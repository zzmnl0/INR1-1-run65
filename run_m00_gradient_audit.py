"""Fixed train-only, no-update gradient audit for the M00/static-ETKF plan."""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from inr_modules.data_managers.FY_dataloader import get_cosmic_dataloader
from inr_modules.data_managers.FY_dataloader import get_dataloaders
from inr_modules.data_managers import SpaceWeatherManager, IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.sliding_dataset import (
    SlidingWindowBatchProcessor, load_representativeness_kernel,
    load_empirical_covariance_targets,
)
from inr_modules.mdia import train_fsia as tf


ROOT = Path(__file__).resolve().parent
RUNS = {
    'explicit_epoch5': ROOT / 'checkpoints_fsia' /
        'run67-m00-seed42-explicit-condition-v1',
    'magnetic_daynight_v2_epoch5': ROOT / 'checkpoints_fsia' /
        'run67-m00-seed42-magnetic-daynight-bias-v2',
}
CHECKPOINTS = {
    label: run / 'epoch_05_background_model.pth'
    for label, run in RUNS.items()
}
OUTPUT = (ROOT / 'checkpoints_fsia' /
          'm00-static-etkf-night-diagnostic-v2' /
          'm00_gradient_audit.json')
MAX_BATCHES = 20
CAP = 0.5


def _sha256_array(value):
    array = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, torch.Tensor)):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return value.detach().cpu().tolist()
            value = value.detach().cpu().item()
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _clone_batch(batch):
    return tuple(item.clone() if isinstance(item, torch.Tensor) else item
                 for item in batch)


def _select_batches(loader, count=MAX_BATCHES):
    total = len(loader)
    if total < 1:
        raise RuntimeError('train loader is empty')
    indices = sorted(set(np.linspace(
        0, total - 1, min(count, total), dtype=np.int64).tolist()))
    wanted = set(indices)
    selected = []
    for index, batch in enumerate(loader):
        if index in wanted:
            selected.append((index, _clone_batch(batch)))
        if index >= indices[-1]:
            break
    if [index for index, _ in selected] != indices:
        raise RuntimeError('fixed batch selection did not cover requested indices')
    return selected


def _batch_identity(batch):
    data = batch[0].detach().cpu().numpy()
    row_ids = (batch[1].detach().cpu().numpy()
               if isinstance(batch[1], torch.Tensor)
               else np.asarray(batch[1]))
    profile_ids = (batch[2].detach().cpu().numpy()
                   if isinstance(batch[2], torch.Tensor)
                   else np.asarray(batch[2]))
    return {
        'point_count': int(len(data)),
        'profile_count': int(np.unique(profile_ids).size),
        'row_ids_sha256': _sha256_array(row_ids),
        'profile_ids_sha256': _sha256_array(profile_ids),
        'row_ids': row_ids.reshape(-1).tolist(),
        'profile_ids': profile_ids.reshape(-1).tolist(),
    }


def _load_config(run):
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    config = copy.deepcopy(manifest['config'])
    resolved = manifest.get('resolved_training', {})
    for key in (
            'resolved_background_stratum_bias_weight',
            'background_stratum_bias_gradient_calibration',
            'resolved_background_magnetic_daynight_weight',
            'background_magnetic_daynight_gradient_calibration'):
        if config.get(key) is None and resolved.get(key) is not None:
            config[key] = resolved[key]
    return config


def _setup(run, checkpoint, scratch):
    config = _load_config(run)
    config['skip_hash_verification'] = True
    device = torch.device(config['device'])
    split = json.loads(Path(config['date_split_manifest']).read_text(
        encoding='utf-8'))['partitions']
    split_days = {'train': list(split['train']),
                  'development': list(split['development'])}
    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'], start_date_str=config['start_date_str'],
        total_hours=config['total_hours'], seq_len=config['seq_len'],
        device=device)
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(
        config['iri_proxy_path'], map_location=device, weights_only=True))
    iri_proxy.eval()
    iri_peak_manager = tf._load_iri_peak_manager(config, device)
    common = dict(
        batch_size=config['batch_size'], bin_size_hours=config['bin_size_hours'],
        num_workers=config.get('num_workers', 0), use_memmap=config.get('use_memmap', True),
        val_ratio=None, split_seed=config['seed'],
        points_per_profile=config.get('profile_points_per_epoch', 8),
        full_validation_profiles=False, split_days=split_days,
        alt_range=tf._observation_alt_range(config),
        sampling_semantics=config.get('background_sampling_semantics',
                                      'legacy_profile_time_bin_v1'),
        max_batches=config.get('background_fixed_updates_per_epoch'))
    fy_loader, _ = get_dataloaders(
        npy_path=config['fy_path'], profile_path=config.get('fy_profile_path'),
        profile_index_path=config.get('fy_profile_index_path'),
        schedule_seed=int(config.get('background_profile_schedule_seed_fy',
                                     config.get('background_profile_schedule_seed', config['seed']))),
        magnetic_sidecar_path=(config.get('m00_observation_qd_sidecar_fy')
                               or config.get('m00_observation_magnetic_sidecar_fy')),
        magnetic_sidecar_source='FY',
        magnetic_sidecar_strict=bool(config.get('m00_observation_strict_magnetic', False)),
        **common)
    split = json.loads(Path(config['date_split_manifest']).read_text(
        encoding='utf-8'))['partitions']
    cosmic_loader, _ = get_cosmic_dataloader(
        cosmic_path=config['cosmic_path'],
        batch_size=config['batch_size'],
        bin_size_hours=config['bin_size_hours'],
        num_workers=config.get('num_workers', 0),
        use_memmap=config.get('use_memmap', True),
        val_ratio=None,
        split_seed=config['seed'],
        points_per_profile=config.get('profile_points_per_epoch', 8),
        profile_index_path=config.get('cosmic_profile_index_path'),
        split_days=split_days,
        alt_range=tf._observation_alt_range(config),
        full_validation_profiles=False,
        sampling_semantics=config.get(
            'background_sampling_semantics', 'legacy_profile_time_bin_v1'),
        schedule_seed=int(config.get(
            'background_profile_schedule_seed_cosmic',
            config.get('background_profile_schedule_seed', config['seed']) + 1)),
        max_batches=config.get('background_fixed_updates_per_epoch'),
        magnetic_sidecar_path=(config.get('m00_observation_qd_sidecar_cosmic')
                               or config.get('m00_observation_magnetic_sidecar_cosmic')),
        magnetic_sidecar_source='COSMIC',
        magnetic_sidecar_strict=bool(config.get(
            'm00_observation_strict_magnetic', False)),
    )
    fy_nb_index = tf.FYNeighborhoodIndex(config['fy_path'], config)
    cosmic_nb_index = tf.COSMICNeighborhoodIndex(config['cosmic_path'], config)
    repr_path = config.get('representativeness_kernel_path')
    repr_kernel = load_representativeness_kernel(repr_path)
    empirical = (load_empirical_covariance_targets(repr_path)
                 if config.get('use_empirical_covariance_loss', False) else None)
    batch_processor = SlidingWindowBatchProcessor(
        sw_manager, device=device, fy_nb_index=fy_nb_index,
        cosmic_nb_index=cosmic_nb_index, representativeness_kernel=repr_kernel,
        representativeness_floor=float(config.get('representativeness_floor', 0.25)),
        empirical_covariance_targets=empirical)
    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    tf._set_stage_mode(model, 'background')
    return config, model, fy_loader, cosmic_loader, sw_manager, batch_processor, iri_peak_manager


def _gradient_vector(loss, parameters):
    if not loss.requires_grad:
        return torch.zeros(sum(parameter.numel() for parameter in parameters),
                           dtype=torch.float32)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True)
    values = [
        (gradient.detach().float().reshape(-1)
         if gradient is not None else
         torch.zeros(parameter.numel(), device=parameter.device))
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(values).cpu()


def _vector_stats(vector, observation):
    vector = vector.float()
    observation = observation.float()
    norm = float(vector.norm())
    obs_norm = float(observation.norm())
    dot = float(torch.dot(vector, observation))
    cosine = dot / (norm * obs_norm) if norm > 0.0 and obs_norm > 0.0 else None
    tanh_vector = torch.tanh(vector)
    tanh_obs = torch.tanh(observation)
    tanh_norm = float(tanh_vector.norm())
    tanh_obs_norm = float(tanh_obs.norm())
    tanh_dot = float(torch.dot(tanh_vector, tanh_obs))
    tanh_cosine = (
        tanh_dot / (tanh_norm * tanh_obs_norm)
        if tanh_norm > 0.0 and tanh_obs_norm > 0.0 else None)
    return {
        'norm': norm,
        'inner_product_with_observation': dot,
        'cosine_with_observation': cosine,
        'tanh_norm': tanh_norm,
        'tanh_cosine_with_observation': tanh_cosine,
    }


def _module_stats(vector, observation, parameter_groups):
    result = {}
    for name, indices in parameter_groups.items():
        selected = vector[indices]
        selected_observation = observation[indices]
        result[name] = _vector_stats(selected, selected_observation)
    return result


def _parameter_groups(model):
    """Return non-overlapping element ranges, not parameter-number ranges."""
    groups = {}
    offset = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split('.', 1)[0]
        indices = torch.arange(offset, offset + parameter.numel(), dtype=torch.long)
        groups.setdefault(group, []).append(indices)
        offset += parameter.numel()
    return {name: torch.cat(chunks) for name, chunks in groups.items()}, offset


def _state_snapshot(model, parameters):
    return {
        'state_dict': {key: value.detach().clone()
                       for key, value in model.state_dict().items()},
        'training': {name: module.training
                     for name, module in model.named_modules()},
        'requires_grad': [parameter.requires_grad for parameter in model.parameters()],
        'grads': [None if parameter.grad is None else parameter.grad.detach().clone()
                  for parameter in model.parameters()],
        'torch_rng': torch.get_rng_state().clone(),
        'cuda_rng': ([state.clone() for state in torch.cuda.get_rng_state_all()]
                     if torch.cuda.is_available() else None),
        'numpy_rng': np.random.get_state(),
        'python_rng': copy.deepcopy(__import__('random').getstate()),
        'parameter_count': int(sum(parameter.numel() for parameter in parameters)),
    }


def _restore_state(model, snapshot):
    model.load_state_dict(snapshot['state_dict'], strict=True)
    for name, training in snapshot['training'].items():
        model.get_submodule(name).train(training) if name else model.train(training)
    for parameter, requires_grad, grad in zip(
            model.parameters(), snapshot['requires_grad'], snapshot['grads']):
        parameter.requires_grad_(requires_grad)
        parameter.grad = None if grad is None else grad.clone()
    torch.set_rng_state(snapshot['torch_rng'])
    if snapshot['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
    np.random.set_state(snapshot['numpy_rng'])
    __import__('random').setstate(snapshot['python_rng'])


def _state_matches(model, parameters, snapshot):
    state_match = all(
        key in model.state_dict()
        and torch.equal(value, model.state_dict()[key])
        for key, value in snapshot['state_dict'].items())
    mode_match = all(
        module.training == training
        for name, training in snapshot['training'].items()
        for module in [model.get_submodule(name) if name else model])
    requires_grad_match = all(
        parameter.requires_grad == expected
        for parameter, expected in zip(
            model.parameters(), snapshot['requires_grad']))
    grads_match = all(
        (parameter.grad is None and expected is None)
        or (parameter.grad is not None and expected is not None
            and torch.equal(parameter.grad, expected))
        for parameter, expected in zip(model.parameters(), snapshot['grads']))
    numpy_state = np.random.get_state()
    numpy_match = (
        numpy_state[0] == snapshot['numpy_rng'][0]
        and np.array_equal(numpy_state[1], snapshot['numpy_rng'][1])
        and numpy_state[2:] == snapshot['numpy_rng'][2:])
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)
    cuda_match = ((cuda_rng is None and snapshot['cuda_rng'] is None)
                  or (cuda_rng is not None and snapshot['cuda_rng'] is not None
                      and len(cuda_rng) == len(snapshot['cuda_rng'])
                      and all(torch.equal(left, right)
                              for left, right in zip(
                                  cuda_rng, snapshot['cuda_rng']))))
    rng_match = (
        torch.equal(torch.get_rng_state(), snapshot['torch_rng'])
        and cuda_match and numpy_match
        and __import__('random').getstate() == snapshot['python_rng'])
    return {
        'state_dict': state_match,
        'training_modes': mode_match,
        'requires_grad': requires_grad_match,
        'parameter_grads': grads_match,
        'model_state_unchanged': all((state_match, mode_match,
                                      requires_grad_match, grads_match)),
        'torch_cuda_numpy_python_rng': rng_match,
        'rng_state_restored': rng_match,
    }


def _supported_stats(vector, observation, point_count, profile_count):
    if vector is None or point_count == 0 or profile_count == 0:
        return {'value': None, 'reason': 'no finite supported points/profiles'}
    stats = _vector_stats(vector, observation)
    if stats['norm'] == 0.0:
        return {'value': None, 'reason': 'zero gradient'}
    return {'value': stats, 'reason': None}


def _group_masks(coords, prediction, target, iri, extras):
    finite = (
        torch.isfinite(coords).all(dim=1)
        & torch.isfinite(prediction.reshape(-1))
        & torch.isfinite(target.reshape(-1))
        & torch.isfinite(iri.reshape(-1)))
    coords = coords.reshape(-1, coords.shape[-1])
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    iri = iri.reshape(-1)
    d = target - iri
    c = prediction - iri
    h = c.square() - 2.0 * c * d
    valid_direction = finite & (d.abs() >= 0.02) & (c.abs() > 1e-6)
    harmful = finite & (h > 1e-8)
    useful = finite & (h < -1e-8)
    local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    low = tf._background_bias_magnetic_masks(coords)[0][1]
    main = (coords[:, 2] >= 200.0) & (coords[:, 2] <= 500.0)
    residual = extras.get('background_residual_raw', c).reshape(-1)
    saturated = torch.isfinite(residual) & (residual.abs() >= 0.95 * CAP)
    masks = {
        'low_magnetic_night_harmful': low & night & main & harmful,
        'low_magnetic_night_useful': low & night & main & valid_direction & useful,
        'low_magnetic_night_saturated': low & night & main & saturated,
        'low_magnetic_night_unsaturated': low & night & main & ~saturated,
        'other_night': (~low) & night & main,
        'day_protection': ~night & main,
    }
    return masks, {
        'direction_valid_count': int((valid_direction & main).sum().detach().cpu()),
        'harmful_count': int((harmful & main).sum().detach().cpu()),
        'useful_count': int((useful & main).sum().detach().cpu()),
        'saturated_count': int((saturated & main).sum().detach().cpu()),
        'h_mean': float(h[finite].mean().detach().cpu()) if finite.any() else None,
    }


def _run_checkpoint(label, run, checkpoint, selected_batches):
    with tempfile.TemporaryDirectory(prefix='m00_grad_') as temp:
        scratch = Path(temp) / label
        scratch.mkdir(parents=True, exist_ok=True)
        config, model, _, _, sw_manager, batch_processor, iri_peak_manager = (
            _setup(run, checkpoint, scratch))
        parameters = [parameter for parameter in model.parameters()
                      if parameter.requires_grad]
        parameter_groups, flat_size = _parameter_groups(model)
        if flat_size != sum(parameter.numel() for parameter in parameters):
            raise AssertionError('parameter ranges do not cover the flat vector')
        snapshot = _state_snapshot(model, parameters)
        current = {}
        records = []
        original_forward = tf._source_forward
        original_record = tf._record_night_v2_execution
        forward_calls = 0

        def forward_wrapper(*args, **kwargs):
            nonlocal forward_calls
            forward_calls += 1
            result = original_forward(*args, **kwargs)
            return result

        tf._source_forward = forward_wrapper
        tf._record_night_v2_execution = lambda *args, **kwargs: None
        try:
            for source, batches in selected_batches.items():
                for batch_index, batch in batches:
                    if len(records) >= MAX_BATCHES * 2 * 2:
                        raise RuntimeError('gradient audit source-batch budget exceeded')
                    current.clear()
                    current['source'] = source
                    before_calls = forward_calls
                    losses = tf._natural_background_source_loss_components(
                        model, batch_processor, batch, source,
                        torch.device(config['device']), config, 4,
                        sw_manager, iri_peak_manager,
                        {'FY': None, 'COSMIC': None}, 1.0, False)
                    coords = losses['coords']
                    target = losses['target']
                    profile_ids = losses['profile_ids']
                    prediction = losses['prediction']
                    extras = losses['extras']
                    observation = _gradient_vector(losses['observation'], parameters)
                    component_vectors = {
                        name: _gradient_vector(value, parameters)
                        for name, value in losses['components'].items()}
                    auxiliary = _gradient_vector(losses['auxiliary'], parameters)
                    masks, counts = _group_masks(
                        coords, prediction, target, extras['ne_iri'], extras)
                    group_vectors, group_counts = {}, {}
                    for name, mask in masks.items():
                        count = int(mask.sum().detach().cpu())
                        ids = profile_ids.reshape(-1)[mask]
                        group_counts[name] = {
                            'point_count': count,
                            'profile_count': int(torch.unique(ids).numel())
                            if count else 0,
                        }
                        if count:
                            loss = tf.profile_huber_loss(
                                prediction, target, profile_ids,
                                delta=float(config.get('huber_delta', 0.2)),
                                valid_mask=mask)
                            group_vectors[name] = _gradient_vector(loss, parameters)
                        else:
                            group_vectors[name] = None
                    d = target.reshape(-1) - extras['ne_iri'].reshape(-1).detach()
                    c = prediction.reshape(-1) - extras['ne_iri'].reshape(-1).detach()
                    h = c.square() - 2.0 * c * d
                    harmful = masks['low_magnetic_night_harmful']
                    h_loss = tf._profile_weighted_mean(
                        torch.relu(h), harmful, profile_ids, torch.ones_like(h))
                    harmful_vector = (_gradient_vector(h_loss, parameters)
                                      if int(harmful.sum()) else None)
                    point_count = int(len(profile_ids))
                    profile_count = int(torch.unique(profile_ids).numel())
                    result = losses
                    row = {
                        'source': source,
                        'batch_index': int(batch_index),
                        'profile_count': profile_count, 'point_count': point_count,
                        'source_batch_loss_calls': 1,
                        'stencil_forward_calls': int(forward_calls - before_calls - 1),
                        'forward_calls': int(forward_calls - before_calls),
                        'losses': {key: float(value.detach().cpu()) for key, value in {
                            'total': losses['total_loss'],
                            'observation': losses['observation'],
                            **losses['components']}.items()},
                        'gradient': {
                            'observation': _vector_stats(observation, observation),
                            'auxiliary': _vector_stats(auxiliary, observation),
                            'components': {
                                name: _vector_stats(vector, observation)
                                for name, vector in component_vectors.items()},
                            'modules': {
                                name: _module_stats(vector, observation, parameter_groups)
                                for name, vector in component_vectors.items()},
                            'harmful_H_diagnostic_only': _supported_stats(
                                harmful_vector, observation,
                                group_counts['low_magnetic_night_harmful']['point_count'],
                                group_counts['low_magnetic_night_harmful']['profile_count']),
                            'groups': {
                                name: (_supported_stats(
                                    vector, observation,
                                    group_counts[name]['point_count'],
                                    group_counts[name]['profile_count'])
                                       if vector is not None else
                                       {'value': None, 'reason': 'no finite supported points/profiles'})
                                for name, vector in group_vectors.items()},
                        },
                        'group_counts': {**group_counts, **counts},
                    }
                    records.append(row)
            if len(records) > MAX_BATCHES * 2 * 2:
                raise RuntimeError('gradient audit source-batch budget exceeded')
        finally:
            tf._source_forward = original_forward
            tf._record_night_v2_execution = original_record
            _restore_state(model, snapshot)
        restore_checks = _state_matches(model, parameters, snapshot)
        return {
            'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'records': records,
            'record_count': len(records),
            'optimizer_step_count': 0,
            'model_state_unchanged': restore_checks['model_state_unchanged'],
            'rng_state_restored': restore_checks['rng_state_restored'],
            'state_restore_checks': restore_checks,
            'source_batch_loss_calls': len(records),
            'stencil_forward_calls': sum(
                row['stencil_forward_calls'] for row in records),
            'forward_calls': forward_calls,
            'selection_identity': [row for source in selected_batches.values()
                                   for _, batch in source for row in [_batch_identity(batch)]],
            'parameter_groups': {
                name: {'start': int(indices[0]), 'stop': int(indices[-1]) + 1,
                       'numel': int(len(indices))}
                for name, indices in parameter_groups.items()},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--selection-output', type=Path)
    args = parser.parse_args()
    output = args.output
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    with tempfile.TemporaryDirectory(prefix='m00_grad_select_') as temp:
        scratch = Path(temp) / 'selection'
        scratch.mkdir(parents=True, exist_ok=True)
        config, _, fy_loader, cosmic_loader, _, _, _ = _setup(
            RUNS['explicit_epoch5'], CHECKPOINTS['explicit_epoch5'], scratch)
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        selected_batches = {
            'FY': _select_batches(fy_loader),
            'COSMIC': _select_batches(cosmic_loader),
        }
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
    selection = {
        source: [
            {'batch_index': int(index), **_batch_identity(batch)}
            for index, batch in batches]
        for source, batches in selected_batches.items()
    }
    selection_manifest = (args.selection_output
                          or output.parent / 'selection_manifest.json')
    selection_manifest.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest.write_text(
        json.dumps(_jsonable({
            'semantics': 'm00_static_etkf_fixed_train_gradient_selection_v2',
            'seed': 42, 'max_batches_per_source': MAX_BATCHES,
            'checkpoints': {key: str(value.resolve())
                            for key, value in CHECKPOINTS.items()},
            'selection': selection,
        }), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    reports = {
        label: _run_checkpoint(label, RUNS[label], CHECKPOINTS[label], selected_batches)
        for label in ('explicit_epoch5', 'magnetic_daynight_v2_epoch5')
    }
    payload = {
        'semantics': 'm00_static_etkf_fixed_train_gradient_audit_v1',
        'seed': 42,
        'split': 'train_only',
        'max_batches_per_source_per_checkpoint': MAX_BATCHES,
        'optimizer_step_count': 0,
        'optimizer_created': False,
        'validation_invocations': 0,
        'calibration_invocations': 0,
        'harmful_H_in_optimizer': False,
        'selection': selection,
        'selection_manifest': str(selection_manifest.resolve()),
        'reports': reports,
        'records_total': sum(report['record_count'] for report in reports.values()),
        'model_state_unchanged': all(
            report['model_state_unchanged'] for report in reports.values()),
        'rng_state_restored': all(
            report['rng_state_restored'] for report in reports.values()),
        'forward_calls': sum(report['forward_calls'] for report in reports.values()),
        'stencil_forward_calls': sum(
            report['stencil_forward_calls'] for report in reports.values()),
        'status': 'completed',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), ensure_ascii=False,
                                 indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'records': {key: value['record_count'] for key, value in reports.items()},
        'optimizer_step_count': 0,
        'model_state_unchanged': all(
            value['model_state_unchanged'] for value in reports.values()),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
