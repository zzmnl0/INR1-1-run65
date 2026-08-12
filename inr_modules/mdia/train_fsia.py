"""Two-stage run66 training for the feature-space local LETKF model."""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from ..config_mdia import get_config_mdia
    from .fsia_model import (
        FSIA_INR_Model,
        solve_density_modes,
    )
    from .physics_losses_mdia import profile_huber_loss, second_difference_loss
    from .sliding_dataset import (
        SlidingWindowBatchProcessor,
        attach_representativeness_weight,
        attach_observation_background,
        empirical_covariance_token_targets,
        load_empirical_covariance_targets,
        load_representativeness_kernel,
        query_observation_payload,
    )
    from .plotting import plot_training_curves
except ImportError:
    from config_mdia import get_config_mdia
    from fsia_model import FSIA_INR_Model, solve_density_modes
    from physics_losses_mdia import profile_huber_loss, second_difference_loss
    from sliding_dataset import (
        SlidingWindowBatchProcessor,
        attach_representativeness_weight,
        attach_observation_background,
        empirical_covariance_token_targets,
        load_empirical_covariance_targets,
        load_representativeness_kernel,
        query_observation_payload,
    )
    from plotting import plot_training_curves

from data_managers import SpaceWeatherManager, IRINeuralProxy
from data_managers.FY_dataloader import (
    FYNeighborhoodIndex,
    COSMICNeighborhoodIndex,
    get_dataloaders,
    get_cosmic_dataloader,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(state, path):
    temp_path = f'{path}.tmp'
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def _validate_date_split(partitions, total_days, development_days,
                         locked_test_days):
    expected = {'train', 'development', 'locked_test'}
    if set(partitions) != expected:
        raise ValueError(f'date split must define {sorted(expected)}')
    normalized = {
        name: np.asarray(days, dtype=np.int64)
        for name, days in partitions.items()}
    for name, days in normalized.items():
        if (days.ndim != 1 or len(np.unique(days)) != len(days)
                or np.any(days < 0) or np.any(days >= total_days)):
            raise ValueError(f'invalid {name} date split')
    if len(normalized['development']) != development_days:
        raise ValueError('development date count mismatch')
    if len(normalized['locked_test']) != locked_test_days:
        raise ValueError('locked-test date count mismatch')
    combined = np.concatenate(list(normalized.values()))
    if len(combined) != total_days or len(np.unique(combined)) != total_days:
        raise ValueError('date split must partition every relative day exactly once')
    return {name: np.sort(days).tolist() for name, days in normalized.items()}


def _load_or_create_date_split(config):
    """Create one deterministic UTC-date split shared by FY and COSMIC."""
    if not config.get('use_date_blocked_split', False):
        return None, None
    path = config.get('date_split_manifest')
    if not path:
        raise ValueError(
            'date_split_manifest is required for date-blocked splitting')
    total_hours = float(config['total_hours'])
    total_days = int(round(total_hours / 24.0))
    if not np.isclose(total_days * 24.0, total_hours):
        raise ValueError('total_hours must contain an integer number of UTC days')
    development_days = int(config.get('development_days', 5))
    locked_test_days = int(config.get('locked_test_days', 5))
    if min(development_days, locked_test_days) < 1:
        raise ValueError('development_days and locked_test_days must be positive')
    if development_days != locked_test_days:
        raise ValueError(
            'M0 requires equal development and locked-test date counts')
    if total_days < 3 * development_days:
        raise ValueError('too few UTC days for blocked train/development/test split')

    identity = {
        'schema_version': 1,
        'seed': int(config['seed']),
        'start_date_str': config['start_date_str'],
        'total_days': total_days,
        'development_days': development_days,
        'locked_test_days': locked_test_days,
    }
    path = os.path.abspath(path)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        for key, value in identity.items():
            if manifest.get(key) != value:
                raise ValueError(f'date split manifest mismatch for {key}')
        partitions = _validate_date_split(
            manifest.get('partitions', {}), total_days,
            development_days, locked_test_days)
    else:
        rng = np.random.default_rng(identity['seed'])
        development = []
        locked_test = []
        for block in np.array_split(
                np.arange(total_days), development_days):
            if len(block) < 2:
                raise ValueError('date block is too small for two held-out dates')
            shuffled = rng.permutation(block)
            development.append(int(shuffled[0]))
            locked_test.append(int(shuffled[1]))
        held_out = set(development + locked_test)
        partitions = _validate_date_split({
            'train': [
                day for day in range(total_days) if day not in held_out],
            'development': development,
            'locked_test': locked_test,
        }, total_days, development_days, locked_test_days)
        start = datetime.strptime(
            identity['start_date_str'], '%Y-%m-%d %H:%M:%S')
        manifest = {
            **identity,
            'partitions': partitions,
            'utc_dates': {
                name: [
                    (start + timedelta(days=day)).strftime('%Y-%m-%d')
                    for day in days]
                for name, days in partitions.items()},
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(
                manifest, stream, ensure_ascii=False, indent=2,
                sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    return partitions, {
        'path': path,
        'sha256': _sha256_file(path),
        'partitions': partitions,
    }


def _loader_profile_ids(loader):
    """Return active profile IDs, respecting an optional screen subset."""
    sampler = loader.batch_sampler
    groups = [
        indices for bin_groups in sampler.profiles_by_bin.values()
        for indices in bin_groups]
    profile_ids = np.array(sorted({
        int(loader.dataset.profile_ids[indices[0]]) for indices in groups
    }), dtype=np.int64)
    if profile_ids.size == 0:
        raise ValueError('loader contains no active profiles')
    return profile_ids


def _record_resolved_training_config(config, covariance_strata):
    path = os.path.join(config['save_dir'], 'run_manifest.json')
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as stream:
        manifest = json.load(stream)
    manifest['resolved_training'] = {
        'analysis_exact_mode_loss': bool(
            config.get('analysis_exact_mode_loss', False)),
        'use_covariance_moment_loss': bool(
            config.get('use_covariance_moment_loss', False)),
        'use_empirical_covariance_loss': bool(
            config.get('use_empirical_covariance_loss', False)),
        'covariance_gradient_target': float(
            config.get('covariance_gradient_target', 0.20)),
        'resolved_covariance_weight': config.get(
            'resolved_covariance_weight'),
        'use_direction_loss': bool(
            config.get('use_direction_loss', False)),
        'direction_gradient_target': float(
            config.get('direction_gradient_target', 0.20)),
        'resolved_direction_weight': config.get(
            'resolved_direction_weight'),
        'use_observation_gram_loss': bool(
            config.get('use_observation_gram_loss', False)),
        'gram_gradient_target': float(
            config.get('gram_gradient_target', 0.02)),
        'gram_calibration_batches': int(
            config.get('gram_calibration_batches', 20)),
        'resolved_gram_weight': config.get('resolved_gram_weight'),
        'gram_gradient_calibration': config.get('gram_gradient_calibration'),
        'representativeness_kernel': config.get(
            'resolved_representativeness_kernel'),
        'covariance_strata': covariance_strata,
        'date_split': config.get('resolved_date_split'),
        'background_training_semantics': config.get(
            'background_training_semantics'),
        'background_loader_schedule': config.get('background_loader_schedule'),
        'background_update_steps': config.get('background_update_steps', []),
    }
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _query_observations(index, coords, profile_ids=None,
                        allowed_profile_ids=None):
    if index is None:
        return None
    exclude = None if profile_ids is None else profile_ids.detach().cpu().numpy()
    return query_observation_payload(
        index, coords, coords.device, exclude_profile_ids=exclude,
        allowed_profile_ids=allowed_profile_ids)


def _unpack_source_batch(batch, device):
    data, _, profile_ids = batch
    data = data.to(device, non_blocking=True)
    return data[:, :4], data[:, 4:5], profile_ids.to(device, non_blocking=True)


def _apply_source_dropout(fy_obs, cosmic_obs, profile_ids, probabilities):
    p_m10, p_m01, p_m11 = probabilities
    if not np.isclose(p_m10 + p_m01 + p_m11, 1.0):
        raise ValueError('source_dropout probabilities must sum to one')
    _, inverse = torch.unique(
        profile_ids.flatten(), sorted=False, return_inverse=True)
    reference = fy_obs if fy_obs is not None else cosmic_obs
    draw = torch.rand(
        inverse.max().item() + 1, device=profile_ids.device,
        dtype=reference['value'].dtype)
    draw = draw[inverse]
    keep_fy = (draw < p_m10) | (draw >= p_m10 + p_m01)
    keep_cosmic = draw >= p_m10
    def apply(payload, keep):
        if payload is None:
            return None
        result = dict(payload)
        if payload['valid_mask'].ndim == 1 and 'query_index' in payload:
            result['valid_mask'] = (
                payload['valid_mask'] & keep[payload['query_index']])
        else:
            result['valid_mask'] = payload['valid_mask'] & keep.unsqueeze(-1)
        return result
    return apply(fy_obs, keep_fy), apply(cosmic_obs, keep_cosmic)


def _apply_balanced_profile_modes(
        fy_obs, cosmic_obs, profile_ids, analysis_epoch):
    unique_ids, inverse = torch.unique(
        profile_ids.flatten(), sorted=False, return_inverse=True)
    cycle = torch.remainder(
        unique_ids.to(torch.int64) + int(analysis_epoch), 4)
    point_cycle = cycle[inverse]

    def apply(payload, keep):
        if payload is None:
            return None
        result = dict(payload)
        if payload['valid_mask'].ndim == 1 and 'query_index' in payload:
            result['valid_mask'] = (
                payload['valid_mask'] & keep[payload['query_index']])
        else:
            result['valid_mask'] = payload['valid_mask'] & keep.unsqueeze(-1)
        return result

    return (
        apply(fy_obs, point_cycle != 1),
        apply(cosmic_obs, point_cycle != 0),
    )


def _balanced_profile_mode_counts(profile_ids, analysis_epoch):
    unique_ids = torch.unique(profile_ids.flatten(), sorted=False)
    cycle = torch.remainder(
        unique_ids.to(torch.int64) + int(analysis_epoch), 4)
    return {
        'M10': int((cycle == 0).sum().item()),
        'M01': int((cycle == 1).sum().item()),
        'M11': int((cycle >= 2).sum().item()),
    }


def _source_mode_for_batch(schedule, batch_index):
    if schedule == 'random_profile':
        return None
    if schedule == 'deterministic_112':
        return ('M10', 'M01', 'M11', 'M11')[batch_index % 4]
    if schedule == 'balanced_profile_112':
        return schedule
    raise ValueError(f'unknown source_mode_schedule: {schedule}')


def _apply_source_mode(fy_obs, cosmic_obs, mode):
    if mode not in ('M10', 'M01', 'M11'):
        raise ValueError(f'unknown source mode: {mode}')

    def apply(payload, keep):
        if payload is None or keep:
            return payload
        result = dict(payload)
        result['valid_mask'] = torch.zeros_like(
            payload['valid_mask'], dtype=torch.bool)
        return result

    return apply(fy_obs, mode != 'M01'), apply(cosmic_obs, mode != 'M10')


def _set_training_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == 'background':
        modules = [
            model.iri_align_net,
            model.background_decoder,
            model.sw_encoder,
        ]
        if model.use_sw_freq:
            modules.extend([model.sw_freq_branch, model.sw_gate])
    elif stage == 'analysis':
        modules = [model.kalman_layer]
        if hasattr(model, 'density_basis_decoder'):
            modules.append(model.density_basis_decoder)
    else:
        raise ValueError(f'unknown training stage: {stage}')
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.iri_proxy.freeze()


def _set_stage_mode(model, stage):
    model.train()
    model.iri_proxy.eval()
    if stage == 'analysis':
        model.iri_align_net.eval()
        model.background_decoder.eval()
        model.sw_encoder.eval()
        if model.use_sw_freq:
            model.sw_freq_branch.eval()
            model.sw_gate.eval()


def _make_optimizer(model, config, stage_epochs):
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = None
    if config.get('scheduler_type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, stage_epochs),
            eta_min=config.get('min_lr', 1e-6),
        )
    return optimizer, scheduler


def _source_forward(model, batch_processor, coords, sw_seq, iri_peak,
                    stage, target_source, profile_ids, apply_dropout, config,
                    iri_peak_manager=None, allowed_profile_ids=None,
                    source_mode=None, analysis_epoch=0):
    allowed_profile_ids = allowed_profile_ids or {}
    if stage == 'background':
        fy_obs = cosmic_obs = None
    elif target_source == 'FY':
        fy_obs = _query_observations(
            batch_processor.fy_nb_index, coords, profile_ids,
            allowed_profile_ids.get('FY'))
        cosmic_obs = _query_observations(
            batch_processor.cosmic_nb_index, coords,
            allowed_profile_ids=allowed_profile_ids.get('COSMIC'))
    else:
        fy_obs = _query_observations(
            batch_processor.fy_nb_index, coords,
            allowed_profile_ids=allowed_profile_ids.get('FY'))
        cosmic_obs = _query_observations(
            batch_processor.cosmic_nb_index, coords, profile_ids,
            allowed_profile_ids.get('COSMIC'))

    if stage == 'analysis' and batch_processor.representativeness_kernel is not None:
        fy_obs = attach_representativeness_weight(
            fy_obs, coords, target_source, 'FY',
            batch_processor.representativeness_kernel,
            batch_processor.representativeness_floor)
        cosmic_obs = attach_representativeness_weight(
            cosmic_obs, coords, target_source, 'COSMIC',
            batch_processor.representativeness_kernel,
            batch_processor.representativeness_floor)

    if stage == 'analysis' and apply_dropout:
        if source_mode is None:
            fy_obs, cosmic_obs = _apply_source_dropout(
                fy_obs, cosmic_obs, profile_ids,
                config['source_dropout'])
        elif source_mode == 'balanced_profile_112':
            fy_obs, cosmic_obs = _apply_balanced_profile_modes(
                fy_obs, cosmic_obs, profile_ids, analysis_epoch)
        else:
            fy_obs, cosmic_obs = _apply_source_mode(
                fy_obs, cosmic_obs, source_mode)
    if stage == 'analysis':
        fy_obs = attach_observation_background(
            fy_obs, model, batch_processor.sw_manager, iri_peak_manager)
        cosmic_obs = attach_observation_background(
            cosmic_obs, model, batch_processor.sw_manager, iri_peak_manager)
    return model(
        coords,
        sw_seq,
        iri_peak=iri_peak,
        observations_fy=fy_obs,
        observations_cosmic=cosmic_obs,
    )


def _structure_losses(model, batch_processor, sw_manager, iri_peak_manager,
                      coords, profile_ids, stage, config,
                      allowed_profile_ids=None):
    n = min(config.get('structure_batch_size', 32), len(coords))
    if n == 0:
        zero = coords.new_zeros(())
        return zero, zero
    chosen = torch.randperm(len(coords), device=coords.device)[:n]
    centers = coords[chosen, :4].clone()
    selected_ids = profile_ids[chosen]
    residual_key = 'background_residual' if stage == 'background' else 'ne_residual'

    def evaluate(triplet):
        flat = triplet.reshape(-1, 4)
        repeated_ids = selected_ids.repeat_interleave(3)
        sw_seq = sw_manager.get_drivers_sequence(flat[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(flat)
                    if iri_peak_manager is not None else None)
        _, _, _, _, extras = _source_forward(
            model, batch_processor, flat, sw_seq, iri_peak,
            stage, 'FY', repeated_ids, False, config, iri_peak_manager,
            allowed_profile_ids)
        return extras[residual_key].reshape(n, 3)

    alt_step = config.get('structure_alt_step_km', 20.0)
    centers[:, 2].clamp_(
        model.alt_min + alt_step, model.alt_max - alt_step)
    vertical = centers[:, None, :].repeat(1, 3, 1)
    vertical[:, 0, 2] = centers[:, 2] - alt_step
    vertical[:, 2, 2] = centers[:, 2] + alt_step

    time_step = config.get('structure_time_step_hours', 1.0)
    centers[:, 3].clamp_(time_step, 720.0 - time_step)
    temporal = centers[:, None, :].repeat(1, 3, 1)
    temporal[:, 0, 3] = centers[:, 3] - time_step
    temporal[:, 2, 3] = centers[:, 3] + time_step

    beta = config.get('structure_huber_beta', 0.05)
    return (
        second_difference_loss(evaluate(vertical), beta),
        second_difference_loss(evaluate(temporal), beta),
    )


def _gradient_norms(data_loss, auxiliary_loss, parameters, components=None):
    def loss_norm(loss):
        if not loss.requires_grad:
            return data_loss.new_zeros(())
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True)
        finite = [grad.square().sum() for grad in grads if grad is not None]
        return torch.sqrt(torch.stack(finite).sum()) if finite else data_loss.new_zeros(())

    data_norm = loss_norm(data_loss)
    auxiliary_norm = loss_norm(auxiliary_loss)
    ratio = auxiliary_norm / data_norm.clamp_min(1e-12)
    component_norms = {
        name: loss_norm(loss).detach()
        for name, loss in (components or {}).items()
    }
    return (
        data_norm.detach(), auxiliary_norm.detach(), ratio.detach(),
        component_norms,
    )


def _query_precision_sum(extras, source):
    precision = extras[f'precision_{source}']
    if precision.ndim == 1:
        query_index = extras[f'observation_query_index_{source}'].long()
        result = extras['ne_bkg'].new_zeros(extras['ne_bkg'].shape[0])
        if query_index.numel():
            result.index_add_(0, query_index, precision)
        return result
    return precision.sum(dim=-1)


def _analysis_active_mask(extras):
    """Queries with at least one physical observation contributing precision."""
    return (_query_precision_sum(extras, 'FY')
            + _query_precision_sum(extras, 'COSMIC')) > 0


_EXACT_MODE_SOURCES = {
    'M10': ('FY',),
    'M01': ('COSMIC',),
    'M11': ('FY', 'COSMIC'),
}
_EXACT_MODE_WEIGHTS = {'M10': 0.25, 'M01': 0.25, 'M11': 0.50}


def observation_gram_whitening_loss(extras, profile_ids, kalman_layer,
                                    return_details=False):
    """Whiten the production-precision weighted observation factor Gram.

    The loss is computed in the seven independent factor coordinates and is
    therefore exactly tied to the ``HX = F @ C`` geometry used by the ETKF.
    Queries with fewer than seven positive-precision tokens are unsupported;
    rank-deficient Gram matrices with sufficient support remain eligible.
    """
    if kalman_layer.anomaly_parameterization != 'orthogonal_factor':
        raise ValueError('observation Gram loss requires orthogonal_factor')
    rank = kalman_layer.n_members - 1
    profile_ids = profile_ids.reshape(-1)
    expected = extras['ne_bkg'].shape[0]
    if profile_ids.numel() != expected:
        raise ValueError('profile_ids must contain one ID per query')
    identity = torch.eye(
        rank, device=extras['ne_bkg'].device, dtype=torch.float32)
    losses, coverage = {}, {}
    total = extras['ne_bkg'].sum() * 0.0

    for mode, sources in _EXACT_MODE_SOURCES.items():
        grams = []
        token_counts = torch.zeros(
            expected, device=extras['ne_bkg'].device, dtype=torch.long)
        for source in sources:
            precomputed = extras.get(f'factor_gram_{source}')
            if precomputed is not None:
                grams.append(precomputed.float())
                precision = extras[f'precision_{source}']
                if precision.ndim == 1:
                    query_index = extras[f'observation_query_index_{source}']
                    token_counts.index_add_(
                        0, query_index.long(), (precision > 0.0).long())
                else:
                    token_counts = token_counts + (precision > 0.0).sum(dim=-1)
                continue
            factors = extras.get(f'observation_factors_{source}')
            if factors is None:
                basis = extras.get(f'basis_{source}')
                if basis is None:
                    raise ValueError(
                        f'observation basis is missing for source {source}')
                factors = kalman_layer.observation_factor_coordinates(
                    basis, extras.get('factor_scales'))
            precision = extras[f'precision_{source}']
            factors = factors.float()
            precision = precision.float()
            grams.append(torch.einsum(
                'bmr,bm,bmk->brk', factors, precision, factors))
            token_counts = token_counts + (precision > 0.0).sum(dim=-1)
        gram = sum(grams)
        trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1)
        eligible = (
            (token_counts >= rank)
            & torch.isfinite(trace)
            & (trace > 1e-12)
            & torch.isfinite(gram).all(dim=-1).all(dim=-1)
        )
        normalized = gram / trace.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1)
        query_loss = (normalized - identity / rank).square().sum(dim=(-2, -1))
        query_loss = torch.where(
            eligible, query_loss, torch.zeros_like(query_loss))
        losses[mode] = _profile_weighted_mean(
            query_loss, eligible, profile_ids,
            torch.ones_like(query_loss))
        coverage[mode] = eligible.float().mean()
        total = total + _EXACT_MODE_WEIGHTS[mode] * losses[mode]

    if return_details:
        return total, losses, coverage
    return total


def exact_mode_profile_losses(extras, target, profile_ids, delta):
    """Profile-balanced M10/M01/M11 losses from one joint forward."""
    losses = {}
    active_masks = {}
    increments = solve_density_modes(extras)
    for mode, sources in _EXACT_MODE_SOURCES.items():
        prediction = extras['ne_bkg'] + increments[mode].unsqueeze(-1)
        active = sum(
            _query_precision_sum(extras, source)
            for source in sources) > 0
        losses[mode] = profile_huber_loss(
            prediction, target, profile_ids, delta=delta, valid_mask=active)
        active_masks[mode] = active
    total = sum(
        _EXACT_MODE_WEIGHTS[mode] * loss
        for mode, loss in losses.items())
    return total, losses, active_masks


def _covariance_stratum_ids(coords):
    local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    high = coords[:, 2] >= 300.0
    return high.to(torch.long) * 2 + night.to(torch.long)


def _profile_weighted_mean(values, valid, profile_ids, weights):
    results = []
    for profile_id in torch.unique(profile_ids):
        mask = valid & (profile_ids == profile_id)
        if mask.any():
            selected_weights = weights[mask]
            results.append(
                (values[mask] * selected_weights).sum()
                / selected_weights.sum().clamp_min(1e-12))
    if not results:
        return values.sum() * 0.0
    return torch.stack(results).mean()


def exact_mode_direction_losses(extras, target, profile_ids, target_source):
    """Penalize only M10/M01/M11 increments opposite to target residuals."""
    if target_source not in ('FY', 'COSMIC'):
        raise ValueError(f'unknown target source: {target_source}')
    desired = target.squeeze(-1) - extras['ne_bkg'].squeeze(-1)
    desired_sign = torch.sign(desired).detach()
    target_r = (
        extras['r_fy'] if target_source == 'FY' else extras['r_cosmic'])
    increments = solve_density_modes(extras)
    losses = {}
    active_masks = {}
    for mode, sources in _EXACT_MODE_SOURCES.items():
        active = (
            sum(_query_precision_sum(extras, source)
                for source in sources) > 0
        ) & (desired.abs() >= 0.05)
        token_loss = F.relu(
            -desired_sign * increments[mode] / torch.sqrt(target_r))
        losses[mode] = _profile_weighted_mean(
            token_loss,
            active,
            profile_ids,
            torch.ones_like(token_loss),
        )
        active_masks[mode] = active
    total = sum(
        _EXACT_MODE_WEIGHTS[mode] * loss
        for mode, loss in losses.items())
    return total, losses, active_masks


def covariance_moment_loss(extras, target, profile_ids, target_source,
                           stratum_weights):
    """Profile-balanced physical residual covariance moment matching."""
    target_r = (
        extras['r_fy'] if target_source == 'FY' else extras['r_cosmic'])
    target_residual = target.squeeze(-1) - extras['ne_bkg'].squeeze(-1)
    target_standardized = torch.clamp(
        target_residual / torch.sqrt(target_r), -3.0, 3.0)
    query_sum = target_residual.new_zeros(target_residual.shape)
    query_sources = target_residual.new_zeros(target_residual.shape)

    for suffix, source_r in (
            ('FY', extras['r_fy']), ('COSMIC', extras['r_cosmic'])):
        innovation = extras[f'innov_{suffix}']
        cross_covariance = extras[f'cross_covariance_{suffix}']
        precision = extras[f'precision_{suffix}']
        # precision = valid * localization / R; recover the localization
        # weight so R standardizes the moment without weighting it twice.
        localization = precision * source_r
        target_moment = (
            target_standardized.unsqueeze(-1)
            * torch.clamp(
                innovation / torch.sqrt(source_r), -3.0, 3.0)
        ).detach()
        predicted_moment = (
            cross_covariance / torch.sqrt(target_r * source_r))
        token_loss = F.smooth_l1_loss(
            predicted_moment, target_moment, beta=1.0, reduction='none')
        weight_sum = localization.sum(dim=-1)
        active = weight_sum > 0
        query_loss = (
            (token_loss * localization).sum(dim=-1)
            / weight_sum.clamp_min(1e-12))
        query_sum = query_sum + torch.where(
            active, query_loss, torch.zeros_like(query_loss))
        query_sources = query_sources + active.to(query_sources.dtype)

    valid_query = query_sources > 0
    query_loss = query_sum / query_sources.clamp_min(1.0)
    cells = _covariance_stratum_ids(extras['query_coords'])
    cell_weights = torch.as_tensor(
        stratum_weights[target_source],
        device=query_loss.device, dtype=query_loss.dtype)
    return _profile_weighted_mean(
        query_loss, valid_query, profile_ids, cell_weights[cells])


def empirical_covariance_loss(
        extras, profile_ids, target_source, empirical_targets):
    """Match model covariance to frozen train-only profile-blocked cells."""
    if empirical_targets is None:
        raise ValueError('empirical covariance targets are required')
    target_r = (
        extras['r_fy'] if target_source == 'FY' else extras['r_cosmic'])
    query_sum = extras['ne_bkg'].new_zeros(extras['ne_bkg'].shape[0])
    query_sources = torch.zeros_like(query_sum)
    for suffix, source_r in (
            ('FY', extras['r_fy']), ('COSMIC', extras['r_cosmic'])):
        precision = extras[f'precision_{suffix}']
        covariance_target, stable = empirical_covariance_token_targets(
            extras['query_coords'], extras[f'observation_coords_{suffix}'],
            extras[f'observation_rho_squared_{suffix}'], precision > 0.0,
            target_source, suffix, empirical_targets)
        scale = torch.sqrt(target_r * source_r)
        token_loss = F.smooth_l1_loss(
            extras[f'cross_covariance_{suffix}'] / scale,
            (covariance_target / scale).detach(),
            beta=1.0, reduction='none')
        weights = precision * source_r * stable.to(precision.dtype)
        weight_sum = weights.sum(dim=-1)
        active = weight_sum > 0.0
        query_loss = (
            (token_loss * weights).sum(dim=-1)
            / weight_sum.clamp_min(1e-12))
        query_sum = query_sum + torch.where(
            active, query_loss, torch.zeros_like(query_loss))
        query_sources = query_sources + active.to(query_sources.dtype)
    valid_query = query_sources > 0.0
    return _profile_weighted_mean(
        query_sum / query_sources.clamp_min(1.0), valid_query, profile_ids,
        torch.ones_like(query_sum))


def _covariance_training_loss(
        extras, target, profile_ids, target_source, batch_processor, config):
    if not (config.get('use_covariance_moment_loss', False)
            or config.get('use_empirical_covariance_loss', False)):
        return extras['ne_bkg'].sum() * 0.0
    if config.get('use_empirical_covariance_loss', False):
        return empirical_covariance_loss(
            extras, profile_ids, target_source,
            batch_processor.empirical_covariance_targets)
    return covariance_moment_loss(
        extras, target, profile_ids, target_source,
        config['covariance_stratum_weights'])


def _paired_analysis_losses(
        model, batch_processor, fy_batch, cosmic_batch, device, config,
        sw_manager, iri_peak_manager, allowed_profile_ids=None,
        source_mode=None, analysis_epoch=0):
    coords, target, profile_ids = _unpack_source_batch(fy_batch, device)
    sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
    iri_peak = (iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
    cosmic_coords, cosmic_target, cosmic_ids = _unpack_source_batch(
        cosmic_batch, device)
    cosmic_sw = sw_manager.get_drivers_sequence(cosmic_coords[:, 3])
    cosmic_peak = (iri_peak_manager.get_iri_peak(cosmic_coords)
                   if iri_peak_manager is not None else None)
    exact_modes = config.get('analysis_exact_mode_loss', False)
    fy_pred, _, _, _, fy_extras = _source_forward(
        model, batch_processor, coords, sw_seq, iri_peak,
        'analysis', 'FY', profile_ids, not exact_modes, config, iri_peak_manager,
        allowed_profile_ids, source_mode, analysis_epoch)
    cosmic_pred, _, _, _, cosmic_extras = _source_forward(
        model, batch_processor, cosmic_coords, cosmic_sw, cosmic_peak,
        'analysis', 'COSMIC', cosmic_ids, not exact_modes, config, iri_peak_manager,
        allowed_profile_ids, source_mode, analysis_epoch)
    delta = config.get('huber_delta', 0.2)
    if exact_modes:
        fy_loss = exact_mode_profile_losses(
            fy_extras, target, profile_ids, delta)[0]
        cosmic_loss = exact_mode_profile_losses(
            cosmic_extras, cosmic_target, cosmic_ids, delta)[0]
    else:
        active_only = config.get('analysis_loss_active_only', False)
        fy_loss = profile_huber_loss(
            fy_pred, target, profile_ids, delta=delta,
            valid_mask=_analysis_active_mask(fy_extras) if active_only else None)
        cosmic_loss = profile_huber_loss(
            cosmic_pred, cosmic_target, cosmic_ids, delta=delta,
            valid_mask=_analysis_active_mask(cosmic_extras) if active_only else None)
    observation_loss = 0.5 * (fy_loss + cosmic_loss)
    covariance_loss = 0.5 * (
        _covariance_training_loss(
            fy_extras, target, profile_ids, 'FY', batch_processor, config)
        + _covariance_training_loss(
            cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC',
            batch_processor, config))
    direction_loss = 0.5 * (
        exact_mode_direction_losses(
            fy_extras, target, profile_ids, 'FY')[0]
        + exact_mode_direction_losses(
            cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC')[0])
    gram_loss = observation_loss.new_zeros(())
    if config.get('use_observation_gram_loss', False):
        gram_loss = 0.5 * (
            observation_gram_whitening_loss(
                fy_extras, profile_ids, model.kalman_layer)
            + observation_gram_whitening_loss(
                cosmic_extras, cosmic_ids, model.kalman_layer))
    return observation_loss, covariance_loss, direction_loss, gram_loss


def _resolve_covariance_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    batches = int(config.get('covariance_calibration_batches', 20))
    target_ratio = float(config.get('covariance_gradient_target', 0.20))
    if batches < 1 or not 0.0 < target_ratio <= 1.0:
        raise ValueError(
            'covariance calibration batches must be positive and target in (0, 1]')
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    ratios = []
    try:
        _set_stage_mode(model, 'analysis')
        cosmic_iter = iter(cosmic_train_loader)
        for batch_index, fy_batch in enumerate(train_loader):
            if batch_index >= batches:
                break
            try:
                cosmic_batch = next(cosmic_iter)
            except StopIteration:
                cosmic_iter = iter(cosmic_train_loader)
                cosmic_batch = next(cosmic_iter)
            observation, covariance, _, _ = _paired_analysis_losses(
                model, batch_processor, fy_batch, cosmic_batch, device,
                config, sw_manager, iri_peak_manager, allowed_profile_ids,
                _source_mode_for_batch(
                    config.get('source_mode_schedule', 'random_profile'),
                    batch_index))
            parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad]
            _, _, covariance_to_observation, _ = _gradient_norms(
                observation, covariance, parameters)
            value = float(covariance_to_observation)
            if np.isfinite(value) and value > 0:
                ratios.append(value)
            model.zero_grad(set_to_none=True)
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if not ratios:
        raise RuntimeError('covariance moment calibration produced no finite gradients')
    minimum_weight = (
        1e-4 if config.get('use_empirical_covariance_loss', False)
        else 1e-3)
    resolved = float(np.clip(
        target_ratio / np.median(ratios), minimum_weight, 1.0))
    print(
        f'[协方差矩] 梯度比中位数={np.median(ratios):.6f}, '
        f'resolved lambda={resolved:.6f}')
    return resolved


def _resolve_direction_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    batches = int(config.get('direction_calibration_batches', 20))
    target_ratio = float(config.get('direction_gradient_target', 0.20))
    if batches < 1 or not 0.0 < target_ratio <= 1.0:
        raise ValueError(
            'direction calibration batches must be positive and target in (0, 1]')
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    ratios = []
    try:
        _set_stage_mode(model, 'analysis')
        cosmic_iter = iter(cosmic_train_loader)
        for batch_index, fy_batch in enumerate(train_loader):
            if batch_index >= batches:
                break
            try:
                cosmic_batch = next(cosmic_iter)
            except StopIteration:
                cosmic_iter = iter(cosmic_train_loader)
                cosmic_batch = next(cosmic_iter)
            observation, _, direction, _ = _paired_analysis_losses(
                model, batch_processor, fy_batch, cosmic_batch, device,
                config, sw_manager, iri_peak_manager, allowed_profile_ids,
                'exact_M10_M01_M11')
            decoder = (model.density_basis_decoder
                       if hasattr(model, 'density_basis_decoder')
                       else model.kalman_layer)
            parameters = [
                parameter for parameter in decoder.parameters()
                if parameter.requires_grad]
            _, _, direction_to_observation, _ = _gradient_norms(
                observation, direction, parameters)
            value = float(direction_to_observation)
            if np.isfinite(value) and value > 0:
                ratios.append(value)
            model.zero_grad(set_to_none=True)
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if not ratios:
        raise RuntimeError('direction calibration produced no finite gradients')
    resolved = float(np.clip(
        target_ratio / np.median(ratios), 1e-4, 1.0))
    print(
        f'[方向损失] 梯度比中位数={np.median(ratios):.6f}, '
        f'resolved lambda={resolved:.6f}')
    return resolved


def _resolve_gram_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    batches = int(config.get('gram_calibration_batches', 20))
    target_ratio = float(config.get('gram_gradient_target', 0.02))
    if batches < 1 or not 0.0 < target_ratio <= 1.0:
        raise ValueError(
            'Gram calibration batches must be positive and target in (0, 1]')
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    ratios = []
    try:
        _set_stage_mode(model, 'analysis')
        cosmic_iter = iter(cosmic_train_loader)
        for batch_index, fy_batch in enumerate(train_loader):
            if batch_index >= batches:
                break
            try:
                cosmic_batch = next(cosmic_iter)
            except StopIteration:
                cosmic_iter = iter(cosmic_train_loader)
                cosmic_batch = next(cosmic_iter)
            observation, _, _, gram = _paired_analysis_losses(
                model, batch_processor, fy_batch, cosmic_batch, device,
                config, sw_manager, iri_peak_manager, allowed_profile_ids,
                'exact_M10_M01_M11')
            parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad]
            _, _, gram_to_observation, _ = _gradient_norms(
                observation, gram, parameters)
            value = float(gram_to_observation)
            if np.isfinite(value) and value > 0:
                ratios.append(value)
            model.zero_grad(set_to_none=True)
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if not ratios:
        raise RuntimeError('observation Gram calibration produced no finite gradients')
    median_ratio = float(np.median(ratios))
    resolved = float(np.clip(target_ratio / median_ratio, 1e-4, 1.0))
    config['gram_gradient_calibration'] = {
        'batches': len(ratios),
        'median_ratio': median_ratio,
        'target_ratio': target_ratio,
    }
    print(
        f'[HX Gram] 梯度比中位数={median_ratio:.6f}, '
        f'resolved lambda={resolved:.6f}')
    return resolved


def train_one_epoch(model, train_loader, batch_processor, optimizer, device,
                    config, epoch, stage, scaler, sw_manager, iri_peak_manager,
                    cosmic_train_loader, allowed_profile_ids=None):
    _set_stage_mode(model, stage)
    use_amp = config.get('use_amp', False) and scaler is not None
    delta = config.get('huber_delta', 0.2)
    # Background must cover both QC-v2 sources completely.  Analysis retains
    # the historical FY-anchored schedule and only cycles COSMIC.
    epoch_length = _background_epoch_length(
        train_loader, cosmic_train_loader, stage)
    fy_iter = iter(train_loader)
    cosmic_iter = iter(cosmic_train_loader)
    stats = {key: 0.0 for key in (
        'total', 'observation', 'covariance', 'direction', 'gram',
        'fy_obs', 'cosmic_obs',
        'fy_m10', 'fy_m01', 'fy_m11',
        'cosmic_m10', 'cosmic_m01', 'cosmic_m11',
        'fy_direction_m10', 'fy_direction_m01', 'fy_direction_m11',
        'cosmic_direction_m10', 'cosmic_direction_m01',
        'cosmic_direction_m11',
        'iri', 'increment',
        'vertical', 'time', 'weighted_iri', 'weighted_increment',
        'weighted_covariance', 'weighted_direction', 'weighted_gram',
        'weighted_vertical', 'weighted_time',
        'gradient_ratio', 'gradient_ratio_covariance',
        'gradient_ratio_direction', 'gradient_ratio_iri',
        'gradient_ratio_increment', 'gradient_ratio_vertical',
        'gradient_ratio_time', 'gradient_ratio_gram',
        'gradient_ratio_fy_m10', 'gradient_ratio_fy_m01',
        'gradient_ratio_fy_m11', 'gradient_ratio_cosmic_m10',
        'gradient_ratio_cosmic_m01', 'gradient_ratio_cosmic_m11',
        'fy_active_fraction', 'cosmic_active_fraction',
        'fy_gram_m10_coverage', 'fy_gram_m01_coverage',
        'fy_gram_m11_coverage', 'cosmic_gram_m10_coverage',
        'cosmic_gram_m01_coverage', 'cosmic_gram_m11_coverage',
        'K_FY_mean', 'K_COSMIC_mean', 'innov_FY_norm',
        'innov_COSMIC_norm', 'inflation', 'ne_delta_abs',
    )}
    diagnostics = []
    audited_batches = 0
    processed_batches = 0
    max_train_batches = config.get('max_train_batches')
    if max_train_batches is not None and int(max_train_batches) < 1:
        raise ValueError('max_train_batches must be positive')
    source_mode_counts = {mode: 0 for mode in ('M10', 'M01', 'M11')}
    exact_mode_loss = (
        stage == 'analysis'
        and config.get('analysis_exact_mode_loss', False))
    started = time.time()

    for batch_idx in range(epoch_length):
        try:
            fy_batch = next(fy_iter)
        except StopIteration:
            fy_iter = iter(train_loader)
            fy_batch = next(fy_iter)
        coords, target, profile_ids = _unpack_source_batch(fy_batch, device)
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)

        try:
            cosmic_batch = next(cosmic_iter)
        except StopIteration:
            cosmic_iter = iter(cosmic_train_loader)
            cosmic_batch = next(cosmic_iter)
        cosmic_coords, cosmic_target, cosmic_ids = _unpack_source_batch(
            cosmic_batch, device)
        cosmic_sw = sw_manager.get_drivers_sequence(cosmic_coords[:, 3])
        cosmic_peak = (iri_peak_manager.get_iri_peak(cosmic_coords)
                       if iri_peak_manager is not None else None)
        source_mode = None
        if stage == 'analysis':
            analysis_epoch = epoch - int(config['background_epochs'])
            analysis_batch = (
                analysis_epoch * len(train_loader)
                + batch_idx)
            if exact_mode_loss:
                source_mode = 'exact_M10_M01_M11'
                for mode in source_mode_counts:
                    source_mode_counts[mode] += 1
            else:
                source_mode = _source_mode_for_batch(
                    config.get('source_mode_schedule', 'random_profile'),
                    analysis_batch)
                if source_mode in source_mode_counts:
                    source_mode_counts[source_mode] += 1
                elif source_mode == 'balanced_profile_112':
                    for ids in (profile_ids, cosmic_ids):
                        for mode, count in _balanced_profile_mode_counts(
                                ids, analysis_epoch).items():
                            source_mode_counts[mode] += count

        with torch.amp.autocast('cuda', enabled=use_amp):
            fy_pred, _, _, _, fy_extras = _source_forward(
                model, batch_processor, coords, sw_seq, iri_peak,
                stage, 'FY', profile_ids,
                stage == 'analysis' and not exact_mode_loss, config,
                iri_peak_manager, allowed_profile_ids, source_mode,
                analysis_epoch if stage == 'analysis' else 0)
            cosmic_pred, _, _, _, cosmic_extras = _source_forward(
                model, batch_processor, cosmic_coords, cosmic_sw, cosmic_peak,
                stage, 'COSMIC', cosmic_ids,
                stage == 'analysis' and not exact_mode_loss, config,
                iri_peak_manager, allowed_profile_ids, source_mode,
                analysis_epoch if stage == 'analysis' else 0)
            zero = fy_pred.sum() * 0.0
            fy_mode_losses = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            cosmic_mode_losses = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            if exact_mode_loss:
                fy_loss, fy_mode_losses, fy_mode_active = (
                    exact_mode_profile_losses(
                        fy_extras, target, profile_ids, delta))
                cosmic_loss, cosmic_mode_losses, cosmic_mode_active = (
                    exact_mode_profile_losses(
                        cosmic_extras, cosmic_target, cosmic_ids, delta))
                fy_active = fy_mode_active['M11']
                cosmic_active = cosmic_mode_active['M11']
            else:
                fy_active = _analysis_active_mask(fy_extras)
                cosmic_active = _analysis_active_mask(cosmic_extras)
                active_only = (
                    stage == 'analysis'
                    and config.get('analysis_loss_active_only', False))
                fy_loss = profile_huber_loss(
                    fy_pred, target, profile_ids, delta=delta,
                    valid_mask=fy_active if active_only else None)
                cosmic_loss = profile_huber_loss(
                    cosmic_pred, cosmic_target, cosmic_ids, delta=delta,
                    valid_mask=cosmic_active if active_only else None)
            observation_loss = 0.5 * (fy_loss + cosmic_loss)
            mode_gradient_components = {
                f'fy_{mode.lower()}': (
                    0.5 * _EXACT_MODE_WEIGHTS[mode] * fy_mode_losses[mode])
                for mode in _EXACT_MODE_SOURCES
            }
            mode_gradient_components.update({
                f'cosmic_{mode.lower()}': (
                    0.5 * _EXACT_MODE_WEIGHTS[mode]
                    * cosmic_mode_losses[mode])
                for mode in _EXACT_MODE_SOURCES
            })
            covariance_loss = observation_loss.new_zeros(())
            if (stage == 'analysis'
                    and (config.get('use_covariance_moment_loss', False)
                         or config.get('use_empirical_covariance_loss', False))):
                covariance_loss = 0.5 * (
                    _covariance_training_loss(
                        fy_extras, target, profile_ids, 'FY',
                        batch_processor, config)
                    + _covariance_training_loss(
                        cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC',
                        batch_processor, config))
            weighted_covariance = (
                float(config.get('resolved_covariance_weight', 0.0))
                * covariance_loss)
            fy_gram_losses = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            cosmic_gram_losses = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            fy_gram_coverage = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            cosmic_gram_coverage = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            gram_loss = observation_loss.new_zeros(())
            if stage == 'analysis' and config.get(
                    'use_observation_gram_loss', False):
                if not exact_mode_loss:
                    raise ValueError(
                        'observation Gram loss requires analysis_exact_mode_loss')
                fy_gram_loss, fy_gram_losses, fy_gram_coverage = (
                    observation_gram_whitening_loss(
                        fy_extras, profile_ids, model.kalman_layer,
                        return_details=True))
                cosmic_gram_loss, cosmic_gram_losses, cosmic_gram_coverage = (
                    observation_gram_whitening_loss(
                        cosmic_extras, cosmic_ids, model.kalman_layer,
                        return_details=True))
                gram_loss = 0.5 * (fy_gram_loss + cosmic_gram_loss)
            weighted_gram = (
                float(config.get('resolved_gram_weight', 0.0)) * gram_loss)
            fy_direction_modes = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            cosmic_direction_modes = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            direction_loss = observation_loss.new_zeros(())
            if (stage == 'analysis'
                    and config.get('use_direction_loss', False)):
                if not exact_mode_loss:
                    raise ValueError(
                        'direction loss requires analysis_exact_mode_loss')
                fy_direction, fy_direction_modes, _ = (
                    exact_mode_direction_losses(
                        fy_extras, target, profile_ids, 'FY'))
                cosmic_direction, cosmic_direction_modes, _ = (
                    exact_mode_direction_losses(
                        cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC'))
                direction_loss = 0.5 * (
                    fy_direction + cosmic_direction)
            weighted_direction = (
                float(config.get('resolved_direction_weight', 0.0))
                * direction_loss)
            data_loss = (
                observation_loss + weighted_covariance + weighted_direction
                + weighted_gram)

            vertical_loss, time_loss = _structure_losses(
                model, batch_processor, sw_manager, iri_peak_manager,
                coords, profile_ids, stage, config, allowed_profile_ids)

            if stage == 'background':
                fy_iri = profile_huber_loss(
                    fy_extras['ne_bkg'], fy_extras['ne_iri'],
                    profile_ids, delta=delta)
                cosmic_iri = profile_huber_loss(
                    cosmic_extras['ne_bkg'], cosmic_extras['ne_iri'],
                    cosmic_ids, delta=delta)
                iri_loss = 0.5 * (fy_iri + cosmic_iri)
                increment_loss = observation_loss.new_zeros(())
                weighted_iri = config.get('w_iri', 0.02) * iri_loss
                weighted_increment = increment_loss
                weighted_vertical = (
                    config.get('w_vertical_background', 0.02) * vertical_loss)
                weighted_time = (
                    config.get('w_time_background', 0.01) * time_loss)
                auxiliary_loss = (
                    weighted_iri + weighted_vertical + weighted_time
                )
            else:
                fy_increment = F.smooth_l1_loss(
                    fy_extras['ne_residual'],
                    torch.zeros_like(fy_extras['ne_residual']), beta=delta)
                cosmic_increment = F.smooth_l1_loss(
                    cosmic_extras['ne_residual'],
                    torch.zeros_like(cosmic_extras['ne_residual']), beta=delta)
                increment_loss = 0.5 * (fy_increment + cosmic_increment)
                iri_loss = observation_loss.new_zeros(())
                weighted_iri = iri_loss
                weighted_increment = (
                    config.get('w_increment', 0.01) * increment_loss)
                weighted_vertical = (
                    config.get('w_vertical_analysis', 0.05) * vertical_loss)
                weighted_time = (
                    config.get('w_time_analysis', 0.02) * time_loss)
                auxiliary_loss = (
                    weighted_increment + weighted_vertical + weighted_time
                )
            total_loss = data_loss + auxiliary_loss
            auxiliary_components = {
                'covariance': weighted_covariance,
                'direction': weighted_direction,
                'gram': weighted_gram,
                'iri': weighted_iri,
                'increment': weighted_increment,
                'vertical': weighted_vertical,
                'time': weighted_time,
            }
            gradient_components = {
                **auxiliary_components,
                **mode_gradient_components,
            }

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f'non-finite {stage} loss at epoch={epoch + 1}, batch={batch_idx}')

        decoder = (model.background_decoder if stage == 'background'
                   else (model.density_basis_decoder
                         if hasattr(model, 'density_basis_decoder')
                         else model.kalman_layer))
        first_stage_epoch = (
            epoch == 0 if stage == 'background'
            else epoch == int(config['background_epochs']))
        audit_batch = first_stage_epoch and batch_idx < 100
        if audit_batch:
            observation_grad, auxiliary_grad, ratio, component_grads = _gradient_norms(
                observation_loss,
                auxiliary_loss + weighted_covariance + weighted_direction
                + weighted_gram,
                [p for p in decoder.parameters() if p.requires_grad],
                gradient_components if max_train_batches is not None else None)
            audited_batches += 1
        else:
            observation_grad = total_loss.new_zeros(())
            auxiliary_grad = total_loss.new_zeros(())
            ratio = total_loss.new_zeros(())
            component_grads = {
                name: total_loss.new_zeros(())
                for name in auxiliary_components
            }

        if audit_batch:
            representativeness_stats = {}
            for target_name, extras in (
                    ('fy_target', fy_extras),
                    ('cosmic_target', cosmic_extras)):
                for observation_source in ('FY', 'COSMIC'):
                    active_precision = (
                        extras[f'precision_{observation_source}'] > 0)
                    values = extras[
                        f'representativeness_{observation_source}'][
                            active_precision]
                    prefix = (
                        f'{target_name}_representativeness_'
                        f'{observation_source.lower()}')
                    representativeness_stats[f'{prefix}_mean'] = (
                        values.mean().item() if len(values) else 1.0)
                    representativeness_stats[f'{prefix}_min'] = (
                        values.min().item() if len(values) else 1.0)
                    representativeness_stats[f'{prefix}_max'] = (
                        values.max().item() if len(values) else 1.0)
            diagnostics.append({
                'epoch': epoch + 1,
                'stage': stage,
                'batch': batch_idx + 1,
                'source_mode': source_mode,
                'fy_obs_raw': fy_loss.item(),
                'cosmic_obs_raw': cosmic_loss.item(),
                'fy_obs_weighted': 0.5 * fy_loss.item(),
                'cosmic_obs_weighted': 0.5 * cosmic_loss.item(),
                'fy_active_fraction': fy_active.float().mean().item(),
                'cosmic_active_fraction': cosmic_active.float().mean().item(),
                'covariance_raw': covariance_loss.item(),
                'covariance_weighted': weighted_covariance.item(),
                'gram_raw': gram_loss.item(),
                'gram_weighted': weighted_gram.item(),
                **{
                    f'fy_gram_{mode.lower()}_raw':
                    fy_gram_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_gram_{mode.lower()}_raw':
                    cosmic_gram_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'fy_gram_{mode.lower()}_coverage':
                    fy_gram_coverage[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_gram_{mode.lower()}_coverage':
                    cosmic_gram_coverage[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                'direction_raw': direction_loss.item(),
                'direction_weighted': weighted_direction.item(),
                'iri_raw': iri_loss.item(),
                'increment_raw': increment_loss.item(),
                'vertical_raw': vertical_loss.item(),
                'time_raw': time_loss.item(),
                'iri_weighted': weighted_iri.item(),
                'increment_weighted': weighted_increment.item(),
                'vertical_weighted': weighted_vertical.item(),
                'time_weighted': weighted_time.item(),
                'observation_grad_norm': observation_grad.item(),
                'auxiliary_grad_norm': auxiliary_grad.item(),
                'auxiliary_to_observation_grad': ratio.item(),
                'fy_hx_effective_rank': fy_extras['hx_effective_rank'].mean().item(),
                'cosmic_hx_effective_rank': cosmic_extras['hx_effective_rank'].mean().item(),
                'fy_hx_condition': fy_extras['hx_condition'].mean().item(),
                'cosmic_hx_condition': cosmic_extras['hx_condition'].mean().item(),
                'fy_hx_numeric_rank': fy_extras['hx_numeric_rank'].float().mean().item(),
                'cosmic_hx_numeric_rank': cosmic_extras['hx_numeric_rank'].float().mean().item(),
                **{
                    f'fy_{mode.lower()}_raw': fy_mode_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_{mode.lower()}_raw':
                    cosmic_mode_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'fy_{mode.lower()}_weighted': (
                        0.5 * _EXACT_MODE_WEIGHTS[mode]
                        * fy_mode_losses[mode].item())
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_{mode.lower()}_weighted': (
                        0.5 * _EXACT_MODE_WEIGHTS[mode]
                        * cosmic_mode_losses[mode].item())
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'fy_direction_{mode.lower()}_raw':
                    fy_direction_modes[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_direction_{mode.lower()}_raw':
                    cosmic_direction_modes[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'{name}_grad_norm': value.item()
                    for name, value in component_grads.items()
                },
                **{
                    f'{name}_to_observation_grad': (
                        value / observation_grad.clamp_min(1e-12)
                    ).item()
                    for name, value in component_grads.items()
                },
                **representativeness_stats,
            })

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.get('grad_clip', 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.get('grad_clip', 1.0))
            optimizer.step()

        stats['total'] += total_loss.item()
        stats['observation'] += observation_loss.item()
        stats['covariance'] += covariance_loss.item()
        stats['direction'] += direction_loss.item()
        stats['gram'] += gram_loss.item()
        stats['fy_obs'] += fy_loss.item()
        stats['cosmic_obs'] += cosmic_loss.item()
        for mode in _EXACT_MODE_SOURCES:
            stats[f'fy_{mode.lower()}'] += fy_mode_losses[mode].item()
            stats[f'cosmic_{mode.lower()}'] += cosmic_mode_losses[mode].item()
            stats[f'fy_direction_{mode.lower()}'] += (
                fy_direction_modes[mode].item())
            stats[f'cosmic_direction_{mode.lower()}'] += (
                cosmic_direction_modes[mode].item())
        stats['fy_active_fraction'] += fy_active.float().mean().item()
        stats['cosmic_active_fraction'] += cosmic_active.float().mean().item()
        stats['iri'] += iri_loss.item()
        stats['increment'] += increment_loss.item()
        stats['vertical'] += vertical_loss.item()
        stats['time'] += time_loss.item()
        stats['weighted_iri'] += weighted_iri.item()
        stats['weighted_increment'] += weighted_increment.item()
        stats['weighted_covariance'] += weighted_covariance.item()
        stats['weighted_direction'] += weighted_direction.item()
        stats['weighted_gram'] += weighted_gram.item()
        stats['weighted_vertical'] += weighted_vertical.item()
        stats['weighted_time'] += weighted_time.item()
        stats['gradient_ratio'] += ratio.item()
        for name, value in component_grads.items():
            stats[f'gradient_ratio_{name}'] += (
                value / observation_grad.clamp_min(1e-12)).item()
        for mode in _EXACT_MODE_SOURCES:
            stats[f'fy_gram_{mode.lower()}_coverage'] += (
                fy_gram_coverage[mode].item())
            stats[f'cosmic_gram_{mode.lower()}_coverage'] += (
                cosmic_gram_coverage[mode].item())
        stats['K_FY_mean'] += fy_extras['K_FY'].mean().item()
        stats['K_COSMIC_mean'] += fy_extras['K_COSMIC'].mean().item()
        stats['innov_FY_norm'] += fy_extras['innov_FY'].norm(dim=-1).mean().item()
        stats['innov_COSMIC_norm'] += fy_extras['innov_COSMIC'].norm(dim=-1).mean().item()
        stats['inflation'] += fy_extras['inflation_scale'].item()
        stats['ne_delta_abs'] += fy_extras['ne_residual'].abs().mean().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f'  [{batch_idx + 1:>5}/{epoch_length}] {stage} '
                f'loss={total_loss.item():.4f} FY={fy_loss.item():.4f} '
                f'COSMIC={cosmic_loss.item():.4f} '
                f'V={vertical_loss.item():.4f} T={time_loss.item():.4f} '
                f'({time.time() - started:.0f}s)')
        processed_batches += 1
        if (max_train_batches is not None
                and processed_batches >= int(max_train_batches)):
            break

    batches = max(1, processed_batches)
    result = {key: value / batches for key, value in stats.items()}
    result['gradient_ratio'] = (
        stats['gradient_ratio'] / audited_batches if audited_batches else 0.0)
    for name in gradient_components:
        result[f'gradient_ratio_{name}'] = (
            stats[f'gradient_ratio_{name}'] / audited_batches
            if audited_batches else 0.0)
    result['gradient_audit_batches'] = audited_batches
    result['processed_batches'] = processed_batches
    result['source_mode_counts'] = source_mode_counts
    result['source_mode_count_unit'] = (
        'batches_all_modes' if exact_mode_loss else
        'profiles' if config.get(
            'source_mode_schedule') == 'balanced_profile_112'
        else 'batches')
    return result['total'], result, diagnostics


def _profile_metrics(predictions, targets, profile_ids):
    predictions = np.concatenate(predictions).reshape(-1)
    targets = np.concatenate(targets).reshape(-1)
    profile_ids = np.concatenate(profile_ids).reshape(-1)
    order = np.argsort(profile_ids, kind='stable')
    sorted_ids = profile_ids[order]
    squared = (predictions[order] - targets[order]) ** 2
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_ids)])
    profile_mse = np.add.reduceat(squared, starts) / counts
    pred_std = float(np.std(predictions))
    target_std = float(np.std(targets))
    pearson_r = (float(np.corrcoef(targets, predictions)[0, 1])
                 if min(pred_std, target_std) > 1e-12 else float('nan'))
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    covariance = float(np.mean(
        (predictions - pred_mean) * (targets - target_mean)))
    ccc = float(
        2.0 * covariance
        / (pred_std ** 2 + target_std ** 2
           + (pred_mean - target_mean) ** 2 + 1e-12))
    return {
        'profile_rmse': float(np.sqrt(profile_mse).mean()),
        'mae': float(np.mean(np.abs(predictions - targets))),
        'rmse': float(np.sqrt(np.mean((predictions - targets) ** 2))),
        'r2': float(
            1.0 - np.sum((predictions - targets) ** 2)
            / (np.sum((targets - targets.mean()) ** 2) + 1e-12)),
        'pearson_r': pearson_r,
        'ccc': ccc,
    }


def _background_stratified_metrics(prediction, raw_prediction, target, coords,
                                   residual_raw, residual, trust_gate,
                                   alt_range=(120.0, 500.0)):
    """Pointwise Background diagnostics by altitude and local day/night."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    raw_prediction = np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    residual_raw = np.asarray(residual_raw, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    trust_gate = np.asarray(trust_gate, dtype=np.float64).reshape(-1)
    local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    finite = (np.isfinite(prediction) & np.isfinite(raw_prediction)
              & np.isfinite(target) & np.isfinite(residual_raw)
              & np.isfinite(residual) & np.isfinite(trust_gate))

    def point_metrics(values, selected):
        selected = selected & finite
        if not selected.any():
            return {'n': 0, 'rmse': None, 'bias': None, 'mae': None}
        error = values[selected] - target[selected]
        return {
            'n': int(selected.sum()),
            'rmse': float(np.sqrt(np.mean(error * error))),
            'bias': float(np.mean(error)),
            'mae': float(np.mean(np.abs(error))),
        }

    cells = {}
    domain_min, domain_max = map(float, alt_range)
    edges = [domain_min] + [
        edge for edge in (200.0, 300.0)
        if domain_min < edge < domain_max] + [domain_max]
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        label = f'{lower:g}-{upper:g}'
        altitude = (coords[:, 2] >= lower) & (
            coords[:, 2] <= upper if index == len(edges) - 2
            else coords[:, 2] < upper)
        for period, period_mask in (('night', night), ('day', ~night)):
            selected = altitude & period_mask
            raw = point_metrics(raw_prediction, selected)
            m00 = point_metrics(prediction, selected)
            cells[f'{label}_{period}'] = {
                'raw_iri': raw,
                'M00': m00,
                'delta_rmse': (
                    None if raw['rmse'] is None or m00['rmse'] is None
                    else float(m00['rmse'] - raw['rmse'])),
                'delta_bias': (
                    None if raw['bias'] is None or m00['bias'] is None
                    else float(m00['bias'] - raw['bias'])),
                'residual_raw_rms': float(np.sqrt(np.mean(
                    residual_raw[selected & finite] ** 2)))
                if (selected & finite).any() else None,
                'residual_rms': float(np.sqrt(np.mean(
                    residual[selected & finite] ** 2)))
                if (selected & finite).any() else None,
                'trust_gate_mean': float(np.mean(trust_gate[selected & finite]))
                if (selected & finite).any() else None,
            }
    transition = ((coords[:, 2] >= 200.0) & (coords[:, 2] < 300.0)
                  & night & finite)
    return {
        'cells': cells,
        'trust_gate': {
            'mean': float(np.mean(trust_gate[finite])) if finite.any() else None,
            'p05': float(np.quantile(trust_gate[finite], 0.05)) if finite.any() else None,
            'p50': float(np.quantile(trust_gate[finite], 0.50)) if finite.any() else None,
            'p95': float(np.quantile(trust_gate[finite], 0.95)) if finite.any() else None,
            'strict_suppression_fraction': float(np.mean(
                trust_gate[finite] <= 1e-8)) if finite.any() else None,
        },
        'raw_residual_rms': float(np.sqrt(np.mean(residual_raw[finite] ** 2)))
        if finite.any() else None,
        'gated_residual_rms': float(np.sqrt(np.mean(residual[finite] ** 2)))
        if finite.any() else None,
        'raw_residual_max_abs': float(np.max(np.abs(residual_raw[finite])))
        if finite.any() else None,
        'gated_residual_max_abs': float(np.max(np.abs(residual[finite])))
        if finite.any() else None,
        'transition_raw_residual_rms': float(np.sqrt(np.mean(
            residual_raw[transition] ** 2))) if transition.any() else None,
        'transition_gated_residual_rms': float(np.sqrt(np.mean(
            residual[transition] ** 2))) if transition.any() else None,
    }


@torch.no_grad()
def _evaluate_source(model, loader, batch_processor, device, stage, source,
                     sw_manager, iri_peak_manager, config,
                     allowed_profile_ids=None):
    predictions, targets, profile_ids_all = [], [], []
    raw_predictions = []
    coords_all = []
    raw_residuals = []
    residuals = []
    trust_gates = []
    max_batches = config.get('max_validation_batches')
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        coords, target, profile_ids = _unpack_source_batch(batch, device)
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
        prediction, _, _, _, extras = _source_forward(
            model, batch_processor, coords, sw_seq, iri_peak,
            stage, source, profile_ids, False, config, iri_peak_manager,
            allowed_profile_ids)
        predictions.append(prediction.cpu().numpy())
        targets.append(target.cpu().numpy())
        profile_ids_all.append(profile_ids.cpu().numpy())
        if stage == 'background':
            raw_predictions.append(extras['ne_iri'].cpu().numpy())
            coords_all.append(coords.cpu().numpy())
            raw_residuals.append(extras['background_residual_raw'].cpu().numpy())
            residuals.append(extras['background_residual'].cpu().numpy())
            trust_gates.append(extras['background_trust_gate'].cpu().numpy())
    result = _profile_metrics(predictions, targets, profile_ids_all)
    if stage == 'background':
        raw = _profile_metrics(raw_predictions, targets, profile_ids_all)
        residual = np.concatenate(predictions).reshape(-1)
        raw_values = np.concatenate(raw_predictions).reshape(-1)
        result.update({
            'raw_iri_profile_rmse': raw['profile_rmse'],
            'raw_iri_rmse': raw['rmse'],
            'raw_iri_mae': raw['mae'],
            'raw_iri_ccc': raw['ccc'],
            'raw_iri_pearson_r': raw['pearson_r'],
            'rmse_change_M00_minus_raw': result['rmse'] - raw['rmse'],
            'profile_rmse_change_M00_minus_raw': (
                result['profile_rmse'] - raw['profile_rmse']),
            'background_residual_rms': float(
                np.sqrt(np.mean((residual - raw_values) ** 2))),
        })
        result['background_stratified'] = _background_stratified_metrics(
            residual, raw_values, np.concatenate(targets).reshape(-1),
            np.concatenate(coords_all), np.concatenate(raw_residuals),
            np.concatenate(residuals), np.concatenate(trust_gates),
            config.get('alt_range', (120.0, 500.0)))
    return result


def validate(model, val_loader, batch_processor, device, config,
             sw_manager, iri_peak_manager, cosmic_val_loader, stage='analysis',
             allowed_profile_ids=None):
    model.eval()
    fy = _evaluate_source(
        model, val_loader, batch_processor, device, stage, 'FY',
        sw_manager, iri_peak_manager, config, allowed_profile_ids)
    cosmic = _evaluate_source(
        model, cosmic_val_loader, batch_processor, device, stage, 'COSMIC',
        sw_manager, iri_peak_manager, config, allowed_profile_ids)
    score = 0.5 * (fy['profile_rmse'] + cosmic['profile_rmse'])
    metrics = {
        'score': score,
        'fy_profile_rmse': fy['profile_rmse'],
        'cosmic_profile_rmse': cosmic['profile_rmse'],
        'mae': 0.5 * (fy['mae'] + cosmic['mae']),
        'rmse': 0.5 * (fy['rmse'] + cosmic['rmse']),
        'r2': 0.5 * (fy['r2'] + cosmic['r2']),
        'fy_ccc': fy['ccc'],
        'cosmic_ccc': cosmic['ccc'],
        'ccc': 0.5 * (fy['ccc'] + cosmic['ccc']),
        'fy_pearson_r': fy['pearson_r'],
        'cosmic_pearson_r': cosmic['pearson_r'],
        'pearson_r': 0.5 * (fy['pearson_r'] + cosmic['pearson_r']),
    }
    if stage == 'background':
        metrics.update({
            'fy_raw_iri_profile_rmse': fy['raw_iri_profile_rmse'],
            'cosmic_raw_iri_profile_rmse': cosmic['raw_iri_profile_rmse'],
            'fy_raw_iri_rmse': fy['raw_iri_rmse'],
            'cosmic_raw_iri_rmse': cosmic['raw_iri_rmse'],
            'fy_raw_iri_ccc': fy['raw_iri_ccc'],
            'cosmic_raw_iri_ccc': cosmic['raw_iri_ccc'],
            'fy_raw_iri_pearson_r': fy['raw_iri_pearson_r'],
            'cosmic_raw_iri_pearson_r': cosmic['raw_iri_pearson_r'],
            'fy_rmse_change_M00_minus_raw': fy[
                'rmse_change_M00_minus_raw'],
            'cosmic_rmse_change_M00_minus_raw': cosmic[
                'rmse_change_M00_minus_raw'],
            'fy_profile_rmse_change_M00_minus_raw': fy[
                'profile_rmse_change_M00_minus_raw'],
            'cosmic_profile_rmse_change_M00_minus_raw': cosmic[
                'profile_rmse_change_M00_minus_raw'],
            'fy_background_residual_rms': fy['background_residual_rms'],
            'cosmic_background_residual_rms': cosmic[
                'background_residual_rms'],
            'fy_background_stratified': fy['background_stratified'],
            'cosmic_background_stratified': cosmic['background_stratified'],
        })
    return score, metrics


def _development_selection_key(metrics, config):
    """Return a JSON-safe lexicographic key; larger is always better."""
    semantics = config.get(
        'checkpoint_selection_semantics', 'mean_profile_rmse_v1')
    if semantics == 'mean_profile_rmse_v1':
        values = (-float(metrics['score']),)
    elif semantics == 'mean_ccc_then_rmse_then_pearson_v1':
        values = (
            float(metrics['ccc']), -float(metrics['rmse']),
            float(metrics['pearson_r']))
    else:
        raise ValueError(f'unsupported checkpoint selection semantics: {semantics}')
    return list(values) if np.isfinite(values).all() else None


def _mad_variance(values, sigma_min=0.05, sigma_max=0.40):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('cannot calibrate R from empty residuals')
    median = np.median(values)
    sigma = float(np.clip(
        1.4826 * np.median(np.abs(values - median)),
        sigma_min, sigma_max))
    return sigma ** 2


def _stratified_variance_table(data, config):
    residual = np.asarray(data['residual'], dtype=np.float64).reshape(-1)
    altitude = np.asarray(data['altitude'], dtype=np.float64).reshape(-1)
    local_time = np.remainder(
        np.asarray(data['local_time'], dtype=np.float64).reshape(-1), 24.0)
    profile_id = np.asarray(data['profile_id']).reshape(-1)
    if not (len(residual) == len(altitude) == len(local_time) == len(profile_id)):
        raise ValueError('R calibration arrays must have equal lengths')

    valid = (
        np.isfinite(residual) & np.isfinite(altitude)
        & np.isfinite(local_time) & np.isfinite(profile_id))
    residual = residual[valid]
    altitude = altitude[valid]
    local_time = local_time[valid]
    profile_id = profile_id[valid]

    sigma_min = float(config.get('r_sigma_min', 0.05))
    sigma_max = float(config.get('r_sigma_max', 0.40))
    min_profiles = int(config.get('r_min_profiles', 200))
    shrinkage_profiles = float(config.get('r_shrinkage_profiles', 200))
    if not (0 < sigma_min <= sigma_max):
        raise ValueError('R sigma bounds must satisfy 0 < min <= max')
    if min_profiles < 1 or shrinkage_profiles < 0:
        raise ValueError('R profile thresholds must be non-negative')

    global_variance = _mad_variance(residual, sigma_min, sigma_max)
    domain_min, domain_max = map(float, config.get(
        'alt_range', (120.0, 500.0)))
    altitude_edges = [domain_min] + [
        edge for edge in (200.0, 300.0)
        if domain_min < edge < domain_max] + [domain_max]
    table = np.full((len(altitude_edges) - 1, 2), global_variance,
                    dtype=np.float64)
    altitude_index = np.searchsorted(
        np.asarray(altitude_edges[1:-1]), altitude, side='right')
    day_index = ((local_time >= 6.0) & (local_time < 18.0)).astype(np.int64)
    cells = []
    for altitude_cell in range(len(altitude_edges) - 1):
        row = []
        for day_cell in range(2):
            cell_mask = (
                (altitude_index == altitude_cell) & (day_index == day_cell))
            point_count = int(cell_mask.sum())
            profile_count = int(np.unique(profile_id[cell_mask]).size)
            raw_variance = (
                _mad_variance(residual[cell_mask], sigma_min, sigma_max)
                if point_count else None)
            fallback = profile_count < min_profiles
            alpha = (
                0.0 if fallback else
                profile_count / (profile_count + shrinkage_profiles))
            if not fallback:
                table[altitude_cell, day_cell] = (
                    alpha * raw_variance + (1.0 - alpha) * global_variance)
            row.append({
                'point_count': point_count,
                'profile_count': profile_count,
                'raw_sigma': (
                    None if raw_variance is None else float(np.sqrt(raw_variance))),
                'raw_variance': (
                    None if raw_variance is None else float(raw_variance)),
                'alpha': float(alpha),
                'fallback_to_global': fallback,
                'final_sigma': float(np.sqrt(table[altitude_cell, day_cell])),
                'final_variance': float(table[altitude_cell, day_cell]),
            })
        cells.append(row)

    metadata = {
        'altitude_bins_km': [
            [float(lower), float(upper)]
            for lower, upper in zip(altitude_edges[:-1], altitude_edges[1:])],
        'local_time_columns': ['night', 'day'],
        'day_definition': '06:00 <= LT < 18:00',
        'global': {
            'point_count': int(residual.size),
            'profile_count': int(np.unique(profile_id).size),
            'sigma': float(np.sqrt(global_variance)),
            'variance': float(global_variance),
        },
        'cells': cells,
        'calibrated_variance_table': table.tolist(),
    }
    return float(global_variance), table, metadata


@torch.no_grad()
def _calibrate_fixed_r(model, fy_loader, cosmic_loader, device, sw_manager,
                       iri_peak_manager, batch_processor, config):
    model.eval()

    def collect(loader, source):
        collected = {
            'residual': [], 'altitude': [], 'local_time': [], 'profile_id': []}
        sampler = loader.batch_sampler
        previous_shuffle = getattr(sampler, 'shuffle', None)
        if previous_shuffle is not None:
            sampler.shuffle = False
        try:
            for batch_index, batch in enumerate(loader):
                max_batches = config.get('r_calibration_batches')
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                coords, target, profile_ids = _unpack_source_batch(batch, device)
                sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
                iri_peak = (iri_peak_manager.get_iri_peak(coords)
                            if iri_peak_manager is not None else None)
                _, _, _, _, extras = _source_forward(
                    model, batch_processor, coords, sw_seq, iri_peak,
                    'background', source, profile_ids, False, config,
                    iri_peak_manager)
                collected['residual'].append(
                    (extras['ne_bkg'] - target).flatten().cpu().numpy())
                collected['altitude'].append(coords[:, 2].cpu().numpy())
                collected['local_time'].append(torch.remainder(
                    coords[:, 3] + coords[:, 1] / 15.0, 24.0).cpu().numpy())
                collected['profile_id'].append(
                    profile_ids.flatten().cpu().numpy())
        finally:
            if previous_shuffle is not None:
                sampler.shuffle = previous_shuffle
        if not collected['residual']:
            raise ValueError(f'{source} R calibration selected no training data')
        return {
            key: np.concatenate(parts) for key, parts in collected.items()}

    fy_global, fy_table, fy_report = _stratified_variance_table(
        collect(fy_loader, 'FY'), config)
    cosmic_global, cosmic_table, cosmic_report = _stratified_variance_table(
        collect(cosmic_loader, 'COSMIC'), config)
    mode = config.get('r_mode', 'global')
    if mode != 'global':
        raise ValueError('v7 density-observation ETKF requires r_mode=global')
    fy_active = np.full((3, 2), fy_global)
    cosmic_active = np.full((3, 2), cosmic_global)
    model.kalman_layer.set_observation_variances(
        fy_global, cosmic_global, fy_active, cosmic_active)

    report = {
        'schema_version': 1,
        'mode': mode,
        'seed': int(config['seed']),
        'train_only': True,
        'points_per_profile': int(config.get('profile_points_per_epoch', 8)),
        'sigma_clip_dex': [
            float(config.get('r_sigma_min', 0.05)),
            float(config.get('r_sigma_max', 0.40)),
        ],
        'min_profiles': int(config.get('r_min_profiles', 200)),
        'shrinkage_profiles': float(config.get('r_shrinkage_profiles', 200)),
        'FY': {**fy_report, 'active_variance_table': fy_active.tolist()},
        'COSMIC': {
            **cosmic_report, 'active_variance_table': cosmic_active.tolist()},
    }
    manifest_path = os.path.join(config['save_dir'], 'run_manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        report['data_identity'] = manifest.get('data_identity', {})
        report['code_identity'] = manifest.get('code_identity', {})
    seed_path = config.get('background_seed_ckpt')
    if seed_path and os.path.isfile(seed_path):
        report['background_seed_checkpoint'] = {
            'path': os.path.abspath(seed_path),
            'sha256': _sha256_file(seed_path),
        }
    output_path = os.path.join(config['save_dir'], 'r_calibration.json')
    temp_path = f'{output_path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_path)
    print(
        f'  冻结有效R [{mode}]: FY={fy_global:.6f}, '
        f'COSMIC={cosmic_global:.6f}')
    return report


def _load_iri_peak_manager(config, device):
    hmf2_path = config.get('iri_hmf2_path')
    nmf2_path = config.get('iri_nmf2_path')
    if not (hmf2_path and nmf2_path
            and os.path.exists(hmf2_path) and os.path.exists(nmf2_path)):
        return None
    from data_managers.iri_peak_manager import IRIPeakManager
    return IRIPeakManager(
        hmf2_path=hmf2_path, nmf2_path=nmf2_path, device=device)


def _resume_needs_analysis_setup(resume_state, start_epoch, background_epochs):
    if resume_state is None:
        return False
    stage = resume_state.get('stage')
    if stage == 'background':
        if start_epoch > background_epochs:
            raise ValueError('background checkpoint has inconsistent completed_epochs')
        return start_epoch == background_epochs
    if stage == 'analysis':
        if start_epoch <= background_epochs:
            raise ValueError('analysis checkpoint has inconsistent completed_epochs')
        return False
    raise ValueError(f'checkpoint has unknown training stage: {stage}')


def _reset_random_seeds(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _architecture_signature(config):
    signature = {
        'alt_range': [float(value) for value in config.get(
            'alt_range', (120.0, 500.0))],
        'model_domain_semantics': config.get(
            'model_domain_semantics', 'legacy_120_500_domain_v1'),
        'basis_dim': int(config.get('basis_dim', 64)),
        'enkf_n_members': int(config.get('enkf_n_members', 8)),
        'enkf_pert_hidden': int(config.get('enkf_pert_hidden', 64)),
        'enkf_anomaly_parameterization': config.get(
            'enkf_anomaly_parameterization', 'legacy_independent'),
        'enkf_scale_init': float(config.get('enkf_scale_init', 1.1)),
        'enkf_scale_condition_max': float(
            config.get('enkf_scale_condition_max', 3.0)),
        'density_basis_semantics': config.get(
            'density_basis_semantics', 'query_conditioned'),
        'analysis_state_semantics': config.get(
            'analysis_state_semantics', 'legacy_feature_increment'),
        'context_semantics': config.get(
            'context_semantics', 'query_conditioning'),
        'mode_basis_semantics': config.get(
            'mode_basis_semantics', 'learned_density_basis'),
        'background_trust_gate_enabled': bool(config.get(
            'background_trust_gate_enabled', False)),
        'background_trust_gate_semantics': config.get(
            'background_trust_gate_semantics', 'disabled'),
        'background_trust_gate_altitude_core_km': float(config.get(
            'background_trust_gate_altitude_core_km', 200.0)),
        'background_trust_gate_altitude_transition_km': float(config.get(
            'background_trust_gate_altitude_transition_km', 100.0)),
        'background_trust_gate_night_cosine_offset': float(config.get(
            'background_trust_gate_night_cosine_offset', 0.2)),
        'background_trust_gate_night_cosine_scale': float(config.get(
            'background_trust_gate_night_cosine_scale', 0.2)),
        'background_trust_gate_dip_core': float(config.get(
            'background_trust_gate_dip_core', 0.25)),
        'background_trust_gate_dip_transition': float(config.get(
            'background_trust_gate_dip_transition', 0.25)),
    }
    if config.get('assimilation_semantics') == 'continuous_physical_local_letkf':
        signature.update({
            'assimilation_semantics': config['assimilation_semantics'],
            'neighbor_directory_semantics': config.get(
                'neighbor_directory_semantics', 'profile_center_legacy'),
            'physical_localization_space_km': float(config.get(
                'physical_localization_space_km', 1800.0)),
            'physical_localization_time_hours': float(config.get(
                'physical_localization_time_hours', 1.5)),
            'observation_chunk_size': int(config.get(
                'observation_chunk_size', 4096)),
        })
    return signature


def _restrict_training_profiles(loader, fraction, seed, source, manifest_path):
    if not 0.0 < fraction <= 1.0:
        raise ValueError('train_profile_fraction must be in (0, 1]')
    sampler = loader.batch_sampler
    profile_groups = [
        indices for groups in sampler.profiles_by_bin.values()
        for indices in groups
    ]
    available = np.array(sorted({
        int(loader.dataset.profile_ids[indices[0]])
        for indices in profile_groups
    }), dtype=np.int64)
    if fraction == 1.0:
        return {'available': int(len(available)), 'selected': int(len(available))}
    if not manifest_path:
        raise ValueError(
            'profile_subset_manifest is required when train_profile_fraction < 1')
    path = os.path.abspath(manifest_path)
    manifest = {}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        if (manifest.get('seed') != int(seed)
                or not np.isclose(manifest.get('fraction', -1.0), fraction)):
            raise ValueError('profile subset manifest seed/fraction mismatch')
    sources = manifest.setdefault('sources', {})
    if source in sources:
        selected = np.asarray(sources[source]['profile_ids'], dtype=np.int64)
        if not np.all(np.isin(selected, available)):
            raise ValueError(
                f'{source} profile subset manifest contains unavailable IDs')
    else:
        rng = np.random.default_rng(
            int(seed) + (0 if source == 'FY' else 1))
        count = max(1, int(round(len(available) * fraction)))
        selected = np.sort(rng.choice(available, count, replace=False))
        sources[source] = {
            'dataset_path': os.path.abspath(loader.dataset.npy_path),
            'available_profiles': int(len(available)),
            'selected_profiles': int(len(selected)),
            'profile_ids': selected.tolist(),
        }
        manifest.update({
            'schema_version': 1,
            'seed': int(seed),
            'fraction': float(fraction),
        })
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        temp_path = f'{path}.tmp'
        with open(temp_path, 'w', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    selected_set = set(selected.tolist())
    sampler.profiles_by_bin = {
        bin_id: [
            indices for indices in groups
            if int(loader.dataset.profile_ids[indices[0]]) in selected_set
        ]
        for bin_id, groups in sampler.profiles_by_bin.items()
    }
    sampler.profiles_by_bin = {
        bin_id: groups for bin_id, groups in sampler.profiles_by_bin.items()
        if groups
    }
    return {'available': int(len(available)), 'selected': int(len(selected))}


def _training_stratum_weights(loader):
    """Profile-balanced inverse frequencies for low/high altitude and day/night."""
    sampler = loader.batch_sampler
    counts = np.zeros(4, dtype=np.float64)
    profiles = 0
    for groups in sampler.profiles_by_bin.values():
        for indices in groups:
            rows = loader.dataset.selected_indices[indices]
            values = np.asarray(loader.dataset.data[rows, :4])
            local_time = np.remainder(
                values[:, 3] + values[:, 1] / 15.0, 24.0)
            night = (local_time < 6.0) | (local_time >= 18.0)
            cells = (values[:, 2] >= 300.0).astype(np.int64) * 2
            cells += night.astype(np.int64)
            profile_counts = np.bincount(cells, minlength=4).astype(np.float64)
            counts += profile_counts / max(profile_counts.sum(), 1.0)
            profiles += 1
    if profiles == 0 or np.any(counts <= 0):
        raise ValueError(
            f'covariance stratum calibration has empty cells: {counts.tolist()}')
    frequencies = counts / counts.sum()
    weights = 1.0 / (4.0 * frequencies)
    return {
        'profiles': profiles,
        'frequencies': frequencies.tolist(),
        'weights': weights.tolist(),
    }


def _load_background_seed(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    expected = model.state_dict()
    background_prefixes = (
        'iri_proxy.', 'iri_align_net.', 'sw_encoder.', 'sw_freq_branch.',
        'sw_gate.', 'background_decoder.')
    required = {
        name for name in expected if name.startswith(background_prefixes)}
    missing = required - set(state)
    mismatched = {
        name for name in required & set(state)
        if state[name].shape != expected[name].shape}
    if missing or mismatched:
        raise RuntimeError(
            'Background seed is architecture-incompatible: '
            f'missing={sorted(missing)}, shape_mismatch={sorted(mismatched)}')
    migrated = dict(expected)
    for name in required:
        migrated[name] = state[name]
    model.load_state_dict(migrated, strict=True)


def _background_epoch_length(fy_loader, cosmic_loader, stage):
    """Return the source schedule length for one training epoch."""
    return (max(len(fy_loader), len(cosmic_loader))
            if stage == 'background' else len(fy_loader))


def _assert_fresh_background_identity(model, sw_manager, iri_peak_manager, device):
    """The zero-initialized Background must start exactly at raw IRI."""
    lower = float(model.alt_min)
    upper = float(model.alt_max)
    coords = torch.tensor(
        [[-11.9, -76.0, lower, 4.0],
         [0.0, 120.0, 0.5 * (lower + upper), 240.0]],
        dtype=torch.float32, device=device)
    sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
    peak = (iri_peak_manager.get_iri_peak(coords)
            if iri_peak_manager is not None else None)
    encoded = model.encode_background(coords, sw_seq, iri_peak=peak)
    if not torch.equal(encoded['ne_bkg'], encoded['ne_iri']):
        error = (encoded['ne_bkg'] - encoded['ne_iri']).abs().max().item()
        raise RuntimeError(
            f'fresh Background invariant failed: M00 != Raw IRI (max={error:.3e})')


def train_fsia(config=None):
    """Train Background first, freeze it, then train ETKF Analysis."""
    config = get_config_mdia() if config is None else config
    if config.get('analysis_state_semantics', 'legacy_feature_increment') != (
            'legacy_feature_increment'):
        raise ValueError(
            'query-local physical states are failed audit shadows; '
            'M2-O legacy_feature_increment is the only trainable model')
    if config.get('r_mode') != 'global':
        raise ValueError('density-observation ETKF requires r_mode=global')
    if not config.get('use_distance_localization', False):
        raise ValueError(
            'density-observation ETKF requires continuous localization')
    if config.get('assimilation_semantics') == 'continuous_physical_local_letkf':
        if not config.get('use_physical_localization', False):
            raise ValueError('M2-V requires physical Gaspari-Cohn localization')
        if config.get('representativeness_kernel_path'):
            raise ValueError('M2-V must not use source-labelled representativeness tables')
        if config.get('use_empirical_covariance_loss', False):
            raise ValueError('M2-V must not use empirical covariance supervision')
        if config.get('neighbor_directory_semantics') != (
                'token_exact_positive_support_v1'):
            raise ValueError('M2-V requires the exact token directory')
        if int(config.get('observation_chunk_size', 4096)) < 1:
            raise ValueError('M2-V observation_chunk_size must be positive')
        model_domain = config.get(
            'model_domain_semantics', 'legacy_120_500_domain_v1')
        if model_domain not in (
                'legacy_120_500_domain_v1', 'strict_200_500_domain_v1'):
            raise ValueError('unrecognized model-domain semantics')
        expected_alt_range = (
            (200.0, 500.0) if model_domain == 'strict_200_500_domain_v1'
            else (120.0, 500.0))
        if tuple(map(float, config.get('alt_range', ()))) != expected_alt_range:
            raise ValueError('model-domain semantics and alt_range disagree')
        expected_format = 13 if model_domain == 'strict_200_500_domain_v1' else 12
        if int(config.get('checkpoint_format_version', 0)) != expected_format:
            raise ValueError('model-domain semantics and checkpoint format disagree')
        if not config.get('eval_only'):
            if model_domain == 'strict_200_500_domain_v1':
                expected_background_semantics = (
                    'qc_v2_date_blocked_train_only_m2w_200_500_continuous_trust_gate_v1'
                    if config.get('background_trust_gate_enabled', False)
                    else 'qc_v2_date_blocked_train_only_m2w_200_500_v1')
            else:
                expected_background_semantics = (
                    'qc_v2_date_blocked_train_only_continuous_trust_gate_v1'
                    if config.get('background_trust_gate_enabled', False)
                    else 'qc_v2_date_blocked_train_only')
            if config.get('background_training_semantics') != expected_background_semantics:
                raise ValueError(
                    'M2-V training requires the configured QC-v2 Background semantics')
            if config.get('background_seed_ckpt'):
                raise ValueError(
                    'QC-v2 Background training cannot use an external seed checkpoint')
        if config.get('background_trust_gate_enabled', False):
            if config.get('background_trust_gate_semantics') != (
                    'fixed_altitude_localtime_dip_smoothstep_v1'):
                raise ValueError('M2-V trust-gate semantic is not recognized')
    if (config.get('use_direction_loss', False)
            and not config.get('analysis_exact_mode_loss', False)):
        raise ValueError(
            'direction loss requires analysis_exact_mode_loss')
    if (config.get('use_covariance_moment_loss', False)
            and config.get('use_empirical_covariance_loss', False)):
        raise ValueError('choose only one covariance supervision loss')
    if (config.get('use_empirical_covariance_loss', False)
            and not config.get('representativeness_kernel_path')):
        raise ValueError(
            'empirical covariance loss requires train-only covariance cells')
    if config.get('use_observation_gram_loss', False):
        required_gram = {
            'analysis_exact_mode_loss': True,
            'basis_dim': 64,
            'enkf_n_members': 8,
            'enkf_anomaly_parameterization': 'orthogonal_factor',
            'density_basis_semantics': 'endpoint_context_symmetric',
        }
        mismatched = {
            key: (config.get(key), expected)
            for key, expected in required_gram.items()
            if config.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                f'observation Gram loss requires the D64/N8 endpoint basis: {mismatched}')
        if config.get('analysis_state_semantics', 'legacy_feature_increment') != (
                'legacy_feature_increment'):
            raise ValueError('observation Gram loss cannot train physical coefficient states')
        if not 0.0 < float(config.get('gram_gradient_target', 0.02)) <= 1.0:
            raise ValueError('gram_gradient_target must be in (0, 1]')
        if int(config.get('gram_calibration_batches', 20)) < 1:
            raise ValueError('gram_calibration_batches must be positive')
    device = torch.device(config['device'])
    _reset_random_seeds(config['seed'], device)

    background_epochs = int(config.get('background_epochs', 5))
    analysis_epochs = int(config.get('analysis_epochs', 5))
    background_only = bool(config.get('background_only', False))
    total_epochs = (background_epochs if background_only
                    else background_epochs + analysis_epochs)
    if background_epochs <= 0 or (not background_only and analysis_epochs <= 0):
        raise ValueError(
            'background_epochs and analysis_epochs must be positive for full training')
    architecture = _architecture_signature(config)
    format_version = int(config.get(
        'checkpoint_format_version',
        8 if architecture['enkf_anomaly_parameterization']
        == 'orthogonal_factor' else 7))

    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device,
    )
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(
        config['iri_proxy_path'], map_location=device, weights_only=True))
    iri_proxy.eval()
    iri_peak_manager = _load_iri_peak_manager(config, device)

    date_split, date_split_identity = _load_or_create_date_split(config)
    if date_split_identity is not None:
        config['resolved_date_split'] = date_split_identity
    loader_split_days = (
        {
            'train': date_split['train'],
            'development': date_split['development'],
        }
        if date_split is not None else None)
    loader_kwargs = dict(
        batch_size=config['batch_size'],
        bin_size_hours=config['bin_size_hours'],
        num_workers=config.get('num_workers', 0),
        use_memmap=config.get('use_memmap', True),
        val_ratio=(
            None if date_split is not None
            else config.get('val_ratio', 0.1)),
        split_seed=config['seed'],
        points_per_profile=config.get('profile_points_per_epoch', 8),
        full_validation_profiles=False,
        split_days=loader_split_days,
        alt_range=config.get('alt_range'),
    )
    train_loader, val_loader = get_dataloaders(
        npy_path=config['fy_path'],
        profile_path=config.get('fy_profile_path'),
        profile_index_path=config.get('fy_profile_index_path'),
        **loader_kwargs,
    )
    cosmic_train_loader, cosmic_val_loader = get_cosmic_dataloader(
        cosmic_path=config['cosmic_path'],
        profile_index_path=config.get('cosmic_profile_index_path'),
        **loader_kwargs,
    )
    config['background_loader_schedule'] = {
        'fy_batches': int(len(train_loader)),
        'cosmic_batches': int(len(cosmic_train_loader)),
        'background_epoch_batches': int(
            max(len(train_loader), len(cosmic_train_loader))),
        'pairing': 'cycle_shorter_source_max_loader_length',
    }
    config['model_domain_data_summary'] = {
        'alt_range_km': [float(value) for value in config['alt_range']],
        'development_sampling': 'stable_8_points_per_profile',
        'FY': {
            'train_points': int(len(train_loader.dataset)),
            'development_points': int(len(val_loader.dataset)),
            'all_domain_points': int(train_loader.dataset.domain_row_count),
            'all_domain_profiles': int(
                train_loader.dataset.domain_profile_count),
        },
        'COSMIC': {
            'train_points': int(len(cosmic_train_loader.dataset)),
            'development_points': int(len(cosmic_val_loader.dataset)),
            'all_domain_points': int(cosmic_train_loader.dataset.domain_row_count),
            'all_domain_profiles': int(
                cosmic_train_loader.dataset.domain_profile_count),
        },
    }
    print(f"[Background批次覆盖] {config['background_loader_schedule']}")
    subset_fraction = float(config.get('train_profile_fraction', 1.0))
    subset_manifest = config.get('profile_subset_manifest')
    profile_subset = {
        'FY': _restrict_training_profiles(
            train_loader, subset_fraction, config['seed'], 'FY',
            subset_manifest),
        'COSMIC': _restrict_training_profiles(
            cosmic_train_loader, subset_fraction, config['seed'], 'COSMIC',
            subset_manifest),
    }
    profile_subset_identity = (
        {
            'path': os.path.abspath(subset_manifest),
            'sha256': _sha256_file(subset_manifest),
        }
        if subset_manifest and os.path.isfile(subset_manifest) else None)
    print(f'[训练profile子集] {profile_subset}')
    allowed_profiles = {
        'train': {
            'FY': _loader_profile_ids(train_loader),
            'COSMIC': _loader_profile_ids(cosmic_train_loader),
        },
        'development': {
            'FY': _loader_profile_ids(val_loader),
            'COSMIC': _loader_profile_ids(cosmic_val_loader),
        },
    }
    if np.intersect1d(
            allowed_profiles['train']['FY'],
            allowed_profiles['development']['FY']).size:
        raise ValueError('FY train/development profile leakage')
    if np.intersect1d(
            allowed_profiles['train']['COSMIC'],
            allowed_profiles['development']['COSMIC']).size:
        raise ValueError('COSMIC train/development profile leakage')
    allowed_profile_summary = {
        split_name: {
            source: int(len(profile_ids))
            for source, profile_ids in sources.items()}
        for split_name, sources in allowed_profiles.items()}
    print(f'[邻域profile隔离] {allowed_profile_summary}')
    query_profile_partitions = (
        allowed_profiles if date_split is not None
        else {'train': None, 'development': None})
    covariance_strata = {
        'FY': _training_stratum_weights(train_loader),
        'COSMIC': _training_stratum_weights(cosmic_train_loader),
    }
    config['covariance_stratum_weights'] = {
        source: report['weights']
        for source, report in covariance_strata.items()}
    print(f'[协方差矩分层] {covariance_strata}')

    fy_nb_index = FYNeighborhoodIndex(config['fy_path'], config)
    cosmic_nb_index = COSMICNeighborhoodIndex(config['cosmic_path'], config)
    representativeness_path = config.get('representativeness_kernel_path')
    representativeness_floor = float(
        config.get('representativeness_floor', 0.25))
    representativeness_kernel = load_representativeness_kernel(
        representativeness_path)
    empirical_covariance_targets = (
        load_empirical_covariance_targets(representativeness_path)
        if config.get('use_empirical_covariance_loss', False) else None)
    representativeness_identity = (
        {
            'path': os.path.abspath(representativeness_path),
            'sha256': _sha256_file(representativeness_path),
            'floor': representativeness_floor,
            'stable_cells': int(representativeness_kernel.sum().item()),
            'total_cells': int(representativeness_kernel.numel()),
        }
        if representativeness_kernel is not None else None)
    config['resolved_representativeness_kernel'] = (
        representativeness_identity)
    batch_processor = SlidingWindowBatchProcessor(
        sw_manager, device=device,
        fy_nb_index=fy_nb_index, cosmic_nb_index=cosmic_nb_index,
        representativeness_kernel=representativeness_kernel,
        representativeness_floor=representativeness_floor,
        empirical_covariance_targets=empirical_covariance_targets)

    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    background_seed = config.get('background_seed_ckpt')
    if background_seed is None and not config.get('eval_only'):
        _assert_fresh_background_identity(
            model, sw_manager, iri_peak_manager, device)
    resume_state = None
    start_epoch = 0
    history = []
    best_scores = {'background': float('inf'), 'analysis': float('inf')}
    best_epochs = {'background': None, 'analysis': None}
    best_selection_keys = {'background': None, 'analysis': None}
    resume_path = config.get('resume_ckpt')
    if resume_path:
        loaded = torch.load(resume_path, map_location=device, weights_only=False)
        if loaded.get('checkpoint_type') != 'run66_training_state':
            if not config.get('eval_only'):
                raise ValueError('run66 cannot resume an older architecture checkpoint')
            model.load_state_dict(loaded, strict=True)
        else:
            if int(loaded.get('format_version', 0)) != format_version:
                raise ValueError(
                    f'checkpoint format does not match expected v{format_version}')
            checkpoint_background_semantics = loaded.get(
                'background_training_semantics')
            if checkpoint_background_semantics != config.get(
                    'background_training_semantics'):
                raise ValueError(
                    'checkpoint Background training semantics differ from config')
            if int(loaded['background_epochs']) != background_epochs:
                raise ValueError('checkpoint background_epochs differs from config')
            checkpoint_r_mode = loaded.get('r_mode')
            if (checkpoint_r_mode is not None
                    and checkpoint_r_mode != config.get('r_mode', 'global')):
                raise ValueError('checkpoint r_mode differs from config')
            checkpoint_localization = loaded.get('use_distance_localization')
            if (checkpoint_localization is not None
                    and bool(checkpoint_localization) != bool(
                        config.get('use_distance_localization', False))):
                raise ValueError(
                    'checkpoint distance localization differs from config')
            checkpoint_source_schedule = loaded.get('source_mode_schedule')
            if (checkpoint_source_schedule is not None
                    and checkpoint_source_schedule != config.get(
                        'source_mode_schedule', 'random_profile')):
                raise ValueError(
                    'checkpoint source mode schedule differs from config')
            if bool(loaded.get('analysis_exact_mode_loss', False)) != bool(
                    config.get('analysis_exact_mode_loss', False)):
                raise ValueError(
                    'checkpoint exact mode loss differs from config')
            if bool(loaded.get(
                    'use_empirical_covariance_loss', False)) != bool(
                    config.get('use_empirical_covariance_loss', False)):
                raise ValueError(
                    'checkpoint empirical covariance loss differs from config')
            if bool(loaded.get('use_direction_loss', False)) != bool(
                    config.get('use_direction_loss', False)):
                raise ValueError(
                    'checkpoint direction loss differs from config')
            if bool(loaded.get('use_observation_gram_loss', False)) != bool(
                    config.get('use_observation_gram_loss', False)):
                raise ValueError(
                    'checkpoint observation Gram loss differs from config')
            if loaded.get('representativeness_kernel') != (
                    representativeness_identity):
                raise ValueError(
                    'checkpoint representativeness kernel differs from config')
            checkpoint_architecture = loaded.get('architecture')
            if checkpoint_architecture is not None:
                checkpoint_architecture = dict(checkpoint_architecture)
                checkpoint_architecture.setdefault(
                    'density_basis_semantics', 'query_conditioned')
                checkpoint_architecture.setdefault(
                    'background_trust_gate_enabled', False)
                checkpoint_architecture.setdefault(
                    'background_trust_gate_semantics', 'disabled')
                checkpoint_architecture.setdefault(
                    'background_trust_gate_altitude_core_km', 200.0)
                checkpoint_architecture.setdefault(
                    'background_trust_gate_altitude_transition_km', 100.0)
                checkpoint_architecture.setdefault(
                    'background_trust_gate_night_cosine_offset', 0.2)
                checkpoint_architecture.setdefault(
                    'background_trust_gate_night_cosine_scale', 0.2)
                checkpoint_architecture.setdefault(
                    'background_trust_gate_dip_core', 0.25)
                checkpoint_architecture.setdefault(
                    'background_trust_gate_dip_transition', 0.25)
                checkpoint_architecture.setdefault(
                    'alt_range', [120.0, 500.0])
                checkpoint_architecture.setdefault(
                    'model_domain_semantics', 'legacy_120_500_domain_v1')
            if (checkpoint_architecture is not None
                    and checkpoint_architecture != architecture):
                raise ValueError(
                    'checkpoint architecture differs from config: '
                    f'{checkpoint_architecture} != {architecture}')
            if loaded.get('date_split_identity') != date_split_identity:
                raise ValueError('checkpoint date split differs from config')
            model.load_state_dict(loaded['model_state_dict'], strict=True)
            start_epoch = int(loaded['completed_epochs'])
            history = list(loaded.get('history', []))
            best_scores.update(loaded.get('best_scores', {}))
            best_epochs.update(loaded.get('best_epochs', {}))
            best_selection_keys.update(loaded.get(
                'best_selection_keys', {}))
            if loaded.get('resolved_covariance_weight') is not None:
                config['resolved_covariance_weight'] = float(
                    loaded['resolved_covariance_weight'])
            if loaded.get('resolved_direction_weight') is not None:
                config['resolved_direction_weight'] = float(
                    loaded['resolved_direction_weight'])
            if loaded.get('resolved_gram_weight') is not None:
                config['resolved_gram_weight'] = float(
                    loaded['resolved_gram_weight'])
            if loaded.get('gram_gradient_calibration') is not None:
                config['gram_gradient_calibration'] = loaded[
                    'gram_gradient_calibration']
            config['background_update_steps'] = list(
                loaded.get('background_update_steps', []))
            resume_state = loaded
            if loaded.get('torch_rng_state') is not None:
                torch.set_rng_state(loaded['torch_rng_state'].cpu())
            if loaded.get('numpy_rng_state') is not None:
                np.random.set_state(loaded['numpy_rng_state'])
            if (device.type == 'cuda'
                    and loaded.get('cuda_rng_state_all') is not None):
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in loaded['cuda_rng_state_all']])

    if config.get('eval_only'):
        return (
            model, [], [], train_loader, val_loader,
            sw_manager, batch_processor, iri_peak_manager,
        )

    os.makedirs(config['save_dir'], exist_ok=True)
    best_background = os.path.join(config['save_dir'], 'best_background_model.pth')
    best_analysis = os.path.join(config['save_dir'], 'best_fsia_model.pth')
    last_state = os.path.join(config['save_dir'], 'last_training_state.pth')

    seeded_background = False
    background_seed = config.get('background_seed_ckpt')
    if background_seed and resume_state is None:
        if not os.path.isfile(background_seed):
            raise FileNotFoundError(
                f'Background seed checkpoint does not exist: {background_seed}')
        _load_background_seed(model, background_seed, device)
        _atomic_torch_save(model.state_dict(), best_background)
        start_epoch = background_epochs
        seeded_background = True
        print(f'  已迁移冻结Background: {background_seed}')

    current_stage = (
        'background' if background_only or start_epoch < background_epochs
        else 'analysis')
    if background_only and (
            start_epoch > background_epochs
            or (resume_state is not None and resume_state.get('stage') != 'background')):
        raise ValueError(
            'background-only cannot resume an Analysis-stage checkpoint')
    if (not background_only and (seeded_background or _resume_needs_analysis_setup(
            resume_state, start_epoch, background_epochs))):
        if not os.path.exists(best_background):
            raise FileNotFoundError(
                'cannot enter Analysis: best_background_model.pth is missing')
        model.load_state_dict(
            torch.load(best_background, map_location=device, weights_only=True),
            strict=True)
        _calibrate_fixed_r(
            model, train_loader, cosmic_train_loader, device,
            sw_manager, iri_peak_manager, batch_processor, config)
        _reset_random_seeds(config.get('analysis_seed', 42), device)

    _set_training_stage(model, current_stage)
    if (current_stage == 'analysis'
            and (config.get('use_covariance_moment_loss', False)
                 or config.get('use_empirical_covariance_loss', False))
            and 'resolved_covariance_weight' not in config):
        config['resolved_covariance_weight'] = _resolve_covariance_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    if (current_stage == 'analysis'
            and config.get('use_direction_loss', False)
            and 'resolved_direction_weight' not in config):
        config['resolved_direction_weight'] = _resolve_direction_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    if (current_stage == 'analysis'
            and config.get('use_observation_gram_loss', False)
            and 'resolved_gram_weight' not in config):
        config['resolved_gram_weight'] = _resolve_gram_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    _record_resolved_training_config(config, covariance_strata)
    stage_epochs = background_epochs if current_stage == 'background' else analysis_epochs
    optimizer, scheduler = _make_optimizer(model, config, stage_epochs)
    scaler = (torch.cuda.amp.GradScaler()
              if config.get('use_amp', False) and device.type == 'cuda' else None)
    if resume_state is not None and resume_state.get('stage') == current_stage:
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        if scheduler is not None and resume_state.get('scheduler_state_dict'):
            scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        if scaler is not None and resume_state.get('scaler_state_dict'):
            scaler.load_state_dict(resume_state['scaler_state_dict'])

    train_losses = [row['total_loss'] for row in history]
    val_losses = [row['val_score'] for row in history]

    for epoch in range(start_epoch, total_epochs):
        stage = 'background' if epoch < background_epochs else 'analysis'
        if stage != current_stage:
            model.load_state_dict(
                torch.load(best_background, map_location=device, weights_only=True),
                strict=True)
            _calibrate_fixed_r(
                model, train_loader, cosmic_train_loader, device,
                sw_manager, iri_peak_manager, batch_processor, config)
            _reset_random_seeds(config.get('analysis_seed', 42), device)
            current_stage = stage
            _set_training_stage(model, stage)
            if ((config.get('use_covariance_moment_loss', False)
                 or config.get('use_empirical_covariance_loss', False))
                    and 'resolved_covariance_weight' not in config):
                config['resolved_covariance_weight'] = _resolve_covariance_weight(
                    model, train_loader, cosmic_train_loader, batch_processor,
                    device, config, sw_manager, iri_peak_manager,
                    query_profile_partitions['train'])
                _record_resolved_training_config(config, covariance_strata)
            if (config.get('use_direction_loss', False)
                    and 'resolved_direction_weight' not in config):
                config['resolved_direction_weight'] = _resolve_direction_weight(
                    model, train_loader, cosmic_train_loader, batch_processor,
                    device, config, sw_manager, iri_peak_manager,
                    query_profile_partitions['train'])
                _record_resolved_training_config(config, covariance_strata)
            if (config.get('use_observation_gram_loss', False)
                    and 'resolved_gram_weight' not in config):
                config['resolved_gram_weight'] = _resolve_gram_weight(
                    model, train_loader, cosmic_train_loader, batch_processor,
                    device, config, sw_manager, iri_peak_manager,
                    query_profile_partitions['train'])
                _record_resolved_training_config(config, covariance_strata)
            optimizer, scheduler = _make_optimizer(model, config, analysis_epochs)
            scaler = (torch.cuda.amp.GradScaler()
                      if config.get('use_amp', False) and device.type == 'cuda' else None)

        print(f'\nEpoch {epoch + 1}/{total_epochs} [{stage}]')
        train_loss, train_metrics, batch_diagnostics = train_one_epoch(
            model, train_loader, batch_processor, optimizer, device,
            config, epoch, stage, scaler, sw_manager, iri_peak_manager,
            cosmic_train_loader, query_profile_partitions['train'])
        if stage == 'background':
            config.setdefault('background_update_steps', []).append(
                int(train_metrics['processed_batches']))
        if batch_diagnostics:
            diagnostics_path = os.path.join(
                config['save_dir'], 'batch_diagnostics.jsonl')
            with open(diagnostics_path, 'a', encoding='utf-8') as stream:
                for row in batch_diagnostics:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        val_score, val_metrics = validate(
            model, val_loader, batch_processor, device, config,
            sw_manager, iri_peak_manager, cosmic_val_loader, stage,
            query_profile_partitions['development'])
        if scheduler is not None:
            scheduler.step()

        if (train_metrics['gradient_audit_batches']
                and train_metrics['gradient_ratio'] > 0.30):
            print(
                f"  警告: 辅助/观测梯度比={train_metrics['gradient_ratio']:.3f}>0.30；"
                '全量训练前应下调对应辅助权重')
        if (stage == 'analysis'
                and config.get('use_observation_gram_loss', False)
                and config.get('max_train_batches') is not None
                and train_metrics['gradient_audit_batches']):
            gram_ratio = train_metrics.get('gradient_ratio_gram', 0.0)
            coverages = [
                train_metrics[f'{target}_gram_{mode}_coverage']
                for target in ('fy', 'cosmic')
                for mode in ('m10', 'm01', 'm11')
            ]
            if not 0.016 <= gram_ratio <= 0.024:
                raise RuntimeError(
                    f'Gram preflight gradient ratio {gram_ratio:.6f} is outside '
                    '[0.016, 0.024]')
            if min(coverages) < 0.50:
                raise RuntimeError(
                    f'Gram preflight eligible coverage is below 0.50: {coverages}')
            if train_metrics['gradient_ratio'] > 0.30:
                raise RuntimeError(
                    'Gram preflight auxiliary/observation gradient ratio exceeds 0.30')
            if len(batch_diagnostics) >= 40:
                first_gram = np.median([
                    row['gram_raw'] for row in batch_diagnostics[:20]])
                last_gram = np.median([
                    row['gram_raw'] for row in batch_diagnostics[-20:]])
                if not last_gram < first_gram:
                    raise RuntimeError(
                        'Gram preflight loss did not decrease over the audited batches')

        history_row = {
            'epoch': epoch + 1,
            'stage': stage,
            'total_loss': train_loss,
            'val_score': val_score,
            **train_metrics,
            **val_metrics,
        }
        history.append(history_row)
        train_losses.append(train_loss)
        val_losses.append(val_score)
        print(
            f"  train={train_loss:.6f} val={val_score:.6f} "
            f"FY={val_metrics['fy_profile_rmse']:.6f} "
            f"COSMIC={val_metrics['cosmic_profile_rmse']:.6f}")

        selection_key = _development_selection_key(val_metrics, config)
        previous_key = best_selection_keys[stage]
        if (selection_key is not None
                and (previous_key is None
                     or tuple(selection_key) > tuple(previous_key))):
            best_scores[stage] = val_score
            best_epochs[stage] = epoch + 1
            best_selection_keys[stage] = selection_key
            _atomic_torch_save(
                model.state_dict(),
                best_background if stage == 'background' else best_analysis)
        if stage == 'analysis':
            _atomic_torch_save(
                model.state_dict(),
                os.path.join(
                    config['save_dir'], f'epoch_{epoch + 1:02d}_model.pth'))

        _atomic_torch_save({
            'checkpoint_type': 'run66_training_state',
            'format_version': format_version,
            'model_domain_semantics': config.get(
                'model_domain_semantics', 'legacy_120_500_domain_v1'),
            'alt_range': [float(value) for value in config.get(
                'alt_range', (120.0, 500.0))],
            'completed_epochs': epoch + 1,
            'background_epochs': background_epochs,
            'analysis_epochs': analysis_epochs,
            'background_training_semantics': config.get(
                'background_training_semantics'),
            'background_trust_gate_enabled': bool(config.get(
                'background_trust_gate_enabled', False)),
            'background_trust_gate_semantics': config.get(
                'background_trust_gate_semantics', 'disabled'),
            'background_trust_gate_parameters': {
                key: float(config.get(key, default))
                for key, default in (
                    ('background_trust_gate_altitude_core_km', 200.0),
                    ('background_trust_gate_altitude_transition_km', 100.0),
                    ('background_trust_gate_night_cosine_offset', 0.2),
                    ('background_trust_gate_night_cosine_scale', 0.2),
                    ('background_trust_gate_dip_core', 0.25),
                    ('background_trust_gate_dip_transition', 0.25),
                )
            },
            'background_only': background_only,
            'background_loader_schedule': config.get(
                'background_loader_schedule'),
            'background_update_steps': config.get(
                'background_update_steps', []),
            'r_mode': config.get('r_mode', 'global'),
            'use_distance_localization': bool(
                config.get('use_distance_localization', False)),
            'assimilation_semantics': config.get(
                'assimilation_semantics', 'legacy_local_etkf'),
            'use_physical_localization': bool(
                config.get('use_physical_localization', False)),
            'neighbor_directory_semantics': config.get(
                'neighbor_directory_semantics', 'profile_center_legacy'),
            'observation_chunk_size': int(config.get(
                'observation_chunk_size', 4096)),
            'source_mode_schedule': config.get(
                'source_mode_schedule', 'random_profile'),
            'analysis_exact_mode_loss': bool(
                config.get('analysis_exact_mode_loss', False)),
            'use_empirical_covariance_loss': bool(
                config.get('use_empirical_covariance_loss', False)),
            'use_direction_loss': bool(
                config.get('use_direction_loss', False)),
            'use_observation_gram_loss': bool(
                config.get('use_observation_gram_loss', False)),
            'representativeness_kernel': representativeness_identity,
            'architecture': architecture,
            'profile_subset': profile_subset,
            'profile_subset_identity': profile_subset_identity,
            'date_split_identity': date_split_identity,
            'allowed_profile_summary': allowed_profile_summary,
            'model_domain_data_summary': config.get(
                'model_domain_data_summary'),
            'resolved_covariance_weight': config.get(
                'resolved_covariance_weight'),
            'resolved_direction_weight': config.get(
                'resolved_direction_weight'),
            'resolved_gram_weight': config.get('resolved_gram_weight'),
            'gram_gradient_calibration': config.get(
                'gram_gradient_calibration'),
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_scores': best_scores,
            'best_epochs': best_epochs,
            'best_selection_keys': best_selection_keys,
            'history': history,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'cuda_rng_state_all': (
                torch.cuda.get_rng_state_all() if device.type == 'cuda' else None),
        }, last_state)
        if not background_only:
            plot_training_curves(
                history,
                save_path=os.path.join(
                    config['save_dir'], 'fsia_training_curves.png'))

    with open(os.path.join(config['save_dir'], 'fsia_training_history.json'),
              'w', encoding='utf-8') as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)
    _record_resolved_training_config(config, covariance_strata)
    background_validation = None
    background_gate_passed = None
    if background_only:
        model.load_state_dict(
            torch.load(best_background, map_location=device, weights_only=True),
            strict=True)
        _, background_validation = validate(
            model, val_loader, batch_processor, device, config,
            sw_manager, iri_peak_manager, cosmic_val_loader, 'background',
            query_profile_partitions['development'])
        comparisons = [
            background_validation['fy_rmse_change_M00_minus_raw'],
            background_validation['cosmic_rmse_change_M00_minus_raw'],
        ]
        background_gate_passed = all(
            value is not None and np.isfinite(value) and value <= 0.0
            for value in comparisons)
        print(
            f'[Background development gate] FY ΔRMSE={comparisons[0]:.6g}, '
            f'COSMIC ΔRMSE={comparisons[1]:.6g}, '
            f'passed={background_gate_passed}')
    summary = {
        'completed_stage': 'background' if background_only else 'analysis',
        'checkpoint_format_version': format_version,
        'model_domain_semantics': config.get(
            'model_domain_semantics', 'legacy_120_500_domain_v1'),
        'alt_range': [float(value) for value in config.get(
            'alt_range', (120.0, 500.0))],
        'model_domain_data_summary': config.get(
            'model_domain_data_summary'),
        'best_scores': {
            stage: (score if np.isfinite(score) else None)
            for stage, score in best_scores.items()},
        'best_epochs': best_epochs,
        'best_selection_keys': best_selection_keys,
        'checkpoint_selection_semantics': config.get(
            'checkpoint_selection_semantics', 'mean_profile_rmse_v1'),
        'checkpoint': best_background if background_only else best_analysis,
        'checkpoint_sha256': _sha256_file(
            best_background if background_only else best_analysis),
        'checkpoint_stage': 'background' if background_only else 'analysis',
        'background_training_semantics': config.get(
            'background_training_semantics'),
        'background_trust_gate_enabled': bool(config.get(
            'background_trust_gate_enabled', False)),
        'background_trust_gate_semantics': config.get(
            'background_trust_gate_semantics', 'disabled'),
        'background_trust_gate_parameters': {
            key: float(config.get(key, default))
            for key, default in (
                ('background_trust_gate_altitude_core_km', 200.0),
                ('background_trust_gate_altitude_transition_km', 100.0),
                ('background_trust_gate_night_cosine_offset', 0.2),
                ('background_trust_gate_night_cosine_scale', 0.2),
                ('background_trust_gate_dip_core', 0.25),
                ('background_trust_gate_dip_transition', 0.25),
            )
        },
        'background_development': background_validation,
        'background_development_gate_passed': background_gate_passed,
        'background_loader_schedule': config.get('background_loader_schedule'),
        'background_update_steps': config.get('background_update_steps', []),
        'r_mode': config.get('r_mode', 'global'),
        'use_distance_localization': bool(
            config.get('use_distance_localization', False)),
        'assimilation_semantics': config.get(
            'assimilation_semantics', 'legacy_local_etkf'),
        'use_physical_localization': bool(
            config.get('use_physical_localization', False)),
        'neighbor_directory_semantics': config.get(
            'neighbor_directory_semantics', 'profile_center_legacy'),
        'observation_chunk_size': int(config.get(
            'observation_chunk_size', 4096)),
        'source_mode_schedule': config.get(
            'source_mode_schedule', 'random_profile'),
        'analysis_exact_mode_loss': bool(
            config.get('analysis_exact_mode_loss', False)),
        'use_empirical_covariance_loss': bool(
            config.get('use_empirical_covariance_loss', False)),
        'use_direction_loss': bool(
            config.get('use_direction_loss', False)),
        'use_observation_gram_loss': bool(
            config.get('use_observation_gram_loss', False)),
        'representativeness_kernel': representativeness_identity,
        'r_fy': (model.kalman_layer.r_fy.item()
                 if not background_only else None),
        'r_cosmic': (model.kalman_layer.r_cosmic.item()
                     if not background_only else None),
        'r_calibration': (
            os.path.join(config['save_dir'], 'r_calibration.json')
            if not background_only else None),
        'architecture': architecture,
        'profile_subset': profile_subset,
        'profile_subset_manifest': profile_subset_identity,
        'date_split': date_split_identity,
        'allowed_profile_summary': allowed_profile_summary,
        'covariance_moment': {
            'enabled': bool(config.get('use_covariance_moment_loss', False)),
            'empirical_cells_enabled': bool(
                config.get('use_empirical_covariance_loss', False)),
            'gradient_target': float(
                config.get('covariance_gradient_target', 0.20)),
            'resolved_weight': config.get('resolved_covariance_weight'),
            'strata': covariance_strata,
        },
        'direction_loss': {
            'enabled': bool(config.get('use_direction_loss', False)),
            'gradient_target': float(
                config.get('direction_gradient_target', 0.20)),
            'resolved_weight': config.get('resolved_direction_weight'),
        },
        'observation_gram_loss': {
            'enabled': bool(config.get('use_observation_gram_loss', False)),
            'gradient_target': float(
                config.get('gram_gradient_target', 0.02)),
            'calibration_batches': int(
                config.get('gram_calibration_batches', 20)),
            'resolved_weight': config.get('resolved_gram_weight'),
            'gradient_calibration': config.get('gram_gradient_calibration'),
        },
    }
    with open(os.path.join(config['save_dir'], 'training_summary.json'),
              'w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    if background_only and not background_gate_passed:
        raise RuntimeError(
            'Background development gate failed: M00 is worse than Raw IRI '
            'for at least one source')

    selected_checkpoint = best_background if background_only else best_analysis
    if not os.path.isfile(selected_checkpoint):
        raise FileNotFoundError(
            f'completed {summary["completed_stage"]} stage lacks checkpoint')
    model.load_state_dict(
        torch.load(selected_checkpoint, map_location=device, weights_only=True),
        strict=True)
    return (
        model, train_losses, val_losses, train_loader, val_loader,
        sw_manager, batch_processor, iri_peak_manager,
    )
