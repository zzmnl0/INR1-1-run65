"""Capture fixed CPU model behavior; never write training artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(record):
    path = Path(record['path'])
    actual = sha256(path)
    expected = record.get('sha256') or record.get('expected_sha256')
    if not expected or actual != expected:
        raise ValueError(f'Source identity mismatch: {path}')
    return path, actual


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise RuntimeError('Set PYTHONDONTWRITEBYTECODE=1 before execution')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    from inr_modules.mdia.checkpoint_io import load_fsia_analysis_checkpoint
    from inr_modules.mdia.fsia_model import FSIA_INR_Model, solve_density_modes
    from inr_modules.mdia.train_fsia import (
        exact_mode_profile_losses, observation_gram_whitening_loss)

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    identity_path = Path(__file__).with_name('model_identity.json')
    identity = read_json(identity_path)
    main_run = identity['main_run']
    main_path = Path(main_run['path']) / 'best_fsia_model.pth'
    main_weight = next(item for item in identity['weights']
                       if Path(item['path']) == main_path)
    iri_record = next(item for item in identity['weights']
                      if item.get('role') == 'iri_proxy')
    _, iri_hash = verify(iri_record)
    runs = [('main_A1S_N8D64', main_run, main_weight)]
    runs.extend((name, record, record['weight'])
                for name, record in identity['sensitivity_runs'].items())
    report = {
        'status': 'RUNNING', 'identity_sha256': sha256(identity_path),
        'script_sha256': sha256(__file__), 'seed': 42,
        'torch': torch.__version__, 'numpy': np.__version__,
        'device': 'cpu', 'threads': 2, 'runs': {},
        'scope': 'Strict weights, synthetic fixed queries, empty fallback, four modes, Gram and gradients',
        'limitations': [
            'Synthetic observations are not a scientific reproduction or full training.',
            'Low-altitude queries cover forward behavior, not the real-data low-altitude training loss.',
            'S6 architectures use manifest-driven strict loading because the production loader requires N8D64.',
            'No optimizer step, resume restoration, or dataset indexing is tested by this script.']}

    def save_report():
        (output / 'model_baseline.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    def tensor_record(tensor):
        value = tensor.detach().cpu().contiguous()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}

    try:
        for name, record, weight_record in runs:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            weight_path, weight_hash = verify(weight_record)
            manifest_path, manifest_hash = verify(record['manifest'])
            summary_path, summary_hash = verify(record['summary'])
            manifest, summary = read_json(manifest_path), read_json(summary_path)
            config = manifest['config']
            if summary.get('checkpoint_sha256') != weight_hash:
                raise ValueError(f'{name}: summary/weight mismatch')
            if summary.get('completed_stage') != 'analysis':
                raise ValueError(f'{name}: incomplete analysis')
            if sha256(config['iri_proxy_path']) != iri_hash:
                raise ValueError(f'{name}: IRI identity mismatch')
            if name == 'main_A1S_N8D64':
                model, config, _, _ = load_fsia_analysis_checkpoint(
                    weight_path, device='cpu', expected_sha256=weight_hash,
                    verify_hashes=True)
                loader = 'production_strict_with_hash_verification'
            else:
                iri = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
                iri.load_state_dict(torch.load(config['iri_proxy_path'],
                                               map_location='cpu', weights_only=True), strict=True)
                iri.eval()
                model = FSIA_INR_Model(iri_proxy=iri, config=config)
                model.load_state_dict(torch.load(weight_path, map_location='cpu',
                                                 weights_only=True), strict=True)
                model.eval()
                loader = 'manifest_architecture_strict_with_source_hash_verification'
            states = {key: tensor_record(value) for key, value in model.state_dict().items()}
            if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
                raise AssertionError(f'{name}: nonfinite weights')
            coords = torch.tensor([[-12., -76.8, 120., 0.], [0., 120., 250., 240.],
                                   [35., 30., 400., 480.], [65., 150., 199.9, 720.]])
            sw = torch.zeros(4, 36, 2)
            peaks = torch.tensor([[300., 11.5]] * 4)
            with torch.no_grad():
                empty, _, _, empty_delta, endpoint = model(coords, sw, iri_peak=peaks)
                if not torch.equal(empty, endpoint['ne_bkg']):
                    raise AssertionError(f'{name}: fallback is not exact')
                if torch.count_nonzero(empty_delta).item():
                    raise AssertionError(f'{name}: empty increment not zero')
                obs_coords = coords.repeat_interleave(8, dim=0)
                obs_coords[:, 2] = torch.linspace(205., 495., 8).repeat(4)
                _, _, _, _, obs_endpoint = model(
                    obs_coords, sw.repeat_interleave(8, dim=0),
                    iri_peak=peaks.repeat_interleave(8, dim=0))
            count = len(obs_coords)
            background = obs_endpoint['ne_bkg'].flatten().detach()
            # Same ragged fields as test_v14_fixed_query_regression._payload, with Gram support.
            payload = {
                'coords': obs_coords, 'background': background,
                'value': background + torch.linspace(-0.15, 0.2, 8).repeat(4),
                'valid_mask': torch.ones(count, dtype=torch.bool),
                'rho_squared': torch.zeros(count), 'localization_weight': torch.ones(count),
                'query_index': torch.arange(4).repeat_interleave(8),
                'row_ptr': torch.arange(0, count + 1, 8),
                'basis_z_background': obs_endpoint['basis_z_background'].detach(),
                'basis_h_sw': obs_endpoint['basis_h_sw'].detach()}
            model.zero_grad(set_to_none=True)
            joint, _, _, _, extras = model(coords, sw, iri_peak=peaks,
                                           observations_fy=payload, observations_cosmic=payload)
            increments = solve_density_modes(extras)
            modes = {'M00': empty.detach()}
            modes.update({key: extras['ne_bkg'] + value.unsqueeze(-1)
                          for key, value in increments.items()})
            if not torch.allclose(joint, modes['M11'], atol=1e-7, rtol=0):
                raise AssertionError(f'{name}: joint solve mismatch')
            ids = torch.arange(4)
            target = empty.detach() + 0.08
            data_loss, losses, active = exact_mode_profile_losses(extras, target, ids, 0.2)
            gram, gram_modes, coverage = observation_gram_whitening_loss(
                extras, ids, model.kalman_layer, return_details=True)
            if not all(value.item() > 0 for value in coverage.values()):
                raise AssertionError(f'{name}: Gram fixture lacks eligible queries')
            # Unit auxiliary weight is a fixed diagnostic, not a replacement training configuration.
            objective = data_loss + gram
            objective.backward()
            gradients = {key: tensor_record(parameter.grad) for key, parameter in model.named_parameters()
                         if parameter.grad is not None}
            grad_norms = {key: float(parameter.grad.norm()) for key, parameter in model.named_parameters()
                          if parameter.grad is not None}
            tensors = dict(coords=coords, sw=sw, iri_peak=peaks, target=target,
                           obs_coords=obs_coords, obs_value=payload['value'], **modes)
            if not all(torch.isfinite(value).all().item() for value in tensors.values()):
                raise AssertionError(f'{name}: nonfinite prediction')
            if not torch.isfinite(objective).item() or not all(np.isfinite(x) for x in grad_norms.values()):
                raise AssertionError(f'{name}: nonfinite loss/gradient')
            if not gradients or not any(x > 0 for x in grad_norms.values()):
                raise AssertionError(f'{name}: no active gradients')
            npz = output / f'{name}.npz'
            np.savez_compressed(npz, **{key: value.detach().cpu().numpy() for key, value in tensors.items()})
            report['runs'][name] = {
                'status': 'PASS', 'checkpoint': str(weight_path), 'weight_sha256': weight_hash,
                'manifest_sha256': manifest_hash, 'summary_sha256': summary_hash,
                'iri_sha256': iri_hash, 'loader': loader,
                'basis_dim': config['basis_dim'], 'enkf_n_members': config['enkf_n_members'],
                'state': states, 'empty_fallback_exact': True, 'empty_increment_zero': True,
                'data_loss': float(data_loss.detach()), 'gram_loss': float(gram.detach()),
                'mode_losses': {k: float(v.detach()) for k, v in losses.items()},
                'gram_mode_losses': {k: float(v.detach()) for k, v in gram_modes.items()},
                'gram_coverage': {k: float(v.detach()) for k, v in coverage.items()},
                'active_modes': {k: v.detach().cpu().tolist() for k, v in active.items()},
                'gradients': gradients, 'gradient_norms': grad_norms,
                'npz': npz.name, 'npz_sha256': sha256(npz)}
            save_report()
            del model
        report['status'] = 'PASS'
        save_report()
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = f'{type(error).__name__}: {error}'
        save_report()
        raise


if __name__ == '__main__':
    main()
