"""M2-R R2 train-only gappy-profile pretraining and frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_etkf_observation_subspace import ROOT, _load, _restrict_profiles
from audit_m2r_representation import _physical_shadow
from evaluate_satellite_development import _cell_ids, _partition_loader
from inr_modules.data_managers.FY_dataloader import COSMICDataset, FY3D_Dataset


def _split(source, profile_id, coords):
    flags = []
    for row in coords.detach().cpu().numpy():
        token = f'{source}:{profile_id}:' + ':'.join(f'{value:.5f}' for value in row)
        flags.append(hashlib.sha256(token.encode()).digest()[0] & 1)
    fit = torch.tensor(flags, device=coords.device, dtype=torch.bool)
    if fit.sum() < 2 or (~fit).sum() < 2:
        order = torch.argsort(coords[:, 2], stable=True)
        fit = torch.zeros(len(coords), device=coords.device, dtype=torch.bool)
        fit[order[::2]] = True
    return fit


def _ridge(modes, residual, fit, ridge):
    matrix = modes[fit]
    system = matrix.T @ matrix + ridge * torch.eye(
        matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.solve(system, matrix.T @ residual[fit])


def _center(coords):
    lon = torch.deg2rad(coords[:, 1])
    return torch.stack([
        coords[:, 0].mean(),
        torch.rad2deg(torch.atan2(torch.sin(lon).mean(), torch.cos(lon).mean())),
        coords[:, 2].median(), coords[:, 3].median(),
    ]).unsqueeze(0)


def _profile_modes(model, coords, sw_manager, peak_manager):
    with torch.no_grad():
        endpoint_peak = peak_manager.get_iri_peak(coords) if peak_manager else None
        endpoint = model.encode_background(
            coords, sw_manager.get_drivers_sequence(coords[:, 3]),
            iri_peak=endpoint_peak)
        center_coords = _center(coords)
        center_peak = peak_manager.get_iri_peak(center_coords) if peak_manager else None
        center = model.encode_background(
            center_coords, sw_manager.get_drivers_sequence(center_coords[:, 3]),
            iri_peak=center_peak)
        reference_coords = model._physical_reference_coords(center_coords)
        flat_reference = reference_coords.reshape(-1, 4)
        reference_peak = (peak_manager.get_iri_peak(flat_reference)
                          if peak_manager else None)
        reference = model.encode_background(
            flat_reference,
            sw_manager.get_drivers_sequence(flat_reference[:, 3]),
            iri_peak=reference_peak)
    hmf2 = center['iri_peak'][:, 0]
    _, chol, _ = model._physical_mode_transform(
        center_coords, hmf2, center['z_background'], center['h_sw'],
        center['ne_bkg'].flatten(),
        reference['z_background'].reshape(1, -1, model.background_state_dim),
        reference['h_sw'].reshape(1, -1, model.sw_out_dim),
        reference['ne_bkg'].reshape(1, -1))
    learned = model._apply_mode_transform(model._physical_mode_raw(
        center_coords, hmf2, coords.unsqueeze(0),
        center['z_background'], center['h_sw'],
        endpoint['z_background'].unsqueeze(0), endpoint['h_sw'].unsqueeze(0),
        endpoint['ne_bkg'].T), chol).squeeze(0)
    _, fixed_chol, _ = model._physical_mode_transform(center_coords, hmf2)
    fixed = model._apply_mode_transform(
        model._physical_mode_raw(center_coords, hmf2, coords.unsqueeze(0)),
        fixed_chol).squeeze(0)
    return learned, fixed, endpoint, center_coords


def _m2o_modes(model, coords, endpoint, center_coords, sw_manager, peak_manager):
    with torch.no_grad():
        peak = peak_manager.get_iri_peak(center_coords) if peak_manager else None
        center = model.encode_background(
            center_coords, sw_manager.get_drivers_sequence(center_coords[:, 3]),
            iri_peak=peak)
        return model._density_basis(
            center_coords, coords.unsqueeze(0), endpoint['ne_bkg'].T,
            center['z_background'], center['h_sw'],
            endpoint['z_background'].unsqueeze(0),
            endpoint['h_sw'].unsqueeze(0)).squeeze(0)


def _rows(loaders):
    for source, loader in loaders.items():
        for data, _, profile_ids in loader:
            for profile_id in torch.unique(profile_ids, sorted=True):
                selected = profile_ids == profile_id
                yield source, int(profile_id), data[selected, :5]


def _bootstrap_upper(differences, seed=42, draws=2000):
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return float(np.quantile(means, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--background-checkpoint', type=Path,
                        default=Path('best_background_model.pth'))
    parser.add_argument('--m2o-checkpoint', type=Path,
                        default=Path('epoch_09_model.pth'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-profiles', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(42)
    run_dir = args.run_dir.resolve()
    background_checkpoint = run_dir / args.background_checkpoint
    m2o_checkpoint = run_dir / args.m2o_checkpoint
    baseline, sw_manager, peak_manager, config = _load(run_dir, m2o_checkpoint)
    model = _physical_shadow(config, background_checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mode_residual.parameters():
        parameter.requires_grad_(True)

    manifest = Path(config['date_split_manifest'])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    split_days = json.loads(manifest.read_text(encoding='utf-8'))['partitions']
    loaders = {
        'FY': _partition_loader(FY3D_Dataset, config, split_days, 'train'),
        'COSMIC': _partition_loader(COSMICDataset, config, split_days, 'train'),
    }
    selected = {
        source: _restrict_profiles(loader, args.max_profiles, 242 + index)
        for index, (source, loader) in enumerate(loaders.items())}
    optimizer = torch.optim.Adam(model.mode_residual.parameters(), lr=args.lr)
    ridge = float(model.kalman_layer.r_fy)
    history = []
    model.train()
    for epoch in range(args.epochs):
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, (source, profile_id, data) in enumerate(_rows(loaders), 1):
            coords, truth = data[:, :4], data[:, 4]
            if len(coords) < 4:
                continue
            learned, _, endpoint, _ = _profile_modes(
                model, coords, sw_manager, peak_manager)
            residual = truth - endpoint['ne_bkg'].flatten().detach()
            fit = _split(source, profile_id, coords)
            coefficients = _ridge(learned, residual, fit, ridge)
            loss = (learned[~fit] @ coefficients - residual[~fit]).square().mean()
            loss.backward()
            losses.append(float(loss.detach()))
            if step % 16 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(float(np.mean(losses)))

    model.eval()
    metrics = {source: defaultdict(lambda: defaultdict(list))
               for source in loaders}
    with torch.no_grad():
        for source, profile_id, data in _rows(loaders):
            coords, truth = data[:, :4], data[:, 4]
            if len(coords) < 4:
                continue
            learned, fixed, endpoint, center_coords = _profile_modes(
                model, coords, sw_manager, peak_manager)
            original = _m2o_modes(
                baseline, coords, endpoint, center_coords, sw_manager, peak_manager)
            residual = truth - endpoint['ne_bkg'].flatten()
            fit = _split(source, profile_id, coords)
            predictions = {
                'M2-R': learned[~fit] @ _ridge(learned, residual, fit, ridge),
                'M0': fixed[~fit] @ _ridge(fixed, residual, fit, ridge),
                'M2-O': original[~fit] @ _ridge(original, residual, fit, ridge),
                'rank1': residual[fit].mean().expand((~fit).sum()),
            }
            altitude, day, _ = _cell_ids(coords[~fit].numpy())
            cells = [f'{"low" if a == 0 else "high"}_{"night" if d == 0 else "day"}'
                     for a, d in zip(altitude, day)]
            for name, prediction in predictions.items():
                error = (prediction - residual[~fit]).square()
                metrics[source]['overall'][name].append(float(error.mean().sqrt()))
                for cell in sorted(set(cells)):
                    selected_cell = torch.tensor(
                        [value == cell for value in cells], dtype=torch.bool)
                    metrics[source][cell][name].append(
                        float(error[selected_cell].mean().sqrt()))

    summary = {}
    overall_rank1_differences = []
    gates = []
    for source, strata in metrics.items():
        summary[source] = {}
        for cell, methods in sorted(strata.items()):
            row = {name: {'profiles': len(values), 'profile_rmse_mean': float(np.mean(values))}
                   for name, values in methods.items()}
            row['M2-R_to_M2-O'] = row['M2-R']['profile_rmse_mean'] / row['M2-O']['profile_rmse_mean']
            summary[source][cell] = row
            if cell == 'overall' or row['M2-R']['profiles'] >= 10:
                gates.append(row['M2-R_to_M2-O'] <= 1.01)
        overall_rank1_differences.extend(
            np.asarray(strata['overall']['M2-R']) -
            np.asarray(strata['overall']['rank1']))
    rank1_upper = _bootstrap_upper(overall_rank1_differences)
    passed = all(gates) and rank1_upper < 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / 'r2_mode_model.pth')
    report = {
        'schema_version': 1, 'partition': 'train',
        'locked_test_accessed': False, 'isr_accessed': False,
        'semantics': {
            'analysis_state': 'query_local_increment_coefficients',
            'context': 'endpoint_conditioning_only'},
        'selected_profiles': {key: len(value) for key, value in selected.items()},
        'training_loss': history, 'metrics': summary,
        'rank1_paired_difference_bootstrap_95_upper': rank1_upper,
        'gates': {'M2O_all_major_strata_within_1pct': all(gates),
                  'significantly_better_than_rank1': rank1_upper < 0,
                  'R2_passed': passed},
    }
    path = args.output_dir / 'r2_gappy_profile_report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
