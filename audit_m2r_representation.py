"""Train-only paired R0 audit for M2-O versus the fixed M2-R representation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_etkf_observation_subspace import (
    ROOT, _geometry, _load, _restrict_profiles,
)
from evaluate_satellite_development import _cell_ids, _cell_name, _partition_loader
from inr_modules.data_managers.FY_dataloader import (
    COSMICDataset, COSMICNeighborhoodIndex, FY3D_Dataset, FYNeighborhoodIndex,
)
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background, query_observation_payload,
)


def _physical_shadow(config, checkpoint):
    physical_config = dict(config)
    physical_config.update({
        'density_basis_semantics': 'endpoint_context_symmetric',
        'analysis_state_semantics': 'query_local_increment_coefficients',
        'context_semantics': 'endpoint_conditioning_only',
        'mode_basis_semantics': 'reference_whitened_physical_modes',
        'enkf_n_members': 8,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
    })
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
    model = FSIA_INR_Model(proxy, physical_config)
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    target = model.state_dict()
    shared = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
        and not key.startswith('kalman_layer.')
    }
    missing, unexpected = model.load_state_dict(shared, strict=False)
    if unexpected or any(
            not key.startswith(('kalman_layer.', 'mode_residual.'))
            for key in missing):
        raise ValueError(
            f'unexpected M2-R shadow state mismatch: {missing}, {unexpected}')
    model.kalman_layer.set_observation_variances(
        float(source['kalman_layer.r_fy']),
        float(source['kalman_layer.r_cosmic']))
    model.eval()
    return model


def _profile_ci(rows, seed=42, draws=2000):
    values = np.asarray([
        np.mean(samples) for _, samples in sorted(rows.items()) if samples
    ], dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return {
        'profiles': int(len(values)),
        'mean_difference': float(values.mean()),
        'bootstrap_95_ci': np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def _posterior_ratio(extras, source):
    innovation = extras[f'innov_{source}']
    precision = extras[f'precision_{source}']
    anomalies = extras[f'obs_anomalies_{source}']
    weights = extras['weights_FY'] + extras['weights_COSMIC']
    reduction = torch.einsum('bmn,bn->bm', anomalies, weights)
    valid = precision > 0
    return (
        (innovation - reduction).abs()[valid].sum().item(),
        innovation.abs()[valid].sum().item(),
    )


def _first_energy(matrix, precision):
    valid = precision > 0
    singular = torch.linalg.svdvals(
        matrix[valid] * precision[valid].sqrt().unsqueeze(-1))
    energy = singular.square()
    return float(energy[0] / energy.sum().clamp_min(1e-12))


def _audit_target(source, loader, baseline, physical, sw_manager,
                  iri_peak_manager, indices, allowed):
    paired = {name: defaultdict(list) for name in ('FY', 'COSMIC', 'joint')}
    strata = {
        name: defaultdict(list) for name in ('FY', 'COSMIC', 'joint')}
    first_energy = {name: {'M2-O': [], 'M2-R': []}
                    for name in ('FY', 'COSMIC', 'joint')}
    posterior = {
        name: {'numerator': 0.0, 'denominator': 0.0}
        for name in ('FY', 'COSMIC')}
    reference = defaultdict(list)
    background_error = 0.0
    with torch.no_grad():
        for data, _, profile_ids in loader:
            coords = data[:, :4]
            sw = sw_manager.get_drivers_sequence(coords[:, 3])
            peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
            raw = {}
            for observation_source in ('FY', 'COSMIC'):
                raw[observation_source] = query_observation_payload(
                    indices[observation_source], coords, torch.device('cpu'),
                    exclude_profile_ids=(profile_ids.numpy()
                                         if observation_source == source else None),
                    allowed_profile_ids=allowed[observation_source])
            baseline_payload = {
                name: attach_observation_background(
                    payload, baseline, sw_manager, iri_peak_manager)
                for name, payload in raw.items()}
            physical_payload = {
                name: attach_observation_background(
                    payload, physical, sw_manager, iri_peak_manager)
                for name, payload in raw.items()}
            baseline_extras = baseline(
                coords, sw, iri_peak=peak,
                observations_fy=baseline_payload['FY'],
                observations_cosmic=baseline_payload['COSMIC'])[4]
            # R0 isolates representation: one point per shadow unit. R1 separately
            # verifies the profile-level shared-state implementation.
            physical_extras = physical(
                coords, sw, iri_peak=peak,
                observations_fy=physical_payload['FY'],
                observations_cosmic=physical_payload['COSMIC'])[4]
            background_error = max(background_error, float(torch.max(torch.abs(
                baseline_extras['ne_bkg'] - physical_extras['ne_bkg']))))
            altitude, day, latitude = _cell_ids(coords.numpy())
            cells = [_cell_name(altitude[i], day[i], latitude[i])
                     for i in range(len(coords))]
            for name in ('FY', 'COSMIC', 'joint'):
                if name == 'joint':
                    base_matrix = torch.cat([
                        baseline_extras['obs_anomalies_FY'],
                        baseline_extras['obs_anomalies_COSMIC']], dim=1)
                    base_precision = torch.cat([
                        baseline_extras['precision_FY'],
                        baseline_extras['precision_COSMIC']], dim=1)
                    phys_matrix = torch.cat([
                        physical_extras['obs_anomalies_FY'],
                        physical_extras['obs_anomalies_COSMIC']], dim=1)
                    phys_precision = torch.cat([
                        physical_extras['precision_FY'],
                        physical_extras['precision_COSMIC']], dim=1)
                else:
                    base_matrix = baseline_extras[f'obs_anomalies_{name}']
                    base_precision = baseline_extras[f'precision_{name}']
                    phys_matrix = physical_extras[f'obs_anomalies_{name}']
                    phys_precision = physical_extras[f'precision_{name}']
                for index in range(len(coords)):
                    base, _ = _geometry(base_matrix[index], base_precision[index])
                    phys, _ = _geometry(phys_matrix[index], phys_precision[index])
                    if base is None or phys is None:
                        continue
                    difference = phys['effective_rank'] - base['effective_rank']
                    paired[name][int(profile_ids[index])].append(difference)
                    strata[name][cells[index]].append((
                        base['effective_rank'], phys['effective_rank']))
                    for label, geometry in (('M2-O', base), ('M2-R', phys)):
                        first_energy[name][label].append(
                            _first_energy(
                                base_matrix[index] if label == 'M2-O'
                                else phys_matrix[index],
                                base_precision[index] if label == 'M2-O'
                                else phys_precision[index]))
            for observation_source in ('FY', 'COSMIC'):
                numerator, denominator = _posterior_ratio(
                    physical_extras, observation_source)
                posterior[observation_source]['numerator'] += numerator
                posterior[observation_source]['denominator'] += denominator
            for key in (
                    'mode_reference_gram_error',
                    'raw_mode_gram_min_eigenvalue',
                    'raw_mode_gram_condition',
                    'mode_reference_effective_rank',
                    'reference_anomalies_effective_rank',
                    'reference_anomalies_first_energy_fraction'):
                reference[key].extend(
                    physical_extras[key].cpu().numpy().reshape(-1).tolist())
    return {
        'paired_effective_rank': {
            name: _profile_ci(values, 42 + index)
            for index, (name, values) in enumerate(paired.items())},
        'strata': {
            name: {
                cell: {
                    'queries': len(values),
                    'M2-O_median': float(np.median([v[0] for v in values])),
                    'M2-R_median': float(np.median([v[1] for v in values])),
                    'relative_change': float(
                        np.median([v[1] for v in values])
                        / np.median([v[0] for v in values]) - 1.0),
                }
                for cell, values in sorted(cells.items())}
            for name, cells in strata.items()},
        'first_mode_proxy_q50': {
            name: {label: float(np.median(values)) if values else None
                   for label, values in groups.items()}
            for name, groups in first_energy.items()},
        'posterior_to_prior_absolute_innovation': {
            name: (values['numerator'] / values['denominator']
                   if values['denominator'] else None)
            for name, values in posterior.items()},
        'reference_diagnostics': {
            key: {
                'min': float(np.min(values)),
                'median': float(np.median(values)),
                'max': float(np.max(values)),
            } for key, values in reference.items()},
        'background_max_abs_difference': background_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('epoch_07_model.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-profiles', type=int, default=128)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else run_dir / args.checkpoint
    baseline, sw_manager, iri_peak_manager, config = _load(run_dir, checkpoint)
    physical = _physical_shadow(config, checkpoint)
    date_manifest = Path(config['date_split_manifest'])
    if not date_manifest.is_absolute():
        date_manifest = ROOT / date_manifest
    with date_manifest.open(encoding='utf-8') as stream:
        split_days = json.load(stream)['partitions']
    loaders = {
        'FY': _partition_loader(FY3D_Dataset, config, split_days, 'train'),
        'COSMIC': _partition_loader(COSMICDataset, config, split_days, 'train'),
    }
    selected = {
        source: _restrict_profiles(loader, args.max_profiles, 142 + index)
        for index, (source, loader) in enumerate(loaders.items())}
    allowed = {source: np.unique(loader.dataset.profile_ids)
               for source, loader in loaders.items()}
    indices = {
        'FY': FYNeighborhoodIndex(config['fy_path'], config),
        'COSMIC': COSMICNeighborhoodIndex(config['cosmic_path'], config),
    }
    targets = {
        source: _audit_target(
            source, loaders[source], baseline, physical, sw_manager,
            iri_peak_manager, indices, allowed)
        for source in ('FY', 'COSMIC')}
    ci_pass = all(
        targets[target]['paired_effective_rank'][observation][
            'bootstrap_95_ci'][0] > 0
        for target in targets for observation in ('FY', 'COSMIC', 'joint'))
    gram_pass = all(
        target['reference_diagnostics']['mode_reference_gram_error']['max'] <= 1e-5
        and target['reference_diagnostics'][
            'raw_mode_gram_min_eigenvalue']['min'] > 1e-6
        for target in targets.values())
    first_mode_pass = all(
        groups['M2-R'] <= groups['M2-O']
        for target in targets.values()
        for groups in target['first_mode_proxy_q50'].values()
        if groups['M2-R'] is not None and groups['M2-O'] is not None)
    strata_pass = all(
        row['relative_change'] >= -0.05
        for target in targets.values()
        for source_rows in target['strata'].values()
        for row in source_rows.values() if row['queries'] >= 10)
    posterior_pass = all(
        ratio is not None and ratio < 1.0
        for target in targets.values()
        for ratio in target['posterior_to_prior_absolute_innovation'].values())
    report = {
        'schema_version': 1,
        'purpose': 'M2-R R0 train-only paired representation audit',
        'partition': 'train',
        'locked_test_accessed': False,
        'isr_accessed': False,
        'checkpoint': str(checkpoint.resolve()),
        'selected_profiles': {source: int(len(ids))
                              for source, ids in selected.items()},
        'semantics': {
            'analysis_state': 'query_local_increment_coefficients',
            'context': 'endpoint_conditioning_only',
            'basis': 'reference_whitened_physical_modes',
            'R0_unit_scope': 'one query per shadow unit; no training',
        },
        'targets': targets,
        'gates': {
            'reference_geometry_passed': gram_pass,
            'all_rank_bootstrap_lower_bounds_positive': ci_pass,
            'first_mode_energy_not_above_M2O': first_mode_pass,
            'major_strata_not_degraded_over_5pct': strata_pass,
            'R0_passed': gram_pass and ci_pass and first_mode_pass and strata_pass,
            'posterior_innovation_all_sources_decreased': posterior_pass,
            'R1_fixed_grid_payload_and_blending_complete': False,
            'progression_to_R2_allowed': False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
