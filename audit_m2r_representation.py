"""Train-only paired R0 audit for M2-O versus the fixed M2-R representation."""

from __future__ import annotations

import argparse
import json
import math
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
from inr_modules.mdia.fsia_model import FSIA_INR_Model, solve_density_modes
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background, build_failed_shadow_reference_context,
    query_observation_payload,
)


def _physical_shadow(config, checkpoint, dictionary):
    physical_config = dict(config)
    physical_config.update({
        'density_basis_semantics': 'endpoint_context_symmetric',
        'analysis_state_semantics': 'query_local_increment_coefficients',
        'context_semantics': 'endpoint_conditioning_only',
        'mode_basis_semantics': 'reference_whitened_physical_modes',
        'enkf_n_members': 8,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'physical_mode_dictionary': dictionary,
        'allow_failed_query_local_shadow': True,
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
        'profile_means': {
            str(profile_id): float(np.mean(samples))
            for profile_id, samples in sorted(rows.items()) if samples},
    }


def _folded_payload(index, coords, profile_ids, excluded, allowed_folds):
    """Query each profile fold independently so observation tokens never cross folds."""
    merged = None
    for fold in (0, 1):
        selected = (profile_ids % 2) == fold
        if not selected.any():
            continue
        part = query_observation_payload(
            index, coords[selected], torch.device('cpu'),
            exclude_profile_ids=(excluded[selected] if excluded is not None else None),
            allowed_profile_ids=allowed_folds[fold])
        if merged is None:
            merged = {
                key: value.new_zeros((len(coords), *value.shape[1:]))
                for key, value in part.items()}
        for key, value in part.items():
            merged[key][selected] = value
    return merged


def _rank1_increment(extras, sources):
    query = extras['query_anomalies']
    weighted_parts, innovation_parts = [], []
    for source in sources:
        precision = extras[f'precision_{source}']
        root = precision.sqrt()
        weighted_parts.append(extras[f'obs_anomalies_{source}'] * root.unsqueeze(-1))
        innovation_parts.append(extras[f'innov_{source}'] * root)
    weighted = torch.cat(weighted_parts, dim=1)
    innovation = torch.cat(innovation_parts, dim=1)
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    rank1 = (u[..., :1] * singular[..., :1].unsqueeze(-2)) @ vh[..., :1, :]
    members = query.shape[1]
    system = ((members - 1) * torch.eye(
        members, dtype=query.dtype, device=query.device).expand(len(query), -1, -1)
        + rank1.transpose(1, 2) @ rank1)
    rhs = torch.einsum('bmn,bm->bn', rank1, innovation)
    weights = torch.cholesky_solve(
        rhs.unsqueeze(-1), torch.linalg.cholesky(system)).squeeze(-1)
    return torch.einsum('bn,bn->b', query, weights)


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
    squared_error = {
        comparison: {mode: defaultdict(list) for mode in ('M10', 'M01', 'M11')}
        for comparison in ('candidate_minus_M2O', 'candidate_minus_rank1')}
    rmse_totals = {
        mode: {'candidate': 0.0, 'M2O': 0.0, 'count': 0}
        for mode in ('M10', 'M01', 'M11')}
    direction = {
        mode: defaultdict(list) for mode in ('M10', 'M01', 'M11')}
    covariance_sign = {
        name: defaultdict(list) for name in ('FY', 'COSMIC')}
    background_error = 0.0
    with torch.no_grad():
        for data, _, profile_ids in loader:
            coords = data[:, :4]
            sw = sw_manager.get_drivers_sequence(coords[:, 3])
            peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
            raw = {}
            for observation_source in ('FY', 'COSMIC'):
                excluded = (profile_ids.numpy()
                            if observation_source == source else None)
                raw[observation_source] = _folded_payload(
                    indices[observation_source], coords, profile_ids,
                    excluded, allowed[observation_source])
            baseline_payload = {
                name: attach_observation_background(
                    payload, baseline, sw_manager, iri_peak_manager)
                for name, payload in raw.items()}
            physical_payload = {
                name: attach_observation_background(
                    payload, physical, sw_manager, iri_peak_manager)
                for name, payload in raw.items()}
            reference_context = (build_failed_shadow_reference_context(
                coords, physical, sw_manager, iri_peak_manager)
                if physical.physical_mode_dictionary in (
                    'endpoint_hmf2_legendre', 'background_adaptive_fixed')
                else None)
            baseline_result = baseline(
                coords, sw, iri_peak=peak,
                observations_fy=baseline_payload['FY'],
                observations_cosmic=baseline_payload['COSMIC'])
            baseline_extras = baseline_result[4]
            physical_result = physical(
                coords, sw, iri_peak=peak,
                observations_fy=physical_payload['FY'],
                observations_cosmic=physical_payload['COSMIC'],
                physical_reference_context=reference_context)
            physical_extras = physical_result[4]
            background_error = max(background_error, float(torch.max(torch.abs(
                baseline_extras['ne_bkg'] - physical_extras['ne_bkg']))))
            desired = data[:, 4] - baseline_extras['ne_bkg'].flatten()
            baseline_modes = solve_density_modes(baseline_extras)
            candidate_modes = solve_density_modes(physical_extras)
            rank1_modes = {
                'M10': _rank1_increment(physical_extras, ('FY',)),
                'M01': _rank1_increment(physical_extras, ('COSMIC',)),
                'M11': _rank1_increment(physical_extras, ('FY', 'COSMIC')),
            }
            profile_array = profile_ids.numpy()
            for index, profile_id in enumerate(profile_array):
                profile_id = int(profile_id)
                if abs(float(desired[index])) >= 0.05:
                    for mode in candidate_modes:
                        direction[mode][profile_id].append(
                            float(torch.sign(candidate_modes[mode][index])
                                  == torch.sign(desired[index]))
                            - float(torch.sign(baseline_modes[mode][index])
                                    == torch.sign(desired[index])))
                for mode in candidate_modes:
                    candidate_error = (
                        physical_extras['ne_bkg'].flatten()[index]
                        + candidate_modes[mode][index] - data[index, 4])
                    baseline_error = (
                        baseline_extras['ne_bkg'].flatten()[index]
                        + baseline_modes[mode][index] - data[index, 4])
                    rank1_error = (
                        physical_extras['ne_bkg'].flatten()[index]
                        + rank1_modes[mode][index] - data[index, 4])
                    squared_error['candidate_minus_M2O'][mode][profile_id].append(
                        float(candidate_error.square() - baseline_error.square()))
                    squared_error['candidate_minus_rank1'][mode][profile_id].append(
                        float(candidate_error.square() - rank1_error.square()))
                    rmse_totals[mode]['candidate'] += float(candidate_error.square())
                    rmse_totals[mode]['M2O'] += float(baseline_error.square())
                    rmse_totals[mode]['count'] += 1
            for observation_source in ('FY', 'COSMIC'):
                innovation = physical_extras[f'innov_{observation_source}']
                precision = physical_extras[f'precision_{observation_source}']
                covariance = physical_extras[f'cross_covariance_{observation_source}']
                baseline_covariance = baseline_extras[
                    f'cross_covariance_{observation_source}']
                for index, profile_id in enumerate(profile_array):
                    selected_tokens = ((precision[index] > 0)
                                       & (innovation[index].abs() >= 0.05)
                                       & (desired[index].abs() >= 0.05))
                    if selected_tokens.any():
                        empirical = desired[index] * innovation[index]
                        candidate_correct = (
                            empirical * covariance[index] > 0)[selected_tokens].float().mean()
                        baseline_correct = (
                            empirical * baseline_covariance[index] > 0)[
                                selected_tokens].float().mean()
                        covariance_sign[observation_source][int(profile_id)].append(
                            float(candidate_correct - baseline_correct))
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
        'paired_squared_error': {
            comparison: {
                mode: _profile_ci(rows, 80 + index)
                for index, (mode, rows) in enumerate(modes.items())}
            for comparison, modes in squared_error.items()},
        'rmse_ratio_candidate_to_M2O': {
            mode: math.sqrt(values['candidate'] / values['M2O'])
            if values['count'] and values['M2O'] > 0 else None
            for mode, values in rmse_totals.items()},
        'paired_direction_accuracy': {
            mode: _profile_ci(rows, 100 + index)
            for index, (mode, rows) in enumerate(direction.items())},
        'paired_covariance_sign_accuracy': {
            name: _profile_ci(rows, 120 + index)
            for index, (name, rows) in enumerate(covariance_sign.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('epoch_07_model.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-profiles', type=int, default=512)
    parser.add_argument(
        '--dictionary', choices=(
            'query_hmf2_legendre', 'endpoint_hmf2_legendre',
            'background_adaptive_fixed'), required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else run_dir / args.checkpoint
    baseline, sw_manager, iri_peak_manager, config = _load(run_dir, checkpoint)
    physical = _physical_shadow(config, checkpoint, args.dictionary)
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
    allowed = {}
    for source, loader in loaders.items():
        profile_ids = np.unique(loader.dataset.profile_ids)
        allowed[source] = tuple(
            profile_ids[profile_ids % 2 == fold] for fold in (0, 1))
    indices = {
        'FY': FYNeighborhoodIndex(config['fy_path'], config),
        'COSMIC': COSMICNeighborhoodIndex(config['cosmic_path'], config),
    }
    targets = {
        source: _audit_target(
            source, loaders[source], baseline, physical, sw_manager,
            iri_peak_manager, indices, allowed)
        for source in ('FY', 'COSMIC')}
    rank_rows = [
        targets[target]['paired_effective_rank'][observation]
        for target in targets for observation in ('FY', 'COSMIC', 'joint')]
    ci_pass = all(
        row is not None and row['bootstrap_95_ci'][0] > 0
        for row in rank_rows)
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
    direction_pass = all(
        row is not None and row['bootstrap_95_ci'][0] > 0
        for target in targets.values()
        for row in target['paired_direction_accuracy'].values())
    covariance_sign_pass = all(
        row is not None and row['bootstrap_95_ci'][0] > 0
        for target in targets.values()
        for row in target['paired_covariance_sign_accuracy'].values())
    rmse_pass = all(
        ratio is not None and ratio <= 1.01
        for target in targets.values()
        for ratio in target['rmse_ratio_candidate_to_M2O'].values())
    rank1_pass = all(
        row is not None and row['bootstrap_95_ci'][1] < 0
        for target in targets.values()
        for row in target['paired_squared_error']['candidate_minus_rank1'].values())
    report = {
        'schema_version': 1,
        'purpose': 'RSR-3 train-only true query-local representation pre-gate',
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
            'dictionary': args.dictionary,
            'scope': 'one independent ETKF problem per target query; no training',
        },
        'targets': targets,
        'gates': {
            'reference_geometry_passed': gram_pass,
            'all_rank_bootstrap_lower_bounds_positive': ci_pass,
            'first_mode_energy_not_above_M2O': first_mode_pass,
            'major_strata_not_degraded_over_5pct': strata_pass,
            'representation_pre_gate_passed': (
                gram_pass and ci_pass and first_mode_pass and strata_pass),
            'posterior_innovation_all_sources_decreased': posterior_pass,
            'direction_accuracy_improved': direction_pass,
            'covariance_sign_accuracy_improved': covariance_sign_pass,
            'squared_error_not_above_M2O': rmse_pass,
            'profile_paired_better_than_rank1': rank1_pass,
            'progression_to_training_allowed': all((
                gram_pass, ci_pass, first_mode_pass, strata_pass,
                posterior_pass, direction_pass, covariance_sign_pass,
                rmse_pass, rank1_pass)),
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
