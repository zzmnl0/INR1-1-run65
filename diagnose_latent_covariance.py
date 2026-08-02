"""Train/validation covariance and latent-rank audit for run66 ETKF."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from inr_modules.data_managers.FY_dataloader import (
    get_cosmic_dataloader,
    get_dataloaders,
)
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    query_observation_payload,
)
from isr_evaluation.main_isr_eval import (
    CONFIG as ISR_CONFIG,
    _load_model_and_managers,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = 'run66-qc2-latent-etkf-density-H-global-localized'


def _profile_groups(loader, max_profiles, seed):
    sampler = loader.batch_sampler
    sampler.shuffle = False
    sampler.points_per_profile = 8
    groups = [
        indices for values in sampler.profiles_by_bin.values()
        for indices in values
    ]
    profile_ids = np.asarray([
        int(loader.dataset.profile_ids[indices[0]]) for indices in groups
    ], dtype=np.int64)
    order = np.argsort(profile_ids, kind='stable')
    groups = [groups[index] for index in order]
    profile_ids = profile_ids[order]
    if len(groups) > max_profiles:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(groups), max_profiles, replace=False))
        groups = [groups[index] for index in keep]
        profile_ids = profile_ids[keep]
    selected = set(profile_ids.tolist())
    sampler.profiles_by_bin = {
        bin_id: [
            indices for indices in values
            if int(loader.dataset.profile_ids[indices[0]]) in selected
        ]
        for bin_id, values in sampler.profiles_by_bin.items()
    }
    sampler.profiles_by_bin = {
        key: values for key, values in sampler.profiles_by_bin.items()
        if values
    }
    return profile_ids


def _spectrum(matrix, requested=(32, 64, 128, 256)):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return {'rows': int(len(matrix)), 'available_components': 0}
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular * singular
    total = energy.sum()
    cumulative = np.cumsum(energy) / total if total > 0 else np.zeros_like(energy)
    return {
        'rows': int(matrix.shape[0]),
        'columns': int(matrix.shape[1]),
        'available_components': int(len(singular)),
        'explained_variance': {
            str(k): float(cumulative[min(k, len(cumulative)) - 1])
            for k in requested if len(cumulative)
        },
        'singular_values': singular[:min(32, len(singular))].tolist(),
    }


def _rank_metrics(anomalies):
    singular = torch.linalg.svdvals(anomalies)
    largest = singular[:, :1].clamp_min(torch.finfo(singular.dtype).eps)
    numeric = (singular > largest * 1e-6).sum(dim=1)
    energy = singular.square()
    probability = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    effective = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum(dim=1))
    return numeric.cpu().numpy(), effective.cpu().numpy()


def _direction_summary(desired, response):
    desired = np.concatenate(desired)
    response = np.concatenate(response)
    if len(desired) == 0:
        return {
            'n': 0, 'comparable_n': 0, 'toward_target_fraction': None,
            'desired_mean': None, 'response_mean': None,
        }
    comparable = (np.abs(desired) > 1e-8) & (np.abs(response) > 1e-8)
    return {
        'n': int(len(desired)),
        'comparable_n': int(comparable.sum()),
        'toward_target_fraction': (
            float(np.mean(desired[comparable] * response[comparable] > 0))
            if comparable.any() else None),
        'desired_mean': float(desired.mean()),
        'response_mean': float(response.mean()),
    }


def _covariance_summary(matches, totals):
    return {
        'comparable_tokens': int(totals),
        'sign_accuracy': float(matches / totals) if totals else None,
    }


def _profile_rate(values, profile_ids, seed=42, bootstrap=1000):
    unique = np.unique(profile_ids)
    rates = np.asarray([
        np.mean(values[profile_ids == profile_id]) for profile_id in unique
    ], dtype=np.float64)
    if not len(rates):
        return {'n_profiles': 0, 'rate': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    draws = rates[rng.integers(0, len(rates), size=(bootstrap, len(rates)))]
    return {
        'n_profiles': int(len(rates)),
        'rate': float(rates.mean()),
        'ci95': np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist(),
    }


def _attribution_summary(arrays, mask, bootstrap=False):
    selected = int(mask.sum())
    if selected == 0:
        return {'n_tokens': 0, 'n_profiles': 0}
    result = {
        'n_tokens': selected,
        'n_profiles': int(np.unique(arrays['profile_id'][mask]).size),
    }
    for key in (
            'innovation_target_agreement',
            'covariance_sign_accuracy',
            'kalman_contribution_toward'):
        values = arrays[key][mask]
        result[key] = float(values.mean())
        if bootstrap:
            result[f'{key}_profile_bootstrap'] = _profile_rate(
                values, arrays['profile_id'][mask])
    return result


def _group_attribution(arrays, values, name):
    rows = []
    for value in np.unique(values):
        mask = values == value
        rows.append({
            'group': int(value),
            **_attribution_summary(arrays, mask),
        })
    return {name: rows}


def _attribution_report(parts):
    if not parts['profile_id']:
        return {'n_tokens': 0, 'representativeness_stop_triggered': False}
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    all_mask = np.ones(len(arrays['profile_id']), dtype=bool)
    night_low = (
        (arrays['altitude'] >= 120.0) & (arrays['altitude'] < 300.0)
        & ((arrays['local_time'] < 6.0) | (arrays['local_time'] >= 18.0)))
    rho = arrays['rho']
    distance_group = np.searchsorted(
        np.quantile(rho, [0.25, 0.5, 0.75]), rho, side='right') + 1
    result = {
        'deadband_dex': 0.05,
        'bootstrap_replicates': 1000,
        'overall': _attribution_summary(arrays, all_mask, bootstrap=True),
        'night_120_300km': _attribution_summary(
            arrays, night_low, bootstrap=True),
        'stratified': {},
    }
    result['stratified'].update(_group_attribution(
        arrays, arrays['date'], 'date'))
    result['stratified'].update(_group_attribution(
        arrays, (np.floor(arrays['altitude'] / 20.0) * 20).astype(int),
        'altitude_20km'))
    result['stratified'].update(_group_attribution(
        arrays, np.floor(arrays['local_time']).astype(int), 'local_time_1h'))
    result['stratified'].update(_group_attribution(
        arrays, distance_group, 'distance_quartile'))
    low = result['night_120_300km']
    result['representativeness_stop_triggered'] = bool(
        low.get('innovation_target_agreement', 1.0) < 0.55
        and low.get('covariance_sign_accuracy', 0.0) >= 0.65)
    return result


def _profile_residual_spectrum(
        model, loader, selected_profiles, sw_manager, iri_peak_manager, device):
    dataset = loader.dataset
    selected = np.isin(dataset.profile_ids, selected_profiles)
    dataset_rows = np.flatnonzero(selected)
    actual_rows = dataset.selected_indices[dataset_rows]
    data = np.asarray(dataset.data[actual_rows, :5], dtype=np.float32)
    profile_ids = dataset.profile_ids[dataset_rows]
    residual_parts = []
    with torch.no_grad():
        for start in range(0, len(data), 2048):
            coords = torch.from_numpy(data[start:start + 2048, :4]).to(device)
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            iri_peak = (
                iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
            background = model.encode_background(
                coords, sw_seq, iri_peak=iri_peak)['ne_bkg']
            residual_parts.append(
                data[start:start + 2048, 4]
                - background.squeeze(-1).cpu().numpy())
    residual = np.concatenate(residual_parts)
    grid = np.arange(150.0, 450.0 + 1e-6, 2.5)
    rows = []
    for profile_id in np.unique(profile_ids):
        mask = profile_ids == profile_id
        altitude = data[mask, 2]
        values = residual[mask]
        order = np.argsort(altitude, kind='stable')
        altitude, unique = np.unique(altitude[order], return_index=True)
        values = values[order][unique]
        if len(altitude) >= 2 and altitude[0] <= grid[0] and altitude[-1] >= grid[-1]:
            rows.append(np.interp(grid, altitude, values))
    result = _spectrum(rows)
    result.update({
        'grid_min_km': float(grid[0]),
        'grid_max_km': float(grid[-1]),
        'grid_step_km': 2.5,
        'eligible_profiles': int(len(rows)),
        'no_extrapolation': True,
    })
    return result


def _audit_loader(
        model, loader, source, index, sw_manager, iri_peak_manager, device,
        selected_profiles):
    suffix = source.upper()
    keyword = 'observations_fy' if source == 'fy' else 'observations_cosmic'
    desired_profile, response_profile = [], []
    desired_low_night, response_low_night = [], []
    desired_height, response_height = [], []
    covariance_matches = covariance_total = 0
    negative_gain = effective_gain = 0
    negative_innovation_correct = negative_innovation_total = 0
    numeric_ranks, effective_ranks, basis_rows = [], [], []
    anomaly_conditions, scale_saturations, factor_scales = [], [], []
    attribution = {key: [] for key in (
        'profile_id', 'altitude', 'local_time', 'date', 'rho',
        'innovation_target_agreement', 'covariance_sign_accuracy',
        'kalman_contribution_toward')}

    with torch.no_grad():
        for batch in loader:
            data, _, profile_ids = batch
            data = data.to(device)
            profile_ids = profile_ids.to(device)
            coords, target = data[:, :4], data[:, 4]
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            iri_peak = (
                iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)

            leave_profile = query_observation_payload(
                index, coords, device,
                exclude_profile_ids=profile_ids.cpu().numpy())
            leave_profile = attach_observation_background(
                leave_profile, model, sw_manager, iri_peak_manager)
            prediction, _, _, _, extras = model(
                coords, sw_seq, iri_peak=iri_peak,
                **{keyword: leave_profile})
            background = extras['ne_bkg'].squeeze(-1)
            desired = target - background
            response = prediction.squeeze(-1) - background
            desired_profile.append(desired.cpu().numpy())
            response_profile.append(response.cpu().numpy())
            local_time = torch.remainder(
                coords[:, 3] + coords[:, 1] / 15.0, 24.0)
            low_night = (
                (coords[:, 2] >= 120.0) & (coords[:, 2] < 300.0)
                & ((local_time < 6.0) | (local_time >= 18.0)))
            desired_low_night.append(desired[low_night].cpu().numpy())
            response_low_night.append(response[low_night].cpu().numpy())

            innovation = extras[f'innov_{suffix}']
            cross = extras[f'cross_covariance_{suffix}']
            precision = extras[f'precision_{suffix}']
            empirical = desired.unsqueeze(-1) * innovation
            comparable = (
                leave_profile['valid_mask'] & (precision > 0)
                & (empirical.abs() > 1e-8) & (cross.abs() > 1e-10))
            covariance_matches += int(
                ((empirical * cross > 0) & comparable).sum().item())
            covariance_total += int(comparable.sum().item())
            effective = leave_profile['valid_mask'] & (precision > 0)
            gain = extras[f'K_{suffix}']
            negative_gain += int(((gain < 0) & effective).sum().item())
            effective_gain += int(effective.sum().item())
            precision_sum = precision.sum(dim=-1)
            innovation_mean = (
                (innovation * precision).sum(dim=-1)
                / precision_sum.clamp_min(1e-12))
            negative_rows = (
                (precision_sum > 0) & (innovation_mean < -1e-8)
                & (response.abs() > 1e-8))
            negative_innovation_correct += int(
                ((response < 0) & negative_rows).sum().item())
            negative_innovation_total += int(negative_rows.sum().item())
            high_confidence = (
                leave_profile['valid_mask'] & (precision > 0)
                & (desired.abs().unsqueeze(-1) >= 0.05)
                & (innovation.abs() >= 0.05)
                & torch.isfinite(cross) & torch.isfinite(gain))
            if high_confidence.any():
                expanded_desired = desired.unsqueeze(-1).expand_as(innovation)
                expanded_profile = profile_ids.unsqueeze(-1).expand_as(
                    leave_profile['profile_id'])
                expanded_altitude = coords[:, 2].unsqueeze(-1).expand_as(
                    innovation)
                expanded_local_time = local_time.unsqueeze(-1).expand_as(
                    innovation)
                expanded_date = (
                    torch.floor(coords[:, 3] / 24.0).to(torch.long) + 1
                ).unsqueeze(-1).expand_as(leave_profile['profile_id'])
                contribution = gain * innovation
                empirical = expanded_desired * innovation
                values = {
                    'profile_id': expanded_profile,
                    'altitude': expanded_altitude,
                    'local_time': expanded_local_time,
                    'date': expanded_date,
                    'rho': leave_profile['rho_squared'].clamp_min(0).sqrt(),
                    'innovation_target_agreement': (
                        expanded_desired * innovation > 0),
                    'covariance_sign_accuracy': empirical * cross > 0,
                    'kalman_contribution_toward': (
                        expanded_desired * contribution > 0),
                }
                for key, value in values.items():
                    attribution[key].append(
                        value[high_confidence].cpu().numpy())

            numeric, effective = _rank_metrics(extras['latent_anomalies'])
            numeric_ranks.append(numeric)
            effective_ranks.append(effective)
            basis_rows.append(extras['query_basis'].cpu().numpy())
            anomaly_conditions.append(
                extras['anomaly_condition'].cpu().numpy())
            scale_saturations.append(
                extras['scale_boundary_saturation'].cpu().numpy())
            if extras['factor_scales'] is not None:
                factor_scales.append(
                    extras['factor_scales'].cpu().numpy().reshape(-1))

            leave_height = query_observation_payload(index, coords, device)
            same_profile = (
                leave_height['profile_id']
                == profile_ids.unsqueeze(-1))
            different_height = (
                leave_height['coords'][..., 2]
                - coords[:, None, 2]).abs() > 1e-3
            leave_height['valid_mask'] &= same_profile & different_height
            leave_height = attach_observation_background(
                leave_height, model, sw_manager, iri_peak_manager)
            height_prediction, _, _, _, height_extras = model(
                coords, sw_seq, iri_peak=iri_peak,
                **{keyword: leave_height})
            height_background = height_extras['ne_bkg'].squeeze(-1)
            desired_height.append((target - height_background).cpu().numpy())
            response_height.append(
                (height_prediction.squeeze(-1) - height_background).cpu().numpy())

    numeric = np.concatenate(numeric_ranks)
    effective = np.concatenate(effective_ranks)
    return {
        'selected_profiles': int(len(selected_profiles)),
        'leave_one_profile': {
            **_direction_summary(desired_profile, response_profile),
            'cross_covariance': _covariance_summary(
                covariance_matches, covariance_total),
            'negative_gain_fraction': (
                float(negative_gain / effective_gain)
                if effective_gain else None),
            'negative_innovation_toward_fraction': (
                float(negative_innovation_correct / negative_innovation_total)
                if negative_innovation_total else None),
            'negative_innovation_queries': negative_innovation_total,
        },
        'leave_one_profile_night_120_300km': _direction_summary(
            desired_low_night, response_low_night),
        'high_confidence_attribution': _attribution_report(attribution),
        'leave_one_height': _direction_summary(
            desired_height, response_height),
        'latent_rank': {
            'numeric_q05_q50_q95': np.quantile(
                numeric, [0.05, 0.5, 0.95]).tolist(),
            'effective_q05_q50_q95': np.quantile(
                effective, [0.05, 0.5, 0.95]).tolist(),
            'collapse_stop_triggered': bool(np.median(effective) < 4.0),
            'condition_q05_q50_q95': np.quantile(
                np.concatenate(anomaly_conditions),
                [0.05, 0.5, 0.95]).tolist(),
            'scale_boundary_saturation_mean': float(np.mean(
                np.concatenate(scale_saturations))),
            'scale_q05_q50_q95': (
                np.quantile(
                    np.concatenate(factor_scales),
                    [0.05, 0.5, 0.95]).tolist()
                if factor_scales else None),
        },
        'basis_spectrum': _spectrum(np.concatenate(basis_rows)),
        'profile_residual_spectrum': _profile_residual_spectrum(
            model, loader, selected_profiles, sw_manager,
            iri_peak_manager, device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=Path,
        default=ROOT / 'checkpoints_fsia' / DEFAULT_RUN
        / 'best_fsia_model.pth')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'isr_validation_outputs' / DEFAULT_RUN
        / 'latent_covariance_audit')
    parser.add_argument('--max-profiles', type=int, default=1000)
    args = parser.parse_args()
    if args.max_profiles < 1:
        raise ValueError('max-profiles must be positive')

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(ISR_CONFIG)
    config['checkpoint_path'] = str(args.checkpoint.resolve())
    (model, sw_manager, cfg, _, iri_peak_manager,
     fy_index, cosmic_index) = _load_model_and_managers(config, device)
    loader_kwargs = {
        'batch_size': min(int(cfg['batch_size']), 512),
        'bin_size_hours': cfg['bin_size_hours'],
        'num_workers': 0,
        'use_memmap': True,
        'val_ratio': cfg.get('val_ratio', 0.1),
        'split_seed': cfg['seed'],
        'points_per_profile': 8,
    }
    fy_loaders = get_dataloaders(
        cfg['fy_path'], profile_path=cfg.get('fy_profile_path'),
        profile_index_path=cfg.get('fy_profile_index_path'), **loader_kwargs)
    cosmic_loaders = get_cosmic_dataloader(
        cfg['cosmic_path'],
        profile_index_path=cfg.get('cosmic_profile_index_path'),
        **loader_kwargs)

    report = {
        'schema_version': 1,
        'checkpoint': str(args.checkpoint.resolve()),
        'seed': 42,
        'max_profiles_per_source_split': args.max_profiles,
        'architecture': {
            'basis_dim': model.kalman_layer.d_model,
            'enkf_n_members': model.kalman_layer.n_members,
            'anomaly_parameterization': (
                model.kalman_layer.anomaly_parameterization),
            'scale_init': model.kalman_layer.scale_init,
            'scale_condition_max': (
                model.kalman_layer.scale_condition_max),
        },
        'splits': {},
    }
    for split_index, split in enumerate(('train', 'validation')):
        report['splits'][split] = {}
        for source_index, (source, loaders, index) in enumerate((
                ('fy', fy_loaders, fy_index),
                ('cosmic', cosmic_loaders, cosmic_index))):
            loader = loaders[split_index]
            selected = _profile_groups(
                loader, args.max_profiles,
                42 + 10 * split_index + source_index)
            report['splits'][split][source.upper()] = _audit_loader(
                model, loader, source, index, sw_manager,
                iri_peak_manager, device, selected)

    collapse = any(
        report['splits'][split][source]['latent_rank'][
            'collapse_stop_triggered']
        for split in report['splits']
        for source in report['splits'][split])
    representativeness = any(
        report['splits'][split][source]['high_confidence_attribution'][
            'representativeness_stop_triggered']
        for split in report['splits']
        for source in report['splits'][split])
    report['stop_gates'] = {
        'latent_effective_rank_below_4': collapse,
        'observation_representativeness_conflict': representativeness,
        'p1_allowed': not collapse and not representativeness,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'latent_covariance_audit.json').open(
            'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        'output': str(output),
        'stop_gates': report['stop_gates'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
