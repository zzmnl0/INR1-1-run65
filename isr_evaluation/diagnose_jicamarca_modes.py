"""Jicamarca M00/M10/M01/M11 diagnostics for the v7 physical-observation ETKF."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    query_observation_payload,
)
from isr_evaluation.isr_loader import load_jicamarca
from isr_evaluation.main_isr_eval import (
    CONFIG as ISR_CONFIG,
    _load_model_and_managers,
    _parse_unix,
)


CHECKPOINT = (
    ROOT / 'checkpoints_fsia'
    / 'run66-qc2-latent-etkf-density-H-global-localized'
    / 'best_fsia_model.pth')
OUTPUT = (
    ROOT / 'isr_validation_outputs'
    / 'run66-qc2-latent-etkf-density-H-global-localized'
    / 'Jicamarca' / 'source_mode_diagnostics')
MODES = ('M00', 'M10', 'M01', 'M11')


def _record_arrays(record, start_unix):
    altitude = np.tile(
        record['alt_1d'][:, None], (1, len(record['ts_1d'])))
    relative_hour = np.tile(
        ((record['ts_1d'] - start_unix) / 3600.0)[None, :],
        (len(record['alt_1d']), 1))
    if (record.get('geo_lat_2d') is not None
            and record.get('geo_lon_2d') is not None):
        latitude = np.asarray(record['geo_lat_2d'], dtype=np.float32)
        longitude = np.asarray(record['geo_lon_2d'], dtype=np.float32)
    else:
        latitude = np.full_like(altitude, record['lat'], dtype=np.float32)
        longitude = np.full_like(altitude, record['lon'], dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        observation = np.log10(record['ne_2d'].astype(np.float64))
    valid = (
        np.isfinite(observation) & np.isfinite(altitude)
        & np.isfinite(relative_hour) & np.isfinite(latitude)
        & np.isfinite(longitude))
    coords = np.column_stack([
        latitude[valid], longitude[valid], altitude[valid], relative_hour[valid],
    ]).astype(np.float32)
    return coords, observation[valid].astype(np.float32)


def _summary(mask, prediction, background, observation):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {'n': 0}
    prediction = prediction[mask]
    background = background[mask]
    observation = observation[mask]
    increment = prediction - background
    desired = observation - background
    comparable = (np.abs(increment) > 1e-8) & (np.abs(desired) > 1e-8)
    negative = desired <= -0.05
    quantiles = np.quantile(increment, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        'n': int(mask.sum()),
        'rmse': float(np.sqrt(np.mean((prediction - observation) ** 2))),
        'background_rmse': float(
            np.sqrt(np.mean((background - observation) ** 2))),
        'increment_mean': float(np.mean(increment)),
        'desired_increment_mean': float(np.mean(desired)),
        'toward_isr_fraction': (
            float(np.mean(increment[comparable] * desired[comparable] > 0))
            if comparable.any() else None),
        'increment_p01': float(quantiles[0]),
        'increment_p05': float(quantiles[1]),
        'increment_p50': float(quantiles[2]),
        'increment_p95': float(quantiles[3]),
        'increment_p99': float(quantiles[4]),
        'negative_desired_n': int(negative.sum()),
        'negative_desired_positive_tail_fraction': (
            float(np.mean(increment[negative] >= 0.05))
            if negative.any() else None),
        'negative_desired_p95': (
            float(np.quantile(increment[negative], 0.95))
            if negative.any() else None),
        'negative_desired_p99': (
            float(np.quantile(increment[negative], 0.99))
            if negative.any() else None),
    }


def _direct_index_coverage(index, coords):
    unique, inverse = np.unique(
        np.asarray(coords)[:, [0, 1, 3]], axis=0, return_inverse=True)
    meta = np.asarray(index.prof_meta)
    direct = np.zeros(len(unique), dtype=bool)
    for row_index, (latitude, longitude, relative_hour) in enumerate(unique):
        dlon = np.abs((meta[:, 1] - longitude + 180.0) % 360.0 - 180.0)
        direct[row_index] = np.any(
            (np.abs(meta[:, 0] - latitude) <= index.dlat)
            & (dlon <= index.dlon)
            & (np.abs(meta[:, 2] - relative_hour) <= index.dt))
    return direct[inverse]


def _query_mean(values, valid):
    count = valid.sum(axis=1)
    return np.divide(
        np.where(valid, values, 0.0).sum(axis=1),
        count,
        out=np.full(len(valid), np.nan, dtype=np.float64),
        where=count > 0)


def _source_arrays(payload, extras, suffix):
    valid = payload['valid_mask'].cpu().numpy()
    innovation = extras[f'innov_{suffix}'].cpu().numpy()
    gain = extras[f'K_{suffix}'].cpu().numpy()
    precision = extras[f'precision_{suffix}'].cpu().numpy()
    cross_covariance = extras[
        f'cross_covariance_{suffix}'].cpu().numpy()
    contribution = gain * innovation
    r_eff = np.full_like(precision, np.nan)
    np.divide(1.0, precision, out=r_eff, where=precision > 0)
    dominant = np.argmax(np.where(valid, np.abs(contribution), -1.0), axis=1)
    rows = np.arange(len(valid))
    effective = valid & (precision > 0)
    precision_sum = np.where(effective, precision, 0.0).sum(axis=1)
    precision_square_sum = np.where(
        effective, precision * precision, 0.0).sum(axis=1)
    effective_sample_size = np.divide(
        precision_sum * precision_sum,
        precision_square_sum,
        out=np.zeros_like(precision_sum),
        where=precision_square_sum > 0)
    return {
        'physical_innovation_mean': _query_mean(innovation, valid),
        'kalman_gain_mean': _query_mean(gain, valid),
        'cross_covariance_mean': _query_mean(cross_covariance, valid),
        'r_eff_mean': _query_mean(r_eff, valid),
        'kalman_contribution_sum': np.sum(
            np.where(valid, contribution, 0.0), axis=1),
        'dominant_profile_id': payload['profile_id'].cpu().numpy()[
            rows, dominant],
        'dominant_rho': np.sqrt(payload['rho_squared'].cpu().numpy()[
            rows, dominant]),
        'valid_tokens': valid.sum(axis=1),
        'effective_tokens': effective.sum(axis=1),
        'precision_sum': precision_sum,
        'effective_sample_size': effective_sample_size,
    }


def _observation_diagnostics(payload, extras, suffix, query_mask,
                             query_offset, query_observation, query_background):
    valid = payload['valid_mask'].cpu().numpy() & query_mask[:, None]
    query_row, token = np.nonzero(valid)
    if not len(query_row):
        return None
    precision = extras[f'precision_{suffix}'].cpu().numpy()[query_row, token]
    gain = extras[f'K_{suffix}'].cpu().numpy()[query_row, token]
    innovation = extras[f'innov_{suffix}'].cpu().numpy()[query_row, token]
    return {
        'source': payload['source'].cpu().numpy()[query_row, token],
        'query_index': query_row.astype(np.int64) + query_offset,
        'query_observation': query_observation[query_row],
        'query_background': query_background[query_row],
        'profile_id': payload['profile_id'].cpu().numpy()[query_row, token],
        'observation_coords': payload['coords'].cpu().numpy()[query_row, token],
        'observation_value': payload['value'].cpu().numpy()[query_row, token],
        'observation_background': (
            payload['background'].cpu().numpy()[query_row, token]),
        'rho_squared': payload['rho_squared'].cpu().numpy()[query_row, token],
        'physical_innovation': innovation,
        'cross_covariance': extras[
            f'cross_covariance_{suffix}'].cpu().numpy()[query_row, token],
        'kalman_gain': gain,
        'precision': precision,
        'r_eff': np.divide(
            1.0, precision, out=np.full_like(precision, np.inf),
            where=precision > 0),
        'kalman_contribution': gain * innovation,
    }


def _latent_rank_metrics(latent_anomalies):
    singular_values = torch.linalg.svdvals(latent_anomalies)
    largest = singular_values[:, :1].clamp_min(torch.finfo(
        singular_values.dtype).eps)
    numeric_rank = (singular_values > largest * 1e-6).sum(dim=1)
    energy = singular_values.square()
    probability = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum(dim=1))
    return (
        singular_values.cpu().numpy(),
        numeric_rank.cpu().numpy(),
        effective_rank.cpu().numpy(),
    )


def _token_group_rows(tokens, selected, labels):
    rows = []
    desired_all = (
        tokens['query_observation'] - tokens['query_background'])
    for label in np.unique(labels[selected]):
        mask = selected & (labels == label)
        desired = desired_all[mask]
        contribution = tokens['kalman_contribution'][mask]
        comparable = (
            np.abs(desired) > 1e-8) & (np.abs(contribution) > 1e-12)
        rows.append({
            'group': int(label),
            'n': int(mask.sum()),
            'innovation_mean': float(
                tokens['physical_innovation'][mask].mean()),
            'cross_covariance_mean': float(
                tokens['cross_covariance'][mask].mean()),
            'kalman_contribution_mean': float(contribution.mean()),
            'conflict_fraction': float(np.mean(
                (desired <= -0.05) & (contribution > 0))),
            'toward_isr_fraction': (
                float(np.mean(desired[comparable] * contribution[comparable] > 0))
                if comparable.any() else None),
        })
    return rows


def _stratified_token_summary(tokens, query_coords):
    query = query_coords[tokens['query_index']]
    day = np.floor(query[:, 3] / 24.0).astype(np.int16) + 1
    altitude_bin = (
        np.floor((query[:, 2] - 120.0) / 20.0) * 20.0 + 120.0
    ).astype(np.int16)
    local_time = np.remainder(query[:, 3] + query[:, 1] / 15.0, 24.0)
    local_time_bin = np.floor(local_time).astype(np.int16)
    result = {}
    for source_code, source_name in ((0, 'FY'), (1, 'COSMIC')):
        selected = (
            (tokens['source'] == source_code)
            & (tokens['precision'] > 0))
        result[source_name] = {
            'date': _token_group_rows(tokens, selected, day),
            'altitude_20km': _token_group_rows(
                tokens, selected, altitude_bin),
            'local_time_1h': _token_group_rows(
                tokens, selected, local_time_bin),
        }
    return result


def _distance_quartile_summary(tokens):
    result = {}
    for source_code, source_name in ((0, 'FY'), (1, 'COSMIC')):
        selected = (
            (tokens['source'] == source_code)
            & (tokens['precision'] > 0))
        if not selected.any():
            result[source_name] = {'n': 0}
            continue
        rho = np.sqrt(tokens['rho_squared'][selected])
        edges = np.quantile(rho, [0.0, 0.25, 0.5, 0.75, 1.0])
        query_index = tokens['query_index'][selected]
        desired = (
            tokens['query_observation'][selected]
            - tokens['query_background'][selected])
        contribution = tokens['kalman_contribution'][selected]
        rows = []
        for quartile in range(4):
            upper = rho <= edges[quartile + 1] if quartile == 3 else (
                rho < edges[quartile + 1])
            mask = (rho >= edges[quartile]) & upper
            conflict = (desired[mask] <= -0.05) & (contribution[mask] > 0)
            comparable = (
                np.abs(desired[mask]) > 1e-8
            ) & (np.abs(contribution[mask]) > 1e-12)
            rows.append({
                'quartile': quartile + 1,
                'n_tokens': int(mask.sum()),
                'n_queries': int(np.unique(query_index[mask]).size),
                'rho_min': float(edges[quartile]),
                'rho_max': float(edges[quartile + 1]),
                'conflict_fraction': float(conflict.mean()),
                'toward_isr_fraction': (
                    float(np.mean(
                        desired[mask][comparable]
                        * contribution[mask][comparable] > 0))
                    if comparable.any() else None),
            })
        conflict_gap = (
            rows[-1]['conflict_fraction'] - rows[0]['conflict_fraction'])
        toward_gap = (
            rows[0]['toward_isr_fraction'] - rows[-1]['toward_isr_fraction']
            if rows[0]['toward_isr_fraction'] is not None
            and rows[-1]['toward_isr_fraction'] is not None else None)
        result[source_name] = {
            'n': int(selected.sum()),
            'quartiles': rows,
            'farthest_minus_nearest_conflict': float(conflict_gap),
            'nearest_minus_farthest_toward': (
                float(toward_gap) if toward_gap is not None else None),
            'distance_stop_triggered': bool(
                conflict_gap >= 0.10
                or (toward_gap is not None and toward_gap >= 0.10)),
        }
    return result


def _top_positive_profiles(tokens, source_code, limit=20):
    selected = (
        (tokens['source'] == source_code)
        & (tokens['precision'] > 0)
        & (tokens['kalman_contribution'] > 0))
    if not selected.any():
        return []
    profile_ids = tokens['profile_id'][selected]
    contributions = tokens['kalman_contribution'][selected]
    rows = []
    for profile_id in np.unique(profile_ids):
        mask = profile_ids == profile_id
        rows.append({
            'profile_id': int(profile_id),
            'positive_contribution_sum': float(contributions[mask].sum()),
            'positive_token_count': int(mask.sum()),
            'innovation_mean': float(
                tokens['physical_innovation'][selected][mask].mean()),
            'rho_mean': float(np.sqrt(
                tokens['rho_squared'][selected][mask]).mean()),
        })
    return sorted(
        rows, key=lambda row: row['positive_contribution_sum'],
        reverse=True)[:limit]


def _qc_profile_audit(rows, data_path, index_path, output, source):
    if not rows or not data_path or not index_path:
        return rows, None
    data = np.load(data_path, mmap_mode='r')
    with np.load(index_path, allow_pickle=True) as index:
        profile_ids = np.asarray(index['profile_id'], dtype=np.int64)
        lookup = {int(profile_id): row for row, profile_id in enumerate(profile_ids)}
        fields = [
            field for field in (
                'pass_profile', 'input_points', 'kept_points', 'h_cut_km',
                'hmf2', 'nmf2', 'peak_count', 'md', 'delta',
                'global_topside_gradient', 'local_topside_gradient',
                'fold_error', 'reason_bits', 'representative_lat',
                'representative_lon', 'representative_time', 'date_code',
                'range_rejected_points', 'original_relative_path',
                'original_profile_id', 'output_start', 'output_end')
            if field in index.files
        ]
        audited = []
        curves = []
        for item in rows:
            profile_id = item['profile_id']
            if profile_id not in lookup:
                audited.append({**item, 'qc_index_found': False})
                curves.append(None)
                continue
            row_index = lookup[profile_id]
            metadata = {}
            for field in fields:
                value = index[field][row_index]
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                metadata[field] = value
            start = int(metadata.get('output_start', -1))
            end = int(metadata.get('output_end', -1))
            curve = (
                np.asarray(data[start:end, :5], dtype=np.float32)
                if 0 <= start < end <= len(data) else None)
            curves.append(curve)
            audited.append({
                **item,
                'qc_index_found': True,
                'qc_metadata': metadata,
                'retained_curve_points': int(len(curve)) if curve is not None else 0,
            })

    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 5, figsize=(15, 14), squeeze=False)
    for axis, item, curve in zip(axes.flat, audited, curves):
        if curve is not None and len(curve):
            axis.plot(curve[:, 4], curve[:, 2], linewidth=1.0)
        axis.set_title(f'{source} profile {item["profile_id"]}', fontsize=8)
        axis.set_xlabel('log10Ne')
        axis.set_ylabel('Altitude (km)')
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(audited):]:
        axis.axis('off')
    figure.tight_layout()
    figure_path = output / f'top_positive_{source.lower()}_profiles.png'
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return audited, figure_path.name


def _observation_summary(arrays, source_code):
    selected = arrays['source'] == source_code
    r_eff = arrays['r_eff'][selected]
    finite_r = r_eff[np.isfinite(r_eff)]
    inconsistent = (
        (arrays['kalman_gain'][selected] < 0)
        & (arrays['cross_covariance'][selected] >= 0))
    return {
        'n': int(selected.sum()),
        'innovation_mean': float(arrays['physical_innovation'][selected].mean()),
        'cross_covariance_mean': float(
            arrays['cross_covariance'][selected].mean()),
        'kalman_contribution_mean': float(
            arrays['kalman_contribution'][selected].mean()),
        'negative_gain_fraction': float(
            np.mean(arrays['kalman_gain'][selected] < 0)),
        'negative_gain_nonnegative_cross_covariance_n': int(inconsistent.sum()),
        'negative_gain_nonnegative_cross_covariance_fraction': float(
            inconsistent.mean()),
        'zero_precision_n': int((~np.isfinite(r_eff)).sum()),
        'r_eff_finite_q05_q50_q95': np.quantile(
            finite_r, [0.05, 0.5, 0.95]).tolist(),
    }


def _counterfactual(model, coords, sw_seq, iri_peak, payload, keyword):
    shifted = dict(payload)
    shifted['value'] = (
        payload['value'] + 0.05 * payload['valid_mask'].to(
            payload['value'].dtype))
    base = model(coords, sw_seq, iri_peak=iri_peak, **{keyword: payload})[0]
    changed = model(
        coords, sw_seq, iri_peak=iri_peak, **{keyword: shifted})[0]
    return ((changed - base) / 0.05).squeeze(-1).cpu().numpy()


def _single_source_checks(arrays, mode, coverage, low_night):
    checks = {}
    for name, mask in (
            ('all', np.ones(len(coverage), dtype=bool)),
            ('covered', coverage),
            ('night_120_300km', low_night),
            ('covered_night_120_300km', coverage & low_night)):
        summary = _summary(
            mask, arrays[f'{mode}_prediction'], arrays['background'],
            arrays['observation'])
        if not summary['n']:
            checks[name] = {'passed': False, **summary}
            continue
        tail = summary['negative_desired_positive_tail_fraction']
        checks[name] = {
            **summary,
            'passed': (
                summary['rmse'] <= 1.01 * summary['background_rmse']
                and summary['toward_isr_fraction'] is not None
                and summary['toward_isr_fraction'] >= 0.55
                and (tail is None or (
                    tail <= 0.10
                    and summary['negative_desired_p95'] <= 0.05
                    and summary['negative_desired_p99'] <= 0.10))),
        }
    checks['passed'] = all(value['passed'] for value in checks.values())
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(ISR_CONFIG)
    config['checkpoint_path'] = str(args.checkpoint.resolve())
    config['run_poker_flat'] = False
    (model, sw_manager, cfg, _, iri_peak_manager,
     fy_index, cosmic_index) = _load_model_and_managers(config, device)
    start_unix = _parse_unix(config['start_date_str'])
    records = load_jicamarca(
        config['jicamarca_dir'], start_unix,
        _parse_unix(config['end_date_str']),
        alt_min=config['alt_min'], alt_max=config['alt_max'],
        err_ratio_max=config['err_ratio_max'])

    saved = {
        key: [] for key in (
            'coords', 'observation', 'background', 'fy_coverage',
            'cosmic_coverage', 'fy_direct', 'cosmic_direct')}
    for mode in MODES:
        saved[f'{mode}_prediction'] = []
        saved[f'{mode}_increment'] = []
    saved['M11_FY_contribution'] = []
    saved['M11_COSMIC_contribution'] = []
    for source in ('fy', 'cosmic'):
        for field in (
                'physical_innovation_mean', 'kalman_gain_mean',
                'cross_covariance_mean', 'r_eff_mean',
                'kalman_contribution_sum', 'dominant_profile_id',
                'dominant_rho', 'valid_tokens', 'effective_tokens',
                'precision_sum', 'effective_sample_size',
                'counterfactual_slope'):
            saved[f'{source}_{field}'] = []
    for field in (
            'latent_singular_values', 'latent_numeric_rank',
            'latent_effective_rank', 'anomaly_condition',
            'scale_boundary_saturation'):
        saved[field] = []
    factor_scales = []
    token_saved = {}
    query_offset = 0

    with torch.no_grad():
        for record in records:
            coords_np, observation = _record_arrays(record, start_unix)
            for start in range(0, len(coords_np), config['batch_size']):
                chunk_np = coords_np[start:start + config['batch_size']]
                chunk_observation = observation[
                    start:start + config['batch_size']]
                coords = torch.from_numpy(chunk_np).to(device)
                sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
                iri_peak = (iri_peak_manager.get_iri_peak(coords)
                            if iri_peak_manager is not None else None)
                fy = query_observation_payload(fy_index, coords, device)
                cosmic = query_observation_payload(cosmic_index, coords, device)
                fy = attach_observation_background(
                    fy, model, sw_manager, iri_peak_manager)
                cosmic = attach_observation_background(
                    cosmic, model, sw_manager, iri_peak_manager)
                kwargs = {
                    'M00': {},
                    'M10': {'observations_fy': fy},
                    'M01': {'observations_cosmic': cosmic},
                    'M11': {
                        'observations_fy': fy,
                        'observations_cosmic': cosmic,
                    },
                }
                extras_by_mode = {}
                background = None
                for mode in MODES:
                    prediction, _, _, increment, extras = model(
                        coords, sw_seq, iri_peak=iri_peak, **kwargs[mode])
                    extras_by_mode[mode] = extras
                    saved[f'{mode}_prediction'].append(
                        prediction.squeeze(-1).cpu().numpy())
                    saved[f'{mode}_increment'].append(
                        increment.squeeze(-1).cpu().numpy())
                    if background is None:
                        background = extras['ne_bkg'].squeeze(-1).cpu().numpy()

                fy_arrays = _source_arrays(fy, extras_by_mode['M10'], 'FY')
                cosmic_arrays = _source_arrays(
                    cosmic, extras_by_mode['M01'], 'COSMIC')
                for field, values in fy_arrays.items():
                    saved[f'fy_{field}'].append(values)
                for field, values in cosmic_arrays.items():
                    saved[f'cosmic_{field}'].append(values)
                saved['M11_FY_contribution'].append(
                    extras_by_mode['M11']['update_FY'].squeeze(-1).cpu().numpy())
                saved['M11_COSMIC_contribution'].append(
                    extras_by_mode['M11']['update_COSMIC'].squeeze(
                        -1).cpu().numpy())
                singular, numeric_rank, effective_rank = _latent_rank_metrics(
                    extras_by_mode['M11']['latent_anomalies'])
                saved['latent_singular_values'].append(singular)
                saved['latent_numeric_rank'].append(numeric_rank)
                saved['latent_effective_rank'].append(effective_rank)
                saved['anomaly_condition'].append(
                    extras_by_mode['M11']['anomaly_condition'].cpu().numpy())
                saved['scale_boundary_saturation'].append(
                    extras_by_mode['M11'][
                        'scale_boundary_saturation'].cpu().numpy())
                if extras_by_mode['M11']['factor_scales'] is not None:
                    factor_scales.append(
                        extras_by_mode['M11']['factor_scales'].cpu().numpy())
                saved['fy_counterfactual_slope'].append(_counterfactual(
                    model, coords, sw_seq, iri_peak, fy, 'observations_fy'))
                saved['cosmic_counterfactual_slope'].append(_counterfactual(
                    model, coords, sw_seq, iri_peak, cosmic,
                    'observations_cosmic'))
                local_time = np.remainder(
                    chunk_np[:, 3] + chunk_np[:, 1] / 15.0, 24.0)
                low_night_chunk = (
                    (chunk_np[:, 2] >= 120.0) & (chunk_np[:, 2] < 300.0)
                    & ((local_time < 6.0) | (local_time >= 18.0)))
                for payload, extras, suffix in (
                        (fy, extras_by_mode['M11'], 'FY'),
                        (cosmic, extras_by_mode['M11'], 'COSMIC')):
                    token_rows = _observation_diagnostics(
                        payload, extras, suffix, low_night_chunk, query_offset,
                        chunk_observation, background)
                    if token_rows is not None:
                        for key, values in token_rows.items():
                            token_saved.setdefault(key, []).append(values)
                saved['coords'].append(chunk_np)
                saved['observation'].append(chunk_observation)
                saved['background'].append(background)
                saved['fy_coverage'].append(
                    fy['valid_mask'].any(dim=1).cpu().numpy())
                saved['cosmic_coverage'].append(
                    cosmic['valid_mask'].any(dim=1).cpu().numpy())
                saved['fy_direct'].append(
                    _direct_index_coverage(fy_index, chunk_np))
                saved['cosmic_direct'].append(
                    _direct_index_coverage(cosmic_index, chunk_np))
                query_offset += len(chunk_np)

    arrays = {key: np.concatenate(values) for key, values in saved.items()}
    token_arrays = {
        key: np.concatenate(values) for key, values in token_saved.items()}
    coords = arrays['coords']
    local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    low_night = (
        (coords[:, 2] >= 120.0) & (coords[:, 2] < 300.0)
        & ((local_time < 6.0) | (local_time >= 18.0)))
    fy_coverage = arrays['fy_coverage'].astype(bool)
    cosmic_coverage = arrays['cosmic_coverage'].astype(bool)
    fy_effective = arrays['fy_effective_tokens'] > 0
    cosmic_effective = arrays['cosmic_effective_tokens'] > 0
    both = fy_coverage & cosmic_coverage
    interaction = (
        arrays['M11_increment'] - arrays['M10_increment']
        - arrays['M01_increment'])
    common = {
        mode: _summary(
            both, arrays[f'{mode}_prediction'], arrays['background'],
            arrays['observation'])
        for mode in ('M10', 'M01', 'M11')
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    top_profiles = {
        'FY': _top_positive_profiles(token_arrays, 0),
        'COSMIC': _top_positive_profiles(token_arrays, 1),
    }
    top_figures = {}
    for source, data_key, index_key in (
            ('FY', 'fy_path', 'fy_profile_index_path'),
            ('COSMIC', 'cosmic_path', 'cosmic_profile_index_path')):
        top_profiles[source], top_figures[source] = _qc_profile_audit(
            top_profiles[source], cfg.get(data_key), cfg.get(index_key),
            output, source)

    report = {
        'schema_version': 7,
        'checkpoint': str(args.checkpoint.resolve()),
        'n_points': int(len(coords)),
        'coverage': {
            'FY': float(np.mean(fy_coverage)),
            'COSMIC': float(np.mean(cosmic_coverage)),
            'FY_positive_precision': float(np.mean(fy_effective)),
            'COSMIC_positive_precision': float(np.mean(cosmic_effective)),
            'both': float(np.mean(both)),
            'neither': float(np.mean(~fy_coverage & ~cosmic_coverage)),
            'fy_index_matches_direct': bool(np.array_equal(
                fy_coverage, arrays['fy_direct'])),
            'cosmic_index_matches_direct': bool(np.array_equal(
                cosmic_coverage, arrays['cosmic_direct'])),
        },
        'low_night_observation_tokens': {
            'total': int(len(token_arrays.get('source', []))),
            'FY': _observation_summary(token_arrays, 0),
            'COSMIC': _observation_summary(token_arrays, 1),
            'file': 'jicamarca_low_night_observation_diagnostics.npz',
        },
        'stratified_observation_audit': _stratified_token_summary(
            token_arrays, coords),
        'localization_distance_audit': _distance_quartile_summary(
            token_arrays),
        'top_positive_profiles': top_profiles,
        'top_positive_profile_figures': top_figures,
        'latent_rank': {
            'numeric_rank_q05_q50_q95': np.quantile(
                arrays['latent_numeric_rank'], [0.05, 0.5, 0.95]).tolist(),
            'effective_rank_q05_q50_q95': np.quantile(
                arrays['latent_effective_rank'], [0.05, 0.5, 0.95]).tolist(),
            'collapse_stop_triggered': bool(
                np.median(arrays['latent_effective_rank']) < 4.0),
            'condition_q05_q50_q95': np.quantile(
                arrays['anomaly_condition'], [0.05, 0.5, 0.95]).tolist(),
            'scale_boundary_saturation_mean': float(np.mean(
                arrays['scale_boundary_saturation'])),
            'scale_q05_q50_q95': (
                np.quantile(
                    np.concatenate(factor_scales),
                    [0.05, 0.5, 0.95]).tolist()
                if factor_scales else None),
        },
        'modes': {
            mode: {
                'all': _summary(
                    np.ones(len(coords), dtype=bool),
                    arrays[f'{mode}_prediction'], arrays['background'],
                    arrays['observation']),
                'night_120_300km': _summary(
                    low_night, arrays[f'{mode}_prediction'],
                    arrays['background'], arrays['observation']),
            } for mode in MODES
        },
        'source_physics': {
            source.upper(): {
                field: float(np.nanmean(arrays[f'{source}_{field}']))
                for field in (
                    'physical_innovation_mean', 'kalman_gain_mean',
                    'cross_covariance_mean', 'r_eff_mean',
                    'kalman_contribution_sum', 'counterfactual_slope')
            } for source in ('fy', 'cosmic')
        },
        'joint_source_contribution_max_abs_error': float(np.max(np.abs(
            arrays['M11_FY_contribution']
            + arrays['M11_COSMIC_contribution']
            - arrays['M11_increment']))),
        'acceptance': {
            'M10': _single_source_checks(
                arrays, 'M10', fy_coverage, low_night),
            'M01': _single_source_checks(
                arrays, 'M01', cosmic_coverage, low_night),
            'M11': {
                'common_coverage': common,
                'within_one_percent_of_best_single': (
                    common['M11'].get('rmse', np.inf)
                    <= 1.01 * min(
                        common['M10'].get('rmse', np.inf),
                        common['M01'].get('rmse', np.inf))),
                'interaction_abs_mean': (
                    float(np.mean(np.abs(interaction[both])))
                    if both.any() else None),
            },
        },
    }
    report['acceptance']['M11']['passed'] = report[
        'acceptance']['M11']['within_one_percent_of_best_single']
    report['acceptance']['passed'] = (
        report['coverage']['fy_index_matches_direct']
        and report['coverage']['cosmic_index_matches_direct']
        and report['acceptance']['M10']['passed']
        and report['acceptance']['M01']['passed']
        and report['acceptance']['M11']['passed'])

    rows = []
    for index in range(len(coords)):
        row = {
            'lat': float(coords[index, 0]),
            'lon': float(coords[index, 1]),
            'alt_km': float(coords[index, 2]),
            'relative_hour': float(coords[index, 3]),
            'local_time': float(local_time[index]),
            'observation': float(arrays['observation'][index]),
            'background': float(arrays['background'][index]),
        }
        for mode in MODES:
            row[f'{mode}_prediction'] = float(
                arrays[f'{mode}_prediction'][index])
            row[f'{mode}_increment'] = float(
                arrays[f'{mode}_increment'][index])
        for source in ('fy', 'cosmic'):
            for field in (
                    'physical_innovation_mean', 'kalman_gain_mean',
                    'cross_covariance_mean', 'r_eff_mean',
                    'kalman_contribution_sum', 'dominant_profile_id',
                    'dominant_rho', 'valid_tokens', 'effective_tokens',
                    'precision_sum', 'effective_sample_size',
                    'counterfactual_slope'):
                row[f'{source}_{field}'] = float(
                    arrays[f'{source}_{field}'][index])
        rows.append(row)

    with (output / 'jicamarca_mode_diagnostics.json').open(
            'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    with (output / 'jicamarca_mode_diagnostics.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / 'jicamarca_mode_diagnostics_arrays.npz', **arrays)
    np.savez_compressed(
        output / 'jicamarca_low_night_observation_diagnostics.npz',
        **token_arrays)
    print(json.dumps({
        'output': str(output),
        'coverage': report['coverage'],
        'acceptance_passed': report['acceptance']['passed'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
