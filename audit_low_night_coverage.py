"""Audit losses layered on FY/COSMIC's intrinsic 120--200 km sparsity."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from estimate_empirical_covariance import (
    BACKGROUND_DEFAULT,
    COSMIC_INDEX_PATH,
    COSMIC_PATH,
    EXPECTED_TRAIN_PROFILES,
    FY_INDEX_PATH,
    FY_PATH,
    FY_REPORT_PATH,
    COSMIC_REPORT_PATH,
    ROOT,
    SOURCE_NAMES,
    _background_residual_cache,
    _configuration,
    _deterministic_npz,
    _lookup_neighbor_residual,
    _sha256,
    _strict_background,
)
from inr_modules.data_managers.FY_dataloader import (
    COSMICNeighborhoodIndex,
    FYNeighborhoodIndex,
    _normalized_profile_distance,
    get_cosmic_dataloader,
    get_dataloaders,
)
from main_fsia import _code_identity, _file_identity
from qc_profile_data import COSMIC_INPUT, FY_DIRS, _parse_fy_time, _read_fy


OUTPUT_DEFAULT = (
    ROOT / 'isr_validation_outputs'
    / 'run66-low-night-coverage-bias-audit')
FY_RAW_ROOT = FY_DIRS[0].parent
PAIR_NAMES = ('FY->FY', 'FY->COSMIC', 'COSMIC->FY', 'COSMIC->COSMIC')
STAGES = (
    'hard_window',
    'raw_height_support',
    'qc_pass',
    'qc_retained_support',
    'sampled8_support',
    'train_only',
    'rho_precision',
    'standard_top8',
)
COHORTS = {
    'low_night_near': (120.0, 200.0, False, 0.0, 0.25),
    'low_night_outer': (120.0, 200.0, False, 0.25, 0.5),
    'peak_night_near': (200.0, 300.0, False, 0.0, 0.25),
    'low_day_near': (120.0, 200.0, True, 0.0, 0.25),
}
SUPPORT_NAMES = ('low_night', 'peak_night', 'low_day')
SUPPORT_INDEX = {'low_night': 0, 'peak_night': 1, 'low_day': 2}
EXPECTED_PRIMARY_PAIRS = {'FY->COSMIC': 359, 'COSMIC->FY': 359}


def _local_time(rows):
    return np.remainder(rows[..., 3] + rows[..., 1] / 15.0, 24.0)


def _support_bits(rows):
    """Return raw/full-profile support for the three audit strata."""
    rows = np.asarray(rows)
    result = np.zeros(3, dtype=bool)
    if rows.ndim != 2 or rows.shape[1] < 5 or not len(rows):
        return result
    valid = np.isfinite(rows[:, :5]).all(axis=1)
    local_time = _local_time(rows)
    day = (local_time >= 6.0) & (local_time < 18.0)
    altitude = rows[:, 2]
    result[0] = np.any(valid & (altitude >= 120.0) & (altitude < 200.0) & ~day)
    result[1] = np.any(valid & (altitude >= 200.0) & (altitude < 300.0) & ~day)
    result[2] = np.any(valid & (altitude >= 120.0) & (altitude < 200.0) & day)
    return result


def _cohort_support_name(cohort):
    low, high, day, _, _ = COHORTS[cohort]
    if low == 200.0:
        return 'peak_night'
    return 'low_day' if day else 'low_night'


def _point_mask(rows, cohort):
    low, high, day_required, _, _ = COHORTS[cohort]
    local_time = _local_time(rows)
    day = (local_time >= 6.0) & (local_time < 18.0)
    return (
        np.isfinite(rows[..., :5]).all(axis=-1)
        & (rows[..., 2] >= low) & (rows[..., 2] < high)
        & (day == day_required)
    )


def _encode_pairs(target_ids, neighbor_ids):
    target = np.asarray(target_ids, dtype=np.uint64)
    neighbor = np.asarray(neighbor_ids, dtype=np.uint64)
    if np.any(target >= 2 ** 32) or np.any(neighbor >= 2 ** 32):
        raise ValueError('profile IDs must fit unsigned 32-bit pair encoding')
    return (target << np.uint64(32)) | neighbor


def _decode_target(keys):
    return (np.asarray(keys, dtype=np.uint64) >> np.uint64(32)).astype(np.int64)


def _decode_neighbor(keys):
    return (
        np.asarray(keys, dtype=np.uint64) & np.uint64(0xffffffff)
    ).astype(np.int64)


def _safe_unique(parts):
    nonempty = [np.asarray(part, dtype=np.uint64) for part in parts if len(part)]
    return np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, np.uint64)


def _date_stratified_bootstrap(rows, statistic, replicates=1000, seed=42):
    rows = np.asarray(rows)
    value = float(statistic(rows))
    if not len(rows):
        return value, None
    dates = np.unique(rows['date'])
    groups = [np.flatnonzero(rows['date'] == date) for date in dates]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = np.concatenate([
            group[rng.integers(0, len(group), len(group))]
            for group in groups
        ])
        samples[index] = statistic(rows[selected])
    return value, np.quantile(samples, [0.025, 0.975]).tolist()


def _rate_difference_rows(low_numerator, low_denominator,
                          control_numerator, control_denominator,
                          target_dates):
    target_ids = np.intersect1d(
        np.unique(_decode_target(low_denominator)),
        np.unique(_decode_target(control_denominator)),
        assume_unique=True)
    dtype = np.dtype([
        ('target_id', '<i8'), ('date', '<i2'),
        ('low_num', '<i4'), ('low_den', '<i4'),
        ('control_num', '<i4'), ('control_den', '<i4'),
    ])
    rows = np.zeros(len(target_ids), dtype=dtype)
    rows['target_id'] = target_ids
    rows['date'] = target_dates(target_ids)
    for keys, field in (
            (low_numerator, 'low_num'), (low_denominator, 'low_den'),
            (control_numerator, 'control_num'),
            (control_denominator, 'control_den')):
        ids, counts = np.unique(_decode_target(keys), return_counts=True)
        positions = np.searchsorted(ids, target_ids)
        present = positions < len(ids)
        clipped = np.minimum(positions, max(len(ids) - 1, 0))
        if len(ids):
            present &= ids[clipped] == target_ids
            rows[field][present] = counts[clipped[present]]
    return rows[(rows['low_den'] > 0) & (rows['control_den'] > 0)]


def _rate_difference(rows):
    low = rows['low_num'].sum() / max(rows['low_den'].sum(), 1)
    control = (
        rows['control_num'].sum() / max(rows['control_den'].sum(), 1))
    return float(low - control)


def _classify(index_ok, qc_result, train_result, top8_ratio, bias_result):
    if not index_ok:
        return '索引实现问题'
    if (
            qc_result['estimable']
            and qc_result['difference'] <= -0.10
            and qc_result['ci95'][1] < -0.10):
        return 'QC额外加剧低层稀疏'
    if (
            train_result['estimable']
            and train_result['difference'] <= -0.05
            and train_result['ci95'][1] < 0.0):
        return 'train划分偶然失衡'
    if top8_ratio < 0.80:
        return '高度无关top-8加剧低层稀疏'
    if bias_result['confirmed'] or bias_result['date_sign_fraction_below_0_80']:
        return '来源偏差或时空代表性问题'
    return '固有低层稀疏，现有流水线未显著额外恶化'


def _qc_metadata(index_path, data_path):
    with np.load(index_path, allow_pickle=False) as loaded:
        metadata = {name: np.asarray(loaded[name]) for name in loaded.files}
    order = np.argsort(metadata['profile_id'], kind='stable')
    metadata = {
        name: values[order] if len(values) == len(order) else values
        for name, values in metadata.items()
    }
    ids = np.asarray(metadata['profile_id'], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'duplicate profile IDs in {index_path}')
    data = np.load(data_path, mmap_mode='r')
    qc_support = np.zeros((len(ids), 3), dtype=bool)
    for position in np.flatnonzero(metadata['pass_profile']):
        start = int(metadata['output_start'][position])
        end = int(metadata['output_end'][position])
        qc_support[position] = _support_bits(data[start:end])
    return metadata, qc_support


def _sampled_support(index, metadata_ids):
    support = np.zeros((len(metadata_ids), 3), dtype=bool)
    positions = np.searchsorted(metadata_ids, index.prof_ids)
    if (
            np.any(positions >= len(metadata_ids))
            or not np.array_equal(metadata_ids[positions], index.prof_ids)):
        raise ValueError('neighborhood profile IDs missing from QC metadata')
    for local, position in enumerate(positions):
        rows = index.prof_abs_data[local][index.prof_valid_mask[local]]
        support[position] = _support_bits(rows)
    return support


def _prepare_cosmic_raw():
    raw = np.load(COSMIC_INPUT, mmap_mode='r')
    profile_ids = np.rint(np.asarray(raw[:, 5])).astype(np.int64)
    order = np.argsort(profile_ids, kind='stable')
    sorted_ids = profile_ids[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_ids)) + 1])
    ends = np.concatenate([starts[1:], [len(order)]])
    return raw, order, sorted_ids[starts], starts, ends


def _ensure_raw_support(state, positions):
    positions = np.unique(np.asarray(positions, dtype=np.int64))
    missing = positions[~state['raw_loaded'][positions]]
    if not len(missing):
        return
    metadata = state['metadata']
    if state['name'] == 'FY':
        for position in missing:
            path = FY_RAW_ROOT / str(
                metadata['original_relative_path'][position]).replace('/', os.sep)
            try:
                raw = _read_fy(path.read_bytes())
                _, relative_hour = _parse_fy_time(path)
                density = raw[:, 3]
                valid_density = np.isfinite(density) & (density > 0.0)
                log_density = np.full(len(density), np.nan, dtype=np.float64)
                log_density[valid_density] = np.log10(density[valid_density])
                rows = np.column_stack([
                    raw[:, 1], raw[:, 0], raw[:, 2],
                    np.full(len(raw), relative_hour),
                    log_density,
                ])
                state['raw_support'][position] = _support_bits(rows)
            except Exception:
                state['raw_support'][position] = False
    else:
        raw, order, raw_ids, starts, ends = state['cosmic_raw']
        ids = metadata['profile_id'][missing]
        raw_positions = np.searchsorted(raw_ids, ids)
        for position, raw_position, profile_id in zip(
                missing, raw_positions, ids):
            if (
                    raw_position >= len(raw_ids)
                    or raw_ids[raw_position] != profile_id):
                continue
            rows = np.asarray(
                raw[order[starts[raw_position]:ends[raw_position]], :5],
                dtype=np.float64)
            state['raw_support'][position] = _support_bits(rows)
    state['raw_loaded'][missing] = True


def _target_queries(cache, cohort, max_profiles=0):
    profile_indices = np.arange(len(cache.ids))
    if max_profiles and len(profile_indices) > max_profiles:
        rng = np.random.default_rng(42)
        profile_indices = np.sort(rng.choice(
            profile_indices, max_profiles, replace=False))
    data = cache.data[profile_indices]
    valid = cache.valid[profile_indices] & _point_mask(data, cohort)
    profile_ids = np.repeat(
        cache.ids[profile_indices], valid.shape[1])[valid.reshape(-1)]
    return {
        'coords': data.reshape(-1, 5)[valid.reshape(-1), :4],
        'residual': cache.residual[profile_indices].reshape(-1)[
            valid.reshape(-1)],
        'profile_id': profile_ids,
    }


def _hard_candidates(coords, state):
    """Yield all profile candidates inside the rectangular hard window."""
    meta = state['meta']
    order = state['meta_time_order']
    sorted_time = state['meta_sorted_time']
    for query, coord in enumerate(coords):
        left = np.searchsorted(sorted_time, coord[3] - 1.5, side='left')
        right = np.searchsorted(sorted_time, coord[3] + 1.5, side='right')
        positions = order[left:right]
        if not len(positions):
            continue
        positions = positions[state['meta_finite'][positions]]
        if not len(positions):
            continue
        dlat = np.abs(meta[positions, 0] - coord[0])
        dlon = np.abs(
            (meta[positions, 1] - coord[1] + 180.0) % 360.0 - 180.0)
        dtime = np.abs(meta[positions, 2] - coord[3])
        keep = (dlat <= 5.0) & (dlon <= 15.0)
        positions = positions[keep]
        if not len(positions):
            continue
        dlat = dlat[keep]
        dlon = dlon[keep]
        dtime = dtime[keep]
        if state['name'] == 'FY':
            selection_distance = (
                dlat / 5.0 + dlon / 15.0 + dtime / 1.5)
        else:
            selection_distance = np.sqrt(
                (dlat / 5.0) ** 2 + (dlon / 15.0) ** 2)
        yield query, positions, selection_distance


def _topk_ids(candidate_ids, distances, eligible, k=8):
    selected = np.flatnonzero(eligible)
    if len(selected) > k:
        selected = selected[np.argsort(
            distances[selected], kind='stable')[:k]]
    return candidate_ids[selected]


def _profile_point_rho(coord, candidate_ids, state, cohort):
    positions = np.searchsorted(state['index'].prof_ids, candidate_ids)
    present = positions < len(state['index'].prof_ids)
    clipped = np.minimum(
        positions, max(len(state['index'].prof_ids) - 1, 0))
    if len(state['index'].prof_ids):
        present &= state['index'].prof_ids[clipped] == candidate_ids
    result = np.zeros(len(candidate_ids), dtype=bool)
    if not np.any(present):
        return result
    data = state['index'].prof_abs_data[clipped[present]]
    valid = state['index'].prof_valid_mask[clipped[present]]
    valid &= _point_mask(data, cohort)
    dlat = np.abs(data[..., 0] - coord[0]) / 5.0
    dlon = np.abs(
        (data[..., 1] - coord[1] + 180.0) % 360.0 - 180.0) / 15.0
    dtime = np.abs(data[..., 3] - coord[3]) / 1.5
    # The production payload uses max norm, not Euclidean norm.
    rho = np.maximum.reduce([dlat, dlon, dtime])
    low, high = COHORTS[cohort][3:]
    result[present] = np.any(
        valid & (rho >= low) & (rho < high) & (rho < 1.0), axis=1)
    return result


def _funnel_for_direction(target, observation, queries, cohort):
    support_column = SUPPORT_INDEX[_cohort_support_name(cohort)]
    stage_parts = {name: [] for name in STAGES}
    stage_parts['qc_any'] = []
    stage_parts['train_any'] = []
    oracle_parts = []
    metadata = observation['metadata']
    ids = metadata['profile_id']
    pass_profile = np.asarray(metadata['pass_profile'], dtype=bool)
    train = np.isin(ids, observation['train_ids'], assume_unique=True)
    same_source = target['name'] == observation['name']
    standard_ids_by_query = np.full(
        (len(queries['coords']), observation['index'].k_prof),
        -1, dtype=np.int64)
    for start in range(0, len(queries['coords']), 512):
        stop = min(start + 512, len(queries['coords']))
        cached = observation['index'].query_profiles_only(
            queries['coords'][start:stop],
            exclude_profile_ids=(
                queries['profile_id'][start:stop] if same_source else None),
            allowed_profile_ids=observation['train_ids'])
        standard_ids_by_query[start:stop] = cached['sel_ids']

    for query, candidate_positions, distances in _hard_candidates(
            queries['coords'], observation):
        candidate_ids = ids[candidate_positions]
        if same_source:
            keep = candidate_ids != queries['profile_id'][query]
            candidate_positions = candidate_positions[keep]
            candidate_ids = candidate_ids[keep]
            distances = distances[keep]
        if not len(candidate_ids):
            continue
        _ensure_raw_support(observation, candidate_positions)
        target_ids = np.full(len(candidate_ids), queries['profile_id'][query])
        keys = _encode_pairs(target_ids, candidate_ids)
        raw_support = observation['raw_support'][
            candidate_positions, support_column]
        passed = raw_support & pass_profile[candidate_positions]
        retained = passed & observation['qc_support'][
            candidate_positions, support_column]
        sampled = retained & observation['sampled_support'][
            candidate_positions, support_column]
        train_support = sampled & train[candidate_positions]
        rho_support = np.zeros(len(candidate_ids), dtype=bool)
        if np.any(train_support):
            rho_support[train_support] = _profile_point_rho(
                queries['coords'][query], candidate_ids[train_support],
                observation, cohort)

        standard_ids = standard_ids_by_query[query]
        standard_ids = standard_ids[standard_ids >= 0]
        standard = rho_support & np.isin(candidate_ids, standard_ids)
        oracle_ids = _topk_ids(
            candidate_ids, distances, train_support)
        oracle = rho_support & np.isin(candidate_ids, oracle_ids)

        masks = {
            'hard_window': np.ones(len(keys), dtype=bool),
            'raw_height_support': raw_support,
            'qc_pass': passed,
            'qc_retained_support': retained,
            'sampled8_support': sampled,
            'train_only': train_support,
            'rho_precision': rho_support,
            'standard_top8': standard,
            'qc_any': pass_profile[candidate_positions],
            'train_any': pass_profile[candidate_positions]
                         & train[candidate_positions],
        }
        for stage, mask in masks.items():
            if np.any(mask):
                stage_parts[stage].append(keys[mask])
        if np.any(oracle):
            oracle_parts.append(keys[oracle])
    result = {stage: _safe_unique(parts) for stage, parts in stage_parts.items()}
    result['oracle_top8'] = _safe_unique(oracle_parts)
    return result


def _stratified_queries(caches, total=1000):
    per_source = total // 2
    rng = np.random.default_rng(42)
    result = {}
    for source in SOURCE_NAMES:
        cache = caches[source]
        valid = cache.valid.reshape(-1)
        coords = cache.data.reshape(-1, 5)[valid, :4]
        profile_ids = np.repeat(
            cache.ids, cache.valid.shape[1])[valid]
        date = np.floor(coords[:, 3] / 24.0).astype(np.int16)
        latitude = np.floor((coords[:, 0] + 90.0) / 30.0).astype(np.int8)
        strata = date.astype(np.int32) * 16 + latitude
        chosen = []
        for stratum in np.unique(strata):
            indices = np.flatnonzero(strata == stratum)
            chosen.append(int(rng.choice(indices)))
        chosen = np.unique(chosen)
        if len(chosen) < per_source:
            pool = np.setdiff1d(
                np.arange(len(coords)), chosen, assume_unique=False)
            chosen = np.concatenate([
                chosen, rng.choice(
                    pool, per_source - len(chosen), replace=False)])
        else:
            chosen = rng.choice(chosen, per_source, replace=False)
        result[source] = {
            'coords': coords[chosen],
            'profile_id': profile_ids[chosen],
        }
    return result


def _direct_index_audit(states, caches, query_count=1000):
    sampled = _stratified_queries(caches, query_count)
    mismatches = []
    checks = 0
    for target_source in SOURCE_NAMES:
        queries = sampled[target_source]
        for observation_source in SOURCE_NAMES:
            observation = states[observation_source]
            index = observation['index']
            same_source = target_source == observation_source
            for start in range(0, len(queries['coords']), 128):
                coords = queries['coords'][start:start + 128]
                exclude = (
                    queries['profile_id'][start:start + 128]
                    if same_source else None)
                actual = index.query_profiles_only(
                    coords, exclude_profile_ids=exclude,
                    allowed_profile_ids=observation['train_ids'])
                for local, coord in enumerate(coords):
                    meta = index.prof_sorted_meta
                    ids = index.prof_sorted_ids
                    dlat = np.abs(meta[:, 0] - coord[0])
                    dlon = np.abs(
                        (meta[:, 1] - coord[1] + 180.0) % 360.0 - 180.0)
                    dtime = np.abs(meta[:, 2] - coord[3])
                    valid = (
                        (dlat <= index.dlat) & (dlon <= index.dlon)
                        & (dtime <= index.dt)
                        & np.isin(ids, observation['train_ids'],
                                  assume_unique=True))
                    if same_source:
                        valid &= ids != exclude[local]
                    distances = _normalized_profile_distance(
                        dlat, dlon, dtime,
                        index.dlat, index.dlon, index.dt)
                    expected = np.sort(distances[valid])[:index.k_prof]
                    observed = np.sort(
                        actual['sel_distance'][local][
                            actual['valid_prof'][local]])
                    checks += 1
                    if (
                            len(expected) != len(observed)
                            or not np.allclose(
                                expected, observed, atol=1e-7, rtol=0.0)):
                        if len(mismatches) < 20:
                            mismatches.append({
                                'target_source': target_source,
                                'observation_source': observation_source,
                                'profile_id': int(
                                    queries['profile_id'][start + local]),
                                'expected': expected.tolist(),
                                'observed': observed.tolist(),
                            })
    return {
        'queries': query_count,
        'source_direction_checks': checks,
        'mismatch_count': len(mismatches),
        'examples': mismatches,
        'passed': not mismatches,
    }


def _target_dates(cache):
    def lookup(profile_ids):
        positions = np.searchsorted(cache.ids, profile_ids)
        if (
                np.any(positions >= len(cache.ids))
                or not np.array_equal(cache.ids[positions], profile_ids)):
            raise ValueError('target profile missing from cache')
        return cache.date[positions]
    return lookup


def _retention_report(rows, threshold, replicates):
    if not len(rows):
        return {
            'profiles': 0, 'difference': 0.0, 'ci95': None,
            'estimable': False,
        }
    value, ci = _date_stratified_bootstrap(
        rows, _rate_difference, replicates=replicates, seed=42)
    return {
        'profiles': int(len(rows)),
        'difference': value,
        'ci95': ci,
        'threshold': threshold,
        'estimable': (
            len(rows) >= 200 and len(np.unique(rows['date'])) >= 15),
    }


def _residual_profile_summary(cache, cohort):
    mask = cache.valid & _point_mask(cache.data, cohort)
    rows = []
    for index in np.flatnonzero(mask.any(axis=1)):
        values = cache.residual[index][mask[index]]
        rows.append((cache.ids[index], cache.date[index], float(values.mean())))
    dtype = np.dtype([
        ('profile_id', '<i8'), ('date', '<i2'), ('residual', '<f8')])
    result = np.asarray(rows, dtype=dtype)
    if not len(result):
        return {}, result
    values = result['residual']
    median = float(np.median(values))
    return {
        'profiles': int(len(result)),
        'mean': float(np.mean(values)),
        'median': median,
        'mad': float(1.4826 * np.median(np.abs(values - median))),
        'p05': float(np.quantile(values, 0.05)),
        'p95': float(np.quantile(values, 0.95)),
        'positive_fraction': float(np.mean(values > 0.0)),
    }, result


def _matched_residual_pairs(states, caches, cohort='low_night_near'):
    parts = []
    for target_source, observation_source in (
            ('FY', 'COSMIC'), ('COSMIC', 'FY')):
        target = caches[target_source]
        observation = caches[observation_source]
        queries = _target_queries(target, cohort)
        for start in range(0, len(queries['coords']), 256):
            coords = queries['coords'][start:start + 256]
            cached = states[observation_source]['index'].query_profiles_only(
                coords,
                allowed_profile_ids=states[observation_source]['train_ids'])
            neighbor_residual, present = _lookup_neighbor_residual(
                observation, cached['sel_ids'])
            data = cached['sel_abs']
            valid = (
                cached['valid_prof'][..., None] & cached['sel_vmask']
                & present[..., None] & _point_mask(data, cohort))
            target_alt = coords[:, None, None, 2]
            valid &= np.abs(data[..., 2] - target_alt) <= 10.0
            dlat = np.abs(
                data[..., 0] - coords[:, None, None, 0]) / 5.0
            dlon = np.abs(
                (data[..., 1] - coords[:, None, None, 1] + 180.0)
                % 360.0 - 180.0) / 15.0
            dtime = np.abs(
                data[..., 3] - coords[:, None, None, 3]) / 1.5
            rho = np.maximum.reduce([dlat, dlon, dtime])
            valid &= rho < 0.25
            if not np.any(valid):
                continue
            localization = 1.0 - 3.0 * rho ** 2 + 2.0 * rho ** 3
            target_residual = queries['residual'][
                start:start + len(coords), None, None]
            difference = (
                target_residual - neighbor_residual
                if target_source == 'FY'
                else neighbor_residual - target_residual)
            target_ids = queries['profile_id'][
                start:start + len(coords), None, None]
            neighbor_ids = cached['sel_ids'][..., None]
            fy_ids = (
                np.broadcast_to(target_ids, valid.shape)
                if target_source == 'FY'
                else np.broadcast_to(neighbor_ids, valid.shape))
            cosmic_ids = (
                np.broadcast_to(neighbor_ids, valid.shape)
                if target_source == 'FY'
                else np.broadcast_to(target_ids, valid.shape))
            parts.append(np.column_stack([
                fy_ids[valid], cosmic_ids[valid],
                difference[valid], localization[valid],
            ]))
    if not parts:
        return np.empty((0, 4), dtype=np.float64)
    tokens = np.concatenate(parts)
    keys = _encode_pairs(tokens[:, 0].astype(np.int64),
                         tokens[:, 1].astype(np.int64))
    unique, inverse = np.unique(keys, return_inverse=True)
    weights = np.bincount(inverse, weights=tokens[:, 3])
    difference = np.bincount(
        inverse, weights=tokens[:, 2] * tokens[:, 3]) / weights
    return np.column_stack([
        _decode_target(unique),
        (unique & np.uint64(0xffffffff)).astype(np.int64),
        difference,
        weights,
    ])


def _bias_report(pairs, fy_cache, replicates):
    dtype = np.dtype([
        ('profile_id', '<i8'), ('date', '<i2'), ('difference', '<f8')])
    if not len(pairs):
        return {
            'unique_profile_pairs': 0, 'fy_target_profiles': 0,
            'effective_dates': 0, 'confirmed': False,
            'date_sign_fraction_below_0_80': False,
        }, np.empty(0, dtype=dtype)
    fy_ids = pairs[:, 0].astype(np.int64)
    unique_fy, inverse = np.unique(fy_ids, return_inverse=True)
    profile_difference = np.bincount(
        inverse, weights=pairs[:, 2]) / np.bincount(inverse)
    dates = _target_dates(fy_cache)(unique_fy)
    rows = np.empty(len(unique_fy), dtype=dtype)
    rows['profile_id'] = unique_fy
    rows['date'] = dates
    rows['difference'] = profile_difference

    def median_stat(selected):
        return float(np.median(selected['difference']))

    median, ci = _date_stratified_bootstrap(
        rows, median_stat, replicates=replicates, seed=42)
    eligible_signs = []
    full_sign = np.sign(median)
    for date in np.unique(rows['date']):
        values = rows['difference'][rows['date'] == date]
        if len(values) >= 2 and np.median(values) != 0.0 and full_sign:
            eligible_signs.append(np.sign(np.median(values)) == full_sign)
    sign_fraction = (
        float(np.mean(eligible_signs)) if eligible_signs else None)
    estimable = len(pairs) >= 200 and len(eligible_signs) >= 15
    confirmed = bool(
        estimable and abs(median) >= 0.05
        and ci[0] * ci[1] > 0.0)
    return {
        'unique_profile_pairs': int(len(pairs)),
        'fy_target_profiles': int(len(rows)),
        'effective_dates': len(eligible_signs),
        'mean_difference': float(np.mean(rows['difference'])),
        'median_difference': median,
        'mad_difference': float(
            1.4826 * np.median(np.abs(rows['difference'] - median))),
        'p05': float(np.quantile(rows['difference'], 0.05)),
        'p95': float(np.quantile(rows['difference'], 0.95)),
        'median_ci95': ci,
        'date_sign_fraction': sign_fraction,
        'estimable': estimable,
        'confirmed': confirmed,
        'date_sign_fraction_below_0_80': bool(
            estimable and sign_fraction < 0.80),
    }, rows


def _cell_rows(results, states, caches):
    rows = []
    for pair in PAIR_NAMES:
        target_source = pair.split('->')[0]
        cache = caches[target_source]
        for cohort in COHORTS:
            for stage in STAGES + ('oracle_top8',):
                keys = results[pair][cohort][stage]
                if not len(keys):
                    continue
                target_ids = _decode_target(keys)
                unique, counts = np.unique(target_ids, return_counts=True)
                positions = np.searchsorted(cache.ids, unique)
                dates = cache.date[positions]
                data = cache.data[positions]
                valid = cache.valid[positions]
                lat = np.nanmean(
                    np.where(valid, data[..., 0], np.nan), axis=1)
                lon = np.nanmean(
                    np.where(valid, data[..., 1], np.nan), axis=1)
                lat_bin = np.floor((lat + 90.0) / 10.0).astype(np.int16)
                lon_bin = np.floor(
                    np.remainder(lon + 180.0, 360.0) / 30.0).astype(np.int16)
                for key in np.unique(
                        np.column_stack([dates, lat_bin, lon_bin]), axis=0):
                    chosen = (
                        (dates == key[0]) & (lat_bin == key[1])
                        & (lon_bin == key[2]))
                    rows.append((
                        pair, cohort, stage, int(key[0]), int(key[1]),
                        int(key[2]), int(chosen.sum()),
                        int(counts[chosen].sum())))
    dtype = np.dtype([
        ('pair', '<U16'), ('cohort', '<U24'), ('stage', '<U24'),
        ('date', '<i2'), ('latitude_bin', '<i2'), ('longitude_bin', '<i2'),
        ('target_profiles', '<i4'), ('ordered_pairs', '<i8'),
    ])
    return np.asarray(rows, dtype=dtype)


def _concentration(keys, cache):
    if not len(keys):
        return {
            'ordered_pairs': 0, 'target_profiles': 0,
            'kish_effective_dates': 0.0,
            'kish_effective_regions': 0.0,
            'top3_date_fraction': 0.0,
            'top3_region_fraction': 0.0,
        }
    target_ids, target_pair_counts = np.unique(
        _decode_target(keys), return_counts=True)
    positions = np.searchsorted(cache.ids, target_ids)
    dates = cache.date[positions]
    data = cache.data[positions]
    valid = cache.valid[positions]
    latitude = np.nanmean(
        np.where(valid, data[..., 0], np.nan), axis=1)
    longitude = np.nanmean(
        np.where(valid, data[..., 1], np.nan), axis=1)
    region = (
        np.floor((latitude + 90.0) / 10.0).astype(np.int32) * 32
        + np.floor(
            np.remainder(longitude + 180.0, 360.0) / 30.0).astype(np.int32))

    def grouped(values):
        _, inverse = np.unique(values, return_inverse=True)
        return np.bincount(inverse, weights=target_pair_counts)

    date_mass = grouped(dates)
    region_mass = grouped(region)

    def kish(mass):
        return float(mass.sum() ** 2 / np.sum(mass ** 2))

    def top3(mass):
        return float(
            np.sort(mass)[-3:].sum() / max(mass.sum(), 1.0))

    return {
        'ordered_pairs': int(len(keys)),
        'target_profiles': int(len(target_ids)),
        'kish_effective_dates': kish(date_mass),
        'kish_effective_regions': kish(region_mass),
        'top3_date_fraction': top3(date_mass),
        'top3_region_fraction': top3(region_mass),
    }


def _attrition_detail(stage, observation_state, cohort):
    metadata = observation_state['metadata']
    ids = np.asarray(metadata['profile_id'], dtype=np.int64)
    raw = stage['raw_height_support']
    passed = stage['qc_pass']
    retained = stage['qc_retained_support']
    sampled = stage['sampled8_support']
    failed_pairs = np.setdiff1d(raw, passed, assume_unique=True)
    retained_loss = np.setdiff1d(passed, retained, assume_unique=True)
    sampled_loss = np.setdiff1d(retained, sampled, assume_unique=True)

    def positions(keys):
        neighbor = _decode_neighbor(keys)
        result = np.searchsorted(ids, neighbor)
        if (
                np.any(result >= len(ids))
                or not np.array_equal(ids[result], neighbor)):
            raise ValueError('attrition profile missing from metadata')
        return result

    failed_positions = positions(failed_pairs) if len(failed_pairs) else np.empty(
        0, dtype=np.int64)
    reason_counts = {}
    reason_names = metadata.get('reason_names', np.empty(0, dtype='U1'))
    reason_masks = metadata.get('reason_masks', np.empty(0, dtype=np.int64))
    for name, mask in zip(reason_names, reason_masks):
        reason_counts[str(name)] = int(np.count_nonzero(
            metadata['reason_bits'][failed_positions] & int(mask)))

    retained_positions = (
        positions(retained_loss) if len(retained_loss)
        else np.empty(0, dtype=np.int64))
    h_cut = np.asarray(metadata['h_cut_km'][retained_positions], dtype=float)
    band_high = COHORTS[cohort][1]
    return {
        'raw_to_qc_pass_rate': len(passed) / max(len(raw), 1),
        'qc_pass_to_retained_rate': len(retained) / max(len(passed), 1),
        'retained_to_sampled8_rate': len(sampled) / max(len(retained), 1),
        'failed_qc_ordered_pairs': int(len(failed_pairs)),
        'failed_qc_reason_pair_counts': reason_counts,
        'retained_height_loss_ordered_pairs': int(len(retained_loss)),
        'retained_height_loss_unique_profiles': int(
            len(np.unique(_decode_neighbor(retained_loss)))),
        'h_cut_at_or_above_band_high_pair_fraction': (
            float(np.mean(h_cut >= band_high)) if len(h_cut) else 0.0),
        'h_cut_median_km': (
            float(np.nanmedian(h_cut)) if len(h_cut) else None),
        'sampled8_height_loss_ordered_pairs': int(len(sampled_loss)),
    }


def _write_outputs(output, report, arrays):
    output.mkdir(parents=True, exist_ok=False)
    npz_path = output / 'low_night_audit_cells.npz'
    json_path = output / 'low_night_audit_report.json'
    _deterministic_npz(npz_path, arrays)
    report['npz_sha256'] = _sha256(npz_path)
    temporary = json_path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False) + '\n',
        encoding='utf-8')
    os.replace(temporary, json_path)
    return {
        'npz': report['npz_sha256'],
        'json': _sha256(json_path),
    }


def run(output, checkpoint=BACKGROUND_DEFAULT, bootstrap=1000,
        direct_queries=1000, max_target_profiles=0):
    if bootstrap < 1 or direct_queries < 2 or max_target_profiles < 0:
        raise ValueError('invalid audit sampling arguments')
    np.random.seed(42)
    torch.manual_seed(42)
    config = _configuration()
    device = torch.device('cpu')
    model, sw_manager, iri_peak_manager = _strict_background(
        config, checkpoint, device)
    loader_kwargs = {
        'batch_size': min(int(config['batch_size']), 512),
        'bin_size_hours': config['bin_size_hours'],
        'num_workers': 0,
        'use_memmap': True,
        'val_ratio': 0.1,
        'split_seed': 42,
        'points_per_profile': 8,
    }
    fy_train, fy_validation = get_dataloaders(
        config['fy_path'], profile_path=None,
        profile_index_path=config['fy_profile_index_path'], **loader_kwargs)
    cosmic_train, cosmic_validation = get_cosmic_dataloader(
        config['cosmic_path'],
        profile_index_path=config['cosmic_profile_index_path'],
        **loader_kwargs)
    train_ids = {
        'FY': np.unique(fy_train.dataset.profile_ids),
        'COSMIC': np.unique(cosmic_train.dataset.profile_ids),
    }
    validation_ids = {
        'FY': np.unique(fy_validation.dataset.profile_ids),
        'COSMIC': np.unique(cosmic_validation.dataset.profile_ids),
    }
    counts = {source: len(ids) for source, ids in train_ids.items()}
    if counts != EXPECTED_TRAIN_PROFILES:
        raise RuntimeError(
            f'train profile identity changed: {counts} != '
            f'{EXPECTED_TRAIN_PROFILES}')
    for source in SOURCE_NAMES:
        if np.intersect1d(
                train_ids[source], validation_ids[source],
                assume_unique=True).size:
            raise AssertionError(f'{source} train/validation overlap')

    indices = {
        'FY': FYNeighborhoodIndex(config['fy_path'], config),
        'COSMIC': COSMICNeighborhoodIndex(config['cosmic_path'], config),
    }
    metadata = {}
    qc_support = {}
    metadata['FY'], qc_support['FY'] = _qc_metadata(FY_INDEX_PATH, FY_PATH)
    metadata['COSMIC'], qc_support['COSMIC'] = _qc_metadata(
        COSMIC_INDEX_PATH, COSMIC_PATH)
    states = {}
    cosmic_raw = _prepare_cosmic_raw()
    for source in SOURCE_NAMES:
        ids = np.asarray(metadata[source]['profile_id'], dtype=np.int64)
        meta = np.column_stack([
            metadata[source]['representative_lat'],
            metadata[source]['representative_lon'],
            metadata[source]['representative_time'],
        ]).astype(np.float64)
        index_positions = np.searchsorted(ids, indices[source].prof_ids)
        if (
                np.any(index_positions >= len(ids))
                or not np.array_equal(
                    ids[index_positions], indices[source].prof_ids)):
            raise ValueError(f'{source} index profile missing from metadata')
        # Passing-profile metadata must match the production neighborhood index.
        meta[index_positions] = indices[source].prof_meta
        finite = np.isfinite(meta).all(axis=1)
        time_order = np.flatnonzero(finite)
        time_order = time_order[
            np.argsort(meta[time_order, 2], kind='stable')]
        states[source] = {
            'name': source,
            'metadata': metadata[source],
            'meta': meta,
            'meta_finite': finite,
            'meta_time_order': time_order,
            'meta_sorted_time': meta[time_order, 2],
            'qc_support': qc_support[source],
            'sampled_support': _sampled_support(indices[source], ids),
            'raw_support': np.zeros((len(ids), 3), dtype=bool),
            'raw_loaded': np.zeros(len(ids), dtype=bool),
            'train_ids': train_ids[source],
            'validation_ids': validation_ids[source],
            'index': indices[source],
            'cosmic_raw': cosmic_raw if source == 'COSMIC' else None,
        }

    caches = {}
    for source in SOURCE_NAMES:
        print(f'[{source}] 计算train-only背景残差缓存...', flush=True)
        caches[source] = _background_residual_cache(
            source, indices[source], train_ids[source], model,
            sw_manager, iri_peak_manager, device)

    index_audit = _direct_index_audit(
        states, caches, query_count=direct_queries)
    if not index_audit['passed']:
        raise AssertionError(
            f'direct top-K audit failed: {index_audit["examples"][:1]}')

    results = {pair: {} for pair in PAIR_NAMES}
    for pair in PAIR_NAMES:
        target_source, observation_source = pair.split('->')
        for cohort in COHORTS:
            queries = _target_queries(
                caches[target_source], cohort,
                max_profiles=max_target_profiles)
            print(
                f'[{pair}/{cohort}] 查询点={len(queries["coords"]):,}',
                flush=True)
            results[pair][cohort] = _funnel_for_direction(
                states[target_source], states[observation_source],
                queries, cohort)

    primary_counts = {
        pair: len(results[pair]['low_night_near']['standard_top8'])
        for pair in ('FY->COSMIC', 'COSMIC->FY')
    }
    if not max_target_profiles and primary_counts != EXPECTED_PRIMARY_PAIRS:
        raise AssertionError(
            f'primary pair identity changed: {primary_counts} != '
            f'{EXPECTED_PRIMARY_PAIRS}')

    qc_reports = {}
    train_reports = {}
    top8_reports = {}
    for pair in PAIR_NAMES:
        target_source = pair.split('->')[0]
        low = results[pair]['low_night_near']
        peak = results[pair]['peak_night_near']
        qc_rows = _rate_difference_rows(
            low['qc_retained_support'], low['raw_height_support'],
            peak['qc_retained_support'], peak['raw_height_support'],
            _target_dates(caches[target_source]))
        qc_reports[pair] = _retention_report(qc_rows, -0.10, bootstrap)
        train_rows = _rate_difference_rows(
            low['train_only'], low['sampled8_support'],
            low['train_any'], low['qc_any'],
            _target_dates(caches[target_source]))
        train_reports[pair] = _retention_report(
            train_rows, -0.05, bootstrap)
        standard_count = len(low['standard_top8'])
        oracle_count = len(low['oracle_top8'])
        top8_reports[pair] = {
            'standard_pairs': standard_count,
            'oracle_pairs': oracle_count,
            'standard_to_oracle_ratio': (
                standard_count / oracle_count if oracle_count else 1.0),
        }

    residual_summaries = {}
    residual_arrays = {}
    for source in SOURCE_NAMES:
        residual_summaries[source], residual_arrays[source] = (
            _residual_profile_summary(
                caches[source], 'low_night_near'))
    matched_pairs = _matched_residual_pairs(states, caches)
    bias, bias_rows = _bias_report(matched_pairs, caches['FY'], bootstrap)

    qc_flag = next(
        (qc_reports[pair] for pair in ('FY->COSMIC', 'COSMIC->FY')
         for value in (qc_reports[pair],)
         if value['estimable'] and value['difference'] <= -0.10
         and value['ci95'][1] < -0.10),
        {'estimable': False, 'difference': 0.0, 'ci95': None})
    train_flag = next(
        (train_reports[pair] for pair in ('FY->COSMIC', 'COSMIC->FY')
         for value in (train_reports[pair],)
         if value['estimable'] and value['difference'] <= -0.05
         and value['ci95'][1] < 0.0),
        {'estimable': False, 'difference': 0.0, 'ci95': None})
    minimum_top8_ratio = min(
        top8_reports[pair]['standard_to_oracle_ratio']
        for pair in ('FY->COSMIC', 'COSMIC->FY'))
    conclusion = _classify(
        index_audit['passed'], qc_flag, train_flag,
        minimum_top8_ratio, bias)

    cells = _cell_rows(results, states, caches)
    stage_counts = {
        pair: {
            cohort: {
                stage: int(len(results[pair][cohort][stage]))
                for stage in STAGES + ('oracle_top8',)
            }
            for cohort in COHORTS
        }
        for pair in PAIR_NAMES
    }
    attrition = {
        pair: {
            cohort: _attrition_detail(
                results[pair][cohort],
                states[pair.split('->')[1]], cohort)
            for cohort in COHORTS
        }
        for pair in PAIR_NAMES
    }
    concentration = {
        pair: {
            cohort: {
                stage: _concentration(
                    results[pair][cohort][stage],
                    caches[pair.split('->')[0]])
                for stage in ('raw_height_support', 'rho_precision',
                              'standard_top8')
            }
            for cohort in COHORTS
        }
        for pair in PAIR_NAMES
    }
    report = {
        'schema_version': 1,
        'seed': 42,
        'mode': 'full' if not max_target_profiles else 'smoke',
        'bootstrap_replicates': bootstrap,
        'direct_query_count': direct_queries,
        'max_target_profiles_per_source': max_target_profiles,
        'assumption': (
            'FY/COSMIC 120-200 km sparsity is intrinsic and is not a failure.'),
        'semantics': {
            'hard_window': {
                'hours': 1.5, 'latitude_degrees': 5.0,
                'longitude_degrees': 15.0,
            },
            'top8_distance': {
                'FY': 'normalized_L1_lat_lon_time',
                'COSMIC': (
                    'normalized_horizontal_L2; time_is_hard_window_only'),
            },
            'localization_rho': 'normalized_max_norm_observation_point',
            'points_per_profile': 8,
            'density_units': 'log10Ne',
        },
        'train_profile_counts': counts,
        'validation_profile_counts': {
            source: len(ids) for source, ids in validation_ids.items()},
        'train_validation_overlap': {'FY': 0, 'COSMIC': 0},
        'index_audit': index_audit,
        'stage_counts': stage_counts,
        'attrition_detail': attrition,
        'date_region_concentration': concentration,
        'qc_retention_vs_peak': qc_reports,
        'train_retention_vs_all_qc': train_reports,
        'top8_vs_height_oracle': top8_reports,
        'low_night_background_residual': residual_summaries,
        'cross_source_residual_bias': bias,
        'decision': {
            'conclusion': conclusion,
            'priority': [
                'index', 'qc', 'train_split', 'top8',
                'source_bias_or_representativeness', 'no_extra_loss',
            ],
        },
        'previous_experiment': {
            'transposed_sign_agreement': 0.9761327608225723,
            'joint_stable_mass_fraction': 0.3579605172363716,
            'joint_stable_cell_fraction': 0.25842696629213485,
            'interpretation': (
                'Cross-source signs largely agree, but stable support is sparse.'),
        },
        'input_identity': {
            'fy_data': _file_identity(FY_PATH),
            'fy_index': _file_identity(FY_INDEX_PATH),
            'fy_qc_report': _file_identity(FY_REPORT_PATH),
            'cosmic_data': _file_identity(COSMIC_PATH),
            'cosmic_index': _file_identity(COSMIC_INDEX_PATH),
            'cosmic_qc_report': _file_identity(COSMIC_REPORT_PATH),
            'cosmic_raw': _file_identity(COSMIC_INPUT),
            'background_checkpoint': _file_identity(checkpoint),
        },
        'code_identity': _code_identity(ROOT),
        'limitations': [
            'Background residual mixes background and observation error.',
            'Neighbor-profile reuse remains a secondary dependence.',
            'Height-aware top-8 is diagnostic only and never enters inference.',
        ],
    }
    arrays = {
        'cell_pair': cells['pair'],
        'cell_cohort': cells['cohort'],
        'cell_stage': cells['stage'],
        'cell_date': cells['date'],
        'cell_latitude_bin': cells['latitude_bin'],
        'cell_longitude_bin': cells['longitude_bin'],
        'cell_target_profiles': cells['target_profiles'],
        'cell_ordered_pairs': cells['ordered_pairs'],
        'fy_residual_profile_id': residual_arrays['FY']['profile_id'],
        'fy_residual_date': residual_arrays['FY']['date'],
        'fy_residual': residual_arrays['FY']['residual'],
        'cosmic_residual_profile_id': residual_arrays['COSMIC']['profile_id'],
        'cosmic_residual_date': residual_arrays['COSMIC']['date'],
        'cosmic_residual': residual_arrays['COSMIC']['residual'],
        'bias_profile_id': bias_rows['profile_id'],
        'bias_date': bias_rows['date'],
        'bias_difference': bias_rows['difference'],
    }
    hashes = _write_outputs(Path(output), report, arrays)
    print(json.dumps({
        'output': str(Path(output).resolve()),
        'hashes': hashes,
        'conclusion': conclusion,
        'primary_pairs': primary_counts,
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        '--background-checkpoint', type=Path, default=BACKGROUND_DEFAULT)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--direct-queries', type=int, default=1000)
    parser.add_argument('--max-target-profiles-per-source', type=int, default=0)
    args = parser.parse_args()
    run(
        args.output.resolve(), args.background_checkpoint.resolve(),
        bootstrap=args.bootstrap, direct_queries=args.direct_queries,
        max_target_profiles=args.max_target_profiles_per_source)


if __name__ == '__main__':
    main()
