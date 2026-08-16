"""Strict, read-only verifier for the M2-W2 P0-A ISR/GIRO QA-v2 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from isr_evaluation.peak_qa import PEAK_STATUSES


class ContractError(RuntimeError):
    """An evaluation artifact does not meet the frozen P0-A contract."""


_PEAK_EXPECTED = {
    'lower_km': 200.0,
    'upper_km': 500.0,
    'coarse_step_km': 10.0,
    'fine_step_km': 1.0,
    'fine_half_window_km': 10.0,
    'min_finite_levels': 5,
    'max_local_gap_km': 20.0,
    'flank_support_km': 30.0,
    'prominence_dex': 0.03,
    'secondary_separation_km': 30.0,
    'near_tie_dex': 0.03,
    'boundary_margin_km': 10.0,
}
_LOW_ALTITUDE_STATUSES = {
    'computed', 'not_applicable', 'insufficient_data',
}
_STRATIFIED_SOURCES = {'analysis': 'M11_log10',
                       'background': 'M00_log10',
                       'iri': 'IRI_log10'}
_STRATIFIED_BINS = ((120.0, 300.0), (300.0, 500.0))
_FULL_ALTITUDE_RANGE = (120.0, 500.0)
_STRATIFIED_PERIODS = ('day', 'night', 'all')
_FULL_ALTITUDE_ALIASES = tuple(f'all_alt_{period}'
                               for period in _STRATIFIED_PERIODS)
_STRATIFIED_METRICS = ('n', 'rmse', 'bias', 'mae', 'pearson_r', 'ccc')
_EXPECTED_STRATIFIED_CONTRACT = {
    'altitude_bins_km': [[120.0, 300.0], [300.0, 500.0]],
    'interval_semantics': 'left_closed_right_open_except_final_right_closed',
    'full_altitude_range_km': [120.0, 500.0],
    'full_altitude_interval_semantics': 'closed',
    'full_altitude_aliases': list(_FULL_ALTITUDE_ALIASES),
    'full_altitude_alias_semantics': (
        'exact_aliases_of_closed_full_altitude_range'),
    'day_local_time_range_hours': [6.0, 18.0],
    'day_interval_semantics': 'left_closed_right_open',
    'periods': ['day', 'night', 'all'],
    'metrics': list(_STRATIFIED_METRICS),
    'source_labels': {
        'analysis': 'M11', 'background': 'M00', 'iri': 'Raw IRI'},
    'low_altitude_diagnostic': {
        'altitude_range_km': [120.0, 200.0],
        'interval_semantics': 'left_closed_right_open',
        'independent_from_stratified_metrics': True,
    },
    'interpretation_warning': (
        '120-300 km mixes 120-200 km without satellite-density targets '
        'and 200-300 km with satellite-density targets; do not treat '
        'the stratum as single-mechanism evidence.'),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path):
    def reject_constant(value):
        raise ContractError(f'{path}: non-finite JSON literal {value!r}')

    try:
        return json.loads(path.read_text(encoding='utf-8'),
                          parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f'{path}: unreadable JSON: {exc}') from exc


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _finite(value, context):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f'{context}: expected finite number')
    _require(math.isfinite(float(value)), f'{context}: non-finite value')


def _validate_contract_common(contract, expected_candidate, expected_baselines,
                              expected_date_split, label):
    _require(contract.get('evaluation_schema_version') == 2,
             f'{label}: evaluation_schema_version must be 2')
    _require(contract.get('token_partitions') == ['train', 'development'],
             f'{label}: token partitions must be train/development only')
    candidate = contract.get('candidate_checkpoint')
    _require(isinstance(candidate, dict), f'{label}: missing candidate checkpoint')
    _require(candidate.get('sha256') == expected_candidate,
             f'{label}: candidate SHA mismatch')
    candidate_split = candidate.get('date_split', {})
    _require(candidate_split.get('sha256') == expected_date_split,
             f'{label}: candidate date-split SHA mismatch')
    baselines = contract.get('baseline_checkpoints')
    _require(isinstance(baselines, list), f'{label}: missing baseline_checkpoints')
    observed = [(item.get('label'), item.get('sha256')) for item in baselines]
    _require(observed == expected_baselines,
             f'{label}: baseline label/SHA sequence mismatch: {observed!r}')
    for item in baselines:
        _require(item.get('date_split', {}).get('sha256') == expected_date_split,
                 f'{label}: baseline date-split SHA mismatch for {item.get("label")}')

    peak = contract.get('peak_search')
    _require(isinstance(peak, dict), f'{label}: missing peak_search contract')
    _require(peak.get('alt_range_km') == [200.0, 500.0],
             f'{label}: peak search range must be 200--500 km')
    for key, expected in _PEAK_EXPECTED.items():
        _require(peak.get(key) == expected,
                 f'{label}: peak_search.{key} mismatch')


def _load_npz(path: Path):
    _require(path.is_file(), f'missing cache: {path}')
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ContractError(f'{path}: unreadable NPZ: {exc}') from exc


def _assert_parallel_arrays(cache, label):
    lengths = {key: int(np.asarray(cache[key]).shape[0]) for key in cache.files}
    _require(lengths, f'{label}: empty cache')
    _require(len(set(lengths.values())) == 1,
             f'{label}: cache arrays have inconsistent lengths {lengths}')
    return next(iter(lengths.values()))


def _array_equal(reference, candidate):
    if reference.dtype.kind in 'fc' or candidate.dtype.kind in 'fc':
        return np.array_equal(reference, candidate, equal_nan=True)
    return np.array_equal(reference, candidate)


def _expected_stratified_keys():
    return {
        f'{source}_alt_{lower:g}-{upper:g}km_{period}'
        for source in _STRATIFIED_SOURCES
        for lower, upper in (*_STRATIFIED_BINS, _FULL_ALTITUDE_RANGE)
        for period in _STRATIFIED_PERIODS
    } | {
        f'{source}_{alias}'
        for source in _STRATIFIED_SOURCES
        for alias in _FULL_ALTITUDE_ALIASES
    }


def _read_stratified_csv(path: Path):
    expected_header = [
        'key', 'source', 'layer', 'n', 'rmse', 'bias', 'mae',
        'pearson_r', 'ccc']
    try:
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames == expected_header,
                     f'ISR: unexpected stratified CSV header in {path}')
            rows = {}
            for row in reader:
                key = row['key']
                _require(key not in rows,
                         f'ISR: duplicate stratified CSV key {key}')
                rows[key] = {
                    'source': row['source'], 'layer': row['layer'],
                    'n': int(row['n']),
                    **{name: float(row[name]) for name in
                       _STRATIFIED_METRICS if name != 'n'},
                }
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f'ISR: unreadable stratified CSV {path}: {exc}') from exc
    return rows


def _metric_values(observation, prediction):
    finite = np.isfinite(observation) & np.isfinite(prediction)
    observation, prediction = observation[finite], prediction[finite]
    error = prediction - observation
    _require(error.size >= 5, 'ISR: stratified cache has fewer than five values')
    obs_mean, pred_mean = observation.mean(), prediction.mean()
    covariance = np.mean(
        (observation - obs_mean) * (prediction - pred_mean))
    denominator = np.sqrt(observation.var() * prediction.var())
    ccc_denominator = (observation.var() + prediction.var()
                       + (obs_mean - pred_mean) ** 2)
    return {
        'n': int(error.size),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'bias': float(np.mean(error)),
        'mae': float(np.mean(np.abs(error))),
        'pearson_r': float(covariance / denominator) if denominator > 1e-30 else None,
        'ccc': float(2.0 * covariance / ccc_denominator)
        if ccc_denominator > 1e-30 else None,
    }


def _compare_metrics(expected, observed, context):
    _require(set(observed) == set(_STRATIFIED_METRICS),
             f'{context}: metric fields mismatch')
    for name in _STRATIFIED_METRICS:
        _finite(expected[name], f'{context}:{name}:expected')
        _finite(observed[name], f'{context}:{name}')
        if name == 'n':
            _require(int(observed[name]) == int(expected[name]),
                     f'{context}:{name} mismatch')
        else:
            _require(math.isclose(float(observed[name]), float(expected[name]),
                                  rel_tol=0.0, abs_tol=1e-12),
                     f'{context}:{name} mismatch')


def _verify_stratified_reports(isr_dir: Path, reports, csv_identities):
    expected_keys = _expected_stratified_keys()
    csv_by_station = {}
    for identity in csv_identities:
        path = isr_dir / identity['relative_path']
        station = path.parent.name
        _require(station not in csv_by_station,
                 f'ISR: duplicate stratified CSV for {station}')
        csv_by_station[station] = _read_stratified_csv(path)
    report_by_station = {report['station']: report for report in reports}
    _require(set(csv_by_station) == set(report_by_station),
             'ISR: stratified CSV station set does not match JSON report')

    with _load_npz(isr_dir / 'isr_evaluation_cache.npz') as cache:
        station_values = np.asarray(cache['station']).astype(str)
        altitude = np.asarray(cache['altitude_km'], dtype=np.float64)
        observation = np.asarray(cache['observation_log10'], dtype=np.float64)
        for station, report in report_by_station.items():
            stratified = report.get('stratified')
            _require(isinstance(stratified, dict),
                     f'ISR {station}: missing stratified metrics')
            _require(set(stratified) == expected_keys,
                     f'ISR {station}: old or incomplete stratified metric keys')
            csv_rows = csv_by_station[station]
            _require(set(csv_rows) == expected_keys,
                     f'ISR {station}: old or incomplete stratified CSV keys')
            for key, values in stratified.items():
                row = csv_rows[key]
                source, layer = key.split('_', 1)
                _require(row['source'] == source and row['layer'] == layer,
                         f'ISR {station}:{key}: CSV labels mismatch')
                _compare_metrics(values, {name: row[name]
                                          for name in _STRATIFIED_METRICS},
                                 f'ISR {station}:{key}:CSV/JSON')

            for source in _STRATIFIED_SOURCES:
                for lower, upper in (*_STRATIFIED_BINS, _FULL_ALTITUDE_RANGE):
                    prefix = f'{source}_alt_{lower:g}-{upper:g}km_'
                    _require(
                        stratified[prefix + 'day']['n']
                        + stratified[prefix + 'night']['n']
                        == stratified[prefix + 'all']['n'],
                        f'ISR {station}:{prefix}: day/night count mismatch')
                canonical_prefix = (
                    f'{source}_alt_{_FULL_ALTITUDE_RANGE[0]:g}-'
                    f'{_FULL_ALTITUDE_RANGE[1]:g}km_')
                alias_prefix = f'{source}_all_alt_'
                _require(
                    stratified[alias_prefix + 'day']['n']
                    + stratified[alias_prefix + 'night']['n']
                    == stratified[alias_prefix + 'all']['n'],
                    f'ISR {station}:{alias_prefix}: day/night count mismatch')
                for period in _STRATIFIED_PERIODS:
                    _compare_metrics(
                        stratified[canonical_prefix + period],
                        stratified[alias_prefix + period],
                        f'ISR {station}:{source}: full-alt alias mismatch')

            station_mask = station_values == station
            for source, cache_key in _STRATIFIED_SOURCES.items():
                # M00 reports use their own finite-pair mask; the shared cache
                # intentionally follows the M11/IRI public mask.
                if source == 'background':
                    continue
                prediction = np.asarray(cache[cache_key], dtype=np.float64)
                for lower, upper in (*_STRATIFIED_BINS, _FULL_ALTITUDE_RANGE):
                    altitude_mask = ((altitude >= lower)
                                     & (altitude <= upper if lower >= 300.0
                                        or (lower, upper) == _FULL_ALTITUDE_RANGE
                                        else altitude < upper))
                    selected = station_mask & altitude_mask
                    expected = _metric_values(
                        observation[selected], prediction[selected])
                    key = f'{source}_alt_{lower:g}-{upper:g}km_all'
                    _compare_metrics(expected, stratified[key],
                                     f'ISR {station}:{key}:cache/JSON')


def _compare_reference_caches(reference_dir: Path, output_dir: Path):
    names = (
        'isr_evaluation_cache.npz',
        'isr_peak_cache.npz',
        'isr_paired_evaluation_cache.npz',
        'isr_paired_evaluation_cache_historical-epoch12.npz',
    )
    for name in names:
        reference_path = reference_dir / name
        output_path = output_dir / name
        _require(reference_path.is_file(), f'missing reference cache: {reference_path}')
        _require(output_path.is_file(), f'missing rerun cache: {output_path}')
        with _load_npz(reference_path) as reference, _load_npz(output_path) as candidate:
            _require(reference.files == candidate.files,
                     f'{name}: cache key sequence changed')
            for key in reference.files:
                left = np.asarray(reference[key])
                right = np.asarray(candidate[key])
                _require(left.shape == right.shape and left.dtype == right.dtype,
                         f'{name}:{key}: shape or dtype changed')
                _require(_array_equal(left, right),
                         f'{name}:{key}: values changed')


def _collect_key_numbers(value, key):
    found = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key and isinstance(item_value, (int, float)):
                found.append(float(item_value))
            found.extend(_collect_key_numbers(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_key_numbers(item, key))
    return found


def _cache_summary(isr_dir: Path):
    summary = {}
    with _load_npz(isr_dir / 'isr_evaluation_cache.npz') as cache:
        count = _assert_parallel_arrays(cache, 'ISR evaluation cache')
        station = np.asarray(cache['station'])
        summary['point_count'] = count
        summary['station_counts'] = {
            str(name): int((station == name).sum()) for name in np.unique(station)
        }
        summary['finite_rates'] = {
            key: float(np.isfinite(np.asarray(cache[key], dtype=np.float64)).mean())
            for key in ('observation_log10', 'M11_log10', 'M00_log10', 'IRI_log10')
        }
    with _load_npz(isr_dir / 'isr_peak_cache.npz') as cache:
        count = _assert_parallel_arrays(cache, 'ISR peak cache')
        summary['peak_count'] = count
        summary['peak_status_counts'] = {
            key: {
                str(status): int(number)
                for status, number in zip(*np.unique(cache[key], return_counts=True))
            }
            for key in cache.files if key.endswith('_status')
        }
    with _load_npz(isr_dir / 'isr_paired_evaluation_cache.npz') as cache:
        summary['primary_paired_count'] = _assert_parallel_arrays(
            cache, 'ISR primary paired cache')
    with _load_npz(
            isr_dir / 'isr_paired_evaluation_cache_historical-epoch12.npz') as cache:
        summary['historical_paired_count'] = _assert_parallel_arrays(
            cache, 'ISR historical paired cache')
    return summary


def _verify_isr(isr_dir: Path, expected_candidate, expected_baselines,
                expected_date_split, reference_dir: Path | None):
    required = (
        'isr_validation_report.txt', 'isr_validation_report.json',
        'isr_evaluation_contract.json', 'isr_evaluation_cache.npz',
        'isr_peak_cache.npz', 'isr_paired_evaluation_cache.npz',
        'isr_paired_evaluation_cache_historical-epoch12.npz',
    )
    for name in required:
        _require((isr_dir / name).is_file(), f'ISR: missing {name}')
    contract = _strict_json(isr_dir / 'isr_evaluation_contract.json')
    reports = _strict_json(isr_dir / 'isr_validation_report.json')
    _validate_contract_common(contract, expected_candidate, expected_baselines,
                              expected_date_split, 'ISR')
    _require(contract.get('stratified_metrics_contract_version') == 4,
             'ISR: stratified_metrics_contract_version must be 4')
    _require(contract.get('stratified_metrics') == _EXPECTED_STRATIFIED_CONTRACT,
             'ISR: stratified metrics contract mismatch')
    _require(isinstance(contract.get('output_artifacts'), dict),
             'ISR: missing output artifact identities')
    csv_identities = contract['output_artifacts'].get('stratified_csvs')
    _require(isinstance(csv_identities, list) and len(csv_identities) == 2,
             'ISR: expected two contracted stratified CSV files')
    for identity in ([contract['output_artifacts'].get('report_text'),
                      contract['output_artifacts'].get('report_json')]
                     + contract['output_artifacts'].get('caches', [])
                     + csv_identities):
        _require(isinstance(identity, dict), 'ISR: malformed output artifact identity')
        path = isr_dir / identity.get('relative_path', '')
        _require(path.is_file(), f'ISR: missing contracted artifact {path}')
        _require(identity.get('sha256') == _sha256(path),
                 f'ISR: artifact SHA mismatch for {path.name}')
        _require(identity.get('size_bytes') == path.stat().st_size,
                 f'ISR: artifact size mismatch for {path.name}')

    geometry = contract.get('poker_geometry', {})
    _require(geometry.get('semantics') == 'wgs84_enu_los_to_ecef_to_geodetic_v1',
             'ISR: Poker geometry semantics mismatch')
    _require(geometry.get('range_scale_to_km') == 1e-3,
             'ISR: Poker range scale mismatch')
    geometry_delta = _collect_key_numbers(
        geometry.get('audit', []), 'legacy_range_as_height_delta_km_median')
    _require(geometry_delta, 'ISR: missing Poker height-delta audit')
    _require(abs(float(np.median(geometry_delta)) - 0.215) <= 0.01,
             'ISR: Poker geometry audit does not reproduce the expected +0.215 km')

    _require(isinstance(reports, list), 'ISR: report must be a station list')
    report_by_station = {report.get('station'): report for report in reports}
    _require(set(report_by_station) == {'Jicamarca', 'PokerFlat'},
             f'ISR: expected Jicamarca/PokerFlat reports, found {sorted(report_by_station)}')
    for station, report in report_by_station.items():
        diagnostic = report.get('low_altitude_diagnostic')
        _require(isinstance(diagnostic, dict),
                 f'ISR {station}: low_altitude_diagnostic must be an object')
        _require(diagnostic.get('status') in _LOW_ALTITUDE_STATUSES,
                 f'ISR {station}: invalid low-altitude diagnostic status')
        _require(diagnostic.get('altitude_range_km') == [120.0, 200.0],
                 f'ISR {station}: low-altitude diagnostic range mismatch')
        if diagnostic.get('status') == 'computed':
            point_metrics = diagnostic.get('point_metrics')
            bootstrap = diagnostic.get('bootstrap')
            _require(set(point_metrics or {}) == {'M11', 'M00', 'Raw IRI'},
                     f'ISR {station}: low-altitude point metrics missing')
            _require(set(bootstrap or {}) == {'M11', 'M00', 'Raw IRI'},
                     f'ISR {station}: low-altitude bootstrap missing')
            for model in ('M11', 'M00', 'Raw IRI'):
                metrics = point_metrics[model]
                _require(set(metrics) == {'n', 'rmse', 'bias', 'mae'},
                         f'ISR {station}: malformed low-altitude {model} metrics')
                for name, value in metrics.items():
                    _finite(value, f'ISR {station}:low-altitude:{model}:{name}')
                _require(bootstrap[model].get('status') == 'computed',
                         f'ISR {station}: low-altitude {model} bootstrap incomplete')
                for name in ('rmse', 'bias', 'mae'):
                    interval = bootstrap[model].get(name, {})
                    _finite(interval.get('estimate'),
                            f'ISR {station}:low-altitude:{model}:{name}:estimate')
                    _require(isinstance(interval.get('ci95'), list)
                             and len(interval['ci95']) == 2,
                             f'ISR {station}: low-altitude {model} CI missing')
                    for bound in interval['ci95']:
                        _finite(bound, f'ISR {station}:low-altitude:{model}:{name}:CI')
        _require(isinstance(report.get('primary_peak_metrics'), dict),
                 f'ISR {station}: missing primary peak metrics')
        _require(isinstance(report.get('legacy_peak_metrics'), dict),
                 f'ISR {station}: missing legacy peak metrics')
        _require(isinstance(report.get('mask_attrition'), dict),
                 f'ISR {station}: missing mask attrition')
        _require(isinstance(report.get('peak_qc_counts'), dict),
                 f'ISR {station}: missing peak QA counts')
        for count_key, metric_key in (
                ('point_n', 'point_rmse'), ('nmf2_n', 'nmf2_mae'),
                ('hmf2_n', 'hmf2_mae')):
            if int(report.get(count_key, 0)) > 0:
                _finite(report.get(metric_key), f'ISR {station}:{metric_key}')

    summary_once = _cache_summary(isr_dir)
    summary_twice = _cache_summary(isr_dir)
    _require(summary_once == summary_twice,
             'ISR: cache summary is not deterministic')
    for status_counts in summary_once['peak_status_counts'].values():
        _require(set(status_counts).issubset(set(PEAK_STATUSES)),
                 'ISR: peak cache contains an unknown status')
    _verify_stratified_reports(isr_dir, reports, csv_identities)
    if reference_dir is not None:
        _compare_reference_caches(reference_dir, isr_dir)
    return {
        'contract_sha256': _sha256(isr_dir / 'isr_evaluation_contract.json'),
        'report_sha256': _sha256(isr_dir / 'isr_validation_report.json'),
        'cache_summary': summary_once,
    }


def _verify_giro(giro_dir: Path, expected_candidate, expected_baselines,
                 expected_date_split):
    required = (
        'giro_peak_contract.json', 'giro_peak_report.json',
        'giro_peak_cache.npz',
    )
    for name in required:
        _require((giro_dir / name).is_file(), f'GIRO: missing {name}')
    contract = _strict_json(giro_dir / 'giro_peak_contract.json')
    report = _strict_json(giro_dir / 'giro_peak_report.json')
    _validate_contract_common(contract, expected_candidate, expected_baselines,
                              expected_date_split, 'GIRO')
    _require(contract.get('quality_thresholds', {}).get('hmf2_public_mask')
             == 'observation_finite_and_all_compared_fields_valid',
             'GIRO: hmF2 public-mask contract mismatch')
    _require(contract.get('quality_thresholds', {}).get('nmf2_public_mask')
             == 'observation_finite_and_all_compared_fields_nmf2_valid',
             'GIRO: NmF2 public-mask contract mismatch')
    _require(isinstance(report.get('hmF2'), dict)
             and isinstance(report.get('NmF2'), dict),
             'GIRO: missing peak reports')
    for quantity in ('hmF2', 'NmF2'):
        _require(isinstance(report[quantity].get('metrics'), dict),
                 f'GIRO: missing {quantity} metrics')
        _require(isinstance(report[quantity].get('legacy_argmax_metrics'), dict),
                 f'GIRO: missing {quantity} legacy metrics')
    _require(set(report.get('candidate_vs_baseline', {}))
             == {label for label, _ in expected_baselines},
             'GIRO: candidate/baseline comparison labels mismatch')
    with _load_npz(giro_dir / 'giro_peak_cache.npz') as cache:
        _require(cache.files, 'GIRO: empty peak cache')
        for key in cache.files:
            if key.endswith('_status'):
                _require(set(map(str, np.unique(cache[key]))).issubset(PEAK_STATUSES),
                         f'GIRO: unknown peak status in {key}')
    return {
        'contract_sha256': _sha256(giro_dir / 'giro_peak_contract.json'),
        'report_sha256': _sha256(giro_dir / 'giro_peak_report.json'),
    }


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=path.parent, delete=False,
                prefix=f'.{path.name}.', suffix='.tmp') as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_baseline(value):
    label, separator, sha256 = value.partition('=')
    if not separator or not label or len(sha256) != 64:
        raise argparse.ArgumentTypeError(
            'baseline must be LABEL=64-character-sha256')
    return label, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isr-dir', type=Path, required=True)
    parser.add_argument('--giro-dir', type=Path, required=True)
    parser.add_argument('--expected-candidate-sha', required=True)
    parser.add_argument('--expected-baseline', type=_parse_baseline,
                        action='append', required=True)
    parser.add_argument('--expected-date-split-sha', required=True)
    parser.add_argument('--reference-isr-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _require(len(args.expected_candidate_sha) == 64,
                 'candidate SHA must be 64 hexadecimal characters')
        _require(len(args.expected_date_split_sha) == 64,
                 'date-split SHA must be 64 hexadecimal characters')
        isr = _verify_isr(args.isr_dir, args.expected_candidate_sha,
                          args.expected_baseline, args.expected_date_split_sha,
                          args.reference_isr_dir)
        giro = _verify_giro(args.giro_dir, args.expected_candidate_sha,
                            args.expected_baseline,
                            args.expected_date_split_sha)
        payload = {
            'acceptance_schema_version': 1,
            'status': 'pass',
            'candidate_checkpoint_sha256': args.expected_candidate_sha,
            'baseline_checkpoints': [
                {'label': label, 'sha256': sha256}
                for label, sha256 in args.expected_baseline
            ],
            'date_split_sha256': args.expected_date_split_sha,
            'isr_directory': str(args.isr_dir.resolve()),
            'giro_directory': str(args.giro_dir.resolve()),
            'isr': isr,
            'giro': giro,
        }
    except ContractError as exc:
        payload = {
            'acceptance_schema_version': 1,
            'status': 'fail',
            'error': str(exc),
        }
        _write_json(args.output, payload)
        print(f'P0-A contract verification failed: {exc}', file=sys.stderr)
        return 1
    _write_json(args.output, payload)
    print(f'P0-A contract verification passed: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
