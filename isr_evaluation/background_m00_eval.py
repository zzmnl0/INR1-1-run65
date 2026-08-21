"""Evaluate a Background-only M00 checkpoint on the canonical ISR cache."""

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch

from inr_modules.mdia.evaluation_stats import paired_group_bootstrap


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metrics(observation, prediction):
    observation = np.asarray(observation, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(observation) & np.isfinite(prediction)
    observation, prediction = observation[finite], prediction[finite]
    if not len(observation):
        return {
            'n': 0, 'bias': None, 'rmse': None, 'mae': None,
            'pearson_r': None, 'ccc': None,
        }
    error = prediction - observation
    obs_mean, pred_mean = observation.mean(), prediction.mean()
    obs_var, pred_var = observation.var(), prediction.var()
    covariance = np.mean(
        (observation - obs_mean) * (prediction - pred_mean))
    denominator = np.sqrt(obs_var * pred_var)
    ccc_denominator = obs_var + pred_var + (obs_mean - pred_mean) ** 2
    return {
        'n': int(len(observation)),
        'bias': float(error.mean()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.mean(np.abs(error))),
        'pearson_r': (
            float(covariance / denominator) if denominator > 1e-30 else None),
        'ccc': (
            float(2.0 * covariance / ccc_denominator)
            if ccc_denominator > 1e-30 else None),
    }


def _reporting_masks(altitude, local_time):
    """Yield the fixed station-specific ISR strata; no station=all aggregate."""
    altitude = np.asarray(altitude, dtype=np.float64)
    local_time = np.asarray(local_time, dtype=np.float64)
    day = (local_time >= 6.0) & (local_time < 18.0)
    ranges = (
        ('low_alt_120_200', (altitude >= 120.0) & (altitude < 200.0)),
        ('main_200_300', (altitude >= 200.0) & (altitude < 300.0)),
        ('main_300_500', (altitude >= 300.0) & (altitude <= 500.0)),
        ('main_200_500', (altitude >= 200.0) & (altitude <= 500.0)),
        ('all_alt', (altitude >= 120.0) & (altitude <= 500.0)),
    )
    for label, altitude_mask in ranges:
        for period, period_mask in (
                ('all', np.ones(len(altitude), dtype=bool)),
                ('day', day), ('night', ~day)):
            yield f'{label}_{period}', altitude_mask & period_mask


def _metric_and_bootstrap_rows(station, altitude, local_time, observation,
                               predictions, unit_ids,
                               candidate_name='M00_A1S', baseline_names=None):
    metric_rows, bootstrap_rows = [], []
    if baseline_names is None:
        baseline_names = ('M00_A0', 'Raw_IRI', 'Static_ETKF')
    common = np.isfinite(observation)
    for values in predictions.values():
        common &= np.isfinite(values)
    for stratum, stratum_mask in _reporting_masks(altitude, local_time):
        selected = common & stratum_mask
        for model_name, values in predictions.items():
            metric_rows.append({
                'station': station,
                'stratum': stratum,
                'model': model_name,
                **_metrics(observation[selected], values[selected]),
            })
        if selected.sum() < 2 or np.unique(unit_ids[selected]).size < 2:
            continue
        for baseline_name in baseline_names:
            bootstrap_rows.append({
                'station': station,
                'stratum': stratum,
                'candidate': candidate_name,
                'baseline': baseline_name,
                **paired_group_bootstrap(
                    observation[selected], predictions[candidate_name][selected],
                    predictions[baseline_name][selected], unit_ids[selected],
                    replicates=2000, seed=42),
            })
    return metric_rows, bootstrap_rows


def _load_background_model(checkpoint, device):
    """Strict-load Background state without Analysis checkpoint helpers."""
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.mdia.fsia_model import FSIA_INR_Model

    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint.parent / 'run_manifest.json'
    summary_path = checkpoint.parent / 'training_summary.json'
    for path in (checkpoint, manifest_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f'required Background artifact is missing: {path}')
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    with summary_path.open(encoding='utf-8') as stream:
        summary = json.load(stream)
    config = manifest.get('config')
    if not isinstance(config, dict):
        raise ValueError('run manifest lacks config object')
    if summary.get('completed_stage') != 'background':
        raise ValueError('M00 ISR requires a completed Background-only run')
    if tuple(map(float, config.get('alt_range', ()))) != (120.0, 500.0):
        raise ValueError('M00 ISR requires the 120--500 km model domain')

    device = torch.device(device)
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    proxy_state = torch.load(
        config['iri_proxy_path'], map_location=device, weights_only=True)
    proxy.load_state_dict(proxy_state, strict=True)
    proxy.eval()
    model = FSIA_INR_Model(iri_proxy=proxy, config=config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(
            torch.isfinite(value).all()
            for value in state.values() if torch.is_tensor(value)):
        raise ValueError('Background checkpoint contains non-finite tensors')
    model.load_state_dict(state, strict=True)
    model.eval()
    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'], start_date_str=config['start_date_str'],
        total_hours=config['total_hours'], seq_len=config['seq_len'],
        device=device)

    peak_manager = None
    hmf2_path, nmf2_path = (
        config.get('iri_hmf2_path'), config.get('iri_nmf2_path'))
    if hmf2_path and nmf2_path and Path(hmf2_path).is_file() and Path(nmf2_path).is_file():
        from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
        peak_manager = IRIPeakManager(
            hmf2_path=hmf2_path, nmf2_path=nmf2_path, device=device)
    return model, sw_manager, peak_manager, config


def _infer_background(model, sw_manager, peak_manager, config, cache,
                      device, batch_size):
    start_time = dt.datetime.fromisoformat(config['start_date_str'])
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=dt.timezone.utc)
    start_unix = start_time.timestamp()
    relative_hour = (
        cache['timestamp'].astype(np.float64) - start_unix) / 3600.0
    coords = np.column_stack([
        cache['lat'], cache['lon'], cache['altitude_km'], relative_hour,
    ]).astype(np.float32)
    if not np.isfinite(coords).all():
        raise ValueError('canonical ISR cache contains non-finite coordinates')
    if ((coords[:, 2] < 120.0) | (coords[:, 2] > 500.0)).any():
        raise ValueError('canonical ISR cache contains coordinates outside 120--500 km')

    m00 = np.empty(len(coords), dtype=np.float32)
    raw_iri = np.empty(len(coords), dtype=np.float32)
    device = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            end = min(start + batch_size, len(coords))
            chunk = torch.from_numpy(coords[start:end]).to(device)
            sw = sw_manager.get_drivers_sequence(chunk[:, 3])
            iri_peak = (
                peak_manager.get_iri_peak(chunk)
                if peak_manager is not None else None)
            background = model.encode_background(chunk, sw, iri_peak=iri_peak)
            m00[start:end] = background['ne_bkg'].flatten().cpu().numpy()
            raw_iri[start:end] = background['ne_iri'].flatten().cpu().numpy()
    if not np.isfinite(m00).all() or not np.isfinite(raw_iri).all():
        raise FloatingPointError('Background inference produced non-finite values')
    return m00.astype(np.float64), raw_iri.astype(np.float64), relative_hour


def run(checkpoint, cache_path, output_dir, device='cpu', batch_size=2048,
        expected_rows=206230, comparison_cache=None,
        candidate_name='M00_A1S'):
    started = time.time()
    cache_path = Path(cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cache_path, allow_pickle=False) as source:
        required = {
            'station', 'key', 'observation', 'M00_log10', 'altitude_km',
            'unit_id', 'timestamp', 'lat', 'lon', 'ETKF_log10',
            'raw_iri_continuous_log10',
        }
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f'canonical ISR cache lacks fields: {missing}')
        cache = {key: source[key] for key in source.files}
    row_count = len(cache['key'])
    if expected_rows is not None and row_count != int(expected_rows):
        raise ValueError(
            f'canonical ISR row count is {row_count}, expected {expected_rows}')
    if any(len(cache[key]) != row_count for key in required):
        raise ValueError('canonical ISR cache columns have unequal lengths')
    if len(np.unique(cache['key'])) != row_count:
        raise ValueError('canonical ISR cache keys are not unique')
    stations = np.unique(cache['station'])
    if set(stations.tolist()) != {'Jicamarca', 'PokerFlat'}:
        raise ValueError(f'unexpected canonical ISR stations: {stations.tolist()}')

    model, sw_manager, peak_manager, config = _load_background_model(
        checkpoint, device)
    m00, raw_iri, relative_hour = _infer_background(
        model, sw_manager, peak_manager, config, cache, device, int(batch_size))
    observation = cache['observation'].astype(np.float64)
    predictions = None
    baseline_names = None
    comparison_identity = None
    if comparison_cache:
        comparison_path = Path(comparison_cache).resolve()
        with np.load(comparison_path, allow_pickle=False) as source:
            required_baseline = {'station', 'key', 'unit_id',
                                 'altitude_km', 'M00_A1S_log10'}
            missing = sorted(required_baseline - set(source.files))
            if missing:
                raise ValueError(
                    f'comparison ISR cache lacks fields: {missing}')
            baseline = {key: source[key] for key in source.files}
        for key in ('station', 'key', 'unit_id', 'altitude_km'):
            if len(baseline[key]) != row_count:
                raise ValueError('comparison ISR cache row count differs')
            if not np.array_equal(baseline[key], cache[key]):
                raise ValueError(
                    f'comparison ISR cache key alignment differs for {key}')
        predictions = {
            candidate_name: m00,
            'Raw_IRI': raw_iri,
            'Static_ETKF': cache['ETKF_log10'].astype(np.float64),
            'M00_A1S': baseline['M00_A1S_log10'].astype(np.float64),
        }
        baseline_names = ['M00_A1S', 'Raw_IRI', 'Static_ETKF']
        comparison_identity = str(comparison_path)
    else:
        predictions = {
            candidate_name: m00,
            'Raw_IRI': raw_iri,
            'Static_ETKF': cache['ETKF_log10'].astype(np.float64),
            'M00_A0': cache['M00_log10'].astype(np.float64),
        }
        baseline_names = ['M00_A0', 'Raw_IRI', 'Static_ETKF']
    local_time = np.remainder(
        relative_hour + cache['lon'].astype(np.float64) / 15.0, 24.0)
    metric_rows, bootstrap_rows = [], []
    for station in stations:
        selected = cache['station'] == station
        units = np.asarray([
            f'{station}|{int(value)}' for value in cache['unit_id'][selected]
        ])
        station_predictions = {
            key: value[selected] for key, value in predictions.items()}
        rows, bootstraps = _metric_and_bootstrap_rows(
            station, cache['altitude_km'][selected], local_time[selected],
            observation[selected], station_predictions, units,
            candidate_name, baseline_names)
        metric_rows.extend(rows)
        bootstrap_rows.extend(bootstraps)

    comparison_rows = []
    acceptance = None
    if comparison_cache:
        metric_index = {
            (row['station'], row['stratum'], row['model']): row
            for row in metric_rows}
        for station in stations.tolist():
            for stratum in (
                    'main_200_500_all', 'main_200_500_day',
                    'main_200_500_night', 'all_alt_all', 'all_alt_day',
                    'all_alt_night'):
                candidate = metric_index[(station, stratum, candidate_name)]
                baseline = metric_index[(station, stratum, 'M00_A1S')]
                comparison_row = {
                    'station': station,
                    'stratum': stratum,
                    'bias_delta_candidate_minus_A1S': float(
                        candidate['bias'] - baseline['bias']),
                    'abs_bias_delta_candidate_minus_A1S': float(
                        abs(candidate['bias']) - abs(baseline['bias'])),
                    'rmse_delta_candidate_minus_A1S': float(
                        candidate['rmse'] - baseline['rmse']),
                    'ccc_delta_candidate_minus_A1S': float(
                        candidate['ccc'] - baseline['ccc']),
                }
                comparison_rows.append(comparison_row)
        primary = {
            row['station']: row for row in comparison_rows
            if row['stratum'] == 'main_200_500_all'}
        jic_bias_ok = bool(
            primary['Jicamarca']['abs_bias_delta_candidate_minus_A1S'] < 0.0)
        poker_bias_ok = bool(
            primary['PokerFlat']['abs_bias_delta_candidate_minus_A1S'] < 0.0)
        acceptance = {
            'candidate': candidate_name,
            'Jicamarca_bias_improved': jic_bias_ok,
            'PokerFlat_bias_improved_vs_A1S': poker_bias_ok,
            'main_rmse_non_degraded': bool(all(
                row['rmse_delta_candidate_minus_A1S'] <= 0.01
                for row in primary.values())),
            'main_ccc_non_degraded': bool(all(
                row['ccc_delta_candidate_minus_A1S'] >= -0.01
                for row in primary.values())),
            'passed': bool(
                jic_bias_ok and poker_bias_ok
                and all(row['rmse_delta_candidate_minus_A1S'] <= 0.01
                        for row in primary.values())
                and all(row['ccc_delta_candidate_minus_A1S'] >= -0.01
                        for row in primary.values())),
        }

    cache_payload = {
        'station': cache['station'], 'key': cache['key'],
        'unit_id': cache['unit_id'], 'timestamp': cache['timestamp'],
        'lat': cache['lat'], 'lon': cache['lon'],
        'altitude_km': cache['altitude_km'], 'observation_log10': observation,
        'candidate_log10': m00,
        'M00_A1S_log10': predictions.get('M00_A1S', m00),
    }
    cache_payload.update({
        'M00_A0_log10': predictions.get('M00_A0', cache['M00_log10']).astype(
            np.float64),
        'Raw_IRI_log10': raw_iri,
        'Static_ETKF_log10': predictions['Static_ETKF'],
    })
    np.savez_compressed(output_dir / 'isr_m00_cache.npz', **cache_payload)
    csv_path = output_dir / 'isr_m00_stratified_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'station', 'stratum', 'model', 'n', 'bias', 'rmse', 'mae',
            'pearson_r', 'ccc'))
        writer.writeheader()
        writer.writerows(metric_rows)
    report = {
        'schema_version': 1,
        'scope': 'Background-only M00; no M11 is computed or reported',
        'candidate_name': candidate_name,
        'checkpoint': str(Path(checkpoint).resolve()),
        'canonical_cache': str(cache_path),
        'comparison_cache': comparison_identity,
        'row_count': row_count,
        'stations': stations.tolist(),
        'stratified_metrics': metric_rows,
        'paired_bootstrap': {
            'replicates': 2000, 'seed': 42,
            'sampling_unit': 'station_time_profile',
            'comparisons': bootstrap_rows,
        },
        'comparison_summary_vs_A1S': comparison_rows,
        'acceptance_vs_A1S': acceptance,
    }
    (output_dir / 'isr_m00_report.json').write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding='utf-8')
    summary_rows = [
        row for row in metric_rows
        if row['stratum'] in ('main_200_500_all', 'all_alt_all')]
    text_lines = [
        'Background-only M00 ISR validation',
        'No new M11 is computed; Analysis must be retrained before M11 evaluation.',
    ]
    for row in summary_rows:
        text_lines.append(
            f"{row['station']} {row['stratum']} {row['model']}: "
            f"n={row['n']} Bias={row['bias']:.6f} RMSE={row['rmse']:.6f} "
            f"MAE={row['mae']:.6f} Pearson={row['pearson_r']:.6f} "
            f"CCC={row['ccc']:.6f}")
    if acceptance is not None:
        text_lines.append(f'{candidate_name} vs A1-S main_200_500 acceptance: '
                          + ('PASS' if acceptance['passed'] else 'FAIL'))
        for row in comparison_rows:
            if row['stratum'] == 'main_200_500_all':
                text_lines.append(
                    f"{row['station']} {candidate_name}-A1S BiasΔ="
                    f"{row['bias_delta_candidate_minus_A1S']:.6f} "
                    f"|Bias|Δ={row['abs_bias_delta_candidate_minus_A1S']:.6f} "
                    f"RMSEΔ={row['rmse_delta_candidate_minus_A1S']:.6f} "
                    f"CCCΔ={row['ccc_delta_candidate_minus_A1S']:.6f}")
    (output_dir / 'isr_m00_report.txt').write_text(
        '\n'.join(text_lines) + '\n', encoding='utf-8')
    audit = {
        'row_count': row_count,
        'unique_key_count': int(len(np.unique(cache['key']))),
        'station_counts': {
            station: int(np.sum(cache['station'] == station))
            for station in stations},
        'finite_output_counts': {
            name: int(np.isfinite(values).sum())
            for name, values in predictions.items()},
        'new_raw_iri_vs_cached_continuous_max_abs': float(np.max(np.abs(
            raw_iri - cache['raw_iri_continuous_log10'].astype(np.float64)))),
        'comparison_cache': comparison_identity,
        'wall_seconds': float(time.time() - started),
        'hash_or_sha_validation': False,
        'acceptance_vs_A1S': acceptance,
    }
    (output_dir / 'isr_m00_audit.json').write_text(
        json.dumps(_json_safe(audit), ensure_ascii=False, indent=2),
        encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--cache', default=(
            'isr_validation_outputs/traditional-static-etkf-raw-iri-full-isr/'
            'isr_etkf_cache.npz'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--expected-rows', type=int, default=206230)
    parser.add_argument(
        '--comparison-cache', default=None,
        help='A1-S M00 cache for strict same-coordinate comparison')
    parser.add_argument('--candidate-name', default='M00_A1S')
    args = parser.parse_args()
    run(args.checkpoint, args.cache, args.output_dir, args.device,
        args.batch_size, args.expected_rows, args.comparison_cache,
        args.candidate_name)


if __name__ == '__main__':
    main()
