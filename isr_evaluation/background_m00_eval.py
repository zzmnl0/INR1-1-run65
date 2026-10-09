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


JOINT_FRESH_LABEL = 'M00_COND_ADAPTER_JOINT_FRESH'
FULL30_ETKF_LABEL = 'Static_ETKF_FULL30'


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
            'prediction_std': None, 'target_std': None,
            'std_ratio': None, 'spread_error': None,
            'centered_rmse': None,
        }
    error = prediction - observation
    obs_mean, pred_mean = observation.mean(), prediction.mean()
    obs_var, pred_var = observation.var(), prediction.var()
    covariance = np.mean(
        (observation - obs_mean) * (prediction - pred_mean))
    denominator = np.sqrt(obs_var * pred_var)
    ccc_denominator = obs_var + pred_var + (obs_mean - pred_mean) ** 2
    target_std = float(np.sqrt(obs_var))
    prediction_std = float(np.sqrt(pred_var))
    std_ratio = float(
        (prediction_std + 1e-4) / (target_std + 1e-4))
    centered_error = (
        (prediction - pred_mean) - (observation - obs_mean))
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
        'prediction_std': prediction_std,
        'target_std': target_std,
        'std_ratio': std_ratio,
        'spread_error': float(abs(np.log(std_ratio))),
        'centered_rmse': float(np.sqrt(np.mean(centered_error ** 2))),
    }


def _reporting_masks(altitude, local_time, cos_sza=None):
    """Yield the fixed station-specific ISR strata; no station=all aggregate."""
    altitude = np.asarray(altitude, dtype=np.float64)
    local_time = np.asarray(local_time, dtype=np.float64)
    if cos_sza is not None:
        cos_sza = np.asarray(cos_sza, dtype=np.float64)
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
        for start in range(0, 24, 3):
            yield (f'{label}_lt_{start:02d}_{start + 3:02d}',
                   altitude_mask & (local_time >= start)
                   & (local_time < start + 3))
        if cos_sza is not None:
            yield f'{label}_sza_day', altitude_mask & (cos_sza > 0.0)
            yield f'{label}_sza_night', altitude_mask & (cos_sza <= 0.0)


def _canonical_baselines(cache, static_etkf_name):
    """Return public baselines without silently re-running Raw IRI."""
    return {
        'Raw_IRI': cache['raw_iri_continuous_log10'].astype(np.float64),
        static_etkf_name: cache['ETKF_log10'].astype(np.float64),
    }


def _component_diagnostic_rows(station, altitude, local_time, base_residual,
                               adapter_delta, total_residual, residual_cap,
                               adapter_cap):
    """Summarize the two M00 residual components on the public ISR strata."""
    base_residual = np.asarray(base_residual, dtype=np.float64)
    adapter_delta = np.asarray(adapter_delta, dtype=np.float64)
    total_residual = np.asarray(total_residual, dtype=np.float64)
    finite = (np.isfinite(base_residual) & np.isfinite(adapter_delta)
              & np.isfinite(total_residual))
    rows = []
    for stratum, stratum_mask in _reporting_masks(altitude, local_time):
        selected = finite & stratum_mask
        base = base_residual[selected]
        adapter = adapter_delta[selected]
        count = int(selected.sum())
        rows.append({
            'station': station,
            'stratum': stratum,
            'n': count,
            'base_residual_mean': float(base.mean()) if count else None,
            'base_residual_rms': (
                float(np.sqrt(np.mean(base ** 2))) if count else None),
            'adapter_delta_mean': float(adapter.mean()) if count else None,
            'adapter_delta_rms': (
                float(np.sqrt(np.mean(adapter ** 2))) if count else None),
            'adapter_cap_saturation_rate': (
                float(np.mean(np.abs(adapter) >= adapter_cap * 0.95))
                if count else None),
            'total_residual_near_cap_rate': (
                float(np.mean(np.abs(total_residual[selected])
                              >= residual_cap * 0.95))
                if count else None),
        })
    return rows


def night_v2_acceptance(metric_rows, candidate_name='M00_NIGHT_V2',
                        reference_name='M00_A1S'):
    """Apply the frozen night-v2 ISR gate to already aligned metric rows."""
    index = {(row['station'], row['stratum'], row['model']): row
             for row in metric_rows}
    checks, reasons = {}, []

    def check(station, stratum, rmse_improvement=None,
              rmse_degradation=None, ccc_improvement=None,
              ccc_degradation=None, bias_improvement=None):
        candidate = index.get((station, stratum, candidate_name))
        reference = index.get((station, stratum, reference_name))
        name = f'{station}:{stratum}'
        values = [row.get(key) for row in (candidate or {}, reference or {})
                  for key in ('rmse', 'ccc', 'bias')]
        if (not candidate or not reference or len(values) != 6
                or not all(isinstance(value, (int, float, np.number))
                           and np.isfinite(value) for value in values)):
            checks[name] = {'passed': False, 'reason': 'missing_or_nonfinite'}
            reasons.append(f'{name}:missing_or_nonfinite')
            return
        rmse_rel = ((reference['rmse'] - candidate['rmse']) /
                    reference['rmse']) if reference['rmse'] > 0 else float('-inf')
        ccc_delta = candidate['ccc'] - reference['ccc']
        abs_bias_gain = abs(reference['bias']) - abs(candidate['bias'])
        passed = all((
            rmse_improvement is None or rmse_rel >= rmse_improvement - 1e-8,
            rmse_degradation is None or rmse_rel >= -rmse_degradation - 1e-8,
            ccc_improvement is None or ccc_delta >= ccc_improvement - 1e-8,
            ccc_degradation is None or ccc_delta >= -ccc_degradation - 1e-8,
            bias_improvement is None or abs_bias_gain >= bias_improvement - 1e-8,
            candidate.get('n') == reference.get('n'),
        ))
        checks[name] = {'passed': passed, 'rmse_improvement_fraction': rmse_rel,
                        'ccc_delta': ccc_delta,
                        'absolute_bias_improvement_dex': abs_bias_gain}
        if not passed:
            reasons.append(f'{name}:threshold_failed')

    for station in ('Jicamarca', 'PokerFlat'):
        check(station, 'main_200_500_night', rmse_improvement=0.001,
              ccc_improvement=0.005,
              bias_improvement=0.01 if station == 'Jicamarca' else None)
        check(station, 'main_200_500_day', rmse_degradation=0.005,
              ccc_degradation=0.005)
        for altitude in ('main_200_300', 'main_300_500'):
            check(station, f'{altitude}_night', rmse_degradation=0.005,
                  ccc_degradation=0.005)
    return {'semantics': 'm00_night_isr_acceptance_v2',
            'candidate': candidate_name, 'reference': reference_name,
            'checks': checks, 'reasons': reasons, 'passed': not reasons}


def _metric_and_bootstrap_rows(station, altitude, local_time, observation,
                               predictions, unit_ids,
                               candidate_name='M00_A1S', baseline_names=None,
                               cos_sza=None):
    metric_rows, bootstrap_rows = [], []
    if baseline_names is None:
        baseline_names = ('M00_A0', 'Raw_IRI', 'Static_ETKF')
    common = np.isfinite(observation)
    for values in predictions.values():
        common &= np.isfinite(values)
    for stratum, stratum_mask in _reporting_masks(
            altitude, local_time, cos_sza):
        selected = common & stratum_mask
        for model_name, values in predictions.items():
            metric_rows.append({
                'station': station,
                'stratum': stratum,
                'model': model_name,
                **_metrics(observation[selected], values[selected]),
            })
        if candidate_name == 'M00_NIGHT_V2' or selected.sum() < 2 or np.unique(unit_ids[selected]).size < 2:
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


def _load_background_model(checkpoint, device, *, allow_unselected=False):
    """Strict-load Background state without Analysis checkpoint helpers."""
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.mdia.fsia_model import FSIA_INR_Model

    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint.parent / 'run_manifest.json'
    summary_path = checkpoint.parent / 'training_summary.json'
    required_paths = (checkpoint, manifest_path)
    if not allow_unselected:
        required_paths = required_paths + (summary_path,)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f'required Background artifact is missing: {path}')
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    summary = {}
    if summary_path.is_file():
        with summary_path.open(encoding='utf-8') as stream:
            summary = json.load(stream)
    config = manifest.get('config')
    if not isinstance(config, dict):
        raise ValueError('run manifest lacks config object')
    completed_stage = summary.get('completed_stage')
    # A full two-stage run retains the selected Background state alongside the
    # Analysis state. Permit only that immutable artifact for M00 evaluation.
    if completed_stage != 'background' and not (
            completed_stage == 'analysis'
            and checkpoint.name == 'best_background_model.pth'):
        raise ValueError('M00 ISR requires a completed Background checkpoint')
    if not allow_unselected and checkpoint.name != 'best_background_model.pth':
        raise ValueError('M00 ISR requires best_background_model.pth')
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


def _background_components(model, coords, background):
    """Read component extras when available, otherwise reproduce the two heads."""
    base = background.get('background_base_residual_raw')
    if base is None:
        decoder_input = torch.cat([
            background['z_background'], background['h_sw'],
            background['alt_n'].unsqueeze(-1),
            background['delta_alt'].unsqueeze(-1),
        ], dim=-1)
        base = model.background_residual_cap * torch.tanh(
            model.background_decoder(decoder_input))
    adapter = background.get('background_adapter_delta')
    if adapter is None:
        if model.background_condition_adapter is None:
            adapter = torch.zeros_like(base)
        else:
            features = model._background_condition_features(
                coords, background['z_background'], background['h_sw'],
                background['alt_n'], background['delta_alt'],
                background['sin_i'])
            adapter = model.background_adapter_residual_cap * torch.tanh(
                model.background_condition_adapter(features))
    return base.flatten(), adapter.flatten()


def _infer_background(model, sw_manager, peak_manager, config, cache,
                      device, batch_size, return_components=False):
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
    inferred_raw_iri = np.empty(len(coords), dtype=np.float32)
    base_residual = np.empty(len(coords), dtype=np.float32)
    adapter_delta = np.empty(len(coords), dtype=np.float32)
    total_residual = np.empty(len(coords), dtype=np.float32)
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
            inferred_raw_iri[start:end] = (
                background['ne_iri'].flatten().cpu().numpy())
            base, adapter = _background_components(model, chunk, background)
            base_residual[start:end] = base.cpu().numpy()
            adapter_delta[start:end] = adapter.cpu().numpy()
            total_residual[start:end] = background[
                'background_residual_raw'].flatten().cpu().numpy()
    if not all(np.isfinite(values).all() for values in (
            m00, inferred_raw_iri, base_residual, adapter_delta,
            total_residual)):
        raise FloatingPointError('Background inference produced non-finite values')
    legacy = (
        m00.astype(np.float64), inferred_raw_iri.astype(np.float64),
        relative_hour)
    if not return_components:
        return legacy
    return (*legacy, {
        'base_residual': base_residual.astype(np.float64),
        'adapter_delta': adapter_delta.astype(np.float64),
        'total_residual': total_residual.astype(np.float64),
        'residual_cap': float(model.background_residual_cap),
        'adapter_cap': float(model.background_adapter_residual_cap),
        'adapter_semantics': model.background_condition_adapter_semantics,
    })


def run(checkpoint, cache_path, output_dir, device='cpu', batch_size=2048,
        expected_rows=206230, comparison_cache=None,
        candidate_name='M00_A1S', reference_name='M00_A1S',
        diagnostic_only=False, static_etkf_name=None, allow_unselected_background=False):
    started = time.time()
    cache_path = Path(cache_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f'ISR output directory must be empty: {output_dir}')
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
        checkpoint, device, allow_unselected=allow_unselected_background)
    m00, inferred_raw_iri, relative_hour, components = _infer_background(
        model, sw_manager, peak_manager, config, cache, device,
        int(batch_size), return_components=True)
    observation = cache['observation'].astype(np.float64)
    if static_etkf_name is None:
        static_etkf_name = (
            FULL30_ETKF_LABEL
            if candidate_name == JOINT_FRESH_LABEL else 'Static_ETKF')
    canonical_baselines = _canonical_baselines(cache, static_etkf_name)
    predictions = None
    baseline_names = None
    comparison_identity = None
    if comparison_cache:
        comparison_path = Path(comparison_cache).resolve()
        with np.load(comparison_path, allow_pickle=False) as source:
            required_baseline = {'station', 'key', 'unit_id', 'altitude_km'}
            missing = sorted(required_baseline - set(source.files))
            if missing:
                raise ValueError(
                    f'comparison ISR cache lacks fields: {missing}')
            baseline = {key: source[key] for key in source.files}
            # Older full30 A1-S2 Background caches store the candidate under
            # the generic ``candidate_log10`` key.  Preserve that cache format
            # while exposing the explicit comparison name to this report.
            if ('M00_A1S2_log10' not in baseline
                    and 'candidate_log10' in baseline):
                baseline['M00_A1S2_log10'] = baseline['candidate_log10']
            reference_field = f'{reference_name}_log10'
            if reference_field not in baseline:
                raise ValueError(
                    f'comparison ISR cache lacks field: {reference_field}')
        for key in ('station', 'key', 'unit_id', 'altitude_km'):
            if len(baseline[key]) != row_count:
                raise ValueError('comparison ISR cache row count differs')
            if not np.array_equal(baseline[key], cache[key]):
                raise ValueError(
                    f'comparison ISR cache key alignment differs for {key}')
            predictions = {
                candidate_name: m00,
                **canonical_baselines,
            }
            baseline_names = ['Raw_IRI', static_etkf_name]
            if reference_name == 'M00_A1S2':
                predictions['M00_A1S2'] = baseline[
                    'M00_A1S2_log10'].astype(np.float64)
                baseline_names.insert(0, 'M00_A1S2')
            else:
                predictions['M00_A1S'] = baseline[
                    'M00_A1S_log10'].astype(np.float64)
                baseline_names.insert(0, 'M00_A1S')
            if 'M00_A1S2_log10' in baseline:
                # Keep the optional A1-S2 column only for the legacy A1-S
                # reference report; the condition-adapter experiment has a
                # strict four-model public comparison.
                if reference_name != 'M00_A1S2':
                    predictions['M00_A1S2'] = baseline[
                        'M00_A1S2_log10'].astype(np.float64)
                    baseline_names.insert(1, 'M00_A1S2')
        comparison_identity = str(comparison_path)
    else:
        predictions = {
            candidate_name: m00,
            **canonical_baselines,
            'M00_A0': cache['M00_log10'].astype(np.float64),
        }
        baseline_names = ['M00_A0', 'Raw_IRI', static_etkf_name]
    local_time = np.remainder(
        relative_hour + cache['lon'].astype(np.float64) / 15.0, 24.0)
    from inr_modules.mdia.fsia_model import _compute_solar_features
    cos_sza = _compute_solar_features(
        torch.as_tensor(cache['lat'], dtype=torch.float64),
        torch.as_tensor(cache['lon'], dtype=torch.float64),
        torch.as_tensor(relative_hour, dtype=torch.float64))[0].numpy()
    metric_rows, bootstrap_rows, component_rows = [], [], []
    date_bootstrap_rows = []
    utc_dates = cache['timestamp'].astype('datetime64[s]').astype('datetime64[D]')
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
            candidate_name, baseline_names, cos_sza[selected])
        metric_rows.extend(rows)
        bootstrap_rows.extend(bootstraps)
        component_rows.extend(_component_diagnostic_rows(
            station, cache['altitude_km'][selected], local_time[selected],
            components['base_residual'][selected],
            components['adapter_delta'][selected],
            components['total_residual'][selected],
            components['residual_cap'], components['adapter_cap']))
        if comparison_cache:
            from run_m00_condition_diagnostic import (
                paired_date_bootstrap_with_abs_bias)
            masks = dict(_reporting_masks(
                cache['altitude_km'][selected], local_time[selected],
                cos_sza[selected]))
            for stratum in ('main_200_500_night', 'main_200_500_day',
                            'main_200_300_night', 'main_300_500_night'):
                mask = masks[stratum]
                date_bootstrap_rows.append({
                    'station': str(station), 'stratum': stratum,
                    'candidate': candidate_name, 'baseline': reference_name,
                    **paired_date_bootstrap_with_abs_bias(
                        observation[selected][mask],
                        station_predictions[candidate_name][mask],
                        station_predictions[reference_name][mask],
                        utc_dates[selected][mask], replicates=2000, seed=42),
                })

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
                baseline = metric_index[(station, stratum, reference_name)]
                comparison_row = {
                    'station': station,
                    'stratum': stratum,
                    'bias_delta_candidate_minus_reference': float(
                        candidate['bias'] - baseline['bias']),
                    'abs_bias_delta_candidate_minus_reference': float(
                        abs(candidate['bias']) - abs(baseline['bias'])),
                    'rmse_delta_candidate_minus_reference': float(
                        candidate['rmse'] - baseline['rmse']),
                    'ccc_delta_candidate_minus_reference': float(
                        candidate['ccc'] - baseline['ccc']),
                }
                # Keep the historical field names for downstream reports.
                comparison_row.update({
                    'bias_delta_candidate_minus_A1S':
                        comparison_row['bias_delta_candidate_minus_reference'],
                    'abs_bias_delta_candidate_minus_A1S':
                        comparison_row['abs_bias_delta_candidate_minus_reference'],
                    'rmse_delta_candidate_minus_A1S':
                        comparison_row['rmse_delta_candidate_minus_reference'],
                    'ccc_delta_candidate_minus_A1S':
                        comparison_row['ccc_delta_candidate_minus_reference'],
                })
                comparison_rows.append(comparison_row)
        primary = {
            row['station']: row for row in comparison_rows
            if row['stratum'] == 'main_200_500_all'}
        jic_bias_ok = bool(
            primary['Jicamarca']['abs_bias_delta_candidate_minus_reference'] < 0.0)
        poker_bias_ok = bool(
            primary['PokerFlat']['abs_bias_delta_candidate_minus_reference'] < 0.0)
        acceptance = {
            'candidate': candidate_name,
            'reference': reference_name,
            'Jicamarca_bias_improved': jic_bias_ok,
            'PokerFlat_bias_improved_vs_reference': poker_bias_ok,
            'PokerFlat_bias_improved_vs_A1S': poker_bias_ok,
            'main_rmse_non_degraded': bool(all(
                row['rmse_delta_candidate_minus_reference'] <= 0.01
                for row in primary.values())),
            'main_ccc_non_degraded': bool(all(
                row['ccc_delta_candidate_minus_reference'] >= -0.01
                for row in primary.values())),
            'passed': bool(
                jic_bias_ok and poker_bias_ok
                and all(row['rmse_delta_candidate_minus_reference'] <= 0.01
                        for row in primary.values())
                and all(row['ccc_delta_candidate_minus_reference'] >= -0.01
                        for row in primary.values())),
        }

    if candidate_name == 'M00_NIGHT_V2':
        acceptance = night_v2_acceptance(metric_rows, candidate_name, reference_name)
    cache_payload = {
        'station': cache['station'], 'key': cache['key'],
        'unit_id': cache['unit_id'], 'timestamp': cache['timestamp'],
        'lat': cache['lat'], 'lon': cache['lon'],
        'altitude_km': cache['altitude_km'], 'observation_log10': observation,
        'candidate_log10': m00,
        f'{candidate_name}_log10': m00,
    }
    for model_name in ('M00_A0', 'M00_A1S', 'M00_A1S2'):
        if model_name in predictions:
            cache_payload[f'{model_name}_log10'] = predictions[model_name]
    cache_payload.update({
        'Raw_IRI_log10': predictions['Raw_IRI'],
        'Static_ETKF_log10': predictions[static_etkf_name],
        f'{static_etkf_name}_log10': predictions[static_etkf_name],
        'background_base_residual_raw': components['base_residual'],
        'background_adapter_delta': components['adapter_delta'],
        'background_residual_raw': components['total_residual'],
    })
    np.savez_compressed(output_dir / 'isr_m00_cache.npz', **cache_payload)
    csv_path = output_dir / 'isr_m00_stratified_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'station', 'stratum', 'model', 'n', 'bias', 'rmse', 'mae',
            'pearson_r', 'ccc', 'prediction_std', 'target_std',
            'std_ratio', 'spread_error', 'centered_rmse'))
        writer.writeheader()
        writer.writerows(metric_rows)
    report = {
        'schema_version': 1,
        'scope': 'Background-only M00; no M11 is computed or reported',
        'candidate_name': candidate_name,
        'reference_name': reference_name if comparison_cache else None,
        'diagnostic_only': bool(diagnostic_only),
        'allow_unselected_background': bool(allow_unselected_background),
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
        'paired_utc_date_bootstrap': {
            'replicates': 2000, 'seed': 42,
            'sampling_unit': 'station_utc_date',
            'comparisons': date_bootstrap_rows,
        },
        'reporting_partitions': {
            'local_time_hours': 3,
            'solar_day_boundary': 'SZA=90 degrees (cos_sza=0)',
        },
        'comparison_summary_vs_reference': comparison_rows,
        'acceptance_vs_reference': acceptance,
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
    if diagnostic_only:
        text_lines.insert(0, 'DIAGNOSTIC_ONLY: not eligible for model promotion')
    for row in summary_rows:
        text_lines.append(
            f"{row['station']} {row['stratum']} {row['model']}: "
            f"n={row['n']} Bias={row['bias']:.6f} RMSE={row['rmse']:.6f} "
            f"MAE={row['mae']:.6f} Pearson={row['pearson_r']:.6f} "
            f"CCC={row['ccc']:.6f}")
    if acceptance is not None:
        text_lines.append(f'{candidate_name} vs {reference_name} '
                          'main_200_500 acceptance: '
                          + ('PASS' if acceptance['passed'] else 'FAIL'))
        for row in comparison_rows:
            if row['stratum'] == 'main_200_500_all':
                text_lines.append(
                    f"{row['station']} {candidate_name}-{reference_name} BiasΔ="
                    f"{row['bias_delta_candidate_minus_A1S']:.6f} "
                    f"|Bias|Δ={row['abs_bias_delta_candidate_minus_reference']:.6f} "
                    f"RMSEΔ={row['rmse_delta_candidate_minus_reference']:.6f} "
                    f"CCCΔ={row['ccc_delta_candidate_minus_reference']:.6f}")
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
            inferred_raw_iri - cache['raw_iri_continuous_log10'].astype(np.float64)))),
        'comparison_cache': comparison_identity,
        'wall_seconds': float(time.time() - started),
        'hash_or_sha_validation': False,
        'acceptance_vs_reference': acceptance,
        'acceptance_vs_A1S': acceptance,
    }
    (output_dir / 'isr_m00_audit.json').write_text(
        json.dumps(_json_safe(audit), ensure_ascii=False, indent=2),
        encoding='utf-8')
    # M11 is deliberately gated by this M00-only report, never by Static ETKF.
    development_passed = False
    summary_path = Path(checkpoint).resolve().parent / 'training_summary.json'
    if summary_path.is_file():
        try:
            with summary_path.open(encoding='utf-8') as stream:
                summary = json.load(stream)
            development_passed = bool(
                summary.get('background_development_gate_passed', False))
            if not development_passed:
                development_passed = any(
                    bool(item.get('eligible', False))
                    for item in summary.get('background_selection_audits', [])
                    if isinstance(item, dict))
        except (OSError, ValueError, TypeError):
            development_passed = False

    recommendation = {
        'schema_version': 1,
        'candidate': candidate_name,
        'reference': reference_name if comparison_cache else None,
        'development_formal_gate_passed': development_passed,
        'conditions': {},
        'passed': False,
        'recommendation': 'do_not_train_m11',
        'reason': 'missing_or_failed_development_gate',
    }
    if comparison_cache and reference_name in predictions:
        index = {
            (row['station'], row['stratum'], row['model']): row
            for row in metric_rows
        }
        conditions = {}
        complete = True
        for station in ('Jicamarca', 'PokerFlat'):
            main = index.get((station, 'main_200_500_all', candidate_name))
            ref = index.get((station, 'main_200_500_all', reference_name))
            raw = index.get((station, 'main_200_500_all', 'Raw_IRI'))
            night = index.get((station, 'main_200_500_night', candidate_name))
            night_ref = index.get(
                (station, 'main_200_500_night', reference_name))
            if not all((main, ref, raw, night, night_ref)):
                complete = False
                continue
            conditions[station] = {
                'main_ccc_strict_vs_a1s2': (
                    main['ccc'] is not None and ref['ccc'] is not None
                    and main['ccc'] > ref['ccc']),
                'main_rmse_non_degrade_vs_a1s2': (
                    main['rmse'] is not None and ref['rmse'] is not None
                    and main['rmse'] - ref['rmse'] <= 0.01),
                'main_abs_bias_non_degrade_vs_a1s2': (
                    main['bias'] is not None and ref['bias'] is not None
                    and abs(main['bias']) - abs(ref['bias']) <= 0.01),
                'night_ccc_strict_vs_a1s2': (
                    night['ccc'] is not None and night_ref['ccc'] is not None
                    and night['ccc'] > night_ref['ccc']),
                'night_spread_strict_vs_a1s2': (
                    night['spread_error'] is not None
                    and night_ref['spread_error'] is not None
                    and night['spread_error'] < night_ref['spread_error']),
                'main_rmse_strict_vs_raw_iri': (
                    main['rmse'] is not None and raw['rmse'] is not None
                    and main['rmse'] < raw['rmse']),
                'main_ccc_strict_vs_raw_iri': (
                    main['ccc'] is not None and raw['ccc'] is not None
                    and main['ccc'] > raw['ccc']),
            }
        recommendation['conditions'] = conditions
        gates = [
            value for station in conditions.values() for value in station.values()]
        recommendation['passed'] = bool(
            development_passed and complete and gates and all(gates))
        if recommendation['passed']:
            recommendation['recommendation'] = (
                'train_m11_from_candidate_background')
            recommendation['reason'] = 'development_and_m00_isr_gates_passed'
        elif complete and development_passed:
            recommendation['reason'] = 'm00_isr_gate_failed'
    (output_dir / 'm11_recommendation.json').write_text(
        json.dumps(_json_safe(recommendation), ensure_ascii=False, indent=2),
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
    parser.add_argument('--reference-name', default='M00_A1S',
                        choices=('M00_A1S', 'M00_A1S2'))
    parser.add_argument('--static-etkf-name', default=None,
                        help='public label for the canonical ETKF baseline')
    parser.add_argument('--diagnostic-only', action='store_true',
                        help='mark output as diagnostic; never a promotion')
    parser.add_argument('--allow-unselected-background', action='store_true',
                        help='explicitly evaluate a user-authorized epoch fallback')
    args = parser.parse_args()
    run(args.checkpoint, args.cache, args.output_dir, args.device,
        args.batch_size, args.expected_rows, args.comparison_cache,
        args.candidate_name, args.reference_name, args.diagnostic_only,
        args.static_etkf_name,
        allow_unselected_background=args.allow_unselected_background)


if __name__ == '__main__':
    main()
