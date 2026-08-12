"""Evaluate M2-W Analysis, Background, and Raw IRI GIRO peaks independently."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from inr_modules.density_units import (
    DENSITY_UNIT_LABEL,
    log10_density_to_display,
)
from inr_modules.data_managers.FY_dataloader import (
    COSMICNeighborhoodIndex, FYNeighborhoodIndex)
from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
from inr_modules.mdia.checkpoint_io import (
    allowed_observation_profile_ids,
    load_fsia_analysis_checkpoint,
)
from inr_modules.mdia.evaluation_stats import (
    paired_group_bootstrap,
    regression_metrics,
)
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    query_observation_payload,
)


_COARSE_STEP = 10.0
_FINE_HALF = 10
_STATION_BATCH = 8
_M2W_DOMAINS = {
    'strict_200_500_domain_v1',
    'hybrid_120_500_model_200_500_observation_v1',
}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _metrics(observation, prediction):
    values = regression_metrics(observation, prediction)
    finite = np.isfinite(observation) & np.isfinite(prediction)
    error = prediction[finite] - observation[finite]
    values.update({
        'bias': float(np.mean(error)) if len(error) else np.nan,
        'mae': float(np.mean(np.abs(error))) if len(error) else np.nan,
    })
    return values


def _record_ids(records):
    return np.asarray([
        f'{lat:.5f}|{lon:.5f}|{rel_hour:.5f}'
        for lat, lon, rel_hour in records[:, :3]
    ])


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _require_m2w_peak_contract(config):
    domain = config.get('model_domain_semantics')
    if domain not in _M2W_DOMAINS:
        raise ValueError(f'GIRO peak evaluation requires an M2-W checkpoint, got {domain}')
    observation = tuple(map(float, config.get('observation_alt_range')
                                or config.get('alt_range', ())))
    peak = tuple(map(float, config.get('peak_search_alt_range')
                         or config.get('alt_range', ())))
    if observation != (200.0, 500.0) or peak != (200.0, 500.0):
        raise ValueError('M2-W GIRO requires 200-500 km observation and peak domains')
    return peak


def _checkpoint_contract(checkpoint, config, summary):
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint.parent / 'run_manifest.json'
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    return {
        'path': str(checkpoint),
        'sha256': _sha256(checkpoint),
        'checkpoint_format_version': int(config['checkpoint_format_version']),
        'model_domain_semantics': config['model_domain_semantics'],
        'model_alt_range_km': list(map(float, config['alt_range'])),
        'observation_alt_range_km': list(map(float, (
            config.get('observation_alt_range') or config['alt_range']))),
        'peak_search_alt_range_km': list(map(float, (
            config.get('peak_search_alt_range') or config['alt_range']))),
        'date_split': summary.get('date_split'),
        'input_data_identity': manifest.get('data_identity'),
    }


def _query_fields(coords_np, model, sw_manager, iri_peak_manager,
                  fy_index, cosmic_index, allowed, device, batch_size=2048):
    outputs = {name: [] for name in ('M11', 'M00', 'IRI')}
    with torch.no_grad():
        for start in range(0, len(coords_np), batch_size):
            coords = torch.from_numpy(
                coords_np[start:start + batch_size]).to(device)
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            iri_peak = iri_peak_manager.get_iri_peak(coords)
            observations = {}
            for name, index in (('fy', fy_index), ('cosmic', cosmic_index)):
                payload = query_observation_payload(
                    index, coords, device,
                    allowed_profile_ids=allowed[name.upper()])
                observations[name] = attach_observation_background(
                    payload, model, sw_manager, iri_peak_manager)
            analysis, _, _, _, extras = model(
                coords, sw_seq, iri_peak=iri_peak,
                observations_fy=observations['fy'],
                observations_cosmic=observations['cosmic'])
            outputs['M11'].append(analysis.squeeze(-1).cpu().numpy())
            outputs['M00'].append(extras['ne_bkg'].squeeze(-1).cpu().numpy())
            outputs['IRI'].append(extras['ne_iri'].squeeze(-1).cpu().numpy())
    return {name: np.concatenate(chunks) for name, chunks in outputs.items()}


def _query_fields_many(coords_np, models, sw_manager, iri_peak_manager,
                       fy_index, cosmic_index, allowed_by_model, device,
                       batch_size=2048):
    """Infer candidate and baseline in one coordinate/token loop."""
    reference_allowed = next(iter(allowed_by_model.values()))
    for allowed in allowed_by_model.values():
        for source in ('FY', 'COSMIC'):
            if not np.array_equal(allowed[source], reference_allowed[source]):
                raise ValueError(
                    'paired GIRO inference requires identical train+development '
                    f'{source} token partitions')
    outputs = {
        label: {name: [] for name in ('M11', 'M00', 'IRI')}
        for label in models
    }
    with torch.no_grad():
        for start in range(0, len(coords_np), batch_size):
            coords = torch.from_numpy(
                coords_np[start:start + batch_size]).to(device)
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            iri_peak = iri_peak_manager.get_iri_peak(coords)
            raw_observations = {
                name: query_observation_payload(
                    index, coords, device,
                    allowed_profile_ids=reference_allowed[name.upper()])
                for name, index in (('fy', fy_index), ('cosmic', cosmic_index))
            }
            for label, model in models.items():
                observations = {
                    name: attach_observation_background(
                        payload,
                        model, sw_manager, iri_peak_manager)
                    for name, payload in raw_observations.items()
                }
                analysis, _, _, _, extras = model(
                    coords, sw_seq, iri_peak=iri_peak,
                    observations_fy=observations['fy'],
                    observations_cosmic=observations['cosmic'])
                outputs[label]['M11'].append(analysis.squeeze(-1).cpu().numpy())
                outputs[label]['M00'].append(
                    extras['ne_bkg'].squeeze(-1).cpu().numpy())
                outputs[label]['IRI'].append(
                    extras['ne_iri'].squeeze(-1).cpu().numpy())
    return {
        label: {name: np.concatenate(chunks) for name, chunks in fields.items()}
        for label, fields in outputs.items()
    }


def _predict_peaks(records, model, managers, allowed, device, alt_range):
    lower, upper = map(float, alt_range)
    coarse_altitudes = np.arange(
        lower, upper + 0.5 * _COARSE_STEP, _COARSE_STEP,
        dtype=np.float32)
    fine_offsets = np.arange(
        -_FINE_HALF, _FINE_HALF + 1, dtype=np.float32)
    result = {
        source: {
            'hmf2': np.full(len(records), np.nan, dtype=np.float32),
            'nmf2': np.full(len(records), np.nan, dtype=np.float32),
        }
        for source in ('M11', 'M00', 'IRI')
    }
    fy_index, cosmic_index, sw_manager, iri_peak_manager = managers
    for start in range(0, len(records), _STATION_BATCH):
        selected = np.arange(start, min(start + _STATION_BATCH, len(records)))
        rows = records[selected]
        n_station, n_coarse = len(rows), len(coarse_altitudes)
        coarse_coords = np.column_stack([
            np.repeat(rows[:, 0], n_coarse),
            np.repeat(rows[:, 1], n_coarse),
            np.tile(coarse_altitudes, n_station),
            np.repeat(rows[:, 2], n_coarse),
        ]).astype(np.float32)
        coarse = _query_fields(
            coarse_coords, model, sw_manager, iri_peak_manager,
            fy_index, cosmic_index, allowed, device)
        coarse_peak = {
            source: coarse_altitudes[
                values.reshape(n_station, n_coarse).argmax(axis=1)]
            for source, values in coarse.items()
        }

        sources = ('M11', 'M00', 'IRI')
        fine_altitudes = np.stack([
            np.clip(coarse_peak[source][:, None] + fine_offsets[None, :],
                    lower, upper)
            for source in sources
        ]).astype(np.float32)
        n_fine = len(fine_offsets)
        fine_coords = np.column_stack([
            np.tile(np.repeat(rows[:, 0], n_fine), len(sources)),
            np.tile(np.repeat(rows[:, 1], n_fine), len(sources)),
            fine_altitudes.reshape(-1),
            np.tile(np.repeat(rows[:, 2], n_fine), len(sources)),
        ]).astype(np.float32)
        fine = _query_fields(
            fine_coords, model, sw_manager, iri_peak_manager,
            fy_index, cosmic_index, allowed, device)
        for source_index, source in enumerate(sources):
            values = fine[source].reshape(len(sources), n_station, n_fine)[
                source_index]
            peak_index = values.argmax(axis=1)
            result[source]['hmf2'][selected] = fine_altitudes[
                source_index, np.arange(n_station), peak_index]
            result[source]['nmf2'][selected] = values[
                np.arange(n_station), peak_index]
        print(f'  {selected[-1] + 1:>6}/{len(records)} records')
    return result


def _predict_peaks_many(records, models, managers, allowed_by_model, device,
                        alt_range):
    """Run independent F2 searches for each model in the same record loop."""
    lower, upper = map(float, alt_range)
    coarse_altitudes = np.arange(
        lower, upper + 0.5 * _COARSE_STEP, _COARSE_STEP, dtype=np.float32)
    fine_offsets = np.arange(-_FINE_HALF, _FINE_HALF + 1, dtype=np.float32)
    labels = tuple(models)
    sources = ('M11', 'M00', 'IRI')
    result = {
        label: {
            source: {
                'hmf2': np.full(len(records), np.nan, dtype=np.float32),
                'nmf2': np.full(len(records), np.nan, dtype=np.float32),
            }
            for source in sources
        }
        for label in labels
    }
    fy_index, cosmic_index, sw_manager, iri_peak_manager = managers
    for start in range(0, len(records), _STATION_BATCH):
        selected = np.arange(start, min(start + _STATION_BATCH, len(records)))
        rows = records[selected]
        n_station, n_coarse = len(rows), len(coarse_altitudes)
        coarse_coords = np.column_stack([
            np.repeat(rows[:, 0], n_coarse),
            np.repeat(rows[:, 1], n_coarse),
            np.tile(coarse_altitudes, n_station),
            np.repeat(rows[:, 2], n_coarse),
        ]).astype(np.float32)
        coarse = _query_fields_many(
            coarse_coords, models, sw_manager, iri_peak_manager, fy_index,
            cosmic_index, allowed_by_model, device)
        coarse_peak = {
            label: {
                source: coarse_altitudes[coarse[label][source].reshape(
                    n_station, n_coarse).argmax(axis=1)]
                for source in sources
            }
            for label in labels
        }
        n_fine = len(fine_offsets)
        fine_altitudes = {}
        coordinate_groups = []
        for label in labels:
            for source in sources:
                altitudes = np.clip(
                    coarse_peak[label][source][:, None] + fine_offsets[None, :],
                    lower, upper).astype(np.float32)
                fine_altitudes[label, source] = altitudes
                coordinate_groups.append(np.column_stack([
                    np.repeat(rows[:, 0], n_fine),
                    np.repeat(rows[:, 1], n_fine),
                    altitudes.reshape(-1),
                    np.repeat(rows[:, 2], n_fine),
                ]).astype(np.float32))
        fine = _query_fields_many(
            np.concatenate(coordinate_groups), models, sw_manager,
            iri_peak_manager, fy_index, cosmic_index, allowed_by_model, device)
        group_size = n_station * n_fine
        for label_index, label in enumerate(labels):
            for source_index, source in enumerate(sources):
                group_index = label_index * len(sources) + source_index
                values = fine[label][source][
                    group_index * group_size:(group_index + 1) * group_size
                ].reshape(n_station, n_fine)
                peak_index = values.argmax(axis=1)
                result[label][source]['hmf2'][selected] = fine_altitudes[
                    label, source][np.arange(n_station), peak_index]
                result[label][source]['nmf2'][selected] = values[
                    np.arange(n_station), peak_index]
        print(f'  {selected[-1] + 1:>6}/{len(records)} records')
    return result


def _evaluate_quantity(records, field, predictions):
    observation = records[:, 3].astype(np.float64)
    units = _record_ids(records)
    metrics = {
        source: _metrics(observation, values[field].astype(np.float64))
        for source, values in predictions.items()
    }
    bootstrap = paired_group_bootstrap(
        observation, predictions['M11'][field], predictions['IRI'][field],
        units, replicates=2000, seed=42)
    return metrics, bootstrap


def _plot_density(path, h_records, n_records, h_prediction, n_prediction,
                  h_metrics, n_metrics):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), dpi=150)
    for row, (records, predictions, metrics, quantity) in enumerate((
            (h_records, h_prediction, h_metrics, 'hmF2 (km)'),
            (n_records, n_prediction, n_metrics,
             f'NmF2 ({DENSITY_UNIT_LABEL})'))):
        truth = records[:, 3]
        for column, source in enumerate(('IRI', 'M00', 'M11')):
            prediction = predictions[source][
                'hmf2' if row == 0 else 'nmf2']
            finite = np.isfinite(truth) & np.isfinite(prediction)
            x, y = truth[finite], prediction[finite]
            if row == 1:
                x = log10_density_to_display(x)
                y = log10_density_to_display(y)
            lower, upper = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
            image = axes[row, column].hist2d(
                x, y, bins=100, range=[[lower, upper], [lower, upper]],
                cmap='turbo', norm=LogNorm(), cmin=1)
            axes[row, column].plot([lower, upper], [lower, upper], 'w--', lw=1)
            value = metrics[source]
            metric_label = 'log10-space ' if row == 1 else ''
            axes[row, column].set_title(
                f'{source} {quantity}\n{metric_label}CCC={value["ccc"]:.4f}  '
                f'RMSE={value["rmse"]:.3f}  R={value["pearson_r"]:.4f}')
            axes[row, column].set_xlabel(f'GIRO {quantity}')
            axes[row, column].set_ylabel(f'Prediction {quantity}')
            fig.colorbar(image[3], ax=axes[row, column], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _load_model_context(checkpoint, device):
    model, config, _, summary = load_fsia_analysis_checkpoint(
        checkpoint, device=device, allow_historical_epoch=True)
    peak_range = _require_m2w_peak_contract(config)
    return model, config, summary, peak_range, allowed_observation_profile_ids(config)


def _paired_model_bootstrap(records, field, candidate, baseline):
    observation = records[:, 3].astype(np.float64)
    units = _record_ids(records)
    return paired_group_bootstrap(
        observation, candidate['M11'][field], baseline['M11'][field], units,
        replicates=2000, seed=42)


def evaluate_giro_peak(checkpoint, save_dir=None, baseline_checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, summary, peak_range, allowed = _load_model_context(
        checkpoint, device)
    candidate_contract = _checkpoint_contract(checkpoint, config, summary)
    baseline_model = baseline_config = baseline_summary = baseline_contract = None
    baseline_allowed = None
    if baseline_checkpoint is not None:
        (baseline_model, baseline_config, baseline_summary,
         baseline_peak_range, baseline_allowed) = _load_model_context(
            baseline_checkpoint, device)
        if baseline_peak_range != peak_range:
            raise ValueError('candidate and baseline peak-search domains differ')
        if (config['fy_path'] != baseline_config['fy_path']
                or config['cosmic_path'] != baseline_config['cosmic_path']
                or config['sw_path'] != baseline_config['sw_path']
                or config['iri_hmf2_path'] != baseline_config['iri_hmf2_path']
                or config['iri_nmf2_path'] != baseline_config['iri_nmf2_path']):
            raise ValueError('paired GIRO evaluation requires shared FY/COSMIC/IRI inputs')
        baseline_contract = _checkpoint_contract(
            baseline_checkpoint, baseline_config, baseline_summary)
    save_dir = Path(save_dir or Path(checkpoint).resolve().parent / 'giro_peak_eval')
    save_dir.mkdir(parents=True, exist_ok=True)

    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'], start_date_str=config['start_date_str'],
        total_hours=config['total_hours'], seq_len=config['seq_len'],
        device=device)
    iri_peak_manager = IRIPeakManager(
        config['iri_hmf2_path'], config['iri_nmf2_path'], device=str(device))
    fy_index = FYNeighborhoodIndex(config['fy_path'], config)
    cosmic_index = COSMICNeighborhoodIndex(config['cosmic_path'], config)
    managers = (fy_index, cosmic_index, sw_manager, iri_peak_manager)

    h_records = np.load(config['giro_hmf2_path']).astype(np.float32)
    n_records = np.load(config['giro_nmf2_path']).astype(np.float32)
    if baseline_model is None:
        print('[GIRO] independent hmF2 profiles')
        h_prediction = _predict_peaks(
            h_records, model, managers, allowed, device, peak_range)
        print('[GIRO] independent NmF2 profiles')
        n_prediction = _predict_peaks(
            n_records, model, managers, allowed, device, peak_range)
        baseline_h_prediction = baseline_n_prediction = None
    else:
        models = {'candidate': model, 'baseline': baseline_model}
        allowed_by_model = {'candidate': allowed, 'baseline': baseline_allowed}
        print('[GIRO] paired independent hmF2 profiles')
        h_predictions = _predict_peaks_many(
            h_records, models, managers, allowed_by_model, device, peak_range)
        print('[GIRO] paired independent NmF2 profiles')
        n_predictions = _predict_peaks_many(
            n_records, models, managers, allowed_by_model, device, peak_range)
        h_prediction, baseline_h_prediction = (
            h_predictions['candidate'], h_predictions['baseline'])
        n_prediction, baseline_n_prediction = (
            n_predictions['candidate'], n_predictions['baseline'])
    h_metrics, h_bootstrap = _evaluate_quantity(
        h_records, 'hmf2', h_prediction)
    n_metrics, n_bootstrap = _evaluate_quantity(
        n_records, 'nmf2', n_prediction)
    comparisons = None
    if baseline_h_prediction is not None:
        baseline_h_metrics, _ = _evaluate_quantity(
            h_records, 'hmf2', baseline_h_prediction)
        baseline_n_metrics, _ = _evaluate_quantity(
            n_records, 'nmf2', baseline_n_prediction)
        comparisons = {
            'hmF2_m11_candidate_vs_baseline_bootstrap': _paired_model_bootstrap(
                h_records, 'hmf2', h_prediction, baseline_h_prediction),
            'NmF2_m11_candidate_vs_baseline_bootstrap': _paired_model_bootstrap(
                n_records, 'nmf2', n_prediction, baseline_n_prediction),
            'baseline_hmF2_metrics': baseline_h_metrics,
            'baseline_NmF2_metrics': baseline_n_metrics,
        }
    contract = {
        'schema_version': 14,
        'token_partitions': ['train', 'development'],
        'candidate_checkpoint': candidate_contract,
        'baseline_checkpoint': baseline_contract,
        'quality_thresholds': {'finite_peak_required': True},
        'peak_search': {
            'semantics': 'independent_coarse_10km_then_fine_1km_per_field_v1',
            'alt_range_km': list(peak_range),
            'coarse_step_km': _COARSE_STEP,
            'fine_half_window_km': _FINE_HALF,
            'bootstrap': {'replicates': 2000, 'seed': 42,
                          'group': 'station_time_record'},
        },
    }
    report = {
        'checkpoint': candidate_contract['path'],
        'checkpoint_sha256': candidate_contract['sha256'],
        'model_domain_semantics': config['model_domain_semantics'],
        'alt_range': list(map(float, config['alt_range'])),
        'observation_alt_range': list(map(float, (
            config.get('observation_alt_range') or config['alt_range']))),
        'peak_search_alt_range': list(peak_range),
        'token_partitions': contract['token_partitions'],
        'peak_search': contract['peak_search']['semantics'],
        'hmF2': {'metrics': h_metrics,
                 'm11_vs_raw_iri_bootstrap': h_bootstrap},
        'NmF2': {'metrics': n_metrics,
                 'm11_vs_raw_iri_bootstrap': n_bootstrap},
        'candidate_vs_baseline': comparisons,
        'passed_m2w_m11_vs_raw_iri_gate': (
            h_bootstrap['decision'] == 'pass'
            and n_bootstrap['decision'] == 'pass'),
    }
    cache = {
        'hm_record_id': _record_ids(h_records),
        'nm_record_id': _record_ids(n_records),
        'hm_observation': h_records[:, 3],
        'nm_observation_log10': n_records[:, 3],
    }
    for prefix, prediction in (('candidate_hm', h_prediction),
                               ('candidate_nm', n_prediction)):
        field = 'hmf2' if prefix.endswith('hm') else 'nmf2'
        for source, values in prediction.items():
            cache[f'{prefix}_{source}'] = values[field]
    if baseline_h_prediction is not None:
        for prefix, prediction in (('baseline_hm', baseline_h_prediction),
                                   ('baseline_nm', baseline_n_prediction)):
            field = 'hmf2' if prefix.endswith('hm') else 'nmf2'
            for source, values in prediction.items():
                cache[f'{prefix}_{source}'] = values[field]
    np.savez_compressed(save_dir / 'giro_peak_cache.npz', **cache)
    with (save_dir / 'giro_peak_contract.json').open('w', encoding='utf-8') as stream:
        json.dump(_json_safe(contract), stream, ensure_ascii=False,
                  indent=2, allow_nan=False)
    with (save_dir / 'giro_peak_report.json').open('w', encoding='utf-8') as stream:
        json.dump(_json_safe(report), stream, ensure_ascii=False,
                  indent=2, allow_nan=False)
    with (save_dir / 'giro_peak_report.txt').open('w', encoding='utf-8') as stream:
        for quantity in ('hmF2', 'NmF2'):
            stream.write(f'[{quantity}]\n')
            for source, values in report[quantity]['metrics'].items():
                stream.write(
                    f'{source}: CCC={values["ccc"]:.6f} '
                    f'RMSE={values["rmse"]:.6f} '
                    f'R={values["pearson_r"]:.6f}\n')
            stream.write(json.dumps(
                _json_safe(report[quantity]['m11_vs_raw_iri_bootstrap']),
                ensure_ascii=False) + '\n')
        if comparisons is not None:
            stream.write('[candidate_vs_baseline]\n')
            stream.write(json.dumps(_json_safe(comparisons), ensure_ascii=False) + '\n')
    _plot_density(
        save_dir / 'giro_peak_density.png', h_records, n_records,
        h_prediction, n_prediction, h_metrics, n_metrics)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save-dir')
    parser.add_argument('--baseline-checkpoint')
    arguments = parser.parse_args()
    evaluate_giro_peak(arguments.checkpoint, arguments.save_dir,
                       arguments.baseline_checkpoint)
