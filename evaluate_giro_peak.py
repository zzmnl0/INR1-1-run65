"""Evaluate M2-W Analysis, Background, and Raw IRI GIRO peaks independently."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

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
            (n_records, n_prediction, n_metrics, 'NmF2 (log10)'))):
        truth = records[:, 3]
        for column, source in enumerate(('IRI', 'M00', 'M11')):
            prediction = predictions[source][
                'hmf2' if row == 0 else 'nmf2']
            finite = np.isfinite(truth) & np.isfinite(prediction)
            x, y = truth[finite], prediction[finite]
            lower, upper = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
            image = axes[row, column].hist2d(
                x, y, bins=100, range=[[lower, upper], [lower, upper]],
                cmap='turbo', norm=LogNorm(), cmin=1)
            axes[row, column].plot([lower, upper], [lower, upper], 'w--', lw=1)
            value = metrics[source]
            axes[row, column].set_title(
                f'{source} {quantity}\nCCC={value["ccc"]:.4f}  '
                f'RMSE={value["rmse"]:.3f}  R={value["pearson_r"]:.4f}')
            axes[row, column].set_xlabel(f'GIRO {quantity}')
            axes[row, column].set_ylabel(f'Prediction {quantity}')
            fig.colorbar(image[3], ax=axes[row, column], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def evaluate_giro_peak(checkpoint, save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, _, summary = load_fsia_analysis_checkpoint(
        checkpoint, device=device,
        require_domain='strict_200_500_domain_v1')
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
    allowed = allowed_observation_profile_ids(config)
    managers = (fy_index, cosmic_index, sw_manager, iri_peak_manager)

    h_records = np.load(config['giro_hmf2_path']).astype(np.float32)
    n_records = np.load(config['giro_nmf2_path']).astype(np.float32)
    print('[GIRO] independent hmF2 profiles')
    h_prediction = _predict_peaks(
        h_records, model, managers, allowed, device, config['alt_range'])
    print('[GIRO] independent NmF2 profiles')
    n_prediction = _predict_peaks(
        n_records, model, managers, allowed, device, config['alt_range'])
    h_metrics, h_bootstrap = _evaluate_quantity(
        h_records, 'hmf2', h_prediction)
    n_metrics, n_bootstrap = _evaluate_quantity(
        n_records, 'nmf2', n_prediction)
    report = {
        'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': summary['checkpoint_sha256'],
        'model_domain_semantics': config['model_domain_semantics'],
        'alt_range': list(map(float, config['alt_range'])),
        'token_partitions': ['train', 'development'],
        'peak_search': 'independent_coarse_and_fine_per_field_v1',
        'hmF2': {'metrics': h_metrics,
                 'm11_vs_raw_iri_bootstrap': h_bootstrap},
        'NmF2': {'metrics': n_metrics,
                 'm11_vs_raw_iri_bootstrap': n_bootstrap},
        'passed_m2w_m11_vs_raw_iri_gate': (
            h_bootstrap['decision'] == 'pass'
            and n_bootstrap['decision'] == 'pass'),
    }
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
    _plot_density(
        save_dir / 'giro_peak_density.png', h_records, n_records,
        h_prediction, n_prediction, h_metrics, n_metrics)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save-dir')
    arguments = parser.parse_args()
    evaluate_giro_peak(arguments.checkpoint, arguments.save_dir)
