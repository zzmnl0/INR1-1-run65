"""Evaluate M2-W epoch checkpoints on both ISR stations in one process."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from isr_evaluation import main_isr_eval as evaluator
from inr_modules.mdia.checkpoint_io import load_fsia_analysis_checkpoint


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_epoch(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(
            torch.isfinite(value).all()
            for value in state.values() if torch.is_tensor(value)):
        raise ValueError(f'non-finite or invalid epoch checkpoint: {path}')
    model.load_state_dict(state, strict=True)
    model.eval()
    return _sha256(path)


def _selection_key(row):
    return (row['mean_point_ccc'], -row['mean_point_rmse'],
            row['mean_point_r'])


def _write_epoch_report(output_dir, reports):
    output_dir.mkdir(parents=True, exist_ok=True)
    from isr_evaluation.plots import save_metrics_report
    save_metrics_report(reports, output_dir / 'isr_validation_report.txt')
    with (output_dir / 'isr_validation_report.json').open(
            'w', encoding='utf-8') as stream:
        json.dump(evaluator._json_safe(reports), stream, ensure_ascii=False,
                  indent=2, allow_nan=False)


def evaluate_sweep(run_dir, output_dir, first_epoch=6, last_epoch=15,
                   preflight_only=False):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    epochs = list(range(first_epoch, last_epoch + 1))
    paths = [run_dir / f'epoch_{epoch:02d}_model.pth' for epoch in epochs]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing epoch checkpoints: {missing}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_checkpoint = run_dir / 'best_fsia_model.pth'
    model, config, _, _ = load_fsia_analysis_checkpoint(
        best_checkpoint, device=device,
        require_domain='strict_200_500_domain_v1')
    epoch_hashes = {
        epoch: _load_epoch(model, path, device)
        for epoch, path in zip(epochs, paths)
    }
    if preflight_only:
        print(f'[preflight] validated epochs {first_epoch:02d}-{last_epoch:02d}')
        return None

    isr_config = dict(evaluator.CONFIG)
    isr_config['checkpoint_path'] = str(best_checkpoint)
    isr_config['alt_min'], isr_config['alt_max'] = map(
        float, config['alt_range'])
    start_unix = evaluator._parse_unix(isr_config['start_date_str'])
    end_unix = evaluator._parse_unix(isr_config['end_date_str'])

    from inr_modules.data_managers.FY_dataloader import (
        COSMICNeighborhoodIndex, FYNeighborhoodIndex)
    from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.mdia.checkpoint_io import allowed_observation_profile_ids
    from isr_evaluation.coord_convert import convert_day_record_cgm
    from isr_evaluation.isr_loader import load_jicamarca, load_poker_flat

    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'], start_date_str=config['start_date_str'],
        total_hours=config['total_hours'], seq_len=config['seq_len'],
        device=device)
    iri_peak_manager = IRIPeakManager(
        config['iri_hmf2_path'], config['iri_nmf2_path'], device=device)
    fy_index = FYNeighborhoodIndex(config['fy_path'], config)
    cosmic_index = COSMICNeighborhoodIndex(config['cosmic_path'], config)
    allowed = allowed_observation_profile_ids(config)
    common_loader = dict(
        start_unix=start_unix, end_unix=end_unix,
        alt_min=isr_config['alt_min'], alt_max=isr_config['alt_max'],
        err_ratio_max=isr_config['err_ratio_max'])
    station_records = {
        'Jicamarca': load_jicamarca(
            data_dir=isr_config['jicamarca_dir'], **common_loader),
        'PokerFlat': load_poker_flat(
            data_dir=isr_config['poker_flat_dir'], **common_loader),
    }
    for record in station_records['PokerFlat']:
        convert_day_record_cgm(record)

    rows = []
    for epoch, checkpoint in zip(epochs, paths):
        print(f'\n[epoch {epoch:02d}] loading {checkpoint}', flush=True)
        _load_epoch(model, checkpoint, device)
        epoch_dir = output_dir / f'epoch_{epoch:02d}'
        epoch_config = dict(isr_config, save_dir=str(epoch_dir),
                            alt_range=config['alt_range'])
        reports = []
        for station, records in station_records.items():
            report = evaluator._process_station(
                station, records, model, sw_manager, start_unix,
                epoch_config, device, model_name=f'M2-W epoch {epoch:02d}',
                iri_peak_manager=iri_peak_manager, fy_nb_index=fy_index,
                cosmic_nb_index=cosmic_index, allowed_profile_ids=allowed)
            if report is not None:
                reports.append(report)
        _write_epoch_report(epoch_dir, reports)
        if len(reports) != 2:
            raise RuntimeError(f'epoch {epoch:02d} lacks two ISR reports')
        row = {
            'epoch': epoch,
            'checkpoint_sha256': epoch_hashes[epoch],
            **{
                f'{report["station"]}_{metric}': float(report[metric])
                for report in reports
                for metric in ('point_ccc', 'point_rmse', 'point_r')
            },
            'mean_point_ccc': float(np.mean([
                report['point_ccc'] for report in reports])),
            'mean_point_rmse': float(np.mean([
                report['point_rmse'] for report in reports])),
            'mean_point_r': float(np.mean([
                report['point_r'] for report in reports])),
        }
        rows.append(row)

    best = max(rows, key=_selection_key)
    summary = {
        'model_domain_semantics': config['model_domain_semantics'],
        'alt_range': list(map(float, config['alt_range'])),
        'epochs': rows,
        'selection': 'mean_station_point_ccc_then_rmse_then_pearson_v1',
        'best_epoch': best['epoch'],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'epoch_isr_sweep_summary.json').open(
            'w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
    with (output_dir / 'epoch_isr_sweep_summary.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[selection] best ISR CCC epoch={best["epoch"]:02d}', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--first-epoch', type=int, default=6)
    parser.add_argument('--last-epoch', type=int, default=15)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    evaluate_sweep(args.run_dir, args.output_dir, args.first_epoch,
                   args.last_epoch, args.preflight_only)
