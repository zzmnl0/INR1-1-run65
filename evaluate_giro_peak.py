"""Evaluate M2-W Analysis, Background, and Raw IRI GIRO peaks independently."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

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
    shared_full30_profile_allowlist,
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
from isr_evaluation.peak_qa import (
    DEFAULT_PEAK_CONTRACT,
    PeakSearchContract,
    results_to_arrays,
    search_peak_profile,
)


_COARSE_STEP = 10.0
_FINE_HALF = 10
_STATION_BATCH = 8
_RESUME_SCHEMA = 'shared_full30_batch_resume_v1'
_RESUME_CHECKPOINT_RECORDS = 1024
_PEAK_CACHE_FIELDS = (
    'nmf2_log10', 'hmf2_km', 'status', 'censoring', 'nmf2_valid',
    'hmf2_valid', 'prominence_dex', 'secondary_peak_delta_dex',
    'n_finite_levels', 'support_lower_km', 'support_upper_km',
    'max_local_gap_km', 'hmf2', 'nmf2', 'legacy_hmf2', 'legacy_nmf2',
)
_M2W_DOMAINS = {
    'strict_200_500_domain_v1',
    'hybrid_120_500_model_200_500_observation_v1',
}
_HISTORICAL_EPOCH12_SHA256 = (
    '486ffe73722cde1ff2909da93898e02d4d1e173700c9e4fa9dd2f0990e2fe56a')


def _peak_contract(alt_range):
    lower, upper = map(float, alt_range)
    if (lower, upper) != (200.0, 500.0):
        raise ValueError('P0-A GIRO peak QA is fixed to 200--500 km')
    return PeakSearchContract(
        lower_km=lower,
        upper_km=upper,
        coarse_step_km=DEFAULT_PEAK_CONTRACT.coarse_step_km,
        fine_step_km=DEFAULT_PEAK_CONTRACT.fine_step_km,
        fine_half_window_km=DEFAULT_PEAK_CONTRACT.fine_half_window_km,
        min_finite_levels=DEFAULT_PEAK_CONTRACT.min_finite_levels,
        max_local_gap_km=DEFAULT_PEAK_CONTRACT.max_local_gap_km,
        flank_support_km=DEFAULT_PEAK_CONTRACT.flank_support_km,
        prominence_dex=DEFAULT_PEAK_CONTRACT.prominence_dex,
        secondary_separation_km=DEFAULT_PEAK_CONTRACT.secondary_separation_km,
        near_tie_dex=DEFAULT_PEAK_CONTRACT.near_tie_dex,
        boundary_margin_km=DEFAULT_PEAK_CONTRACT.boundary_margin_km,
    )


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


def _safe_paired_bootstrap(observation, candidate, baseline, unit_ids):
    finite = (np.isfinite(observation) & np.isfinite(candidate)
              & np.isfinite(baseline))
    if finite.sum() < 2 or np.unique(np.asarray(unit_ids)[finite]).size < 2:
        return {
            'status': 'insufficient_data',
            'n': int(finite.sum()),
            'sampling_units': int(np.unique(np.asarray(unit_ids)[finite]).size),
            'decision': 'inconclusive',
        }
    return paired_group_bootstrap(
        np.asarray(observation)[finite], np.asarray(candidate)[finite],
        np.asarray(baseline)[finite], np.asarray(unit_ids)[finite],
        replicates=2000, seed=42)


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


def _checkpoint_contract(checkpoint, config, summary, verify_hashes=True):
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint.parent / 'run_manifest.json'
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    return {
        'path': str(checkpoint),
        'sha256': _sha256(checkpoint) if verify_hashes else None,
        'checkpoint_format_version': int(config['checkpoint_format_version']),
        'model_domain_semantics': config['model_domain_semantics'],
        'model_alt_range_km': list(map(float, config['alt_range'])),
        'observation_alt_range_km': list(map(float, (
            config.get('observation_alt_range') or config['alt_range']))),
        'peak_search_alt_range_km': list(map(float, (
            config.get('peak_search_alt_range') or config['alt_range']))),
        'date_split': summary.get('date_split'),
        'input_data_identity': manifest.get('data_identity') if verify_hashes else None,
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


def _pack_peak_results(results, legacy_hmf2, legacy_nmf2):
    packed = results_to_arrays(results)
    # Preserve the old finite argmax result solely for QA-v2/legacy comparison.
    packed.update({
        'hmf2': packed['hmf2_km'],
        'nmf2': packed['nmf2_log10'],
        'legacy_hmf2': np.asarray(legacy_hmf2, dtype=np.float32),
        'legacy_nmf2': np.asarray(legacy_nmf2, dtype=np.float32),
    })
    return packed


def _atomic_savez(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resume_metadata(phase, records, labels, sources, peak_contract,
                     resume_contract, checkpoint_records):
    metadata = {
        'schema': _RESUME_SCHEMA,
        'phase': str(phase),
        'record_count': int(len(records)),
        'labels': list(labels),
        'sources': list(sources),
        'station_batch': int(_STATION_BATCH),
        'checkpoint_records': int(checkpoint_records),
        'peak_contract': peak_contract.as_dict(),
        'evaluation_contract': resume_contract,
    }
    return json.dumps(_json_safe(metadata), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    # ponytail: size/mtime avoids hashing large inputs; use content hashes only
    # if the no-hash execution contract is later relaxed.
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _prediction_payload(predictions, expected_count):
    payload = {}
    for label, source_predictions in predictions.items():
        for source, values in source_predictions.items():
            missing = set(_PEAK_CACHE_FIELDS).difference(values)
            if missing:
                raise ValueError(
                    f'GIRO resume prediction lacks fields: {sorted(missing)}')
            for key in _PEAK_CACHE_FIELDS:
                value = np.asarray(values[key])
                if value.ndim == 0 or value.shape[0] != expected_count:
                    raise ValueError(
                        f'GIRO resume prediction {label}/{source}/{key} length '
                        f'mismatch: {value.shape} vs {expected_count}')
                payload[f'prediction__{label}__{source}__{key}'] = value
    return payload


def _prediction_from_npz(data, labels, sources, expected_count):
    predictions = {}
    for label in labels:
        predictions[label] = {}
        for source in sources:
            values = {}
            for key in _PEAK_CACHE_FIELDS:
                name = f'prediction__{label}__{source}__{key}'
                if name not in data:
                    raise ValueError(f'GIRO resume cache lacks {name}')
                value = np.asarray(data[name])
                if value.ndim == 0 or value.shape[0] != expected_count:
                    raise ValueError(
                        f'GIRO resume cache {name} length mismatch: '
                        f'{value.shape} vs {expected_count}')
                values[key] = value
            predictions[label][source] = values
    return predictions


def _save_prediction_cache(path, metadata, records, predictions, start, end):
    records = np.asarray(records)
    expected_count = int(end - start)
    if records.ndim != 2 or records.shape[0] != expected_count:
        raise ValueError(
            f'GIRO resume records shape mismatch: {records.shape} vs '
            f'{expected_count}')
    payload = {
        'metadata_json': np.asarray(metadata),
        'start': np.asarray(int(start), dtype=np.int64),
        'end': np.asarray(int(end), dtype=np.int64),
        'records': records,
    }
    payload.update(_prediction_payload(predictions, expected_count))
    _atomic_savez(path, payload)


def _load_prediction_cache(path, metadata, records, labels, sources,
                           start, end):
    expected_records = np.asarray(records)
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data['metadata_json']).item()) != metadata:
            raise ValueError(f'GIRO resume metadata mismatch: {path}')
        stored_start = int(np.asarray(data['start']).item())
        stored_end = int(np.asarray(data['end']).item())
        if (stored_start, stored_end) != (int(start), int(end)):
            raise ValueError(f'GIRO resume range mismatch: {path}')
        stored_records = np.asarray(data['records'])
        if not np.array_equal(stored_records, expected_records, equal_nan=True):
            raise ValueError(f'GIRO resume records mismatch: {path}')
        return _prediction_from_npz(
            data, labels, sources, expected_count=int(end - start))


def _concat_prediction_chunks(chunks, labels, sources):
    if not chunks:
        raise ValueError('GIRO resume has no prediction chunks')
    return {
        label: {
            source: {
                key: np.concatenate([
                    chunk[label][source][key] for chunk in chunks], axis=0)
                for key in _PEAK_CACHE_FIELDS
            }
            for source in sources
        }
        for label in labels
    }


def _load_resume_chunks(resume_root, phase, records, metadata, labels, sources):
    resume_root = Path(resume_root)
    complete_path = resume_root / f'{phase}_complete.npz'
    if complete_path.is_file():
        prediction = _load_prediction_cache(
            complete_path, metadata, records, labels, sources,
            0, len(records))
        return len(records), [prediction], True

    phase_dir = resume_root / phase
    chunks, completed = [], 0
    if not phase_dir.exists():
        return completed, chunks, False
    shards = []
    for path in phase_dir.glob('part_*.npz'):
        parts = path.stem.split('_')
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            raise ValueError(f'invalid GIRO resume shard name: {path.name}')
        start, end = int(parts[1]), int(parts[2])
        shards.append((start, end, path))
    for start, end, path in sorted(shards):
        if start != completed or end <= start or end > len(records):
            raise ValueError(f'non-contiguous GIRO resume shard: {path.name}')
        chunks.append(_load_prediction_cache(
            path, metadata, records[start:end], labels, sources, start, end))
        completed = end
    if completed < len(records) and completed % _STATION_BATCH:
        raise ValueError('GIRO resume shard ends inside a station batch')
    return completed, chunks, False


def _load_complete_phase(resume_root, phase):
    """Load a published complete cache without initializing model inference."""
    path = Path(resume_root) / f'{phase}_complete.npz'
    if not path.is_file():
        raise FileNotFoundError(f'GIRO complete cache is missing: {path}')
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data['metadata_json']).item()))
        if metadata.get('schema') != _RESUME_SCHEMA:
            raise ValueError(f'GIRO complete cache schema mismatch: {path}')
        if metadata.get('phase') != phase:
            raise ValueError(f'GIRO complete cache phase mismatch: {path}')
        labels = tuple(metadata.get('labels', ()))
        sources = tuple(metadata.get('sources', ()))
        if labels != ('A1S2', 'A1S') or sources != ('M11', 'M00', 'IRI'):
            raise ValueError(f'GIRO complete cache model/source mismatch: {path}')
        records = np.asarray(data['records'])
        count = int(metadata.get('record_count', -1))
        if records.ndim != 2 or records.shape[0] != count:
            raise ValueError(f'GIRO complete cache record count mismatch: {path}')
        if (int(np.asarray(data['start']).item()),
                int(np.asarray(data['end']).item())) != (0, count):
            raise ValueError(f'GIRO complete cache range mismatch: {path}')
        predictions = _prediction_from_npz(data, labels, sources, count)
    return records, predictions, metadata


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2,
                      allow_nan=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_postprocess_status(save_dir, stage, completed):
    print(f'[GIRO postprocess] {stage}')
    _atomic_write_json(Path(save_dir) / 'giro_peak_postprocess_status.json', {
        'schema': 'shared_full30_postprocess_v1',
        'stage': stage,
        'completed_stages': list(completed),
        'status': 'complete' if stage == 'complete' else 'running',
    })


def _predict_peaks_many_resumable(
        records, models, managers, allowed_by_model, device, alt_range,
        resume_root, phase, resume_contract,
        checkpoint_records=_RESUME_CHECKPOINT_RECORDS):
    if resume_root is None:
        return _predict_peaks_many(
            records, models, managers, allowed_by_model, device, alt_range)
    if checkpoint_records <= 0 or checkpoint_records % _STATION_BATCH:
        raise ValueError(
            f'GIRO resume checkpoint size must be a positive multiple of '
            f'{_STATION_BATCH}')
    labels, sources = tuple(models), ('M11', 'M00', 'IRI')
    contract = _peak_contract(alt_range)
    metadata = _resume_metadata(
        phase, records, labels, sources, contract, resume_contract,
        checkpoint_records)
    completed, chunks, complete = _load_resume_chunks(
        resume_root, phase, records, metadata, labels, sources)
    if complete:
        print(f'  resumed complete {phase}: {completed}/{len(records)} records')
        return chunks[0]

    phase_dir = Path(resume_root) / phase
    for start in range(completed, len(records), checkpoint_records):
        end = min(start + checkpoint_records, len(records))
        prediction = _predict_peaks_many(
            records[start:end], models, managers, allowed_by_model, device,
            alt_range)
        _save_prediction_cache(
            phase_dir / f'part_{start}_{end}.npz', metadata,
            records[start:end], prediction, start, end)
        chunks.append(prediction)
    prediction = _concat_prediction_chunks(chunks, labels, sources)
    _save_prediction_cache(
        Path(resume_root) / f'{phase}_complete.npz', metadata, records,
        prediction, 0, len(records))
    return prediction


def _coarse_and_fine_peak_results(coarse_altitudes, coarse_values,
                                  fine_altitudes, fine_values, contract):
    """Apply the common QA contract to model values queried on 10/1 km grids."""
    results, legacy_hmf2, legacy_nmf2 = [], [], []
    for coarse_value, fine_altitude, fine_value in zip(
            coarse_values, fine_altitudes, fine_values):
        raw_index = int(np.nanargmax(fine_value))
        legacy_hmf2.append(float(fine_altitude[raw_index]))
        legacy_nmf2.append(float(fine_value[raw_index]))
        results.append(search_peak_profile(
            np.concatenate([coarse_altitudes, fine_altitude]),
            np.concatenate([coarse_value, fine_value]), contract))
    return results, legacy_hmf2, legacy_nmf2


def _predict_peaks(records, model, managers, allowed, device, alt_range):
    contract = _peak_contract(alt_range)
    lower, upper = contract.lower_km, contract.upper_km
    coarse_altitudes = np.arange(
        lower, upper + 0.5 * contract.coarse_step_km, contract.coarse_step_km,
        dtype=np.float32)
    fine_offsets = np.arange(
        -contract.fine_half_window_km, contract.fine_half_window_km
        + 0.5 * contract.fine_step_km, contract.fine_step_km, dtype=np.float32)
    sources = ('M11', 'M00', 'IRI')
    results = {source: [] for source in sources}
    legacy_hmf2 = {source: [] for source in sources}
    legacy_nmf2 = {source: [] for source in sources}
    fy_index, cosmic_index, sw_manager, iri_peak_manager = managers
    for start in range(0, len(records), _STATION_BATCH):
        selected = np.arange(start, min(start + _STATION_BATCH, len(records)))
        rows = records[selected]
        n_station, n_coarse = len(rows), len(coarse_altitudes)
        coarse_coords = np.column_stack([
            np.repeat(rows[:, 0], n_coarse), np.repeat(rows[:, 1], n_coarse),
            np.tile(coarse_altitudes, n_station), np.repeat(rows[:, 2], n_coarse),
        ]).astype(np.float32)
        coarse = _query_fields(coarse_coords, model, sw_manager, iri_peak_manager,
                               fy_index, cosmic_index, allowed, device)
        coarse_values = {
            source: values.reshape(n_station, n_coarse)
            for source, values in coarse.items()}
        coarse_peak = {
            source: coarse_altitudes[np.nanargmax(values, axis=1)]
            for source, values in coarse_values.items()}
        n_fine = len(fine_offsets)
        fine_altitudes = np.stack([
            np.clip(coarse_peak[source][:, None] + fine_offsets[None, :], lower, upper)
            for source in sources]).astype(np.float32)
        fine_coords = np.column_stack([
            np.tile(np.repeat(rows[:, 0], n_fine), len(sources)),
            np.tile(np.repeat(rows[:, 1], n_fine), len(sources)),
            fine_altitudes.reshape(-1),
            np.tile(np.repeat(rows[:, 2], n_fine), len(sources)),
        ]).astype(np.float32)
        fine = _query_fields(fine_coords, model, sw_manager, iri_peak_manager,
                             fy_index, cosmic_index, allowed, device)
        for source_index, source in enumerate(sources):
            fine_values = fine[source].reshape(len(sources), n_station, n_fine)[source_index]
            block, old_hmf2, old_nmf2 = _coarse_and_fine_peak_results(
                coarse_altitudes, coarse_values[source], fine_altitudes[source_index],
                fine_values, contract)
            results[source].extend(block)
            legacy_hmf2[source].extend(old_hmf2)
            legacy_nmf2[source].extend(old_nmf2)
        print(f'  {selected[-1] + 1:>6}/{len(records)} records')
    return {source: _pack_peak_results(
        results[source], legacy_hmf2[source], legacy_nmf2[source]) for source in sources}


def _predict_peaks_many(records, models, managers, allowed_by_model, device,
                        alt_range):
    """Infer all checkpoints in one record/token loop, with independent peak QA."""
    contract = _peak_contract(alt_range)
    lower, upper = contract.lower_km, contract.upper_km
    coarse_altitudes = np.arange(
        lower, upper + 0.5 * contract.coarse_step_km, contract.coarse_step_km,
        dtype=np.float32)
    fine_offsets = np.arange(
        -contract.fine_half_window_km, contract.fine_half_window_km
        + 0.5 * contract.fine_step_km, contract.fine_step_km, dtype=np.float32)
    labels, sources = tuple(models), ('M11', 'M00', 'IRI')
    results = {label: {source: [] for source in sources} for label in labels}
    legacy_hmf2 = {label: {source: [] for source in sources} for label in labels}
    legacy_nmf2 = {label: {source: [] for source in sources} for label in labels}
    fy_index, cosmic_index, sw_manager, iri_peak_manager = managers
    for start in range(0, len(records), _STATION_BATCH):
        selected = np.arange(start, min(start + _STATION_BATCH, len(records)))
        rows = records[selected]
        n_station, n_coarse = len(rows), len(coarse_altitudes)
        coarse_coords = np.column_stack([
            np.repeat(rows[:, 0], n_coarse), np.repeat(rows[:, 1], n_coarse),
            np.tile(coarse_altitudes, n_station), np.repeat(rows[:, 2], n_coarse),
        ]).astype(np.float32)
        coarse = _query_fields_many(
            coarse_coords, models, sw_manager, iri_peak_manager, fy_index,
            cosmic_index, allowed_by_model, device)
        coarse_values = {
            label: {source: coarse[label][source].reshape(n_station, n_coarse)
                    for source in sources}
            for label in labels}
        fine_altitudes, coordinate_groups = {}, []
        n_fine = len(fine_offsets)
        for label in labels:
            for source in sources:
                coarse_peak = coarse_altitudes[np.nanargmax(
                    coarse_values[label][source], axis=1)]
                altitudes = np.clip(coarse_peak[:, None] + fine_offsets[None, :],
                                    lower, upper).astype(np.float32)
                fine_altitudes[label, source] = altitudes
                coordinate_groups.append(np.column_stack([
                    np.repeat(rows[:, 0], n_fine), np.repeat(rows[:, 1], n_fine),
                    altitudes.reshape(-1), np.repeat(rows[:, 2], n_fine),
                ]).astype(np.float32))
        fine = _query_fields_many(
            np.concatenate(coordinate_groups), models, sw_manager,
            iri_peak_manager, fy_index, cosmic_index, allowed_by_model, device)
        group_size = n_station * n_fine
        for label_index, label in enumerate(labels):
            for source_index, source in enumerate(sources):
                group_index = label_index * len(sources) + source_index
                fine_values = fine[label][source][
                    group_index * group_size:(group_index + 1) * group_size
                ].reshape(n_station, n_fine)
                block, old_hmf2, old_nmf2 = _coarse_and_fine_peak_results(
                    coarse_altitudes, coarse_values[label][source],
                    fine_altitudes[label, source], fine_values, contract)
                results[label][source].extend(block)
                legacy_hmf2[label][source].extend(old_hmf2)
                legacy_nmf2[label][source].extend(old_nmf2)
        print(f'  {selected[-1] + 1:>6}/{len(records)} records')
    return {
        label: {source: _pack_peak_results(
            results[label][source], legacy_hmf2[label][source],
            legacy_nmf2[label][source]) for source in sources}
        for label in labels}


def _peak_values(prediction, source, field, *, primary):
    values = prediction[source][field].astype(np.float64)
    if not primary:
        return prediction[source][f'legacy_{field}'].astype(np.float64)
    valid_key = 'hmf2_valid' if field == 'hmf2' else 'nmf2_valid'
    return np.where(prediction[source][valid_key], values, np.nan)


def _evaluate_quantity(records, field, predictions, *, primary=True):
    observation = records[:, 3].astype(np.float64)
    units = _record_ids(records)
    metrics = {
        source: _metrics(observation, _peak_values(predictions, source, field,
                                                    primary=primary))
        for source in predictions
    }
    candidate = _peak_values(predictions, 'M11', field, primary=primary)
    iri = _peak_values(predictions, 'IRI', field, primary=primary)
    common = np.isfinite(observation) & np.isfinite(candidate) & np.isfinite(iri)
    bootstrap = _safe_paired_bootstrap(
        observation[common], candidate[common], iri[common], units[common])
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


def _is_historical_epoch_checkpoint(checkpoint):
    parts = Path(checkpoint).stem.split('_')
    return len(parts) == 3 and parts[0] == 'epoch' and parts[1].isdigit() \
        and parts[2] == 'model'


def _load_model_context(checkpoint, device, historical_sha256=None,
                        verify_hashes=True, load_allowed=True):
    model, config, _, summary = load_fsia_analysis_checkpoint(
        checkpoint, device=device,
        allow_historical_epoch=historical_sha256 is not None,
        expected_sha256=historical_sha256,
        verify_hashes=verify_hashes)
    peak_range = _require_m2w_peak_contract(config)
    allowed = allowed_observation_profile_ids(config) if load_allowed else None
    return model, config, summary, peak_range, allowed


def _paired_model_bootstrap(records, field, candidate, baseline):
    observation = records[:, 3].astype(np.float64)
    units = _record_ids(records)
    candidate_values = _peak_values(candidate, 'M11', field, primary=True)
    baseline_values = _peak_values(baseline, 'M11', field, primary=True)
    common = (np.isfinite(observation) & np.isfinite(candidate_values)
              & np.isfinite(baseline_values))
    return _safe_paired_bootstrap(
        observation[common], candidate_values[common], baseline_values[common],
        units[common])


_SHARED_FULL30_SEMANTICS = 'shared_full_30d_all_profiles_v1'
_SHARED_FULL30_COUNTS = {'FY': 66612, 'COSMIC': 84364}


def _shared_full30_allowlist(config):
    """Load and validate the A1-S2 Train+Development profile manifest."""
    return shared_full30_profile_allowlist(config, _SHARED_FULL30_COUNTS)
    # Legacy inline implementation retained below only for source compatibility.
    if config.get('profile_split_semantics') != 'source_time_local_block80_dev20_v1':
        raise ValueError('shared-full30 requires the A1-S2 profile split manifest')
    split_path = Path(config.get('profile_split_path', '')).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f'shared-full30 profile split is missing: {split_path}')
    allowed = {}
    audit = {}
    with np.load(split_path, allow_pickle=False) as split:
        for source in ('FY', 'COSMIC'):
            train_key = f'{source}_train_profile_ids'
            dev_key = f'{source}_development_profile_ids'
            if train_key not in split or dev_key not in split:
                raise ValueError(f'shared-full30 manifest lacks {source} partitions')
            train = np.asarray(split[train_key], dtype=np.int64).reshape(-1)
            development = np.asarray(split[dev_key], dtype=np.int64).reshape(-1)
            if np.unique(train).size != train.size:
                raise ValueError(f'{source} Train profile IDs contain duplicates')
            if np.unique(development).size != development.size:
                raise ValueError(f'{source} Development profile IDs contain duplicates')
            if np.intersect1d(train, development).size:
                raise ValueError(f'{source} Train/Development profile IDs overlap')
            ids = np.unique(np.concatenate((train, development)))
            expected = _SHARED_FULL30_COUNTS[source]
            if ids.size != expected:
                raise ValueError(
                    f'{source} shared-full30 count {ids.size} != expected {expected}')
            index_key = f'{source.lower()}_profile_index_path'
            index_path = Path(config.get(index_key, '')).resolve()
            if not index_path.is_file():
                raise FileNotFoundError(f'{source} profile index is missing: {index_path}')
            with np.load(index_path, allow_pickle=False) as index:
                if 'profile_id' not in index:
                    raise ValueError(f'{source} profile index lacks profile_id')
                indexed = np.asarray(index['profile_id'], dtype=np.int64).reshape(-1)
            if not np.all(np.isin(ids, indexed)):
                missing = ids[~np.isin(ids, indexed)][:5].tolist()
                raise ValueError(f'{source} shared-full30 IDs missing from index: {missing}')
            allowed[source] = ids
            audit[source] = {
                'train_count': int(train.size),
                'development_count': int(development.size),
                'all_count': int(ids.size),
                'index_count': int(indexed.size),
                'manifest_path': str(split_path),
            }
    return allowed, audit


def _giro_block_ids(records):
    """Return deterministic geographic-coordinate x UTC-day bootstrap blocks."""
    records = np.asarray(records)
    day = np.floor(records[:, 2].astype(np.float64) / 24.0).astype(np.int64)
    return np.asarray([
        f'{lat:.5f}|{lon:.5f}|day{int(day_value):03d}'
        for (lat, lon), day_value in zip(records[:, :2], day)
    ], dtype=object)


def _giro_mag_band(records):
    """Map the precomputed AACGM latitude column to low/mid/high |sin I|."""
    values = np.asarray(records[:, 4], dtype=np.float64)
    # Round the trigonometric result so exact 0.25/0.5 boundary inputs do not
    # drift into the lower bin because of floating-point representation.
    sini = np.round(np.abs(np.sin(np.deg2rad(values))), 12)
    bands = np.full(values.shape, 'unknown', dtype='<U7')
    finite = np.isfinite(sini)
    bands[finite & (sini < 0.25)] = 'low'
    bands[finite & (sini >= 0.25) & (sini < 0.5)] = 'mid'
    bands[finite & (sini >= 0.5)] = 'high'
    return bands


def _paired_field_summary(records, field, candidate, baseline, source,
                          selection=None):
    records = np.asarray(records)
    if selection is None:
        selection = slice(None)
    records = records[selection]
    observation = records[:, 3].astype(np.float64)
    candidate_values = _peak_values(
        candidate, source, field, primary=True)[selection]
    baseline_values = _peak_values(
        baseline, source, field, primary=True)[selection]
    common = (np.isfinite(observation) & np.isfinite(candidate_values)
              & np.isfinite(baseline_values))
    candidate_metrics = _metrics(observation[common], candidate_values[common])
    baseline_metrics = _metrics(observation[common], baseline_values[common])
    blocks = _giro_block_ids(records)
    bootstrap = _safe_paired_bootstrap(
        observation[common], candidate_values[common], baseline_values[common],
        blocks[common])
    candidate_all = candidate_values
    baseline_all = baseline_values
    observed = np.isfinite(observation)
    candidate_qa = float(np.sum(observed & np.isfinite(candidate_all)) / max(1, observed.sum()))
    baseline_qa = float(np.sum(observed & np.isfinite(baseline_all)) / max(1, observed.sum()))
    threshold = 5.0 if field == 'hmf2' else 0.01
    nondegenerate = bool(
        common.any()
        and candidate_metrics['rmse'] <= baseline_metrics['rmse'] + threshold
        and candidate_metrics['ccc'] >= baseline_metrics['ccc'] - 0.01
        and abs(candidate_metrics['bias']) <= abs(baseline_metrics['bias']) + threshold
        and candidate_qa >= baseline_qa - 0.01)
    rmse_ci = bootstrap.get('delta_rmse_gain', {}).get('ci95', [None, None])
    ccc_ci = bootstrap.get('delta_ccc', {}).get('ci95', [None, None])
    clear_improvement = bool(
        nondegenerate
        and ((rmse_ci[0] is not None and rmse_ci[0] > 0.0)
             or (ccc_ci[0] is not None and ccc_ci[0] > 0.0)))
    return {
        'source': source,
        'n_common': int(common.sum()),
        'sampling_blocks': int(np.unique(blocks[common]).size),
        'candidate': candidate_metrics,
        'baseline': baseline_metrics,
        'candidate_qa_rate': candidate_qa,
        'baseline_qa_rate': baseline_qa,
        'qa_rate_delta': candidate_qa - baseline_qa,
        'bootstrap': bootstrap,
        'nondegenerate': nondegenerate,
        'clear_improvement': clear_improvement,
    }


def _stratified_giro_metrics(records, field, predictions, source):
    observation = records[:, 3].astype(np.float64)
    bands = _giro_mag_band(records)
    blocks = _giro_block_ids(records)
    model_values = {
        label: _peak_values(prediction, source, field, primary=True)
        for label, prediction in predictions.items()
    }
    output = {}
    for band in ('low', 'mid', 'high'):
        selection = bands == band
        common = selection & np.isfinite(observation)
        for values in model_values.values():
            common &= np.isfinite(values)
        band_output = {
            'n_records': int(selection.sum()),
            'n_common': int(common.sum()),
            'sampling_blocks': int(np.unique(blocks[common]).size),
            'models': {},
        }
        for label, values in model_values.items():
            band_output['models'][label] = {
                'n': int(common.sum()),
                'metrics': _metrics(observation[common], values[common]),
            }
        output[band] = band_output
    return output


def _shared_cache(cache, prefix, prediction):
    for source, values in prediction.items():
        for key, value in values.items():
            cache[f'{prefix}_{source}_{key}'] = value


def _write_shared_metrics_csv(path, summaries):
    rows = []
    for quantity, summary in summaries.items():
        for field, values in summary.items():
            rows.append({
                'quantity': quantity,
                'field': field,
                'source': values['source'],
                'n_common': values['n_common'],
                'sampling_blocks': values['sampling_blocks'],
                'candidate_rmse': values['candidate']['rmse'],
                'baseline_rmse': values['baseline']['rmse'],
                'candidate_ccc': values['candidate']['ccc'],
                'baseline_ccc': values['baseline']['ccc'],
                'candidate_bias': values['candidate']['bias'],
                'baseline_bias': values['baseline']['bias'],
                'candidate_qa_rate': values['candidate_qa_rate'],
                'baseline_qa_rate': values['baseline_qa_rate'],
                'nondegenerate': values['nondegenerate'],
                'clear_improvement': values['clear_improvement'],
            })
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
            'quantity', 'field', 'source'])
        writer.writeheader()
        writer.writerows(rows)


def _write_stratified_metrics_csv(path, stratified):
    rows = []
    for quantity, by_source in stratified.items():
        for source, by_band in by_source.items():
            for band, values in by_band.items():
                for label, model_values in values['models'].items():
                    metrics = model_values['metrics']
                    rows.append({
                        'quantity': quantity,
                        'field': source,
                    'magnetic_band': band,
                    'model': label,
                    'n': model_values['n'],
                    'n_common': values['n_common'],
                    'sampling_blocks': values['sampling_blocks'],
                        'rmse': metrics['rmse'],
                        'ccc': metrics['ccc'],
                        'bias': metrics['bias'],
                        'mae': metrics['mae'],
                    })
    fields = ['quantity', 'field', 'magnetic_band', 'model', 'n', 'n_common',
              'sampling_blocks', 'rmse', 'ccc', 'bias', 'mae']
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _postprocess_shared_results(
        save_dir, h_records, n_records, h_predictions, n_predictions,
        candidate_contract, baseline_contract, split_audit, peak_range,
        baseline_label, verify_hashes, log_stages=False):
    """Compute and publish shared-full30 artifacts from cached peak arrays."""
    save_dir = Path(save_dir)
    completed = ['cache_validation']
    summaries, stratified = {}, {}
    for quantity, records, predictions, field in (
            ('hmF2', h_records, h_predictions, 'hmf2'),
            ('NmF2', n_records, n_predictions, 'nmf2')):
        if log_stages:
            _write_postprocess_status(
                save_dir, f'bootstrap_{quantity.lower()}', completed)
        summaries[quantity] = {
            'M11': _paired_field_summary(
                records, field, predictions['A1S2'], predictions['A1S'], 'M11'),
            'M00': _paired_field_summary(
                records, field, predictions['A1S2'], predictions['A1S'], 'M00'),
        }
        stratified[quantity] = {
            source: _stratified_giro_metrics(records, field, predictions, source)
            for source in ('M11', 'M00')
        }
        completed.append(f'bootstrap_{quantity.lower()}')

    iri_metrics = {}
    for quantity, records, predictions, field in (
            ('hmF2', h_records, h_predictions, 'hmf2'),
            ('NmF2', n_records, n_predictions, 'nmf2')):
        observation = records[:, 3].astype(np.float64)
        iri_metrics[quantity] = _metrics(
            observation, _peak_values(
                predictions['A1S2'], 'IRI', field, primary=True))
    region_status = {}
    for quantity in ('hmF2', 'NmF2'):
        field = 'hmf2' if quantity == 'hmF2' else 'nmf2'
        records = h_records if quantity == 'hmF2' else n_records
        predictions = h_predictions if quantity == 'hmF2' else n_predictions
        region_status[quantity] = {}
        bands = _giro_mag_band(records)
        for band in ('low', 'mid', 'high'):
            selection = bands == band
            sub_summary = _paired_field_summary(
                records, field, predictions['A1S2'], predictions['A1S'],
                'M11', selection=selection) if selection.any() else None
            if sub_summary is None or sub_summary['sampling_blocks'] < 30:
                region_status[quantity][band] = {
                    'status': 'insufficient_blocks',
                    'sampling_blocks': 0 if sub_summary is None else
                    sub_summary['sampling_blocks'],
                }
            else:
                region_status[quantity][band] = {
                    'status': 'pass' if sub_summary['nondegenerate'] else 'fail',
                    'sampling_blocks': sub_summary['sampling_blocks'],
                    'nondegenerate': sub_summary['nondegenerate'],
                    'clear_improvement': sub_summary['clear_improvement'],
                }
    m11_nondegenerate = all(
        summaries[q]['M11']['nondegenerate'] for q in ('hmF2', 'NmF2'))
    clear_improvement = any(
        summaries[q]['M11']['clear_improvement'] for q in ('hmF2', 'NmF2'))
    sufficient_regions = all(
        region_status[q][band]['status'] != 'insufficient_blocks'
        for q in ('hmF2', 'NmF2') for band in ('low', 'mid', 'high'))
    regions_nondegenerate = all(
        region_status[q][band].get('status') == 'pass'
        for q in ('hmF2', 'NmF2') for band in ('low', 'mid', 'high'))
    if not m11_nondegenerate:
        decision_label = 'fail'
    elif clear_improvement and sufficient_regions and regions_nondegenerate:
        decision_label = 'clear_improvement_all_regions'
    elif clear_improvement:
        decision_label = 'clear_improvement_region_inconclusive'
    else:
        decision_label = 'mixed_or_nondegenerate'
    contract = {
        'evaluation_schema_version': 3,
        'semantics': _SHARED_FULL30_SEMANTICS,
        'observation_pool': {
            'source': 'A1-S2 profile_split.npz Train+Development',
            'counts': {source: int(values['all_count'])
                       for source, values in split_audit.items()},
            'split_audit': split_audit,
            'candidate_ids_passed_to_model': True,
            'baseline_ids_passed_to_model': True,
            'baseline_native_pool_not_used': True,
        },
        'candidate_checkpoint': candidate_contract,
        'baseline_checkpoint': baseline_contract,
        'peak_search': {
            **_peak_contract(peak_range).as_dict(),
            'alt_range_km': list(map(float, peak_range)),
            'bootstrap': {
                'replicates': 2000,
                'seed': 42,
                'group': 'geographic_coordinate_x_utc_day',
            },
        },
        'qa': {
            'common_mask': 'finite observation and both model fields valid',
            'magnetic_latitude': 'abs(sin(deg2rad(AACGM_latitude_column_4)))',
            'boundaries': {'low': '<0.25', 'mid': '[0.25,0.5)', 'high': '>=0.5'},
        },
        'hash_verification': bool(verify_hashes),
        'locked_test_used': False,
    }
    report = {
        'semantics': _SHARED_FULL30_SEMANTICS,
        'candidate_label': 'A1S2',
        'baseline_label': str(baseline_label),
        'hmF2': {'M11': summaries['hmF2']['M11'],
                 'M00': summaries['hmF2']['M00'],
                 'raw_iri': iri_metrics['hmF2']},
        'NmF2': {'M11': summaries['NmF2']['M11'],
                 'M00': summaries['NmF2']['M00'],
                 'raw_iri': iri_metrics['NmF2']},
        'stratified_magnetic_latitude': stratified,
        'region_status': region_status,
        'decision': {
            'label': decision_label,
            'm11_nondegenerate': m11_nondegenerate,
            'clear_improvement': clear_improvement,
            'sufficient_regions': sufficient_regions,
            'regions_nondegenerate': regions_nondegenerate,
            'static_etkf_available': False,
            'isr_used_for_selection': False,
        },
        'risks': [
            'GIRO records are same-month nowcast observations, not cross-month prediction.',
            'A1-S2 ISR station Bias risk remains an external Pareto consideration.',
            'Magnetic-latitude strata with fewer than 30 geographic-day blocks are diagnostic only.',
        ],
    }
    if log_stages:
        _write_postprocess_status(save_dir, 'writing_artifacts', completed)
    _write_shared_metrics_csv(save_dir / 'giro_peak_paired_metrics.csv', summaries)
    _write_stratified_metrics_csv(
        save_dir / 'giro_peak_stratified_metrics.csv', stratified)
    cache = {
        'hm_record_id': _record_ids(h_records),
        'nm_record_id': _record_ids(n_records),
        'hm_observation': h_records[:, 3],
        'nm_observation_log10': n_records[:, 3],
    }
    _shared_cache(cache, 'a1s2_hm', h_predictions['A1S2'])
    _shared_cache(cache, 'a1s2_nm', n_predictions['A1S2'])
    _shared_cache(cache, 'a1s_hm', h_predictions['A1S'])
    _shared_cache(cache, 'a1s_nm', n_predictions['A1S'])
    _atomic_savez(save_dir / 'giro_peak_cache.npz', cache)
    _atomic_write_json(save_dir / 'giro_peak_contract.json', contract)
    _atomic_write_json(save_dir / 'giro_peak_report.json', report)
    _atomic_write_json(save_dir / 'giro_peak_decision.json', report['decision'])
    with (save_dir / 'giro_peak_report.txt').open('w', encoding='utf-8') as stream:
        stream.write(f'decision={decision_label}\n')
        for quantity in ('hmF2', 'NmF2'):
            for source in ('M11', 'M00'):
                values = summaries[quantity][source]
                stream.write(
                    f"{quantity} {source}: n={values['n_common']} "
                    f"candidate_RMSE={values['candidate']['rmse']:.6f} "
                    f"baseline_RMSE={values['baseline']['rmse']:.6f} "
                    f"candidate_CCC={values['candidate']['ccc']:.6f} "
                    f"baseline_CCC={values['baseline']['ccc']:.6f}\n")
    completed.append('artifacts')
    if log_stages:
        _write_postprocess_status(save_dir, 'visualization', completed)
    _plot_density(save_dir / 'giro_peak_density.png', h_records, n_records,
                  h_predictions['A1S2'], n_predictions['A1S2'],
                  {'M11': summaries['hmF2']['M11']['candidate'],
                   'M00': summaries['hmF2']['M00']['candidate'],
                   'IRI': iri_metrics['hmF2']},
                  {'M11': summaries['NmF2']['M11']['candidate'],
                   'M00': summaries['NmF2']['M00']['candidate'],
                   'IRI': iri_metrics['NmF2']})
    completed.append('visualization')
    if log_stages:
        _write_postprocess_status(save_dir, 'complete', completed)
    return report


def postprocess_giro_peak_shared(resume_dir, save_dir, baseline_label='A1S'):
    """Publish shared-full30 results using complete caches only."""
    if resume_dir is None or save_dir is None:
        raise ValueError('postprocess-only requires --resume-dir and --save-dir')
    save_dir = Path(save_dir)
    expected = {
        'giro_peak_postprocess_status.json', 'giro_peak_paired_metrics.csv',
        'giro_peak_stratified_metrics.csv', 'giro_peak_cache.npz',
        'giro_peak_contract.json', 'giro_peak_report.json',
        'giro_peak_decision.json', 'giro_peak_report.txt',
        'giro_peak_density.png',
    }
    if save_dir.exists():
        unexpected = {path.name for path in save_dir.iterdir()}.difference(expected)
        if unexpected:
            raise FileExistsError(
                f'GIRO output directory contains unrelated artifacts: '
                f'{sorted(unexpected)}')
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_postprocess_status(save_dir, 'loading_complete_caches', [])
    h_records, h_predictions, h_metadata = _load_complete_phase(
        resume_dir, 'hmf2')
    n_records, n_predictions, n_metadata = _load_complete_phase(
        resume_dir, 'nmf2')
    for key in ('evaluation_contract', 'peak_contract', 'labels', 'sources'):
        if h_metadata.get(key) != n_metadata.get(key):
            raise ValueError(f'GIRO complete cache metadata differs for {key}')
    evaluation_contract = h_metadata['evaluation_contract']
    if evaluation_contract.get('semantics') != _SHARED_FULL30_SEMANTICS:
        raise ValueError('GIRO complete caches are not shared-full30 results')
    peak_contract = h_metadata['peak_contract']
    peak_range = (peak_contract['lower_km'], peak_contract['upper_km'])
    split_identity = evaluation_contract.get('inputs', {}).get(
        'profile_split_path')
    split_audit = {
        source: {
            'all_count': int(count),
            'validation': 'complete_cache_metadata',
            'manifest_identity': split_identity,
        }
        for source, count in _SHARED_FULL30_COUNTS.items()
    }
    return _postprocess_shared_results(
        save_dir, h_records, n_records, h_predictions, n_predictions,
        evaluation_contract['candidate'], evaluation_contract['baseline'],
        split_audit, peak_range, baseline_label, verify_hashes=False,
        log_stages=True)


def evaluate_giro_peak_shared(checkpoint, save_dir=None, baseline_checkpoint=None,
                              baseline_label='A1S', verify_hashes=False,
                              resume_dir=None):
    """Evaluate A1-S2 and A1-S on one shared 30-day RO observation pool."""
    if baseline_checkpoint is None:
        raise ValueError('shared-full30 requires --baseline-checkpoint')
    if isinstance(baseline_checkpoint, (list, tuple)):
        if len(baseline_checkpoint) != 1:
            raise ValueError('shared-full30 accepts exactly one baseline checkpoint')
        baseline_checkpoint = baseline_checkpoint[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    candidate_model, candidate_config, candidate_summary, peak_range, _ = (
        _load_model_context(checkpoint, device, verify_hashes=verify_hashes,
                            load_allowed=False))
    baseline_model, baseline_config, baseline_summary, baseline_peak_range, _ = (
        _load_model_context(baseline_checkpoint, device, verify_hashes=verify_hashes,
                            load_allowed=False))
    if tuple(peak_range) != tuple(baseline_peak_range):
        raise ValueError('candidate and baseline peak-search domains differ')
    for key in ('fy_path', 'fy_profile_index_path', 'cosmic_path',
                'cosmic_profile_index_path', 'giro_hmf2_path', 'giro_nmf2_path'):
        if Path(candidate_config.get(key, '')).resolve() != Path(
                baseline_config.get(key, '')).resolve():
            raise ValueError(f'candidate/baseline data path differs for {key}')
    allowed, split_audit = _shared_full30_allowlist(candidate_config)
    for source in ('FY', 'COSMIC'):
        if not np.array_equal(allowed[source], np.sort(allowed[source])):
            raise ValueError(f'{source} shared-full30 IDs are not sorted')
    allowed_by_model = {'A1S2': allowed, 'A1S': allowed}
    candidate_contract = _checkpoint_contract(
        checkpoint, candidate_config, candidate_summary, verify_hashes=verify_hashes)
    baseline_contract = _checkpoint_contract(
        baseline_checkpoint, baseline_config, baseline_summary,
        verify_hashes=verify_hashes)
    resume_contract = {
        'semantics': _SHARED_FULL30_SEMANTICS,
        'candidate': candidate_contract,
        'baseline': baseline_contract,
        'inputs': {
            'candidate_checkpoint': _file_identity(checkpoint),
            'baseline_checkpoint': _file_identity(baseline_checkpoint),
            **{
                key: _file_identity(candidate_config[key])
                for key in (
                    'fy_path', 'fy_profile_index_path', 'cosmic_path',
                    'cosmic_profile_index_path', 'sw_path', 'iri_hmf2_path',
                    'iri_nmf2_path', 'profile_split_path')
            },
        },
    }
    save_dir = Path(save_dir or Path(checkpoint).resolve().parent / 'giro_peak_eval')
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f'GIRO output directory already contains artifacts: {save_dir}')
    save_dir.mkdir(parents=True, exist_ok=True)
    sw_manager = SpaceWeatherManager(
        txt_path=candidate_config['sw_path'],
        start_date_str=candidate_config['start_date_str'],
        total_hours=candidate_config['total_hours'],
        seq_len=candidate_config['seq_len'], device=device)
    iri_peak_manager = IRIPeakManager(
        candidate_config['iri_hmf2_path'], candidate_config['iri_nmf2_path'],
        device=str(device))
    fy_index = FYNeighborhoodIndex(candidate_config['fy_path'], candidate_config)
    cosmic_index = COSMICNeighborhoodIndex(candidate_config['cosmic_path'], candidate_config)
    managers = (fy_index, cosmic_index, sw_manager, iri_peak_manager)
    h_records = np.load(candidate_config['giro_hmf2_path']).astype(np.float32)
    n_records = np.load(candidate_config['giro_nmf2_path']).astype(np.float32)
    for records, label in ((h_records, 'hmF2'), (n_records, 'NmF2')):
        if records.ndim != 2 or records.shape[1] < 5:
            raise ValueError(f'{label} GIRO records must have at least five columns')
        if not np.isfinite(records[:, :3]).all():
            raise ValueError(f'{label} GIRO coordinates contain non-finite values')
    models = {'A1S2': candidate_model, 'A1S': baseline_model}
    print('[GIRO] shared-full30 hmF2 profiles')
    h_predictions = _predict_peaks_many_resumable(
        h_records, models, managers, allowed_by_model, device, peak_range,
        resume_dir, 'hmf2', resume_contract)
    print('[GIRO] shared-full30 NmF2 profiles')
    n_predictions = _predict_peaks_many_resumable(
        n_records, models, managers, allowed_by_model, device, peak_range,
        resume_dir, 'nmf2', resume_contract)
    return _postprocess_shared_results(
        save_dir, h_records, n_records, h_predictions, n_predictions,
        candidate_contract, baseline_contract, split_audit, peak_range,
        baseline_label, verify_hashes)


def evaluate_giro_peak(checkpoint, save_dir=None, baseline_checkpoint=None,
                       baseline_checkpoint_sha256=None, baseline_labels=None,
                       observation_pool=None, skip_sha256=False,
                       resume_dir=None):
    if observation_pool == 'shared-full30':
        labels = list(baseline_labels or ['A1S'])
        if len(labels) != 1:
            raise ValueError('shared-full30 accepts exactly one baseline label')
        return evaluate_giro_peak_shared(
            checkpoint, save_dir=save_dir,
            baseline_checkpoint=baseline_checkpoint,
            baseline_label=labels[0],
            verify_hashes=not skip_sha256,
            resume_dir=resume_dir)
    if observation_pool is not None:
        raise ValueError(f'unknown GIRO observation pool: {observation_pool}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, summary, peak_range, allowed = _load_model_context(
        checkpoint, device)
    candidate_contract = _checkpoint_contract(checkpoint, config, summary)
    if baseline_checkpoint is None:
        baseline_paths = []
    elif isinstance(baseline_checkpoint, (str, os.PathLike)):
        baseline_paths = [str(baseline_checkpoint)]
    else:
        baseline_paths = list(baseline_checkpoint)
    if baseline_checkpoint_sha256 is None:
        baseline_hashes = []
    elif isinstance(baseline_checkpoint_sha256, str):
        baseline_hashes = [baseline_checkpoint_sha256]
    else:
        baseline_hashes = list(baseline_checkpoint_sha256)
    if len(baseline_hashes) > len(baseline_paths):
        raise ValueError('more baseline SHA256 values than baseline checkpoints')
    requested_labels = list(baseline_labels or [])
    if len(requested_labels) > len(baseline_paths):
        raise ValueError('more baseline labels than baseline checkpoints')
    baseline_models, baseline_allowed, baseline_contract = {}, {}, []
    for index, path in enumerate(baseline_paths):
        expected_sha = baseline_hashes[index] if index < len(baseline_hashes) else None
        if _is_historical_epoch_checkpoint(path) and not expected_sha:
            raise ValueError('historical GIRO baseline requires --baseline-checkpoint-sha256')
        if (Path(path).name == 'epoch_12_model.pth'
                and expected_sha != _HISTORICAL_EPOCH12_SHA256):
            raise ValueError('historical epoch12 GIRO baseline SHA256 is not approved')
        (baseline_model, baseline_config, baseline_summary,
         baseline_peak_range, item_allowed) = _load_model_context(
            path, device, expected_sha)
        if (Path(path).name == 'epoch_12_model.pth'
                and baseline_config.get('background_trust_gate_enabled', False)):
            raise ValueError('historical epoch12 GIRO baseline must be gate-off')
        if baseline_peak_range != peak_range:
            raise ValueError('candidate and baseline peak-search domains differ')
        if (config['fy_path'] != baseline_config['fy_path']
                or config['cosmic_path'] != baseline_config['cosmic_path']
                or config['sw_path'] != baseline_config['sw_path']
                or config['iri_hmf2_path'] != baseline_config['iri_hmf2_path']
                or config['iri_nmf2_path'] != baseline_config['iri_nmf2_path']):
            raise ValueError('paired GIRO evaluation requires shared FY/COSMIC/IRI inputs')
        default_label = ('historical_epoch12' if _is_historical_epoch_checkpoint(path)
                         else f"checkpoint_v{baseline_config.get('checkpoint_format_version', index)}")
        label = requested_labels[index] if index < len(requested_labels) else default_label
        if label in baseline_models:
            raise ValueError(f'duplicate baseline label: {label}')
        baseline_models[label] = baseline_model
        baseline_allowed[label] = item_allowed
        baseline_contract.append({
            'label': label,
            **_checkpoint_contract(path, baseline_config, baseline_summary)})
    save_dir = Path(save_dir or Path(checkpoint).resolve().parent / 'giro_peak_eval')
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(
            f'GIRO output directory already contains artifacts: {save_dir}')
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
    if not baseline_models:
        print('[GIRO] independent hmF2 profiles')
        h_prediction = _predict_peaks(
            h_records, model, managers, allowed, device, peak_range)
        print('[GIRO] independent NmF2 profiles')
        n_prediction = _predict_peaks(
            n_records, model, managers, allowed, device, peak_range)
        baseline_h_predictions = baseline_n_predictions = {}
    else:
        models = {'candidate': model, **baseline_models}
        allowed_by_model = {'candidate': allowed, **baseline_allowed}
        print('[GIRO] paired independent hmF2 profiles')
        h_predictions = _predict_peaks_many(
            h_records, models, managers, allowed_by_model, device, peak_range)
        print('[GIRO] paired independent NmF2 profiles')
        n_predictions = _predict_peaks_many(
            n_records, models, managers, allowed_by_model, device, peak_range)
        h_prediction, n_prediction = h_predictions['candidate'], n_predictions['candidate']
        baseline_h_predictions = {
            label: h_predictions[label] for label in baseline_models}
        baseline_n_predictions = {
            label: n_predictions[label] for label in baseline_models}
    h_metrics, h_bootstrap = _evaluate_quantity(
        h_records, 'hmf2', h_prediction, primary=True)
    n_metrics, n_bootstrap = _evaluate_quantity(
        n_records, 'nmf2', n_prediction, primary=True)
    h_legacy_metrics, _ = _evaluate_quantity(
        h_records, 'hmf2', h_prediction, primary=False)
    n_legacy_metrics, _ = _evaluate_quantity(
        n_records, 'nmf2', n_prediction, primary=False)
    comparisons = {}
    for label in baseline_models:
        baseline_h_prediction = baseline_h_predictions[label]
        baseline_n_prediction = baseline_n_predictions[label]
        baseline_h_metrics, _ = _evaluate_quantity(
            h_records, 'hmf2', baseline_h_prediction, primary=True)
        baseline_n_metrics, _ = _evaluate_quantity(
            n_records, 'nmf2', baseline_n_prediction, primary=True)
        comparisons[label] = {
            'hmF2_m11_candidate_vs_baseline_bootstrap': _paired_model_bootstrap(
                h_records, 'hmf2', h_prediction, baseline_h_prediction),
            'NmF2_m11_candidate_vs_baseline_bootstrap': _paired_model_bootstrap(
                n_records, 'nmf2', n_prediction, baseline_n_prediction),
            'baseline_hmF2_metrics': baseline_h_metrics,
            'baseline_NmF2_metrics': baseline_n_metrics,
        }
    contract = {
        'evaluation_schema_version': 2,
        'token_partitions': ['train', 'development'],
        'candidate_checkpoint': candidate_contract,
        'baseline_checkpoints': baseline_contract,
        'quality_thresholds': {'finite_peak_required': True},
        'peak_search': {
            **_peak_contract(peak_range).as_dict(),
            'alt_range_km': list(peak_range),
            'bootstrap': {'replicates': 2000, 'seed': 42,
                          'group': 'station_time_record'},
        },
        'quality_thresholds': {
            'finite_peak_required': True,
            'hmf2_public_mask': 'observation_finite_and_all_compared_fields_valid',
            'nmf2_public_mask': 'observation_finite_and_all_compared_fields_nmf2_valid',
        },
        'giro_input_sha256': {
            'hmf2_path': str(Path(config['giro_hmf2_path']).resolve()),
            'hmf2_sha256': _sha256(config['giro_hmf2_path']),
            'nmf2_path': str(Path(config['giro_nmf2_path']).resolve()),
            'nmf2_sha256': _sha256(config['giro_nmf2_path']),
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
                 'legacy_argmax_metrics': h_legacy_metrics,
                 'm11_vs_raw_iri_bootstrap': h_bootstrap},
        'NmF2': {'metrics': n_metrics,
                 'legacy_argmax_metrics': n_legacy_metrics,
                 'm11_vs_raw_iri_bootstrap': n_bootstrap},
        'candidate_vs_baseline': comparisons,
        'passed_m2w_m11_vs_raw_iri_gate': (
            h_bootstrap.get('decision') == 'pass'
            and n_bootstrap.get('decision') == 'pass'),
        'peak_qc_counts': {
            quantity: {
                source: {
                    'status': {key: int(value) for key, value in zip(
                        *np.unique(prediction[source]['status'], return_counts=True))},
                    'hmf2_valid': int(prediction[source]['hmf2_valid'].sum()),
                    'nmf2_valid': int(prediction[source]['nmf2_valid'].sum()),
                }
                for source in prediction
            }
            for quantity, prediction in (
                ('hmF2_records', h_prediction), ('NmF2_records', n_prediction))
        },
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
            for key, value in values.items():
                cache[f'{prefix}_{source}_{key}'] = value
    for label in baseline_models:
        for prefix, prediction in ((f'baseline_{label}_hm', baseline_h_predictions[label]),
                                   (f'baseline_{label}_nm', baseline_n_predictions[label])):
            for source, values in prediction.items():
                for key, value in values.items():
                    cache[f'{prefix}_{source}_{key}'] = value
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
        if comparisons:
            stream.write('[candidate_vs_baseline]\n')
            stream.write(json.dumps(_json_safe(comparisons), ensure_ascii=False) + '\n')
    _plot_density(
        save_dir / 'giro_peak_density.png', h_records, n_records,
        h_prediction, n_prediction, h_metrics, n_metrics)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--save-dir')
    parser.add_argument('--baseline-checkpoint', action='append')
    parser.add_argument('--baseline-checkpoint-sha256', action='append')
    parser.add_argument('--baseline-label', action='append')
    parser.add_argument('--observation-pool', choices=('shared-full30',))
    parser.add_argument('--skip-sha256', action='store_true')
    parser.add_argument('--resume-dir')
    parser.add_argument('--postprocess-only', action='store_true')
    arguments = parser.parse_args(argv)
    if arguments.postprocess_only:
        labels = list(arguments.baseline_label or ['A1S'])
        if len(labels) != 1:
            parser.error('--postprocess-only accepts exactly one --baseline-label')
        return postprocess_giro_peak_shared(
            arguments.resume_dir, arguments.save_dir, labels[0])
    if arguments.checkpoint is None:
        parser.error('--checkpoint is required unless --postprocess-only is used')
    return evaluate_giro_peak(
        arguments.checkpoint, arguments.save_dir,
        arguments.baseline_checkpoint,
        arguments.baseline_checkpoint_sha256,
        arguments.baseline_label,
        arguments.observation_pool,
        arguments.skip_sha256,
        arguments.resume_dir)


if __name__ == '__main__':
    main()
