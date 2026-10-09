"""Focused checks for shared-full30 GIRO contracts and paired metrics."""

import json

import numpy as np
import pytest

import evaluate_giro_peak as giro


def _shared_config(tmp_path, train=(1, 2), development=(3, 4), indexed=(1, 2, 3, 4)):
    split_path = tmp_path / 'profile_split.npz'
    np.savez(
        split_path,
        FY_train_profile_ids=np.asarray(train, dtype=np.int64),
        FY_development_profile_ids=np.asarray(development, dtype=np.int64),
        COSMIC_train_profile_ids=np.asarray(train, dtype=np.int64),
        COSMIC_development_profile_ids=np.asarray(development, dtype=np.int64),
    )
    config = {
        'profile_split_semantics': 'source_time_local_block80_dev20_v1',
        'profile_split_path': str(split_path),
    }
    for source in ('fy', 'cosmic'):
        path = tmp_path / f'{source}_index.npz'
        np.savez(path, profile_id=np.asarray(indexed, dtype=np.int64))
        config[f'{source}_profile_index_path'] = str(path)
    return config


def test_shared_full30_manifest_is_disjoint_and_index_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(giro, '_SHARED_FULL30_COUNTS', {'FY': 4, 'COSMIC': 4})
    allowed, audit = giro._shared_full30_allowlist(_shared_config(tmp_path))
    assert np.array_equal(allowed['FY'], np.array([1, 2, 3, 4]))
    assert audit['COSMIC']['all_count'] == 4


def test_shared_full30_rejects_overlap_and_missing_index(tmp_path, monkeypatch):
    monkeypatch.setattr(giro, '_SHARED_FULL30_COUNTS', {'FY': 3, 'COSMIC': 3})
    with pytest.raises(ValueError, match='overlap'):
        giro._shared_full30_allowlist(
            _shared_config(tmp_path, train=(1, 2), development=(2, 3), indexed=(1, 2, 3)))
    with pytest.raises(ValueError, match='missing from index'):
        giro._shared_full30_allowlist(
            _shared_config(tmp_path, train=(1, 2), development=(3,), indexed=(1, 2)))


def test_giro_mag_boundaries_and_station_day_blocks():
    records = np.asarray([
        [10.0, 20.0, 0.0, 300.0, np.degrees(np.arcsin(0.25))],
        [10.0, 20.0, 1.0, 300.0, np.degrees(np.arcsin(0.5))],
        [10.0, 20.0, 24.0, 300.0, 0.0],
    ], dtype=np.float32)
    assert giro._giro_mag_band(records).tolist() == ['mid', 'high', 'low']
    blocks = giro._giro_block_ids(records)
    assert blocks[0] == blocks[1]
    assert blocks[0] != blocks[2]


def test_paired_field_summary_uses_one_common_mask():
    n = 40
    records = np.column_stack((
        np.full(n, 10.0), np.full(n, 20.0), np.arange(n, dtype=float),
        np.full(n, 300.0), np.full(n, 0.0)))

    def prediction(offset, valid):
        return {
            'M11': {
                'hmf2': np.full(n, 300.0 + offset),
                'hmf2_valid': np.asarray(valid, dtype=bool),
            },
        }

    candidate = prediction(1.0, [True] * n)
    baseline = prediction(0.0, [True] * (n - 1) + [False])
    summary = giro._paired_field_summary(records, 'hmf2', candidate, baseline, 'M11')
    assert summary['n_common'] == n - 1
    assert summary['sampling_blocks'] == 2
    assert summary['candidate_qa_rate'] == 1.0
    assert summary['baseline_qa_rate'] < 1.0
    selected = np.arange(n) % 2 == 0
    selected_summary = giro._paired_field_summary(
        records, 'hmf2', candidate, baseline, 'M11', selection=selected)
    assert selected_summary['n_common'] == int(selected.sum())
    stratified = giro._stratified_giro_metrics(
        records, 'hmf2', {'A1S2': candidate, 'A1S': baseline}, 'M11')
    assert stratified['low']['n_common'] == n - 1
    assert {values['n'] for values in stratified['low']['models'].values()} == {n - 1}


def test_checkpoint_loader_exposes_hash_opt_out():
    import inspect
    from inr_modules.mdia.checkpoint_io import load_fsia_analysis_checkpoint

    assert 'verify_hashes' in inspect.signature(load_fsia_analysis_checkpoint).parameters


_RESUME_LABELS = ('A1S2', 'A1S')
_RESUME_SOURCES = ('M11', 'M00', 'IRI')


def _resume_records(count=24):
    values = np.arange(count, dtype=np.float32)
    return np.column_stack((values, values + 1, values + 2,
                            values + 3, values + 4))


def _resume_predictions(records):
    values = np.asarray(records[:, 0])
    return {
        label: {
            source: {key: values.copy() for key in giro._PEAK_CACHE_FIELDS}
            for source in _RESUME_SOURCES
        }
        for label in _RESUME_LABELS
    }


def _resume_metadata(records, checkpoint_records=8):
    return giro._resume_metadata(
        'hmf2', records, _RESUME_LABELS, _RESUME_SOURCES,
        giro._peak_contract((200.0, 500.0)), {'run': 'test'},
        checkpoint_records)


def _save_resume_part(root, records, metadata, start, end):
    path = root / 'hmf2' / f'part_{start}_{end}.npz'
    giro._save_prediction_cache(
        path, metadata, records[start:end],
        _resume_predictions(records[start:end]), start, end)
    return path


def test_resume_continues_at_first_missing_batch_and_complete_wins(
        tmp_path, monkeypatch):
    records = _resume_records()
    models = {label: object() for label in _RESUME_LABELS}
    calls = []

    def interrupted(selected, *_args):
        calls.append(selected[:, 0].astype(int).tolist())
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return _resume_predictions(selected)

    monkeypatch.setattr(giro, '_predict_peaks_many', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        giro._predict_peaks_many_resumable(
            records, models, None, None, None, (200.0, 500.0), tmp_path,
            'hmf2', {'run': 'test'}, checkpoint_records=8)
    assert calls == [list(range(8)), list(range(8, 16))]

    calls.clear()

    def resumed(selected, *_args):
        calls.append(selected[:, 0].astype(int).tolist())
        return _resume_predictions(selected)

    monkeypatch.setattr(giro, '_predict_peaks_many', resumed)
    prediction = giro._predict_peaks_many_resumable(
        records, models, None, None, None, (200.0, 500.0), tmp_path,
        'hmf2', {'run': 'test'}, checkpoint_records=8)
    assert calls == [list(range(8, 16)), list(range(16, 24))]
    assert prediction['A1S2']['M11']['hmf2'].tolist() == list(range(24))

    (tmp_path / 'hmf2' / 'part_99_100.npz').write_bytes(b'ignored shard')
    calls.clear()
    giro._predict_peaks_many_resumable(
        records, models, None, None, None, (200.0, 500.0), tmp_path,
        'hmf2', {'run': 'test'}, checkpoint_records=8)
    assert calls == []


def test_resume_shards_sort_numerically_and_bind_full_records(tmp_path):
    records = _resume_records()
    metadata = _resume_metadata(records)
    for start, end in ((0, 8), (8, 16), (16, 24)):
        _save_resume_part(tmp_path, records, metadata, start, end)
    completed, chunks, complete = giro._load_resume_chunks(
        tmp_path, 'hmf2', records, metadata, _RESUME_LABELS,
        _RESUME_SOURCES)
    assert completed == 24
    assert len(chunks) == 3
    assert not complete

    changed = records.copy()
    changed[0, 4] += 1
    with pytest.raises(ValueError, match='records mismatch'):
        giro._load_resume_chunks(
            tmp_path, 'hmf2', changed, metadata, _RESUME_LABELS,
            _RESUME_SOURCES)
    with pytest.raises(ValueError, match='metadata mismatch'):
        giro._load_resume_chunks(
            tmp_path, 'hmf2', records, metadata + 'changed', _RESUME_LABELS,
            _RESUME_SOURCES)


@pytest.mark.parametrize('ranges, message', [
    (((0, 8), (16, 24)), 'non-contiguous'),
    (((0, 16), (8, 24)), 'non-contiguous'),
    (((0, 4),), 'inside a station batch'),
])
def test_resume_rejects_gap_overlap_and_partial_station_batch(
        tmp_path, ranges, message):
    records = _resume_records()
    metadata = _resume_metadata(records)
    for start, end in ranges:
        _save_resume_part(tmp_path, records, metadata, start, end)
    with pytest.raises(ValueError, match=message):
        giro._load_resume_chunks(
            tmp_path, 'hmf2', records, metadata, _RESUME_LABELS,
            _RESUME_SOURCES)


def test_atomic_resume_write_preserves_published_cache_on_failure(
        tmp_path, monkeypatch):
    path = tmp_path / 'published.npz'
    path.write_bytes(b'published')

    def fail_save(*_args, **_kwargs):
        raise RuntimeError('save failed')

    monkeypatch.setattr(giro.np, 'savez_compressed', fail_save)
    with pytest.raises(RuntimeError, match='save failed'):
        giro._atomic_savez(path, {'value': np.arange(2)})
    assert path.read_bytes() == b'published'
    assert not list(tmp_path.glob(f'.{path.name}.*.tmp'))


def test_resume_validates_field_lengths_before_publish(tmp_path):
    records = _resume_records(8)
    prediction = _resume_predictions(records)
    prediction['A1S2']['M11']['hmf2'] = np.arange(7)
    path = tmp_path / 'invalid.npz'
    with pytest.raises(ValueError, match='length mismatch'):
        giro._save_prediction_cache(
            path, _resume_metadata(records), records, prediction, 0, 8)
    assert not path.exists()


def test_cli_forwards_resume_directory_without_enabling_hashes(monkeypatch):
    captured = []
    monkeypatch.setattr(
        giro, 'evaluate_giro_peak', lambda *args: captured.append(args))
    giro.main([
        '--checkpoint', 'candidate.pth',
        '--baseline-checkpoint', 'baseline.pth',
        '--observation-pool', 'shared-full30',
        '--save-dir', 'output',
        '--resume-dir', 'resume',
        '--skip-sha256',
    ])
    assert captured[0][5:] == ('shared-full30', True, 'resume')


def test_postprocess_only_loads_complete_caches_without_evaluation(
        tmp_path, monkeypatch):
    resume = tmp_path / 'resume'
    records = _resume_records(8)
    evaluation_contract = {
        'semantics': giro._SHARED_FULL30_SEMANTICS,
        'candidate': {'path': 'candidate.pth'},
        'baseline': {'path': 'baseline.pth'},
        'inputs': {},
    }
    for phase in ('hmf2', 'nmf2'):
        metadata = giro._resume_metadata(
            phase, records, _RESUME_LABELS, _RESUME_SOURCES,
            giro._peak_contract((200.0, 500.0)), evaluation_contract, 8)
        giro._save_prediction_cache(
            resume / f'{phase}_complete.npz', metadata, records,
            _resume_predictions(records), 0, len(records))

    captured = []
    monkeypatch.setattr(
        giro, '_postprocess_shared_results',
        lambda *args, **kwargs: captured.append((args, kwargs)) or 'done')
    monkeypatch.setattr(
        giro, 'evaluate_giro_peak',
        lambda *_args, **_kwargs: pytest.fail('inference entry point was called'))

    result = giro.main([
        '--postprocess-only', '--resume-dir', str(resume),
        '--save-dir', str(tmp_path / 'output'),
    ])

    assert result == 'done'
    assert captured[0][0][1].shape == records.shape
    assert captured[0][1] == {'verify_hashes': False, 'log_stages': True}
    status = json.loads(
        (tmp_path / 'output' / 'giro_peak_postprocess_status.json').read_text(
            encoding='utf-8'))
    assert status['stage'] == 'loading_complete_caches'
