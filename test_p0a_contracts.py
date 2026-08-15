"""Regression coverage for P0-A ISR finalization and contract verification."""

import json
from pathlib import Path

import numpy as np
import pytest

import isr_evaluation.main_isr_eval as isr_eval
from verify_p0a_contracts import ContractError, _verify_giro, _verify_isr


_CANDIDATE_SHA = 'a' * 64
_V13_SHA = 'b' * 64
_HISTORY_SHA = 'c' * 64
_SPLIT_SHA = 'd' * 64
_BASELINES = [
    ('v13-development', _V13_SHA),
    ('historical-epoch12', _HISTORY_SHA),
]


def _checkpoint(sha256):
    return {
        'path': f'D:/checkpoints/{sha256}.pth',
        'sha256': sha256,
        'date_split': {'sha256': _SPLIT_SHA},
    }


def _station_report(station):
    cache_key = [f'{station}-0', f'{station}-1']
    values = [10.0, 10.1]
    primary = {
        'keys': cache_key,
        'observation_log10': values,
        'candidate_M11_log10': [10.02, 10.12],
        'candidate_M00_log10': [9.98, 10.08],
        'candidate_IRI_log10': [10.0, 10.1],
        'baseline_M11_log10': [10.01, 10.11],
        'baseline_M00_log10': [9.99, 10.09],
        'baseline_IRI_log10': [10.0, 10.1],
        'altitude_km': [250.0, 300.0],
        'unit_id': [1, 2],
    }
    return {
        'station': station,
        'model_name': 'M11',
        'n_days': 1,
        'point_n': 2,
        'point_rmse': 0.1,
        'point_mae': 0.1,
        'point_r': 0.9,
        'point_ccc': 0.9,
        'point_bias': 0.0,
        'nmf2_n': 2,
        'nmf2_mae': 0.1,
        'nmf2_r': 0.9,
        'nmf2_ccc': 0.9,
        'nmf2_bias': 0.0,
        'hmf2_n': 2,
        'hmf2_mae': 2.0,
        'hmf2_ccc': 0.8,
        'hmf2_bias': 0.0,
        'iri_point_n': 2,
        'iri_point_rmse': 0.1,
        'iri_point_mae': 0.1,
        'iri_point_r': 0.9,
        'iri_point_ccc': 0.9,
        'iri_point_bias': 0.0,
        'background_point_n': 2,
        'background_point_rmse': 0.1,
        'background_point_mae': 0.1,
        'background_point_r': 0.9,
        'background_point_ccc': 0.9,
        'background_point_bias': 0.0,
        'iri_nmf2_n': 2,
        'iri_nmf2_mae': 0.1,
        'iri_nmf2_r': 0.9,
        'iri_nmf2_ccc': 0.9,
        'iri_nmf2_bias': 0.0,
        'background_nmf2_n': 2,
        'background_nmf2_mae': 0.1,
        'background_nmf2_r': 0.9,
        'background_nmf2_ccc': 0.9,
        'background_nmf2_bias': 0.0,
        'iri_hmf2_n': 2,
        'iri_hmf2_mae': 2.0,
        'iri_hmf2_ccc': 0.8,
        'iri_hmf2_bias': 0.0,
        'background_hmf2_n': 2,
        'background_hmf2_mae': 2.0,
        'background_hmf2_ccc': 0.8,
        'background_hmf2_bias': 0.0,
        'passed_m2w_m11_vs_raw_iri_gate': True,
        'low_altitude_diagnostic': {
            'status': 'computed', 'altitude_range_km': [120.0, 200.0],
        },
        'primary_peak_metrics': {'analysis': {'hmf2_mae': 2.0}},
        'legacy_peak_metrics': {'analysis': {'hmf2_mae': 3.0}},
        'mask_attrition': {'primary_common_hmf2_n': 2},
        'peak_qc_counts': {'M11': {'status': {'valid': 2}}},
        'evaluation_cache': {
            'keys': cache_key,
            'observation_log10': values,
            'M11_log10': [10.02, 10.12],
            'M00_log10': [9.98, 10.08],
            'IRI_log10': values,
            'altitude_km': [250.0, 300.0],
            'unit_id': [1, 2],
        },
        'peak_cache': {
            'station': [station, station],
            'timestamp': [1, 2],
            'ISR_status': ['valid', 'valid'],
            'M11_status': ['valid', 'flat_peak'],
            'M00_status': ['valid', 'valid'],
            'IRI_status': ['valid', 'valid'],
        },
        'paired_evaluation_cache': primary,
        'additional_paired_evaluation_caches': {
            'historical-epoch12': primary,
        },
    }


def _write_isr(tmp_path):
    output_dir = tmp_path / 'isr'
    output_dir.mkdir(parents=True)
    reports = [_station_report('Jicamarca'), _station_report('PokerFlat')]
    candidate = _checkpoint(_CANDIDATE_SHA)
    baselines = [
        {'label': label, **_checkpoint(sha256)}
        for label, sha256 in _BASELINES
    ]
    _, contract = isr_eval._write_isr_final_artifacts(
        str(output_dir), reports, candidate, baselines,
        [{'path': 'D:/isr/poker.h5', 'sha256': 'e' * 64}],
        [{'date': '2024-09-01', 'geometry_qc': {
            'legacy_range_as_height_delta_km_median': 0.215}}],
        {'peak_search_alt_range': (200.0, 500.0)})
    return output_dir, reports, contract


def _rewrite_contract(output_dir, contract):
    isr_eval._write_json_atomically(
        str(output_dir / 'isr_evaluation_contract.json'), contract)


def _refresh_artifact_identity(output_dir, contract, relative_path):
    path = output_dir / relative_path
    identities = [contract['output_artifacts']['report_text'],
                  contract['output_artifacts']['report_json']]
    identities.extend(contract['output_artifacts']['caches'])
    for identity in identities:
        if identity['relative_path'] == relative_path:
            identity['sha256'] = isr_eval._sha256(str(path))
            identity['size_bytes'] = path.stat().st_size
            return
    raise AssertionError(f'missing artifact identity for {relative_path}')


def test_isr_finalizer_is_non_destructive_and_writes_completion_contract(tmp_path):
    output_dir, reports, contract = _write_isr(tmp_path)
    assert all('evaluation_cache' in report for report in reports)
    assert contract['baseline_checkpoints'][1]['label'] == 'historical-epoch12'
    assert not (output_dir / 'isr_evaluation_contract.json').is_symlink()
    assert (output_dir / 'isr_validation_report.json').is_file()
    assert (output_dir / 'isr_evaluation_contract.json').is_file()
    _verify_isr(output_dir, _CANDIDATE_SHA, _BASELINES, _SPLIT_SHA, None)


def test_isr_finalizer_does_not_leave_contract_when_report_write_fails(tmp_path,
                                                                       monkeypatch):
    output_dir = tmp_path / 'isr'
    output_dir.mkdir()
    original = isr_eval._write_json_atomically

    def fail_report(path, value):
        if path.endswith('isr_validation_report.json'):
            raise OSError('synthetic report write failure')
        return original(path, value)

    monkeypatch.setattr(isr_eval, '_write_json_atomically', fail_report)
    reports = [_station_report('Jicamarca'), _station_report('PokerFlat')]
    with pytest.raises(OSError, match='synthetic'):
        isr_eval._write_isr_final_artifacts(
            str(output_dir), reports, _checkpoint(_CANDIDATE_SHA),
            [{'label': label, **_checkpoint(sha256)}
             for label, sha256 in _BASELINES], [], [],
            {'peak_search_alt_range': (200.0, 500.0)})
    assert not (output_dir / 'isr_evaluation_contract.json').exists()


def test_isr_verifier_rejects_null_low_altitude_diagnostic(tmp_path):
    output_dir, _, _ = _write_isr(tmp_path)
    report_path = output_dir / 'isr_validation_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report[0]['low_altitude_diagnostic'] = None
    isr_eval._write_json_atomically(str(report_path), report)
    contract_path = output_dir / 'isr_evaluation_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    _refresh_artifact_identity(output_dir, contract, 'isr_validation_report.json')
    _rewrite_contract(output_dir, contract)
    with pytest.raises(ContractError, match='low_altitude_diagnostic'):
        _verify_isr(output_dir, _CANDIDATE_SHA, _BASELINES, _SPLIT_SHA, None)


def test_isr_verifier_rejects_unknown_peak_status(tmp_path):
    output_dir, _, _ = _write_isr(tmp_path)
    cache_path = output_dir / 'isr_peak_cache.npz'
    with np.load(cache_path, allow_pickle=False) as cache:
        payload = {key: cache[key] for key in cache.files}
    payload['M11_status'] = np.array(['invalid', 'flat_peak', 'valid', 'valid'])
    np.savez_compressed(cache_path, **payload)
    contract_path = output_dir / 'isr_evaluation_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    _refresh_artifact_identity(output_dir, contract, 'isr_peak_cache.npz')
    _rewrite_contract(output_dir, contract)
    with pytest.raises(ContractError, match='unknown status'):
        _verify_isr(output_dir, _CANDIDATE_SHA, _BASELINES, _SPLIT_SHA, None)


def test_isr_verifier_rejects_missing_sha_nonfinite_cache_and_geometry(tmp_path):
    output_dir, _, _ = _write_isr(tmp_path)
    with pytest.raises(ContractError, match='candidate SHA mismatch'):
        _verify_isr(output_dir, 'f' * 64, _BASELINES, _SPLIT_SHA, None)

    report_path = output_dir / 'isr_validation_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report[0]['point_rmse'] = None
    isr_eval._write_json_atomically(str(report_path), report)
    contract_path = output_dir / 'isr_evaluation_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    _refresh_artifact_identity(output_dir, contract, 'isr_validation_report.json')
    _rewrite_contract(output_dir, contract)
    with pytest.raises(ContractError, match='point_rmse'):
        _verify_isr(output_dir, _CANDIDATE_SHA, _BASELINES, _SPLIT_SHA, None)

    output_dir, _, _ = _write_isr(tmp_path / 'second')
    cache_path = output_dir / 'isr_evaluation_cache.npz'
    with np.load(cache_path, allow_pickle=False) as cache:
        payload = {key: cache[key] for key in cache.files}
    payload['M11_log10'] = payload['M11_log10'][:-1]
    np.savez_compressed(cache_path, **payload)
    contract_path = output_dir / 'isr_evaluation_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    _refresh_artifact_identity(output_dir, contract, 'isr_evaluation_cache.npz')
    _rewrite_contract(output_dir, contract)
    with pytest.raises(ContractError, match='inconsistent lengths'):
        _verify_isr(output_dir, _CANDIDATE_SHA, _BASELINES, _SPLIT_SHA, None)

    output_dir, _, _ = _write_isr(tmp_path / 'third')
    contract_path = output_dir / 'isr_evaluation_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['poker_geometry']['range_scale_to_km'] = 1.0
    _rewrite_contract(output_dir, contract)
    with pytest.raises(ContractError, match='range scale mismatch'):
        _verify_isr(output_dir, _CANDIDATE_SHA, _BASELINES, _SPLIT_SHA, None)


def test_giro_verifier_rejects_wrong_schema(tmp_path):
    output_dir = tmp_path / 'giro'
    output_dir.mkdir()
    contract = {
        'evaluation_schema_version': 1,
        'token_partitions': ['train', 'development'],
        'candidate_checkpoint': _checkpoint(_CANDIDATE_SHA),
        'baseline_checkpoints': [
            {'label': label, **_checkpoint(sha256)}
            for label, sha256 in _BASELINES
        ],
        'peak_search': {
            **{key: value for key, value in {
                'lower_km': 200.0, 'upper_km': 500.0,
                'coarse_step_km': 10.0, 'fine_step_km': 1.0,
                'fine_half_window_km': 10.0, 'min_finite_levels': 5,
                'max_local_gap_km': 20.0, 'flank_support_km': 30.0,
                'prominence_dex': 0.03, 'secondary_separation_km': 30.0,
                'near_tie_dex': 0.03, 'boundary_margin_km': 10.0,
            }.items()},
            'alt_range_km': [200.0, 500.0],
        },
        'quality_thresholds': {
            'hmf2_public_mask': 'observation_finite_and_all_compared_fields_valid',
            'nmf2_public_mask': 'observation_finite_and_all_compared_fields_nmf2_valid',
        },
    }
    report = {
        'hmF2': {'metrics': {}, 'legacy_argmax_metrics': {}},
        'NmF2': {'metrics': {}, 'legacy_argmax_metrics': {}},
        'candidate_vs_baseline': {label: {} for label, _ in _BASELINES},
    }
    (output_dir / 'giro_peak_contract.json').write_text(
        json.dumps(contract), encoding='utf-8')
    (output_dir / 'giro_peak_report.json').write_text(
        json.dumps(report), encoding='utf-8')
    np.savez_compressed(output_dir / 'giro_peak_cache.npz',
                        candidate_hm_M11_status=np.array(['valid']))
    with pytest.raises(ContractError, match='evaluation_schema_version'):
        _verify_giro(output_dir, _CANDIDATE_SHA, _BASELINES, _SPLIT_SHA)
