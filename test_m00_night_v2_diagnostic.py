from pathlib import Path

import run_m00_condition_diagnostic as diagnostic


def _metric(bias=0.03):
    return {"profile_count": 30, "point_count": 60, "bias": bias,
            "rmse": 0.1, "ccc": 0.9, "profile_mean_rmse": 0.1,
            "mean_profile_rmse": 0.1, "date_sufficient_statistics": [{"relative_utc_day": 0}]}


def _replay(cosmic_bias=0.03):
    result = {"metrics": {}, "train_metrics": {}}
    for split in result:
        for source, bias in (("fy", 0.03), ("cosmic", cosmic_bias)):
            exact = {label: {"M00": _metric(bias)}
                     for label in diagnostic.NIGHT_V2_CELLS}
            region = {label: {"M00": _metric(bias)}
                      for label in diagnostic.NIGHT_MAGNETIC_LABELS}
            result[split][f"{source}_background_stratified"] = {
                "magnetic_latitude_cells": exact,
                "magnetic_region_daynight_cells": region,
                "cells": {},
            }
    return result


def _identity(run):
    return {"run": str(Path(run)), "checkpoint": "checkpoint.pth",
            "checkpoint_sha256": "x", "config_sha256": "y",
            "current_input_sha256": {}, "historical_input_sha256": "unavailable"}


def test_night_v2_freezes_exact_support_and_rejects_low_source_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "_identity", _identity)
    monkeypatch.setattr(diagnostic, "_legacy_replay_check",
                        lambda *args: {"passed": True, "checked": 1})
    monkeypatch.setattr(diagnostic, "_ensure_v2_replay",
                        lambda *args: (_replay(), None))
    passed = diagnostic.prepare_night_v2_diagnostic("control", "historical", tmp_path)
    assert passed["can_train"]
    assert "fy:200-300_night_abs_sinI_lt_0.25" in passed["supported_cells"]
    assert (tmp_path / "m00_night_v2_preflight.json").is_file()

    monkeypatch.setattr(diagnostic, "_ensure_v2_replay",
                        lambda *args: (_replay(-0.03), None))
    failed = diagnostic.prepare_night_v2_diagnostic("control", "historical", tmp_path)
    assert not failed["can_train"]
    assert any("opposite corrections" in reason for reason in failed["reasons"])


def test_v2_replay_identity_requires_config_data_and_date_hashes():
    identity = {key: key for key in ('checkpoint_sha256', 'config_sha256',
                                    'current_input_sha256', 'date_split_sha256')}
    replay = {'train_metrics': {}, 'metrics': {}, 'identity': identity.copy()}
    assert diagnostic._replay_is_compatible(replay, identity)
    for key in identity:
        bad = {**identity, key: 'changed'}
        assert not diagnostic._replay_is_compatible(replay, bad)


def test_v2_legacy_missing_or_changed_counts_fail(monkeypatch):
    row = {'epoch': 5, 'fy_profile_rmse': .2,
           'fy_background_stratified': {'cells': {'x': {'M00': _metric()}}}}
    monkeypatch.setattr(diagnostic, '_rows', lambda run: [row])
    import copy
    metrics = copy.deepcopy(row)
    assert diagnostic._legacy_replay_check(None, None, {'metrics': metrics})['passed']
    del metrics['fy_profile_rmse']
    assert not diagnostic._legacy_replay_check(None, None, {'metrics': metrics})['passed']
    metrics = copy.deepcopy(row)
    metrics['fy_background_stratified']['cells']['x']['M00']['profile_count'] = 30.5
    assert not diagnostic._legacy_replay_check(None, None, {'metrics': metrics})['passed']


def test_v2_support_subset_and_bias_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, '_identity', _identity)
    monkeypatch.setattr(diagnostic, '_legacy_replay_check', lambda *args: {'passed': True})
    replay = _replay()
    for split in replay.values():
        split['fy_background_stratified']['magnetic_latitude_cells'][
            '200-300_day_abs_sinI_0.25_0.5']['M00']['profile_count'] = 29
        for source in ('fy', 'cosmic'):
            split[f'{source}_background_stratified']['magnetic_region_daynight_cells'][
                'abs_sinI_lt_0.25_night']['M00']['bias'] = .02
    monkeypatch.setattr(diagnostic, '_ensure_v2_replay', lambda *args: (replay, None))
    result = diagnostic.prepare_night_v2_diagnostic('c', 'h', tmp_path)
    assert result['can_train'] and len(result['supported_cells']) == 23
    for split in replay.values():
        for source in ('fy', 'cosmic'):
            split[f'{source}_background_stratified']['magnetic_region_daynight_cells'][
                'abs_sinI_lt_0.25_night']['M00']['profile_count'] = 29
    assert not diagnostic.prepare_night_v2_diagnostic('c', 'h', tmp_path)['can_train']


def test_v2_manifest_pins_current_hashes_before_training(tmp_path, monkeypatch):
    import run_a1s_natural_epoch as natural
    import json
    data = tmp_path / 'input.npy'
    data.write_bytes(b'fixture')
    config = {'fy_path': str(data), 'night_v2_frozen_input_sha256': {
        str(data.resolve()): diagnostic._sha256(data)}}
    called = []
    monkeypatch.setattr(natural, 'train_fsia', lambda cfg: called.append(dict(cfg)))
    natural.run_config(config, tmp_path / 'output')
    manifest = json.loads((tmp_path / 'output' / 'run_manifest.json').read_text())
    assert manifest['data_identity']['fy_path']['sha256'] == diagnostic._sha256(data)
    assert called[0]['skip_hash_verification'] is False
    assert called[0]['input_data_sha256']['fy_path'] == diagnostic._sha256(data)


def test_date_bootstrap_is_paired_date_not_point_resampling():
    import numpy as np
    y = np.array([1., 2., 3., 4., 5.])
    p, b = y + np.array([.1, .2, .3, .1, .4]), y + .5
    dates = np.array([0, 0, 1, 2, 2])
    result = diagnostic.paired_date_bootstrap_with_abs_bias(y, p, b, dates)
    assert result['estimable'] and result['date_count'] == 3
    assert result['replicates'] == 2000
    assert np.isclose(result['delta']['rmse'], np.sqrt(np.mean((p-y)**2)) - .5)
    assert result['ci95']['rmse'][1] < 0
    assert not diagnostic.paired_date_bootstrap_with_abs_bias(y, p, b, dates * 0)['estimable']
