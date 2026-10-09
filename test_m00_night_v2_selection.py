"""Frozen night-v2 selection contracts, independent of production datasets."""

import copy

from test_m00_low_budget import _night_row
from inr_modules.mdia.train_fsia import (
    night_generalization_v2_audit,
    select_night_generalization_v2,
)


def _support(row):
    return [f'{source}:{cell}' for source in ('fy', 'cosmic')
            for cell in row[f'{source}_background_stratified']['magnetic_latitude_cells']]


def test_v2_boundaries_and_true_global_profile_protection():
    baseline = _night_row(5, 1.0, 0.70)
    candidate = _night_row(1, 0.999, 0.705, low_bias=0.09)
    supported = _support(baseline)
    audit = night_generalization_v2_audit(candidate, baseline, 1, supported)
    assert audit['eligible'], audit
    candidate['fy_profile_rmse'] = 1.006
    assert not night_generalization_v2_audit(
        candidate, baseline, 1, supported)['eligible']


def test_v2_uses_frozen_subset_and_rejects_missing_or_changed_counts():
    baseline = _night_row(5, 1.0, 0.70)
    candidate = _night_row(1, 0.998, 0.706, low_bias=0.08)
    omitted = '200-300_day_abs_sinI_0.25_0.5'
    del baseline['fy_background_stratified']['magnetic_latitude_cells'][omitted]
    del candidate['fy_background_stratified']['magnetic_latitude_cells'][omitted]
    supported = _support(baseline)
    assert night_generalization_v2_audit(
        candidate, baseline, 1, supported)['eligible']
    cell = candidate['fy_background_stratified']['magnetic_latitude_cells'][
        '200-300_night_abs_sinI_lt_0.25']['M00']
    for bad_count in (29, 31, 30.5, float('nan')):
        cell['profile_count'] = bad_count
        assert not night_generalization_v2_audit(
            candidate, baseline, 1, supported)['eligible']
    cell['profile_count'] = 30
    cell['ccc'] = float('nan')
    assert not night_generalization_v2_audit(
        candidate, baseline, 1, supported)['eligible']
    del cell['ccc']
    assert not night_generalization_v2_audit(
        candidate, baseline, 1, supported)['eligible']


def test_v2_tolerance_ties_choose_earlier_and_ignore_saved_passed():
    baseline = _night_row(5, 1.0, 0.70)
    earlier = _night_row(1, 0.998000005, 0.706, low_bias=0.08)
    later = _night_row(2, 0.998, 0.706, low_bias=0.08)
    result = select_night_generalization_v2(
        [later, earlier], baseline, _support(baseline))
    assert result['selected_row']['epoch'] == 1
    forged = _night_row(3, 1.1, 0.60)
    forged['checkpoint_selection'] = {'eligible': True, 'selection_key': [100]}
    result = select_night_generalization_v2(
        [forged, earlier], baseline, _support(baseline))
    assert result['selected_row']['epoch'] == 1


def test_v2_no_eligible_preserves_diagnostic_best_and_all_bias_deltas():
    baseline = _night_row(5, 1.0, 0.70)
    first = copy.deepcopy(baseline)
    first['epoch'] = 1
    second = _night_row(2, 0.9995, 0.70)
    result = select_night_generalization_v2(
        [first, second], baseline, _support(baseline))
    assert not result['eligible']
    assert result['selected_row'] is None
    assert result['diagnostic_row']['epoch'] == 2
    audit = result['diagnostic_audit']
    assert len(audit['cell_abs_bias_deltas']) == 24
    assert audit['max_cell_abs_bias_worsening'] == 0.0


def test_v2_day_high_night_and_per_cell_guards_are_independent():
    baseline = _night_row(5, 1., .70)
    supported = _support(baseline)
    changes = [
        ('200-300_day_abs_sinI_lt_0.25', 'rmse', 2.),
        ('200-300_day_abs_sinI_lt_0.25', 'ccc', .1),
        ('200-300_night_abs_sinI_ge_0.5', 'rmse', 2.),
        ('200-300_night_abs_sinI_ge_0.5', 'ccc', .1),
        ('200-300_day_abs_sinI_lt_0.25', 'bias', .02000002),
    ]
    for cell, key, value in changes:
        candidate = _night_row(1, .998, .706, low_bias=.08)
        candidate['fy_background_stratified']['magnetic_latitude_cells'][cell]['M00'][key] = value
        assert not night_generalization_v2_audit(candidate, baseline, 1, supported)['eligible']


def test_v2_point_counts_and_training_selection_key_match_shared_gate():
    from inr_modules.mdia.train_fsia import _development_selection_key
    baseline, candidate = _night_row(5, 1., .70), _night_row(1, .998, .706, low_bias=.08)
    supported = _support(baseline)
    config = {'background_checkpoint_selection': 'm00_night_generalization_v2',
              'background_night_v2_supported_cells': supported}
    key = _development_selection_key(candidate, config, 'background', baseline)
    assert key == select_night_generalization_v2([candidate], baseline, supported)['selected_audit']['selection_key']
    name = '200-300_day_abs_sinI_lt_0.25'
    baseline['fy_background_stratified']['magnetic_latitude_cells'][name]['M00']['point_count'] = 60
    candidate['fy_background_stratified']['magnetic_latitude_cells'][name]['M00']['point_count'] = 61
    assert not night_generalization_v2_audit(candidate, baseline, 1, supported)['eligible']
