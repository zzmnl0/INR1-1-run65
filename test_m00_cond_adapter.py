"""Focused regression checks for the Background-only M00 ISR reporter."""

import copy

import numpy as np

from isr_evaluation.background_m00_eval import _metrics, _reporting_masks
from inr_modules.mdia.train_fsia import (
    _BACKGROUND_SELECTION_CELLS,
    _BACKGROUND_SELECTION_MAGNETIC_LABELS,
    _joint_fresh_adapter_selection_audit,
)


def test_isr_strata_boundaries():
    altitude = np.array([120., 199.999, 200., 299.999, 300., 500., 500.001])
    local_time = np.array([6., 17.999, 18., 5.999, 0., 23.999, 12.])
    masks = dict(_reporting_masks(altitude, local_time))
    assert masks['low_alt_120_200_day'].tolist() == [True, True, False, False, False, False, False]
    assert masks['main_200_300_night'].tolist() == [False, False, True, True, False, False, False]
    assert masks['main_300_500_night'].tolist() == [False, False, False, False, True, True, False]
    assert masks['main_200_500_day'].tolist() == [False, False, False, False, False, False, False]


def test_metrics_are_finite_and_signed():
    result = _metrics(np.array([1., 2.]), np.array([1.1, 1.9]))
    assert result['n'] == 2
    assert np.isclose(result['bias'], 0.)
    assert np.isfinite(result['rmse'])


def _selector_metrics(bias_scale=1.0):
    metrics = {
        'score': 0.2,
        'rmse': 0.2,
        'ccc': 0.8,
        'pearson_r': 0.8,
        'fy_profile_rmse': 0.2,
        'cosmic_profile_rmse': 0.2,
        'fy_ccc': 0.8,
        'cosmic_ccc': 0.8,
    }
    for source in ('fy', 'cosmic'):
        stratified = {'cells': {}, 'magnetic_latitude_cells': {}}
        for altitude_cell in _BACKGROUND_SELECTION_CELLS:
            stratified['cells'][altitude_cell] = {
                'M00': {'bias': 0.1 * bias_scale, 'profile_count': 40}
            }
            for band in _BACKGROUND_SELECTION_MAGNETIC_LABELS:
                stratified['magnetic_latitude_cells'][
                    f'{altitude_cell}_{band}'] = {
                        'M00': {'bias': 0.1 * bias_scale, 'profile_count': 40}
                    }
        metrics[f'{source}_background_stratified'] = stratified
    return metrics


def test_joint_fresh_selector_gates_and_conflict():
    baseline = _selector_metrics()
    candidate = _selector_metrics(0.5)
    audit = _joint_fresh_adapter_selection_audit(candidate, baseline)
    assert audit['eligible']
    assert audit['mean_abs_bias_improvement_24'] > 0.005

    conflict = copy.deepcopy(candidate)
    conflict['fy_background_stratified']['magnetic_latitude_cells'][
        '200-300_night_abs_sinI_lt_0.25']['M00']['bias'] = 0.1
    conflict['cosmic_background_stratified']['magnetic_latitude_cells'][
        '200-300_night_abs_sinI_lt_0.25']['M00']['bias'] = -0.1
    conflict_audit = _joint_fresh_adapter_selection_audit(conflict, baseline)
    assert not conflict_audit['eligible']
    assert 'source_conflict' in conflict_audit['rejection_reasons']

    no_change = _joint_fresh_adapter_selection_audit(baseline, baseline)
    assert not no_change['eligible']
    assert no_change['selection_key'] is None
    assert no_change['diagnostic_selection_key'] is not None

    nonfinite = copy.deepcopy(candidate)
    nonfinite['fy_profile_rmse'] = float('nan')
    assert not _joint_fresh_adapter_selection_audit(
        nonfinite, baseline)['eligible']
