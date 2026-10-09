import numpy as np
import pytest
from inr_modules.mdia.m00_observation import (
    ObservationCalibration, effective_observation_variance,
    fit_train_calibration, localization_weights,
    m00_observation_error_loss, observation_error_weights,
    observation_stratum_keys, validate_m00_transition)
from inr_modules.mdia.m00_observation import validate_observation_config


def test_train_only_calibration_and_effective_variance():
    c = fit_train_calibration([1, 2, 3], [1, 2, 3], ["FY:a"] * 3)
    corrected, var = c.apply([2], ["FY:a"])
    assert np.isfinite(corrected).all() and (var > 0).all()
    np.testing.assert_allclose(effective_observation_variance([1.0], [2.0]), [3.0])


def test_transition_fail_closed():
    validate_m00_transition({"adapter_route": "stopped", "m00_reference": "A1S_epoch3", "locked_test": "excluded"})
    with pytest.raises(ValueError):
        validate_m00_transition({"adapter_route": "stopped", "m00_reference": "A1S8", "locked_test": "excluded"})


def test_train_only_and_sparse_fallback():
    with pytest.raises(ValueError):
        fit_train_calibration([1, 2], [0, 1], ['FY|200-300|day|low'] * 2,
                              partitions=['Train', 'Development'])
    c = ObservationCalibration(
        {'FY': 0.2}, {'FY': 0.25}, variance_floor=0.01, variance_cap=1.0)
    keys = observation_stratum_keys(
        np.asarray([[10.0, 0.0, 250.0, 12.0]]), 'FY')
    corrected, variance = c.apply([1.0], keys.tolist(), source='FY')
    np.testing.assert_allclose(corrected, [0.8])
    np.testing.assert_allclose(variance, [0.25])


def test_representativeness_localization_and_m00_consumption():
    r_eff = effective_observation_variance([0.2, 0.3], [0.1, 0.2])
    np.testing.assert_allclose(r_eff, [0.3, 0.5])
    weights = observation_error_weights(r_eff)
    assert np.isfinite(weights).all() and np.isclose(weights.mean(), 1.0)
    loc = localization_weights(
        np.asarray([[0.0, 0.0, 250.0, 0.0], [0.0, 0.0, 250.0, 24.0]]),
        np.asarray([0.0, 0.0, 250.0, 0.0]), temporal_scale_hours=2.0)
    np.testing.assert_allclose(loc, [1.0, 1.0])
    cal = ObservationCalibration(
        {'FY|200-300|day|low': 0.1},
        {'FY|200-300|day|low': 0.25})
    import torch
    coords = torch.tensor([[10.0, 0.0, 250.0, 12.0],
                           [10.0, 0.0, 250.0, 12.0]])
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.1, 2.1])
    loss, details = m00_observation_error_loss(
        prediction, target, coords, torch.tensor([7, 7]), 'FY', cal)
    assert details['valid_points'] == 2
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_stage_b_automatically_enables_frozen_rep_loc_weights():
    # Validation only checks existence here; the full loader test consumes a
    # real JSON table and checkpoint in the integration smoke.
    import os
    table = os.path.abspath(__file__)
    seed = os.path.abspath(__file__)
    config = {
        'm00_observation_calibration_semantics': 'train_only_source_stratum_bias_r_v1',
        'm00_observation_bias_path': str(table),
        'm00_observation_seed_checkpoint': str(seed),
        'm00_observation_stage': 'representativeness_localization',
    }
    validate_observation_config(config)
    assert config['m00_representativeness_enabled']
    assert config['m00_localization_enabled']
