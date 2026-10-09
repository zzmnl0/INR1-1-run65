import numpy as np
import torch

from inr_modules.mdia.m00_observation import (
    M00_QD_RELIABILITY_SEMANTICS,
    m00_observation_error_loss,
    observation_stratum_keys,
)


def _calibration(coords, qd, weights):
    keys = observation_stratum_keys(
        coords, "FY", np.ones(len(coords)), qd_latitude_deg=qd,
        strict_magnetic=True).tolist()
    unique = sorted(set(map(str, keys)))
    return {
        "semantics": M00_QD_RELIABILITY_SEMANTICS,
        "bias": {key: 0.0 for key in unique},
        "variance": {key: 1.0 for key in unique},
        "reliability_weight": {
            key: float(weights.get(key, 1.0)) for key in unique},
    }


def test_qd_loss_keeps_point_strata_separate_and_profile_weighted():
    coords = torch.tensor([
        [0.0, 0.0, 250.0, 1.0], [0.0, 1.0, 350.0, 1.0],
        [0.0, 2.0, 250.0, 1.0], [0.0, 3.0, 350.0, 1.0]])
    qd = torch.tensor([10.0, 40.0, 10.0, 40.0])
    prediction = torch.tensor([0.0, 0.0, 1.0, 1.0], requires_grad=True)
    target = torch.zeros(4)
    keys = observation_stratum_keys(
        coords, "FY", torch.ones(4), qd_latitude_deg=qd,
        strict_magnetic=True).tolist()
    cal = _calibration(coords, qd, {str(keys[0]): 4.0, str(keys[1]): 1.0})
    loss, details = m00_observation_error_loss(
        prediction, target, coords, torch.tensor([7, 7, 8, 8]), "FY", cal,
        magnetic_sin_i=torch.ones(4), qd_latitude_deg=qd,
        strict_magnetic=True)
    assert details["aggregation"] == "point_to_profile_stratum_to_source_v1"
    assert details["reliability_applied_once"] is True
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_qd_loss_rejects_missing_exact_key_and_nonfinite_rows():
    coords = torch.tensor([[0.0, 0.0, 250.0, 1.0]])
    qd = torch.tensor([10.0])
    key = str(observation_stratum_keys(
        coords, "FY", torch.ones(1), qd_latitude_deg=qd,
        strict_magnetic=True)[0])
    cal = {
        "semantics": M00_QD_RELIABILITY_SEMANTICS,
        "bias": {key: 0.0}, "variance": {key: 1.0},
        "reliability_weight": {},
    }
    with np.testing.assert_raises((KeyError, ValueError)):
        m00_observation_error_loss(
            torch.tensor([float("nan")]), torch.zeros(1), coords,
            torch.tensor([1]), "FY", cal, magnetic_sin_i=torch.ones(1),
            qd_latitude_deg=qd, strict_magnetic=True)
