import json

import numpy as np
import pytest

from inr_modules.mdia.m00_observation import (
    M00_OBSERVATION_V2_SEMANTICS,
    M00_3D_LOCALIZATION_SEMANTICS,
    ObservationCalibration,
    fit_train_observation_calibration,
    load_magnetic_sidecar,
    localization_weights,
    observation_stratum_keys,
    validate_frozen_m00_config,
    validate_low_dim_assimilation_config,
    validate_strict_v2_calibration,
)


def _sidecar(path, *, mismatch=False, metadata=True):
    coords = np.asarray([
        [10.0, 20.0, 250.0, 1.0],
        [10.0, 20.0, 300.0, 2.0],
        [11.0, 21.0, 350.0, 3.0],
    ], dtype=np.float64)
    payload = {
        "row_index": np.arange(len(coords), dtype=np.int64),
        "profile_id": np.asarray([7, 7, 8], dtype=np.int64),
        "latitude_deg": coords[:, 0],
        "longitude_deg": coords[:, 1] + (1.0 if mismatch else 0.0),
        "altitude_km": coords[:, 2],
        "utc_hours": coords[:, 3],
        "aacgm_latitude_deg": np.asarray([10.0, 20.0, 60.0]),
        "semantics": np.asarray("aacgm_v2"),
        "fallback": np.asarray(False),
    }
    if metadata:
        payload["aacgm_version"] = np.asarray("aacgmv2-2.7.0")
        payload["reference_time_utc"] = np.asarray("2024-09-15T00:00:00Z")
    np.savez(path, **payload)
    return coords, payload["profile_id"]


def test_strict_sidecar_alignment_and_metadata(tmp_path):
    path = tmp_path / "fy_aacgm.npz"
    coords, profiles = _sidecar(path)
    loaded = load_magnetic_sidecar(
        path, expected_rows=3, source="FY", strict_aacgm=True,
        expected_profile_ids=profiles, expected_coords=coords)
    np.testing.assert_array_equal(loaded["row_index"], np.arange(3))
    np.testing.assert_allclose(loaded["aacgm_latitude_deg"], [10.0, 20.0, 60.0])

    with pytest.raises(ValueError):
        _sidecar(tmp_path / "bad_coords.npz", mismatch=True)
        load_magnetic_sidecar(
            tmp_path / "bad_coords.npz", expected_rows=3,
            strict_aacgm=True, expected_profile_ids=profiles,
            expected_coords=coords)
    _sidecar(tmp_path / "bad_meta.npz", metadata=False)
    with pytest.raises(ValueError):
        load_magnetic_sidecar(tmp_path / "bad_meta.npz", strict_aacgm=True)


def test_aacgm_boundaries_and_fit():
    coords = np.asarray([
        [0.0, 0.0, 250.0, 1.0],
        [0.0, 0.0, 250.0, 1.0],
        [0.0, 0.0, 250.0, 1.0],
    ])
    aacgm = np.asarray([np.degrees(np.arcsin(.249)),
                        np.degrees(np.arcsin(.25)),
                        np.degrees(np.arcsin(.5))])
    keys = observation_stratum_keys(
        coords, "FY", aacgm_latitude_deg=aacgm, strict_magnetic=True)
    assert [str(k).split("|")[-1] for k in keys] == ["low", "mid", "high"]
    cal = fit_train_observation_calibration(
        [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], coords, "FY",
        aacgm_latitude_deg=aacgm, strict_magnetic=True,
        partitions=["Train"] * 3)
    assert set(cal.bias) == set(map(str, keys))


def test_strict_calibration_support_and_metadata():
    key = "FY|200-300|day|low"
    payload = {
        "semantics": M00_OBSERVATION_V2_SEMANTICS,
        "train_only": True,
        "locked_test": "excluded",
        "bias": {key: 0.0},
        "variance": {key: 0.1},
        "representativeness": {key: 0.02},
        "representativeness_support": {
            key: {"profiles": 20, "profile_days": 10, "ess": 10}
        },
        "magnetic_coordinate": {
            "semantics": "aacgm_v2",
            "fallback": False,
            "aacgm_version": "aacgmv2-2.7.0",
            "reference_time_utc": "2024-09-15T00:00:00Z",
            "actual_height_semantics": "row_altitude_km_v1",
        },
        "localization_contract": {
            "semantics": M00_3D_LOCALIZATION_SEMANTICS,
            "spatial_scale_km": 1800.0,
            "vertical_scale_km": 100.0,
            "temporal_scale_hours": 1.5,
            "time_semantics": "relative_utc_hours_v1",
            "applies_to": "observation_precision_gain_only",
        },
    }
    validate_strict_v2_calibration(payload)
    bad = json.loads(json.dumps(payload))
    bad["representativeness_support"][key]["profiles"] = 2
    with pytest.raises(ValueError):
        validate_strict_v2_calibration(bad)


def test_localization_is_3d_and_continuous_utc():
    query = np.asarray([[0.0, 179.0, 300.0, 0.0]])
    ref = np.asarray([[0.0, -179.0, 300.0, 25.0]])
    relative = localization_weights(
        query, ref, time_semantics="relative_utc_hours_v1", strict=True)
    cyclic = localization_weights(
        query, ref, time_semantics="local_time_cyclic", strict=True)
    assert relative[0] == 0.0
    assert cyclic[0] > 0.0
    vertical = localization_weights(
        query, np.asarray([[0.0, 179.0, 450.0, 0.0]]),
        time_semantics="relative_utc_hours_v1", strict=True)
    assert vertical[0] == 0.0
    with pytest.raises(ValueError):
        localization_weights(query, None, strict=True)


def test_low_dim_contract_and_stopped_routes():
    validate_low_dim_assimilation_config({})
    with pytest.raises(ValueError):
        validate_low_dim_assimilation_config({"basis_dim": 32})
    with pytest.raises(ValueError):
        validate_frozen_m00_config({
            "background_training_scope": "condition_adapter_only_v1"
        })
    validate_frozen_m00_config({
        "background_training_scope": "condition_adapter_only_v1",
        "eval_only": True,
    })
    with pytest.raises(ValueError):
        validate_frozen_m00_config({"shared_backbone_unfreeze": True})
