"""Small runtime regressions for the frozen M00 v2 sidecar contract."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from inr_modules.data_managers.FY_dataloader import FY3D_Dataset
from inr_modules.mdia.m00_observation import (
    validate_strict_v2_calibration,
)
from inr_modules.mdia.train_fsia import (
    _batch_magnetic_values,
    _resolve_m00_observation_calibration,
    _unpack_source_batch,
    _validate_date_split,
)
from inr_modules.config_mdia import get_config_mdia


def _write_fixture(tmp_path):
    data = np.asarray([
        [10.0, 20.0, 250.0, 49.0, 10.0],
        [10.0, 20.0, 300.0, 50.0, 10.1],
        [11.0, 21.0, 350.0, 51.0, 10.2],
    ], dtype=np.float32)
    np.save(tmp_path / "data.npy", data)
    np.savez(
        tmp_path / "index.npz",
        profile_id=np.asarray([7, 8], dtype=np.int64),
        pass_profile=np.asarray([True, True]),
        output_start=np.asarray([0, 2], dtype=np.int64),
        output_end=np.asarray([2, 3], dtype=np.int64),
    )
    np.savez(
        tmp_path / "magnetic.npz",
        row_index=np.arange(3, dtype=np.int64),
        profile_id=np.asarray([7, 7, 8], dtype=np.int64),
        latitude_deg=data[:, 0],
        longitude_deg=data[:, 1],
        altitude_km=data[:, 2],
        utc_hours=data[:, 3],
        aacgm_latitude_deg=np.asarray([10.0, 20.0, 60.0]),
        magnetic_sin_i=np.asarray([0.1, 0.2, 0.8]),
        semantics=np.asarray("aacgm_v2"),
        aacgm_version=np.asarray("aacgmv2-2.7.0"),
        reference_time_utc=np.asarray("2024-09-15T00:00:00Z"),
        fallback=np.asarray(False),
    )
    return data


def test_dataset_sidecar_stays_out_of_model_features(tmp_path):
    _write_fixture(tmp_path)
    dataset = FY3D_Dataset(
        str(tmp_path / "data.npy"),
        mode="train",
        use_memmap=True,
        profile_index_path=str(tmp_path / "index.npz"),
        magnetic_sidecar_path=str(tmp_path / "magnetic.npz"),
        magnetic_sidecar_source="FY",
        magnetic_sidecar_strict=True,
    )
    batch = [dataset[0], dataset[1]]
    # The fourth item is metadata; the first five data columns are unchanged.
    assert len(batch[0]) == 4
    assert batch[0][0].shape == (5,)
    assert batch[0][3].tolist() == pytest.approx([10.0, 0.1])
    collated = (
        __import__("torch").stack([item[0] for item in batch]),
        __import__("torch").stack([item[1] for item in batch]),
        __import__("torch").stack([item[2] for item in batch]),
        __import__("torch").stack([item[3] for item in batch]),
    )
    sin_i, aacgm = _batch_magnetic_values(collated, {}, "FY")
    assert sin_i.tolist() == pytest.approx([0.1, 0.2])
    assert aacgm.tolist() == pytest.approx([10.0, 20.0])
    coords, target, profile_ids = _unpack_source_batch(collated, "cpu")
    assert coords.shape == (2, 4)
    assert target.shape == (2, 1)
    assert profile_ids.tolist() == [7, 7]


def test_a1s_date_contract_is_20_5_5():
    partitions = {
        "train": list(range(20)),
        "development": list(range(20, 25)),
        "locked_test": list(range(25, 30)),
    }
    assert _validate_date_split(partitions, 30, 5, 5) == partitions
    with pytest.raises(ValueError):
        _validate_date_split({**partitions, "train": list(range(15))}, 30, 5, 5)


def test_strict_calibration_requires_support_for_every_error_key():
    payload = {
        "semantics": "m00_train_only_observation_error_v2",
        "train_only": True,
        "locked_test": "excluded",
        "bias": {"FY|200-300|day|low": 0.0},
        "variance": {"FY|200-300|day|low": 0.1},
        "representativeness": {"FY|200-300|day|low": 0.02},
        "representativeness_support": {
            "FY|200-300|day|low": {
                "profiles": 20,
                "profile_days": 10,
                "ess": 10,
            }
        },
        "magnetic_coordinate": {
            "semantics": "aacgm_abs_sin_lat_025_050_v2",
            "aacgm_version": "aacgmv2-2.7.0",
            "reference_time_utc": "2024-09-15T00:00:00Z",
            "actual_height_semantics": "row_altitude_km_v1",
            "fallback": False,
        },
        "localization_contract": {
            "semantics": "m00_3d_localization_v2",
            "spatial_scale_km": 1800.0,
            "vertical_scale_km": 100.0,
            "temporal_scale_hours": 1.5,
            "time_semantics": "relative_utc_hours_v1",
            "applies_to": "observation_precision_gain_only",
        },
    }
    validate_strict_v2_calibration(payload)
    bad = json.loads(json.dumps(payload))
    bad["representativeness_support"]["FY|200-300|day|low"]["profiles"] = 19
    with pytest.raises(ValueError):
        validate_strict_v2_calibration(bad)


def test_qd_calibration_is_explicit_and_strict_at_resolve(tmp_path):
    config = dict(get_config_mdia())
    config.update({
        "save_dir": str(tmp_path),
        "m00_observation_calibration_semantics": (
            "a1s_m00_qd_residual_empirical_reliability_v1"),
        "magnetic_stratum_semantics": "qd_abs_sin_lat_025_050_v1",
    })
    with pytest.raises(ValueError, match="QD sidecars"):
        _resolve_m00_observation_calibration(config)

    seed = tmp_path / "seed.pth"
    fy_sidecar = tmp_path / "fy_qd.npz"
    cosmic_sidecar = tmp_path / "cosmic_qd.npz"
    for path in (seed, fy_sidecar, cosmic_sidecar):
        path.touch()
    key = "FY|200-300|day|low"
    config.update({
        "m00_observation_seed_checkpoint": str(seed),
        "m00_observation_qd_sidecar_fy": str(fy_sidecar),
        "m00_observation_qd_sidecar_cosmic": str(cosmic_sidecar),
        "m00_observation_calibration": {
            "semantics": "a1s_m00_qd_residual_empirical_reliability_v1",
            "train_only": True,
            "locked_test": "excluded",
            "bias": {key: 0.0},
            "variance": {key: 0.1},
            "representativeness": {},
            "reliability_weight": {},
            "magnetic_coordinate": {
                "semantics": "qd_abs_sin_lat_025_050_v1",
                "qd_converter": "apexpy",
                "apexpy_version": "test",
                "reference_time_utc": "2024-09-01T00:00:00Z",
                "actual_height_semantics": "row_altitude_km_v1",
                "fallback": False,
            },
            "localization_contract": {
                "semantics": "m00_3d_localization_v2",
                "spatial_scale_km": 1800.0,
                "vertical_scale_km": 100.0,
                "temporal_scale_hours": 1.5,
                "time_semantics": "relative_utc_hours_v1",
                "applies_to": "observation_precision_gain_only",
            },
        },
    })
    with pytest.raises(ValueError, match="reliability weights"):
        _resolve_m00_observation_calibration(config)


def test_calibration_detail_loop_does_not_own_analysis_losses():
    source = Path("inr_modules/mdia/train_fsia.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    detail_loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and ast.unparse(node.iter) == "m00_calibration_details.values()"]
    assert len(detail_loops) == 1
    assigned = {target.id for node in ast.walk(detail_loops[0])
                if isinstance(node, ast.Assign) for target in node.targets
                if isinstance(target, ast.Name)}
    assert "covariance_loss" not in assigned
