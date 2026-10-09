import json
import sys
import types

import numpy as np

from generate_qd_sidecar import generate_qd_sidecar
from inr_modules.mdia.m00_observation import load_magnetic_sidecar
from inr_modules.mdia.m00_residual_experiment import _migration_audit


def test_qd_sidecar_is_exactly_row_aligned_and_keeps_dipole(tmp_path, monkeypatch):
    class FakeApex:
        def __init__(self, date):
            self.date = date

        def convert(self, lat, lon, source, dest, *, height, datetime):
            assert source == "geo" and dest == "qd"
            return np.asarray(lat) + 1.0, np.asarray(lon) - 2.0

    monkeypatch.setitem(sys.modules, "apexpy", types.SimpleNamespace(
        __version__="fake-1", Apex=FakeApex))
    data = np.asarray([
        [10.0, 20.0, 250.0, 0.0],
        [11.0, 21.0, 300.0, 0.0],
        [12.0, 22.0, 400.0, 1.0],
    ], dtype=np.float32)
    data_path = tmp_path / "data.npy"
    index_path = tmp_path / "index.npz"
    output_path = tmp_path / "qd.npz"
    np.save(data_path, data)
    np.savez(index_path, profile_id=np.asarray([4, 5]),
             output_start=np.asarray([0, 2]), output_end=np.asarray([2, 3]))
    result = generate_qd_sidecar(str(data_path), str(index_path), str(output_path),
                                 source="FY", include_aacgm_audit=False)
    assert result["semantics"] == "qd_abs_sin_lat_025_050_v1"
    sidecar = load_magnetic_sidecar(str(output_path), expected_rows=3,
                                    strict_qd=True, expected_profile_ids=np.asarray([4, 4, 5]),
                                    expected_coords=data)
    assert np.allclose(sidecar["qd_latitude_deg"], data[:, 0] + 1.0)
    assert np.isfinite(sidecar["magnetic_sin_i"]).all()
    # A second run reuses the row-identity checked chunk cache.
    generate_qd_sidecar(str(data_path), str(index_path), str(output_path),
                        source="FY", include_aacgm_audit=False)


def test_qd_migration_flags_registered_risk():
    qd = np.asarray([60.0] * 19 + [10.0])
    aacgm = np.asarray([10.0] * 19 + [60.0])
    audit = _migration_audit(qd, aacgm, np.arange(20),
                            np.full(20, "FY"), np.zeros((20, 4)))
    assert audit["risk_exceeded"] is True
    assert audit["risk_threshold"] == 0.05
