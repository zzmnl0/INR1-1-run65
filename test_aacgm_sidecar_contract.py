import numpy as np
import pytest

from generate_aacgm_sidecar import generate_sidecar


def test_sidecar_skips_rejected_index_profiles_and_preserves_rows(tmp_path):
    data = np.asarray(
        [[0.0, 0.0, 300.0, 0.0], [1.0, 1.0, 310.0, 0.0],
         [2.0, 2.0, 320.0, 1.0]], dtype=np.float32)
    data_path = tmp_path / "data.npy"
    index_path = tmp_path / "index.npz"
    output_path = tmp_path / "sidecar.npz"
    np.save(data_path, data)
    np.savez(
        index_path,
        profile_id=np.asarray([10, 11, 12]),
        output_start=np.asarray([0, -1, 2]),
        output_end=np.asarray([2, -1, 3]),
    )

    result = generate_sidecar(
        str(data_path), str(index_path), str(output_path),
        source="FY", batch_rows=2,
    )
    assert result["rows"] == 3
    with np.load(output_path, allow_pickle=False) as payload:
        assert np.array_equal(payload["row_index"], np.arange(3))
        assert np.array_equal(payload["profile_id"], np.asarray([10, 10, 12]))
        assert np.isfinite(payload["aacgm_latitude_deg"]).all()
        assert np.isfinite(payload["aacgm_longitude_deg"]).all()


def test_sidecar_rejects_aacgm_undefined_rows(tmp_path):
    data_path = tmp_path / "data.npy"
    index_path = tmp_path / "index.npz"
    output_path = tmp_path / "sidecar.npz"
    np.save(data_path, np.asarray([[5.4996, -42.7299, 348.883, 1.1833333]], dtype=np.float32))
    np.savez(index_path, profile_id=np.asarray([1]),
             output_start=np.asarray([0]), output_end=np.asarray([1]))
    with pytest.raises(ValueError, match="non-finite"):
        generate_sidecar(str(data_path), str(index_path), str(output_path), source="FY")
