"""Focused regression checks for profile QC and NPY+NPZ indexing."""

import tempfile
from pathlib import Path

import numpy as np

from qc_profile_data import (
    FY_DENSITY_MAX_M3,
    FY_DENSITY_MIN_M3,
    _output_rows,
    evaluate_profile,
)
from inr_modules.data_managers.FY_dataloader import _load_profile_index


def test_qc_preserves_observations_and_index_boundaries():
    altitude = np.arange(120.0, 500.1, 2.5)
    density = 1e12 * np.exp(-((altitude - 300.0) / 90.0) ** 2)
    density[20] *= 1.001
    physical = np.column_stack([
        np.zeros_like(altitude), np.zeros_like(altitude), altitude, density,
    ])
    result = evaluate_profile(
        physical, profile_id=7, source='FY3D', relative_hour=72.0)
    assert result.pass_profile, result
    output = _output_rows(physical, result)
    assert np.array_equal(output[:, 2], altitude.astype(np.float32))
    assert np.allclose(output[:, 4], np.log10(density).astype(np.float32))

    with tempfile.TemporaryDirectory() as directory:
        index_path = Path(directory) / 'index.npz'
        np.savez(
            index_path,
            profile_id=np.asarray([7, 8, 9]),
            pass_profile=np.asarray([True, False, True]),
            output_start=np.asarray([0, -1, 3]),
            output_end=np.asarray([3, -1, 5]),
        )
        row_ids, split_population = _load_profile_index(index_path, 5)
    assert np.array_equal(row_ids, [7, 7, 7, 9, 9])
    assert np.array_equal(split_population, [7, 8, 9])


def test_fy_physical_range_is_removed_before_qc_and_output():
    altitude = np.arange(120.0, 500.1, 2.5)
    density = 1e12 * np.exp(-((altitude - 300.0) / 90.0) ** 2)
    density[2] = FY_DENSITY_MIN_M3 / 10.0
    density[-3] = FY_DENSITY_MAX_M3 * 10.0
    physical = np.column_stack([
        np.zeros_like(altitude), np.zeros_like(altitude), altitude, density,
    ])

    result = evaluate_profile(
        physical, profile_id=8, source='FY3D', relative_hour=72.0)
    assert result.pass_profile, result
    assert result.range_rejected_points == 2

    output = _output_rows(physical, result)
    assert len(output) == len(physical) - 2
    assert np.all(output[:, 4] >= np.log10(FY_DENSITY_MIN_M3))
    assert np.all(output[:, 4] <= np.log10(FY_DENSITY_MAX_M3))


if __name__ == '__main__':
    test_qc_preserves_observations_and_index_boundaries()
    test_fy_physical_range_is_removed_before_qc_and_output()
    print('QC profile regression checks passed')
