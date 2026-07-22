"""Verify run65 uses true FY/COSMIC profiles for the September sources."""

import numpy as np

from inr_modules.config_mdia import CONFIG_MDIA
from inr_modules.data_managers.FY_dataloader import (
    COSMICNeighborhoodIndex,
    FYNeighborhoodIndex,
)


def _check(index, expected_profiles):
    assert len(index.prof_starts) == expected_profiles
    assert len(np.unique(index.prof_ids)) == expected_profiles
    assert index.prof_sorted_abs.shape == (expected_profiles, 8, 5)
    assert np.all(np.diff(index.prof_sorted_meta[:, 2]) >= 0)


if __name__ == "__main__":
    _check(FYNeighborhoodIndex(CONFIG_MDIA["fy_path"], CONFIG_MDIA), 75_766)
    _check(COSMICNeighborhoodIndex(CONFIG_MDIA["cosmic_path"], CONFIG_MDIA), 105_182)
    print("FY/COSMIC true-profile grouping verified")
