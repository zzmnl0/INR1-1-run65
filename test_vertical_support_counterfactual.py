"""Regression check for vertical observation masks."""

import numpy as np
import torch

from diagnose_vertical_support_counterfactual import (
    _source_physics,
    _vertical_mask,
)
from isr_evaluation.diagnose_jicamarca_modes import _record_arrays


def test_vertical_masks_preserve_semantics():
    payload = {
        "valid_mask": torch.tensor([[True, True, True, False]]),
        "coords": torch.tensor(
            [[[0.0, 0.0, 190.0, 0.0],
              [0.0, 0.0, 205.0, 0.0],
              [0.0, 0.0, 225.0, 0.0],
              [0.0, 0.0, 190.0, 0.0]]]
        ),
    }
    query = torch.tensor([[0.0, 0.0, 200.0, 0.0]])
    assert torch.equal(
        _vertical_mask(payload, query, "all"),
        payload["valid_mask"],
    )
    assert torch.equal(
        _vertical_mask(payload, query, "same_layer"),
        torch.tensor([[False, True, True, False]]),
    )
    assert torch.equal(
        _vertical_mask(payload, query, "within_20km"),
        torch.tensor([[True, True, False, False]]),
    )


def test_record_arrays_uses_pointwise_poker_flat_geography():
    start_unix = 1_000.0
    record = {
        "alt_1d": np.asarray([120.0, 200.0], dtype=np.float32),
        "ts_1d": np.asarray([1_000.0, 4_600.0]),
        "ne_2d": np.asarray([[1e10, np.nan], [1e11, 1e12]]),
        "lat": None,
        "lon": None,
        "geo_lat_2d": np.asarray([[65.0, np.nan], [66.0, np.nan]]),
        "geo_lon_2d": np.asarray([[-147.0, np.nan], [-146.0, np.nan]]),
    }
    coords, observation = _record_arrays(record, start_unix)
    assert np.array_equal(
        coords,
        np.asarray([
            [65.0, -147.0, 120.0, 0.0],
            [66.0, -146.0, 200.0, 0.0],
        ], dtype=np.float32),
    )
    assert np.allclose(observation, [10.0, 11.0])


def test_source_physics_coverage_uses_group_population():
    arrays = {
        "valid_tokens": np.asarray([1, 0, 1, 1]),
        "physical_innovation_mean": np.asarray([-1.0, 0.0, 1.0, 1.0]),
        "kalman_gain_mean": np.ones(4),
        "cross_covariance_mean": np.ones(4),
        "kalman_contribution_sum": np.asarray([-1.0, 0.0, 1.0, 1.0]),
        "effective_sample_size": np.ones(4),
    }
    result = _source_physics(
        arrays,
        np.asarray([-1.0, -1.0, 1.0, 1.0]),
        np.asarray([True, True, False, False]),
    )
    assert result["n"] == 1
    assert result["coverage"] == 0.5


if __name__ == "__main__":
    test_vertical_masks_preserve_semantics()
    test_record_arrays_uses_pointwise_poker_flat_geography()
    test_source_physics_coverage_uses_group_population()
    print("vertical support regression check passed")
