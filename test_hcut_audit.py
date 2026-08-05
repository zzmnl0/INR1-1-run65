"""Focused regression checks for high-h_cut trigger decomposition."""

import numpy as np

from audit_hcut_raw_curves import (
    _bottom_components,
    _date_bootstrap_rate,
    _negative_raw_support,
    _segmented_shape_cut,
    _stratified_sample,
)


def test_bottom_trigger_decomposition():
    grid = np.arange(120.0, 330.1, 2.5)
    density = 1e11 * np.exp((grid - 120.0) / 100.0)
    density[(grid >= 150.0) & (grid <= 165.0)] = np.linspace(
        1.5e11, 1.2e11, np.count_nonzero((grid >= 150.0) & (grid <= 165.0))
    )
    result = _bottom_components(
        grid, density, hmf2=330.0, nmf2=8e11, regime="night"
    )
    assert result["negative_span"] >= 8.0
    assert result["negative_drop"] > 0.0

    weak = density.copy()
    mask = (grid >= 150.0) & (grid <= 165.0)
    weak[mask] = np.linspace(1.50e11, 1.49e11, np.count_nonzero(mask))
    result = _bottom_components(
        grid, weak, hmf2=330.0, nmf2=8e11, regime="night"
    )
    assert result["negative_span"] >= 8.0
    assert result["negative_drop"] < 0.02 * 8e11


def test_segmented_detection_and_raw_support():
    altitude = np.r_[np.arange(120.0, 170.0, 2.5), np.arange(210.0, 330.1, 2.5)]
    density = 1e11 * np.exp((altitude - 120.0) / 100.0)
    cut = _segmented_shape_cut(
        altitude, density, hmf2=330.0, nmf2=8e11, regime="night"
    )
    assert cut == 120.0

    fraction, supported = _negative_raw_support(
        np.arange(120.0, 150.1, 5.0),
        np.arange(7.0, 0.0, -1.0),
        120.0,
        150.0,
    )
    assert fraction == 1.0 and supported


def test_profile_blocked_bootstrap_rate():
    records = [
        {
            "date_code": 20240901 + index % 10,
            "sample_weight": 1.0,
            "confirmed_false_positive": index < 40,
        }
        for index in range(100)
    ]
    rate, interval = _date_bootstrap_rate(records, 100)
    assert np.isclose(rate, 0.4)
    assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_stratified_sample_keeps_rule_flags():
    rows = [
        {
            "source": "FY",
            "regime_code": 3,
            "dominant_trigger": "negative_gradient",
            "h_cut": 220.0,
            "rule_reason": (
                "weak_negative_gradient" if index < 3 else "none"
            ),
            "rule_false_positive": index < 3,
            "date_code": 20240901 + index,
            "latitude": float(index),
            "profile_id": index,
        }
        for index in range(20)
    ]
    selected, _ = _stratified_sample(rows, 10, 42)
    assert set(range(3)).issubset(selected)


if __name__ == "__main__":
    test_bottom_trigger_decomposition()
    test_segmented_detection_and_raw_support()
    test_profile_blocked_bootstrap_rate()
    test_stratified_sample_keeps_rule_flags()
    print("h_cut audit regression checks passed")
