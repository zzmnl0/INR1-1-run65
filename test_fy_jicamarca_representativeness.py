import numpy as np

from audit_fy_jicamarca_representativeness import (
    _attribution_and_gate,
    _bootstrap_mean,
    _combine_pair_arrays,
    _profile_equal_summary,
)
from audit_fy_jicamarca_metadata_bias import (
    _rank_average,
    _rank_correlation,
)
from validate_fy_hmf2_cross_source import _evaluate


def _pairs():
    return {
        "profile_id": np.array([1, 1, 2, 3]),
        "query_index": np.arange(4),
        "profile_date": np.array([20240901, 20240901, 20240901, 20240902]),
        "mode": np.array(["M10"] * 4),
        "support": np.array(["within_20km"] * 4),
        "high_confidence": np.ones(4, dtype=bool),
        "innovation_toward": np.array([True, True, False, True]),
        "contribution_toward": np.array([True, False, False, True]),
        "contribution": np.array([1.0, -1.0, -1.0, 1.0]),
        "cross_covariance": np.array([1.0, -1.0, 1.0, 1.0]),
        "kalman_gain": np.array([1.0, -1.0, 1.0, 1.0]),
        "rho": np.array([0.1, 0.2, 0.3, 0.4]),
    }


def test_profile_equal_summary_and_bootstrap_are_deterministic():
    pairs = _pairs()
    result = _profile_equal_summary(
        pairs, np.ones(4, dtype=bool), replicates=100, seed=42
    )
    # Profile 1 contributes 1.0, profile 2 contributes 0.0, profile 3 contributes 1.0.
    assert result["innovation_toward_isr"]["value"] == 2.0 / 3.0
    first = _bootstrap_mean([0.0, 1.0, 1.0], [1, 1, 2], 100, 42)
    second = _bootstrap_mean([0.0, 1.0, 1.0], [1, 1, 2], 100, 42)
    assert first == second


def test_gate_stops_on_representativeness_conflict():
    pairs = _pairs()
    summaries = {
        "rho_median": 0.5,
        "modes": {
            "M10": {
                "within_20km_nearest_half": {
                    "innovation_toward_isr": {
                        "value": 0.54,
                        "ci95": [0.49, 0.60],
                    },
                    "contribution_toward_isr": {
                        "value": 0.70,
                        "ci95": [0.60, 0.80],
                    },
                }
            }
        },
    }
    _, gate = _attribution_and_gate(pairs, summaries)
    assert not gate["passed"]
    assert gate["conclusion"] == "FY_Jicamarca_representativeness_conflict"


def test_combines_mode_and_support_groups():
    combined = _combine_pair_arrays({
        "M10_all": {"value": np.array([1])},
        "M10_within": {"value": np.array([2])},
        "M11_all": {"value": np.array([3])},
    })
    assert combined["value"].tolist() == [1, 2, 3]


def test_rank_correlation_handles_ties_and_direction():
    assert _rank_average([1.0, 1.0, 3.0]).tolist() == [0.5, 0.5, 2.0]
    assert np.isclose(_rank_correlation([1, 2, 3], [3, 2, 1]), -1.0)


def test_hmf2_validation_rejects_insufficient_profiles():
    rows = np.zeros(10, dtype=[
        ("date", "i2"), ("hmf2", "f8"), ("direction_agreement", "f8"),
        ("pair_mass", "f8"),
    ])
    assert not _evaluate(rows, 10, 42)["passed"]
