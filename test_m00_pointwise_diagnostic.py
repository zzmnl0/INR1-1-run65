import numpy as np

from inr_modules.mdia.m00_residual_experiment import _migration_audit
from prepare_m00_residual_reliability import _select_profile_queries


def test_profile_migration_uses_fractional_point_contributions():
    coords = np.asarray([
        [0.0, 0.0, 250.0, 0.0],
        [0.0, 1.0, 250.0, 0.0],
    ])
    report = _migration_audit(
        np.asarray([10.0, 60.0]), np.asarray([10.0, 10.0]),
        np.asarray([7, 7]), np.asarray(["FY", "FY"]), coords)
    assert report["profile_matrix"]["low|low"] == 0.5
    assert report["profile_matrix"]["high|low"] == 0.5
    assert report["review_gate_enabled"] is False


def test_localization_queries_keep_distinct_profile_points():
    coords = np.asarray([
        [0.0, 0.0, 200.0, 0.0],
        [0.0, 10.0, 250.0, 1.0],
        [0.0, 20.0, 300.0, 2.0],
        [0.0, 30.0, 350.0, 3.0],
    ])
    query, pids, audit = _select_profile_queries(coords, np.full(4, 9), 1)
    assert query.shape == (4, 4)
    assert np.array_equal(query, coords)
    assert np.array_equal(pids, np.full(4, 9))
    assert audit["query_points"] == 4
