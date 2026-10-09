"""Profile metric meanings and reusable date aggregates."""

import numpy as np

from inr_modules.mdia.train_fsia import _background_stratified_metrics


def test_profile_metrics_do_not_mix_inner_rmse_with_profile_mean_rmse():
    # Profile 1 has cancelling point errors; profile 2 has a constant error.
    prediction = np.array([1., 3., 4.])
    target = np.array([2., 2., 3.])
    coords = np.array([[0., 0., 250., 1.], [0., 0., 250., 1.],
                       [0., 0., 250., 25.]])
    result = _background_stratified_metrics(
        prediction, target, target, coords, prediction - target,
        prediction - target, np.ones(3), profile_ids=np.array([10, 10, 20]))
    cell = result['cells']['200-300_night']['M00']
    assert cell['profile_count'] == 2
    assert cell['point_count'] == 3
    assert np.isclose(cell['mean_profile_rmse'], 1.)
    assert np.isclose(cell['profile_mean_rmse'], np.sqrt(.5))
    assert cell['rmse'] == cell['profile_mean_rmse']
    dates = cell['date_sufficient_statistics']
    assert [row['relative_utc_day'] for row in dates] == [0, 1]
    assert sum(row['profile_count'] for row in dates) == 2
    assert sum(row['point_count'] for row in dates) == 3
    assert np.isclose(sum(row['sum_error_sq'] for row in dates), 1.)
    assert np.isclose(sum(row['sum_profile_rmse'] for row in dates), 2.)
