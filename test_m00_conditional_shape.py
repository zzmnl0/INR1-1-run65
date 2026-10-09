"""Focused, data-free checks for the conditional M00 shape objective."""

import torch

from inr_modules.mdia.train_fsia import background_conditional_shape_loss


def _batch(n=12, altitude=250.0, local_time=12.0):
    lat = torch.linspace(-55.0, 55.0, n)
    lon = torch.linspace(-170.0, 170.0, n)
    coords = torch.stack([
        lat,
        lon,
        torch.full((n,), altitude),
        torch.full((n,), local_time),
    ], dim=1)
    ids = torch.arange(n, dtype=torch.long)
    target = torch.linspace(10.0, 12.0, n)
    prediction = target + torch.sin(torch.arange(n, dtype=torch.float32)) * 0.1
    return prediction, target, coords, ids


def test_profile_replication_does_not_change_loss():
    prediction, target, coords, ids = _batch()
    first = background_conditional_shape_loss(
        prediction, target, coords, ids, min_profiles_per_cell=1)
    repeated = background_conditional_shape_loss(
        torch.cat([prediction, prediction[:3]]),
        torch.cat([target, target[:3]]),
        torch.cat([coords, coords[:3]]),
        torch.cat([ids, ids[:3]]),
        min_profiles_per_cell=1,
    )
    assert torch.allclose(first, repeated, atol=1e-6, rtol=1e-6)


def test_common_offset_does_not_change_anomaly_or_scale():
    prediction, target, coords, ids = _batch()
    first = background_conditional_shape_loss(
        prediction, target, coords, ids, min_profiles_per_cell=1)
    shifted = background_conditional_shape_loss(
        prediction + 3.0, target + 3.0, coords, ids, min_profiles_per_cell=1)
    assert torch.allclose(first, shifted, atol=1e-5, rtol=1e-5)


def test_matching_profile_spread_has_zero_shape_loss():
    prediction, target, coords, ids = _batch()
    prediction = target.clone()
    loss = background_conditional_shape_loss(
        prediction, target, coords, ids, min_profiles_per_cell=1)
    assert float(loss) < 1e-8


def test_collapsed_prediction_spread_is_penalized():
    prediction, target, coords, ids = _batch()
    loss = background_conditional_shape_loss(
        torch.full_like(target, target.mean()), target, coords, ids,
        min_profiles_per_cell=1)
    assert float(loss) > 0.0


def test_altitude_and_day_night_boundaries_are_finite():
    for altitude, local_time in ((200.0, 6.0), (299.999, 18.0),
                                 (300.0, 5.999), (500.0, 23.999)):
        prediction, target, coords, ids = _batch(
            altitude=altitude, local_time=local_time)
        value = background_conditional_shape_loss(
            prediction, target, coords, ids, min_profiles_per_cell=1)
        assert torch.isfinite(value)


if __name__ == '__main__':
    test_profile_replication_does_not_change_loss()
    test_common_offset_does_not_change_anomaly_or_scale()
    test_matching_profile_spread_has_zero_shape_loss()
    test_collapsed_prediction_spread_is_penalized()
    test_altitude_and_day_night_boundaries_are_finite()
    print('ok')
