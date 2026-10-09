"""Focused checks for the shared spatial Development profile split."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from inr_modules.data_managers.FY_dataloader import ProfileTimeBinSampler
from inr_modules.mdia.train_fsia import (
    _shared_spatial_profile_split,
    _spatial_profile_strata,
)


def _metadata(count, *, first_id=0, hour=0.0):
    ids = np.arange(first_id, first_id + count, dtype=np.float64)
    return np.column_stack((
        ids,
        np.zeros(count),
        np.zeros(count),
        np.full(count, hour),
    ))


def test_spatial_strata_boundaries():
    metadata = np.asarray([
        [0, -90.0, 0.0, 6.0],
        [1, -70.0, 30.0, 18.0],
        [2, 90.0, -0.1, 0.0],
    ])
    mocked_sin_i = torch.tensor([0.249999, 0.25, -0.5], dtype=torch.float64)
    with patch(
            'inr_modules.mdia.train_fsia._compute_dip_features',
            return_value=(mocked_sin_i, torch.zeros_like(mocked_sin_i))):
        keys = _spatial_profile_strata(metadata, 'FY')

    assert keys.tolist() == [
        ['FY', 0, 0, 1, 0],   # LT 06 is day; |sin I| < 0.25.
        ['FY', 1, 1, 0, 1],   # LT 20 is night; 0.25 enters mid.
        ['FY', 8, 11, 0, 2],  # longitude wraps; 0.5 enters high.
    ]


def test_spatial_split_threshold_locked_days_and_seed():
    sparse = _shared_spatial_profile_split(
        _metadata(19), 'FY', locked_test_days=[], seed=42)
    assert sparse['development'].size == 0
    assert sparse['train'].size == 19

    metadata = _metadata(41)
    first = _shared_spatial_profile_split(
        metadata, 'FY', locked_test_days=[], seed=42)
    repeated = _shared_spatial_profile_split(
        metadata, 'FY', locked_test_days=[], seed=42)
    other_seed = _shared_spatial_profile_split(
        metadata, 'FY', locked_test_days=[], seed=43)
    assert np.array_equal(first['train'], repeated['train'])
    assert np.array_equal(first['development'], repeated['development'])
    assert first['development'].size == 2
    assert not np.array_equal(first['development'], other_seed['development'])
    assert np.intersect1d(first['train'], first['development']).size == 0

    # Profiles whose representative time is on locked day 1 never enter either set.
    mixed_days = np.vstack((_metadata(20), _metadata(20, first_id=20, hour=24.0)))
    locked = _shared_spatial_profile_split(
        mixed_days, 'FY', locked_test_days=[1], seed=42)
    assert np.all(locked['train'] < 20)
    assert np.all(locked['development'] < 20)
    assert locked['train'].size == 19
    assert locked['development'].size == 1


def test_profile_point_replication_and_short_profile_sampling():
    small = SimpleNamespace(
        profile_ids=np.asarray([10, 10, 11, 11]),
        bin_ids=np.zeros(4, dtype=np.int64),
    )
    replicated = SimpleNamespace(
        profile_ids=np.asarray([10] * 20 + [11, 11]),
        bin_ids=np.zeros(22, dtype=np.int64),
    )
    small_sampler = ProfileTimeBinSampler(
        small, batch_size=16, points_per_profile=4, shuffle=False)
    replicated_sampler = ProfileTimeBinSampler(
        replicated, batch_size=16, points_per_profile=4, shuffle=False)
    assert len(small_sampler.profiles_by_bin[0]) == 2
    assert len(replicated_sampler.profiles_by_bin[0]) == 2

    small_sampler.replace_short_profiles = False
    indices = np.asarray([3, 7])
    assert np.array_equal(small_sampler._sample_profile(indices), indices)
    assert len(set(next(iter(small_sampler)))) == 4

    # Short profiles must remain a finite set of observations even when the
    # training sampler shuffles; padding by replacement creates fake exposure.
    shuffled = ProfileTimeBinSampler(
        small, batch_size=16, points_per_profile=4, shuffle=True,
        schedule_seed=42)
    sampled = list(iter(shuffled))[0]
    assert len(sampled) == len(set(sampled))


if __name__ == '__main__':
    test_spatial_strata_boundaries()
    test_spatial_split_threshold_locked_days_and_seed()
    test_profile_point_replication_and_short_profile_sampling()
    print('shared spatial split checks passed')
