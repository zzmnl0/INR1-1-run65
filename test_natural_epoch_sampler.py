"""Regression checks for the natural complete-profile Background pass."""

from types import SimpleNamespace

import numpy as np

from inr_modules.data_managers.FY_dataloader import ProfileTimeBinSampler


def test_natural_pass_covers_each_profile_once_and_len_matches_plan():
    counts = (3, 5, 2, 6)
    profile_ids = np.concatenate([
        np.full(count, profile, dtype=np.int64)
        for profile, count in enumerate(counts)])
    dataset = SimpleNamespace(
        profile_ids=profile_ids,
        bin_ids=np.arange(len(profile_ids), dtype=np.int64) % 3,
        data=np.zeros((len(profile_ids), 4), dtype=np.float32),
        selected_indices=np.arange(len(profile_ids), dtype=np.int64),
    )
    sampler = ProfileTimeBinSampler(
        dataset, batch_size=7, points_per_profile=None, shuffle=True,
        sampling_semantics='source_complete_profile_pass_v1', schedule_seed=42)

    expected_batches = len(sampler)
    batches = list(iter(sampler))
    consumed = np.concatenate([profile_ids[batch] for batch in batches])

    assert expected_batches == len(batches)
    assert np.array_equal(np.unique(consumed), np.arange(len(counts)))
    assert [int((consumed == profile).sum()) for profile in range(len(counts))] == list(counts)


def test_calibration_sampler_reset_preserves_first_natural_epoch_order():
    from inr_modules.mdia.train_fsia import _reset_natural_sampler_state
    ids = np.repeat(np.arange(20), 8)
    dataset = SimpleNamespace(profile_ids=ids, bin_ids=np.zeros(len(ids), dtype=int),
                              data=np.zeros((len(ids), 4)), selected_indices=np.arange(len(ids)))
    sampler = ProfileTimeBinSampler(dataset, batch_size=32, points_per_profile=8,
                                   shuffle=True, sampling_semantics='source_complete_profile_pass_v1',
                                   schedule_seed=42)
    expected = [list(batch) for batch in sampler]
    list(sampler)  # calibration consumes a plan
    loader = SimpleNamespace(batch_sampler=sampler)
    _reset_natural_sampler_state(loader, loader, start_epoch=0)
    assert [list(batch) for batch in sampler] == expected
