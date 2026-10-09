"""Data-free checks for the sparse-coverage coarse Bias contract."""

import torch

from inr_modules.mdia.train_fsia import background_stratified_bias_loss


def _batch():
    rows = []
    values = []
    ids = []
    for offset, (lat, value) in enumerate(((0.0, 0.1), (-15.0, 0.2),
                                            (45.0, 0.3))):
        for profile in range(8):
            profile_id = offset * 100 + profile
            for altitude in (250.0, 280.0):
                rows.append((lat, 0.0, altitude, 12.0))
                values.append(value)
                ids.append(profile_id)
    return (torch.tensor(values, dtype=torch.float32, requires_grad=True),
            torch.zeros(len(values)), torch.tensor(rows), torch.tensor(ids))


def test_coarse_magnetic_grouping_is_profile_balanced_and_bounded():
    prediction, target, coords, profile_ids = _batch()
    loss, details = background_stratified_bias_loss(
        prediction, target, coords, profile_ids,
        grouping="height_daynight_plus_magnetic_v1",
        magnetic_support_counts={
            "abs_sinI_lt_0.25": 30,
            "abs_sinI_0.25_0.5": 30,
            "abs_sinI_ge_0.5": 30,
        },
        return_details=True)
    assert torch.isfinite(loss)
    assert details["magnetic_active_cell_count"] == 3
    assert all(item["profile_count"] == 8
               for item in details["magnetic"].values())
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_coarse_magnetic_grouping_falls_back_for_sparse_batch_and_legacy_is_stable():
    prediction, target, coords, profile_ids = _batch()
    sparse, details = background_stratified_bias_loss(
        prediction, target, coords, profile_ids,
        grouping="height_daynight_plus_magnetic_v1",
        magnetic_support_counts={
            "abs_sinI_lt_0.25": 30,
            "abs_sinI_0.25_0.5": 30,
            "abs_sinI_ge_0.5": 30,
        },
        magnetic_min_batch_profiles=9,
        return_details=True)
    legacy = background_stratified_bias_loss(
        prediction, target, coords, profile_ids,
        grouping="height_daynight_v1")
    assert details["magnetic_active_cell_count"] == 0
    assert torch.allclose(sparse, legacy)


def test_support_threshold_and_altitude_edges_are_explicit():
    rows = torch.tensor([
        [0.0, 0.0, 200.0, 12.0],
        [0.0, 0.0, 299.9, 12.0],
        [0.0, 0.0, 300.0, 12.0],
        [0.0, 0.0, 500.0, 12.0],
        [0.0, 0.0, 500.1, 12.0],
        [0.0, 0.0, 200.0, 0.0],
        [0.0, 0.0, 300.0, 0.0],
    ])
    prediction = torch.ones(len(rows), requires_grad=True)
    target = torch.zeros(len(rows))
    profile_ids = torch.arange(len(rows))
    _, details = background_stratified_bias_loss(
        prediction, target, rows, profile_ids,
        grouping="height_daynight_plus_magnetic_v1",
        magnetic_support_counts={
            "abs_sinI_lt_0.25": 30,
            "abs_sinI_0.25_0.5": 30,
            "abs_sinI_ge_0.5": 30,
        },
        magnetic_min_batch_profiles=1,
        return_details=True)
    assert details["200-300_day"]["point_count"] == 2
    assert details["300-500_day"]["point_count"] == 2
    assert details["200-300_night"]["point_count"] == 1
    assert details["300-500_night"]["point_count"] == 1
    populated = [item for item in details["magnetic"].values()
                 if item["profile_count"] > 0]
    assert populated and all(item["active"] for item in populated)

    _, sparse_details = background_stratified_bias_loss(
        prediction, target, rows, profile_ids,
        grouping="height_daynight_plus_magnetic_v1",
        magnetic_support_counts={
            "abs_sinI_lt_0.25": 29,
            "abs_sinI_0.25_0.5": 29,
            "abs_sinI_ge_0.5": 29,
        },
        magnetic_min_batch_profiles=1,
        return_details=True)
    assert sparse_details["magnetic_active_cell_count"] == 0


def test_magnetic_daynight_keeps_full_height_loss_and_uses_six_cells():
    prediction, target, coords, profile_ids = _batch()
    coords = coords.clone()
    coords[::2, 3] = 0.0
    support = {f"{mag}_{period}": 30
               for mag in ("abs_sinI_lt_0.25", "abs_sinI_0.25_0.5",
                           "abs_sinI_ge_0.5")
               for period in ("night", "day")}
    loss, details = background_stratified_bias_loss(
        prediction, target, coords, profile_ids,
        grouping="height_daynight_plus_magnetic_daynight_v1",
        magnetic_support_counts=support, magnetic_weight=1.0,
        magnetic_min_batch_profiles=1, return_details=True)
    legacy, legacy_details = background_stratified_bias_loss(
        prediction.detach().clone(), target, coords, profile_ids,
        grouping="height_daynight_v1", return_details=True)
    assert details["magnetic_active_cell_count"] == 6
    assert torch.allclose(loss, legacy + details["magnetic_loss"], atol=1e-6)
    assert torch.allclose(
        torch.as_tensor(details["height_daynight_loss"]),
        torch.as_tensor(legacy_details["height_daynight_loss"]), atol=1e-6)


if __name__ == "__main__":
    test_coarse_magnetic_grouping_is_profile_balanced_and_bounded()
    test_coarse_magnetic_grouping_falls_back_for_sparse_batch_and_legacy_is_stable()
    print("ok")
