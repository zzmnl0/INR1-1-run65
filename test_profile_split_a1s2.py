"""Focused checks for the isolated A1-S2 profile split builder."""

import json

import numpy as np

from inr_modules.data_managers.profile_split import (
    SEMANTICS,
    build_source_split,
    load_profile_split,
    physical_cell_labels,
    save_profile_split,
)


def _rows(profile_ids, hours, *, altitude=250.0, longitude=0.0):
    profile_ids = np.asarray(profile_ids, dtype=np.int64)
    hours = np.asarray(hours, dtype=np.float64)
    return np.column_stack((
        np.zeros(len(profile_ids)),
        np.full(len(profile_ids), longitude),
        np.full(len(profile_ids), altitude),
        hours,
    )), profile_ids


def test_time_local_five_profile_blocks_are_fixed_and_disjoint():
    data, ids = _rows(np.arange(10), np.arange(10, dtype=float) * 0.1)
    first = build_source_split("FY", data, ids, seed=42)
    second = build_source_split("FY", data, ids, seed=42)
    assert np.array_equal(first["train"], second["train"])
    assert np.array_equal(first["development"], second["development"])
    assert len(first["development"]) == 2
    assert not np.intersect1d(first["train"], first["development"]).size
    assert np.array_equal(
        np.sort(np.concatenate((first["train"], first["development"]))), ids)
    assert first["audit"]["grouping"]["complete_five_profile_group_count"] == 2


def test_time_gap_and_short_tail_stay_train():
    ids = np.arange(9)
    hours = np.asarray([0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 2.2, 2.3, 2.4])
    data, ids = _rows(ids, hours)
    result = build_source_split("COSMIC", data, ids, seed=42)
    assert 5 in result["train"]
    assert np.all(np.isin(np.arange(5, 9), result["train"]))
    assert result["audit"]["grouping"]["gap_singleton_train_count"] >= 1


def test_physical_boundaries_and_development_only_is_a_warning():
    # Dipole values are supplied so the boundary test is independent of the
    # coordinate approximation used by the production model.
    data, ids = _rows(np.arange(5), np.arange(5, dtype=float) * 0.05)
    data[0, 2] = 300.0
    data[0, 3] = 6.0
    labels = physical_cell_labels(data, np.asarray([0.25, 0.5, 0.1, 0.6, 0.9]))
    assert labels[0] == "300_500_day_mid"
    assert labels[1] == "200_300_night_high"

    # One profile in each physical cell makes whichever random profile is
    # selected for Development a development-only cell.  This must warn, not
    # abort the split.
    cell_data = np.asarray([
        [0.0, 0.0, 250.0, 0.00],
        [0.0, 0.0, 250.0, 0.05],
        [0.0, 0.0, 250.0, 0.10],
        [0.0, 0.0, 350.0, 0.15],
        [0.0, 0.0, 350.0, 0.20],
    ])
    with np.testing.assert_warns(RuntimeWarning):
        result = build_source_split(
            "FY", cell_data, np.arange(5), seed=42,
            magnetic_sin_i=np.asarray([0.1, 0.3, 0.6, 0.1, 0.3]))
    assert any(
        row["status"] == "development_only_extrapolation"
        for row in result["audit"]["physical_cell_coverage"])


def test_npz_manifest_has_only_train_and_development(tmp_path):
    data, ids = _rows(np.arange(5), np.arange(5, dtype=float) * 0.05)
    split = {"semantics": SEMANTICS, "seed": 42,
             "sources": {"FY": build_source_split("FY", data, ids)},
             "audit": {"ok": True}}
    npz_path = tmp_path / "profile_split.npz"
    audit_path = tmp_path / "profile_split_audit.json"
    save_profile_split(npz_path, split, audit_path=audit_path)
    loaded = load_profile_split(npz_path, expected_seed=42)
    assert set(loaded["sources"]) == {"FY"}
    assert set(loaded["sources"]["FY"]) == {"train", "development"}
    with audit_path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    assert audit["ok"] is True


if __name__ == "__main__":
    test_time_local_five_profile_blocks_are_fixed_and_disjoint()
    test_time_gap_and_short_tail_stay_train()
    test_physical_boundaries_and_development_only_is_a_warning()
    print("A1-S2 profile split checks passed")
