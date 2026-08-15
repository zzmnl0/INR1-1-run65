"""Regression tests for strict train-only FY/COSMIC observation preloading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import inr_modules.data_managers.FY_dataloader as fy_module
from inr_modules.data_managers.FY_dataloader import (
    COSMICNeighborhoodIndex,
    FYNeighborhoodIndex,
)


INDEX_CASES = (
    (FYNeighborhoodIndex, "fy_profile_index_path", "fy_nb_n_alt"),
    (COSMICNeighborhoodIndex, "cosmic_profile_index_path", "cosmic_nb_n_alt"),
)


def _write_products(tmp_path: Path, *, dirty_disallowed: bool = False):
    rows = np.asarray([
        [1.0, 10.0, 190.0, 10.0, 9.00],
        [1.0, 10.0, 220.0, 10.1, 9.10],
        [1.0, 10.0, 300.0, 10.2, 9.20],
        [2.0, 20.0, 210.0, 20.0, 9.30],
        [2.0, 20.0, 280.0, 20.1, 9.40],
        [2.0, 20.0, 450.0, 20.2, 9.50],
        [3.0, 30.0, 205.0, 30.0, 9.60],
        [3.0, 30.0, 260.0, 30.1, 9.70],
        [3.0, 30.0, 410.0, 30.2, 9.80],
        [3.0, 30.0, 510.0, 30.3, 9.90],
    ], dtype=np.float32)
    if dirty_disallowed:
        rows[3, 4] = np.nan
        rows[4, 4] = np.inf
        rows[5, 4] = -np.inf
    data_path = tmp_path / "physical.npy"
    np.save(data_path, rows)

    # Deliberately keep profile metadata out of output-row order.  This matches
    # the audited products and verifies that output_start drives physical reads.
    index_path = tmp_path / "profile_index.npz"
    np.savez(
        index_path,
        profile_id=np.asarray([20, 10, 40, 30], dtype=np.int64),
        pass_profile=np.asarray([True, True, False, True]),
        output_start=np.asarray([3, 0, -1, 6], dtype=np.int64),
        output_end=np.asarray([6, 3, -1, 10], dtype=np.int64),
    )
    return data_path, index_path


def _config(
        index_path, index_key, n_alt_key, allowed_marker=...,
        *, token_only=False):
    config = {
        index_key: str(index_path),
        n_alt_key: 3,
        "observation_alt_range": (200.0, 500.0),
        "neighbor_directory_semantics": "token_exact_positive_support_v1",
    }
    if allowed_marker is not ...:
        config["strict_preload_allowed_profile_ids"] = allowed_marker
    if token_only:
        config["strict_preload_token_only"] = True
    return config


class _ForbiddenRangeGuard:
    """Array facade that raises if a forbidden physical range is indexed."""

    def __init__(self, values, forbidden_start, forbidden_end):
        self._values = values
        self._forbidden = set(range(forbidden_start, forbidden_end))
        self.shape = values.shape
        self.ndim = values.ndim
        self.dtype = values.dtype
        self.read_rows = []

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        row_key = key[0] if isinstance(key, tuple) else key
        universe = np.arange(len(self._values))
        selected = np.asarray(universe[row_key]).reshape(-1)
        selected_rows = set(map(int, selected.tolist()))
        if selected_rows.intersection(self._forbidden):
            raise AssertionError("forbidden profile physical rows were touched")
        self.read_rows.extend(sorted(selected_rows))
        return self._values[key]


@pytest.mark.parametrize("index_class,index_key,n_alt_key", INDEX_CASES)
def test_strict_preload_never_touches_disallowed_density(
        tmp_path, monkeypatch, index_class, index_key, n_alt_key):
    data_path, index_path = _write_products(
        tmp_path, dirty_disallowed=True)
    real_load = fy_module.np.load
    mapped = real_load(data_path, mmap_mode="r")
    guarded = _ForbiddenRangeGuard(mapped, 3, 6)

    def guarded_load(path, *args, **kwargs):
        if Path(path).resolve() == data_path.resolve():
            return guarded
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(fy_module.np, "load", guarded_load)
    index = index_class(
        str(data_path),
        _config(
            index_path, index_key, n_alt_key,
            np.asarray([10, 30], dtype=np.int64)),
    )
    assert set(np.unique(index.prof_ids).tolist()) == {10, 30}
    assert set(np.unique(index.token_profile_ids).tolist()) == {10, 30}
    assert not set(guarded.read_rows).intersection(range(3, 6))
    assert np.isfinite(index.sorted_data).all()


@pytest.mark.parametrize("index_class,index_key,n_alt_key", INDEX_CASES)
@pytest.mark.parametrize(
    "allowed,error",
    [
        ([10, 99], "absent or non-passing"),
        ([10, 40], "absent or non-passing"),
        ([10, 10], "sorted and unique"),
        ([30, 10], "sorted and unique"),
    ],
)
def test_strict_preload_rejects_invalid_allowlists(
        tmp_path, index_class, index_key, n_alt_key, allowed, error):
    data_path, index_path = _write_products(tmp_path)
    with pytest.raises(ValueError, match=error):
        index_class(
            str(data_path),
            _config(index_path, index_key, n_alt_key, allowed),
        )


@pytest.mark.parametrize("index_class,index_key,n_alt_key", INDEX_CASES)
def test_default_path_matches_strict_all_profiles(
        tmp_path, index_class, index_key, n_alt_key):
    data_path, index_path = _write_products(tmp_path)
    default = index_class(
        str(data_path), _config(index_path, index_key, n_alt_key))
    strict = index_class(
        str(data_path),
        _config(index_path, index_key, n_alt_key, [10, 20, 30]),
    )
    array_fields = (
        "sorted_data",
        "prof_starts",
        "prof_ends",
        "prof_ids",
        "prof_meta",
        "prof_abs_data",
        "prof_valid_mask",
        "prof_sorted_idx",
        "prof_sorted_meta",
        "prof_sorted_abs",
        "prof_sorted_vmask",
        "prof_sorted_ids",
        "token_coords",
        "token_values",
        "token_profile_ids",
        "token_ids",
        "bin_starts",
    )
    for field in array_fields:
        assert np.array_equal(
            getattr(default, field), getattr(strict, field), equal_nan=True), field


def test_fy_and_cosmic_strict_paths_are_equivalent(tmp_path):
    data_path, index_path = _write_products(tmp_path)
    allowed = [10, 30]
    fy = FYNeighborhoodIndex(
        str(data_path),
        _config(index_path, "fy_profile_index_path", "fy_nb_n_alt", allowed),
    )
    cosmic = COSMICNeighborhoodIndex(
        str(data_path),
        _config(
            index_path, "cosmic_profile_index_path", "cosmic_nb_n_alt",
            allowed),
    )
    for field in (
            "sorted_data", "prof_ids", "prof_meta", "prof_abs_data",
            "prof_valid_mask", "token_coords", "token_values",
            "token_profile_ids", "token_ids"):
        assert np.array_equal(
            getattr(fy, field), getattr(cosmic, field), equal_nan=True), field


@pytest.mark.parametrize("index_class,index_key,n_alt_key", INDEX_CASES)
def test_compact_strict_tokens_and_query_match_full_bitwise(
        tmp_path, index_class, index_key, n_alt_key):
    data_path, index_path = _write_products(tmp_path)
    allowed = [10, 20, 30]
    full = index_class(
        str(data_path),
        _config(index_path, index_key, n_alt_key, allowed),
    )
    compact = index_class(
        str(data_path),
        _config(
            index_path, index_key, n_alt_key, allowed, token_only=True),
    )

    for field in (
            "prof_ids", "prof_meta", "prof_abs_data", "prof_valid_mask",
            "token_coords", "token_values", "token_profile_ids", "token_ids"):
        assert np.array_equal(
            getattr(full, field), getattr(compact, field), equal_nan=True), field

    queries = np.asarray([
        [1.0, 10.0, 300.0, 10.1],
        [2.0, 20.0, 300.0, 20.1],
        [3.0, 30.0, 300.0, 30.1],
        [80.0, -150.0, 300.0, 10.1],
    ], dtype=np.float32)
    full_payload = full.query_observation_batch(queries)
    compact_payload = compact.query_observation_batch(queries)
    assert full_payload.keys() == compact_payload.keys()
    for key in full_payload:
        assert np.array_equal(
            full_payload[key], compact_payload[key], equal_nan=True), key

    for physical_field in (
            "sorted_data", "prof_starts", "prof_ends", "prof_sorted_idx",
            "prof_sorted_meta", "prof_sorted_abs", "prof_sorted_vmask",
            "prof_sorted_ids", "bin_starts"):
        assert not hasattr(compact, physical_field), physical_field


@pytest.mark.parametrize("index_class,index_key,n_alt_key", INDEX_CASES)
def test_compact_strict_never_touches_disallowed_density(
        tmp_path, monkeypatch, index_class, index_key, n_alt_key):
    data_path, index_path = _write_products(
        tmp_path, dirty_disallowed=True)
    real_load = fy_module.np.load
    mapped = real_load(data_path, mmap_mode="r")
    guarded = _ForbiddenRangeGuard(mapped, 3, 6)

    def guarded_load(path, *args, **kwargs):
        if Path(path).resolve() == data_path.resolve():
            return guarded
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(fy_module.np, "load", guarded_load)
    compact = index_class(
        str(data_path),
        _config(
            index_path, index_key, n_alt_key, [10, 30], token_only=True),
    )
    assert set(np.unique(compact.token_profile_ids).tolist()) == {10, 30}
    assert not set(guarded.read_rows).intersection(range(3, 6))
    assert not hasattr(compact, "sorted_data")


@pytest.mark.parametrize("index_class,index_key,n_alt_key", INDEX_CASES)
def test_compact_strict_rejects_incompatible_configuration(
        tmp_path, index_class, index_key, n_alt_key):
    data_path, index_path = _write_products(tmp_path)

    missing_allowlist = _config(index_path, index_key, n_alt_key)
    missing_allowlist["strict_preload_token_only"] = True
    with pytest.raises(ValueError, match="requires strict_preload_allowed"):
        index_class(str(data_path), missing_allowlist)

    wrong_semantics = _config(
        index_path, index_key, n_alt_key, [10, 30], token_only=True)
    wrong_semantics["neighbor_directory_semantics"] = "profile_top_k"
    with pytest.raises(ValueError, match="token_exact_positive_support_v1"):
        index_class(str(data_path), wrong_semantics)

    non_boolean = _config(index_path, index_key, n_alt_key, [10, 30])
    non_boolean["strict_preload_token_only"] = 1
    with pytest.raises(ValueError, match="must be boolean"):
        index_class(str(data_path), non_boolean)
