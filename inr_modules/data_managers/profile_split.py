"""Deterministic A1-S2 source-local profile split utilities.

The split is made once from profile representative UTC time.  It deliberately
does not know about training, validation, or model code; callers consume the
sorted profile allowlists written by :func:`save_profile_split`.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np


SEMANTICS = "source_time_local_block80_dev20_v1"
SCHEMA_VERSION = 1
SOURCES = ("FY", "COSMIC")
BLOCK_SIZE = 5
BLOCK_HOURS = 0.5

_ALTITUDE_BANDS = (
    ("200_300", 200.0, 300.0, False),
    ("300_500", 300.0, 500.0, True),
)
_MAGNETIC_BANDS = (("low", 0.25), ("mid", 0.50), ("high", 1.01))


def _as_1d(name: str, values, *, dtype=None) -> np.ndarray:
    array = np.asarray(values if values is not None else [])
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _validate_rows(data, profile_ids) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("data must have at least lat, lon, altitude, time columns")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("data must be numeric")
    ids_raw = _as_1d("profile_ids", profile_ids)
    if len(ids_raw) != len(data):
        raise ValueError("data and profile_ids must have equal row counts")
    if ids_raw.dtype.kind not in "iu":
        if (ids_raw.dtype.kind not in "f"
                or not np.isfinite(ids_raw).all()
                or not np.array_equal(ids_raw, np.rint(ids_raw))):
            raise ValueError("profile_ids must contain finite integer values")
    ids = np.asarray(np.rint(ids_raw), dtype=np.int64)
    coords = np.asarray(data[:, :4], dtype=np.float64)
    if len(coords) and not np.isfinite(coords).all():
        raise ValueError("lat/lon/altitude/relative_hour must be finite")
    return coords, ids


def _profile_summary(coords: np.ndarray, profile_ids: np.ndarray):
    """Return sorted profile IDs, median representative times, and row order."""
    order = np.argsort(profile_ids, kind="stable")
    sorted_ids = profile_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    unique_ids = sorted_ids[starts].astype(np.int64, copy=False)
    representative_hours = np.empty(len(unique_ids), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        representative_hours[index] = np.median(coords[order[start:end], 3])
    return unique_ids, representative_hours, order, starts, ends


def _time_local_split(profile_ids: np.ndarray, representative_hours: np.ndarray,
                      *, seed: int, block_hours: float = BLOCK_HOURS):
    """Assign profiles using deterministic consecutive five-profile blocks."""
    if len(profile_ids) != len(representative_hours):
        raise ValueError("profile IDs and representative times must have equal length")
    if not np.isfinite(representative_hours).all():
        raise ValueError("representative profile times must be finite")
    if block_hours <= 0 or not np.isfinite(block_hours):
        raise ValueError("block_hours must be finite and positive")

    # Stable sorting makes the tie-breaker explicit and reproducible.
    order = np.lexsort((profile_ids, representative_hours))
    ordered_ids = profile_ids[order]
    ordered_hours = representative_hours[order]
    train_mask = np.zeros(len(profile_ids), dtype=bool)
    development_mask = np.zeros(len(profile_ids), dtype=bool)
    rng = np.random.default_rng(int(seed))
    groups = []
    cursor = 0
    while cursor < len(order):
        remaining = len(order) - cursor
        if remaining < BLOCK_SIZE:
            chosen = order[cursor:]
            train_mask[chosen] = True
            groups.append({
                "kind": "tail",
                "profile_count": int(len(chosen)),
                "span_hours": (float(ordered_hours[-1] - ordered_hours[cursor])
                                if len(chosen) else 0.0),
                "development_profile_id": None,
            })
            break

        block_end = cursor + BLOCK_SIZE
        span = float(ordered_hours[block_end - 1] - ordered_hours[cursor])
        if span <= float(block_hours) + 1e-12:
            block = order[cursor:block_end]
            dev_position = int(rng.integers(BLOCK_SIZE))
            development_mask[block[dev_position]] = True
            train_mask[block[np.arange(BLOCK_SIZE) != dev_position]] = True
            groups.append({
                "kind": "five_profile_block",
                "profile_count": BLOCK_SIZE,
                "span_hours": span,
                "development_profile_id": int(ordered_ids[cursor + dev_position]),
            })
            cursor = block_end
        else:
            train_mask[order[cursor]] = True
            groups.append({
                "kind": "gap_singleton_train",
                "profile_count": 1,
                "span_hours": 0.0,
                "development_profile_id": None,
            })
            cursor += 1

    train = np.sort(profile_ids[train_mask].astype(np.int64, copy=False))
    development = np.sort(profile_ids[development_mask].astype(np.int64, copy=False))
    if len(train) + len(development) != len(profile_ids):
        raise AssertionError("profile split does not cover every profile")
    if np.intersect1d(train, development).size:
        raise AssertionError("train and Development profile sets overlap")
    return train, development, groups


def dipole_sin_i(latitude_deg, longitude_deg) -> np.ndarray:
    """Compute the smooth dipole ``sin(I)`` feature used by the model."""
    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    pole_lat = np.deg2rad(80.5)
    pole_lon = np.deg2rad(-73.0)
    sin_magnetic_lat = (
        np.sin(lat) * np.sin(pole_lat)
        + np.cos(lat) * np.cos(pole_lat) * np.cos(lon - pole_lon))
    return 2.0 * sin_magnetic_lat / np.sqrt(1.0 + 3.0 * sin_magnetic_lat ** 2)


def physical_cell_labels(data, magnetic_sin_i=None) -> np.ndarray:
    """Return 12-cell labels for rows in the 200--500 km audit domain.

    Rows outside the audit domain are returned as an empty string.  The
    boundaries are ``[200,300)``, ``[300,500]``, LT ``[06,18)``, and absolute
    ``sin(I)`` ``<0.25/[0.25,0.5)/>=0.5``.
    """
    coords = np.asarray(data)
    if coords.ndim != 2 or coords.shape[1] < 4:
        raise ValueError("data must have at least four coordinate columns")
    coords = np.asarray(coords[:, :4], dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("physical-cell coordinates must be finite")
    if magnetic_sin_i is None:
        sin_i = dipole_sin_i(coords[:, 0], coords[:, 1])
    else:
        sin_i = _as_1d("magnetic_sin_i", magnetic_sin_i, dtype=np.float64)
        if len(sin_i) != len(coords) or not np.isfinite(sin_i).all():
            raise ValueError("magnetic_sin_i must match data rows and be finite")
    local_time = np.mod(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    day_label = np.where((local_time >= 6.0) & (local_time < 18.0), "day", "night")
    magnitude = np.abs(sin_i)
    magnetic_label = np.select(
        [magnitude < 0.25, magnitude < 0.50], ["low", "mid"], default="high")
    labels = np.full(len(coords), "", dtype="U32")
    for altitude_name, lower, upper, inclusive_upper in _ALTITUDE_BANDS:
        altitude_mask = coords[:, 2] >= lower
        altitude_mask &= (coords[:, 2] <= upper if inclusive_upper
                          else coords[:, 2] < upper)
        labels[altitude_mask] = np.char.add(
            np.char.add(np.char.add(altitude_name, "_"), day_label[altitude_mask]),
            np.char.add("_", magnetic_label[altitude_mask]))
    return labels


def _time_coverage(relative_hours, profile_ids, train_ids, development_ids):
    hours = np.asarray(relative_hours, dtype=np.float64)
    ids = np.asarray(profile_ids, dtype=np.int64)
    train_ids = np.asarray(train_ids, dtype=np.int64)
    development_ids = np.asarray(development_ids, dtype=np.int64)
    train_rows = np.isin(ids, train_ids)
    development_rows = np.isin(ids, development_ids)

    def counts(bucket, mask):
        result = []
        for value in np.unique(bucket):
            selected = mask & (bucket == value)
            result.append({
                "bin": int(value),
                "profile_count": int(np.unique(ids[selected]).size),
                "row_count": int(selected.sum()),
            })
        return result

    return {
        "utc_day": counts(np.floor(hours / 24.0).astype(np.int64),
                           np.ones(len(ids), dtype=bool)),
        "half_hour": counts(np.floor(hours / 0.5).astype(np.int64),
                             np.ones(len(ids), dtype=bool)),
        "utc_day_by_partition": {
            "train": counts(np.floor(hours / 24.0).astype(np.int64), train_rows),
            "development": counts(np.floor(hours / 24.0).astype(np.int64),
                                   development_rows),
        },
        "half_hour_by_partition": {
            "train": counts(np.floor(hours / 0.5).astype(np.int64), train_rows),
            "development": counts(np.floor(hours / 0.5).astype(np.int64),
                                   development_rows),
        },
    }


def _physical_coverage(data, profile_ids, train_ids, development_ids,
                       magnetic_sin_i=None):
    labels = physical_cell_labels(data, magnetic_sin_i)
    train_ids = np.asarray(train_ids, dtype=np.int64)
    development_ids = np.asarray(development_ids, dtype=np.int64)
    records = []
    warnings_list = []
    expected_cells = [
        f"{altitude}_{period}_{magnetic}"
        for altitude, _, _, _ in _ALTITUDE_BANDS
        for period in ("day", "night")
        for magnetic, _ in _MAGNETIC_BANDS
    ]
    for cell in expected_cells:
        cell_rows = labels == cell
        cell_all = np.unique(np.asarray(profile_ids)[cell_rows])
        cell_train = np.intersect1d(cell_all, train_ids, assume_unique=False)
        cell_development = np.intersect1d(
            cell_all, development_ids, assume_unique=False)
        train_count = int(cell_train.size)
        development_count = int(cell_development.size)
        if development_count and not train_count:
            status = "development_only_extrapolation"
            message = f"{cell}: Development profiles have no Train support"
            warnings_list.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        elif train_count and not development_count:
            status = "train_only"
        elif train_count and development_count:
            status = "train_and_development"
        else:
            status = "empty"
        records.append({
            "cell": cell,
            "train_profile_count": train_count,
            "development_profile_count": development_count,
            "all_profile_count": int(cell_all.size),
            "status": status,
        })
    return records, warnings_list


def build_source_split(source: str, data, profile_ids, *, seed: int = 42,
                       block_hours: float = BLOCK_HOURS,
                       magnetic_sin_i=None):
    """Build one source's fixed train/Development profile allowlists."""
    source = str(source).upper()
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    coords, ids = _validate_rows(data, profile_ids)
    if len(ids) == 0:
        raise ValueError(f"{source} contains no valid profiles")
    unique_ids, representative_hours, _, _, _ = _profile_summary(coords, ids)
    train, development, groups = _time_local_split(
        unique_ids, representative_hours, seed=seed, block_hours=block_hours)
    coverage, coverage_warnings = _physical_coverage(
        coords, ids, train, development, magnetic_sin_i=magnetic_sin_i)
    spans = [group["span_hours"] for group in groups
             if group["kind"] == "five_profile_block"]
    audit = {
        "source": source,
        "seed": int(seed),
        "block_size": BLOCK_SIZE,
        "block_hours": float(block_hours),
        "profile_count": int(len(unique_ids)),
        "train_profile_count": int(len(train)),
        "development_profile_count": int(len(development)),
        "development_fraction": float(len(development) / len(unique_ids)),
        "profile_partition_complete": True,
        "profile_partition_disjoint": True,
        "representative_time_hours": {
            "minimum": float(np.min(representative_hours)),
            "maximum": float(np.max(representative_hours)),
        },
        "grouping": {
            "group_count": int(len(groups)),
            "complete_five_profile_group_count": int(sum(
                group["kind"] == "five_profile_block" for group in groups)),
            "gap_singleton_train_count": int(sum(
                group["kind"] == "gap_singleton_train" for group in groups)),
            "tail_profile_count": int(sum(
                group["kind"] == "tail" for group in groups)),
            "maximum_complete_group_span_hours": (
                float(max(spans)) if spans else 0.0),
            "all_complete_group_spans_within_limit": bool(
                all(span <= block_hours + 1e-12 for span in spans)),
        },
        "time_coverage": _time_coverage(coords[:, 3], ids, train, development),
        "physical_cell_coverage": coverage,
        "warnings": coverage_warnings,
    }
    return {
        "train": train,
        "development": development,
        "profile_ids": unique_ids,
        "representative_hours": representative_hours,
        "audit": audit,
    }


def build_profile_split(sources: Mapping[str, object], *, seed: int = 42,
                        block_hours: float = BLOCK_HOURS):
    """Build FY/COSMIC splits from ``{"data": ..., "profile_ids": ...}``.

    A tuple ``(data, profile_ids)`` is accepted as a compact alternative.  An
    optional ``magnetic_sin_i`` row array may be supplied for exact parity with
    a caller's magnetic-coordinate implementation.
    """
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must be a non-empty mapping")
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "seed": int(seed),
        "block_hours": float(block_hours),
        "sources": {},
    }
    for source, payload in sources.items():
        if isinstance(payload, Mapping):
            if "data" not in payload or "profile_ids" not in payload:
                raise ValueError(f"{source} payload needs data and profile_ids")
            data = payload["data"]
            profile_ids = payload["profile_ids"]
            magnetic_sin_i = payload.get("magnetic_sin_i")
        else:
            try:
                data, profile_ids = payload
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source} payload must be a mapping or (data, profile_ids)") from exc
            magnetic_sin_i = None
        result["sources"][str(source).upper()] = build_source_split(
            str(source), data, profile_ids, seed=seed,
            block_hours=block_hours, magnetic_sin_i=magnetic_sin_i)
    result["audit"] = {
        "semantics": SEMANTICS,
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "source_count": len(result["sources"]),
        "sources": {source: payload["audit"]
                    for source, payload in result["sources"].items()},
        "locked_test": {
            "present": False,
            "reason": "A1-S2 has only train and Development profile partitions",
        },
    }
    return result


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_profile_split(path, split, *, audit_path=None):
    """Atomically write ``profile_split.npz`` and its JSON audit."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if split.get("semantics") != SEMANTICS:
        raise ValueError("split semantics mismatch")
    arrays = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
        "semantics": np.asarray([SEMANTICS]),
        "seed": np.asarray([int(split["seed"])], dtype=np.int64),
    }
    for source, payload in split.get("sources", {}).items():
        source = str(source).upper()
        if source not in SOURCES:
            raise ValueError(f"unsupported source {source}")
        train = np.asarray(payload["train"], dtype=np.int64)
        development = np.asarray(payload["development"], dtype=np.int64)
        if (train.ndim != 1 or development.ndim != 1
                or np.any(np.diff(train) <= 0) or np.any(np.diff(development) <= 0)
                or np.intersect1d(train, development).size):
            raise ValueError(f"{source} profile allowlists must be sorted and disjoint")
        arrays[f"{source}_train_profile_ids"] = train
        arrays[f"{source}_development_profile_ids"] = development
    temporary = path.with_name(path.name + ".tmp")
    np.savez_compressed(temporary, **arrays)
    # numpy appends .npz when given a suffix it does not recognise.
    generated = Path(str(temporary) + ".npz") if not temporary.name.endswith(".npz") else temporary
    os.replace(generated, path)
    if audit_path is None:
        audit_path = path.with_name("profile_split_audit.json")
    audit_path = Path(audit_path).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_path.with_name(audit_path.name + ".tmp")
    with temporary_audit.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(split.get("audit", {})), stream,
                  ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary_audit, audit_path)
    return path, audit_path


def load_profile_split(path, *, expected_seed: Optional[int] = None):
    """Load and validate a split; legacy date/locked-test manifests are rejected."""
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        if any("locked" in name.lower() or "test" in name.lower()
               for name in archive.files):
            raise ValueError("A1-S2 profile split must not contain locked-test arrays")
        required = {"schema_version", "semantics", "seed"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"profile split missing arrays: {sorted(missing)}")
        semantics = str(np.asarray(archive["semantics"]).reshape(-1)[0])
        if semantics != SEMANTICS:
            raise ValueError(f"profile split semantics mismatch: {semantics}")
        seed = int(np.asarray(archive["seed"]).reshape(-1)[0])
        if expected_seed is not None and seed != int(expected_seed):
            raise ValueError("profile split seed mismatch")
        sources = {}
        for source in SOURCES:
            train_key = f"{source}_train_profile_ids"
            development_key = f"{source}_development_profile_ids"
            if train_key not in archive.files and development_key not in archive.files:
                continue
            if train_key not in archive.files or development_key not in archive.files:
                raise ValueError(f"incomplete {source} profile split")
            train = np.asarray(archive[train_key], dtype=np.int64)
            development = np.asarray(archive[development_key], dtype=np.int64)
            if (train.ndim != 1 or development.ndim != 1
                    or np.any(np.diff(train) <= 0)
                    or np.any(np.diff(development) <= 0)
                    or np.intersect1d(train, development).size):
                raise ValueError(f"invalid {source} profile allowlists")
            sources[source] = {"train": train, "development": development}
    if not sources:
        raise ValueError("profile split contains no source allowlists")
    return {"schema_version": SCHEMA_VERSION, "semantics": semantics,
            "seed": seed, "sources": sources}


__all__ = [
    "BLOCK_HOURS",
    "BLOCK_SIZE",
    "SEMANTICS",
    "build_profile_split",
    "build_source_split",
    "dipole_sin_i",
    "load_profile_split",
    "physical_cell_labels",
    "save_profile_split",
]
