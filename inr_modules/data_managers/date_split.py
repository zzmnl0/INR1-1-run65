"""A1-S6 date-split manifest contract.

The A1-S manifest is the source of truth for UTC-date partitions.  A1-S6
uses the existing Development dates unchanged and moves every Locked-test date
to Train.  This module intentionally has no model/data-loader
dependencies so callers can apply the contract before constructing a run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, MutableMapping, Optional


SCHEMA_VERSION = 1
CONTRACT = "A1-S6"
SEMANTICS = "a1s_train_plus_locked_dev5_v1"
PARTITIONS = ("train", "development", "locked_test")


def _as_days(value, name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of UTC day indices")
    try:
        days = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer UTC day indices") from exc
    if any(day < 0 for day in days) or len(set(days)) != len(days):
        raise ValueError(f"{name} must contain unique non-negative day indices")
    return sorted(days)


def _partitions(manifest: Mapping) -> dict[str, list[int]]:
    raw = manifest.get("partitions", manifest)
    if not isinstance(raw, Mapping):
        raise ValueError("date manifest partitions must be an object")
    result = {name: _as_days(raw.get(name, []), name) for name in PARTITIONS}
    combined = [day for days in result.values() for day in days]
    if len(set(combined)) != len(combined):
        raise ValueError("date manifest partitions must be disjoint")
    if "total_days" in manifest:
        try:
            total_days = int(manifest["total_days"])
        except (TypeError, ValueError) as exc:
            raise ValueError("total_days must be an integer") from exc
        if total_days < 0 or set(combined) != set(range(total_days)):
            raise ValueError("date manifest partitions must cover every UTC day")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_a1s_manifest(path: os.PathLike | str) -> dict:
    """Load and validate an A1-S date manifest without modifying it."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, Mapping):
        raise ValueError("date manifest must be a JSON object")
    result = copy.deepcopy(dict(manifest))
    result["partitions"] = _partitions(result)
    return result


def apply_a1s6_contract(manifest: Mapping, *, source_sha256: Optional[str] = None) -> dict:
    """Return an A1-S6 view: former Locked-test dates join Train.

    The input is never mutated.  Development retains the exact set of day
    indices from the source manifest; effective Locked-test is empty.
    """
    if isinstance(manifest, (str, os.PathLike)):
        return load_a1s6_date_manifest(manifest)
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    result: MutableMapping = copy.deepcopy(dict(manifest))
    original = _partitions(result)
    effective = {
        "train": sorted(set(original["train"]) | set(original["locked_test"])),
        "development": list(original["development"]),
        "locked_test": [],
    }
    result["partitions"] = effective
    for name in PARTITIONS:
        result[name] = list(effective[name])
    if "development_days" in result:
        result["a1s6_original_development_days"] = result["development_days"]
        result["development_days"] = len(effective["development"])
    if "locked_test_days" in result:
        result["a1s6_original_locked_test_days"] = result["locked_test_days"]
        result["locked_test_days"] = len(effective["locked_test"])
    if "utc_dates" in result and isinstance(result["utc_dates"], Mapping):
        dates = result["utc_dates"]
        result["utc_dates"] = {
            name: list(dates.get(name, []))
            for name in PARTITIONS
        }
        result["utc_dates"]["train"] = sorted(
            set(dates.get("locked_test", [])) | set(dates.get("train", []))
        )
        result["utc_dates"]["locked_test"] = []
    elif result.get("start_date_str"):
        try:
            start = datetime.strptime(str(result["start_date_str"]), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            start = None
        if start is not None:
            result["utc_dates"] = {
                name: [(start + timedelta(days=day)).strftime("%Y-%m-%d") for day in days]
                for name, days in effective.items()
            }
    original_counts = {name: len(days) for name, days in original.items()}
    effective_counts = {name: len(days) for name, days in effective.items()}
    audit = {
        "contract": CONTRACT,
        "semantics": SEMANTICS,
        "schema_version": SCHEMA_VERSION,
        "locked_test_merged_into_train": True,
        "train_merged_into_locked_test": False,
        "development_preserved": effective["development"] == original["development"],
        "original_partition_counts": original_counts,
        "effective_partition_counts": effective_counts,
        "original_partitions": original,
        "effective_partitions": effective,
    }
    if source_sha256 is not None:
        audit["source_manifest_sha256"] = str(source_sha256)
    result["a1s6_audit"] = audit
    # Keep a conventional top-level audit entry for lightweight consumers;
    # retain any source audit fields when an A1-S manifest already has one.
    prior_audit = result.get("audit")
    result["audit"] = {
        **(dict(prior_audit) if isinstance(prior_audit, Mapping) else {}),
        **audit,
    }
    return dict(result)


def load_a1s6_date_manifest(path: os.PathLike | str) -> dict:
    """Load an A1-S manifest and immediately apply the A1-S6 contract."""
    source = Path(path).resolve()
    return apply_a1s6_contract(load_a1s_manifest(source), source_sha256=_sha256(source))


def write_a1s6_date_manifest(
    source: os.PathLike | str | Mapping,
    destination: os.PathLike | str,
) -> Path:
    """Write a transformed manifest atomically and return its resolved path."""
    if isinstance(source, Mapping):
        transformed = apply_a1s6_contract(source)
    else:
        transformed = load_a1s6_date_manifest(source)
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(transformed, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, target)
    return target


# Short aliases keep call sites readable while preserving one implementation.
merge_locked_test_into_train = apply_a1s6_contract
# Backward-compatible alias retained for callers from the initial A1-S6 draft.
merge_train_into_locked_test = apply_a1s6_contract
build_a1s6_manifest = apply_a1s6_contract
load_date_split_manifest = load_a1s6_date_manifest
load_a1s6_manifest = load_a1s6_date_manifest
transform_date_split_manifest = apply_a1s6_contract


if __name__ == "__main__":
    sample = {"partitions": {"train": [0, 2], "development": [1], "locked_test": [3]}}
    checked = apply_a1s6_contract(sample)
    assert checked["partitions"] == {"train": [0, 2, 3], "development": [1], "locked_test": []}
    assert sample["partitions"]["train"] == [0, 2]
    assert checked["a1s6_audit"]["development_preserved"]
    print("A1-S6 date-split self-check: OK")


__all__ = [
    "CONTRACT",
    "PARTITIONS",
    "SCHEMA_VERSION",
    "SEMANTICS",
    "apply_a1s6_contract",
    "build_a1s6_manifest",
    "load_a1s6_date_manifest",
    "load_a1s6_manifest",
    "load_date_split_manifest",
    "load_a1s_manifest",
    "merge_locked_test_into_train",
    "merge_train_into_locked_test",
    "transform_date_split_manifest",
    "write_a1s6_date_manifest",
]
