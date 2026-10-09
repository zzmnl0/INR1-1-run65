"""Train-only QD/AACGM migration audit without M00 inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import importlib
import json
import os

import numpy as np

from inr_modules.mdia.m00_residual_experiment import _migration_audit, validate_a1s_manifest


def _train_profile_ids(index_path: str, train_days: list[int], n_rows: int,
                       start_date: str = "2024-09-01") -> np.ndarray:
    with np.load(index_path, allow_pickle=False) as index:
        required = ("profile_id", "date_code", "output_start", "output_end")
        missing = [key for key in required if key not in index.files]
        if missing:
            raise ValueError(f"profile index missing {missing}")
        profile_ids = np.asarray(index["profile_id"], dtype=np.int64).reshape(-1)
        date_codes = np.asarray(index["date_code"], dtype=np.int64).reshape(-1)
        starts = np.asarray(index["output_start"], dtype=np.int64).reshape(-1)
        ends = np.asarray(index["output_end"], dtype=np.int64).reshape(-1)
    if not (profile_ids.size == date_codes.size == starts.size == ends.size):
        raise ValueError("profile index arrays are not aligned")
    base = datetime.fromisoformat(str(start_date).replace("Z", "").split()[0])
    train_date_codes = np.asarray([
        int((base + timedelta(days=int(day))).strftime("%Y%m%d"))
        for day in train_days
    ], dtype=np.int64)
    selected = np.isin(date_codes, train_date_codes)
    valid_bounds = ((starts >= 0) & (ends >= starts) & (ends <= n_rows))
    selected &= valid_bounds
    train_ids = np.sort(profile_ids[selected])
    if train_ids.size == 0 or np.unique(train_ids).size != train_ids.size:
        raise ValueError("Train profile IDs are empty or non-unique")
    return train_ids


def audit(source: str, data_path: str, index_path: str, sidecar_path: str,
          train_days: list[int], start_date: str) -> dict:
    raw = np.load(data_path, mmap_mode="r")
    allowed_profiles = _train_profile_ids(
        index_path, train_days, len(raw), start_date=start_date)
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        required = ("row_index", "profile_id", "qd_latitude_deg", "aacgm_latitude_deg")
        missing = [key for key in required if key not in sidecar.files]
        if missing:
            raise ValueError(f"{source} sidecar missing {missing}")
        rows = np.asarray(sidecar["row_index"], dtype=np.int64).reshape(-1)
        profile_ids = np.asarray(sidecar["profile_id"], dtype=np.int64).reshape(-1)
        qd = np.asarray(sidecar["qd_latitude_deg"], dtype=np.float64).reshape(-1)
        aacgm = np.asarray(sidecar["aacgm_latitude_deg"], dtype=np.float64).reshape(-1)
    if rows.size == 0 or np.any(rows < 0) or np.any(rows >= len(raw)):
        raise ValueError(f"{source} sidecar row_index is invalid")
    if not (rows.size == profile_ids.size == qd.size == aacgm.size):
        raise ValueError(f"{source} sidecar arrays are not aligned")
    keep = np.isin(profile_ids, allowed_profiles)
    if not np.any(keep):
        raise ValueError(f"{source} sidecar has no authorized Train rows")
    rows, profile_ids, qd, aacgm = (value[keep] for value in (rows, profile_ids, qd, aacgm))
    coords = np.asarray(raw[rows, :4], dtype=np.float64)
    report = _migration_audit(
        qd, aacgm, profile_ids, np.full(rows.size, source, dtype=object), coords)
    report.update({
        "source": source,
        "sidecar_rows": int(rows.size),
        "train_only": True,
        "authorized_train_profiles": int(allowed_profiles.size),
        "sidecar_train_profiles": int(np.unique(profile_ids).size),
        "qd_accepted_by_user": True,
    })
    if not np.array_equal(np.sort(np.unique(profile_ids)), allowed_profiles):
        raise ValueError(f"{source} sidecar does not cover exactly the Train profile set")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--date-split-manifest", required=True)
    parser.add_argument("--fy-data", required=True)
    parser.add_argument("--fy-index", required=True)
    parser.add_argument("--fy-sidecar", required=True)
    parser.add_argument("--cosmic-data", required=True)
    parser.add_argument("--cosmic-index", required=True)
    parser.add_argument("--cosmic-sidecar", required=True)
    args = parser.parse_args()
    train_days = validate_a1s_manifest(args.date_split_manifest)["train"]
    with open(args.date_split_manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    start_date = manifest.get("start_date_str", "2024-09-01 00:00:00")
    reports = [
        audit("FY", args.fy_data, args.fy_index, args.fy_sidecar, train_days, start_date),
        audit("COSMIC", args.cosmic_data, args.cosmic_index, args.cosmic_sidecar, train_days, start_date),
    ]
    def _version(name: str) -> str | None:
        try:
            return str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except ImportError:
            return None
    payload = {
        "semantics": "a1s_qd_aacgm_migration_review_v3",
        "train_only": True,
        "locked_test": "excluded",
        "sidecar_coordinate_semantics": "qd_abs_sin_lat_025_050_v1",
        "qd_accepted_by_user": True,
        "review_stop": False,
        "converter_provenance": {
            "qd_converter": "apexpy",
            "apexpy_version_runtime": _version("apexpy"),
            "aacgm_version_runtime": _version("aacgmv2"),
        },
        "reports": reports,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
