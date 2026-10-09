"""Generate a strict row-aligned AACGM sidecar for a QC profile dataset.

The QC ``.npy`` file remains untouched.  The sidecar is keyed by the exact
row index and profile id, so calibration code can reject stale or re-ordered
coordinate products instead of silently falling back to geographic latitude.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os

import numpy as np


def _profile_ids(index: np.lib.npyio.NpzFile, n_rows: int) -> np.ndarray:
    required = ("profile_id", "output_start", "output_end")
    missing = [name for name in required if name not in index.files]
    if missing:
        raise ValueError(f"profile index missing {missing}")
    pid = np.asarray(index["profile_id"], dtype=np.int64).reshape(-1)
    start = np.asarray(index["output_start"], dtype=np.int64).reshape(-1)
    end = np.asarray(index["output_end"], dtype=np.int64).reshape(-1)
    if not (pid.size == start.size == end.size):
        raise ValueError("profile index arrays have different lengths")
    partial = (start < 0) != (end < 0)
    if np.any(partial) or np.any(end > n_rows) or np.any((start >= 0) & (end < start)):
        raise ValueError("profile index ranges invalid")
    rows = np.full(n_rows, -1, dtype=np.int64)
    for profile, lo, hi in zip(pid, start, end):
        # QC indices retain rejected profiles as (-1, -1); they have no rows.
        if lo < 0 and hi < 0:
            continue
        if np.any(rows[lo:hi] != -1):
            raise ValueError("profile index ranges overlap")
        rows[lo:hi] = profile
    if np.any(rows < 0):
        raise ValueError("profile index does not cover every QC row")
    return rows


def generate_sidecar(
    data_path: str,
    index_path: str,
    output_path: str,
    *,
    reference_time_utc: str = "2024-09-01T00:00:00",
    source: str,
    batch_rows: int = 100_000,
) -> dict[str, object]:
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    try:
        import aacgmv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("strict AACGM sidecars require aacgmv2") from exc
    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("QC data must be a two-dimensional array with coordinates in the first four columns")
    with np.load(index_path, allow_pickle=True) as index:
        profile_ids = _profile_ids(index, int(data.shape[0]))
    base = datetime.fromisoformat(reference_time_utc.replace("Z", "+00:00")).replace(tzinfo=None)
    n = int(data.shape[0])
    row_index = np.arange(n, dtype=np.int64)
    lat = np.asarray(data[:, 0], dtype=np.float64)
    lon = np.asarray(data[:, 1], dtype=np.float64)
    altitude = np.asarray(data[:, 2], dtype=np.float64)
    utc_hours = np.asarray(data[:, 3], dtype=np.float64)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all() and np.isfinite(altitude).all() and np.isfinite(utc_hours).all()):
        raise ValueError("QC coordinates contain non-finite values")
    aacgm_lat = np.empty(n, dtype=np.float32)
    aacgm_lon = np.empty(n, dtype=np.float32)
    # AACGM accepts vector coordinates but one UTC timestamp per call. The
    # QC rows share exact float32 UTC bins; grouping equal timestamps removes
    # the per-row Python call without changing the continuous-time contract.
    for lo in range(0, n, batch_rows):
        hi = min(n, lo + batch_rows)
        chunk_times = utc_hours[lo:hi]
        for stamp in np.unique(chunk_times):
            rows = np.flatnonzero(chunk_times == stamp) + lo
            when = base + timedelta(hours=float(np.mean(utc_hours[rows])))
            converted = aacgmv2.get_aacgm_coord_arr(
                lat[rows], lon[rows], altitude[rows], when
            )
            aacgm_lat[rows] = np.asarray(converted[0], dtype=np.float32)
            aacgm_lon[rows] = np.asarray(converted[1], dtype=np.float32)
    if not (np.isfinite(aacgm_lat).all() and np.isfinite(aacgm_lon).all()):
        raise ValueError("AACGM conversion produced non-finite values")
    metadata = {
        "source": str(source),
        "semantics": "aacgm_v2",
        "aacgm_version": f"aacgmv2-{getattr(aacgmv2, '__version__', 'unknown')}",
        "reference_time_utc": base.isoformat() + "Z",
        "actual_height_semantics": "row_altitude_km_v1",
        "continuous_time_semantics": "relative_utc_hours_v1",
        "fallback": False,
    }
    payload = {
        "row_index": row_index,
        "profile_id": profile_ids,
        "aacgm_latitude_deg": aacgm_lat,
        "aacgm_longitude_deg": aacgm_lon,
        "latitude_deg": lat.astype(np.float32),
        "longitude_deg": lon.astype(np.float32),
        "altitude_km": altitude.astype(np.float32),
        "utc_hours": utc_hours.astype(np.float32),
        "semantics": np.asarray("aacgm_v2"),
        "aacgm_version": np.asarray(metadata["aacgm_version"]),
        "reference_time_utc": np.asarray(metadata["reference_time_utc"]),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp = output_path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, output_path)
    return {"rows": n, **metadata, "output": os.path.abspath(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True, choices=("FY", "COSMIC"))
    parser.add_argument("--reference-time-utc", default="2024-09-01T00:00:00")
    parser.add_argument("--batch-rows", type=int, default=100_000)
    args = parser.parse_args()
    print(json.dumps(generate_sidecar(
        data_path=args.data,
        index_path=args.index,
        output_path=args.output,
        source=args.source,
        reference_time_utc=args.reference_time_utc,
        batch_rows=args.batch_rows,
    ), indent=2))


if __name__ == "__main__":
    main()
