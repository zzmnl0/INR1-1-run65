"""Generate a resumable, row-aligned QD coordinate sidecar.

QD is calibration metadata only. The generator accepts an explicit authorized
row selector (or an A1-S date manifest) and never silently expands it to the
full source array; this keeps Locked-test rows out of the conversion itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import importlib
import json
import os
from typing import Iterable

import numpy as np

from generate_aacgm_sidecar import _profile_ids
from inr_modules.data_managers.profile_split import dipole_sin_i


QD_SEMANTICS = "qd_abs_sin_lat_025_050_v1"


def _apex_version(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _convert_qd(module: object, lat: np.ndarray, lon: np.ndarray,
                altitude: np.ndarray, when: datetime) -> tuple[np.ndarray, np.ndarray]:
    Apex = getattr(module, "Apex", None)
    if Apex is None:
        raise RuntimeError("apexpy.Apex is unavailable")
    try:
        converter = Apex(date=when)
        qlat, qlon = converter.convert(
            lat, lon, "geo", "qd", height=altitude, datetime=when)
    except Exception as exc:  # pragma: no cover - compiled dependency
        raise RuntimeError(f"apexpy QD conversion failed at {when.isoformat()}") from exc
    return (np.asarray(qlat, dtype=np.float64).reshape(-1),
            np.asarray(qlon, dtype=np.float64).reshape(-1))


def _aacgm_audit(lat: np.ndarray, lon: np.ndarray, altitude: np.ndarray,
                 when: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full(lat.size, np.nan, dtype=np.float32)
    valid = np.zeros(lat.size, dtype=bool)
    reasons = np.full(lat.size, "aacgm_unavailable", dtype="U64")
    try:
        module = importlib.import_module("aacgmv2")
        result = module.get_aacgm_coord_arr(lat, lon, altitude, when)
        candidate = np.asarray(result[0], dtype=np.float64).reshape(-1)
        if candidate.size != lat.size:
            raise ValueError("AACGM output length mismatch")
        finite = np.isfinite(candidate) & (np.abs(candidate) <= 90.0)
        values[finite] = candidate[finite].astype(np.float32)
        valid[finite] = True
        reasons[finite] = ""
        reasons[~finite] = "aacgm_nonfinite_or_out_of_range"
    except Exception as exc:  # AACGM is audit-only in the QD contract.
        reasons[:] = f"aacgm_error:{type(exc).__name__}"[:64]
    return values, valid, reasons


def _aacgm_version() -> str | None:
    """Return installed AACGM provenance without inventing a fallback."""
    try:
        module = importlib.import_module("aacgmv2")
    except ImportError:
        return None
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _load_row_index(path: str, n_rows: int) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    try:
        if isinstance(loaded, np.ndarray):
            values = loaded
        else:
            key = "row_index" if "row_index" in loaded.files else loaded.files[0]
            values = loaded[key]
    finally:
        if hasattr(loaded, "close"):
            loaded.close()
    rows = np.asarray(values, dtype=np.int64).reshape(-1)
    if rows.size == 0 or np.any(rows < 0) or np.any(rows >= n_rows):
        raise ValueError("authorized row selector is empty or out of range")
    if np.unique(rows).size != rows.size:
        raise ValueError("authorized row selector contains duplicates")
    return np.sort(rows)


def _manifest_days(path: str, partitions: Iterable[str]) -> set[int]:
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    raw = manifest.get("partitions", manifest)
    days: set[int] = set()
    for partition in partitions:
        values = raw.get(partition)
        if values is None:
            raise ValueError(f"date manifest lacks partition {partition}")
        days.update(int(value) for value in values)
    return days


def _authorized_rows(data: np.ndarray, index: dict[str, np.ndarray], *,
                     row_index_path: str | None, date_split_manifest: str | None,
                     authorized_partitions: tuple[str, ...], base: datetime) -> np.ndarray:
    n_rows = int(data.shape[0])
    if row_index_path:
        return _load_row_index(row_index_path, n_rows)
    if date_split_manifest:
        days = _manifest_days(date_split_manifest, authorized_partitions)
        if {"date_code", "output_start", "output_end"}.issubset(index):
            wanted = {
                int((base.date() + timedelta(days=int(day))).strftime("%Y%m%d"))
                for day in days
            }
            selected_profiles = np.flatnonzero(
                np.isin(np.asarray(index["date_code"], dtype=np.int64), list(wanted)))
            rows = []
            starts = np.asarray(index["output_start"], dtype=np.int64)
            ends = np.asarray(index["output_end"], dtype=np.int64)
            for profile_index in selected_profiles:
                lo, hi = int(starts[profile_index]), int(ends[profile_index])
                if lo >= 0 and hi > lo:
                    rows.append(np.arange(lo, hi, dtype=np.int64))
            if rows:
                return np.unique(np.concatenate(rows))
            raise ValueError("date manifest selected no indexed source rows")
        utc_hours = np.asarray(data[:, 3], dtype=np.float64)
        if not np.isfinite(utc_hours).all():
            raise ValueError("UTC hours contain non-finite values")
        row_days = np.floor(utc_hours / 24.0).astype(np.int64)
        rows = np.flatnonzero(np.isin(row_days, np.fromiter(days, dtype=np.int64)))
        if rows.size == 0:
            raise ValueError("date manifest selected no source rows")
        return rows.astype(np.int64)
    return np.arange(n_rows, dtype=np.int64)


def generate_qd_sidecar(
    data_path: str,
    index_path: str,
    output_path: str,
    *,
    source: str,
    reference_time_utc: str = "2024-09-01T00:00:00",
    batch_rows: int = 100_000,
    chunk_dir: str | None = None,
    include_aacgm_audit: bool = True,
    row_index_path: str | None = None,
    date_split_manifest: str | None = None,
    authorized_partitions: tuple[str, ...] = ("train", "development"),
) -> dict[str, object]:
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    try:
        apex_module = importlib.import_module("apexpy")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("strict QD sidecars require apexpy; no fallback allowed") from exc

    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("QC data must contain [lat, lon, altitude, utc_hours]")
    with np.load(index_path, allow_pickle=True) as index_payload:
        profile_ids_full = _profile_ids(index_payload, int(data.shape[0]))
        index = {key: np.asarray(index_payload[key]) for key in index_payload.files}
    base = datetime.fromisoformat(reference_time_utc.replace("Z", "+00:00")).replace(tzinfo=None)
    authorized = _authorized_rows(
        data, index,
        row_index_path=row_index_path,
        date_split_manifest=date_split_manifest,
        authorized_partitions=authorized_partitions,
        base=base,
    )
    full_rows = int(data.shape[0])
    n = int(authorized.size)
    lat = np.asarray(data[authorized, 0], dtype=np.float64)
    lon = np.asarray(data[authorized, 1], dtype=np.float64)
    altitude = np.asarray(data[authorized, 2], dtype=np.float64)
    utc_hours = np.asarray(data[authorized, 3], dtype=np.float64)
    coords = np.column_stack((lat, lon, altitude, utc_hours))
    if not np.isfinite(coords).all():
        raise ValueError("authorized QD coordinates contain non-finite values")
    if np.any((lat < -90.0) | (lat > 90.0) | (lon < -360.0) | (lon > 360.0)):
        raise ValueError("authorized QD coordinates out of range")
    profile_ids = profile_ids_full[authorized]
    model_sin_i = np.asarray(dipole_sin_i(lat, lon), dtype=np.float32)
    if not np.isfinite(model_sin_i).all() or np.any(np.abs(model_sin_i) > 1.0):
        raise ValueError("model dipole sin(I) is non-finite or outside [-1,1]")

    qd_lat = np.full(n, np.nan, dtype=np.float32)
    qd_lon = np.full(n, np.nan, dtype=np.float32)
    aacgm_lat = np.full(n, np.nan, dtype=np.float32)
    aacgm_valid = np.zeros(n, dtype=bool)
    aacgm_reason = np.full(n, "aacgm_unavailable", dtype="U64")
    chunk_root = os.path.abspath(chunk_dir or (output_path + ".chunks"))
    os.makedirs(chunk_root, exist_ok=True)

    for lo in range(0, n, batch_rows):
        hi = min(n, lo + batch_rows)
        chunk_path = os.path.join(chunk_root, f"rows_{lo:012d}_{hi:012d}.npz")
        if os.path.isfile(chunk_path):
            with np.load(chunk_path, allow_pickle=False) as cached:
                required = {"row_index", "qd_latitude_deg", "qd_longitude_deg"}
                if not required.issubset(cached.files):
                    raise ValueError(f"QD chunk missing identity arrays: {chunk_path}")
                if not np.array_equal(np.asarray(cached["row_index"], dtype=np.int64), authorized[lo:hi]):
                    raise ValueError(f"QD chunk row identity mismatch: {chunk_path}")
                qd_lat[lo:hi] = cached["qd_latitude_deg"]
                qd_lon[lo:hi] = cached["qd_longitude_deg"]
                if include_aacgm_audit:
                    for key in ("aacgm_latitude_deg", "aacgm_valid_mask", "aacgm_failure_reason"):
                        if key not in cached.files:
                            raise ValueError(f"QD chunk missing {key}: {chunk_path}")
                    aacgm_lat[lo:hi] = cached["aacgm_latitude_deg"]
                    aacgm_valid[lo:hi] = cached["aacgm_valid_mask"]
                    aacgm_reason[lo:hi] = cached["aacgm_failure_reason"].astype("U64")
            continue
        chunk_times = utc_hours[lo:hi]
        for stamp in np.unique(chunk_times):
            local_rows = np.flatnonzero(chunk_times == stamp) + lo
            when = base + timedelta(hours=float(stamp))
            converted = _convert_qd(apex_module, lat[local_rows], lon[local_rows], altitude[local_rows], when)
            if converted[0].size != local_rows.size or converted[1].size != local_rows.size:
                raise ValueError("QD conversion output length mismatch")
            if (not np.isfinite(converted[0]).all()
                    or not np.isfinite(converted[1]).all()
                    or np.any(np.abs(converted[0]) > 90.0)):
                raise ValueError(f"QD conversion produced invalid values at {when.isoformat()}")
            qd_lat[local_rows] = converted[0].astype(np.float32)
            qd_lon[local_rows] = converted[1].astype(np.float32)
            if include_aacgm_audit:
                a_lat, a_valid, a_reason = _aacgm_audit(
                    lat[local_rows], lon[local_rows], altitude[local_rows], when)
                aacgm_lat[local_rows] = a_lat
                aacgm_valid[local_rows] = a_valid
                aacgm_reason[local_rows] = a_reason
        if not np.isfinite(qd_lat[lo:hi]).all():
            raise ValueError(f"QD chunk contains non-finite output: {chunk_path}")
        chunk_payload = {
            "row_index": authorized[lo:hi],
            "qd_latitude_deg": qd_lat[lo:hi],
            "qd_longitude_deg": qd_lon[lo:hi],
        }
        if include_aacgm_audit:
            chunk_payload.update({
                "aacgm_latitude_deg": aacgm_lat[lo:hi],
                "aacgm_valid_mask": aacgm_valid[lo:hi],
                "aacgm_failure_reason": aacgm_reason[lo:hi],
            })
        tmp_chunk = chunk_path + ".tmp.npz"
        np.savez_compressed(tmp_chunk, **chunk_payload)
        os.replace(tmp_chunk, chunk_path)

    metadata = {
        "source": str(source),
        "semantics": QD_SEMANTICS,
        "qd_converter": "apexpy",
        "apexpy_version": _apex_version(apex_module),
        "igrf_semantics": "IGRF14_coefficients_from_apexpy_2.1.1",
        "reference_time_utc": base.isoformat() + "Z",
        "coordinate_input_semantics": "geodetic_lat_lon_actual_height_km_v1",
        "continuous_time_semantics": "relative_utc_hours_v1",
        "fallback": False,
        "authorized_row_selector": "explicit_row_index" if row_index_path else "date_manifest",
        "authorized_partitions": list(authorized_partitions),
        "full_input_rows": full_rows,
        "authorized_rows": n,
        "aacgm_audit_available": bool(include_aacgm_audit),
        "aacgm_version": _aacgm_version() if include_aacgm_audit else None,
        "chunk_dir": chunk_root,
        "chunk_rows": int(batch_rows),
        "chunk_count": int((n + batch_rows - 1) // batch_rows),
    }
    payload = {
        "row_index": authorized,
        "profile_id": profile_ids,
        "qd_latitude_deg": qd_lat,
        "qd_longitude_deg": qd_lon,
        "magnetic_sin_i": model_sin_i,
        "latitude_deg": lat.astype(np.float32),
        "longitude_deg": lon.astype(np.float32),
        "altitude_km": altitude.astype(np.float32),
        "utc_hours": utc_hours.astype(np.float32),
        "semantics": np.asarray(QD_SEMANTICS),
        "apexpy_version": np.asarray(metadata["apexpy_version"]),
        "reference_time_utc": np.asarray(metadata["reference_time_utc"]),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if include_aacgm_audit:
        payload.update({
            "aacgm_latitude_deg": aacgm_lat,
            "aacgm_valid_mask": aacgm_valid,
            "aacgm_failure_reason": aacgm_reason,
        })
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
    parser.add_argument("--chunk-dir", default=None)
    parser.add_argument("--row-index-file", default=None)
    parser.add_argument("--date-split-manifest", default=None)
    parser.add_argument("--authorized-partitions", default="train,development")
    parser.add_argument("--no-aacgm-audit", action="store_true")
    args = parser.parse_args()
    partitions = tuple(part.strip() for part in args.authorized_partitions.split(",") if part.strip())
    print(json.dumps(generate_qd_sidecar(
        args.data, args.index, args.output,
        source=args.source,
        reference_time_utc=args.reference_time_utc,
        batch_rows=args.batch_rows,
        chunk_dir=args.chunk_dir,
        include_aacgm_audit=not args.no_aacgm_audit,
        row_index_path=args.row_index_file,
        date_split_manifest=args.date_split_manifest,
        authorized_partitions=partitions,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
