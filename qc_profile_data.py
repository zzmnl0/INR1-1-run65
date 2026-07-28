"""Build auditable FY/COSMIC profile-QC products without altering observations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parent
START_UTC = datetime(2024, 9, 1, tzinfo=timezone.utc)
FY_DIRS = tuple(
    Path(rf"D:\FYsatellite\EDP_data\202409txt\FY3{mission}")
    for mission in "DEFG"
)
COSMIC_INPUT = Path(
    r"D:\cosmic2\cosmic245-274-September\cosmic_september_2024.npy"
)
FY_OUTPUT = Path(r"D:\FYsatellite\EDP_data\fy_202409_qc_v2.npy")
COSMIC_OUTPUT = COSMIC_INPUT.with_name("cosmic_september_2024_qc.npy")

SOURCE_CODES = {"FY3D": 1, "FY3E": 2, "FY3F": 3, "FY3G": 4, "COSMIC": 10}
FY_SOURCE_CODES = frozenset(SOURCE_CODES[source] for source in ("FY3D", "FY3E", "FY3F", "FY3G"))
FY_DENSITY_MIN_M3 = 5e6
FY_DENSITY_MAX_M3 = 1e13
REGIME_CODES = {"unknown": 0, "day": 1, "twilight": 2, "night": 3}
REASON_BITS = {
    "read_or_time": 1 << 0,
    "points": 1 << 1,
    "altitude_span": 1 << 2,
    "peak_count": 1 << 3,
    "hmf2": 1 << 4,
    "nmf2": 1 << 5,
    "md": 1 << 6,
    "delta": 1 << 7,
    "global_gradient": 1 << 8,
    "local_gradient": 1 << 9,
    "fold_error": 1 << 10,
}
_FY_NAME = re.compile(r"^(\d{8})_(\d{4})(?:_\d+)?\.txt$", re.IGNORECASE)


@dataclass(slots=True)
class QCResult:
    profile_id: int
    pass_profile: bool
    input_points: int
    kept_points: int
    h_cut_km: float
    hmf2: float
    nmf2: float
    peak_count: int
    md: float
    delta: float
    global_topside_gradient: float
    local_topside_gradient: float
    fold_error: float
    reason_bits: int
    representative_lat: float
    representative_lon: float
    representative_time: float
    sza: float
    regime_code: int
    source_code: int
    date_code: int
    range_rejected_points: int = 0
    output_start: int = -1
    output_end: int = -1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_fy_time(path: Path) -> tuple[datetime, float]:
    match = _FY_NAME.match(path.name)
    if not match:
        raise ValueError(f"unsupported FY filename: {path.name}")
    observed = datetime.strptime(
        "".join(match.groups()), "%Y%m%d%H%M"
    ).replace(tzinfo=timezone.utc)
    return observed, (observed - START_UTC).total_seconds() / 3600.0


def _read_fy(content: bytes) -> np.ndarray:
    if not any(
            line.strip() and not line.lstrip().startswith(b"#")
            for line in content.splitlines()):
        raise ValueError("FY profile contains no data rows")
    raw = np.loadtxt(io.StringIO(content.decode("utf-8")), comments="#")
    raw = np.atleast_2d(raw)
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError("FY profile must contain [lon, lat, alt, Ne]")
    return np.asarray(raw[:, :4], dtype=np.float64)


def _solar_zenith(relative_hour: float, lon: float, lat: float) -> float:
    observed = START_UTC + timedelta(hours=float(relative_hour))
    doy = observed.timetuple().tm_yday
    gamma = 2.0 * np.pi * (doy - 1 + (observed.hour - 12) / 24.0) / 366.0
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )
    local_time = np.remainder(relative_hour + lon / 15.0, 24.0)
    hour_angle = np.radians(15.0 * (local_time - 12.0))
    cos_sza = (
        np.sin(np.radians(lat)) * np.sin(declination)
        + np.cos(np.radians(lat)) * np.cos(declination) * np.cos(hour_angle)
    )
    return float(np.degrees(np.arccos(np.clip(cos_sza, -1.0, 1.0))))


def _regime(sza: float) -> str:
    if not np.isfinite(sza):
        return "unknown"
    if sza < 80.0:
        return "day"
    if sza < 100.0:
        return "twilight"
    return "night"


def _odd_window(points: int, maximum: int, minimum: int = 3) -> int:
    window = min(points if points % 2 else points + 1, maximum)
    if window % 2 == 0:
        window -= 1
    return max(minimum, window)


def _collapse_altitudes(altitude: np.ndarray, density: np.ndarray):
    order = np.argsort(altitude, kind="stable")
    altitude = altitude[order]
    density = density[order]
    unique, starts, counts = np.unique(
        altitude, return_index=True, return_counts=True
    )
    averaged = np.add.reduceat(density, starts) / counts
    return unique, averaged


def _gradient(altitude: np.ndarray, density: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(altitude[mask] * 1000.0, density[mask], 1)[0])


def _bottom_cut(
    grid: np.ndarray,
    median30: np.ndarray,
    hmf2: float,
    nmf2: float,
    regime: str,
) -> float:
    h_cut = 120.0
    bottom = (grid >= 120.0) & (grid < hmf2 - 30.0)
    if np.count_nonzero(bottom) < 5:
        return h_cut
    alt = grid[bottom]
    density = median30[bottom]
    distance = max(1, int(round(30.0 / 2.5)))
    if regime == "day":
        peaks, _ = find_peaks(
            density, prominence=0.03 * nmf2, distance=distance
        )
        ceiling = alt[peaks[-1]] - 20.0 if len(peaks) else hmf2 - 30.0
        subset = alt < ceiling
        troughs, _ = find_peaks(
            -density[subset], prominence=(0.02 if len(peaks) else 0.05) * nmf2,
            distance=distance,
        )
        if len(troughs):
            h_cut = max(h_cut, float(alt[subset][troughs[-1]]))
        return h_cut

    troughs, _ = find_peaks(
        -density,
        prominence=(0.03 if regime == "twilight" else 0.02) * nmf2,
        distance=distance,
    )
    if len(troughs):
        h_cut = max(h_cut, float(alt[troughs[-1]]))

    differences = np.diff(density)
    run_start = None
    for index, negative in enumerate(differences < 0):
        run_start = index if negative and run_start is None else run_start
        if run_start is not None and (
            not negative or index == len(differences) - 1
        ):
            run_end = index + int(negative)
            if alt[run_end] - alt[run_start] >= 8.0:
                h_cut = max(h_cut, float(alt[run_end]))
            run_start = None
    return h_cut


def _empty_result(profile_id: int, source_code: int, reason: int) -> QCResult:
    return QCResult(
        profile_id, False, 0, 0, 120.0,
        *(float("nan"), float("nan")), 0,
        *(float("nan"),) * 5, reason,
        *(float("nan"),) * 4, REGIME_CODES["unknown"], source_code, -1,
    )


def _point_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    density: np.ndarray,
    times: np.ndarray,
    source_code: int,
) -> tuple[np.ndarray, int]:
    base = (
        np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
        & np.isfinite(density) & np.isfinite(times)
        & (alt >= 120.0) & (alt <= 500.0)
    )
    if source_code in FY_SOURCE_CODES:
        density_valid = (
            (density >= FY_DENSITY_MIN_M3)
            & (density <= FY_DENSITY_MAX_M3)
        )
        return base & density_valid, int(np.count_nonzero(base & ~density_valid))
    return base & (density > 0), 0


def evaluate_profile(
    physical: np.ndarray,
    profile_id: int,
    source: str,
    relative_hour: float | None = None,
) -> QCResult:
    """Evaluate one profile; physical columns are [lat, lon, alt, Ne, time?]."""
    source_code = SOURCE_CODES[source]
    physical = np.asarray(physical, dtype=np.float64)
    if physical.ndim != 2 or physical.shape[1] < 4:
        return _empty_result(profile_id, source_code, REASON_BITS["read_or_time"])

    lat, lon, alt, density = (physical[:, i] for i in range(4))
    time_values = (
        physical[:, 4] if physical.shape[1] > 4
        else np.full(len(physical), relative_hour, dtype=np.float64)
    )
    finite, range_rejected_points = _point_mask(
        lat, lon, alt, density, time_values, source_code
    )
    input_points = int(len(physical))
    lat, lon, alt, density, time_values = (
        values[finite] for values in (lat, lon, alt, density, time_values)
    )
    if len(alt) == 0:
        result = _empty_result(profile_id, source_code, REASON_BITS["points"])
        result.input_points = input_points
        result.range_rejected_points = range_rejected_points
        return result

    representative_lat = float(np.mean(lat))
    representative_lon = float(np.mean(lon))
    representative_time = float(np.mean(time_values))
    try:
        sza = _solar_zenith(
            representative_time, representative_lon, representative_lat
        )
        observed = START_UTC + timedelta(hours=representative_time)
        date_code = int(observed.strftime("%Y%m%d"))
    except (OverflowError, ValueError):
        sza, date_code = float("nan"), -1
    regime = _regime(sza)
    reason = 0
    span = float(np.ptp(alt)) if len(alt) else 0.0
    if len(alt) < 20:
        reason |= REASON_BITS["points"]
    if span < 150.0:
        reason |= REASON_BITS["altitude_span"]
    if reason:
        result = QCResult(
            profile_id, False, input_points, len(alt), 120.0,
            float("nan"), float("nan"), 0,
            *(float("nan"),) * 5, reason,
            representative_lat, representative_lon, representative_time, sza,
            REGIME_CODES[regime], source_code, date_code,
        )
        result.range_rejected_points = range_rejected_points
        return result

    unique_alt, unique_density = _collapse_altitudes(alt, density)
    grid = np.arange(unique_alt[0], unique_alt[-1] + 1.25, 2.5)
    density_grid = np.interp(grid, unique_alt, unique_density)
    median30 = median_filter(
        density_grid, size=_odd_window(round(30.0 / 2.5), len(grid)),
        mode="nearest",
    )
    sg_window = _odd_window(round(100.0 / 2.5), len(grid), minimum=5)
    detection = savgol_filter(
        median30, sg_window, min(3, sg_window - 2), mode="interp"
    )
    detection = np.clip(detection, 1.0, None)
    peaks, _ = find_peaks(
        detection,
        prominence=0.05 * float(np.max(detection)),
        distance=max(1, int(round(30.0 / 2.5))),
    )
    peak_count = int(len(peaks))
    f2_peaks = peaks[(grid[peaks] >= 200.0) & (grid[peaks] <= 450.0)]
    if len(f2_peaks):
        f2_index = int(f2_peaks[np.argmax(grid[f2_peaks])])
        hmf2, nmf2 = float(grid[f2_index]), float(detection[f2_index])
    else:
        hmf2, nmf2 = float("nan"), float("nan")

    if not 1 <= peak_count <= 5:
        reason |= REASON_BITS["peak_count"]
    if not np.isfinite(hmf2) or not 200.0 <= hmf2 <= 450.0:
        reason |= REASON_BITS["hmf2"]
    if not np.isfinite(nmf2) or not 1e9 <= nmf2 <= 5e12:
        reason |= REASON_BITS["nmf2"]

    h_cut = (
        _bottom_cut(grid, median30, hmf2, nmf2, regime)
        if np.isfinite(hmf2) and np.isfinite(nmf2) else 120.0
    )
    sorted_alt = np.sort(alt)
    gaps = np.flatnonzero(np.diff(sorted_alt) >= 10.0)
    if len(gaps):
        h_cut = max(h_cut, float(sorted_alt[gaps[-1] + 1]))

    kept = alt >= h_cut
    kept_points = int(np.count_nonzero(kept))
    if kept_points < 20:
        reason |= REASON_BITS["points"]
    if kept_points < 2 or float(np.ptp(alt[kept])) < 150.0:
        reason |= REASON_BITS["altitude_span"]

    diagnostic = grid >= h_cut
    grid_qc, density_qc = grid[diagnostic], density_grid[diagnostic]
    background = median_filter(density_qc, size=9, mode="nearest")
    mean_density = float(np.mean(density_qc)) if len(density_qc) else 0.0
    md = (
        float(np.mean(np.abs(density_qc - background)) / mean_density)
        if mean_density > 0 else float("nan")
    )
    top = grid_qc >= 300.0
    delta = (
        float(np.sqrt(np.mean((density_qc[top] - background[top]) ** 2)) / nmf2)
        if np.count_nonzero(top) and np.isfinite(nmf2) and nmf2 > 0
        else float("nan")
    )
    fold = grid_qc >= max(h_cut, 200.0)
    fold_error = (
        float(np.mean(
            np.abs(density_qc[fold] - background[fold])
            / np.clip(background[fold], 1.0, None) > 0.3
        )) if np.count_nonzero(fold) else float("nan")
    )
    global_gradient = _gradient(
        grid_qc, background,
        grid_qc >= max(300.0, hmf2 if np.isfinite(hmf2) else 300.0),
    )
    local_gradient = _gradient(
        grid_qc, background, (grid_qc >= 420.0) & (grid_qc <= 490.0)
    )

    if source.startswith("FY"):
        if not np.isfinite(md) or md >= 0.1:
            reason |= REASON_BITS["md"]
        if not np.isfinite(delta) or delta >= 0.05:
            reason |= REASON_BITS["delta"]
        if not np.isfinite(global_gradient) or global_gradient >= 0.0:
            reason |= REASON_BITS["global_gradient"]
        if not np.isfinite(local_gradient) or local_gradient >= 0.0:
            reason |= REASON_BITS["local_gradient"]
    else:
        if not np.isfinite(md) or not 0.0 < md < 1.5:
            reason |= REASON_BITS["md"]
        if not np.isfinite(delta) or delta >= 0.02:
            reason |= REASON_BITS["delta"]
        if not np.isfinite(local_gradient) or local_gradient > -1e4:
            reason |= REASON_BITS["local_gradient"]
    if not np.isfinite(fold_error) or fold_error > 0.15:
        reason |= REASON_BITS["fold_error"]

    result = QCResult(
        profile_id, reason == 0, input_points, kept_points, h_cut,
        hmf2, nmf2, peak_count, md, delta, global_gradient, local_gradient,
        fold_error, reason, representative_lat, representative_lon,
        representative_time, sza, REGIME_CODES[regime], source_code, date_code,
    )
    result.range_rejected_points = range_rejected_points
    return result


def _output_rows(physical: np.ndarray, result: QCResult) -> np.ndarray:
    lat, lon, alt, density = (physical[:, i] for i in range(4))
    times = (
        physical[:, 4] if physical.shape[1] > 4
        else np.full(len(physical), result.representative_time)
    )
    valid, range_rejected_points = _point_mask(
        lat, lon, alt, density, times, result.source_code
    )
    valid &= alt >= result.h_cut_km
    if range_rejected_points != result.range_rejected_points:
        raise ValueError(
            f"profile {result.profile_id} physical-range count changed between QC passes"
        )
    rows = np.column_stack(
        [lat[valid], lon[valid], alt[valid], times[valid], np.log10(density[valid])]
    ).astype(np.float32)
    if len(rows) != result.kept_points or not np.isfinite(rows).all():
        raise ValueError(f"profile {result.profile_id} changed between QC passes")
    return rows


def _metadata_arrays(
    results: list[QCResult], relative_paths: list[str], original_ids: np.ndarray
) -> dict[str, np.ndarray]:
    fields = tuple(QCResult.__dataclass_fields__)
    arrays = {
        field: np.asarray([getattr(result, field) for result in results])
        for field in fields
    }
    arrays["profile_id"] = arrays["profile_id"].astype(np.int64)
    arrays["pass_profile"] = arrays["pass_profile"].astype(bool)
    arrays["reason_bits"] = arrays["reason_bits"].astype(np.int64)
    arrays["original_relative_path"] = np.asarray(relative_paths, dtype=np.str_)
    arrays["original_profile_id"] = np.asarray(original_ids, dtype=np.int64)
    arrays["reason_names"] = np.asarray(list(REASON_BITS), dtype=np.str_)
    arrays["reason_masks"] = np.asarray(list(REASON_BITS.values()), dtype=np.int64)
    return arrays


def _atomic_commit(temporary: list[Path], final: list[Path]) -> None:
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for destination in final:
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.bak"
                )
                os.replace(destination, backup)
                backups.append((destination, backup))
        for source, destination in zip(temporary, final):
            os.replace(source, destination)
            committed.append(destination)
    except Exception:
        for destination in committed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups:
            os.replace(backup, destination)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def _audit_report(
    source: str,
    results: list[QCResult],
    input_identity: dict,
    output_npy: Path,
    output_npz: Path,
) -> dict:
    passed = np.asarray([item.pass_profile for item in results], dtype=bool)
    pass_rate = float(np.mean(passed)) if len(passed) else 0.0
    gate = (0.20, 0.80) if source == "FY" else (0.50, 0.95)

    def grouped(attribute: str) -> dict:
        total, accepted = Counter(), Counter()
        for item in results:
            key = str(getattr(item, attribute))
            total[key] += 1
            if item.pass_profile:
                accepted[key] += 1
        return {
            key: {
                "total": total[key],
                "passed": accepted[key],
                "pass_rate": accepted[key] / total[key],
            }
            for key in sorted(total)
        }

    reasons = {
        name: int(sum(bool(item.reason_bits & bit) for item in results))
        for name, bit in REASON_BITS.items()
    }
    range_by_source = {}
    for name, code in SOURCE_CODES.items():
        selected = [item for item in results if item.source_code == code]
        if not selected:
            continue
        range_by_source[name] = {
            "profiles": int(sum(item.range_rejected_points > 0 for item in selected)),
            "points": int(sum(item.range_rejected_points for item in selected)),
        }
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "algorithm": {
            "altitude_km": [120.0, 500.0],
            "diagnostic_grid_km": 2.5,
            "background_points": 9,
            "peak_sg_km": 100.0,
            "bottom_median_km": 30.0,
            "point_density_filter": {
                "enabled": source == "FY",
                "unit": "m^-3",
                "minimum": FY_DENSITY_MIN_M3 if source == "FY" else None,
                "maximum": FY_DENSITY_MAX_M3 if source == "FY" else None,
                "action": "remove point, then recompute full-profile QC",
                "reference": (
                    "Tan Guangyuan (2021), FY3-GNOS ionospheric occultation "
                    "product assessment, Section 4.2.2"
                    if source == "FY" else None
                ),
            },
            "reason_bits": REASON_BITS,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "input": input_identity,
        "profiles": {
            "total": len(results),
            "passed": int(passed.sum()),
            "failed": int((~passed).sum()),
            "pass_rate": pass_rate,
        },
        "audit": {
            "required_pass_rate": list(gate),
            "passed": gate[0] <= pass_rate <= gate[1],
        },
        "distributions": {
            "regime_code": grouped("regime_code"),
            "source_code": grouped("source_code"),
            "date_code": grouped("date_code"),
            "failure_reasons": reasons,
            "physical_range_rejections": {
                "profiles": int(sum(
                    item.range_rejected_points > 0 for item in results
                )),
                "points": int(sum(
                    item.range_rejected_points for item in results
                )),
                "by_source": range_by_source,
            },
        },
        "outputs": {
            "npy": {"path": str(output_npy), "sha256": _sha256(output_npy)},
            "npz": {"path": str(output_npz), "sha256": _sha256(output_npz)},
        },
    }


def _save_anomaly_plots(
    selections: dict[str, list[tuple[str, np.ndarray, QCResult]]],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir.with_name(
        f"{output_dir.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    for reason, profiles in selections.items():
        reason_dir = output_dir / reason
        reason_dir.mkdir(parents=True, exist_ok=True)
        for label, physical, result in profiles:
            order = np.argsort(physical[:, 2], kind="stable")
            altitude = physical[order, 2]
            density = physical[order, 3]
            figure, axis = plt.subplots(figsize=(4.5, 6.0))
            axis.plot(np.log10(np.clip(density, 1.0, None)), altitude, ".-", ms=2)
            axis.axhline(result.h_cut_km, color="tab:red", ls="--")
            axis.set(
                xlabel="log10 Ne (m-3)", ylabel="Altitude (km)",
                title=f"{label}\n{reason}; bits={result.reason_bits}",
            )
            axis.grid(alpha=0.25)
            figure.tight_layout()
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            figure.savefig(reason_dir / f"{safe_label}.png", dpi=150)
            plt.close(figure)


def _write_products(
    source: str,
    results: list[QCResult],
    relative_paths: list[str],
    original_ids: np.ndarray,
    output_npy: Path,
    input_identity: dict,
    row_reader,
    stat_checker,
) -> dict:
    output_npz = output_npy.with_name(output_npy.stem + "_index.npz")
    output_report = output_npy.with_name(output_npy.stem + "_report.json")
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(
        np.asarray([
            item.representative_time
            if np.isfinite(item.representative_time) else np.inf
            for item in results
        ]),
        kind="stable",
    )
    total_points = int(sum(results[index].kept_points for index in order
                           if results[index].pass_profile))
    suffix = f".{uuid.uuid4().hex}.tmp"
    temp_npy = output_npy.with_name(output_npy.name + suffix)
    temp_npz = output_npz.with_name(output_npz.name + suffix)
    temp_report = output_report.with_name(output_report.name + suffix)
    temporary = [temp_npy, temp_npz, temp_report]
    anomaly_selections: dict[str, list[tuple[str, np.ndarray, QCResult]]] = {}
    try:
        mapped = np.lib.format.open_memmap(
            temp_npy, mode="w+", dtype=np.float32, shape=(total_points, 5)
        )
        cursor = 0
        for position, index in enumerate(order):
            result = results[index]
            if result.pass_profile:
                physical, label = row_reader(index)
                rows = _output_rows(physical, result)
                mapped[cursor:cursor + len(rows)] = rows
                result.output_start = cursor
                cursor += len(rows)
                result.output_end = cursor
            else:
                reasons = [
                    name for name, bit in REASON_BITS.items()
                    if result.reason_bits & bit
                    and len(anomaly_selections.setdefault(name, [])) < 20
                ]
                if reasons:
                    try:
                        physical, label = row_reader(index)
                    except Exception:
                        pass
                    else:
                        for reason in reasons:
                            anomaly_selections[reason].append(
                                (label, physical, result))
            if position and position % 5000 == 0:
                print(
                    f"[{source}] write {position:,}/{len(order):,}",
                    flush=True)
        mapped.flush()
        del mapped
        if cursor != total_points:
            raise AssertionError("QC output point count mismatch")
        stat_checker()
        loaded = np.load(temp_npy, mmap_mode="r")
        if loaded.shape != (total_points, 5) or not np.isfinite(loaded).all():
            raise ValueError("QC NPY validation failed")
        del loaded

        arrays = _metadata_arrays(results, relative_paths, original_ids)
        with temp_npz.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(temp_npz, allow_pickle=False) as index:
            if len(index["profile_id"]) != len(results):
                raise ValueError("QC NPZ validation failed")

        # Hash final candidates before the three-file transaction.
        staged_npy = output_npy.with_name(output_npy.name + ".hash-stage")
        staged_npz = output_npz.with_name(output_npz.name + ".hash-stage")
        os.replace(temp_npy, staged_npy)
        os.replace(temp_npz, staged_npz)
        report = _audit_report(
            source, results, input_identity, staged_npy, staged_npz
        )
        report["outputs"]["npy"]["path"] = str(output_npy)
        report["outputs"]["npz"]["path"] = str(output_npz)
        with temp_report.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        temporary = [staged_npy, staged_npz, temp_report]
        _atomic_commit(
            temporary, [output_npy, output_npz, output_report]
        )
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)

    try:
        _save_anomaly_plots(
            anomaly_selections,
            output_npy.with_name(output_npy.stem + "_anomalies"),
        )
    except Exception as error:
        print(f"[{source}] anomaly plots skipped: {error}")
    return json.loads(output_report.read_text(encoding="utf-8"))


def process_fy() -> dict:
    files = sorted(
        (
            (directory.name, path.relative_to(directory).as_posix(), path)
            for directory in FY_DIRS for path in directory.glob("*.txt")
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not files:
        raise FileNotFoundError("no FY profile TXT files found")
    results: list[QCResult] = []
    relative_paths: list[str] = []
    file_state: list[tuple[int, int, str]] = []
    aggregate = hashlib.sha256()
    total_size = 0
    for profile_id, (mission, relative, path) in enumerate(files):
        key = f"{mission}/{relative}"
        relative_paths.append(key)
        try:
            content = path.read_bytes()
            observed, relative_hour = _parse_fy_time(path)
            raw = _read_fy(content)
            physical = np.column_stack(
                [raw[:, 1], raw[:, 0], raw[:, 2], raw[:, 3]]
            )
            result = evaluate_profile(
                physical, profile_id, mission, relative_hour
            )
            result.date_code = int(observed.strftime("%Y%m%d"))
        except Exception:
            content = path.read_bytes() if path.exists() else b""
            result = _empty_result(
                profile_id, SOURCE_CODES[mission], REASON_BITS["read_or_time"]
            )
        digest = hashlib.sha256(content).hexdigest()
        stat = path.stat()
        file_state.append((stat.st_size, stat.st_mtime_ns, digest))
        total_size += stat.st_size
        aggregate.update(key.encode("utf-8"))
        aggregate.update(str(stat.st_size).encode("ascii"))
        aggregate.update(bytes.fromhex(digest))
        results.append(result)
        if profile_id and profile_id % 5000 == 0:
            print(f"[FY] QC {profile_id:,}/{len(files):,}", flush=True)

    def read_row(index: int):
        mission, relative, path = files[index]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != file_state[index][2]:
            raise RuntimeError(f"FY input changed: {path}")
        raw = _read_fy(content)
        physical = np.column_stack(
            [raw[:, 1], raw[:, 0], raw[:, 2], raw[:, 3]]
        )
        return physical, f"{mission}_{Path(relative).stem}"

    def check_state():
        for index, (_, _, path) in enumerate(files):
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != file_state[index][:2]:
                raise RuntimeError(f"FY input changed during QC: {path}")

    identity = {
        "directories": [str(path) for path in FY_DIRS],
        "profile_files": len(files),
        "total_bytes": total_size,
        "aggregate_sha256": aggregate.hexdigest(),
    }
    return _write_products(
        "FY", results, relative_paths,
        np.full(len(results), -1, dtype=np.int64),
        FY_OUTPUT, identity, read_row, check_state,
    )


def process_cosmic() -> dict:
    input_sha = _sha256(COSMIC_INPUT)
    stat = COSMIC_INPUT.stat()
    raw = np.load(COSMIC_INPUT, mmap_mode="r")
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError("COSMIC must contain six columns")
    profile_raw = np.asarray(raw[:, 5])
    if (
        not np.isfinite(profile_raw).all()
        or not np.array_equal(profile_raw, np.rint(profile_raw))
    ):
        raise ValueError("COSMIC profile_id must contain finite integers")
    profile_raw = np.rint(profile_raw).astype(np.int64)
    order = np.argsort(profile_raw, kind="stable")
    sorted_ids = profile_raw[order]
    breaks = np.flatnonzero(np.diff(sorted_ids)) + 1
    groups = np.split(order, breaks)
    original_ids = np.asarray([sorted_ids[group[0]] for group in groups])
    results: list[QCResult] = []
    for index, group in enumerate(groups):
        rows = np.asarray(raw[group, :5], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            density = np.power(10.0, rows[:, 4])
        physical = np.column_stack(
            [rows[:, 0], rows[:, 1], rows[:, 2], density, rows[:, 3]]
        )
        results.append(evaluate_profile(
            physical, int(original_ids[index]), "COSMIC"
        ))
        if index and index % 5000 == 0:
            print(f"[COSMIC] QC {index:,}/{len(groups):,}", flush=True)

    def read_row(index: int):
        rows = np.asarray(raw[groups[index], :5], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            density = np.power(10.0, rows[:, 4])
        return np.column_stack(
            [rows[:, 0], rows[:, 1], rows[:, 2], density, rows[:, 3]]
        ), f"COSMIC_{original_ids[index]}"

    def check_state():
        current = COSMIC_INPUT.stat()
        if (
            current.st_size != stat.st_size
            or current.st_mtime_ns != stat.st_mtime_ns
            or _sha256(COSMIC_INPUT) != input_sha
        ):
            raise RuntimeError("COSMIC input changed during QC")

    identity = {
        "path": str(COSMIC_INPUT),
        "rows": int(len(raw)),
        "profile_count": len(groups),
        "size": stat.st_size,
        "sha256": input_sha,
    }
    return _write_products(
        "COSMIC", results, [""] * len(results), original_ids,
        COSMIC_OUTPUT, identity, read_row, check_state,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("fy", "cosmic", "both"), default="both")
    args = parser.parse_args()
    reports = []
    if args.source in ("fy", "both"):
        reports.append(process_fy())
    if args.source in ("cosmic", "both"):
        reports.append(process_cosmic())
    for report in reports:
        summary = report["profiles"] | {"audit_passed": report["audit"]["passed"]}
        print(f"{report['source']}: {json.dumps(summary, ensure_ascii=False)}")
    return 0 if all(report["audit"]["passed"] for report in reports) else 2


if __name__ == "__main__":
    sys.exit(main())
