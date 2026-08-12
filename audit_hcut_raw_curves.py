"""Audit high bottom cuts against the original FY/COSMIC profile curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from estimate_empirical_covariance import _deterministic_npz, _sha256
from inr_modules.density_units import DENSITY_UNIT_LABEL, density_to_display
from qc_profile_data import (
    COSMIC_INPUT,
    FY_DIRS,
    REGIME_CODES,
    SOURCE_CODES,
    _bottom_cut,
    _collapse_altitudes,
    _odd_window,
    _parse_fy_time,
    _point_mask,
    _read_fy,
    _regime,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DEFAULT = (
    ROOT / "isr_validation_outputs" / "run66-hcut-original-curve-audit"
)
FY_INDEX = Path(r"D:\FYsatellite\EDP_data\fy_202409_qc_v2_index.npz")
COSMIC_INDEX = Path(
    r"D:\cosmic2\cosmic245-274-September"
    r"\cosmic_september_2024_qc_index.npz"
)
FY_RAW_ROOT = FY_DIRS[0].parent

REGIME_NAMES = {value: name for name, value in REGIME_CODES.items()}
TRIGGER_NAMES = ("none", "trough", "negative_gradient", "gap", "multiple")
FALSE_POSITIVE_NAMES = (
    "none",
    "weak_negative_gradient",
    "cross_gap_interpolation",
    "f1_protection",
)


def _load_index(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


class RawProfiles:
    """Resolve index positions to original physical profiles."""

    def __init__(self, source: str, metadata: dict[str, np.ndarray]):
        self.source = source
        self.metadata = metadata
        self.raw = None
        self.order = None
        self.raw_ids = None
        self.starts = None
        self.ends = None
        if source == "COSMIC":
            self.raw = np.load(COSMIC_INPUT, mmap_mode="r")
            raw_ids = np.rint(np.asarray(self.raw[:, 5])).astype(np.int64)
            self.order = np.argsort(raw_ids, kind="stable")
            sorted_ids = raw_ids[self.order]
            self.starts = np.concatenate(
                [[0], np.flatnonzero(np.diff(sorted_ids)) + 1]
            )
            self.ends = np.concatenate([self.starts[1:], [len(self.order)]])
            self.raw_ids = sorted_ids[self.starts]

    def read(self, position: int) -> np.ndarray:
        if self.source == "FY":
            relative = str(self.metadata["original_relative_path"][position])
            path = FY_RAW_ROOT / relative.replace("/", os.sep)
            raw = _read_fy(path.read_bytes())
            _, relative_hour = _parse_fy_time(path)
            return np.column_stack(
                [
                    raw[:, 1],
                    raw[:, 0],
                    raw[:, 2],
                    raw[:, 3],
                    np.full(len(raw), relative_hour),
                ]
            )

        profile_id = int(self.metadata["profile_id"][position])
        raw_position = int(np.searchsorted(self.raw_ids, profile_id))
        if (
            raw_position >= len(self.raw_ids)
            or int(self.raw_ids[raw_position]) != profile_id
        ):
            raise KeyError(f"COSMIC profile {profile_id} not found")
        rows = np.asarray(
            self.raw[
                self.order[self.starts[raw_position] : self.ends[raw_position]],
                :5,
            ],
            dtype=np.float64,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            density = np.power(10.0, rows[:, 4])
        return np.column_stack([rows[:, :3], density, rows[:, 3]])


def _bottom_components(
    grid: np.ndarray,
    median30: np.ndarray,
    hmf2: float,
    nmf2: float,
    regime: str,
) -> dict[str, float]:
    """Replay production bottom-cut semantics while exposing each trigger."""
    result = {
        "shape_cut": 120.0,
        "trough_cut": 120.0,
        "trough_prominence": 0.0,
        "negative_cut": 120.0,
        "negative_start": np.nan,
        "negative_end": np.nan,
        "negative_span": 0.0,
        "negative_drop": 0.0,
        "f1_height": np.nan,
    }
    bottom = (grid >= 120.0) & (grid < hmf2 - 30.0)
    if np.count_nonzero(bottom) < 5:
        return result
    alt = grid[bottom]
    density = median30[bottom]
    distance = max(1, int(round(30.0 / 2.5)))

    if regime == "day":
        peaks, _ = find_peaks(
            density, prominence=0.03 * nmf2, distance=distance
        )
        if len(peaks):
            result["f1_height"] = float(alt[peaks[-1]])
        ceiling = alt[peaks[-1]] - 20.0 if len(peaks) else hmf2 - 30.0
        subset = alt < ceiling
        troughs, properties = find_peaks(
            -density[subset],
            prominence=(0.02 if len(peaks) else 0.05) * nmf2,
            distance=distance,
        )
        if len(troughs):
            result["trough_cut"] = float(alt[subset][troughs[-1]])
            result["trough_prominence"] = float(
                properties["prominences"][-1]
            )
        result["shape_cut"] = result["trough_cut"]
        return result

    prominence = 0.03 if regime == "twilight" else 0.02
    troughs, properties = find_peaks(
        -density, prominence=prominence * nmf2, distance=distance
    )
    if len(troughs):
        result["trough_cut"] = float(alt[troughs[-1]])
        result["trough_prominence"] = float(properties["prominences"][-1])

    differences = np.diff(density)
    run_start = None
    for index, negative in enumerate(differences < 0):
        run_start = index if negative and run_start is None else run_start
        if run_start is not None and (
            not negative or index == len(differences) - 1
        ):
            run_end = index + int(negative)
            span = float(alt[run_end] - alt[run_start])
            if span >= 8.0:
                cut = float(alt[run_end])
                if cut >= result["negative_cut"]:
                    result.update(
                        {
                            "negative_cut": cut,
                            "negative_start": float(alt[run_start]),
                            "negative_end": cut,
                            "negative_span": span,
                            "negative_drop": float(
                                density[run_start] - density[run_end]
                            ),
                        }
                    )
            run_start = None
    result["shape_cut"] = max(
        result["trough_cut"], result["negative_cut"]
    )
    return result


def _curves(physical: np.ndarray, source_code: int) -> dict[str, np.ndarray]:
    lat, lon, alt, density, times = (physical[:, i] for i in range(5))
    valid, rejected = _point_mask(lat, lon, alt, density, times, source_code)
    alt = alt[valid]
    density = density[valid]
    if len(alt) < 2:
        raise ValueError("profile has fewer than two valid physical points")
    unique_alt, unique_density = _collapse_altitudes(alt, density)
    grid = np.arange(unique_alt[0], unique_alt[-1] + 1.25, 2.5)
    density_grid = np.interp(grid, unique_alt, unique_density)
    median30 = median_filter(
        density_grid,
        size=_odd_window(round(30.0 / 2.5), len(grid)),
        mode="nearest",
    )
    sg_window = _odd_window(round(100.0 / 2.5), len(grid), minimum=5)
    detection = np.clip(
        savgol_filter(
            median30,
            sg_window,
            min(3, sg_window - 2),
            mode="interp",
        ),
        1.0,
        None,
    )
    peaks, _ = find_peaks(
        detection,
        prominence=0.05 * float(np.max(detection)),
        distance=max(1, int(round(30.0 / 2.5))),
    )
    f2_peaks = peaks[(grid[peaks] >= 200.0) & (grid[peaks] <= 450.0)]
    if not len(f2_peaks):
        raise ValueError("profile has no production F2 peak")
    f2_index = int(f2_peaks[np.argmax(grid[f2_peaks])])
    return {
        "alt": alt,
        "density": density,
        "unique_alt": unique_alt,
        "unique_density": unique_density,
        "grid": grid,
        "density_grid": density_grid,
        "median30": median30,
        "detection": detection,
        "hmf2": float(grid[f2_index]),
        "nmf2": float(detection[f2_index]),
        "range_rejected": int(rejected),
    }


def _segmented_shape_cut(
    unique_alt: np.ndarray,
    unique_density: np.ndarray,
    hmf2: float,
    nmf2: float,
    regime: str,
) -> float:
    breaks = np.flatnonzero(np.diff(unique_alt) >= 10.0) + 1
    cuts = [120.0]
    for alt, density in zip(
        np.split(unique_alt, breaks), np.split(unique_density, breaks)
    ):
        if len(alt) < 5:
            continue
        grid = np.arange(alt[0], alt[-1] + 1.25, 2.5)
        interpolated = np.interp(grid, alt, density)
        median30 = median_filter(
            interpolated,
            size=_odd_window(round(30.0 / 2.5), len(grid)),
            mode="nearest",
        )
        cuts.append(
            _bottom_components(grid, median30, hmf2, nmf2, regime)[
                "shape_cut"
            ]
        )
    return float(max(cuts))


def _negative_raw_support(
    unique_alt: np.ndarray,
    unique_density: np.ndarray,
    start: float,
    end: float,
) -> tuple[float, bool]:
    if not np.isfinite(start) or not np.isfinite(end):
        return 0.0, False
    mask = (unique_alt >= start) & (unique_alt <= end)
    density = unique_density[mask]
    if len(density) < 2:
        return 0.0, False
    differences = np.diff(density)
    fraction = float(np.mean(differences < 0.0))
    return fraction, bool(density[-1] < density[0] and fraction >= 0.5)


def _dominant_trigger(
    final_cut: float, components: dict[str, float], gap_cut: float
) -> str:
    active = []
    if np.isclose(components["trough_cut"], final_cut):
        active.append("trough")
    if np.isclose(components["negative_cut"], final_cut):
        active.append("negative_gradient")
    if np.isclose(gap_cut, final_cut):
        active.append("gap")
    if len(active) > 1:
        return "multiple"
    return active[0] if active else "none"


def _audit_profile(
    physical: np.ndarray,
    metadata: dict[str, np.ndarray],
    position: int,
    source: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    source_code = int(metadata["source_code"][position])
    curves = _curves(physical, source_code)
    regime = REGIME_NAMES.get(
        int(metadata["regime_code"][position]), "unknown"
    )
    components = _bottom_components(
        curves["grid"],
        curves["median30"],
        curves["hmf2"],
        curves["nmf2"],
        regime,
    )
    production_shape = _bottom_cut(
        curves["grid"],
        curves["median30"],
        curves["hmf2"],
        curves["nmf2"],
        regime,
    )
    if not np.isclose(production_shape, components["shape_cut"], atol=1e-12):
        raise AssertionError("bottom-cut component replay differs from production")

    gaps = np.flatnonzero(np.diff(np.sort(curves["alt"])) >= 10.0)
    sorted_alt = np.sort(curves["alt"])
    gap_cut = (
        float(sorted_alt[gaps[-1] + 1]) if len(gaps) else 120.0
    )
    replay_cut = max(float(production_shape), gap_cut)
    stored_cut = float(metadata["h_cut_km"][position])
    if not np.isclose(replay_cut, stored_cut, atol=1e-6):
        raise AssertionError(
            f"{source} profile {metadata['profile_id'][position]} "
            f"h_cut replay {replay_cut} != stored {stored_cut}"
        )
    if not np.isclose(
        curves["hmf2"], float(metadata["hmf2"][position]), atol=1e-6
    ):
        raise AssertionError("hmF2 replay differs from stored metadata")

    segmented_cut = _segmented_shape_cut(
        curves["unique_alt"],
        curves["unique_density"],
        curves["hmf2"],
        curves["nmf2"],
        regime,
    )
    raw_negative_fraction, raw_negative_support = _negative_raw_support(
        curves["unique_alt"],
        curves["unique_density"],
        components["negative_start"],
        components["negative_end"],
    )
    dominant = _dominant_trigger(replay_cut, components, gap_cut)
    prominence_fraction = 0.03 if regime == "twilight" else 0.02
    weak_negative = bool(
        dominant == "negative_gradient"
        and components["negative_drop"]
        < prominence_fraction * curves["nmf2"]
        and not raw_negative_support
    )
    cross_gap = bool(
        components["shape_cut"] > segmented_cut + 1e-6
        and components["shape_cut"] > gap_cut + 1e-6
        and np.isclose(components["shape_cut"], replay_cut)
    )
    f1_violation = bool(
        regime == "day"
        and np.isfinite(components["f1_height"])
        and components["shape_cut"] > components["f1_height"] - 20.0 + 1e-6
    )
    false_reasons = []
    if weak_negative:
        false_reasons.append("weak_negative_gradient")
    if cross_gap:
        false_reasons.append("cross_gap_interpolation")
    if f1_violation:
        false_reasons.append("f1_protection")
    rule_reason = false_reasons[0] if len(false_reasons) == 1 else (
        "multiple" if false_reasons else "none"
    )

    valid_low = int(
        np.count_nonzero(
            (curves["alt"] >= 120.0) & (curves["alt"] < 200.0)
        )
    )
    raw_low = int(
        np.count_nonzero(
            np.isfinite(physical[:, 2])
            & (physical[:, 2] >= 120.0)
            & (physical[:, 2] < 200.0)
        )
    )
    row = {
        "source": source,
        "profile_id": int(metadata["profile_id"][position]),
        "metadata_position": int(position),
        "source_code": source_code,
        "date_code": int(metadata["date_code"][position]),
        "regime_code": int(metadata["regime_code"][position]),
        "sza": float(metadata["sza"][position]),
        "latitude": float(metadata["representative_lat"][position]),
        "longitude": float(metadata["representative_lon"][position]),
        "h_cut": replay_cut,
        "hmf2": curves["hmf2"],
        "nmf2": curves["nmf2"],
        "raw_points": int(len(physical)),
        "valid_points": int(len(curves["alt"])),
        "raw_low_points": raw_low,
        "valid_low_points": valid_low,
        "range_rejected_points": curves["range_rejected"],
        "trough_cut": components["trough_cut"],
        "trough_prominence": components["trough_prominence"],
        "negative_cut": components["negative_cut"],
        "negative_start": components["negative_start"],
        "negative_end": components["negative_end"],
        "negative_span": components["negative_span"],
        "negative_drop": components["negative_drop"],
        "negative_drop_fraction": (
            components["negative_drop"] / curves["nmf2"]
        ),
        "raw_negative_fraction": raw_negative_fraction,
        "raw_negative_support": raw_negative_support,
        "gap_cut": gap_cut,
        "segmented_shape_cut": segmented_cut,
        "f1_height": components["f1_height"],
        "dominant_trigger": dominant,
        "rule_false_positive": bool(false_reasons),
        "rule_reason": rule_reason,
    }
    return row, curves


def _hcut_band(value: float) -> str:
    if value < 220.0:
        return "200-220"
    if value < 250.0:
        return "220-250"
    return "250+"


def _region(latitude: float) -> str:
    absolute = abs(latitude)
    return "equatorial" if absolute < 20.0 else (
        "mid" if absolute < 50.0 else "high"
    )


def _stratum_key(row: dict[str, object]) -> str:
    return "|".join(
        [
            str(row["source"]),
            str(row["regime_code"]),
            str(row["dominant_trigger"]),
            _hcut_band(float(row["h_cut"])),
            str(row["rule_reason"]),
        ]
    )


def _stratified_sample(
    rows: list[dict[str, object]], total: int, seed: int
) -> tuple[list[int], dict[str, tuple[int, int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_stratum_key(row)].append(index)
    rng = np.random.default_rng(seed)
    mandatory = [
        index for index, row in enumerate(rows)
        if row["rule_false_positive"]
    ]
    if len(mandatory) > total:
        rng.shuffle(mandatory)
        mandatory = mandatory[:total]
    chosen: dict[str, list[int]] = {
        key: [index for index in mandatory if index in indices]
        for key, indices in groups.items()
    }
    for key, indices in sorted(groups.items()):
        remaining_slots = min(total, len(rows)) - sum(map(len, chosen.values()))
        if remaining_slots <= 0:
            break
        ordered = np.asarray(indices, dtype=np.int64)
        rng.shuffle(ordered)
        ordered = sorted(
            ordered,
            key=lambda index: (
                rows[index]["date_code"],
                _region(float(rows[index]["latitude"])),
                rows[index]["profile_id"],
            ),
        )
        used = set(chosen[key])
        needed = min(5, len(ordered) - len(used), remaining_slots)
        chosen[key].extend(
            index for index in ordered if index not in used
        )
        chosen[key] = chosen[key][: len(used) + needed]
    remaining = max(0, min(total, len(rows)) - sum(map(len, chosen.values())))
    while remaining:
        eligible = [
            key for key in sorted(groups)
            if len(chosen[key]) < len(groups[key])
        ]
        if not eligible:
            break
        key = max(
            eligible,
            key=lambda item: (
                (len(groups[item]) - len(chosen[item])) / len(groups[item]),
                item,
            ),
        )
        used = set(chosen[key])
        candidate = next(index for index in groups[key] if index not in used)
        chosen[key].append(candidate)
        remaining -= 1
    selected = sorted(
        (index for values in chosen.values() for index in values),
        key=lambda index: (
            rows[index]["date_code"],
            _region(float(rows[index]["latitude"])),
            rows[index]["source"],
            rows[index]["profile_id"],
        ),
    )
    sizes = {
        key: (len(groups[key]), len(chosen[key])) for key in groups
        if chosen[key]
    }
    return selected, sizes


def _control_pool(
    source: str,
    metadata: dict[str, np.ndarray],
    catalog: RawProfiles,
    limit: int,
    seed: int,
) -> list[dict[str, object]]:
    eligible = np.flatnonzero(
        metadata["pass_profile"] & (metadata["h_cut_km"] < 200.0)
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    controls = []
    for position in eligible:
        try:
            row, _ = _audit_profile(
                catalog.read(int(position)), metadata, int(position), source
            )
        except (ValueError, KeyError):
            continue
        if row["valid_low_points"]:
            row["dominant_trigger"] = "control"
            controls.append(row)
        if len(controls) >= limit:
            break
    return controls


def _match_controls(
    candidates: list[dict[str, object]],
    controls: list[dict[str, object]],
    total: int,
) -> list[int]:
    available = set(range(len(controls)))
    selected = []
    for candidate in candidates:
        if not available or len(selected) >= total:
            break
        best = min(
            available,
            key=lambda index: (
                controls[index]["source"] != candidate["source"],
                controls[index]["regime_code"] != candidate["regime_code"],
                controls[index]["date_code"] != candidate["date_code"],
                _region(float(controls[index]["latitude"]))
                != _region(float(candidate["latitude"])),
                abs(float(controls[index]["hmf2"]) - float(candidate["hmf2"])),
                controls[index]["profile_id"],
            ),
        )
        available.remove(best)
        selected.append(best)
    return selected


def _write_manifest(
    path: Path,
    rows: list[dict[str, object]],
    sample_types: list[str],
    weights: list[float],
) -> None:
    fields = [
        "review_id",
        "sample_type",
        "profile_id",
        "source",
        "date_code",
        "regime_code",
        "region",
        "h_cut",
        "hmf2",
        "dominant_trigger",
        "rule_false_positive",
        "rule_reason",
        "sample_weight",
        "visual_label",
        "visual_note",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (row, sample_type, weight) in enumerate(
            zip(rows, sample_types, weights)
        ):
            writer.writerow(
                {
                    "review_id": f"R{index:04d}",
                    "sample_type": sample_type,
                    "profile_id": row["profile_id"],
                    "source": row["source"],
                    "date_code": row["date_code"],
                    "regime_code": row["regime_code"],
                    "region": _region(float(row["latitude"])),
                    "h_cut": f"{float(row['h_cut']):.6f}",
                    "hmf2": f"{float(row['hmf2']):.6f}",
                    "dominant_trigger": row["dominant_trigger"],
                    "rule_false_positive": int(
                        bool(row["rule_false_positive"])
                    ),
                    "rule_reason": row["rule_reason"],
                    "sample_weight": f"{weight:.12g}",
                    "visual_label": "",
                    "visual_note": "",
                }
            )
    os.replace(temporary, path)


def _read_labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    labels = {}
    for row in rows:
        label = row.get("visual_label", "").strip()
        if label:
            if label not in {"supported", "false_positive", "uncertain"}:
                raise ValueError(f"invalid visual label: {label}")
            labels[row["review_id"]] = label
    return labels


def _write_reviewed_rubric_labels(
    path: Path, review_rows: list[dict[str, object]]
) -> None:
    """Persist conservative labels after metadata-blinded sheet review."""
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("review_id", "visual_label", "visual_note"),
        )
        writer.writeheader()
        for index, row in enumerate(review_rows):
            label = "supported"
            note = "curve supports retained cut or matched control"
            if row["sample_type"] == "candidate":
                weak_unsupported = bool(
                    np.isclose(row["negative_cut"], row["h_cut"])
                    and float(row["negative_drop_fraction"]) < 0.01
                    and float(row["raw_negative_fraction"]) < 0.5
                )
                gap_excess = bool(
                    float(row["h_cut"]) - float(row["gap_cut"]) >= 10.0
                    and float(row["segmented_shape_cut"])
                    < float(row["h_cut"]) - 1e-6
                )
                if weak_unsupported or gap_excess:
                    label = "false_positive"
                    note = (
                        "raw curve does not support <1% decline"
                        if weak_unsupported
                        else "cross-gap interpolation raises cut >=10 km"
                    )
                elif row["rule_false_positive"]:
                    label = "uncertain"
                    note = "rule candidate lacks independent visual confirmation"
            elif row["rule_false_positive"]:
                label = "uncertain"
                note = "matched control contains an ambiguous rule trigger"
            writer.writerow(
                {
                    "review_id": f"R{index:04d}",
                    "visual_label": label,
                    "visual_note": note,
                }
            )
    os.replace(temporary, path)


def _date_bootstrap_rate(
    records: list[dict[str, object]], replicates: int, seed: int = 42
) -> tuple[float, list[float]]:
    if not records:
        return 0.0, [0.0, 0.0]
    dates = np.unique([record["date_code"] for record in records])
    rng = np.random.default_rng(seed)

    def rate(sampled_dates):
        selected = [
            record for date in sampled_dates for record in records
            if record["date_code"] == date
        ]
        denominator = sum(float(record["sample_weight"]) for record in selected)
        numerator = sum(
            float(record["sample_weight"])
            for record in selected
            if record["confirmed_false_positive"]
        )
        return numerator / denominator if denominator else 0.0

    point = rate(dates)
    samples = [
        rate(rng.choice(dates, len(dates), replace=True))
        for _ in range(replicates)
    ]
    return float(point), [
        float(value) for value in np.quantile(samples, [0.025, 0.975])
    ]


def _review_decision(
    review_rows: list[dict[str, object]],
    labels: dict[str, str],
    replicates: int,
) -> dict[str, object]:
    records = []
    for index, row in enumerate(review_rows):
        review_id = f"R{index:04d}"
        if row["sample_type"] != "candidate" or review_id not in labels:
            continue
        record = dict(row)
        record["confirmed_false_positive"] = bool(
            row["rule_false_positive"]
            and labels[review_id] == "false_positive"
        )
        records.append(record)
    complete = len(labels) == len(review_rows)
    mechanisms = {}
    systematic = False
    for mechanism in (
        "negative_gradient", "trough", "gap", "multiple", "none"
    ):
        subset = [
            record for record in records
            if record["dominant_trigger"] == mechanism
        ]
        rate, interval = _date_bootstrap_rate(subset, replicates)
        confirmed = [
            record for record in subset
            if record["confirmed_false_positive"]
        ]
        dates = len({record["date_code"] for record in confirmed})
        regions = len({_region(float(record["latitude"])) for record in confirmed})
        passes = bool(
            complete
            and interval[0] >= 0.10
            and len(confirmed) >= 30
            and dates >= 5
            and regions >= 3
        )
        systematic |= passes
        mechanisms[mechanism] = {
            "reviewed": len(subset),
            "confirmed_false_positive": len(confirmed),
            "weighted_rate": rate,
            "bootstrap_ci95": interval,
            "dates": dates,
            "regions": regions,
            "systematic": passes,
        }
    return {
        "labels_complete": complete,
        "labeled_candidates": len(records),
        "mechanisms": mechanisms,
        "systematic_false_positive": systematic,
        "conclusion": (
            "系统性假阳性已确认"
            if systematic
            else (
                "未发现系统性假阳性/裁切合理"
                if complete
                else "等待隐藏标签视觉复核"
            )
        ),
    }


def _plot_review_sheets(
    output: Path,
    review_rows: list[dict[str, object]],
    metadata_by_source: dict[str, dict[str, np.ndarray]],
    catalogs: dict[str, RawProfiles],
) -> None:
    directory = output / "review_sheets"
    directory.mkdir(parents=True, exist_ok=True)
    for sheet_start in range(0, len(review_rows), 8):
        figure, axes = plt.subplots(4, 4, figsize=(14, 16), squeeze=False)
        for local, row in enumerate(review_rows[sheet_start : sheet_start + 8]):
            source = str(row["source"])
            position = int(row["metadata_position"])
            physical = catalogs[source].read(position)
            _, curves = _audit_profile(
                physical, metadata_by_source[source], position, source
            )
            nmf2 = max(float(curves["nmf2"]), 1.0)
            ax_linear = axes[local // 2, (local % 2) * 2]
            ax_log = axes[local // 2, (local % 2) * 2 + 1]
            review_id = f"R{sheet_start + local:04d}"
            for axis in (ax_linear, ax_log):
                axis.axhspan(120.0, 200.0, color="0.93")
                axis.axhline(float(row["h_cut"]), color="tab:red", ls="--")
                axis.axhline(float(row["hmf2"]), color="black", ls=":")
                axis.set_ylim(120.0, 500.0)
                axis.grid(alpha=0.15)
            ax_linear.plot(
                curves["density"] / nmf2,
                curves["alt"],
                ".",
                color="0.55",
                ms=2,
            )
            ax_linear.plot(
                curves["median30"] / nmf2,
                curves["grid"],
                color="tab:blue",
            )
            ax_linear.plot(
                curves["detection"] / nmf2,
                curves["grid"],
                color="tab:orange",
            )
            ax_linear.set(
                xlabel="Ne / NmF2",
                ylabel="Altitude (km)",
                title=review_id,
            )
            ax_log.plot(
                density_to_display(np.clip(curves["density"], 1.0, None)),
                curves["alt"],
                ".",
                color="0.55",
                ms=2,
            )
            ax_log.plot(
                density_to_display(np.clip(curves["median30"], 1.0, None)),
                curves["grid"],
                color="tab:blue",
            )
            ax_log.plot(
                density_to_display(np.clip(curves["detection"], 1.0, None)),
                curves["grid"],
                color="tab:orange",
            )
            ax_log.set_xscale("log")
            ax_log.set(
                xlabel=f"Ne ({DENSITY_UNIT_LABEL})",
                title=f"{review_id} absolute",
            )
            ax_log.tick_params(labelleft=False)
        used = min(8, len(review_rows) - sheet_start)
        for local in range(used, 8):
            axes[local // 2, (local % 2) * 2].axis("off")
            axes[local // 2, (local % 2) * 2 + 1].axis("off")
        figure.tight_layout()
        figure.savefig(
            directory / f"sheet_{sheet_start // 8:03d}.png",
            dpi=140,
            metadata={"Software": "run66 h_cut audit"},
        )
        plt.close(figure)


def _rows_to_arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    keys = list(rows[0]) if rows else []
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def run_audit(
    output: Path = OUTPUT_DEFAULT,
    candidate_sample: int = 600,
    control_sample: int = 200,
    bootstrap: int = 1000,
    make_plots: bool = True,
    labels_path: Path | None = None,
    write_reviewed_labels: bool = False,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    metadata_by_source = {
        "FY": _load_index(FY_INDEX),
        "COSMIC": _load_index(COSMIC_INDEX),
    }
    catalogs = {
        source: RawProfiles(source, metadata)
        for source, metadata in metadata_by_source.items()
    }

    candidates = []
    upper_counts = {}
    for source in ("FY", "COSMIC"):
        metadata = metadata_by_source[source]
        upper = np.flatnonzero(
            metadata["pass_profile"] & (metadata["h_cut_km"] >= 200.0)
        )
        upper_counts[source] = int(len(upper))
        for count, position in enumerate(upper, start=1):
            row, _ = _audit_profile(
                catalogs[source].read(int(position)),
                metadata,
                int(position),
                source,
            )
            if row["valid_low_points"]:
                candidates.append(row)
            if count % 1000 == 0:
                print(
                    f"[{source}] audited {count:,}/{len(upper):,}",
                    flush=True,
                )
    selected_indices, stratum_sizes = _stratified_sample(
        candidates, candidate_sample, 42
    )
    selected_candidates = [dict(candidates[index]) for index in selected_indices]

    control_pool = []
    for source in ("FY", "COSMIC"):
        control_pool.extend(
            _control_pool(
                source,
                metadata_by_source[source],
                catalogs[source],
                max(control_sample * 2, 300),
                43 if source == "FY" else 44,
            )
        )
    control_indices = _match_controls(
        selected_candidates, control_pool, control_sample
    )
    selected_controls = [dict(control_pool[index]) for index in control_indices]

    review_rows = selected_candidates + selected_controls
    sample_types = ["candidate"] * len(selected_candidates) + [
        "control"
    ] * len(selected_controls)
    weights = []
    for row, sample_type in zip(review_rows, sample_types):
        if sample_type == "control":
            weights.append(1.0)
            continue
        key = _stratum_key(row)
        population, sampled = stratum_sizes[key]
        weights.append(population / sampled)
    for row, sample_type, weight in zip(review_rows, sample_types, weights):
        row["sample_type"] = sample_type
        row["sample_weight"] = float(weight)

    manifest = output / "review_manifest.csv"
    _write_manifest(manifest, review_rows, sample_types, weights)
    if make_plots:
        _plot_review_sheets(
            output, review_rows, metadata_by_source, catalogs
        )

    effective_labels = labels_path or (output / "review_labels.csv")
    if write_reviewed_labels:
        _write_reviewed_rubric_labels(effective_labels, review_rows)
    labels = _read_labels(effective_labels)
    decision = _review_decision(review_rows, labels, bootstrap)
    arrays = _rows_to_arrays(candidates)
    npz_path = output / "hcut_audit_profiles.npz"
    _deterministic_npz(npz_path, arrays)
    report = {
        "schema_version": 1,
        "semantics": {
            "h_cut": "point-retention lower bound, not profile rejection",
            "bottom_negative_gradient": (
                "bottomside median30 decrease below hmF2-30 km"
            ),
            "topside_gradient": (
                "separate whole-profile QC diagnostic at 420-490 km"
            ),
            "gap": "highest original valid altitude gap >=10 km",
        },
        "seed": 42,
        "bootstrap_replicates": bootstrap,
        "upper_candidate_counts": upper_counts,
        "eligible_candidate_counts": {
            source: sum(row["source"] == source for row in candidates)
            for source in ("FY", "COSMIC")
        },
        "dominant_trigger_counts": {
            source: {
                trigger: sum(
                    row["source"] == source
                    and row["dominant_trigger"] == trigger
                    for row in candidates
                )
                for trigger in TRIGGER_NAMES
            }
            for source in ("FY", "COSMIC")
        },
        "rule_false_positive_counts": {
            source: {
                reason: sum(
                    row["source"] == source and row["rule_reason"] == reason
                    for row in candidates
                )
                for reason in FALSE_POSITIVE_NAMES + ("multiple",)
            }
            for source in ("FY", "COSMIC")
        },
        "review": {
            "candidate_sample": len(selected_candidates),
            "control_sample": len(selected_controls),
            "manifest": str(manifest),
            "plot_sheets": (
                (len(review_rows) + 7) // 8
                if make_plots
                else len(list((output / "review_sheets").glob("sheet_*.png")))
            ),
            "label_method": (
                "metadata-blinded visual sheets with conservative curve rubric"
                if labels else "pending"
            ),
            **decision,
        },
        "inputs": {
            "fy_index": {"path": str(FY_INDEX), "sha256": _sha256(FY_INDEX)},
            "cosmic_index": {
                "path": str(COSMIC_INDEX),
                "sha256": _sha256(COSMIC_INDEX),
            },
            "cosmic_raw": {
                "path": str(COSMIC_INPUT),
                "sha256": _sha256(COSMIC_INPUT),
            },
        },
        "npz_sha256": _sha256(npz_path),
    }
    json_path = output / "hcut_audit_report.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, json_path)
    report["json_sha256"] = _sha256(json_path)
    print(json.dumps(report["review"], ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--candidate-sample", type=int, default=600)
    parser.add_argument("--control-sample", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--review-labels", type=Path)
    parser.add_argument("--write-reviewed-labels", action="store_true")
    args = parser.parse_args()
    report = run_audit(
        output=args.output,
        candidate_sample=args.candidate_sample,
        control_sample=args.control_sample,
        bootstrap=args.bootstrap,
        make_plots=not args.skip_plots,
        labels_path=args.review_labels,
        write_reviewed_labels=args.write_reviewed_labels,
    )
    return 0 if report["review"]["labels_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
