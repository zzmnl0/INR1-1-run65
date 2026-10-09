"""Small, strict helpers for the A1-S M00 residual experiment.

The module deliberately does not touch the ETKF state.  It only validates the
20/5/5 date contract, builds Train-only empirical reliability tables, and
replays the existing 3-D observation localization for diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from typing import Any, Mapping

import numpy as np

from .m00_observation import localization_weights


DATE_SEMANTICS = "a1s_20_5_5_train_only_v1"
RELIABILITY_SEMANTICS = "a1s_m00_qd_residual_empirical_reliability_v1"
LOCALIZATION_SEMANTICS = "a1s_m00_train_loo_3d_localization_replay_v1"
QD_STRATUM_SEMANTICS = "qd_abs_sin_lat_025_050_v1"


def validate_a1s_manifest(manifest: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the immutable A1-S 20/5/5 date partition."""
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        with open(os.fspath(manifest), encoding="utf-8") as stream:
            payload = json.load(stream)
    partitions = payload.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("A1-S date manifest lacks partitions")
    train = np.asarray(partitions.get("train", ()), dtype=np.int64).reshape(-1)
    development = np.asarray(partitions.get("development", ()), dtype=np.int64).reshape(-1)
    locked = np.asarray(partitions.get("locked_test", ()), dtype=np.int64).reshape(-1)
    if (train.size, development.size, locked.size) != (20, 5, 5):
        raise ValueError("A1-S M00 calibration requires 20 Train, 5 Development, 5 Locked-test dates")
    if any(np.unique(part).size != part.size for part in (train, development, locked)):
        raise ValueError("A1-S date partitions must be unique")
    if np.intersect1d(train, development).size or np.intersect1d(train, locked).size or np.intersect1d(development, locked).size:
        raise ValueError("A1-S date partitions overlap")
    all_days = np.concatenate((train, development, locked))
    if np.unique(all_days).size != 30 or np.any(all_days < 0) or np.any(all_days >= 30):
        raise ValueError("A1-S date partitions must cover relative days 0..29 exactly")
    return {
        "semantics": DATE_SEMANTICS,
        "train": train.tolist(),
        "development": development.tolist(),
        "locked_test": locked.tolist(),
        "locked_test_used": False,
    }


def _magnetic_stratum(magnetic_latitude_deg: np.ndarray) -> np.ndarray:
    lat = np.asarray(magnetic_latitude_deg, dtype=np.float64).reshape(-1)
    if not np.isfinite(lat).all() or np.any(np.abs(lat) > 90.0):
        raise ValueError("magnetic latitude must be finite and lie in [-90, 90]")
    value = np.abs(np.sin(np.deg2rad(lat)))
    return np.where(value < 0.25, "low", np.where(value < 0.5, "mid", "high"))


def _height_band(altitude_km: np.ndarray) -> np.ndarray:
    alt = np.asarray(altitude_km, dtype=np.float64).reshape(-1)
    if not np.isfinite(alt).all():
        raise ValueError("altitude contains non-finite values")
    # Keep keys identical to the existing M00 observation strata contract.
    return np.where(alt < 200.0, "120-200",
                    np.where(alt < 300.0, "200-300",
                             np.where(alt <= 500.0, "300-500", "outside")))


def _day_night(coords: np.ndarray) -> np.ndarray:
    local_hour = np.mod(np.asarray(coords, dtype=np.float64)[:, 1] / 15.0 + np.asarray(coords, dtype=np.float64)[:, 3], 24.0)
    return np.where((local_hour >= 6.0) & (local_hour < 18.0), "day", "night")


def _profile_days(coords: np.ndarray, profile_ids: np.ndarray,
                  source: np.ndarray | None = None) -> dict[tuple[str, int], set[int]]:
    out: dict[tuple[str, int], set[int]] = defaultdict(set)
    profile_ids = np.asarray(profile_ids, dtype=np.int64)
    sources = (np.asarray(source).reshape(-1).astype(str)
               if source is not None else np.full(profile_ids.size, "", dtype=str))
    for src, pid, hour in zip(sources, profile_ids,
                              np.asarray(coords, dtype=np.float64)[:, 3]):
        if not math.isfinite(float(hour)):
            raise ValueError("relative UTC time contains non-finite values")
        out[(str(src), int(pid))].add(int(math.floor(float(hour) / 24.0)))
    return out


@dataclass(frozen=True)
class ReliabilityResult:
    payload: dict[str, Any]
    profile_weights: np.ndarray


def _migration_labels(latitude_deg: np.ndarray) -> np.ndarray:
    values = np.abs(np.sin(np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))))
    return np.where(values < 0.25, "low", np.where(values < 0.5, "mid", "high"))


def _migration_audit(qd_latitude_deg: np.ndarray, aacgm_latitude_deg: np.ndarray,
                     profile_ids: np.ndarray, source: np.ndarray,
                     coords: np.ndarray) -> dict[str, Any]:
    """Audit QD/AACGM migration with row and fractional-profile summaries.

    Each source/profile/height group has unit mass in the profile table and
    distributes it across its observed point-level migration cells. Median
    magnetic latitude is intentionally not used: a profile can cross a stratum.
    """
    qd = np.asarray(qd_latitude_deg, dtype=np.float64).reshape(-1)
    aa = np.asarray(aacgm_latitude_deg, dtype=np.float64).reshape(-1)
    pids = np.asarray(profile_ids, dtype=np.int64).reshape(-1)
    sources = np.asarray(source).reshape(-1).astype(str)
    coords = np.asarray(coords, dtype=np.float64)
    if not (qd.size == aa.size == pids.size == sources.size == coords.shape[0]):
        raise ValueError("QD/AACGM migration arrays are not aligned")
    common = (np.isfinite(qd) & np.isfinite(aa)
              & (np.abs(qd) <= 90.0) & (np.abs(aa) <= 90.0))
    finite_qd = np.isfinite(qd) & (np.abs(qd) <= 90.0)
    finite_aacgm = np.isfinite(aa) & (np.abs(aa) <= 90.0)
    if not np.any(common):
        raise ValueError("QD/AACGM migration has no common finite rows")

    def _cell(q_value: float, a_value: float) -> str:
        return f"{_migration_labels(np.asarray([q_value]))[0]}|{_migration_labels(np.asarray([a_value]))[0]}"

    row_matrix: dict[str, int] = defaultdict(int)
    row_by_source_height: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    grouped_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in np.flatnonzero(common):
        source_name = str(sources[index])
        height = str(_height_band(np.asarray([coords[index, 2]]))[0])
        cell = _cell(float(qd[index]), float(aa[index]))
        row_matrix[cell] += 1
        row_by_source_height[source_name][height][cell] += 1
        grouped_rows[(source_name, height)].append(int(index))

    # Profile-equal matrix: unit mass per profile within each height band,
    # split fractionally over actual point-level migration cells.
    profile_matrix: dict[str, float] = defaultdict(float)
    profile_by_source_height: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    profile_groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for (source_name, height), indices in grouped_rows.items():
        for index in indices:
            profile_groups[(source_name, height, int(pids[index]))].append(index)
    for (source_name, height, _profile_id), indices in profile_groups.items():
        cells: dict[str, int] = defaultdict(int)
        for index in indices:
            cells[_cell(float(qd[index]), float(aa[index]))] += 1
        total = float(len(indices))
        for cell, count in cells.items():
            contribution = float(count) / total
            profile_matrix[cell] += contribution
            profile_by_source_height[source_name][f"{height}|{cell}"] += contribution

    def rates(matrix: Mapping[str, float | int], *, profile_level: bool = False) -> dict[str, float]:
        high_aacgm = sum(float(v) for k, v in matrix.items() if k.endswith("|high"))
        high_qd = sum(float(v) for k, v in matrix.items() if k.startswith("high|"))
        aacgm_high_to_nonhigh = sum(float(v) for k, v in matrix.items()
                                    if k.endswith("|high") and not k.startswith("high|"))
        qd_high_from_nonhigh = sum(float(v) for k, v in matrix.items()
                                   if k.startswith("high|") and not k.endswith("|high"))
        return {
            "aacgm_high_to_nonhigh_fraction": aacgm_high_to_nonhigh / high_aacgm if high_aacgm else 0.0,
            "qd_high_from_nonhigh_fraction": qd_high_from_nonhigh / high_qd if high_qd else 0.0,
            "profile_equal": bool(profile_level),
        }

    by_source_height_rates = {
        source_name: {
            height: rates(cells)
            for height, cells in heights.items()
        }
        for source_name, heights in row_by_source_height.items()
    }
    row_rates = rates(row_matrix)
    profile_rates = rates(profile_matrix, profile_level=True)
    qd_sin = np.abs(np.sin(np.deg2rad(qd[common])))
    aacgm_sin = np.abs(np.sin(np.deg2rad(aa[common])))
    threshold_near = np.minimum(
        np.minimum(np.abs(qd_sin - 0.25), np.abs(qd_sin - 0.5)),
        np.minimum(np.abs(aacgm_sin - 0.25), np.abs(aacgm_sin - 0.5))) <= 0.02
    difference = qd[common] - aa[common]
    finite_difference = difference[np.isfinite(difference)]
    difference_quantiles = np.quantile(finite_difference, [0.0, 0.5, 0.9, 0.95, 1.0]).tolist()
    risk_exceeded = bool(
        row_rates["aacgm_high_to_nonhigh_fraction"] > 0.05
        or row_rates["qd_high_from_nonhigh_fraction"] > 0.05
        or profile_rates["aacgm_high_to_nonhigh_fraction"] > 0.05
        or profile_rates["qd_high_from_nonhigh_fraction"] > 0.05)
    return {
        "semantics": "qd_aacgm_3x3_migration_v2",
        "common_rows": int(np.count_nonzero(common)),
        "common_profiles": int(np.unique(np.column_stack((sources[common], pids[common])), axis=0).shape[0]),
        "matrix": dict(row_matrix),
        "row_matrix": dict(row_matrix),
        "profile_matrix": dict(profile_matrix),
        "by_source_height": {
            src: {band: dict(cell) for band, cell in bands.items()}
            for src, bands in row_by_source_height.items()
        },
        "profile_by_source_height": {
            src: dict(values) for src, values in profile_by_source_height.items()
        },
        "by_source_height_rates": by_source_height_rates,
        "row_rates": row_rates,
        "profile_rates": profile_rates,
        "aacgm_audit": {
            "total_rows": int(qd.size),
            "qd_finite_rows": int(np.count_nonzero(finite_qd)),
            "aacgm_finite_rows": int(np.count_nonzero(finite_aacgm)),
            "common_valid_rows": int(np.count_nonzero(common)),
            "aacgm_invalid_rows": int(np.count_nonzero(~finite_aacgm)),
            "aacgm_invalid_fraction": float(np.mean(~finite_aacgm)),
        },
        "threshold_near": {
            "sin_margin": 0.02,
            "common_rows": int(np.count_nonzero(threshold_near)),
            "fraction": float(np.mean(threshold_near)),
        },
        "qd_minus_aacgm_latitude_deg_quantiles": difference_quantiles,
        "risk_threshold": 0.05,
        "risk_exceeded": risk_exceeded,
        "review_gate_enabled": False,
        "threshold_note": "QD is accepted by user decision; 5% is descriptive only, not a stop gate",
    }


def build_empirical_reliability(
    residual: Any,
    coords: Any,
    profile_ids: Any,
    source: Any,
    aacgm_latitude_deg: Any | None = None,
    *,
    qd_latitude_deg: Any | None = None,
    aacgm_latitude_audit_deg: Any | None = None,
    magnetic_metadata: Mapping[str, Any] | None = None,
    min_profiles: int = 20,
    min_profile_days: int = 10,
    min_ess: float = 10.0,
    weight_cap: float = 4.0,
) -> ReliabilityResult:
    """Build a non-empty empirical reliability table from Train rows only.

    ``residual`` is model-minus-observation audit data.  Its variance is not
    labelled as a separated instrument/representativeness variance.
    """
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    profile_ids = np.asarray(profile_ids, dtype=np.int64).reshape(-1)
    source = np.asarray(source).reshape(-1).astype(str)
    legacy_lat = (None if aacgm_latitude_deg is None else
                  np.asarray(aacgm_latitude_deg, dtype=np.float64).reshape(-1))
    qd_lat = (None if qd_latitude_deg is None else
              np.asarray(qd_latitude_deg, dtype=np.float64).reshape(-1))
    audit_lat = (None if aacgm_latitude_audit_deg is None else
                 np.asarray(aacgm_latitude_audit_deg, dtype=np.float64).reshape(-1))
    n = residual.size
    magnetic_lat = qd_lat if qd_lat is not None else legacy_lat
    if magnetic_lat is None:
        raise ValueError("QD/magnetic latitude is required")
    arrays = (profile_ids, source, magnetic_lat)
    if audit_lat is not None:
        arrays = arrays + (audit_lat,)
    if coords.ndim != 2 or coords.shape[1] < 4 or any(arr.size != n for arr in arrays):
        raise ValueError("residual, coords, profile_ids, source and magnetic arrays must align")
    if not np.isfinite(residual).all() or not np.isfinite(coords[:, :4]).all():
        raise ValueError("residual/coordinates contain non-finite values")
    strata = _magnetic_stratum(magnetic_lat)
    heights = _height_band(coords[:, 2])
    daynight = _day_night(coords)
    profile_day_map = _profile_days(coords, profile_ids, source)
    by_key: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    key_for_row: list[str] = []
    for i in range(n):
        if heights[i] == "outside":
            key_for_row.append("")
            continue
        key = f"{source[i]}|{heights[i]}|{daynight[i]}|{strata[i]}"
        key_for_row.append(key)
        by_key[key][int(profile_ids[i])].append(float(residual[i]))
    table: dict[str, float] = {}
    residual_mean: dict[str, float] = {}
    support: dict[str, dict[str, float | int]] = {}
    raw_reliability: dict[str, float] = {}
    for key, profiles in by_key.items():
        means = np.asarray([np.mean(values) for values in profiles.values()], dtype=np.float64)
        within = []
        for values in profiles.values():
            values = np.asarray(values, dtype=np.float64)
            if values.size > 1:
                med = np.median(values)
                within.append((1.4826 * np.median(np.abs(values - med))) ** 2)
        source_name = key.split("|", 1)[0]
        profile_day_sets = {
            pid: profile_day_map[(source_name, pid)] for pid in profiles
        }
        profile_day_pairs = {
            (int(pid), int(day))
            for pid, days in profile_day_sets.items() for day in days
        }
        independent_dates = {
            int(day) for days in profile_day_sets.values() for day in days
        }
        # Empirical total residual variance: between-profile plus robust within-profile scatter.
        between = float(np.var(means, ddof=1)) if means.size > 1 else 0.0
        within_var = float(np.mean(within)) if within else 0.0
        total = max(between + within_var, 1e-6)
        ess = float(means.size)
        raw_reliability[key] = 1.0 / total
        residual_mean[key] = float(np.mean(means))
        support[key] = {
            "profiles": int(means.size),
            "profile_days": int(len(profile_day_pairs)),
            "independent_dates": int(len(independent_dates)),
            "ess": ess,
            "weighted_ess": ess,
            "ess_method": "profile_equal_within_stratum_v1",
            "between_profile_variance": between,
            "within_profile_mad_variance": within_var,
            "total_empirical_variance": total,
            "reliability_clipped": False,
        }
        table[key] = total
    unsupported = [key for key, item in support.items() if item["profiles"] < min_profiles or item["profile_days"] < min_profile_days or item["ess"] < min_ess]
    if unsupported:
        raise ValueError(f"empirical reliability support below strict threshold: {unsupported[:3]}")
    if not table:
        raise ValueError("empirical reliability table is empty")
    weights_by_source: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for key, value in raw_reliability.items():
        weights_by_source[key.split("|", 1)[0]].append((value, int(support[key]["profiles"])))
    normalized: dict[str, float] = {}
    for key, value in raw_reliability.items():
        source_values = weights_by_source[key.split("|", 1)[0]]
        total_profiles = sum(count for _, count in source_values)
        source_mean = (sum(weight * count for weight, count in source_values)
                       / max(total_profiles, 1))
        normalized[key] = float(np.clip(value / max(source_mean, 1e-12), 1.0 / weight_cap, weight_cap))
        support[key]["reliability_clipped"] = bool(normalized[key] != value / max(source_mean, 1e-12))
    profile_weights = np.ones(n, dtype=np.float64)
    for i, key in enumerate(key_for_row):
        if key:
            profile_weights[i] = normalized[key]
    representativeness = {
        key: float(item["within_profile_mad_variance"])
        for key, item in support.items()
    }
    payload = {
        "semantics": RELIABILITY_SEMANTICS,
        "train_only": True,
        "locked_test": "excluded",
        "magnetic_coordinate": {
            "semantics": "qd_v1" if qd_lat is not None else "aacgm_v2",
            "stratum_semantics": QD_STRATUM_SEMANTICS if qd_lat is not None else "aacgm_abs_sin_lat_025_050_v2",
            "fallback": False,
        },
        "residual_reference": "RO_logNe_minus_M00_A1S_epoch3_logNe",
        "variance_interpretation": "empirical_total_residual_reliability_not_separated_R_obs_R_rep",
        # Keep calibration bias neutral: residual means are an audit statistic,
        # not a label correction.  Training consumes the separate reliability
        # table and cannot silently subtract model residuals from RO targets.
        "bias": {key: 0.0 for key in table},
        "residual_mean": residual_mean,
        "variance": table,
        # This is the robust within-profile component of the empirical
        # residual variance.  It is explicitly labelled empirical below; it
        # is not claimed to be a separated instrument-error estimate.
        "representativeness": representativeness,
        "representativeness_support": support,
        "reliability_weight": normalized,
        "support": support,
        "weight_range": [1.0 / weight_cap, weight_cap],
        "localization_contract": {
            "semantics": "m00_3d_localization_v2",
            "spatial_scale_km": 1800.0,
            "vertical_scale_km": 100.0,
            "temporal_scale_hours": 1.5,
            "time_semantics": "relative_utc_hours_v1",
            "applies_to": "diagnostic_only",
        },
    }
    if qd_lat is not None:
        if magnetic_metadata:
            coordinate = payload["magnetic_coordinate"]
            for key in ("qd_converter", "apexpy_version", "reference_time_utc",
                        "actual_height_semantics", "coordinate_input_semantics"):
                if key in magnetic_metadata:
                    coordinate[key] = magnetic_metadata[key]
        payload["qd_coordinate"] = {"semantics": QD_STRATUM_SEMANTICS,
                                     "rows": int(n),
                                     "finite": True}
    if audit_lat is not None:
        payload["aacgm_audit"] = {
            "finite_rows": int(np.isfinite(audit_lat).sum()),
            "invalid_rows": int((~np.isfinite(audit_lat)).sum()),
        }
        payload["qd_aacgm_migration"] = _migration_audit(
            magnetic_lat, audit_lat, profile_ids, source, coords)
    return ReliabilityResult(payload=payload, profile_weights=profile_weights)


def replay_train_loo_localization(
    query_coords: Any,
    reference_coords: Any,
    query_profile_ids: Any,
    reference_profile_ids: Any,
    *,
    spatial_scale_km: float = 1800.0,
    vertical_scale_km: float = 100.0,
    temporal_scale_hours: float = 1.5,
) -> dict[str, Any]:
    """Replay strict 3-D localization while excluding each query profile."""
    q = np.asarray(query_coords, dtype=np.float64)
    r = np.asarray(reference_coords, dtype=np.float64)
    qid = np.asarray(query_profile_ids, dtype=np.int64).reshape(-1)
    rid = np.asarray(reference_profile_ids, dtype=np.int64).reshape(-1)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] < 4 or r.shape[1] < 4 or q.shape[0] != qid.size or r.shape[0] != rid.size:
        raise ValueError("localization coordinates/profile IDs are not aligned")
    if not np.isfinite(q[:, :4]).all() or not np.isfinite(r[:, :4]).all():
        raise ValueError("localization coordinates contain non-finite values")
    counts = []
    precision_before = []
    precision_after = []
    for i, query in enumerate(q[:, :4]):
        keep = rid != qid[i]
        if not np.any(keep):
            counts.append(0)
            precision_before.append(0.0)
            precision_after.append(0.0)
            continue
        weights = localization_weights(
            query[None, :],
            r[keep, :4],
            spatial_scale_km=spatial_scale_km,
            vertical_scale_km=vertical_scale_km,
            temporal_scale_hours=temporal_scale_hours,
            time_semantics="relative_utc_hours_v1",
            strict_3d=True,
        ).reshape(-1)
        positive = weights > 0.0
        counts.append(int(np.count_nonzero(positive)))
        precision_before.append(float(weights.size))
        precision_after.append(float(weights.sum()))
    return {
        "semantics": LOCALIZATION_SEMANTICS,
        "train_only": True,
        "self_profile_excluded": True,
        "scales": {
            "spatial_scale_km": float(spatial_scale_km),
            "vertical_scale_km": float(vertical_scale_km),
            "temporal_scale_hours": float(temporal_scale_hours),
            "time_semantics": "relative_utc_hours_v1",
        },
        "query_count": int(q.shape[0]),
        "reference_count": int(r.shape[0]),
        "effective_neighbor_count": {
            "mean": float(np.mean(counts)) if counts else 0.0,
            "min": int(min(counts)) if counts else 0,
            "max": int(max(counts)) if counts else 0,
        },
        "precision": {
            "mean_reference_rows": float(np.mean(precision_before)) if precision_before else 0.0,
            "mean_localized_mass": float(np.mean(precision_after)) if precision_after else 0.0,
        },
    }


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)
