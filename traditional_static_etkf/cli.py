"""Prepare and real-data smoke entry points for the traditional static ETKF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .core import (
    ALT_MAX_KM,
    ALT_MIN_KM,
    ALT_STEP_KM,
    DEFAULT_VARIANCE,
    GRID,
    N_MEMBERS,
    R_COSMIC,
    R_FY,
    TIME_LOCALIZATION_HOURS,
    VERTICAL_LENGTH_KM,
    HORIZONTAL_LENGTH_KM,
    analyze_query,
    cycle_time_hours,
    etkf_update,
    generate_static_ensemble,
    iri_at_grid_stencil,
    load_iri_proxy,
)


def load_manifest(path: os.PathLike | str) -> Dict:
    """Read the v14 manifest without performing SHA/hash checks."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = dict(raw.get("config", raw))
    # These are hard contracts of this independent baseline and must be set
    # before constructing either neighborhood index.
    cfg["alt_range"] = (120.0, 500.0)
    cfg["observation_alt_range"] = (200.0, 500.0)
    cfg["physical_localization_space_km"] = 1800.0
    cfg["physical_localization_time_hours"] = 1.5
    cfg["fy_nb_n_alt"] = 8
    cfg["cosmic_nb_n_alt"] = 8
    cfg["neighbor_directory_semantics"] = "token_exact_positive_support_v1"
    cfg["strict_preload_token_only"] = False
    return cfg


def forced_index_config(config: Dict, allowed_profile_ids=None) -> Dict:
    """Return a copy suitable for FY/COSMIC index construction."""
    out = dict(config)
    out["alt_range"] = (120.0, 500.0)
    out["observation_alt_range"] = (200.0, 500.0)
    out["physical_localization_space_km"] = 1800.0
    out["physical_localization_time_hours"] = 1.5
    out["fy_nb_n_alt"] = out["cosmic_nb_n_alt"] = 8
    out["neighbor_directory_semantics"] = "token_exact_positive_support_v1"
    if allowed_profile_ids is not None:
        out["strict_preload_allowed_profile_ids"] = np.asarray(
            allowed_profile_ids, dtype=np.int64)
    return out


def _resolve(path_value, base: Path) -> Path:
    p = Path(str(path_value))
    return p if p.is_absolute() else (base / p).resolve()


def _date_days(manifest_path: Path) -> Dict[str, List[int]]:
    cfg_path = manifest_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    split = raw.get("partitions", raw)
    return {k: [int(x) for x in split.get(k, [])]
            for k in ("train", "development", "locked_test")}


def _split_path(config: Dict, manifest_path: Path) -> Path:
    p = config.get("date_split_manifest")
    if not p:
        raise ValueError("manifest does not specify date_split_manifest")
    return _resolve(p, manifest_path.parent)


def _load_source_index(source: str, config: Dict, allowed_profile_ids=None):
    from inr_modules.data_managers.FY_dataloader import (
        COSMICNeighborhoodIndex,
        FYNeighborhoodIndex,
    )
    cfg = forced_index_config(config, allowed_profile_ids)
    if allowed_profile_ids is not None:
        cfg["strict_preload_token_only"] = True
    if source == "FY":
        return FYNeighborhoodIndex(config["fy_path"], cfg)
    if source == "COSMIC":
        return COSMICNeighborhoodIndex(config["cosmic_path"], cfg)
    raise ValueError(f"unknown source {source}")


def _development_ids_from_profile_index(source: str, config: Dict,
                                        days: Iterable[int]) -> np.ndarray:
    """Get the development allowlist before constructing a strict index."""
    path_key = "fy_profile_index_path" if source == "FY" else "cosmic_profile_index_path"
    with np.load(config[path_key], allow_pickle=False) as idx:
        ids = np.asarray(idx["profile_id"], dtype=np.int64)
        passed = np.asarray(idx["pass_profile"], dtype=bool)
        times = np.asarray(idx["representative_time"], dtype=np.float64)
    days = set(int(x) for x in days)
    finite = np.isfinite(times)
    day_code = np.full(times.shape, -1, dtype=np.int64)
    day_code[finite] = np.floor(times[finite] / 24.0).astype(np.int64)
    return np.sort(np.unique(ids[passed & finite & np.isin(day_code, list(days))]))


def _development_ids(index, days: Iterable[int]) -> np.ndarray:
    days = set(int(x) for x in days)
    ids = np.asarray(getattr(index, "prof_ids", []), dtype=np.int64)
    meta = np.asarray(getattr(index, "prof_meta", []))
    if len(ids) == 0 or meta.ndim != 2 or meta.shape[1] < 3:
        return np.zeros(0, dtype=np.int64)
    mask = np.isfinite(meta[:, 2]) & np.isin(np.floor(meta[:, 2] / 24.0).astype(int), list(days))
    return np.sort(np.unique(ids[mask]))


def _sample_rows(raw: np.ndarray, max_rows: int) -> np.ndarray:
    if len(raw) <= max_rows:
        return np.asarray(raw)
    return np.asarray(raw[np.linspace(0, len(raw) - 1, max_rows, dtype=np.int64)])


def _source_variance(raw_path: str, r_value: float, train_days: Iterable[int], max_profiles: int) -> np.ndarray:
    """Estimate a train-only vertical variance with deterministic bounded I/O.

    The profile-indexed production files are large.  This prepare-stage
    estimate uses an evenly spaced train-date row sample; the R subtraction is
    explicit and the resulting variance is only a static amplitude prior.
    """
    raw = np.load(raw_path, mmap_mode="r")
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise ValueError(f"{raw_path} is not a physical observation table")
    train = set(int(x) for x in train_days)
    # A bounded first pass avoids allocating a full boolean mask for a month.
    step = max(1, len(raw) // max(1, max_profiles * 64))
    rows = np.asarray(raw[::step])
    day = np.floor(rows[:, 3] / 24.0).astype(int)
    rows = rows[np.isin(day, list(train))]
    rows = _sample_rows(rows, max_profiles * 64)
    out = np.full(len(GRID.alt_km), np.nan, dtype=np.float64)
    for j, alt in enumerate(GRID.alt_km):
        mask = np.isfinite(rows[:, 2]) & np.isfinite(rows[:, 4]) & (np.abs(rows[:, 2] - alt) <= 10.0)
        vals = rows[mask, 4].astype(np.float64)
        if len(vals) >= 2:
            out[j] = max(float(np.var(vals, ddof=1)) - r_value, DEFAULT_VARIANCE)
    # Empty layers inherit the closest valid layer; 120--180 km explicitly
    # inherit 200 km, because observations begin at 200 km.
    valid_layers = np.flatnonzero(np.isfinite(out))
    if len(valid_layers) == 0:
        out.fill(DEFAULT_VARIANCE)
    else:
        for j in range(len(out)):
            if not np.isfinite(out[j]):
                out[j] = out[valid_layers[np.argmin(np.abs(valid_layers - j))]]
    out[:5] = out[4]
    return out


def _source_innovation_variance(raw_path: str, r_value: float,
                                train_days: Iterable[int], max_profiles: int,
                                proxy, device: str = "cpu",
                                profile_index_path: Optional[str] = None):
    """Train-only profile-balanced variance of ``y-H(RAW IRI)`` minus R.

    A profile contributes at most one mean residual per 20-km layer.  This is
    deliberately profile-level weighting: dense or long profiles cannot
    dominate the static amplitude prior.  ``max_profiles=0`` means all
    eligible train profiles; positive values are deterministic smoke limits.
    """
    raw = np.load(raw_path, mmap_mode="r")
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise ValueError(f"{raw_path} is not a physical observation table")
    if not profile_index_path:
        raise ValueError("profile_index_path is required for profile-balanced prepare")
    train = set(int(x) for x in train_days)
    with np.load(profile_index_path, allow_pickle=False) as idx:
        passed = np.asarray(idx["pass_profile"], dtype=bool)
        starts = np.asarray(idx["output_start"], dtype=np.int64)
        ends = np.asarray(idx["output_end"], dtype=np.int64)
        rep_time = np.asarray(idx["representative_time"], dtype=np.float64)
    finite_time = np.isfinite(rep_time)
    day_code = np.full(rep_time.shape, -1, dtype=np.int64)
    day_code[finite_time] = np.floor(rep_time[finite_time] / 24.0).astype(np.int64)
    eligible = np.flatnonzero(
        passed & finite_time & np.isin(day_code, list(train)))
    if max_profiles > 0 and len(eligible) > max_profiles:
        selected = eligible[np.linspace(
            0, len(eligible) - 1, max_profiles, dtype=np.int64)]
    else:
        selected = eligible
    profile_values = np.full((len(selected), len(GRID.alt_km)), np.nan,
                             dtype=np.float64)
    raw = np.load(raw_path, mmap_mode="r")
    for batch_start in range(0, len(selected), 512):
        batch_pos = selected[batch_start:batch_start + 512]
        rows, owners = [], []
        for local, pos in enumerate(batch_pos):
            lo = max(0, int(starts[pos]))
            hi = min(len(raw), int(ends[pos]))
            block = np.asarray(raw[lo:hi])
            if block.ndim != 2 or block.shape[1] < 5:
                continue
            block = block[np.isfinite(block[:, :5]).all(axis=1)]
            block = block[(block[:, 2] >= 200.0) & (block[:, 2] <= 500.0)]
            if len(block):
                take = np.linspace(0, len(block) - 1,
                                   min(8, len(block)), dtype=np.int64)
                rows.append(block[take, :5])
                owners.extend([local] * len(take))
        if not rows:
            continue
        rows = np.concatenate(rows, axis=0)
        owners = np.asarray(owners, dtype=np.int64)
        background = iri_at_grid_stencil(
            proxy, rows[:, 0], rows[:, 1], rows[:, 2],
            cycle_time_hours(rows[:, 3]), device=device)
        residual = rows[:, 4].astype(np.float64) - background
        layer = np.rint((rows[:, 2] - ALT_MIN_KM) / ALT_STEP_KM).astype(np.int64)
        layer_ok = (layer >= 0) & (layer < len(GRID.alt_km))
        for local in np.unique(owners[layer_ok]):
            selected_layers = layer_ok & (owners == local)
            for level in np.unique(layer[selected_layers]):
                values = residual[selected_layers & (layer == level)]
                if len(values):
                    profile_values[batch_start + local, level] = values.mean()
    counts = np.isfinite(profile_values).sum(axis=0).astype(np.int64)
    out = np.full(len(GRID.alt_km), np.nan, dtype=np.float64)
    for level in range(len(GRID.alt_km)):
        values = profile_values[:, level]
        values = values[np.isfinite(values)]
        if len(values) >= 2:
            out[level] = max(float(np.var(values, ddof=1)) - r_value,
                             DEFAULT_VARIANCE)
    return out, counts


def prepare(args) -> Dict:
    manifest_path = Path(args.manifest).resolve()
    config = load_manifest(manifest_path)
    split_path = _split_path(config, manifest_path)
    days = _date_days(split_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy = load_iri_proxy(config["iri_proxy_path"], device=args.device)
    var_fy, count_fy = _source_innovation_variance(
        config["fy_path"], R_FY, days["train"], args.max_profiles_per_source,
        proxy, device=args.device, profile_index_path=config.get("fy_profile_index_path"))
    var_cosmic, count_cosmic = _source_innovation_variance(
        config["cosmic_path"], R_COSMIC, days["train"], args.max_profiles_per_source,
        proxy, device=args.device, profile_index_path=config.get("cosmic_profile_index_path"))
    # Combine source-specific R-corrected estimates by effective profile count.
    count_total = count_fy + count_cosmic
    variance = np.full(len(GRID.alt_km), DEFAULT_VARIANCE, dtype=np.float64)
    valid = count_total > 0
    variance[valid] = (
        np.nan_to_num(var_fy[valid], nan=DEFAULT_VARIANCE) * count_fy[valid]
        + np.nan_to_num(var_cosmic[valid], nan=DEFAULT_VARIANCE) * count_cosmic[valid]
    ) / count_total[valid]
    variance = np.maximum(variance, DEFAULT_VARIANCE)
    # 120--180 km have no direct RO observations; inherit the 200-km prior.
    variance[:5] = variance[4]
    std = np.sqrt(variance)
    anomalies = generate_static_ensemble(std, seed=42)
    anomaly_path = output_dir / "static_anomalies.npy"
    np.save(anomaly_path, anomalies)
    metadata = {
        "semantics": "traditional_static_local_etkf",
        "grid_shape": list(GRID.shape),
        "grid_altitude_km": [float(x) for x in GRID.alt_km],
        "state_alt_range_km": [120.0, 500.0],
        "observation_alt_range_km": [200.0, 500.0],
        "n_members": N_MEMBERS,
        "seed": 42,
        "horizontal_length_km": HORIZONTAL_LENGTH_KM,
        "vertical_length_km": VERTICAL_LENGTH_KM,
        "variance_train_only": True,
        "train_days": days["train"],
        "max_profiles_per_source": int(args.max_profiles_per_source),
        "profile_count_by_source": {
            "FY": count_fy.tolist(), "COSMIC": count_cosmic.tolist()},
        "variance_by_altitude_dex2": variance.tolist(),
        "R": {"FY": R_FY, "COSMIC": R_COSMIC},
        "observation_altitude_forced": True,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _payload_for(index, coords, source: str, allowed_ids: np.ndarray, target_id: int):
    # Index APIs require an exclusion value for every query row.
    exclude = np.full(len(coords), int(target_id), dtype=np.int64)
    payload = index.query_observation_batch(
        np.asarray(coords, dtype=np.float32),
        exclude_profile_ids=exclude,
        allowed_profile_ids=np.asarray(allowed_ids, dtype=np.int64),
    )
    p = dict(payload)
    p["source_name"] = source
    return p


def _merge_payloads(payloads: List[Dict]) -> Dict[str, np.ndarray]:
    if not payloads:
        return {"coords": np.zeros((0, 4), dtype=np.float32), "value": np.zeros(0),
                "source": np.zeros(0, dtype=np.int8), "localization_weight": np.zeros(0),
                "valid_mask": np.zeros(0, dtype=bool)}
    keys = ("coords", "value", "source", "localization_weight", "valid_mask",
            "profile_id", "token_id", "query_index")
    result = {}
    for key in keys:
        arrays = [np.asarray(p[key]) for p in payloads if key in p]
        if arrays:
            result[key] = np.concatenate(arrays, axis=0)
    # All smoke calls have one query.  Row pointers are not needed after merge.
    return result


def smoke(args) -> Dict:
    manifest_path = Path(args.manifest).resolve()
    config = load_manifest(manifest_path)
    split_path = _split_path(config, manifest_path)
    days = _date_days(split_path)
    output_dir = Path(args.output_dir).resolve()
    anomaly_path = output_dir / "static_anomalies.npy"
    field = np.load(anomaly_path, mmap_mode="r")
    if field.shape != GRID.shape + (N_MEMBERS,):
        raise ValueError(f"static anomaly shape mismatch: {field.shape}")

    # Loading the frozen proxy is part of the smoke contract even when no
    # observation survives the local window.
    proxy = load_iri_proxy(config["iri_proxy_path"], device=args.device)
    fy_allowed = _development_ids_from_profile_index("FY", config, days["development"])
    cosmic_allowed = _development_ids_from_profile_index("COSMIC", config, days["development"])
    fy_index = _load_source_index("FY", config, fy_allowed if len(fy_allowed) else None)
    cosmic_index = _load_source_index("COSMIC", config, cosmic_allowed if len(cosmic_allowed) else None)
    ids = {
        "FY": _development_ids(fy_index, days["development"]),
        "COSMIC": _development_ids(cosmic_index, days["development"]),
    }
    if args.source:
        source = args.source.upper()
        if source not in ids:
            raise ValueError("source must be FY or COSMIC")
    else:
        source = "FY" if len(ids["FY"]) else "COSMIC"
    if args.profile_id is None:
        if len(ids[source]) == 0:
            raise RuntimeError(f"no development {source} profile found")
        target_id = int(np.sort(ids[source])[0])
    else:
        target_id = int(args.profile_id)

    index = fy_index if source == "FY" else cosmic_index
    all_ids = ids[source]
    if target_id not in set(int(x) for x in all_ids):
        raise ValueError(f"profile {target_id} is not in the development-only {source} allowlist")
    meta_ids = np.asarray(index.prof_ids, dtype=np.int64)
    where = np.flatnonzero(meta_ids == target_id)
    if len(where) == 0:
        raise ValueError(f"profile {target_id} is not present in {source} development index")
    lat, lon, tim = np.asarray(index.prof_meta[where[0]], dtype=np.float64)
    tc = float(cycle_time_hours(tim))
    query = np.array([[lat, lon, 300.0, tc]], dtype=np.float32)
    # Source-specific LOO: only the target source receives the exclusion.
    payloads = []
    for name, idx in (("FY", fy_index), ("COSMIC", cosmic_index)):
        target_exclusion = target_id if name == source else -1
        source_payload = _payload_for(idx, query, name, ids[name], target_exclusion)
        if name == source and target_id in np.asarray(source_payload.get("profile_id", []), dtype=np.int64):
            raise AssertionError(f"target {name} profile was not excluded from its own observations")
        payloads.append(source_payload)
    payload = _merge_payloads(payloads)
    coords = np.asarray(payload.get("coords", np.zeros((0, 4))), dtype=np.float64)
    valid = np.asarray(payload.get("valid_mask", np.zeros(len(coords), dtype=bool)), dtype=bool)
    if len(coords):
        if not np.isfinite(coords).all() or np.any((coords[:, 2] < 200.0) | (coords[:, 2] > 500.0)):
            raise AssertionError("smoke payload contains an observation outside 200--500 km")
    loc = np.asarray(payload.get("localization_weight", np.zeros(len(coords))), dtype=np.float64)
    src = np.asarray(payload.get("source", np.zeros(len(coords), dtype=np.int8)), dtype=np.int8)
    r = np.where(src == 0, R_FY, R_COSMIC)
    precision = np.where(valid, loc / r, 0.0)
    precision[~np.isfinite(precision) | (precision <= 0.0)] = 0.0

    # Query target level and same-location low/high levels in one chunk.
    query_levels = np.array([120.0, 180.0, 200.0, 300.0, 500.0], dtype=np.float64)
    qcoords = np.column_stack([np.full(5, lat), np.full(5, lon), query_levels, np.full(5, tc)])
    q_idx, q_w = GRID.stencil(qcoords[:, 0], qcoords[:, 1], qcoords[:, 2])
    Xq = np.sum(np.asarray(field).reshape(-1, N_MEMBERS)[q_idx] * q_w[..., None], axis=1)
    xb = iri_at_grid_stencil(proxy, qcoords[:, 0], qcoords[:, 1], qcoords[:, 2], np.full(5, tc), device=args.device)
    if len(coords):
        obs_background = iri_at_grid_stencil(proxy, coords[:, 0], coords[:, 1], coords[:, 2], np.full(len(coords), tc), device=args.device)
        # Both the query state and observation anomalies are interpolated from
        # the same mmap-backed grid field; no continuous-coordinate B/H is used.
        oidx, ow = GRID.stencil(coords[:, 0], coords[:, 1], coords[:, 2])
        Y = np.sum(np.asarray(field).reshape(-1, N_MEMBERS)[oidx] * ow[..., None], axis=1)
        innovation = np.asarray(payload["value"], dtype=np.float64) - obs_background
        result = etkf_update(xb, Xq, Y, innovation, precision)
    else:
        from .core import ETKFResult
        result = ETKFResult(xb, Xq, np.zeros(N_MEMBERS), np.eye(N_MEMBERS))
    output = {
        "source": source,
        "profile_id": target_id,
        "cycle_time_hours": tc,
        "query_levels_km": query_levels.tolist(),
        "background": np.asarray(xb).tolist(),
        "analysis": np.asarray(result.analysis).tolist(),
        "increment": (np.asarray(result.analysis) - np.asarray(xb)).tolist(),
        "analysis_spread": np.asarray(result.analysis_anomalies).std(axis=1, ddof=1).tolist(),
        "observation_count": int(len(coords)),
        "positive_precision_count": int(np.count_nonzero(precision > 0.0)),
        "precision": precision.tolist(),
        "precision_sum": float(precision.sum()),
        "observation_counts_by_source": {
            "FY": int(np.count_nonzero(src == 0)),
            "COSMIC": int(np.count_nonzero(src == 1)),
        },
        "finite": bool(np.isfinite(result.analysis).all() and np.isfinite(result.analysis_anomalies).all()),
        "loo_source": source,
        "loo_profile_id": target_id,
        "loo_target_present_after_exclusion": False,
        "observation_altitude_range_km": [float(coords[:, 2].min()), float(coords[:, 2].max())] if len(coords) else None,
    }
    with open(output_dir / "smoke_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("prepare", prepare), ("smoke", smoke)):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True, help="M2-W v14 run_manifest.json")
        p.add_argument("--output-dir", required=True)
        p.set_defaults(func=fn)
        if name == "prepare":
            p.add_argument("--max-profiles-per-source", type=int, default=0,
                            help="0=all train profiles; positive value limits smoke")
            p.add_argument("--device", default="cpu")
        else:
            p.add_argument("--device", default="cpu")
            p.add_argument("--source", choices=("FY", "COSMIC", "fy", "cosmic"))
            p.add_argument("--profile-id", type=int)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
