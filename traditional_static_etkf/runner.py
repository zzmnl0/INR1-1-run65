"""Full-month ISR validation for the independent RAW-IRI static ETKF.

The runner deliberately owns the traditional baseline data flow.  It never
imports M2-W observation-background helpers: the only background passed to an
ETKF system is the frozen RAW IRI proxy evaluated on the fixed grid.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .cli import (_date_days, _load_source_index, _resolve, _split_path,
                  load_manifest)
from .core import (
    ALT_MAX_KM,
    ALT_MIN_KM,
    GRID,
    N_MEMBERS,
    R_BY_SOURCE,
    cycle_time_hours,
    iri_at_grid_stencil,
    iri_query,
    load_iri_proxy,
    solve_ragged_etkf,
)


CANONICAL_DEFAULT = (
    "isr_validation_outputs/"
    "run67-p0a-v14-epoch15-vs-v13-epoch12-isr-qav2-r4-all-alt")
OUTPUT_DEFAULT = "isr_validation_outputs/traditional-static-etkf-raw-iri-full-isr"
DAY_SECONDS = 86400.0


def _station_label(value: str) -> str:
    return "PokerFlat" if value.lower().replace(" ", "") == "pokerflat" else "Jicamarca"


def _date_to_unix(value: str) -> float:
    text = str(value)
    if len(text) == 8 and text.isdigit():
        date = _dt.datetime.strptime(text, "%Y%m%d")
    else:
        date = _dt.datetime.strptime(text, "%Y-%m-%d")
    return date.replace(tzinfo=_dt.timezone.utc).timestamp()


def _key(station: str, date_str: str, timestamp: float, altitude: float) -> str:
    return f"{_station_label(station)}|{date_str}|{int(timestamp)}|{float(altitude):.3f}"


def _profile_ids(config: Mapping, source: str, days: Iterable[int]) -> np.ndarray:
    path_key = "fy_profile_index_path" if source == "FY" else "cosmic_profile_index_path"
    with np.load(config[path_key], allow_pickle=False) as index:
        profile_id = np.asarray(index["profile_id"], dtype=np.int64)
        passed = np.asarray(index["pass_profile"], dtype=bool)
        representative = np.asarray(index["representative_time"], dtype=np.float64)
    days = np.asarray(sorted({int(day) for day in days}), dtype=np.int64)
    finite = np.isfinite(representative)
    date_code = np.full(representative.shape, -1, dtype=np.int64)
    date_code[finite] = np.floor(representative[finite] / 24.0).astype(np.int64)
    return np.sort(np.unique(profile_id[
        passed & finite & np.isin(date_code, days)]))


def _load_online_indices(config: Mapping, days: Mapping[str, Sequence[int]]):
    """Load strict train+development token indexes and audit the split."""
    indexes, allowed, audit = {}, {}, {}
    online_days = list(days.get("train", ())) + list(days.get("development", ()))
    locked_days = days.get("locked_test", ())
    for source in ("FY", "COSMIC"):
        train_ids = _profile_ids(config, source, days.get("train", ()))
        development_ids = _profile_ids(config, source, days.get("development", ()))
        locked_ids = _profile_ids(config, source, locked_days)
        union = np.sort(np.unique(np.concatenate([train_ids, development_ids])))
        overlap = np.intersect1d(union, locked_ids)
        if len(overlap):
            raise AssertionError(f"{source} online pool intersects locked_test: {overlap[:5]}")
        allowed[source] = union
        audit[source] = {
            "train_profiles": int(len(train_ids)),
            "development_profiles": int(len(development_ids)),
            "online_profiles": int(len(union)),
            "locked_test_profiles": int(len(locked_ids)),
            "locked_intersection": int(len(overlap)),
        }
        if len(union):
            indexes[source] = _load_source_index(source, config, union)
        else:
            indexes[source] = None
    audit["online_days"] = sorted({int(day) for day in online_days})
    return indexes, allowed, audit


def _record_points(record: Mapping, start_unix: float | None = None) -> Dict[str, np.ndarray]:
    station = _station_label(record["station"])
    date_str = str(record["date_str"])
    alt = np.asarray(record["alt_1d"], dtype=np.float64)
    ts = np.asarray(record["ts_1d"], dtype=np.float64)
    ne = np.asarray(record["ne_2d"], dtype=np.float64)
    valid = np.isfinite(ne) & (ne > 0.0)
    if record.get("geo_lat_2d") is not None:
        lat_grid = np.asarray(record["geo_lat_2d"], dtype=np.float64)
        lon_grid = np.asarray(record["geo_lon_2d"], dtype=np.float64)
    else:
        lat_grid = np.full(ne.shape, float(record["lat"]), dtype=np.float64)
        lon_grid = np.full(ne.shape, float(record["lon"]), dtype=np.float64)
    i_alt, i_time = np.where(valid)
    if len(i_alt) == 0:
        return {name: np.empty(0) for name in
                ("station", "date", "timestamp", "altitude", "lat", "lon",
                 "observation", "key", "unit_id")}
    timestamps = ts[i_time]
    altitudes = alt[i_alt]
    keys = np.asarray([_key(station, date_str, stamp, height)
                       for stamp, height in zip(timestamps, altitudes)])
    with np.errstate(divide="ignore", invalid="ignore"):
        observation = np.log10(ne[i_alt, i_time])
    return {
        "station": np.full(len(i_alt), station),
        "date": np.full(len(i_alt), date_str),
        "timestamp": timestamps.astype(np.int64),
        "relative_hours": ((timestamps - float(start_unix)) / 3600.0
                           if start_unix is not None else timestamps / 3600.0),
        "altitude": altitudes,
        "lat": lat_grid[i_alt, i_time],
        "lon": lon_grid[i_alt, i_time],
        "observation": observation,
        "key": keys,
        "unit_id": timestamps.astype(np.int64),
    }


def _concat_points(records: Sequence[Mapping]) -> Dict[str, np.ndarray]:
    parts = [_record_points(record) for record in records]
    parts = [part for part in parts if len(part["key"])]
    if not parts:
        return {name: np.empty(0) for name in
                ("station", "date", "timestamp", "altitude", "lat", "lon",
                 "observation", "key", "unit_id")}
    return {name: np.concatenate([part[name] for part in parts])
            for name in parts[0]}


def _decode_grid_nodes(flat: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.int64)
    cycles = np.asarray(cycles, dtype=np.float64)
    ai = flat // (len(GRID.lat_deg) * len(GRID.lon_deg))
    rem = flat % (len(GRID.lat_deg) * len(GRID.lon_deg))
    li = rem // len(GRID.lon_deg)
    oi = rem % len(GRID.lon_deg)
    # Pole states are physically unique and are stored at longitude zero.
    oi = np.where((li == 0) | (li == len(GRID.lat_deg) - 1), 0, oi)
    return np.column_stack([
        GRID.lat_deg[li], GRID.lon_deg[oi], GRID.alt_km[ai], cycles])


def _empty_audit() -> Dict[str, int]:
    return {
        "nodes": 0, "observation_edges": 0, "positive_precision_edges": 0,
        "invalid_edges": 0, "nonfinite_edges": 0, "out_of_domain_edges": 0,
        "fallback_nodes": 0, "factorization_failures": 0,
        "rejected_points": 0,
    }


def _analyze_points(points: Mapping[str, np.ndarray], field: np.ndarray,
                    proxy, indexes: Mapping, allowed: Mapping,
                    device: str = "cpu", node_chunk_size: int = 256):
    """Analyze arbitrary ISR points through unique cycle/grid nodes."""
    n_points = len(points["key"])
    result = {
        "iri_grid": np.full(n_points, np.nan, dtype=np.float64),
        "etkf": np.full(n_points, np.nan, dtype=np.float64),
        "spread": np.full(n_points, np.nan, dtype=np.float64),
        "fy_precision": np.zeros(n_points, dtype=np.float64),
        "cosmic_precision": np.zeros(n_points, dtype=np.float64),
        "status": np.full(n_points, "rejected", dtype="U32"),
    }
    audit = _empty_audit()
    if n_points == 0:
        return result, audit
    coords = np.column_stack([
        points["lat"], points["lon"], points["altitude"]]).astype(np.float64)
    time_hours = np.asarray(points.get("relative_hours", points["timestamp"] / 3600.0),
                            dtype=np.float64)
    cycles = np.asarray(cycle_time_hours(time_hours), dtype=np.float64)
    stencil_index, stencil_weight = GRID.stencil(
        coords[:, 0], coords[:, 1], coords[:, 2])
    node_flat = stencil_index.reshape(-1)
    node_cycles = np.repeat(cycles, 8)
    node_key = np.empty(len(node_flat), dtype=[("cycle", "<i8"), ("flat", "<i8")])
    node_key["cycle"] = np.rint(node_cycles * 2.0).astype(np.int64)
    node_key["flat"] = node_flat
    unique_key, inverse = np.unique(node_key, return_inverse=True)
    node_coords = _decode_grid_nodes(
        unique_key["flat"], unique_key["cycle"].astype(np.float64) / 2.0)
    field_flat = np.asarray(field).reshape(-1, N_MEMBERS)
    node_analysis = np.empty(len(node_coords), dtype=np.float64)
    node_background = np.empty(len(node_coords), dtype=np.float64)
    node_anomaly = np.empty((len(node_coords), N_MEMBERS), dtype=np.float64)
    node_fy_precision = np.zeros(len(node_coords), dtype=np.float64)
    node_cosmic_precision = np.zeros(len(node_coords), dtype=np.float64)
    audit["nodes"] = int(len(node_coords))

    for start in range(0, len(node_coords), int(node_chunk_size)):
        stop = min(start + int(node_chunk_size), len(node_coords))
        qcoords = node_coords[start:stop]
        qflat = unique_key["flat"][start:stop]
        qcycles = qcoords[:, 3]
        xb = iri_query(proxy, qcoords[:, 0], qcoords[:, 1], qcoords[:, 2],
                       qcycles, device=device).astype(np.float64)
        Xq = field_flat[qflat].astype(np.float64)
        payloads = []
        for source in ("FY", "COSMIC"):
            index = indexes.get(source)
            if index is None:
                continue
            payload = index.query_observation_batch(
                qcoords.astype(np.float32),
                allowed_profile_ids=np.asarray(allowed[source], dtype=np.int64))
            payload["source_name"] = source
            payloads.append(payload)
        if payloads:
            coords_e = np.concatenate([np.asarray(p["coords"], dtype=np.float64)
                                       for p in payloads])
            values_e = np.concatenate([np.asarray(p["value"], dtype=np.float64)
                                       for p in payloads])
            valid_e = np.concatenate([np.asarray(p["valid_mask"], dtype=bool)
                                      for p in payloads])
            loc_e = np.concatenate([
                np.asarray(p["localization_weight"], dtype=np.float64)
                for p in payloads])
            source_e = np.concatenate([
                np.full(len(p["coords"]), 0 if p["source_name"] == "FY" else 1,
                        dtype=np.int8) for p in payloads])
            qindex_e = np.concatenate([
                np.asarray(p["query_index"], dtype=np.int64)
                for p in payloads])
            profile_e = np.concatenate([
                np.asarray(p.get("profile_id", np.full(len(p["coords"]), -1)),
                           dtype=np.int64) for p in payloads])
            token_e = np.concatenate([
                np.asarray(p.get("token_id", np.arange(len(p["coords"]))),
                           dtype=np.int64) for p in payloads])
        else:
            coords_e = np.empty((0, 4), dtype=np.float64)
            values_e = np.empty(0, dtype=np.float64)
            valid_e = np.empty(0, dtype=bool)
            loc_e = np.empty(0, dtype=np.float64)
            source_e = np.empty(0, dtype=np.int8)
            qindex_e = np.empty(0, dtype=np.int64)
            profile_e = np.empty(0, dtype=np.int64)
            token_e = np.empty(0, dtype=np.int64)
        audit["observation_edges"] += int(len(coords_e))
        invalid = (~valid_e) | ~np.isfinite(values_e) | ~np.isfinite(loc_e) | (loc_e <= 0.0)
        audit["invalid_edges"] += int(invalid.sum())
        audit["nonfinite_edges"] += int((~np.isfinite(values_e) | ~np.isfinite(loc_e)).sum())
        out_of_domain = ((coords_e[:, 2] < 200.0) | (coords_e[:, 2] > 500.0)) \
            if len(coords_e) else np.zeros(0, dtype=bool)
        audit["out_of_domain_edges"] += int(out_of_domain.sum())
        if len(coords_e) and audit["out_of_domain_edges"]:
            raise ValueError("RO payload contains an observation outside 200--500 km")
        if len(coords_e):
            token_key = np.empty(len(coords_e), dtype=[
                ("source", "<i1"), ("profile", "<i8"), ("token", "<i8"),
                ("cycle", "<i8")])
            token_key["source"] = source_e
            token_key["profile"] = profile_e
            token_key["token"] = token_e
            token_key["cycle"] = np.rint(
                qcoords[qindex_e, 3] * 2.0).astype(np.int64)
            _, first, edge_unique = np.unique(
                token_key, return_index=True, return_inverse=True)
            unique_coords = coords_e[first]
            unique_y = GRID.interpolate(
                field, unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2])
            unique_bg = iri_at_grid_stencil(
                proxy, unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2],
                qcoords[qindex_e[first], 3], device=device)
            Y = unique_y[edge_unique]
            innovation = values_e - unique_bg[edge_unique]
        else:
            Y = np.empty((0, N_MEMBERS), dtype=np.float64)
            innovation = np.empty(0, dtype=np.float64)
            edge_unique = np.empty(0, dtype=np.int64)
        precision = np.zeros(len(coords_e), dtype=np.float64)
        for source_code, source in ((0, "FY"), (1, "COSMIC")):
            mask = (source_e == source_code) & ~invalid
            precision[mask] = loc_e[mask] / R_BY_SOURCE[source]
        audit["positive_precision_edges"] += int((precision > 0).sum())
        source_precision = np.zeros((stop - start, 2), dtype=np.float64)
        for source_code in (0, 1):
            mask = (source_e == source_code) & (precision > 0.0)
            if np.any(mask):
                np.add.at(source_precision[:, source_code], qindex_e[mask], precision[mask])
        local = solve_ragged_etkf(
            xb, Xq, qindex_e, Y, innovation, precision)
        node_analysis[start:stop] = local["analysis"]
        node_anomaly[start:stop] = local["analysis_anomalies"]
        node_background[start:stop] = xb
        node_fy_precision[start:stop] = source_precision[:, 0]
        node_cosmic_precision[start:stop] = source_precision[:, 1]
        audit["fallback_nodes"] += int(np.asarray(local["fallback_mask"]).sum())
        audit["factorization_failures"] += int(local.get("factorization_failures", 0))

    node_pos = inverse.reshape(n_points, 8)
    result["iri_grid"] = np.sum(
        node_background[node_pos] * stencil_weight, axis=1)
    result["etkf"] = np.sum(
        node_analysis[node_pos] * stencil_weight, axis=1)
    analysis_anomaly = np.sum(
        node_anomaly[node_pos] * stencil_weight[..., None], axis=1)
    result["spread"] = analysis_anomaly.std(axis=1, ddof=1)
    result["fy_precision"] = np.sum(
        node_fy_precision[node_pos] * stencil_weight, axis=1)
    result["cosmic_precision"] = np.sum(
        node_cosmic_precision[node_pos] * stencil_weight, axis=1)
    result["status"] = np.where(
        (result["fy_precision"] > 0) & (result["cosmic_precision"] > 0), "dual",
        np.where(result["fy_precision"] > 0, "FY-only",
                 np.where(result["cosmic_precision"] > 0, "COSMIC-only", "zero")))
    if not np.isfinite(result["etkf"]).all():
        raise FloatingPointError("non-finite ISR ETKF result")
    return result, audit


def _load_isr_records(stations: Sequence[str], args):
    from isr_evaluation.isr_loader import load_jicamarca, load_poker_flat
    start_unix = _date_to_unix(args.start_date)
    end_unix = _date_to_unix(args.end_date)
    records = []
    if "Jicamarca" in stations:
        records.extend(load_jicamarca(
            args.jicamarca_dir, start_unix, end_unix,
            alt_min=ALT_MIN_KM, alt_max=ALT_MAX_KM,
            err_ratio_max=args.err_ratio_max,
            fail_on_file_error=True))
    if "PokerFlat" in stations:
        records.extend(load_poker_flat(
            args.poker_flat_dir, start_unix, end_unix,
            alt_min=ALT_MIN_KM, alt_max=ALT_MAX_KM,
            err_ratio_max=args.err_ratio_max,
            beam_select=args.beam_select,
            fail_on_file_error=True))
    records.sort(key=lambda row: (_station_label(row["station"]), row["date_str"]))
    if args.max_days > 0:
        dates = sorted({str(row["date_str"]) for row in records})[:args.max_days]
        records = [row for row in records if str(row["date_str"]) in dates]
    return records


def _write_shard(path: Path, points: Mapping[str, np.ndarray], predictions: Mapping[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**points, **predictions}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(tmp, path)


def _process_records(records, output_dir: Path, field, proxy, indexes, allowed,
                     args):
    shard_dir = output_dir / "shards"
    all_audit = []
    for record in records:
        station = _station_label(record["station"])
        date_str = str(record["date_str"])
        shard = shard_dir / station / f"{date_str}.npz"
        if args.resume and shard.exists():
            all_audit.append({"station": station, "date": date_str, "resumed": True})
            continue
        started = time.perf_counter()
        points = _record_points(record, start_unix=_date_to_unix(args.start_date))
        predictions, audit = _analyze_points(
            points, field, proxy, indexes, allowed, device=args.device,
            node_chunk_size=args.node_chunk_size)
        _write_shard(shard, points, predictions)
        audit.update({"station": station, "date": date_str,
                      "wall_seconds": time.perf_counter() - started})
        all_audit.append(audit)
    return all_audit


def _read_shards(output_dir: Path) -> Dict[str, np.ndarray]:
    paths = sorted((output_dir / "shards").glob("*/*.npz"))
    if not paths:
        raise FileNotFoundError("no ISR shards found")
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            arrays.append({key: np.asarray(data[key]) for key in data.files})
    keys = arrays[0].keys()
    return {key: np.concatenate([row[key] for row in arrays]) for key in keys}


def _strict_join_canonical(shards: Mapping[str, np.ndarray], canonical_path: Path,
                           expected_keys: Sequence[str] | None = None):
    with np.load(canonical_path, allow_pickle=False) as canonical:
        required = {key: np.asarray(canonical[key]) for key in canonical.files}
    shard_keys = np.asarray(shards["key"]).astype(str)
    canonical_keys = np.asarray(required["key"]).astype(str)
    if expected_keys is not None:
        expected = np.asarray(expected_keys).astype(str)
        canonical_mask = np.isin(canonical_keys, expected)
        required = {key: value[canonical_mask] for key, value in required.items()}
        canonical_keys = canonical_keys[canonical_mask]
    if len(np.unique(shard_keys)) != len(shard_keys):
        raise ValueError("traditional ISR result contains duplicate keys")
    if len(np.unique(canonical_keys)) != len(canonical_keys):
        raise ValueError("canonical ISR cache contains duplicate keys")
    shard_map = {key: index for index, key in enumerate(shard_keys)}
    missing = [key for key in canonical_keys if key not in shard_map]
    canonical_key_set = set(canonical_keys)
    extra = [key for key in shard_keys if key not in canonical_key_set]
    if missing or extra:
        raise ValueError(f"ISR key mismatch: missing={len(missing)} extra={len(extra)}")
    order = np.asarray([shard_map[key] for key in canonical_keys], dtype=np.int64)
    result = {key: value for key, value in required.items()}
    for key in ("timestamp", "altitude", "lat", "lon", "observation",
                "iri_grid", "etkf", "spread", "fy_precision",
                "cosmic_precision", "status"):
        result[key] = np.asarray(shards[key])[order]
    result["altitude_km"] = result.pop("altitude")
    result["ETKF_log10"] = result.pop("etkf")
    result["IRI_grid_log10"] = result.pop("iri_grid")
    result["ETKF_background_raw_IRI_log10"] = result["IRI_grid_log10"].copy()
    result["ETKF_spread"] = result.pop("spread")
    result["FY_precision"] = result.pop("fy_precision")
    result["COSMIC_precision"] = result.pop("cosmic_precision")
    result["ETKF_status"] = result.pop("status")
    result["raw_iri_continuous_log10"] = result.pop("IRI_log10")
    return result


def _metrics(observation, prediction):
    observation = np.asarray(observation, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.isfinite(observation) & np.isfinite(prediction)
    obs, pred = observation[mask], prediction[mask]
    if len(obs) == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "mae": np.nan,
                "pearson_r": np.nan, "ccc": np.nan}
    error = pred - obs
    bias = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    if len(obs) < 2 or obs.std() < 1e-12 or pred.std() < 1e-12:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(obs, pred)[0, 1])
    var_o, var_p = obs.var(), pred.var()
    cov = np.mean((obs - obs.mean()) * (pred - pred.mean()))
    ccc = float(2.0 * cov / (var_o + var_p + (obs.mean() - pred.mean()) ** 2 + 1e-30))
    return {"n": int(len(obs)), "bias": bias, "rmse": rmse, "mae": mae,
            "pearson_r": pearson, "ccc": ccc}


def _local_time(timestamp, longitude):
    return (np.asarray(timestamp, dtype=np.float64) % DAY_SECONDS) / 3600.0 \
        + np.asarray(longitude, dtype=np.float64) / 15.0


def _bootstrap(obs, candidate, baseline, units):
    from inr_modules.mdia.evaluation_stats import paired_group_bootstrap
    mask = np.isfinite(obs) & np.isfinite(candidate) & np.isfinite(baseline)
    if np.unique(np.asarray(units)[mask]).size < 2:
        return {"decision": "insufficient_units", "sampling_units": 0,
                "replicates": 2000, "seed": 42}
    return paired_group_bootstrap(np.asarray(obs)[mask], np.asarray(candidate)[mask],
                                  np.asarray(baseline)[mask], np.asarray(units)[mask],
                                  replicates=2000, seed=42)


def _peak_cache(cache: Mapping[str, np.ndarray], canonical_peak_path: Path,
                output_dir: Path):
    from isr_evaluation.peak_qa import DEFAULT_PEAK_CONTRACT, results_to_arrays, search_peak_grid
    station = np.asarray(cache["station"]).astype(str)
    timestamp = np.asarray(cache["unit_id"], dtype=np.int64)
    alt = np.asarray(cache["altitude_km"], dtype=np.float64)
    etkf = np.asarray(cache["ETKF_log10"], dtype=np.float64)
    iri_grid = np.asarray(cache["IRI_grid_log10"], dtype=np.float64)
    values = {"ETKF": {}, "IRI_grid": {}}
    for name, array in (("ETKF", etkf), ("IRI_grid", iri_grid)):
        for index in range(len(array)):
            values[name].setdefault((station[index], int(timestamp[index])), []).append(
                (alt[index], array[index]))
    with np.load(canonical_peak_path, allow_pickle=False) as canonical:
        peak = {key: np.asarray(canonical[key]) for key in canonical.files}
    allowed_groups = set(zip(np.asarray(cache["station"]).astype(str),
                             np.asarray(cache["unit_id"], dtype=np.int64)))
    peak_mask = np.asarray([
        (str(station_name), int(unit_id)) in allowed_groups
        for station_name, unit_id in zip(peak["station"], peak["timestamp"])],
        dtype=bool)
    peak = {key: value[peak_mask] for key, value in peak.items()}
    peak_key = {(str(s), int(t)): i for i, (s, t) in enumerate(
        zip(peak["station"], peak["timestamp"]))}
    additions = {}
    for name, groups in values.items():
        nmf2 = np.full(len(peak_key), np.nan)
        hmf2 = np.full(len(peak_key), np.nan)
        status = np.full(len(peak_key), "missing", dtype="U32")
        for key, rows in groups.items():
            if key not in peak_key:
                raise ValueError(f"ETKF peak key not in canonical cache: {key}")
            rows.sort(key=lambda pair: pair[0])
            altitude = np.asarray([row[0] for row in rows])
            grid = np.asarray([[row[1] for row in rows]]).T
            arrays = results_to_arrays(search_peak_grid(
                grid, altitude, DEFAULT_PEAK_CONTRACT))
            index = peak_key[key]
            nmf2[index] = arrays["nmf2_log10"][0]
            hmf2[index] = arrays["hmf2_km"][0]
            status[index] = arrays["status"][0]
        additions[f"{name}_nmf2_log10"] = nmf2
        additions[f"{name}_hmf2_km"] = hmf2
        additions[f"{name}_status"] = status
    peak.update(additions)
    np.savez_compressed(output_dir / "isr_etkf_peak_cache.npz", **peak)
    return peak


def _report(cache: Mapping[str, np.ndarray], peak: Mapping[str, np.ndarray],
            output_dir: Path, audit_rows: Sequence[Mapping], elapsed: float):
    station = np.asarray(cache["station"]).astype(str)
    altitude = np.asarray(cache["altitude_km"], dtype=np.float64)
    timestamp = np.asarray(cache["unit_id"], dtype=np.int64)
    longitude = np.asarray(cache["lon"], dtype=np.float64)
    observation = np.asarray(cache["observation_log10"], dtype=np.float64)
    local_time = _local_time(timestamp, longitude) % 24.0
    day = (local_time >= 6.0) & (local_time < 18.0)
    unit = np.asarray([f"{s}|{int(t)}" for s, t in zip(station, timestamp)])
    models = {
        "RAW_IRI_grid": np.asarray(cache["IRI_grid_log10"]),
        "Static_ETKF": np.asarray(cache["ETKF_log10"]),
        "M00": np.asarray(cache["M00_log10"]),
        "M11": np.asarray(cache["M11_log10"]),
    }
    strata = {}
    csv_rows = []
    for label, alt_mask in (("200-500km", (altitude >= 200) & (altitude <= 500)),
                            ("200-300km", (altitude >= 200) & (altitude < 300)),
                            ("300-500km", (altitude >= 300) & (altitude <= 500)),
                            ("120-200km", (altitude >= 120) & (altitude < 200))):
        for station_name in sorted(np.unique(station).tolist()):
            station_mask = station == station_name
            for period, period_mask in (("all", np.ones(len(station), dtype=bool)),
                                        ("day", day), ("night", ~day)):
                selected = alt_mask & station_mask & period_mask
                for model_name, prediction in models.items():
                    metrics = _metrics(observation[selected], prediction[selected])
                    key = f"{model_name}|{station_name}|{label}|{period}"
                    strata[key] = metrics
                    csv_rows.append([key, model_name, station_name, label, period,
                                     metrics["n"], metrics["bias"], metrics["rmse"],
                                     metrics["mae"], metrics["pearson_r"], metrics["ccc"]])
    # Compatibility aliases used by the existing ISR reporting convention.
    # ``all_alt`` covers the complete 120--500 km analysis domain.  Keep these
    # aliases station-specific; the combined station aggregate is not needed.
    all_alt_metrics = {}
    for period, period_mask in (("all", np.ones(len(station), dtype=bool)),
                                ("day", day), ("night", ~day)):
        alias = f"all_alt_{period}"
        all_alt_metrics[alias] = {}
        for station_name in sorted(np.unique(station).tolist()):
            selected = ((altitude >= 120.0) & (altitude <= 500.0)
                        & (station == station_name) & period_mask)
            all_alt_metrics[alias][station_name] = {}
            for model_name, prediction in models.items():
                metrics = _metrics(observation[selected], prediction[selected])
                all_alt_metrics[alias][station_name][model_name] = metrics
                key = f"{model_name}|{station_name}|{alias}"
                strata[key] = metrics
                csv_rows.append([key, model_name, station_name, "all_alt", period,
                                 metrics["n"], metrics["bias"], metrics["rmse"],
                                 metrics["mae"], metrics["pearson_r"], metrics["ccc"]])
    bootstrap = {}
    primary = (altitude >= 200) & (altitude <= 500)
    bootstrap["ETKF_vs_RAW_IRI_grid"] = _bootstrap(
        observation[primary], models["Static_ETKF"][primary],
        models["RAW_IRI_grid"][primary], unit[primary])
    bootstrap["ETKF_vs_M00"] = _bootstrap(
        observation[primary], models["Static_ETKF"][primary],
        models["M00"][primary], unit[primary])
    bootstrap["M11_vs_ETKF"] = _bootstrap(
        observation[primary], models["M11"][primary],
        models["Static_ETKF"][primary], unit[primary])
    peak_metrics = {}
    for model_name in ("ETKF", "IRI_grid", "M00", "M11"):
        prefix = "Static_ETKF" if model_name == "ETKF" else model_name
        peak_metrics[prefix] = {
            "nmf2": _metrics(peak["ISR_nmf2_log10"], peak[f"{model_name}_nmf2_log10"]),
            "hmf2": _metrics(peak["ISR_hmf2_km"], peak[f"{model_name}_hmf2_km"]),
        }
    audit = {
        "shards": list(audit_rows),
        "total_rows": int(len(station)),
        "status_counts": {str(k): int(v) for k, v in zip(
            *np.unique(np.asarray(cache["ETKF_status"]).astype(str), return_counts=True))},
        "wall_seconds": float(elapsed),
    }
    report = {"background_semantics": "frozen_RAW_IRI_grid_at_cycle_time",
              "m00_m11_role": "external_comparators_only",
              "primary_altitude_km": [200.0, 500.0],
              "low_altitude_diagnostic_km": [120.0, 200.0],
              "all_alt_metrics": all_alt_metrics,
              "stratified_metrics": strata,
              "bootstrap": bootstrap,
              "peak_metrics": peak_metrics,
              "audit_summary": audit}
    with open(output_dir / "isr_etkf_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=True)
    with open(output_dir / "isr_etkf_report.txt", "w", encoding="utf-8") as stream:
        stream.write("Traditional static-local ETKF ISR validation (RAW IRI background)\n")
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    csv_path = output_dir / "isr_etkf_stratified_metrics.csv"
    try:
        stream = open(csv_path, "w", newline="", encoding="utf-8")
    except PermissionError:
        # A stale Windows reader may hold the previous CSV; preserve the new
        # metrics under an explicit suffix instead of losing the report.
        csv_path = output_dir / "isr_etkf_stratified_metrics_updated.csv"
        stream = open(csv_path, "w", newline="", encoding="utf-8")
    with stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "model", "station", "altitude", "period", "n",
                         "bias", "rmse", "mae", "pearson_r", "ccc"])
        writer.writerows(csv_rows)
    with open(output_dir / "isr_etkf_audit.json", "w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=True)
    return report


def run_isr(args):
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve()
    config = load_manifest(manifest_path)
    split_path = _split_path(config, manifest_path)
    split_days = _date_days(split_path)
    indexes, allowed, split_audit = _load_online_indices(config, split_days)
    proxy = load_iri_proxy(config["iri_proxy_path"], device=args.device)
    field_path = Path(args.static_dir).resolve() / "static_anomalies.npy"
    field = np.load(field_path, mmap_mode="r")
    if field.shape != GRID.shape + (N_MEMBERS,):
        raise ValueError(f"static anomaly shape mismatch: {field.shape}")
    stations = (("Jicamarca",) if args.station == "Jicamarca" else
                ("PokerFlat",) if args.station == "PokerFlat" else
                ("Jicamarca", "PokerFlat"))
    records = _load_isr_records(stations, args)
    audit_rows = _process_records(records, output_dir, field, proxy,
                                  indexes, allowed, args)
    shards = _read_shards(output_dir)
    canonical_path = Path(args.canonical_dir).resolve() / "isr_evaluation_cache.npz"
    expected_keys = shards["key"] if args.max_days > 0 else None
    cache = _strict_join_canonical(shards, canonical_path, expected_keys=expected_keys)
    np.savez_compressed(output_dir / "isr_etkf_cache.npz", **cache)
    canonical_peak = Path(args.canonical_dir).resolve() / "isr_peak_cache.npz"
    if not canonical_peak.exists():
        raise FileNotFoundError(canonical_peak)
    peak = _peak_cache(cache, canonical_peak, output_dir)
    report = _report(cache, peak, output_dir, audit_rows,
                     time.perf_counter() - started)
    cost = {"wall_seconds": time.perf_counter() - started,
            "node_chunk_size": int(args.node_chunk_size),
            "split_audit": split_audit,
            "row_count": int(len(cache["key"]))}
    with open(output_dir / "cost.json", "w", encoding="utf-8") as stream:
        json.dump(cost, stream, ensure_ascii=False, indent=2)
    return {"output_dir": str(output_dir), "rows": int(len(cache["key"])),
            "wall_seconds": cost["wall_seconds"],
            "report": str(output_dir / "isr_etkf_report.json")}


def refresh_report(args):
    """Regenerate metrics from completed ETKF caches without rerunning ETKF."""
    output_dir = Path(args.output_dir).resolve()
    with np.load(output_dir / "isr_etkf_cache.npz", allow_pickle=False) as data:
        cache = {key: np.asarray(data[key]) for key in data.files}
    with np.load(output_dir / "isr_etkf_peak_cache.npz", allow_pickle=False) as data:
        peak = {key: np.asarray(data[key]) for key in data.files}
    audit_path = output_dir / "isr_etkf_audit.json"
    if audit_path.exists():
        with open(audit_path, "r", encoding="utf-8") as stream:
            previous_audit = json.load(stream)
    else:
        previous_audit = {"shards": [], "wall_seconds": 0.0}
    report = _report(cache, peak, output_dir,
                     previous_audit.get("shards", []),
                     float(previous_audit.get("wall_seconds", 0.0)))
    return {"output_dir": str(output_dir),
            "report": str(output_dir / "isr_etkf_report.json"),
            "rows": int(len(cache["key"]))}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("isr")
    command.add_argument("--manifest", required=True)
    command.add_argument("--static-dir", required=True)
    command.add_argument("--canonical-dir", default=CANONICAL_DEFAULT)
    command.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    command.add_argument("--station", choices=("Jicamarca", "PokerFlat", "both"), default="both")
    command.add_argument("--jicamarca-dir", default=r"D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)")
    command.add_argument("--poker-flat-dir", default=r"D:\ISR\DATA\61poker_flat_is_radar(lp)\05min")
    command.add_argument("--start-date", default="2024-09-01")
    command.add_argument("--end-date", default="2024-10-01")
    command.add_argument("--err-ratio-max", type=float, default=0.5)
    command.add_argument("--beam-select", default="max_elm")
    command.add_argument("--device", default="cpu")
    command.add_argument("--node-chunk-size", type=int, default=256)
    command.add_argument("--max-days", type=int, default=0,
                         help="smoke limit; 0 processes every loaded day")
    command.add_argument("--resume", action="store_true")
    command.set_defaults(func=run_isr)
    report_command = sub.add_parser("report")
    report_command.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    report_command.set_defaults(func=refresh_report)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    print(json.dumps(args.func(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
