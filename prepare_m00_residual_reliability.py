"""Train-only A1-S M00 residual/reliability and localization diagnostics.

Use ``--input-npz`` for a deterministic unit/smoke run.  The production path
loads the actual A1-S Background checkpoint and only the 20 Train dates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from inr_modules.config_mdia import get_config_mdia
from inr_modules.mdia.m00_observation import load_magnetic_sidecar
from inr_modules.mdia.m00_residual_experiment import (
    build_empirical_reliability,
    replay_train_loo_localization,
    validate_a1s_manifest,
    write_json,
)


def _load_npz_rows(path: str) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as payload:
        required = ("coords", "target", "prediction", "profile_id")
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(f"input npz missing {missing}")
        coords = np.asarray(payload["coords"], dtype=np.float64)
        target = np.asarray(payload["target"], dtype=np.float64).reshape(-1)
        prediction = np.asarray(payload["prediction"], dtype=np.float64).reshape(-1)
        profile_id = np.asarray(payload["profile_id"], dtype=np.int64).reshape(-1)
        if "qd_latitude_deg" not in payload.files:
            raise ValueError("input npz requires qd_latitude_deg")
        latitude_key = "qd_latitude_deg"
        magnetic = np.asarray(payload[latitude_key], dtype=np.float64).reshape(-1)
        audit = (np.asarray(payload["aacgm_latitude_deg"], dtype=np.float64).reshape(-1)
                 if "aacgm_latitude_deg" in payload.files and latitude_key != "aacgm_latitude_deg"
                 else None)
        source = (np.asarray(payload["source"]).reshape(-1)
                  if "source" in payload.files else np.full(target.size, "FY"))
    arrays = (prediction, profile_id, magnetic, source)
    if audit is not None:
        arrays = arrays + (audit,)
    if coords.ndim != 2 or coords.shape[1] < 4 or any(value.size != target.size for value in arrays):
        raise ValueError("input npz rows are not aligned")
    return coords[:, :4], target, prediction, profile_id, magnetic, audit, source


def _select_profile_queries(
        coords: np.ndarray, profile_ids: np.ndarray, max_queries: int = 256,
        points_per_profile: int = 8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Date-balanced profiles with stable, distinct point coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    profile_ids = np.asarray(profile_ids, dtype=np.int64).reshape(-1)
    unique_profiles = np.unique(profile_ids)
    if unique_profiles.size == 0:
        raise ValueError("localization query pool empty")
    if max_queries < 1 or points_per_profile < 1:
        raise ValueError("localization query limits must be positive")
    profile_index: dict[int, np.ndarray] = {}
    profile_day: dict[int, int] = {}
    for profile_id in unique_profiles:
        indices = np.flatnonzero(profile_ids == profile_id)
        profile_index[int(profile_id)] = indices
        profile_day[int(profile_id)] = int(np.floor(np.median(coords[indices, 3]) / 24.0))
    rng = np.random.default_rng(seed)
    by_day: dict[int, list[int]] = {}
    for profile_id, day in profile_day.items():
        by_day.setdefault(day, []).append(profile_id)
    for values in by_day.values():
        rng.shuffle(values)
    selected: list[int] = []
    target = min(int(max_queries), len(profile_index))
    while len(selected) < target:
        progressed = False
        for day in sorted(by_day):
            if by_day[day]:
                selected.append(by_day[day].pop())
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    query_indices = np.concatenate([
        profile_index[profile_id][:points_per_profile] for profile_id in selected
    ])
    query_profile_ids = np.concatenate([
        np.full(min(points_per_profile, profile_index[profile_id].size), profile_id,
                dtype=np.int64)
        for profile_id in selected
    ])
    return coords[query_indices], query_profile_ids, {
        "seed": int(seed),
        "selection": "stable_first_up_to_8_points_per_profile_round_robin_utc_day_v2",
        "query_profiles": int(len(selected)),
        "query_points": int(query_indices.size),
        "points_per_profile_max": int(points_per_profile),
        "query_days": sorted({profile_day[profile_id] for profile_id in selected}),
    }


def _replay_index_localization(source: str, coords: np.ndarray,
                               profile_ids: np.ndarray, max_queries: int,
                               config: dict) -> dict:
    """Replay production token neighborhoods with a Train profile allow-list."""
    from inr_modules.data_managers.FY_dataloader import (
        COSMICNeighborhoodIndex,
        FYNeighborhoodIndex,
    )

    query_coords, query_profile_ids, selection = _select_profile_queries(
        coords, profile_ids, max_queries)
    index_cls = FYNeighborhoodIndex if source == "FY" else COSMICNeighborhoodIndex
    data_key = "fy_path" if source == "FY" else "cosmic_path"
    allowed_profile_ids = np.unique(np.asarray(profile_ids, dtype=np.int64))
    index_config = dict(config)
    index_config["strict_preload_allowed_profile_ids"] = allowed_profile_ids
    index_config["strict_preload_token_only"] = True
    index = index_cls(index_config[data_key], index_config)
    payload = index.query_observation_batch(
        query_coords.astype(np.float32),
        exclude_profile_ids=query_profile_ids,
        allowed_profile_ids=allowed_profile_ids,
    )
    row_ptr = np.asarray(payload["row_ptr"], dtype=np.int64).reshape(-1)
    valid = np.asarray(payload["valid_mask"], dtype=bool).reshape(-1)
    weights = np.asarray(payload["localization_weight"], dtype=np.float64).reshape(-1)
    counts = np.diff(row_ptr)
    positive = np.zeros(len(query_coords), dtype=np.int64)
    masses = np.zeros(len(query_coords), dtype=np.float64)
    for query_index in range(len(query_coords)):
        start, end = int(row_ptr[query_index]), int(row_ptr[query_index + 1])
        local_valid = valid[start:end]
        local_weights = weights[start:end]
        positive[query_index] = int(np.count_nonzero(local_valid & (local_weights > 0.0)))
        masses[query_index] = float(np.sum(local_weights[local_valid]))
    return {
        "semantics": "a1s_m00_train_loo_3d_localization_replay_v2",
        "train_only": True,
        "self_profile_excluded": True,
        "query_count": int(len(query_coords)),
        "reference_token_count": int(valid.size),
        "allowed_profile_count": int(allowed_profile_ids.size),
        "selection": selection,
        "scales": {
            "spatial_scale_km": float(config.get("physical_localization_space_km", 1800.0)),
            "vertical_scale_km": 100.0,
            "temporal_scale_hours": float(config.get("physical_localization_time_hours", 1.5)),
            "time_semantics": "relative_utc_hours_v1",
        },
        "effective_neighbor_count": {
            "mean": float(np.mean(positive)) if positive.size else 0.0,
            "min": int(np.min(positive)) if positive.size else 0,
            "max": int(np.max(positive)) if positive.size else 0,
        },
        "precision": {
            "mean_reference_rows": float(np.mean(counts)) if counts.size else 0.0,
            "mean_localized_mass": float(np.mean(masses)) if masses.size else 0.0,
        },
    }


def _train_days(manifest_path: str) -> list[int]:
    return list(validate_a1s_manifest(manifest_path)["train"])


def _production_rows(checkpoint: str, manifest_path: str, sidecars: dict[str, str]):
    # Reuse the existing dataset contract; no Development/Locked-test rows are
    # ever passed to the dataset constructor.
    from inr_modules.data_managers.FY_dataloader import COSMICDataset, FY3D_Dataset
    from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.mdia.fsia_model import FSIA_INR_Model

    config = dict(get_config_mdia())
    reference_manifest = os.path.join(
        os.path.dirname(os.path.abspath(checkpoint)), "run_manifest.json")
    if not os.path.isfile(reference_manifest):
        raise FileNotFoundError("A1-S colocated run_manifest.json is required")
    with open(reference_manifest, encoding="utf-8") as stream:
        manifest_payload = json.load(stream)
    reference_config = manifest_payload.get("config")
    if not isinstance(reference_config, dict):
        raise ValueError("A1-S run_manifest lacks its resolved config")
    # Reuse the exact model/data/physics settings used for epoch-3.  The
    # current date manifest is deliberately overlaid below; no hash is read.
    config.update(reference_config)
    config["date_split_manifest"] = manifest_path
    config["background_condition_adapter"] = "disabled"
    config["m00_candidate_semantics"] = "disabled"
    config["background_semantics"] = "learned_v1"
    device = torch.device("cpu")
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(config["iri_proxy_path"], map_location=device, weights_only=True))
    model = FSIA_INR_Model(iri_proxy, config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    days = _train_days(manifest_path)
    alt_range = tuple(config.get("observation_alt_range") or (200.0, 500.0))
    common = dict(mode="train", split_days={"train": days}, val_days=[],
                  bin_size_hours=config["bin_size_hours"], use_memmap=True,
                  val_ratio=None, split_seed=config["seed"], alt_range=alt_range)
    sources = {
        "FY": FY3D_Dataset(config["fy_path"], profile_index_path=config.get("fy_profile_index_path"), profile_path=config.get("fy_profile_path"), magnetic_sidecar_path=sidecars["FY"], magnetic_sidecar_source="FY", magnetic_sidecar_strict=True, **common),
        "COSMIC": COSMICDataset(config["cosmic_path"], profile_index_path=config.get("cosmic_profile_index_path"), magnetic_sidecar_path=sidecars["COSMIC"], magnetic_sidecar_source="COSMIC", magnetic_sidecar_strict=True, **common),
    }
    sw_manager = SpaceWeatherManager(config["sw_path"], config["start_date_str"], config["total_hours"], config["seq_len"], device=device)
    peak_manager = None
    if config.get("iri_hmf2_path") and config.get("iri_nmf2_path") and os.path.isfile(config["iri_hmf2_path"]) and os.path.isfile(config["iri_nmf2_path"]):
        peak_manager = IRIPeakManager(config["iri_hmf2_path"], config["iri_nmf2_path"], device=device)
    rows = []
    for source, dataset in sources.items():
        with np.load(sidecars[source], allow_pickle=False) as sidecar_probe:
            sidecar_rows = np.asarray(sidecar_probe["row_index"], dtype=np.int64).reshape(-1)
        raw_profile_ids = getattr(dataset, "raw_profile_ids", dataset.profile_ids)
        if (sidecar_rows.size == 0 or np.any(sidecar_rows < 0)
                or np.any(sidecar_rows >= len(dataset.data))):
            raise ValueError(f"{source} QD sidecar row index is invalid")
        magnetic = load_magnetic_sidecar(
            sidecars[source], expected_rows=sidecar_rows.size, strict_qd=True,
            expected_profile_ids=raw_profile_ids[sidecar_rows],
            expected_coords=np.asarray(dataset.data[sidecar_rows, :4], dtype=np.float64),
            expected_row_indices=sidecar_rows)
        dense_magnetic = {}
        for key, value in magnetic.items():
            arr = np.asarray(value)
            if arr.ndim == 1 and arr.size == sidecar_rows.size:
                if arr.dtype.kind in "fc":
                    dense = np.full(len(dataset.data), np.nan, dtype=arr.dtype)
                elif arr.dtype.kind == "b":
                    dense = np.zeros(len(dataset.data), dtype=arr.dtype)
                elif arr.dtype.kind in "iu":
                    dense = np.full(len(dataset.data), -1, dtype=arr.dtype)
                else:
                    dense = np.full(len(dataset.data), "", dtype=arr.dtype)
                dense[sidecar_rows] = arr
                dense_magnetic[key] = dense
            else:
                dense_magnetic[key] = value
        magnetic = dense_magnetic
        selected = np.asarray(dataset.selected_indices, dtype=np.int64)
        if (not np.isfinite(np.asarray(magnetic["qd_latitude_deg"], dtype=np.float64)[selected]).all()
                or not np.isfinite(np.asarray(magnetic["magnetic_sin_i"], dtype=np.float64)[selected]).all()):
            raise ValueError(f"{source} QD sidecar does not cover Train rows")
        diagnostic_batch_size = 32768
        for start in range(0, len(selected), diagnostic_batch_size):
            indices = selected[start:start + diagnostic_batch_size]
            data = np.asarray(dataset.data[indices], dtype=np.float32)
            valid = np.isfinite(data[:, :5]).all(axis=1)
            if not np.any(valid):
                continue
            data = data[valid]
            coords = torch.as_tensor(data[:, :4], dtype=torch.float32, device=device)
            sw = sw_manager.get_drivers_sequence(coords[:, 3])
            peak = peak_manager.get_iri_peak(coords) if peak_manager is not None else None
            with torch.no_grad():
                prediction = model.encode_background(coords, sw, iri_peak=peak)["ne_bkg"].reshape(-1).cpu().numpy()
            pids = np.asarray(getattr(dataset, "raw_profile_ids", dataset.profile_ids))[indices][valid]
            qd = magnetic["qd_latitude_deg"][indices][valid]
            aacgm = (magnetic.get("aacgm_latitude_deg", np.full(len(magnetic["row_index"]), np.nan))
                     [indices][valid])
            metadata = {}
            if "metadata" in magnetic:
                try:
                    parsed = json.loads(str(np.asarray(magnetic["metadata"]).reshape(-1)[0]))
                    if isinstance(parsed, dict):
                        metadata = parsed
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
            rows.append((source, data[:, :4].astype(np.float64), data[:, 4].astype(np.float64),
                         prediction.astype(np.float64), pids.astype(np.int64), qd.astype(np.float64),
                         aacgm.astype(np.float64), np.full(len(data), source), metadata))
    return rows, config


def _production_rows_train_only(checkpoint: str, manifest_path: str,
                                sidecars: dict[str, str]):
    """Run the frozen A1-S forward on authorized Train rows only.

    The loader used by historical training scans a full memmap before applying
    date masks.  Diagnostics have a stricter boundary: the sidecar row list is
    the authorization source, so no Development/Locked-test raw row is read.
    """
    from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.mdia.fsia_model import FSIA_INR_Model

    config = dict(get_config_mdia())
    reference_manifest = os.path.join(
        os.path.dirname(os.path.abspath(checkpoint)), "run_manifest.json")
    if not os.path.isfile(reference_manifest):
        raise FileNotFoundError("A1-S colocated run_manifest.json is required")
    with open(reference_manifest, encoding="utf-8") as stream:
        manifest_payload = json.load(stream)
    reference_config = manifest_payload.get("config")
    if not isinstance(reference_config, dict):
        raise ValueError("A1-S run_manifest lacks its resolved config")
    config.update(reference_config)
    config["date_split_manifest"] = manifest_path
    config["background_condition_adapter"] = "disabled"
    config["m00_candidate_semantics"] = "disabled"
    config["background_semantics"] = "learned_v1"
    device = torch.device("cpu")
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    if not os.path.isfile(config["iri_proxy_path"]):
        raise FileNotFoundError("A1-S IRI proxy checkpoint is missing")
    iri_proxy.load_state_dict(torch.load(
        config["iri_proxy_path"], map_location=device, weights_only=True))
    model = FSIA_INR_Model(iri_proxy, config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    if not (config.get("iri_hmf2_path") and config.get("iri_nmf2_path")
            and os.path.isfile(config["iri_hmf2_path"])
            and os.path.isfile(config["iri_nmf2_path"])):
        raise FileNotFoundError("A1-S IRI peak inputs are required for residual replay")
    peak_manager = IRIPeakManager(
        config["iri_hmf2_path"], config["iri_nmf2_path"], device=device)
    sw_manager = SpaceWeatherManager(
        config["sw_path"], config["start_date_str"], config["total_hours"],
        config["seq_len"], device=device)
    train_days = set(_train_days(manifest_path))
    alt_range = tuple(config.get("observation_alt_range") or (200.0, 500.0))
    rows = []
    for source, data_key in (("FY", "fy_path"), ("COSMIC", "cosmic_path")):
        raw = np.load(config[data_key], mmap_mode="r")
        if raw.ndim != 2 or raw.shape[1] < 5:
            raise ValueError(f"{source} raw data lacks [coords,target]")
        with np.load(sidecars[source], allow_pickle=False) as probe:
            sidecar_rows = np.asarray(probe["row_index"], dtype=np.int64).reshape(-1)
        if sidecar_rows.size == 0 or np.unique(sidecar_rows).size != sidecar_rows.size:
            raise ValueError(f"{source} Train QD sidecar row list is invalid")
        sidecar = load_magnetic_sidecar(
            sidecars[source], expected_rows=sidecar_rows.size, strict_qd=True,
            expected_coords=np.asarray(raw[sidecar_rows, :4], dtype=np.float64),
            expected_row_indices=sidecar_rows)
        coords_all = np.asarray(raw[sidecar_rows, :4], dtype=np.float64)
        row_days = np.floor(coords_all[:, 3] / 24.0).astype(np.int64)
        if not set(row_days.tolist()).issubset(train_days):
            raise ValueError(f"{source} sidecar contains non-Train rows")
        if not np.isfinite(np.asarray(raw[sidecar_rows, :5], dtype=np.float64)).all():
            raise ValueError(f"{source} authorized Train rows contain non-finite values")
        alt_mask = ((coords_all[:, 2] >= float(alt_range[0]))
                    & (coords_all[:, 2] <= float(alt_range[1])))
        selected = np.flatnonzero(alt_mask)
        if selected.size == 0:
            raise ValueError(f"{source} Train rows have no configured altitude support")
        pids = np.asarray(sidecar["profile_id"], dtype=np.int64).reshape(-1)
        qd = np.asarray(sidecar["qd_latitude_deg"], dtype=np.float64).reshape(-1)
        aacgm = np.asarray(sidecar.get(
            "aacgm_latitude_deg", np.full(sidecar_rows.size, np.nan)),
            dtype=np.float64).reshape(-1)
        metadata = {}
        if "metadata" in sidecar:
            try:
                parsed = json.loads(str(np.asarray(sidecar["metadata"]).reshape(-1)[0]))
                if isinstance(parsed, dict):
                    metadata = parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
        batch_size = 32768
        for start in range(0, selected.size, batch_size):
            take = selected[start:start + batch_size]
            data = np.asarray(raw[sidecar_rows[take], :5], dtype=np.float32)
            coord = torch.as_tensor(data[:, :4], dtype=torch.float32, device=device)
            sw = sw_manager.get_drivers_sequence(coord[:, 3])
            peak = peak_manager.get_iri_peak(coord)
            with torch.no_grad():
                prediction = model.encode_background(
                    coord, sw, iri_peak=peak)["ne_bkg"].reshape(-1).cpu().numpy()
            rows.append((source, data[:, :4].astype(np.float64),
                         data[:, 4].astype(np.float64), prediction.astype(np.float64),
                         pids[take], qd[take], aacgm[take],
                         np.full(take.size, source), metadata))
    return rows, config


def run(output: str, *, input_npz: str | None = None, checkpoint: str | None = None,
        date_manifest: str | None = None, sidecar_fy: str | None = None,
        sidecar_cosmic: str | None = None, max_localization_queries: int = 256) -> dict:
    if max_localization_queries < 1:
        raise ValueError("max_localization_queries must be positive")
    diagnostic_config: dict | None = None
    if input_npz:
        coords, target, prediction, pids, magnetic, audit, source = _load_npz_rows(input_npz)
        rows = [("mixed", coords, target, prediction, pids, magnetic, audit, source, {})]
    else:
        if not checkpoint or not date_manifest or not sidecar_fy or not sidecar_cosmic:
            raise ValueError("production diagnostics require checkpoint, 20/5/5 manifest and both AACGM sidecars")
        validate_a1s_manifest(date_manifest)
        rows, diagnostic_config = _production_rows_train_only(
            checkpoint, date_manifest, {"FY": sidecar_fy, "COSMIC": sidecar_cosmic})
    grouped = {}
    for item in rows:
        grouped.setdefault(item[0], []).append(item)
    residual_payloads = []
    loc_reports = {}
    for source, chunks in grouped.items():
        coords = np.concatenate([item[1] for item in chunks], axis=0)
        target = np.concatenate([item[2] for item in chunks], axis=0)
        prediction = np.concatenate([item[3] for item in chunks], axis=0)
        pids = np.concatenate([item[4] for item in chunks], axis=0)
        magnetic = np.concatenate([item[5] for item in chunks], axis=0)
        audits = [item[6] for item in chunks]
        audit = (np.concatenate(audits, axis=0)
                 if all(value is not None for value in audits) else None)
        source_values = np.full(target.size, source, dtype=object)
        magnetic_metadata = chunks[0][8] if chunks else {}
        magnetic_metadata = dict(magnetic_metadata)
        if "actual_height_semantics" not in magnetic_metadata:
            magnetic_metadata["actual_height_semantics"] = "row_altitude_km_v1"
        # Do not invent an AACGM provenance value for historical sidecars.  A
        # generator-produced sidecar may provide it explicitly; otherwise the
        # migration report records that the sidecar metadata is incomplete.
        result = build_empirical_reliability(
            target - prediction, coords, pids, source_values,
            qd_latitude_deg=magnetic, aacgm_latitude_audit_deg=audit,
            magnetic_metadata=magnetic_metadata)
        residual_payloads.append(result.payload)
        # Localization is defined on the profile neighborhood, not repeated
        # observation rows.  Keep one finite representative row per Train
        # profile so the full profile pool remains covered without an
        # intractable query-by-row Cartesian product.
        loc_reports[source] = (
            replay_train_loo_localization(coords, coords, pids, pids)
            if diagnostic_config is None else _replay_index_localization(
                source, coords, pids, max_localization_queries, diagnostic_config))
    reliability = residual_payloads[0]
    for other in residual_payloads[1:]:
        for name in ("bias", "residual_mean", "variance", "representativeness",
                     "representativeness_support", "support", "reliability_weight"):
            reliability[name].update(other[name])
    reliability["sources"] = sorted(grouped)
    migrations = [item.get("qd_aacgm_migration") for item in residual_payloads
                  if item.get("qd_aacgm_migration") is not None]
    if migrations:
        reliability["qd_aacgm_migration"] = migrations
    reliability["qd_migration_review_stop"] = False
    reliability["qd_migration_acceptance"] = "user_accepted_qd_v1"
    reliability["localization_diagnostic"] = loc_reports
    reliability["candidate_transition"] = "decoder_last_trust_region_then_peak_relative_lowrank3"
    reliability["m00_frozen_reference"] = "A1-S_epoch3"
    output = os.path.abspath(output)
    write_json(output, reliability)
    write_json(os.path.splitext(output)[0] + "_localization.json", {
        "semantics": "a1s_m00_train_loo_3d_localization_replay_v2",
        "reports": loc_reports,
    })
    return reliability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-npz", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--date-split-manifest", default=None)
    parser.add_argument("--sidecar-fy", default=None)
    parser.add_argument("--sidecar-cosmic", default=None)
    parser.add_argument("--max-localization-queries", type=int, default=256)
    args = parser.parse_args()
    values = vars(args)
    values["date_manifest"] = values.pop("date_split_manifest")
    run(**values)


if __name__ == "__main__":
    main()
