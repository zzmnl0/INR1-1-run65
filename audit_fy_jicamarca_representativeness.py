"""Read-only FY/Jicamarca representativeness audit for run66 G1."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_vertical_support_counterfactual import _validate_payload
from estimate_empirical_covariance import _deterministic_npz, _sha256
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    query_observation_payload,
)
from isr_evaluation.diagnose_jicamarca_modes import (
    _observation_diagnostics,
    _record_arrays,
)
from isr_evaluation.isr_loader import load_jicamarca
from isr_evaluation.main_isr_eval import (
    CONFIG as ISR_CONFIG,
    _load_model_and_managers,
    _parse_unix,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    ROOT / "checkpoints_fsia" / "run66-covfactor-d64-n8-global-localized"
    / "best_fsia_model.pth"
)
OUTPUT = (
    ROOT / "isr_validation_outputs"
    / "run66-direction-correction-g1-fy-jicamarca-representativeness"
)
MANIFEST = CHECKPOINT.parent / "run_manifest.json"
MODES = ("M10", "M11")


def _bootstrap_mean(values, dates, replicates=1000, seed=42):
    """Two-stage date/profile bootstrap; profiles remain the analysis unit."""
    values = np.asarray(values, dtype=np.float64)
    dates = np.asarray(dates)
    if not len(values):
        return [None, None]
    unique_dates = np.unique(dates)
    groups = [np.flatnonzero(dates == day) for day in unique_dates]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_dates = rng.integers(0, len(groups), size=len(groups))
        sampled = []
        for date_index in sampled_dates:
            group = groups[date_index]
            sampled.append(group[rng.integers(0, len(group), size=len(group))])
        estimates[replicate] = np.mean(values[np.concatenate(sampled)])
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def _profile_equal_summary(pairs, mask, replicates=1000, seed=42):
    mask = np.asarray(mask, dtype=bool)
    total_profiles = np.unique(pairs["profile_id"][mask]).size
    selected = mask & pairs["high_confidence"]
    if not selected.any():
        return {
            "n_pairs": int(mask.sum()),
            "n_profiles": int(total_profiles),
            "high_confidence_pairs": 0,
            "high_confidence_profiles": 0,
        }

    date_lookup = {
        int(profile): int(date)
        for profile, date in zip(pairs["profile_id"], pairs["profile_date"])
    }

    def metric(pair_mask, pair_values, offset):
        profile_id = pairs["profile_id"][pair_mask]
        if not len(profile_id):
            return {"value": None, "ci95": [None, None], "n_profiles": 0}
        order = np.argsort(profile_id, kind="stable")
        profile_id = profile_id[order]
        starts = np.flatnonzero(np.r_[True, profile_id[1:] != profile_id[:-1]])
        counts = np.diff(np.r_[starts, len(order)])
        profile_ids = profile_id[starts]
        values = (
            np.add.reduceat(np.asarray(pair_values)[pair_mask][order], starts)
            / counts
        )
        dates = np.asarray([date_lookup[int(profile)] for profile in profile_ids])
        return {
            "value": float(np.mean(values)),
            "ci95": _bootstrap_mean(
                values, dates, replicates=replicates, seed=seed + offset
            ),
            "n_profiles": int(len(profile_ids)),
        }

    return {
        "n_pairs": int(mask.sum()),
        "n_profiles": int(total_profiles),
        "high_confidence_pairs": int(selected.sum()),
        "high_confidence_profiles": int(np.unique(
            pairs["profile_id"][selected]
        ).size),
        "effective_dates": int(np.unique(
            pairs["profile_date"][selected]
        ).size),
        "innovation_toward_isr": metric(
            selected, pairs["innovation_toward"], 1
        ),
        "contribution_toward_isr": metric(
            selected & (np.abs(pairs["contribution"]) > 1e-12),
            pairs["contribution_toward"], 2,
        ),
        "negative_cross_covariance": metric(
            selected, pairs["cross_covariance"] < 0.0, 3
        ),
        "negative_kalman_gain": metric(
            selected, pairs["kalman_gain"] < 0.0, 4
        ),
    }


def _profile_metadata(index_path):
    with np.load(index_path, allow_pickle=True) as index:
        ids = np.asarray(index["profile_id"], dtype=np.int64)
        paths = np.asarray(index["original_relative_path"]).astype(str)
        order = np.argsort(ids)
        metadata = {
            "profile_id": ids[order],
            "path": paths[order],
            "satellite": np.asarray(
                [path.replace("\\", "/").split("/", 1)[0] for path in paths[order]]
            ),
        }
        for key in (
            "date_code", "representative_lat", "representative_lon",
            "representative_time", "h_cut_km", "kept_points",
        ):
            metadata[key] = np.asarray(index[key])[order]
    return metadata


def _attach_profile_metadata(pairs, metadata):
    positions = np.searchsorted(metadata["profile_id"], pairs["profile_id"])
    if (
        np.any(positions >= len(metadata["profile_id"]))
        or not np.array_equal(
            metadata["profile_id"][positions], pairs["profile_id"]
        )
    ):
        raise ValueError("FY diagnostic contains profile_id absent from QC index")
    pairs["profile_date"] = metadata["date_code"][positions].astype(np.int32)
    pairs["satellite"] = metadata["satellite"][positions]
    pairs["original_path"] = metadata["path"][positions]
    return pairs


def _aggregate_profile_query(tokens, query_coords, mode, support="all"):
    token_query = query_coords[np.asarray(tokens["query_index"], dtype=np.int64)]
    vertical_distance_token = np.abs(
        np.asarray(tokens["observation_coords"])[:, 2] - token_query[:, 2]
    )
    query_layer = np.digitize(token_query[:, 2], [200.0, 300.0])
    observation_layer = np.digitize(
        np.asarray(tokens["observation_coords"])[:, 2], [200.0, 300.0]
    )
    selected = (
        (tokens["mode"] == mode)
        & (tokens["precision"] > 0.0)
        & np.isfinite(tokens["precision"])
    )
    if support == "same_layer":
        selected &= query_layer == observation_layer
    elif support == "within_20km":
        selected &= vertical_distance_token <= 20.0
    elif support != "all":
        raise ValueError(f"unknown support: {support}")
    if not selected.any():
        raise ValueError(
            f"{mode}/{support} has no positive-precision FY tokens"
        )

    data = {key: np.asarray(value)[selected] for key, value in tokens.items()}
    query_index = data["query_index"].astype(np.int64)
    profile_id = data["profile_id"].astype(np.int64)
    order = np.lexsort((profile_id, query_index))
    query_index = query_index[order]
    profile_id = profile_id[order]
    boundary = np.r_[
        True,
        (query_index[1:] != query_index[:-1])
        | (profile_id[1:] != profile_id[:-1]),
    ]
    starts = np.flatnonzero(boundary)
    counts = np.diff(np.r_[starts, len(order)])
    weights = data["precision"][order].astype(np.float64)
    weight_sum = np.add.reduceat(weights, starts)

    def weighted(field):
        values = np.asarray(data[field])[order].astype(np.float64)
        return np.add.reduceat(values * weights, starts) / weight_sum

    def summed(field):
        return np.add.reduceat(
            np.asarray(data[field])[order].astype(np.float64), starts
        )

    query_index = query_index[starts]
    profile_id = profile_id[starts]
    observation_coords = np.asarray(data["observation_coords"])[order]
    observation_altitude = np.add.reduceat(
        observation_coords[:, 2] * weights, starts
    ) / weight_sum
    query = query_coords[query_index]
    dlat = np.abs(observation_coords[:, 0] - query_coords[
        np.asarray(data["query_index"])[order], 0
    ])
    dlon = np.abs(
        (
            observation_coords[:, 1]
            - query_coords[np.asarray(data["query_index"])[order], 1]
            + 180.0
        )
        % 360.0
        - 180.0
    )
    horizontal_rho = np.sqrt((dlat / 5.0) ** 2 + (dlon / 15.0) ** 2)
    desired = data["query_observation"][order][starts] - data[
        "query_background"
    ][order][starts]
    innovation = weighted("physical_innovation")
    contribution = summed("kalman_contribution")
    cross_covariance = weighted("cross_covariance")
    kalman_gain = weighted("kalman_gain")
    rho = weighted("rho_squared") ** 0.5
    vertical_distance = np.add.reduceat(
        np.abs(observation_coords[:, 2] - query_coords[
            np.asarray(data["query_index"])[order], 2
        ]) * weights,
        starts,
    ) / weight_sum
    horizontal_rho = np.add.reduceat(
        horizontal_rho * weights, starts
    ) / weight_sum
    local_time = np.remainder(query[:, 3] + query[:, 1] / 15.0, 24.0)

    return {
        "mode": np.full(len(starts), mode),
        "support": np.full(len(starts), support),
        "query_index": query_index,
        "profile_id": profile_id,
        "query_altitude": query[:, 2].astype(np.float64),
        "observation_altitude": observation_altitude,
        "local_time": local_time.astype(np.float64),
        "rho": rho,
        "horizontal_rho": horizontal_rho,
        "vertical_distance": vertical_distance,
        "desired": desired.astype(np.float64),
        "innovation": innovation,
        "cross_covariance": cross_covariance,
        "kalman_gain": kalman_gain,
        "contribution": contribution,
        "precision_mass": weight_sum,
        "token_count": counts.astype(np.int32),
        "high_confidence": (np.abs(desired) >= 0.05)
        & (np.abs(innovation) >= 0.05),
        "innovation_toward": np.sign(innovation) == np.sign(desired),
        "contribution_toward": np.sign(contribution) == np.sign(desired),
    }


def _combine_pair_arrays(pair_by_mode):
    keys = sorted(set.intersection(*(set(value) for value in pair_by_mode.values())))
    return {
        key: np.concatenate([value[key] for value in pair_by_mode.values()])
        for key in keys
    }


def _summaries(pairs, replicates, seed):
    primary_population = (
        (pairs["mode"] == "M10") & (pairs["support"] == "within_20km")
    )
    rho_median = float(np.median(pairs["rho"][primary_population]))
    report = {"rho_median": rho_median, "modes": {}}
    for mode_index, mode in enumerate(MODES):
        mode_mask = pairs["mode"] == mode
        supports = {
            "all": mode_mask & (pairs["support"] == "all"),
            "same_layer": mode_mask & (pairs["support"] == "same_layer"),
            "within_20km": mode_mask & (pairs["support"] == "within_20km"),
            "within_20km_nearest_half": mode_mask
            & (pairs["support"] == "within_20km")
            & (pairs["rho"] <= rho_median),
            "within_20km_farthest_half": mode_mask
            & (pairs["support"] == "within_20km")
            & (pairs["rho"] > rho_median),
        }
        mode_report = {
            name: _profile_equal_summary(
                pairs, mask, replicates, seed + mode_index * 100 + index * 10
            )
            for index, (name, mask) in enumerate(supports.items())
        }
        base = mode_mask & (pairs["support"] == "within_20km")
        mode_report["by_satellite"] = {
            str(value): _profile_equal_summary(
                pairs, base & (pairs["satellite"] == value),
                replicates, seed + 200 + index,
            )
            for index, value in enumerate(sorted(np.unique(pairs["satellite"][base])))
        }
        mode_report["by_date"] = {
            str(int(value)): _profile_equal_summary(
                pairs, base & (pairs["profile_date"] == value),
                replicates, seed + 300 + index,
            )
            for index, value in enumerate(sorted(np.unique(pairs["profile_date"][base])))
        }
        altitude_bin = (
            np.floor((pairs["query_altitude"] - 120.0) / 20.0) * 20.0 + 120.0
        ).astype(np.int16)
        mode_report["by_query_altitude_20km"] = {
            str(int(value)): _profile_equal_summary(
                pairs, base & (altitude_bin == value),
                replicates, seed + 400 + index,
            )
            for index, value in enumerate(sorted(np.unique(altitude_bin[base])))
        }
        lt_bin = np.floor(pairs["local_time"]).astype(np.int16)
        mode_report["by_local_time_1h"] = {
            str(int(value)): _profile_equal_summary(
                pairs, base & (lt_bin == value),
                replicates, seed + 500 + index,
            )
            for index, value in enumerate(sorted(np.unique(lt_bin[base])))
        }
        for field, name, offset in (
            ("rho", "by_rho_quartile", 600),
            ("horizontal_rho", "by_horizontal_distance_quartile", 700),
        ):
            edges = np.quantile(pairs[field][base], [0.0, 0.25, 0.5, 0.75, 1.0])
            rows = {}
            for quartile in range(4):
                upper = (
                    pairs[field] <= edges[quartile + 1]
                    if quartile == 3
                    else pairs[field] < edges[quartile + 1]
                )
                group = base & (pairs[field] >= edges[quartile]) & upper
                rows[str(quartile + 1)] = {
                    "lower": float(edges[quartile]),
                    "upper": float(edges[quartile + 1]),
                    **_profile_equal_summary(
                        pairs, group, replicates,
                        seed + offset + mode_index * 10 + quartile,
                    ),
                }
            mode_report[name] = rows
        vertical_edges = (0.0, 5.0, 10.0, 20.000001)
        mode_report["by_vertical_distance_km"] = {
            f"{vertical_edges[index]:g}-{vertical_edges[index + 1]:g}": (
                _profile_equal_summary(
                    pairs,
                    base
                    & (pairs["vertical_distance"] >= vertical_edges[index])
                    & (pairs["vertical_distance"] < vertical_edges[index + 1]),
                    replicates,
                    seed + 800 + mode_index * 10 + index,
                )
            )
            for index in range(3)
        }
        report["modes"][mode] = mode_report
    return report


def _profile_fraction(pairs, selected, values, replicates, seed):
    profile_ids = pairs["profile_id"][selected]
    if not len(profile_ids):
        return {"value": None, "ci95": [None, None], "n_profiles": 0}
    order = np.argsort(profile_ids, kind="stable")
    profile_ids = profile_ids[order]
    starts = np.flatnonzero(np.r_[True, profile_ids[1:] != profile_ids[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    unique_profiles = profile_ids[starts]
    profile_values = (
        np.add.reduceat(np.asarray(values)[selected][order], starts) / counts
    )
    date_lookup = {
        int(profile): int(date)
        for profile, date in zip(pairs["profile_id"], pairs["profile_date"])
    }
    dates = np.asarray([date_lookup[int(profile)] for profile in unique_profiles])
    return {
        "value": float(np.mean(profile_values)),
        "ci95": _bootstrap_mean(profile_values, dates, replicates, seed),
        "n_profiles": int(len(unique_profiles)),
    }


def _joint_direction_change(pairs, rho_median):
    rows = {}
    for mode in MODES:
        selected = (
            (pairs["mode"] == mode)
            & (pairs["support"] == "within_20km")
            & (pairs["rho"] <= rho_median)
            & pairs["high_confidence"]
            & (np.abs(pairs["contribution"]) > 1e-12)
        )
        rows[mode] = {
            (int(query), int(profile)): bool(direction)
            for query, profile, direction in zip(
                pairs["query_index"][selected],
                pairs["profile_id"][selected],
                pairs["contribution_toward"][selected],
            )
        }
    common = sorted(set(rows["M10"]) & set(rows["M11"]))
    if not common:
        return {"n_pairs": 0}
    m10 = np.asarray([rows["M10"][key] for key in common])
    m11 = np.asarray([rows["M11"][key] for key in common])
    return {
        "n_pairs": int(len(common)),
        "sign_direction_changed_fraction": float(np.mean(m10 != m11)),
        "M10_correct_to_M11_wrong_fraction": float(np.mean(m10 & ~m11)),
        "M10_wrong_to_M11_correct_fraction": float(np.mean(~m10 & m11)),
    }


def _attribution_and_gate(pairs, summaries, replicates=1000, seed=42):
    primary = (
        (pairs["mode"] == "M10")
        & (pairs["support"] == "within_20km")
        & (pairs["rho"] <= summaries["rho_median"])
        & pairs["high_confidence"]
        & (np.abs(pairs["contribution"]) > 1e-12)
    )
    innovation_correct = pairs["innovation_toward"][primary]
    contribution_correct = pairs["contribution_toward"][primary]
    cross_negative = pairs["cross_covariance"][primary] < 0.0
    categories = {
        "observation_conflict_propagated": (~innovation_correct)
        & (~contribution_correct),
        "covariance_corrected_observation_conflict": (~innovation_correct)
        & contribution_correct,
        "negative_covariance_reversal": innovation_correct
        & (~contribution_correct)
        & cross_negative,
        "gain_mixing_or_multitoken_reversal": innovation_correct
        & (~contribution_correct)
        & (~cross_negative),
    }
    attribution = {}
    for index, (name, mask) in enumerate(categories.items()):
        full_mask = np.zeros(len(pairs["profile_id"]), dtype=bool)
        full_mask[primary] = mask
        attribution[name] = {
            "n_pairs": int(mask.sum()),
            "pair_fraction": float(np.mean(mask)) if len(mask) else None,
            "profile_equal_fraction": _profile_fraction(
                pairs, primary, full_mask, replicates, seed + index
            ),
        }
    attribution["M10_to_M11_direction_change"] = _joint_direction_change(
        pairs, summaries["rho_median"]
    )

    gate = summaries["modes"]["M10"]["within_20km_nearest_half"]
    innovation = gate.get("innovation_toward_isr", {})
    contribution = gate.get("contribution_toward_isr", {})
    innovation_value = innovation.get("value")
    innovation_ci = innovation.get("ci95", [None, None])
    if (
        innovation_value is None
        or innovation_value < 0.55
        or innovation_ci[0] is None
        or innovation_ci[0] < 0.50
    ):
        conclusion = "FY_Jicamarca_representativeness_conflict"
        next_step = (
            "audit source/site representativeness using observation metadata only; "
            "do not restore covariance loss or compensate with R"
        )
    elif contribution.get("value", 0.0) < 0.55:
        conclusion = "FY_innovation_usable_but_Kalman_direction_fails"
        next_step = "enter G2 train-only covariance mapping screen"
    else:
        conclusion = "FY_single_source_direction_gate_passed"
        next_step = "plan G3 controlled single-source training"
    return attribution, {
        "criterion": (
            "M10 |dh|<=20 km and nearest rho half: innovation direction >=55% "
            "with bootstrap CI lower bound >=50%"
        ),
        "passed": conclusion != "FY_Jicamarca_representativeness_conflict",
        "conclusion": conclusion,
        "next_step": next_step,
    }


def _top_conflict_profiles(pairs, metadata, limit=20):
    selected = (
        (pairs["mode"] == "M10")
        & (pairs["support"] == "within_20km")
        & pairs["high_confidence"]
        & (~pairs["innovation_toward"])
    )
    if not selected.any():
        return []
    profile_ids = np.unique(pairs["profile_id"][selected])
    rows = []
    for profile_id in profile_ids:
        mask = selected & (pairs["profile_id"] == profile_id)
        position = np.searchsorted(metadata["profile_id"], profile_id)
        rows.append({
            "profile_id": int(profile_id),
            "satellite": str(metadata["satellite"][position]),
            "date": int(metadata["date_code"][position]),
            "original_path": str(metadata["path"][position]),
            "conflict_pairs": int(mask.sum()),
            "conflict_precision_mass": float(
                np.sum(pairs["precision_mass"][mask])
            ),
            "innovation_mean": float(np.mean(pairs["innovation"][mask])),
            "desired_mean": float(np.mean(pairs["desired"][mask])),
            "rho_mean": float(np.mean(pairs["rho"][mask])),
            "vertical_distance_mean_km": float(
                np.mean(pairs["vertical_distance"][mask])
            ),
        })
    return sorted(
        rows,
        key=lambda row: (row["conflict_precision_mass"], row["conflict_pairs"]),
        reverse=True,
    )[:limit]


def _collect(config, checkpoint, batch_size, device):
    config = dict(config)
    config["model_type"] = "fsia"
    config["checkpoint_path"] = str(checkpoint.resolve())
    (
        model, sw_manager, cfg, _, iri_peak_manager, fy_index, cosmic_index
    ) = _load_model_and_managers(config, device)
    start_unix = _parse_unix(config["start_date_str"])
    records = load_jicamarca(
        config["jicamarca_dir"],
        start_unix,
        _parse_unix(config["end_date_str"]),
        alt_min=config["alt_min"],
        alt_max=config["alt_max"],
        err_ratio_max=config["err_ratio_max"],
    )

    query_saved = {key: [] for key in ("coords", "observation", "background")}
    token_saved = {}
    query_offset = 0
    determinism_max_abs = 0.0
    checked_determinism = False
    with torch.no_grad():
        for record in records:
            coords_np, observation = _record_arrays(record, start_unix)
            local_time = np.remainder(
                coords_np[:, 3] + coords_np[:, 1] / 15.0, 24.0
            )
            selected = (
                (coords_np[:, 2] >= 120.0)
                & (coords_np[:, 2] < 300.0)
                & ((local_time < 6.0) | (local_time >= 18.0))
            )
            coords_np = coords_np[selected]
            observation = observation[selected]
            for start in range(0, len(coords_np), batch_size):
                chunk_np = coords_np[start:start + batch_size]
                chunk_observation = observation[start:start + batch_size]
                coords = torch.from_numpy(chunk_np).to(device)
                sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
                iri_peak = (
                    iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None
                )
                fy = attach_observation_background(
                    query_observation_payload(fy_index, coords, device),
                    model, sw_manager, iri_peak_manager,
                )
                cosmic = attach_observation_background(
                    query_observation_payload(cosmic_index, coords, device),
                    model, sw_manager, iri_peak_manager,
                )
                _validate_payload("Jicamarca", "FY", fy)
                _validate_payload("Jicamarca", "COSMIC", cosmic)
                prediction_m10, _, _, _, extras_m10 = model(
                    coords, sw_seq, iri_peak=iri_peak, observations_fy=fy
                )
                prediction_m11, _, _, _, extras_m11 = model(
                    coords, sw_seq, iri_peak=iri_peak,
                    observations_fy=fy, observations_cosmic=cosmic,
                )
                if not checked_determinism:
                    repeated, _, _, _, repeated_extras = model(
                        coords, sw_seq, iri_peak=iri_peak, observations_fy=fy
                    )
                    determinism_max_abs = max(
                        float((prediction_m10 - repeated).abs().max()),
                        float(
                            (
                                extras_m10["K_FY"]
                                - repeated_extras["K_FY"]
                            ).abs().max()
                        ),
                    )
                    checked_determinism = True
                background = extras_m10["ne_bkg"].squeeze(-1).cpu().numpy()
                for mode, extras in (
                    ("M10", extras_m10), ("M11", extras_m11)
                ):
                    rows = _observation_diagnostics(
                        fy, extras, "FY",
                        np.ones(len(chunk_np), dtype=bool),
                        query_offset, chunk_observation, background,
                    )
                    if rows is not None:
                        rows["mode"] = np.full(len(rows["profile_id"]), mode)
                        for key, values in rows.items():
                            token_saved.setdefault(key, []).append(values)
                query_saved["coords"].append(chunk_np)
                query_saved["observation"].append(chunk_observation)
                query_saved["background"].append(background)
                query_offset += len(chunk_np)
                if not (
                    torch.isfinite(prediction_m10).all()
                    and torch.isfinite(prediction_m11).all()
                ):
                    raise ValueError("G1 inference produced non-finite predictions")
    query_arrays = {
        key: np.concatenate(values) for key, values in query_saved.items()
    }
    token_arrays = {
        key: np.concatenate(values) for key, values in token_saved.items()
    }
    return query_arrays, token_arrays, cfg, determinism_max_abs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    query, tokens, cfg, determinism_max_abs = _collect(
        ISR_CONFIG, args.checkpoint, args.batch_size, device
    )
    metadata = _profile_metadata(cfg["fy_profile_index_path"])
    pair_by_mode_support = {
        f"{mode}_{support}": _attach_profile_metadata(
            _aggregate_profile_query(
                tokens, query["coords"], mode, support
            ),
            metadata,
        )
        for mode in MODES
        for support in ("all", "same_layer", "within_20km")
    }
    pairs = _combine_pair_arrays(pair_by_mode_support)
    summaries = _summaries(pairs, args.bootstrap, args.seed)
    attribution, gate = _attribution_and_gate(
        pairs, summaries, args.bootstrap, args.seed
    )
    top_conflicts = _top_conflict_profiles(pairs, metadata)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "fy_jicamarca_representativeness_pairs.npz"
    report_path = output / "fy_jicamarca_representativeness_report.json"
    csv_path = output / "top_conflict_profiles.csv"
    serializable_pairs = {
        key: np.asarray(value)
        for key, value in pairs.items()
        if key != "original_path"
    }
    _deterministic_npz(npz_path, serializable_pairs)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(top_conflicts[0]) if top_conflicts else [
            "profile_id", "satellite", "date", "original_path",
            "conflict_pairs", "conflict_precision_mass", "innovation_mean",
            "desired_mean", "rho_mean", "vertical_distance_mean_km",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_conflicts)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "scope": (
            "Jicamarca night 120-300 km; FY profile is the analysis unit; "
            "M10 and M11 are reported separately"
        ),
        "semantics": {
            "physical_innovation": "FY log10Ne - Background at FY coordinate",
            "desired_increment": "ISR log10Ne - Background at ISR coordinate",
            "bootstrap": (
                "two-stage date/profile bootstrap; 1000 replicates, seed=42"
            ),
            "high_confidence": (
                "|physical innovation|>=0.05 dex and "
                "|desired increment|>=0.05 dex and precision>0"
            ),
            "within_20km": (
                "precision-weighted mean absolute FY/ISR altitude difference"
            ),
        },
        "identity": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "run_manifest": str(args.manifest.resolve()),
            "run_manifest_sha256": _sha256(args.manifest),
            "fy_data": manifest["data_identity"]["fy_path"],
            "fy_profile_index": manifest["data_identity"][
                "fy_profile_index_path"
            ],
            "script_sha256": _sha256(Path(__file__)),
        },
        "n_query_points": int(len(query["coords"])),
        "n_token_rows": int(len(tokens["profile_id"])),
        "n_profile_query_pairs": int(len(pairs["profile_id"])),
        "determinism_max_abs": determinism_max_abs,
        "summaries": summaries,
        "primary_attribution": attribution,
        "gate": gate,
        "top_conflict_profiles": top_conflicts,
        "outputs": {
            "pairs_npz": npz_path.name,
            "pairs_npz_sha256": _sha256(npz_path),
            "top_conflict_csv": csv_path.name,
            "top_conflict_csv_sha256": _sha256(csv_path),
        },
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    print(json.dumps({
        "output": str(output),
        "gate": gate,
        "primary": summaries["modes"]["M10"][
            "within_20km_nearest_half"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
