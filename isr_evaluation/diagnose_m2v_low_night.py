"""Train-only diagnosis of the M2-V low-altitude nighttime Jicamarca error."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inr_modules.data_managers.FY_dataloader import _great_circle_km
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    observation_query_coverage,
    query_observation_payload,
)
from isr_evaluation.isr_loader import load_jicamarca
from isr_evaluation.main_isr_eval import (
    CONFIG as ISR_CONFIG,
    _load_model_and_managers,
    _parse_unix,
)


DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints_fsia" / "run66-m2v-continuous-physical-letkf-restart2"
    / "epoch_13_model.pth"
)
DEFAULT_OUTPUT = (
    ROOT / "isr_validation_outputs" / "run66-m2v-restart2-epoch13-isr"
    / "Jicamarca" / "low_night_diagnosis"
)
BACKGROUND_PREFIXES = (
    "iri_proxy.", "iri_align_net.", "sw_encoder.", "sw_freq_branch.",
    "sw_gate.", "background_decoder.",
)
ALT_EDGES = np.asarray([120.0, 200.0, 300.0, 500.0])
ALT_LABELS = ("120-200", "200-300", "300-500")


def _metrics(prediction, observation):
    prediction = np.asarray(prediction, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(observation)
    if not valid.any():
        return {"n": 0, "rmse": None, "bias": None, "mae": None}
    error = prediction[valid] - observation[valid]
    return {
        "n": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
    }


def _profile_metadata(index_path, data, train_days):
    with np.load(index_path, allow_pickle=False) as index:
        ids = np.asarray(index["profile_id"], dtype=np.int64)
        passed = np.asarray(index["pass_profile"], dtype=bool)
        starts = np.asarray(index["output_start"], dtype=np.int64)
        ends = np.asarray(index["output_end"], dtype=np.int64)
        if "representative_time" in index.files:
            times = np.asarray(index["representative_time"], dtype=np.float64)
        else:
            times = np.asarray([
                np.median(data[start:end, 3]) if start >= 0 and end > start else np.nan
                for start, end in zip(starts, ends)
            ])
        if "representative_lat" in index.files:
            latitudes = np.asarray(index["representative_lat"], dtype=np.float64)
            longitudes = np.asarray(index["representative_lon"], dtype=np.float64)
        else:
            latitudes = np.asarray([
                data[start, 0] if start >= 0 and end > start else np.nan
                for start, end in zip(starts, ends)
            ])
            longitudes = np.asarray([
                data[start, 1] if start >= 0 and end > start else np.nan
                for start, end in zip(starts, ends)
            ])
    finite_time = np.isfinite(times)
    profile_day = np.full(len(times), -1, dtype=np.int64)
    profile_day[finite_time] = np.floor(times[finite_time] / 24.0).astype(np.int64)
    valid = (
        passed & (starts >= 0) & (ends > starts) & finite_time
        & np.isin(profile_day, train_days)
    )
    return {
        "id": ids[valid],
        "start": starts[valid],
        "end": ends[valid],
        "lat": latitudes[valid],
        "lon": longitudes[valid],
    }


def _sample_train_profiles(data_path, index_path, train_days, station_lat,
                           station_lon, seed, max_global_profiles=12000,
                           max_regional_profiles=5000):
    data = np.load(data_path, mmap_mode="r")
    meta = _profile_metadata(index_path, data, train_days)
    distance = _great_circle_km(
        station_lat, station_lon, meta["lat"], meta["lon"])
    regional = np.flatnonzero(distance < 1800.0)
    rng = np.random.default_rng(seed)
    global_selected = rng.choice(
        len(meta["id"]), min(max_global_profiles, len(meta["id"])), replace=False)
    if len(regional) > max_regional_profiles:
        regional = rng.choice(
            regional, max_regional_profiles, replace=False)
    selected = np.unique(np.concatenate([global_selected, regional]))

    rows = []
    profile_ids = []
    profile_regional = []
    regional_set = set(regional.tolist())
    for profile_index in selected:
        start = int(meta["start"][profile_index])
        end = int(meta["end"][profile_index])
        positions = np.linspace(start, end - 1, 8).round().astype(np.int64)
        rows.append(positions)
        profile_ids.extend([int(meta["id"][profile_index])] * len(positions))
        profile_regional.extend([profile_index in regional_set] * len(positions))
    row_index = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    values = np.asarray(data[row_index, :5], dtype=np.float32)
    return values, np.asarray(profile_ids), np.asarray(profile_regional), np.sort(meta["id"])


@torch.no_grad()
def _background_predictions(model, sw_manager, iri_peak_manager, coords_np,
                            device, batch_size=4096):
    iri_parts = []
    background_parts = []
    raw_residual_parts = []
    residual_parts = []
    gate_parts = []
    for start in range(0, len(coords_np), batch_size):
        coords = torch.from_numpy(coords_np[start:start + batch_size]).to(device)
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        peak = (iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
        encoded = model.encode_background(coords, sw_seq, iri_peak=peak)
        iri_parts.append(encoded["ne_iri"].flatten().cpu().numpy())
        background_parts.append(encoded["ne_bkg"].flatten().cpu().numpy())
        raw_residual_parts.append(
            encoded["background_residual_raw"].flatten().cpu().numpy())
        residual_parts.append(
            encoded["background_residual"].flatten().cpu().numpy())
        gate_parts.append(
            encoded["background_trust_gate"].flatten().cpu().numpy())
    return (
        np.concatenate(iri_parts), np.concatenate(background_parts),
        np.concatenate(raw_residual_parts), np.concatenate(residual_parts),
        np.concatenate(gate_parts),
    )


def _satellite_background_audit(model, sw_manager, iri_peak_manager, cfg,
                                train_days, station_lat, station_lon, device):
    result = {}
    allowed = {}
    for source, data_key, index_key, seed in (
        ("FY", "fy_path", "fy_profile_index_path", 42),
        ("COSMIC", "cosmic_path", "cosmic_profile_index_path", 43),
    ):
        values, profile_ids, regional, allowed[source] = _sample_train_profiles(
            cfg[data_key], cfg[index_key], train_days, station_lat, station_lon,
            seed=seed,
        )
        iri, background, raw_residual, residual, trust_gate = _background_predictions(
            model, sw_manager, iri_peak_manager, values[:, :4], device)
        local_time = np.remainder(values[:, 3] + values[:, 1] / 15.0, 24.0)
        night = (local_time < 6.0) | (local_time >= 18.0)
        altitude_index = np.searchsorted(ALT_EDGES[1:-1], values[:, 2], side="right")
        source_result = {
            "sampled_profiles": int(np.unique(profile_ids).size),
            "sampled_points": int(len(values)),
            "regional_profiles": int(np.unique(profile_ids[regional]).size),
            "cells": {},
            "trust_gate": {
                "mean": float(np.mean(trust_gate)),
                "p05": float(np.quantile(trust_gate, 0.05)),
                "p50": float(np.quantile(trust_gate, 0.50)),
                "p95": float(np.quantile(trust_gate, 0.95)),
                "strict_suppression_fraction": float(
                    np.mean(trust_gate <= 1e-8)),
                "raw_residual_rms": float(np.sqrt(np.mean(raw_residual ** 2))),
                "gated_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
                "raw_residual_max_abs": float(np.max(np.abs(raw_residual))),
                "gated_residual_max_abs": float(np.max(np.abs(residual))),
            },
        }
        for scope, scope_mask in (
            ("global", np.ones(len(values), dtype=bool)),
            ("within_1800km_of_jicamarca", regional),
        ):
            source_result["cells"][scope] = {}
            for altitude_cell, altitude_label in enumerate(ALT_LABELS):
                for period, period_mask in (("night", night), ("day", ~night)):
                    selected = scope_mask & (altitude_index == altitude_cell) & period_mask
                    raw = _metrics(iri[selected], values[selected, 4])
                    m00 = _metrics(background[selected], values[selected, 4])
                    source_result["cells"][scope][f"{altitude_label}_{period}"] = {
                        "profiles": int(np.unique(profile_ids[selected]).size),
                        "raw_iri": raw,
                        "M00": m00,
                        "rmse_change_M00_minus_raw": (
                            None if raw["rmse"] is None or m00["rmse"] is None
                            else float(m00["rmse"] - raw["rmse"])
                        ),
                    }
        result[source] = source_result
    return result, allowed


def _jicamarca_low_night_arrays(records, start_unix, train_days,
                                night_only=True, return_day=False):
    coords_parts = []
    observation_parts = []
    uncertainty_parts = []
    day_parts = []
    for record in records:
        altitude = np.tile(
            np.asarray(record["alt_1d"])[:, None], (1, len(record["ts_1d"])))
        relative_hour = np.tile(
            ((np.asarray(record["ts_1d"]) - start_unix) / 3600.0)[None, :],
            (len(record["alt_1d"]), 1),
        )
        latitude = np.full_like(altitude, record["lat"], dtype=np.float64)
        longitude = np.full_like(altitude, record["lon"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            observation = np.log10(np.asarray(record["ne_2d"], dtype=np.float64))
            relative_uncertainty = (
                np.asarray(record["dne_2d"], dtype=np.float64)
                / np.asarray(record["ne_2d"], dtype=np.float64))
        local_time = np.remainder(relative_hour + longitude / 15.0, 24.0)
        day = np.floor(relative_hour / 24.0).astype(np.int64)
        night_mask = (local_time < 6.0) | (local_time >= 18.0)
        valid = (
            np.isfinite(observation) & (altitude >= 120.0) & (altitude < 300.0)
            & (night_mask if night_only else np.ones_like(night_mask, dtype=bool))
            & np.isin(day, train_days)
        )
        coords_parts.append(np.column_stack([
            latitude[valid], longitude[valid], altitude[valid], relative_hour[valid]
        ]).astype(np.float32))
        observation_parts.append(observation[valid].astype(np.float32))
        uncertainty_parts.append(relative_uncertainty[valid].astype(np.float32))
        if return_day:
            day_parts.append(day[valid])
    result = (
        np.concatenate(coords_parts), np.concatenate(observation_parts),
        np.concatenate(uncertainty_parts),
    )
    return result + ((np.concatenate(day_parts),) if return_day else ())


@torch.no_grad()
def _background_jicamarca_gate(model, sw_manager, iri_peak_manager, records,
                               start_unix, train_days, device):
    coords, observation, uncertainty, days = _jicamarca_low_night_arrays(
        records, start_unix, train_days, night_only=False, return_day=True)
    raw, background, raw_residual, residual, trust_gate = _background_predictions(
        model, sw_manager, iri_peak_manager, coords, device)
    local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    result = {'cells': {}, 'all_finite': bool(
        np.isfinite(raw).all() and np.isfinite(background).all())}
    passed = result['all_finite']
    for altitude_label, lower, upper in (
            ('120-200', 120.0, 200.0), ('200-300', 200.0, 300.0)):
        in_altitude = (coords[:, 2] >= lower) & (coords[:, 2] < upper)
        for period, period_mask in (('night', night), ('day', ~night)):
            selected = in_altitude & period_mask
            metrics_raw = _metrics(raw[selected], observation[selected])
            metrics_m00 = _metrics(background[selected], observation[selected])
            n_dates = int(np.unique(days[selected]).size)
            cell_passed = (
                metrics_raw['n'] >= 100 and metrics_m00['n'] >= 100
                and n_dates >= 3
                and metrics_m00['rmse'] is not None
                and metrics_raw['rmse'] is not None
                and metrics_m00['bias'] is not None
                and metrics_raw['bias'] is not None
                and metrics_m00['rmse'] <= metrics_raw['rmse']
                and abs(metrics_m00['bias']) <= abs(metrics_raw['bias']))
            passed = passed and cell_passed
            result['cells'][f'{altitude_label}_{period}'] = {
                'points': metrics_m00['n'],
                'dates': n_dates,
                'raw_iri': metrics_raw,
                'M00': metrics_m00,
                'rmse_change_M00_minus_raw': (
                    None if metrics_raw['rmse'] is None or metrics_m00['rmse'] is None
                    else float(metrics_m00['rmse'] - metrics_raw['rmse'])),
                'passed_hard_gate': cell_passed,
                'trust_gate_mean': float(np.mean(trust_gate[selected]))
                if selected.any() else None,
                'raw_residual_rms': float(np.sqrt(np.mean(
                    raw_residual[selected] ** 2))) if selected.any() else None,
                'gated_residual_rms': float(np.sqrt(np.mean(
                    residual[selected] ** 2))) if selected.any() else None,
            }
            result['cells'][f'{altitude_label}_{period}']['by_date'] = {
                str(int(day)): {
                    'raw_iri': _metrics(
                        raw[selected & (days == day)],
                        observation[selected & (days == day)]),
                    'M00': _metrics(
                        background[selected & (days == day)],
                        observation[selected & (days == day)]),
                }
                for day in np.unique(days[selected])
            }
    result['passed_hard_gate'] = passed
    finite_gate = np.isfinite(trust_gate)
    result['trust_gate'] = {
        'mean': float(np.mean(trust_gate[finite_gate])) if finite_gate.any() else None,
        'p05': float(np.quantile(trust_gate[finite_gate], 0.05))
        if finite_gate.any() else None,
        'p50': float(np.quantile(trust_gate[finite_gate], 0.50))
        if finite_gate.any() else None,
        'p95': float(np.quantile(trust_gate[finite_gate], 0.95))
        if finite_gate.any() else None,
        'strict_suppression_fraction': float(np.mean(
            trust_gate[finite_gate] <= 1e-8)) if finite_gate.any() else None,
        'raw_residual_rms': float(np.sqrt(np.mean(raw_residual ** 2))),
        'gated_residual_rms': float(np.sqrt(np.mean(residual ** 2))),
        'raw_residual_max_abs': float(np.max(np.abs(raw_residual))),
        'gated_residual_max_abs': float(np.max(np.abs(residual))),
    }
    return result


def _append_token_rows(storage, payload, extras, suffix, query_offset, query_coords,
                       query_observation, query_background):
    query_index = payload["query_index"].cpu().numpy().astype(np.int64)
    if not len(query_index):
        return
    observation_coords = payload["coords"].cpu().numpy()
    precision = extras[f"precision_{suffix}"].cpu().numpy()
    innovation = extras[f"innov_{suffix}"].cpu().numpy()
    gain = extras[f"K_{suffix}"].cpu().numpy()
    contribution = gain * innovation
    fields = {
        "source": np.full(len(query_index), suffix),
        "query_index": query_index + query_offset,
        "query_altitude": query_coords[query_index, 2],
        "query_local_time": np.remainder(
            query_coords[query_index, 3] + query_coords[query_index, 1] / 15.0, 24.0),
        "query_observation": query_observation[query_index],
        "query_background": query_background[query_index],
        "observation_altitude": observation_coords[:, 2],
        "observation_local_time": np.remainder(
            observation_coords[:, 3] + observation_coords[:, 1] / 15.0, 24.0),
        "precision": precision,
        "innovation": innovation,
        "gain": gain,
        "contribution": contribution,
        "localization": payload["localization_weight"].cpu().numpy(),
        "space_distance_km": payload["space_distance_km"].cpu().numpy(),
        "time_distance_hours": payload["time_distance_hours"].cpu().numpy(),
    }
    for key, value in fields.items():
        storage.setdefault(key, []).append(value)


def _payload_variant(payload, source, variance_tables, low_observations_only=False,
                     stratified_r=False):
    result = dict(payload)
    if low_observations_only:
        result["valid_mask"] = payload["valid_mask"] & (payload["coords"][:, 2] < 300.0)
    if stratified_r:
        coords = payload["coords"]
        altitude_index = torch.bucketize(
            coords[:, 2].contiguous(), coords.new_tensor([200.0, 300.0]))
        local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
        day_index = ((local_time >= 6.0) & (local_time < 18.0)).long()
        table = coords.new_tensor(variance_tables[source]["table"])
        global_variance = coords.new_tensor(
            variance_tables[source]["global_variance"])
        result["representativeness_weight"] = (
            global_variance / table[altitude_index, day_index]
        )
    return result


@torch.no_grad()
def _letkf_audit(model, sw_manager, iri_peak_manager, fy_index, cosmic_index,
                 allowed, variance_tables, coords_np, observation,
                 isr_relative_uncertainty, device, batch_size=256):
    query = {key: [] for key in (
        "coords", "observation", "ISR_relative_uncertainty",
        "raw_iri", "M00", "M11", "delta",
        "M11_stratified_R", "delta_stratified_R",
        "M11_low_observations_only", "delta_low_observations_only",
        "M11_low_observations_stratified_R",
        "delta_low_observations_stratified_R",
        "delta_FY", "delta_COSMIC", "FY_coverage", "COSMIC_coverage",
    )}
    token = {}
    query_offset = 0
    for start in range(0, len(coords_np), batch_size):
        chunk_np = coords_np[start:start + batch_size]
        chunk_observation = observation[start:start + batch_size]
        chunk_uncertainty = isr_relative_uncertainty[start:start + batch_size]
        coords = torch.from_numpy(chunk_np).to(device)
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        peak = (iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
        fy = query_observation_payload(
            fy_index, coords, device, allowed_profile_ids=allowed["FY"])
        cosmic = query_observation_payload(
            cosmic_index, coords, device, allowed_profile_ids=allowed["COSMIC"])
        fy = attach_observation_background(
            fy, model, sw_manager, iri_peak_manager)
        cosmic = attach_observation_background(
            cosmic, model, sw_manager, iri_peak_manager)
        prediction, _, _, delta, extras = model(
            coords, sw_seq, iri_peak=peak,
            observations_fy=fy, observations_cosmic=cosmic,
        )
        variants = {}
        for name, low_only, stratified in (
            ("stratified_R", False, True),
            ("low_observations_only", True, False),
            ("low_observations_stratified_R", True, True),
        ):
            variant_fy = _payload_variant(
                fy, "FY", variance_tables, low_only, stratified)
            variant_cosmic = _payload_variant(
                cosmic, "COSMIC", variance_tables, low_only, stratified)
            variant_prediction, _, _, variant_delta, _ = model(
                coords, sw_seq, iri_peak=peak,
                observations_fy=variant_fy,
                observations_cosmic=variant_cosmic,
            )
            variants[name] = (
                variant_prediction.flatten().cpu().numpy(),
                variant_delta.flatten().cpu().numpy(),
            )
        background = extras["ne_bkg"].flatten().cpu().numpy()
        query["coords"].append(chunk_np)
        query["observation"].append(chunk_observation)
        query["ISR_relative_uncertainty"].append(chunk_uncertainty)
        query["raw_iri"].append(extras["ne_iri"].flatten().cpu().numpy())
        query["M00"].append(background)
        query["M11"].append(prediction.flatten().cpu().numpy())
        query["delta"].append(delta.flatten().cpu().numpy())
        for name, (variant_prediction, variant_delta) in variants.items():
            query[f"M11_{name}"].append(variant_prediction)
            query[f"delta_{name}"].append(variant_delta)
        query["delta_FY"].append(extras["update_FY"].flatten().cpu().numpy())
        query["delta_COSMIC"].append(
            extras["update_COSMIC"].flatten().cpu().numpy())
        query["FY_coverage"].append(
            observation_query_coverage(fy).cpu().numpy())
        query["COSMIC_coverage"].append(
            observation_query_coverage(cosmic).cpu().numpy())
        _append_token_rows(
            token, fy, extras, "FY", query_offset, chunk_np,
            chunk_observation, background)
        _append_token_rows(
            token, cosmic, extras, "COSMIC", query_offset, chunk_np,
            chunk_observation, background)
        query_offset += len(chunk_np)
    return (
        {key: np.concatenate(parts) for key, parts in query.items()},
        {key: np.concatenate(parts) for key, parts in token.items()},
    )


def _query_summary(query):
    coords = query["coords"]
    local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    desired = query["observation"] - query["M00"]
    delta = query["delta"]
    nonzero = np.abs(delta) > 1e-10
    result = {}
    masks = {
        "all_train_day_low_night": np.ones(len(coords), dtype=bool),
        "120_200km": coords[:, 2] < 200.0,
        "200_300km": coords[:, 2] >= 200.0,
        "deep_night": (local_time < 4.0) | (local_time >= 20.0),
        "night_boundary": ((local_time >= 4.0) & (local_time < 6.0))
                          | ((local_time >= 18.0) & (local_time < 20.0)),
        "120_200km_ISR_relerr_le_0.10": (
            (coords[:, 2] < 200.0) & (query["ISR_relative_uncertainty"] <= 0.10)),
        "120_200km_ISR_relerr_le_0.20": (
            (coords[:, 2] < 200.0) & (query["ISR_relative_uncertainty"] <= 0.20)),
    }
    for name, selected in masks.items():
        raw = _metrics(query["raw_iri"][selected], query["observation"][selected])
        m00 = _metrics(query["M00"][selected], query["observation"][selected])
        analyses = {
            name: _metrics(query[name][selected], query["observation"][selected])
            for name in (
                "M11", "M11_stratified_R", "M11_low_observations_only",
                "M11_low_observations_stratified_R",
            )
        }
        comparable = selected & nonzero & (np.abs(desired) > 1e-10)
        result[name] = {
            "raw_iri": raw,
            "M00": m00,
            **analyses,
            "rmse_change_M00_minus_raw": float(m00["rmse"] - raw["rmse"]),
            **{
                f"rmse_change_{analysis_name}_minus_M00": float(
                    metrics["rmse"] - m00["rmse"])
                for analysis_name, metrics in analyses.items()
            },
            "mean_desired_increment": float(np.mean(desired[selected])),
            "mean_M11_increment": float(np.mean(delta[selected])),
            "toward_ISR_fraction": (
                float(np.mean(delta[comparable] * desired[comparable] > 0))
                if comparable.any() else None
            ),
            "pointwise_improvement_fraction_M11_vs_M00": float(np.mean(
                np.abs(query["M11"][selected] - query["observation"][selected])
                < np.abs(query["M00"][selected] - query["observation"][selected])
            )),
            "FY_coverage": float(np.mean(query["FY_coverage"][selected])),
            "COSMIC_coverage": float(np.mean(query["COSMIC_coverage"][selected])),
        }
    result["joint_contribution_max_abs_error"] = float(np.max(np.abs(
        query["delta_FY"] + query["delta_COSMIC"] - query["delta"]
    )))
    return result


def _token_summary(token):
    result = {}
    desired = token["query_observation"] - token["query_background"]
    observation_day = (
        (token["observation_local_time"] >= 6.0)
        & (token["observation_local_time"] < 18.0)
    )
    altitude_index = np.searchsorted(
        ALT_EDGES[1:-1], token["observation_altitude"], side="right")
    for source in ("FY", "COSMIC"):
        source_mask = token["source"] == source
        source_result = {"altitude": {}, "by_query_altitude": {}}
        total_abs = np.sum(np.abs(token["contribution"][source_mask]))
        total_precision = np.sum(token["precision"][source_mask])
        for altitude_cell, altitude_label in enumerate(ALT_LABELS):
            selected = source_mask & (altitude_index == altitude_cell)
            precision = token["precision"][selected]
            contribution = token["contribution"][selected]
            comparable = selected & (np.abs(desired) > 1e-10) & (
                np.abs(token["contribution"]) > 1e-12)
            source_result["altitude"][altitude_label] = {
                "tokens": int(selected.sum()),
                "precision_sum": float(np.sum(precision)),
                "precision_share": (
                    float(np.sum(precision) / total_precision) if total_precision > 0 else None),
                "signed_contribution_sum": float(np.sum(contribution)),
                "absolute_contribution_sum": float(np.sum(np.abs(contribution))),
                "absolute_contribution_share": (
                    float(np.sum(np.abs(contribution)) / total_abs)
                    if total_abs > 0 else None
                ),
                "precision_weighted_innovation": (
                    float(np.sum(precision * token["innovation"][selected])
                          / np.sum(precision)) if np.sum(precision) > 0 else None
                ),
                "toward_ISR_fraction": (
                    float(np.mean(
                        desired[comparable] * token["contribution"][comparable] > 0
                    )) if comparable.any() else None
                ),
            }
        query_altitude_index = np.searchsorted(
            ALT_EDGES[1:-1], token["query_altitude"], side="right")
        for query_cell, query_label in enumerate(ALT_LABELS[:2]):
            query_selected = source_mask & (query_altitude_index == query_cell)
            query_abs = np.sum(np.abs(token["contribution"][query_selected]))
            query_precision = np.sum(token["precision"][query_selected])
            source_result["by_query_altitude"][query_label] = {}
            for observation_cell, observation_label in enumerate(ALT_LABELS):
                selected = query_selected & (altitude_index == observation_cell)
                source_result["by_query_altitude"][query_label][observation_label] = {
                    "tokens": int(selected.sum()),
                    "precision_share": (
                        float(np.sum(token["precision"][selected]) / query_precision)
                        if query_precision > 0 else None
                    ),
                    "signed_contribution_sum": float(np.sum(
                        token["contribution"][selected])),
                    "absolute_contribution_share": (
                        float(np.sum(np.abs(token["contribution"][selected])) / query_abs)
                        if query_abs > 0 else None
                    ),
                }
        day_selected = source_mask & observation_day
        source_result["day_observation_precision_share_for_night_queries"] = (
            float(np.sum(token["precision"][day_selected]) / total_precision)
            if total_precision > 0 else None
        )
        source_result["day_observation_absolute_contribution_share_for_night_queries"] = (
            float(np.sum(np.abs(token["contribution"][day_selected])) / total_abs)
            if total_abs > 0 else None
        )
        source_result["tokens"] = int(source_mask.sum())
        source_result["unique_queries"] = int(np.unique(
            token["query_index"][source_mask]).size)
        query_low = token["query_altitude"] < 200.0
        for field, edges, output_key in (
            ("space_distance_km", (0.0, 450.0, 900.0, 1350.0, 1800.0),
             "space_distance_for_120_200_queries"),
            ("time_distance_hours", (0.0, 0.375, 0.75, 1.125, 1.5),
             "time_distance_for_120_200_queries"),
        ):
            selected_scope = source_mask & query_low
            total_abs_scope = np.sum(np.abs(token["contribution"][selected_scope]))
            total_precision_scope = np.sum(token["precision"][selected_scope])
            rows = []
            for lower, upper in zip(edges[:-1], edges[1:]):
                selected = (
                    selected_scope & (token[field] >= lower) & (token[field] < upper))
                rows.append({
                    "range": [lower, upper],
                    "tokens": int(selected.sum()),
                    "precision_share": (
                        float(np.sum(token["precision"][selected]) / total_precision_scope)
                        if total_precision_scope > 0 else None
                    ),
                    "signed_contribution_sum": float(np.sum(
                        token["contribution"][selected])),
                    "absolute_contribution_share": (
                        float(np.sum(np.abs(token["contribution"][selected]))
                              / total_abs_scope)
                        if total_abs_scope > 0 else None
                    ),
                })
            source_result[output_key] = rows
        result[source] = source_result
    return result


def _isr_uncertainty_summary(records, start_unix, train_days):
    altitude_parts = []
    local_time_parts = []
    relative_error_parts = []
    for record in records:
        altitude = np.tile(
            np.asarray(record["alt_1d"])[:, None], (1, len(record["ts_1d"])))
        relative_hour = np.tile(
            ((np.asarray(record["ts_1d"]) - start_unix) / 3600.0)[None, :],
            (len(record["alt_1d"]), 1),
        )
        longitude = np.full_like(altitude, record["lon"], dtype=np.float64)
        electron_density = np.asarray(record["ne_2d"], dtype=np.float64)
        density_error = np.asarray(record["dne_2d"], dtype=np.float64)
        relative_error = np.divide(
            density_error, electron_density,
            out=np.full_like(density_error, np.nan),
            where=electron_density > 0,
        )
        day = np.floor(relative_hour / 24.0).astype(np.int64)
        valid = (
            np.isfinite(relative_error) & np.isin(day, train_days)
            & (relative_error > 0.0) & (relative_error < 0.5)
        )
        altitude_parts.append(altitude[valid])
        local_time_parts.append(np.remainder(
            relative_hour[valid] + longitude[valid] / 15.0, 24.0))
        relative_error_parts.append(relative_error[valid])
    altitude = np.concatenate(altitude_parts)
    local_time = np.concatenate(local_time_parts)
    relative_error = np.concatenate(relative_error_parts)
    night = (local_time < 6.0) | (local_time >= 18.0)
    result = {}
    for altitude_cell, altitude_label in enumerate(ALT_LABELS):
        in_altitude = (
            (altitude >= ALT_EDGES[altitude_cell])
            & (altitude < ALT_EDGES[altitude_cell + 1]))
        for period, period_mask in (("night", night), ("day", ~night)):
            selected = in_altitude & period_mask
            values = relative_error[selected]
            result[f"{altitude_label}_{period}"] = {
                "n": int(len(values)),
                "median_dNe_over_Ne": float(np.median(values)),
                "p90_dNe_over_Ne": float(np.quantile(values, 0.90)),
                "p95_dNe_over_Ne": float(np.quantile(values, 0.95)),
            }
    return result


def _r_calibration_summary(checkpoint):
    path = checkpoint.parent / "r_calibration.json"
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    result = {}
    for source in ("FY", "COSMIC"):
        source_report = report[source]
        global_variance = float(source_report["global"]["variance"])
        rows = {}
        for altitude_label, cells in zip(ALT_LABELS, source_report["cells"]):
            for period, cell in zip(("night", "day"), cells):
                final_variance = float(cell["final_variance"])
                rows[f"{altitude_label}_{period}"] = {
                    "points": int(cell["point_count"]),
                    "profiles": int(cell["profile_count"]),
                    "sigma_dex": float(cell["final_sigma"]),
                    "global_sigma_dex": float(np.sqrt(global_variance)),
                    "global_R_precision_overweight_factor": float(
                        final_variance / global_variance),
                }
        result[source] = rows
    return result


def _r_variance_tables(checkpoint):
    with (checkpoint.parent / "r_calibration.json").open(
            encoding="utf-8") as stream:
        report = json.load(stream)
    return {
        source: {
            "global_variance": float(report[source]["global"]["variance"]),
            "table": np.asarray(
                report[source]["calibrated_variance_table"], dtype=np.float64),
        }
        for source in ("FY", "COSMIC")
    }


def _background_seed_identity(checkpoint, seed_path):
    current = torch.load(checkpoint, map_location="cpu", weights_only=True)
    seed = torch.load(seed_path, map_location="cpu", weights_only=True)
    names = [
        name for name in current
        if name.startswith(BACKGROUND_PREFIXES) and name in seed
    ]
    maximum = max(
        float(torch.max(torch.abs(current[name] - seed[name])))
        for name in names
    )
    return {
        "compared_tensors": len(names),
        "max_abs_parameter_difference": maximum,
        "exactly_equal": maximum == 0.0,
        "seed_checkpoint": str(Path(seed_path).resolve()),
    }


def _save_figure(report, output):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    query = report["jicamarca_train_only_LETKF"]["query"]
    names = ("Raw IRI", "M00", "M11")
    rmse = [
        query["all_train_day_low_night"][key]["rmse"]
        for key in ("raw_iri", "M00", "M11")
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(names, rmse, color=("#0072B2", "#E69F00", "#009E73"))
    axes[0].set_ylabel("RMSE (dex)")
    axes[0].set_title("Jicamarca 120-300 km nighttime")

    for source, color in (("FY", "#0072B2"), ("COSMIC", "#D55E00")):
        rows = report["R_calibration"][source]
        axes[1].plot(
            range(3), [rows[f"{label}_night"]["sigma_dex"] for label in ALT_LABELS],
            marker="o", label=f"{source} night", color=color,
        )
        axes[1].plot(
            range(3), [rows[f"{label}_day"]["sigma_dex"] for label in ALT_LABELS],
            marker="x", linestyle="--", label=f"{source} day", color=color,
        )
    axes[1].set_xticks(range(3), ALT_LABELS)
    axes[1].set_ylabel("Robust residual sigma (dex)")
    axes[1].set_title("Train-only M00-satellite residual scale")
    axes[1].legend(fontsize=8)

    width = 0.35
    for offset, source, color in (
        (-width / 2, "FY", "#0072B2"), (width / 2, "COSMIC", "#D55E00")):
        rows = report["jicamarca_train_only_LETKF"]["tokens"][source]["altitude"]
        axes[2].bar(
            np.arange(3) + offset,
            [rows[label]["absolute_contribution_share"] for label in ALT_LABELS],
            width, label=source, color=color,
        )
    axes[2].set_xticks(range(3), ALT_LABELS)
    axes[2].set_ylabel("Absolute LETKF contribution share")
    axes[2].set_title("Observation-height contribution to low queries")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output / "m2v_low_night_diagnosis.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage", choices=("background", "analysis"), default="analysis",
        help="diagnose only Background or the complete Analysis checkpoint")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    np.random.seed(42)
    torch.manual_seed(42)

    config = dict(ISR_CONFIG)
    config["checkpoint_path"] = str(checkpoint)
    config["run_poker_flat"] = False
    config["allow_background_stage"] = args.stage == "background"
    (model, sw_manager, cfg, _, iri_peak_manager,
     fy_index, cosmic_index) = _load_model_and_managers(config, device)
    with (checkpoint.parent / "run_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    summary_path = checkpoint.parent / "training_summary.json"
    summary = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
    if args.stage == "background":
        if summary.get("completed_stage") != "background":
            raise ValueError(
                "Background diagnosis requires a completed Background-only run")
        if summary.get("checkpoint_stage") != "background":
            raise ValueError("checkpoint is not marked as Background stage")
    elif summary and summary.get("completed_stage") != "analysis":
        raise ValueError(
            "Analysis diagnosis requires a completed Analysis checkpoint")
    train_days = np.asarray(
        manifest["resolved_training"]["date_split"]["partitions"]["train"],
        dtype=np.int64,
    )
    start_unix = _parse_unix(config["start_date_str"])
    records = load_jicamarca(
        config["jicamarca_dir"], start_unix,
        _parse_unix(config["end_date_str"]),
        alt_min=config["alt_min"], alt_max=config["alt_max"],
        err_ratio_max=config["err_ratio_max"],
    )
    station_lat = float(records[0]["lat"])
    station_lon = float(records[0]["lon"])

    satellite, allowed = _satellite_background_audit(
        model, sw_manager, iri_peak_manager, cfg, train_days,
        station_lat, station_lon, device,
    )
    if args.stage == "background":
        background_gate = _background_jicamarca_gate(
            model, sw_manager, iri_peak_manager, records,
            start_unix, train_days, device)
        report = {
            "schema_version": 2,
            "stage": "background",
            "scope": "train-only satellite profiles and train-day Jicamarca queries",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()).hexdigest(),
            "background_training_semantics": cfg.get(
                "background_training_semantics"),
            "current_training_data": {
                key: cfg.get(key)
                for key in ("fy_path", "cosmic_path", "use_date_blocked_split")
            },
            "satellite_background_sample": satellite,
            "jicamarca_background_gate": background_gate,
        }
        with (output / "m2v_background_gate.json").open(
                "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps({
            "output": str(output),
            "passed_hard_gate": background_gate["passed_hard_gate"],
        }, ensure_ascii=False, indent=2))
        if not background_gate["passed_hard_gate"]:
            raise SystemExit(1)
        return
    coords, observation, isr_relative_uncertainty = _jicamarca_low_night_arrays(
        records, start_unix, train_days)
    query, token = _letkf_audit(
        model, sw_manager, iri_peak_manager, fy_index, cosmic_index,
        allowed, _r_variance_tables(checkpoint), coords, observation,
        isr_relative_uncertainty, device,
    )
    seed_manifest_path = Path(cfg["background_seed_ckpt"]).parent / "run_manifest.json"
    with seed_manifest_path.open(encoding="utf-8") as stream:
        seed_manifest = json.load(stream)
    report = {
        "schema_version": 1,
        "scope": "train-only satellite profiles and train-day Jicamarca queries",
        "checkpoint": str(checkpoint),
        "background_seed_identity": _background_seed_identity(
            checkpoint, cfg["background_seed_ckpt"]),
        "background_seed_training_data": {
            key: seed_manifest["config"].get(key)
            for key in ("fy_path", "cosmic_path", "use_date_blocked_split")
        },
        "current_training_data": {
            key: cfg.get(key)
            for key in ("fy_path", "cosmic_path", "use_date_blocked_split")
        },
        "R_calibration": _r_calibration_summary(checkpoint),
        "jicamarca_ISR_reported_uncertainty": _isr_uncertainty_summary(
            records, start_unix, train_days),
        "satellite_background_sample": satellite,
        "jicamarca_train_only_LETKF": {
            "query": _query_summary(query),
            "tokens": _token_summary(token),
        },
    }
    report["figure"] = _save_figure(report, output)
    with (output / "m2v_low_night_diagnosis.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    np.savez_compressed(output / "m2v_low_night_query_arrays.npz", **query)
    print(json.dumps({
        "output": str(output),
        "jicamarca": report["jicamarca_train_only_LETKF"]["query"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
