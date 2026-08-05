"""ISR-blind development evaluation for the date-blocked run66 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from inr_modules.data_managers.FY_dataloader import (
    COSMICDataset,
    COSMICNeighborhoodIndex,
    FY3D_Dataset,
    FYNeighborhoodIndex,
    ProfileTimeBinSampler,
)
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
from inr_modules.mdia.fsia_model import FSIA_INR_Model, solve_density_mode
from inr_modules.mdia.sliding_dataset import (
    attach_representativeness_weight,
    attach_observation_background,
    build_m2u_anchor_directories,
    load_representativeness_kernel,
    query_observation_payload,
)


ROOT = Path(__file__).resolve().parent
MODES = ("M00", "M10", "M01", "M11")
FORMULA_TOLERANCE = 2e-6
GAIN_FORMULA_TOLERANCE = 3e-6
N_EMPIRICAL_CELLS = 4 * 3 * 3 * 3 * 4
SOURCE_PAIR_INDEX = {
    ("FY", "FY"): 0,
    ("FY", "COSMIC"): 1,
    ("COSMIC", "FY"): 2,
    ("COSMIC", "COSMIC"): 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stable_cells(path):
    with np.load(path, allow_pickle=False) as cells:
        cell_id = cells["cell_id"]
        stable = cells["stable"]
    if (
        cell_id.shape != (N_EMPIRICAL_CELLS,)
        or stable.shape != cell_id.shape
        or not np.array_equal(cell_id, np.arange(N_EMPIRICAL_CELLS))
    ):
        raise ValueError("empirical covariance cells have an incompatible schema")
    return stable.astype(bool, copy=False)


def _apply_stable_cell_mask(
    payload,
    query_coords,
    target_source,
    observation_source,
    stable_cells,
):
    valid = payload["valid_mask"].bool()
    before = {
        "tokens": int(valid.sum().item()),
        "queries": int(valid.any(dim=-1).sum().item()),
    }
    if stable_cells is None:
        return payload, {**before, "retained_tokens": before["tokens"],
                         "retained_queries": before["queries"]}

    stable = torch.as_tensor(
        stable_cells, dtype=torch.bool, device=valid.device
    )
    query = query_coords[:, :4]
    observation = torch.where(
        valid.unsqueeze(-1),
        payload["coords"][..., :4],
        query[:, None, :],
    )
    boundaries = query.new_tensor([200.0, 300.0])
    target_alt = torch.bucketize(
        query[:, 2].contiguous(), boundaries, right=True
    )[:, None]
    observation_alt = torch.bucketize(
        observation[..., 2].contiguous(), boundaries, right=True
    )
    target_lt = torch.remainder(query[:, 3] + query[:, 1] / 15.0, 24.0)
    observation_lt = torch.remainder(
        observation[..., 3] + observation[..., 1] / 15.0, 24.0
    )
    target_day = ((target_lt >= 6.0) & (target_lt < 18.0))[:, None]
    observation_day = (
        (observation_lt >= 6.0) & (observation_lt < 18.0)
    )
    local_time_class = torch.where(
        target_day & observation_day,
        torch.ones_like(observation_alt),
        torch.where(
            ~target_day & ~observation_day,
            torch.zeros_like(observation_alt),
            torch.full_like(observation_alt, 2),
        ),
    )
    rho = torch.sqrt(torch.clamp(
        torch.where(valid, payload["rho_squared"], 0.0), min=0.0
    ))
    rho_bin = torch.clamp((rho * 4.0).long(), max=3)
    pair = SOURCE_PAIR_INDEX[(target_source, observation_source)]
    cell = (
        ((((pair * 3 + target_alt) * 3 + observation_alt) * 3
           + local_time_class) * 4)
        + rho_bin
    )
    retained = valid & stable[cell]
    result = dict(payload)
    result["valid_mask"] = retained
    return result, {
        **before,
        "retained_tokens": int(retained.sum().item()),
        "retained_queries": int(retained.any(dim=-1).sum().item()),
    }


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return _finite(values.mean()) if values.size else None


def _summarize_records(records, minimum_profiles=200, minimum_dates=5):
    dates = {record["date"] for record in records}
    direction_records = [
        record for record in records
        if math.isfinite(record["direction_fraction"])
    ]
    negative_records = [
        record for record in records
        if math.isfinite(record["negative_correct_fraction"])
    ]
    result = {
        "profiles": len(records),
        "dates": len(dates),
        "estimable": (
            len(records) >= minimum_profiles and len(dates) >= minimum_dates
        ),
        "direction_profiles": len(direction_records),
        "direction_dates": len({
            record["date"] for record in direction_records
        }),
        "direction_estimable": (
            len(direction_records) >= minimum_profiles
            and len({record["date"] for record in direction_records})
            >= minimum_dates
        ),
        "negative_profiles": len(negative_records),
        "negative_dates": len({
            record["date"] for record in negative_records
        }),
        "negative_estimable": (
            len(negative_records) >= minimum_profiles
            and len({record["date"] for record in negative_records})
            >= minimum_dates
        ),
    }
    for key in (
        "rmse",
        "mae",
        "bias",
        "nis",
        "coverage90",
        "precision_coverage",
        "direction_fraction",
        "negative_correct_fraction",
        "negative_positive_tail_fraction",
        "negative_increment_p95",
        "negative_increment_p99",
    ):
        result[key] = _mean([record[key] for record in records])
    return result


def _development_gate(sources, hard_invariants, minimum_direction=0.55):
    """Return the ISR-blind candidate gate used before model freezing."""
    def at_least(value, threshold):
        return value is not None and math.isfinite(value) and value >= threshold

    def at_most(value, threshold):
        return value is not None and math.isfinite(value) and value <= threshold

    self_modes = {'FY': 'M10', 'COSMIC': 'M01'}
    cross_modes = {'FY': 'M01', 'COSMIC': 'M10'}
    self_negative = {
        target: sources[target]['modes'][mode]['negative_correct_fraction']
        for target, mode in self_modes.items()
    }
    cross_direction = {
        target: sources[target]['modes'][mode]['direction_fraction']
        for target, mode in cross_modes.items()
    }
    self_non_degraded = {
        target: at_least(
            sources[target]['modes'][mode][
                'relative_rmse_improvement_vs_M00'], -0.01)
        for target, mode in self_modes.items()
    }
    joint_best_single = {
        target: at_most(
            sources[target]['modes']['M11'][
                'relative_rmse_change_vs_best_single'], 0.01)
        for target in ('FY', 'COSMIC')
    }
    cell_checks = {}
    for target in ('FY', 'COSMIC'):
        cell_checks[target] = {}
        for cell, summary in sources[target]['strata']['M11'].items():
            if summary.get('direction_estimable', False):
                cell_checks[target][cell] = (
                    at_least(
                        summary.get('direction_fraction'), minimum_direction))
    all_estimable_cells = all(
        passed for checks in cell_checks.values() for passed in checks.values())
    result = {
        'self_negative_response_ge_70': {
            target: at_least(value, 0.70)
            for target, value in self_negative.items()
        },
        'cross_source_direction_ge_60': {
            target: at_least(value, 0.60)
            for target, value in cross_direction.items()
        },
        'estimable_m11_cells_direction_ge_55': cell_checks,
        'self_profile_rmse_not_degraded_over_1pct': self_non_degraded,
        'm11_not_worse_than_best_single_over_1pct': joint_best_single,
        'hard_invariants': bool(hard_invariants),
    }
    result['passed'] = all((
        result['hard_invariants'],
        all(result['self_negative_response_ge_70'].values()),
        all(result['cross_source_direction_ge_60'].values()),
        all(result['self_profile_rmse_not_degraded_over_1pct'].values()),
        all(result['m11_not_worse_than_best_single_over_1pct'].values()),
        all_estimable_cells,
    ))
    return result


def _summarize_attribution(records, minimum_profiles=200, minimum_dates=5):
    dates = {record["date"] for record in records}
    result = {
        "profiles": len(records),
        "dates": len(dates),
        "estimable": (
            len(records) >= minimum_profiles and len(dates) >= minimum_dates
        ),
    }
    for key in (
        "tokens",
        "innovation_target_agreement",
        "cross_covariance_sign_accuracy",
        "gain_cross_sign_agreement",
        "kalman_contribution_toward",
        "desired_mean",
        "innovation_mean",
        "cross_covariance_mean",
        "gain_mean",
        "contribution_mean",
    ):
        result[key] = _mean([record[key] for record in records])
    return result


def _single_source_gain(extras, source):
    """Exact per-token gain for the corresponding M10 or M01 solve."""
    query_anomalies = extras["query_anomalies"]
    batch, members = query_anomalies.shape
    eye = torch.eye(
        members,
        device=query_anomalies.device,
        dtype=query_anomalies.dtype,
    ).expand(batch, -1, -1)
    system = (
        max(members - 1, 1) * eye
        + extras[f"ensemble_covariance_{source}"]
    )
    weighted_observations = (
        extras[f"obs_anomalies_{source}"].transpose(1, 2)
        * extras[f"precision_{source}"].unsqueeze(1)
    )
    solved = torch.cholesky_solve(
        weighted_observations,
        torch.linalg.cholesky(system),
    )
    return torch.einsum("bn,bnm->bm", query_anomalies, solved)


def _attribution_record(
    query_mask,
    desired,
    innovation,
    cross_covariance,
    gain,
    precision,
    date,
):
    selected = (
        query_mask[:, None]
        & (precision > 0.0)
        & (np.abs(desired[:, None]) >= 0.05)
        & (np.abs(innovation) >= 0.05)
    )
    if not selected.any():
        return {
            "date": int(date),
            "tokens": math.nan,
            "innovation_target_agreement": math.nan,
            "cross_covariance_sign_accuracy": math.nan,
            "gain_cross_sign_agreement": math.nan,
            "kalman_contribution_toward": math.nan,
            "desired_mean": math.nan,
            "innovation_mean": math.nan,
            "cross_covariance_mean": math.nan,
            "gain_mean": math.nan,
            "contribution_mean": math.nan,
        }
    desired_tokens = np.broadcast_to(desired[:, None], innovation.shape)
    empirical_sign = desired_tokens * innovation
    contribution = gain * innovation
    return {
        "date": int(date),
        "tokens": float(selected.sum()),
        "innovation_target_agreement": float(np.mean(
            empirical_sign[selected] > 0.0
        )),
        "cross_covariance_sign_accuracy": float(np.mean(
            (empirical_sign * cross_covariance)[selected] > 0.0
        )),
        "gain_cross_sign_agreement": float(np.mean(
            (gain * cross_covariance)[selected] > 0.0
        )),
        "kalman_contribution_toward": float(np.mean(
            (desired_tokens * contribution)[selected] > 0.0
        )),
        "desired_mean": float(np.mean(desired_tokens[selected])),
        "innovation_mean": float(np.mean(innovation[selected])),
        "cross_covariance_mean": float(np.mean(cross_covariance[selected])),
        "gain_mean": float(np.mean(gain[selected])),
        "contribution_mean": float(np.mean(contribution[selected])),
    }


def _record(mask, error, increment, desired, variance, active, date):
    selected = np.asarray(mask, dtype=bool)
    high = selected & active & (np.abs(desired) >= 0.05)
    negative = selected & active & (desired <= -0.05)
    negative_increment = increment[negative]
    return {
        "date": int(date),
        "rmse": float(np.sqrt(np.mean(np.square(error[selected])))),
        "mae": float(np.mean(np.abs(error[selected]))),
        "bias": float(np.mean(error[selected])),
        "nis": float(np.mean(np.square(error[selected]) / variance[selected])),
        "coverage90": float(np.mean(
            np.abs(error[selected]) <= 1.6448536269514722
            * np.sqrt(variance[selected])
        )),
        "precision_coverage": float(np.mean(active[selected])),
        "direction_fraction": (
            float(np.mean(np.sign(increment[high]) == np.sign(desired[high])))
            if high.any() else math.nan
        ),
        "negative_correct_fraction": (
            float(np.mean(negative_increment < 0.0))
            if negative.any() else math.nan
        ),
        "negative_positive_tail_fraction": (
            float(np.mean(negative_increment > 0.05))
            if negative.any() else math.nan
        ),
        "negative_increment_p95": (
            float(np.quantile(negative_increment, 0.95))
            if negative.any() else math.nan
        ),
        "negative_increment_p99": (
            float(np.quantile(negative_increment, 0.99))
            if negative.any() else math.nan
        ),
    }


def _cell_ids(coords):
    altitude = np.searchsorted(
        np.asarray([200.0, 300.0]), coords[:, 2], side="right"
    )
    local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    day = ((local_time >= 6.0) & (local_time < 18.0)).astype(np.int8)
    latitude = np.clip(
        np.floor((coords[:, 0] + 90.0) / 10.0).astype(np.int16), 0, 17
    )
    return altitude, day, latitude


def _cell_name(altitude, day, latitude):
    altitude_names = ("120-200", "200-300", "300-500")
    lower = -90 + 10 * int(latitude)
    return (
        f"h{altitude_names[int(altitude)]}_"
        f"{'day' if day else 'night'}_lat{lower:+03d}_{lower + 10:+03d}"
    )


def _query_payload(
    index,
    coords,
    device,
    allowed_profile_ids,
    model,
    sw_manager,
    iri_peak_manager,
    excluded_profile_ids=None,
    target_source=None,
    observation_source=None,
    stable_cells=None,
    representativeness_grid=None,
    representativeness_floor=0.25,
):
    payload = query_observation_payload(
        index,
        coords,
        device,
        exclude_profile_ids=excluded_profile_ids,
        allowed_profile_ids=allowed_profile_ids,
    )
    payload, counts = _apply_stable_cell_mask(
        payload,
        coords,
        target_source,
        observation_source,
        stable_cells,
    )
    payload = attach_representativeness_weight(
        payload,
        coords,
        target_source,
        observation_source,
        representativeness_grid,
        representativeness_floor,
    )
    return (
        attach_observation_background(
            payload, model, sw_manager, iri_peak_manager
        ),
        counts,
    )


def _solve_mode(extras, sources, observation_variance):
    """Solve an M00/M10/M01/M11 ETKF mode from one joint forward."""
    shared = extras.get("mode_increments")
    if shared is not None:
        mode = (
            "M00" if not sources else
            "M10" if tuple(sources) == ("FY",) else
            "M01" if tuple(sources) == ("COSMIC",) else "M11")
        increment = torch.zeros_like(extras["ne_bkg"].squeeze(-1))
        if mode != "M00":
            increment = shared[mode]
        variance = torch.full_like(increment, float(observation_variance))
        return increment, variance
    return solve_density_mode(extras, sources, observation_variance)


def _evaluate_source(
    source,
    loader,
    model,
    sw_manager,
    iri_peak_manager,
    fy_index,
    cosmic_index,
    allowed,
    device,
    stable_cells=None,
    representativeness_grid=None,
    representativeness_floor=0.25,
):
    if getattr(model, 'uses_physical_modes', False):
        raise ValueError(
            'query-local physical states are failed audit shadows; '
            'development evaluation is restricted to M2-O')
    global_records = {mode: [] for mode in MODES}
    cell_records = {mode: defaultdict(list) for mode in MODES}
    attribution_records = {
        source_name: [] for source_name in ("FY", "COSMIC")
    }
    attribution_cells = {
        source_name: defaultdict(list)
        for source_name in ("FY", "COSMIC")
    }
    rank_records = defaultdict(list)
    m2u_records = {
        mode: {'core': [], 'active': [], 'increment': [], 'edge_increment': [],
               'alpha_dispersion': [], 'anchor_count': []}
        for mode in ('M10', 'M01', 'M11')}
    m2u_peak_memory_bytes = 0
    mode_seconds = {mode: 0.0 for mode in MODES}
    m00_max_error = 0.0
    joint_formula_max_error = 0.0
    repeat_inference_max_error = None
    single_gain_formula_max_error = {
        source_name: 0.0 for source_name in ("FY", "COSMIC")
    }
    stable_filter_counts = {
        source_name: {
            "tokens": 0,
            "queries": 0,
            "retained_tokens": 0,
            "retained_queries": 0,
        }
        for source_name in ("FY", "COSMIC")
    }
    target_variance = (
        model.kalman_layer.r_fy
        if source == "FY" else model.kalman_layer.r_cosmic
    )
    started = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            data, _, profile_ids = batch
            data = data.to(device)
            profile_ids = profile_ids.to(device)
            coords = data[:, :4]
            target = data[:, 4]
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            iri_peak = (
                iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None
            )
            excluded = profile_ids.cpu().numpy()
            if getattr(model, "uses_shared_anchor_response", False):
                fy, cosmic = build_m2u_anchor_directories(
                    {"FY": fy_index, "COSMIC": cosmic_index},
                    coords, device, model, sw_manager, iri_peak_manager,
                    allowed_profile_ids=allowed,
                    excluded_profile_ids={source: excluded},
                    space_km=model.m2u_space_support_km,
                    time_h=model.m2u_time_support_h)
                fy_counts = {"directory_tokens": int(
                    fy['valid_mask'].sum().item()) if fy is not None else 0}
                cosmic_counts = {"directory_tokens": int(
                    cosmic['valid_mask'].sum().item()) if cosmic is not None else 0}
            else:
                fy, fy_counts = _query_payload(
                    fy_index,
                    coords,
                    device,
                    allowed["FY"],
                    model,
                    sw_manager,
                    iri_peak_manager,
                    excluded if source == "FY" else None,
                    source,
                    "FY",
                    stable_cells,
                    representativeness_grid,
                    representativeness_floor,
                )
                cosmic, cosmic_counts = _query_payload(
                    cosmic_index,
                    coords,
                    device,
                    allowed["COSMIC"],
                    model,
                    sw_manager,
                    iri_peak_manager,
                    excluded if source == "COSMIC" else None,
                    source,
                    "COSMIC",
                    stable_cells,
                    representativeness_grid,
                    representativeness_floor,
                )
            for observation_source, counts in (
                ("FY", fy_counts),
                ("COSMIC", cosmic_counts),
            ):
                for key, value in counts.items():
                    stable_filter_counts[observation_source][key] += value
            forward_started = time.perf_counter()
            joint_kwargs = (
                {'anchor_observations_fy': fy,
                 'anchor_observations_cosmic': cosmic}
                if getattr(model, "uses_shared_anchor_response", False)
                else {'observations_fy': fy, 'observations_cosmic': cosmic})
            joint_prediction, _, _, _, joint_extras = model(
                coords, sw_seq, iri_peak=iri_peak, **joint_kwargs)
            if repeat_inference_max_error is None:
                repeated_prediction = model(
                    coords, sw_seq, iri_peak=iri_peak, **joint_kwargs)[0]
                repeat_inference_max_error = float(torch.max(torch.abs(
                    repeated_prediction - joint_prediction
                )).item())
            mode_seconds["M11"] += time.perf_counter() - forward_started
            predictions = {}
            variances = {}
            for mode, sources in (
                ("M00", ()),
                ("M10", ("FY",)),
                ("M01", ("COSMIC",)),
                ("M11", ("FY", "COSMIC")),
            ):
                solve_started = time.perf_counter()
                increment, variance = _solve_mode(
                    joint_extras, sources, target_variance
                )
                mode_seconds[mode] += time.perf_counter() - solve_started
                predictions[mode] = (
                    joint_extras["ne_bkg"].squeeze(-1) + increment
                ).cpu().numpy()
                variances[mode] = variance.cpu().numpy()
            active = {
                "M00": np.zeros(len(coords), dtype=bool),
                "M10": (
                    joint_extras["precision_FY"].sum(dim=-1) > 0
                ).cpu().numpy(),
                "M01": (
                    joint_extras["precision_COSMIC"].sum(dim=-1) > 0
                ).cpu().numpy(),
            }
            active["M11"] = active["M10"] | active["M01"]

            if getattr(model, "uses_shared_anchor_response", False):
                m2u_peak_memory_bytes = max(
                    m2u_peak_memory_bytes,
                    int(joint_extras.get('m2u_peak_memory_bytes', 0)))
                for mode in ('M10', 'M01', 'M11'):
                    core = joint_extras[
                        f'shared_anchor_core_{mode}'].any(dim=-1)
                    active_count = joint_extras[
                        f'shared_anchor_active_count_{mode}']
                    alpha = joint_extras[f'shared_anchor_alpha_{mode}']
                    background_weight = joint_extras[
                        f'shared_anchor_background_weight_{mode}']
                    increment = joint_extras['mode_increments'][mode]
                    m2u_records[mode]['core'].extend(
                        core.cpu().numpy().tolist())
                    m2u_records[mode]['active'].extend(
                        active_count.cpu().numpy().tolist())
                    m2u_records[mode]['increment'].extend(
                        increment.cpu().numpy().tolist())
                    edge = (background_weight >= 0.999) & (
                        background_weight < 1.0)
                    m2u_records[mode]['edge_increment'].extend(
                        increment[edge].abs().cpu().numpy().tolist())
                    if alpha.shape[1]:
                        dispersion = 1.0 - alpha.max(dim=-1).values
                        m2u_records[mode]['alpha_dispersion'].extend(
                            dispersion[active_count > 1].cpu().numpy().tolist())
                    m2u_records[mode]['anchor_count'].append(int(
                        joint_extras[f'shared_anchor_count_{mode}']))

            target_np = target.cpu().numpy()
            coords_np = coords.cpu().numpy()
            profile_np = profile_ids.cpu().numpy()
            background = joint_extras["ne_bkg"].squeeze(-1).cpu().numpy()
            desired = target_np - background
            attribution = {}
            for observation_source, mode in (
                ("FY", "M10"),
                ("COSMIC", "M01"),
            ):
                exact_gain = _single_source_gain(
                    joint_extras, observation_source
                )
                innovation = joint_extras[
                    f"innov_{observation_source}"
                ]
                contribution = (exact_gain * innovation).sum(dim=-1)
                single_increment = torch.as_tensor(
                    predictions[mode] - background,
                    device=contribution.device,
                    dtype=contribution.dtype,
                )
                if getattr(model, "uses_shared_anchor_response", False):
                    # M2-U stores anchor-response interpolation, not a query
                    # local token gain.  The exact source increment is the
                    # structural invariant; token attribution is diagnostic.
                    contribution = single_increment
                single_gain_formula_max_error[observation_source] = max(
                    single_gain_formula_max_error[observation_source],
                    float(torch.max(torch.abs(
                        contribution - single_increment
                    )).item()),
                )
                attribution[observation_source] = {
                    "innovation": innovation.cpu().numpy(),
                    "cross_covariance": joint_extras[
                        f"cross_covariance_{observation_source}"
                    ].cpu().numpy(),
                    "gain": exact_gain.cpu().numpy(),
                    "precision": joint_extras[
                        f"precision_{observation_source}"
                    ].cpu().numpy(),
                }
            m00_max_error = max(
                m00_max_error,
                float(np.max(np.abs(predictions["M00"] - background))),
            )
            joint_formula_max_error = max(
                joint_formula_max_error,
                float(np.max(np.abs(
                    predictions["M11"]
                    - joint_prediction.squeeze(-1).cpu().numpy()
                ))),
            )
            altitude_cell, day_cell, latitude_cell = _cell_ids(coords_np)

            rank_extras = joint_extras
            rank_values = {
                "effective_rank": rank_extras[
                    "anomaly_effective_rank"
                ].cpu().numpy(),
                "condition": rank_extras["anomaly_condition"].cpu().numpy(),
                "scale_saturation": rank_extras[
                    "scale_boundary_saturation"
                ].cpu().numpy(),
            }
            for profile_id in np.unique(profile_np):
                profile_mask = profile_np == profile_id
                date = int(np.floor(np.median(coords_np[profile_mask, 3]) / 24.0))
                for key, values in rank_values.items():
                    rank_records[key].append(float(np.mean(values[profile_mask])))
                for observation_source in ("FY", "COSMIC"):
                    attribution_records[observation_source].append(
                        _attribution_record(
                            profile_mask,
                            desired,
                            **attribution[observation_source],
                            date=date,
                        )
                    )
                for mode in MODES:
                    error = predictions[mode] - target_np
                    increment = predictions[mode] - background
                    global_records[mode].append(_record(
                        profile_mask,
                        error,
                        increment,
                        desired,
                        variances[mode],
                        active[mode],
                        date,
                    ))
                    cells = {
                        (
                            int(altitude_cell[index]),
                            int(day_cell[index]),
                            int(latitude_cell[index]),
                        )
                        for index in np.flatnonzero(profile_mask)
                    }
                    for cell in cells:
                        cell_mask = (
                            profile_mask
                            & (altitude_cell == cell[0])
                            & (day_cell == cell[1])
                            & (latitude_cell == cell[2])
                        )
                        cell_records[mode][_cell_name(*cell)].append(_record(
                            cell_mask,
                            error,
                            increment,
                            desired,
                            variances[mode],
                            active[mode],
                            date,
                        ))
                        if mode == "M00":
                            for observation_source in ("FY", "COSMIC"):
                                attribution_cells[observation_source][
                                    _cell_name(*cell)
                                ].append(_attribution_record(
                                    cell_mask,
                                    desired,
                                    **attribution[observation_source],
                                    date=date,
                                ))

    summaries = {
        mode: _summarize_records(global_records[mode]) for mode in MODES
    }
    m00_rmse = summaries["M00"]["rmse"]
    for mode in MODES:
        summaries[mode]["relative_rmse_improvement_vs_M00"] = (
            _finite((m00_rmse - summaries[mode]["rmse"]) / m00_rmse)
            if m00_rmse else None
        )
    best_single = min(summaries["M10"]["rmse"], summaries["M01"]["rmse"])
    summaries["M11"]["relative_rmse_change_vs_best_single"] = _finite(
        (summaries["M11"]["rmse"] - best_single) / best_single
    )
    cells = {
        mode: {
            cell: _summarize_records(records)
            for cell, records in sorted(cell_records[mode].items())
        }
        for mode in MODES
    }
    return {
        "target_source": source,
        "profiles": len(global_records["M00"]),
        "points": len(loader.dataset),
        "modes": summaries,
        "strata": cells,
        "direction_attribution": {
            observation_source: {
                "global": _summarize_attribution(
                    attribution_records[observation_source]
                ),
                "strata": {
                    cell: _summarize_attribution(records)
                    for cell, records in sorted(
                        attribution_cells[observation_source].items()
                    )
                },
            }
            for observation_source in ("FY", "COSMIC")
        },
        "ensemble": {
            "effective_rank_q05": _finite(np.quantile(
                rank_records["effective_rank"], 0.05
            )),
            "effective_rank_median": _finite(np.median(
                rank_records["effective_rank"]
            )),
            "condition_q95": _finite(np.quantile(
                rank_records["condition"], 0.95
            )),
            "scale_saturation_mean": _mean(
                rank_records["scale_saturation"]
            ),
        },
        "m2u_shared_anchor": ({
            mode: {
                "core_query_coverage": _mean(records['core']),
                "core_query_count": int(np.sum(records['core'])),
                "active_anchor_median": _finite(np.median(records['active'])),
                "overlap_query_fraction": _mean(
                    np.asarray(records['active']) > 1),
                "overlap_alpha_dispersion_mean": _mean(
                    records['alpha_dispersion']),
                "increment_rms": _finite(np.sqrt(np.mean(
                    np.square(records['increment'])))),
                "M00_edge_max_abs_increment": (
                    max(records['edge_increment'])
                    if records['edge_increment'] else 0.0),
                "anchor_count_max": max(records['anchor_count'], default=0),
            }
            for mode, records in m2u_records.items()
        } | {"peak_memory_bytes": m2u_peak_memory_bytes}
            if getattr(model, "uses_shared_anchor_response", False) else None),
        "stable_filter": {
            observation_source: {
                **counts,
                "token_recall": (
                    counts["retained_tokens"] / counts["tokens"]
                    if counts["tokens"] else None
                ),
                "query_recall": (
                    counts["retained_queries"] / counts["queries"]
                    if counts["queries"] else None
                ),
                "m00_fallback_fraction": (
                    1.0 - counts["retained_queries"] / len(loader.dataset)
                    if len(loader.dataset) else None
                ),
            }
            for observation_source, counts in stable_filter_counts.items()
        },
        "invariants": {
            "M00_max_abs_difference_from_background": m00_max_error,
            "M00_passed": m00_max_error < 1e-7,
            "joint_formula_max_abs_difference": joint_formula_max_error,
            "joint_formula_tolerance": FORMULA_TOLERANCE,
            "joint_formula_passed": (
                joint_formula_max_error <= FORMULA_TOLERANCE
            ),
            "single_gain_formula_max_abs_difference": (
                single_gain_formula_max_error
            ),
            "single_gain_formula_tolerance": GAIN_FORMULA_TOLERANCE,
            "single_gain_formula_passed": all(
                value <= GAIN_FORMULA_TOLERANCE
                for value in single_gain_formula_max_error.values()
            ),
            "repeat_inference_max_abs_difference": repeat_inference_max_error,
            "repeat_inference_deterministic": repeat_inference_max_error == 0.0,
        },
        "cpu_seconds": {
            "total": time.perf_counter() - started,
            "forward_by_mode": mode_seconds,
        },
    }


def _partition_loader(dataset_class, config, split_days, partition):
    kwargs = {
        "npy_path": config["fy_path"],
        "profile_path": config.get("fy_profile_path"),
        "profile_index_path": config.get("fy_profile_index_path"),
    } if dataset_class is FY3D_Dataset else {
        "npy_path": config["cosmic_path"],
        "profile_index_path": config.get("cosmic_profile_index_path"),
    }
    dataset = dataset_class(
        kwargs.pop("npy_path"),
        mode=partition,
        val_days=[],
        bin_size_hours=config["bin_size_hours"],
        use_memmap=True,
        val_ratio=None,
        split_seed=config["seed"],
        split_days=split_days,
        **kwargs,
    )
    sampler = ProfileTimeBinSampler(
        dataset,
        batch_size=config["batch_size"],
        points_per_profile=None,
        shuffle=False,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("development", "locked_test"),
        default="development",
    )
    parser.add_argument(
        "--stable-covariance-cells",
        type=Path,
        help="train-only empirical cells used for a read-only stable-only mask",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint or run_dir / "best_fsia_model.pth"
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run directory lacks checkpoint or manifest")
    with manifest_path.open(encoding="utf-8") as stream:
        run_manifest = json.load(stream)
    config = dict(run_manifest["config"])
    date_manifest_path = Path(config["date_split_manifest"])
    if not date_manifest_path.is_absolute():
        date_manifest_path = ROOT / date_manifest_path
    with date_manifest_path.open(encoding="utf-8") as stream:
        date_manifest = json.load(stream)
    split_days = date_manifest["partitions"]
    if set(split_days) != {"train", "development", "locked_test"}:
        raise ValueError("date manifest partitions are incomplete")

    device = torch.device("cpu")
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(
        config["iri_proxy_path"], map_location=device, weights_only=True
    ))
    iri_proxy.eval()
    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(
        torch.isfinite(value).all()
        for value in state.values() if torch.is_tensor(value)
    ):
        raise ValueError("checkpoint contains non-finite tensors")
    model.eval()
    sw_manager = SpaceWeatherManager(
        txt_path=config["sw_path"],
        start_date_str=config["start_date_str"],
        total_hours=config["total_hours"],
        seq_len=config["seq_len"],
        device=device,
    )
    iri_peak_manager = None
    hmf2 = config.get("iri_hmf2_path")
    nmf2 = config.get("iri_nmf2_path")
    if hmf2 and nmf2 and Path(hmf2).is_file() and Path(nmf2).is_file():
        from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
        iri_peak_manager = IRIPeakManager(hmf2, nmf2, device=device)

    fy_loader = _partition_loader(
        FY3D_Dataset, config, split_days, args.partition)
    cosmic_loader = _partition_loader(
        COSMICDataset, config, split_days, args.partition)
    allowed = {
        "FY": np.unique(fy_loader.dataset.profile_ids),
        "COSMIC": np.unique(cosmic_loader.dataset.profile_ids),
    }
    fy_index = FYNeighborhoodIndex(config["fy_path"], config)
    cosmic_index = COSMICNeighborhoodIndex(config["cosmic_path"], config)
    stable_cells_path = (
        args.stable_covariance_cells.resolve()
        if args.stable_covariance_cells is not None else None
    )
    stable_cells = (
        _load_stable_cells(stable_cells_path)
        if stable_cells_path is not None else None
    )
    representativeness_path = config.get(
        "representativeness_kernel_path")
    if representativeness_path:
        representativeness_path = Path(representativeness_path)
        if not representativeness_path.is_absolute():
            representativeness_path = ROOT / representativeness_path
        representativeness_path = representativeness_path.resolve()
    representativeness_grid = load_representativeness_kernel(
        representativeness_path)
    representativeness_floor = float(
        config.get("representativeness_floor", 0.25))
    if stable_cells is not None and representativeness_grid is not None:
        raise ValueError(
            "stable-only and continuous representativeness filters conflict")
    report = {
        "schema_version": 1,
        "purpose": (
            f"ISR-blind satellite {args.partition} stable-only counterfactual"
            if stable_cells is not None
            else (
                f"ISR-blind satellite {args.partition} "
                "continuous-representativeness evaluation"
                if representativeness_grid is not None
                else f"ISR-blind satellite {args.partition} evaluation"
            )
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "strict_load": True,
            "all_tensors_finite": True,
        },
        "date_split_manifest": {
            "path": str(date_manifest_path),
            "sha256": _sha256(date_manifest_path),
            "evaluated_partition": args.partition,
            "evaluated_days": split_days[args.partition],
            "locked_test_metrics_accessed": args.partition == "locked_test",
        },
        "profile_counts": {
            "FY": int(len(allowed["FY"])),
            "COSMIC": int(len(allowed["COSMIC"])),
        },
        "stable_covariance_filter": (
            {
                "path": str(stable_cells_path),
                "sha256": _sha256(stable_cells_path),
                "stable_cells": int(stable_cells.sum()),
                "total_cells": int(stable_cells.size),
                "training_or_model_parameters_changed": False,
            }
            if stable_cells is not None else None
        ),
        "representativeness_kernel": (
            {
                "path": str(representativeness_path),
                "sha256": _sha256(representativeness_path),
                "stable_cells": int(
                    representativeness_grid.sum().item()),
                "total_cells": int(
                    representativeness_grid.numel()),
                "floor": representativeness_floor,
            }
            if representativeness_grid is not None else None
        ),
        "sources": {},
    }
    report["sources"]["FY"] = _evaluate_source(
        "FY", fy_loader, model, sw_manager, iri_peak_manager,
        fy_index, cosmic_index, allowed, device, stable_cells,
        representativeness_grid, representativeness_floor,
    )
    report["sources"]["COSMIC"] = _evaluate_source(
        "COSMIC", cosmic_loader, model, sw_manager, iri_peak_manager,
        fy_index, cosmic_index, allowed, device, stable_cells,
        representativeness_grid, representativeness_floor,
    )
    report["passed_hard_invariants"] = all(
        result["invariants"]["M00_passed"]
        and result["invariants"]["joint_formula_passed"]
        and result["invariants"]["single_gain_formula_passed"]
        and result["invariants"]["repeat_inference_deterministic"]
        for result in report["sources"].values()
    )
    report["development_gates"] = _development_gate(
        report["sources"], report["passed_hard_invariants"])
    report["passed_development_gates"] = report["development_gates"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
