"""Read-only M2-M audit of covariance reciprocity and loss gradients."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_etkf_observation_subspace import (
    _geometry,
    _load,
    _restrict_profiles,
)
from evaluate_satellite_development import (
    ROOT,
    _cell_ids,
    _cell_name,
    _partition_loader,
    _sha256,
)
from inr_modules.data_managers.FY_dataloader import (
    COSMICDataset,
    COSMICNeighborhoodIndex,
    FY3D_Dataset,
    FYNeighborhoodIndex,
)
from inr_modules.mdia.fsia_model import (
    _build_kalman_b_input,
    solve_density_modes,
)
from inr_modules.mdia.sliding_dataset import (
    SlidingWindowBatchProcessor,
    attach_observation_background,
    attach_representativeness_weight,
    empirical_covariance_token_targets,
    load_empirical_covariance_targets,
    load_representativeness_kernel,
    query_observation_payload,
)
from inr_modules.mdia.train_fsia import (
    _paired_analysis_losses,
    _set_training_stage,
)


SOURCES = ("FY", "COSMIC")
MODES = ("M10", "M01", "M11")


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "mean": float(values.mean()),
        "q05_q50_q95": np.quantile(values, [0.05, 0.5, 0.95]).tolist(),
    }


def _profile_mean_rows(rows, value_keys):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key in value_keys:
            if math.isfinite(float(row[key])):
                grouped[int(row["profile_id"])][key].append(float(row[key]))
    collapsed = []
    for profile_id, values in grouped.items():
        if all(values[key] for key in value_keys):
            collapsed.append({
                "profile_id": profile_id,
                **{key: float(np.mean(values[key])) for key in value_keys},
            })
    return collapsed


def _reciprocity_summary(rows):
    if not rows:
        return {
            "profile_pairs": 0,
            "sign_reciprocity": None,
            "absolute_ratio": None,
            "max_absolute_error": None,
        }
    grouped = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for row in rows:
        key = (
            int(row["target_profile_id"]),
            int(row["observation_profile_id"]),
            int(row["cell_id"]),
        )
        weight = float(row["weight"])
        grouped[key] += weight * np.asarray([
            float(row["forward"]), float(row["reverse"]), 1.0
        ])
    pairs = []
    for weighted_forward, weighted_reverse, weight in grouped.values():
        if weight > 0:
            pairs.append((weighted_forward / weight, weighted_reverse / weight))
    forward = np.asarray([row[0] for row in pairs], dtype=np.float64)
    reverse = np.asarray([row[1] for row in pairs], dtype=np.float64)
    nonzero = (np.abs(forward) > 1e-12) & (np.abs(reverse) > 1e-12)
    ratios = np.abs(forward[nonzero]) / np.abs(reverse[nonzero])
    return {
        "profile_pairs": len(pairs),
        "nonzero_profile_pairs": int(nonzero.sum()),
        "sign_reciprocity": (
            float(np.mean(forward[nonzero] * reverse[nonzero] > 0.0))
            if nonzero.any() else None
        ),
        "absolute_ratio": _quantiles(ratios),
        "max_absolute_error": float(np.max(np.abs(forward - reverse))),
    }


def _covariance_target_summary(rows):
    if not rows:
        return {
            "profile_pairs": 0,
            "sign_accuracy": None,
            "relative_absolute_error": None,
        }
    grouped = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for row in rows:
        key = (
            int(row["target_profile_id"]),
            int(row["observation_profile_id"]),
            int(row["cell_id"]),
        )
        weight = float(row["weight"])
        grouped[key] += weight * np.asarray([
            float(row["predicted"]), float(row["target"]), 1.0
        ])
    pairs = [
        (values[0] / values[2], values[1] / values[2])
        for values in grouped.values() if values[2] > 0
    ]
    predicted = np.asarray([row[0] for row in pairs], dtype=np.float64)
    target = np.asarray([row[1] for row in pairs], dtype=np.float64)
    nonzero = np.abs(target) > 1e-12
    relative_error = (
        np.abs(predicted[nonzero] - target[nonzero])
        / np.abs(target[nonzero])
    )
    amplitude_ratio = predicted[nonzero] / target[nonzero]
    return {
        "profile_pairs": len(pairs),
        "nonzero_target_pairs": int(nonzero.sum()),
        "sign_accuracy": (
            float(np.mean(predicted[nonzero] * target[nonzero] > 0.0))
            if nonzero.any() else None
        ),
        "relative_absolute_error": _quantiles(relative_error),
        "signed_amplitude_ratio": _quantiles(amplitude_ratio),
    }


def _gradient_vector(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def _gradient_comparison(losses, parameters):
    vectors = {
        name: _gradient_vector(loss, parameters)
        for name, loss in losses.items()
    }
    norms = {name: float(vector.norm()) for name, vector in vectors.items()}

    def cosine(left, right):
        denominator = vectors[left].norm() * vectors[right].norm()
        return (
            float(torch.dot(vectors[left], vectors[right]) / denominator)
            if float(denominator) > 0.0 else math.nan
        )

    return {
        "norms": norms,
        "cosine_observation_covariance": cosine("observation", "covariance"),
        "cosine_observation_direction": cosine("observation", "direction"),
        "cosine_covariance_direction": cosine("covariance", "direction"),
    }


def _direct_cross_covariance(
        model, query_coords, target_coords, target_background,
        sw_seq, iri_peak=None, target_sw_seq=None, target_iri_peak=None):
    """Evaluate C(query,target) without solving an analysis system."""
    background = model.encode_background(query_coords, sw_seq, iri_peak=iri_peak)
    target_context = None
    if model.density_basis_semantics == "endpoint_context_symmetric":
        if target_sw_seq is None:
            raise ValueError("endpoint-context reciprocity requires target drivers")
        target_context = model.encode_background(
            target_coords, target_sw_seq, iri_peak=target_iri_peak)
    query_phi = model._density_basis(
        query_coords,
        query_coords[:, None, :4],
        background["ne_bkg"],
        background["z_background"],
        background["h_sw"],
        background["z_background"][:, None],
        background["h_sw"][:, None],
    ).squeeze(1)
    target_phi = model._density_basis(
        query_coords,
        target_coords[:, None, :4],
        target_background[:, None],
        background["z_background"],
        background["h_sw"],
        (None if target_context is None else
         target_context["z_background"][:, None]),
        (None if target_context is None else target_context["h_sw"][:, None]),
    ).squeeze(1)
    b_input = _build_kalman_b_input(
        background["h_sw"], background["lat_n"], background["cos_sza"],
        background["sin_doy"], background["cos_doy"], background["sin_i"])
    anomalies, _ = model.kalman_layer._eval_perturbations(b_input)
    if model.kalman_layer.anomaly_parameterization == "legacy_independent":
        anomalies = anomalies - anomalies.mean(dim=1, keepdim=True)
        inflation = torch.exp(model.kalman_layer.log_inflation).clamp(0.8, 1.5)
        anomalies = anomalies * torch.sqrt(inflation)
    query_anomalies = torch.einsum("bd,bnd->bn", query_phi, anomalies)
    target_anomalies = torch.einsum("bd,bnd->bn", target_phi, anomalies)
    return torch.einsum(
        "bn,bn->b", query_anomalies, target_anomalies
    ) / max(model.kalman_layer.n_members - 1, 1)


def _empirical_cell_ids(query, observation, rho_squared, pair_index):
    boundaries = query.new_tensor([200.0, 300.0])
    target_altitude = torch.bucketize(
        query[:, 2].contiguous(), boundaries, right=True)[:, None]
    observation_altitude = torch.bucketize(
        observation[..., 2].contiguous(), boundaries, right=True)
    target_lt = torch.remainder(query[:, 3] + query[:, 1] / 15.0, 24.0)[:, None]
    observation_lt = torch.remainder(
        observation[..., 3] + observation[..., 1] / 15.0, 24.0)
    target_day = (target_lt >= 6.0) & (target_lt < 18.0)
    observation_day = (observation_lt >= 6.0) & (observation_lt < 18.0)
    lt_class = torch.where(
        target_day & observation_day,
        torch.ones_like(observation_altitude),
        torch.where(
            ~target_day & ~observation_day,
            torch.zeros_like(observation_altitude),
            torch.full_like(observation_altitude, 2),
        ),
    )
    rho_bin = torch.clamp(
        (torch.sqrt(rho_squared.clamp_min(0.0)) * 4.0).long(), max=3)
    return (
        ((((pair_index * 3 + target_altitude) * 3 + observation_altitude) * 3
          + lt_class) * 4) + rho_bin
    )


def _profile_geometry(matrix, precision, profile_ids):
    rows = []
    for profile_id in torch.unique(profile_ids[precision > 0]):
        selected = (profile_ids == profile_id) & (precision > 0)
        weights = precision[selected]
        rows.append(
            (matrix[selected] * weights[:, None]).sum(0)
            / weights.sum().clamp_min(1e-12)
        )
    if not rows:
        return None
    aggregated = torch.stack(rows)
    geometry, _ = _geometry(
        aggregated, torch.ones(len(rows), device=matrix.device, dtype=matrix.dtype))
    return geometry


def _select_reciprocity_tokens(payload, precision):
    """Choose at most one highest-precision token in each altitude band."""
    selected = []
    for query_index in range(len(precision)):
        valid = precision[query_index] > 0
        altitude = torch.bucketize(
            payload["coords"][query_index, :, 2].contiguous(),
            precision.new_tensor([200.0, 300.0]), right=True)
        for altitude_bin in range(3):
            candidates = torch.nonzero(
                valid & (altitude == altitude_bin), as_tuple=True)[0]
            if len(candidates):
                best = candidates[torch.argmax(precision[query_index, candidates])]
                selected.append((query_index, int(best)))
    return selected


def _source_payload(
        index, coords, profile_ids, target_source, observation_source,
        allowed, model, sw_manager, iri_peak_manager,
        representativeness, floor):
    exclude = profile_ids.numpy() if target_source == observation_source else None
    payload = query_observation_payload(
        index, coords, torch.device("cpu"), exclude_profile_ids=exclude,
        allowed_profile_ids=allowed)
    payload = attach_representativeness_weight(
        payload, coords, target_source, observation_source,
        representativeness, floor)
    return attach_observation_background(
        payload, model, sw_manager, iri_peak_manager)


def _summarize_geometry(rows):
    collapsed = _profile_mean_rows(
        rows, ("numeric_rank", "effective_rank", "condition"))
    return {
        "profiles": len(collapsed),
        "numeric_rank": _quantiles([row["numeric_rank"] for row in collapsed]),
        "effective_rank": _quantiles([
            row["effective_rank"] for row in collapsed]),
        "condition": _quantiles([row["condition"] for row in collapsed]),
    }


def _summarize_endpoint_context(rows):
    result = {}
    for cell, values in rows.items():
        by_profile = defaultdict(list)
        for row in values:
            by_profile[row["profile_id"]].append(row["endpoint_l2"])
        profile_values = np.asarray([
            np.mean(distances) for distances in by_profile.values()],
            dtype=np.float64)
        result[cell] = {
            "profiles": int(len(profile_values)),
            "median_endpoint_l2": (
                float(np.median(profile_values)) if len(profile_values) else None),
            "nonzero_profile_fraction": (
                float(np.mean(profile_values > 1e-8))
                if len(profile_values) else None),
        }
    return result


def _selection_coverage(loader):
    dates = defaultdict(int)
    for groups in loader.batch_sampler.profiles_by_bin.values():
        for indices in groups:
            actual = loader.dataset.selected_indices[int(indices[len(indices) // 2])]
            date = int(math.floor(float(loader.dataset.data[actual, 3]) / 24.0))
            dates[date] += 1
    return {
        "profiles": int(sum(dates.values())),
        "dates": len(dates),
        "profiles_by_date": {
            str(date): count for date, count in sorted(dates.items())
        },
    }


def _summarize_exposure(rows):
    result = {}
    for key, values in sorted(rows.items()):
        profiles = defaultdict(list)
        for row in values:
            profiles[int(row["profile_id"])].append(row)
        collapsed = []
        for profile_id, profile_rows in profiles.items():
            eligible = sum(row["eligible"] for row in profile_rows)
            collapsed.append({
                "profile_id": profile_id,
                "eligible": eligible,
                "wrong_fraction": (
                    sum(row["wrong"] for row in profile_rows) / eligible
                    if eligible else math.nan),
                "huber_derivative": float(np.mean([
                    row["huber_derivative"] for row in profile_rows])),
            })
        wrong = [
            row["wrong_fraction"] for row in collapsed
            if math.isfinite(row["wrong_fraction"])
        ]
        result[key] = {
            "profiles": len(collapsed),
            "eligible_profiles": len(wrong),
            "eligible_points_profile_mean": _quantiles([
                row["eligible"] for row in collapsed]),
            "direction_nonzero_gradient_fraction": (
                float(np.mean(wrong)) if wrong else None
            ),
            "huber_derivative_profile_mean": _quantiles([
                row["huber_derivative"] for row in collapsed]),
        }
    return result


def _audit_partition(
        partition, loaders, model, sw_manager, iri_peak_manager, indices,
        allowed, empirical_targets, representativeness, floor):
    result = {}
    pair_index = {
        ("FY", "FY"): 0, ("FY", "COSMIC"): 1,
        ("COSMIC", "FY"): 2, ("COSMIC", "COSMIC"): 3,
    }
    for target_source in SOURCES:
        reciprocity_rows = {source: [] for source in SOURCES}
        covariance_rows = {source: [] for source in SOURCES}
        geometry_rows = {source: [] for source in SOURCES}
        context_rows = {source: defaultdict(list) for source in SOURCES}
        exposure_rows = defaultdict(list)
        with torch.no_grad():
            for data, _, profile_ids in loaders[target_source]:
                coords, target = data[:, :4], data[:, 4]
                sw = sw_manager.get_drivers_sequence(coords[:, 3])
                peak = (
                    iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
                payloads = {
                    observation_source: _source_payload(
                        indices[observation_source], coords, profile_ids,
                        target_source, observation_source,
                        allowed[observation_source], model, sw_manager,
                        iri_peak_manager, representativeness, floor)
                    for observation_source in SOURCES
                }
                _, _, _, _, extras = model(
                    coords, sw, iri_peak=peak,
                    observations_fy=payloads["FY"],
                    observations_cosmic=payloads["COSMIC"])
                mode_increments = solve_density_modes(extras)
                desired = target - extras["ne_bkg"].squeeze(-1)
                altitude, day, latitude = _cell_ids(coords.numpy())
                cell_names = [
                    _cell_name(altitude[i], day[i], latitude[i])
                    for i in range(len(coords))
                ]
                active = {
                    "M10": extras["precision_FY"].sum(-1) > 0,
                    "M01": extras["precision_COSMIC"].sum(-1) > 0,
                    "M11": (
                        extras["precision_FY"].sum(-1)
                        + extras["precision_COSMIC"].sum(-1)) > 0,
                }
                for mode in MODES:
                    prediction_error = (
                        extras["ne_bkg"].squeeze(-1)
                        + mode_increments[mode] - target)
                    eligible = active[mode] & (desired.abs() >= 0.05)
                    wrong = eligible & (desired * mode_increments[mode] < 0.0)
                    derivative = prediction_error.abs().clamp_max(0.2)
                    for index in range(len(coords)):
                        if active[mode][index]:
                            exposure_rows[f"{mode}/{cell_names[index]}"].append({
                                "profile_id": int(profile_ids[index]),
                                "eligible": float(eligible[index]),
                                "wrong": float(wrong[index]),
                                "huber_derivative": float(derivative[index]),
                            })
                for observation_source in SOURCES:
                    precision = extras[f"precision_{observation_source}"]
                    observation_ids = payloads[observation_source]["profile_id"]
                    if model.density_basis_semantics == "endpoint_context_symmetric":
                        query_context = torch.cat([
                            extras["basis_z_background"], extras["basis_h_sw"]
                        ], dim=-1)
                        observation_context = torch.cat([
                            payloads[observation_source]["basis_z_background"],
                            payloads[observation_source]["basis_h_sw"],
                        ], dim=-1)
                        endpoint_l2 = torch.linalg.vector_norm(
                            observation_context - query_context[:, None, :], dim=-1)
                        for query_index in range(len(coords)):
                            active_tokens = precision[query_index] > 0.0
                            if active_tokens.any():
                                context_rows[observation_source][
                                    cell_names[query_index]].append({
                                        "profile_id": int(profile_ids[query_index]),
                                        "endpoint_l2": float(endpoint_l2[
                                            query_index, active_tokens].mean()),
                                    })
                    for query_index in range(len(coords)):
                        geometry = _profile_geometry(
                            extras[f"obs_anomalies_{observation_source}"][query_index],
                            precision[query_index], observation_ids[query_index])
                        if geometry is not None:
                            geometry_rows[observation_source].append({
                                "profile_id": int(profile_ids[query_index]),
                                **geometry,
                            })
                    covariance_target, stable = empirical_covariance_token_targets(
                        coords, payloads[observation_source]["coords"],
                        payloads[observation_source]["rho_squared"],
                        precision > 0.0, target_source, observation_source,
                        empirical_targets)
                    cells = _empirical_cell_ids(
                        coords, payloads[observation_source]["coords"],
                        payloads[observation_source]["rho_squared"],
                        pair_index[(target_source, observation_source)])
                    selected = _select_reciprocity_tokens(
                        payloads[observation_source], precision)
                    if not selected:
                        continue
                    for query_index, token_index in selected:
                        if stable[query_index, token_index]:
                            covariance_rows[observation_source].append({
                                "target_profile_id": int(profile_ids[query_index]),
                                "observation_profile_id": int(
                                    observation_ids[query_index, token_index]),
                                "cell_id": int(cells[query_index, token_index]),
                                "predicted": float(extras[
                                    f"cross_covariance_{observation_source}"
                                ][query_index, token_index]),
                                "target": float(covariance_target[
                                    query_index, token_index]),
                                "weight": float(
                                    precision[query_index, token_index]),
                            })
                    query_indices = torch.as_tensor(
                        [row[0] for row in selected], dtype=torch.long)
                    token_indices = torch.as_tensor(
                        [row[1] for row in selected], dtype=torch.long)
                    reverse_query = payloads[observation_source]["coords"][
                        query_indices, token_indices]
                    reverse_target = coords[query_indices]
                    reverse_background = extras["ne_bkg"].squeeze(-1)[query_indices]
                    reverse_sw = sw_manager.get_drivers_sequence(reverse_query[:, 3])
                    reverse_peak = (
                        iri_peak_manager.get_iri_peak(reverse_query)
                        if iri_peak_manager is not None else None)
                    target_sw = sw_manager.get_drivers_sequence(reverse_target[:, 3])
                    target_peak = (
                        iri_peak_manager.get_iri_peak(reverse_target)
                        if iri_peak_manager is not None else None)
                    reverse_covariance = _direct_cross_covariance(
                        model, reverse_query, reverse_target, reverse_background,
                        reverse_sw, reverse_peak, target_sw, target_peak)
                    forward_covariance = extras[
                        f"cross_covariance_{observation_source}"
                    ][query_indices, token_indices]
                    for offset, (query_index, token_index) in enumerate(selected):
                        reciprocity_rows[observation_source].append({
                            "target_profile_id": int(profile_ids[query_index]),
                            "observation_profile_id": int(
                                observation_ids[query_index, token_index]),
                            "cell_id": int(cells[query_index, token_index]),
                            "forward": float(forward_covariance[offset]),
                            "reverse": float(reverse_covariance[offset]),
                            "weight": float(precision[query_index, token_index]),
                        })
        result[target_source] = {
            "selected_profiles": int(sum(
                len(values) for values in
                loaders[target_source].batch_sampler.profiles_by_bin.values())),
            "observation_sources": {
                observation_source: {
                    "reciprocity": _reciprocity_summary(
                        reciprocity_rows[observation_source]),
                    "stable_empirical_covariance": _covariance_target_summary(
                        covariance_rows[observation_source]),
                    "unique_profile_observation_geometry": _summarize_geometry(
                        geometry_rows[observation_source]),
                    "endpoint_context_variation": _summarize_endpoint_context(
                        context_rows[observation_source]),
                }
                for observation_source in SOURCES
            },
            "loss_exposure": _summarize_exposure(exposure_rows),
        }
    return result


def _gradient_audit(
        model, loaders, processor, sw_manager, iri_peak_manager,
        allowed, config, batches):
    _set_training_stage(model, "analysis")
    model.eval()
    parameter_groups = {
        "density_basis": list(model.density_basis_decoder.parameters()),
        "ensemble_scale": list(
            model.kalman_layer.covariance_scale_net.parameters()),
    }
    rows = {name: [] for name in parameter_groups}
    for batch_index, (fy_batch, cosmic_batch) in enumerate(zip(
            loaders["FY"], loaders["COSMIC"])):
        if batch_index >= batches:
            break
        losses = _paired_analysis_losses(
            model, processor, fy_batch, cosmic_batch, torch.device("cpu"),
            config, sw_manager, iri_peak_manager, allowed)
        named_losses = dict(zip(
            ("observation", "covariance", "direction"), losses))
        for name, parameters in parameter_groups.items():
            rows[name].append(_gradient_comparison(named_losses, parameters))
    if not rows["density_basis"]:
        raise RuntimeError("gradient audit produced no batches")
    loss_weights = {
        "observation": 1.0,
        "covariance": float(config.get("resolved_covariance_weight", 0.0)),
        "direction": (
            float(config.get("resolved_direction_weight", 0.0))
            if config.get("use_direction_loss", False) else 0.0),
    }
    return {
        "batches": len(rows["density_basis"]),
        "loss_weights_in_checkpoint_run": loss_weights,
        "parameter_groups": {
            name: {
                "norms": {
                    loss: _quantiles([
                        row["norms"][loss] for row in values])
                    for loss in ("observation", "covariance", "direction")
                },
                "weighted_norms": {
                    loss: _quantiles([
                        row["norms"][loss] * loss_weights[loss]
                        for row in values])
                    for loss in ("observation", "covariance", "direction")
                },
                "cosine_observation_covariance": _quantiles([
                    row["cosine_observation_covariance"] for row in values]),
                "cosine_observation_direction": _quantiles([
                    row["cosine_observation_direction"] for row in values]),
                "cosine_covariance_direction": _quantiles([
                    row["cosine_covariance_direction"] for row in values]),
            }
            for name, values in rows.items()
        },
    }


def _decision(report):
    reciprocity = []
    ratios = []
    covariance_accuracy = []
    absolute_errors = []
    for partition in report["partitions"].values():
        for target_source in SOURCES:
            for observation_source in SOURCES:
                values = partition[target_source]["observation_sources"][
                    observation_source]
                reciprocal = values["reciprocity"]
                if reciprocal["sign_reciprocity"] is not None:
                    reciprocity.append(reciprocal["sign_reciprocity"])
                    ratios.append(reciprocal["absolute_ratio"]["q05_q50_q95"][1])
                    absolute_errors.append(reciprocal["max_absolute_error"])
                accuracy = values["stable_empirical_covariance"]["sign_accuracy"]
                if accuracy is not None:
                    covariance_accuracy.append(accuracy)
    reciprocity_failed = (
        len(reciprocity) != 4 * len(report["partitions"])
        or min(reciprocity) < 0.95
        or min(ratios) < 0.8
        or max(ratios) > 1.25
    )
    covariance_failed = (
        len(covariance_accuracy) != 4 * len(report["partitions"])
        or min(covariance_accuracy) < 0.60)
    gradient = report["gradient_audit"]["parameter_groups"]["density_basis"]
    cosine_summary = gradient["cosine_observation_covariance"]
    observation_covariance_cosine = (
        cosine_summary["mean"] if cosine_summary is not None else math.nan)
    gradient_conflict = (
        math.isfinite(observation_covariance_cosine)
        and observation_covariance_cosine < 0.0)
    symmetric = report.get("density_basis_semantics") in (
        "coordinate_local_symmetric", "endpoint_context_symmetric")
    if symmetric:
        observation_norm = gradient["norms"]["observation"]
        covariance_norm = gradient["norms"]["covariance"]
        gradient_failed = any(
            summary is None or not math.isfinite(summary["mean"])
            or summary["mean"] <= 0.0
            for summary in (observation_norm, covariance_norm))
        reciprocity_failed = (
            len(absolute_errors) != 4 * len(report["partitions"])
            or max(absolute_errors) >= 1e-7)
        if reciprocity_failed or gradient_failed:
            conclusion = "对称映射数学预检失败"
            next_step = "停止训练，定位互易误差或零梯度"
        else:
            conclusion = "对称映射数学预检通过"
            next_step = "只允许一次对应语义的五epoch内部screen；仍不读取ISR"
    elif reciprocity_failed or covariance_failed:
        conclusion = "共享density basis互易性或稳定经验协方差映射失败"
        next_step = "设计坐标局地、两端共享的对称density basis；暂不训练"
    elif gradient_conflict:
        conclusion = "观测主损失与经验协方差监督存在梯度冲突"
        next_step = "先修正loss统计与监督语义；暂不修改R、N8或联合ETKF"
    else:
        conclusion = "共享映射通过，优先检查失败单元主损失梯度暴露"
        next_step = "规划profile内高度×昼夜×纬度平衡的单因素主损失"
    return {
        "reciprocity_thresholds": {
            "sign_minimum": 0.95,
            "median_absolute_ratio": [0.8, 1.25],
        },
        "stable_covariance_sign_minimum": 0.60,
        "reciprocity_failed": reciprocity_failed,
        "stable_covariance_mapping_failed": covariance_failed,
        "maximum_absolute_reciprocity_error": (
            max(absolute_errors) if absolute_errors else None),
        "observation_covariance_gradient_conflict": gradient_conflict,
        "conclusion": conclusion,
        "next_step": next_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("epoch_07_model.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-per-source", type=int, default=512)
    parser.add_argument("--gradient-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--density-basis-semantics",
        choices=("query_conditioned", "coordinate_local_symmetric",
                 "endpoint_context_symmetric"),
        default=None)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.profiles_per_source < 1 or args.gradient_batches < 1:
        raise ValueError("profile and gradient batch counts must be positive")
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    model, sw_manager, iri_peak_manager, config = _load(run_dir, checkpoint)
    if args.density_basis_semantics is not None:
        if (args.density_basis_semantics in (
                "coordinate_local_symmetric", "endpoint_context_symmetric")
                and model.kalman_layer.anomaly_parameterization
                != "orthogonal_factor"):
            raise ValueError(
                "coordinate-local symmetric audit requires orthogonal_factor")
        model.density_basis_semantics = args.density_basis_semantics
        model.kalman_layer.coordinate_local_symmetric = (
            args.density_basis_semantics in (
                "coordinate_local_symmetric", "endpoint_context_symmetric"))
        config["density_basis_semantics"] = args.density_basis_semantics
    with (run_dir / "run_manifest.json").open(encoding="utf-8") as stream:
        run_manifest = json.load(stream)
    resolved_training = run_manifest.get("resolved_training", {})
    for key in ("resolved_covariance_weight", "resolved_direction_weight"):
        if resolved_training.get(key) is not None:
            config[key] = float(resolved_training[key])
    split_path = Path(config["date_split_manifest"])
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    with split_path.open(encoding="utf-8") as stream:
        split_days = json.load(stream)["partitions"]
    if set(split_days) != {"train", "development", "locked_test"}:
        raise ValueError("unexpected date partition schema")
    indices = {
        "FY": FYNeighborhoodIndex(config["fy_path"], config),
        "COSMIC": COSMICNeighborhoodIndex(config["cosmic_path"], config),
    }
    empirical_path = Path(config["representativeness_kernel_path"])
    if not empirical_path.is_absolute():
        empirical_path = ROOT / empirical_path
    empirical_targets = load_empirical_covariance_targets(empirical_path)
    representativeness = load_representativeness_kernel(empirical_path)
    partitions = {}
    train_loaders = None
    train_allowed = None
    selected = {}
    selection_coverage = {}
    audit_partitions = ("train",) if args.train_only else (
        "train", "development")
    for partition_index, partition in enumerate(audit_partitions):
        loaders = {
            "FY": _partition_loader(FY3D_Dataset, config, split_days, partition),
            "COSMIC": _partition_loader(COSMICDataset, config, split_days, partition),
        }
        allowed = {
            source: np.unique(loader.dataset.profile_ids)
            for source, loader in loaders.items()
        }
        selected[partition] = {
            source: _restrict_profiles(
                loader, args.profiles_per_source,
                args.seed + partition_index * 10 + source_index)
            for source_index, (source, loader) in enumerate(loaders.items())
        }
        selection_coverage[partition] = {
            source: _selection_coverage(loader)
            for source, loader in loaders.items()
        }
        partitions[partition] = _audit_partition(
            partition, loaders, model, sw_manager, iri_peak_manager,
            indices, allowed, empirical_targets, representativeness,
            float(config["representativeness_floor"]))
        if partition == "train":
            train_loaders, train_allowed = loaders, allowed
    processor = SlidingWindowBatchProcessor(
        sw_manager, device="cpu", fy_nb_index=indices["FY"],
        cosmic_nb_index=indices["COSMIC"],
        representativeness_kernel=representativeness,
        representativeness_floor=float(config["representativeness_floor"]),
        empirical_covariance_targets=empirical_targets)
    gradient = _gradient_audit(
        model, train_loaders, processor, sw_manager, iri_peak_manager,
        train_allowed, config, args.gradient_batches)
    report = {
        "schema_version": 1,
        "purpose": ({
            "coordinate_local_symmetric": (
                "M2-N coordinate-local symmetric mapping preflight"),
            "endpoint_context_symmetric": (
                "M2-O endpoint-context symmetric mapping preflight"),
        }.get(model.density_basis_semantics,
              "M2-M shared mapping reciprocity and loss gradient audit")),
        "research_period": "2024-09",
        "density_basis_semantics": model.density_basis_semantics,
        "seed": args.seed,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "strict_load": True,
            "all_tensors_finite": True,
        },
        "audit_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "data_boundaries": {
            "partitions_accessed": list(audit_partitions),
            "partition_days": {
                partition: list(split_days[partition])
                for partition in audit_partitions
            },
            "locked_test_accessed": False,
            "isr_accessed": False,
            "date_split_manifest": str(split_path.resolve()),
            "date_split_sha256": _sha256(split_path),
        },
        "sampling": {
            "unit": "complete profile with 8 deterministic original points",
            "profiles_per_source_requested": args.profiles_per_source,
            "selected_profiles": {
                partition: {
                    source: int(len(values))
                    for source, values in sources.items()
                }
                for partition, sources in selected.items()
            },
            "selection_coverage": selection_coverage,
        },
        "resolved_training": {
            "use_empirical_covariance_loss": bool(
                config.get("use_empirical_covariance_loss", False)),
            "resolved_covariance_weight": config.get(
                "resolved_covariance_weight"),
            "use_direction_loss": bool(config.get("use_direction_loss", False)),
            "resolved_direction_weight": config.get("resolved_direction_weight"),
        },
        "partitions": partitions,
        "gradient_audit": gradient,
    }
    report["decision"] = _decision(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
