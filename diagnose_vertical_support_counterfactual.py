"""Read-only ISR counterfactuals for vertical observation support."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    query_observation_payload,
)
from isr_evaluation.coord_convert import convert_day_record_cgm
from isr_evaluation.diagnose_jicamarca_modes import (
    _record_arrays,
    _source_arrays,
    _summary,
)
from isr_evaluation.isr_loader import load_jicamarca, load_poker_flat
from isr_evaluation.main_isr_eval import (
    CONFIG as ISR_CONFIG,
    _load_model_and_managers,
    _parse_unix,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    ROOT / "checkpoints_fsia"
    / "run66-covfactor-d64-n8-global-localized"
    / "best_fsia_model.pth"
)
OUTPUT = (
    ROOT / "isr_validation_outputs"
    / "run66-hcut-original-curve-audit"
    / "model_attribution"
)
FILTERS = ("all", "same_layer", "within_20km")
MODES = ("M00", "M10", "M01", "M11")


def _vertical_mask(
    payload: dict[str, torch.Tensor],
    query_coords: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    valid = payload["valid_mask"]
    if mode == "all":
        return valid
    observation_altitude = payload["coords"][..., 2]
    query_altitude = query_coords[:, None, 2]
    if mode == "within_20km":
        return valid & ((observation_altitude - query_altitude).abs() <= 20.0)
    if mode != "same_layer":
        raise ValueError(f"unknown vertical filter: {mode}")
    edges = torch.tensor(
        [120.0, 200.0, 300.0, 500.0001],
        device=query_coords.device,
        dtype=query_coords.dtype,
    )
    query_layer = torch.bucketize(
        query_altitude.contiguous(), edges[1:-1], right=True
    )
    observation_layer = torch.bucketize(
        observation_altitude.contiguous(), edges[1:-1], right=True
    )
    return valid & (query_layer == observation_layer)


def _filtered_payload(payload, query_coords, mode):
    filtered = dict(payload)
    filtered["valid_mask"] = _vertical_mask(payload, query_coords, mode)
    return filtered


def _validate_payload(name, source, payload):
    valid = payload["valid_mask"]
    for field in ("coords", "value", "background", "rho_squared"):
        values = payload[field]
        selected = values[valid] if values.ndim == 2 else values[valid, :]
        if not torch.isfinite(selected).all():
            raise ValueError(
                f"{name} {source} payload has non-finite valid {field}"
            )


def _sample_points(coords, observation, limit, seed):
    if len(coords) <= limit:
        return coords, observation
    local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    altitude_bin = np.floor((coords[:, 2] - 120.0) / 20.0).astype(np.int16)
    day = np.floor(coords[:, 3] / 24.0).astype(np.int16)
    regime = ((local_time >= 6.0) & (local_time < 18.0)).astype(np.int8)
    groups = {}
    for index, key in enumerate(zip(day, altitude_bin, regime)):
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(seed)
    selected = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if not groups[key]:
                continue
            choice = int(rng.integers(len(groups[key])))
            selected.append(groups[key].pop(choice))
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    return coords[selected], observation[selected]


def _load_station(name, config, start_unix, end_unix):
    common = {
        "start_unix": start_unix,
        "end_unix": end_unix,
        "alt_min": config["alt_min"],
        "alt_max": config["alt_max"],
        "err_ratio_max": config["err_ratio_max"],
    }
    if name == "Jicamarca":
        return load_jicamarca(config["jicamarca_dir"], **common)
    if name == "PokerFlat":
        records = load_poker_flat(config["poker_flat_dir"], **common)
        for record in records:
            convert_day_record_cgm(record)
        return records
    raise ValueError(name)


def _flatten_station(records, start_unix, limit, seed):
    coords, observations = [], []
    for record in records:
        record_coords, record_observation = _record_arrays(record, start_unix)
        coords.append(record_coords)
        observations.append(record_observation)
    if not coords:
        raise ValueError("station has no valid ISR observations")
    return _sample_points(
        np.concatenate(coords),
        np.concatenate(observations),
        limit,
        seed,
    )


def _direction_summary(mask, values, background, observation):
    summary = _summary(mask, values, background, observation)
    if not summary["n"]:
        return summary
    increment = values[mask] - background[mask]
    desired = observation[mask] - background[mask]
    selected = (np.abs(increment) > 1e-8) & (np.abs(desired) > 1e-8)
    summary["toward_isr_fraction"] = (
        float(np.mean(np.sign(increment[selected]) == np.sign(desired[selected])))
        if selected.any() else float("nan")
    )
    return summary


def _source_physics(source_arrays, desired, mask=None):
    population = (
        np.ones(len(desired), dtype=bool)
        if mask is None else np.asarray(mask, dtype=bool)
    )
    covered = (source_arrays["valid_tokens"] > 0) & population
    innovation = source_arrays["physical_innovation_mean"]
    contribution = source_arrays["kalman_contribution_sum"]
    comparable_innovation = covered & (np.abs(innovation) > 1e-8) & (
        np.abs(desired) > 1e-8
    )
    comparable_contribution = covered & (np.abs(contribution) > 1e-8) & (
        np.abs(desired) > 1e-8
    )
    return {
        "n": int(covered.sum()),
        "coverage": (
            float(covered.sum() / population.sum())
            if population.any() else float("nan")
        ),
        "innovation_mean": float(np.mean(innovation[covered])) if covered.any()
        else float("nan"),
        "cross_covariance_mean": float(np.mean(
            source_arrays["cross_covariance_mean"][covered]
        )) if covered.any() else float("nan"),
        "kalman_gain_mean": float(np.mean(
            source_arrays["kalman_gain_mean"][covered]
        )) if covered.any() else float("nan"),
        "kalman_contribution_mean": float(np.mean(
            contribution[covered]
        )) if covered.any() else float("nan"),
        "innovation_toward_isr_fraction": (
            float(np.mean(
                np.sign(innovation[comparable_innovation])
                == np.sign(desired[comparable_innovation])
            ))
            if comparable_innovation.any() else float("nan")
        ),
        "contribution_toward_isr_fraction": (
            float(np.mean(
                np.sign(contribution[comparable_contribution])
                == np.sign(desired[comparable_contribution])
            ))
            if comparable_contribution.any() else float("nan")
        ),
        "negative_cross_covariance_fraction": float(np.mean(
            source_arrays["cross_covariance_mean"][covered] < 0.0
        )) if covered.any() else float("nan"),
        "negative_gain_fraction": float(np.mean(
            source_arrays["kalman_gain_mean"][covered] < 0.0
        )) if covered.any() else float("nan"),
        "effective_sample_size_mean": float(np.mean(
            source_arrays["effective_sample_size"][covered]
        )) if covered.any() else 0.0,
    }


def _run_station(
    name,
    records,
    start_unix,
    model,
    sw_manager,
    iri_peak_manager,
    fy_index,
    cosmic_index,
    device,
    limit,
    batch_size,
):
    coords_np, observation = _flatten_station(
        records, start_unix, limit, 42 if name == "Jicamarca" else 43
    )
    saved = {
        vertical_filter: {
            key: [] for key in (
                "coords", "observation", "background",
                *(f"{mode}_prediction" for mode in MODES),
                "fy_valid_tokens", "fy_physical_innovation_mean",
                "fy_kalman_gain_mean", "fy_cross_covariance_mean",
                "fy_kalman_contribution_sum", "fy_effective_sample_size",
                "cosmic_valid_tokens", "cosmic_physical_innovation_mean",
                "cosmic_kalman_gain_mean", "cosmic_cross_covariance_mean",
                "cosmic_kalman_contribution_sum",
                "cosmic_effective_sample_size",
            )
        }
        for vertical_filter in FILTERS
    }
    with torch.no_grad():
        for start in range(0, len(coords_np), batch_size):
            chunk_np = coords_np[start : start + batch_size]
            chunk_observation = observation[start : start + batch_size]
            coords = torch.from_numpy(chunk_np).to(device)
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            iri_peak = (
                iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None
            )
            fy = attach_observation_background(
                query_observation_payload(fy_index, coords, device),
                model,
                sw_manager,
                iri_peak_manager,
            )
            cosmic = attach_observation_background(
                query_observation_payload(cosmic_index, coords, device),
                model,
                sw_manager,
                iri_peak_manager,
            )
            _validate_payload(name, "FY", fy)
            _validate_payload(name, "COSMIC", cosmic)
            for vertical_filter in FILTERS:
                fy_filtered = _filtered_payload(fy, coords, vertical_filter)
                cosmic_filtered = _filtered_payload(
                    cosmic, coords, vertical_filter
                )
                kwargs = {
                    "M00": {},
                    "M10": {"observations_fy": fy_filtered},
                    "M01": {"observations_cosmic": cosmic_filtered},
                    "M11": {
                        "observations_fy": fy_filtered,
                        "observations_cosmic": cosmic_filtered,
                    },
                }
                extras = {}
                background = None
                target = saved[vertical_filter]
                for mode in MODES:
                    try:
                        prediction, _, _, _, mode_extras = model(
                            coords, sw_seq, iri_peak=iri_peak, **kwargs[mode]
                        )
                    except RuntimeError as error:
                        raise RuntimeError(
                            f"{name} filter={vertical_filter} mode={mode} "
                            f"batch={start}:{start + len(chunk_np)}"
                        ) from error
                    extras[mode] = mode_extras
                    target[f"{mode}_prediction"].append(
                        prediction.squeeze(-1).cpu().numpy()
                    )
                    if background is None:
                        background = (
                            mode_extras["ne_bkg"].squeeze(-1).cpu().numpy()
                        )
                for prefix, payload, mode, suffix in (
                    ("fy", fy_filtered, "M10", "FY"),
                    ("cosmic", cosmic_filtered, "M01", "COSMIC"),
                ):
                    source_arrays = _source_arrays(
                        payload, extras[mode], suffix
                    )
                    for field in (
                        "valid_tokens", "physical_innovation_mean",
                        "kalman_gain_mean", "cross_covariance_mean",
                        "kalman_contribution_sum", "effective_sample_size",
                    ):
                        target[f"{prefix}_{field}"].append(
                            source_arrays[field]
                        )
                target["coords"].append(chunk_np)
                target["observation"].append(chunk_observation)
                target["background"].append(background)

    result = {"sample_points": len(coords_np), "filters": {}}
    for vertical_filter, parts in saved.items():
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        coords = arrays["coords"]
        local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
        low_night = (
            (coords[:, 2] >= 120.0) & (coords[:, 2] < 300.0)
            & ((local_time < 6.0) | (local_time >= 18.0))
        )
        desired = arrays["observation"] - arrays["background"]
        result["filters"][vertical_filter] = {
            "modes": {
                mode: {
                    "all": _direction_summary(
                        np.ones(len(coords), dtype=bool),
                        arrays[f"{mode}_prediction"],
                        arrays["background"],
                        arrays["observation"],
                    ),
                    "night_120_300km": _direction_summary(
                        low_night,
                        arrays[f"{mode}_prediction"],
                        arrays["background"],
                        arrays["observation"],
                    ),
                }
                for mode in MODES
            },
            "source_physics": {
                source: {
                    group: _source_physics(
                        {
                            field: arrays[f"{source}_{field}"]
                            for field in (
                                "valid_tokens", "physical_innovation_mean",
                                "kalman_gain_mean", "cross_covariance_mean",
                                "kalman_contribution_sum",
                                "effective_sample_size",
                            )
                        },
                        desired,
                        mask,
                    )
                    for group, mask in (
                        ("all", None),
                        ("night_120_300km", low_night),
                    )
                }
                for source in ("fy", "cosmic")
            },
        }
    return result


def _classify(report):
    jicamarca = report["stations"]["Jicamarca"]["filters"]
    baseline = jicamarca["all"]["modes"]
    local = jicamarca["within_20km"]["modes"]
    improvements = {}
    for mode in ("M10", "M01"):
        base = baseline[mode]["night_120_300km"]
        narrowed = local[mode]["night_120_300km"]
        improvements[mode] = {
            "toward_isr_change": (
                narrowed["toward_isr_fraction"]
                - base["toward_isr_fraction"]
            ),
            "rmse_change": narrowed["rmse"] - base["rmse"],
        }
    cross_altitude = any(
        value["toward_isr_change"] >= 0.10 and value["rmse_change"] < 0.0
        for value in improvements.values()
    )
    report["attribution"] = {
        "within_20km_vs_all": improvements,
        "primary_cause": (
            "unsupported_cross_altitude_covariance"
            if cross_altitude
            else "vertical_restriction_does_not_explain_degradation"
        ),
        "recommended_change": (
            "plan train-only residual-scale continuous vertical localization"
            if cross_altitude
            else "retain current vertical support and audit representativeness/"
                 "source interaction before model changes"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-points-per-station", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    device = torch.device("cpu")
    config = dict(ISR_CONFIG)
    config["model_type"] = "fsia"
    config["checkpoint_path"] = str(args.checkpoint.resolve())
    (
        model,
        sw_manager,
        _,
        _,
        iri_peak_manager,
        fy_index,
        cosmic_index,
    ) = _load_model_and_managers(config, device)
    start_unix = _parse_unix(config["start_date_str"])
    end_unix = _parse_unix(config["end_date_str"])
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "filters": {
            "all": "unchanged observation payload",
            "same_layer": "120-200/200-300/300-500 km layer match",
            "within_20km": "absolute observation-query altitude <=20 km",
        },
        "stations": {},
    }
    failures = []
    for station in ("Jicamarca", "PokerFlat"):
        records = _load_station(station, config, start_unix, end_unix)
        try:
            report["stations"][station] = _run_station(
                station,
                records,
                start_unix,
                model,
                sw_manager,
                iri_peak_manager,
                fy_index,
                cosmic_index,
                device,
                args.max_points_per_station,
                args.batch_size,
            )
        except RuntimeError as error:
            failures.append(f"{station}: {error}")
            report["stations"][station] = {
                "status": "failed",
                "error": str(error),
            }
    if failures:
        report["attribution"] = {
            "primary_cause": "hard_padding_invariant_failure",
            "evidence": failures,
            "recommended_change": (
                "mask or replace invalid observation tokens before density-basis "
                "evaluation, then repeat the read-only vertical counterfactual"
            ),
        }
    else:
        _classify(report)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "vertical_support_counterfactual.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    print(json.dumps(report["attribution"], ensure_ascii=False, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
