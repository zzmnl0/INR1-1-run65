"""Read-only audit of N=8 ETKF and observation-space anomaly geometry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
from inr_modules.mdia.fsia_model import FSIA_INR_Model, solve_density_modes
from inr_modules.mdia.sliding_dataset import (
    attach_representativeness_weight,
    attach_observation_background,
    load_representativeness_kernel,
    query_observation_payload,
)


def _restrict_profiles(loader, maximum, seed):
    sampler = loader.batch_sampler
    sampler.points_per_profile = 8
    groups = [
        indices
        for values in sampler.profiles_by_bin.values()
        for indices in values
    ]
    profile_ids = np.asarray([
        int(loader.dataset.profile_ids[indices[0]]) for indices in groups
    ])
    order = np.argsort(profile_ids, kind="stable")
    profile_ids = profile_ids[order]
    if len(profile_ids) > maximum:
        rng = np.random.default_rng(seed)
        profile_ids = profile_ids[np.sort(
            rng.choice(len(profile_ids), maximum, replace=False)
        )]
    selected = set(profile_ids.tolist())
    sampler.profiles_by_bin = {
        key: [
            indices for indices in values
            if int(loader.dataset.profile_ids[indices[0]]) in selected
        ]
        for key, values in sampler.profiles_by_bin.items()
    }
    sampler.profiles_by_bin = {
        key: values for key, values in sampler.profiles_by_bin.items() if values
    }
    return profile_ids


def _geometry(matrix, precision):
    valid = precision > 0
    weighted = matrix[valid] * precision[valid].sqrt().unsqueeze(-1)
    if weighted.numel() == 0:
        return None, None
    singular = torch.linalg.svdvals(weighted)
    largest = singular[0].clamp_min(torch.finfo(singular.dtype).eps)
    positive = singular[singular > largest * 1e-6]
    energy = positive.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    effective = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum()
    )
    gram = weighted.T @ weighted
    return {
        "tokens": int(valid.sum()),
        "numeric_rank": int(len(positive)),
        "effective_rank": float(effective),
        "condition": float(positive[0] / positive[-1]),
        "first_mode_energy": float(energy[0] / energy.sum().clamp_min(1e-12)),
    }, gram


def _summary(rows):
    if not rows:
        return {"queries": 0}
    result = {"queries": len(rows)}
    for key in (
            "tokens", "numeric_rank", "effective_rank", "condition",
            "first_mode_energy"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"{key}_q05_q50_q95"] = np.quantile(
            values, [0.05, 0.5, 0.95]
        ).tolist()
    result["rank7_fraction"] = float(np.mean([
        row["numeric_rank"] >= 7 for row in rows
    ]))
    return result


def _direction_summary(rows):
    if not rows:
        return {"profiles": 0, "queries": 0, "direction_fraction": None}
    profile_rates = [
        float(np.mean(values)) for values in rows.values() if values
    ]
    return {
        "profiles": len(profile_rates),
        "queries": int(sum(len(values) for values in rows.values())),
        "direction_fraction": float(np.mean(profile_rates)),
    }


def _load(run_dir, checkpoint):
    with (run_dir / "run_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    config = dict(manifest["config"])
    device = torch.device("cpu")
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
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
    hmf2, nmf2 = config.get("iri_hmf2_path"), config.get("iri_nmf2_path")
    if hmf2 and nmf2 and Path(hmf2).is_file() and Path(nmf2).is_file():
        from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
        iri_peak_manager = IRIPeakManager(hmf2, nmf2, device=device)
    return model, sw_manager, iri_peak_manager, config


def _audit_source(
        source, loader, model, sw_manager, iri_peak_manager, indices, allowed,
        representativeness_grid=None, representativeness_floor=0.25):
    rows = {name: [] for name in ("FY", "COSMIC", "joint")}
    cells = {
        name: defaultdict(list) for name in ("FY", "COSMIC", "joint")
    }
    overlaps = []
    overlap_cells = defaultdict(list)
    directions = {
        mode: defaultdict(lambda: defaultdict(list))
        for mode in ("M10", "M01", "M11")
    }
    with torch.no_grad():
        for data, _, profile_ids in loader:
            coords, target = data[:, :4], data[:, 4]
            sw = sw_manager.get_drivers_sequence(coords[:, 3])
            peak = (
                iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None
            )
            payloads = {}
            for observation_source in ("FY", "COSMIC"):
                exclude = (
                    profile_ids.numpy()
                    if observation_source == source else None
                )
                payload = query_observation_payload(
                    indices[observation_source],
                    coords,
                    torch.device("cpu"),
                    exclude_profile_ids=exclude,
                    allowed_profile_ids=allowed[observation_source],
                )
                payload = attach_representativeness_weight(
                    payload, coords, source, observation_source,
                    representativeness_grid, representativeness_floor)
                payloads[observation_source] = attach_observation_background(
                    payload, model, sw_manager, iri_peak_manager
                )
            _, _, _, _, extras = model(
                coords,
                sw,
                iri_peak=peak,
                observations_fy=payloads["FY"],
                observations_cosmic=payloads["COSMIC"],
            )
            coords_np = coords.numpy()
            altitude, day, latitude = _cell_ids(coords_np)
            cell_names = [
                _cell_name(altitude[i], day[i], latitude[i])
                for i in range(len(coords))
            ]
            mode_increments = solve_density_modes(extras)
            desired = target - extras["ne_bkg"].squeeze(-1)
            active = {
                "M10": extras["precision_FY"].sum(-1) > 0,
                "M01": extras["precision_COSMIC"].sum(-1) > 0,
                "M11": (
                    extras["precision_FY"].sum(-1)
                    + extras["precision_COSMIC"].sum(-1)
                ) > 0,
            }
            for index in range(len(coords)):
                source_geometry = {}
                source_grams = {}
                for observation_source in ("FY", "COSMIC"):
                    geometry, gram = _geometry(
                        extras[f"obs_anomalies_{observation_source}"][index],
                        extras[f"precision_{observation_source}"][index],
                    )
                    if geometry is not None:
                        rows[observation_source].append(geometry)
                        cells[observation_source][cell_names[index]].append(
                            geometry
                        )
                        source_geometry[observation_source] = geometry
                        source_grams[observation_source] = gram
                matrices, precisions = [], []
                for observation_source in ("FY", "COSMIC"):
                    matrices.append(
                        extras[f"obs_anomalies_{observation_source}"][index]
                    )
                    precisions.append(
                        extras[f"precision_{observation_source}"][index]
                    )
                geometry, _ = _geometry(
                    torch.cat(matrices), torch.cat(precisions)
                )
                if geometry is not None:
                    rows["joint"].append(geometry)
                    cells["joint"][cell_names[index]].append(geometry)
                if len(source_grams) == 2:
                    fy_gram, cosmic_gram = (
                        source_grams["FY"], source_grams["COSMIC"]
                    )
                    denominator = (
                        torch.linalg.norm(fy_gram)
                        * torch.linalg.norm(cosmic_gram)
                    ).clamp_min(1e-12)
                    overlap = float(
                        (fy_gram * cosmic_gram).sum() / denominator
                    )
                    overlaps.append(overlap)
                    overlap_cells[cell_names[index]].append(overlap)
                for mode in directions:
                    if active[mode][index] and abs(float(desired[index])) >= 0.05:
                        correct = bool(
                            desired[index] * mode_increments[mode][index] > 0
                        )
                        directions[mode][cell_names[index]][
                            int(profile_ids[index])
                        ].append(correct)
    return {
        "observation_anomaly_geometry": {
            name: {
                "global": _summary(rows[name]),
                "strata": {
                    cell: _summary(values)
                    for cell, values in sorted(cells[name].items())
                },
            }
            for name in rows
        },
        "source_gram_overlap_q05_q50_q95": (
            np.quantile(overlaps, [0.05, 0.5, 0.95]).tolist()
            if overlaps else None
        ),
        "source_gram_overlap_strata": {
            cell: {
                "queries": len(values),
                "q05_q50_q95": np.quantile(
                    values, [0.05, 0.5, 0.95]
                ).tolist(),
            }
            for cell, values in sorted(overlap_cells.items())
        },
        "direction_by_stratum": {
            mode: {
                cell: _direction_summary(values)
                for cell, values in sorted(directions[mode].items())
            }
            for mode in directions
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("best_fsia_model.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-profiles", type=int, default=512)
    args = parser.parse_args()
    if args.max_profiles < 1:
        raise ValueError("max-profiles must be positive")
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    model, sw_manager, iri_peak_manager, config = _load(run_dir, checkpoint)
    date_manifest = Path(config["date_split_manifest"])
    if not date_manifest.is_absolute():
        date_manifest = ROOT / date_manifest
    with date_manifest.open(encoding="utf-8") as stream:
        split_days = json.load(stream)["partitions"]
    loaders = {
        "FY": _partition_loader(
            FY3D_Dataset, config, split_days, "development"
        ),
        "COSMIC": _partition_loader(
            COSMICDataset, config, split_days, "development"
        ),
    }
    allowed = {
        source: np.unique(loader.dataset.profile_ids)
        for source, loader in loaders.items()
    }
    selected = {
        source: _restrict_profiles(loader, args.max_profiles, 42 + index)
        for index, (source, loader) in enumerate(loaders.items())
    }
    indices = {
        "FY": FYNeighborhoodIndex(config["fy_path"], config),
        "COSMIC": COSMICNeighborhoodIndex(config["cosmic_path"], config),
    }
    layer = model.kalman_layer
    if layer.anomaly_parameterization != "orthogonal_factor":
        raise ValueError("subspace audit requires orthogonal_factor anomalies")
    coefficients = layer.ensemble_coefficients.double()
    basis = layer.state_basis.double()
    rank = layer.n_members - 1
    representativeness_path = config.get('representativeness_kernel_path')
    if representativeness_path:
        representativeness_path = Path(representativeness_path)
        if not representativeness_path.is_absolute():
            representativeness_path = ROOT / representativeness_path
    representativeness_grid = load_representativeness_kernel(
        representativeness_path)
    representativeness_floor = float(
        config.get('representativeness_floor', 0.25))
    report = {
        "schema_version": 1,
        "purpose": "ISR-blind N8 and observation-space anomaly audit",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "strict_load": True,
            "all_tensors_finite": True,
        },
        "selected_profiles": {
            source: int(len(values)) for source, values in selected.items()
        },
        "analytic_geometry": {
            "members": layer.n_members,
            "maximum_zero_mean_rank": rank,
            "coefficient_zero_mean_max_error": float(
                (coefficients @ torch.ones(layer.n_members, dtype=torch.float64))
                .abs().max()
            ),
            "coefficient_gram_max_error": float(
                (coefficients @ coefficients.T
                 - rank * torch.eye(rank, dtype=torch.float64)).abs().max()
            ),
            "state_basis_gram_max_error": float(
                (basis.T @ basis
                 - torch.eye(rank, dtype=torch.float64)).abs().max()
            ),
        },
        "targets": {
            source: _audit_source(
                source,
                loaders[source],
                model,
                sw_manager,
                iri_peak_manager,
                indices,
                allowed,
                representativeness_grid,
                representativeness_floor,
            )
            for source in ("FY", "COSMIC")
        },
    }
    analytic = report["analytic_geometry"]
    report["analytic_geometry"]["passed"] = all(
        analytic[key] < 1e-7
        for key in (
            "coefficient_zero_mean_max_error",
            "coefficient_gram_max_error",
            "state_basis_gram_max_error",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
