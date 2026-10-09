"""Deterministic, read-only audit for the M00-FCB experiment.

The audit replays the existing profile sampler, evaluates the saved models on
complete profiles, and computes local loss gradients.  It never trains,
updates a parameter, runs ISR, constructs a locked-test loader, or computes
hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from inr_modules.data_managers.FY_dataloader import (
    get_cosmic_dataloader,
    get_dataloaders,
)
from inr_modules.mdia.fcb import FCB_CELL_LABELS, _cell_ids_numpy, fcb_selection_audit
from isr_evaluation.background_m00_eval import _load_background_model


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "checkpoints_fsia" / "run67-m2w-120-500-background-fcb"
SEEDS = tuple(range(42, 47))
GRADIENT_BATCHES = 20
TARGET_SOURCE_SHARE = 1.0 / len(FCB_CELL_LABELS)
TARGET_GLOBAL_SHARE = 1.0 / (2.0 * len(FCB_CELL_LABELS))
MAG_GROUPS = {
    "low": tuple(i for i, label in enumerate(FCB_CELL_LABELS) if "lt_0.25" in label),
    "mid": tuple(i for i, label in enumerate(FCB_CELL_LABELS) if "0.25_0.5" in label),
    "high": tuple(i for i, label in enumerate(FCB_CELL_LABELS) if "ge_0.5" in label),
}


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _finite_tree(value, path="root"):
    bad = []
    if isinstance(value, dict):
        for key, item in value.items():
            bad.extend(_finite_tree(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(_finite_tree(item, f"{path}[{index}]"))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        bad.append(path)
    return bad


def _checkpoint_summary(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state dict: {path}")
    tensors = [v for v in state.values() if torch.is_tensor(v)]
    finite = all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in tensors)
    return {
        "path": str(path.resolve()),
        "tensor_count": len(tensors),
        "element_count": int(sum(v.numel() for v in tensors)),
        "all_finite": bool(finite),
    }


def _split_days(summary: dict) -> dict:
    """Read partition metadata; no locked-test sample is loaded."""
    partitions = summary["date_split"]["partitions"]
    all_days = {name: [int(day) for day in values] for name, values in partitions.items()}
    for left in all_days:
        for right in all_days:
            if left < right and set(all_days[left]).intersection(all_days[right]):
                raise ValueError(f"date partition overlap: {left}/{right}")
    return {"train": all_days["train"], "development": all_days["development"]}


def _contract_path(run_dir: Path, config: dict) -> Path:
    default = (run_dir / "fcb_cell_contract.json").resolve()
    configured = config.get("fcb_cell_contract_path")
    if not configured:
        return default
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if candidate != default:
        raise ValueError(f"contract is outside audited run directory: {candidate}")
    return candidate


def _sampler_profile_ids(loader) -> set[int]:
    ids = set()
    for groups in loader.batch_sampler.profiles_by_bin.values():
        for indices in groups:
            if len(indices):
                ids.add(int(loader.dataset.profile_ids[indices[0]]))
    return ids


def _apply_profile_subset(loader, config: dict, source: str) -> dict:
    sampler = loader.batch_sampler
    available = _sampler_profile_ids(loader)
    fraction = float(config.get("train_profile_fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train_profile_fraction must be in (0,1]")
    manifest_path = config.get("profile_subset_manifest")
    if fraction < 1.0:
        if not manifest_path:
            raise ValueError("profile subset manifest missing")
        path = Path(manifest_path)
        if not path.is_absolute():
            path = ROOT / path
        manifest = _read_json(path)
        if int(manifest.get("seed", -1)) != int(config.get("seed", 42)):
            raise ValueError("profile subset seed mismatch")
        if not np.isclose(float(manifest.get("fraction", -1.0)), fraction):
            raise ValueError("profile subset fraction mismatch")
        selected = {int(x) for x in manifest["sources"][source]["profile_ids"]}
        if not selected.issubset(available):
            raise ValueError(f"subset contains unknown {source} profile")
        sampler.profiles_by_bin = {
            bin_id: [indices for indices in groups
                     if int(loader.dataset.profile_ids[indices[0]]) in selected]
            for bin_id, groups in sampler.profiles_by_bin.items()
        }
        sampler.profiles_by_bin = {k: v for k, v in sampler.profiles_by_bin.items() if v}
        active = selected
    else:
        active = available
    loader._audit_active_profile_ids = active
    return {
        "available": len(available),
        "selected": len(active),
        "profile_ids": active,
        "manifest": str(Path(manifest_path).resolve()) if manifest_path else None,
    }


def _make_loaders(config: dict, summary: dict, points_per_profile=8, full_validation_profiles=False):
    split_days = _split_days(summary)
    common = {
        "batch_size": int(config["batch_size"]),
        "bin_size_hours": float(config.get("bin_size_hours", 0.5)),
        "num_workers": 0,
        "use_memmap": bool(config.get("use_memmap", True)),
        "val_ratio": None,
        "split_seed": int(config.get("seed", 42)),
        "points_per_profile": points_per_profile,
        "split_days": split_days,
        "alt_range": tuple(config.get("observation_alt_range", (200.0, 500.0))),
        "full_validation_profiles": bool(full_validation_profiles),
    }
    fy_train, fy_dev = get_dataloaders(
        npy_path=config["fy_path"],
        val_days=None,
        profile_path=config.get("fy_profile_path"),
        profile_index_path=config.get("fy_profile_index_path"),
        **common,
    )
    cosmic_train, cosmic_dev = get_cosmic_dataloader(
        cosmic_path=config["cosmic_path"],
        profile_index_path=config.get("cosmic_profile_index_path"),
        **common,
    )
    train = {"FY": fy_train, "COSMIC": cosmic_train}
    dev = {"FY": fy_dev, "COSMIC": cosmic_dev}
    subsets = {source: _apply_profile_subset(loader, config, source)
               for source, loader in train.items()}
    for source, loader in dev.items():
        loader._audit_active_profile_ids = _sampler_profile_ids(loader)
    return train, dev, subsets


def _support_profiles(loader, chunk_size=200_000):
    """Recompute source×cell support from complete selected rows."""
    dataset = loader.dataset
    active = set(getattr(loader, "_audit_active_profile_ids", _sampler_profile_ids(loader)))
    sets = [set() for _ in FCB_CELL_LABELS]
    selected = np.asarray(dataset.selected_indices, dtype=np.int64)
    for start in range(0, len(selected), chunk_size):
        rows = np.asarray(dataset.data[selected[start:start + chunk_size], :4], dtype=np.float32)
        pids = np.asarray(dataset.profile_ids[start:start + len(rows)], dtype=np.int64)
        cells = _cell_ids_numpy(rows)
        finite = np.isfinite(rows).all(axis=1) & (cells >= 0)
        if active:
            finite &= np.isin(pids, np.fromiter(active, dtype=np.int64))
        for cell in range(len(FCB_CELL_LABELS)):
            values = pids[finite & (cells == cell)]
            if values.size:
                sets[cell].update(map(int, np.unique(values)))
    return sets, active


def _contract_audit(contract: dict, support: dict, active: dict) -> dict:
    if tuple(contract.get("cells", ())) != FCB_CELL_LABELS:
        raise ValueError("FCB contract cell order differs from code")
    result = {"passed": True, "sources": {}, "mismatches": []}
    for source in ("FY", "COSMIC"):
        total = int(contract["train"][f"{source}_total_profiles"])
        counts = {cell: int(contract["train"][source][cell]) for cell in FCB_CELL_LABELS}
        actual = {cell: len(support[source][i]) for i, cell in enumerate(FCB_CELL_LABELS)}
        active_count = len(active[source])
        weight_errors = {
            cell: abs(counts[cell] * float(contract["train"][f"{source}_weights"][cell]) - total / 12.0)
            for cell in FCB_CELL_LABELS
        }
        mismatches = [cell for cell in FCB_CELL_LABELS if counts[cell] != actual[cell]]
        if active_count != total or mismatches or max(weight_errors.values()) > 1e-8:
            result["passed"] = False
            result["mismatches"].append({
                "source": source,
                "contract_total": total,
                "active_total": active_count,
                "cell_count_mismatches": mismatches,
                "max_weight_identity_error": max(weight_errors.values()),
            })
        result["sources"][source] = {
            "contract_total_profiles": total,
            "active_total_profiles": active_count,
            "contract_cell_counts": counts,
            "recomputed_cell_counts": actual,
            "max_weight_identity_error": max(weight_errors.values()),
        }
    return result


def _empty_replay_state():
    return {
        "batches": 0,
        "cycles": 0,
        "duplicate_profile_occurrences": 0,
        "seen_profiles": set(),
        "raw": np.zeros(len(FCB_CELL_LABELS), dtype=np.int64),
        "exposure": np.zeros(len(FCB_CELL_LABELS), dtype=np.float64),
        "batches_present": np.zeros(len(FCB_CELL_LABELS), dtype=np.int64),
        "sampled_by_cell": [set() for _ in FCB_CELL_LABELS],
    }


def _accumulate_batch(state: dict, batch, weights: np.ndarray):
    data, _, profile_ids = batch
    values = data.detach().cpu().numpy()
    ids = profile_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
    if values.ndim != 2 or values.shape[1] < 5 or len(values) != len(ids):
        raise ValueError("unexpected loader batch schema")
    cells = _cell_ids_numpy(values[:, :4])
    finite = np.isfinite(values[:, :5]).all(axis=1) & (cells >= 0)
    valid_ids = np.unique(ids[finite])
    if not valid_ids.size:
        raise ValueError("batch has no finite in-domain profile")
    state["duplicate_profile_occurrences"] += int(
        len(set(map(int, valid_ids)).intersection(state["seen_profiles"]))
    )
    state["seen_profiles"].update(map(int, valid_ids))
    state["batches"] += 1
    for cell in range(len(FCB_CELL_LABELS)):
        selected = finite & (cells == cell)
        unique_ids = np.unique(ids[selected])
        count = int(unique_ids.size)
        state["raw"][cell] += count
        state["exposure"][cell] += 0.5 * float(weights[cell]) * count / len(valid_ids)
        state["batches_present"][cell] += int(count > 0)
        state["sampled_by_cell"][cell].update(map(int, unique_ids))


def _replay_seed(loaders: dict, contract: dict, seed: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    states = {source: _empty_replay_state() for source in loaders}
    weights = {
        source: np.asarray([contract["train"][f"{source}_weights"][cell]
                            for cell in FCB_CELL_LABELS], dtype=np.float64)
        for source in loaders
    }
    iterators = {source: iter(loaders[source]) for source in loaders}
    steps = max(len(loader) for loader in loaders.values())
    for _ in range(steps):
        for source in ("FY", "COSMIC"):
            try:
                batch = next(iterators[source])
            except StopIteration:
                states[source]["cycles"] += 1
                iterators[source] = iter(loaders[source])
                batch = next(iterators[source])
            _accumulate_batch(states[source], batch, weights[source])
    return {"seed": seed, "steps": steps, "sources": states}


def _aggregate_exposure(replays, support, active, contract):
    rows = []
    total_exposure = sum(float(state["exposure"].sum())
                         for replay in replays for state in replay["sources"].values())
    source_totals = {
        source: sum(float(replay["sources"][source]["exposure"].sum()) for replay in replays)
        for source in ("FY", "COSMIC")
    }
    for source in ("FY", "COSMIC"):
        for index, cell in enumerate(FCB_CELL_LABELS):
            states = [replay["sources"][source] for replay in replays]
            support_count = len(support[source][index])
            raw = int(sum(state["raw"][index] for state in states))
            sampled = set().union(*(state["sampled_by_cell"][index] for state in states))
            weighted = float(sum(state["exposure"][index] for state in states))
            source_share = weighted / source_totals[source] if source_totals[source] else 0.0
            global_share = weighted / total_exposure if total_exposure else 0.0
            batches = [state["batches"] for state in states]
            active_fraction = float(np.mean([
                state["batches_present"][index] / max(state["batches"], 1)
                for state in states
            ]))
            rows.append({
                "source": source,
                "cell": cell,
                "support_profiles": support_count,
                "total_active_profiles": len(active[source]),
                "sampled_support_profiles": len(sampled),
                "sampling_coverage_rate": len(sampled) / support_count if support_count else 0.0,
                "raw_profile_occurrences": raw,
                "exposure_rate": raw / (len(replays) * support_count) if support_count else 0.0,
                "weighted_exposure": weighted,
                "normalized_weighted_share": global_share,
                "target_global_share": TARGET_GLOBAL_SHARE,
                "normalized_source_share": source_share,
                "target_source_share": TARGET_SOURCE_SHARE,
                "relative_source_share_error": source_share / TARGET_SOURCE_SHARE - 1.0
                if TARGET_SOURCE_SHARE else 0.0,
                "cell_active_batch_fraction": active_fraction,
                "mean_batches": float(np.mean(batches)),
                "max_cycles": int(max(state["cycles"] for state in states)),
                "max_duplicate_profile_occurrences": int(max(
                    state["duplicate_profile_occurrences"] for state in states)),
                "contract_support_profiles": int(contract["train"][source][cell]),
            })
    return rows


def _profile_cell_losses(pred, target, coords, profile_ids):
    values = pred.reshape(-1)
    truth = target.reshape(-1)
    cells = _cell_ids_numpy(coords.detach().cpu().numpy())
    ids = profile_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
    out = {}
    for cell in range(len(FCB_CELL_LABELS)):
        selected = np.isfinite(cells) & (cells == cell)
        if not selected.any():
            continue
        selected_t = torch.as_tensor(selected, device=values.device, dtype=torch.bool)
        cell_ids = ids[selected]
        unique = np.unique(cell_ids)
        pieces = []
        for pid in unique:
            mask = selected & (ids == pid)
            mask_t = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
            pieces.append(F.huber_loss(values[mask_t], truth[mask_t], reduction="mean", delta=0.2))
        out[cell] = torch.stack(pieces).mean()
    return out


def _flatten_grads(grads, params):
    pieces = []
    for grad, param in zip(grads, params):
        pieces.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return torch.cat(pieces).to(dtype=torch.float64)


def _cosine(left: torch.Tensor, right: torch.Tensor):
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom) <= 1e-12:
        return None
    return float(torch.dot(left, right) / denom)


def _forward_background(model, sw_manager, iri_peak_manager, data):
    coords = data[:, :4]
    background = model.encode_background(
        coords,
        sw_manager.get_drivers_sequence(coords[:, 3]),
        iri_peak=iri_peak_manager.get_iri_peak(coords),
    )
    return background["ne_bkg"].reshape(-1), background["ne_iri"].reshape(-1)


def _collect_gradient_batches(loader, count=GRADIENT_BATCHES, seed=42):
    np.random.seed(seed)
    result = []
    iterator = iter(loader)
    for _ in range(min(count, len(loader))):
        try:
            result.append(next(iterator))
        except StopIteration:
            break
    return result


def _gradient_replay(checkpoints: dict, loaders: dict, contract: dict) -> dict:
    batches = {source: _collect_gradient_batches(loaders[source]) for source in loaders}
    report = {"batch_count": {source: len(items) for source, items in batches.items()},
              "models": {}, "parameter_unchanged": True}
    for name, checkpoint in checkpoints.items():
        model, sw, iri, _ = _load_background_model(checkpoint, "cpu")
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in params]
        cell_stats = {source: {cell: {"norm": [], "cosine": [], "batches": 0}
                               for cell in FCB_CELL_LABELS} for source in loaders}
        conflicts = {pair: [] for pair in ("low_mid", "low_high", "mid_high")}
        weighted_identity_errors = []
        for source in ("FY", "COSMIC"):
            weights = torch.as_tensor(
                [contract["train"][f"{source}_weights"][cell] for cell in FCB_CELL_LABELS],
                dtype=torch.float64,
            )
            for batch in batches[source]:
                data, _, profile_ids = batch
                finite = torch.isfinite(data[:, :5]).all(dim=1)
                if not bool(finite.any()):
                    continue
                data = data[finite]
                profile_ids = profile_ids[finite]
                prediction, _ = _forward_background(model, sw, iri, data)
                losses = _profile_cell_losses(prediction, data[:, 4:5], data[:, :4], profile_ids)
                if not losses:
                    continue
                total = sum(losses.values())
                total_grad = _flatten_grads(torch.autograd.grad(
                    total, params, retain_graph=True, allow_unused=True), params)
                vectors = {}
                for cell_index, loss in losses.items():
                    vector = _flatten_grads(torch.autograd.grad(
                        loss, params, retain_graph=True, allow_unused=True), params)
                    vectors[cell_index] = vector
                    stat = cell_stats[source][FCB_CELL_LABELS[cell_index]]
                    stat["norm"].append(float(torch.linalg.vector_norm(vector)))
                    cosine = _cosine(vector, total_grad)
                    if cosine is not None:
                        stat["cosine"].append(cosine)
                    stat["batches"] += 1
                weighted_loss = sum(weights[index].to(loss.device) * loss
                                    for index, loss in losses.items())
                weighted_direct = _flatten_grads(torch.autograd.grad(
                    weighted_loss, params, retain_graph=True, allow_unused=True), params)
                weighted_sum = sum(weights[index] * vector for index, vector in vectors.items())
                denom = max(float(torch.linalg.vector_norm(weighted_direct)), 1e-12)
                weighted_identity_errors.append(float(torch.linalg.vector_norm(
                    weighted_direct - weighted_sum) / denom))
                groups = {}
                for group, indices in MAG_GROUPS.items():
                    vectors_in_group = [vectors[i] for i in indices if i in vectors]
                    if vectors_in_group:
                        groups[group] = sum(vectors_in_group)
                for left, right, key in (("low", "mid", "low_mid"),
                                         ("low", "high", "low_high"),
                                         ("mid", "high", "mid_high")):
                    if left in groups and right in groups:
                        value = _cosine(groups[left], groups[right])
                        if value is not None:
                            conflicts[key].append(value)
        unchanged = all(torch.equal(old, new.detach()) for old, new in zip(before, params))
        report["parameter_unchanged"] &= unchanged
        report["models"][name] = {
            "parameter_unchanged": unchanged,
            "cell_gradients": {
                source: {
                    cell: {
                        "batches": stat["batches"],
                        "mean_norm": float(np.mean(stat["norm"])) if stat["norm"] else 0.0,
                        "mean_cosine_to_total": float(np.mean(stat["cosine"]))
                        if stat["cosine"] else None,
                        "finite": bool(all(np.isfinite(stat["norm"]))),
                    }
                    for cell, stat in values.items()
                }
                for source, values in cell_stats.items()
            },
            "conflicts": {
                key: {
                    "effective_batches": len(values),
                    "median": float(np.median(values)) if values else None,
                    "negative_fraction": float(np.mean(np.asarray(values) < 0.0)) if values else None,
                }
                for key, values in conflicts.items()
            },
            "weighted_gradient_identity_max_relative_error": max(weighted_identity_errors)
            if weighted_identity_errors else None,
        }
    return report


def _metric_accumulate(acc, cell, pid, m00, iri, target):
    entry = acc[cell].setdefault(int(pid), [0.0, 0.0, 0])
    residual = float(m00 - target)
    iri_residual = float(iri - target)
    entry[0] += residual * residual
    entry[1] += residual
    entry.append(iri_residual * iri_residual)
    entry.append(iri_residual)
    entry[2] += 1


def _full_profile_metrics(model_path: Path, loaders: dict, label: str) -> dict:
    model, sw, iri_manager, _ = _load_background_model(model_path, "cpu")
    model.eval()
    result = {label: {}}
    with torch.inference_mode():
        for source in ("FY", "COSMIC"):
            result[label][source] = {}
            for split, loader in (("train", loaders[0][source]), ("development", loaders[1][source])):
                acc = {index: {} for index in range(len(FCB_CELL_LABELS))}
                for batch in loader:
                    data, _, profile_ids = batch
                    finite = torch.isfinite(data[:, :5]).all(dim=1)
                    if not bool(finite.any()):
                        continue
                    data = data[finite]
                    profile_ids = profile_ids[finite]
                    m00, raw_iri = _forward_background(model, sw, iri_manager, data)
                    values = data.detach().cpu().numpy()
                    pids = profile_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
                    cells = _cell_ids_numpy(values[:, :4])
                    for cell in range(len(FCB_CELL_LABELS)):
                        selected = cells == cell
                        for pid in np.unique(pids[selected]):
                            mask = selected & (pids == pid)
                            m00_values = m00.detach().cpu().numpy()[mask]
                            iri_values = raw_iri.detach().cpu().numpy()[mask]
                            targets = values[mask, 4]
                            if len(targets):
                                _metric_accumulate(acc, cell, pid,
                                                   float(np.mean(m00_values)),
                                                   float(np.mean(iri_values)),
                                                   float(np.mean(targets)))
                source_split = {}
                for cell, profiles in acc.items():
                    if not profiles:
                        source_split[FCB_CELL_LABELS[cell]] = {
                            "profiles": 0, "m00_profile_rmse": None, "raw_iri_profile_rmse": None,
                            "m00_profile_bias": None, "raw_iri_profile_bias": None,
                        }
                        continue
                    m00_sq = []; m00_bias = []; iri_sq = []; iri_bias = []
                    for values in profiles.values():
                        count = values[2]
                        m00_sq.append(values[0] / count)
                        m00_bias.append(values[1] / count)
                        iri_sq.append(values[3] / count)
                        iri_bias.append(values[4] / count)
                    source_split[FCB_CELL_LABELS[cell]] = {
                        "profiles": len(profiles),
                        "m00_profile_rmse": float(np.sqrt(np.mean(m00_sq))),
                        "raw_iri_profile_rmse": float(np.sqrt(np.mean(iri_sq))),
                        "m00_profile_bias": float(np.mean(m00_bias)),
                        "raw_iri_profile_bias": float(np.mean(iri_bias)),
                    }
                result[label][source][split] = source_split
    return result


def _add_train_dev_deltas(metrics: dict):
    for label, sources in metrics.items():
        for source, splits in sources.items():
            train = splits["train"]
            dev = splits["development"]
            for cell in FCB_CELL_LABELS:
                if train[cell]["m00_profile_rmse"] is None or dev[cell]["m00_profile_rmse"] is None:
                    dev[cell]["train_to_development_rmse_delta"] = None
                else:
                    dev[cell]["train_to_development_rmse_delta"] = (
                        dev[cell]["m00_profile_rmse"] - train[cell]["m00_profile_rmse"])


def _saved_development_audit(history, baseline):
    by_epoch = {int(row["epoch"]): row for row in history
                if row.get("stage") == "background"}
    rows = []
    for epoch in (3, 5):
        if epoch not in by_epoch:
            continue
        audit = fcb_selection_audit(by_epoch[epoch], baseline)
        rows.append({
            "epoch": epoch,
            "eligible": bool(audit["eligible"]),
            "reasons": list(audit["reasons"]),
            "candidate": audit["candidate"],
            "cell_rmse_worsening": audit["cell_rmse_worsening"],
        })
    return rows


def _classify(exposure_rows, metrics, gradient):
    sparse = [row for row in exposure_rows
              if row["exposure_rate"] < 0.8 or
              abs(row["relative_source_share_error"]) > 0.2]
    a1s = metrics.get("A1S", {})
    fcb = metrics.get("FCB_epoch_05", {})
    sparse_not_improved = True
    for row in sparse:
        source, cell = row["source"], row["cell"]
        try:
            a_train = a1s[source]["train"][cell]["m00_profile_rmse"]
            f_train = fcb[source]["train"][cell]["m00_profile_rmse"]
            a_dev = a1s[source]["development"][cell]["m00_profile_rmse"]
            f_dev = fcb[source]["development"][cell]["m00_profile_rmse"]
        except KeyError:
            sparse_not_improved = False
            break
        if any(v is None for v in (a_train, f_train, a_dev, f_dev)):
            sparse_not_improved = False
            break
        if f_train < a_train and f_dev < a_dev:
            continue
        # One non-improved sparse cell is sufficient to avoid claiming a pure
        # exposure explanation; the next experiment must be decided with data.
        sparse_not_improved = False
        break
    if sparse and sparse_not_improved:
        return "exposure_insufficient", "Run one capped sqrt inverse-frequency exposure experiment; do not increase FCB weights."
    exposure_ok = not sparse
    if exposure_ok:
        for source in ("FY", "COSMIC"):
            for cell in FCB_CELL_LABELS:
                try:
                    a_train = a1s[source]["train"][cell]["m00_profile_rmse"]
                    f_train = fcb[source]["train"][cell]["m00_profile_rmse"]
                    a_dev = a1s[source]["development"][cell]["m00_profile_rmse"]
                    f_dev = fcb[source]["development"][cell]["m00_profile_rmse"]
                except KeyError:
                    continue
                if (f_train <= a_train - 0.005 and f_dev >= a_dev):
                    return "data_support_or_domain_shift", "Expand month/geographic/condition coverage; do not increase loss weights."
    for model in gradient.get("models", {}).values():
        pair = model.get("conflicts", {}).get("low_high", {})
        if pair.get("effective_batches", 0) >= 10 and (
                (pair.get("negative_fraction") or 0.0) >= 0.30 or
                (pair.get("median") is not None and pair["median"] < -0.1)):
            return "gradient_conflict", "Inject signed magnetic inclination and LT into the Background decoder; defer PCGrad."
    return "fcb_hypothesis_failed", "Keep A1-S and stop the sparse-weight direction."


def _write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else ["source", "cell"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, audit: dict):
    decision = audit["decision"]
    lines = [
        "M00-FCB exposure audit",
        f"status: {audit['status']}",
        "scope: train-only deterministic sampler replay, local gradient replay, complete-profile train/development forward",
        "locked-test: no loader or sample accessed; training: not run; ISR: not run; hashes: not computed",
        "historical exact replay: impossible because per-batch IDs and epoch-start RNG were not saved",
        f"contract identity: {audit['gates']['contract']}",
        f"train/development profile isolation: {audit['gates']['train_development_profile_isolation']}",
        f"exposure balance: {audit['gates']['exposure_balance']}",
        f"gradient replay: {audit['gates']['gradient_replay']}",
        f"full profile replay: {audit['gates']['full_profile_replay']}",
        f"classification: {decision['classification']}",
        f"decision: {decision['action']}",
        f"next step: {decision['next_step']}",
        "training_summary contradiction: best_epochs.background reports epoch 1 although fcb_development_stop has no eligible epoch; no FCB checkpoint is marked best.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "run_manifest.json")
    config = manifest["config"]
    summary = _read_json(run_dir / "training_summary.json")
    contract_file = _contract_path(run_dir, config)
    contract = _read_json(contract_file)
    history = _read_json(run_dir / "fsia_training_history.json")
    stop = _read_json(run_dir / "fcb_development_stop.json")
    baseline_replay = _read_json(run_dir / "a1s_fcb_development_replay.json")

    baseline_checkpoint = Path(baseline_replay["checkpoint"])
    checkpoint_paths = {
        "A1S": baseline_checkpoint,
        "FCB_epoch_03": run_dir / "epoch_03_background_model.pth",
        "FCB_epoch_05": run_dir / "epoch_05_background_model.pth",
    }
    checkpoint_info = {name: _checkpoint_summary(path) for name, path in checkpoint_paths.items()}

    train_loaders, dev_loaders, subsets = _make_loaders(config, summary, points_per_profile=8)
    support = {}
    active = {}
    dev_profile_sets = {}
    for source in ("FY", "COSMIC"):
        support[source], active[source] = _support_profiles(train_loaders[source])
        dev_profile_sets[source] = set(getattr(dev_loaders[source], "_audit_active_profile_ids", set()))
    profile_isolation = {
        source: not bool(active[source].intersection(dev_profile_sets[source]))
        for source in ("FY", "COSMIC")
    }
    contract_result = _contract_audit(contract, support, active)
    replays = [_replay_seed(train_loaders, contract, seed) for seed in SEEDS]
    exposure_rows = _aggregate_exposure(replays, support, active, contract)
    sparse_rows = [row for row in exposure_rows
                   if row["exposure_rate"] < 0.8 or
                   abs(row["relative_source_share_error"]) > 0.2]
    exposure_passed = contract_result["passed"] and all(profile_isolation.values()) and not sparse_rows

    # Full-profile replay is deliberately separate from the stable 8-point
    # Development selector metrics saved by the training run.
    full_train, full_dev, _ = _make_loaders(
        config, summary, points_per_profile=None, full_validation_profiles=True)
    full_metrics = {}
    for label, checkpoint in checkpoint_paths.items():
        full_metrics.update(_full_profile_metrics(checkpoint, (full_train, full_dev), label))
    _add_train_dev_deltas(full_metrics)

    gradient = _gradient_replay(
        {"A1S": baseline_checkpoint, "FCB_epoch_05": checkpoint_paths["FCB_epoch_05"]},
        train_loaders,
        contract,
    )
    gradient_passed = bool(gradient.get("parameter_unchanged"))
    full_profile_passed = bool(full_metrics) and not _finite_tree(full_metrics)
    classification, next_step = _classify(exposure_rows, full_metrics, gradient)

    eligible_epochs = [int(value) for value in stop.get("eligible_epochs", [])]
    reported_best = summary.get("best_epochs", {}).get("background")
    best_key = summary.get("best_selection_keys", {}).get("background")
    artifact_contradiction = bool(not eligible_epochs and reported_best is not None)
    artifact_passed = all(info["all_finite"] for info in checkpoint_info.values()) and not artifact_contradiction
    all_finite = not _finite_tree({
        "contract": contract_result,
        "exposure": exposure_rows,
        "metrics": full_metrics,
        "gradient": gradient,
    })
    audit = {
        "schema": "m00_fcb_exposure_audit_v2",
        "status": "complete" if all_finite else "failed",
        "provenance": {
            "run_dir": str(run_dir),
            "sampler": "ProfileTimeBinSampler",
            "seeds": list(SEEDS),
            "gradient_batches_per_source": GRADIENT_BATCHES,
            "points_per_profile_exposure": int(config.get("profile_points_per_epoch", 8)),
            "points_per_profile_full_replay": None,
            "train_profile_fraction": float(config.get("train_profile_fraction", 1.0)),
            "profile_subset": {
                source: {key: value for key, value in item.items() if key != "profile_ids"}
                for source, item in subsets.items()
            },
            "contract_path": str(contract_file),
            "locked_test_accessed": False,
            "locked_test_profile_overlap_checked": False,
            "training_run": False,
            "isr_run": False,
            "hashes_computed": False,
            "historical_exact": False,
            "historical_exact_reason": "per-batch IDs and epoch-start RNG were not saved",
            "checkpoints": checkpoint_info,
        },
        "contract": contract_result,
        "partition_audit": {
            "date_partition_disjoint": True,
            "train_development_profile_isolation": profile_isolation,
            "locked_test": "metadata only; no profile loader constructed",
        },
        "exposure": {
            "formula": "0.5*w_source_cell*n_batch_source_cell/N_batch_source",
            "target_source_share": TARGET_SOURCE_SHARE,
            "target_global_share": TARGET_GLOBAL_SHARE,
            "rows": exposure_rows,
            "row_count": len(exposure_rows),
            "sparse_row_count": len(sparse_rows),
            "sparse_rows": [{"source": row["source"], "cell": row["cell"],
                             "exposure_rate": row["exposure_rate"],
                             "relative_source_share_error": row["relative_source_share_error"]}
                            for row in sparse_rows],
            "cycles_are_reported_not_rejected": True,
        },
        "gradient_replay": gradient,
        "full_profile_replay": {
            "metrics": full_metrics,
            "selector_metrics_kept_separate": True,
            "saved_selector_metrics": _saved_development_audit(history, baseline_replay),
        },
        "artifact_consistency": {
            "eligible_epochs": eligible_epochs,
            "reported_best_epoch": reported_best,
            "reported_best_selection_key": best_key if not _finite_tree(best_key) else None,
            "reported_best_selection_key_non_finite": bool(_finite_tree(best_key)),
            "contradiction": artifact_contradiction,
            "passed": artifact_passed,
        },
        "gates": {
            "contract": contract_result["passed"],
            "train_development_profile_isolation": all(profile_isolation.values()),
            "exposure_balance": exposure_passed,
            "gradient_replay": gradient_passed,
            "full_profile_replay": full_profile_passed,
            "artifact_consistency": artifact_passed,
            "overall": False,
        },
        "decision": {
            "action": "keep_a1s_prohibit_fcb_analysis_or_isr",
            "classification": classification,
            "next_step": next_step,
            "reasons": [
                "FCB Development stop audit has no eligible epoch",
                "FCB best checkpoint is not accepted despite training_summary epoch-1 field",
                "audit is diagnostic only; ISR and locked-test are excluded",
            ],
        },
        "finite_values_passed": all_finite,
    }
    if not all_finite:
        raise ValueError(f"audit contains non-finite values: {_finite_tree(audit)[:5]}")
    (run_dir / "fcb_exposure_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(run_dir / "fcb_exposure_by_cell.csv", exposure_rows)
    _write_text(run_dir / "fcb_exposure_audit.txt", audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    audit = run(args.run_dir)
    print(json.dumps({"status": audit["status"], "gates": audit["gates"],
                      "decision": audit["decision"]}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
