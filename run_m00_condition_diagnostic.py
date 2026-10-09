"""Compare completed M00 controls without retraining or reading locked test data."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from run_a1s_natural_epoch import HISTORICAL_RUN, ROOT, run_config

import sys
sys.path.insert(0, str(ROOT / "inr_modules"))
from inr_modules.mdia.train_fsia import select_m00_background_candidate


MAGNETIC_LABELS = (
    "abs_sinI_lt_0.25", "abs_sinI_0.25_0.5", "abs_sinI_ge_0.5")
NIGHT_MAGNETIC_LABELS = tuple(
    f"{mag}_{period}" for mag in MAGNETIC_LABELS
    for period in ("night", "day"))
DIAGNOSTIC_SEMANTICS = "m00_sparse_coverage_bias_diagnostic_v1"
NIGHT_V2_SEMANTICS = "m00_night_generalization_v2"
NIGHT_V2_CELLS = tuple(
    f"{alt}_{period}_{mag}" for alt in ("200-300", "300-500")
    for period in ("night", "day") for mag in MAGNETIC_LABELS)


def _rows(run_dir: Path) -> list[dict]:
    history = json.loads((Path(run_dir) / "fsia_training_history.json").read_text(encoding="utf-8"))
    return [row for row in history if row.get("stage") == "background"]


def _pick(run_dir: Path, baseline: dict) -> dict:
    result = select_m00_background_candidate(_rows(run_dir), baseline)
    if result.get("selected_row") is None:
        raise ValueError(f"no valid selected epoch: {run_dir}")
    return result["selected_row"]


def _metric(item: dict) -> dict:
    m00 = item.get("M00", {})
    return {
        "profile_count": int(m00.get("profile_count", m00.get("n", 0))),
        "rmse": float(m00.get("rmse", "nan")),
        "bias": float(m00.get("bias", "nan")),
    }


def compare(candidate_run: Path, control_run: Path, output: Path | None = None) -> dict:
    baseline_rows = _rows(HISTORICAL_RUN)
    baseline = next(row for row in baseline_rows if int(row["epoch"]) == 3)
    candidate = _pick(candidate_run, baseline)
    control = _pick(control_run, baseline)
    groups = {}
    for source in ("fy", "cosmic"):
        for group_type in ("cells", "magnetic_latitude_cells"):
            candidate_groups = candidate[f"{source}_background_stratified"][group_type]
            control_groups = control[f"{source}_background_stratified"][group_type]
            for group, candidate_data in candidate_groups.items():
                if group not in control_groups:
                    continue
                c = _metric(candidate_data)
                a = _metric(control_groups[group])
                count = min(c["profile_count"], a["profile_count"])
                groups[f"{source.upper()}:{group}"] = {
                    "source": source.upper(), "group": group,
                    "group_type": group_type, "candidate": c, "control": a,
                    "delta_rmse": c["rmse"] - a["rmse"],
                    "delta_bias": c["bias"] - a["bias"],
                    "delta_abs_bias": abs(c["bias"]) - abs(a["bias"]),
                    "sufficient_profiles": count >= 30,
                }
    result = {
        "semantics": "m00_condition_diagnostic_v1",
        "candidate_run": str(Path(candidate_run).resolve()),
        "control_run": str(Path(control_run).resolve()),
        "candidate_epoch": int(candidate["epoch"]),
        "control_epoch": int(control["epoch"]),
        "locked_test_read": False,
        "groups": groups,
        "source_summary": {
            source: {
                "usable_height_day_groups": sum(
                    item["source"] == source and item["group_type"] == "cells"
                    and item["sufficient_profiles"] for item in groups.values()),
                "rmse_improved_groups": sum(
                    item["source"] == source and item["group_type"] == "cells"
                    and item["sufficient_profiles"] and item["delta_rmse"] <= 0
                    for item in groups.values()),
                "abs_bias_improved_groups": sum(
                    item["source"] == source and item["group_type"] == "cells"
                    and item["sufficient_profiles"] and item["delta_abs_bias"] <= 0
                    for item in groups.values()),
            }
            for source in ("FY", "COSMIC")
        },
        "gradient_ratio": {
            "candidate": candidate.get("gradient_ratio_background_stratum_bias"),
            "control": control.get("gradient_ratio_background_stratum_bias"),
            "delta": (
                float(candidate.get("gradient_ratio_background_stratum_bias", float("nan")))
                - float(control.get("gradient_ratio_background_stratum_bias", float("nan")))
            ),
        },
        "magnetic_latitude_note": "|sin I| bins are magnetic-region proxies, not exact magnetic latitude.",
    }
    if output is None:
        output = Path(candidate_run) / "m00_condition_diagnostic.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(result: dict) -> str:
    lines = [
        "# M00 condition diagnostic", "",
        f"Candidate epoch: `{result['candidate_epoch']}`; control epoch: `{result['control_epoch']}`.",
        "", "| source/group | n(candidate/control) | ΔRMSE | ΔBias | Δ|Bias| | usable |",
        "|:---|---:|---:|---:|---:|:---|",
    ]
    for key, item in sorted(result["groups"].items()):
        lines.append(
            f"| {key} | {item['candidate']['profile_count']}/{item['control']['profile_count']} "
            f"| {item['delta_rmse']:.6f} | {item['delta_bias']:.6f} "
            f"| {item['delta_abs_bias']:.6f} | {'yes' if item['sufficient_profiles'] else 'no'} |")
    lines.extend(["", "The magnetic groups use `|sin I|` as a proxy, not exact magnetic latitude.", ""])
    for source, summary in result.get("source_summary", {}).items():
        lines.append(
            f"{source}: {summary['rmse_improved_groups']}/"
            f"{summary['usable_height_day_groups']} height-day groups improved RMSE; "
            f"{summary['abs_bias_improved_groups']}/"
            f"{summary['usable_height_day_groups']} improved absolute Bias.")
    return "\n".join(lines)


def _manifest_config(run_dir: Path) -> dict:
    payload = json.loads((Path(run_dir) / "run_manifest.json").read_text(
        encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"run manifest lacks config: {run_dir}")
    return config


def _selected_checkpoint(run_dir: Path) -> Path:
    summary = json.loads((Path(run_dir) / "training_summary.json").read_text(
        encoding="utf-8"))
    checkpoint = summary.get("background_candidate_checkpoint")
    if checkpoint and Path(checkpoint).is_file():
        return Path(checkpoint).resolve()
    selected = _pick(Path(run_dir), next(
        row for row in _rows(HISTORICAL_RUN) if int(row["epoch"]) == 3))
    return (Path(run_dir) /
            f"epoch_{int(selected['epoch']):02d}_background_model.pth").resolve()


def _ensure_train_replay(run_dir: Path) -> dict:
    """Evaluate one selected checkpoint on train and development splits once."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "m00_sparse_bias_diagnostic"
    replay_path = output_dir / "background_development_replay.json"
    checkpoint = _selected_checkpoint(run_dir)
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if (Path(replay.get("checkpoint", "")).resolve() == checkpoint
                and isinstance(
                    replay.get("background_bias_magnetic_support_counts"),
                    dict)
                and isinstance(replay.get("train_metrics"), dict)):
            return replay
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite incomplete diagnostic output: {output_dir}")
    config = copy.deepcopy(_manifest_config(run_dir))
    config.update({
        "save_dir": str(output_dir),
        "resume_ckpt": str(checkpoint),
        "eval_only": True,
        "background_only": True,
        "m00_diagnostic_train_split": True,
        "locked_test_days": int(config.get("locked_test_days", 5)),
        "skip_hash_verification": True,
    })
    run_config(config, output_dir,
               experiment="M00 sparse-coverage train/development diagnostic")
    if not replay_path.is_file():
        raise RuntimeError(f"diagnostic replay was not written: {replay_path}")
    return json.loads(replay_path.read_text(encoding="utf-8"))


def _coarse_metrics(replay: dict, source: str, split: str) -> dict:
    metrics = replay.get("metrics" if split == "development"
                         else "train_metrics", {})
    source_metrics = metrics.get(f"{source.lower()}_background_stratified", {})
    cells = source_metrics.get("magnetic_region_cells")
    if not isinstance(cells, dict):
        raise ValueError(
            f"{split} replay lacks direct coarse magnetic region metrics for {source}")
    result = {}
    for label in MAGNETIC_LABELS:
        cell = cells.get(label, {}).get("M00", {})
        result[label] = {
            "profile_count": int(cell.get("profile_count", 0)),
            "point_count": int(cell.get("point_count", 0)),
            "bias": float(cell["bias"]) if cell.get("bias") is not None else None,
            "rmse": float(cell["rmse"]) if cell.get("rmse") is not None else None,
        }
    return result


def _night_cells(replay: dict, source: str, split: str) -> dict:
    """Read exact height×local-time×magnetic cells from one replay."""
    metrics = replay.get("metrics" if split == "development"
                         else "train_metrics", {})
    source_metrics = metrics.get(f"{source.lower()}_background_stratified", {})
    cells = source_metrics.get("magnetic_latitude_cells")
    if not isinstance(cells, dict):
        raise ValueError(f"{split} replay lacks magnetic latitude cells for {source}")
    result = {}
    for altitude in ("200-300", "300-500"):
        for period in ("night", "day"):
            for magnetic in MAGNETIC_LABELS:
                item = cells.get(f"{altitude}_{period}_{magnetic}", {}).get("M00", {})
                result[f"{altitude}_{period}_{magnetic}"] = {
                    "profile_count": int(item.get("profile_count", 0)),
                    "bias": (float(item["bias"]) if item.get("bias") is not None else None),
                    "rmse": (float(item["rmse"]) if item.get("rmse") is not None else None),
                }
    return result


def diagnose_night_generalization(
        candidate_run: Path, control_run: Path, output: Path | None = None) -> dict:
    """Gate the one authorized candidate on train/dev night support."""
    replay = _ensure_train_replay(candidate_run)
    regions = {}
    for source in ("FY", "COSMIC"):
        train = _night_cells(replay, source, "train")
        development = _night_cells(replay, source, "development")
        for magnetic in MAGNETIC_LABELS:
            for period in ("night", "day"):
                keys = [f"{alt}_{period}_{magnetic}" for alt in ("200-300", "300-500")]
                def aggregate(table):
                    valid = [table[key] for key in keys
                             if table[key]["profile_count"] >= 1
                             and table[key]["bias"] is not None]
                    if not valid:
                        return {"profile_count": 0, "bias": None, "rmse": None}
                    weights = [item["profile_count"] for item in valid]
                    total = float(sum(weights))
                    return {
                        "profile_count": int(min(item["profile_count"] for item in valid)),
                        "bias": sum(item["bias"] * weight for item, weight in zip(valid, weights)) / total,
                        "rmse": sum(item["rmse"] * weight for item, weight in zip(valid, weights)) / total,
                    }
                train_item, dev_item = aggregate(train), aggregate(development)
                same_sign = (train_item["bias"] is not None
                             and dev_item["bias"] is not None
                             and train_item["bias"] * dev_item["bias"] > 0.0)
                qualified = bool(
                    train_item["profile_count"] >= 30
                    and dev_item["profile_count"] >= 30
                    and same_sign
                    and abs(train_item["bias"]) >= 0.02
                    and abs(dev_item["bias"]) >= 0.02)
                regions[f"{source}:{magnetic}_{period}"] = {
                    "source": source, "magnetic_region": magnetic,
                    "period": period, "train": train_item,
                    "development": dev_item, "same_sign": same_sign,
                    "qualified": qualified,
                }
    conflicts = {}
    for magnetic in MAGNETIC_LABELS:
        fy = regions[f"FY:{magnetic}_night"]
        cosmic = regions[f"COSMIC:{magnetic}_night"]
        conflicts[magnetic] = bool(
            fy["qualified"] and cosmic["qualified"]
            and fy["development"]["bias"] * cosmic["development"]["bias"] < 0.0
            and fy["train"]["bias"] * cosmic["train"]["bias"] < 0.0)
    qualified = [key for key, item in regions.items()
                 if item["qualified"] and item["period"] == "night"]
    conflict = [key for key, value in conflicts.items() if value]
    if conflict:
        status, reason = "diagnostic_not_supported", "opposite FY/COSMIC night Bias directions"
    elif not qualified:
        status, reason = "diagnostic_not_supported", "no source x magnetic night region meets fixed train/dev gate"
    else:
        status, reason = "diagnostic_supported", None
    result = {
        "semantics": "m00_night_generalization_diagnostic_v1",
        "status": status, "reason": reason,
        "candidate_run": str(Path(candidate_run).resolve()),
        "control_run": str(Path(control_run).resolve()),
        "locked_test_read": False, "regions": regions,
        "qualified_regions": qualified,
        "source_conflict_regions": conflict,
        "fixed_rules": {"min_profiles": 30, "min_abs_bias_dex": 0.02,
                        "altitude_range_km": [200.0, 500.0],
                        "magnetic_bins": list(MAGNETIC_LABELS),
                        "periods": ["night", "day"]},
    }
    if output is None:
        output = Path(candidate_run) / "m00_night_generalization_diagnostic.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# M00 night generalization diagnostic", "",
             f"Status: **{status}**", f"Reason: {reason or 'fixed gate passed'}", "",
             "| source/region | train profiles | train Bias | dev profiles | dev Bias | qualified |",
             "|:---|---:|---:|---:|---:|:---|"]
    for key, item in sorted(regions.items()):
        lines.append(
            f"| {key} | {item['train']['profile_count']} | {item['train']['bias']!s} "
            f"| {item['development']['profile_count']} | {item['development']['bias']!s} "
            f"| {'yes' if item['qualified'] else 'no'} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _control_development_cells(control_run: Path) -> dict:
    """Keep the existing control table without another full inference pass."""
    rows = _rows(control_run)
    matches = [item for item in rows if int(item["epoch"]) == 3]
    if len(matches) != 1:
        raise ValueError("historical control must contain exactly one epoch 3")
    row = matches[0]
    result = {}
    for source in ("fy", "cosmic"):
        raw = row.get(f"{source}_background_stratified", {})
        cells = raw.get("magnetic_latitude_cells", {})
        result[source.upper()] = {
            key: _metric(value)
            for key, value in cells.items()
        }
    return result


def diagnose_sparse_coverage(
        candidate_run: Path, control_run: Path, output: Path | None = None) -> dict:
    """Run the fixed train/dev gating diagnostic before a new training run."""
    candidate_run = Path(candidate_run)
    control_run = Path(control_run)
    replay = _ensure_train_replay(candidate_run)
    support = replay.get("background_bias_magnetic_support_counts")
    if not isinstance(support, dict):
        result = {
            "semantics": DIAGNOSTIC_SEMANTICS,
            "status": "diagnostic_not_supported",
            "reason": "missing train magnetic support counts",
            "locked_test_read": False,
        }
    else:
        regions = {}
        for source in ("FY", "COSMIC"):
            train = _coarse_metrics(replay, source, "train")
            development = _coarse_metrics(replay, source, "development")
            source_support = support.get(source, {})
            for label in MAGNETIC_LABELS:
                train_item = train[label]
                dev_item = development[label]
                train_count = int(source_support.get(label, 0))
                dev_count = int(dev_item["profile_count"])
                train_bias = train_item["bias"]
                dev_bias = dev_item["bias"]
                same_sign = (
                    train_bias is not None and dev_bias is not None
                    and train_bias * dev_bias > 0.0)
                qualified = bool(
                    train_count >= 30 and dev_count >= 30
                    and same_sign
                    and abs(float(train_bias or 0.0)) >= 0.02
                    and abs(float(dev_bias or 0.0)) >= 0.02)
                regions[f"{source}:{label}"] = {
                    "source": source,
                    "magnetic_region": label,
                    "train_support_profile_count": train_count,
                    "train": train_item,
                    "development": dev_item,
                    "same_sign": bool(same_sign),
                    "qualified": qualified,
                }
        conflicts = {}
        for label in MAGNETIC_LABELS:
            fy = regions[f"FY:{label}"]
            cosmic = regions[f"COSMIC:{label}"]
            conflicts[label] = bool(
                fy["qualified"] and cosmic["qualified"]
                and fy["development"]["bias"] * cosmic["development"]["bias"] < 0.0
                and fy["train"]["bias"] * cosmic["train"]["bias"] < 0.0)
        qualified_regions = [key for key, item in regions.items()
                             if item["qualified"]]
        conflict_regions = [key for key, value in conflicts.items() if value]
        if conflict_regions:
            status = "diagnostic_not_supported"
            reason = "opposite FY/COSMIC Bias directions in qualified regions"
        elif not qualified_regions:
            status = "diagnostic_not_supported"
            reason = "no source x magnetic region meets fixed train/dev gate"
        else:
            status = "diagnostic_supported"
            reason = None
        result = {
            "semantics": DIAGNOSTIC_SEMANTICS,
            "status": status,
            "reason": reason,
            "candidate_run": str(candidate_run.resolve()),
            "control_run": str(control_run.resolve()),
            "checkpoint": str(_selected_checkpoint(candidate_run)),
            "locked_test_read": False,
            "regions": regions,
            "qualified_regions": qualified_regions,
            "source_conflict_regions": conflict_regions,
            "control_development_magnetic_cells": _control_development_cells(
                control_run),
            "candidate_development": {
                source: _coarse_metrics(replay, source, "development")
                for source in ("FY", "COSMIC")
            },
            "fixed_rules": {
                "min_profiles": 30,
                "min_abs_bias_dex": 0.02,
                "magnetic_bins": list(MAGNETIC_LABELS),
                "altitude_range_km": [200.0, 500.0],
                "proxy_note": "|sin I| is a magnetic-region proxy, not exact magnetic latitude",
            },
        }
    if output is None:
        output = candidate_run / "m00_sparse_coverage_diagnostic.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    output.with_suffix(".md").write_text(_sparse_markdown(result),
                                          encoding="utf-8")
    return result


def _sparse_markdown(result: dict) -> str:
    lines = [
        "# M00 sparse-coverage Bias diagnostic", "",
        f"Status: **{result.get('status')}**",
        f"Reason: {result.get('reason') or 'fixed diagnostic gate passed'}", "",
        "| source/region | train support | train Bias | dev profiles | dev Bias | same sign | qualified |",
        "|:---|---:|---:|---:|---:|:---:|:---|",
    ]
    for key, item in sorted(result.get("regions", {}).items()):
        train = item["train"]
        dev = item["development"]
        lines.append(
            f"| {key} | {item['train_support_profile_count']} "
            f"| {train['bias']!s} | {dev['profile_count']} | {dev['bias']!s} "
            f"| {'yes' if item['same_sign'] else 'no'} "
            f"| {'yes' if item['qualified'] else 'no'} |")
    lines.extend([
        "", "The bins use the fixed dipole `|sin I|` proxy and 200–500 km.",
        "No locked-test profiles or ISR values are used.", "",
    ])
    return "\n".join(lines)


def _sha256(path: Path) -> str | None:
    """Hash one input file; directories deliberately remain provenance, not data."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _input_hashes(config: dict) -> dict:
    """Record current configured files without guessing directory contents."""
    found = {}

    def visit(value, key=""):
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and ("path" in key.lower()
                                     or "file" in key.lower()
                                     or "manifest" in key.lower()):
            candidate = Path(value)
            digest = _sha256(candidate)
            if digest is not None:
                found[str(candidate.resolve())] = digest

    visit(config)
    return found


def _identity(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    config = _manifest_config(run_dir)
    checkpoint = _selected_checkpoint(run_dir)
    return {
        "run": str(run_dir.resolve()), "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": _json_sha256(config),
        "current_input_sha256": _input_hashes(config),
        "date_split_sha256": _sha256(Path(config['date_split_manifest'])),
        "historical_input_sha256": "unavailable",
    }


def _v2_replay_path(output_dir: Path) -> Path:
    return Path(output_dir) / "background_development_replay.json"


def _replay_is_compatible(replay: dict, identity: dict) -> bool:
    replay_identity = replay.get("identity", {})
    return bool(isinstance(replay.get("train_metrics"), dict)
                and isinstance(replay.get("metrics"), dict)
                and all(identity.get(key) is not None
                        and replay_identity.get(key) == identity[key]
                        for key in ('checkpoint_sha256', 'config_sha256',
                                    'current_input_sha256', 'date_split_sha256')))


def _ensure_v2_replay(control_run: Path, output_dir: Path,
                      allow_replay: bool) -> tuple[dict | None, str | None]:
    """Read the one reusable replay; only the caller may authorize its creation."""
    identity = _identity(control_run)
    if Path(identity['checkpoint']).name != 'epoch_05_background_model.pth':
        return None, 'direct control must be the frozen selected epoch 5'
    replay_path = _v2_replay_path(output_dir)
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if _replay_is_compatible(replay, identity):
            return replay, None
        return None, "incompatible cached control replay identity"
    if not allow_replay:
        return None, "compatible control replay is required (replay disabled)"
    if output_dir.exists() and any(output_dir.iterdir()):
        return None, "refusing to overwrite nonempty diagnostic directory"
    config = copy.deepcopy(_manifest_config(control_run))
    config.update({"save_dir": str(output_dir), "resume_ckpt": identity["checkpoint"],
                   "eval_only": True, "background_only": True,
                   "m00_diagnostic_train_split": True,
                   "night_v2_frozen_input_sha256": identity['current_input_sha256'],
                   "skip_hash_verification": False})
    run_config(config, output_dir, experiment="M00 night-v2 control replay")
    if not replay_path.is_file():
        return None, "control replay did not write background_development_replay.json"
    if _identity(control_run) != identity:
        return None, 'control or current data identity changed during replay'
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["identity"] = identity
    replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return replay, None


def _cell_table(replay: dict, source: str, split: str) -> dict:
    metrics = replay["metrics" if split == "development" else "train_metrics"]
    stratified = metrics.get(f"{source}_background_stratified", {})
    cells = stratified.get("magnetic_latitude_cells")
    if not isinstance(cells, dict):
        raise ValueError(f"{split} replay lacks direct height magnetic cells for {source}")
    result = {}
    for label in NIGHT_V2_CELLS:
        entry = cells.get(label, {}).get("M00", {})
        result[label] = {
            "profile_count": _profile_count(entry),
            "point_count": _profile_count(entry, 'point_count'),
            "bias": entry.get("bias"), "rmse": entry.get("rmse"),
            "ccc": entry.get("ccc"),
            "mean_profile_rmse": entry.get("mean_profile_rmse"),
            "profile_mean_rmse": entry.get("profile_mean_rmse", entry.get("rmse")),
            "date_sufficient_statistics": entry.get("date_sufficient_statistics"),
        }
    return result


def _region_table(replay: dict, source: str, split: str) -> dict:
    metrics = replay["metrics" if split == "development" else "train_metrics"]
    cells = metrics.get(f"{source}_background_stratified", {}).get(
        "magnetic_region_daynight_cells")
    if not isinstance(cells, dict):
        raise ValueError(f"{split} replay lacks direct 200-500 magnetic/daynight cells")
    return {label: cells.get(label, {}).get("M00", {})
            for label in NIGHT_MAGNETIC_LABELS}


def _finite_metric(item: dict, key: str) -> bool:
    value = item.get(key)
    return isinstance(value, (int, float)) and math.isfinite(value)


def _profile_count(item: dict, key: str = 'profile_count') -> int:
    value = item.get(key, 0)
    if (not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or value != int(value)):
        raise ValueError(f'invalid {key}: {value}')
    return int(value)


def _legacy_replay_check(control_run: Path, historical_run: Path, replay: dict) -> dict:
    """Compare reproducible dev scalar/cell values; no historical data hash exists."""
    rows = [row for row in _rows(control_run) if int(row['epoch']) == 5]
    if len(rows) != 1:
        return {'checked': 0, 'passed': False, 'mismatches': ['missing_control_epoch5']}
    row = rows[0]
    errors, checked = [], 0
    for source in ("fy", "cosmic"):
        def compare_values(old, new, path):
            nonlocal checked
            if isinstance(old, dict):
                for key, value in old.items():
                    compare_values(value, new.get(key) if isinstance(new, dict) else None,
                                   f'{path}:{key}')
            elif isinstance(old, (int, float)) and math.isfinite(old):
                checked += 1
                tolerance = 0 if path.endswith(('count', ':n')) else 1e-6
                if (not isinstance(new, (int, float)) or not math.isfinite(new)
                        or abs(new - old) > tolerance):
                    errors.append(path)
        prefix = f'{source}_'
        for key, value in row.items():
            training_suffixes = ('obs', 'active_fraction', 'low_altitude',
                                 'low_altitude_profiles', 'low_altitude_points',
                                 'low_altitude_active_fraction')
            if key.startswith(prefix) and key[len(prefix):] not in training_suffixes:
                compare_values(value, replay['metrics'].get(key), key)
    return {"checked": checked, "passed": checked > 0 and not errors,
            "mismatches": errors, "tolerance": 1e-6,
            "historical_input_sha256": "unavailable"}


def paired_date_bootstrap_with_abs_bias(observation, candidate, baseline,
                                        utc_dates, replicates=2000, seed=42) -> dict:
    """Station-local UTC-date paired bootstrap, including absolute-Bias deltas."""
    observed = np.asarray(observation, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    dates = np.asarray(utc_dates).reshape(-1)
    valid = (np.isfinite(observed) & np.isfinite(candidate) & np.isfinite(baseline))
    if not (len(observed) == len(candidate) == len(baseline) == len(dates)):
        raise ValueError("date bootstrap arrays must have equal lengths")
    observed, candidate, baseline, dates = (value[valid] for value in
                                              (observed, candidate, baseline, dates))
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    if unique_dates.size < 2:
        return {"estimable": False, "date_count": int(unique_dates.size),
                "reason": "fewer than two effective UTC dates"}

    def date_statistics(prediction):
        return np.stack([np.bincount(inverse, weights=values,
                                     minlength=len(unique_dates))
                         for values in (np.ones(len(observed)), observed, prediction,
                                        observed ** 2, prediction ** 2,
                                        observed * prediction,
                                        (prediction - observed) ** 2)], axis=1)
    proposed, reference = date_statistics(candidate), date_statistics(baseline)
    def metrics(totals):
        n, y, p, yy, pp, yp, ee = totals.T
        y, p = y / n, p / n
        covariance = yp / n - y * p
        denominator = np.maximum(yy / n - y * y, 0) + np.maximum(pp / n - p * p, 0) + (p - y) ** 2
        return np.stack((np.sqrt(np.maximum(ee / n, 0)),
                         2 * covariance / (denominator + 1e-12), np.abs(p - y)), axis=-1)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(unique_dates), size=(int(replicates), len(unique_dates)))
    samples = metrics(proposed[chosen].sum(axis=1)) - metrics(reference[chosen].sum(axis=1))
    point = metrics(proposed.sum(axis=0)) - metrics(reference.sum(axis=0))
    return {"estimable": True, "date_count": int(unique_dates.size),
            "replicates": int(replicates), "seed": int(seed),
            "delta": {"rmse": float(point[0]), "ccc": float(point[1]),
                      "absolute_bias": float(point[2])},
            "ci95": {"rmse": np.quantile(samples[:, 0], [.025, .975]).tolist(),
                     "ccc": np.quantile(samples[:, 1], [.025, .975]).tolist(),
                     "absolute_bias": np.quantile(samples[:, 2], [.025, .975]).tolist()}}


def prepare_night_v2_diagnostic(control_run: Path, historical_run: Path,
                                output_dir: Path, allow_replay: bool = False) -> dict:
    """Freeze v2 support from one verified control train/development replay."""
    control_run, historical_run, output_dir = map(Path, (control_run, historical_run, output_dir))
    replay, replay_error = _ensure_v2_replay(control_run, output_dir, allow_replay)
    result = {"semantics": NIGHT_V2_SEMANTICS, "control": _identity(control_run),
              "historical": _identity(historical_run), "locked_test_read": False,
              "can_train": False, "replay": {"path": str(_v2_replay_path(output_dir)),
              "used": replay is not None}, "reasons": [], "supported_cells": [],
              "baseline_metrics": {}, "fixed_rules": {"min_profiles": 30,
              "min_abs_bias_dex": 0.02, "altitude_km": [200, 500],
              "day": "[06:00,18:00)", "comparison_tolerance": 1e-8}}
    if replay is None:
        result.update(status="technical_failure")
        result["reasons"].append(replay_error)
    else:
        try:
            result["replay"]["legacy_verification"] = _legacy_replay_check(
                control_run, historical_run, replay)
            if not result["replay"]["legacy_verification"]["passed"]:
                result.update(status="technical_failure")
                result["reasons"].append("legacy control replay mismatch")
            else:
                exact, regions = {}, {}
                for source in ("fy", "cosmic"):
                    train, development = _cell_table(replay, source, "train"), _cell_table(replay, source, "development")
                    result["baseline_metrics"][source] = development
                    for label in NIGHT_V2_CELLS:
                        key = f"{source}:{label}"
                        exact[key] = {"train": train[label], "development": development[label]}
                        if train[label]["profile_count"] >= 30 and development[label]["profile_count"] >= 30:
                            if any(not _finite_metric(entry, metric)
                                   for entry in (train[label], development[label])
                                   for metric in ('bias', 'rmse', 'ccc', 'mean_profile_rmse', 'profile_mean_rmse')):
                                raise ValueError(f'nonfinite supported cell metric: {key}')
                            result["supported_cells"].append(key)
                        if not isinstance(train[label]["date_sufficient_statistics"], list) or not isinstance(development[label]["date_sufficient_statistics"], list):
                            result["reasons"].append(f"missing reusable date statistics: {key}")
                    regions[source] = {"train": _region_table(replay, source, "train"),
                                       "development": _region_table(replay, source, "development")}
                low_qualified, low_signs = [], []
                for source in ("fy", "cosmic"):
                    train, development = regions[source]["train"].get("abs_sinI_lt_0.25_night", {}), regions[source]["development"].get("abs_sinI_lt_0.25_night", {})
                    qualified = (_profile_count(train) >= 30 and _profile_count(development) >= 30
                                 and _finite_metric(train, "bias") and _finite_metric(development, "bias")
                                 and float(train["bias"]) * float(development["bias"]) > 0
                                 and abs(float(train["bias"])) >= .02 and abs(float(development["bias"])) >= .02)
                    if qualified:
                        low_qualified.append(source); low_signs.append(math.copysign(1., float(development["bias"])))
                for mag in ("abs_sinI_lt_0.25", "abs_sinI_ge_0.5"):
                    for altitude in ("200-300", "300-500"):
                        if not any(f"{source}:{altitude}_night_{mag}" in result["supported_cells"] for source in ("fy", "cosmic")):
                            result["reasons"].append(f"missing supported night coverage: {altitude}_{mag}")
                if not any(key.endswith("_day_abs_sinI_lt_0.25") or "_day_" in key for key in result["supported_cells"]):
                    result["reasons"].append("day support set is empty")
                if not low_qualified:
                    result["reasons"].append("no low-magnetic night source meets bias gate")
                if len(low_signs) == 2 and low_signs[0] != low_signs[1]:
                    result["reasons"].append("qualified low-magnetic sources require opposite corrections")
                result["regions"], result["exact_cells"] = regions, exact
                result["status"] = "diagnostic_supported" if not result["reasons"] else "diagnostic_not_supported"
                result["can_train"] = result["status"] == "diagnostic_supported"
        except (KeyError, TypeError, ValueError) as error:
            result.update(status="technical_failure")
            result["reasons"].append(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "m00_night_v2_preflight.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(
        "# M00 night v2 preflight\n\n"
        f"Status: **{result['status']}**\n\n"
        f"Supported cells: {len(result['supported_cells'])}.\n\n"
        + "\n".join(f"- {reason}" for reason in result["reasons"]) + "\n", encoding="utf-8")
    return result


def cached_night_v2_isr_diagnostic(output_file: Path, candidate_cache: Path | None = None) -> dict:
    """Re-stratify existing canonical rows; never load or infer a model."""
    from run_m00_low_budget import (ISR_CACHE, ISR_COMPARISON_CACHE,
                                    SPARSE_BIAS_ISR_OUTPUT, SPARSE_BIAS_RUN)
    from isr_evaluation.background_m00_eval import _metrics, _reporting_masks
    from inr_modules.mdia.fsia_model import _compute_solar_features
    import torch
    paths = {'ETKF': ISR_CACHE, 'A1S': ISR_COMPARISON_CACHE,
             'Coarse': SPARSE_BIAS_ISR_OUTPUT / 'isr_m00_cache.npz'}
    if candidate_cache is not None:
        paths['Night_v2'] = Path(candidate_cache)
    identity = {name: {'path': str(path.resolve()), 'sha256': _sha256(path)}
                for name, path in paths.items()}
    if any(value['sha256'] is None for value in identity.values()):
        raise FileNotFoundError('required cached ISR artifact is missing')
    output_file = Path(output_file)
    if output_file.is_file():
        saved = json.loads(output_file.read_text(encoding='utf-8'))
        if saved.get('identity') != identity:
            raise ValueError('cached ISR diagnostic identity changed; refusing overwrite')
        return saved
    caches = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            caches[name] = {key: data[key] for key in data.files}
    canonical = caches['ETKF']
    if len(canonical['key']) != 206230 or len(np.unique(canonical['key'])) != 206230:
        raise ValueError('canonical ISR row identity/count changed')
    for name, cache in caches.items():
        for key in ('station', 'key', 'timestamp', 'unit_id', 'altitude_km'):
            if not np.array_equal(cache[key], canonical[key]):
                raise ValueError(f'{name} cache alignment mismatch: {key}')
    predictions = {'ETKF': canonical['ETKF_log10'],
                   'Raw_IRI': canonical['raw_iri_continuous_log10'],
                   'A1S': caches['A1S']['M00_A1S_log10'],
                   'Coarse': caches['Coarse']['candidate_log10']}
    if candidate_cache is not None:
        predictions['Night_v2'] = caches['Night_v2']['candidate_log10']
    observed = canonical['observation'].astype(float)
    common = np.isfinite(observed)
    for prediction in predictions.values():
        common &= np.isfinite(prediction)
    dates = canonical['timestamp'].astype('datetime64[s]').astype('datetime64[D]')
    anchor = np.datetime64('2024-09-01T00:00:00', 's').astype(np.int64)
    relative_hours = (canonical['timestamp'].astype(float) - anchor) / 3600.
    local_time = (relative_hours + canonical['lon'] / 15.) % 24.
    cosine = _compute_solar_features(
        torch.as_tensor(canonical['lat'], dtype=torch.float64),
        torch.as_tensor(canonical['lon'], dtype=torch.float64),
        torch.as_tensor(relative_hours, dtype=torch.float64))[0].numpy()
    rows, intervals, saturation = [], [], []
    comparisons = [('Coarse', 'A1S'), ('A1S', 'ETKF'), ('Coarse', 'ETKF')]
    if candidate_cache is not None:
        comparisons.extend(('Night_v2', baseline) for baseline in ('A1S', 'Coarse', 'ETKF'))
    coarse_config = _manifest_config(SPARSE_BIAS_RUN)
    cap = float(coarse_config['background_residual_cap'])
    coarse_residual = caches['Coarse'].get('background_base_residual_raw')
    if coarse_config.get('background_condition_adapter', 'disabled') != 'disabled':
        coarse_residual = None  # Do not substitute a sum of heads for actual bounded residual.
    for station in ('Jicamarca', 'PokerFlat'):
        for stratum, mask in _reporting_masks(canonical['altitude_km'], local_time, cosine):
            selected = mask & common & (canonical['station'] == station)
            for model, prediction in predictions.items():
                rows.append({'station': station, 'stratum': stratum, 'model': model,
                             **_metrics(observed[selected], prediction[selected])})
            for candidate, baseline in comparisons:
                intervals.append({'station': station, 'stratum': stratum,
                                  'candidate': candidate, 'baseline': baseline,
                                  **paired_date_bootstrap_with_abs_bias(
                                      observed[selected], predictions[candidate][selected],
                                      predictions[baseline][selected], dates[selected])})
            if coarse_residual is not None:
                residual = coarse_residual[selected]
                residual = residual[np.isfinite(residual)]
                saturation.append({'station': station, 'stratum': stratum, 'model': 'Coarse',
                                   'n': len(residual), 'cap': cap,
                                   'near_cap_rate': float(np.mean(np.abs(residual) >= .95 * cap))
                                   if len(residual) else None})
    report = {'semantics': 'cached_isr_night_v2_diagnostic_v1', 'identity': identity,
              'row_count': len(observed), 'common_valid_count': int(common.sum()),
              'scope': 'previously viewed diagnostic validation; not a new independent test',
              'stratified_metrics': rows, 'paired_utc_date_bootstrap': intervals,
              'bootstrap': {'replicates': 2000, 'seed': 42, 'sampling_unit': 'station_utc_date'},
              'saturation': saturation, 'saturation_rule': 'abs(residual) >= 0.95 * cap',
              'gaps': ['A1S cached residual unavailable; no model replay performed'],
              'local_time_day': '[06:00,18:00)', 'solar_day': 'SZA < 90 degrees'}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def harmful_correction_metrics(observation, iri, m00, profile_id,
                               include_profile_metrics=True):
    """Classify log10 corrections, then average the classes per profile."""
    y, b, p, ids = (np.asarray(x, dtype=np.float64).reshape(-1)
                    for x in (observation, iri, m00, profile_id))
    if not (len(y) == len(b) == len(p) == len(ids)):
        raise ValueError('diagnostic arrays must have equal lengths')
    finite = (np.isfinite(y) & np.isfinite(b) & np.isfinite(p)
              & np.isfinite(ids))
    d, c = y - b, p - b
    h = c * c - 2.0 * c * d
    near_zero = np.abs(c) <= 1e-6
    # ``d`` is the observation residual.  Keep this mask independent from
    # the correction mask: the H/base classes are intentionally overlapping
    # diagnostics, not a partition of one another.
    near_zero_observation_residual = finite & (np.abs(d) < .02)
    direction = finite & ~near_zero & ~near_zero_observation_residual
    harmful = finite & (h > 1e-8)
    useful = finite & (h < -1e-8)
    reverse = direction & (c * d < 0)
    same_direction_over = direction & (c * d > 0) & (h > 1e-8)
    near_zero_observation_error = near_zero_observation_residual
    mean_keys = ('harmful_fraction', 'useful_fraction', 'reverse_fraction',
                 'same_direction_over_fraction',
                 'near_zero_observation_error_fraction', 'mean_H',
                 'mean_harmful_H', 'mean_abs_c', 'm00_rmse', 'iri_rmse')
    profile_values = []
    profile_count = 0
    means = {key: None for key in mean_keys}
    if finite.any():
        valid_ids = ids[finite]
        order = np.argsort(valid_ids, kind='stable')
        sorted_ids = valid_ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
        counts = np.diff(np.r_[starts, len(sorted_ids)]).astype(np.float64)
        profile_count = int(starts.size)
        def profile_mean(values):
            values = np.asarray(values, dtype=np.float64)[finite][order]
            return np.add.reduceat(values, starts) / counts
        profile_arrays = {
            'harmful_fraction': profile_mean(harmful),
            'useful_fraction': profile_mean(useful),
            'reverse_fraction': profile_mean(reverse),
            'same_direction_over_fraction': profile_mean(same_direction_over),
            'near_zero_observation_error_fraction': profile_mean(
                near_zero_observation_error),
            'mean_H': profile_mean(h),
            'mean_harmful_H': profile_mean(np.maximum(h, 0.0)),
            'mean_abs_c': profile_mean(np.abs(c)),
            'm00_rmse': np.sqrt(profile_mean((p - y) ** 2)),
            'iri_rmse': np.sqrt(profile_mean((b - y) ** 2)),
        }
        means = {key: float(np.mean(profile_arrays[key])) for key in mean_keys}
        if include_profile_metrics:
            for index, uid in zip(starts, sorted_ids[starts]):
                profile_values.append({
                    'profile_id': int(uid) if float(uid).is_integer() else float(uid),
                    'point_count': int(counts[len(profile_values)]),
                    **{key: float(values[len(profile_values)])
                       for key, values in profile_arrays.items()},
                })
    return {
        'point_count': int(finite.sum()),
        'invalid_point_count': int((~finite).sum()),
        'profile_count': profile_count,
        'reverse_point_count': int(reverse.sum()),
        'same_direction_over_point_count': int(same_direction_over.sum()),
        'near_zero_observation_residual_point_count': int(
            near_zero_observation_error.sum()),
        'near_zero_observation_error_point_count': int(
            near_zero_observation_error.sum()),
        'harmful_point_count': int(harmful.sum()),
        'useful_point_count': int(useful.sum()),
        'near_zero_c_count': int((finite & near_zero).sum()),
        'direction_point_count': int(direction.sum()),
        **means,
        'point_h': h,
        'point_reverse': reverse,
        'point_harmful': harmful,
        'point_useful': useful,
        'point_near_zero': finite & near_zero,
        'profile_metrics': profile_values,
    }


def _harmful_strata(coords):
    coords = np.asarray(coords, dtype=np.float64)
    lt = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    day = np.where((lt >= 6) & (lt < 18), 'day', 'night')
    hour = (np.floor(lt / 3).astype(int) * 3).astype(str)
    try:
        import torch
        from inr_modules.mdia.mdia_model import _compute_dip_features
        magnetic = np.abs(_compute_dip_features(
            torch.as_tensor(coords[:, 0]), torch.as_tensor(coords[:, 1]))[0]
            .detach().cpu().numpy())
    except (ImportError, RuntimeError, ValueError):
        # Keep strata usable for synthetic/unit data; production configs have
        # the dip-feature helper available and therefore never use this path.
        magnetic = np.abs(np.sin(np.deg2rad(coords[:, 0])))
    mag = np.where(magnetic < .25, 'abs_sinI_lt_0.25',
                   np.where(magnetic < .5, 'abs_sinI_0.25_0.5',
                            'abs_sinI_ge_0.5'))
    alt = coords[:, 2]
    height = np.where((alt >= 120) & (alt < 200), '120-200',
                      np.where((alt >= 200) & (alt < 300), '200-300',
                               np.where((alt >= 300) & (alt <= 500), '300-500', 'out')))
    return lt, day, hour, mag, height


def _harmful_identity(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': _sha256(path)}


def _manifest_input_identity(run_dir: Path, config: dict) -> dict:
    payload = json.loads((Path(run_dir) / 'run_manifest.json').read_text(
        encoding='utf-8'))
    frozen = payload.get('data_identity')
    if isinstance(frozen, dict) and frozen:
        return {str(item.get('path', key)): item.get('sha256')
                for key, item in frozen.items() if isinstance(item, dict)}
    return _input_hashes(config)


def _harmful_run_identity(run_dir: Path, checkpoint: Path,
                          sampling: dict) -> dict:
    run_dir = Path(run_dir)
    config = _manifest_config(run_dir)
    return {
        'run': str(run_dir.resolve()),
        'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': _sha256(Path(checkpoint)),
        'run_manifest_sha256': _sha256(run_dir / 'run_manifest.json'),
        'config_sha256': _json_sha256(config),
        'current_input_sha256': _manifest_input_identity(run_dir, config),
        'date_split_sha256': _sha256(Path(config['date_split_manifest'])),
        'historical_input_sha256': 'unavailable',
        'sampling': sampling,
        'sampling_sha256': _json_sha256(sampling),
    }


def _grid_coverage(coords, profile_ids, mask):
    coords = np.asarray(coords, dtype=np.float64)
    ids = np.asarray(profile_ids).reshape(-1)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(coords).all(axis=1) & np.isfinite(ids)
    if not valid.any():
        return {'grid_cell_count': 0, 'grid_profile_count': 0,
                'grid_point_count': 0, 'grid_profile_concentration_max': None}
    lat = np.clip(coords[valid, 0], -90.0, 89.999999)
    lon = np.mod(coords[valid, 1] + 180.0, 360.0)
    cells = np.floor((lat + 90.0) / 10.0).astype(np.int64) * 18
    cells += np.floor(lon / 20.0).astype(np.int64)
    unique_cells, inverse = np.unique(cells, return_inverse=True)
    profiles = np.asarray(ids[valid], dtype=np.int64)
    profile_cells = {(int(cell), int(pid))
                     for cell, pid in zip(cells.tolist(), profiles.tolist())}
    counts = np.bincount(
        np.asarray([cell for cell, _ in profile_cells], dtype=np.int64),
        minlength=180)
    total_profiles = len(profile_cells)
    return {
        'grid_cell_count': int(unique_cells.size),
        'grid_profile_count': int(total_profiles),
        'grid_point_count': int(valid.sum()),
        'grid_profile_concentration_max': float(counts.max() / total_profiles)
        if total_profiles else None,
    }


def _date_profile_statistics(observation, iri, m00, profile_ids, dates, selected=None):
    result = {}
    dates = np.asarray(dates).reshape(-1)
    finite = (np.isfinite(observation) & np.isfinite(iri)
              & np.isfinite(m00) & np.isfinite(profile_ids)
              & np.isfinite(dates))
    if selected is not None:
        finite &= np.asarray(selected, dtype=bool)
    for day in np.unique(dates[finite]):
        mask = finite & (dates == day)
        metrics = harmful_correction_metrics(
            observation[mask], iri[mask], m00[mask], profile_ids[mask],
            include_profile_metrics=True)
        result[int(day)] = {
            'harmful_fraction': metrics['harmful_fraction'],
            'mean_H': metrics['mean_H'],
            'profile_count': int(metrics['profile_count']),
            'profile_values': [
                (float(item['harmful_fraction']), float(item['mean_H']))
                for item in metrics['profile_metrics']],
        }
    return result


def paired_date_bootstrap_harmful(candidate, control, mask, replicates=2000,
                                   seed=42):
    """Paired UTC-date bootstrap of profile-equal harmful fraction and H."""
    common = np.asarray(mask, dtype=bool)
    dates = np.floor(np.asarray(candidate['coords'])[:, 3] / 24.0)
    candidate_stats = _date_profile_statistics(
        candidate['observation'], candidate['iri'], candidate['m00'],
        candidate['profile_id'], dates, common)
    control_stats = _date_profile_statistics(
        control['observation'], control['iri'], control['m00'],
        control['profile_id'], dates, common)
    days = sorted(set(candidate_stats) & set(control_stats))
    if len(days) < 2:
        return {'estimable': False, 'date_count': len(days),
                'reason': 'fewer than two effective UTC dates'}
    # A date block contributes all of its profiles.  The point estimate and
    # each bootstrap replicate therefore use the same profile-equal estimator;
    # averaging one scalar per date would silently become date-equal weighting.
    day_profile_deltas = {
        day: np.asarray(candidate_stats[day]['profile_values'], dtype=np.float64)
        - np.asarray(control_stats[day]['profile_values'], dtype=np.float64)
        for day in days
    }
    if any(values.size == 0 for values in day_profile_deltas.values()):
        return {'estimable': False, 'date_count': len(days),
                'reason': 'no effective profiles in one or more dates'}
    all_deltas = np.concatenate([day_profile_deltas[day] for day in days], axis=0)
    point = all_deltas.mean(axis=0)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(days), size=(int(replicates), len(days)))
    samples = np.empty((int(replicates), 2), dtype=np.float64)
    for index, row in enumerate(chosen):
        samples[index] = np.concatenate(
            [day_profile_deltas[days[day_index]] for day_index in row],
            axis=0).mean(axis=0)
    return {
        'estimable': True, 'date_count': len(days),
        'replicates': int(replicates), 'seed': int(seed),
        'delta': {'harmful_fraction': float(point[0]),
                  'mean_H': float(point[1])},
        'ci95': {
            'harmful_fraction': np.quantile(samples[:, 0], [.025, .975]).tolist(),
            'mean_H': np.quantile(samples[:, 1], [.025, .975]).tolist(),
        },
    }


def _reconstruct_v2_sampling(run_dir: Path) -> dict:
    config = _manifest_config(run_dir)
    history_path = Path(run_dir) / 'fsia_training_history.json'
    if not history_path.is_file():
        return {'verified': False, 'reason': 'missing frozen training history'}
    history = json.loads(history_path.read_text(encoding='utf-8'))
    rows = [row for row in history if row.get('stage') == 'background']
    expected_epochs = int(config.get('background_epochs', 5))
    epochs = []
    reasons = []
    for row in rows:
        audit = row.get('natural_epoch_audit') or {}
        epochs.append({
            'epoch': int(row.get('epoch', len(epochs) + 1)),
            'seed': int(config.get('background_profile_schedule_seed',
                                   config.get('seed', 42))),
            'source_profile_totals': audit.get('source_profile_totals'),
            'total_updates': audit.get('total_updates'),
            'missing_profile_counts': audit.get('missing_profile_counts'),
            'duplicate_profile_counts': audit.get('duplicate_profile_counts'),
            'condition_activation': row.get('magnetic_condition_activation'),
        })
        if audit.get('missing_profile_counts') != {'FY': 0, 'COSMIC': 0}:
            reasons.append(f"epoch {row.get('epoch')}: missing profiles")
        if audit.get('duplicate_profile_counts') != {'FY': 0, 'COSMIC': 0}:
            reasons.append(f"epoch {row.get('epoch')}: duplicate profiles")
    return {
        'semantics': config.get('background_sampling_semantics'),
        'seed': int(config.get('background_profile_schedule_seed',
                              config.get('seed', 42))),
        'profile_points_per_epoch': int(config.get('profile_points_per_epoch', 8)),
        'support_min_profiles': int(config.get('background_bias_magnetic_min_profiles', 30)),
        'batch_min_profiles': int(config.get('background_bias_magnetic_min_batch_profiles', 8)),
        'expected_epochs': expected_epochs, 'epochs': epochs,
        'verified': len(epochs) == expected_epochs and not reasons,
        'reasons': reasons or ([] if len(epochs) == expected_epochs else
                               ['incomplete five-epoch history']),
    }


def _run_harmful_replay(run_dir: Path, checkpoint: Path, model_dir: Path) -> None:
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty diagnostic directory: {model_dir}')
    model_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(_manifest_config(run_dir))
    config.update({'save_dir': str(model_dir), 'resume_ckpt': str(checkpoint),
                   'eval_only': True, 'background_only': True,
                   'm00_diagnostic_train_split': True,
                   'm00_point_export_dir': str(model_dir),
                   'm00_point_sample_size': 8,
                   'm00_diagnostic_sampling_semantics':
                   'shuffle_false_first_8_raw_rows_per_profile_v1',
                   'skip_hash_verification': True})
    run_config(config, model_dir, experiment='M00 harmful correction diagnostic')


def _harmful_cache_complete(output: Path, models: dict) -> bool:
    identity_path = output / 'diagnostic_identity.json'
    if not identity_path.is_file():
        return False
    return all((output / name / f'{source.lower()}_{split}.npz').is_file()
               for name in models for source in ('FY', 'COSMIC')
               for split in ('train', 'development'))


def _harmful_cache_hashes(output: Path, models: dict) -> dict:
    return {
        f'{name}/{source}/{split}': _sha256(
            output / name / f'{source.lower()}_{split}.npz')
        for name in models for source in ('FY', 'COSMIC')
        for split in ('train', 'development')
    }


def diagnose_harmful_correction(candidate_run: Path, control_run: Path,
                                candidate_checkpoint: Path, control_checkpoint: Path,
                                output: Path) -> dict:
    """Replay two explicit epochs once, then compute frozen harmful metrics."""
    output = Path(output)
    candidate_checkpoint, control_checkpoint = (Path(candidate_checkpoint),
                                                 Path(control_checkpoint))
    if not candidate_checkpoint.is_file() or not control_checkpoint.is_file():
        raise FileNotFoundError('both explicit epoch-5 checkpoints are required')
    if (candidate_checkpoint.name != 'epoch_05_background_model.pth'
            or control_checkpoint.name != 'epoch_05_background_model.pth'):
        raise ValueError('harmful correction diagnostic requires explicit epoch 5 checkpoints')
    sampling = {
        'shuffle': False, 'points_per_profile': 8,
        'selection': 'first_8_raw_rows_after_stable_profile_order',
        'splits': ['train', 'development'], 'locked_test_read': False,
    }
    models = {'v2_epoch5': (Path(candidate_run), Path(candidate_checkpoint)),
              'explicit_epoch5': (Path(control_run), Path(control_checkpoint))}
    reuse_cache = output.exists() and any(output.iterdir())
    if reuse_cache and not _harmful_cache_complete(output, models):
        raise FileExistsError(f'refusing to overwrite incomplete output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    candidate_identity = _harmful_run_identity(
        candidate_run, candidate_checkpoint, sampling)
    control_identity = _harmful_run_identity(
        control_run, control_checkpoint, sampling)
    if (candidate_identity['date_split_sha256']
            != control_identity['date_split_sha256']):
        raise ValueError('candidate/control date-split identities differ')
    if (candidate_identity['current_input_sha256']
            != control_identity['current_input_sha256']):
        raise ValueError('candidate/control input identities differ')
    identity = {'semantics': 'm00_harmful_correction_identity_v1',
                'candidate': candidate_identity,
                'control': control_identity,
                'historical_input_sha256': 'unavailable'}
    identity_path = output / 'diagnostic_identity.json'
    if reuse_cache:
        saved_identity = json.loads(identity_path.read_text(encoding='utf-8'))
        saved_base = dict(saved_identity)
        saved_base.pop('cache_sha256', None)
        if saved_base != identity:
            raise ValueError('cached diagnostic identity changed; refusing reuse')
    else:
        identity_path.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for name, (run_dir, checkpoint) in models.items():
            _run_harmful_replay(run_dir, checkpoint, output / name)
    expected = {(name, source, split): output / name / f'{source.lower()}_{split}.npz'
                for name in models for source in ('FY', 'COSMIC')
                for split in ('train', 'development')}
    caches = {}
    for name in models:
        caches[name] = {}
        for source in ('FY', 'COSMIC'):
            for split in ('train', 'development'):
                path = expected[(name, source, split)]
                if not path.is_file():
                    raise RuntimeError(f'incomplete point cache: {path}')
                with np.load(path, allow_pickle=False) as data:
                    required = ('source', 'row_index', 'profile_id', 'coords',
                                'observation', 'iri', 'm00', 'raw_residual',
                                'residual', 'trust_gate', 'sampling_shuffle',
                                'point_sample_size', 'duplicate_raw_row_count')
                    if any(key not in data for key in required):
                        raise ValueError(f'incomplete point cache: {path}')
                    cache = {key: np.asarray(data[key]) for key in data.files}
                # Older complete caches encoded the split in the filename;
                # preserve their values while making the identity explicit.
                cache.setdefault('split', np.asarray(split))
                if (str(cache['source'].item()).upper() != source
                        or str(cache['split'].item()).lower() != split):
                    raise ValueError(f'cache source/split identity changed: {path}')
                if (bool(cache['sampling_shuffle'].item())
                        or int(cache['point_sample_size'].item()) != 8):
                    raise ValueError(f'cache sampling identity changed: {path}')
                lengths = {len(cache[key]) for key in (
                    'row_index', 'profile_id', 'coords', 'observation', 'iri',
                    'm00', 'raw_residual', 'residual', 'trust_gate')}
                if len(lengths) != 1 or cache['coords'].ndim != 2 \
                        or cache['coords'].shape[1] < 4:
                    raise ValueError(f'incomplete point cache lengths: {path}')
                if len(np.unique(cache['row_index'])) != len(cache['row_index']):
                    raise ValueError(f'duplicate cache identities: {path}')
                caches[name][(source, split)] = cache
    cache_hashes = _harmful_cache_hashes(output, models)
    if (reuse_cache and saved_identity.get('cache_sha256') is not None
            and saved_identity.get('cache_sha256') != cache_hashes):
        raise ValueError('cached point values changed; refusing reuse')
    identity['cache_sha256'] = cache_hashes
    identity_path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for source in ('FY', 'COSMIC'):
        for split in ('train', 'development'):
            left = caches['v2_epoch5'][(source, split)]
            right = caches['explicit_epoch5'][(source, split)]
            if not np.array_equal(left['row_index'], right['row_index']):
                raise ValueError(f'{source}/{split} cache row identities differ')
            if not np.array_equal(left['profile_id'], right['profile_id']):
                raise ValueError(f'{source}/{split} cache profile identities differ')
            if not np.allclose(left['coords'], right['coords'], atol=0.0, rtol=0.0):
                raise ValueError(f'{source}/{split} cache coordinates differ')
            for key in ('observation', 'iri'):
                if not np.allclose(left[key], right[key], equal_nan=True,
                                   atol=1e-6, rtol=0.0):
                    raise ValueError(f'{source}/{split} {key} identity differs')
    report = {'semantics': 'm00_harmful_correction_diagnostic_v1',
              'identity': identity, 'locked_test_read': False, 'models': {},
              'stratified_metrics': [], 'paired_date_bootstrap': [],
              'bootstrap': {'replicates': 2000, 'seed': 42,
                            'sampling_unit': 'profile_utc_date'},
              'participation': _reconstruct_v2_sampling(candidate_run),
              'validation': {'iri_identity_tolerance_dex': 1e-6,
                             'common_sample_identity': True,
                              'diagnostic_only': True,
                              'training_runs': 0, 'calibration_runs': 0,
                              'isr_runs': 0, 'common_valid_rows': {},
                              'model_valid_rows': {},
                              'gaps': [
                                  'historical training-time input hashes are unavailable',
                                  'locked-test split was not read',
                                  'A1S full replay and ISR inference were not run',
                              ]}}
    h_identity_max = 0.0
    for cache_set in caches.values():
        for cache in cache_set.values():
            finite = (np.isfinite(cache['observation'])
                      & np.isfinite(cache['iri']) & np.isfinite(cache['m00']))
            d = cache['observation'][finite] - cache['iri'][finite]
            c = cache['m00'][finite] - cache['iri'][finite]
            direct = ((cache['m00'][finite] - cache['observation'][finite]) ** 2
                      - (cache['iri'][finite] - cache['observation'][finite]) ** 2)
            if finite.any():
                h_identity_max = max(h_identity_max, float(np.max(
                    np.abs(c * c - 2.0 * c * d - direct))))
    report['validation']['h_identity_max_abs'] = h_identity_max
    timings = {}
    for name in models:
        timing_path = output / name / 'run_timing.json'
        timings[name] = (json.loads(timing_path.read_text(encoding='utf-8'))
                         if timing_path.is_file() else None)
    points_by_model = {
        name: int(sum(len(caches[name][key]['row_index'])
                      for key in caches[name])) for name in models}
    forward_points_by_model = {}
    for name in models:
        replay_path = output / name / 'background_development_replay.json'
        replay = json.loads(replay_path.read_text(encoding='utf-8'))
        total = 0
        for metric_key in ('train_metrics', 'metrics'):
            metrics = replay.get(metric_key, {})
            for source in ('fy', 'cosmic'):
                cells = metrics.get(f'{source}_background_stratified', {}).get(
                    'cells', {})
                total += sum(int(cell.get('M00', {}).get('point_count', 0) or 0)
                             for cell in cells.values())
        forward_points_by_model[name] = int(total)
    report['execution'] = {
        'cache_reused': bool(reuse_cache),
        'prediction_invocations_this_run': 0 if reuse_cache else 2,
        'historical_prediction_invocations': 2,
        'exported_sample_points_by_model': points_by_model,
        'exported_sample_points_total': int(sum(points_by_model.values())),
        'forward_points_by_model': forward_points_by_model,
        'forward_points_total': int(sum(forward_points_by_model.values())),
        'eval_timing': timings,
        'exceptions': [],
    }
    for source in ('FY', 'COSMIC'):
        for split in ('train', 'development'):
            key = f'{source}_{split}'
            masks = {}
            for name in models:
                cache = caches[name][(source, split)]
                masks[name] = (np.isfinite(cache['observation'])
                               & np.isfinite(cache['iri'])
                               & np.isfinite(cache['m00'])
                               & np.isfinite(cache['profile_id'])
                               & np.isfinite(cache['coords']).all(axis=1))
                report['validation']['model_valid_rows'].setdefault(name, {})[key] = int(masks[name].sum())
            report['validation']['common_valid_rows'][key] = int(
                (masks['v2_epoch5'] & masks['explicit_epoch5']).sum())
    for name in models:
        report['models'][name] = {
            'checkpoint': _harmful_identity(models[name][1]),
            'files': [_harmful_identity(expected[(name, source, split)])
                      for source in ('FY', 'COSMIC')
                      for split in ('train', 'development')],
            'cache_rows': {f'{source}_{split}': int(len(caches[name][(source, split)]['row_index']))
                          for source in ('FY', 'COSMIC')
                          for split in ('train', 'development')},
            'invalid_rows': {f'{source}_{split}': int((
                ~np.isfinite(caches[name][(source, split)]['observation'])
                | ~np.isfinite(caches[name][(source, split)]['iri'])
                | ~np.isfinite(caches[name][(source, split)]['m00'])).sum())
                             for source in ('FY', 'COSMIC')
                             for split in ('train', 'development')},
        }
        for (source, split), cache in caches[name].items():
            coords = cache['coords']
            lt, day, hour, mag, height = _harmful_strata(coords)
            finite = (np.isfinite(cache['observation'])
                      & np.isfinite(cache['iri']) & np.isfinite(cache['m00'])
                      & np.isfinite(cache['profile_id'])
                      & np.isfinite(coords).all(axis=1))
            height_masks = {
                '120-200': (coords[:, 2] >= 120) & (coords[:, 2] < 200),
                '200-300': (coords[:, 2] >= 200) & (coords[:, 2] < 300),
                '300-500': (coords[:, 2] >= 300) & (coords[:, 2] <= 500),
                '200-500': (coords[:, 2] >= 200) & (coords[:, 2] <= 500),
            }
            dimensions = [
                ('daynight', [('day', day == 'day'), ('night', day == 'night')]),
                ('lt3h', [(f'{int(value):02d}-{(int(value) + 3) % 24:02d}',
                           hour == str(int(value))) for value in range(0, 24, 3)]),
            ]
            for height_label, height_mask in height_masks.items():
                for period_kind, periods in dimensions:
                    for period_label, period_mask in periods:
                        for magnetic_label in MAGNETIC_LABELS:
                            mask = finite & height_mask & period_mask & (mag == magnetic_label)
                            if not mask.any():
                                continue
                            item = harmful_correction_metrics(
                                cache['observation'][mask], cache['iri'][mask],
                                cache['m00'][mask], cache['profile_id'][mask],
                                include_profile_metrics=False)
                            valid_dates = np.floor(coords[mask, 3] / 24.0)
                            row = {
                                'model': name, 'source': source, 'split': split,
                                'height': height_label, 'period_kind': period_kind,
                                'period': period_label, 'magnetic': magnetic_label,
                                'utc_date_count': int(np.unique(valid_dates).size),
                                **{key: value for key, value in item.items()
                                   if not isinstance(value, (np.ndarray, list, dict))},
                                **_grid_coverage(coords, cache['profile_id'], mask),
                            }
                            row['low_support'] = bool(row['profile_count'] < 30)
                            report['stratified_metrics'].append(row)
            if name == 'v2_epoch5':
                for period_label, period_mask in (('day', day == 'day'),
                                                  ('night', day == 'night')):
                    for magnetic_label in MAGNETIC_LABELS:
                        pair_mask = finite & (coords[:, 2] >= 200) & (coords[:, 2] <= 500) \
                            & period_mask & (mag == magnetic_label)
                        if pair_mask.any():
                            paired = paired_date_bootstrap_harmful(
                                caches['v2_epoch5'][(source, split)],
                                caches['explicit_epoch5'][(source, split)], pair_mask)
                            report['paired_date_bootstrap'].append({
                                'source': source, 'split': split,
                                'height': '200-500', 'period': period_label,
                                'magnetic': magnetic_label, **paired})
    overall = []
    for name in models:
        for (source, split), cache in caches[name].items():
            metrics = harmful_correction_metrics(
                cache['observation'], cache['iri'], cache['m00'],
                cache['profile_id'], include_profile_metrics=False)
            overall.append({
                'model': name, 'source': source, 'split': split,
                **{key: value for key, value in metrics.items()
                   if not isinstance(value, (np.ndarray, list, dict))},
            })
    report['overall_profile_equal_metrics'] = overall
    report['evidence_boundary'] = {
        'observed_facts': [
            'H is evaluated pointwise in log10 space before profile averaging.',
            'A fixed eight-row stable sample is used for every authorized profile.',
            'Train/development and FY/COSMIC remain separate.',
        ],
        'supported_explanations': [
            'The participation audit can establish historical activation exposure, '
            'but prediction-only results cannot identify a gradient mechanism.',
            'A train/development difference in harmful correction is evidence about '
            'generalization, not proof of an optimization constraint.',
        ],
        'unconfirmed_mechanisms': [
            'gradient conflict, loss competition, and causal responsibility of batch organization',
        ],
    }
    report['recommendation'] = (
        '证据不足，暂不建议训练；先审阅逐点缓存、低支持单元与五轮参与重建。'
        if report['participation'].get('verified') else
        '参与重建未验证，暂不建议训练；先修复或核对采样审计。')
    destination = output / 'm00_harmful_correction_report.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x) + '\n', encoding='utf-8')
    with (output / 'm00_harmful_correction_stratified_metrics.csv').open(
            'w', newline='', encoding='utf-8-sig') as stream:
        rows = report['stratified_metrics']
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    participation = report['participation']
    recommendation = report['recommendation']
    summary_lines = []
    for row in report['overall_profile_equal_metrics']:
        summary_lines.append(
            f"| {row['model']} | {row['source']} | {row['split']} | "
            f"{row['profile_count']} | {row['harmful_fraction']:.6g} | "
            f"{row['mean_H']:.6g} | {row['m00_rmse']:.6g} | {row['iri_rmse']:.6g} |")
    (output / 'm00_harmful_correction_report.md').write_text(
        '# M00 夜间有害修正诊断（阶段 A）\n\n'
        '本报告仅使用两个明确 epoch-5 checkpoint 的 eval-only 前向；未训练、未校准、未运行 ISR。\n\n'
        f"参与重建 verified：`{participation.get('verified')}`。\n\n"
        f"缓存复用：`{report['execution']['cache_reused']}`；本次前向次数：`{report['execution']['prediction_invocations_this_run']}`；"
        f"历史双模型前向次数：`{report['execution']['historical_prediction_invocations']}`。\n\n"
        '| model | source | split | profiles | harmful fraction | mean H | M00 RMSE | raw-IRI RMSE |\n'
        '|---|---|---:|---:|---:|---:|---:|---:|\n'
        + '\n'.join(summary_lines) + '\n\n'
        f"建议：{recommendation}\n\n"
        '历史训练时完整数据哈希不可追溯；本次缓存保存当前身份、逐点行索引及 SHA-256。\n',
        encoding='utf-8')
    (output / 'm00_harmful_correction_participation.json').write_text(
        json.dumps(participation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("control_run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sparse-gate", action="store_true")
    parser.add_argument("--night-gate", action="store_true")
    parser.add_argument("--harmful-correction", action="store_true")
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--control-checkpoint", type=Path)
    args = parser.parse_args()
    if args.harmful_correction:
        if not (args.candidate_checkpoint and args.control_checkpoint and args.output):
            parser.error('--harmful-correction requires --candidate-checkpoint, '
                         '--control-checkpoint and --output')
        diagnose_harmful_correction(
            args.candidate_run, args.control_run, args.candidate_checkpoint,
            args.control_checkpoint, args.output)
    elif args.night_gate:
        diagnose_night_generalization(args.candidate_run, args.control_run, args.output)
    elif args.sparse_gate:
        diagnose_sparse_coverage(args.candidate_run, args.control_run, args.output)
    else:
        compare(args.candidate_run, args.control_run, args.output)


if __name__ == "__main__":
    main()
