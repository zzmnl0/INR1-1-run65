"""Qualify the legacy A+C report or the authorized seed42 M00 candidate.

The trainer and this module share ``select_m00_background_candidate``.  The
``seed42-single`` path is the current one-run explicit-condition protocol;
the older ``auto`` path remains available for historical report inspection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from run_a1s_natural_epoch import DATE_SPLIT, HISTORICAL_RUN, ROOT, build_config, run_config

sys.path.insert(0, str(ROOT / "inr_modules"))
from inr_modules.mdia.train_fsia import select_m00_background_candidate


R2_RUN = ROOT / "checkpoints_fsia" / "run67-m2w-120-500-background-a1s-natural-epoch-r2"
A_RUN = ROOT / "checkpoints_fsia" / "run67-m2w-120-500-background-a1s-natural-epoch-a-batch1024"
A_SEED43_RUN = ROOT / "checkpoints_fsia" / "run67-m2w-120-500-background-a1s-natural-epoch-a-batch1024-seed43"
AC_RUN = ROOT / "checkpoints_fsia" / "run67-m2w-120-500-background-a1s-natural-epoch-a-plus-c-bias0025"
AC_SEED43_RUN = ROOT / "checkpoints_fsia" / "run67-m2w-120-500-background-a1s-natural-epoch-a-plus-c-bias0025-seed43"
EXPLICIT_CONDITION_RUN = ROOT / "checkpoints_fsia" / "run67-m00-seed42-explicit-condition-v1"
EXPLICIT_CONDITION_ISR_OUTPUT = ROOT / "isr_validation_outputs" / "m00-seed42-explicit-condition-v1-isr"
SPARSE_BIAS_RUN = ROOT / "checkpoints_fsia" / "run67-m00-seed42-coarse-magnetic-bias-v1"
SPARSE_BIAS_ISR_OUTPUT = ROOT / "isr_validation_outputs" / "m00-seed42-coarse-magnetic-bias-v1-isr"
SPARSE_BIAS_DIAGNOSTIC = SPARSE_BIAS_RUN / "m00_sparse_coverage_diagnostic.json"
NIGHT_BIAS_RUN = ROOT / "checkpoints_fsia" / "run67-m00-seed42-magnetic-daynight-bias-v1"
NIGHT_BIAS_DIAGNOSTIC = ROOT / "checkpoints_fsia" / "m00_seed42_magnetic_daynight_diagnostic.json"
NIGHT_BIAS_PLAN_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_magnetic_daynight_bias_plan.json"
NIGHT_BIAS_PLAN_MD_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_magnetic_daynight_bias_plan.md"
NIGHT_V2_RUN = ROOT / "checkpoints_fsia" / "run67-m00-seed42-magnetic-daynight-bias-v2"
NIGHT_V2_DIAGNOSTIC_DIR = ROOT / "checkpoints_fsia" / "m00-night-v2-diagnostic"
NIGHT_V2_PLAN_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_magnetic_daynight_bias_v2_plan.json"
NIGHT_V2_PLAN_MD_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_magnetic_daynight_bias_v2_plan.md"
NIGHT_V2_BUDGET_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_magnetic_daynight_bias_v2_budget.json"
NIGHT_V2_ISR_OUTPUT = ROOT / "isr_validation_outputs" / "m00-seed42-magnetic-daynight-bias-v2-isr"
NIGHT_V2_OLD_WEIGHT = 0.01226901322277942

IMPROVEMENT_MIN = 0.001
SOURCE_RMSE_MAX_DEGRADATION = 0.01
CCC_MAX_DEGRADATION = 0.005
BIAS_MAX_DEGRADATION = 0.02
MIN_CELL_PROFILES = 30
THRESHOLDS = {
    "improvement_min": IMPROVEMENT_MIN,
    "source_rmse_max_degradation": SOURCE_RMSE_MAX_DEGRADATION,
    "ccc_max_degradation": CCC_MAX_DEGRADATION,
    "cell_bias_max_degradation_dex": BIAS_MAX_DEGRADATION,
    "min_cell_profiles": MIN_CELL_PROFILES,
}
PLAN_PATH = ROOT / "checkpoints_fsia" / "m00_low_budget_plan.json"
PLAN_MD_PATH = ROOT / "checkpoints_fsia" / "m00_low_budget_plan.md"
EXPLICIT_PLAN_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_explicit_condition_plan.json"
EXPLICIT_PLAN_MD_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_explicit_condition_plan.md"
SPARSE_PLAN_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_coarse_magnetic_bias_plan.json"
SPARSE_PLAN_MD_PATH = ROOT / "checkpoints_fsia" / "m00_seed42_coarse_magnetic_bias_plan.md"
ISR_CACHE = ROOT / "isr_validation_outputs" / "traditional-static-etkf-raw-iri-full-isr" / "isr_etkf_cache.npz"
ISR_COMPARISON_CACHE = ROOT / "isr_validation_outputs" / "m2w-background-stratum-bias-a1s-m00-isr" / "isr_m00_cache.npz"
ISR_OUTPUT = ROOT / "isr_validation_outputs" / "m2w-background-a-plus-c-isr"


def _load_history(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "fsia_training_history.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    history = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in history if isinstance(row, dict) and row.get("stage") == "background"]
    if not rows:
        raise ValueError(f"no Background history in {run_dir}")
    return rows


def _historical_row() -> dict:
    rows = _load_history(HISTORICAL_RUN)
    matches = [row for row in rows if int(row.get("epoch", -1)) == 3]
    if len(matches) != 1:
        raise ValueError("historical A1-S epoch 3 is missing or duplicated")
    return matches[0]


def _scalar(row: dict) -> dict:
    try:
        fy_rmse = float(row["fy_profile_rmse"])
        cosmic_rmse = float(row["cosmic_profile_rmse"])
        fy_ccc = float(row.get("fy_ccc", row["ccc"]))
        cosmic_ccc = float(row.get("cosmic_ccc", row["ccc"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing M00 Development scalar") from exc
    values = {
        "epoch": int(row["epoch"]),
        "fy_profile_rmse": fy_rmse,
        "cosmic_profile_rmse": cosmic_rmse,
        "profile_rmse": (fy_rmse + cosmic_rmse) / 2.0,
        "fy_ccc": fy_ccc,
        "cosmic_ccc": cosmic_ccc,
        "mean_ccc": (fy_ccc + cosmic_ccc) / 2.0,
        "val_score": float(row.get("val_score", row.get("score", float("nan")))),
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError(f"non-finite Development scalar at epoch {values['epoch']}")
    return values


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _selected(run_dir: Path, *, allow_ineligible: bool = False) -> tuple[dict, dict, dict]:
    rows = _load_history(run_dir)
    night_rows = [row for row in rows if row.get('checkpoint_selection', {}).get(
        'semantics') == 'm00_night_generalization_v1']
    if night_rows:
        eligible = [row for row in night_rows
                    if row.get('checkpoint_selection', {}).get('eligible')]
        if not eligible:
            if not allow_ineligible:
                raise ValueError(f"no eligible night-generalization epoch in {run_dir}")
            row = max(night_rows, key=lambda item: int(item['epoch']))
            audit = row['checkpoint_selection']
            return _scalar(row), row, {
                'eligible': False,
                'reason': 'no_eligible_night_generalization_epoch',
                'selected_row': row,
                'selected_audit': audit,
                'entries': [{'row': item, 'eligible': bool(
                    item.get('checkpoint_selection', {}).get('eligible'))}
                            for item in night_rows],
            }
        row = max(eligible, key=lambda item: tuple(
            item['checkpoint_selection']['selection_key']))
        audit = row['checkpoint_selection']
        return _scalar(row), row, {
            'eligible': True, 'selected_row': row, 'selected_audit': audit,
            'entries': [{'row': item, 'eligible': bool(
                item.get('checkpoint_selection', {}).get('eligible'))}
                        for item in night_rows],
        }
    result = select_m00_background_candidate(rows, _historical_row())
    row = result.get("selected_row")
    audit = result.get("selected_audit")
    if row is None or audit is None:
        raise ValueError(f"no valid M00 epoch in {run_dir}: {result.get('reason')}")
    return _scalar(row), row, result


def _selection_summary(selection: dict) -> dict:
    """Keep reports small; full epoch rows remain in the run history."""
    return {
        "eligible": bool(selection.get("eligible")),
        "reason": selection.get("reason"),
        "entry_count": len(selection.get("entries", [])),
        "eligible_epochs": [
            int(entry["row"]["epoch"])
            for entry in selection.get("entries", []) if entry.get("eligible")
        ],
        "selected_epoch": (
            int(selection["selected_row"]["epoch"])
            if selection.get("selected_row") is not None else None
        ),
        "selected_audit": selection.get("selected_audit"),
    }


def _training_audit(run_dir: Path, candidate_epoch: int) -> dict:
    """Extract compact protocol evidence; full tensors stay in run artifacts."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    resolved = manifest.get("resolved_training", {})
    history = _load_history(run_dir)
    coverage = []
    for row in history:
        audit = row.get("natural_epoch_audit", {})
        coverage.append({
            "epoch": int(row["epoch"]),
            "updates": int(audit.get("total_updates", row.get("processed_batches", 0))),
            "fy_batches": int(audit.get("fy_batches", 0)),
            "cosmic_batches": int(audit.get("cosmic_batches", 0)),
            "fy_profiles": int(audit.get("unique_profiles_consumed", {}).get("FY", 0)),
            "cosmic_profiles": int(audit.get("unique_profiles_consumed", {}).get("COSMIC", 0)),
            "missing_profiles": audit.get("missing_profile_counts", {}),
            "duplicate_profiles": audit.get("duplicate_profile_counts", {}),
        })
    selected = next(row for row in history if int(row["epoch"]) == int(candidate_epoch))
    calibration = resolved.get("background_stratum_bias_gradient_calibration") or {}
    timing_path = Path(run_dir) / "run_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {
        "status": "not_captured_for_preexisting_run"
    }
    return {
        "batch_size": manifest.get("config", {}).get("batch_size"),
        "bias_gradient_target": resolved.get("background_stratum_bias_gradient_target", manifest.get("config", {}).get("background_stratum_bias_gradient_target")),
        "resolved_bias_weight": resolved.get("resolved_background_stratum_bias_weight"),
        "bias_calibration": {
            key: calibration.get(key)
            for key in ("enabled", "requested_batches", "processed_batches", "valid_batches", "median_ratio", "target_ratio", "resolved_weight", "achieved_median_ratio")
            if key in calibration
        },
        "candidate_gradient_ratio": selected.get("gradient_ratio"),
        "candidate_bias_gradient_ratio": selected.get("gradient_ratio_background_stratum_bias"),
        "background_update_steps": resolved.get("background_update_steps"),
        "loader_schedule": resolved.get("background_loader_schedule", {}),
        "epoch_coverage": coverage,
        "timing": timing,
    }


def _relative(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("relative improvement requires finite positive reference")
    return (reference - candidate) / reference


def evaluate_run(run_dir: Path, label: str, baseline_run: Path | None = None) -> dict:
    """Evaluate one completed run using the trainer's selector and fixed gates."""
    run_dir = Path(run_dir)
    candidate, _, selection = _selected(run_dir)
    historical = _scalar(_historical_row())
    audit = selection["selected_audit"]
    source_relative = dict(audit.get("source_rmse_relative_deltas", {}))
    bias_worsening = dict(audit.get("cell_abs_bias_worsening", {}))
    history_improvement = _relative(candidate["profile_rmse"], historical["profile_rmse"])
    source_passed = bool(source_relative) and all(
        float(value) <= SOURCE_RMSE_MAX_DEGRADATION for value in source_relative.values())
    ccc_delta = float(audit.get("mean_ccc_delta", candidate["mean_ccc"] - historical["mean_ccc"]))
    ccc_passed = ccc_delta >= -CCC_MAX_DEGRADATION
    bias_max = max((float(value) for value in bias_worsening.values()), default=0.0)
    bias_passed = bias_max <= BIAS_MAX_DEGRADATION
    criteria = {
        "candidate_eligible": bool(selection.get("eligible")),
        "history_improvement_passed": history_improvement >= IMPROVEMENT_MIN,
        "source_rmse_passed": source_passed,
        "ccc_passed": ccc_passed,
        "cell_bias_passed": bias_passed,
    }
    report = {
        "semantics": "m00_low_budget_experiment_result_v2",
        "label": label,
        "run": str(run_dir.resolve()),
        "date_split": str(DATE_SPLIT.resolve()),
        "locked_test_read": False,
        "candidate": candidate,
        "candidate_checkpoint": str((run_dir / f"epoch_{candidate['epoch']:02d}_background_model.pth").resolve()),
        "candidate_selection": _jsonable(_selection_summary(selection)),
        "historical_epoch3": historical,
        "history_improvement_fraction": history_improvement,
        "source_rmse_relative_deltas": source_relative,
        "mean_ccc_delta": ccc_delta,
        "cell_abs_bias_worsening": bias_worsening,
        "max_cell_bias_worsening": bias_max,
        "criteria": criteria,
        "baseline_run": str(Path(baseline_run).resolve()) if baseline_run else None,
        "training_audit": _training_audit(run_dir, candidate["epoch"]),
    }
    if baseline_run is not None:
        baseline, _, _ = _selected(Path(baseline_run))
        baseline_improvement = _relative(candidate["profile_rmse"], baseline["profile_rmse"])
        report["baseline_candidate"] = baseline
        report["baseline_improvement_fraction"] = baseline_improvement
        report["baseline_improvement_passed"] = baseline_improvement >= IMPROVEMENT_MIN
        criteria["baseline_improvement_passed"] = report["baseline_improvement_passed"]
    else:
        report["baseline_candidate"] = None
        report["baseline_improvement_fraction"] = None
        report["baseline_improvement_passed"] = None
    report["passed"] = bool(all(criteria.values()))
    reasons = []
    if not report["passed"]:
        for key, passed in criteria.items():
            if passed:
                continue
            if key == "candidate_eligible":
                reasons.append("candidate_ineligible")
            elif key.endswith("_passed"):
                reasons.append(f"{key[:-7]}_failed")
            else:
                reasons.append(key)
    report["decision_reasons"] = reasons
    (run_dir / "m00_low_budget_result.json").write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "m00_low_budget_result.md").write_text(_result_markdown(report), encoding="utf-8")
    return report


def evaluate_night_run(run_dir: Path, control_run: Path) -> dict:
    """Persist the direct-control result for the night-generalization run."""
    run_dir, control_run = Path(run_dir), Path(control_run)
    candidate, candidate_row, selection = _selected(run_dir, allow_ineligible=True)
    control, control_row, _ = _selected(control_run)
    audit = selection['selected_audit']
    report = {
        'semantics': 'm00_night_generalization_result_v1',
        'label': 'M00 seed42 magnetic daynight Bias',
        'run': str(run_dir.resolve()), 'control_run': str(control_run.resolve()),
        'locked_test_read': False, 'candidate': candidate,
        'control_candidate': control,
        'candidate_epoch': int(candidate_row['epoch']),
        'control_epoch': int(control_row['epoch']),
        'candidate_checkpoint': str((run_dir / f"epoch_{candidate['epoch']:02d}_background_model.pth").resolve()),
        'selection': _jsonable(selection),
        'selection_audit': _jsonable(audit),
        'training_audit': _training_audit(run_dir, candidate['epoch']),
        'passed': bool(audit.get('eligible', False)),
        'decision_reasons': list(audit.get('reasons', [])),
    }
    (run_dir / 'm00_night_generalization_result.json').write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# M00 seed42 magnetic daynight Bias', '',
        f"Candidate epoch: `{candidate['epoch']}`; control epoch: `{control['epoch']}`.",
        '', '| metric | candidate | direct control |', '|:---|---:|---:|',
        f"| mean profile RMSE | {candidate['profile_rmse']:.6f} | {control['profile_rmse']:.6f} |",
        f"| mean CCC | {candidate['mean_ccc']:.6f} | {control['mean_ccc']:.6f} |", '',
        f"Decision: **{'PASS' if report['passed'] else 'STOP'}**",
    ]
    if report['decision_reasons']:
        lines.append('Reasons: ' + ', '.join(report['decision_reasons']))
    (run_dir / 'm00_night_generalization_result.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    return report


def _result_markdown(report: dict) -> str:
    candidate = report["candidate"]
    historical = report["historical_epoch3"]
    lines = [
        f"# {report['label']}", "",
        f"Candidate epoch: `{candidate['epoch']}`", "",
        "| metric | candidate | historical epoch 3 |", "|:---|---:|---:|",
        f"| mean profile RMSE | {candidate['profile_rmse']:.6f} | {historical['profile_rmse']:.6f} |",
        f"| FY profile RMSE | {candidate['fy_profile_rmse']:.6f} | {historical['fy_profile_rmse']:.6f} |",
        f"| COSMIC profile RMSE | {candidate['cosmic_profile_rmse']:.6f} | {historical['cosmic_profile_rmse']:.6f} |",
        f"| mean CCC | {candidate['mean_ccc']:.6f} | {historical['mean_ccc']:.6f} |", "",
        f"Historical improvement: {report['history_improvement_fraction'] * 100:.3f}%",
        f"Max key-cell absolute Bias worsening: {report['max_cell_bias_worsening']:.4f} dex",
        f"Decision: **{'PASS' if report['passed'] else 'STOP'}**",
    ]
    if report.get("baseline_improvement_fraction") is not None:
        lines.append(f"A-control improvement: {report['baseline_improvement_fraction'] * 100:.3f}%")
    if report["decision_reasons"]:
        lines.append(f"Reasons: {', '.join(report['decision_reasons'])}")
    return "\n".join(lines) + "\n"


def _manifest_config(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"run manifest lacks config: {path}")
    return config


def _partitions(run_dir: Path, config: dict) -> dict:
    path = config.get("date_split_manifest")
    if path and Path(path).is_file():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload.get("partitions"), dict):
            return payload["partitions"]
    summary = json.loads((Path(run_dir) / "training_summary.json").read_text(encoding="utf-8"))
    return summary.get("date_split", {}).get("partitions", {})


def _assert_comparable(candidate_run: Path, control_run: Path, expected_seed: int) -> dict:
    """Check immutable initialization/data semantics before reusing a control."""
    candidate_config = _manifest_config(candidate_run)
    control_config = _manifest_config(control_run)
    keys = (
        "model_domain_semantics", "background_training_semantics",
        "background_sampling_semantics", "background_fixed_updates_per_epoch",
        "background_only", "development_days", "locked_test_days",
        "background_epochs", "analysis_epochs", "use_date_blocked_split",
        "date_split_semantics", "background_profile_schedule_seed_fy",
        "background_profile_schedule_seed_cosmic",
    )
    mismatches = {
        key: {"candidate": candidate_config.get(key), "control": control_config.get(key)}
        for key in keys if candidate_config.get(key) != control_config.get(key)
    }
    if int(control_config.get("seed", -1)) != int(expected_seed):
        mismatches["control_seed"] = {"expected": expected_seed, "actual": control_config.get("seed")}
    if int(candidate_config.get("seed", -1)) != int(expected_seed):
        mismatches["candidate_seed"] = {"expected": expected_seed, "actual": candidate_config.get("seed")}
    candidate_partitions = _partitions(candidate_run, candidate_config)
    control_partitions = _partitions(control_run, control_config)
    if candidate_partitions != control_partitions:
        mismatches["date_partitions"] = {"candidate": candidate_partitions, "control": control_partitions}
    return {"comparable": not mismatches, "mismatches": mismatches, "seed": int(expected_seed)}


def _assert_full_config_comparable(
        candidate_run: Path, control_run: Path, expected_seed: int,
        allowed_config_keys: set[str]) -> dict:
    """Compare all effective config values, allowing only declared deltas."""
    candidate_config = _manifest_config(candidate_run)
    control_config = _manifest_config(control_run)
    mismatches = _config_mismatches(
        candidate_config, control_config, allowed_config_keys)
    if int(candidate_config.get("seed", -1)) != int(expected_seed):
        mismatches["candidate_seed"] = {
            "expected": expected_seed, "actual": candidate_config.get("seed")}
    if int(control_config.get("seed", -1)) != int(expected_seed):
        mismatches["control_seed"] = {
            "expected": expected_seed, "actual": control_config.get("seed")}
    candidate_partitions = _partitions(candidate_run, candidate_config)
    control_partitions = _partitions(control_run, control_config)
    if candidate_partitions != control_partitions:
        mismatches["date_partitions"] = {
            "candidate": candidate_partitions, "control": control_partitions}
    return {
        "comparable": not mismatches,
        "mismatches": mismatches,
        "seed": int(expected_seed),
        "allowed_config_keys": sorted(allowed_config_keys),
    }


def _config_mismatches(candidate_config: dict, control_config: dict,
                       allowed_config_keys: set[str]) -> dict:
    """Return undeclared effective config differences for preflight checks."""
    ignored = {
        # Output and immutable historical-reference paths are run-local.
        "save_dir", "historical_a1s_reference_checkpoint",
    }
    mismatches = {}
    keys = set(candidate_config) | set(control_config)
    for key in sorted(keys - ignored - set(allowed_config_keys)):
        candidate_value = candidate_config.get(key)
        control_value = control_config.get(key)
        if _jsonable(candidate_value) != _jsonable(control_value):
            mismatches[key] = {
                "candidate": candidate_value, "control": control_value,
            }
    return mismatches


def _is_complete(run_dir: Path, *, require_timing: bool = False) -> bool:
    """Require a complete five-epoch Background artifact, not just 3 files."""
    run_dir = Path(run_dir)
    required = ("run_manifest.json", "fsia_training_history.json",
                "training_summary.json")
    if not all((run_dir / name).is_file() for name in required):
        return False
    if require_timing:
        try:
            timing = json.loads((run_dir / "run_timing.json").read_text(
                encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if timing.get("status") != "completed":
            return False
    try:
        rows = _load_history(run_dir)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    epochs = [int(row.get("epoch", -1)) for row in rows]
    if sorted(epochs) != [1, 2, 3, 4, 5] or len(set(epochs)) != 5:
        return False
    try:
        summary = json.loads((run_dir / "training_summary.json").read_text(
            encoding="utf-8"))
        if summary.get("completed_stage") != "background":
            return False
        if summary.get("background_epochs") not in (None, 5):
            return False
        if summary.get("background_candidate_checkpoint"):
            if not Path(summary["background_candidate_checkpoint"]).is_file():
                return False
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    for epoch in epochs:
        if not (run_dir / f"epoch_{epoch:02d}_background_model.pth").is_file():
            return False
    for row in rows:
        audit = row.get("natural_epoch_audit", {})
        try:
            required_sources = {"FY", "COSMIC"}
            for key in (
                    "source_profile_totals", "unique_profiles_consumed",
                    "missing_profile_counts", "duplicate_profile_counts",
                    "profile_visit_counts"):
                if not isinstance(audit.get(key), dict):
                    return False
            if set(audit["source_profile_totals"]) != required_sources:
                return False
            if set(audit["unique_profiles_consumed"]) != required_sources:
                return False
            if set(audit["missing_profile_counts"]) != required_sources:
                return False
            if set(audit["duplicate_profile_counts"]) != required_sources:
                return False
            for source in required_sources:
                total = int(audit["source_profile_totals"][source])
                unique = int(audit["unique_profiles_consumed"][source])
                if total < 1 or unique != total:
                    return False
                if int(audit["missing_profile_counts"][source]) != 0:
                    return False
                if int(audit["duplicate_profile_counts"][source]) != 0:
                    return False
                visits = audit["profile_visit_counts"].get(source)
                if not isinstance(visits, dict) or len(visits) != total:
                    return False
                if any(int(value) != 1 for value in visits.values()):
                    return False
            if int(audit.get("total_updates", 0)) != int(
                    audit.get("fy_batches", 0)) + int(
                    audit.get("cosmic_batches", 0)):
                return False
            if int(audit.get("total_updates", 0)) < 1:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _write_explicit_plan(payload: dict) -> dict:
    """Persist the single-seed plan without touching the legacy A+C report."""
    payload = _jsonable(payload)
    EXPLICIT_PLAN_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    report = payload.get("development_result") or {}
    candidate = report.get("candidate", {})
    lines = [
        "# M00 seed42 explicit-condition plan", "",
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Technical status: **{payload.get('technical_status', 'unknown')}**",
        "",
        "Configuration: seed42; five natural Background epochs; explicit decoder "
        "geo/local-time conditioning; locked-test read false.",
        f"Training runs used: `{payload.get('training_runs', 0)}`.",
    ]
    if candidate:
        lines.extend([
            "", "| item | value |", "|:---|---:|",
            f"| best epoch | {candidate.get('epoch')} |",
            f"| candidate profile RMSE | {candidate.get('profile_rmse', float('nan')):.9f} |",
            f"| Development improvement | {report.get('baseline_improvement_fraction', float('nan')) * 100:.4f}% |",
            f"| Development gate | {'PASS' if report.get('passed') else 'STOP'} |",
        ])
    if payload.get("isr"):
        lines.append(
            f"ISR: **{'PASS' if payload['isr'].get('passed') else 'FAIL'}** "
            f"(`{payload['isr'].get('report')}`).")
    final = payload.get("final_decision")
    if final:
        lines.append(f"Final adoption: **{final.get('adopted')}**.")
    EXPLICIT_PLAN_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_seed42_single() -> dict:
    """Run and qualify the one authorized seed42 explicit-condition experiment."""
    control_run = AC_RUN
    run_dir = EXPLICIT_CONDITION_RUN
    allowed = {"background_decoder_conditioning"}
    control_config = _manifest_config(control_run)
    if int(control_config.get("seed", -1)) != 42:
        raise ValueError("Development control must be seed42")

    config = None
    if run_dir.exists() and _is_complete(run_dir, require_timing=True):
        comparability = _assert_full_config_comparable(
            run_dir, control_run, 42, allowed)
        if not comparability["comparable"]:
            raise ValueError(
                f"incomparable completed explicit-condition run: "
                f"{comparability['mismatches']}")
        # The completed artifact represents the one authorized training run;
        # reuse must not erase that provenance from the final plan.
        training_runs = 1
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite incomplete output: {run_dir}")
        config = build_config(
            output_run=run_dir,
            reference_run=control_run,
            seed=42,
            date_split_manifest=DATE_SPLIT,
            overrides={
                "batch_size": 1024,
                "background_stratum_bias_gradient_target": 0.025,
                "background_decoder_conditioning": "explicit_geo_lt_v1",
            },
        )
        mismatches = _config_mismatches(config, control_config, allowed)
        if int(config.get("seed", -1)) != 42:
            mismatches["candidate_seed"] = {
                "expected": 42, "actual": config.get("seed")}
        if _jsonable(config.get("date_split_manifest")) != _jsonable(
                control_config.get("date_split_manifest")):
            mismatches["date_split_manifest"] = {
                "candidate": config.get("date_split_manifest"),
                "control": control_config.get("date_split_manifest"),
            }
        if mismatches:
            raise ValueError(f"candidate preflight is not comparable: {mismatches}")
        _write_explicit_plan({
            "semantics": "m00_seed42_explicit_condition_plan_v1",
            "status": "preflight_passed",
            "technical_status": "pending_training",
            "training_runs": 0,
            "control_run": str(control_run.resolve()),
            "candidate_run": str(run_dir.resolve()),
            "allowed_config_keys": sorted(allowed),
            "config": config,
            "thresholds": THRESHOLDS,
            "locked_test_read": False,
        })
        try:
            run_config(config, run_dir,
                       experiment="M00 seed42 explicit decoder conditioning")
        except Exception as exc:
            _write_explicit_plan({
                "semantics": "m00_seed42_explicit_condition_plan_v1",
                "status": "technical_failure",
                "technical_status": "failed",
                "training_runs": 1,
                "candidate_run": str(run_dir.resolve()),
                "error": repr(exc),
                "locked_test_read": False,
            })
            raise
        comparability = _assert_full_config_comparable(
            run_dir, control_run, 42, allowed)
        (run_dir / "comparability.json").write_text(
            json.dumps(comparability, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        if not comparability["comparable"]:
            raise ValueError(
                f"new explicit-condition run is not comparable: "
                f"{comparability['mismatches']}")
        training_runs = 1

    if not _is_complete(run_dir, require_timing=True):
        raise RuntimeError("explicit-condition run is not a complete five-epoch artifact")
    report = evaluate_run(
        run_dir, "M00 seed42 explicit decoder conditioning",
        baseline_run=control_run)
    payload = {
        "semantics": "m00_seed42_explicit_condition_plan_v1",
        "status": "seed42_quality_failed" if not report["passed"] else "development_passed",
        "technical_status": "complete",
        "training_runs": training_runs,
        "control_run": str(control_run.resolve()),
        "candidate_run": str(run_dir.resolve()),
        "development_result": report,
        "thresholds": THRESHOLDS,
        "locked_test_read": False,
        "final_decision": {
            "adopted": "historical_a1s_epoch3",
            "locked_test_read": False,
        },
    }
    if not report["passed"]:
        return _write_explicit_plan(payload)

    try:
        isr = run_isr(run_dir, EXPLICIT_CONDITION_ISR_OUTPUT)
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "technical_status": "isr_failed_to_complete",
            "isr_error": repr(exc),
        })
        return _write_explicit_plan(payload)
    payload["isr"] = isr
    payload["status"] = (
        "m00_adopted_candidate" if isr["passed"]
        else "isr_failed_historical_retained")
    payload["final_decision"] = {
        "adopted": "m00_candidate" if isr["passed"] else "historical_a1s_epoch3",
        "candidate": str(run_dir.resolve()),
        "checkpoint": isr["checkpoint"],
        "locked_test_read": False,
    }
    return _write_explicit_plan(payload)


def _write_sparse_plan(payload: dict) -> dict:
    payload = _jsonable(payload)
    SPARSE_PLAN_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    lines = [
        "# M00 seed42 coarse magnetic Bias plan", "",
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Technical status: **{payload.get('technical_status', 'unknown')}**",
        f"Training runs used: `{payload.get('training_runs', 0)}`.",
        "", "Candidate: explicit geo/local-time decoder + coarse magnetic Bias view.",
        "Diagnostic uses train/development only; locked-test read false.",
    ]
    diagnostic = payload.get("diagnostic", {})
    if diagnostic:
        lines.append(
            f"Diagnostic gate: **{diagnostic.get('status', 'unknown')}** "
            f"({diagnostic.get('reason') or 'supported'}).")
    report = payload.get("development_result")
    if report:
        lines.extend([
            "", f"Best epoch: `{report['candidate']['epoch']}`",
            f"Development improvement vs explicit condition: "
            f"`{report.get('baseline_improvement_fraction', float('nan')) * 100:.4f}%`",
            f"Development gate: **{'PASS' if report.get('passed') else 'STOP'}**",
        ])
    if payload.get("isr"):
        lines.append(
            f"ISR: **{'PASS' if payload['isr'].get('passed') else 'FAIL'}** "
            f"(`{payload['isr'].get('report')}`).")
    final = payload.get("final_decision")
    if final:
        lines.append(f"Final adoption: **{final.get('adopted')}**.")
    SPARSE_PLAN_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _write_night_plan(payload: dict) -> dict:
    payload = _jsonable(payload)
    NIGHT_BIAS_PLAN_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# M00 seed42 magnetic daynight Bias plan', '',
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Technical status: **{payload.get('technical_status', 'unknown')}**",
        f"Training runs used: `{payload.get('training_runs', 0)}`.",
        '', 'Scope: 200–500 km night generalization; locked-test read false.',
    ]
    diagnostic = payload.get('diagnostic', {})
    if diagnostic:
        lines.append(
            f"Diagnostic gate: **{diagnostic.get('status', 'unknown')}** "
            f"({diagnostic.get('reason') or 'fixed gate passed'}).")
    report = payload.get('development_result')
    if report:
        lines.extend([
            '', f"Candidate epoch: `{report.get('candidate_epoch')}`",
            f"Development gate: **{'PASS' if report.get('passed') else 'STOP'}**",
        ])
        if report.get('decision_reasons'):
            lines.append('Reasons: ' + ', '.join(report['decision_reasons']))
    if payload.get('isr'):
        lines.append(
            f"ISR: **{'PASS' if payload['isr'].get('passed') else 'FAIL'}**.")
    if payload.get('final_decision'):
        lines.append(f"Final adoption: **{payload['final_decision'].get('adopted')}**.")
    NIGHT_BIAS_PLAN_MD_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return payload


def _write_night_v2_plan(payload: dict) -> dict:
    payload = _jsonable(payload)
    NIGHT_V2_PLAN_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    result = payload.get('development_result') or {}
    lines = [
        '# M00 night generalization v2', '',
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Protocol compliant: **{bool(payload.get('protocol_compliant', False))}**",
        f"Training launches: `{payload.get('budget', {}).get('training_launches', 0)}`; "
        f"candidate ISR launches: `{payload.get('budget', {}).get('candidate_isr_launches', 0)}`.",
    ]
    if result:
        lines += ['', f"Selected epoch: `{result.get('selected_epoch')}`",
                  f"Diagnostic best epoch: `{result.get('diagnostic_best_epoch')}`"]
    lines += ['', 'Formal M00 remains historical A1S epoch 3.']
    NIGHT_V2_PLAN_MD_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return payload


def _night_v2_budget(event=None) -> dict:
    if NIGHT_V2_BUDGET_PATH.is_file():
        payload = json.loads(NIGHT_V2_BUDGET_PATH.read_text(encoding='utf-8'))
    else:
        payload = {'semantics': 'm00_night_v2_budget_v1',
                   'training_launches': 0, 'candidate_isr_launches': 0,
                   'events': []}
    payload.setdefault('control_replay_creations', 0)
    payload.setdefault('completed_training_updates', 0)
    payload.setdefault('completed_training_elapsed_seconds', 0.0)
    execution_path = NIGHT_V2_RUN / 'night_v2_execution.jsonl'
    execution = []
    if execution_path.is_file():
        execution = [json.loads(line) for line in execution_path.read_text(encoding='utf-8').splitlines()
                     if line.strip()]
    payload['actual_initializations'] = sum(row['event'] == 'model_initialized' for row in execution)
    payload['actual_optimizer_steps'] = sum(row['event'] == 'optimizer_step' for row in execution)
    if event:
        kind, detail = event
        if kind == 'training_launched':
            payload['training_launches'] += 1
        elif kind == 'candidate_isr_launched':
            payload['candidate_isr_launches'] += 1
        elif kind == 'control_replay_created':
            payload['control_replay_creations'] += 1
        elif kind == 'training_completed':
            payload['completed_training_updates'] = int(detail['updates'])
            payload['completed_training_elapsed_seconds'] = float(
                detail['elapsed_seconds'])
        payload['events'].append({'event': kind, 'utc': dt.datetime.now(
            dt.timezone.utc).isoformat(), **detail})
        NIGHT_V2_BUDGET_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return payload


def _night_v2_weight_identity(run_dir: Path) -> dict:
    manifest = json.loads((Path(run_dir) / 'run_manifest.json').read_text(encoding='utf-8'))
    config, resolved = manifest.get('config', {}), manifest.get('resolved_training', {})
    old = resolved.get('resolved_background_stratum_bias_weight',
                       config.get('resolved_background_stratum_bias_weight'))
    new = resolved.get('resolved_background_magnetic_daynight_weight',
                       config.get('resolved_background_magnetic_daynight_weight'))
    valid = (old is not None and new is not None and
             abs(float(old) - NIGHT_V2_OLD_WEIGHT) <= 1e-12 and
             math.isfinite(float(new)) and float(new) > 0)
    return {'valid': valid, 'old_weight': old, 'new_weight': new,
            'old_expected': NIGHT_V2_OLD_WEIGHT}


def _night_v2_complete(run_dir: Path) -> bool:
    if not _is_complete(run_dir, require_timing=True):
        return False
    for row in _load_history(run_dir):
        audit = row.get('natural_epoch_audit', {})
        if audit.get('unique_profiles_consumed') != {'FY': 44625, 'COSMIC': 56099}:
            return False
        if int(audit.get('total_updates', -1)) != 788:
            return False
    return _night_v2_weight_identity(run_dir)['valid']


def evaluate_night_v2(run_dir: Path, diagnostic: dict) -> dict:
    """Recompute every epoch with the shared frozen selector."""
    from inr_modules.mdia.train_fsia import select_night_generalization_v2
    replay = json.loads(Path(diagnostic['replay']['path']).read_text(
        encoding='utf-8'))
    selection = select_night_generalization_v2(
        _load_history(run_dir), replay['metrics'],
        diagnostic['supported_cells'])
    selected = selection.get('selected_row')
    diagnostic_row = selection.get('diagnostic_row')
    report = {
        'semantics': 'm00_night_generalization_result_v2',
        'run': str(Path(run_dir).resolve()),
        'selected_epoch': int(selected['epoch']) if selected else None,
        'diagnostic_best_epoch': int(diagnostic_row['epoch']) if diagnostic_row else None,
        'selection': _jsonable(_selection_summary(selection)),
        'epoch_audits': _jsonable(selection.get('epoch_audits', [])),
        'weight_identity': _night_v2_weight_identity(run_dir),
        'passed': bool(selection.get('eligible') and selected),
    }
    report['decision_reasons'] = [] if report['passed'] else [
        selection.get('reason') or 'no_eligible_epoch']
    (Path(run_dir) / 'm00_night_generalization_result_v2.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def run_night_v2_isr(run_dir: Path, development: dict) -> dict:
    """Launch or reuse the sole v2 candidate inference, then apply its own gate."""
    from isr_evaluation.background_m00_eval import night_v2_acceptance
    from inr_modules.mdia.train_fsia import select_night_generalization_v2
    config = _manifest_config(run_dir)
    replay = json.loads(Path(config['background_selection_baseline']).read_text(encoding='utf-8'))
    recalculated = select_night_generalization_v2(
        _load_history(run_dir), replay['metrics'], config['background_night_v2_supported_cells'])
    if (not recalculated['eligible'] or recalculated['selected_epoch'] != development.get('selected_epoch')
            or not _night_v2_weight_identity(run_dir)['valid']):
        raise ValueError('night-v2 ISR shared selection or weight recheck failed')
    if not development.get('passed') or not development['weight_identity']['valid']:
        raise ValueError('night-v2 ISR requires a recomputed passing Development result and weights')
    epoch = development['selected_epoch']
    checkpoint = Path(run_dir) / f'epoch_{epoch:02d}_background_model.pth'
    best = Path(run_dir) / 'best_background_model.pth'
    if not checkpoint.is_file() or not best.is_file():
        raise FileNotFoundError('selected night-v2 checkpoint or best alias is missing')
    selected_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    best_state = torch.load(best, map_location='cpu', weights_only=True)
    if set(selected_state) != set(best_state) or any(
            not torch.equal(value, best_state[key])
            for key, value in selected_state.items()):
        raise ValueError('best Background alias does not match selected night-v2 epoch')
    report_path = NIGHT_V2_ISR_OUTPUT / 'isr_m00_report.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding='utf-8'))
    else:
        if NIGHT_V2_ISR_OUTPUT.exists() and any(NIGHT_V2_ISR_OUTPUT.iterdir()):
            raise FileExistsError('refusing to overwrite incomplete night-v2 ISR output')
        budget = _night_v2_budget()
        if budget['candidate_isr_launches'] >= 1:
            raise RuntimeError('night-v2 candidate ISR launch budget is exhausted')
        _night_v2_budget(('candidate_isr_launched', {'checkpoint': str(checkpoint.resolve())}))
        subprocess.run([
            sys.executable, '-m', 'isr_evaluation.background_m00_eval',
            '--checkpoint', str(best), '--cache', str(ISR_CACHE),
            '--comparison-cache', str(ISR_COMPARISON_CACHE),
            '--output-dir', str(NIGHT_V2_ISR_OUTPUT),
            '--candidate-name', 'M00_NIGHT_V2', '--reference-name', 'M00_A1S'],
            cwd=ROOT, check=True)
        report = json.loads(report_path.read_text(encoding='utf-8'))
    expected_identity = {
        'checkpoint': best.resolve(),
        'canonical_cache': ISR_CACHE.resolve(),
        'comparison_cache': ISR_COMPARISON_CACHE.resolve(),
    }
    for key, expected in expected_identity.items():
        value = report.get(key)
        if not value or Path(value).resolve() != expected:
            raise ValueError(f'night-v2 ISR {key} identity mismatch')
    if (report.get('candidate_name') != 'M00_NIGHT_V2'
            or report.get('reference_name') != 'M00_A1S'
            or report.get('diagnostic_only') is not False
            or int(report.get('row_count', -1)) != 206230):
        raise ValueError('night-v2 ISR report protocol identity mismatch')
    acceptance = night_v2_acceptance(report.get('stratified_metrics', []))
    from run_m00_condition_diagnostic import cached_night_v2_isr_diagnostic
    cache_report = cached_night_v2_isr_diagnostic(
        NIGHT_V2_ISR_OUTPUT / 'night_v2_all_cached_comparisons.json',
        NIGHT_V2_ISR_OUTPUT / 'isr_m00_cache.npz')
    return {'report': str(report_path.resolve()), 'checkpoint': str(checkpoint.resolve()),
            'passed': acceptance['passed'], 'acceptance': acceptance,
            'all_cached_comparisons': str(NIGHT_V2_ISR_OUTPUT / 'night_v2_all_cached_comparisons.json'),
            'common_valid_count': cache_report['common_valid_count']}


def run_seed42_magnetic_daynight_bias_v2(*, preflight_only=False,
                                         evaluate_only=False) -> dict:
    """Execute the frozen one-launch v2 protocol without hidden retries."""
    from run_m00_condition_diagnostic import prepare_night_v2_diagnostic
    replay_path = NIGHT_V2_DIAGNOSTIC_DIR / 'background_development_replay.json'
    replay_existed = replay_path.is_file()
    diagnostic = prepare_night_v2_diagnostic(
        EXPLICIT_CONDITION_RUN, HISTORICAL_RUN, NIGHT_V2_DIAGNOSTIC_DIR,
        allow_replay=not evaluate_only)
    if not replay_existed and replay_path.is_file():
        _night_v2_budget(('control_replay_created', {
            'replay': str(replay_path.resolve())}))
    base = {'semantics': 'm00_seed42_magnetic_daynight_bias_plan_v2',
            'candidate_run': str(NIGHT_V2_RUN.resolve()), 'diagnostic': diagnostic,
            'protocol_compliant': True, 'formal_m00': 'historical_a1s_epoch3'}
    from run_m00_condition_diagnostic import cached_night_v2_isr_diagnostic
    cached_path = ROOT / 'checkpoints_fsia' / 'm00-night-v2-cached-isr-diagnostic.json'
    cached_night_v2_isr_diagnostic(cached_path)
    base['cached_isr_diagnostic'] = str(cached_path)
    if not diagnostic.get('can_train'):
        return _write_night_v2_plan({**base,
            'status': ('technical_failure' if diagnostic.get('status') == 'technical_failure'
                       else 'diagnostic_not_supported'),
            'budget': _night_v2_budget()})

    allowed = {'background_bias_grouping', 'background_checkpoint_selection',
               'background_night_v2_supported_cells', 'background_selection_baseline',
               'background_selection_baseline_checkpoint',
               'background_magnetic_daynight_gradient_target',
               'background_magnetic_daynight_calibration_batches',
               'resolved_background_magnetic_daynight_weight',
               'resolved_background_stratum_bias_weight',
               'night_v2_frozen_input_sha256', 'input_data_sha256',
               'background_bias_magnetic_min_profiles',
               'background_bias_magnetic_min_batch_profiles',
               'background_bias_magnetic_weight',
               'skip_hash_verification', 'background_stratum_bias_gradient_calibration'}
    control_config = _manifest_config(EXPLICIT_CONDITION_RUN)
    control_manifest = json.loads((EXPLICIT_CONDITION_RUN / 'run_manifest.json').read_text(encoding='utf-8'))
    old_resolved = control_manifest.get('resolved_training', {})
    old_weight = old_resolved.get('resolved_background_stratum_bias_weight')
    if old_weight is None or abs(float(old_weight) - NIGHT_V2_OLD_WEIGHT) > 1e-15:
        raise ValueError('frozen control legacy weight identity mismatch')
    reusable = NIGHT_V2_RUN.exists() and _night_v2_complete(NIGHT_V2_RUN)
    config = _manifest_config(NIGHT_V2_RUN) if reusable else build_config(
        output_run=NIGHT_V2_RUN, reference_run=EXPLICIT_CONDITION_RUN, seed=42,
        date_split_manifest=DATE_SPLIT, overrides={
            'background_bias_grouping': 'height_daynight_plus_magnetic_daynight_v2',
            'background_checkpoint_selection': 'm00_night_generalization_v2',
            'background_decoder_conditioning': control_config['background_decoder_conditioning'],
            'skip_hash_verification': False,
            'background_night_v2_supported_cells': diagnostic['supported_cells'],
            'background_selection_baseline': diagnostic['replay']['path'],
            'background_selection_baseline_checkpoint': diagnostic['control']['checkpoint'],
            'night_v2_frozen_input_sha256': diagnostic['control'][
                'current_input_sha256'],
            'resolved_background_stratum_bias_weight': float(old_weight),
            'background_stratum_bias_gradient_calibration': old_resolved.get(
                'background_stratum_bias_gradient_calibration'),
            'background_magnetic_daynight_gradient_target': 0.0125,
            'background_magnetic_daynight_calibration_batches': 20})
    mismatches = _config_mismatches(config, control_config, allowed)
    if config.get('skip_hash_verification'):
        mismatches['skip_hash_verification'] = {
            'expected': False, 'actual': config.get('skip_hash_verification')}
    if mismatches:
        raise ValueError(f'night-v2 preflight is not comparable: {mismatches}')
    if preflight_only:
        return _write_night_v2_plan({**base, 'status': 'preflight_passed',
                                     'config': config, 'budget': _night_v2_budget()})
    if reusable:
        pass
    elif evaluate_only:
        raise RuntimeError('evaluate-only requires a complete valid night-v2 artifact')
    else:
        if NIGHT_V2_RUN.exists() and any(NIGHT_V2_RUN.iterdir()):
            raise FileExistsError('refusing to overwrite incomplete night-v2 output')
        budget = _night_v2_budget()
        if budget['training_launches'] >= 1:
            raise RuntimeError('night-v2 training launch budget is exhausted')
        _night_v2_budget(('training_launched', {'run': str(NIGHT_V2_RUN.resolve())}))
        try:
            run_config(config, NIGHT_V2_RUN, experiment='M00 night generalization v2')
        except Exception as exc:
            return _write_night_v2_plan({**base, 'status': 'technical_failure',
                                         'error': repr(exc), 'budget': _night_v2_budget()})
        timing = json.loads((NIGHT_V2_RUN / 'run_timing.json').read_text(
            encoding='utf-8'))
        updates = sum(int(row['natural_epoch_audit']['total_updates'])
                      for row in _load_history(NIGHT_V2_RUN))
        _night_v2_budget(('training_completed', {
            'run': str(NIGHT_V2_RUN.resolve()), 'updates': updates,
            'elapsed_seconds': timing['elapsed_seconds']}))
    if not _night_v2_complete(NIGHT_V2_RUN):
        raise RuntimeError('night-v2 artifact failed five-epoch coverage or weight identity')
    development = evaluate_night_v2(NIGHT_V2_RUN, diagnostic)
    payload = {**base, 'development_result': development,
               'budget': _night_v2_budget(),
               'status': 'dev_not_passed' if not development['passed'] else 'development_passed'}
    if evaluate_only or not development['passed']:
        return _write_night_v2_plan(payload)
    try:
        payload['isr'] = run_night_v2_isr(NIGHT_V2_RUN, development)
        payload['status'] = ('single_seed_screening_passed' if payload['isr']['passed']
                             else 'isr_not_passed')
    except Exception as exc:
        payload.update(status='technical_failure', isr_error=repr(exc))
    payload['budget'] = _night_v2_budget()
    payload['next_step'] = ('second_seed_confirmation_recommended'
                            if payload['status'] == 'single_seed_screening_passed' else None)
    return _write_night_v2_plan(payload)


def run_seed42_magnetic_daynight_bias() -> dict:
    """Run exactly one initialized five-epoch night-generalization candidate."""
    control_run = EXPLICIT_CONDITION_RUN
    from run_m00_condition_diagnostic import diagnose_night_generalization

    diagnostic = diagnose_night_generalization(
        control_run, HISTORICAL_RUN, NIGHT_BIAS_DIAGNOSTIC)
    base_payload = {
        'semantics': 'm00_seed42_magnetic_daynight_bias_plan_v1',
        'control_run': str(control_run.resolve()),
        'historical_reference_run': str(HISTORICAL_RUN.resolve()),
        'candidate_run': str(NIGHT_BIAS_RUN.resolve()),
        'diagnostic': diagnostic,
        'training_budget': {'max_new_initialized_training_runs': 1,
                            'epochs': 5, 'seed': 42},
        'locked_test_read': False,
    }
    if diagnostic.get('status') != 'diagnostic_supported':
        return _write_night_plan({
            **base_payload, 'status': 'diagnostic_not_supported',
            'technical_status': 'complete', 'training_runs': 0,
            'final_decision': {'adopted': 'historical_a1s_epoch3',
                               'locked_test_read': False}})

    allowed = {
        'background_bias_grouping',
        'background_bias_magnetic_min_profiles',
        'background_bias_magnetic_min_batch_profiles',
        'background_bias_magnetic_weight',
        'background_bias_magnetic_support_counts',
        'background_bias_magnetic_support_semantics',
        'background_stratum_bias_gradient_target',
        'background_checkpoint_selection',
        'background_selection_baseline',
        'background_selection_baseline_checkpoint',
        'background_bias_magnetic_daynight_labels',
    }
    control_config = _manifest_config(control_run)
    if int(control_config.get('seed', -1)) != 42:
        raise ValueError('explicit-condition Development control must be seed42')
    run_dir = NIGHT_BIAS_RUN
    if run_dir.exists() and _is_complete(run_dir, require_timing=True):
        comparability = _assert_full_config_comparable(run_dir, control_run, 42, allowed)
        if not comparability['comparable']:
            raise ValueError(f'incomparable completed night run: {comparability["mismatches"]}')
        training_runs = 1
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f'refusing to overwrite incomplete output: {run_dir}')
        config = build_config(
            output_run=run_dir, reference_run=control_run, seed=42,
            date_split_manifest=DATE_SPLIT,
            overrides={
                'background_decoder_conditioning': 'explicit_geo_lt_v1',
                'background_bias_grouping': 'height_daynight_plus_magnetic_daynight_v1',
                'background_bias_magnetic_min_profiles': 30,
                'background_bias_magnetic_min_batch_profiles': 8,
                'background_bias_magnetic_weight': 1.0,
                'background_stratum_bias_gradient_target': 0.0125,
                'background_checkpoint_selection': 'm00_night_generalization_v1',
                'background_bias_magnetic_daynight_labels': [
                    'abs_sinI_lt_0.25_night', 'abs_sinI_lt_0.25_day',
                    'abs_sinI_0.25_0.5_night', 'abs_sinI_0.25_0.5_day',
                    'abs_sinI_ge_0.5_night', 'abs_sinI_ge_0.5_day'],
            })
        control_scalar, control_row, _ = _selected(control_run)
        run_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = run_dir / 'direct_control_selection_baseline.json'
        baseline_path.write_text(json.dumps({
            'semantics': 'm00_direct_control_development_replay_v1',
            'score': float(control_row.get('val_score', control_scalar['profile_rmse'])),
            'metrics': control_row,
            'checkpoint': str((control_run / f"epoch_{control_row['epoch']:02d}_background_model.pth").resolve()),
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        config['background_selection_baseline'] = str(baseline_path.resolve())
        config['background_selection_baseline_checkpoint'] = str(
            (control_run / f"epoch_{control_row['epoch']:02d}_background_model.pth").resolve())
        mismatches = _config_mismatches(config, control_config, allowed)
        if int(config.get('seed', -1)) != 42:
            mismatches['candidate_seed'] = {'expected': 42, 'actual': config.get('seed')}
        if mismatches:
            raise ValueError(f'night candidate preflight is not comparable: {mismatches}')
        _write_night_plan({**base_payload, 'status': 'preflight_passed',
                           'technical_status': 'pending_training',
                           'training_runs': 0, 'allowed_config_keys': sorted(allowed),
                           'config': config})
        try:
            run_config(config, run_dir,
                       experiment='M00 seed42 magnetic daynight Bias')
        except Exception as exc:
            return _write_night_plan({**base_payload,
                'status': 'technical_failure', 'technical_status': 'failed',
                'training_runs': 1, 'error': repr(exc)})
        comparability = _assert_full_config_comparable(run_dir, control_run, 42, allowed)
        (run_dir / 'comparability.json').write_text(
            json.dumps(comparability, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if not comparability['comparable']:
            raise ValueError(f'new night run is not comparable: {comparability["mismatches"]}')
        training_runs = 1

    if not _is_complete(run_dir, require_timing=True):
        raise RuntimeError('night candidate is not a complete five-epoch artifact')
    report = evaluate_night_run(run_dir, control_run)
    payload = {**base_payload,
        'status': 'development_passed' if report['passed'] else 'development_quality_failed',
        'technical_status': 'complete', 'training_runs': training_runs,
        'development_result': report,
        'final_decision': {'adopted': 'historical_a1s_epoch3',
                           'candidate': str(run_dir.resolve()),
                           'locked_test_read': False}}
    if not report['passed']:
        return _write_night_plan(payload)
    # Candidate ISR is authorized only after the direct Development gate.
    try:
        isr = run_isr(run_dir, NIGHT_BIAS_RUN / 'isr')
    except Exception as exc:
        return _write_night_plan({**payload, 'status': 'technical_failure',
                                  'technical_status': 'isr_failed_to_complete',
                                  'isr_error': repr(exc)})
    payload['isr'] = isr
    payload['status'] = 'm00_candidate_isr_passed' if isr['passed'] else 'isr_failed_historical_retained'
    payload['final_decision']['adopted'] = 'm00_candidate' if isr['passed'] else 'historical_a1s_epoch3'
    return _write_night_plan(payload)


def run_seed42_coarse_bias() -> dict:
    """Run the one authorized sparse-coverage coarse-Bias candidate."""
    control_run = EXPLICIT_CONDITION_RUN
    diagnostic_control = HISTORICAL_RUN
    from run_m00_condition_diagnostic import diagnose_sparse_coverage

    diagnostic_path = EXPLICIT_CONDITION_RUN / "m00_sparse_coverage_diagnostic.json"
    diagnostic = diagnose_sparse_coverage(
        EXPLICIT_CONDITION_RUN, diagnostic_control, diagnostic_path)
    base_payload = {
        "semantics": "m00_seed42_coarse_magnetic_bias_plan_v1",
        "control_run": str(control_run.resolve()),
        "historical_reference_run": str(HISTORICAL_RUN.resolve()),
        "candidate_run": str(SPARSE_BIAS_RUN.resolve()),
        "diagnostic": diagnostic,
        "thresholds": THRESHOLDS,
        "locked_test_read": False,
    }
    if diagnostic.get("status") != "diagnostic_supported":
        base_payload.update({
            "status": "diagnostic_not_supported",
            "technical_status": "complete",
            "training_runs": 0,
            "final_decision": {
                "adopted": "historical_a1s_epoch3",
                "locked_test_read": False,
            },
        })
        return _write_sparse_plan(base_payload)

    allowed = {
        "background_bias_grouping",
        "background_bias_magnetic_min_profiles",
        "background_bias_magnetic_min_batch_profiles",
        "background_bias_magnetic_weight",
        "background_bias_magnetic_support_counts",
        "background_bias_magnetic_support_semantics",
    }
    control_config = _manifest_config(control_run)
    if int(control_config.get("seed", -1)) != 42:
        raise ValueError("explicit-condition Development control must be seed42")
    run_dir = SPARSE_BIAS_RUN
    config = None
    if run_dir.exists() and _is_complete(run_dir, require_timing=True):
        comparability = _assert_full_config_comparable(
            run_dir, control_run, 42, allowed)
        if not comparability["comparable"]:
            raise ValueError(
                f"incomparable completed coarse-Bias run: "
                f"{comparability['mismatches']}")
        training_runs = 1
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite incomplete output: {run_dir}")
        config = build_config(
            output_run=run_dir,
            reference_run=control_run,
            seed=42,
            date_split_manifest=DATE_SPLIT,
            overrides={
                "background_decoder_conditioning": "explicit_geo_lt_v1",
                "background_bias_grouping": "height_daynight_plus_magnetic_v1",
                "background_bias_magnetic_min_profiles": 30,
                "background_bias_magnetic_min_batch_profiles": 8,
                "background_bias_magnetic_weight": 0.5,
                "background_stratum_bias_gradient_target": 0.025,
            },
        )
        mismatches = _config_mismatches(config, control_config, allowed)
        if int(config.get("seed", -1)) != 42:
            mismatches["candidate_seed"] = {
                "expected": 42, "actual": config.get("seed")}
        if _jsonable(config.get("date_split_manifest")) != _jsonable(
                control_config.get("date_split_manifest")):
            mismatches["date_split_manifest"] = {
                "candidate": config.get("date_split_manifest"),
                "control": control_config.get("date_split_manifest")}
        if mismatches:
            raise ValueError(
                f"coarse-Bias candidate preflight is not comparable: {mismatches}")
        _write_sparse_plan({
            **base_payload,
            "status": "preflight_passed",
            "technical_status": "pending_training",
            "training_runs": 0,
            "allowed_config_keys": sorted(allowed),
            "config": config,
        })
        try:
            run_config(config, run_dir,
                       experiment="M00 seed42 coarse magnetic Bias")
        except Exception as exc:
            return _write_sparse_plan({
                **base_payload,
                "status": "technical_failure",
                "technical_status": "failed",
                "training_runs": 1,
                "error": repr(exc),
            })
        comparability = _assert_full_config_comparable(
            run_dir, control_run, 42, allowed)
        (run_dir / "comparability.json").write_text(
            json.dumps(comparability, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        if not comparability["comparable"]:
            raise ValueError(
                f"new coarse-Bias run is not comparable: "
                f"{comparability['mismatches']}")
        training_runs = 1

    if not _is_complete(run_dir, require_timing=True):
        raise RuntimeError("coarse-Bias run is not a complete five-epoch artifact")
    report = evaluate_run(
        run_dir, "M00 seed42 coarse magnetic Bias",
        baseline_run=control_run)
    payload = {
        **base_payload,
        "status": "development_passed" if report["passed"]
        else "development_quality_failed",
        "technical_status": "complete",
        "training_runs": training_runs,
        "development_result": report,
        "final_decision": {
            "adopted": "historical_a1s_epoch3",
            "locked_test_read": False,
        },
    }
    if not report["passed"]:
        return _write_sparse_plan(payload)
    try:
        isr = run_isr(run_dir, SPARSE_BIAS_ISR_OUTPUT)
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "technical_status": "isr_failed_to_complete",
            "isr_error": repr(exc),
        })
        return _write_sparse_plan(payload)
    payload["isr"] = isr
    payload["status"] = (
        "m00_adopted_candidate" if isr["passed"]
        else "isr_failed_historical_retained")
    payload["final_decision"] = {
        "adopted": "m00_candidate" if isr["passed"]
        else "historical_a1s_epoch3",
        "candidate": str(run_dir.resolve()),
        "checkpoint": isr["checkpoint"],
        "locked_test_read": False,
    }
    return _write_sparse_plan(payload)


def _pin_historical_reference(run_dir: Path) -> None:
    """Keep generated manifests pointed at the immutable formal A1-S epoch 3."""
    path = Path(run_dir) / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    expected = str((HISTORICAL_RUN / "epoch_03_background_model.pth").resolve())
    if config.get("historical_a1s_reference_checkpoint") == expected:
        return
    config["historical_a1s_reference_checkpoint"] = expected
    config["historical_a1s_reference_semantics"] = "a1s_formal_epoch3_v1"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_one(label: str, run_dir: Path, seed: int, control_run: Path) -> dict:
    run_dir = Path(run_dir)
    control_run = Path(control_run)
    comparability = _assert_comparable(control_run, control_run, seed)
    # The control is checked above; candidate comparability is checked again for reuse.
    if run_dir.exists() and _is_complete(run_dir):
        _pin_historical_reference(run_dir)
        comparability = _assert_comparable(run_dir, control_run, seed)
        if not comparability["comparable"]:
            raise ValueError(f"incomparable completed run: {comparability['mismatches']}")
        return evaluate_run(run_dir, label, baseline_run=control_run)
    if run_dir.exists() and any(run_dir.iterdir()):
        allowed = {"date_split_manifest.json"}
        if {path.name for path in run_dir.iterdir()} - allowed:
            raise FileExistsError(f"refusing to overwrite incomplete output: {run_dir}")
    split_manifest = DATE_SPLIT
    if seed != 42:
        run_dir.mkdir(parents=True, exist_ok=True)
        split_manifest = run_dir / "date_split_manifest.json"
        source = json.loads(DATE_SPLIT.read_text(encoding="utf-8"))
        source["seed"] = int(seed)
        split_manifest.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reference_run = A_RUN if seed == 42 else A_SEED43_RUN
    config = build_config(
        output_run=run_dir, reference_run=reference_run, seed=seed,
        date_split_manifest=split_manifest,
        overrides={"batch_size": 1024, "background_stratum_bias_gradient_target": 0.025},
    )
    run_config(config, run_dir, experiment=label)
    _pin_historical_reference(run_dir)
    comparability = _assert_comparable(run_dir, control_run, seed)
    (run_dir / "comparability.json").write_text(
        json.dumps(comparability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not comparability["comparable"]:
        raise ValueError(f"new run is not comparable to control: {comparability['mismatches']}")
    return evaluate_run(run_dir, label, baseline_run=control_run)


def _candidate_checkpoint(run_dir: Path) -> Path:
    summary = json.loads((Path(run_dir) / "training_summary.json").read_text(encoding="utf-8"))
    if summary.get("background_candidate_eligible") is not True:
        raise ValueError("ISR requires a candidate that passed the shared Development selector")
    checkpoint = summary.get("background_candidate_checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"eligible candidate checkpoint is missing: {checkpoint}")
    return Path(checkpoint).resolve()


def _isr_checkpoint(run_dir: Path) -> tuple[Path, Path]:
    """Return (selected epoch checkpoint, evaluator-compatible best alias)."""
    selected = _candidate_checkpoint(run_dir)
    best = selected.parent / "best_background_model.pth"
    if not best.is_file():
        raise ValueError(f"selected candidate lacks best checkpoint: {best}")
    selected_state = torch.load(selected, map_location="cpu", weights_only=True)
    best_state = torch.load(best, map_location="cpu", weights_only=True)
    if set(selected_state) != set(best_state) or any(
            not torch.equal(selected_state[key], best_state[key])
            for key in selected_state):
        raise ValueError("best_background_model.pth is not the selected checkpoint")
    return selected, best


def run_isr(run_dir: Path = AC_RUN, output_dir: Path = ISR_OUTPUT) -> dict:
    """Run exactly one ISR validation for the frozen seed42 candidate."""
    result_path = Path(run_dir) / "m00_low_budget_result.json"
    if not result_path.is_file():
        raise ValueError("ISR requires a persisted Development result")
    development = json.loads(result_path.read_text(encoding="utf-8"))
    if development.get("passed") is not True:
        raise ValueError("ISR requires a passed Development gate")
    checkpoint, evaluator_checkpoint = _isr_checkpoint(run_dir)
    output_dir = Path(output_dir)
    report_path = output_dir / "isr_m00_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse incomplete ISR output: {output_dir}")
        command = [
            sys.executable, "-m", "isr_evaluation.background_m00_eval",
            "--checkpoint", str(evaluator_checkpoint), "--cache", str(ISR_CACHE),
            "--comparison-cache", str(ISR_COMPARISON_CACHE),
            "--output-dir", str(output_dir), "--candidate-name", "M00_A1S_A_PLUS_C",
            "--reference-name", "M00_A1S",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    reported_checkpoint = report.get("checkpoint")
    if (not reported_checkpoint
            or Path(reported_checkpoint).resolve() != evaluator_checkpoint):
        raise ValueError("ISR report checkpoint does not match the frozen candidate")
    acceptance = report.get("acceptance_vs_reference")
    if not isinstance(acceptance, dict) or not isinstance(acceptance.get("passed"), bool):
        raise ValueError("ISR report lacks boolean acceptance_vs_reference.passed")
    return {
        "report": str(report_path.resolve()), "checkpoint": str(checkpoint),
        "isr_checkpoint": str(evaluator_checkpoint),
        "acceptance_vs_reference": acceptance,
        "passed": bool(acceptance["passed"]), "row_count": int(report.get("row_count", 0)),
    }


def _write_plan(payload: dict) -> dict:
    payload.setdefault(
        "timing_status",
        "completed runs predate wall-clock timing artifact; runner now records run_timing.json",
    )
    PLAN_PATH.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PLAN_MD_PATH.write_text(_plan_markdown(payload), encoding="utf-8")
    return payload


def _plan_markdown(payload: dict) -> str:
    experiments = payload.get("experiments", {})
    lines = [
        "# M00 low-budget A+C plan", "",
        f"Status: **{payload.get('status', 'unknown')}**",
        f"Technical status: **{payload.get('technical_status', 'not recorded')}**", "",
        "Configuration: batch size 1024; Bias gradient target 0.025; learning rate unchanged; locked-test read false.",
        f"Timing: {payload.get('timing_status', 'not recorded')}.",
        "", "| run | best epoch | profile RMSE | history improvement | A-control improvement | passed |",
        "|:---|---:|---:|---:|---:|:---|",
    ]
    calibration_lines = []
    for key in ("a_plus_c_seed42", "a_plus_c_seed43"):
        report = experiments.get(key)
        if not report:
            continue
        lines.append(
            f"| {key} | {report['candidate']['epoch']} | "
            f"{report['candidate']['profile_rmse']:.6f} | "
            f"{report['history_improvement_fraction'] * 100:.3f}% | "
            f"{report['baseline_improvement_fraction'] * 100:.3f}% | "
            f"{'yes' if report['passed'] else 'no'} |")
        audit = report.get("training_audit", {})
        calibration_lines.append(
            f"{key} Bias calibration: target={audit.get('bias_gradient_target')}, "
            f"resolved weight={audit.get('resolved_bias_weight')}, "
            f"candidate Bias/observation gradient ratio={audit.get('candidate_bias_gradient_ratio'):.4f}.")
    lines.extend(["", *calibration_lines, "", "Protection gates: source RMSE relative ≤1%; mean CCC drop ≤0.005; key-cell absolute Bias worsening ≤0.02 dex; minimum 30 profiles."])
    paired = payload.get("paired_validation", {})
    if paired:
        lines.append(f"Paired validation: **{'passed' if paired.get('stable_improvement') else 'failed'}**; ISR: `{paired.get('isr_status', 'not run')}`.")
    final = payload.get("final_decision")
    if final:
        lines.append(f"Final adoption: **{final.get('adopted', 'not recorded')}**.")
    diagnostic = experiments.get("condition_diagnostic", {})
    if diagnostic:
        lines.append(f"Condition diagnostic: `{diagnostic.get('report')}` ({diagnostic.get('group_count')} groups; `|sin I|` bins are proxies).")
    return "\n".join(lines) + "\n"


def run_auto() -> dict:
    baseline = evaluate_run(R2_RUN, "natural epoch baseline")
    a42 = evaluate_run(A_RUN, "A control seed42")
    a43 = evaluate_run(A_SEED43_RUN, "A control seed43")
    reports = {"baseline": baseline, "a_control_seed42": a42, "a_control_seed43": a43}
    if not a42.get("candidate_selection", {}).get("eligible", False):
        return _write_plan({"semantics": "m00_low_budget_plan_v2", "status": "a_control_seed42_ineligible", "technical_status": "complete", "thresholds": THRESHOLDS, "experiments": reports})
    ac42 = _run_one("A+C seed42", AC_RUN, 42, A_RUN)
    reports["a_plus_c_seed42"] = ac42
    from run_m00_condition_diagnostic import compare as compare_conditions
    diagnostic_path = AC_RUN / "m00_condition_diagnostic.json"
    condition_diagnostic = compare_conditions(AC_RUN, A_RUN, diagnostic_path)
    reports["condition_diagnostic"] = {
        "report": str(diagnostic_path.resolve()),
        "group_count": len(condition_diagnostic.get("groups", {})),
        "locked_test_read": False,
    }
    if not ac42["passed"]:
        reports["paired_validation"] = {"stable_improvement": False, "reason": "seed42_quality_failure", "isr_status": "skipped_seed42_failure"}
        return _write_plan({"semantics": "m00_low_budget_plan_v2", "status": "seed42_quality_failed", "technical_status": "complete", "thresholds": THRESHOLDS, "experiments": reports, "final_decision": {"adopted": "historical_a1s_epoch3"}})
    ac43 = _run_one("A+C seed43", AC_SEED43_RUN, 43, A_SEED43_RUN)
    reports["a_plus_c_seed43"] = ac43
    stable = bool(ac43["passed"])
    paired = {
        "stable_improvement": stable,
        "final_candidate_run": str(AC_RUN.resolve()),
        "final_candidate_epoch": int(ac42["candidate"]["epoch"]),
    }
    if not stable:
        paired["reason"] = "seed43_quality_failure"
        paired["isr_status"] = "skipped_unstable_improvement"
        return _write_plan({"semantics": "m00_low_budget_plan_v2", "status": "seed43_quality_failed", "technical_status": "complete", "thresholds": THRESHOLDS, "experiments": reports, "paired_validation": paired, "final_decision": {"adopted": "historical_a1s_epoch3"}})
    isr = run_isr(AC_RUN)
    paired["isr"] = isr
    return _write_plan({
        "semantics": "m00_low_budget_plan_v2", "status": "m00_adopted_candidate" if isr["passed"] else "isr_failed_historical_retained", "technical_status": "complete",
        "thresholds": THRESHOLDS,
        "experiments": reports, "paired_validation": paired,
        "final_decision": {"adopted": "m00_candidate" if isr["passed"] else "historical_a1s_epoch3", "candidate": str(AC_RUN.resolve()), "locked_test_read": False},
    })


def attach_isr_result(report_path: Path, plan_path: Path = PLAN_PATH) -> dict:
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    paired = payload.get("paired_validation", {})
    candidate_run = paired.get("final_candidate_run")
    if not candidate_run or not paired.get("stable_improvement"):
        raise ValueError("cannot attach ISR before both paired gates pass")
    checkpoint, evaluator_checkpoint = _isr_checkpoint(Path(candidate_run))
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if Path(report.get("checkpoint", "")).resolve() != evaluator_checkpoint:
        raise ValueError("ISR report checkpoint does not match the frozen candidate")
    acceptance = report.get("acceptance_vs_reference")
    if not isinstance(acceptance, dict) or not isinstance(acceptance.get("passed"), bool):
        raise ValueError("ISR report lacks boolean acceptance_vs_reference.passed")
    paired["isr"] = {"report": str(Path(report_path).resolve()), "checkpoint": str(checkpoint), "isr_checkpoint": str(evaluator_checkpoint), "acceptance_vs_reference": acceptance, "passed": bool(acceptance["passed"])}
    payload["paired_validation"] = paired
    payload["status"] = "m00_adopted_candidate" if acceptance["passed"] else "isr_failed_historical_retained"
    payload["final_decision"] = {"adopted": "m00_candidate" if acceptance["passed"] else "historical_a1s_epoch3", "candidate": str(candidate_run), "locked_test_read": False}
    return _write_plan(payload)


def _self_check() -> None:
    def row(epoch, rmse, ccc=0.7, bias=0.0):
        cells = {name: {"M00": {"bias": bias, "profile_count": 30}} for name in ("200-300_night", "200-300_day", "300-500_night", "300-500_day")}
        return {"stage": "background", "epoch": epoch, "fy_profile_rmse": rmse, "cosmic_profile_rmse": rmse, "fy_ccc": ccc, "cosmic_ccc": ccc, "ccc": ccc, "val_score": rmse, "fy_background_stratified": {"cells": cells}, "cosmic_background_stratified": {"cells": cells}}
    baseline = row(3, 1.0)
    selected = select_m00_background_candidate([row(1, 0.9, bias=0.03), row(2, 0.99)], baseline)
    assert selected["eligible"] and selected["selected_row"]["epoch"] == 2
    tie = select_m00_background_candidate([row(2, 1.0), row(1, 1.0)], baseline)
    assert tie["selected_row"]["epoch"] == 1
    assert abs(_relative(0.999, 1.0) - 0.001) < 1e-12
    print("m00 low-budget self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("baseline", "A+C", "auto", "seed42-single",
                 "seed42-coarse-bias", "seed42-magnetic-daynight-bias",
                 "seed42-magnetic-daynight-bias-v2"),
        default="auto")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--attach-isr")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if (args.preflight_only or args.evaluate_only) and args.experiment != \
            "seed42-magnetic-daynight-bias-v2":
        parser.error('--preflight-only/--evaluate-only are v2-only')
    if args.preflight_only and args.evaluate_only:
        parser.error('--preflight-only and --evaluate-only are mutually exclusive')
    if args.self_check:
        _self_check()
    elif args.attach_isr:
        attach_isr_result(Path(args.attach_isr))
    elif args.experiment == "seed42-single":
        run_seed42_single()
    elif args.experiment == "seed42-coarse-bias":
        run_seed42_coarse_bias()
    elif args.experiment == "seed42-magnetic-daynight-bias":
        run_seed42_magnetic_daynight_bias()
    elif args.experiment == "seed42-magnetic-daynight-bias-v2":
        run_seed42_magnetic_daynight_bias_v2(
            preflight_only=args.preflight_only, evaluate_only=args.evaluate_only)
    elif args.experiment == "auto":
        run_auto()
    elif args.experiment == "baseline":
        report = evaluate_run(R2_RUN, "natural epoch baseline")
        _write_plan({"semantics": "m00_low_budget_plan_v2", "status": "baseline_evaluated", "experiments": {"baseline": report}})
    else:
        _run_one("A+C seed42", AC_RUN, 42, A_RUN)


if __name__ == "__main__":
    main()
