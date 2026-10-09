import json
from pathlib import Path

import pytest

from inr_modules.mdia.train_fsia import (
    _night_generalization_selection_audit,
    m00_natural_selection_audit, select_m00_background_candidate)
from run_m00_low_budget import _relative, _selected, attach_isr_result


CELLS = ("200-300_night", "200-300_day", "300-500_night", "300-500_day")


def _row(epoch, rmse, ccc=0.7, bias=0.0, *, missing=False):
    cells = {
        name: {"M00": {"bias": bias, "profile_count": 30}}
        for name in CELLS
    }
    row = {
        "stage": "background", "epoch": epoch,
        "fy_profile_rmse": rmse, "cosmic_profile_rmse": rmse,
        "fy_ccc": ccc, "cosmic_ccc": ccc, "ccc": ccc,
        "fy_background_stratified": {"cells": cells},
        "cosmic_background_stratified": {"cells": cells},
    }
    if missing:
        row.pop("cosmic_profile_rmse")
    return row


def test_eligible_epoch_beats_ineligible_lower_rmse_and_ties_are_early():
    baseline = _row(3, 1.0)
    result = select_m00_background_candidate(
        [_row(1, 0.9, bias=0.03), _row(2, 0.99)], baseline)
    assert result["eligible"] is True
    assert result["selected_row"]["epoch"] == 2

    tied = select_m00_background_candidate([_row(2, 1.0), _row(1, 1.0)], baseline)
    assert tied["selected_row"]["epoch"] == 1


def test_all_ineligible_falls_back_without_eligibility():
    baseline = _row(3, 1.0)
    result = select_m00_background_candidate([_row(1, 0.9, bias=0.03)], baseline)
    assert result["eligible"] is False
    assert result["selected_row"]["epoch"] == 1
    assert result["reason"] == "no_epoch_passed_protection"


def test_missing_or_nonfinite_metrics_never_pass():
    baseline = _row(3, 1.0)
    missing = m00_natural_selection_audit(_row(1, 0.9, missing=True), baseline, 1)
    assert missing["eligible"] is False
    nonfinite = _row(1, float("nan"))
    assert m00_natural_selection_audit(nonfinite, baseline, 1)["eligible"] is False


def test_improvement_boundary():
    assert abs(_relative(0.999, 1.0) - 0.001) < 1e-12
    assert _relative(1.0, 1.0) == 0.0


def test_isr_checkpoint_mismatch_is_rejected(tmp_path: Path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"checkpoint")
    summary = tmp_path / "training_summary.json"
    summary.write_text(json.dumps({
        "background_candidate_eligible": True,
        "background_candidate_checkpoint": str(candidate),
    }), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "paired_validation": {"stable_improvement": True, "final_candidate_run": str(tmp_path)},
    }), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"checkpoint": str(tmp_path / "other.pth")}), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        attach_isr_result(report, plan)


def test_isr_cannot_attach_before_paired_gate(tmp_path: Path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"checkpoint")
    (tmp_path / "training_summary.json").write_text(json.dumps({
        "background_candidate_eligible": True,
        "background_candidate_checkpoint": str(candidate),
    }), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "paired_validation": {"stable_improvement": False, "final_candidate_run": str(tmp_path)},
    }), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "checkpoint": str(candidate),
        "acceptance_vs_reference": {"passed": False},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="paired gates"):
        attach_isr_result(report, plan)


def _night_row(epoch, rmse, ccc, low_bias=0.10):
    magnetic = ("abs_sinI_lt_0.25", "abs_sinI_0.25_0.5", "abs_sinI_ge_0.5")
    cells = {}
    for source in ("fy", "cosmic"):
        source_cells = {}
        for alt in ("200-300", "300-500"):
            for period in ("night", "day"):
                for mag in magnetic:
                    source_cells[f"{alt}_{period}_{mag}"] = {
                        "M00": {"bias": low_bias if mag == magnetic[0] and period == "night" else 0.0,
                                "rmse": rmse, "ccc": ccc, "profile_count": 30}}
        cells[source] = {"magnetic_latitude_cells": source_cells}
    return {"epoch": epoch, "stage": "background",
            "fy_background_stratified": cells["fy"],
            "cosmic_background_stratified": cells["cosmic"],
            "fy_profile_rmse": rmse, "cosmic_profile_rmse": rmse,
            "fy_ccc": ccc, "cosmic_ccc": ccc, "ccc": ccc}


def test_night_generalization_selection_requires_fixed_improvement():
    baseline = _night_row(3, 1.0, 0.70)
    candidate = _night_row(1, 0.998, 0.706, low_bias=0.08)
    audit = _night_generalization_selection_audit(candidate, baseline, 1)
    assert audit["eligible"] is True
    assert audit["selection_key"][-1] == -1
    failed = _night_generalization_selection_audit(
        _night_row(2, 1.0, 0.70), baseline, 2)
    assert failed["eligible"] is False
    assert "night_macro_rmse_improvement_below_0.1pct" in failed["reasons"]


def test_ineligible_night_epoch_can_be_reported_without_promotion(tmp_path: Path):
    row = _night_row(5, 1.0, 0.70)
    row["val_score"] = 1.0
    row["checkpoint_selection"] = {
        "semantics": "m00_night_generalization_v1",
        "eligible": False,
        "reasons": ["fixed_gate_failed"],
    }
    (tmp_path / "fsia_training_history.json").write_text(
        json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="no eligible"):
        _selected(tmp_path)
    scalar, selected, selection = _selected(tmp_path, allow_ineligible=True)
    assert scalar["epoch"] == selected["epoch"] == 5
    assert selection["eligible"] is False
