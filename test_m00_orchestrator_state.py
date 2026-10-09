import json

import pytest

from run_a1s_m00_residual_correction import (
    _isr_status, _resolve_seed, _selector, _state,
)


def test_state_seed_and_selector_outcomes(tmp_path):
    checkpoint = tmp_path / "best_background_model.pth"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    summary_path = tmp_path / "training_summary.json"
    summary_path.write_text(json.dumps({
        "best_epochs": {"background": 3},
        "checkpoint_stage": "background",
        "checkpoint": str(checkpoint),
    }), encoding="utf-8")
    seed, _ = _resolve_seed(str(summary_path), str(checkpoint))
    assert seed == str(checkpoint.resolve())

    state_path = tmp_path / "experiment_state.json"
    _state(str(state_path), "decoder_train", "running", results=[])
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["stage"] == "decoder_train"
    assert saved["status"] == "running"

    status, selected = _selector({"checkpoint": str(checkpoint)})
    assert (status, selected) == ("passed", str(checkpoint.resolve()))
    status, selected = _selector({
        "best_epochs": {"background": None},
        "background_selection_audits": [{"eligible": False}] * 5,
    })
    assert (status, selected) == ("quality_failed", None)


def test_isr_requires_explicit_boolean_outcome(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"passed": false}', encoding="utf-8")
    assert _isr_status(str(report)) == "quality_failed"
    report.write_text('{"passed": null}', encoding="utf-8")
    with pytest.raises(ValueError, match="boolean passed"):
        _isr_status(str(report))
