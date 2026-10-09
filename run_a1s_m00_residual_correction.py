"""Ordered, resumable A1-S M00 residual-correction candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os

import numpy as np

from inr_modules.config_mdia import get_config_mdia
from inr_modules.mdia.m00_observation import (
    M00_QD_RELIABILITY_SEMANTICS, load_magnetic_sidecar,
    validate_strict_v2_calibration,
)
from inr_modules.mdia.m00_residual_experiment import validate_a1s_manifest, write_json
from inr_modules.mdia.train_fsia import train_fsia

STATE_SEMANTICS = "a1s_m00_residual_correction_state_v1"
STATUSES = {"running", "passed", "quality_failed", "contract_failed",
            "runtime_failed", "awaiting_external_isr"}


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _state(path: str, stage: str, status: str, **values) -> dict:
    if status not in STATUSES:
        raise ValueError(f"unknown experiment status: {status}")
    old = _load_json(path) if os.path.isfile(path) else {}
    if old and old.get("semantics") != STATE_SEMANTICS:
        raise ValueError("experiment state semantics mismatch")
    old.update({"semantics": STATE_SEMANTICS, "stage": stage, "status": status,
                "updated_utc": datetime.now(timezone.utc).isoformat(), **values})
    write_json(path, old)
    return old


def _resolve_seed(summary_path: str, supplied: str | None) -> tuple[str, dict]:
    summary = _load_json(summary_path)
    if summary.get("best_epochs", {}).get("background") != 3:
        raise ValueError("A1-S summary does not select Background epoch 3")
    if summary.get("checkpoint_stage") != "background":
        raise ValueError("A1-S seed is not a Background checkpoint")
    seed = os.path.abspath(str(summary.get("checkpoint") or ""))
    if not seed or not os.path.isfile(seed):
        raise FileNotFoundError("A1-S epoch-3 checkpoint is missing")
    if supplied and os.path.abspath(supplied) != seed:
        raise ValueError("supplied seed differs from A1-S training summary")
    return seed, summary


def _validate_diagnostic(value: dict) -> None:
    if value.get("semantics") != M00_QD_RELIABILITY_SEMANTICS:
        raise ValueError("diagnostic reliability semantics mismatch")
    if value.get("residual_reference") != "RO_logNe_minus_M00_A1S_epoch3_logNe":
        raise ValueError("diagnostic residual reference mismatch")
    validate_strict_v2_calibration(value)
    keys = set(map(str, value.get("variance", {})))
    weights = value.get("reliability_weight", {})
    if not keys or set(map(str, weights)) != keys:
        raise ValueError("reliability weights do not exactly cover variance keys")
    residual_mean = value.get("residual_mean")
    if not isinstance(residual_mean, dict) or set(map(str, residual_mean)) != keys:
        raise ValueError("diagnostic must preserve per-stratum residual means")
    support = value.get("support")
    if not isinstance(support, dict) or set(map(str, support)) != keys:
        raise ValueError("diagnostic support must cover every reliability key")
    for key, item in support.items():
        if int(item.get("profiles", 0)) < 20 or int(item.get("profile_days", 0)) < 10:
            raise ValueError(f"diagnostic support below threshold: {key}")
        if float(item.get("weighted_ess", item.get("ess", 0.0))) < 10.0:
            raise ValueError(f"diagnostic weighted ESS below threshold: {key}")
    if {key.split("|", 1)[0] for key in keys} != {"FY", "COSMIC"}:
        raise ValueError("diagnostic must cover FY and COSMIC")
    numbers = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(numbers).all() or np.any(numbers < .25) or np.any(numbers > 4.):
        raise ValueError("reliability weights violate [0.25, 4] contract")
    reports = value.get("localization_diagnostic")
    if not isinstance(reports, dict) or set(reports) != {"FY", "COSMIC"}:
        raise ValueError("localization diagnostic must cover FY and COSMIC")
    for source, report in reports.items():
        if report.get("semantics") != "a1s_m00_train_loo_3d_localization_replay_v2":
            raise ValueError(f"{source} localization replay is not production token v2")
        if not report.get("train_only") or not report.get("self_profile_excluded"):
            raise ValueError(f"{source} localization replay is not Train-only/self-excluding")
        if int(report.get("allowed_profile_count", 0)) <= 0:
            raise ValueError(f"{source} localization replay has no Train profile allow-list")
        if report.get("selection", {}).get("selection") != (
                "stable_first_up_to_8_points_per_profile_round_robin_utc_day_v2"):
            raise ValueError(f"{source} localization query selection is not registered")
    # QD is the accepted coordinate for this experiment.  Migration to the
    # optional AACGM audit coordinate is descriptive only and never blocks a
    # valid QD diagnostic; row identity and finite-value checks remain strict.
    # The generated Train-only artifact records the converter-level ``qd_v1``
    # label while the stratum contract is stored separately.  Both are the
    # same accepted QD coordinate for this experiment; AACGM/geographic
    # fallbacks remain invalid.
    if value.get("magnetic_coordinate", {}).get("semantics") not in (
            None, "qd_v1", "qd_abs_sin_lat_025_050_v1"):
        raise ValueError("diagnostic magnetic coordinate is not QD")


def _candidate_config(base: dict, seed: str, calibration: dict, output: str,
                      semantics: str, fy: str, cosmic: str, baseline: str) -> dict:
    value = dict(base)
    value.update({
        "save_dir": output, "background_only": True, "background_epochs": 5,
        "background_fixed_updates_per_epoch": 960,
        "analysis_epochs": 0, "background_seed_ckpt": seed,
        "m00_observation_seed_checkpoint": seed, "m00_candidate_semantics": semantics,
        "m00_candidate_residual_cap": .05, "m00_peak_relative_scale_km": 100.,
        "m00_observation_calibration": calibration,
        "m00_observation_calibration_semantics": M00_QD_RELIABILITY_SEMANTICS,
        "m00_observation_training_scope": "m00_observation_error_only_v1",
        "background_training_semantics": "m00_observation_error_only_v1",
        "m00_observation_stage": "bias_r", "m00_representativeness_enabled": False,
        "m00_localization_enabled": False, "m00_observation_qd_sidecar_fy": fy,
        "m00_observation_qd_sidecar_cosmic": cosmic,
        "m00_observation_magnetic_sidecar_fy": fy,
        "m00_observation_magnetic_sidecar_cosmic": cosmic,
        "magnetic_stratum_semantics": "qd_abs_sin_lat_025_050_v1",
        "m00_observation_strict_magnetic": True,
        "background_checkpoint_selection": "bias_constrained_v1",
        "background_selection_baseline": baseline, "checkpoint_semantic": semantics,
        "skip_hash_verification": True,
    })
    return value


def _selector(summary: dict) -> tuple[str, str | None]:
    checkpoint = summary.get("checkpoint")
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise ValueError("selected checkpoint is missing")
        return "passed", os.path.abspath(checkpoint)
    audits = summary.get("background_selection_audits")
    if (summary.get("best_epochs", {}).get("background") is None
            and isinstance(audits, list) and len(audits) >= 5
            and all(item.get("eligible") is False for item in audits[-5:])):
        return "quality_failed", None
    raise ValueError("training ended without a complete selector outcome")


def _isr_status(path: str) -> str:
    report = _load_json(path)
    if not isinstance(report.get("passed"), bool):
        raise ValueError("ISR report lacks an explicit boolean passed outcome")
    return "passed" if report["passed"] else "quality_failed"


def run(*, training_summary: str, manifest: str, sidecar_fy: str,
        sidecar_cosmic: str, diagnostic_json: str, output_root: str,
        loader_sidecar_fy: str | None = None,
        loader_sidecar_cosmic: str | None = None,
        seed: str | None = None, smoke: bool = False,
        isr_reports: dict[str, str] | None = None,
        baseline_replay: str | None = None, state_path: str | None = None) -> dict:
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    state_path = os.path.abspath(state_path or os.path.join(output_root, "experiment_state.json"))
    try:
        _state(state_path, "preflight", "running")
        date_contract = validate_a1s_manifest(manifest)
        seed, seed_summary = _resolve_seed(training_summary, seed)
        baseline_replay = baseline_replay or seed_summary.get("background_selection_baseline")
        if not baseline_replay or not os.path.isfile(baseline_replay):
            raise FileNotFoundError("A1-S Development baseline replay is required")
        config = dict(get_config_mdia())
        reference_manifest = os.path.join(
            os.path.dirname(os.path.abspath(seed)), "run_manifest.json")
        if not os.path.isfile(reference_manifest):
            raise FileNotFoundError("A1-S colocated run_manifest.json is required")
        with open(reference_manifest, encoding="utf-8") as stream:
            reference_payload = json.load(stream)
        reference_config = reference_payload.get("config")
        if not isinstance(reference_config, dict):
            raise ValueError("A1-S run_manifest lacks its resolved config")
        config.update(reference_config)
        config["date_split_manifest"] = os.path.abspath(manifest)
        for path, source, data_key in ((sidecar_fy, "FY", "fy_path"),
                                       (sidecar_cosmic, "COSMIC", "cosmic_path")):
            data = np.load(config[data_key], mmap_mode="r")
            with np.load(path, allow_pickle=False) as sidecar_probe:
                if "row_index" not in sidecar_probe.files:
                    raise ValueError(f"{source} QD sidecar requires row_index")
                sidecar_rows = np.asarray(sidecar_probe["row_index"], dtype=np.int64).reshape(-1)
            if (sidecar_rows.size == 0 or np.any(sidecar_rows < 0)
                    or np.any(sidecar_rows >= len(data))
                    or np.unique(sidecar_rows).size != sidecar_rows.size):
                raise ValueError(f"{source} QD sidecar row_index is invalid")
            load_magnetic_sidecar(
                path, expected_rows=sidecar_rows.size, strict_qd=True,
                expected_coords=np.asarray(data[sidecar_rows, :4], dtype=float),
                expected_row_indices=sidecar_rows)
            row_days = np.floor(np.asarray(data[sidecar_rows, 3], dtype=float) / 24.0)
            if (not np.isfinite(row_days).all()
                    or not set(row_days.astype(np.int64).tolist()).issubset(
                        set(date_contract["train"]))):
                raise ValueError(f"{source} sidecar contains non-Train rows")
        calibration = _load_json(diagnostic_json)
        _validate_diagnostic(calibration)
        loader_sidecar_fy = os.path.abspath(loader_sidecar_fy or sidecar_fy)
        loader_sidecar_cosmic = os.path.abspath(loader_sidecar_cosmic or sidecar_cosmic)
        allowed_loader_days = set(date_contract["train"]) | set(date_contract["development"])
        for path, source, data_key in (
                (loader_sidecar_fy, "FY", "fy_path"),
                (loader_sidecar_cosmic, "COSMIC", "cosmic_path")):
            data = np.load(config[data_key], mmap_mode="r")
            with np.load(path, allow_pickle=False) as sidecar_probe:
                if "row_index" not in sidecar_probe.files:
                    raise ValueError(f"{source} loader QD sidecar requires row_index")
                loader_rows = np.asarray(
                    sidecar_probe["row_index"], dtype=np.int64).reshape(-1)
            if (loader_rows.size == 0 or np.any(loader_rows < 0)
                    or np.any(loader_rows >= len(data))
                    or np.unique(loader_rows).size != loader_rows.size):
                raise ValueError(f"{source} loader QD sidecar row_index is invalid")
            load_magnetic_sidecar(
                path, expected_rows=loader_rows.size, strict_qd=True,
                expected_coords=np.asarray(data[loader_rows, :4], dtype=float),
                expected_row_indices=loader_rows)
            loader_days = np.floor(
                np.asarray(data[loader_rows, 3], dtype=float) / 24.0)
            if (not np.isfinite(loader_days).all()
                    or not set(loader_days.astype(np.int64).tolist()).issubset(
                        allowed_loader_days)):
                raise ValueError(f"{source} loader QD sidecar contains Locked-test rows")
        inputs = {"training_summary": os.path.abspath(training_summary), "seed": seed,
                  "manifest": os.path.abspath(manifest), "sidecar_fy": os.path.abspath(sidecar_fy),
                  "sidecar_cosmic": os.path.abspath(sidecar_cosmic),
                  "loader_sidecar_fy": loader_sidecar_fy,
                  "loader_sidecar_cosmic": loader_sidecar_cosmic,
                  "diagnostic_json": os.path.abspath(diagnostic_json),
                  "train_dates": date_contract["train"]}
        old = _load_json(state_path)
        if old.get("inputs") and old["inputs"] != inputs:
            raise ValueError("resume inputs differ from persisted state")
        results = list(old.get("results", []))
        _state(state_path, "preflight", "passed", inputs=inputs, results=results)
        candidates = (("m00_decoder_last_trust_region_v1", "decoder"),
                      ("m00_peak_relative_lowrank3_v1", "peak"))
        for index, (semantics, short) in enumerate(candidates):
            prior = next((x for x in results if x.get("semantics") == semantics), None)
            if index == 1:
                first = next((x for x in results if x.get("semantics") == candidates[0][0]), None)
                if not first or first.get("status") != "quality_failed":
                    break
            report_path = (isr_reports or {}).get(semantics)
            if prior and prior.get("status") == "awaiting_external_isr":
                if not report_path:
                    return _state(state_path, f"{short}_awaiting_isr",
                                  "awaiting_external_isr", results=results)
                prior.update({"status": _isr_status(report_path),
                              "isr_report": os.path.abspath(report_path)})
                _state(state_path, f"{short}_isr", prior["status"], results=results)
                if prior["status"] == "passed":
                    break
                continue
            if prior and prior.get("status") in {"passed", "quality_failed"}:
                if prior["status"] == "passed":
                    break
                continue
            out = os.path.join(output_root, f"candidate_{short}")
            cfg = _candidate_config(config, seed, calibration, out, semantics,
                                    loader_sidecar_fy, loader_sidecar_cosmic,
                                    baseline_replay)
            cfg["smoke_run"] = bool(smoke)
            _state(state_path, f"{short}_train", "running", candidate=semantics,
                   results=results, last_training_state=os.path.join(out, "last_training_state.pth"))
            train_fsia(cfg)
            # The ISR evaluator consumes the colocated resolved configuration.
            # Keep this manifest local to the candidate; never alter A1-S.
            write_json(os.path.join(out, "run_manifest.json"), {"config": cfg})
            summary_path = os.path.join(out, "training_summary.json")
            summary = _load_json(summary_path) if os.path.isfile(summary_path) else {}
            status, checkpoint = _selector(summary)
            item = {"semantics": semantics, "status": status,
                    "training_summary": os.path.abspath(summary_path), "checkpoint": checkpoint}
            results.append(item)
            _state(state_path, f"{short}_selector", status, results=results)
            if status == "quality_failed":
                continue
            if not report_path:
                item["status"] = "awaiting_external_isr"
                return _state(state_path, f"{short}_awaiting_isr",
                              "awaiting_external_isr", results=results)
            item.update({"status": _isr_status(report_path),
                         "isr_report": os.path.abspath(report_path)})
            _state(state_path, f"{short}_isr", item["status"], results=results)
            if item["status"] == "passed":
                break
        adopted = next((x["semantics"] for x in results if x["status"] == "passed"), "A1-S_M00")
        decision = {"semantics": "a1s_m00_residual_correction_ordered_v1",
                    "m00_reference": "A1-S_epoch3", "m11": "deferred",
                    "locked_test": "excluded", "results": results, "adopted": adopted}
        write_json(os.path.join(output_root, "final_decision.json"), decision)
        _state(state_path, "complete", "passed", results=results, adopted=adopted)
        return decision
    except (OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_json(os.path.join(output_root, "final_decision.json"), {
            "semantics": "a1s_m00_residual_correction_ordered_v1",
            "status": "contract_failed",
            "m00_reference": "A1-S_epoch3",
            "m11": "deferred",
            "locked_test": "excluded",
            "adopted": "A1-S_M00",
            "failure": error,
        })
        _state(state_path, "stopped", "contract_failed", error=error)
        raise
    except RuntimeError as exc:
        error = f"RuntimeError: {exc}"
        write_json(os.path.join(output_root, "final_decision.json"), {
            "semantics": "a1s_m00_residual_correction_ordered_v1",
            "status": "runtime_failed",
            "m00_reference": "A1-S_epoch3",
            "m11": "deferred",
            "locked_test": "excluded",
            "adopted": "A1-S_M00",
            "failure": error,
        })
        _state(state_path, "stopped", "runtime_failed", error=error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sidecar-fy", required=True)
    parser.add_argument("--sidecar-cosmic", required=True)
    parser.add_argument("--loader-sidecar-fy", default=None)
    parser.add_argument("--loader-sidecar-cosmic", default=None)
    parser.add_argument("--diagnostic-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--isr-report-decoder", default=None)
    parser.add_argument("--isr-report-peak", default=None)
    parser.add_argument("--baseline-replay", default=None)
    args = parser.parse_args()
    values = vars(args)
    values["isr_reports"] = {k: v for k, v in (
        ("m00_decoder_last_trust_region_v1", values.pop("isr_report_decoder")),
        ("m00_peak_relative_lowrank3_v1", values.pop("isr_report_peak"))) if v}
    print(json.dumps(run(**values), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
