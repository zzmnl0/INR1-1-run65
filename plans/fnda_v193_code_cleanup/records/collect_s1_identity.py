"""Read-only provenance audit; writes only this stage's JSON evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path, expected=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    result = {"path": str(p), "exists": p.is_file(), "expected_sha256": expected}
    if p.is_file():
        before = p.stat()
        with p.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        after = p.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), p
        result["bytes"] = after.st_size
        result["matches_record"] = result["sha256"] == expected if expected else None
    return result


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8", errors="replace").strip()


def main():
    run = ROOT / "checkpoints_fsia/run67-m2w-120-500-analysis-a1s-10epoch"
    manifest = read(run / "run_manifest.json")
    summary = read(run / "training_summary.json")
    ledger_path = ROOT / "FNDA_V19_slim_work/code_build_ledger.json"
    ledger = read(ledger_path)
    lineage_path = ROOT / "FNDA_V19_slim_work/v19_lineage.json"
    lineage = read(lineage_path)
    code = [identity(r["source"], r["current_sha256"])
            for r in ledger["source_history"]["comparison_to_S6_snapshot"]]
    data = [{"key": key, **identity(value["path"], value.get("sha256"))}
            for key, value in manifest["data_identity"].items()]
    weights = [identity(run / "best_fsia_model.pth", summary["checkpoint_sha256"]),
               identity(run / "epoch_15_model.pth"), identity(run / "best_background_model.pth")]
    for item in ledger["weights"]:
        weights.append({"role": item["role"], **identity(ROOT / "FNDA_V19_slim_package/B_model_code" / item["path"], item["sha256"])})
    weights.append(identity(ROOT / "checkpoints_fsia/run67-m2w-120-500-background-stratum-bias-a1s/best_background_model.pth"))
    figures = []
    package = ROOT / "a1s_submission_v17/run_20260920_v19/submission_v19.3"
    for item in lineage["figures"]:
        folder = package / ("upload_flat" if item["section"] == "main" else "work/clean_SI/figures")
        expected = item["submission"][0]["sha256"] if item["submission"] else None
        figures.append({"section": item["section"], **identity(folder / item["submitted_name"], expected)})
    mappings = []
    for item in lineage["data_and_generators"]:
        mappings.append({"item": item["item"], **{
            key: [identity(x["path"], x.get("sha256")) for x in item[key]]
            for key in ("source_files", "generators")}})
    contracts = {}
    contract_paths = [ROOT / "isr_validation_outputs/m2w-a1s-full30-analysis-isr/isr_evaluation_contract.json",
                      ROOT / "isr_validation_outputs/m2w-a1s2-vs-a1s-giro-full30/giro_peak_contract.json"]
    for p in contract_paths:
        contract = read(p)
        contracts[str(p)] = {"identity": identity(p), "candidate_checkpoint": contract.get("candidate_checkpoint"),
                             "baseline_checkpoint": contract.get("baseline_checkpoint"),
                             "hash_verification": contract.get("hash_verification"),
                             "evaluation_code_sha256": contract.get("evaluation_code_sha256")}
    source_ledger = read(ROOT / "FNDA_V19_slim_work/source_ledger.json")
    sensitivity = {}
    for key in ("FNDA_V8_N4D64", "FNDA_V8_N8D32", "FNDA_V8_N8D64_REPRODUCTION"):
        directory = Path(source_ledger["origin_ids"][key])
        s = read(directory / "training_summary.json")
        sensitivity[key] = {"run": str(directory), "summary": identity(directory / "training_summary.json"),
                            "manifest": identity(directory / "run_manifest.json"),
                            "weight": identity(directory / "best_fsia_model.pth", s.get("checkpoint_sha256")),
                            "architecture": s.get("architecture"), "best_epochs": s.get("best_epochs"),
                            "training_protocol": s.get("training_protocol")}
    f2_prior = identity(package.parent / "submission_v19.2/upload_flat/Figure2_RO_coverage.pdf")
    f2_current = identity(package / "upload_flat/Figure2_RO_coverage.pdf")
    assert f2_prior["sha256"] == f2_current["sha256"]
    record = {
        "schema": 1, "stage": "S1", "collected_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_SOURCE_CONFIRMED_BY_AUTHOR",
        "training_source_confirmation": {"date": "2026-10-09", "source": "direct user statement at S2 authorization",
            "statement": "Current workspace reran N8D64 sensitivity experiment; all parameters and results identical; confirmed training source",
            "rerun_executed_by_this_audit": False},
        "user_decision": "2026-10-09: clean current workspace; retain historical identity and evidence gaps",
        "no_model_execution": True, "no_training_code_mutation": True,
        "git": {"head": git("rev-parse", "HEAD"), "last_commit": git("log", "-1", "--format=%H %aI %s"),
                "tracked_status": git("status", "--short", "--untracked-files=no"), "diff_stat": git("diff", "--stat")},
        "main_run": {"path": str(run), "manifest": identity(run / "run_manifest.json"),
                     "summary": identity(run / "training_summary.json"), "history": identity(run / "fsia_training_history.json"),
                     "training_log": identity(run / "training.log"), "initial_config": manifest["config"],
                     "initial_code_identity": manifest["code_identity"], "resume_events": manifest["resume_events"],
                     "training_environment": manifest["environment"], "training_protocol": summary["training_protocol"],
                     "architecture": summary["architecture"], "best_epochs": summary["best_epochs"],
                     "resolved_training": manifest["resolved_training"], "summary_fields": summary},
        "weights": weights, "training_data_identity": data, "maintenance_code": code,
        "figures": figures, "result_lineage": mappings, "contracts": contracts,
        "sensitivity_runs": sensitivity,
        "figure2_layout_exception": {"v192": f2_prior, "v193": f2_current,
            "record": read(package.parent / "submission_v19.2/checks/layout_changes.json"),
            "verification_scope": "Current bytes equal V19.2; title-only explanation from historical layout record, not new visual review"},
        "evidence_ledgers": [identity(ledger_path), identity(lineage_path), identity(ROOT / "FNDA_V19_slim_work/source_ledger.json"),
                             identity(ROOT / "FNDA_V19_slim_work/author_decisions_20260922.json"),
                             identity(ROOT / "FNDA_V19_slim_work/A1S_weight_resolution_20260922.json")],
        "limitations": ["Historical cache-to-checkpoint byte binding not established",
                        "Historical single frozen snapshot not established; current training source confirmed by user N8D64 rerun",
                        "Current defaults and inherited initial manifest config are not an executable final analysis recipe",
                        "Source lineage hashing is not inference or scientific reproduction",
                        "No complete untracked-file inventory; S2/S3 must establish full maintenance scope"]}
    assert len(code) == 38, len(code)
    assert all(x.get("matches_record") for x in code), "Maintenance source drift"
    assert weights[0]["sha256"] == "94f4da0b265c313a812159560db25c5ae5394b29603d75f47b18d5f6d9bb6dc1"
    target = OUT / "model_identity.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"output": str(target), "code_matches": sum(x.get("matches_record") is True for x in code),
              "data_matches": sum(x.get("matches_record") is True for x in data), "data_total": len(data),
              "lineage_items": len(mappings),
              "lineage_issues": [x for item in mappings for key in ("source_files", "generators") for x in item[key] if not x.get("matches_record")],
              "figure_issues": [x for x in figures if not x.get("matches_record")],
              "weight_issues": [x for x in weights if not x["exists"] or x.get("matches_record") is False]}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
