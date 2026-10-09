"""Parse maintenance sources without importing or executing the training project."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PACKAGES = ("inr_modules", "isr_evaluation", "traditional_static_etkf")


def module_name(path):
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def main():
    paths = set(ROOT.glob("*.py"))
    for package in PACKAGES:
        paths.update((ROOT / package).rglob("*.py"))
    relative = sorted(p.relative_to(ROOT) for p in paths if "__pycache__" not in p.parts)
    modules = {module_name(p): p.as_posix() for p in relative}
    records = {}
    for p in relative:
        full = ROOT / p
        raw = full.read_bytes()
        tree = ast.parse(raw.decode("utf-8-sig"), filename=str(p))
        mod = module_name(p)
        package = mod if p.stem == "__init__" else mod.rpartition(".")[0]
        imports, dynamic, keys = [], [], set()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = package.split(".") if package else []
                    parent = parent[:len(parent) - node.level + 1]
                    base = ".".join(parent + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                names = [base] + [base + "." + a.name for a in node.names]
            for name in names:
                candidate = name
                while candidate and candidate not in modules:
                    candidate = candidate.rpartition(".")[0]
                target = modules.get(candidate)
                if target and target != p.as_posix():
                    imports.append({"line": node.lineno, "target": target, "import": name})
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if any(x in call for x in ("import_module", "spec_from_file_location", "__import__", "subprocess", "runpy")):
                    dynamic.append({"line": node.lineno, "call": ast.unparse(node)[:500]})
                if call.endswith(".get") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    if any(x in call.lower() for x in ("config", "cfg")):
                        keys.add(node.args[0].value)
        records[p.as_posix()] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "lines": len(raw.splitlines()),
            "summary": (ast.get_docstring(tree) or "").split("\n")[0],
            "imports": imports, "dynamic_calls": dynamic, "config_get_keys": sorted(keys),
            "symbols": [{"name": n.name, "line": n.lineno, "kind": type(n).__name__}
                        for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]}
    entrypoints = {
        "train": ["main_fsia.py"], "plot": ["plot_fsia.py"],
        "giro": ["evaluate_giro_peak.py"], "isr": ["isr_evaluation/main_isr_eval.py"],
        "static_etkf": ["traditional_static_etkf/cli.py", "traditional_static_etkf/runner.py"],
        "qc": ["qc_profile_data.py"], "model_selftest": ["inr_modules/mdia/fsia_model.py"]}
    closures = {}
    for workflow, entries in entrypoints.items():
        assert all(e in records for e in entries), entries
        seen, todo = set(), list(entries)
        while todo:
            item = todo.pop()
            if item in seen:
                continue
            seen.add(item)
            todo.extend(x["target"] for x in records[item]["imports"])
            # Package initializers run when a submodule is imported.
            parent = Path(item).parent
            while parent != Path("."):
                init = (parent / "__init__.py").as_posix()
                if init in records:
                    todo.append(init)
                parent = parent.parent
        closures[workflow] = sorted(seen)
    retained = set().union(*(set(v) for v in closures.values()))
    references = []
    text_paths = list(ROOT.glob("*.ps1")) + list(ROOT.glob("*.bat")) + list(ROOT.glob("*.cmd")) + list(ROOT.glob("*.md"))
    for p in text_paths:
        body = p.read_text("utf-8-sig", errors="replace")
        hits = []
        for number, line in enumerate(body.splitlines(), 1):
            if re.search(r"\.py\b|python\s+-m", line, re.I):
                hits.append({"line": number, "text": line.strip()[:600]})
        if hits:
            references.append({"path": str(p.relative_to(ROOT)), "references": hits})
    for path, item in records.items():
        item["callers"] = sorted(k for k,v in records.items() if any(x["target"] == path for x in v["imports"]))
        item["workflows"] = [k for k,v in closures.items() if path in v]
        item["static_scope"] = "reachable_keep_pending_branch_review" if path in retained else "outside_static_closure_not_deletion_evidence"
    archive_names = {
        "run_a1s_natural_epoch.py", "run_a1s_m00_residual_correction.py",
        "prepare_m00_residual_reliability.py", "run_m00_condition_diagnostic.py",
        "run_m00_low_budget.py", "run_m00_luna_repair.py", "test_a1s7_m00_b.py",
        "test_a1s8_m00_preserve.py", "test_natural_epoch_sampler.py", "test_m00_residual_correction.py"}
    keep_extra = {"isr_evaluation/background_m00_eval.py", "inr_modules/data_managers/profile_split.py",
                  "inr_modules/evaluate_giro_peak.py"}
    review_tests = {"test_profile_split_a1s2.py", "test_isr_model_query.py", "test_giro_shared_full30.py",
                    "test_density_plot_units.py", "test_qc_profiles.py", "test_resume_schedule.py",
                    "test_run66_losses.py", "test_v14_fixed_query_regression.py", "test_v14_gram_calibration.py",
                    "test_m2v_physical_localization.py", "test_m2w_domain.py", "test_isr_evaluation_qc.py",
                    "test_isr_loader_geometry.py", "test_qcv2_background_contract.py",
                    "traditional_static_etkf/test_core.py", "traditional_static_etkf/test_runner.py", "verify_profile_grouping.py"}
    for path, item in records.items():
        if path in retained or path in keep_extra:
            disposition = "KEEP"
            reason = "Entry/import closure or reviewed shared evaluation dependency; internal branches reviewed separately"
        elif path in archive_names:
            disposition = "ARCHIVE_CANDIDATE"
            reason = "User-approved separate later research family; migrate only after backup and caller checks"
        elif path in review_tests:
            disposition = "KEEP_TEST"
            reason = "Regression coverage of retained workflow; review individual assertions before changes"
        else:
            disposition = "RETAIN_PENDING_EVIDENCE"
            reason = "Outside entry closure alone is not deletion evidence; historical/manual/shared uses not excluded"
        item["disposition"] = disposition
        item["disposition_reason"] = reason
    duplicate_dirs = []
    for directory in ROOT.iterdir():
        base = next((name for name in PACKAGES if directory.name.startswith(name + " ")), None)
        if base and directory.is_dir():
            items = []
            for p in directory.rglob("*.py"):
                rel = p.relative_to(directory)
                active = ROOT / base / rel
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                match = active.is_file() and hashlib.sha256(active.read_bytes()).hexdigest() == digest
                items.append({"relative":rel.as_posix(),"sha256":digest,"active_exists":active.is_file(),"same_as_active":match})
            duplicate_dirs.append({"path":str(directory.relative_to(ROOT)),"files":items,
                                   "disposition":"RETAIN_PENDING_EVIDENCE", "reason":"Similar directory name does not establish duplicate identity or lack of external users"})
    shell_entries = []
    for p in sorted(ROOT.glob("*.ps1")):
        candidate = any(term in p.stem for term in ("a1s7", "a1s8", "m00_residual", "a1s_natural"))
        shell_entries.append({"path":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                              "disposition":"ARCHIVE_CANDIDATE" if candidate else "RETAIN_PENDING_EVIDENCE"})
    result = {"stage": "S2", "method": "AST plus root command/document references; no project imports or runtime execution",
              "source_confirmed_by": "User N8D64 rerun confirmation, 2026-10-09", "files": records,
              "entrypoints": entrypoints, "closures": closures, "root_command_references": references,
              "duplicate_directory_candidates": duplicate_dirs, "shell_entries": shell_entries,
              "parameter_branch_inventory": {"report": "zero_config_review.md", "json": "zero_config_inventory.json", "csv": "zero_config_inventory.csv",
                                               "scope": "Effective zero/false branches supplement file-level retention"},
              "limitations": ["Conditional imports counted conservatively", "Dynamic paths require manual review",
                              "External source generators mapped separately from S1 lineage", "No observed runtime call coverage"],
              "git_tracked_status": subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT).decode("utf-8").strip()}
    assert module_name(Path("inr_modules/mdia/__init__.py")) == "inr_modules.mdia"
    assert "inr_modules/mdia/train_fsia.py" in closures["train"]
    assert "inr_modules/mdia/fsia_model.py" in closures["train"]
    (OUT / "scope_dependencies.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "scope_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "symbol_scope", "responsibility", "callers", "workflows", "config_get_keys", "serialization_review", "evidence", "disposition", "reason", "validation", "parameter_branch_review"])
        for path, item in records.items():
            writer.writerow([path, "file; symbol detail in scope_dependencies.json", item["summary"], ";".join(item["callers"]),
                             ";".join(item["workflows"]), ";".join(item["config_get_keys"]),
                             "See s2_compat_data_review.md; do not rename state keys/module paths without validation",
                             "SHA256=" + item["sha256"] + "; AST import line evidence in scope_dependencies.json",
                             item["disposition"], item["disposition_reason"], "S3 baseline and S4/S5 checks per scope_inventory.md",
                             "zero_config_review.md / zero_config_inventory.json" if path in ("main_fsia.py", "inr_modules/config_mdia.py", "inr_modules/mdia/train_fsia.py", "inr_modules/mdia/fsia_model.py") else ""])
    print(json.dumps({"files": len(records), "reachable":len(retained),"closures":{k:len(v) for k,v in closures.items()},
                      "outside_closure": sorted(set(records)-retained), "dynamic_files": [k for k,v in records.items() if v['dynamic_calls']]},ensure_ascii=True))


if __name__ == "__main__":
    main()
