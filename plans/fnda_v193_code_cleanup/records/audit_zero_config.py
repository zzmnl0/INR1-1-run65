"""Record zero/false configuration and runtime resolution without importing the project."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inactive(value):
    return value is False or (type(value) in (int, float) and value == 0)


def main():
    source = ROOT / "inr_modules/config_mdia.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    config_node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "CONFIG_MDIA" for t in n.targets))
    defaults = {}
    for key, value in zip(config_node.keys, config_node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            try:
                defaults[key.value] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
    identity = load(OUT / "model_identity.json")
    runs = {"main": identity["main_run"]}
    for name, row in identity["sensitivity_runs"].items():
        manifest = load(row["manifest"]["path"])
        summary = load(row["summary"]["path"])
        runs[name] = {"initial_config": manifest["config"],
                      "resolved_training": manifest.get("resolved_training", {}), "summary_fields": summary}
    keys = {k for k, v in defaults.items() if inactive(v)}
    for run in runs.values():
        keys.update(k for k, v in run["initial_config"].items() if inactive(v))
    keys.update(("w_background_stratum_bias", "w_low_altitude_background_iri", "w_low_altitude_analysis_increment",
                 "use_observation_gram_loss", "use_sw_freq"))
    file_names = ("inr_modules/config_mdia.py", "main_fsia.py", "inr_modules/mdia/train_fsia.py", "inr_modules/mdia/fsia_model.py")
    references = {key: [] for key in keys}
    for name in file_names:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in keys:
                references[node.value].append({"file": name, "line": node.lineno})
    candidates = {"use_covariance_moment_loss", "use_empirical_covariance_loss", "use_direction_loss"}
    active = {"w_background_stratum_bias", "background_stratum_bias_gradient_target",
              "w_low_altitude_background_iri", "w_low_altitude_analysis_increment", "low_altitude_anchor_profiles_per_source",
              "use_observation_gram_loss", "use_sw_freq"}
    rows = []
    for key in sorted(keys):
        if key in active:
            disposition = "KEEP_EFFECTIVE_ACTIVE"
        elif key in candidates:
            disposition = "ARCHIVE_DISABLED_BRANCH_CANDIDATE"
        elif key.startswith("m00_") or key in ("background_trust_gate_enabled", "background_mix_balanced_fraction", "shared_backbone_unfreeze", "localization_enabled"):
            disposition = "REVIEW_WITH_BRANCH_GROUP_KEEP_NEUTRAL_BEHAVIOR"
        else:
            disposition = "KEEP_CONTROL_OR_PENDING_EFFECTIVE_USE"
        run_values = {}
        for name, run in runs.items():
            resolved = run["resolved_training"]
            run_values[name] = {"initial_value": run["initial_config"].get(key, "NOT_RECORDED"),
                               "resolved_same_key": resolved.get(key, "NOT_RECORDED"),
                               "resolved_weights": {k: v for k, v in resolved.items() if k.startswith("resolved_") and "weight" in k},
                               "summary_same_key": run["summary_fields"].get(key, "NOT_RECORDED")}
        rows.append({"key": key, "current_literal_default": defaults.get(key, "NOT_LITERAL"),
                     "runs": run_values, "references": references[key], "disposition": disposition})
    payload = {"scope": "Current CONFIG_MDIA literals plus main and three sensitivity run initial/resolved/summary records",
               "rows": rows, "source_hashes": {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in file_names},
               "inactive_compute_finding": {"id":"Z02/C12", "file":"inr_modules/mdia/train_fsia.py",
                   "producer":"_paired_analysis_losses:2727-2731 computes direction loss unconditionally",
                   "consumer":"Gram calibration at 3282/3347 discards direction return",
                   "action":"Candidate removal with shared return/caller update and Gram baseline verification; not implemented"},
               "limitations": ["No runtime execution", "Zero values in tensors, optimizer counters or time coordinates are not disabled modules",
                               "Resolved runtime values and retained workflows govern removal, not literal defaults"]}
    assert all(run["initial_config"][key] is False for run in runs.values()
               for key in ("use_direction_loss", "use_covariance_moment_loss", "use_empirical_covariance_loss"))
    assert runs["main"]["resolved_training"]["resolved_background_stratum_bias_weight"] > 0
    assert runs["main"]["initial_config"]["w_low_altitude_background_iri"] > 0
    (OUT / "zero_config_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (OUT / "zero_config_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "current_default", *runs.keys(), "disposition", "references"])
        for row in rows:
            writer.writerow([row["key"], row["current_literal_default"],
                             *(json.dumps(row["runs"][name], ensure_ascii=False) for name in runs),
                             row["disposition"], ";".join(f"{x['file']}:{x['line']}" for x in row["references"])])
    print(json.dumps({"parameters":len(rows),"runs":len(runs),"assertions":"PASS","model_executed":False}))


if __name__ == "__main__":
    main()
