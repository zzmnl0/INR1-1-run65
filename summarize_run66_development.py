"""Apply the fixed run66 development qualification gates to epoch reports."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_satellite_development import _sha256


SOURCES = ("FY", "COSMIC")
MODES = ("M10", "M01", "M11")
OBSERVATION_SOURCE = {"M10": "FY", "M01": "COSMIC", "M11": "JOINT"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    epochs = []
    failure_counts = Counter()
    for epoch in range(6, 11):
        path = input_dir / f"development_epoch{epoch:02d}_report.json"
        with path.open(encoding="utf-8") as stream:
            report = json.load(stream)
        source = report["sources"]
        fy, cosmic = source["FY"]["modes"], source["COSMIC"]["modes"]
        failed_strata = []
        for target_source in SOURCES:
            for mode in MODES:
                for cell, values in source[target_source]["strata"][mode].items():
                    if (values["direction_estimable"]
                            and values["direction_fraction"] < 0.55):
                        key = (target_source, OBSERVATION_SOURCE[mode], cell)
                        failure_counts[key] += 1
                        failed_strata.append({
                            "target_source": target_source,
                            "observation_source": OBSERVATION_SOURCE[mode],
                            "cell": cell,
                            "direction_fraction": values["direction_fraction"],
                            "profiles": values["direction_profiles"],
                            "dates": values["direction_dates"],
                        })
        self_negative = {
            "FY": fy["M10"]["negative_correct_fraction"],
            "COSMIC": cosmic["M01"]["negative_correct_fraction"],
        }
        cross_direction = {
            "FY_target_COSMIC_observation": fy["M01"]["direction_fraction"],
            "COSMIC_target_FY_observation": cosmic["M10"]["direction_fraction"],
        }
        rmse = {
            target: {
                mode: source[target]["modes"][mode]["rmse"]
                for mode in ("M00", *MODES)
            }
            for target in SOURCES
        }
        single_rmse_pass = all(
            rmse[target][mode] <= 1.01 * rmse[target]["M00"]
            for target in SOURCES for mode in ("M10", "M01"))
        joint_rmse_pass = all(
            rmse[target]["M11"] <= 1.01 * min(
                rmse[target]["M10"], rmse[target]["M01"])
            for target in SOURCES)
        gates = {
            "self_negative_response": min(self_negative.values()) >= 0.70,
            "cross_source_direction": min(cross_direction.values()) >= 0.60,
            "all_estimable_strata": not failed_strata,
            "single_source_rmse": single_rmse_pass,
            "joint_rmse": joint_rmse_pass,
            "hard_invariants": bool(report["passed_hard_invariants"]),
        }
        epochs.append({
            "epoch": epoch,
            "report": {"path": str(path), "sha256": _sha256(path)},
            "self_negative_response": self_negative,
            "cross_source_direction": cross_direction,
            "failed_strata_count": len(failed_strata),
            "failed_strata": failed_strata,
            "profile_rmse": rmse,
            "gates": gates,
            "qualified": all(gates.values()),
            "equal_weight_joint_rmse": 0.5 * (
                rmse["FY"]["M11"] + rmse["COSMIC"]["M11"]),
            "direction_attribution": {
                target: {
                    observation: source[target]["direction_attribution"][
                        observation]["global"]
                    for observation in SOURCES
                }
                for target in SOURCES
            },
        })
    qualified = [row for row in epochs if row["qualified"]]
    selected = min(
        qualified, key=lambda row: row["equal_weight_joint_rmse"]
    )["epoch"] if qualified else None
    persistent = [
        {
            "target_source": key[0],
            "observation_source": key[1],
            "cell": key[2],
            "failed_epochs": count,
        }
        for key, count in sorted(failure_counts.items()) if count == 5
    ]
    summary = {
        "schema_version": 1,
        "purpose": "M2-N ISR-blind development epoch qualification",
        "research_period": "2024-09",
        "thresholds": {
            "self_negative_response_minimum": 0.70,
            "cross_source_direction_minimum": 0.60,
            "estimable_stratum_direction_minimum": 0.55,
            "single_source_rmse_maximum_vs_M00": 1.01,
            "joint_rmse_maximum_vs_best_single": 1.01,
        },
        "data_boundaries": {
            "partition": "development",
            "locked_test_accessed": False,
            "isr_accessed": False,
        },
        "epochs": epochs,
        "qualified_epochs": [row["epoch"] for row in qualified],
        "selected_epoch": selected,
        "persistent_failed_strata": persistent,
        "conclusion": (
            "M2-N通过全部development资格门禁"
            if qualified else "M2-N对称互易修复不足以通过方向资格门禁"),
        "next_step": (
            "进入三折日期阻断内部验收"
            if qualified else (
                "保持对称核与R/N8不变，先审计端点自身背景上下文缺失及"
                "innovation方向上限；不进入ISR")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
