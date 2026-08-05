"""Summarize run66 development reports with the frozen qualification gates."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


THRESHOLDS = {
    "self_negative_response_minimum": 0.70,
    "cross_source_direction_minimum": 0.60,
    "estimable_stratum_direction_minimum": 0.55,
    "single_source_rmse_maximum_vs_M00": 1.01,
    "joint_rmse_maximum_vs_best_single": 1.01,
}
MODE_SOURCE = {"M10": "FY", "M01": "COSMIC", "M11": "JOINT"}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed_strata(report):
    failed = []
    minimum = THRESHOLDS["estimable_stratum_direction_minimum"]
    for target_source, source_report in report["sources"].items():
        for mode, observation_source in MODE_SOURCE.items():
            for cell, values in source_report["strata"][mode].items():
                direction = values["direction_fraction"]
                if (values["direction_estimable"]
                        and (direction is None or direction < minimum)):
                    failed.append({
                        "target_source": target_source,
                        "observation_source": observation_source,
                        "cell": cell,
                        "direction_fraction": direction,
                        "profiles": values["direction_profiles"],
                        "dates": values["direction_dates"],
                    })
    return failed


def _epoch_summary(epoch, path, report):
    fy = report["sources"]["FY"]
    cosmic = report["sources"]["COSMIC"]
    self_response = {
        "FY": fy["modes"]["M10"]["negative_correct_fraction"],
        "COSMIC": cosmic["modes"]["M01"]["negative_correct_fraction"],
    }
    cross_direction = {
        "FY_target_COSMIC_observation": (
            fy["modes"]["M01"]["direction_fraction"]),
        "COSMIC_target_FY_observation": (
            cosmic["modes"]["M10"]["direction_fraction"]),
    }
    failed = _failed_strata(report)
    rmse = {
        source: {
            mode: values["rmse"]
            for mode, values in source_report["modes"].items()
        }
        for source, source_report in report["sources"].items()
    }
    gates = {
        "self_negative_response": all(
            value >= THRESHOLDS["self_negative_response_minimum"]
            for value in self_response.values()),
        "cross_source_direction": all(
            value >= THRESHOLDS["cross_source_direction_minimum"]
            for value in cross_direction.values()),
        "all_estimable_strata": not failed,
        "single_source_rmse": all(
            values[mode] <= (
                values["M00"]
                * THRESHOLDS["single_source_rmse_maximum_vs_M00"])
            for values in rmse.values() for mode in ("M10", "M01")),
        "joint_rmse": all(
            values["M11"] <= min(values["M10"], values["M01"]) *
            THRESHOLDS["joint_rmse_maximum_vs_best_single"]
            for values in rmse.values()),
        "hard_invariants": bool(report["passed_hard_invariants"]),
    }
    return {
        "epoch": epoch,
        "report": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "self_negative_response": self_response,
        "cross_source_direction": cross_direction,
        "failed_strata_count": len(failed),
        "failed_strata": failed,
        "profile_rmse": rmse,
        "gates": gates,
        "qualified": all(gates.values()),
        "equal_weight_joint_rmse": 0.5 * (
            rmse["FY"]["M11"] + rmse["COSMIC"]["M11"]),
        "direction_attribution": {
            target: {
                observation: values["global"]
                for observation, values in source_report[
                    "direction_attribution"].items()
            }
            for target, source_report in report["sources"].items()
        },
    }


def summarize(report_dir, epochs, purpose):
    rows = []
    for epoch in epochs:
        path = report_dir / f"development_epoch{epoch:02d}_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_epoch_summary(epoch, path, report))
    qualified = [row for row in rows if row["qualified"]]
    selected = min(
        qualified, key=lambda row: row["equal_weight_joint_rmse"],
        default=None)
    failure_counts = Counter(
        (item["target_source"], item["observation_source"], item["cell"])
        for row in rows for item in row["failed_strata"])
    persistent = [
        {
            "target_source": key[0],
            "observation_source": key[1],
            "cell": key[2],
            "failed_epochs": count,
        }
        for key, count in sorted(failure_counts.items())
        if count == len(rows)
    ]
    return {
        "schema_version": 1,
        "purpose": purpose,
        "research_period": "2024-09",
        "thresholds": THRESHOLDS,
        "data_boundaries": {
            "partition": "development",
            "locked_test_accessed": False,
            "isr_accessed": False,
        },
        "epochs": rows,
        "qualified_epochs": [row["epoch"] for row in qualified],
        "selected_epoch": None if selected is None else selected["epoch"],
        "persistent_failed_strata": persistent,
        "conclusion": (
            "At least one epoch passed all frozen development gates."
            if selected else "No epoch passed all frozen development gates."),
        "next_step": (
            "Proceed to three-fold date-blocked internal acceptance."
            if selected else
            "Attribute pre-inversion failures before changing losses, R, or N8."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=range(6, 11))
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    summary = summarize(args.report_dir.resolve(), args.epochs, args.purpose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
