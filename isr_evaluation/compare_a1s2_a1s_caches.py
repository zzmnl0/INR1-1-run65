"""Compare aligned A1-S2 and A1-S ISR caches without rerunning inference."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inr_modules.mdia.evaluation_stats import paired_group_bootstrap
from isr_evaluation.background_m00_eval import _metrics, _reporting_masks


ROOT = Path("isr_validation_outputs")
DEFAULTS = {
    "canonical_cache": ROOT / "traditional-static-etkf-raw-iri-full-isr" / "isr_etkf_cache.npz",
    "a1s2_m00_cache": ROOT / "m2w-a1s2-background-isr" / "isr_m00_cache.npz",
    "a1s2_m11_cache": ROOT / "m2w-a1s2-analysis10-isr" / "isr_evaluation_cache.npz",
    "a1s_m00_cache": ROOT / "m2w-background-stratum-bias-a1s-m00-isr" / "isr_m00_cache.npz",
    "a1s_m11_cache": ROOT / "m2w-background-stratum-bias-a1s-analysis10-isr" / "isr_evaluation_cache.npz",
    "output_dir": ROOT / "m2w-a1s2-analysis10-isr",
}


def _load(path, required):
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(set(required) - set(source.files))
        if missing:
            raise ValueError(f"{path} lacks required fields: {missing}")
        return {name: source[name] for name in source.files}


def _stage_summary(metric_rows, candidate, baseline):
    indexed = {
        (row["station"], row["stratum"], row["model"]): row
        for row in metric_rows
    }
    sites = {}
    for station in ("Jicamarca", "PokerFlat"):
        candidate_row = indexed[(station, "main_200_500_all", candidate)]
        baseline_row = indexed[(station, "main_200_500_all", baseline)]
        sites[station] = {
            "candidate": candidate_row,
            "baseline": baseline_row,
            "abs_bias_delta_candidate_minus_baseline": (
                abs(candidate_row["bias"]) - abs(baseline_row["bias"])),
            "rmse_delta_candidate_minus_baseline": (
                candidate_row["rmse"] - baseline_row["rmse"]),
            "ccc_delta_candidate_minus_baseline": (
                candidate_row["ccc"] - baseline_row["ccc"]),
        }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "sites": sites,
        "both_sites_abs_bias_improved": all(
            row["abs_bias_delta_candidate_minus_baseline"] < 0.0
            for row in sites.values()),
        "rmse_non_degraded": all(
            row["rmse_delta_candidate_minus_baseline"] <= 0.01
            for row in sites.values()),
        "ccc_non_degraded": all(
            row["ccc_delta_candidate_minus_baseline"] >= -0.01
            for row in sites.values()),
    }


def run(args):
    keys = ("station", "key", "unit_id", "altitude_km")
    canonical = _load(args.canonical_cache, keys + (
        "timestamp", "lon", "observation_log10", "ETKF_log10",
        "raw_iri_continuous_log10"))
    caches = {
        "a1s2_m00": _load(args.a1s2_m00_cache, keys + (
            "observation_log10", "candidate_log10")),
        "a1s2_m11": _load(args.a1s2_m11_cache, keys + (
            "observation_log10", "M11_log10")),
        "a1s_m00": _load(args.a1s_m00_cache, keys + (
            "observation_log10", "M00_A1S_log10")),
        "a1s_m11": _load(args.a1s_m11_cache, keys + (
            "observation_log10", "M11_log10")),
    }

    row_count = len(canonical["key"])
    if row_count != 206230 or len(np.unique(canonical["key"])) != row_count:
        raise ValueError("canonical ISR cache row count or key uniqueness differs")
    stations = sorted(str(value) for value in np.unique(canonical["station"]))
    if stations != ["Jicamarca", "PokerFlat"]:
        raise ValueError(f"unexpected canonical stations: {stations}")

    alignment = {}
    observation_max_abs_difference = {}
    for name, cache in caches.items():
        if len(cache["key"]) != row_count:
            raise ValueError(f"{name} row count differs from canonical cache")
        alignment[name] = {}
        for key in keys:
            equal = bool(np.array_equal(cache[key], canonical[key]))
            alignment[name][key] = equal
            if not equal:
                raise ValueError(f"{name} canonical alignment differs for {key}")
        observation_max_abs_difference[name] = float(np.max(np.abs(
            cache["observation_log10"].astype(np.float64)
            - canonical["observation_log10"].astype(np.float64))))

    predictions = {
        "M00_A1S2": caches["a1s2_m00"]["candidate_log10"].astype(np.float64),
        "M11_A1S2": caches["a1s2_m11"]["M11_log10"].astype(np.float64),
        "M00_A1S": caches["a1s_m00"]["M00_A1S_log10"].astype(np.float64),
        "M11_A1S": caches["a1s_m11"]["M11_log10"].astype(np.float64),
        "RAW_IRI": canonical["raw_iri_continuous_log10"].astype(np.float64),
        "Static_ETKF": canonical["ETKF_log10"].astype(np.float64),
    }
    observation = canonical["observation_log10"].astype(np.float64)
    finite_counts = {"observation": int(np.isfinite(observation).sum())}
    finite_counts.update({
        name: int(np.isfinite(values).sum())
        for name, values in predictions.items()
    })
    if any(count != row_count for count in finite_counts.values()):
        raise ValueError(f"non-finite canonical comparison values: {finite_counts}")

    local_time = np.remainder(
        canonical["timestamp"].astype(np.float64) / 3600.0
        + canonical["lon"].astype(np.float64) / 15.0, 24.0)
    altitude = canonical["altitude_km"].astype(np.float64)
    unit_ids = canonical["unit_id"]
    pairs = (
        ("M00_A1S2", "M00_A1S"),
        ("M00_A1S2", "RAW_IRI"),
        ("M00_A1S2", "Static_ETKF"),
        ("M11_A1S2", "M11_A1S"),
        ("M11_A1S2", "RAW_IRI"),
        ("M11_A1S2", "Static_ETKF"),
    )
    metric_rows = []
    bootstrap_rows = []
    stratum_names = None
    for station in stations:
        station_mask = canonical["station"] == station
        masks = list(_reporting_masks(altitude, local_time))
        names = [name for name, _ in masks]
        if stratum_names is None:
            stratum_names = names
        elif names != stratum_names:
            raise ValueError("reporting strata changed between stations")
        for stratum, stratum_mask in masks:
            selected = station_mask & stratum_mask
            if selected.sum() < 2 or np.unique(unit_ids[selected]).size < 2:
                raise ValueError(f"insufficient ISR units for {station}/{stratum}")
            for model, values in predictions.items():
                metric_rows.append({
                    "station": station,
                    "stratum": stratum,
                    "model": model,
                    **_metrics(observation[selected], values[selected]),
                })
            for candidate, baseline in pairs:
                bootstrap_rows.append({
                    "station": station,
                    "stratum": stratum,
                    "candidate": candidate,
                    "baseline": baseline,
                    **paired_group_bootstrap(
                        observation[selected], predictions[candidate][selected],
                        predictions[baseline][selected], unit_ids[selected],
                        replicates=2000, seed=42),
                })

    if len(stratum_names) != 15 or len(metric_rows) != 180 or len(bootstrap_rows) != 180:
        raise ValueError("unexpected unified comparison row count")
    summaries = {
        "background_m00": _stage_summary(metric_rows, "M00_A1S2", "M00_A1S"),
        "analysis_m11": _stage_summary(metric_rows, "M11_A1S2", "M11_A1S"),
    }
    for summary in summaries.values():
        summary["passed"] = bool(
            summary["both_sites_abs_bias_improved"]
            and summary["rmse_non_degraded"]
            and summary["ccc_non_degraded"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "isr_a1s2_vs_a1s_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "station", "stratum", "model", "n", "bias", "rmse", "mae",
            "pearson_r", "ccc", "prediction_std", "target_std",
            "std_ratio", "spread_error", "centered_rmse"))
        writer.writeheader()
        writer.writerows(metric_rows)

    report = {
        "schema_version": 1,
        "scope": "Strict canonical-cache comparison; no inference or locked-test",
        "canonical_cache": str(Path(args.canonical_cache).resolve()),
        "models": list(predictions),
        "strata": stratum_names,
        "metrics": metric_rows,
        "profile_block_bootstrap": {
            "replicates": 2000,
            "seed": 42,
            "group": "station-specific unit_id",
            "comparisons": bootstrap_rows,
        },
        "main_200_500_summary": summaries,
    }
    (output_dir / "isr_a1s2_vs_a1s_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    text_lines = [
        "A1-S2 vs A1-S strict canonical ISR comparison",
        f"Rows={row_count}; strata={len(stratum_names)}; bootstrap=2000 seed=42",
    ]
    indexed = {
        (row["station"], row["stratum"], row["model"]): row
        for row in metric_rows
    }
    for station in stations:
        text_lines.append(f"\n{station} main_200_500_all")
        for model in predictions:
            row = indexed[(station, "main_200_500_all", model)]
            text_lines.append(
                f"  {model}: Bias={row['bias']:.6f} RMSE={row['rmse']:.6f} "
                f"MAE={row['mae']:.6f} CCC={row['ccc']:.6f}")
    for stage, summary in summaries.items():
        text_lines.append(f"\n{stage}: {'PASS' if summary['passed'] else 'FAIL'}")
        for station, row in summary["sites"].items():
            text_lines.append(
                f"  {station}: |Bias|delta={row['abs_bias_delta_candidate_minus_baseline']:.6f} "
                f"RMSEdelta={row['rmse_delta_candidate_minus_baseline']:.6f} "
                f"CCCdelta={row['ccc_delta_candidate_minus_baseline']:.6f}")
    (output_dir / "isr_a1s2_vs_a1s_report.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8")

    audit = {
        "row_count": row_count,
        "unique_key_count": int(len(np.unique(canonical["key"]))),
        "station_counts": {
            station: int(np.sum(canonical["station"] == station))
            for station in stations
        },
        "station_unit_counts": {
            station: int(np.unique(unit_ids[canonical["station"] == station]).size)
            for station in stations
        },
        "canonical_alignment": alignment,
        "observation_max_abs_difference": observation_max_abs_difference,
        "finite_counts": finite_counts,
        "metric_row_count": len(metric_rows),
        "bootstrap_comparison_count": len(bootstrap_rows),
        "locked_test_read": False,
        "hash_or_sha_validation": False,
    }
    (output_dir / "isr_a1s2_vs_a1s_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, audit


def _run_full30(args):
    """Compare one paired FSIA cache with one shared-pool ETKF cache."""
    required_paired = (
        "station", "key", "observation_log10", "candidate_M11_log10",
        "candidate_M00_log10", "baseline_M11_log10", "baseline_M00_log10",
        "altitude_km", "unit_id")
    required_etkf = (
        "station", "key", "observation_log10", "ETKF_log10",
        "raw_iri_continuous_log10", "timestamp", "lon", "altitude_km",
        "unit_id")
    if args.paired_cache is not None:
        paired = _load(args.paired_cache, required_paired)
    else:
        required_model = (
            "station", "key", "observation_log10", "M11_log10",
            "M00_log10", "altitude_km", "unit_id")
        candidate = _load(args.a1s2_cache, required_model)
        baseline = _load(args.a1s_cache, required_model)
        paired = {
            "station": candidate["station"], "key": candidate["key"],
            "observation_log10": candidate["observation_log10"],
            "candidate_M11_log10": candidate["M11_log10"],
            "candidate_M00_log10": candidate["M00_log10"],
            "baseline_M11_log10": baseline["M11_log10"],
            "baseline_M00_log10": baseline["M00_log10"],
            "altitude_km": candidate["altitude_km"],
            "unit_id": candidate["unit_id"],
        }
        for field in ("station", "key", "unit_id", "altitude_km"):
            if not np.array_equal(np.asarray(candidate[field]).astype(str),
                                  np.asarray(baseline[field]).astype(str)):
                raise ValueError(f"A1-S2/A1-S full30 alignment differs for {field}")
        baseline_obs_delta = np.asarray(candidate["observation_log10"], dtype=np.float64) - np.asarray(
            baseline["observation_log10"], dtype=np.float64)
        if float(np.nanmax(np.abs(baseline_obs_delta))) > 1e-6:
            raise ValueError("A1-S2/A1-S full30 observation columns differ")
    etkf = _load(args.etkf_cache, required_etkf)
    expected_rows = 206230
    if len(paired["key"]) != expected_rows or len(etkf["key"]) != expected_rows:
        raise ValueError("full30 cache row count must be 206230")
    if len(np.unique(paired["key"])) != expected_rows:
        raise ValueError("FSIA paired cache contains duplicate keys")
    if len(np.unique(etkf["key"])) != expected_rows:
        raise ValueError("ETKF full30 cache contains duplicate keys")

    alignment = {}
    for field in ("station", "key", "unit_id", "altitude_km"):
        left = np.asarray(paired[field]).astype(str)
        right = np.asarray(etkf[field]).astype(str)
        alignment[field] = bool(np.array_equal(left, right))
        if not alignment[field]:
            raise ValueError(f"full30 cache alignment differs for {field}")
    observation_difference = np.asarray(paired["observation_log10"], dtype=np.float64) - np.asarray(
        etkf["observation_log10"], dtype=np.float64)
    max_observation_difference = float(np.nanmax(np.abs(observation_difference)))
    if not np.isfinite(max_observation_difference) or max_observation_difference > 1e-6:
        raise ValueError("full30 cache observation columns differ")

    candidate_prefix = str(getattr(args, "candidate_prefix", "A1S2_FULL30"))
    baseline_prefix = str(getattr(args, "baseline_prefix", "A1S_FULL30"))
    predictions = {
        f"M00_{baseline_prefix}": np.asarray(paired["baseline_M00_log10"], dtype=np.float64),
        f"M11_{baseline_prefix}": np.asarray(paired["baseline_M11_log10"], dtype=np.float64),
        f"M00_{candidate_prefix}": np.asarray(paired["candidate_M00_log10"], dtype=np.float64),
        f"M11_{candidate_prefix}": np.asarray(paired["candidate_M11_log10"], dtype=np.float64),
        "RAW_IRI": np.asarray(etkf["raw_iri_continuous_log10"], dtype=np.float64),
        "Static_ETKF_FULL30": np.asarray(etkf["ETKF_log10"], dtype=np.float64),
    }
    observation = np.asarray(paired["observation_log10"], dtype=np.float64)
    finite_counts = {"observation": int(np.isfinite(observation).sum())}
    finite_counts.update({name: int(np.isfinite(values).sum())
                          for name, values in predictions.items()})
    if any(count != expected_rows for count in finite_counts.values()):
        raise ValueError(f"full30 comparison contains non-finite values: {finite_counts}")

    stations = sorted(str(value) for value in np.unique(paired["station"]))
    if stations != ["Jicamarca", "PokerFlat"]:
        raise ValueError(f"unexpected full30 stations: {stations}")
    timestamp = np.asarray(etkf["timestamp"], dtype=np.float64)
    longitude = np.asarray(etkf["lon"], dtype=np.float64)
    local_time = np.remainder(timestamp / 3600.0 + longitude / 15.0, 24.0)
    altitude = np.asarray(paired["altitude_km"], dtype=np.float64)
    unit_ids = np.asarray(paired["unit_id"])
    masks = list(_reporting_masks(altitude, local_time))
    if len(masks) != 15:
        raise ValueError("unexpected ISR reporting stratum count")

    pairs = (
        (f"M00_{candidate_prefix}", f"M00_{baseline_prefix}"),
        (f"M00_{candidate_prefix}", "RAW_IRI"),
        (f"M00_{candidate_prefix}", "Static_ETKF_FULL30"),
        (f"M11_{candidate_prefix}", f"M11_{baseline_prefix}"),
        (f"M11_{candidate_prefix}", "RAW_IRI"),
        (f"M11_{candidate_prefix}", "Static_ETKF_FULL30"),
    )
    metric_rows, bootstrap_rows = [], []
    for station in stations:
        station_mask = np.asarray(paired["station"]).astype(str) == station
        for stratum, stratum_mask in masks:
            selected = station_mask & stratum_mask
            if selected.sum() < 2 or np.unique(unit_ids[selected]).size < 2:
                raise ValueError(f"insufficient ISR units for {station}/{stratum}")
            for model, values in predictions.items():
                metric_rows.append({"station": station, "stratum": stratum,
                                    "model": model,
                                    **_metrics(observation[selected], values[selected])})
            for candidate, baseline in pairs:
                bootstrap_rows.append({
                    "station": station, "stratum": stratum,
                    "candidate": candidate, "baseline": baseline,
                    **paired_group_bootstrap(
                        observation[selected], predictions[candidate][selected],
                        predictions[baseline][selected], unit_ids[selected],
                        replicates=2000, seed=42),
                })
    if len(metric_rows) != 180 or len(bootstrap_rows) != 180:
        raise ValueError("unexpected full30 output row count")

    summaries = {
        "background_m00": _stage_summary(
            metric_rows, f"M00_{candidate_prefix}", f"M00_{baseline_prefix}"),
        "analysis_m11": _stage_summary(
            metric_rows, f"M11_{candidate_prefix}", f"M11_{baseline_prefix}"),
    }
    for summary in summaries.values():
        summary["passed"] = bool(
            summary["both_sites_abs_bias_improved"]
            and summary["rmse_non_degraded"]
            and summary["ccc_non_degraded"])

    output_dir = Path(args.full30_output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"full30 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "isr_a1s2_a1s_full30_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "station", "stratum", "model", "n", "bias", "rmse", "mae",
            "pearson_r", "ccc", "prediction_std", "target_std",
            "std_ratio", "spread_error", "centered_rmse"))
        writer.writeheader()
        writer.writerows(metric_rows)
    report = {
        "schema_version": 1,
        "scope": "Shared 30-day profile-pool paired ISR comparison",
        "paired_cache": (str(Path(args.paired_cache).resolve())
                         if args.paired_cache is not None else None),
        "a1s2_cache": (str(Path(args.a1s2_cache).resolve())
                       if args.paired_cache is None else None),
        "a1s_cache": (str(Path(args.a1s_cache).resolve())
                      if args.paired_cache is None else None),
        "etkf_cache": str(Path(args.etkf_cache).resolve()),
        "models": list(predictions),
        "candidate_prefix": candidate_prefix,
        "baseline_prefix": baseline_prefix,
        "strata": [name for name, _ in masks],
        "metrics": metric_rows,
        "profile_block_bootstrap": {
            "replicates": 2000, "seed": 42,
            "group": "station-specific unit_id", "comparisons": bootstrap_rows,
        },
        "main_200_500_summary": summaries,
    }
    (output_dir / "isr_a1s2_a1s_full30_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    indexed = {(row["station"], row["stratum"], row["model"]): row
               for row in metric_rows}
    text_lines = [
        "Shared full30 paired ISR comparison",
        f"Rows={expected_rows}; strata=15; models=6; bootstrap=2000 seed=42",
    ]
    for station in stations:
        text_lines.append(f"\n{station} main_200_500_all")
        for model in predictions:
            row = indexed[(station, "main_200_500_all", model)]
            text_lines.append(
                f"  {model}: Bias={row['bias']:.6f} RMSE={row['rmse']:.6f} "
                f"MAE={row['mae']:.6f} CCC={row['ccc']:.6f}")
    for stage, summary in summaries.items():
        text_lines.append(f"\n{stage}: {'PASS' if summary['passed'] else 'FAIL'}")
        for station, values in summary["sites"].items():
            text_lines.append(
                f"  {station}: |Bias|delta={values['abs_bias_delta_candidate_minus_baseline']:.6f} "
                f"RMSEdelta={values['rmse_delta_candidate_minus_baseline']:.6f} "
                f"CCCdelta={values['ccc_delta_candidate_minus_baseline']:.6f}")
    (output_dir / "isr_a1s2_a1s_full30_report.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8")
    audit = {
        "row_count": expected_rows,
        "unique_key_count": expected_rows,
        "station_counts": {station: int(np.sum(np.asarray(paired["station"]).astype(str) == station))
                            for station in stations},
        "canonical_alignment": alignment,
        "observation_max_abs_difference": max_observation_difference,
        "finite_counts": finite_counts,
        "metric_row_count": len(metric_rows),
        "bootstrap_comparison_count": len(bootstrap_rows),
        "observation_pool_semantics": "shared_full_30d_all_profiles_v1",
        "locked_test_read": False,
        "hash_or_sha_validation": False,
    }
    (output_dir / "isr_a1s2_a1s_full30_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, audit


def main():
    parser = argparse.ArgumentParser()
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default)
    parser.add_argument("--paired-cache", default=None,
                        help="FSIA paired cache from shared-full30 main ISR")
    parser.add_argument("--a1s2-cache", default=None,
                        help="full FSIA A1-S2 cache from shared-full30 ISR")
    parser.add_argument("--a1s-cache", default=None,
                        help="full FSIA A1-S cache from shared-full30 ISR")
    parser.add_argument("--etkf-cache", default=None,
                        help="Static ETKF cache from shared-full30 ISR")
    parser.add_argument(
        "--candidate-prefix", default="A1S2_FULL30",
        help="label prefix for candidate M00/M11 columns")
    parser.add_argument(
        "--baseline-prefix", default="A1S_FULL30",
        help="label prefix for baseline M00/M11 columns")
    parser.add_argument(
        "--full30-output-dir",
        default=ROOT / "m2w-a1s-a1s2-full30-comparison-isr",
        help="output directory for the six-model shared-full30 comparison")
    args = parser.parse_args()
    if args.etkf_cache is None:
        parser.error("--etkf-cache is required for full30 comparison")
    if args.paired_cache is not None and (args.a1s2_cache or args.a1s_cache):
        parser.error("use either --paired-cache or --a1s2-cache/--a1s-cache")
    if args.paired_cache is None and (args.a1s2_cache is None or args.a1s_cache is None):
        parser.error("supply --paired-cache or both full A1-S2 and A1-S caches")
    if args.paired_cache is not None or args.a1s2_cache is not None:
        _run_full30(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
