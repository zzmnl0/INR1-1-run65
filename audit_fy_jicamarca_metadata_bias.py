"""R1 metadata-only audit of FY/Jicamarca representativeness conflicts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from estimate_empirical_covariance import _deterministic_npz, _sha256


ROOT = Path(__file__).resolve().parent
G1_OUTPUT = (
    ROOT / "isr_validation_outputs"
    / "run66-direction-correction-g1-fy-jicamarca-representativeness"
)
OUTPUT = G1_OUTPUT / "metadata-only-audit"
FY_INDEX = Path(r"D:\FYsatellite\EDP_data\fy_202409_qc_v2_index.npz")
NUMERIC_METADATA = (
    "h_cut_km", "hmf2", "nmf2", "peak_count", "md", "delta",
    "global_topside_gradient", "local_topside_gradient", "fold_error",
    "kept_points", "range_rejected_points", "sza",
    "representative_lat", "representative_lon", "representative_time",
)


def _rank_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    stops = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
    return ranks


def _rank_correlation(left, right):
    left = _rank_average(left)
    right = _rank_average(right)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _bootstrap_rank_correlation(left, right, dates, replicates, seed):
    left = np.asarray(left)
    right = np.asarray(right)
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    groups = [np.flatnonzero(dates == day) for day in unique_dates]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = []
        for date_index in rng.integers(0, len(groups), size=len(groups)):
            group = groups[date_index]
            selected.append(group[rng.integers(0, len(group), size=len(group))])
        selected = np.concatenate(selected)
        values[replicate] = _rank_correlation(left[selected], right[selected])
    return np.quantile(values, [0.025, 0.975]).tolist()


def _profile_table(pairs, rho_median, index_path):
    selected = (
        (pairs["mode"] == "M10")
        & (pairs["support"] == "within_20km")
        & (pairs["rho"] <= rho_median)
        & pairs["high_confidence"]
    )
    profile_ids = np.unique(pairs["profile_id"][selected])
    if not len(profile_ids):
        raise ValueError("G1 primary cell has no high-confidence FY profiles")
    with np.load(index_path, allow_pickle=True) as index:
        index_ids = np.asarray(index["profile_id"], dtype=np.int64)
        positions = np.searchsorted(index_ids, profile_ids)
        if not np.array_equal(index_ids[positions], profile_ids):
            raise ValueError("G1 profile_id absent from FY QC index")
        paths = np.asarray(index["original_relative_path"]).astype(str)[positions]
        table = {
            "profile_id": profile_ids,
            "satellite": np.asarray([
                path.replace("\\", "/").split("/", 1)[0] for path in paths
            ]),
            "original_path": paths,
        }
        for field in NUMERIC_METADATA:
            values = np.asarray(index[field])[positions].astype(np.float64)
            if field == "nmf2":
                values = np.log10(np.maximum(values, 1.0))
            table[field] = values

    pair_fields = (
        "innovation_toward", "contribution_toward", "innovation",
        "desired", "rho", "horizontal_rho", "vertical_distance",
        "query_altitude", "local_time", "precision_mass",
    )
    for field in pair_fields:
        table[field] = np.empty(len(profile_ids), dtype=np.float64)
    table["pair_count"] = np.empty(len(profile_ids), dtype=np.int32)
    for row, profile_id in enumerate(profile_ids):
        mask = selected & (pairs["profile_id"] == profile_id)
        for field in pair_fields:
            table[field][row] = float(np.mean(pairs[field][mask]))
        table["pair_count"][row] = int(mask.sum())
    table["profile_date"] = np.asarray(
        [int(path.replace("\\", "/").split("/")[-1][:8])
         for path in table["original_path"]],
        dtype=np.int32,
    )
    table["night_hour_from_midnight"] = np.minimum(
        table["local_time"], 24.0 - table["local_time"]
    )
    table["conflict"] = table["innovation_toward"] < 0.5
    return table


def _numeric_associations(table, replicates, seed):
    outcome = table["innovation_toward"]
    dates = table["profile_date"]
    candidate_fields = (
        *NUMERIC_METADATA,
        "rho", "horizontal_rho", "vertical_distance",
        "query_altitude", "night_hour_from_midnight", "precision_mass",
    )
    rows = []
    for index, field in enumerate(candidate_fields):
        values = np.asarray(table[field], dtype=np.float64)
        finite = np.isfinite(values) & np.isfinite(outcome)
        if finite.sum() < 10:
            continue
        correlation = _rank_correlation(values[finite], outcome[finite])
        ci = _bootstrap_rank_correlation(
            values[finite], outcome[finite], dates[finite],
            replicates, seed + index,
        )
        jackknife = []
        for day in np.unique(dates[finite]):
            keep = finite & (dates != day)
            if keep.sum() >= 8:
                jackknife.append(
                    np.sign(_rank_correlation(values[keep], outcome[keep]))
                    == np.sign(correlation)
                )
        conflict = finite & table["conflict"]
        correct = finite & ~table["conflict"]
        rows.append({
            "field": field,
            "n_profiles": int(finite.sum()),
            "spearman": correlation,
            "ci95": ci,
            "leave_one_date_sign_fraction": (
                float(np.mean(jackknife)) if jackknife else None
            ),
            "conflict_median": (
                float(np.median(values[conflict])) if conflict.any() else None
            ),
            "nonconflict_median": (
                float(np.median(values[correct])) if correct.any() else None
            ),
            "exploratory_candidate": bool(
                abs(correlation) >= 0.40
                and ci[0] * ci[1] > 0.0
                and jackknife
                and np.mean(jackknife) >= 0.80
            ),
        })
    return sorted(rows, key=lambda row: abs(row["spearman"]), reverse=True)


def _categorical_summary(table, field):
    rows = {}
    values = table[field]
    for value in sorted(np.unique(values)):
        selected = values == value
        rows[str(value)] = {
            "n_profiles": int(selected.sum()),
            "direction_rate_mean": float(
                np.mean(table["innovation_toward"][selected])
            ),
            "conflict_profile_fraction": float(
                np.mean(table["conflict"][selected])
            ),
            "precision_mass": float(
                np.sum(table["precision_mass"][selected])
            ),
        }
    return rows


def _matched_differences(table, fields):
    conflict_rows = np.flatnonzero(table["conflict"])
    control_rows = set(np.flatnonzero(~table["conflict"]).tolist())
    matches = []
    for conflict in sorted(
        conflict_rows, key=lambda row: -table["precision_mass"][row]
    ):
        candidates = [
            row for row in control_rows
            if table["profile_date"][row] == table["profile_date"][conflict]
            and table["satellite"][row] == table["satellite"][conflict]
        ]
        if not candidates:
            continue
        distance = [
            abs(table["query_altitude"][row] - table["query_altitude"][conflict])
            / 20.0
            + abs(
                table["night_hour_from_midnight"][row]
                - table["night_hour_from_midnight"][conflict]
            )
            + abs(table["rho"][row] - table["rho"][conflict])
            for row in candidates
        ]
        control = candidates[int(np.argmin(distance))]
        control_rows.remove(control)
        matches.append((conflict, control))
    result = {"n_pairs": len(matches), "fields": {}}
    for field in fields:
        differences = np.asarray([
            table[field][conflict] - table[field][control]
            for conflict, control in matches
        ], dtype=np.float64)
        result["fields"][field] = {
            "conflict_minus_control_median": (
                float(np.median(differences)) if len(differences) else None
            ),
            "differences": differences.tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs", type=Path,
        default=G1_OUTPUT / "fy_jicamarca_representativeness_pairs.npz",
    )
    parser.add_argument(
        "--g1-report", type=Path,
        default=G1_OUTPUT / "fy_jicamarca_representativeness_report.json",
    )
    parser.add_argument("--fy-index", type=Path, default=FY_INDEX)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with np.load(args.pairs, allow_pickle=False) as archive:
        pairs = {key: np.asarray(archive[key]) for key in archive.files}
    g1 = json.loads(args.g1_report.read_text(encoding="utf-8"))
    table = _profile_table(
        pairs, float(g1["summaries"]["rho_median"]), args.fy_index
    )
    associations = _numeric_associations(
        table, args.bootstrap, args.seed
    )
    candidates = [
        row["field"] for row in associations if row["exploratory_candidate"]
    ]
    matched = _matched_differences(
        table,
        (
            "h_cut_km", "hmf2", "nmf2", "md", "delta", "fold_error",
            "sza", "horizontal_rho", "vertical_distance",
        ),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "fy_jicamarca_metadata_profiles.npz"
    report_path = output / "fy_jicamarca_metadata_report.json"
    _deterministic_npz(npz_path, table)
    report = {
        "schema_version": 1,
        "scope": (
            "descriptive discovery only; ISR-derived conflict labels cannot be "
            "used as a runtime rule"
        ),
        "n_profiles": int(len(table["profile_id"])),
        "n_conflict_profiles": int(table["conflict"].sum()),
        "n_nonconflict_profiles": int((~table["conflict"]).sum()),
        "effective_dates": int(np.unique(table["profile_date"]).size),
        "satellite": _categorical_summary(table, "satellite"),
        "date": _categorical_summary(table, "profile_date"),
        "numeric_associations": associations,
        "matched_conflict_control": matched,
        "exploratory_candidates_for_train_only_validation": candidates,
        "gate": {
            "passed": bool(candidates),
            "criterion": (
                "|Spearman|>=0.40, bootstrap CI excludes zero, and "
                "leave-one-date sign agreement>=80%"
            ),
            "conclusion": (
                "candidate_metadata_requires_train_only_cross_source_validation"
                if candidates
                else "no_stable_metadata_candidate_from_G1"
            ),
            "next_step": (
                "validate listed candidates on train-only FY-COSMIC collocations"
                if candidates
                else "do not construct a station-specific FY gate from ISR labels"
            ),
        },
        "identity": {
            "g1_pairs": str(args.pairs.resolve()),
            "g1_pairs_sha256": _sha256(args.pairs),
            "g1_report": str(args.g1_report.resolve()),
            "g1_report_sha256": _sha256(args.g1_report),
            "fy_index": str(args.fy_index.resolve()),
            "fy_index_sha256": _sha256(args.fy_index),
            "script_sha256": _sha256(Path(__file__)),
        },
        "outputs": {
            "profile_npz": npz_path.name,
            "profile_npz_sha256": _sha256(npz_path),
        },
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    print(json.dumps({
        "output": str(output),
        "n_profiles": report["n_profiles"],
        "candidates": candidates,
        "gate": report["gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
