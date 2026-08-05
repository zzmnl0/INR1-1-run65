"""Independent train-only validation of the FY hmF2 representativeness candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from audit_fy_jicamarca_metadata_bias import (
    _bootstrap_rank_correlation,
    _rank_correlation,
)
from estimate_empirical_covariance import (
    BACKGROUND_DEFAULT,
    COSMIC_INDEX_PATH,
    COSMIC_PATH,
    EXPECTED_TRAIN_PROFILES,
    FY_INDEX_PATH,
    FY_PATH,
    _background_residual_cache,
    _configuration,
    _deterministic_npz,
    _localization,
    _lookup_neighbor_residual,
    _sha256,
    _strict_background,
)
from inr_modules.data_managers.FY_dataloader import (
    COSMICNeighborhoodIndex,
    FYNeighborhoodIndex,
    get_cosmic_dataloader,
    get_dataloaders,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "isr_validation_outputs"
    / "run66-direction-correction-g1-fy-jicamarca-representativeness"
    / "metadata-only-audit" / "train-only-hmf2-validation"
)
ROW_DTYPE = np.dtype([
    ("profile_id", "<i8"),
    ("date", "<i2"),
    ("hmf2", "<f8"),
    ("direction_agreement", "<f8"),
    ("target_residual", "<f8"),
    ("cosmic_residual", "<f8"),
    ("neighbor_profiles", "<i4"),
    ("pair_mass", "<f8"),
])


def _train_ids(config):
    kwargs = {
        "batch_size": min(int(config["batch_size"]), 512),
        "bin_size_hours": config["bin_size_hours"],
        "num_workers": 0,
        "use_memmap": True,
        "val_ratio": 0.1,
        "split_seed": 42,
        "points_per_profile": 8,
    }
    fy_train, fy_validation = get_dataloaders(
        config["fy_path"], profile_path=None,
        profile_index_path=config["fy_profile_index_path"], **kwargs,
    )
    cosmic_train, cosmic_validation = get_cosmic_dataloader(
        config["cosmic_path"],
        profile_index_path=config["cosmic_profile_index_path"], **kwargs,
    )
    train = {
        "FY": np.unique(fy_train.dataset.profile_ids),
        "COSMIC": np.unique(cosmic_train.dataset.profile_ids),
    }
    validation = {
        "FY": np.unique(fy_validation.dataset.profile_ids),
        "COSMIC": np.unique(cosmic_validation.dataset.profile_ids),
    }
    counts = {source: len(ids) for source, ids in train.items()}
    if counts != EXPECTED_TRAIN_PROFILES:
        raise RuntimeError(
            f"train profile identity changed: {counts} != "
            f"{EXPECTED_TRAIN_PROFILES}"
        )
    for source in train:
        if np.intersect1d(train[source], validation[source]).size:
            raise AssertionError(f"{source} train/validation overlap")
    return train


def _profile_hmf2(index_path, profile_ids):
    with np.load(index_path, allow_pickle=False) as index:
        ids = np.asarray(index["profile_id"], dtype=np.int64)
        positions = np.searchsorted(ids, profile_ids)
        if not np.array_equal(ids[positions], profile_ids):
            raise ValueError("train FY profile_id absent from QC index")
        return np.asarray(index["hmf2"], dtype=np.float64)[positions]


def _collect_rows(fy, cosmic, cosmic_index, hmf2, batch_profiles=64):
    rows = []
    for start in range(0, len(fy.ids), batch_profiles):
        stop = min(start + batch_profiles, len(fy.ids))
        selected_profiles = np.arange(start, stop)
        data = fy.data[selected_profiles]
        valid = fy.valid[selected_profiles]
        residual = fy.residual[selected_profiles]
        profile_ids = fy.ids[selected_profiles]
        flat_valid = valid.reshape(-1)
        point_data = data.reshape(-1, 5)[flat_valid]
        point_residual = residual.reshape(-1)[flat_valid]
        point_profile = np.repeat(profile_ids, valid.shape[1])[flat_valid]
        local_time = np.remainder(
            point_data[:, 3] + point_data[:, 1] / 15.0, 24.0
        )
        point_keep = (
            (point_data[:, 2] >= 120.0)
            & (point_data[:, 2] < 300.0)
            & ((local_time < 6.0) | (local_time >= 18.0))
            & (np.abs(point_residual) >= 0.05)
        )
        if not point_keep.any():
            continue
        point_data = point_data[point_keep]
        point_residual = point_residual[point_keep]
        point_profile = point_profile[point_keep]
        cached = cosmic_index.query_profiles_only(
            point_data[:, :4], allowed_profile_ids=cosmic.ids
        )
        neighbor_residual, present = _lookup_neighbor_residual(
            cosmic, cached["sel_ids"]
        )
        selected_data = cached["sel_abs"]
        token_valid = (
            cached["valid_prof"][..., None]
            & cached["sel_vmask"]
            & present[..., None]
            & np.isfinite(neighbor_residual)
        )
        payload = cosmic_index.observation_payload_from_cached(cached)
        rho = np.sqrt(np.maximum(
            payload["rho_squared"].reshape(token_valid.shape), 0.0
        ))
        localization = _localization(rho)
        observation_lt = np.remainder(
            selected_data[..., 3] + selected_data[..., 1] / 15.0, 24.0
        )
        token_valid &= (
            (rho < 0.5)
            & (localization > 0.0)
            & (selected_data[..., 2] >= 120.0)
            & (selected_data[..., 2] < 300.0)
            & ((observation_lt < 6.0) | (observation_lt >= 18.0))
            & (
                np.abs(
                    selected_data[..., 2] - point_data[:, None, None, 2]
                ) <= 20.0
            )
            & (np.abs(neighbor_residual) >= 0.05)
        )
        if not token_valid.any():
            continue

        shape = token_valid.shape
        target_grid = np.broadcast_to(
            point_residual[:, None, None], shape
        )
        target_id_grid = np.broadcast_to(
            point_profile[:, None, None], shape
        )
        neighbor_id_grid = np.broadcast_to(
            cached["sel_ids"][..., None], shape
        )
        chosen = token_valid.reshape(-1)
        target_values = target_grid.reshape(-1)[chosen].astype(np.float64)
        cosmic_values = neighbor_residual.reshape(-1)[chosen].astype(np.float64)
        target_ids = target_id_grid.reshape(-1)[chosen]
        neighbor_ids = neighbor_id_grid.reshape(-1)[chosen]
        weights = localization.reshape(-1)[chosen].astype(np.float64)
        keys = np.rec.fromarrays(
            [target_ids, neighbor_ids], names="target,neighbor"
        )
        unique_pairs, inverse = np.unique(keys, return_inverse=True)
        n_pairs = len(unique_pairs)
        weight_sum = np.bincount(inverse, weights=weights, minlength=n_pairs)
        target_pair = np.bincount(
            inverse, weights=target_values * weights, minlength=n_pairs
        ) / weight_sum
        cosmic_pair = np.bincount(
            inverse, weights=cosmic_values * weights, minlength=n_pairs
        ) / weight_sum
        pair_mass = np.bincount(inverse, weights=weights, minlength=n_pairs)
        unique_targets, target_inverse = np.unique(
            unique_pairs["target"], return_inverse=True
        )
        target_count = np.bincount(target_inverse).astype(np.float64)
        direction = np.sign(target_pair) == np.sign(cosmic_pair)
        direction_mean = np.bincount(
            target_inverse, weights=direction, minlength=len(unique_targets)
        ) / target_count
        target_mean = np.bincount(
            target_inverse, weights=target_pair, minlength=len(unique_targets)
        ) / target_count
        cosmic_mean = np.bincount(
            target_inverse, weights=cosmic_pair, minlength=len(unique_targets)
        ) / target_count
        mass = np.bincount(
            target_inverse, weights=pair_mass, minlength=len(unique_targets)
        )
        positions = np.searchsorted(fy.ids, unique_targets)
        block = np.empty(len(unique_targets), dtype=ROW_DTYPE)
        block["profile_id"] = unique_targets
        block["date"] = fy.date[positions]
        block["hmf2"] = hmf2[positions]
        block["direction_agreement"] = direction_mean
        block["target_residual"] = target_mean
        block["cosmic_residual"] = cosmic_mean
        block["neighbor_profiles"] = target_count.astype(np.int32)
        block["pair_mass"] = mass
        rows.append(block)
        if start == 0 or stop % 5000 < batch_profiles:
            print(f"[FY->COSMIC] {stop:,}/{len(fy.ids):,} profiles")
    if not rows:
        return np.empty(0, dtype=ROW_DTYPE)
    result = np.concatenate(rows)
    if np.unique(result["profile_id"]).size != len(result):
        raise AssertionError("target profile appears more than once")
    return result


def _bootstrap_rule(rows, replicates=1000, seed=42):
    dates = np.unique(rows["date"])
    groups = [np.flatnonzero(rows["date"] == day) for day in dates]
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = []
        for date_index in rng.integers(0, len(groups), size=len(groups)):
            group = groups[date_index]
            selected.append(group[rng.integers(0, len(group), size=len(group))])
        sample = rows[np.concatenate(selected)]
        threshold = np.quantile(sample["hmf2"], 0.30)
        retained = sample["hmf2"] >= threshold
        differences[replicate] = (
            sample["direction_agreement"][retained].mean()
            - sample["direction_agreement"].mean()
        )
    return np.quantile(differences, [0.025, 0.975]).tolist()


def _evaluate(rows, replicates, seed):
    if len(rows) < 200 or np.unique(rows["date"]).size < 15:
        return {
            "passed": False,
            "reason": "insufficient train-only collocation profiles/dates",
        }
    threshold = float(np.quantile(rows["hmf2"], 0.30))
    retained = rows["hmf2"] >= threshold
    baseline = float(np.mean(rows["direction_agreement"]))
    filtered = float(np.mean(rows["direction_agreement"][retained]))
    improvement = filtered - baseline
    mass_retained = float(
        rows["pair_mass"][retained].sum() / rows["pair_mass"].sum()
    )
    fold_improvements = []
    for day in np.unique(rows["date"]):
        train = rows["date"] != day
        test = ~train
        if test.sum() < 10:
            continue
        fold_threshold = np.quantile(rows["hmf2"][train], 0.30)
        test_retained = test & (rows["hmf2"] >= fold_threshold)
        if not test_retained.any():
            continue
        fold_improvements.append(
            float(
                rows["direction_agreement"][test_retained].mean()
                - rows["direction_agreement"][test].mean()
            )
        )
    ci = _bootstrap_rule(rows, replicates, seed)
    correlation = _rank_correlation(
        rows["hmf2"], rows["direction_agreement"]
    )
    correlation_ci = _bootstrap_rank_correlation(
        rows["hmf2"], rows["direction_agreement"], rows["date"],
        replicates, seed + 1,
    )
    date_nonnegative = (
        float(np.mean(np.asarray(fold_improvements) >= 0.0))
        if fold_improvements else 0.0
    )
    passed = bool(
        correlation > 0.0
        and improvement >= 0.05
        and ci[0] > 0.0
        and mass_retained >= 0.70
        and date_nonnegative == 1.0
    )
    return {
        "passed": passed,
        "n_profiles": int(len(rows)),
        "effective_dates": int(np.unique(rows["date"]).size),
        "hmf2_30th_percentile_km": threshold,
        "baseline_profile_equal_direction_agreement": baseline,
        "retained_profile_equal_direction_agreement": filtered,
        "improvement": improvement,
        "improvement_ci95": ci,
        "precision_mass_retained": mass_retained,
        "hmf2_spearman": correlation,
        "hmf2_spearman_ci95": correlation_ci,
        "leave_one_date_fold_count": len(fold_improvements),
        "leave_one_date_nonnegative_fraction": date_nonnegative,
        "leave_one_date_improvements": fold_improvements,
        "criterion": (
            "improvement>=5 pp, CI lower>0, precision mass>=70%, "
            "and every eligible leave-one-date fold nonnegative"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--background-checkpoint", type=Path, default=BACKGROUND_DEFAULT
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    config = _configuration()
    train_ids = _train_ids(config)
    model, sw_manager, iri_peak_manager = _strict_background(
        config, args.background_checkpoint, device
    )
    indices = {
        "FY": FYNeighborhoodIndex(str(FY_PATH), config),
        "COSMIC": COSMICNeighborhoodIndex(str(COSMIC_PATH), config),
    }
    caches = {}
    for source in ("FY", "COSMIC"):
        print(f"[{source}] background residuals: {len(train_ids[source]):,}")
        caches[source] = _background_residual_cache(
            source, indices[source], train_ids[source], model,
            sw_manager, iri_peak_manager, device,
        )
    hmf2 = _profile_hmf2(FY_INDEX_PATH, caches["FY"].ids)
    rows = _collect_rows(
        caches["FY"], caches["COSMIC"], indices["COSMIC"], hmf2
    )
    evaluation = _evaluate(rows, args.bootstrap, 42)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "fy_hmf2_cross_source_profiles.npz"
    report_path = output / "fy_hmf2_cross_source_report.json"
    _deterministic_npz(
        npz_path, {name: rows[name] for name in rows.dtype.names}
    )
    report = {
        "schema_version": 1,
        "scope": (
            "train-only FY target to COSMIC neighbor, both night 120-300 km, "
            "|dh|<=20 km, rho<0.5, |residual|>=0.05 dex"
        ),
        "profile_statistical_unit": True,
        "train_profile_counts": {
            source: int(len(ids)) for source, ids in train_ids.items()
        },
        "evaluation": evaluation,
        "conclusion": (
            "hmf2_candidate_independently_supported"
            if evaluation.get("passed")
            else "hmf2_candidate_not_supported"
        ),
        "next_step": (
            "freeze the independently defined rule and perform one-time G1 replay"
            if evaluation.get("passed")
            else "do not build an FY runtime gate from the ISR-discovered candidate"
        ),
        "identity": {
            "background_checkpoint": str(
                args.background_checkpoint.resolve()
            ),
            "background_checkpoint_sha256": _sha256(
                args.background_checkpoint
            ),
            "fy_data_sha256": _sha256(FY_PATH),
            "fy_index_sha256": _sha256(FY_INDEX_PATH),
            "cosmic_data_sha256": _sha256(COSMIC_PATH),
            "cosmic_index_sha256": _sha256(COSMIC_INDEX_PATH),
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
        "conclusion": report["conclusion"],
        "evaluation": evaluation,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
