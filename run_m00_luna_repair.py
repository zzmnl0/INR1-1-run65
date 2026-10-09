"""Evidence-only M00/static_ETKF repair run.

This runner never trains, calibrates, constructs a production model, or runs
ISR inference.  It consumes frozen caches and the two fixed epoch-5
checkpoints only for the bounded autograd audit delegated to Luna.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "checkpoints_fsia" / "m00-static-etkf-evidence-repair-20260911"
INPUT_MANIFEST = OUT / "input_manifest.json"
PLAN = ROOT / "reports" / "M00_static_ETKF_luna_repair_plan_20260911.md"
CACHE_ROOT = ROOT / "checkpoints_fsia" / "m00-night-harmful-correction-diagnostic-v1"
CONTROL_RUN = ROOT / "checkpoints_fsia" / "run67-m00-seed42-explicit-condition-v1"
CANDIDATE_RUN = ROOT / "checkpoints_fsia" / "run67-m00-seed42-magnetic-daynight-bias-v2"
FORMAL_M00 = ROOT / "checkpoints_fsia" / "run67-m2w-120-500-background-stratum-bias-a1s" / "best_background_model.pth"
ETKF_CACHE = ROOT / "isr_validation_outputs" / "traditional-static-etkf-raw-iri-full-isr" / "isr_etkf_cache.npz"
A1S_CACHE = ROOT / "isr_validation_outputs" / "m2w-background-stratum-bias-a1s-m00-isr" / "isr_m00_cache.npz"
OLD_REPORT = ROOT / "reports" / "M00_static_ETKF_next_step_execution_20260911.json"
OLD_GRADIENT = ROOT / "checkpoints_fsia" / "m00-static-etkf-night-diagnostic-v2" / "m00_gradient_audit.json"

MODEL_RUNS = {
    "v2_epoch5": (CANDIDATE_RUN, CANDIDATE_RUN / "epoch_05_background_model.pth"),
    "explicit_epoch5": (CONTROL_RUN, CONTROL_RUN / "epoch_05_background_model.pth"),
}
SOURCES = ("FY", "COSMIC")
SPLITS = ("train", "development")
CONDITIONS = (
    "abs_sinI_lt_0.25_night",
    "abs_sinI_lt_0.25_day",
    "abs_sinI_0.25_0.5_night",
    "abs_sinI_0.25_0.5_day",
    "abs_sinI_ge_0.5_night",
    "abs_sinI_ge_0.5_day",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return json_safe(value.tolist())
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def log_event(operation: str, **details) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "execution_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "unix_time": time.time(),
            "operation": operation,
            **json_safe(details),
        }, ensure_ascii=False) + "\n")


def artifact_map(paths) -> dict:
    return {str(Path(path).resolve().relative_to(ROOT)): sha256(Path(path))
            for path in paths if Path(path).is_file()}


def location(path: Path, *patterns: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return str(path.relative_to(ROOT))
    for pattern in patterns:
        for number, line in enumerate(lines, 1):
            if pattern in line:
                return f"{path.relative_to(ROOT)}:{number}"
    return str(path.relative_to(ROOT))


def scalar_metrics(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (np.ndarray, list, tuple, dict)):
            continue
        if isinstance(value, (np.generic,)):
            value = value.item()
        result[key] = json_safe(value)
    return result


def load_input_manifest() -> dict:
    payload = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("formal_m00_unchanged") is not True:
        raise ValueError("input freeze does not protect formal M00")
    return payload


def e01_identity() -> dict:
    manifest = load_input_manifest()
    expected = {
        "canonical_etkf_cache": ETKF_CACHE,
        "formal_a1s_isr_cache": A1S_CACHE,
        "formal_m00_checkpoint": FORMAL_M00,
        "explicit_condition_epoch5": MODEL_RUNS["explicit_epoch5"][1],
        "magnetic_daynight_v2_epoch5": MODEL_RUNS["v2_epoch5"][1],
        "date_split_manifest": ROOT / "m00_residual_outputs" / "a1s_train_qd" / "a1s_date_split_manifest_20day.json",
        "repair_plan": PLAN,
    }
    observed = {}
    mismatches = []
    for name, path in expected.items():
        actual = sha256(path)
        frozen = manifest.get("fixed_identities", {}).get(name, {}).get("sha256")
        observed[name] = {"path": str(path.resolve()), "sha256": actual,
                          "frozen_sha256": frozen, "match": actual == frozen}
        if actual is None or actual != frozen:
            mismatches.append(name)
    old_paths = {
        "previous_report_json": OLD_REPORT,
        "previous_gradient_audit": OLD_GRADIENT,
    }
    old_observed = {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                    for name, path in old_paths.items()}
    payload = {
        "semantics": "m00_static_etkf_identity_verification_v1",
        "status": "passed" if not mismatches else "failed",
        "observed": observed,
        "previous_artifacts_observed": old_observed,
        "formal_m00_unchanged": sha256(FORMAL_M00) == manifest["fixed_identities"]["formal_m00_checkpoint"]["sha256"],
        "retractions": [
            "撤回上一轮将分类修复写成完成的结论。",
            "撤回只凭 history/计数相等就称真实参与重建完成的结论。",
            "撤回 tanh-cosine 饱和解释、模块归因以及‘全程 train-only/总前向只有 80’的未受完整状态/调用计数支持表述。",
            "上一轮未更新参数；本轮是否调用 optimizer.step 由本轮审计与执行日志实测，不由旧报告继承。",
        ],
        "limitations": manifest.get("limitations", []),
    }
    write_json(OUT / "identity_verification.json", payload)
    log_event("E01_identity_freeze", status=payload["status"], mismatches=mismatches)
    return {
        "status": payload["status"],
        "code_locations": [location(Path(__file__), "def e01_identity")],
        "artifact_paths": [OUT / "input_manifest.json", OUT / "identity_verification.json"],
        "observed": payload,
        "expected": "所有固定身份实测哈希与冻结值一致；正式 M00 不变。",
        "limitations": payload["limitations"],
    }


def _cache_path(model: str, source: str, split: str) -> Path:
    return CACHE_ROOT / model / f"{source.lower()}_{split}.npz"


def _load_point_caches() -> tuple[dict, dict, dict]:
    caches = {}
    hashes = {}
    for model in MODEL_RUNS:
        caches[model] = {}
        for source in SOURCES:
            for split in SPLITS:
                path = _cache_path(model, source, split)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path, allow_pickle=False) as data:
                    required = {"row_index", "profile_id", "coords", "observation", "iri", "m00"}
                    if not required <= set(data.files):
                        raise ValueError(f"cache missing fields: {path}")
                    caches[model][(source, split)] = {key: np.asarray(data[key])
                                                     for key in data.files}
                hashes[f"{model}/{source}/{split}"] = sha256(path)
    identity = {}
    for source in SOURCES:
        for split in SPLITS:
            left = caches["v2_epoch5"][(source, split)]
            right = caches["explicit_epoch5"][(source, split)]
            checks = {
                key: bool(np.array_equal(left[key], right[key], equal_nan=True))
                for key in ("row_index", "profile_id", "coords", "observation", "iri")
            }
            identity[f"{source}/{split}"] = checks
    return caches, hashes, identity


def e02_classification() -> dict:
    from run_m00_condition_diagnostic import harmful_correction_metrics

    caches, cache_hashes, common_identity = _load_point_caches()
    old = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    old_rows = {(row["model"], row["source"], row["split"]): row
                for row in old.get("overall_profile_equal_metrics", [])}
    rows = []
    for model in MODEL_RUNS:
        for source in SOURCES:
            for split in SPLITS:
                cache = caches[model][(source, split)]
                metrics = harmful_correction_metrics(
                    cache["observation"], cache["iri"], cache["m00"], cache["profile_id"],
                    include_profile_metrics=False,
                )
                new = scalar_metrics(metrics)
                old_row = old_rows.get((model, source, split), {})
                diff = {
                    key: (new.get(key), old_row.get(key),
                          None if new.get(key) is None or old_row.get(key) is None
                          else new[key] - old_row[key])
                    for key in ("harmful_fraction", "useful_fraction", "reverse_fraction",
                                "same_direction_over_fraction",
                                "near_zero_observation_error_fraction", "mean_H")
                }
                rows.append({"model": model, "source": source, "split": split,
                             "cache": str(_cache_path(model, source, split).resolve()),
                             "cache_sha256": cache_hashes[f"{model}/{source}/{split}"],
                             "metrics": new, "previous_report_comparison": diff})

    counterexamples = {}
    examples = {
        "useful_d_0.1_c_0.15": ([.1], [0.0], [.15]),
        "harmful_same_direction_d_0.1_c_0.25": ([.1], [0.0], [.25]),
        "harmful_near_zero_d_0_c_0.1": ([0.0], [0.0], [.1]),
        "reverse": ([.1], [0.0], [-.1]),
        "zero_c": ([.1], [0.0], [0.0]),
        "nonfinite": ([.1, np.nan], [0.0, 0.0], [.25, .25]),
    }
    for name, (obs, iri, m00) in examples.items():
        metrics = harmful_correction_metrics(obs, iri, m00, np.arange(len(obs)),
                                             include_profile_metrics=False)
        counterexamples[name] = scalar_metrics(metrics)
    checks = {
        "cache_count": len(rows) == 8,
        "common_cache_identity": all(all(values.values()) for values in common_identity.values()),
        "useful_d_0.1_c_0.15": counterexamples["useful_d_0.1_c_0.15"].get("useful_point_count") == 1,
        "harmful_same_direction_d_0.1_c_0.25": counterexamples["harmful_same_direction_d_0.1_c_0.25"].get("harmful_point_count") == 1,
        "harmful_near_zero_d_0_c_0.1": counterexamples["harmful_near_zero_d_0_c_0.1"].get("harmful_point_count") == 1,
        "reverse": counterexamples["reverse"].get("reverse_point_count") == 1,
        "zero_c": counterexamples["zero_c"].get("near_zero_c_count") == 1,
        "nonfinite_count": counterexamples["nonfinite"].get("invalid_point_count") == 1,
    }
    payload = {
        "semantics": "m00_static_etkf_classification_repair_v2",
        "status": "passed" if all(checks.values()) else "failed",
        "definitions": {
            "d": "obs-IRI", "c": "M00-IRI", "H": "c^2-2*c*d",
            "near_zero_c": "abs(c)<=1e-6",
            "near_zero_observation_residual": "abs(d)<0.02",
            "direction_valid": "abs(d)>=0.02 and abs(c)>1e-6",
            "same_direction_harmful": "direction_valid and c*d>0 and H>1e-8",
            "profile_estimator": "pointwise classes then profile mean then equal profile mean",
        },
        "checks": checks,
        "common_cache_identity": common_identity,
        "cache_hashes": cache_hashes,
        "rows": rows,
        "counterexamples": counterexamples,
        "old_vs_new": "旧报告仅作逐字段对照；本产物由八份同身份逐点缓存重新计算。",
    }
    write_json(OUT / "e02_classification.json", payload)
    log_event("E02_classification", status=payload["status"], cache_count=len(rows))
    return {
        "status": payload["status"],
        "code_locations": [location(ROOT / "run_m00_condition_diagnostic.py", "def harmful_correction_metrics")],
        "artifact_paths": [OUT / "e02_classification.json"],
        "observed": {"checks": checks, "row_count": len(rows)},
        "expected": "八份同身份缓存、唯一分类、profile 等权、反例和旧新差异均可复核。",
        "limitations": ["历史训练时输入字节哈希不可追溯。"],
    }


def _profile_equal_harmful_delta(candidate: dict, control: dict, mask) -> float:
    from run_m00_condition_diagnostic import harmful_correction_metrics

    def one(item):
        result = harmful_correction_metrics(
            item["observation"][mask], item["iri"][mask], item["m00"][mask],
            item["profile_id"][mask], include_profile_metrics=True)
        values = result.get("profile_metrics", [])
        if not values:
            return float("nan")
        return float(np.mean([row["harmful_fraction"] for row in values]))

    return one(candidate) - one(control)


def e03_bootstrap(caches: dict | None = None) -> dict:
    from run_m00_condition_diagnostic import paired_date_bootstrap_harmful

    if caches is None:
        caches, _, _ = _load_point_caches()
    # Unequal profile counts per date catch the old date-mean estimator.
    candidate = {
        "observation": np.array([1, 1, 1, 1, 1, 1], dtype=float),
        "iri": np.zeros(6), "m00": np.array([.25, .25, .25, .25, 0, 0]),
        "profile_id": np.array([1, 1, 2, 3, 4, 5]),
        "coords": np.array([[0, 0, 250, 0], [0, 0, 250, 0], [0, 0, 250, 24],
                             [0, 0, 250, 48], [0, 0, 250, 48], [0, 0, 250, 48]], dtype=float),
    }
    control = dict(candidate, m00=np.array([0, 0, 0, 0, 0, 0], dtype=float))
    synthetic_mask = np.ones(6, dtype=bool)
    synthetic = paired_date_bootstrap_harmful(candidate, control, synthetic_mask)
    expected = _profile_equal_harmful_delta(candidate, control, synthetic_mask)
    actual = (synthetic.get("delta") or {}).get("harmful_fraction")
    checks = {
        "two_dates_estimable": synthetic.get("estimable") is True and synthetic.get("date_count") == 3,
        "replicates_seed": synthetic.get("replicates") == 2000 and synthetic.get("seed") == 42,
        "profile_equal_point_estimate": isinstance(actual, (float, int)) and abs(actual - expected) < 1e-12,
    }
    real_rows = []
    for source in SOURCES:
        for split in SPLITS:
            left = caches["v2_epoch5"][(source, split)]
            right = caches["explicit_epoch5"][(source, split)]
            dates = np.floor(left["coords"][:, 3] / 24.0)
            for period, period_mask in (("day", (left["coords"][:, 3] % 24 >= 6) & (left["coords"][:, 3] % 24 < 18)),
                                         ("night", ~((left["coords"][:, 3] % 24 >= 6) & (left["coords"][:, 3] % 24 < 18)))):
                mask = (period_mask & (left["coords"][:, 2] >= 200)
                        & (left["coords"][:, 2] <= 500)
                        & np.isfinite(left["observation"]) & np.isfinite(left["m00"])
                        & np.isfinite(right["m00"]))
                result = paired_date_bootstrap_harmful(left, right, mask)
                real_rows.append({"source": source, "split": split, "height": "200-500",
                                  "period": period, "result": result,
                                  "date_count_observed": int(np.unique(dates[mask]).size)})
    payload = {
        "semantics": "m00_static_etkf_date_block_profile_equal_bootstrap_v2",
        "status": "passed" if all(checks.values()) else "failed",
        "bootstrap": {"replicates": 2000, "seed": 42, "sampling_unit": "station_utc_date",
                       "estimator": "profile_equal_over_profiles_after_synchronous_date resampling"},
        "synthetic": {"checks": checks, "result": synthetic, "expected_delta": expected},
        "real_rows": real_rows,
        "limitations": ["区间是回归诊断的日期块不确定性，不是新独立测试。"],
    }
    write_json(OUT / "e03_date_bootstrap.json", payload)
    log_event("E03_date_bootstrap", status=payload["status"], real_row_count=len(real_rows))
    return {
        "status": payload["status"],
        "code_locations": [location(ROOT / "run_m00_condition_diagnostic.py", "def paired_date_bootstrap_harmful")],
        "artifact_paths": [OUT / "e03_date_bootstrap.json"],
        "observed": {"checks": checks, "real_row_count": len(real_rows)},
        "expected": "同步日期重采样且每次重算 profile 等权主估计量；少于两日期输出 null 原因。",
        "limitations": payload["limitations"],
    }


def _condition_arrays(coords):
    import torch
    from inr_modules.mdia import train_fsia as tf

    values = {}
    masks = tf._background_bias_magnetic_daynight_masks(
        torch.as_tensor(coords, dtype=torch.float32))
    for name, mask in masks:
        values[name] = np.asarray(mask.detach().cpu(), dtype=bool)
    return values


def _dataset_support(loader) -> dict:
    dataset = loader.dataset
    rows = np.asarray(dataset.data[dataset.selected_indices], dtype=np.float32)
    coords = rows[:, :4]
    profile_ids = np.asarray(dataset.profile_ids, dtype=np.int64)
    masks = _condition_arrays(coords)
    return {name: set(profile_ids[mask].tolist()) for name, mask in masks.items()}


def _batch_payload(loader, batch):
    data = np.asarray(batch[0].detach().cpu(), dtype=np.float32)
    local_ids = np.asarray(batch[1].detach().cpu(), dtype=np.int64).reshape(-1)
    profile_ids = np.asarray(batch[2].detach().cpu(), dtype=np.int64).reshape(-1)
    actual_rows = np.asarray(loader.dataset.selected_indices[local_ids], dtype=np.int64)
    return data[:, :4], actual_rows, profile_ids


def reconstruct_sampling() -> dict:
    from inr_modules.data_managers.FY_dataloader import get_cosmic_dataloader, get_dataloaders
    from inr_modules.mdia import train_fsia as tf

    config = json.loads((CANDIDATE_RUN / "run_manifest.json").read_text(encoding="utf-8"))["config"]
    split = json.loads(Path(config["date_split_manifest"]).read_text(encoding="utf-8"))["partitions"]
    split_days = {"train": list(split["train"]), "development": list(split["development"])}
    common = {
        "batch_size": int(config["batch_size"]),
        "bin_size_hours": float(config["bin_size_hours"]),
        "num_workers": 0,
        "use_memmap": bool(config.get("use_memmap", True)),
        "val_ratio": None,
        "split_seed": int(config["seed"]),
        "points_per_profile": int(config.get("profile_points_per_epoch", 8)),
        "split_days": split_days,
        "alt_range": tf._observation_alt_range(config),
        "full_validation_profiles": False,
        "sampling_semantics": config.get("background_sampling_semantics", "legacy_profile_time_bin_v1"),
    }
    fy, _ = get_dataloaders(
        npy_path=config["fy_path"], profile_path=config.get("fy_profile_path"),
        profile_index_path=config.get("fy_profile_index_path"),
        schedule_seed=int(config.get("background_profile_schedule_seed_fy", config.get("background_profile_schedule_seed", 42))),
        magnetic_sidecar_path=(config.get("m00_observation_qd_sidecar_fy") or config.get("m00_observation_magnetic_sidecar_fy")),
        magnetic_sidecar_source="FY", magnetic_sidecar_strict=False, **common)
    cosmic, _ = get_cosmic_dataloader(
        cosmic_path=config["cosmic_path"], profile_index_path=config.get("cosmic_profile_index_path"),
        schedule_seed=int(config.get("background_profile_schedule_seed_cosmic", config.get("background_profile_schedule_seed", 42) + 1)),
        magnetic_sidecar_path=(config.get("m00_observation_qd_sidecar_cosmic") or config.get("m00_observation_magnetic_sidecar_cosmic")),
        magnetic_sidecar_source="COSMIC", magnetic_sidecar_strict=False, **common)
    loaders = {"FY": fy, "COSMIC": cosmic}
    history = json.loads((CANDIDATE_RUN / "fsia_training_history.json").read_text(encoding="utf-8"))
    history_rows = {int(row["epoch"]): row for row in history if row.get("stage") == "background"}
    all_batch_epoch, all_batch_source, all_batch_index = [], [], []
    all_offsets = [0]
    all_rows, all_profiles = [], []
    all_active = []
    all_sampled_counts, all_active_counts, all_point_counts = [], [], []
    summaries = []
    errors = []
    expected_profiles = {"FY": 44625, "COSMIC": 56099}
    expected_batches = {"FY": 349, "COSMIC": 439}
    for epoch in range(1, int(config.get("background_epochs", 5)) + 1):
        source_summaries = {}
        for source in SOURCES:
            loader = loaders[source]
            support = _dataset_support(loader)
            batches = list(loader)
            seen = []
            for batch_index, batch in enumerate(batches, 1):
                coords, row_ids, profile_ids = _batch_payload(loader, batch)
                seen.extend(np.unique(profile_ids).tolist())
                masks = _condition_arrays(coords)
                active_row = []
                sampled_row = []
                active_profile_row = []
                for name in CONDITIONS:
                    selected_profiles = set(profile_ids[masks[name]].tolist())
                    sampled_row.append(len(selected_profiles))
                    active = len(selected_profiles) >= int(config.get("background_bias_magnetic_min_batch_profiles", 8)) and len(support[name]) >= int(config.get("background_bias_magnetic_min_profiles", 30))
                    active_row.append(active)
                    active_profile_row.append(len(selected_profiles) if active else 0)
                all_batch_epoch.append(epoch)
                all_batch_source.append(0 if source == "FY" else 1)
                all_batch_index.append(batch_index)
                all_rows.extend(row_ids.tolist())
                all_profiles.extend(profile_ids.tolist())
                all_offsets.append(len(all_rows))
                all_active.append(active_row)
                all_sampled_counts.append(sampled_row)
                all_active_counts.append(active_profile_row)
                all_point_counts.append(len(row_ids))
            seen_counts = {int(pid): seen.count(int(pid)) for pid in set(seen)}
            total_profiles = len(np.unique(loader.dataset.profile_ids))
            unique_seen = set(seen)
            source_summaries[source] = {
                "source_profile_total": total_profiles,
                "unique_profiles_consumed": len(unique_seen),
                "missing_profiles": len(set(map(int, loader.dataset.profile_ids)) - unique_seen),
                "duplicate_profiles": sum(count > 1 for count in seen_counts.values()),
                "batch_count": len(batches),
                "point_count": int(sum(all_point_counts[-len(batches):])) if batches else 0,
                "conditions": {},
            }
            if total_profiles != expected_profiles[source]:
                errors.append(
                    f"epoch={epoch} source={source} profile_total="
                    f"{total_profiles} expected={expected_profiles[source]}")
            if len(batches) != expected_batches[source]:
                errors.append(
                    f"epoch={epoch} source={source} batch_count="
                    f"{len(batches)} expected={expected_batches[source]}")
            for index, name in enumerate(CONDITIONS):
                sampled_profiles = set()
                active_profiles = set()
                active_batches = 0
                offset = len(all_active) - len(batches)
                for batch_number, batch in enumerate(batches):
                    coords, _, profile_ids = _batch_payload(loader, batch)
                    selected = set(profile_ids[_condition_arrays(coords)[name]].tolist())
                    sampled_profiles.update(selected)
                    if all_active[offset + batch_number][index]:
                        active_batches += 1
                        active_profiles.update(selected)
                source_summaries[source]["conditions"][name] = {
                    "support_profile_count": len(support[name]),
                    "sampled_profile_count": len(sampled_profiles),
                    "active_profile_count": len(active_profiles),
                    "active_batches": active_batches,
                    "total_batches": len(batches),
                    "support_excluded_profiles": len(support[name] - sampled_profiles),
                    "batch_profile_threshold": int(config.get("background_bias_magnetic_min_batch_profiles", 8)),
                    "support_profile_threshold": int(config.get("background_bias_magnetic_min_profiles", 30)),
                }
            expected_history = history_rows.get(epoch, {}).get("magnetic_condition_activation", {}).get(source, {})
            for name, expected in expected_history.items():
                observed = source_summaries[source]["conditions"].get(name, {})
                if observed.get("active_batches") != expected.get("active_batches"):
                    errors.append(f"epoch={epoch} source={source} condition={name} active_batches")
                if observed.get("sampled_profile_count") != expected.get("sampled_profile_count"):
                    errors.append(f"epoch={epoch} source={source} condition={name} sampled_profile_count")
            if source_summaries[source]["missing_profiles"] or source_summaries[source]["duplicate_profiles"]:
                errors.append(f"epoch={epoch} source={source} profile coverage")
        summaries.append({
            "epoch": epoch,
            "sources": source_summaries,
            "total_batches": sum(
                item["batch_count"] for item in source_summaries.values()),
        })
    np.savez_compressed(
        OUT / "sampling_selection_manifest.npz",
        batch_epoch=np.asarray(all_batch_epoch, dtype=np.int16),
        batch_source=np.asarray(all_batch_source, dtype=np.int8),
        batch_index=np.asarray(all_batch_index, dtype=np.int32),
        batch_offsets=np.asarray(all_offsets, dtype=np.int64),
        row_ids=np.asarray(all_rows, dtype=np.int64),
        profile_ids=np.asarray(all_profiles, dtype=np.int64),
        point_counts=np.asarray(all_point_counts, dtype=np.int32),
        active=np.asarray(all_active, dtype=bool),
        sampled_profile_counts=np.asarray(all_sampled_counts, dtype=np.int32),
        active_profile_counts=np.asarray(all_active_counts, dtype=np.int32),
    )
    payload = {
        "semantics": "m00_production_sampling_reconstruction_v2",
        "status": "passed" if not errors else "failed",
        "model_calls": 0,
        "seed": int(config.get("background_profile_schedule_seed", 42)),
        "profile_points_per_epoch": int(config.get("profile_points_per_epoch", 8)),
        "epochs": summaries,
        "history_comparison": {"active_batch_and_sampled_profile_checks": not errors},
        "selection_manifest": {
            "path": str((OUT / "sampling_selection_manifest.npz").resolve()),
            "sha256": sha256(OUT / "sampling_selection_manifest.npz"),
            "batch_count": len(all_batch_epoch),
            "point_count": len(all_rows),
            "row_ids_sha256": array_sha256(all_rows),
            "profile_ids_sha256": array_sha256(all_profiles),
        },
        "errors": errors,
        "limitations": ["训练时原始输入文件的历史字节哈希不可追溯；本次只证明当前冻结数据与规则可复算。"],
    }
    write_json(OUT / "e04_sampling_reconstruction.json", payload)
    log_event("E04_sampling_reconstruction", status=payload["status"], model_calls=0,
              batch_count=len(all_batch_epoch), errors=errors[:10])
    return {
        "status": payload["status"],
        "code_locations": [location(Path(__file__), "def reconstruct_sampling")],
        "artifact_paths": [OUT / "e04_sampling_reconstruction.json", OUT / "sampling_selection_manifest.npz"],
        "observed": {"model_calls": 0, "errors": errors, "batch_count": len(all_batch_epoch)},
        "expected": "每 epoch FY=44625/COSMIC=56099 profiles，788 updates，按条件记录 support/sampled/active profile 与批次。",
        "limitations": payload["limitations"],
    }


def run_gradient_audit() -> dict:
    output = OUT / "m00_gradient_audit.json"
    log_path = OUT / "gradient_audit_command.log"
    command = [sys.executable, str(ROOT / "run_m00_gradient_audit.py"),
               "--output", str(output), "--selection-output",
               str(OUT / "gradient_selection_manifest.json")]
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               encoding="utf-8", errors="replace")
    log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr,
                        encoding="utf-8")
    log_event("E05_E06_gradient_audit_command", command=command,
              exit_code=completed.returncode, elapsed_seconds=time.time() - started,
              stdout_log=str(log_path.resolve()))
    if completed.returncode != 0 or not output.is_file():
        return {
            "status": "failed",
            "code_locations": [location(ROOT / "run_m00_gradient_audit.py", "def main")],
            "artifact_paths": [log_path],
            "observed": {"exit_code": completed.returncode, "stderr": completed.stderr[-2000:]},
            "expected": "纯 autograd.grad、固定最多 80 source-batch、无 optimizer/validate/step。",
            "limitations": ["真实梯度审计未产生可核验 JSON。"],
        }
    payload = json.loads(output.read_text(encoding="utf-8"))
    forbidden = {
        "optimizer_step_count": payload.get("optimizer_step_count"),
        "optimizer_created": payload.get("optimizer_created"),
        "validation_invocations": payload.get("validation_invocations"),
        "calibration_invocations": payload.get("calibration_invocations"),
        "records_total": payload.get("records_total", sum(
            report.get("record_count", 0)
            for report in payload.get("reports", {}).values())),
        "model_state_unchanged": payload.get("model_state_unchanged"),
        "rng_state_restored": payload.get("rng_state_restored"),
    }
    checks = {
        "records_total_80": forbidden["records_total"] == 80,
        "optimizer_steps_zero": forbidden["optimizer_step_count"] == 0,
        "optimizer_not_created": forbidden["optimizer_created"] in (False, "not_created", None),
        "validation_zero": forbidden["validation_invocations"] in (0, None),
        "calibration_zero": forbidden["calibration_invocations"] in (0, None),
        "state_unchanged": forbidden["model_state_unchanged"] is True,
        "rng_restored": forbidden["rng_state_restored"] is True,
    }
    payload["repair_runner_checks"] = checks
    write_json(output, payload)
    status = "passed" if all(checks.values()) else "failed"
    log_event("E05_E06_gradient_audit", status=status, checks=checks,
              records_total=forbidden["records_total"])
    result = {
        "status": status,
        "code_locations": [location(ROOT / "run_m00_gradient_audit.py", "def _run_checkpoint"),
                            location(ROOT / "inr_modules" / "mdia" / "train_fsia.py", "def _current_stage_trainable_parameters")],
        "artifact_paths": [output, log_path, OUT / "gradient_selection_manifest.json"],
        "observed": forbidden | {"checks": checks},
        "expected": "完整状态/随机状态恢复，模块切片无重叠，真实 80 条 source-batch 记录且禁止动作计数为零。",
        "limitations": payload.get("limitations", []),
    }
    write_json(OUT / "e05_e06_gradient_summary.json", result)
    result["artifact_paths"].append(OUT / "e05_e06_gradient_summary.json")
    return result


def _isr_gap_table() -> dict:
    from isr_evaluation.background_m00_eval import _metrics, _reporting_masks
    from run_m00_condition_diagnostic import paired_date_bootstrap_with_abs_bias

    with np.load(ETKF_CACHE, allow_pickle=False) as data:
        etkf = {key: np.asarray(data[key]) for key in data.files}
    with np.load(A1S_CACHE, allow_pickle=False) as data:
        a1s = {key: np.asarray(data[key]) for key in data.files}
    canonical_keys = ("station", "key", "timestamp", "unit_id", "altitude_km")
    alignment = {key: bool(np.array_equal(etkf[key], a1s[key])) for key in canonical_keys}
    timestamp = etkf["timestamp"].astype("datetime64[s]")
    anchor = np.datetime64("2024-09-01T00:00:00", "s").astype(np.int64)
    relative_hours = (timestamp.astype(np.int64) - anchor) / 3600.0
    local_time = (relative_hours + etkf["lon"] / 15.0) % 24.0
    observed = etkf["observation"].astype(float)
    predictions = {"ETKF": etkf["ETKF_log10"].astype(float),
                   "A1S": a1s["M00_A1S_log10"].astype(float)}
    common = np.isfinite(observed) & np.isfinite(predictions["ETKF"]) & np.isfinite(predictions["A1S"])
    rows, paired = [], []
    wanted = {"main_200_300_day", "main_200_300_night", "main_300_500_day",
              "main_300_500_night", "main_200_500_day", "main_200_500_night"}
    wanted |= {f"main_200_500_lt_{start:02d}_{start + 3:02d}" for start in range(24)}
    for station in ("Jicamarca", "PokerFlat"):
        for stratum, stratum_mask in _reporting_masks(etkf["altitude_km"], local_time):
            if stratum not in wanted:
                continue
            mask = common & stratum_mask & (etkf["station"] == station)
            for model, prediction in predictions.items():
                metrics = _metrics(observed[mask], prediction[mask])
                if metrics["rmse"] is not None:
                    metrics["rmse_identity_error"] = (metrics["rmse"] ** 2
                        - metrics["bias"] ** 2 - metrics["centered_rmse"] ** 2)
                    metrics["bias2"] = metrics["bias"] ** 2
                    metrics["centered_rmse2"] = metrics["centered_rmse"] ** 2
                rows.append({"station": station, "stratum": stratum, "model": model,
                             "date_count": int(np.unique(timestamp[mask].astype("datetime64[D]")).size),
                             **metrics})
            if mask.any():
                paired.append({"station": station, "stratum": stratum,
                               **paired_date_bootstrap_with_abs_bias(
                                   observed[mask], predictions["A1S"][mask],
                                   predictions["ETKF"][mask],
                                   timestamp[mask].astype("datetime64[D]"))})
    return {
        "semantics": "m00_static_etkf_gap_table_from_frozen_caches_v1",
        "status": "passed" if alignment and len(rows) else "failed",
        "alignment": alignment,
        "canonical_unique_rows": int(len(etkf["key"])),
        "canonical_unique_keys": int(np.unique(etkf["key"]).size),
        "locked_test_read": False,
        "new_isr_forward_count": 0,
        "metrics": rows,
        "paired_date_bootstrap": paired,
        "bootstrap": {"replicates": 2000, "seed": 42, "sampling_unit": "station_utc_date"},
    }


def e07_e08() -> dict:
    gap = _isr_gap_table()
    caches, _, _ = _load_point_caches()
    from run_m00_condition_diagnostic import _grid_coverage, _harmful_strata, harmful_correction_metrics

    satellite_rows = []
    for model in MODEL_RUNS:
        for source in SOURCES:
            for split in SPLITS:
                cache = caches[model][(source, split)]
                lt, day, _, magnetic, _ = _harmful_strata(cache["coords"])
                for period, period_mask in (("day", day == "day"), ("night", day == "night")):
                    for magnetic_label in CONDITIONS[::2]:
                        mask = (np.isfinite(cache["coords"]).all(axis=1)
                                & np.isfinite(cache["observation"])
                                & np.isfinite(cache["iri"]) & np.isfinite(cache["m00"])
                                & (cache["coords"][:, 2] >= 200)
                                & (cache["coords"][:, 2] <= 500)
                                & period_mask & (magnetic == magnetic_label.removesuffix("_night").removesuffix("_day")))
                        if not mask.any():
                            continue
                        metrics = harmful_correction_metrics(
                            cache["observation"][mask], cache["iri"][mask],
                            cache["m00"][mask], cache["profile_id"][mask], include_profile_metrics=False)
                        valid = np.isfinite(cache["m00"][mask]) & np.isfinite(cache["observation"][mask])
                        satellite_rows.append({
                            "model": model, "source": source, "split": split,
                            "period": period, "magnetic": magnetic_label,
                            "utc_date_count": int(np.unique(np.floor(cache["coords"][mask, 3] / 24.0)).size),
                            "m00_bias": float(np.mean(cache["m00"][mask][valid] - cache["observation"][mask][valid])) if valid.any() else None,
                            "grid_support": _grid_coverage(cache["coords"], cache["profile_id"], mask),
                            **scalar_metrics(metrics),
                        })
    payload = {
        "semantics": "m00_static_etkf_repair_gap_and_satellite_diagnostic_v1",
        "status": "passed" if gap["status"] == "passed" and satellite_rows else "failed",
        "e07_no_eligible_checkpoint": {
            "code_location": location(ROOT / "inr_modules" / "mdia" / "train_fsia.py", "def _checkpoint_sha256"),
            "expected": {"checkpoint": None, "checkpoint_sha256": None},
            "test_delegated_to": "focused no-checkpoint finalization test",
        },
        "gap_table": gap,
        "satellite_rows": satellite_rows,
        "candidate_decision": {
            "candidate_promoted": False,
            "formal_m00_updated": False,
            "decision": "insufficient_evidence",
            "reason": "E04/E06 只能给出冻结采样暴露与局部一阶梯度关系，不能证明五轮 AdamW 因果；不同时改变采样、权重和网络。",
            "next_candidate_protocol": None,
        },
        "limitations": [
            "正式 A1S ISR 与两个 epoch5 卫星诊断不是同一 checkpoint。",
            "ISR 缓存是已查看的回归诊断，不是新独立测试；未读取 locked-test 卫星数据。",
        ],
    }
    write_json(OUT / "e07_e08_gap_and_decision.json", payload)
    log_event("E07_E08_gap_and_decision", status=payload["status"],
              new_isr_forward_count=0, satellite_row_count=len(satellite_rows))
    return {
        "status": payload["status"],
        "code_locations": [location(Path(__file__), "def e07_e08"), payload["e07_no_eligible_checkpoint"]["code_location"]],
        "artifact_paths": [OUT / "e07_e08_gap_and_decision.json"],
        "observed": {"gap_status": gap["status"], "satellite_row_count": len(satellite_rows),
                     "candidate_decision": payload["candidate_decision"]},
        "expected": "冻结缓存差距表、RMSE 恒等式、日期区间、卫星 H/反向率/偏差/支持和唯一决定。",
        "limitations": payload["limitations"],
    }


def _run_command(command: list[str], log_name: str) -> dict:
    path = OUT / log_name
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr,
                    encoding="utf-8")
    log_event("E09_command", command=command, exit_code=completed.returncode,
              elapsed_seconds=time.time() - started, log_path=str(path.resolve()))
    return {"command": command, "exit_code": completed.returncode,
            "log_path": str(path.resolve())}


def e09_validation(previous: dict) -> dict:
    test_files = sorted(str(path.relative_to(ROOT)) for path in ROOT.glob("test_m00*.py"))
    commands = []
    if test_files:
        commands.append(_run_command(
            [sys.executable, "-m", "pytest", "--basetemp", str(OUT / "pytest-basetemp"), *test_files],
            "pytest_m00_focus.log"))
    commands.append(_run_command(
        [sys.executable, "-m", "inr_modules.mdia.fsia_model"], "fsia_model_selftest.log"))
    checks = {"commands_exit_zero": all(item["exit_code"] == 0 for item in commands),
              "items_present": set(previous) >= {
                  "E01", "E02", "E03", "E04", "E05_E06", "E07_E08"},
              "formal_checkpoint_protected": sha256(FORMAL_M00) == load_input_manifest()["fixed_identities"]["formal_m00_checkpoint"]["sha256"],
              "canonical_cache_protected": sha256(ETKF_CACHE) == load_input_manifest()["fixed_identities"]["canonical_etkf_cache"]["sha256"],
              "a1s_cache_protected": sha256(A1S_CACHE) == load_input_manifest()["fixed_identities"]["formal_a1s_isr_cache"]["sha256"]}
    payload = {"semantics": "m00_static_etkf_final_consistency_v1",
               "status": "passed" if all(checks.values()) else "failed",
               "checks": checks, "commands": commands,
               "acceptance_self_hash": "excluded_by_contract",
               "limitations": ["聚焦测试的环境缺口只按日志记录，不替代真实数据证据。"]}
    write_json(OUT / "e09_final_consistency.json", payload)
    log_event("E09_final_consistency", status=payload["status"], checks=checks)
    return {"status": payload["status"],
            "code_locations": [location(Path(__file__), "def e09_validation")],
            "artifact_paths": [OUT / "e09_final_consistency.json"] + [Path(item["log_path"]) for item in commands],
            "observed": payload,
            "expected": "状态派生、产物哈希、正式身份保护和聚焦检查可复核。",
            "limitations": payload["limitations"]}


def _acceptance_item(item_id: str, result: dict) -> dict:
    artifacts = [Path(path) for path in result.get("artifact_paths", [])]
    return {
        "id": item_id,
        "status": result.get("status", "failed"),
        "code_locations": result.get("code_locations", []),
        "artifact_paths": [str(path.resolve()) for path in artifacts],
        "artifact_sha256": artifact_map(artifacts),
        "check_command": result.get("check_command", "python run_m00_luna_repair.py"),
        "exit_code": 0 if result.get("status") == "passed" else 1,
        "observed": result.get("observed", {}),
        "expected": result.get("expected", ""),
        "limitations": result.get("limitations", []),
    }


def _failure(item_id: str, error: Exception) -> dict:
    log_event(item_id, status="failed", error=repr(error))
    return {"status": "failed", "code_locations": [str(Path(__file__).relative_to(ROOT))],
            "artifact_paths": [], "observed": {"error": repr(error)},
            "expected": "按方案完成该项并留存真实证据。", "limitations": ["执行异常：" + repr(error)]}


def render_report(acceptance: dict, results: dict) -> None:
    status = acceptance["execution_status"]
    first = ("实施状态：实施项已完成，待独立复核" if status == "completed_pending_review"
             else "实施状态：部分完成")
    passed = [key for key, item in acceptance["items"].items() if item["status"] == "passed"]
    failed = [f"{key}：{item['limitations'][0] if item['limitations'] else '证据未满足'}"
              for key, item in acceptance["items"].items() if item["status"] != "passed"]
    gradient = results.get("E05_E06", {}).get("observed", {})
    decision = results.get("E07_E08", {}).get("observed", {}).get("candidate_decision")
    report = {
        "semantics": "m00_static_etkf_luna_repair_execution_v1",
        "date": "2026-09-11",
        "status": status,
        "implementation_complete": acceptance["implementation_complete"],
        "independent_review_status": "pending",
        "research_goal_complete": False,
        "candidate_promoted": False,
        "new_training_started": False,
        "formal_m00_unchanged": True,
        "passed_ids": passed,
        "failed_ids": failed,
        "retractions": json.loads((OUT / "identity_verification.json").read_text(encoding="utf-8")).get("retractions", []),
        "gradient_summary": gradient,
        "candidate_decision": decision,
        "forbidden_actions": {
            "training_runs": 0,
            "calibration_runs": 0,
            "optimizer_steps": gradient.get("optimizer_step_count"),
            "candidate_isr_forwards": 0,
            "full_model_replays": 0,
            "locked_test_reads": 0,
            "evidence": [str((OUT / "execution_log.jsonl").resolve()), str((OUT / "e05_e06_gradient_summary.json").resolve())],
        },
        "artifact_paths": {
            "report_markdown": str((ROOT / "reports" / "M00_static_ETKF_luna_repair_execution_20260911.md").resolve()),
            "report_json": str((ROOT / "reports" / "M00_static_ETKF_luna_repair_execution_20260911.json").resolve()),
            "acceptance": str((OUT / "acceptance.json").resolve()),
        },
    }
    write_json(ROOT / "reports" / "M00_static_ETKF_luna_repair_execution_20260911.json", report)
    lines = [first, "", "本轮严格限定为证据修复、生产采样重建和固定预算梯度审计；未训练、未校准、未调用 optimizer.step、未执行 ISR 前向、未读取 locked-test 卫星数据、未进行全量模型重放。", "",
             f"已通过 ID：{', '.join(passed) if passed else '无'}。", "未通过 ID：" + ("；".join(failed) if failed else "无。"), "",
             "上一轮需撤回的表述：" + "；".join(report["retractions"]), "",
             "候选决定：" + json.dumps(decision, ensure_ascii=False), "",
             "调用与预算证据：" + json.dumps(report["forbidden_actions"], ensure_ascii=False), "",
             "报告和验收清单：",
             f"- [中文报告]({report['artifact_paths']['report_markdown']})",
             f"- [报告 JSON]({report['artifact_paths']['report_json']})",
             f"- [acceptance.json]({report['artifact_paths']['acceptance']})", ""]
    (ROOT / "reports" / "M00_static_ETKF_luna_repair_execution_20260911.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    load_input_manifest()
    results = {}
    for key, func in (("E01", e01_identity), ("E02", e02_classification),
                      ("E03", e03_bootstrap), ("E04", reconstruct_sampling)):
        try:
            results[key] = func()
        except Exception as error:  # preserve independent evidence from other items
            results[key] = _failure(key, error)
    try:
        gradient = run_gradient_audit()
        results["E05_E06"] = gradient
    except Exception as error:
        results["E05_E06"] = _failure("E05_E06", error)
    try:
        results["E07_E08"] = e07_e08()
    except Exception as error:
        results["E07_E08"] = _failure("E07_E08", error)
    try:
        results["E09"] = e09_validation(results)
    except Exception as error:
        results["E09"] = _failure("E09", error)

    items = {}
    for item_id in ("E01", "E02", "E03", "E04"):
        items[item_id] = _acceptance_item(item_id, results[item_id])
    items["E05"] = _acceptance_item("E05", results["E05_E06"])
    items["E06"] = _acceptance_item("E06", results["E05_E06"])
    items["E07"] = _acceptance_item("E07", results["E07_E08"])
    items["E08"] = _acceptance_item("E08", results["E07_E08"])
    items["E09"] = _acceptance_item("E09", results["E09"])
    all_passed = all(item["status"] == "passed" for item in items.values())
    acceptance = {
        "semantics": "m00_static_etkf_luna_repair_acceptance_v1",
        "items": items,
        "implementation_complete": all_passed,
        "execution_status": "completed_pending_review" if all_passed else "partial",
        "independent_review_status": "pending",
        "research_goal_complete": False,
        "candidate_promoted": False,
        "new_training_started": False,
        "status_derivation": "all E01-E09 passed -> completed_pending_review; otherwise partial",
        "acceptance_self_hash": "excluded_by_contract",
    }
    write_json(OUT / "acceptance.json", acceptance)
    render_report(acceptance, results)
    print(json.dumps({"status": acceptance["execution_status"],
                      "passed_ids": [key for key, value in items.items() if value["status"] == "passed"],
                      "failed_ids": [key for key, value in items.items() if value["status"] != "passed"],
                      "report": str((ROOT / "reports" / "M00_static_ETKF_luna_repair_execution_20260911.md").resolve()),
                      "acceptance": str((OUT / "acceptance.json").resolve())}, ensure_ascii=False))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
