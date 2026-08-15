"""Pure contract, statistics, and integrity helpers for the M2-W2 P0-B audit.

This module deliberately contains no model, dataset, or checkpoint loading.  It
is safe to import from unit tests and from the read-only audit runner without
starting inference or mutating training state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


P0B_AUDIT_SCHEMA_VERSION = 1
P0B_CONTRACT_ID = "m2w2_p0b_audit_contract_v1"
P0B_CONTRACT_FILENAME = "p0b_audit_contract_v1.json"
P0B_CACHE_COMPLETION_MARKER = "p0b_cache_acceptance.json"
P0B_COMPLETION_MARKER = "p0b_audit_acceptance.json"
P0B_SOURCES = ("FY", "COSMIC")
P0B_QUERY_SPLITS = ("train", "development")
P0B_COVERAGE_CODES = ("no_token", "FY_only", "COSMIC_only", "joint")
P0B_QUESTION_TERMINAL_STATUSES = (
    "supported",
    "not_supported",
    "mixed",
    "insufficient_evidence",
    "not_applicable",
)
P0B_HEIGHT_DELETE_BANDS = (
    ("drop_200_250", 200.0, 250.0, True, False),
    ("drop_250_300", 250.0, 300.0, True, False),
    ("drop_300_400", 300.0, 400.0, True, False),
    ("drop_400_500", 400.0, 500.0, True, True),
)
P0B_PREDICTIVE_NIS_FORMULA = (
    "d^T P d - (Y^T P d)^T ((N-1)I + Y^T P Y)^-1 (Y^T P d)"
)
P0B_PREDICTIVE_NIS_PRECISION = (
    "P is the unlocalized diagonal-R precision representativeness_weight / "
    "r_variance_dex2 on valid tokens; localization_weight is excluded"
)
P0B_PREDICTIVE_NIS_DOF = (
    "number of tokens in the named mode with unlocalized diagonal-R precision "
    "> 0; only predictive_nis_unlocalized/dof is a per-dof diagnostic"
)
P0B_DEFAULT_OUTPUT_DIRECTORY = (
    "m2w2_artifacts/p0b_error_chain/run67-p0b-v14-readonly-audit-r1"
)
P0B_REQUIRED_BRANCH = "codex/run67-m2w2-p0b"
P0B_BASE_ANCHOR_COMMIT = "5d5c214f4a4e461755332ec322eff1366b718dff"
P0B_IMPLEMENTATION_TAG = "m2w2-run67-p0b-audit-v1"
P0B_PYTHON_EXECUTABLE = Path(
    r"C:\Users\12238\.conda\envs\pytorch_cpu\python.exe")
P0B_CRITICAL_TRACKED_PATHS = (
    "isr_evaluation/audit_m2w2_error_chain.py",
    "isr_evaluation/summarize_m2w2_error_chain.py",
    "m2w2_contracts/p0b_audit_contract_v1.json",
    "inr_modules/mdia/p0b_audit.py",
    "inr_modules/mdia/p0b_counterfactuals.py",
    "inr_modules/mdia/checkpoint_io.py",
    "inr_modules/mdia/fsia_model.py",
    "inr_modules/mdia/sliding_dataset.py",
    "inr_modules/data_managers/FY_dataloader.py",
    "isr_evaluation/isr_loader.py",
    "test_p0b_audit.py",
    "test_p0b_counterfactuals.py",
    "test_p0b_runner.py",
    "test_p0b_summary.py",
    "test_p0b_train_only_index.py",
    "test_p0b_isr_column_filter.py",
    "test_isr_loader_geometry.py",
)
P0B_SUMMARY_ARTIFACTS = (
    "query_regime_summary.csv",
    "query_regime_summary.json",
    "query_source_summary.csv",
    "query_source_summary.json",
    "query_joint_summary.csv",
    "query_joint_summary.json",
    "p0b_six_questions.json",
)
P0B_SUMMARY_MANIFEST = "p0b_summary_manifest.json"
P0B_EXPECTED_TRAIN_PROFILE_COUNTS = {"FY": 44_625, "COSMIC": 56_099}
P0B_EXPECTED_DEVELOPMENT_PROFILE_COUNTS = {"FY": 11_591, "COSMIC": 13_924}
P0B_EXPECTED_FINITE_QUERY_COUNTS = {"train": 141_223, "development": 34_438}
P0B_P0A_CONTRACT_SHA256 = (
    "d8e26bd1f35bbe7c1911659d58a57e09e64e2432bc8adf661bd18a3d69fceb7d"
)
P0B_P0A_CONTRACT_SIZE_BYTES = 71_263
P0B_TRAIN_TOKEN_DIRECTORY_SEMANTICS = (
    "exact_train_only_compact_token_arrays_v1")
P0B_TRAIN_TOKEN_ARRAYS = (
    "token_coords", "token_values", "token_profile_ids", "token_ids")
P0B_TRAIN_TOKEN_DTYPES = {
    "token_coords": "<f4",
    "token_values": "<f4",
    "token_profile_ids": "<i8",
    "token_ids": "<i8",
}

QUERY_ID_FIELDS = (
    "query_id", "sample_key", "station", "date_utc", "batch_id")
TOKEN_ID_FIELDS = (
    "token_row_id",
    "station",
    "date_utc",
    "batch_id",
    "source",
    "profile_id",
    "token_id",
    "profile_split",
)
EDGE_ID_FIELDS = (
    "station",
    "date_utc",
    "batch_id",
    "query_id",
    "token_row_id",
    "source",
)
QUERY_CLOSURE_FIELDS = (
    "M00_log10_ne",
    "M10_log10_ne",
    "M01_log10_ne",
    "M11_log10_ne",
    "no_token_log10_ne",
    "isolated_increment_FY_dex",
    "isolated_increment_COSMIC_dex",
    "joint_update_FY_dex",
    "joint_update_COSMIC_dex",
    "joint_increment_dex",
)
QUERY_PREDICTIVE_NIS_FIELDS = (
    "predictive_nis_unlocalized_FY",
    "predictive_nis_unlocalized_FY_dof",
    "predictive_nis_unlocalized_COSMIC",
    "predictive_nis_unlocalized_COSMIC_dof",
    "predictive_nis_unlocalized_joint",
    "predictive_nis_unlocalized_joint_dof",
)
EDGE_ALGEBRA_FIELDS = (
    "innovation_dex",
    "gain_joint",
    "gain_isolated",
    "contribution_joint_dex",
    "contribution_isolated_dex",
)

P0B_TABLE_DTYPE_NAMES = {
    "query": {
        "query_id": "int64",
        "sample_key": "U64",
        "station": "U16",
        "date_utc": "U8",
        "batch_id": "int64",
        "query_profile_id": "int64",
        "timestamp_unix": "int64",
        "query_split": "U11",
        "latitude_deg": "float32",
        "longitude_deg": "float32",
        "altitude_km": "float32",
        "relative_hour": "float32",
        "local_time_hour": "float32",
        "aacgm_latitude_deg": "float32",
        "aacgm_mlt_hour": "float32",
        "cos_sza": "float32",
        "kp": "float32",
        "f107": "float32",
        "isr_log10_ne": "float32",
        "raw_iri_log10_ne": "float32",
        "M00_log10_ne": "float32",
        "M10_log10_ne": "float32",
        "M01_log10_ne": "float32",
        "M11_log10_ne": "float32",
        "no_token_log10_ne": "float32",
        "isolated_increment_FY_dex": "float32",
        "isolated_increment_COSMIC_dex": "float32",
        "joint_update_FY_dex": "float32",
        "joint_update_COSMIC_dex": "float32",
        "joint_increment_dex": "float32",
        "raw_coverage_code": "U12",
        "coverage_code": "U12",
        "FY_token_count": "int64",
        "FY_unique_profile_count": "int64",
        "FY_unlocalized_precision_sum": "float32",
        "FY_localized_precision_sum": "float32",
        "FY_token_neff": "float32",
        "FY_profile_neff": "float32",
        "FY_max_profile_precision_share": "float32",
        "predictive_nis_unlocalized_FY": "float32",
        "predictive_nis_unlocalized_FY_dof": "int64",
        "COSMIC_token_count": "int64",
        "COSMIC_unique_profile_count": "int64",
        "COSMIC_unlocalized_precision_sum": "float32",
        "COSMIC_localized_precision_sum": "float32",
        "COSMIC_token_neff": "float32",
        "COSMIC_profile_neff": "float32",
        "COSMIC_max_profile_precision_share": "float32",
        "predictive_nis_unlocalized_COSMIC": "float32",
        "predictive_nis_unlocalized_COSMIC_dof": "int64",
        "predictive_nis_unlocalized_joint": "float32",
        "predictive_nis_unlocalized_joint_dof": "int64",
        "CF_drop_200_250_log10_ne": "float32",
        "CF_drop_250_300_log10_ne": "float32",
        "CF_drop_300_400_log10_ne": "float32",
        "CF_drop_400_500_log10_ne": "float32",
        "CF_duplicate_FY_dominant_profile_log10_ne": "float32",
        "CF_duplicate_COSMIC_dominant_profile_log10_ne": "float32",
        "CF_duplicate_both_dominant_profiles_log10_ne": "float32",
        "FY_dominant_profile_id": "int64",
        "COSMIC_dominant_profile_id": "int64",
        "FY_dominant_profile_valid": "bool",
        "COSMIC_dominant_profile_valid": "bool",
    },
    "token": {
        "token_row_id": "int64",
        "station": "U16",
        "date_utc": "U8",
        "batch_id": "int64",
        "source": "U6",
        "profile_id": "int64",
        "token_id": "int64",
        "profile_split": "U5",
        "latitude_deg": "float32",
        "longitude_deg": "float32",
        "altitude_km": "float32",
        "relative_hour": "float32",
        "observation_log10_ne": "float32",
        "background_log10_ne": "float32",
    },
    "edge": {
        "station": "U16",
        "date_utc": "U8",
        "batch_id": "int64",
        "query_id": "int64",
        "token_row_id": "int64",
        "source": "U6",
        "innovation_dex": "float32",
        "r_variance_dex2": "float32",
        "representativeness_weight": "float32",
        "localization_weight": "float32",
        "unlocalized_precision": "float32",
        "localized_precision": "float32",
        "prior_predictive_variance_dex2": "float32",
        "r_standardized_innovation_sq": "float32",
        "predictive_diagonal_nis": "float32",
        "localized_innovation_energy": "float32",
        "gain_joint": "float32",
        "gain_isolated": "float32",
        "contribution_joint_dex": "float32",
        "contribution_isolated_dex": "float32",
        "space_distance_km": "float32",
        "time_distance_hours": "float32",
    },
}

P0B_DECISION_RULES = {
    "independent_unit": (
        "station-time profile with date block retained; query, token, edge, and "
        "height rows are nested technical observations"
    ),
    "eligible_cell_min_station_time_profiles": 30,
    "eligible_cell_min_unique_dates": 2,
    "material_bias_or_increment_abs_dex": 0.02,
    "low_gain_abs_joint_increment_dex": 0.005,
    "Q1_bias_origin": {
        "material_abs_dex_gte": 0.02,
        "allowed_origin_labels": [
            "raw_IRI_preexisting",
            "background_M00_shift",
            "FY_isolated_increment",
            "COSMIC_isolated_increment",
            "joint_interaction",
            "none_above_threshold",
        ],
        "transitions": {
            "raw_IRI_preexisting": (
                "cell abs median(raw_iri_log10_ne - isr_log10_ne)"
            ),
            "background_M00_shift": (
                "cell abs median(M00_log10_ne - raw_iri_log10_ne)"
            ),
            "FY_isolated_increment": (
                "cell abs median(M10_log10_ne - M00_log10_ne)"
            ),
            "COSMIC_isolated_increment": (
                "cell abs median(M01_log10_ne - M00_log10_ne)"
            ),
            "joint_interaction": (
                "cell abs median((M11_log10_ne-M00_log10_ne) - "
                "(M10_log10_ne-M00_log10_ne) - "
                "(M01_log10_ne-M00_log10_ne))"
            ),
        },
        "terminal_logic": (
            "supported for exactly one material origin label across eligible "
            "cells; mixed for multiple or cell-dependent material labels; "
            "not_supported when none is material; insufficient_evidence when "
            "no eligible cell"
        ),
    },
    "Q2_increment_consistency": {
        "material_abs_increment_dex_gte": 0.005,
        "direction_consistency_fraction_gte": 0.75,
        "low_gain_fraction_gte": 0.5,
        "source_cancellation_fraction_gte": 0.5,
        "closure_atol": 5e-06,
        "closure_rtol": 5e-06,
        "direction_reference": (
            "sign of M11-M00 versus sign of the localized-precision-weighted "
            "FY+COSMIC innovation, reduced with equal station-time-profile weight"
        ),
        "closure_role": "hard QA guard only; closure is not attribution evidence",
        "low_gain_and_source_cancellation_role": (
            "explanatory diagnostics for small net increments; they do not "
            "decide direction-consistency status"
        ),
        "terminal_logic": (
            "among material eligible cells, supported when every cell has "
            "defined direction evidence and direction_consistency_fraction "
            "meets the frozen threshold; not_supported when every defined cell "
            "fails and none is undefined; mixed when cells disagree or only "
            "some are undefined; insufficient_evidence when there is no "
            "material eligible cell or no defined direction evidence"
        ),
    },
    "profile_concentration": {
        "max_profile_precision_share_gte": 0.5,
        "profile_neff_over_token_neff_lte": 0.25,
    },
    "duplicate_profile_sensitivity": {
        "median_abs_delta_dex_gte": 0.02,
        "p90_abs_delta_dex_gte": 0.05,
    },
    "Q3_profile_precision_concentration": {
        "terminal_logic": (
            "supported when concentration and duplicate-profile sensitivity "
            "co-occur in at least one eligible cell and neither is contradicted "
            "in another eligible cell; mixed when only one flag occurs or "
            "co-occurrence is cell/source dependent; not_supported when neither "
            "flag occurs in any eligible cell; insufficient_evidence when no "
            "eligible cell"
        ),
    },
    "Q5_low_altitude_drift": {
        "query_altitude_lower_km_inclusive": 120.0,
        "query_altitude_upper_km_exclusive": 200.0,
        "material_abs_M11_minus_M00_dex_gte": 0.02,
        "height_deletion_sensitive_if_median_or_p90_threshold_met": True,
        "terminal_logic": (
            "supported when material low-altitude M11-M00 drift and "
            "high-altitude-token deletion sensitivity co-occur in an eligible "
            "cell without contradiction; mixed when only one occurs or cells "
            "disagree; not_supported when neither occurs in any eligible cell; "
            "insufficient_evidence when no eligible low-altitude cell"
        ),
    },
    "height_deletion_sensitivity": {
        "median_abs_delta_dex_gte": 0.02,
        "p90_abs_delta_dex_gte": 0.05,
    },
    "Q4_maximum_terminal_status": "insufficient_evidence",
    "Q6_required_terminal_status_until_observation_eligibility": (
        "insufficient_evidence"
    ),
    "threshold_mutation_after_cache_start_allowed": False,
}


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 of one regular file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_loads(
        payload: str | bytes | bytearray, *, label: str = "JSON") -> Any:
    """Parse strict UTF-8 JSON and reject every non-finite numeric value."""
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError(f"{label} payload must be str or bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains forbidden non-finite constant: {value}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc

    def reject_nonfinite(item: Any) -> None:
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{label} contains a non-finite numeric value")
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                reject_nonfinite(key)
                reject_nonfinite(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                reject_nonfinite(nested)

    reject_nonfinite(value)
    return value


def artifact_identity(
        path: str | os.PathLike[str],
        root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return path, byte size, and SHA256 for a completed artifact."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if root is None:
        display_path = str(path)
    else:
        root_path = Path(root).resolve()
        try:
            display_path = path.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"artifact is outside declared root: {path}") from exc
    return {
        "path": display_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def current_python_environment() -> dict[str, Any]:
    """Return the exact frozen Python environment used for P0-B publication."""
    executable = Path(sys.executable).resolve()
    if os.path.normcase(str(executable)) != os.path.normcase(
            str(P0B_PYTHON_EXECUTABLE.resolve())):
        raise RuntimeError(
            "P0-B publication must use the frozen pytorch_cpu Python executable: "
            f"{executable}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - frozen environment has torch
        raise RuntimeError("P0-B publication requires the frozen Torch runtime") from exc
    return {
        "executable": str(executable),
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def current_source_provenance(
        paths: Sequence[str | os.PathLike[str]]) -> dict[str, dict[str, Any]]:
    """Hash an exact, duplicate-free source inventory by resolved absolute path."""
    resolved = [Path(path).resolve() for path in paths]
    keys = [str(path) for path in resolved]
    if len(keys) != len(set(map(os.path.normcase, keys))):
        raise ValueError("P0-B source provenance contains duplicate resolved paths")
    return {
        str(path): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in resolved
    }


def build_train_token_directory_identity(
        indexes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Hash the exact train-only compact token arrays used by P0-B inference."""
    if set(indexes) != set(P0B_SOURCES):
        raise ValueError("train token indexes must contain exactly FY and COSMIC")
    result: dict[str, dict[str, Any]] = {}
    for source in P0B_SOURCES:
        index = indexes[source]
        try:
            arrays = {
                name: np.asarray(getattr(index, name))
                for name in P0B_TRAIN_TOKEN_ARRAYS
            }
        except AttributeError as exc:
            raise ValueError(
                f"{source} compact token directory is incomplete") from exc
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
            raise ValueError(f"{source} compact token arrays are empty or misaligned")
        token_rows = next(iter(lengths))
        if arrays["token_coords"].shape != (token_rows, 4):
            raise ValueError(f"{source} token_coords must have shape [N, 4]")
        for name in P0B_TRAIN_TOKEN_ARRAYS[1:]:
            if arrays[name].shape != (token_rows,):
                raise ValueError(f"{source} {name} must have shape [N]")

        digest = hashlib.sha256()
        schema: dict[str, Any] = {}
        for name in P0B_TRAIN_TOKEN_ARRAYS:
            value = arrays[name]
            if value.dtype.str != P0B_TRAIN_TOKEN_DTYPES[name]:
                raise TypeError(
                    f"{source} {name} dtype {value.dtype.str} differs from "
                    f"frozen {P0B_TRAIN_TOKEN_DTYPES[name]}")
            if not np.isfinite(value).all():
                raise ValueError(f"{source} {name} contains non-finite values")
            canonical = np.ascontiguousarray(value)
            digest.update(name.encode("ascii"))
            digest.update(canonical.dtype.str.encode("ascii"))
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes(order="C"))
            schema[name] = {
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
            }
        profile_ids = arrays["token_profile_ids"]
        token_ids = arrays["token_ids"]
        if np.any(profile_ids < 0) or np.any(token_ids < 0):
            raise ValueError(f"{source} token identities must be nonnegative")
        result[source] = {
            "semantics": P0B_TRAIN_TOKEN_DIRECTORY_SEMANTICS,
            "sha256": digest.hexdigest(),
            "token_rows": int(token_rows),
            "unique_profiles": int(len(np.unique(profile_ids))),
            "arrays": schema,
        }
    return validate_train_token_directory_identity(result)


def validate_train_token_directory_identity(
        identity: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and normalize the frozen compact-token identity ledger."""
    if not isinstance(identity, Mapping) or set(identity) != set(P0B_SOURCES):
        raise ValueError("train token directory identity must contain FY and COSMIC")
    normalized: dict[str, dict[str, Any]] = {}
    for source in P0B_SOURCES:
        record = identity[source]
        if not isinstance(record, Mapping) or set(record) != {
                "semantics", "sha256", "token_rows", "unique_profiles",
                "arrays"}:
            raise ValueError(f"invalid train token directory ledger for {source}")
        if record.get("semantics") != P0B_TRAIN_TOKEN_DIRECTORY_SEMANTICS:
            raise ValueError(f"train token directory semantics drifted for {source}")
        digest = record.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"invalid train token directory SHA256 for {source}")
        token_rows = record.get("token_rows")
        unique_profiles = record.get("unique_profiles")
        if (isinstance(token_rows, bool) or not isinstance(token_rows, int)
                or token_rows <= 0):
            raise ValueError(f"invalid train token row count for {source}")
        if (isinstance(unique_profiles, bool)
                or not isinstance(unique_profiles, int)
                or unique_profiles <= 0 or unique_profiles > token_rows):
            raise ValueError(f"invalid train token profile count for {source}")
        arrays = record.get("arrays")
        if not isinstance(arrays, Mapping) or set(arrays) != set(
                P0B_TRAIN_TOKEN_ARRAYS):
            raise ValueError(f"invalid train token array schema for {source}")
        normalized_arrays: dict[str, dict[str, Any]] = {}
        for name in P0B_TRAIN_TOKEN_ARRAYS:
            schema = arrays[name]
            if not isinstance(schema, Mapping) or set(schema) != {"dtype", "shape"}:
                raise ValueError(f"invalid {source} {name} token schema")
            if schema.get("dtype") != P0B_TRAIN_TOKEN_DTYPES[name]:
                raise ValueError(f"invalid {source} {name} token dtype")
            expected_shape = ([token_rows, 4] if name == "token_coords"
                              else [token_rows])
            if schema.get("shape") != expected_shape:
                raise ValueError(f"invalid {source} {name} token shape")
            normalized_arrays[name] = {
                "dtype": str(schema["dtype"]),
                "shape": list(schema["shape"]),
            }
        normalized[source] = {
            "semantics": P0B_TRAIN_TOKEN_DIRECTORY_SEMANTICS,
            "sha256": digest,
            "token_rows": token_rows,
            "unique_profiles": unique_profiles,
            "arrays": normalized_arrays,
        }
    return normalized


def current_git_provenance(
        repo_root: str | os.PathLike[str],
        version_control: Mapping[str, Any]) -> dict[str, Any]:
    """Establish the clean annotated-tag identity required to publish P0-B."""
    root = Path(repo_root).resolve()
    rtk = shutil.which("rtk")
    if rtk is None:
        raise RuntimeError("rtk is required for P0-B publication identity checks")
    required_branch = str(version_control["required_branch"])
    required_tag = str(version_control["required_implementation_tag"])
    base_anchor = str(version_control["base_anchor"]["git_commit"])
    critical_paths = tuple(version_control["critical_tracked_paths"])

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            [rtk, "git", *arguments], cwd=root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return completed.stdout.strip()

    try:
        head = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        tag_commit = run("rev-parse", f"{required_tag}^{{commit}}")
        tag_object = run("rev-parse", f"{required_tag}^{{tag}}")
        tag_type = run("cat-file", "-t", required_tag)
        tracked_status = tracked_git_status(root)
        ancestor = subprocess.run(
            [rtk, "git", "merge-base", "--is-ancestor", base_anchor, head],
            cwd=root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        untracked_critical: list[str] = []
        for relative in critical_paths:
            try:
                run("ls-files", "--error-unmatch", "--", str(relative))
            except subprocess.CalledProcessError:
                untracked_critical.append(str(relative))
        critical_diff = subprocess.run(
            [rtk, "git", "diff", "--quiet", tag_commit, "--",
             *map(str, critical_paths)],
            cwd=root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to establish P0-B publication Git identity") from exc

    result = {
        "status": "computed",
        "head": head,
        "branch": branch,
        "tracked_status": tracked_status,
        "expected_branch": required_branch,
        "base_anchor": base_anchor,
        "base_anchor_is_ancestor": ancestor.returncode == 0,
        "required_implementation_tag": required_tag,
        "implementation_tag_object_sha": tag_object,
        "implementation_tag_type": tag_type,
        "implementation_tag_commit": tag_commit,
        "head_equals_implementation_tag_commit": head == tag_commit,
        "critical_tracked_paths": list(critical_paths),
        "untracked_critical_paths": untracked_critical,
        "critical_paths_clean_against_implementation_commit": (
            critical_diff.returncode == 0),
    }
    failures = []
    if branch != required_branch:
        failures.append("branch")
    if tracked_status:
        failures.append("tracked_status")
    if tag_type != "tag":
        failures.append("annotated_tag")
    if head != tag_commit:
        failures.append("head_equals_tag")
    if untracked_critical:
        failures.append("critical_paths_tracked")
    if critical_diff.returncode != 0:
        failures.append("critical_paths_clean")
    if ancestor.returncode != 0:
        failures.append("base_anchor_is_ancestor")
    if failures:
        raise ValueError(
            "P0-B publication Git identity failed: " + ", ".join(failures))
    return result


def tracked_git_status(repo_root: str | os.PathLike[str]) -> list[str]:
    """Return raw machine Git tracked-status lines without RTK summarization."""
    rtk = shutil.which("rtk")
    if rtk is None:
        raise RuntimeError("rtk is required for P0-B publication identity checks")
    completed = subprocess.run(
        [rtk, "proxy", "git", "status", "--short", "--untracked-files=no"],
        cwd=Path(repo_root).resolve(), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True)
    return completed.stdout.splitlines()


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def atomic_write_json(
        path: str | os.PathLike[str], value: Any, *,
        overwrite: bool = True) -> Path:
    """Serialize strictly, then replace the destination atomically."""
    serialized = _json_text(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # An exclusive hard-link publishes the fully flushed temporary
            # file atomically and fails if the marker already exists.  This is
            # portable across the Windows/NTFS runtime and POSIX worktrees.
            os.link(temporary, path)
            try:
                temporary.unlink()
            except OSError:
                # The destination is already a complete, durable hard link.
                # Best-effort temporary cleanup must not turn that successful
                # publication into an exception while leaving the marker live.
                pass
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def write_completion_marker_atomically(
        marker_path: str | os.PathLike[str],
        payload: Mapping[str, Any],
        artifact_paths: Sequence[str | os.PathLike[str]],
        artifact_root: str | os.PathLike[str] | None = None) -> Path:
    """Write the completion marker only after every declared artifact exists."""
    identities = [
        artifact_identity(path, root=artifact_root) for path in artifact_paths
    ]
    result = dict(payload)
    result["audit_schema_version"] = P0B_AUDIT_SCHEMA_VERSION
    result["completion_marker"] = True
    result["artifacts"] = identities
    return atomic_write_json(marker_path, result, overwrite=False)


def contract_table_schemas(
        contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse and freeze exact NPZ fields/dtypes from the contract cache schema."""
    cache_schema = contract.get("cache_schema", {})
    if cache_schema.get("format") != "npz":
        raise ValueError("P0-B cache schema format must be NPZ")
    if cache_schema.get("allow_pickle") is not False:
        raise ValueError("P0-B cache schema must forbid pickle")
    tables = cache_schema.get("tables", {})
    if set(tables) != set(P0B_TABLE_DTYPE_NAMES):
        raise ValueError("P0-B cache table schemas are incomplete")
    parsed: dict[str, dict[str, Any]] = {}
    for table_name, expected_names in P0B_TABLE_DTYPE_NAMES.items():
        table = tables[table_name]
        required_fields = tuple(table.get("required_fields", ()))
        if required_fields != tuple(expected_names):
            raise ValueError(
                f"P0-B {table_name} required_fields order or membership drifted")
        dtype_names = table.get("dtypes", {})
        if dtype_names != expected_names:
            raise ValueError(f"P0-B {table_name} dtype map drifted")
        dtypes = {name: np.dtype(value) for name, value in dtype_names.items()}
        if any(value.kind == "O" for value in dtypes.values()):
            raise ValueError(f"P0-B {table_name} schema contains object dtype")
        parsed[table_name] = {
            "required_fields": required_fields,
            "dtypes": dtypes,
        }
    return parsed


def validate_p0b_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject contract drift in the fields that define the P0-B experiment."""
    if contract.get("contract_id") != P0B_CONTRACT_ID:
        raise ValueError("unexpected P0-B contract_id")
    if contract.get("audit_schema_version") != P0B_AUDIT_SCHEMA_VERSION:
        raise ValueError("unexpected P0-B audit_schema_version")
    if contract.get("status") != "frozen":
        raise ValueError("P0-B contract is not frozen")

    version_control = contract.get("version_control", {})
    if version_control.get("required_branch") != P0B_REQUIRED_BRANCH:
        raise ValueError("P0-B required Git branch drifted")
    anchor = version_control.get("base_anchor", {})
    if anchor.get("branch") != "codex/run67-m2w2":
        raise ValueError("P0-B base branch drifted")
    if anchor.get("git_commit") != P0B_BASE_ANCHOR_COMMIT:
        raise ValueError("P0-B base anchor commit drifted")
    if version_control.get("required_clean_tracked_worktree") is not True:
        raise ValueError("P0-B runtime must require a clean tracked worktree")
    if version_control.get("untracked_generated_artifacts_allowed") is not True:
        raise ValueError("P0-B generated untracked artifacts policy drifted")
    if version_control.get("explicit_path_staging_only") is not True:
        raise ValueError("P0-B must require explicit-path staging")
    if version_control.get("git_add_all_forbidden") is not True:
        raise ValueError("P0-B must forbid git add-all staging")
    if version_control.get("accepted_history_rewrite_forbidden") is not True:
        raise ValueError("P0-B must forbid accepted-history rewrites")
    if version_control.get("required_implementation_tag") != P0B_IMPLEMENTATION_TAG:
        raise ValueError("P0-B implementation tag drifted")
    if version_control.get("required_implementation_tag_type") != "annotated":
        raise ValueError("P0-B implementation tag must remain annotated")
    if version_control.get("require_head_equals_tag") is not True:
        raise ValueError("P0-B runtime HEAD must equal the implementation tag")
    if tuple(version_control.get("critical_tracked_paths", ())) != (
            P0B_CRITICAL_TRACKED_PATHS):
        raise ValueError("P0-B critical tracked paths drifted")
    if tuple(version_control.get("stage_commit_boundaries", ())) != (
            "plan and status record",
            "frozen P0-B audit contract and integrity helpers",
            "P0-B counterfactual mathematics",
            "strict train-only satellite and ISR loaders",
            "P0-B read-only runner and runner tests",
            "fixed P0-B summarizer and tests",
            "P0-B attribution conclusions and plan update"):
        raise ValueError("P0-B commit boundaries drifted")
    if tuple(version_control.get("runtime_acceptance_requires", ())) != (
            "branch", "git_head", "implementation_tag",
            "implementation_tag_object_sha", "head_equals_tag",
            "critical_paths_tracked", "base_anchor_is_ancestor",
            "empty_tracked_status", "source_file_sha256",
            "train_allowlist_identity",
            "train_token_directory_identity",
            "ISR_column_access_audit",
            "coordinate_enrichment", "profile_cap_status"):
        raise ValueError("P0-B runtime acceptance requirements drifted")

    identity = contract.get("identity", {})
    if tuple(identity.get("runtime_required_identities", ())) != (
            "audit_code_sha256", "git_head", "python_environment",
            "checkpoint_path", "date_split_path", "input_data_sha256",
            "train_allowlist_identity", "train_token_directory_identity",
            "ISR_source_file_sha256",
            "ISR_column_access_audit",
            "coordinate_enrichment",
            "profile_cap_status"):
        raise ValueError("P0-B runtime required identities drifted")

    token_identity = contract.get("train_token_directory_identity_contract", {})
    if token_identity.get("schema_version") != 1:
        raise ValueError("P0-B train token directory identity schema drifted")
    if token_identity.get("semantics") != P0B_TRAIN_TOKEN_DIRECTORY_SEMANTICS:
        raise ValueError("P0-B train token directory semantics drifted")
    if tuple(token_identity.get("sources", ())) != P0B_SOURCES:
        raise ValueError("P0-B train token directory sources drifted")
    if tuple(token_identity.get("arrays_in_digest_order", ())) != (
            P0B_TRAIN_TOKEN_ARRAYS):
        raise ValueError("P0-B train token digest array order drifted")
    if token_identity.get("array_dtypes") != P0B_TRAIN_TOKEN_DTYPES:
        raise ValueError("P0-B train token directory dtypes drifted")
    if token_identity.get("digest_framing") != (
            "array name ASCII, NumPy dtype.str ASCII, little-endian int64 "
            "shape, then C-contiguous payload bytes"):
        raise ValueError("P0-B train token digest framing drifted")
    if tuple(token_identity.get("required_ledgers", ())) != (
            "audit_contract.train_token_directory_identity",
            "audit_contract.runtime_identities.train_token_directory_identity",
            "manifest.train_token_directory_identity",
            "completion_marker.train_token_directory_identity"):
        raise ValueError("P0-B train token directory ledgers drifted")
    if token_identity.get("preflight_full_exact_equality_required") is not True:
        raise ValueError("P0-B preflight/full token identity equality drifted")
    if token_identity.get("end_of_run_recompute_required") is not True:
        raise ValueError("P0-B end-of-run token identity recheck drifted")

    scope = contract.get("data_scope", {})
    tokens = scope.get("satellite_tokens", {})
    queries = scope.get("ISR_queries", {})
    if tokens.get("profile_partition") != "train":
        raise ValueError("P0-B satellite tokens must be train-only")
    if tokens.get("expected_train_profile_counts") != (
            P0B_EXPECTED_TRAIN_PROFILE_COUNTS):
        raise ValueError("P0-B expected train profile counts drifted")
    if tokens.get("development_profiles_allowed") is not False:
        raise ValueError("development satellite profiles must be forbidden")
    if tokens.get("locked_test_profiles_allowed") is not False:
        raise ValueError("locked-test satellite profiles must be forbidden")
    metadata = scope.get("satellite_profile_index_metadata", {})
    if tuple(metadata.get("included_partitions", ())) != P0B_QUERY_SPLITS:
        raise ValueError("profile-index metadata partitions drifted")
    if metadata.get("access_mode") != (
            "read-only classification and aggregate of existing "
            "profile-index metadata"):
        raise ValueError(
            "profile-index metadata must remain read-only classification/aggregate")
    if metadata.get("development_density_values_allowed") is not False:
        raise ValueError("development density values must remain forbidden")
    if metadata.get("development_model_inference_allowed") is not False:
        raise ValueError("development satellite inference must remain forbidden")
    if metadata.get("expected_development_profile_counts") != (
            P0B_EXPECTED_DEVELOPMENT_PROFILE_COUNTS):
        raise ValueError("P0-B expected development profile counts drifted")
    locked_classification = metadata.get(
        "locked_test_classification_metadata", {})
    if tuple(locked_classification.get("allowed_fields", ())) != (
            "profile_id", "pass_profile", "representative_time", "date_code",
            "output_start", "output_end", "kept_points"):
        raise ValueError("locked-test classification metadata fields drifted")
    if locked_classification.get("purpose") != (
            "transient partition classification, output-boundary integrity "
            "validation, and exclusion only"):
        raise ValueError("locked-test classification metadata purpose drifted")
    for key in (
            "persistence_allowed", "aggregation_allowed",
            "density_values_allowed", "model_inference_allowed"):
        if locked_classification.get(key) is not False:
            raise ValueError(
                f"locked-test classification metadata {key} must be false")
    if metadata.get("assimilation_token_effect") != (
            "none; all assimilation tokens remain train-only"):
        raise ValueError("profile-index metadata must not affect train-only tokens")
    if tuple(queries.get("included_date_partitions", ())) != P0B_QUERY_SPLITS:
        raise ValueError("P0-B ISR query partitions drifted")
    if tuple(queries.get("excluded_date_partitions", ())) != ("locked_test",):
        raise ValueError("locked-test ISR dates are not explicitly excluded")
    if queries.get("expected_finite_query_counts") != (
            P0B_EXPECTED_FINITE_QUERY_COUNTS):
        raise ValueError("P0-B expected finite ISR query counts drifted")
    expected_restricted_identity = {
        "schema_version": 1,
        "declared_whole_file_sha256_source": (
            "P0-A ISR input-file declaration"),
        "whole_file_sha256_recompute_from_hdf_bytes_allowed": False,
        "whole_file_size_stat_allowed": True,
        "materialized_allowed_content_identity_required": True,
        "materialized_allowed_content_schema": (
            "isr_allowed_materialized_content_v1"),
        "materialized_allowed_content_exact_fields": [
            "schema", "path", "sha256", "framed_array_count",
            "allowed_time_column_count"],
        "materialized_allowed_content_digest_framing": (
            "SHA256 over the schema prefix and ordered length-framed arrays; "
            "each frame binds logical name, NumPy dtype.str, shape, C order, "
            "and C-contiguous payload bytes"),
        "materialized_allowed_content_scope": [
            "allowed timestamp values",
            "allowed 2-D dataset columns",
            "actually used static geometry arrays"],
        "locked_or_excluded_values_in_content_identity_allowed": False,
        "station_and_global_path_sorted_ledgers_required": True,
    }
    if queries.get("restricted_mixed_file_identity") != (
            expected_restricted_identity):
        raise ValueError("P0-B restricted mixed-file identity contract drifted")
    p0a_identity = scope.get("P0A_ISR_contract_identity", {})
    if p0a_identity.get("source_directory") != (
            "isr_validation_outputs/"
            "run67-p0a-v14-epoch15-vs-v13-epoch12-isr-qav2-r1"):
        raise ValueError("P0-A ISR JSON identity source drifted")
    if p0a_identity.get("access_mode") != (
            "qav2_JSON_contract_identity_only_no_NPZ_or_peak_cache_v1"):
        raise ValueError("P0-A dependency must remain JSON identity-only")
    p0a_sha256 = p0a_identity.get("sha256")
    if (not isinstance(p0a_sha256, str) or len(p0a_sha256) != 64
            or any(value not in "0123456789abcdef" for value in p0a_sha256)):
        raise ValueError("P0-A JSON identity SHA256 must be lowercase hexadecimal")
    if p0a_sha256 != P0B_P0A_CONTRACT_SHA256:
        raise ValueError("P0-A JSON identity SHA256 drifted")
    if p0a_identity.get("size_bytes") != P0B_P0A_CONTRACT_SIZE_BYTES:
        raise ValueError("P0-A JSON identity size drifted")
    if p0a_identity.get("numeric_output_join_allowed") is not False:
        raise ValueError("P0-A numeric output joins must remain forbidden")
    if p0a_identity.get("npz_or_peak_cache_read_allowed") is not False:
        raise ValueError("P0-A NPZ/peak-cache reads must remain forbidden")

    dependency = contract.get("p0a_dependency_exception", {})
    if dependency.get("p0c_locked") is not True:
        raise ValueError("P0-C must remain locked")
    if dependency.get("p0a_overall_status_at_freeze") != (
            "partial_giro_pending_and_fixed_observation_eligibility_pending"):
        raise ValueError("P0-A dependency exception drifted")
    if dependency.get("p0a_isr_status") != (
            "infrastructure_contract_pass_hmf2_eligibility_open"):
        raise ValueError("P0-A ISR status drifted")
    if dependency.get("fixed_observation_eligibility_status") != "pending":
        raise ValueError("fixed observation eligibility status drifted")
    if dependency.get("p0b_result_status_until_p0a_resolution") != "provisional":
        raise ValueError("P0-B provisional status drifted")

    partitioning = contract.get("partitioning", {})
    if partitioning.get("format") != "npz":
        raise ValueError("P0-B table format must be NPZ")
    if partitioning.get("allow_pickle") is not False:
        raise ValueError("P0-B NPZ object loading must be forbidden")
    if tuple(partitioning.get("tables_per_partition", ())) != (
            "query", "token", "edge"):
        raise ValueError("P0-B table triplet drifted")
    if partitioning.get("cache_completion_marker") != P0B_CACHE_COMPLETION_MARKER:
        raise ValueError("P0-B cache completion marker drifted")
    if partitioning.get("final_completion_marker") != P0B_COMPLETION_MARKER:
        raise ValueError("P0-B final completion marker drifted")
    lifecycle = contract.get("output_lifecycle", {})
    if lifecycle.get("default_output_directory") != P0B_DEFAULT_OUTPUT_DIRECTORY:
        raise ValueError("P0-B default output directory drifted")
    if lifecycle.get("start_precondition") != "target directory must not exist":
        raise ValueError("P0-B output start precondition drifted")
    if lifecycle.get("full_audit_precondition") != (
            "an explicit completed preflight_acceptance.json bound to the "
            "identical implementation tag object, HEAD, checkpoint, "
            "P0-A/P0-B/date-split identities, query/profile counts, "
            "coordinate/profile-cap state, source provenance, and Python "
            "environment"):
        raise ValueError("P0-B full-audit preflight dependency drifted")
    if lifecycle.get("preflight_acceptance_filename") != (
            "preflight_acceptance.json"):
        raise ValueError("P0-B preflight acceptance filename drifted")
    if lifecycle.get("preflight_directory_reuse_as_full_output_forbidden") is not True:
        raise ValueError("P0-B preflight/full directory isolation drifted")
    if lifecycle.get("overwrite_allowed") is not False:
        raise ValueError("P0-B output overwrite must remain forbidden")
    if lifecycle.get("failed_directory_policy") != (
            "preserve and permanently retire the failed directory"):
        raise ValueError("P0-B failed-directory policy drifted")
    if lifecycle.get("retry_policy") != (
            "only an explicitly authorized new revision directory may be used"):
        raise ValueError("P0-B retry policy drifted")
    if lifecycle.get("preflight_probe") != {
            "stations": ["Jicamarca", "PokerFlat"],
            "query_splits": ["train", "development"],
            "altitude_bands_km": [
                [120.0, 200.0], [200.0, 300.0], [300.0, 500.0]],
            "final_band_upper_inclusive": True,
            "max_queries_per_cell": 32,
            "cost_class": (
                "strict_full_registry_and_train_token_index_infrastructure_gate_"
                "not_lightweight"),
            "requires_full_isr_query_registry_rebuild": True,
            "requires_full_train_only_compact_token_directories": True,
            "cost_risk_note": (
                "the 12 inference probes are small, but their prerequisite ISR "
                "registry and FY/COSMIC train-only compact indexes have full-scope "
                "I/O and memory cost"),
            "numeric_predictions_persisted": False,
            "npz_written": False,
            "required_paths": [
                "M00", "M10", "M01", "M11", "no_token",
                "counterfactuals", "schema", "closure"],
            "activation_requirements": {
                "positive_localized_precision_edge_sources": ["FY", "COSMIC"],
                "minimum_effective_joint_queries": 1,
                "minimum_valid_dominant_profile_queries_per_source": 1,
                "minimum_positive_localized_precision_edges_per_height_deletion_band": 1,
            },
    }:
        raise ValueError("P0-B strict preflight-probe contract drifted")

    table_contracts = contract_table_schemas(contract)
    for name, required in (
            ("query", QUERY_ID_FIELDS + QUERY_CLOSURE_FIELDS
             + QUERY_PREDICTIVE_NIS_FIELDS),
            ("token", TOKEN_ID_FIELDS),
            ("edge", EDGE_ID_FIELDS + EDGE_ALGEBRA_FIELDS)):
        fields = set(table_contracts[name]["required_fields"])
        missing = set(required).difference(fields)
        if missing:
            raise ValueError(f"P0-B {name} schema lacks fields: {sorted(missing)}")

    nis = contract.get("precision_innovation_gain_semantics", {})
    if nis.get("predictive_nis_unlocalized_formula") != (
            P0B_PREDICTIVE_NIS_FORMULA):
        raise ValueError("P0-B predictive NIS formula drifted")
    if nis.get("predictive_nis_unlocalized_precision") != (
            P0B_PREDICTIVE_NIS_PRECISION):
        raise ValueError("P0-B predictive NIS precision semantics drifted")
    if nis.get("predictive_nis_unlocalized_dof") != P0B_PREDICTIVE_NIS_DOF:
        raise ValueError("P0-B predictive NIS dof semantics drifted")
    if nis.get("predictive_nis_unlocalized_modes") != {
            "FY": "use FY tokens only",
            "COSMIC": "use COSMIC tokens only",
            "joint": (
                "concatenate FY and COSMIC tokens before constructing the "
                "single joint system"),
    }:
        raise ValueError("P0-B predictive NIS source modes drifted")

    coordinates = contract.get("coordinate_enrichment", {})
    if coordinates.get("geographic") != {
            "status": "required",
            "fields": ["latitude_deg", "longitude_deg", "local_time_hour"],
    }:
        raise ValueError("P0-B geographic-coordinate contract drifted")
    if coordinates.get("aacgm") != {
            "status": "required",
            "package": "aacgmv2",
            "version": "2.7.0",
            "method": "ALLOWTRACE",
            "fields": ["aacgm_latitude_deg", "aacgm_mlt_hour"],
    }:
        raise ValueError("P0-B AACGM-coordinate contract drifted")
    if coordinates.get("qd") != {
            "status": "unavailable",
            "package": "apexpy",
            "reason": "not installed in the frozen pytorch_cpu P0-B environment",
            "proxy_substitution_allowed": False,
    }:
        raise ValueError("P0-B QD-coordinate status drifted")
    if coordinates.get("fixed_aggregation_time_coordinate") != "aacgm_mlt_hour":
        raise ValueError("P0-B fixed aggregation must use AACGM MLT")

    fixed = contract.get("fixed_aggregation", {})
    if tuple(fixed.get("dimensions", ())) != (
            "station", "query_split", "MLT_3h", "solar_regime",
            "query_altitude_band", "Kp_activity", "coverage_code"):
        raise ValueError("P0-B fixed aggregation dimensions drifted")
    if fixed.get("profile_equal_weighting") != {
            "enabled": True,
            "profile_id_field": "query_profile_id",
            "within_profile_signed_reducer": "median",
            "within_profile_nonnegative_reducer": "median",
            "within_profile_count_semantics": "raw_query_count",
            "absolute_transform_order": (
                "absolute_value_per_query_before_within_profile_"
                "nonnegative_reduction"),
            "nonnegative_metric_families": [
                "joint_increment_abs", "joint_source_update_abs",
                "height_deletion_abs_delta", "duplicate_profile_abs_delta"],
    }:
        raise ValueError("P0-B profile-equal aggregation contract drifted")
    if fixed.get("edge_diagnostics") != {
            "innovation_center": "median",
            "innovation_scale": "1.4826_mad",
            "tail_sigma": 3.0,
            "zero_mad_tail_rule": "tail_if_abs_deviation_gt_zero",
            "joint_source_sign_epsilon_dex": 0.005,
            "min_vertical_pairs_per_profile": 3,
            "min_profiles_for_vertical_correlation": 30,
            "token_identity_key_fields": ["source", "profile_id", "token_id"],
            "duplicate_token_payload_fields": [
                "profile_split", "latitude_deg", "longitude_deg",
                "altitude_km", "relative_hour", "observation_log10_ne",
                "background_log10_ne", "innovation_dex"],
            "token_height_interval_rule": (
                "positive_adjacent_unique_altitude_spacing_within_source_profile_"
                "then_profile_median"),
            "vertical_innovation_correlation_rule": (
                "pearson_adjacent_unique_altitude_innovation_within_profile_"
                "then_profile_median"),
            "same_altitude_innovation_reducer": "median",
            "innovation_population": (
                "per_satellite_profile_median_then_cell_robust_statistics"),
            "profile_cap_status": "not_applicable_in_v14",
    }:
        raise ValueError("P0-B edge-diagnostic aggregation contract drifted")
    if fixed.get("solar_regime") != {
            "source_field": "cos_sza",
            "day_rule": "cos_sza>0",
            "twilight_rule": "cos(108deg)<=cos_sza<=0",
            "night_rule": "cos_sza<cos(108deg)",
            "cos_108deg": -0.30901699437494734,
    }:
        raise ValueError("P0-B solar-regime aggregation contract drifted")
    q2 = contract.get("decision_rules", {}).get("Q2_increment_consistency", {})
    for key, expected in (
            ("direction_consistency_fraction_gte", 0.75),
            ("low_gain_fraction_gte", 0.5),
            ("source_cancellation_fraction_gte", 0.5)):
        if q2.get(key) != expected:
            raise ValueError(f"P0-B Q2 {key} drifted")

    bands = contract.get("counterfactuals", {}).get(
        "height_deletion", {}).get("bands", ())
    normalized_bands = tuple((
        band.get("name"),
        float(band.get("lower_km")),
        float(band.get("upper_km")),
        band.get("lower_inclusive"),
        band.get("upper_inclusive"),
    ) for band in bands)
    if normalized_bands != P0B_HEIGHT_DELETE_BANDS:
        raise ValueError("P0-B height-deletion bands drifted")
    variants = tuple(contract.get("counterfactuals", {}).get(
        "duplicate_profile", {}).get("variants", ()))
    if variants != (
            "FY_dominant_profile_only",
            "COSMIC_dominant_profile_only",
            "both_sources_each_dominant_profile"):
        raise ValueError("P0-B duplicate-profile variants drifted")

    mode_semantics = contract.get("mode_semantics", {})
    if mode_semantics.get("raw_coverage_code") != (
            "source presence is determined by production-payload token_count > 0, "
            "before any precision test"):
        raise ValueError("P0-B raw coverage semantics drifted")
    if mode_semantics.get("coverage_code") != (
            "effective source presence is determined by localized_precision_sum > 0; "
            "this is the update-capable coverage used in attribution"):
        raise ValueError("P0-B effective coverage semantics drifted")
    zero_precision = contract.get("precision_statistics", {}).get(
        "zero_precision_result", {})
    if (zero_precision.get("dominant_profile_id_npz") != -1
            or zero_precision.get("dominant_profile_valid_npz") is not False):
        raise ValueError("P0-B dominant-profile NPZ sentinel drifted")

    questions = contract.get("six_questions", {})
    if tuple(questions.get("allowed_terminal_statuses", ())) != (
            P0B_QUESTION_TERMINAL_STATUSES):
        raise ValueError("P0-B six-question terminal statuses drifted")
    records = questions.get("questions", ())
    if len(records) != 6 or len({row.get("id") for row in records}) != 6:
        raise ValueError("P0-B must contain six uniquely identified questions")
    if any(row.get("initial_status") != "pending" for row in records):
        raise ValueError("P0-B questions must be pending at contract freeze")
    if contract.get("decision_rules") != P0B_DECISION_RULES:
        raise ValueError("P0-B fixed decision rules drifted")
    summary_artifacts = contract.get("fixed_aggregation", {}).get(
        "summary_artifacts", {})
    if tuple(summary_artifacts.get("manifest_artifacts_in_order", ())) != (
            P0B_SUMMARY_ARTIFACTS):
        raise ValueError("P0-B fixed summary artifacts drifted")
    if summary_artifacts.get("manifest_filename") != P0B_SUMMARY_MANIFEST:
        raise ValueError("P0-B summary manifest filename drifted")
    if summary_artifacts.get("manifest_self_listing_allowed") is not False:
        raise ValueError("P0-B summary manifest must not list itself")
    if tuple(summary_artifacts.get(
            "final_acceptance_artifacts_in_order", ())) != (
            P0B_SUMMARY_ARTIFACTS + (P0B_SUMMARY_MANIFEST,)):
        raise ValueError("P0-B final acceptance artifact list drifted")
    if summary_artifacts.get("final_acceptance_filename") != (
            P0B_COMPLETION_MARKER):
        raise ValueError("P0-B final acceptance filename drifted")
    if summary_artifacts.get("final_acceptance_written_exclusively_last") is not True:
        raise ValueError("P0-B final acceptance must be written exclusively last")
    return contract


def load_p0b_contract(
        path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    """Load and validate the frozen contract, returning its file SHA256."""
    path = Path(path)
    contract = strict_json_loads(path.read_bytes(), label=f"P0-B contract {path}")
    if not isinstance(contract, dict):
        raise ValueError("P0-B contract JSON root must be an object")
    validate_p0b_contract(contract)
    return contract, sha256_file(path)


def _numeric_weights(weights: Any, name: str = "weights") -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return values


def effective_sample_size(weights: Any) -> float:
    """Return ``(sum q)^2 / sum(q^2)`` for finite nonnegative weights."""
    values = _numeric_weights(weights)
    denominator = float(np.dot(values, values))
    if denominator == 0.0:
        return 0.0
    total = float(values.sum())
    result = total * total / denominator
    if not np.isfinite(result):
        raise FloatingPointError("non-finite effective sample size")
    return float(result)


def predictive_nis_unlocalized(
        innovation: Any, observation_anomalies: Any,
        diagonal_r_precision: Any, n_members: int | None = None,
        ) -> tuple[float, int]:
    """Return exact query-level unlocalized predictive NIS and token dof.

    ``observation_anomalies`` is ``Y`` with shape ``[tokens, members]`` and
    ``diagonal_r_precision`` is the unlocalized diagonal ``P``.  The frozen
    Woodbury expression is

    ``d.T P d - (Y.T P d).T ((N-1)I + Y.T P Y)^-1 (Y.T P d)``.

    Localization is intentionally absent.  The returned dof is the number of
    strictly positive precision tokens; callers report NIS/dof separately.
    """
    d = np.asarray(innovation, dtype=np.float64)
    y = np.asarray(observation_anomalies, dtype=np.float64)
    precision = np.asarray(diagonal_r_precision, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("innovation must have shape [tokens]")
    if y.ndim != 2:
        raise ValueError("observation_anomalies must have shape [tokens, members]")
    if precision.ndim != 1:
        raise ValueError("diagonal_r_precision must have shape [tokens]")
    if y.shape[0] != len(d) or len(precision) != len(d):
        raise ValueError("predictive NIS token dimensions disagree")
    inferred_members = int(y.shape[1])
    if n_members is None:
        n_members = inferred_members
    if isinstance(n_members, bool) or not isinstance(n_members, (int, np.integer)):
        raise ValueError("n_members must be an integer")
    n_members = int(n_members)
    if n_members != inferred_members:
        raise ValueError("n_members disagrees with observation_anomalies")
    if n_members < 2:
        raise ValueError("n_members must be at least two")
    if (not np.isfinite(d).all() or not np.isfinite(y).all()
            or not np.isfinite(precision).all()):
        raise ValueError("predictive NIS inputs must be finite")
    if np.any(precision < 0.0):
        raise ValueError("diagonal_r_precision must be nonnegative")

    positive = precision > 0.0
    dof = int(np.count_nonzero(positive))
    if dof == 0:
        return 0.0, 0
    d = d[positive]
    y = y[positive]
    precision = precision[positive]
    weighted_d = precision * d
    rhs = y.T @ weighted_d
    system = ((n_members - 1) * np.eye(n_members, dtype=np.float64)
              + y.T @ (precision[:, None] * y))
    solved = np.linalg.solve(system, rhs)
    diagonal_energy = float(d @ weighted_d)
    result = float(diagonal_energy - rhs @ solved)
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, diagonal_energy)
    if result < -roundoff:
        raise FloatingPointError("predictive NIS became materially negative")
    return max(0.0, result), dof


def validate_query_predictive_nis_fields(
        query: Mapping[str, Any]) -> dict[str, int]:
    """Validate query predictive-NIS storage and frozen dof closure."""
    validate_table_columns(query, QUERY_PREDICTIVE_NIS_FIELDS)
    validate_strict_finite(query, QUERY_PREDICTIVE_NIS_FIELDS)
    dofs: dict[str, np.ndarray] = {}
    for mode in ("FY", "COSMIC", "joint"):
        value = _column(query, f"predictive_nis_unlocalized_{mode}")
        dof = _column(query, f"predictive_nis_unlocalized_{mode}_dof")
        if np.any(value < 0.0):
            raise ValueError(f"{mode} predictive NIS must be nonnegative")
        if (not np.issubdtype(dof.dtype, np.integer)
                and not np.array_equal(dof, np.rint(dof))):
            raise ValueError(f"{mode} predictive NIS dof must contain integers")
        if np.any(dof < 0):
            raise ValueError(f"{mode} predictive NIS dof must be nonnegative")
        if np.any((dof == 0) & (value != 0.0)):
            raise ValueError(f"{mode} zero-dof predictive NIS must be zero")
        dofs[mode] = dof.astype(np.int64, copy=False)
    if not np.array_equal(dofs["joint"], dofs["FY"] + dofs["COSMIC"]):
        raise ValueError("joint predictive NIS dof does not close by source")
    return {
        mode: int(np.sum(values)) for mode, values in dofs.items()
    }


def profile_precision_statistics(
        profile_ids: Any, localized_precision: Any) -> dict[str, Any]:
    """Aggregate localized precision by profile for one query and source."""
    precision = _numeric_weights(localized_precision, "localized_precision")
    ids_raw = np.asarray(profile_ids)
    if ids_raw.ndim != 1 or len(ids_raw) != len(precision):
        raise ValueError("profile_ids and localized_precision must be equal 1-D arrays")
    if not np.issubdtype(ids_raw.dtype, np.integer):
        if (not np.issubdtype(ids_raw.dtype, np.number)
                or not np.isfinite(ids_raw).all()
                or not np.array_equal(ids_raw, np.rint(ids_raw))):
            raise ValueError("profile_ids must contain finite integers")
    ids = ids_raw.astype(np.int64, copy=False)
    if np.any(ids < 0):
        raise ValueError("profile_ids must be nonnegative")
    if len(ids) == 0:
        return {
            "token_count": 0,
            "unique_profile_count": 0,
            "localized_precision_sum": 0.0,
            "token_neff": 0.0,
            "profile_neff": 0.0,
            "max_profile_precision_share": 0.0,
            "dominant_profile_id": None,
        }

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    profile_precision = np.zeros(len(unique_ids), dtype=np.float64)
    np.add.at(profile_precision, inverse, precision)
    total = float(profile_precision.sum())
    if total == 0.0:
        dominant_profile_id = None
        dominant_share = 0.0
    else:
        # np.unique sorts IDs, so np.argmax provides the frozen lowest-ID tie break.
        dominant_index = int(np.argmax(profile_precision))
        dominant_profile_id = int(unique_ids[dominant_index])
        dominant_share = float(profile_precision[dominant_index] / total)
    return {
        "token_count": int(len(ids)),
        "unique_profile_count": int(len(unique_ids)),
        "localized_precision_sum": total,
        "token_neff": effective_sample_size(precision),
        "profile_neff": effective_sample_size(profile_precision),
        "max_profile_precision_share": dominant_share,
        "dominant_profile_id": dominant_profile_id,
    }


def dominant_profile_precision_share(
        profile_ids: Any, localized_precision: Any) -> tuple[int | None, float]:
    """Return the frozen dominant-profile ID and its localized-precision share."""
    stats = profile_precision_statistics(profile_ids, localized_precision)
    return (
        stats["dominant_profile_id"],
        stats["max_profile_precision_share"],
    )


def _column(table: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in table:
        raise ValueError(f"missing table field: {name}")
    value = np.asarray(table[name])
    if value.ndim != 1:
        raise ValueError(f"table field must be one-dimensional: {name}")
    if value.dtype.kind == "O":
        raise ValueError(f"object arrays are forbidden: {name}")
    return value


def validate_table_columns(
        table: Mapping[str, Any], required_fields: Sequence[str]) -> int:
    """Require one-dimensional, non-object, equal-length table columns."""
    arrays = [_column(table, name) for name in required_fields]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("table columns have inconsistent lengths")
    return lengths.pop()


def validate_table_contract_fields(
        table: Mapping[str, Any],
        table_contract: Mapping[str, Any],
        table_name: str) -> int:
    """Require exact fields, frozen dtypes, and finite numeric columns."""
    required = tuple(table_contract.get("required_fields", ()))
    if not required or len(required) != len(set(required)):
        raise ValueError(f"invalid frozen {table_name} required_fields")
    dtype_map = table_contract.get("dtypes", {})
    if set(dtype_map) != set(required):
        raise ValueError(f"invalid frozen {table_name} dtype map")
    actual = set(table)
    expected = set(required)
    if actual != expected:
        raise ValueError(
            f"{table_name} fields differ from frozen contract: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    count = validate_table_columns(table, required)
    for name in required:
        value = _column(table, name)
        expected_dtype = np.dtype(dtype_map[name])
        if value.dtype != expected_dtype:
            raise TypeError(
                f"{table_name}.{name} dtype {value.dtype} differs from "
                f"frozen {expected_dtype}")
        if value.dtype.kind in "biufc":
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite {table_name} field: {name}")
        elif value.dtype.kind not in "US":
            raise ValueError(f"unsupported {table_name} field dtype: {name}")
    return count


def validate_strict_finite(
        table: Mapping[str, Any], numeric_fields: Sequence[str]) -> None:
    """Reject absent, non-numeric, or non-finite core numeric fields."""
    for name in numeric_fields:
        value = _column(table, name)
        if value.dtype.kind not in "biufc":
            raise ValueError(f"core field is not numeric: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"core field is not finite: {name}")


def _single_scope(table: Mapping[str, Any], table_name: str) -> tuple[str, str, int] | None:
    count = validate_table_columns(
        table, ("station", "date_utc", "batch_id"))
    if count == 0:
        return None
    station = np.unique(_column(table, "station"))
    date = np.unique(_column(table, "date_utc"))
    batch = np.unique(_column(table, "batch_id"))
    if len(station) != 1 or len(date) != 1 or len(batch) != 1:
        raise ValueError(f"{table_name} contains more than one partition scope")
    return str(station[0]), str(date[0]), int(batch[0])


def _assert_unique_rows(columns: Sequence[np.ndarray], label: str) -> None:
    rows = list(zip(*(column.tolist() for column in columns)))
    if len(rows) != len(set(rows)):
        raise ValueError(f"duplicate {label}")


def validate_foreign_keys(
        query: Mapping[str, Any],
        token: Mapping[str, Any],
        edge: Mapping[str, Any]) -> dict[str, int]:
    """Validate normalized table identity, deduplication, scope, and FKs."""
    query_count = validate_table_columns(query, QUERY_ID_FIELDS)
    token_count = validate_table_columns(token, TOKEN_ID_FIELDS)
    edge_count = validate_table_columns(edge, EDGE_ID_FIELDS)
    if query_count == 0:
        raise ValueError("query batch must not be empty")

    query_ids = _column(query, "query_id").astype(np.int64, copy=False)
    token_rows = _column(token, "token_row_id").astype(np.int64, copy=False)
    edge_queries = _column(edge, "query_id").astype(np.int64, copy=False)
    edge_tokens = _column(edge, "token_row_id").astype(np.int64, copy=False)
    _assert_unique_rows((query_ids,), "query primary key")
    _assert_unique_rows((_column(query, "sample_key"),), "query sample key")
    _assert_unique_rows((token_rows,), "token primary key")
    _assert_unique_rows((
        _column(token, "source"),
        _column(token, "profile_id").astype(np.int64, copy=False),
        _column(token, "token_id").astype(np.int64, copy=False),
    ), "source-profile-token identity")
    _assert_unique_rows((edge_queries, edge_tokens), "query-token edge")

    query_set = set(query_ids.tolist())
    token_set = set(token_rows.tolist())
    if not set(edge_queries.tolist()).issubset(query_set):
        raise ValueError("edge query_id violates foreign key")
    if not set(edge_tokens.tolist()).issubset(token_set):
        raise ValueError("edge token_row_id violates foreign key")
    if token_set and set(edge_tokens.tolist()) != token_set:
        raise ValueError("token table contains orphan rows")

    query_split = _column(query, "query_split") if "query_split" in query else None
    if query_split is not None and not set(map(str, query_split)).issubset(
            P0B_QUERY_SPLITS):
        raise ValueError("query table contains a forbidden date partition")
    profile_split = _column(token, "profile_split")
    if token_count and set(map(str, profile_split)) != {"train"}:
        raise ValueError("token table contains non-train profiles")

    token_sources = _column(token, "source")
    edge_sources = _column(edge, "source")
    if not set(map(str, token_sources)).issubset(P0B_SOURCES):
        raise ValueError("token table contains an unknown source")
    if not set(map(str, edge_sources)).issubset(P0B_SOURCES):
        raise ValueError("edge table contains an unknown source")
    token_source_by_id = {
        int(row): str(source) for row, source in zip(token_rows, token_sources)
    }
    for row, source in zip(edge_tokens, edge_sources):
        if token_source_by_id[int(row)] != str(source):
            raise ValueError("edge source disagrees with token source")

    query_scope = _single_scope(query, "query")
    token_scope = _single_scope(token, "token")
    edge_scope = _single_scope(edge, "edge")
    for scope in (token_scope, edge_scope):
        if scope is not None and scope != query_scope:
            raise ValueError("table partition scopes disagree")
    return {
        "query_rows": query_count,
        "token_rows": token_count,
        "edge_rows": edge_count,
    }


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    return float(np.max(np.abs(
        actual.astype(np.float64) - expected.astype(np.float64))))


def _require_allclose(
        name: str, actual: np.ndarray, expected: np.ndarray,
        atol: float, rtol: float) -> float:
    error = _max_abs_error(actual, expected)
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"closure failed for {name}: max_abs_error={error:.9g}")
    return error


def validate_query_closures(
        query: Mapping[str, Any], atol: float = 5e-6,
        rtol: float = 5e-6) -> dict[str, Any]:
    """Validate no-token and M00/M10/M01/M11 algebra."""
    validate_table_columns(query, QUERY_CLOSURE_FIELDS)
    validate_strict_finite(query, QUERY_CLOSURE_FIELDS)
    arrays = {name: _column(query, name) for name in QUERY_CLOSURE_FIELDS}
    m00 = arrays["M00_log10_ne"]
    no_token = arrays["no_token_log10_ne"]
    if (m00.dtype != no_token.dtype or m00.shape != no_token.shape
            or m00.tobytes() != no_token.tobytes()):
        raise ValueError("no-token output is not bitwise identical to M00")

    errors = {
        "M10": _require_allclose(
            "M10-M00", arrays["M10_log10_ne"] - m00,
            arrays["isolated_increment_FY_dex"], atol, rtol),
        "M01": _require_allclose(
            "M01-M00", arrays["M01_log10_ne"] - m00,
            arrays["isolated_increment_COSMIC_dex"], atol, rtol),
        "M11_sources": _require_allclose(
            "joint source sum",
            arrays["joint_update_FY_dex"]
            + arrays["joint_update_COSMIC_dex"],
            arrays["M11_log10_ne"] - m00, atol, rtol),
        "M11_increment": _require_allclose(
            "joint increment", arrays["joint_increment_dex"],
            arrays["M11_log10_ne"] - m00, atol, rtol),
    }
    return {"no_token_bitwise_equal": True, "max_abs_errors": errors}


def validate_edge_contribution_closures(
        query: Mapping[str, Any], token: Mapping[str, Any],
        edge: Mapping[str, Any], atol: float = 5e-6,
        rtol: float = 5e-6) -> dict[str, float]:
    """Validate edge algebra and query/source contribution closures."""
    validate_foreign_keys(query, token, edge)
    validate_table_columns(edge, EDGE_ALGEBRA_FIELDS)
    validate_strict_finite(edge, EDGE_ALGEBRA_FIELDS)
    validate_strict_finite(query, (
        "joint_update_FY_dex", "joint_update_COSMIC_dex",
        "isolated_increment_FY_dex", "isolated_increment_COSMIC_dex"))

    innovation = _column(edge, "innovation_dex")
    joint_expected = _column(edge, "gain_joint") * innovation
    isolated_expected = _column(edge, "gain_isolated") * innovation
    errors = {
        "edge_joint_algebra": _require_allclose(
            "edge joint gain x innovation",
            _column(edge, "contribution_joint_dex"),
            joint_expected, atol, rtol),
        "edge_isolated_algebra": _require_allclose(
            "edge isolated gain x innovation",
            _column(edge, "contribution_isolated_dex"),
            isolated_expected, atol, rtol),
    }

    query_ids = _column(query, "query_id").astype(np.int64, copy=False)
    query_row = {int(value): index for index, value in enumerate(query_ids)}
    edge_query = _column(edge, "query_id").astype(np.int64, copy=False)
    row_index = np.asarray([query_row[int(value)] for value in edge_query])
    sources = _column(edge, "source").astype(str)
    for source in P0B_SOURCES:
        selected = sources == source
        joint_sum = np.zeros(len(query_ids), dtype=np.float64)
        isolated_sum = np.zeros(len(query_ids), dtype=np.float64)
        np.add.at(
            joint_sum, row_index[selected],
            _column(edge, "contribution_joint_dex")[selected])
        np.add.at(
            isolated_sum, row_index[selected],
            _column(edge, "contribution_isolated_dex")[selected])
        joint_field = _column(query, f"joint_update_{source}_dex")
        isolated_field = _column(query, f"isolated_increment_{source}_dex")
        errors[f"{source}_joint_sum"] = _require_allclose(
            f"{source} joint edge sum", joint_sum, joint_field, atol, rtol)
        errors[f"{source}_isolated_sum"] = _require_allclose(
            f"{source} isolated edge sum", isolated_sum,
            isolated_field, atol, rtol)
    return errors


def validate_cross_table_diagnostics(
        query: Mapping[str, Any], token: Mapping[str, Any],
        edge: Mapping[str, Any], atol: float = 5e-6,
        rtol: float = 5e-6) -> dict[str, float]:
    """Recompute query coverage and precision summaries from token/edge rows."""
    validate_foreign_keys(query, token, edge)
    query_ids = _column(query, "query_id").astype(np.int64, copy=False)
    edge_query = _column(edge, "query_id").astype(np.int64, copy=False)
    edge_token = _column(edge, "token_row_id").astype(np.int64, copy=False)
    edge_source = _column(edge, "source").astype(str)
    token_rows = _column(token, "token_row_id").astype(np.int64, copy=False)
    token_sources = _column(token, "source").astype(str)
    token_profiles = _column(token, "profile_id").astype(np.int64, copy=False)
    token_lookup = {
        int(row): (source, int(profile))
        for row, source, profile in zip(
            token_rows.tolist(), token_sources.tolist(), token_profiles.tolist())
    }
    unlocalized = _numeric_weights(
        _column(edge, "unlocalized_precision"), "unlocalized_precision")
    localized = _numeric_weights(
        _column(edge, "localized_precision"), "localized_precision")
    maximum_error = 0.0

    def integer_value(field: str, row: int) -> int:
        column = _column(query, field)
        value = column[row]
        if (not np.issubdtype(column.dtype, np.integer)
                and (not np.issubdtype(column.dtype, np.number)
                     or not np.isfinite(value)
                     or float(value) != float(np.rint(value)))):
            raise ValueError(f"cross-table {field} must contain integers")
        return int(value)

    def require_close(label: str, actual: float, expected: float) -> None:
        nonlocal maximum_error
        error = abs(float(actual) - float(expected))
        maximum_error = max(maximum_error, error)
        if not np.isclose(actual, expected, rtol=rtol, atol=atol):
            raise ValueError(
                f"cross-table diagnostic mismatch for {label}: "
                f"actual={actual}, expected={expected}")

    def coverage_code(fy_present: bool, cosmic_present: bool) -> str:
        if fy_present and cosmic_present:
            return "joint"
        if fy_present:
            return "FY_only"
        if cosmic_present:
            return "COSMIC_only"
        return "no_token"

    for query_row, query_id in enumerate(query_ids.tolist()):
        raw_presence: dict[str, bool] = {}
        effective_presence: dict[str, bool] = {}
        for source in P0B_SOURCES:
            selected = (edge_query == query_id) & (edge_source == source)
            selected_rows = np.flatnonzero(selected)
            profiles = np.asarray([
                token_lookup[int(edge_token[index])][1]
                for index in selected_rows
            ], dtype=np.int64)
            if any(token_lookup[int(edge_token[index])][0] != source
                   for index in selected_rows):
                raise ValueError("edge source differs from token source")
            localized_source = localized[selected]
            unlocalized_source = unlocalized[selected]
            stats = profile_precision_statistics(profiles, localized_source)
            raw_presence[source] = len(selected_rows) > 0
            effective_presence[source] = (
                stats["localized_precision_sum"] > 0.0)
            integer_fields = {
                "token_count": stats["token_count"],
                "unique_profile_count": stats["unique_profile_count"],
            }
            for field, expected in integer_fields.items():
                actual = integer_value(f"{source}_{field}", query_row)
                if actual != int(expected):
                    raise ValueError(
                        f"cross-table {source}_{field} mismatch: "
                        f"actual={actual}, expected={expected}")
            floating_fields = {
                "unlocalized_precision_sum": float(unlocalized_source.sum()),
                "localized_precision_sum": stats["localized_precision_sum"],
                "token_neff": stats["token_neff"],
                "profile_neff": stats["profile_neff"],
                "max_profile_precision_share": stats[
                    "max_profile_precision_share"],
            }
            for field, expected in floating_fields.items():
                require_close(
                    f"{source}_{field}",
                    float(_column(query, f"{source}_{field}")[query_row]),
                    expected)
            dominant = stats["dominant_profile_id"]
            expected_valid = dominant is not None
            expected_id = -1 if dominant is None else int(dominant)
            actual_valid = bool(_column(
                query, f"{source}_dominant_profile_valid")[query_row])
            actual_id = integer_value(
                f"{source}_dominant_profile_id", query_row)
            if actual_valid != expected_valid or actual_id != expected_id:
                raise ValueError(
                    f"{source} dominant-profile sentinel/ID mismatch")
            expected_dof = int(np.count_nonzero(unlocalized_source > 0.0))
            actual_dof = integer_value(
                f"predictive_nis_unlocalized_{source}_dof", query_row)
            if actual_dof != expected_dof:
                raise ValueError(f"{source} predictive NIS dof mismatch")

        raw_expected = coverage_code(
            raw_presence["FY"], raw_presence["COSMIC"])
        effective_expected = coverage_code(
            effective_presence["FY"], effective_presence["COSMIC"])
        if str(_column(query, "raw_coverage_code")[query_row]) != raw_expected:
            raise ValueError("raw coverage code does not match edge presence")
        if str(_column(query, "coverage_code")[query_row]) != effective_expected:
            raise ValueError(
                "effective coverage code does not match localized precision")
        joint_dof = integer_value(
            "predictive_nis_unlocalized_joint_dof", query_row)
        source_dof = sum(integer_value(
            f"predictive_nis_unlocalized_{source}_dof", query_row)
            for source in P0B_SOURCES)
        if joint_dof != source_dof:
            raise ValueError("joint predictive NIS dof does not equal source sum")
    return {"max_abs_recomputed_summary_error": maximum_error}


def validate_audit_tables(
        query: Mapping[str, Any], token: Mapping[str, Any],
        edge: Mapping[str, Any], atol: float = 5e-6,
        rtol: float = 5e-6,
        contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the reusable core integrity checks for one table triplet."""
    if contract is not None:
        validate_p0b_contract(contract)
        schemas = contract_table_schemas(contract)
        validate_table_contract_fields(query, schemas["query"], "query")
        validate_table_contract_fields(token, schemas["token"], "token")
        validate_table_contract_fields(edge, schemas["edge"], "edge")
    counts = validate_foreign_keys(query, token, edge)
    query_closure = validate_query_closures(query, atol=atol, rtol=rtol)
    nis_dof_totals = validate_query_predictive_nis_fields(query)
    edge_closure = validate_edge_contribution_closures(
        query, token, edge, atol=atol, rtol=rtol)
    recomputed_diagnostics = validate_cross_table_diagnostics(
        query, token, edge, atol=atol, rtol=rtol)
    return {
        "counts": counts,
        "query_closure": query_closure,
        "predictive_nis_dof_totals": nis_dof_totals,
        "edge_closure_max_abs_errors": edge_closure,
        "recomputed_diagnostics": recomputed_diagnostics,
    }


__all__ = [
    "P0B_AUDIT_SCHEMA_VERSION",
    "P0B_CACHE_COMPLETION_MARKER",
    "P0B_CRITICAL_TRACKED_PATHS",
    "P0B_DECISION_RULES",
    "P0B_EXPECTED_DEVELOPMENT_PROFILE_COUNTS",
    "P0B_EXPECTED_FINITE_QUERY_COUNTS",
    "P0B_EXPECTED_TRAIN_PROFILE_COUNTS",
    "P0B_COMPLETION_MARKER",
    "P0B_CONTRACT_FILENAME",
    "P0B_CONTRACT_ID",
    "P0B_COVERAGE_CODES",
    "P0B_DEFAULT_OUTPUT_DIRECTORY",
    "P0B_HEIGHT_DELETE_BANDS",
    "P0B_IMPLEMENTATION_TAG",
    "P0B_BASE_ANCHOR_COMMIT",
    "P0B_PREDICTIVE_NIS_DOF",
    "P0B_PREDICTIVE_NIS_FORMULA",
    "P0B_PREDICTIVE_NIS_PRECISION",
    "P0B_QUERY_SPLITS",
    "P0B_P0A_CONTRACT_SHA256",
    "P0B_P0A_CONTRACT_SIZE_BYTES",
    "P0B_PYTHON_EXECUTABLE",
    "P0B_REQUIRED_BRANCH",
    "P0B_QUESTION_TERMINAL_STATUSES",
    "P0B_SOURCES",
    "P0B_SUMMARY_ARTIFACTS",
    "P0B_SUMMARY_MANIFEST",
    "P0B_TABLE_DTYPE_NAMES",
    "P0B_TRAIN_TOKEN_ARRAYS",
    "P0B_TRAIN_TOKEN_DIRECTORY_SEMANTICS",
    "P0B_TRAIN_TOKEN_DTYPES",
    "artifact_identity",
    "atomic_write_json",
    "build_train_token_directory_identity",
    "contract_table_schemas",
    "dominant_profile_precision_share",
    "effective_sample_size",
    "current_git_provenance",
    "current_python_environment",
    "current_source_provenance",
    "load_p0b_contract",
    "profile_precision_statistics",
    "predictive_nis_unlocalized",
    "sha256_file",
    "strict_json_loads",
    "tracked_git_status",
    "validate_audit_tables",
    "validate_cross_table_diagnostics",
    "validate_edge_contribution_closures",
    "validate_foreign_keys",
    "validate_p0b_contract",
    "validate_query_closures",
    "validate_query_predictive_nis_fields",
    "validate_strict_finite",
    "validate_table_contract_fields",
    "validate_table_columns",
    "validate_train_token_directory_identity",
    "write_completion_marker_atomically",
]
