"""Read-only M2-W2 P0-B error-chain audit.

The runner binds a strict v14 Analysis checkpoint to the frozen P0-B and P0-A
contracts, rebuilds the registered finite ISR queries from the raw loaders, and
writes normalized query/token/edge NPZ shards.  Satellite observation tokens
are restricted to the frozen *train* partition.  No optimizer, gradient, QC,
checkpoint selection, or satellite development inference is reachable here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inr_modules.mdia.checkpoint_io import load_fsia_analysis_checkpoint
from inr_modules.mdia.fsia_model import (
    _compute_solar_features,
    solve_density_modes,
)
from inr_modules.mdia.p0b_audit import (
    P0B_AUDIT_SCHEMA_VERSION,
    P0B_BASE_ANCHOR_COMMIT,
    P0B_CACHE_COMPLETION_MARKER,
    P0B_CONTRACT_FILENAME,
    P0B_CRITICAL_TRACKED_PATHS,
    P0B_IMPLEMENTATION_TAG,
    P0B_REQUIRED_BRANCH,
    artifact_identity,
    atomic_write_json,
    build_train_token_directory_identity,
    contract_table_schemas,
    load_p0b_contract,
    profile_precision_statistics,
    sha256_file,
    strict_json_loads,
    tracked_git_status,
    validate_audit_tables,
    validate_cross_table_diagnostics,
    validate_train_token_directory_identity,
    write_completion_marker_atomically,
)
from inr_modules.mdia.p0b_counterfactuals import (
    RaggedSourceTerms,
    run_p0b_counterfactual_suite,
)
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    query_observation_payload,
)


EXPECTED_DOMAIN = "hybrid_120_500_model_200_500_observation_v1"
EXPECTED_CHECKPOINT_FORMAT = 14
EXPECTED_MODEL_RANGE = (120.0, 500.0)
EXPECTED_OBSERVATION_RANGE = (200.0, 500.0)
EXPECTED_AACGMV2_VERSION = "2.7.0"
EXPECTED_IMPLEMENTATION_BRANCH = P0B_REQUIRED_BRANCH
EXPECTED_BASE_ANCHOR = P0B_BASE_ANCHOR_COMMIT
EXPECTED_IMPLEMENTATION_TAG = P0B_IMPLEMENTATION_TAG
EXPECTED_PYTHON_EXECUTABLE = Path(
    r"C:\Users\12238\.conda\envs\pytorch_cpu\python.exe")
CRITICAL_TRACKED_PATHS = P0B_CRITICAL_TRACKED_PATHS
SOURCES = ("FY", "COSMIC")
DEFAULT_P0B_CONTRACT = ROOT / "m2w2_contracts" / P0B_CONTRACT_FILENAME
DEFAULT_P0A_DIRECTORY = (
    ROOT / "isr_validation_outputs"
    / "run67-p0a-v14-epoch15-vs-v13-epoch12-isr-qav2-r2"
)
PREFLIGHT_ACCEPTANCE_FILENAME = "preflight_acceptance.json"


QUERY_DTYPES = {
    "query_id": np.int64,
    "sample_key": "U64",
    "station": "U16",
    "date_utc": "U8",
    "batch_id": np.int64,
    "query_profile_id": np.int64,
    "timestamp_unix": np.int64,
    "query_split": "U11",
    "latitude_deg": np.float32,
    "longitude_deg": np.float32,
    "altitude_km": np.float32,
    "relative_hour": np.float32,
    "local_time_hour": np.float32,
    "aacgm_latitude_deg": np.float32,
    "aacgm_mlt_hour": np.float32,
    "cos_sza": np.float32,
    "kp": np.float32,
    "f107": np.float32,
    "isr_log10_ne": np.float32,
    "raw_iri_log10_ne": np.float32,
    "M00_log10_ne": np.float32,
    "M10_log10_ne": np.float32,
    "M01_log10_ne": np.float32,
    "M11_log10_ne": np.float32,
    "no_token_log10_ne": np.float32,
    "isolated_increment_FY_dex": np.float32,
    "isolated_increment_COSMIC_dex": np.float32,
    "joint_update_FY_dex": np.float32,
    "joint_update_COSMIC_dex": np.float32,
    "joint_increment_dex": np.float32,
    "raw_coverage_code": "U12",
    "coverage_code": "U12",
    "FY_token_count": np.int64,
    "FY_unique_profile_count": np.int64,
    "FY_unlocalized_precision_sum": np.float32,
    "FY_localized_precision_sum": np.float32,
    "FY_token_neff": np.float32,
    "FY_profile_neff": np.float32,
    "FY_max_profile_precision_share": np.float32,
    "predictive_nis_unlocalized_FY": np.float32,
    "predictive_nis_unlocalized_FY_dof": np.int64,
    "COSMIC_token_count": np.int64,
    "COSMIC_unique_profile_count": np.int64,
    "COSMIC_unlocalized_precision_sum": np.float32,
    "COSMIC_localized_precision_sum": np.float32,
    "COSMIC_token_neff": np.float32,
    "COSMIC_profile_neff": np.float32,
    "COSMIC_max_profile_precision_share": np.float32,
    "predictive_nis_unlocalized_COSMIC": np.float32,
    "predictive_nis_unlocalized_COSMIC_dof": np.int64,
    "predictive_nis_unlocalized_joint": np.float32,
    "predictive_nis_unlocalized_joint_dof": np.int64,
    "CF_drop_200_250_log10_ne": np.float32,
    "CF_drop_250_300_log10_ne": np.float32,
    "CF_drop_300_400_log10_ne": np.float32,
    "CF_drop_400_500_log10_ne": np.float32,
    "CF_duplicate_FY_dominant_profile_log10_ne": np.float32,
    "CF_duplicate_COSMIC_dominant_profile_log10_ne": np.float32,
    "CF_duplicate_both_dominant_profiles_log10_ne": np.float32,
    "FY_dominant_profile_id": np.int64,
    "COSMIC_dominant_profile_id": np.int64,
    "FY_dominant_profile_valid": np.bool_,
    "COSMIC_dominant_profile_valid": np.bool_,
}

TOKEN_DTYPES = {
    "token_row_id": np.int64,
    "station": "U16",
    "date_utc": "U8",
    "batch_id": np.int64,
    "source": "U6",
    "profile_id": np.int64,
    "token_id": np.int64,
    "profile_split": "U5",
    "latitude_deg": np.float32,
    "longitude_deg": np.float32,
    "altitude_km": np.float32,
    "relative_hour": np.float32,
    "observation_log10_ne": np.float32,
    "background_log10_ne": np.float32,
}

EDGE_DTYPES = {
    "station": "U16",
    "date_utc": "U8",
    "batch_id": np.int64,
    "query_id": np.int64,
    "token_row_id": np.int64,
    "source": "U6",
    "innovation_dex": np.float32,
    "r_variance_dex2": np.float32,
    "representativeness_weight": np.float32,
    "localization_weight": np.float32,
    "unlocalized_precision": np.float32,
    "localized_precision": np.float32,
    "prior_predictive_variance_dex2": np.float32,
    "r_standardized_innovation_sq": np.float32,
    "predictive_diagonal_nis": np.float32,
    "localized_innovation_energy": np.float32,
    "gain_joint": np.float32,
    "gain_isolated": np.float32,
    "contribution_joint_dex": np.float32,
    "contribution_isolated_dex": np.float32,
    "space_distance_km": np.float32,
    "time_distance_hours": np.float32,
}


@dataclass(frozen=True)
class P0AIdentity:
    contract_path: Path
    contract: dict[str, Any]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RawSelection:
    station: str
    date_utc: str
    query_split: str
    keys: np.ndarray
    query_ids: np.ndarray
    timestamps: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    altitudes: np.ndarray
    relative_hours: np.ndarray
    aacgm_latitudes: np.ndarray
    aacgm_mlt_hours: np.ndarray
    observations_log10: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.keys))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_bytes(), label=f"JSON artifact {path}")
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _parse_utc(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc)


def _coordinate_runtime_status(
        frozen_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Bind magnetic-coordinate semantics to the frozen Python environment."""
    coordinate_contract = frozen_contract.get("coordinate_enrichment", {})
    aacgm_contract = coordinate_contract.get("aacgm", {})
    try:
        import aacgmv2
    except ImportError as exc:
        raise RuntimeError("frozen P0-B requires aacgmv2") from exc
    version = str(getattr(aacgmv2, "__version__", ""))
    if version != str(aacgm_contract.get("version")):
        raise RuntimeError(
            "aacgmv2 version differs from the frozen P0-B contract: "
            f"actual={version!r}, expected={aacgm_contract.get('version')!r}")
    if aacgm_contract.get("method") != "ALLOWTRACE":
        raise ValueError("frozen P0-B AACGM method must be ALLOWTRACE")

    qd_contract = coordinate_contract.get("qd", {})
    apex_available = importlib.util.find_spec("apexpy") is not None
    if qd_contract.get("status") != "unavailable" or apex_available:
        raise RuntimeError(
            "P0-B v2 freezes QD as unavailable; ApexPy availability changed")
    return {
        "geographic": {
            "status": "computed",
            "fields": list(coordinate_contract["geographic"]["fields"]),
        },
        "aacgm": {
            "status": "computed",
            "package": "aacgmv2",
            "version": version,
            "method": "ALLOWTRACE",
            "fields": list(aacgm_contract["fields"]),
        },
        "qd": {
            "status": "unavailable",
            "package": "apexpy",
            "reason": str(qd_contract["reason"]),
            "proxy_substitution_used": False,
        },
        "fixed_aggregation_time_coordinate": "aacgm_mlt_hour",
    }


def _compute_aacgm_query_coordinates(
        latitudes: np.ndarray, longitudes: np.ndarray,
        altitudes_km: np.ndarray, timestamps_unix: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite AACGM latitude and magnetic local time per raw query."""
    import aacgmv2

    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    altitudes_km = np.asarray(altitudes_km, dtype=np.float64)
    timestamps_unix = np.asarray(timestamps_unix, dtype=np.int64)
    shapes = {
        latitudes.shape, longitudes.shape, altitudes_km.shape,
        timestamps_unix.shape,
    }
    if len(shapes) != 1 or latitudes.ndim != 1:
        raise ValueError("AACGM query coordinate arrays must be aligned 1-D arrays")
    if not (np.isfinite(latitudes).all() and np.isfinite(longitudes).all()
            and np.isfinite(altitudes_km).all()):
        raise ValueError("AACGM query inputs must be finite")
    magnetic_latitude = np.full(latitudes.shape, np.nan, dtype=np.float64)
    magnetic_mlt = np.full(latitudes.shape, np.nan, dtype=np.float64)
    for timestamp in np.unique(timestamps_unix):
        mask = timestamps_unix == timestamp
        when = dt.datetime.fromtimestamp(
            int(timestamp), tz=dt.timezone.utc)
        mlat, _mlon, mlt = aacgmv2.get_aacgm_coord_arr(
            latitudes[mask], longitudes[mask], altitudes_km[mask], when,
            method="ALLOWTRACE")
        magnetic_latitude[mask] = np.asarray(mlat, dtype=np.float64)
        magnetic_mlt[mask] = np.mod(np.asarray(mlt, dtype=np.float64), 24.0)
    if (not np.isfinite(magnetic_latitude).all()
            or not np.isfinite(magnetic_mlt).all()
            or np.any(magnetic_latitude < -90.0)
            or np.any(magnetic_latitude > 90.0)
            or np.any(magnetic_mlt < 0.0)
            or np.any(magnetic_mlt >= 24.0)):
        raise ValueError("AACGM conversion produced invalid coordinates")
    return (
        magnetic_latitude.astype(np.float32),
        magnetic_mlt.astype(np.float32),
    )


def _normal_station(value: str) -> str:
    normalized = str(value).replace(" ", "")
    if normalized not in ("Jicamarca", "PokerFlat"):
        raise ValueError(f"unsupported ISR station: {value!r}")
    return normalized


def _atomic_savez(path: Path, table: Mapping[str, np.ndarray]) -> Path:
    """Write, verify, and exclusively publish one no-pickle NPZ shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"P0-B NPZ shard already exists: {path}")
    arrays = {key: np.asarray(value) for key, value in table.items()}
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise ValueError("object arrays are forbidden in P0-B shards")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"stale P0-B NPZ temporary file exists: {temporary}")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays):
                raise ValueError("NPZ round-trip field set changed")
            for key, expected in arrays.items():
                actual = loaded[key]
                if actual.dtype.kind == "O" or actual.shape != expected.shape:
                    raise ValueError(f"invalid NPZ round-trip for {key}")
        # A same-directory hard link publishes the fully verified inode while
        # failing atomically if another writer already claimed the shard path.
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def _as_table(rows: Mapping[str, Iterable[Any]], dtypes: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = set(dtypes).difference(rows)
    extra = set(rows).difference(dtypes)
    if missing or extra:
        raise ValueError(
            f"table schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    result = {
        key: np.asarray(list(rows[key]), dtype=dtype)
        for key, dtype in dtypes.items()
    }
    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1:
        raise ValueError("table columns have inconsistent lengths")
    if any(value.dtype.kind == "O" for value in result.values()):
        raise ValueError("object arrays are forbidden")
    return result


def _empty_rows(schema: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {key: [] for key in schema}


def _load_npz_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]).copy() for key in loaded.files}


def _validate_runner_schemas(frozen_contract: Mapping[str, Any]) -> None:
    """Fail before inference if this runner cannot emit the frozen tables."""
    schemas = {
        "query": QUERY_DTYPES,
        "token": TOKEN_DTYPES,
        "edge": EDGE_DTYPES,
    }
    contract_tables = contract_table_schemas(frozen_contract)
    if set(contract_tables) != set(schemas):
        raise ValueError("frozen P0-B table set differs from runner schemas")
    for table_name, schema in schemas.items():
        required = tuple(contract_tables[table_name]["required_fields"])
        if tuple(schema) != required:
            raise ValueError(
                f"runner {table_name} field order/membership differs from frozen schema")
        for field, dtype in schema.items():
            if np.dtype(dtype).kind == "O":
                raise ValueError(
                    f"runner {table_name}.{field} uses forbidden object dtype")
            if np.dtype(dtype) != contract_tables[table_name]["dtypes"][field]:
                raise TypeError(
                    f"runner {table_name}.{field} dtype differs from frozen schema")


def _positive_integer_counts(
        value: Any, expected_keys: Iterable[str], label: str) -> dict[str, int]:
    """Normalize an exact positive-count mapping from the frozen contract."""
    keys = tuple(expected_keys)
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{label} must contain exactly {list(keys)}")
    result: dict[str, int] = {}
    for key in keys:
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise TypeError(f"{label}.{key} must be an integer")
        if int(count) <= 0:
            raise ValueError(f"{label}.{key} must be positive")
        result[key] = int(count)
    return result


def _contract_expected_profile_counts(
        frozen_contract: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Read train/development profile identities only from the frozen contract."""
    scope = frozen_contract.get("data_scope", {})
    train = _positive_integer_counts(
        scope.get("satellite_tokens", {}).get("expected_train_profile_counts"),
        SOURCES,
        "data_scope.satellite_tokens.expected_train_profile_counts",
    )
    development = _positive_integer_counts(
        scope.get("satellite_profile_index_metadata", {}).get(
            "expected_development_profile_counts"),
        SOURCES,
        (
            "data_scope.satellite_profile_index_metadata."
            "expected_development_profile_counts"
        ),
    )
    return {"train": train, "development": development}


def _contract_expected_query_counts(
        frozen_contract: Mapping[str, Any]) -> dict[str, int]:
    """Read only allowed train/development ISR counts from the frozen contract."""
    return _positive_integer_counts(
        frozen_contract.get("data_scope", {}).get("ISR_queries", {}).get(
            "expected_finite_query_counts"),
        ("train", "development"),
        "data_scope.ISR_queries.expected_finite_query_counts",
    )


def _validate_batch_against_contract(
        frozen_contract: Mapping[str, Any],
        query: Mapping[str, np.ndarray], token: Mapping[str, np.ndarray],
        edge: Mapping[str, np.ndarray]) -> dict[str, Any]:
    tables = {"query": query, "token": token, "edge": edge}
    schemas = {
        "query": QUERY_DTYPES,
        "token": TOKEN_DTYPES,
        "edge": EDGE_DTYPES,
    }
    contract_tables = contract_table_schemas(frozen_contract)
    for name, table in tables.items():
        required = tuple(contract_tables[name]["required_fields"])
        if tuple(table) != required:
            raise ValueError(
                f"{name} batch field order/membership differs from frozen contract")
        lengths = set()
        for field, value in table.items():
            array = np.asarray(value)
            if array.ndim != 1 or array.dtype.kind == "O":
                raise ValueError(f"invalid {name}.{field} storage")
            if array.dtype != np.dtype(schemas[name][field]):
                raise TypeError(
                    f"{name}.{field} dtype {array.dtype} differs from "
                    f"{np.dtype(schemas[name][field])}")
            if array.dtype != contract_tables[name]["dtypes"][field]:
                raise TypeError(
                    f"{name}.{field} dtype differs from frozen contract")
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError(f"{name} batch columns are misaligned")
    recomputed = validate_cross_table_diagnostics(query, token, edge)
    validation = validate_audit_tables(
        query, token, edge, contract=frozen_contract)
    if validation.get("recomputed_diagnostics") != recomputed:
        raise RuntimeError("cross-table diagnostic validation is not deterministic")
    return validation


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _python_environment() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    if os.path.normcase(str(executable)) != os.path.normcase(
            str(EXPECTED_PYTHON_EXECUTABLE.resolve())):
        raise RuntimeError(
            "P0-B must use the frozen pytorch_cpu Python executable: "
            f"{executable}")
    return {
        "executable": str(executable),
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _git_provenance(
        *, enforce: bool = True,
        required_implementation_ref: str = EXPECTED_IMPLEMENTATION_TAG,
        ) -> dict[str, Any]:
    rtk = shutil.which("rtk")
    if rtk is None:
        if enforce:
            raise RuntimeError("rtk is required for P0-B git identity checks")
        return {"status": "unavailable", "reason": "rtk_not_found"}

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            [rtk, "git", *arguments], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return completed.stdout.strip()

    try:
        head = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        implementation_commit = run(
            "rev-parse", f"{required_implementation_ref}^{{commit}}")
        implementation_tag_object = run(
            "rev-parse", f"{required_implementation_ref}^{{tag}}")
        implementation_ref_type = run(
            "cat-file", "-t", required_implementation_ref)
        tracked_status = tracked_git_status(ROOT)
        ancestor = subprocess.run(
            [rtk, "git", "merge-base", "--is-ancestor",
             EXPECTED_BASE_ANCHOR, head],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        untracked_critical = []
        for relative_path in CRITICAL_TRACKED_PATHS:
            try:
                run("ls-files", "--error-unmatch", "--", relative_path)
            except subprocess.CalledProcessError:
                untracked_critical.append(relative_path)
        critical_diff = subprocess.run(
            [rtk, "git", "diff", "--quiet", implementation_commit, "--",
             *CRITICAL_TRACKED_PATHS],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        provenance = {
            "status": "computed",
            "head": head,
            "branch": branch,
            "tracked_status": tracked_status,
            "expected_branch": EXPECTED_IMPLEMENTATION_BRANCH,
            "base_anchor": EXPECTED_BASE_ANCHOR,
            "base_anchor_is_ancestor": ancestor.returncode == 0,
            "required_implementation_tag": required_implementation_ref,
            "implementation_tag_object_sha": implementation_tag_object,
            "implementation_tag_type": implementation_ref_type,
            "implementation_tag_commit": implementation_commit,
            "head_equals_implementation_tag_commit": head == implementation_commit,
            "critical_tracked_paths": list(CRITICAL_TRACKED_PATHS),
            "untracked_critical_paths": untracked_critical,
            "critical_paths_clean_against_implementation_commit": (
                critical_diff.returncode == 0),
        }
        if enforce:
            if branch != EXPECTED_IMPLEMENTATION_BRANCH:
                raise ValueError(
                    "P0-B must run from branch "
                    f"{EXPECTED_IMPLEMENTATION_BRANCH}, got {branch!r}")
            if tracked_status:
                raise ValueError(
                    "P0-B requires a clean tracked worktree: "
                    f"{tracked_status[:5]}")
            if implementation_ref_type != "tag":
                raise ValueError(
                    "P0-B implementation ref must be an annotated Git tag")
            if head != implementation_commit:
                raise ValueError(
                    "P0-B runtime HEAD must equal the frozen implementation tag")
            if untracked_critical:
                raise ValueError(
                    "P0-B critical sources/contracts must be tracked: "
                    f"{untracked_critical}")
            if critical_diff.returncode != 0:
                detail = critical_diff.stderr.strip() or (
                    f"returncode={critical_diff.returncode}")
                raise ValueError(
                    "P0-B critical paths differ from the runtime implementation "
                    f"commit: {detail}")
            if ancestor.returncode != 0:
                detail = ancestor.stderr.strip() or f"returncode={ancestor.returncode}"
                raise ValueError(
                    "P0-B HEAD does not descend from the frozen base anchor: "
                    f"{detail}")
        return provenance
    except (OSError, subprocess.CalledProcessError) as exc:
        if enforce:
            raise RuntimeError("unable to establish P0-B git provenance") from exc
        return {"status": "unavailable", "reason": str(exc)}


def _assert_git_provenance_stable(
        initial: Mapping[str, Any], implementation_ref: str) -> dict[str, Any]:
    """Recheck the immutable implementation identity before publishing success."""
    current = _git_provenance(
        enforce=True, required_implementation_ref=implementation_ref)
    for field in (
            "head", "branch", "implementation_tag_object_sha",
            "implementation_tag_commit"):
        if current.get(field) != initial.get(field):
            raise ValueError(
                f"P0-B Git provenance changed during execution: {field}")
    return current


def _source_provenance(contract_path: Path, p0a_contract: Path) -> dict[str, Any]:
    paths = (
        p0a_contract,
        contract_path,
        *(ROOT / relative for relative in CRITICAL_TRACKED_PATHS),
    )
    return {
        str(path.resolve()): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in paths
    }


def _verified_marker_artifacts(
        marker: Mapping[str, Any], root: Path) -> tuple[
            dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Verify and parse each JSON artifact from the exact hashed bytes."""
    records = marker.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("preflight acceptance has no artifact inventory")
    verified: dict[str, dict[str, Any]] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("preflight artifact identity must be an object")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("preflight artifact identity has an invalid path")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("preflight artifact path escapes its directory") from exc
        if relative in verified:
            raise ValueError(f"duplicate preflight artifact identity: {relative}")
        payload = candidate.read_bytes()
        actual = {
            "path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if actual != record:
            raise ValueError(f"preflight artifact identity mismatch: {relative}")
        value = strict_json_loads(
            payload, label=f"preflight artifact {relative}")
        if not isinstance(value, dict):
            raise ValueError(f"preflight JSON root is not an object: {relative}")
        verified[relative] = dict(record)
        parsed[relative] = value
    return verified, parsed


def _profile_count_ledger_from_metadata(
        profile_metadata: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    try:
        return {
            "train": {
                source: int(profile_metadata[source]["train_unique_profiles"])
                for source in SOURCES
            },
            "development": {
                source: int(
                    profile_metadata[source]["development_unique_profiles"])
                for source in SOURCES
            },
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("preflight satellite profile metadata is incomplete") from exc


def _profile_id_sha256(profile_ids: np.ndarray) -> str:
    values = np.asarray(profile_ids, dtype=np.int64)
    if (values.ndim != 1 or len(values) == 0
            or len(np.unique(values)) != len(values)
            or (len(values) > 1 and np.any(values[1:] <= values[:-1]))):
        raise ValueError("profile-ID identity requires a nonempty sorted unique array")
    canonical = values.astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _train_allowlist_identity(
        allowlists: Mapping[str, np.ndarray],
        profile_metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    identity: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        values = np.asarray(allowlists[source], dtype=np.int64)
        digest = _profile_id_sha256(values)
        metadata = profile_metadata[source]
        if (metadata.get("train_profile_id_sha256") != digest
                or int(metadata.get("train_unique_profiles", -1)) != len(values)):
            raise ValueError(f"{source} train allowlist identity differs from metadata")
        identity[source] = {
            "profile_count": int(len(values)),
            "profile_id_sha256": digest,
            "profile_index_sha256": str(metadata["profile_index_sha256"]),
        }
    return identity


def _train_allowlist_identity_from_metadata(
        profile_metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    identity: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        metadata = profile_metadata.get(source)
        if not isinstance(metadata, dict):
            raise ValueError("preflight train allowlist metadata is incomplete")
        count = metadata.get("train_unique_profiles")
        digest = metadata.get("train_profile_id_sha256")
        index_digest = metadata.get("profile_index_sha256")
        if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
                or not isinstance(digest, str) or len(digest) != 64
                or not isinstance(index_digest, str) or len(index_digest) != 64):
            raise ValueError(f"invalid preflight train allowlist identity for {source}")
        identity[source] = {
            "profile_count": count,
            "profile_id_sha256": digest,
            "profile_index_sha256": index_digest,
        }
    return identity


def _validate_preflight_acceptance(
        acceptance_path: Path, *, output_dir: Path, checkpoint: Path,
        expected_checkpoint_sha256: str, frozen_contract: Mapping[str, Any],
        frozen_contract_path: Path, frozen_contract_sha256: str,
        p0a_contract_path: Path, git_provenance: Mapping[str, Any],
        python_environment: Mapping[str, Any],
        coordinate_status: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a completed strict preflight to the pending full audit.

    This gate is intentionally run before the full-audit output directory is
    created.  A missing, mutated, or identity-incompatible preflight therefore
    cannot retire a new full-audit path.
    """
    path = acceptance_path.resolve()
    if path.name != PREFLIGHT_ACCEPTANCE_FILENAME or not path.is_file():
        raise FileNotFoundError(
            f"full P0-B audit requires {PREFLIGHT_ACCEPTANCE_FILENAME}: {path}")
    root = path.parent.resolve()
    resolved_output = output_dir.resolve()
    try:
        resolved_output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "full-audit output directory must not equal or nest under preflight")
    marker_bytes = path.read_bytes()
    marker = strict_json_loads(
        marker_bytes, label="preflight acceptance")
    if not isinstance(marker, dict):
        raise ValueError("preflight acceptance root is not an object")
    marker_identity = {
        "path": str(path),
        "size_bytes": len(marker_bytes),
        "sha256": hashlib.sha256(marker_bytes).hexdigest(),
    }
    if (marker.get("audit_schema_version") != P0B_AUDIT_SCHEMA_VERSION
            or marker.get("completion_marker") is not True
            or marker.get("status") != "preflight_pass"):
        raise ValueError("preflight acceptance is not a completed P0-B v2 pass")
    for key, expected in (
            ("full_audit_complete", False),
            ("read_only_model_inference_executed", True),
            ("numeric_predictions_persisted", False),
            ("npz_written", False),
            ("locked_test_query_rows_written", 0),
            ("satellite_token_partition", "train"),
            ("profile_cap_status", "not_applicable_in_v14")):
        if marker.get(key) != expected:
            raise ValueError(f"preflight acceptance field differs: {key}")

    artifacts, parsed_artifacts = _verified_marker_artifacts(marker, root)
    required_artifacts = {
        "audit_contract.json", "manifest.json", "failure_ledger.json"}
    if set(artifacts) != required_artifacts:
        raise ValueError("preflight acceptance artifact inventory is not exact")
    runtime = parsed_artifacts["audit_contract.json"]
    manifest = parsed_artifacts["manifest.json"]
    failure = parsed_artifacts["failure_ledger.json"]
    if (runtime.get("audit_schema_version") != P0B_AUDIT_SCHEMA_VERSION
            or runtime.get("status") != "runtime_contract_bound"):
        raise ValueError("preflight runtime contract is not bound")
    if (manifest.get("audit_schema_version") != P0B_AUDIT_SCHEMA_VERSION
            or manifest.get("status") != "preflight_pass"
            or manifest.get("preflight_only") is not True):
        raise ValueError("preflight manifest is not a completed preflight pass")
    if (failure.get("audit_schema_version") != P0B_AUDIT_SCHEMA_VERSION
            or failure.get("status") != "no_failures"
            or failure.get("entries") != []):
        raise ValueError("preflight failure ledger is not empty")
    probe_contract = frozen_contract["output_lifecycle"]["preflight_probe"]
    expected_probe_count = (
        len(probe_contract["stations"])
        * len(probe_contract["query_splits"])
        * len(probe_contract["altitude_bands_km"])
    )
    inference_preflight = manifest.get("inference_preflight")
    if (marker.get("stratified_probe_count") != expected_probe_count
            or not isinstance(inference_preflight, dict)
            or inference_preflight.get("status")
            != "stratified_full_chain_pass_no_numeric_persistence"
            or inference_preflight.get("probe_count") != expected_probe_count
            or inference_preflight.get("activation", {}).get("status") != "pass"
            or inference_preflight.get("numeric_predictions_persisted") is not False
            or inference_preflight.get("npz_written") is not False):
        raise ValueError("preflight inference suite is not a complete strict pass")

    expected_queries = _contract_expected_query_counts(frozen_contract)
    expected_profiles = _contract_expected_profile_counts(frozen_contract)
    if marker.get("query_partition_counts") != expected_queries:
        raise ValueError("preflight query counts differ from the frozen contract")
    if marker.get("satellite_profile_counts") != expected_profiles:
        raise ValueError("preflight profile counts differ from the frozen contract")
    raw_registry = manifest.get("raw_registry", {})
    if (raw_registry.get("date_partition_counts") != expected_queries
            or raw_registry.get("allowed_query_count") != sum(
                expected_queries.values())):
        raise ValueError("preflight manifest query registry differs")
    profile_metadata = runtime.get("satellite_profile_metadata")
    if (not isinstance(profile_metadata, dict)
            or manifest.get("satellite_profile_metadata") != profile_metadata
            or _profile_count_ledger_from_metadata(profile_metadata)
            != expected_profiles):
        raise ValueError("preflight profile metadata differs")

    current_checkpoint_sha = sha256_file(checkpoint)
    if current_checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError("full-audit checkpoint SHA256 mismatch")
    if marker.get("candidate_checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("preflight marker checkpoint identity differs")
    runtime_checkpoint = runtime.get("candidate_checkpoint", {})
    runtime_identity_checkpoint = runtime.get("runtime_identities", {}).get(
        "checkpoint_path", {})
    for candidate in (runtime_checkpoint, runtime_identity_checkpoint):
        if (candidate.get("sha256") != expected_checkpoint_sha256
                or Path(candidate.get("path", "")).resolve() != checkpoint.resolve()):
            raise ValueError("preflight checkpoint identity differs from full audit")
    if (manifest.get("checkpoint_sha256_before") != expected_checkpoint_sha256
            or manifest.get("checkpoint_sha256_after")
            != expected_checkpoint_sha256):
        raise ValueError("preflight manifest checkpoint identity differs")

    frozen_binding = runtime.get("frozen_p0b_contract", {})
    if (frozen_binding.get("sha256") != frozen_contract_sha256
            or Path(frozen_binding.get("path", "")).resolve()
            != frozen_contract_path.resolve()
            or sha256_file(frozen_contract_path) != frozen_contract_sha256):
        raise ValueError("preflight P0-B contract identity differs")
    if marker.get("p0b_contract_sha256") != frozen_contract_sha256:
        raise ValueError("preflight marker P0-B contract identity differs")
    expected_p0a = frozen_contract["data_scope"]["P0A_ISR_contract_identity"]
    current_p0a_identity = artifact_identity(p0a_contract_path)
    p0a_binding = runtime.get("p0a_dependency", {})
    if (current_p0a_identity["sha256"] != expected_p0a.get("sha256")
            or current_p0a_identity["size_bytes"] != expected_p0a.get("size_bytes")
            or p0a_binding.get("contract_sha256") != current_p0a_identity["sha256"]
            or p0a_binding.get("contract_size_bytes")
            != current_p0a_identity["size_bytes"]
            or Path(p0a_binding.get("contract_path", "")).resolve()
            != p0a_contract_path.resolve()):
        raise ValueError("preflight P0-A contract identity differs")
    if marker.get("p0a_contract_sha256") != current_p0a_identity["sha256"]:
        raise ValueError("preflight marker P0-A contract identity differs")

    expected_split_sha = frozen_contract["identity"]["date_split_sha256"]
    date_split = runtime.get("date_split", {})
    date_split_path = Path(date_split.get("path", "")).resolve()
    if (date_split.get("sha256") != expected_split_sha
            or sha256_file(date_split_path) != expected_split_sha):
        raise ValueError("preflight date-split identity differs")
    runtime_split_identity = runtime.get("runtime_identities", {}).get(
        "date_split_path", {})
    if (runtime_split_identity.get("sha256") != expected_split_sha
            or Path(runtime_split_identity.get("path", "")).resolve()
            != date_split_path
            or manifest.get("date_split_sha256") != expected_split_sha):
        raise ValueError("preflight date-split ledger differs")
    if marker.get("date_split_sha256") != expected_split_sha:
        raise ValueError("preflight marker date-split identity differs")
    input_data_identity = runtime.get("runtime_identities", {}).get(
        "input_data_sha256")
    if (not isinstance(input_data_identity, dict) or not input_data_identity
            or manifest.get("input_data_sha256") != input_data_identity
            or marker.get("input_data_sha256") != input_data_identity
            or input_data_identity.get("date_split_manifest")
            != expected_split_sha):
        raise ValueError("preflight input-data identity differs")
    allowlist_identity = _train_allowlist_identity_from_metadata(profile_metadata)
    if (runtime.get("train_allowlist_identity") != allowlist_identity
            or runtime.get("runtime_identities", {}).get(
                "train_allowlist_identity") != allowlist_identity
            or manifest.get("train_allowlist_identity") != allowlist_identity
            or marker.get("train_allowlist_identity") != allowlist_identity):
        raise ValueError("preflight train allowlist identity differs")
    token_directory_identity = validate_train_token_directory_identity(
        runtime.get("train_token_directory_identity"))
    if (runtime.get("runtime_identities", {}).get(
                "train_token_directory_identity") != token_directory_identity
            or manifest.get("train_token_directory_identity")
            != token_directory_identity
            or marker.get("train_token_directory_identity")
            != token_directory_identity):
        raise ValueError("preflight train token directory identity differs")

    runtime_git = runtime.get("git")
    if (runtime_git != dict(git_provenance)
            or manifest.get("git") != runtime_git
            or manifest.get("git_after") != runtime_git
            or marker.get("git_head") != git_provenance.get("head")
            or marker.get("implementation_tag_object_sha")
            != git_provenance.get("implementation_tag_object_sha")):
        raise ValueError("preflight Git/tag identity differs from full audit")
    runtime_environment = runtime.get("python_environment")
    if (runtime_environment != dict(python_environment)
            or runtime.get("runtime_identities", {}).get("python_environment")
            != runtime_environment
            or manifest.get("python_environment") != runtime_environment):
        raise ValueError("preflight Python environment differs from full audit")
    if marker.get("python_environment") != runtime_environment:
        raise ValueError("preflight marker Python environment differs")
    coordinate_values = (
        runtime.get("coordinate_enrichment"),
        runtime.get("runtime_identities", {}).get("coordinate_enrichment"),
        manifest.get("coordinate_enrichment"),
        marker.get("coordinate_enrichment"),
    )
    if any(value != dict(coordinate_status) for value in coordinate_values):
        raise ValueError("preflight coordinate identity differs from full audit")
    if (runtime.get("profile_cap_status") != "not_applicable_in_v14"
            or runtime.get("runtime_identities", {}).get("profile_cap_status")
            != "not_applicable_in_v14"
            or manifest.get("profile_cap_status") != "not_applicable_in_v14"):
        raise ValueError("preflight profile-cap status differs")
    column_access = runtime.get("ISR_column_access_audit")
    if (not isinstance(column_access, dict)
            or column_access.get("status") != "pass"
            or column_access.get("excluded_density_columns_materialized") != 0
            or runtime.get("runtime_identities", {}).get(
                "ISR_column_access_audit") != column_access
            or manifest.get("ISR_column_access_audit") != column_access
            or marker.get("ISR_column_access_audit") != column_access):
        raise ValueError("preflight ISR column-access attestation differs")
    isr_source_files = runtime.get("runtime_identities", {}).get(
        "ISR_source_file_sha256")
    if (not isinstance(isr_source_files, list) or not isr_source_files
            or runtime.get("isr_source_files", isr_source_files)
            != isr_source_files
            or manifest.get("isr_source_files") != isr_source_files):
        raise ValueError("preflight materialized ISR source identities differ")

    current_sources = _source_provenance(
        frozen_contract_path, p0a_contract_path)
    if runtime.get("source_provenance") != current_sources:
        raise ValueError("preflight source provenance differs from full audit")
    artifacts_after, parsed_after = _verified_marker_artifacts(marker, root)
    if (artifacts_after != artifacts or parsed_after != parsed_artifacts
            or path.read_bytes() != marker_bytes):
        raise ValueError("preflight acceptance changed while it was validated")
    return {
        "path": str(path),
        "sha256": marker_identity["sha256"],
        "size_bytes": marker_identity["size_bytes"],
        "directory": str(root),
        "git_head": git_provenance["head"],
        "implementation_tag_object_sha": git_provenance[
            "implementation_tag_object_sha"],
        "checkpoint_sha256": expected_checkpoint_sha256,
        "date_split_sha256": expected_split_sha,
        "p0a_contract_sha256": current_p0a_identity["sha256"],
        "p0b_contract_sha256": frozen_contract_sha256,
        "query_partition_counts": expected_queries,
        "satellite_profile_counts": expected_profiles,
        "python_environment": dict(python_environment),
        "coordinate_enrichment": dict(coordinate_status),
        "profile_cap_status": "not_applicable_in_v14",
        "input_data_sha256": dict(input_data_identity),
        "train_allowlist_identity": allowlist_identity,
        "train_token_directory_identity": token_directory_identity,
        "isr_source_files": isr_source_files,
        "ISR_column_access_audit": dict(column_access),
    }


def _resolve_p0a_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    directory = Path(args.p0a_dir).resolve()
    contract = Path(args.p0a_contract).resolve() if args.p0a_contract else (
        directory / "isr_evaluation_contract.json")
    return directory, contract


def _load_p0a_identity(
        contract_path: Path,
        expected_checkpoint_sha256: str,
        expected_date_split_sha256: str,
        expected_artifact_identity: Mapping[str, Any]) -> P0AIdentity:
    """Read only the P0-A qav2 JSON identity; never open a P0-A cache."""
    expected_sha256 = expected_artifact_identity.get("sha256")
    expected_size = expected_artifact_identity.get("size_bytes")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(value not in "0123456789abcdef" for value in expected_sha256)):
        raise ValueError("frozen P0-A JSON identity has an invalid SHA256")
    if (isinstance(expected_size, bool) or not isinstance(expected_size, int)
            or expected_size <= 0):
        raise ValueError("frozen P0-A JSON identity has an invalid size")

    def validate_artifact(stage: str) -> dict[str, Any]:
        identity = artifact_identity(contract_path)
        if (identity["sha256"] != expected_sha256
                or int(identity["size_bytes"]) != expected_size):
            raise ValueError(
                f"P0-A JSON identity mismatch {stage}: "
                f"actual_sha256={identity['sha256']}, "
                f"actual_size_bytes={identity['size_bytes']}, "
                f"expected_sha256={expected_sha256}, "
                f"expected_size_bytes={expected_size}")
        return identity

    identity_before = validate_artifact("before parse")
    contract = _read_json(contract_path)
    identity_after = validate_artifact("after parse")
    if identity_after != identity_before:
        raise RuntimeError("P0-A JSON identity changed while it was parsed")
    if int(contract.get("evaluation_schema_version", -1)) != 2:
        raise ValueError("P0-A evaluation schema must be qav2")
    if tuple(contract.get("token_partitions", ())) != ("train", "development"):
        raise ValueError(
            "P0-A identity requires the frozen train+development token policy")
    candidate = contract.get("candidate_checkpoint", {})
    expected_candidate = {
        "sha256": expected_checkpoint_sha256,
        "checkpoint_format_version": EXPECTED_CHECKPOINT_FORMAT,
        "model_domain_semantics": EXPECTED_DOMAIN,
        "model_alt_range_km": list(EXPECTED_MODEL_RANGE),
        "observation_alt_range_km": list(EXPECTED_OBSERVATION_RANGE),
        "peak_search_alt_range_km": list(EXPECTED_OBSERVATION_RANGE),
    }
    mismatches = {
        key: (candidate.get(key), value)
        for key, value in expected_candidate.items()
        if candidate.get(key) != value
    }
    date_split = candidate.get("date_split", {})
    if date_split.get("sha256") != expected_date_split_sha256:
        mismatches["date_split.sha256"] = (
            date_split.get("sha256"), expected_date_split_sha256)
    if mismatches:
        raise ValueError(f"P0-A candidate contract mismatch: {mismatches}")
    if not isinstance(contract.get("isr_input_files"), list):
        raise ValueError("P0-A qav2 contract lacks ISR source-file identities")
    return P0AIdentity(
        contract_path=contract_path,
        contract=contract,
        sha256=expected_sha256,
        size_bytes=expected_size,
    )


def _validate_identity_contract(
        frozen_contract: Mapping[str, Any], checkpoint: Path,
        expected_sha256: str) -> None:
    identity = frozen_contract["identity"]
    actual_sha = sha256_file(checkpoint)
    if expected_sha256 != identity["candidate_checkpoint_sha256"]:
        raise ValueError("CLI expected checkpoint SHA256 differs from frozen P0-B contract")
    if actual_sha != expected_sha256:
        raise ValueError("candidate checkpoint SHA256 mismatch")
    if int(identity["checkpoint_format_version"]) != EXPECTED_CHECKPOINT_FORMAT:
        raise ValueError("frozen P0-B checkpoint format drifted")
    if identity["model_domain_semantics"] != EXPECTED_DOMAIN:
        raise ValueError("frozen P0-B model domain drifted")
    if tuple(map(float, identity["model_altitude_range_km"])) != EXPECTED_MODEL_RANGE:
        raise ValueError("frozen P0-B model altitude range drifted")
    if tuple(map(float, identity["observation_altitude_range_km"])) != (
            EXPECTED_OBSERVATION_RANGE):
        raise ValueError("frozen P0-B observation altitude range drifted")


def _validate_version_control_contract(
        frozen_contract: Mapping[str, Any],
        implementation_ref: str) -> None:
    version_control = frozen_contract.get("version_control", {})
    if version_control.get("required_branch") != EXPECTED_IMPLEMENTATION_BRANCH:
        raise ValueError("frozen P0-B implementation branch drifted")
    if version_control.get("required_implementation_tag") != (
            EXPECTED_IMPLEMENTATION_TAG):
        raise ValueError("frozen P0-B implementation tag drifted")
    if version_control.get("required_implementation_tag_type") != "annotated":
        raise ValueError("frozen P0-B implementation tag type drifted")
    if implementation_ref != EXPECTED_IMPLEMENTATION_TAG:
        raise ValueError("CLI implementation ref differs from the frozen tag")
    if version_control.get("require_head_equals_tag") is not True:
        raise ValueError("frozen P0-B contract must require HEAD == tag commit")
    if version_control.get("untracked_generated_artifacts_allowed") is not True:
        raise ValueError("frozen P0-B generated-artifact policy drifted")
    if version_control.get("explicit_path_staging_only") is not True:
        raise ValueError("frozen P0-B contract must require explicit-path staging")
    if version_control.get("git_add_all_forbidden") is not True:
        raise ValueError("frozen P0-B contract must forbid git add-all staging")
    if version_control.get("accepted_history_rewrite_forbidden") is not True:
        raise ValueError("frozen P0-B contract must forbid history rewrites")
    if tuple(version_control.get("critical_tracked_paths", ())) != (
            CRITICAL_TRACKED_PATHS):
        raise ValueError("frozen P0-B critical tracked paths drifted")


def _load_date_split(config: Mapping[str, Any], expected_sha256: str) -> tuple[Path, dict[str, list[int]]]:
    path = Path(config["date_split_manifest"]).resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("date-split SHA256 differs from frozen P0-B contract")
    payload = _read_json(path)
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
            "train", "development", "locked_test"}:
        raise ValueError("date-split partitions are incomplete")
    normalized = {
        key: sorted({int(value) for value in values})
        for key, values in partitions.items()
    }
    flattened = [value for values in normalized.values() for value in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("date-split partitions overlap")
    return path, normalized


def _validate_input_data_identity(
        config: Mapping[str, Any], run_manifest: Mapping[str, Any],
        training_summary: Mapping[str, Any],
        expected_date_split_sha256: str) -> dict[str, str]:
    identity = config.get("input_data_sha256")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("strict v14 config lacks input_data_sha256")
    normalized = {str(key): str(value) for key, value in identity.items()}
    if normalized.get("date_split_manifest") != expected_date_split_sha256:
        raise ValueError("input_data_sha256 date split differs from P0-B contract")
    for label, candidate in (
            ("run_manifest.config", run_manifest.get("config", {}).get(
                "input_data_sha256")),
            ("run_manifest.resolved_training", run_manifest.get(
                "resolved_training", {}).get("input_data_sha256")),
            ("training_summary", training_summary.get("input_data_sha256"))):
        if candidate != normalized:
            raise ValueError(f"{label} input_data_sha256 differs from strict config")
    for key, digest in normalized.items():
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest.lower()):
            raise ValueError(f"invalid input SHA256 for {key}")
    return normalized


def _build_train_only_allowlists(
        config: Mapping[str, Any],
        partitions: Mapping[str, list[int]],
        training_summary: Mapping[str, Any],
        start_date: dt.datetime,
        contract_expected_counts: Mapping[str, Mapping[str, int]],
        ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build train whitelists from profile-index metadata, never density rows.

    ``representative_time`` is the frozen profile-level relative hour written by
    QC.  Its floored UTC-relative day is the same split primitive used by the
    training dataset.  Development arrays are used only for aggregate metadata;
    their density files are never opened here.
    """
    allowlists: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    expected_counts = training_summary.get("allowed_profile_summary", {})
    if expected_counts != contract_expected_counts:
        raise ValueError(
            "training summary profile counts differ from the frozen P0-B identity")
    for source in SOURCES:
        index_key = (
            "fy_profile_index_path" if source == "FY"
            else "cosmic_profile_index_path")
        index_path = Path(config[index_key]).resolve()
        with np.load(index_path, allow_pickle=False) as index:
            required = {
                "profile_id", "pass_profile", "representative_time",
                "date_code", "kept_points", "output_start", "output_end",
            }
            missing = required.difference(index.files)
            if missing:
                raise ValueError(
                    f"{source} profile index lacks metadata: {sorted(missing)}")
            profile_ids = np.asarray(index["profile_id"], dtype=np.int64)
            passed = np.asarray(index["pass_profile"], dtype=bool)
            representative_time = np.asarray(
                index["representative_time"], dtype=np.float64)
            date_code = np.asarray(index["date_code"], dtype=np.int64)
            kept_points = np.asarray(index["kept_points"], dtype=np.int64)
            starts = np.asarray(index["output_start"], dtype=np.int64)
            ends = np.asarray(index["output_end"], dtype=np.int64)
        lengths = {
            len(profile_ids), len(passed), len(representative_time),
            len(date_code), len(kept_points), len(starts), len(ends),
        }
        if len(lengths) != 1:
            raise ValueError(f"{source} profile-index metadata is misaligned")
        if len(profile_ids) == 0 or len(np.unique(profile_ids)) != len(profile_ids):
            raise ValueError(f"{source} profile-index IDs must be nonempty and unique")
        if np.any(profile_ids < 0):
            raise ValueError(f"{source} profile-index IDs must be nonnegative")
        if not np.isfinite(representative_time[passed]).all():
            raise ValueError(f"{source} passing profiles lack representative_time")
        if np.any(starts[passed] < 0) or np.any(ends[passed] <= starts[passed]):
            raise ValueError(f"{source} passing profile output boundaries are invalid")
        if np.any(starts[~passed] != -1) or np.any(ends[~passed] != -1):
            raise ValueError(
                f"{source} failed profiles must use -1/-1 output boundaries")
        if np.any(kept_points[passed] != (ends[passed] - starts[passed])):
            raise ValueError(
                f"{source} passing kept_points disagree with output boundaries")

        data_key = "fy_path" if source == "FY" else "cosmic_path"
        physical_path = Path(config[data_key]).resolve()
        physical = np.load(physical_path, mmap_mode="r")
        try:
            if physical.ndim != 2 or physical.shape[1] < 5:
                raise ValueError(
                    f"{source} physical NPY header has an invalid shape")
            physical_row_count = int(physical.shape[0])
        finally:
            del physical
        ordered = np.argsort(starts[passed], kind="stable")
        ordered_starts = starts[passed][ordered]
        ordered_ends = ends[passed][ordered]
        if (len(ordered_starts) == 0 or ordered_starts[0] != 0
                or ordered_ends[-1] != physical_row_count
                or np.any(ordered_starts[1:] != ordered_ends[:-1])):
            raise ValueError(
                f"{source} passing output boundaries do not continuously cover "
                "the physical NPY row count")

        representative_day = np.full(len(profile_ids), -1, dtype=np.int64)
        representative_day[passed] = np.floor(
            representative_time[passed] / 24.0).astype(np.int64)
        expected_date_code = np.asarray([
            int((start_date + dt.timedelta(days=int(day))).strftime("%Y%m%d"))
            for day in representative_day[passed]
        ], dtype=np.int64)
        if not np.array_equal(date_code[passed], expected_date_code):
            raise ValueError(
                f"{source} passing date_code disagrees with representative_time")
        train_mask = passed & np.isin(
            representative_day, np.asarray(partitions["train"], dtype=np.int64))
        development_mask = passed & np.isin(
            representative_day,
            np.asarray(partitions["development"], dtype=np.int64))
        train_ids = np.unique(profile_ids[train_mask])
        development_ids = np.unique(profile_ids[development_mask])

        if len(np.intersect1d(train_ids, development_ids)):
            raise ValueError(f"{source} train/development profile IDs overlap")
        if len(train_ids) == 0:
            raise ValueError(f"{source} train-only allowlist is empty")
        if len(train_ids) > 1 and np.any(train_ids[1:] <= train_ids[:-1]):
            raise ValueError(f"{source} train-only allowlist is not sorted unique")
        expected_train = expected_counts.get("train", {}).get(source)
        expected_development = expected_counts.get("development", {}).get(source)
        if expected_train is None or int(expected_train) != len(train_ids):
            raise ValueError(
                f"{source} train profile-index count differs from training summary")
        if (expected_development is None
                or int(expected_development) != len(development_ids)):
            raise ValueError(
                f"{source} development profile-index count differs from training summary")
        if len(train_ids) != int(contract_expected_counts["train"][source]):
            raise ValueError(f"{source} frozen train profile count changed")
        if len(development_ids) != int(
                contract_expected_counts["development"][source]):
            raise ValueError(f"{source} frozen development profile count changed")
        allowlists[source] = train_ids
        source_metadata = {
            "profile_index_path": str(index_path),
            "profile_index_sha256": sha256_file(index_path),
            "access_semantics": "profile_index_metadata_only_no_density_values_v1",
            "token_partition": "train",
            "train_unique_profiles": int(len(train_ids)),
            "train_profile_id_sha256": _profile_id_sha256(train_ids),
            "train_profile_index_kept_points": int(kept_points[train_mask].sum()),
            "train_date_codes": sorted(
                int(value) for value in np.unique(date_code[train_mask])
                if int(value) > 0),
            "development_metadata_only": True,
            "development_unique_profiles": int(len(development_ids)),
            "development_profile_id_sha256": _profile_id_sha256(development_ids),
            "development_profile_index_kept_points": int(
                kept_points[development_mask].sum()),
            "development_date_codes": sorted(
                int(value) for value in np.unique(date_code[development_mask])
                if int(value) > 0),
            "configured_observation_altitude_range_km": list(
                map(float, config["observation_alt_range"])),
            "development_density_values_read": False,
            "development_model_inference": False,
            "locked_test_rows_inferred": False,
            "classification_only_transient": True,
            "locked_test_metadata_persisted": False,
            "locked_test_metadata_aggregated": False,
            "locked_test_density_values_read": False,
        }
        metadata[source] = source_metadata
    return allowlists, metadata


def _date_partition(
        date_utc: str, start_date: dt.datetime,
        partitions: Mapping[str, list[int]]) -> tuple[str, int]:
    day = dt.datetime.strptime(date_utc, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    day_index = int((day.date() - start_date.date()).days)
    matches = [name for name, values in partitions.items() if day_index in values]
    if len(matches) != 1:
        raise ValueError(
            f"ISR date {date_utc} (day {day_index}) has {len(matches)} partitions")
    return matches[0], day_index


def _included_partition_dates_utc(
        start_date: dt.datetime,
        partitions: Mapping[str, list[int]]) -> set[str]:
    """Return the explicit train/development UTC-date allowlist."""
    allowed: set[str] = set()
    for partition in ("train", "development"):
        for day_index in partitions[partition]:
            allowed.add(
                (start_date + dt.timedelta(days=int(day_index))).strftime("%Y%m%d"))
    if not allowed:
        raise ValueError("train/development ISR date allowlist is empty")
    return allowed


def _select_allowed_isr_files(
        p0a_contract: Mapping[str, Any], start_date: dt.datetime,
        partitions: Mapping[str, list[int]], jicamarca_dir: Path,
        poker_flat_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Enumerate contract-declared ISR candidates without filename-date filtering.

    A file may span midnight or multiple UTC dates.  The raw loader must inspect
    only its timestamp metadata first and slice allowed columns before reading
    density.  Consequently the filename date is never a partition decision.
    """
    del start_date, partitions
    selected: dict[str, list[dict[str, Any]]] = {
        "Jicamarca": [], "PokerFlat": []}
    roots = {
        "Jicamarca": jicamarca_dir.resolve(),
        "PokerFlat": poker_flat_dir.resolve(),
    }
    seen_paths: set[Path] = set()
    for row in p0a_contract.get("isr_input_files", []):
        unresolved = Path(row["path"])
        matches = re.findall(
            r"(jro|pfa)(\d{8})", unresolved.name, flags=re.IGNORECASE)
        if len(matches) != 1:
            raise ValueError(
                "ISR contract filename must declare exactly one UTC date: "
                f"{unresolved.name}")
        prefix, _filename_date = matches[0]

        station = "Jicamarca" if prefix.lower() == "jro" else "PokerFlat"
        expected_suffix = ".hdf5" if station == "Jicamarca" else ".h5"
        if unresolved.suffix.lower() != expected_suffix:
            raise ValueError(
                f"{station} contract file has the wrong suffix: {unresolved.name}")
        path = unresolved.resolve()
        if path in seen_paths:
            raise ValueError(f"duplicate ISR path in P0-A contract: {path}")
        seen_paths.add(path)
        try:
            path.relative_to(roots[station])
        except ValueError as exc:
            raise ValueError(
                f"P0-A {station} path is outside its declared CLI directory") from exc
        normalized = {
            "path": str(path),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        if (len(normalized["sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in normalized["sha256"].lower())):
            raise ValueError(f"invalid ISR SHA256 in P0-A contract: {path}")
        if normalized["size_bytes"] <= 0:
            raise ValueError(f"invalid ISR size in P0-A contract: {path}")
        normalized["sha256"] = normalized["sha256"].lower()
        selected[station].append(normalized)
    for station in selected:
        selected[station].sort(key=lambda value: value["path"])
        if not selected[station]:
            raise ValueError(
                f"P0-A contract declares no {station} ISR candidate files")
    return selected


def _load_raw_isr_records(
        selected_files: Mapping[str, list[dict[str, Any]]],
        start_unix: float, end_unix: float,
        allowed_dates_utc: set[str]) -> tuple[
            dict[str, list[dict[str, Any]]], list[dict[str, Any]],
            dict[str, Any]]:
    """Load explicit HDFs with timestamp-first, allowed-column-only access."""
    from isr_evaluation.isr_loader import load_jicamarca, load_poker_flat

    canonical_by_identity = {
        (str(Path(row["path"]).resolve()), row["sha256"],
         int(row["size_bytes"])): row
        for rows in selected_files.values() for row in rows
    }
    if not allowed_dates_utc or any(
            re.fullmatch(r"\d{8}", value) is None
            for value in allowed_dates_utc):
        raise ValueError("allowed ISR UTC-date whitelist is empty or malformed")
    jicamarca_records, jicamarca_access = load_jicamarca(
            "", start_unix, end_unix,
            alt_min=120.0, alt_max=500.0, err_ratio_max=0.5,
            allowed_dates_utc=allowed_dates_utc,
            file_paths=[row["path"] for row in selected_files["Jicamarca"]],
            source_file_identities=selected_files["Jicamarca"],
            fail_on_file_error=True, return_access_audit=True)
    poker_records, poker_access = load_poker_flat(
            "", start_unix, end_unix,
            alt_min=120.0, alt_max=500.0, err_ratio_max=0.5,
            allowed_dates_utc=allowed_dates_utc,
            file_paths=[row["path"] for row in selected_files["PokerFlat"]],
            source_file_identities=selected_files["PokerFlat"],
            fail_on_file_error=True, return_access_audit=True)
    records = {
        "Jicamarca": jicamarca_records,
        "PokerFlat": poker_records,
    }
    access_audits = {
        "Jicamarca": jicamarca_access,
        "PokerFlat": poker_access,
    }
    for station_records in records.values():
        for record in station_records:
            canonical = []
            for identity in record.get("source_file_identity", []):
                key = (
                    str(Path(identity["path"]).resolve()),
                    str(identity["sha256"]),
                    int(identity["size_bytes"]),
                )
                if key not in canonical_by_identity:
                    raise ValueError(
                        "raw loader opened an ISR file outside the allowed set")
                canonical.append(dict(canonical_by_identity[key]))
            record["source_file_identity"] = canonical
    identities: dict[str, dict[str, Any]] = {}
    for audit in access_audits.values():
        for file_row in audit.get("files", []):
            identity = file_row.get("materialized_source_identity")
            if identity is None:
                continue
            path = str(Path(identity["path"]).resolve())
            normalized = {
                "path": path,
                "sha256": str(identity["sha256"]),
                "size_bytes": int(identity["size_bytes"]),
            }
            if path in identities and identities[path] != normalized:
                raise ValueError("ISR access audit contains conflicting identities")
            identities[path] = normalized
    for station_records in records.values():
        for record in station_records:
            for identity in record.get("source_file_identity", []):
                path = str(Path(identity["path"]).resolve())
                normalized = {
                    "path": path,
                    "sha256": identity["sha256"],
                    "size_bytes": int(identity["size_bytes"]),
                }
                if path in identities and identities[path] != normalized:
                    raise ValueError("ISR source identity changed between records")
                identities[path] = normalized
    return (
        records,
        sorted(identities.values(), key=lambda row: row["path"]),
        access_audits,
    )


def _validate_isr_column_access_audits(
        access_audits: Mapping[str, Any],
        selected_files: Mapping[str, list[dict[str, Any]]],
        allowed_dates_utc: set[str]) -> dict[str, Any]:
    """Validate transient column ledgers and return a non-locked attestation."""
    if set(access_audits) != {"Jicamarca", "PokerFlat"}:
        raise ValueError("ISR column-access audits do not cover both stations")
    expected_allowed = sorted(allowed_dates_utc)
    field_contract = {
        "Jicamarca": {
            "density": ("ne", "dne"),
            "coordinate": (),
        },
        "PokerFlat": {
            "density": ("ne", "dne"),
            "coordinate": ("cgm_lat", "cgm_lon"),
        },
    }
    station_attestations: dict[str, Any] = {}
    global_allowed_content_identities: list[dict[str, Any]] = []
    global_identity_paths: set[str] = set()
    total_density_dataset_column_reads = 0
    total_coordinate_dataset_column_reads = 0
    for station, audit in access_audits.items():
        if (not isinstance(audit, dict)
                or audit.get("isr_column_access_audit_schema_version") != 1
                or audit.get("station") != station
                or audit.get("allowed_dates_utc") != expected_allowed
                or audit.get("excluded_date_values_persisted") is not False):
            raise ValueError(f"invalid ISR column-access audit for {station}")
        declared_paths = {
            str(Path(row["path"]).resolve()): dict(row)
            for row in selected_files[station]
        }
        files = audit.get("files")
        if not isinstance(files, list):
            raise ValueError(f"ISR column-access file ledger is invalid for {station}")
        audited_paths = [str(Path(row.get("path", "")).resolve())
                         for row in files if isinstance(row, dict)]
        if (len(audited_paths) != len(files)
                or len(audited_paths) != len(set(audited_paths))
                or set(audited_paths) != set(declared_paths)):
            raise ValueError(f"ISR candidate-file access ledger differs for {station}")
        allowed_time_columns = 0
        density_dataset_column_reads = 0
        coordinate_dataset_column_reads = 0
        materialized_file_count = 0
        allowed_payload_records: list[dict[str, Any]] = []
        station_allowed_content_identities: list[dict[str, Any]] = []
        station_identity_paths: set[str] = set()
        density_fields = field_contract[station]["density"]
        coordinate_fields = field_contract[station]["coordinate"]
        expected_dataset_fields = (*density_fields, *coordinate_fields)
        for file_row in files:
            path = str(Path(file_row["path"]).resolve())
            segments = file_row.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"ISR file lacks timestamp-segment audit: {path}")
            file_has_allowed = False
            for segment in segments:
                total = segment.get("total_time_columns")
                allowed = segment.get("allowed_time_columns")
                excluded = segment.get("excluded_time_columns")
                if (any(isinstance(value, bool) or not isinstance(value, int)
                        or value < 0 for value in (total, allowed, excluded))
                        or allowed + excluded != total):
                    raise ValueError("ISR timestamp-column accounting is invalid")
                dates = segment.get("allowed_dates_utc")
                if (not isinstance(dates, list)
                        or not set(dates).issubset(allowed_dates_utc)):
                    raise ValueError("ISR materialized date is outside the allowlist")
                reads = segment.get("dataset_reads")
                if not isinstance(reads, list):
                    raise ValueError("ISR dataset-read ledger is invalid")
                if allowed == 0 and reads:
                    raise ValueError("ISR density was read with no allowed columns")
                observed_dataset_fields: list[str] = []
                for read in reads:
                    payload_sha = read.get("materialized_payload_sha256")
                    if (read.get("materialized_column_count") != allowed
                            or read.get("excluded_columns_materialized") != 0
                            or not set(read.get("materialized_dates_utc", ())).issubset(
                                allowed_dates_utc)
                            or not isinstance(payload_sha, str)
                            or len(payload_sha) != 64
                            or any(character not in "0123456789abcdef"
                                   for character in payload_sha)):
                        raise ValueError("ISR density materialized an excluded column")
                    dataset = str(read.get("dataset", ""))
                    observed_dataset_fields.append(dataset)
                    if dataset in density_fields:
                        density_dataset_column_reads += int(
                            read["materialized_column_count"])
                    elif dataset in coordinate_fields:
                        coordinate_dataset_column_reads += int(
                            read["materialized_column_count"])
                    else:
                        raise ValueError(
                            f"unexpected ISR dataset-read field for {station}: "
                            f"{dataset}")
                    allowed_payload_records.append({
                        "path": path,
                        "segment_id": str(segment.get("segment_id", "")),
                        "dataset": dataset,
                        "column_spans_inclusive": read.get(
                            "materialized_column_spans_inclusive"),
                        "payload_sha256": payload_sha,
                    })
                if allowed > 0 and tuple(observed_dataset_fields) != (
                        expected_dataset_fields):
                    raise ValueError(
                        f"ISR dataset-read fields differ for {station}: "
                        f"actual={observed_dataset_fields}, "
                        f"expected={list(expected_dataset_fields)}")
                allowed_time_columns += int(allowed)
                file_has_allowed = file_has_allowed or allowed > 0
            identity = file_row.get("materialized_source_identity")
            allowed_content_identity = file_row.get(
                "materialized_allowed_content_identity")
            if file_has_allowed:
                if not isinstance(identity, dict):
                    raise ValueError("materialized ISR file lacks a verified identity")
                if file_row.get("source_identity_semantics") != (
                        "p0a_contract_attested_no_whole_hdf_reread_v1"):
                    raise ValueError(
                        "P0-B ISR source identity reread semantics are invalid")
                actual = {
                    "path": str(Path(identity.get("path", "")).resolve()),
                    "sha256": str(identity.get("sha256", "")),
                    "size_bytes": int(identity.get("size_bytes", -1)),
                }
                if actual != declared_paths[path]:
                    raise ValueError("materialized ISR identity differs from P0-A")
                if (not isinstance(allowed_content_identity, dict)
                        or set(allowed_content_identity) != {
                            "schema", "path", "sha256", "framed_array_count",
                            "allowed_time_column_count"}):
                    raise ValueError(
                        "materialized ISR file lacks an exact allowed-content identity")
                content_path_value = allowed_content_identity.get("path")
                if (not isinstance(content_path_value, str)
                        or not Path(content_path_value).is_absolute()):
                    raise ValueError("ISR allowed-content identity path is invalid")
                content_path = str(Path(content_path_value).resolve())
                content_sha256 = allowed_content_identity.get("sha256")
                framed_array_count = allowed_content_identity.get(
                    "framed_array_count")
                allowed_column_count = allowed_content_identity.get(
                    "allowed_time_column_count")
                file_allowed_column_count = sum(
                    int(segment["allowed_time_columns"]) for segment in segments)
                if (allowed_content_identity.get("schema")
                        != "isr_allowed_materialized_content_v1"
                        or content_path != path
                        or not isinstance(content_sha256, str)
                        or len(content_sha256) != 64
                        or any(character not in "0123456789abcdef"
                               for character in content_sha256)
                        or isinstance(framed_array_count, bool)
                        or not isinstance(framed_array_count, int)
                        or framed_array_count <= 0
                        or isinstance(allowed_column_count, bool)
                        or not isinstance(allowed_column_count, int)
                        or allowed_column_count <= 0
                        or allowed_column_count != file_allowed_column_count):
                    raise ValueError("ISR allowed-content identity is invalid")
                normalized_content_identity = {
                    "schema": "isr_allowed_materialized_content_v1",
                    "path": content_path,
                    "sha256": content_sha256,
                    "framed_array_count": framed_array_count,
                    "allowed_time_column_count": allowed_column_count,
                }
                normalized_path_key = os.path.normcase(content_path)
                if (normalized_path_key in station_identity_paths
                        or normalized_path_key in global_identity_paths):
                    raise ValueError("duplicate ISR allowed-content identity path")
                station_identity_paths.add(normalized_path_key)
                global_identity_paths.add(normalized_path_key)
                station_allowed_content_identities.append(
                    normalized_content_identity)
                global_allowed_content_identities.append(
                    normalized_content_identity)
                materialized_file_count += 1
            elif identity is not None or allowed_content_identity is not None:
                raise ValueError(
                    "metadata-only ISR file was unexpectedly content-identified")
        totals = audit.get("totals")
        if (not isinstance(totals, dict)
                or totals.get("allowed_time_columns") != allowed_time_columns
                or totals.get("density_dataset_column_reads")
                != density_dataset_column_reads
                or totals.get("excluded_density_columns_materialized") != 0):
            raise ValueError(f"ISR aggregate access ledger differs for {station}")
        station_allowed_content_identities.sort(
            key=lambda row: os.path.normcase(row["path"]))
        station_attestations[station] = {
            "materialized_allowed_source_file_count": materialized_file_count,
            "allowed_time_columns": allowed_time_columns,
            "density_dataset_fields": list(density_fields),
            "coordinate_dataset_fields": list(coordinate_fields),
            "density_dataset_column_reads": density_dataset_column_reads,
            "coordinate_dataset_column_reads": (
                coordinate_dataset_column_reads),
            "materialized_allowed_content_identities": (
                station_allowed_content_identities),
            "excluded_density_columns_materialized": 0,
            "allowed_payload_sha256": hashlib.sha256(json.dumps(
                sorted(
                    allowed_payload_records,
                    key=lambda row: (
                        row["path"], row["segment_id"], row["dataset"],
                        json.dumps(row["column_spans_inclusive"],
                                   separators=(",", ":")))),
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                allow_nan=False).encode("utf-8")).hexdigest(),
        }
        total_density_dataset_column_reads += density_dataset_column_reads
        total_coordinate_dataset_column_reads += coordinate_dataset_column_reads
    global_allowed_content_identities.sort(
        key=lambda row: os.path.normcase(row["path"]))
    return {
        "isr_column_access_audit_schema_version": 1,
        "status": "pass",
        "filter_semantics": (
            "timestamps_metadata_first_then_explicit_2d_column_slice_v1"),
        "allowed_dates_utc": expected_allowed,
        "stations": station_attestations,
        "density_dataset_column_reads": total_density_dataset_column_reads,
        "coordinate_dataset_column_reads": total_coordinate_dataset_column_reads,
        "materialized_allowed_content_identities": (
            global_allowed_content_identities),
        "locked_or_out_of_scope_timestamp_metadata_persisted": False,
        "locked_or_out_of_scope_timestamp_metadata_aggregated": False,
        "excluded_density_columns_materialized": 0,
    }


def _validate_isr_source_identities(
        actual: list[dict[str, Any]],
        selected_files: Mapping[str, list[dict[str, Any]]]) -> None:
    declared = {
        row["path"]: dict(row)
        for rows in selected_files.values() for row in rows
    }
    if not actual:
        raise ValueError("date-filtered raw ISR loader used no source files")
    actual_paths = [row["path"] for row in actual]
    if len(actual_paths) != len(set(actual_paths)):
        raise ValueError("date-filtered raw ISR identities contain duplicates")
    for row in actual:
        if row["path"] not in declared or row != declared[row["path"]]:
            raise ValueError(
                "date-filtered raw ISR identity differs from the P0-A contract")


def _record_coordinate_grids(record: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    altitudes = np.asarray(record["alt_1d"], dtype=np.float64)
    timestamps = np.asarray(record["ts_1d"], dtype=np.float64)
    shape = np.asarray(record["ne_2d"]).shape
    altitude_grid = np.broadcast_to(altitudes[:, None], shape)
    timestamp_grid = np.broadcast_to(timestamps[None, :], shape)
    geo_lat = record.get("geo_lat_2d")
    geo_lon = record.get("geo_lon_2d")
    if geo_lat is None or geo_lon is None:
        latitude_grid = np.full(shape, float(record["lat"]), dtype=np.float64)
        longitude_grid = np.full(shape, float(record["lon"]), dtype=np.float64)
    else:
        latitude_grid = np.asarray(geo_lat, dtype=np.float64)
        longitude_grid = np.asarray(geo_lon, dtype=np.float64)
    return altitude_grid, timestamp_grid, latitude_grid, longitude_grid


def _rebuild_raw_selections(
        records: Mapping[str, list[dict[str, Any]]], start_date: dt.datetime,
        partitions: Mapping[str, list[int]],
        contract_expected_counts: Mapping[str, int],
        ) -> tuple[list[RawSelection], dict[str, Any]]:
    selections: list[RawSelection] = []
    seen: set[str] = set()
    partition_counts = {key: 0 for key in ("train", "development")}
    station_counts = {station: 0 for station in records}
    next_query_id = 0

    for station in ("Jicamarca", "PokerFlat"):
        for record in records[station]:
            date_utc = str(record["date_str"])
            query_split, _ = _date_partition(date_utc, start_date, partitions)
            if query_split == "locked_test":
                raise RuntimeError(
                    "an opened allowed-date ISR file emitted a locked-test record")
            altitude, timestamp, latitude, longitude = _record_coordinate_grids(record)
            density = np.asarray(record["ne_2d"], dtype=np.float64)
            coordinate_mask = np.asarray(
                record.get("coordinate_mask", np.ones(density.shape, dtype=bool)),
                dtype=bool)
            candidate = (
                coordinate_mask
                & np.isfinite(density) & (density > 0.0)
                & np.isfinite(altitude)
                & np.isfinite(timestamp)
                & np.isfinite(latitude)
                & np.isfinite(longitude)
                & (altitude >= EXPECTED_MODEL_RANGE[0])
                & (altitude <= EXPECTED_MODEL_RANGE[1])
            )
            row, column = np.where(candidate)
            keys = np.asarray([
                f"{station}|{date_utc}|{int(timestamp[i, j])}|{altitude[i, j]:.3f}"
                for i, j in zip(row, column)
            ], dtype="U64")
            if len(keys) == 0:
                continue
            if any(key in seen for key in keys.tolist()):
                raise ValueError("raw ISR query construction produced duplicate keys")
            seen.update(keys.tolist())

            observation = np.log10(density[row, column])
            times = timestamp[row, column].astype(np.int64)
            alts = altitude[row, column].astype(np.float32)
            query_latitudes = latitude[row, column].astype(np.float32)
            query_longitudes = longitude[row, column].astype(np.float32)
            aacgm_latitudes, aacgm_mlt_hours = (
                _compute_aacgm_query_coordinates(
                    query_latitudes, query_longitudes, alts, times))
            query_ids = np.arange(
                next_query_id, next_query_id + len(keys), dtype=np.int64)
            next_query_id += len(keys)

            selection = RawSelection(
                station=station,
                date_utc=date_utc,
                query_split=query_split,
                keys=keys,
                query_ids=query_ids,
                timestamps=times,
                latitudes=query_latitudes,
                longitudes=query_longitudes,
                altitudes=alts,
                relative_hours=(
                    (times.astype(np.float64) - start_date.timestamp()) / 3600.0
                ).astype(np.float32),
                aacgm_latitudes=aacgm_latitudes,
                aacgm_mlt_hours=aacgm_mlt_hours,
                observations_log10=observation.astype(np.float32),
            )
            selections.append(selection)
            partition_counts[query_split] += selection.count
            station_counts[station] += selection.count

    if not seen or next_query_id != len(seen):
        raise ValueError("allowed raw ISR query registry is empty or inconsistent")
    allowed_partition_counts = {
        "train": int(partition_counts["train"]),
        "development": int(partition_counts["development"]),
    }
    if allowed_partition_counts != contract_expected_counts:
        raise ValueError(
            "raw ISR train/development query counts differ from the frozen "
            f"registry: actual={allowed_partition_counts}, "
            f"expected={dict(contract_expected_counts)}")
    selections.sort(key=lambda item: (
        0 if item.station == "Jicamarca" else 1, item.date_utc))
    return selections, {
        "query_registry_source": "raw_ISR_after_contract_file_date_filter_v1",
        "registered_query_count": int(len(seen)),
        "station_counts": station_counts,
        "date_partition_counts": allowed_partition_counts,
        "allowed_query_count": int(len(seen)),
        "timestamp_metadata_access": "classification_only_before_density_slice",
        "excluded_date_density_columns_read": 0,
        "locked_test_density_columns_read": 0,
        "p0a_npz_or_cache_read": False,
        "p0a_peak_cache_read": False,
    }


def _load_runtime_managers(
        config: Mapping[str, Any], device: torch.device,
        allowlists: Mapping[str, np.ndarray]):
    from inr_modules.data_managers.FY_dataloader import (
        COSMICNeighborhoodIndex,
        FYNeighborhoodIndex,
    )
    from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager

    sw_manager = SpaceWeatherManager(
        txt_path=config["sw_path"],
        start_date_str=config["start_date_str"],
        total_hours=config["total_hours"],
        seq_len=config["seq_len"],
        device=device,
    )
    iri_peak = IRIPeakManager(
        hmf2_path=config["iri_hmf2_path"],
        nmf2_path=config["iri_nmf2_path"],
        device=device,
    )
    fy_config = dict(config)
    fy_config["neighbor_directory_semantics"] = (
        "token_exact_positive_support_v1")
    fy_config["strict_preload_token_only"] = True
    fy_config["strict_preload_allowed_profile_ids"] = np.asarray(
        allowlists["FY"], dtype=np.int64)
    cosmic_config = dict(config)
    cosmic_config["neighbor_directory_semantics"] = (
        "token_exact_positive_support_v1")
    cosmic_config["strict_preload_token_only"] = True
    cosmic_config["strict_preload_allowed_profile_ids"] = np.asarray(
        allowlists["COSMIC"], dtype=np.int64)
    fy_index = FYNeighborhoodIndex(config["fy_path"], fy_config)
    cosmic_index = COSMICNeighborhoodIndex(config["cosmic_path"], cosmic_config)
    return sw_manager, iri_peak, {"FY": fy_index, "COSMIC": cosmic_index}


def _validate_loaded_train_only_indexes(
        indexes: Mapping[str, Any],
        allowlists: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Prove that strict preload materialized exactly the train allowlists."""
    result: dict[str, Any] = {}
    for source in SOURCES:
        index = indexes[source]
        actual_profiles = np.unique(np.asarray(
            index.token_profile_ids, dtype=np.int64))
        expected_profiles = np.asarray(allowlists[source], dtype=np.int64)
        if not np.array_equal(actual_profiles, expected_profiles):
            raise ValueError(
                f"{source} strict-preload token profiles differ from the "
                "train-only allowlist")
        if len(index.token_values) == 0 or not np.isfinite(
                np.asarray(index.token_values)).all():
            raise ValueError(f"{source} strict-preload token directory is invalid")
        if (getattr(index, "strict_preload_token_only", None) is not True
                or getattr(index, "token_mode", None) is not True):
            raise ValueError(
                f"{source} index is not using strict exact-token preload")
        result[source] = {
            "strict_preload_verified": True,
            "strict_preload_token_only": True,
            "neighbor_directory_semantics": (
                "token_exact_positive_support_v1"),
            "unique_profiles": int(len(actual_profiles)),
            "token_rows": int(len(index.token_values)),
            "profile_partition": "train",
        }
    return result


def _torch_numpy(value: torch.Tensor, dtype=None) -> np.ndarray:
    result = value.detach().cpu().numpy()
    return result.astype(dtype, copy=False) if dtype is not None else result


def _ragged_terms(
        source: str, payload: Mapping[str, torch.Tensor],
        extras: Mapping[str, torch.Tensor]) -> RaggedSourceTerms:
    variance = extras["r_fy" if source == "FY" else "r_cosmic"]
    if (variance.numel() != 1 or not torch.isfinite(variance).all()
            or float(variance.item()) <= 0.0):
        raise ValueError(f"{source} observation variance must be finite and positive")
    query_index = payload["query_index"].long()
    basis64 = extras[f"basis_{source}"].double()
    latent64 = extras["latent_anomalies"][query_index].double()
    observation_anomalies64 = torch.einsum(
        "ed,end->en", basis64, latent64)
    diagnostic = extras[f"obs_anomalies_{source}"]
    if not torch.equal(observation_anomalies64.to(diagnostic.dtype), diagnostic):
        raise ValueError(
            f"{source} pre-cast observation anomalies do not reproduce diagnostics")
    representativeness = extras[f"representativeness_{source}"].double()
    valid = payload.get("valid_mask")
    if valid is None:
        valid64 = torch.ones_like(representativeness, dtype=torch.float64)
    else:
        valid64 = valid.to(dtype=torch.float64)
    diag_precision = (
        valid64 * representativeness / variance.double())
    return RaggedSourceTerms(
        source=source,
        query_index=query_index,
        profile_id=payload["profile_id"].long(),
        altitude_km=payload["coords"][:, 2].double(),
        obs_anomalies=observation_anomalies64,
        innovation=extras[f"innov_{source}"],
        localized_precision=extras[f"precision_{source}"],
        diag_r_precision=diag_precision,
    )


def _validate_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite P0-B tensor: {name}")


def _append_query_rows(
        rows: dict[str, list[Any]], selection: RawSelection, sl: slice,
        batch_id: int, coords: torch.Tensor, sw_seq: torch.Tensor,
        extras: Mapping[str, torch.Tensor], modes: Mapping[str, torch.Tensor],
        no_token: torch.Tensor, suite,
        source_terms: Mapping[str, RaggedSourceTerms]) -> None:
    indices = np.arange(selection.count)[sl]
    query_ids = selection.query_ids[sl].astype(np.int64)
    m00 = extras["ne_bkg"].reshape(-1)
    m10 = m00 + modes["M10"]
    m01 = m00 + modes["M01"]
    m11 = m00 + modes["M11"]
    joint_fy = extras["update_FY"].reshape(-1)
    joint_cosmic = extras["update_COSMIC"].reshape(-1)
    joint_increment = extras["ne_residual"].reshape(-1)
    cos_sza = _compute_solar_features(
        coords[:, 0], coords[:, 1], coords[:, 3])[0]
    kp = 4.0 * (sw_seq[:, -1, 0] + 1.0)
    f107 = 60.0 * sw_seq[:, -1, 1] + 210.0

    per_source: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCES}
    for source in SOURCES:
        terms = source_terms[source]
        unlocalized = terms.diag_r_precision
        for query_index in range(len(indices)):
            mask = terms.query_index == query_index
            profile_ids = _torch_numpy(terms.profile_id[mask], np.int64)
            localized = _torch_numpy(
                terms.localized_precision[mask], np.float64)
            stats = profile_precision_statistics(profile_ids, localized)
            stats["unlocalized_precision_sum"] = float(
                terms.diag_r_precision[mask].sum().item())
            per_source[source].append(stats)

    fy_raw_covered = np.asarray([
        value["token_count"] > 0 for value in per_source["FY"]], dtype=bool)
    cosmic_raw_covered = np.asarray([
        value["token_count"] > 0 for value in per_source["COSMIC"]], dtype=bool)
    fy_covered = np.asarray([
        value["localized_precision_sum"] > 0.0
        for value in per_source["FY"]], dtype=bool)
    cosmic_covered = np.asarray([
        value["localized_precision_sum"] > 0.0
        for value in per_source["COSMIC"]], dtype=bool)
    raw_coverage = np.full(len(indices), "no_token", dtype="U12")
    raw_coverage[fy_raw_covered & ~cosmic_raw_covered] = "FY_only"
    raw_coverage[~fy_raw_covered & cosmic_raw_covered] = "COSMIC_only"
    raw_coverage[fy_raw_covered & cosmic_raw_covered] = "joint"
    coverage = np.full(len(indices), "no_token", dtype="U12")
    coverage[fy_covered & ~cosmic_covered] = "FY_only"
    coverage[~fy_covered & cosmic_covered] = "COSMIC_only"
    coverage[fy_covered & cosmic_covered] = "joint"

    for mode in (*SOURCES, "joint"):
        nis = suite.predictive_nis[mode]
        dof = suite.predictive_nis_valid_dof[mode]
        if nis.dtype != torch.float64:
            raise TypeError(f"{mode} predictive NIS must be calculated in float64")
        if (not torch.isfinite(nis).all() or (nis < 0.0).any()
                or dof.dtype != torch.long or (dof < 0).any()):
            raise ValueError(f"invalid {mode} predictive NIS or dof")

    m11_numpy = _torch_numpy(m11, np.float32)

    values: dict[str, np.ndarray] = {
        "query_id": query_ids,
        "sample_key": selection.keys[sl].astype("U64"),
        "station": np.full(len(indices), selection.station, dtype="U16"),
        "date_utc": np.full(len(indices), selection.date_utc, dtype="U8"),
        "batch_id": np.full(len(indices), batch_id, dtype=np.int64),
        "query_profile_id": selection.timestamps[sl].astype(np.int64),
        "timestamp_unix": selection.timestamps[sl].astype(np.int64),
        "query_split": np.full(len(indices), selection.query_split, dtype="U11"),
        "latitude_deg": selection.latitudes[sl],
        "longitude_deg": selection.longitudes[sl],
        "altitude_km": selection.altitudes[sl],
        "relative_hour": selection.relative_hours[sl],
        "local_time_hour": np.mod(
            (selection.timestamps[sl].astype(np.float64) / 3600.0) % 24.0
            + selection.longitudes[sl].astype(np.float64) / 15.0,
            24.0).astype(np.float32),
        "aacgm_latitude_deg": selection.aacgm_latitudes[sl],
        "aacgm_mlt_hour": selection.aacgm_mlt_hours[sl],
        "cos_sza": _torch_numpy(cos_sza, np.float32),
        "kp": _torch_numpy(kp, np.float32),
        "f107": _torch_numpy(f107, np.float32),
        "isr_log10_ne": selection.observations_log10[sl],
        "raw_iri_log10_ne": _torch_numpy(extras["ne_iri"].reshape(-1), np.float32),
        "M00_log10_ne": _torch_numpy(m00, np.float32),
        "M10_log10_ne": _torch_numpy(m10, np.float32),
        "M01_log10_ne": _torch_numpy(m01, np.float32),
        "M11_log10_ne": m11_numpy,
        "no_token_log10_ne": _torch_numpy(no_token.reshape(-1), np.float32),
        "isolated_increment_FY_dex": _torch_numpy(modes["M10"], np.float32),
        "isolated_increment_COSMIC_dex": _torch_numpy(modes["M01"], np.float32),
        "joint_update_FY_dex": _torch_numpy(joint_fy, np.float32),
        "joint_update_COSMIC_dex": _torch_numpy(joint_cosmic, np.float32),
        "joint_increment_dex": _torch_numpy(joint_increment, np.float32),
        "raw_coverage_code": raw_coverage,
        "coverage_code": coverage,
        "CF_drop_200_250_log10_ne": _torch_numpy(
            m00 + suite.altitude_deletions["drop_200_250"].m11,
            np.float32),
        "CF_drop_250_300_log10_ne": _torch_numpy(
            m00 + suite.altitude_deletions["drop_250_300"].m11,
            np.float32),
        "CF_drop_300_400_log10_ne": _torch_numpy(
            m00 + suite.altitude_deletions["drop_300_400"].m11,
            np.float32),
        "CF_drop_400_500_log10_ne": _torch_numpy(
            m00 + suite.altitude_deletions["drop_400_500"].m11,
            np.float32),
        "CF_duplicate_FY_dominant_profile_log10_ne": _torch_numpy(
            m00 + suite.profile_duplications["duplicate_FY"].m11,
            np.float32),
        "CF_duplicate_COSMIC_dominant_profile_log10_ne": _torch_numpy(
            m00 + suite.profile_duplications["duplicate_COSMIC"].m11,
            np.float32),
        "CF_duplicate_both_dominant_profiles_log10_ne": _torch_numpy(
            m00 + suite.profile_duplications["duplicate_both"].m11,
            np.float32),
        "predictive_nis_unlocalized_FY": _torch_numpy(
            suite.predictive_nis["FY"], np.float32),
        "predictive_nis_unlocalized_FY_dof": _torch_numpy(
            suite.predictive_nis_valid_dof["FY"], np.int64),
        "predictive_nis_unlocalized_COSMIC": _torch_numpy(
            suite.predictive_nis["COSMIC"], np.float32),
        "predictive_nis_unlocalized_COSMIC_dof": _torch_numpy(
            suite.predictive_nis_valid_dof["COSMIC"], np.int64),
        "predictive_nis_unlocalized_joint": _torch_numpy(
            suite.predictive_nis["joint"], np.float32),
        "predictive_nis_unlocalized_joint_dof": _torch_numpy(
            suite.predictive_nis_valid_dof["joint"], np.int64),
    }
    for source in SOURCES:
        stats = per_source[source]
        for field in (
                "token_count", "unique_profile_count",
                "unlocalized_precision_sum", "localized_precision_sum",
                "token_neff", "profile_neff", "max_profile_precision_share"):
            values[f"{source}_{field}"] = np.asarray(
                [row[field] for row in stats])
        dominant = [row["dominant_profile_id"] for row in stats]
        values[f"{source}_dominant_profile_id"] = np.asarray([
            -1 if value is None else value for value in dominant], dtype=np.int64)
        values[f"{source}_dominant_profile_valid"] = np.asarray([
            value is not None for value in dominant], dtype=bool)

    if set(values) != set(QUERY_DTYPES):
        raise ValueError(
            "query producer/schema mismatch: "
            f"missing={sorted(set(QUERY_DTYPES).difference(values))}, "
            f"extra={sorted(set(values).difference(QUERY_DTYPES))}")
    for key in QUERY_DTYPES:
        rows[key].extend(np.asarray(values[key]).tolist())


def _append_token_edge_rows(
        token_rows: dict[str, list[Any]], edge_rows: dict[str, list[Any]],
        selection: RawSelection, sl: slice, batch_id: int,
        extras: Mapping[str, torch.Tensor],
        payloads: Mapping[str, Mapping[str, torch.Tensor]],
        source_terms: Mapping[str, RaggedSourceTerms],
        suite,
        next_token_row_id: int) -> int:
    query_ids = selection.query_ids[sl].astype(np.int64)
    dedup: dict[tuple[str, int, int], tuple[int, tuple[Any, ...]]] = {}
    edge_identity: set[tuple[int, int]] = set()

    for source in SOURCES:
        payload = payloads[source]
        terms = source_terms[source]
        edge_count = terms.edge_count
        if edge_count == 0:
            continue
        query_index = _torch_numpy(payload["query_index"], np.int64)
        profile_id = _torch_numpy(payload["profile_id"], np.int64)
        token_id = _torch_numpy(payload["token_id"], np.int64)
        coords = _torch_numpy(payload["coords"], np.float32)
        value = _torch_numpy(payload["value"], np.float32)
        background = _torch_numpy(payload["background"], np.float32)
        innovation = _torch_numpy(terms.innovation, np.float32)
        localization = _torch_numpy(payload["localization_weight"], np.float32)
        representativeness = _torch_numpy(
            extras[f"representativeness_{source}"], np.float32)
        localized_precision = _torch_numpy(
            terms.localized_precision, np.float32)
        unlocalized_precision = _torch_numpy(
            terms.diag_r_precision, np.float32)
        obs_anomalies = terms.obs_anomalies
        prior_variance = _torch_numpy(
            obs_anomalies.square().sum(dim=-1)
            / float(max(obs_anomalies.shape[-1] - 1, 1)), np.float32)
        r_variance_value = float(
            extras["r_fy" if source == "FY" else "r_cosmic"].item())
        r_variance = np.full(edge_count, r_variance_value, dtype=np.float32)
        r_standardized = _torch_numpy(
            suite.diag_r_standardized_innovation_sq[source], np.float64)
        predictive_diag = innovation.astype(np.float64) ** 2 / (
            prior_variance.astype(np.float64)
            + 1.0 / np.maximum(unlocalized_precision.astype(np.float64), 1e-30))
        localized_energy = localization.astype(np.float64) * r_standardized
        gain_record = suite.edge_gain_coefficients[source]
        helper_joint = gain_record.joint_system_gain_coefficient
        production_joint = extras[f"K_{source}"]
        if (helper_joint.shape != production_joint.shape
                or not torch.isfinite(helper_joint).all()
                or not torch.allclose(
                    helper_joint, production_joint, rtol=5e-6, atol=5e-6)):
            raise ValueError(
                f"{source} helper joint gain disagrees with production K")
        gain_joint = _torch_numpy(helper_joint, np.float32)
        gain_isolated = _torch_numpy(
            gain_record.isolated_system_gain_coefficient, np.float32)
        contribution_joint = gain_joint.astype(np.float64) * innovation.astype(np.float64)
        contribution_isolated = (
            gain_isolated.astype(np.float64) * innovation.astype(np.float64))
        space_distance = _torch_numpy(payload["space_distance_km"], np.float32)
        time_distance = _torch_numpy(payload["time_distance_hours"], np.float32)

        expected_localized = unlocalized_precision * localization
        if not np.allclose(
                localized_precision, expected_localized, rtol=5e-6, atol=5e-6):
            raise ValueError(f"{source} localized precision does not close")

        for edge_index in range(edge_count):
            identity = (source, int(profile_id[edge_index]), int(token_id[edge_index]))
            token_value = (
                float(coords[edge_index, 0]), float(coords[edge_index, 1]),
                float(coords[edge_index, 2]), float(coords[edge_index, 3]),
                float(value[edge_index]), float(background[edge_index]),
            )
            if identity not in dedup:
                row_id = next_token_row_id
                next_token_row_id += 1
                dedup[identity] = (row_id, token_value)
                token_rows["token_row_id"].append(row_id)
                token_rows["station"].append(selection.station)
                token_rows["date_utc"].append(selection.date_utc)
                token_rows["batch_id"].append(batch_id)
                token_rows["source"].append(source)
                token_rows["profile_id"].append(identity[1])
                token_rows["token_id"].append(identity[2])
                token_rows["profile_split"].append("train")
                token_rows["latitude_deg"].append(token_value[0])
                token_rows["longitude_deg"].append(token_value[1])
                token_rows["altitude_km"].append(token_value[2])
                token_rows["relative_hour"].append(token_value[3])
                token_rows["observation_log10_ne"].append(token_value[4])
                token_rows["background_log10_ne"].append(token_value[5])
            else:
                row_id, previous = dedup[identity]
                if not np.array_equal(
                        np.asarray(previous, dtype=np.float32),
                        np.asarray(token_value, dtype=np.float32)):
                    raise ValueError(
                        f"duplicate token identity has inconsistent values: {identity}")
            row_id = dedup[identity][0]
            query_id = int(query_ids[query_index[edge_index]])
            if (query_id, row_id) in edge_identity:
                raise ValueError("duplicate query-token edge in production payload")
            edge_identity.add((query_id, row_id))

            edge_rows["station"].append(selection.station)
            edge_rows["date_utc"].append(selection.date_utc)
            edge_rows["batch_id"].append(batch_id)
            edge_rows["query_id"].append(query_id)
            edge_rows["token_row_id"].append(row_id)
            edge_rows["source"].append(source)
            edge_rows["innovation_dex"].append(innovation[edge_index])
            edge_rows["r_variance_dex2"].append(r_variance[edge_index])
            edge_rows["representativeness_weight"].append(
                representativeness[edge_index])
            edge_rows["localization_weight"].append(localization[edge_index])
            edge_rows["unlocalized_precision"].append(
                unlocalized_precision[edge_index])
            edge_rows["localized_precision"].append(
                localized_precision[edge_index])
            edge_rows["prior_predictive_variance_dex2"].append(
                prior_variance[edge_index])
            edge_rows["r_standardized_innovation_sq"].append(
                r_standardized[edge_index])
            edge_rows["predictive_diagonal_nis"].append(
                predictive_diag[edge_index])
            edge_rows["localized_innovation_energy"].append(
                localized_energy[edge_index])
            edge_rows["gain_joint"].append(gain_joint[edge_index])
            edge_rows["gain_isolated"].append(gain_isolated[edge_index])
            edge_rows["contribution_joint_dex"].append(
                contribution_joint[edge_index])
            edge_rows["contribution_isolated_dex"].append(
                contribution_isolated[edge_index])
            edge_rows["space_distance_km"].append(space_distance[edge_index])
            edge_rows["time_distance_hours"].append(time_distance[edge_index])
    return next_token_row_id


def _infer_batch(
        model: torch.nn.Module, sw_manager, iri_peak_manager,
        indexes: Mapping[str, Any], allowlists: Mapping[str, np.ndarray],
        frozen_contract: Mapping[str, Any], selection: RawSelection, sl: slice,
        batch_id: int, device: torch.device,
        next_token_row_id: int) -> tuple[
            dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray],
            int, dict[str, float]]:
    coords_np = np.column_stack((
        selection.latitudes[sl], selection.longitudes[sl],
        selection.altitudes[sl], selection.relative_hours[sl],
    )).astype(np.float32)
    coords = torch.from_numpy(coords_np).to(device)
    with torch.no_grad():
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        iri_peak = iri_peak_manager.get_iri_peak(coords)
        payloads = {}
        for source in SOURCES:
            payload = query_observation_payload(
                indexes[source], coords, device,
                allowed_profile_ids=allowlists[source])
            payloads[source] = attach_observation_background(
                payload, model, sw_manager, iri_peak_manager)
            if payloads[source]["profile_id"].numel():
                actual_ids = _torch_numpy(payloads[source]["profile_id"], np.int64)
                if not np.isin(actual_ids, allowlists[source]).all():
                    raise ValueError(f"{source} payload escaped train-only allowlist")

        fused, _, _, _, extras = model(
            coords, sw_seq, iri_peak=iri_peak,
            observations_fy=payloads["FY"],
            observations_cosmic=payloads["COSMIC"])
        modes = solve_density_modes(extras)
        no_token, _, _, no_token_delta, no_token_extras = model(
            coords, sw_seq, iri_peak=iri_peak,
            observations_fy=None, observations_cosmic=None)

    for name, value in (
            ("joint_fused", fused), ("M00", extras["ne_bkg"]),
            ("M10_increment", modes["M10"]),
            ("M01_increment", modes["M01"]),
            ("M11_increment", modes["M11"]),
            ("no_token", no_token)):
        _validate_finite_tensor(name, value)
    if not torch.equal(no_token.reshape(-1), extras["ne_bkg"].reshape(-1)):
        raise ValueError("actual no-token forward is not bitwise identical to M00")
    if not torch.equal(no_token_delta, torch.zeros_like(no_token_delta)):
        raise ValueError("actual no-token forward has a nonzero increment")
    if not torch.equal(
            no_token.reshape(-1), no_token_extras["ne_bkg"].reshape(-1)):
        raise ValueError("no-token fused/background identity failed")
    if not torch.allclose(
            fused.reshape(-1),
            extras["ne_bkg"].reshape(-1) + modes["M11"],
            rtol=5e-6, atol=5e-6):
        raise ValueError("joint forward and solve_density_modes M11 disagree")

    source_terms = {
        source: _ragged_terms(source, payloads[source], extras)
        for source in SOURCES
    }
    suite = run_p0b_counterfactual_suite(
        extras["query_anomalies"], source_terms)
    for source, field in (("FY", "M10"), ("COSMIC", "M01")):
        suite_value = suite.baseline.m10 if source == "FY" else suite.baseline.m01
        if not torch.allclose(
                suite_value, modes[field], rtol=5e-6, atol=5e-6):
            raise ValueError(f"P0-B counterfactual adapter disagrees for {field}")
    if not torch.allclose(
            suite.baseline.m11, modes["M11"], rtol=5e-6, atol=5e-6):
        raise ValueError("P0-B counterfactual adapter disagrees for M11")

    query_rows = _empty_rows(QUERY_DTYPES)
    token_rows = _empty_rows(TOKEN_DTYPES)
    edge_rows = _empty_rows(EDGE_DTYPES)
    _append_query_rows(
        query_rows, selection, sl, batch_id, coords, sw_seq,
        extras, modes, no_token, suite, source_terms)
    next_token_row_id = _append_token_edge_rows(
        token_rows, edge_rows, selection, sl, batch_id, extras,
        payloads, source_terms, suite, next_token_row_id)
    query = _as_table(query_rows, QUERY_DTYPES)
    token = _as_table(token_rows, TOKEN_DTYPES)
    edge = _as_table(edge_rows, EDGE_DTYPES)
    validation = _validate_batch_against_contract(
        frozen_contract, query, token, edge)
    return query, token, edge, next_token_row_id, {
        "query_closure_max_abs_error": float(max(
            validation["query_closure"]["max_abs_errors"].values(), default=0.0)),
        "edge_closure_max_abs_error": float(max(
            validation["edge_closure_max_abs_errors"].values(), default=0.0)),
        "recomputed_summary_max_abs_error": float(
            validation["recomputed_diagnostics"][
                "max_abs_recomputed_summary_error"]),
    }


def _write_runtime_contract(
        output_dir: Path, frozen_contract_path: Path, frozen_contract_sha: str,
        checkpoint: Path, checkpoint_sha: str,
        p0a: P0AIdentity, date_split_path: Path,
        input_data_sha256: Mapping[str, str],
        train_allowlist_identity: Mapping[str, Any],
        train_token_directory_identity: Mapping[str, Any],
        isr_source_files: list[dict[str, Any]],
        isr_column_access_audit: Mapping[str, Any],
        required_runtime_identity_names: Iterable[str],
        profile_metadata: Mapping[str, Any], source_provenance: Mapping[str, Any],
        git_provenance: Mapping[str, Any],
        python_environment: Mapping[str, Any],
        coordinate_status: Mapping[str, Any],
        preflight_dependency: Mapping[str, Any] | None) -> Path:
    token_directory_identity = validate_train_token_directory_identity(
        train_token_directory_identity)
    runtime_identities = {
        "audit_code_sha256": sha256_file(SCRIPT_PATH),
        "git_head": git_provenance["head"],
        "python_environment": dict(python_environment),
        "checkpoint_path": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
        },
        "date_split_path": {
            "path": str(date_split_path),
            "sha256": sha256_file(date_split_path),
        },
        "input_data_sha256": dict(input_data_sha256),
        "train_allowlist_identity": dict(train_allowlist_identity),
        "train_token_directory_identity": token_directory_identity,
        "ISR_source_file_sha256": list(isr_source_files),
        "ISR_column_access_audit": dict(isr_column_access_audit),
        "coordinate_enrichment": dict(coordinate_status),
        "profile_cap_status": "not_applicable_in_v14",
    }
    if set(runtime_identities) != set(required_runtime_identity_names):
        raise ValueError("runner runtime identities differ from frozen P0-B contract")
    value = {
        "audit_schema_version": P0B_AUDIT_SCHEMA_VERSION,
        "status": "runtime_contract_bound",
        "created_utc": _utc_now(),
        "frozen_p0b_contract": {
            "path": str(frozen_contract_path),
            "sha256": frozen_contract_sha,
        },
        "candidate_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "checkpoint_format_version": EXPECTED_CHECKPOINT_FORMAT,
            "model_domain_semantics": EXPECTED_DOMAIN,
        },
        "p0a_dependency": {
            "contract_path": str(p0a.contract_path),
            "contract_sha256": p0a.sha256,
            "contract_size_bytes": p0a.size_bytes,
            "access_mode": (
                "qav2_JSON_contract_identity_only_no_NPZ_or_peak_cache_v1"),
            "p0a_npz_or_cache_read": False,
            "p0a_peak_cache_read": False,
            "p0a_token_partitions": list(p0a.contract["token_partitions"]),
            "p0b_token_partitions": ["train"],
            "prediction_semantics_note": (
                "P0-A M11 used train+development tokens; P0-B uses train-only "
                "tokens; no P0-A numeric output is joined or compared"
            ),
        },
        "date_split": {
            "path": str(date_split_path),
            "sha256": sha256_file(date_split_path),
        },
        "satellite_profile_metadata": profile_metadata,
        "train_allowlist_identity": dict(train_allowlist_identity),
        "train_token_directory_identity": token_directory_identity,
        "source_provenance": source_provenance,
        "git": git_provenance,
        "python_environment": dict(python_environment),
        "coordinate_enrichment": dict(coordinate_status),
        "ISR_column_access_audit": dict(isr_column_access_audit),
        "profile_cap_status": "not_applicable_in_v14",
        "preflight_dependency": (
            dict(preflight_dependency) if preflight_dependency is not None else {
                "required": False,
                "reason": "this runtime is the preflight itself",
            }),
        "runtime_identities": runtime_identities,
        "mutation_policy": {
            "training": False,
            "gradients": False,
            "qc": False,
            "satellite_development_inference": False,
            "locked_test_inference": False,
            "checkpoint_selection": False,
        },
    }
    return atomic_write_json(output_dir / "audit_contract.json", value)


def _initial_manifest(
        args: argparse.Namespace, output_dir: Path,
        raw_summary: Mapping[str, Any], profile_metadata: Mapping[str, Any],
        checkpoint_sha: str, model_state_sha: str) -> dict[str, Any]:
    return {
        "audit_schema_version": P0B_AUDIT_SCHEMA_VERSION,
        "status": "preflight_running" if args.preflight_only else "running",
        "preflight_only": bool(args.preflight_only),
        "started_utc": _utc_now(),
        "output_directory": str(output_dir),
        "checkpoint_sha256_before": checkpoint_sha,
        "checkpoint_sha256_after": None,
        "model_state_sha256_before": model_state_sha,
        "model_state_sha256_after": None,
        "raw_registry": dict(raw_summary),
        "satellite_profile_metadata": dict(profile_metadata),
        "counts": {"query_rows": 0, "token_rows": 0, "edge_rows": 0,
                   "batches": 0},
        "max_abs_errors": {
            "query_closure": 0.0,
            "edge_closure": 0.0,
            "recomputed_summary": 0.0,
        },
        "artifacts": [],
    }


def _run_preflight_inference_batch(
        model: torch.nn.Module, sw_manager: Any, iri_peak_manager: Any,
        indexes: Mapping[str, Any], allowlists: Mapping[str, np.ndarray],
        frozen_contract: Mapping[str, Any], selection: RawSelection,
        batch_size: int, device: torch.device) -> dict[str, Any]:
    """Run one non-persisted frozen batch through the complete audit chain."""
    query_count = min(int(batch_size), selection.count)
    if query_count <= 0:
        raise ValueError("P0-B preflight selection is empty")
    query, token, edge, _, errors = _infer_batch(
        model, sw_manager, iri_peak_manager, indexes, allowlists,
        frozen_contract, selection, slice(0, query_count), 0, device, 0)
    positive_edge = np.asarray(edge["localized_precision"], dtype=np.float64) > 0.0
    edge_sources = np.asarray(edge["source"])
    positive_edges_by_source = {
        source: int(np.count_nonzero(positive_edge & (edge_sources == source)))
        for source in SOURCES
    }
    token_altitude_by_id = {
        int(token_row_id): float(altitude_km)
        for token_row_id, altitude_km in zip(
            token["token_row_id"], token["altitude_km"], strict=True)
    }
    positive_edge_altitudes = np.asarray([
        token_altitude_by_id[int(token_row_id)]
        for token_row_id in np.asarray(edge["token_row_id"])[positive_edge]
    ], dtype=np.float64)
    positive_edges_by_height_band: dict[str, int] = {}
    for band in frozen_contract["counterfactuals"]["height_deletion"]["bands"]:
        lower = float(band["lower_km"])
        upper = float(band["upper_km"])
        lower_mask = (
            positive_edge_altitudes >= lower
            if band["lower_inclusive"] else positive_edge_altitudes > lower)
        upper_mask = (
            positive_edge_altitudes <= upper
            if band["upper_inclusive"] else positive_edge_altitudes < upper)
        positive_edges_by_height_band[str(band["name"])] = int(
            np.count_nonzero(lower_mask & upper_mask))
    summary = {
        "status": "full_chain_batch_pass_no_numeric_persistence",
        "station": selection.station,
        "date_utc": selection.date_utc,
        "query_split": selection.query_split,
        "query_rows": int(len(query["query_id"])),
        "token_rows": int(len(token["token_row_id"])),
        "edge_rows": int(len(edge["query_id"])),
        "raw_coverage_codes_observed": sorted(
            set(map(str, query["raw_coverage_code"]))),
        "effective_coverage_codes_observed": sorted(
            set(map(str, query["coverage_code"]))),
        "token_sources_observed": sorted(set(map(str, token["source"]))),
        "positive_localized_precision_edges_by_source": positive_edges_by_source,
        "effective_joint_queries": int(np.count_nonzero(
            np.asarray(query["coverage_code"]) == "joint")),
        "valid_dominant_profile_queries_by_source": {
            source: int(np.count_nonzero(
                np.asarray(query[f"{source}_dominant_profile_valid"], dtype=bool)))
            for source in SOURCES
        },
        "positive_localized_precision_edges_by_height_deletion_band": (
            positive_edges_by_height_band),
        "query_closure_max_abs_error": float(
            errors["query_closure_max_abs_error"]),
        "edge_closure_max_abs_error": float(
            errors["edge_closure_max_abs_error"]),
        "recomputed_summary_max_abs_error": float(
            errors["recomputed_summary_max_abs_error"]),
        "npz_written": False,
        "numeric_predictions_persisted": False,
    }
    del query, token, edge
    gc.collect()
    return summary


def _subset_raw_selection(
        selection: RawSelection, indices: np.ndarray) -> RawSelection:
    indices = np.asarray(indices, dtype=np.int64)
    if (indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0)
            or np.any(indices >= selection.count)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("invalid deterministic preflight selection indices")
    return RawSelection(
        station=selection.station,
        date_utc=selection.date_utc,
        query_split=selection.query_split,
        keys=selection.keys[indices],
        query_ids=selection.query_ids[indices],
        timestamps=selection.timestamps[indices],
        latitudes=selection.latitudes[indices],
        longitudes=selection.longitudes[indices],
        altitudes=selection.altitudes[indices],
        relative_hours=selection.relative_hours[indices],
        aacgm_latitudes=selection.aacgm_latitudes[indices],
        aacgm_mlt_hours=selection.aacgm_mlt_hours[indices],
        observations_log10=selection.observations_log10[indices],
    )


def _run_preflight_inference_suite(
        model: torch.nn.Module, sw_manager: Any, iri_peak_manager: Any,
        indexes: Mapping[str, Any], allowlists: Mapping[str, np.ndarray],
        frozen_contract: Mapping[str, Any], selections: Iterable[RawSelection],
        device: torch.device) -> dict[str, Any]:
    """Probe every frozen station/split/altitude cell without persisting values."""
    probe = frozen_contract["output_lifecycle"]["preflight_probe"]
    candidates = sorted(
        selections,
        key=lambda item: (item.station, item.query_split, item.date_utc))
    max_queries = int(probe["max_queries_per_cell"])
    bands = [tuple(map(float, value)) for value in probe["altitude_bands_km"]]
    results: list[dict[str, Any]] = []
    probe_id = 0
    for station in probe["stations"]:
        for query_split in probe["query_splits"]:
            scoped = [
                item for item in candidates
                if item.station == station and item.query_split == query_split]
            if not scoped:
                raise ValueError(
                    f"preflight lacks selection for {station}/{query_split}")
            for band_index, (lower, upper) in enumerate(bands):
                selected_item = None
                selected_indices = None
                final_band = band_index == len(bands) - 1
                for item in scoped:
                    mask = ((item.altitudes >= lower)
                            & ((item.altitudes <= upper) if final_band
                               else (item.altitudes < upper)))
                    indices = np.flatnonzero(mask)
                    if len(indices):
                        selected_item = item
                        if len(indices) > max_queries:
                            offsets = np.linspace(
                                0, len(indices) - 1, max_queries,
                                dtype=np.int64)
                            indices = indices[offsets]
                        selected_indices = indices
                        break
                if selected_item is None or selected_indices is None:
                    raise ValueError(
                        "preflight lacks altitude coverage for "
                        f"{station}/{query_split}/{lower:g}-{upper:g} km")
                subset = _subset_raw_selection(selected_item, selected_indices)
                result = _run_preflight_inference_batch(
                    model, sw_manager, iri_peak_manager, indexes, allowlists,
                    frozen_contract, subset, max_queries, device)
                result.update({
                    "probe_id": probe_id,
                    "altitude_band_km": [lower, upper],
                    "upper_inclusive": final_band,
                })
                results.append(result)
                probe_id += 1
    activation_contract = probe["activation_requirements"]
    required_sources = tuple(
        activation_contract["positive_localized_precision_edge_sources"])
    source_edge_counts = {
        source: sum(
            int(result["positive_localized_precision_edges_by_source"][source])
            for result in results)
        for source in required_sources
    }
    joint_query_count = sum(
        int(result["effective_joint_queries"]) for result in results)
    dominant_profile_counts = {
        source: sum(
            int(result["valid_dominant_profile_queries_by_source"][source])
            for result in results)
        for source in required_sources
    }
    height_band_counts = {
        str(band["name"]): sum(int(
            result[
                "positive_localized_precision_edges_by_height_deletion_band"
            ][str(band["name"])]) for result in results)
        for band in frozen_contract["counterfactuals"]["height_deletion"]["bands"]
    }
    minimum_joint = int(
        activation_contract["minimum_effective_joint_queries"])
    minimum_dominant = int(
        activation_contract[
            "minimum_valid_dominant_profile_queries_per_source"])
    minimum_height_edges = int(
        activation_contract[
            "minimum_positive_localized_precision_edges_per_height_deletion_band"])
    failures = []
    failures.extend(
        f"no positive localized-precision {source} edge"
        for source, count in source_edge_counts.items() if count <= 0)
    if joint_query_count < minimum_joint:
        failures.append(
            f"effective joint queries {joint_query_count} < {minimum_joint}")
    failures.extend(
        f"valid dominant-profile {source} queries {count} < {minimum_dominant}"
        for source, count in dominant_profile_counts.items()
        if count < minimum_dominant)
    failures.extend(
        f"positive localized-precision edges in {name} {count} < "
        f"{minimum_height_edges}"
        for name, count in height_band_counts.items()
        if count < minimum_height_edges)
    if failures:
        raise ValueError(
            "P0-B preflight did not activate every frozen attribution path: "
            + "; ".join(failures))
    activation = {
        "status": "pass",
        "requirements": dict(activation_contract),
        "observed": {
            "positive_localized_precision_edges_by_source": source_edge_counts,
            "effective_joint_queries": joint_query_count,
            "valid_dominant_profile_queries_by_source": dominant_profile_counts,
            "positive_localized_precision_edges_by_height_deletion_band": (
                height_band_counts),
        },
    }
    return {
        "status": "stratified_full_chain_pass_no_numeric_persistence",
        "probe_count": len(results),
        "stations": list(probe["stations"]),
        "query_splits": list(probe["query_splits"]),
        "altitude_bands_km": [list(value) for value in bands],
        "required_paths": list(probe["required_paths"]),
        "activation": activation,
        "no_token_forward_checked_in_every_probe": True,
        "numeric_predictions_persisted": False,
        "npz_written": False,
        "probes": results,
    }


def _iter_scope_batches(
        selections: Iterable[RawSelection], batch_size: int
        ) -> Iterable[tuple[RawSelection, int, slice]]:
    """Yield collision-free batch IDs within each station/date directory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    next_id: dict[tuple[str, str], int] = {}
    emitted: set[tuple[str, str, int]] = set()
    for selection in selections:
        scope = (selection.station, selection.date_utc)
        for start in range(0, selection.count, batch_size):
            batch_id = next_id.get(scope, 0)
            next_id[scope] = batch_id + 1
            identity = (*scope, batch_id)
            if identity in emitted:
                raise RuntimeError("duplicate station/date/batch identity")
            emitted.add(identity)
            yield selection, batch_id, slice(
                start, min(start + batch_size, selection.count))


def _write_failure_ledger(
        output_dir: Path, exc: BaseException,
        manifest: Mapping[str, Any] | None = None) -> None:
    ledger = {
        "audit_schema_version": P0B_AUDIT_SCHEMA_VERSION,
        "status": "failed_no_restart",
        "failed_utc": _utc_now(),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc().splitlines(),
        "completed_counts": (dict(manifest.get("counts", {}))
                             if manifest is not None else {}),
        "completed_artifacts": (list(manifest.get("artifacts", []))
                                if manifest is not None else []),
        "automatic_restart": False,
    }
    atomic_write_json(output_dir / "failure_ledger.json", ledger)


def _run(args: argparse.Namespace) -> int:
    # Identity and, for a full run, the completed preflight dependency are
    # verified before the output directory is created.  This prevents a missing
    # or incompatible preflight from retiring a new full-audit path.
    git_provenance = _git_provenance(
        enforce=True, required_implementation_ref=args.implementation_ref)
    python_environment = _python_environment()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"P0-B output directory must not exist: {output_dir}")

    checkpoint = Path(args.checkpoint).resolve()
    contract_path = Path(args.p0b_contract).resolve()
    frozen_contract, frozen_contract_sha = load_p0b_contract(contract_path)
    _validate_runner_schemas(frozen_contract)
    coordinate_status = _coordinate_runtime_status(frozen_contract)
    _validate_version_control_contract(
        frozen_contract, args.implementation_ref)
    expected_profile_counts = _contract_expected_profile_counts(frozen_contract)
    expected_query_counts = _contract_expected_query_counts(frozen_contract)
    expected_p0a_identity = frozen_contract["data_scope"][
        "P0A_ISR_contract_identity"]
    _validate_identity_contract(
        frozen_contract, checkpoint, args.expected_checkpoint_sha256)
    expected_date_split_sha = frozen_contract["identity"]["date_split_sha256"]
    p0a_dir, p0a_contract_path = _resolve_p0a_paths(args)
    preflight_dependency = None
    if not args.preflight_only:
        preflight_dependency = _validate_preflight_acceptance(
            Path(args.preflight_acceptance), output_dir=output_dir,
            checkpoint=checkpoint,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            frozen_contract=frozen_contract,
            frozen_contract_path=contract_path,
            frozen_contract_sha256=frozen_contract_sha,
            p0a_contract_path=p0a_contract_path,
            git_provenance=git_provenance,
            python_environment=python_environment,
            coordinate_status=coordinate_status)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] | None = None

    try:
        checkpoint_sha_before = sha256_file(checkpoint)

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model, config, run_manifest, training_summary = (
            load_fsia_analysis_checkpoint(
                checkpoint, device=device, require_domain=EXPECTED_DOMAIN))
        if int(config["checkpoint_format_version"]) != EXPECTED_CHECKPOINT_FORMAT:
            raise ValueError("strict loader returned a non-v14 config")
        if tuple(map(float, config["alt_range"])) != EXPECTED_MODEL_RANGE:
            raise ValueError("strict loader returned the wrong model range")
        if tuple(map(float, config["observation_alt_range"])) != (
                EXPECTED_OBSERVATION_RANGE):
            raise ValueError("strict loader returned the wrong observation range")
        if bool(config.get("background_trust_gate_enabled", False)):
            raise ValueError("P0-B candidate must be gate-off")
        model.eval()
        if any(parameter.requires_grad and parameter.grad is not None
               for parameter in model.parameters()):
            raise ValueError("strictly loaded model unexpectedly has gradients")
        model_state_sha_before = _model_state_sha256(model)

        input_data_sha256 = _validate_input_data_identity(
            config, run_manifest, training_summary, expected_date_split_sha)
        if (preflight_dependency is not None
                and input_data_sha256 != preflight_dependency[
                    "input_data_sha256"]):
            raise ValueError("full-audit input data differ from preflight")
        date_split_path, partitions = _load_date_split(
            config, expected_date_split_sha)
        start_date = _parse_utc(config["start_date_str"])
        if not p0a_dir.is_dir():
            raise FileNotFoundError(p0a_dir)
        p0a = _load_p0a_identity(
            p0a_contract_path, args.expected_checkpoint_sha256,
            expected_date_split_sha, expected_p0a_identity)
        allowlists, profile_metadata = _build_train_only_allowlists(
            config, partitions, training_summary, start_date,
            expected_profile_counts)
        profile_count_ledger = {
            "train": {
                source: int(profile_metadata[source]["train_unique_profiles"])
                for source in SOURCES
            },
            "development": {
                source: int(
                    profile_metadata[source]["development_unique_profiles"])
                for source in SOURCES
            },
        }
        train_allowlist_identity = _train_allowlist_identity(
            allowlists, profile_metadata)
        if (preflight_dependency is not None
                and train_allowlist_identity != preflight_dependency[
                    "train_allowlist_identity"]):
            raise ValueError("full-audit train allowlist differs from preflight")

        end_date = start_date + dt.timedelta(hours=float(config["total_hours"]))
        allowed_isr_dates = {
            (start_date + dt.timedelta(days=int(day))).strftime("%Y%m%d")
            for split in ("train", "development")
            for day in partitions[split]
        }
        selected_isr_files = _select_allowed_isr_files(
            p0a.contract, start_date, partitions,
            Path(args.jicamarca_dir), Path(args.poker_flat_dir))
        records, isr_files, isr_access_audits = _load_raw_isr_records(
            selected_isr_files, start_date.timestamp(), end_date.timestamp(),
            allowed_isr_dates)
        isr_column_access_audit = _validate_isr_column_access_audits(
            isr_access_audits, selected_isr_files, allowed_isr_dates)
        _validate_isr_source_identities(isr_files, selected_isr_files)
        if preflight_dependency is not None:
            if isr_files != preflight_dependency["isr_source_files"]:
                raise ValueError(
                    "full-audit ISR source identities differ from preflight")
            if isr_column_access_audit != preflight_dependency[
                    "ISR_column_access_audit"]:
                raise ValueError(
                    "full-audit ISR column access differs from preflight")
        selections, raw_summary = _rebuild_raw_selections(
            records, start_date, partitions, expected_query_counts)
        del records

        # Both preflight and the full audit must materialize the real compact
        # train-only exact-token directories.  This makes loader identity and
        # memory feasibility part of preflight instead of a deferred claim.
        sw_manager, iri_peak_manager, indexes = _load_runtime_managers(
            config, device, allowlists)
        index_preflight = _validate_loaded_train_only_indexes(
            indexes, allowlists)
        train_token_directory_identity = build_train_token_directory_identity(
            indexes)
        if (preflight_dependency is not None
                and train_token_directory_identity != preflight_dependency[
                    "train_token_directory_identity"]):
            raise ValueError(
                "full-audit train token directory differs from preflight")

        source_provenance = _source_provenance(
            contract_path, p0a_contract_path)
        runtime_contract_path = _write_runtime_contract(
            output_dir, contract_path, frozen_contract_sha,
            checkpoint, checkpoint_sha_before, p0a, date_split_path,
            input_data_sha256, train_allowlist_identity,
            train_token_directory_identity,
            isr_files, isr_column_access_audit,
            frozen_contract["identity"]["runtime_required_identities"],
            profile_metadata, source_provenance, git_provenance,
            python_environment, coordinate_status, preflight_dependency)
        manifest = _initial_manifest(
            args, output_dir, raw_summary, profile_metadata,
            checkpoint_sha_before, model_state_sha_before)
        manifest["input_data_sha256"] = input_data_sha256
        manifest["train_allowlist_identity"] = train_allowlist_identity
        manifest["train_token_directory_identity"] = (
            train_token_directory_identity)
        manifest["python_environment"] = python_environment
        manifest["git"] = git_provenance
        manifest["date_split_sha256"] = expected_date_split_sha
        manifest["isr_source_files"] = isr_files
        manifest["ISR_column_access_audit"] = isr_column_access_audit
        manifest["satellite_index_preflight"] = index_preflight
        manifest["coordinate_enrichment"] = coordinate_status
        manifest["profile_cap_status"] = "not_applicable_in_v14"
        manifest["preflight_dependency"] = (
            dict(preflight_dependency) if preflight_dependency is not None else {
                "required": False,
                "reason": "this runtime is the preflight itself",
            })
        manifest_path = atomic_write_json(output_dir / "manifest.json", manifest)
        failure_path = atomic_write_json(output_dir / "failure_ledger.json", {
            "audit_schema_version": P0B_AUDIT_SCHEMA_VERSION,
            "status": "no_failures",
            "entries": [],
        })

        batch_size = int(frozen_contract["partitioning"]["query_batch_size"])
        if args.batch_size is not None and int(args.batch_size) != batch_size:
            raise ValueError(
                f"batch size is frozen at {batch_size}, got {args.batch_size}")
        allowed_selections = [
            item for item in selections
            if item.query_split in ("train", "development")
        ]
        if not allowed_selections:
            raise ValueError("P0-B has no allowed ISR selection")
        if any(item.query_split == "locked_test" for item in allowed_selections):
            raise ValueError("locked-test ISR selection reached inference")

        if args.preflight_only:
            preflight_summary = _run_preflight_inference_suite(
                model, sw_manager, iri_peak_manager, indexes, allowlists,
                frozen_contract, allowed_selections, device)
            manifest["inference_preflight"] = preflight_summary
            token_directory_after = build_train_token_directory_identity(indexes)
            if token_directory_after != train_token_directory_identity:
                raise ValueError(
                    "train token directory changed during preflight inference")
            checkpoint_after = sha256_file(checkpoint)
            model_state_after = _model_state_sha256(model)
            if checkpoint_after != checkpoint_sha_before:
                raise ValueError("checkpoint SHA256 changed during preflight")
            if model_state_after != model_state_sha_before:
                raise ValueError("model state changed during preflight")
            git_after = _assert_git_provenance_stable(
                git_provenance, args.implementation_ref)
            manifest.update({
                "status": "preflight_pass",
                "completed_utc": _utc_now(),
                "checkpoint_sha256_after": checkpoint_after,
                "model_state_sha256_after": model_state_after,
                "git_after": git_after,
            })
            manifest_path = atomic_write_json(output_dir / "manifest.json", manifest)
            write_completion_marker_atomically(
                output_dir / "preflight_acceptance.json",
                {
                    "status": "preflight_pass",
                    "full_audit_complete": False,
                    "read_only_model_inference_executed": True,
                    "stratified_probe_count": preflight_summary["probe_count"],
                    "implementation_tag_object_sha": git_after[
                        "implementation_tag_object_sha"],
                    "git_head": git_after["head"],
                    "candidate_checkpoint_sha256": checkpoint_after,
                    "date_split_sha256": expected_date_split_sha,
                    "p0a_contract_sha256": p0a.sha256,
                    "p0b_contract_sha256": frozen_contract_sha,
                    "python_environment": python_environment,
                    "input_data_sha256": input_data_sha256,
                    "train_allowlist_identity": train_allowlist_identity,
                    "train_token_directory_identity": (
                        train_token_directory_identity),
                    "query_partition_counts": dict(
                        raw_summary["date_partition_counts"]),
                    "satellite_profile_counts": profile_count_ledger,
                    "numeric_predictions_persisted": False,
                    "npz_written": False,
                    "locked_test_query_rows_written": 0,
                    "satellite_token_partition": "train",
                    "coordinate_enrichment": coordinate_status,
                    "ISR_column_access_audit": isr_column_access_audit,
                    "profile_cap_status": "not_applicable_in_v14",
                },
                [runtime_contract_path, manifest_path, failure_path],
                artifact_root=output_dir,
            )
            return 0

        next_token_row_id = 0
        artifacts: list[Path] = [runtime_contract_path]
        written_paths: set[Path] = set()
        for selection, batch_id, sl in _iter_scope_batches(
                allowed_selections, batch_size):
            query, token, edge, next_token_row_id, errors = _infer_batch(
                model, sw_manager, iri_peak_manager, indexes, allowlists,
                frozen_contract, selection, sl, batch_id, device,
                next_token_row_id)
            directory = (
                output_dir / "batches" / selection.station
                / selection.date_utc)
            paths = {
                "query": directory / f"batch_{batch_id:06d}_query.npz",
                "token": directory / f"batch_{batch_id:06d}_token.npz",
                "edge": directory / f"batch_{batch_id:06d}_edge.npz",
            }
            if any(path in written_paths or path.exists()
                   for path in paths.values()):
                raise FileExistsError(
                    "P0-B shard path collision within a station/date scope")
            tables = {"query": query, "token": token, "edge": edge}
            for name, table in tables.items():
                _atomic_savez(paths[name], table)
                written_paths.add(paths[name])
            persisted = {
                name: _load_npz_table(path) for name, path in paths.items()}
            post_validation = _validate_batch_against_contract(
                frozen_contract, persisted["query"], persisted["token"],
                persisted["edge"])
            for path in paths.values():
                artifacts.append(path)
                manifest["artifacts"].append(
                    artifact_identity(path, root=output_dir))
            manifest["counts"]["query_rows"] += int(len(query["query_id"]))
            manifest["counts"]["token_rows"] += int(len(token["token_row_id"]))
            manifest["counts"]["edge_rows"] += int(len(edge["query_id"]))
            manifest["counts"]["batches"] += 1
            post_summary_error = float(
                post_validation["recomputed_diagnostics"][
                    "max_abs_recomputed_summary_error"])
            for key, value in (
                    ("query_closure", errors["query_closure_max_abs_error"]),
                    ("edge_closure", errors["edge_closure_max_abs_error"]),
                    ("recomputed_summary", max(
                        errors["recomputed_summary_max_abs_error"],
                        post_summary_error))):
                manifest["max_abs_errors"][key] = max(
                    manifest["max_abs_errors"][key], float(value))
            manifest["last_completed_partition"] = {
                "station": selection.station,
                "date_utc": selection.date_utc,
                "batch_id": batch_id,
            }
            atomic_write_json(output_dir / "manifest.json", manifest)

        expected_queries = int(raw_summary["allowed_query_count"])
        if manifest["counts"]["query_rows"] != expected_queries:
            raise ValueError(
                "written query count differs from train+development registry count")
        token_directory_after = build_train_token_directory_identity(indexes)
        if token_directory_after != train_token_directory_identity:
            raise ValueError("train token directory changed during full audit")
        checkpoint_after = sha256_file(checkpoint)
        model_state_after = _model_state_sha256(model)
        if checkpoint_after != checkpoint_sha_before:
            raise ValueError("checkpoint SHA256 changed during audit")
        if model_state_after != model_state_sha_before:
            raise ValueError("model state_dict changed during read-only audit")
        git_after = _assert_git_provenance_stable(
            git_provenance, args.implementation_ref)
        preflight_after = _validate_preflight_acceptance(
            Path(args.preflight_acceptance), output_dir=output_dir,
            checkpoint=checkpoint,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            frozen_contract=frozen_contract,
            frozen_contract_path=contract_path,
            frozen_contract_sha256=frozen_contract_sha,
            p0a_contract_path=p0a_contract_path,
            git_provenance=git_after,
            python_environment=_python_environment(),
            coordinate_status=coordinate_status)
        if preflight_after != preflight_dependency:
            raise ValueError("preflight acceptance identity changed during full audit")
        manifest.update({
            "status": "cache_complete_provisional_not_final_p0b_acceptance",
            "completed_utc": _utc_now(),
            "checkpoint_sha256_after": checkpoint_after,
            "model_state_sha256_after": model_state_after,
            "git_after": git_after,
            "preflight_dependency_after": preflight_after,
            "locked_test_query_rows_written": 0,
            "satellite_token_partition": "train",
        })
        manifest_path = atomic_write_json(output_dir / "manifest.json", manifest)
        artifacts.extend((manifest_path, failure_path))
        write_completion_marker_atomically(
            output_dir / P0B_CACHE_COMPLETION_MARKER,
            {
                "status": "cache_pass_not_final_p0b_acceptance",
                "full_p0b_audit_complete": False,
                "final_p0b_acceptance_written": False,
                "fixed_summary_artifact_written": False,
                "six_question_artifact_written": False,
                "Q6_peak_persistence": "insufficient_evidence",
                "provisional_until_p0a_giro_acceptance": True,
                "counts": manifest["counts"],
                "max_abs_errors": manifest["max_abs_errors"],
                "checkpoint_sha256_before": checkpoint_sha_before,
                "checkpoint_sha256_after": checkpoint_after,
                "model_state_sha256_before": model_state_sha_before,
                "model_state_sha256_after": model_state_after,
                "implementation_tag_object_sha": git_after[
                    "implementation_tag_object_sha"],
                "preflight_dependency": preflight_after,
                "input_data_sha256": input_data_sha256,
                "train_allowlist_identity": train_allowlist_identity,
                "train_token_directory_identity": (
                    train_token_directory_identity),
                "query_partition_counts": dict(
                    raw_summary["date_partition_counts"]),
                "satellite_profile_counts": profile_count_ledger,
                "locked_test_query_rows_written": 0,
                "satellite_token_partition": "train",
                "coordinate_enrichment": coordinate_status,
                "ISR_column_access_audit": isr_column_access_audit,
                "profile_cap_status": "not_applicable_in_v14",
            },
            artifacts,
            artifact_root=output_dir,
        )
        return 0
    except BaseException as exc:
        _write_failure_ledger(output_dir, exc, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen read-only M2-W2 P0-B ISR error-chain audit. "
            "The output directory must not already exist."))
    parser.add_argument("--checkpoint", required=True,
                        help="Strict v14 candidate Analysis checkpoint")
    parser.add_argument(
        "--expected-checkpoint-sha256", required=True,
        help="Required explicit SHA256 for the candidate checkpoint")
    parser.add_argument("--p0b-contract", default=str(DEFAULT_P0B_CONTRACT),
                        help="Frozen P0-B audit contract JSON")
    parser.add_argument(
        "--implementation-ref", default=EXPECTED_IMPLEMENTATION_TAG,
        help=argparse.SUPPRESS)
    parser.add_argument("--p0a-dir", default=str(DEFAULT_P0A_DIRECTORY),
                        help="P0-A qav2-r2 directory (JSON contract only)")
    parser.add_argument("--p0a-contract", default=None,
                        help="Explicit P0-A contract JSON (defaults inside --p0a-dir)")
    parser.add_argument("--output-dir", required=True,
                        help="New P0-B output directory")
    parser.add_argument(
        "--jicamarca-dir",
        default=r"D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)")
    parser.add_argument(
        "--poker-flat-dir",
        default=r"D:\ISR\DATA\61poker_flat_is_radar(lp)\05min")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Must equal the frozen contract value when supplied")
    parser.add_argument(
        "--preflight-only", action="store_true",
        help=(
            "Run the frozen stratified read-only model-inference preflight "
            "across both stations, train/development, and all altitude bands; "
            "no prediction arrays or NPZ shards are persisted"))
    parser.add_argument(
        "--preflight-acceptance", default=None,
        help=(
            "Required for a full audit: the completed preflight_acceptance.json "
            "whose Git, contracts, checkpoint, counts, coordinates, and Python "
            "environment must exactly match this run"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.expected_checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in
            args.expected_checkpoint_sha256.lower()):
        raise ValueError("--expected-checkpoint-sha256 must be 64 hexadecimal characters")
    args.expected_checkpoint_sha256 = args.expected_checkpoint_sha256.lower()
    if args.preflight_only and args.preflight_acceptance is not None:
        raise ValueError("--preflight-acceptance is not valid with --preflight-only")
    if not args.preflight_only and args.preflight_acceptance is None:
        raise ValueError("full P0-B audit requires --preflight-acceptance")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
