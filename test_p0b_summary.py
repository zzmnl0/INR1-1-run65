"""Synthetic integrity and attribution tests for the fixed P0-B summarizer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from inr_modules.mdia import p0b_audit
from isr_evaluation import summarize_m2w2_error_chain as summary


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "m2w2_contracts" / "p0b_audit_contract_v1.json"


def _publication_context_for_inventory(inventory):
    dependency = inventory.identity_validation["preflight_dependency"]
    return {
        "git": dict(inventory.runtime_contract["git"]),
        "python_environment": dict(
            inventory.identity_validation["python_environment"]),
        "source_provenance": dict(
            inventory.identity_validation["source_provenance"]),
        "preflight_acceptance": {
            "path": str(Path(dependency["path"]).resolve()),
            "sha256": dependency["sha256"],
            "size_bytes": dependency["size_bytes"],
        },
    }


@pytest.fixture(autouse=True)
def _synthetic_publication_context(monkeypatch):
    """Keep publication tests independent of the developer's current Git state."""
    def capture(inventory):
        return _publication_context_for_inventory(inventory)

    monkeypatch.setattr(summary, "_capture_current_publication_context", capture)


def _zeros(schema, count):
    return {
        name: np.zeros(count, dtype=dtype)
        for name, dtype in schema["dtypes"].items()
    }


def _synthetic_tables(contract, date_utc, split, batch_id, first_id, count=16):
    schemas = p0b_audit.contract_table_schemas(contract)
    query = _zeros(schemas["query"], count)
    token = _zeros(schemas["token"], count)
    edge = _zeros(schemas["edge"], count)
    query_ids = np.arange(first_id, first_id + count, dtype=np.int64)
    token_ids = np.arange(10_000 + first_id, 10_000 + first_id + count,
                          dtype=np.int64)
    profiles = np.arange(20_000 + first_id, 20_000 + first_id + count,
                         dtype=np.int64)
    timestamps = np.arange(
        1_725_235_200 + first_id * 300,
        1_725_235_200 + (first_id + count) * 300,
        300, dtype=np.int64)

    query.update({
        "query_id": query_ids,
        "sample_key": np.asarray(
            [f"Jicamarca|{date_utc}|{value}|150.000" for value in timestamps],
            dtype="U64"),
        "station": np.full(count, "Jicamarca", dtype="U16"),
        "date_utc": np.full(count, date_utc, dtype="U8"),
        "batch_id": np.full(count, batch_id, dtype=np.int64),
        "query_profile_id": query_ids.copy(),
        "timestamp_unix": timestamps,
        "query_split": np.full(count, split, dtype="U11"),
        "query_profile_id": query_ids.copy(),
        "latitude_deg": np.full(count, -11.95, dtype=np.float32),
        "longitude_deg": np.full(count, -76.87, dtype=np.float32),
        "altitude_km": np.full(count, 150.0, dtype=np.float32),
        "relative_hour": np.arange(count, dtype=np.float32),
        "local_time_hour": np.full(count, 1.0, dtype=np.float32),
        "aacgm_latitude_deg": np.full(count, -1.0, dtype=np.float32),
        "aacgm_mlt_hour": np.full(count, 1.0, dtype=np.float32),
        "cos_sza": np.full(count, -0.5, dtype=np.float32),
        "kp": np.full(count, 1.0, dtype=np.float32),
        "f107": np.full(count, 100.0, dtype=np.float32),
        "isr_log10_ne": np.full(count, 10.0, dtype=np.float32),
        "raw_iri_log10_ne": np.full(count, 10.0, dtype=np.float32),
        "M00_log10_ne": np.full(count, 10.0, dtype=np.float32),
        "M10_log10_ne": np.full(count, 10.08, dtype=np.float32),
        "M01_log10_ne": np.full(count, 10.0, dtype=np.float32),
        "M11_log10_ne": np.full(count, 10.08, dtype=np.float32),
        "no_token_log10_ne": np.full(count, 10.0, dtype=np.float32),
        "isolated_increment_FY_dex": np.full(count, 0.08, dtype=np.float32),
        "isolated_increment_COSMIC_dex": np.zeros(count, dtype=np.float32),
        "joint_update_FY_dex": np.full(count, 0.08, dtype=np.float32),
        "joint_update_COSMIC_dex": np.zeros(count, dtype=np.float32),
        "joint_increment_dex": np.full(count, 0.08, dtype=np.float32),
        "raw_coverage_code": np.full(count, "FY_only", dtype="U12"),
        "coverage_code": np.full(count, "FY_only", dtype="U12"),
        "FY_token_count": np.ones(count, dtype=np.int64),
        "FY_unique_profile_count": np.ones(count, dtype=np.int64),
        "FY_unlocalized_precision_sum": np.full(count, 4.0, dtype=np.float32),
        "FY_localized_precision_sum": np.full(count, 4.0, dtype=np.float32),
        "FY_token_neff": np.ones(count, dtype=np.float32),
        "FY_profile_neff": np.ones(count, dtype=np.float32),
        "FY_max_profile_precision_share": np.ones(count, dtype=np.float32),
        "predictive_nis_unlocalized_FY": np.full(count, 0.1, dtype=np.float32),
        "predictive_nis_unlocalized_FY_dof": np.ones(count, dtype=np.int64),
        "COSMIC_token_count": np.zeros(count, dtype=np.int64),
        "COSMIC_unique_profile_count": np.zeros(count, dtype=np.int64),
        "COSMIC_unlocalized_precision_sum": np.zeros(count, dtype=np.float32),
        "COSMIC_localized_precision_sum": np.zeros(count, dtype=np.float32),
        "COSMIC_token_neff": np.zeros(count, dtype=np.float32),
        "COSMIC_profile_neff": np.zeros(count, dtype=np.float32),
        "COSMIC_max_profile_precision_share": np.zeros(count, dtype=np.float32),
        "predictive_nis_unlocalized_COSMIC": np.zeros(count, dtype=np.float32),
        "predictive_nis_unlocalized_COSMIC_dof": np.zeros(count, dtype=np.int64),
        "predictive_nis_unlocalized_joint": np.full(count, 0.1, dtype=np.float32),
        "predictive_nis_unlocalized_joint_dof": np.ones(count, dtype=np.int64),
        "CF_drop_200_250_log10_ne": np.full(count, 10.11, dtype=np.float32),
        "CF_drop_250_300_log10_ne": np.full(count, 10.08, dtype=np.float32),
        "CF_drop_300_400_log10_ne": np.full(count, 10.08, dtype=np.float32),
        "CF_drop_400_500_log10_ne": np.full(count, 10.08, dtype=np.float32),
        "CF_duplicate_FY_dominant_profile_log10_ne": np.full(
            count, 10.11, dtype=np.float32),
        "CF_duplicate_COSMIC_dominant_profile_log10_ne": np.full(
            count, 10.08, dtype=np.float32),
        "CF_duplicate_both_dominant_profiles_log10_ne": np.full(
            count, 10.11, dtype=np.float32),
        "FY_dominant_profile_id": profiles.copy(),
        "COSMIC_dominant_profile_id": np.full(count, -1, dtype=np.int64),
        "FY_dominant_profile_valid": np.ones(count, dtype=np.bool_),
        "COSMIC_dominant_profile_valid": np.zeros(count, dtype=np.bool_),
    })

    token.update({
        "token_row_id": token_ids,
        "station": np.full(count, "Jicamarca", dtype="U16"),
        "date_utc": np.full(count, date_utc, dtype="U8"),
        "batch_id": np.full(count, batch_id, dtype=np.int64),
        "source": np.full(count, "FY", dtype="U6"),
        "profile_id": profiles,
        "token_id": np.zeros(count, dtype=np.int64),
        "profile_split": np.full(count, "train", dtype="U5"),
        "latitude_deg": np.full(count, -12.0, dtype=np.float32),
        "longitude_deg": np.full(count, -77.0, dtype=np.float32),
        "altitude_km": np.full(count, 225.0, dtype=np.float32),
        "relative_hour": np.arange(count, dtype=np.float32),
        "observation_log10_ne": np.full(count, 10.2, dtype=np.float32),
        "background_log10_ne": np.full(count, 10.0, dtype=np.float32),
    })

    edge.update({
        "station": np.full(count, "Jicamarca", dtype="U16"),
        "date_utc": np.full(count, date_utc, dtype="U8"),
        "batch_id": np.full(count, batch_id, dtype=np.int64),
        "query_id": query_ids,
        "token_row_id": token_ids,
        "source": np.full(count, "FY", dtype="U6"),
        "innovation_dex": np.full(count, 0.2, dtype=np.float32),
        "r_variance_dex2": np.full(count, 0.25, dtype=np.float32),
        "representativeness_weight": np.ones(count, dtype=np.float32),
        "localization_weight": np.ones(count, dtype=np.float32),
        "unlocalized_precision": np.full(count, 4.0, dtype=np.float32),
        "localized_precision": np.full(count, 4.0, dtype=np.float32),
        "prior_predictive_variance_dex2": np.full(
            count, 0.1, dtype=np.float32),
        "r_standardized_innovation_sq": np.full(
            count, 0.16, dtype=np.float32),
        "predictive_diagonal_nis": np.full(
            count, 0.114285715, dtype=np.float32),
        "localized_innovation_energy": np.full(
            count, 0.16, dtype=np.float32),
        "gain_joint": np.full(count, 0.4, dtype=np.float32),
        "gain_isolated": np.full(count, 0.4, dtype=np.float32),
        "contribution_joint_dex": np.full(count, 0.08, dtype=np.float32),
        "contribution_isolated_dex": np.full(count, 0.08, dtype=np.float32),
        "space_distance_km": np.full(count, 100.0, dtype=np.float32),
        "time_distance_hours": np.full(count, 1.0, dtype=np.float32),
    })
    return query, token, edge


def _write_json(path, payload):
    path.write_text(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")


def _rewrite_cache_marker(cache_dir, artifact_paths, payload):
    marker = cache_dir / summary.CACHE_MARKER
    if marker.exists():
        marker.unlink()
    p0b_audit.write_completion_marker_atomically(
        marker, payload, artifact_paths, artifact_root=cache_dir)


def _refresh_shard_identity(cache_dir, relative_path, artifact_paths, marker_payload):
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = p0b_audit.artifact_identity(
        cache_dir / relative_path, root=cache_dir)
    manifest["artifacts"] = [
        identity if row["path"] == relative_path else row
        for row in manifest["artifacts"]
    ]
    _write_json(manifest_path, manifest)
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)


def _make_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    contract, contract_sha = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    head = "a" * 40
    tag_object_sha = "e" * 40
    git = {
        "status": "computed",
        "head": head,
        "branch": contract["version_control"]["required_branch"],
        "tracked_status": [],
        "expected_branch": contract["version_control"]["required_branch"],
        "base_anchor": contract["version_control"]["base_anchor"]["git_commit"],
        "base_anchor_is_ancestor": True,
        "required_implementation_tag": contract["version_control"][
            "required_implementation_tag"],
        "implementation_tag_type": "tag",
        "implementation_tag_commit": head,
        "implementation_tag_object_sha": tag_object_sha,
        "head_equals_implementation_tag_commit": True,
        "critical_tracked_paths": contract["version_control"][
            "critical_tracked_paths"],
        "untracked_critical_paths": [],
        "critical_paths_clean_against_implementation_commit": True,
    }
    input_identity = {
        "date_split_manifest": contract["identity"]["date_split_sha256"],
        "synthetic_input": "c" * 64,
    }
    coordinate_status = {
        "geographic": {
            "status": "computed",
            "fields": list(contract["coordinate_enrichment"]["geographic"][
                "fields"]),
        },
        "aacgm": {
            "status": "computed",
            "package": "aacgmv2",
            "version": contract["coordinate_enrichment"]["aacgm"]["version"],
            "method": "ALLOWTRACE",
            "fields": list(contract["coordinate_enrichment"]["aacgm"]["fields"]),
        },
        "qd": {
            "status": "unavailable",
            "package": "apexpy",
            "reason": contract["coordinate_enrichment"]["qd"]["reason"],
            "proxy_substitution_used": False,
        },
        "fixed_aggregation_time_coordinate": "aacgm_mlt_hour",
    }
    runtime_profile_cap = contract[
        "precision_innovation_gain_semantics"]["profile_cap_status"]
    python_environment = {
        "executable": str(p0b_audit.P0B_PYTHON_EXECUTABLE.resolve()),
        "version": "3.synthetic",
        "implementation": "CPython",
        "numpy_version": "synthetic-numpy",
        "torch_version": "synthetic-torch",
        "torch_cuda_version": None,
    }
    column_access = {
        "isr_column_access_audit_schema_version": 1,
        "status": "pass",
        "excluded_density_columns_materialized": 0,
    }
    satellite_profile_metadata = {
        source: {
            "token_partition": "train",
            "development_metadata_only": True,
            "train_unique_profiles": 10,
            "development_unique_profiles": 5,
            "train_profile_id_sha256": character * 64,
            "profile_index_sha256": (
                "c" if source == "FY" else "d") * 64,
        }
        for source, character in zip(p0b_audit.P0B_SOURCES, ("a", "b"))
    }
    train_allowlist_identity = {
        source: {
            "profile_count": metadata["train_unique_profiles"],
            "profile_id_sha256": metadata["train_profile_id_sha256"],
            "profile_index_sha256": metadata["profile_index_sha256"],
        }
        for source, metadata in satellite_profile_metadata.items()
    }
    train_token_directory_identity = {}
    for index, source in enumerate(p0b_audit.P0B_SOURCES):
        token_rows = 100 + index
        train_token_directory_identity[source] = {
            "semantics": "exact_train_only_compact_token_arrays_v1",
            "sha256": str(index + 1) * 64,
            "token_rows": token_rows,
            "unique_profiles": 10,
            "arrays": {
                "token_coords": {"dtype": "<f4", "shape": [token_rows, 4]},
                "token_values": {"dtype": "<f4", "shape": [token_rows]},
                "token_profile_ids": {"dtype": "<i8", "shape": [token_rows]},
                "token_ids": {"dtype": "<i8", "shape": [token_rows]},
            },
        }
    preflight_path = tmp_path / "preflight_acceptance.json"
    _write_json(preflight_path, {"status": "preflight_pass"})
    preflight_identity = p0b_audit.artifact_identity(preflight_path)
    preflight_dependency = {
        "path": str(preflight_path.resolve()),
        "sha256": preflight_identity["sha256"],
        "size_bytes": preflight_identity["size_bytes"],
        "git_head": head,
        "implementation_tag_object_sha": tag_object_sha,
        "train_allowlist_identity": deepcopy(train_allowlist_identity),
        "train_token_directory_identity": deepcopy(train_token_directory_identity),
        "ISR_column_access_audit": deepcopy(column_access),
    }
    p0a_path = tmp_path / "synthetic_p0a_contract.json"
    _write_json(p0a_path, {"status": "synthetic"})
    source_paths = (
        p0a_path.resolve(), CONTRACT_PATH.resolve(),
        *(ROOT / relative for relative in contract[
            "version_control"]["critical_tracked_paths"]),
    )
    source_provenance = {
        str(path.resolve()): {"sha256": "d" * 64, "size_bytes": 1}
        for path in source_paths
    }
    runtime = {
        "audit_schema_version": 1,
        "status": "runtime_contract_bound",
        "frozen_p0b_contract": {
            "path": str(CONTRACT_PATH.resolve()),
            "sha256": contract_sha,
        },
        "candidate_checkpoint": {
            "sha256": contract["identity"]["candidate_checkpoint_sha256"],
        },
        "date_split": {"sha256": contract["identity"]["date_split_sha256"]},
        "p0a_dependency": {"contract_path": str(p0a_path.resolve())},
        "source_provenance": source_provenance,
        "git": git,
        "python_environment": python_environment,
        "coordinate_enrichment": coordinate_status,
        "ISR_column_access_audit": column_access,
        "train_allowlist_identity": train_allowlist_identity,
        "train_token_directory_identity": train_token_directory_identity,
        "profile_cap_status": runtime_profile_cap,
        "satellite_profile_metadata": satellite_profile_metadata,
        "preflight_dependency": preflight_dependency,
        "runtime_identities": {
            "audit_code_sha256": "d" * 64,
            "git_head": head,
            "python_environment": python_environment,
            "checkpoint_path": {
                "path": "synthetic.pth",
                "sha256": contract["identity"]["candidate_checkpoint_sha256"],
            },
            "date_split_path": {
                "path": "split.json",
                "sha256": contract["identity"]["date_split_sha256"],
            },
            "input_data_sha256": input_identity,
            "train_allowlist_identity": train_allowlist_identity,
            "train_token_directory_identity": train_token_directory_identity,
            "ISR_source_file_sha256": [],
            "ISR_column_access_audit": column_access,
            "coordinate_enrichment": coordinate_status,
            "profile_cap_status": runtime_profile_cap,
        },
    }
    runtime_path = cache_dir / "audit_contract.json"
    failure_path = cache_dir / "failure_ledger.json"
    _write_json(runtime_path, runtime)
    _write_json(failure_path, {
        "audit_schema_version": 1, "status": "no_failures", "entries": []})

    batch_artifacts = []
    dates = (
        ("20240901", "train"), ("20240902", "train"),
        ("20240903", "development"), ("20240904", "development"),
    )
    total_counts = {"query_rows": 0, "token_rows": 0,
                    "edge_rows": 0, "batches": 0}
    for batch_id, (date_utc, split) in enumerate(dates):
        query, token, edge = _synthetic_tables(
            contract, date_utc, split, batch_id, first_id=batch_id * 100)
        directory = cache_dir / "batches" / "Jicamarca" / date_utc
        directory.mkdir(parents=True)
        for table_name, table in (
                ("query", query), ("token", token), ("edge", edge)):
            path = directory / f"batch_{batch_id:06d}_{table_name}.npz"
            np.savez_compressed(path, **table)
            batch_artifacts.append(path)
            total_counts[f"{table_name}_rows"] += len(next(iter(table.values())))
        total_counts["batches"] += 1

    model_sha = "b" * 64
    query_partition_counts = {"train": 32, "development": 32}
    satellite_profile_counts = {
        "train": {source: 10 for source in p0b_audit.P0B_SOURCES},
        "development": {source: 5 for source in p0b_audit.P0B_SOURCES},
    }
    manifest = {
        "audit_schema_version": 1,
        "status": "cache_complete_provisional_not_final_p0b_acceptance",
        "preflight_only": False,
        "checkpoint_sha256_before": contract["identity"][
            "candidate_checkpoint_sha256"],
        "checkpoint_sha256_after": contract["identity"][
            "candidate_checkpoint_sha256"],
        "model_state_sha256_before": model_sha,
        "model_state_sha256_after": model_sha,
        "counts": total_counts,
        "max_abs_errors": {
            "query_closure": 0.0, "edge_closure": 0.0,
            "recomputed_summary": 0.0,
        },
        "artifacts": [
            p0b_audit.artifact_identity(path, root=cache_dir)
            for path in batch_artifacts
        ],
        "input_data_sha256": input_identity,
        "python_environment": python_environment,
        "git": git,
        "git_after": git,
        "date_split_sha256": contract["identity"]["date_split_sha256"],
        "locked_test_query_rows_written": 0,
        "satellite_token_partition": "train",
        "coordinate_enrichment": coordinate_status,
        "ISR_column_access_audit": column_access,
        "train_allowlist_identity": train_allowlist_identity,
        "train_token_directory_identity": train_token_directory_identity,
        "profile_cap_status": runtime_profile_cap,
        "preflight_dependency": preflight_dependency,
        "preflight_dependency_after": preflight_dependency,
        "raw_registry": {"date_partition_counts": query_partition_counts},
        "satellite_profile_metadata": satellite_profile_metadata,
    }
    manifest_path = cache_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    marker_payload = {
        "status": "cache_pass_not_final_p0b_acceptance",
        "full_p0b_audit_complete": False,
        "final_p0b_acceptance_written": False,
        "fixed_summary_artifact_written": False,
        "six_question_artifact_written": False,
        "Q6_peak_persistence": "insufficient_evidence",
        "provisional_until_p0a_giro_acceptance": True,
        "counts": total_counts,
        "max_abs_errors": manifest["max_abs_errors"],
        "checkpoint_sha256_before": contract["identity"][
            "candidate_checkpoint_sha256"],
        "checkpoint_sha256_after": contract["identity"][
            "candidate_checkpoint_sha256"],
        "model_state_sha256_before": model_sha,
        "model_state_sha256_after": model_sha,
        "locked_test_query_rows_written": 0,
        "satellite_token_partition": "train",
        "coordinate_enrichment": coordinate_status,
        "ISR_column_access_audit": column_access,
        "train_allowlist_identity": train_allowlist_identity,
        "train_token_directory_identity": train_token_directory_identity,
        "profile_cap_status": runtime_profile_cap,
        "preflight_dependency": preflight_dependency,
        "implementation_tag_object_sha": tag_object_sha,
        "query_partition_counts": query_partition_counts,
        "satellite_profile_counts": satellite_profile_counts,
    }
    artifact_paths = [runtime_path, *batch_artifacts, manifest_path, failure_path]
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    return cache_dir, contract, artifact_paths, marker_payload


def _question_statuses(cache_dir):
    payload = json.loads((cache_dir / "p0b_six_questions.json").read_text(
        encoding="utf-8"))
    return {row["id"]: row["status"] for row in payload["questions"]}


def test_nonnegative_profile_metrics_transform_before_profile_reduction():
    acc = summary.GroupAccumulator()
    profile_key = ("Jicamarca", "20240901", 1)
    key = (
        "Jicamarca", "train", "00-03", "night", "[200,250)km",
        "[0,2)", "joint")
    for value in (-0.10, 0.10):
        acc.add_value("joint_increment", value, profile_key)
        acc.add_value("joint_increment_abs", abs(value), profile_key)
    row = summary._joint_row(key, acc)
    assert row["joint_increment_median_dex"] == pytest.approx(0.0)
    assert row["joint_increment_abs_median_dex"] == pytest.approx(0.10)


def test_source_totals_preserve_raw_queries_and_label_profile_reduction():
    contract, _ = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    rules = summary._edge_diagnostic_rules(contract)
    acc = summary.GroupAccumulator()
    profile_a = ("Jicamarca", "20240901", 1)
    profile_b = ("Jicamarca", "20240901", 2)
    for token_count, nis, dof, innovation in (
            (1.0, 1.0, 1.0, 0.1),
            (9.0, 2.0, 1.0, 0.2),
            (11.0, 3.0, 1.0, 0.3)):
        acc.add_value("token_count", token_count, profile_a)
        acc.add_value("predictive_nis", nis, profile_a)
        acc.add_value("predictive_nis_dof", dof, profile_a)
        acc.add_value("precision_weighted_innovation", innovation, profile_a)
    acc.add_value("token_count", 3.0, profile_b)
    acc.add_value("predictive_nis", 4.0, profile_b)
    acc.add_value("predictive_nis_dof", 2.0, profile_b)
    acc.add_value("precision_weighted_innovation", 0.4, profile_b)

    key = (
        "Jicamarca", "train", "00-03", "night", "[200,250)km",
        "[0,2)", "joint")
    row = summary._source_row(key, "FY", acc, {}, rules)

    assert row["raw_query_token_count_sum"] == 24
    assert row[
        "profile_equal_token_count_sum_of_within_profile_medians"] == 12
    assert row["profile_equal_token_count_median"] == pytest.approx(6.0)
    assert row["raw_query_predictive_nis_sum"] == pytest.approx(10.0)
    assert row["raw_query_predictive_nis_dof_sum"] == 5
    assert row["raw_query_predictive_nis_per_dof"] == pytest.approx(2.0)
    assert row[
        "profile_equal_predictive_nis_sum_of_within_profile_medians"
    ] == pytest.approx(6.0)
    assert row[
        "profile_equal_predictive_nis_dof_sum_of_within_profile_medians"
    ] == 3
    assert row["profile_equal_predictive_nis_per_dof_mean"] == pytest.approx(2.0)
    assert row[
        "raw_query_precision_weighted_innovation_defined_queries"] == 4
    assert row[
        "profile_equal_precision_weighted_innovation_defined_profiles"] == 2
    assert row[
        "profile_equal_precision_weighted_innovation_median_dex"
    ] == pytest.approx(0.3)
    assert not {
        "token_count_sum", "predictive_nis_sum", "predictive_nis_dof_sum",
        "precision_weighted_innovation_defined_queries",
    }.intersection(row)


def test_synthetic_cache_fixed_summaries_and_six_questions(tmp_path):
    cache_dir, contract, _, _ = _make_cache(tmp_path)
    marker = summary.summarize_cache(cache_dir)
    assert marker.name == summary.FINAL_MARKER
    acceptance = json.loads(marker.read_text(encoding="utf-8"))
    assert acceptance["status"] == "provisional_pass"
    assert acceptance["attribution_complete"] is False
    assert acceptance["scientific_attribution_complete"] is False
    assert acceptance["six_questions_assigned"] is True
    assert acceptance["profile_cap_status"] == contract[
        "precision_innovation_gain_semantics"]["profile_cap_status"]
    assert acceptance["p0c_unlocked"] is False
    expected_artifacts = contract["fixed_aggregation"]["summary_artifacts"][
        "final_acceptance_artifacts_in_order"]
    assert [row["path"] for row in acceptance["artifacts"]] == expected_artifacts
    manifest = json.loads((cache_dir / summary.SUMMARY_MANIFEST).read_text(
        encoding="utf-8"))
    assert [row["path"] for row in manifest["summary_artifacts"]] == list(
        summary.SUMMARY_FILES)
    assert summary.SUMMARY_MANIFEST not in {
        row["path"] for row in manifest["summary_artifacts"]}
    assert acceptance["publication_identity"] == manifest["publication_identity"]
    assert acceptance["publication_identity"]["status"] == "pass"
    assert acceptance["publication_identity"][
        "ISR_column_access_audit"]["status"] == "pass"
    assert acceptance["publication_identity"][
        "ISR_column_access_audit"][
            "excluded_density_columns_materialized"] == 0
    assert set(acceptance["publication_identity"][
        "train_allowlist_identity"]) == set(p0b_audit.P0B_SOURCES)
    assert set(acceptance["publication_identity"][
        "train_token_directory_identity"]) == set(p0b_audit.P0B_SOURCES)
    assert [row["phase"] for row in manifest["publication_context_checks"]] == [
        "before_first_aggregation", "before_second_aggregation_replay",
        "before_summary_publication",
    ]
    assert [row["phase"] for row in acceptance[
            "publication_context_checks"]] == [
        "before_first_aggregation", "before_second_aggregation_replay",
        "before_summary_publication", "immediately_before_final_marker",
    ]
    assert acceptance["publication_identity_stable_across_all_checks"] is True
    statuses = _question_statuses(cache_dir)
    assert statuses == {
        "Q1_bias_origin": "supported",
        "Q2_increment_consistency": "supported",
        "Q3_profile_precision_concentration": "supported",
        "Q4_source_latitude_separation": "insufficient_evidence",
        "Q5_low_altitude_drift": "supported",
        "Q6_peak_persistence": "insufficient_evidence",
    }
    for name in (*summary.SUMMARY_FILES, summary.SUMMARY_MANIFEST,
                 summary.FINAL_MARKER):
        text = (cache_dir / name).read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
    regime = json.loads((cache_dir / "query_regime_summary.json").read_text(
        encoding="utf-8"))
    assert sum(row["query_points"] for row in regime["rows"]) == 64
    assert all(row["station_time_profiles"] == 32 for row in regime["rows"])
    assert all(row["unique_satellite_profiles"] == 32 for row in regime["rows"])
    source = json.loads((cache_dir / "query_source_summary.json").read_text(
        encoding="utf-8"))
    assert source["metric_reduction_order"] == (
        summary._SOURCE_METRIC_REDUCTION_ORDER)


def test_summary_build_is_deterministic(tmp_path):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    inventory = summary.validate_cache_inventory(cache_dir)
    first = summary.build_fixed_summaries(inventory)
    second = summary.build_fixed_summaries(inventory)
    assert [summary._json_text(value) for value in first] == [
        summary._json_text(value) for value in second]


@pytest.mark.parametrize("failure", ["missing", "undeclared", "duplicate"])
def test_inventory_rejects_missing_undeclared_and_duplicate_paths(
        tmp_path, failure):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    target = next((cache_dir / "batches").rglob("*_edge.npz"))
    if failure == "missing":
        target.unlink()
        with pytest.raises(FileNotFoundError):
            summary.validate_cache_inventory(cache_dir)
        return
    if failure == "undeclared":
        shutil.copyfile(target, target.with_name("batch_999999_edge.npz"))
        with pytest.raises(ValueError, match="inventory|incomplete"):
            summary.validate_cache_inventory(cache_dir)
        return
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(deepcopy(manifest["artifacts"][0]))
    _write_json(manifest_path, manifest)
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    with pytest.raises(ValueError, match="duplicate declared"):
        summary.validate_cache_inventory(cache_dir)


def test_inventory_rejects_count_and_identity_mismatch(tmp_path):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["size_bytes"] += 1
    _write_json(manifest_path, manifest)
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    with pytest.raises(ValueError, match="identity differs|identity mismatch"):
        summary.validate_cache_inventory(cache_dir)

    cache_dir2, _, artifact_paths2, marker_payload2 = _make_cache(
        tmp_path / "count")
    manifest_path2 = cache_dir2 / "manifest.json"
    manifest2 = json.loads(manifest_path2.read_text(encoding="utf-8"))
    manifest2["counts"]["query_rows"] += 1
    manifest2["raw_registry"]["date_partition_counts"]["train"] += 1
    _write_json(manifest_path2, manifest2)
    marker_payload2 = deepcopy(marker_payload2)
    marker_payload2["counts"] = deepcopy(manifest2["counts"])
    marker_payload2["query_partition_counts"] = deepcopy(
        manifest2["raw_registry"]["date_partition_counts"])
    _rewrite_cache_marker(cache_dir2, artifact_paths2, marker_payload2)
    inventory2 = summary.validate_cache_inventory(cache_dir2)
    with pytest.raises(ValueError, match="validated row counts"):
        summary.build_fixed_summaries(inventory2)


def test_build_rechecks_shard_identity_before_read(tmp_path):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    inventory = summary.validate_cache_inventory(cache_dir)
    target = inventory.batches[0].paths["query"]
    target.write_bytes(target.read_bytes() + b"tampered-after-inventory")

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        summary.build_fixed_summaries(inventory)


def test_build_rechecks_shard_identity_after_read(tmp_path, monkeypatch):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    inventory = summary.validate_cache_inventory(cache_dir)
    target = inventory.batches[0].paths["query"]
    original = summary.validate_batch_tables
    tampered = {"done": False}

    def tamper_after_read(shard, contract):
        result = original(shard, contract)
        if not tampered["done"]:
            target.write_bytes(target.read_bytes() + b"tampered-during-read")
            tampered["done"] = True
        return result

    monkeypatch.setattr(summary, "validate_batch_tables", tamper_after_read)
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        summary.build_fixed_summaries(inventory)
    assert tampered["done"] is True


@pytest.mark.parametrize("identity_name", ["coordinate_enrichment", "profile_cap_status"])
def test_inventory_rejects_coordinate_or_profile_cap_ledger_drift(
        tmp_path, identity_name):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    marker_payload = deepcopy(marker_payload)
    if identity_name == "coordinate_enrichment":
        marker_payload[identity_name]["aacgm"]["method"] = "BAD"
    else:
        marker_payload[identity_name] = "active"
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    with pytest.raises(ValueError, match="coordinate enrichment|profile-cap status"):
        summary.validate_cache_inventory(cache_dir)


@pytest.mark.parametrize(
    "ledger", ["runtime", "runtime_identities", "manifest", "cache_marker"])
def test_inventory_rejects_isr_column_access_ledger_drift(tmp_path, ledger):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    runtime_path = cache_dir / "audit_contract.json"
    manifest_path = cache_dir / "manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker_payload = deepcopy(marker_payload)
    if ledger == "runtime":
        runtime["ISR_column_access_audit"][
            "excluded_density_columns_materialized"] = 1
        _write_json(runtime_path, runtime)
    elif ledger == "runtime_identities":
        runtime["runtime_identities"]["ISR_column_access_audit"]["status"] = "fail"
        _write_json(runtime_path, runtime)
    elif ledger == "manifest":
        manifest["ISR_column_access_audit"]["status"] = "fail"
        _write_json(manifest_path, manifest)
    else:
        marker_payload["ISR_column_access_audit"]["status"] = "fail"
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    with pytest.raises(ValueError, match="ISR column-access audit"):
        summary.validate_cache_inventory(cache_dir)


@pytest.mark.parametrize(
    "ledger", ["runtime", "manifest", "manifest_after", "cache_marker"])
def test_inventory_rejects_preflight_dependency_ledger_drift(tmp_path, ledger):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    runtime_path = cache_dir / "audit_contract.json"
    manifest_path = cache_dir / "manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker_payload = deepcopy(marker_payload)
    if ledger == "runtime":
        runtime["preflight_dependency"]["sha256"] = "0" * 64
        _write_json(runtime_path, runtime)
    elif ledger == "manifest":
        manifest["preflight_dependency"]["sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
    elif ledger == "manifest_after":
        manifest["preflight_dependency_after"]["sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
    else:
        marker_payload["preflight_dependency"]["sha256"] = "0" * 64
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    with pytest.raises(ValueError, match="preflight dependency"):
        summary.validate_cache_inventory(cache_dir)


@pytest.mark.parametrize(
    ("identity_name", "ledger"),
    [
        (identity_name, ledger)
        for identity_name in (
            "train_allowlist_identity", "train_token_directory_identity")
        for ledger in ("runtime", "runtime_identities", "manifest", "cache_marker")
    ],
)
def test_inventory_rejects_train_identity_four_ledger_drift(
        tmp_path, identity_name, ledger):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    runtime_path = cache_dir / "audit_contract.json"
    manifest_path = cache_dir / "manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker_payload = deepcopy(marker_payload)

    def corrupt(value):
        if identity_name == "train_allowlist_identity":
            value["FY"]["profile_count"] += 1
        else:
            value["FY"]["sha256"] = "0" * 64

    if ledger == "runtime":
        corrupt(runtime[identity_name])
        _write_json(runtime_path, runtime)
    elif ledger == "runtime_identities":
        corrupt(runtime["runtime_identities"][identity_name])
        _write_json(runtime_path, runtime)
    elif ledger == "manifest":
        corrupt(manifest[identity_name])
        _write_json(manifest_path, manifest)
    else:
        corrupt(marker_payload[identity_name])
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    with pytest.raises(ValueError, match=identity_name):
        summary.validate_cache_inventory(cache_dir)


@pytest.mark.parametrize(
    "identity_name",
    ["train_allowlist_identity", "train_token_directory_identity",
     "ISR_column_access_audit"],
)
def test_inventory_cross_binds_preflight_attestation_to_full_identity(
        tmp_path, identity_name):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    runtime_path = cache_dir / "audit_contract.json"
    manifest_path = cache_dir / "manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker_payload = deepcopy(marker_payload)

    def corrupt(dependency):
        if identity_name == "train_allowlist_identity":
            dependency[identity_name]["FY"]["profile_count"] += 1
        elif identity_name == "train_token_directory_identity":
            dependency[identity_name]["FY"]["sha256"] = "0" * 64
        else:
            dependency[identity_name]["status"] = "fail"

    corrupt(runtime["preflight_dependency"])
    corrupt(manifest["preflight_dependency"])
    corrupt(manifest["preflight_dependency_after"])
    corrupt(marker_payload["preflight_dependency"])
    _write_json(runtime_path, runtime)
    _write_json(manifest_path, manifest)
    _rewrite_cache_marker(cache_dir, artifact_paths, marker_payload)
    with pytest.raises(ValueError, match=f"preflight dependency {identity_name}"):
        summary.validate_cache_inventory(cache_dir)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_strict_json_reader_rejects_all_nonfinite_numbers(tmp_path, literal):
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"ignored_extra": {literal}}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        summary._read_json_object(path)


def test_cache_marker_nonfinite_extra_is_rejected_before_inventory(tmp_path):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    marker = cache_dir / summary.CACHE_MARKER
    text = marker.read_text(encoding="utf-8")
    marker.write_text(
        text.rstrip()[:-1] + ', "ignored_extra": Infinity}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        summary.validate_cache_inventory(cache_dir)


@pytest.mark.parametrize("missing", ["profile_equal_weighting", "edge_diagnostics"])
def test_summary_contract_rejects_missing_frozen_aggregation_rules(missing):
    contract, _ = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    contract = deepcopy(contract)
    del contract["fixed_aggregation"][missing]
    with pytest.raises(KeyError):
        summary._validate_summary_contract(contract)


def test_exact_dtype_and_cross_table_validation(tmp_path):
    cache_dir, contract, artifact_paths, marker_payload = _make_cache(tmp_path)
    query_path = next((cache_dir / "batches").rglob("*_query.npz"))
    relative = query_path.relative_to(cache_dir).as_posix()
    with np.load(query_path, allow_pickle=False) as loaded:
        query = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    query["altitude_km"] = query["altitude_km"].astype(np.float64)
    np.savez_compressed(query_path, **query)
    _refresh_shard_identity(
        cache_dir, relative, artifact_paths, marker_payload)
    inventory = summary.validate_cache_inventory(cache_dir)
    shard = next(item for item in inventory.batches
                 if item.paths["query"] == query_path)
    with pytest.raises(TypeError, match="dtype"):
        summary.validate_batch_tables(shard, contract)

    cache_dir2, contract2, artifact_paths2, marker_payload2 = _make_cache(
        tmp_path / "second")
    edge_path = next((cache_dir2 / "batches").rglob("*_edge.npz"))
    relative2 = edge_path.relative_to(cache_dir2).as_posix()
    with np.load(edge_path, allow_pickle=False) as loaded:
        edge = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    edge["query_id"][1] = edge["query_id"][0]
    edge["token_row_id"][1] = edge["token_row_id"][0]
    np.savez_compressed(edge_path, **edge)
    _refresh_shard_identity(
        cache_dir2, relative2, artifact_paths2, marker_payload2)
    inventory2 = summary.validate_cache_inventory(cache_dir2)
    shard2 = next(item for item in inventory2.batches
                  if item.paths["edge"] == edge_path)
    with pytest.raises(ValueError, match="duplicate query-token edge"):
        summary.validate_batch_tables(shard2, contract2)


def test_final_marker_is_last_and_exclusive(tmp_path):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    marker = summary.summarize_cache(cache_dir)
    predecessor_times = [
        (cache_dir / name).stat().st_mtime_ns
        for name in (*summary.SUMMARY_FILES, summary.SUMMARY_MANIFEST)
    ]
    assert marker.stat().st_mtime_ns >= max(predecessor_times)
    with pytest.raises(FileExistsError):
        summary.summarize_cache(cache_dir)


def test_publish_rechecks_complete_inventory_before_final_marker(tmp_path):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    inventory = summary.validate_cache_inventory(cache_dir)
    payloads = summary.build_fixed_summaries(inventory)
    target = inventory.batches[-1].paths["edge"]
    target.write_bytes(target.read_bytes() + b"tampered-before-publication")

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        summary.publish_summaries(inventory, *payloads)
    assert not (cache_dir / summary.FINAL_MARKER).exists()


def test_publication_failure_never_leaves_final_marker(tmp_path, monkeypatch):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    original = summary._atomic_write_text_exclusive
    calls = {"count": 0}

    def fail_third(path, text):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("synthetic summary write failure")
        return original(path, text)

    monkeypatch.setattr(summary, "_atomic_write_text_exclusive", fail_third)
    with pytest.raises(OSError, match="synthetic"):
        summary.summarize_cache(cache_dir)
    assert not (cache_dir / summary.FINAL_MARKER).exists()


@pytest.mark.parametrize(
    "field", ["git", "python_environment", "source_provenance",
              "preflight_acceptance"])
def test_current_publication_identity_mismatch_blocks_before_aggregation(
        tmp_path, monkeypatch, field):
    cache_dir, _, _, _ = _make_cache(tmp_path)

    def mismatched(inventory):
        value = deepcopy(_publication_context_for_inventory(inventory))
        if field == "git":
            value[field]["head"] = "f" * 40
        elif field == "python_environment":
            value[field]["numpy_version"] = "different"
        elif field == "source_provenance":
            next(iter(value[field].values()))["size_bytes"] += 1
        else:
            value[field]["sha256"] = "f" * 64
        return value

    monkeypatch.setattr(summary, "_capture_current_publication_context", mismatched)
    with pytest.raises(ValueError, match="current publication|current preflight"):
        summary.summarize_cache(cache_dir)
    assert not (cache_dir / summary.FINAL_MARKER).exists()
    assert not any((cache_dir / name).exists() for name in summary.SUMMARY_FILES)


@pytest.mark.parametrize("bad_call", [2, 3, 4])
def test_each_late_publication_gate_blocks_final_marker(
        tmp_path, monkeypatch, bad_call):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    calls = {"count": 0}

    def capture(inventory):
        calls["count"] += 1
        value = deepcopy(_publication_context_for_inventory(inventory))
        if calls["count"] == bad_call:
            value["git"]["head"] = "f" * 40
        return value

    monkeypatch.setattr(summary, "_capture_current_publication_context", capture)
    with pytest.raises(ValueError, match="current publication Git identity"):
        summary.summarize_cache(cache_dir)
    assert calls["count"] == bad_call
    assert not (cache_dir / summary.FINAL_MARKER).exists()


def test_cell_metrics_are_station_time_profile_equal_weighted():
    acc = summary.GroupAccumulator()
    dominant = ("Jicamarca", "20240901", 1)
    sparse = ("Jicamarca", "20240901", 2)
    acc.query_points = 10
    acc.station_times.update((dominant, sparse))
    for _ in range(9):
        acc.add_value("raw_error", 1.0, dominant)
    acc.add_value("raw_error", -1.0, sparse)
    assert acc.query_points == 10
    assert summary._metric(acc, "raw_error").tolist() == [1.0, -1.0]
    assert summary._mean(summary._metric(acc, "raw_error")) == pytest.approx(0.0)
    assert summary._rmse(summary._metric(acc, "raw_error")) == pytest.approx(1.0)


def test_source_sort_uses_trailing_source_after_solar_dimension():
    contract, _ = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    fy_key = (
        "Jicamarca", "train", "[00,03)h", "night", "[120,200)km",
        "[0,2)", "joint", "FY")
    cosmic_key = fy_key[:-1] + ("COSMIC",)
    assert summary._fixed_group_sort_key(fy_key, contract) < (
        summary._fixed_group_sort_key(cosmic_key, contract))


@pytest.mark.parametrize(
    ("cos_sza", "expected"),
    ((0.1, "day"), (-0.1, "twilight"), (-0.5, "night")),
)
def test_grouping_uses_aacgm_mlt_and_explicit_solar_regime(cos_sza, expected):
    contract, _ = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    query, _, _ = _synthetic_tables(
        contract, "20240901", "train", 0, first_id=0, count=1)
    query["local_time_hour"][0] = 19.0
    query["aacgm_mlt_hour"][0] = 1.0
    query["cos_sza"][0] = cos_sza
    key = summary._group_key(query, 0, contract)
    assert key[2] == "[00,03)h"
    assert key[3] == expected


def test_token_diagnostics_deduplicate_and_report_vertical_shape():
    contract, _ = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    rules = summary._edge_diagnostic_rules(contract)
    registry = {}
    identities = set()
    for profile_id in range(30):
        for token_id, altitude in enumerate((200.0, 225.0, 250.0, 275.0)):
            identity = ("FY", profile_id, token_id)
            identities.add(identity)
            registry[identity] = {
                "altitude_km": altitude,
                "innovation_dex": profile_id * 0.01 + token_id * 0.1,
            }
    result = summary._token_profile_diagnostics(identities, registry, rules)
    assert result["unique_token_count"] == 120
    assert result["innovation_profile_count"] == 30
    assert result["token_height_interval_profile_count"] == 30
    assert result["token_height_interval_median_km"] == pytest.approx(25.0)
    assert result["vertical_innovation_correlation_defined_profiles"] == 30
    assert result["vertical_innovation_correlation_status"] == "computed"
    assert result["vertical_innovation_correlation_median"] == pytest.approx(1.0)

    empty = summary._token_profile_diagnostics(set(), {}, rules)
    assert empty["innovation_profile_count"] == 0
    assert empty["innovation_status"] == "insufficient_data"
    assert empty["innovation_median_dex"] is None
    assert empty["token_height_interval_status"] == "insufficient_data"
    assert empty["token_height_interval_median_km"] is None
    assert empty["vertical_innovation_correlation_status"] == "insufficient_data"
    assert empty["vertical_innovation_correlation_median"] is None


def test_zero_mad_tail_marks_only_nonzero_deviations():
    contract, _ = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    rules = summary._edge_diagnostic_rules(contract)
    registry = {}
    identities = set()
    for profile_id, innovation in enumerate((0.0, 0.0, 0.0, 0.0, 1.0)):
        identity = ("FY", profile_id, 0)
        identities.add(identity)
        registry[identity] = {
            "altitude_km": 250.0,
            "innovation_dex": innovation,
        }
    result = summary._token_profile_diagnostics(identities, registry, rules)
    assert result["innovation_mad_dex"] == pytest.approx(0.0)
    assert result["innovation_robust_sigma_dex"] == pytest.approx(0.0)
    assert result["innovation_3sigma_tail_profile_count"] == 1
    assert result["innovation_3sigma_tail_profile_fraction"] == pytest.approx(0.2)


def test_cross_shard_token_identity_requires_bitwise_identical_payload():
    contract, _ = p0b_audit.load_p0b_contract(CONTRACT_PATH)
    rules = summary._edge_diagnostic_rules(contract)
    _, token_a, edge_a = _synthetic_tables(
        contract, "20240901", "train", 0, first_id=0, count=1)
    _, token_b, edge_b = _synthetic_tables(
        contract, "20240902", "train", 1, first_id=100, count=1)
    payload_fields = tuple(
        name for name in rules.duplicate_token_payload_fields
        if name != "innovation_dex")
    for name in ("source", "profile_id", "token_id", *payload_fields):
        token_b[name][0] = token_a[name][0]
    registry = {}
    summary._register_token_payloads(token_a, edge_a, registry, rules)
    summary._register_token_payloads(token_b, edge_b, registry, rules)
    assert len(registry) == 1

    registry = {}
    summary._register_token_payloads(token_a, edge_a, registry, rules)
    token_b["observation_log10_ne"][0] += np.float32(0.01)
    with pytest.raises(ValueError, match="non-identical payload"):
        summary._register_token_payloads(token_b, edge_b, registry, rules)


def test_q2_direction_is_not_inferred_from_closure_identity(tmp_path):
    cache_dir, _, artifact_paths, marker_payload = _make_cache(tmp_path)
    for edge_path in sorted((cache_dir / "batches").rglob("*_edge.npz")):
        relative = edge_path.relative_to(cache_dir).as_posix()
        with np.load(edge_path, allow_pickle=False) as loaded:
            edge = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
        edge["innovation_dex"] *= np.float32(-1.0)
        edge["gain_joint"] *= np.float32(-1.0)
        edge["gain_isolated"] *= np.float32(-1.0)
        np.savez_compressed(edge_path, **edge)
        _refresh_shard_identity(
            cache_dir, relative, artifact_paths, marker_payload)
    inventory = summary.validate_cache_inventory(cache_dir)
    questions = summary.build_fixed_summaries(inventory)[3]
    q2 = next(row for row in questions["questions"]
              if row["id"] == "Q2_increment_consistency")
    assert q2["status"] == "not_supported"
    assert q2["evidence"]["material_closure_failures"] == 0
    assert all(
        row["direction_consistency_fraction"] == pytest.approx(0.0)
        for row in q2["evidence"]["material_eligible_cells"])


def test_summarize_reaggregates_twice_before_any_publication(tmp_path, monkeypatch):
    cache_dir, _, _, _ = _make_cache(tmp_path)
    original = summary.build_fixed_summaries
    calls = {"count": 0}

    def divergent(inventory):
        calls["count"] += 1
        result = original(inventory)
        if calls["count"] == 2:
            result = deepcopy(result)
            result[0]["rows"][0]["M00_bias_mean_dex"] += 0.001
        return result

    monkeypatch.setattr(summary, "build_fixed_summaries", divergent)
    with pytest.raises(ValueError, match="two in-memory cache aggregations differ"):
        summary.summarize_cache(cache_dir)
    assert calls["count"] == 2
    assert not any((cache_dir / name).exists() for name in (
        *summary.SUMMARY_FILES, summary.SUMMARY_MANIFEST, summary.FINAL_MARKER))
