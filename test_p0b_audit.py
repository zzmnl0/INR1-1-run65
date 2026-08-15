"""Focused regression tests for the frozen M2-W2 P0-B audit contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from inr_modules.mdia.p0b_audit import (
    P0B_AUDIT_SCHEMA_VERSION,
    P0B_CONTRACT_ID,
    P0B_TABLE_DTYPE_NAMES,
    atomic_write_json,
    build_train_token_directory_identity,
    contract_table_schemas,
    dominant_profile_precision_share,
    effective_sample_size,
    load_p0b_contract,
    predictive_nis_unlocalized,
    profile_precision_statistics,
    strict_json_loads,
    validate_audit_tables,
    validate_cross_table_diagnostics,
    validate_foreign_keys,
    validate_p0b_contract,
    validate_query_closures,
    validate_train_token_directory_identity,
    write_completion_marker_atomically,
)


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "m2w2_contracts" / "p0b_audit_contract_v1.json"


def _scope(count):
    return {
        "station": np.full(count, "Jicamarca", dtype="U10"),
        "date_utc": np.full(count, "20240902", dtype="U8"),
        "batch_id": np.zeros(count, dtype=np.int64),
    }


def _valid_tables():
    m00 = np.asarray([10.0, 11.0], dtype=np.float32)
    isolated_fy = np.asarray([0.10, 0.0], dtype=np.float32)
    isolated_cosmic = np.asarray([0.20, 0.0], dtype=np.float32)
    joint_fy = np.asarray([0.08, 0.0], dtype=np.float32)
    joint_cosmic = np.asarray([0.12, 0.0], dtype=np.float32)
    joint = joint_fy + joint_cosmic
    query = {
        **_scope(2),
        "query_id": np.asarray([100, 101], dtype=np.int64),
        "sample_key": np.asarray(["sample-100", "sample-101"], dtype="U10"),
        "query_split": np.asarray(["train", "train"], dtype="U11"),
        "raw_iri_log10_ne": np.asarray([9.9, 10.8], dtype=np.float32),
        "M00_log10_ne": m00,
        "M10_log10_ne": m00 + isolated_fy,
        "M01_log10_ne": m00 + isolated_cosmic,
        "M11_log10_ne": m00 + joint,
        "no_token_log10_ne": m00.copy(),
        "isolated_increment_FY_dex": isolated_fy,
        "isolated_increment_COSMIC_dex": isolated_cosmic,
        "joint_update_FY_dex": joint_fy,
        "joint_update_COSMIC_dex": joint_cosmic,
        "joint_increment_dex": joint,
        "raw_coverage_code": np.asarray(["joint", "no_token"], dtype="U12"),
        "coverage_code": np.asarray(["joint", "no_token"], dtype="U12"),
        "FY_token_count": np.asarray([1, 0], dtype=np.int64),
        "FY_unique_profile_count": np.asarray([1, 0], dtype=np.int64),
        "FY_unlocalized_precision_sum": np.asarray([2.0, 0.0], dtype=np.float32),
        "FY_localized_precision_sum": np.asarray([1.0, 0.0], dtype=np.float32),
        "FY_token_neff": np.asarray([1.0, 0.0], dtype=np.float32),
        "FY_profile_neff": np.asarray([1.0, 0.0], dtype=np.float32),
        "FY_max_profile_precision_share": np.asarray([1.0, 0.0], dtype=np.float32),
        "FY_dominant_profile_id": np.asarray([7, -1], dtype=np.int64),
        "FY_dominant_profile_valid": np.asarray([True, False], dtype=np.bool_),
        "COSMIC_token_count": np.asarray([1, 0], dtype=np.int64),
        "COSMIC_unique_profile_count": np.asarray([1, 0], dtype=np.int64),
        "COSMIC_unlocalized_precision_sum": np.asarray(
            [4.0, 0.0], dtype=np.float32),
        "COSMIC_localized_precision_sum": np.asarray(
            [3.0, 0.0], dtype=np.float32),
        "COSMIC_token_neff": np.asarray([1.0, 0.0], dtype=np.float32),
        "COSMIC_profile_neff": np.asarray([1.0, 0.0], dtype=np.float32),
        "COSMIC_max_profile_precision_share": np.asarray(
            [1.0, 0.0], dtype=np.float32),
        "COSMIC_dominant_profile_id": np.asarray([9, -1], dtype=np.int64),
        "COSMIC_dominant_profile_valid": np.asarray(
            [True, False], dtype=np.bool_),
        "predictive_nis_unlocalized_FY": np.asarray(
            [0.05, 0.0], dtype=np.float64),
        "predictive_nis_unlocalized_FY_dof": np.asarray(
            [1, 0], dtype=np.int64),
        "predictive_nis_unlocalized_COSMIC": np.asarray(
            [0.08, 0.0], dtype=np.float64),
        "predictive_nis_unlocalized_COSMIC_dof": np.asarray(
            [1, 0], dtype=np.int64),
        "predictive_nis_unlocalized_joint": np.asarray(
            [0.10, 0.0], dtype=np.float64),
        "predictive_nis_unlocalized_joint_dof": np.asarray(
            [2, 0], dtype=np.int64),
    }
    token = {
        **_scope(2),
        "token_row_id": np.asarray([50, 51], dtype=np.int64),
        "source": np.asarray(["FY", "COSMIC"], dtype="U6"),
        "profile_id": np.asarray([7, 9], dtype=np.int64),
        "token_id": np.asarray([0, 1], dtype=np.int64),
        "profile_split": np.asarray(["train", "train"], dtype="U5"),
    }
    innovation = np.asarray([0.2, 0.3], dtype=np.float32)
    gain_joint = np.asarray([0.4, 0.4], dtype=np.float32)
    gain_isolated = np.asarray([0.5, 2.0 / 3.0], dtype=np.float32)
    edge = {
        **_scope(2),
        "query_id": np.asarray([100, 100], dtype=np.int64),
        "token_row_id": np.asarray([50, 51], dtype=np.int64),
        "source": np.asarray(["FY", "COSMIC"], dtype="U6"),
        "innovation_dex": innovation,
        "gain_joint": gain_joint,
        "gain_isolated": gain_isolated,
        "contribution_joint_dex": gain_joint * innovation,
        "contribution_isolated_dex": gain_isolated * innovation,
        "unlocalized_precision": np.asarray([2.0, 4.0], dtype=np.float32),
        "localized_precision": np.asarray([1.0, 3.0], dtype=np.float32),
    }
    return query, token, edge


def _append_row(table, values):
    result = {key: np.asarray(value).copy() for key, value in table.items()}
    for key, value in values.items():
        result[key] = np.concatenate([
            result[key], np.asarray([value], dtype=result[key].dtype)])
    missing = set(result).difference(values)
    for key in missing:
        result[key] = np.concatenate([result[key], result[key][-1:]])
    return result


def test_frozen_contract_loads_and_hashes():
    contract, digest = load_p0b_contract(CONTRACT_PATH)
    assert contract["contract_id"] == P0B_CONTRACT_ID
    assert contract["audit_schema_version"] == P0B_AUDIT_SCHEMA_VERSION
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert contract["data_scope"]["satellite_tokens"][
        "profile_partition"] == "train"
    assert contract["p0a_dependency_exception"]["p0c_locked"] is True
    assert contract["p0a_dependency_exception"]["p0b_result_status_until_p0a_resolution"] == "provisional"
    assert contract["output_lifecycle"]["start_precondition"] == (
        "target directory must not exist")


def _synthetic_token_indexes():
    result = {}
    for offset, source in enumerate(("FY", "COSMIC")):
        index = type("SyntheticTokenIndex", (), {})()
        index.token_coords = np.asarray([
            [1.0 + offset, 2.0, 250.0, 3.0],
            [4.0 + offset, 5.0, 350.0, 6.0],
        ], dtype="<f4")
        index.token_values = np.asarray([9.0, 10.0], dtype="<f4")
        index.token_profile_ids = np.asarray(
            [100 + offset, 100 + offset], dtype="<i8")
        index.token_ids = np.asarray([0, 1], dtype="<i8")
        result[source] = index
    return result


def test_train_token_directory_identity_is_exact_and_mutation_sensitive():
    indexes = _synthetic_token_indexes()
    first = build_train_token_directory_identity(indexes)
    second = build_train_token_directory_identity(indexes)
    assert first == second == validate_train_token_directory_identity(first)
    assert set(first) == {"FY", "COSMIC"}
    assert first["FY"]["token_rows"] == 2
    assert first["FY"]["arrays"]["token_coords"] == {
        "dtype": "<f4", "shape": [2, 4]}

    indexes["FY"].token_values[0] += np.float32(0.25)
    changed = build_train_token_directory_identity(indexes)
    assert changed["FY"]["sha256"] != first["FY"]["sha256"]
    assert changed["COSMIC"] == first["COSMIC"]

    malformed = deepcopy(first)
    malformed["FY"]["sha256"] = "Z" * 64
    with pytest.raises(ValueError, match="SHA256"):
        validate_train_token_directory_identity(malformed)


@pytest.mark.parametrize("payload", [
    '{"value": NaN}',
    '{"value": Infinity}',
    '{"value": -Infinity}',
    '{"nested": [{"value": 1e999}]}',
])
def test_strict_json_rejects_nonfinite_constants_and_overflow(payload):
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_loads(payload, label="synthetic")


def test_contract_loader_rejects_nonfinite_json_before_validation(tmp_path):
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    changed = text.replace(
        '"purpose":', '"forbidden_nonfinite": NaN,\n  "purpose":', 1)
    path = tmp_path / "nonfinite_contract.json"
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_p0b_contract(path)


def test_contract_freezes_train_token_directory_identity_semantics():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    identity = contract["train_token_directory_identity_contract"]
    assert identity["preflight_full_exact_equality_required"] is True
    assert identity["end_of_run_recompute_required"] is True
    assert len(identity["required_ledgers"]) == 4

    changed = deepcopy(contract)
    changed["train_token_directory_identity_contract"][
        "end_of_run_recompute_required"] = False
    with pytest.raises(ValueError, match="end-of-run token identity"):
        validate_p0b_contract(changed)


def test_contract_freezes_restricted_mixed_isr_file_identity_semantics():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    identity = contract["data_scope"]["ISR_queries"][
        "restricted_mixed_file_identity"]
    assert identity["whole_file_sha256_recompute_from_hdf_bytes_allowed"] is False
    assert identity["whole_file_size_stat_allowed"] is True
    assert identity["materialized_allowed_content_identity_required"] is True
    assert identity["materialized_allowed_content_schema"] == (
        "isr_allowed_materialized_content_v1")
    assert "length-framed arrays" in identity[
        "materialized_allowed_content_digest_framing"]
    assert identity["locked_or_excluded_values_in_content_identity_allowed"] is False
    assert identity["station_and_global_path_sorted_ledgers_required"] is True

    changed = deepcopy(contract)
    changed["data_scope"]["ISR_queries"][
        "restricted_mixed_file_identity"][
            "whole_file_sha256_recompute_from_hdf_bytes_allowed"] = True
    with pytest.raises(ValueError, match="restricted mixed-file identity"):
        validate_p0b_contract(changed)


def test_contract_rejects_scope_and_counterfactual_drift():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    changed = deepcopy(contract)
    changed["data_scope"]["satellite_tokens"][
        "profile_partition"] = "train+development"
    with pytest.raises(ValueError, match="train-only"):
        validate_p0b_contract(changed)

    changed = deepcopy(contract)
    changed["data_scope"]["satellite_profile_index_metadata"][
        "development_density_values_allowed"] = True
    with pytest.raises(ValueError, match="development density"):
        validate_p0b_contract(changed)

    changed = deepcopy(contract)
    changed["counterfactuals"]["height_deletion"]["bands"][0][
        "upper_km"] = 251.0
    with pytest.raises(ValueError, match="height-deletion"):
        validate_p0b_contract(changed)

    changed = deepcopy(contract)
    changed["output_lifecycle"]["overwrite_allowed"] = True
    with pytest.raises(ValueError, match="overwrite"):
        validate_p0b_contract(changed)

    changed = deepcopy(contract)
    changed["version_control"]["required_branch"] = "codex/wrong-branch"
    with pytest.raises(ValueError, match="Git branch"):
        validate_p0b_contract(changed)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("untracked_generated_artifacts_allowed", False, "generated untracked"),
    ("explicit_path_staging_only", False, "explicit-path staging"),
    ("git_add_all_forbidden", False, "add-all"),
    ("accepted_history_rewrite_forbidden", False, "history rewrites"),
])
def test_contract_rejects_version_control_policy_drift(field, value, message):
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    changed = deepcopy(contract)
    changed["version_control"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_p0b_contract(changed)


def test_contract_requires_all_p0b_tests_as_immutable_critical_paths():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    expected_tests = {
        "test_p0b_audit.py",
        "test_p0b_counterfactuals.py",
        "test_p0b_runner.py",
        "test_p0b_summary.py",
        "test_p0b_train_only_index.py",
        "test_p0b_isr_column_filter.py",
        "test_isr_loader_geometry.py",
    }
    critical = set(contract["version_control"]["critical_tracked_paths"])
    assert expected_tests <= critical
    changed = deepcopy(contract)
    changed["version_control"]["critical_tracked_paths"].remove(
        "test_p0b_runner.py")
    with pytest.raises(ValueError, match="critical tracked paths"):
        validate_p0b_contract(changed)


@pytest.mark.parametrize("field", [
    "predictive_nis_unlocalized_FY",
    "predictive_nis_unlocalized_FY_dof",
    "predictive_nis_unlocalized_COSMIC",
    "predictive_nis_unlocalized_COSMIC_dof",
    "predictive_nis_unlocalized_joint",
    "predictive_nis_unlocalized_joint_dof",
])
def test_contract_rejects_missing_query_predictive_nis_field(field):
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    changed = deepcopy(contract)
    changed["cache_schema"]["tables"]["query"]["required_fields"].remove(field)
    with pytest.raises(ValueError, match="required_fields"):
        validate_p0b_contract(changed)


def test_contract_exposes_exact_nonobject_dtype_maps_without_p0a_cache_fields():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    schemas = contract_table_schemas(contract)
    assert tuple(schemas) == ("query", "token", "edge")
    assert contract["cache_schema"]["tables"]["query"]["dtypes"] == (
        P0B_TABLE_DTYPE_NAMES["query"])
    assert all(
        dtype.kind != "O"
        for table in schemas.values() for dtype in table["dtypes"].values())
    query_fields = set(schemas["query"]["required_fields"])
    assert not any(field.startswith("P0A_") for field in query_fields)
    assert contract["data_scope"]["P0A_ISR_contract_identity"][
        "npz_or_peak_cache_read_allowed"] is False


def test_contract_rejects_dtype_drift():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    changed = deepcopy(contract)
    changed["cache_schema"]["tables"]["query"]["dtypes"][
        "query_id"] = "float64"
    with pytest.raises(ValueError, match="dtype map"):
        validate_p0b_contract(changed)


def test_contract_freezes_counts_identity_metadata_and_summary_lifecycle():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    scope = contract["data_scope"]
    assert scope["satellite_tokens"]["expected_train_profile_counts"] == {
        "FY": 44625, "COSMIC": 56099}
    assert scope["satellite_profile_index_metadata"][
        "expected_development_profile_counts"] == {
            "FY": 11591, "COSMIC": 13924}
    assert scope["ISR_queries"]["expected_finite_query_counts"] == {
        "train": 141223, "development": 34438}
    classification = scope["satellite_profile_index_metadata"][
        "locked_test_classification_metadata"]
    assert "h_cut_km" not in classification["allowed_fields"]
    assert classification["persistence_allowed"] is False
    assert classification["aggregation_allowed"] is False
    p0a = scope["P0A_ISR_contract_identity"]
    assert len(p0a["sha256"]) == 64 and p0a["size_bytes"] == 71263
    summary = contract["fixed_aggregation"]["summary_artifacts"]
    assert summary["manifest_self_listing_allowed"] is False
    assert summary["final_acceptance_written_exclusively_last"] is True

    changed = deepcopy(contract)
    changed["data_scope"]["ISR_queries"]["expected_finite_query_counts"][
        "train"] += 1
    with pytest.raises(ValueError, match="query counts"):
        validate_p0b_contract(changed)


def test_predictive_nis_unlocalized_matches_direct_system():
    innovation = np.asarray([0.4, -0.2, 0.3], dtype=np.float64)
    anomalies = np.asarray([
        [0.2, -0.1, 0.0],
        [0.3, 0.1, -0.2],
        [-0.1, 0.4, -0.3],
    ], dtype=np.float64)
    precision = np.asarray([2.0, 0.5, 1.5], dtype=np.float64)
    actual, dof = predictive_nis_unlocalized(
        innovation, anomalies, precision, n_members=3)
    system = (2.0 * np.eye(3)
              + anomalies.T @ (precision[:, None] * anomalies))
    rhs = anomalies.T @ (precision * innovation)
    expected = (innovation @ (precision * innovation)
                - rhs @ np.linalg.solve(system, rhs))
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert dof == 3


def test_predictive_nis_unlocalized_zero_tokens_and_bad_inputs():
    value, dof = predictive_nis_unlocalized(
        np.empty(0), np.empty((0, 3)), np.empty(0), n_members=3)
    assert value == 0.0
    assert dof == 0
    value, dof = predictive_nis_unlocalized(
        np.asarray([1.0]), np.zeros((1, 3)), np.asarray([0.0]))
    assert value == 0.0
    assert dof == 0

    with pytest.raises(ValueError, match="nonnegative"):
        predictive_nis_unlocalized(
            np.asarray([1.0]), np.zeros((1, 3)), np.asarray([-1.0]))
    with pytest.raises(ValueError, match="finite"):
        predictive_nis_unlocalized(
            np.asarray([np.nan]), np.zeros((1, 3)), np.asarray([1.0]))
    with pytest.raises(ValueError, match="dimensions disagree"):
        predictive_nis_unlocalized(
            np.asarray([1.0]), np.zeros((2, 3)), np.asarray([1.0]))
    with pytest.raises(ValueError, match="n_members disagrees"):
        predictive_nis_unlocalized(
            np.asarray([1.0]), np.zeros((1, 3)), np.asarray([1.0]),
            n_members=4)

def test_profile_and_token_effective_sample_size():
    precision = np.asarray([1.0, 1.0, 2.0])
    assert effective_sample_size(precision) == pytest.approx(16.0 / 6.0)
    stats = profile_precision_statistics(
        np.asarray([10, 10, 20]), precision)
    assert stats == {
        "token_count": 3,
        "unique_profile_count": 2,
        "localized_precision_sum": 4.0,
        "token_neff": pytest.approx(16.0 / 6.0),
        "profile_neff": pytest.approx(2.0),
        "max_profile_precision_share": pytest.approx(0.5),
        "dominant_profile_id": 10,
    }
    assert dominant_profile_precision_share(
        np.asarray([20, 10]), np.asarray([2.0, 2.0])) == (10, 0.5)
    empty = profile_precision_statistics(
        np.zeros(0, dtype=np.int64), np.zeros(0))
    assert empty["dominant_profile_id"] is None
    assert empty["profile_neff"] == 0.0
    with pytest.raises(ValueError, match="nonnegative"):
        effective_sample_size(np.asarray([1.0, -1.0]))
    with pytest.raises(ValueError, match="finite"):
        effective_sample_size(np.asarray([1.0, np.nan]))


def test_table_foreign_keys_closures_and_strict_no_token_identity():
    query, token, edge = _valid_tables()
    result = validate_audit_tables(query, token, edge)
    assert result["counts"] == {
        "query_rows": 2,
        "token_rows": 2,
        "edge_rows": 2,
    }
    assert result["query_closure"]["no_token_bitwise_equal"] is True
    assert result["recomputed_diagnostics"][
        "max_abs_recomputed_summary_error"] == pytest.approx(0.0)

    changed = {key: value.copy() for key, value in query.items()}
    changed["no_token_log10_ne"][0] = np.nextafter(
        changed["no_token_log10_ne"][0], np.float32(np.inf))
    with pytest.raises(ValueError, match="bitwise"):
        validate_query_closures(changed)


@pytest.mark.parametrize(("field", "bad_value", "message"), [
    ("raw_coverage_code", "no_token", "raw coverage"),
    ("coverage_code", "FY_only", "effective coverage"),
    ("FY_token_count", 2, "token_count"),
    ("FY_unique_profile_count", 2, "unique_profile_count"),
    ("FY_unlocalized_precision_sum", 3.0, "unlocalized_precision_sum"),
    ("FY_localized_precision_sum", 2.0, "localized_precision_sum"),
    ("FY_token_neff", 2.0, "token_neff"),
    ("FY_profile_neff", 2.0, "profile_neff"),
    ("FY_max_profile_precision_share", 0.5, "max_profile_precision_share"),
    ("FY_dominant_profile_id", 8, "dominant-profile"),
    ("FY_dominant_profile_valid", False, "dominant-profile"),
])
def test_cross_table_query_summary_corruption_is_rejected(
        field, bad_value, message):
    query, token, edge = _valid_tables()
    query[field][0] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_audit_tables(query, token, edge)


def test_cross_table_nis_dof_corruption_survives_source_sum_but_is_rejected():
    query, token, edge = _valid_tables()
    query["predictive_nis_unlocalized_FY_dof"][0] = 2
    query["predictive_nis_unlocalized_joint_dof"][0] = 3
    with pytest.raises(ValueError, match="FY predictive NIS dof"):
        validate_audit_tables(query, token, edge)


@pytest.mark.parametrize(("field", "bad_value", "message"), [
    ("unlocalized_precision", -1.0, "unlocalized_precision must be nonnegative"),
    ("localized_precision", np.inf, "localized_precision must be finite"),
])
def test_cross_table_precision_corruption_is_rejected(field, bad_value, message):
    query, token, edge = _valid_tables()
    edge[field][0] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_cross_table_diagnostics(query, token, edge)


def test_raw_and_effective_coverage_are_independent_at_zero_localized_precision():
    query, token, edge = _valid_tables()
    edge["localized_precision"][0] = 0.0
    query["coverage_code"][0] = "COSMIC_only"
    query["FY_localized_precision_sum"][0] = 0.0
    query["FY_token_neff"][0] = 0.0
    query["FY_profile_neff"][0] = 0.0
    query["FY_max_profile_precision_share"][0] = 0.0
    query["FY_dominant_profile_id"][0] = -1
    query["FY_dominant_profile_valid"][0] = False
    result = validate_cross_table_diagnostics(query, token, edge)
    assert result["max_abs_recomputed_summary_error"] == pytest.approx(0.0)


def test_full_contract_validation_rejects_partial_table_schema():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    query, token, edge = _valid_tables()
    with pytest.raises(ValueError, match="fields differ from frozen contract"):
        validate_audit_tables(query, token, edge, contract=contract)


def test_counterfeit_token_and_edge_duplicates_are_rejected():
    query, token, edge = _valid_tables()
    duplicate_token = _append_row(token, {
        "token_row_id": 52,
        "station": "Jicamarca",
        "date_utc": "20240902",
        "batch_id": 0,
        "source": "FY",
        "profile_id": 7,
        "token_id": 0,
        "profile_split": "train",
    })
    with pytest.raises(ValueError, match="source-profile-token"):
        validate_foreign_keys(query, duplicate_token, edge)

    duplicate_edge = _append_row(edge, {
        key: np.asarray(value)[0] for key, value in edge.items()
    })
    with pytest.raises(ValueError, match="query-token edge"):
        validate_foreign_keys(query, token, duplicate_edge)


def test_nontrain_token_is_rejected():
    query, token, edge = _valid_tables()
    token["profile_split"][0] = "devel"
    with pytest.raises(ValueError, match="non-train"):
        validate_foreign_keys(query, token, edge)


def test_atomic_json_and_completion_marker(tmp_path):
    artifact = tmp_path / "batch_000000_query.npz"
    artifact.write_bytes(b"completed artifact")
    marker = tmp_path / "p0b_audit_acceptance.json"
    write_completion_marker_atomically(
        marker,
        {"status": "pass"},
        [artifact],
        artifact_root=tmp_path,
    )
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert record["completion_marker"] is True
    assert record["audit_schema_version"] == P0B_AUDIT_SCHEMA_VERSION
    assert record["artifacts"][0]["path"] == artifact.name
    assert record["artifacts"][0]["size_bytes"] == artifact.stat().st_size
    with pytest.raises(FileExistsError):
        write_completion_marker_atomically(
            marker,
            {"status": "must_not_overwrite"},
            [artifact],
            artifact_root=tmp_path,
        )
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "pass"

    failed_marker = tmp_path / "must_not_exist.json"
    with pytest.raises(FileNotFoundError):
        write_completion_marker_atomically(
            failed_marker,
            {"status": "pass"},
            [tmp_path / "missing.npz"],
            artifact_root=tmp_path,
        )
    assert not failed_marker.exists()

    invalid_json = tmp_path / "nan_forbidden.json"
    with pytest.raises(ValueError):
        atomic_write_json(invalid_json, {"value": float("nan")})
    assert not invalid_json.exists()


def test_exclusive_atomic_json_temp_cleanup_failure_keeps_success(
        tmp_path, monkeypatch):
    marker = tmp_path / "completion.json"
    original_unlink = Path.unlink

    def fail_temporary_cleanup(path, *args, **kwargs):
        if path.name.startswith(f".{marker.name}.") and path.name.endswith(".tmp"):
            raise OSError("synthetic temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary_cleanup)
    result = atomic_write_json(marker, {"status": "pass"}, overwrite=False)
    assert result == marker
    assert json.loads(marker.read_text(encoding="utf-8")) == {"status": "pass"}
