"""Deterministic P0-B cache verification, aggregation, and attribution.

The inference runner writes immutable query/token/edge shards and the
``p0b_cache_acceptance.json`` marker.  This module treats that directory as a
read-only input, validates its complete inventory, and publishes the fixed
P0-B summaries.  The final ``p0b_audit_acceptance.json`` marker is written
exclusively and last.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import numpy as np

from inr_modules.mdia import p0b_audit


SUMMARY_SCHEMA_VERSION = 2
CACHE_MARKER = p0b_audit.P0B_CACHE_COMPLETION_MARKER
FINAL_MARKER = p0b_audit.P0B_COMPLETION_MARKER
SUMMARY_FILES = p0b_audit.P0B_SUMMARY_ARTIFACTS
SUMMARY_MANIFEST = p0b_audit.P0B_SUMMARY_MANIFEST
REPO_ROOT = Path(__file__).resolve().parents[1]

_BATCH_RE = re.compile(
    r"^batches/(?P<station>Jicamarca|PokerFlat)/(?P<date>[0-9]{8})/"
    r"batch_(?P<batch>[0-9]{6})_(?P<table>query|token|edge)\.npz$")

_DIMENSIONS = (
    "station",
    "query_split",
    "MLT_3h",
    "solar_regime",
    "query_altitude_band",
    "Kp_activity",
    "coverage_code",
)
_SOURCE_DIMENSIONS = _DIMENSIONS + ("source",)
_COUNTS = (
    "query_points",
    "station_time_profiles",
    "unique_days",
    "unique_satellite_profiles",
)

_REGIME_FIELDS = _DIMENSIONS + _COUNTS + (
    "FY_unique_satellite_profiles",
    "COSMIC_unique_satellite_profiles",
    "raw_iri_bias_mean_dex",
    "raw_iri_bias_median_dex",
    "raw_iri_rmse_dex",
    "M00_bias_mean_dex",
    "M00_bias_median_dex",
    "M00_rmse_dex",
    "M10_bias_mean_dex",
    "M10_rmse_dex",
    "M01_bias_mean_dex",
    "M01_rmse_dex",
    "M11_bias_mean_dex",
    "M11_rmse_dex",
    "background_increment_mean_dex",
    "background_increment_median_dex",
    "isolated_increment_FY_median_dex",
    "isolated_increment_COSMIC_median_dex",
    "joint_update_FY_mean_dex",
    "joint_update_COSMIC_mean_dex",
    "joint_interaction_mean_dex",
    "joint_interaction_median_dex",
    "joint_increment_mean_dex",
    "joint_increment_median_dex",
    "joint_increment_abs_median_dex",
    "joint_increment_abs_p90_dex",
    "drop_200_250_abs_median_dex",
    "drop_200_250_abs_p90_dex",
    "drop_250_300_abs_median_dex",
    "drop_250_300_abs_p90_dex",
    "drop_300_400_abs_median_dex",
    "drop_300_400_abs_p90_dex",
    "drop_400_500_abs_median_dex",
    "drop_400_500_abs_p90_dex",
    "duplicate_FY_abs_median_dex",
    "duplicate_FY_abs_p90_dex",
    "duplicate_COSMIC_abs_median_dex",
    "duplicate_COSMIC_abs_p90_dex",
    "duplicate_both_abs_median_dex",
    "duplicate_both_abs_p90_dex",
)

_SOURCE_FIELDS = _SOURCE_DIMENSIONS + _COUNTS + (
    "profile_cap_status",
    "queries_with_raw_tokens",
    "queries_with_effective_tokens",
    "raw_query_token_count_sum",
    "profile_equal_token_count_sum_of_within_profile_medians",
    "profile_equal_token_count_median",
    "unique_profile_count_per_query_median",
    "unlocalized_precision_sum_mean",
    "localized_precision_sum_mean",
    "token_neff_median",
    "profile_neff_median",
    "profile_to_token_neff_ratio_median",
    "max_profile_precision_share_median",
    "max_profile_precision_share_p90",
    "concentrated_query_count",
    "concentrated_query_fraction",
    "raw_query_predictive_nis_sum",
    "raw_query_predictive_nis_dof_sum",
    "raw_query_predictive_nis_per_dof",
    "profile_equal_predictive_nis_sum_of_within_profile_medians",
    "profile_equal_predictive_nis_dof_sum_of_within_profile_medians",
    "profile_equal_predictive_nis_per_dof_mean",
    "raw_query_precision_weighted_innovation_defined_queries",
    "profile_equal_precision_weighted_innovation_defined_profiles",
    "profile_equal_precision_weighted_innovation_median_dex",
    "innovation_profile_count",
    "innovation_status",
    "innovation_median_dex",
    "innovation_mad_dex",
    "innovation_robust_sigma_dex",
    "innovation_3sigma_tail_profile_count",
    "innovation_3sigma_tail_profile_fraction",
    "unique_token_count",
    "token_height_interval_profile_count",
    "token_height_interval_status",
    "token_height_interval_median_km",
    "token_height_interval_p90_km",
    "vertical_innovation_correlation_candidate_profiles",
    "vertical_innovation_correlation_defined_profiles",
    "vertical_innovation_correlation_status",
    "vertical_innovation_correlation_median",
    "isolated_increment_mean_dex",
    "isolated_increment_median_dex",
    "joint_source_update_mean_dex",
    "joint_source_update_median_dex",
    "joint_source_update_abs_median_dex",
    "joint_source_update_abs_p90_dex",
    "duplicate_profile_abs_median_dex",
    "duplicate_profile_abs_p90_dex",
    "concentration_defined_profiles",
    "concentration_defined_unique_dates",
    "duplicate_profile_defined_profiles",
    "duplicate_profile_defined_unique_dates",
)

_SOURCE_METRIC_REDUCTION_ORDER = {
    "raw_query_totals": {
        "population": "all_finite_per_query_values_in_the_summary_cell",
        "ordering": (
            "station_date_query_profile_key_lexicographic_then_"
            "deterministic_shard_query_ingestion_order"),
        "within_profile_reducer": "none",
        "cell_reducer": "sum_or_count",
        "fields": [
            "raw_query_token_count_sum",
            "raw_query_predictive_nis_sum",
            "raw_query_predictive_nis_dof_sum",
            "raw_query_predictive_nis_per_dof",
            "raw_query_precision_weighted_innovation_defined_queries",
        ],
    },
    "profile_equal_statistics": {
        "population": "station_date_query_profile",
        "within_profile_reducer": "median_of_finite_per_query_values",
        "cell_reducer_by_field": {
            "profile_equal_token_count_sum_of_within_profile_medians": "sum",
            "profile_equal_token_count_median": "median",
            "profile_equal_predictive_nis_sum_of_within_profile_medians": "sum",
            "profile_equal_predictive_nis_dof_sum_of_within_profile_medians": (
                "sum"),
            "profile_equal_predictive_nis_per_dof_mean": (
                "mean_of_within_profile_median_nis_divided_by_"
                "within_profile_median_positive_dof"),
            "profile_equal_precision_weighted_innovation_defined_profiles": (
                "count"),
            "profile_equal_precision_weighted_innovation_median_dex": (
                "median"),
        },
    },
}

_JOINT_FIELDS = _DIMENSIONS + _COUNTS + (
    "predictive_nis_sum",
    "predictive_nis_dof_sum",
    "predictive_nis_per_dof",
    "joint_increment_mean_dex",
    "joint_increment_median_dex",
    "joint_increment_abs_median_dex",
    "joint_increment_abs_p90_dex",
    "joint_interaction_mean_dex",
    "raw_token_queries",
    "raw_token_low_gain_queries",
    "raw_token_low_gain_fraction",
    "effective_token_queries",
    "effective_token_low_gain_queries",
    "effective_token_low_gain_fraction",
    "raw_token_low_gain_defined_profiles",
    "raw_token_low_gain_defined_unique_dates",
    "raw_token_low_gain_profile_fraction",
    "effective_token_low_gain_defined_profiles",
    "effective_token_low_gain_defined_unique_dates",
    "effective_token_low_gain_profile_fraction",
    "direction_consistency_defined_profiles",
    "direction_consistency_defined_unique_dates",
    "direction_consistency_status",
    "direction_consistency_fraction",
    "joint_source_innovation_sign_defined_profiles",
    "joint_source_innovation_sign_status",
    "joint_source_innovation_same_sign_fraction",
    "joint_source_update_sign_defined_profiles",
    "joint_source_update_sign_status",
    "joint_source_update_same_sign_fraction",
    "source_cancellation_defined_profiles",
    "source_cancellation_defined_unique_dates",
    "source_cancellation_status",
    "source_cancellation_fraction",
    "drop_200_250_abs_median_dex",
    "drop_200_250_abs_p90_dex",
    "drop_250_300_abs_median_dex",
    "drop_250_300_abs_p90_dex",
    "drop_300_400_abs_median_dex",
    "drop_300_400_abs_p90_dex",
    "drop_400_500_abs_median_dex",
    "drop_400_500_abs_p90_dex",
    "duplicate_FY_abs_median_dex",
    "duplicate_FY_abs_p90_dex",
    "duplicate_COSMIC_abs_median_dex",
    "duplicate_COSMIC_abs_p90_dex",
    "duplicate_both_abs_median_dex",
    "duplicate_both_abs_p90_dex",
)


@dataclass(frozen=True)
class BatchShard:
    station: str
    date_utc: str
    batch_id: int
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class CacheInventory:
    cache_dir: Path
    contract: Mapping[str, Any]
    contract_path: Path
    contract_sha256: str
    manifest: Mapping[str, Any]
    cache_acceptance: Mapping[str, Any]
    runtime_contract: Mapping[str, Any]
    identity_validation: Mapping[str, Any]
    batches: tuple[BatchShard, ...]
    declared_batch_identities: Mapping[str, Mapping[str, Any]]
    declared_cache_artifact_identities: Mapping[str, Mapping[str, Any]]
    cache_marker_identity: Mapping[str, Any]


@dataclass
class GroupAccumulator:
    query_points: int = 0
    station_times: set[tuple[str, str, int]] = field(default_factory=set)
    days: set[str] = field(default_factory=set)
    satellite_profiles: set[tuple[str, int]] = field(default_factory=set)
    profiles_by_source: dict[str, set[tuple[str, int]]] = field(
        default_factory=lambda: {source: set() for source in p0b_audit.P0B_SOURCES})
    values: dict[str, dict[tuple[str, str, int], list[float]]] = field(
        default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    token_identities: set[tuple[str, int, int]] = field(default_factory=set)

    def add_value(
            self, name: str, value: float,
            profile_key: tuple[str, str, int]) -> None:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"non-finite aggregation value: {name}")
        self.values.setdefault(name, {}).setdefault(profile_key, []).append(value)

    def add_count(self, name: str, value: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(value)


@dataclass
class AttributionState:
    material_query_count: int = 0
    material_closure_failures: int = 0
    material_closure_max_abs_error: float = 0.0
    raw_token_queries: int = 0
    raw_token_low_gain_queries: int = 0
    effective_token_queries: int = 0
    effective_token_low_gain_queries: int = 0


@dataclass(frozen=True)
class DecisionThresholds:
    minimum_profiles: int
    minimum_days: int
    minimum_defined_profiles: int
    minimum_defined_days: int
    material_bias: float
    material_increment: float
    concentration_share: float
    concentration_neff_ratio: float
    duplicate_median: float
    duplicate_p90: float
    deletion_median: float
    deletion_p90: float
    low_altitude_lower: float
    low_altitude_upper: float
    direction_consistency_fraction: float
    low_gain_fraction: float
    source_cancellation_fraction: float


@dataclass(frozen=True)
class ProfileAggregationRules:
    profile_id_field: str
    signed_reducer: str
    nonnegative_reducer: str
    count_semantics: str
    absolute_transform_order: str
    nonnegative_metric_families: tuple[str, ...]


@dataclass(frozen=True)
class EdgeDiagnosticRules:
    innovation_center: str
    innovation_scale: str
    tail_sigma: float
    zero_mad_tail_rule: str
    joint_source_sign_epsilon: float
    min_vertical_pairs_per_profile: int
    min_profiles_for_vertical_correlation: int
    profile_cap_status: str
    token_identity_key_fields: tuple[str, ...]
    duplicate_token_payload_fields: tuple[str, ...]
    token_height_interval_rule: str
    vertical_innovation_correlation_rule: str
    innovation_population: str
    same_altitude_innovation_reducer: str


def _decision_thresholds(contract: Mapping[str, Any]) -> DecisionThresholds:
    rules = contract["decision_rules"]
    concentration = rules["profile_concentration"]
    duplicate = rules["duplicate_profile_sensitivity"]
    deletion = rules["height_deletion_sensitivity"]
    low = rules["Q5_low_altitude_drift"]
    q2 = rules["Q2_increment_consistency"]
    return DecisionThresholds(
        minimum_profiles=int(rules["eligible_cell_min_station_time_profiles"]),
        minimum_days=int(rules["eligible_cell_min_unique_dates"]),
        minimum_defined_profiles=int(
            rules["eligible_evidence_min_defined_station_time_profiles"]),
        minimum_defined_days=int(
            rules["eligible_evidence_min_defined_unique_dates"]),
        material_bias=float(rules["Q1_bias_origin"]["material_abs_dex_gte"]),
        material_increment=float(
            rules["Q2_increment_consistency"]["material_abs_increment_dex_gte"]),
        concentration_share=float(
            concentration["max_profile_precision_share_gte"]),
        concentration_neff_ratio=float(
            concentration["profile_neff_over_token_neff_lte"]),
        duplicate_median=float(duplicate["median_abs_delta_dex_gte"]),
        duplicate_p90=float(duplicate["p90_abs_delta_dex_gte"]),
        deletion_median=float(deletion["median_abs_delta_dex_gte"]),
        deletion_p90=float(deletion["p90_abs_delta_dex_gte"]),
        low_altitude_lower=float(low["query_altitude_lower_km_inclusive"]),
        low_altitude_upper=float(low["query_altitude_upper_km_exclusive"]),
        direction_consistency_fraction=float(
            q2["direction_consistency_fraction_gte"]),
        low_gain_fraction=float(q2["low_gain_fraction_gte"]),
        source_cancellation_fraction=float(
            q2["source_cancellation_fraction_gte"]),
    )


def _profile_aggregation_rules(
        contract: Mapping[str, Any]) -> ProfileAggregationRules:
    fixed = contract["fixed_aggregation"]["profile_equal_weighting"]
    rules = ProfileAggregationRules(
        profile_id_field=str(fixed["profile_id_field"]),
        signed_reducer=str(fixed["within_profile_signed_reducer"]),
        nonnegative_reducer=str(fixed["within_profile_nonnegative_reducer"]),
        count_semantics=str(fixed["within_profile_count_semantics"]),
        absolute_transform_order=str(fixed["absolute_transform_order"]),
        nonnegative_metric_families=tuple(
            fixed["nonnegative_metric_families"]),
    )
    if rules != ProfileAggregationRules(
            profile_id_field="query_profile_id", signed_reducer="median",
            nonnegative_reducer="median", count_semantics="raw_query_count",
            absolute_transform_order=(
                "absolute_value_per_query_before_within_profile_"
                "nonnegative_reduction"),
            nonnegative_metric_families=(
                "joint_increment_abs", "joint_source_update_abs",
                "height_deletion_abs_delta", "duplicate_profile_abs_delta")):
        raise ValueError("frozen station-time profile equal-weighting rules drifted")
    if fixed.get("enabled") is not True:
        raise ValueError("station-time profile equal weighting is not enabled")
    return rules


def _edge_diagnostic_rules(contract: Mapping[str, Any]) -> EdgeDiagnosticRules:
    fixed = contract["fixed_aggregation"]["edge_diagnostics"]
    rules = EdgeDiagnosticRules(
        innovation_center=str(fixed["innovation_center"]),
        innovation_scale=str(fixed["innovation_scale"]),
        tail_sigma=float(fixed["tail_sigma"]),
        zero_mad_tail_rule=str(fixed["zero_mad_tail_rule"]),
        joint_source_sign_epsilon=float(
            fixed["joint_source_sign_epsilon_dex"]),
        min_vertical_pairs_per_profile=int(
            fixed["min_vertical_pairs_per_profile"]),
        min_profiles_for_vertical_correlation=int(
            fixed["min_profiles_for_vertical_correlation"]),
        profile_cap_status=str(fixed["profile_cap_status"]),
        token_identity_key_fields=tuple(fixed["token_identity_key_fields"]),
        duplicate_token_payload_fields=tuple(
            fixed["duplicate_token_payload_fields"]),
        token_height_interval_rule=str(fixed["token_height_interval_rule"]),
        vertical_innovation_correlation_rule=str(
            fixed["vertical_innovation_correlation_rule"]),
        innovation_population=str(fixed["innovation_population"]),
        same_altitude_innovation_reducer=str(
            fixed["same_altitude_innovation_reducer"]),
    )
    expected = EdgeDiagnosticRules(
        innovation_center="median", innovation_scale="1.4826_mad",
        tail_sigma=3.0,
        zero_mad_tail_rule="tail_if_abs_deviation_gt_zero",
        joint_source_sign_epsilon=0.005,
        min_vertical_pairs_per_profile=3,
        min_profiles_for_vertical_correlation=30,
        profile_cap_status="not_applicable_in_v14",
        token_identity_key_fields=("source", "profile_id", "token_id"),
        duplicate_token_payload_fields=(
            "profile_split", "latitude_deg", "longitude_deg", "altitude_km",
            "relative_hour", "observation_log10_ne", "background_log10_ne",
            "innovation_dex"),
        token_height_interval_rule=(
            "positive_adjacent_unique_altitude_spacing_within_source_profile_"
            "then_profile_median"),
        vertical_innovation_correlation_rule=(
            "pearson_adjacent_unique_altitude_innovation_within_profile_"
            "then_profile_median"),
        innovation_population=(
            "per_satellite_profile_median_then_cell_robust_statistics"),
        same_altitude_innovation_reducer="median",
    )
    if rules != expected:
        raise ValueError("frozen edge diagnostic rules drifted")
    return rules


def _read_json_object(path: Path) -> dict[str, Any]:
    value = p0b_audit.strict_json_loads(
        path.read_bytes(), label=f"P0-B summary input {path}")
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _strict_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be a nonempty normalized POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe or non-normalized artifact path: {value!r}")
    return value


def _artifact_map(records: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{label} artifacts must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{label} artifact entry must be an object")
        path = _strict_relative_path(record.get("path"))
        if path in result:
            raise ValueError(f"duplicate declared artifact path in {label}: {path}")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid artifact size in {label}: {path}")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"invalid artifact SHA256 in {label}: {path}")
        result[path] = {"path": path, "size_bytes": size, "sha256": digest}
    return result


def _verify_identity(path: Path, expected: Mapping[str, Any], root: Path) -> None:
    actual = p0b_audit.artifact_identity(path, root=root)
    if actual != dict(expected):
        raise ValueError(
            f"artifact identity mismatch for {expected.get('path')}: "
            f"expected={dict(expected)}, actual={actual}")


def _verify_batch_shard_identities(
        inventory: CacheInventory, shard: BatchShard) -> None:
    """Recheck one declared triplet without trusting a prior inventory pass."""
    for table_name in ("query", "token", "edge"):
        path = shard.paths[table_name]
        relative = path.relative_to(inventory.cache_dir).as_posix()
        expected = inventory.declared_batch_identities.get(relative)
        if expected is None:
            raise ValueError(f"batch shard is absent from frozen inventory: {relative}")
        _verify_identity(path, expected, inventory.cache_dir)


def _verify_all_declared_batch_identities(inventory: CacheInventory) -> None:
    """Recheck the complete immutable NPZ inventory in deterministic order."""
    for shard in inventory.batches:
        _verify_batch_shard_identities(inventory, shard)


def _resolve_frozen_contract(
        cache_dir: Path, runtime_contract: Mapping[str, Any],
        explicit_contract: Path | None) -> tuple[Mapping[str, Any], Path, str]:
    binding = runtime_contract.get("frozen_p0b_contract")
    if not isinstance(binding, dict):
        raise ValueError("runtime audit_contract lacks frozen_p0b_contract")
    expected_sha = binding.get("sha256")
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)):
        raise ValueError("runtime frozen-contract SHA256 is invalid")
    if explicit_contract is None:
        raw_path = binding.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("runtime frozen-contract path is invalid")
        contract_path = Path(raw_path).resolve()
    else:
        contract_path = explicit_contract.resolve()
    contract, actual_sha = p0b_audit.load_p0b_contract(contract_path)
    if actual_sha != expected_sha:
        raise ValueError("frozen P0-B contract SHA256 differs from runtime binding")
    _validate_summary_contract(contract)
    return contract, contract_path, actual_sha


def _validate_summary_contract(contract: Mapping[str, Any]) -> None:
    fixed = contract.get("fixed_aggregation", {})
    if fixed.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("frozen summary schema version drifted")
    if contract.get("cache_schema", {}).get("cache_schema_version") != 1:
        raise ValueError("frozen raw cache schema version drifted")
    if tuple(fixed.get("dimensions", ())) != _DIMENSIONS:
        raise ValueError("frozen fixed-aggregation dimensions drifted")
    required_counts = tuple(fixed.get("required_counts", ()))
    if required_counts != (
            "query_points", "station_time_profiles", "unique_satellite_profiles"):
        raise ValueError("frozen independent counts drifted")
    artifacts = fixed.get("summary_artifacts", {})
    if tuple(artifacts.get("manifest_artifacts_in_order", ())) != SUMMARY_FILES:
        raise ValueError("frozen summary artifact order drifted")
    if artifacts.get("manifest_filename") != SUMMARY_MANIFEST:
        raise ValueError("frozen summary manifest filename drifted")
    if tuple(artifacts.get("final_acceptance_artifacts_in_order", ())) != (
            SUMMARY_FILES + (SUMMARY_MANIFEST,)):
        raise ValueError("frozen final-acceptance artifact order drifted")
    if artifacts.get("final_acceptance_filename") != FINAL_MARKER:
        raise ValueError("frozen final-acceptance filename drifted")
    if (artifacts.get("manifest_self_listing_allowed") is not False
            or artifacts.get("final_acceptance_written_exclusively_last") is not True):
        raise ValueError("frozen completion ordering semantics drifted")
    _profile_aggregation_rules(contract)
    _edge_diagnostic_rules(contract)
    coordinate = contract["coordinate_enrichment"]
    if (coordinate.get("fixed_aggregation_time_coordinate")
            != "aacgm_mlt_hour"
            or coordinate.get("aacgm", {}).get("status") != "required"
            or tuple(coordinate.get("aacgm", {}).get("fields", ()))
            != ("aacgm_latitude_deg", "aacgm_mlt_hour")):
        raise ValueError("frozen AACGM aggregation contract drifted")
    partitioning = contract.get("partitioning", {})
    if (partitioning.get("cache_completion_marker") != CACHE_MARKER
            or partitioning.get("final_completion_marker") != FINAL_MARKER):
        raise ValueError("frozen cache/final marker names drifted")

    rules = contract["decision_rules"]
    thresholds = _decision_thresholds(contract)
    if (thresholds.minimum_defined_profiles != 30
            or thresholds.minimum_defined_days != 2):
        raise ValueError("frozen defined profile/date evidence gates drifted")
    for name, value in (
            ("direction_consistency_fraction_gte",
             thresholds.direction_consistency_fraction),
            ("low_gain_fraction_gte", thresholds.low_gain_fraction),
            ("source_cancellation_fraction_gte",
             thresholds.source_cancellation_fraction)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"frozen Q2 fraction threshold is invalid: {name}")
    q1 = rules.get("Q1_bias_origin", {})
    if tuple(q1.get("allowed_origin_labels", ())) != (
            "raw_IRI_preexisting", "background_M00_shift",
            "FY_isolated_increment", "COSMIC_isolated_increment",
            "joint_interaction", "none_above_threshold"):
        raise ValueError("frozen Q1 origin labels drifted")
    if (rules.get("Q4_maximum_terminal_status") != "insufficient_evidence"
            or rules.get(
                "Q6_required_terminal_status_until_observation_eligibility")
            != "insufficient_evidence"):
        raise ValueError("frozen Q4/Q6 status caps drifted")
    question_boundaries = {
        row.get("id"): {
            key: row.get(key)
            for key in ("interpretation_boundary", "supports", "does_not_support")
        }
        for row in contract.get("six_questions", {}).get("questions", ())
    }
    if question_boundaries != p0b_audit.P0B_QUESTION_INTERPRETATION_BOUNDARIES:
        raise ValueError("frozen question interpretation boundaries drifted")
    acceptance = contract.get("acceptance", {})
    if any(acceptance.get(key) is not True for key in (
            "defined_profile_and_date_gates_enforced",
            "question_interpretation_boundaries_required",
            "q5_dual_routing_required")):
        raise ValueError("frozen attribution hardening acceptance flags drifted")


def _validate_runtime_identities(
        runtime: Mapping[str, Any], manifest: Mapping[str, Any],
        cache_marker: Mapping[str, Any],
        contract: Mapping[str, Any], *, contract_path: Path,
        repo_root: Path) -> dict[str, Any]:
    """Close runtime Git/checkpoint/input identities across all cache ledgers."""
    if (runtime.get("audit_schema_version")
            != p0b_audit.P0B_AUDIT_SCHEMA_VERSION
            or runtime.get("status") != "runtime_contract_bound"):
        raise ValueError("runtime audit_contract is not a bound P0-B v2 contract")
    git = runtime.get("git")
    identities = runtime.get("runtime_identities")
    if not isinstance(git, dict) or not isinstance(identities, dict):
        raise ValueError("runtime contract lacks Git/runtime identities")
    frozen_vc = contract["version_control"]
    if git.get("status") != "computed":
        raise ValueError("runtime Git identity was not computed")
    if git.get("branch") != frozen_vc["required_branch"]:
        raise ValueError("runtime Git branch differs from the frozen contract")
    if git.get("required_implementation_tag") != frozen_vc[
            "required_implementation_tag"]:
        raise ValueError("runtime Git implementation tag differs from contract")
    if (git.get("implementation_tag_type") != "tag"
            or git.get("head_equals_implementation_tag_commit") is not True
            or git.get("head") != git.get("implementation_tag_commit")):
        raise ValueError("runtime HEAD is not the frozen annotated tag commit")
    tag_object_sha = git.get("implementation_tag_object_sha")
    if not isinstance(tag_object_sha, str) or not tag_object_sha:
        raise ValueError("runtime annotated-tag object SHA is absent")
    if identities.get("git_head") != git.get("head") or manifest.get("git") != git:
        raise ValueError("runtime and manifest Git identities differ")
    git_after = manifest.get("git_after")
    if not isinstance(git_after, dict):
        raise ValueError("manifest lacks the final Git provenance recheck")
    for key in (
            "head", "branch", "implementation_tag_commit",
            "implementation_tag_object_sha"):
        if git_after.get(key) != git.get(key):
            raise ValueError(f"final Git provenance differs at {key}")
    if cache_marker.get("implementation_tag_object_sha") != tag_object_sha:
        raise ValueError("cache marker annotated-tag object SHA differs")
    for key, expected in (
            ("git_head", git.get("head")),
            ("implementation_tag_commit", git.get("implementation_tag_commit"))):
        if key in cache_marker and cache_marker[key] != expected:
            raise ValueError(f"cache marker {key} differs from runtime contract")

    expected_checkpoint = contract["identity"]["candidate_checkpoint_sha256"]
    runtime_checkpoint = runtime.get("candidate_checkpoint", {}).get("sha256")
    identity_checkpoint = identities.get("checkpoint_path", {}).get("sha256")
    checkpoint_values = (
        runtime_checkpoint,
        identity_checkpoint,
        manifest.get("checkpoint_sha256_before"),
        manifest.get("checkpoint_sha256_after"),
        cache_marker.get("checkpoint_sha256_before"),
        cache_marker.get("checkpoint_sha256_after"),
    )
    if any(value != expected_checkpoint for value in checkpoint_values):
        raise ValueError("checkpoint identity differs across P0-B cache ledgers")

    python_environment = runtime.get("python_environment")
    environment_values = (
        python_environment,
        identities.get("python_environment"),
        manifest.get("python_environment"),
    )
    required_environment_fields = {
        "executable", "version", "implementation", "numpy_version",
        "torch_version", "torch_cuda_version",
    }
    if (not isinstance(python_environment, dict)
            or set(python_environment) != required_environment_fields
            or any(value != python_environment for value in environment_values)):
        raise ValueError("Python environment differs across P0-B cache ledgers")
    if os.path.normcase(str(Path(python_environment["executable"]).resolve())) != (
            os.path.normcase(str(p0b_audit.P0B_PYTHON_EXECUTABLE.resolve()))):
        raise ValueError("P0-B cache was not produced by the frozen pytorch_cpu Python")
    for field in ("version", "implementation", "numpy_version", "torch_version"):
        if not isinstance(python_environment[field], str) or not python_environment[field]:
            raise ValueError(f"invalid frozen Python environment field: {field}")
    if (python_environment["torch_cuda_version"] is not None
            and not isinstance(python_environment["torch_cuda_version"], str)):
        raise ValueError("invalid frozen Torch CUDA environment field")

    expected_split = contract["identity"]["date_split_sha256"]
    split_values = (
        runtime.get("date_split", {}).get("sha256"),
        identities.get("date_split_path", {}).get("sha256"),
        manifest.get("date_split_sha256"),
    )
    if any(value != expected_split for value in split_values):
        raise ValueError("date-split identity differs across P0-B cache ledgers")
    input_identity = identities.get("input_data_sha256")
    if not isinstance(input_identity, dict) or not input_identity:
        raise ValueError("runtime input_data_sha256 is absent")
    if manifest.get("input_data_sha256") != input_identity:
        raise ValueError("manifest input_data_sha256 differs from runtime contract")
    if ("input_data_sha256" in cache_marker
            and cache_marker["input_data_sha256"] != input_identity):
        raise ValueError("cache marker input_data_sha256 differs from runtime")
    if input_identity.get("date_split_manifest") != expected_split:
        raise ValueError("input_data_sha256 date split differs from frozen identity")
    if cache_marker.get("locked_test_query_rows_written") != 0:
        raise ValueError("cache marker reports locked-test query output")
    if cache_marker.get("satellite_token_partition") != "train":
        raise ValueError("cache marker token partition is not train-only")
    raw_registry = manifest.get("raw_registry")
    if not isinstance(raw_registry, dict):
        raise ValueError("manifest lacks the frozen raw-query registry")
    query_partition_counts = raw_registry.get("date_partition_counts")
    if (not isinstance(query_partition_counts, dict)
            or set(query_partition_counts) != set(p0b_audit.P0B_QUERY_SPLITS)
            or any(not isinstance(value, int) or value <= 0
                   for value in query_partition_counts.values())):
        raise ValueError("manifest query-partition counts are invalid")
    if sum(query_partition_counts.values()) != manifest.get("counts", {}).get(
            "query_rows"):
        raise ValueError("query-partition counts differ from manifest query rows")
    if cache_marker.get("query_partition_counts") != query_partition_counts:
        raise ValueError("cache marker query-partition counts differ")

    profile_metadata = runtime.get("satellite_profile_metadata")
    if (not isinstance(profile_metadata, dict)
            or manifest.get("satellite_profile_metadata") != profile_metadata):
        raise ValueError("satellite profile metadata differs across cache ledgers")
    satellite_profile_counts = {"train": {}, "development": {}}
    for source in p0b_audit.P0B_SOURCES:
        source_metadata = profile_metadata.get(source)
        if (not isinstance(source_metadata, dict)
                or source_metadata.get("token_partition") != "train"
                or source_metadata.get("development_metadata_only") is not True):
            raise ValueError(f"{source} profile metadata semantics drifted")
        train_count = source_metadata.get("train_unique_profiles")
        development_count = source_metadata.get("development_unique_profiles")
        if (not isinstance(train_count, int) or train_count <= 0
                or not isinstance(development_count, int)
                or development_count <= 0):
            raise ValueError(f"{source} profile counts are invalid")
        satellite_profile_counts["train"][source] = train_count
        satellite_profile_counts["development"][source] = development_count
    if cache_marker.get("satellite_profile_counts") != satellite_profile_counts:
        raise ValueError("cache marker satellite profile counts differ")

    def closed_identity(name: str) -> Mapping[str, Any]:
        value = runtime.get(name)
        ledgers = (
            value, identities.get(name), manifest.get(name), cache_marker.get(name))
        if not isinstance(value, dict) or any(candidate != value for candidate in ledgers):
            raise ValueError(f"{name} differs across P0-B cache ledgers")
        return value

    def valid_sha256(value: Any) -> bool:
        return (isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value))

    train_allowlist_identity = closed_identity("train_allowlist_identity")
    if set(train_allowlist_identity) != set(p0b_audit.P0B_SOURCES):
        raise ValueError("train allowlist identity source inventory is not exact")
    for source in p0b_audit.P0B_SOURCES:
        record = train_allowlist_identity[source]
        metadata = profile_metadata[source]
        if (not isinstance(record, dict)
                or set(record) != {
                    "profile_count", "profile_id_sha256", "profile_index_sha256"}
                or record.get("profile_count") != satellite_profile_counts[
                    "train"][source]
                or record.get("profile_count") != metadata.get(
                    "train_unique_profiles")
                or record.get("profile_id_sha256") != metadata.get(
                    "train_profile_id_sha256")
                or record.get("profile_index_sha256") != metadata.get(
                    "profile_index_sha256")
                or not valid_sha256(record.get("profile_id_sha256"))
                or not valid_sha256(record.get("profile_index_sha256"))):
            raise ValueError(f"invalid {source} train allowlist identity")

    declared_token_directory_identity = closed_identity(
        "train_token_directory_identity")
    train_token_directory_identity = (
        p0b_audit.validate_train_token_directory_identity(
            declared_token_directory_identity))
    if train_token_directory_identity != declared_token_directory_identity:
        raise ValueError("train token-directory identity is not canonical")
    for source in p0b_audit.P0B_SOURCES:
        record = train_token_directory_identity[source]
        if record["unique_profiles"] != train_allowlist_identity[
                source]["profile_count"]:
            raise ValueError(
                f"{source} train token-directory profiles differ from allowlist")

    column_access = runtime.get("ISR_column_access_audit")
    column_access_values = (
        column_access,
        identities.get("ISR_column_access_audit"),
        manifest.get("ISR_column_access_audit"),
        cache_marker.get("ISR_column_access_audit"),
    )
    if (not isinstance(column_access, dict)
            or column_access.get("status") != "pass"
            or column_access.get("excluded_density_columns_materialized") != 0
            or any(value != column_access for value in column_access_values)):
        raise ValueError(
            "ISR column-access audit differs across P0-B cache ledgers")

    preflight_dependency = runtime.get("preflight_dependency")
    preflight_values = (
        preflight_dependency,
        manifest.get("preflight_dependency"),
        manifest.get("preflight_dependency_after"),
        cache_marker.get("preflight_dependency"),
    )
    if (not isinstance(preflight_dependency, dict)
            or any(value != preflight_dependency for value in preflight_values)):
        raise ValueError(
            "preflight dependency differs across full-audit cache ledgers")
    preflight_sha = preflight_dependency.get("sha256")
    preflight_size = preflight_dependency.get("size_bytes")
    if (not isinstance(preflight_dependency.get("path"), str)
            or not preflight_dependency["path"]
            or not isinstance(preflight_sha, str) or len(preflight_sha) != 64
            or any(char not in "0123456789abcdef" for char in preflight_sha)
            or isinstance(preflight_size, bool) or not isinstance(preflight_size, int)
            or preflight_size <= 0):
        raise ValueError("full-audit preflight dependency identity is invalid")
    for name, expected in (
            ("train_allowlist_identity", train_allowlist_identity),
            ("train_token_directory_identity", train_token_directory_identity),
            ("ISR_column_access_audit", column_access)):
        if preflight_dependency.get(name) != expected:
            raise ValueError(
                f"preflight dependency {name} differs from full-audit identity")

    source_provenance = runtime.get("source_provenance")
    p0a_dependency = runtime.get("p0a_dependency")
    if not isinstance(source_provenance, dict) or not isinstance(p0a_dependency, dict):
        raise ValueError("runtime contract lacks source/P0-A provenance")
    expected_source_paths = (
        Path(p0a_dependency.get("contract_path", "")).resolve(),
        contract_path.resolve(),
        *(repo_root / relative for relative in contract[
            "version_control"]["critical_tracked_paths"]),
    )
    expected_by_case = {
        os.path.normcase(str(path.resolve())): str(path.resolve())
        for path in expected_source_paths
    }
    actual_by_case = {
        os.path.normcase(str(Path(path).resolve())): (path, value)
        for path, value in source_provenance.items()
    }
    if len(actual_by_case) != len(source_provenance):
        raise ValueError("runtime source provenance has duplicate resolved paths")
    if set(actual_by_case) != set(expected_by_case):
        raise ValueError("runtime source provenance path inventory is not exact")
    normalized_source_provenance: dict[str, dict[str, Any]] = {}
    for normalized, expected_path in expected_by_case.items():
        declared_path, declared = actual_by_case[normalized]
        if not isinstance(declared, dict):
            raise ValueError(f"invalid source provenance record: {declared_path}")
        digest = declared.get("sha256")
        size = declared.get("size_bytes")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError(f"invalid source provenance identity: {declared_path}")
        normalized_source_provenance[expected_path] = {
            "sha256": digest, "size_bytes": size}
    runner_path = str((repo_root / "isr_evaluation/audit_m2w2_error_chain.py").resolve())
    if identities.get("audit_code_sha256") != normalized_source_provenance[
            runner_path]["sha256"]:
        raise ValueError("runtime audit-code SHA differs from source provenance")
    coordinate_contract = contract["coordinate_enrichment"]
    expected_coordinate = {
        "geographic": {
            "status": "computed",
            "fields": list(coordinate_contract["geographic"]["fields"]),
        },
        "aacgm": {
            "status": "computed",
            "package": str(coordinate_contract["aacgm"]["package"]),
            "version": str(coordinate_contract["aacgm"]["version"]),
            "method": str(coordinate_contract["aacgm"]["method"]),
            "fields": list(coordinate_contract["aacgm"]["fields"]),
        },
        "qd": {
            "status": "unavailable",
            "package": str(coordinate_contract["qd"]["package"]),
            "reason": str(coordinate_contract["qd"]["reason"]),
            "proxy_substitution_used": False,
        },
        "fixed_aggregation_time_coordinate": "aacgm_mlt_hour",
    }
    coordinate_values = (
        runtime.get("coordinate_enrichment"),
        identities.get("coordinate_enrichment"),
        manifest.get("coordinate_enrichment"),
        cache_marker.get("coordinate_enrichment"),
    )
    if any(value != expected_coordinate for value in coordinate_values):
        raise ValueError(
            "coordinate enrichment differs across frozen P0-B cache ledgers")
    expected_profile_cap = contract[
        "precision_innovation_gain_semantics"]["profile_cap_status"]
    profile_cap_values = (
        runtime.get("profile_cap_status"), identities.get("profile_cap_status"),
        manifest.get("profile_cap_status"), cache_marker.get("profile_cap_status"),
    )
    if any(value != expected_profile_cap for value in profile_cap_values):
        raise ValueError("profile-cap status differs across P0-B cache ledgers")
    return {
        "git_head": str(git["head"]),
        "implementation_tag": str(git["required_implementation_tag"]),
        "implementation_tag_object_sha": tag_object_sha,
        "implementation_tag_commit": str(git["implementation_tag_commit"]),
        "checkpoint_sha256": expected_checkpoint,
        "date_split_sha256": expected_split,
        "input_data_sha256": dict(input_identity),
        "query_partition_counts": dict(query_partition_counts),
        "satellite_profile_counts": satellite_profile_counts,
        "python_environment": dict(python_environment),
        "source_provenance": normalized_source_provenance,
        "train_allowlist_identity": dict(train_allowlist_identity),
        "train_token_directory_identity": dict(train_token_directory_identity),
        "ISR_column_access_audit": dict(column_access),
        "preflight_dependency": dict(preflight_dependency),
        "cache_marker_binds_runtime_contract_by_sha256": True,
        "coordinate_enrichment": expected_coordinate,
        "profile_cap_status": expected_profile_cap,
    }


def validate_cache_inventory(
        cache_dir: str | os.PathLike[str],
        contract_path: str | os.PathLike[str] | None = None) -> CacheInventory:
    """Validate completion markers and return the exact declared batch inventory."""
    root = Path(cache_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    final_marker = root / FINAL_MARKER
    if final_marker.exists():
        raise FileExistsError(f"P0-B final acceptance already exists: {final_marker}")

    marker_path = root / CACHE_MARKER
    manifest_path = root / "manifest.json"
    runtime_contract_path = root / "audit_contract.json"
    failure_path = root / "failure_ledger.json"
    for required in (marker_path, manifest_path, runtime_contract_path, failure_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    acceptance = _read_json_object(marker_path)
    manifest = _read_json_object(manifest_path)
    runtime_contract = _read_json_object(runtime_contract_path)
    failure = _read_json_object(failure_path)
    if acceptance.get("audit_schema_version") != p0b_audit.P0B_AUDIT_SCHEMA_VERSION:
        raise ValueError("cache acceptance schema version is not P0-B v1")
    if (acceptance.get("status") != "cache_pass_not_final_p0b_acceptance"
            or acceptance.get("completion_marker") is not True):
        raise ValueError("p0b_cache_acceptance is not a completed pass")
    if manifest.get("audit_schema_version") != p0b_audit.P0B_AUDIT_SCHEMA_VERSION:
        raise ValueError("manifest schema version is not P0-B v1")
    if (manifest.get("status")
            != "cache_complete_provisional_not_final_p0b_acceptance"
            or manifest.get("preflight_only") is not False):
        raise ValueError("manifest is not a completed full-audit pass")
    if (acceptance.get("full_p0b_audit_complete") is not False
            or acceptance.get("final_p0b_acceptance_written") is not False
            or acceptance.get("fixed_summary_artifact_written") is not False
            or acceptance.get("six_question_artifact_written") is not False):
        raise ValueError("cache marker incorrectly claims final P0-B attribution")
    if failure.get("status") != "no_failures" or failure.get("entries") != []:
        raise ValueError("failure ledger is not empty")
    if manifest.get("locked_test_query_rows_written") != 0:
        raise ValueError("manifest reports locked-test query output")
    if manifest.get("satellite_token_partition") != "train":
        raise ValueError("manifest token partition is not train-only")

    contract, resolved_contract_path, contract_sha = _resolve_frozen_contract(
        root, runtime_contract,
        None if contract_path is None else Path(contract_path))
    acceptance_map = _artifact_map(acceptance.get("artifacts"), "cache acceptance")
    manifest_map = _artifact_map(manifest.get("artifacts"), "manifest")
    for required_name in ("audit_contract.json", "manifest.json", "failure_ledger.json"):
        if required_name not in acceptance_map:
            raise ValueError(f"cache acceptance does not declare {required_name}")
    for path, identity in acceptance_map.items():
        declared = root / PurePosixPath(path)
        if not declared.is_file():
            raise FileNotFoundError(declared)
        _verify_identity(declared, identity, root)
    identity_validation = _validate_runtime_identities(
        runtime_contract, manifest, acceptance, contract,
        contract_path=resolved_contract_path, repo_root=REPO_ROOT)

    actual_batch_paths: dict[str, Path] = {}
    batches_root = root / "batches"
    if not batches_root.is_dir():
        raise FileNotFoundError(batches_root)
    for item in batches_root.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        if not _BATCH_RE.fullmatch(relative):
            raise ValueError(f"undeclared or malformed batch artifact path: {relative}")
        if relative in actual_batch_paths:
            raise ValueError(f"duplicate batch artifact path: {relative}")
        actual_batch_paths[relative] = item
    if not actual_batch_paths:
        raise ValueError("cache contains no batch NPZ artifacts")

    manifest_batch_paths = {
        path for path in manifest_map if path.startswith("batches/")}
    acceptance_batch_paths = {
        path for path in acceptance_map if path.startswith("batches/")}
    actual_paths = set(actual_batch_paths)
    if set(manifest_map) != manifest_batch_paths:
        raise ValueError("manifest declares a non-batch artifact")
    if manifest_batch_paths != actual_paths:
        raise ValueError(
            "manifest batch inventory differs from disk: "
            f"missing={sorted(manifest_batch_paths - actual_paths)}, "
            f"undeclared={sorted(actual_paths - manifest_batch_paths)}")
    if acceptance_batch_paths != actual_paths:
        raise ValueError(
            "cache acceptance batch inventory differs from disk: "
            f"missing={sorted(acceptance_batch_paths - actual_paths)}, "
            f"undeclared={sorted(actual_paths - acceptance_batch_paths)}")
    for relative, path in actual_batch_paths.items():
        if manifest_map[relative] != acceptance_map[relative]:
            raise ValueError(f"manifest/cache-acceptance identity differs: {relative}")
        _verify_identity(path, manifest_map[relative], root)

    by_partition: dict[tuple[str, str, int], dict[str, Path]] = {}
    for relative, path in actual_batch_paths.items():
        match = _BATCH_RE.fullmatch(relative)
        assert match is not None
        identity = (
            match.group("station"), match.group("date"),
            int(match.group("batch")))
        table_paths = by_partition.setdefault(identity, {})
        table = match.group("table")
        if table in table_paths:
            raise ValueError(f"duplicate {table} shard for partition {identity}")
        table_paths[table] = path
    expected_tables = set(contract["partitioning"]["tables_per_partition"])
    if expected_tables != {"query", "token", "edge"}:
        raise ValueError("frozen table triplet is not query/token/edge")
    shards: list[BatchShard] = []
    for identity, paths in sorted(by_partition.items()):
        if set(paths) != expected_tables:
            raise ValueError(
                f"incomplete table triplet for {identity}: "
                f"missing={sorted(expected_tables - set(paths))}")
        shards.append(BatchShard(*identity, paths=dict(paths)))

    counts = manifest.get("counts")
    acceptance_counts = acceptance.get("counts")
    if not isinstance(counts, dict) or counts != acceptance_counts:
        raise ValueError("manifest and cache-acceptance counts differ")
    if counts.get("batches") != len(shards):
        raise ValueError("declared batch count differs from disk inventory")
    for name in ("query_rows", "token_rows", "edge_rows", "batches"):
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid manifest count: {name}")

    return CacheInventory(
        cache_dir=root,
        contract=contract,
        contract_path=resolved_contract_path,
        contract_sha256=contract_sha,
        manifest=manifest,
        cache_acceptance=acceptance,
        runtime_contract=runtime_contract,
        identity_validation=identity_validation,
        batches=tuple(shards),
        declared_batch_identities=manifest_map,
        declared_cache_artifact_identities=acceptance_map,
        cache_marker_identity=p0b_audit.artifact_identity(marker_path, root=root),
    )


def _load_npz_no_pickle(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if len(loaded.files) != len(set(loaded.files)):
                raise ValueError(f"duplicate NPZ member name: {path}")
            return {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    except ValueError as exc:
        raise ValueError(f"cannot load no-pickle NPZ {path}: {exc}") from exc


def validate_batch_tables(
        shard: BatchShard, contract: Mapping[str, Any]) -> tuple[
            dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray],
            Mapping[str, Any]]:
    """Load and validate one exact query/token/edge triplet."""
    query = _load_npz_no_pickle(shard.paths["query"])
    token = _load_npz_no_pickle(shard.paths["token"])
    edge = _load_npz_no_pickle(shard.paths["edge"])
    tables = {"query": query, "token": token, "edge": edge}
    schemas = p0b_audit.contract_table_schemas(contract)
    for table_name, table in tables.items():
        table_contract = schemas[table_name]
        p0b_audit.validate_table_contract_fields(
            table, table_contract, table_name)
        for scope_name, expected in (
                ("station", shard.station),
                ("date_utc", shard.date_utc),
                ("batch_id", shard.batch_id)):
            values = np.asarray(table[scope_name])
            if len(values) and (len(np.unique(values)) != 1
                                or str(values[0]) != str(expected)):
                raise ValueError(
                    f"{table_name} {scope_name} differs from its shard path")
    validation = p0b_audit.validate_audit_tables(
        query, token, edge,
        atol=float(contract["acceptance"]["closure_atol"]),
        rtol=float(contract["acceptance"]["closure_rtol"]),
        contract=contract,
    )
    return query, token, edge, validation


def _bin_label(
        value: float, edges: Sequence[float], labels: Sequence[str], name: str,
        *, final_inclusive: bool) -> str:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite {name}")
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if value >= lower and (
                value < upper
                or (final_inclusive and index == len(labels) - 1
                    and value <= upper)):
            return labels[index]
    raise ValueError(f"{name}={value} is outside frozen aggregation edges")


def _group_key(query: Mapping[str, np.ndarray], row: int, contract: Mapping[str, Any]) -> tuple[str, ...]:
    fixed = contract["fixed_aggregation"]
    mlt_edges = tuple(map(float, fixed["MLT_edges_hour"]))
    altitude_edges = tuple(map(float, fixed["query_altitude_edges_km"]))
    kp_edges = tuple(map(float, fixed["Kp_edges"]))
    mlt_labels = tuple(
        f"[{int(lower):02d},{int(upper):02d})h"
        for lower, upper in zip(mlt_edges[:-1], mlt_edges[1:]))
    altitude_labels = tuple(
        f"[{int(lower)},{int(upper)}{']' if index == len(altitude_edges) - 2 else ')'}km"
        for index, (lower, upper) in enumerate(zip(altitude_edges[:-1], altitude_edges[1:])))
    kp_labels = tuple(
        f"[{lower:g},{upper:g}{']' if index == len(kp_edges) - 2 else ')'}"
        for index, (lower, upper) in enumerate(zip(kp_edges[:-1], kp_edges[1:])))
    coverage = str(query["coverage_code"][row])
    if coverage not in p0b_audit.P0B_COVERAGE_CODES:
        raise ValueError(f"unknown effective coverage code: {coverage}")
    split = str(query["query_split"][row])
    if split not in p0b_audit.P0B_QUERY_SPLITS:
        raise ValueError(f"forbidden query split: {split}")
    station = str(query["station"][row])
    allowed_stations = tuple(contract["data_scope"]["ISR_queries"]["stations"])
    if station not in allowed_stations:
        raise ValueError(f"unknown station: {station}")
    solar_contract = fixed["solar_regime"]
    if (solar_contract.get("source_field") != "cos_sza"
            or solar_contract.get("day_rule") != "cos_sza>0"
            or solar_contract.get("twilight_rule")
            != "cos(108deg)<=cos_sza<=0"
            or solar_contract.get("night_rule") != "cos_sza<cos(108deg)"):
        raise ValueError("frozen solar-regime rules drifted")
    cos_sza = float(query["cos_sza"][row])
    twilight_lower = float(solar_contract["cos_108deg"])
    if not np.isfinite(cos_sza) or not -1.0 <= cos_sza <= 1.0:
        raise ValueError("cos_sza is non-finite or outside [-1,1]")
    solar_regime = (
        "day" if cos_sza > 0.0
        else "twilight" if cos_sza >= twilight_lower
        else "night")
    return (
        station,
        split,
        _bin_label(
            query[contract["coordinate_enrichment"][
                "fixed_aggregation_time_coordinate"]][row],
            mlt_edges, mlt_labels, "aacgm_mlt_hour", final_inclusive=False),
        solar_regime,
        _bin_label(
            query["altitude_km"][row], altitude_edges, altitude_labels,
            "altitude_km", final_inclusive=True),
        _bin_label(
            query["kp"][row], kp_edges, kp_labels, "kp",
            final_inclusive=True),
        coverage,
    )


def _fixed_group_sort_key(
        key: tuple[str, ...], contract: Mapping[str, Any]) -> tuple[Any, ...]:
    station_order = {
        name: index for index, name in enumerate(
            contract["data_scope"]["ISR_queries"]["stations"])
    }
    split_order = {
        name: index for index, name in enumerate(p0b_audit.P0B_QUERY_SPLITS)
    }
    coverage_order = {
        name: index for index, name in enumerate(p0b_audit.P0B_COVERAGE_CODES)
    }
    solar_order = {name: index for index, name in enumerate(
        ("night", "twilight", "day"))}
    source_order = {
        name: index for index, name in enumerate(p0b_audit.P0B_SOURCES)
    }

    def lower(label: str) -> float:
        return float(label[1:].split(",", 1)[0])

    result: tuple[Any, ...] = (
        station_order[key[0]], split_order[key[1]], lower(key[2]),
        solar_order[key[3]], lower(key[4]), lower(key[5]),
        coverage_order[key[6]])
    if len(key) == len(_SOURCE_DIMENSIONS):
        result += (source_order[key[len(_DIMENSIONS)]],)
    return result


def _profile_metric_map(
        accumulator: GroupAccumulator, name: str) -> dict[
            tuple[str, str, int], float]:
    """Reduce nested height/query rows before any equal-profile cell statistic."""
    by_profile = accumulator.values.get(name, {})
    return {
        profile_key: float(np.median(np.asarray(values, dtype=np.float64)))
        for profile_key, values in sorted(by_profile.items())
    }


def _defined_profile_support(
        accumulator: GroupAccumulator, *names: str) -> tuple[int, int]:
    """Count unique station-date-profile identities defined for every metric."""
    if not names:
        raise ValueError("defined support requires at least one metric")
    keys = set(_profile_metric_map(accumulator, names[0]))
    for name in names[1:]:
        keys.intersection_update(_profile_metric_map(accumulator, name))
    return len(keys), len({profile_key[1] for profile_key in keys})


def _metric(accumulator: GroupAccumulator, name: str) -> np.ndarray:
    return np.asarray(
        list(_profile_metric_map(accumulator, name).values()), dtype=np.float64)


def _raw_query_metric(accumulator: GroupAccumulator, name: str) -> np.ndarray:
    """Return finite per-query values without the profile-equal reduction."""
    by_profile = accumulator.values.get(name, {})
    return np.asarray([
        value
        for profile_key in sorted(by_profile)
        for value in by_profile[profile_key]
    ], dtype=np.float64)


def _profile_ratio_metric(
        accumulator: GroupAccumulator, numerator: str,
        denominator: str) -> np.ndarray:
    numerators = _profile_metric_map(accumulator, numerator)
    denominators = _profile_metric_map(accumulator, denominator)
    if set(numerators) != set(denominators):
        raise ValueError(
            f"profile support differs for paired metrics: {numerator}/{denominator}")
    return np.asarray([
        numerators[key] / denominators[key]
        for key in sorted(numerators) if denominators[key] > 0.0
    ], dtype=np.float64)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else 0.0


def _p90(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.9, method="linear")) if len(values) else 0.0


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _defined_statistic(
        values: np.ndarray, statistic: str = "mean") -> tuple[int, str, float | None]:
    count = int(len(values))
    if not count:
        return 0, "insufficient_data", None
    if statistic == "mean":
        value = _mean(values)
    elif statistic == "median":
        value = _median(values)
    elif statistic == "p90":
        value = _p90(values)
    else:
        raise ValueError(f"unsupported defined statistic: {statistic}")
    return count, "computed", value


def _base_row(key: tuple[str, ...], accumulator: GroupAccumulator) -> dict[str, Any]:
    return {
        **dict(zip(_DIMENSIONS, key[:len(_DIMENSIONS)])),
        "query_points": int(accumulator.query_points),
        "station_time_profiles": int(len(accumulator.station_times)),
        "unique_days": int(len(accumulator.days)),
        "unique_satellite_profiles": int(len(accumulator.satellite_profiles)),
    }


def _regime_row(key: tuple[str, ...], acc: GroupAccumulator) -> dict[str, Any]:
    row = _base_row(key, acc)
    row.update({
        "FY_unique_satellite_profiles": len(acc.profiles_by_source["FY"]),
        "COSMIC_unique_satellite_profiles": len(acc.profiles_by_source["COSMIC"]),
    })
    for mode, metric_name in (
            ("raw_iri", "raw_error"), ("M00", "M00_error"),
            ("M10", "M10_error"), ("M01", "M01_error"),
            ("M11", "M11_error")):
        values = _metric(acc, metric_name)
        row[f"{mode}_bias_mean_dex"] = _mean(values)
        row[f"{mode}_rmse_dex"] = _rmse(values)
        if mode in ("raw_iri", "M00"):
            row[f"{mode}_bias_median_dex"] = _median(values)
    for output_name, metric_name in (
            ("background_increment_mean_dex", "background_increment"),
            ("joint_update_FY_mean_dex", "joint_update_FY"),
            ("joint_update_COSMIC_mean_dex", "joint_update_COSMIC"),
            ("joint_interaction_mean_dex", "joint_interaction")):
        row[output_name] = _mean(_metric(acc, metric_name))
    joint = _metric(acc, "joint_increment")
    joint_abs = _metric(acc, "joint_increment_abs")
    row.update({
        "background_increment_median_dex": _median(
            _metric(acc, "background_increment")),
        "isolated_increment_FY_median_dex": _median(
            _metric(acc, "isolated_increment_FY")),
        "isolated_increment_COSMIC_median_dex": _median(
            _metric(acc, "isolated_increment_COSMIC")),
        "joint_interaction_median_dex": _median(
            _metric(acc, "joint_interaction")),
    })
    row.update({
        "joint_increment_mean_dex": _mean(joint),
        "joint_increment_median_dex": _median(joint),
        "joint_increment_abs_median_dex": _median(joint_abs),
        "joint_increment_abs_p90_dex": _p90(joint_abs),
    })
    for short_name in (
            "drop_200_250", "drop_250_300", "drop_300_400", "drop_400_500",
            "duplicate_FY", "duplicate_COSMIC", "duplicate_both"):
        values = _metric(acc, f"{short_name}_abs")
        row[f"{short_name}_abs_median_dex"] = _median(values)
        row[f"{short_name}_abs_p90_dex"] = _p90(values)
    return row


def _token_profile_diagnostics(
        token_identities: set[tuple[str, int, int]],
        registry: Mapping[tuple[str, int, int], Mapping[str, Any]],
        rules: EdgeDiagnosticRules) -> dict[str, Any]:
    by_profile: dict[tuple[str, int], list[tuple[float, int, float]]] = {}
    for identity in sorted(token_identities):
        record = registry[identity]
        innovation = record["innovation_dex"]
        if innovation is None:
            raise ValueError(
                f"token used by a summary cell has no edge innovation: {identity}")
        source, profile_id, token_id = identity
        by_profile.setdefault((source, profile_id), []).append((
            float(record["altitude_km"]), token_id, float(innovation)))

    profile_innovations: list[float] = []
    profile_height_intervals: list[float] = []
    vertical_correlations: list[float] = []
    correlation_candidates = 0
    for profile_key in sorted(by_profile):
        rows = sorted(by_profile[profile_key])
        innovations_by_height: dict[float, list[float]] = {}
        for altitude, _token_id, innovation in rows:
            innovations_by_height.setdefault(altitude, []).append(innovation)
        heights = np.asarray(sorted(innovations_by_height), dtype=np.float64)
        innovations = np.asarray([
            np.median(np.asarray(innovations_by_height[altitude], dtype=np.float64))
            for altitude in heights
        ], dtype=np.float64)
        profile_innovations.append(_median(innovations))
        if len(heights) >= 2:
            intervals = np.diff(heights)
            intervals = intervals[intervals > 0.0]
            if len(intervals):
                profile_height_intervals.append(_median(intervals))
        pair_count = max(0, len(innovations) - 1)
        if pair_count >= rules.min_vertical_pairs_per_profile:
            correlation_candidates += 1
            lower = innovations[:-1]
            upper = innovations[1:]
            if np.std(lower) > 0.0 and np.std(upper) > 0.0:
                correlation = float(np.corrcoef(lower, upper)[0, 1])
                if np.isfinite(correlation):
                    vertical_correlations.append(correlation)

    innovation_values = np.asarray(profile_innovations, dtype=np.float64)
    interval_values = np.asarray(profile_height_intervals, dtype=np.float64)
    correlation_values = np.asarray(vertical_correlations, dtype=np.float64)
    innovation_count = int(len(innovation_values))
    if innovation_count:
        center = _median(innovation_values)
        mad = _median(np.abs(innovation_values - center))
        robust_sigma = 1.4826 * mad
        absolute_deviation = np.abs(innovation_values - center)
        if robust_sigma == 0.0:
            if rules.zero_mad_tail_rule != "tail_if_abs_deviation_gt_zero":
                raise ValueError("unsupported frozen zero-MAD tail rule")
            tail = absolute_deviation > 0.0
        else:
            tail = absolute_deviation > rules.tail_sigma * robust_sigma
        tail_count = int(np.count_nonzero(tail))
        innovation_status = "computed"
        tail_fraction: float | None = _fraction(tail_count, innovation_count)
    else:
        center = mad = robust_sigma = None
        tail_count = 0
        innovation_status = "insufficient_data"
        tail_fraction = None
    interval_count, interval_status, interval_median = _defined_statistic(
        interval_values, "median")
    interval_p90 = _p90(interval_values) if interval_count else None
    correlation_count = int(len(correlation_values))
    correlation_computed = (
        correlation_count >= rules.min_profiles_for_vertical_correlation)
    return {
        "innovation_profile_count": innovation_count,
        "innovation_status": innovation_status,
        "innovation_median_dex": center,
        "innovation_mad_dex": mad,
        "innovation_robust_sigma_dex": robust_sigma,
        "innovation_3sigma_tail_profile_count": tail_count,
        "innovation_3sigma_tail_profile_fraction": tail_fraction,
        "unique_token_count": len(token_identities),
        "token_height_interval_profile_count": interval_count,
        "token_height_interval_status": interval_status,
        "token_height_interval_median_km": interval_median,
        "token_height_interval_p90_km": interval_p90,
        "vertical_innovation_correlation_candidate_profiles": (
            correlation_candidates),
        "vertical_innovation_correlation_defined_profiles": correlation_count,
        "vertical_innovation_correlation_status": (
            "computed" if correlation_computed else "insufficient_data"),
        "vertical_innovation_correlation_median": (
            _median(correlation_values) if correlation_computed else None),
    }


def _source_row(
        key: tuple[str, ...], source: str, acc: GroupAccumulator,
        token_registry: Mapping[tuple[str, int, int], Mapping[str, Any]],
        diagnostic_rules: EdgeDiagnosticRules) -> dict[str, Any]:
    row = _base_row(key, acc)
    row["source"] = source
    row["profile_cap_status"] = diagnostic_rules.profile_cap_status
    raw_count = acc.counts.get("raw_tokens", 0)
    effective_count = acc.counts.get("effective_tokens", 0)
    raw_token_count = _raw_query_metric(acc, "token_count")
    token_count = _metric(acc, "token_count")
    unique_count = _metric(acc, "unique_profile_count")
    token_neff = _metric(acc, "token_neff")
    profile_neff = _metric(acc, "profile_neff")
    ratio = _metric(acc, "profile_to_token_neff_ratio")
    max_share = _metric(acc, "effective_max_profile_precision_share")
    raw_nis = _raw_query_metric(acc, "predictive_nis")
    raw_dof = _raw_query_metric(acc, "predictive_nis_dof")
    nis = _metric(acc, "predictive_nis")
    dof = _metric(acc, "predictive_nis_dof")
    raw_innovation = _raw_query_metric(
        acc, "precision_weighted_innovation")
    innovation = _metric(acc, "precision_weighted_innovation")
    isolated = _metric(acc, "isolated_increment")
    joint = _metric(acc, "joint_source_update")
    joint_abs = _metric(acc, "joint_source_update_abs")
    duplicate = _metric(acc, "duplicate_profile_abs")
    concentration_profiles, concentration_dates = _defined_profile_support(
        acc, "profile_to_token_neff_ratio", "effective_max_profile_precision_share")
    duplicate_profiles, duplicate_dates = _defined_profile_support(
        acc, "duplicate_profile_abs")
    concentrated = acc.counts.get("concentrated", 0)
    raw_dof_sum = int(np.rint(np.sum(raw_dof)))
    profile_dof_sum = float(np.sum(dof))
    profile_nis_per_dof = _profile_ratio_metric(
        acc, "predictive_nis", "predictive_nis_dof")
    row.update({
        "queries_with_raw_tokens": raw_count,
        "queries_with_effective_tokens": effective_count,
        "raw_query_token_count_sum": int(np.rint(np.sum(raw_token_count))),
        "profile_equal_token_count_sum_of_within_profile_medians": float(
            np.sum(token_count)),
        "profile_equal_token_count_median": _median(token_count),
        "unique_profile_count_per_query_median": _median(unique_count),
        "unlocalized_precision_sum_mean": _mean(_metric(acc, "unlocalized_precision_sum")),
        "localized_precision_sum_mean": _mean(_metric(acc, "localized_precision_sum")),
        "token_neff_median": _median(token_neff),
        "profile_neff_median": _median(profile_neff),
        "profile_to_token_neff_ratio_median": _median(ratio),
        "max_profile_precision_share_median": _median(max_share),
        "max_profile_precision_share_p90": _p90(max_share),
        "concentrated_query_count": concentrated,
        "concentrated_query_fraction": _fraction(concentrated, effective_count),
        "raw_query_predictive_nis_sum": float(np.sum(raw_nis)),
        "raw_query_predictive_nis_dof_sum": raw_dof_sum,
        "raw_query_predictive_nis_per_dof": (
            float(np.sum(raw_nis)) / raw_dof_sum if raw_dof_sum > 0 else None),
        "profile_equal_predictive_nis_sum_of_within_profile_medians": float(
            np.sum(nis)),
        "profile_equal_predictive_nis_dof_sum_of_within_profile_medians": (
            profile_dof_sum),
        "profile_equal_predictive_nis_per_dof_mean": _mean(
            profile_nis_per_dof),
        "raw_query_precision_weighted_innovation_defined_queries": len(
            raw_innovation),
        "profile_equal_precision_weighted_innovation_defined_profiles": len(
            innovation),
        "profile_equal_precision_weighted_innovation_median_dex": _median(
            innovation),
        "isolated_increment_mean_dex": _mean(isolated),
        "isolated_increment_median_dex": _median(isolated),
        "joint_source_update_mean_dex": _mean(joint),
        "joint_source_update_median_dex": _median(joint),
        "joint_source_update_abs_median_dex": _median(joint_abs),
        "joint_source_update_abs_p90_dex": _p90(joint_abs),
        "duplicate_profile_abs_median_dex": _median(duplicate),
        "duplicate_profile_abs_p90_dex": _p90(duplicate),
        "concentration_defined_profiles": concentration_profiles,
        "concentration_defined_unique_dates": concentration_dates,
        "duplicate_profile_defined_profiles": duplicate_profiles,
        "duplicate_profile_defined_unique_dates": duplicate_dates,
    })
    row.update(_token_profile_diagnostics(
        acc.token_identities, token_registry, diagnostic_rules))
    return row


def _joint_row(key: tuple[str, ...], acc: GroupAccumulator) -> dict[str, Any]:
    row = _base_row(key, acc)
    nis = _metric(acc, "predictive_nis")
    dof = _metric(acc, "predictive_nis_dof")
    joint = _metric(acc, "joint_increment")
    joint_abs = _metric(acc, "joint_increment_abs")
    dof_sum = int(np.rint(np.sum(dof)))
    profile_nis_per_dof = _profile_ratio_metric(
        acc, "predictive_nis", "predictive_nis_dof")
    raw = acc.counts.get("raw_token_queries", 0)
    effective = acc.counts.get("effective_token_queries", 0)
    raw_low = acc.counts.get("raw_token_low_gain_queries", 0)
    effective_low = acc.counts.get("effective_token_low_gain_queries", 0)
    row.update({
        "predictive_nis_sum": float(np.sum(nis)),
        "predictive_nis_dof_sum": dof_sum,
        "predictive_nis_per_dof": _mean(profile_nis_per_dof),
        "joint_increment_mean_dex": _mean(joint),
        "joint_increment_median_dex": _median(joint),
        "joint_increment_abs_median_dex": _median(joint_abs),
        "joint_increment_abs_p90_dex": _p90(joint_abs),
        "joint_interaction_mean_dex": _mean(_metric(acc, "joint_interaction")),
        "raw_token_queries": raw,
        "raw_token_low_gain_queries": raw_low,
        "raw_token_low_gain_fraction": _fraction(raw_low, raw),
        "effective_token_queries": effective,
        "effective_token_low_gain_queries": effective_low,
        "effective_token_low_gain_fraction": _fraction(effective_low, effective),
    })
    for prefix, metric in (
            ("raw_token_low_gain", "raw_token_low_gain_indicator"),
            ("effective_token_low_gain", "effective_token_low_gain_indicator")):
        values = _metric(acc, metric)
        profiles, dates = _defined_profile_support(acc, metric)
        row[f"{prefix}_defined_profiles"] = profiles
        row[f"{prefix}_defined_unique_dates"] = dates
        row[f"{prefix}_profile_fraction"] = _mean(values) if len(values) else None
    for prefix, metric, output in (
            ("direction_consistency", "direction_consistent",
             "direction_consistency_fraction"),
            ("joint_source_innovation_sign", "source_innovation_same_sign",
             "joint_source_innovation_same_sign_fraction"),
            ("joint_source_update_sign", "source_update_same_sign",
             "joint_source_update_same_sign_fraction"),
            ("source_cancellation", "source_cancellation",
             "source_cancellation_fraction")):
        values = _metric(acc, metric)
        profiles, dates = _defined_profile_support(acc, metric)
        row[f"{prefix}_defined_profiles"] = profiles
        if prefix in ("direction_consistency", "source_cancellation"):
            row[f"{prefix}_defined_unique_dates"] = dates
        row[f"{prefix}_status"] = (
            "computed" if profiles else "insufficient_data")
        row[output] = _mean(values) if profiles else None
    for short_name in (
            "drop_200_250", "drop_250_300", "drop_300_400", "drop_400_500",
            "duplicate_FY", "duplicate_COSMIC", "duplicate_both"):
        values = _metric(acc, f"{short_name}_abs")
        row[f"{short_name}_abs_median_dex"] = _median(values)
        row[f"{short_name}_abs_p90_dex"] = _p90(values)
    return row


def _update_common(
        acc: GroupAccumulator, query: Mapping[str, np.ndarray], row: int,
        profile_sets: Mapping[str, set[tuple[str, int]]],
        profile_key: tuple[str, str, int]) -> None:
    acc.query_points += 1
    acc.station_times.add(profile_key)
    acc.days.add(str(query["date_utc"][row]))
    for source in p0b_audit.P0B_SOURCES:
        profiles = profile_sets[source]
        acc.profiles_by_source[source].update(profiles)
        acc.satellite_profiles.update(profiles)


def _station_time_profile_key(
        query: Mapping[str, np.ndarray], row: int,
        rules: ProfileAggregationRules) -> tuple[str, str, int]:
    return (
        str(query["station"][row]), str(query["date_utc"][row]),
        int(query[rules.profile_id_field][row]),
    )


def _scalar_bytes(value: Any) -> tuple[str, bytes]:
    scalar = np.asarray(value)
    return scalar.dtype.str, scalar.tobytes()


def _register_token_payloads(
        token: Mapping[str, np.ndarray], edge: Mapping[str, np.ndarray],
        registry: dict[tuple[str, int, int], dict[str, Any]],
        rules: EdgeDiagnosticRules) -> dict[int, tuple[str, int, int]]:
    lookup: dict[int, tuple[str, int, int]] = {}
    token_payload_fields = tuple(
        name for name in rules.duplicate_token_payload_fields
        if name != "innovation_dex")
    for row in range(len(token["token_row_id"])):
        identity = (
            str(token["source"][row]), int(token["profile_id"][row]),
            int(token["token_id"][row]))
        row_id = int(token["token_row_id"][row])
        lookup[row_id] = identity
        payload = tuple(
            (name, _scalar_bytes(token[name][row]))
            for name in token_payload_fields)
        existing = registry.get(identity)
        if existing is None:
            registry[identity] = {
                "payload": payload,
                "altitude_km": float(token["altitude_km"][row]),
                "innovation_dex": None,
                "innovation_identity": None,
            }
        elif existing["payload"] != payload:
            raise ValueError(
                "duplicate source/profile/token identity has non-identical payload: "
                f"{identity}")
    for row in range(len(edge["query_id"])):
        row_id = int(edge["token_row_id"][row])
        if row_id not in lookup:
            raise ValueError("edge token_row_id is absent from token registry")
        identity = lookup[row_id]
        innovation_identity = _scalar_bytes(edge["innovation_dex"][row])
        existing = registry[identity]
        if existing["innovation_identity"] is None:
            existing["innovation_identity"] = innovation_identity
            existing["innovation_dex"] = float(edge["innovation_dex"][row])
        elif existing["innovation_identity"] != innovation_identity:
            raise ValueError(
                "duplicate source/profile/token identity has non-identical "
                f"innovation_dex: {identity}")
    return lookup


def _query_edge_diagnostics(
        query: Mapping[str, np.ndarray], token: Mapping[str, np.ndarray],
        edge: Mapping[str, np.ndarray],
        token_lookup: Mapping[int, tuple[str, int, int]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {
        int(query_id): {
            "profiles": {source: set() for source in p0b_audit.P0B_SOURCES},
            "tokens": {source: set() for source in p0b_audit.P0B_SOURCES},
            "weighted_innovation_numerator": {
                source: 0.0 for source in p0b_audit.P0B_SOURCES},
            "weighted_innovation_denominator": {
                source: 0.0 for source in p0b_audit.P0B_SOURCES},
            "contribution": {source: 0.0 for source in p0b_audit.P0B_SOURCES},
        }
        for query_id in query["query_id"]
    }
    for index in range(len(edge["query_id"])):
        query_id = int(edge["query_id"][index])
        token_id = int(edge["token_row_id"][index])
        if query_id not in result or token_id not in token_lookup:
            raise ValueError("edge diagnostic lookup violates foreign key")
        source, profile_id, _ = token_lookup[token_id]
        if source != str(edge["source"][index]):
            raise ValueError("edge source differs from token source")
        item = result[query_id]
        item["profiles"][source].add((source, profile_id))
        item["tokens"][source].add(token_lookup[token_id])
        precision = float(edge["localized_precision"][index])
        item["weighted_innovation_numerator"][source] += (
            precision * float(edge["innovation_dex"][index]))
        item["weighted_innovation_denominator"][source] += precision
        item["contribution"][source] += float(
            edge["contribution_joint_dex"][index])
    return result


def build_fixed_summaries(
        inventory: CacheInventory) -> tuple[dict[str, Any], dict[str, Any],
                                             dict[str, Any], dict[str, Any],
                                             dict[str, Any]]:
    """Validate every shard and build the three fixed summaries plus six Qs."""
    regime_groups: dict[tuple[str, ...], GroupAccumulator] = {}
    source_groups: dict[tuple[str, ...], GroupAccumulator] = {}
    joint_groups: dict[tuple[str, ...], GroupAccumulator] = {}
    state = AttributionState()
    global_query_ids: set[int] = set()
    global_sample_keys: set[str] = set()
    global_token_row_ids: set[int] = set()
    token_registry: dict[tuple[str, int, int], dict[str, Any]] = {}
    profile_timestamp_registry: dict[tuple[str, str, int], int] = {}
    totals = {"query_rows": 0, "token_rows": 0, "edge_rows": 0, "batches": 0}
    query_partition_counts = {
        split: 0 for split in p0b_audit.P0B_QUERY_SPLITS}
    thresholds = _decision_thresholds(inventory.contract)
    profile_rules = _profile_aggregation_rules(inventory.contract)
    diagnostic_rules = _edge_diagnostic_rules(inventory.contract)
    atol = float(inventory.contract["acceptance"]["closure_atol"])
    rtol = float(inventory.contract["acceptance"]["closure_rtol"])

    for shard in inventory.batches:
        _verify_batch_shard_identities(inventory, shard)
        query, token, edge, _ = validate_batch_tables(shard, inventory.contract)
        _verify_batch_shard_identities(inventory, shard)
        current_query_ids = set(map(int, query["query_id"]))
        current_samples = set(map(str, query["sample_key"]))
        current_token_ids = set(map(int, token["token_row_id"]))
        if global_query_ids.intersection(current_query_ids):
            raise ValueError("duplicate query_id across batch shards")
        if global_sample_keys.intersection(current_samples):
            raise ValueError("duplicate sample_key across batch shards")
        if global_token_row_ids.intersection(current_token_ids):
            raise ValueError("duplicate token_row_id across batch shards")
        global_query_ids.update(current_query_ids)
        global_sample_keys.update(current_samples)
        global_token_row_ids.update(current_token_ids)
        totals["query_rows"] += len(query["query_id"])
        totals["token_rows"] += len(token["token_row_id"])
        totals["edge_rows"] += len(edge["query_id"])
        totals["batches"] += 1
        for split in p0b_audit.P0B_QUERY_SPLITS:
            query_partition_counts[split] += int(np.count_nonzero(
                np.asarray(query["query_split"]) == split))
        token_lookup = _register_token_payloads(
            token, edge, token_registry, diagnostic_rules)
        diagnostics = _query_edge_diagnostics(
            query, token, edge, token_lookup)

        for row, query_id_raw in enumerate(query["query_id"]):
            query_id = int(query_id_raw)
            key = _group_key(query, row, inventory.contract)
            profile_key = _station_time_profile_key(query, row, profile_rules)
            timestamp = int(query["timestamp_unix"][row])
            existing_timestamp = profile_timestamp_registry.setdefault(
                profile_key, timestamp)
            if existing_timestamp != timestamp:
                raise ValueError(
                    "station/date/query_profile_id maps to multiple timestamps")
            profile_sets = diagnostics[query_id]["profiles"]
            regime = regime_groups.setdefault(key, GroupAccumulator())
            joint_acc = joint_groups.setdefault(key, GroupAccumulator())
            _update_common(regime, query, row, profile_sets, profile_key)
            _update_common(joint_acc, query, row, profile_sets, profile_key)
            observation = float(query["isr_log10_ne"][row])
            m11 = float(query["M11_log10_ne"][row])
            values = {
                "raw_error": float(query["raw_iri_log10_ne"][row]) - observation,
                "M00_error": float(query["M00_log10_ne"][row]) - observation,
                "M10_error": float(query["M10_log10_ne"][row]) - observation,
                "M01_error": float(query["M01_log10_ne"][row]) - observation,
                "M11_error": m11 - observation,
                "background_increment": (
                    float(query["M00_log10_ne"][row])
                    - float(query["raw_iri_log10_ne"][row])),
                "joint_update_FY": float(query["joint_update_FY_dex"][row]),
                "joint_update_COSMIC": float(query["joint_update_COSMIC_dex"][row]),
                "isolated_increment_FY": float(
                    query["isolated_increment_FY_dex"][row]),
                "isolated_increment_COSMIC": float(
                    query["isolated_increment_COSMIC_dex"][row]),
                "joint_increment": float(query["joint_increment_dex"][row]),
                "joint_interaction": (
                    float(query["joint_increment_dex"][row])
                    - float(query["isolated_increment_FY_dex"][row])
                    - float(query["isolated_increment_COSMIC_dex"][row])),
                "drop_200_250": float(query["CF_drop_200_250_log10_ne"][row]) - m11,
                "drop_250_300": float(query["CF_drop_250_300_log10_ne"][row]) - m11,
                "drop_300_400": float(query["CF_drop_300_400_log10_ne"][row]) - m11,
                "drop_400_500": float(query["CF_drop_400_500_log10_ne"][row]) - m11,
                "duplicate_FY": float(
                    query["CF_duplicate_FY_dominant_profile_log10_ne"][row]) - m11,
                "duplicate_COSMIC": float(
                    query["CF_duplicate_COSMIC_dominant_profile_log10_ne"][row]) - m11,
                "duplicate_both": float(
                    query["CF_duplicate_both_dominant_profiles_log10_ne"][row]) - m11,
            }
            for name, value in values.items():
                regime.add_value(name, value, profile_key)
            for name in (
                    "joint_increment", "drop_200_250", "drop_250_300",
                    "drop_300_400", "drop_400_500", "duplicate_FY",
                    "duplicate_COSMIC", "duplicate_both"):
                regime.add_value(f"{name}_abs", abs(values[name]), profile_key)
            for name in (
                    "joint_increment", "joint_interaction", "drop_200_250",
                    "drop_250_300", "drop_300_400", "drop_400_500",
                    "duplicate_FY", "duplicate_COSMIC", "duplicate_both"):
                joint_acc.add_value(name, values[name], profile_key)
            for name in (
                    "joint_increment", "drop_200_250", "drop_250_300",
                    "drop_300_400", "drop_400_500", "duplicate_FY",
                    "duplicate_COSMIC", "duplicate_both"):
                joint_acc.add_value(
                    f"{name}_abs", abs(values[name]), profile_key)
            joint_acc.add_value(
                "predictive_nis", query["predictive_nis_unlocalized_joint"][row],
                profile_key)
            joint_acc.add_value(
                "predictive_nis_dof",
                query["predictive_nis_unlocalized_joint_dof"][row], profile_key)

            increment = values["joint_increment"]
            raw_has_token = str(query["raw_coverage_code"][row]) != "no_token"
            effective_has_token = str(query["coverage_code"][row]) != "no_token"
            low_gain = abs(increment) < thresholds.material_increment
            if raw_has_token:
                state.raw_token_queries += 1
                joint_acc.add_count("raw_token_queries")
                joint_acc.add_value(
                    "raw_token_low_gain_indicator", float(low_gain), profile_key)
                if low_gain:
                    state.raw_token_low_gain_queries += 1
                    joint_acc.add_count("raw_token_low_gain_queries")
            if effective_has_token:
                state.effective_token_queries += 1
                joint_acc.add_count("effective_token_queries")
                joint_acc.add_value(
                    "effective_token_low_gain_indicator", float(low_gain),
                    profile_key)
                if low_gain:
                    state.effective_token_low_gain_queries += 1
                    joint_acc.add_count("effective_token_low_gain_queries")

            if abs(increment) >= thresholds.material_increment:
                state.material_query_count += 1
                expected_by_source = {
                    source: float(query[f"joint_update_{source}_dex"][row])
                    for source in p0b_audit.P0B_SOURCES
                }
                for source in p0b_audit.P0B_SOURCES:
                    actual = diagnostics[query_id]["contribution"][source]
                    expected = expected_by_source[source]
                    error = abs(actual - expected)
                    state.material_closure_max_abs_error = max(
                        state.material_closure_max_abs_error, error)
                    if not np.isclose(actual, expected, atol=atol, rtol=rtol):
                        state.material_closure_failures += 1

            innovation_by_source: dict[str, float | None] = {}
            total_innovation_numerator = 0.0
            total_innovation_denominator = 0.0
            for source in p0b_audit.P0B_SOURCES:
                source_denominator = diagnostics[query_id][
                    "weighted_innovation_denominator"][source]
                source_numerator = diagnostics[query_id][
                    "weighted_innovation_numerator"][source]
                if source_denominator > 0.0:
                    innovation_by_source[source] = (
                        source_numerator / source_denominator)
                    total_innovation_numerator += source_numerator
                    total_innovation_denominator += source_denominator
                else:
                    innovation_by_source[source] = None
            epsilon = diagnostic_rules.joint_source_sign_epsilon
            if total_innovation_denominator > 0.0:
                net_innovation = (
                    total_innovation_numerator / total_innovation_denominator)
                if abs(increment) >= epsilon and abs(net_innovation) >= epsilon:
                    joint_acc.add_value(
                        "direction_consistent",
                        float(np.sign(increment) == np.sign(net_innovation)),
                        profile_key)
            fy_innovation = innovation_by_source["FY"]
            cosmic_innovation = innovation_by_source["COSMIC"]
            if (fy_innovation is not None and cosmic_innovation is not None
                    and abs(fy_innovation) >= epsilon
                    and abs(cosmic_innovation) >= epsilon):
                joint_acc.add_value(
                    "source_innovation_same_sign",
                    float(np.sign(fy_innovation) == np.sign(cosmic_innovation)),
                    profile_key)
            fy_update = values["joint_update_FY"]
            cosmic_update = values["joint_update_COSMIC"]
            if abs(fy_update) >= epsilon and abs(cosmic_update) >= epsilon:
                same_update_sign = np.sign(fy_update) == np.sign(cosmic_update)
                joint_acc.add_value(
                    "source_update_same_sign", float(same_update_sign), profile_key)
                joint_acc.add_value(
                    "source_cancellation", float(not same_update_sign), profile_key)

            for source in p0b_audit.P0B_SOURCES:
                source_key = key + (source,)
                source_acc = source_groups.setdefault(source_key, GroupAccumulator())
                source_only_profiles = {
                    candidate: (profile_sets[candidate]
                                if candidate == source else set())
                    for candidate in p0b_audit.P0B_SOURCES
                }
                _update_common(
                    source_acc, query, row, source_only_profiles, profile_key)
                source_acc.token_identities.update(
                    diagnostics[query_id]["tokens"][source])
                token_count = int(query[f"{source}_token_count"][row])
                localized = float(query[f"{source}_localized_precision_sum"][row])
                source_acc.add_value("token_count", token_count, profile_key)
                source_acc.add_value(
                    "unique_profile_count",
                    query[f"{source}_unique_profile_count"][row], profile_key)
                source_acc.add_value(
                    "unlocalized_precision_sum",
                    query[f"{source}_unlocalized_precision_sum"][row], profile_key)
                source_acc.add_value(
                    "localized_precision_sum", localized, profile_key)
                source_acc.add_value(
                    "token_neff", query[f"{source}_token_neff"][row], profile_key)
                source_acc.add_value(
                    "profile_neff", query[f"{source}_profile_neff"][row],
                    profile_key)
                source_acc.add_value(
                    "max_profile_precision_share",
                    query[f"{source}_max_profile_precision_share"][row], profile_key)
                source_acc.add_value(
                    "predictive_nis",
                    query[f"predictive_nis_unlocalized_{source}"][row], profile_key)
                source_acc.add_value(
                    "predictive_nis_dof",
                    query[f"predictive_nis_unlocalized_{source}_dof"][row],
                    profile_key)
                source_acc.add_value(
                    "isolated_increment",
                    query[f"isolated_increment_{source}_dex"][row], profile_key)
                source_acc.add_value(
                    "joint_source_update", query[f"joint_update_{source}_dex"][row],
                    profile_key)
                source_acc.add_value(
                    "joint_source_update_abs",
                    abs(float(query[f"joint_update_{source}_dex"][row])),
                    profile_key)
                duplicate_name = f"duplicate_{source}"
                dominant_valid = bool(query[f"{source}_dominant_profile_valid"][row])
                if dominant_valid:
                    source_acc.add_value(
                        "duplicate_profile", values[duplicate_name], profile_key)
                    source_acc.add_value(
                        "duplicate_profile_abs", abs(values[duplicate_name]),
                        profile_key)
                if token_count > 0:
                    source_acc.add_count("raw_tokens")
                if localized > 0.0:
                    source_acc.add_count("effective_tokens")
                    token_neff = float(query[f"{source}_token_neff"][row])
                    profile_neff = float(query[f"{source}_profile_neff"][row])
                    ratio = profile_neff / token_neff
                    max_share = float(
                        query[f"{source}_max_profile_precision_share"][row])
                    concentrated = (
                        max_share >= thresholds.concentration_share
                        or ratio <= thresholds.concentration_neff_ratio)
                    source_acc.add_value(
                        "profile_to_token_neff_ratio", ratio, profile_key)
                    source_acc.add_value(
                        "effective_max_profile_precision_share", max_share,
                        profile_key)
                    if concentrated:
                        source_acc.add_count("concentrated")
                    denominator = diagnostics[query_id][
                        "weighted_innovation_denominator"][source]
                    if denominator > 0.0:
                        source_acc.add_value(
                            "precision_weighted_innovation",
                            diagnostics[query_id][
                                "weighted_innovation_numerator"][source]
                            / denominator, profile_key)

    if state.material_closure_failures:
        raise ValueError(
            "material source contribution closure failed; closure is a hard QA "
            f"guard ({state.material_closure_failures} failures)")
    if totals != inventory.manifest["counts"]:
        raise ValueError(
            f"validated row counts differ from manifest: {totals} != "
            f"{inventory.manifest['counts']}")
    if query_partition_counts != inventory.identity_validation[
            "query_partition_counts"]:
        raise ValueError(
            "recomputed query-partition counts differ from the runtime registry")

    regime_keys = sorted(
        regime_groups, key=lambda key: _fixed_group_sort_key(
            key, inventory.contract))
    source_keys = sorted(
        source_groups, key=lambda key: _fixed_group_sort_key(
            key, inventory.contract))
    joint_keys = sorted(
        joint_groups, key=lambda key: _fixed_group_sort_key(
            key, inventory.contract))
    regime_rows = [_regime_row(key, regime_groups[key]) for key in regime_keys]
    source_rows = [
        _source_row(
            key[:-1], key[-1], source_groups[key], token_registry,
            diagnostic_rules)
        for key in source_keys
    ]
    joint_rows = [_joint_row(key, joint_groups[key]) for key in joint_keys]
    questions = _build_six_questions(
        regime_rows, source_rows, joint_rows, state, inventory.contract,
        thresholds)
    q5_routing_status = next(
        record["evidence"]["routing_status"]
        for record in questions["questions"]
        if record["id"] == "Q5_low_altitude_drift")
    integrity = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "validated_batches": totals["batches"],
        "validated_counts": totals,
        "global_unique_query_ids": len(global_query_ids),
        "global_unique_sample_keys": len(global_sample_keys),
        "global_unique_token_row_ids": len(global_token_row_ids),
        "global_unique_source_profile_token_identities": len(token_registry),
        "duplicate_source_profile_token_payloads_consistent": True,
        "station_time_profile_equal_weighting": True,
        "profile_cap_status": diagnostic_rules.profile_cap_status,
        "query_partition_counts": query_partition_counts,
        "satellite_profile_counts": inventory.identity_validation[
            "satellite_profile_counts"],
        "all_batch_schema_dtype_cross_table_checks_passed": True,
        "batch_identity_rechecked_before_and_after_each_shard_read": True,
        "batch_identity_checks_per_aggregation": 2 * 3 * totals["batches"],
        "defined_profile_and_date_gates_enforced": True,
        "question_interpretation_boundaries_complete": True,
        "q5_dual_routing_computed": True,
        "q5_routing_status": q5_routing_status,
        "runtime_identity_validation": dict(inventory.identity_validation),
    }
    source_payload = _table_payload(
        "query_source", _SOURCE_DIMENSIONS, source_rows)
    source_payload["metric_reduction_order"] = _SOURCE_METRIC_REDUCTION_ORDER
    return (
        _table_payload("query_regime", _DIMENSIONS, regime_rows),
        source_payload,
        _table_payload("query_joint", _DIMENSIONS, joint_rows),
        questions,
        integrity,
    )


def _table_payload(table: str, dimensions: Sequence[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "table": table,
        "dimensions": list(dimensions),
        "rows": rows,
    }


def _build_six_questions(
        regime_rows: Sequence[Mapping[str, Any]],
        source_rows: Sequence[Mapping[str, Any]],
        joint_rows: Sequence[Mapping[str, Any]], state: AttributionState,
        contract: Mapping[str, Any],
        thresholds: DecisionThresholds) -> dict[str, Any]:
    allowed = set(p0b_audit.P0B_QUESTION_TERMINAL_STATUSES)
    eligible_cells: list[dict[str, Any]] = []
    signatures: list[tuple[str, ...]] = []
    for row in regime_rows:
        if (int(row["station_time_profiles"]) < thresholds.minimum_profiles
                or int(row["unique_days"]) < thresholds.minimum_days):
            continue
        components = {
            "raw_IRI_preexisting": abs(float(
                row["raw_iri_bias_median_dex"])),
            "background_M00_shift": abs(float(
                row["background_increment_median_dex"])),
            "FY_isolated_increment": abs(float(
                row["isolated_increment_FY_median_dex"])),
            "COSMIC_isolated_increment": abs(float(
                row["isolated_increment_COSMIC_median_dex"])),
            "joint_interaction": abs(float(
                row["joint_interaction_median_dex"])),
        }
        labels = tuple(sorted(
            name
            for name, value in components.items()
            if abs(value) >= thresholds.material_bias))
        signatures.append(labels)
        eligible_cells.append({
            "cell": {name: row[name] for name in _DIMENSIONS},
            "station_time_profiles": int(row["station_time_profiles"]),
            "unique_days": int(row["unique_days"]),
            "absolute_median_components_dex": components,
            "origin_labels": list(labels),
        })
    if not eligible_cells:
        q1_status = "insufficient_evidence"
    elif all(not signature for signature in signatures):
        q1_status = "not_supported"
    elif (len(set(signatures)) == 1 and len(signatures[0]) == 1):
        q1_status = "supported"
    else:
        q1_status = "mixed"

    def evidence_status(
            value: float | None, threshold: float, profiles: int,
            dates: int) -> tuple[str, bool | None]:
        if (profiles < thresholds.minimum_defined_profiles
                or dates < thresholds.minimum_defined_days
                or value is None):
            return "insufficient_evidence", None
        passed = float(value) >= threshold
        return ("supported" if passed else "not_supported"), passed

    q2_material_cells: list[dict[str, Any]] = []
    q2_direction_flags: list[bool] = []
    q2_ineligible_direction_cells = 0
    for row in joint_rows:
        if (int(row["station_time_profiles"]) < thresholds.minimum_profiles
                or int(row["unique_days"]) < thresholds.minimum_days
                or float(row["joint_increment_abs_median_dex"])
                < thresholds.material_increment):
            continue
        direction_fraction = row["direction_consistency_fraction"]
        direction_profiles = int(row["direction_consistency_defined_profiles"])
        direction_dates = int(row["direction_consistency_defined_unique_dates"])
        direction_status, direction_pass = evidence_status(
            direction_fraction, thresholds.direction_consistency_fraction,
            direction_profiles, direction_dates)
        if direction_pass is not None:
            q2_direction_flags.append(direction_pass)
        else:
            q2_ineligible_direction_cells += 1
        effective_low_gain = row["effective_token_low_gain_profile_fraction"]
        raw_low_gain = row["raw_token_low_gain_profile_fraction"]
        raw_low_gain_status, raw_low_gain_pass = evidence_status(
            raw_low_gain, thresholds.low_gain_fraction,
            int(row["raw_token_low_gain_defined_profiles"]),
            int(row["raw_token_low_gain_defined_unique_dates"]))
        effective_low_gain_status, effective_low_gain_pass = evidence_status(
            effective_low_gain, thresholds.low_gain_fraction,
            int(row["effective_token_low_gain_defined_profiles"]),
            int(row["effective_token_low_gain_defined_unique_dates"]))
        cancellation_fraction = row["source_cancellation_fraction"]
        cancellation_status, cancellation_pass = evidence_status(
            cancellation_fraction, thresholds.source_cancellation_fraction,
            int(row["source_cancellation_defined_profiles"]),
            int(row["source_cancellation_defined_unique_dates"]))
        q2_material_cells.append({
            "cell": {name: row[name] for name in _DIMENSIONS},
            "station_time_profiles": int(row["station_time_profiles"]),
            "unique_days": int(row["unique_days"]),
            "joint_increment_abs_median_dex": float(
                row["joint_increment_abs_median_dex"]),
            "direction_consistency_defined_profiles": direction_profiles,
            "direction_consistency_defined_unique_dates": direction_dates,
            "direction_consistency_fraction": direction_fraction,
            "direction_evidence_status": direction_status,
            "direction_consistency_meets_threshold": direction_pass,
            "raw_token_low_gain_defined_profiles": int(
                row["raw_token_low_gain_defined_profiles"]),
            "raw_token_low_gain_defined_unique_dates": int(
                row["raw_token_low_gain_defined_unique_dates"]),
            "raw_token_low_gain_fraction": raw_low_gain,
            "raw_token_low_gain_mechanism_status": raw_low_gain_status,
            "raw_token_low_gain_mechanism_supported": raw_low_gain_pass,
            "effective_token_low_gain_defined_profiles": int(
                row["effective_token_low_gain_defined_profiles"]),
            "effective_token_low_gain_defined_unique_dates": int(
                row["effective_token_low_gain_defined_unique_dates"]),
            "effective_token_low_gain_fraction": effective_low_gain,
            "effective_token_low_gain_mechanism_status": effective_low_gain_status,
            "effective_token_low_gain_mechanism_supported": effective_low_gain_pass,
            "source_cancellation_defined_profiles": int(
                row["source_cancellation_defined_profiles"]),
            "source_cancellation_defined_unique_dates": int(
                row["source_cancellation_defined_unique_dates"]),
            "source_cancellation_fraction": cancellation_fraction,
            "source_cancellation_mechanism_status": cancellation_status,
            "source_cancellation_mechanism_supported": cancellation_pass,
            "source_edge_and_joint_source_sum_closure_pass": True,
        })
    if not q2_material_cells:
        q2_status = "insufficient_evidence"
    elif not q2_direction_flags:
        q2_status = "insufficient_evidence"
    elif q2_ineligible_direction_cells:
        q2_status = "mixed"
    elif all(q2_direction_flags):
        q2_status = "supported"
    elif not any(q2_direction_flags):
        q2_status = "not_supported"
    else:
        q2_status = "mixed"

    q3_cells: list[dict[str, Any]] = []
    q3_cell_statuses: list[str] = []
    q3_ineligible_evidence_cells: list[dict[str, Any]] = []
    for row in source_rows:
        if (int(row["station_time_profiles"]) < thresholds.minimum_profiles
                or int(row["unique_days"]) < thresholds.minimum_days
                or int(row["queries_with_effective_tokens"]) == 0):
            continue
        concentration_profiles = int(row["concentration_defined_profiles"])
        concentration_dates = int(row["concentration_defined_unique_dates"])
        duplicate_profiles = int(row["duplicate_profile_defined_profiles"])
        duplicate_dates = int(row["duplicate_profile_defined_unique_dates"])
        if (concentration_profiles < thresholds.minimum_defined_profiles
                or concentration_dates < thresholds.minimum_defined_days
                or duplicate_profiles < thresholds.minimum_defined_profiles
                or duplicate_dates < thresholds.minimum_defined_days):
            q3_ineligible_evidence_cells.append({
                "cell": {name: row[name] for name in _SOURCE_DIMENSIONS},
                "station_time_profiles": int(row["station_time_profiles"]),
                "unique_days": int(row["unique_days"]),
                "concentration_defined_profiles": concentration_profiles,
                "concentration_defined_unique_dates": concentration_dates,
                "duplicate_profile_defined_profiles": duplicate_profiles,
                "duplicate_profile_defined_unique_dates": duplicate_dates,
                "status": "insufficient_evidence",
            })
            continue
        concentration = (
            float(row["max_profile_precision_share_median"])
            >= thresholds.concentration_share
            or float(row["profile_to_token_neff_ratio_median"])
            <= thresholds.concentration_neff_ratio)
        duplication_material = (
            float(row["duplicate_profile_abs_median_dex"])
            >= thresholds.duplicate_median
            or float(row["duplicate_profile_abs_p90_dex"])
            >= thresholds.duplicate_p90)
        cell_status = (
            "supported" if concentration and duplication_material
            else "mixed" if concentration or duplication_material
            else "not_supported")
        q3_cell_statuses.append(cell_status)
        q3_cells.append({
            "cell": {name: row[name] for name in _SOURCE_DIMENSIONS},
            "station_time_profiles": int(row["station_time_profiles"]),
            "unique_days": int(row["unique_days"]),
            "concentration_defined_profiles": concentration_profiles,
            "concentration_defined_unique_dates": concentration_dates,
            "duplicate_profile_defined_profiles": duplicate_profiles,
            "duplicate_profile_defined_unique_dates": duplicate_dates,
            "max_profile_precision_share_median": float(
                row["max_profile_precision_share_median"]),
            "profile_to_token_neff_ratio_median": float(
                row["profile_to_token_neff_ratio_median"]),
            "duplicate_profile_abs_median_dex": float(
                row["duplicate_profile_abs_median_dex"]),
            "duplicate_profile_abs_p90_dex": float(
                row["duplicate_profile_abs_p90_dex"]),
            "concentration": concentration,
            "duplication_material": duplication_material,
            "cell_status": cell_status,
        })
    if not q3_cells:
        q3_status = "insufficient_evidence"
    elif len(set(q3_cell_statuses)) == 1:
        q3_status = q3_cell_statuses[0]
    else:
        q3_status = "mixed"

    low_altitude_evidence: list[dict[str, Any]] = []
    eligible_low_altitude_statuses: list[str] = []
    eligible_analysis_material: list[bool] = []
    eligible_background_material: list[bool] = []
    regime_by_cell: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for regime_row in regime_rows:
        regime_key = tuple(regime_row[name] for name in _DIMENSIONS)
        if regime_key in regime_by_cell:
            raise ValueError("duplicate regime row for Q5 cell")
        regime_by_cell[regime_key] = regime_row
    q5_rule = contract["decision_rules"]["Q5_low_altitude_drift"]
    background_threshold = float(
        q5_rule["background_path_material_abs_M00_bias_dex_gte"])
    q5_routes = q5_rule["routing"]
    low_altitude_label = (
        f"[{thresholds.low_altitude_lower:g},"
        f"{thresholds.low_altitude_upper:g})km")
    for row in joint_rows:
        if row["query_altitude_band"] != low_altitude_label:
            continue
        cell_key = tuple(row[name] for name in _DIMENSIONS)
        regime_row = regime_by_cell.get(cell_key)
        if regime_row is None:
            raise ValueError("Q5 joint row has no matching regime row")
        enough = (
            int(row["station_time_profiles"]) >= thresholds.minimum_profiles
            and int(row["unique_days"]) >= thresholds.minimum_days)
        metrics: dict[str, dict[str, float | bool]] = {}
        deletion_sensitive = False
        for name in (
                "drop_200_250", "drop_250_300", "drop_300_400",
                "drop_400_500"):
            median = float(row[f"{name}_abs_median_dex"])
            p90 = float(row[f"{name}_abs_p90_dex"])
            this_material = (
                median >= thresholds.deletion_median
                or p90 >= thresholds.deletion_p90)
            deletion_sensitive = deletion_sensitive or this_material
            metrics[name] = {
                "median_abs_delta_dex": median,
                "p90_abs_delta_dex": p90,
                "material": this_material,
            }
        material_drift = (
            float(row["joint_increment_abs_median_dex"])
            >= thresholds.material_bias)
        cell_status = (
            "supported" if material_drift and deletion_sensitive
            else "mixed" if material_drift or deletion_sensitive
            else "not_supported")
        analysis_material = enough and material_drift and deletion_sensitive
        background_material = (
            enough and abs(float(regime_row["M00_bias_median_dex"]))
            >= background_threshold)
        if not enough:
            route = q5_routes["insufficient_evidence"]
        elif analysis_material and background_material:
            route = q5_routes["analysis_supported_background_material"]
        elif analysis_material:
            route = q5_routes["analysis_supported_background_not_material"]
        elif background_material:
            route = q5_routes["analysis_not_supported_background_material"]
        else:
            route = q5_routes["analysis_not_supported_background_not_material"]
        low_altitude_evidence.append({
            "cell": {name: row[name] for name in _DIMENSIONS},
            "station_time_profiles": int(row["station_time_profiles"]),
            "unique_days": int(row["unique_days"]),
            "eligible": enough,
            "joint_increment_abs_median_dex": float(
                row["joint_increment_abs_median_dex"]),
            "raw_iri_bias_median_dex": float(
                regime_row["raw_iri_bias_median_dex"]),
            "raw_iri_rmse_dex": float(regime_row["raw_iri_rmse_dex"]),
            "M00_bias_median_dex": float(regime_row["M00_bias_median_dex"]),
            "M00_rmse_dex": float(regime_row["M00_rmse_dex"]),
            "M00_minus_IRI_median_dex": float(
                regime_row["background_increment_median_dex"]),
            "material_low_altitude_drift": material_drift,
            "height_deletion_sensitive": deletion_sensitive,
            "analysis_path_status": (
                cell_status if enough else "insufficient_evidence"),
            "background_path_material": background_material,
            "routing_status": route,
            "cell_status": cell_status if enough else "ineligible",
            "height_deletion_counterfactuals": metrics,
        })
        if enough:
            eligible_low_altitude_statuses.append(cell_status)
            eligible_analysis_material.append(analysis_material)
            eligible_background_material.append(background_material)
    if not eligible_low_altitude_statuses:
        q5_status = "insufficient_evidence"
    elif len(set(eligible_low_altitude_statuses)) == 1:
        q5_status = eligible_low_altitude_statuses[0]
    else:
        q5_status = "mixed"
    if not eligible_low_altitude_statuses:
        q5_routing_status = q5_routes["insufficient_evidence"]
    elif any(eligible_analysis_material) and any(eligible_background_material):
        q5_routing_status = q5_routes["analysis_supported_background_material"]
    elif any(eligible_analysis_material):
        q5_routing_status = q5_routes["analysis_supported_background_not_material"]
    elif any(eligible_background_material):
        q5_routing_status = q5_routes["analysis_not_supported_background_material"]
    else:
        q5_routing_status = q5_routes[
            "analysis_not_supported_background_not_material"]

    raw_low_fraction = _fraction(
        state.raw_token_low_gain_queries, state.raw_token_queries)
    effective_low_fraction = _fraction(
        state.effective_token_low_gain_queries, state.effective_token_queries)
    records = [
        {
            "id": "Q1_bias_origin",
            "status": q1_status,
            "decision_rule": (
                "Eligible cells require >=30 station-time profiles and >=2 days; "
                "raw IRI, M00-background shift, isolated FY/COSMIC increments, "
                "and joint interaction are material at |median|>=0.02 dex; exactly "
                "one common origin is supported and cell-dependent labels are mixed."),
            "evidence": {"eligible_cells": eligible_cells},
        },
        {
            "id": "Q2_increment_consistency",
            "status": q2_status,
            "decision_rule": (
                "Closure is a hard QA guard, not attribution evidence. In each "
                "material eligible cell, the station-time-profile-equal fraction "
                "whose M11-M00 direction matches the localized-precision-weighted "
                "innovation must be >=0.75. All cells pass => supported, all fail "
                "=> not_supported, disagreement or partial undefinedness => mixed. "
                "Low-gain and FY/COSMIC cancellation fractions only explain small "
                "increments and do not decide direction status."),
            "evidence": {
                "material_eligible_cells": q2_material_cells,
                "material_query_count": state.material_query_count,
                "material_closure_failures": state.material_closure_failures,
                "material_closure_max_abs_error": state.material_closure_max_abs_error,
                "closure_atol": float(contract["acceptance"]["closure_atol"]),
                "closure_rtol": float(contract["acceptance"]["closure_rtol"]),
                "raw_token_queries": state.raw_token_queries,
                "raw_token_low_gain_queries": state.raw_token_low_gain_queries,
                "raw_token_low_gain_fraction": raw_low_fraction,
                "effective_token_queries": state.effective_token_queries,
                "effective_token_low_gain_queries": (
                    state.effective_token_low_gain_queries),
                "effective_token_low_gain_fraction": effective_low_fraction,
                "direction_consistency_fraction_gte": (
                    thresholds.direction_consistency_fraction),
                "low_gain_fraction_gte": thresholds.low_gain_fraction,
                "source_cancellation_fraction_gte": (
                    thresholds.source_cancellation_fraction),
                "ineligible_direction_material_cells": (
                    q2_ineligible_direction_cells),
            },
        },
        {
            "id": "Q3_profile_precision_concentration",
            "status": q3_status,
            "decision_rule": (
                "Eligible cells require >=30 station-time profiles and >=2 days; "
                "concentration and duplicate evidence each additionally require "
                ">=30 defined profiles across >=2 dates. "
                "Concentration is max_profile_precision_share>=0.5 or "
                "profile_neff/token_neff<=0.25. Duplication is material when "
                "median |delta|>=0.02 dex or P90>=0.05 dex. Both imply supported; "
                "one implies mixed; neither implies not_supported."),
            "evidence": {
                "eligible_cells": q3_cells,
                "ineligible_evidence_cells": q3_ineligible_evidence_cells,
            },
        },
        {
            "id": "Q4_source_latitude_separation",
            "status": "insufficient_evidence",
            "decision_rule": "Forced insufficient_evidence in P0-B.",
            "evidence": {
                "reason": (
                    "Two fixed ISR stations do not independently randomize source "
                    "coverage versus latitude/magnetic topology."),
                "query_source_summary": "query_source_summary.json",
            },
        },
        {
            "id": "Q5_low_altitude_drift",
            "status": q5_status,
            "decision_rule": (
                "Within 120<=altitude<200 km, an eligible fixed cell requires >=30 "
                "station-time profiles and >=2 days. Material drift requires "
                "median |M11-M00|>=0.02 dex and high-altitude height deletion is "
                "sensitive when median |delta|>=0.02 dex or P90>=0.05 dex; both "
                "must co-occur for supported."),
            "evidence": {
                "eligible_and_ineligible_cells": low_altitude_evidence,
                "routing_status": q5_routing_status,
            },
        },
        {
            "id": "Q6_peak_persistence",
            "status": "insufficient_evidence",
            "decision_rule": (
                "Forced insufficient_evidence until observation-only hmF2 "
                "eligibility is fixed and P0-A closes."),
            "evidence": {
                "fixed_observation_eligibility_status": contract[
                    "p0a_dependency_exception"][
                        "fixed_observation_eligibility_status"],
                "p0a_overall_status_at_freeze": contract[
                    "p0a_dependency_exception"]["p0a_overall_status_at_freeze"],
            },
        },
    ]
    for record in records:
        record.update(p0b_audit.P0B_QUESTION_INTERPRETATION_BOUNDARIES[
            record["id"]])
    if len(records) != 6 or {record["id"] for record in records} != {
            item["id"] for item in contract["six_questions"]["questions"]}:
        raise ValueError("six-question IDs differ from the frozen contract")
    if any(record["status"] not in allowed for record in records):
        raise ValueError("six-question status is outside the frozen allowed set")
    if any({
            key: record.get(key)
            for key in ("interpretation_boundary", "supports", "does_not_support")
    } != p0b_audit.P0B_QUESTION_INTERPRETATION_BOUNDARIES[record["id"]]
           for record in records):
        raise ValueError("six-question interpretation boundary differs from contract")
    if next(row for row in records if row["id"] == "Q4_source_latitude_separation")[
            "status"] != "insufficient_evidence":
        raise ValueError("Q4 must remain insufficient_evidence")
    if next(row for row in records if row["id"] == "Q6_peak_persistence")[
            "status"] != "insufficient_evidence":
        raise ValueError("Q6 must remain insufficient_evidence")
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "provisional",
        "allowed_terminal_statuses": list(
            p0b_audit.P0B_QUESTION_TERMINAL_STATUSES),
        "thresholds": {
            "material_bias_dex": thresholds.material_bias,
            "material_increment_dex": thresholds.material_increment,
            "concentrated_max_profile_share": thresholds.concentration_share,
            "concentrated_profile_to_token_neff_ratio": (
                thresholds.concentration_neff_ratio),
            "duplicate_profile_median_abs_dex": thresholds.duplicate_median,
            "duplicate_profile_p90_abs_dex": thresholds.duplicate_p90,
            "height_deletion_median_abs_dex": thresholds.deletion_median,
            "height_deletion_p90_abs_dex": thresholds.deletion_p90,
            "minimum_station_time_profiles": thresholds.minimum_profiles,
            "minimum_unique_days": thresholds.minimum_days,
            "minimum_defined_station_time_profiles": (
                thresholds.minimum_defined_profiles),
            "minimum_defined_unique_dates": thresholds.minimum_defined_days,
            "direction_consistency_fraction": (
                thresholds.direction_consistency_fraction),
            "low_gain_fraction": thresholds.low_gain_fraction,
            "source_cancellation_fraction": (
                thresholds.source_cancellation_fraction),
        },
        "questions": records,
    }


def _json_text(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n"


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise ValueError(
                f"CSV row schema mismatch: missing={sorted(set(fields) - set(row))}, "
                f"extra={sorted(set(row) - set(fields))}")
        writer.writerow(row)
    return stream.getvalue()


def _atomic_write_text_exclusive(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def _verify_publication_control_ledgers(inventory: CacheInventory) -> None:
    """Re-read and re-hash immutable control ledgers at every publication gate."""
    root = inventory.cache_dir
    marker_path = root / CACHE_MARKER
    if p0b_audit.artifact_identity(marker_path, root=root) != dict(
            inventory.cache_marker_identity):
        raise ValueError("cache completion marker changed before publication")
    if _read_json_object(marker_path) != dict(inventory.cache_acceptance):
        raise ValueError("cache completion marker payload changed before publication")
    for filename, expected_payload in (
            ("audit_contract.json", inventory.runtime_contract),
            ("manifest.json", inventory.manifest)):
        path = root / filename
        expected_identity = inventory.declared_cache_artifact_identities.get(filename)
        if expected_identity is None:
            raise ValueError(f"cache marker does not bind {filename}")
        _verify_identity(path, expected_identity, root)
        if _read_json_object(path) != dict(expected_payload):
            raise ValueError(f"{filename} payload changed before publication")


def _capture_current_publication_context(
        inventory: CacheInventory) -> dict[str, Any]:
    """Capture current process/repository/source identity; tests monkeypatch this."""
    source_paths = tuple(inventory.identity_validation["source_provenance"])
    dependency = inventory.identity_validation["preflight_dependency"]
    preflight_path = Path(dependency["path"]).resolve()
    return {
        "git": p0b_audit.current_git_provenance(
            REPO_ROOT, inventory.contract["version_control"]),
        "python_environment": p0b_audit.current_python_environment(),
        "source_provenance": p0b_audit.current_source_provenance(source_paths),
        "preflight_acceptance": {
            "path": str(preflight_path),
            "sha256": p0b_audit.sha256_file(preflight_path),
            "size_bytes": int(preflight_path.stat().st_size),
        },
    }


def _validate_publication_context(
        inventory: CacheInventory, phase: str) -> dict[str, Any]:
    """Require the current publisher to equal the immutable cache runtime."""
    _verify_publication_control_ledgers(inventory)
    current = _capture_current_publication_context(inventory)
    expected_git = inventory.runtime_contract["git"]
    if current.get("git") != expected_git:
        raise ValueError("current publication Git identity differs from cache runtime")
    expected_environment = inventory.identity_validation["python_environment"]
    if current.get("python_environment") != expected_environment:
        raise ValueError(
            "current publication Python environment differs from cache runtime")
    expected_sources = inventory.identity_validation["source_provenance"]
    if current.get("source_provenance") != expected_sources:
        raise ValueError(
            "current publication source provenance differs from cache runtime")
    dependency = inventory.identity_validation["preflight_dependency"]
    expected_preflight = {
        "path": str(Path(dependency["path"]).resolve()),
        "sha256": dependency["sha256"],
        "size_bytes": dependency["size_bytes"],
    }
    if current.get("preflight_acceptance") != expected_preflight:
        raise ValueError(
            "current preflight acceptance identity differs from cache runtime")
    return {
        "status": "pass",
        "phase": phase,
        "git": dict(current["git"]),
        "python_environment": dict(current["python_environment"]),
        "source_provenance": dict(current["source_provenance"]),
        "preflight_dependency": dict(dependency),
        "train_allowlist_identity": dict(
            inventory.identity_validation["train_allowlist_identity"]),
        "train_token_directory_identity": dict(
            inventory.identity_validation["train_token_directory_identity"]),
        "ISR_column_access_audit": dict(
            inventory.identity_validation["ISR_column_access_audit"]),
    }


def _publication_identity(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable identity portion of a phase-labelled publication check."""
    return {key: value for key, value in context.items() if key != "phase"}


def publish_summaries(
        inventory: CacheInventory, regime: Mapping[str, Any],
        source: Mapping[str, Any], joint: Mapping[str, Any],
        questions: Mapping[str, Any], integrity: Mapping[str, Any], *,
        prior_publication_checks: Sequence[Mapping[str, Any]] = ()) -> Path:
    """Publish all deterministic summaries, then the exclusive final marker."""
    root = inventory.cache_dir
    destinations = [root / name for name in SUMMARY_FILES] + [
        root / SUMMARY_MANIFEST, root / FINAL_MARKER]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"P0-B summary output already exists: {existing[0]}")
    publication_check = _validate_publication_context(
        inventory, "before_summary_publication")
    all_checks = [dict(value) for value in prior_publication_checks]
    if any(_publication_identity(value) != _publication_identity(publication_check)
           for value in all_checks):
        raise ValueError("P0-B publication identity changed across aggregation phases")
    all_checks.append(publication_check)
    payloads = {
        "query_regime_summary.json": _json_text(regime),
        "query_source_summary.json": _json_text(source),
        "query_joint_summary.json": _json_text(joint),
        "p0b_six_questions.json": _json_text(questions),
        "query_regime_summary.csv": _csv_text(
            regime["rows"], _REGIME_FIELDS),
        "query_source_summary.csv": _csv_text(
            source["rows"], _SOURCE_FIELDS),
        "query_joint_summary.csv": _csv_text(joint["rows"], _JOINT_FIELDS),
    }
    written: list[Path] = []
    for name in SUMMARY_FILES:
        written.append(_atomic_write_text_exclusive(root / name, payloads[name]))
    statuses = {
        record["id"]: record["status"] for record in questions["questions"]
    }
    q5_routing_status = next(
        record["evidence"]["routing_status"]
        for record in questions["questions"]
        if record["id"] == "Q5_low_altitude_drift")
    cache_identity = p0b_audit.artifact_identity(
        root / CACHE_MARKER, root=root)
    summary_manifest_payload = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "summary_complete_provisional_not_final_acceptance",
        "input_cache_acceptance": cache_identity,
        "input_manifest": p0b_audit.artifact_identity(
            root / "manifest.json", root=root),
        "input_runtime_contract": p0b_audit.artifact_identity(
            root / "audit_contract.json", root=root),
        "frozen_contract": {
            "path": str(inventory.contract_path),
            "sha256": inventory.contract_sha256,
        },
        "validated_integrity": dict(integrity),
        "six_question_statuses": statuses,
        "six_questions_assigned": len(statuses) == 6,
        "scientific_attribution_complete": False,
        "defined_profile_and_date_gates_enforced": True,
        "question_interpretation_boundaries_complete": True,
        "q5_routing_status": q5_routing_status,
        "profile_cap_status": _edge_diagnostic_rules(
            inventory.contract).profile_cap_status,
        "coordinate_enrichment": inventory.identity_validation[
            "coordinate_enrichment"],
        "publication_identity": _publication_identity(publication_check),
        "publication_context_checks": [
            {"phase": value["phase"], "status": value["status"]}
            for value in all_checks
        ],
        "summary_artifacts": [
            p0b_audit.artifact_identity(path, root=root) for path in written
        ],
    }
    summary_manifest_path = p0b_audit.atomic_write_json(
        root / SUMMARY_MANIFEST, summary_manifest_payload, overwrite=False)
    marker_payload = {
        "status": "provisional_pass",
        "attribution_complete": False,
        "scientific_attribution_complete": False,
        "six_questions_assigned": len(statuses) == 6,
        "provisional_until_fixed_observation_eligibility": True,
        "p0c_unlocked": False,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "defined_profile_and_date_gates_enforced": True,
        "question_interpretation_boundaries_complete": True,
        "q5_routing_status": q5_routing_status,
        "cache_acceptance": cache_identity,
        "frozen_contract": {
            "path": str(inventory.contract_path),
            "sha256": inventory.contract_sha256,
        },
        "validated_integrity": dict(integrity),
        "six_question_statuses": statuses,
        "profile_cap_status": _edge_diagnostic_rules(
            inventory.contract).profile_cap_status,
        "runtime_profile_cap_status": inventory.identity_validation[
            "profile_cap_status"],
        "coordinate_enrichment": inventory.identity_validation[
            "coordinate_enrichment"],
    }
    _verify_all_declared_batch_identities(inventory)
    final_publication_check = _validate_publication_context(
        inventory, "immediately_before_final_marker")
    if _publication_identity(final_publication_check) != _publication_identity(
            publication_check):
        raise ValueError("P0-B publication identity changed before final marker")
    all_checks.append(final_publication_check)
    marker_payload["batch_identity_rechecked_immediately_before_marker"] = True
    marker_payload["publication_identity"] = _publication_identity(
        final_publication_check)
    marker_payload["publication_context_checks"] = [
        {"phase": value["phase"], "status": value["status"]}
        for value in all_checks
    ]
    marker_payload["publication_identity_stable_across_all_checks"] = True
    return p0b_audit.write_completion_marker_atomically(
        root / FINAL_MARKER, marker_payload,
        written + [summary_manifest_path], artifact_root=root)


def summarize_cache(
        cache_dir: str | os.PathLike[str],
        contract_path: str | os.PathLike[str] | None = None) -> Path:
    inventory = validate_cache_inventory(cache_dir, contract_path=contract_path)
    first_publication_check = _validate_publication_context(
        inventory, "before_first_aggregation")
    first = build_fixed_summaries(inventory)
    replay_publication_check = _validate_publication_context(
        inventory, "before_second_aggregation_replay")
    if _publication_identity(replay_publication_check) != _publication_identity(
            first_publication_check):
        raise ValueError("P0-B publication identity changed before aggregation replay")
    second = build_fixed_summaries(inventory)
    labels = ("query_regime", "query_source", "query_joint", "six_questions",
              "integrity")
    for label, first_payload, second_payload in zip(labels, first, second):
        if _json_text(first_payload) != _json_text(second_payload):
            raise ValueError(
                "two in-memory cache aggregations differ after canonical field "
                f"serialization: {label}")
    regime, source, joint, questions, integrity = first
    integrity["in_memory_aggregation_replays"] = 2
    integrity["canonical_field_serialization_replay_equal"] = True
    return publish_summaries(
        inventory, regime, source, joint, questions, integrity,
        prior_publication_checks=(
            first_publication_check, replay_publication_check))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize an immutable M2-W2 P0-B audit cache.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--contract", default=None,
        help="Optional explicit frozen P0-B contract; must match runtime SHA256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summarize_cache(args.cache_dir, contract_path=args.contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
