"""Canonical, side-effect-free contracts for the hybrid M2-W v14 run."""

import math
import statistics

HYBRID_DOMAIN_SEMANTICS = 'hybrid_120_500_model_200_500_observation_v1'
TRAINING_PROTOCOL_REVISION = 'v14_hybrid_gram_calibration_v2'
GRAM_WEIGHT_RESOLUTION_SEMANTICS = (
    'target_over_median_no_lower_clip_reject_gt_one_v1')
GRADIENT_PARAMETER_SCOPE = (
    'all_current_stage_trainable_parameters_v1')
CALIBRATION_PRECISION = 'match_analysis_autocast_v1'

INPUT_DATA_IDENTITY_KEYS = (
    'fy_path', 'fy_profile_path', 'fy_profile_index_path',
    'fy_qc_report_path', 'cosmic_path', 'cosmic_profile_index_path',
    'cosmic_qc_report_path', 'iri_proxy_path', 'iri_hmf2_path',
    'iri_nmf2_path', 'sw_path', 'representativeness_kernel_path',
    'date_split_manifest',
)


def _range(value):
    return [float(item) for item in value]


def canonical_v14_low_altitude_protocol():
    """Fixed low-altitude IRI prior, independent of Gram calibration."""
    return {
        'range_km': [120.0, 200.0],
        'semantics': 'soft_iri_background_zero_analysis_increment_v1',
        'anchor_levels_km': [float(value) for value in range(120, 200, 10)],
        'profiles_per_source': 16,
        'background_weight': 0.02,
        'analysis_weight': 0.01,
        'low_altitude_gradient_ratio_max': 0.25,
    }


def build_v14_low_altitude_protocol(config):
    """Normalize the configured prior so it can be compared byte-for-byte."""
    return {
        'range_km': _range(config.get('low_altitude_prior_range') or ()),
        'semantics': config.get('low_altitude_prior_semantics'),
        'anchor_levels_km': _range(
            config.get('low_altitude_anchor_levels_km') or ()),
        'profiles_per_source': int(
            config.get('low_altitude_anchor_profiles_per_source', 0)),
        'background_weight': float(
            config.get('w_low_altitude_background_iri', 0.0)),
        'analysis_weight': float(
            config.get('w_low_altitude_analysis_increment', 0.0)),
        'low_altitude_gradient_ratio_max': float(
            config.get('low_altitude_gradient_ratio_max', 0.0)),
    }


def canonical_v14_training_protocol(smoke_run=False):
    """The complete immutable v14 protocol for either smoke or full training."""
    return {
        'training_protocol_revision': TRAINING_PROTOCOL_REVISION,
        'seed': 42,
        'smoke_run': bool(smoke_run),
        'background_epochs': 1 if smoke_run else 5,
        'analysis_epochs': 1 if smoke_run else 10,
        'background_trust_gate_enabled': False,
        'use_observation_gram_loss': True,
        'analysis_exact_mode_loss': True,
        'gram_gradient_target': 0.02,
        'gram_calibration_batches': 20,
        'gram_weight_resolution_semantics': GRAM_WEIGHT_RESOLUTION_SEMANTICS,
        'gradient_parameter_scope': GRADIENT_PARAMETER_SCOPE,
        'calibration_precision': CALIBRATION_PRECISION,
        'gram_require_all_batches_finite_positive': True,
        'gram_preflight_ratio_bounds': [0.016, 0.024],
        'gram_coverage_policy': 'diagnostic_only_v1',
        'smoke_auxiliary_gradient_ratio_max': 0.25,
        'low_altitude_anchor_selection': (
            'first_16_unique_profiles_per_source_per_batch_first_record_v1'),
        'low_altitude_anchor_grouping': 'source_profile_id_profile_balanced_v1',
        'low_altitude_neighbor_profile_semantics': (
            'synthetic_query_no_target_profile_exclusion_v1'),
        'low_altitude_prior_protocol': canonical_v14_low_altitude_protocol(),
    }


def build_v14_training_protocol(config):
    """Normalize a run configuration using the same shape as the canonical one."""
    return {
        'training_protocol_revision': config.get('training_protocol_revision'),
        'seed': int(config.get('seed', -1)),
        'smoke_run': bool(config.get('smoke_run', False)),
        'background_epochs': int(config.get('background_epochs', -1)),
        'analysis_epochs': int(config.get('analysis_epochs', -1)),
        'background_trust_gate_enabled': bool(
            config.get('background_trust_gate_enabled', False)),
        'use_observation_gram_loss': bool(
            config.get('use_observation_gram_loss', False)),
        'analysis_exact_mode_loss': bool(
            config.get('analysis_exact_mode_loss', False)),
        'gram_gradient_target': float(config.get('gram_gradient_target', 0.0)),
        'gram_calibration_batches': int(
            config.get('gram_calibration_batches', 0)),
        'gram_weight_resolution_semantics': config.get(
            'gram_weight_resolution_semantics'),
        'gradient_parameter_scope': config.get('gradient_parameter_scope'),
        'calibration_precision': config.get('calibration_precision'),
        'gram_require_all_batches_finite_positive': bool(config.get(
            'gram_require_all_batches_finite_positive', False)),
        'gram_preflight_ratio_bounds': _range(
            config.get('gram_preflight_ratio_bounds') or ()),
        'gram_coverage_policy': config.get('gram_coverage_policy'),
        'smoke_auxiliary_gradient_ratio_max': float(config.get(
            'smoke_auxiliary_gradient_ratio_max', 0.0)),
        'low_altitude_anchor_selection': config.get(
            'low_altitude_anchor_selection',
            'first_16_unique_profiles_per_source_per_batch_first_record_v1'),
        'low_altitude_anchor_grouping': config.get(
            'low_altitude_anchor_grouping',
            'source_profile_id_profile_balanced_v1'),
        'low_altitude_neighbor_profile_semantics': config.get(
            'low_altitude_neighbor_profile_semantics',
            'synthetic_query_no_target_profile_exclusion_v1'),
        'low_altitude_prior_protocol': build_v14_low_altitude_protocol(config),
    }


def canonical_v14_config(smoke_run=False):
    """Config fields that must match the fixed protocol before training."""
    protocol = canonical_v14_training_protocol(smoke_run)
    return {
        'checkpoint_format_version': 14,
        'model_domain_semantics': HYBRID_DOMAIN_SEMANTICS,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (200.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
        'low_altitude_prior_range': (120.0, 200.0),
        'low_altitude_prior_semantics': (
            'soft_iri_background_zero_analysis_increment_v1'),
        'low_altitude_anchor_levels_km': tuple(float(value) for value in range(120, 200, 10)),
        'low_altitude_anchor_profiles_per_source': 16,
        'w_low_altitude_background_iri': 0.02,
        'w_low_altitude_analysis_increment': 0.01,
        'low_altitude_gradient_ratio_max': 0.25,
        **{key: value for key, value in protocol.items()
           if key != 'low_altitude_prior_protocol'},
    }


def resolve_v14_gram_weight(raw_ratios, target_ratio):
    """Return ``target / median`` or reject an unsafe v14 calibration."""
    ratios = [float(value) for value in raw_ratios]
    target_ratio = float(target_ratio)
    if (not ratios or not all(math.isfinite(value) and value > 0.0
                              for value in ratios)):
        raise ValueError('v14 Gram calibration requires finite positive ratios')
    if not math.isfinite(target_ratio) or not 0.0 < target_ratio <= 1.0:
        raise ValueError('Gram gradient target must be finite and in (0, 1]')
    median = float(statistics.median(ratios))
    resolved = target_ratio / median
    # ``float`` is IEEE-754 binary64; this direct lower bound is the float32
    # underflow condition without importing a tensor framework into the contract.
    if (not math.isfinite(resolved) or resolved <= 0.0
            or resolved < 1.401298464324817e-45 or resolved > 1.0):
        raise ValueError('v14 Gram calibration resolved an invalid weight')
    return float(resolved), median


def valid_v14_gram_calibration(record, protocol):
    """Validate the persisted 20-batch v14 calibration without model imports."""
    if not isinstance(record, dict) or not isinstance(protocol, dict):
        return False
    expected_batches = int(protocol['gram_calibration_batches'])
    ratios = record.get('raw_ratios')
    if (expected_batches != 20
            or record.get('requested_batches') != expected_batches
            or record.get('processed_batches') != expected_batches
            or record.get('valid_batches') != expected_batches
            or not isinstance(ratios, list) or len(ratios) != expected_batches):
        return False
    try:
        resolved, median = resolve_v14_gram_weight(
            ratios, protocol['gram_gradient_target'])
        achieved = float(record['achieved_median_ratio'])
        recorded_median = float(record['median_ratio'])
        recorded_weight = float(record['resolved_weight'])
    except (KeyError, TypeError, ValueError):
        return False
    lower, upper = [float(value) for value in protocol['gram_preflight_ratio_bounds']]
    return (
        math.isclose(recorded_median, median, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(recorded_weight, resolved, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(achieved, resolved * median,
                         rel_tol=1e-9, abs_tol=1e-12)
        and lower <= achieved <= upper
        and record.get('parameter_tensor_count') == 8
        and record.get('parameter_count') == 18119
        and record.get('gradient_parameter_scope')
        == protocol['gradient_parameter_scope']
        and record.get('calibration_precision') == protocol['calibration_precision']
        and record.get('weight_resolution_semantics')
        == protocol['gram_weight_resolution_semantics']
        and record.get('require_all_batches_finite_positive') is True
    )
