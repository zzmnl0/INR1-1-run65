"""Two-stage run66 training for the feature-space local LETKF model."""

import hashlib
import json
import os
import sys
import time
import copy
import random
from datetime import datetime, timedelta
from typing import Mapping

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from ..config_mdia import get_config_mdia
    from .fsia_model import (
        FSIA_INR_Model,
        solve_density_modes,
    )
    from .mdia_model import _compute_dip_features
    from .physics_losses_mdia import profile_huber_loss, second_difference_loss
    from .sliding_dataset import (
        SlidingWindowBatchProcessor,
        attach_representativeness_weight,
        attach_observation_background,
        empirical_covariance_token_targets,
        load_empirical_covariance_targets,
        load_representativeness_kernel,
        query_observation_payload,
    )
    from .plotting import plot_training_curves
    from .m00_observation import (
        M00_OBSERVATION_V2_SEMANTICS,
        M00_QD_RELIABILITY_SEMANTICS,
        validate_observation_config,
        validate_low_dim_assimilation_config,
        validate_frozen_m00_config,
        load_observation_calibration,
        load_magnetic_sidecar,
        m00_observation_error_loss,
        calibration_to_dict,
        validate_strict_v2_calibration,
    )
except ImportError:
    from config_mdia import get_config_mdia
    from fsia_model import FSIA_INR_Model, solve_density_modes
    from mdia_model import _compute_dip_features
    from physics_losses_mdia import profile_huber_loss, second_difference_loss
    from sliding_dataset import (
        SlidingWindowBatchProcessor,
        attach_representativeness_weight,
        attach_observation_background,
        empirical_covariance_token_targets,
        load_empirical_covariance_targets,
        load_representativeness_kernel,
        query_observation_payload,
    )
    from plotting import plot_training_curves
    from m00_observation import (
        M00_OBSERVATION_V2_SEMANTICS,
        M00_QD_RELIABILITY_SEMANTICS,
        validate_observation_config,
        validate_low_dim_assimilation_config,
        validate_frozen_m00_config,
        load_observation_calibration,
        load_magnetic_sidecar,
        m00_observation_error_loss,
        calibration_to_dict,
        validate_strict_v2_calibration,
    )

try:
    from .v14_contract import (
        A2S1_DOMAIN_SEMANTICS,
        CALIBRATION_PRECISION,
        GRADIENT_PARAMETER_SCOPE,
        HYBRID_DOMAIN_SEMANTICS,
        INPUT_DATA_IDENTITY_KEYS,
        build_a2s1_training_protocol,
        build_v14_low_altitude_protocol,
        build_v14_training_protocol,
        canonical_a2s1_config,
        canonical_a2s1_training_protocol,
        canonical_v14_config,
        canonical_v14_low_altitude_protocol,
        canonical_v14_training_protocol,
        resolve_v14_gram_weight,
        valid_v14_gram_calibration,
    )
except ImportError:
    from v14_contract import (
        A2S1_DOMAIN_SEMANTICS,
        CALIBRATION_PRECISION,
        GRADIENT_PARAMETER_SCOPE,
        HYBRID_DOMAIN_SEMANTICS,
        INPUT_DATA_IDENTITY_KEYS,
        build_a2s1_training_protocol,
        build_v14_low_altitude_protocol,
        build_v14_training_protocol,
        canonical_a2s1_config,
        canonical_a2s1_training_protocol,
        canonical_v14_config,
        canonical_v14_low_altitude_protocol,
        canonical_v14_training_protocol,
        resolve_v14_gram_weight,
        valid_v14_gram_calibration,
    )

from data_managers import SpaceWeatherManager, IRINeuralProxy
from data_managers.FY_dataloader import (
    COSMICDataset,
    FYNeighborhoodIndex,
    FY3D_Dataset,
    COSMICNeighborhoodIndex,
    ProfileTimeBinSampler,
    get_dataloaders,
    get_cosmic_dataloader,
)


_HYBRID_DOMAIN_SEMANTICS = HYBRID_DOMAIN_SEMANTICS
_A2S1_DOMAIN_SEMANTICS = A2S1_DOMAIN_SEMANTICS
_STRICT_LOW_PRIOR_DOMAINS = (_HYBRID_DOMAIN_SEMANTICS, _A2S1_DOMAIN_SEMANTICS)
_HYBRID_LOW_ALTITUDE_PRIOR_SEMANTICS = (
    'soft_iri_background_zero_analysis_increment_v1')
_HYBRID_LOW_ALTITUDE_LEVELS = tuple(float(value) for value in range(120, 200, 10))
_NATURAL_BACKGROUND_SAMPLING = 'source_complete_profile_pass_v1'
_NATURAL_BACKGROUND_TRAINING = 'a1s_natural_epoch_train_only_v1'


def _natural_background_enabled(config):
    return str(config.get('background_sampling_semantics', '')).replace(
        '-', '_') == _NATURAL_BACKGROUND_SAMPLING


def _reset_natural_sampler_state(train_loader, cosmic_train_loader,
                                  start_epoch=0):
    """Make formal epoch seeds independent of calibration/diagnostic passes."""
    for loader in (train_loader, cosmic_train_loader):
        sampler = getattr(loader, 'batch_sampler', None)
        if (sampler is None
                or str(getattr(sampler, 'sampling_semantics', '')).replace(
                    '-', '_') != _NATURAL_BACKGROUND_SAMPLING):
            continue
        sampler._natural_batches = None
        sampler._iteration = int(start_epoch)


def _observation_alt_range(config):
    """Satellite targets and tokens; legacy contracts reuse model bounds."""
    return tuple(map(float, config.get('observation_alt_range')
                     or config.get('alt_range', (120.0, 500.0))))


def _low_altitude_prior_protocol(config):
    """Return the immutable low-altitude protocol, if enabled."""
    if config.get('model_domain_semantics') not in _STRICT_LOW_PRIOR_DOMAINS:
        return None
    return build_v14_low_altitude_protocol(config)


def _training_protocol_signature(config):
    """Return the immutable training protocol stored with resumable state."""
    domain = config.get('model_domain_semantics')
    if domain == _A2S1_DOMAIN_SEMANTICS:
        return build_a2s1_training_protocol(config)
    if domain != _HYBRID_DOMAIN_SEMANTICS:
        return None
    return build_v14_training_protocol(config)


def _validate_hybrid_low_altitude_protocol(config):
    """Reject strict low-prior runs that could silently change their contract."""
    domain = config.get('model_domain_semantics')
    if domain not in _STRICT_LOW_PRIOR_DOMAINS:
        return
    smoke = bool(config.get('smoke_run', False))
    expected = (canonical_a2s1_config(smoke)
                if domain == _A2S1_DOMAIN_SEMANTICS
                else canonical_v14_config(smoke))
    expected_protocol = (canonical_a2s1_training_protocol(smoke)
                         if domain == _A2S1_DOMAIN_SEMANTICS
                         else canonical_v14_training_protocol(smoke))
    expected.update({'basis_dim': 64, 'enkf_n_members': 8, 'r_mode': 'global'})
    mismatches = {}
    for key, expected_value in expected.items():
        actual = config.get(key)
        if isinstance(expected_value, tuple):
            actual = tuple(map(float, actual or ()))
            matches = actual == expected_value
        elif isinstance(expected_value, float):
            try:
                actual = float(actual)
                matches = bool(np.isclose(actual, expected_value))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected_value
        if not matches:
            mismatches[key] = (actual, expected_value)
    if config.get('background_trust_gate_enabled', False):
        mismatches['background_trust_gate_enabled'] = (True, False)
    if _low_altitude_prior_protocol(config) != canonical_v14_low_altitude_protocol():
        mismatches['low_altitude_prior_protocol'] = (
            _low_altitude_prior_protocol(config), canonical_v14_low_altitude_protocol())
    if _training_protocol_signature(config) != expected_protocol:
        mismatches['training_protocol'] = (
            _training_protocol_signature(config),
            expected_protocol)
    if (config.get('training_protocol') is not None
            and config.get('training_protocol')
            != expected_protocol):
        mismatches['config.training_protocol'] = (
            config.get('training_protocol'),
            expected_protocol)
    if mismatches:
        raise ValueError(f'v14 training contract mismatch: {mismatches}')


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(path, skip=False):
    """Return a checkpoint hash, or None when no checkpoint was selected."""
    return None if skip or not path else _sha256_file(path)


def _resolve_input_data_sha256(config):
    """Verify each manifest-pinned v14 input before a state can be restored."""
    if config.get('skip_hash_verification', False):
        return None
    path = os.path.join(config['save_dir'], 'run_manifest.json')
    if not os.path.isfile(path):
        if config.get('model_domain_semantics') in _STRICT_LOW_PRIOR_DOMAINS:
            raise FileNotFoundError('strict low-prior training requires a run manifest')
        return None
    with open(path, encoding='utf-8') as stream:
        manifest = json.load(stream)
    identities = manifest.get('data_identity', {})
    if not isinstance(identities, dict):
        raise ValueError('run manifest data_identity must be an object')
    result = {}
    for key, identity in identities.items():
        if not isinstance(identity, dict) or not identity.get('path'):
            raise ValueError(f'run manifest has invalid input identity for {key}')
        expected = identity.get('sha256')
        actual = _sha256_file(identity['path'])
        if actual != expected:
            raise ValueError(f'input SHA256 differs from run manifest for {key}')
        result[key] = actual
    if config.get('model_domain_semantics') in _STRICT_LOW_PRIOR_DOMAINS:
        if manifest.get('config', {}).get('input_data_sha256') != result:
            raise ValueError(
                'strict low-prior run manifest config input_data_sha256 differs from data identity')
        required = set(INPUT_DATA_IDENTITY_KEYS) & {
            key for key in INPUT_DATA_IDENTITY_KEYS
            if config.get(key) and os.path.isfile(config[key])}
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(f'strict low-prior input identity lacks {missing}')
        if ('date_split_manifest' not in result
                and not _profile_split_enabled(config)):
            raise ValueError('strict low-prior input identity must include date_split_manifest')
    return result


def _atomic_torch_save(state, path):
    temp_path = f'{path}.tmp'
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def _validate_date_split(partitions, total_days, development_days,
                         locked_test_days):
    expected = {'train', 'development', 'locked_test'}
    if set(partitions) != expected:
        raise ValueError(f'date split must define {sorted(expected)}')
    normalized = {
        name: np.asarray(days, dtype=np.int64)
        for name, days in partitions.items()}
    for name, days in normalized.items():
        if (days.ndim != 1 or len(np.unique(days)) != len(days)
                or np.any(days < 0) or np.any(days >= total_days)):
            raise ValueError(f'invalid {name} date split')
    if len(normalized['development']) != development_days:
        raise ValueError('development date count mismatch')
    if len(normalized['locked_test']) != locked_test_days:
        raise ValueError('locked-test date count mismatch')
    combined = np.concatenate(list(normalized.values()))
    if len(combined) != total_days or len(np.unique(combined)) != total_days:
        raise ValueError('date split must partition every relative day exactly once')
    return {name: np.sort(days).tolist() for name, days in normalized.items()}


_A1S6_DATE_SPLIT_SEMANTICS = 'a1s_train_plus_locked_dev5_v1'
_A1S_NATIVE_DATE_SPLIT_SEMANTICS = 'a1s_native_date_split_v1'


def _load_or_create_date_split(config):
    """Create one deterministic UTC-date split shared by FY and COSMIC."""
    if not config.get('use_date_blocked_split', False):
        return None, None
    path = config.get('date_split_manifest')
    if not path:
        raise ValueError(
            'date_split_manifest is required for date-blocked splitting')
    total_hours = float(config['total_hours'])
    total_days = int(round(total_hours / 24.0))
    if not np.isclose(total_days * 24.0, total_hours):
        raise ValueError('total_hours must contain an integer number of UTC days')
    development_days = int(config.get('development_days', 5))
    locked_test_days = int(config.get('locked_test_days', 5))
    raw_date_semantics = config.get('date_split_semantics')
    date_semantics = (
        '' if raw_date_semantics in (None, '')
        else str(raw_date_semantics).replace('-', '_'))
    if date_semantics not in ('', _A1S6_DATE_SPLIT_SEMANTICS,
                              _A1S_NATIVE_DATE_SPLIT_SEMANTICS):
        raise ValueError(f'unsupported date-split semantics: {date_semantics}')
    a1s6 = date_semantics == _A1S6_DATE_SPLIT_SEMANTICS
    if min(development_days, locked_test_days) < 1:
        raise ValueError('development_days and locked_test_days must be positive')
    if development_days != locked_test_days:
        raise ValueError(
            'M0 requires equal development and locked-test date counts')
    if total_days < 3 * development_days:
        raise ValueError('too few UTC days for blocked train/development/test split')

    identity = {
        'schema_version': 1,
        'seed': int(config['seed']),
        'start_date_str': config['start_date_str'],
        'total_days': total_days,
        'development_days': development_days,
        'locked_test_days': locked_test_days,
    }
    if a1s6:
        identity['semantics'] = _A1S6_DATE_SPLIT_SEMANTICS
    path = os.path.abspath(path)
    if a1s6 and not os.path.isfile(path):
        raise FileNotFoundError(
            f'A1-S6 requires the existing A1-S date manifest: {path}')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        for key, value in identity.items():
            if key == 'semantics':
                continue
            if manifest.get(key) != value:
                raise ValueError(f'date split manifest mismatch for {key}')
        source_partitions = _validate_date_split(
            manifest.get('partitions', {}), total_days,
            development_days, locked_test_days)
        if a1s6:
            partitions = {
                'train': sorted(set(source_partitions['train']) | set(source_partitions['locked_test'])),
                'development': list(source_partitions['development']),
                'locked_test': [],
            }
        else:
            partitions = source_partitions
    else:
        rng = np.random.default_rng(identity['seed'])
        development = []
        locked_test = []
        for block in np.array_split(
                np.arange(total_days), development_days):
            if len(block) < 2:
                raise ValueError('date block is too small for two held-out dates')
            shuffled = rng.permutation(block)
            development.append(int(shuffled[0]))
            locked_test.append(int(shuffled[1]))
        held_out = set(development + locked_test)
        partitions = _validate_date_split({
            'train': [
                day for day in range(total_days) if day not in held_out],
            'development': development,
            'locked_test': locked_test,
        }, total_days, development_days, locked_test_days)
        start = datetime.strptime(
            identity['start_date_str'], '%Y-%m-%d %H:%M:%S')
        manifest = {
            **identity,
            'partitions': partitions,
            'utc_dates': {
                name: [
                    (start + timedelta(days=day)).strftime('%Y-%m-%d')
                    for day in days]
                for name, days in partitions.items()},
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(
                manifest, stream, ensure_ascii=False, indent=2,
                sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    if a1s6:
        if set(partitions['development']) & set(partitions['train']):
            raise ValueError('A1-S6 train/development date leakage')
        if len(partitions['train']) + len(partitions['development']) != total_days:
            raise ValueError('A1-S6 effective split must cover all dates')
        audit_path = os.path.join(config.get('save_dir', os.path.dirname(path)),
                                  'a1s6_date_split_audit.json')
        audit = {
            'semantics': _A1S6_DATE_SPLIT_SEMANTICS,
            'source_manifest': path,
            'source_partitions': source_partitions if 'source_partitions' in locals() else None,
            'effective_partitions': partitions,
            'former_locked_test_dates': list(source_partitions['locked_test']) if 'source_partitions' in locals() else [],
            'profile_partition_complete': True,
            'profile_partition_disjoint': True,
        }
        os.makedirs(os.path.dirname(audit_path) or '.', exist_ok=True)
        with open(audit_path + '.tmp', 'w', encoding='utf-8') as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(audit_path + '.tmp', audit_path)
        return partitions, {
            **identity,
            'path': path,
            'sha256': (None if config.get('skip_hash_verification', False)
                       else _sha256_file(path)),
            'partitions': partitions,
            'former_locked_test_dates': list(source_partitions['locked_test']) if 'source_partitions' in locals() else [],
            'locked_test': [],
        }
    return partitions, {
        'path': path,
        'sha256': (None if config.get('skip_hash_verification', False)
                   else _sha256_file(path)),
        'partitions': partitions,
    }


def _loader_profile_ids(loader):
    """Return active profile IDs, respecting an optional screen subset."""
    sampler = loader.batch_sampler
    groups = [
        indices for bin_groups in sampler.profiles_by_bin.values()
        for indices in bin_groups]
    profile_ids = np.array(sorted({
        int(loader.dataset.profile_ids[indices[0]]) for indices in groups
    }), dtype=np.int64)
    if profile_ids.size == 0:
        raise ValueError('loader contains no active profiles')
    return profile_ids


_A1S2_PROFILE_SPLIT_SEMANTICS = 'source_time_local_block80_dev20_v1'
_A1S2_BACKGROUND_SEMANTICS = (
    'qc_v2_source_time_local_block80_dev20_train_only_v1')
_COND_ADAPTER_SEMANTICS = 'lt_signed_abs_sini_periodic_lon_v1'
_COND_ADAPTER_SCOPE = 'condition_adapter_only_v1'
_COND_ADAPTER_SELECTOR = 'cond_adapter_bias_constrained_v1'
_A1S7_BACKGROUND_SEMANTICS = 'a1s7_m00_b_shared_dual_maglat_v1'
_A1S7_ADAPTER_SEMANTICS = 'maglat2_smooth_v1'
_A1S7_ADAPTER_ALIAS = 'dual_maglat_smooth_v1'
_A1S7_ADAPTER_SCOPE = 'condition_adapter_only_dual_maglat_v1'
_A1S7_GATE_SEMANTICS = 'abs_sinI_025_050_smoothstep_v1'
_A1S7_SAMPLING_SEMANTICS = 'source_date_physical_stratum_balanced_v1'
_A1S7_CHECKPOINT_SEMANTICS = 'a1s7_m00_b_v1'
_A1S8_BACKGROUND_SEMANTICS = 'a1s8_m00_preserve_mixed_gate_v1'
_A1S8_SAMPLING_SEMANTICS = 'a1s8_mixed_gate_aware_v1'
_A1S8_GATE_BALANCE_SEMANTICS = 'effective_gate_mass_clipped_v1'
_A1S8_ANCHOR_SEMANTICS = 'a1s_output_trust_region_v1'
_A1S8_CHECKPOINT_SEMANTICS = 'a1s8_m00_preserve_mixed_gate_v1'
_A1S8_SOURCE_MIX_SEMANTICS = 'fy3_cosmic1_v1'
_A1S9_BACKGROUND_SEMANTICS = 'a1s9_m00_strict_baseline_v1'
_A1S9_CHECKPOINT_SEMANTICS = 'a1s9_m00_strict_baseline_v1'
_A1S9_ADAPTER_SCOPE = _A1S7_ADAPTER_SCOPE
_M00_OBSERVATION_SCOPE = 'm00_observation_error_only_v1'
_M00_OBSERVATION_BACKGROUND_SEMANTICS = _M00_OBSERVATION_SCOPE
_M00_DECODER_TRUST_SCOPE = 'm00_decoder_last_trust_region_v1'
_M00_PEAK_LOWRANK_SCOPE = 'm00_peak_relative_lowrank3_v1'
_M00_CANDIDATE_SCOPES = (_M00_DECODER_TRUST_SCOPE, _M00_PEAK_LOWRANK_SCOPE)
_COND_ADAPTER_JOINT_FRESH_SCOPE = 'condition_adapter_joint_fresh_v1'
_COND_ADAPTER_JOINT_FRESH_SELECTOR = (
    'cond_adapter_joint_fresh_bias_constrained_v1')
_CONDITIONAL_SHAPE_LOSS = 'conditional_profile_centered_scale_v1'
_CONDITIONAL_SHAPE_SELECTOR = 'conditional_shape_constrained_v1'
_M00_NATURAL_SELECTOR = 'm00_natural_rmse_protected_v1'
_M00_NIGHT_GENERALIZATION_SELECTOR = 'm00_night_generalization_v1'
_M00_NIGHT_GENERALIZATION_V2_SELECTOR = 'm00_night_generalization_v2'
_BACKGROUND_BIAS_GROUPING_LEGACY = 'height_daynight_v1'
_BACKGROUND_BIAS_GROUPING_MAGNETIC = 'height_daynight_plus_magnetic_v1'
_BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT = (
    'height_daynight_plus_magnetic_daynight_v1')
_BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2 = (
    'height_daynight_plus_magnetic_daynight_v2')
_BACKGROUND_BIAS_GROUPINGS = (
    _BACKGROUND_BIAS_GROUPING_LEGACY,
    _BACKGROUND_BIAS_GROUPING_MAGNETIC,
    _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT,
    _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2,
)
_BACKGROUND_BIAS_MAGNETIC_LABELS = (
    'abs_sinI_lt_0.25', 'abs_sinI_0.25_0.5', 'abs_sinI_ge_0.5')
_BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_LABELS = tuple(
    f'{mag}_{period}'
    for mag in _BACKGROUND_BIAS_MAGNETIC_LABELS
    for period in ('night', 'day'))
_BACKGROUND_BIAS_MAGNETIC_SEMANTICS = (
    'dipole_abs_sin_inclination_bins_<0.25_[0.25,0.5)_>=0.5_v1')
_BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_SEMANTICS = (
    'dipole_abs_sin_inclination_bins_x_local_time_daynight_v1')
_BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_V2_SEMANTICS = (
    'dipole_abs_sin_inclination_bins_x_local_time_daynight_v2')
_CAUSAL_HISTORY_ANALYSIS_SCOPE = 'causal_history_only_v1'
_CAUSAL_HISTORY_OUTPUT_SEMANTICS = 'causal_context_background_v1'
_CAUSAL_PAST_TIME_SUPPORT = 'causal_past_lag_0p5_1p5_v1'
_CAUSAL_FROZEN_R_SOURCE = 'a1s2_frozen_global_v1'
_A1S_FROZEN_R_SOURCE = 'a1s_frozen_global_v1'
_CAUSAL_BACKGROUND_ANCHORS = ('a1s2_m00_v1', 'a1s_m00_v1')


def _resolve_m00_observation_calibration(config):
    """Load and freeze the optional Train-only M00 observation table once."""
    semantics = str(config.get(
        'm00_observation_calibration_semantics', 'disabled'))
    qd_strata = str(config.get(
        'magnetic_stratum_semantics', 'disabled')).startswith('qd_')
    if (semantics == M00_QD_RELIABILITY_SEMANTICS) != qd_strata:
        raise ValueError(
            'QD reliability calibration requires the explicit QD magnetic '
            'stratum semantics, and no other calibration may use QD strata')
    if semantics == M00_QD_RELIABILITY_SEMANTICS:
        missing = [source for source in ('fy', 'cosmic')
                   if not config.get(f'm00_observation_qd_sidecar_{source}')]
        if missing:
            raise ValueError(
                'QD reliability calibration requires FY/COSMIC QD sidecars: '
                + ', '.join(missing))
    validate_observation_config(config)
    if semantics == 'disabled':
        config['m00_calibration_applied'] = False
        return None
    payload = config.get('m00_observation_calibration')
    if payload is None:
        merged = {}
        for path, table_kind in (
                (config.get('m00_observation_bias_path'), 'bias'),
                (config.get('m00_observation_r_path'), 'variance')):
            if path:
                loaded = load_observation_calibration(
                    path, table_kind=table_kind).to_dict()
                for key in ('bias', 'variance', 'representativeness',
                            'localization'):
                    if loaded.get(key):
                        merged.setdefault(key, {}).update(loaded[key])
                merged.setdefault('semantics', loaded.get('semantics'))
                merged.setdefault('variance_floor',
                                  loaded.get('variance_floor', 1e-6))
                merged.setdefault('variance_cap',
                                  loaded.get('variance_cap', 1e6))
        payload = merged
    payload = calibration_to_dict(payload)
    if not payload.get('bias') and not payload.get('variance'):
        raise ValueError('M00 calibration table has no bias or variance cells')
    config['m00_observation_calibration'] = payload
    if semantics in {
            M00_OBSERVATION_V2_SEMANTICS,
            M00_QD_RELIABILITY_SEMANTICS,
    }:
        validate_strict_v2_calibration(payload)
        config['m00_observation_strict_magnetic'] = True
    config['m00_calibration_applied'] = True
    config['m00_calibration_frozen'] = True
    config['m00_calibration_semantics'] = semantics
    artifact = os.path.join(config.get('save_dir', '.'),
                            'm00_observation_calibration_resolved.json')
    os.makedirs(os.path.dirname(artifact) or '.', exist_ok=True)
    with open(f'{artifact}.tmp', 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(f'{artifact}.tmp', artifact)
    config['m00_calibration_artifact'] = artifact
    return payload
_A1S2_FROZEN_R = {'FY': 0.0339414440, 'COSMIC': 0.0218319837}
_A1S_FROZEN_R = {'FY': 0.03446704712846726, 'COSMIC': 0.02256251445677149}


def _a1s7_enabled(config):
    """Whether the explicit A1-S7 M00-B contract is active."""
    if '_A1S8_BACKGROUND_SEMANTICS' in globals() and _a1s8_enabled(config):
        return False
    return (str(config.get('background_training_semantics', '')).replace(
        '-', '_') == _A1S7_BACKGROUND_SEMANTICS
        or config.get('background_condition_adapter') in (
            _A1S7_ADAPTER_SEMANTICS, _A1S7_ADAPTER_ALIAS)
        or config.get('checkpoint_semantic') == _A1S7_CHECKPOINT_SEMANTICS)


def _a1s8_enabled(config):
    """Whether the explicit A1-S8 M00-preserve contract is active."""
    semantic = str(config.get('background_training_semantics', '')).replace(
        '-', '_')
    sampling = str(config.get('background_sampling_semantics', '')).replace(
        '-', '_')
    return bool(
        semantic == _A1S8_BACKGROUND_SEMANTICS
        and sampling == _A1S8_SAMPLING_SEMANTICS
        and config.get('checkpoint_semantic') == _A1S8_CHECKPOINT_SEMANTICS)


def _a1s9_enabled(config):
    """Whether the strict A1-S9 M00 baseline-preserve contract is active."""
    semantic = str(config.get('background_training_semantics', '')).replace(
        '-', '_')
    return bool(semantic == _A1S9_BACKGROUND_SEMANTICS
                or config.get('checkpoint_semantic') == _A1S9_CHECKPOINT_SEMANTICS
                or config.get('a1s9_status') == 'active')


def _resolve_a1s8_update_budget(config):
    """Resolve A1-S8's fixed update budget, preferring explicit metadata."""
    if not _a1s8_enabled(config):
        return None
    value = config.get('background_fixed_updates_per_epoch')
    if value is None:
        # A1-S8 must keep the A1-S optimizer budget.  The seed summary is the
        # authoritative source when a caller did not pass it explicitly.
        seed = config.get('background_adapter_seed_checkpoint')
        if seed:
            summary_path = os.path.join(
                os.path.dirname(os.path.abspath(seed)),
                'training_summary.json')
            if os.path.isfile(summary_path):
                with open(summary_path, encoding='utf-8') as stream:
                    summary = json.load(stream)
                value = summary.get('background_loader_schedule', {}).get(
                    'background_epoch_batches')
    if value is None:
        value = 960
    value = int(value)
    if value < 1:
        raise ValueError('A1-S8 fixed update budget must be positive')
    config['background_fixed_updates_per_epoch'] = value
    return value


def _resolve_a1s7_update_budget(config):
    """Resolve A1-S7's fixed per-epoch update budget from A1-S metadata."""
    if not _a1s7_enabled(config):
        return None
    value = config.get('background_fixed_updates_per_epoch')
    if value is not None:
        value = int(value)
        if value < 1:
            raise ValueError('A1-S7 fixed update budget must be positive')
        return value
    seed = config.get('background_adapter_seed_checkpoint')
    if not seed:
        raise ValueError('A1-S7 cannot resolve fixed updates without seed checkpoint')
    summary_path = os.path.join(os.path.dirname(os.path.abspath(seed)),
                                'training_summary.json')
    if not os.path.isfile(summary_path):
        raise FileNotFoundError('A1-S7 seed training_summary.json is missing')
    with open(summary_path, encoding='utf-8') as stream:
        summary = json.load(stream)
    schedule = summary.get('background_loader_schedule', {})
    value = schedule.get('background_epoch_batches')
    if value is None:
        raise ValueError('A1-S7 seed summary lacks background_epoch_batches')
    value = int(value)
    if value < 1:
        raise ValueError('A1-S7 seed update budget must be positive')
    config['background_fixed_updates_per_epoch'] = value
    return value


def _profile_split_enabled(config):
    """Return whether the explicit A1-S2 profile split is requested."""
    semantics = config.get('profile_split_semantics')
    if semantics is None:
        return False
    normalized = str(semantics).replace('-', '_')
    if normalized != _A1S2_PROFILE_SPLIT_SEMANTICS:
        raise ValueError(f'unsupported profile-split semantics: {semantics}')
    return True


def _reindex_profile_dataset(dataset, profile_ids, source, partition):
    """Restrict an all-profile dataset and rebuild its sampler index state."""
    requested = np.asarray(profile_ids, dtype=np.int64)
    if (requested.ndim != 1 or requested.size == 0
            or np.any(np.diff(requested) <= 0)):
        raise ValueError(
            f'{source} {partition} profile allowlist must be sorted and non-empty')
    base_ids = np.asarray(dataset.profile_ids, dtype=np.int64)
    available = np.unique(base_ids)
    missing = np.setdiff1d(requested, available, assume_unique=True)
    if missing.size:
        raise ValueError(
            f'{source} {partition} profile split contains unavailable IDs '
            f'(first={missing[:5].tolist()})')
    mask = np.isin(base_ids, requested, assume_unique=False)
    if not mask.any():
        raise ValueError(f'{source} {partition} profile allowlist is empty after QC')
    dataset.selected_indices = np.asarray(
        dataset.selected_indices[mask], dtype=np.int64)
    dataset.profile_ids = base_ids[mask]
    dataset.bin_ids = np.asarray(dataset.bin_ids[mask], dtype=np.int64)
    order = np.argsort(dataset.bin_ids, kind='stable')
    sorted_bins = dataset.bin_ids[order]
    unique_bins, split_points = np.unique(sorted_bins, return_index=True)
    dataset.indices_by_bin = {
        int(bin_id): group
        for bin_id, group in zip(
            unique_bins, np.split(order, split_points[1:]))}
    return dataset


def _new_all_profile_dataset(config, source):
    suffix = str(source).lower()
    """Construct a full-domain dataset before applying explicit allowlists."""
    common = dict(
        mode='train',
        val_days=[],
        bin_size_hours=config['bin_size_hours'],
        use_memmap=config.get('use_memmap', True),
        val_ratio=None,
        split_seed=config['seed'],
        split_days=None,
        alt_range=_observation_alt_range(config),
        magnetic_sidecar_path=(config.get(
            f'm00_observation_qd_sidecar_{suffix}') or config.get(
            f'm00_observation_magnetic_sidecar_{suffix}')),
        magnetic_sidecar_source=source,
        magnetic_sidecar_strict=bool(config.get(
            'm00_observation_strict_magnetic', False)),
    )
    if source == 'FY':
        return FY3D_Dataset(
            config['fy_path'],
            profile_path=config.get('fy_profile_path'),
            profile_index_path=config.get('fy_profile_index_path'),
            **common)
    if source == 'COSMIC':
        return COSMICDataset(
            config['cosmic_path'],
            profile_index_path=config.get('cosmic_profile_index_path'),
            **common)
    raise ValueError(f'unknown profile-split source: {source}')


def _load_profile_split_loaders(config):
    """Build Train/Development loaders from one fixed source-local manifest."""
    path = config.get('profile_split_path')
    if not path:
        raise ValueError('profile_split_path is required for A1-S2 profile split')
    try:
        from ..data_managers.profile_split import (
            build_profile_split, load_profile_split, save_profile_split)
    except ImportError:
        from data_managers.profile_split import (
            build_profile_split, load_profile_split, save_profile_split)
    expected_seed = int(config.get('profile_split_seed', config['seed']))
    # Build the fixed manifest once from the same QC/domain-filtered profiles
    # that will feed the loaders.  Subsequent epochs only read this manifest.
    if not os.path.isfile(path):
        source_payloads = {}
        for source in ('FY', 'COSMIC'):
            full_dataset = _new_all_profile_dataset(config, source)
            source_payloads[source] = {
                # ``data`` remains the raw memmap; profile_ids correspond to
                # selected_indices after NaN/domain filtering.
                'data': np.asarray(
                    full_dataset.data[full_dataset.selected_indices]),
                'profile_ids': np.asarray(full_dataset.profile_ids),
            }
        split = build_profile_split(
            source_payloads,
            seed=expected_seed,
            block_hours=float(config.get(
                'profile_split_time_block_hours', 0.5)),
        )
        save_profile_split(path, split)
    split = load_profile_split(path, expected_seed=expected_seed)
    sources = split.get('sources', {})
    required = ('FY', 'COSMIC')
    missing_sources = [source for source in required if source not in sources]
    if missing_sources:
        raise ValueError(
            f'profile split lacks required source allowlists: {missing_sources}')

    datasets = {}
    for source in required:
        train_ids = np.asarray(sources[source]['train'], dtype=np.int64)
        development_ids = np.asarray(
            sources[source]['development'], dtype=np.int64)
        if np.intersect1d(train_ids, development_ids).size:
            raise ValueError(f'{source} train/development profile leakage')
        train_dataset = _new_all_profile_dataset(config, source)
        development_dataset = _new_all_profile_dataset(config, source)
        available = np.unique(np.asarray(train_dataset.profile_ids, dtype=np.int64))
        union = np.union1d(train_ids, development_ids)
        if not np.array_equal(union, available):
            missing = np.setdiff1d(available, union, assume_unique=True)
            extra = np.setdiff1d(union, available, assume_unique=True)
            raise ValueError(
                f'{source} profile split does not cover the active domain '
                f'(missing={missing[:5].tolist()}, extra={extra[:5].tolist()})')
        _reindex_profile_dataset(train_dataset, train_ids, source, 'train')
        _reindex_profile_dataset(
            development_dataset, development_ids, source, 'development')
        datasets[source] = (train_dataset, development_dataset)

    points = int(config.get('profile_points_per_epoch', 8))
    if points < 1:
        raise ValueError('profile_points_per_epoch must be positive')
    batch_size = int(config['batch_size'])
    num_workers = int(config.get('num_workers', 0))
    sampling_semantics = config.get(
        'background_sampling_semantics', 'legacy_profile_time_bin_v1')
    schedule_seed = int(config.get(
        'background_profile_schedule_seed', config.get('seed', 42)))
    fy_schedule_seed = int(config.get(
        'background_profile_schedule_seed_fy', schedule_seed))
    cosmic_schedule_seed = int(config.get(
        'background_profile_schedule_seed_cosmic', schedule_seed + 1))
    max_batches = config.get('background_fixed_updates_per_epoch')
    if max_batches is not None:
        max_batches = int(max_batches)
    loaders = {}
    for source, (train_dataset, development_dataset) in datasets.items():
        loaders[source] = (
            DataLoader(
                train_dataset,
                batch_sampler=ProfileTimeBinSampler(
                train_dataset, batch_size=batch_size,
                points_per_profile=points, shuffle=True,
                sampling_semantics=sampling_semantics,
                schedule_seed=(fy_schedule_seed if source == 'FY'
                               else cosmic_schedule_seed),
                max_batches=max_batches),
                num_workers=num_workers, pin_memory=False),
            DataLoader(
                development_dataset,
                batch_sampler=ProfileTimeBinSampler(
                    development_dataset, batch_size=batch_size,
                    points_per_profile=points, shuffle=False),
                num_workers=num_workers, pin_memory=False),
        )

    resolved = {
        'semantics': str(split['semantics']),
        'seed': int(split['seed']),
        'path': os.path.abspath(path),
        'sources': {
            source: {
                'train_profiles': int(len(sources[source]['train'])),
                'development_profiles': int(
                    len(sources[source]['development'])),
            }
            for source in required
        },
    }
    config['resolved_profile_split'] = resolved
    return loaders['FY'][0], loaders['FY'][1], loaders['COSMIC'][0], loaders['COSMIC'][1]


def _record_resolved_training_config(config, covariance_strata):
    path = os.path.join(config['save_dir'], 'run_manifest.json')
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as stream:
        manifest = json.load(stream)
    manifest['resolved_training'] = {
        'analysis_exact_mode_loss': bool(config.get('analysis_exact_mode_loss', False)),
        'use_covariance_moment_loss': bool(config.get('use_covariance_moment_loss', False)),
        'use_empirical_covariance_loss': bool(config.get('use_empirical_covariance_loss', False)),
        'covariance_gradient_target': float(config.get('covariance_gradient_target', 0.20)),
        'resolved_covariance_weight': config.get('resolved_covariance_weight'),
        'use_direction_loss': bool(config.get('use_direction_loss', False)),
        'direction_gradient_target': float(config.get('direction_gradient_target', 0.20)),
        'resolved_direction_weight': config.get('resolved_direction_weight'),
        'use_observation_gram_loss': bool(config.get('use_observation_gram_loss', False)),
        'gram_gradient_target': float(config.get('gram_gradient_target', 0.02)),
        'gram_calibration_batches': int(config.get('gram_calibration_batches', 20)),
        'resolved_gram_weight': config.get('resolved_gram_weight'),
        'gram_gradient_calibration': config.get('gram_gradient_calibration'),
        'background_stratum_bias_gradient_target': float(config.get('background_stratum_bias_gradient_target', 0.0)),
        'background_stratum_bias_calibration_batches': int(config.get('background_stratum_bias_calibration_batches', 20)),
        'resolved_background_stratum_bias_weight': config.get('resolved_background_stratum_bias_weight'),
        'background_stratum_bias_gradient_calibration': config.get('background_stratum_bias_gradient_calibration'),
        'background_magnetic_daynight_gradient_target': float(config.get('background_magnetic_daynight_gradient_target', 0.0125)),
        'background_magnetic_daynight_calibration_batches': int(config.get('background_magnetic_daynight_calibration_batches', 20)),
        'resolved_background_magnetic_daynight_weight': config.get('resolved_background_magnetic_daynight_weight'),
        'background_magnetic_daynight_gradient_calibration': config.get('background_magnetic_daynight_gradient_calibration'),
        'background_bias_grouping': config.get(
            'background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY),
        'background_bias_magnetic_min_profiles': int(config.get(
            'background_bias_magnetic_min_profiles', 30)),
        'background_bias_magnetic_min_batch_profiles': int(config.get(
            'background_bias_magnetic_min_batch_profiles', 8)),
        'background_bias_magnetic_weight': float(config.get(
            'background_bias_magnetic_weight', 0.5)),
        'background_bias_magnetic_support_counts': config.get(
            'background_bias_magnetic_support_counts'),
        'background_bias_magnetic_support_semantics': config.get(
            'background_bias_magnetic_support_semantics'),
        'background_shape_loss': config.get('background_shape_loss', 'disabled'),
        'background_shape_gradient_target': float(config.get('background_shape_gradient_target', 0.02)),
        'background_shape_calibration_batches': int(config.get('background_shape_calibration_batches', 20)),
        'background_shape_min_profiles_per_cell': int(config.get('background_shape_min_profiles_per_cell', 8)),
        'resolved_background_shape_weight': config.get('resolved_background_shape_weight'),
        'background_shape_gradient_calibration': config.get('background_shape_gradient_calibration'),
        'background_checkpoint_selection': config.get('background_checkpoint_selection'),
        'background_selection_baseline': config.get('background_selection_baseline'),
        'background_selection_baseline_checkpoint': config.get('background_selection_baseline_checkpoint'),
        'representativeness_kernel': config.get('resolved_representativeness_kernel'),
        'covariance_strata': covariance_strata,
        'date_split': config.get('resolved_date_split'),
        'profile_split': config.get('resolved_profile_split'),
        'input_data_sha256': config.get('input_data_sha256'),
        'model_domain_data_summary': config.get('model_domain_data_summary'),
        'low_altitude_prior_protocol': config.get('low_altitude_prior_protocol'),
        'training_protocol': config.get('training_protocol'),
        'low_altitude_anchor_history': config.get('low_altitude_anchor_history'),
        'background_training_semantics': config.get('background_training_semantics'),
        'checkpoint_semantic': config.get('checkpoint_semantic'),
        'background_decoder_conditioning': config.get(
            'background_decoder_conditioning', 'legacy_v1'),
        'background_condition_adapter': config.get('background_condition_adapter'),
        'background_adapter_gate_semantics': config.get('background_adapter_gate_semantics'),
        'background_sampling_semantics': config.get('background_sampling_semantics'),
        'background_fixed_updates_per_epoch': config.get('background_fixed_updates_per_epoch'),
        'a1s7_m11_deferred_contract': config.get('a1s7_m11_deferred_contract'),
        'a1s8_m11_deferred_contract': config.get('a1s8_m11_deferred_contract'),
        'a1s8_exposure_audit': config.get('a1s8_exposure_audit'),
        'a1s8_contract': {
            'background_anchor_semantics': config.get('background_anchor_semantics'),
            'background_anchor_gradient_target': float(config.get('background_anchor_gradient_target', 0.10)),
            'background_anchor_calibration_batches': int(config.get('background_anchor_calibration_batches', 20)),
            'background_anchor_warmup_epochs': int(config.get('background_anchor_warmup_epochs', 1)),
            'background_gate_balance_semantics': config.get('background_gate_balance_semantics'),
            'background_source_mix_schedule': config.get('background_source_mix_schedule'),
            'background_mix_legacy_fraction': float(config.get('background_mix_legacy_fraction', 1.0)),
            'background_mix_balanced_fraction': float(config.get('background_mix_balanced_fraction', 0.0)),
            'background_exposure_weight_cap': float(config.get('background_exposure_weight_cap', 3.0)),
            'background_gradient_ratio_min': float(config.get('background_gradient_ratio_min', 0.25)),
            'background_gradient_ratio_max': float(config.get('background_gradient_ratio_max', 4.0)),
            'background_low_effective_mass_min': float(config.get('background_low_effective_mass_min', 0.25)),
            'background_low_effective_mass_max': float(config.get('background_low_effective_mass_max', 0.35)),
            'background_high_effective_mass_min': float(config.get('background_high_effective_mass_min', 0.65)),
            'background_high_effective_mass_max': float(config.get('background_high_effective_mass_max', 0.75)),
        },
        'background_loader_schedule': config.get('background_loader_schedule'),
        'background_update_steps': config.get('background_update_steps', []),
        'background_component_audit': config.get('background_component_audit'),
        'analysis_training_scope': config.get('analysis_training_scope'),
        'analysis_output_semantics': config.get('analysis_output_semantics'),
        'analysis_observation_time_support': config.get('analysis_observation_time_support'),
        'r_source': config.get('r_source'),
        'm00_observation_calibration_semantics': config.get(
            'm00_observation_calibration_semantics', 'disabled'),
        'm00_observation_seed_checkpoint': config.get(
            'm00_observation_seed_checkpoint'),
        'm00_observation_training_scope': config.get(
            'm00_observation_training_scope', 'disabled'),
        'm00_observation_background_semantics': config.get(
            'background_training_semantics'),
        'm00_observation_stage': config.get('m00_observation_stage', 'bias_r'),
        'm00_observation_strict_magnetic': bool(config.get(
            'm00_observation_strict_magnetic', False)),
        'm00_observation_magnetic_sidecar_fy': config.get(
            'm00_observation_magnetic_sidecar_fy'),
        'm00_observation_magnetic_sidecar_cosmic': config.get(
            'm00_observation_magnetic_sidecar_cosmic'),
        'm00_observation_qd_sidecar_fy': config.get(
            'm00_observation_qd_sidecar_fy'),
        'm00_observation_qd_sidecar_cosmic': config.get(
            'm00_observation_qd_sidecar_cosmic'),
        'm00_observation_aacgm_version': config.get(
            'm00_observation_aacgm_version', 'aacgmv2'),
        'm00_observation_aacgm_reference_time_utc': config.get(
            'm00_observation_aacgm_reference_time_utc'),
        'm00_observation_aacgm_actual_height_semantics': config.get(
            'm00_observation_aacgm_actual_height_semantics',
            'row_altitude_km_v1'),
        'm00_localization_semantics': config.get(
            'm00_localization_semantics', 'm00_3d_localization_v2'),
        'm00_localization_time_semantics': config.get(
            'm00_localization_time_semantics', 'relative_utc_hours_v1'),
        'm00_localization_vertical_scale_km': float(config.get(
            'm00_localization_vertical_scale_km', 100.0)),
        'm00_calibration_applied': bool(config.get('m00_calibration_applied', False)),
        'm00_calibration_frozen': bool(config.get('m00_calibration_frozen', False)),
        'm00_calibration_artifact': config.get('m00_calibration_artifact'),
        'm00_representativeness_enabled': bool(config.get(
            'm00_representativeness_enabled', False)),
        'm00_localization_enabled': bool(config.get(
            'm00_localization_enabled', False)),
    }
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(temp_path, path)

def _write_a1s7_deferred_contract(config):
    """Persist the M11 contract without enabling any Analysis training."""
    if not _a1s7_enabled(config):
        return None
    path = os.path.join(config['save_dir'], 'a1s7_m11_deferred_contract.json')
    contract = {
        'status': 'deferred',
        'analysis_background_frozen': True,
        'analysis_state_semantics': 'shared_etkf_v1',
        'analysis_observation_bias_semantics': 'source_stratum_bias_v1',
        'analysis_adapter_semantics': 'physical_conditioned_lowrank_v1',
        'analysis_source_heads': 'forbidden',
        'analysis_target_profile_loo': 'preserve_a1s',
        'analysis_development_neighbor_policy': 'preserve_a1s',
    }
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(temp_path, path)
    config['a1s7_m11_deferred_contract'] = path
    return path


def _write_a1s8_deferred_contract(config):
    """Persist the deferred M11 contract for the A1-S8 M00-only run."""
    if not _a1s8_enabled(config):
        return None
    path = os.path.join(config['save_dir'], 'a1s8_m11_deferred_contract.json')
    contract = {
        'status': 'deferred',
        'semantic': _A1S8_BACKGROUND_SEMANTICS,
        'analysis_background_frozen': True,
        'analysis_state_semantics': 'shared_etkf_v1',
        'analysis_observation_bias_semantics': 'source_stratum_bias_v1',
        'analysis_adapter_semantics': 'physical_conditioned_lowrank_v1',
        'analysis_source_heads': 'forbidden',
        'analysis_target_profile_loo': 'preserve_a1s',
        'analysis_development_neighbor_policy': 'preserve_a1s',
    }
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(temp_path, path)
    config['a1s8_m11_deferred_contract'] = path
    return path


def _write_m00_transition_contract(config, status='active'):
    """Record the A1-S9-to-observation-error hand-off without hashes."""
    if (config.get('m00_observation_calibration_semantics', 'disabled')
            == 'disabled' and status == 'active'):
        return None
    path = os.path.join(config['save_dir'],
                        'm00_observation_transition_contract.json')
    contract = {
        'semantic': 'a1s9_failed_to_m00_observation_error_v1',
        'status': str(status),
        'm00_reference': 'A1S_epoch3',
        'adapter_route': 'stopped',
        'locked_test': 'excluded',
        'calibration_train_only': True,
        'stage_a': 'bias_r',
        'stage_b': 'representativeness_localization',
        'analysis_enabled': False,
        'sha256': None,
    }
    temporary = f'{path}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(temporary, path)
    config['m00_transition_contract'] = path
    return path


def _write_frozen_v2_contracts(config):
    """Write auditable v2 contracts without changing the frozen M00 model."""
    save_dir = config.get('save_dir')
    if not save_dir:
        return {}
    os.makedirs(save_dir, exist_ok=True)
    base = {
        'low_dim_assimilation_semantics': 'shared_latent_etkf_v1',
        'basis_dim': 64,
        'enkf_n_members': 8,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'analysis_state_semantics': 'legacy_feature_increment',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'mode_mapping': 'M10:FY,M01:COSMIC,M11:FY+COSMIC',
        'm00_status': 'frozen_a1s_epoch3',
        'adapter_route': 'stopped_permanently',
        'shared_backbone_unfreeze': False,
        'locked_test': 'excluded',
        'localization_enabled': False,
        'magnetic_stratum_semantics': config.get(
            'magnetic_stratum_semantics',
            'aacgm_abs_sin_lat_025_050_v2'),
        'm00_observation_strict_magnetic': bool(config.get(
            'm00_observation_strict_magnetic', False)),
        'm00_observation_magnetic_sidecar_fy': config.get(
            'm00_observation_magnetic_sidecar_fy'),
        'm00_observation_magnetic_sidecar_cosmic': config.get(
            'm00_observation_magnetic_sidecar_cosmic'),
    }
    localization = {
        'semantics': config.get('m00_localization_semantics',
                                'm00_3d_localization_v2'),
        'spatial_scale_km': float(config.get('m00_localization_space_scale_km',
                                             config.get('physical_localization_space_km', 1800.0))),
        'vertical_scale_km': float(config.get('m00_localization_vertical_scale_km', 100.0)),
        'temporal_scale_hours': float(config.get('m00_localization_time_scale_hours',
                                                  config.get('physical_localization_time_hours', 1.5))),
        'time_semantics': config.get(
            'm00_localization_time_semantics', 'relative_utc_hours_v1'),
        'enabled': False,
        'applies_to': 'observation_precision_gain_only',
    }
    deferred = {
        'status': 'deferred',
        'analysis_background_frozen': True,
        'analysis_state_semantics': 'shared_etkf_v1',
        'analysis_observation_bias_semantics': 'source_stratum_bias_v1',
        'analysis_adapter_semantics': 'physical_conditioned_lowrank_v1',
        'analysis_source_heads': 'forbidden',
        'analysis_target_profile_loo': 'preserve_a1s',
        'analysis_development_neighbor_policy': 'preserve_a1s',
    }
    calibration = config.get('m00_observation_calibration')
    if isinstance(calibration, Mapping):
        resolved = dict(calibration)
        resolved.setdefault('semantics', config.get(
            'm00_observation_calibration_semantics', 'disabled'))
        resolved.setdefault('train_only', True)
        resolved.setdefault('locked_test', 'excluded')
    else:
        resolved = {
            'status': 'not_loaded',
            'semantics': config.get('m00_observation_calibration_semantics', 'disabled'),
            'train_only': True,
            'locked_test': 'excluded',
        }
    payloads = {
        'a1s_m00_frozen_contract_v2.json': base,
        'm00_observation_calibration_resolved_v2.json': resolved,
        'm00_localization_contract_v2.json': localization,
        'm11_deferred_contract_v2.json': deferred,
    }
    paths = {}
    for name, payload in payloads.items():
        path = os.path.join(save_dir, name)
        tmp = f'{path}.tmp'
        with open(tmp, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(tmp, path)
        paths[name] = path
    config['m00_v2_contracts'] = paths
    return paths


def _write_a1s8_exposure_audit(config, history):
    """Atomically persist per-epoch A1-S8 exposure/gradient diagnostics."""
    if not _a1s8_enabled(config):
        return None
    path = os.path.join(config['save_dir'], 'a1s8_exposure_audit.json')

    def _json_safe(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return value

    payload = {
        'semantic': _A1S8_SAMPLING_SEMANTICS,
        'history': [row for row in history if row.get('a1s8_exposure')],
    }
    payload = _json_safe(payload)
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write('\n')
    os.replace(temp_path, path)
    config['a1s8_exposure_audit'] = path
    return path


def _query_observations(index, coords, profile_ids=None,
                        allowed_profile_ids=None, time_support=None):
    if index is None:
        return None
    exclude = None if profile_ids is None else profile_ids.detach().cpu().numpy()
    return query_observation_payload(
        index, coords, coords.device, exclude_profile_ids=exclude,
        allowed_profile_ids=allowed_profile_ids, time_support=time_support)


def _unpack_source_batch(batch, device):
    data, _, profile_ids = batch[:3]
    data = data.to(device, non_blocking=True)
    return data[:, :4], data[:, 4:5], profile_ids.to(device, non_blocking=True)


def _batch_magnetic_values(batch, config, source):
    """Return batch-aligned (dipole sin-I, QD/AACGM latitude) values."""
    data = batch[0]
    sample_index = batch[1] if len(batch) > 1 else None
    aacgm = None
    sin_i = None
    if len(batch) > 3 and batch[3] is not None:
        metadata = batch[3]
        if getattr(metadata, 'ndim', 0) != 2 or metadata.shape[1] < 2:
            raise ValueError(f'{source} magnetic batch metadata must be [B,2]')
        aacgm = metadata[:, 0]
        sin_i = metadata[:, 1]
    if aacgm is None and getattr(data, 'ndim', 0) == 2 and data.shape[1] >= 6:
        aacgm = data[:, 5]
    if sin_i is None and getattr(data, 'ndim', 0) == 2 and data.shape[1] >= 8:
        sin_i = data[:, 7]
    suffix = str(source).lower()
    latitude_name = (
        f'm00_observation_qd_latitude_{suffix}'
        if str(config.get('magnetic_stratum_semantics', '')).startswith('qd_')
        else f'm00_observation_aacgm_latitude_{suffix}')
    for name, current in (
            (latitude_name, aacgm),
            (f'm00_observation_magnetic_sin_i_{suffix}', sin_i)):
        if current is not None:
            continue
        values = config.get(name)
        if values is None:
            continue
        values = values if hasattr(values, 'shape') else np.asarray(values)
        values = values.reshape(-1)
        if values.size == len(data):
            selected = values
        elif sample_index is not None:
            indices = np.asarray(sample_index.detach().cpu() if hasattr(
                sample_index, 'detach') else sample_index, dtype=np.int64).reshape(-1)
            if indices.size != len(data) or np.any(indices < 0) or np.any(indices >= values.size):
                raise ValueError(f'{source} magnetic sidecar cannot align batch indices')
            selected = values[indices]
        else:
            raise ValueError(f'{source} magnetic sidecar cannot align batch')
        if name.endswith('aacgm_latitude_' + suffix):
            aacgm = selected
        else:
            sin_i = selected
    return sin_i, aacgm


def _low_altitude_anchor_batch(coords, profile_ids, config):
    """Build deterministic IRI-only anchors from batch coordinates, never targets."""
    levels = tuple(config.get('low_altitude_anchor_levels_km', ()))
    maximum = int(config.get('low_altitude_anchor_profiles_per_source', 0))
    if not levels or maximum <= 0 or len(coords) == 0:
        empty_coords = coords.new_empty((0, 4))
        return empty_coords, profile_ids.new_empty((0,))
    # The batch sampler is randomized; preserve its first appearance order so
    # the anchor selection is deterministic for a fixed sampler state.
    chosen, seen = [], set()
    for index, profile_id in enumerate(profile_ids.detach().reshape(-1).cpu().tolist()):
        profile_id = int(profile_id)
        if profile_id not in seen:
            seen.add(profile_id)
            chosen.append(index)
            if len(chosen) == maximum:
                break
    if not chosen:
        empty_coords = coords.new_empty((0, 4))
        return empty_coords, profile_ids.new_empty((0,))
    selected = torch.as_tensor(chosen, device=coords.device, dtype=torch.long)
    base = coords.index_select(0, selected).repeat_interleave(len(levels), dim=0).clone()
    base[:, 2] = torch.as_tensor(levels, device=coords.device,
                                 dtype=coords.dtype).repeat(len(chosen))
    groups = profile_ids.reshape(-1).index_select(0, selected).repeat_interleave(
        len(levels))
    return base, groups


def _low_altitude_prior_loss(model, batch_processor, coords, profile_ids,
                             source, stage, config, sw_manager,
                             iri_peak_manager, allowed_profile_ids):
    """Profile-balanced v14 IRI anchor, separate from satellite observation loss."""
    anchors, groups = _low_altitude_anchor_batch(coords, profile_ids, config)
    if len(anchors) == 0:
        zero = coords.sum() * 0.0
        return zero, {'profiles': 0, 'points': 0, 'active_fraction': 0.0}
    sw_seq = sw_manager.get_drivers_sequence(anchors[:, 3])
    iri_peak = (iri_peak_manager.get_iri_peak(anchors)
                if iri_peak_manager is not None else None)
    prediction, _, _, _, extras = _source_forward(
        model, batch_processor, anchors, sw_seq, iri_peak, stage, source,
        # Synthetic/no query profile IDs retain all train-only observation
        # tokens for the Analysis zero-increment anchor.
        None, False, config, iri_peak_manager, allowed_profile_ids)
    delta = float(config.get('huber_delta', 0.2))
    if stage == 'background':
        values = extras['ne_bkg']
        target = extras['ne_iri'].detach()
    else:
        values = prediction - extras['ne_bkg']
        target = torch.zeros_like(values)
    loss = profile_huber_loss(values, target, groups, delta=delta)
    active = (_analysis_active_mask(extras) if stage == 'analysis'
              else torch.ones(len(anchors), device=anchors.device, dtype=torch.bool))
    return loss, {
        'profiles': int(torch.unique(groups).numel()),
        'points': int(len(anchors)),
        'active_fraction': float(active.float().mean().detach().cpu()),
    }


def _apply_source_dropout(fy_obs, cosmic_obs, profile_ids, probabilities):
    p_m10, p_m01, p_m11 = probabilities
    if not np.isclose(p_m10 + p_m01 + p_m11, 1.0):
        raise ValueError('source_dropout probabilities must sum to one')
    _, inverse = torch.unique(
        profile_ids.flatten(), sorted=False, return_inverse=True)
    reference = fy_obs if fy_obs is not None else cosmic_obs
    draw = torch.rand(
        inverse.max().item() + 1, device=profile_ids.device,
        dtype=reference['value'].dtype)
    draw = draw[inverse]
    keep_fy = (draw < p_m10) | (draw >= p_m10 + p_m01)
    keep_cosmic = draw >= p_m10
    def apply(payload, keep):
        if payload is None:
            return None
        result = dict(payload)
        if payload['valid_mask'].ndim == 1 and 'query_index' in payload:
            result['valid_mask'] = (
                payload['valid_mask'] & keep[payload['query_index']])
        else:
            result['valid_mask'] = payload['valid_mask'] & keep.unsqueeze(-1)
        return result
    return apply(fy_obs, keep_fy), apply(cosmic_obs, keep_cosmic)


def _apply_balanced_profile_modes(
        fy_obs, cosmic_obs, profile_ids, analysis_epoch):
    unique_ids, inverse = torch.unique(
        profile_ids.flatten(), sorted=False, return_inverse=True)
    cycle = torch.remainder(
        unique_ids.to(torch.int64) + int(analysis_epoch), 4)
    point_cycle = cycle[inverse]

    def apply(payload, keep):
        if payload is None:
            return None
        result = dict(payload)
        if payload['valid_mask'].ndim == 1 and 'query_index' in payload:
            result['valid_mask'] = (
                payload['valid_mask'] & keep[payload['query_index']])
        else:
            result['valid_mask'] = payload['valid_mask'] & keep.unsqueeze(-1)
        return result

    return (
        apply(fy_obs, point_cycle != 1),
        apply(cosmic_obs, point_cycle != 0),
    )


def _balanced_profile_mode_counts(profile_ids, analysis_epoch):
    unique_ids = torch.unique(profile_ids.flatten(), sorted=False)
    cycle = torch.remainder(
        unique_ids.to(torch.int64) + int(analysis_epoch), 4)
    return {
        'M10': int((cycle == 0).sum().item()),
        'M01': int((cycle == 1).sum().item()),
        'M11': int((cycle >= 2).sum().item()),
    }


def _source_mode_for_batch(schedule, batch_index):
    if schedule == 'random_profile':
        return None
    if schedule == 'deterministic_112':
        return ('M10', 'M01', 'M11', 'M11')[batch_index % 4]
    if schedule == 'balanced_profile_112':
        return schedule
    raise ValueError(f'unknown source_mode_schedule: {schedule}')


def _apply_source_mode(fy_obs, cosmic_obs, mode):
    if mode not in ('M10', 'M01', 'M11'):
        raise ValueError(f'unknown source mode: {mode}')

    def apply(payload, keep):
        if payload is None or keep:
            return payload
        result = dict(payload)
        result['valid_mask'] = torch.zeros_like(
            payload['valid_mask'], dtype=torch.bool)
        return result

    return apply(fy_obs, mode != 'M01'), apply(cosmic_obs, mode != 'M10')


def _set_training_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == 'background':
        adapter_semantics = getattr(
            model, 'background_condition_adapter_semantics', 'disabled')
        adapter_scope = getattr(model, 'background_training_scope', None)
        candidate_scope = getattr(model, 'm00_candidate_semantics', 'disabled')
        observation_scope = getattr(model, 'm00_observation_training_scope',
                                    'disabled')
        if candidate_scope in _M00_CANDIDATE_SCOPES:
            if observation_scope not in ('disabled', _M00_OBSERVATION_SCOPE):
                raise ValueError('M00 candidate requires an observation-compatible scope')
            if candidate_scope == _M00_DECODER_TRUST_SCOPE:
                modules = [model.background_decoder[-1]]
            else:
                head = getattr(model, 'm00_peak_relative_head', None)
                if head is None:
                    raise ValueError('peak-relative candidate requires m00_peak_relative_head')
                modules = [head]
        elif observation_scope == _M00_OBSERVATION_SCOPE:
            modules = [model.iri_align_net, model.background_decoder]
        elif (adapter_semantics == _A1S7_ADAPTER_SEMANTICS
                and adapter_scope == _A1S7_ADAPTER_SCOPE):
            low = getattr(model, 'background_condition_adapter_low', None)
            high = getattr(model, 'background_condition_adapter_high', None)
            if low is None or high is None:
                raise ValueError('A1-S7 requires low/high condition adapters')
            modules = [low, high]
        elif (adapter_semantics == _COND_ADAPTER_SEMANTICS
                and adapter_scope == _COND_ADAPTER_SCOPE):
            adapter = getattr(model, 'background_condition_adapter', None)
            if adapter is None:
                raise ValueError('condition adapter requested but model has none')
            modules = [adapter]
        else:
            modules = [
                model.iri_align_net,
                model.background_decoder,
                model.sw_encoder,
            ]
            if model.use_sw_freq:
                modules.extend([model.sw_freq_branch, model.sw_gate])
            if (adapter_semantics == _COND_ADAPTER_SEMANTICS
                    and adapter_scope == _COND_ADAPTER_JOINT_FRESH_SCOPE):
                adapter = getattr(model, 'background_condition_adapter', None)
                if adapter is None:
                    raise ValueError('joint condition adapter requested but model has none')
                modules.append(adapter)
    elif stage == 'analysis':
        modules = [model.kalman_layer]
        if hasattr(model, 'density_basis_decoder'):
            modules.append(model.density_basis_decoder)
    else:
        raise ValueError(f'unknown training stage: {stage}')
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.iri_proxy.freeze()


def _current_stage_trainable_parameters(model):
    """The sole parameter scope for optimizer, calibration, and audits."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _current_stage_parameter_counts(model):
    parameters = _current_stage_trainable_parameters(model)
    return len(parameters), sum(parameter.numel() for parameter in parameters)


def _set_stage_mode(model, stage):
    model.train()
    model.iri_proxy.eval()
    if stage == 'analysis':
        model.iri_align_net.eval()
        model.background_decoder.eval()
        model.sw_encoder.eval()
        if model.use_sw_freq:
            model.sw_freq_branch.eval()
            model.sw_gate.eval()
    elif (stage == 'background'
          and getattr(model, 'm00_candidate_semantics', 'disabled')
          in _M00_CANDIDATE_SCOPES):
        model.iri_align_net.eval()
        model.background_decoder.eval()
        if getattr(model, 'm00_candidate_semantics', 'disabled') == _M00_DECODER_TRUST_SCOPE:
            model.background_decoder[-1].train()
        else:
            model.m00_peak_relative_head.train()
        model.sw_encoder.eval()
        if model.use_sw_freq:
            model.sw_freq_branch.eval()
            model.sw_gate.eval()
    elif (stage == 'background'
          and getattr(model, 'm00_observation_training_scope', 'disabled')
          == _M00_OBSERVATION_SCOPE):
        model.iri_align_net.train()
        model.background_decoder.train()
        model.sw_encoder.eval()
        if model.use_sw_freq:
            model.sw_freq_branch.eval()
            model.sw_gate.eval()
        if getattr(model, 'background_condition_adapter', None) is not None:
            model.background_condition_adapter.eval()
        if getattr(model, 'background_condition_adapter_low', None) is not None:
            model.background_condition_adapter_low.eval()
        if getattr(model, 'background_condition_adapter_high', None) is not None:
            model.background_condition_adapter_high.eval()
    elif (stage == 'background'
          and getattr(model, 'background_condition_adapter_semantics',
                      'disabled') == _A1S7_ADAPTER_SEMANTICS
          and getattr(model, 'background_training_scope', None)
          == _A1S7_ADAPTER_SCOPE):
        model.iri_align_net.eval()
        model.background_decoder.eval()
        model.sw_encoder.eval()
        if model.use_sw_freq:
            model.sw_freq_branch.eval()
            model.sw_gate.eval()
        model.background_condition_adapter_low.train()
        model.background_condition_adapter_high.train()
    elif (stage == 'background'
          and getattr(model, 'background_condition_adapter_semantics',
                      'disabled') == _COND_ADAPTER_SEMANTICS
          and getattr(model, 'background_training_scope', None)
          == _COND_ADAPTER_SCOPE):
        model.iri_align_net.eval()
        model.background_decoder.eval()
        model.sw_encoder.eval()
        if model.use_sw_freq:
            model.sw_freq_branch.eval()
            model.sw_gate.eval()
        model.background_condition_adapter.train()


def _background_component_audit(model):
    """Return the explicit Background parameter scope for fresh-run records."""
    names = (
        'iri_align_net', 'background_decoder', 'sw_encoder', 'sw_freq_branch',
        'sw_gate', 'background_condition_adapter',
        'background_condition_adapter_low',
        'background_condition_adapter_high', 'm00_peak_relative_head')
    components = {}
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue
        parameters = list(module.parameters())
        components[name] = {
            'parameter_tensors': len(parameters),
            'parameter_count': sum(parameter.numel() for parameter in parameters),
            'trainable_tensors': sum(
                parameter.requires_grad for parameter in parameters),
            'trainable_parameter_count': sum(
                parameter.numel() for parameter in parameters
                if parameter.requires_grad),
        }
    return components


def _make_optimizer(model, config, stage_epochs):
    optimizer = optim.AdamW(
        _current_stage_trainable_parameters(model),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = None
    if config.get('scheduler_type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, stage_epochs),
            eta_min=config.get('min_lr', 1e-6),
        )
    return optimizer, scheduler


def _source_forward(model, batch_processor, coords, sw_seq, iri_peak,
                    stage, target_source, profile_ids, apply_dropout, config,
                    iri_peak_manager=None, allowed_profile_ids=None,
                    source_mode=None, analysis_epoch=0):
    allowed_profile_ids = allowed_profile_ids or {}
    causal_time_support = (
        config.get('analysis_observation_time_support')
        if config.get('analysis_training_scope')
        == _CAUSAL_HISTORY_ANALYSIS_SCOPE else None)
    if stage == 'background':
        fy_obs = cosmic_obs = None
    elif target_source == 'FY':
        fy_obs = _query_observations(
            batch_processor.fy_nb_index, coords, profile_ids,
            allowed_profile_ids.get('FY'), causal_time_support)
        cosmic_obs = _query_observations(
            batch_processor.cosmic_nb_index, coords,
            allowed_profile_ids=allowed_profile_ids.get('COSMIC'),
            time_support=causal_time_support)
    else:
        fy_obs = _query_observations(
            batch_processor.fy_nb_index, coords,
            allowed_profile_ids=allowed_profile_ids.get('FY'),
            time_support=causal_time_support)
        cosmic_obs = _query_observations(
            batch_processor.cosmic_nb_index, coords, profile_ids,
            allowed_profile_ids.get('COSMIC'), causal_time_support)

    if stage == 'analysis' and batch_processor.representativeness_kernel is not None:
        fy_obs = attach_representativeness_weight(
            fy_obs, coords, target_source, 'FY',
            batch_processor.representativeness_kernel,
            batch_processor.representativeness_floor)
        cosmic_obs = attach_representativeness_weight(
            cosmic_obs, coords, target_source, 'COSMIC',
            batch_processor.representativeness_kernel,
            batch_processor.representativeness_floor)

    if stage == 'analysis' and apply_dropout:
        if source_mode is None:
            fy_obs, cosmic_obs = _apply_source_dropout(
                fy_obs, cosmic_obs, profile_ids,
                config['source_dropout'])
        elif source_mode == 'balanced_profile_112':
            fy_obs, cosmic_obs = _apply_balanced_profile_modes(
                fy_obs, cosmic_obs, profile_ids, analysis_epoch)
        else:
            fy_obs, cosmic_obs = _apply_source_mode(
                fy_obs, cosmic_obs, source_mode)
    if stage == 'analysis':
        fy_obs = attach_observation_background(
            fy_obs, model, batch_processor.sw_manager, iri_peak_manager)
        cosmic_obs = attach_observation_background(
            cosmic_obs, model, batch_processor.sw_manager, iri_peak_manager)
    return model(
        coords,
        sw_seq,
        iri_peak=iri_peak,
        observations_fy=fy_obs,
        observations_cosmic=cosmic_obs,
    )


def _structure_losses(model, batch_processor, sw_manager, iri_peak_manager,
                      coords, profile_ids, stage, config,
                      allowed_profile_ids=None):
    n = min(config.get('structure_batch_size', 32), len(coords))
    if n == 0:
        zero = coords.new_zeros(())
        return zero, zero
    chosen = torch.randperm(len(coords), device=coords.device)[:n]
    centers = coords[chosen, :4].clone()
    selected_ids = profile_ids[chosen]
    residual_key = 'background_residual' if stage == 'background' else 'ne_residual'

    def evaluate(triplet):
        flat = triplet.reshape(-1, 4)
        repeated_ids = selected_ids.repeat_interleave(3)
        sw_seq = sw_manager.get_drivers_sequence(flat[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(flat)
                    if iri_peak_manager is not None else None)
        _, _, _, _, extras = _source_forward(
            model, batch_processor, flat, sw_seq, iri_peak,
            stage, 'FY', repeated_ids, False, config, iri_peak_manager,
            allowed_profile_ids)
        return extras[residual_key].reshape(n, 3)

    alt_step = config.get('structure_alt_step_km', 20.0)
    centers[:, 2].clamp_(
        model.alt_min + alt_step, model.alt_max - alt_step)
    vertical = centers[:, None, :].repeat(1, 3, 1)
    vertical[:, 0, 2] = centers[:, 2] - alt_step
    vertical[:, 2, 2] = centers[:, 2] + alt_step

    time_step = config.get('structure_time_step_hours', 1.0)
    centers[:, 3].clamp_(time_step, 720.0 - time_step)
    temporal = centers[:, None, :].repeat(1, 3, 1)
    temporal[:, 0, 3] = centers[:, 3] - time_step
    temporal[:, 2, 3] = centers[:, 3] + time_step

    beta = config.get('structure_huber_beta', 0.05)
    return (
        second_difference_loss(evaluate(vertical), beta),
        second_difference_loss(evaluate(temporal), beta),
    )


def _gradient_norms(data_loss, auxiliary_loss, parameters, components=None):
    def loss_norm(loss):
        if not loss.requires_grad:
            return data_loss.new_zeros(())
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True)
        finite = [grad.square().sum() for grad in grads if grad is not None]
        return torch.sqrt(torch.stack(finite).sum()) if finite else data_loss.new_zeros(())

    data_norm = loss_norm(data_loss)
    auxiliary_norm = loss_norm(auxiliary_loss)
    ratio = auxiliary_norm / data_norm.clamp_min(1e-12)
    component_norms = {
        name: loss_norm(loss).detach()
        for name, loss in (components or {}).items()
    }
    return (
        data_norm.detach(), auxiliary_norm.detach(), ratio.detach(),
        component_norms,
    )


def _module_gradient_norm(module):
    """Return a finite L2 norm for a module's currently accumulated grads."""
    if module is None:
        return 0.0
    values = [parameter.grad.detach().float().norm()
              for parameter in module.parameters()
              if parameter.grad is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).square().sum()).cpu())


def _smoke_json_value(value):
    if isinstance(value, dict):
        return {key: _smoke_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_smoke_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_smoke_preflight(config, stage, epoch, batch, ratios, coverages,
                           losses=None):
    """Atomically record each stage gate before the first optimizer step."""
    path = os.path.join(config['save_dir'], 'smoke_preflight.json')
    current = {'schema_version': 14, 'stages': {}}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as stream:
            current = json.load(stream)
    lower, upper = config.get('gram_preflight_ratio_bounds', (0.016, 0.024))
    auxiliary_limit = float(config['smoke_auxiliary_gradient_ratio_max'])
    total = ratios['total_auxiliary_to_observation']
    low = ratios['low_altitude_to_observation']
    gram = ratios['gram_to_observation']
    failures = []
    inclusive = stage == 'background'
    if not np.isfinite(total) or total < 0.0 or total > auxiliary_limit:
        failures.append('total auxiliary gradient ratio outside [0, 0.25]')
    if not np.isfinite(low) or low < 0.0 or low > auxiliary_limit:
        failures.append('low-altitude gradient ratio outside [0, 0.25]')
    if not inclusive and (total <= 0.0 or low <= 0.0):
        failures.append('Analysis low-altitude and total auxiliary ratios must be positive')
    calibration = config.get('gram_gradient_calibration')
    if stage == 'analysis':
        achieved = None if calibration is None else calibration.get(
            'achieved_median_ratio')
        if not np.isfinite(gram) or gram <= 0.0:
            failures.append('single-batch Gram gradient ratio must be finite and positive')
        if (achieved is None or not np.isfinite(achieved)
                or not float(lower) <= float(achieved) <= float(upper)):
            failures.append(
                '20-batch Gram achieved median ratio is outside '
                f'[{float(lower):.3f}, {float(upper):.3f}]')
        if (calibration is None
                or calibration.get('requested_batches') != 20
                or calibration.get('processed_batches') != 20
                or calibration.get('valid_batches') != 20):
            failures.append('v14 Gram calibration is not a complete 20/20 record')
    stage_record = {
        'epoch': int(epoch + 1),
        'batch': int(batch + 1),
        'losses': _smoke_json_value(losses or {}),
        'ratios': _smoke_json_value(ratios),
        'gram_calibration': _smoke_json_value(
            calibration if stage == 'analysis' else None),
        'gram_coverage_diagnostic': _smoke_json_value(coverages),
        'coverage_policy': 'diagnostic_only_v1',
        'passed': not failures,
        'failures': failures,
    }
    current.setdefault('stages', {})[stage] = stage_record
    temporary = f'{path}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(current, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
    os.replace(temporary, path)
    if failures:
        raise RuntimeError('; '.join(failures))


def _query_precision_sum(extras, source):
    precision = extras[f'precision_{source}']
    if precision.ndim == 1:
        query_index = extras[f'observation_query_index_{source}'].long()
        result = extras['ne_bkg'].new_zeros(extras['ne_bkg'].shape[0])
        if query_index.numel():
            result.index_add_(0, query_index, precision)
        return result
    return precision.sum(dim=-1)


def _analysis_active_mask(extras):
    """Queries with at least one physical observation contributing precision."""
    return (_query_precision_sum(extras, 'FY')
            + _query_precision_sum(extras, 'COSMIC')) > 0


_EXACT_MODE_SOURCES = {
    'M10': ('FY',),
    'M01': ('COSMIC',),
    'M11': ('FY', 'COSMIC'),
}
_EXACT_MODE_WEIGHTS = {'M10': 0.25, 'M01': 0.25, 'M11': 0.50}


def observation_gram_whitening_loss(extras, profile_ids, kalman_layer,
                                    return_details=False):
    """Whiten the production-precision weighted observation factor Gram.

    The loss is computed in the seven independent factor coordinates and is
    therefore exactly tied to the ``HX = F @ C`` geometry used by the ETKF.
    Queries with fewer than seven positive-precision tokens are unsupported;
    rank-deficient Gram matrices with sufficient support remain eligible.
    """
    if kalman_layer.anomaly_parameterization != 'orthogonal_factor':
        raise ValueError('observation Gram loss requires orthogonal_factor')
    rank = kalman_layer.n_members - 1
    profile_ids = profile_ids.reshape(-1)
    expected = extras['ne_bkg'].shape[0]
    if profile_ids.numel() != expected:
        raise ValueError('profile_ids must contain one ID per query')
    identity = torch.eye(
        rank, device=extras['ne_bkg'].device, dtype=torch.float32)
    losses, coverage = {}, {}
    total = extras['ne_bkg'].sum() * 0.0

    for mode, sources in _EXACT_MODE_SOURCES.items():
        grams = []
        token_counts = torch.zeros(
            expected, device=extras['ne_bkg'].device, dtype=torch.long)
        for source in sources:
            precomputed = extras.get(f'factor_gram_{source}')
            if precomputed is not None:
                grams.append(precomputed.float())
                precision = extras[f'precision_{source}']
                if precision.ndim == 1:
                    query_index = extras[f'observation_query_index_{source}']
                    token_counts.index_add_(
                        0, query_index.long(), (precision > 0.0).long())
                else:
                    token_counts = token_counts + (precision > 0.0).sum(dim=-1)
                continue
            factors = extras.get(f'observation_factors_{source}')
            if factors is None:
                basis = extras.get(f'basis_{source}')
                if basis is None:
                    raise ValueError(
                        f'observation basis is missing for source {source}')
                factors = kalman_layer.observation_factor_coordinates(
                    basis, extras.get('factor_scales'))
            precision = extras[f'precision_{source}']
            factors = factors.float()
            precision = precision.float()
            grams.append(torch.einsum(
                'bmr,bm,bmk->brk', factors, precision, factors))
            token_counts = token_counts + (precision > 0.0).sum(dim=-1)
        gram = sum(grams)
        trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1)
        eligible = (
            (token_counts >= rank)
            & torch.isfinite(trace)
            & (trace > 1e-12)
            & torch.isfinite(gram).all(dim=-1).all(dim=-1)
        )
        normalized = gram / trace.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1)
        query_loss = (normalized - identity / rank).square().sum(dim=(-2, -1))
        query_loss = torch.where(
            eligible, query_loss, torch.zeros_like(query_loss))
        losses[mode] = _profile_weighted_mean(
            query_loss, eligible, profile_ids,
            torch.ones_like(query_loss))
        coverage[mode] = eligible.float().mean()
        total = total + _EXACT_MODE_WEIGHTS[mode] * losses[mode]

    if return_details:
        return total, losses, coverage
    return total


def exact_mode_profile_losses(extras, target, profile_ids, delta):
    """Profile-balanced M10/M01/M11 losses from one joint forward."""
    losses = {}
    active_masks = {}
    increments = solve_density_modes(extras)
    for mode, sources in _EXACT_MODE_SOURCES.items():
        prediction = extras['ne_bkg'] + increments[mode].unsqueeze(-1)
        active = sum(
            _query_precision_sum(extras, source)
            for source in sources) > 0
        losses[mode] = profile_huber_loss(
            prediction, target, profile_ids, delta=delta, valid_mask=active)
        active_masks[mode] = active
    total = sum(
        _EXACT_MODE_WEIGHTS[mode] * loss
        for mode, loss in losses.items())
    return total, losses, active_masks


def _covariance_stratum_ids(coords):
    local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    high = coords[:, 2] >= 300.0
    return high.to(torch.long) * 2 + night.to(torch.long)


def _profile_weighted_mean(values, valid, profile_ids, weights):
    results = []
    for profile_id in torch.unique(profile_ids):
        mask = valid & (profile_ids == profile_id)
        if mask.any():
            selected_weights = weights[mask]
            results.append(
                (values[mask] * selected_weights).sum()
                / selected_weights.sum().clamp_min(1e-12))
    if not results:
        return values.sum() * 0.0
    return torch.stack(results).mean()


def _background_bias_grouping_name(grouping):
    grouping = str(grouping or _BACKGROUND_BIAS_GROUPING_LEGACY).replace(
        '-', '_')
    if grouping not in _BACKGROUND_BIAS_GROUPINGS:
        raise ValueError(f'unknown Background Bias grouping: {grouping}')
    return grouping


def _background_bias_magnetic_masks(coords):
    """Return the fixed coarse |sin(I)| proxy masks used by M00."""
    sin_i, _ = _compute_dip_features(coords[:, 0], coords[:, 1])
    abs_sin_i = sin_i.abs()
    return (
        ('abs_sinI_lt_0.25', abs_sin_i < 0.25),
        ('abs_sinI_0.25_0.5',
         (abs_sin_i >= 0.25) & (abs_sin_i < 0.5)),
        ('abs_sinI_ge_0.5', abs_sin_i >= 0.5),
    )


def _background_bias_magnetic_daynight_masks(coords):
    """Return six coarse magnetic-proxy/local-time masks."""
    local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    magnetic = dict(_background_bias_magnetic_masks(coords))
    return tuple(
        (f'{label}_{period}', mask & (night if period == 'night' else ~night))
        for label, mask in magnetic.items()
        for period in ('night', 'day'))


def background_stratified_bias_loss(
        prediction, target, coords, profile_ids,
        alt_range=(200.0, 500.0), return_details=False,
        grouping=_BACKGROUND_BIAS_GROUPING_LEGACY,
        magnetic_support_counts=None, magnetic_min_profiles=30,
        magnetic_min_batch_profiles=8, magnetic_weight=0.5):
    """Profile-balanced source-local mean-Bias penalty.

    ``height_daynight_v1`` is the historical four-cell loss.  The M00
    ``height_daynight_plus_magnetic_v1`` is the historical coarse magnetic
    blend.  ``height_daynight_plus_magnetic_daynight_v1`` keeps the complete
    height/daynight term and adds a separate six-cell magnetic/daynight term.
    The v2 name exposes those two tensors independently to the trainer.
    """
    grouping = _background_bias_grouping_name(grouping)
    magnetic_min_profiles = int(magnetic_min_profiles)
    magnetic_min_batch_profiles = int(magnetic_min_batch_profiles)
    magnetic_weight = float(magnetic_weight)
    if (magnetic_min_profiles < 1 or magnetic_min_batch_profiles < 1
            or not 0.0 <= magnetic_weight <= 1.0):
        raise ValueError('invalid magnetic Background Bias grouping thresholds')
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    coords = coords.reshape(-1, coords.shape[-1])
    profile_ids = profile_ids.reshape(-1)
    if not (len(prediction) == len(target) == len(coords) == len(profile_ids)):
        raise ValueError('Background Bias inputs must have equal lengths')
    lower, upper = map(float, alt_range)
    if not lower < upper:
        raise ValueError('Background Bias altitude range must be increasing')
    altitude = coords[:, 2]
    local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    valid = (
        torch.isfinite(prediction) & torch.isfinite(target)
        & torch.isfinite(coords).all(dim=1)
        & torch.isfinite(profile_ids.to(dtype=coords.dtype))
        & (altitude >= lower) & (altitude <= upper))
    errors = prediction - target
    # The public labels intentionally remain stable for manifest/diagnostic use.
    cells = (
        ('200-300_day', (altitude >= max(lower, 200.0))
         & (altitude < min(upper, 300.0)) & ~night),
        ('200-300_night', (altitude >= max(lower, 200.0))
         & (altitude < min(upper, 300.0)) & night),
        ('300-500_day', (altitude >= max(lower, 300.0))
         & (altitude <= min(upper, 500.0)) & ~night),
        ('300-500_night', (altitude >= max(lower, 300.0))
         & (altitude <= min(upper, 500.0)) & night),
    )
    losses = []
    details = {}
    for label, cell_mask in cells:
        selected = valid & cell_mask
        if not selected.any():
            details[label] = {
                'profile_count': 0, 'point_count': 0,
                'mean_bias': None, 'squared_bias': None,
            }
            continue
        profile_means = []
        for profile_id in torch.unique(profile_ids[selected], sorted=False):
            profile_mask = selected & (profile_ids == profile_id)
            if profile_mask.any():
                profile_means.append(errors[profile_mask].mean())
        if not profile_means:
            details[label] = {
                'profile_count': 0, 'point_count': 0,
                'mean_bias': None, 'squared_bias': None,
            }
            continue
        mean_bias = torch.stack(profile_means).mean()
        squared_bias = mean_bias.square()
        losses.append(squared_bias)
        details[label] = {
            'profile_count': len(profile_means),
            'point_count': int(selected.sum().detach().cpu()),
            'mean_bias': float(mean_bias.detach().cpu()),
            'squared_bias': float(squared_bias.detach().cpu()),
        }
    zero = prediction[torch.isfinite(prediction)].sum() * 0.0
    height_loss = torch.stack(losses).mean() if losses else zero
    magnetic_losses = []
    magnetic_details = {}
    support = magnetic_support_counts or {}
    if grouping in (_BACKGROUND_BIAS_GROUPING_MAGNETIC,
                    _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT,
                    _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2):
        magnetic_masks = (_background_bias_magnetic_masks(coords)
                          if grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC
                          else _background_bias_magnetic_daynight_masks(coords))
        magnetic_altitude = ((altitude >= max(lower, 200.0))
                             & (altitude <= min(upper, 500.0)))
        for label, magnetic_mask in magnetic_masks:
            selected = valid & magnetic_altitude & magnetic_mask
            profile_means = []
            if selected.any():
                for profile_id in torch.unique(
                        profile_ids[selected], sorted=False):
                    profile_mask = selected & (profile_ids == profile_id)
                    if profile_mask.any():
                        profile_means.append(errors[profile_mask].mean())
            profile_count = len(profile_means)
            support_count = int(support.get(label, 0))
            active = (support_count >= magnetic_min_profiles
                      and profile_count >= magnetic_min_batch_profiles)
            entry = {
                'profile_count': profile_count,
                'point_count': int(selected.sum().detach().cpu()),
                'support_profile_count': support_count,
                'active': bool(active),
                'mean_bias': None,
                'squared_bias': None,
            }
            if profile_means:
                mean_bias = torch.stack(profile_means).mean()
                entry['mean_bias'] = float(mean_bias.detach().cpu())
                entry['squared_bias'] = float(
                    mean_bias.square().detach().cpu())
                if active:
                    magnetic_losses.append(mean_bias.square())
            magnetic_details[label] = entry
    magnetic_loss = torch.stack(magnetic_losses).mean() if magnetic_losses else zero
    if grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC and magnetic_losses:
        result = ((1.0 - magnetic_weight) * height_loss
                  + magnetic_weight * magnetic_loss)
    elif (grouping in (_BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT,
                       _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2)
          and magnetic_losses):
        # Keep the legacy height/daynight objective at full strength.  The
        # calibrated outer weight controls the combined auxiliary objective;
        # magnetic_weight is the fixed conditional-term multiplier.
        result = (height_loss + magnetic_weight * magnetic_loss
                  if grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT
                  else height_loss)
    else:
        result = height_loss
    if return_details:
        details['active_cell_count'] = len(losses)
        details['grouping'] = grouping
        details['height_daynight_loss'] = float(height_loss.detach().cpu())
        details['magnetic_active_cell_count'] = len(magnetic_losses)
        details['magnetic_loss'] = float(magnetic_loss.detach().cpu())
        details['magnetic_weight'] = magnetic_weight
        details['combined_loss'] = float(result.detach().cpu())
        details['_height_loss_tensor'] = height_loss
        details['_magnetic_loss_tensor'] = magnetic_loss
        details['magnetic'] = magnetic_details
        details['magnetic_support_semantics'] = (
            _BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_V2_SEMANTICS
            if grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2 else
            (_BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_SEMANTICS
             if grouping in (_BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT,
                             _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2)
             else _BACKGROUND_BIAS_MAGNETIC_SEMANTICS
             if grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC else None))
        details['valid_profile_count'] = int(
            torch.unique(profile_ids[valid], sorted=False).numel())
        return result, details
    return result


def background_profile_stratum_loss(prediction, target, coords, profile_ids,
                                    delta=0.2, alt_range=(200.0, 500.0),
                                    return_details=False):
    """Average profile Huber losses equally over supported 12 physical cells."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    coords = coords.reshape(-1, coords.shape[-1])
    profile_ids = profile_ids.reshape(-1)
    if not (len(prediction) == len(target) == len(coords) == len(profile_ids)):
        raise ValueError('Background profile-stratum inputs must align')
    lower, upper = map(float, alt_range)
    valid = (torch.isfinite(prediction) & torch.isfinite(target)
             & torch.isfinite(coords).all(dim=1))
    valid &= (coords[:, 2] >= lower) & (coords[:, 2] <= upper)
    if not valid.any():
        result = prediction.sum() * 0.0
        return (result, {'active_cells': 0}) if return_details else result
    errors = F.huber_loss(prediction[valid], target[valid], reduction='none',
                          delta=delta)
    ids, inverse = torch.unique(profile_ids[valid], sorted=False,
                                return_inverse=True)
    n_profiles = int(ids.numel())
    sums = torch.zeros(n_profiles, device=errors.device, dtype=errors.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, errors)
    counts.scatter_add_(0, inverse, torch.ones_like(errors))
    profile_loss = sums / counts.clamp_min(1.0)
    profile_coords = torch.zeros((n_profiles, coords.shape[1]),
                                 device=coords.device, dtype=coords.dtype)
    profile_coords.scatter_add_(
        0, inverse[:, None].expand(-1, coords.shape[1]), coords[valid])
    profile_coords /= counts.clamp_min(1.0)[:, None]
    alt_band = (profile_coords[:, 2] >= 300.0).to(torch.long)
    local_hour = torch.remainder(profile_coords[:, 3] + profile_coords[:, 1] / 15.0,
                                 24.0)
    day_band = ((local_hour < 6.0) | (local_hour >= 18.0)).to(torch.long)
    mag = torch.sin(torch.deg2rad(profile_coords[:, 0])).abs()
    mag_band = torch.where(mag < 0.25, 0, torch.where(mag < 0.5, 1, 2))
    cell_ids = alt_band * 6 + day_band * 3 + mag_band
    losses = []
    active = []
    for cell in range(12):
        selected = cell_ids == cell
        if selected.any():
            losses.append(profile_loss[selected].mean())
            active.append(cell)
    if not losses:
        result = prediction.sum() * 0.0
        details = {'active_cells': 0, 'cell_ids': []}
    else:
        result = torch.stack(losses).mean()
        details = {'active_cells': len(active), 'cell_ids': active}
    return (result, details) if return_details else result


def _background_physical_strata(coords):
    """Return six altitude/day/magnetic-latitude stratum IDs."""
    altitude = coords[:, 2]
    local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    day = ((local_time >= 6.0) & (local_time < 18.0)).to(torch.long)
    altitude_band = (altitude >= 300.0).to(torch.long)
    magnetic = torch.sin(torch.deg2rad(coords[:, 0])).abs()
    magnetic_band = torch.where(
        magnetic < 0.25, torch.zeros_like(magnetic, dtype=torch.long),
        torch.where(magnetic < 0.5,
                    torch.ones_like(magnetic, dtype=torch.long),
                    torch.full_like(magnetic, 2, dtype=torch.long)))
    return altitude_band * 6 + day * 3 + magnetic_band


def _background_magnetic_support_counts(loader, *, daynight=False):
    """Count train profiles in the fixed coarse magnetic proxy bins."""
    dataset = getattr(loader, 'dataset', None)
    if dataset is None or not hasattr(dataset, 'selected_indices'):
        raise RuntimeError('Background magnetic support requires a profile dataset')
    indices = np.asarray(dataset.selected_indices, dtype=np.int64)
    profile_ids = np.asarray(dataset.profile_ids, dtype=np.int64).reshape(-1)
    if len(indices) != len(profile_ids):
        raise RuntimeError('Background magnetic support dataset is misaligned')
    rows = np.asarray(dataset.data[indices, :4], dtype=np.float64)
    finite = (np.isfinite(rows).all(axis=1)
              & (rows[:, 2] >= 200.0) & (rows[:, 2] <= 500.0))
    if not finite.any():
        return {label: 0 for label in _BACKGROUND_BIAS_MAGNETIC_LABELS}
    lat = torch.as_tensor(rows[finite, 0], dtype=torch.float64)
    lon = torch.as_tensor(rows[finite, 1], dtype=torch.float64)
    sin_i, _ = _compute_dip_features(lat, lon)
    abs_sin_i = np.abs(sin_i.detach().cpu().numpy())
    ids = profile_ids[finite]
    local_time = np.remainder(rows[finite, 3] + rows[finite, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    base_masks = (
        abs_sin_i < 0.25,
        (abs_sin_i >= 0.25) & (abs_sin_i < 0.5),
        abs_sin_i >= 0.5,
    )
    if daynight:
        labels = _BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_LABELS
        masks = tuple(
            mask & (night if period == 'night' else ~night)
            for mask in base_masks for period in ('night', 'day'))
    else:
        labels, masks = _BACKGROUND_BIAS_MAGNETIC_LABELS, base_masks
    return {
        label: int(np.unique(ids[mask]).size)
        for label, mask in zip(labels, masks)
    }


def _resolve_background_bias_support_counts(
        train_loader, cosmic_train_loader, config):
    """Freeze train-only magnetic support before calibration or training."""
    grouping = _background_bias_grouping_name(
        config.get('background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY))
    if (grouping == _BACKGROUND_BIAS_GROUPING_LEGACY
            and not config.get('m00_diagnostic_train_split', False)):
        config['background_bias_magnetic_support_counts'] = None
        config['background_bias_magnetic_support_semantics'] = None
        return None
    if (grouping in (_BACKGROUND_BIAS_GROUPING_MAGNETIC,
                     _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT,
                     _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2)
            and not _natural_background_enabled(config)):
        raise ValueError(
            'magnetic Background Bias grouping is only supported by natural '
            'Background training')
    daynight = grouping in (_BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT,
                            _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2)
    support = {
        'FY': _background_magnetic_support_counts(
            train_loader, daynight=daynight),
        'COSMIC': _background_magnetic_support_counts(
            cosmic_train_loader, daynight=daynight),
    }
    config['background_bias_magnetic_support_counts'] = support
    config['background_bias_magnetic_support_semantics'] = (
        _BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_V2_SEMANTICS
        if grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2 else
        (_BACKGROUND_BIAS_MAGNETIC_DAYNIGHT_SEMANTICS
         if grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT
         else _BACKGROUND_BIAS_MAGNETIC_SEMANTICS)
        if grouping in (_BACKGROUND_BIAS_GROUPING_MAGNETIC,
                        _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT,
                        _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2)
        else 'diagnostic_only_train_support_v1')
    return support


def _background_bias_kwargs(config, source):
    """Resolve the frozen grouping contract for one source call."""
    grouping = _background_bias_grouping_name(
        config.get('background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY))
    support = config.get('background_bias_magnetic_support_counts') or {}
    return {
        'grouping': grouping,
        'magnetic_support_counts': support.get(source, {}),
        'magnetic_min_profiles': int(config.get(
            'background_bias_magnetic_min_profiles', 30)),
        'magnetic_min_batch_profiles': int(config.get(
            'background_bias_magnetic_min_batch_profiles', 8)),
        'magnetic_weight': float(config.get(
            'background_bias_magnetic_weight', 0.5)),
    }


def background_gate_aware_loss(
        prediction, target, coords, profile_ids, gate, delta=0.2,
        alt_range=(200.0, 500.0), return_details=False):
    """Profile/stratum loss with effective low/high gate-mass normalization.

    The two branches are averaged independently, so a source with mostly high
    magnetic-latitude samples cannot drown the low-latitude adapter.  The
    normalization denominators are detached to keep the loss scale stable.
    """
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    coords = coords.reshape(-1, coords.shape[-1])
    profile_ids = profile_ids.reshape(-1)
    gate = gate.reshape(-1).to(dtype=prediction.dtype)
    if not (len(prediction) == len(target) == len(coords)
            == len(profile_ids) == len(gate)):
        raise ValueError('gate-aware Background inputs must align')
    lower, upper = map(float, alt_range)
    if not lower < upper:
        raise ValueError('gate-aware Background altitude range increasing')
    finite = (torch.isfinite(prediction) & torch.isfinite(target)
              & torch.isfinite(coords).all(dim=1)
              & torch.isfinite(gate)
              & (coords[:, 2] >= lower) & (coords[:, 2] <= upper))
    if not finite.any():
        zero = prediction.sum() * 0.0
        details = {
            'active_strata': 0, 'low_effective_mass': 0.0,
            'high_effective_mass': 0.0, 'low_profiles': 0,
            'high_profiles': 0, 'low_effective_mass_raw': 0.0,
            'high_effective_mass_raw': 0.0,
        }
        return (zero, details) if return_details else zero
    prediction = prediction[finite]
    target = target[finite]
    coords = coords[finite]
    profile_ids = profile_ids[finite]
    gate = gate[finite].clamp(0.0, 1.0)
    error = F.huber_loss(prediction, target, reduction='none', delta=delta)
    weights = (1.0 - gate, gate)
    profile_unique, inverse = torch.unique(
        profile_ids, sorted=False, return_inverse=True)
    n_profiles = int(profile_unique.numel())
    strata = _background_physical_strata(coords)
    profile_strata = torch.zeros(
        n_profiles, device=coords.device, dtype=torch.long)
    profile_strata.scatter_(0, inverse, strata)
    branch_losses = []
    branch_masses = []
    branch_profiles = []
    active_strata = set()
    for branch_weight in weights:
        weighted_sum = torch.zeros(
            n_profiles, device=error.device, dtype=error.dtype)
        mass = torch.zeros_like(weighted_sum)
        weighted_sum.scatter_add_(0, inverse, error * branch_weight)
        mass.scatter_add_(0, inverse, branch_weight)
        supported = mass > 1e-8
        profile_loss = weighted_sum / mass.detach().clamp_min(1e-8)
        stratum_losses = []
        for stratum in range(12):
            selected = supported & (profile_strata == stratum)
            if selected.any():
                stratum_losses.append(profile_loss[selected].mean())
                active_strata.add(stratum)
        if stratum_losses:
            branch_losses.append(torch.stack(stratum_losses).mean())
        branch_masses.append(float(branch_weight.detach().sum().cpu()))
        branch_profiles.append(int(supported.sum().detach().cpu()))
    result = (torch.stack(branch_losses).mean() if branch_losses
              else prediction.sum() * 0.0)
    total_mass = sum(branch_masses)
    if total_mass > 0.0:
        normalized_masses = [mass / total_mass for mass in branch_masses]
    else:
        normalized_masses = [0.0, 0.0]
    details = {
        'active_strata': len(active_strata),
        'low_effective_mass': normalized_masses[0],
        'high_effective_mass': normalized_masses[1],
        'low_effective_mass_raw': branch_masses[0],
        'high_effective_mass_raw': branch_masses[1],
        'low_profiles': branch_profiles[0],
        'high_profiles': branch_profiles[1],
    }
    return (result, details) if return_details else result


def background_adapter_anchor_loss(
        effective_delta, coords, profile_ids, delta=0.2,
        alt_range=(200.0, 500.0), return_details=False):
    """A1-S output trust-region loss for the adapter's effective M00 delta."""
    values = effective_delta.reshape(-1)
    coords = coords.reshape(-1, coords.shape[-1])
    profile_ids = profile_ids.reshape(-1)
    if not (len(values) == len(coords) == len(profile_ids)):
        raise ValueError('anchor inputs must align')
    lower, upper = map(float, alt_range)
    valid = (torch.isfinite(values) & torch.isfinite(coords).all(dim=1)
             & (coords[:, 2] >= lower) & (coords[:, 2] <= upper))
    if not valid.any():
        zero = values.sum() * 0.0
        details = {'profiles': 0, 'active_strata': 0}
        return (zero, details) if return_details else zero
    values = values[valid]
    coords = coords[valid]
    profile_ids = profile_ids[valid]
    errors = F.huber_loss(values, torch.zeros_like(values), reduction='none',
                          delta=delta)
    unique, inverse = torch.unique(
        profile_ids, sorted=False, return_inverse=True)
    sums = torch.zeros(
        unique.numel(), device=values.device, dtype=values.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, errors)
    counts.scatter_add_(0, inverse, torch.ones_like(errors))
    profile_loss = sums / counts.detach().clamp_min(1.0)
    strata = _background_physical_strata(coords)
    profile_strata = torch.zeros_like(unique, dtype=torch.long)
    profile_strata.scatter_(0, inverse, strata)
    losses = [profile_loss[profile_strata == cell].mean()
              for cell in range(12) if (profile_strata == cell).any()]
    result = torch.stack(losses).mean() if losses else values.sum() * 0.0
    details = {'profiles': int(unique.numel()),
               'active_strata': len(losses)}
    return (result, details) if return_details else result


def a1s8_source_mix(update_index, config=None):
    """Return deterministic 3:1 legacy/balanced schedule for one update."""
    config = {} if config is None else config
    semantics = str(config.get('background_source_mix_schedule',
                               _A1S8_SOURCE_MIX_SEMANTICS)).replace('-', '_')
    if semantics in ('equal_v1', 'disabled'):
        return 0.5, 0.5, 'equal'
    if semantics != _A1S8_SOURCE_MIX_SEMANTICS:
        raise ValueError(f'unsupported A1-S8 source-mix semantics: {semantics}')
    legacy = float(config.get('background_mix_legacy_fraction', 0.75))
    balanced = float(config.get('background_mix_balanced_fraction', 0.25))
    if not (np.isfinite(legacy) and np.isfinite(balanced)
            and legacy >= 0.0 and balanced >= 0.0
            and abs(legacy + balanced - 1.0) <= 1e-6):
        raise ValueError('A1-S8 source-mix fractions must sum to one')
    return legacy, balanced, ('balanced' if int(update_index) % 4 == 3
                              else 'legacy')


def background_conditional_shape_loss(
        prediction, target, coords, profile_ids, delta=0.2,
        min_profiles_per_cell=8, target_std_min=1e-4,
        return_details=False):
    """Profile-centered anomaly and spread loss in the fixed 12 M00 cells."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    coords = coords.reshape(-1, coords.shape[-1])
    profile_ids = profile_ids.reshape(-1)
    if not (len(prediction) == len(target) == len(coords) == len(profile_ids)):
        raise ValueError('Background shape inputs must have equal lengths')
    if min_profiles_per_cell < 1 or target_std_min <= 0.0:
        raise ValueError('Background shape profile/std thresholds must be positive')

    altitude = coords[:, 2]
    local_time = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    day = (local_time >= 6.0) & (local_time < 18.0)
    sin_i, _ = _compute_dip_features(coords[:, 0], coords[:, 1])
    abs_sin_i = sin_i.abs()
    finite = (
        torch.isfinite(prediction) & torch.isfinite(target)
        & torch.isfinite(coords).all(dim=1)
        & torch.isfinite(profile_ids.to(dtype=coords.dtype))
        & (altitude >= 200.0) & (altitude <= 500.0))
    altitude_cells = (
        ('200-300', (altitude >= 200.0) & (altitude < 300.0)),
        ('300-500', (altitude >= 300.0) & (altitude <= 500.0)),
    )
    period_cells = (('night', ~day), ('day', day))
    magnetic_cells = (
        ('abs_sinI_lt_0.25', abs_sin_i < 0.25),
        ('abs_sinI_0.25_0.5',
         (abs_sin_i >= 0.25) & (abs_sin_i < 0.5)),
        ('abs_sinI_ge_0.5', abs_sin_i >= 0.5),
    )
    losses = []
    details = {}
    for altitude_label, altitude_mask in altitude_cells:
        for period_label, period_mask in period_cells:
            for magnetic_label, magnetic_mask in magnetic_cells:
                label = f'{altitude_label}_{period_label}_{magnetic_label}'
                selected = finite & altitude_mask & period_mask & magnetic_mask
                profile_predictions, profile_targets = [], []
                for profile_id in torch.unique(profile_ids[selected], sorted=False):
                    profile_mask = selected & (profile_ids == profile_id)
                    profile_predictions.append(prediction[profile_mask].mean())
                    profile_targets.append(target[profile_mask].mean())
                profile_count = len(profile_predictions)
                if profile_count:
                    pred_values = torch.stack(profile_predictions)
                    target_values = torch.stack(profile_targets)
                    target_std = target_values.std(unbiased=False)
                else:
                    pred_values = prediction.new_empty((0,))
                    target_values = target.new_empty((0,))
                    target_std = target.new_zeros(())
                active = (
                    profile_count >= int(min_profiles_per_cell)
                    and bool(torch.isfinite(target_std))
                    and float(target_std.detach().cpu()) > float(target_std_min))
                if active:
                    pred_centered = pred_values - pred_values.mean()
                    target_centered = target_values - target_values.mean()
                    anomaly = F.smooth_l1_loss(
                        pred_centered, target_centered, beta=float(delta))
                    pred_std = pred_values.std(unbiased=False)
                    log_ratio = torch.log(
                        (pred_std + target_std_min)
                        / (target_std + target_std_min))
                    scale = F.smooth_l1_loss(
                        log_ratio, torch.zeros_like(log_ratio), beta=1.0)
                    cell_loss = 0.5 * (anomaly + scale)
                    losses.append(cell_loss)
                    details[label] = {
                        'active': True,
                        'profile_count': profile_count,
                        'point_count': int(selected.sum().detach().cpu()),
                        'anomaly_loss': float(anomaly.detach().cpu()),
                        'scale_loss': float(scale.detach().cpu()),
                        'prediction_std': float(pred_std.detach().cpu()),
                        'target_std': float(target_std.detach().cpu()),
                    }
                else:
                    details[label] = {
                        'active': False,
                        'profile_count': profile_count,
                        'point_count': int(selected.sum().detach().cpu()),
                        'anomaly_loss': None,
                        'scale_loss': None,
                        'prediction_std': None,
                        'target_std': (
                            float(target_std.detach().cpu()) if profile_count else None),
                    }
    result = torch.stack(losses).mean() if losses else prediction.sum() * 0.0
    if return_details:
        details['active_cell_count'] = len(losses)
        return result, details
    return result


def exact_mode_direction_losses(extras, target, profile_ids, target_source):
    """Penalize only M10/M01/M11 increments opposite to target residuals."""
    if target_source not in ('FY', 'COSMIC'):
        raise ValueError(f'unknown target source: {target_source}')
    desired = target.squeeze(-1) - extras['ne_bkg'].squeeze(-1)
    desired_sign = torch.sign(desired).detach()
    target_r = (
        extras['r_fy'] if target_source == 'FY' else extras['r_cosmic'])
    increments = solve_density_modes(extras)
    losses = {}
    active_masks = {}
    for mode, sources in _EXACT_MODE_SOURCES.items():
        active = (
            sum(_query_precision_sum(extras, source)
                for source in sources) > 0
        ) & (desired.abs() >= 0.05)
        token_loss = F.relu(
            -desired_sign * increments[mode] / torch.sqrt(target_r))
        losses[mode] = _profile_weighted_mean(
            token_loss,
            active,
            profile_ids,
            torch.ones_like(token_loss),
        )
        active_masks[mode] = active
    total = sum(
        _EXACT_MODE_WEIGHTS[mode] * loss
        for mode, loss in losses.items())
    return total, losses, active_masks


def covariance_moment_loss(extras, target, profile_ids, target_source,
                           stratum_weights):
    """Profile-balanced physical residual covariance moment matching."""
    target_r = (
        extras['r_fy'] if target_source == 'FY' else extras['r_cosmic'])
    target_residual = target.squeeze(-1) - extras['ne_bkg'].squeeze(-1)
    target_standardized = torch.clamp(
        target_residual / torch.sqrt(target_r), -3.0, 3.0)
    query_sum = target_residual.new_zeros(target_residual.shape)
    query_sources = target_residual.new_zeros(target_residual.shape)

    for suffix, source_r in (
            ('FY', extras['r_fy']), ('COSMIC', extras['r_cosmic'])):
        innovation = extras[f'innov_{suffix}']
        cross_covariance = extras[f'cross_covariance_{suffix}']
        precision = extras[f'precision_{suffix}']
        # precision = valid * localization / R; recover the localization
        # weight so R standardizes the moment without weighting it twice.
        localization = precision * source_r
        target_moment = (
            target_standardized.unsqueeze(-1)
            * torch.clamp(
                innovation / torch.sqrt(source_r), -3.0, 3.0)
        ).detach()
        predicted_moment = (
            cross_covariance / torch.sqrt(target_r * source_r))
        token_loss = F.smooth_l1_loss(
            predicted_moment, target_moment, beta=1.0, reduction='none')
        weight_sum = localization.sum(dim=-1)
        active = weight_sum > 0
        query_loss = (
            (token_loss * localization).sum(dim=-1)
            / weight_sum.clamp_min(1e-12))
        query_sum = query_sum + torch.where(
            active, query_loss, torch.zeros_like(query_loss))
        query_sources = query_sources + active.to(query_sources.dtype)

    valid_query = query_sources > 0
    query_loss = query_sum / query_sources.clamp_min(1.0)
    cells = _covariance_stratum_ids(extras['query_coords'])
    cell_weights = torch.as_tensor(
        stratum_weights[target_source],
        device=query_loss.device, dtype=query_loss.dtype)
    return _profile_weighted_mean(
        query_loss, valid_query, profile_ids, cell_weights[cells])


def empirical_covariance_loss(
        extras, profile_ids, target_source, empirical_targets):
    """Match model covariance to frozen train-only profile-blocked cells."""
    if empirical_targets is None:
        raise ValueError('empirical covariance targets are required')
    target_r = (
        extras['r_fy'] if target_source == 'FY' else extras['r_cosmic'])
    query_sum = extras['ne_bkg'].new_zeros(extras['ne_bkg'].shape[0])
    query_sources = torch.zeros_like(query_sum)
    for suffix, source_r in (
            ('FY', extras['r_fy']), ('COSMIC', extras['r_cosmic'])):
        precision = extras[f'precision_{suffix}']
        covariance_target, stable = empirical_covariance_token_targets(
            extras['query_coords'], extras[f'observation_coords_{suffix}'],
            extras[f'observation_rho_squared_{suffix}'], precision > 0.0,
            target_source, suffix, empirical_targets)
        scale = torch.sqrt(target_r * source_r)
        token_loss = F.smooth_l1_loss(
            extras[f'cross_covariance_{suffix}'] / scale,
            (covariance_target / scale).detach(),
            beta=1.0, reduction='none')
        weights = precision * source_r * stable.to(precision.dtype)
        weight_sum = weights.sum(dim=-1)
        active = weight_sum > 0.0
        query_loss = (
            (token_loss * weights).sum(dim=-1)
            / weight_sum.clamp_min(1e-12))
        query_sum = query_sum + torch.where(
            active, query_loss, torch.zeros_like(query_loss))
        query_sources = query_sources + active.to(query_sources.dtype)
    valid_query = query_sources > 0.0
    return _profile_weighted_mean(
        query_sum / query_sources.clamp_min(1.0), valid_query, profile_ids,
        torch.ones_like(query_sum))


def _covariance_training_loss(
        extras, target, profile_ids, target_source, batch_processor, config):
    if not (config.get('use_covariance_moment_loss', False)
            or config.get('use_empirical_covariance_loss', False)):
        return extras['ne_bkg'].sum() * 0.0
    if config.get('use_empirical_covariance_loss', False):
        return empirical_covariance_loss(
            extras, profile_ids, target_source,
            batch_processor.empirical_covariance_targets)
    return covariance_moment_loss(
        extras, target, profile_ids, target_source,
        config['covariance_stratum_weights'])


def _paired_analysis_losses(
        model, batch_processor, fy_batch, cosmic_batch, device, config,
        sw_manager, iri_peak_manager, allowed_profile_ids=None,
        source_mode=None, analysis_epoch=0):
    coords, target, profile_ids = _unpack_source_batch(fy_batch, device)
    sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
    iri_peak = (iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
    cosmic_coords, cosmic_target, cosmic_ids = _unpack_source_batch(
        cosmic_batch, device)
    cosmic_sw = sw_manager.get_drivers_sequence(cosmic_coords[:, 3])
    cosmic_peak = (iri_peak_manager.get_iri_peak(cosmic_coords)
                   if iri_peak_manager is not None else None)
    exact_modes = config.get('analysis_exact_mode_loss', False)
    fy_pred, _, _, _, fy_extras = _source_forward(
        model, batch_processor, coords, sw_seq, iri_peak,
        'analysis', 'FY', profile_ids, not exact_modes, config, iri_peak_manager,
        allowed_profile_ids, source_mode, analysis_epoch)
    cosmic_pred, _, _, _, cosmic_extras = _source_forward(
        model, batch_processor, cosmic_coords, cosmic_sw, cosmic_peak,
        'analysis', 'COSMIC', cosmic_ids, not exact_modes, config, iri_peak_manager,
        allowed_profile_ids, source_mode, analysis_epoch)
    delta = config.get('huber_delta', 0.2)
    if exact_modes:
        fy_loss = exact_mode_profile_losses(
            fy_extras, target, profile_ids, delta)[0]
        cosmic_loss = exact_mode_profile_losses(
            cosmic_extras, cosmic_target, cosmic_ids, delta)[0]
    else:
        active_only = config.get('analysis_loss_active_only', False)
        fy_loss = profile_huber_loss(
            fy_pred, target, profile_ids, delta=delta,
            valid_mask=_analysis_active_mask(fy_extras) if active_only else None)
        cosmic_loss = profile_huber_loss(
            cosmic_pred, cosmic_target, cosmic_ids, delta=delta,
            valid_mask=_analysis_active_mask(cosmic_extras) if active_only else None)
    observation_loss = 0.5 * (fy_loss + cosmic_loss)
    covariance_loss = 0.5 * (
        _covariance_training_loss(
            fy_extras, target, profile_ids, 'FY', batch_processor, config)
        + _covariance_training_loss(
            cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC',
            batch_processor, config))
    direction_loss = 0.5 * (
        exact_mode_direction_losses(
            fy_extras, target, profile_ids, 'FY')[0]
        + exact_mode_direction_losses(
            cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC')[0])
    gram_loss = observation_loss.new_zeros(())
    if config.get('use_observation_gram_loss', False):
        gram_loss = 0.5 * (
            observation_gram_whitening_loss(
                fy_extras, profile_ids, model.kalman_layer)
            + observation_gram_whitening_loss(
                cosmic_extras, cosmic_ids, model.kalman_layer))
    return observation_loss, covariance_loss, direction_loss, gram_loss


def _resolve_covariance_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    batches = int(config.get('covariance_calibration_batches', 20))
    target_ratio = float(config.get('covariance_gradient_target', 0.20))
    if batches < 1 or not 0.0 < target_ratio <= 1.0:
        raise ValueError(
            'covariance calibration batches must be positive and target in (0, 1]')
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    ratios = []
    try:
        _set_stage_mode(model, 'analysis')
        cosmic_iter = iter(cosmic_train_loader)
        for batch_index, fy_batch in enumerate(train_loader):
            if batch_index >= batches:
                break
            try:
                cosmic_batch = next(cosmic_iter)
            except StopIteration:
                cosmic_iter = iter(cosmic_train_loader)
                cosmic_batch = next(cosmic_iter)
            observation, covariance, _, _ = _paired_analysis_losses(
                model, batch_processor, fy_batch, cosmic_batch, device,
                config, sw_manager, iri_peak_manager, allowed_profile_ids,
                _source_mode_for_batch(
                    config.get('source_mode_schedule', 'random_profile'),
                    batch_index))
            _, _, covariance_to_observation, _ = _gradient_norms(
                observation, covariance, _current_stage_trainable_parameters(model))
            value = float(covariance_to_observation)
            if np.isfinite(value) and value > 0:
                ratios.append(value)
            model.zero_grad(set_to_none=True)
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if not ratios:
        raise RuntimeError('covariance moment calibration produced no finite gradients')
    minimum_weight = (
        1e-4 if config.get('use_empirical_covariance_loss', False)
        else 1e-3)
    resolved = float(np.clip(
        target_ratio / np.median(ratios), minimum_weight, 1.0))
    print(
        f'[协方差矩] 梯度比中位数={np.median(ratios):.6f}, '
        f'resolved lambda={resolved:.6f}')
    return resolved


def _resolve_direction_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    batches = int(config.get('direction_calibration_batches', 20))
    target_ratio = float(config.get('direction_gradient_target', 0.20))
    if batches < 1 or not 0.0 < target_ratio <= 1.0:
        raise ValueError(
            'direction calibration batches must be positive and target in (0, 1]')
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    ratios = []
    try:
        _set_stage_mode(model, 'analysis')
        cosmic_iter = iter(cosmic_train_loader)
        for batch_index, fy_batch in enumerate(train_loader):
            if batch_index >= batches:
                break
            try:
                cosmic_batch = next(cosmic_iter)
            except StopIteration:
                cosmic_iter = iter(cosmic_train_loader)
                cosmic_batch = next(cosmic_iter)
            observation, _, direction, _ = _paired_analysis_losses(
                model, batch_processor, fy_batch, cosmic_batch, device,
                config, sw_manager, iri_peak_manager, allowed_profile_ids,
                'exact_M10_M01_M11')
            _, _, direction_to_observation, _ = _gradient_norms(
                observation, direction, _current_stage_trainable_parameters(model))
            value = float(direction_to_observation)
            if np.isfinite(value) and value > 0:
                ratios.append(value)
            model.zero_grad(set_to_none=True)
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if not ratios:
        raise RuntimeError('direction calibration produced no finite gradients')
    resolved = float(np.clip(
        target_ratio / np.median(ratios), 1e-4, 1.0))
    print(
        f'[方向损失] 梯度比中位数={np.median(ratios):.6f}, '
        f'resolved lambda={resolved:.6f}')
    return resolved


def _resolve_background_stratum_bias_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    """Calibrate the Background mean-Bias term against its data loss."""
    batches = int(config.get('background_stratum_bias_calibration_batches', 20))
    target_ratio = float(config.get(
        'background_stratum_bias_gradient_target', 0.0))
    if batches < 1 or not 0.0 <= target_ratio <= 1.0:
        raise ValueError(
            'Background Bias calibration requires batches >= 1 and target in [0, 1]')
    if target_ratio == 0.0:
        config['w_background_stratum_bias'] = 0.0
        config['resolved_background_stratum_bias_weight'] = 0.0
        config['background_stratum_bias_gradient_calibration'] = {
            'enabled': False, 'requested_batches': batches,
            'target_ratio': 0.0,
        }
        return 0.0

    module_modes = [(module, module.training) for module in model.modules()]
    requires_grad = [
        (parameter, bool(parameter.requires_grad))
        for parameter in model.parameters()]
    state_dict = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()}
    gradients = [
        (parameter, None if parameter.grad is None
         else parameter.grad.detach().clone())
        for parameter in model.parameters()]
    transient = {
        name: _clone_calibration_value(getattr(model.kalman_layer, name))
        for name in ('last_member_weights', 'last_inflation_scale')
        if hasattr(model.kalman_layer, name)}
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    ratios = []
    cell_counts = []
    processed = 0
    try:
        _set_training_stage(model, 'background')
        _set_stage_mode(model, 'background')
        parameters = _current_stage_trainable_parameters(model)
        if not parameters:
            raise RuntimeError('Background Bias calibration has no trainable parameters')
        fy_iter = iter(train_loader)
        cosmic_iter = iter(cosmic_train_loader)
        for batch_index in range(batches):
            try:
                fy_batch = next(fy_iter)
            except StopIteration:
                fy_iter = iter(train_loader)
                fy_batch = next(fy_iter)
            try:
                cosmic_batch = next(cosmic_iter)
            except StopIteration:
                cosmic_iter = iter(cosmic_train_loader)
                cosmic_batch = next(cosmic_iter)
            coords, target, profile_ids = _unpack_source_batch(fy_batch, device)
            cosmic_coords, cosmic_target, cosmic_ids = _unpack_source_batch(
                cosmic_batch, device)
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            cosmic_sw = sw_manager.get_drivers_sequence(cosmic_coords[:, 3])
            iri_peak = (iri_peak_manager.get_iri_peak(coords)
                        if iri_peak_manager is not None else None)
            cosmic_peak = (iri_peak_manager.get_iri_peak(cosmic_coords)
                           if iri_peak_manager is not None else None)
            with torch.amp.autocast(
                    'cuda', enabled=bool(config.get('use_amp', False))
                    and device.type == 'cuda'):
                _, _, _, _, fy_extras = _source_forward(
                    model, batch_processor, coords, sw_seq, iri_peak,
                    'background', 'FY', profile_ids, False, config,
                    iri_peak_manager, allowed_profile_ids)
                _, _, _, _, cosmic_extras = _source_forward(
                    model, batch_processor, cosmic_coords, cosmic_sw,
                    cosmic_peak, 'background', 'COSMIC', cosmic_ids, False,
                    config, iri_peak_manager, allowed_profile_ids)
                fy_observation = profile_huber_loss(
                    fy_extras['ne_bkg'], target, profile_ids,
                    delta=config.get('huber_delta', 0.2))
                cosmic_observation = profile_huber_loss(
                    cosmic_extras['ne_bkg'], cosmic_target,
                    cosmic_ids, delta=config.get('huber_delta', 0.2))
                observation = 0.5 * (fy_observation + cosmic_observation)
                fy_bias, fy_details = background_stratified_bias_loss(
                    fy_extras['ne_bkg'], target, coords, profile_ids,
                    return_details=True,
                    **_background_bias_kwargs(config, 'FY'))
                cosmic_bias, cosmic_details = background_stratified_bias_loss(
                    cosmic_extras['ne_bkg'], cosmic_target,
                    cosmic_coords, cosmic_ids, return_details=True,
                    **_background_bias_kwargs(config, 'COSMIC'))
                bias = 0.5 * (fy_bias + cosmic_bias)
            observation_grad, bias_grad, ratio, _ = _gradient_norms(
                observation, bias, parameters)
            values = tuple(float(value.detach().cpu()) for value in (
                observation_grad, bias_grad, ratio))
            processed += 1
            cell_counts.append({
                'batch': batch_index + 1,
                'FY': fy_details['active_cell_count'],
                'COSMIC': cosmic_details['active_cell_count'],
            })
            if all(np.isfinite(value) and value > 0.0 for value in values):
                ratios.append(values[2])
    finally:
        model.load_state_dict(state_dict, strict=True)
        for parameter, gradient in gradients:
            parameter.grad = (
                None if gradient is None else gradient.clone())
        for parameter, enabled in requires_grad:
            parameter.requires_grad_(enabled)
        for module, training in module_modes:
            module.train(training)
        for name, value in transient.items():
            setattr(model.kalman_layer, name, _clone_calibration_value(value))
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if not ratios:
        raise RuntimeError(
            'Background Bias calibration produced no finite positive gradients')
    median_ratio = float(np.median(ratios))
    resolved = float(np.clip(target_ratio / median_ratio, 1e-4, 1.0))
    config['w_background_stratum_bias'] = resolved
    config['resolved_background_stratum_bias_weight'] = resolved
    config['background_stratum_bias_gradient_calibration'] = {
        'enabled': True,
        'requested_batches': batches,
        'processed_batches': processed,
        'valid_batches': len(ratios),
        'raw_ratios': [float(value) for value in ratios],
        'median_ratio': median_ratio,
        'target_ratio': target_ratio,
        'resolved_weight': resolved,
        'achieved_median_ratio': resolved * median_ratio,
        'active_cells': cell_counts,
    }
    print(
        f'[Background分层Bias] 梯度比中位数={median_ratio:.6f}, '
        f'resolved lambda={resolved:.6f}')
    return resolved


def _resolve_background_magnetic_daynight_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    """Calibrate only the independent v2 magnetic/daynight loss term."""
    batches = int(config.get('background_magnetic_daynight_calibration_batches', 20))
    target = float(config.get('background_magnetic_daynight_gradient_target', 0.0125))
    if batches < 1 or not 0.0 < target <= 1.0:
        raise ValueError('invalid v2 magnetic/daynight calibration settings')
    if config.get('background_bias_grouping') != _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2:
        raise ValueError('v2 magnetic/daynight calibration requires v2 grouping')
    modules = [(m, m.training) for m in model.modules()]
    params_state = [(p, bool(p.requires_grad),
                     None if p.grad is None else p.grad.detach().clone())
                    for p in model.parameters()]
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state(); nrng = np.random.get_state()
    python_rng = random.getstate()
    transient = {name: _clone_calibration_value(getattr(model.kalman_layer, name))
                 for name in ('last_member_weights', 'last_inflation_scale')
                 if hasattr(getattr(model, 'kalman_layer', None), name)}
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    ratios = []
    try:
        _set_training_stage(model, 'background'); _set_stage_mode(model, 'background')
        parameters = _current_stage_trainable_parameters(model)
        fy_iter, cosmic_iter = iter(train_loader), iter(cosmic_train_loader)
        for index in range(batches):
            try: fy_batch = next(fy_iter)
            except StopIteration: fy_iter = iter(train_loader); fy_batch = next(fy_iter)
            try: cosmic_batch = next(cosmic_iter)
            except StopIteration: cosmic_iter = iter(cosmic_train_loader); cosmic_batch = next(cosmic_iter)
            terms = []
            for batch, source in ((fy_batch, 'FY'), (cosmic_batch, 'COSMIC')):
                coords, target_y, ids = _unpack_source_batch(batch, device)
                sw = sw_manager.get_drivers_sequence(coords[:, 3])
                peak = (iri_peak_manager.get_iri_peak(coords)
                        if iri_peak_manager is not None else None)
                _, _, _, _, extras = _source_forward(
                    model, batch_processor, coords, sw, peak, 'background',
                    source, ids, False, config, iri_peak_manager,
                    allowed_profile_ids)
                obs = profile_huber_loss(extras['ne_bkg'], target_y, ids,
                                         delta=config.get('huber_delta', 0.2))
                _, detail = background_stratified_bias_loss(
                    extras['ne_bkg'], target_y, coords, ids, return_details=True,
                    **_background_bias_kwargs(config, source))
                terms.append((obs, detail['_magnetic_loss_tensor']))
            observation = 0.5 * (terms[0][0] + terms[1][0])
            magnetic = 0.5 * (terms[0][1] + terms[1][1])
            obs_grad, mag_grad, ratio, _ = _gradient_norms(
                observation, magnetic, parameters)
            value = float(ratio.detach().cpu())
            if all(np.isfinite(v) and v > 0.0 for v in (
                    float(obs_grad), float(mag_grad), value)):
                ratios.append(value)
            model.zero_grad(set_to_none=True)
    finally:
        model.load_state_dict(state, strict=True)
        for p, enabled, grad in params_state:
            p.requires_grad_(enabled); p.grad = None if grad is None else grad.clone()
        for module, training in modules: module.train(training)
        torch.set_rng_state(rng); np.random.set_state(nrng)
        random.setstate(python_rng)
        for name, value in transient.items():
            setattr(model.kalman_layer, name, _clone_calibration_value(value))
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
    if not ratios:
        raise RuntimeError('v2 magnetic/daynight calibration produced no finite gradients')
    median = float(np.median(ratios))
    resolved = target / median
    if not np.isfinite(resolved) or not 1e-4 <= resolved <= 1.0:
        raise RuntimeError('v2 magnetic/daynight calibration target is out of bounds')
    config['resolved_background_magnetic_daynight_weight'] = float(resolved)
    config['background_magnetic_daynight_gradient_calibration'] = {
        'enabled': True, 'requested_batches': batches, 'processed_batches': batches,
        'valid_batches': len(ratios), 'raw_ratios': ratios,
        'median_ratio': median, 'target_ratio': target,
        'resolved_weight': float(resolved),
        'achieved_median_ratio': float(resolved * median),
    }
    return float(resolved)


def _resolve_background_shape_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    """Calibrate the conditional-shape gradient to the fixed 2% target."""
    batches = int(config.get('background_shape_calibration_batches', 20))
    target_ratio = float(config.get('background_shape_gradient_target', 0.02))
    min_profiles = int(config.get('background_shape_min_profiles_per_cell', 8))
    if batches < 1 or not 0.0 < target_ratio <= 1.0 or min_profiles < 1:
        raise ValueError('Background shape calibration configuration is invalid')

    module_modes = [(module, module.training) for module in model.modules()]
    requires_grad = [(parameter, bool(parameter.requires_grad))
                     for parameter in model.parameters()]
    state_dict = {name: value.detach().clone()
                  for name, value in model.state_dict().items()}
    gradients = [(parameter, None if parameter.grad is None
                  else parameter.grad.detach().clone())
                 for parameter in model.parameters()]
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    ratios, active_cells = [], []
    try:
        _set_training_stage(model, 'background')
        _set_stage_mode(model, 'background')
        parameters = _current_stage_trainable_parameters(model)
        if not parameters:
            raise RuntimeError('Background shape calibration has no parameters')
        fy_iter, cosmic_iter = iter(train_loader), iter(cosmic_train_loader)
        for batch_index in range(batches):
            try:
                fy_batch = next(fy_iter)
            except StopIteration:
                fy_iter = iter(train_loader)
                fy_batch = next(fy_iter)
            try:
                cosmic_batch = next(cosmic_iter)
            except StopIteration:
                cosmic_iter = iter(cosmic_train_loader)
                cosmic_batch = next(cosmic_iter)
            coords, target, profile_ids = _unpack_source_batch(fy_batch, device)
            cosmic_coords, cosmic_target, cosmic_ids = _unpack_source_batch(
                cosmic_batch, device)
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            cosmic_sw = sw_manager.get_drivers_sequence(cosmic_coords[:, 3])
            iri_peak = (iri_peak_manager.get_iri_peak(coords)
                        if iri_peak_manager is not None else None)
            cosmic_peak = (iri_peak_manager.get_iri_peak(cosmic_coords)
                           if iri_peak_manager is not None else None)
            with torch.amp.autocast(
                    'cuda', enabled=bool(config.get('use_amp', False))
                    and device.type == 'cuda'):
                _, _, _, _, fy_extras = _source_forward(
                    model, batch_processor, coords, sw_seq, iri_peak,
                    'background', 'FY', profile_ids, False, config,
                    iri_peak_manager, allowed_profile_ids)
                _, _, _, _, cosmic_extras = _source_forward(
                    model, batch_processor, cosmic_coords, cosmic_sw, cosmic_peak,
                    'background', 'COSMIC', cosmic_ids, False, config,
                    iri_peak_manager, allowed_profile_ids)
                fy_observation = profile_huber_loss(
                    fy_extras['ne_bkg'], target, profile_ids,
                    delta=config.get('huber_delta', 0.2))
                cosmic_observation = profile_huber_loss(
                    cosmic_extras['ne_bkg'], cosmic_target, cosmic_ids,
                    delta=config.get('huber_delta', 0.2))
                observation = 0.5 * (fy_observation + cosmic_observation)
                fy_shape, fy_details = background_conditional_shape_loss(
                    fy_extras['ne_bkg'], target, coords, profile_ids,
                    delta=config.get('huber_delta', 0.2),
                    min_profiles_per_cell=min_profiles, return_details=True)
                cosmic_shape, cosmic_details = background_conditional_shape_loss(
                    cosmic_extras['ne_bkg'], cosmic_target, cosmic_coords,
                    cosmic_ids, delta=config.get('huber_delta', 0.2),
                    min_profiles_per_cell=min_profiles, return_details=True)
                shape = 0.5 * (fy_shape + cosmic_shape)
            observation_grad, shape_grad, ratio, _ = _gradient_norms(
                observation, shape, parameters)
            values = tuple(float(value.detach().cpu()) for value in (
                observation_grad, shape_grad, ratio))
            if not all(np.isfinite(value) and value > 0.0 for value in values):
                raise RuntimeError(
                    'Background shape calibration requires finite positive '
                    f'gradients at batch {batch_index + 1}: {values}')
            ratios.append(values[2])
            active_cells.append({
                'batch': batch_index + 1,
                'FY': int(fy_details['active_cell_count']),
                'COSMIC': int(cosmic_details['active_cell_count']),
            })
    finally:
        model.load_state_dict(state_dict, strict=True)
        for parameter, gradient in gradients:
            parameter.grad = None if gradient is None else gradient.clone()
        for parameter, enabled in requires_grad:
            parameter.requires_grad_(enabled)
        for module, training in module_modes:
            module.train(training)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    median_ratio = float(np.median(ratios))
    resolved = float(target_ratio / median_ratio)
    if not np.isfinite(resolved) or resolved <= 0.0 or resolved > 1.0:
        raise RuntimeError(
            'Background shape calibration requires a finite weight in (0, 1]; '
            f'got {resolved!r}')
    calibration = {
        'semantics': 'target_over_median_reject_gt_one_v1',
        'requested_batches': batches,
        'processed_batches': len(ratios),
        'raw_ratios': [float(value) for value in ratios],
        'median_ratio': median_ratio,
        'target_ratio': target_ratio,
        'resolved_weight': resolved,
        'achieved_median_ratio': resolved * median_ratio,
        'active_cells': active_cells,
    }
    config['resolved_background_shape_weight'] = resolved
    config['background_shape_gradient_calibration'] = calibration
    contract = {
        'semantics': _CONDITIONAL_SHAPE_LOSS,
        'cells': {
            'altitude_km': ['[200,300)', '[300,500]'],
            'local_time': {'day': '[06,18)', 'night': '[18,24) U [0,06)'},
            'abs_sin_i': ['[0,0.25)', '[0.25,0.5)', '[0.5,1]'],
        },
        'profile_first': True,
        'source_weights': {'FY': 0.5, 'COSMIC': 0.5},
        'min_profiles_per_cell': min_profiles,
        'target_std_min': 1e-4,
        'calibration': calibration,
    }
    with open(os.path.join(config['save_dir'], 'background_shape_contract.json'),
              'w', encoding='utf-8') as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2)
    print(
        f'[Background conditional shape] gradient median={median_ratio:.6f}, '
        f'resolved lambda={resolved:.6f}')
    return resolved


def _resolve_v14_gram_weight_from_ratios(raw_ratios, target_ratio):
    """Resolve the v14 coefficient without silently changing the target."""
    try:
        return resolve_v14_gram_weight(raw_ratios, target_ratio)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _valid_v14_gram_calibration(record, protocol):
    """True only for a complete, protocol-matching persisted calibration."""
    return valid_v14_gram_calibration(record, protocol)


def _validate_v14_resume_state(loaded, training_protocol, input_data_sha256):
    """Reject smoke or protocol-drifted v14 states before loading parameters."""
    if loaded.get('smoke_run', False) or loaded.get(
            'training_protocol', {}).get('smoke_run', False):
        raise ValueError('v14 smoke training states are never resumable')
    if loaded.get('training_protocol') != training_protocol:
        raise ValueError('v14 checkpoint training protocol differs from config')
    if loaded.get('input_data_sha256') != input_data_sha256:
        raise ValueError('v14 checkpoint input data SHA256 differs from config')
    if (loaded.get('stage') == 'analysis'
            and not _valid_v14_gram_calibration(
                loaded.get('gram_gradient_calibration'), training_protocol)):
        raise ValueError(
            'v14 Analysis resume requires a complete matching Gram calibration')


def _clone_calibration_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def _resolve_gram_weight(
        model, train_loader, cosmic_train_loader, batch_processor, device,
        config, sw_manager, iri_peak_manager, allowed_profile_ids=None):
    batches = int(config.get('gram_calibration_batches', 20))
    target_ratio = float(config.get('gram_gradient_target', 0.02))
    if batches < 1 or not 0.0 < target_ratio <= 1.0:
        raise ValueError(
            'Gram calibration batches must be positive and target in (0, 1]')

    # v12/v13 checkpoints retain their historical clipped calibration rule.
    if config.get('model_domain_semantics') not in _STRICT_LOW_PRIOR_DOMAINS:
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        ratios = []
        try:
            _set_stage_mode(model, 'analysis')
            cosmic_iter = iter(cosmic_train_loader)
            for batch_index, fy_batch in enumerate(train_loader):
                if batch_index >= batches:
                    break
                try:
                    cosmic_batch = next(cosmic_iter)
                except StopIteration:
                    cosmic_iter = iter(cosmic_train_loader)
                    cosmic_batch = next(cosmic_iter)
                observation, _, _, gram = _paired_analysis_losses(
                    model, batch_processor, fy_batch, cosmic_batch, device,
                    config, sw_manager, iri_peak_manager, allowed_profile_ids,
                    'exact_M10_M01_M11')
                _, _, gram_to_observation, _ = _gradient_norms(
                    observation, gram, _current_stage_trainable_parameters(model))
                value = float(gram_to_observation)
                if np.isfinite(value) and value > 0:
                    ratios.append(value)
        finally:
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        if not ratios:
            raise RuntimeError('observation Gram calibration produced no finite gradients')
        median_ratio = float(np.median(ratios))
        resolved = float(np.clip(target_ratio / median_ratio, 1e-4, 1.0))
        config['gram_gradient_calibration'] = {
            'batches': len(ratios),
            'median_ratio': median_ratio,
            'target_ratio': target_ratio,
        }
        return resolved

    if batches != 20:
        raise ValueError('v14 Gram calibration requires exactly 20 batches')
    module_modes = [(module, module.training) for module in model.modules()]
    state_dict = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    gradients = [
        (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
        for parameter in model.parameters()
    ]
    transient = {
        name: _clone_calibration_value(getattr(model.kalman_layer, name))
        for name in ('last_member_weights', 'last_inflation_scale')
        if hasattr(model.kalman_layer, name)
    }
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    ratios = []
    processed = 0
    try:
        _set_stage_mode(model, 'analysis')
        parameters = _current_stage_trainable_parameters(model)
        if not parameters:
            raise RuntimeError('v14 Gram calibration has no trainable Analysis parameters')
        fy_iter = iter(train_loader)
        cosmic_iter = iter(cosmic_train_loader)
        for batch_index in range(batches):
            try:
                fy_batch = next(fy_iter)
                cosmic_batch = next(cosmic_iter)
            except StopIteration as error:
                raise RuntimeError(
                    'v14 Gram calibration requires exactly 20 paired FY/COSMIC '
                    f'batches; loader ended at batch {batch_index + 1}') from error
            with torch.amp.autocast(
                    'cuda', enabled=bool(config.get('use_amp', False))
                    and device.type == 'cuda'):
                observation, _, _, gram = _paired_analysis_losses(
                    model, batch_processor, fy_batch, cosmic_batch, device,
                    config, sw_manager, iri_peak_manager, allowed_profile_ids,
                    'exact_M10_M01_M11')
            observation_grad, gram_grad, ratio, _ = _gradient_norms(
                observation, gram, parameters)
            values = tuple(float(value.detach().cpu()) for value in (
                observation_grad, gram_grad, ratio))
            if (not all(np.isfinite(value) and value > 0.0 for value in values)):
                raise RuntimeError(
                    'v14 Gram calibration requires finite positive observation, '
                    f'Gram, and ratio gradients at batch {batch_index + 1}: {values}')
            ratios.append(values[2])
            processed += 1
    finally:
        model.load_state_dict(state_dict, strict=True)
        for parameter, gradient in gradients:
            parameter.grad = (None if gradient is None else gradient.clone())
        for module, training in module_modes:
            module.train(training)
        for name, value in transient.items():
            setattr(model.kalman_layer, name, _clone_calibration_value(value))
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if processed != batches or len(ratios) != batches:
        raise RuntimeError(
            f'v14 Gram calibration processed {processed}/{batches} batches')
    resolved, median_ratio = _resolve_v14_gram_weight_from_ratios(
        ratios, target_ratio)
    quantiles = np.quantile(np.asarray(ratios, dtype=np.float64),
                            [0.0, 0.25, 0.50, 0.75, 1.0])
    tensor_count, parameter_count = _current_stage_parameter_counts(model)
    config['gram_gradient_calibration'] = {
        'requested_batches': batches,
        'processed_batches': processed,
        'valid_batches': len(ratios),
        'raw_ratios': [float(value) for value in ratios],
        'raw_ratio_summary': {
            key: float(value) for key, value in zip(
                ('min', 'q1', 'median', 'q3', 'max'), quantiles)
        },
        'median_ratio': median_ratio,
        'target_ratio': target_ratio,
        'resolved_weight': resolved,
        'achieved_median_ratio': resolved * median_ratio,
        'parameter_tensor_count': tensor_count,
        'parameter_count': parameter_count,
        'gradient_parameter_scope': GRADIENT_PARAMETER_SCOPE,
        'calibration_precision': CALIBRATION_PRECISION,
        'weight_resolution_semantics': (
            'target_over_median_no_lower_clip_reject_gt_one_v1'),
        'require_all_batches_finite_positive': True,
    }
    print(
        f'[HX Gram] 梯度比中位数={median_ratio:.6f}, '
        f'resolved lambda={resolved:.9f}')
    return resolved


def train_one_epoch(model, train_loader, batch_processor, optimizer, device,
                    config, epoch, stage, scaler, sw_manager, iri_peak_manager,
                    cosmic_train_loader, allowed_profile_ids=None):
    _set_stage_mode(model, stage)
    use_amp = config.get('use_amp', False) and scaler is not None
    delta = config.get('huber_delta', 0.2)
    # Background must cover both QC-v2 sources completely.  Analysis retains
    # the historical FY-anchored schedule and only cycles COSMIC.
    epoch_length = _background_epoch_length(
        train_loader, cosmic_train_loader, stage,
        config.get('model_domain_semantics'))
    fy_iter = iter(train_loader)
    cosmic_iter = iter(cosmic_train_loader)
    a1s8_low_geographic_day_blocks = set()
    stats = {key: 0.0 for key in (
        'total', 'observation', 'covariance', 'direction', 'gram',
        'fy_obs', 'cosmic_obs',
        'fy_m10', 'fy_m01', 'fy_m11',
        'cosmic_m10', 'cosmic_m01', 'cosmic_m11',
        'fy_direction_m10', 'fy_direction_m01', 'fy_direction_m11',
        'cosmic_direction_m10', 'cosmic_direction_m01',
        'cosmic_direction_m11',
        'iri', 'increment', 'low_altitude',
        'vertical', 'time', 'weighted_iri', 'weighted_increment',
        'weighted_low_altitude', 'low_altitude_profiles',
        'low_altitude_points', 'low_altitude_active_fraction',
        'fy_low_altitude', 'cosmic_low_altitude',
        'fy_low_altitude_profiles', 'cosmic_low_altitude_profiles',
        'fy_low_altitude_points', 'cosmic_low_altitude_points',
        'fy_low_altitude_active_fraction',
        'cosmic_low_altitude_active_fraction',
        'weighted_covariance', 'weighted_direction', 'weighted_gram',
        'background_stratum_bias', 'weighted_background_stratum_bias',
        'background_stratum_bias_active_cells',
        'background_shape', 'weighted_background_shape',
        'background_shape_active_cells',
        'weighted_vertical', 'weighted_time',
        'gradient_ratio', 'gradient_ratio_covariance',
        'gradient_ratio_direction', 'gradient_ratio_iri',
        'gradient_ratio_increment', 'gradient_ratio_low_altitude',
        'gradient_ratio_vertical',
        'gradient_ratio_time', 'gradient_ratio_gram',
        'gradient_ratio_background_stratum_bias',
            'gradient_ratio_background_shape',
            'background_adapter_gate_mean',
            'background_adapter_gate_min',
            'background_adapter_gate_max',
            'background_adapter_effective_abs_mean',
            'background_adapter_effective_abs_max',
            'background_residual_cap_saturation',
            'gradient_ratio_fy_m10', 'gradient_ratio_fy_m01',
        'gradient_ratio_fy_m11', 'gradient_ratio_cosmic_m10',
        'gradient_ratio_cosmic_m01', 'gradient_ratio_cosmic_m11',
        'fy_active_fraction', 'cosmic_active_fraction',
        'fy_gram_m10_coverage', 'fy_gram_m01_coverage',
        'fy_gram_m11_coverage', 'cosmic_gram_m10_coverage',
        'cosmic_gram_m01_coverage', 'cosmic_gram_m11_coverage',
        'K_FY_mean', 'K_COSMIC_mean', 'innov_FY_norm',
        'innov_COSMIC_norm', 'inflation', 'ne_delta_abs',
            )}
    if (_a1s8_enabled(config)
            or getattr(model, 'm00_candidate_semantics', 'disabled')
            in _M00_CANDIDATE_SCOPES):
        stats.update({
            'legacy_updates': 0.0,
            'balanced_updates': 0.0,
            'low_effective_gate_mass': 0.0,
            'high_effective_gate_mass': 0.0,
            'low_gate_profiles': 0.0,
            'high_gate_profiles': 0.0,
            'anchor_loss': 0.0,
            'anchor_weight': 0.0,
            'adapter_amplitude': 0.0,
            'adapter_cap_saturation': 0.0,
            'adapter_gradient_low': 0.0,
            'adapter_gradient_high': 0.0,
            'adapter_gradient_ratio': 0.0,
            'a1s8_ess': 0.0,
            'candidate_gradient_norm': 0.0,
        })
    m00_calibration_enabled = bool(
        config.get('m00_calibration_applied', False)
        or (config.get('m00_observation_calibration')
            and config.get('m00_observation_calibration_semantics', 'disabled')
            != 'disabled'))
    if m00_calibration_enabled:
        stats.update({
            'm00_calibration_loss': 0.0,
            'm00_calibration_points': 0.0,
            'm00_calibration_profiles': 0.0,
            'm00_effective_variance': 0.0,
            'm00_weight_max': 0.0,
            'm00_bias_mean': 0.0,
        })
    diagnostics = []
    audited_batches = 0
    processed_batches = 0
    max_train_batches = config.get('max_train_batches')
    if max_train_batches is not None and int(max_train_batches) < 1:
        raise ValueError('max_train_batches must be positive')
    source_mode_counts = {mode: 0 for mode in ('M10', 'M01', 'M11')}
    exact_mode_loss = (
        stage == 'analysis'
        and config.get('analysis_exact_mode_loss', False))
    started = time.time()

    for batch_idx in range(epoch_length):
        try:
            fy_batch = next(fy_iter)
        except StopIteration:
            fy_iter = iter(train_loader)
            fy_batch = next(fy_iter)
        coords, target, profile_ids = _unpack_source_batch(fy_batch, device)
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)

        try:
            cosmic_batch = next(cosmic_iter)
        except StopIteration:
            cosmic_iter = iter(cosmic_train_loader)
            cosmic_batch = next(cosmic_iter)
        cosmic_coords, cosmic_target, cosmic_ids = _unpack_source_batch(
            cosmic_batch, device)
        cosmic_sw = sw_manager.get_drivers_sequence(cosmic_coords[:, 3])
        cosmic_peak = (iri_peak_manager.get_iri_peak(cosmic_coords)
                       if iri_peak_manager is not None else None)
        if _a1s8_enabled(config) and stage == 'background':
            for sample_coords in (coords, cosmic_coords):
                flat_coords = sample_coords.reshape(-1, sample_coords.shape[-1])
                finite = torch.isfinite(flat_coords).all(dim=1)
                low = torch.sin(torch.deg2rad(flat_coords[:, 0])).abs() < 0.25
                selected = flat_coords[finite & low]
                if selected.numel():
                    # Coordinates are continuous; one-degree geo bins and UTC day
                    # provide a stable, conservative regional support audit.
                    lat = torch.round(selected[:, 0]).to(torch.int64).cpu().tolist()
                    lon = torch.round(selected[:, 1]).to(torch.int64).cpu().tolist()
                    day = torch.floor(selected[:, 3] / 24.0).to(torch.int64).cpu().tolist()
                    a1s8_low_geographic_day_blocks.update(zip(lat, lon, day))
        source_mode = None
        if stage == 'analysis':
            analysis_epoch = epoch - int(config['background_epochs'])
            analysis_batch = (
                analysis_epoch * len(train_loader)
                + batch_idx)
            if exact_mode_loss:
                source_mode = 'exact_M10_M01_M11'
                for mode in source_mode_counts:
                    source_mode_counts[mode] += 1
            else:
                source_mode = _source_mode_for_batch(
                    config.get('source_mode_schedule', 'random_profile'),
                    analysis_batch)
                if source_mode in source_mode_counts:
                    source_mode_counts[source_mode] += 1
                elif source_mode == 'balanced_profile_112':
                    for ids in (profile_ids, cosmic_ids):
                        for mode, count in _balanced_profile_mode_counts(
                                ids, analysis_epoch).items():
                            source_mode_counts[mode] += count

        with torch.amp.autocast('cuda', enabled=use_amp):
            fy_pred, _, _, _, fy_extras = _source_forward(
                model, batch_processor, coords, sw_seq, iri_peak,
                stage, 'FY', profile_ids,
                stage == 'analysis' and not exact_mode_loss, config,
                iri_peak_manager, allowed_profile_ids, source_mode,
                analysis_epoch if stage == 'analysis' else 0)
            cosmic_pred, _, _, _, cosmic_extras = _source_forward(
                model, batch_processor, cosmic_coords, cosmic_sw, cosmic_peak,
                stage, 'COSMIC', cosmic_ids,
                stage == 'analysis' and not exact_mode_loss, config,
                iri_peak_manager, allowed_profile_ids, source_mode,
                analysis_epoch if stage == 'analysis' else 0)
            zero = fy_pred.sum() * 0.0
            fy_mode_losses = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            cosmic_mode_losses = {
                mode: zero for mode in _EXACT_MODE_SOURCES}
            if exact_mode_loss:
                fy_loss, fy_mode_losses, fy_mode_active = (
                    exact_mode_profile_losses(
                        fy_extras, target, profile_ids, delta))
                cosmic_loss, cosmic_mode_losses, cosmic_mode_active = (
                    exact_mode_profile_losses(
                        cosmic_extras, cosmic_target, cosmic_ids, delta))
                fy_active = fy_mode_active['M11']
                cosmic_active = cosmic_mode_active['M11']
                observation_loss = 0.5 * (fy_loss + cosmic_loss)
            else:
                fy_active = _analysis_active_mask(fy_extras)
                cosmic_active = _analysis_active_mask(cosmic_extras)
                active_only = (
                    stage == 'analysis'
                    and config.get('analysis_loss_active_only', False))
                fy_gate_details = {}
                cosmic_gate_details = {}
                if _a1s8_enabled(config):
                    fy_loss, fy_gate_details = background_gate_aware_loss(
                        fy_pred, target, coords, profile_ids,
                        fy_extras['background_adapter_gate'], delta=delta,
                        alt_range=(200.0, 500.0), return_details=True)
                    cosmic_loss, cosmic_gate_details = background_gate_aware_loss(
                        cosmic_pred, cosmic_target, cosmic_coords, cosmic_ids,
                        cosmic_extras['background_adapter_gate'], delta=delta,
                        alt_range=(200.0, 500.0), return_details=True)
                elif _a1s7_enabled(config):
                    fy_loss = background_profile_stratum_loss(
                        fy_pred, target, coords, profile_ids, delta=delta,
                        alt_range=(200.0, 500.0))
                    cosmic_loss = background_profile_stratum_loss(
                        cosmic_pred, cosmic_target, cosmic_coords, cosmic_ids,
                        delta=delta, alt_range=(200.0, 500.0))
                else:
                    fy_loss = profile_huber_loss(
                        fy_pred, target, profile_ids, delta=delta,
                        valid_mask=fy_active if active_only else None)
                    cosmic_loss = profile_huber_loss(
                        cosmic_pred, cosmic_target, cosmic_ids, delta=delta,
                        valid_mask=cosmic_active if active_only else None)
                if stage == 'background' and _a1s8_enabled(config):
                    # The tuple is the legacy/balanced sampler schedule,
                    # not FY/COSMIC weights. Keep source losses equal.
                    _, _, mix_kind = a1s8_source_mix(batch_idx, config)
                    fy_mix = cosmic_mix = 0.5
                    observation_loss = 0.5 * (fy_loss + cosmic_loss)
                else:
                    fy_mix, cosmic_mix, mix_kind = 0.5, 0.5, 'equal'
                    observation_loss = 0.5 * (fy_loss + cosmic_loss)
                if _a1s8_enabled(config) and stage == 'background':
                    stats[f'{mix_kind}_updates'] += 1.0
                    for details in (fy_gate_details, cosmic_gate_details):
                        total_mass = (
                            details.get('low_effective_mass', 0.0)
                            + details.get('high_effective_mass', 0.0))
                        if total_mass > 0.0:
                            # FY and COSMIC contribute equally; average their
                            # within-source gate fractions before epoch mean.
                            stats['low_effective_gate_mass'] += 0.5 * (
                                details.get('low_effective_mass', 0.0)
                                / total_mass)
                            stats['high_effective_gate_mass'] += 0.5 * (
                                details.get('high_effective_mass', 0.0)
                                / total_mass)
                        stats['low_gate_profiles'] += float(
                            details.get('low_profiles', 0))
                        stats['high_gate_profiles'] += float(
                            details.get('high_profiles', 0))
            mode_gradient_components = {
                f'fy_{mode.lower()}': (
                    0.5 * _EXACT_MODE_WEIGHTS[mode] * fy_mode_losses[mode])
                for mode in _EXACT_MODE_SOURCES
            }
            mode_gradient_components.update({
                f'cosmic_{mode.lower()}': (
                    0.5 * _EXACT_MODE_WEIGHTS[mode]
                    * cosmic_mode_losses[mode])
                for mode in _EXACT_MODE_SOURCES
            })
        if m00_calibration_enabled:
            calibration = config.get('m00_observation_calibration')
            if not calibration:
                raise RuntimeError('M00 calibration was enabled but not loaded')
            m00_calibration_details = {'FY': {}, 'COSMIC': {}}
            fy_magnetic_sin_i, fy_aacgm_latitude = _batch_magnetic_values(
                fy_batch, config, 'FY')
            cosmic_magnetic_sin_i, cosmic_aacgm_latitude = _batch_magnetic_values(
                cosmic_batch, config, 'COSMIC')
            use_qd_strata = str(config.get(
                'magnetic_stratum_semantics', '')).startswith('qd_')
            fy_loss, m00_calibration_details['FY'] = m00_observation_error_loss(
                fy_pred, target, coords, profile_ids, 'FY', calibration,
                delta=delta, magnetic_sin_i=fy_magnetic_sin_i,
                aacgm_latitude_deg=(
                    None if use_qd_strata else fy_aacgm_latitude),
                qd_latitude_deg=(
                    fy_aacgm_latitude if use_qd_strata else None),
                strict_magnetic=bool(config.get(
                    'm00_observation_strict_magnetic', False)),
                use_representativeness=bool(config.get(
                    'm00_representativeness_enabled', False)),
                use_localization=bool(config.get(
                    'm00_localization_enabled', False)),
                weight_cap=float(config.get(
                    'm00_observation_weight_cap', 4.0)))
            cosmic_loss, m00_calibration_details['COSMIC'] = (
                m00_observation_error_loss(
                    cosmic_pred, cosmic_target, cosmic_coords, cosmic_ids,
                    'COSMIC', calibration, delta=delta,
                    magnetic_sin_i=cosmic_magnetic_sin_i,
                    aacgm_latitude_deg=(
                        None if use_qd_strata else cosmic_aacgm_latitude),
                    qd_latitude_deg=(
                        cosmic_aacgm_latitude if use_qd_strata else None),
                    strict_magnetic=bool(config.get(
                        'm00_observation_strict_magnetic', False)),
                    use_representativeness=bool(config.get(
                        'm00_representativeness_enabled', False)),
                    use_localization=bool(config.get(
                        'm00_localization_enabled', False)),
                    weight_cap=float(config.get(
                        'm00_observation_weight_cap', 4.0))))
            stats['m00_calibration_loss'] += float(
                (0.5 * (fy_loss + cosmic_loss)).detach().cpu())
            for details in m00_calibration_details.values():
                stats['m00_calibration_points'] += float(
                    details.get('valid_points', 0))
                stats['m00_calibration_profiles'] += float(
                    details.get('valid_profiles', 0))
                stats['m00_effective_variance'] += float(
                    details.get('effective_variance_mean', 0.0))
                stats['m00_weight_max'] = max(
                    stats['m00_weight_max'],
                    float(details.get('weight_max', 0.0)))
                stats['m00_bias_mean'] += float(
                    details.get('bias_mean', 0.0))
        covariance_loss = observation_loss.new_zeros(())
        if (stage == 'analysis'
                and (config.get('use_covariance_moment_loss', False)
                     or config.get('use_empirical_covariance_loss', False))):
            covariance_loss = 0.5 * (
                _covariance_training_loss(
                    fy_extras, target, profile_ids, 'FY',
                    batch_processor, config)
                + _covariance_training_loss(
                    cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC',
                    batch_processor, config))
        weighted_covariance = (
            float(config.get('resolved_covariance_weight', 0.0))
            * covariance_loss)
        fy_gram_losses = {
            mode: zero for mode in _EXACT_MODE_SOURCES}
        cosmic_gram_losses = {
            mode: zero for mode in _EXACT_MODE_SOURCES}
        fy_gram_coverage = {
            mode: zero for mode in _EXACT_MODE_SOURCES}
        cosmic_gram_coverage = {
            mode: zero for mode in _EXACT_MODE_SOURCES}
        gram_loss = observation_loss.new_zeros(())
        if stage == 'analysis' and config.get(
                'use_observation_gram_loss', False):
            if not exact_mode_loss:
                raise ValueError(
                    'observation Gram loss requires analysis_exact_mode_loss')
            fy_gram_loss, fy_gram_losses, fy_gram_coverage = (
                observation_gram_whitening_loss(
                    fy_extras, profile_ids, model.kalman_layer,
                    return_details=True))
            cosmic_gram_loss, cosmic_gram_losses, cosmic_gram_coverage = (
                observation_gram_whitening_loss(
                    cosmic_extras, cosmic_ids, model.kalman_layer,
                    return_details=True))
            gram_loss = 0.5 * (fy_gram_loss + cosmic_gram_loss)
        weighted_gram = (
            float(config.get('resolved_gram_weight', 0.0)) * gram_loss)
        fy_direction_modes = {
            mode: zero for mode in _EXACT_MODE_SOURCES}
        cosmic_direction_modes = {
            mode: zero for mode in _EXACT_MODE_SOURCES}
        direction_loss = observation_loss.new_zeros(())
        if (stage == 'analysis'
                and config.get('use_direction_loss', False)):
            if not exact_mode_loss:
                raise ValueError(
                    'direction loss requires analysis_exact_mode_loss')
            fy_direction, fy_direction_modes, _ = (
                exact_mode_direction_losses(
                    fy_extras, target, profile_ids, 'FY'))
            cosmic_direction, cosmic_direction_modes, _ = (
                exact_mode_direction_losses(
                    cosmic_extras, cosmic_target, cosmic_ids, 'COSMIC'))
            direction_loss = 0.5 * (
                fy_direction + cosmic_direction)
        weighted_direction = (
            float(config.get('resolved_direction_weight', 0.0))
            * direction_loss)
        background_stratum_bias_loss = observation_loss.new_zeros(())
        background_bias_details = {
            'FY': {'active_cell_count': 0},
            'COSMIC': {'active_cell_count': 0},
        }
        background_anchor_loss = observation_loss.new_zeros(())
        background_anchor_details = {
            'FY': {'profiles': 0, 'active_strata': 0},
            'COSMIC': {'profiles': 0, 'active_strata': 0},
        }
        weighted_background_anchor = observation_loss.new_zeros(())
        weighted_background_stratum_bias = observation_loss.new_zeros(())
        if stage == 'background':
            fy_bias, background_bias_details['FY'] = (
                background_stratified_bias_loss(
                    fy_extras['ne_bkg'], target, coords, profile_ids,
                    return_details=True,
                    **_background_bias_kwargs(config, 'FY')))
            cosmic_bias, background_bias_details['COSMIC'] = (
                background_stratified_bias_loss(
                    cosmic_extras['ne_bkg'], cosmic_target,
                    cosmic_coords, cosmic_ids, return_details=True,
                    **_background_bias_kwargs(config, 'COSMIC')))
            background_stratum_bias_loss = (
                fy_mix * fy_bias + cosmic_mix * cosmic_bias
                if _a1s8_enabled(config)
                else 0.5 * (fy_bias + cosmic_bias))
            weighted_background_stratum_bias = (
                float(config.get(
                    'resolved_background_stratum_bias_weight',
                    config.get('w_background_stratum_bias', 0.0)))
                * background_stratum_bias_loss)
        background_shape_loss = observation_loss.new_zeros(())
        background_shape_details = {
            'FY': {'active_cell_count': 0},
            'COSMIC': {'active_cell_count': 0},
        }
        if (_a1s8_enabled(config)
                or getattr(model, 'm00_candidate_semantics', 'disabled')
                in _M00_CANDIDATE_SCOPES):
            fy_anchor, background_anchor_details['FY'] = (
                background_adapter_anchor_loss(
                            fy_extras['background_adapter_effective_delta'],
                            coords, profile_ids, delta=delta,
                            return_details=True))
            cosmic_anchor, background_anchor_details['COSMIC'] = (
                background_adapter_anchor_loss(
                    cosmic_extras['background_adapter_effective_delta'],
                    cosmic_coords, cosmic_ids, delta=delta,
                    return_details=True))
            background_anchor_loss = (
                fy_mix * fy_anchor + cosmic_mix * cosmic_anchor)
            anchor_target = float(config.get(
                'background_anchor_gradient_target', 0.10))
            warmup_epochs = max(1, int(config.get(
                'background_anchor_warmup_epochs', 1)))
            anchor_weight = float(config.get(
                'resolved_background_anchor_weight', anchor_target))
            anchor_weight *= min(1.0, (int(epoch) + 1) / warmup_epochs)
            calibration_batches = max(
                1, int(config.get(
                    'background_anchor_calibration_batches', 20)))
            if int(epoch) < warmup_epochs:
                anchor_weight *= min(
                    1.0, (int(batch_idx) + 1) / calibration_batches)
            weighted_background_anchor = (
                anchor_weight * background_anchor_loss)
            if (stage == 'background' and config.get('background_shape_loss')
                    == _CONDITIONAL_SHAPE_LOSS):
                minimum = int(config.get(
                    'background_shape_min_profiles_per_cell', 8))
                fy_shape, background_shape_details['FY'] = (
                    background_conditional_shape_loss(
                        fy_extras['ne_bkg'], target, coords, profile_ids,
                        delta=delta, min_profiles_per_cell=minimum,
                        return_details=True))
                cosmic_shape, background_shape_details['COSMIC'] = (
                    background_conditional_shape_loss(
                        cosmic_extras['ne_bkg'], cosmic_target, cosmic_coords,
                        cosmic_ids, delta=delta,
                        min_profiles_per_cell=minimum, return_details=True))
                background_shape_loss = 0.5 * (fy_shape + cosmic_shape)
        weighted_background_shape = (
            float(config.get('resolved_background_shape_weight') or 0.0)
            * background_shape_loss)
        data_loss = (
            observation_loss + weighted_covariance + weighted_direction
            + weighted_gram)

        vertical_loss, time_loss = _structure_losses(
            model, batch_processor, sw_manager, iri_peak_manager,
            coords, profile_ids, stage, config, allowed_profile_ids)

        low_altitude_loss = observation_loss.new_zeros(())
        low_altitude_details = {
            'FY': {'profiles': 0, 'points': 0, 'active_fraction': 0.0},
            'COSMIC': {'profiles': 0, 'points': 0, 'active_fraction': 0.0},
        }
        fy_low_altitude = observation_loss.new_zeros(())
        cosmic_low_altitude = observation_loss.new_zeros(())
        if _low_altitude_prior_protocol(config) is not None:
            fy_low_altitude, low_altitude_details['FY'] = (
                _low_altitude_prior_loss(
                    model, batch_processor, coords, profile_ids, 'FY', stage,
                    config, sw_manager, iri_peak_manager, allowed_profile_ids))
            cosmic_low_altitude, low_altitude_details['COSMIC'] = (
                _low_altitude_prior_loss(
                    model, batch_processor, cosmic_coords, cosmic_ids,
                    'COSMIC', stage, config, sw_manager, iri_peak_manager,
                    allowed_profile_ids))
            low_altitude_loss = 0.5 * (
                fy_low_altitude + cosmic_low_altitude)

        if stage == 'background':
            fy_iri = profile_huber_loss(
                fy_extras['ne_bkg'], fy_extras['ne_iri'],
                profile_ids, delta=delta)
            cosmic_iri = profile_huber_loss(
                cosmic_extras['ne_bkg'], cosmic_extras['ne_iri'],
                cosmic_ids, delta=delta)
            iri_loss = ((fy_mix * fy_iri + cosmic_mix * cosmic_iri)
                        if _a1s8_enabled(config)
                        else 0.5 * (fy_iri + cosmic_iri))
            increment_loss = observation_loss.new_zeros(())
            weighted_iri = config.get('w_iri', 0.02) * iri_loss
            weighted_increment = increment_loss
            weighted_low_altitude = (
                config.get('w_low_altitude_background_iri', 0.0)
                * low_altitude_loss)
            weighted_vertical = (
                config.get('w_vertical_background', 0.02) * vertical_loss)
            weighted_time = (
                config.get('w_time_background', 0.01) * time_loss)
            auxiliary_loss = (
                weighted_iri + weighted_low_altitude + weighted_vertical
                + weighted_time + weighted_background_stratum_bias
                + weighted_background_shape + weighted_background_anchor
            )
        else:
            fy_increment = F.smooth_l1_loss(
                fy_extras['ne_residual'],
                torch.zeros_like(fy_extras['ne_residual']), beta=delta)
            cosmic_increment = F.smooth_l1_loss(
                cosmic_extras['ne_residual'],
                torch.zeros_like(cosmic_extras['ne_residual']), beta=delta)
            increment_loss = 0.5 * (fy_increment + cosmic_increment)
            iri_loss = observation_loss.new_zeros(())
            weighted_iri = iri_loss
            weighted_increment = (
                config.get('w_increment', 0.01) * increment_loss)
            weighted_low_altitude = (
                config.get('w_low_altitude_analysis_increment', 0.0)
                * low_altitude_loss)
            weighted_vertical = (
                config.get('w_vertical_analysis', 0.05) * vertical_loss)
            weighted_time = (
                config.get('w_time_analysis', 0.02) * time_loss)
            auxiliary_loss = (
                weighted_increment + weighted_low_altitude + weighted_vertical
                + weighted_time
            )
        total_loss = data_loss + auxiliary_loss
        auxiliary_components = {
            'covariance': weighted_covariance,
            'direction': weighted_direction,
            'gram': weighted_gram,
            'background_stratum_bias': weighted_background_stratum_bias,
            'background_shape': weighted_background_shape,
            'iri': weighted_iri,
            'increment': weighted_increment,
            'low_altitude': weighted_low_altitude,
            'vertical': weighted_vertical,
            'time': weighted_time,
        }
        gradient_components = {
            **auxiliary_components,
            **mode_gradient_components,
        }

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f'non-finite {stage} loss at epoch={epoch + 1}, batch={batch_idx}')

        first_stage_epoch = (
            epoch == 0 if stage == 'background'
            else epoch == int(config['background_epochs']))
        audit_batch = first_stage_epoch and batch_idx < 100
        if audit_batch:
            observation_grad, auxiliary_grad, ratio, component_grads = _gradient_norms(
                observation_loss,
                auxiliary_loss + weighted_covariance + weighted_direction
                + weighted_gram,
                _current_stage_trainable_parameters(model),
                gradient_components)
            audited_batches += 1
        else:
            observation_grad = total_loss.new_zeros(())
            auxiliary_grad = total_loss.new_zeros(())
            ratio = total_loss.new_zeros(())
            component_grads = {
                name: total_loss.new_zeros(())
                for name in auxiliary_components
            }

        if audit_batch:
            representativeness_stats = {}
            for target_name, extras in (
                    ('fy_target', fy_extras),
                    ('cosmic_target', cosmic_extras)):
                for observation_source in ('FY', 'COSMIC'):
                    active_precision = (
                        extras[f'precision_{observation_source}'] > 0)
                    values = extras[
                        f'representativeness_{observation_source}'][
                            active_precision]
                    prefix = (
                        f'{target_name}_representativeness_'
                        f'{observation_source.lower()}')
                    representativeness_stats[f'{prefix}_mean'] = (
                        values.mean().item() if len(values) else 1.0)
                    representativeness_stats[f'{prefix}_min'] = (
                        values.min().item() if len(values) else 1.0)
                    representativeness_stats[f'{prefix}_max'] = (
                        values.max().item() if len(values) else 1.0)
            diagnostics.append({
                'epoch': epoch + 1,
                'stage': stage,
                'batch': batch_idx + 1,
                'source_mode': source_mode,
                'fy_obs_raw': fy_loss.item(),
                'cosmic_obs_raw': cosmic_loss.item(),
                'fy_obs_weighted': 0.5 * fy_loss.item(),
                'cosmic_obs_weighted': 0.5 * cosmic_loss.item(),
                'fy_active_fraction': fy_active.float().mean().item(),
                'cosmic_active_fraction': cosmic_active.float().mean().item(),
                'covariance_raw': covariance_loss.item(),
                'covariance_weighted': weighted_covariance.item(),
                'gram_raw': gram_loss.item(),
                'gram_weighted': weighted_gram.item(),
                'background_stratum_bias_raw': (
                    background_stratum_bias_loss.item()),
                'background_stratum_bias_weighted': (
                    weighted_background_stratum_bias.item()),
                'background_stratum_bias_active_cells': {
                    source: int(details.get('active_cell_count', 0))
                    for source, details in background_bias_details.items()
                },
                'background_stratum_bias_magnetic_active_cells': {
                    source: int(details.get('magnetic_active_cell_count', 0))
                    for source, details in background_bias_details.items()
                },
                'background_stratum_bias_height_raw': {
                    source: float(details.get('height_daynight_loss', 0.0))
                    for source, details in background_bias_details.items()
                },
                'background_stratum_bias_magnetic_raw': {
                    source: float(details.get('magnetic_loss', 0.0))
                    for source, details in background_bias_details.items()
                },
                'background_shape_raw': background_shape_loss.item(),
                'background_shape_weighted': weighted_background_shape.item(),
                'background_shape_active_cells': {
                    source: int(details.get('active_cell_count', 0))
                    for source, details in background_shape_details.items()
                },
                **{
                    f'fy_gram_{mode.lower()}_raw':
                    fy_gram_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_gram_{mode.lower()}_raw':
                    cosmic_gram_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'fy_gram_{mode.lower()}_coverage':
                    fy_gram_coverage[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_gram_{mode.lower()}_coverage':
                    cosmic_gram_coverage[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                'direction_raw': direction_loss.item(),
                'direction_weighted': weighted_direction.item(),
                'iri_raw': iri_loss.item(),
                'increment_raw': increment_loss.item(),
                'low_altitude_raw': low_altitude_loss.item(),
                'low_altitude_weighted': weighted_low_altitude.item(),
                'fy_low_altitude_raw': fy_low_altitude.item(),
                'cosmic_low_altitude_raw': cosmic_low_altitude.item(),
                'fy_low_altitude_profiles': low_altitude_details['FY']['profiles'],
                'cosmic_low_altitude_profiles': (
                    low_altitude_details['COSMIC']['profiles']),
                'fy_low_altitude_points': low_altitude_details['FY']['points'],
                'cosmic_low_altitude_points': low_altitude_details['COSMIC']['points'],
                'fy_low_altitude_active_fraction': (
                    low_altitude_details['FY']['active_fraction']),
                'cosmic_low_altitude_active_fraction': (
                    low_altitude_details['COSMIC']['active_fraction']),
                'vertical_raw': vertical_loss.item(),
                'time_raw': time_loss.item(),
                'iri_weighted': weighted_iri.item(),
                'increment_weighted': weighted_increment.item(),
                'vertical_weighted': weighted_vertical.item(),
                'time_weighted': weighted_time.item(),
                'observation_grad_norm': observation_grad.item(),
                'auxiliary_grad_norm': auxiliary_grad.item(),
                'auxiliary_to_observation_grad': ratio.item(),
                'fy_hx_effective_rank': fy_extras['hx_effective_rank'].mean().item(),
                'cosmic_hx_effective_rank': cosmic_extras['hx_effective_rank'].mean().item(),
                'fy_hx_condition': fy_extras['hx_condition'].mean().item(),
                'cosmic_hx_condition': cosmic_extras['hx_condition'].mean().item(),
                'fy_hx_numeric_rank': fy_extras['hx_numeric_rank'].float().mean().item(),
                'cosmic_hx_numeric_rank': cosmic_extras['hx_numeric_rank'].float().mean().item(),
                **{
                    f'fy_{mode.lower()}_raw': fy_mode_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_{mode.lower()}_raw':
                    cosmic_mode_losses[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'fy_{mode.lower()}_weighted': (
                        0.5 * _EXACT_MODE_WEIGHTS[mode]
                        * fy_mode_losses[mode].item())
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_{mode.lower()}_weighted': (
                        0.5 * _EXACT_MODE_WEIGHTS[mode]
                        * cosmic_mode_losses[mode].item())
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'fy_direction_{mode.lower()}_raw':
                    fy_direction_modes[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'cosmic_direction_{mode.lower()}_raw':
                    cosmic_direction_modes[mode].item()
                    for mode in _EXACT_MODE_SOURCES
                },
                **{
                    f'{name}_grad_norm': value.item()
                    for name, value in component_grads.items()
                },
                **{
                    f'{name}_to_observation_grad': (
                        value / observation_grad.clamp_min(1e-12)
                    ).item()
                    for name, value in component_grads.items()
                },
                **representativeness_stats,
            })

        if (config.get('smoke_run', False) and first_stage_epoch
                and batch_idx == 0):
            _write_smoke_preflight(
                config, stage, epoch, batch_idx,
                {
                    'total_auxiliary_to_observation': float(ratio.detach().cpu()),
                    'low_altitude_to_observation': float(
                        component_grads['low_altitude']
                        / observation_grad.clamp_min(1e-12)),
                    'gram_to_observation': float(
                        component_grads['gram']
                        / observation_grad.clamp_min(1e-12)),
                },
                {
                    f'{target}_{mode}': float(
                        (fy_gram_coverage if target == 'fy'
                         else cosmic_gram_coverage)[mode].detach().cpu())
                    for target in ('fy', 'cosmic')
                    for mode in _EXACT_MODE_SOURCES
                },
                {
                    'observation': float(observation_loss.detach().cpu()),
                    'gram': float(gram_loss.detach().cpu()),
                    'low_altitude': float(low_altitude_loss.detach().cpu()),
                    'total': float(total_loss.detach().cpu()),
                })

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                _current_stage_trainable_parameters(model),
                config.get('grad_clip', 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                _current_stage_trainable_parameters(model),
                config.get('grad_clip', 1.0))
            optimizer.step()

        if ((_a1s8_enabled(config)
             or getattr(model, 'm00_candidate_semantics', 'disabled')
             in _M00_CANDIDATE_SCOPES) and stage == 'background'):
            stats['anchor_loss'] += float(
                background_anchor_loss.detach().cpu())
            stats['anchor_weight'] += float(anchor_weight)
            adapter_values = torch.cat((
                fy_extras['background_adapter_effective_delta'].reshape(-1),
                cosmic_extras['background_adapter_effective_delta'].reshape(-1)))
            finite_adapter = adapter_values[torch.isfinite(adapter_values)]
            if finite_adapter.numel():
                stats['adapter_amplitude'] += float(
                    finite_adapter.abs().mean().detach().cpu())
                cap = (float(config.get('m00_candidate_residual_cap', 0.05))
                       if getattr(model, 'm00_candidate_semantics', 'disabled')
                       in _M00_CANDIDATE_SCOPES else float(config.get(
                           'background_adapter_residual_cap', 0.1)))
                stats['adapter_cap_saturation'] += float(
                    (finite_adapter.abs() >= cap * 0.999).float().mean().cpu())
            low_grad = _module_gradient_norm(getattr(
                model, 'background_condition_adapter_low', None))
            high_grad = _module_gradient_norm(getattr(
                model, 'background_condition_adapter_high', None))
            stats['adapter_gradient_low'] += low_grad
            stats['adapter_gradient_high'] += high_grad
            stats['adapter_gradient_ratio'] += low_grad / max(high_grad, 1e-12)
            if getattr(model, 'm00_candidate_semantics', 'disabled') in _M00_CANDIDATE_SCOPES:
                stats['candidate_gradient_norm'] += _module_gradient_norm(
                    model.background_decoder[-1]
                    if getattr(model, 'm00_candidate_semantics') == _M00_DECODER_TRUST_SCOPE
                    else model.m00_peak_relative_head)
            gate_values = torch.cat((
                fy_extras['background_adapter_gate'].reshape(-1),
                cosmic_extras['background_adapter_gate'].reshape(-1))).detach()
            if gate_values.numel():
                gate_weights = torch.cat((1.0 - gate_values, gate_values))
                mass = gate_weights.sum()
                stats['a1s8_ess'] += float(
                    mass.square().div(
                        gate_weights.square().sum().clamp_min(1e-12)).cpu())

        stats['total'] += total_loss.item()
        stats['observation'] += observation_loss.item()
        stats['covariance'] += covariance_loss.item()
        stats['direction'] += direction_loss.item()
        stats['gram'] += gram_loss.item()
        stats['fy_obs'] += fy_loss.item()
        stats['cosmic_obs'] += cosmic_loss.item()
        for mode in _EXACT_MODE_SOURCES:
            stats[f'fy_{mode.lower()}'] += fy_mode_losses[mode].item()
            stats[f'cosmic_{mode.lower()}'] += cosmic_mode_losses[mode].item()
            stats[f'fy_direction_{mode.lower()}'] += (
                fy_direction_modes[mode].item())
            stats[f'cosmic_direction_{mode.lower()}'] += (
                cosmic_direction_modes[mode].item())
        stats['fy_active_fraction'] += fy_active.float().mean().item()
        stats['cosmic_active_fraction'] += cosmic_active.float().mean().item()
        stats['iri'] += iri_loss.item()
        stats['increment'] += increment_loss.item()
        stats['low_altitude'] += low_altitude_loss.item()
        stats['weighted_low_altitude'] += weighted_low_altitude.item()
        stats['fy_low_altitude'] += fy_low_altitude.item()
        stats['cosmic_low_altitude'] += cosmic_low_altitude.item()
        for source, prefix in (('FY', 'fy'), ('COSMIC', 'cosmic')):
            details = low_altitude_details[source]
            stats[f'{prefix}_low_altitude_profiles'] += details['profiles']
            stats[f'{prefix}_low_altitude_points'] += details['points']
            stats[f'{prefix}_low_altitude_active_fraction'] += (
                details['active_fraction'])
        stats['low_altitude_profiles'] += sum(
            details['profiles'] for details in low_altitude_details.values())
        stats['low_altitude_points'] += sum(
            details['points'] for details in low_altitude_details.values())
        stats['low_altitude_active_fraction'] += 0.5 * sum(
            details['active_fraction'] for details in low_altitude_details.values())
        stats['vertical'] += vertical_loss.item()
        stats['time'] += time_loss.item()
        stats['weighted_iri'] += weighted_iri.item()
        stats['weighted_increment'] += weighted_increment.item()
        stats['weighted_covariance'] += weighted_covariance.item()
        stats['weighted_direction'] += weighted_direction.item()
        stats['weighted_gram'] += weighted_gram.item()
        stats['background_stratum_bias'] += (
            background_stratum_bias_loss.item())
        stats['weighted_background_stratum_bias'] += (
            weighted_background_stratum_bias.item())
        stats['background_stratum_bias_active_cells'] += 0.5 * sum(
            details.get('active_cell_count', 0)
            for details in background_bias_details.values())
        stats['background_shape'] += background_shape_loss.item()
        stats['weighted_background_shape'] += weighted_background_shape.item()
        stats['background_shape_active_cells'] += 0.5 * sum(
            details.get('active_cell_count', 0)
            for details in background_shape_details.values())
        stats['weighted_vertical'] += weighted_vertical.item()
        stats['weighted_time'] += weighted_time.item()
        stats['gradient_ratio'] += ratio.item()
        for name, value in component_grads.items():
            stats[f'gradient_ratio_{name}'] += (
                value / observation_grad.clamp_min(1e-12)).item()
        for mode in _EXACT_MODE_SOURCES:
            stats[f'fy_gram_{mode.lower()}_coverage'] += (
                fy_gram_coverage[mode].item())
            stats[f'cosmic_gram_{mode.lower()}_coverage'] += (
                cosmic_gram_coverage[mode].item())
        stats['K_FY_mean'] += fy_extras['K_FY'].mean().item()
        stats['K_COSMIC_mean'] += fy_extras['K_COSMIC'].mean().item()
        stats['innov_FY_norm'] += fy_extras['innov_FY'].norm(dim=-1).mean().item()
        stats['innov_COSMIC_norm'] += fy_extras['innov_COSMIC'].norm(dim=-1).mean().item()
        stats['inflation'] += fy_extras['inflation_scale'].item()
        stats['ne_delta_abs'] += fy_extras['ne_residual'].abs().mean().item()
        if stage == 'background' and (
                _a1s7_enabled(config) or _a1s8_enabled(config)):
            adapter_gate = torch.cat((
                fy_extras['background_adapter_gate'].reshape(-1),
                cosmic_extras['background_adapter_gate'].reshape(-1)))
            adapter_effective = torch.cat((
                fy_extras['background_adapter_effective_delta'].reshape(-1),
                cosmic_extras['background_adapter_effective_delta'].reshape(-1)))
            residual = torch.cat((
                fy_extras['background_residual_raw'].reshape(-1),
                cosmic_extras['background_residual_raw'].reshape(-1)))
            cap = float(config.get('background_residual_cap', 0.5))
            stats['background_adapter_gate_mean'] += adapter_gate.mean().item()
            stats['background_adapter_gate_min'] += adapter_gate.min().item()
            stats['background_adapter_gate_max'] += adapter_gate.max().item()
            stats['background_adapter_effective_abs_mean'] += (
                adapter_effective.abs().mean().item())
            stats['background_adapter_effective_abs_max'] += (
                adapter_effective.abs().max().item())
            stats['background_residual_cap_saturation'] += (
                (residual.abs() >= cap - 1e-6).float().mean().item())

        if (batch_idx + 1) % 100 == 0:
            print(
                f'  [{batch_idx + 1:>5}/{epoch_length}] {stage} '
                f'loss={total_loss.item():.4f} FY={fy_loss.item():.4f} '
                f'COSMIC={cosmic_loss.item():.4f} '
                f'V={vertical_loss.item():.4f} T={time_loss.item():.4f} '
                f'({time.time() - started:.0f}s)')
        processed_batches += 1
        if (max_train_batches is not None
                and processed_batches >= int(max_train_batches)):
            break

    batches = max(1, processed_batches)
    result = {key: value / batches for key, value in stats.items()}
    result['gradient_ratio'] = (
        stats['gradient_ratio'] / audited_batches if audited_batches else 0.0)
    for name in gradient_components:
        result[f'gradient_ratio_{name}'] = (
            stats[f'gradient_ratio_{name}'] / audited_batches
            if audited_batches else 0.0)
    result['gradient_audit_batches'] = audited_batches
    result['processed_batches'] = processed_batches
    result['source_mode_counts'] = source_mode_counts
    result['source_mode_count_unit'] = (
        'batches_all_modes' if exact_mode_loss else
        'profiles' if config.get(
            'source_mode_schedule') == 'balanced_profile_112'
        else 'batches')
    if _a1s8_enabled(config):
        ratio = float(result.get('adapter_gradient_ratio', 0.0))
        result['a1s8_exposure'] = {
            'legacy_updates': int(round(result.get(
                'legacy_updates', 0.0) * batches)),
            'balanced_updates': int(round(result.get(
                'balanced_updates', 0.0) * batches)),
            'low_effective_gate_mass': float(
                result.get('low_effective_gate_mass', 0.0)),
            'high_effective_gate_mass': float(
                result.get('high_effective_gate_mass', 0.0)),
            'low_geographic_day_blocks': int(
                len(a1s8_low_geographic_day_blocks)),
            'gradient_ratio': ratio,
            'gradient_ratio_in_bounds': bool(
                float(config.get('background_gradient_ratio_min', 0.25))
                <= ratio <= float(config.get(
                    'background_gradient_ratio_max', 4.0))),
            'effective_sample_size': float(result.get('a1s8_ess', 0.0)),
            'anchor_loss': float(result.get('anchor_loss', 0.0)),
            'anchor_weight': float(result.get('anchor_weight', 0.0)),
            'adapter_amplitude': float(result.get(
                'adapter_amplitude', 0.0)),
            'adapter_cap_saturation': float(result.get(
                'adapter_cap_saturation', 0.0)),
        }
    return result['total'], result, diagnostics


def _record_night_v2_execution(config, event, **details):
    if config.get('background_bias_grouping') != _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2:
        return
    path = os.path.join(config['save_dir'], 'night_v2_execution.jsonl')
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'event': event, 'unix_time': time.time(), **details}) + '\n')


def _natural_background_source_loss_components(
        model, batch_processor, batch, source, device, config, epoch,
        sw_manager, iri_peak_manager, allowed_profile_ids, scale, use_amp):
    """Build the natural Background losses without training side effects."""
    coords, target, profile_ids = _unpack_source_batch(batch, device)
    sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
    iri_peak = (iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
    with torch.amp.autocast('cuda', enabled=use_amp):
        pred, _, _, _, extras = _source_forward(
            model, batch_processor, coords, sw_seq, iri_peak, 'background',
            source, profile_ids, False, config, iri_peak_manager,
            allowed_profile_ids, None, 0)
        delta = float(config.get('huber_delta', 0.2))
        observation = profile_huber_loss(pred, target, profile_ids, delta=delta)
        iri = profile_huber_loss(extras['ne_bkg'], extras['ne_iri'],
                                 profile_ids, delta=delta)
        bias, bias_details = background_stratified_bias_loss(
            extras['ne_bkg'], target, coords, profile_ids,
            return_details=True, **_background_bias_kwargs(config, source))
        v2_bias = (str(config.get('background_bias_grouping', ''))
                   == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2)
        shape = observation.new_zeros(())
        shape_details = {'active_cell_count': 0}
        if config.get('background_shape_loss') == _CONDITIONAL_SHAPE_LOSS:
            shape, shape_details = background_conditional_shape_loss(
                extras['ne_bkg'], target, coords, profile_ids, delta=delta,
                min_profiles_per_cell=int(config.get(
                    'background_shape_min_profiles_per_cell', 8)),
                return_details=True)
        vertical, temporal = _structure_losses(
            model, batch_processor, sw_manager, iri_peak_manager, coords,
            profile_ids, 'background', config, allowed_profile_ids)
        low_altitude = observation.new_zeros(())
        low_details = {'profiles': 0, 'points': 0, 'active_fraction': 0.0}
        if _low_altitude_prior_protocol(config) is not None:
            low_altitude, low_details = _low_altitude_prior_loss(
                model, batch_processor, coords, profile_ids, source,
                'background', config, sw_manager, iri_peak_manager,
                allowed_profile_ids)
        weighted_iri = float(config.get('w_iri', 0.02)) * iri
        old_bias_weight = float(config.get(
            'resolved_background_stratum_bias_weight',
            config.get('w_background_stratum_bias', 0.0)))
        magnetic_bias_weight = float(config.get(
            'resolved_background_magnetic_daynight_weight', 0.0) or 0.0)
        height_bias = bias_details.get('_height_loss_tensor', bias)
        magnetic_bias = bias_details.get(
            '_magnetic_loss_tensor', observation.new_zeros(()))
        weighted_bias = old_bias_weight * (height_bias if v2_bias else bias)
        weighted_magnetic_bias = (magnetic_bias_weight * magnetic_bias
                                  if v2_bias else observation.new_zeros(()))
        weighted_shape = float(config.get(
            'resolved_background_shape_weight', 0.0) or 0.0) * shape
        weighted_low = float(config.get(
            'w_low_altitude_background_iri', 0.0)) * low_altitude
        weighted_vertical = float(config.get('w_vertical_background', 0.02)) * vertical
        weighted_time = float(config.get('w_time_background', 0.01)) * temporal
        auxiliary = (weighted_iri + weighted_bias + weighted_magnetic_bias
                     + weighted_shape + weighted_low + weighted_vertical
                     + weighted_time)
        source_loss = observation + auxiliary
        total_loss = float(scale) * source_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f'non-finite natural Background loss for {source}')
    return {
        'coords': coords, 'target': target, 'profile_ids': profile_ids,
        'prediction': pred, 'extras': extras, 'total_loss': total_loss,
        'source_loss': source_loss, 'observation': observation,
        'auxiliary': auxiliary,
        'components': {
            'iri': weighted_iri,
            'background_stratum_bias': weighted_bias,
            'background_magnetic_daynight': weighted_magnetic_bias,
            'background_shape': weighted_shape, 'low_altitude': weighted_low,
            'vertical': weighted_vertical, 'time': weighted_time,
        },
        'details': {
            'bias': bias, 'weighted_bias': weighted_bias,
            'magnetic_bias': magnetic_bias,
            'weighted_magnetic_bias': weighted_magnetic_bias,
            'shape': shape, 'weighted_shape': weighted_shape,
            'vertical': vertical, 'weighted_vertical': weighted_vertical,
            'temporal': temporal, 'weighted_time': weighted_time,
            'iri': iri, 'weighted_iri': weighted_iri,
            'low_altitude': low_altitude, 'weighted_low': weighted_low,
            'bias_details': bias_details, 'shape_details': shape_details,
            'low_details': low_details,
        },
    }


def _natural_background_source_loss(
        model, batch_processor, batch, source, device, config, epoch,
        optimizer, scaler, sw_manager, iri_peak_manager,
        allowed_profile_ids, scale, use_amp):
    """Run one complete-pass source batch with the original A1-S loss."""
    coords, target, profile_ids = _unpack_source_batch(batch, device)
    sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
    iri_peak = (iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
    with torch.amp.autocast('cuda', enabled=use_amp):
        pred, _, _, _, extras = _source_forward(
            model, batch_processor, coords, sw_seq, iri_peak, 'background',
            source, profile_ids, False, config, iri_peak_manager,
            allowed_profile_ids, None, 0)
        delta = float(config.get('huber_delta', 0.2))
        observation = profile_huber_loss(
            pred, target, profile_ids, delta=delta)
        iri = profile_huber_loss(
            extras['ne_bkg'], extras['ne_iri'], profile_ids, delta=delta)
        bias, bias_details = background_stratified_bias_loss(
            extras['ne_bkg'], target, coords, profile_ids,
            return_details=True,
            **_background_bias_kwargs(config, source))
        v2_bias = (str(config.get('background_bias_grouping', ''))
                   == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2)
        shape = observation.new_zeros(())
        shape_details = {'active_cell_count': 0}
        if config.get('background_shape_loss') == _CONDITIONAL_SHAPE_LOSS:
            shape, shape_details = background_conditional_shape_loss(
                extras['ne_bkg'], target, coords, profile_ids,
                delta=delta,
                min_profiles_per_cell=int(config.get(
                    'background_shape_min_profiles_per_cell', 8)),
                return_details=True)
        vertical, temporal = _structure_losses(
            model, batch_processor, sw_manager, iri_peak_manager, coords,
            profile_ids, 'background', config, allowed_profile_ids)
        low_altitude = observation.new_zeros(())
        low_details = {'profiles': 0, 'points': 0, 'active_fraction': 0.0}
        if _low_altitude_prior_protocol(config) is not None:
            low_altitude, low_details = _low_altitude_prior_loss(
                model, batch_processor, coords, profile_ids, source,
                'background', config, sw_manager, iri_peak_manager,
                allowed_profile_ids)
        weighted_iri = float(config.get('w_iri', 0.02)) * iri
        old_bias_weight = float(config.get(
            'resolved_background_stratum_bias_weight',
            config.get('w_background_stratum_bias', 0.0)))
        magnetic_bias_weight = float(config.get(
            'resolved_background_magnetic_daynight_weight', 0.0) or 0.0)
        height_bias = bias_details.get('_height_loss_tensor', bias)
        magnetic_bias = bias_details.get(
            '_magnetic_loss_tensor', observation.new_zeros(()))
        weighted_bias = old_bias_weight * (height_bias if v2_bias else bias)
        weighted_magnetic_bias = (magnetic_bias_weight * magnetic_bias
                                  if v2_bias else observation.new_zeros(()))
        weighted_shape = float(config.get(
            'resolved_background_shape_weight', 0.0) or 0.0) * shape
        weighted_low = float(config.get(
            'w_low_altitude_background_iri', 0.0)) * low_altitude
        weighted_vertical = float(config.get(
            'w_vertical_background', 0.02)) * vertical
        weighted_time = float(config.get(
            'w_time_background', 0.01)) * temporal
        auxiliary = (weighted_iri + weighted_bias + weighted_magnetic_bias
                     + weighted_shape
                     + weighted_low + weighted_vertical + weighted_time)
        source_loss = observation + auxiliary
        total_loss = float(scale) * source_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError(
            f'non-finite natural Background loss for {source}')
    parameters = _current_stage_trainable_parameters(model)
    observation_grad, auxiliary_grad, ratio, component_grads = _gradient_norms(
        observation, auxiliary, parameters, {
            'iri': weighted_iri,
            'background_stratum_bias': weighted_bias,
            'background_magnetic_daynight': weighted_magnetic_bias,
            'background_shape': weighted_shape,
            'low_altitude': weighted_low,
            'vertical': weighted_vertical,
            'time': weighted_time,
        })
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, config.get('grad_clip', 1.0))
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.get('grad_clip', 1.0))
        optimizer.step()
    _record_night_v2_execution(config, 'optimizer_step', epoch=int(epoch + 1), source=source)
    unique_profiles = int(torch.unique(profile_ids).numel())
    return {
        'total': float(total_loss.detach().cpu()),
        'source_loss': float(source_loss.detach().cpu()),
        'observation': float(observation.detach().cpu()),
        'iri': float(iri.detach().cpu()),
        'weighted_iri': float(weighted_iri.detach().cpu()),
        'bias': float(bias.detach().cpu()),
        'weighted_bias': float(weighted_bias.detach().cpu()),
        'magnetic_bias': float(magnetic_bias.detach().cpu()),
        'weighted_magnetic_bias': float(weighted_magnetic_bias.detach().cpu()),
        'shape': float(shape.detach().cpu()),
        'weighted_shape': float(weighted_shape.detach().cpu()),
        'vertical': float(vertical.detach().cpu()),
        'weighted_vertical': float(weighted_vertical.detach().cpu()),
        'time': float(temporal.detach().cpu()),
        'weighted_time': float(weighted_time.detach().cpu()),
        'low_altitude': float(low_altitude.detach().cpu()),
        'weighted_low_altitude': float(weighted_low.detach().cpu()),
        'profiles': unique_profiles,
        'points': int(coords.shape[0]),
        'bias_active_cells': int(bias_details.get('active_cell_count', 0)),
        'bias_magnetic_active_cells': int(
            bias_details.get('magnetic_active_cell_count', 0)),
        'bias_height_daynight_raw': float(
            bias_details.get('height_daynight_loss', 0.0)),
        'bias_magnetic_raw': float(
            bias_details.get('magnetic_loss', 0.0)),
        'magnetic_conditions': bias_details.get('magnetic', {}),
        'bias_combined_raw': float(
            bias_details.get('combined_loss', bias.detach().cpu())),
        'shape_active_cells': int(shape_details.get('active_cell_count', 0)),
        'low_altitude_details': low_details,
        'observation_grad': float(observation_grad.cpu()),
        'auxiliary_grad': float(auxiliary_grad.cpu()),
        'gradient_ratio': float(ratio.cpu()),
        'component_grads': {
            key: float(value.cpu()) for key, value in component_grads.items()},
        'profile_ids': [int(value) for value in torch.unique(
            profile_ids).detach().cpu().tolist()],
    }


def train_one_epoch_natural_background(
        model, train_loader, batch_processor, optimizer, device, config,
        epoch, scaler, sw_manager, iri_peak_manager, cosmic_train_loader,
        allowed_profile_ids=None):
    """Complete FY/COSMIC profile passes with no source recycling."""
    if not _natural_background_enabled(config):
        raise ValueError('natural Background sampler is not enabled')
    if config.get('m00_observation_calibration_semantics', 'disabled') != 'disabled':
        raise ValueError('natural A1-S run must keep M00 calibration disabled')
    fy_batches = int(len(train_loader))
    cosmic_batches = int(len(cosmic_train_loader))
    if fy_batches < 1 or cosmic_batches < 1:
        raise RuntimeError('natural Background requires non-empty FY/COSMIC loaders')
    total_updates = fy_batches + cosmic_batches
    source_profile_totals = {
        'FY': len(_loader_profile_ids(train_loader)),
        'COSMIC': len(_loader_profile_ids(cosmic_train_loader)),
    }
    authorized_profiles = {
        'FY': set(map(int, _loader_profile_ids(train_loader))),
        'COSMIC': set(map(int, _loader_profile_ids(cosmic_train_loader))),
    }
    if min(source_profile_totals.values()) < 1:
        raise RuntimeError('natural Background has no authorized profiles')
    if config.get('background_bias_grouping') == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2:
        if source_profile_totals != {'FY': 44625, 'COSMIC': 56099} or total_updates != 788:
            raise RuntimeError('night-v2 frozen profile/update counts changed before epoch')
    source_iters = {'FY': iter(train_loader), 'COSMIC': iter(cosmic_train_loader)}
    used_batches = {'FY': 0, 'COSMIC': 0}
    seen_profiles = {'FY': [], 'COSMIC': []}
    source_points = {'FY': 0, 'COSMIC': 0}
    source_scale_sums = {'FY': 0.0, 'COSMIC': 0.0}
    source_raw_loss_sums = {'FY': 0.0, 'COSMIC': 0.0}
    source_scaled_loss_sums = {'FY': 0.0, 'COSMIC': 0.0}
    magnetic_conditions = {'FY': {}, 'COSMIC': {}}
    magnetic_fallback_batches = {'FY': 0, 'COSMIC': 0}
    stats = {
        'total': 0.0, 'observation': 0.0, 'iri': 0.0,
        'weighted_iri': 0.0, 'background_stratum_bias': 0.0,
        'weighted_background_stratum_bias': 0.0,
        'background_magnetic_daynight': 0.0,
        'weighted_background_magnetic_daynight': 0.0,
        'background_shape': 0.0, 'weighted_background_shape': 0.0,
        'vertical': 0.0, 'weighted_vertical': 0.0,
        'time': 0.0, 'weighted_time': 0.0,
        'low_altitude': 0.0, 'weighted_low_altitude': 0.0,
        'low_altitude_profiles': 0.0, 'low_altitude_points': 0.0,
        'low_altitude_active_fraction': 0.0,
        'fy_low_altitude': 0.0, 'cosmic_low_altitude': 0.0,
        'fy_low_altitude_profiles': 0.0, 'cosmic_low_altitude_profiles': 0.0,
        'fy_low_altitude_points': 0.0, 'cosmic_low_altitude_points': 0.0,
        'fy_low_altitude_active_fraction': 0.0,
        'cosmic_low_altitude_active_fraction': 0.0,
        'gradient_ratio': 0.0, 'gradient_ratio_low_altitude': 0.0,
        'gradient_ratio_vertical': 0.0, 'gradient_ratio_time': 0.0,
        'gradient_ratio_iri': 0.0,
        'gradient_ratio_background_stratum_bias': 0.0,
        'gradient_ratio_background_magnetic_daynight': 0.0,
        'gradient_ratio_background_shape': 0.0,
        'background_stratum_bias_active_cells': 0.0,
        'background_stratum_bias_magnetic_active_cells': 0.0,
        'background_stratum_bias_height_raw': 0.0,
        'background_stratum_bias_magnetic_raw': 0.0,
        'background_shape_active_cells': 0.0,
        'fy_obs': 0.0, 'cosmic_obs': 0.0,
        'fy_active_fraction': 1.0, 'cosmic_active_fraction': 1.0,
        'gradient_audit_batches': 0,
    }
    diagnostics = []
    started = time.time()
    use_amp = config.get('use_amp', False) and scaler is not None
    max_train_batches = config.get('max_train_batches')
    if max_train_batches is not None:
        raise ValueError('natural Background does not support max_train_batches')
    for batch_idx in range(total_updates):
        available = [source for source, total in (
            ('FY', fy_batches), ('COSMIC', cosmic_batches))
                     if used_batches[source] < total]
        if not available:
            break
        if len(available) == 1:
            source = available[0]
        else:
            fy_progress = used_batches['FY'] / fy_batches
            cosmic_progress = used_batches['COSMIC'] / cosmic_batches
            source = 'FY' if fy_progress <= cosmic_progress else 'COSMIC'
        batch = next(source_iters[source])
        used_batches[source] += 1
        n_profiles = int(torch.unique(batch[2]).numel())
        scale = (float(total_updates) / 2.0
                 * float(n_profiles) / float(source_profile_totals[source]))
        result = _natural_background_source_loss(
            model, batch_processor, batch, source, device, config, epoch,
            optimizer, scaler, sw_manager, iri_peak_manager,
            allowed_profile_ids, scale, use_amp)
        seen_profiles[source].extend(result.pop('profile_ids'))
        source_points[source] += int(result['points'])
        source_scale_sums[source] += float(scale)
        source_raw_loss_sums[source] += float(result['source_loss'])
        source_scaled_loss_sums[source] += float(result['total'])
        magnetic_fallback_batches[source] += int(not result['bias_magnetic_active_cells'])
        for label, condition in result.get('magnetic_conditions', {}).items():
            entry = magnetic_conditions[source].setdefault(label, {
                'support_profile_count': condition['support_profile_count'],
                'active_batches': 0, 'sampled_profile_count': 0, 'bias_sum': 0.0})
            entry['active_batches'] += int(condition['active'])
            count = condition['profile_count']
            entry['sampled_profile_count'] += count
            if count:
                entry['bias_sum'] += condition['mean_bias'] * count
        stats['total'] += result['total']
        stats['observation'] += result['observation']
        stats['iri'] += result['iri']
        stats['weighted_iri'] += result['weighted_iri']
        stats['background_stratum_bias'] += result['bias']
        stats['weighted_background_stratum_bias'] += result['weighted_bias']
        stats['background_magnetic_daynight'] += result.get(
            'magnetic_bias', 0.0)
        stats['weighted_background_magnetic_daynight'] += result.get(
            'weighted_magnetic_bias', 0.0)
        stats['background_shape'] += result['shape']
        stats['weighted_background_shape'] += result['weighted_shape']
        stats['vertical'] += result['vertical']
        stats['weighted_vertical'] += result['weighted_vertical']
        stats['time'] += result['time']
        stats['weighted_time'] += result['weighted_time']
        stats['low_altitude'] += result['low_altitude']
        stats['weighted_low_altitude'] += result['weighted_low_altitude']
        stats['background_stratum_bias_active_cells'] += result['bias_active_cells']
        stats['background_stratum_bias_magnetic_active_cells'] += result[
            'bias_magnetic_active_cells']
        stats['background_stratum_bias_height_raw'] += result[
            'bias_height_daynight_raw']
        stats['background_stratum_bias_magnetic_raw'] += result[
            'bias_magnetic_raw']
        stats['background_shape_active_cells'] += result['shape_active_cells']
        stats['gradient_ratio'] += result['gradient_ratio']
        for key in ('low_altitude', 'vertical', 'time', 'iri',
                    'background_stratum_bias', 'background_shape',
                    'background_magnetic_daynight'):
            stats[f'gradient_ratio_{key}'] += result['component_grads'].get(
                key, 0.0) / max(result['observation_grad'], 1e-12)
        stats['gradient_audit_batches'] += 1
        prefix = source.lower()
        stats[f'{prefix}_obs'] += result['observation']
        low = result['low_altitude_details']
        stats[f'{prefix}_low_altitude'] += result['low_altitude']
        stats[f'{prefix}_low_altitude_profiles'] += low['profiles']
        stats[f'{prefix}_low_altitude_points'] += low['points']
        stats[f'{prefix}_low_altitude_active_fraction'] += low['active_fraction']
        stats['low_altitude_profiles'] += low['profiles']
        stats['low_altitude_points'] += low['points']
        stats['low_altitude_active_fraction'] += low['active_fraction']
        if batch_idx < 100:
            diagnostics.append({
                'epoch': int(epoch + 1), 'batch': int(batch_idx + 1),
                'source': source, 'profiles': n_profiles,
                'points': result['points'], 'source_scale': scale,
                'source_loss': result['source_loss'],
                'weighted_loss': result['total'],
            })
        if (batch_idx + 1) % 100 == 0:
            print(f'  [{batch_idx + 1:>5}/{total_updates}] natural Background '
                  f'{source} loss={result["source_loss"]:.4f} '
                  f'({time.time() - started:.0f}s)')
    if used_batches != {'FY': fy_batches, 'COSMIC': cosmic_batches}:
        raise RuntimeError(
            f'natural Background did not exhaust both loaders: {used_batches}')
    profile_visit_counts = {}
    duplicate_profile_counts = {}
    for source, ids in seen_profiles.items():
        counts = {}
        for profile in ids:
            key = str(profile)
            counts[key] = counts.get(key, 0) + 1
        profile_visit_counts[source] = counts
        duplicate_profile_counts[source] = sum(
            max(0, count - 1) for count in counts.values())
    divisor = float(total_updates)
    for source, cells in magnetic_conditions.items():
        for entry in cells.values():
            entry['total_source_batches'] = used_batches[source]
            entry['active_fraction'] = entry['active_batches'] / used_batches[source]
            entry['sampled_profile_mean_bias'] = (
                entry.pop('bias_sum') / entry['sampled_profile_count']
                if entry['sampled_profile_count'] else None)
    metrics = {key: value / divisor for key, value in stats.items()
               if isinstance(value, (int, float))}
    metrics.update({
        'processed_batches': total_updates,
        'magnetic_condition_activation': magnetic_conditions,
        'magnetic_no_active_group_fallback_fraction': {
            source: magnetic_fallback_batches[source] / used_batches[source]
            for source in used_batches},
        'gradient_audit_batches': int(stats['gradient_audit_batches']),
        'source_mode_counts': {'M10': 0, 'M01': 0, 'M11': 0},
        'natural_epoch_audit': {
            'semantics': 'source_complete_profile_pass_v1',
            'fy_batches': fy_batches, 'cosmic_batches': cosmic_batches,
            'total_updates': total_updates,
            'source_profile_totals': source_profile_totals,
            'sampled_points': source_points,
            'unique_profiles_consumed': {
                source: len(set(ids)) for source, ids in seen_profiles.items()},
            'missing_profile_counts': {
                source: len(authorized_profiles[source] - set(ids))
                for source, ids in seen_profiles.items()},
            'duplicate_profile_counts': {
                source: int(duplicate_profile_counts[source])
                for source in ('FY', 'COSMIC')},
            'profile_visit_counts': profile_visit_counts,
            'source_scale_sums': source_scale_sums,
            'source_raw_loss_means': {
                source: source_raw_loss_sums[source] / max(1, used_batches[source])
                for source in ('FY', 'COSMIC')},
            'source_scaled_loss_sums': source_scaled_loss_sums,
        },
    })
    return metrics['total'], metrics, diagnostics


def _profile_metrics(predictions, targets, profile_ids):
    predictions = np.concatenate(predictions).reshape(-1)
    targets = np.concatenate(targets).reshape(-1)
    profile_ids = np.concatenate(profile_ids).reshape(-1)
    order = np.argsort(profile_ids, kind='stable')
    sorted_ids = profile_ids[order]
    squared = (predictions[order] - targets[order]) ** 2
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_ids)])
    profile_mse = np.add.reduceat(squared, starts) / counts
    pred_std = float(np.std(predictions))
    target_std = float(np.std(targets))
    pearson_r = (float(np.corrcoef(targets, predictions)[0, 1])
                 if min(pred_std, target_std) > 1e-12 else float('nan'))
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    covariance = float(np.mean(
        (predictions - pred_mean) * (targets - target_mean)))
    ccc = float(
        2.0 * covariance
        / (pred_std ** 2 + target_std ** 2
           + (pred_mean - target_mean) ** 2 + 1e-12))
    return {
        'profile_rmse': float(np.sqrt(profile_mse).mean()),
        'mae': float(np.mean(np.abs(predictions - targets))),
        'rmse': float(np.sqrt(np.mean((predictions - targets) ** 2))),
        'r2': float(
            1.0 - np.sum((predictions - targets) ** 2)
            / (np.sum((targets - targets.mean()) ** 2) + 1e-12)),
        'pearson_r': pearson_r,
        'ccc': ccc,
    }


def _background_stratified_metrics(prediction, raw_prediction, target, coords,
                                   residual_raw, residual, trust_gate,
                                   alt_range=(120.0, 500.0),
                                   profile_ids=None):
    """Profile-balanced Background diagnostics by cell and magnetic latitude."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    raw_prediction = np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    residual_raw = np.asarray(residual_raw, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    trust_gate = np.asarray(trust_gate, dtype=np.float64).reshape(-1)
    if profile_ids is not None:
        profile_ids = np.asarray(profile_ids).reshape(-1)
        if len(profile_ids) != len(prediction):
            raise ValueError('Background diagnostic profile IDs have wrong length')
    local_time = np.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    night = (local_time < 6.0) | (local_time >= 18.0)
    finite = (np.isfinite(prediction) & np.isfinite(raw_prediction)
              & np.isfinite(target) & np.isfinite(residual_raw)
              & np.isfinite(residual) & np.isfinite(trust_gate)
              & np.isfinite(coords).all(axis=1))
    if profile_ids is not None:
        finite &= np.isfinite(profile_ids)

    def point_metrics(values, selected):
        selected = selected & finite
        if not selected.any():
            return {
                'n': 0, 'point_count': 0, 'profile_count': 0,
                'rmse': None, 'profile_mean_rmse': None,
                'mean_profile_rmse': None, 'date_sufficient_statistics': [],
                'bias': None, 'mae': None,
                'pearson_r': None, 'ccc': None,
                'prediction_std': None, 'target_std': None,
                'std_ratio': None, 'abs_log_std_ratio': None,
                'spread_error': None, 'centered_rmse': None,
            }
        point_count = int(selected.sum())
        if profile_ids is not None:
            order = np.argsort(profile_ids[selected], kind='stable')
            selected_ids = profile_ids[selected][order]
            starts = np.r_[0, np.flatnonzero(np.diff(selected_ids)) + 1]
            counts = np.diff(np.r_[starts, len(selected_ids)])
            p, y = values[selected][order], target[selected][order]
            prediction_values = np.add.reduceat(p, starts) / counts
            target_values = np.add.reduceat(y, starts) / counts
            profile_rmse = np.sqrt(np.add.reduceat((p - y) ** 2, starts) / counts)
            dates = np.floor(coords[selected, 3][order][starts] / 24).astype(int)
        else:
            prediction_values = values[selected]
            target_values = target[selected]
            counts = np.ones(point_count, dtype=int)
            profile_rmse = np.abs(prediction_values - target_values)
            dates = np.floor(coords[selected, 3] / 24).astype(int)
        error = prediction_values - target_values
        date_statistics = []
        for day in np.unique(dates):
            mask = dates == day
            p, y, e = prediction_values[mask], target_values[mask], error[mask]
            date_statistics.append({
                'relative_utc_day': int(day), 'profile_count': int(mask.sum()),
                'point_count': int(counts[mask].sum()),
                'sum_prediction': float(p.sum()), 'sum_target': float(y.sum()),
                'sum_prediction_sq': float(np.square(p).sum()),
                'sum_target_sq': float(np.square(y).sum()),
                'sum_prediction_target': float((p * y).sum()),
                'sum_error_sq': float(np.square(e).sum()),
                'sum_abs_error': float(np.abs(e).sum()),
                'sum_profile_rmse': float(profile_rmse[mask].sum()),
            })
        profile_count = int(error.size) if profile_ids is not None else point_count
        prediction_std = float(np.std(prediction_values))
        target_std = float(np.std(target_values))
        std_ratio = float((prediction_std + 1e-4) / (target_std + 1e-4))
        abs_log_std_ratio = float(abs(np.log(std_ratio)))
        centered_error = (
            prediction_values - prediction_values.mean()
            - (target_values - target_values.mean()))
        pearson = (
            float(np.corrcoef(prediction_values, target_values)[0, 1])
            if min(prediction_std, target_std) > 1e-12 else None)
        covariance = float(np.mean(
            (prediction_values - prediction_values.mean())
            * (target_values - target_values.mean())))
        ccc = float(
            2.0 * covariance
            / (prediction_std ** 2 + target_std ** 2
               + (prediction_values.mean() - target_values.mean()) ** 2
               + 1e-12))
        return {
            'n': profile_count,
            'point_count': point_count,
            'profile_count': profile_count,
            'rmse': float(np.sqrt(np.mean(error * error))),
            'profile_mean_rmse': float(np.sqrt(np.mean(error * error))),
            'mean_profile_rmse': float(profile_rmse.mean()),
            'date_sufficient_statistics': date_statistics,
            'bias': float(np.mean(error)),
            'mae': float(np.mean(np.abs(error))),
            'pearson_r': pearson,
            'ccc': ccc,
            'prediction_std': prediction_std,
            'target_std': target_std,
            'std_ratio': std_ratio,
            'abs_log_std_ratio': abs_log_std_ratio,
            'spread_error': abs_log_std_ratio,
            'centered_rmse': float(np.sqrt(np.mean(centered_error ** 2))),
        }

    cells = {}
    domain_min, domain_max = map(float, alt_range)
    edges = [domain_min] + [
        edge for edge in (200.0, 300.0)
        if domain_min < edge < domain_max] + [domain_max]
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        label = f'{lower:g}-{upper:g}'
        altitude = (coords[:, 2] >= lower) & (
            coords[:, 2] <= upper if index == len(edges) - 2
            else coords[:, 2] < upper)
        for period, period_mask in (('night', night), ('day', ~night)):
            selected = altitude & period_mask
            raw = point_metrics(raw_prediction, selected)
            m00 = point_metrics(prediction, selected)
            cells[f'{label}_{period}'] = {
                'raw_iri': raw,
                'M00': m00,
                'delta_rmse': (
                    None if raw['rmse'] is None or m00['rmse'] is None
                    else float(m00['rmse'] - raw['rmse'])),
                'delta_bias': (
                    None if raw['bias'] is None or m00['bias'] is None
                    else float(m00['bias'] - raw['bias'])),
                'residual_raw_rms': float(np.sqrt(np.mean(
                    residual_raw[selected & finite] ** 2)))
                if (selected & finite).any() else None,
                'residual_rms': float(np.sqrt(np.mean(
                    residual[selected & finite] ** 2)))
                if (selected & finite).any() else None,
                'trust_gate_mean': float(np.mean(trust_gate[selected & finite]))
                if (selected & finite).any() else None,
            }
    magnetic_cells = {}
    latitude = torch.as_tensor(coords[:, 0], dtype=torch.float64)
    longitude = torch.as_tensor(coords[:, 1], dtype=torch.float64)
    sin_i, _ = _compute_dip_features(latitude, longitude)
    abs_sin_i = np.abs(sin_i.detach().cpu().numpy())
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        label = f'{lower:g}-{upper:g}'
        altitude = (coords[:, 2] >= lower) & (
            coords[:, 2] <= upper if index == len(edges) - 2
            else coords[:, 2] < upper)
        for period, period_mask in (('night', night), ('day', ~night)):
            for magnetic_label, magnetic_mask in (
                    ('abs_sinI_lt_0.25', abs_sin_i < 0.25),
                    ('abs_sinI_0.25_0.5',
                     (abs_sin_i >= 0.25) & (abs_sin_i < 0.5)),
                    ('abs_sinI_ge_0.5', abs_sin_i >= 0.5)):
                selected = altitude & period_mask & magnetic_mask
                magnetic_cells[
                    f'{label}_{period}_{magnetic_label}'] = {
                        'raw_iri': point_metrics(raw_prediction, selected),
                        'M00': point_metrics(prediction, selected),
                    }
    magnetic_region_cells = {}
    magnetic_region_mask = ((coords[:, 2] >= 200.0)
                            & (coords[:, 2] <= 500.0))
    for magnetic_label, magnetic_mask in (
            ('abs_sinI_lt_0.25', abs_sin_i < 0.25),
            ('abs_sinI_0.25_0.5',
             (abs_sin_i >= 0.25) & (abs_sin_i < 0.5)),
            ('abs_sinI_ge_0.5', abs_sin_i >= 0.5)):
        selected = magnetic_region_mask & magnetic_mask
        magnetic_region_cells[magnetic_label] = {
            'raw_iri': point_metrics(raw_prediction, selected),
            'M00': point_metrics(prediction, selected),
        }
    magnetic_region_daynight_cells = {}
    for magnetic_label, magnetic_mask in (
            ('abs_sinI_lt_0.25', abs_sin_i < 0.25),
            ('abs_sinI_0.25_0.5',
             (abs_sin_i >= 0.25) & (abs_sin_i < 0.5)),
            ('abs_sinI_ge_0.5', abs_sin_i >= 0.5)):
        for period, period_mask in (('night', night), ('day', ~night)):
            selected = magnetic_region_mask & period_mask & magnetic_mask
            magnetic_region_daynight_cells[f'{magnetic_label}_{period}'] = {
                'raw_iri': point_metrics(raw_prediction, selected),
                'M00': point_metrics(prediction, selected),
            }
    transition = ((coords[:, 2] >= 200.0) & (coords[:, 2] < 300.0)
                  & night & finite)
    return {
        'cells': cells,
        'magnetic_latitude_cells': magnetic_cells,
        'magnetic_region_cells': magnetic_region_cells,
        'magnetic_region_daynight_cells': magnetic_region_daynight_cells,
        'magnetic_latitude_definition': (
            'dipole_abs_sin_inclination_bins_<0.25_[0.25,0.5)_>=0.5'),
        'trust_gate': {
            'mean': float(np.mean(trust_gate[finite])) if finite.any() else None,
            'p05': float(np.quantile(trust_gate[finite], 0.05)) if finite.any() else None,
            'p50': float(np.quantile(trust_gate[finite], 0.50)) if finite.any() else None,
            'p95': float(np.quantile(trust_gate[finite], 0.95)) if finite.any() else None,
            'strict_suppression_fraction': float(np.mean(
                trust_gate[finite] <= 1e-8)) if finite.any() else None,
        },
        'raw_residual_rms': float(np.sqrt(np.mean(residual_raw[finite] ** 2)))
        if finite.any() else None,
        'gated_residual_rms': float(np.sqrt(np.mean(residual[finite] ** 2)))
        if finite.any() else None,
        'raw_residual_max_abs': float(np.max(np.abs(residual_raw[finite])))
        if finite.any() else None,
        'gated_residual_max_abs': float(np.max(np.abs(residual[finite])))
        if finite.any() else None,
        'transition_raw_residual_rms': float(np.sqrt(np.mean(
            residual_raw[transition] ** 2))) if transition.any() else None,
        'transition_gated_residual_rms': float(np.sqrt(np.mean(
            residual[transition] ** 2))) if transition.any() else None,
    }


@torch.no_grad()
def _evaluate_source(model, loader, batch_processor, device, stage, source,
                     sw_manager, iri_peak_manager, config,
                     allowed_profile_ids=None):
    predictions, targets, profile_ids_all = [], [], []
    raw_predictions = []
    coords_all = []
    raw_residuals = []
    residuals = []
    trust_gates = []
    point_row_indices = []
    magnetic_sin_i_all = []
    aacgm_latitude_all = []
    magnetic_metadata_missing = False
    max_batches = config.get('max_validation_batches')
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        coords, target, profile_ids = _unpack_source_batch(batch, device)
        if stage == 'background' and config.get('m00_point_export_dir'):
            # Dataset index is stable; convert it to the raw row index below.
            point_row_indices.append(np.asarray(batch[1].detach().cpu())
                                     if torch.is_tensor(batch[1])
                                     else np.asarray(batch[1]))
        batch_sin_i, batch_aacgm = _batch_magnetic_values(
            batch, config, source)
        if batch_sin_i is None or batch_aacgm is None:
            magnetic_metadata_missing = True
        else:
            magnetic_sin_i_all.append(
                batch_sin_i.detach().cpu().numpy()
                if torch.is_tensor(batch_sin_i) else np.asarray(batch_sin_i))
            aacgm_latitude_all.append(
                batch_aacgm.detach().cpu().numpy()
                if torch.is_tensor(batch_aacgm) else np.asarray(batch_aacgm))
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
        prediction, _, _, _, extras = _source_forward(
            model, batch_processor, coords, sw_seq, iri_peak,
            stage, source, profile_ids, False, config, iri_peak_manager,
            allowed_profile_ids)
        predictions.append(prediction.cpu().numpy())
        targets.append(target.cpu().numpy())
        profile_ids_all.append(profile_ids.cpu().numpy())
        if stage == 'background' or (
                stage == 'analysis'
                and config.get('analysis_training_scope')
                == _CAUSAL_HISTORY_ANALYSIS_SCOPE):
            raw_predictions.append(extras['ne_iri'].cpu().numpy())
            if stage == 'analysis':
                raw_predictions[-1] = extras['ne_bkg'].cpu().numpy()
            coords_all.append(coords.cpu().numpy())
            if stage == 'background':
                raw_residuals.append(
                    extras['background_residual_raw'].cpu().numpy())
                residuals.append(extras['background_residual'].cpu().numpy())
                trust_gates.append(
                    extras['background_trust_gate'].cpu().numpy())
            else:
                raw_residuals.append(
                    extras['ne_residual'].cpu().numpy())
                residuals.append(extras['ne_residual'].cpu().numpy())
                trust_gates.append(np.ones_like(
                    extras['ne_residual'].cpu().numpy()))
    result = _profile_metrics(predictions, targets, profile_ids_all)
    if stage == 'background' and (
            config.get('m00_calibration_applied', False)
            or (config.get('m00_observation_calibration')
                and config.get('m00_observation_calibration_semantics', 'disabled')
                != 'disabled')):
        calibration = config.get('m00_observation_calibration')
        if not calibration:
            raise RuntimeError('M00 calibration was enabled but not loaded')
        eval_prediction = torch.as_tensor(
            np.concatenate(predictions), device=device)
        eval_target = torch.as_tensor(np.concatenate(targets), device=device)
        eval_coords = torch.as_tensor(np.concatenate(coords_all), device=device)
        eval_ids = torch.as_tensor(np.concatenate(profile_ids_all), device=device)
        eval_sin_i = (
            np.concatenate(magnetic_sin_i_all).reshape(-1)
            if magnetic_sin_i_all and not magnetic_metadata_missing else None)
        eval_aacgm = (
            np.concatenate(aacgm_latitude_all).reshape(-1)
            if aacgm_latitude_all and not magnetic_metadata_missing else None)
        use_qd_strata = str(config.get('magnetic_stratum_semantics', '')).startswith('qd_')
        calibrated_loss, calibrated_details = m00_observation_error_loss(
            eval_prediction, eval_target, eval_coords, eval_ids, source,
            calibration,
            delta=float(config.get('huber_delta', 0.2)),
            magnetic_sin_i=eval_sin_i,
            aacgm_latitude_deg=None if use_qd_strata else eval_aacgm,
            qd_latitude_deg=eval_aacgm if use_qd_strata else None,
            strict_magnetic=bool(config.get('m00_observation_strict_magnetic', False)),
            use_representativeness=bool(config.get(
                'm00_representativeness_enabled', False)),
            use_localization=bool(config.get(
                'm00_localization_enabled', False)),
            weight_cap=float(config.get('m00_observation_weight_cap', 4.0)))
        result['m00_calibrated_loss'] = float(calibrated_loss.detach().cpu())
        result['m00_calibration_details'] = calibrated_details
    if stage == 'background':
        raw = _profile_metrics(raw_predictions, targets, profile_ids_all)
        residual = np.concatenate(predictions).reshape(-1)
        raw_values = np.concatenate(raw_predictions).reshape(-1)
        all_targets = np.concatenate(targets).reshape(-1)
        all_coords = np.concatenate(coords_all)
        all_profile_ids = np.concatenate(profile_ids_all).reshape(-1)
        result.update({
            'raw_iri_profile_rmse': raw['profile_rmse'],
            'raw_iri_rmse': raw['rmse'],
            'raw_iri_mae': raw['mae'],
            'raw_iri_ccc': raw['ccc'],
            'raw_iri_pearson_r': raw['pearson_r'],
            'rmse_change_M00_minus_raw': result['rmse'] - raw['rmse'],
            'profile_rmse_change_M00_minus_raw': (
                result['profile_rmse'] - raw['profile_rmse']),
            'background_residual_rms': float(
                np.sqrt(np.mean((residual - raw_values) ** 2))),
        })
        result['background_stratified'] = _background_stratified_metrics(
            residual, raw_values, all_targets, all_coords,
            np.concatenate(raw_residuals), np.concatenate(residuals),
            np.concatenate(trust_gates), _observation_alt_range(config),
            all_profile_ids)
        export_dir = config.get('m00_point_export_dir')
        if export_dir:
            rows = np.concatenate(point_row_indices).reshape(-1)
            selected = np.asarray(loader.dataset.selected_indices, dtype=np.int64)
            if rows.size != all_coords.shape[0] or np.any((rows < 0) | (rows >= selected.size)):
                raise ValueError('point diagnostic dataset indices are invalid')
            raw_rows = selected[rows]
            # The harmful-correction diagnostic has one deterministic sample
            # per profile, independent of DataLoader ordering.  Keep the
            # lowest raw row indices (the existing shuffle=False convention)
            # and reject duplicate identities only by de-duplicating them in
            # the exported cache.
            sample_limit = max(1, int(config.get('m00_point_sample_size', 8)))
            _, first_positions = np.unique(raw_rows, return_index=True)
            first_positions = np.sort(first_positions)
            order = first_positions[np.lexsort((
                raw_rows[first_positions], all_profile_ids[first_positions]))]
            keep = []
            per_profile = {}
            for position in order:
                profile = int(all_profile_ids[position])
                count = per_profile.get(profile, 0)
                if count < sample_limit:
                    keep.append(int(position))
                    per_profile[profile] = count + 1
            keep = np.asarray(keep, dtype=np.int64)
            keep = keep[np.argsort(raw_rows[keep], kind='stable')]
            duplicate_count = int(rows.size - first_positions.size)
            np.savez_compressed(
                os.path.join(str(export_dir), f'{source.lower()}_{config.get("m00_point_export_split", "development")}.npz'),
                source=np.asarray(source), row_index=raw_rows[keep],
                split=np.asarray(config.get('m00_point_export_split', 'development')),
                profile_id=all_profile_ids[keep].astype(np.int64), coords=all_coords[keep],
                observation=all_targets[keep].astype(np.float64), iri=raw_values[keep].astype(np.float64),
                m00=residual[keep].astype(np.float64),
                raw_residual=(all_targets[keep] - raw_values[keep]).astype(np.float64),
                residual=(residual[keep] - all_targets[keep]).astype(np.float64),
                trust_gate=np.concatenate(trust_gates).reshape(-1)[keep].astype(np.float64),
                sampling_shuffle=np.asarray(False), point_sample_size=np.asarray(sample_limit),
                duplicate_raw_row_count=np.asarray(duplicate_count))
    elif (stage == 'analysis'
          and config.get('analysis_training_scope')
          == _CAUSAL_HISTORY_ANALYSIS_SCOPE):
        baseline = _profile_metrics(raw_predictions, targets, profile_ids_all)
        prediction = np.concatenate(predictions).reshape(-1)
        baseline_values = np.concatenate(raw_predictions).reshape(-1)
        all_targets = np.concatenate(targets).reshape(-1)
        all_profile_ids = np.concatenate(profile_ids_all).reshape(-1)
        anchor = str(config.get(
            'causal_background_anchor', 'a1s2_m00_v1')).replace('-', '_')
        result.update({
            # Source-neutral fields are used by new causal-anchor runs.
            'anchor_m00_profile_rmse': baseline['profile_rmse'],
            'anchor_m00_rmse': baseline['rmse'],
            'anchor_m00_ccc': baseline['ccc'],
            'profile_rmse_improvement_over_anchor_m00': (
                baseline['profile_rmse'] - result['profile_rmse']),
            # Keep the A1-S4 schema readable by existing reports/tests.
            'a1s2_m00_profile_rmse': baseline['profile_rmse'],
            'a1s2_m00_rmse': baseline['rmse'],
            'a1s2_m00_ccc': baseline['ccc'],
            'profile_rmse_improvement_over_a1s2_m00': (
                baseline['profile_rmse'] - result['profile_rmse']),
            'causal_background_anchor': anchor,
        })
        result['causal_stratified'] = _background_stratified_metrics(
            prediction, baseline_values, all_targets,
            np.concatenate(coords_all), np.concatenate(raw_residuals),
            np.concatenate(residuals), np.concatenate(trust_gates),
            _observation_alt_range(config), all_profile_ids)
    return result


@torch.no_grad()
def _development_low_altitude_diagnostics(
        model, loaders, batch_processor, device, stage, config, sw_manager,
        iri_peak_manager, allowed_profile_ids=None):
    """Finite-only v14 drift diagnostic; never used in checkpoint selection."""
    if _low_altitude_prior_protocol(config) is None:
        return None
    _set_stage_mode(model, stage)
    result = {}
    for source, loader in loaders.items():
        try:
            batch = next(iter(loader))
        except StopIteration:
            raise ValueError(f'{source} development loader is empty')
        coords, _, profile_ids = _unpack_source_batch(batch, device)
        loss, details = _low_altitude_prior_loss(
            model, batch_processor, coords, profile_ids, source, stage, config,
            sw_manager, iri_peak_manager, allowed_profile_ids)
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise FloatingPointError(
                f'non-finite {source} development low-altitude anchor diagnostic')
        result[source] = {'raw_loss': value, **details}
    result['equal_source_raw_loss'] = float(np.mean([
        result['FY']['raw_loss'], result['COSMIC']['raw_loss']]))
    return result


def validate(model, val_loader, batch_processor, device, config,
             sw_manager, iri_peak_manager, cosmic_val_loader, stage='analysis',
             allowed_profile_ids=None):
    model.eval()
    fy = _evaluate_source(
        model, val_loader, batch_processor, device, stage, 'FY',
        sw_manager, iri_peak_manager, config, allowed_profile_ids)
    cosmic = _evaluate_source(
        model, cosmic_val_loader, batch_processor, device, stage, 'COSMIC',
        sw_manager, iri_peak_manager, config, allowed_profile_ids)
    score = 0.5 * (fy['profile_rmse'] + cosmic['profile_rmse'])
    metrics = {
        'score': score,
        'fy_profile_rmse': fy['profile_rmse'],
        'cosmic_profile_rmse': cosmic['profile_rmse'],
        'mae': 0.5 * (fy['mae'] + cosmic['mae']),
        'rmse': 0.5 * (fy['rmse'] + cosmic['rmse']),
        'r2': 0.5 * (fy['r2'] + cosmic['r2']),
        'fy_ccc': fy['ccc'],
        'cosmic_ccc': cosmic['ccc'],
        'ccc': 0.5 * (fy['ccc'] + cosmic['ccc']),
        'fy_pearson_r': fy['pearson_r'],
        'cosmic_pearson_r': cosmic['pearson_r'],
        'pearson_r': 0.5 * (fy['pearson_r'] + cosmic['pearson_r']),
    }
    if stage == 'background':
        fy_cells = fy['background_stratified']['cells']
        cosmic_cells = cosmic['background_stratified']['cells']
        conflicts = {}
        for cell in sorted(set(fy_cells) & set(cosmic_cells)):
            fy_bias = fy_cells[cell]['M00'].get('bias')
            cosmic_bias = cosmic_cells[cell]['M00'].get('bias')
            conflicts[cell] = bool(
                fy_bias is not None and cosmic_bias is not None
                and np.isfinite(fy_bias) and np.isfinite(cosmic_bias)
                and abs(fy_bias) >= 0.02 and abs(cosmic_bias) >= 0.02
                and fy_bias * cosmic_bias < 0.0)
        metrics.update({
            'fy_raw_iri_profile_rmse': fy['raw_iri_profile_rmse'],
            'cosmic_raw_iri_profile_rmse': cosmic['raw_iri_profile_rmse'],
            'fy_raw_iri_rmse': fy['raw_iri_rmse'],
            'cosmic_raw_iri_rmse': cosmic['raw_iri_rmse'],
            'fy_raw_iri_ccc': fy['raw_iri_ccc'],
            'cosmic_raw_iri_ccc': cosmic['raw_iri_ccc'],
            'fy_raw_iri_pearson_r': fy['raw_iri_pearson_r'],
            'cosmic_raw_iri_pearson_r': cosmic['raw_iri_pearson_r'],
            'fy_rmse_change_M00_minus_raw': fy[
                'rmse_change_M00_minus_raw'],
            'cosmic_rmse_change_M00_minus_raw': cosmic[
                'rmse_change_M00_minus_raw'],
            'fy_profile_rmse_change_M00_minus_raw': fy[
                'profile_rmse_change_M00_minus_raw'],
            'cosmic_profile_rmse_change_M00_minus_raw': cosmic[
                'profile_rmse_change_M00_minus_raw'],
            'fy_background_residual_rms': fy['background_residual_rms'],
            'cosmic_background_residual_rms': cosmic[
                'background_residual_rms'],
            'fy_background_stratified': fy['background_stratified'],
            'cosmic_background_stratified': cosmic['background_stratified'],
            'background_source_conflict': {
                'cells': conflicts,
                'any': bool(any(conflicts.values())),
                'criterion': (
                    'opposite FY/COSMIC M00 Bias with both absolute Bias >= 0.02 dex'),
            },
        })
    elif (config.get('analysis_training_scope')
          == _CAUSAL_HISTORY_ANALYSIS_SCOPE):
        metrics.update({
            'causal_background_anchor': config.get(
                'causal_background_anchor', 'a1s2_m00_v1'),
            'fy_anchor_m00_profile_rmse': fy['anchor_m00_profile_rmse'],
            'cosmic_anchor_m00_profile_rmse': cosmic[
                'anchor_m00_profile_rmse'],
            'fy_anchor_m00_ccc': fy['anchor_m00_ccc'],
            'cosmic_anchor_m00_ccc': cosmic['anchor_m00_ccc'],
            # Backward-compatible A1-S4 names.
            'fy_a1s2_m00_profile_rmse': fy['a1s2_m00_profile_rmse'],
            'cosmic_a1s2_m00_profile_rmse': cosmic[
                'a1s2_m00_profile_rmse'],
            'fy_a1s2_m00_ccc': fy['a1s2_m00_ccc'],
            'cosmic_a1s2_m00_ccc': cosmic['a1s2_m00_ccc'],
            'fy_causal_stratified': fy['causal_stratified'],
            'cosmic_causal_stratified': cosmic['causal_stratified'],
        })
    return score, metrics


_BACKGROUND_SELECTION_CELLS = (
    '200-300_night', '200-300_day',
    '300-500_night', '300-500_day',
)
_BACKGROUND_SELECTION_MAGNETIC_LABELS = (
    'abs_sinI_lt_0.25', 'abs_sinI_0.25_0.5', 'abs_sinI_ge_0.5',
)


def _night_generalization_cells(metrics, label):
    """Extract the fixed 24 source/height/daynight/magnetic cells."""
    if not isinstance(metrics, dict):
        raise ValueError(f'{label} metrics must be an object')
    values = {}
    for source in ('fy', 'cosmic'):
        try:
            cells = metrics[f'{source}_background_stratified'][
                'magnetic_latitude_cells']
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f'{label} lacks {source}_background_stratified magnetic cells') from exc
        for altitude in ('200-300', '300-500'):
            for period in ('night', 'day'):
                for magnetic in _BACKGROUND_SELECTION_MAGNETIC_LABELS:
                    name = f'{altitude}_{period}_{magnetic}'
                    try:
                        item = cells[name]['M00']
                        row = {
                            key: float(item[key])
                            for key in ('bias', 'rmse', 'ccc', 'profile_count')}
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f'{label} lacks finite {source} {name}') from exc
                    if not np.isfinite(list(row.values())).all():
                        raise ValueError(f'{label} has non-finite {source} {name}')
                    row['profile_count'] = int(row['profile_count'])
                    values[(source, name)] = row
    if len(values) != 24:
        raise ValueError(f'{label} must provide exactly 24 night-generalization cells')
    return values


def _night_generalization_selection_audit(metrics, baseline_metrics, epoch):
    """Fixed direct-control Development gate for the one-run night experiment."""
    epsilon = 1e-8
    candidate = _night_generalization_cells(metrics, 'candidate')
    baseline = _night_generalization_cells(baseline_metrics, 'control')
    reasons = []
    supported = {
        key for key, row in baseline.items() if row['profile_count'] >= 30}
    if len(supported) != 24:
        reasons.append(f'unsupported_control_cells:{24 - len(supported)}')
    if any(candidate[key]['profile_count'] < 30 for key in supported):
        reasons.append('candidate_cell_support_below_30')
    finite_cells = [key for key in supported
                    if np.isfinite(list(candidate[key].values())).all()]
    if len(finite_cells) != len(supported):
        reasons.append('non_finite_candidate_cells')

    def macro(keys, metric):
        return float(np.mean([candidate[key][metric] for key in keys]))

    night_keys = [key for key in supported if '_night_' in key[1]]
    day_keys = [key for key in supported if '_day_' in key[1]]
    low_night = [key for key in night_keys if key[1].endswith(
        'abs_sinI_lt_0.25')]
    high_night = [key for key in night_keys if key[1].endswith(
        'abs_sinI_ge_0.5')]
    if not night_keys or not day_keys or not low_night or not high_night:
        reasons.append('missing_fixed_night_or_day_cells')

    def delta(keys, metric):
        return float(np.mean([candidate[key][metric] - baseline[key][metric]
                              for key in keys]))

    night_rmse_delta = delta(night_keys, 'rmse') if night_keys else float('nan')
    night_ccc_delta = delta(night_keys, 'ccc') if night_keys else float('nan')
    day_rmse_delta = delta(day_keys, 'rmse') if day_keys else float('nan')
    day_ccc_delta = delta(day_keys, 'ccc') if day_keys else float('nan')
    low_abs_bias_delta = (
        float(np.mean([abs(candidate[key]['bias']) - abs(baseline[key]['bias'])
                       for key in low_night])) if low_night else float('nan'))
    high_rmse_delta = delta(high_night, 'rmse') if high_night else float('nan')
    high_ccc_delta = delta(high_night, 'ccc') if high_night else float('nan')
    if night_keys:
        base_night_rmse = float(np.mean([baseline[key]['rmse'] for key in night_keys]))
        if not macro(night_keys, 'rmse') <= base_night_rmse * (1.0 - 0.001) + epsilon:
            reasons.append('night_macro_rmse_improvement_below_0.1pct')
        if night_ccc_delta < 0.005 - epsilon:
            reasons.append('night_macro_ccc_improvement_below_0.005')
    if not low_night or low_abs_bias_delta > -0.01 + epsilon:
        reasons.append('low_magnetic_night_abs_bias_not_reduced_0.01')
    if high_night and (high_rmse_delta > epsilon or high_ccc_delta < -epsilon):
        reasons.append('high_magnetic_night_degradation')
    if day_keys and day_rmse_delta > 0.005 * float(np.mean(
            [baseline[key]['rmse'] for key in day_keys])) + epsilon:
        reasons.append('day_macro_rmse_degradation_over_0.5pct')
    if day_keys and day_ccc_delta < -0.005 - epsilon:
        reasons.append('day_macro_ccc_degradation_over_0.005')
    source_overall = {}
    for source in ('fy', 'cosmic'):
        keys = [key for key in supported if key[0] == source]
        c_rmse = float(np.mean([candidate[key]['rmse'] for key in keys]))
        b_rmse = float(np.mean([baseline[key]['rmse'] for key in keys]))
        c_ccc = float(np.mean([candidate[key]['ccc'] for key in keys]))
        b_ccc = float(np.mean([baseline[key]['ccc'] for key in keys]))
        source_overall[source] = {
            'candidate_rmse': c_rmse, 'baseline_rmse': b_rmse,
            'candidate_ccc': c_ccc, 'baseline_ccc': b_ccc,
        }
        if c_rmse > b_rmse * 1.005 + epsilon:
            reasons.append(f'{source}_rmse_degradation_over_0.5pct')
        if c_ccc < b_ccc - 0.005 - epsilon:
            reasons.append(f'{source}_ccc_degradation_over_0.005')
    cell_worsening = {}
    for key in supported:
        abs_bias_delta = abs(candidate[key]['bias']) - abs(baseline[key]['bias'])
        if abs_bias_delta > 0.02 + epsilon:
            cell_worsening[f'{key[0]}:{key[1]}'] = abs_bias_delta
    if cell_worsening:
        reasons.append('cell_abs_bias_worsening_over_0.02')
    finite_key = all(np.isfinite(value) for value in (
        night_rmse_delta, night_ccc_delta, low_abs_bias_delta,
        high_rmse_delta, high_ccc_delta))
    if not finite_key:
        reasons.append('non_finite_selection_key')
    return {
        'semantics': _M00_NIGHT_GENERALIZATION_SELECTOR,
        'epoch': int(epoch),
        'eligible': not reasons,
        'reasons': list(dict.fromkeys(reasons)),
        'supported_cell_count': len(supported),
        'night_macro_rmse_delta': night_rmse_delta,
        'night_macro_ccc_delta': night_ccc_delta,
        'day_macro_rmse_delta': day_rmse_delta,
        'day_macro_ccc_delta': day_ccc_delta,
        'low_magnetic_night_abs_bias_delta': low_abs_bias_delta,
        'high_magnetic_night_rmse_delta': high_rmse_delta,
        'high_magnetic_night_ccc_delta': high_ccc_delta,
        'source_overall': source_overall,
        'cell_abs_bias_worsening': cell_worsening,
        'selection_key': ([
            -night_rmse_delta, night_ccc_delta, -int(epoch)]
            if finite_key and not reasons else None),
    }


def night_generalization_v2_audit(metrics, baseline_metrics, epoch, supported_cells):
    """The frozen, shared night-v2 gate; malformed candidates fail closed."""
    epsilon = 1e-8
    audit = {'semantics': _M00_NIGHT_GENERALIZATION_V2_SELECTOR,
             'epoch': int(epoch), 'eligible': False, 'reasons': [],
             'night_macro_rmse': None, 'night_macro_ccc': None,
             'selection_key': None, 'cell_abs_bias_deltas': {},
             'max_cell_abs_bias_worsening': None, 'source_overall': {}}
    reasons = audit['reasons']
    candidate, baseline = {}, {}
    try:
        if not supported_cells or len(set(supported_cells)) != len(supported_cells):
            raise ValueError('empty_or_duplicate_frozen_support')
        allowed = {f'{source}:{height}_{period}_{mag}'
                   for source in ('fy', 'cosmic')
                   for height in ('200-300', '300-500')
                   for period in ('night', 'day')
                   for mag in _BACKGROUND_SELECTION_MAGNETIC_LABELS}
        if not set(supported_cells) <= allowed:
            raise ValueError('invalid_frozen_support')
        for name in supported_cells:
            source, cell = name.split(':', 1)
            for label, raw, destination in (
                    ('candidate', metrics, candidate),
                    ('control', baseline_metrics, baseline)):
                item = raw[f'{source}_background_stratified'][
                    'magnetic_latitude_cells'][cell]['M00']
                row = {key: float(item[key]) for key in
                       ('rmse', 'ccc', 'bias', 'profile_count')}
                if 'point_count' in item:
                    row['point_count'] = float(item['point_count'])
                if (not np.isfinite(list(row.values())).all()
                        or row['profile_count'] < 30
                        or row['profile_count'] != int(row['profile_count'])
                        or ('point_count' in row and (
                            row['point_count'] < row['profile_count']
                            or row['point_count'] != int(row['point_count'])))):
                    raise ValueError(f'{label}_invalid_cell:{name}')
                destination[name] = row
            if candidate[name]['profile_count'] != baseline[name]['profile_count']:
                reasons.append(f'candidate_count_changed:{name}')
            if candidate[name].get('point_count') != baseline[name].get('point_count'):
                reasons.append(f'candidate_point_count_changed:{name}')
        night = [key for key in supported_cells if '_night_' in key]
        day = [key for key in supported_cells if '_day_' in key]
        low = [key for key in night if key.endswith('abs_sinI_lt_0.25')]
        high = [key for key in night if key.endswith('abs_sinI_ge_0.5')]
        if not day or any(not any(f':{height}_night_' in key for key in group)
                          for group in (low, high)
                          for height in ('200-300', '300-500')):
            raise ValueError('insufficient_frozen_night_day_coverage')
        def mean(rows, keys, metric, absolute=False):
            return float(np.mean([abs(rows[key][metric]) if absolute
                                  else rows[key][metric] for key in keys]))
        def check(label, keys, metric, maximum_delta, absolute=False):
            before = mean(baseline, keys, metric, absolute)
            after = mean(candidate, keys, metric, absolute)
            delta = after - before
            audit[label] = {'control': before, 'candidate': after, 'delta': delta}
            if delta > maximum_delta(before) + epsilon:
                reasons.append(label)
        check('night_macro_rmse_gate', night, 'rmse', lambda b: -0.001 * b)
        # Negating the delta turns lower-bound CCC gates into the same tolerance rule.
        for label, keys, minimum in (('night_macro_ccc_gate', night, .005),
                                     ('high_night_ccc_gate', high, 0.),
                                     ('day_macro_ccc_gate', day, -.005)):
            before, after = mean(baseline, keys, 'ccc'), mean(candidate, keys, 'ccc')
            audit[label] = {'control': before, 'candidate': after, 'delta': after - before}
            if after - before < minimum - epsilon:
                reasons.append(label)
        check('low_night_abs_bias_gate', low, 'bias', lambda b: -.01, True)
        check('high_night_rmse_gate', high, 'rmse', lambda b: 0.)
        check('day_macro_rmse_gate', day, 'rmse', lambda b: .005 * b)
        audit['night_macro_rmse'] = mean(candidate, night, 'rmse')
        audit['night_macro_ccc'] = mean(candidate, night, 'ccc')
        for source in ('fy', 'cosmic'):
            before = float(baseline_metrics[f'{source}_profile_rmse'])
            after = float(metrics[f'{source}_profile_rmse'])
            if not np.isfinite([before, after]).all():
                raise ValueError(f'non_finite_{source}_mean_profile_rmse')
            audit['source_overall'][source] = {
                'control_mean_profile_rmse': before, 'candidate_mean_profile_rmse': after}
            if after > before * 1.005 + epsilon:
                reasons.append(f'{source}_mean_profile_rmse_gate')
        deltas = {key: abs(candidate[key]['bias']) - abs(baseline[key]['bias'])
                  for key in supported_cells}
        audit['cell_abs_bias_deltas'] = deltas
        audit['max_cell_abs_bias_worsening'] = max(deltas.values())
        if max(deltas.values()) > .02 + epsilon:
            reasons.append('cell_abs_bias_worsening_over_0.02')
        audit['supported_cell_count'] = len(supported_cells)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        reasons.append(f'invalid_required_metrics:{exc}')
    audit['eligible'] = not reasons
    if audit['eligible']:
        audit['selection_key'] = [-audit['night_macro_rmse'],
                                  audit['night_macro_ccc'], -int(epoch)]
    return audit


def night_generalization_v2_better(candidate_audit, current_audit):
    """Tolerance-aware ordering, including diagnostic-only finite candidates."""
    if any(candidate_audit.get(key) is None or not np.isfinite(candidate_audit[key])
           for key in ('night_macro_rmse', 'night_macro_ccc')):
        return False
    if current_audit is None:
        return True
    for key, direction in (('night_macro_rmse', -1), ('night_macro_ccc', 1)):
        difference = candidate_audit[key] - current_audit[key]
        if abs(difference) > 1e-8:
            return difference * direction > 0
    return candidate_audit['epoch'] < current_audit['epoch']


def select_night_generalization_v2(rows, baseline_metrics, supported_cells):
    result = {'eligible': False, 'selected_row': None, 'selected_audit': None,
              'diagnostic_row': None, 'diagnostic_audit': None, 'entries': []}
    for row in rows:
        if row.get('stage', 'background') != 'background':
            continue
        audit = night_generalization_v2_audit(
            row, baseline_metrics, row['epoch'], supported_cells)
        result['entries'].append({'row': row, 'audit': audit, 'eligible': audit['eligible']})
        if night_generalization_v2_better(audit, result['diagnostic_audit']):
            result.update(diagnostic_row=row, diagnostic_audit=audit)
        if audit['eligible'] and night_generalization_v2_better(audit, result['selected_audit']):
            result.update(eligible=True, selected_row=row, selected_audit=audit)
    result['selected_epoch'] = (result['selected_row']['epoch'] if result['eligible'] else None)
    result['diagnostic_best_epoch'] = (result['diagnostic_row']['epoch']
                                       if result['diagnostic_row'] else None)
    result['epoch_audits'] = [entry['audit'] for entry in result['entries']]
    result['reason'] = 'eligible_epoch_selected' if result['eligible'] else 'no_eligible_epoch'
    return result


def _causal_history_selection_audit(metrics):
    """Apply the fixed causal-history Development gate to its M00 anchor."""
    epsilon = 1e-12
    reasons = []
    source_overall = {}
    for source in ('fy', 'cosmic'):
        candidate_rmse = float(metrics[f'{source}_profile_rmse'])
        baseline_rmse_key = f'{source}_anchor_m00_profile_rmse'
        baseline_rmse = float(
            metrics[baseline_rmse_key] if baseline_rmse_key in metrics
            else metrics[f'{source}_a1s2_m00_profile_rmse'])
        candidate_ccc = float(metrics[f'{source}_ccc'])
        baseline_ccc_key = f'{source}_anchor_m00_ccc'
        baseline_ccc = float(
            metrics[baseline_ccc_key] if baseline_ccc_key in metrics
            else metrics[f'{source}_a1s2_m00_ccc'])
        values = np.asarray(
            [candidate_rmse, baseline_rmse, candidate_ccc, baseline_ccc],
            dtype=np.float64)
        if not np.isfinite(values).all():
            reasons.append(f'{source}_non_finite_overall')
        if candidate_rmse > baseline_rmse + epsilon:
            reasons.append(f'{source}_profile_rmse_degradation')
        if candidate_ccc < baseline_ccc - epsilon:
            reasons.append(f'{source}_ccc_degradation')
        source_overall[source] = {
            'candidate_profile_rmse': candidate_rmse,
            'baseline_profile_rmse': baseline_rmse,
            'candidate_ccc': candidate_ccc,
            'baseline_ccc': baseline_ccc,
        }
    mean_improvement = float(np.mean([
        row['baseline_profile_rmse'] - row['candidate_profile_rmse']
        for row in source_overall.values()]))
    if mean_improvement < 0.005 - epsilon:
        reasons.append('insufficient_mean_profile_rmse_improvement')

    night_pairs, band_pairs, cell_worsening = [], {label: [] for label in (
        'abs_sinI_lt_0.25', 'abs_sinI_ge_0.5')}, {}
    expected_cells = 0
    for source in ('fy', 'cosmic'):
        cells = metrics[f'{source}_causal_stratified'][
            'magnetic_latitude_cells']
        for name, payload in cells.items():
            if not (name.startswith(('200-300_', '300-500_'))):
                continue
            candidate = payload['M00']
            baseline = payload['raw_iri']
            values = np.asarray([
                candidate.get('bias'), candidate.get('rmse'),
                candidate.get('ccc'), baseline.get('bias'),
                baseline.get('rmse'), baseline.get('ccc')], dtype=np.float64)
            if not np.isfinite(values).all():
                reasons.append(f'non_finite_cell:{source}:{name}')
                continue
            expected_cells += 1
            if abs(values[0]) > abs(values[3]) + 0.01 + epsilon:
                cell_worsening[f'{source}:{name}:abs_bias'] = float(
                    abs(values[0]) - abs(values[3]))
            if values[1] > values[4] + 0.01 + epsilon:
                cell_worsening[f'{source}:{name}:rmse'] = float(
                    values[1] - values[4])
            if values[2] < values[5] - 0.02 - epsilon:
                cell_worsening[f'{source}:{name}:ccc'] = float(
                    values[2] - values[5])
            if '_night_' in name:
                pair = (float(values[2]), float(values[5]),
                        float(values[1]), float(values[4]))
                night_pairs.append(pair)
                for label in band_pairs:
                    if name.endswith(label):
                        band_pairs[label].append(pair)
    if expected_cells != 24:
        reasons.append(f'expected_24_finite_cells_got_{expected_cells}')
    if len(night_pairs) != 12:
        reasons.append(f'expected_12_finite_night_cells_got_{len(night_pairs)}')
    if night_pairs:
        night = np.asarray(night_pairs, dtype=np.float64)
        night_ccc_delta = float(np.mean(night[:, 0] - night[:, 1]))
        night_candidate_rmse = float(np.mean(night[:, 2]))
        night_baseline_rmse = float(np.mean(night[:, 3]))
        if night_ccc_delta < 0.005 - epsilon:
            reasons.append('insufficient_night_macro_ccc_improvement')
        if not night_candidate_rmse < night_baseline_rmse - epsilon:
            reasons.append('night_macro_rmse_not_improved')
    else:
        night_ccc_delta = float('-inf')
        night_candidate_rmse = float('inf')
        night_baseline_rmse = float('inf')
    for label, pairs in band_pairs.items():
        if not pairs:
            reasons.append(f'missing_night_band:{label}')
        elif np.mean([item[0] - item[1] for item in pairs]) < -epsilon:
            reasons.append(f'night_band_ccc_degradation:{label}')
    if cell_worsening:
        reasons.append('source_cell_degradation')
    key = [
        night_ccc_delta,
        -night_candidate_rmse,
        -float(metrics['score']),
        float(metrics['ccc']),
    ]
    finite_key = np.isfinite(key).all()
    if not finite_key:
        reasons.append('non_finite_selection_key')
    return {
        'eligible': not reasons,
        'reasons': reasons,
        'source_overall': source_overall,
        'mean_profile_rmse_improvement': mean_improvement,
        'night_macro_ccc_improvement': night_ccc_delta,
        'night_macro_candidate_rmse': night_candidate_rmse,
        'night_macro_baseline_rmse': night_baseline_rmse,
        'source_cell_worsening': cell_worsening,
        'selection_key': key if finite_key and not reasons else None,
    }


def _background_selection_values(metrics, label):
    """Validate and extract the fixed bias-constrained-v1 metric schema."""
    if not isinstance(metrics, dict):
        raise ValueError(f'{label} metrics must be an object')
    scalars = {}
    for key in ('score', 'ccc', 'rmse', 'pearson_r'):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{label} metrics lacks finite {key}') from exc
        if not np.isfinite(value):
            raise ValueError(f'{label} metrics lacks finite {key}')
        scalars[key] = value

    cells, magnetic = {}, {}
    for source in ('fy', 'cosmic'):
        source_key = f'{source}_background_stratified'
        try:
            source_metrics = metrics[source_key]
            source_cells = source_metrics['cells']
            source_magnetic = source_metrics['magnetic_latitude_cells']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'{label} metrics lacks {source_key} schema') from exc
        for cell in _BACKGROUND_SELECTION_CELLS:
            try:
                item = source_cells[cell]['M00']
                bias = float(item['bias'])
                count = int(item['profile_count'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f'{label} metrics lacks {source.upper()} {cell} M00 Bias') from exc
            if not np.isfinite(bias) or count < 0:
                raise ValueError(
                    f'{label} metrics has invalid {source.upper()} {cell} M00 Bias')
            cells[(source, cell)] = {'bias': bias, 'profile_count': count}
        for cell in _BACKGROUND_SELECTION_CELLS:
            for magnetic_label in _BACKGROUND_SELECTION_MAGNETIC_LABELS:
                key = f'{cell}_{magnetic_label}'
                try:
                    item = source_magnetic[key]['M00']
                    bias = float(item['bias'])
                    count = int(item['profile_count'])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f'{label} metrics lacks {source.upper()} {key} M00 Bias') from exc
                if not np.isfinite(bias) or count < 0:
                    raise ValueError(
                        f'{label} metrics has invalid {source.upper()} {key} M00 Bias')
                magnetic[(source, key)] = {
                    'bias': bias, 'profile_count': count}
    return {'scalars': scalars, 'cells': cells, 'magnetic': magnetic}


def _m00_natural_scalar_values(metrics, label):
    """Read the common M00 Development scalars used by train and reports."""
    if not isinstance(metrics, dict):
        raise ValueError(f'{label} metrics must be an object')
    values = {}
    try:
        values['fy_profile_rmse'] = float(metrics['fy_profile_rmse'])
        values['cosmic_profile_rmse'] = float(metrics['cosmic_profile_rmse'])
        values['fy_ccc'] = float(metrics.get('fy_ccc', metrics['ccc']))
        values['cosmic_ccc'] = float(metrics.get('cosmic_ccc', metrics['ccc']))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{label} metrics lacks M00 Development scalars') from exc
    values['profile_rmse'] = float(np.mean([
        values['fy_profile_rmse'], values['cosmic_profile_rmse']]))
    values['mean_ccc'] = float(np.mean([
        values['fy_ccc'], values['cosmic_ccc']]))
    if (not np.isfinite(list(values.values())).all()
            or values['profile_rmse'] <= 0.0):
        raise ValueError(f'{label} metrics has non-finite M00 Development scalars')
    return values


def _m00_natural_cell_values(metrics, label):
    """Read the eight height x local-time M00 cells used for protection."""
    values = {}
    if not isinstance(metrics, dict):
        raise ValueError(f'{label} metrics must be an object')
    for source in ('fy', 'cosmic'):
        try:
            cells = metrics[f'{source}_background_stratified']['cells']
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f'{label} metrics lacks {source.upper()} Background cells') from exc
        for cell in _BACKGROUND_SELECTION_CELLS:
            try:
                item = cells[cell]['M00']
                bias = float(item['bias'])
                profiles = int(item.get('profile_count', item.get('n', 0)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f'{label} metrics lacks {source.upper()} {cell} M00 cell') from exc
            values[(source, cell)] = {'bias': bias, 'profile_count': profiles}
    return values


def m00_natural_selection_audit(metrics, baseline_metrics, epoch):
    """Apply the shared natural M00 protection gates for one epoch."""
    thresholds = {
        'source_rmse_relative_degradation_max': 0.01,
        'mean_ccc_degradation_max': 0.005,
        'cell_abs_bias_degradation_max': 0.02,
        'min_cell_profiles': 30,
    }
    reasons = []
    candidate = baseline = None
    candidate_cells = baseline_cells = None
    try:
        candidate = _m00_natural_scalar_values(metrics, 'candidate')
        candidate_cells = _m00_natural_cell_values(metrics, 'candidate')
    except ValueError as exc:
        reasons.append('candidate_schema_or_finite')
        candidate_error = str(exc)
    else:
        candidate_error = None
    try:
        baseline = _m00_natural_scalar_values(baseline_metrics, 'historical')
        baseline_cells = _m00_natural_cell_values(baseline_metrics, 'historical')
    except ValueError as exc:
        reasons.append('historical_schema_or_finite')
        baseline_error = str(exc)
    else:
        baseline_error = None

    audit = {
        'semantics': _M00_NATURAL_SELECTOR,
        'epoch': int(epoch),
        'eligible': False,
        'rejection_reasons': reasons,
        'thresholds': thresholds,
        'selection_key': None,
        'candidate_error': candidate_error,
        'historical_error': baseline_error,
    }
    if candidate is None or baseline is None:
        return audit

    rmse_relative = {
        source.upper(): float(
            (candidate[f'{source}_profile_rmse']
             - baseline[f'{source}_profile_rmse'])
            / baseline[f'{source}_profile_rmse'])
        for source in ('fy', 'cosmic')
    }
    ccc_delta = float(candidate['mean_ccc'] - baseline['mean_ccc'])
    audit.update({
        'candidate': candidate,
        'historical': baseline,
        'source_rmse_relative_deltas': rmse_relative,
        'mean_ccc_delta': ccc_delta,
    })
    if any(value > thresholds['source_rmse_relative_degradation_max']
           for value in rmse_relative.values()):
        reasons.append('source_rmse_protection_failed')
    if ccc_delta < -thresholds['mean_ccc_degradation_max']:
        reasons.append('ccc_protection_failed')

    cell_worsening = {}
    if candidate_cells is not None and baseline_cells is not None:
        for key in sorted(baseline_cells):
            candidate_cell = candidate_cells.get(key)
            baseline_cell = baseline_cells[key]
            if (candidate_cell is None
                    or candidate_cell['profile_count'] < thresholds['min_cell_profiles']
                    or baseline_cell['profile_count'] < thresholds['min_cell_profiles']):
                reasons.append(
                    f'insufficient_cell_profiles:{key[0].upper()}:{key[1]}')
                continue
            candidate_bias = candidate_cell['bias']
            baseline_bias = baseline_cell['bias']
            if not np.isfinite([candidate_bias, baseline_bias]).all():
                reasons.append(f'nonfinite_cell:{key[0].upper()}:{key[1]}')
                continue
            worsening = float(abs(candidate_bias) - abs(baseline_bias))
            if worsening > thresholds['cell_abs_bias_degradation_max']:
                cell_worsening[f'{key[0].upper()}:{key[1]}'] = worsening
    if cell_worsening:
        reasons.append('cell_bias_protection_failed')
    reasons = list(dict.fromkeys(reasons))
    finite_key = np.isfinite([
        -candidate['profile_rmse'], candidate['mean_ccc'], -float(epoch)]).all()
    if not finite_key:
        reasons.append('nonfinite_selection_key')
    eligible = not reasons
    audit.update({
        'cell_abs_bias_worsening': cell_worsening,
        'eligible': bool(eligible),
        'rejection_reasons': reasons,
        'selection_key': (
            [-candidate['profile_rmse'], candidate['mean_ccc'], -int(epoch)]
            if eligible else None),
    })
    return audit


def select_m00_background_candidate(rows, baseline_metrics):
    """Select the shared best natural M00 epoch, with a safe fallback."""
    entries = []
    for row in rows:
        if row.get('stage', 'background') != 'background':
            continue
        epoch = row.get('epoch')
        try:
            audit = m00_natural_selection_audit(row, baseline_metrics, epoch)
        except (TypeError, ValueError):
            continue
        candidate = audit.get('candidate')
        if candidate is None:
            continue
        entries.append({
            'row': row,
            'audit': audit,
            'eligible': bool(audit['eligible']),
            'fallback_key': [
                -candidate['profile_rmse'], candidate['mean_ccc'], -int(epoch)],
        })
    if not entries:
        return {
            'selected_row': None,
            'selected_audit': None,
            'eligible': False,
            'entries': [],
            'reason': 'no_valid_background_epoch',
        }
    eligible_entries = [entry for entry in entries if entry['eligible']]
    pool = eligible_entries or entries
    selected = max(pool, key=lambda entry: tuple(entry['fallback_key']))
    return {
        'selected_row': selected['row'],
        'selected_audit': selected['audit'],
        'eligible': bool(eligible_entries),
        'entries': entries,
        'reason': None if eligible_entries else 'no_epoch_passed_protection',
    }


def _background_bias_selection_audit(metrics, baseline_metrics):
    """Apply the fixed A1-S eligibility gates and return a JSON-safe audit."""
    baseline = _background_selection_values(baseline_metrics, 'baseline')
    try:
        candidate = _background_selection_values(metrics, 'candidate')
    except ValueError as exc:
        return {
            'semantics': 'bias_constrained_v1',
            'eligible': False,
            'rejection_reasons': ['candidate_schema_or_finite'],
            'error': str(exc),
            'selection_key': None,
        }

    epsilon = 1e-12
    thresholds = {
        'rmse_degradation_max': 0.01,
        'ccc_degradation_max': 0.01,
        'cell_abs_bias_degradation_max': 0.02,
        'magnetic_min_profiles': 30,
        'mean_abs_bias_improvement_8_min': 0.005,
        'mean_abs_bias_improvement_24_min': 0.005,
        'source_conflict_abs_bias_min': 0.02,
    }
    rmse_delta = (
        candidate['scalars']['rmse'] - baseline['scalars']['rmse'])
    ccc_delta = candidate['scalars']['ccc'] - baseline['scalars']['ccc']
    cell_worsening = {}
    for key, item in candidate['cells'].items():
        degradation = abs(item['bias']) - abs(baseline['cells'][key]['bias'])
        if degradation > thresholds['cell_abs_bias_degradation_max'] + epsilon:
            cell_worsening[f'{key[0].upper()}:{key[1]}'] = float(degradation)

    conflicts = {}
    for cell in _BACKGROUND_SELECTION_CELLS:
        fy_bias = candidate['cells'][('fy', cell)]['bias']
        cosmic_bias = candidate['cells'][('cosmic', cell)]['bias']
        conflicts[cell] = bool(
            abs(fy_bias) >= thresholds['source_conflict_abs_bias_min'] - epsilon
            and abs(cosmic_bias) >=
            thresholds['source_conflict_abs_bias_min'] - epsilon
            and fy_bias * cosmic_bias < 0.0)

    baseline_mean_8 = float(np.mean([
        abs(item['bias']) for item in baseline['cells'].values()]))
    candidate_mean_8 = float(np.mean([
        abs(item['bias']) for item in candidate['cells'].values()]))
    compared_magnetic = [
        key for key in candidate['magnetic']
        if candidate['magnetic'][key]['profile_count'] >=
        thresholds['magnetic_min_profiles']
        and baseline['magnetic'][key]['profile_count'] >=
        thresholds['magnetic_min_profiles']
    ]
    magnetic_worsening = {}
    for key in compared_magnetic:
        degradation = (
            abs(candidate['magnetic'][key]['bias'])
            - abs(baseline['magnetic'][key]['bias']))
        if degradation > thresholds['cell_abs_bias_degradation_max'] + epsilon:
            magnetic_worsening[f'{key[0].upper()}:{key[1]}'] = float(degradation)
    if compared_magnetic:
        baseline_mean_24 = float(np.mean([
            abs(baseline['magnetic'][key]['bias'])
            for key in compared_magnetic]))
        candidate_mean_24 = float(np.mean([
            abs(candidate['magnetic'][key]['bias'])
            for key in compared_magnetic]))
        magnetic_improvement = baseline_mean_24 - candidate_mean_24
    else:
        baseline_mean_24 = candidate_mean_24 = magnetic_improvement = None
    cell_improvement = baseline_mean_8 - candidate_mean_8

    reasons = []
    if rmse_delta > thresholds['rmse_degradation_max'] + epsilon:
        reasons.append('rmse_degradation')
    if ccc_delta < -thresholds['ccc_degradation_max'] - epsilon:
        reasons.append('ccc_degradation')
    if cell_worsening:
        reasons.append('cell_abs_bias_worsening')
    if magnetic_worsening:
        reasons.append('magnetic_abs_bias_worsening')
    if any(conflicts.values()):
        reasons.append('source_conflict')
    if cell_improvement < (
            thresholds['mean_abs_bias_improvement_8_min'] - epsilon):
        reasons.append('insufficient_mean_abs_bias_improvement_8')
    if magnetic_improvement is None:
        reasons.append('no_magnetic_cells_with_30_profiles')
    elif magnetic_improvement < (
            thresholds['mean_abs_bias_improvement_24_min'] - epsilon):
        reasons.append('insufficient_mean_abs_bias_improvement_24')

    eligible = not reasons
    key = ([
        -candidate_mean_8, -candidate_mean_24,
        candidate['scalars']['ccc'], -candidate['scalars']['rmse'],
        candidate['scalars']['pearson_r'],
    ] if eligible else None)
    return {
        'semantics': 'bias_constrained_v1',
        'eligible': eligible,
        'rejection_reasons': reasons,
        'thresholds': thresholds,
        'rmse_delta_vs_a0': float(rmse_delta),
        'ccc_delta_vs_a0': float(ccc_delta),
        'mean_abs_bias_8': candidate_mean_8,
        'baseline_mean_abs_bias_8': baseline_mean_8,
        'mean_abs_bias_improvement_8': float(cell_improvement),
        'mean_abs_bias_24': candidate_mean_24,
        'baseline_mean_abs_bias_24': baseline_mean_24,
        'mean_abs_bias_improvement_24': (
            None if magnetic_improvement is None
            else float(magnetic_improvement)),
        'magnetic_cells_compared': len(compared_magnetic),
        'cell_abs_bias_worsening': cell_worsening,
        'magnetic_abs_bias_worsening': magnetic_worsening,
        'source_conflict_cells': conflicts,
        'selection_key': key,
    }


def _joint_fresh_source_cell_values(metrics, label):
    """Read only the FY/COSMIC 24 magnetic source cells for fresh selection."""
    if not isinstance(metrics, dict):
        raise ValueError(f'{label} metrics must be an object')
    values = {}
    for source in ('fy', 'cosmic'):
        try:
            cells = metrics[f'{source}_background_stratified'][
                'magnetic_latitude_cells']
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f'{label} metrics lacks {source} magnetic source cells') from exc
        for altitude_cell in _BACKGROUND_SELECTION_CELLS:
            for magnetic_label in _BACKGROUND_SELECTION_MAGNETIC_LABELS:
                cell = f'{altitude_cell}_{magnetic_label}'
                try:
                    item = cells[cell]['M00']
                    bias = float(item['bias'])
                    profiles = int(item['profile_count'])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f'{label} metrics lacks {source.upper()} {cell} M00 Bias') from exc
                if not np.isfinite(bias) or profiles < 0:
                    raise ValueError(
                        f'{label} metrics has invalid {source.upper()} {cell} M00 Bias')
                values[(source, cell)] = {
                    'bias': bias,
                    'profile_count': profiles,
                }
    if len(values) != 24:
        raise ValueError(f'{label} metrics must provide FY/COSMIC 24 source cells')
    return values


def _joint_fresh_adapter_selection_audit(metrics, baseline_metrics):
    """Apply the Joint-Fresh Development gates without using ISR metrics."""
    try:
        baseline = _background_selection_values(baseline_metrics, 'baseline')
        candidate = _background_selection_values(metrics, 'candidate')
    except ValueError as exc:
        return {
            'semantics': _COND_ADAPTER_JOINT_FRESH_SELECTOR,
            'eligible': False,
            'rejection_reasons': ['candidate_schema_or_finite'],
            'error': str(exc),
            'selection_key': None,
        }

    epsilon = 1e-12
    thresholds = {
        'rmse_degradation_max': 0.01,
        'ccc_degradation_max': 0.01,
        'source_cell_abs_bias_degradation_max': 0.01,
        'magnetic_min_profiles': 30,
        'mean_abs_bias_improvement_24_min': 0.005,
        'magnetic_strict_improvement_epsilon': epsilon,
        'mid_abs_bias_degradation_max': 0.002,
        'source_conflict_abs_bias_min': 0.02,
    }

    def _source_scalar(container, key, label):
        try:
            value = float(container[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{label} metrics lacks finite {key}') from exc
        if not np.isfinite(value):
            raise ValueError(f'{label} metrics lacks finite {key}')
        return value

    try:
        source_rmse = {
            source: _source_scalar(
                metrics, f'{source}_profile_rmse', 'candidate')
            for source in ('fy', 'cosmic')
        }
        baseline_source_rmse = {
            source: _source_scalar(
                baseline_metrics, f'{source}_profile_rmse', 'baseline')
            for source in ('fy', 'cosmic')
        }
        source_ccc = {
            source: _source_scalar(metrics, f'{source}_ccc', 'candidate')
            for source in ('fy', 'cosmic')
        }
        baseline_source_ccc = {
            source: _source_scalar(
                baseline_metrics, f'{source}_ccc', 'baseline')
            for source in ('fy', 'cosmic')
        }
    except ValueError as exc:
        return {
            'semantics': _COND_ADAPTER_JOINT_FRESH_SELECTOR,
            'eligible': False,
            'rejection_reasons': ['candidate_schema_or_finite'],
            'error': str(exc),
            'selection_key': None,
            'diagnostic_selection_key': None,
        }

    rmse_delta = {
        source: source_rmse[source] - baseline_source_rmse[source]
        for source in ('fy', 'cosmic')
    }
    ccc_delta = {
        source: source_ccc[source] - baseline_source_ccc[source]
        for source in ('fy', 'cosmic')
    }

    # The 24 source/altitude/day/magnetic cells are the selector's primary
    # bias population.  A cell with fewer than 30 Development profiles is
    # retained for the macro mean, but does not trigger the local worsening
    # gate by itself.
    candidate_mean = float(np.mean([
        abs(item['bias']) for item in candidate['magnetic'].values()]))
    baseline_mean = float(np.mean([
        abs(item['bias']) for item in baseline['magnetic'].values()]))
    if not np.isfinite(candidate_mean) or not np.isfinite(baseline_mean):
        return {
            'semantics': _COND_ADAPTER_JOINT_FRESH_SELECTOR,
            'eligible': False,
            'rejection_reasons': ['candidate_schema_or_finite'],
            'selection_key': None,
        }
    improvement = baseline_mean - candidate_mean

    cell_worsening = {}
    compared_cells = []
    for key, item in candidate['magnetic'].items():
        base_item = baseline['magnetic'][key]
        if (item['profile_count'] >= thresholds['magnetic_min_profiles']
                and base_item['profile_count'] >= thresholds['magnetic_min_profiles']):
            compared_cells.append(key)
            degradation = abs(item['bias']) - abs(base_item['bias'])
            if degradation > (
                    thresholds['source_cell_abs_bias_degradation_max'] + epsilon):
                cell_worsening[f'{key[0].upper()}:{key[1]}'] = float(degradation)

    # Aggregate low/mid/high magnetic bands over supported source cells.
    band_values = {}
    band_support = {}
    for band in _BACKGROUND_SELECTION_MAGNETIC_LABELS:
        keys = [
            key for key in compared_cells
            if key[1].endswith(f'_{band}')
        ]
        if not keys:
            band_values[band] = None
            band_support[band] = 0
            continue
        band_values[band] = (
            float(np.mean([abs(candidate['magnetic'][key]['bias']) for key in keys])),
            float(np.mean([abs(baseline['magnetic'][key]['bias']) for key in keys])),
        )
        band_support[band] = len(keys)

    conflicts = {}
    for altitude_cell in _BACKGROUND_SELECTION_CELLS:
        for band in _BACKGROUND_SELECTION_MAGNETIC_LABELS:
            key_name = f'{altitude_cell}_{band}'
            fy = candidate['magnetic'].get(('fy', key_name))
            cosmic = candidate['magnetic'].get(('cosmic', key_name))
            conflicts[key_name] = bool(
                fy is not None and cosmic is not None
                and abs(fy['bias']) >= thresholds['source_conflict_abs_bias_min'] - epsilon
                and abs(cosmic['bias']) >= thresholds['source_conflict_abs_bias_min'] - epsilon
                and fy['bias'] * cosmic['bias'] < 0.0)

    reasons = []
    for source in ('fy', 'cosmic'):
        if rmse_delta[source] > thresholds['rmse_degradation_max'] + epsilon:
            reasons.append(f'{source}_rmse_degradation')
        if ccc_delta[source] < -thresholds['ccc_degradation_max'] - epsilon:
            reasons.append(f'{source}_ccc_degradation')
    if cell_worsening:
        reasons.append('source_cell_abs_bias_worsening')
    if improvement < thresholds['mean_abs_bias_improvement_24_min'] - epsilon:
        reasons.append('insufficient_mean_abs_bias_improvement_24')
    low = band_values[_BACKGROUND_SELECTION_MAGNETIC_LABELS[0]]
    mid = band_values[_BACKGROUND_SELECTION_MAGNETIC_LABELS[1]]
    high = band_values[_BACKGROUND_SELECTION_MAGNETIC_LABELS[2]]
    if low is None:
        reasons.append('no_supported_low_magnetic_cells')
    if high is None:
        reasons.append('no_supported_high_magnetic_cells')
    if mid is None:
        reasons.append('no_supported_mid_magnetic_cells')
    if low is not None and not (low[0] < low[1] - epsilon):
        reasons.append('low_magnetic_not_strictly_improved')
    if high is not None and not (high[0] < high[1] - epsilon):
        reasons.append('high_magnetic_not_strictly_improved')
    if mid is not None and mid[0] - mid[1] > (
            thresholds['mid_abs_bias_degradation_max'] + epsilon):
        reasons.append('mid_magnetic_degradation')
    if any(conflicts.values()):
        reasons.append('source_conflict')

    eligible = not reasons
    worst_source_rmse = max(source_rmse.values())
    mean_ccc = float(np.mean(list(source_ccc.values())))
    diagnostic_key = None
    if low is not None and high is not None:
        diagnostic_key = [
            -max(low[0], high[0]),
            -candidate_mean,
            -worst_source_rmse,
            mean_ccc,
        ]
    key = diagnostic_key if eligible else None
    return {
        'semantics': _COND_ADAPTER_JOINT_FRESH_SELECTOR,
        'eligible': eligible,
        'rejection_reasons': reasons,
        'thresholds': thresholds,
        'source_profile_rmse': source_rmse,
        'baseline_source_profile_rmse': baseline_source_rmse,
        'source_profile_rmse_delta': rmse_delta,
        'source_ccc': source_ccc,
        'baseline_source_ccc': baseline_source_ccc,
        'source_ccc_delta': ccc_delta,
        'source_cell_count': len(candidate['magnetic']),
        'compared_source_cells': len(compared_cells),
        'source_cell_abs_bias_worsening': cell_worsening,
        'source_conflict_cells': conflicts,
        'mean_abs_bias_24': candidate_mean,
        'baseline_mean_abs_bias_24': baseline_mean,
        'mean_abs_bias_improvement_24': improvement,
        'magnetic_band_abs_bias': {
            band: None if band_values[band] is None else {
                'candidate': band_values[band][0],
                'baseline': band_values[band][1],
                'delta': band_values[band][0] - band_values[band][1],
                'cell_count': band_support[band],
            }
            for band in _BACKGROUND_SELECTION_MAGNETIC_LABELS
        },
        'selection_key': key,
        'diagnostic_selection_key': diagnostic_key,
    }


def _conditional_shape_cell_values(metrics, label):
    """Extract the fixed FY/COSMIC 24-cell shape metric population."""
    values = {}
    for source in ('fy', 'cosmic'):
        try:
            cells = metrics[f'{source}_background_stratified'][
                'magnetic_latitude_cells']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'{label} lacks {source} magnetic cells') from exc
        for altitude_cell in _BACKGROUND_SELECTION_CELLS:
            for band in _BACKGROUND_SELECTION_MAGNETIC_LABELS:
                cell = f'{altitude_cell}_{band}'
                try:
                    item = cells[cell]['M00']
                    row = {
                        key: float(item[key])
                        for key in ('bias', 'rmse', 'ccc', 'pearson_r',
                                    'std_ratio', 'spread_error', 'centered_rmse')}
                    row['profile_count'] = int(item['profile_count'])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f'{label} lacks finite {source.upper()} {cell} shape metrics') from exc
                if row['profile_count'] < 30 or not np.isfinite(
                        list(row.values())).all():
                    raise ValueError(
                        f'{label} has unsupported/non-finite {source.upper()} {cell}')
                values[(source, cell)] = row
    if len(values) != 24:
        raise ValueError(f'{label} must provide exactly 24 source cells')
    return values


def _conditional_shape_selection_audit(metrics, baseline_metrics):
    """Apply the conditional-shape Development gates; ISR is never consulted."""
    try:
        candidate = _conditional_shape_cell_values(metrics, 'candidate')
        baseline = _conditional_shape_cell_values(baseline_metrics, 'baseline')
        source_rmse = {
            source: float(metrics[f'{source}_profile_rmse'])
            for source in ('fy', 'cosmic')}
        baseline_rmse = {
            source: float(baseline_metrics[f'{source}_profile_rmse'])
            for source in ('fy', 'cosmic')}
        source_ccc = {
            source: float(metrics[f'{source}_ccc'])
            for source in ('fy', 'cosmic')}
        baseline_ccc = {
            source: float(baseline_metrics[f'{source}_ccc'])
            for source in ('fy', 'cosmic')}
        if not np.isfinite(
                list(source_rmse.values()) + list(baseline_rmse.values())
                + list(source_ccc.values()) + list(baseline_ccc.values())).all():
            raise ValueError('non-finite source metrics')
    except (KeyError, TypeError, ValueError) as exc:
        return {
            'semantics': _CONDITIONAL_SHAPE_SELECTOR,
            'eligible': False,
            'rejection_reasons': ['candidate_schema_support_or_finite'],
            'error': str(exc),
            'selection_key': None,
            'diagnostic_selection_key': None,
        }

    epsilon = 1e-12
    thresholds = {
        'source_rmse_degradation_max': 0.01,
        'source_ccc_degradation_max': 0.01,
        'night_ccc_improvement_min': 0.005,
        'mid_night_ccc_degradation_max': 0.005,
        'mid_night_spread_degradation_max': 0.02,
        'cell_abs_bias_degradation_max': 0.01,
        'cell_rmse_degradation_max': 0.01,
        'cell_ccc_degradation_max': 0.02,
        'day_ccc_degradation_max': 0.005,
        'day_spread_degradation_max': 0.02,
        'min_profiles_per_cell': 30,
    }

    def macro(rows, key):
        return float(np.mean([row[key] for row in rows]))

    night_keys = [key for key in candidate if '_night_' in key[1]]
    day_keys = [key for key in candidate if '_day_' in key[1]]
    band_keys = {
        band: [key for key in night_keys if key[1].endswith(f'_{band}')]
        for band in _BACKGROUND_SELECTION_MAGNETIC_LABELS}
    night = {
        name: {
            'candidate': macro([candidate[key] for key in night_keys], metric),
            'baseline': macro([baseline[key] for key in night_keys], metric),
        }
        for name, metric in (('ccc', 'ccc'), ('spread_error', 'spread_error'))}
    day = {
        name: {
            'candidate': macro([candidate[key] for key in day_keys], metric),
            'baseline': macro([baseline[key] for key in day_keys], metric),
        }
        for name, metric in (('ccc', 'ccc'), ('spread_error', 'spread_error'))}
    bands = {}
    for band, keys in band_keys.items():
        bands[band] = {
            name: {
                'candidate': macro([candidate[key] for key in keys], metric),
                'baseline': macro([baseline[key] for key in keys], metric),
            }
            for name, metric in (('ccc', 'ccc'), ('spread_error', 'spread_error'))}

    cell_worsening = {}
    for key, row in candidate.items():
        base = baseline[key]
        failures = {}
        if abs(row['bias']) - abs(base['bias']) > 0.01 + epsilon:
            failures['abs_bias_delta'] = abs(row['bias']) - abs(base['bias'])
        if row['rmse'] - base['rmse'] > 0.01 + epsilon:
            failures['rmse_delta'] = row['rmse'] - base['rmse']
        if row['ccc'] - base['ccc'] < -0.02 - epsilon:
            failures['ccc_delta'] = row['ccc'] - base['ccc']
        if failures:
            cell_worsening[f'{key[0].upper()}:{key[1]}'] = {
                name: float(value) for name, value in failures.items()}

    reasons = []
    for source in ('fy', 'cosmic'):
        if source_rmse[source] - baseline_rmse[source] > 0.01 + epsilon:
            reasons.append(f'{source}_profile_rmse_degradation')
        if source_ccc[source] - baseline_ccc[source] < -0.01 - epsilon:
            reasons.append(f'{source}_ccc_degradation')
    if night['ccc']['candidate'] - night['ccc']['baseline'] < 0.005 - epsilon:
        reasons.append('insufficient_night_macro_ccc_improvement')
    if not (night['spread_error']['candidate']
            < night['spread_error']['baseline'] - epsilon):
        reasons.append('night_macro_spread_not_improved')
    low, mid, high = _BACKGROUND_SELECTION_MAGNETIC_LABELS
    for band in (low, high):
        if bands[band]['ccc']['candidate'] < bands[band]['ccc']['baseline'] - epsilon:
            reasons.append(f'{band}_night_ccc_degradation')
        if (bands[band]['spread_error']['candidate']
                > bands[band]['spread_error']['baseline'] + epsilon):
            reasons.append(f'{band}_night_spread_degradation')
    if (bands[mid]['ccc']['candidate'] - bands[mid]['ccc']['baseline']
            < -0.005 - epsilon):
        reasons.append('mid_night_ccc_degradation')
    if (bands[mid]['spread_error']['candidate']
            - bands[mid]['spread_error']['baseline'] > 0.02 + epsilon):
        reasons.append('mid_night_spread_degradation')
    if day['ccc']['candidate'] - day['ccc']['baseline'] < -0.005 - epsilon:
        reasons.append('day_macro_ccc_degradation')
    if (day['spread_error']['candidate'] - day['spread_error']['baseline']
            > 0.02 + epsilon):
        reasons.append('day_macro_spread_degradation')
    if cell_worsening:
        reasons.append('source_cell_degradation')

    overall_spread = macro(list(candidate.values()), 'spread_error')
    overall_ccc = float(np.mean(list(source_ccc.values())))
    overall_rmse = float(np.mean(list(source_rmse.values())))
    worst_low_high_spread = max(
        bands[low]['spread_error']['candidate'],
        bands[high]['spread_error']['candidate'])
    diagnostic_key = [
        -worst_low_high_spread,
        night['ccc']['candidate'],
        -overall_spread,
        overall_ccc,
        -overall_rmse,
    ]
    return {
        'semantics': _CONDITIONAL_SHAPE_SELECTOR,
        'eligible': not reasons,
        'rejection_reasons': reasons,
        'thresholds': thresholds,
        'source_profile_rmse_delta': {
            source: source_rmse[source] - baseline_rmse[source]
            for source in ('fy', 'cosmic')},
        'source_ccc_delta': {
            source: source_ccc[source] - baseline_ccc[source]
            for source in ('fy', 'cosmic')},
        'night_macro': night,
        'night_magnetic_bands': bands,
        'day_macro': day,
        'source_cell_worsening': cell_worsening,
        'selection_key': diagnostic_key if not reasons else None,
        'diagnostic_selection_key': diagnostic_key,
    }


def _a1s8_selection_audit(metrics, baseline_metrics, exposure=None,
                          config=None, gradient_oob_streak=0):
    """Apply the fixed A1-S selector plus A1-S8 adapter/exposure guards."""
    config = config or {}
    exposure = exposure or {}
    base = _background_bias_selection_audit(metrics, baseline_metrics)
    audit = dict(base)
    reasons = list(base.get('rejection_reasons', []))

    def _finite(name, default=None):
        value = exposure.get(name, metrics.get(name, default))
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    cap_saturation = _finite('adapter_cap_saturation')
    if cap_saturation is None or cap_saturation > 0.05:
        reasons.append('adapter_cap_saturation')

    gradient_ratio = _finite('gradient_ratio')
    ratio_min = float(config.get('background_gradient_ratio_min', 0.25))
    ratio_max = float(config.get('background_gradient_ratio_max', 4.0))
    ratio_ok = (gradient_ratio is not None
                and ratio_min <= gradient_ratio <= ratio_max)
    if not ratio_ok:
        reasons.append('adapter_gradient_ratio')
    if gradient_oob_streak >= 2:
        reasons.append('adapter_gradient_ratio_consecutive')

    low_mass = _finite('low_effective_gate_mass')
    high_mass = _finite('high_effective_gate_mass')
    low_bounds = (float(config.get('background_low_effective_mass_min', 0.25)),
                  float(config.get('background_low_effective_mass_max', 0.35)))
    high_bounds = (
        float(config.get('background_high_effective_mass_min', 0.65)),
        float(config.get('background_high_effective_mass_max', 0.75)))
    if low_mass is None or not low_bounds[0] <= low_mass <= low_bounds[1]:
        reasons.append('low_effective_gate_mass')
    if high_mass is None or not high_bounds[0] <= high_mass <= high_bounds[1]:
        reasons.append('high_effective_gate_mass')

    low_blocks = exposure.get('low_geographic_day_blocks',
                              metrics.get('low_geographic_day_blocks'))
    try:
        low_blocks = int(low_blocks)
    except (TypeError, ValueError):
        low_blocks = 0
    regional_allowed = low_blocks >= 30
    if not regional_allowed:
        reasons.append('low_latitude_under_supported')

    finite_values = [cap_saturation, gradient_ratio, low_mass, high_mass]
    if any(value is None for value in finite_values):
        reasons.append('a1s8_nonfinite_exposure')
    # Preserve order while avoiding duplicate diagnostics from base/a1s8 checks.
    reasons = list(dict.fromkeys(reasons))
    audit.update({
        'semantics': 'a1s8_m00_preserve_bias_constrained_v1',
        'base_semantics': base.get('semantics'),
        'eligible': bool(base.get('eligible', False) and not reasons),
        'rejection_reasons': reasons,
        'a1s8_thresholds': {
            'adapter_cap_saturation_max': 0.05,
            'gradient_ratio_min': ratio_min,
            'gradient_ratio_max': ratio_max,
            'low_effective_mass': list(low_bounds),
            'high_effective_mass': list(high_bounds),
            'low_geographic_day_blocks_min': 30,
        },
        'adapter_cap_saturation': cap_saturation,
        'gradient_ratio': gradient_ratio,
        'gradient_oob_streak': int(gradient_oob_streak),
        'low_effective_gate_mass': low_mass,
        'high_effective_gate_mass': high_mass,
        'low_geographic_day_blocks': low_blocks,
        'regional_conclusion_allowed': regional_allowed,
        'selection_key': (base.get('selection_key')
                          if base.get('eligible', False) and not reasons
                          else None),
    })
    return audit



def _load_background_selection_baseline(config):
    semantics = config.get('background_checkpoint_selection', 'legacy')
    if semantics == 'legacy':
        return None
    if semantics not in (
            'bias_constrained_v1', _COND_ADAPTER_SELECTOR,
            _COND_ADAPTER_JOINT_FRESH_SELECTOR, _CONDITIONAL_SHAPE_SELECTOR,
            _M00_NIGHT_GENERALIZATION_SELECTOR, _M00_NIGHT_GENERALIZATION_V2_SELECTOR):
        raise ValueError(
            f'unsupported Background checkpoint selection: {semantics}')
    path = config.get('background_selection_baseline')
    if (semantics == _CONDITIONAL_SHAPE_SELECTOR and not path
            and config.get('background_selection_baseline_checkpoint')):
        return None
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f'Background selection baseline replay is missing: {path}')
    with open(path, encoding='utf-8') as stream:
        replay = json.load(stream)
    if not isinstance(replay, dict) or not isinstance(replay.get('metrics'), dict):
        raise ValueError('Background selection baseline replay lacks metrics object')
    try:
        replay_score = float(replay['score'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            'Background selection baseline replay lacks finite score') from exc
    if not np.isfinite(replay_score):
        raise ValueError('Background selection baseline replay lacks finite score')
    if semantics == _M00_NIGHT_GENERALIZATION_V2_SELECTOR:
        audit = night_generalization_v2_audit(
            replay['metrics'], replay['metrics'], 0,
            config.get('background_night_v2_supported_cells'))
        if any(reason.startswith('invalid_required_metrics:') for reason in audit['reasons']):
            raise ValueError(f'invalid night-v2 baseline: {audit["reasons"]}')
    elif semantics == _M00_NIGHT_GENERALIZATION_SELECTOR:
        _night_generalization_cells(replay['metrics'], 'baseline')
    elif semantics == _COND_ADAPTER_JOINT_FRESH_SELECTOR:
        _joint_fresh_source_cell_values(replay['metrics'], 'baseline')
    elif semantics == _CONDITIONAL_SHAPE_SELECTOR:
        _conditional_shape_cell_values(replay['metrics'], 'baseline')
    else:
        _background_selection_values(replay['metrics'], 'baseline')
    return replay['metrics']


def _load_m00_natural_selection_baseline(config):
    """Load the fixed historical A1-S epoch used by the natural selector."""
    path = config.get('m00_selection_reference_history')
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f'natural M00 selection history is missing: {path}')
    with open(path, encoding='utf-8') as stream:
        history = json.load(stream)
    epoch = int(config.get('m00_selection_reference_epoch', 3))
    rows = [
        row for row in history
        if isinstance(row, dict)
        and row.get('stage') == 'background'
        and int(row.get('epoch', -1)) == epoch
    ]
    if len(rows) != 1:
        raise ValueError(
            f'natural M00 selection history must contain one Background epoch {epoch}')
    # Fail at configuration time instead of silently selecting against a bad baseline.
    baseline_audit = m00_natural_selection_audit(rows[0], rows[0], epoch)
    if baseline_audit.get('candidate') is None:
        raise ValueError(
            'natural M00 selection history lacks finite baseline metrics')
    config['m00_selection_reference_history'] = os.path.abspath(path)
    return rows[0]


def _replay_conditional_shape_baseline(
        model, val_loader, cosmic_val_loader, batch_processor, device, config,
        sw_manager, iri_peak_manager, allowed_profile_ids):
    """Evaluate A1-S2 epoch 3 on this run's exact Development manifest."""
    checkpoint = config.get('background_selection_baseline_checkpoint')
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f'conditional-shape baseline checkpoint is missing: {checkpoint}')
    state = {name: value.detach().clone()
             for name, value in model.state_dict().items()}
    module_modes = [(module, module.training) for module in model.modules()]
    try:
        loaded = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(loaded, strict=True)
        score, metrics = validate(
            model, val_loader, batch_processor, device, config, sw_manager,
            iri_peak_manager, cosmic_val_loader, 'background',
            allowed_profile_ids)
        _conditional_shape_cell_values(metrics, 'baseline')
    finally:
        model.load_state_dict(state, strict=True)
        for module, training in module_modes:
            module.train(training)
    replay = {
        'score': float(score),
        'metrics': metrics,
        'checkpoint': os.path.abspath(checkpoint),
        'profile_split': config.get('resolved_profile_split'),
        'semantics': _CONDITIONAL_SHAPE_SELECTOR,
    }
    path = os.path.join(
        config['save_dir'], 'a1s2_conditional_shape_development_replay.json')
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(replay, stream, ensure_ascii=False, indent=2)
    config['background_selection_baseline'] = path
    return metrics


def _development_selection_key(metrics, config, stage=None,
                               baseline_metrics=None):
    """Return a JSON-safe lexicographic key; larger is always better."""
    selection_semantics = config.get(
        'background_checkpoint_selection', 'legacy')
    if stage == 'background' and selection_semantics == _M00_NIGHT_GENERALIZATION_V2_SELECTOR:
        return night_generalization_v2_audit(
            metrics, baseline_metrics, metrics.get('epoch', 0),
            config.get('background_night_v2_supported_cells'))['selection_key']
    if (stage == 'background'
            and selection_semantics == _M00_NATURAL_SELECTOR):
        if baseline_metrics is None:
            raise ValueError(
                'm00 natural selector requires historical baseline metrics')
        return m00_natural_selection_audit(
            metrics, baseline_metrics, metrics.get('epoch', 0)).get(
                'selection_key')
    if stage == 'background' and selection_semantics in (
            'bias_constrained_v1', _COND_ADAPTER_SELECTOR,
            _COND_ADAPTER_JOINT_FRESH_SELECTOR, _CONDITIONAL_SHAPE_SELECTOR,
            _M00_NIGHT_GENERALIZATION_SELECTOR):
        if baseline_metrics is None:
            raise ValueError('bias_constrained_v1 requires A0 baseline metrics')
        if _a1s8_enabled(config):
            return _a1s8_selection_audit(
                metrics, baseline_metrics,
                exposure=metrics.get('a1s8_exposure'),
                config=config).get('selection_key')
        if selection_semantics == _COND_ADAPTER_JOINT_FRESH_SELECTOR:
            return _joint_fresh_adapter_selection_audit(
                metrics, baseline_metrics)['selection_key']
        if selection_semantics == _CONDITIONAL_SHAPE_SELECTOR:
            return _conditional_shape_selection_audit(
                metrics, baseline_metrics)['selection_key']
        if selection_semantics == _M00_NIGHT_GENERALIZATION_SELECTOR:
            return _night_generalization_selection_audit(
                metrics, baseline_metrics, metrics.get('epoch', 0))[
                    'selection_key']
        return _background_bias_selection_audit(
            metrics, baseline_metrics)['selection_key']
    semantics = config.get(
        'checkpoint_selection_semantics', 'mean_profile_rmse_v1')
    if semantics == 'mean_profile_rmse_v1':
        values = (-float(metrics['score']),)
    elif semantics == 'mean_ccc_then_rmse_then_pearson_v1':
        values = (
            float(metrics['ccc']), -float(metrics['rmse']),
            float(metrics['pearson_r']))
    else:
        raise ValueError(f'unsupported checkpoint selection semantics: {semantics}')
    return list(values) if np.isfinite(values).all() else None


def _mad_variance(values, sigma_min=0.05, sigma_max=0.40):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('cannot calibrate R from empty residuals')
    median = np.median(values)
    sigma = float(np.clip(
        1.4826 * np.median(np.abs(values - median)),
        sigma_min, sigma_max))
    return sigma ** 2


def _stratified_variance_table(data, config):
    residual = np.asarray(data['residual'], dtype=np.float64).reshape(-1)
    altitude = np.asarray(data['altitude'], dtype=np.float64).reshape(-1)
    local_time = np.remainder(
        np.asarray(data['local_time'], dtype=np.float64).reshape(-1), 24.0)
    profile_id = np.asarray(data['profile_id']).reshape(-1)
    if not (len(residual) == len(altitude) == len(local_time) == len(profile_id)):
        raise ValueError('R calibration arrays must have equal lengths')

    valid = (
        np.isfinite(residual) & np.isfinite(altitude)
        & np.isfinite(local_time) & np.isfinite(profile_id))
    residual = residual[valid]
    altitude = altitude[valid]
    local_time = local_time[valid]
    profile_id = profile_id[valid]

    sigma_min = float(config.get('r_sigma_min', 0.05))
    sigma_max = float(config.get('r_sigma_max', 0.40))
    min_profiles = int(config.get('r_min_profiles', 200))
    shrinkage_profiles = float(config.get('r_shrinkage_profiles', 200))
    if not (0 < sigma_min <= sigma_max):
        raise ValueError('R sigma bounds must satisfy 0 < min <= max')
    if min_profiles < 1 or shrinkage_profiles < 0:
        raise ValueError('R profile thresholds must be non-negative')

    global_variance = _mad_variance(residual, sigma_min, sigma_max)
    domain_min, domain_max = _observation_alt_range(config)
    altitude_edges = [domain_min] + [
        edge for edge in (200.0, 300.0)
        if domain_min < edge < domain_max] + [domain_max]
    table = np.full((len(altitude_edges) - 1, 2), global_variance,
                    dtype=np.float64)
    altitude_index = np.searchsorted(
        np.asarray(altitude_edges[1:-1]), altitude, side='right')
    day_index = ((local_time >= 6.0) & (local_time < 18.0)).astype(np.int64)
    cells = []
    for altitude_cell in range(len(altitude_edges) - 1):
        row = []
        for day_cell in range(2):
            cell_mask = (
                (altitude_index == altitude_cell) & (day_index == day_cell))
            point_count = int(cell_mask.sum())
            profile_count = int(np.unique(profile_id[cell_mask]).size)
            raw_variance = (
                _mad_variance(residual[cell_mask], sigma_min, sigma_max)
                if point_count else None)
            fallback = profile_count < min_profiles
            alpha = (
                0.0 if fallback else
                profile_count / (profile_count + shrinkage_profiles))
            if not fallback:
                table[altitude_cell, day_cell] = (
                    alpha * raw_variance + (1.0 - alpha) * global_variance)
            row.append({
                'point_count': point_count,
                'profile_count': profile_count,
                'raw_sigma': (
                    None if raw_variance is None else float(np.sqrt(raw_variance))),
                'raw_variance': (
                    None if raw_variance is None else float(raw_variance)),
                'alpha': float(alpha),
                'fallback_to_global': fallback,
                'final_sigma': float(np.sqrt(table[altitude_cell, day_cell])),
                'final_variance': float(table[altitude_cell, day_cell]),
            })
        cells.append(row)

    metadata = {
        'altitude_bins_km': [
            [float(lower), float(upper)]
            for lower, upper in zip(altitude_edges[:-1], altitude_edges[1:])],
        'local_time_columns': ['night', 'day'],
        'day_definition': '06:00 <= LT < 18:00',
        'global': {
            'point_count': int(residual.size),
            'profile_count': int(np.unique(profile_id).size),
            'sigma': float(np.sqrt(global_variance)),
            'variance': float(global_variance),
        },
        'cells': cells,
        'calibrated_variance_table': table.tolist(),
    }
    return float(global_variance), table, metadata


@torch.no_grad()
def _calibrate_fixed_r(model, fy_loader, cosmic_loader, device, sw_manager,
                       iri_peak_manager, batch_processor, config):
    if config.get('r_source') in (_CAUSAL_FROZEN_R_SOURCE,
                                  _A1S_FROZEN_R_SOURCE):
        if config.get('r_mode', 'global') != 'global':
            raise ValueError('causal history analysis requires global frozen R')
        frozen = (_A1S_FROZEN_R if config.get('r_source') == _A1S_FROZEN_R_SOURCE
                  else _A1S2_FROZEN_R)
        fixed = {
            'FY': float(config.get('r_fy_fixed') or frozen['FY']),
            'COSMIC': float(config.get('r_cosmic_fixed') or frozen['COSMIC']),
        }
        for source, expected in frozen.items():
            if (not np.isfinite(fixed[source])
                    or not np.isclose(fixed[source], expected, rtol=0.0,
                                      atol=1e-12)):
                raise ValueError(
                    f'causal history frozen {source} R must equal {expected:.10f}')
        model.kalman_layer.set_observation_variances(
            fixed['FY'], fixed['COSMIC'])
        config['resolved_r_source'] = config.get('r_source')
        config['resolved_r_values'] = dict(fixed)
        report = {
            'semantics': config.get('r_source'),
            'scope': 'fixed global R; no recalibration',
            'r_fy': fixed['FY'],
            'r_cosmic': fixed['COSMIC'],
        }
        output_path = os.path.join(config['save_dir'], 'r_calibration.json')
        temporary = f'{output_path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, output_path)
        return
    model.eval()

    def collect(loader, source):
        collected = {
            'residual': [], 'altitude': [], 'local_time': [], 'profile_id': []}
        sampler = loader.batch_sampler
        previous_shuffle = getattr(sampler, 'shuffle', None)
        if previous_shuffle is not None:
            sampler.shuffle = False
        try:
            for batch_index, batch in enumerate(loader):
                max_batches = config.get('r_calibration_batches')
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                coords, target, profile_ids = _unpack_source_batch(batch, device)
                sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
                iri_peak = (iri_peak_manager.get_iri_peak(coords)
                            if iri_peak_manager is not None else None)
                _, _, _, _, extras = _source_forward(
                    model, batch_processor, coords, sw_seq, iri_peak,
                    'background', source, profile_ids, False, config,
                    iri_peak_manager)
                collected['residual'].append(
                    (extras['ne_bkg'] - target).flatten().cpu().numpy())
                collected['altitude'].append(coords[:, 2].cpu().numpy())
                collected['local_time'].append(torch.remainder(
                    coords[:, 3] + coords[:, 1] / 15.0, 24.0).cpu().numpy())
                collected['profile_id'].append(
                    profile_ids.flatten().cpu().numpy())
        finally:
            if previous_shuffle is not None:
                sampler.shuffle = previous_shuffle
        if not collected['residual']:
            raise ValueError(f'{source} R calibration selected no training data')
        return {
            key: np.concatenate(parts) for key, parts in collected.items()}

    fy_global, fy_table, fy_report = _stratified_variance_table(
        collect(fy_loader, 'FY'), config)
    cosmic_global, cosmic_table, cosmic_report = _stratified_variance_table(
        collect(cosmic_loader, 'COSMIC'), config)
    mode = config.get('r_mode', 'global')
    if mode != 'global':
        raise ValueError('v7 density-observation ETKF requires r_mode=global')
    fy_active = np.full((3, 2), fy_global)
    cosmic_active = np.full((3, 2), cosmic_global)
    model.kalman_layer.set_observation_variances(
        fy_global, cosmic_global, fy_active, cosmic_active)

    report = {
        'schema_version': 1,
        'mode': mode,
        'seed': int(config['seed']),
        'train_only': True,
        'points_per_profile': int(config.get('profile_points_per_epoch', 8)),
        'sigma_clip_dex': [
            float(config.get('r_sigma_min', 0.05)),
            float(config.get('r_sigma_max', 0.40)),
        ],
        'min_profiles': int(config.get('r_min_profiles', 200)),
        'shrinkage_profiles': float(config.get('r_shrinkage_profiles', 200)),
        'FY': {**fy_report, 'active_variance_table': fy_active.tolist()},
        'COSMIC': {
            **cosmic_report, 'active_variance_table': cosmic_active.tolist()},
    }
    manifest_path = os.path.join(config['save_dir'], 'run_manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        report['data_identity'] = manifest.get('data_identity', {})
        report['code_identity'] = manifest.get('code_identity', {})
    seed_path = config.get('background_seed_ckpt')
    if (seed_path and os.path.isfile(seed_path)
            and not config.get('skip_hash_verification', False)):
        report['background_seed_checkpoint'] = {
            'path': os.path.abspath(seed_path),
            'sha256': _sha256_file(seed_path),
        }
    elif seed_path and os.path.isfile(seed_path):
        report['background_seed_checkpoint'] = {
            'path': os.path.abspath(seed_path),
        }
    output_path = os.path.join(config['save_dir'], 'r_calibration.json')
    temp_path = f'{output_path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_path)
    print(
        f'  冻结有效R [{mode}]: FY={fy_global:.6f}, '
        f'COSMIC={cosmic_global:.6f}')
    return report


def _load_iri_peak_manager(config, device):
    hmf2_path = config.get('iri_hmf2_path')
    nmf2_path = config.get('iri_nmf2_path')
    if not (hmf2_path and nmf2_path
            and os.path.exists(hmf2_path) and os.path.exists(nmf2_path)):
        return None
    from data_managers.iri_peak_manager import IRIPeakManager
    return IRIPeakManager(
        hmf2_path=hmf2_path, nmf2_path=nmf2_path, device=device)


def _resume_needs_analysis_setup(resume_state, start_epoch, background_epochs):
    if resume_state is None:
        return False
    stage = resume_state.get('stage')
    if stage == 'background':
        if start_epoch > background_epochs:
            raise ValueError('background checkpoint has inconsistent completed_epochs')
        return start_epoch == background_epochs
    if stage == 'analysis':
        if start_epoch <= background_epochs:
            raise ValueError('analysis checkpoint has inconsistent completed_epochs')
        return False
    raise ValueError(f'checkpoint has unknown training stage: {stage}')


def _reset_random_seeds(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _architecture_signature(config):
    signature = {
        'checkpoint_format_version': int(config.get(
            'checkpoint_format_version', 12)),
        'alt_range': [float(value) for value in config.get(
            'alt_range', (120.0, 500.0))],
        'observation_alt_range': [float(value) for value in (
            config.get('observation_alt_range') or config.get(
                'alt_range', (120.0, 500.0)))],
        'peak_search_alt_range': [float(value) for value in (
            config.get('peak_search_alt_range') or config.get(
                'alt_range', (120.0, 500.0)))],
        'model_domain_semantics': config.get(
            'model_domain_semantics', 'legacy_120_500_domain_v1'),
        'basis_dim': int(config.get('basis_dim', 64)),
        'enkf_n_members': int(config.get('enkf_n_members', 8)),
        'enkf_pert_hidden': int(config.get('enkf_pert_hidden', 64)),
        'enkf_anomaly_parameterization': config.get(
            'enkf_anomaly_parameterization', 'legacy_independent'),
        'enkf_scale_init': float(config.get('enkf_scale_init', 1.1)),
        'enkf_scale_condition_max': float(
            config.get('enkf_scale_condition_max', 3.0)),
        'density_basis_semantics': config.get(
            'density_basis_semantics', 'query_conditioned'),
        'analysis_state_semantics': config.get(
            'analysis_state_semantics', 'legacy_feature_increment'),
        'context_semantics': config.get(
            'context_semantics', 'query_conditioning'),
        'mode_basis_semantics': config.get(
            'mode_basis_semantics', 'learned_density_basis'),
        'background_trust_gate_enabled': bool(config.get(
            'background_trust_gate_enabled', False)),
        'background_trust_gate_semantics': config.get(
            'background_trust_gate_semantics', 'disabled'),
        'background_trust_gate_altitude_core_km': float(config.get(
            'background_trust_gate_altitude_core_km', 200.0)),
        'background_trust_gate_altitude_transition_km': float(config.get(
            'background_trust_gate_altitude_transition_km', 100.0)),
        'background_trust_gate_night_cosine_offset': float(config.get(
            'background_trust_gate_night_cosine_offset', 0.2)),
        'background_trust_gate_night_cosine_scale': float(config.get(
            'background_trust_gate_night_cosine_scale', 0.2)),
        'background_trust_gate_dip_core': float(config.get(
            'background_trust_gate_dip_core', 0.25)),
        'background_trust_gate_dip_transition': float(config.get(
            'background_trust_gate_dip_transition', 0.25)),
        'background_decoder_conditioning': config.get(
            'background_decoder_conditioning', 'legacy_v1'),
    }
    if config.get('assimilation_semantics') == 'continuous_physical_local_letkf':
        signature.update({
            'assimilation_semantics': config['assimilation_semantics'],
            'neighbor_directory_semantics': config.get(
                'neighbor_directory_semantics', 'profile_center_legacy'),
            'physical_localization_space_km': float(config.get(
                'physical_localization_space_km', 1800.0)),
            'physical_localization_time_hours': float(config.get(
                'physical_localization_time_hours', 1.5)),
            'observation_chunk_size': int(config.get(
                'observation_chunk_size', 4096)),
        })
    return signature


def _restrict_training_profiles(loader, fraction, seed, source, manifest_path):
    if not 0.0 < fraction <= 1.0:
        raise ValueError('train_profile_fraction must be in (0, 1]')
    sampler = loader.batch_sampler
    profile_groups = [
        indices for groups in sampler.profiles_by_bin.values()
        for indices in groups
    ]
    available = np.array(sorted({
        int(loader.dataset.profile_ids[indices[0]])
        for indices in profile_groups
    }), dtype=np.int64)
    if fraction == 1.0:
        return {'available': int(len(available)), 'selected': int(len(available))}
    if not manifest_path:
        raise ValueError(
            'profile_subset_manifest is required when train_profile_fraction < 1')
    path = os.path.abspath(manifest_path)
    manifest = {}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        if (manifest.get('seed') != int(seed)
                or not np.isclose(manifest.get('fraction', -1.0), fraction)):
            raise ValueError('profile subset manifest seed/fraction mismatch')
    sources = manifest.setdefault('sources', {})
    if source in sources:
        selected = np.asarray(sources[source]['profile_ids'], dtype=np.int64)
        if not np.all(np.isin(selected, available)):
            raise ValueError(
                f'{source} profile subset manifest contains unavailable IDs')
    else:
        rng = np.random.default_rng(
            int(seed) + (0 if source == 'FY' else 1))
        count = max(1, int(round(len(available) * fraction)))
        selected = np.sort(rng.choice(available, count, replace=False))
        sources[source] = {
            'dataset_path': os.path.abspath(loader.dataset.npy_path),
            'available_profiles': int(len(available)),
            'selected_profiles': int(len(selected)),
            'profile_ids': selected.tolist(),
        }
        manifest.update({
            'schema_version': 1,
            'seed': int(seed),
            'fraction': float(fraction),
        })
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        temp_path = f'{path}.tmp'
        with open(temp_path, 'w', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    selected_set = set(selected.tolist())
    sampler.profiles_by_bin = {
        bin_id: [
            indices for indices in groups
            if int(loader.dataset.profile_ids[indices[0]]) in selected_set
        ]
        for bin_id, groups in sampler.profiles_by_bin.items()
    }
    sampler.profiles_by_bin = {
        bin_id: groups for bin_id, groups in sampler.profiles_by_bin.items()
        if groups
    }
    return {'available': int(len(available)), 'selected': int(len(selected))}


def _training_stratum_weights(loader):
    """Profile-balanced inverse frequencies for low/high altitude and day/night."""
    sampler = loader.batch_sampler
    counts = np.zeros(4, dtype=np.float64)
    profiles = 0
    for groups in sampler.profiles_by_bin.values():
        for indices in groups:
            rows = loader.dataset.selected_indices[indices]
            values = np.asarray(loader.dataset.data[rows, :4])
            local_time = np.remainder(
                values[:, 3] + values[:, 1] / 15.0, 24.0)
            night = (local_time < 6.0) | (local_time >= 18.0)
            cells = (values[:, 2] >= 300.0).astype(np.int64) * 2
            cells += night.astype(np.int64)
            profile_counts = np.bincount(cells, minlength=4).astype(np.float64)
            counts += profile_counts / max(profile_counts.sum(), 1.0)
            profiles += 1
    if profiles == 0 or np.any(counts <= 0):
        raise ValueError(
            f'covariance stratum calibration has empty cells: {counts.tolist()}')
    frequencies = counts / counts.sum()
    weights = 1.0 / (4.0 * frequencies)
    return {
        'profiles': profiles,
        'frequencies': frequencies.tolist(),
        'weights': weights.tolist(),
    }


def _load_background_seed(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    expected = model.state_dict()
    background_prefixes = (
        'iri_proxy.', 'iri_align_net.', 'sw_encoder.', 'sw_freq_branch.',
        'sw_gate.', 'background_decoder.')
    adapter_prefixes = (
        'background_condition_adapter.',
        'background_condition_adapter_low.',
        'background_condition_adapter_high.',
    )
    required = {
        name for name in expected if name.startswith(background_prefixes)}
    optional = {
        name for name in expected if name.startswith(adapter_prefixes)}
    # Background checkpoints contain the complete model state (including
    # frozen Analysis tensors); only truly unknown keys are incompatible.
    unexpected = set(state) - set(expected)
    if unexpected:
        raise RuntimeError(
            'Background seed contains unexpected parameters: '
            f'{sorted(unexpected)[:8]}')
    missing = required - set(state)
    mismatched = {
        name for name in required & set(state)
        if state[name].shape != expected[name].shape}
    adapter_mismatched = {
        name for name in optional & set(state)
        if state[name].shape != expected[name].shape}
    if missing or mismatched or adapter_mismatched:
        raise RuntimeError(
            'Background seed is architecture-incompatible: '
            f'missing={sorted(missing)}, '
            f'shape_mismatch={sorted(mismatched | adapter_mismatched)}')
    migrated = dict(expected)
    for name in required | (optional & set(state)):
        migrated[name] = state[name]
    model.load_state_dict(migrated, strict=True)


def _background_epoch_length(fy_loader, cosmic_loader, stage,
                             model_domain=None):
    """Return the source schedule length for one training epoch."""
    return (max(len(fy_loader), len(cosmic_loader))
            if (stage == 'background'
                or model_domain == _A2S1_DOMAIN_SEMANTICS)
            else len(fy_loader))


def _validate_a2s1_loader_lengths(config, fy_loader, cosmic_loader):
    if (config.get('model_domain_semantics') == _A2S1_DOMAIN_SEMANTICS
            and not config.get('smoke_run', False)
            and len(fy_loader) != len(cosmic_loader)):
        raise RuntimeError(
            'formal A2S1 requires equal FY/COSMIC loader lengths; '
            f'got FY={len(fy_loader)}, COSMIC={len(cosmic_loader)}')


def _assert_fresh_background_identity(model, sw_manager, iri_peak_manager, device):
    """The zero-initialized Background must start exactly at raw IRI."""
    lower = float(model.alt_min)
    upper = float(model.alt_max)
    coords = torch.tensor(
        [[-11.9, -76.0, lower, 4.0],
         [0.0, 120.0, 0.5 * (lower + upper), 240.0]],
        dtype=torch.float32, device=device)
    sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
    peak = (iri_peak_manager.get_iri_peak(coords)
            if iri_peak_manager is not None else None)
    encoded = model.encode_background(coords, sw_seq, iri_peak=peak)
    if not torch.equal(encoded['ne_bkg'], encoded['ne_iri']):
        error = (encoded['ne_bkg'] - encoded['ne_iri']).abs().max().item()
        raise RuntimeError(
            f'fresh Background invariant failed: M00 != Raw IRI (max={error:.3e})')


def train_fsia(config=None):
    """Train Background first, freeze it, then train ETKF Analysis."""
    config = get_config_mdia() if config is None else config
    m00_calibration = _resolve_m00_observation_calibration(config)
    validate_low_dim_assimilation_config(config)
    validate_frozen_m00_config(config)
    _write_frozen_v2_contracts(config)
    _write_m00_transition_contract(
        config, status='active' if m00_calibration is not None else 'pending')
    _resolve_a1s7_update_budget(config)
    _resolve_a1s8_update_budget(config)
    if _natural_background_enabled(config):
        if config.get('background_training_scope') != 'full_background_v1':
            raise ValueError(
                'natural A1-S Background requires full_background_v1 scope')
        if config.get('background_condition_adapter', 'disabled') != 'disabled':
            raise ValueError(
                'natural A1-S Background does not allow condition adapters')
        if config.get('background_fixed_updates_per_epoch') is not None:
            raise ValueError(
                'natural A1-S Background requires no fixed update budget')
        if not bool(config.get('background_only', False)):
            raise ValueError(
                'natural A1-S run is Background-only in this experiment')
        if str(config.get('m00_observation_calibration_semantics',
                          'disabled')) != 'disabled':
            raise ValueError(
                'natural A1-S run requires M00 calibration to be disabled')
        if config.get('background_checkpoint_selection') not in (
                _M00_NATURAL_SELECTOR, _M00_NIGHT_GENERALIZATION_SELECTOR,
                _M00_NIGHT_GENERALIZATION_V2_SELECTOR):
            raise ValueError(
                'natural A1-S run requires a shared M00 selector')
        if not config.get('m00_selection_reference_history'):
            raise ValueError(
                'natural A1-S run requires a historical M00 selection history')
    if _a1s8_enabled(config):
        if config.get('background_condition_adapter') not in (
                _A1S7_ADAPTER_SEMANTICS, _A1S7_ADAPTER_ALIAS):
            raise ValueError('A1-S8 requires the dual magnetic-latitude adapters')
        if config.get('background_training_scope') != _A1S7_ADAPTER_SCOPE:
            raise ValueError('A1-S8 requires adapter-only Background scope')
        if str(config.get('background_adapter_gate_semantics', '')).replace(
                '-', '_') != _A1S7_GATE_SEMANTICS:
            raise ValueError('A1-S8 gate semantics mismatch')
        if str(config.get('background_gate_balance_semantics', '')).replace(
                '-', '_') != _A1S8_GATE_BALANCE_SEMANTICS:
            raise ValueError('A1-S8 gate-balance semantics mismatch')
        if str(config.get('background_anchor_semantics', '')).replace(
                '-', '_') != _A1S8_ANCHOR_SEMANTICS:
            raise ValueError('A1-S8 anchor semantics mismatch')
        if str(config.get('background_source_mix_schedule', '')).replace(
                '-', '_') != _A1S8_SOURCE_MIX_SEMANTICS:
            raise ValueError('A1-S8 source-mix semantics mismatch')
        if int(config.get('background_fixed_updates_per_epoch', 0)) != 960:
            raise ValueError('A1-S8 requires exactly 960 updates per epoch')
        if not bool(config.get('background_only', False)):
            raise ValueError('A1-S8 is Background-only; Analysis/M11 is deferred')
    if _a1s9_enabled(config):
        if config.get('background_condition_adapter') not in (
                _A1S7_ADAPTER_SEMANTICS, _A1S7_ADAPTER_ALIAS):
            raise ValueError('A1-S9 requires the dual magnetic-latitude adapters')
        if config.get('background_training_scope') != _A1S9_ADAPTER_SCOPE:
            raise ValueError('A1-S9 requires adapter-only Background scope')
        if float(config.get('background_adapter_residual_cap', 0.0)) > 0.05:
            raise ValueError('A1-S9 adapter cap must be at most 0.05 dex')
        if float(config.get('background_anchor_gradient_target', 0.0)) != 0.20:
            raise ValueError('A1-S9 requires anchor target 0.20')
        if int(config.get('background_fixed_updates_per_epoch', 0)) != 960:
            raise ValueError('A1-S9 requires exactly 960 updates per epoch')
        if config.get('background_checkpoint_selection') != 'bias_constrained_v1':
            raise ValueError('A1-S9 requires the strict A1-S selector')
        if not bool(config.get('background_only', False)):
            raise ValueError('A1-S9 is Background-only; Analysis/M11 is deferred')
    observation_semantics = str(config.get(
        'm00_observation_calibration_semantics', 'disabled'))
    m00_observation_route = observation_semantics != 'disabled'
    m00_candidate_semantics = str(config.get('m00_candidate_semantics', 'disabled'))
    if m00_candidate_semantics in _M00_CANDIDATE_SCOPES:
        if not bool(config.get('background_only', False)):
            raise ValueError('M00 candidate route is Background-only; Analysis/M11 is deferred')
        if int(config.get('background_epochs', 5)) != (1 if config.get('smoke_run', False) else 5):
            raise ValueError('M00 candidate route requires five Background epochs')
        if config.get('analysis_epochs') not in (0, None):
            raise ValueError('M00 candidate route does not train Analysis')
        seed = config.get('background_seed_ckpt') or config.get('m00_observation_seed_checkpoint')
        if not seed or not os.path.isfile(seed):
            raise FileNotFoundError('M00 candidate route requires the A1-S epoch-3 Background seed')
    elif m00_candidate_semantics != 'disabled':
        raise ValueError(f'unsupported M00 candidate semantics: {m00_candidate_semantics}')
    if observation_semantics != 'disabled':
        if config.get('m00_observation_training_scope', 'disabled') \
                != _M00_OBSERVATION_SCOPE:
            raise ValueError(
                'M00 observation route requires m00_observation_error_only_v1 scope')
        if not bool(config.get('background_only', False)):
            raise ValueError('M00 observation route cannot enter Analysis/M11')
        expected_background_epochs = (
            1 if bool(config.get('smoke_run', False)) else 5)
        if int(config.get('background_epochs', 5)) != expected_background_epochs:
            raise ValueError(
                'M00 observation route requires five Background epochs '
                '(one in smoke mode)')
        allowed_analysis_epochs = (0, None, 1, 10) if bool(
            config.get('smoke_run', False)) else (0, None, 10)
        if config.get('analysis_epochs') not in allowed_analysis_epochs:
            raise ValueError('M00 observation route does not train Analysis')
        seed = config.get('m00_observation_seed_checkpoint') or config.get(
            'background_seed_ckpt')
        if not seed or not os.path.isfile(seed):
            raise FileNotFoundError(
                'M00 observation route requires an A1-S Background seed')
        if config.get('background_training_semantics') != (
                _M00_OBSERVATION_BACKGROUND_SEMANTICS):
            raise ValueError(
                'M00 observation route requires its explicit Background semantics')
    causal_history_analysis_only = (
        config.get('analysis_training_scope')
        == _CAUSAL_HISTORY_ANALYSIS_SCOPE)
    raw_iri_analysis_only = (
        config.get('background_semantics') == 'raw_iri_fixed_v1'
        and config.get('background_training_scope') == 'analysis_only_v1')
    if raw_iri_analysis_only:
        # Keep the v14/A1-S2 5+10 bookkeeping contract while explicitly
        # skipping the Background optimizer loop.  Using zero here would
        # make a raw epoch count look like a legacy "no Background" run and
        # would also invalidate the hybrid checkpoint contract.
        if int(config.get('background_epochs', 5)) != 5:
            raise ValueError(
                'raw_iri_fixed_v1 Analysis-only runs must retain '
                'background_epochs=5 bookkeeping')
        if int(config.get('analysis_epochs', 10)) != 10:
            raise ValueError(
                'raw_iri_fixed_v1 Analysis-only runs require '
                'analysis_epochs=10')
        if config.get('background_residual_enabled', False):
            raise ValueError(
                'raw_iri_fixed_v1 Analysis-only runs require '
                'background residual disabled')
        if not config.get('analysis_seed_background'):
            raise ValueError(
                'raw_iri_fixed_v1 Analysis-only runs require '
                'analysis_seed_background')
    if causal_history_analysis_only:
        causal_background_anchor = str(
            config.get('causal_background_anchor', 'a1s2_m00_v1')).replace('-', '_')
        if causal_background_anchor not in _CAUSAL_BACKGROUND_ANCHORS:
            raise ValueError(
                'causal history requires causal_background_anchor in '
                f'{_CAUSAL_BACKGROUND_ANCHORS!r}')
        config['causal_background_anchor'] = causal_background_anchor
        expected = {
            'analysis_output_semantics': _CAUSAL_HISTORY_OUTPUT_SEMANTICS,
            'analysis_observation_time_support': _CAUSAL_PAST_TIME_SUPPORT,
        'r_source': (_CAUSAL_FROZEN_R_SOURCE, _A1S_FROZEN_R_SOURCE),
        }
        mismatched = {
            key: config.get(key) for key, value in expected.items()
            if (config.get(key) not in value if isinstance(value, tuple)
                else config.get(key) != value)}
        if mismatched:
            raise ValueError(
                f'causal history Analysis configuration mismatch: {mismatched}')
        if int(config.get('background_epochs', 5)) != 5:
            raise ValueError(
                'causal history Analysis-only runs retain background_epochs=5')
        if int(config.get('analysis_epochs', 10)) != 10:
            raise ValueError('causal history Analysis-only runs require 10 epochs')
        if not config.get('analysis_seed_background'):
            raise ValueError(
                'causal history Analysis-only runs require an A1-S Background')
        # Resume is allowed only from the persisted causal Analysis state;
        # external Background seeding remains rejected for this scope.
        if config.get('background_seed_ckpt'):
            raise ValueError(
                'causal history Analysis-only runs reject resume/background seed')
    adapter_semantics = config.get('background_condition_adapter', 'disabled')
    if adapter_semantics == _A1S7_ADAPTER_ALIAS:
        adapter_semantics = _A1S7_ADAPTER_SEMANTICS
        config['background_condition_adapter'] = adapter_semantics
    if adapter_semantics not in (
            'disabled', _COND_ADAPTER_SEMANTICS,
            _A1S7_ADAPTER_SEMANTICS):
        raise ValueError(
            f'unsupported Background condition-adapter semantics: {adapter_semantics}')
    if adapter_semantics == _COND_ADAPTER_SEMANTICS:
        adapter_scope = config.get('background_training_scope')
        if adapter_scope not in (
                _COND_ADAPTER_SCOPE, _COND_ADAPTER_JOINT_FRESH_SCOPE,
                _A1S7_ADAPTER_SCOPE):
            raise ValueError(
                'condition adapter requires a supported '
                'background_training_scope')
        cap = float(config.get('background_adapter_residual_cap', 0.1))
        if not np.isfinite(cap) or cap <= 0.0 or cap > 0.5:
            raise ValueError('background adapter residual cap must be in (0, 0.5]')
        if adapter_scope == _A1S7_ADAPTER_SCOPE:
            if adapter_semantics != _A1S7_ADAPTER_SEMANTICS:
                raise ValueError(
                    'A1-S7 adapter scope requires dual_maglat_smooth_v1')
            if config.get('background_checkpoint_selection') != (
                    'bias_constrained_v1'):
                raise ValueError(
                    'A1-S7 must retain bias_constrained_v1 selector')
        elif adapter_scope == _COND_ADAPTER_SCOPE:
            seed = config.get('background_adapter_seed_checkpoint')
            if not seed or not os.path.isfile(seed):
                raise FileNotFoundError(
                    'condition adapter requires background_adapter_seed_checkpoint')
            # Reuse the existing seed-migration path; it permits only missing
            # adapter keys and avoids restoring an optimizer/scheduler state.
            config['background_seed_ckpt'] = seed
        else:
            forbidden = [
                key for key in (
                    'background_seed_ckpt', 'background_adapter_seed_checkpoint',
                    'resume_ckpt') if config.get(key)]
            if forbidden:
                raise ValueError(
                    'condition_adapter_joint_fresh_v1 rejects seed/resume inputs: '
                    f'{forbidden}')
            if config.get('background_adapter_rng_seed') is None:
                raise ValueError(
                    'condition_adapter_joint_fresh_v1 requires '
                    'background_adapter_rng_seed')
            if config.get('background_checkpoint_selection') != (
                    _COND_ADAPTER_JOINT_FRESH_SELECTOR):
                raise ValueError(
                    'joint fresh adapter requires background_checkpoint_selection='
                    f'{_COND_ADAPTER_JOINT_FRESH_SELECTOR!r}')
    # A1-S7 is deliberately a Background-only migration from the selected
    # A1-S Background weights. Keep its date split/selector contract explicit
    # so a later resume cannot silently become A1-S6/A1-S2.
    if _a1s7_enabled(config):
        if adapter_semantics != _A1S7_ADAPTER_SEMANTICS:
            raise ValueError('A1-S7 requires dual_maglat_smooth_v1 adapter')
        if config.get('background_training_scope') != _A1S7_ADAPTER_SCOPE:
            raise ValueError('A1-S7 requires adapter-only dual scope')
        if _profile_split_enabled(config):
            raise ValueError('A1-S7 must use the A1-S date split, not profile split')
        expected_background_epochs = (
            1 if bool(config.get('smoke_run', False)) else 5)
        if int(config.get('background_epochs', 5)) != expected_background_epochs:
            raise ValueError(
                f'A1-S7 requires exactly {expected_background_epochs} '
                'Background epochs')
        if not bool(config.get('background_only', False)):
            raise ValueError('A1-S7 does not train Analysis/M11')
        if config.get('background_checkpoint_selection') != 'bias_constrained_v1':
            raise ValueError('A1-S7 must use bias_constrained_v1 selector')
        if config.get('background_adapter_gate_semantics',
                       _A1S7_GATE_SEMANTICS) != _A1S7_GATE_SEMANTICS:
            raise ValueError('A1-S7 gate semantics mismatch')
        if config.get('background_sampling_semantics',
                       _A1S7_SAMPLING_SEMANTICS) != _A1S7_SAMPLING_SEMANTICS:
            raise ValueError('A1-S7 sampling semantics mismatch')
        cap = float(config.get('background_adapter_residual_cap', 0.1))
        if not np.isfinite(cap) or cap <= 0.0 or cap > 0.5:
            raise ValueError('A1-S7 adapter residual cap must be in (0, 0.5]')
        seed = config.get('background_adapter_seed_checkpoint')
        if not seed or not os.path.isfile(seed):
            raise FileNotFoundError(
                'A1-S7 requires background_adapter_seed_checkpoint')
        # Weight-only migration: _load_background_seed ignores optimizer and
        # scheduler state and permits the new zero-initialized adapter keys.
        config['background_seed_ckpt'] = seed
        config['checkpoint_semantic'] = _A1S7_CHECKPOINT_SEMANTICS

    shape_semantics = config.get('background_shape_loss', 'disabled')
    if config.get('background_bias_grouping') == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2:
        legacy_weight = config.get('resolved_background_stratum_bias_weight')
        if legacy_weight is None:
            raise ValueError('v2 requires the frozen legacy Background Bias weight')
        if not np.isclose(float(legacy_weight), 0.01226901322277942,
                          rtol=0.0, atol=1e-15):
            raise ValueError('v2 legacy Background Bias weight identity mismatch')
        target = float(config.get('background_magnetic_daynight_gradient_target', 0.0125))
        if not 0.0 < target <= 1.0:
            raise ValueError('v2 magnetic/daynight gradient target must be in (0, 1]')
        if int(config.get('background_magnetic_daynight_calibration_batches', 20)) < 1:
            raise ValueError('v2 magnetic/daynight calibration batches must be positive')
    if shape_semantics not in ('disabled', _CONDITIONAL_SHAPE_LOSS):
        raise ValueError(f'unsupported Background shape loss: {shape_semantics}')
    if shape_semantics == _CONDITIONAL_SHAPE_LOSS:
        if config.get('background_checkpoint_selection') != _CONDITIONAL_SHAPE_SELECTOR:
            raise ValueError(
                'conditional shape loss requires conditional_shape_constrained_v1')
        if not 0.0 < float(config.get(
                'background_shape_gradient_target', 0.02)) <= 1.0:
            raise ValueError('background_shape_gradient_target must be in (0, 1]')
        if int(config.get('background_shape_calibration_batches', 20)) < 1:
            raise ValueError('background_shape_calibration_batches must be positive')
        if int(config.get('background_shape_min_profiles_per_cell', 8)) < 1:
            raise ValueError('background_shape_min_profiles_per_cell must be positive')
    if config.get('analysis_state_semantics', 'legacy_feature_increment') != (
            'legacy_feature_increment'):
        raise ValueError(
            'query-local physical states are failed audit shadows; '
            'M2-O legacy_feature_increment is the only trainable model')
    if config.get('r_mode') != 'global':
        raise ValueError('density-observation ETKF requires r_mode=global')
    if not config.get('use_distance_localization', False):
        raise ValueError(
            'density-observation ETKF requires continuous localization')
    if config.get('assimilation_semantics') == 'continuous_physical_local_letkf':
        if not config.get('use_physical_localization', False):
            raise ValueError('M2-V requires physical Gaspari-Cohn localization')
        if config.get('representativeness_kernel_path'):
            raise ValueError('M2-V must not use source-labelled representativeness tables')
        if config.get('use_empirical_covariance_loss', False):
            raise ValueError('M2-V must not use empirical covariance supervision')
        if config.get('neighbor_directory_semantics') != (
                'token_exact_positive_support_v1'):
            raise ValueError('M2-V requires the exact token directory')
        if int(config.get('observation_chunk_size', 4096)) < 1:
            raise ValueError('M2-V observation_chunk_size must be positive')
        model_domain = config.get(
            'model_domain_semantics', 'legacy_120_500_domain_v1')
        if model_domain not in (
                'legacy_120_500_domain_v1', 'strict_200_500_domain_v1',
                *_STRICT_LOW_PRIOR_DOMAINS):
            raise ValueError('unrecognized model-domain semantics')
        expected_alt_range = {
            'legacy_120_500_domain_v1': (120.0, 500.0),
            'strict_200_500_domain_v1': (200.0, 500.0),
            _HYBRID_DOMAIN_SEMANTICS: (120.0, 500.0),
            _A2S1_DOMAIN_SEMANTICS: (120.0, 500.0),
        }[model_domain]
        if tuple(map(float, config.get('alt_range', ()))) != expected_alt_range:
            raise ValueError('model-domain semantics and alt_range disagree')
        expected_observation_range = {
            'legacy_120_500_domain_v1': (120.0, 500.0),
            'strict_200_500_domain_v1': (200.0, 500.0),
            _HYBRID_DOMAIN_SEMANTICS: (200.0, 500.0),
            _A2S1_DOMAIN_SEMANTICS: (120.0, 500.0),
        }[model_domain]
        expected_peak_range = {
            'legacy_120_500_domain_v1': (120.0, 500.0),
            'strict_200_500_domain_v1': (200.0, 500.0),
            _HYBRID_DOMAIN_SEMANTICS: (200.0, 500.0),
            _A2S1_DOMAIN_SEMANTICS: (200.0, 500.0),
        }[model_domain]
        if tuple(map(float, config.get('observation_alt_range') or ())) != expected_observation_range:
            raise ValueError('model-domain semantics and observation_alt_range disagree')
        if tuple(map(float, config.get('peak_search_alt_range') or ())) != expected_peak_range:
            raise ValueError('model-domain semantics and peak_search_alt_range disagree')
        expected_format = {
            'legacy_120_500_domain_v1': 12,
            'strict_200_500_domain_v1': 13,
            _HYBRID_DOMAIN_SEMANTICS: 14,
            _A2S1_DOMAIN_SEMANTICS: 15,
        }[model_domain]
        if int(config.get('checkpoint_format_version', 0)) != expected_format:
            raise ValueError('model-domain semantics and checkpoint format disagree')
        if not config.get('eval_only'):
            if model_domain == _A2S1_DOMAIN_SEMANTICS:
                expected_background_semantics = (
                    'qc_v2_date_blocked_train_only_a2s1_120_500_low_alt_prior_v1')
            elif model_domain == _HYBRID_DOMAIN_SEMANTICS:
                expected_background_semantics = (
                    'qc_v2_date_blocked_train_only_m2w_120_500_obs_200_500_low_alt_prior_v1')
            elif model_domain == 'strict_200_500_domain_v1':
                expected_background_semantics = (
                    'qc_v2_date_blocked_train_only_m2w_200_500_continuous_trust_gate_v1'
                    if config.get('background_trust_gate_enabled', False)
                    else 'qc_v2_date_blocked_train_only_m2w_200_500_v1')
            else:
                expected_background_semantics = (
                    'qc_v2_date_blocked_train_only_continuous_trust_gate_v1'
                    if config.get('background_trust_gate_enabled', False)
                    else 'qc_v2_date_blocked_train_only')
        profile_split_background = (
            model_domain == _HYBRID_DOMAIN_SEMANTICS
            and _profile_split_enabled(config)
            and config.get('background_training_semantics')
            == _A1S2_BACKGROUND_SEMANTICS)
        raw_iri_background = (
            model_domain == _HYBRID_DOMAIN_SEMANTICS
            and raw_iri_analysis_only
            and config.get('background_training_semantics')
            == 'raw_iri_fixed_v1')
        if (not config.get('eval_only')
                and config.get('background_training_semantics')
                != expected_background_semantics
                and not (profile_split_background or raw_iri_background
                         or _natural_background_enabled(config)
                         or _a1s7_enabled(config) or _a1s8_enabled(config)
                         or m00_observation_route)):
            raise ValueError(
                'M2-V training requires the configured QC-v2 Background semantics')
    if config.get('model_domain_semantics') == _A2S1_DOMAIN_SEMANTICS:
        _validate_hybrid_low_altitude_protocol(config)
    if (config.get('background_seed_ckpt')
            and adapter_semantics not in (_COND_ADAPTER_SEMANTICS,
                                          _A1S7_ADAPTER_SEMANTICS)
            and not m00_observation_route
            and not raw_iri_analysis_only):
            raise ValueError(
                'QC-v2 Background training cannot use an external seed checkpoint')
    if config.get('background_trust_gate_enabled', False):
        if config.get('background_trust_gate_semantics') != (
                'fixed_altitude_localtime_dip_smoothstep_v1'):
            raise ValueError('M2-V trust-gate semantic is not recognized')
        _validate_hybrid_low_altitude_protocol(config)
    if (config.get('use_direction_loss', False)
            and not config.get('analysis_exact_mode_loss', False)):
        raise ValueError(
            'direction loss requires analysis_exact_mode_loss')
    if (config.get('use_covariance_moment_loss', False)
            and config.get('use_empirical_covariance_loss', False)):
        raise ValueError('choose only one covariance supervision loss')
    if (config.get('use_empirical_covariance_loss', False)
            and not config.get('representativeness_kernel_path')):
        raise ValueError(
            'empirical covariance loss requires train-only covariance cells')
    if config.get('use_observation_gram_loss', False):
        required_gram = {
            'analysis_exact_mode_loss': True,
            'basis_dim': 64,
            'enkf_n_members': 8,
            'enkf_anomaly_parameterization': 'orthogonal_factor',
            'density_basis_semantics': 'endpoint_context_symmetric',
        }
        mismatched = {
            key: (config.get(key), expected)
            for key, expected in required_gram.items()
            if config.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                f'observation Gram loss requires the D64/N8 endpoint basis: {mismatched}')
        if config.get('analysis_state_semantics', 'legacy_feature_increment') != (
                'legacy_feature_increment'):
            raise ValueError('observation Gram loss cannot train physical coefficient states')
        if not 0.0 < float(config.get('gram_gradient_target', 0.02)) <= 1.0:
            raise ValueError('gram_gradient_target must be in (0, 1]')
        if int(config.get('gram_calibration_batches', 20)) < 1:
            raise ValueError('gram_calibration_batches must be positive')
    device = torch.device(config['device'])
    _reset_random_seeds(config['seed'], device)

    background_epochs = int(config.get('background_epochs', 5))
    analysis_epochs = int(config.get('analysis_epochs', 5))
    background_only = bool(config.get('background_only', False))
    selection_semantics = config.get('background_checkpoint_selection', 'legacy')
    m00_selection_baseline = None
    if selection_semantics == _M00_NATURAL_SELECTOR:
        m00_selection_baseline = _load_m00_natural_selection_baseline(config)
        background_selection_baseline = None
    else:
        background_selection_baseline = _load_background_selection_baseline(config)
    total_epochs = (background_epochs if background_only
                    else background_epochs + analysis_epochs)
    if (background_epochs <= 0 and not (
            raw_iri_analysis_only or causal_history_analysis_only)) or (
            not background_only and analysis_epochs <= 0):
        raise ValueError(
            'background_epochs and analysis_epochs must be positive for full training')
    architecture = _architecture_signature(config)
    training_protocol = _training_protocol_signature(config)
    format_version = int(config.get(
        'checkpoint_format_version',
        8 if architecture['enkf_anomaly_parameterization']
        == 'orthogonal_factor' else 7))

    date_split, date_split_identity = _load_or_create_date_split(config)
    if date_split_identity is not None:
        config['resolved_date_split'] = date_split_identity
    # Recompute immutable input identities before any model state is restored.
    input_data_sha256 = _resolve_input_data_sha256(config)
    config['input_data_sha256'] = input_data_sha256

    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device,
    )
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(
        config['iri_proxy_path'], map_location=device, weights_only=True))
    iri_proxy.eval()
    iri_peak_manager = _load_iri_peak_manager(config, device)

    # Keep dataset constructors from mutating the immutable split identity.
    # Some legacy loaders normalize/filter their ``split_days`` lists in place.
    loader_split_days = (
        {
            'train': list(date_split['train']),
            'development': list(date_split['development']),
        }
        if date_split is not None else None)
    loader_kwargs = dict(
        batch_size=config['batch_size'],
        bin_size_hours=config['bin_size_hours'],
        num_workers=config.get('num_workers', 0),
        use_memmap=config.get('use_memmap', True),
        val_ratio=(
            None if date_split is not None
            else config.get('val_ratio', 0.1)),
        split_seed=config['seed'],
        points_per_profile=config.get('profile_points_per_epoch', 8),
        full_validation_profiles=False,
        split_days=loader_split_days,
        alt_range=_observation_alt_range(config),
        sampling_semantics=config.get(
            'background_sampling_semantics',
            'legacy_profile_time_bin_v1'),
        schedule_seed=int(config.get(
            'background_profile_schedule_seed', config.get('seed', 42))),
        max_batches=config.get('background_fixed_updates_per_epoch'),
    )
    schedule_seed = int(loader_kwargs['schedule_seed'])
    fy_schedule_seed = int(config.get(
        'background_profile_schedule_seed_fy', schedule_seed))
    cosmic_schedule_seed = int(config.get(
        'background_profile_schedule_seed_cosmic', schedule_seed + 1))
    loader_kwargs['schedule_seed'] = fy_schedule_seed
    if _profile_split_enabled(config):
        (
            train_loader,
            val_loader,
            cosmic_train_loader,
            cosmic_val_loader,
        ) = _load_profile_split_loaders(config)
    else:
        fy_loader_kwargs = dict(
            loader_kwargs,
        magnetic_sidecar_path=(config.get('m00_observation_qd_sidecar_fy') or
                               config.get('m00_observation_magnetic_sidecar_fy')),
            magnetic_sidecar_source='FY',
            magnetic_sidecar_strict=bool(config.get(
                'm00_observation_strict_magnetic', False)),
        )
        cosmic_loader_kwargs = dict(
            loader_kwargs,
            schedule_seed=cosmic_schedule_seed,
        magnetic_sidecar_path=(config.get('m00_observation_qd_sidecar_cosmic') or
                               config.get('m00_observation_magnetic_sidecar_cosmic')),
            magnetic_sidecar_source='COSMIC',
            magnetic_sidecar_strict=bool(config.get(
                'm00_observation_strict_magnetic', False)),
        )
        train_loader, val_loader = get_dataloaders(
            npy_path=config['fy_path'],
            profile_path=config.get('fy_profile_path'),
            profile_index_path=config.get('fy_profile_index_path'),
            **fy_loader_kwargs,
        )
        cosmic_train_loader, cosmic_val_loader = get_cosmic_dataloader(
            cosmic_path=config['cosmic_path'],
            profile_index_path=config.get('cosmic_profile_index_path'),
            **cosmic_loader_kwargs,
        )
    if date_split is not None and date_split_identity is not None:
        expected_partitions = {
            'train': list(date_split['train']),
            'development': list(date_split['development']),
            'locked_test': list(date_split.get('locked_test', [])),
        }
        if date_split_identity.get('partitions') != expected_partitions:
            raise RuntimeError(
                'date split identity changed while constructing loaders')
    _validate_a2s1_loader_lengths(
        config, train_loader, cosmic_train_loader)
    if _a1s7_enabled(config):
        fixed_updates = int(config['background_fixed_updates_per_epoch'])
        if len(train_loader) != fixed_updates or len(cosmic_train_loader) != fixed_updates:
            raise RuntimeError(
                'A1-S7 loader update count differs from the fixed A1-S budget')
    if _natural_background_enabled(config) and config.get(
            'background_fixed_updates_per_epoch') is not None:
        raise RuntimeError(
            'natural Background cannot be combined with a fixed update budget')
    schedule_natural = _natural_background_enabled(config)
    config['background_loader_schedule'] = {
        'fy_batches': int(len(train_loader)),
        'cosmic_batches': int(len(cosmic_train_loader)),
        'background_epoch_batches': int(
            len(train_loader) + len(cosmic_train_loader)
            if schedule_natural else
            max(len(train_loader), len(cosmic_train_loader))),
        'pairing': ('source_complete_pass_proportional_interleave_v1'
                    if schedule_natural else
                    'paired_equal_fixed_budget_v1' if _a1s7_enabled(config)
                    else 'cycle_shorter_source_max_loader_length'),
        'sampling_semantics': config.get(
            'background_sampling_semantics', 'legacy_profile_time_bin_v1'),
        'schedule_seed': int(config.get(
            'background_profile_schedule_seed', config.get('seed', 42))),
        'schedule_seed_fy': int(config.get(
            'background_profile_schedule_seed_fy', config.get(
                'background_profile_schedule_seed', config.get('seed', 42)))),
        'schedule_seed_cosmic': int(config.get(
            'background_profile_schedule_seed_cosmic',
            int(config.get('background_profile_schedule_seed',
                           config.get('seed', 42))) + 1)),
        'fixed_updates_per_epoch': config.get(
            'background_fixed_updates_per_epoch'),
        'epoch_semantics': (
            'source_complete_profile_pass_v1' if schedule_natural
            else 'paired_or_cycled_loader_v1'),
    }
    config['model_domain_data_summary'] = {
        'model_alt_range_km': [float(value) for value in config['alt_range']],
        'observation_alt_range_km': [
            float(value) for value in _observation_alt_range(config)],
        'peak_search_alt_range_km': [float(value) for value in (
            config.get('peak_search_alt_range') or config['alt_range'])],
        'development_sampling': 'stable_8_points_per_profile',
        'FY': {
            'train_points': int(len(train_loader.dataset)),
            'development_points': int(len(val_loader.dataset)),
            'all_domain_points': int(train_loader.dataset.domain_row_count),
            'all_domain_profiles': int(
                train_loader.dataset.domain_profile_count),
        },
        'COSMIC': {
            'train_points': int(len(cosmic_train_loader.dataset)),
            'development_points': int(len(cosmic_val_loader.dataset)),
            'all_domain_points': int(cosmic_train_loader.dataset.domain_row_count),
            'all_domain_profiles': int(
                cosmic_train_loader.dataset.domain_profile_count),
        },
    }
    if str(config.get('date_split_semantics', '')).replace('-', '_') == _A1S6_DATE_SPLIT_SEMANTICS:
        audit_path = os.path.join(config['save_dir'], 'a1s6_date_split_audit.json')
        if os.path.isfile(audit_path):
            with open(audit_path, encoding='utf-8') as stream:
                audit = json.load(stream)
            audit['sources'] = {
                'FY': {'train_profiles': int(len(_loader_profile_ids(train_loader))),
                       'development_profiles': int(len(_loader_profile_ids(val_loader))),
                       'train_points': int(len(train_loader.dataset)),
                       'development_points': int(len(val_loader.dataset))},
                'COSMIC': {'train_profiles': int(len(_loader_profile_ids(cosmic_train_loader))),
                           'development_profiles': int(len(_loader_profile_ids(cosmic_val_loader))),
                           'train_points': int(len(cosmic_train_loader.dataset)),
                           'development_points': int(len(cosmic_val_loader.dataset))},
            }
            with open(audit_path + '.tmp', 'w', encoding='utf-8') as stream:
                json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write('\n')
            os.replace(audit_path + '.tmp', audit_path)
    print(f"[Background批次覆盖] {config['background_loader_schedule']}")
    subset_fraction = float(config.get('train_profile_fraction', 1.0))
    subset_manifest = config.get('profile_subset_manifest')
    profile_subset = {
        'FY': _restrict_training_profiles(
            train_loader, subset_fraction, config['seed'], 'FY',
            subset_manifest),
        'COSMIC': _restrict_training_profiles(
            cosmic_train_loader, subset_fraction, config['seed'], 'COSMIC',
            subset_manifest),
    }
    _resolve_background_bias_support_counts(
        train_loader, cosmic_train_loader, config)
    profile_subset_identity = (
        {
            'path': os.path.abspath(subset_manifest),
            'sha256': (None if config.get('skip_hash_verification', False)
                       else _sha256_file(subset_manifest)),
        }
        if subset_manifest and os.path.isfile(subset_manifest) else None)
    print(f'[训练profile子集] {profile_subset}')
    allowed_profiles = {
        'train': {
            'FY': _loader_profile_ids(train_loader),
            'COSMIC': _loader_profile_ids(cosmic_train_loader),
        },
        'development': {
            'FY': _loader_profile_ids(val_loader),
            'COSMIC': _loader_profile_ids(cosmic_val_loader),
        },
    }
    if np.intersect1d(
            allowed_profiles['train']['FY'],
            allowed_profiles['development']['FY']).size:
        raise ValueError('FY train/development profile leakage')
    if np.intersect1d(
            allowed_profiles['train']['COSMIC'],
            allowed_profiles['development']['COSMIC']).size:
        raise ValueError('COSMIC train/development profile leakage')
    allowed_profile_summary = {
        split_name: {
            source: int(len(profile_ids))
            for source, profile_ids in sources.items()}
        for split_name, sources in allowed_profiles.items()}
    print(f'[邻域profile隔离] {allowed_profile_summary}')
    if _profile_split_enabled(config):
        # Development targets remain held out, while Analysis neighbor
        # tokens are restricted to Train profiles only.
        query_profile_partitions = {
            'train': allowed_profiles['train'],
            'development': allowed_profiles['train'],
        }
    else:
        query_profile_partitions = (
            allowed_profiles if date_split is not None
            else {'train': None, 'development': None})
    covariance_strata = {
        'FY': _training_stratum_weights(train_loader),
        'COSMIC': _training_stratum_weights(cosmic_train_loader),
    }
    config['covariance_stratum_weights'] = {
        source: report['weights']
        for source, report in covariance_strata.items()}
    print(f'[协方差矩分层] {covariance_strata}')

    fy_nb_index = FYNeighborhoodIndex(config['fy_path'], config)
    cosmic_nb_index = COSMICNeighborhoodIndex(config['cosmic_path'], config)
    representativeness_path = config.get('representativeness_kernel_path')
    representativeness_floor = float(
        config.get('representativeness_floor', 0.25))
    representativeness_kernel = load_representativeness_kernel(
        representativeness_path)
    empirical_covariance_targets = (
        load_empirical_covariance_targets(representativeness_path)
        if config.get('use_empirical_covariance_loss', False) else None)
    representativeness_identity = (
        {
            'path': os.path.abspath(representativeness_path),
            'sha256': (None if config.get('skip_hash_verification', False)
                       else _sha256_file(representativeness_path)),
            'floor': representativeness_floor,
            'stable_cells': int(representativeness_kernel.sum().item()),
            'total_cells': int(representativeness_kernel.numel()),
        }
        if representativeness_kernel is not None else None)
    config['resolved_representativeness_kernel'] = (
        representativeness_identity)
    batch_processor = SlidingWindowBatchProcessor(
        sw_manager, device=device,
        fy_nb_index=fy_nb_index, cosmic_nb_index=cosmic_nb_index,
        representativeness_kernel=representativeness_kernel,
        representativeness_floor=representativeness_floor,
        empirical_covariance_targets=empirical_covariance_targets)

    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    _record_night_v2_execution(config, 'model_initialized', seed=int(config.get('seed', 42)),
                               background_seed_checkpoint=config.get('background_seed_ckpt'),
                               resume_checkpoint=config.get('resume_ckpt'))
    if (background_selection_baseline is None
            and config.get('background_checkpoint_selection')
            == _CONDITIONAL_SHAPE_SELECTOR):
        background_selection_baseline = _replay_conditional_shape_baseline(
            model, val_loader, cosmic_val_loader, batch_processor, device, config,
            sw_manager, iri_peak_manager, query_profile_partitions['development'])
    background_seed = config.get('background_seed_ckpt')
    if (background_seed is None and not (
            raw_iri_analysis_only or causal_history_analysis_only)
            and not config.get('eval_only')):
        _assert_fresh_background_identity(
            model, sw_manager, iri_peak_manager, device)
    resume_state = None
    start_epoch = 0
    history = []
    best_scores = {'background': float('inf'), 'analysis': float('inf')}
    best_epochs = {'background': None, 'analysis': None}
    best_selection_keys = {'background': None, 'analysis': None}
    resume_path = config.get('resume_ckpt')
    if resume_path:
        loaded = torch.load(resume_path, map_location=device, weights_only=False)
        if loaded.get('checkpoint_type') != 'run66_training_state':
            if not config.get('eval_only'):
                raise ValueError('run66 cannot resume an older architecture checkpoint')
            model.load_state_dict(loaded, strict=True)
        else:
            if config.get('model_domain_semantics') in _STRICT_LOW_PRIOR_DOMAINS:
                _validate_v14_resume_state(
                    loaded, training_protocol, input_data_sha256)
            if int(loaded.get('format_version', 0)) != format_version:
                raise ValueError(
                    f'checkpoint format does not match expected v{format_version}')
            checkpoint_background_semantics = loaded.get(
                'background_training_semantics')
            if checkpoint_background_semantics != config.get(
                    'background_training_semantics'):
                raise ValueError(
                    'checkpoint Background training semantics differ from config')
            checkpoint_bias_grouping = loaded.get(
                'background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY)
            if checkpoint_bias_grouping != config.get(
                    'background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY):
                raise ValueError(
                    'checkpoint Background Bias grouping differs from config')
            if checkpoint_bias_grouping == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2:
                checkpoint_weight = loaded.get('resolved_background_magnetic_daynight_weight')
                config_weight = config.get('resolved_background_magnetic_daynight_weight')
                if (checkpoint_weight is None or config_weight is not None
                        and not np.isclose(float(checkpoint_weight), float(config_weight),
                                           rtol=0.0, atol=1e-15)):
                    raise ValueError('checkpoint v2 magnetic/daynight weight differs from config')
            if loaded.get('m00_observation_calibration_semantics', 'disabled') != (
                    config.get('m00_observation_calibration_semantics', 'disabled')):
                raise ValueError(
                    'checkpoint M00 observation semantics differ from config')
            if loaded.get('m00_observation_training_scope', 'disabled') != (
                    config.get('m00_observation_training_scope', 'disabled')):
                raise ValueError(
                    'checkpoint M00 observation scope differs from config')
            if loaded.get('m00_observation_stage', 'bias_r') != (
                    config.get('m00_observation_stage', 'bias_r')):
                raise ValueError(
                    'checkpoint M00 observation stage differs from config')
            if loaded.get('m00_observation_seed_checkpoint') != (
                    config.get('m00_observation_seed_checkpoint')):
                raise ValueError(
                    'checkpoint M00 observation seed differs from config')
            if int(loaded['background_epochs']) != background_epochs:
                raise ValueError('checkpoint background_epochs differs from config')
            if (config.get('model_domain_semantics') in _STRICT_LOW_PRIOR_DOMAINS
                    and int(loaded.get('analysis_epochs', -1)) != analysis_epochs):
                raise ValueError('v14 checkpoint analysis_epochs differs from config')
            checkpoint_r_mode = loaded.get('r_mode')
            if (checkpoint_r_mode is not None
                    and checkpoint_r_mode != config.get('r_mode', 'global')):
                raise ValueError('checkpoint r_mode differs from config')
            checkpoint_localization = loaded.get('use_distance_localization')
            if (checkpoint_localization is not None
                    and bool(checkpoint_localization) != bool(
                        config.get('use_distance_localization', False))):
                raise ValueError(
                    'checkpoint distance localization differs from config')
            checkpoint_source_schedule = loaded.get('source_mode_schedule')
            if (checkpoint_source_schedule is not None
                    and checkpoint_source_schedule != config.get(
                        'source_mode_schedule', 'random_profile')):
                raise ValueError(
                    'checkpoint source mode schedule differs from config')
            if bool(loaded.get('analysis_exact_mode_loss', False)) != bool(
                    config.get('analysis_exact_mode_loss', False)):
                raise ValueError(
                    'checkpoint exact mode loss differs from config')
            if bool(loaded.get(
                    'use_empirical_covariance_loss', False)) != bool(
                    config.get('use_empirical_covariance_loss', False)):
                raise ValueError(
                    'checkpoint empirical covariance loss differs from config')
            if bool(loaded.get('use_direction_loss', False)) != bool(
                    config.get('use_direction_loss', False)):
                raise ValueError(
                    'checkpoint direction loss differs from config')
            if bool(loaded.get('use_observation_gram_loss', False)) != bool(
                    config.get('use_observation_gram_loss', False)):
                raise ValueError(
                    'checkpoint observation Gram loss differs from config')
            if loaded.get('representativeness_kernel') != (
                    representativeness_identity):
                raise ValueError(
                    'checkpoint representativeness kernel differs from config')
            checkpoint_architecture = loaded.get('architecture')
            if checkpoint_architecture is not None:
                checkpoint_architecture = dict(checkpoint_architecture)
                if config.get('model_domain_semantics') not in _STRICT_LOW_PRIOR_DOMAINS:
                    checkpoint_architecture.setdefault(
                        'density_basis_semantics', 'query_conditioned')
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_enabled', False)
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_semantics', 'disabled')
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_altitude_core_km', 200.0)
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_altitude_transition_km', 100.0)
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_night_cosine_offset', 0.2)
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_night_cosine_scale', 0.2)
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_dip_core', 0.25)
                    checkpoint_architecture.setdefault(
                        'background_trust_gate_dip_transition', 0.25)
                    checkpoint_architecture.setdefault(
                        'background_decoder_conditioning', 'legacy_v1')
                    checkpoint_architecture.setdefault(
                        'alt_range', [120.0, 500.0])
                    checkpoint_architecture.setdefault(
                        'observation_alt_range', checkpoint_architecture['alt_range'])
                    checkpoint_architecture.setdefault(
                        'peak_search_alt_range', checkpoint_architecture['alt_range'])
                    checkpoint_architecture.setdefault(
                        'checkpoint_format_version', format_version)
                    checkpoint_architecture.setdefault(
                        'model_domain_semantics', 'legacy_120_500_domain_v1')
            if (checkpoint_architecture is not None
                    and checkpoint_architecture != architecture):
                raise ValueError(
                    'checkpoint architecture differs from config: '
                    f'{checkpoint_architecture} != {architecture}')
            if loaded.get('date_split_identity') != date_split_identity:
                raise ValueError('checkpoint date split differs from config')
            if (_profile_split_enabled(config)
                    and loaded.get('profile_split_identity')
                    != config.get('resolved_profile_split')):
                raise ValueError('checkpoint profile split differs from config')
            model.load_state_dict(loaded['model_state_dict'], strict=True)
            start_epoch = int(loaded['completed_epochs'])
            history = list(loaded.get('history', []))
            best_scores.update(loaded.get('best_scores', {}))
            best_epochs.update(loaded.get('best_epochs', {}))
            best_selection_keys.update(loaded.get(
                'best_selection_keys', {}))
            if loaded.get('resolved_covariance_weight') is not None:
                config['resolved_covariance_weight'] = float(
                    loaded['resolved_covariance_weight'])
            if loaded.get('resolved_direction_weight') is not None:
                config['resolved_direction_weight'] = float(
                    loaded['resolved_direction_weight'])
            if loaded.get('resolved_gram_weight') is not None:
                config['resolved_gram_weight'] = float(
                    loaded['resolved_gram_weight'])
            if loaded.get('gram_gradient_calibration') is not None:
                config['gram_gradient_calibration'] = loaded[
                    'gram_gradient_calibration']
            if loaded.get('resolved_background_stratum_bias_weight') is not None:
                config['resolved_background_stratum_bias_weight'] = float(
                    loaded['resolved_background_stratum_bias_weight'])
                config['w_background_stratum_bias'] = float(
                    loaded['resolved_background_stratum_bias_weight'])
            if loaded.get('background_stratum_bias_gradient_calibration') is not None:
                config['background_stratum_bias_gradient_calibration'] = loaded[
                    'background_stratum_bias_gradient_calibration']
            if loaded.get('resolved_background_magnetic_daynight_weight') is not None:
                config['resolved_background_magnetic_daynight_weight'] = float(
                    loaded['resolved_background_magnetic_daynight_weight'])
            if loaded.get('background_magnetic_daynight_gradient_calibration') is not None:
                config['background_magnetic_daynight_gradient_calibration'] = loaded[
                    'background_magnetic_daynight_gradient_calibration']
            if loaded.get('resolved_background_shape_weight') is not None:
                config['resolved_background_shape_weight'] = float(
                    loaded['resolved_background_shape_weight'])
            if loaded.get('background_shape_gradient_calibration') is not None:
                config['background_shape_gradient_calibration'] = loaded[
                    'background_shape_gradient_calibration']
            config['background_update_steps'] = list(
                loaded.get('background_update_steps', []))
            resume_state = loaded
            if loaded.get('torch_rng_state') is not None:
                torch.set_rng_state(loaded['torch_rng_state'].cpu())
            if loaded.get('numpy_rng_state') is not None:
                np.random.set_state(loaded['numpy_rng_state'])
            if (device.type == 'cuda'
                    and loaded.get('cuda_rng_state_all') is not None):
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in loaded['cuda_rng_state_all']])

    if config.get('eval_only'):
        if background_only:
            config['m00_point_export_split'] = 'development'
            score, metrics = validate(
                model, val_loader, batch_processor, device, config,
                sw_manager, iri_peak_manager, cosmic_val_loader, 'background',
                query_profile_partitions['development'])
            replay = {
                'score': float(score),
                'metrics': metrics,
                'checkpoint': config.get('resume_ckpt'),
                'background_stratum_bias_gradient_target': float(config.get(
                    'background_stratum_bias_gradient_target', 0.0)),
                'background_bias_grouping': config.get(
                    'background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY),
                'background_bias_magnetic_support_counts': config.get(
                    'background_bias_magnetic_support_counts'),
            }
            if config.get('m00_diagnostic_train_split', False):
                config['m00_point_export_split'] = 'train'
                train_score, train_metrics = validate(
                    model, train_loader, batch_processor, device, config,
                    sw_manager, iri_peak_manager, cosmic_train_loader,
                    'background', query_profile_partitions['train'])
                replay['train_score'] = float(train_score)
                replay['train_metrics'] = train_metrics
                replay['train_profile_summary'] = {
                    source: int(len(query_profile_partitions['train'][source]))
                    for source in ('FY', 'COSMIC')
                }
            replay_path = os.path.join(
                config['save_dir'], 'background_development_replay.json')
            with open(replay_path, 'w', encoding='utf-8') as stream:
                json.dump(replay, stream, ensure_ascii=False, indent=2)
            print(f'  已写入 Background development 重放: {replay_path}')
        return (
            model, [], [], train_loader, val_loader,
            sw_manager, batch_processor, iri_peak_manager,
        )

    os.makedirs(config['save_dir'], exist_ok=True)
    deferred_contract_path = _write_a1s7_deferred_contract(config)
    if _a1s8_enabled(config):
        deferred_contract_path = _write_a1s8_deferred_contract(config)
    best_background = os.path.join(config['save_dir'], 'best_background_model.pth')
    best_analysis = os.path.join(config['save_dir'], 'best_fsia_model.pth')
    last_state = os.path.join(config['save_dir'], 'last_training_state.pth')

    seeded_background = False
    background_seed = config.get('background_seed_ckpt')
    if m00_observation_route:
        background_seed = (config.get('m00_observation_seed_checkpoint')
                           or background_seed)
    if _a1s8_enabled(config) or _a1s9_enabled(config):
        background_seed = (config.get('background_adapter_seed_checkpoint')
                           or background_seed)
    if raw_iri_analysis_only or causal_history_analysis_only:
        # This is an internal seed path, distinct from the legacy
        # --background-seed option (which remains rejected by main_fsia).
        background_seed = config.get('analysis_seed_background')
    elif _a1s8_enabled(config) or _a1s9_enabled(config):
        background_seed = config.get('background_adapter_seed_checkpoint')
    if background_seed and resume_state is None:
        if not os.path.isfile(background_seed):
            raise FileNotFoundError(
                f'Background seed checkpoint does not exist: {background_seed}')
        _load_background_seed(model, background_seed, device)
        if (getattr(model, 'm00_candidate_semantics', 'disabled') ==
                _M00_DECODER_TRUST_SCOPE and resume_state is None):
            model.capture_m00_decoder_baseline()
        if m00_observation_route:
            # Keep the selected A1-S seed auditable without pretending it is a
            # formally selected epoch in the new route.
            _atomic_torch_save(
                model.state_dict(),
                os.path.join(config['save_dir'], 'm00_observation_seed_model.pth'))
        else:
            _atomic_torch_save(model.state_dict(), best_background)
        if causal_history_analysis_only:
            replay_score, replay_metrics = validate(
                model, val_loader, batch_processor, device, config,
                sw_manager, iri_peak_manager, cosmic_val_loader, 'background',
                query_profile_partitions['development'])
            anchor = str(config.get(
                'causal_background_anchor', 'a1s2_m00_v1')).replace('-', '_')
            replay_name = {
                'a1s2_m00_v1': 'a1s2_m00_development_replay.json',
                'a1s_m00_v1': 'a1s_m00_development_replay.json',
            }.get(anchor, 'anchor_m00_development_replay.json')
            replay_path = os.path.join(config['save_dir'], replay_name)
            with open(replay_path, 'w', encoding='utf-8') as stream:
                json.dump({
                    'semantics': f'{anchor}_fixed_development_replay_v1',
                    'causal_background_anchor': anchor,
                    'checkpoint': os.path.abspath(background_seed),
                    'score': float(replay_score),
                    'metrics': replay_metrics,
                    'profile_split': config.get('resolved_profile_split'),
                }, stream, ensure_ascii=False, indent=2)
        # An adapter run starts at epoch zero from the migrated A1-S2 state;
        # the legacy external-seed path skips Background and enters Analysis.
        if raw_iri_analysis_only or causal_history_analysis_only:
            start_epoch = background_epochs
        else:
            start_epoch = (
            0 if (m00_observation_route
                  or getattr(model, 'm00_candidate_semantics', 'disabled')
                  in _M00_CANDIDATE_SCOPES
                  or config.get('background_condition_adapter') in (
                    _COND_ADAPTER_SEMANTICS, _A1S7_ADAPTER_SEMANTICS)
                      )
                else background_epochs)
        seeded_background = True
        print(f'  已迁移冻结Background: {background_seed}')

    current_stage = (
        'background' if background_only or start_epoch < background_epochs
        else 'analysis')
    if background_only and (
            start_epoch > background_epochs
            or (resume_state is not None and resume_state.get('stage') != 'background')):
        raise ValueError(
            'background-only cannot resume an Analysis-stage checkpoint')
    if (not background_only and (seeded_background or _resume_needs_analysis_setup(
            resume_state, start_epoch, background_epochs))):
        if not os.path.exists(best_background):
            raise FileNotFoundError(
                'cannot enter Analysis: best_background_model.pth is missing')
        model.load_state_dict(
            torch.load(best_background, map_location=device, weights_only=True),
            strict=True)
        _calibrate_fixed_r(
            model, train_loader, cosmic_train_loader, device,
            sw_manager, iri_peak_manager, batch_processor, config)
        _reset_random_seeds(config.get('analysis_seed', 42), device)

    _set_training_stage(model, current_stage)
    if (current_stage == 'background' and
            getattr(model, 'm00_candidate_semantics', 'disabled')
            in _M00_CANDIDATE_SCOPES):
        component_audit = _background_component_audit(model)
        candidate = getattr(model, 'm00_candidate_semantics')
        expected = ({'background_decoder'} if candidate == _M00_DECODER_TRUST_SCOPE
                    else {'m00_peak_relative_head'})
        forbidden = [
            name for name, details in component_audit.items()
            if name not in expected and details.get('trainable_parameter_count', 0)]
        missing = [
            name for name in expected
            if component_audit.get(name, {}).get('trainable_parameter_count', 0) == 0]
        if forbidden or missing:
            raise RuntimeError(
                f'M00 candidate scope violation: expected={sorted(expected)}, '
                f'missing={missing}, forbidden={forbidden}')
        if (candidate == _M00_DECODER_TRUST_SCOPE and any(
                parameter.requires_grad for parameter in model.background_decoder[:-1].parameters())):
            raise RuntimeError('decoder-last candidate unexpectedly unfreezes decoder trunk')
        config['background_component_audit'] = {
            'scope': candidate,
            'components': component_audit,
        }
    if (current_stage == 'background' and m00_observation_route
            and getattr(model, 'm00_candidate_semantics', 'disabled')
            not in _M00_CANDIDATE_SCOPES):
        component_audit = _background_component_audit(model)
        expected = {'iri_align_net', 'background_decoder'}
        frozen = [
            name for name in expected
            if component_audit.get(name, {}).get('trainable_parameter_count', 0)
            != component_audit.get(name, {}).get('parameter_count', 0)]
        forbidden = [
            name for name, details in component_audit.items()
            if name not in expected and details.get('trainable_parameter_count', 0)]
        if frozen or forbidden:
            raise RuntimeError(
                'M00 observation route scope violation: '
                f'trainable_expected={frozen}, forbidden={forbidden}')
        config['background_component_audit'] = {
            'scope': _M00_OBSERVATION_SCOPE,
            'components': component_audit,
        }
        print(
            f'[Epoch0 Background component audit] scope={_M00_OBSERVATION_SCOPE} '
            'components=[iri_align_net, background_decoder]')
    elif (current_stage == 'background'
            and config.get('background_training_scope')
            == _A1S7_ADAPTER_SCOPE):
        component_audit = _background_component_audit(model)
        expected = {
            'background_condition_adapter_low',
            'background_condition_adapter_high',
        }
        shared = {
            'iri_align_net', 'background_decoder', 'sw_encoder',
            'sw_freq_branch', 'sw_gate',
        }
        if not model.use_sw_freq:
            shared -= {'sw_freq_branch', 'sw_gate'}
        missing_trainable = [
            name for name in expected
            if component_audit.get(name, {}).get('trainable_parameter_count', 0)
            != component_audit.get(name, {}).get('parameter_count', 0)]
        drift = [
            name for name in shared
            if component_audit.get(name, {}).get('trainable_parameter_count', 0)]
        if missing_trainable or drift:
            raise RuntimeError(
                f'A1-S7 adapter-only scope violation: '
                f'adapter={missing_trainable}, shared={drift}')
        config['background_component_audit'] = {
            'scope': _A1S7_ADAPTER_SCOPE,
            'components': component_audit,
        }
        print(
            f'[Epoch0 Background component audit] scope={_A1S7_ADAPTER_SCOPE} '
            f'components={sorted(expected)}')
    elif (current_stage == 'background'
            and config.get('background_training_scope')
            == _COND_ADAPTER_JOINT_FRESH_SCOPE):
        component_audit = _background_component_audit(model)
        expected = {
            'iri_align_net', 'background_decoder', 'sw_encoder',
            'background_condition_adapter'}
        if model.use_sw_freq:
            expected.update(('sw_freq_branch', 'sw_gate'))
        frozen = [
            name for name in expected
            if component_audit.get(name, {}).get('trainable_parameter_count', 0)
            != component_audit.get(name, {}).get('parameter_count', 0)]
        if frozen:
            raise RuntimeError(
                f'joint fresh Background scope left components frozen: {frozen}')
        config['background_component_audit'] = {
            'scope': _COND_ADAPTER_JOINT_FRESH_SCOPE,
            'adapter_rng_seed': int(config['background_adapter_rng_seed']),
            'components': component_audit,
        }
        print(
            '[Epoch0 Background component audit] '
            f"scope={_COND_ADAPTER_JOINT_FRESH_SCOPE} "
            f"components={sorted(expected)}")
    if (current_stage == 'background'
            and 'resolved_background_stratum_bias_weight' not in config):
        _resolve_background_stratum_bias_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    if (current_stage == 'background'
            and config.get('background_bias_grouping')
            == _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2
            and config.get('resolved_background_magnetic_daynight_weight') is None):
        _resolve_background_magnetic_daynight_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    if (current_stage == 'background'
            and config.get('background_shape_loss') == _CONDITIONAL_SHAPE_LOSS
            and config.get('resolved_background_shape_weight') is None):
        _resolve_background_shape_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    if current_stage == 'background' and _natural_background_enabled(config):
        _reset_natural_sampler_state(
            train_loader, cosmic_train_loader, start_epoch=start_epoch)
    if (current_stage == 'analysis'
            and (config.get('use_covariance_moment_loss', False)
                 or config.get('use_empirical_covariance_loss', False))
            and 'resolved_covariance_weight' not in config):
        config['resolved_covariance_weight'] = _resolve_covariance_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    if (current_stage == 'analysis'
            and config.get('use_direction_loss', False)
            and 'resolved_direction_weight' not in config):
        config['resolved_direction_weight'] = _resolve_direction_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    if (current_stage == 'analysis'
            and config.get('use_observation_gram_loss', False)
            and 'resolved_gram_weight' not in config):
        config['resolved_gram_weight'] = _resolve_gram_weight(
            model, train_loader, cosmic_train_loader, batch_processor, device,
            config, sw_manager, iri_peak_manager,
            query_profile_partitions['train'])
    _record_resolved_training_config(config, covariance_strata)
    stage_epochs = background_epochs if current_stage == 'background' else analysis_epochs
    optimizer, scheduler = _make_optimizer(model, config, stage_epochs)
    scaler = (torch.cuda.amp.GradScaler()
              if config.get('use_amp', False) and device.type == 'cuda' else None)
    if resume_state is not None and resume_state.get('stage') == current_stage:
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        if scheduler is not None and resume_state.get('scheduler_state_dict'):
            scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        if scaler is not None and resume_state.get('scaler_state_dict'):
            scaler.load_state_dict(resume_state['scaler_state_dict'])

    train_losses = [row['total_loss'] for row in history]
    val_losses = [row['val_score'] for row in history]

    for epoch in range(start_epoch, total_epochs):
        stage = 'background' if epoch < background_epochs else 'analysis'
        if stage != current_stage:
            if (current_stage == 'background'
                    and config.get(
                        'background_checkpoint_selection', 'legacy') in (
                            _M00_NATURAL_SELECTOR, 'bias_constrained_v1',
                            _COND_ADAPTER_SELECTOR,
                            _COND_ADAPTER_JOINT_FRESH_SELECTOR,
                            _CONDITIONAL_SHAPE_SELECTOR,
                            _M00_NIGHT_GENERALIZATION_SELECTOR)
                    and best_epochs['background'] is None):
                raise RuntimeError(
                    'Background checkpoint selector found no eligible epoch; '
                    'Analysis cannot start')
            model.load_state_dict(
                torch.load(best_background, map_location=device, weights_only=True),
                strict=True)
            _calibrate_fixed_r(
                model, train_loader, cosmic_train_loader, device,
                sw_manager, iri_peak_manager, batch_processor, config)
            _reset_random_seeds(config.get('analysis_seed', 42), device)
            current_stage = stage
            _set_training_stage(model, stage)
            if ((config.get('use_covariance_moment_loss', False)
                 or config.get('use_empirical_covariance_loss', False))
                    and 'resolved_covariance_weight' not in config):
                config['resolved_covariance_weight'] = _resolve_covariance_weight(
                    model, train_loader, cosmic_train_loader, batch_processor,
                    device, config, sw_manager, iri_peak_manager,
                    query_profile_partitions['train'])
                _record_resolved_training_config(config, covariance_strata)
            if (config.get('use_direction_loss', False)
                    and 'resolved_direction_weight' not in config):
                config['resolved_direction_weight'] = _resolve_direction_weight(
                    model, train_loader, cosmic_train_loader, batch_processor,
                    device, config, sw_manager, iri_peak_manager,
                    query_profile_partitions['train'])
                _record_resolved_training_config(config, covariance_strata)
            if (config.get('use_observation_gram_loss', False)
                    and 'resolved_gram_weight' not in config):
                config['resolved_gram_weight'] = _resolve_gram_weight(
                    model, train_loader, cosmic_train_loader, batch_processor,
                    device, config, sw_manager, iri_peak_manager,
                    query_profile_partitions['train'])
                _record_resolved_training_config(config, covariance_strata)
            optimizer, scheduler = _make_optimizer(model, config, analysis_epochs)
            scaler = (torch.cuda.amp.GradScaler()
                      if config.get('use_amp', False) and device.type == 'cuda' else None)

        print(f'\nEpoch {epoch + 1}/{total_epochs} [{stage}]')
        if stage == 'background' and _natural_background_enabled(config):
            train_loss, train_metrics, batch_diagnostics = (
                train_one_epoch_natural_background(
                    model, train_loader, batch_processor, optimizer, device,
                    config, epoch, scaler, sw_manager, iri_peak_manager,
                    cosmic_train_loader, query_profile_partitions['train']))
        else:
            train_loss, train_metrics, batch_diagnostics = train_one_epoch(
                model, train_loader, batch_processor, optimizer, device,
                config, epoch, stage, scaler, sw_manager, iri_peak_manager,
                cosmic_train_loader, query_profile_partitions['train'])
        if stage == 'background':
            config.setdefault('background_update_steps', []).append(
                int(train_metrics['processed_batches']))
        if _low_altitude_prior_protocol(config) is not None:
            config.setdefault('low_altitude_anchor_history', []).append({
                'epoch': epoch + 1,
                'stage': stage,
                'raw_loss': train_metrics['low_altitude'],
                'weighted_loss': train_metrics['weighted_low_altitude'],
                'profiles_per_batch': train_metrics['low_altitude_profiles'],
                'points_per_batch': train_metrics['low_altitude_points'],
                'active_fraction': train_metrics[
                    'low_altitude_active_fraction'],
                'FY': {
                    'raw_loss': train_metrics['fy_low_altitude'],
                    'profiles_per_batch': train_metrics[
                        'fy_low_altitude_profiles'],
                    'points_per_batch': train_metrics['fy_low_altitude_points'],
                    'active_fraction': train_metrics[
                        'fy_low_altitude_active_fraction'],
                },
                'COSMIC': {
                    'raw_loss': train_metrics['cosmic_low_altitude'],
                    'profiles_per_batch': train_metrics[
                        'cosmic_low_altitude_profiles'],
                    'points_per_batch': train_metrics[
                        'cosmic_low_altitude_points'],
                    'active_fraction': train_metrics[
                        'cosmic_low_altitude_active_fraction'],
                },
                'gradient_ratio_to_observation': train_metrics[
                    'gradient_ratio_low_altitude'],
            })
        if batch_diagnostics:
            diagnostics_path = os.path.join(
                config['save_dir'], 'batch_diagnostics.jsonl')
            with open(diagnostics_path, 'a', encoding='utf-8') as stream:
                for row in batch_diagnostics:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        val_score, val_metrics = validate(
            model, val_loader, batch_processor, device, config,
            sw_manager, iri_peak_manager, cosmic_val_loader, stage,
            query_profile_partitions['development'])
        low_altitude_development = _development_low_altitude_diagnostics(
            model, {'FY': val_loader, 'COSMIC': cosmic_val_loader},
            batch_processor, device, stage, config, sw_manager,
            iri_peak_manager, query_profile_partitions['train'])
        if low_altitude_development is not None:
            val_metrics['low_altitude_diagnostic'] = low_altitude_development
        if scheduler is not None:
            scheduler.step()

        smoke_auxiliary_limit = float(config.get(
            'smoke_auxiliary_gradient_ratio_max', 0.30))
        if (train_metrics['gradient_audit_batches']
                and train_metrics['gradient_ratio'] > smoke_auxiliary_limit):
            print(
                f"  警告: 辅助/观测梯度比={train_metrics['gradient_ratio']:.3f}>"
                f"{smoke_auxiliary_limit:.2f}；"
                '全量训练前应下调对应辅助权重')
        if (_low_altitude_prior_protocol(config) is not None
                and config.get('max_train_batches') is not None
                and train_metrics['gradient_audit_batches']
                and not config.get('smoke_run', False)):
            low_ratio = train_metrics['gradient_ratio_low_altitude']
            maximum = float(config['low_altitude_gradient_ratio_max'])
            if not np.isfinite(low_ratio) or low_ratio > maximum:
                raise RuntimeError(
                    f'low-altitude preflight gradient ratio {low_ratio!r} '
                    f'is outside [0, {maximum}]')
            if stage == 'analysis' and low_ratio <= 0.0:
                raise RuntimeError(
                    'low-altitude Analysis anchor has zero gradient; '
                    'check train-only token coverage before full training')
        history_row = {
            'epoch': epoch + 1,
            'stage': stage,
            'total_loss': train_loss,
            'val_score': val_score,
            **train_metrics,
            **val_metrics,
        }
        print(
            f"  train={train_loss:.6f} val={val_score:.6f} "
            f"FY={val_metrics['fy_profile_rmse']:.6f} "
            f"COSMIC={val_metrics['cosmic_profile_rmse']:.6f}")

        if _a1s8_enabled(config) and stage == 'background':
            exposure = train_metrics.get('a1s8_exposure', {})
            ratio = exposure.get('gradient_ratio')
            ratio_min = float(config.get('background_gradient_ratio_min', 0.25))
            ratio_max = float(config.get('background_gradient_ratio_max', 4.0))
            try:
                ratio_ok = (np.isfinite(float(ratio))
                            and ratio_min <= float(ratio) <= ratio_max)
            except (TypeError, ValueError):
                ratio_ok = False
            config['_a1s8_gradient_oob_streak'] = (
                0 if ratio_ok else
                int(config.get('_a1s8_gradient_oob_streak', 0)) + 1)

        selection_semantics = config.get(
            'background_checkpoint_selection', 'legacy')
        if (stage == 'background'
                and selection_semantics == _M00_NATURAL_SELECTOR):
            selection_audit = m00_natural_selection_audit(
                val_metrics, m00_selection_baseline, epoch + 1)
            selection_key = selection_audit['selection_key']
            history_row['checkpoint_selection'] = selection_audit
        elif stage == 'background' and selection_semantics == _M00_NIGHT_GENERALIZATION_V2_SELECTOR:
            selection_audit = night_generalization_v2_audit(
                val_metrics, background_selection_baseline, epoch + 1,
                config['background_night_v2_supported_cells'])
            selection_key = selection_audit['selection_key']
            history_row['checkpoint_selection'] = selection_audit
        elif stage == 'background' and selection_semantics in (
                'bias_constrained_v1', _COND_ADAPTER_SELECTOR,
                _COND_ADAPTER_JOINT_FRESH_SELECTOR,
                _CONDITIONAL_SHAPE_SELECTOR,
                _M00_NIGHT_GENERALIZATION_SELECTOR):
            if _a1s8_enabled(config):
                exposure = train_metrics.get('a1s8_exposure', {})
                streak = int(config.get('_a1s8_gradient_oob_streak', 0))
                selection_audit = _a1s8_selection_audit(
                    val_metrics, background_selection_baseline,
                    exposure=exposure, config=config,
                    gradient_oob_streak=streak)
            else:
                selection_audit = (
                    _joint_fresh_adapter_selection_audit(
                        val_metrics, background_selection_baseline)
                    if selection_semantics == _COND_ADAPTER_JOINT_FRESH_SELECTOR
                    else _conditional_shape_selection_audit(
                        val_metrics, background_selection_baseline)
                    if selection_semantics == _CONDITIONAL_SHAPE_SELECTOR
                    else (_night_generalization_selection_audit(
                        val_metrics, background_selection_baseline,
                        epoch + 1)
                          if selection_semantics ==
                          _M00_NIGHT_GENERALIZATION_SELECTOR
                          else _background_bias_selection_audit(
                              val_metrics, background_selection_baseline)))
            selection_key = selection_audit['selection_key']
            history_row['checkpoint_selection'] = selection_audit
        elif (stage == 'analysis'
              and config.get('analysis_training_scope')
              == _CAUSAL_HISTORY_ANALYSIS_SCOPE):
            selection_audit = _causal_history_selection_audit(val_metrics)
            selection_key = selection_audit['selection_key']
            history_row['checkpoint_selection'] = selection_audit
        else:
            selection_key = _development_selection_key(
                val_metrics, config, stage=stage)
        history.append(history_row)
        train_losses.append(train_loss)
        val_losses.append(val_score)
        previous_key = best_selection_keys[stage]
        better = previous_key is None or (selection_key is not None
                                          and tuple(selection_key) > tuple(previous_key))
        if stage == 'background' and selection_semantics == _M00_NIGHT_GENERALIZATION_V2_SELECTOR:
            previous_audit = next((row['checkpoint_selection'] for row in history
                                   if row['epoch'] == best_epochs[stage]), None)
            better = night_generalization_v2_better(selection_audit, previous_audit)
        if selection_key is not None and better:
            best_scores[stage] = val_score
            best_epochs[stage] = epoch + 1
            best_selection_keys[stage] = selection_key
            _atomic_torch_save(
                model.state_dict(),
                best_background if stage == 'background' else best_analysis)
        if stage == 'background':
            _atomic_torch_save(
                model.state_dict(),
                os.path.join(
                    config['save_dir'],
                    f'epoch_{epoch + 1:02d}_background_model.pth'))
        else:
            _atomic_torch_save(
                model.state_dict(),
                os.path.join(
                    config['save_dir'], f'epoch_{epoch + 1:02d}_model.pth'))

        _atomic_torch_save({
            'checkpoint_type': 'run66_training_state',
            'format_version': format_version,
            'model_domain_semantics': config.get(
                'model_domain_semantics', 'legacy_120_500_domain_v1'),
            'alt_range': [float(value) for value in config.get(
                'alt_range', (120.0, 500.0))],
            'observation_alt_range': [float(value) for value in (
                config.get('observation_alt_range') or config.get(
                    'alt_range', (120.0, 500.0)))],
            'peak_search_alt_range': [float(value) for value in (
                config.get('peak_search_alt_range') or config.get(
                    'alt_range', (120.0, 500.0)))],
            'completed_epochs': epoch + 1,
            'background_epochs': background_epochs,
            'analysis_epochs': analysis_epochs,
            'background_training_semantics': config.get(
                'background_training_semantics'),
            'background_trust_gate_enabled': bool(config.get(
                'background_trust_gate_enabled', False)),
            'background_trust_gate_semantics': config.get(
                'background_trust_gate_semantics', 'disabled'),
            'background_trust_gate_parameters': {
                key: float(config.get(key, default))
                for key, default in (
                    ('background_trust_gate_altitude_core_km', 200.0),
                    ('background_trust_gate_altitude_transition_km', 100.0),
                    ('background_trust_gate_night_cosine_offset', 0.2),
                    ('background_trust_gate_night_cosine_scale', 0.2),
                    ('background_trust_gate_dip_core', 0.25),
                    ('background_trust_gate_dip_transition', 0.25),
                )
            },
            'background_only': background_only,
            'smoke_run': bool(config.get('smoke_run', False)),
            'background_loader_schedule': config.get(
                'background_loader_schedule'),
            'background_update_steps': config.get(
                'background_update_steps', []),
            'training_protocol': training_protocol,
            'low_altitude_anchor_history': config.get(
                'low_altitude_anchor_history', []),
            'r_mode': config.get('r_mode', 'global'),
            'use_distance_localization': bool(
                config.get('use_distance_localization', False)),
            'assimilation_semantics': config.get(
                'assimilation_semantics', 'legacy_local_etkf'),
            'use_physical_localization': bool(
                config.get('use_physical_localization', False)),
            'neighbor_directory_semantics': config.get(
                'neighbor_directory_semantics', 'profile_center_legacy'),
            'observation_chunk_size': int(config.get(
                'observation_chunk_size', 4096)),
            'source_mode_schedule': config.get(
                'source_mode_schedule', 'random_profile'),
            'analysis_training_scope': config.get('analysis_training_scope'),
            'analysis_output_semantics': config.get('analysis_output_semantics'),
            'analysis_observation_time_support': config.get(
                'analysis_observation_time_support'),
            'causal_background_anchor': config.get(
                'causal_background_anchor', 'a1s2_m00_v1'),
            'r_source': config.get('r_source'),
            'm00_observation_calibration_semantics': config.get(
                'm00_observation_calibration_semantics', 'disabled'),
            'm00_observation_seed_checkpoint': config.get(
                'm00_observation_seed_checkpoint'),
            'm00_observation_training_scope': config.get(
                'm00_observation_training_scope', 'disabled'),
            'm00_observation_stage': config.get(
                'm00_observation_stage', 'bias_r'),
            'm00_candidate_semantics': config.get(
                'm00_candidate_semantics', 'disabled'),
            'm00_candidate_residual_cap': float(config.get(
                'm00_candidate_residual_cap', 0.05)),
            'analysis_exact_mode_loss': bool(
                config.get('analysis_exact_mode_loss', False)),
            'use_empirical_covariance_loss': bool(
                config.get('use_empirical_covariance_loss', False)),
            'use_direction_loss': bool(
                config.get('use_direction_loss', False)),
            'use_observation_gram_loss': bool(
                config.get('use_observation_gram_loss', False)),
            'representativeness_kernel': representativeness_identity,
            'architecture': architecture,
            'profile_subset': profile_subset,
            'profile_subset_identity': profile_subset_identity,
            'date_split_identity': date_split_identity,
            'profile_split_identity': config.get('resolved_profile_split'),
            'input_data_sha256': input_data_sha256,
            'allowed_profile_summary': allowed_profile_summary,
            'model_domain_data_summary': config.get(
                'model_domain_data_summary'),
            'resolved_covariance_weight': config.get(
                'resolved_covariance_weight'),
            'resolved_direction_weight': config.get(
                'resolved_direction_weight'),
            'resolved_gram_weight': config.get('resolved_gram_weight'),
            'gram_gradient_calibration': config.get(
                'gram_gradient_calibration'),
            'background_stratum_bias_gradient_target': float(config.get(
                'background_stratum_bias_gradient_target', 0.0)),
            'background_stratum_bias_calibration_batches': int(config.get(
                'background_stratum_bias_calibration_batches', 20)),
            'resolved_background_stratum_bias_weight': config.get(
                'resolved_background_stratum_bias_weight'),
            'background_stratum_bias_gradient_calibration': config.get(
                'background_stratum_bias_gradient_calibration'),
            'background_magnetic_daynight_gradient_target': float(config.get(
                'background_magnetic_daynight_gradient_target', 0.0125)),
            'background_magnetic_daynight_calibration_batches': int(config.get(
                'background_magnetic_daynight_calibration_batches', 20)),
            'resolved_background_magnetic_daynight_weight': config.get(
                'resolved_background_magnetic_daynight_weight'),
            'background_magnetic_daynight_gradient_calibration': config.get(
                'background_magnetic_daynight_gradient_calibration'),
            'background_bias_grouping': config.get(
                'background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY),
            'background_bias_magnetic_min_profiles': int(config.get(
                'background_bias_magnetic_min_profiles', 30)),
            'background_bias_magnetic_min_batch_profiles': int(config.get(
                'background_bias_magnetic_min_batch_profiles', 8)),
            'background_bias_magnetic_weight': float(config.get(
                'background_bias_magnetic_weight', 0.5)),
            'background_bias_magnetic_support_counts': config.get(
                'background_bias_magnetic_support_counts'),
            'background_bias_magnetic_support_semantics': config.get(
                'background_bias_magnetic_support_semantics'),
            'background_shape_loss': config.get(
                'background_shape_loss', 'disabled'),
            'background_shape_gradient_target': float(config.get(
                'background_shape_gradient_target', 0.02)),
            'background_shape_calibration_batches': int(config.get(
                'background_shape_calibration_batches', 20)),
            'background_shape_min_profiles_per_cell': int(config.get(
                'background_shape_min_profiles_per_cell', 8)),
            'resolved_background_shape_weight': config.get(
                'resolved_background_shape_weight'),
            'background_shape_gradient_calibration': config.get(
                'background_shape_gradient_calibration'),
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_scores': best_scores,
            'best_epochs': best_epochs,
            'best_selection_keys': best_selection_keys,
            'history': history,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'cuda_rng_state_all': (
                torch.cuda.get_rng_state_all() if device.type == 'cuda' else None),
        }, last_state)
        if not background_only:
            plot_training_curves(
                history,
                save_path=os.path.join(
                    config['save_dir'], 'fsia_training_curves.png'))

    with open(os.path.join(config['save_dir'], 'fsia_training_history.json'),
              'w', encoding='utf-8') as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)
    if _a1s8_enabled(config):
        _write_a1s8_exposure_audit(config, history)
    _record_resolved_training_config(config, covariance_strata)
    background_candidate = None
    background_candidate_epoch = None
    background_candidate_key = None
    background_candidate_eligible = None
    background_candidate_selection_audit = None
    selection_semantics = config.get('background_checkpoint_selection', 'legacy')
    if background_only:
        if selection_semantics == _M00_NATURAL_SELECTOR:
            candidate_result = select_m00_background_candidate(
                history, m00_selection_baseline)
            candidate_row = candidate_result['selected_row']
            background_candidate_eligible = candidate_result['eligible']
            background_candidate_selection_audit = candidate_result[
                'selected_audit']
            if candidate_row is not None:
                background_candidate_epoch = int(candidate_row['epoch'])
                background_candidate = os.path.join(
                    config['save_dir'],
                    f'epoch_{background_candidate_epoch:02d}_background_model.pth')
                background_candidate_key = candidate_result[
                    'selected_audit'].get('selection_key')
                if background_candidate_key is None:
                    candidate_values = candidate_result[
                        'selected_audit'].get('candidate', {})
                    background_candidate_key = [
                        -float(candidate_values['profile_rmse']),
                        float(candidate_values['mean_ccc']),
                        -background_candidate_epoch,
                    ]
        elif selection_semantics == _M00_NIGHT_GENERALIZATION_V2_SELECTOR:
            candidate_result = select_night_generalization_v2(
                history, background_selection_baseline,
                config['background_night_v2_supported_cells'])
            background_candidate_eligible = candidate_result['eligible']
            background_candidate_selection_audit = candidate_result['selected_audit']
            background_candidate_epoch = candidate_result['selected_epoch']
            if background_candidate_epoch is not None:
                background_candidate = os.path.join(config['save_dir'],
                    f'epoch_{background_candidate_epoch:02d}_background_model.pth')
                background_candidate_key = candidate_result['selected_audit']['selection_key']
        else:
            candidate_rows = [
                row for row in history
                if row.get('stage') == 'background'
                and np.isfinite(float(row.get('fy_profile_rmse', np.nan)))
                and np.isfinite(float(row.get('cosmic_profile_rmse', np.nan)))
                and np.isfinite(float(row.get('ccc', np.nan)))
            ]
            if candidate_rows:
                candidate_row = min(
                    candidate_rows,
                    key=lambda row: (
                        (float(row['fy_profile_rmse'])
                         + float(row['cosmic_profile_rmse'])) / 2.0,
                        -float(row['ccc']),
                        int(row['epoch']),
                    ))
                background_candidate_epoch = int(candidate_row['epoch'])
                background_candidate = os.path.join(
                    config['save_dir'],
                    f'epoch_{background_candidate_epoch:02d}_background_model.pth')
                background_candidate_key = [
                    (float(candidate_row['fy_profile_rmse'])
                     + float(candidate_row['cosmic_profile_rmse'])) / 2.0,
                    float(candidate_row['ccc']),
                    background_candidate_epoch,
                ]
    diagnostic_checkpoint = None
    diagnostic_only = False
    no_eligible_background = False
    if background_only and best_epochs['background'] is None:
        if selection_semantics in (
                _COND_ADAPTER_JOINT_FRESH_SELECTOR,
                _CONDITIONAL_SHAPE_SELECTOR):
            diagnostic_rows = [
                row for row in history
                if row.get('stage') == 'background'
                and row.get('checkpoint_selection', {}).get(
                    'diagnostic_selection_key') is not None
            ]
            if diagnostic_rows:
                diagnostic_row = max(
                    diagnostic_rows,
                    key=lambda row: tuple(row['checkpoint_selection'][
                        'diagnostic_selection_key']))
                diagnostic_epoch = int(diagnostic_row['epoch'])
                diagnostic_checkpoint = os.path.join(
                    config['save_dir'],
                    f'epoch_{diagnostic_epoch:02d}_background_model.pth')
                if not os.path.isfile(diagnostic_checkpoint):
                    raise FileNotFoundError(
                        f'diagnostic Background checkpoint missing: '
                        f'{diagnostic_checkpoint}')
                diagnostic_only = True
                print(
                    '[Background selector] no eligible epoch; '
                    f'using epoch {diagnostic_epoch} for DIAGNOSTIC_ONLY ISR')
            else:
                raise RuntimeError(
                    'Background selector found no finite diagnostic epoch')
        elif selection_semantics in (
                _M00_NATURAL_SELECTOR, 'bias_constrained_v1',
                _COND_ADAPTER_SELECTOR, _M00_NIGHT_GENERALIZATION_SELECTOR,
                _M00_NIGHT_GENERALIZATION_V2_SELECTOR):
            if (_a1s8_enabled(config) or m00_observation_route
                    or _natural_background_enabled(config)):
                # Gated M00/natural experiments persist the complete audit and
                # return without fabricating a checkpoint when no epoch passes.
                no_eligible_background = True
                print(
                    '[M00 selector] no eligible Background epoch; '
                    'stopping without checkpoint or ISR')
            else:
                raise RuntimeError(
                    'Background checkpoint selector found no eligible epoch; '
                    'see fsia_training_history.json checkpoint_selection audits')
    background_validation = None
    background_gate_passed = None
    # A gated Background run may finish with no eligible epoch.  Keep the
    # selected checkpoint empty so summary finalization does not hash a path
    # that was intentionally never created.
    background_checkpoint = (
        diagnostic_checkpoint
        or (None if no_eligible_background else best_background))
    if background_only and not no_eligible_background:
        if selection_semantics == _M00_NIGHT_GENERALIZATION_V2_SELECTOR:
            background_validation = candidate_result['selected_row']
        else:
            model.load_state_dict(
                torch.load(background_checkpoint, map_location=device,
                           weights_only=True),
                strict=True)
            _, background_validation = validate(
                model, val_loader, batch_processor, device, config,
                sw_manager, iri_peak_manager, cosmic_val_loader, 'background',
                query_profile_partitions['development'])
        comparisons = [
            background_validation['fy_rmse_change_M00_minus_raw'],
            background_validation['cosmic_rmse_change_M00_minus_raw'],
        ]
        background_gate_passed = all(
            value is not None and np.isfinite(value) and value <= 0.0
            for value in comparisons)
        print(
            f'[Background development gate] FY ΔRMSE={comparisons[0]:.6g}, '
            f'COSMIC ΔRMSE={comparisons[1]:.6g}, '
            f'passed={background_gate_passed}')
    summary = {
        'completed_stage': 'background' if background_only else 'analysis',
        'smoke_run': bool(config.get('smoke_run', False)),
        'checkpoint_format_version': format_version,
        'model_domain_semantics': config.get(
            'model_domain_semantics', 'legacy_120_500_domain_v1'),
        'alt_range': [float(value) for value in config.get(
            'alt_range', (120.0, 500.0))],
        'observation_alt_range': [float(value) for value in (
            config.get('observation_alt_range') or config.get(
                'alt_range', (120.0, 500.0)))],
        'peak_search_alt_range': [float(value) for value in (
            config.get('peak_search_alt_range') or config.get(
                'alt_range', (120.0, 500.0)))],
        'model_domain_data_summary': config.get(
            'model_domain_data_summary'),
        'best_scores': {
            stage: (score if np.isfinite(score) else None)
            for stage, score in best_scores.items()},
        'best_epochs': best_epochs,
        'best_selection_keys': best_selection_keys,
        'background_candidate_checkpoint': background_candidate,
        'background_candidate_epoch': background_candidate_epoch,
        'background_candidate_selection_key': background_candidate_key,
        'background_candidate_eligible': background_candidate_eligible,
        'background_candidate_selection_audit': (
            background_candidate_selection_audit),
        'checkpoint_selection_semantics': config.get(
            'checkpoint_selection_semantics', 'mean_profile_rmse_v1'),
        'background_checkpoint_selection': config.get(
            'background_checkpoint_selection', 'legacy'),
        'm00_selection_reference_history': config.get(
            'm00_selection_reference_history'),
        'm00_selection_reference_epoch': config.get(
            'm00_selection_reference_epoch'),
        'background_selection_baseline': config.get(
            'background_selection_baseline'),
        'background_selection_audits': [
            row['checkpoint_selection'] for row in history
            if row.get('stage') == 'background'
            and 'checkpoint_selection' in row],
        'checkpoint': background_checkpoint if background_only else best_analysis,
        'diagnostic_only': bool(diagnostic_only),
        'checkpoint_sha256': _checkpoint_sha256(
            background_checkpoint if background_only else best_analysis,
            skip=config.get('skip_hash_verification', False)),
        'checkpoint_stage': 'background' if background_only else 'analysis',
        'background_training_semantics': config.get(
            'background_training_semantics'),
        'checkpoint_semantic': config.get('checkpoint_semantic',
                                          'a1s_legacy_v1'),
        'background_condition_adapter': config.get(
            'background_condition_adapter', 'disabled'),
        'background_adapter_gate_semantics': config.get(
            'background_adapter_gate_semantics'),
        'background_sampling_semantics': config.get(
            'background_sampling_semantics'),
        'background_fixed_updates_per_epoch': config.get(
            'background_fixed_updates_per_epoch'),
        'background_stratum_bias_gradient_target': float(config.get(
            'background_stratum_bias_gradient_target', 0.0)),
        'background_stratum_bias_calibration_batches': int(config.get(
            'background_stratum_bias_calibration_batches', 20)),
        'resolved_background_stratum_bias_weight': config.get(
            'resolved_background_stratum_bias_weight'),
        'background_stratum_bias_gradient_calibration': config.get(
            'background_stratum_bias_gradient_calibration'),
        'background_magnetic_daynight_gradient_target': float(config.get(
            'background_magnetic_daynight_gradient_target', 0.0125)),
        'background_magnetic_daynight_calibration_batches': int(config.get(
            'background_magnetic_daynight_calibration_batches', 20)),
        'resolved_background_magnetic_daynight_weight': config.get(
            'resolved_background_magnetic_daynight_weight'),
        'background_magnetic_daynight_gradient_calibration': config.get(
            'background_magnetic_daynight_gradient_calibration'),
        'background_bias_grouping': config.get(
            'background_bias_grouping', _BACKGROUND_BIAS_GROUPING_LEGACY),
        'background_bias_magnetic_min_profiles': int(config.get(
            'background_bias_magnetic_min_profiles', 30)),
        'background_bias_magnetic_min_batch_profiles': int(config.get(
            'background_bias_magnetic_min_batch_profiles', 8)),
        'background_bias_magnetic_weight': float(config.get(
            'background_bias_magnetic_weight', 0.5)),
        'background_bias_magnetic_support_counts': config.get(
            'background_bias_magnetic_support_counts'),
        'background_bias_magnetic_support_semantics': config.get(
            'background_bias_magnetic_support_semantics'),
        'a1s7_m11_deferred_contract': config.get(
            'a1s7_m11_deferred_contract'),
        'background_adapter_audit': {
            'adapter_semantics': config.get(
                'background_condition_adapter', 'disabled'),
            'gate_semantics': config.get(
                'background_adapter_gate_semantics'),
            'residual_cap_dex': float(config.get(
                'background_adapter_residual_cap', 0.1)),
            'epochs': [
                {
                    key: row.get(key)
                    for key in (
                        'epoch', 'background_adapter_gate_mean',
                        'background_adapter_gate_min',
                        'background_adapter_gate_max',
                        'background_adapter_effective_abs_mean',
                        'background_adapter_effective_abs_max',
                        'background_residual_cap_saturation')
                }
                for row in history
                if row.get('stage') == 'background'
            ],
        },
        'background_trust_gate_enabled': bool(config.get(
            'background_trust_gate_enabled', False)),
        'background_trust_gate_semantics': config.get(
            'background_trust_gate_semantics', 'disabled'),
        'background_trust_gate_parameters': {
            key: float(config.get(key, default))
            for key, default in (
                ('background_trust_gate_altitude_core_km', 200.0),
                ('background_trust_gate_altitude_transition_km', 100.0),
                ('background_trust_gate_night_cosine_offset', 0.2),
                ('background_trust_gate_night_cosine_scale', 0.2),
                ('background_trust_gate_dip_core', 0.25),
                ('background_trust_gate_dip_transition', 0.25),
            )
        },
        'background_development': background_validation,
        'background_development_gate_passed': background_gate_passed,
        'background_loader_schedule': config.get('background_loader_schedule'),
        'background_update_steps': config.get('background_update_steps', []),
        'background_component_audit': config.get('background_component_audit'),
        'training_protocol': training_protocol,
        'low_altitude_prior_protocol': _low_altitude_prior_protocol(config),
        'low_altitude_anchor_history': config.get(
            'low_altitude_anchor_history', []),
        'r_mode': config.get('r_mode', 'global'),
        'use_distance_localization': bool(
            config.get('use_distance_localization', False)),
        'assimilation_semantics': config.get(
            'assimilation_semantics', 'legacy_local_etkf'),
        'use_physical_localization': bool(
            config.get('use_physical_localization', False)),
        'neighbor_directory_semantics': config.get(
            'neighbor_directory_semantics', 'profile_center_legacy'),
        'observation_chunk_size': int(config.get(
            'observation_chunk_size', 4096)),
        'source_mode_schedule': config.get(
            'source_mode_schedule', 'random_profile'),
        'analysis_training_scope': config.get('analysis_training_scope'),
        'analysis_output_semantics': config.get('analysis_output_semantics'),
        'analysis_observation_time_support': config.get(
            'analysis_observation_time_support'),
        'r_source': config.get('r_source'),
        'm00_observation_calibration_semantics': config.get(
            'm00_observation_calibration_semantics', 'disabled'),
        'm00_observation_seed_checkpoint': config.get(
            'm00_observation_seed_checkpoint'),
        'm00_observation_training_scope': config.get(
            'm00_observation_training_scope', 'disabled'),
        'm00_observation_background_semantics': config.get(
            'background_training_semantics'),
        'm00_observation_stage': config.get('m00_observation_stage', 'bias_r'),
        'm00_calibration_applied': bool(config.get('m00_calibration_applied', False)),
        'm00_calibration_frozen': bool(config.get('m00_calibration_frozen', False)),
        'm00_representativeness_enabled': bool(config.get(
            'm00_representativeness_enabled', False)),
        'm00_localization_enabled': bool(config.get(
            'm00_localization_enabled', False)),
        'm00_calibration_artifact': config.get('m00_calibration_artifact'),
        'm00_transition_contract': config.get('m00_transition_contract'),
        'analysis_exact_mode_loss': bool(
            config.get('analysis_exact_mode_loss', False)),
        'use_empirical_covariance_loss': bool(
            config.get('use_empirical_covariance_loss', False)),
        'use_direction_loss': bool(
            config.get('use_direction_loss', False)),
        'use_observation_gram_loss': bool(
            config.get('use_observation_gram_loss', False)),
        'representativeness_kernel': representativeness_identity,
        'r_fy': (model.kalman_layer.r_fy.item()
                 if not background_only else None),
        'r_cosmic': (model.kalman_layer.r_cosmic.item()
                     if not background_only else None),
        'r_calibration': (
            os.path.join(config['save_dir'], 'r_calibration.json')
            if not background_only else None),
        'architecture': architecture,
        'profile_subset': profile_subset,
        'profile_subset_manifest': profile_subset_identity,
        'date_split': date_split_identity,
        'profile_split': config.get('resolved_profile_split'),
        'input_data_sha256': input_data_sha256,
        'allowed_profile_summary': allowed_profile_summary,
        'covariance_moment': {
            'enabled': bool(config.get('use_covariance_moment_loss', False)),
            'empirical_cells_enabled': bool(
                config.get('use_empirical_covariance_loss', False)),
            'gradient_target': float(
                config.get('covariance_gradient_target', 0.20)),
            'resolved_weight': config.get('resolved_covariance_weight'),
            'strata': covariance_strata,
        },
        'direction_loss': {
            'enabled': bool(config.get('use_direction_loss', False)),
            'gradient_target': float(
                config.get('direction_gradient_target', 0.20)),
            'resolved_weight': config.get('resolved_direction_weight'),
        },
        'observation_gram_loss': {
            'enabled': bool(config.get('use_observation_gram_loss', False)),
            'gradient_target': float(
                config.get('gram_gradient_target', 0.02)),
            'calibration_batches': int(
                config.get('gram_calibration_batches', 20)),
            'resolved_weight': config.get('resolved_gram_weight'),
            'gradient_calibration': config.get('gram_gradient_calibration'),
        },
        'background_shape_loss': {
            'semantics': config.get('background_shape_loss', 'disabled'),
            'gradient_target': float(config.get(
                'background_shape_gradient_target', 0.02)),
            'calibration_batches': int(config.get(
                'background_shape_calibration_batches', 20)),
            'min_profiles_per_cell': int(config.get(
                'background_shape_min_profiles_per_cell', 8)),
            'resolved_weight': config.get('resolved_background_shape_weight'),
            'gradient_calibration': config.get(
                'background_shape_gradient_calibration'),
        },
    }
    if _a1s8_enabled(config):
        summary['a1s8_m11_deferred_contract'] = config.get(
            'a1s8_m11_deferred_contract')
        summary['a1s8_exposure_audit'] = config.get('a1s8_exposure_audit')
        summary['a1s8_contract'] = {
            'background_anchor_semantics': config.get(
                'background_anchor_semantics'),
            'background_anchor_gradient_target': float(config.get(
                'background_anchor_gradient_target', 0.10)),
            'background_anchor_calibration_batches': int(config.get(
                'background_anchor_calibration_batches', 20)),
            'background_anchor_warmup_epochs': int(config.get(
                'background_anchor_warmup_epochs', 1)),
            'background_gate_balance_semantics': config.get(
                'background_gate_balance_semantics'),
            'background_source_mix_schedule': config.get(
                'background_source_mix_schedule'),
            'background_mix_legacy_fraction': float(config.get(
                'background_mix_legacy_fraction', 1.0)),
            'background_mix_balanced_fraction': float(config.get(
                'background_mix_balanced_fraction', 0.0)),
            'background_exposure_weight_cap': float(config.get(
                'background_exposure_weight_cap', 3.0)),
            'background_gradient_ratio_min': float(config.get(
                'background_gradient_ratio_min', 0.25)),
            'background_gradient_ratio_max': float(config.get(
                'background_gradient_ratio_max', 4.0)),
            'background_low_effective_mass_min': float(config.get(
                'background_low_effective_mass_min', 0.25)),
            'background_low_effective_mass_max': float(config.get(
                'background_low_effective_mass_max', 0.35)),
            'background_high_effective_mass_min': float(config.get(
                'background_high_effective_mass_min', 0.65)),
            'background_high_effective_mass_max': float(config.get(
                'background_high_effective_mass_max', 0.75)),
        }
    with open(os.path.join(config['save_dir'], 'training_summary.json'),
              'w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    if no_eligible_background:
        summary.update({
            'checkpoint': None,
            'checkpoint_stage': None,
            'no_eligible_checkpoint': True,
            'stop_reason': 'no_eligible_background_development_epoch',
        })
        summary_path = os.path.join(config['save_dir'], 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        return (
            model, train_losses, val_losses, train_loader, val_loader,
            sw_manager, batch_processor, iri_peak_manager,
        )

    if (background_only and not background_gate_passed
            and config.get('background_condition_adapter', 'disabled')
            != _COND_ADAPTER_SEMANTICS
            and config.get('background_shape_loss') != _CONDITIONAL_SHAPE_LOSS):
        raise RuntimeError(
            'Background development gate failed: M00 is worse than Raw IRI '
            'for at least one source')

    selected_checkpoint = background_checkpoint if background_only else best_analysis
    if not os.path.isfile(selected_checkpoint):
        # An Analysis selector is allowed to reject every finite epoch.  Keep
        # this fail-closed: do not fabricate a checkpoint or continue into
        # evaluation/ISR with an unselected model.
        if not background_only and best_epochs.get('analysis') is None:
            summary.update({
                'checkpoint': None,
                'checkpoint_stage': None,
                'no_eligible_checkpoint': True,
                'stop_reason': 'no_eligible_analysis_development_epoch',
            })
            summary_path = os.path.join(
                config['save_dir'], 'training_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2)
            print(
                '[Analysis selector] no eligible Development epoch; '
                'stopping without checkpoint or ISR')
            return (
                model, train_losses, val_losses, train_loader, val_loader,
                sw_manager, batch_processor, iri_peak_manager,
            )
        raise FileNotFoundError(
            f'completed {summary["completed_stage"]} stage lacks checkpoint')
    model.load_state_dict(
        torch.load(selected_checkpoint, map_location=device, weights_only=True),
        strict=True)
    return (
        model, train_losses, val_losses, train_loader, val_loader,
        sw_manager, batch_processor, iri_peak_manager,
    )
