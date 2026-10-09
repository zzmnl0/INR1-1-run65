"""Strict manifest-backed loading for complete FSIA Analysis checkpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .v14_contract import (
    A2S1_DOMAIN_SEMANTICS,
    canonical_a2s1_config,
    canonical_a2s1_training_protocol,
    HYBRID_DOMAIN_SEMANTICS,
    canonical_v14_config,
    canonical_v14_training_protocol,
    valid_v14_gram_calibration,
)
from .m00_observation import (
    validate_frozen_m00_config,
    validate_low_dim_assimilation_config,
)


_DOMAIN_CONTRACTS = {
    'legacy_120_500_domain_v1': {
        'format_version': 12,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (120.0, 500.0),
        'peak_search_alt_range': (120.0, 500.0),
    },
    'strict_200_500_domain_v1': {
        'format_version': 13,
        'alt_range': (200.0, 500.0),
        'observation_alt_range': (200.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
    },
    'hybrid_120_500_model_200_500_observation_v1': {
        'format_version': 14,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (200.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
    },
    A2S1_DOMAIN_SEMANTICS: {
        'format_version': 15,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (120.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
    },
}

_STRICT_LOW_PRIOR_DOMAINS = (HYBRID_DOMAIN_SEMANTICS, A2S1_DOMAIN_SEMANTICS)

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_v14_input_identity(manifest, summary):
    # A1-S2 deliberately has no date/locked-test manifest. Its ISR contract
    # is the explicit profile split; callers that opt out of hash verification
    # must not be forced to re-hash the large input products here.
    config = manifest.get('config', {})
    if config.get('profile_split_semantics') == (
            'source_time_local_block80_dev20_v1'):
        if not summary.get('profile_split'):
            raise ValueError('A1-S2 summary lacks profile_split identity')
        return
    identities = manifest.get('data_identity')
    expected = manifest.get('config', {}).get('input_data_sha256')
    resolved = manifest.get('resolved_training', {})
    if not isinstance(identities, dict) or not isinstance(expected, dict):
        raise ValueError('v14 manifest lacks input_data_sha256 identities')
    actual = {}
    for key, identity in identities.items():
        if not isinstance(identity, dict) or not identity.get('path'):
            raise ValueError(f'v14 manifest input identity is invalid for {key}')
        digest = _sha256(identity['path'])
        if digest != identity.get('sha256'):
            raise ValueError(f'v14 input SHA256 differs from manifest for {key}')
        actual[key] = digest
    if 'date_split_manifest' not in actual:
        raise ValueError('v14 manifest lacks date_split_manifest identity')
    if actual != expected:
        raise ValueError('v14 manifest config input_data_sha256 differs from data identity')
    if summary.get('input_data_sha256') != actual:
        raise ValueError('v14 summary input_data_sha256 differs from data identity')
    if resolved.get('input_data_sha256') != actual:
        raise ValueError('v14 resolved training input_data_sha256 differs from data identity')


def load_fsia_analysis_checkpoint(checkpoint, device='cpu', require_domain=None,
                                  allow_historical_epoch=False,
                                  expected_sha256=None, verify_hashes=True):
    """Load a finite Analysis model using only its colocated run contract."""
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint.parent / 'run_manifest.json'
    summary_path = checkpoint.parent / 'training_summary.json'
    for path in (checkpoint, manifest_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f'required Analysis artifact is missing: {path}')
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    with summary_path.open(encoding='utf-8') as stream:
        summary = json.load(stream)
    config = manifest.get('config')
    if isinstance(config, dict):
        validate_low_dim_assimilation_config(config)
        # Loading is evaluation-only; stopped adapter routes may be inspected
        # but are never eligible for resume or training.
        validate_frozen_m00_config(config, eval_only=True)
    if not isinstance(config, dict):
        raise ValueError('run manifest lacks a config object')
    if summary.get('completed_stage') != 'analysis':
        raise ValueError('complete Analysis checkpoint requires completed_stage=analysis')
    if summary.get('checkpoint_stage') not in (None, 'analysis'):
        raise ValueError('training summary checkpoint_stage is not analysis')
    actual_sha = None
    if verify_hashes:
        actual_sha = _sha256(checkpoint)
        if summary.get('checkpoint_sha256') != actual_sha:
            parts = checkpoint.stem.split('_')
            is_historical_epoch = (len(parts) == 3 and parts[0] == 'epoch'
                                   and parts[1].isdigit() and parts[2] == 'model')
            if not (allow_historical_epoch and is_historical_epoch
                    and expected_sha256 == actual_sha):
                raise ValueError('checkpoint SHA256 differs from training summary')
        summary = dict(summary)
        summary['checkpoint'] = str(checkpoint)
    if verify_hashes:
        summary['checkpoint_sha256'] = actual_sha

    domain = config.get(
        'model_domain_semantics', 'legacy_120_500_domain_v1')
    if require_domain is not None and domain != require_domain:
        raise ValueError(f'checkpoint domain is {domain}, expected {require_domain}')
    if domain not in _DOMAIN_CONTRACTS:
        raise ValueError(f'unsupported model-domain semantics: {domain}')
    domain_contract = _DOMAIN_CONTRACTS[domain]
    expected_version = domain_contract['format_version']
    expected_alt_range = domain_contract['alt_range']
    actual_alt_range = tuple(map(float, config.get(
        'alt_range', expected_alt_range if domain ==
        'legacy_120_500_domain_v1' else ())))
    legacy_domain_fields = domain not in _STRICT_LOW_PRIOR_DOMAINS
    actual_observation_alt_range = tuple(map(float, config.get(
        'observation_alt_range', expected_alt_range if legacy_domain_fields else ())))
    actual_peak_search_alt_range = tuple(map(float, config.get(
        'peak_search_alt_range', expected_alt_range if legacy_domain_fields else ())))
    required = {
        'checkpoint_format_version': expected_version,
        'basis_dim': 64,
        'enkf_n_members': 8,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'r_mode': 'global',
        'use_distance_localization': True,
        'use_physical_localization': True,
        'assimilation_semantics': 'continuous_physical_local_letkf',
        'neighbor_directory_semantics': 'token_exact_positive_support_v1',
        'physical_localization_space_km': 1800.0,
        'physical_localization_time_hours': 1.5,
        'representativeness_floor': 1.0,
        'use_empirical_covariance_loss': False,
    }
    def matches(actual, expected):
        if isinstance(expected, float):
            return actual is not None and np.isclose(float(actual), expected)
        return actual == expected

    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if not matches(config.get(key), expected)
    }
    if actual_alt_range != expected_alt_range:
        mismatches['alt_range'] = (actual_alt_range, expected_alt_range)
    if actual_observation_alt_range != domain_contract['observation_alt_range']:
        mismatches['observation_alt_range'] = (
            actual_observation_alt_range,
            domain_contract['observation_alt_range'])
    if actual_peak_search_alt_range != domain_contract['peak_search_alt_range']:
        mismatches['peak_search_alt_range'] = (
            actual_peak_search_alt_range,
            domain_contract['peak_search_alt_range'])
    if config.get('representativeness_kernel_path') is not None:
        mismatches['representativeness_kernel_path'] = (
            config.get('representativeness_kernel_path'), None)
    if (config.get('background_trust_gate_enabled', False)
            and config.get('background_trust_gate_semantics') !=
            'fixed_altitude_localtime_dip_smoothstep_v1'):
        mismatches['background_trust_gate_semantics'] = (
            config.get('background_trust_gate_semantics'),
            'fixed_altitude_localtime_dip_smoothstep_v1')
    expected_protocol = None
    if domain in _STRICT_LOW_PRIOR_DOMAINS:
        expected_protocol = (
            canonical_a2s1_training_protocol(False)
            if domain == A2S1_DOMAIN_SEMANTICS
            else canonical_v14_training_protocol(False))
        expected_config = (
            canonical_a2s1_config(False)
            if domain == A2S1_DOMAIN_SEMANTICS
            else canonical_v14_config(False))
        for key, expected in expected_config.items():
            actual = config.get(key)
            if isinstance(expected, tuple):
                actual = tuple(map(float, actual or ()))
                matches = actual == expected
            elif isinstance(expected, float):
                try:
                    actual = float(actual)
                    matches = bool(np.isclose(actual, expected))
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                mismatches[key] = (actual, expected)
        if config.get('background_seed_ckpt') is not None:
            mismatches['background_seed_ckpt'] = (
                config.get('background_seed_ckpt'), None)
        if config.get('training_protocol') != expected_protocol:
            mismatches['config.training_protocol'] = (
                config.get('training_protocol'), expected_protocol)
        if config.get('smoke_run', False) or summary.get('smoke_run', False):
            mismatches['smoke_run'] = (True, False)
        expected_summary = {
            'checkpoint_format_version': expected_version,
            'model_domain_semantics': domain,
            'alt_range': list(domain_contract['alt_range']),
            'observation_alt_range': list(
                domain_contract['observation_alt_range']),
            'peak_search_alt_range': list(
                domain_contract['peak_search_alt_range']),
            'training_protocol': expected_protocol,
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                mismatches[f'training_summary.{key}'] = (
                    summary.get(key), expected)
    profile_split_semantics = 'source_time_local_block80_dev20_v1'
    if config.get('profile_split_semantics') == profile_split_semantics:
        if not summary.get('profile_split'):
            mismatches['training_summary.profile_split'] = (
                summary.get('profile_split'), 'required')
    elif not summary.get('date_split'):
        mismatches['training_summary.date_split'] = (
            summary.get('date_split'), 'required')
    if domain in _STRICT_LOW_PRIOR_DOMAINS:
        resolved = manifest.get('resolved_training')
        if not isinstance(resolved, dict):
            mismatches['manifest.resolved_training'] = (resolved, 'required object')
        else:
            if resolved.get('training_protocol') != expected_protocol:
                mismatches['manifest.resolved_training.training_protocol'] = (
                    resolved.get('training_protocol'), expected_protocol)
            calibration = resolved.get('gram_gradient_calibration')
            if not valid_v14_gram_calibration(calibration, expected_protocol):
                mismatches['manifest.resolved_training.gram_gradient_calibration'] = (
                    calibration, 'complete valid 20-batch calibration')
            summary_calibration = summary.get(
                'observation_gram_loss', {}).get('gradient_calibration')
            if summary_calibration != calibration:
                mismatches['training_summary.observation_gram_loss.gradient_calibration'] = (
                    summary_calibration, calibration)
    if mismatches:
        raise ValueError(f'FSIA checkpoint contract mismatch: {mismatches}')
    if domain in _STRICT_LOW_PRIOR_DOMAINS and verify_hashes:
        _verify_v14_input_identity(manifest, summary)

    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    from inr_modules.mdia.fsia_model import FSIA_INR_Model
    device = torch.device(device)
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(
        config['iri_proxy_path'], map_location=device, weights_only=True))
    iri_proxy.eval()
    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(
            torch.isfinite(value).all()
            for value in state.values() if torch.is_tensor(value)):
        raise ValueError('FSIA checkpoint contains non-finite tensors')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, manifest, summary


def allowed_observation_profile_ids(config):
    """Return train+development satellite profile IDs for external inference."""
    from inr_modules.data_managers.FY_dataloader import (
        COSMICDataset, FY3D_Dataset)

    if config.get('profile_split_semantics') == (
            'source_time_local_block80_dev20_v1'):
        split_path = Path(config['profile_split_path']).resolve()
        with np.load(split_path, allow_pickle=False) as split:
            return {
                source: np.unique(np.concatenate((
                    np.asarray(split[f'{source}_train_profile_ids'], dtype=np.int64),
                    np.asarray(split[f'{source}_development_profile_ids'],
                               dtype=np.int64))))
                for source in ('FY', 'COSMIC')
            }

    with Path(config['date_split_manifest']).open(encoding='utf-8') as stream:
        partitions = json.load(stream)['partitions']
    days = set(partitions['train']) | set(partitions['development'])
    if str(config.get('date_split_semantics', '')).replace('-', '_') == 'a1s_train_plus_locked_dev5_v1':
        days |= set(partitions.get('locked_test', []))
    days = sorted(days)
    common = dict(
        mode='observation', val_days=[],
        bin_size_hours=config['bin_size_hours'], use_memmap=True,
        val_ratio=None, split_seed=config['seed'],
        split_days={'observation': days}, alt_range=(
            config.get('observation_alt_range') or config['alt_range']))
    fy = FY3D_Dataset(
        config['fy_path'], profile_path=config.get('fy_profile_path'),
        profile_index_path=config.get('fy_profile_index_path'), **common)
    cosmic = COSMICDataset(
        config['cosmic_path'],
        profile_index_path=config.get('cosmic_profile_index_path'), **common)
    return {'FY': np.unique(fy.profile_ids),
            'COSMIC': np.unique(cosmic.profile_ids)}


def shared_full30_profile_allowlist(config, expected_counts=None):
    """Validate and return the A1-S2 Train+Development full-domain allowlist."""
    semantics = 'source_time_local_block80_dev20_v1'
    if config.get('profile_split_semantics') != semantics:
        raise ValueError('shared-full30 requires the A1-S2 profile split manifest')
    split_path = Path(config.get('profile_split_path', '')).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f'shared-full30 profile split is missing: {split_path}')
    expected = expected_counts or {'FY': 66612, 'COSMIC': 84364}
    allowed, audit = {}, {}
    with np.load(split_path, allow_pickle=False) as split:
        for source in ('FY', 'COSMIC'):
            train = np.asarray(split[f'{source}_train_profile_ids'], dtype=np.int64).reshape(-1)
            development = np.asarray(split[f'{source}_development_profile_ids'], dtype=np.int64).reshape(-1)
            if np.unique(train).size != train.size or np.unique(development).size != development.size:
                raise ValueError(f'{source} profile IDs contain duplicates')
            if np.intersect1d(train, development).size:
                raise ValueError(f'{source} Train/Development profile IDs overlap')
            ids = np.unique(np.concatenate((train, development)))
            if ids.size != expected[source]:
                raise ValueError(f'{source} shared-full30 count {ids.size} != expected {expected[source]}')
            index_path = Path(config.get(f'{source.lower()}_profile_index_path', '')).resolve()
            if not index_path.is_file():
                raise FileNotFoundError(f'{source} profile index is missing: {index_path}')
            with np.load(index_path, allow_pickle=False) as index:
                indexed = np.asarray(index['profile_id'], dtype=np.int64).reshape(-1)
            if not np.all(np.isin(ids, indexed)):
                raise ValueError(f'{source} shared-full30 IDs missing from index')
            allowed[source] = ids
            audit[source] = {'train_count': int(train.size), 'development_count': int(development.size),
                             'all_count': int(ids.size), 'index_count': int(indexed.size),
                             'manifest_path': str(split_path)}
    return allowed, audit
