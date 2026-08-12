"""Strict manifest-backed loading for complete FSIA Analysis checkpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


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
}

_HYBRID_LOW_ALTITUDE_PROTOCOL = {
    'low_altitude_prior_range': (120.0, 200.0),
    'low_altitude_prior_semantics': (
        'soft_iri_background_zero_analysis_increment_v1'),
    'low_altitude_anchor_levels_km': tuple(float(value) for value in range(120, 200, 10)),
    'low_altitude_anchor_profiles_per_source': 16,
    'w_low_altitude_background_iri': 0.02,
    'w_low_altitude_analysis_increment': 0.01,
}

_HYBRID_TRAINING_PROTOCOL = {
    'seed': 42,
    'smoke_run': False,
    'background_epochs': 5,
    'analysis_epochs': 10,
    'background_trust_gate_enabled': False,
    'low_altitude_anchor_selection': (
        'first_16_unique_profiles_per_source_per_batch_first_record_v1'),
    'low_altitude_anchor_grouping': 'source_profile_id_profile_balanced_v1',
    'low_altitude_neighbor_profile_semantics': (
        'synthetic_query_no_target_profile_exclusion_v1'),
    'low_altitude_prior_protocol': {
        'range_km': [120.0, 200.0],
        'semantics': 'soft_iri_background_zero_analysis_increment_v1',
        'anchor_levels_km': [float(value) for value in range(120, 200, 10)],
        'profiles_per_source': 16,
        'background_weight': 0.02,
        'analysis_weight': 0.01,
        'gradient_ratio_max': 0.25,
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_fsia_analysis_checkpoint(checkpoint, device='cpu', require_domain=None,
                                  allow_historical_epoch=False):
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
    if not isinstance(config, dict):
        raise ValueError('run manifest lacks a config object')
    if summary.get('completed_stage') != 'analysis':
        raise ValueError('complete Analysis checkpoint requires completed_stage=analysis')
    if summary.get('checkpoint_stage') not in (None, 'analysis'):
        raise ValueError('training summary checkpoint_stage is not analysis')
    actual_sha = _sha256(checkpoint)
    if summary.get('checkpoint_sha256') != actual_sha:
        parts = checkpoint.stem.split('_')
        is_historical_epoch = (len(parts) == 3 and parts[0] == 'epoch'
                               and parts[1].isdigit() and parts[2] == 'model')
        if not (allow_historical_epoch and is_historical_epoch):
            raise ValueError('checkpoint SHA256 differs from training summary')
        summary = dict(summary)
        summary['checkpoint'] = str(checkpoint)
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
    legacy_domain_fields = domain != 'hybrid_120_500_model_200_500_observation_v1'
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
    if domain == 'hybrid_120_500_model_200_500_observation_v1':
        if config.get('background_trust_gate_enabled', False):
            mismatches['background_trust_gate_enabled'] = (True, False)
        for key, expected in _HYBRID_LOW_ALTITUDE_PROTOCOL.items():
            actual = config.get(key)
            if isinstance(expected, tuple):
                actual = tuple(map(float, actual or ()))
            elif isinstance(expected, float):
                actual = None if actual is None else float(actual)
            if (not np.isclose(actual, expected)
                    if isinstance(expected, float) else actual != expected):
                mismatches[key] = (actual, expected)
        for key, expected in {
                'background_epochs': 5,
                'analysis_epochs': 10,
                'seed': 42,
                'background_seed_ckpt': None,
        }.items():
            if config.get(key) != expected:
                mismatches[key] = (config.get(key), expected)
        if config.get('smoke_run', False) or summary.get('smoke_run', False):
            mismatches['smoke_run'] = (True, False)
        expected_summary = {
            'checkpoint_format_version': 14,
            'model_domain_semantics': domain,
            'alt_range': [120.0, 500.0],
            'observation_alt_range': [200.0, 500.0],
            'peak_search_alt_range': [200.0, 500.0],
            'training_protocol': _HYBRID_TRAINING_PROTOCOL,
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                mismatches[f'training_summary.{key}'] = (
                    summary.get(key), expected)
        if not isinstance(manifest.get('data_identity'), dict):
            mismatches['manifest.data_identity'] = (
                manifest.get('data_identity'), 'non-empty object')
        if not summary.get('date_split'):
            mismatches['training_summary.date_split'] = (
                summary.get('date_split'), 'required')
    if mismatches:
        raise ValueError(f'FSIA checkpoint contract mismatch: {mismatches}')

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

    with Path(config['date_split_manifest']).open(encoding='utf-8') as stream:
        partitions = json.load(stream)['partitions']
    days = sorted(set(partitions['train']) | set(partitions['development']))
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
