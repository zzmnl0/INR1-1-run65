"""Strict manifest-backed loading for complete FSIA Analysis checkpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


_DOMAIN_CONTRACTS = {
    'legacy_120_500_domain_v1': (12, (120.0, 500.0)),
    'strict_200_500_domain_v1': (13, (200.0, 500.0)),
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
    expected_version, expected_alt_range = _DOMAIN_CONTRACTS[domain]
    actual_alt_range = tuple(map(float, config.get(
        'alt_range', expected_alt_range if domain ==
        'legacy_120_500_domain_v1' else ())))
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
    if config.get('representativeness_kernel_path') is not None:
        mismatches['representativeness_kernel_path'] = (
            config.get('representativeness_kernel_path'), None)
    if (config.get('background_trust_gate_enabled', False)
            and config.get('background_trust_gate_semantics') !=
            'fixed_altitude_localtime_dip_smoothstep_v1'):
        mismatches['background_trust_gate_semantics'] = (
            config.get('background_trust_gate_semantics'),
            'fixed_altitude_localtime_dip_smoothstep_v1')
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
        split_days={'observation': days}, alt_range=config['alt_range'])
    fy = FY3D_Dataset(
        config['fy_path'], profile_path=config.get('fy_profile_path'),
        profile_index_path=config.get('fy_profile_index_path'), **common)
    cosmic = COSMICDataset(
        config['cosmic_path'],
        profile_index_path=config.get('cosmic_profile_index_path'), **common)
    return {'FY': np.unique(fy.profile_ids),
            'COSMIC': np.unique(cosmic.profile_ids)}
