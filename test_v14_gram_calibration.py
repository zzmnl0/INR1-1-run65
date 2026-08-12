"""Focused regression coverage for the v14 Gram calibration contract."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from inr_modules.mdia import train_fsia as train_module
from inr_modules.mdia.train_fsia import (
    _current_stage_parameter_counts,
    _current_stage_trainable_parameters,
    _gradient_norms,
    _low_altitude_prior_protocol,
    _resolve_gram_weight,
    _resolve_input_data_sha256,
    _set_training_stage,
    _training_protocol_signature,
    _validate_v14_resume_state,
    _write_smoke_preflight,
)
from inr_modules.mdia.v14_contract import (
    HYBRID_DOMAIN_SEMANTICS,
    canonical_v14_config,
    canonical_v14_low_altitude_protocol,
    canonical_v14_training_protocol,
    resolve_v14_gram_weight,
    valid_v14_gram_calibration,
)


class _Proxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class _TinyAnalysisModel(torch.nn.Module):
    """Shape-matched stand-in for the v14 Analysis parameter contract."""
    def __init__(self):
        super().__init__()
        self.iri_proxy = _Proxy()
        self.iri_align_net = torch.nn.Linear(1, 1)
        self.background_decoder = torch.nn.Linear(1, 1)
        self.sw_encoder = torch.nn.Linear(1, 1)
        self.use_sw_freq = False
        self.kalman_layer = torch.nn.Module()
        self.kalman_layer.covariance_scale_net = torch.nn.Sequential(
            torch.nn.Linear(69, 64), torch.nn.ReLU(), torch.nn.Linear(64, 7))
        self.kalman_layer.last_member_weights = torch.tensor([1.0])
        self.kalman_layer.last_inflation_scale = torch.tensor(1.0)
        self.density_basis_decoder = torch.nn.Sequential(
            torch.nn.Linear(140, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))


def _full_config(**overrides):
    config = canonical_v14_config(False)
    config.update({
        'basis_dim': 64,
        'enkf_n_members': 8,
        'r_mode': 'global',
        'use_amp': False,
    })
    config.update(overrides)
    return config


def _calibration(raw_ratio=10.0):
    raw = [float(raw_ratio)] * 20
    resolved, median = resolve_v14_gram_weight(raw, 0.02)
    return {
        'requested_batches': 20,
        'processed_batches': 20,
        'valid_batches': 20,
        'raw_ratios': raw,
        'raw_ratio_summary': {
            'min': median, 'q1': median, 'median': median,
            'q3': median, 'max': median,
        },
        'median_ratio': median,
        'target_ratio': 0.02,
        'resolved_weight': resolved,
        'achieved_median_ratio': resolved * median,
        'parameter_tensor_count': 8,
        'parameter_count': 18119,
        'gradient_parameter_scope': (
            'all_current_stage_trainable_parameters_v1'),
        'calibration_precision': 'match_analysis_autocast_v1',
        'weight_resolution_semantics': (
            'target_over_median_no_lower_clip_reject_gt_one_v1'),
        'require_all_batches_finite_positive': True,
    }


def _equal_numpy_state(left, right):
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


def test_v14_weight_is_not_clipped_and_invalid_ratios_reject():
    weight, median = resolve_v14_gram_weight([6923.897949] * 20, 0.02)
    assert median == pytest.approx(6923.897949)
    assert weight == pytest.approx(2.8885463284820188e-6)
    assert weight < 1e-4
    for ratios in ([0.0], [-1.0], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            resolve_v14_gram_weight(ratios, 0.02)
    with pytest.raises(ValueError):
        resolve_v14_gram_weight([0.01], 0.02)


def test_v14_protocol_is_canonical_and_calibration_tampering_rejects():
    config = _full_config()
    protocol = canonical_v14_training_protocol(False)
    assert _training_protocol_signature(config) == protocol
    assert _low_altitude_prior_protocol(config) == canonical_v14_low_altitude_protocol()
    record = _calibration()
    assert valid_v14_gram_calibration(record, protocol)
    for key, value in (
            ('processed_batches', 19),
            ('parameter_count', 1),
            ('resolved_weight', 0.1),
            ('gradient_parameter_scope', 'decoder_only')):
        tampered = copy.deepcopy(record)
        tampered[key] = value
        assert not valid_v14_gram_calibration(tampered, protocol)


def test_analysis_scope_includes_kalman_and_decoder_gradients():
    model = _TinyAnalysisModel()
    _set_training_stage(model, 'analysis')
    assert _current_stage_parameter_counts(model) == (8, 18119)
    parameters = _current_stage_trainable_parameters(model)
    observation = model.density_basis_decoder[0].weight.sum()
    kalman_only = model.kalman_layer.covariance_scale_net[0].weight.sum()
    _, auxiliary_norm, ratio, _ = _gradient_norms(
        observation, kalman_only, parameters)
    assert auxiliary_norm.item() > 0.0
    assert ratio.item() > 0.0


def test_v14_calibration_processes_exactly_twenty_and_preserves_state(monkeypatch):
    model = _TinyAnalysisModel()
    _set_training_stage(model, 'analysis')
    model.eval()
    model.kalman_layer.train()
    model.kalman_layer.last_member_weights = torch.tensor([2.0])
    model.kalman_layer.last_inflation_scale = torch.tensor(3.0)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    state_before = {key: value.detach().clone()
                    for key, value in model.state_dict().items()}
    grads_before = [parameter.grad.detach().clone()
                    for parameter in model.parameters()]
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def fake_losses(model, _processor, fy_batch, _cosmic_batch, *_args, **_kwargs):
        observation = model.density_basis_decoder[0].weight.reshape(-1)[0]
        gram = float(fy_batch) * model.kalman_layer.covariance_scale_net[
            0].weight.reshape(-1)[0]
        return observation, observation * 0.0, observation * 0.0, gram

    monkeypatch.setattr(train_module, '_paired_analysis_losses', fake_losses)
    config = _full_config()
    raw = 6923.897949
    weight = _resolve_gram_weight(
        model, [raw] * 20, [0] * 20, None, torch.device('cpu'), config,
        None, None)
    assert weight == pytest.approx(2.8885463284820188e-6)
    record = config['gram_gradient_calibration']
    assert record['processed_batches'] == record['valid_batches'] == 20
    assert record['parameter_tensor_count'] == 8
    assert record['parameter_count'] == 18119
    assert valid_v14_gram_calibration(
        record, canonical_v14_training_protocol(False))
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])
    for parameter, gradient in zip(model.parameters(), grads_before):
        assert torch.equal(parameter.grad, gradient)
    assert not model.training and model.kalman_layer.training
    assert torch.equal(model.kalman_layer.last_member_weights, torch.tensor([2.0]))
    assert torch.equal(model.kalman_layer.last_inflation_scale, torch.tensor(3.0))
    assert _equal_numpy_state(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    with pytest.raises(RuntimeError, match='20 paired'):
        _resolve_gram_weight(
            model, [raw] * 19, [0] * 20, None, torch.device('cpu'),
            _full_config(), None, None)


def test_smoke_preflight_uses_calibration_not_single_batch_or_coverage(tmp_path):
    config = _full_config(save_dir=str(tmp_path), smoke_run=True)
    config['gram_gradient_calibration'] = _calibration()
    _write_smoke_preflight(
        config, 'analysis', 1, 0,
        {'total_auxiliary_to_observation': 0.10,
         'low_altitude_to_observation': 0.01,
         'gram_to_observation': 0.80},
        {'fy_M10': 0.34375, 'fy_M01': 0.6, 'fy_M11': 0.7,
         'cosmic_M10': 0.8, 'cosmic_M01': 0.9, 'cosmic_M11': 1.0})
    saved = json.loads((tmp_path / 'smoke_preflight.json').read_text(
        encoding='utf-8'))
    assert saved['stages']['analysis']['passed']
    assert saved['stages']['analysis']['gram_coverage_diagnostic']['fy_M10'] < 0.5
    with pytest.raises(RuntimeError, match='total auxiliary'):
        _write_smoke_preflight(
            config, 'background', 0, 0,
            {'total_auxiliary_to_observation': 0.251,
             'low_altitude_to_observation': 0.01,
             'gram_to_observation': 0.0}, {})
    saved = json.loads((tmp_path / 'smoke_preflight.json').read_text(
        encoding='utf-8'))
    assert not saved['stages']['background']['passed']


def test_v14_resume_and_input_identity_rejections(tmp_path):
    protocol = canonical_v14_training_protocol(False)
    hashes = {'date_split_manifest': 'a' * 64}
    background = {
        'smoke_run': False,
        'stage': 'background',
        'training_protocol': protocol,
        'input_data_sha256': hashes,
    }
    _validate_v14_resume_state(background, protocol, hashes)
    analysis = dict(background, stage='analysis')
    with pytest.raises(ValueError, match='Gram calibration'):
        _validate_v14_resume_state(analysis, protocol, hashes)
    analysis['gram_gradient_calibration'] = _calibration()
    _validate_v14_resume_state(analysis, protocol, hashes)
    with pytest.raises(ValueError, match='smoke'):
        _validate_v14_resume_state(
            dict(background, smoke_run=True), protocol, hashes)
    with pytest.raises(ValueError, match='input data SHA256'):
        _validate_v14_resume_state(background, protocol, {'date_split_manifest': 'b' * 64})

    data = tmp_path / 'input.bin'
    split = tmp_path / 'date_split_manifest.json'
    data.write_bytes(b'input')
    split.write_text('{"partitions": {}}', encoding='utf-8')
    identity = {
        'date_split_manifest': {
            'path': str(split), 'sha256': hashlib.sha256(split.read_bytes()).hexdigest()},
        'fy_path': {'path': str(data), 'sha256': hashlib.sha256(data.read_bytes()).hexdigest()},
    }
    mapping = {key: value['sha256'] for key, value in identity.items()}
    (tmp_path / 'run_manifest.json').write_text(json.dumps({
        'config': {'input_data_sha256': mapping}, 'data_identity': identity}),
        encoding='utf-8')
    config = {
        'save_dir': str(tmp_path),
        'model_domain_semantics': HYBRID_DOMAIN_SEMANTICS,
        'date_split_manifest': str(split),
        'fy_path': str(data),
    }
    assert _resolve_input_data_sha256(config) == mapping
    data.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='input SHA256'):
        _resolve_input_data_sha256(config)
