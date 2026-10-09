"""Focused contracts for the A1-S3 Raw-IRI Analysis-only experiment."""

import torch
from torch import nn

from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.train_fsia import _make_optimizer, _set_training_stage


def _raw_model():
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
    config = {
        'alt_range': (120.0, 500.0),
        'seq_len': 36,
        'basis_dim': 16,
        'sw_hidden_dim': 8,
        'sw_lstm_layers': 1,
        'sw_out_dim': 16,
        'enkf_n_members': 8,
        'enkf_pert_hidden': 8,
        'use_sw_freq': False,
        'background_semantics': 'raw_iri_fixed_v1',
        'background_training_scope': 'analysis_only_v1',
        'background_residual_enabled': False,
        'r_source': 'raw_iri_train_only_recalibrated_v1',
    }
    return FSIA_INR_Model(proxy, config)


def _inputs(batch=3):
    coords = torch.tensor([
        [-12.0, -76.8, 250.0, 48.0],
        [10.0, 120.0, 300.0, 120.0],
        [45.0, 20.0, 400.0, 360.0],
    ][:batch])
    return coords, torch.zeros(batch, 36, 2)


def test_raw_iri_background_is_exactly_iri():
    model = _raw_model().eval()
    # Make the learned residual nonzero so the test verifies the switch,
    # rather than the conservative zero initialization of a fresh model.
    with torch.no_grad():
        model.background_decoder[-1].weight.fill_(0.25)
        model.background_decoder[-1].bias.fill_(0.1)
    coords, sw_seq = _inputs()
    m11, _, _, _, extras = model(coords, sw_seq)
    torch.testing.assert_close(extras['ne_bkg'], extras['ne_iri'], rtol=0, atol=1e-7)
    torch.testing.assert_close(m11, extras['ne_iri'], rtol=0, atol=1e-7)
    torch.testing.assert_close(extras['background_residual'], torch.zeros_like(extras['ne_iri']))


def test_no_observation_analysis_falls_back_to_raw_iri():
    model = _raw_model().eval()
    coords, sw_seq = _inputs()
    prediction, _, _, _, extras = model(coords, sw_seq)
    torch.testing.assert_close(prediction, extras['ne_bkg'], rtol=0, atol=1e-7)
    torch.testing.assert_close(extras['ne_bkg'], extras['ne_iri'], rtol=0, atol=1e-7)


class _StageModel(nn.Module):
    """Small module-shaped double for stage/optimizer scope contracts."""

    def __init__(self):
        super().__init__()
        self.iri_proxy = _FrozenLinear(1, 1)
        self.iri_align_net = nn.Linear(1, 2)
        self.background_decoder = nn.Linear(1, 2)
        self.sw_encoder = nn.Linear(1, 2)
        self.kalman_layer = nn.Linear(1, 2)
        self.density_basis_decoder = nn.Linear(1, 2)
        self.use_sw_freq = False

    def train(self, mode=True):
        super().train(mode)
        return self


class _FrozenLinear(nn.Linear):
    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def test_analysis_only_freezes_background_and_optimizer_scope():
    model = _StageModel()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    _set_training_stage(model, 'analysis')
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {
        'kalman_layer.weight', 'kalman_layer.bias',
        'density_basis_decoder.weight', 'density_basis_decoder.bias',
    }
    optimizer, _ = _make_optimizer(model, {'lr': 3e-4, 'weight_decay': 1e-4}, 10)
    optimized = {name for group in optimizer.param_groups for parameter in group['params']
                 for name, value in model.named_parameters() if parameter is value}
    assert optimized == trainable
    loss = sum(value.square().sum() for value in model.parameters() if value.requires_grad)
    loss.backward()
    optimizer.step()
    for name, value in model.named_parameters():
        if name not in trainable:
            torch.testing.assert_close(value, before[name])
