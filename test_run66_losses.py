"""Runnable regression checks for the v7 density-observation ETKF."""

import torch

from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model, NeuralETKFLayer
from inr_modules.mdia.physics_losses_mdia import (
    profile_huber_loss,
    second_difference_loss,
)
from inr_modules.mdia.sliding_dataset import attach_observation_background


def _layer():
    layer = NeuralETKFLayer(
        d_model=1, b_net_in=6, n_members=3, pert_hidden=2,
        r_fy=0.04, r_cosmic=0.04)
    fixed = torch.tensor([[[-1.0], [0.0], [1.0]]])
    layer._eval_perturbations = lambda _: fixed
    return layer


def _observations(value, valid=True):
    return {
        'value': torch.tensor([[value]]),
        'background': torch.zeros(1, 1),
        'valid_mask': torch.tensor([[valid]]),
        'rho_squared': torch.zeros(1, 1),
    }


def _run_layer(fy=None, cosmic=None):
    layer = _layer()
    empty_phi = torch.zeros(1, 0, 1)
    sources = {}
    if fy is not None:
        sources['FY'] = (fy, torch.ones(1, 1, 1))
    if cosmic is not None:
        sources['COSMIC'] = (cosmic, torch.ones(1, 1, 1))
    result = layer(
        torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1),
        sources, *[torch.zeros(1) for _ in range(5)])
    return layer, result


def test_analytic_kalman_and_sign():
    _, result = _run_layer(fy=_observations(0.2))
    expected = (1.0 / 1.04) * 0.2
    assert abs(result['delta_FY'].item() - expected) < 1e-6
    _, negative = _run_layer(fy=_observations(-0.2))
    assert negative['delta_FY'].item() < 0
    _, zero = _run_layer(fy=_observations(0.0))
    assert abs(zero['delta_FY'].item()) < 1e-7


def test_joint_source_and_square_root_invariants():
    _, joint = _run_layer(
        fy=_observations(0.1), cosmic=_observations(-0.03))
    assert torch.allclose(
        joint['delta_FY'] + joint['delta_COSMIC'],
        torch.einsum(
            'bn,bn->b', joint['query_anomalies'],
            joint['weights_FY'] + joint['weights_COSMIC']),
        atol=1e-7)
    _, swapped = _run_layer(
        fy=_observations(-0.03), cosmic=_observations(0.1))
    assert torch.allclose(
        joint['delta_FY'] + joint['delta_COSMIC'],
        swapped['delta_FY'] + swapped['delta_COSMIC'], atol=1e-7)
    system_inverse = torch.linalg.inv(joint['system'])
    expected_covariance = 2.0 * system_inverse
    actual_covariance = (
        joint['analysis_anomalies'].transpose(1, 2)
        @ joint['analysis_anomalies'] / 2.0)
    prior = torch.tensor([[[-1.0], [0.0], [1.0]]])
    prior_columns = prior.transpose(1, 2)
    expected_latent_covariance = (
        prior_columns @ expected_covariance @ prior_columns.transpose(1, 2)
        / 2.0)
    assert torch.allclose(
        actual_covariance,
        expected_latent_covariance,
        atol=1e-6)


def test_profile_loss_and_curvature():
    target = torch.zeros(4, 1)
    prediction = torch.tensor([[0.0], [0.4], [0.2], [0.2]])
    ids = torch.tensor([0, 0, 1, 1])
    original = profile_huber_loss(prediction, target, ids, delta=0.2)
    duplicated = profile_huber_loss(
        torch.cat([prediction[:2].repeat_interleave(3, 0), prediction[2:]]),
        torch.zeros(8, 1),
        torch.tensor([0] * 6 + [1] * 2),
        delta=0.2)
    assert torch.allclose(original, duplicated)
    assert second_difference_loss(torch.tensor([[0.0, 1.0, 2.0]])) == 0
    assert second_difference_loss(torch.tensor([[0.0, 2.0, 0.0]])) > 0


def _model():
    config = {
        'alt_range': (120.0, 500.0), 'seq_len': 4, 'basis_dim': 8,
        'sw_hidden_dim': 4, 'sw_lstm_layers': 1, 'sw_out_dim': 8,
        'tau_kp_init': 8.0, 'tau_solar_init': 72.0,
        'enkf_n_members': 4, 'enkf_pert_hidden': 8, 'use_sw_freq': False,
    }
    return FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), config)


def _model_observation(coords, background, source):
    return {
        'coords': coords[:, None, :],
        'value': background.detach() + 0.1,
        'background': background.detach(),
        'valid_mask': torch.ones(len(coords), 1, dtype=torch.bool),
        'profile_id': torch.arange(len(coords))[:, None],
        'source': torch.full((len(coords), 1), source, dtype=torch.int8),
        'rho_squared': torch.zeros(len(coords), 1),
    }


def test_model_m00_padding_continuity_and_gradients():
    torch.manual_seed(42)
    model = _model()
    coords = torch.tensor([
        [-12.0, -76.8, 250.0, 48.0],
        [-12.0, -76.8, 251.0, 48.0],
    ])
    sw = torch.zeros(2, 4, 2)
    peak = torch.tensor([[300.0, 11.5], [300.0, 11.5]])
    m00, _, _, delta00, extras = model(coords, sw, iri_peak=peak)
    assert torch.equal(m00, extras['ne_bkg'])
    assert torch.equal(delta00, torch.zeros_like(delta00))
    fy = _model_observation(coords, extras['ne_bkg'], 0)
    padded = {key: value.clone() for key, value in fy.items()}
    for key, fill in (
            ('coords', 0.0), ('value', 0.0), ('background', 0.0),
            ('rho_squared', 1.0), ('profile_id', -1), ('source', 0)):
        shape = list(padded[key].shape)
        shape[1] = 1
        padded[key] = torch.cat(
            [padded[key], torch.full(
                shape, fill, dtype=padded[key].dtype)], dim=1)
    padded['valid_mask'] = torch.cat([
        padded['valid_mask'], torch.zeros(2, 1, dtype=torch.bool)], dim=1)
    regular = model(coords, sw, iri_peak=peak, observations_fy=fy)[0]
    with_padding = model(
        coords, sw, iri_peak=peak, observations_fy=padded)[0]
    assert torch.allclose(regular, with_padding, atol=1e-7)
    shifted = coords.clone()
    shifted[:, 2] += 1e-3
    shifted_output = model(
        shifted, sw, iri_peak=peak, observations_fy=fy)[0]
    assert torch.isfinite(shifted_output).all()
    assert (shifted_output - regular).abs().max() < 1e-3

    for source_name, source_code in (
            ('observations_fy', 0), ('observations_cosmic', 1)):
        model.zero_grad()
        observation = _model_observation(
            coords, extras['ne_bkg'], source_code)
        output = model(
            coords, sw, iri_peak=peak, **{source_name: observation})[0]
        output.sum().backward()
        assert model.density_basis_decoder[-1].weight.grad.abs().sum() > 0
        assert model.kalman_layer.P_w2.grad.abs().sum() > 0


def test_observation_background_deduplication():
    class _SW:
        @staticmethod
        def get_drivers_sequence(relative_hour):
            return relative_hour[:, None, None].expand(-1, 2, 2)

    class _Background:
        @staticmethod
        def encode_background(coords, sw_seq, iri_peak=None):
            return {'ne_bkg': (
                coords[:, :3].sum(dim=1, keepdim=True)
                + sw_seq[:, 0, 0:1])}

    coords = torch.tensor([[
        [-12.0, -76.8, 250.0, 48.0],
        [-12.0, -76.8, 250.0, 48.0],
        [-12.0, -76.8, 260.0, 48.0],
    ]])
    payload = {
        'coords': coords,
        'value': torch.zeros(1, 3),
        'valid_mask': torch.tensor([[True, True, False]]),
    }
    attached = attach_observation_background(payload, _Background(), _SW())
    direct = _Background.encode_background(
        coords[:, :2].reshape(-1, 4),
        _SW.get_drivers_sequence(coords[:, :2, 3].reshape(-1)))['ne_bkg']
    assert torch.equal(attached['background'][0, :2], direct.flatten())
    assert attached['background'][0, 2] == 0


if __name__ == '__main__':
    test_analytic_kalman_and_sign()
    test_joint_source_and_square_root_invariants()
    test_profile_loss_and_curvature()
    test_model_m00_padding_continuity_and_gradients()
    test_observation_background_deduplication()
    print('run66 v7 ETKF regression checks passed')
