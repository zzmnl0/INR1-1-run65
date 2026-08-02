"""Runnable regression checks for the v7/v8 density-observation ETKF."""

import torch

from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import (
    FSIA_INR_Model,
    NeuralETKFLayer,
    solve_density_mode,
    solve_density_modes,
)
from inr_modules.mdia.physics_losses_mdia import (
    profile_huber_loss,
    second_difference_loss,
)
from inr_modules.mdia.sliding_dataset import attach_observation_background
from inr_modules.mdia.train_fsia import (
    covariance_moment_loss,
    empirical_covariance_loss,
    exact_mode_direction_losses,
    exact_mode_profile_losses,
)


def _layer():
    layer = NeuralETKFLayer(
        d_model=1, b_net_in=6, n_members=3, pert_hidden=2,
        r_fy=0.04, r_cosmic=0.04)
    fixed = torch.tensor([[[-1.0], [0.0], [1.0]]])
    layer._eval_perturbations = lambda _: (fixed, None)
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


def test_exact_mode_solver_and_profile_losses():
    query_basis = torch.tensor(
        [[0.8, -0.2], [0.3, 0.6]], requires_grad=True)
    latent = torch.tensor([
        [[1.0, 0.0], [-0.5, 0.5], [-0.5, -0.5]],
        [[1.0, 0.0], [-0.5, 0.5], [-0.5, -0.5]],
    ])
    fy_anomalies = torch.tensor([
        [[1.0, -0.5, -0.5]],
        [[0.2, -0.1, -0.1]],
    ])
    cosmic_anomalies = torch.tensor([
        [[-0.4, 0.2, 0.2]],
        [[0.6, -0.3, -0.3]],
    ])
    extras = {
        'query_basis': query_basis,
        'latent_anomalies': latent,
        'obs_anomalies_FY': fy_anomalies,
        'obs_anomalies_COSMIC': cosmic_anomalies,
        'precision_FY': torch.tensor([[4.0], [0.0]]),
        'precision_COSMIC': torch.tensor([[0.0], [5.0]]),
        'innov_FY': torch.tensor([[-0.3], [0.2]]),
        'innov_COSMIC': torch.tensor([[0.1], [0.25]]),
        'ne_bkg': torch.zeros(2, 1),
        'r_fy': torch.tensor(0.04),
        'r_cosmic': torch.tensor(0.04),
    }
    m00 = solve_density_mode(extras, ())
    m10 = solve_density_mode(extras, ('FY',))
    m01 = solve_density_mode(extras, ('COSMIC',))
    m11 = solve_density_mode(extras, ('FY', 'COSMIC'))
    modes = solve_density_modes(extras)
    swapped = solve_density_mode(extras, ('COSMIC', 'FY'))
    assert torch.equal(m00, torch.zeros_like(m00))
    assert torch.allclose(m11, swapped, atol=1e-7)
    assert torch.allclose(modes['M10'], m10, atol=1e-7)
    assert torch.allclose(modes['M01'], m01, atol=1e-7)
    assert torch.allclose(modes['M11'], m11, atol=1e-7)
    assert m10[0] < 0 and m10[1] == 0
    assert m01[0] == 0 and m01[1] > 0

    target = torch.tensor([[-0.2], [0.2]])
    total, losses, active = exact_mode_profile_losses(
        extras, target, torch.tensor([0, 1]), delta=0.2)
    expected = (
        0.25 * losses['M10']
        + 0.25 * losses['M01']
        + 0.50 * losses['M11'])
    assert torch.allclose(total, expected)
    assert active['M10'].tolist() == [True, False]
    assert active['M01'].tolist() == [False, True]
    assert active['M11'].tolist() == [True, True]
    m10_grad = torch.autograd.grad(
        losses['M10'], query_basis, retain_graph=True)[0]
    m01_grad = torch.autograd.grad(
        losses['M01'], query_basis, retain_graph=True)[0]
    assert torch.isfinite(m10_grad).all() and torch.isfinite(m01_grad).all()
    assert m10_grad[0].abs().sum() > 0 and m10_grad[1].abs().sum() == 0
    assert m01_grad[0].abs().sum() == 0 and m01_grad[1].abs().sum() > 0

    correct_direction = exact_mode_direction_losses(
        extras, target, torch.tensor([0, 1]), 'FY')[0]
    assert correct_direction == 0
    reverse_target = -target
    direction_total, direction_modes, direction_active = (
        exact_mode_direction_losses(
            extras, reverse_target, torch.tensor([0, 1]), 'FY'))
    assert direction_total > 0
    assert direction_modes['M10'] > 0
    assert direction_modes['M01'] > 0
    assert direction_active['M11'].tolist() == [True, True]
    direction_grad = torch.autograd.grad(
        direction_total, query_basis, retain_graph=True)[0]
    assert torch.isfinite(direction_grad).all()
    assert direction_grad.abs().sum() > 0
    below_threshold = exact_mode_direction_losses(
        extras, torch.full((2, 1), 0.04),
        torch.tensor([0, 1]), 'FY')[0]
    assert below_threshold == 0

    try:
        solve_density_mode(extras, ('FY', 'FY'))
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate sources must be rejected')


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
    masked_prediction = torch.tensor(
        [[0.4], [99.0], [0.2], [-99.0]], requires_grad=True)
    masked_ids = torch.tensor([0, 0, 1, 2])
    valid = torch.tensor([True, False, True, False])
    masked = profile_huber_loss(
        masked_prediction, target, masked_ids, delta=0.2, valid_mask=valid)
    expected = profile_huber_loss(
        masked_prediction[[0, 2]], target[[0, 2]],
        masked_ids[[0, 2]], delta=0.2)
    assert torch.allclose(masked, expected)
    masked.backward()
    assert masked_prediction.grad[[1, 3]].abs().sum() == 0
    assert masked_prediction.grad[[0, 2]].abs().sum() > 0
    all_invalid = profile_huber_loss(
        masked_prediction, target, masked_ids, delta=0.2,
        valid_mask=torch.zeros(4, dtype=torch.bool))
    assert all_invalid == 0
    try:
        profile_huber_loss(
            masked_prediction, target, masked_ids, delta=0.2,
            valid_mask=torch.ones(3, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError('invalid mask shape must be rejected')
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
            ('coords', float('nan')), ('value', float('nan')),
            ('background', float('nan')), ('rho_squared', float('nan')),
            ('profile_id', -1), ('source', 0)):
        shape = list(padded[key].shape)
        shape[1] = 3
        padded[key] = torch.cat(
            [padded[key], torch.full(
                shape, fill, dtype=padded[key].dtype)], dim=1)
    padded['valid_mask'] = torch.cat([
        padded['valid_mask'], torch.zeros(2, 3, dtype=torch.bool)], dim=1)
    for key in ('coords', 'value', 'background', 'rho_squared'):
        padded[key][:, 1] = 0.0
        padded[key][:, 2] = 1e20
    regular = model(coords, sw, iri_peak=peak, observations_fy=fy)[0]
    with_padding = model(
        coords, sw, iri_peak=peak, observations_fy=padded)[0]
    assert torch.allclose(regular, with_padding, atol=1e-7)
    assert torch.equal(padded['coords'][:, 1], torch.zeros_like(
        padded['coords'][:, 1]))
    assert torch.equal(
        padded['coords'][:, 2],
        torch.full_like(padded['coords'][:, 2], 1e20))
    assert torch.isnan(padded['coords'][:, 3]).all()

    invalid = {key: value[:, 1:].clone() for key, value in padded.items()}
    for source_name in ('observations_fy', 'observations_cosmic'):
        model.zero_grad(set_to_none=True)
        output, _, _, increment, _ = model(
            coords, sw, iri_peak=peak, **{source_name: invalid})
        assert torch.equal(output, m00)
        assert torch.equal(increment, torch.zeros_like(increment))
        assert torch.isfinite(output).all()
        output.sum().backward()
        for parameter in (
                *model.density_basis_decoder.parameters(),
                model.kalman_layer.P_w1, model.kalman_layer.P_w2):
            assert parameter.grad is None or torch.count_nonzero(
                parameter.grad) == 0

    mixed = {key: value.clone() for key, value in fy.items()}
    mixed['valid_mask'][1, 0] = False
    mixed['coords'][1, 0] = float('nan')
    mixed['value'][1, 0] = float('nan')
    mixed['background'][1, 0] = float('nan')
    mixed['rho_squared'][1, 0] = float('nan')
    mixed_output = model(
        coords, sw, iri_peak=peak, observations_fy=mixed)[0]
    assert torch.allclose(mixed_output[0], regular[0], atol=1e-7)
    assert torch.equal(mixed_output[1], m00[1])

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


def test_factorized_anomaly_geometry_and_gradients():
    for members, minimum_rank in ((8, 4.0), (16, 8.5)):
        layer = NeuralETKFLayer(
            d_model=64, b_net_in=6, n_members=members, pert_hidden=8,
            anomaly_parameterization='orthogonal_factor')
        coefficients = layer.ensemble_coefficients
        rank = members - 1
        assert (coefficients @ torch.ones(
            members, dtype=coefficients.dtype)).abs().max() < 1e-7
        assert torch.allclose(
            coefficients @ coefficients.T,
            rank * torch.eye(rank, dtype=coefficients.dtype), atol=1e-7)
        assert torch.allclose(
            layer.state_basis.T @ layer.state_basis,
            torch.eye(rank, dtype=layer.state_basis.dtype), atol=1e-7)
        anomalies, scales = layer._eval_perturbations(torch.randn(3, 6))
        covariance = anomalies.transpose(1, 2) @ anomalies / rank
        factor = layer.state_basis.to(scales.dtype)[None] * scales[:, None]
        assert torch.allclose(
            covariance, factor @ factor.transpose(1, 2), atol=1e-6)
        singular = torch.linalg.svdvals(anomalies)[:, :rank]
        probability = singular.square()
        probability = probability / probability.sum(dim=1, keepdim=True)
        effective_rank = torch.exp(
            -(probability * probability.log()).sum(dim=1))
        assert effective_rank.min() >= minimum_rank
        assert (singular[:, 0] / singular[:, -1]).max() <= 3.001

    try:
        NeuralETKFLayer(
            d_model=4, n_members=6,
            anomaly_parameterization='orthogonal_factor')
    except ValueError:
        pass
    else:
        raise AssertionError('N-1 > basis_dim must be rejected')

    torch.manual_seed(42)
    config = {
        'alt_range': (120.0, 500.0), 'seq_len': 4, 'basis_dim': 8,
        'sw_hidden_dim': 4, 'sw_lstm_layers': 1, 'sw_out_dim': 8,
        'tau_kp_init': 8.0, 'tau_solar_init': 72.0,
        'enkf_n_members': 4, 'enkf_pert_hidden': 8, 'use_sw_freq': False,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
    }
    model = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), config)
    clone = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), config)
    clone.load_state_dict(model.state_dict(), strict=True)
    coords = torch.tensor([
        [-12.0, -76.8, 250.0, 48.0],
        [-12.0, -76.8, 251.0, 48.0],
    ])
    sw = torch.zeros(2, 4, 2)
    peak = torch.tensor([[300.0, 11.5], [300.0, 11.5]])
    background = model(coords, sw, iri_peak=peak)[4]['ne_bkg']
    for source_name, source_code in (
            ('observations_fy', 0), ('observations_cosmic', 1)):
        model.zero_grad()
        observation = _model_observation(coords, background, source_code)
        output = model(
            coords, sw, iri_peak=peak, **{source_name: observation})[0]
        output.sum().backward()
        assert model.density_basis_decoder[-1].weight.grad.abs().sum() > 0
        scale_grad = model.kalman_layer.covariance_scale_net[-1].weight.grad
        assert scale_grad is not None
        assert torch.isfinite(scale_grad).all() and scale_grad.abs().sum() > 0
    fy = _model_observation(coords, background, 0)
    cosmic = _model_observation(coords, background, 1)
    _, _, _, _, extras = model(
        coords, sw, iri_peak=peak,
        observations_fy=fy, observations_cosmic=cosmic)
    assert torch.allclose(
        solve_density_mode(extras, ('FY', 'COSMIC')).unsqueeze(-1),
        extras['ne_residual'], atol=1e-7)
    _, mode_losses, _ = exact_mode_profile_losses(
        extras, background.detach() + 0.25,
        torch.tensor([0, 1]), delta=0.2)
    parameters = (
        model.density_basis_decoder[-1].weight,
        model.kalman_layer.covariance_scale_net[-1].weight,
    )
    for mode in ('M10', 'M01', 'M11'):
        gradients = torch.autograd.grad(
            mode_losses[mode], parameters, retain_graph=True)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
    assert torch.equal(
        model(coords, sw, iri_peak=peak)[0],
        clone(coords, sw, iri_peak=peak)[0])


def test_coordinate_local_symmetric_covariance_and_compatibility():
    torch.manual_seed(42)
    base_config = {
        'alt_range': (120.0, 500.0), 'seq_len': 4, 'basis_dim': 8,
        'sw_hidden_dim': 4, 'sw_lstm_layers': 1, 'sw_out_dim': 8,
        'tau_kp_init': 8.0, 'tau_solar_init': 72.0,
        'enkf_n_members': 4, 'enkf_pert_hidden': 8, 'use_sw_freq': False,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
    }
    legacy = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), base_config)
    explicit_legacy = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), {
            **base_config, 'density_basis_semantics': 'query_conditioned'})
    explicit_legacy.load_state_dict(legacy.state_dict(), strict=True)
    symmetric = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), {
            **base_config,
            'density_basis_semantics': 'coordinate_local_symmetric'})
    symmetric.load_state_dict(legacy.state_dict(), strict=True)
    endpoint = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), {
            **base_config,
            'density_basis_semantics': 'endpoint_context_symmetric'})
    endpoint.load_state_dict(legacy.state_dict(), strict=True)
    coords = torch.tensor([
        [-12.0, -76.8, 180.0, 48.0],
        [65.0, 147.0, 320.0, 240.0],
    ])
    reverse = coords.flip(0)
    background = torch.tensor([[10.7], [11.4]])
    reverse_background = background.flip(0)
    z = torch.randn(2, 8)
    h_sw = torch.randn(2, 8)
    forward_phi = symmetric._density_basis(
        coords, coords[:, None], background, z, h_sw).squeeze(1)
    target_phi = symmetric._density_basis(
        coords, reverse[:, None], reverse_background, z, h_sw).squeeze(1)
    reverse_query_phi = symmetric._density_basis(
        reverse, reverse[:, None], reverse_background,
        z.flip(0), h_sw.flip(0)).squeeze(1)
    reverse_target_phi = symmetric._density_basis(
        reverse, coords[:, None], background,
        z.flip(0), h_sw.flip(0)).squeeze(1)
    anomalies_a, _ = symmetric.kalman_layer._eval_perturbations(
        torch.randn(2, 13))
    anomalies_b, _ = symmetric.kalman_layer._eval_perturbations(
        torch.randn(2, 13))
    assert torch.equal(anomalies_a, anomalies_b)

    def covariance(left, right, anomalies):
        left_y = torch.einsum('bd,bnd->bn', left, anomalies)
        right_y = torch.einsum('bd,bnd->bn', right, anomalies)
        return (left_y * right_y).sum(-1) / 3.0

    forward = covariance(forward_phi, target_phi, anomalies_a)
    backward = covariance(
        reverse_query_phi, reverse_target_phi, anomalies_b).flip(0)
    assert (forward - backward).abs().max() < 1e-7

    endpoint_query_phi = endpoint._density_basis(
        coords, coords[:, None], background, z, h_sw,
        z[:, None], h_sw[:, None]).squeeze(1)
    endpoint_target_phi = endpoint._density_basis(
        coords, reverse[:, None], reverse_background, z, h_sw,
        z.flip(0)[:, None], h_sw.flip(0)[:, None]).squeeze(1)
    endpoint_reverse_query_phi = endpoint._density_basis(
        reverse, reverse[:, None], reverse_background, z.flip(0), h_sw.flip(0),
        z.flip(0)[:, None], h_sw.flip(0)[:, None]).squeeze(1)
    endpoint_reverse_target_phi = endpoint._density_basis(
        reverse, coords[:, None], background, z.flip(0), h_sw.flip(0),
        z[:, None], h_sw[:, None]).squeeze(1)
    endpoint_forward = covariance(
        endpoint_query_phi, endpoint_target_phi, anomalies_a)
    endpoint_backward = covariance(
        endpoint_reverse_query_phi, endpoint_reverse_target_phi,
        anomalies_b).flip(0)
    assert (endpoint_forward - endpoint_backward).abs().max() < 1e-7

    sw = torch.zeros(2, 4, 2)
    peak = torch.tensor([[300.0, 11.5], [300.0, 11.5]])
    assert torch.equal(
        legacy(coords, sw, iri_peak=peak)[0],
        explicit_legacy(coords, sw, iri_peak=peak)[0])
    m00, _, _, delta00, extras = symmetric(coords, sw, iri_peak=peak)
    assert torch.equal(m00, extras['ne_bkg'])
    assert torch.equal(delta00, torch.zeros_like(delta00))
    observation = _model_observation(coords, extras['ne_bkg'], 0)
    zero = {key: value.clone() for key, value in observation.items()}
    zero['value'] = zero['background'].clone()
    assert torch.equal(
        symmetric(coords, sw, iri_peak=peak, observations_fy=zero)[0], m00)
    valid = symmetric(
        coords, sw, iri_peak=peak, observations_fy=observation)
    repeated = symmetric(
        coords, sw, iri_peak=peak, observations_fy=observation)
    assert torch.equal(valid[0], repeated[0])
    assert torch.allclose(
        valid[4]['update_FY'] + valid[4]['update_COSMIC'],
        valid[4]['ne_residual'], atol=1e-7)
    symmetric.zero_grad(set_to_none=True)
    valid[0].sum().backward()
    basis_grad = symmetric.density_basis_decoder[-1].weight.grad
    assert torch.isfinite(basis_grad).all() and basis_grad.abs().sum() > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in symmetric.kalman_layer.covariance_scale_net.parameters())

    endpoint_background = endpoint.encode_background(coords, sw, iri_peak=peak)
    endpoint_observation = _model_observation(
        coords, endpoint_background['ne_bkg'], 0)
    endpoint_observation['basis_z_background'] = (
        endpoint_background['z_background'][:, None])
    endpoint_observation['basis_h_sw'] = endpoint_background['h_sw'][:, None]
    endpoint_result = endpoint(
        coords, sw, iri_peak=peak, observations_fy=endpoint_observation)
    endpoint_repeat = endpoint(
        coords, sw, iri_peak=peak, observations_fy=endpoint_observation)
    assert torch.equal(endpoint_result[0], endpoint_repeat[0])
    actual_endpoint_query_phi = endpoint._density_basis(
        coords, coords[:, None], endpoint_background['ne_bkg'],
        endpoint_background['z_background'], endpoint_background['h_sw'],
        endpoint_background['z_background'][:, None],
        endpoint_background['h_sw'][:, None]).squeeze(1)
    assert torch.allclose(
        endpoint_result[4]['ne_residual'],
        torch.einsum(
            'bd,bd->b',
            actual_endpoint_query_phi,
            endpoint_result[4]['latent_increment']).unsqueeze(-1),
        atol=1e-7)
    endpoint_zero = {
        key: value.clone() for key, value in endpoint_observation.items()}
    endpoint_zero['value'] = endpoint_zero['background'].clone()
    assert torch.equal(
        endpoint(coords, sw, iri_peak=peak,
                 observations_fy=endpoint_zero)[0],
        endpoint_background['ne_bkg'])
    endpoint.zero_grad(set_to_none=True)
    endpoint_result[0].sum().backward()
    endpoint_basis_grad = endpoint.density_basis_decoder[0].weight.grad
    assert (torch.isfinite(endpoint_basis_grad).all()
            and endpoint_basis_grad.abs().sum() > 0)

    try:
        FSIA_INR_Model(
            IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), {
                **base_config,
                'enkf_anomaly_parameterization': 'legacy_independent',
                'density_basis_semantics': 'coordinate_local_symmetric'})
    except ValueError:
        pass
    else:
        raise AssertionError('symmetric basis must reject query-conditioned anomalies')


def test_observation_background_deduplication():
    class _SW:
        @staticmethod
        def get_drivers_sequence(relative_hour):
            return relative_hour[:, None, None].expand(-1, 2, 2)

    class _Background:
        density_basis_semantics = 'endpoint_context_symmetric'
        sw_out_dim = 2

        class kalman_layer:
            d_model = 3

        @staticmethod
        def encode_background(coords, sw_seq, iri_peak=None):
            base = coords[:, :3].sum(dim=1, keepdim=True)
            return {
                'ne_bkg': base + sw_seq[:, 0, 0:1],
                'z_background': base.expand(-1, 3),
                'h_sw': sw_seq[:, 0, :],
            }

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
    assert torch.equal(
        attached['basis_z_background'][0, 0],
        attached['basis_z_background'][0, 1])
    assert torch.count_nonzero(attached['basis_z_background'][0, 2]) == 0
    assert torch.count_nonzero(attached['basis_h_sw'][0, 2]) == 0


def _covariance_extras(cross, innovation, precision=None):
    if precision is None:
        precision = torch.ones_like(innovation)
    zeros = torch.zeros(cross.shape[0], 1)
    base_coords = torch.tensor([
        [-12.0, -76.0, 180.0, 48.0],
        [-12.0, -76.0, 350.0, 60.0],
    ])
    repeats = (cross.shape[0] + 1) // 2
    return {
        'r_fy': torch.tensor(0.04),
        'r_cosmic': torch.tensor(0.04),
        'ne_bkg': zeros,
        'query_coords': base_coords.repeat_interleave(3, 0)
        if cross.shape[0] == 6 else base_coords.repeat(repeats, 1)[
            :cross.shape[0]],
        'innov_FY': innovation,
        'innov_COSMIC': torch.zeros_like(innovation),
        'cross_covariance_FY': cross,
        'cross_covariance_COSMIC': torch.zeros_like(cross),
        'precision_FY': precision,
        'precision_COSMIC': torch.zeros_like(precision),
        'observation_coords_FY': base_coords.repeat(repeats, 1)[
            :cross.shape[0], None].expand(-1, cross.shape[1], -1),
        'observation_coords_COSMIC': base_coords.repeat(repeats, 1)[
            :cross.shape[0], None].expand(-1, cross.shape[1], -1),
        'observation_rho_squared_FY': torch.zeros_like(cross),
        'observation_rho_squared_COSMIC': torch.zeros_like(cross),
    }


def test_covariance_moment_direction_mask_and_profile_balance():
    cross = torch.tensor(
        [[0.01, -0.01], [0.02, 0.02]], requires_grad=True)
    innovation = torch.tensor([[0.2, -0.2], [0.2, 0.2]])
    target = torch.tensor([[0.2], [-0.2]], requires_grad=True)
    ids = torch.tensor([0, 1])
    strata = {'FY': [1.0] * 4, 'COSMIC': [1.0] * 4}
    loss = covariance_moment_loss(
        _covariance_extras(cross, innovation), target, ids, 'FY', strata)
    loss.backward()
    assert cross.grad[0, 0] < 0
    assert cross.grad[0, 1] > 0
    assert target.grad is None or target.grad.abs().sum() == 0

    padded_cross = torch.cat([
        cross.detach(), torch.full((2, 1), 99.0)], dim=1)
    padded_innovation = torch.cat([
        innovation, torch.full((2, 1), 99.0)], dim=1)
    padded_precision = torch.cat([
        torch.ones_like(innovation), torch.zeros(2, 1)], dim=1)
    padded = covariance_moment_loss(
        _covariance_extras(
            padded_cross, padded_innovation, padded_precision),
        target.detach(), ids, 'FY', strata)
    assert torch.allclose(loss.detach(), padded)

    duplicated = covariance_moment_loss(
        _covariance_extras(
            cross.detach().repeat_interleave(3, 0),
            innovation.repeat_interleave(3, 0)),
        target.detach().repeat_interleave(3, 0),
        ids.repeat_interleave(3), 'FY', strata)
    assert torch.allclose(loss.detach(), duplicated)


def test_empirical_covariance_loss_uses_frozen_stable_cells():
    cross = torch.zeros(2, 1, requires_grad=True)
    extras = _covariance_extras(
        cross, torch.zeros_like(cross), torch.ones_like(cross))
    stable = torch.zeros(4, 3, 3, 3, 4, dtype=torch.bool)
    covariance = torch.zeros(4, 3, 3, 3, 4)
    stable[0, 0, 0, 0, 0] = True
    covariance[0, 0, 0, 0, 0] = 0.02
    stable[0, 2, 2, 1, 0] = True
    covariance[0, 2, 2, 1, 0] = -0.02
    targets = {'stable': stable, 'covariance': covariance}
    ids = torch.tensor([0, 1])
    loss = empirical_covariance_loss(extras, ids, 'FY', targets)
    loss.backward()
    assert cross.grad[0, 0] < 0
    assert cross.grad[1, 0] > 0

    padded = dict(extras)
    for key in (
            'cross_covariance_FY', 'precision_FY',
            'observation_rho_squared_FY'):
        padded[key] = torch.cat([
            extras[key].detach(), torch.full((2, 1), 99.0)], dim=1)
    padded['precision_FY'][:, 1] = 0.0
    padded['observation_coords_FY'] = torch.cat([
        extras['observation_coords_FY'],
        torch.full((2, 1, 4), 99.0)], dim=1)
    padded_loss = empirical_covariance_loss(padded, ids, 'FY', targets)
    assert torch.allclose(loss.detach(), padded_loss)

    duplicated = dict(extras)
    for key in (
            'cross_covariance_FY', 'precision_FY',
            'observation_rho_squared_FY'):
        duplicated[key] = extras[key].detach().repeat(1, 3)
    duplicated['observation_coords_FY'] = (
        extras['observation_coords_FY'].repeat(1, 3, 1))
    duplicated_loss = empirical_covariance_loss(
        duplicated, ids, 'FY', targets)
    assert torch.allclose(loss.detach(), duplicated_loss)


if __name__ == '__main__':
    test_analytic_kalman_and_sign()
    test_joint_source_and_square_root_invariants()
    test_exact_mode_solver_and_profile_losses()
    test_profile_loss_and_curvature()
    test_model_m00_padding_continuity_and_gradients()
    test_factorized_anomaly_geometry_and_gradients()
    test_observation_background_deduplication()
    test_covariance_moment_direction_mask_and_profile_balance()
    test_empirical_covariance_loss_uses_frozen_stable_cells()
    print('run66 v7/v8 ETKF regression checks passed')
