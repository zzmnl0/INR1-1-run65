"""Minimal R0/R1 checks for the M2-R physical coefficient shadow path."""

import torch

from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import (
    FSIA_INR_Model, _compute_local_time_features, solve_density_modes,
)
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background, build_failed_shadow_reference_context,
)


def _model(physical, dictionary='m2r_residual_legendre', basis_dim=8):
    config = {
        'alt_range': (120.0, 500.0), 'seq_len': 4, 'basis_dim': basis_dim,
        'sw_hidden_dim': 4, 'sw_lstm_layers': 1, 'sw_out_dim': 8,
        'tau_kp_init': 8.0, 'tau_solar_init': 72.0,
        'enkf_n_members': 8, 'enkf_pert_hidden': 8, 'use_sw_freq': False,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
    }
    if physical:
        config.update({
            'density_basis_semantics': 'endpoint_context_symmetric',
            'analysis_state_semantics': 'query_local_increment_coefficients',
            'context_semantics': 'endpoint_conditioning_only',
            'mode_basis_semantics': 'reference_whitened_physical_modes',
            'physical_mode_dictionary': dictionary,
            'allow_failed_query_local_shadow': dictionary != 'm2r_residual_legendre',
        })
    return FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), config)


class _SW:
    def get_drivers_sequence(self, time):
        return torch.zeros(len(time), 4, 2, device=time.device)


class _Peak:
    def get_iri_peak(self, coords):
        hmf2 = 300.0 + 0.2 * coords[:, 0] + 0.1 * coords[:, 3]
        return torch.stack([hmf2, torch.full_like(hmf2, 11.5)], dim=-1)


def _observations(model, coords, background):
    token_coords = torch.tensor([
        [-12.2, -76.5, 170.0, 48.2],
        [-11.8, -77.0, 360.0, 47.8],
        [64.8, 147.3, 180.0, 240.2],
        [65.2, 146.7, 390.0, 239.8],
    ])
    per_row = torch.stack([
        token_coords[:2], token_coords[:2].flip(0),
        token_coords[2:], token_coords[2:].flip(0),
    ])
    values = torch.tensor([
        [0.08, -0.05], [-0.05, 0.08],
        [0.06, -0.04], [-0.04, 0.06],
    ])
    return {
        'coords': per_row,
        'value': background.detach() + values,
        'background': background.detach().expand(-1, 2),
        'valid_mask': torch.ones(4, 2, dtype=torch.bool),
        'profile_id': torch.tensor([
            [101, 102], [102, 101], [201, 202], [202, 201]]),
        'source': torch.zeros(4, 2, dtype=torch.int8),
        'rho_squared': torch.full((4, 2), 0.04),
        'representativeness_weight': torch.ones(4, 2),
        'basis_z_background': torch.zeros(
            4, 2, model.background_state_dim),
        'basis_h_sw': torch.zeros(4, 2, model.sw_out_dim),
    }


def test_m2r_reference_modes_and_query_local_state():
    torch.manual_seed(42)
    model = _model(True)
    coords = torch.tensor([
        [-12.0, -76.8, 170.0, 48.0],
        [-12.0, -76.8, 360.0, 48.0],
        [65.0, 147.0, 180.0, 240.0],
        [65.0, 147.0, 390.0, 240.0],
    ])
    sw = torch.zeros(4, 4, 2)
    peak = torch.tensor([[300.0, 11.5]]).expand(4, -1)
    background = model(coords, sw, iri_peak=peak)[4]
    fy = _observations(model, coords, background['ne_bkg'])
    result = model(
        coords, sw, iri_peak=peak, observations_fy=fy)
    repeated = model(
        coords, sw, iri_peak=peak, observations_fy=fy)
    extras = result[4]

    assert extras['mode_reference_gram_error'].max() <= 1e-5
    assert torch.isfinite(extras['raw_mode_gram_condition']).all()
    assert extras['raw_mode_gram_min_eigenvalue'].min() > 1e-6
    assert extras['coefficient_analysis_query'].shape == (4, 7)
    assert not torch.equal(
        extras['coefficient_analysis_query'][0],
        extras['coefficient_analysis_query'][1])
    assert torch.equal(result[0], repeated[0])
    assert torch.allclose(
        extras['update_FY'] + extras['update_COSMIC'],
        extras['ne_residual'], atol=1e-7)
    assert torch.isfinite(extras['joint_observation_anomalies_effective_rank']).all()
    assert torch.linalg.eigvalsh(extras['system']).min() > 0
    modes = solve_density_modes(extras)
    assert torch.allclose(
        modes['M11'].unsqueeze(-1), extras['ne_residual'], atol=1e-7)

    token_reordered = {
        key: (value.flip(1) if value.ndim >= 2 else value)
        for key, value in fy.items()}
    reordered = model(
        coords, sw, iri_peak=peak, observations_fy=token_reordered)
    assert torch.allclose(result[0], reordered[0], atol=1e-7)
    relabeled = {key: value.clone() for key, value in fy.items()}
    relabeled['source'].fill_(1)
    assert torch.equal(
        result[0], model(
            coords, sw, iri_peak=peak,
            observations_fy=relabeled)[0])

    row_order = torch.tensor([3, 2, 1, 0])
    row_reordered = model(
        coords[row_order], sw[row_order], iri_peak=peak[row_order],
        observations_fy={key: value[row_order] for key, value in fy.items()})
    assert torch.allclose(result[0][row_order], row_reordered[0], atol=1e-7)
    single = model(
        coords[:1], sw[:1], iri_peak=peak[:1],
        observations_fy={key: value[:1] for key, value in fy.items()})
    assert torch.equal(
        extras['coefficient_analysis_query'][0],
        single[4]['coefficient_analysis_query'][0])
    assert torch.allclose(result[0][0], single[0][0], atol=1e-6)

    padded = {}
    for key, value in fy.items():
        shape = list(value.shape)
        shape[1] = 1
        fill = False if value.dtype == torch.bool else 0
        padded[key] = torch.cat([
            value, torch.full(shape, fill, dtype=value.dtype)], dim=1)
    padded['coords'][:, -1] = float('nan')
    padded['value'][:, -1] = float('nan')
    padded_result = model(
        coords, sw, iri_peak=peak, observations_fy=padded)
    assert torch.equal(result[0], padded_result[0])

    zero = {key: value.clone() for key, value in fy.items()}
    zero['value'] = zero['background'].clone()
    zero_result = model(
        coords, sw, iri_peak=peak, observations_fy=zero)
    assert torch.equal(zero_result[0], zero_result[4]['ne_bkg'])
    assert torch.equal(zero_result[3], torch.zeros_like(zero_result[3]))

    model.zero_grad(set_to_none=True)
    result[0].sum().backward()
    gradient = model.kalman_layer.covariance_scale_net[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_m2r_does_not_change_legacy_state_dict_or_output():
    torch.manual_seed(7)
    implicit = _model(False)
    explicit = _model(False)
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    coords = torch.tensor([[-12.0, -76.8, 250.0, 48.0]])
    sw = torch.zeros(1, 4, 2)
    peak = torch.tensor([[300.0, 11.5]])
    assert torch.equal(
        implicit(coords, sw, iri_peak=peak)[0],
        explicit(coords, sw, iri_peak=peak)[0])
    rho = torch.linspace(0.0, 1.0, 10001).square()
    assert implicit.kalman_layer._localization_precision(rho).min() >= 0


def test_d64_endpoint_payload_enters_query_local_etkf():
    torch.manual_seed(3)
    coords = torch.tensor([
        [-12.0, -76.8, 180.0, 48.0],
        [65.0, 147.0, 380.0, 240.0],
    ])
    sw_manager, peak_manager = _SW(), _Peak()
    sw = sw_manager.get_drivers_sequence(coords[:, 3])
    peak = peak_manager.get_iri_peak(coords)
    raw = {
        'coords': coords[:, None, :].clone(),
        'value': torch.tensor([[10.6], [11.1]]),
        'valid_mask': torch.ones(2, 1, dtype=torch.bool),
        'rho_squared': torch.zeros(2, 1),
    }
    for dictionary in ('endpoint_hmf2_legendre', 'background_adaptive_fixed'):
        model = _model(True, dictionary, basis_dim=64)
        payload = attach_observation_background(
            raw, model, sw_manager, peak_manager)
        assert payload['basis_z_background'].shape == (2, 1, 64)
        reference = build_failed_shadow_reference_context(
            coords, model, sw_manager, peak_manager)
        result = model(
            coords, sw, iri_peak=peak, observations_fy=payload,
            physical_reference_context=reference)
        assert result[4]['coefficient_analysis_query'].shape == (2, 7)
        assert torch.isfinite(result[0]).all()
        background = model.encode_endpoint_context(coords, sw, peak)
        _, chol, _ = model._physical_mode_transform(
            coords, peak[:, 0], background['z_background'], background['h_sw'],
            background['ne_bkg'].flatten(),
            reference_z=reference['basis_z_background'],
            reference_h_sw=reference['basis_h_sw'],
            reference_background=reference['background'],
            reference_endpoint=reference)
        manual = model._apply_mode_transform(
            model._physical_mode_raw(
                coords, peak[:, 0], payload['coords'],
                background['z_background'], background['h_sw'],
                payload['basis_z_background'], payload['basis_h_sw'],
                payload['background'], payload), chol)
        assert torch.equal(result[4]['basis_FY'], manual)


def test_endpoint_height_absolute_height_and_lst_semantics():
    model = _model(True, 'background_adaptive_fixed')
    center = torch.tensor([[0.0, 0.0, 300.0, 24.0]])
    target = torch.tensor([[[0.0, 0.0, 180.0, 24.0],
                            [0.0, 0.0, 180.0, 24.0]]])
    endpoint = {
        'basis_delta_alt': torch.tensor([[0.0, -0.5]]),
        'basis_background_dh': torch.tensor([[0.01, 0.01]]),
        'basis_cos_sza': torch.zeros(1, 2),
        'basis_sin_lst': torch.zeros(1, 2),
        'basis_cos_lst': torch.ones(1, 2),
    }
    modes = model._physical_mode_raw(
        center, torch.tensor([300.0]), target, target_endpoint=endpoint)
    assert not torch.equal(modes[0, 0, 2], modes[0, 1, 2])

    target_alt = target.clone()
    target_alt[0, 1, 2] = 360.0
    endpoint['basis_delta_alt'] = torch.zeros(1, 2)
    modes = model._physical_mode_raw(
        center, torch.tensor([300.0]), target_alt, target_endpoint=endpoint)
    assert not torch.equal(modes[0, 0, 3], modes[0, 1, 3])

    lon = torch.tensor([0.0, 30.0, 179.999, -180.001])
    time = torch.full((4,), 24.0)
    sin_lst, cos_lst = _compute_local_time_features(lon, time)
    assert not torch.equal(sin_lst[0], sin_lst[1])
    assert torch.allclose(sin_lst[2], sin_lst[3], atol=1e-6)
    assert torch.allclose(cos_lst[2], cos_lst[3], atol=1e-6)


def test_reference_endpoint_context_is_required_not_query_fallback():
    model = _model(True, 'endpoint_hmf2_legendre')
    coords = torch.tensor([[0.0, 0.0, 250.0, 24.0]])
    sw = torch.zeros(1, 4, 2)
    peak = torch.tensor([[300.0, 11.5]])
    try:
        model(coords, sw, iri_peak=peak)
    except ValueError as error:
        assert 'reference endpoint context' in str(error)
    else:
        raise AssertionError('reference query-context fallback was accepted')


def test_failed_query_local_dictionaries_require_explicit_audit_flag():
    config = {
        'alt_range': (120.0, 500.0), 'seq_len': 4, 'basis_dim': 8,
        'sw_hidden_dim': 4, 'sw_lstm_layers': 1, 'sw_out_dim': 8,
        'enkf_n_members': 8, 'enkf_pert_hidden': 8, 'use_sw_freq': False,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'analysis_state_semantics': 'query_local_increment_coefficients',
        'context_semantics': 'endpoint_conditioning_only',
        'mode_basis_semantics': 'reference_whitened_physical_modes',
        'physical_mode_dictionary': 'endpoint_hmf2_legendre',
    }
    try:
        FSIA_INR_Model(
            IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), config)
    except ValueError as error:
        assert 'failed RSR-3' in str(error)
    else:
        raise AssertionError('failed query-local dictionary was enabled')
