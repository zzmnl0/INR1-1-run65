import numpy as np
import torch
import json
import hashlib

from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.sliding_dataset import (
    build_m2u_anchor_directories,
    validate_m2u_statistics_report,
)
from inr_modules.mdia.train_fsia import observation_gram_whitening_loss
from inr_modules.mdia.m2u_anchor_etkf import (
    accumulate_anchor_terms,
    anchor_mixing_weights,
    blend_anchor_increments,
    flat_top_cover,
    normalized_support_distance_squared,
    pairwise_representativeness_weight,
    solve_anchor_weights,
    sparsemax,
)


def test_sparsemax_has_one_hot_core_and_continuous_simplex():
    scores = torch.tensor([[3.0, 0.0, -2.0], [0.2, 0.1, -1.0]])
    weights = sparsemax(scores)
    assert torch.equal(weights[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.all(weights >= 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_flat_cover_fades_single_anchor_to_background():
    anchor = torch.tensor([[0.0, 0.0, 200.0, 0.0]])
    query = torch.tensor([
        [0.0, 0.0, 200.0, 0.0],
        [0.0, 0.0, 200.0, 0.75],
        [0.0, 0.0, 200.0, 1.5],
    ])
    cover = flat_top_cover(query, anchor).squeeze(-1)
    assert torch.equal(cover[:2], torch.ones(2))
    assert torch.equal(cover[2:], torch.zeros(1))
    mixed = anchor_mixing_weights(
        query, anchor, None, None, ell_e=1.0, ell_a=1.0, tau=1.0)
    assert torch.equal(mixed['beta'][0], torch.ones(1))
    assert torch.equal(mixed['beta'][1], torch.ones(1))
    assert torch.equal(mixed['beta'][2], torch.zeros(1))
    assert torch.equal(mixed['background_weight'], torch.tensor([0.0, 0.0, 1.0]))


def test_anchor_terms_and_shared_increment():
    anomalies = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    precision = torch.ones(1, 2)
    innovation = torch.tensor([[2.0, -1.0]])
    covariance, rhs = accumulate_anchor_terms(
        anomalies, precision, innovation)
    weights = solve_anchor_weights(covariance, rhs, base_rank=1)
    query_anomalies = torch.tensor([[0.4, -0.2], [0.4, -0.2]])
    beta = torch.ones(2, 1)
    increment = blend_anchor_increments(query_anomalies, beta, weights)
    assert torch.equal(increment[0], increment[1])
    assert torch.isfinite(increment).all()


def test_pairwise_representativeness_uses_anchor_source_and_spherical_rho():
    target = torch.tensor([
        [0.0, 0.0, 200.0, 0.0],
        [0.0, 0.0, 200.0, 0.0],
    ])
    observation = torch.tensor([[0.0, 0.0, 200.0, 0.0]])
    rho_squared = normalized_support_distance_squared(target, observation)
    stable = torch.zeros(4, 3, 3, 3, 4)
    stable[0] = 1.0
    weight = pairwise_representativeness_weight(
        target, observation, rho_squared,
        torch.tensor([True, False]), 'FY', stable, floor=0.25)
    assert torch.equal(weight[:, 0], torch.tensor([1.0, 0.25]))


def test_normalized_support_distance_reaches_time_boundary():
    target = torch.tensor([[0.0, 0.0, 200.0, 0.0]])
    observation = torch.tensor([[0.0, 0.0, 200.0, 1.5]])
    assert torch.equal(
        normalized_support_distance_squared(target, observation),
        torch.ones(1, 1))


def test_m2u_model_uses_pairwise_basis_and_frozen_representativeness():
    config = {
        'alt_range': (120.0, 500.0), 'seq_len': 36, 'basis_dim': 64,
        'sw_hidden_dim': 16, 'sw_lstm_layers': 1, 'sw_out_dim': 16,
        'tau_kp_init': 8.0, 'tau_solar_init': 72.0,
        'enkf_n_members': 8, 'enkf_pert_hidden': 32,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'analysis_state_semantics': 'shared_anchor_response',
        'context_semantics': 'shared_error_state',
        'representativeness_floor': 0.25,
        'use_sw_freq': False,
    }
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
    model = FSIA_INR_Model(proxy, config)
    model.set_m2u_representativeness_kernel(
        torch.zeros(4, 3, 3, 3, 4),
        torch.zeros(4, 3, 3, 3, 4))
    query = torch.tensor([
        [0.0, 0.0, 250.0, 24.0],
        [0.0, 0.0, 250.0, 24.0],
    ])
    anchors = torch.tensor([
        [0.0, 0.0, 250.0, 24.0],
        [2.0, 3.0, 320.0, 24.5],
    ])
    query_sw = torch.zeros(2, 36, 2)
    anchor_sw = torch.zeros(2, 36, 2)
    query_peak = torch.tensor([[300.0, 11.5], [300.0, 11.5]])
    anchor_peak = torch.tensor([[300.0, 11.5], [300.0, 11.5]])
    with torch.no_grad():
        endpoint = model.encode_background(
            anchors, anchor_sw, iri_peak=anchor_peak)
    base = {
        'coords': anchors.unsqueeze(0),
        'value': (endpoint['ne_bkg'].squeeze(-1) + 0.1).unsqueeze(0),
        'background': endpoint['ne_bkg'].squeeze(-1).unsqueeze(0),
        'valid_mask': torch.ones(1, 2, dtype=torch.bool),
        'rho_squared': torch.zeros(1, 2),
        'profile_id': torch.tensor([[1, 2]]),
        'source': torch.zeros(1, 2, dtype=torch.int8),
        'basis_z_background': endpoint['z_background'].unsqueeze(0),
        'basis_h_sw': endpoint['h_sw'].unsqueeze(0),
    }
    catalog = dict(base)
    catalog['self_observation_pool'] = base
    catalog['joint_observation_pool'] = base
    prediction, _, _, delta, extras = model(
        query, query_sw, iri_peak=query_peak,
        anchor_observations_fy=catalog)
    assert torch.isfinite(prediction).all() and torch.isfinite(delta).all()
    assert torch.equal(delta[0], delta[1])
    assert extras['m2u_hx_effective_rank_M11'].shape == (2,)
    assert torch.isfinite(extras['m2u_hx_effective_rank_M11']).all()
    assert torch.isfinite(extras['m2u_gram_loss_sum_M11']).all()
    gram_loss, _, coverage = observation_gram_whitening_loss(
        extras, torch.tensor([1, 2]), model.kalman_layer, return_details=True)
    assert torch.isfinite(gram_loss)
    assert set(coverage) == {'M10', 'M01', 'M11'}
    eta = model._m2u_state_eta(
        endpoint['z_background'], endpoint['h_sw'], anchors,
        {'iri_peak': query_peak})
    assert eta.shape == (2, 16) and torch.isfinite(eta).all()
    active_rep = extras['representativeness_FY'][
        extras['precision_FY'] > 0]
    assert active_rep.numel() and torch.allclose(
        active_rep, torch.full_like(active_rep, 0.25))
    cosmic_base = dict(base)
    cosmic_base['value'] = base['background'] - 0.1
    cosmic_base['source'] = torch.ones(1, 2, dtype=torch.int8)
    cosmic_catalog = dict(cosmic_base)
    cosmic_catalog['self_observation_pool'] = cosmic_base
    cosmic_catalog['joint_observation_pool'] = cosmic_base
    _, _, _, _, both = model(
        query, query_sw, iri_peak=query_peak,
        anchor_observations_fy=catalog,
        anchor_observations_cosmic=cosmic_catalog)
    assert torch.allclose(
        extras['mode_increments']['M10'],
        both['mode_increments']['M10'], atol=1e-6, rtol=0.0)
    _, _, _, _, cosmic_only = model(
        query, query_sw, iri_peak=query_peak,
        anchor_observations_cosmic=cosmic_catalog)
    assert torch.allclose(
        cosmic_only['mode_increments']['M01'],
        both['mode_increments']['M01'], atol=1e-6, rtol=0.0)
    clone = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), config)
    clone.load_state_dict(model.state_dict(), strict=True)
    clone_prediction = clone(
        query, query_sw, iri_peak=query_peak,
        anchor_observations_fy=catalog,
        anchor_observations_cosmic=cosmic_catalog)[0]
    assert torch.allclose(
        clone_prediction, both['ne_bkg'] + both['ne_residual'],
        atol=1e-7, rtol=0.0)


def test_m2u_statistics_validator_requires_the_frozen_schema(tmp_path):
    cell_count = int(np.prod((4, 3, 3, 3, 4)))
    npz_path = tmp_path / 'empirical_covariance_cells.npz'
    np.savez(
        npz_path,
        cell_id=np.arange(cell_count, dtype=np.int16),
        stable=np.ones(cell_count, dtype=bool),
        covariance=np.ones(cell_count, dtype=np.float32),
        correlation=np.ones(cell_count, dtype=np.float32),
    )
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    report = {
        'mode': 'full',
        'split_mode': 'date_blocked_train',
        'npz_sha256': digest,
        'window': {
            'localization_semantics': 'm2u',
            'hours': 1.5,
            'space_km': 1800.0,
            'top_profiles': None,
            'points_per_profile': 8,
        },
        'cross_source_gate': {'passed': True},
        'input_identity': {},
    }
    (tmp_path / 'empirical_covariance_report.json').write_text(
        json.dumps(report), encoding='utf-8')
    result = validate_m2u_statistics_report(npz_path, expected={})
    assert result['shadow'] is False


def test_anchor_directories_expand_observations_around_union_anchors():
    class Index:
        def __init__(self, longitude_offset, source):
            self.longitude_offset = longitude_offset
            self.source = source

        def query_observation_directory(self, coords, **_):
            coords = np.asarray(coords, dtype=np.float32).copy()
            coords[:, 1] += self.longitude_offset
            count = len(coords)
            return {
                'coords': coords[:, None, :4],
                'value': np.ones((count, 1), dtype=np.float32),
                'valid_mask': np.ones((count, 1), dtype=bool),
                'profile_id': np.arange(count, dtype=np.int64)[:, None]
                + self.source * 100,
                'source': np.full((count, 1), self.source, dtype=np.int8),
                'rho_squared': np.zeros((count, 1), dtype=np.float32),
            }

    class Model:
        density_basis_semantics = 'none'

        @staticmethod
        def encode_background(coords, *_args, **_kwargs):
            return {'ne_bkg': torch.zeros(len(coords), 1)}

    class Weather:
        @staticmethod
        def get_drivers_sequence(time):
            return torch.zeros(len(time), 1, 2)

    fy, cosmic = build_m2u_anchor_directories(
        {'FY': Index(0.0, 0), 'COSMIC': Index(10.0, 1)},
        torch.tensor([[0.0, 0.0, 250.0, 0.0]]), torch.device('cpu'),
        Model(), Weather())
    assert fy['self_observation_pool']['coords'].shape[1] == 1
    assert fy['joint_observation_pool']['coords'].shape[1] == 2
    assert cosmic['self_observation_pool']['coords'].shape[1] == 1
    assert cosmic['joint_observation_pool']['coords'].shape[1] == 2


if __name__ == '__main__':
    test_sparsemax_has_one_hot_core_and_continuous_simplex()
    test_flat_cover_fades_single_anchor_to_background()
    test_anchor_terms_and_shared_increment()
    test_pairwise_representativeness_uses_anchor_source_and_spherical_rho()
    test_normalized_support_distance_reaches_time_boundary()
    test_m2u_model_uses_pairwise_basis_and_frozen_representativeness()
    test_anchor_directories_expand_observations_around_union_anchors()
    print('M2-U anchor ETKF tests passed')
