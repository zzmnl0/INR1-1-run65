import numpy as np
import torch
import pytest

from inr_modules.data_managers.FY_dataloader import _token_observation_payload
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model, NeuralETKFLayer
from inr_modules.mdia.sliding_dataset import attach_observation_background
from isr_evaluation.main_isr_eval import _require_m2v_config


def test_exact_token_support_and_batch_invariance():
    token_coords = np.array([
        [0.0, 10.0, 250.0, 1.0],  # token is supported; profile center is not consulted
        [0.0, 0.0, 260.0, 1.0],
        [0.0, 0.0, 270.0, 3.0],
    ], dtype=np.float32)
    values = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    profile_ids = np.array([10, 11, 12], dtype=np.int64)
    token_ids = np.array([0, 0, 0], dtype=np.int64)
    queries = np.array([[0.0, 0.0, 250.0, 1.0], [0.0, 0.0, 250.0, 1.2]], dtype=np.float32)
    whole = _token_observation_payload(
        token_coords, values, profile_ids, token_ids, queries, 0)
    first = _token_observation_payload(
        token_coords, values, profile_ids, token_ids, queries[:1], 0)
    assert whole['profile_id'][0] == first['profile_id'][0]
    assert whole['localization_weight'][0] > 0.0
    assert 11 in whole['profile_id']
    assert 12 not in whole['profile_id']  # time support is excluded
    assert whole['row_ptr'].tolist() == [0, 2, 4]
    assert whole['query_index'].tolist() == [0, 0, 1, 1]


def test_d64_n8_rank_and_physical_precision():
    layer = NeuralETKFLayer(
        d_model=64, n_members=8, anomaly_parameterization='orthogonal_factor')
    b, m = 2, 3
    z = torch.zeros(b, 64)
    h = torch.zeros(b, 64)
    phi_q = torch.randn(b, 64)
    coords = torch.randn(5, 64)
    payload = {
        'value': torch.ones(5), 'background': torch.zeros(5),
        'valid_mask': torch.ones(5, dtype=torch.bool),
        'rho_squared': torch.zeros(5),
        'localization_weight': torch.full((5,), 0.5),
        'query_index': torch.tensor([0, 0, 1, 1, 0]),
        'row_ptr': torch.tensor([0, 3, 5]),
    }
    out = layer(
        z, h, phi_q,
        {'FY': (payload, coords), 'COSMIC': (payload, coords)},
        torch.zeros(b), torch.zeros(b), torch.zeros(b), torch.ones(b), torch.zeros(b))
    assert out['latent_anomalies'].shape == (b, 8, 64)
    assert out['query_anomalies'].shape == (b, 8)
    assert torch.linalg.matrix_rank(out['latent_anomalies'][0]) <= 7
    assert torch.allclose(out['precision_FY'], torch.full((5,), 0.5 / 0.04))
    assert out['factor_gram_FY'].shape == (b, 7, 7)
    assert out['hx_numeric_rank_M11'].max() <= 7


def test_chunked_and_single_pass_statistics_match():
    torch.manual_seed(7)
    kwargs = dict(d_model=64, n_members=8,
                  anomaly_parameterization='orthogonal_factor')
    layer_one = NeuralETKFLayer(**kwargs, observation_chunk_size=1)
    layer_all = NeuralETKFLayer(**kwargs, observation_chunk_size=4096)
    layer_all.load_state_dict(layer_one.state_dict())
    b, t = 3, 11
    z = torch.zeros(b, 64)
    h = torch.zeros(b, 64)
    phi_q = torch.randn(b, 64)
    phi = torch.randn(t, 64)
    payload = {
        'value': torch.randn(t), 'background': torch.randn(t),
        'valid_mask': torch.ones(t, dtype=torch.bool),
        'rho_squared': torch.zeros(t),
        'localization_weight': torch.rand(t),
        'query_index': torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2]),
        'row_ptr': torch.tensor([0, 4, 7, 11]),
    }
    args = (z, h, phi_q, {'FY': (payload, phi), 'COSMIC': (payload, phi)},
            torch.zeros(b), torch.zeros(b), torch.zeros(b), torch.ones(b),
            torch.zeros(b))
    one = layer_one(*args)
    all_ = layer_all(*args)
    for key in ('ensemble_covariance_FY', 'ensemble_rhs_FY',
                'factor_gram_FY', 'delta_FY'):
        assert torch.allclose(one[key], all_[key], atol=2e-5, rtol=2e-5)


def test_m2v_model_accepts_flat_token_payload():
    config = {
        'alt_range': (120.0, 500.0), 'seq_len': 36, 'basis_dim': 64,
        'sw_hidden_dim': 8, 'sw_lstm_layers': 1, 'sw_out_dim': 16,
        'enkf_n_members': 8, 'enkf_pert_hidden': 16,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'assimilation_semantics': 'continuous_physical_local_letkf',
        'observation_chunk_size': 1, 'use_physical_localization': True,
        'physical_localization_space_km': 1800.0,
        'physical_localization_time_hours': 1.5,
    }
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
    model = FSIA_INR_Model(proxy, config).eval()
    coords = torch.tensor([[0.0, 0.0, 250.0, 1.0],
                           [0.0, 0.0, 270.0, 1.2]])
    sw = torch.zeros(2, 36, 2)
    payload = {
        'coords': torch.tensor([[0.0, 0.0, 250.0, 1.0],
                                [0.0, 0.0, 270.0, 1.2]]),
        'value': torch.tensor([10.1, 10.2]),
        'valid_mask': torch.ones(2, dtype=torch.bool),
        'rho_squared': torch.zeros(2),
        'localization_weight': torch.ones(2),
        'query_index': torch.tensor([0, 1]),
        'row_ptr': torch.tensor([0, 1, 2]),
    }
    class _SW:
        @staticmethod
        def get_drivers_sequence(times):
            return torch.zeros(len(times), 36, 2)

    payload = attach_observation_background(payload, model, _SW())
    fused, _, _, _, extras = model(
        coords, sw, observations_fy=payload,
        observations_cosmic=None)
    assert fused.shape == (2, 1)
    assert extras['precision_FY'].shape == (2,)
    assert extras['factor_gram_FY'].shape == (2, 7, 7)
    assert torch.isfinite(fused).all()


def test_isr_config_rejects_m2o_semantics():
    config = {
        'checkpoint_format_version': 12, 'basis_dim': 64,
        'enkf_n_members': 8,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'r_mode': 'global', 'use_distance_localization': True,
        'assimilation_semantics': 'legacy_local_etkf',
        'neighbor_directory_semantics': 'profile_center_legacy',
        'physical_localization_space_km': 1800.0,
        'physical_localization_time_hours': 1.5,
        'representativeness_kernel_path': None,
    }
    with pytest.raises(ValueError, match='M2-V'):
        _require_m2v_config(config)
