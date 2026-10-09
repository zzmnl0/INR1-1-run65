"""Focused A1-S4 causal-history contracts."""

import numpy as np
import pytest
import torch

from inr_modules.data_managers.FY_dataloader import _token_observation_payload
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.train_fsia import (
    _calibrate_fixed_r,
    _causal_history_selection_audit,
    _make_optimizer,
    _set_training_stage,
)
from isr_evaluation import main_isr_eval, model_query


def test_causal_token_windows_are_half_open_and_disjoint():
    token_times = np.array([10.5, 10.0, 9.75, 9.5, 8.5], dtype=np.float32)
    token_coords = np.column_stack((
        np.zeros(5), np.zeros(5), np.full(5, 300.0), token_times,
    )).astype(np.float32)
    args = (
        token_coords, np.ones(5, dtype=np.float32),
        np.arange(5, dtype=np.int64), np.arange(5, dtype=np.int64),
        np.array([[0.0, 0.0, 300.0, 10.0]], dtype=np.float32), 0,
    )
    past = _token_observation_payload(
        *args, time_support='causal_past_lag_0p5_1p5_v1')
    current = _token_observation_payload(
        *args, time_support='causal_current_lag_0_0p5_v1')
    assert past['token_id'].tolist() == [3]       # lag=0.5 included; 1.5 excluded
    assert current['token_id'].tolist() == [1, 2]  # lag=0 included; 0.5 excluded
    assert not np.intersect1d(past['token_id'], current['token_id']).size


def test_isr_candidate_uses_checkpoint_causal_time_support(
        monkeypatch, tmp_path):
    expected = 'causal_past_lag_0p5_1p5_v1'
    fy_index, cosmic_index = object(), object()
    observed = {}

    def capture_payload(index, coords, device, **kwargs):
        source = 'FY' if index is fy_index else 'COSMIC'
        observed[source] = kwargs.get('time_support')
        return {}

    monkeypatch.setattr(
        model_query, 'query_observation_payload', capture_payload)
    monkeypatch.setattr(
        model_query, 'observation_query_coverage',
        lambda payload: torch.ones(1, dtype=torch.bool))
    monkeypatch.setattr(
        model_query, 'attach_observation_background',
        lambda payload, *args: payload)

    class StopAfterPayloads(RuntimeError):
        pass

    class CausalModel:
        alt_min = 120.0
        alt_max = 500.0
        analysis_output_semantics = 'causal_context_background_v1'
        analysis_observation_time_support = expected

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            raise StopAfterPayloads

    class SpaceWeather:
        @staticmethod
        def get_drivers_sequence(relative_hours):
            return torch.zeros((len(relative_hours), 1, 2))

    record = {
        'date_str': '20240901',
        'alt_1d': np.array([300.0], dtype=np.float32),
        'ts_1d': np.array([0.0], dtype=np.float64),
        'ne_2d': np.ones((1, 1), dtype=np.float32),
        'lat': 0.0,
        'lon': 0.0,
    }
    with pytest.raises(StopAfterPayloads):
        main_isr_eval._process_station(
            'test', [record], CausalModel(), SpaceWeather(), 0.0,
            {'save_dir': str(tmp_path), 'batch_size': 1},
            torch.device('cpu'), fy_nb_index=fy_index,
            cosmic_nb_index=cosmic_index)

    assert observed == {'FY': expected, 'COSMIC': expected}


def _causal_model():
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
    return FSIA_INR_Model(proxy, {
        'alt_range': (120.0, 500.0),
        'seq_len': 36,
        'basis_dim': 16,
        'sw_hidden_dim': 8,
        'sw_lstm_layers': 1,
        'sw_out_dim': 16,
        'enkf_n_members': 8,
        'enkf_pert_hidden': 8,
        'use_sw_freq': False,
        'analysis_output_semantics': 'causal_context_background_v1',
        'analysis_observation_time_support': 'causal_past_lag_0p5_1p5_v1',
    })


def test_no_history_falls_back_and_only_analysis_is_trainable():
    model = _causal_model().eval()
    coords = torch.tensor([[-12.0, -76.8, 300.0, 48.0]])
    prediction, _, _, increment, extras = model(
        coords, torch.zeros(1, 36, 2))
    torch.testing.assert_close(prediction, extras['ne_bkg'], rtol=0, atol=0)
    torch.testing.assert_close(increment, torch.zeros_like(increment), rtol=0, atol=0)
    torch.testing.assert_close(
        extras['ne_causal_background'], extras['ne_bkg'], rtol=0, atol=0)

    _set_training_stage(model, 'analysis')
    trainable = {name for name, value in model.named_parameters()
                 if value.requires_grad}
    assert trainable
    assert all(name.startswith(('kalman_layer.', 'density_basis_decoder.'))
               for name in trainable)
    optimizer, _ = _make_optimizer(
        model, {'lr': 3e-4, 'weight_decay': 1e-4}, 10)
    assert {id(value) for group in optimizer.param_groups for value in group['params']} == {
        id(value) for value in model.parameters() if value.requires_grad}


def test_causal_r_is_fixed_without_loader_recalibration(tmp_path):
    model = _causal_model()
    _calibrate_fixed_r(
        model, None, None, torch.device('cpu'), None, None, None, {
            'save_dir': str(tmp_path),
            'r_source': 'a1s2_frozen_global_v1',
            'r_mode': 'global',
            'r_fy_fixed': 0.0339414440,
            'r_cosmic_fixed': 0.0218319837,
        })
    assert model.kalman_layer.r_fy.item() == np.float32(0.0339414440)
    assert model.kalman_layer.r_cosmic.item() == np.float32(0.0218319837)


def test_causal_development_selector_accepts_joint_improvement():
    magnetic = {}
    for altitude in ('200-300', '300-500'):
        for period in ('night', 'day'):
            for band in (
                    'abs_sinI_lt_0.25', 'abs_sinI_0.25_0.5',
                    'abs_sinI_ge_0.5'):
                magnetic[f'{altitude}_{period}_{band}'] = {
                    'M00': {'bias': 0.0, 'rmse': 0.18, 'ccc': 0.72},
                    'raw_iri': {'bias': 0.0, 'rmse': 0.20, 'ccc': 0.70},
                }
    metrics = {'score': 0.18, 'ccc': 0.72}
    for source in ('fy', 'cosmic'):
        metrics.update({
            f'{source}_profile_rmse': 0.18,
            f'{source}_a1s2_m00_profile_rmse': 0.20,
            f'{source}_ccc': 0.72,
            f'{source}_a1s2_m00_ccc': 0.70,
            f'{source}_causal_stratified': {
                'magnetic_latitude_cells': magnetic},
        })
    audit = _causal_history_selection_audit(metrics)
    assert audit['eligible']
    assert audit['selection_key'] is not None
