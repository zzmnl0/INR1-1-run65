"""Regression check that ISR queries pass FY and COSMIC neighborhoods."""

import numpy as np
import torch

from isr_evaluation.model_query import query_model_grid
from isr_evaluation.main_isr_eval import (
    _compute_stratified_metrics,
    _require_m2o_config,
    _resolve_checkpoint,
)
from inr_modules.mdia.visualization_mdia import _infer_grid


def test_query_model_grid_passes_both_sources():
    class Index:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def query_observation_batch(self, coords, exclude_profile_ids=None):
            self.calls += 1
            batch = len(coords)
            return {
                'coords': np.repeat(coords[:, None, :4], 2, axis=1),
                'value': np.full((batch, 2), self.value, np.float32),
                'valid_mask': np.ones((batch, 2), bool),
                'profile_id': np.zeros((batch, 2), np.int64),
                'source': np.zeros((batch, 2), np.int8),
                'rho_squared': np.zeros((batch, 2), np.float32),
            }

    class SpaceWeather:
        def get_drivers_sequence(self, rel_hour):
            return torch.zeros(len(rel_hour), 3, 2)

    class Peaks:
        def get_iri_peak(self, coords):
            return torch.tensor([[300.0, 11.5]]).expand(len(coords), -1)

    class Model:
        def eval(self):
            return self

        @staticmethod
        def encode_background(coords, sw_seq, iri_peak=None):
            return {'ne_bkg': torch.zeros(len(coords), 1)}

        def __call__(self, coords, sw_seq, **kwargs):
            assert kwargs['observations_fy']['value'].mean() == 1
            assert kwargs['observations_cosmic']['value'].mean() == 2
            assert kwargs['iri_peak'] is not None
            pred = torch.ones(len(coords), 1)
            return pred, pred, pred, pred, {
                'ne_bkg': torch.zeros_like(pred),
                'ne_iri': torch.full_like(pred, -1.0),
            }

    fy_index, cosmic_index = Index(1), Index(2)
    record = {
        'alt_1d': np.array([200.0, 300.0], np.float32),
        'ts_1d': np.array([0.0, 3600.0]),
        'ne_2d': np.ones((2, 2), np.float32),
        'lat': 10.0,
        'lon': 20.0,
    }
    pred, bkg, iri = query_model_grid(
        Model(), SpaceWeather(), record, 0.0, torch.device('cpu'),
        batch_size=2, iri_peak_manager=Peaks(),
        fy_nb_index=fy_index, cosmic_nb_index=cosmic_index)
    np.testing.assert_array_equal(pred, np.ones((2, 2), np.float32))
    np.testing.assert_array_equal(bkg, np.zeros((2, 2), np.float32))
    np.testing.assert_array_equal(iri, -np.ones((2, 2), np.float32))
    assert fy_index.calls == cosmic_index.calls == 2


def test_infer_grid_reports_raw_background_and_four_modes():
    class SpaceWeather:
        def get_drivers_sequence(self, rel_hour):
            return torch.zeros(len(rel_hour), 3, 2)

    class Index:
        def __init__(self, value):
            self.value = value

        def query_observation_batch(self, coords, exclude_profile_ids=None):
            batch = len(coords)
            return {
                'coords': np.repeat(coords[:, None, :4], 2, axis=1),
                'value': np.full((batch, 2), self.value, np.float32),
                'valid_mask': np.ones((batch, 2), bool),
                'profile_id': np.zeros((batch, 2), np.int64),
                'source': np.zeros((batch, 2), np.int8),
                'rho_squared': np.zeros((batch, 2), np.float32),
            }

    class Model:
        def eval(self):
            return self

        @staticmethod
        def encode_background(coords, sw_seq, iri_peak=None):
            return {'ne_bkg': torch.full((len(coords), 1), 10.0)}

        def __call__(self, coords, sw_seq, **kwargs):
            background = torch.full((len(coords), 1), 10.0, device=coords.device)
            analysis = background.clone()
            if kwargs.get('observations_fy') is not None:
                present = kwargs['observations_fy']['valid_mask'].any(dim=1)
                analysis = analysis + 0.1 * present.reshape(-1, 1)
            if kwargs.get('observations_cosmic') is not None:
                present = kwargs['observations_cosmic']['valid_mask'].any(dim=1)
                analysis = analysis + 0.2 * present.reshape(-1, 1)
            extras = {
                'ne_bkg': background,
                'ne_iri': torch.full_like(background, 9.0),
            }
            return analysis, analysis, analysis, analysis - background, extras

    coords = np.array([
        [0.0, 0.0, 200.0, 0.0],
        [0.0, 0.0, 250.0, 0.0],
        [0.0, 0.0, 300.0, 0.0],
    ], dtype=np.float32)
    result = _infer_grid(
        Model(), coords, torch.zeros(1, 3, 2), torch.device('cpu'),
        SpaceWeather(), vis_batch=2, fy_nb_index=Index(1),
        cosmic_nb_index=Index(2))

    np.testing.assert_allclose(result['ne_iri'], 9.0)
    np.testing.assert_allclose(result['background'], 10.0)
    np.testing.assert_allclose(result['m00'], 10.0)
    np.testing.assert_allclose(result['m10'], 10.1)
    np.testing.assert_allclose(result['m01'], 10.2)
    np.testing.assert_allclose(result['m11'], 10.3)
    np.testing.assert_allclose(result['ne_delta'], 0.3, atol=1e-6)


def test_stratified_metrics_keep_three_model_stages():
    n = 8
    metrics = _compute_stratified_metrics(
        alt_all=np.linspace(200.0, 400.0, n),
        lon_all=np.zeros(n),
        rh_all=np.arange(n, dtype=np.float64),
        pred_all=np.full(n, 10.3),
        obs_all=np.full(n, 10.4),
        background_all=np.full(n, 10.1),
        iri_all=np.full(n, 10.0),
    )
    assert 'analysis_all_alt_all' in metrics
    assert 'background_all_alt_all' in metrics
    assert 'iri_all_alt_all' in metrics


def test_isr_requires_explicit_frozen_m2o_epoch():
    try:
        _resolve_checkpoint({'checkpoint_path': None, 'model_type': 'fsia'}, {})
    except ValueError as error:
        assert 'development' in str(error)
    else:
        raise AssertionError('ambiguous RMSE-best checkpoint was accepted')

    _require_m2o_config({
        'basis_dim': 64,
        'enkf_n_members': 8,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'r_mode': 'global',
        'use_distance_localization': True,
    })


if __name__ == '__main__':
    test_query_model_grid_passes_both_sources()
    test_infer_grid_reports_raw_background_and_four_modes()
    test_stratified_metrics_keep_three_model_stages()
    test_isr_requires_explicit_frozen_m2o_epoch()
