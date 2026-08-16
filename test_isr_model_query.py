"""Regression check that ISR queries pass FY and COSMIC neighborhoods."""

import numpy as np
import torch

from isr_evaluation.model_query import query_model_grid
from isr_evaluation.main_isr_eval import (
    _compute_stratified_bootstrap,
    _compute_stratified_metrics,
    _require_m2v_config,
    _resolve_checkpoint,
)
from inr_modules.mdia.sliding_dataset import observation_query_coverage
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


def test_query_model_grid_does_not_use_isr_missingness_as_coordinate_mask():
    class SpaceWeather:
        def get_drivers_sequence(self, rel_hour):
            return torch.zeros(len(rel_hour), 3, 2)

    class Model:
        alt_min, alt_max = 120.0, 500.0

        def eval(self):
            return self

        def __call__(self, coords, sw_seq, **kwargs):
            output = torch.full((len(coords), 1), 11.0)
            return output, output, output, output, {
                'ne_bkg': output - 0.1,
                'ne_iri': output - 0.2,
            }

    record = {
        'alt_1d': np.array([200.0, 300.0], np.float32),
        'ts_1d': np.array([0.0, 3600.0]),
        'ne_2d': np.array([[1.0, np.nan], [np.nan, 1.0]], np.float32),
        'lat': 10.0,
        'lon': 20.0,
        'coordinate_mask': np.ones((2, 2), dtype=bool),
    }
    prediction, background, iri = query_model_grid(
        Model(), SpaceWeather(), record, 0.0, torch.device('cpu'), batch_size=8)
    np.testing.assert_allclose(prediction, 11.0)
    np.testing.assert_allclose(background, 10.9)
    np.testing.assert_allclose(iri, 10.8)


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
    assert 'analysis_alt_120-300km_all' in metrics
    assert 'analysis_alt_120-500km_day' in metrics
    assert 'analysis_alt_120-500km_night' in metrics
    assert 'analysis_alt_120-500km_all' in metrics
    assert 'background_alt_120-300km_all' in metrics
    assert 'iri_alt_120-300km_all' in metrics


def test_stratified_metrics_and_bootstrap_share_two_height_bins():
    alt = np.array([120.0, 199.999, 200.0, 299.999, 300.0, 500.0])
    metrics = _compute_stratified_metrics(
        alt_all=alt,
        lon_all=np.zeros_like(alt),
        rh_all=np.zeros_like(alt),
        pred_all=np.zeros_like(alt),
        obs_all=np.zeros_like(alt),
        background_all=np.zeros_like(alt),
        iri_all=np.zeros_like(alt),
    )
    assert metrics['analysis_alt_120-300km_all']['n'] == 4
    assert np.isnan(metrics['analysis_alt_120-300km_all']['rmse'])
    assert metrics['analysis_alt_300-500km_all']['n'] == 2
    assert metrics['analysis_alt_120-500km_all']['n'] == 6
    assert metrics['analysis_alt_120-500km_all']['bias'] == 0.0
    assert not any('120-200km' in key or '200-300km' in key
                   for key in metrics)

    bootstrap = _compute_stratified_bootstrap(
        altitude=alt,
        longitude=np.zeros_like(alt),
        rel_hour=np.array([6.0, 7.0, 8.0, 9.0, 0.0, 1.0]),
        observation=np.linspace(10.0, 10.5, len(alt)),
        analysis=np.linspace(10.1, 10.6, len(alt)),
        raw_iri=np.linspace(9.9, 10.4, len(alt)),
        unit_ids=np.arange(len(alt)),
        alt_range=(120.0, 500.0),
    )
    assert set(bootstrap) == {
        'alt_120-300km_day', 'alt_120-300km_all',
        'alt_300-500km_night', 'alt_300-500km_all',
        'alt_120-500km_day', 'alt_120-500km_night',
        'alt_120-500km_all'}


def test_isr_requires_explicit_frozen_m2v_epoch():
    try:
        _resolve_checkpoint({'checkpoint_path': None, 'model_type': 'fsia'}, {})
    except ValueError as error:
        assert '完整Analysis checkpoint' in str(error)
    else:
        raise AssertionError('ambiguous RMSE-best checkpoint was accepted')

    _require_m2v_config({
        'checkpoint_format_version': 12,
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
        'representativeness_kernel_path': None,
        'use_empirical_covariance_loss': False,
    })


def test_observation_query_coverage_supports_ragged_m2v_payload():
    coverage = observation_query_coverage({
        'valid_mask': torch.ones(3, dtype=torch.bool),
        'row_ptr': torch.tensor([0, 2, 2, 3]),
    })
    torch.testing.assert_close(
        coverage, torch.tensor([True, False, True]))


if __name__ == '__main__':
    test_query_model_grid_passes_both_sources()
    test_infer_grid_reports_raw_background_and_four_modes()
    test_stratified_metrics_keep_three_model_stages()
    test_isr_requires_explicit_frozen_m2v_epoch()
    test_observation_query_coverage_supports_ragged_m2v_payload()
