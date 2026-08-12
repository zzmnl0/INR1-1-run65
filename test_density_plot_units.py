"""Focused checks for physical electron-density display units."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import evaluate_giro_peak
from inr_modules.density_units import (
    density_to_display,
    log10_density_to_display,
)
from inr_modules.mdia.visualization_mdia import _extract_isr_profile
from isr_evaluation import plots as isr_plots


def test_density_display_conversions_preserve_nan():
    np.testing.assert_allclose(
        log10_density_to_display([10.0, 11.0, 12.0]),
        [0.1, 1.0, 10.0])
    np.testing.assert_allclose(
        density_to_display([1e10, 1e11, 1e12]),
        [0.1, 1.0, 10.0])
    assert np.isnan(log10_density_to_display([np.nan])[0])


def test_isr_nmf2_scatter_uses_physical_values(tmp_path, monkeypatch):
    original_close = plt.close
    monkeypatch.setattr(isr_plots.plt, 'close', lambda *_args, **_kwargs: None)
    observed = np.array([10.0, 11.0, 12.0])
    predicted = np.array([10.1, 11.1, 12.1])
    isr_plots.plot_nmf2_scatter(
        observed, predicted, 'Test', tmp_path / 'nmf2.png')

    figure = plt.gcf()
    axis = figure.axes[0]
    offsets = np.asarray(axis.collections[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], log10_density_to_display(observed))
    np.testing.assert_allclose(offsets[:, 1], log10_density_to_display(predicted))
    assert '10^{11}' in axis.get_xlabel() and 'log' not in axis.get_xlabel().lower()
    original_close(figure)


def test_isr_time_altitude_uses_physical_density_and_delta(tmp_path, monkeypatch):
    original_close = plt.close
    monkeypatch.setattr(isr_plots.plt, 'close', lambda *_args, **_kwargs: None)
    observed = np.array([[1e10, 1e11], [1e12, 2e11]], dtype=np.float64)
    model = observed * 1.1
    day_record = {
        'ne_2d': observed,
        'alt_1d': np.array([200.0, 300.0]),
        'ts_1d': np.array([0.0, 3600.0]),
        'station': 'Test',
        'date_str': '2024-09-01',
        'plot_segs': [],
    }
    isr_plots.plot_time_altitude_comparison(
        day_record, np.log10(observed * 0.9), np.log10(observed),
        np.log10(model), tmp_path / 'time_altitude.png')

    figure = plt.gcf()
    panels = figure.axes[:6]
    np.testing.assert_allclose(
        np.asarray(panels[0].collections[0].get_array()).ravel(),
        density_to_display(observed).ravel())
    np.testing.assert_allclose(
        np.asarray(panels[4].collections[0].get_array()).ravel(),
        (density_to_display(model) - density_to_display(observed)).ravel())
    colorbar_labels = [axis.get_ylabel() for axis in figure.axes[6:]]
    assert colorbar_labels and all('log' not in label.lower()
                                   for label in colorbar_labels)
    original_close(figure)


def test_giro_nmf2_histogram_uses_physical_values(tmp_path, monkeypatch):
    captured = []
    original_hist2d = Axes.hist2d
    original_close = plt.close

    def capture_hist2d(axis, x, y, *args, **kwargs):
        captured.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return original_hist2d(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, 'hist2d', capture_hist2d)
    monkeypatch.setattr(evaluate_giro_peak.plt, 'close',
                        lambda *_args, **_kwargs: None)
    h_truth = np.array([220.0, 260.0, 300.0, 340.0])
    n_truth = np.array([10.0, 10.5, 11.0, 11.5])
    h_records = np.column_stack([np.zeros((4, 3)), h_truth])
    n_records = np.column_stack([np.zeros((4, 3)), n_truth])
    h_prediction = {
        source: {'hmf2': h_truth + offset}
        for source, offset in zip(('IRI', 'M00', 'M11'), (10.0, 5.0, 1.0))
    }
    n_prediction = {
        source: {'nmf2': n_truth + offset}
        for source, offset in zip(('IRI', 'M00', 'M11'), (0.2, 0.1, 0.05))
    }
    metric = {'ccc': 0.9, 'rmse': 0.1, 'pearson_r': 0.95}
    metrics = {source: metric for source in ('IRI', 'M00', 'M11')}
    evaluate_giro_peak._plot_density(
        tmp_path / 'giro.png', h_records, n_records,
        h_prediction, n_prediction, metrics, metrics)

    assert len(captured) == 6
    np.testing.assert_allclose(captured[3][0],
                               log10_density_to_display(n_truth))
    figure = plt.gcf()
    assert '10^{11}' in figure.axes[3].get_xlabel()
    assert 'log' not in figure.axes[3].get_xlabel().lower()
    original_close(figure)


def test_isr_profile_extraction_returns_physical_display_units():
    record = {
        'ts_1d': np.array([1725148800.0]),
        'ne_2d': np.array([[1e10], [1e11], [1e12]]),
        'alt_1d': np.array([200.0, 300.0, 400.0]),
    }
    density, altitude = _extract_isr_profile(record, 0.0)
    np.testing.assert_allclose(density, [0.1, 1.0, 10.0])
    np.testing.assert_allclose(altitude, [200.0, 300.0, 400.0])
