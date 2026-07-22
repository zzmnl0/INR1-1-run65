"""Regression check that ISR queries pass FY and COSMIC neighborhoods."""

import numpy as np
import torch

from isr_evaluation.model_query import query_model_grid


def test_query_model_grid_passes_both_sources():
    class Index:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def query_batch_np(self, coords):
            self.calls += 1
            return (np.full((len(coords), 2, 10), self.value, np.float32),
                    np.ones(len(coords), np.float32))

    class SpaceWeather:
        def get_drivers_sequence(self, rel_hour):
            return torch.zeros(len(rel_hour), 3, 2)

    class Peaks:
        def get_iri_peak(self, coords):
            return torch.tensor([[300.0, 11.5]]).expand(len(coords), -1)

    class Model:
        def eval(self):
            return self

        def __call__(self, coords, sw_seq, **kwargs):
            assert kwargs['neighbors_feats'].mean() == 1
            assert kwargs['neighbors_feats_cosmic'].mean() == 2
            assert kwargs['has_obs'].all()
            assert kwargs['has_obs_cosmic'].all()
            assert kwargs['iri_peak'] is not None
            pred = torch.ones(len(coords), 1)
            return pred, pred, pred, pred, {'ne_bkg': torch.zeros_like(pred)}

    fy_index, cosmic_index = Index(1), Index(2)
    record = {
        'alt_1d': np.array([200.0, 300.0], np.float32),
        'ts_1d': np.array([0.0, 3600.0]),
        'ne_2d': np.ones((2, 2), np.float32),
        'lat': 10.0,
        'lon': 20.0,
    }
    pred, iri = query_model_grid(
        Model(), SpaceWeather(), record, 0.0, torch.device('cpu'),
        batch_size=2, iri_peak_manager=Peaks(),
        fy_nb_index=fy_index, cosmic_nb_index=cosmic_index)
    np.testing.assert_array_equal(pred, np.ones((2, 2), np.float32))
    np.testing.assert_array_equal(iri, np.zeros((2, 2), np.float32))
    assert fy_index.calls == cosmic_index.calls == 2


if __name__ == '__main__':
    test_query_model_grid_passes_both_sources()
