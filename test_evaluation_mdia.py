"""Regression check for the seven-value evaluation batch interface."""

import numpy as np
import torch

from inr_modules.mdia.evaluation_mdia import _collect_predictions


def test_collect_predictions_passes_both_observation_sources():
    class Processor:
        def process_batch(self, _):
            batch = 2
            return (
                torch.zeros(batch, 4), torch.tensor([[1.0], [2.0]]),
                torch.zeros(batch, 3, 2), torch.ones(batch, 1, 10),
                torch.ones(batch), torch.full((batch, 1, 10), 2.0),
                torch.ones(batch))

    class PeakManager:
        def get_iri_peak(self, coords):
            return torch.tensor([[300.0, 11.5]]).expand(len(coords), -1)

    class Model:
        def eval(self):
            return self

        def __call__(self, coords, sw_seq, **kwargs):
            assert kwargs['neighbors_feats'] is not None
            assert kwargs['neighbors_feats_cosmic'] is not None
            assert kwargs['has_obs'] is not None
            assert kwargs['has_obs_cosmic'] is not None
            assert kwargs['iri_peak'] is not None
            pred = torch.ones(len(coords), 1)
            return pred, pred, pred, pred, {'ne_bkg': torch.zeros_like(pred)}

    pred, bkg, target = _collect_predictions(
        Model(), [None], Processor(), PeakManager())
    np.testing.assert_array_equal(pred, [1.0, 1.0])
    np.testing.assert_array_equal(bkg, [0.0, 0.0])
    np.testing.assert_array_equal(target, [1.0, 2.0])


if __name__ == '__main__':
    test_collect_predictions_passes_both_observation_sources()
