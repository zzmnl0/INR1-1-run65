"""Regression check for the profile-aware evaluation batch interface."""

import numpy as np
import torch

from inr_modules.mdia.evaluation_mdia import _collect_predictions


def test_collect_predictions_passes_both_observation_sources():
    class Processor:
        class SW:
            @staticmethod
            def get_drivers_sequence(time):
                return torch.zeros(len(time), 3, 2)

        sw_manager = SW()

        def process_batch(self, _):
            batch = 2
            payload = {
                'coords': torch.zeros(batch, 1, 4),
                'value': torch.ones(batch, 1),
                'valid_mask': torch.ones(batch, 1, dtype=torch.bool),
                'profile_id': torch.arange(batch)[:, None],
                'source': torch.zeros(batch, 1, dtype=torch.int8),
                'rho_squared': torch.zeros(batch, 1),
            }
            return (
                torch.zeros(batch, 4), torch.tensor([[1.0], [2.0]]),
                torch.zeros(batch, 3, 2), payload, payload,
                torch.tensor([10, 11]))

    class PeakManager:
        def get_iri_peak(self, coords):
            return torch.tensor([[300.0, 11.5]]).expand(len(coords), -1)

    class Model:
        def eval(self):
            return self

        @staticmethod
        def encode_background(coords, sw_seq, iri_peak=None):
            return {'ne_bkg': torch.zeros(len(coords), 1)}

        def __call__(self, coords, sw_seq, **kwargs):
            assert kwargs['observations_fy']['background'] is not None
            assert kwargs['observations_cosmic']['background'] is not None
            assert kwargs['iri_peak'] is not None
            pred = torch.ones(len(coords), 1)
            return pred, pred, pred, pred, {
                'ne_bkg': torch.zeros_like(pred),
                'ne_iri': torch.full_like(pred, -1.0),
            }

    pred, bkg, iri, target = _collect_predictions(
        Model(), [None], Processor(), PeakManager())
    np.testing.assert_array_equal(pred, [1.0, 1.0])
    np.testing.assert_array_equal(bkg, [0.0, 0.0])
    np.testing.assert_array_equal(iri, [-1.0, -1.0])
    np.testing.assert_array_equal(target, [1.0, 2.0])


if __name__ == '__main__':
    test_collect_predictions_passes_both_observation_sources()
