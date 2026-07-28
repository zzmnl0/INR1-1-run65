"""Focused regression check for physical source-attribution diagnostics."""

import numpy as np
import torch

from isr_evaluation.diagnose_jicamarca_modes import (
    _observation_diagnostics,
    _observation_summary,
    _query_mean,
    _source_arrays,
    _summary,
)


def test_physical_source_attribution():
    payload = {
        'valid_mask': torch.tensor([[True, True, False]]),
        'profile_id': torch.tensor([[10, 11, -1]]),
        'rho_squared': torch.tensor([[0.0, 0.25, 1.0]]),
    }
    extras = {
        'innov_FY': torch.tensor([[-0.2, 0.1, 0.0]]),
        'K_FY': torch.tensor([[0.5, 0.25, 0.0]]),
        'precision_FY': torch.tensor([[25.0, 10.0, 0.0]]),
        'cross_covariance_FY': torch.tensor([[0.4, -0.1, 0.0]]),
    }
    result = _source_arrays(payload, extras, 'FY')
    assert np.allclose(result['kalman_contribution_sum'], [-0.075])
    assert np.allclose(result['r_eff_mean'], [0.07])
    assert np.array_equal(result['dominant_profile_id'], [10])
    assert np.allclose(result['dominant_rho'], [0.0])

    report = _summary(
        np.array([True, True]),
        np.array([9.9, 10.1]),
        np.array([10.0, 10.0]),
        np.array([9.8, 10.2]))
    assert report['n'] == 2
    assert report['toward_isr_fraction'] == 1.0
    assert np.isnan(_query_mean(
        np.zeros((1, 2)), np.zeros((1, 2), dtype=bool))[0])
    payload.update({
        'source': torch.tensor([[0, 0, 0]], dtype=torch.int8),
        'coords': torch.zeros(1, 3, 4),
        'value': torch.tensor([[9.8, 10.1, 0.0]]),
        'background': torch.tensor([[10.0, 10.0, 0.0]]),
    })
    token_rows = _observation_diagnostics(
        payload, extras, 'FY', np.array([True]), 7,
        np.array([9.9]), np.array([10.0]))
    assert np.array_equal(token_rows['query_index'], [7, 7])
    assert np.allclose(token_rows['kalman_contribution'], [-0.1, 0.025])
    summary = _observation_summary(token_rows, 0)
    assert summary['n'] == 2
    assert summary['negative_gain_nonnegative_cross_covariance_n'] == 0


if __name__ == '__main__':
    test_physical_source_attribution()
    print('Source diagnostic regression checks passed')
