"""Focused regression checks for the read-only M2-M audit."""

import torch

from audit_m2m_mapping_loss import (
    _covariance_target_summary,
    _direct_cross_covariance,
    _gradient_comparison,
    _reciprocity_summary,
)
from test_run66_losses import _model, _model_observation


def test_profile_pair_reciprocity_ignores_token_duplication():
    row = {
        "target_profile_id": 1,
        "observation_profile_id": 2,
        "cell_id": 3,
        "forward": 0.2,
        "reverse": 0.1,
        "weight": 1.0,
    }
    base = _reciprocity_summary([row])
    duplicate = _reciprocity_summary([
        {**row, "weight": 0.5}, {**row, "weight": 0.5}
    ])
    assert base == duplicate
    assert base["sign_reciprocity"] == 1.0
    assert base["absolute_ratio"]["q05_q50_q95"][1] == 2.0


def test_covariance_target_summary_recovers_sign_and_scale():
    rows = [{
        "target_profile_id": 1,
        "observation_profile_id": 2,
        "cell_id": 3,
        "predicted": -0.3,
        "target": -0.2,
        "weight": 1.0,
    }]
    result = _covariance_target_summary(rows)
    assert result["sign_accuracy"] == 1.0
    assert abs(
        result["relative_absolute_error"]["q05_q50_q95"][1] - 0.5
    ) < 1e-12
    assert abs(
        result["signed_amplitude_ratio"]["q05_q50_q95"][1] - 1.5
    ) < 1e-12


def test_gradient_comparison_detects_opposition():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = {
        "observation": parameter.sum(),
        "covariance": -parameter.sum(),
        "direction": (2.0 * parameter).sum(),
    }
    result = _gradient_comparison(losses, [parameter])
    assert abs(result["cosine_observation_covariance"] + 1.0) < 2e-7
    assert abs(result["cosine_observation_direction"] - 1.0) < 2e-7


def test_direct_cross_covariance_matches_forward_operator():
    torch.manual_seed(42)
    model = _model().eval()
    query = torch.tensor([
        [-12.0, -76.8, 250.0, 48.0],
        [20.0, 120.0, 310.0, 96.0],
    ])
    target = query.clone()
    target[:, 2] += 10.0
    sw = torch.zeros(2, 4, 2)
    peak = torch.tensor([[300.0, 11.5], [300.0, 11.5]])
    target_background = model(target, sw, iri_peak=peak)[4]["ne_bkg"]
    observations = _model_observation(target, target_background, 0)
    extras = model(
        query, sw, iri_peak=peak, observations_fy=observations)[4]
    direct = _direct_cross_covariance(
        model, query, target, target_background.squeeze(-1), sw, peak)
    assert torch.allclose(
        direct, extras["cross_covariance_FY"].squeeze(-1), atol=1e-7)
