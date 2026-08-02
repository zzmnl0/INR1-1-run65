"""Focused checks for the read-only observation-subspace audit."""

import torch

from audit_etkf_observation_subspace import _direction_summary, _geometry


def test_observation_geometry_detects_redundancy():
    orthogonal = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    full, _ = _geometry(orthogonal, torch.ones(2))
    assert full["numeric_rank"] == 2
    assert abs(full["effective_rank"] - 2.0) < 1e-6

    repeated = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    collapsed, _ = _geometry(repeated, torch.ones(2))
    assert collapsed["numeric_rank"] == 1
    assert abs(collapsed["effective_rank"] - 1.0) < 1e-6

    assert _geometry(orthogonal, torch.zeros(2)) == (None, None)

    profile_weighted = _direction_summary({
        1: [True, False],
        2: [True],
    })
    assert profile_weighted["profiles"] == 2
    assert profile_weighted["queries"] == 3
    assert abs(profile_weighted["direction_fraction"] - 0.75) < 1e-12
