import math

import numpy as np
import torch

from evaluate_satellite_development import (
    N_EMPIRICAL_CELLS,
    _apply_stable_cell_mask,
    _attribution_record,
    _single_source_gain,
    _solve_mode,
    _summarize_records,
)
from inr_modules.mdia.fsia_model import NeuralETKFLayer
from inr_modules.mdia.sliding_dataset import (
    REPRESENTATIVENESS_SHAPE,
    attach_representativeness_weight,
)


def test_development_summary_is_profile_balanced_and_marks_sparse_cells():
    base = {
        "date": 3,
        "rmse": 1.0,
        "mae": 1.0,
        "bias": 0.0,
        "nis": 1.0,
        "coverage90": 1.0,
        "precision_coverage": 1.0,
        "direction_fraction": 1.0,
        "negative_correct_fraction": math.nan,
        "negative_positive_tail_fraction": math.nan,
        "negative_increment_p95": math.nan,
        "negative_increment_p99": math.nan,
    }
    other = {**base, "date": 8, "rmse": 3.0, "mae": 3.0}
    result = _summarize_records([base, other])
    assert result["rmse"] == 2.0
    assert result["mae"] == 2.0
    assert result["profiles"] == 2
    assert result["dates"] == 2
    assert not result["estimable"]
    assert result["direction_profiles"] == 2
    assert result["direction_dates"] == 2
    assert not result["direction_estimable"]


def test_mode_solver_recovers_closed_form_etkf_mean_and_variance():
    latent = torch.tensor([[
        [1.0, 0.0],
        [-0.5, 0.5],
        [-0.5, -0.5],
    ]])
    basis = torch.tensor([[0.8, -0.2]])
    obs_anomalies = torch.tensor([[[1.0, -0.5, -0.5]]])
    precision = torch.tensor([[4.0]])
    innovation = torch.tensor([[-0.3]])
    extras = {
        "query_basis": basis,
        "latent_anomalies": latent,
        "obs_anomalies_FY": obs_anomalies,
        "precision_FY": precision,
        "innov_FY": innovation,
    }
    increment, variance = _solve_mode(extras, ("FY",), 0.04)
    query = torch.einsum("bd,bnd->bn", basis, latent)
    system = (
        2.0 * torch.eye(3).unsqueeze(0)
        + torch.einsum(
            "bmn,bm,bmk->bnk",
            obs_anomalies,
            precision,
            obs_anomalies,
        )
    )
    rhs = torch.einsum(
        "bmn,bm,bm->bn", obs_anomalies, precision, innovation
    )
    expected_weight = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    expected_increment = (query * expected_weight).sum(-1)
    expected_variance = (
        query * torch.linalg.solve(
            system, query.unsqueeze(-1)
        ).squeeze(-1)
    ).sum(-1) + 0.04
    assert torch.allclose(increment, expected_increment, atol=1e-7)
    assert torch.allclose(variance, expected_variance, atol=1e-7)


def test_exact_gain_reconstructs_single_source_increment_and_attribution():
    query = torch.tensor([[0.9, -0.3, -0.6]])
    obs_anomalies = torch.tensor([[
        [1.0, -0.5, -0.5],
        [-0.2, 0.7, -0.5],
    ]])
    precision = torch.tensor([[4.0, 2.0]])
    innovation = torch.tensor([[-0.3, 0.2]])
    covariance = torch.einsum(
        "bmn,bm,bmk->bnk",
        obs_anomalies,
        precision,
        obs_anomalies,
    )
    extras = {
        "query_anomalies": query,
        "obs_anomalies_FY": obs_anomalies,
        "precision_FY": precision,
        "innov_FY": innovation,
        "ensemble_covariance_FY": covariance,
    }
    gain = _single_source_gain(extras, "FY")
    increment = _solve_mode(extras, ("FY",), None)
    assert torch.allclose(
        (gain * innovation).sum(dim=-1), increment, atol=1e-7
    )

    record = _attribution_record(
        np.array([True]),
        np.array([-0.2]),
        innovation.numpy(),
        np.array([[0.1, -0.1]]),
        gain.numpy(),
        precision.numpy(),
        date=3,
    )
    assert record["tokens"] == 2.0
    assert record["innovation_target_agreement"] == 0.5
    assert all(math.isfinite(value) for value in record.values())


def test_stable_cell_mask_filters_tokens_without_mutating_payload():
    query = torch.tensor([[0.0, 0.0, 150.0, 0.0]])
    payload = {
        "coords": torch.tensor([[
            [0.0, 0.0, 160.0, 0.0],
            [0.0, 0.0, 250.0, 12.0],
            [float("nan"), 0.0, 0.0, 0.0],
        ]]),
        "valid_mask": torch.tensor([[True, True, False]]),
        "rho_squared": torch.tensor([[0.01, 0.04, float("nan")]]),
    }
    stable = np.zeros(N_EMPIRICAL_CELLS, dtype=bool)
    stable[108] = True  # FY target -> COSMIC, low-low, night-night, rho<0.25
    filtered, counts = _apply_stable_cell_mask(
        payload, query, "FY", "COSMIC", stable
    )
    assert filtered["valid_mask"].tolist() == [[True, False, False]]
    assert payload["valid_mask"].tolist() == [[True, True, False]]
    assert counts == {
        "tokens": 2,
        "queries": 1,
        "retained_tokens": 1,
        "retained_queries": 1,
    }


def test_continuous_representativeness_weight_and_precision():
    query = torch.tensor([
        [0.0, 0.0, 199.999, 6.0],
        [0.0, 0.0, 200.001, 6.0],
    ])
    payload = {
        "coords": torch.tensor([
            [[0.0, 0.0, 250.0, 6.0]],
            [[0.0, 0.0, 250.0, 6.0]],
        ]),
        "value": torch.ones(2, 1),
        "background": torch.zeros(2, 1),
        "valid_mask": torch.ones(2, 1, dtype=torch.bool),
        "rho_squared": torch.zeros(2, 1),
    }
    grid = torch.zeros(REPRESENTATIVENESS_SHAPE)
    weighted = attach_representativeness_weight(
        payload, query, "FY", "FY", grid, floor=0.25
    )
    assert torch.allclose(
        weighted["representativeness_weight"],
        torch.full((2, 1), 0.25),
    )
    assert "representativeness_weight" not in payload
    varying = torch.zeros(REPRESENTATIVENESS_SHAPE)
    varying[0, 1] = 1.0
    continuous = attach_representativeness_weight(
        payload, query, "FY", "FY", varying, floor=0.25
    )["representativeness_weight"]
    difference = torch.abs(continuous[0] - continuous[1]).max()
    assert 0.0 < difference < 1e-4

    layer = NeuralETKFLayer(d_model=2, b_net_in=3, n_members=3)
    terms = layer._source_terms(
        torch.ones(2, 3, 2),
        torch.ones(2, 1, 2),
        weighted,
        torch.tensor(0.04),
    )
    assert torch.allclose(terms[4], torch.full((2, 1), 6.25))
