import torch

from inr_modules.mdia.m2u_anchor_etkf import (
    accumulate_anchor_terms,
    anchor_mixing_weights,
    blend_anchor_increments,
    flat_top_cover,
    solve_anchor_weights,
    sparsemax,
)


def test_sparsemax_has_one_hot_core_and_continuous_simplex():
    scores = torch.tensor([[3.0, 0.0, -2.0], [0.2, 0.1, -1.0]])
    weights = sparsemax(scores)
    assert torch.equal(weights[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.all(weights >= 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_flat_cover_fades_single_anchor_to_background():
    anchor = torch.tensor([[0.0, 0.0, 200.0, 0.0]])
    query = torch.tensor([
        [0.0, 0.0, 200.0, 0.0],
        [0.0, 0.0, 200.0, 0.75],
        [0.0, 0.0, 200.0, 1.5],
    ])
    cover = flat_top_cover(query, anchor).squeeze(-1)
    assert torch.equal(cover[:2], torch.ones(2))
    assert torch.equal(cover[2:], torch.zeros(1))
    mixed = anchor_mixing_weights(
        query, anchor, None, None, ell_e=1.0, ell_a=1.0, tau=1.0)
    assert torch.equal(mixed['beta'][0], torch.ones(1))
    assert torch.equal(mixed['beta'][1], torch.ones(1))
    assert torch.equal(mixed['beta'][2], torch.zeros(1))
    assert torch.equal(mixed['background_weight'], torch.tensor([0.0, 0.0, 1.0]))


def test_anchor_terms_and_shared_increment():
    anomalies = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    precision = torch.ones(1, 2)
    innovation = torch.tensor([[2.0, -1.0]])
    covariance, rhs = accumulate_anchor_terms(
        anomalies, precision, innovation)
    weights = solve_anchor_weights(covariance, rhs, base_rank=1)
    query_anomalies = torch.tensor([[0.4, -0.2], [0.4, -0.2]])
    beta = torch.ones(2, 1)
    increment = blend_anchor_increments(query_anomalies, beta, weights)
    assert torch.equal(increment[0], increment[1])
    assert torch.isfinite(increment).all()


if __name__ == '__main__':
    test_sparsemax_has_one_hot_core_and_continuous_simplex()
    test_flat_cover_fades_single_anchor_to_background()
    test_anchor_terms_and_shared_increment()
    print('M2-U anchor ETKF tests passed')
