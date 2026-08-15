"""Unit tests for the pure P0-B ETKF counterfactual layer."""

import torch

from inr_modules.mdia.p0b_counterfactuals import (
    PREREGISTERED_ALTITUDE_DELETION_BANDS,
    RaggedSourceTerms,
    aggregate_ragged_source_terms,
    delete_altitude_band,
    diag_r_standardized_innovation_sq,
    duplicate_max_precision_profile,
    empty_ragged_source_terms,
    predictive_nis_low_rank,
    predictive_nis_valid_dof,
    recompute_edge_gain_coefficients,
    run_p0b_counterfactual_suite,
    solve_etkf_counterfactual_modes,
)


DTYPE = torch.float64


def _terms(source, query_index, profile_id, altitude_km, anomalies,
           innovation, localized_precision, diag_r_precision=None):
    localized_precision = torch.tensor(localized_precision, dtype=DTYPE)
    if diag_r_precision is None:
        diag_r_precision = localized_precision.clone()
    else:
        diag_r_precision = torch.tensor(diag_r_precision, dtype=DTYPE)
    return RaggedSourceTerms(
        source=source,
        query_index=torch.tensor(query_index, dtype=torch.long),
        profile_id=torch.tensor(profile_id, dtype=torch.long),
        altitude_km=torch.tensor(altitude_km, dtype=DTYPE),
        obs_anomalies=torch.tensor(anomalies, dtype=DTYPE),
        innovation=torch.tensor(innovation, dtype=DTYPE),
        localized_precision=localized_precision,
        diag_r_precision=diag_r_precision,
    )


def _explicit_mode(query_anomaly, source_terms):
    members = query_anomaly.numel()
    system = (members - 1) * torch.eye(members, dtype=DTYPE)
    rhs = torch.zeros(members, dtype=DTYPE)
    for terms in source_terms:
        y = terms.obs_anomalies
        p = terms.localized_precision
        d = terms.innovation
        system = system + y.T @ torch.diag(p) @ y
        rhs = rhs + y.T @ (p * d)
    return query_anomaly @ torch.linalg.solve(system, rhs)


def test_ragged_modes_match_explicit_query_solves_and_close_exactly():
    query_anomalies = torch.tensor([
        [0.7, -0.2, 0.4],
        [-0.1, 0.9, 0.3],
        [0.2, 0.1, -0.5],
    ], dtype=DTYPE)
    fy = _terms(
        "FY",
        query_index=[0, 0, 1],
        profile_id=[10, 11, 12],
        altitude_km=[220, 330, 410],
        anomalies=[[1.0, 0.2, -0.1], [0.3, -0.4, 0.8],
                   [0.5, 0.1, 0.7]],
        innovation=[0.4, -0.7, 0.2],
        localized_precision=[1.2, 0.5, 2.0],
    )
    cosmic = _terms(
        "COSMIC",
        query_index=[0, 1, 1],
        profile_id=[20, 21, 22],
        altitude_km=[260, 290, 450],
        anomalies=[[0.4, 0.9, -0.2], [-0.6, 0.3, 0.2],
                   [0.2, -0.5, 1.0]],
        innovation=[0.6, -0.3, 0.8],
        localized_precision=[0.8, 1.1, 0.4],
    )
    solution = solve_etkf_counterfactual_modes(
        query_anomalies, {"FY": fy, "COSMIC": cosmic})

    for query in range(3):
        fy_q = fy.subset(fy.query_index == query)
        cosmic_q = cosmic.subset(cosmic.query_index == query)
        expected_m10 = _explicit_mode(query_anomalies[query], [fy_q])
        expected_m01 = _explicit_mode(query_anomalies[query], [cosmic_q])
        expected_m11 = _explicit_mode(
            query_anomalies[query], [fy_q, cosmic_q])
        torch.testing.assert_close(solution.m10[query], expected_m10)
        torch.testing.assert_close(solution.m01[query], expected_m01)
        torch.testing.assert_close(solution.m11[query], expected_m11)

    assert torch.equal(solution.m00, torch.zeros(3, dtype=DTYPE))
    assert torch.equal(
        solution.m11,
        solution.joint_fy_contribution
        + solution.joint_cosmic_contribution)
    assert torch.equal(
        solution.joint_closure_error, torch.zeros(3, dtype=DTYPE))
    # Isolated increments are intentionally not used as joint contributions.
    assert not torch.allclose(solution.m11, solution.m10 + solution.m01)


def test_four_preregistered_altitude_deletions_have_exact_boundaries():
    terms = _terms(
        "FY",
        query_index=list(range(10)),
        profile_id=list(range(10)),
        altitude_km=[199.9, 200.0, 249.9, 250.0, 299.9,
                     300.0, 399.9, 400.0, 500.0, 500.1],
        anomalies=[[1.0, 0.0]] * 10,
        innovation=[1.0] * 10,
        localized_precision=[1.0] * 10,
    )
    expected_deleted = {
        "drop_200_250": {1, 2},
        "drop_250_300": {3, 4},
        "drop_300_400": {5, 6},
        "drop_400_500": {7, 8},
    }
    assert len(PREREGISTERED_ALTITUDE_DELETION_BANDS) == 4
    all_deleted = set()
    for band in PREREGISTERED_ALTITUDE_DELETION_BANDS:
        deleted = set(torch.nonzero(
            band.deletion_mask(terms.altitude_km),
            as_tuple=False).squeeze(-1).tolist())
        assert deleted == expected_deleted[band.name]
        assert not (all_deleted & deleted)
        all_deleted |= deleted
        retained = delete_altitude_band(terms, band)
        assert retained.edge_count == terms.edge_count - len(deleted)
    assert all_deleted == set(range(1, 9))


def test_profile_duplication_uses_query_source_sum_and_minimum_id_tie():
    fy = _terms(
        "FY",
        query_index=[0, 0, 0, 0, 1, 1, 1, 2, 2],
        profile_id=[9, 9, 3, 3, 8, 7, 7, 6, 5],
        altitude_km=[210, 220, 230, 240, 250, 260, 270, 280, 290],
        anomalies=[[1.0, 0.0]] * 9,
        innovation=[1.0] * 9,
        # q0: profiles 9 and 3 tie at 2.0 -> choose 3.
        # q1: profile 7 totals 1.1 and beats profile 8 at 1.0.
        # q2: all profiles have zero precision -> select nothing.
        localized_precision=[
            1.0, 1.0, 0.5, 1.5, 1.0, 0.4, 0.7, 0.0, 0.0],
    )
    duplicated = duplicate_max_precision_profile(fy, n_queries=4)
    assert duplicated.selected.tolist() == [True, True, False, False]
    assert duplicated.selected_profile_id[:2].tolist() == [3, 7]
    assert duplicated.selected_profile_id[2:].tolist() == [-1, -1]
    assert duplicated.terms.edge_count == fy.edge_count + 4
    torch.testing.assert_close(
        duplicated.terms.localized_precision.sum(),
        fy.localized_precision.sum() + torch.tensor(3.1, dtype=DTYPE))
    assert duplicated.terms.profile_id[-4:].tolist() == [3, 3, 7, 7]


def test_edge_gain_coefficients_close_isolated_and_joint_contributions():
    query_anomalies = torch.tensor([
        [0.7, -0.2, 0.4],
        [-0.1, 0.9, 0.3],
    ], dtype=DTYPE)
    sources = {
        "FY": _terms(
            "FY", [0, 0, 1], [1, 2, 3], [220, 310, 450],
            [[1.0, 0.2, -0.1], [0.3, -0.4, 0.8],
             [0.5, 0.1, 0.7]],
            [0.4, -0.7, 0.2], [1.2, 0.5, 2.0]),
        "COSMIC": _terms(
            "COSMIC", [0, 1, 1], [4, 5, 6], [260, 290, 420],
            [[0.4, 0.9, -0.2], [-0.6, 0.3, 0.2],
             [0.2, -0.5, 1.0]],
            [0.6, -0.3, 0.8], [0.8, 1.1, 0.4]),
    }
    solution = solve_etkf_counterfactual_modes(query_anomalies, sources)
    gains = recompute_edge_gain_coefficients(query_anomalies, sources)
    isolated_expected = {"FY": solution.m10, "COSMIC": solution.m01}
    joint_expected = {
        "FY": solution.joint_fy_contribution,
        "COSMIC": solution.joint_cosmic_contribution,
    }
    for source, terms in sources.items():
        isolated = torch.zeros(2, dtype=DTYPE).index_add(
            0, terms.query_index,
            gains[source].isolated_system_gain_coefficient
            * terms.innovation)
        joint = torch.zeros(2, dtype=DTYPE).index_add(
            0, terms.query_index,
            gains[source].joint_system_gain_coefficient
            * terms.innovation)
        torch.testing.assert_close(isolated, isolated_expected[source])
        torch.testing.assert_close(joint, joint_expected[source])


def test_float32_repeated_edges_match_float64_accumulate_then_float32_solve():
    edge_count = 513
    query_anomalies = torch.tensor(
        [[0.35, -0.8, 0.45]], dtype=torch.float32)
    anomaly_row = torch.tensor(
        [0.03125, -0.046875, 0.078125], dtype=torch.float32)
    anomalies = anomaly_row.repeat(edge_count, 1)
    # Vary small weights/innovations so float32 accumulation would measurably
    # differ from the frozen float64 index-add path.
    precision = torch.linspace(
        1.0e-4, 2.5e-3, edge_count, dtype=torch.float32)
    innovation = torch.linspace(
        -0.75, 0.55, edge_count, dtype=torch.float32)
    fy = RaggedSourceTerms(
        source="FY",
        query_index=torch.zeros(edge_count, dtype=torch.long),
        profile_id=torch.arange(edge_count, dtype=torch.long),
        altitude_km=torch.full((edge_count,), 300.0, dtype=torch.float32),
        obs_anomalies=anomalies,
        innovation=innovation,
        localized_precision=precision,
        diag_r_precision=precision,
    )
    cosmic = empty_ragged_source_terms(
        "COSMIC", 3, dtype=torch.float32)
    sources = {"FY": fy, "COSMIC": cosmic}

    outer64 = torch.einsum(
        "en,e,em->enm", anomalies.double(), precision.double(),
        anomalies.double())
    covariance64 = torch.zeros(1, 3, 3, dtype=torch.float64).index_add(
        0, fy.query_index, outer64)
    rhs64 = torch.zeros(1, 3, dtype=torch.float64).index_add(
        0, fy.query_index,
        anomalies.double()
        * (precision.double() * innovation.double()).unsqueeze(-1))
    covariance32 = covariance64.float()
    rhs32 = rhs64.float()
    system32 = (2.0 * torch.eye(3, dtype=torch.float32).unsqueeze(0)
                + covariance32)
    chol32 = torch.linalg.cholesky(system32)
    weights32 = torch.cholesky_solve(
        rhs32.unsqueeze(-1), chol32).squeeze(-1)
    expected_increment = query_anomalies[0] @ weights32[0]

    aggregated = aggregate_ragged_source_terms(
        fy, n_queries=1, query_anomalies=query_anomalies)
    assert torch.equal(aggregated.covariance, covariance32)
    assert torch.equal(aggregated.rhs, rhs32)
    solution = solve_etkf_counterfactual_modes(query_anomalies, sources)
    torch.testing.assert_close(
        solution.m10[0], expected_increment, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        solution.m11[0], expected_increment, rtol=0.0, atol=1e-7)

    gains = recompute_edge_gain_coefficients(query_anomalies, sources)["FY"]
    weighted_y = anomalies * precision.unsqueeze(-1)
    solved_y = torch.cholesky_solve(
        weighted_y.unsqueeze(-1),
        chol32[fy.query_index],
    ).squeeze(-1)
    expected_gain = torch.einsum(
        "en,en->e", query_anomalies[fy.query_index], solved_y)
    assert torch.equal(
        gains.isolated_system_gain_coefficient, expected_gain)
    assert torch.equal(gains.joint_system_gain_coefficient, expected_gain)
    closed = torch.zeros(1, dtype=torch.float32).index_add(
        0, fy.query_index,
        gains.joint_system_gain_coefficient * innovation)
    torch.testing.assert_close(closed, solution.m11, rtol=0.0, atol=5e-6)


def test_suite_duplicates_fy_cosmic_and_both_independently():
    query_anomalies = torch.tensor([[0.6, -0.4]], dtype=DTYPE)
    fy = _terms(
        "FY", [0, 0], [2, 1], [220, 230],
        [[1.0, 0.2], [0.4, -0.7]], [0.8, -0.2], [1.0, 1.0],
        diag_r_precision=[2.0, 2.0])
    cosmic = _terms(
        "COSMIC", [0], [4], [310], [[-0.2, 0.9]], [0.5], [0.6],
        diag_r_precision=[1.5])
    suite = run_p0b_counterfactual_suite(
        query_anomalies, {"FY": fy, "COSMIC": cosmic})

    assert set(suite.profile_duplications) == {
        "duplicate_FY", "duplicate_COSMIC", "duplicate_both"}
    assert suite.selected_profile_ids["FY"].item() == 1
    assert suite.selected_profile_ids["COSMIC"].item() == 4
    # Each both-source result is a fresh joint solve, not a sum of the two
    # single-duplication counterfactual outputs.
    assert torch.isfinite(
        suite.profile_duplications["duplicate_both"].m11).all()
    assert set(suite.altitude_deletions) == {
        band.name for band in PREREGISTERED_ALTITUDE_DELETION_BANDS}


def test_diag_r_standardized_innovation_sq_is_unlocalized():
    innovation = torch.tensor([2.0, -3.0], dtype=DTYPE)
    diag_r_precision = torch.tensor([0.5, 4.0], dtype=DTYPE)
    localized_precision = torch.tensor([0.05, 0.0], dtype=DTYPE)
    actual = diag_r_standardized_innovation_sq(
        innovation, diag_r_precision)
    torch.testing.assert_close(
        actual, torch.tensor([2.0, 36.0], dtype=DTYPE))
    assert not torch.equal(
        actual, innovation.square() * localized_precision)


def test_predictive_nis_low_rank_matches_explicit_dense_inverse():
    anomalies = torch.tensor([
        [1.0, 0.2, -0.3],
        [0.1, -0.7, 0.5],
        [0.4, 0.8, -0.2],
        [-0.6, 0.3, 0.9],
        [0.2, -0.1, 0.7],
    ], dtype=DTYPE)
    innovation = torch.tensor([0.5, -0.2, 0.9, -0.4, 0.3], dtype=DTYPE)
    precision = torch.tensor([2.0, 0.7, 1.3, 3.0, 0.4], dtype=DTYPE)
    query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    low_rank = predictive_nis_low_rank(
        anomalies, innovation, precision, query_index, n_queries=3)

    expected = torch.zeros(3, dtype=DTYPE)
    members = anomalies.shape[1]
    for query in range(3):
        mask = query_index == query
        if not mask.any():
            continue
        y = anomalies[mask]
        r = torch.diag(precision[mask].reciprocal())
        predictive_covariance = r + y @ y.T / (members - 1)
        d = innovation[mask]
        expected[query] = d @ torch.linalg.solve(predictive_covariance, d)
    torch.testing.assert_close(low_rank, expected, rtol=1e-12, atol=1e-12)
    assert predictive_nis_valid_dof(
        precision, query_index, n_queries=3).tolist() == [3, 2, 0]
    precision_with_zero = precision.clone()
    precision_with_zero[1] = 0.0
    assert predictive_nis_valid_dof(
        precision_with_zero, query_index,
        n_queries=3).tolist() == [2, 2, 0]


def test_predictive_nis_float32_high_precision_is_nonnegative_float64():
    generator = torch.Generator().manual_seed(42015)
    edge_count = 256
    members = 8
    anomalies = torch.randn(
        edge_count, members, generator=generator, dtype=torch.float32)
    coefficient = torch.randn(
        members, generator=generator, dtype=torch.float32)
    innovation = anomalies @ coefficient
    precision = torch.full(
        (edge_count,), 1.0e8, dtype=torch.float32)
    query_index = torch.zeros(edge_count, dtype=torch.long)

    actual = predictive_nis_low_rank(
        anomalies, innovation, precision, query_index, n_queries=1)
    y = anomalies.double()
    d = innovation.double()
    p = precision.double()
    rhs = y.T @ (p * d)
    system = ((members - 1) * torch.eye(members, dtype=torch.float64)
              + y.T @ (p[:, None] * y))
    expected = d @ (p * d) - rhs @ torch.linalg.solve(system, rhs)

    assert actual.dtype == torch.float64
    assert actual.item() >= 0.0
    torch.testing.assert_close(actual[0], expected, rtol=1e-5, atol=1e-5)


def test_precast_float64_anomalies_reproduce_v14_ragged_accumulation():
    generator = torch.Generator().manual_seed(67)
    queries, members, dimensions, edges = 3, 5, 7, 1025
    query_anomalies = torch.randn(
        queries, members, generator=generator, dtype=torch.float32)
    query_index = torch.randint(
        0, queries, (edges,), generator=generator, dtype=torch.long)
    basis = torch.randn(
        edges, dimensions, generator=generator, dtype=torch.float32)
    latent = torch.randn(
        queries, members, dimensions, generator=generator,
        dtype=torch.float32)
    precast = torch.einsum(
        "ed,end->en", basis.double(), latent[query_index].double())
    precision = torch.logspace(
        -5, 5, edges, dtype=torch.float32)
    innovation = torch.linspace(
        -0.75, 0.55, edges, dtype=torch.float32)
    terms = RaggedSourceTerms(
        source="FY",
        query_index=query_index,
        profile_id=torch.arange(edges, dtype=torch.long),
        altitude_km=torch.full((edges,), 300.0, dtype=torch.float32),
        obs_anomalies=precast,
        innovation=innovation,
        localized_precision=precision,
        diag_r_precision=precision,
    )

    covariance64 = torch.zeros(
        queries, members, members, dtype=torch.float64).index_add(
            0, query_index,
            torch.einsum(
                "en,e,em->enm", precast, precision.double(), precast))
    rhs64 = torch.zeros(queries, members, dtype=torch.float64).index_add(
        0, query_index,
        precast * (precision.double() * innovation.double()).unsqueeze(-1))
    actual = aggregate_ragged_source_terms(
        terms, n_queries=queries, query_anomalies=query_anomalies)
    assert torch.equal(actual.covariance, covariance64.float())
    assert torch.equal(actual.rhs, rhs64.float())

    postcast_terms = RaggedSourceTerms(
        source=terms.source,
        query_index=terms.query_index,
        profile_id=terms.profile_id,
        altitude_km=terms.altitude_km,
        obs_anomalies=precast.float(),
        innovation=terms.innovation,
        localized_precision=terms.localized_precision,
        diag_r_precision=terms.diag_r_precision,
    )
    postcast = aggregate_ragged_source_terms(
        postcast_terms, n_queries=queries,
        query_anomalies=query_anomalies)
    assert not torch.equal(postcast.covariance, actual.covariance)


def test_empty_sources_produce_zero_modes_and_predictive_nis():
    query_anomalies = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    sources = {
        source: empty_ragged_source_terms(source, 2, dtype=DTYPE)
        for source in ("FY", "COSMIC")
    }
    suite = run_p0b_counterfactual_suite(query_anomalies, sources)
    for value in (
            suite.baseline.m00, suite.baseline.m10, suite.baseline.m01,
            suite.baseline.m11, suite.predictive_nis["FY"],
            suite.predictive_nis["COSMIC"], suite.predictive_nis["joint"]):
        assert torch.equal(value, torch.zeros(2, dtype=DTYPE))
    for dof in suite.predictive_nis_valid_dof.values():
        assert torch.equal(dof, torch.zeros(2, dtype=torch.long))
