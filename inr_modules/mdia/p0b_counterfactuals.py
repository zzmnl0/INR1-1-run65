"""Pure P0-B ETKF counterfactual mathematics.

This module deliberately has no model, loader, checkpoint, or filesystem
dependencies.  It consumes ragged observation-edge terms that have already
been produced by a frozen forward pass and recomputes source-isolated and
joint ensemble-space analyses.

Two different innovation diagnostics are kept explicit:

``diag_r_standardized_innovation_sq``
    The per-edge, *unlocalized* diagonal-R quantity ``innovation**2 / R``.

``predictive_nis_low_rank``
    The query-level predictive NIS ``d.T @ S^-1 @ d`` with
    ``S = R + Y @ Y.T / (N - 1)``, evaluated without constructing dense S.

Neither quantity uses the localized precision employed by the assimilation
solve.  This separation prevents a localization-weighted energy from being
misreported as NIS.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch


SOURCES = ("FY", "COSMIC")


@dataclass(frozen=True)
class AltitudeDeletionBand:
    """A preregistered half-open altitude band, except the final closed band."""

    name: str
    lower_km: float
    upper_km: float
    include_upper: bool = False

    def deletion_mask(self, altitude_km: torch.Tensor) -> torch.Tensor:
        upper = (altitude_km <= self.upper_km
                 if self.include_upper else altitude_km < self.upper_km)
        return (altitude_km >= self.lower_km) & upper


PREREGISTERED_ALTITUDE_DELETION_BANDS: Tuple[AltitudeDeletionBand, ...] = (
    AltitudeDeletionBand("drop_200_250", 200.0, 250.0),
    AltitudeDeletionBand("drop_250_300", 250.0, 300.0),
    AltitudeDeletionBand("drop_300_400", 300.0, 400.0),
    AltitudeDeletionBand("drop_400_500", 400.0, 500.0,
                         include_upper=True),
)


@dataclass(frozen=True)
class RaggedSourceTerms:
    """Ragged observation-edge terms for one source.

    All edge tensors have leading dimension ``E``.  ``obs_anomalies`` has
    shape ``[E, N]`` and contains observation-space ensemble anomalies.  For
    strict v14 ragged replay it must be the pre-cast float64 value reconstructed
    from the saved observation basis and latent anomalies; converting the
    already rounded diagnostic float32 anomaly back to float64 is insufficient.
    ``localized_precision`` is the exact nonnegative precision used by the
    ETKF solve.  ``diag_r_precision`` is the unlocalized diagonal ``1 / R``
    used only by innovation diagnostics.
    """

    source: str
    query_index: torch.Tensor
    profile_id: torch.Tensor
    altitude_km: torch.Tensor
    obs_anomalies: torch.Tensor
    innovation: torch.Tensor
    localized_precision: torch.Tensor
    diag_r_precision: torch.Tensor

    @property
    def edge_count(self) -> int:
        return int(self.query_index.numel())

    @property
    def member_count(self) -> int:
        return int(self.obs_anomalies.shape[1])

    def subset(self, keep: torch.Tensor) -> "RaggedSourceTerms":
        """Return an order-preserving edge subset."""
        if keep.ndim != 1 or keep.shape[0] != self.edge_count:
            raise ValueError("edge subset mask must have shape [E]")
        if keep.dtype != torch.bool:
            raise TypeError("edge subset mask must be boolean")
        return RaggedSourceTerms(
            source=self.source,
            query_index=self.query_index[keep],
            profile_id=self.profile_id[keep],
            altitude_km=self.altitude_km[keep],
            obs_anomalies=self.obs_anomalies[keep],
            innovation=self.innovation[keep],
            localized_precision=self.localized_precision[keep],
            diag_r_precision=self.diag_r_precision[keep],
        )


@dataclass(frozen=True)
class AggregatedSourceTerms:
    """Per-query ensemble covariance and right-hand-side terms."""

    covariance: torch.Tensor
    rhs: torch.Tensor
    localized_precision_sum: torch.Tensor


@dataclass(frozen=True)
class ModeSolution:
    """Source-isolated and joint ETKF density increments."""

    m00: torch.Tensor
    m10: torch.Tensor
    m01: torch.Tensor
    m11: torch.Tensor
    joint_fy_contribution: torch.Tensor
    joint_cosmic_contribution: torch.Tensor
    joint_closure_error: torch.Tensor


@dataclass(frozen=True)
class EdgeGainCoefficients:
    """Per-edge coefficients multiplying innovation in ETKF increments."""

    isolated_system_gain_coefficient: torch.Tensor
    joint_system_gain_coefficient: torch.Tensor


@dataclass(frozen=True)
class ProfileDuplication:
    """One-source profile duplication and its deterministic selection."""

    terms: RaggedSourceTerms
    selected_profile_id: torch.Tensor
    selected: torch.Tensor


@dataclass(frozen=True)
class CounterfactualSuite:
    """All preregistered P0-B mathematical counterfactuals."""

    baseline: ModeSolution
    altitude_deletions: Dict[str, ModeSolution]
    profile_duplications: Dict[str, ModeSolution]
    selected_profile_ids: Dict[str, torch.Tensor]
    selected_profile_masks: Dict[str, torch.Tensor]
    edge_gain_coefficients: Dict[str, EdgeGainCoefficients]
    diag_r_standardized_innovation_sq: Dict[str, torch.Tensor]
    predictive_nis: Dict[str, torch.Tensor]
    predictive_nis_valid_dof: Dict[str, torch.Tensor]


def empty_ragged_source_terms(source: str, n_members: int, *,
                              device=None,
                              dtype=torch.float64) -> RaggedSourceTerms:
    """Construct a typed empty source payload for queries with no edges."""
    if source not in SOURCES:
        raise ValueError(f"unsupported source: {source}")
    if n_members < 2:
        raise ValueError("ETKF ensemble size N must be at least two")
    return RaggedSourceTerms(
        source=source,
        query_index=torch.empty(0, dtype=torch.long, device=device),
        profile_id=torch.empty(0, dtype=torch.long, device=device),
        altitude_km=torch.empty(0, dtype=dtype, device=device),
        obs_anomalies=torch.empty(
            0, n_members, dtype=dtype, device=device),
        innovation=torch.empty(0, dtype=dtype, device=device),
        localized_precision=torch.empty(0, dtype=dtype, device=device),
        diag_r_precision=torch.empty(0, dtype=dtype, device=device),
    )


def _validate_terms(terms: RaggedSourceTerms, *, n_queries: int,
                    n_members: int, device: torch.device) -> None:
    if terms.source not in SOURCES:
        raise ValueError(f"unsupported source: {terms.source}")
    edge_count = terms.edge_count
    one_dimensional = {
        "query_index": terms.query_index,
        "profile_id": terms.profile_id,
        "altitude_km": terms.altitude_km,
        "innovation": terms.innovation,
        "localized_precision": terms.localized_precision,
        "diag_r_precision": terms.diag_r_precision,
    }
    for name, tensor in one_dimensional.items():
        if tensor.ndim != 1 or tensor.shape[0] != edge_count:
            raise ValueError(f"{terms.source} {name} must have shape [E]")
        if tensor.device != device:
            raise ValueError(f"{terms.source} {name} is on the wrong device")
    if terms.query_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_index must have an integer dtype")
    if terms.profile_id.dtype not in (torch.int32, torch.int64):
        raise TypeError("profile_id must have an integer dtype")
    if terms.obs_anomalies.shape != (edge_count, n_members):
        raise ValueError(
            f"{terms.source} obs_anomalies must have shape [E, N]")
    if terms.obs_anomalies.device != device:
        raise ValueError(f"{terms.source} obs_anomalies is on the wrong device")
    if edge_count:
        if int(terms.query_index.min()) < 0:
            raise ValueError("query_index must be nonnegative")
        if int(terms.query_index.max()) >= n_queries:
            raise ValueError("query_index exceeds the query count")
    floating = (
        terms.altitude_km,
        terms.obs_anomalies,
        terms.innovation,
        terms.localized_precision,
        terms.diag_r_precision,
    )
    if any(not torch.isfinite(tensor).all() for tensor in floating):
        raise ValueError(f"{terms.source} edge terms must be finite")
    if (terms.localized_precision < 0).any():
        raise ValueError("localized_precision must be nonnegative")
    if (terms.diag_r_precision < 0).any():
        raise ValueError("diag_r_precision must be nonnegative")


def _validate_inputs(query_anomalies: torch.Tensor,
                     sources: Mapping[str, RaggedSourceTerms]) -> None:
    if query_anomalies.ndim != 2:
        raise ValueError("query_anomalies must have shape [Q, N]")
    if query_anomalies.shape[1] < 2:
        raise ValueError("ETKF ensemble size N must be at least two")
    if not query_anomalies.is_floating_point():
        raise TypeError("query_anomalies must be floating point")
    if not torch.isfinite(query_anomalies).all():
        raise ValueError("query_anomalies must be finite")
    if set(sources) != set(SOURCES):
        raise ValueError("sources must contain exactly FY and COSMIC")
    n_queries, n_members = query_anomalies.shape
    for source in SOURCES:
        if sources[source].source != source:
            raise ValueError(f"source key/payload mismatch for {source}")
        _validate_terms(
            sources[source], n_queries=n_queries, n_members=n_members,
            device=query_anomalies.device)


def aggregate_ragged_source_terms(
        terms: RaggedSourceTerms, *, n_queries: int,
        query_anomalies: torch.Tensor) -> AggregatedSourceTerms:
    """Aggregate ragged edges into covariance/RHS terms by query."""
    if query_anomalies.ndim != 2 or query_anomalies.shape[0] != n_queries:
        raise ValueError("query_anomalies and n_queries disagree")
    n_members = query_anomalies.shape[1]
    _validate_terms(
        terms, n_queries=n_queries, n_members=n_members,
        device=query_anomalies.device)
    dtype = query_anomalies.dtype
    accum_dtype = (torch.float64 if dtype in (
        torch.float16, torch.bfloat16, torch.float32) else dtype)
    query_index = terms.query_index.long()
    anomalies = terms.obs_anomalies.to(dtype=accum_dtype)
    precision = terms.localized_precision.to(dtype=accum_dtype)
    innovation = terms.innovation.to(dtype=accum_dtype)
    covariance = torch.zeros(
        n_queries, n_members, n_members, dtype=accum_dtype,
        device=query_anomalies.device)
    rhs = torch.zeros(
        n_queries, n_members, dtype=accum_dtype,
        device=query_anomalies.device)
    precision_sum = torch.zeros(
        n_queries, dtype=accum_dtype, device=query_anomalies.device)
    if terms.edge_count:
        covariance = covariance.index_add(
            0, query_index,
            torch.einsum("en,e,em->enm", anomalies, precision, anomalies))
        rhs = rhs.index_add(
            0, query_index,
            anomalies * (precision * innovation).unsqueeze(-1))
        precision_sum = precision_sum.index_add(
            0, query_index, precision)
    # Match the production ragged path: float64 index-add accumulation is
    # completed before the ensemble-space system is cast back and solved in
    # the query dtype.
    return AggregatedSourceTerms(
        covariance.to(dtype=dtype),
        rhs.to(dtype=dtype),
        precision_sum.to(dtype=dtype),
    )


def _solve(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    chol = torch.linalg.cholesky(system)
    return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)


def solve_etkf_counterfactual_modes(
        query_anomalies: torch.Tensor,
        sources: Mapping[str, RaggedSourceTerms]) -> ModeSolution:
    """Recompute M00/M10/M01/M11 from frozen ragged edge terms.

    M10 and M01 are source-isolated analyses with their own systems.  The two
    joint-source contributions are solved against the *same* M11 system.
    M11 is constructed as their sum, making the reported joint closure exact.
    """
    _validate_inputs(query_anomalies, sources)
    n_queries, n_members = query_anomalies.shape
    aggregated = {
        source: aggregate_ragged_source_terms(
            sources[source], n_queries=n_queries,
            query_anomalies=query_anomalies)
        for source in SOURCES
    }
    base = (n_members - 1) * torch.eye(
        n_members, dtype=query_anomalies.dtype,
        device=query_anomalies.device).expand(n_queries, -1, -1)
    fy_system = base + aggregated["FY"].covariance
    cosmic_system = base + aggregated["COSMIC"].covariance
    joint_system = (base + aggregated["FY"].covariance
                    + aggregated["COSMIC"].covariance)

    fy_isolated_weights = _solve(fy_system, aggregated["FY"].rhs)
    cosmic_isolated_weights = _solve(
        cosmic_system, aggregated["COSMIC"].rhs)
    m10 = torch.einsum("qn,qn->q", query_anomalies,
                       fy_isolated_weights)
    m01 = torch.einsum("qn,qn->q", query_anomalies,
                       cosmic_isolated_weights)

    fy_joint_weights = _solve(joint_system, aggregated["FY"].rhs)
    cosmic_joint_weights = _solve(
        joint_system, aggregated["COSMIC"].rhs)
    fy_contribution = torch.einsum(
        "qn,qn->q", query_anomalies, fy_joint_weights)
    cosmic_contribution = torch.einsum(
        "qn,qn->q", query_anomalies, cosmic_joint_weights)
    m11 = fy_contribution + cosmic_contribution
    closure_error = m11 - (fy_contribution + cosmic_contribution)
    return ModeSolution(
        m00=query_anomalies.new_zeros(n_queries),
        m10=m10,
        m01=m01,
        m11=m11,
        joint_fy_contribution=fy_contribution,
        joint_cosmic_contribution=cosmic_contribution,
        joint_closure_error=closure_error,
    )


def recompute_edge_gain_coefficients(
        query_anomalies: torch.Tensor,
        sources: Mapping[str, RaggedSourceTerms],
        ) -> Dict[str, EdgeGainCoefficients]:
    """Recompute isolated- and joint-system gain coefficients per edge.

    For edge ``i`` belonging to query ``q``, the coefficient is

    ``p_i * x_q.T @ A_q^-1 @ y_i``.

    Multiplying it by the edge innovation and summing by query/source closes
    to the corresponding isolated increment or joint-source contribution.
    """
    _validate_inputs(query_anomalies, sources)
    n_queries, n_members = query_anomalies.shape
    aggregated = {
        source: aggregate_ragged_source_terms(
            sources[source], n_queries=n_queries,
            query_anomalies=query_anomalies)
        for source in SOURCES
    }
    base = (n_members - 1) * torch.eye(
        n_members, dtype=query_anomalies.dtype,
        device=query_anomalies.device).expand(n_queries, -1, -1)
    joint_system = (base + aggregated["FY"].covariance
                    + aggregated["COSMIC"].covariance)
    joint_chol = torch.linalg.cholesky(joint_system)
    result = {}
    for source in SOURCES:
        isolated_system = base + aggregated[source].covariance
        isolated_chol = torch.linalg.cholesky(isolated_system)
        terms = sources[source]
        query_index = terms.query_index.long()
        anomalies = terms.obs_anomalies.to(dtype=query_anomalies.dtype)
        precision = terms.localized_precision.to(dtype=query_anomalies.dtype)
        if terms.edge_count:
            # Preserve the v14 production operation order exactly: form p_i y_i,
            # solve each edge against its query Cholesky factor, then dot with
            # x_q.  The algebraically equivalent adjoint solve is not bitwise
            # identical in float32.
            weighted = anomalies * precision.unsqueeze(-1)
            isolated_solved = torch.cholesky_solve(
                weighted.unsqueeze(-1),
                isolated_chol[query_index],
            ).squeeze(-1)
            joint_solved = torch.cholesky_solve(
                weighted.unsqueeze(-1),
                joint_chol[query_index],
            ).squeeze(-1)
            isolated_coefficient = torch.einsum(
                "en,en->e", query_anomalies[query_index], isolated_solved)
            joint_coefficient = torch.einsum(
                "en,en->e", query_anomalies[query_index], joint_solved)
        else:
            isolated_coefficient = query_anomalies.new_zeros(0)
            joint_coefficient = query_anomalies.new_zeros(0)
        result[source] = EdgeGainCoefficients(
            isolated_system_gain_coefficient=isolated_coefficient,
            joint_system_gain_coefficient=joint_coefficient,
        )
    return result


def delete_altitude_band(terms: RaggedSourceTerms,
                         band: AltitudeDeletionBand) -> RaggedSourceTerms:
    """Delete all edges in one preregistered observation-altitude band."""
    return terms.subset(~band.deletion_mask(terms.altitude_km))


def solve_altitude_deletion_counterfactuals(
        query_anomalies: torch.Tensor,
        sources: Mapping[str, RaggedSourceTerms]) -> Dict[str, ModeSolution]:
    """Solve the four preregistered observation-height deletion variants."""
    _validate_inputs(query_anomalies, sources)
    return {
        band.name: solve_etkf_counterfactual_modes(
            query_anomalies,
            {source: delete_altitude_band(sources[source], band)
             for source in SOURCES})
        for band in PREREGISTERED_ALTITUDE_DELETION_BANDS
    }


def duplicate_max_precision_profile(
        terms: RaggedSourceTerms, *, n_queries: int) -> ProfileDuplication:
    """Duplicate one maximum-total-precision profile per query and source.

    A tie in summed localized precision is resolved by the smallest numeric
    ``profile_id``.  Every edge belonging to the selected profile is appended
    once in its original order.
    """
    selected_profile_id = torch.full(
        (n_queries,), -1, dtype=terms.profile_id.dtype,
        device=terms.profile_id.device)
    selected = torch.zeros(
        n_queries, dtype=torch.bool, device=terms.profile_id.device)
    for query in range(n_queries):
        query_mask = terms.query_index == query
        if not query_mask.any():
            continue
        profile_ids, inverse = torch.unique(
            terms.profile_id[query_mask], sorted=True, return_inverse=True)
        # Selection is a diagnostic identity decision, so accumulate in float64
        # even when the production precision is float32.
        totals = torch.zeros(
            profile_ids.numel(), dtype=torch.float64,
            device=terms.localized_precision.device)
        totals = totals.index_add(
            0, inverse.long(), terms.localized_precision[query_mask].double())
        maximum = totals.max()
        if maximum <= 0:
            continue
        # profile_ids is sorted, so the first exact maximum is the minimum ID.
        selected_index = torch.nonzero(
            totals == maximum, as_tuple=False)[0, 0]
        selected_profile_id[query] = profile_ids[selected_index]
        selected[query] = True

    if not terms.edge_count:
        return ProfileDuplication(terms, selected_profile_id, selected)
    duplicate_mask = (
        selected[terms.query_index.long()]
        & (terms.profile_id
           == selected_profile_id[terms.query_index.long()]))
    duplicate_indices = torch.nonzero(
        duplicate_mask, as_tuple=False).squeeze(-1)
    all_indices = torch.cat((
        torch.arange(
            terms.edge_count, device=terms.query_index.device,
            dtype=torch.long),
        duplicate_indices,
    ))
    duplicated = RaggedSourceTerms(
        source=terms.source,
        query_index=terms.query_index[all_indices],
        profile_id=terms.profile_id[all_indices],
        altitude_km=terms.altitude_km[all_indices],
        obs_anomalies=terms.obs_anomalies[all_indices],
        innovation=terms.innovation[all_indices],
        localized_precision=terms.localized_precision[all_indices],
        diag_r_precision=terms.diag_r_precision[all_indices],
    )
    return ProfileDuplication(
        duplicated, selected_profile_id, selected)


def solve_profile_duplication_counterfactuals(
        query_anomalies: torch.Tensor,
        sources: Mapping[str, RaggedSourceTerms],
        duplications: Mapping[str, ProfileDuplication] = None,
        ) -> Tuple[Dict[str, ModeSolution], Dict[str, ProfileDuplication]]:
    """Solve FY-only, COSMIC-only, and both-source duplication variants."""
    _validate_inputs(query_anomalies, sources)
    n_queries = query_anomalies.shape[0]
    if duplications is None:
        duplications = {
            source: duplicate_max_precision_profile(
                sources[source], n_queries=n_queries)
            for source in SOURCES
        }
    elif set(duplications) != set(SOURCES):
        raise ValueError("duplications must contain exactly FY and COSMIC")
    variants = {
        "duplicate_FY": {
            "FY": duplications["FY"].terms,
            "COSMIC": sources["COSMIC"],
        },
        "duplicate_COSMIC": {
            "FY": sources["FY"],
            "COSMIC": duplications["COSMIC"].terms,
        },
        "duplicate_both": {
            "FY": duplications["FY"].terms,
            "COSMIC": duplications["COSMIC"].terms,
        },
    }
    return ({
        name: solve_etkf_counterfactual_modes(query_anomalies, variant)
        for name, variant in variants.items()
    }, dict(duplications))


def diag_r_standardized_innovation_sq(
        innovation: torch.Tensor,
        diag_r_precision: torch.Tensor) -> torch.Tensor:
    """Return per-edge unlocalized ``innovation**2 / R`` values."""
    if innovation.shape != diag_r_precision.shape:
        raise ValueError("innovation and diag_r_precision shapes differ")
    if not torch.isfinite(innovation).all():
        raise ValueError("innovation must be finite")
    if (not torch.isfinite(diag_r_precision).all()
            or (diag_r_precision < 0).any()):
        raise ValueError("diag_r_precision must be finite and nonnegative")
    return innovation.square() * diag_r_precision


def predictive_nis_low_rank(
        obs_anomalies: torch.Tensor, innovation: torch.Tensor,
        diag_r_precision: torch.Tensor, query_index: torch.Tensor,
        *, n_queries: int) -> torch.Tensor:
    """Compute query predictive NIS through the exact low-rank formula.

    For query observations ``d`` and anomaly matrix ``Y`` this evaluates

    ``d.T @ (R + Y @ Y.T / (N - 1))^-1 @ d``

    via Woodbury using ``(N - 1) I + Y.T @ R^-1 @ Y``.  Localization is
    intentionally absent; callers must pass the unlocalized diagonal-R
    precision, not the ETKF localized precision.  Inputs are promoted and the
    complete diagnostic is returned in float64 regardless of model dtype.
    """
    if obs_anomalies.ndim != 2:
        raise ValueError("obs_anomalies must have shape [E, N]")
    edge_count, n_members = obs_anomalies.shape
    if n_members < 2:
        raise ValueError("ETKF ensemble size N must be at least two")
    for name, tensor in (
            ("innovation", innovation),
            ("diag_r_precision", diag_r_precision),
            ("query_index", query_index)):
        if tensor.ndim != 1 or tensor.shape[0] != edge_count:
            raise ValueError(f"{name} must have shape [E]")
        if tensor.device != obs_anomalies.device:
            raise ValueError(f"{name} is on the wrong device")
    if query_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_index must have an integer dtype")
    if edge_count:
        if int(query_index.min()) < 0 or int(query_index.max()) >= n_queries:
            raise ValueError("query_index is outside [0, n_queries)")
    if (not torch.isfinite(obs_anomalies).all()
            or not torch.isfinite(innovation).all()
            or not torch.isfinite(diag_r_precision).all()):
        raise ValueError("predictive NIS inputs must be finite")
    if (diag_r_precision < 0).any():
        raise ValueError("diag_r_precision must be nonnegative")

    # Predictive NIS is a diagnostic, not the production Analysis solve.  Keep
    # the Woodbury accumulation, solve, and final subtract in float64 to avoid
    # catastrophic cancellation (which can otherwise create large negative NIS
    # values for high precision and d approximately in span(Y)).
    dtype = torch.float64
    anomalies = obs_anomalies.to(dtype=dtype)
    precision = diag_r_precision.to(dtype=dtype)
    innovations = innovation.to(dtype=dtype)
    indices = query_index.long()
    covariance = torch.zeros(
        n_queries, n_members, n_members, dtype=dtype,
        device=obs_anomalies.device)
    rhs = torch.zeros(
        n_queries, n_members, dtype=dtype, device=obs_anomalies.device)
    diagonal_energy = torch.zeros(
        n_queries, dtype=dtype, device=obs_anomalies.device)
    if edge_count:
        covariance = covariance.index_add(
            0, indices,
            torch.einsum(
                "en,e,em->enm", anomalies, precision, anomalies))
        rhs = rhs.index_add(
            0, indices,
            anomalies * (precision * innovations).unsqueeze(-1))
        diagonal_energy = diagonal_energy.index_add(
            0, indices, innovations.square() * precision)
    base = (n_members - 1) * torch.eye(
        n_members, dtype=dtype,
        device=obs_anomalies.device).expand(n_queries, -1, -1)
    solved_rhs = _solve(base + covariance, rhs)
    correction = torch.einsum("qn,qn->q", rhs, solved_rhs)
    raw = diagonal_energy - correction
    tolerance = (
        256.0 * torch.finfo(torch.float64).eps
        * (diagonal_energy.abs() + correction.abs() + 1.0))
    if not torch.isfinite(raw).all() or not torch.isfinite(tolerance).all():
        raise FloatingPointError("predictive NIS produced non-finite intermediates")
    materially_negative = raw < -tolerance
    if materially_negative.any():
        minimum = float(raw.min().item())
        worst_tolerance = float(tolerance[materially_negative].max().item())
        raise FloatingPointError(
            "predictive NIS became materially negative: "
            f"minimum={minimum:.9g}, tolerance={worst_tolerance:.9g}")
    return torch.where(raw < 0.0, torch.zeros_like(raw), raw)


def predictive_nis_valid_dof(
        diag_r_precision: torch.Tensor, query_index: torch.Tensor, *,
        n_queries: int) -> torch.Tensor:
    """Count positive unlocalized diagonal-R precision edges per query."""
    if diag_r_precision.ndim != 1:
        raise ValueError("diag_r_precision must have shape [E]")
    if query_index.ndim != 1 or query_index.shape != diag_r_precision.shape:
        raise ValueError("query_index must have shape [E]")
    if query_index.device != diag_r_precision.device:
        raise ValueError("query_index is on the wrong device")
    if query_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_index must have an integer dtype")
    if (not torch.isfinite(diag_r_precision).all()
            or (diag_r_precision < 0).any()):
        raise ValueError("diag_r_precision must be finite and nonnegative")
    if query_index.numel():
        if int(query_index.min()) < 0 or int(query_index.max()) >= n_queries:
            raise ValueError("query_index is outside [0, n_queries)")
    result = torch.zeros(
        n_queries, dtype=torch.long, device=query_index.device)
    if query_index.numel():
        result = result.index_add(
            0, query_index.long(), (diag_r_precision > 0).long())
    return result


def _source_predictive_nis(terms: RaggedSourceTerms,
                           n_queries: int) -> torch.Tensor:
    return predictive_nis_low_rank(
        terms.obs_anomalies,
        terms.innovation,
        terms.diag_r_precision,
        terms.query_index,
        n_queries=n_queries,
    )


def run_p0b_counterfactual_suite(
        query_anomalies: torch.Tensor,
        sources: Mapping[str, RaggedSourceTerms]) -> CounterfactualSuite:
    """Run all frozen P0-B mathematical variants and diagnostics."""
    _validate_inputs(query_anomalies, sources)
    n_queries = query_anomalies.shape[0]
    duplications = {
        source: duplicate_max_precision_profile(
            sources[source], n_queries=n_queries)
        for source in SOURCES
    }
    duplication_solutions, duplications = (
        solve_profile_duplication_counterfactuals(
            query_anomalies, sources, duplications))
    joint_anomalies = torch.cat(
        [sources[source].obs_anomalies for source in SOURCES], dim=0)
    joint_innovation = torch.cat(
        [sources[source].innovation for source in SOURCES], dim=0)
    joint_precision = torch.cat(
        [sources[source].diag_r_precision for source in SOURCES], dim=0)
    joint_query_index = torch.cat(
        [sources[source].query_index for source in SOURCES], dim=0)
    predictive_nis = {
        source: _source_predictive_nis(sources[source], n_queries)
        for source in SOURCES
    }
    predictive_nis["joint"] = predictive_nis_low_rank(
        joint_anomalies, joint_innovation, joint_precision,
        joint_query_index, n_queries=n_queries)
    predictive_dof = {
        source: predictive_nis_valid_dof(
            sources[source].diag_r_precision,
            sources[source].query_index,
            n_queries=n_queries)
        for source in SOURCES
    }
    predictive_dof["joint"] = predictive_nis_valid_dof(
        joint_precision, joint_query_index, n_queries=n_queries)
    return CounterfactualSuite(
        baseline=solve_etkf_counterfactual_modes(query_anomalies, sources),
        altitude_deletions=solve_altitude_deletion_counterfactuals(
            query_anomalies, sources),
        profile_duplications=duplication_solutions,
        selected_profile_ids={
            source: duplications[source].selected_profile_id
            for source in SOURCES
        },
        selected_profile_masks={
            source: duplications[source].selected
            for source in SOURCES
        },
        edge_gain_coefficients=recompute_edge_gain_coefficients(
            query_anomalies, sources),
        diag_r_standardized_innovation_sq={
            source: diag_r_standardized_innovation_sq(
                sources[source].innovation,
                sources[source].diag_r_precision)
            for source in SOURCES
        },
        predictive_nis=predictive_nis,
        predictive_nis_valid_dof=predictive_dof,
    )


__all__ = [
    "AltitudeDeletionBand",
    "PREREGISTERED_ALTITUDE_DELETION_BANDS",
    "RaggedSourceTerms",
    "AggregatedSourceTerms",
    "ModeSolution",
    "EdgeGainCoefficients",
    "ProfileDuplication",
    "CounterfactualSuite",
    "empty_ragged_source_terms",
    "aggregate_ragged_source_terms",
    "solve_etkf_counterfactual_modes",
    "recompute_edge_gain_coefficients",
    "delete_altitude_band",
    "solve_altitude_deletion_counterfactuals",
    "duplicate_max_precision_profile",
    "solve_profile_duplication_counterfactuals",
    "diag_r_standardized_innovation_sq",
    "predictive_nis_low_rank",
    "predictive_nis_valid_dof",
    "run_p0b_counterfactual_suite",
]
