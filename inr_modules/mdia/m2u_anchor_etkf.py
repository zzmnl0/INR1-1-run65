"""Core tensor operations for the M2-U shared-anchor response layer.

The module is intentionally independent of the data indexes and the INR model.
It keeps the mathematical contract testable with synthetic tensors:

* anchor-local ETKF systems are accumulated from positive precisions;
* Sparsemax gives a continuous, possibly one-hot anchor partition;
* a flat-top physical cover fades the final increment to the M00 background;
* M10, M01 and M11 use separate anchor catalogs.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def spherical_distance_km(coords_a: torch.Tensor,
                          coords_b: torch.Tensor) -> torch.Tensor:
    """Return pairwise great-circle distance for ``[..., 2]`` lat/lon tensors."""
    if coords_a.shape[-1] < 2 or coords_b.shape[-1] < 2:
        raise ValueError('coordinates need latitude and longitude')
    lat_a = torch.deg2rad(coords_a[..., 0])
    lon_a = torch.deg2rad(coords_a[..., 1])
    lat_b = torch.deg2rad(coords_b[..., 0])
    lon_b = torch.deg2rad(coords_b[..., 1])
    dlat = lat_b - lat_a
    dlon = torch.remainder(lon_b - lon_a + math.pi, 2.0 * math.pi) - math.pi
    hav = (
        torch.sin(dlat * 0.5).square()
        + torch.cos(lat_a) * torch.cos(lat_b) * torch.sin(dlon * 0.5).square()
    ).clamp(0.0, 1.0)
    return 2.0 * 6371.0 * torch.asin(torch.sqrt(hav))


def _flat_top(value: torch.Tensor) -> torch.Tensor:
    """C1 flat-top taper: one in the core, zero at the support boundary."""
    value = value.clamp_min(0.0)
    transition = ((value - 0.5) * 2.0).clamp(0.0, 1.0)
    taper = 1.0 - transition.square() * (3.0 - 2.0 * transition)
    return torch.where(value <= 0.5, torch.ones_like(value), taper).masked_fill(
        value >= 1.0, 0.0)


def flat_top_cover(query_coords: torch.Tensor,
                   anchor_coords: torch.Tensor,
                   space_scale_km: float = 1800.0,
                   time_scale_h: float = 1.5) -> torch.Tensor:
    """Return ``[Q, A]`` physical cover weights.

    Coordinates use ``[lat, lon, altitude, relative_hour]``.  Altitude is
    deliberately absent from this guard; vertical relation is carried by the
    endpoint basis and the error-state representation.
    """
    if query_coords.ndim != 2 or anchor_coords.ndim != 2:
        raise ValueError('query_coords and anchor_coords must be rank-2')
    if query_coords.shape[-1] < 4 or anchor_coords.shape[-1] < 4:
        raise ValueError('coordinates need lat, lon, altitude and time')
    distance = spherical_distance_km(
        query_coords[:, None, :2], anchor_coords[None, :, :2])
    time_distance = torch.abs(
        query_coords[:, None, 3] - anchor_coords[None, :, 3])
    return _flat_top(distance / float(space_scale_km)) * _flat_top(
        time_distance / float(time_scale_h))


def state_distance_squared(query_eta: torch.Tensor,
                           anchor_eta: torch.Tensor,
                           ell_e: float,
                           ell_a: float,
                           valid_query: Optional[torch.Tensor] = None,
                           valid_anchor: Optional[torch.Tensor] = None,
                           eps: float = 1e-8) -> torch.Tensor:
    """Return normalized ``[Q, A]`` direction-plus-amplitude distances."""
    if query_eta.ndim != 2 or anchor_eta.ndim != 2:
        raise ValueError('eta tensors must be rank-2')
    if query_eta.shape[-1] < 2 or anchor_eta.shape[-1] < 2:
        raise ValueError('eta must contain direction and amplitude')
    if ell_e <= 0.0 or ell_a <= 0.0:
        raise ValueError('state length scales must be positive')
    rank = query_eta.shape[-1] - 1
    if anchor_eta.shape[-1] != rank + 1:
        raise ValueError('query and anchor eta dimensions differ')
    direction = query_eta[:, None, :rank] - anchor_eta[None, :, :rank]
    amplitude = query_eta[:, None, rank] - anchor_eta[None, :, rank]
    distance = direction.square().sum(dim=-1) / float(ell_e) ** 2
    distance = distance + amplitude.square() / float(ell_a) ** 2
    if valid_query is not None:
        distance = distance.masked_fill(~valid_query[:, None], 0.0)
    if valid_anchor is not None:
        distance = distance.masked_fill(~valid_anchor[None, :], 0.0)
    return distance.clamp_min(eps)


def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the probability simplex.

    This implementation uses only PyTorch tensor operations and returns exact
    zeros/ones in the one-hot core up to floating-point arithmetic.
    """
    if scores.ndim == 0:
        raise ValueError('scores must have at least one dimension')
    dim = dim if dim >= 0 else scores.ndim + dim
    if not 0 <= dim < scores.ndim:
        raise ValueError('invalid sparsemax dimension')
    shifted = scores - scores.max(dim=dim, keepdim=True).values
    sorted_scores, _ = torch.sort(shifted, dim=dim, descending=True)
    count = sorted_scores.shape[dim]
    view = [1] * scores.ndim
    view[dim] = count
    rank = torch.arange(1, count + 1, device=scores.device,
                        dtype=scores.dtype).view(view)
    cumulative = sorted_scores.cumsum(dim=dim)
    support = 1.0 + rank * sorted_scores > cumulative
    support_count = support.to(scores.dtype).sum(dim=dim, keepdim=True).clamp_min(1.0)
    tau = (cumulative.gather(dim, support_count.to(torch.long) - 1)
           - 1.0) / support_count
    return torch.clamp(shifted - tau, min=0.0)


def calibrate_temperature(score_gaps: torch.Tensor,
                          default: float = 1.0,
                          eps: float = 1e-3) -> float:
    """Freeze a positive train-only Sparsemax temperature from top-gap samples."""
    values = score_gaps.detach().reshape(-1).float()
    values = values[torch.isfinite(values) & (values > 0.0)]
    if values.numel() == 0:
        value = float(default)
    else:
        value = float(torch.median(values).item())
    if not math.isfinite(value) or value <= eps:
        value = float(default)
    if not math.isfinite(value) or value <= eps:
        raise ValueError('Sparsemax temperature calibration is non-positive')
    return value


def anchor_scores(query_coords: torch.Tensor,
                  anchor_coords: torch.Tensor,
                  query_eta: Optional[torch.Tensor],
                  anchor_eta: Optional[torch.Tensor],
                  ell_e: float,
                  ell_a: float,
                  tau: float,
                  space_scale_km: float = 1800.0,
                  time_scale_h: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Sparsemax scores and flat-top cover for one mode."""
    cover = flat_top_cover(query_coords, anchor_coords, space_scale_km,
                           time_scale_h)
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError('anchor temperature must be finite and positive')
    if query_eta is None or anchor_eta is None:
        distance = torch.zeros_like(cover)
    else:
        distance = state_distance_squared(query_eta, anchor_eta, ell_e, ell_a)
    scores = torch.log(cover.clamp_min(torch.finfo(cover.dtype).tiny))
    scores = (scores - 0.5 * distance) / float(tau)
    return scores.masked_fill(cover <= 0.0, -torch.inf), cover


def anchor_mixing_weights(query_coords: torch.Tensor,
                          anchor_coords: torch.Tensor,
                          query_eta: Optional[torch.Tensor],
                          anchor_eta: Optional[torch.Tensor],
                          ell_e: float,
                          ell_a: float,
                          tau: float,
                          space_scale_km: float = 1800.0,
                          time_scale_h: float = 1.5,
                          anchor_mask: Optional[torch.Tensor] = None
                          ) -> Dict[str, torch.Tensor]:
    """Compute ``alpha`` and final background/anchor partition weights."""
    scores, cover = anchor_scores(
        query_coords, anchor_coords, query_eta, anchor_eta, ell_e, ell_a,
        tau, space_scale_km, time_scale_h)
    positive = cover > 0.0
    if anchor_mask is not None:
        if anchor_mask.ndim != 1 or anchor_mask.shape[0] != cover.shape[1]:
            raise ValueError('anchor_mask must match anchor dimension')
        positive = positive & anchor_mask[None, :].bool()
    alpha = torch.zeros_like(cover)
    for row in range(scores.shape[0]):
        if positive[row].any():
            alpha[row, positive[row]] = sparsemax(scores[row, positive[row]])
    beta = alpha * cover
    background = (1.0 - beta.sum(dim=-1)).clamp(0.0, 1.0)
    return {
        'scores': scores,
        'cover': cover,
        'alpha': alpha,
        'beta': beta,
        'background_weight': background,
    }


def accumulate_anchor_terms(obs_anomalies: torch.Tensor,
                            precision: torch.Tensor,
                            innovation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate ``C`` and ``b`` for ``[A, M, N]`` observation anomalies."""
    if obs_anomalies.ndim != 3:
        raise ValueError('obs_anomalies must have shape [A, M, N]')
    if precision.shape != obs_anomalies.shape[:2]:
        raise ValueError('precision shape does not match observation anomalies')
    if innovation.shape != precision.shape:
        raise ValueError('innovation shape does not match precision')
    weighted = obs_anomalies * precision.unsqueeze(-1)
    covariance = torch.einsum('amn,amk->ank', weighted, obs_anomalies)
    rhs = torch.einsum('amn,am,am->an', obs_anomalies, precision, innovation)
    return covariance, rhs


def solve_anchor_weights(covariance: torch.Tensor,
                         rhs: torch.Tensor,
                         base_rank: int = 7) -> torch.Tensor:
    """Solve one or more anchor ETKF systems in ensemble space."""
    if covariance.ndim != 3 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError('covariance must have shape [A, N, N]')
    if rhs.shape != covariance.shape[:2]:
        raise ValueError('rhs shape does not match covariance')
    members = covariance.shape[-1]
    eye = torch.eye(members, device=covariance.device, dtype=covariance.dtype)
    system = covariance + float(base_rank) * eye.unsqueeze(0)
    chol = torch.linalg.cholesky(system)
    return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)


def solve_anchor_sources(source_terms: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                         mode: str,
                         base_rank: int = 7) -> Dict[str, torch.Tensor]:
    """Solve mode-specific anchor systems from ``(C, b)`` source terms.

    ``source_terms`` maps ``FY`` and ``COSMIC`` to ``[A,N,N]`` covariance and
    ``[A,N]`` right-hand-side tensors.  The returned weights retain the exact
    M10/M01/M11 distinction; a caller may then interpolate the anchor weights
    spatially without rebuilding a query-local system.
    """
    if mode not in ('M10', 'M01', 'M11'):
        raise ValueError('mode must be M10, M01 or M11')
    required = ('FY',) if mode == 'M10' else ('COSMIC',) if mode == 'M01' else ('FY', 'COSMIC')
    missing = [source for source in required if source not in source_terms]
    if missing:
        raise ValueError(f'missing source terms for {mode}: {missing}')
    first_cov, first_rhs = source_terms[required[0]]
    members = first_cov.shape[-1]
    eye = torch.eye(members, device=first_cov.device, dtype=first_cov.dtype)
    system = float(base_rank) * eye.unsqueeze(0) + first_cov
    if mode == 'M11':
        system = system + source_terms['COSMIC'][0]
    chol = torch.linalg.cholesky(system)
    result = {
        f'w_{required[0]}': torch.cholesky_solve(
            first_rhs.unsqueeze(-1), chol).squeeze(-1),
        'system': system,
    }
    if mode == 'M11':
        cosmic_rhs = source_terms['COSMIC'][1]
        result['w_COSMIC'] = torch.cholesky_solve(
            cosmic_rhs.unsqueeze(-1), chol).squeeze(-1)
    if mode == 'M11':
        result['w_total'] = result['w_FY'] + result['w_COSMIC']
    else:
        result['w_total'] = result[f'w_{required[0]}']
    return result


def blend_anchor_increments(query_anomalies: torch.Tensor,
                            beta: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
    """Decode shared anchor weights into ``[Q]`` analysis increments."""
    if query_anomalies.ndim != 2 or weights.ndim != 2:
        raise ValueError('query_anomalies and weights must be rank-2')
    if beta.shape != (query_anomalies.shape[0], weights.shape[0]):
        raise ValueError('beta shape does not match query/anchor dimensions')
    anchor_increment = torch.einsum('qn,an->qa', query_anomalies, weights)
    return torch.einsum('qa,qa->q', beta, anchor_increment)


__all__ = [
    'accumulate_anchor_terms',
    'anchor_mixing_weights',
    'anchor_scores',
    'blend_anchor_increments',
    'calibrate_temperature',
    'flat_top_cover',
    'solve_anchor_weights',
    'sparsemax',
    'spherical_distance_km',
    'state_distance_squared',
]
