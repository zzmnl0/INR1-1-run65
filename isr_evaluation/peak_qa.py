"""Shared F2-peak search and quality-control contract for ISR and GIRO.

The module deliberately keeps peak *selection* separate from peak *validity*.
Every field is searched independently on the same 200--500 km coarse/fine
grid, but a finite argmax is not automatically a scientifically usable hmF2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


PEAK_STATUSES = (
    'valid',
    'lower_censored',
    'upper_censored',
    'boundary_sensitive',
    'insufficient_levels',
    'insufficient_bracketing',
    'gap_at_peak',
    'flat_peak',
    'ambiguous_multipeak',
    'nonfinite',
)


@dataclass(frozen=True)
class PeakSearchContract:
    """Versioned, model-independent F2 peak-search contract."""

    lower_km: float = 200.0
    upper_km: float = 500.0
    coarse_step_km: float = 10.0
    fine_step_km: float = 1.0
    fine_half_window_km: float = 10.0
    min_finite_levels: int = 5
    max_local_gap_km: float = 20.0
    flank_support_km: float = 30.0
    prominence_dex: float = 0.03
    secondary_separation_km: float = 30.0
    near_tie_dex: float = 0.03
    boundary_margin_km: float = 10.0
    semantics: str = 'f2_peak_qc_coarse10_fine1_no_extrapolation_v2'

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_PEAK_CONTRACT = PeakSearchContract()


@dataclass(frozen=True)
class PeakResult:
    """Peak value plus the evidence required to use it in a metric."""

    nmf2_log10: float = np.nan
    hmf2_km: float = np.nan
    status: str = 'nonfinite'
    censoring: str = 'none'
    nmf2_valid: bool = False
    hmf2_valid: bool = False
    prominence_dex: float = np.nan
    secondary_peak_delta_dex: float = np.nan
    n_finite_levels: int = 0
    support_lower_km: float = np.nan
    support_upper_km: float = np.nan
    max_local_gap_km: float = np.nan

    def as_dict(self) -> dict:
        return asdict(self)


def _empty(status: str, *, n_finite_levels: int = 0,
           support_lower_km: float = np.nan,
           support_upper_km: float = np.nan,
           max_local_gap_km: float = np.nan,
           censoring: str = 'none') -> PeakResult:
    return PeakResult(
        status=status,
        censoring=censoring,
        n_finite_levels=int(n_finite_levels),
        support_lower_km=float(support_lower_km),
        support_upper_km=float(support_upper_km),
        max_local_gap_km=float(max_local_gap_km),
    )


def _coalesce_sorted(altitude: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable-sort profile samples and average exact duplicate heights."""
    order = np.argsort(altitude, kind='stable')
    altitude, values = altitude[order], values[order]
    unique, inverse = np.unique(altitude, return_inverse=True)
    if len(unique) == len(altitude):
        return altitude, values
    summed = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique, summed / counts


def _interpolate_no_extrapolation(altitude: np.ndarray, values: np.ndarray,
                                  targets: np.ndarray, max_gap_km: float) -> np.ndarray:
    """Linearly interpolate only inside observed brackets no wider than ``max_gap``."""
    output = np.full(np.asarray(targets).shape, np.nan, dtype=np.float64)
    if altitude.size == 0:
        return output
    targets = np.asarray(targets, dtype=np.float64)
    right = np.searchsorted(altitude, targets, side='left')
    exact = ((right < altitude.size)
             & np.isclose(altitude[np.minimum(right, altitude.size - 1)], targets,
                          rtol=0.0, atol=1e-7))
    output[exact] = values[right[exact]]
    between = (~exact) & (right > 0) & (right < altitude.size)
    if not np.any(between):
        return output
    left_index = right[between] - 1
    right_index = right[between]
    span = altitude[right_index] - altitude[left_index]
    allowed = span <= float(max_gap_km) + 1e-9
    selected = np.flatnonzero(between)[allowed]
    if selected.size:
        li = right[selected] - 1
        ri = right[selected]
        fraction = (targets[selected] - altitude[li]) / (altitude[ri] - altitude[li])
        output[selected] = values[li] + fraction * (values[ri] - values[li])
    return output


def _max_gap_in_window(altitude: np.ndarray, lower: float, upper: float) -> float:
    if altitude.size < 2:
        return np.nan
    left, right = altitude[:-1], altitude[1:]
    intersects = (right >= lower) & (left <= upper)
    if not np.any(intersects):
        return np.nan
    return float(np.max(right[intersects] - left[intersects]))


def _near_tied_secondary(fine_altitude: np.ndarray, fine_values: np.ndarray,
                          peak_altitude: float, peak_value: float,
                          contract: PeakSearchContract) -> float:
    finite = np.isfinite(fine_values)
    candidates: list[float] = []
    for index in range(1, len(fine_values) - 1):
        if not (finite[index - 1] and finite[index] and finite[index + 1]):
            continue
        if (fine_values[index] >= fine_values[index - 1]
                and fine_values[index] >= fine_values[index + 1]
                and abs(fine_altitude[index] - peak_altitude)
                >= contract.secondary_separation_km - 1e-8):
            candidates.append(float(fine_values[index]))
    if not candidates:
        return np.nan
    return float(peak_value - max(candidates))


def search_peak_profile(altitude_km: Iterable[float], log10_density: Iterable[float],
                        contract: PeakSearchContract = DEFAULT_PEAK_CONTRACT) -> PeakResult:
    """Search one F2 profile without interpolating through gaps or outside support.

    ``flat_peak`` and ``ambiguous_multipeak`` retain an NmF2 value because its
    amplitude is still meaningful, while hmF2 is intentionally excluded from the
    primary hmF2 metric.  Censored and bracketing failures retain no primary peak.
    """
    altitude = np.asarray(altitude_km, dtype=np.float64).reshape(-1)
    values = np.asarray(log10_density, dtype=np.float64).reshape(-1)
    if altitude.size != values.size:
        raise ValueError('altitude_km and log10_density must have equal length')
    finite = np.isfinite(altitude) & np.isfinite(values)
    if not np.any(finite):
        return _empty('nonfinite')
    altitude, values = _coalesce_sorted(altitude[finite], values[finite])
    in_domain = ((altitude >= contract.lower_km)
                 & (altitude <= contract.upper_km))
    altitude, values = altitude[in_domain], values[in_domain]
    n_levels = len(altitude)
    if n_levels == 0:
        return _empty('nonfinite')
    support_lower, support_upper = float(altitude[0]), float(altitude[-1])
    max_gap = _max_gap_in_window(altitude, contract.lower_km, contract.upper_km)
    if n_levels < contract.min_finite_levels:
        return _empty('insufficient_levels', n_finite_levels=n_levels,
                      support_lower_km=support_lower, support_upper_km=support_upper,
                      max_local_gap_km=max_gap)

    coarse_altitude = np.arange(
        contract.lower_km, contract.upper_km + 0.5 * contract.coarse_step_km,
        contract.coarse_step_km, dtype=np.float64)
    coarse_values = _interpolate_no_extrapolation(
        altitude, values, coarse_altitude, contract.max_local_gap_km)
    if not np.any(np.isfinite(coarse_values)):
        return _empty('insufficient_bracketing', n_finite_levels=n_levels,
                      support_lower_km=support_lower, support_upper_km=support_upper,
                      max_local_gap_km=max_gap)
    coarse_peak = coarse_altitude[int(np.nanargmax(coarse_values))]
    fine_altitude = np.arange(
        max(contract.lower_km, coarse_peak - contract.fine_half_window_km),
        min(contract.upper_km, coarse_peak + contract.fine_half_window_km)
        + 0.5 * contract.fine_step_km,
        contract.fine_step_km, dtype=np.float64)
    fine_values = _interpolate_no_extrapolation(
        altitude, values, fine_altitude, contract.max_local_gap_km)
    if not np.any(np.isfinite(fine_values)):
        return _empty('insufficient_bracketing', n_finite_levels=n_levels,
                      support_lower_km=support_lower, support_upper_km=support_upper,
                      max_local_gap_km=max_gap)
    peak_index = int(np.nanargmax(fine_values))
    peak_altitude = float(fine_altitude[peak_index])
    peak_value = float(fine_values[peak_index])

    # A field that reaches the domain end cannot identify the exterior side.
    if peak_altitude <= contract.lower_km + 1e-8 or peak_altitude <= support_lower + 1e-8:
        return PeakResult(peak_value, peak_altitude, 'lower_censored', 'lower',
                          False, False, n_finite_levels=n_levels,
                          support_lower_km=support_lower,
                          support_upper_km=support_upper,
                          max_local_gap_km=max_gap)
    if peak_altitude >= contract.upper_km - 1e-8 or peak_altitude >= support_upper - 1e-8:
        return PeakResult(peak_value, peak_altitude, 'upper_censored', 'upper',
                          False, False, n_finite_levels=n_levels,
                          support_lower_km=support_lower,
                          support_upper_km=support_upper,
                          max_local_gap_km=max_gap)
    if (peak_altitude <= contract.lower_km + contract.boundary_margin_km
            or peak_altitude >= contract.upper_km - contract.boundary_margin_km):
        return PeakResult(peak_value, peak_altitude, 'boundary_sensitive', 'none',
                          False, False, n_finite_levels=n_levels,
                          support_lower_km=support_lower,
                          support_upper_km=support_upper,
                          max_local_gap_km=max_gap)

    flank_targets = np.asarray([
        peak_altitude - contract.flank_support_km,
        peak_altitude + contract.flank_support_km], dtype=np.float64)
    flank_values = _interpolate_no_extrapolation(
        altitude, values, flank_targets, contract.max_local_gap_km)
    if not np.all(np.isfinite(flank_values)):
        if not np.isfinite(flank_values[0]) and flank_targets[0] <= support_lower + 1e-8:
            status, censoring = 'lower_censored', 'lower'
        elif not np.isfinite(flank_values[1]) and flank_targets[1] >= support_upper - 1e-8:
            status, censoring = 'upper_censored', 'upper'
        else:
            status, censoring = 'insufficient_bracketing', 'none'
        return PeakResult(peak_value, peak_altitude, status, censoring, False, False,
                          n_finite_levels=n_levels, support_lower_km=support_lower,
                          support_upper_km=support_upper, max_local_gap_km=max_gap)

    local_gap = _max_gap_in_window(
        altitude, peak_altitude - contract.flank_support_km,
        peak_altitude + contract.flank_support_km)
    if local_gap > contract.max_local_gap_km + 1e-8:
        return PeakResult(peak_value, peak_altitude, 'gap_at_peak', 'none',
                          False, False, n_finite_levels=n_levels,
                          support_lower_km=support_lower,
                          support_upper_km=support_upper,
                          max_local_gap_km=local_gap)

    prominence = float(peak_value - max(flank_values))
    # A competing F2 peak can be tens of kilometres away from the fine window;
    # inspect the whole supported coarse grid rather than only ±10 km.
    secondary_delta = _near_tied_secondary(
        coarse_altitude, coarse_values, peak_altitude, peak_value, contract)
    if np.isfinite(secondary_delta) and secondary_delta <= contract.near_tie_dex + 1e-12:
        return PeakResult(peak_value, peak_altitude, 'ambiguous_multipeak', 'none',
                          True, False, prominence, secondary_delta, n_levels,
                          support_lower, support_upper, local_gap)
    if prominence < contract.prominence_dex - 1e-12:
        return PeakResult(peak_value, peak_altitude, 'flat_peak', 'none',
                          True, False, prominence, secondary_delta, n_levels,
                          support_lower, support_upper, local_gap)
    return PeakResult(peak_value, peak_altitude, 'valid', 'none', True, True,
                      prominence, secondary_delta, n_levels, support_lower,
                      support_upper, local_gap)


def search_peak_grid(log10_grid: np.ndarray, altitude_km: Iterable[float],
                     contract: PeakSearchContract = DEFAULT_PEAK_CONTRACT) -> list[PeakResult]:
    """Apply :func:`search_peak_profile` independently to every grid column."""
    values = np.asarray(log10_grid, dtype=np.float64)
    altitude = np.asarray(altitude_km, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != altitude.size:
        raise ValueError('grid altitude dimension does not match altitude_km')
    return [search_peak_profile(altitude, values[:, index], contract)
            for index in range(values.shape[1])]


def results_to_arrays(results: Iterable[PeakResult]) -> dict[str, np.ndarray]:
    """Serialize result fields as deterministic arrays suitable for an NPZ cache."""
    values = list(results)
    return {
        'nmf2_log10': np.asarray([item.nmf2_log10 for item in values], dtype=np.float32),
        'hmf2_km': np.asarray([item.hmf2_km for item in values], dtype=np.float32),
        'status': np.asarray([item.status for item in values], dtype='U32'),
        'censoring': np.asarray([item.censoring for item in values], dtype='U16'),
        'nmf2_valid': np.asarray([item.nmf2_valid for item in values], dtype=bool),
        'hmf2_valid': np.asarray([item.hmf2_valid for item in values], dtype=bool),
        'prominence_dex': np.asarray([item.prominence_dex for item in values], dtype=np.float32),
        'secondary_peak_delta_dex': np.asarray(
            [item.secondary_peak_delta_dex for item in values], dtype=np.float32),
        'n_finite_levels': np.asarray([item.n_finite_levels for item in values], dtype=np.int16),
        'support_lower_km': np.asarray([item.support_lower_km for item in values], dtype=np.float32),
        'support_upper_km': np.asarray([item.support_upper_km for item in values], dtype=np.float32),
        'max_local_gap_km': np.asarray([item.max_local_gap_km for item in values], dtype=np.float32),
    }
