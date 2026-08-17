"""Independent traditional static-local ETKF primitives.

This module deliberately contains no M2-W/P0 state or learned features.  The
state is a 1 degree spherical grid and the only observation operator is the
grid trilinear stencil defined below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


ALT_MIN_KM = 120.0
ALT_MAX_KM = 500.0
ALT_STEP_KM = 20.0
N_ALT = 20
N_LAT = 181
N_LON = 360
N_MEMBERS = 8
N_FACTORS = 7
SEED = 42
HORIZONTAL_LENGTH_KM = 1800.0
VERTICAL_LENGTH_KM = 100.0
TIME_BIN_HOURS = 0.5
TIME_LOCALIZATION_HOURS = 1.5
R_FY = 0.03347056359052658
R_COSMIC = 0.021563315764069557
R_BY_SOURCE = {"FY": R_FY, "COSMIC": R_COSMIC}
DEFAULT_VARIANCE = 1.0e-4
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Grid:
    """The fixed 120--500 km, 1 degree spherical grid."""

    alt_km: np.ndarray = field(default_factory=lambda: np.arange(120.0, 500.0 + 20.0, 20.0))
    lat_deg: np.ndarray = field(default_factory=lambda: np.arange(-90.0, 90.0 + 1.0, 1.0))
    lon_deg: np.ndarray = field(default_factory=lambda: np.arange(0.0, 360.0, 1.0))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (len(self.alt_km), len(self.lat_deg), len(self.lon_deg))

    @property
    def altitudes(self):
        return self.alt_km

    @property
    def latitudes(self):
        return self.lat_deg

    @property
    def longitudes(self):
        return self.lon_deg

    def validate(self) -> None:
        if self.shape != (N_ALT, N_LAT, N_LON):
            raise ValueError(f"unexpected grid shape {self.shape}")

    def stencil(self, lat_deg, lon_deg, alt_km):
        """Return sparse trilinear corners and weights for ``(lat, lon, alt)``.

        Returned indices are flattened state indices and have shape ``(Q, 8)``;
        unused corners at a pole are retained with zero weight.  Longitude is
        periodic, while latitude/altitude are validated rather than clipped.
        """
        lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
        lon = np.asarray(lon_deg, dtype=np.float64).reshape(-1)
        alt = np.asarray(alt_km, dtype=np.float64).reshape(-1)
        if not (lat.shape == lon.shape == alt.shape):
            raise ValueError("lat, lon and altitude must have equal shape")
        if not np.isfinite(np.stack([lat, lon, alt], axis=1)).all():
            raise ValueError("non-finite observation coordinate")
        if np.any((lat < -90.0) | (lat > 90.0)):
            raise ValueError("latitude outside [-90, 90]")
        if np.any((alt < ALT_MIN_KM) | (alt > ALT_MAX_KM)):
            raise ValueError("altitude outside [120, 500] km")

        # Longitude is the sole periodic coordinate.  Values exactly 360 and
        # negative values are valid equivalents; no other coordinate is clipped.
        lon = np.mod(lon, 360.0)
        a = (alt - ALT_MIN_KM) / ALT_STEP_KM
        a0 = np.floor(a).astype(np.int64)
        a0 = np.minimum(a0, N_ALT - 2)  # 500 km belongs to the final cell
        a1 = a0 + 1
        wa = a - a0
        # At 500 km the last upper endpoint is the last grid level.
        wa = np.where(alt >= ALT_MAX_KM, 1.0, wa)
        l = lat + 90.0
        l0 = np.floor(l).astype(np.int64)
        l0 = np.minimum(l0, N_LAT - 2)
        l1 = l0 + 1
        wl = l - l0
        wl = np.where(lat >= 90.0, 1.0, wl)
        o = lon
        o0 = np.floor(o).astype(np.int64) % N_LON
        o1 = (o0 + 1) % N_LON
        wo = o - np.floor(o)

        # Corner ordering: altitude, latitude, longitude binary bits.
        idx = np.empty((len(lat), 8), dtype=np.int64)
        weights = np.empty((len(lat), 8), dtype=np.float64)
        k = 0
        for dk, wk in ((0, 1.0 - wa), (1, wa)):
            for dl, wl_w in ((0, 1.0 - wl), (1, wl)):
                for do, wo_w in ((0, 1.0 - wo), (1, wo)):
                    ai = a0 + dk
                    li = np.minimum(l0 + dl, N_LAT - 1)
                    oi = (o0 + do) % N_LON
                    idx[:, k] = ((ai * N_LAT + li) * N_LON + oi)
                    weights[:, k] = wk * wl_w * wo_w
                    k += 1

        # A spherical grid has one pole, not 360 independent pole states.
        pole = np.abs(np.abs(lat) - 90.0) < 1.0e-12
        if pole.any():
            pole_rows = np.flatnonzero(pole)
            for row in pole_rows:
                li = 0 if lat[row] < 0.0 else N_LAT - 1
                lower = (a0[row] * N_LAT + li) * N_LON
                upper = (a1[row] * N_LAT + li) * N_LON
                idx[row, :] = lower
                idx[row, 1] = upper
                weights[row, :] = 0.0
                weights[row, 0] = 1.0 - wa[row]
                weights[row, 1] = wa[row]
        # Numerical round-off must never make a non-convex interpolation.
        weights /= weights.sum(axis=1, keepdims=True)
        return idx, weights

    def interpolate(self, field: np.ndarray, lat_deg, lon_deg, alt_km) -> np.ndarray:
        """Apply the same sparse H to a field ``[..., 8]`` or scalar field."""
        arr = np.asarray(field)
        if arr.shape[:3] != self.shape:
            raise ValueError(f"field must start with grid shape {self.shape}")
        idx, weights = self.stencil(lat_deg, lon_deg, alt_km)
        flat = arr.reshape((-1,) + arr.shape[3:])
        values = flat[idx]
        if arr.ndim == 3:
            return np.sum(values * weights, axis=1)
        return np.sum(values * weights[..., None], axis=1)


GRID = Grid()
GRID.validate()


def grid_stencil(lat_deg, lon_deg, alt_km):
    """Compatibility function exposing the canonical H stencil."""
    return GRID.stencil(lat_deg, lon_deg, alt_km)


def trilinear_stencil(lat_deg, lon_deg, alt_km):
    return GRID.stencil(lat_deg, lon_deg, alt_km)


def cycle_time_hours(time_hours):
    """Nearest independent 0.5 h analysis cycle (no cycling of analysis)."""
    t = np.asarray(time_hours, dtype=np.float64)
    return TIME_BIN_HOURS * np.floor(t / TIME_BIN_HOURS + 0.5)


def _helmert_coefficients() -> np.ndarray:
    # H is a 7x8 orthonormal Helmert contrast matrix.  C has zero row sums and
    # C.T@C=7I; C@C.T cannot equal 7I for an 8x7 matrix (rank obstruction).
    h = np.zeros((N_FACTORS, N_MEMBERS), dtype=np.float64)
    for k in range(1, N_MEMBERS):
        h[k - 1, :k] = 1.0 / np.sqrt(k * (k + 1.0))
        h[k - 1, k] = -k / np.sqrt(k * (k + 1.0))
    return np.sqrt(7.0) * h.T


HELMERT = _helmert_coefficients()


def _sphere_coordinates(grid: Grid = GRID) -> np.ndarray:
    lat = np.deg2rad(grid.lat_deg)[None, :, None]
    lon = np.deg2rad(grid.lon_deg)[None, None, :]
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.broadcast_to(np.sin(lat), (1, len(grid.lat_deg), len(grid.lon_deg)))
    surface = np.stack([np.broadcast_to(x, z.shape), np.broadcast_to(y, z.shape), z], axis=-1)
    return np.broadcast_to(surface, (len(grid.alt_km),) + surface.shape[1:])


def generate_static_ensemble(
    target_std,
    seed: int = SEED,
    grid: Grid = GRID,
    chunk_size: int = 32768,
) -> np.ndarray:
    """Generate deterministic 7-factor spherical RFF anomalies and 8 members.

    ``target_std`` may be a scalar, one value per height, or a full grid.  The
    output is ``float32 [20,181,360,8]`` and is built in chunks so no dense B/H
    is ever materialized.
    """
    std = np.asarray(target_std, dtype=np.float64)
    if std.ndim == 0:
        std = np.full(grid.shape, float(std))
    elif std.shape == (N_ALT,):
        std = np.broadcast_to(std[:, None, None], grid.shape)
    elif std.shape != grid.shape:
        raise ValueError(f"target_std must be scalar, ({N_ALT},), or {grid.shape}")
    std = np.maximum(np.where(np.isfinite(std), std, 0.0), 0.0)

    rng = np.random.RandomState(seed)
    frequencies = rng.normal(size=(N_FACTORS, 64, 4)).astype(np.float64)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(N_FACTORS, 64)).astype(np.float64)
    xyz = _sphere_coordinates(grid).reshape(-1, 3)
    alt = np.repeat(grid.alt_km, N_LAT * N_LON).astype(np.float64)
    z = np.empty((len(xyz), 4), dtype=np.float64)
    z[:, :3] = EARTH_RADIUS_KM * xyz / HORIZONTAL_LENGTH_KM
    z[:, 3] = alt / VERTICAL_LENGTH_KM
    out = np.empty((len(xyz), N_MEMBERS), dtype=np.float32)
    flat_std = std.reshape(-1)
    for start in range(0, len(xyz), int(chunk_size)):
        stop = min(start + int(chunk_size), len(xyz))
        zz = z[start:stop]
        factors = np.empty((stop - start, N_FACTORS), dtype=np.float64)
        for f in range(N_FACTORS):
            # RFF average over 64 fixed frequencies for this factor.
            factors[:, f] = np.sqrt(2.0 / 64.0) * np.cos(
                zz @ frequencies[f].T + phases[f]
            ).sum(axis=1)
        raw = factors @ HELMERT.T
        raw -= raw.mean(axis=1, keepdims=True)
        sample_std = raw.std(axis=1, ddof=1)
        scale = np.divide(
            flat_std[start:stop], sample_std,
            out=np.zeros_like(sample_std), where=sample_std > 1.0e-12)
        out[start:stop] = (raw * scale[:, None]).astype(np.float32)
    out = out.reshape(grid.shape + (N_MEMBERS,))
    # Explicit pole broadcast also protects against future feature changes.
    out[:, 0, :, :] = out[:, 0, :1, :]
    out[:, -1, :, :] = out[:, -1, :1, :]
    out -= out.mean(axis=-1, keepdims=True).astype(np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite static ensemble")
    return out


def make_static_ensemble(target_std, seed: int = SEED, **kwargs):
    return generate_static_ensemble(target_std, seed=seed, **kwargs)


def load_iri_proxy(checkpoint_path, device: str = "cpu"):
    """Strictly load and freeze the independent [4,128,128,128,128,1] proxy."""
    import torch
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy

    model = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    raw = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    state = raw.get("state_dict", raw.get("model_state_dict", raw)) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise ValueError("IRI checkpoint must contain a state dictionary")
    # Accept a conventional module. prefix while retaining strict loading.
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError("IRI proxy contains non-finite weights")
    model.freeze()
    return model


def iri_query(proxy, lat_deg, lon_deg, alt_km, time_hours, device: str = "cpu") -> np.ndarray:
    """Query the proxy; grid longitudes are converted to its [-180,180] domain."""
    import torch
    lat, lon, alt, tim = [np.asarray(v, dtype=np.float32).reshape(-1)
                          for v in (lat_deg, lon_deg, alt_km, time_hours)]
    if not (lat.shape == lon.shape == alt.shape == tim.shape):
        raise ValueError("IRI query arrays must have equal shape")
    if not np.isfinite(np.stack((lat, lon, alt, tim), axis=1)).all():
        raise ValueError("IRI query coordinates must be finite")
    if np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("IRI latitude outside [-90, 90]")
    if np.any((alt < ALT_MIN_KM) | (alt > ALT_MAX_KM)):
        raise ValueError("IRI altitude outside [120, 500] km")
    if np.any((tim < 0.0) | (tim > 720.0)):
        raise ValueError("IRI time outside [0, 720] hours")
    lon_proxy = ((lon + 180.0) % 360.0) - 180.0
    # Geographic longitude is undefined at a pole.  Canonicalizing it prevents
    # the neural proxy from returning different values for the same grid state.
    lon_proxy[np.isclose(np.abs(lat), 90.0, atol=1e-6, rtol=0.0)] = 0.0
    try:
        model_device = next(proxy.parameters()).device
    except StopIteration:
        try:
            model_device = next(proxy.buffers()).device
        except StopIteration:
            model_device = torch.device(device)
    x = torch.from_numpy(np.stack([lat, lon_proxy, alt, tim], axis=1)).to(
        model_device)
    with torch.no_grad():
        y = proxy(x).detach().cpu().numpy().reshape(-1)
    if not np.isfinite(y).all():
        raise FloatingPointError("IRI proxy returned non-finite values")
    return y


def iri_at_grid_stencil(proxy, lat_deg, lon_deg, alt_km, time_hours,
                        device: str = "cpu", chunk_size: int = 65536) -> np.ndarray:
    """Evaluate cycle-time IRI at unique H corners, then interpolate."""
    idx, weights = GRID.stencil(lat_deg, lon_deg, alt_km)
    flat = idx.reshape(-1)
    tc = np.broadcast_to(np.asarray(time_hours, dtype=np.float64).reshape(-1),
                         np.asarray(lat_deg).reshape(-1).shape)
    if not np.isfinite(tc).all() or np.any((tc < 0.0) | (tc > 720.0)):
        raise ValueError("IRI time outside [0, 720] hours")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    tc = cycle_time_hours(tc)
    time_code = np.rint(np.repeat(tc, 8) / TIME_BIN_HOURS).astype(np.int64)
    pairs = np.column_stack((time_code, flat))
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    unique_time = unique[:, 0].astype(np.float64) * TIME_BIN_HOURS
    unique_flat = unique[:, 1].astype(np.int64)
    ai = unique_flat // (N_LAT * N_LON)
    rem = unique_flat % (N_LAT * N_LON)
    li = rem // N_LON
    oi = rem % N_LON
    unique_values = np.empty(len(unique), dtype=np.float64)
    for start in range(0, len(unique), int(chunk_size)):
        stop = min(start + int(chunk_size), len(unique))
        unique_values[start:stop] = iri_query(
            proxy, GRID.lat_deg[li[start:stop]], GRID.lon_deg[oi[start:stop]],
            GRID.alt_km[ai[start:stop]], unique_time[start:stop], device=device)
    vals = unique_values[inverse].reshape(-1, 8)
    return np.sum(vals * weights, axis=1)


@dataclass
class ETKFResult:
    analysis: np.ndarray
    analysis_anomalies: np.ndarray
    weights: np.ndarray
    transform: np.ndarray

    @property
    def xa(self):
        return self.analysis

    @property
    def X_a(self):
        return self.analysis_anomalies

    def __iter__(self):
        yield self.analysis
        yield self.analysis_anomalies
        yield self.weights
        yield self.transform


@dataclass
class RaggedETKFAudit:
    status: np.ndarray
    positive_precision_count: np.ndarray
    precision_sum: np.ndarray
    minimum_eigenvalue: np.ndarray
    condition_number: np.ndarray
    max_absolute_increment: np.ndarray
    failure_count: int
    failure_queries: np.ndarray

    @property
    def min_eigenvalue(self):
        return self.minimum_eigenvalue

    @property
    def max_abs_increment(self):
        return self.max_absolute_increment


@dataclass
class RaggedETKFResult:
    analysis: np.ndarray
    analysis_anomalies: np.ndarray
    weights: np.ndarray
    transform: np.ndarray
    audit: RaggedETKFAudit

    @property
    def xa(self):
        return self.analysis

    @property
    def X_a(self):
        return self.analysis_anomalies


def _etkf_factor(A, b):
    """Factor one finite 8x8 ETKF system without numerical fallback."""
    A = (np.asarray(A, dtype=np.float64) +
         np.asarray(A, dtype=np.float64).T) * 0.5
    b = np.asarray(b, dtype=np.float64)
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise FloatingPointError("non-finite ETKF system")
    try:
        chol = np.linalg.cholesky(A)
        w = np.linalg.solve(chol.T, np.linalg.solve(chol, b))
        eigval, eigvec = np.linalg.eigh(A)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError("ETKF decomposition failed") from exc
    if (not np.isfinite(w).all() or not np.isfinite(eigval).all()
            or np.any(eigval <= 0.0)):
        raise FloatingPointError("invalid ETKF decomposition")
    T = (eigvec * np.sqrt(7.0 / eigval)) @ eigvec.T
    if not np.isfinite(T).all():
        raise FloatingPointError("non-finite ETKF transform")
    return w, T, eigval


def ragged_etkf_update(xb, query_anomalies, query_index, obs_anomalies,
                       innovation, precision, strict=True) -> RaggedETKFResult:
    """Solve independent float64 ETKF systems from ragged observation edges."""
    xb = np.asarray(xb, dtype=np.float64)
    X = np.asarray(query_anomalies, dtype=np.float64)
    index = np.asarray(query_index)
    Y = np.asarray(obs_anomalies, dtype=np.float64)
    d = np.asarray(innovation, dtype=np.float64)
    p = np.asarray(precision, dtype=np.float64)
    if xb.ndim != 1:
        raise ValueError("xb must have shape [Q]")
    if X.shape != (len(xb), N_MEMBERS):
        raise ValueError("query_anomalies must have shape [Q, 8]")
    if index.ndim != 1:
        raise ValueError("query_index must have shape [E]")
    if index.size and not np.issubdtype(index.dtype, np.integer):
        raise TypeError("query_index must have an integer dtype")
    edge_count = len(index)
    if Y.shape != (edge_count, N_MEMBERS):
        raise ValueError("obs_anomalies must have shape [E, 8]")
    if d.shape != (edge_count,) or p.shape != (edge_count,):
        raise ValueError("innovation and precision must have shape [E]")
    if edge_count and (index.min() < 0 or index.max() >= len(xb)):
        raise ValueError("query_index is outside [0, Q)")
    if (not np.isfinite(xb).all() or not np.isfinite(X).all()
            or not np.isfinite(Y).all() or not np.isfinite(d).all()
            or not np.isfinite(p).all()):
        raise ValueError("ragged ETKF inputs must be finite")
    if np.any(p < 0.0):
        raise ValueError("precision must be nonnegative")

    query_count = len(xb)
    covariance = np.zeros((query_count, N_MEMBERS, N_MEMBERS), dtype=np.float64)
    rhs = np.zeros((query_count, N_MEMBERS), dtype=np.float64)
    positive_count = np.zeros(query_count, dtype=np.int64)
    precision_sum = np.zeros(query_count, dtype=np.float64)
    positive = p > 0.0
    if positive.any():
        positive_index = index[positive].astype(np.int64, copy=False)
        positive_y = Y[positive]
        positive_d = d[positive]
        positive_p = p[positive]
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(
                covariance, positive_index,
                positive_p[:, None, None] *
                positive_y[:, :, None] * positive_y[:, None, :])
            np.add.at(rhs, positive_index,
                      positive_y * (positive_p * positive_d)[:, None])
            np.add.at(precision_sum, positive_index, positive_p)
        np.add.at(positive_count, positive_index, 1)

    analysis = xb.copy()
    analysis_anomalies = X.copy()
    weights = np.zeros((query_count, N_MEMBERS), dtype=np.float64)
    transform = np.broadcast_to(
        np.eye(N_MEMBERS, dtype=np.float64),
        (query_count, N_MEMBERS, N_MEMBERS)).copy()
    status = np.full(query_count, "no_observation", dtype="<U17")
    minimum_eigenvalue = np.full(query_count, 7.0, dtype=np.float64)
    condition_number = np.ones(query_count, dtype=np.float64)
    max_absolute_increment = np.zeros(query_count, dtype=np.float64)
    identity = np.eye(N_MEMBERS, dtype=np.float64)

    for query in np.flatnonzero(positive_count):
        try:
            w, T, eigval = _etkf_factor(7.0 * identity + covariance[query],
                                        rhs[query])
            with np.errstate(over="ignore", invalid="ignore"):
                increment = X[query] @ w
                Xa = X[query] @ T
                Xa -= Xa.mean()
            if not np.isfinite(increment) or not np.isfinite(Xa).all():
                raise FloatingPointError("non-finite ETKF analysis")
            analysis[query] += increment
            analysis_anomalies[query] = Xa
            weights[query] = w
            transform[query] = T
            status[query] = "ok"
            minimum_eigenvalue[query] = eigval[0]
            condition_number[query] = eigval[-1] / eigval[0]
            max_absolute_increment[query] = abs(increment)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            if strict:
                raise FloatingPointError(
                    f"ETKF numerical failure for query {query}: {exc}") from exc
            status[query] = "numerical_failure"
            analysis[query] = np.nan
            analysis_anomalies[query] = np.nan
            weights[query] = np.nan
            transform[query] = np.nan
            minimum_eigenvalue[query] = np.nan
            condition_number[query] = np.nan
            max_absolute_increment[query] = np.nan

    failure_queries = np.flatnonzero(status == "numerical_failure")
    audit = RaggedETKFAudit(
        status=status,
        positive_precision_count=positive_count,
        precision_sum=precision_sum,
        minimum_eigenvalue=minimum_eigenvalue,
        condition_number=condition_number,
        max_absolute_increment=max_absolute_increment,
        failure_count=int(len(failure_queries)),
        failure_queries=failure_queries,
    )
    return RaggedETKFResult(analysis, analysis_anomalies, weights, transform,
                            audit)


def local_etkf_update(xb, X, Y, innovation, precision) -> ETKFResult:
    """Deterministic square-root ensemble-space ETKF for one local query.

    ``Y`` is the observation anomaly matrix, not a learned observation
    payload.  Use :func:`analyze_query` when only a static grid is available.
    """
    return etkf_update(xb, X, Y, innovation, precision)


def etkf_update(xb, X, Y, innovation, precision) -> ETKFResult:
    """ETKF with ``X`` state anomalies, ``Y`` observation anomalies."""
    xb = np.asarray(xb, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")
    if X.shape[1] != N_MEMBERS and X.shape[0] == N_MEMBERS:
        X = X.T
    if Y.shape[1] != N_MEMBERS and Y.shape[0] == N_MEMBERS:
        Y = Y.T
    if X.shape[1] != N_MEMBERS or Y.shape[1] != N_MEMBERS:
        raise ValueError("ETKF requires eight ensemble members")
    d = np.asarray(innovation, dtype=np.float64).reshape(-1)
    p = np.asarray(precision, dtype=np.float64).reshape(-1)
    if Y.shape[0] != len(d) or len(d) != len(p) or X.shape[0] != len(xb):
        raise ValueError("ETKF shapes are inconsistent")
    if (not np.isfinite(xb).all() or not np.isfinite(X).all()
            or not np.isfinite(Y).all() or not np.isfinite(d).all()
            or not np.isfinite(p).all()):
        raise ValueError("ETKF inputs must be finite")
    if np.any(p < 0.0):
        raise ValueError("precision must be nonnegative")
    positive = p > 0.0
    if not positive.any():
        return ETKFResult(xb.copy(), X.copy(), np.zeros(N_MEMBERS), np.eye(N_MEMBERS))
    y = Y[positive]
    dd = d[positive]
    pp = p[positive]
    with np.errstate(over="ignore", invalid="ignore"):
        A = (7.0 * np.eye(N_MEMBERS, dtype=np.float64)
             + y.T @ (pp[:, None] * y))
        b = y.T @ (pp * dd)
    w, T, _ = _etkf_factor(A, b)
    with np.errstate(over="ignore", invalid="ignore"):
        xa = xb + X @ w
        Xa = X @ T
        Xa -= Xa.mean(axis=1, keepdims=True)
    if not np.isfinite(xa).all() or not np.isfinite(Xa).all():
        raise FloatingPointError("non-finite ETKF analysis")
    return ETKFResult(xa, Xa, w, T)


def local_etkf(*args, **kwargs):
    return etkf_update(*args, **kwargs)


def analyze_query(xb, Xq, obs_coords, observations, obs_background, precision) -> ETKFResult:
    """Apply H to a static ensemble and run one local ETKF query."""
    Xq = np.asarray(Xq, dtype=np.float64)
    if Xq.ndim != 2 or Xq.shape[1] != N_MEMBERS:
        raise ValueError("Xq must be [state, 8]")
    coords = np.asarray(obs_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError("obs_coords must contain lat, lon and altitude")
    idx, weights = GRID.stencil(coords[:, 0], coords[:, 1], coords[:, 2])
    Y = np.sum(Xq[idx] * weights[..., None], axis=1)
    d = np.asarray(observations, dtype=np.float64) - np.asarray(obs_background, dtype=np.float64)
    return etkf_update(np.asarray(xb), Xq, Y, d, precision)


# Small compatibility surface used by the independent prepare/smoke CLI.
def baseline_config(config):
    result = dict(config)
    result.update(alt_range=(120.0, 500.0),
                  observation_alt_range=(200.0, 500.0),
                  physical_localization_space_km=1800.0,
                  physical_localization_time_hours=1.5,
                  fy_nb_n_alt=8, cosmic_nb_n_alt=8,
                  neighbor_directory_semantics="token_exact_positive_support_v1")
    return result


def cycle_time(hours):
    return cycle_time_hours(hours)


def stencil(coords):
    coords = np.asarray(coords)
    return GRID.stencil(coords[:, 0], coords[:, 1], coords[:, 2])


def apply_h(field, coords):
    coords = np.asarray(coords)
    return GRID.interpolate(field, coords[:, 0], coords[:, 1], coords[:, 2])


def load_iri(path, device="cpu"):
    return load_iri_proxy(path, device)


def iri_at(proxy, coords, device="cpu"):
    coords = np.asarray(coords, dtype=np.float64)
    return iri_at_grid_stencil(
        proxy, coords[:, 0], coords[:, 1], coords[:, 2],
        cycle_time_hours(coords[:, 3]), device=device)


def prepare_anomalies(path, sigma, seed=SEED):
    members = generate_static_ensemble(sigma, seed=seed)
    np.save(Path(path), members, allow_pickle=False)
    return members


def payload_terms(payload, source, anomaly_field, proxy, query_cycles,
                  device="cpu"):
    if source not in R_BY_SOURCE or any("representativeness" in key.lower()
                                        for key in payload):
        raise ValueError("invalid traditional ETKF source/payload")
    required = {"coords", "value", "valid_mask", "localization_weight",
                "query_index"}
    if not required.issubset(payload):
        raise ValueError(f"payload missing {sorted(required-set(payload))}")
    coords = np.asarray(payload["coords"], dtype=np.float64)
    if len(coords) and np.any((coords[:, 2] < 200.0) | (coords[:, 2] > 500.0)):
        raise ValueError("observations must be within 200-500 km")
    index = np.asarray(payload["query_index"], dtype=np.int64)
    values = np.asarray(payload["value"], dtype=np.float64)
    loc = np.asarray(payload["localization_weight"], dtype=np.float64)
    valid = np.asarray(payload["valid_mask"], dtype=bool)
    keep = valid & np.isfinite(values) & np.isfinite(loc) & (loc > 0)
    coords, index, values, loc = (array[keep] for array in
                                  (coords, index, values, loc))
    if len(index):
        anomalies = apply_h(anomaly_field, coords)
        background_coords = coords.copy()
        background_coords[:, 3] = np.asarray(query_cycles)[index]
        innovation = values - iri_at(proxy, background_coords, device=device)
    else:
        anomalies, innovation = np.empty((0, 8)), np.empty(0)
    return index, anomalies, innovation, loc / R_BY_SOURCE[source]


def solve_ragged_etkf(background, query_anomalies, query_index,
                      obs_anomalies, innovation, precision, strict=False):
    """Dictionary compatibility wrapper for the audited ragged ETKF solve.

    With ``strict=False`` a numerical factorization failure is represented by
    NaNs and counted explicitly; it is never replaced by the IRI background.
    ``fallback_mask`` therefore identifies only exact no-observation queries.
    """
    background = np.asarray(background, dtype=np.float64)
    result = ragged_etkf_update(
        background, query_anomalies, query_index, obs_anomalies, innovation,
        precision, strict=strict)
    fallback_mask = result.audit.status == "no_observation"
    return {
        "analysis": result.analysis,
        "increment": result.analysis - background,
        "transform": result.transform,
        "analysis_anomalies": result.analysis_anomalies,
        "weights": result.weights,
        "spread": np.sqrt(
            np.sum(result.analysis_anomalies ** 2, axis=1) / 7.0),
        "precision_sum": result.audit.precision_sum,
        "fallback_mask": fallback_mask,
        "factorization_failures": result.audit.failure_count,
        "audit": result.audit,
    }


def solve(background, query_anomalies, terms):
    """Solve ragged observation terms while preserving the legacy dictionary."""
    background = np.asarray(background, dtype=np.float64).reshape(-1)
    query_anomalies = np.asarray(query_anomalies, dtype=np.float64)
    indices, anomalies, innovations, precisions = [], [], [], []
    for index, obs_anomalies, innovation, precision in terms:
        indices.append(np.asarray(index))
        anomalies.append(np.asarray(obs_anomalies))
        innovations.append(np.asarray(innovation))
        precisions.append(np.asarray(precision))
    query_index = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    obs_anomalies = (np.concatenate(anomalies, axis=0) if anomalies
                     else np.empty((0, N_MEMBERS), dtype=np.float64))
    innovation = (np.concatenate(innovations) if innovations
                  else np.empty(0, dtype=np.float64))
    precision = (np.concatenate(precisions) if precisions
                 else np.empty(0, dtype=np.float64))
    result = ragged_etkf_update(
        background, query_anomalies, query_index, obs_anomalies, innovation,
        precision)
    return {"analysis": result.analysis,
            "increment": result.analysis-background,
            "transform": result.transform,
            "analysis_anomalies": result.analysis_anomalies,
            "spread": np.sqrt(np.sum(result.analysis_anomalies**2, axis=1)/7.0),
            "precision_sum": result.audit.precision_sum,
            "audit": result.audit}


__all__ = [
    "Grid", "GRID", "ETKFResult", "RaggedETKFAudit", "RaggedETKFResult",
    "N_ALT", "N_LAT", "N_LON", "N_MEMBERS",
    "ALT_MIN_KM", "ALT_MAX_KM", "ALT_STEP_KM", "HORIZONTAL_LENGTH_KM",
    "VERTICAL_LENGTH_KM", "TIME_BIN_HOURS", "TIME_LOCALIZATION_HOURS",
    "R_FY", "R_COSMIC",
    "cycle_time_hours", "grid_stencil", "trilinear_stencil",
    "generate_static_ensemble", "make_static_ensemble", "load_iri_proxy",
    "iri_query", "iri_at_grid_stencil", "etkf_update", "ragged_etkf_update",
    "local_etkf_update",
    "local_etkf", "analyze_query", "HELMERT", "R_BY_SOURCE",
    "baseline_config", "cycle_time", "stencil", "apply_h", "load_iri",
    "iri_at", "prepare_anomalies", "payload_terms", "solve_ragged_etkf",
    "solve",
]
