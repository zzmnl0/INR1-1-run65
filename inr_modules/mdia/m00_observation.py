"""Train-only observation-error calibration for the M00 Background loss.

The module is intentionally independent of the data loaders.  A calibration
table is fitted from Train rows, frozen, and then consumed by both training
and evaluation.  Development and Locked-test rows are never accepted by the
fit helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np


M00_OBSERVATION_SEMANTICS = "m00_train_only_observation_error_v1"
M00_BIAS_R_SEMANTICS = "m00_train_only_bias_r_v1"
M00_REP_LOC_SEMANTICS = "m00_train_only_representativeness_localization_v1"
M00_OBSERVATION_V2_SEMANTICS = "m00_train_only_observation_error_v2"
M00_QD_RELIABILITY_SEMANTICS = "a1s_m00_qd_residual_empirical_reliability_v1"
M00_3D_LOCALIZATION_SEMANTICS = "m00_3d_localization_v2"
AACGM_MAGNETIC_STRATUM_SEMANTICS = "aacgm_abs_sin_lat_025_050_v2"
QD_MAGNETIC_STRATUM_SEMANTICS = "qd_abs_sin_lat_025_050_v1"
DIPOLE_MODEL_MAGNETIC_SEMANTICS = "dipole_abs_sin_i_v1"
LOW_DIM_ASSIMILATION_CONTRACT = {
    "low_dim_assimilation_semantics": "shared_latent_etkf_v1",
    "basis_dim": 64,
    "enkf_n_members": 8,
    "enkf_anomaly_parameterization": "orthogonal_factor",
    "analysis_state_semantics": "legacy_feature_increment",
    "density_basis_semantics": "endpoint_context_symmetric",
    "mode_mapping": "M10:FY,M01:COSMIC,M11:FY+COSMIC",
}
_STOPPED_ADAPTER_SEMANTICS = {
    "maglat2_smooth_v1",
    "dual_maglat_smooth_v1",
    "condition_adapter_only_v1",
    "condition_adapter_joint_fresh_v1",
}


def validate_low_dim_assimilation_config(config: Mapping[str, Any]) -> None:
    """Fail closed if the shared 64-D ETKF contract drifts."""
    for key, expected in LOW_DIM_ASSIMILATION_CONTRACT.items():
        actual = config.get(key, expected)
        if key in {"basis_dim", "enkf_n_members"}:
            try:
                actual = int(actual)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer") from exc
        if actual != expected:
            raise ValueError(
                f"low-dimensional assimilation contract mismatch for {key}: "
                f"{actual!r} != {expected!r}")


def validate_frozen_m00_config(config: Mapping[str, Any], *,
                               eval_only: bool = False) -> None:
    """Reject the permanently stopped adapter/backbone-unfreeze routes."""
    if bool(config.get("shared_backbone_unfreeze", False)):
        raise ValueError("shared backbone unfreeze is forbidden by frozen M00 contract")
    if eval_only or bool(config.get("eval_only", False)):
        return
    adapter = str(config.get("background_condition_adapter", "disabled"))
    semantics = str(config.get("background_condition_adapter_semantics", "disabled"))
    scope = str(config.get("background_training_scope", ""))
    if (adapter in _STOPPED_ADAPTER_SEMANTICS
            or semantics in _STOPPED_ADAPTER_SEMANTICS
            or scope in _STOPPED_ADAPTER_SEMANTICS):
        raise ValueError("A1-S7/A1-S8 adapter route is permanently stopped")


def load_magnetic_sidecar(path: str | os.PathLike[str], *, expected_rows: int | None = None,
                          source: str | None = None, strict_aacgm: bool = False,
                          strict_qd: bool = False,
                          expected_profile_ids: Any | None = None,
                          expected_coords: Any | None = None,
                          expected_row_indices: Any | None = None) -> dict[str, np.ndarray]:
    """Load a strict row-aligned magnetic-coordinate sidecar.

    Required arrays are ``row_index``, ``profile_id`` and (strict coordinate
    mode) ``aacgm_latitude_deg`` or ``qd_latitude_deg``;
    optional ``magnetic_latitude_deg``/``magnetic_longitude_deg`` are retained.
    Missing or fallback-filled values are rejected instead of silently using
    geographic latitude.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"magnetic sidecar is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if strict_aacgm and "qd_latitude_deg" in payload.files and "aacgm_latitude_deg" not in payload.files:
            # The loader flag predates the QD contract; auto-detect the new
            # strict coordinate while preserving the public call signature.
            strict_qd = True
        required = ("row_index", "profile_id")
        if any(name not in payload for name in required):
            raise ValueError("magnetic sidecar requires row_index/profile_id")
        if strict_aacgm and "aacgm_latitude_deg" not in payload:
            raise ValueError("strict AACGM sidecar requires aacgm_latitude_deg")
        if strict_qd and "qd_latitude_deg" not in payload:
            raise ValueError("strict QD sidecar requires qd_latitude_deg")
        if strict_qd and "magnetic_sin_i" not in payload:
            raise ValueError("strict QD sidecar requires independent model magnetic_sin_i")
        if not strict_aacgm and not strict_qd and "magnetic_sin_i" not in payload and "aacgm_latitude_deg" not in payload and "qd_latitude_deg" not in payload:
            raise ValueError("magnetic sidecar requires magnetic_sin_i or aacgm_latitude_deg")
        out = {name: np.asarray(payload[name]) for name in payload.files}
    rows = np.asarray(out["row_index"], dtype=np.int64).reshape(-1)
    profiles = np.asarray(out["profile_id"], dtype=np.int64).reshape(-1)
    if "magnetic_sin_i" in out:
        sin_i = np.asarray(out["magnetic_sin_i"], dtype=np.float64).reshape(-1)
    elif "aacgm_latitude_deg" in out:
        sin_i = np.sin(np.deg2rad(np.asarray(out["aacgm_latitude_deg"], dtype=np.float64).reshape(-1)))
    elif "qd_latitude_deg" in out:
        # QD is a calibration coordinate; this fallback exists only for old
        # sidecars that did not persist the independent model dipole field.
        sin_i = np.sin(np.deg2rad(np.asarray(out["qd_latitude_deg"], dtype=np.float64).reshape(-1)))
    else:
        sin_i = np.empty(0, dtype=np.float64)
    if not (rows.size == profiles.size == sin_i.size):
        raise ValueError("magnetic sidecar arrays must have equal length")
    if expected_rows is not None and rows.size != int(expected_rows):
        raise ValueError("magnetic sidecar row count differs from source data")
    if strict_aacgm or strict_qd:
        if expected_row_indices is None:
            if not np.array_equal(rows, np.arange(rows.size, dtype=np.int64)):
                raise ValueError("strict AACGM sidecar row_index must be contiguous source order")
        else:
            authorized = np.asarray(expected_row_indices, dtype=np.int64).reshape(-1)
            if authorized.size != rows.size or not np.array_equal(rows, authorized):
                raise ValueError("strict magnetic sidecar row_index is not the authorized subset")
    if rows.size and (np.unique(rows).size != rows.size or np.any(rows < 0)):
        raise ValueError("magnetic sidecar row_index must be unique and non-negative")
    if not (np.isfinite(sin_i).all() and np.all(np.abs(sin_i) <= 1.0)):
        raise ValueError("magnetic sidecar magnetic_sin_i must be finite in [-1,1]")
    if source is not None and "source" in out:
        labels = np.asarray(out["source"]).astype(str).reshape(-1)
        if labels.size != rows.size or np.any(labels != str(source)):
            raise ValueError("magnetic sidecar source labels do not match")
    if expected_profile_ids is not None:
        expected = np.asarray(expected_profile_ids, dtype=np.int64).reshape(-1)
        if expected.size != profiles.size or not np.array_equal(expected, profiles):
            raise ValueError("magnetic sidecar profile_id is not row-aligned")
    if expected_coords is not None:
        coords = np.asarray(expected_coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] != rows.size or coords.shape[1] < 4:
            raise ValueError("expected_coords must align with sidecar rows")
        # Sidecars must carry the source coordinates used by AACGM conversion;
        # compare exact float values when present, never nearest-neighbour join.
        coord_names = ("latitude_deg", "longitude_deg", "altitude_km", "utc_hours")
        if all(name in out for name in coord_names):
            side_coords = np.column_stack([np.asarray(out[name], dtype=np.float64).reshape(-1)
                                           for name in coord_names])
            if side_coords.shape != coords[:, :4].shape or not np.array_equal(side_coords, coords[:, :4]):
                raise ValueError("magnetic sidecar coordinates are not row-aligned")
        elif strict_aacgm or strict_qd:
            raise ValueError("strict magnetic sidecar lacks source coordinate columns")
    if strict_aacgm or strict_qd:
        metadata = out.get("metadata")
        semantics = out.get("semantics")
        if semantics is not None:
            sem = str(np.asarray(semantics).reshape(-1)[0])
            allowed = {"aacgm_v2", AACGM_MAGNETIC_STRATUM_SEMANTICS}
            if strict_qd:
                allowed.add(QD_MAGNETIC_STRATUM_SEMANTICS)
            if sem not in allowed:
                raise ValueError("unsupported magnetic sidecar semantics")
            if strict_qd and sem != QD_MAGNETIC_STRATUM_SEMANTICS:
                raise ValueError("strict QD sidecar semantics mismatch")
        if "aacgm_version" not in out and "aacgmv2_version" not in out:
            # Metadata can be represented as a scalar array or JSON string.
            if metadata is None:
                raise ValueError("strict magnetic sidecar lacks converter version metadata")
    out["row_index"], out["profile_id"], out["magnetic_sin_i"] = rows, profiles, sin_i
    if "aacgm_latitude_deg" in out:
        lat = np.asarray(out["aacgm_latitude_deg"], dtype=np.float64).reshape(-1)
        if lat.size != rows.size:
            raise ValueError("AACGM latitude must align with sidecar rows")
        # QD sidecars may carry AACGM as an optional audit.  AACGM can be
        # undefined for individual rows; QD remains the strict coordinate.
        if strict_qd and not strict_aacgm:
            finite = np.isfinite(lat)
            if np.any(np.abs(lat[finite]) > 90.0):
                raise ValueError("finite AACGM audit latitude must be in [-90,90]")
        elif not np.isfinite(lat).all() or np.any(np.abs(lat) > 90.0):
            raise ValueError("AACGM latitude must align and be finite in [-90,90]")
        out["aacgm_latitude_deg"] = lat
    if "qd_latitude_deg" in out:
        qd = np.asarray(out["qd_latitude_deg"], dtype=np.float64).reshape(-1)
        if qd.size != rows.size or not np.isfinite(qd).all() or np.any(np.abs(qd) > 90.0):
            raise ValueError("QD latitude must align and be finite in [-90,90]")
        out["qd_latitude_deg"] = qd
    if strict_aacgm or strict_qd:
        _validate_strict_sidecar_metadata(out, expected_rows=rows.size)
    return out


def _validate_strict_sidecar_metadata(payload: Mapping[str, Any], *,
                                      expected_rows: int) -> None:
    """Validate provenance fields required by the strict AACGM contract."""
    def scalar(name: str, aliases: tuple[str, ...] = ()) -> Any:
        for key in (name, *aliases):
            if key in payload:
                value = np.asarray(payload[key]).reshape(-1)
                if value.size != 1:
                    raise ValueError(f"AACGM sidecar {key} must be scalar metadata")
                return value[0].item()
        return None

    semantics = scalar("semantics")
    if semantics is None or str(semantics) not in {
            "aacgm_v2", AACGM_MAGNETIC_STRATUM_SEMANTICS,
            QD_MAGNETIC_STRATUM_SEMANTICS}:
        raise ValueError("unsupported magnetic sidecar semantics")
    metadata = scalar("metadata")
    metadata_map = {}
    if metadata is not None:
        try:
            parsed = json.loads(str(metadata))
            if isinstance(parsed, Mapping):
                metadata_map = parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata_map = {}
    version = scalar("aacgm_version", ("aacgmv2_version", "converter_version"))
    if version is None:
        version = metadata_map.get("aacgm_version", metadata_map.get("converter_version"))
    if str(semantics) == QD_MAGNETIC_STRATUM_SEMANTICS:
        qd_converter = metadata_map.get("qd_converter")
        apex_version = metadata_map.get("apexpy_version")
        if str(qd_converter).lower() != "apexpy" or not str(apex_version).strip():
            raise ValueError("strict QD sidecar requires apexpy provenance")
    elif version is None or "aacgmv2" not in str(version).lower():
        raise ValueError("strict AACGM sidecar requires aacgmv2 converter version")
    reference_time = scalar("reference_time_utc", ("aacgm_reference_time_utc",))
    if reference_time is None:
        reference_time = metadata_map.get("reference_time_utc")
    if reference_time is None or not str(reference_time).strip():
        raise ValueError("strict AACGM sidecar requires reference_time_utc")
    fallback = scalar("fallback", ("fallback_used",))
    if fallback not in (None, False, 0, "", "none", "False", "false"):
        raise ValueError("strict AACGM sidecar cannot use geographic fallback")
    coordinate_name = "qd_latitude_deg" if str(semantics) == QD_MAGNETIC_STRATUM_SEMANTICS else "aacgm_latitude_deg"
    for name in ("row_index", "profile_id", coordinate_name):
        if name not in payload:
            raise ValueError(f"strict magnetic sidecar missing {name}")
        arr = np.asarray(payload[name]).reshape(-1)
        if arr.size != int(expected_rows):
            raise ValueError(f"strict magnetic sidecar {name} length mismatch")
    for name in ("latitude_deg", "longitude_deg", "altitude_km", "utc_hours"):
        if name not in payload:
            raise ValueError(f"strict AACGM sidecar missing source field {name}")
        values = np.asarray(payload[name], dtype=np.float64).reshape(-1)
        if values.size != int(expected_rows) or not np.isfinite(values).all():
            raise ValueError(f"strict AACGM sidecar source field {name} is invalid")

def _finite_mapping(values: Mapping[str, Any], name: str) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    out: dict[str, float] = {}
    for key, value in values.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value: {key}")
        out[str(key)] = value
    return out


def _lookup(table: Mapping[str, float], key: str, source: str,
            default: float) -> float:
    """Look up exact stratum first, then progressively coarser levels."""
    if not table:
        return float(default)
    key = str(key)
    candidates = [key]
    # Both ``source|...`` and the historical ``source:...`` spelling occur in
    # old audit artifacts; accepting either keeps calibration files portable.
    parts = key.replace(":", "|").split("|")
    if parts:
        for end in range(len(parts) - 1, 0, -1):
            candidates.append("|".join(parts[:end]))
        candidates.append(parts[0])
    candidates.extend((str(source), "default", "__default__"))
    for candidate in candidates:
        if candidate in table:
            return float(table[candidate])
    return float(default)


def _as_numpy(values: Any, *, dtype=float) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def observation_stratum_keys(coords: Any, source: str,
                             magnetic_sin_i: Any | None = None,
                             *, aacgm_latitude_deg: Any | None = None,
                             qd_latitude_deg: Any | None = None,
                             strict_magnetic: bool = False) -> np.ndarray:
    """Return source×height×day/night×magnetic-latitude keys for coordinates.

    Coordinates use the project convention ``[lat_deg, lon_deg, alt_km,
    utc_hours]``.  If no magnetic-inclination feature is supplied, geographic
    latitude is the deterministic fallback used by the existing M00 strata.
    """
    arr = _as_numpy(coords, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] < 4:
        raise ValueError("coords must have shape [..., 4] or larger")
    flat = arr.reshape(-1, arr.shape[-1])
    if not np.isfinite(flat[:, :4]).all():
        raise ValueError("non-finite coordinates")
    if strict_magnetic and aacgm_latitude_deg is None and qd_latitude_deg is None:
        raise ValueError("strict magnetic-latitude contract requires QD/AACGM latitude")
    coordinate = qd_latitude_deg if qd_latitude_deg is not None else aacgm_latitude_deg
    if coordinate is not None:
        magnetic_lat = _as_numpy(coordinate, dtype=float).reshape(-1)
        if (magnetic_lat.size != flat.shape[0] or not np.isfinite(magnetic_lat).all()
                or np.any(np.abs(magnetic_lat) > 90.0)):
            raise ValueError("magnetic latitude must align and be finite in [-90,90]")
        mag = np.abs(np.sin(np.deg2rad(magnetic_lat)))
    elif magnetic_sin_i is None:
        if strict_magnetic:
            raise ValueError("strict magnetic-latitude contract requires AACGM magnetic latitude")
        mag = np.abs(np.sin(np.deg2rad(flat[:, 0])))
    else:
        mag = np.abs(_as_numpy(magnetic_sin_i, dtype=float)).reshape(-1)
        if mag.size != flat.shape[0] or not np.isfinite(mag).all():
            raise ValueError("magnetic_sin_i must align with coords")
    alt = flat[:, 2]
    alt_band = np.where(alt < 200.0, "120-200",
                        np.where(alt < 300.0, "200-300",
                                 np.where(alt <= 500.0, "300-500", "other")))
    local_hour = np.mod(flat[:, 3] + flat[:, 1] / 15.0, 24.0)
    time_band = np.where((local_hour >= 6.0) & (local_hour < 18.0),
                         "day", "night")
    mag_band = np.where(mag < 0.25, "low",
                        np.where(mag < 0.5, "mid", "high"))
    return np.asarray([
        f"{source}|{a}|{t}|{m}"
        for a, t, m in zip(alt_band, time_band, mag_band)
    ], dtype=object).reshape(arr.shape[:-1])


@dataclass(frozen=True)
class ObservationCalibration:
    """Frozen finite bias/variance table and optional error-model factors."""

    bias: Mapping[str, float]
    variance: Mapping[str, float]
    representativeness: Mapping[str, float] = field(default_factory=dict)
    localization: Mapping[str, float] = field(default_factory=dict)
    semantics: str = M00_OBSERVATION_SEMANTICS
    variance_floor: float = 1e-6
    variance_cap: float = 1e6

    def __post_init__(self) -> None:
        object.__setattr__(self, "bias", _finite_mapping(self.bias, "bias"))
        object.__setattr__(self, "variance",
                           _finite_mapping(self.variance, "variance"))
        object.__setattr__(self, "representativeness",
                           _finite_mapping(self.representativeness,
                                           "representativeness"))
        object.__setattr__(self, "localization",
                           _finite_mapping(self.localization, "localization"))
        floor, cap = float(self.variance_floor), float(self.variance_cap)
        if not (math.isfinite(floor) and floor > 0 and
                math.isfinite(cap) and cap >= floor):
            raise ValueError("invalid variance floor/cap")
        object.__setattr__(self, "variance_floor", floor)
        object.__setattr__(self, "variance_cap", cap)
        if any(v <= 0 for v in self.variance.values()):
            raise ValueError("observation variance must be positive")
        if any(v < 0 for v in self.representativeness.values()):
            raise ValueError("representativeness variance must be non-negative")
        if any(v < 0 for v in self.localization.values()):
            raise ValueError("localization factors must be non-negative")

    def lookup_arrays(self, keys: Sequence[str], source: str, *, strict: bool = False) -> tuple[np.ndarray, ...]:
        if strict:
            missing = [str(k) for k in keys
                       if str(k) not in self.bias
                       or str(k) not in self.variance
                       or (self.semantics != M00_QD_RELIABILITY_SEMANTICS
                           and str(k) not in self.representativeness)
                       or (self.localization and str(k) not in self.localization)]
            if missing:
                raise KeyError(f"strict calibration missing exact keys: {missing[:3]}")
            bias = np.asarray([self.bias[str(k)] for k in keys], float)
            r_obs = np.asarray([self.variance[str(k)] for k in keys], float)
            r_rep = np.asarray([
                self.representativeness.get(str(k), 0.0) for k in keys], float)
            loc = np.asarray([
                self.localization[str(k)] if self.localization else 1.0
                for k in keys], float)
        else:
            bias = np.asarray([_lookup(self.bias, k, source, 0.0) for k in keys], float)
            r_obs = np.asarray([_lookup(self.variance, k, source, 1.0) for k in keys], float)
            r_rep = np.asarray([_lookup(self.representativeness, k, source, 0.0) for k in keys], float)
            loc = np.asarray([_lookup(self.localization, k, source, 1.0) for k in keys], float)
        r_obs = np.clip(r_obs, self.variance_floor, self.variance_cap)
        r_rep = np.clip(r_rep, 0.0, self.variance_cap)
        if not (np.isfinite(bias).all() and np.isfinite(r_obs).all()
                and np.isfinite(r_rep).all() and np.isfinite(loc).all()):
            raise ValueError("calibration lookup is non-finite")
        if (loc < 0).any():
            raise ValueError("calibration localization is negative")
        return bias, r_obs, r_rep, loc

    def apply(self, values: Sequence[float], keys: Sequence[str],
              source: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        if len(values) != len(keys):
            raise ValueError("values and keys must have equal length")
        x = _as_numpy(values, dtype=float).reshape(-1)
        if not np.isfinite(x).all():
            raise ValueError("non-finite observations")
        src = source or (str(keys[0]).split("|")[0] if keys else "default")
        b, r, _, _ = self.lookup_arrays(keys, src, strict=self.semantics == M00_OBSERVATION_V2_SEMANTICS)
        return x - b, r

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": self.semantics,
            "bias": dict(self.bias),
            "variance": dict(self.variance),
            "representativeness": dict(self.representativeness),
            "localization": dict(self.localization),
            "variance_floor": self.variance_floor,
            "variance_cap": self.variance_cap,
        }


def calibration_to_dict(calibration: ObservationCalibration | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(calibration, ObservationCalibration):
        return calibration.to_dict()
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration must be a mapping or ObservationCalibration")
    normalized = ObservationCalibration(
        calibration.get("bias", {}),
        calibration.get("variance", calibration.get("r_obs", {})),
        calibration.get("representativeness", calibration.get("r_rep", {})),
        calibration.get("localization", {}),
        calibration.get("semantics", M00_OBSERVATION_SEMANTICS),
        calibration.get("variance_floor", 1e-6),
        calibration.get("variance_cap", 1e6),
    ).to_dict()
    # Preserve non-numeric contract/audit metadata used by strict-v2 loaders.
    for key in ("train_only", "locked_test", "date_manifest", "train_dates",
                "representativeness_support", "magnetic_coordinate",
                "qd_coordinate", "qd_aacgm_migration", "aacgm_audit",
                "localization_contract", "fit", "r_obs_semantics",
                "r_rep_semantics", "reliability_weight",
                "variance_interpretation", "residual_reference"):
        if key in calibration:
            normalized[key] = calibration[key]
    return normalized


def load_observation_calibration(path: str | os.PathLike[str],
                                 table_kind: str | None = None
                                 ) -> ObservationCalibration:
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"M00 observation calibration is missing: {path}")
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("M00 observation calibration JSON must be an object")
    # Bias and R are often exported as separate flat tables.  Accept those
    # files as shorthand while retaining the full contract for combined files.
    if not any(key in payload for key in (
            "bias", "variance", "r_obs", "representativeness",
            "r_rep", "localization", "semantics")):
        if table_kind not in (None, "bias", "variance", "r_obs",
                              "representativeness", "r_rep", "localization"):
            raise ValueError(f"unknown flat calibration table kind: {table_kind}")
        if table_kind == "bias":
            return ObservationCalibration(payload, {})
        if table_kind in ("representativeness", "r_rep"):
            return ObservationCalibration({}, {}, representativeness=payload)
        if table_kind == "localization":
            return ObservationCalibration({}, {}, localization=payload)
        # A flat R table is the only untyped shorthand retained for backwards
        # compatibility; explicit --m00-observation-r-path passes this kind.
        return ObservationCalibration({}, payload)
    calibration = ObservationCalibration(
        payload.get("bias", {}),
        payload.get("variance", payload.get("r_obs", {})),
        payload.get("representativeness", payload.get("r_rep", {})),
        payload.get("localization", {}),
        payload.get("semantics", M00_OBSERVATION_SEMANTICS),
        payload.get("variance_floor", 1e-6),
        payload.get("variance_cap", 1e6),
    )
    if calibration.semantics in {M00_OBSERVATION_V2_SEMANTICS,
                                 M00_QD_RELIABILITY_SEMANTICS}:
        validate_strict_v2_calibration(payload)
    return calibration


def validate_strict_v2_calibration(payload: Mapping[str, Any]) -> None:
    """Fail closed for the non-empty Train-only v2 calibration contract."""
    if not isinstance(payload, Mapping):
        raise ValueError("strict v2 calibration must be an object")
    if payload.get("semantics") not in {M00_OBSERVATION_V2_SEMANTICS,
                                         M00_QD_RELIABILITY_SEMANTICS}:
        raise ValueError("strict calibration semantics mismatch")
    if payload.get("train_only") is not True or payload.get("locked_test") != "excluded":
        raise ValueError("strict v2 calibration must be Train-only")
    rep = payload.get("representativeness", {})
    if not isinstance(rep, Mapping):
        raise ValueError("strict calibration representativeness must be a mapping")
    if not all(math.isfinite(float(v)) and float(v) >= 0.0 for v in rep.values()):
        raise ValueError("strict representativeness contains invalid values")
    support = payload.get("representativeness_support", payload.get("support", {}))
    if str(payload.get("semantics")) != M00_QD_RELIABILITY_SEMANTICS:
        if not isinstance(rep, Mapping) or not rep:
            raise ValueError("strict v2 calibration requires non-empty representativeness")
        if not isinstance(support, Mapping) or not support:
            raise ValueError("strict v2 calibration lacks representativeness support audit")
    if not isinstance(support, Mapping):
        raise ValueError("strict calibration support audit must be a mapping")
    for key, item in support.items():
        if (not isinstance(item, Mapping)
                or int(item.get("profiles", 0)) < 20
                or int(item.get("profile_days", 0)) < 10
                or float(item.get("ess", 0.0)) < 10.0):
            raise ValueError(f"strict calibration support is insufficient: {key}")
    if rep and set(map(str, rep)) - set(map(str, support)):
        raise ValueError("strict calibration support does not cover all keys")
    if str(payload.get("semantics")) == M00_QD_RELIABILITY_SEMANTICS:
        weights = payload.get("reliability_weight")
        if not isinstance(weights, Mapping) or set(map(str, weights)) != set(map(str, payload.get("variance", {}))):
            raise ValueError("QD reliability calibration requires exact reliability weights")
    magnetic = payload.get("magnetic_coordinate")
    if not isinstance(magnetic, Mapping) or magnetic.get("semantics") not in {
            "aacgm_v2", "qd_v1", AACGM_MAGNETIC_STRATUM_SEMANTICS,
            QD_MAGNETIC_STRATUM_SEMANTICS}:
        raise ValueError("strict v2 calibration lacks magnetic-coordinate provenance")
    if magnetic.get("fallback") not in (None, False, "none"):
        raise ValueError("strict v2 calibration cannot use magnetic-coordinate fallback")
    coordinate_semantics = str(magnetic.get("semantics"))
    if coordinate_semantics in {"qd_v1", QD_MAGNETIC_STRATUM_SEMANTICS}:
        if str(magnetic.get("qd_converter", "")).lower() != "apexpy":
            raise ValueError("strict QD calibration lacks apexpy converter")
        if not str(magnetic.get("apexpy_version", "")).strip():
            raise ValueError("strict QD calibration lacks apexpy version")
    elif "aacgmv2" not in str(magnetic.get("aacgm_version", "")).lower():
        raise ValueError("strict v2 calibration lacks AACGM converter version")
    if not str(magnetic.get("reference_time_utc", "")).strip():
        raise ValueError("strict v2 calibration lacks AACGM reference time")
    if magnetic.get("actual_height_semantics") not in (
            "row_altitude_km_v1", "actual_altitude_km_v1"):
        raise ValueError("strict v2 calibration lacks actual-height semantics")
    loc = payload.get("localization_contract")
    if not isinstance(loc, Mapping) or loc.get("semantics") != M00_3D_LOCALIZATION_SEMANTICS:
        raise ValueError("strict calibration lacks 3-D localization contract")
    for key in ("spatial_scale_km", "vertical_scale_km", "temporal_scale_hours"):
        if not math.isfinite(float(loc.get(key, float("nan")))) or float(loc[key]) <= 0:
            raise ValueError(f"strict v2 localization contract missing positive {key}")
    if loc.get("time_semantics") not in ("relative_utc_hours_v1", "utc_hours"):
        raise ValueError("strict v2 localization requires continuous UTC semantics")
    if loc.get("applies_to") not in (None, "observation_precision_gain_only", "diagnostic_only"):
        raise ValueError("localization may only affect observation precision/gain")


def fit_train_calibration(values: Sequence[float], residuals: Sequence[float],
                          keys: Sequence[str], *, shrinkage: float = 10.0,
                          variance_floor: float = 1e-6,
                          variance_cap: float = 1e6,
                          partitions: Sequence[str] | None = None
                          ) -> ObservationCalibration:
    """Fit a shrink-to-zero table from Train rows only."""
    if len(values) != len(residuals) or len(values) != len(keys):
        raise ValueError("calibration inputs must have equal length")
    if partitions is not None:
        if len(partitions) != len(keys):
            raise ValueError("partition labels must align with calibration rows")
        if any(str(part).lower().replace("-", "_") != "train"
               for part in partitions):
            raise ValueError("M00 calibration may only be fitted from Train rows")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")
    residual_arr = _as_numpy(residuals, dtype=float).reshape(-1)
    if not np.isfinite(residual_arr).all():
        raise ValueError("non-finite residual")
    groups: dict[str, list[float]] = {}
    for key, residual in zip(keys, residual_arr):
        groups.setdefault(str(key), []).append(float(residual))
    bias, variance = {}, {}
    for key, group in groups.items():
        arr = np.asarray(group, dtype=float)
        bias[key] = float(arr.mean() * len(arr) / (len(arr) + shrinkage))
        variance[key] = float(np.clip(max(float(arr.var()), variance_floor),
                                      variance_floor, variance_cap))
    return ObservationCalibration(bias, variance,
                                  variance_floor=variance_floor,
                                  variance_cap=variance_cap)


def fit_train_observation_calibration(values: Sequence[float],
                                      residuals: Sequence[float],
                                      coords: Any, source: str,
                                      partitions: Sequence[str] | None = None,
                                      magnetic_sin_i: Any | None = None,
                                      aacgm_latitude_deg: Any | None = None,
                                      strict_magnetic: bool = False,
                                      **kwargs: Any) -> ObservationCalibration:
    keys = observation_stratum_keys(
        coords, source, magnetic_sin_i,
        aacgm_latitude_deg=aacgm_latitude_deg,
        strict_magnetic=strict_magnetic).reshape(-1).tolist()
    return fit_train_calibration(values, residuals, keys,
                                 partitions=partitions, **kwargs)


def effective_observation_variance(r_obs: Any, r_rep: Any = 0.0,
                                   variance_floor: float = 1e-6,
                                   variance_cap: float = 1e6) -> np.ndarray:
    """Combine observation and representativeness variance additively."""
    a, b = np.broadcast_arrays(_as_numpy(r_obs, dtype=float),
                               _as_numpy(r_rep, dtype=float))
    floor, cap = float(variance_floor), float(variance_cap)
    if not (floor > 0 and cap >= floor):
        raise ValueError("invalid variance floor/cap")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("invalid variance arrays")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("variance must be non-negative")
    out = np.clip(a + b, floor, cap)
    if not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError("effective variance must be positive and finite")
    return out


def observation_error_weights(r_obs: Any, r_rep: Any = 0.0,
                              *, localization: Any = 1.0,
                              representativeness_weight: Any = 1.0,
                              variance_floor: float = 1e-6,
                              variance_cap: float = 1e6,
                              weight_cap: float = 4.0) -> np.ndarray:
    """Return finite mean-one precision weights with conservative clipping."""
    variance = effective_observation_variance(
        r_obs, r_rep, variance_floor, variance_cap)
    factors = np.asarray(localization, float) * np.asarray(
        representativeness_weight, float)
    factors = np.broadcast_to(factors, variance.shape)
    if not np.isfinite(factors).all() or (factors < 0).any():
        raise ValueError("invalid localization/representativeness weights")
    precision = factors / variance
    if not np.isfinite(precision).all() or not np.any(precision > 0):
        raise ValueError("observation precision has no valid positive rows")
    median = float(np.median(precision[precision > 0]))
    weights = np.clip(precision / median, 0.0, float(weight_cap))
    mean = float(weights.mean())
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("observation weights have zero effective sample size")
    return weights / mean


def representativeness_weights(spread: Any, *, floor: float = 1e-6,
                               cap: float = 4.0) -> np.ndarray:
    spread = _as_numpy(spread, dtype=float)
    if not np.isfinite(spread).all() or (spread < 0).any():
        raise ValueError("representativeness spread must be finite/non-negative")
    return observation_error_weights(np.ones_like(spread), spread,
                                     variance_floor=floor, weight_cap=cap)


def localization_weights(coords: Any, reference_coords: Any | None = None,
                          *, spatial_scale_km: float = 1800.0,
                          vertical_scale_km: float = 100.0,
                          temporal_scale_hours: float = 1.5,
                          time_semantics: str = "local_time_cyclic",
                          strict_3d: bool = False, strict: bool | None = None) -> np.ndarray:
    """Continuous Gaussian space/time localization factors."""
    arr = _as_numpy(coords, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] < 4:
        raise ValueError("coords must have shape [..., 4] or larger")
    if strict is not None:
        strict_3d = bool(strict)
    if reference_coords is None:
        if strict_3d:
            raise ValueError("strict 3-D localization requires reference_coords")
        return np.ones(arr.shape[:-1], dtype=float)
    ref = _as_numpy(reference_coords, dtype=float)
    if ref.shape[-1] < 4:
        raise ValueError("reference_coords must have four columns")
    arr, ref = np.broadcast_arrays(arr, ref)
    lat1, lon1, _alt1, time1 = [arr[..., i] for i in range(4)]
    lat2, lon2, _alt2, time2 = [ref[..., i] for i in range(4)]
    if not np.isfinite(arr[..., :4]).all() or not np.isfinite(ref[..., :4]).all():
        raise ValueError("non-finite localization coordinates")
    if spatial_scale_km <= 0 or vertical_scale_km <= 0 or temporal_scale_hours <= 0:
        raise ValueError("localization scales must be positive")
    if time_semantics not in {"utc_hours", "relative_utc_hours_v1", "local_time_cyclic"}:
        raise ValueError("unknown localization time semantics")
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = ((lon2 - lon1 + 180.0) % 360.0 - 180.0) * rad
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    distance = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    dt = np.abs(time2 - time1)
    if time_semantics == "local_time_cyclic":
        dt = np.mod(dt, 24.0)
        dt = np.minimum(dt, 24.0 - dt)
    vertical = np.abs(_alt2 - _alt1)
    def gc(x):
        u = np.abs(np.asarray(x, dtype=float))
        q = 2.0 * u
        first = 1.0 - (5.0 / 3.0) * q**2 + (5.0 / 8.0) * q**3 + 0.5 * q**4 - 0.25 * q**5
        safe = np.maximum(q, 1e-12)
        second = 4.0 - 5.0*q + (5.0/3.0)*q**2 + (5.0/8.0)*q**3 - 0.5*q**4 + (1.0/12.0)*q**5 - 2.0/(3.0*safe)
        return np.where(q <= 1.0, first, np.where(q < 2.0, second, 0.0)).clip(0.0, 1.0)
    out = gc(distance / spatial_scale_km) * gc(vertical / vertical_scale_km) * gc(dt / temporal_scale_hours)
    if not np.isfinite(out).all():
        raise ValueError("non-finite localization weights")
    return out


def _m00_qd_profile_stratum_loss(
        prediction: Any, target: Any, coords: Any, profile_ids: Any,
        source: str, cal: ObservationCalibration,
        reliability_by_key: Mapping[str, float], *, delta: float,
        magnetic_sin_i: Any | None, aacgm_latitude_deg: Any | None,
        qd_latitude_deg: Any | None) -> tuple[Any, dict[str, Any]]:
    """Apply the QD empirical reliability exactly once at profile/stratum level.

    A profile remains the statistical unit.  Its points are first grouped by
    their actual physical stratum, then each group contributes in proportion
    to its point exposure within that profile.  This prevents mixed-height or
    mixed-magnetic-latitude profiles from being represented by one midpoint.
    """
    import torch
    import torch.nn.functional as F

    pred = prediction.reshape(-1)
    truth = target.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    c = coords.reshape(-1, coords.shape[-1])
    ids = profile_ids.reshape(-1)
    if ids.numel() != pred.numel() or c.shape[0] != pred.numel():
        raise ValueError("QD calibration rows must align with prediction")
    if not bool(torch.isfinite(pred).all() and torch.isfinite(truth).all()):
        raise ValueError("QD M00 calibration received non-finite prediction/target")
    keys = observation_stratum_keys(
        c, source, magnetic_sin_i, aacgm_latitude_deg=aacgm_latitude_deg,
        qd_latitude_deg=qd_latitude_deg, strict_magnetic=True,
    ).reshape(-1).tolist()
    bias, _r_obs, _r_rep, _loc = cal.lookup_arrays(keys, source, strict=True)
    del bias  # QD residual reliability is an exposure weight, not label debiasing.
    if not reliability_by_key:
        raise ValueError("QD calibration requires a non-empty reliability table")
    missing = sorted({str(key) for key in keys} - set(reliability_by_key))
    if missing:
        raise KeyError(f"QD reliability table missing exact keys: {missing[:3]}")
    id_np = ids.detach().cpu().numpy().reshape(-1)
    groups: dict[tuple[int, str], int] = {}
    group_ids = []
    group_profiles = []
    group_keys = []
    for pid, key in zip(id_np.tolist(), keys):
        token = (int(pid), str(key))
        group_ids.append(groups.setdefault(token, len(groups)))
        if group_ids[-1] == len(group_profiles):
            group_profiles.append(int(pid))
            group_keys.append(str(key))
    group_ids_t = torch.as_tensor(group_ids, device=pred.device, dtype=torch.long)
    unique_profiles, profile_inverse = torch.unique(
        ids, sorted=False, return_inverse=True)
    group_count = len(groups)
    rel_np = np.asarray([float(reliability_by_key[key]) for key in group_keys],
                        dtype=float)
    if (not np.isfinite(rel_np).all() or np.any(rel_np < 0.25)
            or np.any(rel_np > 4.0)):
        raise ValueError("QD reliability weights must be finite in [0.25, 4]")
    point_loss = F.huber_loss(pred, truth, reduction="none", delta=delta)
    group_sum = torch.zeros(group_count, device=pred.device, dtype=point_loss.dtype)
    group_n = torch.zeros_like(group_sum)
    group_sum.scatter_add_(0, group_ids_t, point_loss)
    group_n.scatter_add_(0, group_ids_t, torch.ones_like(point_loss))
    group_mean = group_sum / group_n.clamp_min(torch.finfo(point_loss.dtype).eps)
    group_profile_inverse = torch.as_tensor(
        [int(np.flatnonzero(unique_profiles.detach().cpu().numpy() == pid)[0])
         for pid in group_profiles], device=pred.device, dtype=torch.long)
    profile_n = torch.zeros(unique_profiles.numel(), device=pred.device, dtype=point_loss.dtype)
    profile_n.scatter_add_(0, profile_inverse, torch.ones_like(point_loss))
    alpha = group_n / profile_n[group_profile_inverse].clamp_min(
        torch.finfo(point_loss.dtype).eps)
    rel = torch.as_tensor(rel_np, device=pred.device, dtype=point_loss.dtype)
    profile_mass = torch.zeros_like(profile_n)
    profile_mass.scatter_add_(0, group_profile_inverse, alpha * rel)
    numerator = torch.sum(group_mean * alpha * rel)
    denominator = profile_mass.sum().clamp_min(torch.finfo(point_loss.dtype).eps)
    loss = numerator / denominator
    mass = profile_mass.detach().cpu().numpy()
    details = {
        "valid_points": int(pred.numel()),
        "valid_profiles": int(unique_profiles.numel()),
        "valid_profile_strata": int(group_count),
        "effective_variance_mean": 1.0,
        "weight_mean": 1.0,
        "weight_max": 1.0,
        "bias_mean": 0.0,
        "reliability_ess": float((mass.sum() ** 2) / max(np.square(mass).sum(), 1e-12)),
        "aggregation": "point_to_profile_stratum_to_source_v1",
        "reliability_applied_once": True,
        "representativeness_applied": False,
        "localization_applied": False,
    }
    return loss, details


def m00_observation_error_loss(prediction: Any, target: Any, coords: Any,
                               profile_ids: Any, source: str,
                               calibration: ObservationCalibration | Mapping[str, Any],
                               *, delta: float = 0.2,
                               valid_mask: Any | None = None,
                               magnetic_sin_i: Any | None = None,
                               aacgm_latitude_deg: Any | None = None,
                               qd_latitude_deg: Any | None = None,
                               strict_magnetic: bool = False,
                               use_representativeness: bool = True,
                               use_localization: bool = True,
                               weight_cap: float = 4.0) -> tuple[Any, dict[str, Any]]:
    """Torch profile-balanced Huber loss using frozen M00 error calibration."""
    import torch
    import torch.nn.functional as F

    reliability_by_key = {}
    if isinstance(calibration, ObservationCalibration):
        cal = calibration
    else:
        reliability_by_key = {
            str(key): float(value)
            for key, value in calibration.get('reliability_weight', {}).items()
        }
        cal = ObservationCalibration(
            calibration.get("bias", {}),
            calibration.get("variance", calibration.get("r_obs", {})),
            calibration.get("representativeness", calibration.get("r_rep", {})),
            calibration.get("localization", {}),
            calibration.get("semantics", M00_OBSERVATION_SEMANTICS),
            calibration.get("variance_floor", 1e-6),
            calibration.get("variance_cap", 1e6),
        )
    if cal.semantics == M00_QD_RELIABILITY_SEMANTICS:
        return _m00_qd_profile_stratum_loss(
            prediction, target, coords, profile_ids, source, cal,
            reliability_by_key, delta=delta, magnetic_sin_i=magnetic_sin_i,
            aacgm_latitude_deg=aacgm_latitude_deg,
            qd_latitude_deg=qd_latitude_deg)

    pred = prediction.reshape(-1)
    truth = target.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    c = coords.reshape(-1, coords.shape[-1])
    ids = profile_ids.reshape(-1)
    if ids.numel() != pred.numel():
        if profile_ids.ndim == 1 and pred.numel() % profile_ids.numel() == 0:
            ids = profile_ids.reshape(-1, 1).expand(
                -1, pred.numel() // profile_ids.numel()).reshape(-1)
        else:
            raise ValueError("profile_ids must align with prediction")
    if c.shape[0] != pred.numel():
        raise ValueError("coords must align with prediction")
    keys = observation_stratum_keys(
        c, source, magnetic_sin_i, aacgm_latitude_deg=aacgm_latitude_deg,
        qd_latitude_deg=qd_latitude_deg,
        strict_magnetic=(strict_magnetic or cal.semantics in {
            M00_OBSERVATION_V2_SEMANTICS, M00_QD_RELIABILITY_SEMANTICS})).reshape(-1).tolist()
    strict_cal = cal.semantics in {M00_OBSERVATION_V2_SEMANTICS,
                                   M00_QD_RELIABILITY_SEMANTICS}
    bias, r_obs, r_rep, loc = cal.lookup_arrays(keys, source, strict=strict_cal)
    if not use_representativeness:
        r_rep.fill(0.0)
    if not use_localization:
        loc.fill(1.0)
    weights = observation_error_weights(
        r_obs, r_rep, localization=loc, weight_cap=weight_cap,
        variance_floor=cal.variance_floor, variance_cap=cal.variance_cap)
    device = pred.device
    b = torch.as_tensor(bias, device=device, dtype=pred.dtype)
    w = torch.as_tensor(weights, device=device, dtype=pred.dtype)
    valid = torch.isfinite(pred) & torch.isfinite(truth) & torch.isfinite(w)
    if valid_mask is not None:
        valid &= valid_mask.reshape(-1).to(device=device, dtype=torch.bool)
    if not bool(valid.any()):
        raise ValueError(f"M00 {source} calibration has zero valid rows")
    point = F.huber_loss(pred[valid], truth[valid] - b[valid],
                        reduction="none", delta=delta) * w[valid]
    valid_ids = ids[valid]
    _, inverse = torch.unique(valid_ids, sorted=False, return_inverse=True)
    sums = torch.zeros(int(inverse.max().item()) + 1, device=device,
                       dtype=point.dtype)
    norm = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, point)
    norm.scatter_add_(0, inverse, w[valid])
    profile_loss = sums / norm.clamp_min(torch.finfo(point.dtype).eps)
    if reliability_by_key:
        reliability = torch.as_tensor(
            [reliability_by_key.get(str(key), 1.0) for key in keys],
            device=pred.device, dtype=pred.dtype)
        reliability = reliability[valid]
        profile_reliability = torch.zeros_like(sums)
        # Average reliability with the same observation weights used for the
        # profile loss.  An unweighted sum divided by ``norm`` would distort
        # every profile whenever the calibrated observation variance differs
        # from one.
        profile_reliability.scatter_add_(0, inverse, reliability * w[valid])
        profile_reliability = profile_reliability / norm.clamp_min(
            torch.finfo(point.dtype).eps)
        profile_reliability = profile_reliability.clamp(0.25, 4.0)
        profile_loss = profile_loss * profile_reliability
        loss = profile_loss.sum() / profile_reliability.sum().clamp_min(
            torch.finfo(point.dtype).eps)
    else:
        loss = profile_loss.mean()
    valid_np = valid.detach().cpu().numpy()
    details = {
        "valid_points": int(valid.sum().item()),
        "valid_profiles": int(torch.unique(valid_ids).numel()),
        "effective_variance_mean": float(np.mean(
            effective_observation_variance(r_obs, r_rep))),
        "weight_mean": float(weights[valid_np].mean()),
        "weight_max": float(weights.max()),
        "bias_mean": float(np.mean(bias)),
        "reliability_ess": float(
            (profile_reliability.sum().detach().cpu().item() ** 2)
            / max(profile_reliability.square().sum().detach().cpu().item(), 1e-12)
        ) if reliability_by_key else float(torch.unique(valid_ids).numel()),
    }
    return loss, details


def apply_m00_observation_calibration(*args: Any, **kwargs: Any):
    """Compatibility alias used by training/evaluation callers."""
    return m00_observation_error_loss(*args, **kwargs)


def validate_m00_transition(summary: Mapping[str, Any]) -> None:
    validate_low_dim_assimilation_config(summary)
    validate_frozen_m00_config(summary)
    """Reject a hand-off from a failed adapter route with the wrong seed."""
    if summary.get("adapter_route") in {"stopped", "failed"} and summary.get(
            "m00_reference") != "A1S_epoch3":
        raise ValueError("failed adapter route may only transition from A1S_epoch3")
    if summary.get("locked_test") not in ("excluded", None):
        raise ValueError("Locked-test must remain excluded")


def validate_observation_config(config: Mapping[str, Any]) -> None:
    validate_low_dim_assimilation_config(config)
    validate_frozen_m00_config(config)
    """Fail closed at the training boundary for optional M00 calibration."""
    sem = str(config.get("m00_observation_calibration_semantics", "disabled"))
    allowed = {
        "disabled", "train_only_source_stratum_bias_r_v1",
        M00_OBSERVATION_SEMANTICS, M00_BIAS_R_SEMANTICS,
        M00_OBSERVATION_V2_SEMANTICS,
        M00_REP_LOC_SEMANTICS,
        M00_QD_RELIABILITY_SEMANTICS,
    }
    if sem not in allowed:
        raise ValueError(f"unsupported M00 observation semantics: {sem}")
    enabled = sem != "disabled"
    if sem in {M00_OBSERVATION_V2_SEMANTICS, M00_QD_RELIABILITY_SEMANTICS}:
        sidecars = (
            config.get("m00_observation_qd_sidecar_fy") or
            config.get("m00_observation_magnetic_sidecar_fy"),
            config.get("m00_observation_qd_sidecar_cosmic") or
            config.get("m00_observation_magnetic_sidecar_cosmic"),
        )
        for sidecar in sidecars:
            if sidecar is not None and not os.path.isfile(os.fspath(sidecar)):
                raise FileNotFoundError(f"AACGM sidecar missing: {sidecar}")
        if not all(sidecars) and not all(
                config.get(name) is not None for name in (
                    "m00_observation_aacgm_latitude_fy",
                    "m00_observation_aacgm_latitude_cosmic")):
            raise ValueError("strict magnetic calibration requires FY/COSMIC sidecars or arrays")
    bias_path = config.get("m00_observation_bias_path")
    r_path = config.get("m00_observation_r_path")
    if enabled and not (bias_path or r_path or config.get("m00_observation_calibration")):
        raise ValueError("M00 calibration requires a frozen Train-only table/path")
    for path in (bias_path, r_path):
        if path is not None and not os.path.isfile(os.fspath(path)):
            raise FileNotFoundError(f"M00 calibration file is missing: {path}")
    if enabled:
        seed = config.get('m00_observation_seed_checkpoint') or config.get(
            'background_seed_ckpt')
        if not seed:
            raise ValueError(
                'M00 observation calibration requires an A1-S Background seed')
        if not os.path.isfile(os.fspath(seed)):
            raise FileNotFoundError(f"M00 observation seed is missing: {seed}")
    stage = str(config.get("m00_observation_stage", "bias_r"))
    if stage not in {"bias_r", "representativeness_localization", "full"}:
        raise ValueError(f"unsupported M00 observation stage: {stage}")
    # Stage semantics are authoritative.  Stage A uses calibrated Bias/R;
    # Stage B and full additionally enable the frozen representativeness and
    # localization factors at the loss call site.
    config['m00_representativeness_enabled'] = stage in {
        'representativeness_localization', 'full'}
    config['m00_localization_enabled'] = stage in {
        'representativeness_localization', 'full'}
    if config.get("m00_transition_status") == "stopped" and enabled:
        raise ValueError("stopped M00 route cannot enable calibration")


if __name__ == "__main__":
    c = fit_train_calibration([1, 2], [0.1, -0.1],
                              ["FY|200-300|day|low"] * 2,
                              partitions=["Train", "Train"])
    assert np.isfinite(effective_observation_variance([1.0], [2.0])).all()
    print(c.to_dict()["semantics"])
