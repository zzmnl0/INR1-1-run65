"""Build a Train-only M00 observation Bias/R table.

The table is deliberately derived from the raw IRI proxy and the active
Train dates only.  It never opens Development or Locked-test rows and does
not calculate file hashes.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from inr_modules.config_mdia import get_config_mdia
from inr_modules.data_managers.FY_dataloader import COSMICDataset, FY3D_Dataset
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.m00_observation import (
    M00_BIAS_R_SEMANTICS,
    observation_stratum_keys,
)


def _train_days(path: str) -> list[int]:
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("date split manifest lacks partitions")
    train = np.asarray(partitions.get("train", []), dtype=np.int64)
    development = np.asarray(partitions.get("development", []), dtype=np.int64)
    locked = np.asarray(partitions.get("locked_test", []), dtype=np.int64)
    # A1-S is fixed at 20 Train / 5 Development / 5 Locked-test over 30 days.
    # Keep this calibration route fail-closed so a stale 15-day manifest cannot
    # silently produce an apparently valid M00 table.
    if len(train) != 20 or len(development) != 5 or len(locked) != 5:
        raise ValueError(
            "A1-S M00 calibration requires exactly 20 Train, 5 Development "
            "and 5 Locked-test dates")
    covered = np.concatenate((train, development, locked))
    if (np.unique(covered).size != 30 or np.any(covered < 0)
            or np.any(covered >= 30)):
        raise ValueError("A1-S date manifest must cover relative days 0..29 exactly once")
    if train.ndim != 1 or np.unique(train).size != train.size:
        raise ValueError("Train dates must be unique")
    if np.intersect1d(train, development).size or np.intersect1d(train, locked).size:
        raise ValueError("date manifest has Train leakage")
    # A1-S source manifests still list Locked-test dates; they are audited for
    # disjointness above and never passed to a dataset or loader.
    return train.tolist()


def _dataset(config, source: str, train_days: list[int], alt_range):
    common = dict(
        mode="train",
        split_days={"train": train_days},
        val_days=[],
        bin_size_hours=config["bin_size_hours"],
        use_memmap=True,
        val_ratio=None,
        split_seed=config["seed"],
        alt_range=alt_range,
    )
    if source == "FY":
        return FY3D_Dataset(
            config["fy_path"],
            profile_index_path=config.get("fy_profile_index_path"),
            profile_path=config.get("fy_profile_path"),
            **common,
        )
    return COSMICDataset(
        config["cosmic_path"],
        profile_index_path=config.get("cosmic_profile_index_path"),
        **common,
    )


def _accumulate(dataset, source: str, proxy, device, batch_rows: int,
                max_rows: int | None = None, *, magnetic_sin_i=None,
                aacgm_latitude_deg=None, strict_magnetic: bool = False):
    stats = defaultdict(lambda: [0, 0.0, 0.0])
    profile_residuals = defaultdict(lambda: defaultdict(list))
    profile_days = defaultdict(lambda: defaultdict(set))
    selected = np.asarray(dataset.selected_indices, dtype=np.int64)
    if max_rows is not None:
        selected = selected[: int(max_rows)]
    rows_seen = 0
    with torch.no_grad():
        for start in range(0, len(selected), batch_rows):
            indices = selected[start:start + batch_rows]
            data = np.asarray(dataset.data[indices], dtype=np.float32)
            finite = np.isfinite(data[:, :5]).all(axis=1)
            if not finite.any():
                continue
            data = data[finite]
            coords = torch.as_tensor(data[:, :4], device=device)
            target = data[:, 4].astype(np.float64)
            prediction = proxy(coords).reshape(-1).cpu().numpy().astype(np.float64)
            residual = target - prediction
            batch_mag = (None if magnetic_sin_i is None else
                         np.asarray(magnetic_sin_i)[indices][finite])
            batch_aacgm = (None if aacgm_latitude_deg is None else
                           np.asarray(aacgm_latitude_deg)[indices][finite])
            keys = observation_stratum_keys(
                data[:, :4], source, magnetic_sin_i=batch_mag,
                aacgm_latitude_deg=batch_aacgm,
                strict_magnetic=strict_magnetic).reshape(-1)
            profile_source = getattr(dataset, 'raw_profile_ids', dataset.profile_ids)
            profile_ids = np.asarray(profile_source[indices])[finite]
            for key, value, pid, row in zip(
                    keys.tolist(), residual.tolist(), profile_ids.tolist(),
                    data.tolist()):
                item = stats[str(key)]
                item[0] += 1
                item[1] += value
                item[2] += value * value
                profile_residuals[str(key)][int(pid)].append(float(value))
                profile_days[str(key)][int(pid)].add(
                    int(np.floor(float(row[3]) / 24.0)))
            rows_seen += int(len(data))
            if rows_seen and rows_seen % (batch_rows * 10) < batch_rows:
                print(f"[{source}] calibration rows={rows_seen:,}", flush=True)
    rep = {}
    rep_support = {}
    for key, by_profile in profile_residuals.items():
        within = [float(np.var(v, ddof=1)) for v in by_profile.values() if len(v) > 1]
        if within:
            rep[key] = float(max(np.median(within), 1e-4))
        rep_support[key] = {
            'profiles': int(len(by_profile)),
            'profiles_with_variance': int(len(within)),
            'profile_days': int(sum(
                len(days) for days in profile_days[key].values())),
            'ess': float(len(by_profile)),
        }
    return stats, rows_seen, rep, rep_support, profile_residuals


def build_calibration(output: str, *, date_manifest: str | None = None,
                      batch_rows: int = 8192, shrinkage: float = 10.0,
                      variance_floor: float = 1e-4,
                      variance_cap: float = 1e6,
                      max_rows: int | None = None,
                      strict_v2: bool = False,
                      magnetic_sidecar_fy: str | None = None,
                      magnetic_sidecar_cosmic: str | None = None) -> dict:
    config = get_config_mdia()
    manifest_path = date_manifest or config.get("date_split_manifest")
    if not manifest_path or not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"date split manifest is missing: {manifest_path}")
    train_days = _train_days(manifest_path)
    if not train_days:
        raise ValueError("Train date partition is empty")
    device = torch.device("cpu")
    # The active 120--500 model trains M00 against satellite observations in
    # the strict 200--500 km target domain.  The base config leaves this
    # override unset, so keep the route fail-closed to that explicit domain.
    observation_range = tuple(config.get("observation_alt_range") or (200.0, 500.0))
    if len(observation_range) != 2 or not observation_range[0] < observation_range[1]:
        raise ValueError(f"invalid observation altitude range: {observation_range}")
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    proxy.load_state_dict(torch.load(
        config["iri_proxy_path"], map_location=device, weights_only=True))
    proxy.eval()
    bias, variance, representativeness, rep_support = {}, {}, {}, {}
    row_counts = {}
    for source in ("FY", "COSMIC"):
        dataset = _dataset(config, source, train_days, tuple(observation_range))
        sidecar_path = magnetic_sidecar_fy if source == 'FY' else magnetic_sidecar_cosmic
        magnetic = None
        if strict_v2:
            if not sidecar_path:
                raise ValueError(f'strict v2 requires {source} AACGM sidecar')
            from inr_modules.mdia.m00_observation import load_magnetic_sidecar
            magnetic = load_magnetic_sidecar(
                sidecar_path, expected_rows=len(dataset.data), strict_aacgm=True,
                expected_profile_ids=getattr(dataset, 'raw_profile_ids', None),
                expected_coords=np.asarray(dataset.data[:, :4], dtype=np.float64))
        magnetic_sin_i = None if magnetic is None else magnetic['magnetic_sin_i']
        aacgm_latitude = None if magnetic is None else magnetic['aacgm_latitude_deg']
        stats, rows, rep, support, profile_residuals = _accumulate(
            dataset, source, proxy, device, batch_rows, max_rows,
            magnetic_sin_i=magnetic_sin_i,
            aacgm_latitude_deg=aacgm_latitude,
            strict_magnetic=strict_v2)
        if not stats:
            raise ValueError(f"{source} Train rows produced no calibration cells")
        if strict_v2 and set(map(str, stats)) - set(rep):
            missing_rep = sorted(set(map(str, stats)) - set(rep))
            raise ValueError(f"{source} strata lack profile support for R_rep: {missing_rep[:3]}")
        if strict_v2:
            unsupported = sorted(
                key for key, item in support.items()
                if int(item.get('profiles', 0)) < 20
                or int(item.get('profile_days', 0)) < 10
                or float(item.get('ess', 0.0)) < 10.0)
            if unsupported:
                raise ValueError(
                    f"{source} strata have insufficient Train support: {unsupported[:3]}")
        row_counts[source] = rows
        # R_obs is the variance of profile means; within-profile variance is
        # emitted separately as R_rep to avoid counting the same scatter twice.
        for key, by_profile in profile_residuals.items():
            profile_means = np.asarray(
                [np.mean(values) for values in by_profile.values()],
                dtype=np.float64)
            if profile_means.size > 1:
                variance[key] = float(np.clip(
                    max(float(np.var(profile_means, ddof=1)), variance_floor),
                    variance_floor, variance_cap))
        for key, (count, total, _total_sq) in stats.items():
            mean = total / count
            bias[key] = float(mean * count / (count + shrinkage))
            # ``variance`` is populated above from profile means.  Keep a
            # conservative floor only for the single-profile compatibility
            # case; strict-v2 support checks reject that case before writing.
            variance.setdefault(key, float(variance_floor))
        representativeness.update(rep)
        rep_support.update(support)
    if strict_v2 and not representativeness:
        raise ValueError('strict v2 requires non-empty Train-only representativeness table')
    payload = {
        "semantics": ("m00_train_only_observation_error_v2" if strict_v2
                       else M00_BIAS_R_SEMANTICS),
        "train_only": True,
        "locked_test": "excluded",
        "date_manifest": os.path.abspath(manifest_path),
        "train_dates": train_days,
        "observation_alt_range_km": [float(v) for v in observation_range],
        "rows": row_counts,
        "bias": bias,
        "variance": variance,
        "representativeness": representativeness,
        "representativeness_support": rep_support,
        "localization": {},
        "magnetic_coordinate": {
            "semantics": "aacgm_v2" if strict_v2 else "legacy_geographic_fallback",
            "fallback": False if strict_v2 else True,
            "aacgm_version": "aacgmv2",
            "reference_time_utc": "2024-09-15T00:00:00Z",
            "actual_height_semantics": "row_altitude_km_v1",
        },
        "localization_contract": {"semantics": "m00_3d_localization_v2",
                                  "spatial_scale_km": 1800.0,
                                  "vertical_scale_km": 100.0,
                                  "temporal_scale_hours": 1.5,
                                  "time_semantics": "relative_utc_hours_v1",
                                  "applies_to": "observation_precision_gain_only"},
        "variance_floor": float(variance_floor),
        "variance_cap": float(variance_cap),
        "fit": {
            "residual_reference": "target_minus_raw_iri_proxy",
            "shrinkage": float(shrinkage),
            "batch_rows": int(batch_rows),
            "max_rows_per_source": max_rows,
        },
    }
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output + ".tmp", "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(output + ".tmp", output)
    print(f"wrote Train-only M00 calibration: {output}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--date-split-manifest", default=None)
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    parser.add_argument("--variance-floor", type=float, default=1e-4)
    parser.add_argument("--variance-cap", type=float, default=1e6)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    parser.add_argument("--strict-v2", action="store_true")
    parser.add_argument("--magnetic-sidecar-fy", default=None)
    parser.add_argument("--magnetic-sidecar-cosmic", default=None)
    args = parser.parse_args()
    if args.batch_rows < 1 or args.shrinkage < 0:
        raise ValueError("batch-rows must be positive and shrinkage non-negative")
    build_calibration(
        args.output,
        date_manifest=args.date_split_manifest,
        batch_rows=args.batch_rows,
        shrinkage=args.shrinkage,
        variance_floor=args.variance_floor,
        variance_cap=args.variance_cap,
        max_rows=args.max_rows_per_source,
        strict_v2=args.strict_v2,
        magnetic_sidecar_fy=args.magnetic_sidecar_fy,
        magnetic_sidecar_cosmic=args.magnetic_sidecar_cosmic,
    )


if __name__ == "__main__":
    main()
