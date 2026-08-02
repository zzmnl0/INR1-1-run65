"""Profile-level audit of the M2-L COSMIC top-8 ordering correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_satellite_development import _partition_loader
from inr_modules.data_managers.FY_dataloader import (
    COSMICDataset,
    COSMICNeighborhoodIndex,
    FY3D_Dataset,
    _allowed_profile_mask,
)


ROOT = Path(__file__).resolve().parent


def _sample_profile_queries(dataset, samples, seed):
    profile_ids = np.asarray(dataset.profile_ids, dtype=np.int64)
    unique_ids, first = np.unique(profile_ids, return_index=True)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(
        len(unique_ids), size=min(samples, len(unique_ids)), replace=False))
    coords = []
    selected_ids = []
    for index in first[chosen]:
        values, _, profile_id = dataset[int(index)]
        coords.append(values[:4].numpy())
        selected_ids.append(int(profile_id))
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(selected_ids, dtype=np.int64),
    )


def _legacy_cosmic_selection(index, coords, allowed_profile_ids,
                             exclude_profile_ids=None):
    """Reproduce the pre-M2-L spatial-only COSMIC top-k selection."""
    selected = np.full((len(coords), index.k_prof), -1, dtype=np.int64)
    for row, coord in enumerate(coords):
        lo, hi = index._cand_slice(float(coord[3]))
        metadata = index.prof_sorted_meta[lo:hi]
        profile_ids = index.prof_sorted_ids[lo:hi]
        if not len(metadata):
            continue
        dt = np.abs(metadata[:, 2] - coord[3])
        dlat = np.abs(metadata[:, 0] - coord[0])
        dlon = np.abs(
            (metadata[:, 1] - coord[1] + 180.0) % 360.0 - 180.0)
        valid = (
            (dt <= index.dt)
            & (dlat <= index.dlat)
            & (dlon <= index.dlon)
            & _allowed_profile_mask(profile_ids, allowed_profile_ids)
        )
        if exclude_profile_ids is not None:
            valid &= profile_ids != exclude_profile_ids[row]
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            continue
        distance = (
            (dlat[candidates] / index.dlat) ** 2
            + (dlon[candidates] / index.dlon) ** 2
        )
        chosen = candidates[np.argsort(distance)[:index.k_prof]]
        selected[row, :len(chosen)] = profile_ids[chosen]
    return selected


def _distance_summary(index, coords, selected_ids):
    flat_ids = selected_ids.ravel()
    valid = flat_ids >= 0
    if not valid.any():
        return {
            'tokens': 0,
            'normalized_time_median': None,
            'normalized_time_p95': None,
            'normalized_space_l2_median': None,
        }
    query = np.repeat(coords, selected_ids.shape[1], axis=0)[valid]
    ids = flat_ids[valid]
    order = np.argsort(index.prof_ids)
    sorted_ids = index.prof_ids[order]
    positions = np.searchsorted(sorted_ids, ids)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(
            sorted_ids[positions], ids):
        raise ValueError('selected COSMIC profile ID is absent from the index')
    metadata = index.prof_meta[order[positions]]
    dt = np.abs(metadata[:, 2] - query[:, 3]) / index.dt
    dlat = np.abs(metadata[:, 0] - query[:, 0]) / index.dlat
    dlon = np.abs(
        (metadata[:, 1] - query[:, 1] + 180.0) % 360.0 - 180.0
    ) / index.dlon
    return {
        'tokens': int(len(ids)),
        'normalized_time_median': float(np.median(dt)),
        'normalized_time_p95': float(np.quantile(dt, 0.95)),
        'normalized_space_l2_median': float(np.median(np.sqrt(
            dlat ** 2 + dlon ** 2))),
    }


def _comparison(old_ids, new_ids):
    replaced = 0
    denominator = 0
    exact = 0
    for old_row, new_row in zip(old_ids, new_ids):
        old_set = set(old_row[old_row >= 0].tolist())
        new_set = set(new_row[new_row >= 0].tolist())
        mass = max(len(old_set), len(new_set))
        denominator += mass
        replaced += mass - len(old_set & new_set)
        exact += old_set == new_set
    return {
        'queries': int(len(old_ids)),
        'exact_profile_set_fraction': float(exact / len(old_ids)),
        'profile_replacement_fraction': (
            float(replaced / denominator) if denominator else None),
        'ordered_slot_change_fraction': float(np.mean(old_ids != new_ids)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-source', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_source < 1:
        raise ValueError('samples-per-source must be positive')

    run_dir = args.run_dir.resolve()
    with (run_dir / 'run_manifest.json').open(encoding='utf-8') as stream:
        config = json.load(stream)['config']
    date_manifest = Path(config['date_split_manifest'])
    if not date_manifest.is_absolute():
        date_manifest = ROOT / date_manifest
    with date_manifest.open(encoding='utf-8') as stream:
        partitions = json.load(stream)['partitions']
    if set(partitions) != {'train', 'development', 'locked_test'}:
        raise ValueError('date manifest partitions are incomplete')

    fy = _partition_loader(
        FY3D_Dataset, config, partitions, 'development').dataset
    cosmic = _partition_loader(
        COSMICDataset, config, partitions, 'development').dataset
    allowed = np.unique(np.asarray(cosmic.profile_ids, dtype=np.int64))
    index = COSMICNeighborhoodIndex(config['cosmic_path'], config)

    report = {
        'schema_version': 1,
        'purpose': 'M2-L development profile-level COSMIC neighbor reordering audit',
        'partition': 'development',
        'locked_test_accessed': False,
        'seed': args.seed,
        'samples_per_source_requested': args.samples_per_source,
        'old_metric': 'normalized latitude/longitude L2',
        'new_metric': 'normalized latitude/longitude/time L1',
        'sources': {},
    }
    for source_index, (source, dataset) in enumerate((('FY', fy), ('COSMIC', cosmic))):
        coords, target_ids = _sample_profile_queries(
            dataset, args.samples_per_source, args.seed + source_index)
        exclude = target_ids if source == 'COSMIC' else None
        old_ids = _legacy_cosmic_selection(index, coords, allowed, exclude)
        current = index.query_profiles_only(
            coords,
            exclude_profile_ids=exclude,
            allowed_profile_ids=allowed,
        )
        new_ids = np.where(
            current['valid_prof'], current['sel_ids'], -1)
        report['sources'][source] = {
            'sampled_profiles': int(len(coords)),
            'sample_profile_ids_sha256': hashlib.sha256(
                target_ids.tobytes()).hexdigest(),
            'comparison': _comparison(old_ids, new_ids),
            'old_selection': _distance_summary(index, coords, old_ids),
            'new_selection': _distance_summary(index, coords, new_ids),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
