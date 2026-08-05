"""M2-S S1/S2: one balanced-exposure retry after an H2 attribution."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_etkf_observation_subspace import ROOT, _load, _restrict_profiles
from audit_m2r_representation import _physical_shadow
from audit_m2s_conflict import SEED, _profile_record, _rmse
from evaluate_satellite_development import _partition_loader
from inr_modules.data_managers.FY_dataloader import COSMICDataset, FY3D_Dataset
from pretrain_m2r_modes import _profile_modes, _ridge, _rows


def _balanced_order(records, fold):
    groups = defaultdict(list)
    for record in records:
        if record['fold'] == fold:
            groups[f"{record['source']}_{record['cell']}"] .append(record)
    names = sorted(groups)
    if len(names) < 8:
        raise ValueError(f'balanced retry requires all 8 source/cell groups, got {names}')
    offsets = defaultdict(int)
    while True:
        for name in names:
            values = groups[name]
            yield values[offsets[name] % len(values)]
            offsets[name] += 1


def _train(shadow, records, fold, steps, lr, sw, peak):
    model = copy.deepcopy(shadow)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mode_residual.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.mode_residual.parameters(), lr=lr)
    ridge = float(model.kalman_layer.r_fy)
    stream = _balanced_order(records, fold)
    model.train()
    for _ in range(steps):
        record = next(stream)
        learned, _, _, _ = _profile_modes(model, record['coords'], sw, peak)
        fit = record['fit']
        coefficients = _ridge(learned, record['residual'], fit, ridge)
        loss = (learned[~fit] @ coefficients - record['residual'][~fit]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _bootstrap_upper(values, seed, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return float(np.quantile(means, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-profiles', type=int, default=512)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_dir = args.run_dir.resolve()
    baseline, sw, peak, config = _load(run_dir, run_dir / 'epoch_09_model.pth')
    shadow = _physical_shadow(config, run_dir / 'best_background_model.pth')
    date_manifest = Path(config['date_split_manifest'])
    if not date_manifest.is_absolute():
        date_manifest = ROOT / date_manifest
    partitions = json.loads(date_manifest.read_text(encoding='utf-8'))['partitions']
    loaders = {
        'FY': _partition_loader(FY3D_Dataset, config, partitions, 'train'),
        'COSMIC': _partition_loader(COSMICDataset, config, partitions, 'train'),
    }
    selected = {source: _restrict_profiles(loader, args.max_profiles, 542 + index)
                for index, (source, loader) in enumerate(loaders.items())}
    records = []
    for source, profile_id, data in _rows(loaders):
        record = _profile_record(shadow, baseline, source, profile_id, data, sw, peak)
        if record is not None:
            records.append(record)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results, gates = [], []
    for index, (train_fold, test_fold) in enumerate((('A', 'B'), ('B', 'A'))):
        model = _train(shadow, records, train_fold, args.steps, args.lr, sw, peak)
        torch.save(model.state_dict(), output / f'h2_balanced_train_{train_fold}.pth')
        fold_rows = []
        ridge = float(model.kalman_layer.r_fy)
        with torch.no_grad():
            for record in records:
                if record['fold'] != test_fold:
                    continue
                modes, _, _, _ = _profile_modes(model, record['coords'], sw, peak)
                fit = record['fit']
                rank1 = float((record['residual'][fit].mean() -
                               record['residual'][~fit]).square().mean().sqrt())
                fold_rows.append({
                    'source': record['source'], 'cell': record['cell'],
                    'profile_id': record['profile_id'],
                    'M2-S': _rmse(modes, record, ridge),
                    'M2-O': _rmse(record['m2o'], record, ridge),
                    'rank1': rank1,
                })
        strata = {}
        fold_gate = True
        for source in ('FY', 'COSMIC'):
            for cell in ('overall', 'low_day', 'low_night', 'high_day', 'high_night'):
                values = [row for row in fold_rows if row['source'] == source and
                          (cell == 'overall' or row['cell'] == cell)]
                if not values:
                    continue
                ratio = np.mean([row['M2-S'] for row in values]) / np.mean(
                    [row['M2-O'] for row in values])
                strata[f'{source}_{cell}'] = {'profiles': len(values),
                                              'M2S_to_M2O': float(ratio)}
                if cell == 'overall' or len(values) >= 10:
                    fold_gate &= ratio <= 1.01
        rank1_upper = _bootstrap_upper(
            [row['M2-S'] - row['rank1'] for row in fold_rows], SEED + index)
        fold_gate &= rank1_upper < 0
        gates.append(bool(fold_gate))
        results.append({'train_fold': train_fold, 'test_fold': test_fold,
                        'strata': strata,
                        'rank1_paired_bootstrap_95_upper': rank1_upper,
                        'passed': bool(fold_gate), 'profiles': fold_rows})
    report = {
        'schema_version': 1, 'partition': 'train',
        'locked_test_accessed': False, 'development_accessed': False,
        'isr_accessed': False,
        'semantics': {'analysis_state': 'query_local_increment_coefficients',
                      'change': 'profile_source_altitude_day_equal_exposure_only'},
        'selected_profile_ids': {key: value.tolist() for key, value in selected.items()},
        'steps': args.steps, 'lr': args.lr, 'folds': results,
        'gates': {'both_folds_passed': all(gates),
                  'progression_to_S3_allowed': all(gates)},
    }
    (output / 's1_h2_balanced_report.json').write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(output / 's1_h2_balanced_report.json')


if __name__ == '__main__':
    main()
