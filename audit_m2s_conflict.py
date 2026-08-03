"""M2-S S0 train-only support, gradient-conflict, and gauge audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_etkf_observation_subspace import ROOT, _load, _restrict_profiles
from audit_m2r_representation import _physical_shadow
from evaluate_satellite_development import _partition_loader
from inr_modules.data_managers.FY_dataloader import COSMICDataset, FY3D_Dataset
from pretrain_m2r_modes import _m2o_modes, _profile_modes, _ridge, _rows, _split


SEED = 42
SNAPSHOTS = (0, 25, 50, 100)
THRESHOLDS = {
    'support_gap_reduction_fraction': 0.50,
    'source_effect_smd': 0.20,
    'minimum_matched_profiles_per_major_stratum': 30,
    'gauge_spearman_absolute': 0.30,
    'gradient_norm_ratio': 3.0,
    'gradient_conflict_fraction': 0.50,
    'gradient_low_conflict_fraction': 0.25,
    'matching_caliper_standardized_distance': 2.5,
}


def _fold(source, profile_id):
    token = f'{source}:{profile_id}:{SEED}'.encode()
    return 'A' if hashlib.sha256(token).digest()[0] & 1 == 0 else 'B'


def _cell(coords):
    altitude = 'low' if float(coords[:, 2].median()) < 300.0 else 'high'
    lon = float(coords[:, 1].median())
    time = float(coords[:, 3].median())
    lst = (time + lon / 15.0) % 24.0
    day = 'day' if 6.0 <= lst < 18.0 else 'night'
    return f'{altitude}_{day}'


def _effective_rank(matrix):
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    if not float(energy.sum().detach()):
        return 0.0
    probability = energy / energy.sum()
    return float(torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum()).detach())


def _principal_angle(fit_modes, holdout_modes):
    left = torch.linalg.svd(fit_modes, full_matrices=False).Vh.T
    right = torch.linalg.svd(holdout_modes, full_matrices=False).Vh.T
    singular = torch.linalg.svdvals(left.T @ right).clamp(0.0, 1.0)
    return float(torch.rad2deg(torch.acos(singular.min())).detach())


def _spearman(x, y):
    if len(x) < 4:
        return None
    rx = np.argsort(np.argsort(np.asarray(x), kind='stable'), kind='stable')
    ry = np.argsort(np.argsort(np.asarray(y), kind='stable'), kind='stable')
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap(values, statistic=np.mean, seed=SEED, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    samples = [statistic(values[rng.integers(0, len(values), len(values))])
               for _ in range(draws)]
    return {'estimate': float(statistic(values)),
            'ci95': np.quantile(samples, [0.025, 0.975]).tolist(),
            'profiles': int(len(values))}


def _bootstrap_spearman(x, y, seed=SEED, draws=1000):
    x, y = np.asarray(x), np.asarray(y)
    estimate = _spearman(x, y)
    if estimate is None:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(x), len(x))
        value = _spearman(x[indices], y[indices])
        if value is not None:
            values.append(value)
    return {'estimate': estimate,
            'ci95': np.quantile(values, [0.025, 0.975]).tolist(),
            'profiles': int(len(x))}


def _profile_record(model, baseline, source, profile_id, data, sw, peak):
    coords, truth = data[:, :4], data[:, 4]
    if len(coords) < 4:
        return None
    learned, fixed, endpoint, center = _profile_modes(model, coords, sw, peak)
    original = _m2o_modes(baseline, coords, endpoint, center, sw, peak)
    residual = truth - endpoint['ne_bkg'].flatten().detach()
    fit = _split(source, profile_id, coords)
    altitude = coords[:, 2].sort().values
    gaps = altitude[1:] - altitude[:-1]
    fit_alt, holdout_alt = coords[fit, 2], coords[~fit, 2]
    fit_distance = torch.cdist(holdout_alt[:, None], fit_alt[:, None]).min(1).values
    lst = torch.remainder(coords[:, 3] + coords[:, 1] / 15.0, 24.0)
    features = [
        len(coords), float(altitude[-1] - altitude[0]), float(gaps.max()),
        float(fit_distance.mean()), float(torch.sin(math.pi * lst / 12).mean()),
        float(torch.cos(math.pi * lst / 12).mean()),
        float(coords[:, 0].mean()), float(endpoint['sin_i'].mean()),
        float(endpoint['sin_doy'].mean()), float(endpoint['cos_doy'].mean()),
        float(endpoint['iri_peak'][:, 0].mean()),
        float(endpoint['ne_bkg'].mean()), float(endpoint['kp_eff'].mean()),
        float(endpoint['f107_eff'].mean()),
    ]
    return {
        'source': source, 'profile_id': int(profile_id), 'fold': _fold(source, profile_id),
        'cell': _cell(coords), 'date_block': int(float(coords[:, 3].median()) // 24),
        'coords': coords, 'truth': truth, 'residual': residual,
        'fit': fit, 'fixed': fixed, 'm2o': original, 'r2': learned,
        'features': features,
    }


def _predict(modes, record, ridge):
    fit = record['fit']
    return modes[~fit] @ _ridge(modes, record['residual'], fit, ridge)


def _rmse(modes, record, ridge):
    return float((_predict(modes, record, ridge) - record['residual'][~record['fit']])
                 .square().mean().sqrt().detach())


def _load_r2(shadow, checkpoint):
    model = copy.deepcopy(shadow)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _train(shadow, records, sources, fold, steps, lr, sw, peak):
    model = copy.deepcopy(shadow)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mode_residual.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.mode_residual.parameters(), lr=lr)
    eligible = [r for r in records if r['source'] in sources and r['fold'] == fold]
    if not eligible:
        raise ValueError(f'empty training fold {fold} for {sources}')
    model.train()
    ridge = float(model.kalman_layer.r_fy)
    for step in range(steps):
        record = eligible[step % len(eligible)]
        learned, _, _, _ = _profile_modes(
            model, record['coords'], sw, peak)
        loss = (_predict(learned, record, ridge) -
                record['residual'][~record['fit']]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _leave_source_out(shadow, records, steps, lr, sw, peak):
    ridge = float(shadow.kalman_layer.r_fy)
    result = []
    for train_fold, test_fold in (('A', 'B'), ('B', 'A')):
        models = {
            label: _train(shadow, records, sources, train_fold, steps, lr, sw, peak)
            for label, sources in (
                ('FY-only', {'FY'}), ('COSMIC-only', {'COSMIC'}),
                ('pooled', {'FY', 'COSMIC'}))
        }
        for record in records:
            if record['fold'] != test_fold:
                continue
            row = {'train_fold': train_fold, 'test_fold': test_fold,
                   'source': record['source'], 'cell': record['cell'],
                   'profile_id': record['profile_id'], 'rmse': {}}
            for label, model in models.items():
                learned, _, _, _ = _profile_modes(
                    model, record['coords'], sw, peak)
                row['rmse'][label] = _rmse(learned, record, ridge)
            result.append(row)
    return result


def _gradient(model, records, sw, peak, ridge):
    losses = []
    for record in records[:8]:
        learned, _, _, _ = _profile_modes(model, record['coords'], sw, peak)
        losses.append((_predict(learned, record, ridge) -
                       record['residual'][~record['fit']]).square().mean())
    if not losses:
        return None
    gradients = torch.autograd.grad(torch.stack(losses).mean(),
                                    tuple(model.mode_residual.parameters()),
                                    allow_unused=False)
    return torch.cat([value.flatten() for value in gradients]).detach()


def _gradient_audit(shadow, records, steps, lr, sw, peak):
    model = copy.deepcopy(shadow)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mode_residual.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.mode_residual.parameters(), lr=lr)
    groups = defaultdict(list)
    for record in records:
        groups[f"{record['source']}_{record['cell']}"] .append(record)
    ridge = float(model.kalman_layer.r_fy)
    result = {}
    ordered = sorted(records, key=lambda row: (row['source'], row['profile_id']))
    for step in range(max(SNAPSHOTS) + 1):
        if step in SNAPSHOTS:
            vectors = {name: _gradient(model, values, sw, peak, ridge)
                       for name, values in sorted(groups.items())}
            pairs = {}
            norms = {name: float(vector.norm()) for name, vector in vectors.items()
                     if vector is not None}
            for left in sorted(vectors):
                for right in sorted(vectors):
                    if left >= right or vectors[left] is None or vectors[right] is None:
                        continue
                    cosine = torch.nn.functional.cosine_similarity(
                        vectors[left], vectors[right], dim=0)
                    pairs[f'{left}|{right}'] = float(cosine)
            result[str(step)] = {'cosines': pairs, 'norms': norms}
        if step == max(SNAPSHOTS):
            break
        record = ordered[step % len(ordered)]
        learned, _, _, _ = _profile_modes(model, record['coords'], sw, peak)
        loss = (_predict(learned, record, ridge) -
                record['residual'][~record['fit']]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return result


def _match(records):
    fy = [r for r in records if r['source'] == 'FY']
    cosmic = [r for r in records if r['source'] == 'COSMIC']
    matrix = np.asarray([r['features'] for r in records], dtype=np.float64)
    mean, scale = matrix.mean(0), matrix.std(0)
    scale[scale < 1e-8] = 1.0
    used, pairs = set(), []
    for left in sorted(fy, key=lambda r: (r['cell'], r['profile_id'])):
        candidates = [r for r in cosmic if r['cell'] == left['cell']
                      and r['profile_id'] not in used]
        if not candidates:
            continue
        x = (np.asarray(left['features']) - mean) / scale
        distances = [float(np.linalg.norm(
            x - (np.asarray(r['features']) - mean) / scale) / math.sqrt(len(x)))
                     for r in candidates]
        index = int(np.argmin(distances))
        if distances[index] <= THRESHOLDS['matching_caliper_standardized_distance']:
            right = candidates[index]
            used.add(right['profile_id'])
            pairs.append((left, right, distances[index]))
    return pairs


def _summarize(records, ridge):
    rows = []
    for record in records:
        methods = {
            'rank1': float((record['residual'][record['fit']].mean() -
                            record['residual'][~record['fit']]).square().mean().sqrt()),
            'M0': _rmse(record['fixed'], record, ridge),
            'M2-O': _rmse(record['m2o'], record, ridge),
            'R2': _rmse(record['r2'], record, ridge),
        }
        fit = record['fit']
        fit_matrix = record['r2'][fit]
        system = fit_matrix.T @ fit_matrix + ridge * torch.eye(fit_matrix.shape[1])
        leverage = torch.diagonal(fit_matrix @ torch.linalg.solve(system, fit_matrix.T))
        rows.append({
            'source': record['source'], 'profile_id': record['profile_id'],
            'fold': record['fold'], 'cell': record['cell'], 'rmse': methods,
            'support_features': record['features'],
            'residual_mean': float(record['residual'].mean()),
            'residual_slope': float(np.polyfit(record['coords'][:, 2].numpy(),
                                               record['residual'].numpy(), 1)[0]),
            'effective_rank_fit': _effective_rank(record['r2'][fit]),
            'effective_rank_holdout': _effective_rank(record['r2'][~fit]),
            'principal_angle_degrees': _principal_angle(record['r2'][fit],
                                                        record['r2'][~fit]),
            'ridge_condition': float(torch.linalg.cond(system)),
            'ridge_leverage_max': float(leverage.max()),
        })
    return rows


def _attribution(rows, matches, gradients, leave_source_out):
    r2_error = {(r['source'], r['profile_id']): r['rmse']['R2'] for r in rows}
    all_gap = np.mean([r['rmse']['R2'] for r in rows if r['source'] == 'COSMIC']) - np.mean(
        [r['rmse']['R2'] for r in rows if r['source'] == 'FY'])
    matched_gap = np.mean([r2_error[('COSMIC', c['profile_id'])] -
                           r2_error[('FY', f['profile_id'])] for f, c, _ in matches]) if matches else math.nan
    reduction = 1.0 - abs(matched_gap) / max(abs(all_gap), 1e-12)
    matched_residuals = [value for pair in matches
                         for record in pair[:2] for value in record['residual'].tolist()]
    pooled_scale = float(np.std(matched_residuals)) if matched_residuals else 0.0
    source_effects = [(float(cosmic['residual'].mean()) - float(fy['residual'].mean())) /
                      pooled_scale for fy, cosmic, _ in matches if pooled_scale > 0]
    angle_ci = _bootstrap_spearman(
        [r['principal_angle_degrees'] for r in rows], [r['rmse']['R2'] for r in rows])
    condition_ci = _bootstrap_spearman(
        [r['ridge_condition'] for r in rows], [r['rmse']['R2'] for r in rows])
    cosines = [value for snapshot in gradients.values()
               for key, value in snapshot['cosines'].items()
               if key.split('|')[0].split('_')[0] != key.split('|')[1].split('_')[0]]
    negative_fraction = float(np.mean(np.asarray(cosines) < 0)) if cosines else 0.0
    norm_ratios = []
    for snapshot in gradients.values():
        norms = [value for value in snapshot['norms'].values() if value > 1e-12]
        if norms:
            norm_ratios.append(max(norms) / min(norms))
    norm_ratio = max(norm_ratios) if norm_ratios else 1.0
    counts = defaultdict(lambda: defaultdict(int))
    for fy, cosmic, _ in matches:
        counts[fy['cell']]['FY'] += 1
        counts[cosmic['cell']]['COSMIC'] += 1
    enough_h4 = bool(counts) and all(min(group.values()) >=
        THRESHOLDS['minimum_matched_profiles_per_major_stratum']
        for group in counts.values() if len(group) == 2)
    shrink = [abs(all_gap) - abs(r2_error[('COSMIC', cosmic['profile_id'])] -
                                  r2_error[('FY', fy['profile_id'])])
              for fy, cosmic, _ in matches]
    shrink_ci = _bootstrap(shrink)
    source_ci = _bootstrap(source_effects)
    h1 = bool(bool(matches) and reduction >= THRESHOLDS['support_gap_reduction_fraction']
              and shrink_ci is not None and shrink_ci['ci95'][0] > 0)
    effects_by_block = defaultdict(list)
    for fy, cosmic, _ in matches:
        effects_by_block[fy['date_block']].append(
            float(cosmic['residual'].mean() - fy['residual'].mean()))
    block_means = [float(np.mean(values)) for values in effects_by_block.values()]
    block_same_direction = (len(block_means) >= 2 and
                            (all(value > 0 for value in block_means) or
                             all(value < 0 for value in block_means)))
    h4 = bool(enough_h4 and source_effects and block_same_direction
              and abs(float(np.mean(source_effects))) >= THRESHOLDS['source_effect_smd']
              and source_ci is not None
              and (source_ci['ci95'][0] > 0 or source_ci['ci95'][1] < 0))
    def stable_gauge(value):
        return (value is not None
                and abs(value['estimate']) >= THRESHOLDS['gauge_spearman_absolute']
                and (value['ci95'][0] > 0 or value['ci95'][1] < 0))
    negative_ci = _bootstrap([float(value < 0) for value in cosines])
    cosine_ci = _bootstrap(cosines)
    h5 = bool(stable_gauge(angle_ci) or stable_gauge(condition_ci))
    low_conflict = bool(
        cosine_ci is not None and negative_ci is not None and
        (cosine_ci['ci95'][0] >= 0 or
         negative_ci['ci95'][1] < THRESHOLDS['gradient_low_conflict_fraction']))
    h2 = bool(low_conflict and norm_ratio > THRESHOLDS['gradient_norm_ratio'])
    means = defaultdict(dict)
    for source in ('FY', 'COSMIC'):
        for label in ('FY-only', 'COSMIC-only', 'pooled'):
            values = [row['rmse'][label] for row in leave_source_out
                      if row['source'] == source]
            means[source][label] = float(np.mean(values)) if values else None
    cross_harm = bool(leave_source_out) and all((
        means['FY']['FY-only'] < means['FY']['pooled'],
        means['COSMIC']['COSMIC-only'] < means['COSMIC']['pooled'],
        means['FY']['COSMIC-only'] > means['FY']['pooled'],
        means['COSMIC']['FY-only'] > means['COSMIC']['pooled'],
    ))
    conflict = bool(cosine_ci is not None and negative_ci is not None and
                    (cosine_ci['ci95'][1] < 0 or
                     negative_ci['ci95'][0] > THRESHOLDS['gradient_conflict_fraction']))
    h3 = bool(conflict
              and cross_harm and not h1)
    ordered = [('H1', h1), ('H4', h4), ('H5', h5), ('H2', h2), ('H3', h3)]
    selected = next((name for name, passed in ordered if passed), 'inconclusive')
    return {
        'selected_hypothesis': selected,
        'progression_to_FiLM_allowed': selected == 'H3',
        'H1': {'passed': h1, 'all_gap': float(all_gap),
               'matched_gap': float(matched_gap), 'gap_reduction_fraction': float(reduction),
               'shrink_bootstrap': shrink_ci},
        'H4': {'passed': bool(h4), 'sample_sufficient': enough_h4,
               'matched_source_effect_mean_smd': float(np.mean(source_effects)) if source_effects else None,
               'source_effect_bootstrap': source_ci,
               'matched_counts': counts, 'date_block_means': block_means,
               'cross_date_blocks_same_direction': block_same_direction},
        'H5': {'passed': h5, 'rho_angle_error': angle_ci,
               'rho_condition_error': condition_ci},
        'H2': {'passed': h2, 'low_conflict': low_conflict,
               'gradient_norm_ratio': float(norm_ratio)},
        'H3': {'passed': h3, 'cross_source_negative_cosine_fraction': negative_fraction,
               'negative_fraction_bootstrap': negative_ci,
               'cosine_mean_bootstrap': cosine_ci, 'gradient_conflict': conflict,
               'leave_source_out_means': means, 'cross_source_harm': cross_harm},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--r2-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-profiles', type=int, default=128)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_dir = args.run_dir.resolve()
    baseline, sw, peak, config = _load(run_dir, run_dir / 'epoch_09_model.pth')
    shadow = _physical_shadow(config, run_dir / 'best_background_model.pth')
    r2 = _load_r2(shadow, args.r2_checkpoint)
    manifest_path = Path(config['date_split_manifest'])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    partitions = json.loads(manifest_path.read_text(encoding='utf-8'))['partitions']
    loaders = {
        'FY': _partition_loader(FY3D_Dataset, config, partitions, 'train'),
        'COSMIC': _partition_loader(COSMICDataset, config, partitions, 'train'),
    }
    selected = {source: _restrict_profiles(loader, args.max_profiles, 542 + index)
                for index, (source, loader) in enumerate(loaders.items())}
    records = []
    for source, profile_id, data in _rows(loaders):
        record = _profile_record(r2, baseline, source, profile_id, data, sw, peak)
        if record is not None:
            records.append(record)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_manifest = {
        'schema_version': 1, 'partition': 'train', 'seed': SEED,
        'locked_test_accessed': False, 'development_accessed': False,
        'isr_accessed': False, 'max_profiles_per_source': args.max_profiles,
        'steps': args.steps, 'lr': args.lr, 'snapshots': SNAPSHOTS,
        'thresholds': THRESHOLDS,
        'selected_profile_ids': {key: value.tolist() for key, value in selected.items()},
        'semantics': {'analysis_state': 'query_local_increment_coefficients',
                      'physical_modes': 'source_shared',
                      'source_labels': 'diagnostics_only'},
    }
    (output / 's0_manifest.json').write_text(
        json.dumps(frozen_manifest, indent=2), encoding='utf-8')
    gradients = _gradient_audit(shadow, records, args.steps, args.lr, sw, peak)
    leave_source_out = _leave_source_out(
        shadow, records, args.steps, args.lr, sw, peak)
    ridge = float(shadow.kalman_layer.r_fy)
    detail = _summarize(records, ridge)
    pairs = _match(records)
    attribution = _attribution(detail, pairs, gradients, leave_source_out)
    report = {
        'manifest': frozen_manifest,
        'profiles': detail,
        'gradient_snapshots': gradients,
        'leave_source_out': leave_source_out,
        'matching': {'pairs': [{'FY': left['profile_id'], 'COSMIC': right['profile_id'],
                                'cell': left['cell'], 'distance': distance}
                               for left, right, distance in pairs]},
        'attribution': attribution,
    }
    temporary = output / 's0_report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output / 's0_report.json')
    (output / 's0_attribution.json').write_text(
        json.dumps(attribution, indent=2, allow_nan=False), encoding='utf-8')
    print(output / 's0_attribution.json')


if __name__ == '__main__':
    main()
