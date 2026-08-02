"""Estimate train-only, profile-blocked Background residual covariance."""

import argparse
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from inr_modules.config_mdia import get_config_mdia
from inr_modules.data_managers.FY_dataloader import (
    COSMICNeighborhoodIndex,
    FYNeighborhoodIndex,
    get_cosmic_dataloader,
    get_dataloaders,
)
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.train_fsia import (
    _load_background_seed,
    _load_iri_peak_manager,
)
from main_fsia import _code_identity, _file_identity


ROOT = Path(__file__).resolve().parent
OUTPUT_DEFAULT = (
    ROOT / 'isr_validation_outputs'
    / 'run66-empirical-covariance-train-only')
BACKGROUND_DEFAULT = (
    ROOT / 'checkpoints_fsia' / 'run66-etkf-loss'
    / 'best_background_model.pth')
FY_PATH = Path(r'D:\FYsatellite\EDP_data\fy_202409_qc_v2.npy')
FY_INDEX_PATH = Path(
    r'D:\FYsatellite\EDP_data\fy_202409_qc_v2_index.npz')
FY_REPORT_PATH = Path(
    r'D:\FYsatellite\EDP_data\fy_202409_qc_v2_report.json')
COSMIC_PATH = Path(
    r'D:\cosmic2\cosmic245-274-September'
    r'\cosmic_september_2024_qc.npy')
COSMIC_INDEX_PATH = Path(
    r'D:\cosmic2\cosmic245-274-September'
    r'\cosmic_september_2024_qc_index.npz')
COSMIC_REPORT_PATH = Path(
    r'D:\cosmic2\cosmic245-274-September'
    r'\cosmic_september_2024_qc_report.json')

SOURCE_NAMES = ('FY', 'COSMIC')
PAIR_NAMES = ('FY->FY', 'FY->COSMIC', 'COSMIC->FY', 'COSMIC->COSMIC')
ALT_NAMES = ('120-200', '200-300', '300-500')
LT_NAMES = ('night-night', 'day-day', 'mixed')
RHO_NAMES = ('0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1')
N_CELLS = len(PAIR_NAMES) * 3 * 3 * 3 * 4
EXPECTED_TRAIN_PROFILES = {'FY': 59958, 'COSMIC': 75958}
EXPECTED_DATE_BLOCKED_TRAIN_PROFILES = {'FY': 44625, 'COSMIC': 56099}

ROW_DTYPE = np.dtype([
    ('target_id', '<i8'),
    ('date', '<i2'),
    ('loc_r', '<f8'),
    ('loc_d', '<f8'),
    ('loc_rd', '<f8'),
    ('loc_r2', '<f8'),
    ('loc_d2', '<f8'),
    ('raw_r', '<f8'),
    ('raw_d', '<f8'),
    ('raw_rd', '<f8'),
    ('raw_r2', '<f8'),
    ('raw_d2', '<f8'),
    ('pair_count', '<i4'),
    ('pair_mass', '<f8'),
])


@dataclass
class ProfileCache:
    source: str
    ids: np.ndarray
    data: np.ndarray
    valid: np.ndarray
    residual: np.ndarray
    date: np.ndarray


def _cell_id(pair, target_alt, observation_alt, lt_class, rho_bin):
    target_alt = np.asarray(target_alt, dtype=np.int16)
    observation_alt = np.asarray(observation_alt, dtype=np.int16)
    lt_class = np.asarray(lt_class, dtype=np.int16)
    rho_bin = np.asarray(rho_bin, dtype=np.int16)
    return ((((int(pair) * 3 + target_alt) * 3 + observation_alt) * 3
             + lt_class) * 4 + rho_bin)


def _decode_cell(cell):
    rho_bin = cell % 4
    cell //= 4
    lt_class = cell % 3
    cell //= 3
    observation_alt = cell % 3
    cell //= 3
    target_alt = cell % 3
    pair = cell // 3
    return pair, target_alt, observation_alt, lt_class, rho_bin


def _altitude_bin(altitude):
    return np.digitize(altitude, [200.0, 300.0]).astype(np.int8)


def _local_time(coords):
    return np.remainder(coords[..., 3] + coords[..., 1] / 15.0, 24.0)


def _localization(rho):
    return 1.0 - 3.0 * rho ** 2 + 2.0 * rho ** 3


def _moment_summary(rows, prefix):
    means = {
        name: float(np.mean(rows[f'{prefix}_{name}']))
        for name in ('r', 'd', 'rd', 'r2', 'd2')
    }
    covariance = means['rd'] - means['r'] * means['d']
    var_r = max(0.0, means['r2'] - means['r'] ** 2)
    var_d = max(0.0, means['d2'] - means['d'] ** 2)
    denominator = np.sqrt(var_r * var_d)
    correlation = (
        float(covariance / denominator) if denominator > 1e-15 else 0.0)
    products = (
        (rows[f'{prefix}_r'] - means['r'])
        * (rows[f'{prefix}_d'] - means['d']))
    direction = (
        float(np.sign(correlation) * np.mean(np.sign(products)))
        if correlation else 0.0)
    return {
        'mean_target': means['r'],
        'mean_observation': means['d'],
        'variance_target': var_r,
        'variance_observation': var_d,
        'covariance': float(covariance),
        'correlation': correlation,
        'directional_excess': direction,
    }


def _bootstrap_cell(rows, replicates, seed):
    """Date-stratified target-profile bootstrap with recomputed centering."""
    dates = np.unique(rows['date'])
    groups = [np.flatnonzero(rows['date'] == day) for day in dates]
    values = np.column_stack([
        rows['loc_r'], rows['loc_d'], rows['loc_rd'],
        rows['loc_r2'], rows['loc_d2'],
    ])
    rng = np.random.default_rng(seed)
    covariance_values = np.empty(replicates, dtype=np.float64)
    correlation_values = np.empty(replicates, dtype=np.float64)
    direction_values = np.empty(replicates, dtype=np.float64)
    total = len(rows)

    for start in range(0, replicates, 20):
        count = min(20, replicates - start)
        sampled_by_date = [
            indices[rng.integers(0, len(indices), size=(count, len(indices)))]
            for indices in groups
        ]
        sums = np.zeros((count, 5), dtype=np.float64)
        for sampled in sampled_by_date:
            sums += values[sampled].sum(axis=1)
        means = sums / total
        covariance = means[:, 2] - means[:, 0] * means[:, 1]
        var_r = np.maximum(0.0, means[:, 3] - means[:, 0] ** 2)
        var_d = np.maximum(0.0, means[:, 4] - means[:, 1] ** 2)
        denominator = np.sqrt(var_r * var_d)
        correlation = np.divide(
            covariance, denominator, out=np.zeros_like(covariance),
            where=denominator > 1e-15)
        signs = np.zeros(count, dtype=np.float64)
        for sampled in sampled_by_date:
            centered = (
                (rows['loc_r'][sampled] - means[:, None, 0])
                * (rows['loc_d'][sampled] - means[:, None, 1]))
            signs += np.sign(centered).sum(axis=1)
        direction = np.sign(correlation) * signs / total
        stop = start + count
        covariance_values[start:stop] = covariance
        correlation_values[start:stop] = correlation
        direction_values[start:stop] = direction

    quantiles = [0.025, 0.975]
    return {
        'covariance_ci95': np.quantile(
            covariance_values, quantiles).tolist(),
        'correlation_ci95': np.quantile(
            correlation_values, quantiles).tolist(),
        'directional_excess_ci95': np.quantile(
            direction_values, quantiles).tolist(),
    }


def _date_sign_stability(rows, full_sign):
    signs = []
    for day in np.unique(rows['date']):
        selected = rows[rows['date'] == day]
        if len(selected) < 2:
            continue
        correlation = _moment_summary(selected, 'loc')['correlation']
        if correlation:
            signs.append(np.sign(correlation) == full_sign)
    return {
        'eligible_dates': len(signs),
        'sign_fraction': float(np.mean(signs)) if signs else None,
    }


def _strict_background(config, checkpoint, device):
    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device,
    )
    iri_proxy = IRINeuralProxy(
        layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(
        config['iri_proxy_path'], map_location=device, weights_only=True))
    iri_proxy.eval()
    iri_peak_manager = _load_iri_peak_manager(config, device)
    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not all(
            torch.isfinite(value).all().item()
            for value in state.values() if torch.is_tensor(value)):
        raise ValueError('Background checkpoint contains non-finite tensors')
    _load_background_seed(model, checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, sw_manager, iri_peak_manager


def _background_residual_cache(
        source, index, allowed_ids, model, sw_manager, iri_peak_manager,
        device, batch_size=4096):
    allowed = np.asarray(allowed_ids, dtype=np.int64)
    selected = np.searchsorted(index.prof_ids, allowed)
    if (np.any(selected >= len(index.prof_ids))
            or not np.array_equal(index.prof_ids[selected], allowed)):
        raise ValueError(f'{source} train profile is missing from neighborhood index')
    data = np.asarray(index.prof_abs_data[selected], dtype=np.float32)
    valid = np.asarray(index.prof_valid_mask[selected], dtype=bool)
    residual = np.full(valid.shape, np.nan, dtype=np.float32)
    flat_data = data.reshape(-1, 5)
    flat_valid = valid.reshape(-1)
    valid_rows = np.flatnonzero(flat_valid)
    with torch.no_grad():
        for start in range(0, len(valid_rows), batch_size):
            rows = valid_rows[start:start + batch_size]
            coords = torch.from_numpy(flat_data[rows, :4]).to(device)
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            iri_peak = (
                iri_peak_manager.get_iri_peak(coords)
                if iri_peak_manager is not None else None)
            background = model.encode_background(
                coords, sw_seq, iri_peak=iri_peak)['ne_bkg']
            residual.reshape(-1)[rows] = (
                flat_data[rows, 4]
                - background.squeeze(-1).cpu().numpy())
    representative_time = np.nanmean(
        np.where(valid, data[..., 3], np.nan), axis=1)
    date = (np.floor(representative_time / 24.0) + 1).astype(np.int16)
    if not np.isfinite(residual[valid]).all():
        raise FloatingPointError(f'{source} residual cache is non-finite')
    return ProfileCache(
        source=source, ids=allowed, data=data, valid=valid,
        residual=residual, date=date)


def _lookup_neighbor_residual(cache, selected_ids):
    safe_ids = np.maximum(selected_ids, 0)
    positions = np.searchsorted(cache.ids, safe_ids)
    present = positions < len(cache.ids)
    clipped = np.minimum(positions, max(0, len(cache.ids) - 1))
    present &= cache.ids[clipped] == safe_ids
    residual = np.zeros(
        selected_ids.shape + (cache.residual.shape[1],), dtype=np.float32)
    residual[present] = cache.residual[clipped[present]]
    return residual, present


def _reduce_profile_pairs(inverse, target_values, observation_values, weights):
    """Reduce tokens to localization-weighted and unweighted profile-pair moments."""
    n_pairs = int(np.max(inverse)) + 1
    weight_sum = np.bincount(inverse, weights=weights, minlength=n_pairs)
    token_count = np.bincount(inverse, minlength=n_pairs).astype(np.float64)
    values = (
        target_values,
        observation_values,
        target_values * observation_values,
        target_values ** 2,
        observation_values ** 2,
    )
    localized = np.column_stack([
        np.bincount(inverse, weights=value * weights, minlength=n_pairs)
        / weight_sum
        for value in values
    ])
    unweighted = np.column_stack([
        np.bincount(inverse, weights=value, minlength=n_pairs)
        / token_count
        for value in values
    ])
    return localized, unweighted, weight_sum / token_count


def _append_profile_rows(
        target, target_indices, observation, observation_index, pair_index,
        rows_by_cell, neighbor_sets, pair_counts, pair_masses):
    profile_data = target.data[target_indices]
    profile_valid = target.valid[target_indices]
    profile_residual = target.residual[target_indices]
    profile_ids = target.ids[target_indices]
    point_valid = profile_valid.reshape(-1)
    point_data = profile_data.reshape(-1, 5)[point_valid]
    point_residual = profile_residual.reshape(-1)[point_valid]
    point_profile = np.repeat(
        profile_ids, profile_valid.shape[1])[point_valid]
    exclude = (
        point_profile if target.source == observation.source else None)
    cached = observation_index.query_profiles_only(
        point_data[:, :4], exclude_profile_ids=exclude,
        allowed_profile_ids=observation.ids)
    neighbor_residual, present = _lookup_neighbor_residual(
        observation, cached['sel_ids'])
    if not np.all(present[cached['valid_prof']]):
        raise AssertionError('top-K returned a profile outside train-only cache')
    selected_data = cached['sel_abs']
    token_valid = (
        cached['valid_prof'][..., None] & cached['sel_vmask']
        & present[..., None] & np.isfinite(neighbor_residual))
    payload = observation_index.observation_payload_from_cached(cached)
    rho = payload['rho_squared'].reshape(token_valid.shape)
    rho = np.sqrt(np.maximum(rho, 0.0))
    localization = _localization(rho)
    token_valid &= (rho < 1.0) & (localization > 0.0)
    if not token_valid.any():
        return

    shape = token_valid.shape
    target_grid = np.broadcast_to(
        point_residual[:, None, None], shape)
    target_id_grid = np.broadcast_to(
        point_profile[:, None, None], shape)
    neighbor_id_grid = np.broadcast_to(
        cached['sel_ids'][..., None], shape)
    target_alt = np.broadcast_to(
        _altitude_bin(point_data[:, 2])[:, None, None], shape)
    observation_alt = _altitude_bin(selected_data[..., 2])
    target_day = np.broadcast_to(
        ((_local_time(point_data) >= 6.0)
         & (_local_time(point_data) < 18.0))[:, None, None], shape)
    observation_lt = _local_time(selected_data)
    observation_day = (observation_lt >= 6.0) & (observation_lt < 18.0)
    lt_class = np.where(
        target_day & observation_day, 1,
        np.where(~target_day & ~observation_day, 0, 2)).astype(np.int8)
    rho_bin = np.minimum((rho * 4.0).astype(np.int8), 3)
    cell = _cell_id(
        pair_index, target_alt, observation_alt, lt_class, rho_bin)

    selected = token_valid.reshape(-1)
    target_values = target_grid.reshape(-1)[selected].astype(np.float64)
    observation_values = neighbor_residual.reshape(-1)[selected].astype(
        np.float64)
    weights = localization.reshape(-1)[selected].astype(np.float64)
    target_ids = target_id_grid.reshape(-1)[selected]
    neighbor_ids = neighbor_id_grid.reshape(-1)[selected]
    cells = cell.reshape(-1)[selected].astype(np.int16)

    pair_keys = np.rec.fromarrays(
        [target_ids, neighbor_ids, cells], names='target,neighbor,cell')
    unique_pairs, inverse = np.unique(pair_keys, return_inverse=True)
    localized_pair, raw_pair, pair_quality = _reduce_profile_pairs(
        inverse, target_values, observation_values, weights)

    for cell_value in np.unique(unique_pairs['cell']):
        selected_pairs = unique_pairs['cell'] == cell_value
        neighbor_sets[int(cell_value)].update(
            unique_pairs['neighbor'][selected_pairs].tolist())
        pair_counts[int(cell_value)] += int(selected_pairs.sum())
        pair_masses[int(cell_value)] += float(
            pair_quality[selected_pairs].sum())

    target_cell_keys = np.rec.fromarrays(
        [unique_pairs['target'], unique_pairs['cell']],
        names='target,cell')
    unique_target_cells, target_inverse = np.unique(
        target_cell_keys, return_inverse=True)
    n_target_cells = len(unique_target_cells)
    neighbor_count = np.bincount(
        target_inverse, minlength=n_target_cells).astype(np.float64)
    localized = np.column_stack([
        np.bincount(
            target_inverse, weights=localized_pair[:, column],
            minlength=n_target_cells) / neighbor_count
        for column in range(5)
    ])
    raw = np.column_stack([
        np.bincount(
            target_inverse, weights=raw_pair[:, column],
            minlength=n_target_cells) / neighbor_count
        for column in range(5)
    ])
    mass = np.bincount(
        target_inverse, weights=pair_quality, minlength=n_target_cells)
    positions = np.searchsorted(
        target.ids, unique_target_cells['target'])
    dates = target.date[positions]

    for cell_value in np.unique(unique_target_cells['cell']):
        chosen = np.flatnonzero(unique_target_cells['cell'] == cell_value)
        block = np.empty(len(chosen), dtype=ROW_DTYPE)
        block['target_id'] = unique_target_cells['target'][chosen]
        block['date'] = dates[chosen]
        for column, name in enumerate(('r', 'd', 'rd', 'r2', 'd2')):
            block[f'loc_{name}'] = localized[chosen, column]
            block[f'raw_{name}'] = raw[chosen, column]
        block['pair_count'] = neighbor_count[chosen].astype(np.int32)
        block['pair_mass'] = mass[chosen]
        rows_by_cell[int(cell_value)].append(block)


def _cell_report(
        cell, rows, neighbor_count, pair_count, pair_mass, replicates, seed):
    pair, target_alt, observation_alt, lt_class, rho_bin = _decode_cell(cell)
    localized = _moment_summary(rows, 'loc')
    raw = _moment_summary(rows, 'raw')
    date_stability = _date_sign_stability(
        rows, np.sign(localized['correlation']))
    estimable = (
        len(rows) >= 200
        and neighbor_count >= 200
        and pair_count >= 1000
        and date_stability['eligible_dates'] >= 15)
    bootstrap = (
        _bootstrap_cell(rows, replicates, seed + cell)
        if estimable else {
            'covariance_ci95': None,
            'correlation_ci95': None,
            'directional_excess_ci95': None,
        })
    correlation_ci = bootstrap['correlation_ci95']
    direction_ci = bootstrap['directional_excess_ci95']
    stable = bool(
        estimable
        and abs(localized['correlation']) >= 0.10
        and correlation_ci[0] * correlation_ci[1] > 0.0
        and date_stability['sign_fraction'] >= 0.80
        and direction_ci[0] > 0.0
        and localized['correlation'] * raw['correlation'] > 0.0)
    return {
        'cell_id': cell,
        'pair': PAIR_NAMES[pair],
        'target_altitude': ALT_NAMES[target_alt],
        'observation_altitude': ALT_NAMES[observation_alt],
        'local_time': LT_NAMES[lt_class],
        'rho': RHO_NAMES[rho_bin],
        'target_profiles': int(len(rows)),
        'neighbor_profiles': int(neighbor_count),
        'ordered_profile_pairs': int(pair_count),
        'localization_pair_mass': float(pair_mass),
        'eligible_dates': date_stability['eligible_dates'],
        'date_sign_fraction': date_stability['sign_fraction'],
        'localized': localized,
        'unweighted': raw,
        **bootstrap,
        'estimable': estimable,
        'stable': stable,
    }


def _cross_source_gate(cells):
    by_key = {
        (
            PAIR_NAMES.index(cell['pair']),
            ALT_NAMES.index(cell['target_altitude']),
            ALT_NAMES.index(cell['observation_altitude']),
            LT_NAMES.index(cell['local_time']),
            RHO_NAMES.index(cell['rho']),
        ): cell
        for cell in cells
    }
    comparisons = []
    for target_alt in range(3):
        for observation_alt in range(3):
            for lt_class in range(3):
                for rho_bin in range(4):
                    fy_cosmic = by_key[
                        (1, target_alt, observation_alt, lt_class, rho_bin)]
                    cosmic_fy = by_key[
                        (2, observation_alt, target_alt, lt_class, rho_bin)]
                    if not (
                            fy_cosmic['estimable']
                            and cosmic_fy['estimable']):
                        continue
                    mass = np.sqrt(
                        fy_cosmic['localization_pair_mass']
                        * cosmic_fy['localization_pair_mass'])
                    same_sign = (
                        fy_cosmic['localized']['correlation']
                        * cosmic_fy['localized']['correlation'] > 0.0)
                    joint_stable = bool(
                        same_sign and fy_cosmic['stable']
                        and cosmic_fy['stable'])
                    comparisons.append({
                        'fy_to_cosmic_cell': fy_cosmic['cell_id'],
                        'cosmic_to_fy_cell': cosmic_fy['cell_id'],
                        'mass': float(mass),
                        'same_sign': same_sign,
                        'joint_stable': joint_stable,
                    })
    total_mass = sum(item['mass'] for item in comparisons)
    sign_mass = sum(
        item['mass'] for item in comparisons if item['same_sign'])
    stable_mass = sum(
        item['mass'] for item in comparisons if item['joint_stable'])
    sign_fraction = sign_mass / total_mass if total_mass else 0.0
    stable_mass_fraction = stable_mass / total_mass if total_mass else 0.0
    stable_cell_fraction = (
        sum(item['joint_stable'] for item in comparisons) / len(comparisons)
        if comparisons else 0.0)

    diagonal = []
    for altitude in range(2):
        for rho_bin in range(2):
            forward = by_key[(1, altitude, altitude, 0, rho_bin)]
            reverse = by_key[(2, altitude, altitude, 0, rho_bin)]
            passed = bool(
                forward['estimable'] and reverse['estimable']
                and forward['correlation_ci95'][0] > 0.0
                and reverse['correlation_ci95'][0] > 0.0)
            diagonal.append({
                'altitude': ALT_NAMES[altitude],
                'rho': RHO_NAMES[rho_bin],
                'fy_to_cosmic_cell': forward['cell_id'],
                'cosmic_to_fy_cell': reverse['cell_id'],
                'passed': passed,
            })

    gates = {
        'transposed_sign_agreement_at_least_0.80': sign_fraction >= 0.80,
        'joint_stable_mass_at_least_0.60': stable_mass_fraction >= 0.60,
        'joint_stable_cells_at_least_0.50': stable_cell_fraction >= 0.50,
        'positive_low_night_diagonal': all(
            item['passed'] for item in diagonal),
    }
    passed = all(gates.values())
    strong_disagreement = bool(
        comparisons and sign_fraction < 0.80
        and sum(
            cells[index]['stable']
            for item in comparisons
            for index in (
                item['fy_to_cosmic_cell'], item['cosmic_to_fy_cell']))
        >= len(comparisons))
    if passed:
        conclusion = '跨源稳定结构已确认，可规划基于单元经验协方差的监督'
    elif strong_disagreement:
        conclusion = '来源间结构冲突，优先定位FY/COSMIC残差偏差或代表性问题'
    else:
        conclusion = '经验结构弱或不稳定，停止协方差监督'
    return {
        'comparisons': comparisons,
        'transposed_sign_agreement': float(sign_fraction),
        'joint_stable_mass_fraction': float(stable_mass_fraction),
        'joint_stable_cell_fraction': float(stable_cell_fraction),
        'positive_diagonal': diagonal,
        'gates': gates,
        'passed': passed,
        'conclusion': conclusion,
    }


def _deterministic_npz(path, arrays):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with zipfile.ZipFile(
            temporary, mode='w', compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f'{name}.npy')
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, buffer.getvalue())
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_outputs(output, report, cells):
    output.mkdir(parents=True, exist_ok=False)
    npz_path = output / 'empirical_covariance_cells.npz'
    json_path = output / 'empirical_covariance_report.json'
    arrays = {
        'cell_id': np.arange(N_CELLS, dtype=np.int16),
        'target_profiles': np.asarray([
            cell['target_profiles'] for cell in cells], dtype=np.int32),
        'neighbor_profiles': np.asarray([
            cell['neighbor_profiles'] for cell in cells], dtype=np.int32),
        'ordered_profile_pairs': np.asarray([
            cell['ordered_profile_pairs'] for cell in cells], dtype=np.int64),
        'localization_pair_mass': np.asarray([
            cell['localization_pair_mass'] for cell in cells], dtype=np.float64),
        'eligible_dates': np.asarray([
            cell['eligible_dates'] for cell in cells], dtype=np.int16),
        'date_sign_fraction': np.asarray([
            np.nan if cell['date_sign_fraction'] is None
            else cell['date_sign_fraction'] for cell in cells],
            dtype=np.float64),
        'estimable': np.asarray([
            cell['estimable'] for cell in cells], dtype=bool),
        'stable': np.asarray([
            cell['stable'] for cell in cells], dtype=bool),
    }
    arrays['covariance'] = np.asarray([
        np.nan if cell['localized'] is None
        else cell['localized']['covariance'] for cell in cells])
    arrays['correlation'] = np.asarray([
        np.nan if cell['localized'] is None
        else cell['localized']['correlation'] for cell in cells])
    arrays['unweighted_covariance'] = np.asarray([
        np.nan if cell['unweighted'] is None
        else cell['unweighted']['covariance'] for cell in cells])
    arrays['unweighted_correlation'] = np.asarray([
        np.nan if cell['unweighted'] is None
        else cell['unweighted']['correlation'] for cell in cells])
    arrays['correlation_ci95'] = np.asarray([
        cell['correlation_ci95'] if cell['correlation_ci95'] is not None
        else [np.nan, np.nan] for cell in cells])
    arrays['directional_excess_ci95'] = np.asarray([
        cell['directional_excess_ci95']
        if cell['directional_excess_ci95'] is not None
        else [np.nan, np.nan] for cell in cells])
    _deterministic_npz(npz_path, arrays)
    report['npz_sha256'] = _sha256(npz_path)
    temporary = json_path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False) + '\n',
        encoding='utf-8')
    os.replace(temporary, json_path)
    return {
        'npz': report['npz_sha256'],
        'json': _sha256(json_path),
    }


def _configuration():
    config = dict(get_config_mdia())
    config.update({
        'fy_path': str(FY_PATH),
        'fy_profile_path': None,
        'fy_profile_index_path': str(FY_INDEX_PATH),
        'cosmic_path': str(COSMIC_PATH),
        'cosmic_profile_index_path': str(COSMIC_INDEX_PATH),
        'device': 'cpu',
        'seed': 42,
        'r_mode': 'global',
        'use_distance_localization': True,
    })
    return config


def run(output, checkpoint, bootstrap=1000, max_profiles=0,
        date_split_manifest=None):
    if bootstrap < 1 or max_profiles < 0:
        raise ValueError('bootstrap must be positive and max-profiles non-negative')
    required = (
        FY_PATH, FY_INDEX_PATH, FY_REPORT_PATH,
        COSMIC_PATH, COSMIC_INDEX_PATH, COSMIC_REPORT_PATH, checkpoint)
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f'missing required inputs: {missing}')
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cpu')
    config = _configuration()
    model, sw_manager, iri_peak_manager = _strict_background(
        config, checkpoint, device)
    split_days = None
    split_identity = None
    if date_split_manifest is not None:
        date_split_manifest = Path(date_split_manifest).resolve()
        with date_split_manifest.open(encoding='utf-8') as stream:
            split_document = json.load(stream)
        split_days = split_document['partitions']
        if set(split_days) != {'train', 'development', 'locked_test'}:
            raise ValueError('date split manifest partitions are incomplete')
        split_identity = _file_identity(date_split_manifest)

    loader_kwargs = {
        'batch_size': min(int(config['batch_size']), 512),
        'bin_size_hours': config['bin_size_hours'],
        'num_workers': 0,
        'use_memmap': True,
        'val_ratio': None if split_days is not None else 0.1,
        'split_seed': 42,
        'points_per_profile': 8,
        'split_days': split_days,
    }
    fy_train, fy_validation = get_dataloaders(
        config['fy_path'], profile_path=None,
        profile_index_path=config['fy_profile_index_path'], **loader_kwargs)
    cosmic_train, cosmic_validation = get_cosmic_dataloader(
        config['cosmic_path'],
        profile_index_path=config['cosmic_profile_index_path'],
        **loader_kwargs)
    train_ids = {
        'FY': np.unique(fy_train.dataset.profile_ids),
        'COSMIC': np.unique(cosmic_train.dataset.profile_ids),
    }
    counts = {source: len(ids) for source, ids in train_ids.items()}
    expected_counts = (
        EXPECTED_DATE_BLOCKED_TRAIN_PROFILES
        if split_days is not None else EXPECTED_TRAIN_PROFILES)
    if counts != expected_counts:
        raise RuntimeError(
            f'train profile identity changed: {counts} != '
            f'{expected_counts}')
    validation_ids = {
        'FY': np.unique(fy_validation.dataset.profile_ids),
        'COSMIC': np.unique(cosmic_validation.dataset.profile_ids),
    }
    for source in SOURCE_NAMES:
        overlap = np.intersect1d(
            train_ids[source], validation_ids[source], assume_unique=True)
        if len(overlap):
            raise AssertionError(
                f'{source} train/validation profile overlap: {len(overlap)}')
    if max_profiles:
        rng = np.random.default_rng(42)
        for source in SOURCE_NAMES:
            if len(train_ids[source]) > max_profiles:
                train_ids[source] = np.sort(rng.choice(
                    train_ids[source], max_profiles, replace=False))

    indices = {
        'FY': FYNeighborhoodIndex(config['fy_path'], config),
        'COSMIC': COSMICNeighborhoodIndex(config['cosmic_path'], config),
    }
    caches = {}
    for source in SOURCE_NAMES:
        print(f'[{source}] 计算 {len(train_ids[source]):,} 条profile背景残差...')
        caches[source] = _background_residual_cache(
            source, indices[source], train_ids[source], model,
            sw_manager, iri_peak_manager, device)

    rows_by_cell = [[] for _ in range(N_CELLS)]
    neighbor_sets = [set() for _ in range(N_CELLS)]
    pair_counts = np.zeros(N_CELLS, dtype=np.int64)
    pair_masses = np.zeros(N_CELLS, dtype=np.float64)
    for target_index, target_source in enumerate(SOURCE_NAMES):
        target = caches[target_source]
        for observation_index, observation_source in enumerate(SOURCE_NAMES):
            pair = target_index * 2 + observation_index
            print(f'[{PAIR_NAMES[pair]}] 构造profile-blocked充分统计量...')
            for start in range(0, len(target.ids), 64):
                selected = np.arange(
                    start, min(start + 64, len(target.ids)))
                _append_profile_rows(
                    target, selected, caches[observation_source],
                    indices[observation_source], pair, rows_by_cell,
                    neighbor_sets, pair_counts, pair_masses)

    cells = []
    for cell in range(N_CELLS):
        rows = (
            np.concatenate(rows_by_cell[cell])
            if rows_by_cell[cell] else np.empty(0, dtype=ROW_DTYPE))
        if len(rows):
            if len(np.unique(rows['target_id'])) != len(rows):
                raise AssertionError(
                    f'cell {cell} contains repeated target profile rows')
            cells.append(_cell_report(
                cell, rows, len(neighbor_sets[cell]),
                int(pair_counts[cell]), float(pair_masses[cell]),
                bootstrap, 42))
        else:
            pair, target_alt, observation_alt, lt_class, rho_bin = (
                _decode_cell(cell))
            cells.append({
                'cell_id': cell,
                'pair': PAIR_NAMES[pair],
                'target_altitude': ALT_NAMES[target_alt],
                'observation_altitude': ALT_NAMES[observation_alt],
                'local_time': LT_NAMES[lt_class],
                'rho': RHO_NAMES[rho_bin],
                'target_profiles': 0,
                'neighbor_profiles': 0,
                'ordered_profile_pairs': 0,
                'localization_pair_mass': 0.0,
                'eligible_dates': 0,
                'date_sign_fraction': None,
                'localized': None,
                'unweighted': None,
                'covariance_ci95': None,
                'correlation_ci95': None,
                'directional_excess_ci95': None,
                'estimable': False,
                'stable': False,
            })
    gate = _cross_source_gate(cells)
    identities = {
        'fy_data': _file_identity(FY_PATH),
        'fy_index': _file_identity(FY_INDEX_PATH),
        'fy_qc_report': _file_identity(FY_REPORT_PATH),
        'cosmic_data': _file_identity(COSMIC_PATH),
        'cosmic_index': _file_identity(COSMIC_INDEX_PATH),
        'cosmic_qc_report': _file_identity(COSMIC_REPORT_PATH),
        'background_checkpoint': _file_identity(checkpoint),
    }
    if split_identity is not None:
        identities['date_split_manifest'] = split_identity
    report = {
        'schema_version': 1,
        'seed': 42,
        'bootstrap_replicates': bootstrap,
        'mode': 'full' if not max_profiles else 'smoke',
        'max_profiles_per_source': max_profiles,
        'train_profile_counts_before_smoke_limit': counts,
        'validation_profile_counts': {
            source: len(ids) for source, ids in validation_ids.items()},
        'train_validation_profile_overlap': {'FY': 0, 'COSMIC': 0},
        'split_mode': (
            'date_blocked_train' if split_days is not None
            else 'legacy_random_profile'),
        'analyzed_profile_counts': {
            source: len(caches[source].ids) for source in SOURCE_NAMES},
        'window': {
            'hours': 1.5, 'latitude_degrees': 5.0,
            'longitude_degrees': 15.0, 'top_profiles': 8,
            'points_per_profile': 8,
        },
        'cell_definitions': {
            'source_pairs': PAIR_NAMES,
            'altitude_km': ALT_NAMES,
            'local_time': LT_NAMES,
            'rho': RHO_NAMES,
        },
        'estimability': {
            'target_profiles_min': 200,
            'neighbor_profiles_min': 200,
            'ordered_profile_pairs_min': 1000,
            'eligible_dates_min': 15,
        },
        'stability': {
            'absolute_correlation_min': 0.10,
            'date_sign_fraction_min': 0.80,
            'bootstrap_ci_excludes_zero': True,
            'directional_excess_ci_lower_positive': True,
            'weighted_unweighted_same_sign': True,
        },
        'input_identity': identities,
        'code_identity': _code_identity(ROOT),
        'cells': cells,
        'cross_source_gate': gate,
        'conclusion': gate['conclusion'],
        'limitations': [
            'Background residual includes background and observation error.',
            'Bootstrap conditions on the reused neighbor-profile database.',
            'Same-source profile-correlated errors are diagnostic only.',
        ],
    }
    hashes = _write_outputs(Path(output), report, cells)
    print(json.dumps({
        'output': str(Path(output).resolve()),
        'hashes': hashes,
        'conclusion': gate['conclusion'],
        'gates': gate['gates'],
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        '--background-checkpoint', type=Path, default=BACKGROUND_DEFAULT)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument(
        '--max-profiles-per-source', type=int, default=0,
        help='smoke-only cap; zero uses every train profile')
    parser.add_argument(
        '--date-split-manifest', type=Path,
        help='use only the train partition from this UTC-date manifest')
    args = parser.parse_args()
    run(
        args.output.resolve(), args.background_checkpoint.resolve(),
        bootstrap=args.bootstrap,
        max_profiles=args.max_profiles_per_source,
        date_split_manifest=args.date_split_manifest)


if __name__ == '__main__':
    main()
