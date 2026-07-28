"""Two-stage run66 training for the feature-space ETKF model."""

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from ..config_mdia import get_config_mdia
    from .fsia_model import FSIA_INR_Model
    from .physics_losses_mdia import profile_huber_loss, second_difference_loss
    from .sliding_dataset import (
        SlidingWindowBatchProcessor,
        attach_observation_background,
        query_observation_payload,
    )
    from .plotting import plot_training_curves
except ImportError:
    from config_mdia import get_config_mdia
    from fsia_model import FSIA_INR_Model
    from physics_losses_mdia import profile_huber_loss, second_difference_loss
    from sliding_dataset import (
        SlidingWindowBatchProcessor,
        attach_observation_background,
        query_observation_payload,
    )
    from plotting import plot_training_curves

from data_managers import SpaceWeatherManager, IRINeuralProxy
from data_managers.FY_dataloader import (
    FYNeighborhoodIndex,
    COSMICNeighborhoodIndex,
    get_dataloaders,
    get_cosmic_dataloader,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(state, path):
    temp_path = f'{path}.tmp'
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def _query_observations(index, coords, profile_ids=None):
    if index is None:
        return None
    exclude = None if profile_ids is None else profile_ids.detach().cpu().numpy()
    return query_observation_payload(
        index, coords, coords.device, exclude_profile_ids=exclude)


def _unpack_source_batch(batch, device):
    data, _, profile_ids = batch
    data = data.to(device, non_blocking=True)
    return data[:, :4], data[:, 4:5], profile_ids.to(device, non_blocking=True)


def _apply_source_dropout(fy_obs, cosmic_obs, profile_ids, probabilities):
    p_m10, p_m01, p_m11 = probabilities
    if not np.isclose(p_m10 + p_m01 + p_m11, 1.0):
        raise ValueError('source_dropout probabilities must sum to one')
    _, inverse = torch.unique(
        profile_ids.flatten(), sorted=False, return_inverse=True)
    reference = fy_obs if fy_obs is not None else cosmic_obs
    draw = torch.rand(
        inverse.max().item() + 1, device=profile_ids.device,
        dtype=reference['value'].dtype)
    draw = draw[inverse]
    keep_fy = (draw < p_m10) | (draw >= p_m10 + p_m01)
    keep_cosmic = draw >= p_m10
    def apply(payload, keep):
        if payload is None:
            return None
        result = dict(payload)
        result['valid_mask'] = payload['valid_mask'] & keep.unsqueeze(-1)
        return result
    return apply(fy_obs, keep_fy), apply(cosmic_obs, keep_cosmic)


def _set_training_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == 'background':
        modules = [
            model.iri_align_net,
            model.background_decoder,
            model.sw_encoder,
        ]
        if model.use_sw_freq:
            modules.extend([model.sw_freq_branch, model.sw_gate])
    elif stage == 'analysis':
        modules = [
            model.kalman_layer,
            model.density_basis_decoder,
        ]
    else:
        raise ValueError(f'unknown training stage: {stage}')
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.iri_proxy.freeze()


def _set_stage_mode(model, stage):
    model.train()
    model.iri_proxy.eval()
    if stage == 'analysis':
        model.iri_align_net.eval()
        model.background_decoder.eval()
        model.sw_encoder.eval()
        if model.use_sw_freq:
            model.sw_freq_branch.eval()
            model.sw_gate.eval()


def _make_optimizer(model, config, stage_epochs):
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = None
    if config.get('scheduler_type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, stage_epochs),
            eta_min=config.get('min_lr', 1e-6),
        )
    return optimizer, scheduler


def _source_forward(model, batch_processor, coords, sw_seq, iri_peak,
                    stage, target_source, profile_ids, apply_dropout, config,
                    iri_peak_manager=None):
    if stage == 'background':
        fy_obs = cosmic_obs = None
    elif target_source == 'FY':
        fy_obs = _query_observations(
            batch_processor.fy_nb_index, coords, profile_ids)
        cosmic_obs = _query_observations(
            batch_processor.cosmic_nb_index, coords)
    else:
        fy_obs = _query_observations(batch_processor.fy_nb_index, coords)
        cosmic_obs = _query_observations(
            batch_processor.cosmic_nb_index, coords, profile_ids)

    if stage == 'analysis' and apply_dropout:
        fy_obs, cosmic_obs = _apply_source_dropout(
            fy_obs, cosmic_obs, profile_ids, config['source_dropout'])
    if stage == 'analysis':
        fy_obs = attach_observation_background(
            fy_obs, model, batch_processor.sw_manager, iri_peak_manager)
        cosmic_obs = attach_observation_background(
            cosmic_obs, model, batch_processor.sw_manager, iri_peak_manager)

    return model(
        coords,
        sw_seq,
        iri_peak=iri_peak,
        observations_fy=fy_obs,
        observations_cosmic=cosmic_obs,
    )


def _structure_losses(model, batch_processor, sw_manager, iri_peak_manager,
                      coords, profile_ids, stage, config):
    n = min(config.get('structure_batch_size', 32), len(coords))
    if n == 0:
        zero = coords.new_zeros(())
        return zero, zero
    chosen = torch.randperm(len(coords), device=coords.device)[:n]
    centers = coords[chosen, :4].clone()
    selected_ids = profile_ids[chosen]
    residual_key = 'background_residual' if stage == 'background' else 'ne_residual'

    def evaluate(triplet):
        flat = triplet.reshape(-1, 4)
        repeated_ids = selected_ids.repeat_interleave(3)
        sw_seq = sw_manager.get_drivers_sequence(flat[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(flat)
                    if iri_peak_manager is not None else None)
        _, _, _, _, extras = _source_forward(
            model, batch_processor, flat, sw_seq, iri_peak,
            stage, 'FY', repeated_ids, False, config, iri_peak_manager)
        return extras[residual_key].reshape(n, 3)

    alt_step = config.get('structure_alt_step_km', 20.0)
    centers[:, 2].clamp_(
        model.alt_min + alt_step, model.alt_max - alt_step)
    vertical = centers[:, None, :].repeat(1, 3, 1)
    vertical[:, 0, 2] = centers[:, 2] - alt_step
    vertical[:, 2, 2] = centers[:, 2] + alt_step

    time_step = config.get('structure_time_step_hours', 1.0)
    centers[:, 3].clamp_(time_step, 720.0 - time_step)
    temporal = centers[:, None, :].repeat(1, 3, 1)
    temporal[:, 0, 3] = centers[:, 3] - time_step
    temporal[:, 2, 3] = centers[:, 3] + time_step

    beta = config.get('structure_huber_beta', 0.05)
    return (
        second_difference_loss(evaluate(vertical), beta),
        second_difference_loss(evaluate(temporal), beta),
    )


def _gradient_norms(data_loss, auxiliary_loss, parameters):
    data_grads = torch.autograd.grad(
        data_loss, parameters, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(
        auxiliary_loss, parameters, retain_graph=True, allow_unused=True)

    def norm(grads):
        finite = [grad.square().sum() for grad in grads if grad is not None]
        return torch.sqrt(torch.stack(finite).sum()) if finite else data_loss.new_zeros(())

    data_norm = norm(data_grads)
    auxiliary_norm = norm(aux_grads)
    ratio = auxiliary_norm / data_norm.clamp_min(1e-12)
    return data_norm.detach(), auxiliary_norm.detach(), ratio.detach()


def train_one_epoch(model, train_loader, batch_processor, optimizer, device,
                    config, epoch, stage, scaler, sw_manager, iri_peak_manager,
                    cosmic_train_loader):
    _set_stage_mode(model, stage)
    use_amp = config.get('use_amp', False) and scaler is not None
    delta = config.get('huber_delta', 0.2)
    cosmic_iter = iter(cosmic_train_loader)
    stats = {key: 0.0 for key in (
        'total', 'observation', 'fy_obs', 'cosmic_obs', 'iri', 'increment',
        'vertical', 'time', 'weighted_iri', 'weighted_increment',
        'weighted_vertical', 'weighted_time', 'gradient_ratio',
        'K_FY_mean', 'K_COSMIC_mean', 'innov_FY_norm',
        'innov_COSMIC_norm', 'inflation', 'ne_delta_abs',
    )}
    diagnostics = []
    audited_batches = 0
    started = time.time()

    for batch_idx, fy_batch in enumerate(train_loader):
        coords, target, profile_ids = _unpack_source_batch(fy_batch, device)
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)

        try:
            cosmic_batch = next(cosmic_iter)
        except StopIteration:
            cosmic_iter = iter(cosmic_train_loader)
            cosmic_batch = next(cosmic_iter)
        cosmic_coords, cosmic_target, cosmic_ids = _unpack_source_batch(
            cosmic_batch, device)
        cosmic_sw = sw_manager.get_drivers_sequence(cosmic_coords[:, 3])
        cosmic_peak = (iri_peak_manager.get_iri_peak(cosmic_coords)
                       if iri_peak_manager is not None else None)

        with torch.amp.autocast('cuda', enabled=use_amp):
            fy_pred, _, _, _, fy_extras = _source_forward(
                model, batch_processor, coords, sw_seq, iri_peak,
                stage, 'FY', profile_ids, stage == 'analysis', config,
                iri_peak_manager)
            cosmic_pred, _, _, _, cosmic_extras = _source_forward(
                model, batch_processor, cosmic_coords, cosmic_sw, cosmic_peak,
                stage, 'COSMIC', cosmic_ids, stage == 'analysis', config,
                iri_peak_manager)
            fy_loss = profile_huber_loss(
                fy_pred, target, profile_ids, delta=delta)
            cosmic_loss = profile_huber_loss(
                cosmic_pred, cosmic_target, cosmic_ids, delta=delta)
            observation_loss = 0.5 * (fy_loss + cosmic_loss)

            vertical_loss, time_loss = _structure_losses(
                model, batch_processor, sw_manager, iri_peak_manager,
                coords, profile_ids, stage, config)

            if stage == 'background':
                fy_iri = profile_huber_loss(
                    fy_extras['ne_bkg'], fy_extras['ne_iri'],
                    profile_ids, delta=delta)
                cosmic_iri = profile_huber_loss(
                    cosmic_extras['ne_bkg'], cosmic_extras['ne_iri'],
                    cosmic_ids, delta=delta)
                iri_loss = 0.5 * (fy_iri + cosmic_iri)
                increment_loss = observation_loss.new_zeros(())
                weighted_iri = config.get('w_iri', 0.02) * iri_loss
                weighted_increment = increment_loss
                weighted_vertical = (
                    config.get('w_vertical_background', 0.02) * vertical_loss)
                weighted_time = (
                    config.get('w_time_background', 0.01) * time_loss)
                auxiliary_loss = (
                    weighted_iri + weighted_vertical + weighted_time
                )
            else:
                fy_increment = F.smooth_l1_loss(
                    fy_extras['ne_residual'],
                    torch.zeros_like(fy_extras['ne_residual']), beta=delta)
                cosmic_increment = F.smooth_l1_loss(
                    cosmic_extras['ne_residual'],
                    torch.zeros_like(cosmic_extras['ne_residual']), beta=delta)
                increment_loss = 0.5 * (fy_increment + cosmic_increment)
                iri_loss = observation_loss.new_zeros(())
                weighted_iri = iri_loss
                weighted_increment = (
                    config.get('w_increment', 0.01) * increment_loss)
                weighted_vertical = (
                    config.get('w_vertical_analysis', 0.05) * vertical_loss)
                weighted_time = (
                    config.get('w_time_analysis', 0.02) * time_loss)
                auxiliary_loss = (
                    weighted_increment + weighted_vertical + weighted_time
                )
            total_loss = observation_loss + auxiliary_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f'non-finite {stage} loss at epoch={epoch + 1}, batch={batch_idx}')

        decoder = (model.background_decoder if stage == 'background'
                   else model.density_basis_decoder)
        first_stage_epoch = (
            epoch == 0 if stage == 'background'
            else epoch == int(config['background_epochs']))
        audit_batch = first_stage_epoch and batch_idx < 100
        if audit_batch:
            observation_grad, auxiliary_grad, ratio = _gradient_norms(
                observation_loss, auxiliary_loss,
                [p for p in decoder.parameters() if p.requires_grad])
            audited_batches += 1
        else:
            observation_grad = total_loss.new_zeros(())
            auxiliary_grad = total_loss.new_zeros(())
            ratio = total_loss.new_zeros(())

        if audit_batch:
            diagnostics.append({
                'epoch': epoch + 1,
                'stage': stage,
                'batch': batch_idx + 1,
                'fy_obs_raw': fy_loss.item(),
                'cosmic_obs_raw': cosmic_loss.item(),
                'fy_obs_weighted': 0.5 * fy_loss.item(),
                'cosmic_obs_weighted': 0.5 * cosmic_loss.item(),
                'iri_raw': iri_loss.item(),
                'increment_raw': increment_loss.item(),
                'vertical_raw': vertical_loss.item(),
                'time_raw': time_loss.item(),
                'iri_weighted': weighted_iri.item(),
                'increment_weighted': weighted_increment.item(),
                'vertical_weighted': weighted_vertical.item(),
                'time_weighted': weighted_time.item(),
                'observation_grad_norm': observation_grad.item(),
                'auxiliary_grad_norm': auxiliary_grad.item(),
                'auxiliary_to_observation_grad': ratio.item(),
            })

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.get('grad_clip', 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.get('grad_clip', 1.0))
            optimizer.step()

        stats['total'] += total_loss.item()
        stats['observation'] += observation_loss.item()
        stats['fy_obs'] += fy_loss.item()
        stats['cosmic_obs'] += cosmic_loss.item()
        stats['iri'] += iri_loss.item()
        stats['increment'] += increment_loss.item()
        stats['vertical'] += vertical_loss.item()
        stats['time'] += time_loss.item()
        stats['weighted_iri'] += weighted_iri.item()
        stats['weighted_increment'] += weighted_increment.item()
        stats['weighted_vertical'] += weighted_vertical.item()
        stats['weighted_time'] += weighted_time.item()
        stats['gradient_ratio'] += ratio.item()
        stats['K_FY_mean'] += fy_extras['K_FY'].mean().item()
        stats['K_COSMIC_mean'] += fy_extras['K_COSMIC'].mean().item()
        stats['innov_FY_norm'] += fy_extras['innov_FY'].norm(dim=-1).mean().item()
        stats['innov_COSMIC_norm'] += fy_extras['innov_COSMIC'].norm(dim=-1).mean().item()
        stats['inflation'] += fy_extras['inflation_scale'].item()
        stats['ne_delta_abs'] += fy_extras['ne_residual'].abs().mean().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f'  [{batch_idx + 1:>5}/{len(train_loader)}] {stage} '
                f'loss={total_loss.item():.4f} FY={fy_loss.item():.4f} '
                f'COSMIC={cosmic_loss.item():.4f} '
                f'V={vertical_loss.item():.4f} T={time_loss.item():.4f} '
                f'({time.time() - started:.0f}s)')

    batches = max(1, len(train_loader))
    result = {key: value / batches for key, value in stats.items()}
    result['gradient_ratio'] = (
        stats['gradient_ratio'] / audited_batches if audited_batches else 0.0)
    result['gradient_audit_batches'] = audited_batches
    return result['total'], result, diagnostics


def _profile_metrics(predictions, targets, profile_ids):
    predictions = np.concatenate(predictions).reshape(-1)
    targets = np.concatenate(targets).reshape(-1)
    profile_ids = np.concatenate(profile_ids).reshape(-1)
    order = np.argsort(profile_ids, kind='stable')
    sorted_ids = profile_ids[order]
    squared = (predictions[order] - targets[order]) ** 2
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_ids)])
    profile_mse = np.add.reduceat(squared, starts) / counts
    return {
        'profile_rmse': float(np.sqrt(profile_mse).mean()),
        'mae': float(np.mean(np.abs(predictions - targets))),
        'rmse': float(np.sqrt(np.mean((predictions - targets) ** 2))),
        'r2': float(
            1.0 - np.sum((predictions - targets) ** 2)
            / (np.sum((targets - targets.mean()) ** 2) + 1e-12)),
    }


@torch.no_grad()
def _evaluate_source(model, loader, batch_processor, device, stage, source,
                     sw_manager, iri_peak_manager, config):
    predictions, targets, profile_ids_all = [], [], []
    for batch in loader:
        coords, target, profile_ids = _unpack_source_batch(batch, device)
        sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
        iri_peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
        prediction, _, _, _, _ = _source_forward(
            model, batch_processor, coords, sw_seq, iri_peak,
            stage, source, profile_ids, False, config, iri_peak_manager)
        predictions.append(prediction.cpu().numpy())
        targets.append(target.cpu().numpy())
        profile_ids_all.append(profile_ids.cpu().numpy())
    return _profile_metrics(predictions, targets, profile_ids_all)


def validate(model, val_loader, batch_processor, device, config,
             sw_manager, iri_peak_manager, cosmic_val_loader, stage='analysis'):
    model.eval()
    fy = _evaluate_source(
        model, val_loader, batch_processor, device, stage, 'FY',
        sw_manager, iri_peak_manager, config)
    cosmic = _evaluate_source(
        model, cosmic_val_loader, batch_processor, device, stage, 'COSMIC',
        sw_manager, iri_peak_manager, config)
    score = 0.5 * (fy['profile_rmse'] + cosmic['profile_rmse'])
    return score, {
        'score': score,
        'fy_profile_rmse': fy['profile_rmse'],
        'cosmic_profile_rmse': cosmic['profile_rmse'],
        'mae': 0.5 * (fy['mae'] + cosmic['mae']),
        'rmse': 0.5 * (fy['rmse'] + cosmic['rmse']),
        'r2': 0.5 * (fy['r2'] + cosmic['r2']),
    }


def _mad_variance(values, sigma_min=0.05, sigma_max=0.40):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('cannot calibrate R from empty residuals')
    median = np.median(values)
    sigma = float(np.clip(
        1.4826 * np.median(np.abs(values - median)),
        sigma_min, sigma_max))
    return sigma ** 2


def _stratified_variance_table(data, config):
    residual = np.asarray(data['residual'], dtype=np.float64).reshape(-1)
    altitude = np.asarray(data['altitude'], dtype=np.float64).reshape(-1)
    local_time = np.remainder(
        np.asarray(data['local_time'], dtype=np.float64).reshape(-1), 24.0)
    profile_id = np.asarray(data['profile_id']).reshape(-1)
    if not (len(residual) == len(altitude) == len(local_time) == len(profile_id)):
        raise ValueError('R calibration arrays must have equal lengths')

    valid = (
        np.isfinite(residual) & np.isfinite(altitude)
        & np.isfinite(local_time) & np.isfinite(profile_id))
    residual = residual[valid]
    altitude = altitude[valid]
    local_time = local_time[valid]
    profile_id = profile_id[valid]

    sigma_min = float(config.get('r_sigma_min', 0.05))
    sigma_max = float(config.get('r_sigma_max', 0.40))
    min_profiles = int(config.get('r_min_profiles', 200))
    shrinkage_profiles = float(config.get('r_shrinkage_profiles', 200))
    if not (0 < sigma_min <= sigma_max):
        raise ValueError('R sigma bounds must satisfy 0 < min <= max')
    if min_profiles < 1 or shrinkage_profiles < 0:
        raise ValueError('R profile thresholds must be non-negative')

    global_variance = _mad_variance(residual, sigma_min, sigma_max)
    table = np.full((3, 2), global_variance, dtype=np.float64)
    altitude_index = np.searchsorted(
        np.asarray([200.0, 300.0]), altitude, side='right')
    day_index = ((local_time >= 6.0) & (local_time < 18.0)).astype(np.int64)
    cells = []
    for altitude_cell in range(3):
        row = []
        for day_cell in range(2):
            cell_mask = (
                (altitude_index == altitude_cell) & (day_index == day_cell))
            point_count = int(cell_mask.sum())
            profile_count = int(np.unique(profile_id[cell_mask]).size)
            raw_variance = (
                _mad_variance(residual[cell_mask], sigma_min, sigma_max)
                if point_count else None)
            fallback = profile_count < min_profiles
            alpha = (
                0.0 if fallback else
                profile_count / (profile_count + shrinkage_profiles))
            if not fallback:
                table[altitude_cell, day_cell] = (
                    alpha * raw_variance + (1.0 - alpha) * global_variance)
            row.append({
                'point_count': point_count,
                'profile_count': profile_count,
                'raw_sigma': (
                    None if raw_variance is None else float(np.sqrt(raw_variance))),
                'raw_variance': (
                    None if raw_variance is None else float(raw_variance)),
                'alpha': float(alpha),
                'fallback_to_global': fallback,
                'final_sigma': float(np.sqrt(table[altitude_cell, day_cell])),
                'final_variance': float(table[altitude_cell, day_cell]),
            })
        cells.append(row)

    metadata = {
        'altitude_bins_km': [[120.0, 200.0], [200.0, 300.0], [300.0, 500.0]],
        'local_time_columns': ['night', 'day'],
        'day_definition': '06:00 <= LT < 18:00',
        'global': {
            'point_count': int(residual.size),
            'profile_count': int(np.unique(profile_id).size),
            'sigma': float(np.sqrt(global_variance)),
            'variance': float(global_variance),
        },
        'cells': cells,
        'calibrated_variance_table': table.tolist(),
    }
    return float(global_variance), table, metadata


@torch.no_grad()
def _calibrate_fixed_r(model, fy_loader, cosmic_loader, device, sw_manager,
                       iri_peak_manager, batch_processor, config):
    model.eval()

    def collect(loader, source):
        collected = {
            'residual': [], 'altitude': [], 'local_time': [], 'profile_id': []}
        sampler = loader.batch_sampler
        previous_shuffle = getattr(sampler, 'shuffle', None)
        if previous_shuffle is not None:
            sampler.shuffle = False
        try:
            for batch_index, batch in enumerate(loader):
                max_batches = config.get('r_calibration_batches')
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                coords, target, profile_ids = _unpack_source_batch(batch, device)
                sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
                iri_peak = (iri_peak_manager.get_iri_peak(coords)
                            if iri_peak_manager is not None else None)
                _, _, _, _, extras = _source_forward(
                    model, batch_processor, coords, sw_seq, iri_peak,
                    'background', source, profile_ids, False, config,
                    iri_peak_manager)
                collected['residual'].append(
                    (extras['ne_bkg'] - target).flatten().cpu().numpy())
                collected['altitude'].append(coords[:, 2].cpu().numpy())
                collected['local_time'].append(torch.remainder(
                    coords[:, 3] + coords[:, 1] / 15.0, 24.0).cpu().numpy())
                collected['profile_id'].append(
                    profile_ids.flatten().cpu().numpy())
        finally:
            if previous_shuffle is not None:
                sampler.shuffle = previous_shuffle
        if not collected['residual']:
            raise ValueError(f'{source} R calibration selected no training data')
        return {
            key: np.concatenate(parts) for key, parts in collected.items()}

    fy_global, fy_table, fy_report = _stratified_variance_table(
        collect(fy_loader, 'FY'), config)
    cosmic_global, cosmic_table, cosmic_report = _stratified_variance_table(
        collect(cosmic_loader, 'COSMIC'), config)
    mode = config.get('r_mode', 'global')
    if mode != 'global':
        raise ValueError('v7 density-observation ETKF requires r_mode=global')
    fy_active = np.full((3, 2), fy_global)
    cosmic_active = np.full((3, 2), cosmic_global)
    model.kalman_layer.set_observation_variances(
        fy_global, cosmic_global, fy_active, cosmic_active)

    report = {
        'schema_version': 1,
        'mode': mode,
        'seed': int(config['seed']),
        'train_only': True,
        'points_per_profile': int(config.get('profile_points_per_epoch', 8)),
        'sigma_clip_dex': [
            float(config.get('r_sigma_min', 0.05)),
            float(config.get('r_sigma_max', 0.40)),
        ],
        'min_profiles': int(config.get('r_min_profiles', 200)),
        'shrinkage_profiles': float(config.get('r_shrinkage_profiles', 200)),
        'FY': {**fy_report, 'active_variance_table': fy_active.tolist()},
        'COSMIC': {
            **cosmic_report, 'active_variance_table': cosmic_active.tolist()},
    }
    manifest_path = os.path.join(config['save_dir'], 'run_manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        report['data_identity'] = manifest.get('data_identity', {})
        report['code_identity'] = manifest.get('code_identity', {})
    seed_path = config.get('background_seed_ckpt')
    if seed_path and os.path.isfile(seed_path):
        report['background_seed_checkpoint'] = {
            'path': os.path.abspath(seed_path),
            'sha256': _sha256_file(seed_path),
        }
    output_path = os.path.join(config['save_dir'], 'r_calibration.json')
    temp_path = f'{output_path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_path)
    print(
        f'  冻结有效R [{mode}]: FY={fy_global:.6f}, '
        f'COSMIC={cosmic_global:.6f}')
    return report


def _load_iri_peak_manager(config, device):
    hmf2_path = config.get('iri_hmf2_path')
    nmf2_path = config.get('iri_nmf2_path')
    if not (hmf2_path and nmf2_path
            and os.path.exists(hmf2_path) and os.path.exists(nmf2_path)):
        return None
    from data_managers.iri_peak_manager import IRIPeakManager
    return IRIPeakManager(
        hmf2_path=hmf2_path, nmf2_path=nmf2_path, device=device)


def _resume_needs_analysis_setup(resume_state, start_epoch, background_epochs):
    if resume_state is None:
        return False
    stage = resume_state.get('stage')
    if stage == 'background':
        if start_epoch > background_epochs:
            raise ValueError('background checkpoint has inconsistent completed_epochs')
        return start_epoch == background_epochs
    if stage == 'analysis':
        if start_epoch <= background_epochs:
            raise ValueError('analysis checkpoint has inconsistent completed_epochs')
        return False
    raise ValueError(f'checkpoint has unknown training stage: {stage}')


def _reset_random_seeds(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _load_background_seed(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    expected = model.state_dict()
    background_prefixes = (
        'iri_proxy.', 'iri_align_net.', 'sw_encoder.', 'sw_freq_branch.',
        'sw_gate.', 'background_decoder.')
    required = {
        name for name in expected if name.startswith(background_prefixes)}
    missing = required - set(state)
    mismatched = {
        name for name in required & set(state)
        if state[name].shape != expected[name].shape}
    if missing or mismatched:
        raise RuntimeError(
            'Background seed is architecture-incompatible: '
            f'missing={sorted(missing)}, shape_mismatch={sorted(mismatched)}')
    migrated = dict(expected)
    for name in required:
        migrated[name] = state[name]
    model.load_state_dict(migrated, strict=True)


def train_fsia(config=None):
    """Train Background first, freeze it, then train ETKF Analysis."""
    config = get_config_mdia() if config is None else config
    if config.get('r_mode') != 'global':
        raise ValueError('v7 density-observation ETKF requires r_mode=global')
    if not config.get('use_distance_localization', False):
        raise ValueError(
            'v7 density-observation ETKF requires continuous localization')
    device = torch.device(config['device'])
    _reset_random_seeds(config['seed'], device)

    background_epochs = int(config.get('background_epochs', 5))
    analysis_epochs = int(config.get('analysis_epochs', 5))
    total_epochs = background_epochs + analysis_epochs
    if background_epochs <= 0 or analysis_epochs <= 0:
        raise ValueError('background_epochs and analysis_epochs must both be positive')

    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device,
    )
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(
        config['iri_proxy_path'], map_location=device, weights_only=True))
    iri_proxy.eval()
    iri_peak_manager = _load_iri_peak_manager(config, device)

    loader_kwargs = dict(
        batch_size=config['batch_size'],
        bin_size_hours=config['bin_size_hours'],
        num_workers=config.get('num_workers', 0),
        use_memmap=config.get('use_memmap', True),
        val_ratio=config.get('val_ratio', 0.1),
        split_seed=config['seed'],
        points_per_profile=config.get('profile_points_per_epoch', 8),
    )
    train_loader, val_loader = get_dataloaders(
        npy_path=config['fy_path'],
        profile_path=config.get('fy_profile_path'),
        profile_index_path=config.get('fy_profile_index_path'),
        **loader_kwargs,
    )
    cosmic_train_loader, cosmic_val_loader = get_cosmic_dataloader(
        cosmic_path=config['cosmic_path'],
        profile_index_path=config.get('cosmic_profile_index_path'),
        **loader_kwargs,
    )

    fy_nb_index = FYNeighborhoodIndex(config['fy_path'], config)
    cosmic_nb_index = COSMICNeighborhoodIndex(config['cosmic_path'], config)
    batch_processor = SlidingWindowBatchProcessor(
        sw_manager, device=device,
        fy_nb_index=fy_nb_index, cosmic_nb_index=cosmic_nb_index)

    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    resume_state = None
    start_epoch = 0
    history = []
    best_scores = {'background': float('inf'), 'analysis': float('inf')}
    best_epochs = {'background': None, 'analysis': None}
    resume_path = config.get('resume_ckpt')
    if resume_path:
        loaded = torch.load(resume_path, map_location=device, weights_only=False)
        if loaded.get('checkpoint_type') != 'run66_training_state':
            if not config.get('eval_only'):
                raise ValueError('run66 cannot resume an older architecture checkpoint')
            model.load_state_dict(loaded, strict=True)
        else:
            if int(loaded.get('format_version', 0)) != 7:
                raise ValueError(
                    'checkpoint is not v7 density-observation ETKF')
            if int(loaded['background_epochs']) != background_epochs:
                raise ValueError('checkpoint background_epochs differs from config')
            checkpoint_r_mode = loaded.get('r_mode')
            if (checkpoint_r_mode is not None
                    and checkpoint_r_mode != config.get('r_mode', 'global')):
                raise ValueError('checkpoint r_mode differs from config')
            checkpoint_localization = loaded.get('use_distance_localization')
            if (checkpoint_localization is not None
                    and bool(checkpoint_localization) != bool(
                        config.get('use_distance_localization', False))):
                raise ValueError(
                    'checkpoint distance localization differs from config')
            model.load_state_dict(loaded['model_state_dict'], strict=True)
            start_epoch = int(loaded['completed_epochs'])
            history = list(loaded.get('history', []))
            best_scores.update(loaded.get('best_scores', {}))
            best_epochs.update(loaded.get('best_epochs', {}))
            resume_state = loaded
            if loaded.get('torch_rng_state') is not None:
                torch.set_rng_state(loaded['torch_rng_state'].cpu())
            if loaded.get('numpy_rng_state') is not None:
                np.random.set_state(loaded['numpy_rng_state'])
            if (device.type == 'cuda'
                    and loaded.get('cuda_rng_state_all') is not None):
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in loaded['cuda_rng_state_all']])

    if config.get('eval_only'):
        return (
            model, [], [], train_loader, val_loader,
            sw_manager, batch_processor, iri_peak_manager,
        )

    os.makedirs(config['save_dir'], exist_ok=True)
    best_background = os.path.join(config['save_dir'], 'best_background_model.pth')
    best_analysis = os.path.join(config['save_dir'], 'best_fsia_model.pth')
    last_state = os.path.join(config['save_dir'], 'last_training_state.pth')

    seeded_background = False
    background_seed = config.get('background_seed_ckpt')
    if background_seed and resume_state is None:
        if not os.path.isfile(background_seed):
            raise FileNotFoundError(
                f'Background seed checkpoint does not exist: {background_seed}')
        _load_background_seed(model, background_seed, device)
        _atomic_torch_save(model.state_dict(), best_background)
        start_epoch = background_epochs
        seeded_background = True
        print(f'  已迁移冻结Background: {background_seed}')

    current_stage = 'background' if start_epoch < background_epochs else 'analysis'
    if (seeded_background or _resume_needs_analysis_setup(
            resume_state, start_epoch, background_epochs)):
        if not os.path.exists(best_background):
            raise FileNotFoundError(
                'cannot enter Analysis: best_background_model.pth is missing')
        model.load_state_dict(
            torch.load(best_background, map_location=device, weights_only=True),
            strict=True)
        _calibrate_fixed_r(
            model, train_loader, cosmic_train_loader, device,
            sw_manager, iri_peak_manager, batch_processor, config)
        _reset_random_seeds(config.get('analysis_seed', 42), device)

    _set_training_stage(model, current_stage)
    stage_epochs = background_epochs if current_stage == 'background' else analysis_epochs
    optimizer, scheduler = _make_optimizer(model, config, stage_epochs)
    scaler = (torch.cuda.amp.GradScaler()
              if config.get('use_amp', False) and device.type == 'cuda' else None)
    if resume_state is not None and resume_state.get('stage') == current_stage:
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        if scheduler is not None and resume_state.get('scheduler_state_dict'):
            scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        if scaler is not None and resume_state.get('scaler_state_dict'):
            scaler.load_state_dict(resume_state['scaler_state_dict'])

    train_losses = [row['total_loss'] for row in history]
    val_losses = [row['val_score'] for row in history]

    for epoch in range(start_epoch, total_epochs):
        stage = 'background' if epoch < background_epochs else 'analysis'
        if stage != current_stage:
            model.load_state_dict(
                torch.load(best_background, map_location=device, weights_only=True),
                strict=True)
            _calibrate_fixed_r(
                model, train_loader, cosmic_train_loader, device,
                sw_manager, iri_peak_manager, batch_processor, config)
            _reset_random_seeds(config.get('analysis_seed', 42), device)
            current_stage = stage
            _set_training_stage(model, stage)
            optimizer, scheduler = _make_optimizer(model, config, analysis_epochs)
            scaler = (torch.cuda.amp.GradScaler()
                      if config.get('use_amp', False) and device.type == 'cuda' else None)

        print(f'\nEpoch {epoch + 1}/{total_epochs} [{stage}]')
        train_loss, train_metrics, batch_diagnostics = train_one_epoch(
            model, train_loader, batch_processor, optimizer, device,
            config, epoch, stage, scaler, sw_manager, iri_peak_manager,
            cosmic_train_loader)
        if batch_diagnostics:
            diagnostics_path = os.path.join(
                config['save_dir'], 'batch_diagnostics.jsonl')
            with open(diagnostics_path, 'a', encoding='utf-8') as stream:
                for row in batch_diagnostics:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        val_score, val_metrics = validate(
            model, val_loader, batch_processor, device, config,
            sw_manager, iri_peak_manager, cosmic_val_loader, stage)
        if scheduler is not None:
            scheduler.step()

        if (train_metrics['gradient_audit_batches']
                and train_metrics['gradient_ratio'] > 0.30):
            print(
                f"  警告: 辅助/观测梯度比={train_metrics['gradient_ratio']:.3f}>0.30；"
                '全量训练前应下调对应辅助权重')

        history_row = {
            'epoch': epoch + 1,
            'stage': stage,
            'total_loss': train_loss,
            'val_score': val_score,
            **train_metrics,
            **val_metrics,
        }
        history.append(history_row)
        train_losses.append(train_loss)
        val_losses.append(val_score)
        print(
            f"  train={train_loss:.6f} val={val_score:.6f} "
            f"FY={val_metrics['fy_profile_rmse']:.6f} "
            f"COSMIC={val_metrics['cosmic_profile_rmse']:.6f}")

        if val_score < best_scores[stage]:
            best_scores[stage] = val_score
            best_epochs[stage] = epoch + 1
            torch.save(
                model.state_dict(),
                best_background if stage == 'background' else best_analysis)

        _atomic_torch_save({
            'checkpoint_type': 'run66_training_state',
            'format_version': 7,
            'completed_epochs': epoch + 1,
            'background_epochs': background_epochs,
            'analysis_epochs': analysis_epochs,
            'r_mode': config.get('r_mode', 'global'),
            'use_distance_localization': bool(
                config.get('use_distance_localization', False)),
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_scores': best_scores,
            'best_epochs': best_epochs,
            'history': history,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'cuda_rng_state_all': (
                torch.cuda.get_rng_state_all() if device.type == 'cuda' else None),
        }, last_state)
        plot_training_curves(
            history,
            save_path=os.path.join(config['save_dir'], 'fsia_training_curves.png'))

    with open(os.path.join(config['save_dir'], 'fsia_training_history.json'),
              'w', encoding='utf-8') as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)
    summary = {
        'best_scores': {
            stage: (score if np.isfinite(score) else None)
            for stage, score in best_scores.items()},
        'best_epochs': best_epochs,
        'checkpoint': best_analysis,
        'checkpoint_sha256': _sha256_file(best_analysis),
        'r_mode': config.get('r_mode', 'global'),
        'use_distance_localization': bool(
            config.get('use_distance_localization', False)),
        'r_fy': model.kalman_layer.r_fy.item(),
        'r_cosmic': model.kalman_layer.r_cosmic.item(),
        'r_calibration': os.path.join(config['save_dir'], 'r_calibration.json'),
    }
    with open(os.path.join(config['save_dir'], 'training_summary.json'),
              'w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    model.load_state_dict(
        torch.load(best_analysis, map_location=device, weights_only=True),
        strict=True)
    return (
        model, train_losses, val_losses, train_loader, val_loader,
        sw_manager, batch_processor, iri_peak_manager,
    )
