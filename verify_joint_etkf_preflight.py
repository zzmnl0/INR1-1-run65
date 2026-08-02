"""Actual-data forward/backward and 100-step overfit check for joint ETKF."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from inr_modules.config_mdia import get_config_mdia
from inr_modules.data_managers import IRINeuralProxy, SpaceWeatherManager
from inr_modules.data_managers.FY_dataloader import (
    COSMICDataset,
    COSMICNeighborhoodIndex,
    FY3D_Dataset,
    FYNeighborhoodIndex,
    ProfileTimeBinSampler,
)
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.physics_losses_mdia import profile_huber_loss
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    query_observation_payload,
)
from inr_modules.mdia.train_fsia import (
    _load_background_seed,
    _load_iri_peak_manager,
    _set_stage_mode,
    _set_training_stage,
    _unpack_source_batch,
)


ROOT = Path(__file__).resolve().parent
FY_DATA = Path(r'D:\FYsatellite\EDP_data\fy_202409_qc_v2.npy')
FY_INDEX = Path(r'D:\FYsatellite\EDP_data\fy_202409_qc_v2_index.npz')
COSMIC_DATA = Path(
    r'D:\cosmic2\cosmic245-274-September\cosmic_september_2024_qc.npy')
COSMIC_INDEX = Path(
    r'D:\cosmic2\cosmic245-274-September\cosmic_september_2024_qc_index.npz')
BACKGROUND = (
    ROOT / 'checkpoints_fsia' / 'run66-etkf-loss'
    / 'best_background_model.pth')


def _prepare_batch(batch, source, device, config, fy_index, cosmic_index,
                   sw_manager, iri_peak_manager, model):
    coords, target, profile_ids = _unpack_source_batch(batch, device)
    fy_exclude = profile_ids if source == 'FY' else None
    cosmic_exclude = profile_ids if source == 'COSMIC' else None
    fy_observations = query_observation_payload(
        fy_index, coords, device,
        exclude_profile_ids=(
            None if fy_exclude is None else fy_exclude.cpu().numpy()))
    cosmic_observations = query_observation_payload(
        cosmic_index, coords, device,
        exclude_profile_ids=(
            None if cosmic_exclude is None else cosmic_exclude.cpu().numpy()))
    fy_observations = attach_observation_background(
        fy_observations, model, sw_manager, iri_peak_manager)
    cosmic_observations = attach_observation_background(
        cosmic_observations, model, sw_manager, iri_peak_manager)
    return {
        'coords': coords,
        'target': target,
        'profile_ids': profile_ids,
        'sw_seq': sw_manager.get_drivers_sequence(coords[:, 3]),
        'iri_peak': (iri_peak_manager.get_iri_peak(coords)
                     if iri_peak_manager is not None else None),
        'fy_observations': fy_observations,
        'cosmic_observations': cosmic_observations,
    }


def _forward(model, prepared, use_fy=True, use_cosmic=True):
    return model(
        prepared['coords'],
        prepared['sw_seq'],
        iri_peak=prepared['iri_peak'],
        observations_fy=(
            prepared['fy_observations'] if use_fy else None),
        observations_cosmic=(
            prepared['cosmic_observations'] if use_cosmic else None),
    )


def _loss(model, fy_batch, cosmic_batch, delta):
    fy_prediction = _forward(model, fy_batch)[0]
    cosmic_prediction = _forward(model, cosmic_batch)[0]
    fy_loss = profile_huber_loss(
        fy_prediction, fy_batch['target'], fy_batch['profile_ids'], delta)
    cosmic_loss = profile_huber_loss(
        cosmic_prediction, cosmic_batch['target'],
        cosmic_batch['profile_ids'], delta)
    return 0.5 * (fy_loss + cosmic_loss), fy_loss, cosmic_loss


def _gradient_sum(parameter):
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        return None
    return float(gradient.abs().sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--enkf-members', type=int, default=8)
    parser.add_argument(
        '--anomaly-parameterization',
        choices=('legacy_independent', 'orthogonal_factor'),
        default='orthogonal_factor')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 8:
        raise ValueError('steps must be positive and batch-size must be at least 8')

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(get_config_mdia())
    config.update({
        'device': str(device),
        'batch_size': args.batch_size,
        'num_workers': 0,
        'use_amp': False,
        'fy_path': str(FY_DATA),
        'fy_profile_path': None,
        'fy_profile_index_path': str(FY_INDEX),
        'cosmic_path': str(COSMIC_DATA),
        'cosmic_profile_index_path': str(COSMIC_INDEX),
        'r_mode': 'global',
        'use_distance_localization': True,
        'enkf_n_members': args.enkf_members,
        'enkf_anomaly_parameterization': args.anomaly_parameterization,
    })
    for path in (FY_DATA, FY_INDEX, COSMIC_DATA, COSMIC_INDEX, BACKGROUND):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset_kwargs = {
        'mode': 'train',
        'bin_size_hours': config['bin_size_hours'],
        'use_memmap': True,
        'val_ratio': config['val_ratio'],
        'split_seed': 42,
    }
    fy_dataset = FY3D_Dataset(
        str(FY_DATA), profile_path=None, profile_index_path=str(FY_INDEX),
        **dataset_kwargs)
    cosmic_dataset = COSMICDataset(
        str(COSMIC_DATA), profile_index_path=str(COSMIC_INDEX),
        **dataset_kwargs)
    sampler_kwargs = {
        'batch_size': args.batch_size,
        'points_per_profile': config['profile_points_per_epoch'],
        'shuffle': True,
    }
    fy_loader = DataLoader(
        fy_dataset,
        batch_sampler=ProfileTimeBinSampler(fy_dataset, **sampler_kwargs))
    cosmic_loader = DataLoader(
        cosmic_dataset,
        batch_sampler=ProfileTimeBinSampler(cosmic_dataset, **sampler_kwargs))
    fy_index = FYNeighborhoodIndex(str(FY_DATA), config)
    cosmic_index = COSMICNeighborhoodIndex(str(COSMIC_DATA), config)

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
    model = FSIA_INR_Model(iri_proxy, config).to(device)
    _load_background_seed(model, str(BACKGROUND), device)
    iri_peak_manager = _load_iri_peak_manager(config, device)
    _set_training_stage(model, 'analysis')
    _set_stage_mode(model, 'analysis')

    fy_batch = _prepare_batch(
        next(iter(fy_loader)), 'FY', device, config, fy_index, cosmic_index,
        sw_manager, iri_peak_manager, model)
    cosmic_batch = _prepare_batch(
        next(iter(cosmic_loader)), 'COSMIC', device, config, fy_index,
        cosmic_index, sw_manager, iri_peak_manager, model)
    def coverage_of(batch, source):
        return float(batch[f'{source}_observations'][
            'valid_mask'].any(dim=1).float().mean())

    coverage = {
        'fy_on_fy_batch': coverage_of(fy_batch, 'fy'),
        'cosmic_on_fy_batch': coverage_of(fy_batch, 'cosmic'),
        'fy_on_cosmic_batch': coverage_of(cosmic_batch, 'fy'),
        'cosmic_on_cosmic_batch': coverage_of(cosmic_batch, 'cosmic'),
    }
    if not (coverage['fy_on_fy_batch'] + coverage['fy_on_cosmic_batch']
            and coverage['cosmic_on_fy_batch']
            + coverage['cosmic_on_cosmic_batch']):
        raise RuntimeError(f'preflight batches lack source coverage: {coverage}')

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=config['lr'], weight_decay=config['weight_decay'])
    delta = config['huber_delta']
    initial = [float(value.detach()) for value in
               _loss(model, fy_batch, cosmic_batch, delta)]

    gradient_report = {}
    for source, batch, flags in (
            ('FY', fy_batch, (True, False)),
            ('COSMIC', cosmic_batch, (False, True))):
        model.zero_grad(set_to_none=True)
        prediction = _forward(
            model, batch, use_fy=flags[0], use_cosmic=flags[1])[0]
        profile_huber_loss(
            prediction, batch['target'], batch['profile_ids'], delta).backward()
        perturbation_parameters = (
            ('P_w1', 'P_b1', 'P_w2', 'P_b2')
            if args.anomaly_parameterization == 'legacy_independent'
            else (
                'covariance_scale_net.0.weight',
                'covariance_scale_net.0.bias',
                'covariance_scale_net.2.weight',
                'covariance_scale_net.2.bias',
            ))
        named_parameters = dict(model.kalman_layer.named_parameters())
        gradient_report[source] = {
            'density_basis': _gradient_sum(
                model.density_basis_decoder[-1].weight),
            'perturbations': sum(
                _gradient_sum(named_parameters[name]) or 0.0
                for name in perturbation_parameters),
        }
        if any(value is None or value <= 0
               for value in gradient_report[source].values()):
            raise RuntimeError(
                f'non-finite or zero {source} gradient: '
                f'{gradient_report[source]}')

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = _loss(model, fy_batch, cosmic_batch, delta)
        if not torch.isfinite(total):
            raise FloatingPointError(f'non-finite loss at step {step + 1}')
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            config['grad_clip'])
        optimizer.step()

    final = [float(value.detach()) for value in
             _loss(model, fy_batch, cosmic_batch, delta)]
    reduction = 1.0 - final[0] / initial[0]
    result = {
        'device': str(device),
        'steps': args.steps,
        'coverage': coverage,
        'initial': {'total': initial[0], 'FY': initial[1], 'COSMIC': initial[2]},
        'final': {'total': final[0], 'FY': final[1], 'COSMIC': final[2]},
        'observation_loss_reduction': reduction,
        'critical_gradient_l1': gradient_report,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.steps >= 100 and reduction < 0.20:
        raise AssertionError(
            f'100-step observation loss reduction {reduction:.2%} is below 20%')


if __name__ == '__main__':
    main()
