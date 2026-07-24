"""
FSIA-INR 训练脚本

Active path: frozen IRI background, local FY/COSMIC profile encoders,
NeuralETKFLayer, and a bounded fusion decoder. Training uses one AdamW group,
FY/COSMIC validation, height-adaptive IRI regularization, and profile peak
alignment. The best checkpoint is selected by validation density loss.
"""

import os
import sys
import json
import hashlib
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from ..config_mdia import get_config_mdia
    from .fsia_model import FSIA_INR_Model
    from .physics_losses_mdia import (
        combined_mdia_physics_loss,
        profile_peak_alignment_loss,
    )
    from .sliding_dataset import SlidingWindowBatchProcessor
    from .plotting import plot_training_curves
except ImportError:
    _mdia_dir = os.path.dirname(os.path.abspath(__file__))
    _inr_dir = os.path.dirname(_mdia_dir)
    for _p in [_inr_dir, _mdia_dir]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from config_mdia import get_config_mdia
    from fsia_model import FSIA_INR_Model
    from physics_losses_mdia import (
        combined_mdia_physics_loss,
        profile_peak_alignment_loss,
    )
    from sliding_dataset import SlidingWindowBatchProcessor
    from plotting import plot_training_curves

from data_managers import SpaceWeatherManager, IRINeuralProxy
from data_managers.FY_dataloader import (
    FY3D_Dataset, TimeBinSampler, FYNeighborhoodIndex,
    COSMICNeighborhoodIndex, get_cosmic_dataloader,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _remaining_phase_counts(completed_epochs, total_epochs, warmup_epochs):
    if not 0 <= completed_epochs <= total_epochs:
        raise ValueError('completed_epochs must be within [0, total_epochs]')
    warmup_remaining = max(0, min(total_epochs, warmup_epochs) - completed_epochs)
    uncertainty_remaining = max(0, total_epochs - max(completed_epochs, warmup_epochs))
    return warmup_remaining, uncertainty_remaining


def _atomic_torch_save(state, path):
    temp_path = f'{path}.tmp'
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def _optimizer_parameter_names(model):
    return [name for name, param in model.named_parameters()
            if param.requires_grad]


def _load_optimizer_state(optimizer, saved_state, model, saved_names=None):
    """Load optimizer state, dropping parameters frozen by repository cleanup."""
    try:
        optimizer.load_state_dict(saved_state)
        return False
    except ValueError as original_error:
        if len(saved_state['param_groups']) != 1:
            raise original_error

        old_ids = saved_state['param_groups'][0]['params']
        if saved_names is None:
            # Training states written before format v2 contained every
            # non-IRI parameter and did not record names.
            saved_names = [
                name for name, _ in model.named_parameters()
                if not name.startswith('iri_proxy.')
            ]
        if len(saved_names) != len(old_ids):
            raise original_error

        current_template = optimizer.state_dict()
        if len(current_template['param_groups']) != 1:
            raise original_error
        current_ids = current_template['param_groups'][0]['params']
        current_names = _optimizer_parameter_names(model)
        if len(current_names) != len(current_ids):
            raise original_error

        old_id_by_name = dict(zip(saved_names, old_ids))
        migrated_state = {}
        for name, current_id in zip(current_names, current_ids):
            old_id = old_id_by_name.get(name)
            if old_id in saved_state['state']:
                migrated_state[current_id] = saved_state['state'][old_id]

        migrated_group = dict(saved_state['param_groups'][0])
        migrated_group['params'] = current_ids
        optimizer.load_state_dict({
            'state': migrated_state,
            'param_groups': [migrated_group],
        })
        return True


# ======================== SubsetTimeBinSampler ========================

class SubsetTimeBinSampler(TimeBinSampler):
    def __init__(self, subset: Subset, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False):
        base_dataset = subset.dataset
        subset_indices = subset.indices
        original_to_subset = {orig: sub for sub, orig in enumerate(subset_indices)}

        filtered_by_bin = {}
        for bin_id, indices in base_dataset.indices_by_bin.items():
            filtered = [original_to_subset[i] for i in indices if i in original_to_subset]
            if filtered:
                filtered_by_bin[bin_id] = np.array(filtered)

        class _TempDS:
            def __init__(self, ibb):
                self.indices_by_bin = ibb

        self.dataset = _TempDS(filtered_by_bin)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


def _query_cosmic_neighbors(batch_processor, coords):
    """Query the formal local COSMIC index for arbitrary model coordinates."""
    index = batch_processor.cosmic_nb_index
    if index is None:
        return None, None
    feats, has_obs = index.query_batch_np(coords.detach().cpu().numpy())
    return (torch.from_numpy(feats).to(coords.device, non_blocking=True),
            torch.from_numpy(has_obs).to(coords.device, non_blocking=True))


# ======================== 训练一个 Epoch ========================

def train_one_epoch(model, train_loader, batch_processor,
                    optimizer, device, config, epoch, scaler=None,
                    sw_manager=None, iri_peak_manager=None,
                    cosmic_train_loader=None, cosmic_iter=None):
    """训练一个 epoch。"""
    model.train()

    warmup_epochs      = config.get('uncertainty_warmup_epochs', 5)
    use_uncertainty    = config.get('use_uncertainty', True) and (epoch >= warmup_epochs)
    # NLL 渐进激活：前 nll_ramp_epochs 个 NLL epoch 内线性混合 Huber→NLL，防冷启动梯度爆炸
    _nll_ramp_epochs   = config.get('nll_ramp_epochs', 2)
    _nll_epoch_idx     = epoch - warmup_epochs          # 0 = 首个 NLL epoch
    _nll_ramp          = min(1.0, max(0.0, (_nll_epoch_idx + 1) / _nll_ramp_epochs)) \
                         if use_uncertainty else 0.0    # 0.0→1.0 线性升
    use_amp            = config.get('use_amp', False) and scaler is not None
    physics_freq       = config.get('physics_loss_freq', 5)

    stats = {k: 0.0 for k in [
        'total', 'mse', 'nll', 'bkg', 'physics_total', 'profile_align',
        'iri_struct', 'trust_iri_mean', 'gate_mean',
        'alt_weight_mean', 'low_alt_frac',
        'K_FY_mean', 'K_COSMIC_mean', 'b_mean', 'r_fy_mean',
        'innov_FY_norm', 'innov_COSMIC_norm',
        'inflation', 'r_ref_FY', 'r_ref_COSMIC', 'member_w_max',
        'ne_delta_abs', 'ne_bkg_mean', 'ne_fused_mean', 'delta_pos_frac',
        'csm_mse',
    ]}
    num_batches = 0

    log_interval  = 100
    total_batches = len(train_loader)
    t0 = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        (coords, target_ne, sw_seq,
         neighbors_feats, has_obs,
         neighbors_feats_cosmic, has_obs_cosmic) = batch_processor.process_batch(batch_data)

        compute_physics = (batch_idx % physics_freq == 0)

        # ---- IRI peak structural reference ----
        iri_peak_batch = None
        if iri_peak_manager is not None:
            with torch.no_grad():
                iri_peak_batch = iri_peak_manager.get_iri_peak(coords.detach())  # [B, 2]

        with torch.amp.autocast('cuda', enabled=use_amp):
            h_sw_shared, _, _ = model.sw_encoder(sw_seq[:1])
            h_sw_shared = h_sw_shared.expand(coords.shape[0], -1).contiguous()
            Ne_fused, log_var, _, _, extras = model(
                coords, sw_seq,
                precomputed_h_sw=h_sw_shared,
                iri_peak=iri_peak_batch,
                neighbors_feats=neighbors_feats,
                has_obs=has_obs,
                neighbors_feats_cosmic=neighbors_feats_cosmic,
                has_obs_cosmic=has_obs_cosmic,
            )

            pure_mse = F.mse_loss(Ne_fused, target_ne)

        # ---- [DIAG] 步骤一：每 diag_interval 个 batch 打印融合内部统计量 ----
        diag_interval = config.get('diag_interval', 500)
        if batch_idx % diag_interval == 0:
            with torch.no_grad():
                _g  = extras['gate'].mean().item()      if extras.get('gate')      is not None else float('nan')
                _gd = extras['gate_data'].mean().item() if extras.get('gate_data') is not None else float('nan')
                _gp = extras['gate_phys'].mean().item() if extras.get('gate_phys') is not None else float('nan')
                _delta     = extras['ne_residual'].abs().mean().item()
                _delta_pos = (extras['ne_residual'] > 0).float().mean().item()
                _bkg       = extras['ne_bkg'].mean().item()
                _fused     = Ne_fused.mean().item()
                _target_mean = target_ne.mean().item()
                _delta_vs_target = _fused - _target_mean   # 正=过高估计，负=低估
                print(f"  [DIAG b{batch_idx:>5}] "
                      f"gate={_g:.3f}(d={_gd:.3f}/p={_gp:.3f})  "
                      f"|Ne_delta|={_delta:.4f}  delta>0={_delta_pos:.2f}  "
                      f"bkg={_bkg:.3f}  fused={_fused:.3f}  target={_target_mean:.3f}  "
                      f"fused-target={_delta_vs_target:+.3f}")

        # ---- run26 (A1): 一次性计算 trust_iri，供 L_shape / bkg_trust / L_peak_iri 共用 ----
        # trust_iri = (1 - K_FY.mean(dim=-1)).detach().clamp(0,1)  ∈ [B]
        # FY 主导样本(K_FY 大)→ trust_iri 小 → IRI anchor 弱化（允许偏离错误的 IRI）
        # IRI 主导样本(K_FY 小)→ trust_iri 大 → IRI anchor 强（守住 IRI 形态/值）
        _K_FY_full = extras.get('K_FY')
        if _K_FY_full is not None and _K_FY_full.numel() > 0 and _K_FY_full.dim() > 1:
            trust_iri_global = (1.0 - _K_FY_full.mean(dim=-1)).detach().clamp(0.0, 1.0)  # [B]
        else:
            trust_iri_global = None

        with torch.amp.autocast('cuda', enabled=use_amp):
            # P0-2: 低高度样本降权（alt<160km→0.3, alt>200km→1.0，平滑过渡）
            alt_batch   = coords[:, 2]                                          # [B]
            alt_weight  = config.get('alt_weight_low', 0.3) + \
                          (1.0 - config.get('alt_weight_low', 0.3)) * \
                          torch.sigmoid((alt_batch - config.get('alt_weight_center', 175.0))
                                        / config.get('alt_weight_scale', 12.0))  # [B]
            alt_weight  = alt_weight.unsqueeze(1)                               # [B,1]

            if use_uncertainty:
                # 渐进 clamp：NLL 冷启动时收紧区间，随 ramp 逐步放开
                # ramp=1.0 时 min_eff=log_var_min(-6)；ramp 期间线性插值到 log_var_min_init(-2)
                _lv_min_full = config.get('log_var_min', -6.0)
                _lv_min_init = config.get('log_var_min_init', -2.0)
                _lv_min_eff  = _lv_min_init + (_lv_min_full - _lv_min_init) * _nll_ramp
                log_var_clamped = torch.clamp(
                    log_var,
                    min=_lv_min_eff,
                    max=config.get('log_var_max', 4.0))
                precision = torch.exp(-log_var_clamped)
                raw_nll   = 0.5 * precision * (Ne_fused - target_ne) ** 2 \
                            + 0.5 * log_var_clamped                             # [B,1]
                # 组合样本权重（各因子独立，默认=1.0，向后兼容）
                _composite_w = alt_weight                                        # [B, 1]
                nll_loss  = (raw_nll * _composite_w).mean()
                log_var_reg = config.get('log_var_regularization', 0.001)
                loss_nll_term = nll_loss + log_var_reg * (log_var_clamped ** 2).mean()
                # Huber 分量（ramp 期间保留作为稳定基座）
                raw_huber = F.huber_loss(Ne_fused, target_ne,
                                         reduction='none', delta=0.3)
                loss_huber_term = (raw_huber * alt_weight).mean()
                # 线性混合：ramp=0 → 纯 Huber；ramp=1 → 纯 NLL
                loss_main = _nll_ramp * loss_nll_term + (1.0 - _nll_ramp) * loss_huber_term
                nll_val = nll_loss.item()
            else:
                raw_huber = F.huber_loss(Ne_fused, target_ne,
                                         reduction='none', delta=0.3)           # [B,1]
                loss_main = (raw_huber * alt_weight).mean()
                nll_val = pure_mse.item()

        if compute_physics:
            physics_loss, phy_dict = combined_mdia_physics_loss(
                pred_ne=Ne_fused,
                ne_bkg=extras['ne_bkg'],
                coords=coords,
                w_bkg_low=config.get('w_bkg_low', 0.25),
                w_bkg_high=config.get('w_bkg_high', 0.02),
                w_bkg_transition=config.get('w_bkg_transition', 250.0),
                w_bkg_sharpness=config.get('w_bkg_sharpness', 25.0),
                trust_iri=trust_iri_global,
            )

            # ---- 1-C: L_iri_struct（h_iri_aligned 重建 IRI 场，防止对齐网络丢弃结构）----
            w_is = config.get('w_iri_struct', 0.0)
            loss_iri_struct = torch.tensor(0.0, device=device)
            if w_is > 0:
                h_iri_aligned = extras.get('h_iri_aligned')
                ne_bkg_detach  = extras['ne_bkg'].detach()
                if h_iri_aligned is not None and h_iri_aligned.requires_grad:
                    ne_iri_recon   = model.iri_recon_head(h_iri_aligned)
                    loss_iri_struct = F.mse_loss(ne_iri_recon, ne_bkg_detach)
                    physics_loss    = physics_loss + w_is * loss_iri_struct
            phy_dict['iri_struct'] = loss_iri_struct.item() \
                if isinstance(loss_iri_struct, torch.Tensor) else 0.0

            # ---- 廓线-峰高对齐损失（使用 hmF2_det，避免重复 detach）----
            w_pa = config.get('w_profile_align', 0.0)
            if w_pa > 0:
                coords_peak = coords.detach().clone()
                coords_peak[:, 2] = extras['hmF2_det']   # detached 融合峰高，更稳定
                coords_peak.requires_grad_(True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    Ne_at_peak, _, _, _, _ = model(
                        coords_peak, sw_seq,
                        precomputed_h_sw=h_sw_shared.detach(),
                        iri_peak=iri_peak_batch,
                    )
                loss_pa = profile_peak_alignment_loss(Ne_at_peak, coords_peak)
                physics_loss = physics_loss + w_pa * loss_pa
                phy_dict['profile_align'] = loss_pa.item()
            else:
                phy_dict['profile_align'] = 0.0

        else:
            physics_loss = 0.0
            phy_dict = {k: 0.0 for k in [
                'bkg', 'physics_total', 'profile_align', 'iri_struct']}

        total_loss = config.get('w_obs', 1.0) * loss_main + physics_loss
        trust_iri_mean_val = (
            trust_iri_global.mean().item()
            if trust_iri_global is not None else 0.0)
        gate = extras.get('gate')

        # ── run64: COSMIC-2 Two-pass 训练 ──
        _w_cosmic   = config.get('w_cosmic', 0.0)
        csm_mse_val = 0.0
        if _w_cosmic > 0 and cosmic_train_loader is not None:
            try:
                csm_batch_data = next(cosmic_iter)
            except (StopIteration, TypeError):
                cosmic_iter    = iter(cosmic_train_loader)
                csm_batch_data = next(cosmic_iter)
            csm_batch_data = csm_batch_data.to(device, non_blocking=True)
            csm_coords    = csm_batch_data[:, :4]    # [B_c, 4]
            csm_target_ne = csm_batch_data[:, 4:5]   # [B_c, 1]
            csm_sw_seq    = sw_manager.get_drivers_sequence(csm_coords[:, 3])
            csm_neighbors, csm_has_obs = _query_cosmic_neighbors(
                batch_processor, csm_coords)
            csm_iri_peak = None
            if iri_peak_manager is not None:
                with torch.no_grad():
                    csm_iri_peak = iri_peak_manager.get_iri_peak(csm_coords.detach())
            with torch.amp.autocast('cuda', enabled=use_amp):
                csm_h_sw, _, _ = model.sw_encoder(csm_sw_seq[:1])
                csm_h_sw = csm_h_sw.expand(csm_coords.shape[0], -1).contiguous()
                csm_Ne, csm_log_var, _, _, csm_extras = model(
                    csm_coords, csm_sw_seq,
                    precomputed_h_sw=csm_h_sw,
                    iri_peak=csm_iri_peak,
                    neighbors_feats_cosmic=csm_neighbors,
                    has_obs_cosmic=csm_has_obs,
                )
                csm_pure_mse = F.mse_loss(csm_Ne, csm_target_ne)
            csm_mse_val = csm_pure_mse.item()
            # trust_iri from COSMIC K_FY
            _K_FY_csm = csm_extras.get('K_FY')
            if (_K_FY_csm is not None and _K_FY_csm.numel() > 0
                    and _K_FY_csm.dim() > 1):
                trust_iri_csm = (1.0 - _K_FY_csm.mean(dim=-1)).detach().clamp(0.0, 1.0)
            else:
                trust_iri_csm = None
            # COSMIC alt weight + main loss
            with torch.amp.autocast('cuda', enabled=use_amp):
                csm_alt_w = (
                    config.get('alt_weight_low', 0.3)
                    + (1.0 - config.get('alt_weight_low', 0.3))
                    * torch.sigmoid((csm_coords[:, 2]
                                     - config.get('alt_weight_center', 175.0))
                                    / config.get('alt_weight_scale', 12.0))
                ).unsqueeze(1)
                if use_uncertainty:
                    csm_lvc = torch.clamp(csm_log_var, min=_lv_min_eff,
                                          max=config.get('log_var_max', 4.0))
                    csm_prec = torch.exp(-csm_lvc)
                    csm_nll_t = (0.5 * csm_prec * (csm_Ne - csm_target_ne) ** 2
                                 + 0.5 * csm_lvc)
                    csm_nll_loss = ((csm_nll_t * csm_alt_w).mean()
                                    + config.get('log_var_regularization', 0.001)
                                    * (csm_lvc ** 2).mean())
                    csm_hub_t    = F.huber_loss(csm_Ne, csm_target_ne,
                                                 reduction='none', delta=0.3)
                    csm_hub_loss = (csm_hub_t * csm_alt_w).mean()
                    csm_loss_main = _nll_ramp * csm_nll_loss + (1.0 - _nll_ramp) * csm_hub_loss
                else:
                    csm_hub_t     = F.huber_loss(csm_Ne, csm_target_ne,
                                                  reduction='none', delta=0.3)
                    csm_loss_main = (csm_hub_t * csm_alt_w).mean()
            csm_total = config.get('w_obs', 1.0) * csm_loss_main
            # COSMIC physics (same freq as FY)
            if compute_physics:
                csm_phy, _ = combined_mdia_physics_loss(
                    pred_ne=csm_Ne,
                    ne_bkg=csm_extras['ne_bkg'],
                    coords=csm_coords,
                    w_bkg_low=config.get('w_bkg_low', 0.25),
                    w_bkg_high=config.get('w_bkg_high', 0.02),
                    w_bkg_transition=config.get('w_bkg_transition', 250.0),
                    w_bkg_sharpness=config.get('w_bkg_sharpness', 25.0),
                    trust_iri=trust_iri_csm,
                )
                csm_total = csm_total + csm_phy
                w_is_csm = config.get('w_iri_struct', 0.0)
                if w_is_csm > 0:
                    csm_h_iri_a = csm_extras.get('h_iri_aligned')
                    if csm_h_iri_a is not None and csm_h_iri_a.requires_grad:
                        csm_total = csm_total + w_is_csm * F.mse_loss(
                            model.iri_recon_head(csm_h_iri_a),
                            csm_extras['ne_bkg'].detach())
            total_loss = total_loss + _w_cosmic * csm_total

        # ── 守卫：非有限 loss 跳过 backward，防止 NaN 写入参数 ──
        if not torch.isfinite(total_loss):
            #print(f"  [WARN] batch {batch_idx}: loss={total_loss.item()}, 跳过 backward")
            optimizer.zero_grad()
            if use_amp and scaler is not None:
                scaler.update()
            num_batches += 1
            continue

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(total_loss).backward()
            if config.get('grad_clip'):
                scaler.unscale_(optimizer)
                # 守卫：NaN 梯度跳过 step（scaler 不一定捕获全部 NaN）
                _has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if _has_nan_grad:
                    print(f"  [WARN] batch {batch_idx}: NaN gradient，跳过 step")
                    optimizer.zero_grad()
                    scaler.update()
                    num_batches += 1
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if config.get('grad_clip'):
                _has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if _has_nan_grad:
                    print(f"  [WARN] batch {batch_idx}: NaN gradient，跳过 step")
                    optimizer.zero_grad()
                    num_batches += 1
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()

        stats['total'] += total_loss.item()
        stats['mse']   += pure_mse.item()
        stats['nll']   += nll_val
        for k in ['bkg', 'physics_total', 'profile_align', 'iri_struct']:
            stats[k] += phy_dict.get(k, 0.0)
        if gate is not None:
            stats['gate_mean'] += gate.mean().item()
        # P0-D 监控
        stats['alt_weight_mean'] += alt_weight.mean().item()
        stats['low_alt_frac']    += (coords[:, 2] < 200).float().mean().item()
        stats['trust_iri_mean'] += trust_iri_mean_val
        with torch.no_grad():
            if extras.get('K_FY') is not None:
                stats['K_FY_mean']         += extras['K_FY'].mean().item()
                stats['K_COSMIC_mean']     += extras['K_COSMIC'].mean().item()
                stats['b_mean']            += extras['b'].mean().item()
                stats['r_fy_mean']         += extras['r_fy'].mean().item()
                stats['innov_FY_norm']     += extras['innov_FY'].norm(dim=-1).mean().item()
                stats['innov_COSMIC_norm'] += extras['innov_COSMIC'].norm(dim=-1).mean().item()
            _infl = extras.get('inflation_scale')
            if _infl is not None:
                stats['inflation'] += _infl.item()
            _r_ref_FY = extras.get('r_ref_FY')
            if _r_ref_FY is not None:
                stats['r_ref_FY']   += _r_ref_FY.detach().item()
            _r_ref_csm = extras.get('r_ref_COSMIC')
            if _r_ref_csm is not None:
                stats['r_ref_COSMIC'] += _r_ref_csm.detach().item()
        with torch.no_grad():
            _mw = extras.get('member_weights')
            if _mw is not None:
                stats['member_w_max'] += _mw.max(dim=-1).values.mean().item()
        # [DIAG] 步骤一：融合问题定位指标累积
        with torch.no_grad():
            stats['ne_delta_abs']   += extras['ne_residual'].abs().mean().item()
            stats['ne_bkg_mean']    += extras['ne_bkg'].mean().item()
            stats['ne_fused_mean']  += Ne_fused.mean().item()
            stats['delta_pos_frac'] += (extras['ne_residual'] > 0).float().mean().item()
        stats['csm_mse'] += csm_mse_val
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed  = time.time() - t0
            phy_str  = f"{phy_dict['physics_total']:.4f}" if compute_physics else 'skip'
            mode_str = 'NLL' if use_uncertainty else 'MSE'
            print(f"  [{batch_idx+1:>5}/{total_batches}] Loss={total_loss.item():.4f}  "
                  f"MSE={pure_mse.item():.4f}  Phy={phy_str}  "
                  f"Mode={mode_str}  ({elapsed:.0f}s)")

    return stats['total'] / num_batches, {k: v / num_batches for k, v in stats.items()}


# ======================== 验证 ========================

def validate(model, val_loader, batch_processor, device, config,
             sw_manager=None, iri_peak_manager=None, cosmic_val_loader=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_data in val_loader:
            (coords, target_ne, sw_seq,
             nb_feats_v, has_obs_v,
             nb_csm_v, has_csm_v) = batch_processor.process_batch(batch_data)
            h_sw_v, _, _ = model.sw_encoder(sw_seq[:1])
            h_sw_v = h_sw_v.expand(coords.shape[0], -1).contiguous()
            _iri_peak_val = None
            if iri_peak_manager is not None:
                _iri_peak_val = iri_peak_manager.get_iri_peak(coords)
            Ne_fused, _, _, _, _ = model(coords, sw_seq, precomputed_h_sw=h_sw_v,
                                         iri_peak=_iri_peak_val,
                                         neighbors_feats=nb_feats_v,
                                         has_obs=has_obs_v,
                                         neighbors_feats_cosmic=nb_csm_v,
                                         has_obs_cosmic=has_csm_v)
            batch_loss = F.mse_loss(Ne_fused, target_ne)
            total_loss += batch_loss.item()
            all_preds.append(Ne_fused.cpu())
            all_targets.append(target_ne.cpu())

    n = len(val_loader)
    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    mae  = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2   = 1 - np.sum((all_preds - all_targets) ** 2) / \
               (np.sum((all_targets - np.mean(all_targets)) ** 2) + 1e-12)
    val_ne_loss = total_loss / n
    val_dict = {'loss': val_ne_loss, 'mae': mae, 'rmse': rmse, 'r2': r2}

    # run64: COSMIC-2 验证集 MSE
    _w_cosmic_val = config.get('w_cosmic', 0.0)
    val_csm_loss  = 0.0
    if _w_cosmic_val > 0 and cosmic_val_loader is not None and sw_manager is not None:
        csm_tot, csm_n = 0.0, 0
        with torch.no_grad():
            for csm_batch in cosmic_val_loader:
                csm_batch = csm_batch.to(device)
                csm_c     = csm_batch[:, :4]
                csm_t     = csm_batch[:, 4:5]
                csm_sw    = sw_manager.get_drivers_sequence(csm_c[:, 3])
                csm_nb, csm_has = _query_cosmic_neighbors(batch_processor, csm_c)
                csm_hw, _, _ = model.sw_encoder(csm_sw[:1])
                csm_hw = csm_hw.expand(csm_c.shape[0], -1).contiguous()
                _csm_ip = None
                if iri_peak_manager is not None:
                    _csm_ip = iri_peak_manager.get_iri_peak(csm_c)
                csm_Ne_v, _, _, _, _ = model(
                    csm_c, csm_sw, precomputed_h_sw=csm_hw, iri_peak=_csm_ip,
                    neighbors_feats_cosmic=csm_nb, has_obs_cosmic=csm_has)
                csm_tot += F.mse_loss(csm_Ne_v, csm_t).item()
                csm_n   += 1
        if csm_n > 0:
            val_csm_loss = csm_tot / csm_n
    val_combined = val_ne_loss + _w_cosmic_val * val_csm_loss
    val_dict['val_csm_mse']  = val_csm_loss
    val_dict['val_combined'] = val_combined

    return val_ne_loss, val_dict


# ======================== 主训练函数 ========================

def train_fsia(config=None):
    """
    FSIA-INR 主训练函数

    Returns:
        model, train_losses, val_losses, 各管理器
    """
    if config is None:
        config = get_config_mdia()

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(config['device'])

    print(f"\n{'='*60}")
    print('FSIA-INR 训练流程')
    print(f"设备: {device}")
    print(f"{'='*60}\n")

    # ========== 步骤 1: 数据管理器 ==========
    print('[步骤 1] 初始化数据管理器...')
    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device,
    )

    # ========== 步骤 2: IRI 代理场 ==========
    print('\n[步骤 2] 加载 IRI 神经代理场...')
    if not os.path.exists(config['iri_proxy_path']):
        raise FileNotFoundError(f"IRI 代理未找到: {config['iri_proxy_path']}")

    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    state = torch.load(config['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(state)
    iri_proxy.eval()
    print('  IRI 代理已加载（FSIA 完全冻结，仅提取隐状态作为特征 Token）')

    # ========== 步骤 2.5: IRI 预计算峰参数管理器 ==========
    print('\n[步骤 2.5] 初始化 IRI 峰参数管理器...')
    iri_peak_manager = None
    _hmf2_p = config.get('iri_hmf2_path')
    _nmf2_p = config.get('iri_nmf2_path')
    if (_hmf2_p and _nmf2_p
            and os.path.exists(_hmf2_p) and os.path.exists(_nmf2_p)):
        from data_managers.iri_peak_manager import IRIPeakManager
        iri_peak_manager = IRIPeakManager(
            hmf2_path=_hmf2_p,
            nmf2_path=_nmf2_p,
            device=device,
        )
        print('  IRI 峰参数管理器已加载（结构参考）')
    else:
        print('  警告: iri_hmf2_path/iri_nmf2_path 未配置或文件不存在，'
              '将使用中性峰值参考 (300km, 11.5)')

    # ========== 步骤 3: 数据集 ==========
    print('\n[步骤 3] 准备数据集...')
    full_dataset = FY3D_Dataset(
        npy_path=config['fy_path'],
        mode='train',
        val_days=[],
        bin_size_hours=config['bin_size_hours'],
        use_memmap=config.get('use_memmap', True),
    )

    total_n = len(full_dataset)
    val_n   = int(total_n * config.get('val_ratio', 0.1))
    train_n = total_n - val_n
    print(f'  总样本: {total_n} | 训练: {train_n} | 验证: {val_n}')

    gen = torch.Generator().manual_seed(config['seed'])
    train_ds, val_ds = random_split(full_dataset, [train_n, val_n], generator=gen)

    train_sampler = SubsetTimeBinSampler(train_ds, config['batch_size'],
                                         shuffle=True, drop_last=False)
    val_sampler   = SubsetTimeBinSampler(val_ds,   config['batch_size'],
                                         shuffle=False, drop_last=False)

    dl_kwargs = {
        'num_workers': config.get('num_workers', 0),
        'pin_memory':  config.get('pin_memory', device.type == 'cuda'),
    }
    if config.get('num_workers', 0) > 0:
        dl_kwargs['prefetch_factor']     = config.get('prefetch_factor', 2)
        dl_kwargs['persistent_workers']  = config.get('persistent_workers', False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,   **dl_kwargs)
    print(f'  训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}')

    # ========== 步骤 3b: FY 邻域索引（run61）==========
    print('\n[步骤 3b] 构建 FY 邻域索引（run61）...')
    fy_nb_index = FYNeighborhoodIndex(config['fy_path'], config)
    print(f'  FYNeighborhoodIndex 已构建 (profiles={len(fy_nb_index.prof_starts):,}, '
          f'dt={fy_nb_index.dt}h, dlat={fy_nb_index.dlat}°, '
          f'dlon={fy_nb_index.dlon}°, k_max={fy_nb_index.k_max})')

    # ========== 步骤 3c: COSMIC-2 邻域索引（run64）==========
    print('\n[步骤 3c] 构建 COSMIC-2 邻域索引（run64）...')
    cosmic_nb_index  = None
    cosmic_path_cfg  = config.get('cosmic_path', '')
    if cosmic_path_cfg and os.path.exists(cosmic_path_cfg):
        cosmic_nb_index = COSMICNeighborhoodIndex(cosmic_path_cfg, config)
        print(f'  COSMICNeighborhoodIndex 已构建 '
              f'(profiles={len(cosmic_nb_index.prof_starts):,}, '
              f'dt={cosmic_nb_index.dt}h, dlat={cosmic_nb_index.dlat}°, '
              f'dlon={cosmic_nb_index.dlon}°, k_max={cosmic_nb_index.k_max})')
    else:
        if config.get('w_cosmic', 0.0) > 0:
            raise FileNotFoundError(f'COSMIC 数据文件不存在: {cosmic_path_cfg}')
        print('  w_cosmic=0，跳过 COSMIC 邻域索引')

    batch_processor = SlidingWindowBatchProcessor(sw_manager, device=device,
                                                  fy_nb_index=fy_nb_index,
                                                  cosmic_nb_index=cosmic_nb_index)

    # ========== 步骤 3d: COSMIC-2 数据加载器（run64）==========
    print('\n[步骤 3d] 准备 COSMIC-2 数据加载器（run64）...')
    cosmic_train_loader = None
    cosmic_val_loader   = None
    if cosmic_path_cfg and os.path.exists(cosmic_path_cfg) and config.get('w_cosmic', 0.0) > 0:
        _csm_val_days = config.get('cosmic_val_days', [4, 14, 24])
        cosmic_train_loader, cosmic_val_loader = get_cosmic_dataloader(
            cosmic_path=cosmic_path_cfg,
            val_days=_csm_val_days,
            batch_size=config.get('batch_size', 2048),
            bin_size_hours=config.get('bin_size_hours', 1.0),
            num_workers=config.get('num_workers', 0),
            use_memmap=config.get('use_memmap', True),
        )
        print(f'  COSMIC 训练批次: {len(cosmic_train_loader)}'
              f' | 验证批次: {len(cosmic_val_loader)}'
              f' | val_days={_csm_val_days}')
    else:
        print('  w_cosmic=0 或 cosmic_path 未配置/不存在，跳过 COSMIC 数据加载')

    # ========== 步骤 4: FSIA-INR 模型 ==========
    print('\n[步骤 4] 初始化 FSIA-INR 模型...')
    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    iri_total        = sum(p.numel() for n, p in model.named_parameters() if 'iri_proxy' in n)
    print(f'  总参数:      {total_params:,}')
    print(f'  可训练:      {trainable_params:,}')
    print(f'  IRI 冻结:    {iri_total:,}（完全冻结，不参与优化）')
    print(f'  初始 τ_kp:   {model.sw_encoder.tau_kp.item():.2f} h')
    print(f'  初始 τ_solar:{model.sw_encoder.tau_solar.item():.2f} h')

    # ========== 步骤 4.5: 断点续训（可选）==========
    _resume_ckpt = config.get('resume_ckpt')
    resume_state = None
    start_epoch = 0
    if _resume_ckpt:
        if not os.path.exists(_resume_ckpt):
            raise FileNotFoundError(f'续训 checkpoint 不存在: {_resume_ckpt}')
        loaded = torch.load(_resume_ckpt, map_location='cpu', weights_only=False)
        if (isinstance(loaded, dict)
                and loaded.get('checkpoint_type') == 'fsia_training_state'):
            resume_state = loaded
            model.load_state_dict(loaded['model_state_dict'], strict=True)
            start_epoch = int(loaded['completed_epochs'])
            saved_warmup = int(loaded['uncertainty_warmup_epochs'])
            if saved_warmup != int(config.get('uncertainty_warmup_epochs', 5)):
                raise ValueError('续训 checkpoint 与当前 uncertainty_warmup_epochs 不一致')
            print(f'\n[步骤 4.5] 已加载完整训练状态: {_resume_ckpt}')
        else:
            completed = config.get('resume_completed_epochs')
            if completed is None and config.get('eval_only'):
                completed = 0
            elif completed is None:
                raise ValueError('旧 raw state_dict 续训必须设置 resume_completed_epochs')
            model.load_state_dict(loaded, strict=True)
            start_epoch = int(completed)
            print(f'\n[步骤 4.5] 已加载旧 raw state_dict: {_resume_ckpt}')
            print('  注意: 旧 checkpoint 不含优化器/scheduler，二者将重新初始化')

        _cosmic_dead = (
            torch.count_nonzero(
                model.cosmic_obs_encoder.input_proj.weight).item() == 0
            and torch.count_nonzero(
                model.proj_pre.weight[:, -model.kalman_layer.d_model:]).item() == 0
        )
        if config.get('w_cosmic', 0.0) > 0 and _cosmic_dead:
            model._initialize_cosmic_bootstrap()
            print('  检测到旧 checkpoint COSMIC 全零入口，已仅重启 COSMIC 梯度路径')

        warmup_left, uncertainty_left = _remaining_phase_counts(
            start_epoch, int(config['epochs']),
            int(config.get('uncertainty_warmup_epochs', 5)))
        if start_epoch >= int(config['epochs']):
            raise ValueError(f'已完成 {start_epoch} epochs，目标总轮数为 {config["epochs"]}')
        print(f'  已完成: {start_epoch}/{config["epochs"]}  '
              f'剩余 warmup={warmup_left}  uncertainty={uncertainty_left}')

    if config.get('eval_only'):
        print('\n[评估模式] 已加载数据上下文与 checkpoint，跳过优化器和训练循环')
        return (model, [], [], train_loader, val_loader,
                sw_manager, batch_processor, iri_peak_manager)

    # ========== 步骤 5: 优化器 ==========
    print('\n[步骤 5] 配置优化器...')
    lr = config['lr']

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=config.get('weight_decay', 1e-4),
    )
    print(f'  LR: {lr:.2e}  (IRI 完全冻结，不在优化器中)')

    if config.get('scheduler_type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None

    use_amp = config.get('use_amp', False)
    scaler  = None
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print('  启用混合精度训练（AMP）')
    elif use_amp:
        print('  AMP 仅支持 CUDA，已禁用')
        use_amp = False

    if resume_state is not None:
        optimizer_migrated = _load_optimizer_state(
            optimizer,
            resume_state['optimizer_state_dict'],
            model,
            resume_state.get('optimizer_param_names'),
        )
        if optimizer_migrated:
            print('  已剔除冻结参数的旧优化器状态')
        if scheduler is not None and resume_state.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        if scaler is not None and resume_state.get('scaler_state_dict') is not None:
            scaler.load_state_dict(resume_state['scaler_state_dict'])
        rng_state = resume_state.get('rng_state', {})
        if rng_state.get('numpy') is not None:
            np.random.set_state(rng_state['numpy'])
        if rng_state.get('torch') is not None:
            torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and rng_state.get('cuda') is not None:
            torch.cuda.set_rng_state_all(rng_state['cuda'])
        print('  优化器、scheduler、AMP及随机状态已恢复')

    # ========== 步骤 6: 训练循环 ==========
    print(f'\n[步骤 6] 开始训练 (目标总轮数 {config["epochs"]}, '
          f'从 Epoch {start_epoch + 1} 继续)...')
    warmup = config.get('uncertainty_warmup_epochs', 5)
    if config.get('use_uncertainty'):
        print(f'  前 {warmup} 轮使用 Huber loss，之后启用 NLL + 不确定性学习')
    print(f'  物理损失每 {config.get("physics_loss_freq", 5)} 个 batch 计算一次')

    history = list(resume_state.get('history', [])) if resume_state else []
    train_losses = [row['total_loss'] for row in history]
    val_losses = [row['val_mse'] for row in history]
    if resume_state:
        best_val_ne = float(resume_state['best_val_ne'])
        best_epoch = resume_state.get('best_epoch')
        ne_patience_counter = int(resume_state.get('ne_patience_counter', 0))
    else:
        best_val_ne_cfg = config.get('resume_best_val')
        best_val_ne = (float(best_val_ne_cfg)
                       if best_val_ne_cfg is not None else float('inf'))
        best_epoch = config.get('resume_best_epoch')
        ne_patience_counter = 0
    best_ckpt = os.path.join(config['save_dir'], 'best_fsia_model.pth')
    last_state_ckpt = os.path.join(config['save_dir'], 'last_training_state.pth')

    for epoch in range(start_epoch, config['epochs']):
        epoch_t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['epochs']}")

        cosmic_train_iter = iter(cosmic_train_loader) if cosmic_train_loader is not None else None
        train_loss, train_dict = train_one_epoch(
            model, train_loader, batch_processor,
            optimizer, device, config, epoch, scaler,
            sw_manager=sw_manager,
            iri_peak_manager=iri_peak_manager,
            cosmic_train_loader=cosmic_train_loader,
            cosmic_iter=cosmic_train_iter,
        )
        train_losses.append(train_loss)

        val_loss, val_metrics = validate(
            model, val_loader, batch_processor, device, config,
            sw_manager=sw_manager,
            iri_peak_manager=iri_peak_manager,
            cosmic_val_loader=cosmic_val_loader,
        )
        val_losses.append(val_loss)

        epoch_elapsed = time.time() - epoch_t0
        print(f"  Epoch time:   {epoch_elapsed:.1f}s  ({epoch_elapsed/60:.1f}min)")
        print(f"  Train Loss:   {train_dict['total']:.6f}")
        print(f"    MSE:          {train_dict['mse']:.6f}")
        print(f"    NLL:          {train_dict['nll']:.6f}")
        print(f"    Bkg Trust:    {train_dict['bkg']:.6f}")
        if train_dict.get('iri_struct', 0.0) > 0:
            print(f"    IRI Struct:   {train_dict['iri_struct']:.4f}")
        _k_fy = train_dict.get('K_FY_mean', 0.0)
        _k_csm = train_dict.get('K_COSMIC_mean', 0.0)
        _b_da = train_dict.get('b_mean', 0.0)
        _r_fy = train_dict.get('r_fy_mean', 0.0)
        print(f"  [DA] K_FY={_k_fy:.3f}  K_COSMIC={_k_csm:.3f}  b={_b_da:.3f}  r_fy={_r_fy:.3f}"
              f"  innov_FY={train_dict.get('innov_FY_norm',0.0):.3f}"
              f"  innov_COSMIC={train_dict.get('innov_COSMIC_norm',0.0):.3f}")
        print(f"  Val Ne Loss:  {val_loss:.6f}")
        print(f"    MAE={val_metrics['mae']:.6f}  RMSE={val_metrics['rmse']:.6f}"
              f"  R²={val_metrics['r2']:.4f}")

        tau_kp_val    = model.sw_encoder.tau_kp.item()
        tau_solar_val = model.sw_encoder.tau_solar.item()
        gate_mean_v   = train_dict['gate_mean']
        print(f"  τ_kp:{tau_kp_val:.2f}h  τ_solar:{tau_solar_val:.2f}h"
              f"  gate_mean:{gate_mean_v:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        _bkg_mse_ratio = train_dict['bkg'] / max(train_dict['mse'], 1e-8)
        _bias_warn = " ⚠ 全局正偏置!" if train_dict['delta_pos_frac'] > 0.85 else ""
        print(f"  [DIAG] |Ne_delta|={train_dict['ne_delta_abs']:.4f}  "
              f"delta>0={train_dict['delta_pos_frac']:.3f}{_bias_warn}  "
              f"bkg={train_dict['ne_bkg_mean']:.3f}  fused={train_dict['ne_fused_mean']:.3f}  "
              f"bkg/mse={_bkg_mse_ratio:.2f}x")

        history.append({
            'epoch': epoch + 1,
            'train_mse': train_dict['mse'],
            'val_mse':   val_loss,
            'val_cosmic_mse': val_metrics.get('val_csm_mse'),
            'val_combined': val_metrics.get('val_combined', val_loss),
            'train_nll': train_dict['nll'],
            'total_loss': train_dict['total'],
            'bkg':              train_dict['bkg'],
            'profile_align':    train_dict['profile_align'],
            'iri_struct':       train_dict['iri_struct'],
            'tau_kp':           tau_kp_val,
            'tau_solar':        tau_solar_val,
            'gate_mean':        gate_mean_v,
            'K_FY_mean':        train_dict.get('K_FY_mean', 0.0),
            'K_COSMIC_mean':    train_dict.get('K_COSMIC_mean', 0.0),
            'b_mean':           train_dict.get('b_mean', 0.0),
            'r_fy_mean':        train_dict.get('r_fy_mean', 0.0),
            'inflation':        train_dict.get('inflation', 0.0),
            'member_w_max':     train_dict.get('member_w_max', 0.0),
        })

        if scheduler:
            scheduler.step()

        _val_combined = val_metrics.get('val_combined', val_loss)  # run64: FY + w_cosmic*COSMIC
        if _val_combined < best_val_ne:
            best_val_ne          = _val_combined
            best_epoch           = epoch + 1
            ne_patience_counter  = 0
            torch.save(model.state_dict(), best_ckpt)
            _csm_str = (f"  COSMIC MSE={val_metrics.get('val_csm_mse', 0.0):.6f}"
                        f"  combined={_val_combined:.6f}")
            print(f"  [*] 最佳模型已保存 (val_combined): {best_ckpt}")
            print(f"  [*] Ne MSE={val_loss:.6f}{_csm_str}")
        else:
            ne_patience_counter += 1

        print(f"  [早停] ne_pat={ne_patience_counter}/{config.get('ne_patience', 4)}")

        plot_training_curves(
            history,
            save_path=os.path.join(config['save_dir'], 'fsia_training_curves.png')
        )

        if (epoch + 1) % config.get('save_interval', 5) == 0:
            ckpt_path = os.path.join(config['save_dir'], f'fsia_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)

        _atomic_torch_save({
            'checkpoint_type': 'fsia_training_state',
            'format_version': 2,
            'completed_epochs': epoch + 1,
            'target_epochs': int(config['epochs']),
            'uncertainty_warmup_epochs': int(warmup),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_param_names': _optimizer_parameter_names(model),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_ne': best_val_ne,
            'best_epoch': best_epoch,
            'ne_patience_counter': ne_patience_counter,
            'history': history,
            'rng_state': {
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
        }, last_state_ckpt)
        print(f'  [断点] 完整训练状态已保存: {last_state_ckpt}')

        if (config.get('early_stopping')
                and ne_patience_counter >= config.get('ne_patience', 4)):
            print(f'\n[早停] 触发 ne_pat={ne_patience_counter}')
            break

    print(f"\n{'='*60}")
    print(f"训练完成！最佳联合验证 MSE: {best_val_ne:.6f}")

    hist_path = os.path.join(config['save_dir'], 'fsia_training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存: {hist_path}")

    summary = {
        'best_epoch': best_epoch,
        'best_val_combined': best_val_ne,
        'best_epoch_metrics': next(
            (row for row in history if row.get('epoch') == best_epoch), None),
        'fy_profiles': len(fy_nb_index.prof_starts),
        'cosmic_profiles': (len(cosmic_nb_index.prof_starts)
                            if cosmic_nb_index is not None else 0),
        'checkpoint_path': os.path.abspath(best_ckpt),
        'checkpoint_sha256': _sha256_file(best_ckpt),
    }
    summary_path = os.path.join(config['save_dir'], 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"训练摘要已保存: {summary_path}")

    return (model, train_losses, val_losses,
            train_loader, val_loader,
            sw_manager, batch_processor, iri_peak_manager)


# ======================== 入口 ========================
if __name__ == '__main__':
    _module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _module_dir not in sys.path:
        sys.path.insert(0, _module_dir)
    cfg = get_config_mdia()
    train_fsia(cfg)
