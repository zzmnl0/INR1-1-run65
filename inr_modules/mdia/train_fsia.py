"""
FSIA-INR 训练脚本

主要特性：
    1. FSIA_INR_Model — IRI 完全冻结 + CrossSourceAttention + PeakHead + FusionDecoder
    2. 单组 AdamW 优化器（IRI 冻结，无需分组 LR）
    3. 双域早停：Ne MSE + GIRO 峰值 MAE 独立计数
    4. 物理损失：5 项（bkg/res_smooth/horiz_smooth/uplift/depletion）+ 廓线对齐
    5. 检查点：best_fsia_model.pth（以 Ne MSE 改善为保存条件）
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
        peak_field_smooth_loss,
        pearson_r_shape_loss,
        banded_pearson_loss,
        ne_vert_smooth_loss,
    )
    from .sliding_dataset import SlidingWindowBatchProcessor
    from .plotting import plot_training_curves
    from .giro_dataloader import build_giro_loaders, compute_giro_loss
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
        peak_field_smooth_loss,
        pearson_r_shape_loss,
        banded_pearson_loss,
        ne_vert_smooth_loss,
    )
    from sliding_dataset import SlidingWindowBatchProcessor
    from plotting import plot_training_curves
    from giro_dataloader import build_giro_loaders, compute_giro_loss

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
                    giro_hmf2_loader=None, giro_nmf2_loader=None, sw_manager=None,
                    iri_peak_manager=None,
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

    stats = {k: 0.0 for k in ['total', 'mse', 'nll', 'bkg',
                                'uplift', 'depletion', 'eia_crest', 'physics_total',
                                'profile_align', 'peak_smooth', 'ne_vert_smooth',  # run41
                                'iri_struct',                           # 1-C
                                'l_shape',                              # run22 Pearson-r 形态约束
                                # run25: DA 自动加权 IRI 峰锚定（无新硬编码 w_*）
                                'peak_iri_hmf2',     # hmF2 vs IRI MSE，权重 = trust_iri = 1-K_FY.mean()
                                'peak_iri_nmf2',     # NmF2 banded Pearson 形状损失（IRI 系统偏差不锚定值）
                                'trust_iri_mean',    # batch 平均 IRI 信任度（1 - K_FY 均值）
                                'alpha_eff_mean',    # run26 (B2): batch 平均 PeakHead 自适应 detach 系数
                                'giro_hmf2', 'giro_nmf2', 'gate_mean',
                                'giro_ne_val', 'giro_argmax',
                                'alt_weight_mean',                      # P0-D 监控
                                'low_alt_frac',
                                # run22 DA 监控键（run28：MHDK→NeuralETKFLayer，物理意义升级）
                                'K_FY_mean',        # K_eff_FY 均值（ens_var/(ens_var+r_fy)）
                                'K_vert_mean',      # K_eff_vert 均值
                                'b_mean',           # ensemble 方差均值（≡ 背景误差协方差对角元）
                                'r_fy_mean',        # FY 观测误差均值（含底侧+夜间物理先验）
                                'innov_FY_norm',    # FY 新息 L2 范数均值
                                'innov_vert_norm',  # 垂直新息 L2 范数均值
                                # run28 NeuralETKFLayer ensemble 监控
                                'inflation',        # 当前 inflation 因子 = exp(log_inflation)
                                # run59: H 学习参考 R 监控（run60: alpha_vert_night/eq 已删除）
                                'r_ref_FY',         # H_FY 梯度路径基准 R（均等学日/夜）
                                'r_ref_vert',       # H_vert 梯度路径基准 R
                                'member_w_max',    # max ensemble member weight（dominance 监控）
                                # [DIAG] 融合问题定位监控指标
                                'ne_delta_abs',    # |Ne_delta| 均值：decoder 修正幅度
                                'ne_bkg_mean',     # IRI 背景 Ne 均值
                                'ne_fused_mean',   # 融合 Ne 均值
                                'delta_pos_frac',  # Ne_delta>0 比例（健康值 0.4~0.6）
                                'hmf2_iri_diff',   # |hmF2_fused - hmF2_IRI| 均值 (km)
                                'csm_mse',         # run64: COSMIC-2 MSE（Two-pass）
                                ]}
    num_batches = 0

    hmf2_iter        = iter(giro_hmf2_loader) if giro_hmf2_loader is not None else None
    nmf2_iter        = iter(giro_nmf2_loader) if giro_nmf2_loader is not None else None
    giro_direct_iter = iter(giro_hmf2_loader) if giro_hmf2_loader is not None else None  # 1-D
    use_giro  = (hmf2_iter is not None or nmf2_iter is not None) and sw_manager is not None
    log_interval  = 100
    total_batches = len(train_loader)
    t0 = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        (coords, target_ne, sw_seq,
         neighbors_feats, has_obs,
         neighbors_feats_cosmic, has_obs_cosmic) = batch_processor.process_batch(batch_data)

        compute_physics = (batch_idx % physics_freq == 0)

        # ---- IRI 峰参数查询（PeakHead 背景输入）----
        iri_peak_batch = None
        if iri_peak_manager is not None:
            with torch.no_grad():
                iri_peak_batch = iri_peak_manager.get_iri_peak(coords.detach())  # [B, 2]

        with torch.amp.autocast('cuda', enabled=use_amp):
            h_sw_shared, _, _ = model.sw_encoder(sw_seq[:1])
            h_sw_shared = h_sw_shared.expand(coords.shape[0], -1).contiguous()
            Ne_fused, log_var, _ne_placeholder, Ne_delta, extras = model(
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
                _hmf2_pred = extras.get('peak_params', {}).get('hmF2')
                if _hmf2_pred is not None and iri_peak_batch is not None:
                    _hmf2_str = (f"hmF2_fused={_hmf2_pred.mean().item():.1f}km"
                                 f"(IRI={iri_peak_batch[:,0].mean().item():.1f}km"
                                 f" diff={(_hmf2_pred - iri_peak_batch[:,0]).abs().mean().item():.1f}km)")
                elif _hmf2_pred is not None:
                    _hmf2_str = f"hmF2_fused={_hmf2_pred.mean().item():.1f}km"
                else:
                    _hmf2_str = "hmF2=N/A"
                _target_mean = target_ne.mean().item()
                _delta_vs_target = _fused - _target_mean   # 正=过高估计，负=低估
                print(f"  [DIAG b{batch_idx:>5}] "
                      f"gate={_g:.3f}(d={_gd:.3f}/p={_gp:.3f})  "
                      f"|Ne_delta|={_delta:.4f}  delta>0={_delta_pos:.2f}  "
                      f"bkg={_bkg:.3f}  fused={_fused:.3f}  target={_target_mean:.3f}  "
                      f"fused-target={_delta_vs_target:+.3f}  "
                      f"{_hmf2_str}")

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
            _peak      = extras.get('peak_params', {})
            hmF2_fused = _peak.get('hmF2')   # [B] 融合峰高（IRI+GIRO修正）
            NmF2_fused = _peak.get('NmF2')
            physics_loss, phy_dict = combined_mdia_physics_loss(
                pred_ne=Ne_fused,
                ne_bkg=extras['ne_bkg'],
                ne_residual=Ne_delta,
                coords=coords,
                lat=coords[:, 0].detach(),
                hmF2_pred=hmF2_fused,
                NmF2_pred=NmF2_fused,
                sin_I=extras.get('sin_I'),
                w_bkg=config.get('w_bkg', 0.1),
                w_uplift=config.get('w_uplift', 0.0),
                w_depletion=config.get('w_depletion', 0.0),
                w_eia_crest=config.get('w_eia_crest', 0.0),
                uplift_threshold_km=config.get('uplift_threshold_km', 380.0),
                # P0-C: 高度自适应 bkg 权重
                use_adaptive_bkg=True,
                w_bkg_low=config.get('w_bkg_low', 0.25),
                w_bkg_high=config.get('w_bkg_high', 0.02),
                w_bkg_transition=config.get('w_bkg_transition', 250.0),
                w_bkg_sharpness=config.get('w_bkg_sharpness', 25.0),
                # v3.1: 夜间增强 + EIA 日间门控
                w_bkg_night=config.get('w_bkg_night', 0.0),
                uplift_lt_sigma=config.get('uplift_lt_sigma', 5.0),
                # run26 (A1): DA 不确定性派生的 IRI 信任度，per-sample 加权 bkg_trust
                trust_iri=trust_iri_global,
            )

            # ---- 峰场平滑（run65 P1：giro_mode mini-forward，coords_ps 独立 requires_grad）----
            # peak_field_smooth_loss 需要 coords 梯度（grad path: coords → DualFreqSpatialNet →
            # h_spatial → PeakHead → hmF2）。P1 移除主批次 coords.requires_grad，改用 n_s 点的
            # 独立 mini-forward（giro_mode=True 仅运行 SW + Spatial + PeakHead，~10× 更快）。
            w_ps = config.get('w_peak_smooth', 0.0)
            loss_peak_smooth = torch.tensor(0.0, device=device)
            if w_ps > 0:
                n_s = min(config.get('peak_smooth_samples', 256), coords.shape[0])
                idx_ps = torch.randperm(coords.shape[0], device=device)[:n_s]
                coords_ps = coords[idx_ps].detach().clone().requires_grad_(True)
                iri_peak_ps = iri_peak_batch[idx_ps] if iri_peak_batch is not None else None
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, _, _, _, extras_ps = model(
                        coords_ps, sw_seq[idx_ps],
                        precomputed_h_sw=h_sw_shared[idx_ps].detach(),
                        iri_peak=iri_peak_ps,
                        giro_mode=True,
                    )
                hmF2_ps = extras_ps.get('peak_params', {}).get('hmF2')
                if hmF2_ps is not None:
                    loss_peak_smooth = peak_field_smooth_loss(
                        hmF2_ps, coords_ps,
                        sin_I=extras_ps.get('sin_I'),
                        cos_I=extras_ps.get('cos_I'),
                        lat_geo_deg=coords_ps[:, 0],
                    )
                    physics_loss = physics_loss + w_ps * loss_peak_smooth
            phy_dict['peak_smooth'] = loss_peak_smooth.item()

            # ---- run41: Ne 垂直方向二阶导平滑（直接约束 Ne_fused 在 alt 方向的高频波动）----
            w_nv = config.get('w_ne_vert_smooth', 0.0)
            loss_ne_vert = torch.tensor(0.0, device=device)
            if w_nv > 0:
                loss_ne_vert = ne_vert_smooth_loss(Ne_fused, coords)
                physics_loss = physics_loss + w_nv * loss_ne_vert
            phy_dict['ne_vert_smooth'] = loss_ne_vert.item() if isinstance(loss_ne_vert, torch.Tensor) else float(loss_ne_vert)

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
            if w_pa > 0 and hmF2_fused is not None:
                coords_peak = coords.detach().clone()
                coords_peak[:, 2] = extras['hmF2_det']   # detached 融合峰高，更稳定
                coords_peak.requires_grad_(True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    Ne_at_peak, _, _, _, _ = model(
                        coords_peak, sw_seq,
                        precomputed_h_sw=h_sw_shared.detach(),
                        iri_peak=iri_peak_batch,
                    )
                loss_pa = profile_peak_alignment_loss(
                    Ne_at_peak, coords_peak,
                    NmF2_pred=NmF2_fused,
                    w_val=config.get('w_profile_align_val', 0.0))
                physics_loss = physics_loss + w_pa * loss_pa
                phy_dict['profile_align'] = loss_pa.item()
            else:
                phy_dict['profile_align'] = 0.0

        else:
            physics_loss = 0.0
            phy_dict = {k: 0.0 for k in ['bkg', 'uplift', 'depletion',
                                           'eia_crest', 'physics_total',
                                           'profile_align', 'peak_smooth', 'iri_struct']}

        giro_loss = 0.0
        giro_dict = {'giro_hmf2': 0.0, 'giro_nmf2': 0.0, 'giro_ne_val': 0.0, 'giro_argmax': 0.0}
        giro_freq = config.get('giro_loss_freq', 1)
        compute_giro = use_giro and (batch_idx % giro_freq == 0)
        if compute_giro:
            _giro_loss, _giro_dict, hmf2_iter, nmf2_iter = compute_giro_loss(
                model, sw_manager,
                hmf2_iter, giro_hmf2_loader,
                nmf2_iter, giro_nmf2_loader,
                device, config,
                iri_peak_manager=iri_peak_manager,
            )
            giro_loss = _giro_loss
            giro_dict = _giro_dict

            # ---- 1-D: GIRO → Ne_fused 直接约束 + 弱 argmax（run61: 默认关闭，无邻域数据时跳过）----
            if config.get('w_giro_ne_val', 0) > 0 or config.get('w_giro_argmax', 0) > 0:
                _direct_loss, _direct_dict, giro_direct_iter = _compute_giro_direct_loss(
                    model, giro_direct_iter, giro_hmf2_loader,
                    sw_manager, iri_peak_manager, device, config)
                giro_loss = giro_loss + _direct_loss
                giro_dict['giro_ne_val']  = _direct_dict.get('giro_ne_val', 0.0)
                giro_dict['giro_argmax']  = _direct_dict.get('giro_argmax', 0.0)

        total_loss = config.get('w_obs', 1.0) * loss_main + physics_loss + giro_loss

        # ---- run22: L_shape — Pearson-r 形态约束（每批次，IRI 形态守恒）----
        # run26 (A1): trust_iri 加权 — IRI 主导时强守形态，FY 主导时弱化（允许偏离错误的 IRI）
        w_shape = config.get('w_shape', 0.05)
        l_shape_val = torch.tensor(0.0, device=device)
        if w_shape > 0:
            l_shape_val = pearson_r_shape_loss(
                Ne_fused, extras['ne_bkg'], trust_iri=trust_iri_global)
            total_loss  = total_loss + w_shape * l_shape_val

        # ---- run25: L_peak_iri — DA 自动加权 IRI 峰锚定（无新硬编码 w_*）----
        # trust_iri = (1 - K_FY.mean()).detach()  → IRI 信任度，由 DA 不确定性自动决定
        # hmF2: IRI 值可信 → MSE 锚定（按 trust_iri 加权）
        # NmF2: IRI 系统偏差仅 Pearson 可信 → banded Pearson 形状（不锚定值）
        l_peak_iri_hmf2 = torch.tensor(0.0, device=device)
        l_peak_iri_nmf2 = torch.tensor(0.0, device=device)
        trust_iri_mean_val = 0.0
        _peak_p = extras.get('peak_params', {})
        _hmF2_f = _peak_p.get('hmF2')
        _NmF2_f = _peak_p.get('NmF2')
        if (_hmF2_f is not None and _NmF2_f is not None
                and trust_iri_global is not None
                and iri_peak_batch is not None):
            # run26: 复用 trust_iri_global（已在 forward 后统一计算）
            trust_iri = trust_iri_global
            trust_iri_mean_val = trust_iri.mean().item()

            # hmF2 anchor（归一化 km → unitless）
            hmF2_resid = (_hmF2_f - iri_peak_batch[:, 0]) / 100.0              # [B]
            l_peak_iri_hmf2 = (trust_iri * hmF2_resid.pow(2)).mean()

            # NmF2 banded Pearson（3 带：低/中/高纬，min_samples=10）
            l_peak_iri_nmf2 = trust_iri.mean() * banded_pearson_loss(
                _NmF2_f,
                iri_peak_batch[:, 1],
                coords[:, 0],     # lat_geo_deg
            )

            total_loss = total_loss + l_peak_iri_hmf2 + l_peak_iri_nmf2

        # ---- Gate 熵正则（防止 gate 崩塌至 0 或 1）----
        # 注：使用 gate_data（sigmoid 输出，恒 ∈ (0,1)）而非合并 gate，
        # 因为修改 C 的 lt_gate_modulation 可使合并 gate 超过 1，
        # 导致 log(1 - gate + eps) = log(负数) = NaN。
        gate = extras.get('gate')
        gate_data_ent = extras.get('gate_data')   # sigmoid 输出，恒 ∈ (0,1)
        _gate_for_ent = gate_data_ent if gate_data_ent is not None else gate
        if _gate_for_ent is not None and config.get('w_gate_entropy', 0.0) > 0:
            eps = 1e-6
            gate_ent = (_gate_for_ent * torch.log(_gate_for_ent + eps) +
                        (1 - _gate_for_ent) * torch.log(1 - _gate_for_ent + eps)).mean()
            total_loss = total_loss + config['w_gate_entropy'] * gate_ent

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
                csm_Ne, csm_log_var, _, csm_Ne_delta, csm_extras = model(
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
                _csm_pk = csm_extras.get('peak_params', {})
                csm_phy, _ = combined_mdia_physics_loss(
                    pred_ne=csm_Ne,
                    ne_bkg=csm_extras['ne_bkg'],
                    ne_residual=csm_Ne_delta,
                    coords=csm_coords,
                    lat=csm_coords[:, 0].detach(),
                    hmF2_pred=_csm_pk.get('hmF2'),
                    NmF2_pred=_csm_pk.get('NmF2'),
                    sin_I=csm_extras.get('sin_I'),
                    w_bkg=config.get('w_bkg', 0.1),
                    w_uplift=config.get('w_uplift', 0.0),
                    w_depletion=config.get('w_depletion', 0.0),
                    w_eia_crest=config.get('w_eia_crest', 0.0),
                    uplift_threshold_km=config.get('uplift_threshold_km', 380.0),
                    use_adaptive_bkg=True,
                    w_bkg_low=config.get('w_bkg_low', 0.25),
                    w_bkg_high=config.get('w_bkg_high', 0.02),
                    w_bkg_transition=config.get('w_bkg_transition', 250.0),
                    w_bkg_sharpness=config.get('w_bkg_sharpness', 25.0),
                    w_bkg_night=config.get('w_bkg_night', 0.0),
                    uplift_lt_sigma=config.get('uplift_lt_sigma', 5.0),
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
            # COSMIC L_shape
            if w_shape > 0:
                csm_total = csm_total + w_shape * pearson_r_shape_loss(
                    csm_Ne, csm_extras['ne_bkg'], trust_iri=trust_iri_csm)
            # COSMIC L_peak_iri anchor
            _csm_pp = csm_extras.get('peak_params', {})
            _csm_hf = _csm_pp.get('hmF2')
            _csm_nf = _csm_pp.get('NmF2')
            if (_csm_hf is not None and _csm_nf is not None
                    and trust_iri_csm is not None and csm_iri_peak is not None):
                _csm_hr  = (_csm_hf - csm_iri_peak[:, 0]) / 100.0
                csm_total = csm_total + (trust_iri_csm * _csm_hr.pow(2)).mean()
                csm_total = csm_total + trust_iri_csm.mean() * banded_pearson_loss(
                    _csm_nf, csm_iri_peak[:, 1], csm_coords[:, 0])
            # COSMIC gate entropy
            _csm_gd  = csm_extras.get('gate_data')
            _csm_ge  = csm_extras.get('gate') if _csm_gd is None else _csm_gd
            if _csm_ge is not None and config.get('w_gate_entropy', 0.0) > 0:
                _eps = 1e-6
                csm_total = csm_total + config['w_gate_entropy'] * (
                    _csm_ge * torch.log(_csm_ge + _eps)
                    + (1 - _csm_ge) * torch.log(1 - _csm_ge + _eps)).mean()
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
        for k in ['bkg', 'uplift', 'depletion', 'eia_crest', 'physics_total',
                   'profile_align', 'peak_smooth', 'iri_struct']:
            stats[k] += phy_dict.get(k, 0.0)
        stats['giro_hmf2']   += giro_dict.get('giro_hmf2', 0.0)
        stats['giro_nmf2']   += giro_dict.get('giro_nmf2', 0.0)
        stats['giro_ne_val'] += giro_dict.get('giro_ne_val', 0.0)
        stats['giro_argmax'] += giro_dict.get('giro_argmax', 0.0)
        if gate is not None:
            stats['gate_mean'] += gate.mean().item()
        # P0-D 监控
        stats['alt_weight_mean'] += alt_weight.mean().item()
        stats['low_alt_frac']    += (coords[:, 2] < 200).float().mean().item()
        # run22 DA 监控
        stats['l_shape'] += l_shape_val.item() if isinstance(l_shape_val, torch.Tensor) else float(l_shape_val)
        # run25 DA 自动加权 IRI 峰锚定监控
        stats['peak_iri_hmf2']  += l_peak_iri_hmf2.item() if isinstance(l_peak_iri_hmf2, torch.Tensor) else 0.0
        stats['peak_iri_nmf2']  += l_peak_iri_nmf2.item() if isinstance(l_peak_iri_nmf2, torch.Tensor) else 0.0
        stats['trust_iri_mean'] += trust_iri_mean_val
        # run26 (B2): sample-adaptive α 监控
        with torch.no_grad():
            _ae = extras.get('alpha_eff')
            if _ae is not None and isinstance(_ae, torch.Tensor) and _ae.numel() > 0:
                stats['alpha_eff_mean'] += _ae.mean().item()
        with torch.no_grad():
            if extras.get('K_FY') is not None:
                stats['K_FY_mean']       += extras['K_FY'].mean().item()
                stats['K_vert_mean']     += extras['K_vert'].mean().item()
                stats['b_mean']          += extras['b'].mean().item()
                stats['r_fy_mean']       += extras['r_fy'].mean().item()
                stats['innov_FY_norm']   += extras['innov_FY'].norm(dim=-1).mean().item()
                stats['innov_vert_norm'] += extras['innov_vert'].norm(dim=-1).mean().item()
            # run28 NeuralETKFLayer ensemble 监控（per-batch 累加，per-epoch 平均）
            _infl = extras.get('inflation_scale')
            if _infl is not None:
                stats['inflation'] += _infl.item()
            # run59 监控：参考 R
            _r_ref_FY = extras.get('r_ref_FY')
            if _r_ref_FY is not None:
                stats['r_ref_FY']   += _r_ref_FY.detach().item()
            _r_ref_v = extras.get('r_ref_vert')
            if _r_ref_v is not None:
                stats['r_ref_vert'] += _r_ref_v.detach().item()
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
            _hf_pred = extras.get('peak_params', {}).get('hmF2')
            if _hf_pred is not None and iri_peak_batch is not None:
                stats['hmf2_iri_diff'] += (_hf_pred - iri_peak_batch[:, 0]).abs().mean().item()
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
             giro_val_loaders=None, sw_manager=None, iri_peak_manager=None,
             cosmic_val_loader=None):
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

    # 阶段一：GIRO 验证集峰值指标（hmF2/NmF2 MAE）
    # GIRO batch shape: [B, 5] = [lat_geo, lon_geo, rel_hour, value, lat_aacgm]
    # coords_g: [B, 5] = [lat_geo, lon_geo, alt_dummy, rel_hour, lat_aacgm]
    _DUMMY_ALT = 300.0
    val_giro_peak_mae = float('inf')
    if giro_val_loaders is not None and sw_manager is not None:
        giro_hmf2_val, giro_nmf2_val = giro_val_loaders
        hmf2_errs, nmf2_errs = [], []
        with torch.no_grad():
            if giro_hmf2_val is not None:
                for raw_batch in giro_hmf2_val:
                    raw_batch = raw_batch.to(device)
                    lat_g, lon_g, rel_h, hmf2_target = (
                        raw_batch[:, 0], raw_batch[:, 1],
                        raw_batch[:, 2], raw_batch[:, 3])
                    alt_d  = torch.full_like(lat_g, _DUMMY_ALT)
                    lat_aa = raw_batch[:, 4]
                    coords_g = torch.stack([lat_g, lon_g, alt_d, rel_h, lat_aa], dim=1)
                    sw_seq_g = sw_manager.get_drivers_sequence(rel_h)
                    iri_peak_v = None
                    if iri_peak_manager is not None:
                        iri_peak_v = iri_peak_manager.get_iri_peak(coords_g)
                    _, _, _, _, g_extras = model(coords_g, sw_seq_g,
                                                 giro_mode=True, iri_peak=iri_peak_v)
                    pred_hmf2 = g_extras.get('peak_params', {}).get('hmF2')
                    if pred_hmf2 is not None:
                        hmf2_errs.append((pred_hmf2 - hmf2_target).abs().cpu())
            if giro_nmf2_val is not None:
                for raw_batch in giro_nmf2_val:
                    raw_batch = raw_batch.to(device)
                    lat_g, lon_g, rel_h, nmf2_target = (
                        raw_batch[:, 0], raw_batch[:, 1],
                        raw_batch[:, 2], raw_batch[:, 3])
                    alt_d  = torch.full_like(lat_g, _DUMMY_ALT)
                    lat_aa = raw_batch[:, 4]
                    coords_g = torch.stack([lat_g, lon_g, alt_d, rel_h, lat_aa], dim=1)
                    sw_seq_g = sw_manager.get_drivers_sequence(rel_h)
                    iri_peak_v = None
                    if iri_peak_manager is not None:
                        iri_peak_v = iri_peak_manager.get_iri_peak(coords_g)
                    _, _, _, _, g_extras = model(coords_g, sw_seq_g,
                                                 giro_mode=True, iri_peak=iri_peak_v)
                    pred_nmf2 = g_extras.get('peak_params', {}).get('NmF2')
                    if pred_nmf2 is not None:
                        nmf2_errs.append((pred_nmf2 - nmf2_target).abs().cpu())

        hmf2_mae = torch.cat(hmf2_errs).mean().item() if hmf2_errs else 0.0
        nmf2_mae = torch.cat(nmf2_errs).mean().item() if nmf2_errs else 0.0
        # 归一化：hmF2 MAE (km) / 50 使量级与 NmF2 MAE (log10) 相当
        val_giro_peak_mae = hmf2_mae / 50.0 + nmf2_mae
        val_dict['val_giro_peak_mae'] = val_giro_peak_mae
        val_dict['val_hmf2_mae'] = hmf2_mae
        val_dict['val_nmf2_mae'] = nmf2_mae

    return val_ne_loss, val_giro_peak_mae, val_dict


# ======================== 1-D: GIRO → Ne_fused 直接约束 ========================

def _compute_giro_direct_loss(model, direct_iter, giro_hmf2_loader,
                               sw_manager, iri_peak_manager, device, config):
    """
    GIRO → Ne_fused 直接约束（Plan 1-D）

    在 GIRO 观测峰高 hmF2_obs 处运行完整前向（非 giro_mode），约束：
      L_giro_ne_val  : MSE(Ne_fused(hmF2_obs), NmF2_fused.detach())  峰值自洽
      L_giro_argmax  : relu(Ne_up - Ne_pk) + relu(Ne_dn - Ne_pk)    弱 argmax 峰结构

    使用 precomputed_h_sw 避免重复 SW 编码；3B 坐标拼接减少 forward 次数。

    Returns:
        l_direct:    scalar Tensor 或 0.0
        direct_dict: {'giro_ne_val': float, 'giro_argmax': float}
        direct_iter: (更新后的迭代器)
    """
    w_ne_val = config.get('w_giro_ne_val', 0.0)
    w_argmax = config.get('w_giro_argmax', 0.0)
    Dh       = config.get('giro_argmax_dh', 10.0)

    zero_dict = {'giro_ne_val': 0.0, 'giro_argmax': 0.0}

    if (w_ne_val <= 0 and w_argmax <= 0) or giro_hmf2_loader is None:
        return 0.0, zero_dict, direct_iter

    # 从 hmF2 loader 获取一个 GIRO batch
    try:
        raw_batch = next(direct_iter)
    except StopIteration:
        direct_iter = iter(giro_hmf2_loader)
        try:
            raw_batch = next(direct_iter)
        except StopIteration:
            return 0.0, zero_dict, direct_iter

    raw_batch = raw_batch.to(device)
    lat_g    = raw_batch[:, 0]
    lon_g    = raw_batch[:, 1]
    rel_h    = raw_batch[:, 2]
    hmF2_obs = raw_batch[:, 3]   # GIRO 观测峰高 [km]
    lat_aa   = raw_batch[:, 4]

    # 构建 3B coords：hmF2_obs / hmF2_obs+Δh / hmF2_obs-Δh
    def _make_coords(alt_vals):
        return torch.stack([lat_g, lon_g, alt_vals, rel_h, lat_aa], dim=1)

    coords_pk = _make_coords(hmF2_obs)
    coords_up = _make_coords(hmF2_obs + Dh)
    coords_dn = _make_coords(hmF2_obs - Dh)
    coords_3B = torch.cat([coords_pk, coords_up, coords_dn])     # [3B, 5]

    # 预计算 h_sw（GIRO 站点共享相同时刻，拼接复用）
    sw_seq_g = sw_manager.get_drivers_sequence(rel_h)             # [B, seq_len, 2]
    with torch.no_grad():
        h_sw_g, _, _ = model.sw_encoder(sw_seq_g)                 # [B, 64]
    h_sw_3B  = h_sw_g.repeat(3, 1)                                # [3B, 64]
    sw_seq_3B = sw_seq_g.repeat(3, 1, 1)                          # [3B, seq_len, 2]

    # IRI 峰参数（3B coords）
    iri_peak_3B = None
    if iri_peak_manager is not None:
        with torch.no_grad():
            iri_peak_3B = iri_peak_manager.get_iri_peak(coords_3B)

    # 完整前向（非 giro_mode），复用 h_sw
    Ne_3B, _, _, _, extras_3B = model(
        coords_3B, sw_seq_3B,
        precomputed_h_sw=h_sw_3B,
        iri_peak=iri_peak_3B,
    )

    B_g   = coords_pk.shape[0]
    Ne_pk = Ne_3B[:B_g].squeeze(-1)           # [B]
    Ne_up = Ne_3B[B_g:2*B_g].squeeze(-1)      # [B]
    Ne_dn = Ne_3B[2*B_g:].squeeze(-1)         # [B]

    l_direct   = torch.tensor(0.0, device=device)
    l_ne_val_v = 0.0
    l_argmax_v = 0.0

    # L_giro_ne_val：Ne_fused 在观测峰高处应与 PeakHead 预测幅度一致
    # NmF2_fused（从 full forward）是 PeakHead 的预测，已被 GIRO NmF2 监督训练
    # → 间接将 GIRO NmF2 观测信息注入 3D 场
    if w_ne_val > 0:
        nmf2_fused = extras_3B.get('peak_params', {}).get('NmF2')
        if nmf2_fused is not None:
            nmf2_target = nmf2_fused[:B_g].detach()
            l_ne_val    = F.mse_loss(Ne_pk, nmf2_target)
            l_direct    = l_direct + w_ne_val * l_ne_val
            l_ne_val_v  = l_ne_val.item()

    # L_giro_argmax：弱 argmax——Ne_fused 在观测峰高处应为局部极大值
    # 只在违反峰结构时激活（relu），对正常峰形无梯度
    if w_argmax > 0:
        l_argmax   = (torch.relu(Ne_up - Ne_pk) +
                      torch.relu(Ne_dn - Ne_pk)).mean()
        l_direct   = l_direct + w_argmax * l_argmax
        l_argmax_v = l_argmax.item()

    return l_direct, {'giro_ne_val': l_ne_val_v, 'giro_argmax': l_argmax_v}, direct_iter


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
            total_hours=config['total_hours'],
            device=device,
        )
        print('  IRI 峰参数管理器已加载（作为 PeakHead 背景输入）')
    else:
        print('  警告: iri_hmf2_path/iri_nmf2_path 未配置或文件不存在，'
              'PeakHead 将使用中性值 (300km, 11.5)')

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

    # ========== 步骤 3.5: GIRO 数据 ==========
    print('\n[步骤 3.5] 载入 GIRO 电离层测高仪数据...')
    giro_hmf2_loader, giro_nmf2_loader = build_giro_loaders(config, device=None)
    if giro_hmf2_loader is None and giro_nmf2_loader is None:
        print('  未配置 GIRO 数据，跳过')

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
    _resume_ckpt   = config.get('resume_ckpt')
    _resume_epochs = config.get('resume_epochs')
    if _resume_ckpt:
        if os.path.exists(_resume_ckpt):
            model.load_state_dict(torch.load(_resume_ckpt, map_location=device), strict=True)
            print(f'\n[步骤 4.5] 已从检查点加载权重: {_resume_ckpt}')
            _cosmic_dead = (
                torch.count_nonzero(
                    model.cosmic_obs_encoder.input_proj.weight).item() == 0
                and torch.count_nonzero(
                    model.proj_pre.weight[:, -model.kalman_layer.d_model:]).item() == 0
            )
            if config.get('w_cosmic', 0.0) > 0 and _cosmic_dead:
                model._initialize_cosmic_bootstrap()
                print('  检测到旧 checkpoint COSMIC 全零入口，已仅重启 COSMIC 梯度路径')
        else:
            print(f'\n[步骤 4.5] 警告: 检查点不存在 ({_resume_ckpt})，从头训练')
        if _resume_epochs is not None:
            config = dict(config)
            config['epochs'] = int(_resume_epochs)
            print(f'  续训轮数: {config["epochs"]}')

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

    # ========== 步骤 6: 训练循环 ==========
    print(f'\n[步骤 6] 开始训练 ({config["epochs"]} epochs)...')
    warmup = config.get('uncertainty_warmup_epochs', 5)
    if config.get('use_uncertainty'):
        print(f'  前 {warmup} 轮使用 Huber loss，之后启用 NLL + 不确定性学习')
    print(f'  物理损失每 {config.get("physics_loss_freq", 5)} 个 batch 计算一次')

    train_losses, val_losses, history = [], [], []
    best_val_ne      = float('inf')
    best_val_peak    = float('inf')
    best_epoch       = None
    ne_patience_counter   = 0
    peak_patience_counter = 0
    best_ckpt = os.path.join(config['save_dir'], 'best_fsia_model.pth')

    # 构建 GIRO 验证集（用于双域早停）
    giro_hmf2_val_loader = None
    giro_nmf2_val_loader = None
    try:
        from .giro_dataloader import build_giro_loaders as _build_giro
    except ImportError:
        from giro_dataloader import build_giro_loaders as _build_giro
    _val_cfg = dict(config)
    _val_cfg['giro_batch_size'] = min(config.get('giro_batch_size', 256), 512)
    giro_hmf2_val_loader, giro_nmf2_val_loader = _build_giro(_val_cfg, device=None)

    for epoch in range(config['epochs']):
        epoch_t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['epochs']}")

        cosmic_train_iter = iter(cosmic_train_loader) if cosmic_train_loader is not None else None
        train_loss, train_dict = train_one_epoch(
            model, train_loader, batch_processor,
            optimizer, device, config, epoch, scaler,
            giro_hmf2_loader=giro_hmf2_loader,
            giro_nmf2_loader=giro_nmf2_loader,
            sw_manager=sw_manager,
            iri_peak_manager=iri_peak_manager,
            cosmic_train_loader=cosmic_train_loader,
            cosmic_iter=cosmic_train_iter,
        )
        train_losses.append(train_loss)

        giro_val_loaders = (giro_hmf2_val_loader, giro_nmf2_val_loader) \
            if (giro_hmf2_val_loader is not None or giro_nmf2_val_loader is not None) \
            else None
        val_loss, val_peak_mae, val_metrics = validate(
            model, val_loader, batch_processor, device, config,
            giro_val_loaders=giro_val_loaders,
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
        print(f"    EIA Uplift:   {train_dict['uplift']:.6f}")
        print(f"    Trough Depl:  {train_dict['depletion']:.6f}")
        if giro_hmf2_loader is not None:
            print(f"    GIRO hmF2:    {train_dict['giro_hmf2']:.4f}")
            print(f"    GIRO Ne_val:  {train_dict['giro_ne_val']:.4f}  "
                  f"argmax={train_dict['giro_argmax']:.4f}")
        if giro_nmf2_loader is not None:
            print(f"    GIRO NmF2:    {train_dict['giro_nmf2']:.6f}")
        if train_dict.get('iri_struct', 0.0) > 0:
            print(f"    IRI Struct:   {train_dict['iri_struct']:.4f}")
        if train_dict.get('l_shape', 0.0) > 0:
            print(f"    L_shape:      {train_dict['l_shape']:.4f}")
        # run22 DA 监控
        _k_fy = train_dict.get('K_FY_mean', 0.0)
        _k_vt = train_dict.get('K_vert_mean', 0.0)
        _b_da = train_dict.get('b_mean', 0.0)
        _r_fy = train_dict.get('r_fy_mean', 0.0)
        print(f"  [DA] K_FY={_k_fy:.3f}  K_vert={_k_vt:.3f}  b={_b_da:.3f}  r_fy={_r_fy:.3f}"
              f"  innov_FY={train_dict.get('innov_FY_norm',0.0):.3f}"
              f"  innov_vert={train_dict.get('innov_vert_norm',0.0):.3f}")
        print(f"  Val Ne Loss:  {val_loss:.6f}")
        print(f"    MAE={val_metrics['mae']:.6f}  RMSE={val_metrics['rmse']:.6f}"
              f"  R²={val_metrics['r2']:.4f}")
        if val_peak_mae < float('inf'):
            print(f"  Val Peak MAE: {val_peak_mae:.4f}"
                  f"  (hmF2={val_metrics.get('val_hmf2_mae', 0):.1f}km"
                  f"  NmF2={val_metrics.get('val_nmf2_mae', 0):.4f})")

        tau_kp_val    = model.sw_encoder.tau_kp.item()
        tau_solar_val = model.sw_encoder.tau_solar.item()
        gate_mean_v   = train_dict['gate_mean']
        print(f"  τ_kp:{tau_kp_val:.2f}h  τ_solar:{tau_solar_val:.2f}h"
              f"  gate_mean:{gate_mean_v:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        # [DIAG] 步骤一：融合诊断摘要
        _bkg_mse_ratio = train_dict['bkg'] / max(train_dict['mse'], 1e-8)
        _hmf2_warn = " ⚠ PeakHead过冲!" if train_dict['hmf2_iri_diff'] > 20.0 else ""
        _bias_warn = " ⚠ 全局正偏置!" if train_dict['delta_pos_frac'] > 0.85 else ""
        print(f"  [DIAG] |Ne_delta|={train_dict['ne_delta_abs']:.4f}  "
              f"delta>0={train_dict['delta_pos_frac']:.3f}{_bias_warn}  "
              f"bkg={train_dict['ne_bkg_mean']:.3f}  fused={train_dict['ne_fused_mean']:.3f}  "
              f"bkg/mse={_bkg_mse_ratio:.2f}x  "
              f"hmF2_iri_diff={train_dict['hmf2_iri_diff']:.1f}km{_hmf2_warn}")

        history.append({
            'epoch': epoch + 1,
            'train_mse': train_dict['mse'],
            'val_mse':   val_loss,
            'val_cosmic_mse': val_metrics.get('val_csm_mse'),
            'val_combined': val_metrics.get('val_combined', val_loss),
            'train_nll': train_dict['nll'],
            'total_loss': train_dict['total'],
            'bkg':              train_dict['bkg'],
            'uplift':           train_dict['uplift'],
            'depletion':        train_dict['depletion'],
            'profile_align':    train_dict['profile_align'],
            'peak_smooth':      train_dict['peak_smooth'],
            'iri_struct':       train_dict['iri_struct'],
            'giro_hmf2':        train_dict['giro_hmf2'],
            'giro_nmf2':        train_dict['giro_nmf2'],
            'giro_ne_val':      train_dict['giro_ne_val'],
            'giro_argmax':      train_dict['giro_argmax'],
            'val_giro_peak_mae': val_peak_mae if val_peak_mae < float('inf') else None,
            'tau_kp':           tau_kp_val,
            'tau_solar':        tau_solar_val,
            'gate_mean':        gate_mean_v,
            'l_shape':          train_dict.get('l_shape', 0.0),
            'K_FY_mean':        train_dict.get('K_FY_mean', 0.0),
            'K_vert_mean':      train_dict.get('K_vert_mean', 0.0),
            'b_mean':           train_dict.get('b_mean', 0.0),
            'r_fy_mean':        train_dict.get('r_fy_mean', 0.0),
            # run28 NeuralETKFLayer
            'inflation':        train_dict.get('inflation', 0.0),
            'member_w_max':    train_dict.get('member_w_max', 0.0),
        })

        if scheduler:
            scheduler.step()

        # 双域早停：Ne MSE（+ COSMIC，run64 val_combined）+ GIRO 峰值指标各自独立计数
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

        if val_peak_mae < best_val_peak:
            best_val_peak          = val_peak_mae
            peak_patience_counter  = 0
        else:
            peak_patience_counter += 1

        print(f"  [早停] ne_pat={ne_patience_counter}/{config.get('ne_patience', 4)}"
              f"  peak_pat={peak_patience_counter}/{config.get('peak_patience', 4)}")

        plot_training_curves(
            history,
            save_path=os.path.join(config['save_dir'], 'fsia_training_curves.png')
        )

        if (epoch + 1) % config.get('save_interval', 5) == 0:
            ckpt_path = os.path.join(config['save_dir'], f'fsia_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)

        if config.get('early_stopping'):
            if (ne_patience_counter >= config.get('ne_patience', 4) or
                    peak_patience_counter >= config.get('peak_patience', 4)):
                print(f'\n[早停] 触发 ne_pat={ne_patience_counter}'
                      f' peak_pat={peak_patience_counter}')
                break

    print(f"\n{'='*60}")
    print(f"训练完成！最佳 Ne MSE: {best_val_ne:.6f}  最佳峰值 MAE: {best_val_peak:.4f}")

    hist_path = os.path.join(config['save_dir'], 'fsia_training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存: {hist_path}")

    summary = {
        'best_epoch': best_epoch,
        'best_val_combined': best_val_ne,
        'best_val_peak_mae': (best_val_peak
                              if best_val_peak < float('inf') else None),
        'best_epoch_metrics': (history[best_epoch - 1]
                               if best_epoch is not None else None),
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
