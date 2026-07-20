"""
FSIA-INR GIRO 峰参数评估脚本（run65）

对全部 GIRO 站点位置运行模型，提取 Ne_fused 廓线峰值，与 IRI 基准及 GIRO 真值对比。

高度采样策略（粗-精两阶段）：
    Pass 1：250–500 km，步长 10 km（26 层），确定粗峰位置
    Pass 2：粗峰 ±10 km，步长  1 km（21 层），精确到 1 km
    总计 47 次前向 vs 原始 251 次，约 5× 加速

FY 邻域（run61 — 剖面缓存）：
    每个 GIRO 站点 (lat/lon/time) 仅调用一次 query_profiles_only；
    47 个高度层共享同一套 top-K 剖面，仅重新计算 dalt_f。

COSMIC-2 邻域（run64）：
    若 config['cosmic_path'] 已配置，同样预查 COSMIC 剖面并传入模型。
    路径缺失或构建失败时自动退化（COSMIC 贡献为零）。

IRI 基准：IRIPeakManager.get_iri_peak()，三线性插值于 3h×1°×2° 预计算网格。

指标：RMSE / BIAS / MAE / Pearson R / CCC（hmF2 单位 km，NmF2 单位 log10）

输出：
    <save_dir>/giro_peak_eval/giro_peak_report.txt
    <save_dir>/giro_peak_eval/giro_peak_density.png

使用方式：
    cd FSIA_INR18
    python evaluate_giro_peak.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'inr_modules'))

# =====================================================================
# 用户配置（与 main_fsia.py 的 save_dir 保持一致）
# =====================================================================
_SAVE_DIR  = r'D:\code11\IRI01\IRI03\INR1-1\FSIA_INR18\checkpoints_fsia\run65'
_CKPT_PATH = os.path.join(_SAVE_DIR, 'best_fsia_model.pth')

# 每批处理的 GIRO 站点数（每批 = N_STA × 47 次模型前向）
_N_STA_BATCH = 8

# 两阶段高度参数
_ALT_COARSE_START = 250.0   # km
_ALT_COARSE_STOP  = 500.0   # km
_ALT_COARSE_STEP  =  10.0   # km → 26 层
_ALT_FINE_HALF    =  10     # ±10 km → 21 层
# =====================================================================


def _expand_cache(cached: dict, n: int) -> dict:
    """
    将 query_profiles_only 返回的 [B, ...] 缓存沿 axis=0 每条重复 n 次 → [B*n, ...]。
    用于把 B 个站点缓存扩展到 B*n 个（站点×高度层）的扁平批次。
    'K_p' 为标量，保持不变。
    """
    out = {}
    for k, v in cached.items():
        if k == 'K_p':
            out[k] = v
        else:
            out[k] = np.repeat(v, n, axis=0)
    return out


def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient"""
    mu_t = np.mean(y_true)
    mu_p = np.mean(y_pred)
    s2_t = np.var(y_true)
    s2_p = np.var(y_pred)
    s_tp = np.mean((y_true - mu_t) * (y_pred - mu_p))
    return float(2.0 * s_tp / (s2_t + s2_p + (mu_t - mu_p) ** 2 + 1e-12))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """返回 (RMSE, BIAS, MAE, Pearson_R, CCC)"""
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    mae  = float(np.mean(np.abs(diff)))
    r, _ = pearsonr(y_true, y_pred)
    ccc  = _ccc(y_true, y_pred)
    return rmse, bias, mae, float(r), ccc


def evaluate_giro_peak(config=None):
    from inr_modules.config_mdia import get_config_mdia
    from inr_modules.mdia.fsia_model import FSIA_INR_Model
    from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.data_managers.FY_dataloader import (
        FYNeighborhoodIndex, COSMICNeighborhoodIndex,
    )

    if config is None:
        config = get_config_mdia()

    device = torch.device(config['device'])

    # ----------------------------------------------------------------
    # 初始化组件
    # ----------------------------------------------------------------
    print('[评估] 初始化 IRIPeakManager...')
    iri_mgr = IRIPeakManager(
        config['iri_hmf2_path'], config['iri_nmf2_path'], device=str(device))

    print('[评估] 初始化 SpaceWeatherManager...')
    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device,
    )

    print('[评估] 构建 FYNeighborhoodIndex...')
    fy_nb = FYNeighborhoodIndex(config['fy_path'], config)

    # ----------------------------------------------------------------
    # COSMIC-2 邻域索引（run64，可选）
    # ----------------------------------------------------------------
    cosmic_nb = None
    cosmic_path = config.get('cosmic_path', '')
    if cosmic_path and os.path.exists(cosmic_path):
        try:
            cosmic_nb = COSMICNeighborhoodIndex(cosmic_path, config)
            print(f'[评估] COSMICNeighborhoodIndex 已构建 '
                  f'(dt={cosmic_nb.dt}h, k_prof={cosmic_nb.k_prof})')
        except Exception as _e:
            print(f'[评估] 警告: COSMICNeighborhoodIndex 构建失败 ({_e})，跳过 COSMIC')
    else:
        print('[评估] cosmic_path 未配置或不存在，COSMIC 邻域贡献为零')

    print('[评估] 构建 IRI 神经代理...')
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    proxy_state = torch.load(config['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(proxy_state)
    iri_proxy.eval()

    print('[评估] 加载模型...')
    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=config).to(device)
    ckpt  = torch.load(_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f'  checkpoint : {_CKPT_PATH}')

    # ----------------------------------------------------------------
    # 加载 GIRO 真值
    # 列格式：[lat_geo, lon_geo, rel_hour, value, lat_aacgm]
    # ----------------------------------------------------------------
    giro_h = np.load(config['giro_hmf2_path']).astype(np.float32)  # hmF2 km
    giro_n = np.load(config['giro_nmf2_path']).astype(np.float32)  # NmF2 log10
    print(f'[评估] GIRO hmF2 样本数: {len(giro_h):,}')
    print(f'[评估] GIRO NmF2 样本数: {len(giro_n):,}')

    # ----------------------------------------------------------------
    # 高度网格定义
    # ----------------------------------------------------------------
    alts_coarse = np.arange(
        _ALT_COARSE_START, _ALT_COARSE_STOP + _ALT_COARSE_STEP,
        _ALT_COARSE_STEP, dtype=np.float32)              # [26]
    n_c = len(alts_coarse)

    fine_offsets = np.arange(
        -_ALT_FINE_HALF, _ALT_FINE_HALF + 1, 1.0, dtype=np.float32)  # [21]
    n_f = len(fine_offsets)

    save_dir = os.path.join(_SAVE_DIR, 'giro_peak_eval')
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 核心：对一批 GIRO 数据运行两阶段廓线采样
    # giro_data: [N, 5]  列 [lat, lon, rel_hour, value, lat_aacgm]
    # 返回：hmF2_model [N], NmF2_model [N], hmF2_iri [N], NmF2_iri [N]
    # ----------------------------------------------------------------
    def _predict_peaks(giro_data: np.ndarray):
        N = len(giro_data)
        hmF2_model = np.full(N, np.nan, dtype=np.float32)
        NmF2_model = np.full(N, np.nan, dtype=np.float32)
        hmF2_iri   = np.full(N, np.nan, dtype=np.float32)
        NmF2_iri   = np.full(N, np.nan, dtype=np.float32)

        lats_all  = giro_data[:, 0]
        lons_all  = giro_data[:, 1]
        times_all = giro_data[:, 2]

        log_every = max(1, (N // 10 // _N_STA_BATCH) * _N_STA_BATCH)

        for sta_start in range(0, N, _N_STA_BATCH):
            batch_idx = np.arange(sta_start, min(sta_start + _N_STA_BATCH, N))
            Ns = len(batch_idx)

            lats  = lats_all [batch_idx].astype(np.float32)
            lons  = lons_all [batch_idx].astype(np.float32)
            times = times_all[batch_idx].astype(np.float32)

            # ---- IRI 基准（直接查询，不依赖高度）----
            coords_dummy_t = torch.from_numpy(
                np.stack([lats, lons, np.full(Ns, 300.0, dtype=np.float32), times], axis=1)
            ).to(device)
            with torch.no_grad():
                iri_peak_sta = iri_mgr.get_iri_peak(coords_dummy_t).cpu().numpy()  # [Ns, 2]
            hmF2_iri[batch_idx] = iri_peak_sta[:, 0]
            NmF2_iri[batch_idx] = iri_peak_sta[:, 1]

            # ---- FY Phase 1：剖面搜索（每站点仅一次）----
            coords_dummy_np = np.stack(
                [lats, lons, np.full(Ns, 300.0, dtype=np.float32), times], axis=1)
            cached_fy_sta = fy_nb.query_profiles_only(coords_dummy_np)  # [Ns, ...]

            # ---- COSMIC Phase 1：剖面搜索（run64，可选）----
            cached_csm_sta = None
            if cosmic_nb is not None:
                cached_csm_sta = cosmic_nb.query_profiles_only(coords_dummy_np)

            # ---- sw_seq（每时间点计算一次，后续 repeat_interleave 复用）----
            times_t    = torch.from_numpy(times).to(device)
            sw_seq_sta = sw_manager.get_drivers_sequence(times_t)    # [Ns, seq, 2]

            iri_peak_sta_t = torch.from_numpy(iri_peak_sta).to(device)  # [Ns, 2]

            # ==============================================================
            # Pass 1：粗高度网格（步长 10 km，26 层）
            # ==============================================================
            cached_fy_c = _expand_cache(cached_fy_sta, n_c)       # [Ns*n_c, ...]
            alts_q_c    = np.tile(alts_coarse, Ns)                 # [Ns*n_c]

            feats_fy_c, hobs_fy_c = fy_nb.featurize_with_cached(cached_fy_c, alts_q_c)

            if cached_csm_sta is not None:
                cached_csm_c = _expand_cache(cached_csm_sta, n_c)
                feats_csm_c, hobs_csm_c = cosmic_nb.featurize_with_cached(
                    cached_csm_c, alts_q_c)
                nb_feats_csm_c  = torch.from_numpy(feats_csm_c).to(device)
                has_obs_csm_c   = torch.from_numpy(hobs_csm_c).to(device)
            else:
                nb_feats_csm_c = None
                has_obs_csm_c  = None

            coords_c_np = np.stack([
                np.repeat(lats,  n_c),
                np.repeat(lons,  n_c),
                alts_q_c,
                np.repeat(times, n_c),
            ], axis=1).astype(np.float32)                          # [Ns*n_c, 4]

            coords_c_t  = torch.from_numpy(coords_c_np).to(device)
            sw_seq_c    = sw_seq_sta.repeat_interleave(n_c, dim=0)         # [Ns*n_c, seq, 2]
            iri_peak_c  = iri_peak_sta_t.repeat_interleave(n_c, dim=0)    # [Ns*n_c, 2]
            nb_feats_c  = torch.from_numpy(feats_fy_c).to(device)
            has_obs_c   = torch.from_numpy(hobs_fy_c).to(device)

            with torch.no_grad():
                Ne_c, _, _, _, _ = model(
                    coords_c_t, sw_seq_c,
                    iri_peak=iri_peak_c,
                    neighbors_feats=nb_feats_c,
                    has_obs=has_obs_c,
                    neighbors_feats_cosmic=nb_feats_csm_c,
                    has_obs_cosmic=has_obs_csm_c,
                )                                                  # [Ns*n_c, 1]

            Ne_c_np      = Ne_c.cpu().numpy().reshape(Ns, n_c)    # [Ns, 26]
            coarse_idx   = Ne_c_np.argmax(axis=1)                 # [Ns]
            coarse_alt   = alts_coarse[coarse_idx]                # [Ns] km

            # ==============================================================
            # Pass 2：精细高度网格（粗峰 ±10 km，步长 1 km，21 层）
            # ==============================================================
            fine_alts = np.clip(
                coarse_alt[:, None] + fine_offsets[None, :],
                _ALT_COARSE_START, _ALT_COARSE_STOP
            ).astype(np.float32)                                   # [Ns, 21]
            alts_q_f = fine_alts.ravel()                           # [Ns*n_f]

            cached_fy_f = _expand_cache(cached_fy_sta, n_f)
            feats_fy_f, hobs_fy_f = fy_nb.featurize_with_cached(cached_fy_f, alts_q_f)

            if cached_csm_sta is not None:
                cached_csm_f = _expand_cache(cached_csm_sta, n_f)
                feats_csm_f, hobs_csm_f = cosmic_nb.featurize_with_cached(
                    cached_csm_f, alts_q_f)
                nb_feats_csm_f  = torch.from_numpy(feats_csm_f).to(device)
                has_obs_csm_f   = torch.from_numpy(hobs_csm_f).to(device)
            else:
                nb_feats_csm_f = None
                has_obs_csm_f  = None

            coords_f_np = np.stack([
                np.repeat(lats,  n_f),
                np.repeat(lons,  n_f),
                alts_q_f,
                np.repeat(times, n_f),
            ], axis=1).astype(np.float32)                          # [Ns*n_f, 4]

            coords_f_t  = torch.from_numpy(coords_f_np).to(device)
            sw_seq_f    = sw_seq_sta.repeat_interleave(n_f, dim=0)
            iri_peak_f  = iri_peak_sta_t.repeat_interleave(n_f, dim=0)
            nb_feats_f  = torch.from_numpy(feats_fy_f).to(device)
            has_obs_f   = torch.from_numpy(hobs_fy_f).to(device)

            with torch.no_grad():
                Ne_f, _, _, _, _ = model(
                    coords_f_t, sw_seq_f,
                    iri_peak=iri_peak_f,
                    neighbors_feats=nb_feats_f,
                    has_obs=has_obs_f,
                    neighbors_feats_cosmic=nb_feats_csm_f,
                    has_obs_cosmic=has_obs_csm_f,
                )                                                  # [Ns*n_f, 1]

            Ne_f_np    = Ne_f.cpu().numpy().reshape(Ns, n_f)      # [Ns, 21]
            fine_idx   = Ne_f_np.argmax(axis=1)                   # [Ns]
            row_idx    = np.arange(Ns)

            hmF2_model[batch_idx] = fine_alts[row_idx, fine_idx]
            NmF2_model[batch_idx] = Ne_f_np  [row_idx, fine_idx]

            if sta_start % log_every == 0:
                print(f'  [进度] {min(sta_start + _N_STA_BATCH, N):>5}/{N} 站点完成')

        return hmF2_model, NmF2_model, hmF2_iri, NmF2_iri

    # ----------------------------------------------------------------
    # hmF2 评估
    # ----------------------------------------------------------------
    print('\n[评估] 运行 hmF2 廓线峰值计算...')
    hmF2_pred, _, hmF2_iri_h, _ = _predict_peaks(giro_h)
    hmF2_true = giro_h[:, 3]

    ok_h = np.isfinite(hmF2_pred) & np.isfinite(hmF2_true)
    m_iri_h = _metrics(hmF2_true[ok_h], hmF2_iri_h[ok_h])
    m_inr_h = _metrics(hmF2_true[ok_h], hmF2_pred[ok_h])

    # ----------------------------------------------------------------
    # NmF2 评估（GIRO NmF2 站点独立运行）
    # ----------------------------------------------------------------
    print('\n[评估] 运行 NmF2 廓线峰值计算...')
    _, NmF2_pred, _, NmF2_iri_n = _predict_peaks(giro_n)
    NmF2_true = giro_n[:, 3]

    ok_n = np.isfinite(NmF2_pred) & np.isfinite(NmF2_true)
    m_iri_n = _metrics(NmF2_true[ok_n], NmF2_iri_n[ok_n])
    m_inr_n = _metrics(NmF2_true[ok_n], NmF2_pred[ok_n])

    # ----------------------------------------------------------------
    # 文本报告
    # ----------------------------------------------------------------
    W = 75

    def _row(label, m):
        return (f'  {label:<22}'
                f'  {m[0]:>8.3f}'
                f'  {m[1]:>+8.3f}'
                f'  {m[2]:>8.3f}'
                f'  {m[3]:>8.4f}'
                f'  {m[4]:>8.4f}')

    cosmic_tag = f'COSMIC={os.path.basename(cosmic_path)}' if cosmic_nb is not None else 'COSMIC=off'
    lines = [
        '=' * W,
        '   FSIA-INR × GIRO 峰参数评估报告（run65）',
        '=' * W,
        f'  Checkpoint  : {_CKPT_PATH}',
        f'  {cosmic_tag}',
        f'  hmF2 有效样本: {ok_h.sum():,} / {len(hmF2_true):,}',
        f'  NmF2 有效样本: {ok_n.sum():,} / {len(NmF2_true):,}',
        f'  高度采样     : Pass1 {_ALT_COARSE_START:.0f}–{_ALT_COARSE_STOP:.0f} km '
        f'步长 {_ALT_COARSE_STEP:.0f} km ({n_c} 层)  '
        f'Pass2 ±{_ALT_FINE_HALF} km 步长 1 km ({n_f} 层)',
        '-' * W,
        f'  {"模型":<22}  {"RMSE":>8}  {"BIAS":>8}  {"MAE":>8}  {"Pearson_R":>8}  {"CCC":>8}',
        '-' * W,
        '[hmF2 (km)]',
        _row('IRI',        m_iri_h),
        _row('FSIA-INR',   m_inr_h),
        '-' * W,
        '[NmF2 (log10)]',
        _row('IRI',        m_iri_n),
        _row('FSIA-INR',   m_inr_n),
        '=' * W,
    ]
    report = '\n'.join(lines)
    print('\n' + report)

    rpt_path = os.path.join(save_dir, 'giro_peak_report.txt')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n[评估] 报告已保存: {rpt_path}')

    # ----------------------------------------------------------------
    # 2×2 密度图（hist2d + LogNorm + 色标）
    # ----------------------------------------------------------------
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)

    panels = [
        # (ax,       x_true,            y_pred,           metrics,  label,       unit)
        (axes[0, 0], hmF2_true[ok_h], hmF2_iri_h[ok_h],  m_iri_h, 'IRI',       'hmF2 (km)'),
        (axes[0, 1], hmF2_true[ok_h], hmF2_pred[ok_h],   m_inr_h, 'FSIA-INR',  'hmF2 (km)'),
        (axes[1, 0], NmF2_true[ok_n], NmF2_iri_n[ok_n],  m_iri_n, 'IRI',       'NmF2 (log10)'),
        (axes[1, 1], NmF2_true[ok_n], NmF2_pred[ok_n],   m_inr_n, 'FSIA-INR',  'NmF2 (log10)'),
    ]

    for ax, xt, xp, m, lbl, unit in panels:
        lo = float(min(xt.min(), xp.min()))
        hi = float(max(xt.max(), xp.max()))
        plot_range = [[lo, hi], [lo, hi]]

        h = ax.hist2d(xt, xp, bins=100, range=plot_range,
                      cmap='turbo', norm=LogNorm(), cmin=1)
        ax.plot([lo, hi], [lo, hi], 'w--', lw=1.5, alpha=0.9, label='1:1')
        ax.set_title(
            f'{lbl}   RMSE={m[0]:.2f}  BIAS={m[1]:+.2f}\n'
            f'R={m[3]:.4f}  CCC={m[4]:.4f}',
            fontsize=10)
        ax.set_xlabel(f'GIRO 真值 {unit}', fontsize=10)
        ax.set_ylabel(f'预测值 {unit}',    fontsize=10)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.4)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        plt.colorbar(h[3], cax=cax, label='计数 (对数)')

    plt.suptitle('FSIA-INR × GIRO 峰参数评估（Ne 廓线 argmax）',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'giro_peak_density.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[评估] 密度图已保存: {fig_path}')

    return {
        'hmF2': {'iri': m_iri_h, 'inr': m_inr_h},
        'NmF2': {'iri': m_iri_n, 'inr': m_inr_n},
    }


if __name__ == '__main__':
    evaluate_giro_peak()
