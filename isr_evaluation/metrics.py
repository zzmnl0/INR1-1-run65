"""
ISR 验证指标计算模块

计算 MDIA-INR 预测值与 ISR 观测值之间的：
  - 逐点：RMSE (log10)、MAE (log10)、Pearson R (log10)、Bias (log10)
  - NmF2：MAE (log10)、Pearson R (log10)
  - hmF2：MAE (km)、Bias (km)
"""

import numpy as np


# ==================== 基础工具 ====================

def _valid_pair(a, b):
    """返回两组数组中同时有效（有限值）的掩码。"""
    return np.isfinite(a) & np.isfinite(b)


def _ccc(obs, pred):
    """Lin's Concordance Correlation Coefficient (ρ_c).

    ρ_c = 2·Cov(x,y) / (Var(x) + Var(y) + (μ_x − μ_y)²)

    Returns NaN when either array is constant or has < 2 points.
    """
    n = len(obs)
    if n < 2:
        return np.nan
    mu_o, mu_p = obs.mean(), pred.mean()
    var_o = obs.var(ddof=0)
    var_p = pred.var(ddof=0)
    cov   = np.mean((obs - mu_o) * (pred - mu_p))
    denom = var_o + var_p + (mu_o - mu_p) ** 2
    if denom < 1e-30:
        return np.nan
    return float(2.0 * cov / denom)


# ==================== NmF2 / hmF2 ====================

def extract_isr_nmf2_hmf2(ne_2d, alt_1d, f2_alt_min=150.0):
    """
    从 ISR ne_2d (m⁻³) 中逐时刻提取 NmF2 和 hmF2。

    Args:
        ne_2d:       [n_alt, n_time] float32 — ISR 线性密度 (m⁻³)，NaN=缺测
        alt_1d:      [n_alt] float32 — 高度 (km)
        f2_alt_min:  F2 层搜索下限 (km)

    Returns:
        nmf2_log10: [n_time] float32 — log10(NmF2)
        hmf2_km:    [n_time] float32 — hmF2 (km)
    """
    n_alt, n_time = ne_2d.shape
    f2_mask = alt_1d >= f2_alt_min

    nmf2_log10 = np.full(n_time, np.nan, dtype=np.float32)
    hmf2_km    = np.full(n_time, np.nan, dtype=np.float32)

    grid_f2 = ne_2d.copy().astype(np.float64)
    grid_f2[~f2_mask, :] = np.nan
    grid_f2[grid_f2 <= 0] = np.nan

    for t in range(n_time):
        col = grid_f2[:, t]
        if not np.isfinite(col).any():
            continue
        idx = np.nanargmax(col)
        nmf2_log10[t] = np.log10(col[idx])
        hmf2_km[t]    = alt_1d[idx]

    return nmf2_log10, hmf2_km


def compute_nmf2_hmf2_metrics(isr_nmf2_log10, isr_hmf2_km,
                               model_nmf2_log10, model_hmf2_km):
    """
    计算 NmF2 和 hmF2 指标。

    Args:
        isr_nmf2_log10:   [n_time] float32
        isr_hmf2_km:      [n_time] float32
        model_nmf2_log10: [n_time] float32
        model_hmf2_km:    [n_time] float32

    Returns:
        dict:
            'nmf2_n'    : int
            'nmf2_mae'  : float  — log10 units
            'nmf2_r'    : float  — Pearson R (log10 空间)
            'nmf2_bias' : float  — log10 均值偏差
            'hmf2_n'    : int
            'hmf2_mae'  : float  — km
            'hmf2_bias' : float  — km 均值偏差
    """
    result = {}

    # NmF2
    mask_n = _valid_pair(isr_nmf2_log10, model_nmf2_log10)
    result['nmf2_n'] = int(mask_n.sum())
    if result['nmf2_n'] >= 2:
        obs_n  = isr_nmf2_log10[mask_n].astype(np.float64)
        pred_n = model_nmf2_log10[mask_n].astype(np.float64)
        result['nmf2_mae']  = float(np.mean(np.abs(pred_n - obs_n)))
        result['nmf2_bias'] = float(np.mean(pred_n - obs_n))
        result['nmf2_ccc']  = _ccc(obs_n, pred_n)
        if obs_n.std() < 1e-12 or pred_n.std() < 1e-12:
            result['nmf2_r'] = np.nan
        else:
            result['nmf2_r'] = float(np.corrcoef(obs_n, pred_n)[0, 1])
    else:
        result['nmf2_mae']  = np.nan
        result['nmf2_r']    = np.nan
        result['nmf2_ccc']  = np.nan
        result['nmf2_bias'] = np.nan

    # hmF2
    mask_h = _valid_pair(isr_hmf2_km, model_hmf2_km)
    result['hmf2_n'] = int(mask_h.sum())
    if result['hmf2_n'] >= 2:
        obs_h  = isr_hmf2_km[mask_h].astype(np.float64)
        pred_h = model_hmf2_km[mask_h].astype(np.float64)
        result['hmf2_mae']  = float(np.mean(np.abs(pred_h - obs_h)))
        result['hmf2_bias'] = float(np.mean(pred_h - obs_h))
        result['hmf2_ccc']  = _ccc(obs_h, pred_h)
    else:
        result['hmf2_mae']  = np.nan
        result['hmf2_bias'] = np.nan
        result['hmf2_ccc']  = np.nan

    return result
