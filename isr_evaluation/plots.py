"""
ISR 验证可视化模块

生成三类图表：
  1. plot_time_altitude_comparison — ISR / Raw IRI / Background / Analysis / 两类误差六列图
  2. plot_nmf2_scatter             — NmF2 散点图（Pearson CC 标注）
  3. save_metrics_report           — 指标汇总文本报告
"""

import os
import numpy as np
import matplotlib
import matplotlib.font_manager as _fm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from inr_modules.density_units import (
    DENSITY_UNIT_LABEL,
    density_to_display,
    log10_density_to_display,
)


def _setup_font():
    """
    动态检测系统已安装的 CJK 字体并设置到 matplotlib。
    优先级：Microsoft YaHei > SimHei > STXihei > FangSong > 系统回退 DejaVu Sans。
    同时关闭 unicode_minus（避免坐标轴负号显示为方框）。
    """
    candidates = ['Microsoft YaHei', 'SimHei', 'STXihei', 'FangSong',
                  'Heiti TC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    available  = {f.name for f in _fm.fontManager.ttflist}
    chosen     = [f for f in candidates if f in available]
    matplotlib.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
    matplotlib.rcParams['font.family']     = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False


_setup_font()


# ==================== 颜色范围辅助 ====================

def _safe_log10_range(ne_m3_2d, default_lo=8.0, default_hi=12.5):
    """从线性 ne 数组确定 log10 显示范围。"""
    pos = ne_m3_2d[ne_m3_2d > 0]
    if len(pos) == 0:
        return default_lo, default_hi
    lo = max(np.log10(np.nanpercentile(pos, 2)),  default_lo)
    hi = min(np.log10(np.nanpercentile(pos, 98)), default_hi)
    if hi - lo < 0.3:
        mid = (hi + lo) / 2.0
        lo, hi = mid - 0.15, mid + 0.15
    return lo, hi


# ==================== 时间-高度对比图 ====================

def plot_time_altitude_comparison(day_record, ne_iri_log10,
                                  background_log10_2d, model_log10_2d,
                                  save_path):
    """
    绘制六列时间-高度对比图：
        Col 1: ISR 观测（物理电子密度）
        Col 2: Raw IRI
        Col 3: FNDA Background (M00)
        Col 4: FSIA-INR Analysis (M11)
        Col 5: Analysis - ISR 误差（物理电子密度, bwr）
        Col 6: Raw IRI - ISR 误差（物理电子密度, bwr）

    Args:
        day_record:      DayRecord dict，含 'ne_2d' (m⁻³), 'alt_1d', 'ts_1d',
                         'station', 'date_str', 'plot_segs'
        ne_iri_log10:    [n_alt, n_time] float32 — Raw IRI log10(Ne)
        background_log10_2d: [n_alt, n_time] float32 — FNDA Background
        model_log10_2d:  [n_alt, n_time] float32 — FSIA-INR M11
        save_path:       输出 PNG 完整路径
    """
    import pandas as pd
    import matplotlib.dates as mdates

    ne_2d     = day_record['ne_2d']
    alt_1d    = day_record['alt_1d']
    ts_1d     = day_record['ts_1d']
    station   = day_record.get('station', 'ISR')
    date_str  = day_record.get('date_str', '')
    plot_segs = day_record.get('plot_segs', [])

    # ---- 物理电子密度全网格（单位 10^11 m^-3）----
    isr_display_full = np.where(
        ne_2d > 0, density_to_display(ne_2d), np.nan)

    # Ne 色标范围
    lo, hi = _safe_log10_range(ne_2d)
    for arr in [ne_iri_log10, background_log10_2d, model_log10_2d]:
        fin = arr[np.isfinite(arr)]
        if len(fin):
            lo = min(lo, float(np.percentile(fin, 2)))
            hi = max(hi, float(np.percentile(fin, 98)))
    lo = max(lo, 8.0);  hi = min(hi, 13.0)
    if hi - lo < 0.3:
        mid = (lo + hi) / 2;  lo, hi = mid - 0.15, mid + 0.15
    ne_norm = LogNorm(vmin=10.0 ** (lo - 11.0),
                      vmax=10.0 ** (hi - 11.0))

    iri_display_full = log10_density_to_display(ne_iri_log10)
    bkg_display_full = log10_density_to_display(background_log10_2d)
    model_display_full = log10_density_to_display(model_log10_2d)

    # 误差色标范围：取物理密度差的共同 95th 分位。
    mdia_err_full = model_display_full - isr_display_full
    iri_err_full = iri_display_full - isr_display_full
    all_err       = np.concatenate([mdia_err_full[np.isfinite(mdia_err_full)],
                                    iri_err_full [np.isfinite(iri_err_full )]])
    err_max = max(float(np.percentile(np.abs(all_err), 95))
                  if len(all_err) else 1.0, 0.05)

    # 时间戳 → 列索引映射
    ts_dict = {int(round(float(t))): i for i, t in enumerate(ts_1d)}

    fig, axes = plt.subplots(1, 6, figsize=(33, 6), sharey=True)
    alt_lo, alt_hi = float(alt_1d.min()), float(alt_1d.max())

    ne_mesh_ref   = None   # Ne 色标参考（Col 0–3 共用）
    err_mesh_ref  = None   # 误差色标参考（Col 4–5 共用）

    # ---- 六列按 segment 分段渲染，无拉伸、无填充 ----
    for seg in plot_segs:
        seg_ts  = seg['ts']       # [n_t] Unix s
        seg_alt = seg['alt_km']   # [n_a] km
        seg_ne  = seg['ne']       # [n_a, n_t] m⁻³

        ts_idx  = np.array([ts_dict.get(int(round(float(t))), -1) for t in seg_ts])
        valid_t = ts_idx >= 0
        if not valid_t.any():
            continue
        ts_idx_v  = ts_idx[valid_t]
        seg_ne_v  = seg_ne[:, valid_t]
        ts_dt_seg = pd.to_datetime(seg_ts[valid_t], unit='s').values

        alt_idx = np.array([int(np.argmin(np.abs(alt_1d - a))) for a in seg_alt])

        isr_display = np.where(
            seg_ne_v > 0, density_to_display(seg_ne_v), np.nan)
        ne_iri_sub = log10_density_to_display(
            ne_iri_log10[np.ix_(alt_idx, ts_idx_v)])
        ne_bkg_sub = log10_density_to_display(
            background_log10_2d[np.ix_(alt_idx, ts_idx_v)])
        ne_pred_sub = log10_density_to_display(
            model_log10_2d[np.ix_(alt_idx, ts_idx_v)])
        mdia_err = ne_pred_sub - isr_display
        iri_err = ne_iri_sub - isr_display

        def _pm(ax, data, cmap, **norm_kwargs):
            return ax.pcolormesh(ts_dt_seg, seg_alt, data,
                                 cmap=cmap, shading='auto', **norm_kwargs)

        m = _pm(axes[0], isr_display, 'plasma', norm=ne_norm)
        if ne_mesh_ref is None:  ne_mesh_ref  = m
        _pm(axes[1], ne_iri_sub, 'plasma', norm=ne_norm)
        _pm(axes[2], ne_bkg_sub, 'plasma', norm=ne_norm)
        _pm(axes[3], ne_pred_sub, 'plasma', norm=ne_norm)
        m = _pm(axes[4], mdia_err, 'bwr', vmin=-err_max, vmax=err_max)
        if err_mesh_ref is None: err_mesh_ref = m
        _pm(axes[5], iri_err, 'bwr', vmin=-err_max, vmax=err_max)

    # 回退：无 plot_segs 时用合并网格
    if ne_mesh_ref is None:
        t_dt = pd.to_datetime(ts_1d, unit='s').values
        ne_mesh_ref = axes[0].pcolormesh(
            t_dt, alt_1d, isr_display_full, cmap='plasma',
            shading='auto', norm=ne_norm)
        axes[1].pcolormesh(t_dt, alt_1d, iri_display_full, cmap='plasma',
                           shading='auto', norm=ne_norm)
        axes[2].pcolormesh(t_dt, alt_1d, bkg_display_full, cmap='plasma',
                           shading='auto', norm=ne_norm)
        axes[3].pcolormesh(t_dt, alt_1d, model_display_full, cmap='plasma',
                           shading='auto', norm=ne_norm)
        err_mesh_ref = axes[4].pcolormesh(t_dt, alt_1d, mdia_err_full, cmap='bwr',
                                          shading='auto', vmin=-err_max, vmax=err_max)
        axes[5].pcolormesh(t_dt, alt_1d, iri_err_full,  cmap='bwr',
                           shading='auto', vmin=-err_max, vmax=err_max)

    # Colorbars — 标签用 mathtext 避免 Unicode 渲染问题
    _ne_label = f'Ne ({DENSITY_UNIT_LABEL})'
    _err_label = f'$\\Delta$Ne ({DENSITY_UNIT_LABEL})'
    for ax, ref, label in [
        (axes[0], ne_mesh_ref,  _ne_label),
        (axes[1], ne_mesh_ref,  _ne_label),
        (axes[2], ne_mesh_ref,  _ne_label),
        (axes[3], ne_mesh_ref,  _ne_label),
        (axes[4], err_mesh_ref, _err_label),
        (axes[5], err_mesh_ref, _err_label),
    ]:
        if ref is not None:
            plt.colorbar(ref, ax=ax, label=label, pad=0.02, shrink=0.85)

    # 轴修饰
    titles = [
        f'{station} ISR',
        'Raw IRI',
        'FNDA Background',
        'FSIA-INR M11',
        'M11 - ISR Error',
        'IRI - ISR Error',
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('UT (h)', fontsize=9)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.set_ylim(alt_lo, alt_hi)
    axes[0].set_ylabel('Altitude (km)', fontsize=10)

    plt.suptitle(f'{station}  {date_str}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [plots] 时间-高度图已保存: {save_path}')


# ==================== NmF2 散点图 ====================

def plot_nmf2_scatter(isr_nmf2_log10_all, model_nmf2_log10_all,
                      station, save_path):
    """
    绘制 NmF2 散点图，标注 Pearson CC 和样本数。

    Args:
        isr_nmf2_log10_all:   [N] float32 — ISR log10(NmF2)（所有天拼接）
        model_nmf2_log10_all: [N] float32 — 模型 log10(NmF2)
        station:              站点名称（用于标题）
        save_path:            输出 PNG 完整路径
    """
    mask = np.isfinite(isr_nmf2_log10_all) & np.isfinite(model_nmf2_log10_all)
    obs_log10 = isr_nmf2_log10_all[mask].astype(np.float64)
    pred_log10 = model_nmf2_log10_all[mask].astype(np.float64)
    obs = log10_density_to_display(obs_log10)
    pred = log10_density_to_display(pred_log10)
    n    = len(obs)

    fig, ax = plt.subplots(figsize=(6, 6))

    if n >= 2:
        r = float(np.corrcoef(obs_log10, pred_log10)[0, 1])
        ax.scatter(obs, pred, s=12, alpha=0.5, color='steelblue', linewidths=0)
        lower = min(obs.min(), pred.min())
        upper = max(obs.max(), pred.max())
        pad = max(0.05 * (upper - lower), 0.01 * upper, 1e-3)
        lo = max(0.0, lower - pad)
        hi = upper + pad
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(0.05, 0.92, f'CC (log10) = {r:.3f}\nN = {n}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', fontsize=12)

    ax.set_xlabel(f'ISR NmF2 ({DENSITY_UNIT_LABEL})', fontsize=11)
    ax.set_ylabel(f'Prediction NmF2 ({DENSITY_UNIT_LABEL})', fontsize=11)
    ax.set_title(f'{station}  NmF2 Scatter', fontsize=12, fontweight='bold')
    ax.set_aspect('equal', 'box')
    if n >= 2:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [plots] NmF2 散点图已保存: {save_path}')


# ==================== hmF2 / NmF2 vs 地方时对比图（连续时间轴）====================

def plot_peak_lt_comparison(
    lt_all,
    isr_hmf2, model_hmf2, background_hmf2, iri_hmf2,
    isr_nmf2, model_nmf2, background_nmf2, iri_nmf2,
    station, model_name, save_path,
):
    """
    绘制 hmF2 和 NmF2 随地方时变化的拟合对比图（连续时间轴，含日期区分）。

    上图（hmF2）: 横轴 LT 连续日期时间，纵轴高度 (km)
    下图（NmF2）: 横轴 LT 连续日期时间，纵轴物理峰值密度

    四条曲线：ISR 观测、Raw IRI、FNDA Background、M11 Analysis。
    排序后按时间先后绘制，x 轴刻度同时显示日期和 LT 小时，让日内日间循环
    与日间变化在同一连续时间轴上清晰呈现。

    Args:
        lt_all      : [N] float — LT Unix 时间戳 (s)，= UT_unix + lon/15*3600
        isr_hmf2    : [N] float — ISR hmF2 (km)，NaN=无效
        model_hmf2  : [N] float — 模型 hmF2 (km)
        iri_hmf2    : [N] float — IRI hmF2 (km)
        isr_nmf2    : [N] float — ISR log10(NmF2)
        model_nmf2  : [N] float — 模型 log10(NmF2)
        iri_nmf2    : [N] float — IRI log10(NmF2)
        station     : str
        model_name  : str
        save_path   : str
    """
    import pandas as pd
    import matplotlib.dates as mdates

    def _segmented_values(times, values):
        """Insert NaNs at local-date and long-cadence discontinuities."""
        values = np.asarray(values, dtype=np.float64).copy()
        finite_time = np.isfinite(times)
        valid_times = times[finite_time]
        cadence = np.diff(valid_times)
        cadence = cadence[cadence > 0]
        median_cadence = float(np.median(cadence)) if cadence.size else 0.0
        break_after = max(3.0 * median_cadence, 900.0)
        for index in range(1, len(times)):
            if not (np.isfinite(times[index]) and np.isfinite(times[index - 1])):
                values[index] = np.nan
                continue
            date_changed = int(np.floor(times[index] / 86400.0)) != int(
                np.floor(times[index - 1] / 86400.0))
            if date_changed or times[index] - times[index - 1] > break_after:
                values[index] = np.nan
        return values

    # LT Unix 时间戳 → pandas Timestamp（用于 matplotlib datetime 轴）
    lt_unix = np.asarray(lt_all,     dtype=np.float64)
    ih      = np.asarray(isr_hmf2,   dtype=np.float64)
    mh      = np.asarray(model_hmf2, dtype=np.float64)
    bh      = np.asarray(background_hmf2, dtype=np.float64)
    rh      = np.asarray(iri_hmf2,   dtype=np.float64)
    in_     = np.asarray(isr_nmf2,   dtype=np.float64)
    mn      = np.asarray(model_nmf2, dtype=np.float64)
    bn      = np.asarray(background_nmf2, dtype=np.float64)
    rn      = np.asarray(iri_nmf2,   dtype=np.float64)

    # 按时间排序（各天数据可能乱序拼接）
    sort_idx = np.argsort(lt_unix)
    lt_unix  = lt_unix[sort_idx]
    ih, mh, bh, rh = ih[sort_idx], mh[sort_idx], bh[sort_idx], rh[sort_idx]
    in_, mn, bn, rn = in_[sort_idx], mn[sort_idx], bn[sort_idx], rn[sort_idx]

    lt_dt = pd.to_datetime(lt_unix, unit='s')   # LT datetime array

    # Each field retains its own finite mask.  A missing background/model value
    # must not erase an otherwise valid ISR or IRI curve.
    h_values = [_segmented_values(lt_unix, values)
                for values in (ih, rh, bh, mh)]
    in_plot = log10_density_to_display(in_)
    mn_plot = log10_density_to_display(mn)
    bn_plot = log10_density_to_display(bn)
    rn_plot = log10_density_to_display(rn)
    n_values = [_segmented_values(lt_unix, values)
                for values in (in_plot, rn_plot, bn_plot, mn_plot)]

    fig, (ax_h, ax_n) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # ---- hmF2 ----
    ax_h.scatter(lt_dt[np.isfinite(ih)], ih[np.isfinite(ih)],
                 s=10, alpha=0.5, color='#1f77b4', linewidths=0,
                 label='ISR', zorder=3)
    ax_h.plot(lt_dt, h_values[1],
              color='black', lw=1.2, ls='--', label='Raw IRI', zorder=4)
    ax_h.plot(lt_dt, h_values[2],
              color='gray', lw=1.2, ls=':', label='FNDA Background', zorder=4)
    ax_h.plot(lt_dt, h_values[3],
              color='#2ca02c', lw=1.5, ls='-', label=model_name, zorder=5)

    # ---- NmF2 ----
    ax_n.scatter(lt_dt[np.isfinite(in_plot)], in_plot[np.isfinite(in_plot)],
                 s=10, alpha=0.5, color='#1f77b4', linewidths=0,
                 label='ISR', zorder=3)
    ax_n.plot(lt_dt, n_values[1],
              color='black', lw=1.2, ls='--', label='Raw IRI', zorder=4)
    ax_n.plot(lt_dt, n_values[2],
              color='gray', lw=1.2, ls=':', label='FNDA Background', zorder=4)
    ax_n.plot(lt_dt, n_values[3],
              color='#2ca02c', lw=1.5, ls='-', label=model_name, zorder=5)

    # ---- 轴格式：显示日期 + LT 小时 ----
    # 主刻度：每天，副刻度：每6h
    ax_n.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_n.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax_n.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax_n.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))

    # 次刻度标签小一点，错开显示
    for label in ax_n.xaxis.get_minorticklabels():
        label.set_fontsize(7)
        label.set_color('#555555')
    for label in ax_n.xaxis.get_majorticklabels():
        label.set_rotation(30)
        label.set_ha('right')

    ax_n.tick_params(axis='x', which='minor', length=4)
    ax_n.tick_params(axis='x', which='major', length=7)

    for ax in (ax_h, ax_n):
        ax.grid(True, which='major', alpha=0.35, lw=0.8)
        ax.grid(True, which='minor', alpha=0.15, lw=0.5)
        ax.legend(fontsize=9, loc='best')

    ax_h.set_ylabel('hmF2 (km)', fontsize=11)
    ax_n.set_ylabel(f'NmF2 ({DENSITY_UNIT_LABEL})', fontsize=11)
    ax_n.set_xlabel('Local Time', fontsize=11)

    ax_h.set_title(f'{station}  hmF2 vs Local Time', fontsize=11, fontweight='bold')
    ax_n.set_title(f'{station}  NmF2 vs Local Time', fontsize=11, fontweight='bold')

    n_pts_h = int(np.isfinite(ih).sum())
    n_pts_n = int(np.isfinite(in_plot).sum())
    fig.suptitle(
        f'{station}  |  hmF2: N={n_pts_h}  NmF2: N={n_pts_n}  (横轴为地方时)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [plots] hmF2/NmF2 vs LT 对比图已保存: {save_path}')


# ==================== 文本指标报告 ====================

def _fmt(v, fmt='.4f'):
    return f'{v:{fmt}}' if np.isfinite(v) else 'N/A'


def save_metrics_report(station_reports, save_path):
    """
    将多站点指标汇总写入文本报告（Raw IRI / Background / Analysis）。

    每个 station_report dict 需含：
        Analysis 指标使用无前缀键；Background 使用 ``background_`` 前缀；
        Raw IRI 使用 ``iri_`` 前缀。
    """
    lines = []
    lines.append('=' * 94)
    lines.append('FSIA-INR × ISR 验证报告（Raw IRI / FNDA Background / M11 Analysis）')
    lines.append('=' * 94)

    for rep in station_reports:
        st = rep['station']
        lines.append(f'\n站点: {st}   有效天数: {rep["n_days"]}')
        model_name = rep.get('model_name', 'FSIA-INR M11')
        lines.append('-' * 82)
        lines.append(
            f'  {"指标":<28} {"Raw IRI":>14}  {"Background":>14}  {model_name:>14}')
        lines.append(f'  {"-"*28} {"-"*14}  {"-"*14}  {"-"*14}')

        def row(label, iri_val, background_val, analysis_val, fmt='.4f'):
            iv = _fmt(iri_val, fmt)
            bv = _fmt(background_val, fmt)
            av = _fmt(analysis_val, fmt)
            lines.append(f'  {label:<28} {iv:>14}  {bv:>14}  {av:>14}')

        lines.append(
            f'\n  【逐点统计】  IRI N={rep.get("iri_point_n", 0)}  '
            f'Background N={rep.get("background_point_n", 0)}  '
            f'Analysis N={rep["point_n"]}')
        row('RMSE (log10 Ne)', rep.get('iri_point_rmse', np.nan),
            rep.get('background_point_rmse', np.nan), rep['point_rmse'])
        row('MAE (log10 Ne)', rep.get('iri_point_mae', np.nan),
            rep.get('background_point_mae', np.nan), rep['point_mae'])
        row('Pearson R (log10)', rep.get('iri_point_r', np.nan),
            rep.get('background_point_r', np.nan), rep['point_r'])
        row('CCC (log10)', rep.get('iri_point_ccc', np.nan),
            rep.get('background_point_ccc', np.nan), rep.get('point_ccc', np.nan))
        row('Bias (log10 Ne)', rep.get('iri_point_bias', np.nan),
            rep.get('background_point_bias', np.nan), rep['point_bias'])

        lines.append(
            f'\n  【NmF2 统计】  IRI N={rep.get("iri_nmf2_n", 0)}  '
            f'Background N={rep.get("background_nmf2_n", 0)}  '
            f'Analysis N={rep["nmf2_n"]}')
        row('MAE (log10 NmF2)', rep.get('iri_nmf2_mae', np.nan),
            rep.get('background_nmf2_mae', np.nan), rep['nmf2_mae'])
        row('Pearson R (log10 NmF2)', rep.get('iri_nmf2_r', np.nan),
            rep.get('background_nmf2_r', np.nan), rep['nmf2_r'])
        row('CCC (log10 NmF2)', rep.get('iri_nmf2_ccc', np.nan),
            rep.get('background_nmf2_ccc', np.nan), rep.get('nmf2_ccc', np.nan))
        row('Bias (log10 NmF2)', rep.get('iri_nmf2_bias', np.nan),
            rep.get('background_nmf2_bias', np.nan), rep['nmf2_bias'])

        lines.append(
            f'\n  【hmF2 统计】  IRI N={rep.get("iri_hmf2_n", 0)}  '
            f'Background N={rep.get("background_hmf2_n", 0)}  '
            f'Analysis N={rep["hmf2_n"]}')
        row('MAE (km)', rep.get('iri_hmf2_mae', np.nan),
            rep.get('background_hmf2_mae', np.nan), rep['hmf2_mae'], '.2f')
        row('CCC (km space)', rep.get('iri_hmf2_ccc', np.nan),
            rep.get('background_hmf2_ccc', np.nan), rep.get('hmf2_ccc', np.nan))
        row('Bias (km)', rep.get('iri_hmf2_bias', np.nan),
            rep.get('background_hmf2_bias', np.nan), rep['hmf2_bias'], '+.2f')

    lines.append('\n' + '=' * 94)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [plots] 指标报告已保存: {save_path}')
