"""
FSIA-INR 可视化模块

功能：
  1. plot_global_slice      — Raw IRI / Background / M10 / M01 / M11 / increment
  2. plot_altitude_profile  — 指定位置的五模式 EDP 廓线
  3. plot_hmf2_nmf2_map     — 从最终 M11 廓线提取 hmF2 / NmF2
"""

import os
import datetime
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from .sliding_dataset import (
    attach_observation_background,
    observation_query_coverage,
    query_observation_payload,
)
import matplotlib.colors as mcolors

# 中文字体配置（Windows：微软雅黑/黑体；Linux/Mac：回退 DejaVu Sans）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STXihei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 地理坐标降级模式：不再使用 aacgmv2；coords 保持 [N, 4]（Lat_geo, Lon_geo, Alt, Time）

# 2024-09-01 00:00:00 UTC — 训练数据起始时刻的 Unix 时间戳（供 ISR 时刻对齐使用）
_SEP01_2024_UNIX = 1725148800.0


def _rel_to_datestr(rel_hour):
    """将 rel_hour（从 2024-09-01 00:00 UTC 起算的小时数）转换为 'YYYY-MM-DD' 字符串。"""
    base = datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc)
    dt   = base + datetime.timedelta(hours=float(rel_hour))
    return dt.strftime('%Y-%m-%d')


def _extract_isr_profile(isr_record, rel_hour, tol_sec=1800):
    """
    从 load_jicamarca 返回的 DayRecord 中提取最接近 rel_hour 的 Ne 廓线。

    Args:
        isr_record: DayRecord dict（含 ts_1d / ne_2d / alt_1d）
        rel_hour:   相对小时数（从 2024-09-01 00:00 UTC 起算）
        tol_sec:    最大时间容差（秒），超出则视为无近邻数据（默认 1800 = 30 min）

    Returns:
        (ne_log10 [M], alt_km [M])  — 有效观测点的 log₁₀(Ne) 与对应高度
        或 (None, None) 若无近邻数据或全部无效
    """
    target_unix = _SEP01_2024_UNIX + rel_hour * 3600.0
    ts     = isr_record['ts_1d']
    dt_arr = np.abs(ts - target_unix)
    t_idx  = int(np.argmin(dt_arr))
    if dt_arr[t_idx] > tol_sec:
        return None, None

    ne_col  = isr_record['ne_2d'][:, t_idx]   # m⁻³（无效处已为 NaN）
    alt_col = isr_record['alt_1d']             # km
    valid   = (ne_col > 0) & np.isfinite(ne_col)
    if not valid.any():
        return None, None
    return np.log10(ne_col[valid].astype(np.float64)), alt_col[valid]


# ======================== 内部辅助 ========================

def _infer_grid(model, coords_np, sw_seq_single, device, sw_manager,
                vis_batch=1024,
                iri_peak_manager=None, fy_nb_index=None,
                cosmic_nb_index=None, include_single_source=True):
    """
    在大网格上分批推理，返回各分量 numpy 数组。

    Args:
        model:          MDIA_INR_Model（eval 模式）
        coords_np:      [N, 4] float32 numpy — (Lat_geo, Lon_geo, Alt, Time)
        sw_seq_single:  [1, seq_len, 2] tensor（单一时刻 SW 序列）
        device:         torch.device
        vis_batch:      每批点数，默认 1024

    Returns:
        dict:
            'ne_iri'     [N] — Raw IRI
            'background' [N] — FNDA Background (M00)
            'm00/m10/m01/m11' [N] — 四模式输出
            'ne_delta'   [N] — M11 − Background（log10 空间）
    """
    model.eval()
    N = len(coords_np)
    buffers = {
        key: np.empty(N, dtype=np.float32)
        for key in ('ne_iri', 'background', 'm00', 'm10', 'm01', 'm11')
    }
    fy_coverage = np.zeros(N, dtype=np.float32)
    cosmic_coverage = np.zeros(N, dtype=np.float32)
    hmf2_buf      = np.full(N, 300.0, dtype=np.float32)   # 默认 300 km
    nmf2_buf      = np.full(N, 11.5,  dtype=np.float32)   # 默认 11.5 log10

    # 地理坐标降级：coords_np 保持 [N, 4]，不再追加 AACGM 列

    with torch.no_grad():
        for start in range(0, N, vis_batch):
            end = min(start + vis_batch, N)
            chunk = torch.from_numpy(coords_np[start:end]).to(device)
            n = end - start
            sw_chunk = sw_seq_single.expand(n, -1, -1)

            iri_peak = None
            if iri_peak_manager is not None:
                iri_peak = iri_peak_manager.get_iri_peak(chunk)
            common = {'iri_peak': iri_peak}
            m00, _, _, _, extras = model(chunk, sw_chunk, **common)
            fy_kwargs = {}
            cosmic_kwargs = {}
            if fy_nb_index is not None:
                observations = query_observation_payload(
                    fy_nb_index, chunk, device)
                fy_coverage[start:end] = observation_query_coverage(
                    observations).float().cpu().numpy()
                fy_kwargs = {'observations_fy': attach_observation_background(
                    observations, model, sw_manager, iri_peak_manager)}
            if cosmic_nb_index is not None:
                observations = query_observation_payload(
                    cosmic_nb_index, chunk, device)
                cosmic_coverage[start:end] = observation_query_coverage(
                    observations).float().cpu().numpy()
                cosmic_kwargs = {
                    'observations_cosmic': attach_observation_background(
                        observations, model, sw_manager, iri_peak_manager)}
            if include_single_source:
                m10 = model(chunk, sw_chunk, **common, **fy_kwargs)[0]
                m01 = model(chunk, sw_chunk, **common, **cosmic_kwargs)[0]
            else:
                m10 = m01 = m00
            m11 = model(
                chunk, sw_chunk, **common, **fy_kwargs, **cosmic_kwargs)[0]

            buffers['ne_iri'][start:end] = extras.get(
                'ne_iri', extras['ne_bkg']).reshape(-1).cpu().numpy()
            buffers['background'][start:end] = extras['ne_bkg'].reshape(-1).cpu().numpy()
            for key, value in (
                    ('m00', m00), ('m10', m10), ('m01', m01), ('m11', m11)):
                buffers[key][start:end] = value.reshape(-1).cpu().numpy()
            _cp   = extras.get('peak_params', {})
            _hmf2 = _cp.get('hmF2')
            _nmf2 = _cp.get('NmF2')
            if _hmf2 is not None:
                hmf2_buf[start:end] = _hmf2.cpu().numpy()
            if _nmf2 is not None:
                nmf2_buf[start:end] = _nmf2.cpu().numpy()

    return {
        **buffers,
        'ne_fused':   buffers['m11'],
        'ne_bkg':     buffers['background'],
        'ne_delta':   buffers['m11'] - buffers['background'],
        'has_fy':     fy_coverage,
        'has_cosmic': cosmic_coverage,
        'hmf2_f2':    hmf2_buf,
        'nmf2_f2':    nmf2_buf,   # log10 单位；plot 时用 10** 转换为 el/m³
    }


def _infer_analysis_peaks(model, lat_flat, lon_flat, global_time,
                          sw_seq_single, device, sw_manager, iri_peak_manager,
                          fy_nb_index, cosmic_nb_index):
    """Derive hmF2/NmF2 from the final M11 density columns."""
    alts = np.arange(model.alt_min, model.alt_max + 0.1, 10.0, dtype=np.float32)
    n_points = len(lat_flat)
    coords = np.column_stack([
        np.tile(lat_flat, len(alts)),
        np.tile(lon_flat, len(alts)),
        np.repeat(alts, n_points),
        np.full(len(alts) * n_points, global_time, dtype=np.float32),
    ]).astype(np.float32)
    result = _infer_grid(
        model, coords, sw_seq_single, device, sw_manager, vis_batch=4096,
        iri_peak_manager=iri_peak_manager,
        fy_nb_index=fy_nb_index, cosmic_nb_index=cosmic_nb_index,
        include_single_source=False)
    columns = result['m11'].reshape(len(alts), n_points)
    peak_index = np.argmax(columns, axis=0)
    return alts[peak_index], columns[peak_index, np.arange(n_points)]


def _get_sw_seq(sw_manager, global_time, device):
    """获取单一时刻的 SW 序列 [1, seq_len, 2]"""
    time_t = torch.tensor([global_time], dtype=torch.float32, device=device)
    return sw_manager.get_drivers_sequence(time_t)


def _sw_display_values(sw_seq_single):
    """从 SW 序列末端提取 Kp / F10.7 展示值"""
    kp_raw   = sw_seq_single[0, -1, 0].item()
    f107_raw = sw_seq_single[0, -1, 1].item()
    kp_disp   = (kp_raw + 1.0) / 2.0 * 9.0
    f107_disp = f107_raw * 60.0 + 210.0
    return kp_disp, f107_disp


def _discrete_norm(cmap_name, vmin, vmax, n=12):
    """
    创建线性等间距离散 BoundaryNorm + colormap。

    Args:
        cmap_name: matplotlib colormap 名称字符串
        vmin/vmax: 数据范围
        n:         离散级数（默认 12）

    Returns:
        (cmap, norm)
    """
    bounds = np.linspace(vmin, vmax, n + 1)
    try:
        cmap = matplotlib.colormaps[cmap_name].resampled(n)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap(cmap_name, n)  # noqa: deprecated but safe fallback
    norm = mcolors.BoundaryNorm(bounds, ncolors=n)
    return cmap, norm


def _discrete_norm_log(cmap_name, vmin_log10, vmax_log10, n=10):
    """
    创建对数间距离散 BoundaryNorm + colormap（用于跨数量级数据，如 NmF2 el/m³）。

    Args:
        cmap_name:          matplotlib colormap 名称字符串
        vmin_log10/vmax_log10: log10 空间的上下限（如 10.5, 13.0）
        n:                  离散级数（默认 10）

    Returns:
        (cmap, norm, bounds)  — bounds 为 el/m³ 实际值数组，供 colorbar 标注
    """
    bounds = np.logspace(vmin_log10, vmax_log10, n + 1)
    try:
        cmap = matplotlib.colormaps[cmap_name].resampled(n)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap(cmap_name, n)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n)
    return cmap, norm, bounds


def _discrete_norm_sym_white(cmap_name, vmax, n=20, n_white=4):
    """
    创建中心带纯白色的对称离散 colormap + norm（用于残差 ΔNe 面板）。

    中心 n_white 级替换为纯白，使零值附近形成明显白色中心带，
    正负异号区域分别映射到冷暖两端，与背景白色形成强对比。

    Args:
        cmap_name: 基础散度 colormap 名称（如 'RdBu_r'）
        vmax:      对称范围正半轴最大值（el/m³）
        n:         离散总级数（默认 20）
        n_white:   中心替换为纯白的级数（默认 4，即中心 ±10% 范围）

    Returns:
        (cmap, norm)
    """
    bounds = np.linspace(-vmax, vmax, n + 1)
    try:
        base_colors = matplotlib.colormaps[cmap_name](np.linspace(0, 1, n))
    except (AttributeError, KeyError):
        base_colors = plt.cm.get_cmap(cmap_name, n)(np.linspace(0, 1, n))
    mid = n // 2
    hw  = n_white // 2
    base_colors[mid - hw: mid + hw] = [1.0, 1.0, 1.0, 1.0]
    cmap = mcolors.ListedColormap(base_colors)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n)
    return cmap, norm


def _fmt_sym_ticks(cb, vmax_lin):
    """
    为对称线性离散 colorbar（如残差 ΔNe）设置科学记数法刻度标签。

    刻度均匀分布于 [-vmax_lin, +vmax_lin]，标签归一化到最近的 10^n 量级。

    Args:
        cb:       matplotlib Colorbar 对象
        vmax_lin: 正半轴最大值（el/m³）

    Returns:
        exp (int) — 所用量级幂次，供调用方更新 colorbar label
    """
    if vmax_lin <= 0:
        return 0
    exp   = int(np.floor(np.log10(vmax_lin + 1e-30)))
    scale = 10.0 ** exp
    tick_vals = np.linspace(-vmax_lin, vmax_lin, 7)
    cb.set_ticks(tick_vals)
    cb.set_ticklabels([f'{v / scale:.1f}' for v in tick_vals], fontsize=7)
    return exp


def _fmt_log_ticks(cb, bounds):
    """
    为对数离散 colorbar 设置科学记数法刻度标签（每隔一个 bound 显示一个刻度）。

    Args:
        cb:     matplotlib Colorbar 对象
        bounds: _discrete_norm_log 返回的 bounds 数组（el/m³ 实际值）
    """
    tick_vals = bounds[::2]   # 每隔一个，避免标签拥挤
    cb.set_ticks(tick_vals)
    labels = []
    for v in tick_vals:
        exp  = int(np.floor(np.log10(v + 1e-30)))
        coef = v / 10.0 ** exp
        if abs(coef - 1.0) < 0.05:
            labels.append(f'$10^{{{exp}}}$')
        else:
            labels.append(f'${coef:.1f}\\!\\times\\!10^{{{exp}}}$')
    cb.set_ticklabels(labels, fontsize=7)


# ======================== 全球纬经度切片 ========================

def plot_global_slice(model, sw_manager, device, target_day, target_hour,
                      save_dir, alt_levels=None, model_name='MDIA-INR',
                      iri_peak_manager=None, fy_nb_index=None,
                      cosmic_nb_index=None):
    """
    绘制全球纬经度切片图（多高度层）。

    每个高度层一行，六列布局：
        Raw IRI | Background(M00) | FY(M10) | COSMIC(M01) |
        FY+COSMIC(M11) | M11-Background

    Args:
        model:        MDIA_INR_Model 或 FSIA_INR_Model
        sw_manager:   SpaceWeatherManager
        device:       torch.device
        target_day:   天数（相对于 start_date，0 索引，0 = 第 1 天）
        target_hour:  整点小时（0–23）
        save_dir:     保存目录
        alt_levels:   高度列表 (km)，默认 [200, 300, 400]
        model_name:   模型名称，用于标题和标签（默认 'MDIA-INR'）
    """
    os.makedirs(save_dir, exist_ok=True)
    if alt_levels is None:
        alt_levels = [250, 300, 350, 400, 450]

    global_time = target_day * 24.0 + target_hour
    print(f'[可视化] 全球切片  Day {target_day}  {target_hour:02d}:00 UT  '
          f'高度: {alt_levels} km')

    sw_seq_single = _get_sw_seq(sw_manager, global_time, device)
    kp_disp, f107_disp = _sw_display_values(sw_seq_single)

    # 91 × 180 空间网格
    lat_grid = np.linspace(-90, 90, 91)
    lon_grid = np.linspace(-180, 180, 180)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    extent = [-180, 180, -90, 90]
    n_pts = LAT.size

    n_rows = len(alt_levels)
    fig, axes = plt.subplots(n_rows, 6, figsize=(30, 4.5 * n_rows))
    axes = np.atleast_2d(axes)  # 保证二维索引 [row, col]

    col_titles = [
        'Raw IRI', 'FNDA Background = M00', 'M10 FY only',
        'M01 COSMIC only', 'M11 FY + COSMIC', 'M11 − Background',
    ]

    # ---- 固定色标范围（统一所有高度层）----
    # Ne 面板：7×10⁹ ~ 4×10¹² el/m³（log10: 9.845 ~ 12.602）
    _ne_lo = np.log10(7e9)
    _ne_hi = np.log10(4e12)
    ne_cmap, ne_norm, ne_bounds = _discrete_norm_log('jet', _ne_lo, _ne_hi, n=20)

    # Analysis increment in log10 space.
    _dl_max = 0.5
    dl_cmap, dl_norm = _discrete_norm_sym_white('RdBu_r', _dl_max, n=19, n_white=1)

    # ---- 单遍推理 + 绘图 ----
    for row_idx, alt in enumerate(alt_levels):
        coords_np = np.column_stack([
            LAT.flatten().astype(np.float32),
            LON.flatten().astype(np.float32),
            np.full(n_pts, alt,         dtype=np.float32),
            np.full(n_pts, global_time, dtype=np.float32),
        ])

        result    = _infer_grid(model, coords_np, sw_seq_single, device,
                                sw_manager,
                                iri_peak_manager=iri_peak_manager,
                                fy_nb_index=fy_nb_index,
                                cosmic_nb_index=cosmic_nb_index)
        iri_map   = result['ne_iri'].reshape(LAT.shape)
        bkg_map   = result['background'].reshape(LAT.shape)
        m10_map   = result['m10'].reshape(LAT.shape)
        m01_map   = result['m01'].reshape(LAT.shape)
        fuse_map  = result['m11'].reshape(LAT.shape)
        delta_map = result['ne_delta'].reshape(LAT.shape)   # log10 残差
        iri_lin   = 10.0 ** iri_map
        bkg_lin   = 10.0 ** bkg_map
        m10_lin   = 10.0 ** m10_map
        m01_lin   = 10.0 ** m01_map
        fuse_lin  = 10.0 ** fuse_map

        # (data, cmap_d, norm_d, bounds_or_None, col_title, cb_mode)
        # cb_mode: 'log' → log el/m³; 'sym_lin' → symmetric linear el/m³; 'log10' → log10 units
        maps_cfg = [
            (iri_lin,   ne_cmap,  ne_norm,  ne_bounds, col_titles[0], 'log'),
            (bkg_lin,   ne_cmap,  ne_norm,  ne_bounds, col_titles[1], 'log'),
            (m10_lin,   ne_cmap,  ne_norm,  ne_bounds, col_titles[2], 'log'),
            (m01_lin,   ne_cmap,  ne_norm,  ne_bounds, col_titles[3], 'log'),
            (fuse_lin,  ne_cmap,  ne_norm,  ne_bounds, col_titles[4], 'log'),
            (delta_map, dl_cmap,  dl_norm,  None,      col_titles[5], 'log10'),
        ]

        for col_idx, (data, cmap_d, norm_d, bounds, col_title, cb_mode) in enumerate(maps_cfg):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(data, extent=extent, origin='lower',
                           cmap=cmap_d, norm=norm_d, aspect='auto')
            if row_idx == 0:
                ax.set_title(col_title, fontsize=11, fontweight='bold')
            if col_idx == 0:
                coverage = np.maximum(result['has_fy'], result['has_cosmic']).mean()
                ax.set_ylabel(
                    f'{alt} km\ncoverage={coverage:.1%}\nLat',
                    fontweight='bold')
            if row_idx < n_rows - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Longitude (°)', fontsize=9)
            cb = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
            if cb_mode == 'log':
                cb.set_label('el/m³', fontsize=8)
                _fmt_log_ticks(cb, bounds)
            else:  # 'log10'
                tick_vals = np.linspace(-_dl_max, _dl_max, 5)
                cb.set_ticks(tick_vals)
                cb.set_ticklabels([f'{v:+.2f}' for v in tick_vals], fontsize=7)
                cb.set_label(r'$\Delta \log_{10} N_e$', fontsize=8)

    plt.suptitle(
        f'{model_name} 全球切片  Day {target_day}  {target_hour:02d}:00 UT\n'
        f'Kp = {kp_disp:.1f}   F10.7 = {f107_disp:.1f}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    fname = f'global_slice_day{target_day:02d}_h{target_hour:02d}.png'
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  全球切片已保存: {save_path}')


# ======================== 垂直 EDP 廓线 ========================

def plot_altitude_profile(model, sw_manager, device, lat, lon, time_hour,
                          save_dir, config, model_name='MDIA-INR',
                          iri_peak_manager=None, isr_record=None,
                          time_hours=None, fy_nb_index=None,
                          cosmic_nb_index=None):
    """
    绘制指定位置的垂直 EDP 廓线。

    五条曲线（+ 可选 ISR 真值）：
        Raw IRI、FNDA Background(M00)、M10、M01、M11
        ISR Ne    (红色散点)    — ISR 实测真值（isr_record 不为 None 时叠绘）

    Args:
        model:            MDIA_INR_Model 或 FSIA_INR_Model
        sw_manager:       SpaceWeatherManager
        device:           torch.device
        lat:              纬度 (°)
        lon:              经度 (°)
        time_hour:        相对小时数（从 start_date 起算）；time_hours 为 None 时使用
        save_dir:         保存目录
        config:           配置字典（读取 alt_range）
        model_name:       模型名称，用于标题和标签（默认 'MDIA-INR'）
        iri_peak_manager: IRIPeakManager（可选）
        isr_record:       load_jicamarca 返回的 DayRecord dict（可选，不为 None 时
                          在每个面板上叠绘 ISR 实测廓线作为真值）
        time_hours:       list of float — 多时刻相对小时数；提供时创建多面板图，
                          每个面板对应一个时刻（忽略 time_hour 参数）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 确定绘图时刻列表
    t_list = list(time_hours) if time_hours is not None else [time_hour]

    alt_min, alt_max = config.get('alt_range', (120.0, 500.0))
    alts = np.linspace(alt_min, alt_max, 76, dtype=np.float32)  # ~5 km 步长

    n_panels = len(t_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 10),
                             sharey=True, squeeze=False)
    axes = axes[0]   # shape: (n_panels,)

    for col_idx, t_hour in enumerate(t_list):
        ax = axes[col_idx]
        sw_seq_single = _get_sw_seq(sw_manager, t_hour, device)

        coords_np = np.column_stack([
            np.full(len(alts), lat,    dtype=np.float32),
            np.full(len(alts), lon,    dtype=np.float32),
            alts,
            np.full(len(alts), t_hour, dtype=np.float32),
        ])

        result = _infer_grid(model, coords_np, sw_seq_single, device,
                             sw_manager,
                             iri_peak_manager=iri_peak_manager,
                             fy_nb_index=fy_nb_index,
                             cosmic_nb_index=cosmic_nb_index)

        day = int(t_hour // 24)
        hr  = int(t_hour % 24)
        lt_float = (hr + lon / 15.0) % 24.0
        lt_h = int(lt_float)
        lt_m = int(round((lt_float - lt_h) * 60))
        if lt_m == 60:
            lt_h = (lt_h + 1) % 24
            lt_m = 0

        ax.plot(result['ne_iri'], alts, color='black', lw=1.5, ls='--',
                label='Raw IRI')
        ax.plot(result['background'], alts, color='gray', lw=1.5, ls=':',
                label='FNDA Background = M00')
        ax.plot(result['m10'], alts, color='tab:green', lw=1.3,
                label='M10 FY')
        ax.plot(result['m01'], alts, color='tab:orange', lw=1.3,
                label='M01 COSMIC')
        ax.plot(result['m11'], alts, color='blue', lw=2,
                label=f'M11 {model_name}')

        # ISR 真值叠绘
        if isr_record is not None:
            ne_log, alt_isr = _extract_isr_profile(isr_record, t_hour)
            if ne_log is not None:
                ax.plot(ne_log, alt_isr, color='red', lw=1.5, alpha=0.85,
                        label='Jicamarca ISR')
            else:
                print(f'  [EDP] {hr:02d}:00 UT (LT {lt_h:02d}:{lt_m:02d}) — 无近邻 ISR 数据（容差 30 min）')

        ax.set_xlabel(r'$\log_{10}$ Ne (m$^{-3}$)', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel('Altitude (km)', fontsize=12)
        ax.set_title(f'{hr:02d}:00 UT  (LT {lt_h:02d}:{lt_m:02d})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(alt_min, alt_max)
        covered = np.maximum(result['has_fy'], result['has_cosmic']) > 0.5
        changed = np.abs(result['m11'] - result['background']) > 1e-6
        print(
            f'  [EDP] {hr:02d}:00 coverage={covered.mean():.1%}, '
            f'M11 changed={changed.mean():.1%}, '
            f'|M11-B| mean={np.abs(result["ne_delta"]).mean():.4f} dex')

    # 总标题
    day0      = int(t_list[0] // 24)
    date_disp = _rel_to_datestr(t_list[0])
    plt.suptitle(
        f'{model_name} EDP Profile  —  Lat={lat:.1f}°  Lon={lon:.1f}°\n'
        f'Day {day0}  ({date_disp})',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    # 文件命名
    if n_panels == 1:
        hr0   = int(t_list[0] % 24)
        fname = f'edp_lat{lat:.0f}_lon{lon:.0f}_d{day0:02d}h{hr0:02d}.png'
    else:
        hrs_tag = '_'.join(f'{int(t % 24):02d}' for t in t_list)
        fname   = f'edp_lat{lat:.0f}_lon{lon:.0f}_d{day0:02d}_h{hrs_tag}.png'

    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  EDP 廓线已保存: {save_path}')


# ======================== hmF2 + NmF2 全时间步合并图 ========================

def plot_hmf2_nmf2_map(model, sw_manager, device, time_steps, save_dir,
                       label=None, model_name='MDIA-INR', iri_peak_manager=None,
                       fy_nb_index=None, cosmic_nb_index=None):
    """
    绘制 hmF2 + NmF2 全球分布合并图（所有时间步纵向排列，每行左右两列）。

    左列：F2 峰高 hmF2 (km)
    右列：F2 峰值电子密度 NmF2 (el/m³，由 log10 还原）
    两列均使用离散颜色条，所有时间步共享同一色标便于横向比较。

    建议：同一天的若干时次组成一个 time_steps，按日分画布调用；
    label 参数用于文件命名区分（如 'day04'）。

    Args:
        model:       MDIA_INR_Model 或 FSIA_INR_Model（eval 模式）
        sw_manager:  SpaceWeatherManager
        device:      torch.device
        time_steps:  list of (target_day, target_hour) — 同一画布内按顺序排列的时间步
        save_dir:    保存目录
        config:      配置字典
        label:       文件名后缀标识（如 'day04'）；None 时用 'all_timesteps'
        model_name:  模型名称，用于标题（默认 'MDIA-INR'）
    """
    os.makedirs(save_dir, exist_ok=True)

    n_rows = len(time_steps)
    fig, axes = plt.subplots(n_rows, 2, figsize=(24, 4.5 * n_rows))
    axes = np.atleast_2d(axes)  # 保证二维索引 [row, col]

    lat_grid = np.linspace(-90, 90, 91)
    lon_grid = np.linspace(-180, 180, 180)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    extent = [-180, 180, -90, 90]

    # 所有时间步共享统一色标
    hmf2_cmap, hmf2_norm = _discrete_norm('jet', 330, 480, n=20)
    # NmF2: log10 空间 [10.5, 13.0] → el/m³ 对数间距 20 级
    nmf2_cmap, nmf2_norm, nmf2_bounds = _discrete_norm_log('jet', 10.5, 12.1, n=20)

    for row_idx, (target_day, target_hour) in enumerate(time_steps):
        global_time = target_day * 24.0 + target_hour
        print(f'[可视化] hmF2/NmF2  Day {target_day}  {target_hour:02d}:00 UT')

        sw_seq_single = _get_sw_seq(sw_manager, global_time, device)
        kp_disp, f107_disp = _sw_display_values(sw_seq_single)

        hmf2, nmf2 = _infer_analysis_peaks(
            model, LAT.flatten().astype(np.float32),
            LON.flatten().astype(np.float32), global_time,
            sw_seq_single, device, sw_manager, iri_peak_manager,
            fy_nb_index, cosmic_nb_index)
        hmf2_map = hmf2.reshape(LAT.shape)
        nmf2_map = (10.0 ** nmf2).reshape(LAT.shape)

        row_label = (f'Day {target_day}  {target_hour:02d}:00 UT\n'
                     f'Kp={kp_disp:.1f}  F10.7={f107_disp:.1f}')

        # ---------- 左列：hmF2 ----------
        ax_h = axes[row_idx, 0]
        im_h = ax_h.imshow(hmf2_map, extent=extent, origin='lower',
                           cmap=hmf2_cmap, norm=hmf2_norm, aspect='auto')
        if row_idx == 0:
            ax_h.set_title('F2 峰高  hmF2', fontsize=11, fontweight='bold')
        ax_h.set_ylabel(row_label, fontsize=8)
        ax_h.set_xticks([-120, -60, 0, 60, 120])
        if row_idx < n_rows - 1:
            ax_h.set_xticklabels([])
        else:
            ax_h.set_xlabel('Longitude (°)', fontsize=9)
        cb_h = plt.colorbar(im_h, ax=ax_h, pad=0.02, shrink=0.85)
        cb_h.set_label('km', fontsize=9)

        # ---------- 右列：NmF2 ----------
        ax_n = axes[row_idx, 1]
        im_n = ax_n.imshow(nmf2_map, extent=extent, origin='lower',
                           cmap=nmf2_cmap, norm=nmf2_norm, aspect='auto')
        if row_idx == 0:
            ax_n.set_title('F2 峰值电子密度  NmF2', fontsize=11, fontweight='bold')
        ax_n.set_yticks([])
        ax_n.set_xticks([-120, -60, 0, 60, 120])
        if row_idx < n_rows - 1:
            ax_n.set_xticklabels([])
        else:
            ax_n.set_xlabel('Longitude (°)', fontsize=9)
        cb_n = plt.colorbar(im_n, ax=ax_n, pad=0.02, shrink=0.85)
        cb_n.set_label('el/m³', fontsize=9)
        _fmt_log_ticks(cb_n, nmf2_bounds)

    plt.suptitle(f'{model_name}  F2 峰高 hmF2 与峰值电子密度 NmF2 全球分布',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    tag   = label if label is not None else 'all_timesteps'
    fname = f'hmf2_nmf2_{tag}.png'
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  hmF2/NmF2 合并图已保存: {save_path}')
