"""
MDIA-INR / FSIA-INR 局地时垂直切片可视化

功能：
    plot_vertical_slice_lt — 固定经度扇区，多地方时垂直（纬度×高度）切片图
                              横轴：地理纬度 90°N → 90°S
                              纵轴：高度 120–500 km
                              色标：与 plot_global_slice 一致（log 离散 jet，7×10⁹–4×10¹² el/m³）

时间约定：
    输入为模型世界时（UT）的 global_time（小数小时，距 start_date 起算）。
    地方时由 LT = UT + lon/15 精确换算，以 HH:MM 格式显示。

独立运行：
    cd MDIA_INR
    python inr_modules/mdia/visualization_mdia_lt.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STXihei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 从同目录导入辅助函数（与 visualization_mdia.py 共享色标工具）
try:
    from .visualization_mdia import (
        _infer_grid, _get_sw_seq, _sw_display_values,
        _discrete_norm_log, _fmt_log_ticks,
    )
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from visualization_mdia import (
        _infer_grid, _get_sw_seq, _sw_display_values,
        _discrete_norm_log, _fmt_log_ticks,
    )


# ======================== 辅助：从 Ne 剖面计算 hmF2 ========================

def _hmf2_from_ne(ne_lin, alt_grid):
    """
    从离散 Ne_fused 剖面精确计算 F2 峰高 hmF2。

    步骤：
      1. argmax 沿高度轴粗定位峰格点 k（精度 = 格点间距，5 km）
      2. 对 k-1, k, k+1 三点做抛物线插值，求亚格点峰位（~1 km 精度）
         δ = -Δh/2 × (Ne[k+1] − Ne[k-1]) / (Ne[k-1] − 2Ne[k] + Ne[k+1])

    Args:
        ne_lin:   [n_lat, n_alt] float32，线性 Ne（el/m³）
        alt_grid: [n_alt] float32，高度坐标（km），均匀间隔
    Returns:
        hmf2:     [n_lat] float32（km）
    """
    n_lat, n_alt = ne_lin.shape
    dh    = float(alt_grid[1] - alt_grid[0])
    k_arr = np.argmax(ne_lin, axis=1)           # [n_lat]  粗定位
    hmf2  = alt_grid[k_arr].astype(np.float64)

    for i in range(n_lat):
        k = k_arr[i]
        if 1 <= k <= n_alt - 2:
            n0, n1, n2 = float(ne_lin[i, k-1]), float(ne_lin[i, k]), float(ne_lin[i, k+1])
            denom = n0 - 2.0 * n1 + n2          # < 0 → 凹峰（正常 F2 层）
            if denom < 0.0:
                delta = -0.5 * dh * (n2 - n0) / denom
                hmf2[i] = alt_grid[k] + float(np.clip(delta, -dh, dh))

    return hmf2.astype(np.float32)


# ======================== 垂直切片（纬度 × 高度）========================

def plot_vertical_slice_lt(
        model, sw_manager, device,
        global_times,
        lon_sector,
        save_dir, config,
        model_name='FSIA-INR',
        n_lat=181,
        n_alt=77,
        iri_peak_manager=None,
):
    """
    绘制固定经度扇区、多世界时时刻的垂直（纬度×高度）切片图。

    时间处理：
        输入 global_times 为模型世界时（UT），距 start_date 起算的小数小时列表。
        地方时由 LT = UT + lon/15 精确换算，以 HH:MM 格式显示于子图标题。

    布局：2 行 × 3 列，按行优先填充 global_times（需恰好 6 个）。
    色标与 plot_global_slice Ne 面板完全一致：
        log 离散，jet colormap，7×10⁹–4×10¹² el/m³，20 级。
    叠加白色虚线：从 Ne_fused 场中逐纬度计算的 hmF2（抛物线插值精化）。

    Args:
        model:        MDIA_INR_Model 或 FSIA_INR_Model（eval 模式）
        sw_manager:   SpaceWeatherManager
        device:       torch.device
        global_times: list of float，6 个 global_time（UT 小时，距 start_date 起算）
        lon_sector:   经度扇区 (°)，如 -165.0
        save_dir:     输出目录
        config:       配置字典
        model_name:   模型名称，用于标题
        n_lat:        纬度格点数（默认 181，步长 1°）
        n_alt:        高度格点数（默认 77，120–500 km，步长 5 km）
    """
    if len(global_times) != 6:
        raise ValueError(f'global_times 需要恰好 6 个时刻，收到 {len(global_times)} 个')

    os.makedirs(save_dir, exist_ok=True)

    # ---- 坐标网格 ----
    lat_grid = np.linspace(-90, 90,  n_lat, dtype=np.float32)   # S→N
    alt_grid = np.linspace(120, 500, n_alt, dtype=np.float32)   # 120–500 km
    LAT_2D, ALT_2D = np.meshgrid(lat_grid, alt_grid, indexing='ij')  # [n_lat, n_alt]

    # ---- 固定色标（与 plot_global_slice 一致）----
    _ne_lo = np.log10(7e9)
    _ne_hi = np.log10(4e12)
    ne_cmap, ne_norm, ne_bounds = _discrete_norm_log('jet', _ne_lo, _ne_hi, n=20)

    # ---- global_time → LT/UT 标签（LT = UT + lon/15，精确到分钟）----
    lt_offset = lon_sector / 15.0

    def _gt_to_labels(gt):
        ut_h   = gt % 24.0
        ut_day = int(gt // 24)
        lt_h   = (ut_h + lt_offset) % 24.0
        lt_int = int(lt_h)
        lt_min = round((lt_h - lt_int) * 60)
        if lt_min == 60:
            lt_int += 1
            lt_min  = 0
        return f'{lt_int % 24:02d}:{lt_min:02d}', ut_day, ut_h

    # ---- 画布（2 行 × 3 列）----
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    ims = []
    iri_panels = []   # 缓存 IRI 面板数据，供第二张画布使用

    for idx, global_time in enumerate(global_times):
        ax = axes_flat[idx]
        lt_str, ut_day, ut_h = _gt_to_labels(global_time)
        ut_int = int(ut_h)
        ut_min = round((ut_h % 1) * 60)

        print(f'[垂直切片] global_time={global_time:.2f}h  →  '
              f'UT {ut_int:02d}:{ut_min:02d} Day {ut_day}  LT = {lt_str}')

        sw_seq_single      = _get_sw_seq(sw_manager, global_time, device)
        kp_disp, f107_disp = _sw_display_values(sw_seq_single)

        # ---- 构造 coords [n_lat×n_alt, 4] ----
        coords_np = np.column_stack([
            LAT_2D.flatten(),
            np.full(LAT_2D.size, lon_sector,  dtype=np.float32),
            ALT_2D.flatten(),
            np.full(LAT_2D.size, global_time, dtype=np.float32),
        ]).astype(np.float32)

        # ---- 推理 ----
        result    = _infer_grid(model, coords_np, sw_seq_single, device,
                                iri_peak_manager=iri_peak_manager)
        ne_lin    = (10.0 ** result['ne_fused']).reshape(n_lat, n_alt)
        hmf2_line = _hmf2_from_ne(ne_lin, alt_grid)          # [n_lat] 廓线 argmax

        # PeakHead 直接输出 hmF2（FSIA v3.0，km；无 iri_peak_manager 时为 NaN）
        _hmf2_pk_raw = result['hmf2_f2'].reshape(n_lat, n_alt)[:, 0]   # [n_lat]
        _has_pk = not np.all(np.isnan(_hmf2_pk_raw))

        # IRI 背景：同步提取，缓存供第二张画布
        ne_iri   = (10.0 ** result['ne_bkg']).reshape(n_lat, n_alt)
        hmf2_iri = _hmf2_from_ne(ne_iri, alt_grid)
        iri_panels.append((lt_str, ut_int, ut_min, ut_day,
                           kp_disp, f107_disp, ne_iri, hmf2_iri))

        # ---- pcolormesh(x=lat, y=alt, C[n_alt, n_lat]) ----
        im = ax.pcolormesh(
            lat_grid, alt_grid, ne_lin.T,
            cmap=ne_cmap, norm=ne_norm, shading='auto',
        )
        ims.append(im)

        # ---- hmF2 叠加（两条线）----
        ax.plot(lat_grid, hmf2_line, color='white',  lw=1.4, ls='--',
                alpha=0.85, label='hmF2 (profile)')
        if _has_pk:
            ax.plot(lat_grid, _hmf2_pk_raw, color='orange', lw=1.2, ls=':',
                    alpha=0.80, label='hmF2 (PeakHead)')

        # ---- 子图标题 ----
        ax.set_title(
            f'LT = {lt_str}   (UT {ut_int:02d}:{ut_min:02d} Day {ut_day})\n'
            f'Kp = {kp_disp:.1f}   F10.7 = {f107_disp:.1f}',
            fontsize=9,
        )

        # ---- 坐标轴：横轴 -90→90（北极在右）----
        ax.set_xlim(-90, 90)
        ax.set_ylim(120, 500)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(['60°N', '30°N', '0°', '30°S', '60°S'], fontsize=8)
        ax.set_yticks([150, 200, 250, 300, 350, 400, 450, 500])
        ax.set_yticklabels(['150', '200', '250', '300', '350', '400', '450', '500'], fontsize=8)

        if idx >= 3:
            ax.set_xlabel('Latitude', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Altitude (km)', fontsize=10)

        ax.grid(True, alpha=0.2, lw=0.5, color='white')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right',
                      framealpha=0.55, labelcolor='white', facecolor='#333333')

    # ---- 共享 colorbar ----
    fig.subplots_adjust(right=0.87, hspace=0.40, wspace=0.08)
    cbar_ax = fig.add_axes([0.895, 0.10, 0.016, 0.76])
    cb = fig.colorbar(ims[0], cax=cbar_ax)
    cb.set_label('Electron Density (el/m³)', fontsize=10, labelpad=6)
    _fmt_log_ticks(cb, ne_bounds)
    cb.ax.tick_params(labelsize=8)

    # ---- 总标题 ----
    lt_str0, _, _ = _gt_to_labels(global_times[0])
    lt_str1, _, _ = _gt_to_labels(global_times[-1])
    _ref_day = int(min(global_times) // 24)
    fig.suptitle(
        f'{model_name}  垂直电子密度切片\n'
        f'经度扇区 Lon = {lon_sector:.1f}°   LT 范围 {lt_str0}–{lt_str1}   '
        f'参考日期 2024-09-{1 + _ref_day:02d}（第 {_ref_day} 日）\n'
        f'横轴：纬度 90°N → 90°S     纵轴：高度 120–500 km     虚线：Ne_fused hmF2',
        fontsize=11, fontweight='bold', y=0.995,
    )

    # ---- 保存 ----
    lon_tag   = f'{int(abs(lon_sector))}{"W" if lon_sector < 0 else "E"}'
    gt_tag    = f'gt{int(global_times[0])}-{int(global_times[-1])}'
    save_path = os.path.join(save_dir, f'vertical_slice_lt_lon{lon_tag}_{gt_tag}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  垂直切片已保存: {save_path}')

    # =====================================================================
    # 第二张画布：IRI 背景电子密度（与第一张布局/色标完全一致）
    # =====================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    axes2_flat  = axes2.flatten()
    ims2 = []

    for idx, (lt_str, ut_int, ut_min, ut_day,
              kp_disp, f107_disp, ne_iri, hmf2_iri) in enumerate(iri_panels):
        ax2 = axes2_flat[idx]

        im2 = ax2.pcolormesh(
            lat_grid, alt_grid, ne_iri.T,
            cmap=ne_cmap, norm=ne_norm, shading='auto',
        )
        ims2.append(im2)

        ax2.plot(lat_grid, hmf2_iri, color='white', lw=1.4, ls='--', alpha=0.85, label='hmF2')

        ax2.set_title(
            f'LT = {lt_str}   (UT {ut_int:02d}:{ut_min:02d} Day {ut_day})\n'
            f'Kp = {kp_disp:.1f}   F10.7 = {f107_disp:.1f}',
            fontsize=9,
        )

        ax2.set_xlim(-90, 90)
        ax2.set_ylim(120, 500)
        ax2.set_xticks([-60, -30, 0, 30, 60])
        ax2.set_xticklabels(['60°N', '30°N', '0°', '30°S', '60°S'], fontsize=8)
        ax2.set_yticks([150, 200, 250, 300, 350, 400, 450, 500])
        ax2.set_yticklabels(['150', '200', '250', '300', '350', '400', '450', '500'], fontsize=8)

        if idx >= 3:
            ax2.set_xlabel('Latitude', fontsize=10)
        if idx % 3 == 0:
            ax2.set_ylabel('Altitude (km)', fontsize=10)

        ax2.grid(True, alpha=0.2, lw=0.5, color='white')
        if idx == 0:
            ax2.legend(fontsize=7, loc='upper right',
                       framealpha=0.55, labelcolor='white', facecolor='#333333')

    fig2.subplots_adjust(right=0.87, hspace=0.40, wspace=0.08)
    cbar_ax2 = fig2.add_axes([0.895, 0.10, 0.016, 0.76])
    cb2 = fig2.colorbar(ims2[0], cax=cbar_ax2)
    cb2.set_label('Electron Density (el/m³)', fontsize=10, labelpad=6)
    _fmt_log_ticks(cb2, ne_bounds)
    cb2.ax.tick_params(labelsize=8)

    fig2.suptitle(
        f'IRI 背景  垂直电子密度切片\n'
        f'经度扇区 Lon = {lon_sector:.1f}°   LT 范围 {lt_str0}–{lt_str1}   '
        f'参考日期 2024-09-{1 + _ref_day:02d}（第 {_ref_day} 日）\n'
        f'横轴：纬度 90°N → 90°S     纵轴：高度 120–500 km     虚线：IRI hmF2',
        fontsize=11, fontweight='bold', y=0.995,
    )

    save_path_iri = os.path.join(save_dir, f'vertical_slice_iri_lon{lon_tag}_{gt_tag}.png')
    plt.savefig(save_path_iri, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  IRI 切片已保存: {save_path_iri}')

    return save_path, save_path_iri


# ======================== LT-纬度分布图 ========================

def plot_lt_lat_map(
        model, sw_manager, device,
        target_day,
        lon_sector,
        save_dir, config,
        model_name='FSIA-INR',
        n_lat=181,
        n_alt=77,
        n_lt=49,
        iri_peak_manager=None,
):
    """
    绘制固定经度扇区、某一日全天地方时-纬度分布图（2 行 × 2 列）。

    行 1：IRI 背景   — NmF2 | hmF2（廓线 argmax）
    行 2：model_name — NmF2 | hmF2（从最终 Ne_fused 廓线 argmax + 抛物线插值精化）

    色标：
        NmF2  log10 离散 jet，8×10¹⁰ – 4×10¹² el/m³，20 级
        hmF2  线性离散 plasma，200 – 500 km，15 级（步长 20 km）

    扫描：lt_grid = linspace(0,24,n_lt)（默认 0.5h，49 步）
          UT = LT − lon/15；global_time = (day+floor(UT/24))×24 + UT%24

    Args:
        target_day:  0-indexed 日序（4 = 2024-09-05）
        lon_sector:  固定经度 (°)
        n_lt:        LT 扫描格点数（默认 49，步长 0.5h）
    """
    import matplotlib.colors as _mcolors

    os.makedirs(save_dir, exist_ok=True)

    lat_grid = np.linspace(-90, 90,  n_lat, dtype=np.float32)
    alt_grid = np.linspace(120, 500, n_alt, dtype=np.float32)
    lt_grid  = np.linspace(0,   24,  n_lt,  dtype=np.float32)
    LAT_2D, ALT_2D = np.meshgrid(lat_grid, alt_grid, indexing='ij')  # [n_lat, n_alt]

    lt_offset = lon_sector / 15.0   # LT→UT 偏移（lon=-165° → -11h）

    def _lt_to_global(lt):
        ut = lt - lt_offset
        return (target_day + int(ut // 24)) * 24.0 + (ut % 24)

    # FSIA 和 IRI 各自的峰值网格
    nmf2_grid     = np.zeros((n_lat, n_lt), dtype=np.float32)
    hmf2_grid     = np.zeros((n_lat, n_lt), dtype=np.float32)
    nmf2_iri_grid = np.zeros((n_lat, n_lt), dtype=np.float32)
    hmf2_iri_grid = np.zeros((n_lat, n_lt), dtype=np.float32)

    print(f'[LT-lat图] 扫描 {n_lt} 步（步长 {24/(n_lt-1):.2f}h），'
          f'lon={lon_sector:.1f}°，Day {target_day}（2024-09-{1+target_day:02d}）')
    for j, lt in enumerate(lt_grid):
        if j % 10 == 0:
            print(f'  {j+1}/{n_lt}  LT={lt:.1f}h')
        global_time = _lt_to_global(float(lt))
        sw_seq      = _get_sw_seq(sw_manager, global_time, device)

        coords_np = np.column_stack([
            LAT_2D.flatten(),
            np.full(LAT_2D.size, lon_sector,  dtype=np.float32),
            ALT_2D.flatten(),
            np.full(LAT_2D.size, global_time, dtype=np.float32),
        ]).astype(np.float32)

        result = _infer_grid(model, coords_np, sw_seq, device,
                             iri_peak_manager=iri_peak_manager)

        ne_fsia = (10.0 ** result['ne_fused']).reshape(n_lat, n_alt)
        nmf2_grid[:, j] = np.max(ne_fsia, axis=1)

        # hmF2 来自最终模型输出 Ne_fused 廓线 argmax + 抛物线插值精化
        # （不使用 PeakHead 中间产物 result['hmf2_f2']）
        hmf2_grid[:, j] = _hmf2_from_ne(ne_fsia, alt_grid)

        ne_iri  = (10.0 ** result['ne_bkg']).reshape(n_lat, n_alt)
        nmf2_iri_grid[:, j] = np.max(ne_iri,  axis=1)
        hmf2_iri_grid[:, j] = _hmf2_from_ne(ne_iri,  alt_grid)

    # ---- 色标定义 ----
    # NmF2：log10 离散 jet，8×10¹⁰ – 4×10¹²
    _ne_lo = np.log10(8e10)
    _ne_hi = np.log10(4e12)
    ne_cmap, ne_norm, ne_bounds = _discrete_norm_log('jet', _ne_lo, _ne_hi, n=20)

    # hmF2：线性离散 plasma，200–500 km，15 级（20 km/级）
    _hm_bounds = np.linspace(200, 500, 21)                  # 16 边界 → 15 区间
    _hm_cmap   = plt.get_cmap('plasma', 20)
    _hm_norm   = _mcolors.BoundaryNorm(_hm_bounds, ncolors=_hm_cmap.N)

    # ---- 画布（2 行 × 2 列）----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

    def _setup_ax(ax, title, row):
        ax.set_xlim(0, 24)
        ax.set_ylim(-90, 90)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(['00', '03', '06', '09', '12', '15', '18', '21', '24'], fontsize=8)
        ax.set_yticklabels(['60°S', '30°S', '0°', '30°N', '60°N'], fontsize=8)
        if row == 1:
            ax.set_xlabel('Local Time (h)', fontsize=10)
        ax.set_ylabel('Geographic Latitude', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25, lw=0.5)

    def _add_nmf2_cb(im, ax):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('NmF2 (el/m³)', fontsize=9, labelpad=5)
        _fmt_log_ticks(cb, ne_bounds)
        cb.ax.tick_params(labelsize=7)

    def _add_hmf2_cb(im, ax):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('hmF2 (km)', fontsize=9, labelpad=5)
        cb.set_ticks([200, 250, 300, 350, 400, 450, 500])
        cb.ax.tick_params(labelsize=7)

    # 行 0：IRI
    im00 = axes[0, 0].pcolormesh(lt_grid, lat_grid, nmf2_iri_grid,
                                  cmap=ne_cmap, norm=ne_norm, shading='auto')
    _add_nmf2_cb(im00, axes[0, 0])
    _setup_ax(axes[0, 0], '(a) IRI  NmF2', row=0)

    im01 = axes[0, 1].pcolormesh(lt_grid, lat_grid, hmf2_iri_grid,
                                  cmap=_hm_cmap, norm=_hm_norm, shading='auto')
    _add_hmf2_cb(im01, axes[0, 1])
    _setup_ax(axes[0, 1], '(b) IRI  hmF2', row=0)

    # 行 1：FSIA
    im10 = axes[1, 0].pcolormesh(lt_grid, lat_grid, nmf2_grid,
                                  cmap=ne_cmap, norm=ne_norm, shading='auto')
    _add_nmf2_cb(im10, axes[1, 0])
    _setup_ax(axes[1, 0], f'(c) {model_name}  NmF2', row=1)

    im11 = axes[1, 1].pcolormesh(lt_grid, lat_grid, hmf2_grid,
                                  cmap=_hm_cmap, norm=_hm_norm, shading='auto')
    _add_hmf2_cb(im11, axes[1, 1])
    _setup_ax(axes[1, 1], f'(d) {model_name}  hmF2', row=1)

    lon_tag = f'{int(abs(lon_sector))}{"W" if lon_sector < 0 else "E"}'
    fig.suptitle(
        f'F2 层峰值参数   地方时–纬度分布\n'
        f'经度扇区 Lon = {lon_sector:.1f}°   日期 2024-09-{1+target_day:02d}（第 {target_day} 日）\n'
        f'上行：IRI 背景     下行：{model_name}',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(save_dir, f'lt_lat_map_lon{lon_tag}_day{target_day:02d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  LT-纬度分布图已保存: {save_path}')
    return save_path


# ======================== 独立运行入口 ========================

if __name__ == '__main__':
    # __file__ = .../MDIA_INR/inr_modules/mdia/visualization_mdia_lt.py
    # 上两级到达 MDIA_INR/，inr_modules 作为包从此处导入
    _here      = os.path.dirname(os.path.abspath(__file__))
    _mdia_root = os.path.abspath(os.path.join(_here, '../..'))
    if _mdia_root not in sys.path:
        sys.path.insert(0, _mdia_root)

    # ================================================================
    # 配置区（按需修改）
    # ================================================================
    CONFIG = {
        # 模型类型：'fsia' 或 'mdia'
        'model_type':      'fsia',

        # 检查点路径（None = 自动推断 {save_dir}/best_fsia_model.pth）
        'checkpoint_path': r"D:\code11\IRI01\IRI03\INR1-1\FSIA_INR18\checkpoints_fsia\run58\best_fsia_model.pth",

        # 输出目录（None = {save_dir}/plots/lt_slice/）
        'save_dir_out':    r'D:\code11\IRI01\IRI03\INR1-1\FSIA_INR18\checkpoints_fsia\run58\plots\lt_slice10',

        # 经度扇区（°）
        'lon_sector':     -165.0,

        # 6 个 global_time（UT 小时，距 2024-09-01 00:00 UT 起算），2×3 行优先排列
        # 地方时由 LT = UT + lon/15 精确换算后显示（HH:MM）
        # 示例：lon=-165° → LT offset=-11h
        #   119 → UT 23:00 Day4 → LT 12:00
        #   120 → UT 00:00 Day5 → LT 13:00  ...  124 → LT 17:00
        'global_times':   [119.0, 120.0, 121.0, 122.0, 123.0, 124.0],

        # 模型名称（标题显示）
        'model_name':     'FSIA-INR',
    }
    # ================================================================

    from inr_modules.config_mdia import get_config_mdia, update_config_mdia
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.data_managers.iri_peak_manager import IRIPeakManager

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[可视化] 使用设备: {device}')

    mdia_cfg   = get_config_mdia()
    model_type = CONFIG['model_type']

    # FSIA 检查点目录 override（与 main_fsia.py 保持一致）
    if model_type == 'fsia':
        update_config_mdia(save_dir=os.path.join(_mdia_root, 'checkpoints_fsia', 'run57'))

    # ---- 加载 IRI 代理 ----
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    iri_proxy.load_state_dict(torch.load(mdia_cfg['iri_proxy_path'], map_location=device))
    iri_proxy.eval()

    # ---- IRIPeakManager（FSIA v2.2+，文件不存在时自动降级为中性后备值）----
    iri_peak_manager = None
    if model_type == 'fsia':
        try:
            iri_peak_manager = IRIPeakManager(
                hmf2_path=mdia_cfg.get('iri_hmf2_path', ''),
                nmf2_path=mdia_cfg.get('iri_nmf2_path', ''),
                device=device,
            )
            print('[可视化] IRIPeakManager 已加载')
        except Exception as _e:
            print(f'[可视化] IRIPeakManager 初始化失败（{_e}），PeakHead 使用中性后备值')

    # ---- 加载模型 ----
    if model_type == 'fsia':
        from inr_modules.mdia.fsia_model import FSIA_INR_Model
        model     = FSIA_INR_Model(iri_proxy=iri_proxy, config=mdia_cfg).to(device)
        _def_ckpt = os.path.join(mdia_cfg['save_dir'], 'best_fsia_model.pth')
    else:
        from inr_modules.mdia.mdia_model import MDIA_INR_Model
        model     = MDIA_INR_Model(iri_proxy=iri_proxy, config=mdia_cfg).to(device)
        _def_ckpt = os.path.join(mdia_cfg['save_dir'], 'best_mdia_model.pth')

    ckpt_path = CONFIG['checkpoint_path'] or _def_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'检查点不存在: {ckpt_path}')
    _sd = torch.load(ckpt_path, map_location=device)

    # ---- run27 兼容：检查点用 MultiHeadDiagonalKalmanLayer (n_heads>1)，
    # 当前 FSIA_INR_Model 默认装配 NeuralETKFLayer。检测到 H_FY_w 为 3D
    # ([H, d, d]) 时，先把 model.kalman_layer 替换为 MHDK 再 load。
    # FSIA_INR_Model.forward 调用签名为 13 实参（含 alt_km, hmF2_det，run28 NeuralETKFLayer
    # 专用），MHDK 只取前 11 个 → 用 *args/**kwargs 包一层吞掉多余参数。----
    if model_type == 'fsia':
        _hfy = _sd.get('kalman_layer.H_FY_w', None)
        if _hfy is not None and _hfy.dim() == 3:
            from inr_modules.mdia.fsia_model import MultiHeadDiagonalKalmanLayer
            _n_heads, _d_in_b, _d_model = _sd['kalman_layer.B_w1'].shape
            _, _d_in_r, _ = _sd['kalman_layer.R_w1'].shape
            _hidden = _sd['kalman_layer.B_w2'].shape[1]
            _hg_hidden = _sd['kalman_layer.head_gate.0.bias'].shape[0]
            print(f'[可视化] 检测到 MHDK 检查点：n_heads={_n_heads}, d={_d_model}, '
                  f'b_in={_d_in_b}, r_in={_d_in_r}, hidden={_hidden}, head_gate_h={_hg_hidden}')

            class _MHDKShim(MultiHeadDiagonalKalmanLayer):
                """吞掉 FSIA_INR_Model 传给 NeuralETKFLayer 的额外 (alt_km, hmF2_det) 参数。"""
                def forward(self, f_iri, h_obs_FY, h_res, h_sw,
                            lat_n, cos_SZA, sin_doy, cos_doy, sin_I,
                            alt_n, delta_alt_n, *extra, **kw):
                    return super().forward(
                        f_iri, h_obs_FY, h_res, h_sw,
                        lat_n, cos_SZA, sin_doy, cos_doy, sin_I,
                        alt_n, delta_alt_n,
                    )

                @torch.no_grad()
                def compute_proxy_trust_iri(self, h_sw, lat_n, cos_SZA, sin_doy, cos_doy,
                                            sin_I, alt_n, delta_alt_iri, *extra, **kw):
                    return super().compute_proxy_trust_iri(
                        h_sw, lat_n, cos_SZA, sin_doy, cos_doy, sin_I, alt_n, delta_alt_iri,
                    )

            model.kalman_layer = _MHDKShim(
                d_model=_d_model,
                b_net_in=_d_in_b,
                r_fy_net_in=_d_in_r,
                n_heads=_n_heads,
                head_gate_hidden=_hg_hidden,
                hidden=_hidden,
            ).to(device)

        # ---- N-adaptive：检查点 N_members 可能与当前 model 不同（如 run56=8, run57=4）----
        _ckpt_P_w1 = _sd.get('kalman_layer.P_w1')
        if _ckpt_P_w1 is not None and _ckpt_P_w1.shape[0] != model.enkf_n_members:
            _ckpt_n = int(_ckpt_P_w1.shape[0])
            print(f'[可视化] 检查点 N={_ckpt_n} ≠ 当前 N={model.enkf_n_members}，'
                  f'自适应重建 NeuralETKFLayer(n_members={_ckpt_n})')
            from inr_modules.mdia.fsia_model import NeuralETKFLayer
            _kl = model.kalman_layer
            model.kalman_layer = NeuralETKFLayer(
                d_model    = _kl.d_model,
                b_net_in   = _kl.b_net_in,
                r_fy_net_in= _kl.r_fy_net_in,
                n_members  = _ckpt_n,
                pert_hidden= _kl.pert_hidden,
                n_rank_h   = _kl.n_rank_h,
            ).to(device)
            model.enkf_n_members = _ckpt_n

        # ---- PeakHead spatial_dim-adaptive（run<58: 144D 用 h_spatial[64]，run58+: 90D）----
        _ckpt_peak_w = _sd.get('peak_head.net.0.weight')
        if _ckpt_peak_w is not None:
            _ckpt_peak_indim = int(_ckpt_peak_w.shape[1])
            _curr_peak_indim = model.peak_head.net[0].in_features
            if _ckpt_peak_indim != _curr_peak_indim:
                _ckpt_spatial = _ckpt_peak_indim - 64 - 2 - 14   # sw=64, iri=2, sh=14
                print(f'[可视化] 检查点 PeakHead 输入 {_ckpt_peak_indim}D ≠ 当前 {_curr_peak_indim}D，'
                      f'自适应重建 PeakHead(spatial_dim={_ckpt_spatial})')
                from inr_modules.mdia.fsia_model import PeakHead as _PeakHead
                _ph = model.peak_head
                model.peak_head = _PeakHead(
                    spatial_dim=_ckpt_spatial,
                    sw_dim=64,
                    hmf2_range=(_ph.lo_h, _ph.lo_h + _ph.span_h),
                    nmf2_range=(_ph.lo_n, _ph.lo_n + _ph.span_n),
                ).to(device)

    _missing, _unexpected = model.load_state_dict(_sd, strict=False)
    if _missing:
        print(f'[可视化] ⚠ 检查点缺失 {len(_missing)} 个键（新架构组件使用初始值）:')
        for _k in _missing[:10]:
            print(f'         - {_k}')
        if len(_missing) > 10:
            print(f'         ... 共 {len(_missing)} 个')
    if _unexpected:
        print(f'[可视化] ⚠ 检查点多余 {len(_unexpected)} 个键（已忽略）:')
        for _k in _unexpected[:5]:
            print(f'         - {_k}')
    model.eval()
    print(f'[可视化] 已加载: {ckpt_path}')

    # ---- SpaceWeatherManager ----
    sw_manager = SpaceWeatherManager(
        txt_path=mdia_cfg['sw_path'],
        start_date_str=mdia_cfg['start_date_str'],
        total_hours=mdia_cfg['total_hours'],
        seq_len=mdia_cfg['seq_len'],
        device=device,
    )

    # ---- 输出目录 ----
    save_dir_out = (CONFIG['save_dir_out']
                    or os.path.join(mdia_cfg['save_dir'], 'plots', 'lt_slice'))

    # ---- 垂直切片图 ----
    plot_vertical_slice_lt(
        model             = model,
        sw_manager        = sw_manager,
        device            = device,
        global_times      = CONFIG['global_times'],
        lon_sector        = CONFIG['lon_sector'],
        save_dir          = save_dir_out,
        config            = mdia_cfg,
        model_name        = CONFIG['model_name'],
        iri_peak_manager  = iri_peak_manager,
    )

    # ---- LT-纬度分布图（Day4，0.5h 分辨率）----
    plot_lt_lat_map(
        model             = model,
        sw_manager        = sw_manager,
        device            = device,
        target_day        = 4,
        lon_sector        = CONFIG['lon_sector'],
        save_dir          = save_dir_out,
        config            = mdia_cfg,
        model_name        = CONFIG['model_name'],
        iri_peak_manager  = iri_peak_manager,
    )
    print('[可视化] 完成')
