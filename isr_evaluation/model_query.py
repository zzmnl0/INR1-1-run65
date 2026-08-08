"""
MDIA-INR 模型查询模块

在 ISR 观测坐标（纬度、经度、高度、时间）上推理 MDIA-INR，
返回与 ISR DayRecord 对齐的 log10 电子密度网格。
"""

import numpy as np
import torch
from inr_modules.mdia.sliding_dataset import (
    attach_observation_background,
    observation_query_coverage,
    query_observation_payload,
)

# 地理坐标降级模式：不再使用 aacgmv2；coords 保持 [M, 4]（Lat_geo, Lon_geo, Alt, Time）


def _unix_to_relhour(unix_times, start_unix):
    """将 Unix 时间戳转换为相对于训练起始时刻的小时数。"""
    return (unix_times - start_unix) / 3600.0


def query_model_grid(model, sw_manager, day_record, start_unix, device,
                     batch_size=2048, iri_peak_manager=None,
                     fy_nb_index=None, cosmic_nb_index=None,
                     allowed_profile_ids=None):
    """
    在 DayRecord 的 (alt, time) 网格上推理 FSIA-INR，返回 log10(Ne) 网格。

    Args:
        model:            FSIA_INR_Model（eval 模式）
        sw_manager:       SpaceWeatherManager
        day_record:       isr_loader 返回的 DayRecord dict，需含字段：
                            'alt_1d'      [n_alt]  float32  km
                            'ts_1d'       [n_time] float64  Unix s
                            'ne_2d'       [n_alt, n_time] float32  m⁻³  (NaN=缺测)
                            'lat'         scalar or None  (Jicamarca: 单一地理经纬度)
                            'lon'         scalar or None
                            'geo_lat_2d'  [n_alt, n_time] or None  (Poker Flat: 逐点地理纬度)
                            'geo_lon_2d'  [n_alt, n_time] or None
        start_unix:       训练起始 Unix 时间戳（由 start_date_str 解析）
        device:           torch.device
        batch_size:       推理分块大小
        iri_peak_manager: IRIPeakManager or None (IRI structural reference)
        fy_nb_index:      FYNeighborhoodIndex 或 None
        cosmic_nb_index:  COSMICNeighborhoodIndex 或 None

    Returns:
        analysis, background, raw_iri: three [n_alt, n_time] log10(Ne) grids.
        NaN where ne_2d is NaN or coordinates are invalid.
    """
    model.eval()

    alt_1d = day_record['alt_1d']    # [n_alt]
    ts_1d  = day_record['ts_1d']     # [n_time]
    ne_2d  = day_record['ne_2d']     # [n_alt, n_time]

    n_alt, n_time = ne_2d.shape

    # 构建坐标网格
    alts_2d  = np.tile(alt_1d[:, None], (1, n_time))          # [n_alt, n_time]
    rh_2d    = np.tile(
        _unix_to_relhour(ts_1d, start_unix)[None, :], (n_alt, 1)
    ).astype(np.float32)                                       # [n_alt, n_time]

    # 经纬度：标量（Jicamarca）或逐点数组（Poker Flat）
    geo_lat_2d = day_record.get('geo_lat_2d')
    geo_lon_2d = day_record.get('geo_lon_2d')
    if geo_lat_2d is not None and geo_lon_2d is not None:
        lat_2d = geo_lat_2d
        lon_2d = geo_lon_2d
    else:
        lat_scalar = day_record.get('lat', 0.0)
        lon_scalar = day_record.get('lon', 0.0)
        lat_2d = np.full((n_alt, n_time), lat_scalar, dtype=np.float32)
        lon_2d = np.full((n_alt, n_time), lon_scalar, dtype=np.float32)

    # 有效掩码（ne_2d 非 NaN 且坐标有效）
    valid_mask = (
        np.isfinite(ne_2d) &
        np.isfinite(lat_2d) &
        np.isfinite(lon_2d) &
        np.isfinite(alts_2d) &
        np.isfinite(rh_2d)
    )

    ne_pred_log10 = np.full((n_alt, n_time), np.nan, dtype=np.float32)
    ne_bkg_log10 = np.full((n_alt, n_time), np.nan, dtype=np.float32)
    ne_iri_log10 = np.full((n_alt, n_time), np.nan, dtype=np.float32)

    if not valid_mask.any():
        return ne_pred_log10, ne_bkg_log10, ne_iri_log10

    # 展平有效点
    lat_flat = lat_2d[valid_mask].astype(np.float32)
    lon_flat = lon_2d[valid_mask].astype(np.float32)
    alt_flat = alts_2d[valid_mask].astype(np.float32)
    rh_flat  = rh_2d[valid_mask].astype(np.float32)

    # 构建坐标数组：[M, 4]（地理坐标降级模式，不追加 AACGM 列）
    coords_np = np.column_stack([lat_flat, lon_flat, alt_flat, rh_flat])  # [M, 4]

    M = len(coords_np)

    pred_flat = np.empty(M, dtype=np.float32)
    bkg_flat = np.empty(M, dtype=np.float32)
    iri_flat = np.empty(M, dtype=np.float32)
    fy_covered = 0
    cosmic_covered = 0

    with torch.no_grad():
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            chunk = torch.from_numpy(coords_np[start:end]).to(device)

            rh_tensor = torch.from_numpy(rh_flat[start:end]).to(device)
            sw_seq = sw_manager.get_drivers_sequence(rh_tensor)  # [n, seq_len, 2]

            iri_peak = None
            if iri_peak_manager is not None:
                iri_peak = iri_peak_manager.get_iri_peak(chunk)

            model_kwargs = {'iri_peak': iri_peak}
            if fy_nb_index is not None:
                observations = query_observation_payload(
                    fy_nb_index, chunk, device,
                    allowed_profile_ids=(allowed_profile_ids or {}).get('FY'))
                fy_covered += int(
                    observation_query_coverage(observations).sum().item())
                model_kwargs['observations_fy'] = attach_observation_background(
                    observations, model, sw_manager, iri_peak_manager)
            if cosmic_nb_index is not None:
                observations = query_observation_payload(
                    cosmic_nb_index, chunk, device,
                    allowed_profile_ids=(allowed_profile_ids or {}).get('COSMIC'))
                cosmic_covered += int(
                    observation_query_coverage(observations).sum().item())
                model_kwargs['observations_cosmic'] = (
                    attach_observation_background(
                        observations, model, sw_manager, iri_peak_manager))

            ne_fused, _, _, _, extras = model(chunk, sw_seq, **model_kwargs)
            pred_flat[start:end] = ne_fused.reshape(-1).cpu().numpy()
            bkg_flat[start:end] = extras['ne_bkg'].reshape(-1).cpu().numpy()
            iri_flat[start:end] = extras['ne_iri'].reshape(-1).cpu().numpy()

    if fy_nb_index is not None or cosmic_nb_index is not None:
        print(f'    邻域覆盖: FY={fy_covered}/{M} ({fy_covered/M:.2%}), '
              f'COSMIC={cosmic_covered}/{M} ({cosmic_covered/M:.2%})')

    ne_pred_log10[valid_mask] = pred_flat
    ne_bkg_log10[valid_mask] = bkg_flat
    ne_iri_log10[valid_mask] = iri_flat
    return ne_pred_log10, ne_bkg_log10, ne_iri_log10


def extract_model_nmf2_hmf2(ne_log10_grid, alt_1d, f2_alt_min=150.0):
    """
    从 log10(Ne) 网格中逐时刻提取 NmF2 和 hmF2。

    策略：在 alt >= f2_alt_min 的范围内取最大值作为 F2 峰。

    Args:
        ne_log10_grid: [n_alt, n_time] float32 — log10(Ne)，NaN=缺测
        alt_1d:        [n_alt] float32 — 高度 (km)
        f2_alt_min:    F2 层搜索下限 (km)，排除 E/F1 层干扰

    Returns:
        nmf2_log10: [n_time] float32 — log10(NmF2)，无峰时为 NaN
        hmf2_km:    [n_time] float32 — hmF2 (km)，无峰时为 NaN
    """
    n_alt, n_time = ne_log10_grid.shape
    f2_mask = alt_1d >= f2_alt_min               # [n_alt] bool

    nmf2_log10 = np.full(n_time, np.nan, dtype=np.float32)
    hmf2_km    = np.full(n_time, np.nan, dtype=np.float32)

    grid_f2 = ne_log10_grid.copy()
    grid_f2[~f2_mask, :] = np.nan               # 屏蔽 F2 层以下

    for t in range(n_time):
        col = grid_f2[:, t]
        if not np.isfinite(col).any():
            continue
        idx = np.nanargmax(col)
        nmf2_log10[t] = col[idx]
        hmf2_km[t]    = alt_1d[idx]

    return nmf2_log10, hmf2_km
