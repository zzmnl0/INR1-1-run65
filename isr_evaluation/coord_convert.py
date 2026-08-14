"""
坐标转换模块

Poker Flat ISR 数据中的 cgm_lat / cgm_long 为 AACGM 地磁坐标，
需通过 aacgmv2 转换为地理坐标后才能与 MDIA 模型坐标系对齐。

依赖：aacgmv2（已在 poker_flat_to_nc.py 中使用，conda 环境已安装）
"""

import numpy as np
import datetime


def cgm_to_geo_batch(cgm_lats, cgm_lons, alts_km, unix_times):
    """
    将 AACGM 地磁坐标批量转换为地理坐标。

    使用观测时间段的中值时刻作为 IGRF 历元参考（IGRF 月内变化可忽略）。

    Args:
        cgm_lats:   [N] ndarray — AACGM 地磁纬度 (degrees)
        cgm_lons:   [N] ndarray — AACGM 地磁经度 (degrees)
        alts_km:    [N] ndarray — 高度 (km)，用于 IGRF 磁场计算
        unix_times: [N] ndarray — Unix 秒时间戳（用于确定参考历元）

    Returns:
        geo_lats: [N] ndarray — 地理纬度，无效处为 NaN
        geo_lons: [N] ndarray — 地理经度，无效处为 NaN
    """
    try:
        import aacgmv2
    except ImportError:
        raise ImportError(
            "aacgmv2 未安装。请在 conda 环境中执行: pip install aacgmv2"
        )

    geo_lats = np.full(len(cgm_lats), np.nan, dtype=np.float64)
    geo_lons = np.full(len(cgm_lons), np.nan, dtype=np.float64)

    valid = (np.isfinite(cgm_lats) & np.isfinite(cgm_lons)
             & np.isfinite(alts_km) & np.isfinite(unix_times))
    if not valid.any():
        return geo_lats, geo_lons

    # 以有效观测的中值时刻作为 IGRF 参考历元
    ref_unix = float(np.median(unix_times[valid]))
    ref_dt   = datetime.datetime.utcfromtimestamp(ref_unix)

    try:
        g_lat, g_lon, _ = aacgmv2.convert_latlon_arr(
            cgm_lats[valid], cgm_lons[valid], alts_km[valid],
            ref_dt, method_code='A2G'
        )
        geo_lats[valid] = g_lat
        geo_lons[valid] = g_lon
    except Exception as e:
        print(f'  [coord_convert] aacgmv2 转换失败: {e}')

    return geo_lats, geo_lons


def convert_day_record_cgm(day_record):
    """
    Convert AACGM coordinates for a diagnostic cross-check only.

    P0-A uses radar line-of-sight WGS84 coordinates as the model coordinates.
    AACGM inverse conversion is retained to expose disagreements, never to replace
    the LOS coordinates or to determine model-query coverage.

    Args:
        day_record: dict（由 isr_loader.load_poker_flat 返回，含 cgm_lat_2d 等字段）

    Returns:
        同一 dict，添加 'aacgm_geo_lat_2d', 'aacgm_geo_lon_2d' and a small
        geometry diagnostic.  It never overwrites ``geo_lat_2d``/``geo_lon_2d``.
        若原始 cgm 字段为 None 则跳过。
    """
    cgm_lat_2d = day_record.get('cgm_lat_2d')
    cgm_lon_2d = day_record.get('cgm_lon_2d')
    ts_1d      = day_record.get('ts_1d')
    alt_1d     = day_record.get('alt_1d')

    if cgm_lat_2d is None or cgm_lon_2d is None:
        day_record['aacgm_geo_lat_2d'] = None
        day_record['aacgm_geo_lon_2d'] = None
        day_record.setdefault('geometry_qc', {})['aacgm_crosscheck'] = {
            'status': 'not_available'}
        return day_record

    n_alt, n_time = cgm_lat_2d.shape

    # 构建对应的 alts 和 times 2D 数组
    alts_2d  = np.tile(alt_1d[:, None], (1, n_time))   # [n_alt, n_time]
    times_2d = np.tile(ts_1d[None, :],  (n_alt, 1))    # [n_alt, n_time]

    geo_lat, geo_lon = cgm_to_geo_batch(
        cgm_lat_2d.flatten(),
        cgm_lon_2d.flatten(),
        alts_2d.flatten(),
        times_2d.flatten()
    )

    aacgm_lat = geo_lat.reshape(n_alt, n_time).astype(np.float32)
    aacgm_lon = geo_lon.reshape(n_alt, n_time).astype(np.float32)
    day_record['aacgm_geo_lat_2d'] = aacgm_lat
    day_record['aacgm_geo_lon_2d'] = aacgm_lon
    primary_lat = day_record.get('geo_lat_2d')
    primary_lon = day_record.get('geo_lon_2d')
    diagnostic = {'status': 'computed'}
    if primary_lat is not None and primary_lon is not None:
        valid = (np.isfinite(primary_lat) & np.isfinite(primary_lon)
                 & np.isfinite(aacgm_lat) & np.isfinite(aacgm_lon))
        if valid.any():
            # Small-angle surface distance is sufficient for a cross-check; it is
            # explicitly not used in the model coordinate path.
            dlat = np.deg2rad(aacgm_lat[valid] - primary_lat[valid])
            dlon = np.deg2rad(aacgm_lon[valid] - primary_lon[valid])
            mean_lat = np.deg2rad(0.5 * (aacgm_lat[valid] + primary_lat[valid]))
            diagnostic.update({
                'n_compared': int(valid.sum()),
                'median_horizontal_difference_km': float(np.median(
                    6371.0 * np.hypot(dlat, np.cos(mean_lat) * dlon))),
            })
        else:
            diagnostic['status'] = 'insufficient_data'
    else:
        diagnostic['status'] = 'primary_los_coordinate_missing'
    day_record.setdefault('geometry_qc', {})['aacgm_crosscheck'] = diagnostic
    return day_record
