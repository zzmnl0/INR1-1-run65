"""
ISR 数据加载模块

支持两个站点：
  - Jicamarca  (~12°S，低纬磁赤道)   — 单 beam，固定站址，HDF5 Array Layout 格式
  - Poker Flat (~65°N，高纬极光带)   — 多 beam，cgm 地磁坐标，HDF5 Array Layout 格式

每个站点返回 DayRecord 列表（按 UTC 自然日分组），每个 DayRecord 包含：
  - 2D 网格数据 (alt × time)：用于时间-高度对比图
  - 平展有效观测 1D 数组：用于逐点统计指标
"""

import h5py
import numpy as np
import pandas as pd
import glob
import os
import traceback
import hashlib


_WGS84_A_KM = 6378.137
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
_WGS84_B_KM = _WGS84_A_KM * (1.0 - _WGS84_F)
_WGS84_EP2 = ((_WGS84_A_KM ** 2 - _WGS84_B_KM ** 2)
              / (_WGS84_B_KM ** 2))
_POKER_RANGE_TO_KM = 1.0e-3


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _decode_hdf_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


def _read_experiment_geometry(hdf_file):
    """Read mandatory Poker Flat station geometry from Madrigal metadata."""
    try:
        parameters = hdf_file['Metadata/Experiment Parameters'][:]
    except KeyError as exc:
        raise ValueError('Poker Flat HDF lacks Metadata/Experiment Parameters') from exc
    metadata = {
        _decode_hdf_text(row['name']).strip().lower(): _decode_hdf_text(row['value']).strip()
        for row in parameters
    }
    required = {
        'instrument latitude': 'latitude_deg',
        'instrument longitude': 'longitude_deg',
        'instrument altitude': 'altitude_km',
    }
    result = {}
    for source_key, output_key in required.items():
        if source_key not in metadata:
            raise ValueError(f'Poker Flat HDF lacks {source_key!r} metadata')
        try:
            result[output_key] = float(metadata[source_key])
        except ValueError as exc:
            raise ValueError(f'invalid Poker Flat {source_key!r} metadata') from exc
    if not (-90.0 <= result['latitude_deg'] <= 90.0):
        raise ValueError('Poker Flat station latitude is outside physical bounds')
    if not (-1.0 <= result['altitude_km'] <= 20.0):
        raise ValueError('Poker Flat station altitude is outside physical bounds')
    result['longitude_deg'] = ((result['longitude_deg'] + 180.0) % 360.0) - 180.0
    return result


def _constant_beam_parameter(beam, name, *, integer=False):
    values = np.asarray(beam[f'1D Parameters/{name}'][:]).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f'Poker Flat beam {name} is missing or nonfinite')
    reference = values[0]
    tolerance = 0.0 if integer else 1e-6
    if not np.all(np.isclose(values, reference, rtol=0.0, atol=tolerance)):
        raise ValueError(
            f'Poker Flat beam {name} varies within a segment; '
            'the P0-A LOS geometry contract rejects time-varying beam angles')
    return int(reference) if integer else float(reference)


def _validate_beam_geometry(beam, station_geometry):
    """Validate the declared meter-scale range and immutable beam direction."""
    azimuth_deg = _constant_beam_parameter(beam, 'azm')
    elevation_deg = _constant_beam_parameter(beam, 'elm')
    beam_id = _constant_beam_parameter(beam, 'beamid', integer=True)
    range_m = np.asarray(beam['range'][:], dtype=np.float64).reshape(-1)
    if (range_m.size == 0 or not np.all(np.isfinite(range_m))
            or np.any(range_m < 1.0e4) or np.any(range_m > 2.0e6)):
        raise ValueError(
            'Poker Flat range metadata is incompatible with the explicit '
            'meter-to-kilometre scale contract (1e-3)')
    # Madrigal stores valid clockwise azimuths in either [-180, 180] or
    # [0, 360); normalize the representation rather than rejecting a direction.
    azimuth_deg = azimuth_deg % 360.0
    if not (0.0 <= azimuth_deg < 360.0 and 0.0 <= elevation_deg <= 90.0):
        raise ValueError('Poker Flat beam azimuth/elevation is outside physical bounds')
    if not np.isfinite(station_geometry['latitude_deg']):
        raise ValueError('Poker Flat station geometry is nonfinite')
    return {
        'beam_id': beam_id,
        'azimuth_deg': azimuth_deg,
        'elevation_deg': elevation_deg,
        'slant_range_m': range_m,
        'range_scale_to_km': _POKER_RANGE_TO_KM,
        'range_unit_semantics': 'HDF_range_treated_as_meters_explicit_1e-3_v1',
        'range_min_m': float(range_m.min()),
        'range_max_m': float(range_m.max()),
    }


def _geodetic_to_ecef(lat_deg, lon_deg, altitude_km):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    radius = _WGS84_A_KM / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    x = (radius + altitude_km) * np.cos(lat) * np.cos(lon)
    y = (radius + altitude_km) * np.cos(lat) * np.sin(lon)
    z = (radius * (1.0 - _WGS84_E2) + altitude_km) * sin_lat
    return x, y, z


def _ecef_to_geodetic(x, y, z):
    longitude = np.arctan2(y, x)
    planar = np.hypot(x, y)
    theta = np.arctan2(z * _WGS84_A_KM, planar * _WGS84_B_KM)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    latitude = np.arctan2(
        z + _WGS84_EP2 * _WGS84_B_KM * sin_theta ** 3,
        planar - _WGS84_E2 * _WGS84_A_KM * cos_theta ** 3,
    )
    radius = _WGS84_A_KM / np.sqrt(1.0 - _WGS84_E2 * np.sin(latitude) ** 2)
    altitude = planar / np.cos(latitude) - radius
    return np.rad2deg(latitude), ((np.rad2deg(longitude) + 180.0) % 360.0) - 180.0, altitude


def _wgs84_los_to_geodetic(station_lat_deg, station_lon_deg, station_alt_km,
                           slant_range_m, azimuth_deg, elevation_deg):
    """Transform an ENU line-of-sight ray to WGS84 geodetic coordinates."""
    ranges_km = np.asarray(slant_range_m, dtype=np.float64) * _POKER_RANGE_TO_KM
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    east = ranges_km * np.cos(elevation) * np.sin(azimuth)
    north = ranges_km * np.cos(elevation) * np.cos(azimuth)
    up = ranges_km * np.sin(elevation)
    lat = np.deg2rad(float(station_lat_deg))
    lon = np.deg2rad(float(station_lon_deg))
    station_x, station_y, station_z = _geodetic_to_ecef(
        float(station_lat_deg), float(station_lon_deg), float(station_alt_km))
    dx = -np.sin(lon) * east - np.sin(lat) * np.cos(lon) * north + np.cos(lat) * np.cos(lon) * up
    dy = np.cos(lon) * east - np.sin(lat) * np.sin(lon) * north + np.cos(lat) * np.sin(lon) * up
    dz = np.cos(lat) * north + np.sin(lat) * up
    return _ecef_to_geodetic(station_x + dx, station_y + dy, station_z + dz)


# ======================== 内部辅助 ========================

def _qmask(ne, dne, err_ratio_max):
    """质量掩码：ne > 0, dne > 0, dne/ne < threshold, 有限值。"""
    ratio = np.where(ne > 0, dne / ne, np.inf)
    return (ne > 0) & np.isfinite(ne) & (dne > 0) & (dne < ne) & (ratio < err_ratio_max)


def _merge_to_grid(alt_segs, ts_segs, data_segs_dict, tol=1.0):
    """
    将多个 (alt, time, data...) 段合并为对齐的 2D 网格。

    Args:
        alt_segs:       list of [n_alt_i] arrays
        ts_segs:        list of [n_time_i] Unix 秒数组
        data_segs_dict: {name: list of [n_alt_i, n_time_i] arrays}
        tol:            高度匹配容差 (km)

    Returns:
        alt_1d:  [n_alt] union 高度轴
        ts_1d:   [n_time] 排序时间轴
        grids:   {name: [n_alt, n_time] ndarray}
    """
    # Union 高度轴（1 km 容差合并近似相同高度）
    all_alts = np.sort(np.unique(np.concatenate(alt_segs).astype(np.float32)))
    union = []
    for a in all_alts:
        if not union or (a - union[-1]) > tol:
            union.append(float(a))
    alt_1d = np.array(union, dtype=np.float32)

    # 合并时间轴并排序
    ts_all = np.concatenate(ts_segs)
    order  = np.argsort(ts_all, kind='stable')
    ts_1d  = ts_all[order]

    n_alt  = len(alt_1d)
    n_time = len(ts_1d)

    grids = {}
    for name, data_list in data_segs_dict.items():
        grid = np.full((n_alt, n_time), np.nan, dtype=np.float64)
        col  = 0
        for seg_i, alts_seg in enumerate(alt_segs):
            nt       = len(ts_segs[seg_i])
            seg_data = data_list[seg_i]       # [n_alt_i, nt]
            for row_i, a in enumerate(alts_seg):
                best = int(np.argmin(np.abs(alt_1d - a)))
                if abs(alt_1d[best] - a) <= tol:
                    grid[best, col:col + nt] = seg_data[row_i, :]
            col += nt
        grids[name] = grid[:, order]

    return alt_1d, ts_1d, grids


def _build_record(date_str, station, lat, lon,
                  alt_1d, ts_1d, ne_2d, dne_2d,
                  cgm_lat_2d=None, cgm_lon_2d=None,
                  geo_lat_2d=None, geo_lon_2d=None,
                  plot_segs=None, source_file_identity=None,
                  geometry_qc=None):
    """
    构建 DayRecord dict，同时生成平展有效观测数组。
    ne_2d / dne_2d 已完成质量过滤（无效处为 NaN）。

    plot_segs: list of {'alt_km', 'ts', 'ne'} — 原始分段，供绘图时分段 pcolormesh 使用，
               避免不同高度门合并后产生横条纹。仅用于可视化，不用于指标计算。
    """
    i_alt, i_t = np.where(np.isfinite(ne_2d))
    n_alt, n_time = ne_2d.shape
    alts_2d = np.tile(np.asarray(alt_1d)[:, None], (1, n_time))
    times_2d = np.tile(np.asarray(ts_1d)[None, :], (n_alt, 1))
    if geo_lat_2d is not None and geo_lon_2d is not None:
        coordinate_mask = (
            np.isfinite(geo_lat_2d) & np.isfinite(geo_lon_2d)
            & np.isfinite(alts_2d) & np.isfinite(times_2d))
    else:
        coordinate_mask = (
            np.isfinite(float(lat)) & np.isfinite(float(lon))
            & np.isfinite(alts_2d) & np.isfinite(times_2d))
    return {
        'date_str': date_str,
        'station': station,
        'lat':     lat,
        'lon':     lon,
        'alt_1d':  alt_1d,
        'ts_1d':   ts_1d,
        'ne_2d':   ne_2d,
        'dne_2d':  dne_2d,
        'cgm_lat_2d': cgm_lat_2d,
        'cgm_lon_2d': cgm_lon_2d,
        'geo_lat_2d': geo_lat_2d,
        'geo_lon_2d': geo_lon_2d,
        'coordinate_mask': coordinate_mask,
        'observation_mask': np.isfinite(ne_2d),
        'plot_segs': plot_segs or [],  # [{'alt_km', 'ts', 'ne'}, ...]
        'ne_flat':     ne_2d[i_alt, i_t],
        'dne_flat':    dne_2d[i_alt, i_t],
        'alt_flat':    alt_1d[i_alt],
        'ts_flat':     ts_1d[i_t],
        'source_file_identity': source_file_identity or [],
        'geometry_qc': geometry_qc or {},
    }


# ======================== Jicamarca ========================

def load_jicamarca(data_dir, start_unix, end_unix,
                   alt_min=120.0, alt_max=500.0, err_ratio_max=0.5):
    """
    读取 Jicamarca IS Radar HDF5 数据，按 UTC 日期分组，返回 DayRecord 列表。

    数据格式：HDF5 Array Layout
      gdalt       : [n_alt] km
      timestamps  : [n_time] Unix 秒
      ne / dne    : [n_alt, n_time] m-3
      gdlatr/gdlonr: 固定站址地理坐标

    Args:
        data_dir:      HDF5 文件目录（含 *.hdf5）
        start_unix:    MDIA 训练起始 Unix 时间戳
        end_unix:      MDIA 训练结束 Unix 时间戳
        alt_min/max:   高度过滤范围 (km)
        err_ratio_max: 最大 dne/ne 比值（误差棒过滤）

    Returns:
        list of DayRecord dicts，按日期排序
    """
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
    if not h5_files:
        print(f'  [Jicamarca] 未找到 .hdf5 文件: {data_dir}')
        return []

    print(f'  [Jicamarca] 找到 {len(h5_files)} 个 HDF5 文件')

    day_groups = {}   # date_str → list of segment dicts
    lat_val = lon_val = None

    for fp in h5_files:
        try:
            with h5py.File(fp, 'r') as f:
                al     = f['Data/Array Layout']
                gdalt  = al['gdalt'][:]
                ts     = al['timestamps'][:]
                ne     = al['2D Parameters/ne'][:]
                dne    = al['2D Parameters/dne'][:]
                gdlatr = float(al['1D Parameters/gdlatr'][0])
                gdlonr = float(al['1D Parameters/gdlonr'][0])
                file_identity = {
                    'path': os.path.abspath(fp),
                    'sha256': _sha256(fp),
                    'size_bytes': int(os.path.getsize(fp)),
                }

            if lat_val is None:
                lat_val, lon_val = gdlatr, gdlonr

            # 过滤 gdalt 无效行
            valid_alt = np.isfinite(gdalt)
            gdalt = gdalt[valid_alt]
            ne    = ne[valid_alt, :]
            dne   = dne[valid_alt, :]

            # 按 UTC 自然日分组
            dates_utc = pd.to_datetime(ts, unit='s', utc=True).normalize()
            for day in dates_utc.unique():
                date_str = day.strftime('%Y%m%d')
                mask     = np.asarray(dates_utc == day)
                day_groups.setdefault(date_str, []).append({
                    'gdalt': gdalt.astype(np.float32),
                    'ts':    ts[mask],
                    'ne':    ne[:, mask],
                    'dne':   dne[:, mask],
                    'file_identity': file_identity,
                })
        except Exception as e:
            print(f'  [Jicamarca] 读取失败 {os.path.basename(fp)}: {e}')
            traceback.print_exc()

    records = []
    for date_str in sorted(day_groups.keys()):
        segs = day_groups[date_str]

        # 粗略时间范围过滤
        all_ts = np.concatenate([s['ts'] for s in segs])
        if all_ts.max() < start_unix or all_ts.min() > end_unix:
            continue

        alt_segs = [s['gdalt']  for s in segs]
        ts_segs  = [s['ts']     for s in segs]
        ne_segs  = [s['ne']     for s in segs]
        dne_segs = [s['dne']    for s in segs]

        # 保留原始分段供绘图用（各段独立 pcolormesh，消除横条纹）
        plot_segs = []
        for s in segs:
            alt_s = s['gdalt']
            ts_s  = s['ts']
            ne_s  = s['ne'].copy().astype(np.float64)
            qm    = _qmask(s['ne'], s['dne'], err_ratio_max)
            ne_s[~qm] = np.nan
            amask = (alt_s >= alt_min) & (alt_s <= alt_max)
            tmask = (ts_s  >= start_unix) & (ts_s <= end_unix)
            if not amask.any() or not tmask.any():
                continue
            ne_seg = ne_s[amask][:, tmask]
            if np.any(np.isfinite(ne_seg)):
                plot_segs.append({'alt_km': alt_s[amask].astype(np.float32),
                                  'ts':     ts_s[tmask],
                                  'ne':     ne_seg.astype(np.float32)})

        alt_1d, ts_1d, grids = _merge_to_grid(
            alt_segs, ts_segs, {'ne': ne_segs, 'dne': dne_segs})

        ne_2d  = grids['ne']
        dne_2d = grids['dne']

        # 质量过滤
        qmask = _qmask(ne_2d, dne_2d, err_ratio_max)
        ne_2d[~qmask]  = np.nan
        dne_2d[~qmask] = np.nan

        # 高度过滤
        alt_mask = (alt_1d >= alt_min) & (alt_1d <= alt_max)
        alt_1d = alt_1d[alt_mask]
        ne_2d  = ne_2d[alt_mask, :]
        dne_2d = dne_2d[alt_mask, :]

        # 时间范围过滤
        ts_mask = (ts_1d >= start_unix) & (ts_1d <= end_unix)
        ts_1d  = ts_1d[ts_mask]
        ne_2d  = ne_2d[:, ts_mask]
        dne_2d = dne_2d[:, ts_mask]

        if ts_1d.size == 0 or alt_1d.size == 0:
            continue
        if not np.any(np.isfinite(ne_2d)):
            continue

        records.append(_build_record(
            date_str, 'Jicamarca', lat_val, lon_val,
            alt_1d, ts_1d, ne_2d, dne_2d,
            plot_segs=plot_segs,
            source_file_identity=[s['file_identity'] for s in segs],
            geometry_qc={'semantics': 'fixed_station_geodetic_coordinate_v1',
                         'station_latitude_deg': float(lat_val),
                         'station_longitude_deg': float(lon_val)}))

    print(f'  [Jicamarca] 有效天数: {len(records)}')
    return records


# ======================== Poker Flat ========================

def load_poker_flat(data_dir, start_unix, end_unix,
                    alt_min=120.0, alt_max=500.0, err_ratio_max=0.5,
                    beam_select='max_elm'):
    """
    读取 Poker Flat IS Radar HDF5 数据，按 UTC 日期分组，返回 DayRecord 列表。

    数据格式：HDF5 多 beam Array Layout
      range       : [n_alt] meters（除以 1000 得 km，近似高度）
      timestamps  : [n_time] Unix 秒
      ne / dne    : [n_alt, n_time] m-3
      cgm_lat / cgm_long : [n_alt, n_time] AACGM 地磁坐标（需后续转换）

    Args:
        beam_select: 'max_elm' 选最大仰角 beam（最接近垂直，默认）
                     'all' 合并所有 beam
                     int   按 beamid 指定

    Returns:
        list of DayRecord dicts（含 cgm_lat_2d / cgm_lon_2d，待坐标转换）
    """
    h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
    if not h5_files:
        print(f'  [Poker Flat] 未找到 .h5 文件: {data_dir}')
        return []

    print(f'  [Poker Flat] 找到 {len(h5_files)} 个 HDF5 文件')

    day_groups = {}   # date_str → list of segment dicts

    for fp in h5_files:
        try:
            with h5py.File(fp, 'r') as f:
                layout = f['Data/Array Layout']
                station_geometry = _read_experiment_geometry(f)
                file_identity = {
                    'path': os.path.abspath(fp),
                    'sha256': _sha256(fp),
                    'size_bytes': int(os.path.getsize(fp)),
                }
                beams  = []
                for bname in layout.keys():
                    b = layout[bname]
                    beam_geometry = _validate_beam_geometry(b, station_geometry)
                    geo_lat_1d, geo_lon_1d, geo_alt_1d = _wgs84_los_to_geodetic(
                        station_geometry['latitude_deg'],
                        station_geometry['longitude_deg'],
                        station_geometry['altitude_km'],
                        beam_geometry['slant_range_m'],
                        beam_geometry['azimuth_deg'],
                        beam_geometry['elevation_deg'],
                    )
                    beams.append({
                        'azm':      beam_geometry['azimuth_deg'],
                        'elm':      beam_geometry['elevation_deg'],
                        'beamid':   beam_geometry['beam_id'],
                        'range_m':  beam_geometry['slant_range_m'],
                        'alt_km':   geo_alt_1d.astype(np.float32),
                        'geo_lat_1d': geo_lat_1d.astype(np.float32),
                        'geo_lon_1d': geo_lon_1d.astype(np.float32),
                        'geometry_qc': {
                            **station_geometry,
                            **{key: value for key, value in beam_geometry.items()
                               if key != 'slant_range_m'},
                            'legacy_range_as_height_delta_km_median': float(np.median(
                                geo_alt_1d - beam_geometry['slant_range_m']
                                * _POKER_RANGE_TO_KM)),
                        },
                        'ts':       b['timestamps'][:],
                        'ne':       b['2D Parameters/ne'][:],
                        'dne':      b['2D Parameters/dne'][:],
                        'cgm_lat':  b['2D Parameters/cgm_lat'][:],
                        'cgm_lon':  b['2D Parameters/cgm_long'][:],
                        'file_identity': file_identity,
                    })

            if not beams:
                continue

            # ---- beam 选择 ----
            if beam_select == 'max_elm':
                beams = [max(beams, key=lambda b: b['elm'])]
            elif beam_select == 'all':
                pass
            elif isinstance(beam_select, int):
                beams = [b for b in beams if b['beamid'] == beam_select]
                if not beams:
                    continue

            for b in beams:
                # ISR density quality is distinct from model-coordinate validity.
                # AACGM is retained for a diagnostic cross-check only and must not
                # erase the LOS geodetic coordinate when it is unavailable.
                qmask = _qmask(b['ne'], b['dne'], err_ratio_max)

                ne  = b['ne'].copy()
                dne = b['dne'].copy()
                cgm_lat = b['cgm_lat'].copy()
                cgm_lon = b['cgm_lon'].copy()
                ne[~qmask]      = np.nan
                dne[~qmask]     = np.nan

                ts = b['ts']
                alt_km = b['alt_km']
                geo_lat = np.tile(b['geo_lat_1d'][:, None], (1, len(ts)))
                geo_lon = np.tile(b['geo_lon_1d'][:, None], (1, len(ts)))

                dates_utc = pd.to_datetime(ts, unit='s', utc=True).normalize()
                for day in dates_utc.unique():
                    date_str = day.strftime('%Y%m%d')
                    mask     = np.asarray(dates_utc == day)
                    day_groups.setdefault(date_str, []).append({
                        'alt_km':   alt_km,
                        'ts':       ts[mask],
                        'ne':       ne[:, mask],
                        'dne':      dne[:, mask],
                        'cgm_lat':  cgm_lat[:, mask],
                        'cgm_lon':  cgm_lon[:, mask],
                        'geo_lat':  geo_lat[:, mask],
                        'geo_lon':  geo_lon[:, mask],
                        'elm':      b['elm'],
                        'azm':      b['azm'],
                        'beamid':   b['beamid'],
                        'geometry_qc': b['geometry_qc'],
                        'file_identity': b['file_identity'],
                    })

        except Exception as e:
            print(f'  [Poker Flat] 读取失败 {os.path.basename(fp)}: {e}')
            traceback.print_exc()

    records = []
    for date_str in sorted(day_groups.keys()):
        segs = day_groups[date_str]

        all_ts = np.concatenate([s['ts'] for s in segs])
        if all_ts.max() < start_unix or all_ts.min() > end_unix:
            continue

        alt_segs    = [s['alt_km']   for s in segs]
        ts_segs     = [s['ts']       for s in segs]
        ne_segs     = [s['ne']       for s in segs]
        dne_segs    = [s['dne']      for s in segs]
        cgmlat_segs = [s['cgm_lat']  for s in segs]
        cgmlon_segs = [s['cgm_lon']  for s in segs]
        geolat_segs = [s['geo_lat']  for s in segs]
        geolon_segs = [s['geo_lon']  for s in segs]

        # 保留原始分段供绘图用
        plot_segs = []
        for s in segs:
            alt_s = s['alt_km']
            ts_s  = s['ts']
            ne_s  = s['ne'].copy().astype(np.float64)
            # ne/dne 在入组前已质量过滤（NaN化），此处直接用
            amask = (alt_s >= alt_min) & (alt_s <= alt_max)
            tmask = (ts_s  >= start_unix) & (ts_s <= end_unix)
            if not amask.any() or not tmask.any():
                continue
            ne_seg = ne_s[amask][:, tmask]
            if np.any(np.isfinite(ne_seg)):
                plot_segs.append({'alt_km': alt_s[amask].astype(np.float32),
                                  'ts':     ts_s[tmask],
                                  'ne':     ne_seg.astype(np.float32)})

        alt_1d, ts_1d, grids = _merge_to_grid(
            alt_segs, ts_segs,
            {'ne': ne_segs, 'dne': dne_segs,
             'cgm_lat': cgmlat_segs, 'cgm_lon': cgmlon_segs,
             'geo_lat': geolat_segs, 'geo_lon': geolon_segs})

        ne_2d      = grids['ne']
        dne_2d     = grids['dne']
        cgm_lat_2d = grids['cgm_lat'].astype(np.float32)
        cgm_lon_2d = grids['cgm_lon'].astype(np.float32)
        geo_lat_2d = grids['geo_lat'].astype(np.float32)
        geo_lon_2d = grids['geo_lon'].astype(np.float32)

        # 高度过滤
        alt_mask = (alt_1d >= alt_min) & (alt_1d <= alt_max)
        alt_1d     = alt_1d[alt_mask]
        ne_2d      = ne_2d[alt_mask, :]
        dne_2d     = dne_2d[alt_mask, :]
        cgm_lat_2d = cgm_lat_2d[alt_mask, :]
        cgm_lon_2d = cgm_lon_2d[alt_mask, :]
        geo_lat_2d = geo_lat_2d[alt_mask, :]
        geo_lon_2d = geo_lon_2d[alt_mask, :]

        # 时间范围过滤
        ts_mask = (ts_1d >= start_unix) & (ts_1d <= end_unix)
        ts_1d      = ts_1d[ts_mask]
        ne_2d      = ne_2d[:, ts_mask]
        dne_2d     = dne_2d[:, ts_mask]
        cgm_lat_2d = cgm_lat_2d[:, ts_mask]
        cgm_lon_2d = cgm_lon_2d[:, ts_mask]
        geo_lat_2d = geo_lat_2d[:, ts_mask]
        geo_lon_2d = geo_lon_2d[:, ts_mask]

        if ts_1d.size == 0 or alt_1d.size == 0:
            continue
        if not np.any(np.isfinite(ne_2d)):
            continue

        rec = _build_record(
            date_str, 'Poker Flat', None, None,
            alt_1d, ts_1d, ne_2d, dne_2d,
            cgm_lat_2d=cgm_lat_2d, cgm_lon_2d=cgm_lon_2d,
            geo_lat_2d=geo_lat_2d, geo_lon_2d=geo_lon_2d,
            plot_segs=plot_segs,
            source_file_identity=[s['file_identity'] for s in segs],
            geometry_qc={
                'semantics': 'wgs84_enu_los_to_ecef_to_geodetic_v1',
                'selected_beams': [s['geometry_qc'] for s in segs],
                'legacy_height_delta_km_median': float(np.median([
                    s['geometry_qc']['legacy_range_as_height_delta_km_median']
                    for s in segs])),
            })

        # 记录所选 beam 仰角（用于标题）
        rec['elm'] = float(segs[0]['elm'])
        rec['azm'] = float(segs[0]['azm'])
        rec['beamid'] = int(segs[0]['beamid'])
        records.append(rec)

    print(f'  [Poker Flat] 有效天数: {len(records)}  '
          f'(beam: {beam_select})')
    return records
