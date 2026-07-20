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


def _resample_to_std_alt(alt_1d, grids_dict, step_km=5.0):
    """
    将任意非均匀高度轴上的多个 2D 网格，插值到统一步长的标准高度轴。
    对每个时刻独立插值，只在有效数据范围内插值（不外推）。

    Args:
        alt_1d:      [n_alt] float32 — 原始高度轴 (km)，可非均匀
        grids_dict:  {name: [n_alt, n_time] ndarray} — 待重采样的数组
        step_km:     标准高度步长 (km)，默认 5 km

    Returns:
        alt_new:     [n_alt_new] float32
        new_grids:   {name: [n_alt_new, n_time] ndarray}  dtype 与输入一致
    """
    a_min = float(np.ceil(alt_1d.min() / step_km) * step_km)
    a_max = float(np.floor(alt_1d.max() / step_km) * step_km)
    if a_max <= a_min:
        return alt_1d, grids_dict

    alt_new  = np.arange(a_min, a_max + step_km * 0.1, step_km, dtype=np.float32)
    n_time   = next(iter(grids_dict.values())).shape[1]
    new_grids = {
        k: np.full((len(alt_new), n_time), np.nan, dtype=v.dtype)
        for k, v in grids_dict.items()
    }

    for t in range(n_time):
        # 以 ne（第一个键）确定哪些高度行有效
        ne_key = list(grids_dict.keys())[0]
        valid = np.isfinite(grids_dict[ne_key][:, t])
        if valid.sum() < 2:
            continue
        av = alt_1d[valid].astype(np.float64)
        for k, grid in grids_dict.items():
            fp = grid[:, t][valid].astype(np.float64)
            # np.interp 不外推，超出范围返回端点值；手动置 NaN
            interped = np.interp(alt_new.astype(np.float64), av, fp)
            out_of_range = (alt_new < av[0]) | (alt_new > av[-1])
            interped[out_of_range] = np.nan
            new_grids[k][:, t] = interped.astype(grid.dtype)

    return alt_new, new_grids


def _build_record(date_str, station, lat, lon,
                  alt_1d, ts_1d, ne_2d, dne_2d,
                  cgm_lat_2d=None, cgm_lon_2d=None,
                  plot_segs=None):
    """
    构建 DayRecord dict，同时生成平展有效观测数组。
    ne_2d / dne_2d 已完成质量过滤（无效处为 NaN）。

    plot_segs: list of {'alt_km', 'ts', 'ne'} — 原始分段，供绘图时分段 pcolormesh 使用，
               避免不同高度门合并后产生横条纹。仅用于可视化，不用于指标计算。
    """
    i_alt, i_t = np.where(np.isfinite(ne_2d))
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
        'geo_lat_2d': None,
        'geo_lon_2d': None,
        'plot_segs': plot_segs or [],  # [{'alt_km', 'ts', 'ne'}, ...]
        'ne_flat':     ne_2d[i_alt, i_t],
        'dne_flat':    dne_2d[i_alt, i_t],
        'alt_flat':    alt_1d[i_alt],
        'ts_flat':     ts_1d[i_t],
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
            plot_segs=plot_segs))

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
                beams  = []
                for bname in layout.keys():
                    b = layout[bname]
                    beams.append({
                        'azm':      float(b['1D Parameters/azm'][0]),
                        'elm':      float(b['1D Parameters/elm'][0]),
                        'beamid':   int(b['1D Parameters/beamid'][0]),
                        'range_km': b['range'][:] / 1000.0,
                        'ts':       b['timestamps'][:],
                        'ne':       b['2D Parameters/ne'][:],
                        'dne':      b['2D Parameters/dne'][:],
                        'cgm_lat':  b['2D Parameters/cgm_lat'][:],
                        'cgm_lon':  b['2D Parameters/cgm_long'][:],
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
                # 质量掩码（含 cgm 有效性）
                qmask = (_qmask(b['ne'], b['dne'], err_ratio_max)
                         & np.isfinite(b['cgm_lat']) & np.isfinite(b['cgm_lon']))

                ne  = b['ne'].copy()
                dne = b['dne'].copy()
                cgm_lat = b['cgm_lat'].copy()
                cgm_lon = b['cgm_lon'].copy()
                ne[~qmask]      = np.nan
                dne[~qmask]     = np.nan
                cgm_lat[~qmask] = np.nan
                cgm_lon[~qmask] = np.nan

                ts       = b['ts']
                range_km = b['range_km'].astype(np.float32)

                dates_utc = pd.to_datetime(ts, unit='s', utc=True).normalize()
                for day in dates_utc.unique():
                    date_str = day.strftime('%Y%m%d')
                    mask     = np.asarray(dates_utc == day)
                    day_groups.setdefault(date_str, []).append({
                        'range_km': range_km,
                        'ts':       ts[mask],
                        'ne':       ne[:, mask],
                        'dne':      dne[:, mask],
                        'cgm_lat':  cgm_lat[:, mask],
                        'cgm_lon':  cgm_lon[:, mask],
                        'elm':      b['elm'],
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

        alt_segs    = [s['range_km'] for s in segs]
        ts_segs     = [s['ts']       for s in segs]
        ne_segs     = [s['ne']       for s in segs]
        dne_segs    = [s['dne']      for s in segs]
        cgmlat_segs = [s['cgm_lat']  for s in segs]
        cgmlon_segs = [s['cgm_lon']  for s in segs]

        # 保留原始分段供绘图用
        plot_segs = []
        for s in segs:
            alt_s = s['range_km']
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
             'cgm_lat': cgmlat_segs, 'cgm_lon': cgmlon_segs})

        ne_2d      = grids['ne']
        dne_2d     = grids['dne']
        cgm_lat_2d = grids['cgm_lat'].astype(np.float32)
        cgm_lon_2d = grids['cgm_lon'].astype(np.float32)

        # 高度过滤
        alt_mask = (alt_1d >= alt_min) & (alt_1d <= alt_max)
        alt_1d     = alt_1d[alt_mask]
        ne_2d      = ne_2d[alt_mask, :]
        dne_2d     = dne_2d[alt_mask, :]
        cgm_lat_2d = cgm_lat_2d[alt_mask, :]
        cgm_lon_2d = cgm_lon_2d[alt_mask, :]

        # 时间范围过滤
        ts_mask = (ts_1d >= start_unix) & (ts_1d <= end_unix)
        ts_1d      = ts_1d[ts_mask]
        ne_2d      = ne_2d[:, ts_mask]
        dne_2d     = dne_2d[:, ts_mask]
        cgm_lat_2d = cgm_lat_2d[:, ts_mask]
        cgm_lon_2d = cgm_lon_2d[:, ts_mask]

        if ts_1d.size == 0 or alt_1d.size == 0:
            continue
        if not np.any(np.isfinite(ne_2d)):
            continue

        rec = _build_record(
            date_str, 'Poker Flat', None, None,
            alt_1d, ts_1d, ne_2d, dne_2d,
            cgm_lat_2d=cgm_lat_2d, cgm_lon_2d=cgm_lon_2d,
            plot_segs=plot_segs)

        # 记录所选 beam 仰角（用于标题）
        rec['elm'] = float(segs[0]['elm'])
        records.append(rec)

    print(f'  [Poker Flat] 有效天数: {len(records)}  '
          f'(beam: {beam_select})')
    return records
