"""
GIRO/DIDBase 电离层测高仪数据预处理脚本

将 GIRO .txt 文件批量解析并转换为 .npy 格式，供训练时快速载入。

输出文件：
  giro_hmf2.npy  — shape (N, 5)  float32
                   列：[lat_geo, lon_geo, rel_hour, hmf2_km, lat_aacgm]
                   条件：hmF2 有效（QD=="//）且 CS >= cs_threshold
  giro_nmf2.npy  — shape (N, 5)  float32
                   列：[lat_geo, lon_geo, rel_hour, nmf2_log10, lat_aacgm]
                   条件：foF2 有效（QD=="//"）且 CS >= cs_threshold
                   nmf2_log10 = log10(1.24e10 × foF2²)

数据格式说明：
  列头：  #Time  CS  foF2 QD  hmF2 QD
  每行固定 6 列（时间戳 / CS / foF2 / foF2_QD / hmF2 / hmF2_QD）
  "---" 表示无数据，"__" 表示 QD 字段对应数据无效
  CS=-1：置信度未知；CS=0：低质量；CS=50-100：自动定标质量分

用法：
  python preprocess_giro.py \\
      --giro_dir  D:/c_shuju/GIRO_hmf2 \\
      --output_dir D:/c_shuju/GIRO_hmf2/processed \\
      [--cs_threshold -1]   # 默认 -1 保留全部（QD 列为有效性主判据）
"""

import os
import sys
import glob
import argparse
import numpy as np
from datetime import datetime, timezone

# AACGM 磁纬转换（可选，安装 aacgmv2 后自动启用）
try:
    import aacgmv2
    _HAS_AACGM = True
except ImportError:
    _HAS_AACGM = False

# ======================== 常量 ========================

# 训练基准时刻（UTC）
_START_UTC = datetime(2024, 9, 1, 0, 0, 0, tzinfo=timezone.utc)

# AACGM 参考历元（与 FY 数据 / 可视化保持一致）
_AACGM_EPOCH = datetime(2024, 9, 15)

# NmF2 换算因子：NmF2 [m⁻³] = K × foF2² [MHz²]
# 等离子体频率公式：fp [Hz] = 8.979 × sqrt(Ne [m⁻³])
# → Ne = (foF2×1e6 / 8.979)² ≈ 1.2407e10 × foF2²
_NMF2_K = 1.2407e10
_NMF2_LOG10_K = np.log10(_NMF2_K)   # ≈ 10.0937


# ======================== 头部解析 ========================

def _parse_header(lines):
    """从 # 注释行提取站点地理坐标（十进制度，经度转换为 -180~+180）"""
    lat, lon = None, None
    for line in lines:
        if line.startswith('# Location: GEO'):
            # 格式示例：# Location: GEO 33.43N 126.3E, URSI-Code JJ433 JEJU
            geo_part = line.split('GEO')[1].strip().split(',')[0].strip()
            parts = geo_part.split()
            lat_str, lon_str = parts[0], parts[1]

            # 纬度：正 = N，负 = S
            lat_val = float(lat_str[:-1])
            if 'S' in lat_str:
                lat_val = -lat_val

            # 经度：0-360E → -180~+180
            lon_val = float(lon_str[:-1])
            if lon_val > 180.0:
                lon_val -= 360.0

            lat, lon = lat_val, lon_val
            break
    return lat, lon


# ======================== 数据行解析 ========================

def _parse_file(filepath, cs_threshold):
    """
    解析单个 GIRO .txt 文件。

    Returns:
        station_lat (float): 地理纬度
        station_lon (float): 地理经度（-180~+180）
        records (list): 每条记录 [rel_hour, hmF2_km_or_nan, nmF2_log10_or_nan]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 头部：以 '#' 开头的行
    header_lines = [l.strip() for l in lines if l.startswith('#')]
    lat, lon = _parse_header(header_lines)
    if lat is None:
        return None, None, []

    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        tokens = line.split()
        if len(tokens) != 6:
            continue  # 格式不符，跳过

        ts_str, cs_str, fof2_str, fof2_qd, hmf2_str, hmf2_qd = tokens

        # 时间戳 → 相对小时
        try:
            ts = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                tzinfo=timezone.utc)
            rel_hour = (ts - _START_UTC).total_seconds() / 3600.0
        except ValueError:
            continue

        # CS 过滤：cs_threshold < 0 时保留全部（包括 CS=-1）
        try:
            cs = int(cs_str)
        except ValueError:
            continue
        if cs_threshold >= 0 and cs < cs_threshold:
            continue

        # foF2：有效时 QD=="//"，否则为 "---" / QD=="__"
        if fof2_str == '---' or fof2_qd != '//':
            nmf2_log10 = np.nan
        else:
            try:
                fof2 = float(fof2_str)
                nmf2_log10 = _NMF2_LOG10_K + 2.0 * np.log10(fof2)
            except (ValueError, FloatingPointError):
                nmf2_log10 = np.nan

        # hmF2：有效时 QD=="//"，否则为 "---" / QD=="__"
        if hmf2_str == '---' or hmf2_qd != '//':
            hmf2_km = np.nan
        else:
            try:
                hmf2_km = float(hmf2_str)
            except ValueError:
                hmf2_km = np.nan

        records.append([rel_hour, hmf2_km, nmf2_log10])

    return lat, lon, records


# ======================== AACGM 转换 ========================

def _get_aacgm_lat(lat, lon, alt_km=300.0):
    """计算地理坐标对应的 AACGM 磁纬（参考历元 2024-09-15）"""
    if not _HAS_AACGM:
        return lat  # 回退：使用地理纬度
    try:
        aacgm_lats, _, _ = aacgmv2.convert_latlon_arr(
            np.array([lat]),
            np.array([lon if lon >= 0 else lon + 360]),  # aacgmv2 需要 0-360
            np.array([alt_km]),
            dtime=_AACGM_EPOCH,
            method_code='G2A'
        )
        val = float(aacgm_lats[0])
        return val if np.isfinite(val) else lat
    except Exception:
        return lat


# ======================== 主处理函数 ========================

def preprocess_giro(giro_dir, output_dir, cs_threshold=-1):
    """
    批量处理 GIRO .txt 文件，输出 hmF2 和 NmF2 监督数据。

    Args:
        giro_dir:     包含 *.txt 的目录
        output_dir:   npy 输出目录
        cs_threshold: 置信度阈值（<0 保留全部，含 CS=-1 未知）
    """
    os.makedirs(output_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(giro_dir, '*.txt')))
    if not txt_files:
        raise FileNotFoundError(f"在 {giro_dir} 中未找到 .txt 文件")

    print(f"\nGIRO 数据预处理")
    print(f"  输入目录：{giro_dir}")
    print(f"  输出目录：{output_dir}")
    print(f"  CS 阈值：{cs_threshold}  (aacgmv2: {'已安装' if _HAS_AACGM else '未安装，使用地理纬度'})")
    print(f"  站点数：{len(txt_files)}\n")

    # 收集所有记录
    hmf2_rows = []   # [lat_geo, lon_geo, rel_hour, hmf2_km, lat_aacgm]
    nmf2_rows = []   # [lat_geo, lon_geo, rel_hour, nmf2_log10, lat_aacgm]

    for fpath in txt_files:
        station_id = os.path.basename(fpath).replace('.txt', '')
        lat, lon, records = _parse_file(fpath, cs_threshold)

        if lat is None or not records:
            print(f"  [跳过] {station_id}: 解析失败或无记录")
            continue

        # AACGM：每站只算一次（地理位置固定）
        aacgm_lat = _get_aacgm_lat(lat, lon)

        n_hmf2 = n_nmf2 = 0
        for rel_hour, hmf2_km, nmf2_log10 in records:
            if not np.isnan(hmf2_km):
                hmf2_rows.append([lat, lon, rel_hour, hmf2_km, aacgm_lat])
                n_hmf2 += 1
            if not np.isnan(nmf2_log10):
                nmf2_rows.append([lat, lon, rel_hour, nmf2_log10, aacgm_lat])
                n_nmf2 += 1

        print(f"  {station_id:8s}  lat={lat:7.2f}  lon={lon:8.2f}  "
              f"aacgm={aacgm_lat:7.2f}  hmF2={n_hmf2:5d}  NmF2={n_nmf2:5d}")

    # 转换为 numpy array
    hmf2_arr = np.array(hmf2_rows, dtype=np.float32)  # (N, 5)
    nmf2_arr = np.array(nmf2_rows, dtype=np.float32)  # (N, 5)

    # 保存
    hmf2_path = os.path.join(output_dir, 'giro_hmf2.npy')
    nmf2_path = os.path.join(output_dir, 'giro_nmf2.npy')
    np.save(hmf2_path, hmf2_arr)
    np.save(nmf2_path, nmf2_arr)

    print(f"\n输出汇总：")
    print(f"  giro_hmf2.npy : {hmf2_arr.shape}  "
          f"({os.path.getsize(hmf2_path)/1024:.1f} KB)  → {hmf2_path}")
    print(f"  giro_nmf2.npy : {nmf2_arr.shape}  "
          f"({os.path.getsize(nmf2_path)/1024:.1f} KB)  → {nmf2_path}")
    print(f"\n列说明：")
    print(f"  giro_hmf2.npy  [lat_geo, lon_geo, rel_hour, hmf2_km,    lat_aacgm]")
    print(f"  giro_nmf2.npy  [lat_geo, lon_geo, rel_hour, nmf2_log10, lat_aacgm]")
    print(f"  rel_hour: 相对于 2024-09-01 00:00 UTC 的小时数")
    print(f"  nmf2_log10 = log10(1.24e10 * foF2^2 [MHz^2])")

    return hmf2_path, nmf2_path


# ======================== 入口 ========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIRO 数据预处理')
    parser.add_argument('--giro_dir', default=r'D:\c_shuju\GIRO_hmf2',
                        help='包含 GIRO .txt 文件的目录')
    parser.add_argument('--output_dir', default=r'D:\c_shuju\GIRO_hmf2\processed',
                        help='npy 输出目录')
    parser.add_argument('--cs_threshold', type=int, default=-1,
                        help='置信度阈值（-1 = 保留全部含未知；50 = 仅高置信度）')
    args = parser.parse_args()

    preprocess_giro(
        giro_dir=args.giro_dir,
        output_dir=args.output_dir,
        cs_threshold=args.cs_threshold,
    )
