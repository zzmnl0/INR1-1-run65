"""
FY 卫星 EDP 数据预处理：添加 AACGM 磁纬列

将 fy_202409_clean.npy（N×5，[Lat_geo, Lon_geo, Alt, RelHour, NeLog10]）
转换为 fy_202409_with_aacgm.npy（N×7，新增 [Lat_aacgm, Lon_aacgm_unused]）

原理：
    - AACGM 磁纬比地理纬度更好地描述 EIA 等磁场相关结构的空间分布
    - EIA 峰值位置在磁赤道 ±15–20° 处（地理上因磁赤道倾角最大偏移 25°）
    - 使用单一参考历元（2024-09-15）：一个月内 AACGM 坐标变化 < 0.05°，可忽略

依赖：
    pip install aacgmv2

用法：
    python add_aacgm_to_fy.py [--input PATH] [--output PATH] [--batch_size N]
"""

import numpy as np
import argparse
from datetime import datetime
import os

# 默认路径
DEFAULT_INPUT  = r'D:\FYsatellite\EDP_data\fy_202409_clean.npy'
DEFAULT_OUTPUT = r'D:\FYsatellite\EDP_data\fy_202409_with_aacgm.npy'

# AACGM 参考历元（2024-09-15，位于训练月份中央）
_REF_DATETIME = datetime(2024, 9, 15, 0, 0, 0)

# alt 参考值（AACGM 依赖高度，使用 F 区典型高度 300 km）
_AACGM_REF_ALT_KM = 300.0


def add_aacgm_columns(input_path, output_path, batch_size=50000):
    """
    加载 FY EDP npy，计算每点的 AACGM 磁纬/磁经，保存扩展 npy。

    Args:
        input_path:  输入文件路径（N×5 float32）
        output_path: 输出文件路径（N×7 float32）
        batch_size:  aacgmv2 分批调用大小（避免内存溢出）
    """
    try:
        import aacgmv2
    except ImportError:
        raise ImportError(
            'aacgmv2 未安装。请运行: pip install aacgmv2\n'
            '若在 conda 环境中: conda activate pytorch_cpu && pip install aacgmv2'
        )

    print(f'加载输入数据: {input_path}')
    data = np.load(input_path)
    N, C = data.shape
    print(f'  数据形状: {data.shape}')

    if C < 5:
        raise ValueError(f'输入数据列数 {C} < 5，期望 [Lat, Lon, Alt, RelHour, NeLog10]')

    if C >= 7:
        print('  检测到数据已含 7+ 列，无需重新处理。直接保存到输出路径。')
        np.save(output_path, data.astype(np.float32))
        print(f'  已复制到: {output_path}')
        return

    lat_geo = data[:, 0].astype(np.float64)
    lon_geo = data[:, 1].astype(np.float64)
    # AACGM 使用固定参考高度，避免每点高度导致的额外误差
    alt_ref = np.full(N, _AACGM_REF_ALT_KM, dtype=np.float64)

    lat_aacgm = np.full(N, np.nan, dtype=np.float64)
    lon_aacgm = np.full(N, np.nan, dtype=np.float64)

    print(f'计算 AACGM 坐标（参考历元: {_REF_DATETIME.date()}, 高度: {_AACGM_REF_ALT_KM} km）...')
    print(f'  总样本: {N:,}  批次大小: {batch_size:,}')

    n_batches = (N + batch_size - 1) // batch_size
    for i in range(n_batches):
        s = i * batch_size
        e = min(s + batch_size, N)

        try:
            mlat, mlon, _ = aacgmv2.convert_latlon_arr(
                lat_geo[s:e], lon_geo[s:e], alt_ref[s:e],
                _REF_DATETIME,
                method_code='G2A'   # Geographic → AACGM
            )
            lat_aacgm[s:e] = mlat
            lon_aacgm[s:e] = mlon
        except Exception as ex:
            print(f'  警告: 批次 {i+1}/{n_batches} 转换失败 ({ex})，填充 NaN')

        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            pct = (i + 1) / n_batches * 100
            nan_count = np.sum(np.isnan(lat_aacgm[:e]))
            print(f'  {i+1:>4}/{n_batches} ({pct:5.1f}%)  NaN 点数: {nan_count:,}')

    # NaN 回退：使用地理纬度（极盖区 / 高纬可能超出 AACGM 有效范围）
    nan_mask = np.isnan(lat_aacgm)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        print(f'  {nan_count:,} 个点 AACGM 转换失败，回退使用地理纬度')
        lat_aacgm[nan_mask] = lat_geo[nan_mask]
        lon_aacgm[nan_mask] = lon_geo[nan_mask]

    # 构建输出数据（N×7）
    out_data = np.column_stack([
        data[:, :5],                        # 原始 5 列不变
        lat_aacgm.astype(np.float32),       # 列 5: AACGM 磁纬
        lon_aacgm.astype(np.float32),       # 列 6: AACGM 磁经（供参考，模型未使用）
    ]).astype(np.float32)

    print(f'\n输出数据形状: {out_data.shape}')
    print(f'AACGM 磁纬范围: [{np.nanmin(lat_aacgm):.2f}°, {np.nanmax(lat_aacgm):.2f}°]')
    print(f'地理纬度范围: [{lat_geo.min():.2f}°, {lat_geo.max():.2f}°]')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, out_data)
    print(f'\n已保存: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FY EDP 数据添加 AACGM 磁纬列')
    parser.add_argument('--input',      default=DEFAULT_INPUT,
                        help=f'输入 npy 路径（默认: {DEFAULT_INPUT}）')
    parser.add_argument('--output',     default=DEFAULT_OUTPUT,
                        help=f'输出 npy 路径（默认: {DEFAULT_OUTPUT}）')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='每批转换样本数（默认: 50000）')
    args = parser.parse_args()

    add_aacgm_columns(args.input, args.output, args.batch_size)
