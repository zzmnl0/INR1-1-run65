"""
R-STMRF 滑动窗口数据整合工具

设计思路：
    - 保留现有的 FY_dataloader.py 时间分箱策略（TimeBinSampler）
    - 在训练循环中，通过 sw_manager 和 tec_manager 动态获取历史序列
    - 提供辅助函数简化数据流处理
"""

import torch
import numpy as np


class SlidingWindowBatchProcessor:
    """
    滑动窗口批次处理器

    职责：
        - 接收 FY 数据批次（原始格式）
        - 动态查询 sw_manager 和 tec_manager
        - 返回 R-STMRF 所需的完整数据包

    使用方式：
        在训练循环中，替代原有的数据预处理逻辑
    """

    def __init__(self, sw_manager, tec_manager=None, device='cuda',
                 fy_nb_index=None, cosmic_nb_index=None,
                 fy_precomputed=None, csm_precomputed=None):
        """
        Args:
            sw_manager:       SpaceWeatherManager 实例
            tec_manager:      TECDataManager 实例（可选，已不用于前向传播）
            device:           计算设备
            fy_nb_index:      FYNeighborhoodIndex 实例（run61，可选）
            cosmic_nb_index:  COSMICNeighborhoodIndex 实例（run64，可选）
            fy_precomputed:   precompute_all 返回的 FY 邻域预计算数据（run65 P2，可选）
            csm_precomputed:  precompute_all 返回的 COSMIC 邻域预计算数据（run65 P2，可选）
        """
        self.sw_manager = sw_manager
        self.tec_manager = tec_manager  # 保留字段供 MDIA 向后兼容
        self.device = device
        self.fy_nb_index = fy_nb_index          # run61: FY 邻域索引
        self.cosmic_nb_index = cosmic_nb_index  # run64: COSMIC 邻域索引
        self.fy_precomputed  = fy_precomputed   # run65 P2: 预计算 FY 邻域数据
        self.csm_precomputed = csm_precomputed  # run65 P2: 预计算 COSMIC 邻域数据

    def process_batch(self, batch_item):
        """
        处理一个批次的 FY 数据（新架构：移除 TEC 在线加载）

        TEC 梯度方向现在通过 TecGradientBank 离线预计算 + 时间插值获得，
        不再需要在此处加载 TEC 地图序列。

        支持两种输入形式（run65 P2）：
            - 单张量：[Batch, 5 or 7]（backward compat）
            - 元组：  (batch_data [Batch, 5 or 7], batch_ds_idx [Batch])
                      batch_ds_idx 用于 O(1) 预计算邻域查询

        支持两种 FY 数据格式：
            5 列：[Lat_geo, Lon_geo, Alt, Time, Ne_Log]
                  → coords [Batch, 4]（无 AACGM）
            7 列：[Lat_geo, Lon_geo, Alt, Time, Ne_Log, Lat_aacgm, Lon_aacgm]
                  → coords [Batch, 5]（含 AACGM 磁纬，由 add_aacgm_to_fy.py 生成）

        Args:
            batch_item: [Batch, 5/7] Tensor，或 (batch_data, batch_ds_idx) 元组

        Returns:
            coords:    [Batch, 4] 或 [Batch, 5] — (Lat_geo, Lon_geo, Alt, Time[, Lat_aacgm])
            target_ne: [Batch, 1] 真值 Ne（对数）
            sw_seq:    [Batch, Seq, 2] 空间天气序列
        """
        # 解包元组（run65 P2：DataLoader 现返回 (data, ds_idx)）
        if isinstance(batch_item, (tuple, list)):
            batch_data, batch_ds_idx = batch_item
            batch_ds_idx_np = batch_ds_idx.numpy()
        else:
            batch_data = batch_item
            batch_ds_idx_np = None

        batch_data = batch_data.to(self.device, non_blocking=True)

        ncols = batch_data.shape[1]

        if ncols >= 7:
            # 7 列：cols [0,1,2,3] = geo coords; col 4 = Ne_Log; cols [5,6] = AACGM
            # 将 AACGM lat (col 5) 追加为 coords 第 5 列
            coords    = torch.cat([batch_data[:, :4], batch_data[:, 5:6]], dim=1)  # [B, 5]
            target_ne = batch_data[:, 4:5]  # [Batch, 1]
        else:
            # 5 列（原始格式）
            coords    = batch_data[:, :4]   # [Batch, 4]
            target_ne = batch_data[:, 4:5]  # [Batch, 1]

        # 查询空间天气序列
        sw_seq = self.sw_manager.get_drivers_sequence(coords[:, 3])  # [Batch, Seq, 2]

        # run61/run65: FY 邻域观测查询（P2：优先用预计算数据）
        if self.fy_nb_index is not None:
            coords_np = coords.detach().cpu().numpy()
            if (self.fy_precomputed is not None and batch_ds_idx_np is not None):
                # P2: O(1) 索引，仅重算 Δalt delta 特征
                nb_feats_np, has_obs_np = self.fy_nb_index.query_batch_precomputed(
                    batch_ds_idx_np, coords_np[:, 2],
                    self.fy_precomputed,
                    coords_np[:, 0], coords_np[:, 1], coords_np[:, 3],
                )
            else:
                # fallback: 在线剖面搜索
                nb_feats_np, has_obs_np = self.fy_nb_index.query_batch_np(coords_np)
            neighbors_feats = torch.from_numpy(nb_feats_np).to(self.device)
            has_obs = torch.from_numpy(has_obs_np).to(self.device)
        else:
            neighbors_feats = None
            has_obs = None

        # run64/run65: COSMIC-2 邻域观测查询（P2：优先用预计算数据）
        if self.cosmic_nb_index is not None:
            coords_np_csm = coords.detach().cpu().numpy()
            if (self.csm_precomputed is not None and batch_ds_idx_np is not None):
                # P2: O(1) 索引
                nb_csm_np, has_csm_np = self.cosmic_nb_index.query_batch_precomputed(
                    batch_ds_idx_np, coords_np_csm[:, 2],
                    self.csm_precomputed,
                    coords_np_csm[:, 0], coords_np_csm[:, 1], coords_np_csm[:, 3],
                )
            else:
                # fallback: 在线剖面搜索
                nb_csm_np, has_csm_np = self.cosmic_nb_index.query_batch_np(coords_np_csm)
            neighbors_feats_cosmic = torch.from_numpy(nb_csm_np).to(self.device)
            has_obs_cosmic = torch.from_numpy(has_csm_np).to(self.device)
        else:
            neighbors_feats_cosmic = None
            has_obs_cosmic = None

        return (coords, target_ne, sw_seq,
                neighbors_feats, has_obs,
                neighbors_feats_cosmic, has_obs_cosmic)


def get_r_stmrf_dataloaders(fy_path, val_days, batch_size, bin_size_hours,
                              sw_manager, tec_manager, num_workers=0):
    """
    获取 R-STMRF 专用的 DataLoader

    Note:
        实际上使用原有的 get_dataloaders 即可，数据处理在训练循环中进行

    Args:
        fy_path: FY 数据路径
        val_days: 验证集日期
        batch_size: 批次大小
        bin_size_hours: 时间分箱大小
        sw_manager: 空间天气管理器
        tec_manager: TEC 管理器
        num_workers: 工作进程数

    Returns:
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        batch_processor: 批次处理器实例
    """
    # 导入原有的 DataLoader 工厂函数
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from data_managers.FY_dataloader import get_dataloaders

    # 获取原有的 DataLoader
    train_loader, val_loader = get_dataloaders(
        npy_path=fy_path,
        val_days=val_days,
        batch_size=batch_size,
        bin_size_hours=bin_size_hours,
        num_workers=num_workers
    )

    # 创建批次处理器
    device = next(iter(train_loader)).device if torch.cuda.is_available() else 'cpu'
    batch_processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)

    return train_loader, val_loader, batch_processor


# ======================== 辅助函数 ========================
def collate_with_sequences(batch, sw_manager, tec_manager, device='cuda'):
    """
    自定义 Collate 函数（可选方案）

    如果希望在 DataLoader 层面整合序列数据，可以使用此函数作为 collate_fn

    注意：新架构中，TEC 梯度通过 TecGradientBank 离线预计算，不再在此处加载

    Args:
        batch: List of samples from Dataset
        sw_manager: SpaceWeatherManager
        tec_manager: TECDataManager
        device: 计算设备

    Returns:
        collated_data: dict
    """
    # 将批次堆叠为 Tensor
    batch_data = torch.stack([item for item in batch], dim=0)  # [Batch, 5]

    # 使用批次处理器
    processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)
    (coords, target_ne, sw_seq,
     neighbors_feats, has_obs,
     neighbors_feats_cosmic, has_obs_cosmic) = processor.process_batch(batch_data)

    return {
        'coords': coords,
        'target_ne': target_ne,
        'sw_seq': sw_seq,
        'neighbors_feats': neighbors_feats,
        'has_obs': has_obs,
        'neighbors_feats_cosmic': neighbors_feats_cosmic,
        'has_obs_cosmic': has_obs_cosmic,
    }


# ======================== 使用示例 ========================
if __name__ == '__main__':
    print("="*60)
    print("滑动窗口数据处理器测试")
    print("="*60)

    # 模拟数据管理器
    class DummySWManager:
        def __init__(self, seq_len=6, device='cuda'):
            self.seq_len = seq_len
            self.device = device

        def get_drivers_sequence(self, time_batch):
            batch_size = time_batch.shape[0]
            return torch.randn(batch_size, self.seq_len, 2).to(self.device)

    class DummyTECManager:
        def __init__(self, seq_len=6, device='cuda'):
            self.seq_len = seq_len
            self.device = device

        def get_tec_map_sequence(self, time_batch):
            batch_size = time_batch.shape[0]
            return torch.rand(batch_size, self.seq_len, 1, 181, 361).to(self.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建管理器
    sw_manager = DummySWManager(seq_len=6, device=device)
    tec_manager = DummyTECManager(seq_len=6, device=device)

    # 创建批次处理器
    processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device=device)

    # 模拟 FY 批次数据
    batch_size = 128
    batch_data = torch.randn(batch_size, 5)
    batch_data[:, 0] = batch_data[:, 0] * 90  # Lat
    batch_data[:, 1] = batch_data[:, 1] * 180  # Lon
    batch_data[:, 2] = 200 + batch_data[:, 2].abs() * 100  # Alt
    batch_data[:, 3] = batch_data[:, 3].abs() * 100  # Time
    batch_data[:, 4] = batch_data[:, 4] + 11.0  # Ne_Log

    print(f"\n输入批次数据: {batch_data.shape}")

    # 处理批次
    (coords, target_ne, sw_seq,
     neighbors_feats, has_obs,
     neighbors_feats_cosmic, has_obs_cosmic) = processor.process_batch(batch_data)

    print("\n输出形状:")
    print(f"  coords: {coords.shape}")
    print(f"  target_ne: {target_ne.shape}")
    print(f"  sw_seq: {sw_seq.shape}")
    print(f"  neighbors_feats: {neighbors_feats}")
    print(f"  has_obs: {has_obs}")
    print(f"  neighbors_feats_cosmic: {neighbors_feats_cosmic}")
    print(f"  has_obs_cosmic: {has_obs_cosmic}")
    print(f"\n注意: TEC 梯度方向现在通过 TecGradientBank 外部提供")

    print("\n数据范围检查:")
    print(f"  Lat: [{coords[:, 0].min().item():.2f}, {coords[:, 0].max().item():.2f}]")
    print(f"  Lon: [{coords[:, 1].min().item():.2f}, {coords[:, 1].max().item():.2f}]")
    print(f"  Alt: [{coords[:, 2].min().item():.2f}, {coords[:, 2].max().item():.2f}]")
    print(f"  Time: [{coords[:, 3].min().item():.2f}, {coords[:, 3].max().item():.2f}]")

    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)
