"""Assemble FY targets, space-weather history, and local FY/COSMIC profiles."""

import torch


class SlidingWindowBatchProcessor:
    """
    滑动窗口批次处理器

    Receives one FY batch and returns the tensors consumed by FSIA training
    and evaluation.
    """

    def __init__(self, sw_manager, device='cuda',
                 fy_nb_index=None, cosmic_nb_index=None,
                 fy_precomputed=None, csm_precomputed=None):
        """
        Args:
            sw_manager:       SpaceWeatherManager 实例
            device:           计算设备
            fy_nb_index:      FYNeighborhoodIndex 实例（run61，可选）
            cosmic_nb_index:  COSMICNeighborhoodIndex 实例（run64，可选）
            fy_precomputed:   precompute_all 返回的 FY 邻域预计算数据（run65 P2，可选）
            csm_precomputed:  precompute_all 返回的 COSMIC 邻域预计算数据（run65 P2，可选）
        """
        self.sw_manager = sw_manager
        self.device = device
        self.fy_nb_index = fy_nb_index          # run61: FY 邻域索引
        self.cosmic_nb_index = cosmic_nb_index  # run64: COSMIC 邻域索引
        self.fy_precomputed  = fy_precomputed   # run65 P2: 预计算 FY 邻域数据
        self.csm_precomputed = csm_precomputed  # run65 P2: 预计算 COSMIC 邻域数据

    def process_batch(self, batch_item):
        """
        Convert one FY batch and query the local FY/COSMIC profile indexes.

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
