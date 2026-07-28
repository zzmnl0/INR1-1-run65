"""Assemble FY targets, space-weather history, and local FY/COSMIC profiles."""

import torch


def query_observation_payload(index, coords, device, exclude_profile_ids=None):
    if index is None:
        return None
    payload = index.query_observation_batch(
        coords.detach().cpu().numpy(),
        exclude_profile_ids=exclude_profile_ids)
    return {
        key: torch.from_numpy(value).to(device, non_blocking=True)
        for key, value in payload.items()
    }


def attach_observation_background(
        payload, model, sw_manager, iri_peak_manager=None, chunk_size=4096):
    """Evaluate the exact shared Background at every valid observation point."""
    if payload is None:
        return None
    valid = payload['valid_mask']
    background = payload['value'].new_zeros(payload['value'].shape)
    flat_coords = payload['coords'][valid]
    valid_indices = valid.flatten().nonzero(as_tuple=True)[0]
    flat_background = background.flatten()
    if len(flat_coords):
        unique_coords, inverse = torch.unique(
            flat_coords, dim=0, sorted=False, return_inverse=True)
    else:
        unique_coords, inverse = flat_coords, torch.empty(
            0, device=flat_coords.device, dtype=torch.long)
    unique_background = background.new_empty(len(unique_coords))
    with torch.no_grad():
        for start in range(0, len(unique_coords), chunk_size):
            coords = unique_coords[start:start + chunk_size]
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
            encoded = model.encode_background(coords, sw_seq, iri_peak=peak)
            unique_background[start:start + len(coords)] = (
                encoded['ne_bkg'].flatten())
    flat_background[valid_indices] = unique_background[inverse]
    result = dict(payload)
    result['background'] = background
    return result


class SlidingWindowBatchProcessor:
    """
    滑动窗口批次处理器

    Receives one FY batch and returns the tensors consumed by FSIA training
    and evaluation.
    """

    def __init__(self, sw_manager, device='cuda',
                 fy_nb_index=None, cosmic_nb_index=None):
        """
        Args:
            sw_manager:       SpaceWeatherManager 实例
            device:           计算设备
            fy_nb_index:      FYNeighborhoodIndex 实例（run61，可选）
            cosmic_nb_index:  COSMICNeighborhoodIndex 实例（run64，可选）
        """
        self.sw_manager = sw_manager
        self.device = device
        self.fy_nb_index = fy_nb_index          # run61: FY 邻域索引
        self.cosmic_nb_index = cosmic_nb_index  # run64: COSMIC 邻域索引

    def process_batch(self, batch_item, query_neighbors=True):
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
            if len(batch_item) == 3:
                batch_data, _, profile_ids = batch_item
            else:
                batch_data, _ = batch_item
                profile_ids = torch.arange(len(batch_data), dtype=torch.long)
            profile_ids_np = profile_ids.cpu().numpy()
        else:
            batch_data = batch_item
            profile_ids = torch.arange(len(batch_data), dtype=torch.long)
            profile_ids_np = None

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
        if query_neighbors and self.fy_nb_index is not None:
            observations_fy = query_observation_payload(
                self.fy_nb_index, coords, self.device,
                exclude_profile_ids=profile_ids_np)
        else:
            observations_fy = None

        # run64/run65: COSMIC-2 邻域观测查询（P2：优先用预计算数据）
        if query_neighbors and self.cosmic_nb_index is not None:
            observations_cosmic = query_observation_payload(
                self.cosmic_nb_index, coords, self.device)
        else:
            observations_cosmic = None

        return (coords, target_ne, sw_seq,
                observations_fy, observations_cosmic,
                profile_ids.to(self.device, non_blocking=True))
