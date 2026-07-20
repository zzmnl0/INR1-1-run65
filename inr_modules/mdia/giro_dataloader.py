"""
GIRO/DIDBase 电离层测高仪数据载入模块

提供 Dataset / DataLoader 以及训练循环中的 GIRO 监督损失计算辅助函数。

数据格式（由 preprocess_giro.py 生成）：
  giro_hmf2.npy — (N, 5) float32: [lat_geo, lon_geo, rel_hour, hmf2_km,    lat_aacgm]
  giro_nmf2.npy — (N, 5) float32: [lat_geo, lon_geo, rel_hour, nmf2_log10, lat_aacgm]

典型训练用法：
  from giro_dataloader import build_giro_loaders, compute_giro_loss

  giro_hmf2_loader, giro_nmf2_loader = build_giro_loaders(config, device)
  giro_hmf2_iter = iter(giro_hmf2_loader)  # 循环迭代器
  giro_nmf2_iter = iter(giro_nmf2_loader)

  # 在训练循环中：
  giro_loss, giro_dict = compute_giro_loss(
      model, sw_manager,
      giro_hmf2_iter, giro_hmf2_loader,
      giro_nmf2_iter, giro_nmf2_loader,
      device, config,
  )
  total_loss = fy_loss + giro_loss
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# GIRO 查询时的假高度（hmF2 是 NeQuick 解码器参数，与 alt 无关；此处仅为 forward() 正常运行）
_DUMMY_ALT_KM = 300.0


# ======================== Dataset ========================

class GIRODataset(Dataset):
    """
    GIRO 监督数据集（hmF2 或 NmF2）

    npy 文件格式（均为 float32，shape (N, 5)）：
      hmF2 文件：[lat_geo, lon_geo, rel_hour, hmf2_km,    lat_aacgm]
      NmF2 文件：[lat_geo, lon_geo, rel_hour, nmf2_log10, lat_aacgm]

    Args:
        npy_path:     .npy 文件路径
        time_range:   (t_min, t_max)  过滤 rel_hour（默认 0-720h）
        device:       数据预置设备（None = 返回 CPU tensor，训练时再移动）
    """
    def __init__(self, npy_path, time_range=(0.0, 720.0), device=None):
        data = np.load(npy_path)  # (N, 5)
        if data.ndim != 2 or data.shape[1] != 5:
            raise ValueError(f"期望 (N, 5) 格式，实际为 {data.shape}: {npy_path}")

        # 时间窗口过滤
        t = data[:, 2]
        mask = (t >= time_range[0]) & (t <= time_range[1])
        data = data[mask]

        self.data = torch.tensor(data, dtype=torch.float32)
        if device is not None:
            self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # [lat_geo, lon_geo, rel_hour, value, lat_aacgm]


# ======================== DataLoader 构建 ========================

def build_giro_loaders(config, device=None):
    """
    根据配置构建 GIRO hmF2 / NmF2 DataLoader。
    若对应路径为 None 或权重为 0，则返回 None。

    Returns:
        (hmf2_loader, nmf2_loader)  — 任一可能为 None
    """
    total_hours = config.get('total_hours', 720.0)
    time_range = (0.0, total_hours)
    batch_size = config.get('giro_batch_size', 256)

    def _build(path_key, weight_key):
        path = config.get(path_key)
        w = config.get(weight_key, 0.0)
        if not path or w <= 0:
            return None
        if not __import__('os').path.exists(path):
            print(f"  警告: GIRO 文件未找到: {path}，跳过 {weight_key}")
            return None
        ds = GIRODataset(path, time_range=time_range)
        print(f"  GIRO {path_key}: {len(ds)} 条记录  batch_size={batch_size}")
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          drop_last=False, num_workers=0, pin_memory=False)

    hmf2_loader = _build('giro_hmf2_path', 'w_giro_hmf2')
    nmf2_loader = _build('giro_nmf2_path', 'w_giro_nmf2')
    return hmf2_loader, nmf2_loader


# ======================== 训练循环辅助 ========================

def _safe_next(it, loader):
    """从迭代器取下一批；耗尽后重新建立迭代器（循环采样）"""
    try:
        return next(it), it
    except StopIteration:
        new_it = iter(loader)
        return next(new_it), new_it


def _build_giro_coords(batch, device):
    """
    将 GIRO DataLoader 返回的 batch 转换为模型 coords [B, 5]。

    batch [B, 5]: [lat_geo, lon_geo, rel_hour, value, lat_aacgm]
    coords [B, 5]: [lat_geo, lon_geo, alt_dummy, rel_hour, lat_aacgm]
    """
    batch = batch.to(device)
    lat_geo  = batch[:, 0]    # [B]
    lon_geo  = batch[:, 1]    # [B]
    rel_hour = batch[:, 2]    # [B]
    value    = batch[:, 3]    # [B]  — hmF2_km 或 nmF2_log10
    lat_aacgm = batch[:, 4]   # [B]

    alt_dummy = torch.full_like(lat_geo, _DUMMY_ALT_KM)
    coords = torch.stack([lat_geo, lon_geo, alt_dummy, rel_hour, lat_aacgm], dim=1)  # [B, 5]
    return coords, value


def _get_peak_pred(extras):
    """
    从 FSIA-INR extras 中提取 PeakHead 预测的 hmF2 / NmF2。

    extras['peak_params'] = {'hmF2': [B], 'NmF2': [B]}

    Returns:
        (hmF2_pred, NmF2_pred) — 任一可能为 None（若 peak_params 键不存在）
    """
    pp = extras.get('peak_params', {})
    return pp.get('hmF2'), pp.get('NmF2')


def compute_giro_loss(
        model, sw_manager,
        hmf2_iter, hmf2_loader,
        nmf2_iter, nmf2_loader,
        device, config,
        iri_peak_manager=None):
    """
    从 GIRO 迭代器取一个 mini-batch，前向查询模型，计算 hmF2 / NmF2 监督损失。

    模型 extras['peak_params'] 中直接提供 hmF2 和 NmF2，无需在特定高度再次采样。

    Args:
        iri_peak_manager: IRIPeakManager 实例（可选）；提供时传入 PeakHead 的 IRI 背景

    Returns:
        total_giro_loss (Tensor | 0.0)
        loss_dict       (dict) — {'giro_hmf2': float, 'giro_nmf2': float}
        (hmf2_iter, nmf2_iter)  — 更新后的迭代器
    """
    w_hmf2 = config.get('w_giro_hmf2', 0.0)
    w_nmf2 = config.get('w_giro_nmf2', 0.0)
    loss_dict = {'giro_hmf2': 0.0, 'giro_nmf2': 0.0}
    total_loss = 0.0

    # --- hmF2 监督 ---
    if hmf2_loader is not None and w_hmf2 > 0:
        batch, hmf2_iter = _safe_next(hmf2_iter, hmf2_loader)
        coords_g, hmf2_target = _build_giro_coords(batch, device)
        sw_seq_g = sw_manager.get_drivers_sequence(coords_g[:, 3])  # [B, seq_len, 2]

        _iri_peak_g = None
        if iri_peak_manager is not None:
            _iri_peak_g = iri_peak_manager.get_iri_peak(coords_g)
        _, _, _, _, extras_g = model(coords_g, sw_seq_g,
                                     giro_mode=True, iri_peak=_iri_peak_g)
        hmf2_pred, _ = _get_peak_pred(extras_g)
        if hmf2_pred is not None:
            hmf2_target = hmf2_target.clamp(150.0, 600.0)  # 剔除极端异常值
            # 特征动态非对称权重：基于 PeakHead 输出（来自 h_spatial + h_sw 的网络特征）
            # 预测峰高越低 → sigmoid 值越大 → 额外惩罚越强（自纠正动态）
            # detach 避免循环梯度：权重放大梯度但不参与 hmF2_pred 的二阶反传
            base_asym = config.get('giro_hmf2_asym_weight', 2.0)
            hmf2_center = config.get('giro_hmf2_asym_center', 350.0)
            asym_extra = 2.0 * torch.sigmoid(
                -(hmf2_pred.detach() - hmf2_center) / 60.0)   # [B]: 0→2
            asym_w_spatial = base_asym + asym_extra            # [B]: base→base+2
            diff = hmf2_pred - hmf2_target
            weights = torch.where(diff < 0, asym_w_spatial, torch.ones_like(diff))
            loss_hmf2 = (weights * diff ** 2).mean()
            loss_dict['giro_hmf2'] = loss_hmf2.item()
            total_loss = total_loss + w_hmf2 * loss_hmf2

    # --- NmF2 监督 ---
    if nmf2_loader is not None and w_nmf2 > 0:
        batch, nmf2_iter = _safe_next(nmf2_iter, nmf2_loader)
        coords_g, nmf2_target = _build_giro_coords(batch, device)
        sw_seq_g = sw_manager.get_drivers_sequence(coords_g[:, 3])  # [B, seq_len, 2]

        _iri_peak_g = None
        if iri_peak_manager is not None:
            _iri_peak_g = iri_peak_manager.get_iri_peak(coords_g)
        _, _, _, _, extras_g = model(coords_g, sw_seq_g,
                                     giro_mode=True, iri_peak=_iri_peak_g)
        _, nmf2_pred = _get_peak_pred(extras_g)
        if nmf2_pred is not None:
            loss_nmf2 = F.mse_loss(nmf2_pred, nmf2_target)
            loss_dict['giro_nmf2'] = loss_nmf2.item()
            total_loss = total_loss + w_nmf2 * loss_nmf2

    return total_loss, loss_dict, hmf2_iter, nmf2_iter
