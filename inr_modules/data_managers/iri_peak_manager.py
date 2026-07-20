"""
IRI-2020 预计算峰参数管理器

提供 PeakHead 所需的 IRI 背景锚点（双通道）：
    [hmF2_IRI_km, NmF2_IRI_log10]

数据格式（D:/IRI/data01/edp_peak_npy/readme.txt）：
    IRI_hmF2_*.npy — (241, 181, 181) float32 km
    IRI_NmF2_*.npy — (241, 181, 181) float32 m⁻³ → 转 log10
    时间: 241步×3h; 纬度: 181点 1°步; 经度: 181点 2°步; NaN=无效

NaN-mask 加权插值（Mask-Normalized Interpolation）：
    data_filled = nan_to_num(data, 0.0)
    mask        = (~isnan).float()
    result      = grid_sample(data_filled) / grid_sample(mask).clamp(1e-6)
    fallback    = [300.0 km, 11.5 log10]（仅当邻居全为 NaN）
"""

import os
import numpy as np
import torch
import torch.nn.functional as F


class IRIPeakManager:
    """
    IRI-2020 预计算峰参数管理器（双通道，NaN-mask 三线性插值）

    存储结构（GPU tensor）：
        data_filled: [1, 2, 241, 181, 181]  ch0=hmF2_km, ch1=NmF2_log10
        mask:        [1, 2, 241, 181, 181]  float32 (0/1)

    grid_sample 坐标归一化（align_corners=True）：
        norm_t = rel_hour / 360.0 - 1.0   (idx∈[0,240], 步长3h)
        norm_h = lat_geo  / 90.0           (idx∈[0,180], 步长1°)
        norm_w = lon_geo  / 180.0          (idx∈[0,180], 步长2°)
        grid:  [1, 1, 1, B, 3]  (x=lon_norm, y=lat_norm, z=time_norm)
        output:[1, 2, 1, 1, B] → squeeze → [B, 2]
    """

    # fallback 常量
    _FALLBACK_HMF2  = 300.0   # km
    _FALLBACK_NMF2  = 11.5    # log10
    _NMF2_CLIP_MIN  = 9.0
    _NMF2_CLIP_MAX  = 13.0

    def __init__(self, hmf2_path: str, nmf2_path: str,
                 total_hours: float = 720.0, device='cpu'):
        """
        Args:
            hmf2_path:   IRI_hmF2_*.npy 路径
            nmf2_path:   IRI_NmF2_*.npy 路径（单位 m⁻³，自动转 log10）
            total_hours: 训练窗口总时长（用于边界检查）
            device:      目标设备
        """
        if not os.path.exists(hmf2_path):
            raise FileNotFoundError(f'IRI hmF2 文件未找到: {hmf2_path}')
        if not os.path.exists(nmf2_path):
            raise FileNotFoundError(f'IRI NmF2 文件未找到: {nmf2_path}')

        self.device = torch.device(device)

        # ---- 加载 hmF2 ----
        hmf2_raw = np.load(hmf2_path).astype(np.float32)   # (241,181,181) km
        assert hmf2_raw.shape == (241, 181, 181), \
            f'hmF2 shape 异常: {hmf2_raw.shape}，期望 (241, 181, 181)'

        mask_h = (~np.isnan(hmf2_raw)).astype(np.float32)
        hmf2_filled = np.nan_to_num(hmf2_raw, nan=0.0)

        # ---- 加载并转换 NmF2 ----
        nmf2_raw = np.load(nmf2_path).astype(np.float32)   # (241,181,181) m⁻³
        assert nmf2_raw.shape == (241, 181, 181), \
            f'NmF2 shape 异常: {nmf2_raw.shape}，期望 (241, 181, 181)'

        mask_n = (~np.isnan(nmf2_raw)).astype(np.float32)
        # m⁻³ → log10，clip(1) 防止 log(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            nmf2_log = np.log10(np.clip(nmf2_raw, 1.0, None))
        nmf2_log = np.clip(nmf2_log, self._NMF2_CLIP_MIN, self._NMF2_CLIP_MAX)
        nmf2_filled = np.nan_to_num(nmf2_log, nan=0.0)

        # ---- 构建 [1, 2, 241, 181, 181] tensor ----
        data_np = np.stack([hmf2_filled, nmf2_filled], axis=0)   # (2,241,181,181)
        mask_np = np.stack([mask_h,      mask_n     ], axis=0)   # (2,241,181,181)

        self.data_filled = torch.from_numpy(data_np).unsqueeze(0).to(self.device)
        self.mask        = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        fallback = torch.tensor(
            [[[[self._FALLBACK_HMF2]]], [[[self._FALLBACK_NMF2]]]],
            dtype=torch.float32, device=self.device)              # [2,1,1,1]
        self.fallback = fallback.unsqueeze(0)                     # [1,2,1,1,1]

        print(f'  [IRIPeakManager] hmF2: {hmf2_raw.shape}, '
              f'valid={mask_h.mean()*100:.1f}%')
        print(f'  [IRIPeakManager] NmF2: {nmf2_raw.shape}, '
              f'valid={mask_n.mean()*100:.1f}%')

    @torch.no_grad()
    def get_iri_peak(self, coords: torch.Tensor) -> torch.Tensor:
        """
        批量三线性插值查询 IRI 峰参数

        Args:
            coords: [B, 4+] — col0=lat_geo(°), col1=lon_geo(°), col3=rel_hour(h)
        Returns:
            [B, 2] — [hmF2_IRI_km, NmF2_IRI_log10]，无 NaN
        """
        lat      = coords[:, 0].to(self.device)   # [B]
        lon      = coords[:, 1].to(self.device)   # [B]
        rel_hour = coords[:, 3].to(self.device)   # [B]

        B = lat.shape[0]

        # 归一化到 [-1, 1]（align_corners=True）
        norm_t = rel_hour / 360.0 - 1.0   # time: [0,720]h → [-1,1]
        norm_h = lat      / 90.0           # lat:  [-90,90]° → [-1,1]
        norm_w = lon      / 180.0          # lon: [-180,180]° → [-1,1]

        # grid: [1, 1, 1, B, 3]  (x=lon, y=lat, z=time)
        grid = torch.stack([norm_w, norm_h, norm_t], dim=-1)   # [B, 3]
        grid = grid.view(1, 1, 1, B, 3)

        # NaN-mask 加权插值
        interp_val = F.grid_sample(
            self.data_filled, grid,
            mode='bilinear', padding_mode='border',
            align_corners=True)                   # [1, 2, 1, 1, B]

        interp_mask = F.grid_sample(
            self.mask, grid,
            mode='bilinear', padding_mode='border',
            align_corners=True)                   # [1, 2, 1, 1, B]

        # 归一化（仅用有效邻居）
        result = interp_val / interp_mask.clamp(min=1e-6)       # [1, 2, 1, 1, B]

        # fallback：当所有邻居均无效时使用
        no_valid = (interp_mask < 0.01)
        result   = torch.where(no_valid, self.fallback, result)  # broadcast

        # [1, 2, 1, 1, B] → [B, 2]
        return result.squeeze(0).squeeze(1).squeeze(1).T.contiguous()


# ======================== 自测 ========================
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    hmf2_path = r'D:\IRI\data01\edp\IRI_hmF2_20240901_20241001.npy'
    nmf2_path = r'D:\IRI\data01\edp\IRI_NmF2_20240901_20241001.npy'

    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    mgr = IRIPeakManager(hmf2_path, nmf2_path, device=device)

    B = 64
    coords = __import__('torch').zeros(B, 4)
    coords[:, 0] = __import__('torch').linspace(-60, 60, B)    # lat
    coords[:, 1] = __import__('torch').linspace(-180, 180, B)  # lon
    coords[:, 3] = __import__('torch').linspace(0, 720, B)     # rel_hour

    out = mgr.get_iri_peak(coords)
    print(f'\nget_iri_peak output shape: {out.shape}')
    print(f'hmF2 range: [{out[:,0].min():.1f}, {out[:,0].max():.1f}] km')
    print(f'NmF2 range: [{out[:,1].min():.2f}, {out[:,1].max():.2f}] log10')
    assert out.shape == (B, 2), 'shape 错误'
    assert not out.isnan().any(), '存在 NaN'
    assert (out[:, 0] >= 200).all() and (out[:, 0] <= 550).all(), 'hmF2 越界'
    assert (out[:, 1] >= 9.0).all()  and (out[:, 1] <= 13.0).all(), 'NmF2 越界'
    print('Step 2 自测通过')
