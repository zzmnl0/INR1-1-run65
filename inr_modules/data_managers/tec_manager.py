"""
TEC（总电子含量）数据管理器
支持时序窗口和3D grid_sample
"""
import torch
import torch.nn.functional as F
import numpy as np
import os


class TECDataManager:
    """
    TEC数据管理器

    功能:
    - 加载TEC地图数据（保持原始分辨率）
    - 提供时序窗口采样
    - 支持双线性插值

    数据规格:
    - 原始: (720, 71, 73) - Time × Lat × Lon
    - Lat: -87.5 ~ 87.5, 步长 2.5° (71个点)
    - Lon: -180 ~ 180, 步长 5° (73个点)
    - 纬度填充后: (720, 73, 73) - 填充到 -90 ~ 90
    """

    def __init__(self, tec_map_path, total_hours, seq_len=12, device='cuda'):
        """
        Args:
            tec_map_path: TEC数据文件路径
            total_hours: 总时长（小时）
            seq_len: 时序窗口长度
            device: 计算设备
        """
        print(f"[TECDataManager] 加载数据...")

        if not os.path.exists(tec_map_path):
            raise FileNotFoundError(f"TEC文件未找到: {tec_map_path}")

        self.device = device
        self.total_hours = float(total_hours)
        self.seq_len = int(seq_len)

        # 加载原始数据 [Time, Lat, Lon]
        # 预期形状: (720, 71, 73)
        raw_data = np.load(tec_map_path).astype(np.float32)
        print(f"  原始TEC数据形状: {raw_data.shape}")

        tec_tensor = torch.from_numpy(raw_data).unsqueeze(1)  # [Time, 1, Lat, Lon]

        # 仅纬度填充（保持原始分辨率，不做上采样/降采样）
        # 原始: 71个纬度点 (-87.5 ~ 87.5)
        # 填充: 前后各填充1个点，覆盖 -90 ~ 90
        # 结果: 73个纬度点
        pad_tensor = F.pad(tec_tensor, (0, 0, 1, 1), mode='replicate')
        # pad_tensor: [Time, 1, 73, 73]

        # 保持原始分辨率：73×73
        self.target_h = 73  # 纬度点数（填充后）
        self.target_w = 73  # 经度点数（原始）

        # 转换为3D体积 [1, 1, Time, Lat, Lon]
        self.tec_vol = pad_tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(device)

        # 归一化统计
        self.min_tec = torch.min(self.tec_vol)
        self.max_tec = torch.max(self.tec_vol)
        self.denom = self.max_tec - self.min_tec + 1e-6

        print(f"  TEC管理器就绪. 形状: {self.tec_vol.shape}")
        print(f"  分辨率: Lat={self.target_h}, Lon={self.target_w}")
        print(f"  归一化范围: Min={self.min_tec:.2f}, Max={self.max_tec:.2f}")
    
    def get_tec_sequence(self, lat, lon, time_end):
        """
        获取TEC历史序列
        
        Args:
            lat: [Batch] 纬度
            lon: [Batch] 经度
            time_end: [Batch] 结束时间（小时）
            
        Returns:
            [Batch, Seq_Len, 1] TEC序列（归一化）
        """
        batch_size = lat.shape[0]
        
        # 生成时间序列: [t - seq_len + 1, ..., t]
        offsets = torch.arange(self.seq_len, device=self.device).flip(0)
        time_seq = time_end.unsqueeze(1) - offsets.unsqueeze(0)  # [Batch, Seq]
        
        # 归一化坐标到[-1, 1]
        t_norm = (time_seq / self.total_hours) * 2.0 - 1.0
        lat_norm = (lat / 90.0).unsqueeze(1).expand(-1, self.seq_len)
        lon_norm = (lon / 180.0).unsqueeze(1).expand(-1, self.seq_len)
        
        # 构建采样网格 [1, 1, 1, Batch*Seq, 3] (D, H, W) -> (Time, Lat, Lon)
        flat_size = batch_size * self.seq_len
        grid = torch.stack([lon_norm, lat_norm, t_norm], dim=-1).view(1, 1, 1, flat_size, 3)
        
        # Grid sample
        val = F.grid_sample(
            self.tec_vol, grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        # 重塑为 [Batch, Seq_Len, 1]
        val = val.view(batch_size, self.seq_len, 1)
        
        # 归一化到[0, 1]
        val_norm = (val - self.min_tec) / self.denom
        val_norm = torch.clamp(val_norm, 0.0, 1.0)

        return val_norm

    def get_tec_map_sequence(self, time_end):
        """
        获取 TEC 地图序列（用于 ConvLSTM）

        Args:
            time_end: [Batch] 结束时间（小时）

        Returns:
            [Batch, Seq_Len, 1, H, W] TEC 地图序列（归一化）
        """
        batch_size = time_end.shape[0]
        device = time_end.device

        # 生成时间序列: [t - seq_len + 1, ..., t]
        offsets = torch.arange(self.seq_len, device=device).flip(0)  # [Seq]
        time_seq = time_end.unsqueeze(1) - offsets.unsqueeze(0)  # [Batch, Seq]

        # 将时间归一化到 [0, T-1] 索引范围
        time_indices = time_seq.clamp(0, self.total_hours - 1)  # [Batch, Seq]

        # 转换为整数索引（四舍五入）
        time_indices_int = time_indices.round().long()  # [Batch, Seq]

        # 从 tec_vol [1, 1, Time, Lat, Lon] 中提取地图
        # 使用高级索引一次性提取所有地图（避免循环）
        # tec_vol: [1, 1, Time, Lat, Lon] -> [Time, Lat, Lon]
        tec_data = self.tec_vol.squeeze(0).squeeze(0)  # [Time, Lat, Lon]

        # 展平索引以便批量提取
        flat_indices = time_indices_int.view(-1)  # [Batch * Seq]

        # 批量索引：[Batch*Seq, Lat, Lon]
        flat_maps = tec_data[flat_indices]

        # 重塑为 [Batch, Seq, Lat, Lon]
        tec_maps = flat_maps.view(batch_size, self.seq_len, self.target_h, self.target_w)

        # 添加通道维度：[Batch, Seq, 1, Lat, Lon]
        tec_maps = tec_maps.unsqueeze(2)

        # 归一化到 [0, 1]
        tec_maps_norm = (tec_maps - self.min_tec) / self.denom
        tec_maps_norm = torch.clamp(tec_maps_norm, 0.0, 1.0)

        return tec_maps_norm

    def get_tec_map_sequence_by_hours(self, hour_indices):
        """
        获取指定整点小时的 TEC 地图序列（用于 HourlyTECContextCache）

        此方法专为多时间尺度架构设计：ConvLSTM 只在 1h 整点运行，
        输出低频、平滑的空间调制特征，不参与分钟级建模。

        Args:
            hour_indices: list or tensor of integer hour indices [0, 720)
                         例如: [10, 11, 15] 表示 10h, 11h, 15h 三个整点

        Returns:
            tec_sequences: [N_hours, Seq_Len, 1, H, W] TEC 地图序列（归一化）
                          每个序列: [hour-seq_len+1, ..., hour]
        """
        # 转换为 tensor
        if isinstance(hour_indices, list):
            hour_indices = torch.tensor(hour_indices, dtype=torch.long, device=self.device)
        elif isinstance(hour_indices, torch.Tensor):
            hour_indices = hour_indices.long().to(self.device)
        else:
            raise TypeError(f"hour_indices 必须是 list 或 tensor，当前类型: {type(hour_indices)}")

        n_hours = hour_indices.shape[0]

        # 生成时间序列: [h - seq_len + 1, ..., h]
        offsets = torch.arange(self.seq_len, device=self.device).flip(0)  # [Seq]
        time_seq = hour_indices.unsqueeze(1) - offsets.unsqueeze(0)  # [N_hours, Seq]

        # 限制在有效范围内
        time_seq = time_seq.clamp(0, int(self.total_hours) - 1)  # [N_hours, Seq]

        # 从 tec_vol [1, 1, Time, Lat, Lon] 中提取地图
        tec_data = self.tec_vol.squeeze(0).squeeze(0)  # [Time, Lat, Lon]

        # 展平索引以便批量提取
        flat_indices = time_seq.view(-1)  # [N_hours * Seq]

        # 批量索引：[N_hours*Seq, Lat, Lon]
        flat_maps = tec_data[flat_indices]

        # 重塑为 [N_hours, Seq, Lat, Lon]
        tec_maps = flat_maps.view(n_hours, self.seq_len, self.target_h, self.target_w)

        # 添加通道维度：[N_hours, Seq, 1, Lat, Lon]
        tec_maps = tec_maps.unsqueeze(2)

        # 归一化到 [0, 1]
        tec_maps_norm = (tec_maps - self.min_tec) / self.denom
        tec_maps_norm = torch.clamp(tec_maps_norm, 0.0, 1.0)

        return tec_maps_norm
