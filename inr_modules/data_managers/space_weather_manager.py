"""
空间天气数据管理器（Kp, F10.7）
支持时序窗口和零填充
"""
import torch
import numpy as np
import pandas as pd


class SpaceWeatherManager:
    """
    空间天气数据管理器
    
    功能:
    - 加载并预处理Kp/F10.7数据
    - 生成滑动时间窗口
    - 支持零填充处理
    """
    
    def __init__(self, txt_path, start_date_str, total_hours, seq_len, device):
        """
        Args:
            txt_path: 空间天气数据文件路径
            start_date_str: 起始日期字符串
            total_hours: 总时长（小时）
            seq_len: 时序窗口长度
            device: 计算设备
        """
        print(f"[SpaceWeatherManager] 加载数据 (窗口长度={seq_len})...")
        
        self.device = device
        self.total_hours = int(total_hours)
        self.seq_len = int(seq_len)
        
        # 加载并预处理数据
        df = self._load_and_process_data(txt_path, start_date_str)
        
        # 提取Kp和F10.7
        kp_col = 'Kp' if 'Kp' in df.columns else df.columns[1]
        f107_col = 'F10_7' if 'F10_7' in df.columns else df.columns[2]
        
        kp_raw = df[kp_col].fillna(0).values.astype(np.float32)
        f107_raw = df[f107_col].fillna(150.0).values.astype(np.float32)
        
        # 归一化: Kp -> [-1, 1], F10.7 -> 中心化
        kp_norm = (kp_raw / 8.0) * 2.0 - 1.0
        f107_norm = (f107_raw - 210.0) / 60.0
        
        # 转换为张量 [Total_Hours, 2]
        self.raw_data = torch.tensor(
            np.stack([kp_norm, f107_norm], axis=1),
            device=device, dtype=torch.float32
        )
        
        # 预计算滑动窗口
        self._build_sliding_windows()
        
        print(f"  空间天气管理器就绪. 窗口形状: {self.sw_windows.shape}")
        print(f"  时间0处Kp序列示例: {self.sw_windows[0, :, 0]}")
    
    def _load_and_process_data(self, txt_path, start_date_str):
        """加载并预处理CSV数据"""
        # 读取CSV
        try:
            df = pd.read_csv(txt_path, sep='\t')
            if 'Datetime' not in df.columns:
                df = pd.read_csv(txt_path, sep=r'\s+')
        except Exception as e:
            raise RuntimeError(f"读取CSV失败: {e}")
        
        # 解析时间
        if 'Datetime' in df.columns:
            df['dt'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        else:
            df['dt'] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # 清洗数据
        df = df.dropna(subset=['dt']).sort_values(by='dt')
        if df['dt'].duplicated().any():
            df = df.drop_duplicates(subset=['dt'], keep='first')
        
        # 时间范围截断与补全
        start_date = pd.to_datetime(start_date_str)
        end_date = start_date + pd.Timedelta(hours=self.total_hours - 1)
        df = df[(df['dt'] >= start_date) & (df['dt'] <= end_date)]
        
        # 重采样到小时分辨率
        full_time_grid = pd.date_range(start=start_date, periods=self.total_hours, freq='h')
        df = df.set_index('dt').reindex(full_time_grid, method='ffill')
        if df.iloc[0].isnull().any():
            df = df.bfill()
        
        return df
    
    def _build_sliding_windows(self):
        """构建滑动窗口（零填充）"""
        pad_size = self.seq_len - 1
        
        # 零填充（假设归一化后0代表均值状态）
        padding = torch.zeros(pad_size, 2, device=self.device)
        
        # 拼接: [Seq-1 + Total_Hours, 2]
        data_padded = torch.cat([padding, self.raw_data], dim=0)
        
        # Unfold创建滑动窗口: [Total_Hours, Seq_Len, 2]
        self.sw_windows = data_padded.unfold(0, self.seq_len, 1).permute(0, 2, 1).contiguous()
    
    def get_drivers_sequence(self, time_batch):
        """
        获取空间天气驱动序列
        
        Args:
            time_batch: [Batch] 时间（小时）
            
        Returns:
            [Batch, Seq_Len, 2] 历史序列 (Kp, F10.7)
        """
        # 向下取整获取索引
        time_idx = torch.floor(time_batch).long()
        
        # 边界保护
        time_idx = torch.clamp(time_idx, 0, self.total_hours - 1)
        
        # 直接索引预计算好的窗口
        return self.sw_windows[time_idx]
