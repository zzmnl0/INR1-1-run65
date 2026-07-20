"""
双尺度空间天气编码器（Dual-Scale Space Weather Encoder）

核心创新：可学习 EWMA 时间常数
    - τ_kp：Kp 指数时间常数，初始化 8h（环电流恢复时间尺度）
    - τ_solar：F10.7 指数时间常数，初始化 648h（27 天太阳自转周期）

动机：
    - 原始 Kp 数据由 3h 填充至 1h，存在阶跃误差
    - 原始 F10.7 数据按日存储，相邻时刻存在日间阶跃
    - EWMA 平滑消除阶跃，同时保留物理意义上的"记忆效应"
    - 可学习时间常数允许模型自适应最优平滑时间尺度

架构：
    Input: sw_seq [Batch, Seq, 2] — (Kp_norm, F10.7_norm)
        Seq 维度: [t-Seq+1, ..., t]（最新时刻在最后）

    Output:
        h_sw:    [Batch, sw_out_dim] — 融合 SW 上下文向量
        kp_eff:  [Batch]             — EWMA 有效 Kp 值
        f107_eff:[Batch]             — EWMA 有效 F10.7 值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DualScaleSWEncoder(nn.Module):
    """
    双尺度空间天气编码器

    分支一：Kp → EWMA(τ_kp) → LSTM_storm → z_storm [32]
    分支二：F10.7 → EWMA(τ_solar) → LSTM_solar → z_solar [32]
    融合：concat(z_storm, z_solar) → LayerNorm → Linear → h_sw [64]
    """

    def __init__(self, seq_len=168, sw_hidden_dim=32, sw_lstm_layers=2, sw_out_dim=64,
                 tau_kp_init=8.0, tau_solar_init=648.0):
        """
        Args:
            seq_len:         输入序列长度（时间步数）
            sw_hidden_dim:   每个分支 LSTM 隐层维度
            sw_lstm_layers:  Kp LSTM 层数（storm-scale 记忆更深）
            sw_out_dim:      输出维度（通常 = 2 * sw_hidden_dim）
            tau_kp_init:     Kp EWMA 时间常数初始值（小时）
            tau_solar_init:  F10.7 EWMA 时间常数初始值（小时）
        """
        super().__init__()
        self.seq_len = seq_len
        self.sw_hidden_dim = sw_hidden_dim

        # ==================== 可学习 EWMA 时间常数 ====================
        # 参数化：tau_kp    = exp(_log_tau_kp)    + 3.0  → init: log(tau_kp_init - 3)
        #         tau_solar  = exp(_log_tau_solar)  + 48.0 → init: log(tau_solar_init - 48)
        # 提高 offset 下限：强制保留物理滞后效应，防止网络将 tau 压至 0（"只看瞬时值"）
        # tau_kp    最小 3h（磁暴响应不可能在 3h 内归零）
        # tau_solar 最小 48h（F10.7 跨日变化，亚日平滑无物理意义）
        # 使用 exp 而非 softplus：对大初始值给出正确初始梯度
        self._log_tau_kp = nn.Parameter(
            torch.tensor(float(np.log(max(tau_kp_init - 3.0, 0.1)))))
        self._log_tau_solar = nn.Parameter(
            torch.tensor(float(np.log(max(tau_solar_init - 48.0, 0.1)))))

        # ==================== Storm-Scale LSTM（Kp 分支）====================
        # Kp 有较短记忆（磁暴快速演变），使用 2 层 LSTM
        self.lstm_storm = nn.LSTM(
            input_size=1,
            hidden_size=sw_hidden_dim,
            num_layers=sw_lstm_layers,
            batch_first=True,
            dropout=0.1 if sw_lstm_layers > 1 else 0.0
        )

        # ==================== Solar-Scale LSTM（F10.7 分支）====================
        # F10.7 变化缓慢，使用 1 层 LSTM
        self.lstm_solar = nn.LSTM(
            input_size=1,
            hidden_size=sw_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # ==================== 融合投影 ====================
        combined_dim = sw_hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, sw_out_dim),
            nn.Tanh()
        )

        # ==================== 多窗口统计融合 ====================
        # 拼接 EWMA-LSTM 输出与三个时窗均值（短/中/全期），捕获多时间尺度活动强度
        # 三窗: 最近8步 + 最近24步 + 全序列均值，各 [B,2] → 共 6D
        self.win_fusion = nn.Sequential(
            nn.LayerNorm(sw_out_dim + 6),
            nn.Linear(sw_out_dim + 6, sw_out_dim),
            nn.Tanh(),
        )

        # ==================== 缓存 dt 矩阵（方案 B）====================
        # _dt_lower[t,i] = max(t-i, 0)：下三角存真实距离，上三角钳为 0
        # _dt_mask[t,i]  = 1 if t>=i else 0：显式掩码，避免 exp(-inf/tau) 导致的 NaN 梯度
        # 注册为 buffer：随模型 .to(device) 自动迁移，不参与梯度计算
        t_idx = torch.arange(seq_len, dtype=torch.float32)
        i_idx = torch.arange(seq_len, dtype=torch.float32)
        dt = t_idx.unsqueeze(1) - i_idx.unsqueeze(0)    # [Seq, Seq]
        self.register_buffer('_dt_lower', dt.clamp(min=0.0))         # [Seq, Seq]
        self.register_buffer('_dt_mask',  (dt >= 0.0).float())        # [Seq, Seq]

        self._init_weights()

    def _init_weights(self):
        """初始化 LSTM 权重（正交初始化减少梯度消失）"""
        for lstm in [self.lstm_storm, self.lstm_solar]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # 遗忘门偏置初始化为 1（促进长程记忆）
                    n = param.shape[0]
                    param.data[n // 4:n // 2].fill_(1.0)

    @property
    def tau_kp(self):
        """Kp EWMA 时间常数（小时），硬下限 3h（磁暴响应不可能在 3h 内完全归零）
        参数化：tau = exp(log_tau_param) + 3，使 tau_init=8h 时 log_tau_param=log(5)≈1.61"""
        return torch.exp(self._log_tau_kp) + 3.0

    @property
    def tau_solar(self):
        """F10.7 EWMA 时间常数（小时），硬下限 48h（跨日平滑，亚日分量无物理意义）
        参数化：tau = exp(log_tau_param) + 48，使 tau_init=72h 时 log_tau_param=log(24)≈3.18"""
        return torch.exp(self._log_tau_solar) + 48.0

    def _ewma_smooth(self, seq, tau):
        """
        因果 EWMA 平滑（向量化实现，使用预缓存 dt 矩阵）

        在序列每个位置 t，计算 [0..t] 范围内的指数加权均值：
            smooth[t] = sum_{i=0}^{t} w_{t,i} * seq[i]
            w_{t,i} = exp(-(t-i)/tau) / sum_{j=0}^{t} exp(-(t-j)/tau)

        Args:
            seq: [Batch, Seq] 原始序列（oldest 在左）
            tau: scalar 时间常数（以时间步为单位）

        Returns:
            smoothed: [Batch, Seq] EWMA 平滑序列
        """
        # _dt_lower: 下三角为 t-i，上三角钳为 0（避免 exp(-inf/τ) 的 NaN 梯度）
        # _dt_mask:  下三角为 1，上三角为 0（显式掩盖因果之外的位置）
        W = torch.exp(-self._dt_lower / tau) * self._dt_mask  # [Seq, Seq]
        W = W / (W.sum(dim=1, keepdim=True) + 1e-8)           # 归一化每行

        # 矩阵乘法: smoothed[b, t] = sum_i seq[b, i] * W[t, i]
        smoothed = torch.einsum('bi,ti->bt', seq, W)   # [Batch, Seq]
        return smoothed

    def forward(self, sw_seq, mask=None):
        """
        前向传播

        Args:
            sw_seq: [Batch, Seq, 2] 空间天气序列
                    - Channel 0: Kp_norm   (归一化, (kp/8)*2-1 ∈ [-1,1])
                    - Channel 1: F107_norm (归一化, (f107-210)/60)
                    - 时间方向：sw_seq[:, 0, :] 最旧，sw_seq[:, -1, :] 最新
            mask: [Batch, Seq] 可选掩码（True 表示无效），保留兼容性

        Returns:
            h_sw:     [Batch, sw_out_dim] 融合 SW 上下文
            kp_eff:   [Batch] EWMA 有效 Kp（最新时刻的 EWMA 输出）
            f107_eff: [Batch] EWMA 有效 F10.7
        """
        if mask is not None:
            sw_seq = sw_seq.masked_fill(mask.unsqueeze(-1), 0.0)

        # 分离双通道
        kp_seq = sw_seq[:, :, 0]    # [Batch, Seq]
        f107_seq = sw_seq[:, :, 1]  # [Batch, Seq]

        # ==================== EWMA 平滑 ====================
        # tau 在 Seq 维度以"时间步"为单位，因此直接使用小时数
        kp_smoothed = self._ewma_smooth(kp_seq, self.tau_kp)       # [Batch, Seq]
        f107_smoothed = self._ewma_smooth(f107_seq, self.tau_solar)  # [Batch, Seq]

        # 提取最新时刻的 EWMA 有效值
        kp_eff = kp_smoothed[:, -1]    # [Batch]
        f107_eff = f107_smoothed[:, -1]  # [Batch]

        # ==================== LSTM 编码 ====================
        # LSTM 输入: EWMA 平滑后的序列，形状 [Batch, Seq, 1]
        _, (h_storm, _) = self.lstm_storm(kp_smoothed.unsqueeze(-1))
        z_storm = h_storm[-1]   # 取最后一层的最终隐状态 [Batch, hidden]

        _, (h_solar, _) = self.lstm_solar(f107_smoothed.unsqueeze(-1))
        z_solar = h_solar[-1]   # [Batch, hidden]

        # ==================== 融合投影 ====================
        combined = torch.cat([z_storm, z_solar], dim=-1)   # [Batch, 2*hidden]
        h_sw = self.fusion(combined)                        # [Batch, sw_out_dim]

        # ==================== 多窗口统计特征 ====================
        def _win_mean(seq, n):
            k = min(n, seq.size(1))
            return seq[:, -k:, :].mean(dim=1)              # [Batch, 2]

        multi_win = torch.cat([
            _win_mean(sw_seq, 8),
            _win_mean(sw_seq, 24),
            sw_seq.mean(dim=1),
        ], dim=-1)                                          # [Batch, 6]
        h_sw = self.win_fusion(torch.cat([h_sw, multi_win], dim=-1))

        return h_sw, kp_eff, f107_eff


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print('=' * 60)
    print('DualScaleSWEncoder 测试')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = DualScaleSWEncoder(
        seq_len=168, sw_hidden_dim=32, sw_lstm_layers=2, sw_out_dim=64,
        tau_kp_init=8.0, tau_solar_init=648.0
    ).to(device)

    print(f'参数量: {sum(p.numel() for p in encoder.parameters()):,}')
    print(f'初始 tau_kp:    {encoder.tau_kp.item():.2f} h')
    print(f'初始 tau_solar: {encoder.tau_solar.item():.2f} h')

    B = 32
    sw_seq = torch.randn(B, 168, 2).to(device)
    h_sw, kp_eff, f107_eff = encoder(sw_seq)

    print(f'\n输出形状:')
    print(f'  h_sw:    {h_sw.shape}')
    print(f'  kp_eff:  {kp_eff.shape}')
    print(f'  f107_eff:{f107_eff.shape}')

    loss = h_sw.sum()
    loss.backward()
    print(f'\n梯度测试 (tau_kp grad): {encoder._log_tau_kp.grad.item():.6f}')
    print('测试通过!')
