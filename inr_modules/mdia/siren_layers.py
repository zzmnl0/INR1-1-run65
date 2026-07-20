"""
SIREN (Sinusoidal Representation Networks) 基础层

Reference:
    Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
    NEURIPS 2020

核心特性:
    1. 使用 sin 激活函数替代 ReLU
    2. 特殊的权重初始化策略确保梯度稳定性
    3. 适合学习高频细节和周期性现象
"""

import torch
import torch.nn as nn
import numpy as np


class SIRENLayer(nn.Module):
    """
    SIREN 单层：线性投影 + sin 激活

    特殊之处：
        - 权重初始化遵循 SIREN 论文的均匀分布策略
        - 第一层和后续层使用不同的初始化范围
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            bias: 是否使用偏置
            is_first: 是否为网络第一层（初始化策略不同）
            omega_0: 频率因子（论文推荐 30）
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """
        SIREN 初始化策略：

        第一层：
            W ~ Uniform(-1/n_in, 1/n_in)

        后续层：
            W ~ Uniform(-sqrt(6/n_in)/omega_0, sqrt(6/n_in)/omega_0)

        偏置：
            b ~ Uniform(-sqrt(1/n_in), sqrt(1/n_in))
        """
        with torch.no_grad():
            if self.is_first:
                # 第一层：均匀分布 [-1/n, 1/n]
                bound = 1 / self.linear.in_features
            else:
                # 后续层：考虑 omega_0
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)

            if self.linear.bias is not None:
                # 偏置：与输入维度相关
                bias_bound = np.sqrt(1 / self.linear.in_features)
                self.linear.bias.uniform_(-bias_bound, bias_bound)

    def forward(self, x):
        """
        前向传播：sin(omega_0 * Wx + b)

        注意：只有第一层应用 omega_0 频率缩放
        """
        if self.is_first:
            return torch.sin(self.omega_0 * self.linear(x))
        else:
            return torch.sin(self.linear(x))


class SIRENNet(nn.Module):
    """
    多层 SIREN 网络

    构建策略：
        - 第一层标记 is_first=True
        - 中间层使用 sin 激活
        - 最后一层可选择线性输出或 sin 输出
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=False, omega_0=30.0, omega_hidden=30.0):
        """
        Args:
            in_features: 输入维度
            hidden_features: 隐层维度
            hidden_layers: 隐层数量
            out_features: 输出维度
            outermost_linear: 最后一层是否使用线性（不使用 sin）
            omega_0: 第一层频率因子
            omega_hidden: 后续层频率因子（通常与 omega_0 相同）
        """
        super().__init__()

        self.net = []

        # 第一层（特殊初始化）
        self.net.append(SIRENLayer(in_features, hidden_features,
                                    is_first=True, omega_0=omega_0))

        # 中间层
        for _ in range(hidden_layers):
            self.net.append(SIRENLayer(hidden_features, hidden_features,
                                        is_first=False, omega_0=omega_hidden))

        # 最后一层
        if outermost_linear:
            # 线性输出（用于回归任务）
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / omega_hidden
                final_linear.weight.uniform_(-bound, bound)
                if final_linear.bias is not None:
                    final_linear.bias.uniform_(-bound, bound)
            self.net.append(final_linear)
        else:
            # sin 输出
            self.net.append(SIRENLayer(hidden_features, out_features,
                                        is_first=False, omega_0=omega_hidden))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [Batch, in_features]

        Returns:
            [Batch, out_features]
        """
        return self.net(x)


class ModulatedSIRENNet(nn.Module):
    """
    可调制的 SIREN 网络（用于 R-STMRF）

    特点：
        - SIREN 生成基函数特征
        - 支持外部调制信号（FiLM-style）
        - 用于 Spatial/Temporal Basis Network
    """

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, omega_0=30.0):
        """
        Args:
            in_features: 输入坐标维度
            hidden_features: 隐层维度
            hidden_layers: 隐层数量
            out_features: 基函数维度
            omega_0: 频率因子
        """
        super().__init__()

        self.siren = SIRENNet(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            outermost_linear=False,  # 保持 sin 输出
            omega_0=omega_0
        )

    def forward(self, coords):
        """
        生成基函数特征（未调制）

        Args:
            coords: [Batch, in_features] 归一化坐标

        Returns:
            [Batch, out_features] 基函数特征
        """
        return self.siren(coords)


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("="*60)
    print("SIREN 层测试")
    print("="*60)

    # 测试单层
    layer = SIRENLayer(3, 64, is_first=True, omega_0=30.0)
    x = torch.randn(128, 3)
    out = layer(x)
    print(f"\n单层测试:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 测试多层网络
    net = SIRENNet(
        in_features=3,
        hidden_features=128,
        hidden_layers=3,
        out_features=64,
        outermost_linear=False
    )
    out_net = net(x)
    print(f"\n多层网络测试:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {out_net.shape}")
    print(f"  参数量: {sum(p.numel() for p in net.parameters()):,}")

    # 测试梯度流动
    loss = out_net.sum()
    loss.backward()
    print(f"\n梯度测试:")
    print(f"  第一层梯度范数: {layer.linear.weight.grad.norm().item():.6f}")

    # 测试可调制网络
    mod_net = ModulatedSIRENNet(
        in_features=5,  # (Lat, Lon, Alt, sin_lt, cos_lt)
        hidden_features=128,
        hidden_layers=3,
        out_features=64
    )
    coords = torch.randn(256, 5)
    basis = mod_net(coords)
    print(f"\n可调制网络测试:")
    print(f"  输入: {coords.shape}")
    print(f"  基函数: {basis.shape}")

    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)
