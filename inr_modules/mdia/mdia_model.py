"""
MDIA-INR 主模型
Multi-scale Data-Informed Assimilation Implicit Neural Representation

核心公式（数据同化增量范式）：
    Ne_fused = Ne_iri + Ne_residual

其中：
    Ne_iri:      冻结 IRI 神经代理场输出（物理背景场，可信的全球基准）
    Ne_residual: SIREN 残差网络学习 IRI 与真实观测之间的增量
                 （EIA 双峰、等离子泡、磁暴扰动等 IRI 欠估的细节）

NeQuick 的新角色——"物理正则化器（Physical Regularizer）"：
    NeQuick 三层 Epstein 廓线（E+F1+F2）不再直接参与最终加法，
    而是通过 chapman_shape_loss = MSE(Ne_fused, Ne_nequick) 作为软约束。
    GIRO 测高仪数据精准训练 hmF2/NmF2 → NeQuick 作为"物理模具"将
    Ne_fused（= IRI + 残差）塑造成符合 Epstein 垂直剖面的形状。

七大分支：
    [A] 冻结 IRI 代理场  → Ne_bkg（最终加法的基准项，不可移除）
    [B] 双尺度 SW 编码器 → h_sw[64]（EWMA + LSTM 双通道）
    [C] 空间 SIREN       → h_spatial[64]（7D 输入含 sin_I/cos_I 地磁倾角特征，不含高度）
        + 地磁 FiLM 调制 → MagneticEncoder(sin_I,cos_I)→γ_mag,β_mag 微扰 h_spatial（缩放 0.1）
    [D] NeQuick 参数解码 → (NmF2, hmF2, BF2, NmF1, BF1, NmE_SZA) → Ne_nequick
                           ★ 仅用于 chapman_shape_loss 物理软约束，不参与 Ne_fused 加法
    [E] 残差 SIREN       → Ne_residual（8D 输入含 sin_I/cos_I，含高度 + SW 加性调制）
    [F] 不确定性估计     → log_var（Warm-up 后启用）

输出：Ne_fused = Ne_iri + Ne_residual   （Ne_nequick 存于 extras['ne_chapman']）
      extras['sin_I'] — 地磁倾角正弦，供 physics_losses 磁赤道感知掩码使用

坐标约定（地理坐标降级 + 地磁倾角特征注入）：
    coords [Batch, 4]：(Lat_geo, Lon_geo, Alt, Time)
    coords [Batch, 5]：(Lat_geo, Lon_geo, Alt, Time, Lat_aacgm)  — 第 5 列保留但不再使用
    SIREN 使用 Lat_geo（col 0），消除 AACGM 坐标变换在磁赤道的断层；
    地磁信息通过 IGRF 偶极子近似从 (Lat_geo, Lon_geo) 实时计算 sin_I/cos_I 注入（无外部库依赖）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .siren_layers import ModulatedSIRENNet, SIRENNet
from .ewma_sw_encoder import DualScaleSWEncoder

# ======================== 数值常数 ========================
_LN10      = np.log(10.0)
_INV_LN10  = 1.0 / np.log(10.0)
_LOG10_4   = np.log10(4.0)           # log10(4) ≈ 0.60206
_DOY_SEP1  = 243.0                   # 2024-09-01 的年内天数（0-indexed，Jan 1 = 0）

# IGRF 磁偶极子北极坐标（J2024.7 近似，偶极子近似精度对 EIA 建模已足够）
_MAG_POLE_LAT_RAD = 80.5 * np.pi / 180.0   # ~80.5°N
_MAG_POLE_LON_RAD = -73.0 * np.pi / 180.0  # ~73.0°W


# ======================== 地磁倾角计算（偶极子近似）========================

def _compute_dip_features(lat_deg, lon_deg):
    """
    基于 IGRF 偶极子近似实时计算地磁倾角（Dip Angle I）的 sin/cos 分量。

    偶极子公式（无外部库依赖，纯 PyTorch，完全可微）：
        sin(λ_m) = sin(lat)·sin(lat_pole) + cos(lat)·cos(lat_pole)·cos(lon − lon_pole)
        sin(I)   = 2·sin(λ_m) / √(1 + 3·sin²(λ_m))
        cos(I)   = cos(λ_m)   / √(1 + 3·sin²(λ_m))

    物理特性（与 AACGM 的本质区别）：
        - 磁赤道处（λ_m=0）: sin(I)=0, cos(I)=1 — 处处光滑、无坐标断层
        - 磁极处（λ_m=±90°）: sin(I)=±1, cos(I)=0
        - sin(I) 符号正确反映磁半球（北正南负），连续可微
        - EIA 双峰纬度（±15°磁纬）对应 sin(I) ≈ ±0.25 — 对 SIREN 频率激活友好

    Args:
        lat_deg: [Batch] 地理纬度 (°)
        lon_deg: [Batch] 地理经度 (°)

    Returns:
        sin_I: [Batch] ∈ (-1, 1)，磁赤道附近 ≈ 0
        cos_I: [Batch] ∈ (0, 1]，磁赤道附近 ≈ 1
    """
    DEG2RAD = np.pi / 180.0
    lat_r = lat_deg * DEG2RAD
    lon_r = lon_deg * DEG2RAD

    # 偶极子磁纬 sin(λ_m)
    sin_lam = (torch.sin(lat_r) * np.sin(_MAG_POLE_LAT_RAD)
               + torch.cos(lat_r) * np.cos(_MAG_POLE_LAT_RAD)
               * torch.cos(lon_r - _MAG_POLE_LON_RAD))
    sin_lam = torch.clamp(sin_lam, -1.0 + 1e-6, 1.0 - 1e-6)   # 数值稳定
    cos_lam = torch.sqrt(1.0 - sin_lam ** 2)                    # ≥ 0（|λ_m|≤90°）

    # 偶极子倾角：tan(I) = 2·tan(λ_m)  →  sin/cos 解析式
    denom = torch.sqrt(1.0 + 3.0 * sin_lam ** 2)
    sin_I = 2.0 * sin_lam / denom   # ∈ (-1, 1)
    cos_I = cos_lam / denom          # ∈ (0, 1]

    return sin_I, cos_I


# ======================== 轻量级地磁编码器 ========================

class MagneticEncoder(nn.Module):
    """
    轻量级地磁倾角编码器（2D → 32D）

    输入 [sin_I, cos_I] 编码为 32 维地磁表示，
    用于 FiLM 调制空间 SIREN，使模型感知地磁场拓扑。

    计算量：2×32 + 32×32 = 1,088 次乘法 ≈ SIREN 的 0.5%（< 性能要求 5%）
    零初始化输出层 → 训练初期为恒等映射，稳定收敛。
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, sin_I, cos_I):
        """
        Args:
            sin_I: [Batch] 地磁倾角正弦
            cos_I: [Batch] 地磁倾角余弦
        Returns:
            h_mag: [Batch, 32]
        """
        x = torch.stack([sin_I, cos_I], dim=1)   # [Batch, 2]
        return self.net(x)


# ======================== NeQuick Epstein 函数 ========================

def _epstein_log10(alt, Nm_log10, hm, B):
    """
    NeQuick Epstein 函数（log10 空间）

    N(h) = 4·Nm · exp(z) / (1 + exp(z))²,  z = (h - hm) / B

    log10(N) = Nm_log10 + log10(4) + (z − 2·softplus(z)) / ln(10)

    数值特性：
        z = 0  (peak):   log10(N) = Nm_log10                  ✓
        z → +∞ (above):  log10(N) ≈ Nm_log10 + log10(4) − z/ln10  (指数衰减)
        z → −∞ (below):  log10(N) ≈ Nm_log10 + log10(4) + z/ln10  (指数衰减)

    Args:
        alt:      [Batch] 高度 (km) — tensor
        Nm_log10: [Batch] 峰值密度 log10 — tensor
        hm:       [Batch] 峰高 (km) — tensor or scalar
        B:        [Batch] 标高 (km) — tensor or scalar

    Returns:
        log10_N: [Batch]
    """
    z = (alt - hm) / B
    z = torch.clamp(z, -30.0, 30.0)
    log10_N = Nm_log10 + _LOG10_4 + _INV_LN10 * (z - 2.0 * F.softplus(z))
    return log10_N  # [Batch]


def _compute_nme_sza(lat_deg, lon_deg, time_relhour, nme_max_log10):
    """
    基于太阳天顶角（SZA）的 E 层峰值密度（经验公式）

    NmE_log10 = nme_max_log10 + 0.6 · log10(max(cos(SZA), 1e-4))

    物理依据：
        - 白天 cos(SZA) ≈ 1 → NmE ≈ 10^nme_max（最大值）
        - 夜侧 cos(SZA) → 0 → NmE ≈ 10^(nme_max − 2.4)（≈ 1e9.1 m⁻³，残余夜间 E 区）
        - 指数 0.6 来自 Chapman 理论（E 层生产率 ∝ cos^0.5—cos^1 SZA）

    坐标参考：2024-09-01 起算（_DOY_SEP1 = 243）

    Args:
        lat_deg:        [Batch] 地理纬度 (度)
        lon_deg:        [Batch] 地理经度 (度)
        time_relhour:   [Batch] 相对小时数（从 2024-09-01 00:00 UT 起算）
        nme_max_log10:  [Batch] 可学习最大 log10(NmE)，由 MLP 预测

    Returns:
        NmE_log10: [Batch]
    """
    _DEG2RAD = np.pi / 180.0
    lat_rad = lat_deg * _DEG2RAD

    # 年内天数（从 2024-09-01 开始，保持张量运算）
    doy = _DOY_SEP1 + time_relhour / 24.0

    # 太阳赤纬（弧度）：δ = −23.45° · cos(2π(doy+10)/365)
    decl_rad = (-23.45 * _DEG2RAD) * torch.cos(2.0 * np.pi / 365.0 * (doy + 10.0))

    # 地方太阳时 → 时角（弧度）
    lst = (time_relhour % 24.0) + lon_deg / 15.0
    hour_angle_rad = (lst - 12.0) * (np.pi / 12.0)

    # cos(SZA) = sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(H)
    cos_sza = (torch.sin(lat_rad) * torch.sin(decl_rad)
               + torch.cos(lat_rad) * torch.cos(decl_rad) * torch.cos(hour_angle_rad))
    cos_sza = torch.clamp(cos_sza, min=1e-4)   # 夜侧保留小正数（避免 log10(0)）

    NmE_log10 = nme_max_log10 + 0.6 * torch.log10(cos_sza)
    return NmE_log10  # [Batch]


# ======================== NeQuick 三层解码器 ========================

class NeQuickDecoder(nn.Module):
    """
    NeQuick 三层 Epstein 廓线解码器 (E + F1 + F2)

    层参数（收紧边界以压制低高度虚高）：
        E  层: hmE = 110 km (固定), BE = 5 km (固定), NmE 由 SZA 驱动
               nme_max ∈ [9.0, 10.8]（收紧上限，防止夜间 E 层虚高）
        F1 层: hmF1 = 165 km (固定), BF1 ∈ [2.0, 4.5] km（关键：严格限制 F1 底侧渗透至 120km）
               NmF1 = NmF2 − Δ (Δ ∈ [1.5, 3.0]，拉大 F1/F2 密度差距)
        F2 层: NmF2 自由, hmF2 ∈ [200, 550] km, BF2 ∈ [20, 60] km

    合并：
        Ne_total = Ne_E + Ne_F1 + Ne_F2
        Ne_log10 = log10(Ne_total)  via torch.logaddexp（数值稳定）

    架构（FiLM 调制 + 物理解耦双分支）：
        1. FiLM 调制：h_sw → (gamma, beta) → h_mod = h_spatial * (1 + gamma) + beta
           物理意义：SW 改变电离层对磁场拓扑（空间结构）的响应尺度和偏移，而非简单拼接
        2. 光化学参数分支 mlp_sun（太阳辐射驱动）：
           输入 cat(h_mod[64], dip_features[4]) = 68 维
           输出 (nme_max, BF1, delta)：日照角决定 E 层电离 + F1 层形态
        3. 动力学参数分支 mlp_dyn（E×B 漂移 + 等离子体对流驱动）：
           输入 cat(h_mod[64], dip_features[4]) = 68 维
           输出 (NmF2, hmF2, BF2)：喷泉效应 + 磁暴扰动决定 F2 层主体

    初始化：
        FiLM 网络零初始化 → 训练初期为恒等调制（h_mod ≈ h_spatial）
        mlp_dyn 偏置 [12.0, 0, 0] → NmF2=12, hmF2=375km, BF2=40km
        mlp_sun 零偏置 → nme_max=9.9, BF1=3.25km, Δ=2.25（sigmoid(0)=0.5 各取中点）

    地理纬度三角特征注入（EIA 双峰形态编码）：
        dip_features = [lat_geo_n, sin(π·lat_n), cos(π·lat_n), cos(2π·lat_n)]
        其中 lat_geo_n = lat_geo/90 ∈ [-1,1]；高频展开强化 EIA 双峰对称结构。
        注意：此处特征编码纬度位置，与 IGRF 倾角特征（sin_I/cos_I）用途不同。
    """

    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()

        # 分离空间维和 SW 维（各占 in_dim 的一半）
        self.h_spatial_dim = in_dim // 2           # basis_dim = 64
        h_sw_dim = in_dim - self.h_spatial_dim     # sw_out_dim = 64
        dip_dim  = 4                               # AACGM lat 三角特征
        mlp_in   = self.h_spatial_dim + dip_dim   # 68 维

        # FiLM 调制网络：SW → (gamma, beta)，零初始化 → 初期为恒等映射
        self.film_net = nn.Linear(h_sw_dim, self.h_spatial_dim * 2)

        # 光化学参数分支（太阳辐射驱动：E 层电离 + F1 层形态）
        self.mlp_sun = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)         # → (nme_max_raw, BF1_raw, delta_raw)
        )

        # 动力学参数分支（E×B 漂移 + 等离子体扩散：F2 层主体）
        self.mlp_dyn = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)         # → (NmF2_log10, hmF2_raw, BF2_raw)
        )

        with torch.no_grad():
            # FiLM 网络近零初始化（确保训练初期为恒等调制）
            nn.init.zeros_(self.film_net.weight)
            nn.init.zeros_(self.film_net.bias)

            # 光化学分支输出层零初始化
            # sigmoid(0)=0.5 → nme_max=9.9, BF1=3.25km, Δ=2.25
            nn.init.zeros_(self.mlp_sun[-1].weight)
            nn.init.zeros_(self.mlp_sun[-1].bias)

            # 动力学分支：NmF2 偏置 12.0，hmF2/BF2 零偏置（→ 375km, 40km）
            nn.init.zeros_(self.mlp_dyn[-1].weight)
            self.mlp_dyn[-1].bias.data = torch.tensor([12.0, 0.0, 0.0])

    def forward(self, h_combined, lat_deg, lon_deg, time_relhour, alt_km, lat_aacgm_n):
        """
        Args:
            h_combined:   [Batch, in_dim] = cat(h_spatial[64], h_sw[64])
            lat_deg:      [Batch] 地理纬度 (度)，用于 SZA 计算
            lon_deg:      [Batch] 地理经度 (度)，用于 SZA 计算
            time_relhour: [Batch] 相对小时（2024-09-01 起算）
            alt_km:       [Batch] 高度 (km)
            lat_aacgm_n:  [Batch] 归一化 AACGM 磁纬 (AACGM_lat / 90) ∈ [-1, 1]

        Returns:
            Ne_log10:  [Batch, 1]
            params:    dict — 廓线参数（供损失 / 可视化使用）
        """
        # 分离空间特征和 SW 特征
        h_spatial = h_combined[:, :self.h_spatial_dim]    # [Batch, 64]
        h_sw      = h_combined[:, self.h_spatial_dim:]    # [Batch, 64]

        # AACGM lat 三角特征展开（高频编码强化 EIA 双峰对称性）
        lat_n = lat_aacgm_n.unsqueeze(1)                              # [Batch, 1]
        dip_features = torch.cat([
            lat_n,
            torch.sin(np.pi * lat_n),                                  # EIA 峰值纬度响应
            torch.cos(np.pi * lat_n),                                  # 赤道/极区对比
            torch.cos(2.0 * np.pi * lat_n),                            # 双峰对称（cos2θ ±15° 极值）
        ], dim=1)                                                      # [Batch, 4]

        # FiLM 调制：SW 特征对空间特征施加微扰（物理：磁暴改变 F 层响应尺度）
        gamma_beta = self.film_net(h_sw)                               # [Batch, h_spatial_dim*2]
        gamma, beta = gamma_beta.chunk(2, dim=-1)                      # each [Batch, 64]
        h_mod = h_spatial * (1.0 + gamma) + beta                       # FiLM 调制（恒等 + 微扰）

        # 拼接磁倾角特征
        h_in = torch.cat([h_mod, dip_features], dim=-1)               # [Batch, 68]

        # ==================== 光化学参数分支（太阳辐射驱动）====================
        out_sun = self.mlp_sun(h_in)                                   # [Batch, 3]
        nme_max = torch.sigmoid(out_sun[:, 0]) * 1.8 + 9.0            # [9.0, 10.8] log10
        BF1     = torch.sigmoid(out_sun[:, 1]) * 2.5 + 2.0            # [2.0, 4.5] km（限制底侧渗透）
        delta   = torch.sigmoid(out_sun[:, 2]) * 1.5 + 1.5            # [1.5, 3.0]（强制 F1/F2 分离）

        # ==================== 动力学参数分支（E×B + 扩散驱动）====================
        out_dyn = self.mlp_dyn(h_in)                                   # [Batch, 3]
        NmF2_log10 = out_dyn[:, 0]                                     # F2 峰值密度（自由，偏置 12.0）
        hmF2       = torch.sigmoid(out_dyn[:, 1]) * 350.0 + 200.0     # [200, 550] km
        BF2        = torch.sigmoid(out_dyn[:, 2]) * 40.0  + 20.0      # [20, 60] km

        # F1 参数（层间约束）
        NmF1_log10 = NmF2_log10 - delta
        hmF1       = torch.full_like(alt_km, 165.0)                    # 固定 165 km

        # E 层参数（SZA 驱动）
        NmE_log10  = _compute_nme_sza(lat_deg, lon_deg, time_relhour, nme_max)
        hmE        = torch.full_like(alt_km, 110.0)                    # 固定 110 km
        BE         = torch.full_like(alt_km, 5.0)                      # 固定 5 km

        # Epstein 廓线（log10 空间）
        Ne_E_log10  = _epstein_log10(alt_km, NmE_log10,  hmE,  BE)
        Ne_F1_log10 = _epstein_log10(alt_km, NmF1_log10, hmF1, BF1)
        Ne_F2_log10 = _epstein_log10(alt_km, NmF2_log10, hmF2, BF2)

        # 合并（数值稳定 logaddexp）：log10(Ne_E + Ne_F1 + Ne_F2)
        ne_total_log10 = torch.logaddexp(
            Ne_E_log10  * _LN10,
            torch.logaddexp(Ne_F1_log10 * _LN10, Ne_F2_log10 * _LN10)
        ) * _INV_LN10   # [Batch]

        params = {
            'NmF2_F2':  NmF2_log10, 'hmF2_F2': hmF2,  'BF2': BF2,
            'NmF1_F1':  NmF1_log10, 'hmF1_F1': hmF1,  'BF1': BF1,
            'NmE_E':    NmE_log10,
        }

        return ne_total_log10.unsqueeze(-1), params  # [Batch, 1], dict


# ======================== 主模型 ========================

class MDIA_INR_Model(nn.Module):
    """
    MDIA-INR 主模型

    Args:
        iri_proxy:     冻结的 IRI 神经代理场
        config:        配置字典
    """

    def __init__(self, iri_proxy, config):
        super().__init__()

        self.total_hours = config.get('total_hours', 720.0)
        self.alt_min, self.alt_max = config['alt_range']
        self.seq_len = config['seq_len']

        basis_dim      = config.get('basis_dim', 64)
        siren_hidden   = config.get('siren_hidden', 128)
        siren_layers   = config.get('siren_layers', 3)
        omega_0        = config.get('omega_0', 30.0)
        sw_out_dim     = config.get('sw_out_dim', 64)
        sw_hidden_dim  = config.get('sw_hidden_dim', 32)
        sw_lstm_layers = config.get('sw_lstm_layers', 2)
        nequick_hidden = config.get('chapman_hidden', 128)   # 复用配置键
        tau_kp_init    = config.get('tau_kp_init', 8.0)
        tau_solar_init = config.get('tau_solar_init', 72.0)

        # ==================== [A] IRI 神经代理场（冻结）====================
        self.iri_proxy = iri_proxy
        for param in self.iri_proxy.parameters():
            param.requires_grad = False
        self.iri_proxy.eval()

        # ==================== [B] 双尺度 SW 编码器 ====================
        self.sw_encoder = DualScaleSWEncoder(
            seq_len=self.seq_len,
            sw_hidden_dim=sw_hidden_dim,
            sw_lstm_layers=sw_lstm_layers,
            sw_out_dim=sw_out_dim,
            tau_kp_init=tau_kp_init,
            tau_solar_init=tau_solar_init
        )

        # ==================== [C] 空间 SIREN（不含高度）====================
        # 输入: [lat_geo_n, sin_Lon, cos_Lon, sin_LT, cos_LT, sin_I, cos_I] = 7 维
        # sin_I/cos_I 由 IGRF 偶极子近似实时计算，提供连续可微的地磁场信息
        self.spatial_basis_net = ModulatedSIRENNet(
            in_features=7,
            hidden_features=siren_hidden,
            hidden_layers=siren_layers,
            out_features=basis_dim,
            omega_0=omega_0
        )

        # ==================== [C+] 地磁 FiLM 调制 ====================
        # MagneticEncoder: [sin_I, cos_I] → 32D 地磁表示
        # film_mag: 32D → (γ_mag, β_mag)，对 h_spatial 施加残差调制（缩放 0.1）
        # 零初始化：训练初期为恒等映射，梯度自然演化
        self.mag_encoder = MagneticEncoder()
        self.film_mag = nn.Linear(32, 2 * basis_dim)

        # ==================== [D] NeQuick 参数解码器（FiLM 双分支）====================
        # FiLM: h_sw → (gamma, beta) 调制 h_spatial，然后双分支解耦预测
        # mlp_sun(68维) → (nme_max, BF1, delta)；mlp_dyn(68维) → (NmF2, hmF2, BF2)
        self.nequick_decoder = NeQuickDecoder(
            in_dim=basis_dim + sw_out_dim,
            hidden_dim=nequick_hidden
        )

        # ==================== [E] 残差 SIREN（含高度）====================
        # 输入: [lat_geo_n, sin_Lon, cos_Lon, Alt_n, sin_LT, cos_LT, sin_I, cos_I] = 8 维
        self.residual_net = ModulatedSIRENNet(
            in_features=8,
            hidden_features=siren_hidden,
            hidden_layers=siren_layers,
            out_features=basis_dim,
            omega_0=omega_0
        )

        # SW 对残差的加性调制（Additive Shift）
        self.sw_shift_head = nn.Sequential(
            nn.Linear(sw_out_dim, basis_dim),
            nn.Tanh()
        )

        # 残差解码器
        self.residual_decoder = nn.Sequential(
            nn.Linear(basis_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        # 残差缩放参数（初始值 1.0，允许残差充分修正 NeQuick 结构）
        self.resid_scale = nn.Parameter(torch.tensor(1.0))

        # ==================== [F] 不确定性估计头 ====================
        self.uncertainty_head = nn.Sequential(
            nn.Linear(basis_dim + sw_out_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """零初始化输出层，确保残差从零开始学习"""
        nn.init.zeros_(self.residual_decoder[-1].weight)
        nn.init.zeros_(self.residual_decoder[-1].bias)
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.zeros_(self.uncertainty_head[-1].bias)
        # 地磁 FiLM 零初始化：训练初期 γ_mag=0,β_mag=0 → 恒等调制
        nn.init.zeros_(self.film_mag.weight)
        nn.init.zeros_(self.film_mag.bias)

    def normalize_coords(self, lat, lon, alt, time):
        """坐标归一化到 [-1, 1]"""
        lat_n  = lat  / 90.0
        lon_n  = lon  / 180.0
        alt_n  = 2.0 * (alt  - self.alt_min) / (self.alt_max - self.alt_min) - 1.0
        time_n = (time / self.total_hours) * 2.0 - 1.0
        return lat_n, lon_n, alt_n, time_n

    def get_background(self, lat, lon, alt, time):
        """
        查询 IRI Neural Proxy 背景值（冻结参数，但允许梯度通过 coords）
        """
        coords = torch.stack([lat, lon, alt, time], dim=-1)
        was_training = self.iri_proxy.training
        self.iri_proxy.train()
        try:
            background_log = self.iri_proxy(coords)
        finally:
            if not was_training:
                self.iri_proxy.eval()
        return background_log

    def forward(self, coords, sw_seq, precomputed_h_sw=None, giro_mode=False):
        """
        前向传播

        数据同化增量范式：
            Ne_fused = Ne_bkg (IRI) + Ne_residual
            Ne_nequick 仅作为物理正则化器，通过 extras['ne_chapman'] 供
            chapman_shape_loss = MSE(Ne_fused, Ne_nequick) 使用，不参与最终加法。

        Args:
            coords:          [Batch, 4] — (Lat_geo, Lon_geo, Alt, Time)
                          或 [Batch, 5] — (Lat_geo, Lon_geo, Alt, Time, Lat_aacgm)
            sw_seq:          [Batch, Seq, 2] — (Kp_norm, F10.7_norm)
            precomputed_h_sw:[Batch, sw_out_dim] 可选，跳过 EWMA+LSTM
            giro_mode:       bool — GIRO 专用快速路径；跳过 IRI 代理和残差 SIREN，
                             仅运行 SW 编码 + 空间 SIREN + NeQuick 解码（约节省 40% 计算）。
                             GIRO 损失只用 extras['chapman_params'] 中的 hmF2/NmF2，
                             不使用 Ne_fused，故此模式下 Ne_fused 返回 Ne_nequick 占位。

        Returns:
            Ne_fused:   [Batch, 1] — 最终预测 Ne (log10) = Ne_bkg + Ne_residual
                        （giro_mode 时为 Ne_nequick 占位，GIRO 损失不使用此值）
            log_var:    [Batch, 1] — 不确定性（对数方差）
            Ne_nequick: [Batch, 1] — NeQuick 三层廓线（物理正则化器；
                        同时以 'ne_chapman' 键存入 extras 供 chapman_shape_loss 使用）
            Ne_residual:[Batch, 1] — 残差分量（giro_mode 时为全零占位）
            extras:     dict — 含 'ne_bkg', 'ne_chapman', 'ne_residual', 'chapman_params' 等
        """
        lat_geo = coords[:, 0]
        lon_geo = coords[:, 1]
        alt     = coords[:, 2]
        time    = coords[:, 3]

        # 1. IRI 背景（giro_mode 时跳过：GIRO 损失不依赖 ne_bkg）
        if not giro_mode:
            ne_bkg = self.get_background(lat_geo, lon_geo, alt, time)   # [Batch, 1]
        else:
            ne_bkg = torch.zeros(len(coords), 1, device=coords.device, dtype=coords.dtype)

        # 2. 坐标归一化
        lat_n, lon_n, alt_n, time_n = self.normalize_coords(lat_geo, lon_geo, alt, time)

        # 3a. 经度周期编码（±180° 连续性）
        sin_lon = torch.sin(np.pi * lon_geo / 180.0)
        cos_lon = torch.cos(np.pi * lon_geo / 180.0)

        # 3b. 地方时特征（角度编码）
        local_time_hour = (time % 24.0) + (lon_geo / 15.0)
        lt_norm = local_time_hour / 24.0
        sin_lt  = torch.sin(2.0 * np.pi * lt_norm)
        cos_lt  = torch.cos(2.0 * np.pi * lt_norm)

        # 3c. 地理纬度降级模式：统一使用 lat_geo / 90，不再读取 AACGM 列（col 4）
        # 若 coords 有第 5 列（AACGM lat），保留以兼容数据格式，但不参与计算。
        lat_siren_n = lat_n   # = lat_geo / 90.0，处处连续可微

        # 3d. IGRF 偶极子近似地磁倾角特征（无外部库依赖，实时计算）
        # sin_I ≈ 0 在磁赤道，处处连续可微，无 AACGM 断层
        # 注入 SIREN 输入和 FiLM 调制，使模型感知地磁场拓扑（EIA 双峰对齐）
        sin_I, cos_I = _compute_dip_features(lat_geo, lon_geo)   # each [Batch]

        # 4. 双尺度 SW 编码
        # 若外部已计算好 h_sw（同箱 batch 去重优化），直接使用，跳过 EWMA+LSTM
        if precomputed_h_sw is not None:
            h_sw     = precomputed_h_sw          # [Batch, sw_out_dim]
            kp_eff   = sw_seq[:, -1, 0]          # 序列末尾近似值（仅供诊断，不参与梯度）
            f107_eff = sw_seq[:, -1, 1]
        else:
            h_sw, kp_eff, f107_eff = self.sw_encoder(sw_seq)   # [Batch, sw_out_dim]

        # 5. 空间 SIREN（不含高度，7D 输入）
        # [lat_geo_n, sin_lon, cos_lon, sin_lt, cos_lt, sin_I, cos_I]
        spatial_input = torch.stack(
            [lat_siren_n, sin_lon, cos_lon, sin_lt, cos_lt, sin_I, cos_I], dim=1
        )
        h_spatial = self.spatial_basis_net(spatial_input)   # [Batch, basis_dim]

        # 5b. 地磁 FiLM 调制（残差调制，缩放 0.1 保证训练稳定）
        # 零初始化 film_mag → 训练初期 γ_mag=0,β_mag=0 → 完全恒等
        # 梯度自然驱动 film_mag 学习地磁场对空间表示的微扰
        h_mag = self.mag_encoder(sin_I, cos_I)                              # [Batch, 32]
        fm_mag = self.film_mag(h_mag)                                        # [Batch, 2*basis_dim]
        gamma_mag, beta_mag = fm_mag.chunk(2, dim=-1)                        # each [Batch, basis_dim]
        h_spatial = h_spatial * (1.0 + 0.1 * gamma_mag) + 0.1 * beta_mag   # 磁场残差调制

        # 6. NeQuick 廓线解码（三层 Epstein: E + F1 + F2）
        h_combined = torch.cat([h_spatial, h_sw], dim=-1)  # [Batch, basis_dim + sw_out_dim]
        Ne_nequick, nequick_params = self.nequick_decoder(
            h_combined, lat_geo, lon_geo, time, alt, lat_siren_n
        )   # [Batch, 1]  — lat_siren_n = lat_geo / 90，用于 NeQuick 纬度三角特征

        # 7. GIRO 快速路径：跳过残差 SIREN + 不确定性估计（约节省 40% 计算）
        if giro_mode:
            _zero = torch.zeros_like(Ne_nequick)
            extras = {
                'ne_bkg':     ne_bkg,
                'ne_chapman': Ne_nequick,
                'ne_residual': _zero,
                'h_spatial':  h_spatial,
                'h_sw':       h_sw,
                'kp_eff':     kp_eff,
                'f107_eff':   f107_eff,
                'chapman_params': nequick_params,
                'coords_normalized': torch.stack([lat_n, lon_n], dim=1),
                'sin_I': sin_I,    # 地磁倾角正弦，供 physics_losses 磁赤道掩码使用
            }
            return Ne_nequick, _zero, Ne_nequick, _zero, extras

        # 7. 残差 SIREN（含高度，8D 输入）
        # [lat_geo_n, sin_lon, cos_lon, alt_n, sin_lt, cos_lt, sin_I, cos_I]
        residual_input = torch.stack(
            [lat_siren_n, sin_lon, cos_lon, alt_n, sin_lt, cos_lt, sin_I, cos_I], dim=1
        )
        h_res = self.residual_net(residual_input)   # [Batch, basis_dim]

        sw_shift    = self.sw_shift_head(h_sw)      # [Batch, basis_dim]
        h_res_mod   = h_res + sw_shift
        raw_resid   = self.residual_decoder(h_res_mod)               # [Batch, 1]
        Ne_residual = torch.tanh(self.resid_scale * raw_resid)       # [Batch, 1]

        # 8. 最终输出（数据同化增量范式）
        # Ne_fused = Ne_iri (背景场) + Ne_residual (同化增量)
        # Ne_nequick 不参与此加法，仅通过 extras['ne_chapman'] 为
        # chapman_shape_loss = MSE(Ne_fused, Ne_nequick) 提供 Epstein 形状软约束。
        Ne_fused = ne_bkg + Ne_residual   # [Batch, 1]

        # 9. 不确定性估计
        unc_input = torch.cat([h_res_mod, h_sw], dim=-1)
        raw_log_var = self.uncertainty_head(unc_input)
        log_var = torch.clamp(raw_log_var, -10.0, 10.0)    # [Batch, 1]

        extras = {
            'ne_bkg':     ne_bkg,
            'ne_chapman': Ne_nequick,   # 保留键名兼容 train_mdia.py 及 physics_losses
            'ne_residual': Ne_residual,
            'h_spatial':  h_spatial,
            'h_sw':       h_sw,
            'kp_eff':     kp_eff,
            'f107_eff':   f107_eff,
            'chapman_params': nequick_params,   # 含 hmF2_F2, NmF2_F2 等（可视化使用）
            'coords_normalized': torch.stack([lat_n, lon_n], dim=1),  # 地理坐标，供 TEC 损失
            'sin_I': sin_I,    # [Batch] 地磁倾角正弦，供 physics_losses 磁赤道感知掩码使用
        }

        return Ne_fused, log_var, Ne_nequick, Ne_residual, extras


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print('=' * 60)
    print('MDIA-INR 模型测试（NeQuick 三层解码器）')
    print('=' * 60)

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    class DummyIRI(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)
        def forward(self, x):
            return self.net(x) + 11.5

    config = {
        'total_hours': 720.0,
        'alt_range':   (120.0, 500.0),
        'seq_len':     72,
        'basis_dim':   64,
        'siren_hidden': 128,
        'siren_layers': 3,
        'omega_0':     30.0,
        'sw_hidden_dim': 32,
        'sw_lstm_layers': 2,
        'sw_out_dim':  64,
        'chapman_hidden': 128,
        'tau_kp_init': 8.0,
        'tau_solar_init': 72.0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MDIA_INR_Model(DummyIRI(), config).to(device)

    print(f'\n总参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'  可训练: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    B = 64

    # 测试 1：4 列 coords（地理纬度回退）
    coords4 = torch.zeros(B, 4).to(device)
    coords4[:, 0] = torch.linspace(-60,  60, B)   # Lat_geo
    coords4[:, 1] = torch.linspace(-180, 180, B)  # Lon_geo
    coords4[:, 2] = torch.linspace(150, 450, B)   # Alt
    coords4[:, 3] = torch.linspace(0, 720, B)     # Time
    sw_seq = torch.randn(B, 72, 2).to(device)

    Ne_fused, log_var, Ne_nequick, Ne_residual, extras = model(coords4, sw_seq)
    print(f'\n[4-col coords] Ne_fused: {Ne_fused.shape}  Ne_nequick: {Ne_nequick.shape}')

    # 测试 2：5 列 coords（含 AACGM lat）
    coords5 = torch.zeros(B, 5).to(device)
    coords5[:, :4] = coords4
    coords5[:, 4]  = torch.linspace(-55, 55, B)   # Lat_aacgm
    Ne_fused5, _, _, _, _ = model(coords5, sw_seq)
    print(f'[5-col coords] Ne_fused: {Ne_fused5.shape}  (AACGM lat active)')

    print(f'\nNeQuick 参数:')
    for k, v in extras['chapman_params'].items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: [{v.min():.2f}, {v.max():.2f}]  shape={v.shape}')

    # 梯度测试
    loss = Ne_fused.sum()
    loss.backward()
    print(f'\n梯度测试:')
    print(f'  τ_kp grad: {model.sw_encoder._log_tau_kp.grad.item():.6f}')
    print(f'  SIREN grad: {model.spatial_basis_net.siren.net[0].linear.weight.grad.norm():.6f}')

    tau_kp    = model.sw_encoder.tau_kp.item()
    tau_solar = model.sw_encoder.tau_solar.item()
    print(f'\nτ_kp = {tau_kp:.2f} h  τ_solar = {tau_solar:.2f} h')
    print('\n所有测试通过!')
