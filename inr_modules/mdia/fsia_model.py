"""
FSIA-INR run61 — FYObsEncoder 邻域观测特征 + IRI passthrough 峰场

run61 核心重构（相对 run60）：

    设计动机：
        h_spatial（DualFreqSpatialNet 输出）只编码查询点坐标，
        不含实际 FY 观测值。FY 损失将 h_spatial 推向 FY-optimal，
        导致 PeakHead 使用被 FY 污染的特征来估计 hmF2，产生根本设计缺陷。

    三路特征重构：
        h_IRI（全覆盖）: IRI 神经代理 → f_iri   ← 现有路径保留
        h_FY（轨道覆盖）: FYNeighborhoodIndex → FYObsEncoder → h_FY
                          无邻居时 h_FY=0 → ETKF innovation=0 → update=0
                          → Ne_fused≈ne_bkg（IRI baseline，安全退化）
        h_SW（全覆盖）: DualScaleSWEncoder → h_sw   ← 现有路径保留

    PeakHead 彻底绕过：
        hmF2_fused = hmF2_IRI（直接来自 IRIPeakManager）
        NmF2_fused = NmF2_IRI（直接来自 IRIPeakManager）
        不再依赖 h_spatial

    删除的模块（不再实例化）：
        DualFreqSpatialNet (spatial_basis_net)
        MagneticEncoder + film_mag（h_spatial MagFiLM）
        proj_delta（h_spatial 3D 注入）
        AltitudeMultiScaleEmbedding + ModulatedSIRENNet + film_mag_res（残差 SIREN）
        res_head_day / res_head_night（日夜双头）
        PeakHead（peak_head，逻辑保留但绕过）

    CRF proj_pre 维度：192（f_iri+h_obs_FY+h_res）→ 128（f_iri+h_FY）

输出范式不变: Ne_fused = Ne_bkg + tanh(FusionDecoder(h_decode_mod, alt_n, δh_n)) × gate
gate_phys_net 输入改为 h_FY（替代原 h_spatial）


run55 核心改动（相对 run54，两处互补修改）：

    方案 A — 夜间样本显式加权（损失层，train_fsia.py）：
        _composite_w = alt_weight × (1 + w_night_boost × night_gate) × B_weight
        night_gate = 0.5 × (1 − cos_SZA).clamp(0, 1)
        动机：日间 EIA 修正幅度大 → 梯度被日间主导 → 显式补偿夜间梯度。
        w_night_boost=0（默认）退化为 run54 行为。

    方案 B — Residual SIREN 日夜残差双头（架构层，fsia_model.py）：
        h_day_delta   = res_head_day(h_res)           [B, 64]  零初始化
        h_night_delta = res_head_night(h_res)          [B, 64]  零初始化
        h_res += (1 − night_gate) × h_day_delta
               + night_gate       × h_night_delta
        动机：h_res 主干被日间 EIA 主导；双头强制日夜模式分离。
        零初始化 → 起步 h_res 不变（IRI baseline 保持）；
        训练后 day/night head 逐步学习各自的修正方向。

run54 改动（保留）：
    1. R_vert 物理先验（磁赤道日夜区分）
    2. B-加权 NLL 损失（DDA Theorem 3）
        night_gate_v = 0.5 × (1 − cos_SZA).clamp(0,1)                  [B]
        mag_eq_gate  = exp(−(sin_I / σ_eq)²)                           [B] ∈ (0,1]
        r_vert_prior = α_vert_night × night_gate_v
                     + α_vert_eq × night_gate_v × mag_eq_gate           [B]
        r_vert = softplus(log_R_vert + r_vert_prior)                    [B, d]（由 [d] 升为 [B,d]）
        动机：磁赤道区域背景场（h_res）日夜偏差方向相反，夜间 R_vert 增大
              → 降低 h_res 在夜间分析中的权重 → 防止日间 EIA 模式污染夜间同化。
        初始 α_vert_night=0.5, α_vert_eq=0.8：
            日间低纬（cos_SZA≈1）→ r_vert_prior ≈ 0（继承 log_R_vert 静态先验）
            夜间低纬（cos_SZA≈-1）→ r_vert_prior ≈ 0.5 + 0.8×mag_eq_gate
            夜间磁赤道（sin_I≈0）  → r_vert_prior ≈ 1.3（2.01× 基线）

    2. B-加权训练损失（DDA Theorem 3 启发）：
        b_weight = (ens_var.mean(dim=-1) / global_mean).clamp(0.2, 5.0)  [B,1]
        loss_nll = (raw_nll × alt_weight × (1 + w_b_cov × (b_weight−1))).mean()
        动机：背景误差协方差大的样本（ensemble 方差大）观测信息更稀缺，
              给予更高损失权重以加速同化；w_b_cov=0 退化为原损失（向后兼容）。

run53 改动（保留）：
    1. h_res FiLM 调制 + 2. R_FY α_night_base

输出范式: Ne_fused = Ne_bkg + tanh(FusionDecoder(h_decode_mod, alt_n, δh_n)) × gate
    h_decode_mod = (1 + γ_d) ⊙ h_decode + β_d          (regime FiLM，run52)
    γ_d, β_d = regime_film_net(ne_bkg_n, cos_SZA, kp_eff, f107_eff)
    h_res_mod  = (1 + γ_r) ⊙ h_res + β_r               (h_res FiLM，run53 新增)
    γ_r, β_r = res_film_net(below_dist, night_gate, below_dist×night_gate, alt_n)
gate     = gate_data × gate_phys
gate_data 输入: cat(h_analysis[64], h_sw[64], log_var[1], |Ne_delta_raw|[1],
                    regime_desc[6])  = 136D
regime_desc: [alt_n, |sin_I|, kp_eff, f107_eff, cos_SZA, lat_n]

giro_mode 快速路径：跳过 IRI proxy + Residual SIREN + KalmanLayer；gate=None
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from .siren_layers import ModulatedSIRENNet
    from .ewma_sw_encoder import DualScaleSWEncoder
    from .mdia_model import (
        _LN10, _INV_LN10, _LOG10_4, _DOY_SEP1,
        _MAG_POLE_LAT_RAD, _MAG_POLE_LON_RAD,
        _compute_dip_features,
        MagneticEncoder,
    )
except ImportError:
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from siren_layers import ModulatedSIRENNet
    from ewma_sw_encoder import DualScaleSWEncoder
    from mdia_model import (
        _LN10, _INV_LN10, _LOG10_4, _DOY_SEP1,
        _MAG_POLE_LAT_RAD, _MAG_POLE_LON_RAD,
        _compute_dip_features,
        MagneticEncoder,
    )


# ======================== Solar Zenith Angle 计算（run25：替代 LT）========================

# Sep 1 day-of-year（非闰年/2024 闰年差 1，可忽略）— 与训练数据集起始日对齐
_DOY_SEP1_FLOAT = 245.0

def _compute_solar_features(lat_deg, lon_deg, rel_hour):
    """
    计算太阳天顶角余弦 + day-of-year 编码（run25：物理上替代 LT 当地时编码）

    使用 Spencer 7阶傅里叶公式计算太阳赤纬，精度 < 0.01°。
    所有运算可微（floor 仅用于离散化 day index，不影响梯度路径）。

    Args:
        lat_deg:  [B] 地理纬度（度）
        lon_deg:  [B] 地理经度（度，∈ [-180, 180]）
        rel_hour: [B] 自数据集起始的相对小时（用作 day-of-year 累加）
    Returns:
        cos_SZA:  [B] 太阳天顶角余弦 ∈ [-1, 1]
                       cos_SZA > 0 → 日照面（白天）
                       cos_SZA < 0 → 阴影面（夜间）
        sin_doy:  [B] sin(2π·doy/365)
        cos_doy:  [B] cos(2π·doy/365)
    """
    # ---- day-of-year + 年角 ----
    doy     = (rel_hour / 24.0).floor() + _DOY_SEP1_FLOAT
    doy_mod = doy % 365.0
    gamma   = 2.0 * math.pi * doy_mod / 365.0                                # [B] 年角

    # ---- 太阳赤纬（Spencer 1971 7阶傅里叶级数）----
    delta = (0.006918
             - 0.399912 * torch.cos(gamma)
             + 0.070257 * torch.sin(gamma)
             - 0.006758 * torch.cos(2.0 * gamma)
             + 0.000907 * torch.sin(2.0 * gamma)
             - 0.002697 * torch.cos(3.0 * gamma)
             + 0.001480 * torch.sin(3.0 * gamma))                            # [B] rad

    # ---- 当地真太阳时 + cos(SZA) ----
    LST        = (rel_hour + lon_deg / 15.0) % 24.0                          # [B]
    hour_angle = (LST - 12.0) * (math.pi / 12.0)
    lat_rad    = lat_deg * (math.pi / 180.0)
    cos_SZA = (torch.sin(delta) * torch.sin(lat_rad)
               + torch.cos(delta) * torch.cos(lat_rad) * torch.cos(hour_angle))

    # ---- 季节相位（doy_phase 与 gamma 完全同义，直接复用）----
    sin_doy = torch.sin(gamma)
    cos_doy = torch.cos(gamma)

    return cos_SZA, sin_doy, cos_doy


# ======================== NeuralETKFLayer 辅助函数 ========================

def _build_kalman_b_input(h_sw, lat_n, cos_SZA, sin_doy, cos_doy, sin_I):
    """B_net / PerturbationNet 输入构造（h_sw + 太阳/地磁 regime 特征）"""
    return torch.cat([
        h_sw,
        lat_n.unsqueeze(-1),
        cos_SZA.unsqueeze(-1),
        sin_doy.unsqueeze(-1),
        cos_doy.unsqueeze(-1),
        sin_I.unsqueeze(-1),
    ], dim=-1)                                                                  # [B, b_net_in=69]


def _build_kalman_r_fy_input(alt_n, delta_alt, cos_SZA, sin_doy, cos_doy, lat_n_abs, h_sw):
    """R_FY_net 输入构造。proxy K_FY 用 delta_alt_iri；主路径用 delta_alt_n。"""
    return torch.cat([
        alt_n.unsqueeze(-1),
        delta_alt.unsqueeze(-1),
        cos_SZA.unsqueeze(-1),
        sin_doy.unsqueeze(-1),
        cos_doy.unsqueeze(-1),
        lat_n_abs.unsqueeze(-1),
        h_sw,
    ], dim=-1)                                                                  # [B, r_fy_net_in=70]
# ======================== NeuralETKFLayer (run28) ========================

class NeuralETKFLayer(nn.Module):
    """
    神经化集合卡尔曼滤波（ETKF 形式）— run28，替代 MHDK 作为 DA 核心。

    通过 N 个并行 PerturbationNet 生成 ensemble 扰动 δ_n（n=1..N），
    用 ETKF transform 在 N 维子空间求逆做最优更新（不需要 d×d 求逆）：

        δ^(n)        = PerturbationNet_n(b_in)             [B, d]    n=1..N
        X            = δ - δ_mean                          [B, N, d] centered
        X_inf        = sqrt(inflation) × X                            inflation 防 spread 坍塌
        HX_*         = einsum('bnd,do->bno', X_inf, H_*)   [B, N, d]
        HXR_*        = HX_* / R_*                                     pre-whitened
        M_*          = HXR_* @ HX_*^T / (N-1)              [B, N, N]
        T_*          = inv(I_N + M_*)                                 N=8 → 8×8 求逆便宜
        innov_*      = h_obs_* - H_* @ f_iri               [B, d]
        w_*          = T_* @ HXR_* @ innov_*               [B, N]
        x_a          = LN(f_iri + δ_mean + X_inf^T @ (w_FY + w_vert))

    R_FY 物理先验（FY Abel 反演 + LT 调制）：
        below_dist  = relu(hmF2 - alt) / (hmF2 - alt_floor).clamp_min(30)   ∈ [0, 1]
                       0  在 hmF2 处；1  在 alt_floor=120km 处（最深底侧，最大不确定性）
        night_gate  = 0.5 × (1 - cos_SZA).clamp(0, 1)
        r_fy_prior  = below_dist × (α_day + α_night × night_gate)            标量先验
        r_fy        = softplus(R_FY_net(in) + r_fy_prior)
    初始 α_day=0.5, α_night=1.5：
        在 alt=hmF2 处       r_fy ≈ softplus(0)=0.69      → 1.00× 基线
        在 alt=alt_floor 白天 r_fy ≈ softplus(0.5)=0.97   → 1.40× 基线
        在 alt=alt_floor 夜间 r_fy ≈ softplus(2.0)=2.13   → 3.07× 基线
    物理意义：alt=120km（同化范围下界）处不确定性最大；越接近 hmF2 越小；夜间额外加成。
    可学习的 α_* → 网络可微调先验强度。

    LT/区域局地化：通过 PerturbationNet 输入条件（cos_SZA / sin_I / lat_n / h_sw）
    隐式实现，每个 ensemble member 自适应学习不同 regime 的协方差结构。

    n_members=1 退化为单成员 KF（消融对照）。
    返回 7-tuple，与 MHDK 接口完全一致；监控量物理意义升级：
        K_FY/K_vert  → K_eff_*: 对角等效卡尔曼增益（ens_var/(ens_var+R)）
        b            → ens_var: ensemble 方差，等价于背景误差协方差对角元
    """

    def __init__(self, d_model: int = 64, b_net_in: int = 69, r_fy_net_in: int = 70,
                 n_members: int = 8, pert_hidden: int = 64,
                 n_rank_h: int = 8):
        super().__init__()
        self.d_model     = d_model
        self.n_members   = n_members
        self.b_net_in    = b_net_in
        self.r_fy_net_in = r_fy_net_in
        self.pert_hidden = pert_hidden

        # ---- N 个并行 PerturbationNet（向量化 Parameter[N, in, out]）----
        # 输入: b_net_in（同 MHDK B_net 输入）= h_sw + lat_n + cos_SZA + sin_doy + cos_doy + sin_I
        self.P_w1 = nn.Parameter(torch.empty(n_members, b_net_in, pert_hidden))
        self.P_b1 = nn.Parameter(torch.empty(n_members, pert_hidden))
        self.P_w2 = nn.Parameter(torch.empty(n_members, pert_hidden, d_model))
        self.P_b2 = nn.Parameter(torch.empty(n_members, d_model))

        # ---- R_FY: 纯数据驱动，不确定性由 ETKF 自动学习 ----
        self.R_FY_net = nn.Sequential(
            nn.Linear(r_fy_net_in, 64), nn.SiLU(), nn.Linear(64, d_model))

        # ---- FY observation operator (global H, zero-init) ----
        self.H_FY_w   = nn.Parameter(torch.zeros(d_model, d_model))

        # ---- run56: SALR-H 低秩空间自适应观测算子 ----
        _salr_cond = 17
        self.n_rank_h  = n_rank_h
        self.H_FY_u    = nn.Parameter(torch.zeros(n_rank_h, d_model))
        self.H_FY_v    = nn.Parameter(torch.zeros(n_rank_h, d_model))
        self.H_FY_a    = nn.Linear(_salr_cond, n_rank_h, bias=True)
        nn.init.zeros_(self.H_FY_a.weight); nn.init.zeros_(self.H_FY_a.bias)

        # ---- run64: COSMIC observation channel (replaces vert channel) ----
        # H_COSMIC_w zero-init -> COSMIC channel inactive at start -> IRI baseline
        self.H_COSMIC_w  = nn.Parameter(torch.zeros(d_model, d_model))
        self.H_COSMIC_u  = nn.Parameter(torch.zeros(n_rank_h, d_model))
        self.H_COSMIC_v  = nn.Parameter(torch.zeros(n_rank_h, d_model))
        self.H_COSMIC_a  = nn.Linear(_salr_cond, n_rank_h, bias=True)
        nn.init.zeros_(self.H_COSMIC_a.weight); nn.init.zeros_(self.H_COSMIC_a.bias)
        # R_COSMIC: 纯数据驱动，与 R_FY 独立网络
        self.R_COSMIC_net = nn.Sequential(
            nn.Linear(r_fy_net_in, 64), nn.SiLU(), nn.Linear(64, d_model))
        self.log_r_ref_COSMIC = nn.Parameter(torch.zeros(()))

        # ---- Inflation（可学习标量；exp 保证 > 0，初始 = 1.0）----
        self.log_inflation = nn.Parameter(torch.zeros(()))

        # ---- run59: H 学习参考 R（解耦 H 梯度与物理先验）----
        # softplus(0)+0.1 ≈ 0.79；r_ref 可学习 → 自动校准基准增益
        self.log_r_ref_FY = nn.Parameter(torch.zeros(()))   # r_ref_FY = softplus + 0.1

        # ---- 输出归一化 ----
        self.norm = nn.LayerNorm(d_model)

        # 监控量（外部读取）
        self.last_member_weights  = None
        self.last_head_weights    = None    # 兼容别名 = last_member_weights
        self.last_inflation_scale = None

        self._reset_kalman_params()

    def _reset_kalman_params(self):
        """PerturbationNet 用 nn.Linear 默认风格 uniform 初始化（per-member fan_in）。
        H_FY_w / H_COSMIC_w 已在 __init__ 中零初始化。"""
        for w, b, fan_in in [
            (self.P_w1, self.P_b1, self.b_net_in),
            (self.P_w2, self.P_b2, self.pert_hidden),
        ]:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)

    # ---------- 向量化扰动生成 ----------
    def _eval_perturbations(self, b_input):
        """N 个并行扰动 → ensemble 扰动 [B, N, d]"""
        h = torch.einsum('bi,nio->bno', b_input, self.P_w1) + self.P_b1.unsqueeze(0)
        h = F.silu(h)
        delta = torch.einsum('bni,nio->bno', h, self.P_w2) + self.P_b2.unsqueeze(0)
        return delta                                                            # [B, N, d]

    def _eval_R_FY(self, r_input):
        """R_FY = softplus(R_FY_net(in)) — 纯数据驱动，ETKF 自动学习不确定性。"""
        return F.softplus(self.R_FY_net(r_input))                               # [B, d]

    def _eval_R_COSMIC(self, r_input):
        """R_COSMIC = softplus(R_COSMIC_net(in)) — 独立网络，纯数据驱动。"""
        return F.softplus(self.R_COSMIC_net(r_input))                           # [B, d]

    def forward(self, f_iri, h_obs_FY, h_sw,
                lat_n, cos_SZA, sin_doy, cos_doy, sin_I, alt_n, delta_alt_n,
                alt_km, hmF2_km,
                sh_feats=None, has_obs=None,
                h_obs_COSMIC=None, has_obs_cosmic=None):
        b_in = _build_kalman_b_input(h_sw, lat_n, cos_SZA, sin_doy, cos_doy, sin_I)
        r_in = _build_kalman_r_fy_input(alt_n, delta_alt_n,
                                         cos_SZA, sin_doy, cos_doy, lat_n.abs(), h_sw)

        # ---- Step 1: ensemble perturbations + centering + inflation ----
        delta      = self._eval_perturbations(b_in)
        delta_mean = delta.mean(dim=1, keepdim=True)
        X          = delta - delta_mean
        inflation  = torch.exp(self.log_inflation)
        X_inf      = X * torch.sqrt(inflation)

        # ---- Step 2: observation error covariances (data-driven) ----
        r_fy = self._eval_R_FY(r_in)
        use_cosmic = (h_obs_COSMIC is not None)
        if use_cosmic:
            r_cosmic = self._eval_R_COSMIC(r_in)
        else:
            r_cosmic = torch.zeros_like(r_fy)

        # ---- Step 3: SALR-H observation projection ----
        if sh_feats is not None:
            h_cond = torch.cat([sh_feats,
                                cos_SZA.unsqueeze(-1),
                                sin_I.unsqueeze(-1),
                                alt_n.unsqueeze(-1)], dim=-1)
            a_FY     = self.H_FY_a(h_cond)
            a_COSMIC = self.H_COSMIC_a(h_cond)
        else:
            a_FY = a_COSMIC = None

        HX_FY_b     = torch.einsum('bnd,do->bno', X_inf, self.H_FY_w)
        HX_COSMIC_b = torch.einsum('bnd,do->bno', X_inf, self.H_COSMIC_w)
        if a_FY is not None:
            vx_FY = torch.einsum('bnd,kd->bnk', X_inf, self.H_FY_v)
            HX_FY = HX_FY_b + torch.einsum('bnk,kd->bnd',
                                             vx_FY * a_FY.unsqueeze(1), self.H_FY_u)
            vx_COSMIC = torch.einsum('bnd,kd->bnk', X_inf, self.H_COSMIC_v)
            HX_COSMIC = HX_COSMIC_b + torch.einsum('bnk,kd->bnd',
                                                     vx_COSMIC * a_COSMIC.unsqueeze(1),
                                                     self.H_COSMIC_u)
        else:
            HX_FY, HX_COSMIC = HX_FY_b, HX_COSMIC_b

        eps = 1e-6
        r_ref_FY     = F.softplus(self.log_r_ref_FY)    + 0.1
        r_ref_COSMIC = F.softplus(self.log_r_ref_COSMIC) + 0.1

        HXR_FY     = HX_FY     / (r_ref_FY     + eps)
        HXR_COSMIC = HX_COSMIC / (r_ref_COSMIC + eps)

        N   = self.n_members
        I_N = torch.eye(N, device=X.device, dtype=X.dtype).unsqueeze(0)
        M_FY     = torch.einsum('bnd,bmd->bnm', HXR_FY,     HX_FY    ) / max(N - 1, 1)
        M_COSMIC = torch.einsum('bnd,bmd->bnm', HXR_COSMIC, HX_COSMIC) / max(N - 1, 1)
        T_FY     = torch.linalg.inv(I_N + M_FY)
        T_COSMIC = torch.linalg.inv(I_N + M_COSMIC)

        # ---- Step 4: innovations + weights + posterior ----
        innov_FY = h_obs_FY - torch.einsum('bd,do->bo', f_iri, self.H_FY_w)
        if a_FY is not None:
            vf_FY    = torch.einsum('bd,kd->bk', f_iri, self.H_FY_v)
            innov_FY = innov_FY - torch.einsum('bk,kd->bd', vf_FY * a_FY, self.H_FY_u)
        w_FY_pre  = torch.einsum('bnd,bd->bn', HXR_FY, innov_FY)
        w_FY_base = torch.einsum('bnm,bm->bn', T_FY, w_FY_pre)

        if use_cosmic:
            innov_COSMIC = h_obs_COSMIC - torch.einsum('bd,do->bo', f_iri, self.H_COSMIC_w)
            if a_COSMIC is not None:
                vf_COSMIC    = torch.einsum('bd,kd->bk', f_iri, self.H_COSMIC_v)
                innov_COSMIC = innov_COSMIC - torch.einsum('bk,kd->bd',
                                                            vf_COSMIC * a_COSMIC,
                                                            self.H_COSMIC_u)
            w_COSMIC_pre  = torch.einsum('bnd,bd->bn', HXR_COSMIC, innov_COSMIC)
            w_COSMIC_base = torch.einsum('bnm,bm->bn', T_COSMIC, w_COSMIC_pre)
        else:
            innov_COSMIC  = torch.zeros_like(innov_FY)
            w_COSMIC_base = torch.zeros(X.shape[0], N, device=X.device, dtype=X.dtype)

        # run59: inference gain suppression via physical prior scaling
        phy_scale_FY = (r_ref_FY / (r_fy.detach() + eps)).clamp(0.0, 1.0).mean(-1)
        w_FY   = w_FY_base * phy_scale_FY.unsqueeze(-1)

        if use_cosmic:
            phy_scale_COSMIC = (r_ref_COSMIC / (r_cosmic.detach() + eps)).clamp(0.0, 1.0).mean(-1)
            w_COSMIC = w_COSMIC_base * phy_scale_COSMIC.unsqueeze(-1)
        else:
            w_COSMIC = w_COSMIC_base

        # run63: mask update when no observations present
        if has_obs is not None:
            w_FY = w_FY * has_obs.unsqueeze(-1)
        if has_obs_cosmic is not None and use_cosmic:
            w_COSMIC = w_COSMIC * has_obs_cosmic.unsqueeze(-1)

        w_total    = w_FY + w_COSMIC
        update     = torch.einsum('bn,bnd->bd', w_total, X_inf)
        h_analysis = self.norm(f_iri + delta_mean.squeeze(1) + update)

        # ---- monitoring quantities ----
        ens_var      = X.pow(2).sum(dim=1) / max(N - 1, 1)
        K_eff_FY     = ens_var / (ens_var + r_fy     + eps)
        K_eff_COSMIC = ens_var / (ens_var + r_cosmic  + eps)

        w_abs = w_total.abs()
        member_weights = w_abs / (w_abs.sum(dim=-1, keepdim=True) + eps)
        self.last_member_weights  = member_weights
        self.last_head_weights    = member_weights
        self.last_inflation_scale = inflation.detach()

        return (h_analysis, K_eff_FY, K_eff_COSMIC,
                ens_var, r_fy, innov_FY, innov_COSMIC)


# ======================== MultiScaleAdaptiveGate ========================

class MultiScaleAdaptiveGate(nn.Module):
    """
    多尺度自适应门控（替代 AssimilationGate）

    gate = gate_data × gate_phys

    gate_data（数据驱动）：
        cat(h_fused[64], h_sw[64], log_var[1], |Ne_delta_raw|[1], regime_desc[6]) = 136D
        → Linear(136→32) → SiLU → Linear(32→1) → sigmoid
        初始化: weight=0, bias=logit(0.95)≈2.944 → gate_data≈0.95
        regime_desc = [alt_n, |sin_I|, kp_eff, f107_eff, cos_SZA, lat_n]

    gate_phys（物理先验，空间自适应）：
        三高斯结构，参数由 h_spatial 残差超网络生成

        g(δ) = Σ_k w_k(x)·exp(-((δ - c_k)/σ_k(x))²)
        gate_phys = sigmoid(g)
        δ = (alt_km - hmF2_fused.detach()) / H_scale

        c_k 固定 buffer: [0.0, -1.5, 2.0]   (F2峰/底部/顶部)
        w_k(x) = w_prior_k + gate_phys_net(h_spatial)[:, k]
        σ_k(x) = exp(log_σ_prior_k + gate_phys_net(h_spatial)[:, k+3])

        物理先验 buffer：
            w_prior      = [+2.5, -1.5, -1.0]
            log_σ_prior  = [ln0.8, ln0.5, ln1.2]

        gate_phys_net: Linear(64→32) → SiLU → Linear(32→6)
            最后层 weight=0, bias=0 → 初期 Δ=0 → 纯先验
            训练后自然学习低纬/高纬/不同经度的差异

    初始 gate_phys 典型值（纯先验）：
        δ≈0   (F2峰)    → ≈ 0.92  高增益同化
        δ≈-1.5 (底部)   → ≈ 0.21  抑制
        δ≈+2  (深顶部)  → ≈ 0.27  衰减

    参数量: gate_net=4257 + gate_phys_net=2278 = 6535 可训练（in_dim=130）
    """

    def __init__(self, in_dim: int = 130, basis_dim: int = 64,
                 h_scale: float = 100.0):
        super().__init__()
        self.h_scale = h_scale

        # gate_data（输入 128→130，含 log_var + |Ne_delta_raw|）
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, math.log(0.95 / 0.05))  # ≈2.944

        # gate_phys 残差超网络
        # P0-B: 输入扩展 basis_dim+2 (sin_LT/cos_LT)
        # run25: → basis_dim+3 (cos_SZA + sin_doy + cos_doy)，物理上具备日期+真实日夜感知
        self.gate_phys_net = nn.Sequential(
            nn.Linear(basis_dim + 3, 32), nn.SiLU(), nn.Linear(32, 6))
        nn.init.zeros_(self.gate_phys_net[-1].weight)
        nn.init.zeros_(self.gate_phys_net[-1].bias)

        # 物理先验（固定，不可学习）
        # P0-B: 顶层先验从 -1.0 → -0.3，松弛对顶侧（alt > hmF2）的过度抑制
        self.register_buffer('w_prior',
            torch.tensor([2.5, -1.5, -0.3]))
        self.register_buffer('log_sigma_prior',
            torch.tensor([math.log(0.8), math.log(0.5), math.log(1.2)]))
        self.register_buffer('centers',
            torch.tensor([0.0, -1.5, 2.0]))

    def forward(self, h_fused, h_sw, alt_km, hmF2_det, h_spatial,
                log_var_det, ne_delta_raw_abs,
                cos_SZA, sin_doy, cos_doy, regime_desc):
        """
        Args:
            h_fused, h_sw:      [B, 64]
            alt_km, hmF2_det:   [B]   实际高度 / 融合峰高（已 detach）
            h_spatial:          [B, 64]  双频空间特征（gate_phys 区域先验来源）
            log_var_det:        [B, 1]   log_var.detach()（不确定性感知）
            ne_delta_raw_abs:   [B, 1]   |Ne_delta_raw|.detach()（残差幅度感知）
            cos_SZA, sin_doy, cos_doy: [B, 1]  太阳天顶角余弦 + 季节相位
            regime_desc:        [B, R]   显式 regime 描述符 [alt_n, |sin_I|, kp_eff, f107_eff, cos_SZA, lat_n]
        Returns:
            gate, gate_data, gate_phys:  各 [B, 1]
        """
        # gate_data: cat(h_fused, h_sw, log_var, |Ne_delta_raw|, regime_desc)
        gate_data = torch.sigmoid(self.gate_net(torch.cat(
            [h_fused, h_sw, log_var_det, ne_delta_raw_abs, regime_desc], dim=-1)))  # [B, 1]

        # gate_phys — 空间自适应参数（cos_SZA + sin_doy/cos_doy 替代 sin_LT/cos_LT）
        gate_phys_input = torch.cat([h_spatial, cos_SZA, sin_doy, cos_doy], dim=-1)  # [B, 67]
        dp    = self.gate_phys_net(gate_phys_input)                # [B, 6]
        w     = self.w_prior     + dp[:, :3]                       # [B, 3]
        sigma = torch.exp(self.log_sigma_prior + dp[:, 3:])        # [B, 3] > 0

        delta = ((alt_km - hmF2_det) / self.h_scale).unsqueeze(-1) # [B, 1]
        gauss = torch.exp(-((delta - self.centers) / sigma) ** 2)  # [B, 3]
        gate_phys = torch.sigmoid(
            (gauss * w).sum(dim=-1, keepdim=True))                 # [B, 1]

        return gate_data * gate_phys, gate_data, gate_phys


# ======================== AltitudeMultiScaleEmbedding ========================

class AltitudeMultiScaleEmbedding(nn.Module):
    """
    固定多尺度高度嵌入（参数量=0）。

    将归一化高度 alt_n ∈ [-1, 1] 展开为 sin/cos 多频特征：
        [sin(π·alt_n), sin(2π·alt_n), sin(4π·alt_n),
         cos(π·alt_n), cos(2π·alt_n), cos(4π·alt_n)]  → 6D

    注入 Residual SIREN 后，为高度方向提供显式多尺度先验，
    约束 EDP 廓线不产生高频垂直振荡。
    """

    def __init__(self, freqs=(1, 2, 4)):
        super().__init__()
        self.register_buffer('freqs',
                             torch.tensor(freqs, dtype=torch.float32) * math.pi)

    def forward(self, alt_n):   # alt_n: [B] normalized ∈ [-1, 1]
        x = alt_n.unsqueeze(-1) * self.freqs            # [B, 3]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [B, 6]


# ======================== DualFreqSpatialNet ========================

class DualFreqSpatialNet(nn.Module):
    """
    双频并行空间 SIREN：ω₀=10（全局）+ ω₀=30（局部），自适应门控融合。

    低频分支（ω₀=10）：捕获大尺度背景结构（mid-lat 驼峰、极区梯度）
    高频分支（ω₀=30）：捕获精细结构（EIA 双峰、等离子泡）
    门控融合（sigmoid gate）：gate·h_high + (1-gate)·h_low，初始 gate=0.5

    相比单一 ω₀=30 的改进：
        - 避免过低纬度区域的高频过拟合
        - 中纬度/高纬度结构由低频分支主导，低纬度精细结构由高频分支主导
        - 门控由 concat(h_low, h_high) 计算，自适应选择最优混合比
    """

    def __init__(self, in_features, hidden_dim, out_dim, n_layers,
                 omega_low=10.0, omega_high=30.0):
        super().__init__()
        self.siren_low = ModulatedSIRENNet(
            in_features=in_features,
            hidden_features=hidden_dim,
            hidden_layers=n_layers,
            out_features=out_dim,
            omega_0=omega_low,
        )
        self.siren_high = ModulatedSIRENNet(
            in_features=in_features,
            hidden_features=hidden_dim,
            hidden_layers=n_layers,
            out_features=out_dim,
            omega_0=omega_high,
        )
        self.gate_net = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim // 2),
            nn.SiLU(),
            nn.Linear(out_dim // 2, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)  # sigmoid(0)=0.5 → 初始等权融合（run13 回退）

    def forward(self, x):
        h_low  = self.siren_low(x)
        h_high = self.siren_high(x)
        gate   = torch.sigmoid(self.gate_net(torch.cat([h_low, h_high], dim=-1)))
        return gate * h_high + (1.0 - gate) * h_low


# ======================== PeakHead ========================

class PeakHead(nn.Module):
    """
    IRI-GIRO 融合 F2 峰场（run28-B 经典版 logit 残差范式）

    架构:
        输入: cat(LN(h_spatial), LN(h_sw), hmF2_IRI_n, NmF2_IRI_n, sh_feats[14]) = 144D
        网络: Linear(144→64) → SiLU → Linear(64→2)，输出层零初始化
        公式: fused = sigmoid(logit(IRI_n) + δ) × span + lo
        起步: δ=0 → fused = IRI（精确退化）
        范围: hmF2 ∈ [200, 550] km；NmF2 ∈ [9, 13] log10

    Args（forward）:
        h_spatial: [B, 64]
        h_sw:      [B, 64]
        iri_peak:  [B, 2]  [hmF2_IRI_km, NmF2_IRI_log10]
        sh_feats:  [B, 14] MODIPSHBasis 输出
    Returns:
        hmF2_fused [B], NmF2_fused [B]
    """

    def __init__(self, spatial_sw_dim: int = 128,
                 hmf2_range=(200.0, 550.0),
                 nmf2_range=(9.0, 13.0),
                 sh_dim: int = 14):
        super().__init__()
        in_dim = spatial_sw_dim + 2 + sh_dim   # 128 + 2 + 14 = 144D

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)   # δ=0 → fused=IRI 起步

        # 跨模态归一化：h_spatial（SIREN 末层线性，无界）与 h_sw（Tanh 有界）
        _half = spatial_sw_dim // 2   # 64
        self.ln_spatial = nn.LayerNorm(_half)
        self.ln_sw      = nn.LayerNorm(_half)

        self.lo_h   = hmf2_range[0]
        self.span_h = hmf2_range[1] - hmf2_range[0]
        self.lo_n   = nmf2_range[0]
        self.span_n = nmf2_range[1] - nmf2_range[0]

    def forward(self, h_spatial, h_sw, iri_peak, sh_feats):
        hmF2_IRI = iri_peak[:, 0]
        NmF2_IRI = iri_peak[:, 1]

        hmF2_n = ((hmF2_IRI - self.lo_h) / self.span_h).clamp(0.001, 0.999)
        NmF2_n = ((NmF2_IRI - self.lo_n) / self.span_n).clamp(0.001, 0.999)

        x = torch.cat([
            self.ln_spatial(h_spatial),
            self.ln_sw(h_sw),
            hmF2_n.unsqueeze(-1),
            NmF2_n.unsqueeze(-1),
            sh_feats,
        ], dim=-1)                                          # [B, 144]
        delta = self.net(x)                                 # [B, 2]

        hmF2_fused = torch.sigmoid(torch.logit(hmF2_n) + delta[:, 0]) * self.span_h + self.lo_h
        NmF2_fused = torch.sigmoid(torch.logit(NmF2_n) + delta[:, 1]) * self.span_n + self.lo_n
        return hmF2_fused, NmF2_fused


# ======================== MODIPSHBasis ========================

class MODIPSHBasis(nn.Module):
    """
    零参数 MODIP 球谐基函数编码器（仿 IRI SHU-2015，LT→SZA 化）。

    输出 14D 固定基函数（无参数）：
        P1-P4(sin_MODIP)              [4D] 纬度结构（P₂ 捕获 EIA 双峰）
        P1·cos_SZA, P1·sin_doy        [2D] 主日/季变化
        P2·cos_SZA, P2·sin_doy        [2D] EIA 日/季变化（关键项）
        P2·cos2_SZA, P2·cross_SZA_doy [2D] P₂ × 半日 / 季-日交叉
        P3·cos_SZA, P3·sin_doy        [2D] 三阶纬度 × 日/季
        cos2_SZA, cross_SZA_doy       [2D] 背景半日 / 季-日交叉

    sin(MODIP) = tan(I) / sqrt(tan²(I) + cos(lat_geo))  (Rawer 1963；全程可微)
    """

    def forward(self, sin_I, cos_I, lat_geo_deg, cos_SZA, sin_doy):
        """
        Args:
            sin_I, cos_I: [B]  地磁倾角 sin/cos（_compute_dip_features 输出）
            lat_geo_deg:  [B]  地理纬度 (°)
            cos_SZA:      [B]  太阳天顶角余弦 ∈ [-1,1]
            sin_doy:      [B]  sin(2π·doy/365) 季节相位
        Returns:
            sh_feats: [B, 14]
        """
        cos_lat = torch.cos(lat_geo_deg * (math.pi / 180.0))
        tan_I   = sin_I / (cos_I + 1e-7)
        sm      = tan_I / torch.sqrt(tan_I ** 2 + cos_lat + 1e-7)            # [B] ∈ (-1,1)

        # Legendre P1-P4(sin_MODIP)
        P1 = sm
        P2 = (3.0 * sm ** 2 - 1.0) * 0.5                                     # EIA 双峰模板
        P3 = (5.0 * sm ** 3 - 3.0 * sm) * 0.5
        P4 = (35.0 * sm ** 4 - 30.0 * sm ** 2 + 3.0) * 0.125

        cos2_SZA      = 2.0 * cos_SZA ** 2 - 1.0                             # 2倍频谐波代理
        cross_SZA_doy = 2.0 * cos_SZA * sin_doy                              # 季节-日交叉项

        return torch.stack([
            P1, P2, P3, P4,
            P1 * cos_SZA, P1 * sin_doy,
            P2 * cos_SZA, P2 * sin_doy,
            P2 * cos2_SZA, P2 * cross_SZA_doy,
            P3 * cos_SZA, P3 * sin_doy,
            cos2_SZA, cross_SZA_doy,
        ], dim=-1)                                                            # [B, 14]


# ======================== SpectralSWBranch (run40, 复活自 run28-A) ========================

class SpectralSWBranch(nn.Module):
    """
    SW 频域分支（精简版，无 Transformer）— run40 自适应频域门控

    流程:
        sw_seq [B, L, 2]
            └─ rFFT magnitude          → [B, L//2+1, 2]
            └─ Linear(2→d) + SiLU       → [B, n_freq, d]
            └─ Attention-weighted pool  → [B, d]

    设计动机:
        run28-A vs run28-B ISR 指标对比显示：
            - SpectralSWBranch 在某些 regime 下提供有价值的信号
              （Jicamarca night CCC: A 0.554 > B 0.493；PokerFlat 120-300km day CCC: A 0.795 > B 0.719）
            - 但在其他 regime 下成为噪声源（B 在 Jicamarca day / PokerFlat night 反超）
        run40 重新引入但**配合 regime-aware sw_gate**（69D 输入含 cos_SZA/sin_doy/cos_doy），
        让网络自动学习何时启用频域信号。

    参数量: ~3K（vs run28-A 初版 Transformer ~50K，−94%）
    """

    def __init__(self, seq_len: int = 36, d_model: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq_bins = seq_len // 2 + 1
        self.freq_proj = nn.Linear(2, d_model)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 32), nn.SiLU(), nn.Linear(32, 1),
        )

    def forward(self, sw_seq):
        """sw_seq: [B, L, 2]; returns h_sw_freq [B, d_model]"""
        F_mag   = torch.fft.rfft(sw_seq, dim=1).abs()         # [B, n_freq, 2]
        Z_f     = F.silu(self.freq_proj(F_mag))               # [B, n_freq, d]
        weights = F.softmax(self.attn_pool(Z_f), dim=1)       # [B, n_freq, 1]
        return (weights * Z_f).sum(dim=1)                     # [B, d]


# ======================== FYObsEncoder（run61 新增）========================

class FYObsEncoder(nn.Module):
    """
    FY 邻域观测编码器（run61）

    从 K 个最近 FY 邻居（10D 特征）提取观测特征 h_FY [B, d_model]。
    无邻居时（has_obs=False）输出 zeros → ETKF 退化为 IRI baseline。

    输入: neighbors_feats [B, K, 10]，has_obs [B]
        10D = [ne_k_n, lat_k_n, sin_lon_k, cos_lon_k, alt_k_n,
               Δlat_n, Δlon_sin, Δlon_cos, Δt_n, Δalt_n]
        Δalt_n = (alt_k - alt_q) / 190  — 垂直相对位置（run61 fix）

    零初始化 input_proj → 训练初期 keys≈0 → attention uniform →
    h_FY 接近零 → ETKF innovation≈0 → update=0 → IRI baseline 起步。
    """
    def __init__(self, feat_dim: int = 10, d_model: int = 64,
                 n_heads: int = 4, k_max: int = 64):
        super().__init__()
        self.d_model = d_model
        self.k_max   = k_max
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm  = nn.LayerNorm(d_model)
        # 零初始化：训练初期 h_FY ≈ 0 → ETKF update ≈ 0 → IRI baseline 起步
        nn.init.zeros_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, neighbors_feats, has_obs):
        """
        Args:
            neighbors_feats: [B, K, 10]
            has_obs:         [B] float32, 1.0 if has FY neighbors else 0.0
        Returns:
            h_FY: [B, d_model], zeros where has_obs=0
        """
        B, K, _ = neighbors_feats.shape
        # 检测有效 FY 样本（has_obs>0）
        valid_mask = (has_obs > 0.5)   # [B] bool
        if not valid_mask.any():
            return torch.zeros(B, self.d_model,
                               device=neighbors_feats.device,
                               dtype=neighbors_feats.dtype)

        # 仅对有效样本计算注意力（节省计算）
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]  # [V]
        nb_valid  = neighbors_feats[valid_idx]             # [V, K, feat_dim]

        keys = F.silu(self.input_proj(nb_valid))          # [V, K, d]
        q    = self.query.expand(len(valid_idx), -1, -1)  # [V, 1, d]
        out, _ = self.attn(q, keys, keys)                  # [V, 1, d]
        h_valid = self.norm(out.squeeze(1))               # [V, d]

        # 写回结果，has_obs=0 的位置保持 0
        h_FY = torch.zeros(B, self.d_model,
                           device=neighbors_feats.device,
                           dtype=neighbors_feats.dtype)
        h_FY[valid_idx] = h_valid
        return h_FY


# ======================== 主模型 ========================

class FSIA_INR_Model(nn.Module):
    """
    FSIA-INR 主模型（v2）

    与 v1 的关键差异：
        1. 空间 SIREN → DualFreqSpatialNet（双频 ω₀=10/30，门控融合）
        2. 删除 NeQuick 解码器，改用 PeakHead（数据驱动峰值预测）
        3. Residual SIREN 输入维度 8 → 14（注入 6D 多尺度高度嵌入）
        4. extras['peak_params'] = {'hmF2': [B], 'NmF2': [B]}（新增）
        5. extras['chapman_params'] = peak_params（向后兼容 giro_dataloader）
        6. extras['ne_chapman'] = zeros（向后兼容，不再是 NeQuick 输出）

    完全兼容接口：
        - forward() 返回与 MDIA-INR 完全相同的 5-tuple
        - giro_dataloader.py 无需修改（chapman_params 别名）
        - physics_losses_mdia.py ne_chapman=None guard 已添加

    Args:
        iri_proxy:  IRINeuralProxy 实例（已从磁盘加载权重）
        config:     配置字典（CONFIG_MDIA）
    """

    def __init__(self, iri_proxy, config):
        super().__init__()

        self.total_hours = config.get('total_hours', 720.0)
        self.alt_min, self.alt_max = config['alt_range']
        self.seq_len = config['seq_len']

        basis_dim       = config.get('basis_dim', 64)
        sw_out_dim      = config.get('sw_out_dim', 64)
        sw_hidden_dim   = config.get('sw_hidden_dim', 32)
        sw_lstm_layers  = config.get('sw_lstm_layers', 2)
        tau_kp_init     = config.get('tau_kp_init', 8.0)
        tau_solar_init  = config.get('tau_solar_init', 72.0)

        # IRI proxy 末层隐层维度（与代理网络架构匹配：[4,128,128,128,128,1]）
        iri_hidden_dim = 128

        # ==================== [A] IRI 代理场（完全冻结）====================
        self.iri_proxy = iri_proxy
        for param in self.iri_proxy.parameters():
            param.requires_grad = False

        # ---- IRI 峰对齐特征网络 ----
        # 输入: cat(h_iri[128], delta_alt_iri[1], NmF2_IRI_n[1]) = 130D
        self.iri_align_net = nn.Sequential(
            nn.Linear(iri_hidden_dim + 2, 128),
            nn.SiLU(),
            nn.Linear(128, basis_dim),
        )
        # frame_offset 注入（坐标系差，run61: hmF2_fused=hmF2_IRI → frame_offset=0，但保留路径）
        self.proj_frame_offset = nn.Linear(1, basis_dim)
        # IRI 结构重建头：监督 h_iri_aligned 保留 IRI 场信息
        self.iri_recon_head = nn.Linear(basis_dim, 1)

        # ==================== [B] 双尺度 SW 编码器（含多窗口统计）====================
        self.sw_encoder = DualScaleSWEncoder(
            seq_len=self.seq_len,
            sw_hidden_dim=sw_hidden_dim,
            sw_lstm_layers=sw_lstm_layers,
            sw_out_dim=sw_out_dim,
            tau_kp_init=tau_kp_init,
            tau_solar_init=tau_solar_init,
        )

        # ==================== [B+] SW 频域分支（run40，复活自 run28-A 简化版）====================
        self.use_sw_freq    = bool(config.get('use_sw_freq', True))
        sw_gate_bias_init   = float(config.get('sw_gate_bias_init', -2.0))
        if self.use_sw_freq:
            self.sw_freq_branch = SpectralSWBranch(
                seq_len=self.seq_len,
                d_model=sw_out_dim,
            )
            self.sw_gate = nn.Sequential(
                nn.Linear(sw_out_dim + 5, 32),
                nn.SiLU(),
                nn.Linear(32, sw_out_dim),
            )
            nn.init.zeros_(self.sw_gate[-1].weight)
            nn.init.constant_(self.sw_gate[-1].bias, sw_gate_bias_init)

        # ==================== [C] FYObsEncoder（run61 核心：替代 h_spatial）====================
        # FY 邻域观测 → h_FY [B, basis_dim]
        # 无 FY 覆盖时 h_FY=0 → ETKF innovation=0 → update=0 → IRI baseline
        self.fy_obs_encoder = FYObsEncoder(
            feat_dim=10,
            d_model=basis_dim,
            n_heads=config.get('fy_enc_heads', 4),
            k_max=config.get('fy_nb_kmax', 64),
        )

        # ==================== [D] MODIP 球谐基（零参数，用于 SALR-H）====================
        self.sh_basis = MODIPSHBasis()

        # ==================== [C+] COSMICObsEncoder（run64：第三数据源）====================
        self.cosmic_obs_encoder = FYObsEncoder(
            feat_dim=10,
            d_model=basis_dim,
            n_heads=config.get('fy_enc_heads', 4),
            k_max=config.get('cosmic_nb_k_prof', 8) * config.get('cosmic_nb_n_alt', 8),
        )

        # ==================== [G] NeuralETKFLayer（run64：COSMIC 替代 vert 通道）====================
        enkf_n_members   = int(config.get('enkf_n_members', 8))
        enkf_pert_hidden = int(config.get('enkf_pert_hidden', 64))
        enkf_n_rank_h    = int(config.get('enkf_n_rank_h', 8))
        self.kalman_layer = NeuralETKFLayer(
            d_model=basis_dim,
            b_net_in=sw_out_dim + 5,
            r_fy_net_in=sw_out_dim + 6,
            n_members=enkf_n_members,
            pert_hidden=enkf_pert_hidden,
            n_rank_h=enkf_n_rank_h,
        )
        self.enkf_n_members = enkf_n_members

        # ==================== [H] FusionDecoder (run52: FiLM-Conditioned) ====================
        # run52 改动：regime 信号改用 FiLM 乘法路径调制 h_decode，而非 concat 加法路径。
        # 动机：run51 中 regime 4D / (64+6)D = 8.6%，被 h_decode 主导无法翻转修正符号；
        #       FiLM 通过 γ⊙h_decode + β 乘法控制每个通道，使 cos_SZA/ne_bkg_n 可有效
        #       压制或翻转夜间/IRI-高估场景的正向修正，修复密度图左上角拖尾。
        #
        # regime_film_net: (ne_bkg_n[1], cos_SZA[1], kp_eff[1], f107_eff[1]) = 4D
        #                  → Linear(4→32) → SiLU → Linear(32→128) → γ[64], β[64]
        # FiLM 残差形式: h_decode_mod = (1 + γ) ⊙ h_decode + β
        #   零初始化输出层 → γ=0, β=0 → h_decode_mod = h_decode（起步退化保证）
        # fusion_decoder: cat(h_decode_mod[64], alt_n[1], delta_alt_n[1]) = 66D → 64 → 1
        #   零初始化输出层 → Ne_delta_raw=0 → Ne_fused=Ne_bkg（IRI baseline 起步）
        self.regime_film_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, basis_dim * 2),   # → γ[64] ‖ β[64]
        )
        self.fusion_decoder = nn.Sequential(
            nn.Linear(basis_dim + 2, 64),   # 66D: h_decode_mod[64] + alt_n + delta_alt_n
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # ==================== [H+] Channel-wise Residual Fusion (CRF) ====================
        # run64: proj_pre 输入 192D（f_iri[64]+h_FY[64]+h_COSMIC[64]）
        self.proj_pre  = nn.Linear(basis_dim * 3, basis_dim)
        self.crf_alpha = nn.Parameter(torch.full((basis_dim,), 5.0))

        # ==================== 动态同化增益门控 (MultiScaleAdaptiveGate) ====================
        gate_h_scale    = config.get('gate_h_scale', 100.0)
        gate_regime_dim = int(config.get('gate_regime_dim', 6))   # run51: 4→6 (+cos_SZA, lat_n)
        self.gate_regime_dim = gate_regime_dim
        self.assim_gate = MultiScaleAdaptiveGate(
            in_dim=basis_dim + sw_out_dim + 2 + gate_regime_dim,   # 128 + 2 + 6 = 136
            basis_dim=basis_dim,
            h_scale=gate_h_scale,
        )

        # ==================== [F] 不确定性估计头 ====================
        self.uncertainty_head = nn.Sequential(
            nn.Linear(basis_dim + sw_out_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self._initialize_weights()
        self._initialize_cosmic_bootstrap()

    # ------------------------------------------------------------------ #

    def normalize_coords(self, lat, lon, alt, time):
        """坐标归一化到 [-1, 1]"""
        lat_n  = lat  / 90.0
        lon_n  = lon  / 180.0
        alt_n  = 2.0 * (alt  - self.alt_min) / (self.alt_max - self.alt_min) - 1.0
        time_n = (time / self.total_hours) * 2.0 - 1.0
        return lat_n, lon_n, alt_n, time_n

    def get_background_with_features(self, lat, lon, alt, time):
        """
        查询冻结的 IRI proxy，返回背景场和末层隐状态。
        IRI proxy 完全冻结，使用 torch.no_grad() 避免构建无效计算图。
        """
        coords_iri = torch.stack([lat, lon, alt, time], dim=-1)
        with torch.no_grad():
            ne_bkg, h_iri = self.iri_proxy(coords_iri, return_features=True)
        return ne_bkg, h_iri   # [B,1], [B,128]

    def forward(self, coords, sw_seq, precomputed_h_sw=None, giro_mode=False,
                iri_peak=None, neighbors_feats=None, has_obs=None,
                neighbors_feats_cosmic=None, has_obs_cosmic=None):
        """
        前向传播（run61：FYObsEncoder 替代 DualFreqSpatialNet）

        增量范式: Ne_fused = Ne_bkg + tanh(FusionDecoder(h_decode_mod, alt_n, δh_n)) × gate

        Args:
            coords:           [Batch, 4] 或 [Batch, 5] — (Lat_geo, Lon_geo, Alt, Time[, Lat_aacgm])
            sw_seq:           [Batch, Seq, 2] — (Kp_norm, F10.7_norm)
            precomputed_h_sw: [Batch, sw_out_dim] 可选优化
            giro_mode:        bool — GIRO 快速路径（仅运行 h_sw + IRI passthrough）
            iri_peak:         [Batch, 2] 可选 — [hmF2_IRI_km, NmF2_IRI_log10]，None → fallback (300, 11.5)
            neighbors_feats:  [Batch, K, 9] FY 邻域特征（可选）
            has_obs:          [Batch] float32，1.0 若有 FY 邻居（可选）

        Returns:
            Ne_fused:      [Batch, 1] — 最终预测（giro_mode 时为零占位）
            log_var:       [Batch, 1] — 不确定性对数方差（giro_mode 时为零）
            ne_placeholder:[Batch, 1] — 零占位（向后兼容第3位返回）
            Ne_delta:      [Batch, 1] — FusionDecoder 增量（giro_mode 时为零）
            extras:        dict
        """
        lat_geo = coords[:, 0]
        lon_geo = coords[:, 1]
        alt     = coords[:, 2]
        time    = coords[:, 3]
        B = coords.shape[0]

        # ---- 1. 坐标归一化 ----
        lat_n, lon_n, alt_n, time_n = self.normalize_coords(lat_geo, lon_geo, alt, time)

        # ---- 2. 周期编码（太阳天顶角/季节）----
        cos_SZA, sin_doy, cos_doy = _compute_solar_features(lat_geo, lon_geo, time)

        # ---- 3. IGRF 偶极子倾角特征 ----
        sin_I, cos_I = _compute_dip_features(lat_geo, lon_geo)

        # ---- 4. SW 编码 ----
        if precomputed_h_sw is not None:
            h_sw_time = precomputed_h_sw
            kp_eff    = sw_seq[:, -1, 0]
            f107_eff  = sw_seq[:, -1, 1]
        else:
            h_sw_time, kp_eff, f107_eff = self.sw_encoder(sw_seq)

        # ---- 4b. SW 频域分支（run40）----
        if self.use_sw_freq:
            h_sw_freq = self.sw_freq_branch(sw_seq)                          # [B, 64]
            sw_gate_in = torch.cat([
                h_sw_time,
                alt_n.unsqueeze(-1),
                sin_I.abs().unsqueeze(-1),
                cos_SZA.unsqueeze(-1),
                sin_doy.unsqueeze(-1),
                cos_doy.unsqueeze(-1),
            ], dim=-1)                                                        # [B, 69]
            sw_g = torch.sigmoid(self.sw_gate(sw_gate_in))                   # [B, 64]
            h_sw = sw_g * h_sw_freq + (1.0 - sw_g) * h_sw_time               # [B, 64]
        else:
            h_sw_freq = None
            sw_g      = None
            h_sw      = h_sw_time

        # ---- 5. MODIP 球谐基函数（零参数，SALR-H 用）----
        sh_feats = self.sh_basis(sin_I, cos_I, lat_geo, cos_SZA, sin_doy)  # [B, 14]

        # ---- IRI 峰场直接 passthrough（run61：绕过 PeakHead）----
        if iri_peak is not None:
            _iri_peak = iri_peak
        else:
            _iri_peak = torch.stack([
                torch.full((B,), 300.0, device=coords.device),
                torch.full((B,), 11.5,  device=coords.device),
            ], dim=-1)
        hmF2_IRI_km = _iri_peak[:, 0]                                    # [B]
        NmF2_IRI_n  = (_iri_peak[:, 1] - 11.0) / 2.0                    # [B] 归一化

        # run61: PeakHead 完全绕过 — hmF2_fused = hmF2_IRI，NmF2_fused = NmF2_IRI
        # 消除 h_spatial 污染 PeakHead 的根本缺陷
        hmF2_fused = hmF2_IRI_km.clone()
        NmF2_fused = _iri_peak[:, 1].clone()
        peak_params = {'hmF2': hmF2_fused, 'NmF2': NmF2_fused}

        delta_alt_iri = (alt - hmF2_IRI_km.detach()) / 190.0             # [B]

        # ---- GIRO 快速路径：仅返回 IRI passthrough 峰参数 ----
        if giro_mode:
            _zero_1 = torch.zeros(B, 1, device=coords.device, dtype=h_sw.dtype)
            _zero_d = torch.zeros(B, self.kalman_layer.d_model,
                                  device=coords.device, dtype=h_sw.dtype)
            _zero_iri = torch.zeros(B, 128, device=coords.device, dtype=h_sw.dtype)
            _zero_B   = torch.zeros(B, device=coords.device, dtype=h_sw.dtype)
            extras = {
                'ne_bkg':        _zero_1,
                'ne_chapman':    _zero_1,
                'ne_residual':   _zero_1,
                'h_spatial':     _zero_d,   # run61: h_FY; giro_mode → zero
                'h_sw':          h_sw,
                'h_fused':       _zero_d,
                'h_iri':         _zero_iri,
                'h_iri_aligned': _zero_d,
                'gate':          None,
                'gate_data':     None,
                'gate_phys':     None,
                'hmF2_det':      hmF2_fused.detach(),
                'kp_eff':        kp_eff,
                'f107_eff':      f107_eff,
                'peak_params':   peak_params,
                'chapman_params': peak_params,
                'sin_I':         sin_I,
                'cos_I':         cos_I,
                'hmF2_IRI_km':   hmF2_IRI_km,
                'NmF2_IRI_log10':_iri_peak[:, 1],
                'h_analysis':    _zero_d,
                'K_FY':          _zero_d,
                'K_COSMIC':      _zero_d,
                'K_vert':        _zero_d,   # backward-compat alias
                'b':             _zero_d,
                'r_fy':          _zero_d,
                'innov_FY':      _zero_d,
                'innov_COSMIC':  _zero_d,
                'innov_vert':    _zero_d,   # backward-compat alias
                'h_COSMIC':      _zero_d,
                'has_obs_cosmic': None,
                'alpha_eff':     _zero_B,
                'h_decode':      _zero_d,
                'h_pre':         _zero_d,
                'crf_alpha':     None,
                'regime_desc':   None,
                'head_weights':  None,
                'member_weights': None,
                'inflation_scale': None,
                'cos_SZA':       cos_SZA.detach(),
                'h_sw_time':     h_sw_time,
                'h_sw_freq':     h_sw_freq if self.use_sw_freq else None,
                'sw_g':          sw_g if self.use_sw_freq else None,
            }
            return _zero_1, _zero_1, _zero_1, _zero_1, extras

        # ---- Step A: IRI proxy（完全冻结，no_grad 提取 3D 结构隐状态）----
        ne_bkg, h_iri = self.get_background_with_features(lat_geo, lon_geo, alt, time)

        # ---- Step A2: IRI 峰对齐特征 ----
        h_iri_aligned = self.iri_align_net(torch.cat([
            h_iri,
            delta_alt_iri.unsqueeze(-1),
            NmF2_IRI_n.unsqueeze(-1),
        ], dim=-1))                                                    # [B, 64]

        # ---- Step C: 结构坐标（run61: hmF2_fused=hmF2_IRI → frame_offset=0 恒成立）----
        hmF2_det    = hmF2_fused.detach()                             # [B]
        delta_alt_n = (alt - hmF2_det) / 190.0                        # [B]
        frame_offset = (hmF2_det - hmF2_IRI_km.detach()) / 190.0      # [B] ≡ 0

        f_iri = h_iri_aligned + self.proj_frame_offset(
            frame_offset.unsqueeze(-1))                               # [B, 64]

        # ---- Step 5b: FY 邻域观测编码（run61 核心）----
        if neighbors_feats is not None:
            h_FY = self.fy_obs_encoder(neighbors_feats, has_obs)      # [B, basis_dim]
        else:
            h_FY = torch.zeros(B, self.kalman_layer.d_model,
                               device=coords.device, dtype=h_sw.dtype)
            if has_obs is None:
                has_obs = torch.zeros(B, device=coords.device, dtype=h_sw.dtype)

        # ---- Step 5c: COSMIC 邻域观测编码（run64）----
        if neighbors_feats_cosmic is not None:
            h_COSMIC = self.cosmic_obs_encoder(neighbors_feats_cosmic, has_obs_cosmic)
        else:
            h_COSMIC = torch.zeros(B, self.kalman_layer.d_model,
                                   device=coords.device, dtype=h_sw.dtype)
            if has_obs_cosmic is None:
                has_obs_cosmic = torch.zeros(B, device=coords.device, dtype=h_sw.dtype)

        # ---- 9. NeuralETKFLayer: 神经化集合卡尔曼同化（run64：FY + COSMIC 双源）----
        (h_analysis, K_FY, K_COSMIC,
         b_val, r_fy, innov_FY, innov_COSMIC) = self.kalman_layer(
            f_iri, h_FY, h_sw,
            lat_n, cos_SZA, sin_doy, cos_doy, sin_I,
            alt_n, delta_alt_n,
            alt, hmF2_det,
            sh_feats=sh_feats,
            has_obs=has_obs,
            h_obs_COSMIC=h_COSMIC,
            has_obs_cosmic=has_obs_cosmic,
        )

        # ---- CRF：三路 token（f_iri + h_FY + h_COSMIC）→ h_pre（run64: 192D）----
        h_pre     = self.proj_pre(torch.cat([f_iri, h_FY, h_COSMIC], dim=-1))  # [B, 64]
        crf_alpha = torch.sigmoid(self.crf_alpha)                      # [64] ∈(0,1)
        h_decode  = crf_alpha * h_analysis + (1.0 - crf_alpha) * h_pre # [B, 64]

        # ---- FiLM-Conditioned Fusion Decoder（run52）----
        ne_bkg_n = ne_bkg.detach() / 12.0 - 1.0
        film_in  = torch.stack([
            ne_bkg_n.squeeze(-1),
            cos_SZA,
            kp_eff,
            f107_eff,
        ], dim=-1)                                                      # [B, 4]
        film_out = self.regime_film_net(film_in)                       # [B, 128]
        gamma, beta = film_out[:, :64], film_out[:, 64:]
        h_decode_mod = (1.0 + gamma) * h_decode + beta                 # [B, 64]

        fusion_in    = torch.cat([
            h_decode_mod,
            alt_n.unsqueeze(-1),
            delta_alt_n.unsqueeze(-1),
        ], dim=-1)                                                      # [B, 66]
        Ne_delta_raw = self.fusion_decoder(fusion_in)

        # ---- 不确定性估计 ----
        unc_in  = torch.cat([h_analysis, h_sw], dim=-1)
        log_var = torch.clamp(self.uncertainty_head(unc_in), -10.0, 10.0)

        # ---- Gate（数据驱动 × 物理先验）----
        regime_desc = torch.stack([
            alt_n,
            sin_I.abs(),
            kp_eff,
            f107_eff,
            cos_SZA,
            lat_n,
        ], dim=-1)                                                      # [B, 6]

        # run61: gate_phys_net 接收 h_FY（替代原 h_spatial）
        _, gate_data, gate_phys = self.assim_gate(
            h_analysis, h_sw, alt, hmF2_det, h_FY,
            log_var_det=log_var.detach(),
            ne_delta_raw_abs=Ne_delta_raw.detach().abs(),
            cos_SZA=cos_SZA.unsqueeze(-1),
            sin_doy=sin_doy.unsqueeze(-1),
            cos_doy=cos_doy.unsqueeze(-1),
            regime_desc=regime_desc,
        )
        gate     = gate_data * gate_phys                               # [B, 1]
        Ne_delta = torch.tanh(Ne_delta_raw) * gate                     # [B, 1]

        # ---- 最终输出 ----
        Ne_fused = ne_bkg + Ne_delta
        ne_placeholder = torch.zeros_like(ne_bkg)

        _alpha_eff_out = (has_obs if has_obs is not None
                          else torch.zeros(B, device=coords.device))

        extras = {
            'ne_bkg':         ne_bkg,
            'ne_chapman':     ne_placeholder,
            'ne_residual':    Ne_delta,
            'h_spatial':      h_FY,            # run61: h_FY（语义更新，键名向后兼容）
            'h_sw':           h_sw,
            'h_fused':        h_analysis,
            'h_iri':          h_iri,
            'h_iri_aligned':  h_iri_aligned,
            'gate':           gate,
            'gate_data':      gate_data,
            'gate_phys':      gate_phys,
            'hmF2_det':       hmF2_det,
            'delta_alt_iri':  delta_alt_iri,
            'frame_offset':   frame_offset,
            'hmF2_IRI_km':    hmF2_IRI_km,
            'NmF2_IRI_log10': _iri_peak[:, 1],
            'kp_eff':         kp_eff,
            'f107_eff':       f107_eff,
            'peak_params':    peak_params,
            'chapman_params': peak_params,
            'sin_I':          sin_I,
            'cos_I':          cos_I,
            'h_analysis':     h_analysis,
            'K_FY':           K_FY,
            'K_COSMIC':       K_COSMIC,
            'K_vert':         K_COSMIC,   # backward-compat alias
            'b':              b_val,
            'r_fy':           r_fy,
            'innov_FY':       innov_FY,
            'innov_COSMIC':   innov_COSMIC,
            'innov_vert':     innov_COSMIC,   # backward-compat alias
            'alpha_eff':      _alpha_eff_out,
            'h_decode':       h_decode,
            'h_pre':          h_pre,
            'crf_alpha':      crf_alpha,
            'regime_desc':    regime_desc,
            'head_weights':     getattr(self.kalman_layer, 'last_head_weights', None),
            'member_weights':   getattr(self.kalman_layer, 'last_member_weights', None),
            'inflation_scale':  getattr(self.kalman_layer, 'last_inflation_scale', None),
            'alpha_r_day':      getattr(self.kalman_layer, 'alpha_r_day', None),
            'alpha_r_night':    getattr(self.kalman_layer, 'alpha_r_night', None),
            'r_ref_FY':   F.softplus(self.kalman_layer.log_r_ref_FY)     + 0.1,
            'r_ref_COSMIC': F.softplus(self.kalman_layer.log_r_ref_COSMIC) + 0.1,
            'r_ref_vert': F.softplus(self.kalman_layer.log_r_ref_COSMIC)   + 0.1,  # backward-compat
            'cos_SZA':          cos_SZA.detach(),
            'h_sw_time':      h_sw_time,
            'h_sw_freq':      h_sw_freq if self.use_sw_freq else None,
            'sw_g':           sw_g      if self.use_sw_freq else None,
            # run61/64 monitoring
            'h_FY':           h_FY,
            'h_COSMIC':       h_COSMIC,
            'has_obs':        has_obs,
            'has_obs_cosmic': has_obs_cosmic,
        }

        return Ne_fused, log_var, ne_placeholder, Ne_delta, extras

    def _initialize_weights(self):
        """关键层零初始化，确保训练初期退化为 IRI 先验（run61）"""
        # regime_film_net 输出层零初始化：γ=0, β=0 → h_decode_mod = h_decode（恒等起步）
        nn.init.zeros_(self.regime_film_net[-1].weight)
        nn.init.zeros_(self.regime_film_net[-1].bias)
        # FusionDecoder 输出层零初始化：Ne_delta 初始为 0
        nn.init.zeros_(self.fusion_decoder[-1].weight)
        nn.init.zeros_(self.fusion_decoder[-1].bias)
        # iri_align_net 输出层零初始化：初期 h_iri_aligned≈0
        nn.init.zeros_(self.iri_align_net[-1].weight)
        nn.init.zeros_(self.iri_align_net[-1].bias)
        # proj_frame_offset 零初始化：初期 frame_offset 投影=0（run61: frame_offset=0 恒成立）
        nn.init.zeros_(self.proj_frame_offset.weight)
        nn.init.zeros_(self.proj_frame_offset.bias)
        # iri_recon_head 零初始化：L_iri_struct 从零开始监督
        nn.init.zeros_(self.iri_recon_head.weight)
        nn.init.zeros_(self.iri_recon_head.bias)
        # uncertainty_head 输出层零初始化：初期 log_var=0
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.zeros_(self.uncertainty_head[-1].bias)
        # H_FY / H_COSMIC 零初始化（NeuralETKFLayer）：训练初期 update=0 → Ne_fused ≈ ne_bkg
        nn.init.zeros_(self.kalman_layer.H_FY_w)
        nn.init.zeros_(self.kalman_layer.H_COSMIC_w)
        # CRF: proj_pre 零初始化 → 初期 h_pre=0 → h_decode ≈ sigmoid(5)·h_analysis ≈ 0.9933·h_analysis
        nn.init.zeros_(self.proj_pre.weight)
        nn.init.zeros_(self.proj_pre.bias)

    def _initialize_cosmic_bootstrap(self):
        """Break the all-zero COSMIC encoder/H/proj_pre gradient deadlock."""
        self.cosmic_obs_encoder.input_proj.reset_parameters()
        nn.init.xavier_uniform_(
            self.proj_pre.weight[:, -self.kalman_layer.d_model:], gain=0.01)


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print('=' * 60)
    print('FSIA-INR run64 (FYObsEncoder + COSMICObsEncoder + IRI passthrough)')
    print('=' * 60)

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_managers.irinc_neural_proxy import IRINeuralProxy

    config = {
        'total_hours':     720.0,
        'alt_range':       (120.0, 500.0),
        'seq_len':         36,
        'basis_dim':       64,
        'sw_hidden_dim':   32,
        'sw_lstm_layers':  2,
        'sw_out_dim':      64,
        'tau_kp_init':     8.0,
        'tau_solar_init':  72.0,
        # run28-B core
        'enkf_n_members':  8,
        'enkf_pert_hidden': 64,
        'enkf_n_rank_h':   8,
        'gate_h_scale':    100.0,
        'gate_regime_dim': 6,
        # run40: SW 频域分支
        'use_sw_freq':         True,
        'sw_gate_bias_init':  -1.0,
        # run61: FYObsEncoder 超参数
        'fy_enc_heads': 4,
        'fy_nb_kmax':   64,
        # run64: COSMIC
        'cosmic_nb_k_prof': 8,
        'cosmic_nb_n_alt':  8,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    model = FSIA_INR_Model(iri_proxy, config).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'\n总参数量:    {total_params:,}')
    print(f'可训练参数:  {trainable_params:,}')

    # run64: 核心模块验证
    print(f'\nrun64 核心模块（应全为 True）:')
    print(f'  有 fy_obs_encoder:     {hasattr(model, "fy_obs_encoder")}  (FYObsEncoder)')
    print(f'  有 cosmic_obs_encoder: {hasattr(model, "cosmic_obs_encoder")}  (COSMICObsEncoder)')
    print(f'  有 kalman_layer:       {hasattr(model, "kalman_layer")}  (NeuralETKFLayer)')
    print(f'  有 fusion_decoder:     {hasattr(model, "fusion_decoder")}')
    print(f'  有 uncertainty_head:   {hasattr(model, "uncertainty_head")}')
    print(f'  有 assim_gate:         {hasattr(model, "assim_gate")}  (MultiScaleAdaptiveGate)')
    print(f'  有 proj_pre:           {hasattr(model, "proj_pre")}  (CRF 192D)')
    print(f'  有 crf_alpha:          {hasattr(model, "crf_alpha")}  (CRF)')
    print(f'  有 iri_align_net:      {hasattr(model, "iri_align_net")}')

    # FYObsEncoder 验证
    print(f'\nFYObsEncoder:')
    enc = model.fy_obs_encoder
    print(f'  input_proj: {enc.input_proj.in_features}→{enc.input_proj.out_features}  (应 10→64)')
    print(f'  input_proj w max: {enc.input_proj.weight.abs().max().item():.2e}  (应=0, 零初始化)')

    # COSMICObsEncoder 验证
    print(f'\nCOSMICObsEncoder (run64 新增):')
    cenc = model.cosmic_obs_encoder
    print(f'  input_proj: {cenc.input_proj.in_features}→{cenc.input_proj.out_features}  (应 10→64)')
    print(f'  input_proj w max: {cenc.input_proj.weight.abs().max().item():.2e}  (应>0, COSMIC 启动)')

    # CRF 维度验证
    print(f'\nCRF:')
    print(f'  proj_pre 输入维度:   {model.proj_pre.in_features}  (应=192 = 64×3，run64)')
    print(f'  crf_alpha shape:    {tuple(model.crf_alpha.shape)}  (应=(64,))')
    print(f'  sigmoid(crf_alpha): {torch.sigmoid(model.crf_alpha).mean().item():.4f}  (应≈0.9933)')

    # NeuralETKFLayer 验证
    print(f'\nNeuralETKFLayer:')
    print(f'  enkf_n_members:       {model.enkf_n_members}  (应=8)')
    print(f'  H_FY_w abs.max:       {model.kalman_layer.H_FY_w.abs().max().item():.2e}  (应=0)')
    print(f'  H_COSMIC_w abs.max:   {model.kalman_layer.H_COSMIC_w.abs().max().item():.2e}  (应=0)')
    print(f'  有 H_COSMIC_a:        {hasattr(model.kalman_layer, "H_COSMIC_a")}  (应=True)')
    print(f'  有 R_COSMIC_net:      {hasattr(model.kalman_layer, "R_COSMIC_net")}  (应=True)')
    print(f'  无 log_R_vert:        {not hasattr(model.kalman_layer, "log_R_vert")}  (run64 移除)')

    B = 64
    K = 32
    coords = torch.zeros(B, 4, device=device)
    coords[:, 0] = torch.linspace(-60, 60, B)
    coords[:, 1] = torch.linspace(-180, 180, B)
    coords[:, 2] = torch.linspace(150, 450, B)
    coords[:, 3] = torch.linspace(0, 720, B)
    sw_seq = torch.randn(B, 36, 2, device=device)

    iri_peak_test = torch.stack([
        torch.full((B,), 300.0, device=device),
        torch.full((B,), 11.5,  device=device)
    ], dim=-1)

    # 测试 1: 无任何观测（退化为 IRI baseline）
    Ne_fused_nobs, log_var, _, Ne_delta, extras_nobs = model(
        coords, sw_seq, iri_peak=iri_peak_test)

    print(f'\n=== 测试 1: 无观测（IRI baseline）===')
    print(f'  Ne_fused shape:     {Ne_fused_nobs.shape}')
    print(f'  |Ne_delta| max:     {Ne_delta.abs().max().item():.2e}  (应≈0)')
    ne_diff_nobs = (Ne_fused_nobs - extras_nobs["ne_bkg"]).abs().max().item()
    print(f'  |Ne_fused-ne_bkg|:  {ne_diff_nobs:.2e}  (应≈0)')

    # 测试 2: 有 FY 邻居
    neighbors_feats = torch.randn(B, K, 10, device=device)
    has_obs = torch.ones(B, device=device)
    Ne_fused_obs, _, _, Ne_delta_obs, extras_obs = model(
        coords, sw_seq, iri_peak=iri_peak_test,
        neighbors_feats=neighbors_feats, has_obs=has_obs)

    print(f'\n=== 测试 2: 有 FY 邻居 ===')
    print(f'  Ne_fused shape:     {Ne_fused_obs.shape}')
    print(f'  h_FY shape:         {extras_obs["h_FY"].shape}  (应=[{B}, 64])')
    print(f'  innov_FY norm:      {extras_obs["innov_FY"].norm(dim=-1).mean().item():.4f}')

    # 测试 3: 有 COSMIC 邻居
    nb_csm = torch.randn(B, K, 10, device=device)
    has_csm = torch.ones(B, device=device)
    Ne_fused_csm, _, _, _, extras_csm = model(
        coords, sw_seq, iri_peak=iri_peak_test,
        neighbors_feats=neighbors_feats, has_obs=has_obs,
        neighbors_feats_cosmic=nb_csm, has_obs_cosmic=has_csm)
    print(f'\n=== 测试 3: FY + COSMIC ===')
    print(f'  Ne_fused shape:     {Ne_fused_csm.shape}')
    print(f'  h_COSMIC shape:     {extras_csm["h_COSMIC"].shape}  (应=[{B}, 64])')
    print(f'  K_COSMIC shape:     {extras_csm["K_COSMIC"].shape}')
    print(f'  innov_COSMIC norm:  {extras_csm["innov_COSMIC"].norm(dim=-1).mean().item():.4f}')

    # 测试 4: 混合（50% 有 FY 观测，50% 无）
    has_obs_mixed = (torch.rand(B, device=device) > 0.5).float()
    Ne_fused_mix, _, _, _, _ = model(
        coords, sw_seq, iri_peak=iri_peak_test,
        neighbors_feats=neighbors_feats, has_obs=has_obs_mixed)
    print(f'\n=== 测试 4: 混合 FY 观测 ===')
    print(f'  has_obs=1 数量:     {has_obs_mixed.sum().int().item()} / {B}')
    print(f'  Ne_fused shape:     {Ne_fused_mix.shape}')

    # 梯度测试（FY+COSMIC 联合）
    model.zero_grad()
    loss = Ne_fused_csm.sum()
    loss.backward()
    iri_any_grad = any(p.grad is not None for p in model.iri_proxy.parameters())
    print(f'\n=== 梯度测试 ===')
    print(f'  IRI proxy 任意参数有梯度:   {iri_any_grad}  (应=False)')
    print(f'  H_FY_w grad:               {model.kalman_layer.H_FY_w.grad is not None}')
    print(f'  H_COSMIC_w grad:           {model.kalman_layer.H_COSMIC_w.grad is not None}')
    print(f'  cosmic_obs_encoder grad:   {model.cosmic_obs_encoder.query.grad is not None}')

    # gate 验证
    gate = extras_csm.get('gate')
    print(f'\n=== Gate ===')
    if gate is not None:
        print(f'  gate shape: {gate.shape}')
        print(f'  gate range: [{gate.min().item():.4f}, {gate.max().item():.4f}]')

    # giro_mode
    model.eval()
    with torch.no_grad():
        out_giro = model(coords, sw_seq, giro_mode=True, iri_peak=iri_peak_test)
    giro_hmf2 = out_giro[4]['peak_params']['hmF2']
    iri_hmf2  = iri_peak_test[:, 0]
    print(f'\n=== giro_mode 测试 ===')
    print(f'  peak_params hmF2 = iri_peak hmF2: {torch.allclose(giro_hmf2, iri_hmf2)}')
    print(f'  gate: {out_giro[4]["gate"]}  (应=None)')

    # 兼容性
    assert 'hmF2' in extras_csm['chapman_params']
    assert 'K_vert' in extras_csm      # backward-compat alias
    assert 'innov_vert' in extras_csm  # backward-compat alias
    print(f'\n  chapman_params alias: OK')
    print(f'  K_vert / innov_vert backward-compat: OK')

    print('\n所有测试通过!')
