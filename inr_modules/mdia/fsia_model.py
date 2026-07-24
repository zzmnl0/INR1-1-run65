"""FSIA-INR model with frozen IRI background and local FY/COSMIC assimilation.

The active path is: IRI hidden state + two profile encoders -> NeuralETKFLayer ->
channel-wise fusion -> bounded density increment. IRI hmF2/NmF2 are passed through
as structural references; no trainable peak head is present.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ewma_sw_encoder import DualScaleSWEncoder
    from .mdia_model import _compute_dip_features
except ImportError:
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from ewma_sw_encoder import DualScaleSWEncoder
    from mdia_model import _compute_dip_features


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
    """Build the shared FY/COSMIC gain-calibration input."""
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
    """Ensemble-space FY/COSMIC update over the encoded IRI state.

    Perturbation networks generate a centered ensemble. Independent global
    observation operators form FY and COSMIC innovations; source masks are
    applied separately before the two member-weight updates are combined.
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

        # Frozen inference calibration. Its output is detached below, so these
        # weights cannot receive gradients in the current architecture.
        self.R_FY_net = nn.Sequential(
            nn.Linear(r_fy_net_in, 64), nn.SiLU(), nn.Linear(64, d_model))
        self.R_FY_net.requires_grad_(False)

        # ---- FY observation operator (global H, zero-init) ----
        self.H_FY_w   = nn.Parameter(torch.zeros(d_model, d_model))

        # run65 checkpoint compatibility only. These SALR-H factors were all
        # zero-initialized, so their product had identically zero gradients.
        # Keep the state keys for strict loading, but exclude them from training.
        _salr_cond = 17
        self.n_rank_h  = n_rank_h
        self.H_FY_u    = nn.Parameter(
            torch.zeros(n_rank_h, d_model), requires_grad=False)
        self.H_FY_v    = nn.Parameter(
            torch.zeros(n_rank_h, d_model), requires_grad=False)
        self.H_FY_a    = nn.Linear(_salr_cond, n_rank_h, bias=True)
        nn.init.zeros_(self.H_FY_a.weight)
        nn.init.zeros_(self.H_FY_a.bias)
        self.H_FY_a.requires_grad_(False)

        # ---- run64: COSMIC observation channel (replaces vert channel) ----
        # H_COSMIC_w zero-init -> COSMIC channel inactive at start -> IRI baseline
        self.H_COSMIC_w  = nn.Parameter(torch.zeros(d_model, d_model))
        self.H_COSMIC_u  = nn.Parameter(
            torch.zeros(n_rank_h, d_model), requires_grad=False)
        self.H_COSMIC_v  = nn.Parameter(
            torch.zeros(n_rank_h, d_model), requires_grad=False)
        self.H_COSMIC_a  = nn.Linear(_salr_cond, n_rank_h, bias=True)
        nn.init.zeros_(self.H_COSMIC_a.weight)
        nn.init.zeros_(self.H_COSMIC_a.bias)
        self.H_COSMIC_a.requires_grad_(False)
        # Independent frozen COSMIC inference calibration.
        self.R_COSMIC_net = nn.Sequential(
            nn.Linear(r_fy_net_in, 64), nn.SiLU(), nn.Linear(64, d_model))
        self.R_COSMIC_net.requires_grad_(False)
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
        """Return the fixed FY gain-calibration field."""
        return F.softplus(self.R_FY_net(r_input))                               # [B, d]

    def _eval_R_COSMIC(self, r_input):
        """Return the fixed COSMIC gain-calibration field."""
        return F.softplus(self.R_COSMIC_net(r_input))                           # [B, d]

    def forward(self, f_iri, h_obs_FY, h_sw,
                lat_n, cos_SZA, sin_doy, cos_doy, sin_I, alt_n, delta_alt_n,
                has_obs=None, h_obs_COSMIC=None, has_obs_cosmic=None):
        b_in = _build_kalman_b_input(h_sw, lat_n, cos_SZA, sin_doy, cos_doy, sin_I)
        r_in = _build_kalman_r_fy_input(alt_n, delta_alt_n,
                                         cos_SZA, sin_doy, cos_doy, lat_n.abs(), h_sw)

        # ---- Step 1: ensemble perturbations + centering + inflation ----
        delta      = self._eval_perturbations(b_in)
        delta_mean = delta.mean(dim=1, keepdim=True)
        X          = delta - delta_mean
        inflation  = torch.exp(self.log_inflation)
        X_inf      = X * torch.sqrt(inflation)

        # ---- Step 2: frozen source-specific gain calibration ----
        r_fy = self._eval_R_FY(r_in)
        use_cosmic = (h_obs_COSMIC is not None)
        if use_cosmic:
            r_cosmic = self._eval_R_COSMIC(r_in)
        else:
            r_cosmic = torch.zeros_like(r_fy)

        # ---- Step 3: global observation projection ----
        HX_FY     = torch.einsum('bnd,do->bno', X_inf, self.H_FY_w)
        HX_COSMIC = torch.einsum('bnd,do->bno', X_inf, self.H_COSMIC_w)

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
        w_FY_pre  = torch.einsum('bnd,bd->bn', HXR_FY, innov_FY)
        w_FY_base = torch.einsum('bnm,bm->bn', T_FY, w_FY_pre)

        if use_cosmic:
            innov_COSMIC = h_obs_COSMIC - torch.einsum('bd,do->bo', f_iri, self.H_COSMIC_w)
            w_COSMIC_pre  = torch.einsum('bnd,bd->bn', HXR_COSMIC, innov_COSMIC)
            w_COSMIC_base = torch.einsum('bnm,bm->bn', T_COSMIC, w_COSMIC_pre)
        else:
            innov_COSMIC  = torch.zeros_like(innov_FY)
            w_COSMIC_base = torch.zeros(X.shape[0], N, device=X.device, dtype=X.dtype)

        # Inference gain suppression. Detach preserves run65 behavior and makes
        # the calibration networks fixed rather than trainable.
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
        self.last_inflation_scale = inflation.detach()

        return (h_analysis, K_eff_FY, K_eff_COSMIC,
                ens_var, r_fy, innov_FY, innov_COSMIC)


# ======================== MultiScaleAdaptiveGate ========================

class MultiScaleAdaptiveGate(nn.Module):
    """Multiply a learned regime gate by a profile-height prior gate."""

    def __init__(self, in_dim: int = 130, basis_dim: int = 64,
                 h_scale: float = 100.0):
        super().__init__()
        self.h_scale = h_scale

        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, math.log(0.95 / 0.05))  # ≈2.944

        self.gate_phys_net = nn.Sequential(
            nn.Linear(basis_dim + 3, 32), nn.SiLU(), nn.Linear(32, 6))
        nn.init.zeros_(self.gate_phys_net[-1].weight)
        nn.init.zeros_(self.gate_phys_net[-1].bias)

        self.register_buffer('w_prior',
            torch.tensor([2.5, -1.5, -0.3]))
        self.register_buffer('log_sigma_prior',
            torch.tensor([math.log(0.8), math.log(0.5), math.log(1.2)]))
        self.register_buffer('centers',
            torch.tensor([0.0, -1.5, 2.0]))

    def forward(self, h_fused, h_sw, alt_km, hmF2_det, h_profile,
                log_var_det, ne_delta_raw_abs,
                cos_SZA, sin_doy, cos_doy, regime_desc):
        """Return total, learned, and profile-height gates, each shaped [B, 1]."""
        gate_data = torch.sigmoid(self.gate_net(torch.cat(
            [h_fused, h_sw, log_var_det, ne_delta_raw_abs, regime_desc], dim=-1)))  # [B, 1]

        gate_phys_input = torch.cat([h_profile, cos_SZA, sin_doy, cos_doy], dim=-1)  # [B, 67]
        dp    = self.gate_phys_net(gate_phys_input)                # [B, 6]
        w     = self.w_prior     + dp[:, :3]                       # [B, 3]
        sigma = torch.exp(self.log_sigma_prior + dp[:, 3:])        # [B, 3] > 0

        delta = ((alt_km - hmF2_det) / self.h_scale).unsqueeze(-1) # [B, 1]
        gauss = torch.exp(-((delta - self.centers) / sigma) ** 2)  # [B, 3]
        gate_phys = torch.sigmoid(
            (gauss * w).sum(dim=-1, keepdim=True))                 # [B, 1]

        return gate_data * gate_phys, gate_data, gate_phys


# ======================== SpectralSWBranch ========================

class SpectralSWBranch(nn.Module):
    """Encode rFFT magnitudes with attention-weighted frequency pooling."""

    def __init__(self, d_model: int = 64):
        super().__init__()
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


# ======================== FYObsEncoder ========================

class FYObsEncoder(nn.Module):
    """Encode K local 10-D profile samples; return zero when unobserved."""
    def __init__(self, feat_dim: int = 10, d_model: int = 64,
                 n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
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
    """Feature-space FY/COSMIC assimilation over a frozen IRI background."""

    def __init__(self, iri_proxy, config):
        super().__init__()

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
        self.iri_proxy.freeze()

        # ---- IRI 峰对齐特征网络 ----
        # 输入: cat(h_iri[128], delta_alt_iri[1], NmF2_IRI_n[1]) = 130D
        self.iri_align_net = nn.Sequential(
            nn.Linear(iri_hidden_dim + 2, 128),
            nn.SiLU(),
            nn.Linear(128, basis_dim),
        )
        # The weight is a frozen run65 checkpoint key; the input is always zero.
        self.proj_frame_offset = nn.Linear(1, basis_dim)
        self.proj_frame_offset.weight.requires_grad_(False)
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
        )

        # ==================== [C+] COSMICObsEncoder（run64：第三数据源）====================
        self.cosmic_obs_encoder = FYObsEncoder(
            feat_dim=10,
            d_model=basis_dim,
            n_heads=config.get('fy_enc_heads', 4),
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

    def get_background_with_features(self, lat, lon, alt, time):
        """
        查询冻结的 IRI proxy，返回背景场和末层隐状态。
        IRI proxy 完全冻结，使用 torch.no_grad() 避免构建无效计算图。
        """
        coords_iri = torch.stack([lat, lon, alt, time], dim=-1)
        with torch.no_grad():
            ne_bkg, h_iri = self.iri_proxy(coords_iri, return_features=True)
        return ne_bkg, h_iri   # [B,1], [B,128]

    def forward(self, coords, sw_seq, precomputed_h_sw=None,
                iri_peak=None, neighbors_feats=None, has_obs=None,
                neighbors_feats_cosmic=None, has_obs_cosmic=None):
        """
        前向传播（run61：FYObsEncoder 替代 DualFreqSpatialNet）

        增量范式: Ne_fused = Ne_bkg + tanh(FusionDecoder(h_decode_mod, alt_n, δh_n)) × gate

        Args:
            coords:           [Batch, 4] 或 [Batch, 5] — (Lat_geo, Lon_geo, Alt, Time[, Lat_aacgm])
            sw_seq:           [Batch, Seq, 2] — (Kp_norm, F10.7_norm)
            precomputed_h_sw: [Batch, sw_out_dim] 可选优化
            iri_peak:         [Batch, 2] 可选 — [hmF2_IRI_km, NmF2_IRI_log10]，None → fallback (300, 11.5)
            neighbors_feats:  [Batch, K, 9] FY 邻域特征（可选）
            has_obs:          [Batch] float32，1.0 若有 FY 邻居（可选）

        Returns:
            Ne_fused:      [Batch, 1] — 最终预测
            log_var:       [Batch, 1] — 不确定性对数方差
            ne_placeholder:[Batch, 1] — 零占位（向后兼容第3位返回）
            Ne_delta:      [Batch, 1] — FusionDecoder 增量
            extras:        dict
        """
        lat_geo = coords[:, 0]
        lon_geo = coords[:, 1]
        alt     = coords[:, 2]
        time    = coords[:, 3]
        B = coords.shape[0]

        # ---- 1. 坐标归一化 ----
        lat_n = lat_geo / 90.0
        alt_n = 2.0 * (alt - self.alt_min) / (self.alt_max - self.alt_min) - 1.0

        # ---- 2. 周期编码（太阳天顶角/季节）----
        cos_SZA, sin_doy, cos_doy = _compute_solar_features(lat_geo, lon_geo, time)

        # ---- 3. IGRF 偶极子倾角特征 ----
        sin_I, _ = _compute_dip_features(lat_geo, lon_geo)

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

        # ---- IRI peak structural reference (direct passthrough) ----
        if iri_peak is not None:
            _iri_peak = iri_peak
        else:
            _iri_peak = torch.stack([
                torch.full((B,), 300.0, device=coords.device),
                torch.full((B,), 11.5,  device=coords.device),
            ], dim=-1)
        hmF2_IRI_km = _iri_peak[:, 0]                                    # [B]
        NmF2_IRI_n  = (_iri_peak[:, 1] - 11.0) / 2.0                    # [B] 归一化

        # No trainable peak head: keep the IRI hmF2/NmF2 references unchanged.
        hmF2_fused = hmF2_IRI_km.clone()
        NmF2_fused = _iri_peak[:, 1].clone()
        peak_params = {'hmF2': hmF2_fused, 'NmF2': NmF2_fused}

        delta_alt_iri = (alt - hmF2_IRI_km.detach()) / 190.0             # [B]

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
        # hmF2 is passed through from IRI, so frame_offset is identically zero.
        # Use the learned bias directly and skip the dead matrix multiply.
        f_iri = h_iri_aligned + self.proj_frame_offset.bias            # [B, 64]

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
        gate, gate_data, gate_phys = self.assim_gate(
            h_analysis, h_sw, alt, hmF2_det, h_FY,
            log_var_det=log_var.detach(),
            ne_delta_raw_abs=Ne_delta_raw.detach().abs(),
            cos_SZA=cos_SZA.unsqueeze(-1),
            sin_doy=sin_doy.unsqueeze(-1),
            cos_doy=cos_doy.unsqueeze(-1),
            regime_desc=regime_desc,
        )
        Ne_delta = torch.tanh(Ne_delta_raw) * gate                     # [B, 1]

        # ---- 最终输出 ----
        Ne_fused = ne_bkg + Ne_delta
        ne_placeholder = torch.zeros_like(ne_bkg)

        extras = {
            'ne_bkg':         ne_bkg,
            'ne_residual':    Ne_delta,
            'h_iri_aligned':  h_iri_aligned,
            'gate':           gate,
            'gate_data':      gate_data,
            'gate_phys':      gate_phys,
            'hmF2_det':       hmF2_det,
            'peak_params':    peak_params,
            'K_FY':           K_FY,
            'K_COSMIC':       K_COSMIC,
            'b':              b_val,
            'r_fy':           r_fy,
            'innov_FY':       innov_FY,
            'innov_COSMIC':   innov_COSMIC,
            'member_weights':   getattr(self.kalman_layer, 'last_member_weights', None),
            'inflation_scale':  getattr(self.kalman_layer, 'last_inflation_scale', None),
            'r_ref_FY':   F.softplus(self.kalman_layer.log_r_ref_FY)     + 0.1,
            'r_ref_COSMIC': F.softplus(self.kalman_layer.log_r_ref_COSMIC) + 0.1,
            'h_FY':           h_FY,
            'h_COSMIC':       h_COSMIC,
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
    print('FSIA-INR (FY/COSMIC local-profile assimilation)')
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

    print(f'\n核心模块（应全为 True）:')
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
    print(f'\nCOSMICObsEncoder:')
    cenc = model.cosmic_obs_encoder
    print(f'  input_proj: {cenc.input_proj.in_features}→{cenc.input_proj.out_features}  (应 10→64)')
    print(f'  input_proj w max: {cenc.input_proj.weight.abs().max().item():.2e}  (应>0, COSMIC 启动)')

    # CRF 维度验证
    print(f'\nCRF:')
    print(f'  proj_pre 输入维度:   {model.proj_pre.in_features}  (应=192 = 64×3)')
    print(f'  crf_alpha shape:    {tuple(model.crf_alpha.shape)}  (应=(64,))')
    print(f'  sigmoid(crf_alpha): {torch.sigmoid(model.crf_alpha).mean().item():.4f}  (应≈0.9933)')

    # NeuralETKFLayer 验证
    print(f'\nNeuralETKFLayer:')
    print(f'  enkf_n_members:       {model.enkf_n_members}  (应=8)')
    print(f'  H_FY_w abs.max:       {model.kalman_layer.H_FY_w.abs().max().item():.2e}  (应=0)')
    print(f'  H_COSMIC_w abs.max:   {model.kalman_layer.H_COSMIC_w.abs().max().item():.2e}  (应=0)')
    print(f'  H_COSMIC_a 已冻结:    {not model.kalman_layer.H_COSMIC_a.weight.requires_grad}  (应=True)')
    print(f'  有 R_COSMIC_net:      {hasattr(model.kalman_layer, "R_COSMIC_net")}  (应=True)')
    print(f'  R_COSMIC_net 已冻结:  {not model.kalman_layer.R_COSMIC_net[0].weight.requires_grad}  (应=True)')

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

    assert torch.isfinite(Ne_fused_csm).all()

    print('\n所有测试通过!')
