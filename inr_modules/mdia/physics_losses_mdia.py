"""
FSIA-INR 物理约束损失函数

损失函数体系（共 5 项物理约束）：

1. L_bkg       — IRI 背景信任损失（高度+地方时自适应；夜间全局增强，低高度强约束）
2. L_uplift    — 赤道 ExB 上涌先验（磁赤道感知 sin_I + 日间 LT 门控；仅白天激活）
3. L_depletion — 赤道喷泉耗散先验（磁赤道感知 sin_I + 日间 LT 门控；仅白天激活）
4. L_profile   — 廓线-峰高对齐（∂Ne_fused/∂h=0 @ hmF2_pred，单独调用 profile_peak_alignment_loss）

run65 移除：L_smooth（∂²Ne_delta/∂h²）和 L_horiz（∂Ne/∂lat,lon）— SIREN 输出本身 C∞
平滑，两项在主批次计算 coords.requires_grad=True，占 backward 耗时约 40%，移除后加速显著。

TEC（tec_direction_consistency_loss）和 NeQuick 相关项（chapman_shape_loss）已从 FSIA-INR
中完全移除：TEC 仅约束积分量、分辨率不足；NeQuick 被 PeakHead 替代。

GIRO 监督（hmF2 / NmF2）通过 giro_dataloader.py::compute_giro_loss() 在训练循环中独立计算。

背景信任权重设计（v3.1，run25 SZA 化）：
    - 移除 lambda_B（sin_I/lat 赤道抑制）：赤道 IRI 误差已由 GIRO 监督显式修正
    - 改为 SZA + 高度 双维度自适应（run25: LT → cos_SZA，含日期变化）：
        w_bkg_eff = w_alt_factor + w_bkg_night × night_gate × (2 - _ratio)
        w_alt_factor  = w_bkg_high × _ratio + w_bkg_low × (1 - _ratio)        (高度因子)
        night_gate    = sigmoid(-cos_SZA × 5)                                  (夜间=1，正午=0)
        (2 - _ratio)  ≈ 2 at 120km, ≈ 1 at 400km                              (低高度夜间额外加倍)
    - 赤道 ExB 先验（uplift/depletion）限定在日间（lt_day_gate = relu(cos_SZA)）

梯度连续性设计原则：
    - 所有掩码使用 exp(-(x/σ)²)，禁止硬阈值 Boolean 掩码
    - 禁止 torch.abs(lat)，对称函数用 lat**2/σ² 计算
"""

import math as _math

import torch
import torch.nn.functional as F


# ======================== Solar features（run25：替代 LT 计算）========================

_DOY_SEP1_FLOAT = 245.0  # Sep 1 day-of-year — 与 fsia_model._compute_solar_features 同源

def _compute_cos_sza_from_coords(coords):
    """
    从 coords [B,4] 计算 cos(SZA)。run25：替代 LT 余弦门控。

    Returns:
        cos_SZA: [B] ∈ [-1, 1]
    """
    lat_deg  = coords[:, 0].detach()
    lon_deg  = coords[:, 1].detach()
    rel_hour = coords[:, 3].detach()

    doy_mod = ((rel_hour / 24.0).floor() + _DOY_SEP1_FLOAT) % 365.0
    gamma   = 2.0 * _math.pi * doy_mod / 365.0
    delta   = (0.006918
               - 0.399912 * torch.cos(gamma)
               + 0.070257 * torch.sin(gamma)
               - 0.006758 * torch.cos(2.0 * gamma)
               + 0.000907 * torch.sin(2.0 * gamma)
               - 0.002697 * torch.cos(3.0 * gamma)
               + 0.001480 * torch.sin(3.0 * gamma))                     # [B] rad
    LST     = (rel_hour + lon_deg / 15.0) % 24.0
    HA      = (LST - 12.0) * (_math.pi / 12.0)
    lat_rad = lat_deg * (_math.pi / 180.0)
    cos_SZA = (torch.sin(delta) * torch.sin(lat_rad)
               + torch.cos(delta) * torch.cos(lat_rad) * torch.cos(HA))
    return cos_SZA


def banded_pearson_loss(x, y, lat_geo_deg, eps=1e-8, min_samples=10):
    """
    分纬度带 Pearson 损失（run25：NmF2 仅形状约束 — 系统偏差不锚定）

    分 3 带：[0,30°) 低纬 EIA / [30°,50°) 中纬 / [50°,90°] 高纬
    每带至少 min_samples 才计入；各带 (1 - Pearson_r) 等权平均。

    Args:
        x, y:           [B] — 待对比量（NmF2_fused, NmF2_IRI_log10）
        lat_geo_deg:    [B] — 地理纬度（度）
        min_samples:    每带最低样本数阈值
    Returns:
        scalar Tensor — 各带 (1 - Pearson_r) 平均；全跳过时返回 0
    """
    abs_lat = lat_geo_deg.abs()
    bands = [
        abs_lat <  30.0,                            # 低纬（EIA / 赤道）
        (abs_lat >= 30.0) & (abs_lat < 50.0),       # 中纬
        abs_lat >= 50.0,                             # 高纬
    ]
    losses = []
    for mask in bands:
        if mask.sum().item() < min_samples:
            continue
        xb = x[mask]; yb = y[mask]
        xc = xb - xb.mean()
        yc = yb - yb.mean()
        cov   = (xc * yc).mean()
        std_x = (xc.pow(2).mean() + eps).sqrt()
        std_y = (yc.pow(2).mean() + eps).sqrt()
        losses.append(1.0 - cov / (std_x * std_y + eps))
    if not losses:
        return torch.tensor(0.0, device=x.device)
    return torch.stack(losses).mean()


# ======================== 1. 背景信任损失（v3.1：LT + 高度自适应，移除 lambda_B）========================

def background_trust_loss(pred_ne, ne_bkg, lat, sin_I=None):
    """
    IRI 背景信任损失（兼容接口，内部已迁移至 combined_mdia_physics_loss 自适应实现）

    保留此函数用于独立调用场景；训练循环通过 combined_mdia_physics_loss 调用。
    """
    diff_sq = (pred_ne - ne_bkg.detach()) ** 2
    return torch.mean(diff_sq)


# ======================== 2. 赤道 ExB 上涌先验损失 ========================

def trough_uplift_loss(hmF2_pred, lat_deg,
                       threshold_km=380.0, equator_halfwidth=5.0,
                       sin_I=None, lt_day_gate=None):
    """
    赤道 ExB 上涌先验约束（EIA trough uplift prior）

    物理依据：喷泉效应（日间）使赤道附近 hmF2 显著高于中纬度（通常 350-450 km）。
    日间 LT 门控防止将夜间等离子体下漂方向错误地约束为上涌。

    L_uplift = mean(smooth_mask × lt_day_gate × relu(threshold - hmF2)²)

    threshold_km 调整历史：
        300km (run13-16) → 日间赤道 hmF2 通常 >300km，loss 不激活，无效
        380km (run17+)   → 覆盖典型日间赤道 hmF2 范围，梯度持续有效

    Args:
        hmF2_pred:        [Batch] F2 峰高预测 (km)，保留梯度
        lat_deg:          [Batch] 地理纬度 (度)，已 detach
        threshold_km:     低于此值才惩罚 (默认 380 km)
        equator_halfwidth:高斯掩码半宽度 (度, 默认 5°)；用于 sin_I 空间，对应 ±5° 磁纬
        sin_I:            [Batch] 地磁倾角正弦（可选）；提供时用磁赤道作为惩罚中心
        lt_day_gate:      [Batch] 日间门控权重 ∈ [0,1]（可选）；
                          LT=12 时≈1（最大惩罚），LT=0/24 时≈0（夜间不激活）

    Returns:
        scalar loss
    """
    if sin_I is not None:
        sigma_I = _math.sin(equator_halfwidth * _math.pi / 180.0)
        smooth_mask = torch.exp(-(sin_I / sigma_I) ** 2).detach()
    else:
        smooth_mask = torch.exp(-(lat_deg / equator_halfwidth) ** 2).detach()

    if lt_day_gate is not None:
        smooth_mask = smooth_mask * lt_day_gate.detach()

    return torch.mean(smooth_mask * torch.relu(threshold_km - hmF2_pred) ** 2)


# ======================== 3. 赤道喷泉耗散损失 ========================

def trough_depletion_loss(NmF2_pred, lat_deg, equator_sigma=7.0, sin_I=None, lt_day_gate=None):
    """
    赤道喷泉耗散约束（Fountain Effect Trough Depletion Prior）

    物理依据：喷泉效应（日间）将赤道等离子体向两侧 EIA 双峰输运，
    使磁赤道处 NmF2 低于 EIA 双峰（赤道波谷）。夜间 EIA 消散，不应施压。

    L_depletion = mean(weight × lt_day_gate × NmF2_pred)

    Args:
        NmF2_pred:     [Batch] F2 峰值密度预测 (log10 空间)，保留梯度
        lat_deg:       [Batch] 地理纬度 (度)，已 detach
        equator_sigma: 高斯掩码宽度 (度, 默认 7°)；用于 sin_I 空间
        sin_I:         [Batch] 地磁倾角正弦（可选）；提供时惩罚磁赤道
        lt_day_gate:   [Batch] 日间门控权重 ∈ [0,1]（可选）；夜间为 0

    Returns:
        scalar loss
    """
    if sin_I is not None:
        sigma_I = _math.sin(equator_sigma * _math.pi / 180.0)
        depletion_weight = torch.exp(-(sin_I / sigma_I) ** 2).detach()
    else:
        depletion_weight = torch.exp(-(lat_deg / equator_sigma) ** 2).detach()

    if lt_day_gate is not None:
        depletion_weight = depletion_weight * lt_day_gate.detach()

    return torch.mean(depletion_weight * NmF2_pred)


# ======================== 4. EIA 双峰增强损失 ========================

def eia_crest_enhance_loss(NmF2_pred, sin_I, lt_day_gate=None,
                           crest_sinI=0.22, crest_sigma=0.10,
                           nmf2_floor=11.0):
    """
    EIA 双峰增强先验（Equatorial Ionization Anomaly Crest Enhancement Prior）

    物理依据：喷泉效应将等离子体从磁赤道输运至 ±15° 磁纬（EIA 双峰），
    日间 EIA 波峰 NmF2 明显高于赤道波谷。
    本损失约束 EIA 波峰位置（|sin_I| ≈ 0.22，对应 ±12° 磁纬）日间 NmF2 不低于下限，
    配合 trough_depletion_loss（压低赤道）共同建立"波谷-双峰"对比结构。

    L_crest = mean(crest_mask × lt_day_gate × relu(nmf2_floor - NmF2_pred)²)

    Args:
        NmF2_pred:   [Batch] F2 峰值密度预测 (log10)，保留梯度
        sin_I:       [Batch] 地磁倾角正弦
        lt_day_gate: [Batch] 日间门控权重（与 uplift/depletion 共享，夜间为 0）
        crest_sinI:  EIA 波峰中心 |sin_I|（默认 0.22，与 EIAMechanismHead 一致）
        crest_sigma: 波峰宽度（默认 0.10）
        nmf2_floor:  EIA 波峰 NmF2 下限 (log10, 默认 11.0)

    Returns:
        scalar loss
    """
    crest_mask = torch.exp(-((sin_I.abs() - crest_sinI) / crest_sigma) ** 2).detach()
    if lt_day_gate is not None:
        crest_mask = crest_mask * lt_day_gate.detach()
    return torch.mean(crest_mask * torch.relu(nmf2_floor - NmF2_pred) ** 2)


# ======================== 5. KGE 效率系数损失 ========================

def kge_loss(pred, target, eps=1e-8):
    """
    Kling-Gupta Efficiency（KGE）基损失

    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
        r = Pearson 相关系数（模式匹配）
        α = σ_pred/σ_obs（方差比，惩罚方差坍缩/膨胀）
        β = μ_pred/μ_obs（均值偏差比）

    loss = 1 - KGE，最小化 ≡ 最大化 KGE

    特点：
        α 项直接惩罚 NmF2 双峰抹平（σ_pred << σ_obs 时 (α-1)²大）
        r 项鼓励正确空间模式（赤道波谷 + 双峰）
        β 项消除系统性偏差

    注意：KGE 是批次级统计量，批次异质性（LT/纬度混合）会引入噪声；
          建议以小权重（~0.05）辅助 MSE，而非单独使用。

    Args:
        pred:   [B, 1] 或 [B] 预测值（log10 Ne 或 NmF2）
        target: [B, 1] 或 [B] 观测值，已 detach

    Returns:
        scalar loss = 1 - KGE
    """
    p = pred.view(-1)
    o = target.view(-1).detach()

    mu_p = p.mean()
    mu_o = o.mean()
    sigma_p = p.std() + eps
    sigma_o = o.std() + eps

    r = ((p - mu_p) * (o - mu_o)).mean() / (sigma_p * sigma_o)
    alpha = sigma_p / sigma_o
    beta  = mu_p / (mu_o.abs() + eps)

    kge = 1.0 - torch.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2 + eps)
    return 1.0 - kge


# ======================== 8. 廓线-峰高对齐损失 ========================

# 高度归一化系数（模型内部 alt_n = (alt - 310) / 190），用于量纲对齐
_ALT_RANGE_HALF = 190.0


def profile_peak_alignment_loss(Ne_at_peak, coords_peak, NmF2_pred=None, w_val=0.0):
    """
    廓线-峰高对齐：强迫 3D Ne_fused 在 h=hmF2_pred 处满足 ∂Ne/∂h = 0（极大值必要条件）。

    物理意义：
        PeakHead 已通过 GIRO 监督学习了正确峰高，此损失将该 2D 参数"锚入"3D 场：
        在 hmF2_pred 高度处，Ne_fused 关于高度的一阶导数应为零（密度极大值条件）。
        彻底打通 PeakHead 与 FusionDecoder 的壁垒，解决 FY 顶部数据下拉剖面问题。

    Loss = mean((∂Ne_fused/∂h|_{h=hmF2} × ALT_RANGE_HALF)²)
         + w_val × MSE(Ne_fused|_{h=hmF2}, NmF2_pred)

    梯度流：
        loss → ∂Ne/∂h → Ne_at_peak → FusionDecoder / SIREN / CrossAttn 权重
        w_val 部分仅对 Ne_at_peak（3D 场）求梯度，NmF2_pred 已 detach

    调用方须确保 coords_peak.requires_grad=True，且 coords_peak[:, 2] = hmF2_pred.detach()

    Args:
        Ne_at_peak:  [B, 1]  在 alt=hmF2_pred 处前向传播得到的 Ne_fused
        coords_peak: [B, 4]  requires_grad=True；col-2 = hmF2_pred.detach() (km)
        NmF2_pred:   [B]     峰值密度预测（log10），须已 detach；可选值对齐
        w_val:       float   值对齐损失权重（0 = 仅导数约束）

    Returns:
        scalar loss
    """
    if not coords_peak.requires_grad:
        return torch.tensor(0.0, device=Ne_at_peak.device)

    with torch.amp.autocast('cuda', enabled=False):
        Ne_fp32     = Ne_at_peak.float()
        coords_fp32 = coords_peak.float()
        try:
            grad = torch.autograd.grad(
                Ne_fp32.sum(), coords_fp32,
                create_graph=True, retain_graph=True
            )[0]          # [B, 4]
        except RuntimeError:
            return torch.tensor(0.0, device=Ne_at_peak.device)

    # ∂Ne/∂alt_km 归一化至 alt_n 量纲，使损失与其他物理损失量级相当
    dNe_dalt_norm = grad[:, 2] * _ALT_RANGE_HALF   # [B]
    loss = (dNe_dalt_norm ** 2).mean()

    if w_val > 0 and NmF2_pred is not None:
        # 值对齐：Ne_fused 在峰高处的值应等于 PeakHead 预测的 NmF2（均为 log10）
        loss = loss + w_val * F.mse_loss(Ne_at_peak.squeeze(-1), NmF2_pred.detach())

    return loss


# ======================== 组合物理损失 ========================

def combined_mdia_physics_loss(
    pred_ne, ne_bkg, ne_residual,
    coords, lat,
    hmF2_pred=None,
    NmF2_pred=None,
    sin_I=None,
    w_bkg=0.1,               # 保留兼容性（已由 w_bkg_low/w_bkg_high/w_bkg_night 替代）
    w_uplift=0.0,
    w_depletion=0.0,
    w_eia_crest=0.0,          # EIA 双峰增强（run17+）；配合 w_depletion 建立波谷-双峰结构
    uplift_threshold_km=380.0, # 赤道 hmF2 上涌阈值（run17+：380km，300km 时 loss 通常不激活）
    # 高度自适应 bkg 权重（P0-C，始终激活）
    use_adaptive_bkg=True,   # 保留参数兼容性，内部始终使用高度自适应
    w_bkg_low=0.25,          # 低高度（<transition）约束权重
    w_bkg_high=0.02,         # 高高度（>transition）约束权重
    w_bkg_transition=250.0,  # 过渡中心 km
    w_bkg_sharpness=25.0,    # 过渡宽度 km
    # 夜间背景增强（v3.1：替代 lambda_B）
    w_bkg_night=0.0,         # 夜间背景信任增益上限（LT=0 时叠加，LT=12 时为 0）
    # EIA 日间门控（v3.1：uplift/depletion 限定在白天）
    uplift_lt_sigma=5.0,     # 日间 LT 高斯门控半宽 (h)；LT=7/17h 时权重≈0.14
    # run26: DA 不确定性派生的 IRI 信任度（per-sample 加权 bkg_trust）
    trust_iri=None,          # [B] ∈ [0,1]；None 时退化为 run25 行为
):
    """
    FSIA-INR 组合物理损失（run65：3 项，移除 residual_smooth 和 horizontal_smooth）

    背景权重公式（run25：SZA + 高度双维度自适应，cos_SZA 替代 LT 含日期变化）：
        _ratio      = sigmoid((alt_km - w_bkg_transition) / w_bkg_sharpness)
        w_alt       = w_bkg_high × _ratio + w_bkg_low × (1 - _ratio)          高度因子
        night_gate  = sigmoid(-cos_SZA × 5)                                    夜间 1，正午 0
        night_amp   = w_bkg_night × night_gate × (2 - _ratio)                 低高度倍率≈2，高高度≈1
        w_bkg_eff   = w_alt + night_amp

    EIA uplift/depletion 日间门控（run25：cos_SZA 替代 LT 高斯）：
        lt_day_gate = relu(cos_SZA).clamp(0,1)                                 正午≈1，夜间精确为 0

    Args:
        pred_ne:             [Batch, 1] Ne_fused（最终预测）
        ne_bkg:              [Batch, 1] IRI 背景（已 detach）
        ne_residual:         [Batch, 1] Ne_delta（FusionDecoder 增量输出，保留以备兼容）
        coords:              [Batch, 4] 坐标 [lat, lon, alt_km, rel_hour]
        lat:                 [Batch]    地理纬度 (度)（仍用于 uplift/depletion 回退）
        hmF2_pred:           [Batch]    F2 峰高预测 (km)
        NmF2_pred:           [Batch]    F2 峰值密度预测 (log10)
        sin_I:               [Batch]    地磁倾角正弦（可选，uplift/depletion 磁赤道感知）
        w_bkg_night:         夜间背景信任增益（0=禁用，推荐 0.10-0.15）
        uplift_lt_sigma:     日间 LT 门控宽度 (h)

    Returns:
        total_loss: scalar
        loss_dict:  {'bkg', 'uplift', 'depletion', 'eia_crest', 'physics_total'}
    """
    loss_dict = {}

    # 1. 背景信任损失（高度 + 地方时自适应，v3.1 移除 lambda_B）
    alt_km_phys = coords[:, 2:3]                                          # [B,1]
    _ratio      = torch.sigmoid(
        (alt_km_phys - w_bkg_transition) / w_bkg_sharpness)              # [B,1]
    w_alt       = w_bkg_high * _ratio + w_bkg_low * (1.0 - _ratio)       # [B,1]

    # run25: 用太阳天顶角余弦替代 LT 门控（真实日夜判定，含日期变化）
    _cos_sza = _compute_cos_sza_from_coords(coords)                      # [B]

    # 夜间门控：cos_SZA<0(夜)→1.0, cos_SZA>0(日)→0.0（sigmoid 光滑过渡，β=5 控制陡峭度）
    night_gate = torch.sigmoid(-_cos_sza * 5.0).unsqueeze(-1)            # [B,1]

    # 夜间增益叠加：低高度倍率(2-_ratio)≈2, 高高度≈1
    night_amp = w_bkg_night * night_gate * (2.0 - _ratio)                # [B,1]

    w_bkg_eff = w_alt + night_amp                                         # [B,1]
    raw_bkg   = (pred_ne - ne_bkg.detach()) ** 2                          # [B,1]

    # run26 (A1): DA 不确定性加权——FY 主导样本(K_FY 大→trust_iri 小)
    # 弱化 IRI 锚定，允许偏离错误的 IRI（修复 Jicamarca 夜间 IRI 高估同质化）
    if trust_iri is not None:
        w_bkg_eff = w_bkg_eff * trust_iri.detach().view(-1, 1)            # [B,1]
    loss_bkg  = (w_bkg_eff * raw_bkg).mean()
    loss_dict['bkg'] = loss_bkg.item()

    # EIA 日间门控（供 uplift/depletion 使用；run25：cos_SZA-based）
    # cos_SZA>0 → 日间，等于 cos_SZA 强度；cos_SZA<0 → 夜间精确为 0
    _lt_day_gate = torch.relu(_cos_sza).clamp(0.0, 1.0).detach()          # [B]

    # 2. 赤道 ExB 上涌先验（日间门控：夜间不激活）
    if w_uplift > 0 and hmF2_pred is not None:
        try:
            loss_uplift = trough_uplift_loss(
                hmF2_pred, lat.detach(), threshold_km=uplift_threshold_km,
                sin_I=sin_I, lt_day_gate=_lt_day_gate)
            loss_dict['uplift'] = loss_uplift.item()
        except RuntimeError:
            loss_uplift = torch.tensor(0.0, device=pred_ne.device)
            loss_dict['uplift'] = 0.0
    else:
        loss_uplift = torch.tensor(0.0, device=pred_ne.device)
        loss_dict['uplift'] = 0.0

    # 3. 赤道喷泉耗散约束（日间门控：夜间不激活）
    if w_depletion > 0 and NmF2_pred is not None:
        try:
            loss_depletion = trough_depletion_loss(
                NmF2_pred, lat.detach(), sin_I=sin_I,
                lt_day_gate=_lt_day_gate)
            loss_dict['depletion'] = loss_depletion.item()
        except RuntimeError:
            loss_depletion = torch.tensor(0.0, device=pred_ne.device)
            loss_dict['depletion'] = 0.0
    else:
        loss_depletion = torch.tensor(0.0, device=pred_ne.device)
        loss_dict['depletion'] = 0.0

    # 4. EIA 双峰增强（日间门控，配合 depletion 建立赤道波谷 + ±15° 双峰结构）
    if w_eia_crest > 0 and NmF2_pred is not None and sin_I is not None:
        try:
            loss_eia_crest = eia_crest_enhance_loss(
                NmF2_pred, sin_I, lt_day_gate=_lt_day_gate)
            loss_dict['eia_crest'] = loss_eia_crest.item()
        except RuntimeError:
            loss_eia_crest = torch.tensor(0.0, device=pred_ne.device)
            loss_dict['eia_crest'] = 0.0
    else:
        loss_eia_crest = torch.tensor(0.0, device=pred_ne.device)
        loss_dict['eia_crest'] = 0.0

    # 加权总和（bkg 已在内部自适应加权，不再外乘 w_bkg）
    total_loss = (loss_bkg +
                  w_uplift * loss_uplift +
                  w_depletion * loss_depletion +
                  w_eia_crest * loss_eia_crest)

    loss_dict['physics_total'] = total_loss.item()

    return total_loss, loss_dict


# ======================== 7. 峰场空间平滑损失 ========================

def peak_field_smooth_loss(hmF2_fused, coords,
                            sin_I=None, cos_I=None, lat_geo_deg=None):
    """
    峰场空间平滑约束（run28-C 各向异性版）

    L_smooth = mean(w_lat × |∂hmF2/∂lat|²  +  |∂hmF2/∂lon|²)
        其中 w_lat = sin(MODIP)²  ∈ [0, 1]

    物理语义：
        EIA 喷泉效应在磁赤道附近（|sin_MODIP| 小）造成 hmF2 ~4-7 km/° 的合理纬向梯度
        （赤道高 ~410km，驼峰低 ~350km）。各向同性平滑会把这个真实结构当噪声压平。
        改用 sin_MODIP² 加权 lat 项：
            磁赤道 sin_MODIP=0 → w_lat=0 → 完全允许 EIA 纬向梯度
            磁极   sin_MODIP=1 → w_lat=1 → 保持高纬空间平滑
        lon 项始终保留（lon 方向是 LT 变化，结构主要由 SW/SZA 驱动，需平滑）。

    Fallback：若 sin_I/cos_I/lat_geo_deg 任一为 None，退化为各向同性（旧版行为）。

    梯度路径（PeakHead 严格 2D 后成立）：
        coords[:,0/1] → spatial_input → DualFreqSpatialNet → h_spatial
        → PeakHead(h_spatial, h_sw, iri_peak) → hmF2_fused

    调用约定：
        - 仅在 compute_physics 批次内调用（coords.requires_grad=True 已设置）
        - 调用方传入随机子采样后的 hmF2_fused[idx] 和 coords[idx]
        - sin_I/cos_I/lat_geo_deg 同样 [idx] 子采样后传入；可全 None 退化

    Args:
        hmF2_fused:    [N]     融合峰高 (km)，含梯度（PeakHead 输出的子采样）
        coords:        [N, 4+] requires_grad=True（同批次 coords 的子采样）
        sin_I, cos_I:  [N] 地磁倾角 sin/cos（_compute_dip_features 输出）
        lat_geo_deg:   [N] 地理纬度 (°)

    Returns:
        scalar loss
    """
    if not coords.requires_grad:
        return torch.tensor(0.0, device=hmF2_fused.device)

    with torch.amp.autocast('cuda', enabled=False):
        hf32 = hmF2_fused.float()
        cf32 = coords.float()
        try:
            grad = torch.autograd.grad(
                hf32.sum(), cf32,
                create_graph=True, retain_graph=True,
                only_inputs=True,
            )[0]   # [N, 4+]
        except RuntimeError:
            return torch.tensor(0.0, device=hmF2_fused.device)

    grad_lat_sq = grad[:, 0] ** 2
    grad_lon_sq = grad[:, 1] ** 2

    # run28-C: sin(MODIP)² 加权 lat 项 — EIA 区域允许大梯度
    if sin_I is not None and cos_I is not None and lat_geo_deg is not None:
        with torch.amp.autocast('cuda', enabled=False):
            sI = sin_I.float()
            cI = cos_I.float()
            cos_lat = torch.cos(lat_geo_deg.float() * (_math.pi / 180.0))
            tan_I = sI / (cI + 1e-7)
            sm = tan_I / torch.sqrt(tan_I ** 2 + cos_lat + 1e-7)   # sin(MODIP) ∈ (-1,1)
            w_lat = (sm ** 2).detach()                              # 防止 mask 反传影响 PeakHead
        return (w_lat * grad_lat_sq + grad_lon_sq).mean()

    # 退化：各向同性（旧版）
    return (grad_lat_sq + grad_lon_sq).mean()


# ======================== run22: Pearson-r 形态约束损失 ========================

def pearson_r_shape_loss(Ne_fused, ne_bkg, eps: float = 1e-8, trust_iri=None):
    """
    批次级 Pearson-r 形态约束（run22 新增；run26 增加 trust_iri 加权）

    L_shape = trust_iri.mean() × (1 - pearson_r(Ne_fused, ne_bkg.detach()))

    物理语义：
        IRI 主结构保持（形态守恒）。ne_bkg.detach() 作为参考形状，
        Loss→0 当 Ne_fused 与 IRI 背景场高度相关（形态一致）。
        FY 主要修正幅度（振幅 / 偏置），不改变垂直剖面形态。

    run26 改进（A1：DA 不确定性加权）：
        当 K_FY 大（FY 主导）→ trust_iri 小 → L_shape 弱化 → 允许偏离 IRI 形态
        当 K_FY 小（IRI 主导）→ trust_iri 大 → L_shape 强 → 严格守 IRI 形态
        修复 Jicamarca 夜间 IRI 高估时形态守恒导致的 bias 加重问题

    调用约定：
        每批次调用（与 L_FY 同频）；w_shape≈0.05。
        Ne_fused 含梯度；ne_bkg.detach() 已切断梯度（不计入 loss 计算图）。
        trust_iri 由调用方计算，已 detach。

    Args:
        Ne_fused:  [B, 1] 融合预测（含梯度）
        ne_bkg:    [B, 1] IRI 背景（.detach() 内部调用，调用方可直接传原张量）
        eps:       数值稳定项
        trust_iri: [B] DA 不确定性派生的 IRI 信任度 ∈ [0,1]，已 detach；
                   None 时退化为 run22 行为（无加权）

    Returns:
        scalar ∈ [0, 2]（trust_iri=None）；trust_iri 提供时再乘 trust_iri.mean()
    """
    x = Ne_fused.squeeze() - Ne_fused.squeeze().mean()
    y = ne_bkg.detach().squeeze() - ne_bkg.detach().squeeze().mean()
    r = (x * y).sum() / (x.norm() * y.norm() + eps)
    raw_loss = 1.0 - r
    if trust_iri is not None:
        return trust_iri.detach().mean() * raw_loss
    return raw_loss


# ======================== run41: Ne 垂直方向平滑损失 ========================

def ne_vert_smooth_loss(Ne_fused, coords):
    """
    Ne 垂直方向二阶导平滑（run41 — 解决 EDP 廓线不够平滑问题）

    L_vert = mean((∂²Ne_fused / ∂alt²)²)

    物理意义：
        电离层 EDP 廓线在 alt 方向上应物理平滑（Chapman 廓线为光滑指数衰减）。
        FusionDecoder 输入仅有 alt_n 和 delta_alt_n 两维 alt 信号，对 alt 响应粒度粗，
        可能产生高频 alt 波动。本损失直接约束 Ne_fused 二阶导，互补 residual_smooth_loss
        （后者作用于 Ne_delta，前者作用于 Ne_fused = ne_bkg + Ne_delta）。

    Args:
        Ne_fused: [B, 1]  最终融合 Ne（含梯度）
        coords:   [B, 4+] requires_grad=True，col 2 = alt (km)

    Returns:
        scalar loss
    """
    if not coords.requires_grad:
        return torch.tensor(0.0, device=Ne_fused.device)
    with torch.amp.autocast('cuda', enabled=False):
        Nf32 = Ne_fused.float()
        Cf32 = coords.float()
        try:
            grad_1 = torch.autograd.grad(
                Nf32.sum(), Cf32,
                create_graph=True, retain_graph=True,
                only_inputs=True,
            )[0][:, 2]                                          # [B] ∂N/∂alt
            grad_2 = torch.autograd.grad(
                grad_1.sum(), Cf32,
                create_graph=True, retain_graph=True,
                only_inputs=True,
            )[0][:, 2]                                          # [B] ∂²N/∂alt²
        except RuntimeError:
            return torch.tensor(0.0, device=Ne_fused.device)
    return (grad_2 ** 2).mean()


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print('=' * 60)
    print('FSIA-INR 物理损失测试')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 128

    coords = torch.randn(B, 4, requires_grad=True, device=device)
    coords.data[:, 0] = coords.data[:, 0] * 90   # Lat
    coords.data[:, 1] = coords.data[:, 1] * 180  # Lon
    coords.data[:, 2] = 200 + torch.abs(coords.data[:, 2]) * 100  # Alt
    coords.data[:, 3] = torch.abs(coords.data[:, 3]) * 100        # Time

    pred_ne     = torch.randn(B, 1, requires_grad=True, device=device) + 11.5
    ne_bkg      = torch.randn(B, 1, device=device) + 11.3
    ne_residual = torch.randn(B, 1, requires_grad=True, device=device) * 0.1
    lat         = coords[:, 0].detach()
    hmF2        = torch.full((B,), 320.0, device=device)
    NmF2        = torch.full((B,), 11.5,  device=device)

    print('\n[测试] 背景信任损失')
    l = background_trust_loss(pred_ne, ne_bkg, lat)
    print(f'  L_bkg = {l.item():.6f}')

    print('[测试] 组合损失')
    total, loss_dict = combined_mdia_physics_loss(
        pred_ne, ne_bkg, ne_residual, coords, lat,
        hmF2_pred=hmF2, NmF2_pred=NmF2,
        w_bkg=0.1, w_uplift=0.01, w_depletion=0.005,
    )
    print(f'  Total physics loss: {total.item():.6f}')
    for k, v in loss_dict.items():
        print(f'    {k}: {v:.6f}')

    total.backward()
    print(f'\n  pred_ne.grad norm: {pred_ne.grad.norm().item():.6f}')
    print('\n所有测试通过!')
