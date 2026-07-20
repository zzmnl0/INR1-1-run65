# CLAUDE.md

FSIA-INR — feature-space neural data assimilation for 3D ionospheric Ne reconstruction.

## Commands

All commands run from `FSIA_INR18/`:

```bash
python main_fsia.py                                  # train
python plot_fsia.py                                  # standalone viz
python isr_evaluation/main_isr_eval.py               # ISR validation (Jicamarca + Poker Flat)
python -m inr_modules.mdia.fsia_model                # model self-test

# One-time GIRO preprocessing (before first run)
python inr_modules/mdia/preprocess_giro.py \
    --giro_dir D:/c_shuju/GIRO_hmf2 \
    --output_dir D:/c_shuju/GIRO_hmf2/processed
```

**Run number**: edit `save_dir` on `main_fsia.py:46`. Current = **run28** (NeuralETKFLayer).

---

## Architecture (run28)

**核心范式**：神经化集合卡尔曼滤波（Neural ETKF）+ 3 级融合

```
公式: Ne_fused = Ne_bkg + tanh(FusionDecoder(h_decode, alt_n, delta_alt_n)) × gate

Level 1 — 2D 融合峰场
    IRIPeakManager → [hmF2_IRI, NmF2_IRI]
    PeakHead(h_spatial, h_sw, iri_peak, sh_feats[14]) → hmF2_fused, NmF2_fused
    GIRO 监督 + peak_field_smooth_loss + GIRO→Ne 直接约束

Level 2 — 结构坐标桥
    delta_alt_n  = (alt − hmF2_fused.detach()) / 190
    f_spatial_3d = h_spatial + proj_delta(delta_alt_n)
    frame_offset = (hmF2_det − hmF2_IRI) / 190

Level 3 — Neural ETKF DA
    f_iri    = iri_align_net(cat(h_iri, delta_alt_iri, NmF2_IRI_n))
               + proj_frame_offset(frame_offset)
    h_obs_FY = h_obs_context (Phase2) | h_spatial (Phase1)
    NeuralETKFLayer(f_iri, h_obs_FY, h_res, h_sw, …, alt_km, hmF2_km)
        → h_analysis [B,64]
    h_decode = α·h_analysis + (1−α)·h_pre   (CRF, run26 Step 1)
    log_var  = uncertainty_head(h_analysis, h_sw)
    gate     = MultiScaleAdaptiveGate(h_decode, h_sw, log_var.detach(), …, regime_desc)
    Ne_fused = Ne_bkg + tanh(FusionDecoder(h_decode, …)) × gate
    L_iri_struct = MSE(iri_recon_head(h_iri_aligned), ne_bkg.detach())
    L_shape      = 1 − Pearson-r(Ne_fused, ne_bkg)
```

**模型参数量**: ~447K 总（~396K 可训练 + ~50K IRI 冻结）

### 四路特征（输入 NeuralETKFLayer）

| Token | 来源 | 维度 |
|---|---|---|
| `f_iri` (x_b) | IRI proxy 末层 [128] + IRI 峰参数 [2]，经 iri_align_net + proj_frame_offset | 64 |
| `h_obs_FY` | Phase1: h_spatial；Phase2: ObsCrossAttn → h_obs_context | 64 |
| `h_res` (y_vert) | Residual SIREN(15D, 含 AltEmbed 6D) | 64 |
| `h_sw` | DualScaleSWEncoder(EWMA + win_fusion) — 仅时域 (run28-B 删除频域分支) | 64 |

---

## NeuralETKFLayer (run28 DA 核心)

ETKF 形式，N=8 个并行 PerturbationNet 生成 ensemble，N 维子空间求逆做最优更新：

```python
δ^(n)   = PerturbationNet_n(b_in)         n=1..N      [B, d]
X       = δ - δ_mean                                  [B, N, d]
X_inf   = sqrt(inflation) × X                         (可学习 inflation)
HX_*    = einsum('bnd,do->bno', X_inf, H_*)          [B, N, d]
HXR_*   = HX_* / R_*                                  pre-whitened
M_*     = HXR_* @ HX_*^T / (N-1)                     [B, N, N]
T_*     = inv(I_N + M_*)                              N=8 求逆便宜
innov_* = h_obs_* − H_* @ f_iri                      [B, d]
w_*     = T_* @ HXR_* @ innov_*                      [B, N]
x_a     = LN(f_iri + δ_mean + X_inf^T @ (w_FY + w_vert))
```

**R_FY 物理先验**（FY Abel 反演 + LT 调制；alt_floor=120km 处不确定性最大）：

```
below_dist = relu(hmF2_km − alt_km) / (hmF2_km − 120).clamp_min(30)   ∈ [0, 1]
night_gate = 0.5 × (1 − cos_SZA).clamp(0, 1)
r_fy_prior = below_dist × (α_day + α_night × night_gate)
r_fy       = softplus(R_FY_net(in) + r_fy_prior)
```

初始 α_day=0.5, α_night=1.5（可学习）：
- alt=hmF2 处       → 1.00× 基线 (softplus(0) ≈ 0.69)
- alt=120km 白天    → 1.41× 基线
- alt=120km 夜间    → 3.07× 基线

**返回 7-tuple**（与 MHDK 接口同构，物理意义升级）：

| 位 | 量 | 含义 |
|---|---|---|
| 0 | `h_analysis` | 后验状态（卡尔曼分析场）|
| 1 | `K_eff_FY`   | ens_var/(ens_var+r_fy)，对角等效 K |
| 2 | `K_eff_vert` | 同上垂直 |
| 3 | `ens_var`    | (X²).sum(N)/(N-1)，背景误差协方差对角元 |
| 4 | `r_fy`       | 含物理先验 |
| 5 | `innov_FY`   | h_obs_FY − H_FY @ f_iri |
| 6 | `innov_vert` | h_res − H_vert @ f_iri |

**关键 Parameter 结构**：

```python
P_w1 / P_w2  : Parameter[N, in, out]    PerturbationNet 向量化
H_FY_w / H_vert_w : Parameter[d, d]     单个共享，零初始化
log_R_vert        : Parameter[d]
α_day / α_night   : scalar Parameter    R_FY 物理先验
log_inflation     : scalar Parameter
alt_floor_km      : buffer = 120        同化范围下界
```

**初始训练动力学**：H 零初始化 → HX=0 → update=0 → h_analysis ≈ LN(f_iri+δ_mean) → Ne_fused ≈ ne_bkg（IRI baseline 起步）；H 通过 ∂update/∂H 反传从零学起。

---

## 关键组件

### PeakHead（IRI-GIRO 融合峰场）

```
输入: cat(LN(h_spatial), LN(h_sw), hmF2_IRI_n, NmF2_IRI_n, sh_feats[14]) = 144D
网络: hmf2_bias_net + nmf2_bias_net 完全独立（各 Linear(144→64) → SiLU → Linear(64→5)）输出层零初始化
公式 (run32 IRI 形态模板 + SH 低秩 bias 解耦):

    sh_hmf2 = [P1, P3, P1·cos_SZA, P1·sin_doy, 1.0]              单峰基（hmF2 馒头）
    sh_nmf2 = [P2, P2·cos_SZA, P2·sin_doy, P2·cos2_SZA, 1.0]     双峰基（NmF2 驼峰）
    coef_h, coef_n: [B, 5]   两个独立 net 各自输出
    raw_h  = (sh_hmf2 × coef_h).sum(-1)                          SH 投影
    raw_n  = (sh_nmf2 × coef_n).sum(-1)
    bias_h = 60  × tanh(raw_h / 2.0)                             ∈ ±60 km   (软限幅)
    bias_n = 0.4 × tanh(raw_n / 2.0)                             ∈ ±0.4 log10
    hmF2_fused = clamp(hmF2_IRI + bias_h, 200, 550)              IRI 形态完全保留
    NmF2_fused = clamp(NmF2_IRI + bias_n, 9,   13)
范围: hmF2 ∈ [200, 550] km；NmF2 ∈ [9, 13] log10
起步: coef=0 → bias=0 → fused = IRI（精确退化）
```

**run32 设计理由（数据特性匹配）**：

| 数据源 | bias | Pearson-r | 角色 |
|---|---|---|---|
| IRI | 大 | **高**（形态合理）+ 全面覆盖 | **形态模板** |
| FY/GIRO | **小** | 低（站点稀疏） | **数值校准** |

- **形态层**：IRI 直接作为 hmF2/NmF2 形态模板（无修改），物理 Pearson-r 完全继承
- **数值层**：bias 通过 5 维 SH 系数空间低秩展开，强制平滑 → 自动避免逐点过拟合 GIRO 站
- **物理解耦**：hmF2 用 P1/P3 单峰基（馒头形态），NmF2 用 P2 双峰基（驼峰形态），两个独立 net 互不传染
- **双重保形态**：架构层 + `peak_shape_loss` (w=0.1/0.05，已弱化)

主路径中 `h_spatial.detach()` α 软隔离（α 由 EnKF proxy K_FY 自适应生成）。`giro_mode` 仍需 `iri_peak`。

### MODIPSHBasis（零参数 14D 球谐基）

仿 IRI SHU-2015，sin(MODIP) 基础 + cos_SZA/sin_doy 谐波；输出 14D 注入 PeakHead。

### MultiScaleAdaptiveGate

```
gate = gate_data × gate_phys     ∈ (0, 1)，无 clamp

gate_data: cat(h_decode[64], h_sw[64], log_var.det, |Ne_delta|.det, regime_desc[4]) = 134D
           → Linear(134→32) → SiLU → Linear(32→1) → sigmoid
           bias=logit(0.95)≈2.944
gate_phys: 三高斯混合 g(δ) where δ = (alt − hmF2_det) / 100
           centers [0, -1.5, 2.0]；残差超网络 gate_phys_net(h_spatial)→[B,6]，零初始化
regime_desc = [alt_n, |sin_I|, kp_eff, f107_eff]
```

### CRF (Channel-wise Residual Fusion, run26 Step 1)

```
h_pre    = proj_pre(cat(f_iri, h_obs_FY, h_res))    proj_pre 零初始化 → h_pre=0
α        = sigmoid(crf_alpha)     [64]，可学习；初始 ≈ 0.9933
h_decode = α ⊙ h_analysis + (1-α) ⊙ h_pre
```

### DualFreqSpatialNet

```
输入: [lat_geo_n, sin_Lon, cos_Lon, cos_SZA, sin_doy, cos_doy, sin_I]   7D
两路并行 SIREN: ω₀=10 全局 + ω₀=30 精细
gate_net 末层零初始化 → 初始等权融合 0.5
```

### DualScaleSWEncoder (run28-B：仅时域)

```
时域 (DualScaleSWEncoder):
    learnable EWMA: τ = exp(log_tau) + offset    τ_kp+3h, τ_solar+48h
    win_fusion: 3 窗口均值 (last-8, last-24, full)
    → h_sw [64]
```

**SpectralSWBranch 已删除（run28-B）**：18h SW 时窗 rFFT 最低非零频率 ≈ 1.33 cycle/day，无法分离 24h 日变化（不在频谱内）；实际拾取 6h/3h 高次谐波噪声，与时域 EWMA 互补价值低；run26 Step4 引入后指标轻微变差，run27/28 简化版恶化更明显，故彻底回退。

### AltitudeMultiScaleEmbedding + Residual SIREN

```
freqs = (1, 2, 4) × π → 6D（零参数）
Residual SIREN 输入: 9D + 6D AltEmbed = 15D
    [lat_n, sin_lon, cos_lon, alt_n, cos_SZA, sin_doy, cos_doy, sin_I, cos_I, AltEmbed×6]
```

### IRIPeakManager（数据管理器）

```
文件: inr_modules/data_managers/iri_peak_manager.py
数据: IRI_hmF2_*.npy + IRI_NmF2_*.npy  shape (241, 181, 181)
分辨率: 3h × 1° × 2°；NmF2 单位 m⁻³ 自动转 log10
插值: NaN-mask 三线性 (F.grid_sample align_corners=True)
fallback: [300km, 11.5] (邻居全 NaN 时)
接口: get_iri_peak(coords [B,4+]) → [B, 2]    @torch.no_grad()
归一化 (align_corners=True):
    norm_t = rel_hour / 360 - 1     norm_h = lat/90      norm_w = lon/180
```

### `_compute_solar_features(lat, lon, rel_hour)`

Spencer 1971 7阶傅里叶级数太阳赤纬 → cos(SZA) + sin/cos(2π·doy/365)；纯 PyTorch 可微。

---

## 数据流（每批次）

```
[FY 观测 B×5] → coords[B,4], target_ne[B,1], sw_seq[B,36,2]
    ├─► IRIPeakManager → iri_peak [B,2]                    (no_grad)
    ├─► sw_encoder (DualScaleSWEncoder) → h_sw [B,64]
    ├─► DualFreqSpatialNet + MagFiLM → h_spatial [B,64]
    ├─► MODIPSHBasis → sh_feats [B,14]                     (无参数)
    │
    │  [giro_mode → 仅运行 PeakHead，跳过下面]
    │
    ├─► IRI proxy → ne_bkg [B,1], h_iri [B,128]            (no_grad)
    ├─► proxy K_FY (kalman_layer.compute_proxy_trust_iri)  → α_eff [B,1]
    ├─► PeakHead(α-detached h_spatial, h_sw, iri_peak, sh_feats)
    │       → hmF2_fused, NmF2_fused
    │       delta_alt_n   = (alt - hmF2_det) / 190
    │       frame_offset  = (hmF2_det - hmF2_IRI) / 190
    │
    ├─► iri_align_net(cat(h_iri, delta_alt_iri, NmF2_IRI_n)) + proj_frame_offset
    │       → f_iri [B,64]
    ├─► ResidualSIREN(15D) + film_mag_res → h_res [B,64]
    │
    ├─► NeuralETKFLayer(f_iri, h_obs_FY, h_res, h_sw,
    │                   …, alt_km, hmF2_det)
    │       → h_analysis, K_eff_FY, K_eff_vert, ens_var, r_fy, innov_FY, innov_vert
    │
    ├─► CRF: h_decode = sigmoid(crf_alpha) × h_analysis + (1-…) × h_pre
    ├─► FusionDecoder(cat(h_decode, alt_n, delta_alt_n)) → Ne_delta_raw
    ├─► uncertainty_head(h_analysis, h_sw) → log_var       (先于 gate)
    ├─► MultiScaleAdaptiveGate(…, log_var.det, regime_desc)
    │       → gate, gate_data, gate_phys
    ├─► Ne_delta = tanh(Ne_delta_raw) × gate
    └─► Ne_fused = ne_bkg + Ne_delta

    每 batch:  L_shape (Pearson-r)
    每 giro_loss_freq=2:    GIRO peak (giro_mode) + GIRO→Ne 直接约束 (3B 全前向)
    每 physics_loss_freq=10: 5 项 + L_iri_struct + peak_smooth + profile_align
```

---

## model.forward() 接口

```python
model.forward(
    coords,                  # [B, 4] or [B, 5]: [lat_geo, lon_geo, alt_km, rel_hour, (lat_aacgm)]
    sw_seq,                  # [B, seq_len, 2]: [Kp_norm, F10.7_norm]
    precomputed_h_sw=None,   # [B, 64] 可选优化
    giro_mode=False,         # True → 跳过 IRI proxy + Residual SIREN + KalmanLayer
    iri_peak=None,           # [B, 2] [hmF2_IRI_km, NmF2_IRI_log10]，None → fallback (300, 11.5)
    voxel_feats=None, dist_weights=None, valid_mask=None,   # Plan9 Phase2
)
```

**返回**: `(Ne_fused [B,1], log_var [B,1], zeros_placeholder, Ne_delta [B,1], extras)`

### extras 主要键

| 键 | shape | 含义 |
|---|---|---|
| `ne_bkg` | [B,1] | IRI 背景 Ne |
| `ne_residual` | [B,1] | = Ne_delta |
| `h_spatial`, `h_sw`, `h_iri`, `h_iri_aligned` | [B,*] | 各层特征 |
| `h_analysis` / `h_fused` | [B,64] | 卡尔曼分析场（同义别名）|
| `h_decode`, `h_pre`, `crf_alpha` | * | CRF 监控 |
| `gate`, `gate_data`, `gate_phys` | [B,1] | 自适应门控（giro_mode 为 None）|
| `regime_desc` | [B,4] | [alt_n, \|sin_I\|, kp_eff, f107_eff] |
| `peak_params` / `chapman_params` | dict | `{'hmF2', 'NmF2'}` 各 [B] |
| `hmF2_det` | [B] | hmF2_fused.detach() |
| `delta_alt_iri`, `frame_offset` | [B] | 监控 |
| `alpha_eff` | [B] | sample-adaptive PeakHead α |
| `K_FY`, `K_vert`, `b`, `r_fy`, `innov_FY`, `innov_vert` | [B,64] | EnKF 监控（giro_mode 为 zeros）|
| `member_weights` / `head_weights` | [B,N] | ensemble 成员相对贡献（别名等价）|
| `inflation_scale`, `alpha_r_day`, `alpha_r_night` | scalar | EnKF 调参监控 |
| `h_sw_time` | [B,64] | = h_sw（保留键以兼容下游） |
| `sin_I`, `cos_I` | [B] | IGRF 偶极子近似地磁倾角 sin/cos（peak_field_smooth_loss 各向异性 mask 用） |
| `hmF2_IRI_km`, `NmF2_IRI_log10` | [B] | run31 IRI 峰参数（peak_shape_loss 形态参考） |

### IRI proxy 调用规则

- `forward(x, return_features=False)` → **单张量** `ne_log10 [B,1]`（不是元组，不可解包）
- `forward(x, return_features=True)`  → `(ne_log10 [B,1], h [B,128])`
- 全部包在 `torch.no_grad()` 内

---

## 物理损失

### `combined_mdia_physics_loss`（5 项，每 10 batch）

| 损失 | 公式 | 权重 |
|---|---|---|
| `bkg` | LT+高度自适应 `w_eff = w_alt + w_night × night_gate × (2−ratio)` × ‖Ne−Ne_bkg‖² | `w_bkg_low=0.25, w_bkg_high=0.02, w_bkg_night=0.15` |
| `residual_smooth` | ‖∂²Ne_delta/∂h²‖² | `w_residual_smooth=0.05` |
| `horizontal_smooth` | anisotropic ‖∇⊥Ne_delta‖² | `w_horizontal_smooth=0.01` |
| `uplift` | mean(sin_I × lt_day_gate × relu(300−hmF2)²) | `w_uplift=0.01` |
| `depletion` | mean(sin_I × lt_day_gate × NmF2_pred) | `w_depletion=0.005` |

`uplift_lt_sigma=5.0` → 日间门控 `exp(−((LT−12)/5)²)`，正午=1，LT=7/17h≈0.14，夜间≈0。

### 单独调用的损失

| 损失 | 公式 | 调用频率 / 权重 |
|---|---|---|
| `pearson_r_shape_loss` | 1 − Pearson-r(Ne_fused, ne_bkg.detach()) | 每 batch / `w_shape=0.05` |
| `peak_shape_loss` (hmF2) | 1 − Pearson-r(hmF2_fused, hmF2_IRI.detach()) — run32 形态守恒（架构已强保，权重弱化） | 每 batch / `w_hmf2_shape=0.1` |
| `peak_shape_loss` (NmF2) | 1 − Pearson-r(NmF2_fused, NmF2_IRI.detach()) — run32 形态守恒 | 每 batch / `w_nmf2_shape=0.05` |
| `profile_peak_alignment_loss` | mean((∂Ne/∂h × 190)²) @ hmF2_det | 随 Physics / `w_profile_align=0.1, w_val=0.3` |
| `peak_field_smooth_loss` | mean(sin_MODIP² × \|∂hmF2/∂lat\|² + \|∂hmF2/∂lon\|²) — run28-C 各向异性，磁赤道 weight=0 允许 EIA 大梯度 | 随 Physics（256 子采样）/ `w_peak_smooth=0.01` |
| `L_iri_struct` | MSE(iri_recon_head(h_iri_aligned), ne_bkg.detach()) | 随 Physics / `w_iri_struct=0.05` |
| `L_giro_ne_val` | MSE(Ne_fused@hmF2_obs, NmF2_fused.detach()) | 随 GIRO / `w_giro_ne_val=0.5` |
| `L_giro_argmax` | mean(relu(Ne_up−Ne_pk) + relu(Ne_dn−Ne_pk)) @ Δh=10km | 随 GIRO / `w_giro_argmax=0.2` |
| Gate entropy | gate·log(gate) + (1−gate)·log(1−gate) | 每 batch / `w_gate_entropy=0.01` |

---

## 训练循环

- **Warmup**：epoch 0–1 用 Huber loss；epoch ≥ 2 切换 NLL（`uncertainty_warmup_epochs=2`, `log_var_regularization=0.01`）
- **双域早停**：Ne MSE 验证 + GIRO peak MAE 各自独立 patience=4，任一触发 stop；仅保存 `best_fsia_model.pth`（Ne MSE 改善条件）
- **GIRO val peak MAE**: `val_giro_peak_mae = hmf2_mae/50 + nmf2_mae`
- **优化器**：单组 AdamW + CosineAnnealingLR

`train_fsia()` 返回 `(model, train_losses, val_losses, train_loader, val_loader, sw_manager, batch_processor)`。

### stats / history 主要键

```
基本: total, mse, nll, l_shape, gate_mean, member_w_max
Physics: bkg, residual_smooth, horizontal_smooth, uplift, depletion, physics_total,
         iri_struct, profile_align, peak_smooth
GIRO: giro_hmf2, giro_nmf2, giro_ne_val, giro_argmax
EnKF: K_FY_mean, K_vert_mean, b_mean (=ens_var), r_fy_mean,
      innov_FY_norm, innov_vert_norm,
      inflation, alpha_r_day, alpha_r_night
SW:   tau_kp, tau_solar
诊断: ne_delta_abs, ne_bkg_mean, ne_fused_mean, delta_pos_frac,
      hmf2_iri_diff, alt_weight_mean, low_alt_frac, obs_ctx_frac
```

---

## 关键文件

| 文件 | 作用 |
|---|---|
| `inr_modules/config_mdia.py` | 所有超参数（FSIA 独立副本）|
| `inr_modules/data_managers/iri_peak_manager.py` | IRI 预计算峰参数 |
| `inr_modules/data_managers/irinc_neural_proxy.py` | IRI 神经代理（完全冻结）|
| `inr_modules/data_managers/space_weather_manager.py` | Kp/F10.7 时间序列 |
| `inr_modules/data_managers/FY_dataloader.py` | FY-3 卫星 EDP + TimeBinSampler |
| `inr_modules/mdia/fsia_model.py` | 主模型（FSIA_INR_Model + NeuralETKFLayer + 所有子模块）|
| `inr_modules/mdia/mdia_model.py` | 工具函数依赖（`_compute_dip_features` 等）|
| `inr_modules/mdia/train_fsia.py` | 训练循环 + 验证 + 双域早停 |
| `inr_modules/mdia/physics_losses_mdia.py` | 5+4 项物理损失 |
| `inr_modules/mdia/giro_dataloader.py` | GIRO 测高仪监督 |
| `inr_modules/mdia/ewma_sw_encoder.py` | DualScaleSWEncoder |
| `inr_modules/mdia/sliding_dataset.py` | SlidingWindowBatchProcessor |
| `inr_modules/mdia/visualization_mdia.py` | 全球切片 + EDP + hmF2 图 |
| `inr_modules/mdia/evaluation_mdia.py` | 2-panel 散点图 |

---

## 配置（`inr_modules/config_mdia.py`）

`update_config_mdia(**kwargs)` 运行时覆盖。

### 必填路径

```python
'fy_path':         r'D:\FYsatellite\EDP_data\fy_202409_clean.npy',
'iri_proxy_path':  r'D:\code11\IRI01\output_results\iri_september_full_proxy.pth',
'sw_path':         r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
'giro_hmf2_path':  r'D:\c_shuju\GIRO_hmf2\processed\giro_hmf2.npy',
'giro_nmf2_path':  r'D:\c_shuju\GIRO_hmf2\processed\giro_nmf2.npy',
'iri_hmf2_path':   r'D:\IRI\data01\edp\IRI_hmF2_20240901_20241001.npy',
'iri_nmf2_path':   r'D:\IRI\data01\edp\IRI_NmF2_20240901_20241001.npy',
```

### FSIA 专有键

```python
# 双频空间 SIREN
'omega_low': 10.0, 'omega_high': 30.0
# 多尺度高度嵌入
'alt_embed_freqs': [1, 2, 4]
# PeakHead 输出范围
'peak_hmf2_range': [200.0, 550.0], 'peak_nmf2_range': [9.0, 13.0]
# run32: PeakHead IRI 形态模板 + SH 低秩 bias 限幅
'peak_bias_h_max': 60.0, 'peak_bias_n_max': 0.4
# run32: 峰场形态守恒（架构已保形态，权重弱化）
'w_hmf2_shape': 0.1, 'w_nmf2_shape': 0.05
# MultiScaleAdaptiveGate
'gate_h_scale': 100.0, 'w_gate_entropy': 0.01
# 早停
'ne_patience': 4, 'peak_patience': 4
# 廓线对齐
'w_profile_align': 0.05, 'w_profile_align_val': 0.1
# 峰场平滑
'w_peak_smooth': 0.01, 'peak_smooth_samples': 256
# 物理损失（背景信任 LT+高度自适应）
'w_bkg_low': 0.12, 'w_bkg_night': 0.08, 'uplift_lt_sigma': 5.0
# IRI 梯度路径闭合
'w_iri_struct': 0.05
# GIRO 直接约束
'w_giro_ne_val': 0.5, 'w_giro_argmax': 0.2, 'giro_argmax_dh': 10.0
# IRI 形态守恒
'w_shape': 0.05
# run28 NeuralETKFLayer
'enkf_n_members':  8       # ensemble 成员数（消融可设 1/4/16）
'enkf_pert_hidden': 64     # PerturbationNet 隐层
# 兼容键（不再使用，保留以免 config 报错）
'fsia_nhead': 4, 'fsia_nlayers': 2, 'fsia_dim_ff': 256
```

---

## 数据格式

- **FY/训练 coords**: `[B, 4]` — `[lat_geo°, lon_geo°, alt_km, rel_hour]`
- **GIRO coords**: `[B, 5]` — `[lat_geo, lon_geo, alt_dummy=300, rel_hour, lat_aacgm]`
- `lat_siren_n = lat_geo / 90`（无 AACGM）
- **alt_range**: (120, 500) km
- **IRI 峰参数 npy**: `(241, 181, 181)` = 时间×纬度×经度，分辨率 3h×1°×2°；hmF2(km)；NmF2(m⁻³, 自动 →log10)；NaN=无效

---

## Critical Gotchas

- **EWMA τ**: `tau = exp(log_tau) + offset` — **必须 `exp`，不得 `softplus`**
- **IRI proxy 单张量返回**: `forward(x, return_features=False)` → 单张量；不可 `a, b = proxy(x)` 解包
- **giro_mode 仍依赖 iri_peak**: PeakHead 始终需要 `iri_peak`
- **peak_field_smooth_loss 梯度路径**: 子采样 idx 须同时作用于 `hmF2_fused` 和 `coords`
- **profile_peak_alignment_loss**: 必须用 `extras['hmF2_det']`（已 detach），不能用 `peak_params['hmF2']`
- **gate 无 clamp**: `gate = gate_data × gate_phys`，gate_phys ∈ (0,1) → gate ∈ (0,1)。**不得恢复 lt_gate_modulation**（曾导致 gate>1 → clamp 伪影）
- **H_FY/H_vert 零初始化**: 训练初期 update=0，从 IRI baseline 起步逐步学习
- **h_analysis vs h_fused**: `extras['h_fused']` 是 `extras['h_analysis']` 的别名
- **member_weights vs head_weights**: 同一量的两个键名（向后兼容）
- **R_FY 物理先验**: 用绝对深度 `(hmF2-alt)/(hmF2-120)` ∈ [0,1]，alt=120km 处最大；不要用 `delta_alt_n`（190 半量程归一化语义错）
- **所有掩码用 `exp(−x²/σ²)`** — 禁止硬 Boolean 掩码或 `torch.abs(lat)`
- **giro_dataloader 版本差异**: FSIA 读 `extras['peak_params']['hmF2'/'NmF2']`；MDIA 读 `extras['chapman_params']['hmF2_F2'/'NmF2_F2']`
- **checkpoint 不向前兼容**: run28 用 `NeuralETKFLayer` 替换 MHDK；旧 checkpoint 无法加载，必须从头训练

---

## 历史里程碑

| Run | 关键变化 |
|---|---|
| run18 | IRITrustNet + LT-FiLM（已废弃）|
| run22 | DiagonalKalmanLayer 替代 CrossSourceAttention；移除 IRITrustNet/LT-FiLM |
| run24 | PeakHead 梯度隔离；R_FY_net 对称化（含 h_sw）|
| run26 | T3Time 启发：CRF + Regime-Aware Gating + MHDK 向量化 + SpectralSWBranch |
| **run28-B** | **回退 SpectralSWBranch（频率分辨率不足）；保留 NeuralETKFLayer + R_FY 物理先验 + CRF + MHDK 向量化** |
| **run28-C** | (已废弃) PeakHead 拆分 + EIA 物理先验，损坏 hmF2 形态；回退 |
| **run29** | (已废弃) ETKF Ensemble NLL (D2+D4)，训练 NaN 不收敛；回退 |
| **run30** | **基线 = run28-B + peak_field_smooth_loss 各向异性 (sin_MODIP² 加权 lat 项)**；保留 NeuralETKFLayer + R_FY 物理先验 + CRF；主损失维持 single-σ NLL |
| **run31** | **PeakHead α/β 加性范式 + peak_shape_loss 双重保 IRI 形态**：fused = clamp(α×IRI + β, lo, hi)；α∈[0.85,1.15], β∈±60km (hmF2)；α∈[0.95,1.05], β∈±0.5 (NmF2)；w_hmf2_shape=0.3, w_nmf2_shape=0.1 |
| **run32** | **PeakHead 形态-数值解耦：IRI 形态完全保留 + SH 低秩 bias 校准**：fused = IRI + bias_max × tanh(Σ(sh × coef)/2)；hmF2 用单峰基 P1/P3，NmF2 用双峰基 P2；hmf2_bias_net / nmf2_bias_net 完全独立；bias_h ±60km, bias_n ±0.4 log10；w_hmf2_shape=0.1, w_nmf2_shape=0.05 |
| **run34** | **结构保持型神经数据同化 — StructuralDABranch 替代 NeuralETKFLayer + CRF + FusionDecoder + uncertainty_head + MultiScaleAdaptiveGate；IRI 形态完全保留为模板，网络仅学有界 bias 修正 + 不确定性贝叶斯加权；5 个 regime-adaptive 修复全部嵌入；R_FY 物理先验吸收到 σ_obs；新增结构保持损失 monotonic / eia_arch / ne_smooth；PeakHead 极简化（仅 SW-driven 微调）；delta_alt_n 改用底部锚定归一化 (alt-hmF2)/(hmF2-120) |
| **run28** | **NeuralETKFLayer + R_FY 物理先验（底侧 + 夜间增强）** |

---

## ISR Validation

输出目录: `FSIA_INR/isr_validation_outputs/`

- Jicamarca: `D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)`
- Poker Flat: `D:\ISR\DATA\61poker_flat_is_radar(lp)\05min`
