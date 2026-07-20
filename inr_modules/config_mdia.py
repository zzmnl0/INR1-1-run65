"""
FSIA-INR 模型配置文件

FSIA-INR: Feature-Space Informed Assimilation INR
核心架构：
    1. IRI proxy 完全冻结，末层隐状态注入 CrossSourceAttention（方案B：IRI 为 Query 残差交叉注意力）
    2. 融合四路 64D 特征 Token：[f_iri, f_spatial, f_vertical, f_sw]
    3. FusionDecoder: cat(h_fused, alt_n, delta_alt_n) → Ne_delta，增量叠加范式
    4. 数据驱动 PeakHead（替代 NeQuick）直接监督 hmF2/NmF2
    5. 可学习 EWMA 时间常数（τ_kp min 3h, τ_solar min 48h）消除 Kp/F10.7 阶跃
    6. TEC 已完全移除（IGS 分辨率不足，无法约束 EIA 结构）
"""

import torch
import os

_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = 'cuda' if _CUDA_AVAILABLE else 'cpu'

if _CUDA_AVAILABLE:
    _BATCH_SIZE = 4096
    _NUM_WORKERS = 4
    _USE_AMP = True
    _PIN_MEMORY = True
    _PREFETCH_FACTOR = 2
    _PERSISTENT_WORKERS = True
else:
    _BATCH_SIZE = 2048
    _NUM_WORKERS = 0
    _USE_AMP = False
    _PIN_MEMORY = False
    _PREFETCH_FACTOR = 2
    _PERSISTENT_WORKERS = False


CONFIG_MDIA = {
    # ==================== 数据路径 ====================
    'fy_path': r'D:\FYsatellite\EDP_data\fy_202409_clean1.npy',
    'iri_proxy_path': r'D:\code11\IRI01\output_results\iri_september_full_proxy.pth',
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'save_dir': './checkpoints_fsia/run6',

    # ==================== GIRO 电离层测高仪数据（可选）====================
    # 由 preprocess_giro.py 生成，运行一次即可
    # giro_hmf2.npy  shape (N, 5): [lat_geo, lon_geo, rel_hour, hmf2_km,    lat_aacgm]
    # giro_nmf2.npy  shape (N, 5): [lat_geo, lon_geo, rel_hour, nmf2_log10, lat_aacgm]
    'giro_hmf2_path': r'D:\c_shuju\GIRO_hmf2\processed\giro_hmf2_clean.npy',
    'giro_nmf2_path': r'D:\c_shuju\GIRO_hmf2\processed\giro_nmf2_clean.npy',

    # ==================== 数据规格 ====================
    'total_hours': 720.0,
    'start_date_str': '2024-09-01 00:00:00',
    'bin_size_hours': 0.5,

    # ==================== 物理参数 ====================
    'lat_range': (-90.0, 90.0),
    'lon_range': (-180.0, 180.0),
    'alt_range': (120.0, 500.0),

    # ==================== 时序学习参数 ====================
    # seq_len 控制 SpaceWeatherManager 的历史窗口
    # 72h（3天）：覆盖 9×τ_kp（磁暴完整响应），对单月训练窗口足够
    # 168h 对单月数据无信息增益，但带来 (168/72)²≈5.4× 更多 EWMA 计算量
    'seq_len': 36,

    # EWMA 时间常数初始值（可学习，此处仅为初始化参考）
    'tau_kp_init': 8.0,        # 小时，环电流恢复时间尺度
    'tau_solar_init': 72.0,    # 小时，3天F10.7均值（匹配单月训练窗口；648h在30天数据中无统计对比度）

    # ==================== SIREN 架构参数 ====================
    'basis_dim': 64,      # 空间基函数 / 残差网络输出维度
    'siren_hidden': 128,  # SIREN 隐层维度
    'siren_layers': 3,    # SIREN 隐层数量
    'omega_0': 30.0,      # SIREN 频率因子

    # ==================== SW 编码器参数 ====================
    'sw_hidden_dim': 32,   # 每个 LSTM 的隐层维度（Kp + F10.7 各自 32 → 合并 64）
    'sw_lstm_layers': 2,   # Kp LSTM 层数（storm-scale 需要更深的记忆）
    'sw_out_dim': 64,      # SW 编码器输出维度

    # ==================== 训练超参数 ====================
    'batch_size': _BATCH_SIZE,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 10,
    'seed': 42,
    'device': _DEVICE,
    'num_workers': _NUM_WORKERS,
    'pin_memory': _PIN_MEMORY,
    'prefetch_factor': _PREFETCH_FACTOR,
    'persistent_workers': _PERSISTENT_WORKERS,
    'use_memmap': True,

    # ==================== 学习率调度 ====================
    'scheduler_type': 'cosine',
    'min_lr': 1e-6,

    # ==================== 数据划分 ====================
    'val_ratio': 0.1,

    # ==================== 损失函数权重 ====================
    'w_obs': 1.0,           # 观测损失（NLL 或 MSE）权重
    'w_bkg': 0.02,          # 非自适应 fallback（use_adaptive_bkg=True 时不生效；实际由 w_bkg_low/w_bkg_high 控制）
    'w_residual_smooth': 0.15,  # run41: 0.05→0.15 加强残差垂直平滑（∂²Ne_delta/∂h²）
    'w_horizontal_smooth': 0.0, # run50: 0.05→0.0（SIREN 固有平滑冗余，与 NLL 梯度冲突）
    'w_uplift': 0.0,              # run50: 0.03→0.0（GIRO 赤道站已覆盖 EIA，冗余）
    'w_depletion': 0.0,          # run50: 0.005→0.0（同上）
    'w_eia_crest': 0.0,          # run50: 0.005→0.0（EIA 先验归零，GIRO 替代）
    'uplift_threshold_km': 380.0,  # 上涌阈值 km（run17+: 380km，原300km日间赤道通常>300故无效）

    # GIRO 监督损失权重
    # hmF2 MSE 量纲为 km²（均值约 300²=9e4），FY MSE 在 log10 空间（约 0.1）
    # 需大幅降权：w_giro_hmf2=0.0001 对应约 9e4*1e-4=9，与 FY 损失量级相当
    'w_giro_hmf2': 0.0001,  # hmF2 监督（km²）：0.0001×(50km)²=0.25，与 FY log10 MSE(~0.1) 量级相当
    'giro_hmf2_asym_weight': 1.5,   # hmF2 非对称损失基准权重（赤道附近基准值）
    'giro_hmf2_asym_center': 310.0, # 特征动态权重中心点 (km)；预测低于此值时惩罚递增
    'w_giro_nmf2': 0.3,     # NmF2 监督（log10 空间，与 FY 量纲相同）
    'giro_batch_size': 256,  # GIRO mini-batch 大小（独立于 FY batch_size）
    'giro_loss_freq': 2,     # 每批次计算（已移除 ×freq 补偿，每次梯度平稳，不触发 grad_clip 截断）

    # 物理损失计算频率（每 N 个 batch 计算一次，加速训练）
    'physics_loss_freq': 10,

    # ==================== 不确定性学习 ====================
    'use_uncertainty': True,
    'uncertainty_warmup_epochs': 2,
    'log_var_min': -6.0,        # 最大精度 exp(6)≈403，恢复表达能力（原-10风险高；-4过保守）
    'log_var_min_init': -2.0,   # NLL 冷启动时收紧下限，随 ramp 线性放开到 log_var_min
    'log_var_max': 6.0,         # 对称设置
    'log_var_regularization': 0.01,
    'nll_ramp_epochs': 1,       # NLL ramp 周期：1 epoch 内完成 Huber→NLL 过渡（FY NaN已修复，冷启动风险低）

    # ==================== 模型保存 ====================
    'save_interval': 5,
    'early_stopping': True,
    'patience': 8,

    # ==================== 梯度裁剪 ====================
    'grad_clip': 1.0,

    # ==================== 混合精度训练 ====================
    'use_amp': _USE_AMP,

    # ==================== FSIA-INR 注意力参数 ====================
    'fsia_nhead': 4,    # CrossSourceAttention 注意力头数
    'fsia_nlayers': 2,  # 保留兼容性（方案C中未使用 FFN 层）
    'fsia_dim_ff': 256, # 保留兼容性（方案C中未使用 FFN）

    # ==================== run40 SW 频域分支（复活 SpectralSWBranch + regime-aware sw_gate）====================
    # 基于 run28-A vs run28-B ISR 指标对比：A/B 在不同 regime 互补
    # run40 重新引入 SpectralSWBranch，但 sw_gate 输入扩展到 69D（含 cos_SZA/sin_doy/cos_doy）
    # 让网络自适应学习何时启用频域信号（低纬夜间/高纬日间用 freq；其他用 time）
    # bias_init -2 (vs run28-A -5)：起步 sigmoid≈0.12，让 freq 信号更快被网络学到
    'use_sw_freq':       True,    # 启用 SpectralSWBranch（False 退化为 run28-B 仅时域）
    'sw_gate_bias_init': -1.0,    # run41: -2.0→-1.0；sigmoid(-1)≈0.27 起步 freq 贡献 27%

    # ==================== run41 Ne 垂直平滑（解决 EDP 廓线不平滑问题）====================
    'w_ne_vert_smooth': 0.05,     # ∂²Ne_fused/∂alt² 二阶导平滑（与 w_residual_smooth 互补）

    # ==================== 双域早停 ====================
    'ne_patience':    4,     # Ne 点值 MSE 早停 patience
    'peak_patience':  12,    # hmF2/NmF2 峰值指标早停 patience（run60: 4→12，允许峰值充分收敛）

    # ==================== 廓线-峰高对齐损失（Profile-Peak Alignment）====================
    # 在 hmF2_pred 高度构造虚拟坐标，额外前向传播后计算 ∂Ne/∂h=0 约束
    # 打通 PeakHead（2D）与 FusionDecoder（3D）的壁垒，解决 FY 顶部数据下拉剖面问题
    # 每 physics_loss_freq 个 batch 计算一次，复用 h_sw_shared.detach()
    'w_profile_align':     0.1,   # 一阶导数零点损失（∂Ne/∂h=0 @ hmF2，平滑正则）
    'w_profile_align_val': 0.0,   # 值对齐权重（run15: 置零，避免 hmF2_fused 偏差时与 giro_direct 冲突）

    # ==================== FSIA-INR v2：双频空间 SIREN + PeakHead ====================
    'omega_low':  10.0,      # 低频空间 SIREN ω₀（全局结构）
    'omega_high': 30.0,      # 高频空间 SIREN ω₀（精细结构）
    'alt_embed_freqs': [1, 2, 4],       # 多尺度高度嵌入频率（×π）
    'peak_hmf2_range': [200.0, 550.0],  # PeakHead hmF2 输出范围 (km)
    'peak_nmf2_range': [9.0,   13.0],   # PeakHead NmF2 输出范围 (log10)

    # ==================== IRI 预计算峰参数（PeakHead 双通道背景输入）====================
    # 用途：PeakHead(h_spatial, h_sw, hmF2_IRI_n, NmF2_IRI_n)
    # 训练初期 PeakHead 输出 = IRI（零初始化保证）；GIRO 训练后 = IRI + 空间修正
    # 数据格式：shape (241, 181, 181), 3h×1°×2°, NaN=无效
    # 详见 D:\IRI\data01\edp_peak_npy\readme.txt
    'iri_hmf2_path': r"D:\IRI\data01\edp_peak_npy\IRI_hmF2_20240901_20241001.npy",
    'iri_nmf2_path': r"D:\IRI\data01\edp_peak_npy\IRI_NmF2_20240901_20241001.npy",

    # ==================== 峰场空间平滑（保证 GIRO 修正的空间连续性）====================
    # 约束 ∂hmF2_fused/∂lat 和 ∂hmF2_fused/∂lon，防止 GIRO 站点间突变
    # 每 physics_loss_freq 批次计算一次，随机子采样降低二阶导开销
    'w_peak_smooth':       0.01,
    'peak_smooth_samples': 256,

    # ==================== IRI 峰对齐特征（iri_align_net）====================
    # 输入: cat(h_iri[128], delta_alt_iri[1], NmF2_IRI_n[1]) → 130D → 64D
    # NmF2_IRI_n    = (NmF2_IRI - 11.0) / 2.0   (归一化幅度)
    'w_iri_struct': 0.005,         # L_iri_struct：iri_recon_head(h_iri_aligned) → ne_bkg 重建损失

    # ==================== Phase 1-D：GIRO → Ne_fused 直接约束 ====================
    # 修复关键缺陷：当前 giro_mode=True 路径完全跳过 Ne_fused 计算
    # 在 GIRO 观测峰高处运行完整前向，约束 Ne_fused 场结构
    'w_giro_ne_val':  0.5,        # L_giro_ne_val：MSE(Ne_fused(hmF2_obs), NmF2_fused.detach())
    'w_giro_argmax':  0.2,        # L_giro_argmax：弱 argmax，relu(Ne_up-Ne_pk)+relu(Ne_dn-Ne_pk)
    'giro_argmax_dh': 10.0,       # Δh (km)，弱 argmax 偏移量

    # ==================== Phase 2：局部曲率约束（收敛后启用）====================
    'w_curvature': 0.0,           # L4：ReLU(∂²Ne/∂h²) @ hmF2_det；Phase 2 启用时设为 0.05

    # ==================== Plan9：4D 体素池化观测上下文 ====================
    # FY 观测从"训练期 loss 信号"升级为"推断期输入特征"
    # Phase 1（obs_warmup_epochs 之前）：h_obs_context=None，主干先收敛
    # Phase 2（obs_warmup_epochs 之后）：ObsVoxelPool 激活，渐进融合 FY 上下文
    'use_obs_context':    True,    # 是否启用 Plan9 观测上下文（False → 退化为 FSIA-INR v3.0）
    'obs_warmup_epochs':  2,       # Phase 1 预热轮数（第 0..1 epoch 关闭观测上下文）
    # 体素分辨率
    'voxel_lat_res':      2.0,     # 纬度体素分辨率 (°)
    'voxel_lon_res':      5.0,     # 经度体素分辨率 (°)
    'voxel_alt_res':      30.0,    # 高度体素分辨率 (km)
    'voxel_time_res':     0.5,     # 时间体素分辨率 (h)
    # 各向异性软距离权重参数
    'obs_sigma_horiz':    800.0,   # 水平相关长度 (km)，EIA 水平相关尺度
    'obs_sigma_vert':     50.0,    # 垂直相关长度 (km)，标高相关
    'obs_sigma_time':     1.5,     # 时间相关长度 (h)，平静期
    'obs_kp_scale':       3.0,     # Kp 自适应 σ_time 衰减尺度
    # 查询参数
    'obs_K_max':          8,       # 每查询点最大有效体素数
    'obs_max_ctx_voxels': 5000,    # 上下文体素集最大规模（超出随机子采样）
    'obs_n_sigma':        3.0,     # 软距离截断阈值（n × σ_time 时间窗口半径）

    # ==================== run22: 神经数据同化（DiagonalKalmanLayer）====================
    # IRITrustNet / LT-FiLM / lt_gate_modulation 已全部移除
    # B_net(h_sw,lat,sin_lt,cos_lt,sin_I→b[64]) + R_FY_net(alt_n,delta_alt_n,sin_lt,cos_lt→r_fy[64])
    # K = b/(b+r+ε)  (对角近似 Kalman 增益，∈(0,1))
    # LT 信息通过 B_net 的 sin_lt/cos_lt 显式进入，不再作为独立调制
    # use_obs_context=False: Phase1（h_spatial 作为 FY 代理）
    # use_obs_context=True:  Phase2（ObsVoxelPool 启用真实 FY 观测上下文）
    'w_shape':   0.0,     # run50: 0.005→0.0（背景信任已覆盖 IRI 锚定，Pearson-r 冗余）

    # ==================== run32：PeakHead IRI 形态模板 + SH 低秩 bias 解耦 ====================
    # 设计：IRI 形态完全保留（无修改）；FY/GIRO 仅修正 bias，通过 5 维 SH 系数空间低秩展开
    # hmF2 单峰基: [P1, P3, P1·cos_SZA, P1·sin_doy, const]   (单峰馒头形态)
    # NmF2 双峰基: [P2, P2·cos_SZA, P2·sin_doy, P2·cos2_SZA, const]  (双峰驼峰形态)
    # fused = IRI + bias_max × tanh(Σ(coef × sh_basis) / 2.0)
    'peak_bias_h_max':    120.0,   # hmF2 bias 限幅: ±60 km
    'peak_bias_n_max':    1.6,    # NmF2 bias 限幅: ±0.4 log10 (~2.5× 数值)
    # 形态守恒损失（架构已保形态，权重弱化）
    'w_hmf2_shape':       0.005,    # hmF2 形态守恒（双重保险，权重已降低）
    'w_nmf2_shape':       0.005,   # NmF2 形态守恒（同上）

    # ==================== run25：PeakHead 部分 detach + DA 自动加权 IRI 峰锚定 ====================
    # peak_alpha_detach: PeakHead 输入 h_spatial 部分 detach 系数
    #   α=0   (run24)：完全 detach，hmF2 仅 GIRO 监督，但 GIRO 稀疏区域退化
    #   α=1.0 (≤run23)：完全连通，FY 轨道偏差污染 hmF2
    #   α=0.3 (run25)：30% FY 梯度通过；70% 阻断 — 折衷修复 GIRO 稀疏区
    'peak_alpha_detach': 0.3,
    # 注：L_peak_iri (hmF2 IRI MSE 锚定 + NmF2 banded Pearson 形状) 无独立 w_*；
    #     权重 = trust_iri = (1 - K_FY.mean()).detach()，由 DA 不确定性自动决定。

    # P0-C: 高度自适应 bkg 损失（过渡中心上移至 F1/F2 交界 250km）
    'w_bkg_low':          0.25,    # 低高度 bkg 权重
    'w_bkg_high':         0.02,    # 高高度 bkg 权重（弱约束，允许充分修正）
    'w_bkg_transition':  250.0,    # 过渡中心 km（F1/F2 交界）
    'w_bkg_sharpness':    25.0,    # 过渡宽度 km

    # v3.1: 夜间背景信任增强（run60: 置零——推理时 phy_scale_FY/phy_scale_vert 已双重约束夜间，
    # bkg_loss 端不再叠加第三层夜间先验；w_bkg_low=0.25 提供高度维基础 IRI 信任）
    'w_bkg_night':        0.0,     # run60: 0.15→0（去除训练端夜间冗余约束）

    # v3.1: EIA 日间 LT 门控（uplift/depletion 仅白天激活）
    # lt_day_gate = exp(-((LT-12)/uplift_lt_sigma)²)
    #   LT=12(noon): gate=1.0；LT=7/17h: gate≈0.14；LT=4/20h: gate≈0.04
    'uplift_lt_sigma':     5.0,    # 日间门控半宽 (h)

    # P0-D: 训练样本连续软权重降权低高度（E/F1 过渡区 FY GNOS 反演可靠性边界）
    'alt_weight_center':  190.0,   # sigmoid 中心 (km)
    'alt_weight_scale':    20.0,   # sigmoid 宽度 (km)
    'alt_weight_low':       0.3,   # 低高度最低权重（alt→120km 时趋近此值）

    # ==================== run61: FY 邻域观测编码 ====================
    # FYNeighborhoodIndex 时空检索参数
    'fy_nb_dt':       1.5,   # 邻域时间半径（小时）
    'fy_nb_dlat':     5.0,   # 邻域纬度半径（度）
    'fy_nb_dlon':    15.0,   # 邻域经度半径（度）
    'fy_nb_kmax':     64,    # 最大邻居数 K = k_prof × n_alt（接口不变）
    # 剖面级聚合参数（run61 profile-level）
    'fy_nb_k_prof':    8,    # 最近剖面数
    'fy_nb_n_alt':     8,    # 每剖面高度采样数
    'fy_nb_dt_break': 0.05,  # 掩星剖面断点阈值（小时，=3 min）
    # FYObsEncoder 超参数
    'fy_enc_heads': 4,       # 注意力头数
    # run61: 关闭 GIRO 直接约束（调试阶段，PeakHead 绕过后 Ne_val 语义待重建）
    'w_giro_ne_val':  0.0,
    'w_giro_argmax':  0.0,

    # ==================== run64: COSMIC-2 第三数据源 ====================
    'cosmic_path':    r'D:\cosmic2\cosmic245-274-September\cosmic_september_2024.npy',
    'cosmic_nb_dt':       1.5,   # 邻域时间半径（小时）
    'cosmic_nb_dlat':     5.0,   # 邻域纬度半径（度）
    'cosmic_nb_dlon':    15.0,   # 邻域经度半径（度）
    'cosmic_nb_k_prof':   8,     # 最近掩星剖面数
    'cosmic_nb_n_alt':    8,     # 每剖面高度采样数
    'w_cosmic':           1.0,   # COSMIC 损失相对 FY 的权重系数

    # ==================== 断点续训 ====================
    'resume_ckpt':   None,   # 检查点路径；None = 从头训练
    'resume_epochs': None,   # 续训轮数；None = 使用 config['epochs']

}


def get_config_mdia():
    os.makedirs(CONFIG_MDIA['save_dir'], exist_ok=True)
    return CONFIG_MDIA


def update_config_mdia(**kwargs):
    CONFIG_MDIA.update(kwargs)


def print_config_mdia():
    print('\n' + '=' * 60)
    print('FSIA-INR 配置参数')
    print('=' * 60)

    categories = {
        '数据路径': ['fy_path', 'iri_proxy_path', 'sw_path',
                    'giro_hmf2_path', 'giro_nmf2_path', 'save_dir'],
        '数据规格': ['total_hours', 'start_date_str', 'bin_size_hours'],
        '物理参数': ['lat_range', 'lon_range', 'alt_range'],
        '时序参数': ['seq_len', 'tau_kp_init', 'tau_solar_init'],
        'SIREN 架构': ['basis_dim', 'siren_hidden', 'siren_layers', 'omega_0',
                       'omega_low', 'omega_high'],
        'SW 编码器': ['sw_hidden_dim', 'sw_lstm_layers', 'sw_out_dim'],
        '训练超参数': ['batch_size', 'lr', 'weight_decay', 'epochs', 'device',
                       'num_workers', 'use_memmap'],
        '损失权重': ['w_obs', 'w_bkg', 'w_residual_smooth',
                    'w_horizontal_smooth', 'w_uplift', 'w_depletion', 'physics_loss_freq'],
        'GIRO 监督': ['w_giro_hmf2', 'w_giro_nmf2', 'giro_batch_size'],
        '不确定性': ['use_uncertainty', 'uncertainty_warmup_epochs'],
        '其他': ['save_interval', 'early_stopping', 'patience', 'grad_clip', 'use_amp'],
    }

    for cat, keys in categories.items():
        print(f'\n【{cat}】')
        for k in keys:
            if k in CONFIG_MDIA:
                print(f'  {k:35s}: {CONFIG_MDIA[k]}')

    print('\n' + '=' * 60 + '\n')


if __name__ == '__main__':
    print_config_mdia()
