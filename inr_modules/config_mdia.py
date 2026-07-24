"""
FSIA-INR 模型配置文件

FSIA-INR: Feature-Space Informed Assimilation INR
核心架构：
    1. IRI proxy 完全冻结，提供背景密度与隐状态
    2. FY/COSMIC 局部 profile 分别编码后进入 NeuralETKFLayer
    3. FusionDecoder 输出对 IRI 背景的有界增量
    4. 可学习 EWMA 时间常数编码空间天气历史
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
    # clean1 supplies physical columns; clean3's seventh column supplies profile_id only.
    'fy_profile_path': r'D:\FYsatellite\EDP_data\fy_202409_clean3.npy',
    'iri_proxy_path': r'D:\code11\IRI01\output_results\iri_september_full_proxy.pth',
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'save_dir': './checkpoints_fsia/run6',

    # ==================== GIRO 独立评估数据（训练不读取）====================
    # 由 preprocess_giro.py 生成，供 evaluate_giro_peak.py 使用
    # giro_hmf2.npy  shape (N, 5): [lat_geo, lon_geo, rel_hour, hmf2_km,    lat_aacgm]
    # giro_nmf2.npy  shape (N, 5): [lat_geo, lon_geo, rel_hour, nmf2_log10, lat_aacgm]
    'giro_hmf2_path': r'D:\c_shuju\GIRO_hmf2\processed\giro_hmf2_clean.npy',
    'giro_nmf2_path': r'D:\c_shuju\GIRO_hmf2\processed\giro_nmf2_clean.npy',

    # ==================== 数据规格 ====================
    'total_hours': 720.0,
    'start_date_str': '2024-09-01 00:00:00',
    'bin_size_hours': 0.5,

    # ==================== 物理参数 ====================
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
    # 物理损失计算频率（每 N 个 batch 计算一次，加速训练）
    'physics_loss_freq': 10,

    # ==================== 不确定性学习 ====================
    'use_uncertainty': True,
    'uncertainty_warmup_epochs': 5,
    'log_var_min': -6.0,        # 最大精度 exp(6)≈403，恢复表达能力（原-10风险高；-4过保守）
    'log_var_min_init': -2.0,   # NLL 冷启动时收紧下限，随 ramp 线性放开到 log_var_min
    'log_var_max': 6.0,         # 对称设置
    'log_var_regularization': 0.01,
    'nll_ramp_epochs': 1,       # NLL ramp 周期：1 epoch 内完成 Huber→NLL 过渡（FY NaN已修复，冷启动风险低）

    # ==================== 模型保存 ====================
    'save_interval': 5,
    'early_stopping': True,

    # ==================== 梯度裁剪 ====================
    'grad_clip': 1.0,

    # ==================== 混合精度训练 ====================
    'use_amp': _USE_AMP,

    # ==================== run40 SW 频域分支（复活 SpectralSWBranch + regime-aware sw_gate）====================
    # 基于 run28-A vs run28-B ISR 指标对比：A/B 在不同 regime 互补
    # run40 重新引入 SpectralSWBranch，但 sw_gate 输入扩展到 69D（含 cos_SZA/sin_doy/cos_doy）
    # 让网络自适应学习何时启用频域信号（低纬夜间/高纬日间用 freq；其他用 time）
    # bias_init -2 (vs run28-A -5)：起步 sigmoid≈0.12，让 freq 信号更快被网络学到
    'use_sw_freq':       True,    # 启用 SpectralSWBranch（False 退化为 run28-B 仅时域）
    'sw_gate_bias_init': -1.0,    # run41: -2.0→-1.0；sigmoid(-1)≈0.27 起步 freq 贡献 27%

    # ==================== 早停 ====================
    'ne_patience': 4,

    # ==================== 廓线-峰高对齐损失（Profile-Peak Alignment）====================
    # 在 hmF2_pred 高度构造虚拟坐标，额外前向传播后计算 ∂Ne/∂h=0 约束
    # 约束 3D 场在 IRI hmF2 参考高度附近保持峰值结构
    # 每 physics_loss_freq 个 batch 计算一次，复用 h_sw_shared.detach()
    'w_profile_align':     0.1,   # 一阶导数零点损失（∂Ne/∂h=0 @ hmF2，平滑正则）

    # ==================== IRI 预计算峰参数（结构参考）====================
    # 数据格式：shape (241, 181, 181), 3h×1°×2°, NaN=无效
    # 详见 D:\IRI\data01\edp_peak_npy\readme.txt
    'iri_hmf2_path': r"D:\IRI\data01\edp_peak_npy\IRI_hmF2_20240901_20241001.npy",
    'iri_nmf2_path': r"D:\IRI\data01\edp_peak_npy\IRI_NmF2_20240901_20241001.npy",

    # ==================== IRI 峰对齐特征（iri_align_net）====================
    # 输入: cat(h_iri[128], delta_alt_iri[1], NmF2_IRI_n[1]) → 130D → 64D
    # NmF2_IRI_n    = (NmF2_IRI - 11.0) / 2.0   (归一化幅度)
    'w_iri_struct': 0.005,         # L_iri_struct：iri_recon_head(h_iri_aligned) → ne_bkg 重建损失

    # P0-C: 高度自适应 bkg 损失（过渡中心上移至 F1/F2 交界 250km）
    'w_bkg_low':          0.25,    # 低高度 bkg 权重
    'w_bkg_high':         0.02,    # 高高度 bkg 权重（弱约束，允许充分修正）
    'w_bkg_transition':  250.0,    # 过渡中心 km（F1/F2 交界）
    'w_bkg_sharpness':    25.0,    # 过渡宽度 km

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
    # FYObsEncoder 超参数
    'fy_enc_heads': 4,       # 注意力头数
    # ==================== run64: COSMIC-2 第三数据源 ====================
    'cosmic_path':    r'D:\cosmic2\cosmic245-274-September\cosmic_september_2024.npy',
    'cosmic_nb_dt':       1.5,   # 邻域时间半径（小时）
    'cosmic_nb_dlat':     5.0,   # 邻域纬度半径（度）
    'cosmic_nb_dlon':    15.0,   # 邻域经度半径（度）
    'cosmic_nb_k_prof':   8,     # 最近掩星剖面数
    'cosmic_nb_n_alt':    8,     # 每剖面高度采样数
    'w_cosmic':           1.0,   # COSMIC 损失相对 FY 的权重系数

    # ==================== 断点续训 ====================
    'resume_ckpt': None,              # last_training_state.pth；旧 raw state_dict 也可兼容
    'resume_completed_epochs': None,  # 仅旧 raw state_dict 必填；完整状态自动读取

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
        '数据路径': ['fy_path', 'fy_profile_path', 'iri_proxy_path', 'sw_path',
                    'giro_hmf2_path', 'giro_nmf2_path', 'save_dir'],
        '数据规格': ['total_hours', 'start_date_str', 'bin_size_hours'],
        '物理参数': ['alt_range'],
        '时序参数': ['seq_len', 'tau_kp_init', 'tau_solar_init'],
        'SIREN 架构': ['basis_dim', 'siren_hidden', 'siren_layers', 'omega_0',
                       'omega_low', 'omega_high'],
        'SW 编码器': ['sw_hidden_dim', 'sw_lstm_layers', 'sw_out_dim'],
        '训练超参数': ['batch_size', 'lr', 'weight_decay', 'epochs', 'device',
                       'num_workers', 'use_memmap'],
        '损失权重': ['w_obs', 'w_bkg_low', 'w_bkg_high',
                    'w_profile_align', 'physics_loss_freq'],
        '不确定性': ['use_uncertainty', 'uncertainty_warmup_epochs'],
        '其他': ['save_interval', 'early_stopping', 'ne_patience',
                 'grad_clip', 'use_amp'],
    }

    for cat, keys in categories.items():
        print(f'\n【{cat}】')
        for k in keys:
            if k in CONFIG_MDIA:
                print(f'  {k:35s}: {CONFIG_MDIA[k]}')

    print('\n' + '=' * 60 + '\n')


if __name__ == '__main__':
    print_config_mdia()
