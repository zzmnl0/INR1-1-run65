"""
FSIA-INR 模型配置文件

FSIA-INR: Feature-Space Informed Assimilation INR
核心架构：
    1. IRI proxy 完全冻结，提供背景密度与隐状态
    2. FY/COSMIC 原始 log10Ne 通过共享密度观测算子进入低维联合 ETKF
    3. 共享仿射基函数解码器将低维分析状态映射为连续密度增量
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
    'fy_profile_index_path': None,
    'iri_proxy_path': r'D:\code11\IRI01\output_results\iri_september_full_proxy.pth',
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'save_dir': './checkpoints_fsia/run66-qc2-latent-etkf-density-H-global-localized',

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
    'enkf_n_members': 8,  # ETKF集合成员数；分析异常秩至多为N-1
    'enkf_pert_hidden': 64,
    'enkf_anomaly_parameterization': 'legacy_independent',
    'enkf_scale_init': 1.1,
    'enkf_scale_condition_max': 3.0,
    'density_basis_semantics': 'query_conditioned',
    'analysis_state_semantics': 'legacy_feature_increment',
    'context_semantics': 'query_conditioning',
    'mode_basis_semantics': 'learned_density_basis',
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
    'seed': 42,
    'analysis_seed': 42,
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
    'use_date_blocked_split': False,
    'date_split_manifest': None,
    'development_days': 5,
    'locked_test_days': 5,

    # ==================== run66 两阶段损失 ====================
    'background_epochs': 5,
    'analysis_epochs': 5,
    'profile_points_per_epoch': 8,
    'train_profile_fraction': 1.0,
    'profile_subset_manifest': None,
    'huber_delta': 0.2,
    'w_iri': 0.02,
    'w_increment': 0.01,
    'w_vertical_background': 0.02,
    'w_time_background': 0.01,
    'w_vertical_analysis': 0.05,
    'w_time_analysis': 0.02,
    'analysis_loss_active_only': False,
    'analysis_exact_mode_loss': False,
    'use_covariance_moment_loss': False,
    'use_empirical_covariance_loss': False,
    'covariance_gradient_target': 0.20,
    'covariance_calibration_batches': 20,
    'use_direction_loss': False,
    'direction_gradient_target': 0.04,
    'direction_calibration_batches': 20,
    'structure_batch_size': 32,
    'structure_alt_step_km': 20.0,
    'structure_time_step_hours': 1.0,
    'structure_huber_beta': 0.05,
    'background_residual_cap': 0.5,
    # 固定有效观测方差的初值；Background 阶段结束后由训练集残差稳健校准。
    'r_fy_init': 0.04,
    'r_cosmic_init': 0.04,
    'r_mode': 'global',
    'use_distance_localization': True,
    'representativeness_kernel_path': None,
    'representativeness_floor': 0.25,
    'r_calibration_batches': None,
    'r_sigma_min': 0.05,
    'r_sigma_max': 0.40,
    'r_min_profiles': 200,
    'r_shrinkage_profiles': 200,
    'background_seed_ckpt': (
        './checkpoints_fsia/run66-etkf-loss/best_background_model.pth'),
    'source_dropout': (0.25, 0.25, 0.50),  # M10, M01, M11
    'source_mode_schedule': 'random_profile',

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

    # ==================== IRI 预计算峰参数（结构参考）====================
    # 数据格式：shape (241, 181, 181), 3h×1°×2°, NaN=无效
    # 详见 D:\IRI\data01\edp_peak_npy\readme.txt
    'iri_hmf2_path': r"D:\IRI\data01\edp_peak_npy\IRI_hmF2_20240901_20241001.npy",
    'iri_nmf2_path': r"D:\IRI\data01\edp_peak_npy\IRI_NmF2_20240901_20241001.npy",

    # ==================== IRI 峰对齐特征（iri_align_net）====================
    # 输入: cat(h_iri[128], delta_alt_iri[1], NmF2_IRI_n[1]) → 130D → 64D
    # NmF2_IRI_n    = (NmF2_IRI - 11.0) / 2.0   (归一化幅度)
    # ==================== run61: FY 邻域观测编码 ====================
    # FYNeighborhoodIndex 时空检索参数
    'fy_nb_dt':       1.5,   # 邻域时间半径（小时）
    'fy_nb_dlat':     5.0,   # 邻域纬度半径（度）
    'fy_nb_dlon':    15.0,   # 邻域经度半径（度）
    # 剖面级聚合参数（run61 profile-level）
    'fy_nb_k_prof':    8,    # 最近剖面数
    'fy_nb_n_alt':     8,    # 每剖面高度采样数
    # ==================== run64: COSMIC-2 第三数据源 ====================
    'cosmic_path':    r'D:\cosmic2\cosmic245-274-September\cosmic_september_2024.npy',
    'cosmic_profile_index_path': None,
    'cosmic_nb_dt':       1.5,   # 邻域时间半径（小时）
    'cosmic_nb_dlat':     5.0,   # 邻域纬度半径（度）
    'cosmic_nb_dlon':    15.0,   # 邻域经度半径（度）
    'cosmic_nb_k_prof':   8,     # 最近掩星剖面数
    'cosmic_nb_n_alt':    8,     # 每剖面高度采样数
    'use_cosmic':         True,

    # ==================== 断点续训 ====================
    'resume_ckpt': None,

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
        '数据路径': ['fy_path', 'fy_profile_path', 'fy_profile_index_path',
                    'cosmic_path', 'cosmic_profile_index_path',
                    'iri_proxy_path', 'sw_path',
                    'giro_hmf2_path', 'giro_nmf2_path', 'save_dir'],
        '数据规格': ['total_hours', 'start_date_str', 'bin_size_hours'],
        '物理参数': ['alt_range'],
        '时序参数': ['seq_len', 'tau_kp_init', 'tau_solar_init'],
        'SIREN 架构': ['basis_dim', 'enkf_n_members', 'enkf_pert_hidden',
                       'enkf_anomaly_parameterization', 'enkf_scale_init',
                       'enkf_scale_condition_max', 'density_basis_semantics',
                       'analysis_state_semantics', 'context_semantics',
                       'mode_basis_semantics',
                       'siren_hidden', 'siren_layers', 'omega_0',
                       'omega_low', 'omega_high'],
        'SW 编码器': ['sw_hidden_dim', 'sw_lstm_layers', 'sw_out_dim'],
        '训练超参数': ['batch_size', 'lr', 'weight_decay', 'background_epochs',
                       'analysis_epochs', 'seed', 'analysis_seed', 'device',
                       'num_workers', 'use_memmap', 'train_profile_fraction',
                       'profile_subset_manifest'],
        '损失权重': ['huber_delta', 'w_iri', 'w_increment',
                    'w_vertical_background', 'w_time_background',
                    'w_vertical_analysis', 'w_time_analysis',
                    'analysis_loss_active_only',
                    'analysis_exact_mode_loss',
                    'use_covariance_moment_loss',
                    'use_empirical_covariance_loss',
                    'covariance_gradient_target',
                    'covariance_calibration_batches',
                    'use_direction_loss',
                    'direction_gradient_target',
                    'direction_calibration_batches'],
        '同化控制': ['background_residual_cap',
                    'r_fy_init', 'r_cosmic_init', 'r_mode',
                    'use_distance_localization',
                    'representativeness_kernel_path',
                    'representativeness_floor',
                    'r_calibration_batches', 'r_sigma_min', 'r_sigma_max',
                    'r_min_profiles', 'r_shrinkage_profiles',
                    'background_seed_ckpt', 'source_dropout',
                    'source_mode_schedule'],
        '其他': ['grad_clip', 'use_amp'],
    }

    for cat, keys in categories.items():
        print(f'\n【{cat}】')
        for k in keys:
            if k in CONFIG_MDIA:
                print(f'  {k:35s}: {CONFIG_MDIA[k]}')

    print('\n' + '=' * 60 + '\n')


if __name__ == '__main__':
    print_config_mdia()
