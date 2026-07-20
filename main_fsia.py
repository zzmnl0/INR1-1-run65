"""
FSIA-INR 主程序入口 — run22 (Neural Data Assimilation)

Feature-Space Informed Assimilation INR
特征空间信息同化隐式神经表示

run22 改进要点（相对于 run18）：
    1. CrossSourceAttention → DiagonalKalmanLayer（神经 DA，对角 Kalman 增益）
    2. B_net(h_sw,lat,sin_lt,cos_lt,sin_I) → b[64]（背景误差协方差）
    3. R_FY_net(alt_n,delta_alt_n,sin_lt,cos_lt) → r_fy[64]（观测误差协方差）
    4. 移除 IRITrustNet（循环依赖）、LT-FiLM、lt_gate_modulation
    5. L_shape = 1 - Pearson-r(Ne_fused, ne_bkg) — IRI 形态守恒约束

运行方式：
    cd FSIA_INR
    python main_fsia.py

检查点输出：./checkpoints_fsia/run22/best_fsia_model.pth
可视化：    python plot_fsia.py

数据路径（在 inr_modules/config_mdia.py 中修改）：
    fy_path, iri_proxy_path, sw_path, giro_hmf2_path, giro_nmf2_path
"""

import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
inr_modules_dir = os.path.join(current_dir, 'inr_modules')
if inr_modules_dir not in sys.path:
    sys.path.insert(0, inr_modules_dir)

from inr_modules.config_mdia import get_config_mdia, print_config_mdia, update_config_mdia
from inr_modules.mdia.train_fsia import train_fsia
from inr_modules.mdia.evaluation_mdia import evaluate_and_save_report, evaluate_parity
from inr_modules.mdia.visualization_mdia import (plot_global_slice, plot_altitude_profile,
                                                  plot_hmf2_nmf2_map)


def main():
    # ==================== 断点续训接口 ====================
    # 从头训练：_RESUME_CKPT = None
    # 续训：    _RESUME_CKPT = r"...\best_fsia_model.pth"，_RESUME_EPOCHS = N
    _RESUME_CKPT   = None
    _RESUME_EPOCHS = 5

    # ==================== 加载配置 ====================
    config = get_config_mdia()

    # FSIA 检查点目录（每次新训练实验递增 run 编号）
    update_config_mdia(save_dir=r"D:\code11\IRI01\IRI03\INR1-1-run65\checkpoints_fsia\run65")

    if _RESUME_CKPT is not None:
        update_config_mdia(resume_ckpt=_RESUME_CKPT, resume_epochs=_RESUME_EPOCHS)
        print(f'\n[续训模式] 检查点: {_RESUME_CKPT}  续训 {_RESUME_EPOCHS} 轮')
    # run61 重构（基于 run60）：FY 邻域观测特征替代坐标 SIREN
    #   核心缺陷修复：h_spatial（DualFreqSpatialNet）只编码坐标，不含 FY 观测值，
    #     FY 损失将 h_spatial 推向 FY-optimal → 污染 PeakHead hmF2 估计
    #   A. 新增 FYObsEncoder：K 个最近 FY 邻居（9D）→ CrossAttn → h_FY [B,64]
    #      无覆盖时 h_FY=0 → ETKF innovation=0 → update=0 → IRI baseline（安全退化）
    #   B. PeakHead 完全绕过：hmF2_fused=hmF2_IRI, NmF2_fused=NmF2_IRI（直接 passthrough）
    #   C. 删除 DualFreqSpatialNet/AltitudeMultiScaleEmbedding/ResidualSIREN/MagFiLM 等
    #   D. CRF proj_pre 192D→128D（f_iri+h_FY，去掉 h_res）
    #   E. 关闭 GIRO 直接约束（w_giro_ne_val/argmax=0，调试阶段）
    # update_config_mdia(epochs=3, batch_size=512)  # 快速调试

    print_config_mdia()
    print('\n[FSIA-INR] 配置覆盖:')
    print(f'  save_dir    : {config["save_dir"]}')
    print(f'  fsia_nhead  : {config["fsia_nhead"]}')
    print(f'  fsia_nlayers: {config["fsia_nlayers"]}')
    print(f'  fsia_dim_ff : {config["fsia_dim_ff"]}')

    # ==================== 路径检查 ====================
    required_paths = ['fy_path', 'iri_proxy_path', 'sw_path']
    missing = [k for k in required_paths
               if config.get(k) and not os.path.exists(config[k])]
    if missing:
        print('\n以下数据文件缺失，无法继续训练:')
        for k in missing:
            print(f'  {k}: {config[k]}')
        return

    # ==================== 开始训练 ====================
    print('\n' + '=' * 60)
    print('开始 FSIA-INR 训练')
    print('=' * 60)

    results         = train_fsia(config)
    model           = results[0]
    train_losses    = results[1]
    val_losses      = results[2]
    train_loader    = results[3]
    val_loader      = results[4]
    sw_manager      = results[5]
    batch_processor = results[6]

    print(f'\n训练完成！最终验证损失: {val_losses[-1]:.6f}')
    best_ckpt = os.path.join(config['save_dir'], 'best_fsia_model.pth')
    print(f'最佳模型保存于: {best_ckpt}')

    # ==================== 加载最佳模型 ====================
    device = torch.device(config['device'])
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print('已加载最佳模型权重用于评估')

    # ==================== 模型信息 ====================
    print(f'\n最终 EWMA 时间常数:')
    print(f'  τ_kp:    {model.sw_encoder.tau_kp.item():.2f} h')
    print(f'  τ_solar: {model.sw_encoder.tau_solar.item():.2f} h')

    # ==================== 评估 ====================
    print('\n' + '=' * 60)
    print('开始评估')
    print('=' * 60)

    save_dir = config['save_dir']
    evaluate_and_save_report(
        model, train_loader, val_loader, batch_processor, device, save_dir)
    evaluate_parity(
        model, val_loader, batch_processor, device, save_dir)

    # ==================== 可视化 ====================
    print('\n' + '=' * 60)
    print('开始可视化')
    print('=' * 60)

    _vis_days   = [4, 14, 24]
    _vis_hours  = [6, 12, 18]
    _alt_levels = [250, 300, 350, 400, 450]

    for vis_day in _vis_days:
        for vis_hour in _vis_hours:
            plot_global_slice(
                model, sw_manager, device,
                target_day=vis_day, target_hour=vis_hour,
                save_dir=save_dir, config=config,
                alt_levels=_alt_levels, model_name='FSIA-INR')

        plot_hmf2_nmf2_map(
            model, sw_manager, device,
            time_steps=[(vis_day, h) for h in _vis_hours],
            save_dir=save_dir, config=config,
            label=f'day{vis_day:02d}', model_name='FSIA-INR')

    # ---- Jicamarca EDP + ISR 真值（Sep 5, 05/10/15/20 UT）----
    _jic_record = None
    try:
        import datetime as _dt
        _isr_eval_dir = os.path.join(current_dir, 'isr_evaluation')
        if _isr_eval_dir not in sys.path:
            sys.path.insert(0, _isr_eval_dir)
        from isr_loader import load_jicamarca as _load_jic
        _JIC_DIR   = r'D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)'
        _START_UNX = _dt.datetime(2024,  9, 1, tzinfo=_dt.timezone.utc).timestamp()
        _END_UNX   = _dt.datetime(2024, 10, 1, tzinfo=_dt.timezone.utc).timestamp()
        if os.path.isdir(_JIC_DIR):
            _isr_recs  = _load_jic(_JIC_DIR, _START_UNX, _END_UNX)
            _jic_record = next((r for r in _isr_recs if r['date_str'] == '20240905'), None)
            if _jic_record is None:
                print('[EDP] 未找到 Jicamarca 20240905 记录，ISR 叠绘跳过')
        else:
            print(f'[EDP] Jicamarca 数据目录不存在: {_JIC_DIR}，ISR 叠绘跳过')
    except Exception as _e:
        print(f'[EDP] Jicamarca 数据加载失败: {_e}')

    plot_altitude_profile(
        model, sw_manager, device,
        lat=-11.9, lon=-76.0,
        time_hour=4 * 24.0 + 5,
        time_hours=[4 * 24.0 + h for h in [5, 10, 15, 20]],
        save_dir=save_dir, config=config,
        model_name='FSIA-INR',
        isr_record=_jic_record)


if __name__ == '__main__':
    main()
