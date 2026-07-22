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

import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
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

_PROFILE_FIXED_DIR = Path(current_dir) / 'checkpoints_fsia' / 'run65-profile-fixed'
_OLD_RUN65_CKPT = Path(current_dir) / 'checkpoints_fsia' / 'run65' / 'best_fsia_model.pth'
_RESUME_CKPT = str(_PROFILE_FIXED_DIR / 'best_fsia_model.pth')
# 当前日志显示 Epoch 1 已完整保存，Epoch 2 仅运行到 batch 200，故从全局 Epoch 2 重跑。
_RESUME_COMPLETED_EPOCHS = 1
_RESUME_BEST_VAL = 0.146658
_RESUME_BEST_PEAK = 0.6334


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _file_identity(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'sha256': digest.hexdigest()}


def _code_identity(root):
    digest = hashlib.sha256()
    files = sorted(Path(root).rglob('*.py'))
    for path in files:
        rel = path.relative_to(root).as_posix().encode('utf-8')
        digest.update(rel)
        digest.update(path.read_bytes())
    return {'python_files': len(files), 'sha256': digest.hexdigest()}


def _write_run_manifest(config):
    data_keys = [key for key, value in config.items()
                 if key.endswith('_path') and value and os.path.isfile(value)]
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': config,
        'data_identity': {key: _file_identity(config[key]) for key in data_keys},
        'code_identity': _code_identity(current_dir),
        'environment': {
            'python': sys.version,
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cuda_device': (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
        },
    }
    path = Path(config['save_dir']) / 'run_manifest.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def _record_resume_manifest(config):
    path = Path(config['save_dir']) / 'run_manifest.json'
    if not path.exists():
        return
    with path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    manifest.setdefault('resume_events', []).append({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': _file_identity(config['resume_ckpt']),
        'code_identity': _code_identity(current_dir),
    })
    temp_path = path.with_suffix('.json.tmp')
    with temp_path.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _strict_load_finite(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    bad = [name for name, value in state.items()
           if torch.is_tensor(value) and not torch.isfinite(value).all()]
    if bad:
        raise ValueError(f'checkpoint contains non-finite parameters: {bad[:5]}')


def main(eval_only=False):
    # ==================== 断点续训接口 ====================
    # config['epochs'] 始终表示目标总轮数；旧 raw state_dict 必须显式给出已完成轮数。

    # ==================== 加载配置 ====================
    config = get_config_mdia()

    # FSIA 检查点目录（每次新训练实验递增 run 编号）
    update_config_mdia(
        save_dir=str(_PROFILE_FIXED_DIR),
        resume_ckpt=None,
        resume_completed_epochs=None,
        resume_best_val=None,
        resume_best_peak=None,
        resume_best_epoch=None,
        eval_only=False,
    )

    if eval_only:
        update_config_mdia(
            eval_only=True,
            resume_ckpt=os.path.join(config['save_dir'], 'best_fsia_model.pth'))
    elif _RESUME_CKPT is not None:
        last_state = _PROFILE_FIXED_DIR / 'last_training_state.pth'
        resume_path = str(last_state) if last_state.exists() else _RESUME_CKPT
        update_config_mdia(
            resume_ckpt=resume_path,
            resume_completed_epochs=_RESUME_COMPLETED_EPOCHS,
            resume_best_val=_RESUME_BEST_VAL,
            resume_best_peak=_RESUME_BEST_PEAK,
            resume_best_epoch=_RESUME_COMPLETED_EPOCHS,
        )
        print(f'\n[续训模式] 检查点: {resume_path}')
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
    required_paths = ['fy_path', 'fy_profile_path', 'iri_proxy_path', 'sw_path']
    if config.get('w_cosmic', 0.0) > 0:
        required_paths.append('cosmic_path')
    missing = [k for k in required_paths
               if not config.get(k) or not os.path.exists(config[k])]
    if missing:
        print('\n以下数据文件缺失，无法继续训练:')
        for k in missing:
            print(f'  {k}: {config[k]}')
        return

    best_ckpt = Path(config['save_dir']) / 'best_fsia_model.pth'
    if best_ckpt.resolve() == _OLD_RUN65_CKPT.resolve():
        raise RuntimeError('拒绝覆盖历史 run65 checkpoint')
    is_resume = bool(config.get('resume_ckpt')) and not eval_only
    if is_resume:
        if not os.path.exists(config['resume_ckpt']):
            raise FileNotFoundError(f"续训 checkpoint 不存在: {config['resume_ckpt']}")
        _record_resume_manifest(config)
    elif not eval_only:
        Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
        if best_ckpt.exists():
            raise FileExistsError(f'新训练目录已有 checkpoint，请换用空目录: {best_ckpt}')
        _write_run_manifest(config)

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
    iri_peak_manager = results[7]

    if val_losses:
        print(f'\n训练完成！最终验证损失: {val_losses[-1]:.6f}')
    best_ckpt = os.path.join(config['save_dir'], 'best_fsia_model.pth')
    print(f'最佳模型保存于: {best_ckpt}')

    # ==================== 加载最佳模型 ====================
    device = torch.device(config['device'])
    if os.path.exists(best_ckpt):
        _strict_load_finite(model, best_ckpt, device)
        print('已严格加载最佳模型，且参数全部有限')

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
        model, train_loader, val_loader, batch_processor, device, save_dir,
        iri_peak_manager=iri_peak_manager)
    evaluate_parity(
        model, val_loader, batch_processor, device, save_dir,
        iri_peak_manager=iri_peak_manager)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true',
                        help='load best checkpoint and skip training')
    args = parser.parse_args()
    _PROFILE_FIXED_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _PROFILE_FIXED_DIR / 'training.log'
    log_mode = 'a' if args.eval_only or _RESUME_CKPT else 'x'
    with log_path.open(log_mode, encoding='utf-8', buffering=1) as log_stream:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_stream)), \
                contextlib.redirect_stderr(_Tee(sys.stderr, log_stream)):
            main(eval_only=args.eval_only)
