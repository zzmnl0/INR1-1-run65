"""Train, resume, evaluate, and visualize the FSIA-INR model."""

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

_DEFAULT_RUN_NAME = 'run66-qc2-latent-etkf-density-H-global-localized'
_DEFAULT_BACKGROUND_SEED = (
    Path(current_dir) / 'checkpoints_fsia' / 'run66-etkf-loss'
    / 'best_background_model.pth')
_OLD_RUN65_CKPT = Path(current_dir) / 'checkpoints_fsia' / 'run65' / 'best_fsia_model.pth'


def _run_directory(run_name):
    if Path(run_name).name != run_name or run_name in ('', '.', '..'):
        raise ValueError('run_name must be one directory name')
    return Path(current_dir) / 'checkpoints_fsia' / run_name


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
    data_keys = [
        key for key in (
            'fy_path', 'fy_profile_path', 'fy_profile_index_path',
            'fy_qc_report_path', 'cosmic_path', 'cosmic_profile_index_path',
            'cosmic_qc_report_path', 'iri_proxy_path',
            'iri_hmf2_path', 'iri_nmf2_path', 'sw_path')
        if config.get(key) and os.path.isfile(config[key])
    ]
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


def main(eval_only=False, resume_ckpt=None, run_name=_DEFAULT_RUN_NAME,
         r_mode='global', background_seed=None, qc_data=True,
         distance_localization=True):
    # ==================== 加载配置 ====================
    config = get_config_mdia()
    run_dir = _run_directory(run_name)
    background_seed = (
        str(_DEFAULT_BACKGROUND_SEED)
        if background_seed is None else background_seed)

    # FSIA 检查点目录（每次新训练实验递增 run 编号）
    update_config_mdia(
        save_dir=str(run_dir),
        resume_ckpt=None,
        eval_only=False,
        r_mode=r_mode,
        use_distance_localization=distance_localization,
        background_seed_ckpt=background_seed,
    )
    if qc_data:
        update_config_mdia(
            fy_path=r'D:\FYsatellite\EDP_data\fy_202409_qc_v2.npy',
            fy_profile_path=None,
            fy_profile_index_path=(
                r'D:\FYsatellite\EDP_data\fy_202409_qc_v2_index.npz'),
            fy_qc_report_path=(
                r'D:\FYsatellite\EDP_data\fy_202409_qc_v2_report.json'),
            cosmic_path=(
                r'D:\cosmic2\cosmic245-274-September'
                r'\cosmic_september_2024_qc.npy'),
            cosmic_profile_index_path=(
                r'D:\cosmic2\cosmic245-274-September'
                r'\cosmic_september_2024_qc_index.npz'),
            cosmic_qc_report_path=(
                r'D:\cosmic2\cosmic245-274-September'
                r'\cosmic_september_2024_qc_report.json'),
        )

    if eval_only:
        update_config_mdia(
            eval_only=True,
            resume_ckpt=os.path.join(config['save_dir'], 'best_fsia_model.pth'))
    elif resume_ckpt is not None:
        last_state = run_dir / 'last_training_state.pth'
        resume_path = (
            str(last_state) if resume_ckpt == 'auto' and last_state.exists()
            else resume_ckpt)
        if resume_path == 'auto':
            raise FileNotFoundError(f'未找到自动续训状态: {last_state}')
        update_config_mdia(resume_ckpt=resume_path)
        print(f'\n[续训模式] 检查点: {resume_path}')
    # update_config_mdia(background_epochs=1, analysis_epochs=1, batch_size=512)

    print_config_mdia()
    print(f'\n[FSIA-INR] save_dir: {config["save_dir"]}')

    # ==================== 路径检查 ====================
    required_paths = ['fy_path', 'iri_proxy_path', 'sw_path']
    required_paths.append(
        'fy_profile_index_path'
        if config.get('fy_profile_index_path') else 'fy_profile_path')
    if config.get('use_cosmic', True):
        required_paths.append('cosmic_path')
        if config.get('cosmic_profile_index_path'):
            required_paths.append('cosmic_profile_index_path')
    if qc_data:
        required_paths.extend(['fy_qc_report_path', 'cosmic_qc_report_path'])
    missing = [k for k in required_paths
               if not config.get(k) or not os.path.exists(config[k])]
    if missing:
        print('\n以下数据文件缺失，无法继续训练:')
        for k in missing:
            print(f'  {k}: {config[k]}')
        return
    if qc_data:
        for source, data_key, index_key, report_key in (
                ('FY', 'fy_path', 'fy_profile_index_path', 'fy_qc_report_path'),
                ('COSMIC', 'cosmic_path', 'cosmic_profile_index_path',
                 'cosmic_qc_report_path')):
            with open(config[report_key], encoding='utf-8') as stream:
                report = json.load(stream)
            if not report.get('audit', {}).get('passed'):
                raise RuntimeError(f'{source} QC未通过审计门禁，拒绝启动训练')
            expected = report.get('outputs', {})
            for key, output_name in ((data_key, 'npy'), (index_key, 'npz')):
                actual = _file_identity(config[key])['sha256']
                if actual != expected.get(output_name, {}).get('sha256'):
                    raise RuntimeError(f'{source} QC {output_name} SHA256不匹配')

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
        model, train_loader, val_loader, batch_processor, save_dir,
        iri_peak_manager=iri_peak_manager)
    evaluate_parity(
        model, val_loader, batch_processor, save_dir,
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
                save_dir=save_dir,
                alt_levels=_alt_levels, model_name='FSIA-INR',
                iri_peak_manager=iri_peak_manager,
                fy_nb_index=batch_processor.fy_nb_index,
                cosmic_nb_index=batch_processor.cosmic_nb_index)

        plot_hmf2_nmf2_map(
            model, sw_manager, device,
            time_steps=[(vis_day, h) for h in _vis_hours],
            save_dir=save_dir,
            label=f'day{vis_day:02d}', model_name='FSIA-INR',
            iri_peak_manager=iri_peak_manager,
            fy_nb_index=batch_processor.fy_nb_index,
            cosmic_nb_index=batch_processor.cosmic_nb_index)

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
        iri_peak_manager=iri_peak_manager,
        isr_record=_jic_record,
        fy_nb_index=batch_processor.fy_nb_index,
        cosmic_nb_index=batch_processor.cosmic_nb_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true',
                        help='load best checkpoint and skip training')
    parser.add_argument(
        '--resume', nargs='?', const='auto', default=None,
        help='resume last_training_state.pth, or resume from an explicit path')
    parser.add_argument('--run-name', default=_DEFAULT_RUN_NAME,
                        help='output directory name under checkpoints_fsia')
    parser.add_argument('--r-mode', choices=('global', 'stratified'),
                        default='global')
    parser.add_argument('--distance-localization', action='store_true',
                        help='inflate fixed R using deterministic local distance')
    parser.add_argument('--background-seed', default=None,
                        help='shared Background checkpoint for Analysis-only runs')
    parser.add_argument(
        '--qc-data', action='store_true',
        help='use audited FY/COSMIC NPY+NPZ QC products')
    args = parser.parse_args()
    run_dir = _run_directory(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'training.log'
    log_mode = 'a' if args.eval_only or args.resume else 'x'
    with log_path.open(log_mode, encoding='utf-8', buffering=1) as log_stream:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_stream)), \
                contextlib.redirect_stderr(_Tee(sys.stderr, log_stream)):
            main(
                eval_only=args.eval_only,
                resume_ckpt=args.resume,
                run_name=args.run_name,
                r_mode=args.r_mode,
                background_seed=args.background_seed,
                qc_data=args.qc_data,
                distance_localization=args.distance_localization,
            )
