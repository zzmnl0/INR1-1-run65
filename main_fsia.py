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

_DEFAULT_RUN_NAME = 'run66-m2v-qcv2-dateblocked-background'
_TRUST_GATE_SEMANTICS = 'fixed_altitude_localtime_dip_smoothstep_v1'
_TRUST_GATE_BACKGROUND_SEMANTICS = (
    'qc_v2_date_blocked_train_only_continuous_trust_gate_v1')
_M2W_DOMAIN_SEMANTICS = 'strict_200_500_domain_v1'
_M2W_RUN_SEMANTICS = 'M2-W_continuous_physical_local_letkf_200_500'
_M2W_BACKGROUND_SEMANTICS = 'qc_v2_date_blocked_train_only_m2w_200_500_v1'
_M2W_GATE_BACKGROUND_SEMANTICS = (
    'qc_v2_date_blocked_train_only_m2w_200_500_continuous_trust_gate_v1')
_M2V_DOMAIN_SEMANTICS = 'legacy_120_500_domain_v1'
_M2W_HYBRID_DOMAIN_SEMANTICS = (
    'hybrid_120_500_model_200_500_observation_v1')
_M2W_HYBRID_RUN_SEMANTICS = (
    'M2-W_continuous_physical_local_letkf_120_500_obs_200_500')
_M2W_HYBRID_BACKGROUND_SEMANTICS = (
    'qc_v2_date_blocked_train_only_m2w_120_500_obs_200_500_low_alt_prior_v1')
_LOW_ALTITUDE_PRIOR_SEMANTICS = (
    'soft_iri_background_zero_analysis_increment_v1')
_HYBRID_LOW_ALTITUDE_LEVELS = tuple(float(value) for value in range(120, 200, 10))
_DOMAIN_SPECS = {
    _M2V_DOMAIN_SEMANTICS: {
        'checkpoint_format_version': 12,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (120.0, 500.0),
        'peak_search_alt_range': (120.0, 500.0),
        'run_semantics': None,
    },
    _M2W_DOMAIN_SEMANTICS: {
        'checkpoint_format_version': 13,
        'alt_range': (200.0, 500.0),
        'observation_alt_range': (200.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
        'run_semantics': _M2W_RUN_SEMANTICS,
    },
    _M2W_HYBRID_DOMAIN_SEMANTICS: {
        'checkpoint_format_version': 14,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (200.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
        'run_semantics': _M2W_HYBRID_RUN_SEMANTICS,
    },
}
_CLI_DOMAIN_SPECS = {
    '200-500': _M2W_DOMAIN_SEMANTICS,
    '120-500-obs-200-500': _M2W_HYBRID_DOMAIN_SEMANTICS,
}
_UNSET = object()
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
            'iri_hmf2_path', 'iri_nmf2_path', 'sw_path',
            'representativeness_kernel_path')
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
        'target_stage': (
            'background' if config.get('background_only', False)
            else 'analysis'),
        'background_trust_gate_enabled': bool(
            config.get('background_trust_gate_enabled', False)),
        'background_trust_gate_semantics': config.get(
            'background_trust_gate_semantics', 'disabled'),
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
         r_mode='global', background_seed=_UNSET, qc_data=True,
         distance_localization=True, basis_dim=None, enkf_members=None,
         anomaly_parameterization=None, density_basis_semantics=None,
         train_profile_fraction=None,
         profile_subset_manifest=None, covariance_moment=None,
         empirical_covariance_loss=None,
         covariance_gradient_target=None,
         observation_gram_loss=None, gram_gradient_target=None,
         direction_loss=None,
         post_train_evaluation=True, date_blocked_split=None,
         date_split_manifest=None, source_mode_schedule=None,
         analysis_epochs=None, background_epochs=None, max_train_batches=None,
         max_validation_batches=None,
         w_time_analysis=None, analysis_active_only_loss=None,
         analysis_exact_mode_loss=None, representativeness_kernel=None,
         background_only=False,
         include_isr_overlay=False, background_trust_gate=None,
         model_domain=None, smoke_run=False):
    # ==================== 加载配置 ====================
    config = get_config_mdia()
    run_dir = _run_directory(run_name)
    prior_config = {}
    manifest_path = run_dir / 'run_manifest.json'
    if (eval_only or resume_ckpt is not None) and manifest_path.is_file():
        with manifest_path.open(encoding='utf-8') as stream:
            prior_config = json.load(stream).get('config', {})
    recorded_domain = prior_config.get(
        'model_domain_semantics', _M2V_DOMAIN_SEMANTICS)
    model_domain_semantics = (
        _CLI_DOMAIN_SPECS[model_domain] if model_domain is not None
        else recorded_domain)
    if model_domain_semantics not in _DOMAIN_SPECS:
        raise ValueError(f'unsupported model-domain semantics: {model_domain_semantics}')
    if prior_config and model_domain_semantics != recorded_domain:
        raise ValueError('model domain differs from the run manifest')
    domain_spec = _DOMAIN_SPECS[model_domain_semantics]
    m2w_domain = model_domain_semantics != _M2V_DOMAIN_SEMANTICS
    hybrid_domain = model_domain_semantics == _M2W_HYBRID_DOMAIN_SEMANTICS
    if model_domain_semantics == _M2V_DOMAIN_SEMANTICS:
        alt_range = tuple(prior_config.get('alt_range', config['alt_range']))
        observation_alt_range = tuple(prior_config.get(
            'observation_alt_range') or alt_range)
        peak_search_alt_range = tuple(prior_config.get(
            'peak_search_alt_range') or alt_range)
        checkpoint_format_version = int(prior_config.get(
            'checkpoint_format_version', config['checkpoint_format_version']))
        run_semantics = prior_config.get('run_semantics', config['run_semantics'])
    else:
        alt_range = domain_spec['alt_range']
        observation_alt_range = domain_spec['observation_alt_range']
        peak_search_alt_range = domain_spec['peak_search_alt_range']
        checkpoint_format_version = domain_spec['checkpoint_format_version']
        run_semantics = domain_spec['run_semantics']
    basis_dim = int(
        prior_config.get('basis_dim', config['basis_dim'])
        if basis_dim is None else basis_dim)
    enkf_members = int(
        prior_config.get('enkf_n_members', config['enkf_n_members'])
        if enkf_members is None else enkf_members)
    anomaly_parameterization = (
        prior_config.get(
            'enkf_anomaly_parameterization',
            config['enkf_anomaly_parameterization'])
        if anomaly_parameterization is None else anomaly_parameterization)
    density_basis_semantics = (
        prior_config.get(
            'density_basis_semantics', config['density_basis_semantics'])
        if density_basis_semantics is None else density_basis_semantics)
    train_profile_fraction = float(
        prior_config.get(
            'train_profile_fraction', config['train_profile_fraction'])
        if train_profile_fraction is None else train_profile_fraction)
    if profile_subset_manifest is None:
        profile_subset_manifest = prior_config.get('profile_subset_manifest')
    date_blocked_split = bool(
        prior_config.get('use_date_blocked_split',
                         config.get('use_date_blocked_split', True))
        if date_blocked_split is None else date_blocked_split)
    if date_split_manifest is None:
        date_split_manifest = prior_config.get(
            'date_split_manifest', config.get('date_split_manifest'))
    if date_blocked_split and not date_split_manifest:
        date_split_manifest = str(run_dir / 'date_split_manifest.json')
    covariance_moment = bool(
        prior_config.get('use_covariance_moment_loss', False)
        if covariance_moment is None else covariance_moment)
    empirical_covariance_loss = bool(
        prior_config.get('use_empirical_covariance_loss', False)
        if empirical_covariance_loss is None else empirical_covariance_loss)
    covariance_gradient_target = float(
        prior_config.get(
            'covariance_gradient_target',
            config.get('covariance_gradient_target', 0.20))
        if covariance_gradient_target is None else covariance_gradient_target)
    if not 0.0 < covariance_gradient_target <= 1.0:
        raise ValueError('covariance_gradient_target must be in (0, 1]')
    observation_gram_loss = bool(
        prior_config.get(
            'use_observation_gram_loss',
            config.get('use_observation_gram_loss', False))
        if observation_gram_loss is None else observation_gram_loss)
    gram_gradient_target = float(
        prior_config.get(
            'gram_gradient_target', config.get('gram_gradient_target', 0.02))
        if gram_gradient_target is None else gram_gradient_target)
    if not 0.0 < gram_gradient_target <= 1.0:
        raise ValueError('gram_gradient_target must be in (0, 1]')
    direction_loss = bool(
        prior_config.get('use_direction_loss', False)
        if direction_loss is None else direction_loss)
    source_mode_schedule = (
        prior_config.get(
            'source_mode_schedule', config.get(
                'source_mode_schedule', 'random_profile'))
        if source_mode_schedule is None else source_mode_schedule)
    analysis_epochs = int(
        prior_config.get('analysis_epochs', config['analysis_epochs'])
        if analysis_epochs is None else analysis_epochs)
    background_epochs = int(
        prior_config.get('background_epochs', config['background_epochs'])
        if background_epochs is None else background_epochs)
    if background_epochs <= 0 or analysis_epochs <= 0:
        raise ValueError('background_epochs and analysis_epochs must be positive')
    if smoke_run:
        if max_train_batches not in (None, 1) or max_validation_batches not in (None, 1):
            raise ValueError('v14 smoke runs require one train and one validation batch')
        max_train_batches = 1
        max_validation_batches = 1
    if smoke_run and (background_epochs, analysis_epochs) != (1, 1):
        raise ValueError('v14 smoke runs require exactly 1 Background and 1 Analysis epoch')
    if hybrid_domain and not smoke_run and (background_epochs, analysis_epochs) != (5, 10):
        raise ValueError('v14 full training requires exactly 5 Background and 10 Analysis epochs')
    max_validation_batches = (
        prior_config.get('max_validation_batches')
        if max_validation_batches is None else max_validation_batches)
    if max_validation_batches is not None:
        max_validation_batches = int(max_validation_batches)
        if max_validation_batches <= 0:
            raise ValueError('max_validation_batches must be positive')
    w_time_analysis = float(
        prior_config.get('w_time_analysis', config['w_time_analysis'])
        if w_time_analysis is None else w_time_analysis)
    if w_time_analysis < 0.0:
        raise ValueError('w_time_analysis must be non-negative')
    analysis_active_only_loss = bool(
        prior_config.get(
            'analysis_loss_active_only',
            config.get('analysis_loss_active_only', False))
        if analysis_active_only_loss is None else analysis_active_only_loss)
    analysis_exact_mode_loss = bool(
        prior_config.get(
            'analysis_exact_mode_loss',
            config.get('analysis_exact_mode_loss', False))
        if analysis_exact_mode_loss is None else analysis_exact_mode_loss)
    if representativeness_kernel is None:
        representativeness_kernel = prior_config.get(
            'representativeness_kernel_path',
            config.get('representativeness_kernel_path'))
    representativeness_floor = float(
        prior_config.get(
            'representativeness_floor',
            config.get('representativeness_floor', 0.25)))
    if not 0.0 < representativeness_floor <= 1.0:
        raise ValueError('representativeness_floor must be in (0, 1]')
    recorded_gate = bool(prior_config.get(
        'background_trust_gate_enabled', False))
    if background_trust_gate is None:
        background_trust_gate = recorded_gate
    else:
        background_trust_gate = bool(background_trust_gate)
        if prior_config and background_trust_gate != recorded_gate:
            raise ValueError(
                'background trust-gate flag differs from the run manifest')
    if hybrid_domain and background_trust_gate:
        raise ValueError('hybrid v14 training requires gate-off')
    if background_trust_gate:
        recorded_gate_semantics = prior_config.get(
            'background_trust_gate_semantics', _TRUST_GATE_SEMANTICS)
        if recorded_gate_semantics != _TRUST_GATE_SEMANTICS:
            raise ValueError('unsupported Background trust-gate semantics')
        background_gate_semantics = _TRUST_GATE_SEMANTICS
        background_training_semantics = (
            _M2W_GATE_BACKGROUND_SEMANTICS if m2w_domain
            else _TRUST_GATE_BACKGROUND_SEMANTICS)
    else:
        background_gate_semantics = 'disabled'
    if background_seed is _UNSET:
        # Resume/evaluation inherits the target run's recorded semantics;
        # a fresh QC-v2 run intentionally has no external Background seed.
        background_seed = prior_config.get('background_seed_ckpt')
    prior_background_semantics = prior_config.get(
        'background_training_semantics')
    if background_seed is not None:
        background_seed = str(background_seed)
        if (m2w_domain or background_trust_gate
                or prior_background_semantics == 'qc_v2_date_blocked_train_only'
                or (not prior_config and run_name == _DEFAULT_RUN_NAME)):
            raise ValueError(
                'QC-v2 Background run cannot be combined with an external seed')
        background_training_semantics = 'frozen_external_seed'
    else:
        if not background_trust_gate:
            background_training_semantics = (
                prior_background_semantics
                or (_M2W_HYBRID_BACKGROUND_SEMANTICS if hybrid_domain
                    else _M2W_BACKGROUND_SEMANTICS if m2w_domain
                    else config.get('background_training_semantics',
                                    'qc_v2_date_blocked_train_only')))

    # FSIA 检查点目录（每次新训练实验递增 run 编号）
    update_config_mdia(
        save_dir=str(run_dir),
        resume_ckpt=None,
        eval_only=False,
        r_mode=r_mode,
        use_distance_localization=distance_localization,
        background_seed_ckpt=background_seed,
        background_training_semantics=background_training_semantics,
        background_only=bool(background_only),
        smoke_run=bool(smoke_run),
        background_trust_gate_enabled=bool(background_trust_gate),
        background_trust_gate_semantics=background_gate_semantics,
        basis_dim=basis_dim,
        enkf_n_members=enkf_members,
        enkf_anomaly_parameterization=anomaly_parameterization,
        density_basis_semantics=density_basis_semantics,
        train_profile_fraction=train_profile_fraction,
        profile_subset_manifest=profile_subset_manifest,
        use_covariance_moment_loss=bool(covariance_moment),
        use_empirical_covariance_loss=bool(empirical_covariance_loss),
        covariance_gradient_target=covariance_gradient_target,
        use_observation_gram_loss=bool(observation_gram_loss),
        gram_gradient_target=gram_gradient_target,
        use_direction_loss=bool(direction_loss),
        use_date_blocked_split=date_blocked_split,
        date_split_manifest=date_split_manifest,
        source_mode_schedule=source_mode_schedule,
        background_epochs=background_epochs,
        analysis_epochs=analysis_epochs,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
        w_time_analysis=w_time_analysis,
        analysis_loss_active_only=analysis_active_only_loss,
        analysis_exact_mode_loss=analysis_exact_mode_loss,
        representativeness_kernel_path=representativeness_kernel,
        representativeness_floor=representativeness_floor,
        include_isr_overlay=bool(include_isr_overlay),
        model_domain_semantics=model_domain_semantics,
        alt_range=alt_range,
        observation_alt_range=observation_alt_range,
        peak_search_alt_range=peak_search_alt_range,
        low_altitude_prior_range=(120.0, 200.0) if hybrid_domain else None,
        low_altitude_prior_semantics=(
            _LOW_ALTITUDE_PRIOR_SEMANTICS if hybrid_domain else None),
        low_altitude_anchor_levels_km=(
            _HYBRID_LOW_ALTITUDE_LEVELS if hybrid_domain else ()),
        low_altitude_anchor_profiles_per_source=(16 if hybrid_domain else 0),
        w_low_altitude_background_iri=(0.02 if hybrid_domain else 0.0),
        w_low_altitude_analysis_increment=(0.01 if hybrid_domain else 0.0),
        low_altitude_gradient_ratio_max=(0.25 if hybrid_domain else 0.25),
        smoke_auxiliary_gradient_ratio_max=(0.25 if hybrid_domain else 0.30),
        checkpoint_format_version=checkpoint_format_version,
        run_semantics=run_semantics,
        checkpoint_selection_semantics=(
            'mean_ccc_then_rmse_then_pearson_v1' if m2w_domain
            else prior_config.get('checkpoint_selection_semantics',
                                  'mean_profile_rmse_v1')),
    )
    if smoke_run:
        update_config_mdia(r_calibration_batches=1, gram_calibration_batches=1)
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
        summary_path = run_dir / 'training_summary.json'
        if summary_path.is_file():
            with summary_path.open(encoding='utf-8') as stream:
                training_summary = json.load(stream)
            if training_summary.get('completed_stage') != 'analysis':
                raise ValueError(
                    'eval-only requires a completed Analysis checkpoint; '
                    'Background-only artifacts are not valid Analysis models')
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
    if representativeness_kernel:
        required_paths.append('representativeness_kernel_path')
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
    best_ckpt = os.path.join(
        config['save_dir'],
        'best_background_model.pth' if background_only
        else 'best_fsia_model.pth')
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
    if background_only or not post_train_evaluation:
        return

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
    from inr_modules.mdia.checkpoint_io import allowed_observation_profile_ids
    visualization_profiles = allowed_observation_profile_ids(config)

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
                cosmic_nb_index=batch_processor.cosmic_nb_index,
                allowed_profile_ids=visualization_profiles)

        plot_hmf2_nmf2_map(
            model, sw_manager, device,
            time_steps=[(vis_day, h) for h in _vis_hours],
            save_dir=save_dir,
            label=f'day{vis_day:02d}', model_name='FSIA-INR',
            iri_peak_manager=iri_peak_manager,
            fy_nb_index=batch_processor.fy_nb_index,
            cosmic_nb_index=batch_processor.cosmic_nb_index,
            allowed_profile_ids=visualization_profiles)

    # ---- Jicamarca EDP + ISR 真值（Sep 5, 05/10/15/20 UT）----
    _jic_record = None
    if config.get('include_isr_overlay', False):
        try:
            import datetime as _dt
            _isr_eval_dir = os.path.join(current_dir, 'isr_evaluation')
            if _isr_eval_dir not in sys.path:
                sys.path.insert(0, _isr_eval_dir)
            from isr_loader import load_jicamarca as _load_jic
            _JIC_DIR = r'D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)'
            _START_UNX = _dt.datetime(
                2024, 9, 1, tzinfo=_dt.timezone.utc).timestamp()
            _END_UNX = _dt.datetime(
                2024, 10, 1, tzinfo=_dt.timezone.utc).timestamp()
            if os.path.isdir(_JIC_DIR):
                _isr_recs = _load_jic(_JIC_DIR, _START_UNX, _END_UNX)
                _jic_record = next(
                    (r for r in _isr_recs
                     if r['date_str'] == '20240905'), None)
                if _jic_record is None:
                    print('[EDP] 未找到 Jicamarca 20240905 记录，ISR 叠绘跳过')
            else:
                print(f'[EDP] Jicamarca 数据目录不存在: {_JIC_DIR}，ISR 叠绘跳过')
        except Exception as _e:
            print(f'[EDP] Jicamarca 数据加载失败: {_e}')
    else:
        print('[EDP] 模型内优化模式：未读取ISR叠绘数据')

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
        cosmic_nb_index=batch_processor.cosmic_nb_index,
        allowed_profile_ids=visualization_profiles)


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
    parser.add_argument('--distance-localization',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='enable continuous physical localization')
    parser.add_argument('--background-seed', default=argparse.SUPPRESS,
                        help='shared Background checkpoint for Analysis-only runs')
    parser.add_argument('--basis-dim', type=int, default=None,
                        help='low-dimensional ETKF state size')
    parser.add_argument('--enkf-members', type=int, default=None,
                        help='ETKF ensemble member count')
    parser.add_argument(
        '--anomaly-parameterization',
        choices=('legacy_independent', 'orthogonal_factor'),
        default=None,
        help='ensemble anomaly generator (v7 legacy or v8 covariance factor)')
    parser.add_argument(
        '--density-basis-semantics',
        choices=('query_conditioned', 'coordinate_local_symmetric',
                 'endpoint_context_symmetric'),
        default=None,
        help='legacy, coordinate-only, or endpoint-context symmetric basis')
    parser.add_argument('--train-profile-fraction', type=float, default=None,
                        help='deterministic fraction of training profiles to use')
    parser.add_argument('--profile-subset-manifest', default=None,
                        help='shared JSON profile subset manifest for screen runs')
    parser.add_argument(
        '--date-blocked-split', action='store_true', default=None,
        help='use a shared UTC-date train/development/locked-test manifest')
    parser.add_argument(
        '--date-split-manifest', default=None,
        help='path to the deterministic UTC-date split manifest')
    parser.add_argument(
        '--qc-data', action=argparse.BooleanOptionalAction, default=True,
        help='use audited FY/COSMIC NPY+NPZ QC products')
    parser.add_argument(
        '--train-only', action='store_true',
        help='stop after strict-loading the best checkpoint')
    parser.add_argument(
        '--background-only', action='store_true',
        help='run or resume only the QC-v2 Background stage; never enter Analysis')
    parser.add_argument(
        '--background-trust-gate', action=argparse.BooleanOptionalAction,
        default=None,
        help='enable the fixed low-altitude nighttime Background trust gate')
    parser.add_argument(
        '--model-domain', choices=tuple(_CLI_DOMAIN_SPECS), default=None,
        help=('M2-W domain preset: strict 200-500 km, or 120-500 km model '
              'with 200-500 km satellite observations'))
    parser.add_argument('--smoke', action='store_true',
                        help='strict v14 1+1 epoch preflight; not eligible for evaluation')
    parser.add_argument(
        '--covariance-moment', action='store_true', default=None,
        help='train Analysis with profile-balanced covariance moment matching')
    parser.add_argument(
        '--empirical-covariance-loss', action='store_true', default=None,
        help='match density-basis covariance to frozen train-only cells')
    parser.add_argument(
        '--covariance-gradient-target', type=float, default=None,
        help='target empirical-covariance/observation gradient ratio')
    parser.add_argument(
        '--observation-gram-loss', action='store_true', default=None,
        help='train Analysis with production-precision HX Gram whitening')
    parser.add_argument(
        '--gram-gradient-target', type=float, default=None,
        help='target HX Gram/observation gradient ratio')
    parser.add_argument(
        '--direction-loss', action='store_true', default=None,
        help='penalize exact M10/M01/M11 increments opposite to target residuals')
    parser.add_argument(
        '--source-mode-schedule',
        choices=(
            'random_profile', 'deterministic_112',
            'balanced_profile_112'),
        default=None,
        help=(
            'random profiles, repeating M10/M01/M11/M11 batches, or '
            'profile-balanced M10/M01/M11/M11 across Analysis epochs'))
    parser.add_argument(
        '--background-epochs', type=int, default=None,
        help='number of Background epochs (default: configured value)')
    parser.add_argument(
        '--analysis-epochs', type=int, default=None,
        help='number of Analysis epochs (default: configured value)')
    parser.add_argument(
        '--max-train-batches', type=int, default=None,
        help='diagnostic-only cap on batches per epoch')
    parser.add_argument(
        '--max-validation-batches', type=int, default=None,
        help='diagnostic-only cap per development source')
    parser.add_argument(
        '--w-time-analysis', type=float, default=None,
        help='Analysis temporal second-difference weight')
    parser.add_argument(
        '--analysis-active-only-loss', action='store_true', default=None,
        help='exclude zero-precision M00 queries from the Analysis profile loss')
    parser.add_argument(
        '--analysis-exact-mode-loss', action='store_true', default=None,
        help='optimize exact profile-balanced M10/M01/M11 losses each batch')
    parser.add_argument(
        '--representativeness-kernel', default=None,
        help='train-only empirical cells for continuous precision weighting')
    parser.add_argument(
        '--include-isr-overlay', action='store_true',
        help='final-only: load ISR for profile overlays after model freeze')
    args = parser.parse_args()
    run_dir = _run_directory(args.run_name)
    if not args.eval_only and not args.resume and run_dir.exists():
        raise FileExistsError(
            'new training requires a nonexistent run directory; '
            f'use a new --run-name: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'training.log'
    stale_preflight_log = (
        log_path.exists()
        and not (run_dir / 'run_manifest.json').exists()
        and not (run_dir / 'best_background_model.pth').exists()
        and not (run_dir / 'best_fsia_model.pth').exists())
    log_mode = 'a' if args.eval_only or args.resume or stale_preflight_log else 'x'
    with log_path.open(log_mode, encoding='utf-8', buffering=1) as log_stream:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_stream)), \
                contextlib.redirect_stderr(_Tee(sys.stderr, log_stream)):
            main(
                eval_only=args.eval_only,
                resume_ckpt=args.resume,
                run_name=args.run_name,
                r_mode=args.r_mode,
                background_seed=getattr(args, 'background_seed', _UNSET),
                qc_data=args.qc_data,
                distance_localization=args.distance_localization,
                basis_dim=args.basis_dim,
                enkf_members=args.enkf_members,
                anomaly_parameterization=args.anomaly_parameterization,
                density_basis_semantics=args.density_basis_semantics,
                train_profile_fraction=args.train_profile_fraction,
                profile_subset_manifest=args.profile_subset_manifest,
                covariance_moment=args.covariance_moment,
                empirical_covariance_loss=args.empirical_covariance_loss,
                covariance_gradient_target=args.covariance_gradient_target,
                observation_gram_loss=args.observation_gram_loss,
                gram_gradient_target=args.gram_gradient_target,
                post_train_evaluation=not args.train_only,
                date_blocked_split=args.date_blocked_split,
                date_split_manifest=args.date_split_manifest,
                source_mode_schedule=args.source_mode_schedule,
                analysis_epochs=args.analysis_epochs,
                background_epochs=args.background_epochs,
                max_train_batches=args.max_train_batches,
                max_validation_batches=args.max_validation_batches,
                w_time_analysis=args.w_time_analysis,
                analysis_active_only_loss=args.analysis_active_only_loss,
                analysis_exact_mode_loss=args.analysis_exact_mode_loss,
                direction_loss=args.direction_loss,
                representativeness_kernel=args.representativeness_kernel,
                background_only=args.background_only,
                include_isr_overlay=args.include_isr_overlay,
                background_trust_gate=args.background_trust_gate,
                model_domain=args.model_domain,
                smoke_run=args.smoke,
            )
