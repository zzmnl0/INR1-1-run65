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
from inr_modules.mdia.m00_observation import validate_observation_config
from inr_modules.mdia.v14_contract import (
    A2S1_DOMAIN_SEMANTICS,
    canonical_a2s1_training_protocol,
    canonical_v14_training_protocol,
)
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
_A2S1_DOMAIN_SEMANTICS = A2S1_DOMAIN_SEMANTICS
_M2W_HYBRID_RUN_SEMANTICS = (
    'M2-W_continuous_physical_local_letkf_120_500_obs_200_500')
_M2W_HYBRID_BACKGROUND_SEMANTICS = (
    'qc_v2_date_blocked_train_only_m2w_120_500_obs_200_500_low_alt_prior_v1')
_A2S1_RUN_SEMANTICS = 'A2S1_a1s_legacy_120_500_low_prior_v1'
_A2S1_BACKGROUND_SEMANTICS = (
    'qc_v2_date_blocked_train_only_a2s1_120_500_low_alt_prior_v1')
_PROFILE_SPLIT_SEMANTICS = 'source_time_local_block80_dev20_v1'
_A1S6_DATE_SPLIT_SEMANTICS = 'a1s_train_plus_locked_dev5_v1'
_A1S_NATIVE_DATE_SPLIT_SEMANTICS = 'a1s_native_date_split_v1'
_A1S2_BACKGROUND_SEMANTICS = (
    'qc_v2_source_time_local_block80_dev20_train_only_v1')
_A1S7_BACKGROUND_SEMANTICS = 'a1s7_m00_b_shared_dual_maglat_v1'
_A1S7_CHECKPOINT_SEMANTICS = 'a1s7_m00_b_v1'
_A1S7_ADAPTER_SEMANTICS = 'maglat2_smooth_v1'
_A1S7_ADAPTER_ALIAS = 'dual_maglat_smooth_v1'
_A1S7_ADAPTER_SCOPE = 'condition_adapter_only_dual_maglat_v1'
_A1S7_GATE_SEMANTICS = 'abs_sinI_025_050_smoothstep_v1'
_A1S7_SAMPLING_SEMANTICS = 'source_date_physical_stratum_balanced_v1'
_A1S8_BACKGROUND_SEMANTICS = 'a1s8_m00_preserve_mixed_gate_v1'
_A1S8_CHECKPOINT_SEMANTICS = 'a1s8_m00_preserve_mixed_gate_v1'
_A1S8_ADAPTER_SEMANTICS = _A1S7_ADAPTER_SEMANTICS
_A1S8_ADAPTER_SCOPE = _A1S7_ADAPTER_SCOPE
_A1S8_GATE_SEMANTICS = _A1S7_GATE_SEMANTICS
_A1S8_SAMPLING_SEMANTICS = 'a1s8_mixed_gate_aware_v1'
_A1S8_GATE_BALANCE_SEMANTICS = 'effective_gate_mass_clipped_v1'
_A1S8_ANCHOR_SEMANTICS = 'a1s_output_trust_region_v1'
_A1S8_SOURCE_MIX_SEMANTICS = 'fy3_cosmic1_v1'
_A1S9_BACKGROUND_SEMANTICS = 'a1s9_m00_strict_baseline_v1'
_A1S9_CHECKPOINT_SEMANTICS = 'a1s9_m00_strict_baseline_v1'
_M00_OBSERVATION_SCOPE = 'm00_observation_error_only_v1'
_M00_OBSERVATION_BACKGROUND_SEMANTICS = _M00_OBSERVATION_SCOPE
_JOINT_FRESH_SCOPE = 'condition_adapter_joint_fresh_v1'
_JOINT_FRESH_SELECTOR = 'cond_adapter_joint_fresh_bias_constrained_v1'
_CONDITIONAL_SHAPE_LOSS = 'conditional_profile_centered_scale_v1'
_CONDITIONAL_SHAPE_SELECTOR = 'conditional_shape_constrained_v1'
_RAW_IRI_BACKGROUND_SEMANTICS = 'raw_iri_fixed_v1'
_ANALYSIS_ONLY_SCOPE = 'analysis_only_v1'
_RAW_IRI_R_SOURCE = 'raw_iri_train_only_recalibrated_v1'
_CAUSAL_HISTORY_SCOPE = 'causal_history_only_v1'
_CAUSAL_HISTORY_OUTPUT = 'causal_context_background_v1'
_CAUSAL_HISTORY_TIME_SUPPORT = 'causal_past_lag_0p5_1p5_v1'
_CAUSAL_FROZEN_R_SOURCE = 'a1s2_frozen_global_v1'
_A1S_FROZEN_R_SOURCE = 'a1s_frozen_global_v1'
_CAUSAL_BACKGROUND_ANCHORS = ('a1s2_m00_v1', 'a1s_m00_v1')
_LOW_ALTITUDE_PRIOR_SEMANTICS = (
    'soft_iri_background_zero_analysis_increment_v1')
_HYBRID_LOW_ALTITUDE_LEVELS = tuple(float(value) for value in range(120, 200, 10))
_M2W_V14_DATE_SPLIT_SHA256 = (
    '0576d83c72cbcfd5994e3afcb192429f965bf423a937f88a1db26eb09e21dbab')
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
    _A2S1_DOMAIN_SEMANTICS: {
        'checkpoint_format_version': 15,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (120.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
        'run_semantics': _A2S1_RUN_SEMANTICS,
    },
}
_CLI_DOMAIN_SPECS = {
    '200-500': _M2W_DOMAIN_SEMANTICS,
    '120-500-obs-200-500': _M2W_HYBRID_DOMAIN_SEMANTICS,
    'a2s1-120-500': _A2S1_DOMAIN_SEMANTICS,
}
_UNSET = object()
_OLD_RUN65_CKPT = Path(current_dir) / 'checkpoints_fsia' / 'run65' / 'best_fsia_model.pth'


def _run_directory(run_name):
    if Path(run_name).name != run_name or run_name in ('', '.', '..'):
        raise ValueError('run_name must be one directory name')
    return Path(current_dir) / 'checkpoints_fsia' / run_name


def _is_transient_log_scaffold(run_dir):
    """Allow retrying a run stopped before it wrote any experiment artifact."""
    return (run_dir.is_dir()
            and (run_dir / 'training.log').is_file()
            and {path.name for path in run_dir.iterdir()} == {'training.log'})


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


def _file_identity(path, include_hash=True):
    path = Path(path)
    stat = path.stat()
    identity = {'path': str(path.resolve()), 'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns}
    if include_hash:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        identity['sha256'] = digest.hexdigest()
    return identity


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
        'representativeness_kernel_path', 'date_split_manifest',
        'm00_observation_magnetic_sidecar_fy',
        'm00_observation_magnetic_sidecar_cosmic')
        if config.get(key) and os.path.isfile(config[key])
    ]
    skip_hashes = bool(config.get('skip_hash_verification', False))
    data_identity = {
        key: _file_identity(config[key], include_hash=not skip_hashes)
        for key in data_keys}
    config['input_data_sha256'] = (
        None if skip_hashes else {
            key: identity['sha256']
            for key, identity in data_identity.items()})
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': config,
        'data_identity': data_identity,
        'code_identity': None if skip_hashes else _code_identity(current_dir),
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
        'checkpoint': _file_identity(
            config['resume_ckpt'],
            include_hash=not config.get('skip_hash_verification', False)),
        'code_identity': (
            None if config.get('skip_hash_verification', False)
            else _code_identity(current_dir)),
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
         background_condition_adapter=None,
         background_adapter_seed_checkpoint=None,
         background_adapter_rng_seed=None,
         background_training_scope=None,
         background_adapter_residual_cap=None,
         distance_localization=True, basis_dim=None, enkf_members=None,
         anomaly_parameterization=None, density_basis_semantics=None,
         train_profile_fraction=None,
         profile_subset_manifest=None, covariance_moment=None,
         empirical_covariance_loss=None,
         covariance_gradient_target=None,
         observation_gram_loss=None, gram_gradient_target=None,
         direction_loss=None,
        post_train_evaluation=True, date_blocked_split=None,
        date_split_manifest=None, date_split_semantics=None,
        profile_split_semantics=None,
         profile_split_path=None, source_mode_schedule=None,
         analysis_epochs=None, background_epochs=None, max_train_batches=None,
         max_validation_batches=None,
         w_time_analysis=None, analysis_active_only_loss=None,
         analysis_exact_mode_loss=None, representativeness_kernel=None,
         background_only=False,
         include_isr_overlay=False, background_trust_gate=None,
         background_stratum_bias_gradient_target=None,
         background_checkpoint_selection=None,
         background_selection_baseline=None,
         background_selection_baseline_checkpoint=None,
         background_shape_loss=None, background_shape_gradient_target=None,
         background_shape_calibration_batches=None,
         background_shape_min_profiles_per_cell=None,
         background_semantics=None, background_residual_enabled=None,
    background_adapter_gate_semantics=None,
    background_sampling_semantics=None,
    background_fixed_updates_per_epoch=None,
    background_anchor_semantics=None,
    background_gate_balance_semantics=None,
    background_source_mix_schedule=None,
    background_mix_legacy_fraction=None,
    background_mix_balanced_fraction=None,
    background_anchor_gradient_target=None,
    background_anchor_calibration_batches=None,
    background_anchor_warmup_epochs=None,
    background_exposure_weight_cap=None,
    background_gradient_ratio_min=None,
    background_gradient_ratio_max=None,
    background_low_effective_mass_min=None,
    background_low_effective_mass_max=None,
    background_high_effective_mass_min=None,
    background_high_effective_mass_max=None,
    analysis_seed_background=None, analysis_training_scope=None,
         analysis_output_semantics=None,
    analysis_observation_time_support=None, causal_background_anchor=None,
    r_source=None,
         r_fy_fixed=None, r_cosmic_fixed=None,
         model_domain=None, smoke_run=False, skip_hash_verification=False,
         m00_observation_calibration_semantics=None,
         m00_observation_bias_path=None, m00_observation_r_path=None,
         m00_observation_seed_checkpoint=None,
         m00_observation_stage=None, m00_observation_weight_cap=None,
         m00_observation_training_scope=None,
         m00_representativeness_enabled=None,
         m00_localization_enabled=None):
    # ==================== 加载配置 ====================
    config = get_config_mdia()
    run_dir = _run_directory(run_name)
    prior_config = {}
    manifest_path = run_dir / 'run_manifest.json'
    if (eval_only or resume_ckpt is not None) and manifest_path.is_file():
        with manifest_path.open(encoding='utf-8') as stream:
            prior_config = json.load(stream).get('config', {})
    if background_condition_adapter is None:
        background_condition_adapter = prior_config.get(
            'background_condition_adapter',
            config.get('background_condition_adapter', 'disabled'))
    if m00_observation_calibration_semantics is None:
        m00_observation_calibration_semantics = prior_config.get(
            'm00_observation_calibration_semantics',
            config.get('m00_observation_calibration_semantics', 'disabled'))
    if m00_observation_bias_path is None:
        m00_observation_bias_path = prior_config.get(
            'm00_observation_bias_path',
            config.get('m00_observation_bias_path'))
    if m00_observation_r_path is None:
        m00_observation_r_path = prior_config.get(
            'm00_observation_r_path',
            config.get('m00_observation_r_path'))
    if m00_observation_seed_checkpoint is None:
        m00_observation_seed_checkpoint = prior_config.get(
            'm00_observation_seed_checkpoint',
            config.get('m00_observation_seed_checkpoint'))
    if m00_observation_stage is None:
        m00_observation_stage = prior_config.get(
            'm00_observation_stage', config.get('m00_observation_stage', 'bias_r'))
    if m00_observation_weight_cap is None:
        m00_observation_weight_cap = prior_config.get(
            'm00_observation_weight_cap',
            config.get('m00_observation_weight_cap', 4.0))
    if m00_observation_training_scope is None:
        m00_observation_training_scope = prior_config.get(
            'm00_observation_training_scope',
            config.get('m00_observation_training_scope', 'disabled'))
    if m00_representativeness_enabled is None:
        m00_representativeness_enabled = prior_config.get(
            'm00_representativeness_enabled',
            config.get('m00_representativeness_enabled', False))
    if m00_localization_enabled is None:
        m00_localization_enabled = prior_config.get(
            'm00_localization_enabled', config.get('m00_localization_enabled', False))

    m00_observation_route = str(
        m00_observation_calibration_semantics).replace('-', '_') != 'disabled'
    if m00_observation_route:
        # The route is a Background-only fine-tune of the selected A1-S M00.
        # --background-seed remains a shorthand, while the explicit field is
        # persisted so resume cannot silently change the route.
        if m00_observation_seed_checkpoint is None and background_seed is not _UNSET:
            m00_observation_seed_checkpoint = background_seed
        if not m00_observation_seed_checkpoint:
            raise ValueError(
                'M00 observation-error route requires the A1-S epoch-3 '
                'Background seed checkpoint')
        m00_observation_seed_checkpoint = str(m00_observation_seed_checkpoint)
        background_seed = m00_observation_seed_checkpoint
        background_condition_adapter = 'disabled'
        background_training_scope = _M00_OBSERVATION_SCOPE
        background_only = True
    if background_adapter_seed_checkpoint is None:
        background_adapter_seed_checkpoint = prior_config.get(
            'background_adapter_seed_checkpoint',
            config.get('background_adapter_seed_checkpoint'))
    if background_adapter_rng_seed is None:
        background_adapter_rng_seed = prior_config.get(
            'background_adapter_rng_seed',
            config.get('background_adapter_rng_seed', 42))
    if background_training_scope is None:
        background_training_scope = prior_config.get(
            'background_training_scope',
            config.get('background_training_scope', 'full_background_v1'))
    if background_adapter_residual_cap is None:
        background_adapter_residual_cap = float(prior_config.get(
            'background_adapter_residual_cap',
            config.get('background_adapter_residual_cap', 0.1)))
    if background_anchor_semantics is None:
        background_anchor_semantics = prior_config.get(
            'background_anchor_semantics',
            config.get('background_anchor_semantics', 'disabled'))
    if background_gate_balance_semantics is None:
        background_gate_balance_semantics = prior_config.get(
            'background_gate_balance_semantics',
            config.get('background_gate_balance_semantics', 'disabled'))
    if background_source_mix_schedule is None:
        background_source_mix_schedule = prior_config.get(
            'background_source_mix_schedule',
            config.get('background_source_mix_schedule', 'equal_v1'))
    if background_mix_legacy_fraction is None:
        background_mix_legacy_fraction = prior_config.get(
            'background_mix_legacy_fraction',
            config.get('background_mix_legacy_fraction', 1.0))
    if background_mix_balanced_fraction is None:
        background_mix_balanced_fraction = prior_config.get(
            'background_mix_balanced_fraction',
            config.get('background_mix_balanced_fraction', 0.0))
    if background_anchor_gradient_target is None:
        background_anchor_gradient_target = prior_config.get(
            'background_anchor_gradient_target',
            config.get('background_anchor_gradient_target', 0.10))
    if background_anchor_calibration_batches is None:
        background_anchor_calibration_batches = prior_config.get(
            'background_anchor_calibration_batches',
            config.get('background_anchor_calibration_batches', 20))
    if background_anchor_warmup_epochs is None:
        background_anchor_warmup_epochs = prior_config.get(
            'background_anchor_warmup_epochs',
            config.get('background_anchor_warmup_epochs', 1))
    if background_exposure_weight_cap is None:
        background_exposure_weight_cap = prior_config.get(
            'background_exposure_weight_cap',
            config.get('background_exposure_weight_cap', 3.0))
    if background_gradient_ratio_min is None:
        background_gradient_ratio_min = prior_config.get(
            'background_gradient_ratio_min',
            config.get('background_gradient_ratio_min', 0.25))
    if background_gradient_ratio_max is None:
        background_gradient_ratio_max = prior_config.get(
            'background_gradient_ratio_max',
            config.get('background_gradient_ratio_max', 4.0))
    if background_low_effective_mass_min is None:
        background_low_effective_mass_min = prior_config.get(
            'background_low_effective_mass_min',
            config.get('background_low_effective_mass_min', 0.25))
    if background_low_effective_mass_max is None:
        background_low_effective_mass_max = prior_config.get(
            'background_low_effective_mass_max',
            config.get('background_low_effective_mass_max', 0.35))
    if background_high_effective_mass_min is None:
        background_high_effective_mass_min = prior_config.get(
            'background_high_effective_mass_min',
            config.get('background_high_effective_mass_min', 0.65))
    if background_high_effective_mass_max is None:
        background_high_effective_mass_max = prior_config.get(
            'background_high_effective_mass_max',
            config.get('background_high_effective_mass_max', 0.75))

    background_condition_adapter = str(background_condition_adapter).replace('-', '_')
    if background_condition_adapter == _A1S7_ADAPTER_ALIAS:
        background_condition_adapter = _A1S7_ADAPTER_SEMANTICS
    background_training_scope = str(background_training_scope).replace('-', '_')
    background_anchor_semantics = str(background_anchor_semantics).replace('-', '_')
    background_gate_balance_semantics = str(background_gate_balance_semantics).replace('-', '_')
    background_source_mix_schedule = str(background_source_mix_schedule).replace('-', '_')
    if background_semantics is None:
        background_semantics = prior_config.get(
            'background_semantics', config.get('background_semantics', 'learned_v1'))
    background_semantics = str(background_semantics).replace('-', '_')
    if str(m00_observation_calibration_semantics) != 'disabled':
        m00_observation_training_scope = 'm00_observation_error_only_v1'
        background_only = True
    a1s8_requested = (
        background_semantics == _A1S8_BACKGROUND_SEMANTICS
        or background_anchor_semantics == _A1S8_ANCHOR_SEMANTICS
        or background_gate_balance_semantics == _A1S8_GATE_BALANCE_SEMANTICS
        or background_source_mix_schedule == _A1S8_SOURCE_MIX_SEMANTICS
        or background_sampling_semantics == _A1S8_SAMPLING_SEMANTICS)
    a1s9_requested = (background_semantics == _A1S9_BACKGROUND_SEMANTICS)
    if background_residual_enabled is None:
        background_residual_enabled = prior_config.get(
            'background_residual_enabled', config.get('background_residual_enabled', True))
    background_residual_enabled = bool(background_residual_enabled)
    if analysis_seed_background is None:
        analysis_seed_background = prior_config.get(
            'analysis_seed_background', config.get('analysis_seed_background'))
    if r_source is None:
        r_source = prior_config.get('r_source', config.get('r_source', 'checkpoint'))
    r_source = str(r_source).replace('-', '_')
    if analysis_training_scope is None:
        analysis_training_scope = prior_config.get(
            'analysis_training_scope',
            config.get('analysis_training_scope', 'full_analysis_v1'))
    if analysis_output_semantics is None:
        analysis_output_semantics = prior_config.get(
            'analysis_output_semantics',
            config.get('analysis_output_semantics', 'm11_feature_letkf_v1'))
    if analysis_observation_time_support is None:
        analysis_observation_time_support = prior_config.get(
            'analysis_observation_time_support',
            config.get('analysis_observation_time_support', 'symmetric_legacy_v1'))
    if causal_background_anchor is None:
        causal_background_anchor = prior_config.get(
            'causal_background_anchor',
            config.get('causal_background_anchor', 'a1s2_m00_v1'))
    analysis_training_scope = str(analysis_training_scope).replace('-', '_')
    analysis_output_semantics = str(analysis_output_semantics).replace('-', '_')
    analysis_observation_time_support = str(
        analysis_observation_time_support).replace('-', '_')
    causal_background_anchor = str(causal_background_anchor).replace('-', '_')
    causal_history_analysis_only = (
        analysis_training_scope == _CAUSAL_HISTORY_SCOPE)
    if causal_history_analysis_only:
        if causal_background_anchor not in _CAUSAL_BACKGROUND_ANCHORS:
            raise ValueError(
                'causal history requires causal_background_anchor in '
                f'{_CAUSAL_BACKGROUND_ANCHORS!r}')
        required = {
            'analysis_output_semantics': (
                analysis_output_semantics, _CAUSAL_HISTORY_OUTPUT),
            'analysis_observation_time_support': (
                analysis_observation_time_support, _CAUSAL_HISTORY_TIME_SUPPORT),
        'r_source': (r_source, (_CAUSAL_FROZEN_R_SOURCE,
                                _A1S_FROZEN_R_SOURCE)),
        }
        mismatched = {
            key: actual for key, (actual, expected) in required.items()
            if (actual not in expected if isinstance(expected, tuple)
                else actual != expected)}
        if mismatched:
            raise ValueError(
                f'causal history Analysis configuration mismatch: {mismatched}')
        if not analysis_seed_background:
            raise ValueError(
                'causal_history_only_v1 requires an A1-S Background checkpoint')
        # A causal-history run normally starts from a frozen Background seed,
        # but a persisted Analysis state is safe to resume: it restores the
        # same model/R/split contract and does not re-enter Background.
        if background_seed is not _UNSET or background_only:
            raise ValueError(
                'causal_history_only_v1 rejects Background seed/resume/background-only')
        r_fy_fixed = 0.0339414440 if r_fy_fixed is None else float(r_fy_fixed)
        r_cosmic_fixed = (
            0.0218319837 if r_cosmic_fixed is None else float(r_cosmic_fixed))
    raw_iri_analysis_only = background_semantics == _RAW_IRI_BACKGROUND_SEMANTICS
    if raw_iri_analysis_only:
        if background_training_scope != _ANALYSIS_ONLY_SCOPE:
            raise ValueError('raw_iri_fixed_v1 requires analysis_only_v1 scope')
        if background_residual_enabled:
            raise ValueError('raw_iri_fixed_v1 requires background residual disabled')
        if not analysis_seed_background:
            raise ValueError('raw_iri_fixed_v1 requires an A1-S2 analysis seed checkpoint')
        if background_seed is not _UNSET:
            raise ValueError(
                'raw_iri_fixed_v1 cannot combine with --background-seed')
        if r_source != _RAW_IRI_R_SOURCE:
            raise ValueError('raw_iri_fixed_v1 requires train-only Raw-IRI R calibration')
        if background_only:
            raise ValueError('raw_iri_fixed_v1 is Analysis-only, not Background-only')
    elif background_training_scope == _ANALYSIS_ONLY_SCOPE:
        raise ValueError('analysis_only_v1 scope requires raw_iri_fixed_v1 semantics')
    if background_condition_adapter not in (
            'disabled', 'lt_signed_abs_sini_periodic_lon_v1',
            _A1S7_ADAPTER_SEMANTICS):
        raise ValueError(
            f'unsupported Background condition-adapter semantics: {background_condition_adapter}')
    if a1s9_requested:
        if background_condition_adapter != _A1S7_ADAPTER_SEMANTICS:
            raise ValueError('A1-S9 requires dual magnetic-latitude adapters')
        if background_training_scope != _A1S7_ADAPTER_SCOPE:
            raise ValueError('A1-S9 requires adapter-only Background scope')
        background_adapter_gate_semantics = _A1S7_GATE_SEMANTICS
        background_sampling_semantics = 'legacy_profile_time_bin_v1'
        background_anchor_semantics = _A1S8_ANCHOR_SEMANTICS
        background_anchor_gradient_target = 0.20
        background_adapter_residual_cap = 0.05
        background_fixed_updates_per_epoch = 960
        background_training_semantics = _A1S9_BACKGROUND_SEMANTICS
        background_checkpoint_selection = 'bias_constrained_v1'
        if not background_adapter_seed_checkpoint:
            raise ValueError('A1-S9 requires the A1-S epoch-3 Background seed')
    elif a1s8_requested:
        if background_condition_adapter != _A1S8_ADAPTER_SEMANTICS:
            raise ValueError('A1-S8 requires dual magnetic-latitude adapters')
        if background_training_scope != _A1S8_ADAPTER_SCOPE:
            raise ValueError('A1-S8 requires condition_adapter_only_dual_maglat_v1')
        if background_adapter_gate_semantics is None:
            background_adapter_gate_semantics = _A1S8_GATE_SEMANTICS
        else:
            background_adapter_gate_semantics = str(
                background_adapter_gate_semantics).replace('-', '_')
        if background_sampling_semantics is None:
            background_sampling_semantics = _A1S8_SAMPLING_SEMANTICS
        else:
            background_sampling_semantics = str(
                background_sampling_semantics).replace('-', '_')
        if background_adapter_gate_semantics != _A1S8_GATE_SEMANTICS:
            raise ValueError('A1-S8 gate semantics mismatch')
        if background_sampling_semantics != _A1S8_SAMPLING_SEMANTICS:
            raise ValueError('A1-S8 sampling semantics mismatch')
        background_anchor_semantics = _A1S8_ANCHOR_SEMANTICS
        background_gate_balance_semantics = _A1S8_GATE_BALANCE_SEMANTICS
        background_source_mix_schedule = _A1S8_SOURCE_MIX_SEMANTICS
        background_training_semantics = _A1S8_BACKGROUND_SEMANTICS
    if (background_condition_adapter == _A1S7_ADAPTER_SEMANTICS
            and not a1s8_requested and not a1s9_requested):
        if background_training_scope != _A1S7_ADAPTER_SCOPE:
            raise ValueError(
                'A1-S7 adapter requires condition_adapter_only_dual_maglat_v1')
        background_adapter_gate_semantics = (
            _A1S7_GATE_SEMANTICS if background_adapter_gate_semantics is None
            else str(background_adapter_gate_semantics).replace('-', '_'))
        background_sampling_semantics = (
            _A1S7_SAMPLING_SEMANTICS if background_sampling_semantics is None
            else str(background_sampling_semantics).replace('-', '_'))
        if background_adapter_gate_semantics != _A1S7_GATE_SEMANTICS:
            raise ValueError('A1-S7 gate semantics mismatch')
        if background_sampling_semantics != _A1S7_SAMPLING_SEMANTICS:
            raise ValueError('A1-S7 sampling semantics mismatch')
        background_training_semantics = _A1S7_BACKGROUND_SEMANTICS
    elif background_adapter_gate_semantics is None:
        background_adapter_gate_semantics = prior_config.get(
            'background_adapter_gate_semantics',
            config.get('background_adapter_gate_semantics'))
    if background_sampling_semantics is None:
        background_sampling_semantics = prior_config.get(
            'background_sampling_semantics',
            config.get('background_sampling_semantics'))
    if background_fixed_updates_per_epoch is None:
        background_fixed_updates_per_epoch = prior_config.get(
            'background_fixed_updates_per_epoch',
            config.get('background_fixed_updates_per_epoch'))
    if a1s8_requested:
        if background_fixed_updates_per_epoch is None:
            background_fixed_updates_per_epoch = 960
        if int(background_fixed_updates_per_epoch) != 960:
            raise ValueError('A1-S8 requires exactly 960 Background updates/epoch')
        if not background_only:
            raise ValueError('A1-S8 is Background-only; Analysis/M11 is deferred')
    if background_training_scope not in (
        'full_background_v1', 'condition_adapter_only_v1', _JOINT_FRESH_SCOPE,
            _ANALYSIS_ONLY_SCOPE, _A1S7_ADAPTER_SCOPE,
            _M00_OBSERVATION_SCOPE):
        raise ValueError(
            f'unsupported Background training scope: {background_training_scope}')
    if background_training_scope == _JOINT_FRESH_SCOPE:
        if background_condition_adapter != 'lt_signed_abs_sini_periodic_lon_v1':
            raise ValueError(
                'Joint-Fresh requires matching condition-adapter and training-scope semantics')
        if background_seed is not _UNSET or background_adapter_seed_checkpoint or resume_ckpt:
            raise ValueError(
                'Joint-Fresh cannot use --background-seed, adapter seed checkpoint, or --resume')
        if not isinstance(background_adapter_rng_seed, int) or background_adapter_rng_seed < 0:
            raise ValueError(
                'Joint-Fresh background_adapter_rng_seed must be a non-negative integer')
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
    a2s1_domain = model_domain_semantics == _A2S1_DOMAIN_SEMANTICS
    strict_low_prior_domain = hybrid_domain or a2s1_domain
    training_protocol = (
        canonical_a2s1_training_protocol(smoke_run) if a2s1_domain
        else canonical_v14_training_protocol(smoke_run)
        if hybrid_domain else {})
    if a2s1_domain and background_sampling_semantics is None:
        background_sampling_semantics = 'legacy_profile_time_bin_v1'
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
    if date_split_semantics is None:
        date_split_semantics = prior_config.get(
            'date_split_semantics', config.get('date_split_semantics'))
    if date_split_semantics is not None:
        date_split_semantics = str(date_split_semantics).replace('-', '_')
        if date_split_semantics not in (_A1S6_DATE_SPLIT_SEMANTICS,
                                        _A1S_NATIVE_DATE_SPLIT_SEMANTICS):
            raise ValueError(
                f'unsupported date-split semantics: {date_split_semantics}')
    if profile_split_semantics is None:
        profile_split_semantics = prior_config.get(
            'profile_split_semantics',
            config.get('profile_split_semantics'))
    if profile_split_semantics is not None:
        profile_split_semantics = str(profile_split_semantics).replace('-', '_')
        if profile_split_semantics != _PROFILE_SPLIT_SEMANTICS:
            raise ValueError(
                f'unsupported profile-split semantics: {profile_split_semantics}')
    if profile_split_path is None:
        profile_split_path = prior_config.get(
            'profile_split_path', config.get('profile_split_path'))
    if profile_split_semantics == _PROFILE_SPLIT_SEMANTICS:
        if date_blocked_split is True:
            raise ValueError(
                'profile split cannot be combined with --date-blocked-split')
        # A1-S2 supersedes the legacy UTC-date train/development/locked-test
        # contract; the explicit profile manifest is the only split source.
        date_blocked_split = False
        date_split_manifest = None
    if date_split_semantics == _A1S6_DATE_SPLIT_SEMANTICS:
        if profile_split_semantics is not None:
            raise ValueError('A1-S6 date split cannot be combined with profile split')
        date_blocked_split = True
        if not date_split_manifest:
            date_split_manifest = str(Path('isr_validation_outputs/run66-modelonly-m0') / 'date_split_manifest.json')
        if not profile_split_path:
            profile_split_path = str(run_dir / 'profile_split.npz')
    date_blocked_split = bool(
        prior_config.get('use_date_blocked_split',
                         config.get('use_date_blocked_split', True))
        if date_blocked_split is None else date_blocked_split)
    if date_split_manifest is None:
        date_split_manifest = prior_config.get(
            'date_split_manifest', config.get('date_split_manifest'))
    if date_blocked_split and not date_split_manifest:
        date_split_manifest = str(run_dir / 'date_split_manifest.json')
    if profile_split_semantics == _PROFILE_SPLIT_SEMANTICS:
        # Do not let the legacy default date manifest leak into an A1-S2 run.
        date_blocked_split = False
        date_split_manifest = None
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
    background_stratum_bias_gradient_target = float(
        prior_config.get(
            'background_stratum_bias_gradient_target',
            (0.05 if background_condition_adapter == _A1S7_ADAPTER_SEMANTICS
             else config.get('background_stratum_bias_gradient_target', 0.0)))
        if background_stratum_bias_gradient_target is None
        else background_stratum_bias_gradient_target)
    if not 0.0 <= background_stratum_bias_gradient_target <= 1.0:
        raise ValueError(
            'background_stratum_bias_gradient_target must be in [0, 1]')
    background_stratum_bias_calibration_batches = int(
        prior_config.get(
            'background_stratum_bias_calibration_batches',
            config.get('background_stratum_bias_calibration_batches', 20)))
    if background_stratum_bias_calibration_batches < 1:
        raise ValueError(
            'background_stratum_bias_calibration_batches must be positive')
    background_checkpoint_selection = (
        prior_config.get(
            'background_checkpoint_selection',
            config.get('background_checkpoint_selection', 'legacy'))
        if background_checkpoint_selection is None
        else str(background_checkpoint_selection).replace('-', '_'))
    if background_checkpoint_selection not in (
        'legacy', 'bias_constrained_v1',
        'cond_adapter_bias_constrained_v1', _JOINT_FRESH_SELECTOR,
        _CONDITIONAL_SHAPE_SELECTOR):
        raise ValueError('unsupported Background checkpoint selection')
    if background_selection_baseline is None:
        background_selection_baseline = prior_config.get(
            'background_selection_baseline',
            config.get('background_selection_baseline'))
    if background_selection_baseline_checkpoint is None:
        background_selection_baseline_checkpoint = prior_config.get(
            'background_selection_baseline_checkpoint',
            config.get('background_selection_baseline_checkpoint'))
    if (background_checkpoint_selection in (
            'bias_constrained_v1', 'cond_adapter_bias_constrained_v1',
            _JOINT_FRESH_SELECTOR)
            and not background_selection_baseline):
        raise ValueError(
            'Background checkpoint selection requires --background-selection-baseline')
    if (background_checkpoint_selection == _CONDITIONAL_SHAPE_SELECTOR
            and not (background_selection_baseline
                     or background_selection_baseline_checkpoint)):
        raise ValueError(
            'conditional-shape selection requires a baseline replay or checkpoint')
    background_shape_loss = str(
        prior_config.get('background_shape_loss', 'disabled')
        if background_shape_loss is None else background_shape_loss).replace('-', '_')
    if background_shape_loss not in ('disabled', _CONDITIONAL_SHAPE_LOSS):
        raise ValueError('unsupported Background shape loss')
    background_shape_gradient_target = float(
        prior_config.get('background_shape_gradient_target', 0.02)
        if background_shape_gradient_target is None
        else background_shape_gradient_target)
    background_shape_calibration_batches = int(
        prior_config.get('background_shape_calibration_batches', 20)
        if background_shape_calibration_batches is None
        else background_shape_calibration_batches)
    background_shape_min_profiles_per_cell = int(
        prior_config.get('background_shape_min_profiles_per_cell', 8)
        if background_shape_min_profiles_per_cell is None
        else background_shape_min_profiles_per_cell)
    if (background_shape_loss == _CONDITIONAL_SHAPE_LOSS
            and background_checkpoint_selection != _CONDITIONAL_SHAPE_SELECTOR):
        raise ValueError('conditional shape loss requires its constrained selector')
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
    if raw_iri_analysis_only and background_epochs != 5:
        raise ValueError(
            'raw_iri_fixed_v1 requires A1-S2 background_epochs=5 bookkeeping')
    validate_observation_config(config)
    if smoke_run:
        if max_train_batches not in (None, 1) or max_validation_batches not in (None, 1):
            raise ValueError('v14 smoke runs require one train and one validation batch')
        max_train_batches = 1
        max_validation_batches = 1
    if smoke_run and (background_epochs, analysis_epochs) != (1, 1):
        raise ValueError('v14 smoke runs require exactly 1 Background and 1 Analysis epoch')
    if strict_low_prior_domain and not smoke_run and (background_epochs, analysis_epochs) != (5, 10):
        raise ValueError('strict low-prior full training requires exactly 5 Background and 10 Analysis epochs')
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
    if strict_low_prior_domain and background_trust_gate:
        raise ValueError('strict low-prior training requires gate-off')
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
        if ((m2w_domain or background_trust_gate)
                and not m00_observation_route
                or prior_background_semantics == 'qc_v2_date_blocked_train_only'
                or (not prior_config and run_name == _DEFAULT_RUN_NAME)):
            raise ValueError(
                'QC-v2 Background run cannot be combined with an external seed')
        background_training_semantics = 'frozen_external_seed'
    else:
        if not background_trust_gate:
            background_training_semantics = (
                _A1S9_BACKGROUND_SEMANTICS if a1s9_requested else
                _A1S8_BACKGROUND_SEMANTICS if a1s8_requested else
                _A1S7_BACKGROUND_SEMANTICS
                if background_condition_adapter == _A1S7_ADAPTER_SEMANTICS
                else prior_background_semantics
                or (_A2S1_BACKGROUND_SEMANTICS if a2s1_domain else
                    _M2W_HYBRID_BACKGROUND_SEMANTICS if hybrid_domain
                    else _M2W_BACKGROUND_SEMANTICS if m2w_domain
                    else config.get('background_training_semantics',
                                    'qc_v2_date_blocked_train_only')))

    if (profile_split_semantics == _PROFILE_SPLIT_SEMANTICS
            and background_seed is None):
        background_training_semantics = _A1S2_BACKGROUND_SEMANTICS
    if raw_iri_analysis_only:
        background_training_semantics = _RAW_IRI_BACKGROUND_SEMANTICS

    # The M00 observation-error route is explicitly hash-free by contract.
    if str(m00_observation_calibration_semantics) != 'disabled':
        skip_hash_verification = True
        background_training_semantics = _M00_OBSERVATION_BACKGROUND_SEMANTICS

    # FSIA 检查点目录（每次新训练实验递增 run 编号）
    update_config_mdia(
        save_dir=str(run_dir),
        resume_ckpt=None,
        eval_only=False,
        r_mode=r_mode,
        use_distance_localization=distance_localization,
        background_seed_ckpt=background_seed,
        m00_observation_seed_checkpoint=m00_observation_seed_checkpoint,
        background_condition_adapter=background_condition_adapter,
        background_adapter_seed_checkpoint=background_adapter_seed_checkpoint,
        background_adapter_rng_seed=background_adapter_rng_seed,
        background_training_scope=background_training_scope,
            background_adapter_residual_cap=background_adapter_residual_cap,
        background_adapter_gate_semantics=background_adapter_gate_semantics,
        background_sampling_semantics=background_sampling_semantics,
        background_fixed_updates_per_epoch=background_fixed_updates_per_epoch,
        background_anchor_semantics=background_anchor_semantics,
        background_gate_balance_semantics=background_gate_balance_semantics,
        background_source_mix_schedule=background_source_mix_schedule,
        background_mix_legacy_fraction=float(background_mix_legacy_fraction),
        background_mix_balanced_fraction=float(background_mix_balanced_fraction),
        background_anchor_gradient_target=float(background_anchor_gradient_target),
        background_anchor_calibration_batches=int(background_anchor_calibration_batches),
        background_anchor_warmup_epochs=int(background_anchor_warmup_epochs),
        background_exposure_weight_cap=float(background_exposure_weight_cap),
        background_gradient_ratio_min=float(background_gradient_ratio_min),
        background_gradient_ratio_max=float(background_gradient_ratio_max),
        background_low_effective_mass_min=float(background_low_effective_mass_min),
        background_low_effective_mass_max=float(background_low_effective_mass_max),
        background_high_effective_mass_min=float(background_high_effective_mass_min),
        background_high_effective_mass_max=float(background_high_effective_mass_max),
        background_semantics=background_semantics,
        background_residual_enabled=background_residual_enabled,
        analysis_seed_background=analysis_seed_background,
        analysis_training_scope=analysis_training_scope,
        analysis_output_semantics=analysis_output_semantics,
        analysis_observation_time_support=analysis_observation_time_support,
        causal_background_anchor=causal_background_anchor,
        r_source=r_source,
        r_fy_fixed=r_fy_fixed,
        r_cosmic_fixed=r_cosmic_fixed,
        background_training_semantics=background_training_semantics,
        checkpoint_semantic=(
            _A1S9_CHECKPOINT_SEMANTICS if a1s9_requested else
            _A1S8_CHECKPOINT_SEMANTICS if a1s8_requested else
            _A1S7_CHECKPOINT_SEMANTICS
            if background_condition_adapter == _A1S7_ADAPTER_SEMANTICS
            else prior_config.get(
                'checkpoint_semantic', config.get('checkpoint_semantic', 'a1s_legacy_v1'))),
        background_only=bool(background_only),
        smoke_run=bool(smoke_run),
        skip_hash_verification=bool(skip_hash_verification),
        m00_observation_calibration_semantics=(
            m00_observation_calibration_semantics),
        m00_observation_bias_path=m00_observation_bias_path,
        m00_observation_r_path=m00_observation_r_path,
        m00_observation_stage=m00_observation_stage,
        m00_observation_weight_cap=float(m00_observation_weight_cap),
        m00_observation_training_scope=m00_observation_training_scope,
        m00_representativeness_enabled=bool(m00_representativeness_enabled),
        m00_localization_enabled=bool(m00_localization_enabled),
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
        date_split_semantics=date_split_semantics,
        profile_split_semantics=profile_split_semantics,
        profile_split_path=profile_split_path,
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
        background_stratum_bias_gradient_target=(
            background_stratum_bias_gradient_target),
        background_stratum_bias_calibration_batches=(
            background_stratum_bias_calibration_batches),
        background_checkpoint_selection=background_checkpoint_selection,
        background_selection_baseline=background_selection_baseline,
        background_selection_baseline_checkpoint=(
            background_selection_baseline_checkpoint),
        background_shape_loss=background_shape_loss,
        background_shape_gradient_target=background_shape_gradient_target,
        background_shape_calibration_batches=background_shape_calibration_batches,
        background_shape_min_profiles_per_cell=(
            background_shape_min_profiles_per_cell),
        model_domain_semantics=model_domain_semantics,
        alt_range=alt_range,
        observation_alt_range=observation_alt_range,
        peak_search_alt_range=peak_search_alt_range,
        low_altitude_prior_range=(120.0, 200.0) if strict_low_prior_domain else None,
        low_altitude_prior_semantics=(
            _LOW_ALTITUDE_PRIOR_SEMANTICS if strict_low_prior_domain else None),
        low_altitude_anchor_levels_km=(
            _HYBRID_LOW_ALTITUDE_LEVELS if strict_low_prior_domain else ()),
        low_altitude_anchor_profiles_per_source=(16 if strict_low_prior_domain else 0),
        w_low_altitude_background_iri=(0.02 if strict_low_prior_domain else 0.0),
        w_low_altitude_analysis_increment=(0.01 if strict_low_prior_domain else 0.0),
        low_altitude_gradient_ratio_max=(0.25 if strict_low_prior_domain else 0.25),
        smoke_auxiliary_gradient_ratio_max=(0.25 if strict_low_prior_domain else 0.30),
        **{key: training_protocol[key] for key in (
            'training_protocol_revision',
            'gram_weight_resolution_semantics',
            'gradient_parameter_scope',
            'calibration_precision',
            'gram_require_all_batches_finite_positive',
            'gram_preflight_ratio_bounds',
            'gram_coverage_policy',
            'low_altitude_anchor_selection',
            'low_altitude_anchor_grouping',
            'low_altitude_neighbor_profile_semantics',
        ) if key in training_protocol},
        **({'background_source_quota_semantics': 'legacy_cycle_shorter_v1',
            'background_profile_schedule_version': 'legacy_v1'}
           if a2s1_domain else {}),
        training_protocol=(training_protocol if strict_low_prior_domain else None),
        checkpoint_format_version=checkpoint_format_version,
        run_semantics=run_semantics,
        checkpoint_selection_semantics=(
            'mean_ccc_then_rmse_then_pearson_v1' if m2w_domain
            else prior_config.get('checkpoint_selection_semantics',
                                  'mean_profile_rmse_v1')),
    )
    if smoke_run:
        update_config_mdia(r_calibration_batches=1)
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
            expected_stage = 'background' if background_only else 'analysis'
            recorded_stage = training_summary.get('completed_stage')
            compatible_background_replay = (
                background_only and recorded_stage == 'analysis'
                and (run_dir / 'best_background_model.pth').is_file())
            if (recorded_stage != expected_stage
                    and not compatible_background_replay):
                raise ValueError(
                    f'eval-only requires a completed {expected_stage} checkpoint')
        checkpoint_name = (
            'best_background_model.pth' if background_only
            else 'best_fsia_model.pth')
        update_config_mdia(
            eval_only=True,
            resume_ckpt=os.path.join(config['save_dir'], checkpoint_name))
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
    if date_blocked_split:
        required_paths.append('date_split_manifest')
    if qc_data:
        required_paths.extend(['fy_qc_report_path', 'cosmic_qc_report_path'])
    if representativeness_kernel:
        required_paths.append('representativeness_kernel_path')
    if m00_observation_route:
        required_paths.append('m00_observation_seed_checkpoint')
        if m00_observation_bias_path:
            required_paths.append('m00_observation_bias_path')
        if m00_observation_r_path:
            required_paths.append('m00_observation_r_path')
    if raw_iri_analysis_only or causal_history_analysis_only:
        required_paths.append('analysis_seed_background')
    if background_checkpoint_selection in (
            'bias_constrained_v1', 'cond_adapter_bias_constrained_v1',
            _JOINT_FRESH_SELECTOR):
        required_paths.append('background_selection_baseline')
    if (background_checkpoint_selection == _CONDITIONAL_SHAPE_SELECTOR
            and not background_selection_baseline):
        required_paths.append('background_selection_baseline_checkpoint')
    missing = [k for k in required_paths
               if not config.get(k) or not os.path.exists(config[k])]
    if missing:
        print('\n以下数据文件缺失，无法继续训练:')
        for k in missing:
            print(f'  {k}: {config[k]}')
        return
    if (strict_low_prior_domain and profile_split_semantics != _PROFILE_SPLIT_SEMANTICS
            and not config.get('skip_hash_verification', False)):
        split_sha256 = _file_identity(config['date_split_manifest'])['sha256']
        if split_sha256 != _M2W_V14_DATE_SPLIT_SHA256:
            raise RuntimeError(
                'strict low-prior date-split manifest SHA256 differs from the approved '
                'train+development contract')
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
            if not config.get('skip_hash_verification', False):
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
        if strict_low_prior_domain:
            resume_state = torch.load(
                config['resume_ckpt'], map_location='cpu', weights_only=False)
            if isinstance(resume_state, dict) and resume_state.get('smoke_run', False):
                raise ValueError('v14 smoke training states are never resumable')
        _record_resume_manifest(config)
    elif not eval_only:
        # The CLI checked freshness before creating this one-file tee-log
        # scaffold.  Do not mistake that transient scaffold for a prior run.
        # A completed/partial hybrid run may be resumed from its persisted
        # state.  Fresh runs still require a new directory, but resume must
        # not be rejected merely because the directory already contains the
        # checkpoint and manifest being resumed.
        if strict_low_prior_domain and not is_resume and run_dir.exists() and not _is_transient_log_scaffold(run_dir):
            raise FileExistsError(
                'strict low-prior smoke/full training requires a previously nonexistent '
                f'target directory: {run_dir}')
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
    # Joint-Fresh may intentionally have no eligible Development epoch.  In
    # that case train_fsia records one finite diagnostic checkpoint instead
    # of fabricating a formal ``best_background_model.pth``.
    if background_only:
        summary_path = os.path.join(config['save_dir'], 'training_summary.json')
        if os.path.isfile(summary_path):
            with open(summary_path, encoding='utf-8') as stream:
                training_summary = json.load(stream)
            if (training_summary.get('diagnostic_only')
                    and training_summary.get('checkpoint')):
                best_ckpt = training_summary['checkpoint']
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
    parser.add_argument(
        '--background-condition-adapter',
        choices=('disabled', 'lt_signed_abs_sini_periodic_lon_v1',
                 _A1S7_ADAPTER_SEMANTICS, _A1S7_ADAPTER_ALIAS),
        default=None, help='explicit M00 condition-adapter semantics')
    parser.add_argument(
        '--background-adapter-seed-checkpoint', default=None,
        help='A1-S Background checkpoint used to seed an adapter-only run')
    parser.add_argument(
        '--background-adapter-rng-seed', type=int, default=None,
        help='independent seed for fresh condition-adapter initialization')
    parser.add_argument('--background-adapter-gate-semantics', default=None)
    parser.add_argument('--background-sampling-semantics', default=None)
    parser.add_argument('--background-fixed-updates-per-epoch', type=int,
                        default=None)
    parser.add_argument('--background-anchor-semantics', default=None)
    parser.add_argument('--background-gate-balance-semantics', default=None)
    parser.add_argument('--background-source-mix-schedule', default=None)
    parser.add_argument('--background-mix-legacy-fraction', type=float,
                        default=None)
    parser.add_argument('--background-mix-balanced-fraction', type=float,
                        default=None)
    parser.add_argument('--background-anchor-gradient-target', type=float,
                        default=None)
    parser.add_argument('--background-anchor-calibration-batches', type=int,
                        default=None)
    parser.add_argument('--background-anchor-warmup-epochs', type=int,
                        default=None)
    parser.add_argument('--background-exposure-weight-cap', type=float,
                        default=None)
    parser.add_argument('--background-gradient-ratio-min', type=float,
                        default=None)
    parser.add_argument('--background-gradient-ratio-max', type=float,
                        default=None)
    parser.add_argument('--background-low-effective-mass-min', type=float,
                        default=None)
    parser.add_argument('--background-low-effective-mass-max', type=float,
                        default=None)
    parser.add_argument('--background-high-effective-mass-min', type=float,
                        default=None)
    parser.add_argument('--background-high-effective-mass-max', type=float,
                        default=None)
    parser.add_argument(
        '--background-training-scope', default=None,
        help='Background parameter scope (e.g. condition_adapter_only_v1)')
    parser.add_argument(
        '--background-adapter-residual-cap', type=float, default=None,
        help='maximum absolute condition-adapter residual in dex')
    parser.add_argument(
        '--background-semantics', choices=('learned_v1', 'raw_iri_fixed_v1',
                                           _A1S9_BACKGROUND_SEMANTICS),
        default=None, help='explicit Background semantics')
    parser.add_argument(
        '--background-residual-enabled', action=argparse.BooleanOptionalAction,
        default=None, help='enable learned Background residual')
    parser.add_argument(
        '--analysis-seed-background', default=None,
        help='A1-S Background checkpoint for explicit Analysis-only runs')
    parser.add_argument(
        '--analysis-training-scope', default=None,
        help='Analysis parameter/time-support scope')
    parser.add_argument(
        '--analysis-output-semantics', default=None,
        help='meaning of the trained Analysis output')
    parser.add_argument(
        '--analysis-observation-time-support', default=None,
        help='observation token time-support semantics')
    parser.add_argument(
        '--causal-background-anchor', default=None,
        help='causal-history M00 anchor semantics (A1-S compatible)')
    parser.add_argument(
        '--r-source', default=None, help='observation-R calibration semantics')
    parser.add_argument('--r-fy-fixed', type=float, default=None)
    parser.add_argument('--r-cosmic-fixed', type=float, default=None)
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
        '--profile-split-semantics',
        choices=(_PROFILE_SPLIT_SEMANTICS,),
        default=None,
        help='fixed source-specific time-local profile split semantics')
    parser.add_argument(
        '--date-split-semantics',
        choices=(_A1S6_DATE_SPLIT_SEMANTICS,
                 _A1S_NATIVE_DATE_SPLIT_SEMANTICS),
        default=None,
        help='A1-S6 train-plus-former-locked date semantics')
    parser.add_argument(
        '--profile-split-path', default=None,
        help='path to the fixed A1-S2 profile split manifest')
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
        '--background-stratum-bias-gradient-target', type=float, default=None,
        help=('Background source-by-stratum mean-Bias gradient ratio target; '
              '0 disables the optional loss'))
    parser.add_argument(
        '--background-checkpoint-selection',
        choices=('legacy', 'bias-constrained-v1',
                 'cond-adapter-bias-constrained-v1',
                 'cond-adapter-joint-fresh-bias-constrained-v1',
                 'conditional-shape-constrained-v1'),
        default=None,
        help='Background checkpoint selector')
    parser.add_argument(
        '--background-selection-baseline', default=None,
        help='A0 background_development_replay.json used by bias-constrained-v1')
    parser.add_argument(
        '--background-selection-baseline-checkpoint', default=None,
        help='A1-S Background checkpoint replayed for conditional-shape selection')
    parser.add_argument(
        '--background-shape-loss',
        choices=('disabled', 'conditional-profile-centered-scale-v1'),
        default=None, help='conditional Background anomaly/spread loss semantics')
    parser.add_argument(
        '--background-shape-gradient-target', type=float, default=None,
        help='shape/observation gradient ratio target')
    parser.add_argument(
        '--background-shape-calibration-batches', type=int, default=None,
        help='fixed paired batches used for shape gradient calibration')
    parser.add_argument(
        '--background-shape-min-profiles-per-cell', type=int, default=None,
        help='minimum profiles required by a training shape cell')
    parser.add_argument(
        '--model-domain', choices=tuple(_CLI_DOMAIN_SPECS), default=None,
        help=('M2-W domain preset: strict 200-500 km, or 120-500 km model '
              'with 200-500 km satellite observations, or A2S1 120-500 km '
              'observations with a 200-500 km peak search'))
    parser.add_argument('--smoke', action='store_true',
                        help='strict v14 1+1 epoch preflight; not eligible for evaluation')
    parser.add_argument(
        '--skip-hash-verification', action='store_true',
        help='do not compute or verify input/checkpoint SHA256 for this run')
    parser.add_argument('--m00-observation-calibration-semantics',
                        default=None)
    parser.add_argument('--m00-observation-bias-path', default=None)
    parser.add_argument('--m00-observation-r-path', default=None)
    parser.add_argument(
        '--m00-observation-seed-checkpoint', default=None,
        help='A1-S epoch-3 Background checkpoint for the M00 observation route')
    parser.add_argument('--m00-observation-stage',
                        choices=('bias_r', 'representativeness_localization',
                                 'full'), default=None)
    parser.add_argument('--m00-observation-weight-cap', type=float,
                        default=None)
    parser.add_argument('--m00-representativeness-enabled',
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--m00-localization-enabled',
                        action=argparse.BooleanOptionalAction, default=None)
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
    if (not args.eval_only and not args.resume and run_dir.exists()
            and not _is_transient_log_scaffold(run_dir)):
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
            background_condition_adapter=args.background_condition_adapter,
            background_adapter_seed_checkpoint=
            args.background_adapter_seed_checkpoint,
                background_adapter_rng_seed=args.background_adapter_rng_seed,
                background_adapter_gate_semantics=args.background_adapter_gate_semantics,
                background_sampling_semantics=args.background_sampling_semantics,
                background_fixed_updates_per_epoch=args.background_fixed_updates_per_epoch,
                background_anchor_semantics=args.background_anchor_semantics,
                background_gate_balance_semantics=args.background_gate_balance_semantics,
                background_source_mix_schedule=args.background_source_mix_schedule,
                background_mix_legacy_fraction=args.background_mix_legacy_fraction,
                background_mix_balanced_fraction=args.background_mix_balanced_fraction,
                background_anchor_gradient_target=args.background_anchor_gradient_target,
                background_anchor_calibration_batches=args.background_anchor_calibration_batches,
                background_anchor_warmup_epochs=args.background_anchor_warmup_epochs,
                background_exposure_weight_cap=args.background_exposure_weight_cap,
                background_gradient_ratio_min=args.background_gradient_ratio_min,
                background_gradient_ratio_max=args.background_gradient_ratio_max,
                background_low_effective_mass_min=args.background_low_effective_mass_min,
                background_low_effective_mass_max=args.background_low_effective_mass_max,
                background_high_effective_mass_min=args.background_high_effective_mass_min,
                background_high_effective_mass_max=args.background_high_effective_mass_max,
                background_training_scope=args.background_training_scope,
            background_adapter_residual_cap=args.background_adapter_residual_cap,
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
                date_split_semantics=args.date_split_semantics,
                profile_split_semantics=args.profile_split_semantics,
                profile_split_path=args.profile_split_path,
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
                background_stratum_bias_gradient_target=(
                    args.background_stratum_bias_gradient_target),
                background_checkpoint_selection=(
                    args.background_checkpoint_selection),
            background_selection_baseline=(
                args.background_selection_baseline),
            background_selection_baseline_checkpoint=(
                args.background_selection_baseline_checkpoint),
            background_shape_loss=args.background_shape_loss,
            background_shape_gradient_target=(
                args.background_shape_gradient_target),
            background_shape_calibration_batches=(
                args.background_shape_calibration_batches),
            background_shape_min_profiles_per_cell=(
                args.background_shape_min_profiles_per_cell),
            background_semantics=args.background_semantics,
            background_residual_enabled=args.background_residual_enabled,
            analysis_seed_background=args.analysis_seed_background,
            analysis_training_scope=args.analysis_training_scope,
            analysis_output_semantics=args.analysis_output_semantics,
            analysis_observation_time_support=(
                args.analysis_observation_time_support),
            causal_background_anchor=args.causal_background_anchor,
            r_source=args.r_source,
            r_fy_fixed=args.r_fy_fixed,
            r_cosmic_fixed=args.r_cosmic_fixed,
            model_domain=args.model_domain,
                smoke_run=args.smoke,
                skip_hash_verification=args.skip_hash_verification,
                m00_observation_calibration_semantics=(
                    args.m00_observation_calibration_semantics),
                m00_observation_bias_path=args.m00_observation_bias_path,
                m00_observation_r_path=args.m00_observation_r_path,
                m00_observation_seed_checkpoint=(
                    args.m00_observation_seed_checkpoint),
                m00_observation_stage=args.m00_observation_stage,
                m00_observation_weight_cap=args.m00_observation_weight_cap,
                m00_representativeness_enabled=(
                    args.m00_representativeness_enabled),
                m00_localization_enabled=args.m00_localization_enabled,
            )
