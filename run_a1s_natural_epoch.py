"""Run the authorized fresh A1-S natural-epoch Background experiment."""

import copy
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'inr_modules'))

from inr_modules.mdia.train_fsia import train_fsia


REFERENCE_RUN = ROOT / 'checkpoints_fsia' / (
    'run67-m2w-120-500-background-stratum-bias-a1s')
HISTORICAL_RUN = REFERENCE_RUN
OUTPUT_RUN = ROOT / 'checkpoints_fsia' / (
    'run67-m2w-120-500-background-a1s-natural-epoch-r2')
DATE_SPLIT = ROOT / 'm00_residual_outputs' / 'a1s_train_qd' / (
    'a1s_date_split_manifest_20day.json')


def build_config(output_run=OUTPUT_RUN, reference_run=REFERENCE_RUN,
                 seed=42, overrides=None, date_split_manifest=DATE_SPLIT):
    with (reference_run / 'run_manifest.json').open(encoding='utf-8') as stream:
        config = copy.deepcopy(json.load(stream)['config'])
    existing = ({path.name for path in output_run.iterdir()}
                if output_run.exists() else set())
    if existing - {'date_split_manifest.json'}:
        raise FileExistsError(f'refusing to overwrite {output_run}')
    if not date_split_manifest.is_file():
        raise FileNotFoundError(date_split_manifest)

    config.update({
        'save_dir': str(output_run),
        'date_split_manifest': str(date_split_manifest),
        'date_split_semantics': 'a1s_native_date_split_v1',
        'use_date_blocked_split': True,
        'development_days': 5,
        'locked_test_days': 5,
        'background_only': True,
        'background_epochs': 5,
        'analysis_epochs': 10,
        'background_training_scope': 'full_background_v1',
        'background_training_semantics': 'a1s_natural_epoch_train_only_v1',
        'background_sampling_semantics': 'source_complete_profile_pass_v1',
        'background_checkpoint_selection': 'm00_natural_rmse_protected_v1',
        'background_selection_baseline': None,
        'm00_selection_reference_history': str(
            HISTORICAL_RUN / 'fsia_training_history.json'),
        'm00_selection_reference_epoch': 3,
        'background_fixed_updates_per_epoch': None,
        'background_profile_schedule_seed': int(seed),
        'background_profile_schedule_seed_fy': int(seed),
        'background_profile_schedule_seed_cosmic': int(seed) + 1,
        'background_condition_adapter': 'disabled',
        'background_decoder_conditioning': 'legacy_v1',
        'background_bias_grouping': 'height_daynight_v1',
        'background_bias_magnetic_min_profiles': 30,
        'background_bias_magnetic_min_batch_profiles': 8,
        'background_bias_magnetic_weight': 0.5,
        'background_bias_magnetic_daynight_labels': None,
        'background_bias_magnetic_support_counts': None,
        'background_bias_magnetic_support_semantics': None,
        'background_seed_ckpt': None,
        'resume_ckpt': None,
        'checkpoint_semantic': 'a1s_natural_epoch_background_v1',
        'm00_observation_calibration_semantics': 'disabled',
        'm00_observation_training_scope': 'disabled',
        'm00_candidate_semantics': 'disabled',
        'm00_representativeness_enabled': False,
        'm00_localization_enabled': False,
        'm00_observation_reliability_semantics': 'disabled',
        'm00_observation_bias_correction_semantics': 'disabled',
        'skip_hash_verification': True,
        'historical_a1s_reference_checkpoint': str(
            HISTORICAL_RUN / 'epoch_03_background_model.pth'),
        'historical_a1s_reference_semantics': 'a1s_formal_epoch3_v1',
    })
    for key in (
            'm00_observation_qd_sidecar_fy',
            'm00_observation_qd_sidecar_cosmic',
            'm00_observation_magnetic_sidecar_fy',
            'm00_observation_magnetic_sidecar_cosmic'):
        config.pop(key, None)
    config.pop('input_data_sha256', None)
    # Recalibrate this train-only weight for every fresh experiment.
    config.pop('resolved_background_stratum_bias_weight', None)
    config.pop('background_stratum_bias_gradient_calibration', None)
    config['seed'] = int(seed)
    if overrides:
        config.update(overrides)
    return config


def run_config(config, output_run, experiment='A1-S M00 natural epoch'):
    output_run.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1,
        'experiment': experiment,
        'semantics': {
            'background_epoch_semantics': 'source_complete_profile_pass_v1',
            'background_source_schedule': (
                'proportional_interleave_no_repeat_v1'),
            'background_fixed_updates_per_epoch': None,
            'date_split': 'a1s_native_date_split_v1',
            'sha256': 'not_computed_or_verified',
        },
        'config': config,
    }
    frozen_inputs = config.get('night_v2_frozen_input_sha256')
    if frozen_inputs is not None:
        if not isinstance(frozen_inputs, dict) or not frozen_inputs:
            raise ValueError('night-v2 requires nonempty frozen input hashes')
        identities = {}
        for key, value in config.items():
            if isinstance(value, str) and Path(value).is_file():
                path = str(Path(value).resolve())
                if path in frozen_inputs:
                    identities[key] = {'path': path, 'sha256': frozen_inputs[path]}
        config['skip_hash_verification'] = False
        config['input_data_sha256'] = {
            key: identity['sha256'] for key, identity in identities.items()}
        manifest['data_identity'] = identities
        manifest['semantics']['sha256'] = 'current_inputs_frozen_and_verified_v2'
    with (output_run / 'run_manifest.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    started = time.perf_counter()
    status = 'completed'
    try:
        train_fsia(config)
    except Exception:
        status = 'failed'
        raise
    finally:
        (output_run / 'run_timing.json').write_text(
            json.dumps({
                'status': status,
                'elapsed_seconds': time.perf_counter() - started,
                'semantics': 'wall_clock_train_fsia_v1',
            }, indent=2) + '\n',
            encoding='utf-8')


def main():
    config = build_config()
    run_config(config, OUTPUT_RUN)


if __name__ == '__main__':
    main()
