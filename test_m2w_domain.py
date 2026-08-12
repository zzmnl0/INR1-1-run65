import hashlib
import json

import numpy as np
import pytest

from evaluate_giro_peak import _predict_peaks
from inr_modules.data_managers.FY_dataloader import (
    FY3D_Dataset,
    FYNeighborhoodIndex,
    get_dataloaders,
)
from inr_modules.mdia.evaluation_stats import paired_group_bootstrap
from inr_modules.mdia.checkpoint_io import load_fsia_analysis_checkpoint
from inr_modules.mdia.train_fsia import (
    _architecture_signature,
    _development_selection_key,
    _low_altitude_anchor_batch,
    _validate_hybrid_low_altitude_protocol,
)
from isr_evaluation.main_isr_eval import (
    _analysis_common_mask,
    _candidate_baseline_common_mask,
)
from isr_evaluation.metrics import extract_grid_nmf2_hmf2
from select_m2w_background import select_background


def _write_profiles(tmp_path):
    rows = np.asarray([
        [0, 0, 199.9, 72, 10, 1],
        [0, 0, 200.0, 72, 10, 1],
        [0, 0, 250.0, 72, 10, 2],
        [0, 0, 300.0, 72, 10, 2],
        [0, 0, 500.0, 72, 10, 3],
        [0, 0, 500.1, 72, 10, 3],
    ], dtype=np.float32)
    data_path = tmp_path / 'profiles.npy'
    np.save(data_path, rows)
    index_path = tmp_path / 'profiles_index.npz'
    np.savez(index_path,
             profile_id=np.asarray([1, 2, 3]),
             pass_profile=np.ones(3, dtype=bool),
             output_start=np.asarray([0, 2, 4]),
             output_end=np.asarray([2, 4, 6]))
    return data_path, index_path


def test_strict_domain_boundaries_and_stable_development_sampling(tmp_path):
    data_path, index_path = _write_profiles(tmp_path)
    dataset = FY3D_Dataset(
        str(data_path), mode='train', val_days=[], val_ratio=None,
        profile_index_path=str(index_path), alt_range=(200.0, 500.0))
    assert np.array_equal(dataset.data[dataset.selected_indices, 2],
                          [200.0, 250.0, 300.0, 500.0])
    train, development = get_dataloaders(
        str(data_path), batch_size=16, val_ratio=0.34,
        profile_index_path=str(index_path), alt_range=(200.0, 500.0),
        full_validation_profiles=False)
    assert train.batch_sampler.points_per_profile == 8
    assert development.batch_sampler.points_per_profile == 8


def test_exact_tokens_respect_domain_and_profile_whitelist(tmp_path):
    data_path, index_path = _write_profiles(tmp_path)
    index = FYNeighborhoodIndex(str(data_path), {
        'fy_profile_index_path': str(index_path),
        'alt_range': (200.0, 500.0),
        'neighbor_directory_semantics': 'token_exact_positive_support_v1',
        'physical_localization_space_km': 1800.0,
        'physical_localization_time_hours': 1.5,
    })
    assert np.all((index.token_coords[:, 2] >= 200.0)
                  & (index.token_coords[:, 2] <= 500.0))
    payload = index.query_observation_batch(
        np.asarray([[0, 0, 275, 72]], dtype=np.float32),
        allowed_profile_ids=np.asarray([2]))
    assert set(payload['profile_id'].tolist()) == {2}


def test_v13_signature_and_lexicographic_selection():
    signature = _architecture_signature({
        'alt_range': (200.0, 500.0),
        'model_domain_semantics': 'strict_200_500_domain_v1',
    })
    assert signature['alt_range'] == [200.0, 500.0]
    assert signature['model_domain_semantics'] == 'strict_200_500_domain_v1'
    config = {'checkpoint_selection_semantics':
              'mean_ccc_then_rmse_then_pearson_v1'}
    better_ccc = _development_selection_key(
        {'ccc': 0.8, 'rmse': 2.0, 'pearson_r': 0.1}, config)
    lower_ccc = _development_selection_key(
        {'ccc': 0.7, 'rmse': 0.1, 'pearson_r': 0.99}, config)
    assert tuple(better_ccc) > tuple(lower_ccc)
    assert _development_selection_key(
        {'ccc': np.nan, 'rmse': 1.0, 'pearson_r': 1.0}, config) is None


def test_analysis_loader_rejects_background_and_wrong_v13_domain(tmp_path):
    checkpoint = tmp_path / 'best_fsia_model.pth'
    checkpoint.write_bytes(b'not-a-checkpoint')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path / 'run_manifest.json').write_text(json.dumps({
        'config': {'model_domain_semantics': 'strict_200_500_domain_v1',
                   'alt_range': [120.0, 500.0]}}), encoding='utf-8')
    summary = {'completed_stage': 'background',
               'checkpoint_sha256': digest}
    (tmp_path / 'training_summary.json').write_text(
        json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError, match='completed_stage=analysis'):
        load_fsia_analysis_checkpoint(checkpoint)
    summary['completed_stage'] = 'analysis'
    (tmp_path / 'training_summary.json').write_text(
        json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError, match='contract mismatch'):
        load_fsia_analysis_checkpoint(checkpoint)


def test_analysis_loader_historical_epoch_requires_opt_in(tmp_path):
    checkpoint = tmp_path / 'epoch_12_model.pth'
    checkpoint.write_bytes(b'historical-analysis-checkpoint')
    (tmp_path / 'run_manifest.json').write_text(json.dumps({
        'config': {'model_domain_semantics': 'strict_200_500_domain_v1',
                   'alt_range': [120.0, 500.0]}}), encoding='utf-8')
    (tmp_path / 'training_summary.json').write_text(json.dumps({
        'completed_stage': 'analysis', 'checkpoint_sha256': 'best-model-sha'}),
        encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256'):
        load_fsia_analysis_checkpoint(checkpoint)
    with pytest.raises(ValueError, match='contract mismatch'):
        load_fsia_analysis_checkpoint(checkpoint, allow_historical_epoch=True)


def test_isr_pair_mask_does_not_depend_on_background():
    observation = np.asarray([10.0, 11.0, 12.0])
    analysis = np.asarray([10.0, np.nan, 12.0])
    raw_iri = np.asarray([10.0, 11.0, 12.0])
    expected = np.asarray([True, False, True])
    assert np.array_equal(
        _analysis_common_mask(observation, analysis, raw_iri), expected)


def test_giro_fields_choose_independent_peaks(monkeypatch):
    centers = {'M11': 250.0, 'M00': 300.0, 'IRI': 350.0}

    def fake_query(coords, *args, **kwargs):
        altitude = coords[:, 2]
        return {name: -(altitude - center) ** 2
                for name, center in centers.items()}

    monkeypatch.setattr('evaluate_giro_peak._query_fields', fake_query)
    records = np.asarray([[0, 0, 72, 300]], dtype=np.float32)
    peaks = _predict_peaks(
        records, None, (None, None, None, None), {}, None,
        (200.0, 500.0))
    assert {name: float(values['hmf2'][0])
            for name, values in peaks.items()} == centers


def test_grouped_bootstrap_is_deterministic_and_ordered():
    observation = np.arange(12, dtype=np.float64)
    units = np.repeat(np.arange(6), 2)
    baseline = observation + 2.0
    passed = paired_group_bootstrap(
        observation, observation, baseline, units, replicates=200, seed=42)
    repeated = paired_group_bootstrap(
        observation, observation, baseline, units, replicates=200, seed=42)
    failed = paired_group_bootstrap(
        observation, baseline, observation, units, replicates=200, seed=42)
    uncertain = paired_group_bootstrap(
        observation, baseline, baseline, units, replicates=200, seed=42)
    assert passed == repeated and passed['decision'] == 'pass'
    assert failed['decision'] == 'fail'
    assert uncertain['decision'] == 'inconclusive'


def test_background_candidate_tie_prefers_gate_off(tmp_path):
    run_dirs = []
    for enabled in (False, True):
        run_dir = tmp_path / f'gate-{enabled}'
        run_dir.mkdir()
        checkpoint = run_dir / 'best_background_model.pth'
        checkpoint.write_bytes(b'finite-placeholder')
        summary = {
            'completed_stage': 'background',
            'checkpoint_format_version': 13,
            'model_domain_semantics': 'strict_200_500_domain_v1',
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': hashlib.sha256(
                checkpoint.read_bytes()).hexdigest(),
            'background_development_gate_passed': True,
            'background_trust_gate_enabled': enabled,
            'background_development': {
                'fy_ccc': 0.5, 'cosmic_ccc': 0.5,
                'ccc': 0.5, 'rmse': 0.2, 'pearson_r': 0.6,
                'fy_pearson_r': 0.6, 'cosmic_pearson_r': 0.6,
            },
        }
        (run_dir / 'training_summary.json').write_text(
            json.dumps(summary), encoding='utf-8')
        run_dirs.append(run_dir)
    result = select_background(run_dirs)
    assert result['winner']['trust_gate_enabled'] is False


def _v14_config(**overrides):
    config = {
        'model_domain_semantics': 'hybrid_120_500_model_200_500_observation_v1',
        'checkpoint_format_version': 14,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (200.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
        'low_altitude_prior_range': (120.0, 200.0),
        'low_altitude_prior_semantics':
        'soft_iri_background_zero_analysis_increment_v1',
        'low_altitude_anchor_levels_km': tuple(range(120, 200, 10)),
        'low_altitude_anchor_profiles_per_source': 16,
        'w_low_altitude_background_iri': 0.02,
        'w_low_altitude_analysis_increment': 0.01,
        'low_altitude_gradient_ratio_max': 0.25,
        'background_epochs': 5,
        'analysis_epochs': 10,
        'seed': 42,
        'basis_dim': 64,
        'enkf_n_members': 8,
        'r_mode': 'global',
        'background_trust_gate_enabled': False,
    }
    config.update(overrides)
    return config


def test_v14_domain_contract_and_observation_tokens(tmp_path):
    _validate_hybrid_low_altitude_protocol(_v14_config())
    with pytest.raises(ValueError, match='contract mismatch'):
        _validate_hybrid_low_altitude_protocol(
            _v14_config(observation_alt_range=(120.0, 500.0)))
    data_path, index_path = _write_profiles(tmp_path)
    index = FYNeighborhoodIndex(str(data_path), {
        **_v14_config(), 'fy_profile_index_path': str(index_path),
        'neighbor_directory_semantics': 'token_exact_positive_support_v1',
        'physical_localization_space_km': 1800.0,
        'physical_localization_time_hours': 1.5,
    })
    assert np.all((index.token_coords[:, 2] >= 200.0)
                  & (index.token_coords[:, 2] <= 500.0))
    strict = FY3D_Dataset(
        str(data_path), mode='train', val_days=[], val_ratio=None,
        profile_index_path=str(index_path), alt_range=(200.0, 500.0))
    hybrid_observation = FY3D_Dataset(
        str(data_path), mode='train', val_days=[], val_ratio=None,
        profile_index_path=str(index_path),
        alt_range=_v14_config()['observation_alt_range'])
    assert np.array_equal(strict.profile_ids, hybrid_observation.profile_ids)
    assert np.array_equal(strict.data[strict.selected_indices],
                          hybrid_observation.data[hybrid_observation.selected_indices])


def test_v14_anchors_are_deterministic_bounded_and_low_only():
    config = _v14_config()
    coords = np.asarray([
        [1, 2, 250, 10], [3, 4, 300, 20], [5, 6, 400, 30],
        [7, 8, 450, 40], [1, 2, 260, 11],
    ], dtype=np.float32)
    import torch
    coordinate_tensor = torch.from_numpy(coords)
    profiles = torch.tensor([9, 8, 7, 6, 9])
    anchors, groups = _low_altitude_anchor_batch(
        coordinate_tensor, profiles, config)
    repeated, repeated_groups = _low_altitude_anchor_batch(
        coordinate_tensor, profiles, config)
    assert torch.equal(anchors, repeated) and torch.equal(groups, repeated_groups)
    assert len(torch.unique(groups)) == 4
    assert len(anchors) == 4 * 8
    assert set(anchors[:, 2].tolist()) == set(range(120, 200, 10))


def test_v14_dual_peak_excludes_150_km_peak():
    altitude = np.arange(120.0, 501.0, 1.0)
    field = np.maximum(
        20.0 - ((altitude - 150.0) / 5.0) ** 2,
        12.0 - ((altitude - 310.0) / 8.0) ** 2,
    )[:, None]
    nmf2, hmf2 = extract_grid_nmf2_hmf2(field, altitude, (200.0, 500.0))
    assert np.isfinite(nmf2[0]) and hmf2[0] == 310.0


def test_candidate_baseline_mask_requires_both_models_and_200_500():
    observation = np.asarray([1.0, 2.0, 3.0, 4.0])
    candidate = np.asarray([1.0, 2.0, np.nan, 4.0])
    baseline = np.asarray([1.0, np.nan, 3.0, 4.0])
    altitude = np.asarray([199.9, 250.0, 300.0, 500.0])
    assert np.array_equal(
        _candidate_baseline_common_mask(
            observation, candidate, baseline, altitude),
        np.asarray([False, False, False, True]),
    )
