import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diagnose_latent_covariance import _rank_metrics, _spectrum
from inr_modules.data_managers.FY_dataloader import (
    FY3D_Dataset,
    _profile_representative_days,
)
from inr_modules.mdia.fsia_model import NeuralETKFLayer
from inr_modules.mdia.train_fsia import (
    _apply_balanced_profile_modes,
    _apply_source_mode,
    _architecture_signature,
    _balanced_profile_mode_counts,
    _gradient_norms,
    _load_or_create_date_split,
    _query_observations,
    _restrict_training_profiles,
    _source_mode_for_batch,
)


def _fake_loader():
    profile_ids = np.repeat(np.arange(10), 2)
    groups = [np.array([2 * index, 2 * index + 1]) for index in range(10)]
    return SimpleNamespace(
        dataset=SimpleNamespace(
            profile_ids=profile_ids,
            npy_path='fake.npy',
        ),
        batch_sampler=SimpleNamespace(profiles_by_bin={0: groups}),
    )


def test_profile_subset_manifest_is_deterministic():
    manifest = Path(tempfile.mkdtemp()) / 'profiles.json'
    first = _fake_loader()
    summary = _restrict_training_profiles(
        first, 0.3, 42, 'FY', str(manifest))
    selected_first = [
        int(first.dataset.profile_ids[group[0]])
        for group in first.batch_sampler.profiles_by_bin[0]
    ]
    second = _fake_loader()
    _restrict_training_profiles(second, 0.3, 42, 'FY', str(manifest))
    selected_second = [
        int(second.dataset.profile_ids[group[0]])
        for group in second.batch_sampler.profiles_by_bin[0]
    ]
    assert summary == {'available': 10, 'selected': 3}
    assert selected_first == selected_second
    assert json.loads(manifest.read_text())['sources']['FY'][
        'selected_profiles'] == 3


def test_date_split_manifest_is_blocked_and_deterministic():
    manifest = Path(tempfile.mkdtemp()) / 'dates.json'
    config = {
        'use_date_blocked_split': True,
        'date_split_manifest': str(manifest),
        'total_hours': 720.0,
        'start_date_str': '2024-09-01 00:00:00',
        'development_days': 5,
        'locked_test_days': 5,
        'seed': 42,
    }
    first, first_identity = _load_or_create_date_split(config)
    first_bytes = manifest.read_bytes()
    second, second_identity = _load_or_create_date_split(config)
    assert first == second
    assert first_identity == second_identity
    assert first_bytes == manifest.read_bytes()
    assert {name: len(days) for name, days in first.items()} == {
        'train': 20, 'development': 5, 'locked_test': 5}
    combined = sum((days for days in first.values()), [])
    assert sorted(combined) == list(range(30))
    assert len(set(combined)) == 30
    for block in np.array_split(np.arange(30), 5):
        assert len(set(block) & set(first['development'])) == 1
        assert len(set(block) & set(first['locked_test'])) == 1

    with pytest.raises(ValueError, match='seed'):
        _load_or_create_date_split({**config, 'seed': 43})


def test_explicit_date_split_keeps_complete_profiles():
    days = _profile_representative_days(
        np.array([23.9, 24.1, 48.0, 49.0, 72.0, 73.0]),
        np.array([10, 10, 20, 20, 30, 30]))
    np.testing.assert_array_equal(days, [1, 1, 2, 2, 3, 3])

    root = Path(tempfile.mkdtemp())
    data_path = root / 'profiles.npy'
    data = np.array([
        [0, 0, 200, 23.9, 10, 10],
        [0, 0, 210, 24.1, 10, 10],
        [0, 0, 200, 48.0, 10, 20],
        [0, 0, 210, 49.0, 10, 20],
        [0, 0, 200, 72.0, 10, 30],
        [0, 0, 210, 73.0, 10, 30],
    ], dtype=np.float32)
    np.save(data_path, data)
    split = {'train': [1, 3], 'development': [2], 'locked_test': [0]}
    train = FY3D_Dataset(
        str(data_path), mode='train', val_ratio=None, split_days=split)
    development = FY3D_Dataset(
        str(data_path), mode='development', val_ratio=None, split_days=split)
    assert set(np.unique(train.profile_ids)) == {10, 30}
    assert set(np.unique(development.profile_ids)) == {20}
    assert not set(train.profile_ids) & set(development.profile_ids)


def test_observation_query_forwards_profile_allowlist_and_exclusion():
    class Index:
        def query_observation_batch(
                self, coords, exclude_profile_ids=None,
                allowed_profile_ids=None):
            self.exclude = exclude_profile_ids
            self.allowed = allowed_profile_ids
            return {
                'value': np.zeros((len(coords), 1), dtype=np.float32),
                'valid_mask': np.zeros((len(coords), 1), dtype=bool),
            }

    index = Index()
    coords = torch.zeros(2, 4)
    excluded = torch.tensor([3, 4])
    allowed = np.array([1, 2, 5], dtype=np.int64)
    _query_observations(index, coords, excluded, allowed)
    np.testing.assert_array_equal(index.exclude, [3, 4])
    np.testing.assert_array_equal(index.allowed, allowed)


def test_architecture_signature_and_dimension_validation():
    signature = _architecture_signature({
        'basis_dim': 128,
        'enkf_n_members': 16,
        'enkf_pert_hidden': 64,
    })
    assert signature['basis_dim'] == 128
    assert signature['enkf_n_members'] == 16
    assert signature['enkf_pert_hidden'] == 64
    assert signature['model_domain_semantics'] == 'legacy_120_500_domain_v1'
    assert signature['alt_range'] == [120.0, 500.0]
    try:
        NeuralETKFLayer(d_model=64, n_members=1)
    except ValueError:
        pass
    else:
        raise AssertionError('n_members=1 must be rejected')


def test_rank_and_spectrum_helpers():
    anomalies = torch.zeros(2, 8, 64)
    anomalies[:, :, :7] = torch.eye(8, 7)
    numeric, effective = _rank_metrics(anomalies)
    assert np.all(numeric == 7)
    assert np.all(effective > 1)
    spectrum = _spectrum(np.eye(64))
    assert spectrum['available_components'] == 64
    assert np.isclose(spectrum['explained_variance']['64'], 1.0)


def test_deterministic_source_mode_schedule_and_masks():
    assert [_source_mode_for_batch('deterministic_112', index)
            for index in range(8)] == [
                'M10', 'M01', 'M11', 'M11',
                'M10', 'M01', 'M11', 'M11']
    assert _source_mode_for_batch('random_profile', 0) is None
    assert _source_mode_for_batch(
        'balanced_profile_112', 0) == 'balanced_profile_112'
    payload = {
        'value': torch.ones(3, 2),
        'valid_mask': torch.tensor([
            [True, False], [True, True], [False, True]])}
    fy, cosmic = _apply_source_mode(payload, payload, 'M10')
    assert fy is payload
    assert not cosmic['valid_mask'].any()
    fy, cosmic = _apply_source_mode(payload, payload, 'M01')
    assert not fy['valid_mask'].any()
    assert cosmic is payload
    fy, cosmic = _apply_source_mode(payload, payload, 'M11')
    assert fy is payload and cosmic is payload


def test_balanced_profile_modes_cover_each_source():
    profile_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    payload = {
        'value': torch.ones(8, 1),
        'valid_mask': torch.ones(8, 1, dtype=torch.bool),
    }
    exposure = {profile_id: [] for profile_id in range(4)}
    for epoch in range(5):
        fy, cosmic = _apply_balanced_profile_modes(
            payload, payload, profile_ids, epoch)
        for profile_id in exposure:
            selected = profile_ids == profile_id
            keep_fy = bool(fy['valid_mask'][selected].all())
            keep_cosmic = bool(cosmic['valid_mask'][selected].all())
            exposure[profile_id].append((keep_fy, keep_cosmic))
    for modes in exposure.values():
        assert (True, False) in modes
        assert (False, True) in modes
        assert (True, True) in modes
    assert payload['valid_mask'].all()
    assert _balanced_profile_mode_counts(profile_ids, 0) == {
        'M10': 1, 'M01': 1, 'M11': 2}


def test_gradient_audit_reports_each_weighted_auxiliary_component():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    observation = parameter.square().sum()
    increment = 0.1 * parameter[0].square()
    vertical = 0.2 * parameter[1].square()
    auxiliary = increment + vertical
    observation_norm, auxiliary_norm, ratio, components = _gradient_norms(
        observation, auxiliary, [parameter], {
            'increment': increment,
            'vertical': vertical,
        })
    assert torch.isfinite(torch.stack([
        observation_norm, auxiliary_norm, ratio, *components.values()])).all()
    assert observation_norm > 0
    assert auxiliary_norm > 0
    assert components['increment'] > 0
    assert components['vertical'] > components['increment']
    _, _, _, scaled = _gradient_norms(
        observation, 0.25 * auxiliary, [parameter], {
            'auxiliary': 0.25 * auxiliary,
        })
    assert torch.allclose(
        scaled['auxiliary'], 0.25 * auxiliary_norm, rtol=1e-6, atol=1e-8)


if __name__ == '__main__':
    test_profile_subset_manifest_is_deterministic()
    test_date_split_manifest_is_blocked_and_deterministic()
    test_explicit_date_split_keeps_complete_profiles()
    test_observation_query_forwards_profile_allowlist_and_exclusion()
    test_architecture_signature_and_dimension_validation()
    test_rank_and_spectrum_helpers()
    test_deterministic_source_mode_schedule_and_masks()
    test_balanced_profile_modes_cover_each_source()
    test_gradient_audit_reports_each_weighted_auxiliary_component()
    print('experiment control checks passed')
