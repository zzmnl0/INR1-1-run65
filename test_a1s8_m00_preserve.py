"""Data-free A1-S8 contract checks."""

import numpy as np
import torch

from inr_modules.mdia import train_fsia as train


def _a1s8_config(**overrides):
    config = {
        'background_training_semantics': train._A1S8_BACKGROUND_SEMANTICS,
        'background_sampling_semantics': train._A1S8_SAMPLING_SEMANTICS,
        'checkpoint_semantic': train._A1S8_CHECKPOINT_SEMANTICS,
        'background_source_mix_schedule': train._A1S8_SOURCE_MIX_SEMANTICS,
        'background_mix_legacy_fraction': 0.75,
        'background_mix_balanced_fraction': 0.25,
        'background_fixed_updates_per_epoch': 960,
    }
    config.update(overrides)
    return config


def test_source_mix_is_deterministic_three_to_one():
    config = _a1s8_config()
    values = [train.a1s8_source_mix(i, config) for i in range(8)]
    assert [v[2] for v in values] == [
        'legacy', 'legacy', 'legacy', 'balanced',
        'legacy', 'legacy', 'legacy', 'balanced']
    assert all(np.isclose(v[0], 0.75) and np.isclose(v[1], 0.25)
               for v in values)


def test_gate_aware_loss_reports_common_mass_and_is_finite():
    coords = torch.tensor([
        [0.0, 0.0, 250.0, 1.0],
        [0.0, 0.0, 250.0, 1.0],
        [60.0, 20.0, 350.0, 12.0],
        [60.0, 20.0, 350.0, 12.0],
    ])
    profile_ids = torch.tensor([1, 1, 2, 2])
    prediction = torch.tensor([1.0, 1.0, 2.0, 2.0])
    target = torch.zeros_like(prediction)
    gate = torch.tensor([0.0, 0.0, 1.0, 1.0])
    value, details = train.background_gate_aware_loss(
        prediction, target, coords, profile_ids, gate, return_details=True)
    assert torch.isfinite(value)
    assert details['low_effective_mass'] > 0.0
    assert details['high_effective_mass'] > 0.0
    assert np.isclose(details['low_effective_mass']
                      + details['high_effective_mass'], 1.0)
    assert details['low_profiles'] == details['high_profiles'] == 1


def test_zero_adapter_anchor_is_zero():
    coords = torch.tensor([[0.0, 0.0, 250.0, 1.0], [60.0, 20.0, 350.0, 12.0]])
    ids = torch.tensor([1, 2])
    loss, details = train.background_adapter_anchor_loss(
        torch.zeros(2), coords, ids, return_details=True)
    assert loss.item() == 0.0
    assert details['profiles'] == 2


def test_a1s8_semantics_fail_closed_and_budget_fixed():
    config = _a1s8_config()
    assert train._a1s8_enabled(config)
    config['checkpoint_semantic'] = 'a1s7_m00_b_v1'
    assert not train._a1s8_enabled(config)
    config = _a1s8_config(background_fixed_updates_per_epoch=0)
    try:
        train._resolve_a1s8_update_budget(config)
    except ValueError:
        pass
    else:
        raise AssertionError('non-positive A1-S8 update budget must fail closed')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
    print('A1-S8 M00-preserve contract checks passed')
