"""Small, data-free A1-S7 contract checks."""

from types import SimpleNamespace

import numpy as np
import torch

from inr_modules.config_mdia import get_config_mdia
from inr_modules.data_managers.FY_dataloader import ProfileTimeBinSampler
from inr_modules.mdia.fsia_model import FSIA_INR_Model, background_adapter_gate
from inr_modules.mdia.train_fsia import _set_training_stage


class _DummyIRI(torch.nn.Module):
    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, coords, return_features=False):
        out = coords.new_zeros((coords.shape[0], 1))
        features = coords.new_zeros((coords.shape[0], 128))
        return (out, features) if return_features else out


def _s7_model():
    config = get_config_mdia()
    config.update({
        'background_condition_adapter': 'maglat2_smooth_v1',
        'background_training_scope': 'condition_adapter_only_dual_maglat_v1',
        'background_adapter_gate_semantics': 'abs_sinI_025_050_smoothstep_v1',
        'background_adapter_residual_cap': 0.1,
    })
    return FSIA_INR_Model(_DummyIRI(), config)


def test_gate_boundaries_and_continuity():
    values = torch.tensor([0.0, 0.25, 0.375, 0.5, 0.75])
    gate = background_adapter_gate(values)
    assert torch.equal(gate[[0, 1]], torch.zeros(2))
    assert torch.isclose(gate[2], torch.tensor(0.5))
    assert torch.equal(gate[[3, 4]], torch.ones(2))
    near = background_adapter_gate(torch.tensor([0.25 - 1e-5, 0.25, 0.25 + 1e-5]))
    assert torch.all(near[1:] >= near[:-1])


def test_zero_initialised_adapters_and_freeze_scope():
    model = _s7_model()
    for adapter in (model.background_condition_adapter_low,
                    model.background_condition_adapter_high):
        assert torch.equal(adapter[-1].weight, torch.zeros_like(adapter[-1].weight))
        assert torch.equal(adapter[-1].bias, torch.zeros_like(adapter[-1].bias))
    _set_training_stage(model, 'background')
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all('background_condition_adapter_low' in name
               or 'background_condition_adapter_high' in name
               for name in trainable)


def test_adapter_seed_is_reproducible_and_local():
    first = _s7_model()
    second = _s7_model()
    for left, right in zip(first.background_condition_adapter_low.parameters(),
                           second.background_condition_adapter_low.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(first.background_condition_adapter_high.parameters(),
                           second.background_condition_adapter_high.parameters()):
        assert torch.equal(left, right)


def test_sparse_profile_sampling_is_unique_and_budgeted():
    data = np.array([
        [-1.0, 10.0, 250.0, 0.0, 1.0],
        [-1.0, 10.0, 260.0, 0.1, 1.0],
        [-1.0, 10.0, 270.0, 0.2, 1.0],
        [55.0, 20.0, 350.0, 12.0, 1.0],
        [55.0, 20.0, 360.0, 12.1, 1.0],
    ], dtype=np.float32)
    dataset = SimpleNamespace(
        data=data,
        selected_indices=np.arange(len(data), dtype=np.int64),
        profile_ids=np.array([10, 10, 10, 20, 20], dtype=np.int64),
        bin_ids=np.array([0, 0, 0, 4, 4], dtype=np.int64),
        indices_by_bin={0: np.array([0, 1, 2]), 4: np.array([3, 4])},
    )
    sampler = ProfileTimeBinSampler(
        dataset, batch_size=8, points_per_profile=8, shuffle=True,
        sampling_semantics='source_date_physical_stratum_balanced_v1',
        schedule_seed=42, max_batches=3)
    batches = list(iter(sampler))
    assert len(batches) == 3
    for batch in batches:
        assert len(batch) <= 8
        assert len(batch) == len(set(batch))
    assert set(sampler.last_profile_order.tolist()) == {10, 20}


if __name__ == '__main__':
    test_gate_boundaries_and_continuity()
    test_zero_initialised_adapters_and_freeze_scope()
    test_sparse_profile_sampling_is_unique_and_budgeted()
    print('A1-S7 M00-B contract checks passed')
