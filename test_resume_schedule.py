"""Regression checks for global-epoch resume semantics and full state round-trip."""

import tempfile
from pathlib import Path

import torch

from inr_modules.mdia.train_fsia import (
    _atomic_torch_save,
    _load_optimizer_state,
    _optimizer_parameter_names,
    _remaining_phase_counts,
)


def test_resume_phase_counts():
    assert _remaining_phase_counts(2, 10, 5) == (3, 5)
    assert _remaining_phase_counts(6, 10, 5) == (0, 4)


def test_training_state_round_trip():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()

    state = {
        'checkpoint_type': 'fsia_training_state',
        'completed_epochs': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'state.pth'
        _atomic_torch_save(state, path)
        loaded = torch.load(path, map_location='cpu', weights_only=False)

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=10)
    restored.load_state_dict(loaded['model_state_dict'], strict=True)
    restored_optimizer.load_state_dict(loaded['optimizer_state_dict'])
    restored_scheduler.load_state_dict(loaded['scheduler_state_dict'])
    assert loaded['completed_epochs'] == 1
    assert restored_optimizer.state_dict()['state']
    assert restored_scheduler.last_epoch == scheduler.last_epoch


def test_optimizer_state_drops_newly_frozen_parameter():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    old_names = _optimizer_parameter_names(model)
    old_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model(torch.ones(1, 2)).sum().backward()
    old_optimizer.step()

    model[0].requires_grad_(False)
    new_optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=3e-4,
    )
    migrated = _load_optimizer_state(
        new_optimizer, old_optimizer.state_dict(), model, old_names)

    assert migrated
    assert len(new_optimizer.param_groups[0]['params']) == 2
    assert len(new_optimizer.state_dict()['state']) == 2


if __name__ == '__main__':
    test_resume_phase_counts()
    test_training_state_round_trip()
    test_optimizer_state_drops_newly_frozen_parameter()
