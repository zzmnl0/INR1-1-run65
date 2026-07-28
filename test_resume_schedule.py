"""Regression check for run66 two-stage training-state persistence."""

import tempfile
from pathlib import Path

import torch

from inr_modules.mdia.train_fsia import (
    _atomic_torch_save,
    _resume_needs_analysis_setup,
)


def test_training_state_round_trip():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    state = {
        'checkpoint_type': 'run66_training_state',
        'completed_epochs': 2,
        'background_epochs': 5,
        'analysis_epochs': 5,
        'stage': 'background',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'state.pth'
        _atomic_torch_save(state, path)
        loaded = torch.load(path, map_location='cpu', weights_only=False)

    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(loaded['model_state_dict'], strict=True)
    assert loaded['checkpoint_type'] == 'run66_training_state'
    assert loaded['completed_epochs'] < loaded['background_epochs']
    assert loaded['stage'] == 'background'


def test_stage_boundary_setup():
    background = {'stage': 'background'}
    analysis = {'stage': 'analysis'}
    assert not _resume_needs_analysis_setup(background, 2, 5)
    assert _resume_needs_analysis_setup(background, 5, 5)
    assert not _resume_needs_analysis_setup(analysis, 6, 5)
    try:
        _resume_needs_analysis_setup(analysis, 5, 5)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid Analysis checkpoint position was accepted')


if __name__ == '__main__':
    test_training_state_round_trip()
    test_stage_boundary_setup()
