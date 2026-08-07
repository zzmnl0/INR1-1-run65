"""Small contract checks for the QC-v2 Background-only lifecycle."""

import torch

from inr_modules.config_mdia import get_config_mdia
from inr_modules.mdia.train_fsia import (
    _background_epoch_length,
    _resume_needs_analysis_setup,
)
from inr_modules.mdia.fsia_model import background_trust_gate


class _Loader:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


def test_qcv2_defaults_start_without_legacy_seed():
    config = get_config_mdia()
    assert config['background_seed_ckpt'] is None
    assert config['background_training_semantics'] == (
        'qc_v2_date_blocked_train_only')
    assert config['save_dir'].endswith(
        'run66-m2v-qcv2-dateblocked-background')


def test_background_cycles_shorter_source_to_max_loader_length():
    fy = _Loader(2)
    cosmic = _Loader(3)
    assert _background_epoch_length(fy, cosmic, 'background') == 3
    assert _background_epoch_length(fy, cosmic, 'analysis') == 2


def test_background_resume_is_explicitly_marked_for_analysis_transition():
    assert _resume_needs_analysis_setup(
        {'stage': 'background'}, 5, 5)
    assert not _resume_needs_analysis_setup(
        {'stage': 'analysis'}, 6, 5)


def test_background_trust_gate_has_physical_core_and_continuous_transition():
    coords = torch.tensor([
        [-11.9, -76.0, 150.0, 5.0],   # Jicamarca, deep local night
        [-11.9, -76.0, 200.0, 5.0],
        [-11.9, -76.0, 250.0, 5.0],
        [-11.9, -76.0, 300.0, 5.0],
        [-11.9, -76.0, 150.0, 17.0],  # local day
    ])
    gate = background_trust_gate(coords, enabled=True)
    assert gate[0].item() == 0.0
    assert gate[1].item() == 0.0
    assert 0.0 < gate[2].item() < 1.0
    assert gate[3].item() == 1.0
    assert gate[4].item() == 1.0
    assert torch.isfinite(gate).all()
    assert torch.all((gate >= 0.0) & (gate <= 1.0))


def test_background_trust_gate_disabled_is_exact_identity():
    coords = torch.tensor([[-11.9, -76.0, 150.0, 5.0]])
    assert torch.equal(
        background_trust_gate(coords, enabled=False),
        torch.ones(1),
    )
