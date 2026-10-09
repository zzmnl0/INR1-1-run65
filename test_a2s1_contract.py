"""Focused T0 contract checks; no production data or training is required."""

import numpy as np
import torch

from inr_modules.data_managers.FY_dataloader import _altitude_mask
from inr_modules.mdia import checkpoint_io
from inr_modules.mdia.train_fsia import (
    _A2S1_DOMAIN_SEMANTICS,
    _background_epoch_length,
    _low_altitude_anchor_batch,
    _validate_a2s1_loader_lengths,
    _validate_hybrid_low_altitude_protocol,
)
from inr_modules.mdia.v14_contract import (
    A2S1_DOMAIN_SEMANTICS,
    canonical_a2s1_config,
    canonical_a2s1_training_protocol,
    canonical_v14_config,
)


def test_a2s1_is_distinct_from_v14_and_keeps_low_prior():
    config = canonical_a2s1_config()
    config['training_protocol'] = canonical_a2s1_training_protocol()
    config.update({
        'basis_dim': 64,
        'enkf_n_members': 8,
        'r_mode': 'global',
        'background_trust_gate_enabled': False,
    })
    assert A2S1_DOMAIN_SEMANTICS == _A2S1_DOMAIN_SEMANTICS
    assert config['checkpoint_format_version'] == 15
    assert config['observation_alt_range'] == (120.0, 500.0)
    assert config['peak_search_alt_range'] == (200.0, 500.0)
    assert config['training_protocol_revision'] == 'a2s1_a1s_legacy_120_500_v1'
    assert config['training_protocol']['pairing'] == (
        'cycle_shorter_source_max_loader_length')
    _validate_hybrid_low_altitude_protocol(config)


def test_v14_contract_remains_unchanged_and_loader_maps_are_strict():
    old = canonical_v14_config()
    assert old['checkpoint_format_version'] == 14
    assert old['observation_alt_range'] == (200.0, 500.0)
    assert checkpoint_io._DOMAIN_CONTRACTS[
        'hybrid_120_500_model_200_500_observation_v1']['format_version'] == 14
    assert checkpoint_io._DOMAIN_CONTRACTS[A2S1_DOMAIN_SEMANTICS] == {
        'format_version': 15,
        'alt_range': (120.0, 500.0),
        'observation_alt_range': (120.0, 500.0),
        'peak_search_alt_range': (200.0, 500.0),
    }


def test_a2s1_boundaries_are_inclusive_and_prior_stops_below_200():
    altitudes = np.asarray([119.999, 120.0, 199.9, 200.0, 500.0, 500.001])
    assert _altitude_mask(altitudes, (120.0, 500.0)).tolist() == [
        False, True, True, True, True, False]

    config = canonical_a2s1_config()
    coords = torch.zeros((3, 4), dtype=torch.float32)
    profile_ids = torch.tensor([10, 10, 11], dtype=torch.long)
    anchors, groups = _low_altitude_anchor_batch(coords, profile_ids, config)
    assert anchors.shape == (16, 4)
    assert torch.unique(anchors[:, 2]).tolist() == list(range(120, 200, 10))
    assert float(anchors[:, 2].max()) < 200.0
    assert torch.unique(groups).tolist() == [10, 11]


def test_a2s1_keeps_max_schedule_and_rejects_unequal_formal_loaders():
    config = {'model_domain_semantics': A2S1_DOMAIN_SEMANTICS}
    fy_loader = [None] * 3
    cosmic_loader = [None] * 5
    assert _background_epoch_length(
        fy_loader, cosmic_loader, 'analysis', A2S1_DOMAIN_SEMANTICS) == 5
    try:
        _validate_a2s1_loader_lengths(config, fy_loader, cosmic_loader)
    except RuntimeError as error:
        assert 'equal FY/COSMIC loader lengths' in str(error)
    else:
        raise AssertionError('formal A2S1 accepted unequal loader lengths')
    _validate_a2s1_loader_lengths(
        dict(config, smoke_run=True), fy_loader, cosmic_loader)
