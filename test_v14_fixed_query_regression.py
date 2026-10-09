"""Frozen-v14 fixed-query M00/M11 numerical regression."""

from pathlib import Path

import pytest
import torch

from inr_modules.mdia.checkpoint_io import load_fsia_analysis_checkpoint
from inr_modules.mdia.fsia_model import solve_density_mode


CHECKPOINT = Path(
    'checkpoints_fsia/run66-m2w-120-500-obs-200-500-gate-off/'
    'best_fsia_model.pth')
if not CHECKPOINT.is_file():
    pytest.skip('frozen v14 checkpoint is not present', allow_module_level=True)


def _payload(coords, background, endpoint):
    background = background.flatten()
    count = len(coords)
    return {
        'coords': coords.clone(),
        'value': background + 0.1,
        'background': background,
        'valid_mask': torch.ones(count, dtype=torch.bool),
        'rho_squared': torch.zeros(count),
        'localization_weight': torch.ones(count),
        'query_index': torch.arange(count),
        'row_ptr': torch.arange(count + 1),
        'basis_z_background': endpoint['basis_z_background'],
        'basis_h_sw': endpoint['basis_h_sw'],
    }


def test_frozen_v14_fixed_query_m00_m11():
    model, _, _, summary = load_fsia_analysis_checkpoint(
        CHECKPOINT, device='cpu',
        require_domain='hybrid_120_500_model_200_500_observation_v1')
    assert summary['checkpoint_sha256'] == (
        '655db1733498c86fcf200e284abb96e50fd546040b2fca4b83e05dd3eed32e8a')
    coords = torch.tensor([
        [-12.0, -76.8, 120.0, 0.0],
        [0.0, 120.0, 250.0, 240.0],
        [35.0, 30.0, 400.0, 480.0],
        [65.0, 150.0, 199.9, 720.0],
    ])
    sw = torch.zeros(4, 36, 2)
    iri_peak = torch.tensor([[300.0, 11.5]] * 4)
    with torch.no_grad():
        m00, _, _, _, endpoint = model(coords, sw, iri_peak=iri_peak)
        payload = _payload(coords, endpoint['ne_bkg'], endpoint)
        m11, _, _, _, extras = model(
            coords, sw, iri_peak=iri_peak,
            observations_fy=payload, observations_cosmic=payload)
        m11_from_modes = endpoint['ne_bkg'] + solve_density_mode(
            extras, ('FY', 'COSMIC')).unsqueeze(-1)
    expected_m00 = torch.tensor([
        9.46693229675293, 11.981146812438965,
        11.606842041015625, 11.684921264648438])
    expected_m11 = torch.tensor([
        9.509805679321289, 12.011528968811035,
        11.681675910949707, 11.710855484008789])
    assert torch.allclose(m00[:, 0], expected_m00, atol=1e-6, rtol=0.0)
    assert torch.allclose(m11[:, 0], expected_m11, atol=1e-6, rtol=0.0)
    assert torch.allclose(m11, m11_from_modes, atol=1e-7, rtol=0.0)
