"""Focused synthetic contracts for the M00 night-v2 loss."""
import torch
import pytest
import numpy as np
from types import SimpleNamespace
from inr_modules.mdia import train_fsia as training

from inr_modules.mdia.train_fsia import (
    _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2,
    background_stratified_bias_loss,
)


def _batch():
    # [lat, lon, altitude, UTC hour], two profiles per supported cell.
    coords, target, ids = [], [], []
    for cell, hour in enumerate((0.0, 12.0, 0.0, 12.0, 0.0, 12.0)):
        lat = (0.1 if cell % 3 == 0 else 30.0 if cell % 3 == 1 else 70.0)
        alt = 250.0 if cell < 3 else 400.0
        for profile in range(8):
            coords.append([lat, 0.0, alt, hour])
            target.append(0.0)
            ids.append(cell * 8 + profile)
    return (torch.tensor(coords), torch.tensor(target), torch.tensor(ids))


def test_v2_equal_profile_cells_and_finite_fallback():
    coords, target, ids = _batch()
    prediction = torch.ones(len(ids), requires_grad=True)
    support = {f'{mag}_{period}': 30 for mag in (
        'abs_sinI_lt_0.25', 'abs_sinI_0.25_0.5', 'abs_sinI_ge_0.5')
               for period in ('night', 'day')}
    value, details = background_stratified_bias_loss(
        prediction, target, coords, ids, grouping=
        _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2,
        magnetic_support_counts=support, return_details=True)
    assert torch.isfinite(value)
    assert details['magnetic_active_cell_count'] > 0
    assert torch.isclose(details['_height_loss_tensor'], torch.tensor(1.0))
    assert torch.isclose(details['_magnetic_loss_tensor'], torch.tensor(1.0))
    value.backward()
    assert torch.isfinite(prediction.grad).all()

    invalid = prediction.detach().clone().requires_grad_()
    bad = coords.clone(); bad[0, 2] = float('nan')
    fallback, _ = background_stratified_bias_loss(
        invalid, target, bad, ids, grouping=
        _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2,
        magnetic_support_counts={}, return_details=True)
    assert torch.isfinite(fallback)


def _conditional(count=8, support_count=30, repeat=1):
    # Same low-dip location; day and night must not cancel one another.
    coords = torch.tensor([[0., 0., 250., hour]
                           for hour in (0., 12.) for _ in range(count)])
    ids = torch.arange(2 * count)
    errors = torch.cat((torch.ones(count), -torch.ones(count)))
    if repeat > 1:
        index = torch.cat((torch.zeros(repeat, dtype=torch.long),
                           torch.arange(1, len(ids))))
        coords, ids, errors = coords[index], ids[index], errors[index]
    errors.requires_grad_()
    support = {f'{mag}_{period}': support_count for mag in (
        'abs_sinI_lt_0.25', 'abs_sinI_0.25_0.5', 'abs_sinI_ge_0.5')
               for period in ('night', 'day')}
    old, detail = background_stratified_bias_loss(
        errors, torch.zeros_like(errors), coords, ids,
        grouping=_BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2,
        magnetic_support_counts=support, return_details=True)
    return errors, coords, ids, old, detail


def test_v2_support_boundaries_profile_weights_and_old_weight_gradient():
    for count, support, active in ((7, 30, False), (8, 29, False), (8, 30, True)):
        errors, coords, ids, old, detail = _conditional(count, support)
        assert bool(detail['magnetic_active_cell_count']) == active
        assert detail['_magnetic_loss_tensor'].item() == float(active)
        legacy = background_stratified_bias_loss(
            errors, torch.zeros_like(errors), coords, ids)
        weighted = .01226901322277942 * old + 0. * detail['_magnetic_loss_tensor']
        legacy_weighted = .01226901322277942 * legacy
        assert torch.equal(weighted, legacy_weighted)
        assert torch.equal(torch.autograd.grad(weighted, errors, retain_graph=True)[0],
                           torch.autograd.grad(legacy_weighted, errors)[0])
    _, _, _, _, normal = _conditional()
    _, _, _, _, unequal = _conditional(repeat=17)
    assert normal['_magnetic_loss_tensor'].item() == 1.
    assert torch.equal(normal['_magnetic_loss_tensor'], unequal['_magnetic_loss_tensor'])
    # Source objectives are squared before source averaging.
    assert .5 * (normal['_magnetic_loss_tensor'] + unequal['_magnetic_loss_tensor']) == 1.


def test_v2_all_nonfinite_fallback_has_finite_zero_gradient():
    prediction = torch.tensor([float('nan'), float('inf')], requires_grad=True)
    zero, detail = background_stratified_bias_loss(
        prediction, torch.zeros(2), torch.zeros(2, 4), torch.arange(2),
        grouping=_BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2,
        return_details=True)
    loss = zero + detail['_magnetic_loss_tensor']
    assert loss.item() == 0.
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_v2_calibration_changes_only_new_weight_and_restores_state(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(.1)
    model.weight.grad = torch.ones_like(model.weight)
    model.eval()
    _, coords, ids, _, _ = _conditional()
    batch = (coords, torch.zeros(len(ids)), ids)
    monkeypatch.setattr(training, '_set_training_stage', lambda *args: None)
    monkeypatch.setattr(training, '_set_stage_mode', lambda *args: model.train())
    monkeypatch.setattr(training, '_current_stage_trainable_parameters', lambda m: list(m.parameters()))
    monkeypatch.setattr(training, '_unpack_source_batch', lambda b, device: b)
    def forward(*args):
        torch.rand(1); np.random.random()
        return None, None, None, None, {'ne_bkg': model(torch.ones(len(ids), 1)).reshape(-1)}
    monkeypatch.setattr(training, '_source_forward', forward)
    config = {
        'background_bias_grouping': _BACKGROUND_BIAS_GROUPING_MAGNETIC_DAYNIGHT_V2,
        'background_magnetic_daynight_calibration_batches': 20,
        'resolved_background_stratum_bias_weight': .01226901322277942,
        'background_bias_magnetic_support_counts': {
            source: {f'{mag}_{period}': 30 for mag in (
                'abs_sinI_lt_0.25', 'abs_sinI_0.25_0.5', 'abs_sinI_ge_0.5')
                     for period in ('night', 'day')} for source in ('FY', 'COSMIC')}}
    rng, nrng = torch.get_rng_state(), np.random.get_state()
    value = training._resolve_background_magnetic_daynight_weight(
        model, [batch], [batch], None, torch.device('cpu'), config,
        SimpleNamespace(get_drivers_sequence=lambda times: None), None)
    assert value > 0
    assert config['resolved_background_stratum_bias_weight'] == .01226901322277942
    assert config['background_magnetic_daynight_gradient_calibration']['processed_batches'] == 20
    assert torch.equal(torch.get_rng_state(), rng)
    assert np.array_equal(np.random.get_state()[1], nrng[1])
    assert not model.training
    assert torch.equal(model.weight.grad, torch.ones_like(model.weight))
    config['background_bias_magnetic_support_counts'] = {}
    with pytest.raises(RuntimeError, match='no finite gradients'):
        training._resolve_background_magnetic_daynight_weight(
            model, [batch], [batch], None, torch.device('cpu'), config,
            SimpleNamespace(get_drivers_sequence=lambda times: None), None)


if __name__ == '__main__':
    test_v2_equal_profile_cells_and_finite_fallback()
    print('ok')
