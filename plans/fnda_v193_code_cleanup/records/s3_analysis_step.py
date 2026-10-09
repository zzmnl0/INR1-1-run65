"""Isolated real-weight Analysis step and serialization baseline; no production writes."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
from pathlib import Path
import json
import random
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from inr_modules.mdia.checkpoint_io import load_fsia_analysis_checkpoint
from inr_modules.mdia.train_fsia import (
    _set_training_stage, _set_stage_mode, _make_optimizer,
    exact_mode_profile_losses, observation_gram_whitening_loss,
)

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
out = Path(__file__).parent / 's3_baseline_20261009' / 'analysis_step_v4'
out.mkdir(exist_ok=False)
identity = json.loads(Path(__file__).with_name('model_identity.json').read_text(encoding='utf-8'))
weight = Path(identity['main_run']['path']) / 'best_fsia_model.pth'
digest = hashlib.sha256(weight.read_bytes()).hexdigest()
model, config, _, _ = load_fsia_analysis_checkpoint(
    weight, device='cpu', expected_sha256=digest, verify_hashes=True)
_set_training_stage(model, 'analysis')
_set_stage_mode(model, 'analysis')
optimizer, scheduler = _make_optimizer(model, config, 10)
before = {k: v.detach().clone() for k, v in model.state_dict().items()}
frozen = {k for k, v in model.named_parameters() if not v.requires_grad}
arrays = np.load(out.parent / 'real_weights' / 'main_A1S_N8D64.npz')
coords, sw, peaks, target, obs_coords = [torch.from_numpy(arrays[k]) for k in
    ('coords', 'sw', 'iri_peak', 'target', 'obs_coords')]
with torch.no_grad():
    _, _, _, _, endpoint = model(obs_coords, sw.repeat_interleave(8, 0),
                                iri_peak=peaks.repeat_interleave(8, 0))
payload = dict(coords=obs_coords, background=endpoint['ne_bkg'].flatten().detach(),
    value=torch.from_numpy(arrays['obs_value']), valid_mask=torch.ones(32, dtype=torch.bool),
    rho_squared=torch.zeros(32), localization_weight=torch.ones(32),
    query_index=torch.arange(4).repeat_interleave(8), row_ptr=torch.arange(0, 33, 8),
    basis_z_background=endpoint['basis_z_background'].detach(),
    basis_h_sw=endpoint['basis_h_sw'].detach())
_, _, _, _, extras = model(coords, sw, iri_peak=peaks,
    observations_fy=payload, observations_cosmic=payload)
data_loss, _, _ = exact_mode_profile_losses(extras, target, torch.arange(4), 0.2)
gram, _, _ = observation_gram_whitening_loss(
    extras, torch.arange(4), model.kalman_layer, return_details=True)
manifest = json.loads((weight.parent / 'run_manifest.json').read_text(encoding='utf-8'))
gram_weight = float(manifest['resolved_training']['resolved_gram_weight'])
loss = data_loss + gram_weight * gram
assert torch.isfinite(loss)
optimizer.zero_grad(set_to_none=True)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
if scheduler is not None:
    scheduler.step()
changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
assert changed and not (set(changed) & frozen)
assert all(torch.isfinite(v).all() for v in model.state_dict().values())
state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
    scheduler=scheduler.state_dict() if scheduler else None,
    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state())
torch.save(state, out / 'diagnostic_state.pth')
saved = torch.load(out / 'diagnostic_state.pth', map_location='cpu', weights_only=False)
expected = (random.random(), np.random.rand(), torch.rand(3))
restored, _, _, _ = load_fsia_analysis_checkpoint(
    weight, device='cpu', expected_sha256=digest, verify_hashes=True)
_set_training_stage(restored, 'analysis')
_set_stage_mode(restored, 'analysis')
restored.load_state_dict(saved['model'], strict=True)
opt2, sch2 = _make_optimizer(restored, config, 10)
opt2.load_state_dict(saved['optimizer'])
if sch2 is not None:
    sch2.load_state_dict(saved['scheduler'])
def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b
assert equal(model.state_dict(), restored.state_dict())
assert equal(optimizer.state_dict(), opt2.state_dict())
assert equal(state['scheduler'], sch2.state_dict() if sch2 else None)
random.setstate(saved['python_rng'])
np.random.set_state(saved['numpy_rng'])
torch.set_rng_state(saved['torch_rng'])
actual = (random.random(), np.random.rand(), torch.rand(3))
assert equal(expected, actual)
assert hashlib.sha256(weight.read_bytes()).hexdigest() == digest
report = dict(status='PASS', objective='fixed synthetic data loss + resolved Gram weight * Gram',
    full_production_loss=False, loss=float(loss.detach()), gram_weight=gram_weight,
    changed_parameter_count=len(changed), changed_parameters=changed,
    frozen_parameter_count=len(frozen), frozen_unchanged=True,
    strict_model_optimizer_scheduler_rng_roundtrip=True,
    scheduler_type=type(scheduler).__name__ if scheduler else None,
    production_weight_unchanged=True, weight_sha256=digest)
(out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'changed_parameters'}))
