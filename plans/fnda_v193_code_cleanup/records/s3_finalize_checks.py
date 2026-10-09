"""Verify S3 source stability and collect existing acceptance evidence."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
scope = read(P / 'scope_dependencies.json')
sources = {k: sha(ROOT / k) == v['sha256'] for k, v in scope['files'].items()}
assert len(sources) == 160 and all(sources.values())
backup = read(P / 'backup_manifest.json')
model = read(P / 's3_baseline_20261009/real_weights/model_baseline.json')
step = read(P / 's3_baseline_20261009/analysis_step_v4/result.json')
assert model['status'] == step['status'] == 'PASS'
weights = {k: sha(v['checkpoint']) == v['weight_sha256'] for k, v in model['runs'].items()}
assert all(weights.values())
identity = read(P / 'model_identity.json')
identity_weights = {v['path']: sha(v['path']) == v['sha256'] for v in identity['weights'] if v.get('exists') and v.get('sha256')}
assert all(identity_weights.values())
base = P / 's3_baseline_20261009/real_weights'
a = np.load(base / 'main_A1S_N8D64.npz')
b = np.load(base / 'FNDA_V8_N8D64_REPRODUCTION.npz')
assert set(a.files) == set(b.files)
arrays = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
assert all(arrays.values())
git_status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
saved_status = scope['git_tracked_status']
if isinstance(saved_status, str):
    saved_status = saved_status.splitlines()
assert '\n'.join(git_status).strip() == '\n'.join(saved_status).strip(), (git_status, saved_status)
result = dict(stage='S3', status='COMPLETE_WITH_USER_ACCEPTED_EXCLUSIONS', next_stage='S4_PENDING_USER_DECISION',
    backup_status=backup['status'], backup_path=backup['destination'], backup_counts=backup['counts'],
    backup_bytes=backup['copied_bytes'], restore_sample_count=len(backup['restore_sample']),
    source_hash_count=len(sources), source_hashes_unchanged=sources,
    formal_weights_unchanged=weights, identity_weights_unchanged=identity_weights,
    tracked_git_status_unchanged=True, regression_passed=79, regression_warnings=242,
    real_weight_runs={k: v['status'] for k, v in model['runs'].items()},
    main_and_N8D64_reproduction_arrays_exact=arrays, analysis_step=step,
    limitations=['300 unreadable paths protected in place by user decision',
    'Registered-in-place assets are not copied disaster recovery',
    'Analysis step uses synthetic data and partial diagnostic objective, not full training resume',
    'Process command lines unavailable; source stability verified with hashes',
    'No full paper metrics recomputation or full training'])
(P / 'S3_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status', 'source_hash_count', 'tracked_git_status_unchanged', 'next_stage')}))
