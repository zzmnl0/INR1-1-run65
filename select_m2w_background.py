"""Select the eligible M2-W Background candidate before Analysis resume."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate(run_dir):
    run_dir = Path(run_dir).resolve()
    with (run_dir / 'training_summary.json').open(encoding='utf-8') as stream:
        summary = json.load(stream)
    if (summary.get('completed_stage') != 'background'
            or summary.get('checkpoint_format_version') != 13
            or summary.get('model_domain_semantics') !=
            'strict_200_500_domain_v1'):
        raise ValueError(f'{run_dir} is not a complete M2-W Background run')
    checkpoint = run_dir / Path(summary['checkpoint']).name
    if not checkpoint.is_file() or _sha256(checkpoint) != summary.get(
            'checkpoint_sha256'):
        raise ValueError(f'{run_dir} Background checkpoint identity mismatch')
    metrics = summary.get('background_development') or {}
    values = np.asarray([
        metrics['ccc'], metrics['rmse'], metrics['pearson_r'],
    ], dtype=np.float64)
    eligible = bool(summary.get('background_development_gate_passed'))
    eligible = eligible and np.isfinite(values).all()
    return {
        'run_dir': str(run_dir),
        'checkpoint': str(checkpoint),
        'eligible': bool(eligible),
        'mean_ccc': float(values[0]),
        'mean_rmse': float(values[1]),
        'mean_pearson_r': float(values[2]),
        'trust_gate_enabled': bool(summary.get(
            'background_trust_gate_enabled', False)),
        'selection_key': [float(values[0]), -float(values[1]),
                          float(values[2]),
                          int(not summary.get(
                              'background_trust_gate_enabled', False))],
    }


def select_background(run_dirs):
    candidates = [_candidate(path) for path in run_dirs]
    eligible = [item for item in candidates if item['eligible']]
    if not eligible:
        raise RuntimeError('no M2-W Background candidate passed both source gates')
    winner = max(eligible, key=lambda item: tuple(item['selection_key']))
    return {'candidates': candidates, 'winner': winner}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', nargs=2)
    parser.add_argument('--output')
    arguments = parser.parse_args()
    result = select_background(arguments.run_dirs)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(text + '\n', encoding='utf-8')
    print(text)
