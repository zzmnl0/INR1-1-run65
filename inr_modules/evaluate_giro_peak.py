"""Compatibility entry point for the M2-W GIRO peak evaluator."""

import argparse
import sys
from pathlib import Path

import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from evaluate_giro_peak import (  # noqa: E402
    _require_m2w_peak_contract,
    evaluate_giro_peak,
)
from inr_modules.mdia.checkpoint_io import (  # noqa: E402
    load_fsia_analysis_checkpoint,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save-dir')
    parser.add_argument('--baseline-checkpoint')
    parser.add_argument('--preflight-only', action='store_true')
    arguments = parser.parse_args()
    if arguments.preflight_only:
        _, config, _, summary = load_fsia_analysis_checkpoint(
            arguments.checkpoint, device=torch.device('cpu'),
            allow_historical_epoch=True)
        _require_m2w_peak_contract(config)
        print('[preflight] M2-W Analysis checkpoint passed: '
              f'v{summary["checkpoint_format_version"]}, '
              f'alt_range={config["alt_range"]}')
    else:
        evaluate_giro_peak(
            arguments.checkpoint, arguments.save_dir,
            arguments.baseline_checkpoint)
