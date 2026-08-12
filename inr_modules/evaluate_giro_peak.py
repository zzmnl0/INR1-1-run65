"""Compatibility entry point for the M2-W GIRO peak evaluator."""

import argparse
import sys
from pathlib import Path

import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from evaluate_giro_peak import evaluate_giro_peak  # noqa: E402
from inr_modules.mdia.checkpoint_io import (  # noqa: E402
    load_fsia_analysis_checkpoint,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save-dir')
    parser.add_argument('--preflight-only', action='store_true')
    arguments = parser.parse_args()
    if arguments.preflight_only:
        _, config, _, summary = load_fsia_analysis_checkpoint(
            arguments.checkpoint, device=torch.device('cpu'),
            require_domain='strict_200_500_domain_v1',
            allow_historical_epoch=True)
        print('[preflight] M2-W Analysis checkpoint passed: '
              f'v{summary["checkpoint_format_version"]}, '
              f'alt_range={config["alt_range"]}')
    else:
        evaluate_giro_peak(arguments.checkpoint, arguments.save_dir)
