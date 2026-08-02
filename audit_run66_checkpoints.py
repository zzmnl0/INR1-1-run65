"""Strict, finite, deterministic audit for run66 epoch checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from audit_etkf_observation_subspace import _load
from evaluate_satellite_development import _sha256
from inr_modules.mdia.sliding_dataset import attach_observation_background


def _tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=range(6, 11))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = []
    for epoch in args.epochs:
        checkpoint = run_dir / f"epoch_{epoch:02d}_model.pth"
        model, sw_manager, iri_peak_manager, _ = _load(run_dir, checkpoint)
        coords = torch.tensor([
            [-12.0, -76.8, 180.0, 48.0],
            [65.0, 147.0, 250.0, 240.0],
            [0.0, 179.0, 320.0, 360.0],
            [30.0, -179.0, 450.0, 600.0],
        ])
        sw = sw_manager.get_drivers_sequence(coords[:, 3])
        peak = (
            iri_peak_manager.get_iri_peak(coords)
            if iri_peak_manager is not None else None)
        with torch.no_grad():
            background = model(coords, sw, iri_peak=peak)[4]["ne_bkg"]
            observation = {
                "coords": coords[:, None],
                "value": background + 0.1,
                "background": background,
                "valid_mask": torch.ones(4, 1, dtype=torch.bool),
                "rho_squared": torch.zeros(4, 1),
            }
            observation = attach_observation_background(
                observation, model, sw_manager, iri_peak_manager)
            first = model(
                coords, sw, iri_peak=peak, observations_fy=observation)
            second = model(
                coords, sw, iri_peak=peak, observations_fy=observation)
        first_tensors = list(_tensors(first))
        second_tensors = list(_tensors(second))
        finite = all(torch.isfinite(value).all() for value in first_tensors)
        repeat_error = max(
            (float((left - right).abs().max())
             for left, right in zip(first_tensors, second_tensors)
             if left.numel()),
            default=0.0)
        rows.append({
            "epoch": epoch,
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
            "strict_load": True,
            "all_tensors_finite": finite,
            "outputs_finite": finite,
            "repeat_inference_max_abs_difference": repeat_error,
            "repeat_inference_deterministic": repeat_error == 0.0,
        })
    report = {
        "schema_version": 1,
        "checkpoints": rows,
        "all_passed": all(
            row["all_tensors_finite"]
            and row["repeat_inference_deterministic"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
