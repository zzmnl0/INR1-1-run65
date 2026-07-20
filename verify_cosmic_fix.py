"""Real-data smoke check for the repaired run65 COSMIC branch."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from inr_modules.config_mdia import get_config_mdia
from inr_modules.data_managers.FY_dataloader import COSMICNeighborhoodIndex
from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.train_fsia import _query_cosmic_neighbors


OLD_CHECKPOINT = Path(
    r"D:\code11\IRI01\IRI03\INR1-1\FSIA_INR18\checkpoints_fsia\run65"
    r"\best_fsia_model0.pth")
EXPECTED_OLD_SHA256 = "dd23cff3d1050ad7b333eef37661389f30818bf47471d872ddbe4ad8549c7fce"
OUT_DIR = Path("checkpoints/run65_cosmic_fix_smoke")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(cfg, device):
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    proxy.load_state_dict(torch.load(
        cfg["iri_proxy_path"], map_location=device, weights_only=True), strict=True)
    model = FSIA_INR_Model(proxy, cfg).to(device)
    model.load_state_dict(torch.load(
        OLD_CHECKPOINT, map_location=device, weights_only=True), strict=True)
    model._initialize_cosmic_bootstrap()
    return model


def forward(model, coords, sw_seq, iri_peak, neighbors=None, has_obs=None,
            fy_neighbors=None, fy_has_obs=None):
    return model(
        coords, sw_seq, iri_peak=iri_peak,
        neighbors_feats=fy_neighbors, has_obs=fy_has_obs,
        neighbors_feats_cosmic=neighbors, has_obs_cosmic=has_obs)[0]


def main():
    torch.manual_seed(20240965)
    np.random.seed(20240965)
    device = torch.device("cpu")
    cfg = dict(get_config_mdia())

    old_sha_before = sha256(OLD_CHECKPOINT)
    assert old_sha_before == EXPECTED_OLD_SHA256

    cosmic_index = COSMICNeighborhoodIndex(cfg["cosmic_path"], cfg)
    sample_idx = np.linspace(
        0, len(cosmic_index.sorted_data) - 1, 32, dtype=np.int64)
    sample = np.array(cosmic_index.sorted_data[sample_idx], dtype=np.float32)
    coords = torch.from_numpy(sample[:, :4]).to(device)
    target = torch.from_numpy(sample[:, 4:5]).to(device)
    neighbors, has_obs = _query_cosmic_neighbors(
        SimpleNamespace(cosmic_nb_index=cosmic_index), coords)
    assert has_obs.all(), "selected COSMIC samples must have local coverage"

    sw_manager = SpaceWeatherManager(
        cfg["sw_path"], cfg["start_date_str"], cfg["total_hours"],
        cfg["seq_len"], device)
    peak_manager = IRIPeakManager(
        cfg["iri_hmf2_path"], cfg["iri_nmf2_path"],
        cfg["total_hours"], device)
    sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
    iri_peak = peak_manager.get_iri_peak(coords)
    model = load_model(cfg, device)

    model.eval()
    m00 = forward(model, coords, sw_seq, iri_peak)
    m01_before = forward(model, coords, sw_seq, iri_peak, neighbors, has_obs)
    dummy_fy = torch.zeros_like(neighbors)
    no_fy = torch.zeros_like(has_obs)
    m01_with_empty_fy = forward(
        model, coords, sw_seq, iri_peak, neighbors, has_obs, dummy_fy, no_fy)
    mask_independence = float(
        (m01_before - m01_with_empty_fy).abs().max().detach())
    assert mask_independence <= 1e-7

    model.train()
    loss = F.mse_loss(
        forward(model, coords, sw_seq, iri_peak, neighbors, has_obs), target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    d = model.kalman_layer.d_model
    gradients = {
        "cosmic_input_proj": float(
            model.cosmic_obs_encoder.input_proj.weight.grad.abs().max()),
        "cosmic_query": float(model.cosmic_obs_encoder.query.grad.abs().max()),
        "h_cosmic_w": float(model.kalman_layer.H_COSMIC_w.grad.abs().max()),
        "proj_pre_cosmic": float(model.proj_pre.weight.grad[:, -d:].abs().max()),
    }
    assert all(value > 0 for value in gradients.values()), gradients

    params = list(model.cosmic_obs_encoder.parameters()) + [
        model.kalman_layer.H_COSMIC_w,
        *model.kalman_layer.R_COSMIC_net.parameters(),
        model.kalman_layer.log_r_ref_COSMIC,
    ]
    optimizer = torch.optim.Adam(params, lr=1e-2)
    loss_before = float(loss.detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        pred = forward(model, coords, sw_seq, iri_peak, neighbors, has_obs)
        step_loss = F.mse_loss(pred, target)
        step_loss.backward()
        optimizer.step()
    loss_after = float(step_loss.detach())
    assert loss_after < loss_before, (loss_before, loss_after)

    model.eval()
    with torch.inference_mode():
        m01_after = forward(model, coords, sw_seq, iri_peak, neighbors, has_obs)
        m00_after = forward(model, coords, sw_seq, iri_peak)
        repeat = forward(model, coords, sw_seq, iri_peak, neighbors, has_obs)
    cosmic_effect = float((m01_after - m00_after).abs().max())
    deterministic_diff = float((m01_after - repeat).abs().max())
    assert cosmic_effect > 1e-6
    assert deterministic_diff <= 1e-7

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT_DIR / "smoke_overfit_model.pth"
    torch.save(model.state_dict(), checkpoint)
    reload_model = FSIA_INR_Model(
        IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]), cfg)
    reload_model.load_state_dict(torch.load(
        checkpoint, map_location="cpu", weights_only=True), strict=True)

    report = {
        "status": "passed",
        "scope": "single-batch gradient and 40-step COSMIC-only overfit smoke",
        "coverage": float(has_obs.mean()),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "gradients": gradients,
        "mask_independence_max_abs": mask_independence,
        "cosmic_effect_max_abs": cosmic_effect,
        "deterministic_repeat_max_abs": deterministic_diff,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "old_checkpoint_sha256_before": old_sha_before,
        "old_checkpoint_sha256_after": sha256(OLD_CHECKPOINT),
    }
    assert report["old_checkpoint_sha256_after"] == EXPECTED_OLD_SHA256
    (OUT_DIR / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
