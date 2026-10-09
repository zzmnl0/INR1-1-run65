import json
import numpy as np
import torch

from inr_modules.config_mdia import get_config_mdia
from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model
from inr_modules.mdia.m00_observation import m00_observation_error_loss, observation_stratum_keys
from inr_modules.mdia.m00_residual_experiment import (
    build_empirical_reliability,
    replay_train_loo_localization,
    validate_a1s_manifest,
)


def _manifest():
    return {"partitions": {
        "train": list(range(20)),
        "development": list(range(20, 25)),
        "locked_test": list(range(25, 30)),
    }}


def test_a1s_manifest_and_reliability_are_train_only():
    contract = validate_a1s_manifest(_manifest())
    assert contract["locked_test_used"] is False
    n = 40
    coords = np.column_stack((
        np.zeros(n), np.linspace(-20, 20, n), np.full(n, 250.0),
        np.repeat(np.arange(10), 4),
    ))
    pids = np.repeat(np.arange(20), 2)
    residual = np.sin(np.arange(n) / 5.0) * 0.02
    aacgm = np.full(n, 30.0)
    result = build_empirical_reliability(
        residual, coords, pids, np.full(n, "FY"), aacgm,
        min_profiles=2, min_profile_days=1, min_ess=2,
    )
    assert result.payload["bias"]
    assert result.payload["reliability_weight"]
    assert np.isfinite(result.profile_weights).all()


def test_localization_excludes_self_profile_and_is_3d():
    coords = np.asarray([
        [0.0, 0.0, 250.0, 0.0],
        [0.0, 0.1, 250.0, 0.25],
        [0.0, 0.2, 400.0, 0.5],
    ])
    result = replay_train_loo_localization(
        coords[:2], coords, np.asarray([1, 2]), np.asarray([1, 2, 3]),
    )
    assert result["self_profile_excluded"] is True
    assert result["effective_neighbor_count"]["max"] >= 1


def test_candidate_zero_init_and_scope():
    config = dict(get_config_mdia())
    config.update({
        "m00_candidate_semantics": "m00_decoder_last_trust_region_v1",
        "background_condition_adapter": "disabled",
        "adapter_route": "stopped_permanently",
        "alt_range": (120.0, 500.0),
    })
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1])
    model = FSIA_INR_Model(proxy, config)
    model.capture_m00_decoder_baseline()
    coords = torch.tensor([[0.0, 0.0, 250.0, 4.0], [10.0, 20.0, 350.0, 8.0]])
    sw = torch.zeros(2, config["seq_len"], 2)
    peak = torch.tensor([[300.0, 11.5], [310.0, 11.4]])
    with torch.no_grad():
        out = model.encode_background(coords, sw, iri_peak=peak)
    assert torch.allclose(out["m00_candidate_delta"], torch.zeros_like(out["m00_candidate_delta"]))
    assert torch.allclose(out["ne_bkg"], out["ne_iri"] + out["background_residual"])


def test_profile_reliability_changes_profile_contribution():
    prediction = torch.zeros(4)
    target = torch.ones(4)
    coords = torch.tensor([[0.0, 0.0, 250.0, 0.0], [0.0, 0.1, 250.0, 0.0],
                           [0.0, 0.2, 250.0, 0.0], [0.0, 0.3, 250.0, 0.0]])
    pids = torch.tensor([1, 1, 2, 2])
    key = str(observation_stratum_keys(
        coords.numpy(), "FY", magnetic_sin_i=np.full(4, 0.8))[0])
    base = {"semantics": "m00_train_only_observation_error_v1",
            "bias": {key: 0.0}, "variance": {key: 1.0},
            "reliability_weight": {key: 4.0}}
    loss, _ = m00_observation_error_loss(
        prediction, target, coords, pids, "FY", base,
        magnetic_sin_i=torch.full((4,), 0.8),
        use_representativeness=False, use_localization=False,
    )
    assert torch.isfinite(loss)
