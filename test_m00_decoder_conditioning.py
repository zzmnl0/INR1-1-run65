"""Data-free checks for the explicit M00 decoder conditioning contract."""

import pytest
import torch

from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
from inr_modules.mdia.fsia_model import FSIA_INR_Model


def _config(conditioning="legacy_v1"):
    return {
        "alt_range": (120.0, 500.0),
        "seq_len": 6,
        "basis_dim": 8,
        "sw_hidden_dim": 8,
        "sw_lstm_layers": 1,
        "sw_out_dim": 8,
        "enkf_n_members": 8,
        "enkf_pert_hidden": 8,
        "use_sw_freq": False,
        "background_decoder_conditioning": conditioning,
    }


def _model(conditioning):
    proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 1])
    return FSIA_INR_Model(proxy, _config(conditioning))


def test_legacy_and_explicit_decoder_shapes_and_forward_are_finite():
    legacy = _model("legacy_v1")
    explicit = _model("explicit_geo_lt_v1")
    assert legacy.background_decoder[0].in_features == 8 + 8 + 2
    assert explicit.background_decoder[0].in_features == 8 + 8 + 2 + 7
    coords = torch.tensor([
        [-20.0, -120.0, 250.0, 10.0],
        [20.0, 120.0, 350.0, 30.0],
    ])
    sw = torch.randn(2, 6, 2)
    iri_peak = torch.tensor([[300.0, 11.5], [320.0, 11.0]])
    for model in (legacy, explicit):
        background = model.encode_background(coords, sw, iri_peak=iri_peak)
        assert torch.isfinite(background["ne_bkg"]).all()
        assert torch.max(torch.abs(background["background_residual_raw"])) <= 0.5


def test_unknown_decoder_conditioning_is_rejected():
    with pytest.raises(ValueError, match="background decoder conditioning"):
        _model("not-a-contract")


if __name__ == "__main__":
    test_legacy_and_explicit_decoder_shapes_and_forward_are_finite()
    test_unknown_decoder_conditioning_is_rejected()
    print("ok")
