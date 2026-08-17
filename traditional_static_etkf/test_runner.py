import numpy as np
import pytest

from traditional_static_etkf.core import GRID, generate_static_ensemble
from traditional_static_etkf.runner import (
    _analyze_points,
    _key,
    _strict_join_canonical,
)


def test_canonical_key_format_and_strict_order(tmp_path):
    canonical = {
        "station": np.asarray(["Jicamarca", "PokerFlat"]),
        "key": np.asarray([_key("Jicamarca", "20240902", 10, 200),
                            _key("Poker Flat", "20240902", 20, 220)]),
        "observation_log10": np.asarray([1.0, 2.0]),
        "M11_log10": np.asarray([3.0, 4.0]),
        "M00_log10": np.asarray([5.0, 6.0]),
        "IRI_log10": np.asarray([7.0, 8.0]),
        "altitude_km": np.asarray([200.0, 220.0]),
        "unit_id": np.asarray([10, 20]),
    }
    canonical_path = tmp_path / "canonical.npz"
    np.savez(canonical_path, **canonical)
    shards = {
        "station": np.asarray(["PokerFlat", "Jicamarca"]),
        "date": np.asarray(["20240902", "20240902"]),
        "timestamp": np.asarray([20, 10]),
        "altitude": np.asarray([220.0, 200.0]),
        "lat": np.asarray([1.0, 2.0]),
        "lon": np.asarray([3.0, 4.0]),
        "observation": np.asarray([2.0, 1.0]),
        "key": canonical["key"][::-1],
        "unit_id": np.asarray([20, 10]),
        "iri_grid": np.asarray([8.1, 7.1]),
        "etkf": np.asarray([8.2, 7.2]),
        "spread": np.asarray([.2, .1]),
        "fy_precision": np.asarray([1., 2.]),
        "cosmic_precision": np.asarray([3., 4.]),
        "status": np.asarray(["dual", "FY-only"]),
    }
    joined = _strict_join_canonical(shards, canonical_path)
    assert joined["key"].tolist() == canonical["key"].tolist()
    assert np.allclose(joined["ETKF_log10"], [7.2, 8.2])
    assert np.allclose(joined["IRI_grid_log10"], [7.1, 8.1])
    with pytest.raises(ValueError, match="duplicate"):
        bad = dict(shards)
        bad["key"] = np.asarray([shards["key"][0], shards["key"][0]])
        _strict_join_canonical(bad, canonical_path)


def test_raw_iri_no_observation_fallback_and_chunk_invariance():
    import torch

    class ConstantIRI(torch.nn.Module):
        def forward(self, values):
            return torch.ones((values.shape[0], 1), dtype=values.dtype)

    field = generate_static_ensemble(np.asarray([.1] * 20), seed=42)
    points = {
        "key": np.asarray(["a", "b"]),
        "station": np.asarray(["Jicamarca", "Jicamarca"]),
        "date": np.asarray(["20240902", "20240902"]),
        "timestamp": np.asarray([0, 1800], dtype=np.int64),
        "altitude": np.asarray([180., 300.]),
        "lat": np.asarray([0., 1.]),
        "lon": np.asarray([0., 359.5]),
        "observation": np.asarray([np.nan, np.nan]),
        "unit_id": np.asarray([0, 1800], dtype=np.int64),
    }
    first, audit_first = _analyze_points(
        points, field, ConstantIRI(), {"FY": None, "COSMIC": None},
        {"FY": np.empty(0, dtype=np.int64), "COSMIC": np.empty(0, dtype=np.int64)},
        node_chunk_size=1)
    second, audit_second = _analyze_points(
        points, field, ConstantIRI(), {"FY": None, "COSMIC": None},
        {"FY": np.empty(0, dtype=np.int64), "COSMIC": np.empty(0, dtype=np.int64)},
        node_chunk_size=64)
    assert np.allclose(first["etkf"], second["etkf"])
    assert np.allclose(first["iri_grid"], 1.0)
    assert np.allclose(first["etkf"], first["iri_grid"])
    assert np.allclose(first["fy_precision"], 0.0)
    assert np.all(first["status"] == "zero")
    assert audit_first["fallback_nodes"] == audit_second["fallback_nodes"]


def test_200_km_observation_has_low_altitude_cross_covariance():
    import torch

    class ConstantIRI(torch.nn.Module):
        def forward(self, values):
            return torch.ones((values.shape[0], 1), dtype=values.dtype)

    class OneTokenIndex:
        def query_observation_batch(self, qcoords, allowed_profile_ids=None):
            count = len(qcoords)
            return {
                "coords": np.column_stack([
                    qcoords[:, 0], qcoords[:, 1], np.full(count, 200.0), qcoords[:, 3]
                ]).astype(np.float32),
                "value": np.full(count, 2.0, dtype=np.float32),
                "valid_mask": np.ones(count, dtype=bool),
                "localization_weight": np.ones(count, dtype=np.float32),
                "profile_id": np.arange(count, dtype=np.int64),
                "token_id": np.arange(count, dtype=np.int64),
                "query_index": np.arange(count, dtype=np.int64),
            }

    field = generate_static_ensemble(np.asarray([.1] * 20), seed=42)
    points = {
        "key": np.asarray(["low", "obs"]),
        "timestamp": np.asarray([0, 0], dtype=np.int64),
        "relative_hours": np.asarray([0., 0.]),
        "altitude": np.asarray([180., 200.]),
        "lat": np.asarray([0., 0.]),
        "lon": np.asarray([0., 0.]),
    }
    result, _ = _analyze_points(
        points, field, ConstantIRI(), {"FY": OneTokenIndex(), "COSMIC": None},
        {"FY": np.asarray([1]), "COSMIC": np.empty(0, dtype=np.int64)})
    assert np.all(result["status"] == "FY-only")
    assert abs(result["etkf"][0] - result["iri_grid"][0]) > 1e-6
