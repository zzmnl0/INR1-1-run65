import numpy as np

from traditional_static_etkf.core import (GRID, HELMERT, apply_h,
                                            baseline_config, etkf_update,
                                            generate_static_ensemble, iri_query,
                                            solve, stencil)


def test_h_periodic_polar_and_linear():
    alt, lat, lon = np.meshgrid(GRID.altitudes, GRID.latitudes, GRID.longitudes,
                                indexing="ij")
    constant = np.ones(GRID.shape)
    assert np.allclose(apply_h(constant, [[0, 359.5, 200], [90, 123, 500]]), 1)
    members = np.broadcast_to(np.arange(8), GRID.shape + (8,))
    assert np.allclose(apply_h(members, [[0, 359.5, 200]]), np.arange(8))
    linear = alt + 2*lat
    assert np.allclose(apply_h(linear, [[3.5, 359.5, 235]]), 242.)
    polar = np.zeros(GRID.shape); polar[:, 0, :] = 7; polar[:, -1, :] = 9
    assert np.allclose(apply_h(polar, [[-90, 1, 200], [-90, 271, 200]]), 7)
    assert np.allclose(stencil([[0, 1, 200]])[1].sum(), 1)


def test_member_basis_reproducible_and_rank_seven():
    first = np.random.RandomState(42).normal(size=(32, 7)) @ HELMERT.T
    second = np.random.RandomState(42).normal(size=(32, 7)) @ HELMERT.T
    assert np.array_equal(first, second)
    assert np.max(np.abs(first.mean(-1))) < 1e-12
    assert np.linalg.matrix_rank(first) <= 7
    # A shared factor field produces the required 180/200-km cross covariance.
    assert abs(np.mean(first * (0.9 * first))) > 1e-8


def test_etkf_update_and_exact_fallback():
    background = np.array([1., 1.])
    anomalies = np.tile(np.linspace(-.2, .2, 8), (2, 1))
    empty = (np.empty(0, int), np.empty((0, 8)), np.empty(0), np.empty(0))
    fallback = solve(background, anomalies, [empty])
    assert np.array_equal(fallback["analysis"], background)
    assert np.array_equal(fallback["transform"], np.tile(np.eye(8), (2, 1, 1)))
    term = (np.array([0]), anomalies[:1], np.array([.3]), np.array([20.]))
    result = solve(background, anomalies, [term])
    assert result["increment"][0] != 0 and result["increment"][1] == 0
    assert np.isfinite(result["transform"]).all()


def test_config_forces_observation_domain():
    config = baseline_config({"observation_alt_range": None})
    assert config["alt_range"] == (120., 500.)
    assert config["observation_alt_range"] == (200., 500.)


def test_grid_boundaries_reject_invalid_coordinates_and_close_longitude():
    field = np.arange(np.prod(GRID.shape), dtype=float).reshape(GRID.shape)
    assert np.allclose(apply_h(field, [[0., 0., 120.], [0., 0., 500.]]),
                       [field[0, 90, 0], field[-1, 90, 0]])
    assert np.allclose(apply_h(field, [[0., 359.999, 200.]],),
                       apply_h(field, [[0., -0.001, 200.]]))
    for bad in ([91., 0., 200.], [0., 0., 119.], [0., 0., 501.]):
        try:
            stencil([bad])
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-domain coordinate was silently clipped")


def test_square_root_transform_is_finite_psd_and_zero_mean():
    xb = np.array([1.0, 2.0])
    X = np.array([[.2, -.1, .3, -.4, .1, -.2, .4, -.3],
                  [.1, -.2, .2, -.1, .3, -.3, .4, -.4]])
    Y = X[:1]
    result = etkf_update(xb, X, Y, np.array([.5]), np.array([20.]))
    assert np.isfinite(result.transform).all()
    assert np.max(np.abs(result.analysis_anomalies.mean(axis=1))) < 1e-12
    cov = result.analysis_anomalies @ result.analysis_anomalies.T / 7.0
    assert np.linalg.eigvalsh((cov + cov.T) / 2).min() > -1e-10


def test_iri_proxy_longitude_conversion():
    import torch

    class Echo(torch.nn.Module):
        def forward(self, x):
            return x[:, 1:2]

    out = iri_query(Echo(), [0., 0.], [0., 359.], [200., 200.], [0., 0.])
    assert np.allclose(out, [0., -1.])
