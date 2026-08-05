import hashlib

import numpy as np

from audit_m2l_neighbor_reordering import _legacy_cosmic_selection
from estimate_empirical_covariance import (
    ROW_DTYPE,
    _cell_id,
    _bootstrap_cell,
    _cross_source_gate,
    _deterministic_npz,
    _date_sign_stability,
    _moment_summary,
    _reduce_profile_pairs,
)
from inr_modules.data_managers.FY_dataloader import (
    COSMICNeighborhoodIndex,
    FYNeighborhoodIndex,
    _allowed_profile_mask,
    _normalized_profile_distance,
)


def _rows(target, observation):
    result = np.zeros(len(target), dtype=ROW_DTYPE)
    result['target_id'] = np.arange(len(target))
    result['date'] = np.arange(len(target)) % 30 + 1
    for prefix in ('loc', 'raw'):
        result[f'{prefix}_r'] = target
        result[f'{prefix}_d'] = observation
        result[f'{prefix}_rd'] = target * observation
        result[f'{prefix}_r2'] = target ** 2
        result[f'{prefix}_d2'] = observation ** 2
    result['pair_count'] = 8
    result['pair_mass'] = 4.0
    return result


def test_centered_covariance_sign_and_source_bias():
    target = np.linspace(-2.0, 2.0, 600)
    positive = _moment_summary(_rows(target + 10.0, target - 7.0), 'loc')
    negative = _moment_summary(_rows(target + 10.0, -target - 7.0), 'loc')
    assert positive['correlation'] > 0.999
    assert negative['correlation'] < -0.999


def test_cell_ids_do_not_overflow_int8_inputs():
    values = []
    for pair in range(4):
        for target_alt in range(3):
            for observation_alt in range(3):
                for lt_class in range(3):
                    for rho_bin in range(4):
                        values.append(int(_cell_id(
                            pair, np.int8(target_alt),
                            np.int8(observation_alt), np.int8(lt_class),
                            np.int8(rho_bin))))
    assert values == list(range(432))


def test_independent_profile_bootstrap_contains_zero():
    rng = np.random.default_rng(42)
    target = rng.normal(size=1200)
    observation = rng.normal(size=1200)
    interval = _bootstrap_cell(
        _rows(target, observation), replicates=300, seed=42)[
            'correlation_ci95']
    assert interval[0] < 0.0 < interval[1]


def test_profile_pair_token_duplication_invariance():
    inverse = np.array([0, 0, 1, 1])
    target = np.array([-1.0, 1.0, 2.0, 4.0])
    observation = np.array([-2.0, 2.0, 3.0, 5.0])
    weights = np.array([0.2, 0.8, 0.4, 0.6])
    original = _reduce_profile_pairs(
        inverse, target, observation, weights)
    duplicated = _reduce_profile_pairs(
        np.repeat(inverse, 3),
        np.repeat(target, 3),
        np.repeat(observation, 3),
        np.repeat(weights, 3))
    for expected, actual in zip(original, duplicated):
        assert np.allclose(expected, actual)


def test_sparse_dates_remain_eligible_blocks():
    rows = _rows(
        np.tile([-1.0, 1.0], 15),
        np.tile([-2.0, 2.0], 15))
    rows['date'] = np.repeat(np.arange(1, 16), 2)
    stability = _date_sign_stability(rows, 1.0)
    assert stability == {'eligible_dates': 15, 'sign_fraction': 1.0}


def test_allowed_profiles_are_filtered_before_top_k():
    index = COSMICNeighborhoodIndex.__new__(COSMICNeighborhoodIndex)
    index.dt = index.dlat = index.dlon = 1.0
    index.k_prof = 2
    index.n_alt = 1
    index.prof_sorted_meta = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ], dtype=np.float32)
    index.prof_sorted_abs = np.zeros((3, 1, 5), dtype=np.float32)
    index.prof_sorted_vmask = np.ones((3, 1), dtype=bool)
    index.prof_sorted_ids = np.array([10, 20, 30])
    index._cand_slice = lambda _: (0, 3)
    coords = np.array([[0.0, 0.0, 200.0, 0.0]], dtype=np.float32)
    default = index.query_profiles_only(coords)
    default_explicit = index.query_profiles_only(
        coords, allowed_profile_ids=None)
    restricted = index.query_profiles_only(
        coords, allowed_profile_ids=np.array([20, 30]))
    assert default['sel_ids'][0].tolist() == [10, 20]
    assert np.array_equal(default['sel_ids'], default_explicit['sel_ids'])
    assert np.array_equal(
        default['valid_prof'], default_explicit['valid_prof'])
    assert restricted['sel_ids'][0].tolist() == [20, 30]
    assert _allowed_profile_mask(
        np.array([10, 20, 30]), np.array([20, 30])).tolist() == [
            False, True, True]


def test_cosmic_top_k_uses_same_space_time_distance_as_fy():
    index = COSMICNeighborhoodIndex.__new__(COSMICNeighborhoodIndex)
    index.dt = index.dlat = index.dlon = 1.0
    index.k_prof = 1
    index.n_alt = 1
    index.prof_sorted_meta = np.array([
        [0.0, 0.0, 0.9],
        [0.2, 0.0, 0.0],
    ], dtype=np.float32)
    index.prof_sorted_abs = np.zeros((2, 1, 5), dtype=np.float32)
    index.prof_sorted_vmask = np.ones((2, 1), dtype=bool)
    index.prof_sorted_ids = np.array([10, 20])
    index._cand_slice = lambda _: (0, 2)

    result = index.query_profiles_only(
        np.array([[0.0, 0.0, 200.0, 0.0]], dtype=np.float32))

    assert result['sel_ids'][0, 0] == 20
    assert np.isclose(result['sel_distance'][0, 0], 0.2)
    assert np.allclose(
        _normalized_profile_distance(
            np.array([0.0, 0.2]), np.zeros(2), np.array([0.9, 0.0]),
            1.0, 1.0, 1.0),
        [0.9, 0.2],
    )


def test_fy_and_cosmic_top_k_paths_are_equivalent():
    metadata = np.array([
        [0.0, 0.0, 0.9],
        [0.2, 0.0, 0.0],
        [0.1, 0.0, 0.4],
    ], dtype=np.float32)
    profile_data = np.zeros((3, 1, 5), dtype=np.float32)
    valid = np.ones((3, 1), dtype=bool)
    profile_ids = np.array([10, 20, 30])
    coords = np.array([[0.0, 0.0, 200.0, 0.0]], dtype=np.float32)

    fy = FYNeighborhoodIndex.__new__(FYNeighborhoodIndex)
    fy.dt = fy.dlat = fy.dlon = 1.0
    fy.k_prof = 2
    fy.n_alt = 1
    fy.t_min = 0.0
    fy.bin_size = 1.0
    fy.prof_sorted_meta = metadata
    fy.prof_sorted_abs = profile_data
    fy.prof_sorted_vmask = valid
    fy.prof_sorted_ids = profile_ids
    fy._cand_slice = lambda _lo, _hi: (0, 3)

    cosmic = COSMICNeighborhoodIndex.__new__(COSMICNeighborhoodIndex)
    cosmic.dt = cosmic.dlat = cosmic.dlon = 1.0
    cosmic.k_prof = 2
    cosmic.n_alt = 1
    cosmic.prof_sorted_meta = metadata
    cosmic.prof_sorted_abs = profile_data
    cosmic.prof_sorted_vmask = valid
    cosmic.prof_sorted_ids = profile_ids
    cosmic._cand_slice = lambda _time: (0, 3)

    fy_result = fy.query_profiles_only(coords)
    cosmic_result = cosmic.query_profiles_only(coords)

    assert np.array_equal(fy_result['sel_ids'], cosmic_result['sel_ids'])
    assert np.array_equal(
        fy_result['valid_prof'], cosmic_result['valid_prof'])
    assert np.allclose(
        fy_result['sel_distance'], cosmic_result['sel_distance'])


def test_fy_cosmic_wraparound_exclusion_and_allowlist_are_equivalent():
    metadata = np.array([
        [0.0, -179.9, 0.0],
        [0.0, 179.7, 0.1],
        [0.0, 170.0, 0.0],
    ], dtype=np.float32)
    profile_data = np.zeros((3, 1, 5), dtype=np.float32)
    valid = np.ones((3, 1), dtype=bool)
    profile_ids = np.array([10, 20, 30])
    coords = np.array([[0.0, 179.9, 200.0, 0.0]], dtype=np.float32)
    allowed = np.array([10, 30])
    excluded = np.array([10])

    fy = FYNeighborhoodIndex.__new__(FYNeighborhoodIndex)
    fy.dt = fy.dlat = 1.0
    fy.dlon = 15.0
    fy.k_prof = 2
    fy.n_alt = 1
    fy.t_min = 0.0
    fy.bin_size = 1.0
    fy.prof_sorted_meta = metadata
    fy.prof_sorted_abs = profile_data
    fy.prof_sorted_vmask = valid
    fy.prof_sorted_ids = profile_ids
    fy._cand_slice = lambda _lo, _hi: (0, 3)

    cosmic = COSMICNeighborhoodIndex.__new__(COSMICNeighborhoodIndex)
    cosmic.dt = cosmic.dlat = 1.0
    cosmic.dlon = 15.0
    cosmic.k_prof = 2
    cosmic.n_alt = 1
    cosmic.prof_sorted_meta = metadata
    cosmic.prof_sorted_abs = profile_data
    cosmic.prof_sorted_vmask = valid
    cosmic.prof_sorted_ids = profile_ids
    cosmic._cand_slice = lambda _time: (0, 3)

    kwargs = {
        'exclude_profile_ids': excluded,
        'allowed_profile_ids': allowed,
    }
    fy_result = fy.query_profiles_only(coords, **kwargs)
    cosmic_result = cosmic.query_profiles_only(coords, **kwargs)

    assert np.array_equal(fy_result['sel_ids'], cosmic_result['sel_ids'])
    assert fy_result['sel_ids'][0].tolist() == [30, -1]
    assert np.allclose(
        fy_result['sel_distance'], cosmic_result['sel_distance'])


def test_legacy_cosmic_selection_reproduces_spatial_only_ordering():
    index = COSMICNeighborhoodIndex.__new__(COSMICNeighborhoodIndex)
    index.dt = index.dlat = index.dlon = 1.0
    index.k_prof = 1
    index.n_alt = 1
    index.prof_sorted_meta = np.array([
        [0.0, 0.0, 0.9],
        [0.2, 0.0, 0.0],
    ], dtype=np.float32)
    index.prof_sorted_abs = np.zeros((2, 1, 5), dtype=np.float32)
    index.prof_sorted_vmask = np.ones((2, 1), dtype=bool)
    index.prof_sorted_ids = np.array([10, 20])
    index._cand_slice = lambda _time: (0, 2)
    coords = np.array([[0.0, 0.0, 200.0, 0.0]], dtype=np.float32)

    legacy = _legacy_cosmic_selection(
        index, coords, np.array([10, 20]))
    current = index.query_profiles_only(coords)

    assert legacy[0, 0] == 10
    assert current['sel_ids'][0, 0] == 20


def test_deterministic_npz(tmp_path):
    first = tmp_path / 'first.npz'
    second = tmp_path / 'second.npz'
    arrays = {'b': np.arange(3), 'a': np.array([1.5])}
    _deterministic_npz(first, arrays)
    _deterministic_npz(second, arrays)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()).digest()


def test_cross_source_gate_compares_all_estimable_transpose_cells():
    cells = []
    for cell_id in range(432):
        pair, remainder = divmod(cell_id, 108)
        target_alt, remainder = divmod(remainder, 36)
        observation_alt, remainder = divmod(remainder, 12)
        lt_class, rho_bin = divmod(remainder, 4)
        cells.append({
            'cell_id': cell_id,
            'pair': ('FY->FY', 'FY->COSMIC', 'COSMIC->FY',
                     'COSMIC->COSMIC')[pair],
            'target_altitude': ('120-200', '200-300', '300-500')[target_alt],
            'observation_altitude': (
                '120-200', '200-300', '300-500')[observation_alt],
            'local_time': ('night-night', 'day-day', 'mixed')[lt_class],
            'rho': ('0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1')[rho_bin],
            'estimable': pair in (1, 2),
            'stable': True,
            'localization_pair_mass': 1.0,
            'localized': {'correlation': 0.2},
            'correlation_ci95': [0.1, 0.3],
        })
    gate = _cross_source_gate(cells)
    assert len(gate['comparisons']) == 108
    assert gate['passed']
