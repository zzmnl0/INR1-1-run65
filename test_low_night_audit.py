from types import SimpleNamespace

import numpy as np

from audit_low_night_coverage import (
    _bias_report,
    _attrition_detail,
    _classify,
    _decode_target,
    _encode_pairs,
    _hard_candidates,
    _rate_difference,
    _safe_unique,
    _support_bits,
    _topk_ids,
)


def test_support_bits_distinguish_height_and_day():
    rows = np.array([
        [0.0, 0.0, 150.0, 0.0, 10.0],
        [0.0, 0.0, 250.0, 0.0, 10.0],
        [0.0, 0.0, 150.0, 12.0, 10.0],
    ])
    assert _support_bits(rows).tolist() == [True, True, True]


def test_pair_encoding_and_profile_duplication_invariance():
    original = _encode_pairs([1, 1, 2], [10, 11, 10])
    duplicated = np.repeat(original, 5)
    assert np.array_equal(_safe_unique([original]), _safe_unique([duplicated]))
    assert _decode_target(_safe_unique([original])).tolist() == [1, 1, 2]


def test_hard_window_wraps_longitude():
    meta = np.array([
        [0.0, -179.0, 10.0],
        [0.0, -150.0, 10.0],
    ])
    order = np.array([0, 1])
    state = {
        'name': 'FY',
        'meta': meta,
        'meta_finite': np.ones(2, dtype=bool),
        'meta_time_order': order,
        'meta_sorted_time': meta[:, 2],
    }
    found = list(_hard_candidates(
        np.array([[0.0, 179.0, 150.0, 10.0]]), state))
    assert found[0][1].tolist() == [0]


def test_height_oracle_topk_excludes_unsupported_profiles():
    ids = np.arange(10)
    distances = np.arange(10, dtype=float)
    support = np.array([False, False] + [True] * 8)
    assert _topk_ids(ids, distances, support).tolist() == list(range(2, 10))


def test_rate_difference_uses_aggregate_profile_pairs():
    dtype = np.dtype([
        ('low_num', '<i4'), ('low_den', '<i4'),
        ('control_num', '<i4'), ('control_den', '<i4'),
    ])
    rows = np.array([(5, 10, 8, 10), (5, 10, 8, 10)], dtype=dtype)
    assert np.isclose(_rate_difference(rows), -0.3)


def test_bias_and_decision_priority():
    fy_cache = SimpleNamespace(
        ids=np.arange(250, dtype=np.int64),
        date=np.arange(250, dtype=np.int16) % 30 + 1,
    )
    pairs = np.column_stack([
        np.arange(250),
        np.arange(1000, 1250),
        np.full(250, 0.08),
        np.ones(250),
    ])
    bias, _ = _bias_report(pairs, fy_cache, replicates=50)
    assert bias['confirmed']
    neutral = {
        'estimable': False, 'difference': 0.0, 'ci95': None}
    assert _classify(True, neutral, neutral, 1.0, bias) == (
        '来源偏差或时空代表性问题')
    assert _classify(False, neutral, neutral, 0.0, bias) == '索引实现问题'


def test_attrition_distinguishes_qc_and_sampled_height_loss():
    metadata = {
        'profile_id': np.array([10, 11, 12]),
        'reason_bits': np.array([0, 4, 0]),
        'reason_names': np.array(['points']),
        'reason_masks': np.array([4]),
        'h_cut_km': np.array([210.0, 120.0, 120.0]),
    }
    pair = lambda neighbor: _encode_pairs([1], [neighbor])
    stage = {
        'raw_height_support': _safe_unique([pair(10), pair(11), pair(12)]),
        'qc_pass': _safe_unique([pair(10), pair(12)]),
        'qc_retained_support': _safe_unique([pair(12)]),
        'sampled8_support': np.empty(0, dtype=np.uint64),
    }
    result = _attrition_detail(
        stage, {'metadata': metadata}, 'low_night_near')
    assert result['failed_qc_reason_pair_counts']['points'] == 1
    assert result['h_cut_at_or_above_band_high_pair_fraction'] == 1.0
    assert result['sampled8_height_loss_ordered_pairs'] == 1
