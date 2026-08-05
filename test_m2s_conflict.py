import torch

from audit_m2s_conflict import THRESHOLDS, _fold, _match
from run_m2s_h2 import _balanced_order


def test_profile_fold_is_deterministic_and_source_scoped():
    assert _fold('FY', 17) == _fold('FY', 17)
    assert _fold('FY', 17) in {'A', 'B'}
    assert any(_fold('FY', value) != _fold('FY', 17) for value in range(18, 64))


def test_support_matching_uses_only_preregistered_features():
    base = {'cell': 'low_night', 'features': [0.0] * 14,
            'coords': torch.zeros(4, 4), 'residual': torch.zeros(4),
            'date_block': 0}
    fy = dict(base, source='FY', profile_id=1)
    cosmic = dict(base, source='COSMIC', profile_id=2)
    pairs = _match([fy, cosmic])
    assert [(left['profile_id'], right['profile_id']) for left, right, _ in pairs] == [(1, 2)]
    assert THRESHOLDS['matching_caliper_standardized_distance'] == 2.5


def test_h2_order_is_source_cell_balanced():
    records = [{'source': source, 'cell': cell, 'fold': 'A',
                'profile_id': index}
               for index, (source, cell) in enumerate(
                   (source, cell) for source in ('FY', 'COSMIC')
                   for cell in ('low_day', 'low_night', 'high_day', 'high_night'))]
    stream = _balanced_order(records, 'A')
    first_cycle = [next(stream) for _ in range(8)]
    assert len({(row['source'], row['cell']) for row in first_cycle}) == 8
