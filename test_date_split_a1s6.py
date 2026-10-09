import json
from pathlib import Path

from inr_modules.data_managers.date_split import apply_a1s6_contract, load_a1s_manifest


def test_a1s6_merges_former_locked_test_without_mutating_source():
    path = Path('isr_validation_outputs/run66-modelonly-m0/date_split_manifest.json')
    source = load_a1s_manifest(path)
    original = json.loads(json.dumps(source))
    effective = apply_a1s6_contract(source)
    assert effective['partitions']['development'] == [3, 8, 17, 22, 29]
    assert effective['partitions']['locked_test'] == []
    assert effective['partitions']['train'] == sorted(
        original['partitions']['train'] + original['partitions']['locked_test'])
    assert effective['utc_dates']['train'] == sorted(
        original['utc_dates']['train'] + original['utc_dates']['locked_test'])
    assert effective['utc_dates']['locked_test'] == []
    assert source == original
    assert set(effective['partitions']['train']) | set(effective['partitions']['development']) == set(range(30))
