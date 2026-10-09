"""Small contract checks for the A1-S native causal-past run."""
import json
from pathlib import Path

from inr_modules.mdia.train_fsia import _A1S_FROZEN_R, _A1S_FROZEN_R_SOURCE


def test_a1s_native_manifest_is_complete_and_disjoint():
    path = Path("isr_validation_outputs/run66-modelonly-m0/date_split_manifest.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = data["partitions"]
    sets = {key: set(value) for key, value in parts.items()}
    assert {len(value) for value in parts.values()} == {20, 5}
    assert not (sets["train"] & sets["development"])
    assert not (sets["train"] & sets["locked_test"])
    assert not (sets["development"] & sets["locked_test"])
    assert set().union(*sets.values()) == set(range(30))


def test_a1s_frozen_r_contract():
    assert _A1S_FROZEN_R_SOURCE == "a1s_frozen_global_v1"
    assert _A1S_FROZEN_R == {
        "FY": 0.03446704712846726,
        "COSMIC": 0.02256251445677149,
    }
