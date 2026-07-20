"""Run the existing P2 four-mode acceptance against the repaired workspace."""

import importlib.util
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
CHECKPOINT = WORKSPACE / "checkpoints/run65_cosmic_fix_smoke/smoke_overfit_model.pth"
OUTPUT = WORKSPACE / "checkpoints/run65_cosmic_fix_smoke/four_mode"
ACCEPTANCE = Path(
    r"D:\code11\IRI01\IRI03\INR1-2\INR1-3\data\teacher_label_cache.py")


def main():
    spec = importlib.util.spec_from_file_location("run65_p2_acceptance", ACCEPTANCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FSIA_ROOT = WORKSPACE
    module.CHECKPOINT = CHECKPOINT
    module.EXPECTED_CHECKPOINT_SHA256 = module._sha256(CHECKPOINT)
    report = module.run_smoke(OUTPUT, batch_size=256)
    print(json.dumps({
        "status": report["status"],
        "checkpoint": report["checkpoint"],
        "coverage": report["coverage"],
        "deltas": report["deltas"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
