"""Create the missing local manifest for the forced spatial-dev5 Analysis run."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checkpoints_fsia/run67-m2w-120-500-spatial-dev5-background-analysis/run_manifest.json"
RUN_DIR = ROOT / "checkpoints_fsia/run67-m2w-120-500-spatial-dev5-analysis-forced-epoch05-retry"


def main():
    target = RUN_DIR / "run_manifest.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    summary = json.loads((RUN_DIR / "training_summary.json").read_text(encoding="utf-8"))
    manifest["config"]["save_dir"] = str(RUN_DIR)
    manifest["config"]["resume_ckpt"] = str(RUN_DIR / "resume_background_state.pth")
    manifest["config"]["background_only"] = False
    resolved = manifest["resolved_training"]
    for key in tuple(resolved):
        if key in summary:
            resolved[key] = summary[key]
    gram = summary.get("observation_gram_loss", {})
    if "gradient_calibration" in gram:
        resolved["gram_gradient_calibration"] = gram["gradient_calibration"]
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
