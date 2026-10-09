"""Run explicitly selected baseline checks in a new, isolated output directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
RECORDS = Path(__file__).resolve().parent
TESTS = ["test_resume_schedule.py", "test_m2w_domain.py", "test_v14_gram_calibration.py", "test_qc_profiles.py",
         "test_profile_split_a1s2.py", "test_density_plot_units.py", "test_giro_shared_full30.py", "test_isr_evaluation_qc.py",
         "test_isr_loader_geometry.py", "test_isr_model_query.py", "traditional_static_etkf/test_core.py",
         "traditional_static_etkf/test_runner.py"]


def main():
    backup = json.loads((RECORDS / "backup_manifest.json").read_text("utf-8"))
    assert backup["status"] == "VERIFIED_READABLE_SCOPE_WITH_USER_ACCEPTED_EXCLUSIONS"
    output = RECORDS / "s3_baseline_20261009"
    output.mkdir(exist_ok=False)
    for name in ("temp", "mpl"):
        (output/name).mkdir()
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg",
               TMP=str(output/"temp"), TEMP=str(output/"temp"), MPLCONFIGDIR=str(output/"mpl"),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    commands = [
        ("regression", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(output/"pytest_temp"),
                        "--junitxml", str(output/"regression.xml"), *TESTS]),
        ("real_weights", [sys.executable, str(RECORDS/"s3_capture_model_baseline.py"), str(output/"real_weights")])]
    result = {"created_utc":datetime.now(timezone.utc).isoformat(), "output":str(output), "jobs":[],
              "scope":"Pre-cleanup baseline, no full training or scientific acceptance"}
    for name, cmd in commands:
        print("START " + name, flush=True)
        with (output/(name+".log")).open("wb") as log:
            try:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = "TIMEOUT_600_SECONDS"
        result["jobs"].append({"name":name,"command":cmd,"returncode":code,"log":str(output/(name+".log"))})
        (output/"run_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print("END " + name + " " + str(code),flush=True)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
