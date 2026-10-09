"""Create a new verified source snapshot; never move or delete source files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
RECORDS = Path(__file__).resolve().parent
SKIP = {".git", ".aws", ".codex", ".agents", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
COPY_EXT = {".py", ".ps1", ".bat", ".cmd", ".sh", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".md", ".txt", ".tex", ".bib", ".cls", ".sty", ".ipynb", ".csv", ".log", ".err", ".diff", ".patch", ".cff",
            ".js", ".mjs", ".html", ".xml", ".pth"}
SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|(?i:ZENODO_ACCESS_TOKEN|GITHUB_TOKEN)\s*=\s*['\"][A-Za-z0-9_-]{30,}")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    assert destination.parent == ROOT.parent and destination.name.startswith(ROOT.name + "_backup_v193_"), destination
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing backup")
    assert shutil.disk_usage(destination.parent).free > 10 * 1024**3
    destination.mkdir()
    state = {"created_utc": datetime.now(timezone.utc).isoformat(), "root": str(ROOT), "destination": str(destination),
             "status": "IN_PROGRESS", "copied": [], "registered_in_place": [], "excluded": [], "unreadable": [], "links": [],
             "scope_decision": "User approved 2026-10-09: unreadable historical temporary directories remain protected in place",
             "run_environment": {"python": sys.version, "executable": sys.executable}, "restore_sample": []}
    sources = [(ROOT, Path("project"))]
    identity = json.loads((RECORDS / "model_identity.json").read_text("utf-8"))
    external = set()
    for item in identity["result_lineage"]:
        for entry in item["generators"]:
            p = Path(entry["path"])
            if not p.is_relative_to(ROOT):
                external.add(p.parent)
    for run in identity["sensitivity_runs"].values():
        external.add(Path(run["run"]).parents[1])
    for p in (Path(r"C:\Users\12238\.codex\skills\nature-figure\scripts"),
              Path(r"C:\Users\12238\.agents\skills\nature-figure\scripts")):
        if p.is_dir():
            external.add(p)
    external = {p for p in external if not any(p != q and p.is_relative_to(q) for q in external)}
    for p in sorted(external):
        sources.append((p, Path("external") / p.drive.rstrip(":") / Path(*p.parts[1:])))
    state["source_roots"] = [{"source":str(p),"snapshot_prefix":str(q)} for p,q in sources]
    seen = set()
    processed = 0
    for source_root, prefix in sources:
        for folder, dirs, files in os.walk(source_root, followlinks=False, onerror=lambda e: state["unreadable"].append({"path":e.filename,"error":str(e)})):
            retained = []
            for name in dirs:
                p = Path(folder) / name
                if name in SKIP:
                    state["excluded"].append({"path":str(p),"reason":"metadata/credentials/runtime cache; git captured separately"})
                elif p.is_symlink() or p.is_junction():
                    state["links"].append({"path":str(p),"target":str(p.resolve())})
                else:
                    retained.append(name)
            dirs[:] = retained
            for name in files:
                p = Path(folder) / name
                if p in seen:
                    continue
                seen.add(p)
                if p.is_symlink():
                    state["links"].append({"path":str(p),"target":str(p.resolve())})
                    continue
                if name.lower().startswith(".env") or p.suffix.lower() in {".pem", ".key"} or re.search(r"(?i)(access[_-]?token|credentials|secret)[^.]*\.(txt|json|ini)$", name):
                    state["excluded"].append({"path":str(p),"reason":"credential filename; not copied or content-logged"})
                    continue
                try:
                    before = p.stat()
                    copy = p.suffix.lower() in COPY_EXT or (p.suffix == "" and name in {"LICENSE", "requirements", ".gitignore", ".gitattributes"})
                    if copy and p.suffix.lower() != ".pth" and before.st_size < 16 * 1024**2 and SECRET.search(p.read_bytes()):
                        state["excluded"].append({"path":str(p),"reason":"credential pattern; not copied or content-logged"})
                        continue
                    sha = digest(p)
                    record = {"path":str(p),"bytes":before.st_size,"mtime_ns":before.st_mtime_ns,"sha256":sha}
                    if copy:
                        target = destination / prefix / p.relative_to(source_root)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(p,target)
                        if digest(target) != sha:
                            raise RuntimeError(f"Copy hash mismatch: {p}")
                        record["snapshot"] = str(target.relative_to(destination))
                        state["copied"].append(record)
                    else:
                        state["registered_in_place"].append(record)
                    after = p.stat()
                    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                        raise RuntimeError(f"Source changed during snapshot: {p}")
                except PermissionError as exc:
                    state["unreadable"].append({"path":str(p),"error":str(exc)})
                processed += 1
                if processed % 5000 == 0:
                    print(json.dumps({"files_processed":processed,"copied":len(state["copied"])}),flush=True)
    # External training inputs are intentionally registered, not copied as a new dataset.
    for item in identity["training_data_identity"]:
        p = Path(item["path"])
        if p not in seen:
            state["registered_in_place"].append({"path":str(p),"bytes":p.stat().st_size,"sha256":digest(p),"role":"external training input"})
    gitdir = destination / "git_state"
    gitdir.mkdir()
    for filename, command in (("head.txt",["rev-parse","HEAD"]),("status.txt",["status","--short","--untracked-files=all"]),
                              ("unstaged.patch",["diff","--binary"]),("staged.patch",["diff","--cached","--binary"]),
                              ("tracked_files.txt",["ls-files"])):
        completed = subprocess.run(["git",*command],cwd=ROOT,capture_output=True,check=True)
        if SECRET.search(completed.stdout):
            raise RuntimeError("Git export includes credential pattern; snapshot retained for restricted review")
        (gitdir/filename).write_bytes(completed.stdout)
    subprocess.run(["git","bundle","create",str(gitdir/"history.bundle"),"--all"],cwd=ROOT,check=True,capture_output=True)
    verified = subprocess.run(["git","bundle","verify",str(gitdir/"history.bundle")],cwd=ROOT,check=True,capture_output=True)
    (gitdir/"bundle_verify.log").write_bytes(verified.stdout+verified.stderr)
    env = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,check=True)
    (destination/"requirements_observed.txt").write_bytes(env.stdout)
    sample_root = destination / "restore_sample"
    sample_root.mkdir()
    required = {ROOT/"main_fsia.py",ROOT/"inr_modules/config_mdia.py",ROOT/"inr_modules/mdia/fsia_model.py",ROOT/"inr_modules/mdia/train_fsia.py"}
    for row in state["copied"]:
        if Path(row["path"]) in required:
            sample = sample_root / Path(row["path"]).relative_to(ROOT)
            sample.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(destination/row["snapshot"],sample)
            assert digest(sample)==row["sha256"]
            state["restore_sample"].append({"source":row["snapshot"],"restored":str(sample),"sha256":row["sha256"]})
    assert len(state["restore_sample"])==4
    state["status"] = "VERIFIED_READABLE_SCOPE_WITH_USER_ACCEPTED_EXCLUSIONS"
    state["counts"] = {k:len(state[k]) for k in ("copied","registered_in_place","unreadable","excluded","links")}
    state["copied_bytes"] = sum(x["bytes"] for x in state["copied"])
    text = json.dumps(state,ensure_ascii=False,indent=2)+"\n"
    (destination/"backup_manifest.json").write_text(text,encoding="utf-8")
    (RECORDS/"backup_manifest.json").write_text(text,encoding="utf-8")
    print(json.dumps({"status":state["status"],"destination":str(destination),"counts":state["counts"],"copied_bytes":state["copied_bytes"]}),flush=True)


if __name__ == "__main__":
    main()
