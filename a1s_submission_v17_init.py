"""V17-A0: baseline verification and isolated run creation.

Read-only checks on the V16 delivery, the frozen V15/V14/V13 baselines and the
V17 plan/prompt/review inputs, followed by the creation of an isolated
``a1s_submission_v17/run_<timestamp>`` tree that inherits the V16 sources.

No training, no model forward pass, no bootstrap, no metric recomputation.
"""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parent
V16 = ROOT / 'a1s_submission_v16/run_20260919_224919'
V15 = ROOT / 'build/a1s_submission_v15/run_20260919_202003'
V14 = ROOT / 'build/a1s_submission_v14/run_20260919_180611'
V13 = ROOT / 'build/a1s_submission_v13/run_20260919_025039'
OUTDIR = ROOT / 'build/output'

PLAN = OUTDIR / 'FNDA_v17_revision_plan_v17.1_20260920.md'
PROMPT = OUTDIR / 'FNDA_v17a_execution_prompt_20260920.md'
PLAN_VERIFY = OUTDIR / 'FNDA_v17_plan_verification_v17.1_20260920.json'
REVIEWS = [OUTDIR / 'FNDA_v17_science_review_20260919.md',
           OUTDIR / 'FNDA_v17_language_review_20260919.md',
           OUTDIR / 'FNDA_v17_response_review_20260919.md']

ORIG_TEX = Path('D:/code11/IRI01/IRI03/jgr-paper-review/paper/run65_ieee/run65_agu_final/fsia_inr_run65_agu_en.tex')
ORIG_PDF = Path('D:/code11/IRI01/IRI03/jgr-paper-review/paper/run65_ieee/run65_agu_final/Zhang-FNDA.pdf')
REVIEWER_PDFS = [
    Path('D:/code11/IRI01/IRI03/jgr-paper-review/2026JA035830+(Editor+-+Toshi+Nishimura)_+Decision+Letter/'
         '3_reviewer_attachment_1_1784775202_convrt.pdf'),
    Path('D:/code11/IRI01/IRI03/jgr-paper-review/2026JA035830+(Editor+-+Toshi+Nishimura)_+Decision+Letter/'
         '2_reviewer_attachment_1_1784216780_convrt.pdf'),
]

COPY_FROM_V16 = ['paper', 'sources', 'reading', 'response', 'inputs']
COPY_FILES = ['README_V16.md', 'BUILD.md', 'REVISION_MAP.md', 'SOURCE_MAP_V16.md',
              'SOURCE_DATA_INDEX.md', 'AUTHOR_TODOS.md', 'delivery_manifest.json']
TEX_CACHE_SRC = V16 / 'tex_cache'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(base):
    manifest = json.loads((base / 'delivery_manifest.json').read_text(encoding='utf-8'))
    bad = [n for n, h in manifest['files'].items() if not (base / n).is_file() or sha(base / n) != h]
    return {'path': str(base), 'declared_files': len(manifest['files']), 'mismatches': bad,
            'status': 'zero_difference' if not bad else 'differs',
            'manifest_sha256': sha(base / 'delivery_manifest.json')}


def file_facts(path, copy_to=None):
    p = Path(path)
    rec = {'path': str(p), 'exists': p.is_file()}
    if rec['exists']:
        rec['bytes'] = p.stat().st_size
        rec['sha256'] = sha(p)
        if copy_to is not None:
            copy_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, copy_to)
            rec['copied_to'] = str(copy_to.relative_to(RUN))
    return rec


def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run = ROOT / 'a1s_submission_v17' / f'run_{stamp}'
    for sub in ['paper', 'sources', 'inputs', 'checks', 'previews', 'reading', 'response',
                'data', 'history']:
        (run / sub).mkdir(parents=True, exist_ok=True)

    global RUN
    RUN = run

    checks = {'stage': 'V17-A0', 'run': str(run), 'generated_at': datetime.now().astimezone().isoformat()}

    # ---- 1. Baselines -----------------------------------------------------
    checks['baseline_v16'] = verify_manifest(V16)
    checks['baseline_v15'] = verify_manifest(V15)
    checks['baseline_v14'] = verify_manifest(V14)
    checks['baseline_v13'] = verify_manifest(V13)
    checks['protected'] = {
        'v16': {'path': str(V16), 'files': sum(1 for p in V16.rglob('*') if p.is_file())},
        'v15': {'path': str(V15), 'files': sum(1 for p in V15.rglob('*') if p.is_file())},
        'v14': {'path': str(V14), 'files': sum(1 for p in V14.rglob('*') if p.is_file())},
        'v13': {'path': str(V13), 'files': sum(1 for p in V13.rglob('*') if p.is_file())},
    }

    # ---- 2. V16 delivered PDFs and sources --------------------------------
    checks['v16_deliverables'] = {
        rel: file_facts(V16 / rel) for rel in [
            'paper/manuscript_v16.pdf', 'paper/manuscript_v16.tex', 'paper/body_v15.tex',
            'paper/methods_v14.tex', 'paper/design_v14.tex', 'paper/isr_v13.tex',
            'paper/giro_results_v16.tex',
            'paper/tables_v8/table_giro_three_method_v16.tex',
            'paper/tables_v8/table_profile_window_three_method_v16.tex',
            'paper/supporting_information_v16_zh.pdf', 'paper/supporting_information_v16_zh.tex',
            'paper/figures/figure1_architecture_v16.png',
            'reading/manuscript_v16_zh_reading.md',
        ]}

    # ---- 3. Plan, prompt, verification, reviews ---------------------------
    checks['inputs'] = {}
    for p in [PLAN, PROMPT, PLAN_VERIFY] + REVIEWS:
        dst = run / 'inputs' / p.name
        checks['inputs'][p.name] = file_facts(p, copy_to=dst)
    checks['author_figure1_in_v16'] = file_facts(V16 / 'paper/figures/figure1_architecture_v16.png',
                                                 copy_to=run / 'inputs/figure1_architecture_v16.png')
    checks['author_figure1_in_v16']['v16_sha256'] = sha(V16 / 'paper/figures/figure1_architecture_v16.png')
    checks['original_submission'] = {
        'tex': file_facts(ORIG_TEX), 'pdf': file_facts(ORIG_PDF),
        'reviewer_pdfs': [file_facts(p) for p in REVIEWER_PDFS],
    }

    # ---- 4. Frozen statistics --------------------------------------------
    frozen = {}
    for rel in ['data/peak_statistics/metrics.csv', 'data/peak_statistics/paired_contrasts.csv',
                'data/peak_statistics/coverage.csv',
                'data/support_strata_v14/stratified_metrics.csv',
                'data/support_strata_v14/paired_contrasts.csv',
                'data/support_strata_v14/support_group_counts.csv',
                'data/profile_window_metrics_v15.csv',
                'data/frozen_v12/update_gain_v11.csv',
                'data/frozen_v12/frozen_statistics/paired_contrasts.csv',
                'data/giro_station_coverage_v15.csv']:
        frozen[rel] = file_facts(V15 / rel)
    checks['frozen_statistics'] = frozen

    # ---- 5. Inherit V16 sources ------------------------------------------
    inherited = []
    for name in COPY_FROM_V16:
        src = V16 / name
        if not src.is_dir():
            continue
        dst = run / name
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.aux', '*.log', '*.blg',
                                                      '*.out', '*.xdv', '*.fls', '*.fdb_latexmk',
                                                      '*.synctex.gz', '*.pyc', '*.bbl', '*.bcf',
                                                      '*.run.xml'))
        inherited.append(name)
    for name in COPY_FILES:
        src = V16 / name
        if src.is_file():
            shutil.copy2(src, run / name)
            inherited.append(name)
    checks['inherited_from_v16'] = inherited
    if TEX_CACHE_SRC.is_dir():
        shutil.copytree(TEX_CACHE_SRC, run / 'tex_cache', dirs_exist_ok=True)

    # ---- 6. Confirm the inherited sources are byte-identical -------------
    identical = {}
    for name in inherited:
        src = V16 / name
        dst = run / name
        if src.is_file():
            identical[name] = sha(src) == sha(dst)
        else:
            diffs = []
            for p in sorted(src.rglob('*')):
                if not p.is_file():
                    continue
                rel = p.relative_to(src)
                q = dst / rel
                if not q.is_file() or sha(p) != sha(q):
                    diffs.append(rel.as_posix())
            identical[name] = {'copied': True, 'differences': diffs}
    checks['inherited_identical'] = identical

    (run / 'checks/s0_verification.json').write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (run / 'RUN_ROOT.txt').write_text(str(run) + '\n', encoding='utf-8')
    print(json.dumps({'run': str(run),
                      'v16': checks['baseline_v16']['status'],
                      'v15': checks['baseline_v15']['status'],
                      'v14': checks['baseline_v14']['status'],
                      'v13': checks['baseline_v13']['status'],
                      'inherited': inherited}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
