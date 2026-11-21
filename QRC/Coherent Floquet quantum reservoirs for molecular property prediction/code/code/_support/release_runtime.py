"""Reconstruct original relative paths in a disposable execution directory.

Categorized release files are never used as writable experiment directories.
Only explicitly invoked workflow entry points execute research code.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / 'index/RUNTIME_PATHS.csv'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes directory: {relative}')
    return path


def load_paths():
    with MAP.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def prepare_runtime():
    runtime = Path(os.environ.get('DTC_QRC_RUNTIME_DIR', str(ROOT / 'runs/runtime'))).resolve()
    # The runtime must never be an ancestor of the released code or data.
    if ROOT.is_relative_to(runtime) or any(runtime.is_relative_to(ROOT / n)
                                         for n in ('code', 'data', 'figures', 'index')):
        raise ValueError('Choose a separate scratch directory or runs/runtime.')
    marker = runtime / 'RELEASE_INPUTS.json'
    fingerprint = digest(MAP)
    if marker.exists():
        if json.loads(marker.read_text())['runtime_map_sha256'] != fingerprint:
            raise RuntimeError('This runtime belongs to another release. Set DTC_QRC_RUNTIME_DIR to a new directory.')
        return runtime
    if runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError('Incomplete or occupied runtime. Choose a new empty DTC_QRC_RUNTIME_DIR.')
    rows = load_paths()
    # Check inputs and available space before copying anything.
    needed = 0
    for row in rows:
        source = safe_path(ROOT, row['release_path'])
        if digest(source) != row['sha256']:
            raise RuntimeError(f'Release input changed: {row["release_path"]}')
        needed += source.stat().st_size
    ancestor = runtime.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < needed + 64 * 1024 * 1024:
        raise RuntimeError(f'Runtime preparation needs at least {(needed / 1024**2) + 64:.0f} MiB free; computation needs additional space.')
    runtime.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = safe_path(ROOT, row['release_path'])
        target = safe_path(runtime, row['runtime_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (runtime / 'audit').mkdir(exist_ok=True)
    marker.write_text(json.dumps({'runtime_map_sha256': fingerprint, 'copied_files': len(rows)}, indent=2) + '\n')
    return runtime


WORKFLOWS = {
    'graph': ('tools/run_graph_case.py', ['--system', 'qrc'],
              'Graph classification: --dataset BACE|BBBP --t 30 --seed 0 --decoder rbf_svr|linear --features frozen|fresh'),
    'gap': ('tools/run_hopping_case.py', ['--grid', 'main'],
            'Gap prediction and main reset grid: --g 0.84 --seed 0 --threads 4 [--features-only]'),
    'graph-reset': ('tools/run_graph_case.py', ['--system', 'qrc', '--p', '0', '.01', '.02', '.04', '.06', '.08', '.10', '.12', '.16', '.20', '.30', '.40', '.60', '.80'],
                    'Graph reset scan: --dataset BACE|BBBP --t 6|12|20|30 --seed 0 --decoder rbf_svr|linear'),
    'gap-reset': ('tools/run_hopping_case.py', ['--grid', 'main'],
                  'Gap reset scan: --g 0.84 --seed 0 --threads 4; original 27-p grid, two decoders and two readouts'),
    'drive': ('tools/run_hopping_case.py', ['--grid', 'six'],
              'Floquet drive scan: --g 0.88 --seed 0 --threads 4; six p values, four targets, final/TM3, Ridge/RBF-SVR'),
    'dephasing-k8': ('code/QRC-J-mechanism-diagnostics/experiments/18_coherence_interaction_ablation/run_ablation.py', [],
                     'Original K=8: features --task bace|hopping --variant coherent|dephase_half|event_dephase --seed 0 --device cuda:0; or evaluate --task all'),
    'dephasing-k32': ('code/BFS-QRC/reviewer_supplement_20260905/run_dephasing_full_k.py', [],
                      'Full-dataset dephasing driver: --task bace|hopping|all --device cuda:0 [--features-only|--decoders-only]. Retains original K settings; see documentation.'),
    'hardware': ('code/BFS-QRC/Shenglian_Q6T6L3_g_scan_cloud/train_connected_correlation_gscan.py', [],
                 'Fit the archived hardware connected-correlation comparison locally. No cloud acquisition is performed.'),
    'esn-width': ('tools/run_graph_case.py', ['--system', 'esn'],
                  'ESN width comparison: --dataset BACE|BBBP --t 12|30 --width 78|500|1000|2000 --seed 0'),
    'figures': ('tools/rebuild_figures.py', [], 'Render the ten publication figures from the saved data, vector sources, and diagram code.'),
    'evidence': ('tools/verify_evidence.py', [], 'Recalculate metrics from saved predictions and counts; no training or cloud acquisition.'),
}


def launch(workflow):
    target, defaults, description = WORKFLOWS[workflow]
    args = sys.argv[1:]
    if '--help' in args or '-h' in args or '--describe' in args:
        print(description)
        print('Runtime target:', target)
        print('Default arguments:', ' '.join(defaults) or '(none)')
        print('Full option reference: index/REPRODUCTION.md. Outputs: runs/runtime/.')
        return
    runtime = prepare_runtime()
    script = safe_path(runtime, target)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    print('Execution directory:', runtime, flush=True)
    subprocess.run([sys.executable, str(script), *defaults, *args], cwd=runtime, env=env, check=True)


if __name__ == '__main__':
    print(prepare_runtime())
