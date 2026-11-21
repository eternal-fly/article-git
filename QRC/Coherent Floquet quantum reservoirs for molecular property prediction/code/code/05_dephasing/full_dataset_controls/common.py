from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
ROOT = HERE.parents[1]
FINAL = ROOT / "BFS-QRC/FINAL"
MIG = FINAL / "17_hamiltonian_setting_fair_migration"
CONTROL = FINAL / "6_controlled_graph_streams"
HOP = ROOT / "Transition/shnitsel_a01_surface_hopping"
SEEDS = (0, 10, 20, 30, 40, 50)
LAGS = (1, 2, 4, 6, 8, 12, 16)
FAMILIES = ("delayed_motif", "distant_pair")
for folder in (MIG, CONTROL, HOP, ROOT / "spatrace_dtc_qrc/01_algorithm_validation"):
    sys.path.insert(0, str(folder))


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **values)
    partial.replace(path)


def hopping_data():
    with np.load(HOP / "data/processed/a01_windows.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def crossed_interval(values, label, draws=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed("review-bootstrap-v1", label))
    means = np.empty(draws)
    for start in range(0, draws, 512):
        count = min(512, draws - start)
        si = rng.integers(values.shape[0], size=(count, values.shape[0]))
        ti = rng.integers(values.shape[1], size=(count, values.shape[1]))
        means[start:start + count] = values[si[:, :, None], ti[:, None, :]].mean((1, 2))
    return dict(mean=float(values.mean()), ci_low=float(np.quantile(means, .025)),
                ci_high=float(np.quantile(means, .975)))
