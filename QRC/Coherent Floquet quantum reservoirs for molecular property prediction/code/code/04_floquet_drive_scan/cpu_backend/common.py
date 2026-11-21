"""Frozen protocol, source access, and atomic runtime artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOP = ROOT / "Transition/shnitsel_a01_surface_hopping"
ALG = ROOT / "spatrace_dtc_qrc/01_algorithm_validation"
TASK8 = ROOT / "Transition/task8_q12_xz_post_tm3_z2"
SELECTION = ROOT / "BFS-QRC/reviewer_supplement_20260905/hopping/qrc_validation_selected_p.csv"
for path in (HOP, ALG, TASK8):
    sys.path.insert(0, str(path))
sys.path.insert(0, str(HERE))

from a01_core import load_config, TARGET_NAMES, reset_suffix_weights

SEEDS = (0, 10, 20, 30, 40, 50)
GS = tuple(round(.02 * i, 2) for i in range(50))
Q, T, LAYERS, W, ANGLE = 15, 20, 3, .5, .5
READOUTS = ("final", "tm3")
DECODERS = ("linear", "rbf_svr")
VERSION = "hopping-q15-t20-l3-gscan-frozen-p-v1"


def jwrite(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def cwrite(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.csv")
    pd.DataFrame(value).to_csv(temporary, index=False)
    temporary.replace(path)


def nwrite(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pkey(p):
    return f"p{round(float(p)*10000):05d}"


def case_dir(g, seed):
    return HERE / "cases" / f"g{g:.2f}" / f"seed{seed:03d}"


def dataset():
    with np.load(HOP / "data/processed/a01_windows.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def choices():
    path = HERE / "frozen_p.csv"
    return pd.read_csv(path if path.exists() else SELECTION)


def widths():
    result = {0.: 360}
    for row in choices().to_dict("records"):
        p = round(float(row["p"]), 8)
        result[p] = max(result.get(p, 0), 120 if row["readout"] == "final" else 360)
    return dict(sorted(result.items()))


def validate_inputs():
    cfg = load_config()
    assert (cfg.q, cfg.history_steps, cfg.layers, cfg.W, cfg.angle_scale, cfg.seeds) == (Q, T, LAYERS, W, ANGLE, SEEDS)
    data = dataset()
    assert data["tokens"].shape == (2376, 20, 45)
    assert data["targets"].shape == (2376, 4)
    assert np.isfinite(data["tokens"]).all() and np.isfinite(data["targets"]).all()
    assert not data["tokens"][..., 1::3].any()
    assert tuple(data["target_names"].astype(str)) == TARGET_NAMES
    ids = []
    rows = []
    for split, count, trajectories in ((0, 1424, 178), (1, 472, 59), (2, 480, 60)):
        mask = data["split"] == split
        unique, repeats = np.unique(data["trajectory_id"][mask], return_counts=True)
        assert mask.sum() == count and len(unique) == trajectories and np.all(repeats == 8)
        ids.append(set(unique))
        rows.append(dict(split=split, windows=count, trajectories=trajectories, windows_per_trajectory=8))
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i+1, 3))
    selected = choices()
    assert len(selected) == 16 and not selected.duplicated(["readout", "decoder", "target"]).any()
    expected = dict(zip(TARGET_NAMES, (.9, .65, .8, .85)))
    main = selected[selected.readout.eq("final") & selected.decoder.eq("rbf_svr")]
    assert all(np.isclose(r.p, expected[r.target]) for r in main.itertuples())
    return rows


def freeze():
    frozen = HERE / "protocol.json"
    paths = [SELECTION, HOP/"config.json", HOP/"data/processed/a01_windows.npz",
             HOP/"a01_core.py", HOP/"evaluate_strict_axes.py", ALG/"hamiltonian_dtc_qrc.py",
             TASK8/"task8_core.py"]
    paths += [HERE / name for name in ("common.py", "backend.py", "cpu_kernel.cpp", "run.py",
                                      "verify.py", "plot_results.py", "launch.py", "PROTOCOL.md")]
    paths += [HOP / "features" / f"qrc_post_tm3_suffixes_seed{s:03d}.npz" for s in SEEDS]
    fingerprints = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    payload = dict(version=VERSION, q=Q, t=T, layers=LAYERS, W=W, angle_scale=ANGLE,
                   g=GS, seeds=SEEDS, targets=TARGET_NAMES, readouts=READOUTS, decoders=DECODERS,
                   reset_position="between events, absent after final event and in TM3",
                   selected_p=choices().to_dict("records"), input_audit=validate_inputs(),
                   expected_feature_cases=300, expected_metric_rows=9600, source_sha256=fingerprints)
    payload = json.loads(json.dumps(payload))
    if frozen.exists():
        assert json.loads(frozen.read_text()) == payload, "Frozen source/protocol changed"
    else:
        cwrite(HERE / "frozen_p.csv", choices())
        jwrite(frozen, payload)
        cwrite(HERE / "input_audit.csv", payload["input_audit"])
    return payload
