"""Isolated fixed-grid protocol using the audited original implementation."""
from pathlib import Path
import sys
import os
import json

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "hopping_gscan_fixed_p_20260908"
sys.path.insert(0, str(PRIOR))
import common as base
import backend
from a01_core import (_best_ridge, _best_svr, _refit_ridge, _refit_svr,
                      regression_metrics, mix_suffix_bank)

np, pd = base.np, base.pd
GS, SEEDS = base.GS, base.SEEDS
TARGET_NAMES, READOUTS, DECODERS = base.TARGET_NAMES, base.READOUTS, base.DECODERS
Q, T, LAYERS, W = base.Q, base.T, base.LAYERS, base.W
PS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
VERSION = "hopping-q15-t20-six-fixed-p-gscan-v1"
jwrite, cwrite, nwrite, sha, pkey = base.jwrite, base.cwrite, base.nwrite, base.sha, base.pkey
dataset = base.dataset


def folder_for(g, seed):
    return HERE / "cases" / f"g{g:.2f}" / f"seed{seed:03d}"


def verify_prior():
    base.freeze()
    report = json.loads((PRIOR / "completion_audit.json").read_text())
    assert report["passed"] and report["metric_rows"] == 9600
    assert json.loads((PRIOR / "feature_cache_audit.json").read_text())["passed"]
    assert json.loads((PRIOR / "backend_verification.json").read_text())["passed"]


def freeze():
    verify_prior()
    paths = [HERE / f for f in ("PROTOCOL.md", "grid_common.py", "grid_features.py",
             "grid_run.py", "grid_plot.py", "grid_verify.py")]
    paths += [PRIOR / f for f in ("protocol.json", "metrics_all_seeds.csv", "feature_cache_audit.csv",
                                 "completion_audit.json")]
    prior = pd.read_csv(PRIOR / "metrics_all_seeds.csv")
    reused = prior[np.isclose(prior.p.to_numpy()[:, None], np.asarray(PS)[None, :]).any(axis=1)]
    assert len(reused) == 6900 and reused.groupby(["g", "seed"]).size().eq(23).all()
    value = dict(version=VERSION, p=list(PS), g=list(GS), seeds=list(SEEDS),
                 q=Q, t=T, layers=LAYERS, W=W, input_scale=base.ANGLE,
                 targets=list(TARGET_NAMES), readouts=list(READOUTS), decoders=list(DECODERS),
                 validation_selects="decoder hyperparameters only; no p or g selection",
                 reused_rows=6900, new_rows=21900, total_rows=28800,
                 prior_protocol_sha256=sha(PRIOR / "protocol.json"),
                 sources={str(p.relative_to(base.ROOT)): sha(p) for p in paths})
    path = HERE / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == value, "Frozen protocol/source changed"
    else:
        jwrite(path, value)
        cwrite(HERE / "reuse_inventory.csv", reused)
    return value
