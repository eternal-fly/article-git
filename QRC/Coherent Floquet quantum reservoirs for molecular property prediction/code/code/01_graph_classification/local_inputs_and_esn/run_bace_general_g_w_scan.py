#!/usr/bin/env python3
"""BACE/general one-factor g and W scans at validation-chosen Q12/T30.

The scan isolates closed DTC-QRC Floquet-parameter sensitivity.  It scans g at
the manuscript W=0.5 anchor and W at the prespecified g=0.84 anchor.  Every
test point is retained; validation-only selections are supplementary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from decoder_ablation import fit_svr
from protocol import (
    LAYERS,
    OBSERVABLES,
    PROTOCOL_VERSION,
    QRC_BACKEND,
    RESULT_ROOT,
    SEEDS,
    split_path,
    token_cache_path,
)


if str(QRC_BACKEND) not in sys.path:
    sys.path.insert(0, str(QRC_BACKEND))

from gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu  # noqa: E402


SCAN_PROTOCOL = "bace-general-q12-t30-closed-svr-one-factor-g-w-v1"
DATASET = "BACE"
VARIANT = "canonical_general"
Q = 12
T = 30
BASE_G = 0.84
BASE_W = 0.50
G_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.84, 0.90, 0.95, 1.00)
W_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BATCH_SIZE = 32

SCAN_ROOT = RESULT_ROOT / "bace_general_g_w_scan"
FEATURE_ROOT = SCAN_ROOT / "features"
RUN_ROOT = SCAN_ROOT / "runs"


def _tag(prefix: str, value: float) -> str:
    return f"{prefix}{float(value):.2f}".replace(".", "p")


def parameter_pairs() -> list[tuple[float, float]]:
    pairs = [(float(g), BASE_W) for g in G_VALUES]
    pairs.extend((BASE_G, float(w)) for w in W_VALUES if float(w) != BASE_W)
    if len(pairs) != len(G_VALUES) + len(W_VALUES) - 1 or len(set(pairs)) != len(pairs):
        raise AssertionError("invalid one-factor g/W scan plan")
    return pairs


def case_rows(mode: str = "full") -> pd.DataFrame:
    pairs = [(BASE_G, BASE_W)] if mode == "smoke" else parameter_pairs()
    seeds = (0,) if mode == "smoke" else SEEDS
    rows = [
        {
            "dataset": DATASET,
            "token_variant": VARIANT,
            "q": Q,
            "t": T,
            "g": g,
            "W": w,
            "seed": int(seed),
            "on_g_axis": bool(w == BASE_W),
            "on_W_axis": bool(g == BASE_G),
        }
        for g, w in pairs
        for seed in seeds
    ]
    frame = pd.DataFrame(rows)
    expected = len(pairs) * len(seeds)
    if len(frame) != expected or frame.duplicated(["g", "W", "seed"]).any():
        raise RuntimeError("invalid g/W scan cases")
    return frame


def feature_path(row: dict | pd.Series) -> Path:
    return (
        FEATURE_ROOT
        / _tag("g", float(row["g"]))
        / _tag("W", float(row["W"]))
        / f"features_seed{int(row['seed']):03d}.npz"
    )


def result_dir(row: dict | pd.Series) -> Path:
    return (
        RUN_ROOT
        / _tag("g", float(row["g"]))
        / _tag("W", float(row["W"]))
        / "SVR_RBF"
        / f"seed{int(row['seed']):03d}"
    )


def _parameter_hash(params: dict) -> str:
    digest = hashlib.sha256()
    for key in ("J", "h"):
        value = np.ascontiguousarray(params[key])
        digest.update(key.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_inputs() -> tuple[np.ndarray, np.ndarray, str, str]:
    path = token_cache_path(DATASET, VARIANT)
    with np.load(path, allow_pickle=False) as archive:
        tokens = np.ascontiguousarray(
            archive["tokens"][:, :T, : 3 * Q], dtype=np.float32
        )
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if tokens.shape[1:] != (T, 3 * Q):
        raise RuntimeError(f"invalid BACE/general crop {tokens.shape}")
    return tokens, labels, token_sha, str(metadata["dataset_fingerprint"])


def load_feature(row: dict | pd.Series) -> np.ndarray:
    path = feature_path(row)
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        identity = {
            "protocol": str(archive["scan_protocol"].item()),
            "g": float(archive["g"]),
            "W": float(archive["W"]),
            "seed": int(archive["seed"]),
            "q": int(archive["q"]),
            "t": int(archive["t"]),
        }
    expected = {
        "protocol": SCAN_PROTOCOL,
        "g": float(np.float32(row["g"])),
        "W": float(np.float32(row["W"])),
        "seed": int(row["seed"]),
        "q": Q,
        "t": T,
    }
    if identity != expected or features.shape[1] != Q * (Q + 1) // 2:
        raise RuntimeError(f"stale or invalid feature cache {path}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"non-finite feature cache {path}")
    return features


@torch.inference_mode()
def compute_feature(
    row: dict | pd.Series,
    *,
    device: str,
    force: bool = False,
) -> Path:
    path = feature_path(row)
    if path.exists() and not force:
        try:
            load_feature(row)
            print(f"[cached] {path}", flush=True)
            return path
        except Exception:
            pass
    tokens, labels, token_sha, dataset_fingerprint = _load_inputs()
    cfg = DTCQRCConfig(
        num_qubits=Q,
        features_per_qubit=3,
        g=float(row["g"]),
        W_scale=float(row["W"]),
        layers_per_injection=LAYERS,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    features_t, params = compute_qrc_features_gpu(
        torch.from_numpy(tokens),
        cfg,
        seed=int(row["seed"]),
        batch_size=BATCH_SIZE,
        device=device,
        progress=False,
    )
    features = features_t.numpy().astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        features=features,
        labels=labels,
        scan_protocol=np.asarray(SCAN_PROTOCOL),
        base_protocol=np.asarray(PROTOCOL_VERSION),
        dataset=np.asarray(DATASET),
        token_variant=np.asarray(VARIANT),
        token_sha256=np.asarray(token_sha),
        dataset_fingerprint=np.asarray(dataset_fingerprint),
        q=np.int16(Q),
        t=np.int16(T),
        g=np.float32(row["g"]),
        W=np.float32(row["W"]),
        seed=np.int32(row["seed"]),
        layers=np.int16(LAYERS),
        observables=np.asarray(OBSERVABLES),
        parameter_sha256=np.asarray(_parameter_hash(params)),
        J=params["J"],
        h=params["h"],
    )
    partial.replace(path)
    load_feature(row)
    print(
        f"[feature] g={row['g']:.2f} W={row['W']:.2f} "
        f"seed={int(row['seed'])} shape={features.shape}",
        flush=True,
    )
    return path


def evaluate_case(row: dict | pd.Series, *, force: bool = False) -> dict:
    destination = result_dir(row)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not force:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("scan_protocol") == SCAN_PROTOCOL:
            return payload
    features = load_feature(row)
    _, labels, token_sha, _ = _load_inputs()
    split = json.loads(split_path(DATASET, int(row["seed"])).read_text(encoding="utf-8"))
    metrics = fit_svr(
        features,
        labels,
        split,
        seed=int(row["seed"]),
        destination=destination,
        decoder_protocol=SCAN_PROTOCOL,
    )
    payload = {
        "scan_protocol": SCAN_PROTOCOL,
        "base_protocol": PROTOCOL_VERSION,
        "dataset": DATASET,
        "token_variant": VARIANT,
        "q": Q,
        "t": T,
        "g": float(row["g"]),
        "W": float(row["W"]),
        "seed": int(row["seed"]),
        "on_g_axis": bool(row["on_g_axis"]),
        "on_W_axis": bool(row["on_W_axis"]),
        "reservoir": "closed DTC-QRC",
        "observables": OBSERVABLES,
        "token_sha256": token_sha,
        "q_t_selection": (
            "Q12/T30 maximizes six-seed mean validation AUC in the completed "
            "BACE/general fixed-Q/T SVR grid"
        ),
        **metrics,
    }
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def collect(strict: bool = True) -> pd.DataFrame:
    rows = []
    missing = []
    for row in case_rows("full").to_dict("records"):
        path = result_dir(row) / "metrics.json"
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scan_protocol") != SCAN_PROTOCOL:
            raise RuntimeError(f"stale result {path}")
        rows.append(payload)
    if strict and missing:
        raise FileNotFoundError(f"missing {len(missing)} results; first={missing[0]}")
    return pd.DataFrame(rows)


def _validation_select(group: pd.DataFrame, parameter: str) -> pd.Series:
    return group.sort_values(
        ["val_auc", "val_brier", "val_acc", parameter],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).iloc[0]


def summarize(strict: bool = True) -> dict:
    frame = collect(strict=strict)
    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SCAN_ROOT / "metrics_all.csv", index=False)
    anchor = frame[(frame.g == BASE_G) & (frame.W == BASE_W)][
        ["seed", "val_auc", "test_auc"]
    ].rename(columns={"val_auc": "anchor_val_auc", "test_auc": "anchor_test_auc"})
    if len(anchor) != len(SEEDS):
        raise RuntimeError("incomplete g=0.84, W=0.5 anchor")
    summaries = {}
    for axis, parameter, anchor_column in (
        ("g", "g", "on_g_axis"),
        ("W", "W", "on_W_axis"),
    ):
        selected = frame[frame[anchor_column].astype(bool)].copy()
        curve = (
            selected.groupby(parameter, as_index=False)
            .agg(
                val_auc_mean=("val_auc", "mean"),
                val_auc_std=("val_auc", "std"),
                test_auc_mean=("test_auc", "mean"),
                test_auc_std=("test_auc", "std"),
                test_acc_mean=("test_acc", "mean"),
                test_brier_mean=("test_brier", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        curve.to_csv(SCAN_ROOT / f"{axis}_scan_curve_summary.csv", index=False)
        globally_selected = curve.sort_values(
            ["val_auc_mean", parameter],
            ascending=[False, True],
            kind="mergesort",
        ).head(1)
        globally_selected.to_csv(
            SCAN_ROOT / f"validation_selected_{axis}_global.csv", index=False
        )
        paired = selected.merge(anchor, on="seed", validate="many_to_one")
        paired["delta_val_auc_vs_anchor"] = paired.val_auc - paired.anchor_val_auc
        paired["delta_test_auc_vs_anchor"] = paired.test_auc - paired.anchor_test_auc
        paired.to_csv(
            SCAN_ROOT / f"{axis}_scan_paired_vs_anchor_by_seed.csv", index=False
        )
        paired_summary = (
            paired.groupby(parameter, as_index=False)
            .agg(
                mean_delta_val_auc=("delta_val_auc_vs_anchor", "mean"),
                std_delta_val_auc=("delta_val_auc_vs_anchor", "std"),
                mean_delta_test_auc=("delta_test_auc_vs_anchor", "mean"),
                std_delta_test_auc=("delta_test_auc_vs_anchor", "std"),
                favourable_test_seeds=(
                    "delta_test_auc_vs_anchor", lambda x: int(np.sum(x > 0))
                ),
                tied_test_seeds=(
                    "delta_test_auc_vs_anchor", lambda x: int(np.sum(x == 0))
                ),
                n_seeds=("seed", "nunique"),
            )
        )
        paired_summary.to_csv(
            SCAN_ROOT / f"{axis}_scan_paired_vs_anchor_summary.csv", index=False
        )
        picked = []
        for _, group in selected.groupby("seed"):
            picked.append(_validation_select(group, parameter).to_dict())
        picked_frame = pd.DataFrame(picked)
        picked_frame.to_csv(
            SCAN_ROOT / f"validation_selected_{axis}_by_seed.csv", index=False
        )
        summaries[axis] = {
            "curve_rows": len(curve),
            "selected_rows": len(picked_frame),
            "global_validation_selected_value": float(
                globally_selected.iloc[0][parameter]
            ),
            "global_validation_selected_test_auc_mean": float(
                globally_selected.iloc[0].test_auc_mean
            ),
            "selected_test_auc_mean": float(picked_frame.test_auc.mean()),
        }
    status = {
        "scan_protocol": SCAN_PROTOCOL,
        "q": Q,
        "t": T,
        "g_values": list(G_VALUES),
        "W_values": list(W_VALUES),
        "unique_parameter_points": len(parameter_pairs()),
        "seeds": list(SEEDS),
        "result_rows": len(frame),
        "summaries": summaries,
    }
    (SCAN_ROOT / "summary_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    return status


def audit(strict: bool = True) -> dict:
    errors = []
    valid_features = 0
    valid_results = 0
    for row in case_rows("full").to_dict("records"):
        try:
            load_feature(row)
            valid_features += 1
        except Exception as error:
            errors.append(f"feature {row}: {error}")
        path = result_dir(row) / "metrics.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = [payload.get(key) for key in ("val_auc", "test_auc", "test_acc", "test_brier")]
            if payload.get("scan_protocol") == SCAN_PROTOCOL and np.isfinite(values).all():
                valid_results += 1
            else:
                errors.append(f"invalid result {path}")
        else:
            errors.append(f"missing result {path}")
    expected = len(parameter_pairs()) * len(SEEDS)
    report = {
        "scan_protocol": SCAN_PROTOCOL,
        "expected": expected,
        "valid_features": valid_features,
        "valid_results": valid_results,
        "errors": errors,
    }
    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    (SCAN_ROOT / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if strict and errors:
        raise AssertionError(errors[0])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cases", "features", "evaluate", "all", "summarize", "audit"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    cases = case_rows(args.mode).to_dict("records")
    if args.stage == "cases":
        result = {"cases": len(cases), "unique_parameter_points": len({(r['g'], r['W']) for r in cases})}
    elif args.stage in ("features", "evaluate", "all"):
        if args.stage in ("features", "all"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA unavailable; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
                )
            for index, row in enumerate(cases, start=1):
                compute_feature(row, device=args.device, force=args.force)
                print(f"feature case {index}/{len(cases)}", flush=True)
        if args.stage in ("evaluate", "all"):
            for index, row in enumerate(cases, start=1):
                evaluate_case(row, force=args.force)
                print(f"SVR case {index}/{len(cases)}", flush=True)
        result = {"completed_cases": len(cases)}
    elif args.stage == "summarize":
        result = summarize(strict=args.strict)
    else:
        result = audit(strict=args.strict)
    print(json.dumps({"stage": args.stage, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
