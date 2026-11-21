"""Decoder-sensitivity control for the frozen J/coherence feature banks.

BACE uses a train-standardized, validation-selected L2-logistic probe.
SHNITSEL uses validation-selected ridge regression and refits on
train+validation.  No reservoir parameter, p value, event length, or test
result is selected here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from evaluate_ablation import (
    _atomic_csv,
    _load_bank,
    mix_bace_bank,
    mix_hopping_bank,
)
from protocol import (
    BACE_Q,
    BACE_SEEDS,
    BACE_T,
    HOPPING_Q,
    HOPPING_ROOT,
    HOPPING_SEEDS,
    HOPPING_T,
    P_VALUES,
    PROTOCOL_VERSION,
    ROOT,
    VARIANTS,
    bace_split_path,
    bace_token_path,
)


if str(HOPPING_ROOT) not in sys.path:
    sys.path.insert(0, str(HOPPING_ROOT))

from a01_core import (  # noqa: E402
    TARGET_NAMES,
    _best_ridge,
    _refit_ridge,
    load_config,
    regression_metrics,
)


RESULTS = ROOT / "results"
BACE_LOGISTIC_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _fit_bace_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
) -> dict:
    train, validation, test = (
        np.asarray(split[key], dtype=np.int64)
        for key in ("train_indices", "val_indices", "test_indices")
    )
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_validation = scaler.transform(features[validation])
    x_test = scaler.transform(features[test])
    best_key = None
    best_model = None
    best_row = None
    for c_value in BACE_LOGISTIC_C_GRID:
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=int(seed),
        ).fit(x_train, labels[train])
        probability = model.predict_proba(x_validation)[:, 1]
        auc = float(roc_auc_score(labels[validation], probability))
        brier = float(np.mean((probability - labels[validation]) ** 2))
        key = (auc, -brier, -float(c_value))
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_row = {"C": float(c_value), "val_auc": auc, "val_brier": brier}
    if best_model is None or best_row is None:
        raise RuntimeError("no valid logistic candidate")
    probability = best_model.predict_proba(x_test)[:, 1]
    return {
        **best_row,
        "test_auc": float(roc_auc_score(labels[test], probability)),
        "test_brier": float(np.mean((probability - labels[test]) ** 2)),
        "test_acc": float(accuracy_score(labels[test], probability >= 0.5)),
    }


def evaluate_bace_linear() -> Path:
    output = RESULTS / "bace_linear_metrics_by_seed.csv"
    with np.load(bace_token_path(), allow_pickle=False) as package:
        labels = package["labels"].astype(np.int64)
    rows: list[dict] = []
    for seed in BACE_SEEDS:
        split = json.loads(bace_split_path(seed).read_text(encoding="utf-8"))
        for variant in VARIANTS:
            bank = _load_bank("bace", variant.name, seed)
            if bank.shape != (
                BACE_T,
                len(labels),
                BACE_Q * (BACE_Q + 1) // 2,
            ):
                raise ValueError((variant.name, seed, bank.shape))
            for p in P_VALUES:
                metrics = _fit_bace_logistic(
                    mix_bace_bank(bank, p), labels, split, seed=seed
                )
                rows.append(
                    {
                        "task": "BACE",
                        "variant": variant.name,
                        "j_scale": variant.j_scale,
                        "dephasing_strength": variant.dephasing_strength,
                        "seed": seed,
                        "q": BACE_Q,
                        "t": BACE_T,
                        "p": p,
                        "decoder": "l2_logistic",
                        "readout": "final",
                        **metrics,
                        "protocol_version": PROTOCOL_VERSION,
                    }
                )
        print(f"[BACE linear] seed={seed}", flush=True)
    _atomic_csv(pd.DataFrame(rows), output)
    return output


def evaluate_hopping_linear() -> Path:
    metrics_output = RESULTS / "hopping_linear_metrics_by_seed.csv"
    cfg = load_config()
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        targets = package["targets"].astype(np.float64)
        split = package["split"].astype(np.int8)
    train, validation, test = split == 0, split == 1, split == 2
    alpha_grid = tuple(map(float, cfg.payload["decoders"]["ridge_alpha_grid"]))
    metric_rows: list[dict] = []
    for variant in VARIANTS:
        for seed in HOPPING_SEEDS:
            bank = _load_bank("hopping", variant.name, seed)
            expected_width = 3 * HOPPING_Q * (HOPPING_Q + 1) // 2
            if bank.shape != (HOPPING_T, len(targets), expected_width):
                raise ValueError((variant.name, seed, bank.shape))
            for p in P_VALUES:
                for readout in ("final", "tm3"):
                    features = mix_hopping_bank(bank, p, readout)
                    for target_index, target_name in enumerate(TARGET_NAMES):
                        y = targets[:, target_index]
                        selected = _best_ridge(
                            features[train],
                            y[train],
                            features[validation],
                            y[validation],
                            alpha_grid,
                        )
                        prediction = _refit_ridge(
                            features[train],
                            y[train],
                            features[validation],
                            y[validation],
                            features[test],
                            selected["alpha"],
                        )
                        row = {
                            "task": "SHNITSEL_A01",
                            "target": target_name,
                            "variant": variant.name,
                            "j_scale": variant.j_scale,
                            "dephasing_strength": variant.dephasing_strength,
                            "seed": seed,
                            "q": HOPPING_Q,
                            "t": HOPPING_T,
                            "p": p,
                            "decoder": "ridge",
                            "readout": readout,
                            "feature_dim": features.shape[1],
                            **selected,
                            **regression_metrics(y[test], prediction),
                            "protocol_version": PROTOCOL_VERSION,
                        }
                        metric_rows.append(row)
            print(f"[hopping linear] {variant.name} seed={seed}", flush=True)
    _atomic_csv(pd.DataFrame(metric_rows), metrics_output)
    return metrics_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("bace", "hopping", "all"), default="all")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.task in ("bace", "all"):
        print(evaluate_bace_linear())
    if args.task in ("hopping", "all"):
        print(evaluate_hopping_linear())


if __name__ == "__main__":
    main()
