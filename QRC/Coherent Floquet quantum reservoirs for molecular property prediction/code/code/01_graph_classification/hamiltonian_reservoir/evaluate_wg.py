"""Evaluate the prespecified closed/open W-g grid with a linear readout.

Every grid cell is retained.  The canonical (g=.84, W=.5) point is evaluated
as a separately tagged anchor and is never inserted into the uniform heat-map
grid or used to select a working point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hamiltonian_qrc import load_suffix_bank, mix_suffix_bank
from protocol import (
    CLOSED_P,
    G,
    OPEN_P,
    PROTOCOL_VERSION,
    RESULT_ROOT,
    SEEDS,
    W,
    WG_DATASET,
    WG_Q,
    WG_T,
    WG_TOKEN_VARIANT,
    WG_VALUES,
    qrc_bank_path,
    source_split_path,
    source_token_path,
)
from readouts import fit_linear_bank


ROOT = RESULT_ROOT / "w_g_heatmaps"


def _csv_tuple(value: str, cast=float) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _case_root(g: float, W_value: float, seed: int, *, anchor: bool) -> Path:
    family = "canonical_anchor" if anchor else "uniform_grid"
    return (
        ROOT / family / f"g{float(g):.3f}_W{float(W_value):.3f}"
        / f"seed{int(seed):03d}"
    )


def evaluate_case(
    g: float,
    W_value: float,
    seed: int,
    *,
    device: str,
    anchor: bool,
    force: bool,
) -> pd.DataFrame:
    destination = _case_root(g, W_value, seed, anchor=anchor)
    table_path = destination / "metrics.csv"
    if table_path.exists() and not force:
        table = pd.read_csv(table_path)
        if set(np.round(table.p, 8)) == {CLOSED_P, OPEN_P}:
            return table

    token_path = source_token_path(WG_DATASET, WG_TOKEN_VARIANT)
    with np.load(token_path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    bank, _ = load_suffix_bank(
        qrc_bank_path(
            WG_DATASET, WG_TOKEN_VARIANT, WG_Q, WG_T, seed,
            g=g, W_value=W_value,
        ),
        q=WG_Q,
        t=WG_T,
        seed=seed,
        g=g,
        W_scale=W_value,
        n_samples=len(labels),
        expected_metadata={
            "dataset": WG_DATASET,
            "token_variant": WG_TOKEN_VARIANT,
            "source_token_sha256": token_sha,
        },
    )
    split = json.loads(source_split_path(WG_DATASET, seed).read_text(encoding="utf-8"))
    probabilities = (CLOSED_P, OPEN_P)
    features = np.stack([mix_suffix_bank(bank, p) for p in probabilities])
    table, artifacts = fit_linear_bank(
        features,
        labels,
        split,
        seed=seed,
        device=device,
        row_values=probabilities,
    )
    table = table.rename(columns={"row_value": "p"})
    table.insert(0, "is_canonical_anchor", bool(anchor))
    table.insert(0, "W", float(W_value))
    table.insert(0, "g", float(g))
    table.insert(0, "seed", int(seed))
    table.insert(0, "t", WG_T)
    table.insert(0, "q", WG_Q)
    table.insert(0, "token_variant", WG_TOKEN_VARIANT)
    table.insert(0, "dataset", WG_DATASET)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "artifacts.npz",
        weights=artifacts["weights"],
        biases=artifacts["biases"],
        test_indices=artifacts["test_indices"],
        test_probabilities=artifacts["test_probabilities"],
        p_values=np.asarray(probabilities, dtype=np.float32),
        protocol_version=np.asarray(PROTOCOL_VERSION),
    )
    _atomic_csv(table, table_path)
    return table


def consolidate() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(ROOT.glob("*/*/seed*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no W-g metrics below {ROOT}")
    full = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = (
        full.groupby(
            ["dataset", "token_variant", "q", "t", "g", "W", "p", "is_canonical_anchor"],
            as_index=False,
        )
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", "std"),
            val_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["is_canonical_anchor", "p", "g", "W"])
    )
    _atomic_csv(full, ROOT / "metrics_all_seeds.csv")
    _atomic_csv(summary, ROOT / "summary_w_g.csv")
    return full, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g-values", default=",".join(map(str, WG_VALUES)))
    parser.add_argument("--W-values", default=",".join(map(str, WG_VALUES)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-anchor", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    g_values = _csv_tuple(args.g_values, float)
    W_values = _csv_tuple(args.W_values, float)
    seeds = _csv_tuple(args.seeds, int)
    for g in g_values:
        for W_value in W_values:
            for seed in seeds:
                evaluate_case(
                    g, W_value, seed,
                    device=args.device, anchor=False, force=args.force,
                )
    if not args.skip_anchor:
        for seed in seeds:
            evaluate_case(G, W, seed, device=args.device, anchor=True, force=args.force)
    _, summary = consolidate()
    print(summary.to_string(index=False))
    print("all W-g test cells retained; no grid cell selected")


if __name__ == "__main__":
    main()
