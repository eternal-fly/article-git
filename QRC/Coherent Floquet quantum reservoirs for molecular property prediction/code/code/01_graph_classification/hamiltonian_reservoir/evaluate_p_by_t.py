"""Evaluate a fixed-Q p-by-T diagnostic from Hamiltonian suffix banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hamiltonian_qrc import load_suffix_bank, mix_suffix_bank
from protocol import (
    DECODERS,
    G,
    OPEN_P,
    PRIMARY_TOKEN_VARIANT,
    PROTOCOL_VERSION,
    PT_P_VALUES,
    PT_Q,
    PT_T_VALUES,
    RESULT_ROOT,
    SEEDS,
    W,
    qrc_bank_path,
    source_split_path,
    source_token_path,
)
from readouts import fit_linear_bank, fit_svr


def _result_root(token_variant: str) -> Path:
    """Keep the published general-token path stable and isolate mirrors."""
    if token_variant == PRIMARY_TOKEN_VARIANT:
        return RESULT_ROOT / "p_by_T_diagnostic"
    if token_variant == "canonical_global_prior":
        return RESULT_ROOT / "p_by_T_global_prior_diagnostic"
    raise ValueError(f"unsupported token variant: {token_variant}")


def _csv_tuple(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _load_case(
    dataset: str, t: int, seed: int, token_variant: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    token_path = source_token_path(dataset, token_variant)
    with np.load(token_path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    bank_path = qrc_bank_path(
        dataset, token_variant, PT_Q, t, seed, g=G, W_value=W
    )
    bank, _ = load_suffix_bank(
        bank_path,
        q=PT_Q,
        t=t,
        seed=seed,
        g=G,
        W_scale=W,
        n_samples=len(labels),
        expected_metadata={
            "dataset": dataset.upper(),
            "token_variant": token_variant,
            "source_token_sha256": token_sha,
        },
    )
    split = json.loads(source_split_path(dataset, seed).read_text(encoding="utf-8"))
    return bank, labels, split


def _case_root(dataset: str, t: int, seed: int, token_variant: str) -> Path:
    return (
        _result_root(token_variant) / dataset.upper()
        / f"Q{PT_Q:02d}" / f"T{int(t):02d}"
        / f"seed{int(seed):03d}"
    )


def evaluate_linear(
    dataset: str, t: int, seed: int, *, token_variant: str,
    device: str, force: bool
) -> pd.DataFrame:
    destination = _case_root(dataset, t, seed, token_variant) / "linear"
    table_path = destination / "metrics.csv"
    if table_path.exists() and not force:
        table = pd.read_csv(table_path)
        if set(np.round(table.p, 8)) == set(np.round(PT_P_VALUES, 8)):
            return table
    bank, labels, split = _load_case(dataset, t, seed, token_variant)
    features = np.stack([mix_suffix_bank(bank, p) for p in PT_P_VALUES])
    table, artifacts = fit_linear_bank(
        features,
        labels,
        split,
        seed=seed,
        device=device,
        row_values=PT_P_VALUES,
    )
    table = table.rename(columns={"row_value": "p"})
    table.insert(0, "decoder", "linear")
    table.insert(0, "seed", int(seed))
    table.insert(0, "t", int(t))
    table.insert(0, "q", PT_Q)
    table.insert(0, "token_variant", token_variant)
    table.insert(0, "dataset", dataset.upper())
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "artifacts.npz",
        weights=artifacts["weights"],
        biases=artifacts["biases"],
        test_indices=artifacts["test_indices"],
        test_probabilities=artifacts["test_probabilities"],
        p_values=np.asarray(PT_P_VALUES, dtype=np.float32),
        protocol_version=np.asarray(PROTOCOL_VERSION),
    )
    _atomic_csv(table, table_path)
    return table


def evaluate_svr(
    dataset: str, t: int, seed: int, *, token_variant: str, force: bool
) -> pd.DataFrame:
    destination = _case_root(dataset, t, seed, token_variant) / "rbf_svr"
    table_path = destination / "metrics.csv"
    if table_path.exists() and not force:
        table = pd.read_csv(table_path)
        if set(np.round(table.p, 8)) == set(np.round(PT_P_VALUES, 8)):
            return table
    bank, labels, split = _load_case(dataset, t, seed, token_variant)
    rows = []
    for p in PT_P_VALUES:
        features = mix_suffix_bank(bank, p)
        metrics, artifacts = fit_svr(features, labels, split, seed=seed)
        p_dir = destination / f"p{float(p):.3f}"
        p_dir.mkdir(parents=True, exist_ok=True)
        artifacts["screen"].to_csv(p_dir / "validation_screen.csv", index=False)
        joblib.dump(
            {
                "protocol_version": PROTOCOL_VERSION,
                "scaler": artifacts["scaler"],
                "model": artifacts["model"],
                "seed": int(seed),
                "p": float(p),
            },
            p_dir / "model.joblib",
        )
        pd.DataFrame(
            {
                "sample_index": artifacts["test_indices"],
                "score": artifacts["test_scores"],
            }
        ).to_csv(p_dir / "test_scores.csv", index=False)
        rows.append(
            {
                "dataset": dataset.upper(),
                "token_variant": token_variant,
                "q": PT_Q,
                "t": int(t),
                "seed": int(seed),
                "decoder": "rbf_svr",
                "p": float(p),
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    _atomic_csv(table, table_path)
    return table


def consolidate(
    tables: list[pd.DataFrame], token_variant: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.concat(tables, ignore_index=True)
    keys = ["dataset", "decoder", "t", "seed"]
    baseline = (
        full[np.isclose(full.p, 0.0)][keys + ["test_auc"]]
        .rename(columns={"test_auc": "test_auc_p0"})
    )
    full = full.merge(baseline, on=keys, how="left", validate="many_to_one")
    full["delta_test_auc_vs_p0"] = full.test_auc - full.test_auc_p0
    summary = (
        full.groupby(["dataset", "decoder", "q", "t", "p"], as_index=False)
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", "std"),
            delta_auc_mean=("delta_test_auc_vs_p0", "mean"),
            delta_auc_sd=("delta_test_auc_vs_p0", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "decoder", "t", "p"])
    )
    result_root = _result_root(token_variant)
    result_root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(full, result_root / "metrics_all_seeds.csv")
    _atomic_csv(summary, result_root / "summary_p_by_T.csv")
    return full, summary


def load_all_case_tables(token_variant: str) -> list[pd.DataFrame]:
    """Collect every completed dataset/T/seed/decoder case.

    This prevents a dataset-filtered invocation from replacing the top-level
    consolidated table with only its own subset.
    """
    result_root = _result_root(token_variant)
    paths = sorted(result_root.glob("*/Q12/T*/seed*/*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no case metrics below {result_root}")
    return [pd.read_csv(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="BACE,BBBP")
    parser.add_argument(
        "--token-variant",
        choices=(PRIMARY_TOKEN_VARIANT, "canonical_global_prior"),
        default=PRIMARY_TOKEN_VARIANT,
    )
    parser.add_argument("--t-values", default=",".join(map(str, PT_T_VALUES)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--decoders", default=",".join(DECODERS))
    parser.add_argument("--linear-device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = tuple(value.upper() for value in _csv_tuple(args.datasets))
    t_values = _csv_tuple(args.t_values, int)
    seeds = _csv_tuple(args.seeds, int)
    decoders = _csv_tuple(args.decoders)
    token_variant = args.token_variant
    tables = []
    for dataset in datasets:
        for t in t_values:
            for seed in seeds:
                if "linear" in decoders:
                    tables.append(
                        evaluate_linear(
                            dataset, t, seed, token_variant=token_variant,
                            device=args.linear_device, force=args.force
                        )
                    )
                if "rbf_svr" in decoders:
                    tables.append(
                        evaluate_svr(
                            dataset, t, seed,
                            token_variant=token_variant, force=args.force
                        )
                    )
    # Re-scan case artifacts so sequential dataset invocations accumulate.
    _, summary = consolidate(load_all_case_tables(token_variant), token_variant)
    print(summary.to_string(index=False))
    print(f"main fixed open p remains {OPEN_P:.2f}; no p-by-T cell was selected")


if __name__ == "__main__":
    main()
