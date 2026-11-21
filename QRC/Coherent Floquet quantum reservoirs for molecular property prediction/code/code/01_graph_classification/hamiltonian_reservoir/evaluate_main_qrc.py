"""Train the missing closed/open QRC readouts on the fixed-Q/T grid.

The Hamiltonian suffix banks are frozen.  Each case trains the prespecified
linear and/or RBF-SVR decoder at p=0 and p=0.10.  Case outputs are resumable;
the top-level tables are rebuilt from every completed case after each run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hamiltonian_qrc import load_suffix_bank, mix_suffix_bank
from protocol import (
    CLOSED_P,
    DATASETS,
    DECODERS,
    G,
    OPEN_P,
    PRIMARY_TOKEN_VARIANT,
    PROTOCOL_VERSION,
    PT_Q,
    PT_T_VALUES,
    Q_VALUES,
    RESULT_ROOT,
    SEEDS,
    T_VALUES,
    TOKEN_VARIANTS,
    W,
    qrc_bank_path,
    source_split_path,
    source_token_path,
)
from readouts import fit_linear_bank, fit_svr


ROOT = RESULT_ROOT / "main_fixed_q_t" / "qrc"
P_VALUES = (CLOSED_P, OPEN_P)


def _csv_tuple(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _case_root(dataset: str, variant: str, q: int, t: int, seed: int, decoder: str) -> Path:
    return (
        ROOT / dataset.upper() / variant / f"Q{int(q):02d}" / f"T{int(t):02d}"
        / f"seed{int(seed):03d}" / decoder
    )


def _load_case(dataset: str, variant: str, q: int, t: int, seed: int):
    token_path = source_token_path(dataset, variant)
    with np.load(token_path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    bank, _ = load_suffix_bank(
        qrc_bank_path(dataset, variant, q, t, seed, g=G, W_value=W),
        q=q,
        t=t,
        seed=seed,
        g=G,
        W_scale=W,
        n_samples=len(labels),
        expected_metadata={
            "dataset": dataset.upper(),
            "token_variant": variant,
            "source_token_sha256": token_sha,
        },
    )
    split = json.loads(source_split_path(dataset, seed).read_text(encoding="utf-8"))
    return bank, labels, split


def _valid_existing(path: Path, decoder: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    table = pd.read_csv(path)
    required = {"dataset", "token_variant", "q", "t", "seed", "decoder", "p", "test_auc"}
    if not required.issubset(table.columns):
        return None
    if set(np.round(table["p"], 8)) != set(np.round(P_VALUES, 8)):
        return None
    if set(table["decoder"]) != {decoder}:
        return None
    return table


def _reuse_pt_case(
    dataset: str, variant: str, q: int, t: int, seed: int, decoder: str
) -> pd.DataFrame | None:
    """Reuse the exact same frozen states/readout already trained in p-by-T.

    The vectorized linear heads are independent, so selecting the p=0/.10 rows
    from the 14-p bank is mathematically identical to training a two-row bank.
    SVR is fitted independently at every p in both scripts.
    """
    if q != PT_Q or t not in PT_T_VALUES:
        return None
    diagnostic = (
        RESULT_ROOT / (
            "p_by_T_diagnostic"
            if variant == PRIMARY_TOKEN_VARIANT
            else "p_by_T_global_prior_diagnostic"
        )
        / dataset.upper() / f"Q{q:02d}" / f"T{t:02d}" / f"seed{seed:03d}"
        / decoder / "metrics.csv"
    )
    if not diagnostic.exists():
        return None
    source = pd.read_csv(diagnostic)
    table = source[np.isclose(source["p"], CLOSED_P) | np.isclose(source["p"], OPEN_P)].copy()
    if set(np.round(table["p"], 8)) != set(np.round(P_VALUES, 8)):
        return None
    table["system"] = np.where(
        np.isclose(table["p"], CLOSED_P), "closed_dtc_qrc", "open_dtc_qrc"
    )
    table["protocol_version"] = PROTOCOL_VERSION
    table["artifact_source"] = str(diagnostic.resolve())
    return table


def evaluate_linear(
    dataset: str, variant: str, q: int, t: int, seed: int, *, device: str, force: bool
) -> pd.DataFrame:
    destination = _case_root(dataset, variant, q, t, seed, "linear")
    table_path = destination / "metrics.csv"
    if not force:
        existing = _valid_existing(table_path, "linear")
        if existing is not None:
            return existing
        reused = _reuse_pt_case(dataset, variant, q, t, seed, "linear")
        if reused is not None:
            _atomic_csv(reused, table_path)
            return reused
    bank, labels, split = _load_case(dataset, variant, q, t, seed)
    features = np.stack([mix_suffix_bank(bank, p) for p in P_VALUES])
    table, artifacts = fit_linear_bank(
        features, labels, split, seed=seed, device=device, row_values=P_VALUES
    )
    table = table.rename(columns={"row_value": "p"})
    for name, value in reversed(
        (
            ("dataset", dataset.upper()),
            ("token_variant", variant),
            ("q", int(q)),
            ("t", int(t)),
            ("seed", int(seed)),
            ("decoder", "linear"),
        )
    ):
        table.insert(0, name, value)
    table["system"] = np.where(
        np.isclose(table["p"], CLOSED_P), "closed_dtc_qrc", "open_dtc_qrc"
    )
    table["protocol_version"] = PROTOCOL_VERSION
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "artifacts.npz",
        weights=artifacts["weights"],
        biases=artifacts["biases"],
        test_indices=artifacts["test_indices"],
        test_probabilities=artifacts["test_probabilities"],
        p_values=np.asarray(P_VALUES, dtype=np.float32),
        protocol_version=np.asarray(PROTOCOL_VERSION),
    )
    _atomic_csv(table, table_path)
    return table


def evaluate_linear_group(
    dataset: str,
    variant: str,
    q: int,
    t_values: tuple[int, ...],
    seed: int,
    *,
    device: str,
    force: bool,
) -> list[pd.DataFrame]:
    """Train all missing T/p affine heads in one equivalent vectorized call."""
    completed: dict[int, pd.DataFrame] = {}
    pending = []
    for t in t_values:
        table_path = _case_root(dataset, variant, q, t, seed, "linear") / "metrics.csv"
        if not force:
            existing = _valid_existing(table_path, "linear")
            if existing is not None:
                completed[t] = existing
                continue
            reused = _reuse_pt_case(dataset, variant, q, t, seed, "linear")
            if reused is not None:
                _atomic_csv(reused, table_path)
                completed[t] = reused
                continue
        pending.append(t)
    if pending:
        feature_rows = []
        labels = split = None
        for t in pending:
            bank, case_labels, case_split = _load_case(dataset, variant, q, t, seed)
            if labels is None:
                labels, split = case_labels, case_split
            elif not np.array_equal(labels, case_labels) or split != case_split:
                raise RuntimeError("labels/split changed within a fixed dataset/seed group")
            feature_rows.extend(mix_suffix_bank(bank, p) for p in P_VALUES)
        raw, artifacts = fit_linear_bank(
            np.stack(feature_rows),
            labels,
            split,
            seed=seed,
            device=device,
            row_values=tuple(float(index) for index in range(len(feature_rows))),
        )
        for t_index, t in enumerate(pending):
            start = t_index * len(P_VALUES)
            stop = start + len(P_VALUES)
            table = raw.iloc[start:stop].copy().reset_index(drop=True)
            table["p"] = P_VALUES
            table = table.drop(columns="row_value")
            for name, value in reversed(
                (
                    ("dataset", dataset.upper()),
                    ("token_variant", variant),
                    ("q", int(q)),
                    ("t", int(t)),
                    ("seed", int(seed)),
                    ("decoder", "linear"),
                )
            ):
                table.insert(0, name, value)
            table["system"] = ["closed_dtc_qrc", "open_dtc_qrc"]
            table["protocol_version"] = PROTOCOL_VERSION
            destination = _case_root(dataset, variant, q, t, seed, "linear")
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination / "artifacts.npz",
                weights=artifacts["weights"][start:stop],
                biases=artifacts["biases"][start:stop],
                test_indices=artifacts["test_indices"],
                test_probabilities=artifacts["test_probabilities"][start:stop],
                p_values=np.asarray(P_VALUES, dtype=np.float32),
                protocol_version=np.asarray(PROTOCOL_VERSION),
            )
            _atomic_csv(table, destination / "metrics.csv")
            completed[t] = table
    return [completed[t] for t in t_values]


def evaluate_svr(
    dataset: str, variant: str, q: int, t: int, seed: int, *, force: bool
) -> pd.DataFrame:
    destination = _case_root(dataset, variant, q, t, seed, "rbf_svr")
    table_path = destination / "metrics.csv"
    if not force:
        existing = _valid_existing(table_path, "rbf_svr")
        if existing is not None:
            return existing
        reused = _reuse_pt_case(dataset, variant, q, t, seed, "rbf_svr")
        if reused is not None:
            _atomic_csv(reused, table_path)
            return reused
    bank, labels, split = _load_case(dataset, variant, q, t, seed)
    rows = []
    for p in P_VALUES:
        metrics, artifacts = fit_svr(mix_suffix_bank(bank, p), labels, split, seed=seed)
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
            {"sample_index": artifacts["test_indices"], "score": artifacts["test_scores"]}
        ).to_csv(p_dir / "test_scores.csv", index=False)
        rows.append(
            {
                "dataset": dataset.upper(),
                "token_variant": variant,
                "q": int(q),
                "t": int(t),
                "seed": int(seed),
                "decoder": "rbf_svr",
                "p": float(p),
                "system": "closed_dtc_qrc" if np.isclose(p, CLOSED_P) else "open_dtc_qrc",
                "protocol_version": PROTOCOL_VERSION,
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    _atomic_csv(table, table_path)
    return table


def consolidate() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(ROOT.glob("*/canonical_*/Q*/T*/seed*/*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no completed cases below {ROOT}")
    full = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    identity = ["dataset", "token_variant", "q", "t", "seed", "decoder", "system", "p"]
    duplicated = full.duplicated(identity, keep=False)
    if duplicated.any():
        raise RuntimeError(f"duplicate main QRC rows: {full.loc[duplicated, identity].head().to_dict('records')}")
    summary = (
        full.groupby(["dataset", "token_variant", "q", "t", "decoder", "system", "p"], as_index=False)
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", "std"),
            val_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "token_variant", "q", "t", "decoder", "p"])
    )
    output = RESULT_ROOT / "main_fixed_q_t"
    _atomic_csv(full, output / "qrc_metrics_all_seeds.csv")
    _atomic_csv(summary, output / "qrc_summary.csv")
    return full, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--variants", default=",".join(TOKEN_VARIANTS))
    parser.add_argument("--q-values", default=",".join(map(str, Q_VALUES)))
    parser.add_argument("--t-values", default=",".join(map(str, T_VALUES)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--decoders", default=",".join(DECODERS))
    parser.add_argument("--linear-device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = tuple(value.upper() for value in _csv_tuple(args.datasets))
    variants = _csv_tuple(args.variants)
    q_values = _csv_tuple(args.q_values, int)
    t_values = _csv_tuple(args.t_values, int)
    seeds = _csv_tuple(args.seeds, int)
    decoders = _csv_tuple(args.decoders)
    total = len(datasets) * len(variants) * len(q_values) * len(t_values) * len(seeds) * len(decoders)
    done = 0
    if "linear" in decoders:
        for dataset in datasets:
            for variant in variants:
                for q in q_values:
                    for seed in seeds:
                        evaluate_linear_group(
                            dataset, variant, q, t_values, seed,
                            device=args.linear_device, force=args.force,
                        )
                        done += len(t_values)
                        print(
                            f"[main-qrc-linear] {done}/{total} "
                            f"{dataset}/{variant}/Q{q}/seed{seed}/all-T",
                            flush=True,
                        )
    if "rbf_svr" in decoders:
        for dataset in datasets:
            for variant in variants:
                for q in q_values:
                    for t in t_values:
                        for seed in seeds:
                            evaluate_svr(dataset, variant, q, t, seed, force=args.force)
                            done += 1
                            print(
                                f"[main-qrc-svr] {done}/{total} "
                                f"{dataset}/{variant}/Q{q}/T{t}/seed{seed}",
                                flush=True,
                            )
    full, summary = consolidate()
    print(f"completed rows={len(full)} summary_rows={len(summary)}", flush=True)


if __name__ == "__main__":
    main()
