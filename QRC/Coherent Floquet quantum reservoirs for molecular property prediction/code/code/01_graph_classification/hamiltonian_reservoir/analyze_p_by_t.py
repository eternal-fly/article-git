"""Read-only statistical analysis of every p-by-T case artifact."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from hamiltonian_qrc import load_suffix_bank, mix_suffix_bank
from protocol import (
    G,
    OPEN_P,
    PRIMARY_TOKEN_VARIANT,
    PT_Q,
    RESULT_ROOT,
    SEEDS,
    W,
    qrc_bank_path,
    source_split_path,
    source_token_path,
)


ROOT = RESULT_ROOT / "p_by_T_diagnostic"


def load_all() -> pd.DataFrame:
    paths = sorted(ROOT.glob("*/Q12/T*/seed*/*/metrics.csv"))
    if len(paths) != 96:
        raise RuntimeError(f"expected 96 case tables, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["dataset", "decoder", "t", "seed"]
    baseline = (
        frame[np.isclose(frame.p, 0.0)][keys + ["test_auc"]]
        .rename(columns={"test_auc": "test_auc_p0"})
    )
    frame = frame.merge(baseline, on=keys, validate="many_to_one")
    frame["delta_test_auc_vs_p0"] = frame.test_auc - frame.test_auc_p0
    return frame


def bootstrap_ci(values: np.ndarray, seed: int = 20260828) -> tuple[float, float]:
    value = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = value[rng.integers(0, len(value), size=(20_000, len(value)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def operating_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, decoder, t), group in frame.groupby(["dataset", "decoder", "t"]):
        summary = (
            group.groupby("p", as_index=False)
            .agg(
                test_auc=("test_auc", "mean"),
                test_auc_sd=("test_auc", "std"),
                val_auc=("val_auc", "mean"),
                delta=("delta_test_auc_vs_p0", "mean"),
            )
        )
        test_best = summary.loc[summary.test_auc.idxmax()]
        val_best = summary.loc[summary.val_auc.idxmax()]
        p0 = summary[np.isclose(summary.p, 0.0)].iloc[0]
        fixed = summary[np.isclose(summary.p, OPEN_P)].iloc[0]
        fixed_rows = group[np.isclose(group.p, OPEN_P)]
        low, high = bootstrap_ci(fixed_rows.delta_test_auc_vs_p0.to_numpy())
        rows.append(
            {
                "dataset": dataset,
                "decoder": decoder,
                "T": int(t),
                "AUC_p0": p0.test_auc,
                "AUC_p.10": fixed.test_auc,
                "delta_p.10": fixed.delta,
                "delta_p.10_CI_low": low,
                "delta_p.10_CI_high": high,
                "wins_p.10": int((fixed_rows.delta_test_auc_vs_p0 > 0).sum()),
                "best_test_p_descriptive": float(test_best.p),
                "best_test_AUC_descriptive": test_best.test_auc,
                "best_validation_p": float(val_best.p),
                "test_AUC_at_validation_p": float(
                    summary[np.isclose(summary.p, val_best.p)].iloc[0].test_auc
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "decoder", "T"])


def aggregate_fixed_p(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fixed = frame[np.isclose(frame.p, OPEN_P)]
    for (dataset, decoder), group in fixed.groupby(["dataset", "decoder"]):
        # Seed is the inferential unit: first average the four T values inside seed.
        seed_delta = group.groupby("seed").delta_test_auc_vs_p0.mean().to_numpy()
        low, high = bootstrap_ci(seed_delta)
        rows.append(
            {
                "dataset": dataset,
                "decoder": decoder,
                "mean_AUC_p.10": group.test_auc.mean(),
                "mean_delta_p.10": seed_delta.mean(),
                "delta_CI_low": low,
                "delta_CI_high": high,
                "seed_wins": int((seed_delta > 0).sum()),
                "n_seeds": len(seed_delta),
            }
        )
    return pd.DataFrame(rows)


def global_linear_validation(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[frame.decoder == "linear"]
        .groupby("p", as_index=False)
        .agg(
            mean_validation_auc=("val_auc", "mean"),
            mean_test_auc=("test_auc", "mean"),
            mean_test_delta=("delta_test_auc_vs_p0", "mean"),
        )
        .sort_values("mean_validation_auc", ascending=False)
    )


def feature_scale_table() -> pd.DataFrame:
    rows = []
    for dataset in ("BACE", "BBBP"):
        with np.load(source_token_path(dataset, PRIMARY_TOKEN_VARIANT), allow_pickle=False) as archive:
            n_samples = len(archive["labels"])
            token_sha = str(archive["tokens_sha256"].item())
        for t in (6, 12, 20, 30):
            for seed in SEEDS:
                bank, _ = load_suffix_bank(
                    qrc_bank_path(
                        dataset, PRIMARY_TOKEN_VARIANT, PT_Q, t, seed,
                        g=G, W_value=W,
                    ),
                    q=PT_Q,
                    t=t,
                    seed=seed,
                    g=G,
                    W_scale=W,
                    n_samples=n_samples,
                    expected_metadata={
                        "dataset": dataset,
                        "token_variant": PRIMARY_TOKEN_VARIANT,
                        "source_token_sha256": token_sha,
                    },
                )
                split = json.loads(source_split_path(dataset, seed).read_text(encoding="utf-8"))
                train = np.asarray(split["train_indices"], dtype=np.int64)
                base_scale = None
                for p in (0.0, 0.1, 0.8):
                    features = mix_suffix_bank(bank, p)[train]
                    std = features.std(axis=0, ddof=1)
                    scale = float(np.sqrt(np.mean(std**2)))
                    if base_scale is None:
                        base_scale = scale
                    rows.append(
                        {
                            "dataset": dataset,
                            "T": t,
                            "seed": seed,
                            "p": p,
                            "rms_feature_std": scale,
                            "scale_ratio_vs_p0": scale / max(base_scale, 1e-15),
                        }
                    )
    return (
        pd.DataFrame(rows)
        .groupby(["dataset", "T", "p"], as_index=False)
        .agg(
            rms_feature_std=("rms_feature_std", "mean"),
            scale_ratio_vs_p0=("scale_ratio_vs_p0", "mean"),
        )
    )


def main() -> None:
    frame = load_all()
    print("OPERATING TABLE")
    print(operating_table(frame).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFIXED p=.10 AGGREGATED OVER T")
    print(aggregate_fixed_p(frame).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nGLOBAL PRIMARY-LINEAR VALIDATION RANKING")
    print(
        global_linear_validation(frame).head(14).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print("\nFEATURE-SCALE CONTRACTION")
    print(feature_scale_table().to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
