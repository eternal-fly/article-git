#!/usr/bin/env python3
"""Connected-ZZ diagnostic for the canonical-v3 Q6/T6/L3 ideal/hardware g scan."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_ideal_observable_gscan as base


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "connected_correlation_training_results"
FAMILY_COLUMNS = {"Z": 6, "CZZ": 15, "Z+CZZ": 21}
PROTOCOL = "canonical-v3-q6t6l3-connected-zz-linear-v1"
HARDWARE_FILES = {
    0.40: tuple(
        HERE / "repeat_observable_training_results" / "features"
        / ("hardware_sl_g040_z_zz.npz" if repeat == 1 else f"hardware_sl_g040_r{repeat}_z_zz.npz")
        for repeat in (1, 2, 3)
    ),
    0.60: tuple(
        HERE / "repeat_observable_training_results" / "features"
        / ("hardware_sl_g060_z_zz.npz" if repeat == 1 else f"hardware_sl_g060_r{repeat}_z_zz.npz")
        for repeat in (1, 2, 3)
    ),
    0.84: tuple(
        base.QRC_ROOT / "Quafu_mainline_Q6T6L3_experiment" / "local_results_five_tasks"
        / f"hardware_features_shenglian_0{repeat}_q6_t6_l3_batch0{repeat}.npz"
        for repeat in (1, 2, 3)
    ),
    1.00: tuple(
        HERE / "repeat_observable_training_results" / "features"
        / ("hardware_sl_g100_z_zz.npz" if repeat == 1 else f"hardware_sl_g100_r{repeat}_z_zz.npz")
        for repeat in (1, 2, 3)
    ),
}


def connected_families(features: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(features, dtype=np.float32)
    if raw.shape != (400, 21):
        raise AssertionError(raw.shape)
    z = raw[:, :6]
    zz = raw[:, 6:]
    products = np.stack(
        [z[:, left] * z[:, right] for left in range(6) for right in range(left + 1, 6)],
        axis=1,
    ).astype(np.float32)
    czz = (zz - products).astype(np.float32)
    return {
        "Z": np.ascontiguousarray(z),
        "CZZ": np.ascontiguousarray(czz),
        "Z+CZZ": np.ascontiguousarray(np.concatenate([z, czz], axis=1)),
    }


def load_hardware(path: Path, bundle: dict) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as package:
        if not np.array_equal(package["indices"], bundle["indices"]):
            raise AssertionError(f"Index mismatch: {path}")
        if not np.array_equal(package["labels"], bundle["labels"]):
            raise AssertionError(f"Label mismatch: {path}")
        if "token_hashes" in package.files and not np.array_equal(
            package["token_hashes"].astype(str), bundle["token_hashes"]
        ):
            raise AssertionError(f"Token-hash mismatch: {path}")
        return package["features"].astype(np.float32)


def train() -> pd.DataFrame:
    bundle = base.load_bundle()
    ideal_matrices = {
        g_value: base.load_ideal(path, g_value, bundle)[0]
        for g_value, path in base.IDEAL_FILES.items()
    }
    hardware_matrices = {
        (g_value, repeat): load_hardware(path, bundle)
        for g_value, paths in HARDWARE_FILES.items()
        for repeat, path in enumerate(paths, start=1)
    }
    rows = []
    for seed in base.SEEDS:
        split = base.common.split_rows(bundle, seed)
        specifications = []
        for g_value, matrix in ideal_matrices.items():
            for family, features in connected_families(matrix).items():
                specifications.append(
                    {"domain": "ideal", "g": g_value, "repeat": 0,
                     "family": family, "features": features}
                )
        for (g_value, repeat), matrix in hardware_matrices.items():
            for family, features in connected_families(matrix).items():
                specifications.append(
                    {"domain": "hardware", "g": g_value, "repeat": repeat,
                     "family": family, "features": features}
                )
        fitted = base.observable.train_batched_heads(
            specifications, bundle["labels"], split, seed
        )
        for item, result in zip(specifications, fitted):
            fit = result["fit"]
            metric = result["test_metrics"]
            rows.append(
                {
                    "domain": item["domain"],
                    "g": item["g"],
                    "repeat": item["repeat"],
                    "family": item["family"],
                    "feature_dim": item["features"].shape[1],
                    "seed": seed,
                    "n_test": len(split["test"]),
                    "best_epoch": fit["best_epoch"],
                    "val_auc": fit["val_auc"],
                    "val_loss": fit["val_loss"],
                    "val_acc": fit["val_acc"],
                    "test_auc": metric["auc"],
                    "test_loss": metric["loss"],
                    "test_acc": metric["acc"],
                    "readout": f"single torch.nn.Linear({item['features'].shape[1]},1) layer",
                    "weight_decay": 0.0,
                    "selection_rule": base.common.SELECTION_RULE,
                    "protocol": PROTOCOL,
                }
            )
        print(f"[trained] seed={seed:02d}; ideal and 12 hardware feature matrices")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["domain", "g", "repeat", "family", "seed"])
    table.to_csv(OUTPUT_DIR / "connected_observable_metrics_by_seed.csv", index=False)
    return table


def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ideal = table[table["domain"] == "ideal"]
    ideal_summary = (
        ideal.groupby(["g", "family", "feature_dim"], as_index=False)
        .agg(auc_mean=("test_auc", "mean"), auc_sd=("test_auc", "std"),
             nested_seed_count=("seed", "nunique"))
        .sort_values(["g", "family"])
    )
    ideal_summary.to_csv(OUTPUT_DIR / "ideal_connected_summary.csv", index=False)

    hardware = table[table["domain"] == "hardware"]
    hardware_batch = (
        hardware.groupby(["g", "repeat", "family", "feature_dim"], as_index=False)
        .agg(auc_mean=("test_auc", "mean"), auc_sd_nested=("test_auc", "std"),
             nested_seed_count=("seed", "nunique"))
        .sort_values(["g", "repeat", "family"])
    )
    hardware_batch.to_csv(OUTPUT_DIR / "hardware_connected_batch_means.csv", index=False)
    hardware_summary = (
        hardware_batch.groupby(["g", "family", "feature_dim"], as_index=False)
        .agg(auc_mean=("auc_mean", "mean"), auc_sd=("auc_mean", "std"),
             hardware_batch_count=("repeat", "nunique"))
        .sort_values(["g", "family"])
    )
    hardware_summary.to_csv(OUTPUT_DIR / "hardware_connected_summary.csv", index=False)

    combined = ideal_summary.merge(
        hardware_summary, on=["g", "family", "feature_dim"],
        suffixes=("_ideal", "_hardware")
    )
    combined["hardware_minus_ideal_auc"] = combined["auc_mean_hardware"] - combined["auc_mean_ideal"]
    combined.to_csv(OUTPUT_DIR / "connected_ideal_hardware_retention.csv", index=False)
    return ideal_summary, hardware_batch, hardware_summary


def contrast(summary: pd.DataFrame, batch: bool) -> pd.DataFrame:
    index = ["g", "repeat"] if batch else ["g"]
    wide = summary.pivot_table(index=index, columns="family", values="auc_mean").reset_index()
    wide["CZZ_minus_Z"] = wide["CZZ"] - wide["Z"]
    return wide


def plot(ideal: pd.DataFrame, hardware_batch: pd.DataFrame, hardware: pd.DataFrame) -> None:
    ideal_wide = contrast(ideal, batch=False)
    hardware_wide = contrast(hardware_batch, batch=True)
    hardware_contrast = (
        hardware_wide.groupby("g", as_index=False)
        .agg(CZZ_minus_Z_mean=("CZZ_minus_Z", "mean"),
             CZZ_minus_Z_sd=("CZZ_minus_Z", "std"))
    )
    ideal_seed = pd.read_csv(OUTPUT_DIR / "connected_observable_metrics_by_seed.csv")
    ideal_seed = ideal_seed[ideal_seed["domain"] == "ideal"].pivot_table(
        index=["g", "seed"], columns="family", values="test_auc"
    ).reset_index()
    ideal_seed["CZZ_minus_Z"] = ideal_seed["CZZ"] - ideal_seed["Z"]
    ideal_contrast = ideal_seed.groupby("g", as_index=False).agg(
        CZZ_minus_Z_mean=("CZZ_minus_Z", "mean"),
        CZZ_minus_Z_sd=("CZZ_minus_Z", "std"),
    )
    ideal_wide.to_csv(OUTPUT_DIR / "ideal_connected_contrast.csv", index=False)
    hardware_wide.to_csv(OUTPUT_DIR / "hardware_connected_contrast_by_batch.csv", index=False)

    mpl.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
         "font.size": 7, "axes.linewidth": 0.8, "axes.spines.top": True,
         "axes.spines.right": True, "svg.fonttype": "none", "pdf.fonttype": 42,
         "legend.frameon": True, "legend.facecolor": "white",
         "legend.edgecolor": "#B8B8B8", "legend.framealpha": 1.0,
         "legend.fancybox": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    colors = {"ideal": "#5E60CE", "hardware": "#D95F59"}
    ax = axes[0]
    ideal_c = ideal[ideal["family"] == "CZZ"].sort_values("g")
    hardware_c = hardware[hardware["family"] == "CZZ"].sort_values("g")
    ax.errorbar(ideal_c["g"], ideal_c["auc_mean"], yerr=ideal_c["auc_sd"],
                color=colors["ideal"], marker="o", lw=1.35, ms=4.3, capsize=2.2,
                label="Ideal (10 nested splits)")
    ax.errorbar(hardware_c["g"], hardware_c["auc_mean"], yerr=hardware_c["auc_sd"],
                color=colors["hardware"], marker="s", lw=1.35, ms=4.3, capsize=2.2,
                label="Hardware (3 batches)")
    cpoints = hardware_batch[hardware_batch["family"] == "CZZ"]
    ax.scatter(cpoints["g"], cpoints["auc_mean"], s=9, facecolor="white",
               edgecolor=colors["hardware"], linewidth=0.6, zorder=4)
    ax.axhline(0.5, color="#777777", lw=0.8, ls=(0, (3, 2)))
    ax.axvspan(0.7, 0.9, color="#D9D9D9", alpha=0.28, zorder=-1)
    ax.set(xlabel="Transverse-field parameter, $g$", ylabel="Connected-$ZZ$ test ROC–AUC",
           xlim=(0.35, 1.05), ylim=(0.38, 0.78))
    ax.set_xticks([0.4, 0.6, 0.84, 1.0], ["0.40", "0.60", "0.84", "1.00"])
    ax.legend(loc="lower left", borderpad=0.35)
    ax.text(-0.16, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    ax = axes[1]
    ax.errorbar(ideal_contrast["g"], ideal_contrast["CZZ_minus_Z_mean"],
                yerr=ideal_contrast["CZZ_minus_Z_sd"], color=colors["ideal"],
                marker="o", lw=1.35, ms=4.3, capsize=2.2, label="Ideal")
    ax.errorbar(hardware_contrast["g"], hardware_contrast["CZZ_minus_Z_mean"],
                yerr=hardware_contrast["CZZ_minus_Z_sd"], color=colors["hardware"],
                marker="s", lw=1.35, ms=4.3, capsize=2.2, label="Hardware")
    ax.scatter(hardware_wide["g"], hardware_wide["CZZ_minus_Z"], s=9,
               facecolor="white", edgecolor=colors["hardware"], linewidth=0.6, zorder=4)
    ax.axhline(0.0, color="#777777", lw=0.8, ls=(0, (3, 2)))
    ax.axvspan(0.7, 0.9, color="#D9D9D9", alpha=0.28, zorder=-1)
    ax.set(xlabel="Transverse-field parameter, $g$", ylabel="$\mathrm{AUC}_{C^{ZZ}}-\mathrm{AUC}_{Z}$",
           xlim=(0.35, 1.05), ylim=(-0.22, 0.22))
    ax.set_xticks([0.4, 0.6, 0.84, 1.0], ["0.40", "0.60", "0.84", "1.00"])
    ax.legend(loc="upper left", borderpad=0.35)
    ax.text(-0.16, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=8)
    for current in axes:
        current.tick_params(direction="out", length=3, width=0.8)
        current.grid(axis="y", color="#E8E8E8", lw=0.5, zorder=-2)
        for spine in current.spines.values():
            spine.set_visible(True)
    stem = OUTPUT_DIR / "connected_zz_ideal_hardware_gscan"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(ideal: pd.DataFrame, hardware: pd.DataFrame) -> None:
    lines = [
        "# Connected-ZZ diagnostic", "",
        "Connected correlations are computed as Cij=<ZiZj>-<Zi><Zj> from the same",
        "stored Z and ZZ features; no new hardware jobs or generated data are used.", "",
        "| g | Ideal Z | Ideal CZZ | Hardware Z | Hardware CZZ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for g_value in sorted(base.IDEAL_FILES):
        i = ideal[ideal["g"] == g_value].set_index("family")
        h = hardware[hardware["g"] == g_value].set_index("family")
        lines.append(
            f"| {g_value:.2f} | {i.loc['Z','auc_mean']:.3f} | {i.loc['CZZ','auc_mean']:.3f} "
            f"| {h.loc['Z','auc_mean']:.3f} | {h.loc['CZZ','auc_mean']:.3f} |"
        )
    (OUTPUT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    table = train()
    ideal, hardware_batch, hardware = summarize(table)
    plot(ideal, hardware_batch, hardware)
    write_report(ideal, hardware)
    print(ideal.to_string(index=False))
    print("\nHardware")
    print(hardware.to_string(index=False))


if __name__ == "__main__":
    main()
