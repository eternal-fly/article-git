#!/usr/bin/env python3
"""Train ideal Q6/T6/L3 Z, ZZ, and Z+ZZ readouts across the hardware g scan."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
QRC_ROOT = HERE.parent
COMMON_DIR = QRC_ROOT / "Quafu_mainline_Q6T6L3_experiment" / "local"
DATASET_PATH = HERE / "bace_bal400_canonical_v3_q6_t6.npz"
OUTPUT_DIR = HERE / "ideal_observable_training_results"
HARDWARE_DIR = HERE / "repeat_observable_training_results"
IDEAL_FILES = {
    0.40: HERE / "local_training_results" / "ideal_g040_z_zz.npz",
    0.60: HERE / "local_training_results" / "ideal_g060_z_zz.npz",
    0.84: QRC_ROOT / "PRAPPLIED_used_data_only" / "07_hardware_mainline"
    / "input_data" / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz",
    1.00: HERE / "local_training_results" / "ideal_g100_z_zz.npz",
}
FAMILIES = {
    "Z": np.arange(0, 6, dtype=np.int64),
    "ZZ": np.arange(6, 21, dtype=np.int64),
    "Z+ZZ": np.arange(0, 21, dtype=np.int64),
}
SEEDS = tuple(range(0, 100, 10))
EXPECTED_DATASET_SHA256 = "00b3fdbab64cb95f61b8580bd7e975ee9f1c9d7df1b44ce60ebe61a643a3eb04"
PROTOCOL = "canonical-v3-q6t6l3-ideal-gscan-observable-linear-v1"

sys.path.insert(0, str(COMMON_DIR))
import run_local_analysis as common  # noqa: E402
import run_observable_family_ablation as observable  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value: object) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def load_bundle() -> dict:
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise AssertionError("Dataset SHA-256 differs from the hardware scan")
    with np.load(DATASET_PATH, allow_pickle=False) as package:
        bundle = {
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "token_hashes": package["token_hashes"].astype(str),
            "splits": json.loads(scalar_text(package["split_indices_json"])),
        }
    if len(bundle["indices"]) != 400:
        raise AssertionError(len(bundle["indices"]))
    common.DATASET_PATH = DATASET_PATH
    return bundle


def load_ideal(path: Path, expected_g: float, bundle: dict) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as package:
        if not np.array_equal(package["indices"], bundle["indices"]):
            raise AssertionError(f"Index mismatch in {path}")
        if not np.array_equal(package["labels"], bundle["labels"]):
            raise AssertionError(f"Label mismatch in {path}")
        if not np.array_equal(package["token_hashes"].astype(str), bundle["token_hashes"]):
            raise AssertionError(f"Token-hash mismatch in {path}")
        features = package["features"].astype(np.float32)
        if features.shape != (400, 21):
            raise AssertionError(features.shape)
        if "metadata_json" in package.files:
            metadata = json.loads(scalar_text(package["metadata_json"]))
            observed_g = float(metadata["g"])
            observed_w = float(metadata["W"])
            q, t, layers = int(metadata["q"]), int(metadata["t"]), int(metadata["layers"])
            reservoir_seed = int(metadata["reservoir_seed"])
            dataset_sha = str(metadata["dataset_sha256"])
        else:
            observed_g = float(package["g"])
            observed_w = float(package["W"])
            q, t, layers = 6, 6, 3
            reservoir_seed = int(package["reservoir_seed"])
            dataset_sha = EXPECTED_DATASET_SHA256
        if not np.isclose(observed_g, expected_g, atol=1e-6):
            raise AssertionError((observed_g, expected_g))
        if not np.isclose(observed_w, 0.60, atol=1e-6):
            raise AssertionError(observed_w)
        if (q, t, layers, reservoir_seed) != (6, 6, 3, 0):
            raise AssertionError((q, t, layers, reservoir_seed))
        if dataset_sha != EXPECTED_DATASET_SHA256:
            raise AssertionError(dataset_sha)
        audit = {
            "g": expected_g,
            "source_sha256": sha256_file(path),
            "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "shape": list(features.shape),
            "W": observed_w,
            "q": q,
            "t": t,
            "layers": layers,
            "reservoir_seed": reservoir_seed,
            "dataset_sha256": dataset_sha,
        }
        return features, audit


def train_all() -> pd.DataFrame:
    bundle = load_bundle()
    rows: list[dict] = []
    audits: list[dict] = []
    matrices: dict[float, np.ndarray] = {}
    for g_value, path in IDEAL_FILES.items():
        features, audit = load_ideal(path, g_value, bundle)
        audits.append(audit)
        matrices[g_value] = features

    for seed in SEEDS:
        split = common.split_rows(bundle, seed)
        specifications = []
        for g_value in IDEAL_FILES:
            features = matrices[g_value]
            for family, columns in FAMILIES.items():
                selected = np.ascontiguousarray(features[:, columns], dtype=np.float32)
                specifications.append(
                    {
                        "g": g_value,
                        "family": family,
                        "features": selected,
                    }
                )
        fitted = observable.train_batched_heads(
            specifications, bundle["labels"], split, seed
        )
        for item, result in zip(specifications, fitted):
            g_value = float(item["g"])
            family = str(item["family"])
            selected = item["features"]
            fit = result["fit"]
            metric = result["test_metrics"]
            rows.append(
                {
                    "g": g_value,
                    "family": family,
                    "feature_dim": selected.shape[1],
                    "seed": seed,
                    "n_test": len(split["test"]),
                    "best_epoch": fit["best_epoch"],
                    "val_auc": fit["val_auc"],
                    "val_loss": fit["val_loss"],
                    "val_acc": fit["val_acc"],
                    "test_auc": metric["auc"],
                    "test_loss": metric["loss"],
                    "test_acc": metric["acc"],
                    "readout": f"single torch.nn.Linear({selected.shape[1]},1) layer",
                    "weight_decay": 0.0,
                    "selection_rule": common.SELECTION_RULE,
                    "implementation": "paper batched-independent-head trainer",
                    "protocol": PROTOCOL,
                }
            )
        print(f"[trained] seed={seed:02d}; 4 g points x 3 observable families")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["g", "family", "seed"])
    table.to_csv(OUTPUT_DIR / "ideal_observable_metrics_by_seed.csv", index=False)
    (OUTPUT_DIR / "ideal_feature_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    return table


def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        table.groupby(["g", "family", "feature_dim"], as_index=False)
        .agg(
            auc_mean=("test_auc", "mean"),
            auc_sd_nested_seeds=("test_auc", "std"),
            loss_mean=("test_loss", "mean"),
            acc_mean=("test_acc", "mean"),
            nested_seed_count=("seed", "nunique"),
        )
        .sort_values(["g", "family"])
    )
    summary.to_csv(OUTPUT_DIR / "ideal_observable_summary.csv", index=False)

    wide = table.pivot_table(index=["g", "seed"], columns="family", values="test_auc").reset_index()
    wide["ZZ_minus_Z"] = wide["ZZ"] - wide["Z"]
    wide["ZplusZZ_minus_best_single"] = wide["Z+ZZ"] - wide[["Z", "ZZ"]].max(axis=1)
    wide.to_csv(OUTPUT_DIR / "ideal_observable_paired_contrasts_by_seed.csv", index=False)
    contrasts = (
        wide.groupby("g", as_index=False)
        .agg(
            ZZ_minus_Z_mean=("ZZ_minus_Z", "mean"),
            ZZ_minus_Z_sd=("ZZ_minus_Z", "std"),
            combined_minus_best_mean=("ZplusZZ_minus_best_single", "mean"),
            combined_minus_best_sd=("ZplusZZ_minus_best_single", "std"),
            nested_seed_count=("seed", "nunique"),
        )
        .sort_values("g")
    )
    contrasts.to_csv(OUTPUT_DIR / "ideal_observable_paired_contrasts_summary.csv", index=False)
    return summary, wide, contrasts


def load_hardware_contrasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    batch = pd.read_csv(HARDWARE_DIR / "hardware_batch_means.csv")
    wide = batch.pivot_table(index=["g", "repeat"], columns="family", values="auc_mean").reset_index()
    wide["ZZ_minus_Z"] = wide["ZZ"] - wide["Z"]
    summary = (
        wide.groupby("g", as_index=False)
        .agg(
            ZZ_minus_Z_mean=("ZZ_minus_Z", "mean"),
            ZZ_minus_Z_sd=("ZZ_minus_Z", "std"),
            hardware_batch_count=("repeat", "nunique"),
        )
        .sort_values("g")
    )
    return wide, summary


def build_retention_table(ideal: pd.DataFrame) -> pd.DataFrame:
    hardware = pd.read_csv(HARDWARE_DIR / "hardware_gscan_z_zz_source_data.csv")
    merged = ideal.merge(
        hardware[["g", "family", "auc_mean", "auc_sd_across_batches"]],
        on=["g", "family"], suffixes=("_ideal", "_hardware")
    )
    merged["hardware_minus_ideal_auc"] = merged["auc_mean_hardware"] - merged["auc_mean_ideal"]
    ideal_signal = merged["auc_mean_ideal"] - 0.5
    hardware_signal = merged["auc_mean_hardware"] - 0.5
    merged["above_chance_retention_ratio"] = np.where(
        ideal_signal > 0, hardware_signal / ideal_signal, np.nan
    )
    merged.to_csv(OUTPUT_DIR / "ideal_hardware_observable_retention.csv", index=False)
    return merged


def plot(summary: pd.DataFrame, table: pd.DataFrame, ideal_contrasts: pd.DataFrame) -> None:
    hardware_wide, hardware_contrasts = load_hardware_contrasts()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )
    colors = {"Z": "#4C78A8", "ZZ": "#E07A5F", "Z+ZZ": "#2A9D8F"}
    markers = {"Z": "o", "ZZ": "s", "Z+ZZ": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    ax = axes[0]
    for family in ("Z", "ZZ", "Z+ZZ"):
        values = summary[summary["family"] == family].sort_values("g")
        ax.errorbar(
            values["g"], values["auc_mean"], yerr=values["auc_sd_nested_seeds"],
            color=colors[family], marker=markers[family], lw=1.35, ms=4.3,
            capsize=2.2, label=family, zorder=3,
        )
        nested = table[table["family"] == family]
        ax.scatter(
            nested["g"] + {"Z": -0.008, "ZZ": 0.0, "Z+ZZ": 0.008}[family],
            nested["test_auc"], s=7, facecolor="none", edgecolor=colors[family],
            linewidth=0.5, alpha=0.35, zorder=2,
        )
    ax.axhline(0.5, color="#777777", lw=0.8, ls=(0, (3, 2)))
    ax.axvspan(0.7, 0.9, color="#D9D9D9", alpha=0.28, zorder=-1)
    ax.set(xlabel="Transverse-field parameter, $g$", ylabel="Ideal test ROC–AUC", xlim=(0.35, 1.05), ylim=(0.42, 0.76))
    ax.set_xticks([0.4, 0.6, 0.84, 1.0], ["0.40", "0.60", "0.84", "1.00"])
    ax.legend(loc="lower left", ncol=3, handlelength=1.5, columnspacing=0.9)
    ax.text(-0.16, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    ax = axes[1]
    ideal = ideal_contrasts.sort_values("g")
    hardware = hardware_contrasts.sort_values("g")
    ax.errorbar(
        ideal["g"], ideal["ZZ_minus_Z_mean"], yerr=ideal["ZZ_minus_Z_sd"],
        color="#5E60CE", marker="o", lw=1.35, ms=4.3, capsize=2.2,
        label="Ideal (10 nested splits)", zorder=3,
    )
    ax.errorbar(
        hardware["g"], hardware["ZZ_minus_Z_mean"], yerr=hardware["ZZ_minus_Z_sd"],
        color="#D95F59", marker="s", lw=1.35, ms=4.3, capsize=2.2,
        label="Hardware (3 batches)", zorder=3,
    )
    ax.scatter(
        hardware_wide["g"], hardware_wide["ZZ_minus_Z"], s=9,
        facecolor="white", edgecolor="#D95F59", linewidth=0.6, zorder=4,
    )
    ax.axhline(0.0, color="#777777", lw=0.8, ls=(0, (3, 2)))
    ax.axvspan(0.7, 0.9, color="#D9D9D9", alpha=0.28, zorder=-1)
    ax.set(xlabel="Transverse-field parameter, $g$", ylabel="$\mathrm{AUC}_{ZZ}-\mathrm{AUC}_{Z}$", xlim=(0.35, 1.05), ylim=(-0.16, 0.16))
    ax.set_xticks([0.4, 0.6, 0.84, 1.0], ["0.40", "0.60", "0.84", "1.00"])
    ax.legend(loc="upper left")
    ax.text(-0.16, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=8)
    for current in axes:
        current.tick_params(direction="out", length=3, width=0.8)

    stem = OUTPUT_DIR / "ideal_hardware_observable_information_shift"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, contrasts: pd.DataFrame, retention: pd.DataFrame) -> None:
    lines = [
        "# Ideal observable-family g scan",
        "",
        "All four ideal feature matrices contain the same 400 canonical-v3 molecules,",
        "Q=T=6, L=3, W=0.6, and reservoir seed 0. Each family receives an",
        "independent unregularized Linear(D,1) readout under the same ten stored",
        "scaffold splits and validation ranking protocol.",
        "",
        "| g | Z AUC | ZZ AUC | Z+ZZ AUC | ZZ-Z |",
        "|---:|---:|---:|---:|---:|",
    ]
    for g_value in sorted(IDEAL_FILES):
        part = summary[summary["g"] == g_value].set_index("family")
        contrast = contrasts[contrasts["g"] == g_value].iloc[0]
        lines.append(
            f"| {g_value:.2f} | {part.loc['Z','auc_mean']:.3f} +/- {part.loc['Z','auc_sd_nested_seeds']:.3f} "
            f"| {part.loc['ZZ','auc_mean']:.3f} +/- {part.loc['ZZ','auc_sd_nested_seeds']:.3f} "
            f"| {part.loc['Z+ZZ','auc_mean']:.3f} +/- {part.loc['Z+ZZ','auc_sd_nested_seeds']:.3f} "
            f"| {contrast.ZZ_minus_Z_mean:+.3f} |"
        )
    lines.extend(["", "## Hardware retention relative to ideal", ""])
    for g_value in sorted(IDEAL_FILES):
        part = retention[retention["g"] == g_value].set_index("family")
        lines.append(
            f"- g={g_value:.2f}: hardware-ideal AUC is "
            f"{part.loc['Z','hardware_minus_ideal_auc']:+.3f} for Z and "
            f"{part.loc['ZZ','hardware_minus_ideal_auc']:+.3f} for ZZ."
        )
    lines.extend(
        [
            "",
            "Interpretation must account for unequal family widths (Z: 6, ZZ: 15,",
            "Z+ZZ: 21) and for the fact that the ten ideal seeds are nested split/readout",
            "analyses rather than independent reservoirs.",
        ]
    )
    (OUTPUT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    table = train_all()
    summary, _, contrasts = summarize(table)
    retention = build_retention_table(summary)
    plot(summary, table, contrasts)
    write_report(summary, contrasts, retention)
    print(summary[["g", "family", "auc_mean", "auc_sd_nested_seeds"]].to_string(index=False))
    print("\nPaired ZZ-Z contrasts")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
