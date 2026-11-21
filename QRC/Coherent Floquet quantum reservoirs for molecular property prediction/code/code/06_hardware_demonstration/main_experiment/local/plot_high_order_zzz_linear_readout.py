#!/usr/bin/env python3
"""Plot ideal and hardware effects of diagonal third-order Z correlations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT / "local_results_five_tasks" / "high_order_zzz_linear"
)
FIGURE_DIR = RESULT_DIR / "figures"
METRICS_PATH = RESULT_DIR / "high_order_zzz_linear_metrics_by_seed.csv"
DELTA_PATH = RESULT_DIR / "high_order_zzz_vs_z_zz_paired_by_seed.csv"

TASKS = (
    "01_q6_t6_l3_batch01",
    "02_q6_t6_l3_batch02",
    "03_q6_t6_l3_batch03",
    "04_q6_t6_l1_base_batch01",
    "05_q6_t6_l1_fold3_batch01",
)
TASK_LABELS = (
    "L3\nbatch 1",
    "L3\nbatch 2",
    "L3\nbatch 3",
    "L1\nbase",
    "L1\nfold3",
)
FAMILIES = ("Z+ZZ", "ZZZ", "Z+ZZ+ZZZ")
FAMILY_LABELS = {
    "Z+ZZ": r"$Z+ZZ$ ($D=21$)",
    "ZZZ": r"$ZZZ$ ($D=20$)",
    "Z+ZZ+ZZZ": r"$Z+ZZ+ZZZ$ ($D=41$)",
}
COLORS = {
    "Z+ZZ": "#3C5488",
    "ZZZ": "#4DBBD5",
    "Z+ZZ+ZZZ": "#DC0000",
}
MARKERS = {"Z+ZZ": "o", "ZZZ": "D", "Z+ZZ+ZZZ": "^"}
OFFSETS = {"Z+ZZ": -0.19, "ZZZ": 0.0, "Z+ZZ+ZZZ": 0.19}
DOMAINS = ("ideal", "baihua", "shenglian")
DOMAIN_LABELS = {
    "ideal": "Ideal QRC",
    "baihua": "Baihua",
    "shenglian": "Shenglian",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Noto Sans CJK JP",
                "sans-serif",
            ],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.15,
            "lines.markersize": 4.2,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(METRICS_PATH)
    delta = pd.read_csv(DELTA_PATH)
    if len(metrics) != 300:
        raise AssertionError(f"Expected 300 metric rows, got {len(metrics)}")
    if set(metrics["observable_family"]) != set(FAMILIES):
        raise AssertionError("Observable-family mismatch")
    if sorted(metrics["seed"].unique().tolist()) != list(
        range(0, 100, 10)
    ):
        raise AssertionError("Seed mismatch")
    numeric = metrics[
        [
            "ideal_test_auc",
            "hardware_test_auc",
            "ideal_test_acc",
            "hardware_test_acc",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise AssertionError("Nonfinite metric")
    return metrics, delta


def values_for(
    data: pd.DataFrame,
    domain: str,
    task_key: str,
    family: str,
) -> np.ndarray:
    rows = data[
        (data["task_key"] == task_key)
        & (data["observable_family"] == family)
    ]
    if domain == "ideal":
        rows = rows[rows["backend"] == "baihua"]
        column = "ideal_test_auc"
    else:
        rows = rows[rows["backend"] == domain]
        column = "hardware_test_auc"
    values = rows.sort_values("seed")[column].to_numpy(dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise AssertionError((domain, task_key, family, values))
    return values


def delta_values_for(
    data: pd.DataFrame,
    domain: str,
    task_key: str,
    family: str,
) -> np.ndarray:
    rows = data[
        (data["task_key"] == task_key)
        & (data["observable_family"] == family)
    ]
    if domain == "ideal":
        rows = rows[rows["backend"] == "baihua"]
        column = "delta_ideal_auc_vs_z_zz"
    else:
        rows = rows[rows["backend"] == domain]
        column = "delta_hardware_auc_vs_z_zz"
    values = rows.sort_values("seed")[column].to_numpy(dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise AssertionError((domain, task_key, family, values))
    return values


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 450},
        ".tiff": {
            "dpi": 600,
            "pil_kwargs": {"compression": "tiff_lzw"},
        },
    }.items():
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def draw_auc(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.45), sharex=True)
    xbase = np.arange(len(TASKS), dtype=float)
    for panel, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        for family in FAMILIES:
            task_values = [
                values_for(data, domain, task_key, family)
                for task_key in TASKS
            ]
            means = np.asarray([values.mean() for values in task_values])
            deviations = np.asarray(
                [values.std(ddof=1) for values in task_values]
            )
            ax.errorbar(
                xbase + OFFSETS[family],
                means,
                yerr=deviations,
                fmt=MARKERS[family],
                ms=4.7,
                capsize=2.0,
                lw=0.95,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
                zorder=3,
            )
            jitter = np.linspace(-0.034, 0.034, 10)
            ax.scatter(
                np.concatenate(
                    [
                        np.full(10, x + OFFSETS[family]) + jitter
                        for x in xbase
                    ]
                ),
                np.concatenate(task_values),
                s=5.5,
                color=COLORS[family],
                alpha=0.27,
                linewidths=0,
                zorder=2.5,
            )
        ax.axhline(0.5, color="#C8C8C8", ls="--", lw=0.8)
        ax.set_ylim(0.30, 0.88)
        ax.set_ylabel("Test ROC--AUC")
        ax.grid(axis="y", color="#EAEAEA", lw=0.6)
        ax.text(
            0.01,
            0.94,
            DOMAIN_LABELS[domain],
            transform=ax.transAxes,
            va="top",
            fontsize=6.2,
        )
        ax.text(
            -0.13,
            1.02,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            weight="bold",
        )
    axes[-1].set_xticks(xbase, TASK_LABELS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.998),
        fontsize=5.1,
        columnspacing=0.55,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=0.75)
    save_figure(
        fig,
        FIGURE_DIR / "hardware_high_order_zzz_linear_readout_en",
    )


def draw_delta(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.10), sharex=True)
    xbase = np.arange(len(TASKS), dtype=float)
    delta_families = ("ZZZ", "Z+ZZ+ZZZ")
    delta_offsets = {"ZZZ": -0.10, "Z+ZZ+ZZZ": 0.10}
    for panel, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        for family in delta_families:
            task_values = [
                delta_values_for(data, domain, task_key, family)
                for task_key in TASKS
            ]
            means = np.asarray([values.mean() for values in task_values])
            deviations = np.asarray(
                [values.std(ddof=1) for values in task_values]
            )
            ax.errorbar(
                xbase + delta_offsets[family],
                means,
                yerr=deviations,
                fmt=MARKERS[family],
                ms=4.7,
                capsize=2.0,
                lw=0.95,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
                zorder=3,
            )
            jitter = np.linspace(-0.032, 0.032, 10)
            ax.scatter(
                np.concatenate(
                    [
                        np.full(10, x + delta_offsets[family]) + jitter
                        for x in xbase
                    ]
                ),
                np.concatenate(task_values),
                s=5.5,
                color=COLORS[family],
                alpha=0.27,
                linewidths=0,
                zorder=2.5,
            )
        ax.axhline(0.0, color="#8D8D8D", ls="--", lw=0.8)
        ax.set_ylim(-0.25, 0.25)
        ax.set_ylabel(r"$\Delta$AUC vs $Z+ZZ$")
        ax.grid(axis="y", color="#EAEAEA", lw=0.6)
        ax.text(
            0.01,
            0.94,
            DOMAIN_LABELS[domain],
            transform=ax.transAxes,
            va="top",
            fontsize=6.2,
        )
        ax.text(
            -0.13,
            1.02,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            weight="bold",
        )
    axes[-1].set_xticks(xbase, TASK_LABELS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.998),
        fontsize=5.2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=0.75)
    save_figure(
        fig,
        FIGURE_DIR / "hardware_high_order_zzz_delta_en",
    )


def write_reports(
    metrics: pd.DataFrame,
    delta: pd.DataFrame,
) -> None:
    metrics.to_csv(
        FIGURE_DIR / "hardware_high_order_zzz_source_data.csv",
        index=False,
    )
    delta.to_csv(
        FIGURE_DIR / "hardware_high_order_zzz_delta_source_data.csv",
        index=False,
    )
    summary = (
        metrics.groupby(
            ["backend", "task_key", "observable_family", "feature_dim"],
            observed=True,
            as_index=False,
        )
        .agg(
            ideal_auc_mean=("ideal_test_auc", "mean"),
            ideal_auc_sd_nested=("ideal_test_auc", "std"),
            hardware_auc_mean=("hardware_test_auc", "mean"),
            hardware_auc_sd_nested=("hardware_test_auc", "std"),
        )
    )
    delta_summary = (
        delta.groupby(
            ["backend", "task_key", "observable_family", "feature_dim"],
            observed=True,
            as_index=False,
        )
        .agg(
            delta_ideal_auc_mean=("delta_ideal_auc_vs_z_zz", "mean"),
            delta_hardware_auc_mean=(
                "delta_hardware_auc_vs_z_zz",
                "mean",
            ),
        )
    )
    report = summary.merge(
        delta_summary,
        on=["backend", "task_key", "observable_family", "feature_dim"],
        validate="one_to_one",
    )
    report["observable_family"] = pd.Categorical(
        report["observable_family"],
        categories=FAMILIES,
        ordered=True,
    )
    report = report.sort_values(
        ["backend", "task_key", "observable_family"]
    ).reset_index(drop=True)
    report.to_csv(
        RESULT_DIR / "high_order_zzz_final_summary.csv",
        index=False,
    )

    lines = [
        "# High-order ZZZ linear-readout diagnostic",
        "",
        "The same single affine linear readout is fitted separately to ideal, "
        "Baihua, and Shenglian feature domains. `ZZZ` denotes raw third-order "
        "Pauli-string expectations, not connected cumulants.",
        "",
        "| Backend | Task | Family | D | Ideal AUC | Hardware AUC | Hardware delta vs Z+ZZ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in report.iterrows():
        lines.append(
            f"| {row['backend'].title()} | `{row['task_key']}` | "
            f"{row['observable_family']} | {int(row['feature_dim'])} | "
            f"{row['ideal_auc_mean']:.3f} | "
            f"{row['hardware_auc_mean']:.3f} | "
            f"{row['delta_hardware_auc_mean']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "`Z+ZZ` (D=21) and `ZZZ` (D=20) are near-matched in retained "
            "feature dimension but are not an exactly fixed-D comparison. "
            "The augmented D=41 family asks whether third-order moments add "
            "held-out discrimination; it is not a fixed measurement-cost "
            "comparison.",
            "",
            "The ten scaffold/readout seeds reuse each hardware feature "
            "matrix and are not independent device repetitions.",
            "",
        ]
    )
    (RESULT_DIR / "HIGH_ORDER_ZZZ_RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    legend = """# Figure legend

**Diagonal third-order correlations provide an observable-family sensitivity
test on ideal and hardware QRC features.** Test ROC–AUC for **a,** ideal QRC,
**b,** Baihua, and **c,** Shenglian using low-order `Z+ZZ` (D=21),
third-order-only `ZZZ` (D=20), or augmented `Z+ZZ+ZZZ` (D=41) feature
families. Every feature domain receives a separately fitted single affine
linear readout with the same scaffold splits, training seeds, and
validation-only checkpoint selection. Large symbols and error bars show mean
± sample standard deviation across ten nested scaffold/readout seeds; light
points show individual seed values. These seeds are not independent hardware
repetitions. Dashed lines mark chance. `ZZZ` denotes raw Pauli-string
expectations rather than connected cumulants.

Source data: `hardware_high_order_zzz_source_data.csv`.
"""
    (
        FIGURE_DIR / "hardware_high_order_zzz_linear_readout_legend.md"
    ).write_text(legend, encoding="utf-8")


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics, delta = load_data()
    write_reports(metrics, delta)
    draw_auc(metrics)
    draw_delta(delta)
    print(
        "[saved]",
        FIGURE_DIR / "hardware_high_order_zzz_linear_readout_en.pdf",
    )
    print("[saved]", RESULT_DIR / "HIGH_ORDER_ZZZ_RESULTS.md")


if __name__ == "__main__":
    main()
