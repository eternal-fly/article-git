#!/usr/bin/env python3
"""Plot the domain-matched two-layer-MLP decoder sensitivity experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "local_results_five_tasks" / "mlp_sensitivity"
FIGURE_DIR = RESULT_DIR / "figures"
METRICS_PATH = RESULT_DIR / "two_layer_mlp_metrics_by_seed.csv"
COMPARISON_PATH = RESULT_DIR / "two_layer_mlp_vs_linear_summary.csv"

BACKENDS = ("baihua", "shenglian")
BACKEND_LABELS = {"baihua": "Baihua", "shenglian": "Shenglian"}
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
COLORS = {
    "ideal": "#3C5488",
    "baihua": "#4DBBD5",
    "shenglian": "#DC0000",
}
MARKERS = {"ideal": "o", "baihua": "D", "shenglian": "^"}
OFFSETS = {"ideal": -0.19, "baihua": 0.0, "shenglian": 0.19}


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
            "legend.fontsize": 6.2,
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


def load_metrics() -> pd.DataFrame:
    data = pd.read_csv(METRICS_PATH)
    required = {
        "backend",
        "task_key",
        "seed",
        "ideal_test_auc",
        "ideal_test_acc",
        "hardware_test_auc",
        "hardware_test_acc",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    if len(data) != 100:
        raise AssertionError(f"Expected 100 rows, got {len(data)}")
    if set(data["backend"]) != set(BACKENDS):
        raise AssertionError("Backend mismatch")
    if set(data["task_key"]) != set(TASKS):
        raise AssertionError("Task mismatch")
    if sorted(data["seed"].unique().tolist()) != list(range(0, 100, 10)):
        raise AssertionError("Seed mismatch")
    return data


def values_for(
    data: pd.DataFrame,
    source: str,
    task_key: str,
    metric: str,
) -> np.ndarray:
    if source == "ideal":
        column = f"ideal_test_{metric}"
        rows = data[
            (data["backend"] == "baihua")
            & (data["task_key"] == task_key)
        ].sort_values("seed")
    else:
        column = f"hardware_test_{metric}"
        rows = data[
            (data["backend"] == source)
            & (data["task_key"] == task_key)
        ].sort_values("seed")
    values = rows[column].to_numpy(dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise AssertionError((source, task_key, metric, values))
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


def draw(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.35), sharex=True)
    xbase = np.arange(len(TASKS), dtype=float)
    sources = ("ideal", "baihua", "shenglian")
    displays = {
        "ideal": "Ideal QRC",
        "baihua": "Baihua",
        "shenglian": "Shenglian",
    }
    metric_specs = (
        ("auc", "Test ROC--AUC", (0.30, 0.88)),
        ("acc", "Test accuracy", (0.30, 0.82)),
    )

    for ax, (metric, ylabel, ylim) in zip(axes, metric_specs):
        for source in sources:
            task_values = [
                values_for(data, source, task_key, metric)
                for task_key in TASKS
            ]
            means = np.asarray([values.mean() for values in task_values])
            deviations = np.asarray(
                [values.std(ddof=1) for values in task_values]
            )
            color = COLORS[source]
            ax.errorbar(
                xbase + OFFSETS[source],
                means,
                yerr=deviations,
                fmt=MARKERS[source],
                ms=5.2,
                capsize=2.2,
                lw=1.0,
                color=color,
                label=displays[source],
                zorder=3,
            )
            jitter = np.linspace(-0.038, 0.038, 10)
            ax.scatter(
                np.concatenate(
                    [
                        np.full(10, x + OFFSETS[source]) + jitter
                        for x in xbase
                    ]
                ),
                np.concatenate(task_values),
                s=6.5,
                color=color,
                alpha=0.30,
                linewidths=0,
                zorder=2.5,
            )

        ax.axhline(0.5, color="#C8C8C8", ls="--", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#EAEAEA", lw=0.6)

    axes[1].set_xticks(xbase, TASK_LABELS)
    axes[0].text(-0.13, 1.03, "a", transform=axes[0].transAxes, weight="bold")
    axes[1].text(-0.13, 1.03, "b", transform=axes[1].transAxes, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=5.7,
        columnspacing=0.7,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.0)
    save_figure(
        fig,
        FIGURE_DIR / "hardware_five_task_two_layer_mlp_en",
    )


def write_outputs(data: pd.DataFrame) -> None:
    data.to_csv(
        FIGURE_DIR
        / "hardware_five_task_two_layer_mlp_source_data.csv",
        index=False,
    )
    summary = (
        data.groupby(
            ["backend", "task_key", "layers", "fold_factor"],
            as_index=False,
        )
        .agg(
            nested_seed_count=("seed", "nunique"),
            ideal_auc_mean=("ideal_test_auc", "mean"),
            ideal_auc_sd_nested=("ideal_test_auc", "std"),
            ideal_acc_mean=("ideal_test_acc", "mean"),
            ideal_acc_sd_nested=("ideal_test_acc", "std"),
            hardware_auc_mean=("hardware_test_auc", "mean"),
            hardware_auc_sd_nested=("hardware_test_auc", "std"),
            hardware_acc_mean=("hardware_test_acc", "mean"),
            hardware_acc_sd_nested=("hardware_test_acc", "std"),
        )
    )
    comparison = pd.read_csv(COMPARISON_PATH)
    lines = [
        "# Two-layer MLP decoder sensitivity",
        "",
        "Architecture: `Linear(21,21) -> ReLU -> Linear(21,1)`. Ideal, "
        "Baihua, and Shenglian readouts are fitted separately using the same "
        "splits, seeds, optimization settings, and validation ranking.",
        "",
        "| Backend | Task | Ideal AUC | Hardware AUC | Hardware AUC SD | MLP − linear hardware AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    comparison_lookup = comparison.set_index(["backend", "task_key"])
    for _, row in summary.iterrows():
        delta = comparison_lookup.loc[
            (row["backend"], row["task_key"]),
            "mlp_minus_linear_hardware_auc_mean",
        ]
        lines.append(
            f"| {BACKEND_LABELS[row['backend']]} | `{row['task_key']}` | "
            f"{row['ideal_auc_mean']:.3f} | {row['hardware_auc_mean']:.3f} | "
            f"{row['hardware_auc_sd_nested']:.3f} | {delta:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Large symbols and error bars in the figure show mean ± sample "
            "standard deviation across ten nested scaffold/readout seeds. "
            "Those seeds reuse each hardware feature matrix and are not "
            "independent hardware repetitions.",
            "",
        ]
    )
    (RESULT_DIR / "TWO_LAYER_MLP_RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    legend = """# Figure legend

**Two-layer nonlinear readouts retain processor-dependent hardware
performance.** **a,** Test ROC–AUC and **b,** test accuracy for ideal QRC,
Baihua, and Shenglian using `Linear(21,21) -> ReLU -> Linear(21,1)` readouts
fitted separately in each feature domain. Large symbols and error bars show
mean ± sample standard deviation across ten nested scaffold/readout seeds;
light points show individual seed values. These seeds reuse one hardware
feature array per task and are not independent device repetitions. Dashed
lines mark chance. Each task contains 400 molecular circuits with 1,024 shots
per circuit.

Source data: `hardware_five_task_two_layer_mlp_source_data.csv`.
"""
    (
        FIGURE_DIR / "hardware_five_task_two_layer_mlp_legend.md"
    ).write_text(legend, encoding="utf-8")


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    write_outputs(data)
    draw(data)
    print(
        "[saved]",
        FIGURE_DIR / "hardware_five_task_two_layer_mlp_en.pdf",
    )
    print("[saved]", RESULT_DIR / "TWO_LAYER_MLP_RESULTS.md")


if __name__ == "__main__":
    main()
