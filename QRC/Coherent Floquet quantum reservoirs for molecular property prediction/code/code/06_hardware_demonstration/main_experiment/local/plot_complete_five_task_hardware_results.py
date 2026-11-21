#!/usr/bin/env python3
"""Summarize and plot the complete five-task hardware experiment.

The manuscript-facing result fits the same single linear readout separately
on ideal, Baihua, and Shenglian training features. A distinct supporting
figure applies the ideal readout unchanged to hardware and is explicitly
labelled as a zero-calibration coordinate-transfer diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "local_results_five_tasks"
FIGURE_DIR = RESULT_DIR / "figures"
METRICS_PATH = RESULT_DIR / "five_task_ideal_vs_hardware_metrics_by_seed.csv"

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

# Exact visual vocabulary used by the existing PR Applied hardware figure.
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
        "hardware_trained_test_auc",
        "hardware_trained_test_acc",
        "frozen_ideal_head_hardware_test_auc",
        "frozen_ideal_head_hardware_test_acc",
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
    protocol: str,
) -> np.ndarray:
    if source == "ideal":
        column = f"ideal_test_{metric}"
        rows = data[
            (data["backend"] == "baihua")
            & (data["task_key"] == task_key)
        ].sort_values("seed")
    else:
        if protocol == "end_to_end":
            column = f"hardware_trained_test_{metric}"
        elif protocol == "zero_calibration":
            column = f"frozen_ideal_head_hardware_test_{metric}"
        else:
            raise ValueError(protocol)
        rows = data[
            (data["backend"] == source)
            & (data["task_key"] == task_key)
        ].sort_values("seed")
    values = rows[column].to_numpy(dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise AssertionError((source, task_key, metric, protocol, values))
    return values


def draw_protocol_figure(
    data: pd.DataFrame,
    *,
    protocol: str,
    stem: str,
) -> None:
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
                values_for(data, source, task_key, metric, protocol)
                for task_key in TASKS
            ]
            means = np.asarray([values.mean() for values in task_values])
            standard_deviations = np.asarray(
                [values.std(ddof=1) for values in task_values]
            )
            color = COLORS[source]
            ax.errorbar(
                xbase + OFFSETS[source],
                means,
                yerr=standard_deviations,
                fmt=MARKERS[source],
                ms=5.2,
                capsize=2.2,
                lw=1.0,
                color=color,
                label=displays[source],
                zorder=3,
            )

            jitter = np.linspace(-0.038, 0.038, 10)
            point_x = np.concatenate(
                [
                    np.full(10, x + OFFSETS[source]) + jitter
                    for x in xbase
                ]
            )
            point_y = np.concatenate(task_values)
            ax.scatter(
                point_x,
                point_y,
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
    axes[0].text(
        -0.13,
        1.03,
        "a",
        transform=axes[0].transAxes,
        weight="bold",
    )
    axes[1].text(
        -0.13,
        1.03,
        "b",
        transform=axes[1].transAxes,
        weight="bold",
    )
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
    save_figure(fig, FIGURE_DIR / stem)


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


def build_source_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identifiers = ["backend", "task_key", "layers", "fold_factor", "seed", "n_test"]
    end_to_end = data[
        identifiers
        + [
            "ideal_test_auc",
            "ideal_test_acc",
            "hardware_trained_test_auc",
            "hardware_trained_test_acc",
            "hardware_minus_ideal_auc",
            "hardware_best_epoch",
            "hardware_val_auc",
            "hardware_val_loss",
            "hardware_val_acc",
            "readout",
            "selection_rule",
        ]
    ].copy()
    end_to_end["protocol"] = (
        "same linear-readout architecture fitted separately in each "
        "ideal/hardware feature domain"
    )
    end_to_end.to_csv(
        FIGURE_DIR / "hardware_five_task_linear_readout_source_data.csv",
        index=False,
    )

    zero_calibration = data[
        identifiers
        + [
            "ideal_test_auc",
            "ideal_test_acc",
            "frozen_ideal_head_hardware_test_auc",
            "frozen_ideal_head_hardware_test_acc",
            "frozen_transfer_minus_ideal_auc",
            "diagnostic_protocol",
            "selection_rule",
        ]
    ].copy()
    zero_calibration.to_csv(
        FIGURE_DIR
        / "hardware_five_task_zero_calibration_transfer_source_data.csv",
        index=False,
    )
    return end_to_end, zero_calibration


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["backend", "task_key", "layers", "fold_factor"], as_index=False)
        .agg(
            nested_seed_count=("seed", "nunique"),
            ideal_auc_mean=("ideal_test_auc", "mean"),
            ideal_auc_sd_nested=("ideal_test_auc", "std"),
            ideal_acc_mean=("ideal_test_acc", "mean"),
            ideal_acc_sd_nested=("ideal_test_acc", "std"),
            hardware_auc_mean=("hardware_trained_test_auc", "mean"),
            hardware_auc_sd_nested=("hardware_trained_test_auc", "std"),
            hardware_acc_mean=("hardware_trained_test_acc", "mean"),
            hardware_acc_sd_nested=("hardware_trained_test_acc", "std"),
            hardware_minus_ideal_auc_mean=("hardware_minus_ideal_auc", "mean"),
            zero_calibration_auc_mean=(
                "frozen_ideal_head_hardware_test_auc",
                "mean",
            ),
            zero_calibration_auc_sd_nested=(
                "frozen_ideal_head_hardware_test_auc",
                "std",
            ),
        )
    )
    summary.to_csv(RESULT_DIR / "final_hardware_results_by_task.csv", index=False)

    aggregate_rows = []
    for backend in BACKENDS:
        backend_summary = summary[summary["backend"] == backend].set_index("task_key")
        l3 = backend_summary.loc[list(TASKS[:3]), "hardware_auc_mean"]
        l3_zero = backend_summary.loc[
            list(TASKS[:3]), "zero_calibration_auc_mean"
        ]
        base = backend_summary.loc[TASKS[3]]
        fold3 = backend_summary.loc[TASKS[4]]
        aggregate_rows.append(
            {
                "backend": backend,
                "l3_acquisition_batch_count": 3,
                "l3_auc_mean_across_batch_means": l3.mean(),
                "l3_auc_sd_across_batch_means": l3.std(ddof=1),
                "l3_zero_calibration_auc_mean_across_batch_means": (
                    l3_zero.mean()
                ),
                "l1_base_auc_mean_nested": base["hardware_auc_mean"],
                "l1_fold3_auc_mean_nested": fold3["hardware_auc_mean"],
                "l1_fold3_minus_base_auc_mean_nested": (
                    fold3["hardware_auc_mean"] - base["hardware_auc_mean"]
                ),
                "l1_base_zero_calibration_auc_mean_nested": (
                    base["zero_calibration_auc_mean"]
                ),
                "l1_fold3_zero_calibration_auc_mean_nested": (
                    fold3["zero_calibration_auc_mean"]
                ),
                "hardware_samples_per_task": 400,
                "shots_per_circuit": 1024,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        RESULT_DIR / "final_hardware_results_summary.csv",
        index=False,
    )
    return summary


def write_reports(summary: pd.DataFrame) -> None:
    lines = [
        "# Final five-task hardware results",
        "",
        "The manuscript-facing comparison fits the same single linear readout "
        "separately on ideal, Baihua, and Shenglian training features. Model "
        "selection is validation AUC, then validation loss, then validation "
        "accuracy. The frozen ideal-head result is retained only as a "
        "zero-calibration feature-coordinate diagnostic.",
        "",
        "## End-to-end linear-readout test results",
        "",
        "| Backend | Task | Ideal AUC | Hardware AUC | AUC SD | Ideal acc. | Hardware acc. | Acc. SD | Hardware − ideal AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {BACKEND_LABELS[row['backend']]} | `{row['task_key']}` | "
            f"{row['ideal_auc_mean']:.3f} | {row['hardware_auc_mean']:.3f} | "
            f"{row['hardware_auc_sd_nested']:.3f} | "
            f"{row['ideal_acc_mean']:.3f} | {row['hardware_acc_mean']:.3f} | "
            f"{row['hardware_acc_sd_nested']:.3f} | "
            f"{row['hardware_minus_ideal_auc_mean']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Zero-calibration coordinate-transfer diagnostic",
            "",
            "| Backend | Task | Frozen ideal-head AUC | Nested-seed SD |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {BACKEND_LABELS[row['backend']]} | `{row['task_key']}` | "
            f"{row['zero_calibration_auc_mean']:.3f} | "
            f"{row['zero_calibration_auc_sd_nested']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Shenglian retains useful end-to-end task information across the "
            "five tasks, whereas Baihua is weak and batch dependent. The "
            "base/fold3 task-performance response is processor dependent: "
            "fold3 improves the Baihua mean but reduces the Shenglian mean. "
            "The lower zero-calibration results show substantial ideal–hardware "
            "coordinate mismatch and must not be reported as the final "
            "hardware-trained classifier AUC.",
            "",
            "The ten scaffold/readout seeds are nested analyses of the same "
            "hardware feature array and are not ten hardware repetitions. The "
            "three L3 task folders are described as acquisition batches unless "
            "distinct calibration windows are independently documented.",
            "",
        ]
    )
    (RESULT_DIR / "FINAL_HARDWARE_RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    primary_legend = """# Main figure legend

**Domain-matched linear readout reveals processor-dependent hardware task
performance.** **a,** Test ROC–AUC and **b,** test accuracy for ideal QRC,
Baihua, and Shenglian across three L3 acquisition batches and the paired L1
base/fold3 tasks. The same single linear-readout architecture, scaffold splits,
training seeds, and validation selection rule are used throughout; each
readout is fitted on its own ideal or hardware training feature matrix. Large
symbols and error bars show mean ± sample standard deviation across ten nested
scaffold/readout seeds; light points show the ten seed values. These seeds
reuse one hardware feature array per task and are not independent device
repetitions. Horizontal dashed lines mark chance. Each task comprises 400
molecular circuits and 1,024 shots per circuit.

Source data: `hardware_five_task_linear_readout_source_data.csv`.
"""
    (FIGURE_DIR / "hardware_five_task_linear_readout_legend.md").write_text(
        primary_legend,
        encoding="utf-8",
    )

    diagnostic_legend = """# Supporting figure legend

**Zero-calibration ideal-to-hardware readout transfer diagnoses feature-coordinate
preservation.** Panels follow the main hardware figure, but the ideal-feature
linear readout is applied unchanged to each hardware feature matrix. The
result therefore measures coordinate transfer rather than the final task
performance available after fitting a hardware-domain linear readout.

Source data: `hardware_five_task_zero_calibration_transfer_source_data.csv`.
"""
    (
        FIGURE_DIR
        / "hardware_five_task_zero_calibration_transfer_legend.md"
    ).write_text(diagnostic_legend, encoding="utf-8")


def write_contract() -> None:
    contract = """# Hardware figure contract

Core conclusion: domain-matched linear readouts reveal useful but
processor-dependent task information on Shenglian and weaker, batch-dependent
performance on Baihua; zero-calibration ideal-head transfer is a separate
feature-coordinate diagnostic.

Figure archetype: quantitative grid matching the existing PR Applied hardware
figure.

Target/output: PR Applied single-column figure, Python/matplotlib, 3.45 ×
4.35 in, editable SVG/PDF, 600 dpi LZW-TIFF, and 450 dpi PNG.

Panel map:

- a: test ROC–AUC for ideal QRC, Baihua, and Shenglian.
- b: test accuracy for the same tasks and protocols.

Statistics: large symbols are means and error bars are sample standard
deviations across ten nested scaffold/readout seeds. Light points are the ten
seed values. Seeds are not independent hardware repetitions. No inferential
P value is used.

Source data: `figures/hardware_five_task_linear_readout_source_data.csv`.

Image integrity: all plotted elements are produced directly from CSV outputs
by Python; no generated or manually edited data graphics are used. SVG/PDF
text remains editable.

Reviewer risks: acquisition folders do not establish independent calibration
windows; hardware-specific linear-readout fitting and frozen ideal-head
transfer answer different questions and must not be pooled.
"""
    (RESULT_DIR / "HARDWARE_FIGURE_CONTRACT.md").write_text(
        contract,
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    build_source_tables(data)
    summary = summarize(data)
    write_reports(summary)
    write_contract()

    draw_protocol_figure(
        data,
        protocol="end_to_end",
        stem="hardware_five_task_linear_readout_en",
    )
    # Keep the previous public stem synchronized with the corrected primary
    # protocol so old links cannot silently point to the frozen-head result.
    draw_protocol_figure(
        data,
        protocol="end_to_end",
        stem="hardware_five_task_robustness",
    )
    draw_protocol_figure(
        data,
        protocol="zero_calibration",
        stem="hardware_five_task_zero_calibration_transfer_en",
    )

    print(
        "[saved]",
        FIGURE_DIR / "hardware_five_task_linear_readout_en.pdf",
    )
    print(
        "[saved]",
        FIGURE_DIR / "hardware_five_task_zero_calibration_transfer_en.pdf",
    )
    print("[saved]", RESULT_DIR / "FINAL_HARDWARE_RESULTS.md")


if __name__ == "__main__":
    main()
