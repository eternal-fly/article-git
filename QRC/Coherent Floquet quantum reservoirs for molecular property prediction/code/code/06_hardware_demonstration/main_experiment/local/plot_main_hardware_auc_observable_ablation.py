#!/usr/bin/env python3
"""Build the manuscript hardware AUC and Shenglian observable-family figure.

Panel a reports the domain-matched, full Z+ZZ linear-readout AUC for ideal,
Baihua, and Shenglian features. Panel b decomposes the Shenglian feature
matrix into Z-only, ZZ-only, and Z+ZZ readouts. All graphics and manuscript
source-data files are generated directly from stored CSV outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
RESULT_DIR = EXPERIMENT_ROOT / "local_results_five_tasks"
ABLATION_DIR = RESULT_DIR / "observable_family_ablation"
FIGURE_DIR = RESULT_DIR / "figures"
MANUSCRIPT_DIR = PROJECT_ROOT / "PRL" / "ch"
MANUSCRIPT_FIGURE_DIR = MANUSCRIPT_DIR / "fig_applied"
MANUSCRIPT_TABLE_DIR = MANUSCRIPT_DIR / "evidence_tables"

MAIN_METRICS = RESULT_DIR / "five_task_ideal_vs_hardware_metrics_by_seed.csv"
ABLATION_METRICS = (
    ABLATION_DIR / "observable_family_metrics_by_seed_shenglian.csv"
)

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
SEEDS = tuple(range(0, 100, 10))

DOMAIN_SERIES = ("ideal", "baihua", "shenglian")
DOMAIN_LABELS = {
    "ideal": "Ideal QRC",
    "baihua": "Baihua",
    "shenglian": "Shenglian",
}
DOMAIN_COLORS = {
    "ideal": "#3C5488",
    "baihua": "#4DBBD5",
    "shenglian": "#DC0000",
}
DOMAIN_MARKERS = {"ideal": "o", "baihua": "D", "shenglian": "^"}
DOMAIN_OFFSETS = {"ideal": -0.19, "baihua": 0.0, "shenglian": 0.19}

FAMILIES = ("Z", "ZZ", "Z+ZZ")
FAMILY_DIMS = {"Z": 6, "ZZ": 15, "Z+ZZ": 21}
FAMILY_LABELS = {"Z": r"$Z$ ($D=6$)", "ZZ": r"$ZZ$ ($D=15$)", "Z+ZZ": r"$Z+ZZ$ ($D=21$)"}
FAMILY_COLORS = {"Z": "#00A087", "ZZ": "#E64B35", "Z+ZZ": "#3C5488"}
FAMILY_MARKERS = {"Z": "s", "ZZ": "o", "Z+ZZ": "D"}
FAMILY_OFFSETS = {"Z": -0.16, "ZZ": 0.0, "Z+ZZ": 0.16}

OUTPUT_STEM = "mainline_hardware_q6_auc_observable_ablation_en"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.1,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.75,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "lines.linewidth": 1.05,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#B8B8B8",
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def load_main_metrics() -> pd.DataFrame:
    data = pd.read_csv(MAIN_METRICS)
    required = {
        "backend",
        "task_key",
        "seed",
        "ideal_test_auc",
        "hardware_trained_test_auc",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Main hardware metrics lack {sorted(missing)}")
    if len(data) != 100:
        raise AssertionError(f"Expected 100 main rows, found {len(data)}")
    if set(data["backend"]) != {"baihua", "shenglian"}:
        raise AssertionError("Unexpected hardware backend")
    if set(data["task_key"]) != set(TASKS):
        raise AssertionError("Unexpected task set")
    if tuple(sorted(data["seed"].unique())) != SEEDS:
        raise AssertionError("Unexpected main-analysis seeds")
    return data


def load_ablation_metrics() -> pd.DataFrame:
    data = pd.read_csv(ABLATION_METRICS)
    required = {
        "domain",
        "task_key",
        "observable_family",
        "feature_dim",
        "seed",
        "test_auc",
        "readout",
        "selection_rule",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Observable ablation lacks {sorted(missing)}")
    data = data[data["domain"] == "shenglian"].copy()
    expected = len(TASKS) * len(FAMILIES) * len(SEEDS)
    if len(data) != expected:
        raise AssertionError((len(data), expected))
    if set(data["task_key"]) != set(TASKS):
        raise AssertionError("Unexpected ablation task set")
    if set(data["observable_family"]) != set(FAMILIES):
        raise AssertionError("Unexpected observable families")
    if tuple(sorted(data["seed"].unique())) != SEEDS:
        raise AssertionError("Unexpected ablation seeds")
    observed_dims = (
        data[["observable_family", "feature_dim"]]
        .drop_duplicates()
        .set_index("observable_family")["feature_dim"]
        .astype(int)
        .to_dict()
    )
    if observed_dims != FAMILY_DIMS:
        raise AssertionError((observed_dims, FAMILY_DIMS))
    return data


def main_values(
    data: pd.DataFrame,
    series: str,
    task_key: str,
) -> np.ndarray:
    if series == "ideal":
        rows = data[
            (data["backend"] == "baihua")
            & (data["task_key"] == task_key)
        ].sort_values("seed")
        column = "ideal_test_auc"
    else:
        rows = data[
            (data["backend"] == series)
            & (data["task_key"] == task_key)
        ].sort_values("seed")
        column = "hardware_trained_test_auc"
    values = rows[column].to_numpy(dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise AssertionError((series, task_key, values))
    return values


def family_values(
    data: pd.DataFrame,
    family: str,
    task_key: str,
) -> np.ndarray:
    rows = data[
        (data["observable_family"] == family)
        & (data["task_key"] == task_key)
    ].sort_values("seed")
    values = rows["test_auc"].to_numpy(dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise AssertionError((family, task_key, values))
    return values


def draw_group(
    ax: plt.Axes,
    xbase: np.ndarray,
    values_by_series: dict[str, list[np.ndarray]],
    *,
    labels: dict[str, str],
    colors: dict[str, str],
    markers: dict[str, str],
    offsets: dict[str, float],
) -> None:
    for series, task_values in values_by_series.items():
        means = np.asarray([values.mean() for values in task_values])
        standard_deviations = np.asarray(
            [values.std(ddof=1) for values in task_values]
        )
        ax.errorbar(
            xbase + offsets[series],
            means,
            yerr=standard_deviations,
            fmt=markers[series],
            ms=5.0,
            capsize=2.1,
            lw=0.95,
            color=colors[series],
            label=labels[series],
            zorder=3,
        )
        jitter = np.linspace(-0.034, 0.034, len(SEEDS))
        for x, values in zip(xbase, task_values):
            ax.scatter(
                x + offsets[series] + jitter,
                values,
                s=6.0,
                color=colors[series],
                alpha=0.26,
                linewidths=0,
                zorder=2.5,
            )


def draw_figure(main: pd.DataFrame, ablation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.55), sharex=True)
    xbase = np.arange(len(TASKS), dtype=float)

    main_values_by_series = {
        series: [main_values(main, series, task) for task in TASKS]
        for series in DOMAIN_SERIES
    }
    draw_group(
        axes[0],
        xbase,
        main_values_by_series,
        labels=DOMAIN_LABELS,
        colors=DOMAIN_COLORS,
        markers=DOMAIN_MARKERS,
        offsets=DOMAIN_OFFSETS,
    )

    family_values_by_series = {
        family: [family_values(ablation, family, task) for task in TASKS]
        for family in FAMILIES
    }
    draw_group(
        axes[1],
        xbase,
        family_values_by_series,
        labels=FAMILY_LABELS,
        colors=FAMILY_COLORS,
        markers=FAMILY_MARKERS,
        offsets=FAMILY_OFFSETS,
    )

    for panel, ax in enumerate(axes):
        ax.axhline(0.5, color="#BDBDBD", ls="--", lw=0.8, zorder=1)
        ax.grid(axis="y", color="#EAEAEA", lw=0.6, zorder=0)
        ax.set_ylabel("Test ROC–AUC")
        ax.set_ylim(0.30, 0.90)
        ax.text(
            -0.13,
            1.03,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            weight="bold",
            fontsize=8.2,
        )
        ax.legend(
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=0.65,
            handletextpad=0.35,
            borderaxespad=0,
            borderpad=0.35,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)

    axes[0].text(
        0.99,
        0.04,
        "Full $Z+ZZ$ readout",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=6.2,
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "linewidth": 0.5,
            "alpha": 1.0,
        },
    )
    axes[1].text(
        0.99,
        0.04,
        "Shenglian",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=6.2,
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "linewidth": 0.5,
            "alpha": 1.0,
        },
    )
    axes[1].set_xticks(xbase, TASK_LABELS)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.1)
    save_figure(fig)


def save_figure(fig: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for destination in (FIGURE_DIR, MANUSCRIPT_FIGURE_DIR):
        stem = destination / OUTPUT_STEM
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
        fig.savefig(
            stem.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)


def build_source_data(
    main: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series in DOMAIN_SERIES:
        for task_key in TASKS:
            values = main_values(main, series, task_key)
            for seed, value in zip(SEEDS, values):
                rows.append(
                    {
                        "panel": "a",
                        "panel_role": "domain-matched full Z+ZZ AUC",
                        "series": DOMAIN_LABELS[series],
                        "domain": series,
                        "observable_family": "Z+ZZ",
                        "feature_dim": 21,
                        "task_key": task_key,
                        "seed": seed,
                        "test_auc": value,
                        "nested_seed_mean": float(values.mean()),
                        "nested_seed_sd": float(values.std(ddof=1)),
                        "repeat_definition": (
                            "nested scaffold/readout analysis; not an "
                            "independent hardware repetition"
                        ),
                    }
                )
    for family in FAMILIES:
        for task_key in TASKS:
            values = family_values(ablation, family, task_key)
            for seed, value in zip(SEEDS, values):
                rows.append(
                    {
                        "panel": "b",
                        "panel_role": "Shenglian observable-family AUC",
                        "series": family,
                        "domain": "shenglian",
                        "observable_family": family,
                        "feature_dim": FAMILY_DIMS[family],
                        "task_key": task_key,
                        "seed": seed,
                        "test_auc": value,
                        "nested_seed_mean": float(values.mean()),
                        "nested_seed_sd": float(values.std(ddof=1)),
                        "repeat_definition": (
                            "nested scaffold/readout analysis on one fixed "
                            "feature matrix; not an independent hardware "
                            "repetition"
                        ),
                    }
                )
    source = pd.DataFrame(rows)
    if len(source) != 300:
        raise AssertionError(len(source))
    for destination in (
        FIGURE_DIR / f"{OUTPUT_STEM}_source_data.csv",
        MANUSCRIPT_TABLE_DIR / f"{OUTPUT_STEM}_source_data.csv",
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(destination, index=False)
    return source


def copy_manuscript_tables() -> None:
    inputs = {
        "observable_family_summary.csv":
            ABLATION_DIR / "observable_family_summary.csv",
        "observable_family_paired_effects.csv":
            ABLATION_DIR / "observable_family_paired_effects.csv",
        "observable_family_depth_effects.csv":
            ABLATION_DIR / "observable_family_depth_effects.csv",
    }
    for output_name, source_path in inputs.items():
        table = pd.read_csv(source_path)
        table.to_csv(MANUSCRIPT_TABLE_DIR / output_name, index=False)


def write_metadata(source: pd.DataFrame) -> None:
    contract = """# Main hardware AUC figure contract

Core conclusion: canonical-v3 hardware features retain task-readable
directions most clearly on Shenglian, and Shenglian's unitary-preserving
folded-depth loss is concentrated in the pairwise ZZ readout.

Evidence chain:

- Panel a reports domain-matched full Z+ZZ AUC for ideal, Baihua, and
  Shenglian features across all five circuit conditions.
- Panel b decomposes the Shenglian result into Z-only, ZZ-only, and Z+ZZ
  readouts under the same splits, training schedule, and validation ranking.

Archetype: two-panel stacked quantitative grid.

Backend/export: Python/matplotlib only; single-column editable PDF/SVG,
450-dpi PNG, and 600-dpi LZW-TIFF.

Review risks: Z and ZZ have unequal widths (6 and 15); the readout is fitted
separately to each feature domain and observable family; the ten seeds are
nested scaffold/readout analyses rather than hardware repetitions; raw ZZ
expectations are not connected correlators and do not establish entanglement.
"""
    (RESULT_DIR / "MAIN_HARDWARE_AUC_FIGURE_CONTRACT.md").write_text(
        contract,
        encoding="utf-8",
    )
    legend = """# Main figure legend

**Canonical-v3 hardware task readability and its observable-family
decomposition.** **a,** Test ROC-AUC from the full 21-dimensional Z+ZZ
features for ideal simulation, Baihua, and Shenglian. **b,** Shenglian test
ROC-AUC after retaining the six one-body Z features, the 15 two-body ZZ
features, or all 21 Z+ZZ features. Every domain and family receives one
bias-containing Linear(D,1) readout under the same scaffold splits, ten
readout seeds, 500 epochs, and validation ranking. Large markers and error
bars show means and sample standard deviations; pale points show nested-seed
values. The seeds reuse one feature matrix per hardware task and are not
independent device repetitions. Fold3 preserves the ideal L1 unitary while
increasing the nominal CX count from 60 to 180. Z and ZZ have unequal feature
dimensions, and raw ZZ expectations do not constitute an entanglement
measure. Dashed lines mark chance.

Source data: `mainline_hardware_q6_auc_observable_ablation_en_source_data.csv`.
"""
    (FIGURE_DIR / f"{OUTPUT_STEM}_legend.md").write_text(
        legend,
        encoding="utf-8",
    )
    qa = {
        "core_conclusion_defined": True,
        "archetype": "two-panel stacked quantitative grid",
        "backend": "Python/matplotlib",
        "manual_or_generative_plot_editing": False,
        "source_row_count": int(len(source)),
        "panel_a_rows": int((source["panel"] == "a").sum()),
        "panel_b_rows": int((source["panel"] == "b").sum()),
        "tasks": list(TASKS),
        "nested_seed_count_per_cell": 10,
        "observable_family_dimensions": FAMILY_DIMS,
        "independent_hardware_repeat_claim": False,
        "raw_ZZ_entanglement_claim": False,
    }
    (RESULT_DIR / "MAIN_HARDWARE_AUC_FIGURE_QA.json").write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    main_metrics = load_main_metrics()
    ablation_metrics = load_ablation_metrics()
    source = build_source_data(main_metrics, ablation_metrics)
    copy_manuscript_tables()
    draw_figure(main_metrics, ablation_metrics)
    write_metadata(source)
    print(
        f"Wrote {OUTPUT_STEM} with {len(source)} source rows to "
        f"{MANUSCRIPT_FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
