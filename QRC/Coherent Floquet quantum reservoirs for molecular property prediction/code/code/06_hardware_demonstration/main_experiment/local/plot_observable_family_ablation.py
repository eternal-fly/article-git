#!/usr/bin/env python3
"""Summarize and plot the Q6 Z-only versus ZZ-only observable ablation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "local_results_five_tasks" / "observable_family_ablation"
FIGURE_DIR = RESULT_DIR / "figures"
BACKENDS = ("baihua", "shenglian")
DOMAINS = ("ideal", "baihua", "shenglian")
DOMAIN_LABELS = {"ideal": "Ideal", "baihua": "Baihua", "shenglian": "Shenglian"}
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
FAMILIES = ("Z", "ZZ", "Z+ZZ")
COLORS = {"Z": "#00A087", "ZZ": "#E64B35", "Z+ZZ": "#3C5488"}
MARKERS = {"Z": "s", "ZZ": "o", "Z+ZZ": "D"}
OFFSETS = {"Z": -0.16, "ZZ": 0.0, "Z+ZZ": 0.16}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.1,
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
    frames = [
        pd.read_csv(RESULT_DIR / f"observable_family_metrics_by_seed_{backend}.csv")
        for backend in BACKENDS
    ]
    data = pd.concat(frames, ignore_index=True)
    ideal = (
        data[data["domain"] == "ideal"]
        .sort_values(["task_key", "observable_family", "seed"])
        .drop_duplicates(["task_key", "observable_family", "seed"])
    )
    measured = data[data["domain"] != "ideal"].copy()
    data = pd.concat([ideal, measured], ignore_index=True)
    expected_rows = len(DOMAINS) * len(TASKS) * len(FAMILIES) * 10
    if len(data) != expected_rows:
        raise AssertionError((len(data), expected_rows))
    if set(data["domain"]) != set(DOMAINS):
        raise AssertionError("Domain mismatch")
    if set(data["observable_family"]) != set(FAMILIES):
        raise AssertionError("Observable-family mismatch")
    if not np.isfinite(data["test_auc"]).all():
        raise AssertionError("Nonfinite AUC")
    return data


def percentile_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def build_tables(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        data.groupby(
            ["domain", "task_key", "layers", "fold_factor", "observable_family", "feature_dim"],
            as_index=False,
        )
        .agg(
            nested_seed_count=("seed", "nunique"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd_nested=("test_auc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_sd_nested=("test_acc", "std"),
        )
    )
    summary.to_csv(RESULT_DIR / "observable_family_summary.csv", index=False)

    paired_rows = []
    comparisons = (
        ("Z+ZZ", "Z"),
        ("Z+ZZ", "ZZ"),
        ("ZZ", "Z"),
    )
    for (domain, task_key), group in data.groupby(["domain", "task_key"]):
        pivot = group.pivot(index="seed", columns="observable_family", values="test_auc")
        pivot = pivot.loc[sorted(pivot.index), list(FAMILIES)]
        if pivot.shape != (10, 3) or pivot.isna().any().any():
            raise AssertionError((domain, task_key, pivot))
        for comparison_index, (left, right) in enumerate(comparisons):
            values = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            low, high = percentile_interval(
                values,
                seed=73_000 + 100 * list(DOMAINS).index(domain)
                + 10 * list(TASKS).index(task_key)
                + comparison_index,
            )
            paired_rows.append(
                {
                    "domain": domain,
                    "task_key": task_key,
                    "left_family": left,
                    "right_family": right,
                    "contrast": f"{left} minus {right}",
                    "nested_seed_count": len(values),
                    "paired_auc_difference_mean": float(values.mean()),
                    "paired_auc_difference_sd_nested": float(values.std(ddof=1)),
                    "nested_seed_bootstrap_ci_low": low,
                    "nested_seed_bootstrap_ci_high": high,
                    "positive_seed_count": int(np.sum(values > 0)),
                    "zero_seed_count": int(np.sum(values == 0)),
                    "negative_seed_count": int(np.sum(values < 0)),
                    "inference_boundary": (
                        "descriptive split/readout sensitivity; not a hardware-repeat CI"
                        if domain != "ideal"
                        else "descriptive paired scaffold/readout-seed interval"
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(RESULT_DIR / "observable_family_paired_effects.csv", index=False)

    aggregate_rows = []
    for domain in DOMAINS:
        for family in FAMILIES:
            selected = summary[
                (summary["domain"] == domain)
                & (summary["observable_family"] == family)
            ].set_index("task_key")
            l3 = selected.loc[list(TASKS[:3]), "test_auc_mean"]
            aggregate_rows.append(
                {
                    "domain": domain,
                    "observable_family": family,
                    "feature_dim": int(selected.iloc[0]["feature_dim"]),
                    "l3_mean_across_acquisition_batch_means": float(l3.mean()),
                    "l3_sd_across_acquisition_batch_means": float(l3.std(ddof=1)),
                    "l1_base_auc_mean_nested": float(
                        selected.loc[TASKS[3], "test_auc_mean"]
                    ),
                    "l1_fold3_auc_mean_nested": float(
                        selected.loc[TASKS[4], "test_auc_mean"]
                    ),
                    "l1_fold3_minus_base_auc_mean_nested": float(
                        selected.loc[TASKS[4], "test_auc_mean"]
                        - selected.loc[TASKS[3], "test_auc_mean"]
                    ),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        RESULT_DIR / "observable_family_aggregate_summary.csv",
        index=False,
    )

    depth_rows = []
    for (domain, family), group in data.groupby(
        ["domain", "observable_family"]
    ):
        selected = group[group["task_key"].isin(TASKS[3:])].pivot(
            index="seed",
            columns="task_key",
            values="test_auc",
        )
        selected = selected.loc[sorted(selected.index), list(TASKS[3:])]
        if selected.shape != (10, 2) or selected.isna().any().any():
            raise AssertionError((domain, family, selected))
        values = (
            selected[TASKS[4]] - selected[TASKS[3]]
        ).to_numpy(dtype=float)
        low, high = percentile_interval(
            values,
            seed=91_000
            + 100 * list(DOMAINS).index(domain)
            + list(FAMILIES).index(family),
        )
        depth_rows.append(
            {
                "domain": domain,
                "observable_family": family,
                "feature_dim": int(
                    group["feature_dim"].drop_duplicates().item()
                ),
                "nested_seed_count": len(values),
                "fold3_minus_base_auc_mean": float(values.mean()),
                "fold3_minus_base_auc_sd_nested": float(values.std(ddof=1)),
                "nested_seed_bootstrap_ci_low": low,
                "nested_seed_bootstrap_ci_high": high,
                "positive_seed_count": int(np.sum(values > 0)),
                "zero_seed_count": int(np.sum(values == 0)),
                "negative_seed_count": int(np.sum(values < 0)),
                "inference_boundary": (
                    "one paired hardware feature acquisition; nested-seed "
                    "analysis is not a hardware-repeat CI"
                    if domain != "ideal"
                    else "ideal-equivalent base/fold3 feature matrices"
                ),
            }
        )
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(RESULT_DIR / "observable_family_depth_effects.csv", index=False)

    data.to_csv(FIGURE_DIR / "observable_family_ablation_source_data.csv", index=False)
    return summary, paired, aggregate, depth


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
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


def draw(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), sharey=True)
    xbase = np.arange(len(TASKS), dtype=float)
    for panel_index, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        panel = data[data["domain"] == domain]
        for family in FAMILIES:
            values = []
            for task_key in TASKS:
                selected = panel[
                    (panel["task_key"] == task_key)
                    & (panel["observable_family"] == family)
                ].sort_values("seed")["test_auc"].to_numpy(dtype=float)
                if len(selected) != 10:
                    raise AssertionError((domain, task_key, family, len(selected)))
                values.append(selected)
            means = np.asarray([item.mean() for item in values])
            sds = np.asarray([item.std(ddof=1) for item in values])
            ax.plot(
                xbase + OFFSETS[family],
                means,
                color=COLORS[family],
                marker=MARKERS[family],
                markersize=4.4,
                label=family,
                zorder=3,
            )
            ax.errorbar(
                xbase + OFFSETS[family],
                means,
                yerr=sds,
                fmt="none",
                ecolor=COLORS[family],
                elinewidth=0.9,
                capsize=2.0,
                zorder=2.8,
            )
            jitter = np.linspace(-0.025, 0.025, 10)
            for x, seed_values in zip(xbase, values):
                ax.scatter(
                    x + OFFSETS[family] + jitter,
                    seed_values,
                    s=5.2,
                    color=COLORS[family],
                    alpha=0.20,
                    linewidths=0,
                    zorder=2.4,
                )
        ax.axhline(0.5, color="#BDBDBD", ls="--", lw=0.8, zorder=1)
        ax.grid(axis="y", color="#EAEAEA", lw=0.6, zorder=0)
        ax.set_xticks(xbase, TASK_LABELS)
        ax.set_title(DOMAIN_LABELS[domain], pad=5)
        ax.set_ylim(0.30, 0.90)
        ax.text(
            -0.13,
            1.04,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            weight="bold",
            fontsize=8.2,
        )
    axes[0].set_ylabel("Test ROC–AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.45,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.0)
    save_figure(fig, FIGURE_DIR / "hardware_observable_family_ablation_en")


def write_metadata(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    aggregate: pd.DataFrame,
    depth: pd.DataFrame,
) -> None:
    contract = """# Observable-family ablation figure contract

Core conclusion: ideal L3 performance benefits from combining Z and ZZ,
whereas the hardware partition is processor and depth dependent. Shenglian's
strong L1-base result is concentrated in ZZ, and its folded-depth loss is
dominated by degradation of ZZ-only readability; Baihua remains weak and
batch dependent.

Evidence chain: panels a-c show the complete paired Z-only, ZZ-only, and Z+ZZ
test-AUC results for ideal, Baihua, and Shenglian across all five circuit
conditions. Large markers are nested-seed means, error bars are sample standard
deviations, and light points are the ten nested scaffold/readout analyses.

Archetype: three-panel quantitative grid.

Backend/export: Python/matplotlib only; editable PDF/SVG, 450-dpi PNG, and
600-dpi LZW-TIFF.

Review risks: Z and ZZ have unequal retained dimensions (6 versus 15);
domain-matched linear readouts diagnose task readability rather than
ideal-coordinate preservation; nested seeds and acquisition folders are not
independent device repeats; paired intervals are descriptive analysis
sensitivity and not hardware-repeat confidence intervals.
"""
    (RESULT_DIR / "OBSERVABLE_FAMILY_FIGURE_CONTRACT.md").write_text(
        contract,
        encoding="utf-8",
    )
    legend = """# Figure legend

**One- and two-body observables contribute differently across ideal and
hardware feature domains.** Test ROC–AUC for readouts retaining only the six
one-body Z features, only the 15 two-body ZZ features, or all 21 Z+ZZ features
in the (a) ideal, (b) Baihua, and (c) Shenglian domains. Each domain and
observable family receives its own single Linear(D,1) readout under the same
scaffold splits, ten readout seeds, 500 training epochs, and validation
ranking. Large markers and error bars show mean ± sample standard deviation;
light points are the nested seed values. These are analysis seeds applied to a
fixed feature matrix, not hardware repetitions. Z and ZZ are not
dimension-matched (D=6 and D=15). Dashed lines mark chance.

Source data: `observable_family_ablation_source_data.csv`.
"""
    (FIGURE_DIR / "hardware_observable_family_ablation_legend.md").write_text(
        legend,
        encoding="utf-8",
    )
    qa = {
        "summary_row_count": int(len(summary)),
        "paired_effect_row_count": int(len(paired)),
        "aggregate_summary_row_count": int(len(aggregate)),
        "depth_effect_row_count": int(len(depth)),
        "source_data_row_count": 450,
        "domains": list(DOMAINS),
        "tasks": list(TASKS),
        "observable_families": {"Z": 6, "ZZ": 15, "Z+ZZ": 21},
        "nested_seed_count_per_cell": 10,
        "independent_hardware_repeat_claim": False,
        "figure_backend": "Python/matplotlib",
        "manual_or_generative_plot_editing": False,
    }
    (RESULT_DIR / "OBSERVABLE_FAMILY_QA.json").write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    summary, paired, aggregate, depth = build_tables(data)
    draw(data)
    write_metadata(summary, paired, aggregate, depth)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
