#!/usr/bin/env python3
"""Publication figures for six-seed closed-reservoir fixed-history scans."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "outputs" / "strict_decoder_only"
FIGURE_ROOT = ROOT / "figures" / "strict_six_seed"
SOURCE_ROOT = FIGURE_ROOT / "source_data"
TARGETS = ("gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs")
TARGET_LABELS = {
    "gap_1fs": r"$\Delta E$, 1 fs",
    "gap_5fs": r"$\Delta E$, 5 fs",
    "gap_10fs": r"$\Delta E$, 10 fs",
    "min_gap_next_5fs": "Minimum gap, next 5 fs",
}
SYSTEMS = ("qrc", "esn_standard", "esn_second_order_recurrent", "fp_cvr")
SYSTEM_LABELS = {
    "qrc": "Closed DTC-QRC",
    "esn_standard": "Standard ESN",
    "esn_second_order_recurrent": "Second-order ESN",
    "fp_cvr": "FP-CVR",
}
SYSTEM_COLORS = {
    "qrc": "#244A86",
    "esn_standard": "#D3744F",
    "esn_second_order_recurrent": "#2A9D8F",
    "fp_cvr": "#8A7FA8",
}
SYSTEM_MARKERS = {"qrc": "o", "esn_standard": "s", "esn_second_order_recurrent": "^", "fp_cvr": "D"}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 7.6,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.22, 1.06, label, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_ROOT / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def absolute_history_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.25), sharex=True)
    for row, decoder in enumerate(("rbf_svr", "linear")):
        for column, target in enumerate(TARGETS):
            ax = axes[row, column]
            frame = summary[(summary.decoder == decoder) & (summary.target == target)]
            for system in SYSTEMS:
                sub = frame[frame.system == system].sort_values("t")
                x = sub.t.to_numpy(float)
                y = sub.mae_ev_mean.to_numpy(float)
                sd = sub.mae_ev_sample_std.to_numpy(float)
                ax.plot(x, y, color=SYSTEM_COLORS[system], marker=SYSTEM_MARKERS[system],
                        ms=2.7, lw=1.0)
                ax.fill_between(x, y - sd, y + sd, color=SYSTEM_COLORS[system],
                                alpha=0.08, linewidth=0)
            ax.grid(axis="y", color="#E8E8E8", lw=0.5)
            ax.set_title(TARGET_LABELS[target])
            ax.set_xticks([1, 4, 8, 12, 16, 20])
            panel_label(ax, chr(ord("a") + row * 4 + column))
            if column == 0:
                ax.set_ylabel(("RBF-SVR" if row == 0 else "Linear") + "\ntest MAE (eV)")
            if row == 1:
                ax.set_xlabel("Visible frames, T")
    handles = [
        Line2D([0], [0], color=SYSTEM_COLORS[system], marker=SYSTEM_MARKERS[system],
               lw=1.0, ms=3, label=SYSTEM_LABELS[system])
        for system in SYSTEMS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Closed reservoirs at matched physical history length (p=0, final readout)",
                 x=0.075, y=0.995, ha="left", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.995, top=0.90, bottom=0.18,
                        wspace=0.34, hspace=0.35)
    save_figure(fig, "fig_closed_hopping_fixed_history_absolute")


def paired_history_figure(paired: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.25), sharex=True, sharey="row")
    baselines = ("esn_standard", "esn_second_order_recurrent", "fp_cvr")
    for row, decoder in enumerate(("rbf_svr", "linear")):
        for column, target in enumerate(TARGETS):
            ax = axes[row, column]
            frame = paired[(paired.decoder == decoder) & (paired.target == target)]
            for baseline in baselines:
                sub = frame[frame.baseline == baseline].sort_values("t")
                x = sub.t.to_numpy(float)
                y = sub.mean_qrc_minus_baseline_trajectory_mae_ev.to_numpy(float)
                low = sub.crossed_bootstrap_95ci_low_ev.to_numpy(float)
                high = sub.crossed_bootstrap_95ci_high_ev.to_numpy(float)
                ax.plot(x, y, color=SYSTEM_COLORS[baseline], marker=SYSTEM_MARKERS[baseline],
                        ms=2.7, lw=1.0)
                ax.fill_between(x, low, high, color=SYSTEM_COLORS[baseline],
                                alpha=0.08, linewidth=0)
            ax.axhline(0.0, color="#858585", lw=0.7, ls="--")
            ax.grid(axis="y", color="#E8E8E8", lw=0.5)
            ax.set_title(TARGET_LABELS[target])
            ax.set_xticks([1, 4, 8, 12, 16, 20])
            panel_label(ax, chr(ord("a") + row * 4 + column))
            if column == 0:
                ax.set_ylabel(("RBF-SVR" if row == 0 else "Linear") + "\nQRC − control MAE (eV)")
            if row == 1:
                ax.set_xlabel("Visible frames, T")
    handles = [
        Line2D([0], [0], color=SYSTEM_COLORS[baseline], marker=SYSTEM_MARKERS[baseline],
               lw=1.0, ms=3, label=SYSTEM_LABELS[baseline])
        for baseline in baselines
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Paired QRC effects are evaluated only within the same T",
                 x=0.075, y=0.995, ha="left", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.995, top=0.90, bottom=0.18,
                        wspace=0.22, hspace=0.35)
    save_figure(fig, "fig_closed_hopping_fixed_history_paired_effects")


def main() -> None:
    configure_style()
    metrics = pd.read_csv(DATA_ROOT / "reservoir_metrics_all_fixed_histories.csv")
    summary = pd.read_csv(DATA_ROOT / "reservoir_summary_all_fixed_histories.csv")
    paired = pd.read_csv(DATA_ROOT / "qrc_minus_controls_paired_by_fixed_history.csv")
    if set(metrics.seed) != {0, 10, 20, 30, 40, 50}:
        raise RuntimeError("six-seed contract is incomplete")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(SOURCE_ROOT / "closed_hopping_seed_metrics_by_fixed_T.csv", index=False)
    summary.to_csv(SOURCE_ROOT / "closed_hopping_summary_by_fixed_T.csv", index=False)
    paired.to_csv(SOURCE_ROOT / "closed_hopping_paired_effects_by_fixed_T.csv", index=False)
    absolute_history_figure(summary)
    paired_history_figure(paired)
    manifest = {
        "backend": "Python/matplotlib",
        "protocol": "six paired seeds; p=0; final readout; every T retained",
        "history_selection": False,
        "figures": [
            "fig_closed_hopping_fixed_history_absolute",
            "fig_closed_hopping_fixed_history_paired_effects",
        ],
        "formats": ["svg", "pdf", "png", "tiff"],
    }
    (FIGURE_ROOT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(FIGURE_ROOT)


if __name__ == "__main__":
    main()
