#!/usr/bin/env python3
"""Six-seed replacement for the architecture + hopping main-evidence figure."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_revised_hopping import (
    FIGURES,
    HOPPING_ROOT,
    METHOD_COLORS,
    SOURCE_DATA,
    TARGET_COLORS,
    TARGET_LABELS,
    TARGET_ORDER,
    apply_style,
    draw_architecture,
    panel_label,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = HOPPING_ROOT / "outputs" / "strict_decoder_only"
    paired = pd.read_csv(strict / "qrc_minus_esn_paired_all_fixed_axes.csv")
    optima = pd.read_csv(strict / "qrc_p_scan_optima_by_fixed_decoder_readout.csv")
    classical = pd.read_csv(HOPPING_ROOT / "outputs" / "classical_test.csv")
    raw_history_svr = classical[classical.model == "svr_history_10fs"].copy()
    if len(raw_history_svr) != len(TARGET_ORDER):
        raise RuntimeError("fixed raw-history SVR rows are incomplete")
    return optima, paired, raw_history_svr


def plot() -> None:
    apply_style()
    optima, paired, classical = load_data()
    fixed_effect = optima[
        (optima.decoder == "rbf_svr") & (optima.readout == "final")
    ].copy()
    scan = paired[
        (paired.decoder == "rbf_svr") & (paired.readout == "final")
    ].copy()
    if len(fixed_effect) != len(TARGET_ORDER) or set(fixed_effect.num_seeds) != {6}:
        raise RuntimeError("six-seed summary is incomplete")

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    fixed_effect.to_csv(
        SOURCE_DATA / "hopping_six_seed_rbf_final_descriptive_p_star.csv", index=False
    )
    scan.to_csv(SOURCE_DATA / "hopping_six_seed_rbf_final_p_scan.csv", index=False)
    classical.to_csv(SOURCE_DATA / "hopping_fixed_raw_history_svr.csv", index=False)

    fig = plt.figure(figsize=(7.0, 4.85))
    grid = fig.add_gridspec(
        2, 3, height_ratios=(0.86, 1.65), left=0.075, right=0.995,
        top=0.97, bottom=0.14, hspace=0.40, wspace=0.45,
    )
    architecture = fig.add_subplot(grid[0, :])
    draw_architecture(architecture)
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]

    # b: absolute performance at each target's descriptive scan optimum.
    ax = axes[0]
    x = np.arange(len(TARGET_ORDER))
    width = 0.19
    closed_color = "#A65F21"
    methods = (
        ("qrc", r"DTC-QRC at $p^*$", -1.5 * width),
        ("closed", "Closed DTC-QRC", -0.5 * width),
        ("esn", "Matched ESN", 0.5 * width),
        ("classical", "Raw-history\nRBF-SVR", 1.5 * width),
    )
    for system, label, offset in methods:
        means, errors = [], []
        for target in TARGET_ORDER:
            if system == "classical":
                row = classical[classical.target == target].iloc[0]
                means.append(float(row.mae_ev)); errors.append(np.nan)
            elif system in ("qrc", "closed", "esn"):
                row = fixed_effect[fixed_effect.target == target].iloc[0]
                if system == "qrc":
                    means.append(float(row.qrc_mae_ev_mean))
                    errors.append(float(row.qrc_mae_ev_sample_std))
                elif system == "closed":
                    means.append(float(row.closed_qrc_mae_ev_mean))
                    errors.append(float(row.closed_qrc_mae_ev_sample_std))
                else:
                    means.append(float(row.esn_mae_ev_mean))
                    errors.append(float(row.esn_mae_ev_sample_std))
        color = closed_color if system == "closed" else METHOD_COLORS[system]
        ax.bar(x + offset, means, width, color=color, label=label,
               edgecolor="white", linewidth=0.4)
        finite = np.isfinite(errors)
        if finite.any():
            ax.errorbar((x + offset)[finite], np.asarray(means)[finite],
                        yerr=np.asarray(errors)[finite], fmt="none", ecolor="#333333",
                        elinewidth=0.7, capsize=1.8)
    ordered = fixed_effect.set_index("target").loc[list(TARGET_ORDER)]
    for x_value, (_, row) in zip(x - 1.5 * width, ordered.iterrows()):
        ax.text(x_value, float(row.qrc_mae_ev_mean) + 0.025,
                rf"$p^*={float(row.p_star_test_grid_argmin):g}$",
                ha="center", va="bottom", fontsize=4.6,
                color=METHOD_COLORS["qrc"], rotation=90)
    ax.set_xticks(x, [TARGET_LABELS[target] for target in TARGET_ORDER])
    ax.tick_params(axis="x", labelrotation=24)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title(r"Target-wise descriptive $p^*$", fontsize=6.8)
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    panel_label(ax, "b")

    # c: paired effects at target-wise p*, with scan multiplicity visible.
    ax = axes[1]
    y = np.arange(len(TARGET_ORDER))[::-1]
    y_esn, y_closed = y + 0.11, y - 0.11
    ax.hlines(y_esn, ordered.simultaneous_95band_low_ev,
              ordered.simultaneous_95band_high_ev,
              color="#9FB2CA", linewidth=5.0, alpha=0.45)
    ax.hlines(y_esn, ordered.pointwise_95ci_low_ev,
              ordered.pointwise_95ci_high_ev,
              color=METHOD_COLORS["qrc"], linewidth=1.35)
    ax.plot(ordered.mean_qrc_minus_esn_trajectory_mae_ev, y_esn, marker="s",
            color=METHOD_COLORS["qrc"], linestyle="none", markersize=4.0)
    ax.hlines(y_closed, ordered.closed_simultaneous_95band_low_ev,
              ordered.closed_simultaneous_95band_high_ev,
              color="#E7B982", linewidth=5.0, alpha=0.45)
    ax.hlines(y_closed, ordered.closed_pointwise_95ci_low_ev,
              ordered.closed_pointwise_95ci_high_ev,
              color=closed_color, linewidth=1.35)
    ax.plot(ordered.mean_p_star_minus_closed_trajectory_mae_ev, y_closed,
            marker="o", color=closed_color, linestyle="none", markersize=4.0)
    ax.axvline(0.0, color="#8C8C8C", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [TARGET_LABELS[target] for target in TARGET_ORDER])
    ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlabel(r"DTC-QRC($p^*$) $-$ baseline MAE (eV)")
    ax.set_title("Pointwise CI; scan-wide simultaneous band", fontsize=6.6)
    ax.grid(axis="x", color="#E7E7E7", lw=0.55)
    ax.legend(
        handles=[
            Line2D([0], [0], color=METHOD_COLORS["qrc"], lw=1.35,
                   marker="s", ms=3.5, label=r"$p^*$ QRC − ESN"),
            Line2D([0], [0], color=closed_color, lw=1.35,
                   marker="o", ms=3.5, label=r"$p^*$ QRC − closed"),
            Line2D([0], [0], color="#999999", lw=5.0, alpha=0.35,
                   label="Pale: scan-wide band"),
        ],
        loc="best", fontsize=4.6, handlelength=1.4, labelspacing=0.25,
    )
    panel_label(ax, "c")

    # d: full p response; every target-specific p* remains visibly post hoc.
    ax = axes[2]
    for target in TARGET_ORDER:
        sub = scan[scan.target == target].sort_values("p")
        x_p = sub.p.to_numpy(float)
        mean = sub.mean_qrc_minus_esn_trajectory_mae_ev.to_numpy(float)
        low = sub.crossed_bootstrap_95ci_low_ev.to_numpy(float)
        high = sub.crossed_bootstrap_95ci_high_ev.to_numpy(float)
        ax.plot(x_p, mean, color=TARGET_COLORS[target], marker="o", ms=2.8,
                lw=1.05, label=TARGET_LABELS[target].replace("\n", " "))
        ax.fill_between(x_p, low, high, color=TARGET_COLORS[target], alpha=0.10,
                        linewidth=0)
        selected = fixed_effect[fixed_effect.target == target].iloc[0]
        ax.plot(float(selected.p_star_test_grid_argmin),
                float(selected.mean_qrc_minus_esn_trajectory_mae_ev),
                marker="*", ms=6.0, color=TARGET_COLORS[target],
                markeredgecolor="white", markeredgewidth=0.35, zorder=5)
    ax.axhline(0.0, color="#8C8C8C", linestyle="--", linewidth=0.8)
    ax.set_xscale("linear")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"])
    ax.set_xlabel("Reset probability, p")
    ax.set_ylabel("QRC − ESN trajectory MAE (eV)")
    ax.set_title("Complete fixed-axis reset response")
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    ax.legend(loc="best", fontsize=4.8, handlelength=1.25,
              labelspacing=0.25)
    panel_label(ax, "d")

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["qrc"], lw=5,
               label=r"DTC-QRC at target $p^*$"),
        Line2D([0], [0], color=closed_color, lw=5, label="Closed DTC-QRC"),
        Line2D([0], [0], color=METHOD_COLORS["esn"], lw=5, label="Matched ESN"),
        Line2D([0], [0], color=METHOD_COLORS["classical"], lw=5,
               label="Fixed raw-history RBF-SVR"),
        Line2D([0], [0], color="#555555", marker="*", ls="none", ms=6,
               label=r"Target-wise descriptive $p^*$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.01))
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "fig2_architecture_surface_hopping"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    plot()
