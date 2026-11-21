#!/usr/bin/env python3
"""Build the fixed-architecture cross-event and coherence main figure.

The script reads only manuscript source-data tables.  It deliberately excludes
all nondefault interaction-strength diagnostics from the application paper.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


if (ROOT / "source_data" / "supplement").exists():
    SOURCE = ROOT / "source_data" / "supplement"
    FIGURES = ROOT / "fig"
    OUTPUT_NAME = "fig4_cross_event_coherence"
elif (ROOT.parent / "source_data" / "paper" / "supplement").exists():
    SOURCE = ROOT.parent / "source_data" / "paper" / "supplement"
    FIGURES = ROOT.parent / "figures" / "main"
    OUTPUT_NAME = "04_cross_event_coherence"
else:
    raise FileNotFoundError("fixed-architecture source-data directory not found")

BLUE = "#31598F"
ORANGE = "#D0872E"
TEAL = "#4A918B"
RED = "#D95F45"
GRAY = "#8C8C8C"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # The source canvas is exported at its final 89-mm single-column
            # width, so these point sizes are also the printed point sizes.
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#B8B8B8",
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.24,
        1.075,
        label,
        transform=ax.transAxes,
        fontsize=8.7,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def cross_event_panel(ax: plt.Axes, frame: pd.DataFrame, family: str, title: str) -> None:
    styles = {
        "open_dtc_qrc": (BLUE, "o", "DTC-QRC"),
        "fp_cvr": (RED, "s", "FP-CVR"),
        "matched_esn": (TEAL, "^", "Matched ESN"),
    }
    subset = frame[frame["family"] == family]
    for method, (color, marker, label) in styles.items():
        selected = subset[subset["method"] == method].sort_values("lag")
        x = selected["lag"].to_numpy(float)
        y = selected["test_auc_mean"].to_numpy(float)
        sd = selected["test_auc_sd"].to_numpy(float)
        ax.plot(x, y, color=color, marker=marker, lw=1.35, ms=3.7, label=label, zorder=3)
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, lw=0, zorder=1)
    ax.axhline(0.5, color=GRAY, lw=0.85, ls="--")
    ax.set_xticks([1, 2, 4, 6, 8, 12, 16])
    ax.set_ylim(0.43, 1.01)
    ax.set_xlabel("Event separation")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title(title, pad=4.0)
    ax.set_box_aspect(1)
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)


def dephasing_bace_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for p, color, marker, label in (
        (0.0, BLUE, "o", "Closed, p=0"),
        (0.1, ORANGE, "s", "Dissipative, p=0.10"),
    ):
        selected = frame[np.isclose(frame["p"], p)].sort_values("control_value")
        x = selected["control_value"].to_numpy(float)
        y = selected["mean"].to_numpy(float)
        low = selected["ci_low"].to_numpy(float)
        high = selected["ci_high"].to_numpy(float)
        ax.plot(x, y, color=color, marker=marker, lw=1.35, ms=3.8, label=label, zorder=3)
        ax.fill_between(x, low, high, color=color, alpha=0.15, lw=0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(0.44, 0.84)
    ax.set_xlabel(r"Event dephasing, $\lambda$")
    ax.set_ylabel("BACE test ROC-AUC")
    ax.set_title("Drug-molecule\nevent stream", pad=3.0)
    ax.set_box_aspect(1)
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)


def dephasing_hopping_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    final = frame[(frame["readout"] == "final") & (frame["decoder"] == "RBF-SVR")]
    for p, color, marker, label in (
        (0.0, BLUE, "o", "Closed, p=0"),
        (0.1, ORANGE, "s", "Dissipative, p=0.10"),
    ):
        selected = final[np.isclose(final["p"], p)].sort_values("control_value")
        x = selected["control_value"].to_numpy(float)
        y = selected["mean"].to_numpy(float)
        low = selected["ci_low"].to_numpy(float)
        high = selected["ci_high"].to_numpy(float)
        ax.plot(x, y, color=color, marker=marker, lw=1.35, ms=3.8, label=label, zorder=3)
        ax.fill_between(x, low, high, color=color, alpha=0.15, lw=0)
    ax.axhline(1.0, color=GRAY, lw=0.85, ls="--")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(0.97, 1.36)
    ax.set_xlabel(r"Event dephasing, $\lambda$")
    ax.set_ylabel("Target-balanced MAE ratio")
    ax.set_title("Surface-hopping\nevent stream", pad=3.0)
    ax.set_box_aspect(1)
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)


def main() -> None:
    set_style()
    cross = pd.read_csv(SOURCE / "controlled_cross_event_p010" / "summary_by_lag.csv")
    bace = pd.read_csv(
        SOURCE / "coherence_dephasing_fixed_architecture" / "mechanism_bace_auc.csv"
    )
    hopping = pd.read_csv(
        SOURCE
        / "coherence_dephasing_fixed_architecture"
        / "mechanism_hopping_ratio_summary.csv"
    )

    # Export at the final APS single-column width (89 mm).  The taller canvas
    # gives every data region a square box instead of relying on REVTeX to
    # shrink a wide 4:3 source figure.
    fig, axes = plt.subplots(2, 2, figsize=(3.50, 4.25))
    cross_event_panel(axes[0, 0], cross, "delayed_motif", "Delayed motif")
    cross_event_panel(axes[0, 1], cross, "distant_pair", "Distant event pair")
    dephasing_bace_panel(axes[1, 0], bace)
    dephasing_hopping_panel(axes[1, 1], hopping)
    for label, ax in zip("abcd", axes.flat):
        panel_label(ax, label)

    cross_handles = [
        Line2D([0], [0], color=BLUE, marker="o", lw=1.35, ms=3.7, label="DTC-QRC"),
        Line2D([0], [0], color=RED, marker="s", lw=1.35, ms=3.7, label="FP-CVR"),
        Line2D([0], [0], color=TEAL, marker="^", lw=1.35, ms=3.7, label="Matched ESN"),
    ]
    reset_handles = [
        Line2D([0], [0], color=BLUE, marker="o", lw=1.35, ms=3.7, label="Closed, p=0"),
        Line2D(
            [0], [0], color=ORANGE, marker="s", lw=1.35, ms=3.7,
            label="Dissipative, p=0.10",
        ),
    ]
    # Matplotlib fills multi-column legends column-first.  This ordering makes
    # the first displayed row the three model baselines and the second row the
    # two channel conditions.
    legend_handles = [
        cross_handles[0], reset_handles[0],
        cross_handles[1], reset_handles[1],
        cross_handles[2],
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.012),
        columnspacing=0.68,
        handlelength=1.25,
        handletextpad=0.32,
        borderpad=0.26,
    )
    fig.subplots_adjust(
        left=0.165,
        right=0.985,
        top=0.952,
        bottom=0.165,
        hspace=0.50,
        wspace=0.40,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / OUTPUT_NAME
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    for suffix in (".pdf", ".svg", ".png", ".tiff"):
        print(stem.with_suffix(suffix))


if __name__ == "__main__":
    main()
