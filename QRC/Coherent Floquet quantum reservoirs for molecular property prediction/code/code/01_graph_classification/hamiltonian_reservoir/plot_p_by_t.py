"""Publication figures for the fixed-Q p-by-T diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_p_by_t import ROOT, load_all
from protocol import FIGURE_ROOT, OPEN_P, PT_P_VALUES, PT_T_VALUES


OUTPUT = FIGURE_ROOT / "p_by_T_diagnostic"
DATASET_LABELS = {"BACE": "BACE", "BBBP": "BBBP"}
DECODER_LABELS = {"linear": "Linear readout", "rbf_svr": "RBF-SVR"}
T_COLORS = {
    6: "#3B5B92",
    12: "#2A9D8F",
    20: "#E9A23B",
    30: "#C95A49",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["dataset", "decoder", "t", "p"], as_index=False)
        .agg(mean_auc=("test_auc", "mean"), sd_auc=("test_auc", "std"))
        .sort_values(["dataset", "decoder", "t", "p"])
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.16, 1.05, letter, transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="top", ha="left",
    )


def export(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=600, bbox_inches="tight")


def plot_curves(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.35), sharex=True, sharey=True)
    letters = iter("abcd")
    for row, dataset in enumerate(("BACE", "BBBP")):
        for column, decoder in enumerate(("linear", "rbf_svr")):
            ax = axes[row, column]
            subset = table[(table.dataset == dataset) & (table.decoder == decoder)]
            for t in PT_T_VALUES:
                values = subset[subset.t == t]
                x = values.p.to_numpy()
                mean = values.mean_auc.to_numpy()
                sd = values.sd_auc.to_numpy()
                ax.plot(
                    x, mean, marker="o", color=T_COLORS[t], label=f"T={t}",
                    markeredgecolor="white", markeredgewidth=0.35,
                )
                ax.fill_between(x, mean - sd, mean + sd, color=T_COLORS[t], alpha=0.11, linewidth=0)
            ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
            ax.axvline(OPEN_P, color="#222222", linestyle=":", linewidth=0.9, zorder=0)
            ax.text(
                OPEN_P + 0.008, 0.505, "fixed p=.10", rotation=90,
                fontsize=6.5, va="bottom", ha="left", color="#333333",
            )
            ax.set_title(f"{DATASET_LABELS[dataset]} · {DECODER_LABELS[decoder]}")
            ax.grid(color="#E6E6E6", linewidth=0.55, alpha=0.85)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-0.012, 0.82)
            ax.set_ylim(0.48, 0.84)
            panel_letter(ax, next(letters))
            if row == 1:
                ax.set_xlabel("Reset probability, p")
            if column == 0:
                ax.set_ylabel("Test ROC-AUC")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.52, 1.01))
    fig.subplots_adjust(top=0.88, wspace=0.18, hspace=0.28)
    export(fig, "p_by_T_auc_curves")
    plt.close(fig)


def plot_heatmaps(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.9), sharex=True, sharey=True)
    letters = iter("abcd")
    p_values = np.asarray(PT_P_VALUES)
    t_values = np.asarray(PT_T_VALUES)
    image = None
    for row, dataset in enumerate(("BACE", "BBBP")):
        for column, decoder in enumerate(("linear", "rbf_svr")):
            ax = axes[row, column]
            subset = table[(table.dataset == dataset) & (table.decoder == decoder)]
            matrix = (
                subset.pivot(index="p", columns="t", values="mean_auc")
                .reindex(index=p_values, columns=t_values)
                .to_numpy()
            )
            image = ax.imshow(
                matrix, origin="lower", aspect="auto", interpolation="nearest",
                cmap="viridis", vmin=0.50, vmax=0.82,
            )
            fixed_index = int(np.flatnonzero(np.isclose(p_values, OPEN_P))[0])
            ax.axhline(fixed_index, color="white", linestyle=":", linewidth=1.0)
            ax.set_title(f"{DATASET_LABELS[dataset]} · {DECODER_LABELS[decoder]}")
            ax.set_xticks(np.arange(len(t_values)), t_values)
            ax.set_yticks(np.arange(len(p_values)), [f"{value:.2f}" for value in p_values])
            panel_letter(ax, next(letters))
            if row == 1:
                ax.set_xlabel("Visible events, T")
            if column == 0:
                ax.set_ylabel("Reset probability, p")
    if image is None:
        raise RuntimeError("no heat-map image")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.026, pad=0.025)
    colorbar.set_label("Mean test ROC-AUC (6 seeds)")
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.09, top=0.92, wspace=0.18, hspace=0.23)
    export(fig, "p_by_T_auc_heatmaps")
    plt.close(fig)


def main() -> None:
    configure()
    frame = load_all()
    table = summary(frame)
    plot_curves(table)
    plot_heatmaps(table)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()

