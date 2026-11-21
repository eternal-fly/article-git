"""Plot the Q12 general-token reset-by-event-length diagnostic.

Every curve and heat-map row retains a fixed event length.  The script never
averages AUC or method effects across T.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from protocol import MAIN_ROOT, ROOT


SOURCE_FILE = MAIN_ROOT / "results" / "p_by_T_diagnostic" / "summary_p_by_T.csv"
OUT = ROOT / "figures" / "revised_evidence"
SOURCE = OUT / "source_data"
T_VALUES = (6, 12, 20, 30)
T_COLORS = {6: "#A8C5DA", 12: "#628BB8", 20: "#394B77", 30: "#A54F73"}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")


def save(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig3_reset_memory_by_exact_t"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(SOURCE_FILE)
    data = data[(data.decoder == "rbf_svr") & data.t.isin(T_VALUES)].copy()
    SOURCE.mkdir(parents=True, exist_ok=True)
    data.to_csv(SOURCE / "reset_memory_q12_general_rbf_svr.csv", index=False)
    p_values = sorted(data.p.unique())
    x = np.arange(len(p_values))
    p_to_x = {round(float(p), 8): index for index, p in enumerate(p_values)}
    label_values = {0.0, 0.02, 0.06, 0.1, 0.2, 0.4, 0.8}
    tick_labels = [f"{p:g}" if round(float(p), 8) in label_values else "" for p in p_values]
    maximum = float(np.nanmax(np.abs(data.delta_auc_mean)))

    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex="col")
    images = []
    for col, dataset in enumerate(("BACE", "BBBP")):
        line_ax = axes[0, col]
        frame = data[data.dataset == dataset]
        for t in T_VALUES:
            sub = frame[frame.t == t].sort_values("p")
            px = np.asarray([p_to_x[round(float(p), 8)] for p in sub.p])
            y = sub.test_auc_mean.to_numpy(dtype=float)
            sd = sub.test_auc_sd.to_numpy(dtype=float)
            line_ax.plot(px, y, color=T_COLORS[t], marker="o", ms=2.8,
                         lw=1.25, label=f"T={t}")
            line_ax.fill_between(px, y - sd, y + sd, color=T_COLORS[t],
                                 alpha=0.10, linewidth=0)
        line_ax.axvline(p_to_x[0.1], color="#D9822B", lw=0.9, ls="--")
        line_ax.set_title(f"{dataset}, Q=12, general local tokens")
        line_ax.set_ylabel("Test ROC-AUC")
        line_ax.grid(axis="y", color="#E7E7E7", lw=0.55)

        heat_ax = axes[1, col]
        matrix = np.empty((len(T_VALUES), len(p_values)), dtype=float)
        for row, t in enumerate(T_VALUES):
            sub = frame[frame.t == t].set_index("p")
            matrix[row] = [sub.loc[p, "delta_auc_mean"] for p in p_values]
        image = heat_ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                               vmin=-maximum, vmax=maximum, interpolation="nearest")
        images.append(image)
        heat_ax.axvline(p_to_x[0.1], color="#222222", lw=0.8, ls="--")
        heat_ax.set_yticks(np.arange(len(T_VALUES)))
        heat_ax.set_yticklabels([str(t) for t in T_VALUES])
        heat_ax.set_ylabel("Visible events, T")
        heat_ax.set_xlabel("Reset probability, p")
        heat_ax.set_xticks(x)
        heat_ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    for index, ax in enumerate(axes.flat):
        panel_label(ax, chr(ord("a") + index))
    handles = [Line2D([0], [0], color=T_COLORS[t], marker="o", lw=1.25,
                      ms=2.8, label=f"T={t}") for t in T_VALUES]
    handles.append(Line2D([0], [0], color="#D9822B", ls="--", lw=0.9,
                          label="Fixed p=0.10"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.43, -0.005))
    colorbar = fig.colorbar(images[-1], ax=axes[1, :], fraction=0.028, pad=0.025)
    colorbar.set_label(r"Mean $\Delta$AUC vs p=0" + "\n(same T)", fontsize=6.8,
                       labelpad=4)
    fig.subplots_adjust(left=0.10, right=0.91, top=0.92, bottom=0.18,
                        wspace=0.27, hspace=0.34)
    save(fig)
    print(OUT / "fig3_reset_memory_by_exact_t.pdf")


if __name__ == "__main__":
    main()
