"""Export the existing ESN width scan as two panels in one manuscript column."""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qrc_revision_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SOURCE = PAPER / "source_data" / "main"
OUT = PAPER / "fig" / "fig8_esn_width_single_column"


def main():
    summary = pd.read_csv(SOURCE / "wide_esn_summary.csv")
    qrc = pd.read_csv(SOURCE / "wide_esn_qrc_reference.csv")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.5,
        "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.65,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": True,
        "savefig.facecolor": "white",
    })
    width, height = 3.42, 2.23
    fig = plt.figure(figsize=(width, height))
    side, bottom = 1.23, 0.70
    axes = [fig.add_axes([left / width, bottom / height,
                         side / width, side / height])
            for left in [0.43, 2.10]]
    colors = {12: "#6A8EC1", 30: "#B6554D"}
    markers = {12: "o", 30: "s"}
    plotted = []
    for ax, dataset in zip(axes, ["BACE", "BBBP"]):
        for t in [12, 30]:
            rows = summary[(summary.dataset == dataset) & (summary.t == t)].sort_values("width")
            x, y, sd = (rows[c].to_numpy() for c in ["width", "test_auc_mean", "test_auc_sd"])
            assert list(x) == [78, 500, 1000, 2000]
            assert (rows.n_seeds == 6).all()
            curve, = ax.plot(x, y, color=colors[t], marker=markers[t],
                             ms=2.6, lw=1.0, clip_on=False)
            ax.fill_between(x, y - sd, y + sd, color=colors[t], alpha=0.13, linewidth=0)
            ref = qrc[(qrc.dataset == dataset) & (qrc.t == t)].iloc[0]
            ax.axhline(ref.test_auc_mean, color=colors[t], ls="--", lw=0.8)
            np.testing.assert_array_equal(curve.get_ydata(), y)
            np.testing.assert_array_equal(curve.get_xdata(), x)
            for node, mean, spread in zip(x, y, sd):
                plotted.append({"dataset": dataset, "t": t, "width": int(node),
                                "mean": float(mean), "sample_sd": float(spread),
                                "band_low": float(mean - spread),
                                "band_high": float(mean + spread),
                                "qrc_mean": float(ref.test_auc_mean)})
        ax.set(xscale="log", xlim=(65, 2500), ylim=(0.64, 0.84),
               xlabel="ESN nodes", title=dataset)
        ax.set_xticks([78, 500, 1000, 2000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks([0.65, 0.70, 0.75, 0.80])
        ax.grid(axis="y", color="#E7E7E7", lw=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.labelpad = 3
        for label in ax.get_xticklabels():
            label.set_rotation(40)
            label.set_ha("right")
    axes[0].set_ylabel("Test ROC-AUC", labelpad=3)
    for left, letter in zip([0.025, 1.70], ["a", "b"]):
        fig.text(left / width, 2.015 / height, letter,
                 fontsize=8, fontweight="bold")
    handles = [Line2D([0], [0], color=colors[t], marker=markers[t], lw=1,
                      ms=2.6, label=f"ESN, T={t}") for t in [12, 30]]
    handles += [Line2D([0], [0], color=colors[t], ls="--", lw=0.8,
                       label=f"QRC, T={t}") for t in [12, 30]]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.52, 0.008), columnspacing=2.2,
               handlelength=2, labelspacing=0.35, borderaxespad=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT.with_suffix("." + suffix), dpi=600)
    pd.DataFrame(plotted).to_csv(HERE / "plotted_esn_values.csv", index=False)
    (HERE / "figure_dimensions.json").write_text(json.dumps({
        "canvas_inches": [width, height], "axes_inches": [side, side],
        "n_panels": len(axes), "n_esn_means": len(plotted),
        "tick_font_pt": 6.5, "label_font_pt": 7, "panel_font_pt": 8,
    }, indent=2) + "\n")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
