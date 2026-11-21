"""Create the revised architecture and SHNITSEL main-evidence figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from protocol import HOPPING_ROOT, ROOT


FIGURES = ROOT / "figures" / "revised_evidence"
SOURCE_DATA = FIGURES / "source_data"

TARGET_ORDER = ("gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs")
TARGET_LABELS = {
    "gap_1fs": "Gap 1 fs",
    "gap_5fs": "Gap 5 fs",
    "gap_10fs": "Gap 10 fs",
    "min_gap_next_5fs": "Next-5-fs\nminimum",
}
TARGET_COLORS = {
    "gap_1fs": "#24508A",
    "gap_5fs": "#438C8E",
    "gap_10fs": "#88507F",
    "min_gap_next_5fs": "#D37B26",
}
METHOD_COLORS = {
    "qrc": "#D9822B",
    "esn": "#287D78",
    "classical": "#7D7D7D",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.08),
    )
    ax.add_patch(patch)
    ax.text(xy[0] + 0.02, xy[1] + height - 0.045, title, color=color,
            fontsize=7.5, fontweight="bold", va="top")
    ax.text(xy[0] + 0.02, xy[1] + height - 0.145, body, color="#202020",
            fontsize=6.1, va="top", linespacing=1.18)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.85,
            color="#4A4A4A",
        )
    )


def draw_architecture(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(
        ax, (0.012, 0.56), 0.18, 0.36,
        "Graph-derived stream",
        r"BFS atom/bond events" + "\n" + r"$x_1,\ldots,x_T$",
        "#24508A",
    )
    _box(
        ax, (0.012, 0.08), 0.18, 0.36,
        "Physical-time stream",
        r"distances and velocities" + "\n" + r"$x(t-19),\ldots,x(t)$",
        "#438C8E",
    )
    _box(
        ax, (0.235, 0.27), 0.235, 0.46,
        "Direct Hamiltonian setting",
        r"$\bigotimes_i e^{-\mathrm{i}(x_iX_i+y_iY_i+z_iZ_i)/2}$"
        + "\nno micro-time-step serialization",
        "#6B5AA6",
    )
    _box(
        ax, (0.505, 0.27), 0.235, 0.46,
        "DTC reservoir + reset",
        r"$U_F^L$: drive, $ZZ$, disorder" + "\n"
        + r"$\mathrm{R}_p(\rho)=(1-p)\rho+p\rho_0$, $\rho_0=|0^Q\rangle\!\langle0^Q|$",
        "#D9822B",
    )
    _box(
        ax, (0.775, 0.27), 0.205, 0.46,
        "Measured representation",
        r"all $Z_i$ and $Z_iZ_j$" + "\n" + r"final or TM3 $\rightarrow$ RBF-SVR",
        "#287D78",
    )
    _arrow(ax, (0.192, 0.735), (0.235, 0.58))
    _arrow(ax, (0.192, 0.255), (0.235, 0.42))
    _arrow(ax, (0.47, 0.50), (0.505, 0.50))
    _arrow(ax, (0.74, 0.50), (0.775, 0.50))
    ax.text(
        0.50,
        0.015,
        "The interface changes; the reservoir, observable family, and decoder protocol remain fixed.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    panel_label(ax, "a", x=-0.015, y=0.91)


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    summary = pd.read_csv(HOPPING_ROOT / "outputs" / "best_reservoir_summary.csv")
    classical = pd.read_csv(HOPPING_ROOT / "outputs" / "classical_test.csv")
    input_matched = classical[
        ~classical.input_scope.str.contains("target-informed", na=False)
    ].copy()
    input_matched = input_matched.loc[
        input_matched.groupby("target").validation_mae_ev.idxmin()
    ].copy()
    target_informed = classical[
        classical.input_scope.str.contains("target-informed", na=False)
    ].copy()
    paired = pd.read_csv(
        HOPPING_ROOT / "outputs" / "paired_trajectory_comparisons.csv"
    )
    paired = paired[paired.comparison == "qrc_minus_esn"].copy()
    p_scan = pd.read_csv(
        HOPPING_ROOT / "figures" / "source_data" / "qrc_svr_p_scan_envelope.csv"
    )
    baseline = (
        p_scan[np.isclose(p_scan.p, 0.0)][["seed", "target", "validation_mae_ev"]]
        .rename(columns={"validation_mae_ev": "p0_validation_mae_ev"})
    )
    p_scan = p_scan.merge(baseline, on=["seed", "target"], validate="many_to_one")
    p_scan["validation_mae_ratio_to_p0"] = (
        p_scan.validation_mae_ev / p_scan.p0_validation_mae_ev
    )
    selected = pd.read_csv(
        HOPPING_ROOT / "outputs" / "reservoir_best_validation_selected.csv"
    )
    selected = selected[selected.system == "qrc"].copy()
    return summary, input_matched, target_informed, paired, p_scan, selected


def plot() -> None:
    apply_style()
    summary, classical, target_informed, paired, p_scan, selected = load_data()
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SOURCE_DATA / "hopping_main_reservoir_summary.csv", index=False)
    classical.to_csv(SOURCE_DATA / "hopping_input_matched_classical.csv", index=False)
    target_informed.to_csv(SOURCE_DATA / "hopping_target_informed_controls.csv", index=False)
    paired.to_csv(SOURCE_DATA / "hopping_qrc_esn_paired_effects.csv", index=False)
    p_scan.to_csv(SOURCE_DATA / "hopping_validation_p_scan.csv", index=False)
    selected.to_csv(SOURCE_DATA / "hopping_validation_selected_qrc.csv", index=False)

    fig = plt.figure(figsize=(7.0, 4.85))
    grid = fig.add_gridspec(
        2, 3, height_ratios=(0.86, 1.65), left=0.075, right=0.995,
        top=0.97, bottom=0.14, hspace=0.40, wspace=0.45,
    )
    architecture = fig.add_subplot(grid[0, :])
    draw_architecture(architecture)
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]

    # b: held-out performance under validation-selected p/readout.
    ax = axes[0]
    x = np.arange(len(TARGET_ORDER))
    width = 0.24
    for offset, (system, label) in zip(
        (-width, 0.0, width),
        (("qrc", "DTC-QRC"), ("esn", "Matched ESN"), ("classical", "Best input-matched\nclassical")),
    ):
        means: list[float] = []
        errors: list[float] = []
        for target in TARGET_ORDER:
            if system == "classical":
                row = classical[classical.target == target].iloc[0]
                means.append(float(row.mae_ev))
                errors.append(np.nan)
            else:
                row = summary[(summary.system == system) & (summary.target == target)].iloc[0]
                means.append(float(row.mae_ev_mean))
                errors.append(float(row.mae_ev_sample_std))
        ax.bar(
            x + offset, means, width, color=METHOD_COLORS[system], label=label,
            edgecolor="white", linewidth=0.4,
        )
        if system != "classical":
            ax.errorbar(
                x + offset, means, yerr=errors, fmt="none", ecolor="#333333",
                elinewidth=0.7, capsize=1.8,
            )
    ax.set_xticks(x, [TARGET_LABELS[target] for target in TARGET_ORDER])
    ax.tick_params(axis="x", labelrotation=24)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title(r"SHNITSEL A01, $Q=15$, $T=20$")
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    panel_label(ax, "b")

    # c: trajectory-paired effect at the identical T=20 interface.
    ax = axes[1]
    paired_ordered = paired.set_index("target").loc[list(TARGET_ORDER)]
    y = np.arange(len(TARGET_ORDER))[::-1]
    ax.hlines(
        y,
        paired_ordered.bootstrap_95ci_low_ev,
        paired_ordered.bootstrap_95ci_high_ev,
        color=METHOD_COLORS["qrc"],
        linewidth=1.35,
    )
    ax.plot(
        paired_ordered.mean_mae_difference_ev,
        y,
        marker="s",
        color=METHOD_COLORS["qrc"],
        linestyle="none",
        markersize=4.0,
    )
    ax.axvline(0.0, color="#8C8C8C", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [TARGET_LABELS[target] for target in TARGET_ORDER])
    ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlabel(r"DTC-QRC $-$ ESN trajectory MAE (eV)")
    ax.set_title("Paired effects at the same T=20")
    ax.grid(axis="x", color="#E7E7E7", lw=0.55)
    panel_label(ax, "c")

    # d: validation-only reset response, normalized within target and seed.
    ax = axes[2]
    p_source_rows: list[pd.DataFrame] = []
    for target in TARGET_ORDER:
        sub = (
            p_scan[p_scan.target == target]
            .groupby("p", as_index=False)
            .validation_mae_ratio_to_p0.agg(["mean", "std"])
            .reset_index()
        )
        p_source_rows.append(sub.assign(target=target))
        color = TARGET_COLORS[target]
        ax.plot(sub.p, sub["mean"], color=color, marker="o", ms=2.8, lw=1.1,
                label=TARGET_LABELS[target].replace("\n", " "))
        ax.fill_between(
            sub.p.to_numpy(float),
            (sub["mean"] - sub["std"]).to_numpy(float),
            (sub["mean"] + sub["std"]).to_numpy(float),
            color=color,
            alpha=0.11,
            linewidth=0,
        )
        chosen = selected[selected.target == target]
        chosen_p = float(chosen.selected_p.mean())
        chosen_rows = p_scan[
            (p_scan.target == target)
            & p_scan.seed.isin(chosen.seed)
        ].merge(
            chosen[["seed", "selected_p"]], on="seed", validate="many_to_one"
        )
        chosen_rows = chosen_rows[np.isclose(chosen_rows.p, chosen_rows.selected_p)]
        ax.plot(
            chosen_p,
            float(chosen_rows.validation_mae_ratio_to_p0.mean()),
            marker="*",
            ms=7.0,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=5,
        )
    pd.concat(p_source_rows, ignore_index=True).to_csv(
        SOURCE_DATA / "hopping_validation_p_scan_summary.csv", index=False
    )
    ax.axhline(1.0, color="#8C8C8C", linestyle="--", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=0.005, linscale=0.7)
    ax.set_xlim(-0.0015, 1.5)
    ax.set_xticks([0.0, 0.01, 0.04, 0.16, 0.64, 1.0])
    ax.set_xticklabels(["0", ".01", ".04", ".16", ".64", "1"])
    ax.tick_params(axis="x", labelrotation=24)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_xlabel("Reset probability, p")
    ax.set_ylabel("Validation MAE / p=0 MAE")
    ax.set_title("Validation-only reset response")
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    ax.legend(loc="lower left", ncol=2, fontsize=4.8, handlelength=1.25,
              columnspacing=0.7, labelspacing=0.25)
    panel_label(ax, "d")

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["qrc"], lw=5, label="DTC-QRC"),
        Line2D([0], [0], color=METHOD_COLORS["esn"], lw=5, label="Matched ESN"),
        Line2D([0], [0], color=METHOD_COLORS["classical"], lw=5,
               label="Best input-matched classical"),
        Line2D([0], [0], color="#555555", marker="*", ls="none", ms=6,
               label="Validation-selected p"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "fig2_architecture_surface_hopping"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    plot()
