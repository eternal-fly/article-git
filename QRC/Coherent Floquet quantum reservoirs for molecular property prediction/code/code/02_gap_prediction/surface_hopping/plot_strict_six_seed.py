#!/usr/bin/env python3
"""Publication figures for the six-seed fixed-axis hopping benchmark."""

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
HIGH_P_UNIFORM_GRID = np.linspace(0.5, 1.0, 11)
TARGET_LABELS = {
    "gap_1fs": r"$\Delta E$, 1 fs",
    "gap_5fs": r"$\Delta E$, 5 fs",
    "gap_10fs": r"$\Delta E$, 10 fs",
    "min_gap_next_5fs": "Minimum gap,\nnext 5 fs",
}
TARGET_COLORS = {
    "gap_1fs": "#345995",
    "gap_5fs": "#2A9D8F",
    "gap_10fs": "#D8872E",
    "min_gap_next_5fs": "#9B5A8C",
}
SYSTEM_COLORS = {"qrc": "#244A86", "esn": "#D3744F"}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=9,
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


def set_rowwise_effect_limits(
    axes: np.ndarray,
    paired: pd.DataFrame,
    effect_column: str,
) -> None:
    """Use one honest scale per decoder row, shared by final and TM3."""
    for row, decoder in enumerate(("rbf_svr", "linear")):
        frame = paired[paired.decoder == decoder]
        lower = min(
            0.0,
            float(frame[effect_column].min()),
            float(frame.crossed_bootstrap_95ci_low_ev.min()),
        )
        upper = max(
            0.0,
            float(frame[effect_column].max()),
            float(frame.crossed_bootstrap_95ci_high_ev.max()),
        )
        span = max(upper - lower, 0.05)
        axes[row, 0].set_ylim(lower - 0.055 * span, upper + 0.055 * span)


def effect_figure(
    paired: pd.DataFrame,
    *,
    effect_column: str = "mean_qrc_minus_esn_trajectory_mae_ev",
    ylabel: str = "QRC − ESN trajectory MAE (eV)",
    title: str = "Six-seed SHNITSEL comparison at the identical T=20 interface",
    stem: str = "fig_hopping_qrc_esn_effect_by_fixed_axis",
    rowwise_y_scale: bool = False,
    linear_p_axis: bool = False,
) -> None:
    panels = (
        ("rbf_svr", "final", "RBF-SVR, final"),
        ("rbf_svr", "tm3", "RBF-SVR, TM3"),
        ("linear", "final", "Linear, final"),
        ("linear", "tm3", "Linear, TM3"),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(7.05, 4.65), sharex=True,
        sharey="row" if rowwise_y_scale else True,
    )
    for index, (ax, (decoder, readout, panel_title)) in enumerate(
        zip(axes.flat, panels)
    ):
        frame = paired[(paired.decoder == decoder) & (paired.readout == readout)]
        for target in TARGETS:
            sub = frame[frame.target == target].sort_values("p")
            physical_p = sub.p.to_numpy(float)
            y = sub[effect_column].to_numpy(float)
            low = sub.crossed_bootstrap_95ci_low_ev.to_numpy(float)
            high = sub.crossed_bootstrap_95ci_high_ev.to_numpy(float)
            ax.plot(physical_p, y, color=TARGET_COLORS[target], marker="o", ms=2.7,
                    lw=1.1, label=TARGET_LABELS[target].replace("\n", " "))
            ax.fill_between(
                physical_p, low, high, color=TARGET_COLORS[target], alpha=0.10,
                linewidth=0,
            )
        ax.axhline(0.0, color="#858585", lw=0.75, ls="--")
        if linear_p_axis:
            ticks = np.linspace(0.0, 1.0, 11)
            ax.set_xscale("linear")
            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                ["0" if np.isclose(tick, 0.0) else
                 "1" if np.isclose(tick, 1.0) else f"{tick:.1f}"
                 for tick in ticks]
            )
        else:
            ax.set_xscale("symlog", linthresh=0.005, linscale=0.75)
            ax.set_xlim(-0.0015, 1.35)
            ax.set_xticks([0, 0.01, 0.04, 0.1, 0.32, 1.0])
            ax.set_xticklabels(["0", ".01", ".04", ".10", ".32", "1"])
        ax.grid(axis="y", color="#E8E8E8", lw=0.55)
        ax.set_title(panel_title, loc="left")
        panel_label(ax, chr(ord("a") + index))
    if rowwise_y_scale:
        set_rowwise_effect_limits(axes, paired, effect_column)
    fig.supylabel(ylabel, x=0.012, fontsize=7.5)
    xlabel = "Reset probability, p"
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 1].set_xlabel(xlabel)
    handles = [
        Line2D([0], [0], color=TARGET_COLORS[target], marker="o", lw=1.1,
               ms=3, label=TARGET_LABELS[target].replace("\n", " "))
        for target in TARGETS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, x=0.08, y=0.995, ha="left", fontsize=9,
                 fontweight="bold")
    figure_notes = []
    if linear_p_axis:
        figure_notes.append("physical linear p axis (0–1)")
    if rowwise_y_scale:
        figure_notes.append("independent y-scale by decoder row")
    if figure_notes:
        fig.text(
            0.985, 0.965, "; ".join(figure_notes).capitalize(),
            ha="right", va="top", fontsize=5.5, color="#666666",
        )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.91,
                        bottom=0.17,
                        wspace=0.20, hspace=0.33)
    save_figure(fig, stem)


def high_p_zoom_figure(
    paired: pd.DataFrame,
    *,
    effect_column: str = "mean_qrc_minus_esn_trajectory_mae_ev",
    ylabel: str = "QRC − ESN trajectory MAE (eV)",
    title: str = "Dense high-reset scan at the identical T=20 interface",
    stem: str = "fig_hopping_qrc_esn_effect_high_p_uniform_zoom",
    rowwise_y_scale: bool = False,
) -> pd.DataFrame:
    """Resolve the possible crossover on the uniform p=0.50,...,1.00 grid."""
    panels = (
        ("rbf_svr", "final", "RBF-SVR, final"),
        ("rbf_svr", "tm3", "RBF-SVR, TM3"),
        ("linear", "final", "Linear, final"),
        ("linear", "tm3", "Linear, TM3"),
    )
    keep = np.logical_or.reduce([
        np.isclose(paired.p.to_numpy(float), p) for p in HIGH_P_UNIFORM_GRID
    ])
    high_p = paired.loc[keep].copy()
    observed = np.sort(high_p.p.unique().astype(float))
    if len(observed) != len(HIGH_P_UNIFORM_GRID) or not np.allclose(
        observed, HIGH_P_UNIFORM_GRID
    ):
        raise RuntimeError(f"incomplete uniform high-p grid: {observed}")

    fig, axes = plt.subplots(
        2, 2, figsize=(7.05, 4.65), sharex=True,
        sharey="row" if rowwise_y_scale else True,
    )
    for index, (ax, (decoder, readout, panel_title)) in enumerate(
        zip(axes.flat, panels)
    ):
        frame = high_p[
            (high_p.decoder == decoder) & (high_p.readout == readout)
        ]
        for target in TARGETS:
            sub = frame[frame.target == target].sort_values("p")
            x = sub.p.to_numpy(float)
            y = sub[effect_column].to_numpy(float)
            low = sub.crossed_bootstrap_95ci_low_ev.to_numpy(float)
            high = sub.crossed_bootstrap_95ci_high_ev.to_numpy(float)
            ax.plot(
                x, y, color=TARGET_COLORS[target], marker="o", ms=2.9,
                lw=1.15, label=TARGET_LABELS[target].replace("\n", " "),
            )
            ax.fill_between(
                x, low, high, color=TARGET_COLORS[target], alpha=0.10,
                linewidth=0,
            )
        ax.axhline(0.0, color="#858585", lw=0.75, ls="--")
        ax.set_xlim(0.49, 1.01)
        ax.set_xticks(np.arange(0.5, 1.01, 0.1))
        ax.set_xticks(HIGH_P_UNIFORM_GRID, minor=True)
        ax.grid(axis="y", color="#E8E8E8", lw=0.55)
        ax.set_title(panel_title, loc="left")
        panel_label(ax, chr(ord("a") + index))
    if rowwise_y_scale:
        set_rowwise_effect_limits(axes, high_p, effect_column)
    fig.supylabel(ylabel, x=0.012, fontsize=7.5)
    axes[1, 0].set_xlabel("Reset probability, p")
    axes[1, 1].set_xlabel("Reset probability, p")
    handles = [
        Line2D(
            [0], [0], color=TARGET_COLORS[target], marker="o", lw=1.1,
            ms=3, label=TARGET_LABELS[target].replace("\n", " "),
        )
        for target in TARGETS
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(title, x=0.08, y=0.995, ha="left", fontsize=9,
                 fontweight="bold")
    if rowwise_y_scale:
        fig.text(
            0.985, 0.965, "Independent y-scale by decoder row",
            ha="right", va="top", fontsize=5.5, color="#666666",
        )
    fig.subplots_adjust(
        left=0.10, right=0.99, top=0.91, bottom=0.17,
        wspace=0.20, hspace=0.33,
    )
    save_figure(fig, stem)
    return high_p


def optimum_figure(optima: pd.DataFrame) -> None:
    panels = (
        ("rbf_svr", "final", "RBF-SVR, final"),
        ("rbf_svr", "tm3", "RBF-SVR, TM3"),
        ("linear", "final", "Linear, final"),
        ("linear", "tm3", "Linear, TM3"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.75), sharex=True)
    y = np.arange(len(TARGETS), dtype=float)[::-1]
    for index, (ax, (decoder, readout, title)) in enumerate(zip(axes.flat, panels)):
        frame = (
            optima[(optima.decoder == decoder) & (optima.readout == readout)]
            .set_index("target").loc[list(TARGETS)]
        )
        comparisons = (
            (
                0.11, frame.mean_qrc_minus_esn_trajectory_mae_ev.to_numpy(float),
                frame.pointwise_95ci_low_ev.to_numpy(float),
                frame.pointwise_95ci_high_ev.to_numpy(float),
                frame.simultaneous_95band_low_ev.to_numpy(float),
                frame.simultaneous_95band_high_ev.to_numpy(float),
                SYSTEM_COLORS["qrc"], "s",
            ),
            (
                -0.11, frame.mean_p_star_minus_closed_trajectory_mae_ev.to_numpy(float),
                frame.closed_pointwise_95ci_low_ev.to_numpy(float),
                frame.closed_pointwise_95ci_high_ev.to_numpy(float),
                frame.closed_simultaneous_95band_low_ev.to_numpy(float),
                frame.closed_simultaneous_95band_high_ev.to_numpy(float),
                "#D8872E", "o",
            ),
        )
        all_limits = []
        for offset, center, point_low, point_high, simultaneous_low, simultaneous_high, color, marker in comparisons:
            positions = y + offset
            ax.hlines(positions, simultaneous_low, simultaneous_high, color=color,
                      lw=5.0, alpha=0.20, zorder=1)
            ax.hlines(positions, point_low, point_high, color=color,
                      lw=1.35, zorder=2)
            ax.plot(center, positions, marker=marker, linestyle="none", ms=3.8,
                    color=color, zorder=3)
            all_limits.extend((simultaneous_low, simultaneous_high))
        esn_center = comparisons[0][1]
        span = max(float(np.ptp(np.concatenate(all_limits))), 0.05)
        for y_value, x_value, p_star in zip(
            y, esn_center, frame.p_star_test_grid_argmin.to_numpy(float)
        ):
            ax.text(x_value + 0.018 * span, y_value + 0.22,
                    rf"$p^*={p_star:g}$", fontsize=5.2,
                    color=SYSTEM_COLORS["qrc"], ha="left", va="bottom")
        ax.axvline(0.0, color="#858585", lw=0.75, ls="--")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="#E8E8E8", lw=0.55)
        ax.set_yticks(y, [TARGET_LABELS[target] for target in TARGETS])
        panel_label(ax, chr(ord("a") + index))
    axes[1, 0].set_xlabel(r"QRC($p^*$) − baseline trajectory MAE (eV)")
    axes[1, 1].set_xlabel(r"QRC($p^*$) − baseline trajectory MAE (eV)")
    handles = [
        Line2D([0], [0], color=SYSTEM_COLORS["qrc"], marker="s", lw=1.35,
               ms=3.8, label=r"$p^*$ QRC − standard ESN"),
        Line2D([0], [0], color="#D8872E", marker="o", lw=1.35,
               ms=3.8, label=r"$p^*$ QRC − closed QRC"),
        Line2D([0], [0], color="#777777", lw=5.0, alpha=0.20,
               label="Pale width: scan-wide 95% band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Descriptive target-wise p* after the complete reset scan at fixed T=20",
                 x=0.08, y=0.995, ha="left", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.18, right=0.99, top=0.91, bottom=0.17,
                        wspace=0.22, hspace=0.42)
    save_figure(fig, "fig_hopping_scan_optimum_by_fixed_axis")


def main() -> None:
    configure_style()
    metrics = pd.read_csv(DATA_ROOT / "reservoir_metrics_all_fixed_axes.csv")
    summary = pd.read_csv(DATA_ROOT / "reservoir_summary_all_fixed_axes.csv")
    paired = pd.read_csv(DATA_ROOT / "qrc_minus_esn_paired_all_fixed_axes.csv")
    closed_paired = pd.read_csv(
        DATA_ROOT / "qrc_minus_closed_paired_all_fixed_axes.csv"
    )
    optima = pd.read_csv(DATA_ROOT / "qrc_p_scan_optima_by_fixed_decoder_readout.csv")
    if set(metrics.seed) != {0, 10, 20, 30, 40, 50}:
        raise RuntimeError("six-seed contract is incomplete")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(SOURCE_ROOT / "hopping_seed_metrics_all_fixed_axes.csv", index=False)
    summary.to_csv(SOURCE_ROOT / "hopping_summary_all_fixed_axes.csv", index=False)
    paired.to_csv(SOURCE_ROOT / "hopping_paired_effects_all_fixed_axes.csv", index=False)
    closed_paired.to_csv(
        SOURCE_ROOT / "hopping_qrc_minus_closed_effects_all_fixed_axes.csv",
        index=False,
    )
    optima.to_csv(SOURCE_ROOT / "hopping_p_scan_optima_by_fixed_axis.csv", index=False)
    effect_figure(paired, linear_p_axis=True)
    high_p = high_p_zoom_figure(paired)
    high_p.to_csv(
        SOURCE_ROOT / "hopping_paired_effects_high_p_uniform_grid.csv", index=False
    )
    closed_ylabel = "QRC(p) − closed QRC trajectory MAE (eV)"
    effect_figure(
        closed_paired,
        effect_column="mean_qrc_minus_closed_trajectory_mae_ev",
        ylabel=closed_ylabel,
        title="Complete reset scan relative to closed QRC at fixed T=20",
        stem="fig_hopping_qrc_closed_effect_by_fixed_axis",
        rowwise_y_scale=True,
        linear_p_axis=True,
    )
    closed_high_p = high_p_zoom_figure(
        closed_paired,
        effect_column="mean_qrc_minus_closed_trajectory_mae_ev",
        ylabel=closed_ylabel,
        title="Dense high-reset scan relative to closed QRC at fixed T=20",
        stem="fig_hopping_qrc_closed_effect_high_p_uniform_zoom",
        rowwise_y_scale=True,
    )
    closed_high_p.to_csv(
        SOURCE_ROOT / "hopping_qrc_minus_closed_effects_high_p_uniform_grid.csv",
        index=False,
    )
    optimum_figure(optima)
    manifest = {
        "backend": "Python/matplotlib",
        "protocol": "six paired seeds; T=20 fixed",
        "physical_axis_selection": (
            "complete p curves retained; p* is a descriptive test-grid argmin, "
            "never validation or deployment selection"
        ),
        "closed_comparison_scale": (
            "complete 27-point tiny-p plus uniform-0.05 grid; independent y limits for RBF-SVR and "
            "Linear rows, with final/TM3 sharing one scale inside each row"
        ),
        "closed_comparison_x_layout": (
            "physical p values are plotted without remapping on a linear 0-to-1 "
            "axis with uniformly spaced 0.1 ticks"
        ),
        "figures": [
            "fig_hopping_qrc_esn_effect_by_fixed_axis",
            "fig_hopping_qrc_esn_effect_high_p_uniform_zoom",
            "fig_hopping_qrc_closed_effect_by_fixed_axis",
            "fig_hopping_qrc_closed_effect_high_p_uniform_zoom",
            "fig_hopping_scan_optimum_by_fixed_axis",
        ],
        "formats": ["svg", "pdf", "png", "tiff"],
        "uncertainty": (
            "20,000-draw crossed seed-by-trajectory pointwise intervals; "
            f"max-deviation simultaneous band across all {metrics[metrics.system == 'qrc'].p.nunique()} p values for p* effects"
        ),
    }
    (FIGURE_ROOT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(FIGURE_ROOT)


if __name__ == "__main__":
    main()
