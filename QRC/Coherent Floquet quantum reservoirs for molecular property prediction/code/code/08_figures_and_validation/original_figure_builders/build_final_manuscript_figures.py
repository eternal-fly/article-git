#!/usr/bin/env python3
"""Build the submission-facing figures and compact source-data summaries.

This script consumes only the audited CSV bundle distributed with the paper.
It keeps the application comparisons and the mechanism interventions separate:
graph-stream QRC--ESN comparisons are always made at identical T, whereas the
surface-hopping reset response is expressed relative to the closed QRC.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "source_data" / "main").exists():
    SOURCE = ROOT / "source_data"
    FIG = ROOT / "fig"
elif (ROOT.parent / "source_data" / "paper" / "main").exists():
    SOURCE = ROOT.parent / "source_data" / "paper"
    FIG = ROOT.parent / "figures" / "main"
else:
    raise FileNotFoundError("submission source-data tree not found")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BLUE = "#38598C"
LIGHT_BLUE = "#9EB8D2"
ORANGE = "#D08732"
TEAL = "#4B918A"
MAGENTA = "#9D5672"
RED = "#D85B43"
PURPLE = "#7965A8"
GRAY = "#8B8B8B"
GRID = "#E8E8E8"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
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


def panel_label(ax: plt.Axes, label: str, x: float = -0.17, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def finish(ax: plt.Axes, *, xgrid: bool = False) -> None:
    ax.grid(axis="y", color=GRID, lw=0.55, zorder=0)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.45, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)


def save_all(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / name
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


def build_graph_application() -> None:
    data = pd.read_csv(SOURCE / "main" / "graph_stream_q12_curves.csv")
    styles = {
        "open_dtc_qrc": (MAGENTA, "o", r"Dissipative DTC-QRC, $p=0.10$"),
        "closed_dtc_qrc": (BLUE, "o", r"Closed DTC-QRC, $p=0$"),
        "matched_esn": (TEAL, "^", "Matched 78-node ESN"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 4.45), sharex=True)
    for ax, dataset, label in zip(axes, ["BACE", "BBBP"], "ab"):
        for system, (color, marker, name) in styles.items():
            d = data[(data.dataset == dataset) & (data.system == system)].sort_values("t")
            x = d.t.to_numpy(float)
            y = d.test_auc_mean.to_numpy(float)
            sd = d.test_auc_sd.to_numpy(float)
            ax.plot(x, y, color=color, marker=marker, ms=3.5, lw=1.25, label=name, zorder=3)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12, lw=0, zorder=1)
        ax.axhline(0.5, color=GRAY, ls="--", lw=0.8)
        ax.axvspan(20.01, 30.8, color=ORANGE, alpha=0.055, lw=0)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(f"{dataset}, local chemical-prior tokens")
        ax.set_ylim(0.48, 0.835 if dataset == "BACE" else 0.815)
        ax.set_xlim(1.5, 30.8)
        finish(ax)
        panel_label(ax, label)
    axes[-1].set_xlabel("Visible events, $T$")
    axes[-1].set_xticks([2, 6, 10, 16, 20, 24, 30])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.54, -0.005), handlelength=1.8)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.96, bottom=0.19, hspace=0.28)
    save_all(fig, "fig3_graph_stream_same_t")


def select_hopping_rbf_final() -> pd.DataFrame:
    metrics = pd.read_csv(SOURCE / "main" / "hopping_six_seed_metrics_all_fixed_axes.csv")
    effects_esn = pd.read_csv(
        SOURCE / "main" / "hopping_six_seed_qrc_esn_paired_all_fixed_axes.csv"
    )
    effects_closed = pd.read_csv(
        SOURCE / "supplement" / "hopping_qrc_minus_closed_effects_all_fixed_axes.csv"
    )
    qrc = metrics[
        (metrics.system == "qrc")
        & (metrics.readout == "final")
        & (metrics.decoder == "rbf_svr")
    ]
    ranked = (
        qrc.groupby(["target", "p"], as_index=False)
        .agg(validation_mae_mean_ev=("validation_mae_ev", "mean"))
        .sort_values(["target", "validation_mae_mean_ev", "p"])
    )
    selected = ranked.groupby("target", as_index=False).first()
    rows: list[dict[str, float | str | int]] = []
    for row in selected.itertuples(index=False):
        target = row.target
        p = float(row.p)
        for system, mask in (
            ("dissipative_dtc_qrc", (metrics.system == "qrc") & np.isclose(metrics.p, p)),
            ("closed_dtc_qrc", (metrics.system == "qrc") & np.isclose(metrics.p, 0.0)),
            ("matched_esn", metrics.system == "esn"),
        ):
            d = metrics[
                mask
                & (metrics.target == target)
                & (metrics.readout == "final")
                & (metrics.decoder == "rbf_svr")
            ]
            rows.append(
                {
                    "target": target,
                    "validation_selected_p": p,
                    "system": system,
                    "test_mae_mean_ev": d.mae_ev.mean(),
                    "test_mae_sd_ev": d.mae_ev.std(ddof=1),
                    "n_seeds": d.seed.nunique(),
                    "validation_selection": "minimum six-seed mean validation MAE; ties favor smaller p",
                }
            )
        e = effects_esn[
            (effects_esn.target == target)
            & np.isclose(effects_esn.p, p)
            & (effects_esn.readout == "final")
            & (effects_esn.decoder == "rbf_svr")
        ].iloc[0]
        c = effects_closed[
            (effects_closed.target == target)
            & np.isclose(effects_closed.p, p)
            & (effects_closed.readout == "final")
            & (effects_closed.decoder == "rbf_svr")
        ].iloc[0]
        for baseline, record, key in (
            ("matched_esn", e, "mean_qrc_minus_esn_trajectory_mae_ev"),
            ("closed_dtc_qrc", c, "mean_qrc_minus_closed_trajectory_mae_ev"),
        ):
            rows.append(
                {
                    "target": target,
                    "validation_selected_p": p,
                    "system": f"effect_vs_{baseline}",
                    "test_mae_mean_ev": record[key],
                    "test_mae_sd_ev": np.nan,
                    "n_seeds": int(record.num_seeds),
                    "validation_selection": "paired effect; 95% crossed seed-by-trajectory bootstrap CI in ci columns",
                    "ci95_low_ev": record.crossed_bootstrap_95ci_low_ev,
                    "ci95_high_ev": record.crossed_bootstrap_95ci_high_ev,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE / "main" / "hopping_validation_selected_rbf_final.csv", index=False)
    return out


def build_hopping_application(selected: pd.DataFrame) -> None:
    order = ["gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs"]
    labels = ["Gap\n1 fs", "Gap\n5 fs", "Gap\n10 fs", "Next-5-fs\nminimum"]
    systems = [
        ("dissipative_dtc_qrc", ORANGE, "Dissipative DTC-QRC"),
        ("closed_dtc_qrc", BLUE, "Closed DTC-QRC"),
        ("matched_esn", TEAL, "Matched ESN"),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(3.42, 3.28))
    x = np.arange(len(order))
    width = 0.24
    for j, (system, color, label) in enumerate(systems):
        d = selected[selected.system == system].set_index("target").loc[order]
        ax.bar(
            x + (j - 1) * width,
            d.test_mae_mean_ev,
            width,
            yerr=d.test_mae_sd_ev,
            capsize=1.8,
            color=color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.45,
            label=label,
            zorder=3,
        )
    ax.set_ylabel("Test MAE (eV)")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0.0, 1.90)
    finish(ax)
    panel_label(ax, "a", x=-0.16, y=1.04)
    fig.suptitle("SHNITSEL A01, validation-frozen reset", y=0.985, fontsize=8.2)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.5,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.72, bottom=0.18)
    save_all(fig, "fig4_hopping_validation_selected")


def build_reset_scan_grid() -> None:
    graph = pd.read_csv(SOURCE / "main" / "reset_memory_q12_general_rbf_svr.csv")
    hopping = pd.read_csv(
        SOURCE / "supplement" / "hopping_qrc_minus_closed_effects_all_fixed_axes.csv"
    )
    hopping_p_grid = np.round(np.arange(0.0, 1.0001, 0.05), 10)
    hopping = hopping[
        hopping["p"].round(10).isin(hopping_p_grid)
    ].copy()
    hopping_cell_sizes = hopping.groupby(["readout", "decoder", "target"]).size()
    if len(hopping_cell_sizes) != 16 or not (hopping_cell_sizes == 21).all():
        raise ValueError("surface-hopping reset figure requires every 0.05 cell from p=0 to 1")

    colors = {6: LIGHT_BLUE, 12: "#6E8FB8", 20: BLUE, 30: MAGENTA}
    targets = {
        "gap_1fs": (BLUE, "o", r"$\Delta E$, 1 fs"),
        "gap_5fs": (TEAL, "o", r"$\Delta E$, 5 fs"),
        "gap_10fs": (ORANGE, "s", r"$\Delta E$, 10 fs"),
        "min_gap_next_5fs": (MAGENTA, "o", "Minimum gap, next 5 fs"),
    }
    # Share x only within a column.  The graph experiment retains its original
    # measured 0--0.8 axis, while SHNITSEL uses the uniform 0.05 grid on 0--1.
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.65), sharex="col")

    for row, dataset in enumerate(["BACE", "BBBP"]):
        ax = axes[row, 0]
        for t, color in colors.items():
            d = graph[(graph.dataset == dataset) & (graph.t == t)].sort_values("p")
            x = d.p.to_numpy(float)
            y = d.test_auc_mean.to_numpy(float)
            sd = d.test_auc_sd.to_numpy(float)
            ax.plot(x, y, color=color, marker="o", ms=2.7, lw=1.15, label=f"T={t}")
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10, lw=0)
        ax.axvline(0.10, color=ORANGE, ls="--", lw=1.0)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(f"{dataset} graph, Q=12")
        ax.set_ylim((0.645, 0.83) if dataset == "BACE" else (0.615, 0.805))
        finish(ax)

    hopping_panels = [
        (axes[0, 1], "final", "rbf_svr", "RBF-SVR, final"),
        (axes[0, 2], "tm3", "rbf_svr", "RBF-SVR, TM3"),
        (axes[1, 1], "final", "linear", "Linear, final"),
        (axes[1, 2], "tm3", "linear", "Linear, TM3"),
    ]
    for ax, readout, decoder, title in hopping_panels:
        for target, (color, marker, name) in targets.items():
            d = hopping[
                (hopping.readout.str.lower() == readout)
                & (hopping.decoder.str.lower() == decoder)
                & (hopping.target == target)
            ].sort_values("p")
            x = d.p.to_numpy(float)
            y = d.mean_qrc_minus_closed_trajectory_mae_ev.to_numpy(float)
            lo = d.crossed_bootstrap_95ci_low_ev.to_numpy(float)
            hi = d.crossed_bootstrap_95ci_high_ev.to_numpy(float)
            ax.plot(x, y, color=color, marker=marker, markevery=1, ms=2.2, lw=1.05, label=name, zorder=3)
            ax.fill_between(x, lo, hi, color=color, alpha=0.08, lw=0, zorder=1)
        ax.axhline(0, color=GRAY, ls="--", lw=0.75)
        ax.set_title(title)
        finish(ax)

    axes[0, 1].set_ylabel(r"$\Delta$MAE vs closed QRC (eV)")
    axes[1, 1].set_ylabel(r"$\Delta$MAE vs closed QRC (eV)")
    for ax in axes[:, 0]:
        ax.set_xlim(-0.012, 0.82)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    for ax in axes[:, 1:].flat:
        ax.set_xlim(-0.015, 1.015)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    for ax in axes[1, :]:
        ax.set_xlabel("Reset probability, $p$")
    axes[0, 1].set_ylim(-1.0, 0.20)
    axes[0, 2].set_ylim(-1.0, 0.20)
    axes[1, 1].set_ylim(-0.24, 0.06)
    axes[1, 2].set_ylim(-0.24, 0.06)
    for label, ax in zip("abcdef", axes.flat):
        panel_label(ax, label, x=-0.16, y=1.04)

    graph_handles = [
        Line2D([0], [0], color=colors[t], marker="o", lw=1.15, ms=3, label=f"T={t}")
        for t in [6, 12, 20, 30]
    ]
    graph_handles.append(
        Line2D([0], [0], color=ORANGE, ls="--", lw=1.0, label=r"Fixed $p=0.10$")
    )
    target_handles = [
        Line2D([0], [0], color=c, marker=m, lw=1.1, ms=3, label=n)
        for c, m, n in targets.values()
    ]
    graph_legend = fig.legend(
        handles=graph_handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.995),
        ncol=3,
        title="Graph-event prefixes",
        title_fontsize=6.7,
        columnspacing=0.75,
        handlelength=1.5,
        borderpad=0.35,
    )
    fig.add_artist(graph_legend)
    fig.legend(
        handles=target_handles,
        loc="upper center",
        bbox_to_anchor=(0.705, 0.995),
        ncol=2,
        title="SHNITSEL targets",
        title_fontsize=6.7,
        columnspacing=0.9,
        handlelength=1.5,
        borderpad=0.35,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.79, bottom=0.12, hspace=0.34, wspace=0.34)
    save_all(fig, "fig5_reset_scans_2x3")


def build_cross_event() -> None:
    data = pd.read_csv(SOURCE / "supplement" / "controlled_cross_event_p010" / "summary_by_lag.csv")
    styles = {
        "open_dtc_qrc": (BLUE, "o", "DTC-QRC"),
        "fp_cvr": (RED, "s", "FP-CVR"),
        "matched_esn": (TEAL, "^", "Matched ESN"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 4.25), sharex=True)
    for ax, family, title, label in zip(axes, ["delayed_motif", "distant_pair"], ["Delayed motif", "Distant event pair"], "ab"):
        for method, (color, marker, name) in styles.items():
            d = data[(data.family == family) & (data.method == method)].sort_values("lag")
            x = d.lag.to_numpy(float)
            y = d.test_auc_mean.to_numpy(float)
            sd = d.test_auc_sd.to_numpy(float)
            ax.plot(x, y, color=color, marker=marker, ms=3.2, lw=1.2, label=name, zorder=3)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12, lw=0)
        ax.axhline(0.5, color=GRAY, ls="--", lw=0.8)
        ax.set_ylim(0.43, 1.01)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(title)
        finish(ax)
        panel_label(ax, label)
    axes[-1].set_xlabel("Event separation")
    axes[-1].set_xticks([1, 2, 4, 6, 8, 12, 16])
    axes[0].legend(loc="upper left", ncol=1)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.96, bottom=0.11, hspace=0.26)
    save_all(fig, "fig7_cross_event_readability")


def build_dephasing() -> None:
    bace = pd.read_csv(SOURCE / "supplement" / "coherence_dephasing_fixed_architecture" / "mechanism_bace_auc.csv")
    hopping = pd.read_csv(SOURCE / "supplement" / "coherence_dephasing_fixed_architecture" / "mechanism_hopping_ratio_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    for p, color, marker, label in ((0.0, BLUE, "o", "Closed QRC"), (0.1, ORANGE, "s", "Dissipative QRC")):
        d = bace[np.isclose(bace.p, p)].sort_values("control_value")
        axes[0].plot(d.control_value, d["mean"], color=color, marker=marker, ms=3.7, lw=1.2, label=label)
        axes[0].fill_between(d.control_value, d.ci_low, d.ci_high, color=color, alpha=0.13, lw=0)
    axes[0].set_ylabel("BACE test ROC-AUC")
    axes[0].set_title("Drug-molecule event stream")
    axes[0].set_ylim(0.44, 0.84)
    finish(axes[0])
    panel_label(axes[0], "a", x=-0.13, y=1.02)

    h = hopping[(hopping.readout.str.lower() == "final") & (hopping.decoder.str.lower() == "rbf-svr")]
    for p, color, marker, label in ((0.0, BLUE, "o", "Closed QRC"), (0.1, ORANGE, "s", "Dissipative QRC")):
        d = h[np.isclose(h.p, p)].sort_values("control_value")
        axes[1].plot(d.control_value, d["mean"], color=color, marker=marker, ms=3.7, lw=1.2, label=label)
        axes[1].fill_between(d.control_value, d.ci_low, d.ci_high, color=color, alpha=0.13, lw=0)
    axes[1].axhline(1, color=GRAY, ls="--", lw=0.8)
    axes[1].set_ylabel("Target-balanced MAE ratio")
    axes[1].set_title("Surface-hopping event stream")
    axes[1].set_ylim(0.97, 1.36)
    finish(axes[1])
    panel_label(axes[1], "b", x=-0.13, y=1.02)
    for ax in axes:
        ax.set_xlabel(r"Event dephasing, $\lambda$")
        ax.set_xticks([0, 0.5, 1.0])
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", lw=1.2, ms=3.7, label="Closed QRC"),
        Line2D([0], [0], color=ORANGE, marker="s", lw=1.2, ms=3.7, label="Dissipative QRC"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        columnspacing=1.4,
        handlelength=1.7,
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.91, bottom=0.25, wspace=0.28)
    save_all(fig, "fig8_event_dephasing")


def write_graph_table_source() -> None:
    curves = pd.read_csv(SOURCE / "main" / "graph_stream_q12_curves.csv")
    effects = pd.read_csv(SOURCE / "main" / "graph_stream_q12_long_stream_exact_contrasts.csv")
    rows = []
    for dataset in ["BACE", "BBBP"]:
        for t in [24, 30]:
            d = curves[(curves.dataset == dataset) & (curves.t == t)].set_index("system")
            e = effects[(effects.dataset == dataset) & (effects.t == t) & (effects.comparison == "dissipative_minus_matched_esn")].iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "t": t,
                    "dissipative_p": 0.10,
                    "dissipative_qrc_auc_mean": d.loc["open_dtc_qrc", "test_auc_mean"],
                    "dissipative_qrc_auc_sd": d.loc["open_dtc_qrc", "test_auc_sd"],
                    "closed_qrc_auc_mean": d.loc["closed_dtc_qrc", "test_auc_mean"],
                    "closed_qrc_auc_sd": d.loc["closed_dtc_qrc", "test_auc_sd"],
                    "matched_esn_auc_mean": d.loc["matched_esn", "test_auc_mean"],
                    "matched_esn_auc_sd": d.loc["matched_esn", "test_auc_sd"],
                    "qrc_minus_esn_mean": e.delta_auc_mean,
                    "qrc_minus_esn_ci95_low": e.ci_low,
                    "qrc_minus_esn_ci95_high": e.ci_high,
                }
            )
    pd.DataFrame(rows).to_csv(SOURCE / "main" / "graph_stream_q12_application_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "BACE",
                "model": "simple full-graph GNN reference",
                "test_auc_mean": 0.8199465649,
                "test_auc_sd": 0.0259724218,
                "n_seeds": 10,
                "provenance": "Q12T12_QRC_CRC_GNN_comparison; three-layer GCN; hidden widths 128/64/32; ReLU; dropout 0.2; sum pooling; validation-selected epoch",
            },
            {
                "dataset": "BBBP",
                "model": "simple full-graph GNN reference",
                "test_auc_mean": 0.8299278489,
                "test_auc_sd": 0.0409930857,
                "n_seeds": 10,
                "provenance": "three-layer GIN; hidden widths 128/64/32; ReLU; dropout 0.2; sum pooling; validation-selected epoch; independent scaffold seeds",
            },
        ]
    ).to_csv(SOURCE / "main" / "full_graph_gnn_reference.csv", index=False)


def main() -> None:
    set_style()
    write_graph_table_source()
    selected = select_hopping_rbf_final()
    build_graph_application()
    build_hopping_application(selected)
    build_reset_scan_grid()
    build_cross_event()
    build_dephasing()
    print("Built final manuscript figures and source-data summaries.")


if __name__ == "__main__":
    main()
