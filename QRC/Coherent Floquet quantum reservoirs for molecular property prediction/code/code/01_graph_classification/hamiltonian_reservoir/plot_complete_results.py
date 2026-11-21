"""Generate the complete BACE/BBBP/surface-hopping result figure bundle."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from protocol import OPEN_P, T_VALUES, W, G, WORKSPACE


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "complete_results_bundle"
SOURCE = OUT / "source_data"
HOP_OPEN = WORKSPACE / "Transition" / "shnitsel_a01_surface_hopping"
HOP_CLOSED = WORKSPACE / "Transition" / "shnitsel_a01_surface_hopping_closed"

COLORS = {
    "closed_dtc_qrc": "#0F4D92",
    "open_dtc_qrc": "#E28E2C",
    "matched_esn": "#42949E",
    "qrc": "#0F4D92",
    "esn": "#42949E",
    "classical": "#9B9B9B",
    "second_esn": "#7C6CCF",
    "fp_cvr": "#D9544D",
}
LABELS = {
    "closed_dtc_qrc": "Closed DTC-QRC",
    "open_dtc_qrc": "Dissipative DTC-QRC (p=0.10)",
    "matched_esn": "Matched-width ESN",
}
MARKERS = {"closed_dtc_qrc": "o", "open_dtc_qrc": "s", "matched_esn": "^"}
TARGET_ORDER = ["gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs"]
TARGET_LABELS = {
    "gap_1fs": "Gap, 1 fs",
    "gap_5fs": "Gap, 5 fs",
    "gap_10fs": "Gap, 10 fs",
    "min_gap_next_5fs": "Min. gap\n(next 5 fs)",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6.2,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#B8B8B8",
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
        }
    )


def panel_label(ax, label: str, x: float = -0.14, y: float = 1.06, va: str = "bottom") -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va=va)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        OUT / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def plot_main_curves(summary: pd.DataFrame, variant: str, stem: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.5), sharex=True, sharey=True)
    columns = [(12, "linear"), (16, "linear"), (12, "rbf_svr"), (16, "rbf_svr")]
    for row, dataset in enumerate(["BACE", "BBBP"]):
        for col, (q, decoder) in enumerate(columns):
            ax = axes[row, col]
            frame = summary[
                (summary.dataset == dataset)
                & (summary.token_variant == variant)
                & (summary.q == q)
                & (summary.decoder == decoder)
            ]
            for system in ["closed_dtc_qrc", "open_dtc_qrc", "matched_esn"]:
                sub = frame[frame.system == system].sort_values("t")
                x = sub.t.to_numpy()
                y = sub.test_auc_mean.to_numpy()
                sd = sub.test_auc_sd.to_numpy()
                ax.plot(
                    x,
                    y,
                    color=COLORS[system],
                    marker=MARKERS[system],
                    ms=2.8,
                    lw=1.25,
                    label=LABELS[system],
                )
                ax.fill_between(x, y - sd, y + sd, color=COLORS[system], alpha=0.12, linewidth=0)
            ax.axhline(0.5, color="#B7B7B7", ls="--", lw=0.7, zorder=0)
            ax.set_xlim(1.5, 30.5)
            ax.set_ylim(0.46, 0.92)
            ax.set_xticks(T_VALUES)
            ax.set_xticklabels([str(t) for t in T_VALUES], rotation=45, ha="right")
            ax.grid(axis="y", color="#E8E8E8", lw=0.55)
            if row == 0:
                decoder_label = "Linear" if decoder == "linear" else "RBF-SVR"
                ax.set_title(f"Q{q} · {decoder_label}")
            if col == 0:
                ax.set_ylabel(f"{dataset}\nTest ROC-AUC")
            if row == 1:
                ax.set_xlabel("Visible events, T")
            panel_label(ax, chr(ord("a") + row * 4 + col))
    handles = [
        Line2D([0], [0], color=COLORS[s], marker=MARKERS[s], lw=1.4, ms=3.2, label=LABELS[s])
        for s in ["closed_dtc_qrc", "open_dtc_qrc", "matched_esn"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.92, bottom=0.17, wspace=0.18, hspace=0.24)
    save_figure(fig, stem)


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draw].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plot_fixed_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("closed_dtc_qrc", "matched_esn", "Closed − ESN", COLORS["closed_dtc_qrc"], "o"),
        ("open_dtc_qrc", "matched_esn", "Open − ESN", COLORS["open_dtc_qrc"], "s"),
        ("open_dtc_qrc", "closed_dtc_qrc", "Open − closed", "#666666", "D"),
    ]
    rows = []
    panels = [("BACE", "canonical_general"), ("BBBP", "canonical_general"),
              ("BACE", "canonical_global_prior"), ("BBBP", "canonical_global_prior")]
    categories = [(12, 12, "linear"), (16, 16, "linear"), (12, 12, "rbf_svr"), (16, 16, "rbf_svr")]
    for panel_index, (dataset, variant) in enumerate(panels):
        for cat_index, (q, t, decoder) in enumerate(categories):
            frame = metrics[
                (metrics.dataset == dataset) & (metrics.token_variant == variant)
                & (metrics.q == q) & (metrics.t == t) & (metrics.decoder == decoder)
            ].pivot(index="seed", columns="system", values="test_auc").sort_index()
            for effect_index, (left, right, label, color, marker) in enumerate(specs):
                estimate, low, high = bootstrap_mean_ci(
                    (frame[left] - frame[right]).to_numpy(),
                    20260829 + panel_index * 100 + cat_index * 10 + effect_index,
                )
                rows.append(
                    {
                        "dataset": dataset, "token_variant": variant, "q": q, "t": t,
                        "decoder": decoder, "effect": label, "estimate": estimate,
                        "ci_low": low, "ci_high": high, "n_seeds": len(frame),
                        "color": color, "marker": marker,
                    }
                )
    effects = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey=True)
    cat_labels = ["Q12/T12\nLinear", "Q16/T16\nLinear", "Q12/T12\nSVR", "Q16/T16\nSVR"]
    offsets = {"Closed − ESN": -0.18, "Open − ESN": 0.0, "Open − closed": 0.18}
    for index, ((dataset, variant), ax) in enumerate(zip(panels, axes.flat)):
        frame = effects[(effects.dataset == dataset) & (effects.token_variant == variant)]
        for cat_index, (q, t, decoder) in enumerate(categories):
            sub = frame[(frame.q == q) & (frame.t == t) & (frame.decoder == decoder)]
            for _, row in sub.iterrows():
                y = len(categories) - 1 - cat_index + offsets[row.effect]
                ax.plot([row.ci_low, row.ci_high], [y, y], color=row.color, lw=1.15)
                ax.plot(row.estimate, y, marker=row.marker, color=row.color, ms=3.5)
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.set_yticks(np.arange(len(categories)))
        ax.set_yticklabels(cat_labels[::-1])
        ax.grid(axis="x", color="#E8E8E8", lw=0.55)
        token_label = "General tokens" if variant == "canonical_general" else "Global-prior tokens"
        ax.set_title(f"{dataset} · {token_label}")
        ax.set_xlabel("Paired Δ test ROC-AUC")
        panel_label(ax, chr(ord("a") + index))
    handles = [Line2D([0], [0], color=c, marker=m, lw=1.2, ms=3.5, label=l) for _, _, l, c, m in specs]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.19, right=0.99, top=0.91, bottom=0.18, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig3_fixed_point_paired_effects")
    return effects.drop(columns=["color", "marker"])


def plot_p_by_t() -> pd.DataFrame:
    sources = [
        ("canonical_general", RESULTS / "p_by_T_diagnostic" / "summary_p_by_T.csv"),
        ("canonical_global_prior", RESULTS / "p_by_T_global_prior_diagnostic" / "summary_p_by_T.csv"),
    ]
    frames = []
    for variant, path in sources:
        frame = pd.read_csv(path)
        frame["token_variant"] = variant
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    row_specs = [
        ("canonical_general", "BACE"), ("canonical_general", "BBBP"),
        ("canonical_global_prior", "BACE"), ("canonical_global_prior", "BBBP"),
    ]
    p_values = sorted(data.p.unique())
    x = np.arange(len(p_values))
    p_index = {round(float(p), 8): i for i, p in enumerate(p_values)}
    t_colors = {6: "#B4C0E4", 12: "#7884B4", 20: "#484878", 30: "#9A4D8E"}
    ymin = float(data.delta_auc_mean.min() - data.delta_auc_sd.max() - 0.015)
    ymax = float(data.delta_auc_mean.max() + data.delta_auc_sd.max() + 0.015)
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.8), sharex=True, sharey=True)
    for row, (variant, dataset) in enumerate(row_specs):
        for col, decoder in enumerate(["linear", "rbf_svr"]):
            ax = axes[row, col]
            frame = data[(data.token_variant == variant) & (data.dataset == dataset) & (data.decoder == decoder)]
            for t in [6, 12, 20, 30]:
                sub = frame[frame.t == t].sort_values("p")
                px = np.asarray([p_index[round(float(p), 8)] for p in sub.p])
                y = sub.delta_auc_mean.to_numpy()
                sd = sub.delta_auc_sd.to_numpy()
                ax.plot(px, y, color=t_colors[t], marker="o", ms=2.2, lw=1.1, label=f"T={t}")
                ax.fill_between(px, y - sd, y + sd, color=t_colors[t], alpha=0.12, linewidth=0)
            ax.axhline(0, color="#888888", lw=0.75, ls="--")
            ax.axvline(p_index[round(OPEN_P, 8)], color=COLORS["open_dtc_qrc"], lw=0.85, ls=":")
            ax.set_ylim(ymin, ymax)
            ax.grid(axis="y", color="#E8E8E8", lw=0.5)
            if row == 0:
                ax.set_title("Linear" if decoder == "linear" else "RBF-SVR")
            token = "General" if variant == "canonical_general" else "Global prior"
            if col == 0:
                ax.set_ylabel(f"{dataset} · {token}\nΔAUC vs p=0")
            if row == 3:
                ax.set_xlabel("Reset probability, p")
            panel_label(ax, chr(ord("a") + row * 2 + col))
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p:g}" for p in p_values], rotation=45, ha="right")
    handles = [Line2D([0], [0], color=t_colors[t], marker="o", lw=1.1, ms=2.5, label=f"T={t}") for t in [6, 12, 20, 30]]
    handles.append(Line2D([0], [0], color=COLORS["open_dtc_qrc"], ls=":", lw=1, label="Fixed p=0.10"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.11, wspace=0.14, hspace=0.24)
    save_figure(fig, "fig4_p_by_t_dissipation")
    return data


def plot_esn_width(width: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=True, sharey=True)
    columns = [
        ("canonical_general", "linear", "General · Linear"),
        ("canonical_general", "rbf_svr", "General · RBF-SVR"),
        ("canonical_global_prior", "linear", "Global prior · Linear"),
        ("canonical_global_prior", "rbf_svr", "Global prior · RBF-SVR"),
    ]
    line_specs = [
        (12, 12, "#B4C0E4", "o", "Q12/T12"),
        (12, 30, "#484878", "o", "Q12/T30"),
        (16, 12, "#E9A6A1", "s", "Q16/T12"),
        (16, 30, "#B64342", "s", "Q16/T30"),
    ]
    for row, dataset in enumerate(["BACE", "BBBP"]):
        for col, (variant, decoder, title) in enumerate(columns):
            ax = axes[row, col]
            for q, t, color, marker, label in line_specs:
                sub = width[
                    (width.dataset == dataset) & (width.token_variant == variant)
                    & (width.decoder == decoder) & (width.q == q) & (width.t == t)
                ].sort_values("width")
                ax.plot(sub.width, sub.test_auc_mean, color=color, marker=marker, ms=3, lw=1.1, label=label)
                ax.fill_between(
                    sub.width.to_numpy(),
                    (sub.test_auc_mean - sub.test_auc_sd).to_numpy(),
                    (sub.test_auc_mean + sub.test_auc_sd).to_numpy(),
                    color=color, alpha=0.10, linewidth=0,
                )
            ax.set_xscale("log")
            ax.set_xticks([78, 136, 500, 1000, 2000])
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.set_ylim(0.48, 0.92)
            ax.grid(axis="y", color="#E8E8E8", lw=0.5)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{dataset}\nTest ROC-AUC")
            if row == 1:
                ax.set_xlabel("ESN nodes")
            panel_label(ax, chr(ord("a") + row * 4 + col))
    handles = [Line2D([0], [0], color=c, marker=m, lw=1.1, ms=3, label=l) for _, _, c, m, l in line_specs]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.17, wspace=0.16, hspace=0.25)
    save_figure(fig, "fig5_esn_width_scan_separate")


def plot_wg(wg: pd.DataFrame) -> None:
    # Keep this figure reproducible when plot_wg is called independently of main().
    apply_style()
    grid = wg[~wg.is_canonical_anchor.astype(bool)].copy()
    matrices = []
    for p in [0.0, 0.1]:
        matrices.append(grid[np.isclose(grid.p, p)].pivot(index="g", columns="W", values="test_auc_mean").sort_index().sort_index(axis=1))
    delta = matrices[1] - matrices[0]
    vmin = min(float(m.min().min()) for m in matrices)
    vmax = max(float(m.max().max()) for m in matrices)
    dlim = max(abs(float(delta.min().min())), abs(float(delta.max().max())))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    titles = ["Closed DTC-QRC (p=0)", "Dissipative DTC-QRC (p=0.10)", "Dissipative − closed"]
    for index, (ax, matrix, title) in enumerate(zip(axes, [*matrices, delta], titles)):
        if index < 2:
            im = ax.imshow(matrix.to_numpy(), origin="lower", aspect="equal", cmap="Blues", vmin=vmin, vmax=vmax, extent=(-0.05, 1.05, -0.05, 1.05))
        else:
            im = ax.imshow(matrix.to_numpy(), origin="lower", aspect="equal", cmap="RdBu_r", vmin=-dlim, vmax=dlim, extent=(-0.05, 1.05, -0.05, 1.05))
        ax.plot(W, G, marker="*", color="#FFD700", mec="#333333", mew=0.4, ms=7, zorder=4)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Disorder scale, W")
        if index == 0:
            ax.set_ylabel("Drive, g")
        else:
            ax.set_yticklabels([])
        ax.set_title(title)
        panel_label(ax, chr(ord("a") + index))
        for spine in ax.spines.values():
            spine.set_visible(True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Test ROC-AUC" if index < 2 else "ΔAUC")
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.17, wspace=0.28)
    save_figure(fig, "fig6_w_g_closed_open_heatmaps")


def _best_input_matched_classical(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[~frame.input_scope.str.contains("target-informed", na=False)].copy()
    return frame.loc[frame.groupby("target").validation_mae_ev.idxmin()].copy()


def plot_hopping_performance() -> tuple[pd.DataFrame, pd.DataFrame]:
    open_summary = pd.read_csv(HOP_OPEN / "outputs" / "best_reservoir_summary.csv")
    open_classical = _best_input_matched_classical(HOP_OPEN / "outputs" / "classical_test.csv")
    closed_summary = pd.read_csv(HOP_CLOSED / "outputs" / "reservoir_summary.csv")
    closed_classical = _best_input_matched_classical(HOP_CLOSED / "outputs" / "classical_selected_test.csv")
    open_pair = pd.read_csv(HOP_OPEN / "outputs" / "paired_trajectory_comparisons.csv")
    open_pair = open_pair[open_pair.comparison == "qrc_minus_esn"].copy()
    open_pair["protocol"] = "Dissipative, W=0.5"
    closed_pair = pd.read_csv(HOP_CLOSED / "outputs" / "paired_trajectory_comparisons.csv")
    closed_pair = closed_pair[(closed_pair.decoder == "svr") & (closed_pair.comparison == "qrc_minus_esn_standard")].copy()
    closed_pair["protocol"] = "Closed, W=0.6"
    paired = pd.concat([open_pair, closed_pair], ignore_index=True, sort=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.05, 1.45, 1.25]})
    x = np.arange(len(TARGET_ORDER))
    # a: dissipative validation-selected protocol
    ax = axes[0]
    methods = [("qrc", "QRC", COLORS["qrc"]), ("esn", "ESN", COLORS["esn"]), ("classical", "Best classical", COLORS["classical"])]
    width_bar = 0.24
    for i, (method, label, color) in enumerate(methods):
        means, errors = [], []
        for target in TARGET_ORDER:
            if method == "classical":
                row = open_classical[open_classical.target == target].iloc[0]
                means.append(row.mae_ev); errors.append(np.nan)
            else:
                row = open_summary[(open_summary.system == method) & (open_summary.target == target)].iloc[0]
                means.append(row.mae_ev_mean); errors.append(row.mae_ev_sample_std)
        offset = (i - 1) * width_bar
        ax.bar(x + offset, means, width_bar, color=color, label=label, edgecolor="white", linewidth=0.4)
        if method != "classical":
            ax.errorbar(x + offset, means, yerr=errors, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.8)
    ax.set_xticks(x); ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right")
    ax.set_ylabel("Test MAE (eV)"); ax.set_title("Dissipative protocol\nW=0.5; validation-selected p/readout")
    ax.grid(axis="y", color="#E8E8E8", lw=0.5); panel_label(ax, "a", x=0.02, y=0.98, va="top")
    # b: independent closed protocol
    ax = axes[1]
    closed_methods = [
        ("qrc", "QRC", COLORS["qrc"]), ("esn_standard", "Standard ESN", COLORS["esn"]),
        ("esn_second_order_recurrent", "2nd-order ESN", COLORS["second_esn"]),
        ("fp_cvr", "FP-CVR", COLORS["fp_cvr"]), ("classical", "Best classical", COLORS["classical"]),
    ]
    width_bar = 0.16
    for i, (method, label, color) in enumerate(closed_methods):
        means, errors = [], []
        for target in TARGET_ORDER:
            if method == "classical":
                row = closed_classical[closed_classical.target == target].iloc[0]
                means.append(row.mae_ev); errors.append(np.nan)
            else:
                row = closed_summary[(closed_summary.system == method) & (closed_summary.decoder == "svr") & (closed_summary.target == target)].iloc[0]
                means.append(row.mae_ev_mean); errors.append(row.mae_ev_sample_std)
        offset = (i - 2) * width_bar
        ax.bar(x + offset, means, width_bar, color=color, label=label, edgecolor="white", linewidth=0.35)
        if method != "classical":
            ax.errorbar(x + offset, means, yerr=errors, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5)
    ax.set_xticks(x); ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right")
    ax.set_title("Independent coherent protocol\np=0; W=0.6; SVR")
    ax.grid(axis="y", color="#E8E8E8", lw=0.5); panel_label(ax, "b", x=0.02, y=0.98, va="top")
    # c: trajectory-paired QRC minus matched standard ESN
    ax = axes[2]
    protocol_specs = [("Dissipative, W=0.5", COLORS["open_dtc_qrc"], "s", -0.13), ("Closed, W=0.6", COLORS["closed_dtc_qrc"], "o", 0.13)]
    for protocol, color, marker, offset in protocol_specs:
        sub = paired[paired.protocol == protocol].set_index("target").loc[TARGET_ORDER]
        y = np.arange(len(TARGET_ORDER))[::-1] + offset
        ax.hlines(y, sub.bootstrap_95ci_low_ev, sub.bootstrap_95ci_high_ev, color=color, lw=1.2)
        ax.plot(sub.mean_mae_difference_ev, y, marker=marker, color=color, ls="none", ms=3.8, label=protocol)
    ax.axvline(0, color="#888888", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(TARGET_ORDER))[::-1]); ax.set_yticklabels([TARGET_LABELS[t].replace("\n", " ") for t in TARGET_ORDER])
    ax.set_xlabel("QRC − ESN trajectory MAE (eV)"); ax.set_title("Paired trajectory effects\n95% bootstrap CI")
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.4)
    ax.grid(axis="x", color="#E8E8E8", lw=0.5); panel_label(ax, "c", x=0.02, y=0.98, va="top")
    handles = [
        Line2D([0], [0], color=COLORS["qrc"], lw=5, label="QRC"), Line2D([0], [0], color=COLORS["esn"], lw=5, label="Standard ESN"),
        Line2D([0], [0], color=COLORS["second_esn"], lw=5, label="2nd-order ESN"), Line2D([0], [0], color=COLORS["fp_cvr"], lw=5, label="FP-CVR"),
        Line2D([0], [0], color=COLORS["classical"], lw=5, label="Best input-matched classical"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.30, wspace=0.36)
    save_figure(fig, "fig7_surface_hopping_performance")
    return paired, pd.concat([open_classical.assign(protocol="dissipative"), closed_classical.assign(protocol="closed")], ignore_index=True)


def plot_hopping_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    p_scan = pd.read_csv(HOP_OPEN / "figures" / "source_data" / "qrc_svr_p_scan_envelope.csv")
    selected_p = pd.read_csv(HOP_OPEN / "figures" / "source_data" / "selected_qrc_p.csv")
    tm3 = pd.read_csv(HOP_OPEN / "figures" / "source_data" / "tm3_final_comparison.csv")
    history = pd.read_csv(HOP_CLOSED / "outputs" / "history_scan_validation.csv")
    history = history[(history.system == "qrc") & (history.decoder == "svr")]
    history_summary = history.groupby(["target", "history_steps", "history_fs"], as_index=False).validation_mae_ev.agg(["mean", "std"]).reset_index()
    target_colors = {"gap_1fs": "#0F4D92", "gap_5fs": "#42949E", "gap_10fs": "#9A4D8E", "min_gap_next_5fs": "#E28E2C"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    # a
    ax = axes[0, 0]
    for target in TARGET_ORDER:
        sub = p_scan[p_scan.target == target].groupby("p").validation_mae_ev.agg(["mean", "std"]).reset_index()
        ax.plot(sub.p, sub["mean"], color=target_colors[target], marker="o", ms=2.5, lw=1.1, label=TARGET_LABELS[target].replace("\n", " "))
        ax.fill_between(sub.p, sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=target_colors[target], alpha=0.12, linewidth=0)
    ax.set_xscale("symlog", linthresh=0.005)
    p_ticks = [0, .0025, .01, .04, .16, .64, 1.0]
    ax.set_xticks(p_ticks)
    ax.set_xticklabels(["0", ".0025", ".01", ".04", ".16", ".64", "1"])
    ax.set_xlabel("Reset probability, p"); ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("Dissipative QRC · SVR\nbest of final/TM3 at each p")
    ax.grid(axis="y", color="#E8E8E8", lw=0.5); panel_label(ax, "a")
    # b
    ax = axes[0, 1]
    long_p = selected_p.melt(id_vars="target", var_name="seed", value_name="selected_p")
    for i, target in enumerate(TARGET_ORDER):
        vals = long_p[long_p.target == target].selected_p.to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=target_colors[target], s=16, zorder=3)
        ax.plot([i - 0.16, i + 0.16], [np.mean(vals), np.mean(vals)], color="#333333", lw=1)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Validation-selected p"); ax.set_ylim(-0.05, 1.08)
    ax.set_title("Memory selected by forecast target\nthree reservoir seeds")
    ax.grid(axis="y", color="#E8E8E8", lw=0.5); panel_label(ax, "b")
    # c
    ax = axes[1, 0]
    width_bar = 0.34
    for i, (decoder, color) in enumerate([("linear", "#7884B4"), ("svr", "#E28E2C")]):
        sub = tm3[tm3.decoder == decoder].set_index("target").loc[TARGET_ORDER]
        ax.bar(np.arange(4) + (i - 0.5) * width_bar, sub.tm3_minus_final_mae_ev, width_bar, color=color, label=decoder.upper())
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=30, ha="right")
    ax.set_ylabel("TM3 − final test MAE (eV)"); ax.set_title("Temporal readout effect\nnegative favors TM3")
    ax.legend(ncol=2, loc="lower right"); ax.grid(axis="y", color="#E8E8E8", lw=0.5); panel_label(ax, "c")
    # d
    ax = axes[1, 1]
    for target in TARGET_ORDER:
        sub = history_summary[history_summary.target == target].sort_values("history_steps")
        ax.plot(sub.history_fs, sub["mean"], color=target_colors[target], marker="o", ms=2.5, lw=1.1, label=TARGET_LABELS[target].replace("\n", " "))
        ax.fill_between(sub.history_fs, sub["mean"] - sub["std"], sub["mean"] + sub["std"], color=target_colors[target], alpha=0.12, linewidth=0)
    ax.set_xlabel("Visible history (fs)"); ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("Closed QRC · p=0, W=0.6\nSVR history scan")
    ax.grid(axis="y", color="#E8E8E8", lw=0.5); panel_label(ax, "d")
    handles = [Line2D([0], [0], color=target_colors[t], marker="o", lw=1.1, ms=2.5, label=TARGET_LABELS[t].replace("\n", " ")) for t in TARGET_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.18, wspace=0.28, hspace=0.43)
    save_figure(fig, "fig8_surface_hopping_memory_diagnostics")
    return long_p, history_summary


def write_bundle_notes(files: list[str]) -> None:
    captions = """# Complete result figure captions

## Fig. 1 | Fair fixed-Q/T comparison with general molecular tokens

Test ROC-AUC for closed DTC-QRC, fixed-p open DTC-QRC and matched-width ESN at every identical visible event length T. Curves show mean and shaded sample SD across six paired seeds. Q12 and Q16 are displayed separately; linear and RBF-SVR use the same frozen states. The dashed line marks chance.

## Fig. 2 | Global chemical-prior token supplement

Same fixed-Q/T comparison as Fig. 1 using the mirrored global-prior token protocol. No Q, T, p or model width is selected from test performance.

## Fig. 3 | Paired effects at prespecified fixed-size points

Within-seed test ROC-AUC differences at Q12/T12 and Q16/T16. Points are mean paired differences and intervals are 95% bootstrap intervals from 10,000 resamples of six paired seeds. Positive QRC-minus-ESN values favor QRC; positive open-minus-closed values favor p=0.10.

## Fig. 4 | Dissipation-history interaction

Change in test ROC-AUC relative to the same-T closed state across the complete p grid at Q12. Curves and ribbons show mean and sample SD across six paired seeds. The orange dotted line marks the globally prespecified p=0.10. The plot is diagnostic and does not select p from test performance.

## Fig. 5 | Independent ESN capacity scaling

Test ROC-AUC versus ESN node count at T12 and T30. Matched widths (78 nodes for Q12; 136 for Q16) and 500/1000/2000-node controls are kept separate from the equal-width main comparison. Curves show mean and sample SD across six paired seeds.

## Fig. 6 | Closed and open Floquet parameter landscapes

BACE/general/Q12/T30 test ROC-AUC across the complete 0.1-spaced W-g grid for p=0 and fixed p=0.10, plus their paired difference. Cells show six-seed means; the star marks the separately evaluated canonical point g=0.84, W=0.5 and is not a heat-map cell.

## Fig. 7 | Surface-hopping prediction performance

Validation-selected dissipative QRC and matched ESN, the independent closed p=0 protocol, and input-matched classical controls on 60 held-out SHNITSEL A01 trajectories. Reservoir bars are mean ± sample SD across three seeds; deterministic classical models have no invented error bars. The paired panel reports trajectory-bootstrap 95% intervals. The closed W=0.6 and dissipative W=0.5 experiments are separate protocols, not a one-factor ablation.

## Fig. 8 | Surface-hopping memory diagnostics

Validation MAE across the exact p scan, validation-selected p for three reservoir seeds, TM3-minus-final test MAE, and the independent closed-QRC history scan. Negative TM3-minus-final values favor temporal readout. No test trajectory enters normalization or model selection.

Source data are provided in the source_data directory.
"""
    qa = f"""# Figure bundle QA

- Backend: Python/matplotlib exclusively.
- Core claim: dissipation is task-, history- and decoder-dependent; all primary QRC/ESN comparisons use identical T.
- Main classification statistics: mean ± sample SD across n=6 paired seeds.
- Surface-hopping statistics: mean ± sample SD across n=3 reservoir seeds; paired intervals use 10,000 bootstrap resamples of n=60 complete test trajectories.
- Closed/open hopping protocols are visibly separated because W and model-selection scope differ.
- Wide ESNs appear only in the independent capacity figure.
- The p-by-T and W-g test grids are fully retained and are not used for operating-point selection.
- SVG text is editable; PDF uses TrueType fonts; TIFF is 600 dpi with LZW compression.
- Generated stems: {', '.join(files)}.
"""
    (OUT / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    (OUT / "FIGURE_QA.md").write_text(qa, encoding="utf-8")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    main_summary = pd.read_csv(RESULTS / "main_fixed_q_t" / "main_pointwise_summary.csv")
    main_metrics = pd.read_csv(RESULTS / "main_fixed_q_t" / "main_pointwise_metrics_all_seeds.csv")
    width = pd.read_csv(RESULTS / "main_fixed_q_t" / "esn_width_summary.csv")
    wg = pd.read_csv(RESULTS / "w_g_heatmaps" / "summary_w_g.csv")
    plot_main_curves(main_summary, "canonical_general", "fig1_general_fair_fixed_q_t")
    plot_main_curves(main_summary, "canonical_global_prior", "fig2_global_prior_fair_fixed_q_t")
    effects = plot_fixed_effects(main_metrics)
    pt = plot_p_by_t()
    plot_esn_width(width)
    plot_wg(wg)
    paired_hop, classical_hop = plot_hopping_performance()
    selected_p, closed_history = plot_hopping_diagnostics()
    main_summary.to_csv(SOURCE / "main_pointwise_summary.csv", index=False)
    effects.to_csv(SOURCE / "fixed_point_paired_effects.csv", index=False)
    pt.to_csv(SOURCE / "p_by_t_summary.csv", index=False)
    width.to_csv(SOURCE / "esn_width_summary.csv", index=False)
    wg.to_csv(SOURCE / "w_g_summary.csv", index=False)
    paired_hop.to_csv(SOURCE / "surface_hopping_paired_effects.csv", index=False)
    classical_hop.to_csv(SOURCE / "surface_hopping_selected_classical.csv", index=False)
    selected_p.to_csv(SOURCE / "surface_hopping_selected_p.csv", index=False)
    closed_history.to_csv(SOURCE / "surface_hopping_closed_history_summary.csv", index=False)
    stems = [
        "fig1_general_fair_fixed_q_t", "fig2_global_prior_fair_fixed_q_t",
        "fig3_fixed_point_paired_effects", "fig4_p_by_t_dissipation",
        "fig5_esn_width_scan_separate", "fig6_w_g_closed_open_heatmaps",
        "fig7_surface_hopping_performance", "fig8_surface_hopping_memory_diagnostics",
    ]
    write_bundle_notes(stems)
    manifest = {
        "status": "complete", "backend": "python-matplotlib", "figure_stems": stems,
        "formats": ["svg", "pdf", "png", "tiff"], "raster_dpi": 600,
        "main_seeds": 6, "hopping_reservoir_seeds": 3, "hopping_test_trajectories": 60,
        "canonical": {"g": G, "W": W, "open_p": OPEN_P},
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
