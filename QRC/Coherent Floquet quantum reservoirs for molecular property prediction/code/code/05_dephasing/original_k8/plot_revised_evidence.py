"""Create the revised molecular-event evidence figures from frozen results.

The primary graph-derived comparison uses one fixed physical width (Q=12),
identical event length T for QRC and ESN, the general local-token protocol, and
RBF-SVR throughout.  Capacity-expanded ESNs are kept in a separate figure and
their width is selected on validation data within each seed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from protocol import MAIN_ROOT, ROOT


MAIN_RESULTS = MAIN_ROOT / "results" / "main_fixed_q_t"
OUT = ROOT / "figures" / "revised_evidence"
SOURCE = OUT / "source_data"
T_VALUES = (2, 4, 6, 8, 10, 12, 16, 20, 24, 30)
SYSTEMS = ("closed_dtc_qrc", "open_dtc_qrc", "matched_esn")
COLORS = {
    "closed_dtc_qrc": "#214D8C",
    "open_dtc_qrc": "#D9822B",
    "matched_esn": "#2A8C82",
}
LABELS = {
    "closed_dtc_qrc": "Closed DTC-QRC",
    "open_dtc_qrc": "Dissipative DTC-QRC, p=0.10",
    "matched_esn": "Matched-width ESN",
}
MARKERS = {"closed_dtc_qrc": "o", "open_dtc_qrc": "s", "matched_esn": "^"}


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
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def bootstrap_mean_ci(values: np.ndarray, key: str, draws: int = 20_000):
    values = np.asarray(values, dtype=float)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = samples.mean(axis=1)
    return float(values.mean()), *[float(x) for x in np.quantile(means, [0.025, 0.975])]


def load_primary(q: int) -> pd.DataFrame:
    raw = pd.read_csv(MAIN_RESULTS / "main_pointwise_metrics_all_seeds.csv")
    return raw[
        (raw.token_variant == "canonical_general")
        & (raw.decoder == "rbf_svr")
        & (raw.q == q)
        & raw.system.isin(SYSTEMS)
    ].copy()


def summarize_curves(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["dataset", "q", "t", "system"], as_index=False)
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", lambda x: float(np.std(x, ddof=1))),
            validation_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )


def paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    pivot = raw.pivot_table(
        index=["dataset", "q", "t", "seed"], columns="system", values="test_auc"
    ).dropna()
    pivot["delta_open_minus_esn"] = pivot.open_dtc_qrc - pivot.matched_esn
    rows = []
    for (dataset, q, t), group in pivot.groupby(level=[0, 1, 2]):
        values = group.delta_open_minus_esn.to_numpy()
        mean, low, high = bootstrap_mean_ci(values, f"{dataset}-{q}-{t}")
        rows.append(
            {
                "dataset": dataset,
                "q": q,
                "t": t,
                "delta_auc_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "wins": int(np.sum(values > 0)),
                "n_seeds": len(values),
            }
        )
    return pd.DataFrame(rows)


def exact_long_stream_contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    """Paired contrasts at each long-stream event length; never pool across T."""
    pivot = raw.pivot_table(
        index=["dataset", "q", "t", "seed"], columns="system", values="test_auc"
    ).dropna()
    comparisons = (
        ("dissipative_minus_matched_esn", "open_dtc_qrc", "matched_esn"),
        ("dissipative_minus_closed", "open_dtc_qrc", "closed_dtc_qrc"),
        ("closed_minus_matched_esn", "closed_dtc_qrc", "matched_esn"),
    )
    rows = []
    for (dataset, q, t), frame in pivot.groupby(level=[0, 1, 2]):
        if int(t) not in (24, 30):
            continue
        for label, left, right in comparisons:
            values = (frame[left] - frame[right]).to_numpy(dtype=float)
            mean, low, high = bootstrap_mean_ci(
                values, f"long-{dataset}-{q}-{t}-{label}"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "q": int(q),
                    "t": int(t),
                    "comparison": label,
                    "delta_auc_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "positive_seeds": int(np.sum(values > 0)),
                    "zero_seeds": int(np.sum(np.isclose(values, 0.0))),
                    "n_seeds": len(values),
                    "information_scope": "same T only",
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def plot_primary(q: int, stem_name: str) -> None:
    raw = load_primary(q)
    curves = summarize_curves(raw)
    effects = paired_effects(raw)
    long_stream = exact_long_stream_contrasts(raw)
    SOURCE.mkdir(parents=True, exist_ok=True)
    curves.to_csv(SOURCE / f"graph_stream_q{q}_curves.csv", index=False)
    effects.to_csv(SOURCE / f"graph_stream_q{q}_paired_by_t.csv", index=False)
    long_stream.to_csv(
        SOURCE / f"graph_stream_q{q}_long_stream_exact_contrasts.csv", index=False
    )

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), sharex="col")
    for col, dataset in enumerate(("BACE", "BBBP")):
        ax = axes[0, col]
        frame = curves[curves.dataset == dataset]
        for system in SYSTEMS:
            sub = frame[frame.system == system].sort_values("t")
            x = sub.t.to_numpy(dtype=float)
            y = sub.test_auc_mean.to_numpy(dtype=float)
            sd = sub.test_auc_sd.to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=COLORS[system],
                marker=MARKERS[system],
                ms=3.4,
                lw=1.35,
                label=LABELS[system],
            )
            ax.fill_between(x, y - sd, y + sd, color=COLORS[system], alpha=0.12, linewidth=0)
        ax.axhline(0.5, color="#999999", ls="--", lw=0.7)
        ax.axvspan(22.0, 30.7, color="#D9822B", alpha=0.055, linewidth=0)
        ax.set_title(f"{dataset}, general local tokens, Q={q}")
        ax.set_ylabel("Test ROC-AUC")
        ax.set_ylim(0.48, 0.84 if dataset == "BACE" else 0.82)
        ax.grid(axis="y", color="#E7E7E7", lw=0.55)

        effect_ax = axes[1, col]
        sub = effects[effects.dataset == dataset].sort_values("t")
        x = sub.t.to_numpy(dtype=float)
        y = sub.delta_auc_mean.to_numpy(dtype=float)
        effect_ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - sub.ci_low.to_numpy(), sub.ci_high.to_numpy() - y)),
            color=COLORS["open_dtc_qrc"],
            marker="s",
            ms=3.5,
            lw=1.1,
            capsize=2.0,
        )
        effect_ax.axhline(0, color="#777777", ls="--", lw=0.75)
        effect_ax.axvspan(22.0, 30.7, color="#D9822B", alpha=0.055, linewidth=0)
        effect_ax.grid(axis="y", color="#E7E7E7", lw=0.55)
        effect_ax.set_ylabel(r"Paired $\Delta$AUC, dissipative QRC $-$ ESN")
        effect_ax.set_xlabel("Visible molecular events, T")
        long_stream = sub[sub.t.isin([24, 30])]
        exact_lines = []
        for row in long_stream.itertuples():
            exact_lines.append(
                f"T={int(row.t)}: {row.delta_auc_mean:+.3f} "
                f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}], "
                f"{int(row.wins)}/{int(row.n_seeds)}"
            )
        effect_ax.text(
            0.98,
            0.05,
            "Exact-T paired effects\n" + "\n".join(exact_lines),
            transform=effect_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.3,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
        )

    for ax in axes.flat:
        ax.set_xticks(T_VALUES)
        ax.set_xlim(1.3, 30.7)
    for index, ax in enumerate(axes.flat):
        panel_label(ax, chr(ord("a") + index))
    handles = [
        Line2D(
            [0], [0], color=COLORS[s], marker=MARKERS[s], lw=1.35, ms=3.4, label=LABELS[s]
        )
        for s in SYSTEMS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.92, bottom=0.16, wspace=0.25, hspace=0.32)
    save_figure(fig, OUT / stem_name)


def select_wide_esn(width: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    groups = ["dataset", "token_variant", "q", "t", "decoder", "seed"]
    for _, frame in width.groupby(groups, sort=True):
        chosen = frame.sort_values(
            ["val_auc", "val_brier", "width"], ascending=[False, True, True]
        ).iloc[0]
        selected_rows.append(chosen)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def plot_wide_esn_control() -> None:
    width = pd.read_csv(MAIN_RESULTS / "esn_width_metrics_all_seeds.csv")
    width = width[
        (width.token_variant == "canonical_general")
        & (width.decoder == "rbf_svr")
        & (width.q == 12)
        & width.t.isin([12, 30])
    ].copy()
    qrc = load_primary(12)
    qrc = qrc[
        (qrc.system == "open_dtc_qrc") & qrc.t.isin([12, 30])
    ].copy()
    selected = select_wide_esn(width)
    SOURCE.mkdir(parents=True, exist_ok=True)
    width.to_csv(SOURCE / "wide_esn_all_seed_metrics.csv", index=False)
    selected.to_csv(SOURCE / "wide_esn_validation_selected.csv", index=False)

    summary = (
        width.groupby(["dataset", "t", "width"], as_index=False)
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", lambda x: float(np.std(x, ddof=1))),
            val_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    qrc_summary = (
        qrc.groupby(["dataset", "t"], as_index=False)
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", lambda x: float(np.std(x, ddof=1))),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(SOURCE / "wide_esn_summary.csv", index=False)
    qrc_summary.to_csv(SOURCE / "wide_esn_qrc_reference.csv", index=False)

    paired_rows = []
    merged = qrc.merge(
        selected[["dataset", "t", "seed", "width", "test_auc"]].rename(
            columns={"test_auc": "esn_test_auc", "width": "selected_width"}
        ),
        on=["dataset", "t", "seed"],
        validate="one_to_one",
    )
    merged["delta_qrc_minus_selected_wide_esn"] = merged.test_auc - merged.esn_test_auc
    for (dataset, t), frame in merged.groupby(["dataset", "t"]):
        mean, low, high = bootstrap_mean_ci(
            frame.delta_qrc_minus_selected_wide_esn.to_numpy(), f"wide-{dataset}-{t}"
        )
        paired_rows.append(
            {
                "dataset": dataset,
                "t": t,
                "delta_auc_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "qrc_wins": int(np.sum(frame.delta_qrc_minus_selected_wide_esn > 0)),
                "n_seeds": len(frame),
            }
        )
    paired = pd.DataFrame(paired_rows)
    merged.to_csv(SOURCE / "wide_esn_validation_selected_paired_seed.csv", index=False)
    paired.to_csv(SOURCE / "wide_esn_validation_selected_effects.csv", index=False)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    t_colors = {12: "#6A8EC1", 30: "#B6554D"}
    for col, dataset in enumerate(("BACE", "BBBP")):
        ax = axes[col]
        for t in (12, 30):
            sub = summary[(summary.dataset == dataset) & (summary.t == t)].sort_values("width")
            ax.plot(
                sub.width,
                sub.test_auc_mean,
                color=t_colors[t],
                marker="o",
                ms=3.4,
                lw=1.3,
                label=f"ESN, T={t}",
            )
            ax.fill_between(
                sub.width.to_numpy(),
                (sub.test_auc_mean - sub.test_auc_sd).to_numpy(),
                (sub.test_auc_mean + sub.test_auc_sd).to_numpy(),
                color=t_colors[t],
                alpha=0.12,
                linewidth=0,
            )
            ref = qrc_summary[(qrc_summary.dataset == dataset) & (qrc_summary.t == t)].iloc[0]
            ax.axhline(ref.test_auc_mean, color=t_colors[t], ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks([78, 500, 1000, 2000])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel("ESN nodes")
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(dataset)
        ax.grid(axis="y", color="#E7E7E7", lw=0.55)

    ax = axes[2]
    positions = np.arange(len(paired))
    y = paired.delta_auc_mean.to_numpy()
    ax.errorbar(
        y,
        positions,
        xerr=np.vstack((y - paired.ci_low.to_numpy(), paired.ci_high.to_numpy() - y)),
        fmt="o",
        color="#333333",
        ms=3.8,
        lw=1.1,
        capsize=2,
    )
    ax.axvline(0, color="#777777", ls="--", lw=0.75)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{row.dataset}, T={int(row.t)}" for row in paired.itertuples()])
    ax.set_xlabel(r"QRC $-$ validation-selected wide ESN, $\Delta$AUC")
    ax.set_title("Capacity control")
    ax.grid(axis="x", color="#E7E7E7", lw=0.55)

    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index))
    handles = [
        Line2D([0], [0], color=t_colors[t], marker="o", lw=1.3, ms=3.4, label=f"ESN, T={t}")
        for t in (12, 30)
    ]
    handles.extend(
        Line2D([0], [0], color=t_colors[t], ls="--", lw=0.9, label=f"QRC, T={t}")
        for t in (12, 30)
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.25, wspace=0.42)
    save_figure(fig, OUT / "figS_wide_esn_capacity_control")


def main() -> None:
    plot_primary(12, "fig1_graph_stream_rbf_svr_q12")
    plot_primary(16, "figS_graph_stream_rbf_svr_q16")
    plot_wide_esn_control()
    print(OUT)


if __name__ == "__main__":
    main()
