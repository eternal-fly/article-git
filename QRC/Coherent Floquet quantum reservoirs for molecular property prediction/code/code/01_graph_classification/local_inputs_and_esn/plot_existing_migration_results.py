#!/usr/bin/env python3
"""Publication-ready visual overview of completed migration results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Publication contract: editable vector text and consistent sans-serif fonts.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results"
OUTPUT_ROOT = RESULT_ROOT / "figures" / "existing_migration_overview"

CURVE_PATH = (
    RESULT_ROOT
    / "decoder_q_t_scan"
    / "svr_fixed_q_t"
    / "test_curve_summary.csv"
)
QRC_AFFINE_PATH = RESULT_ROOT / "qrc_q_t_summary.csv"
ESN_AFFINE_PATH = RESULT_ROOT / "esn_q_t_width_summary.csv"
G_PATH = RESULT_ROOT / "bace_general_g_w_scan" / "g_scan_curve_summary.csv"
W_PATH = RESULT_ROOT / "bace_general_g_w_scan" / "W_scan_curve_summary.csv"

SEEDS = 6
Q = 12
Q_VALUES = (12, 16)
MATCHED_WIDTH = {12: 78, 16: 136}
T_VALUES = (2, 4, 6, 8, 10, 12, 16, 20, 24, 30)
SYSTEMS = ("QRC_OPEN", "QRC_CLOSED", "CRC_MATCHED")

TASKS = (
    ("BACE", "canonical_general", "BACE · source tokens"),
    ("BACE", "canonical_global_prior", "BACE · global-prior tokens"),
    ("BBBP", "canonical_general", "BBBP · source tokens"),
    ("BBBP", "canonical_global_prior", "BBBP · global-prior tokens"),
)
HEATMAP_SYSTEMS = (
    "QRC_OPEN",
    "QRC_CLOSED",
    "ESN_MATCHED",
)
SYSTEM_LABELS = {
    "QRC_OPEN": "Open DTC–QRC",
    "QRC_CLOSED": "Closed DTC–QRC",
    "CRC_MATCHED": "ESN (matched width)",
    "ESN_MATCHED": "ESN (matched)",
}
COLORS = {
    "QRC_OPEN": "#0F4D92",
    "QRC_CLOSED": "#42949E",
    "CRC_MATCHED": "#767676",
}
LINESTYLES = {"QRC_OPEN": "-", "QRC_CLOSED": "--", "CRC_MATCHED": ":"}
MARKERS = {"QRC_OPEN": "o", "QRC_CLOSED": "s", "CRC_MATCHED": "D"}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def save_bundle(fig: plt.Figure, base: Path) -> dict[str, int]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
        "tiff": base.with_suffix(".tiff"),
        "png": base.with_suffix(".png"),
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {name: path.stat().st_size for name, path in paths.items()}


def load_curve_data(q: int) -> pd.DataFrame:
    data = pd.read_csv(CURVE_PATH)
    data = data[
        (data.q == int(q))
        & data.system.isin(SYSTEMS)
        & data.t.isin(T_VALUES)
    ].copy()
    expected = len(TASKS) * len(SYSTEMS) * len(T_VALUES)
    keys = ["dataset", "token_variant", "q", "t", "system"]
    if len(data) != expected or data.duplicated(keys).any():
        raise RuntimeError(f"invalid fixed-Q curve table: {len(data)} != {expected}")
    if not (data.n_seeds == SEEDS).all():
        raise RuntimeError("fixed-Q curve does not contain six seeds everywhere")
    numeric = data[["test_auc_mean", "test_auc_std"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise RuntimeError("non-finite fixed-Q curve metrics")
    return data.sort_values(keys)


def plot_fixed_q_t_curves(
    data: pd.DataFrame, *, q: int
) -> tuple[Path, dict[str, int]]:
    apply_style()
    width_in = 183 / 25.4
    height_in = 132 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width_in, height_in), sharex=True, sharey=True)
    handles = []
    labels = []
    for panel, (ax, (dataset, variant, title)) in enumerate(zip(axes.flat, TASKS)):
        subset = data[(data.dataset == dataset) & (data.token_variant == variant)]
        for system in SYSTEMS:
            line = subset[subset.system == system].sort_values("t")
            if tuple(line.t.astype(int)) != T_VALUES:
                raise RuntimeError(f"missing T points for {dataset}/{variant}/{system}")
            x = line.t.to_numpy(dtype=float)
            mean = line.test_auc_mean.to_numpy(dtype=float)
            std = line.test_auc_std.to_numpy(dtype=float)
            artist, = ax.plot(
                x,
                mean,
                color=COLORS[system],
                linestyle=LINESTYLES[system],
                marker=MARKERS[system],
                markersize=3.2,
                linewidth=1.35,
                markerfacecolor="white" if system != "QRC_OPEN" else COLORS[system],
                markeredgewidth=0.75,
                label=(
                    f"ESN ({MATCHED_WIDTH[int(q)]} nodes)"
                    if system == "CRC_MATCHED"
                    else SYSTEM_LABELS[system]
                ),
                zorder=3,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=COLORS[system],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
            if panel == 0:
                handles.append(artist)
                labels.append(
                    f"ESN ({MATCHED_WIDTH[int(q)]} nodes)"
                    if system == "CRC_MATCHED"
                    else SYSTEM_LABELS[system]
                )
        ax.set_title(title, pad=5)
        ax.set_xlim(1.2, 30.8)
        ax.set_ylim(0.54, 0.90)
        ax.set_xticks(T_VALUES)
        ax.set_yticks((0.6, 0.7, 0.8, 0.9))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.65)
        ax.text(
            0.98,
            0.05,
            "mean ± s.d.; n = 6",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="#606060",
        )
        add_panel_label(ax, chr(ord("a") + panel))
    for ax in axes[-1, :]:
        ax.set_xlabel("Molecular event-prefix length, T")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test ROC–AUC")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.005),
        ncol=3,
        handlelength=2.8,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.91, wspace=0.18, hspace=0.31)
    base = OUTPUT_ROOT / f"fixed_Q{int(q)}_same_width_T_curves"
    sizes = save_bundle(fig, base)
    data.to_csv(
        OUTPUT_ROOT / f"source_data_fixed_Q{int(q)}_same_width_T_curves.csv",
        index=False,
    )
    return base, sizes


def load_overview_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qrc_affine = pd.read_csv(QRC_AFFINE_PATH)
    qrc_affine = qrc_affine[
        (qrc_affine.q == Q)
        & (qrc_affine.t == 30)
        & qrc_affine.system.isin(("QRC_OPEN", "QRC_CLOSED"))
    ].copy()
    qrc_affine["decoder"] = "AFFINE_ORIGINAL"
    esn_affine = pd.read_csv(ESN_AFFINE_PATH)
    esn_affine = esn_affine[
        (esn_affine.q == Q)
        & (esn_affine.t == 30)
        & (esn_affine.width == 78)
    ].copy()
    esn_affine["decoder"] = "AFFINE_ORIGINAL"
    esn_affine["system"] = "ESN_MATCHED"
    svr = pd.read_csv(CURVE_PATH)
    svr = svr[
        (svr.q == Q)
        & (svr.t == 30)
        & svr.system.isin(("QRC_OPEN", "QRC_CLOSED", "CRC_MATCHED"))
    ].copy()
    svr["decoder"] = "SVR_RBF"
    svr["system"] = svr.system.replace({"CRC_MATCHED": "ESN_MATCHED"})
    common = [
        "dataset",
        "token_variant",
        "system",
        "q",
        "t",
        "test_auc_mean",
        "test_auc_std",
        "test_acc_mean",
        "n_seeds",
        "decoder",
    ]
    decoder = pd.concat(
        [qrc_affine[common], esn_affine[common], svr[common]],
        ignore_index=True,
    )
    expected = len(TASKS) * 2 * len(HEATMAP_SYSTEMS)
    keys = ["dataset", "token_variant", "decoder", "system"]
    if len(decoder) != expected or decoder.duplicated(keys).any():
        raise RuntimeError(f"invalid decoder summary: {len(decoder)} != {expected}")
    if not (decoder.n_seeds == SEEDS).all():
        raise RuntimeError("decoder overview does not contain six seeds everywhere")
    g_scan = pd.read_csv(G_PATH).sort_values("g")
    w_scan = pd.read_csv(W_PATH).sort_values("W")
    if len(g_scan) != 12 or len(w_scan) != 10:
        raise RuntimeError("incomplete one-factor Floquet scans")
    if not (g_scan.n_seeds == SEEDS).all() or not (w_scan.n_seeds == SEEDS).all():
        raise RuntimeError("Floquet scans do not contain six seeds everywhere")
    for frame in (decoder, g_scan, w_scan):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if not np.isfinite(numeric).all():
            raise RuntimeError("non-finite overview metrics")
    return decoder.sort_values(keys), g_scan, w_scan


def decoder_matrix(decoder: pd.DataFrame, decoder_name: str) -> np.ndarray:
    matrix = np.empty((len(TASKS), len(HEATMAP_SYSTEMS)), dtype=float)
    selected = decoder[decoder.decoder == decoder_name]
    for row, (dataset, variant, _) in enumerate(TASKS):
        task = selected[(selected.dataset == dataset) & (selected.token_variant == variant)]
        for col, system in enumerate(HEATMAP_SYSTEMS):
            values = task.loc[task.system == system, "test_auc_mean"].to_numpy()
            if len(values) != 1:
                raise RuntimeError(f"missing decoder cell {dataset}/{variant}/{system}")
            matrix[row, col] = values[0]
    return matrix


def annotated_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> mpl.image.AxesImage:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    for row, col in np.ndindex(matrix.shape):
        value = float(matrix[row, col])
        red, green, blue, _ = cmap(norm(value))
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        ax.text(
            col,
            row,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if luminance < 0.48 else "#272727",
        )
    ax.set_xticks(range(len(HEATMAP_SYSTEMS)))
    ax.set_xticklabels(["Open\nQRC", "Closed\nQRC", "78-node\nESN"], rotation=0)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([title.replace(" · ", "\n") for _, _, title in TASKS])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(HEATMAP_SYSTEMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TASKS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_decoder_and_floquet(
    decoder: pd.DataFrame,
    g_scan: pd.DataFrame,
    w_scan: pd.DataFrame,
) -> tuple[Path, dict[str, int]]:
    apply_style()
    affine = decoder_matrix(decoder, "AFFINE_ORIGINAL")
    svr = decoder_matrix(decoder, "SVR_RBF")
    lower = float(min(affine.min(), svr.min()))
    upper = float(max(affine.max(), svr.max()))
    padding = max(0.003, 0.03 * (upper - lower))
    norm = mpl.colors.Normalize(vmin=lower - padding, vmax=upper + padding)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "migration_auc", ["#F5F6F8", "#AFC8DF", "#3775BA", "#0F4D92"]
    )

    width_in = 183 / 25.4
    height_in = 137 / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.02, 1.0), hspace=0.53)
    top = outer[0].subgridspec(1, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.40)
    bottom = outer[1].subgridspec(1, 2, wspace=0.42)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_colorbar = fig.add_subplot(top[0, 2])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    image = annotated_matrix(ax_a, affine, cmap=cmap, norm=norm)
    annotated_matrix(ax_b, svr, cmap=cmap, norm=norm)
    # The row order is shared; repeating long task labels would intrude into panel a.
    ax_b.set_yticklabels([])
    ax_a.set_title(r"Linear readout (fixed $Q=12$, $T=30$)", pad=5)
    ax_b.set_title(r"RBF–SVR (fixed $Q=12$, $T=30$)", pad=5)
    colorbar = fig.colorbar(image, cax=ax_colorbar)
    colorbar.set_label("Mean test ROC–AUC")

    ax_c.plot(
        g_scan.g,
        g_scan.test_auc_mean,
        color="#0F4D92",
        marker="o",
        markersize=3.2,
        linewidth=1.4,
    )
    ax_c.fill_between(
        g_scan.g,
        g_scan.test_auc_mean - g_scan.test_auc_std,
        g_scan.test_auc_mean + g_scan.test_auc_std,
        color="#0F4D92",
        alpha=0.12,
        linewidth=0,
    )
    ax_c.axvline(0.84, color="#767676", linestyle=":", linewidth=1.0)
    ax_c.text(0.84, 0.755, "anchor", rotation=90, va="bottom", ha="right", fontsize=5.5, color="#606060")
    ax_c.set_xlabel(r"Floquet pulse parameter, $g$  ($W=0.5$)")
    ax_c.set_ylabel("Test ROC–AUC")
    ax_c.set_title("BACE/source: one-factor g sensitivity", pad=5)

    ax_d.plot(
        w_scan.W,
        w_scan.test_auc_mean,
        color="#42949E",
        marker="s",
        markersize=3.2,
        linewidth=1.4,
    )
    ax_d.fill_between(
        w_scan.W,
        w_scan.test_auc_mean - w_scan.test_auc_std,
        w_scan.test_auc_mean + w_scan.test_auc_std,
        color="#42949E",
        alpha=0.12,
        linewidth=0,
    )
    ax_d.axvline(0.50, color="#767676", linestyle=":", linewidth=1.0)
    ax_d.text(0.50, 0.755, "anchor", rotation=90, va="bottom", ha="right", fontsize=5.5, color="#606060")
    ax_d.set_xlabel(r"Disorder strength, $W$  ($g=0.84$)")
    ax_d.set_ylabel("Test ROC–AUC")
    ax_d.set_title("BACE/source: one-factor W sensitivity", pad=5)

    sensitivity_min = float(
        min(
            (g_scan.test_auc_mean - g_scan.test_auc_std).min(),
            (w_scan.test_auc_mean - w_scan.test_auc_std).min(),
        )
    )
    sensitivity_max = float(
        max(
            (g_scan.test_auc_mean + g_scan.test_auc_std).max(),
            (w_scan.test_auc_mean + w_scan.test_auc_std).max(),
        )
    )
    margin = max(0.008, 0.08 * (sensitivity_max - sensitivity_min))
    for ax in (ax_c, ax_d):
        ax.set_ylim(sensitivity_min - margin, sensitivity_max + margin)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.65)
        ax.text(
            0.98,
            0.05,
            "mean ± s.d.; n = 6",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="#606060",
        )
    ax_c.set_xticks((0.1, 0.3, 0.5, 0.7, 0.9, 1.0))
    ax_d.set_xticks((0.0, 0.2, 0.4, 0.6, 0.8))

    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(ax, label)
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.10, top=0.96)
    base = OUTPUT_ROOT / "decoder_and_floquet_overview"
    sizes = save_bundle(fig, base)
    decoder.to_csv(OUTPUT_ROOT / "source_data_decoder_summary.csv", index=False)
    g_scan.to_csv(OUTPUT_ROOT / "source_data_g_scan.csv", index=False)
    w_scan.to_csv(OUTPUT_ROOT / "source_data_W_scan.csv", index=False)
    return base, sizes


def qa_export(base: Path, sizes: dict[str, int]) -> dict:
    svg_path = base.with_suffix(".svg")
    svg = svg_path.read_text(encoding="utf-8")
    required = ("svg", "pdf", "tiff", "png")
    return {
        "base": base.name,
        "exports": sizes,
        "all_exports_nonempty": all(sizes.get(name, 0) > 0 for name in required),
        "editable_svg_text": "<text" in svg,
        "python_only": True,
        "passes": all(sizes.get(name, 0) > 0 for name in required) and "<text" in svg,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    curve_exports = {}
    for q in Q_VALUES:
        curves = load_curve_data(q)
        curve_base, curve_sizes = plot_fixed_q_t_curves(curves, q=q)
        curve_exports[q] = (curve_base, curve_sizes)
    decoder, g_scan, w_scan = load_overview_data()
    overview_base, overview_sizes = plot_decoder_and_floquet(decoder, g_scan, w_scan)

    contracts = {
        "decoder_and_floquet_overview": {
            "core_conclusion": (
                "At the same Q12/T30 event representation and matched 78-dimensional classical "
                "reservoir, migration outcomes depend on task and decoder, while completed BACE "
                "one-factor scans support a broad Floquet operating region."
            ),
            "archetype": "quantitative grid",
            "role": "decoder sensitivity and reservoir robustness",
            "backend": "Python/matplotlib only",
            "final_size_mm": [183, 137],
            "statistics": "mean ± sample s.d. across six paired seeds; test ROC-AUC",
            "reviewer_risk": (
                "Every top-row system uses the identical Q12/T30 event prefix; no system-specific "
                "Q/T selection is used. The one-factor g/W curves are not the pending two-dimensional "
                "interaction scan, and no test maximum is treated as selected."
            ),
        },
    }
    for q in Q_VALUES:
        contracts[f"fixed_Q{q}_same_width_T_curves"] = {
            "core_conclusion": (
                f"At fixed Q={q} and matched width={MATCHED_WIDTH[q]}, QRC and ESN are compared "
                "at every identical event-prefix length T without selecting or collapsing T."
            ),
            "archetype": "quantitative grid",
            "role": "primary fixed-capacity event-history evidence",
            "backend": "Python/matplotlib only",
            "final_size_mm": [183, 132],
            "statistics": "mean ± sample s.d. across six paired seeds; test ROC-AUC",
            "reviewer_risk": (
                "Increasing T adds chemical information as well as memory demand, so the curve "
                "is an effective event-history-capacity comparison, not a pure synthetic memory score."
            ),
        }
    (OUTPUT_ROOT / "figure_contracts.json").write_text(
        json.dumps(contracts, indent=2, sort_keys=True), encoding="utf-8"
    )
    qa = {
        "decoder_and_floquet_overview": qa_export(overview_base, overview_sizes),
        "source_data": {
            path.name: path.stat().st_size
            for path in OUTPUT_ROOT.glob("source_data_*.csv")
        },
        "excluded_incomplete_result": "pending two-dimensional g-by-W scan",
    }
    for q, (curve_base, curve_sizes) in curve_exports.items():
        qa[f"fixed_Q{q}_same_width_T_curves"] = qa_export(
            curve_base, curve_sizes
        )
    qa["passes"] = all(item["passes"] for key, item in qa.items() if isinstance(item, dict) and "passes" in item)
    (OUTPUT_ROOT / "figure_qa.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not qa["passes"]:
        raise RuntimeError(f"figure QA failed: {qa}")
    print(json.dumps(qa, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
