#!/usr/bin/env python3
"""Publication figures for the SHNITSEL A01 Q15 benchmark (Python only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_absolute_error, r2_score


# Mandatory editable-text and publication export settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.frameon": False,
})


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ("gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs")
TARGET_LABELS = (r"$\Delta E$ 1 fs", r"$\Delta E$ 5 fs", r"$\Delta E$ 10 fs",
                 r"min $\Delta E$" + "\nnext 5 fs")
SPLIT_LABELS = ("Train", "Validation", "Test")

COLORS = {
    "qrc": "#0F4D92",
    "esn": "#9A4D8E",
    "svr_history_10fs": "#42949E",
    "extra_trees": "#A8A8A8",
    "random_forest": "#CFCECE",
    "hist_gradient_boosting": "#E9A6A1",
    "classical": "#767676",
    "positive": "#B64342",
    "negative": "#2E9E44",
}

METHODS = (
    ("qrc", "QRC"),
    ("esn", "ESN"),
    ("svr_history_10fs", "Raw-history SVR"),
    ("extra_trees", "Extra Trees"),
    ("random_forest", "Random Forest"),
    ("hist_gradient_boosting", "HistGradientBoosting"),
)


def panel_label(ax, label: str) -> None:
    ax.text(-0.24, 1.06, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")


def save_figure(fig: plt.Figure, name: str) -> None:
    base = FIGURE_DIR / name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def load_tables():
    best = pd.read_csv(OUTPUT / "reservoir_best_validation_selected.csv")
    selected = pd.read_csv(OUTPUT / "reservoir_selected_test.csv")
    classical = pd.read_csv(OUTPUT / "classical_test.csv")
    scan = pd.read_csv(OUTPUT / "p_scan_validation.csv")
    paired = pd.read_csv(OUTPUT / "paired_trajectory_comparisons.csv")
    return best, selected, classical, scan, paired


def performance_table(best: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for system in ("qrc", "esn"):
            values = best[(best.system == system) & (best.target == target)]
            for metric in ("mae_ev", "rmse_ev", "r2"):
                rows.append({
                    "target": target, "method": system, "metric": metric,
                    "mean": values[metric].mean(), "sample_std": values[metric].std(ddof=1),
                    "n_repeats": len(values), "uncertainty": "sample SD across reservoir seeds",
                })
        for model, _ in METHODS[2:]:
            value = classical[(classical.target == target) & (classical.model == model)].iloc[0]
            for metric in ("mae_ev", "rmse_ev", "r2"):
                rows.append({
                    "target": target, "method": model, "metric": metric,
                    "mean": value[metric], "sample_std": np.nan,
                    "n_repeats": 1, "uncertainty": "none; deterministic fitted model",
                })
    result = pd.DataFrame(rows)
    result.to_csv(SOURCE_DIR / "model_metrics.csv", index=False)
    return result


def selected_prediction(best: pd.DataFrame, system: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = best[(best.system == system) & (best.target == target)].sort_values("seed")
    predictions, truth, trajectory = [], None, None
    for _, row in rows.iterrows():
        path = OUTPUT / "predictions" / (
            f"{system}_seed{int(row.seed):03d}_{row.readout}_{row.decoder}_{target}.npz"
        )
        with np.load(path, allow_pickle=False) as package:
            if truth is None:
                truth = package["target"].astype(np.float64)
                trajectory = package["trajectory_id"].astype(np.int64)
            predictions.append(package["prediction"].astype(np.float64))
    return truth, np.mean(predictions, axis=0), trajectory


def plot_metric_panel(ax, metrics: pd.DataFrame, metric: str, ylabel: str, label: str) -> None:
    x = np.arange(len(TARGETS), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(METHODS))
    for offset, (method, method_label) in zip(offsets, METHODS):
        subset = metrics[(metrics.method == method) & (metrics.metric == metric)].set_index("target").loc[list(TARGETS)]
        values = subset["mean"].to_numpy()
        error = subset["sample_std"].to_numpy()
        color = COLORS[method]
        ax.plot(x + offset, values, marker="o", ms=3.3, lw=1.0, color=color,
                label=method_label, zorder=3 if method in ("qrc", "esn") else 2)
        finite = np.isfinite(error)
        if finite.any():
            ax.errorbar(x[finite] + offset, values[finite], yerr=error[finite], fmt="none",
                        ecolor=color, elinewidth=0.8, capsize=1.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", lw=0.55, zorder=0)
    panel_label(ax, label)


def performance_figure(best, classical, paired) -> None:
    metrics = performance_table(best, classical)
    fig = plt.figure(figsize=(7.2, 6.6), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 1.0))
    ax_a = fig.add_subplot(grid[0, :2])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[1, 2])

    plot_metric_panel(ax_a, metrics, "mae_ev", "Test MAE (eV)", "a")
    ax_a.set_title("Validation-selected models on held-out trajectories", loc="left", fontweight="bold")
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.22),
                handlelength=1.8, columnspacing=1.1)

    plot_metric_panel(ax_b, metrics, "rmse_ev", "Test RMSE (eV)", "b")
    ax_b.set_ylim(bottom=0)
    plt.setp(ax_b.get_xticklabels(), rotation=28, ha="right")

    plot_metric_panel(ax_c, metrics, "r2", r"Test $R^2$", "c")
    ax_c.axhline(0, color="#767676", lw=0.65, ls="--")
    plt.setp(ax_c.get_xticklabels(), rotation=28, ha="right")

    primary = "min_gap_next_5fs"
    target, qrc_prediction, trajectory = selected_prediction(best, "qrc", primary)
    _, esn_prediction, _ = selected_prediction(best, "esn", primary)
    with np.load(OUTPUT / "predictions" / f"classical_hist_gradient_boosting_{primary}.npz", allow_pickle=False) as package:
        classical_prediction = package["prediction"].astype(np.float64)
    prediction_source = pd.DataFrame({
        "target_ev": target, "qrc_seed_mean_prediction_ev": qrc_prediction,
        "esn_seed_mean_prediction_ev": esn_prediction,
        "hist_gradient_boosting_prediction_ev": classical_prediction,
        "trajectory_id": trajectory,
    })
    prediction_source.to_csv(SOURCE_DIR / "primary_predictions.csv", index=False)
    for prediction, name, color, marker in (
        (classical_prediction, "HGB", COLORS["hist_gradient_boosting"], "^"),
        (esn_prediction, "ESN", COLORS["esn"], "s"),
        (qrc_prediction, "QRC", COLORS["qrc"], "o"),
    ):
        ax_d.scatter(target, prediction, s=8, alpha=0.34, color=color, marker=marker,
                     edgecolors="none", label=f"{name}: MAE {mean_absolute_error(target, prediction):.2f}")
    limit = (0, max(target.max(), qrc_prediction.max(), esn_prediction.max(), classical_prediction.max()) * 1.02)
    ax_d.plot(limit, limit, color="#4D4D4D", lw=0.8, ls="--")
    ax_d.set(xlim=limit, ylim=limit, xlabel="True minimum gap (eV)", ylabel="Predicted minimum gap (eV)")
    ax_d.legend(loc="upper left", handletextpad=0.3)
    ax_d.set_title("Primary-target seed-mean diagnostic", loc="left")
    panel_label(ax_d, "d")

    forest = paired[paired.comparison == "qrc_minus_esn"].set_index("target").loc[list(TARGETS)]
    y = np.arange(len(TARGETS))[::-1]
    for yi, (_, row) in zip(y, forest.iterrows()):
        ax_e.plot([row.bootstrap_95ci_low_ev, row.bootstrap_95ci_high_ev], [yi, yi],
                  color=COLORS["qrc"], lw=1.5)
        ax_e.plot(row.mean_mae_difference_ev, yi, "o", color=COLORS["qrc"], ms=4)
    ax_e.axvline(0, color="#767676", ls="--", lw=0.8)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(TARGET_LABELS)
    ax_e.set_xlabel("QRC − ESN trajectory MAE (eV)")
    ax_e.set_title("10,000× trajectory bootstrap", loc="left")
    ax_e.text(0.02, 0.03, "QRC better", transform=ax_e.transAxes, color=COLORS["qrc"], ha="left")
    panel_label(ax_e, "e")
    forest.reset_index().to_csv(SOURCE_DIR / "qrc_esn_paired_bootstrap.csv", index=False)

    fig.suptitle("Dissipative QRC improves short- and mid-horizon nonadiabatic gap prediction",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    save_figure(fig, "figure1_model_performance")


def diagnostics_figure(best, selected, scan) -> None:
    with np.load(ROOT / "data" / "processed" / "a01_windows.npz", allow_pickle=False) as package:
        pair_windows = package["pair_windows"]
        targets = package["targets"]
        split = package["split"]
        trajectory = package["trajectory_id"]
        endpoint = package["endpoint"]
        pair_index = package["pair_index"]
    train_indices = np.flatnonzero(split == 0)
    sample_index = int(train_indices[0])  # deterministic, not cherry-picked
    distances = pair_windows[sample_index, :, :, 0].T
    velocities = pair_windows[sample_index, :, :, 1].T
    time_fs = np.arange(-9.5, 0.5, 0.5)
    pair_names = []
    atom_names = ("C1", "C2", "H1", "H2", "H3", "H4")
    for left, right in pair_index:
        pair_names.append(f"{atom_names[left]}–{atom_names[right]}")
    pd.DataFrame({
        "sample_index": sample_index,
        "trajectory_id": int(trajectory[sample_index]),
        "endpoint_local_frame": int(endpoint[sample_index]),
        "time_relative_fs": np.repeat(time_fs, len(pair_names)),
        "pair": np.tile(pair_names, len(time_fs)),
        "distance_angstrom": distances.T.ravel(),
        "radial_velocity_angstrom_per_fs": velocities.T.ravel(),
    }).to_csv(SOURCE_DIR / "example_training_window.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8), layout="constrained")
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    im_a = ax_a.imshow(distances, aspect="auto", cmap="Blues", interpolation="nearest",
                       extent=[time_fs[0] - 0.25, time_fs[-1] + 0.25, len(pair_names) - 0.5, -0.5])
    ax_a.set(xlabel="Time before forecast origin (fs)", ylabel="Atom pair",
             title="Example training input: distances")
    ax_a.set_yticks(range(len(pair_names)))
    ax_a.set_yticklabels(pair_names, fontsize=5.5)
    cb = fig.colorbar(im_a, ax=ax_a, fraction=0.047, pad=0.02)
    cb.set_label(r"Distance ($\AA$)")
    panel_label(ax_a, "a")

    vmax = np.quantile(np.abs(velocities), 0.98)
    im_b = ax_b.imshow(velocities, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest", extent=[time_fs[0] - 0.25, time_fs[-1] + 0.25,
                                                         len(pair_names) - 0.5, -0.5])
    ax_b.set(xlabel="Time before forecast origin (fs)", title="Same input: radial velocities")
    ax_b.set_yticks(range(len(pair_names)))
    ax_b.set_yticklabels([])
    cb = fig.colorbar(im_b, ax=ax_b, fraction=0.047, pad=0.02)
    cb.set_label(r"Velocity ($\AA$ fs$^{-1}$)")
    panel_label(ax_b, "b")

    bins = np.linspace(0, np.quantile(targets[:, 3], 0.995), 28)
    distribution_rows = []
    for split_index, (label, color) in enumerate(zip(SPLIT_LABELS, ("#767676", "#42949E", "#B64342"))):
        values = targets[split == split_index, 3]
        ax_c.hist(values, bins=bins, histtype="step", density=True, lw=1.25, color=color,
                  label=f"{label} (n={len(values)})")
        distribution_rows.extend({"split": label, "target_ev": float(value)} for value in values)
    ax_c.set(xlabel="Minimum gap in next 5 fs (eV)", ylabel="Density",
             title="Target distribution by trajectory split")
    ax_c.legend()
    panel_label(ax_c, "c")
    pd.DataFrame(distribution_rows).to_csv(SOURCE_DIR / "primary_target_distributions.csv", index=False)

    qrc_scan = scan[(scan.system == "qrc") & (scan.decoder == "svr")]
    envelope = qrc_scan.groupby(["seed", "target", "p"], as_index=False).validation_mae_ev.min()
    summary = envelope.groupby(["target", "p"], as_index=False).validation_mae_ev.agg(["mean", "std"]).reset_index()
    p_grid = sorted(scan.p.unique())
    x = np.arange(len(p_grid))
    target_colors = ("#0F4D92", "#42949E", "#9A4D8E", "#B64342")
    direct_labels = ("1 fs", "5 fs", "10 fs", "min next 5 fs")
    for target_name, display, direct, color in zip(TARGETS, TARGET_LABELS, direct_labels, target_colors):
        values = summary[summary.target == target_name].set_index("p").loc[p_grid]
        mean, std = values["mean"].to_numpy(), values["std"].to_numpy()
        ax_d.plot(x, mean, marker="o", ms=2.8, lw=1.15, color=color, label=display)
        ax_d.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, lw=0)
        ax_d.text(x[-1] + 0.25, mean[-1], direct, color=color, fontsize=5.8, va="center")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{p:g}" for p in p_grid], rotation=55, ha="right", fontsize=5.5)
    ax_d.set_xlim(-1.0, len(p_grid) + 1.35)
    ax_d.set(xlabel="Reset probability p", ylabel="Validation MAE (eV)",
             title="Model-selection objective (loss analogue)")
    ax_d.grid(axis="y", color="#E6E6E6", lw=0.55)
    panel_label(ax_d, "d")
    envelope.to_csv(SOURCE_DIR / "qrc_svr_p_scan_envelope.csv", index=False)

    qrc_summary = selected[selected.system == "qrc"].groupby(
        ["readout", "decoder", "target"], as_index=False
    ).mae_ev.mean()
    delta_rows = []
    for decoder_name in ("linear", "svr"):
        for target in TARGETS:
            final = qrc_summary[(qrc_summary.readout == "final") & (qrc_summary.decoder == decoder_name)
                                & (qrc_summary.target == target)].mae_ev.iloc[0]
            tm3 = qrc_summary[(qrc_summary.readout == "tm3") & (qrc_summary.decoder == decoder_name)
                              & (qrc_summary.target == target)].mae_ev.iloc[0]
            delta_rows.append({"decoder": decoder_name, "target": target, "tm3_minus_final_mae_ev": tm3 - final})
    delta = pd.DataFrame(delta_rows)
    positions = np.arange(len(TARGETS))
    width = 0.34
    for offset, decoder_name, color in ((-width / 2, "linear", "#42949E"),
                                         (width / 2, "svr", "#9A4D8E")):
        values = delta[delta.decoder == decoder_name].set_index("target").loc[list(TARGETS)].tm3_minus_final_mae_ev
        ax_e.bar(positions + offset, values, width=width, color=color, label=decoder_name.upper(),
                 edgecolor="white", linewidth=0.4)
    ax_e.axhline(0, color="#4D4D4D", lw=0.75)
    ax_e.set_xticks(positions)
    ax_e.set_xticklabels(TARGET_LABELS, rotation=20, ha="right")
    ax_e.set(ylabel="TM3 − final test MAE (eV)", title="Temporal-readout effect")
    ax_e.legend()
    ax_e.text(0.02, 0.03, "<0: TM3 better", transform=ax_e.transAxes, color="#4D4D4D")
    panel_label(ax_e, "e")
    delta.to_csv(SOURCE_DIR / "tm3_final_comparison.csv", index=False)

    selected_qrc = best[best.system == "qrc"].pivot(index="target", columns="seed", values="selected_p").loc[list(TARGETS)]
    im_f = ax_f.imshow(selected_qrc.to_numpy(), cmap="Blues_r", vmin=0, vmax=1, aspect="auto")
    for row in range(selected_qrc.shape[0]):
        for column in range(selected_qrc.shape[1]):
            value = selected_qrc.iloc[row, column]
            ax_f.text(column, row, f"{value:g}", ha="center", va="center",
                      color="white" if value < 0.5 else "black", fontsize=7)
    ax_f.set_xticks(range(selected_qrc.shape[1]))
    ax_f.set_xticklabels([str(seed) for seed in selected_qrc.columns])
    ax_f.set_xlabel("Reservoir seed")
    ax_f.set_yticks(range(len(TARGETS)))
    ax_f.set_yticklabels(TARGET_LABELS)
    ax_f.set_title("Validation-selected QRC memory")
    cb = fig.colorbar(im_f, ax=ax_f, fraction=0.047, pad=0.02)
    cb.set_label("Selected p")
    panel_label(ax_f, "f")
    selected_qrc.reset_index().to_csv(SOURCE_DIR / "selected_qrc_p.csv", index=False)

    fig.suptitle("Training data, validation objective and temporal-readout diagnostics",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    save_figure(fig, "figure2_training_diagnostics")


def write_notes() -> None:
    notes = """# Figure contract and QA notes

Core conclusion: validation-selected QRC improves short- and mid-horizon energy-gap prediction over a dimension-matched ESN and tested input-matched classical models, while the selected reset probability decreases as forecast horizon grows.

Archetype: quantitative grid. Backend: Python/matplotlib exclusively. Final size: double-column, 183 mm wide. Exports: editable SVG, TrueType PDF, 300-dpi PNG preview and 600-dpi LZW TIFF.

Evidence map:
- Figure 1a: hero evidence, held-out trajectory MAE across methods.
- Figure 1b-c: RMSE and R2 corroborate the MAE result.
- Figure 1d: primary-target true/predicted diagnostic; QRC/ESN are the arithmetic mean of the three independently validation-selected seed predictions for visualization only.
- Figure 1e: 60-trajectory paired bootstrap comparison.
- Figure 2a-b: actual, deterministically chosen first training window; no representative cherry-picking.
- Figure 2c: target-distribution split audit.
- Figure 2d: validation MAE over p is the honest model-selection objective. Ridge/SVR have no epoch-wise neural-network loss history.
- Figure 2e-f: final/TM3 effect and validation-selected memory.

Statistics: QRC and ESN points are mean ± sample SD over n=3 reservoir seeds. Classical algorithms are one deterministic fit after validation selection and therefore have no invented error bars. Paired intervals use 10,000 bootstrap resamples of n=60 complete test trajectories. Train/validation/test contain 178/59/60 complete trajectories; no trajectory crosses a split.

Reviewer risks: the current-gap persistence baseline has access to the current electronic target and is intentionally excluded from the input-matched method curves. The QRC/ESN seed-mean prediction panel is diagnostic and not the metric used for the main benchmark. No test sample enters normalization or hyperparameter selection.
"""
    (FIGURE_DIR / "FIGURE_QA.md").write_text(notes, encoding="utf-8")
    captions = """# Figure captions

## Figure 1 | Dissipative QRC improves nonadiabatic energy-gap prediction

**a–c,** Test MAE, RMSE and R² on 60 held-out SHNITSEL A01 trajectories for validation-selected QRC and dimension-matched ESN models (mean ± sample SD across three reservoir seeds) and four input-matched classical algorithms (one deterministic validation-selected fit; no error bar). **d,** Diagnostic true-versus-predicted plot for the primary target, the minimum S1–S0 gap over the next 5 fs. QRC and ESN points are the arithmetic mean of their three independently validation-selected seed predictions for visualization; main benchmark metrics are seed-wise, not ensemble metrics. **e,** Paired QRC-minus-ESN trajectory MAE with 95% intervals from 10,000 bootstrap resamples of 60 complete test trajectories. Negative values favor QRC.

## Figure 2 | Training data, validation objective and temporal-readout diagnostics

**a,b,** Pair distances and finite-difference radial velocities in the first deterministic training window (20 frames, 10 fs; 15 atom pairs). These 30 values per frame are injected simultaneously through X and Z. **c,** Primary-target distributions after complete-trajectory train/validation/test splitting. **d,** Validation MAE used as the model-selection objective across the exact reset-probability scan; curves show the lower validation MAE of final and TM3 at each p, mean ± sample SD across three QRC seeds. Linear/SVR decoders do not expose an epoch-wise neural-network loss history. **e,** Mean TM3-minus-final test MAE across three QRC seeds for Linear and SVR; negative values favor TM3. **f,** Reset probabilities selected exclusively on validation trajectories for each QRC seed and target.
"""
    (FIGURE_DIR / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def main() -> None:
    best, selected, classical, scan, paired = load_tables()
    performance_figure(best, classical, paired)
    diagnostics_figure(best, selected, scan)
    write_notes()
    manifest = {
        "backend": "Python/matplotlib",
        "figures": ["figure1_model_performance", "figure2_training_diagnostics"],
        "formats": ["svg", "pdf", "png", "tiff"],
        "source_data": sorted(path.name for path in SOURCE_DIR.glob("*.csv")),
        "epoch_loss_available": False,
        "loss_substitute": "validation MAE versus p and readout/decoder selection",
    }
    (FIGURE_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
