"""Summarize and plot the frozen coherence/interaction ablation.

The script is deliberately read-only with respect to feature banks and model
outputs.  It writes derived statistics, panel-level source data, a compact
machine-readable report, and publication exports for the mechanism figure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from protocol import P_VALUES, ROOT


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SOURCE_DATA = FIGURES / "source_data"
REPORTS = ROOT / "reports"

COLORS = {0.0: "#214D8C", 0.1: "#D9822B"}
P_LABELS = {0.0: "Closed, p=0", 0.1: "Dissipative, p=0.10"}
TARGET_LABELS = {
    "gap_1fs": "Gap, 1 fs",
    "gap_5fs": "Gap, 5 fs",
    "gap_10fs": "Gap, 10 fs",
    "min_gap_next_5fs": "Minimum gap, next 5 fs",
}
INTERACTION_VARIANTS = {
    "interaction_off": 0.0,
    "interaction_half": 0.5,
    "coherent": 1.0,
}
DEPHASING_VARIANTS = {
    "coherent": 0.0,
    "dephase_half": 0.5,
    "event_dephase": 1.0,
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
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = samples.mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _summary(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    rows: list[dict] = []
    for key, group in frame.groupby(groups, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value].to_numpy(dtype=np.float64)
        stable_key = "|".join(str(item) for item in key).encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(stable_key).digest()[:4], "little")
        mean, low, high = bootstrap_mean_ci(
            values, seed=(20260829 + offset) % (2**32 - 1)
        )
        rows.append(
            {
                **dict(zip(groups, key)),
                "metric": value,
                "mean": mean,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "ci_low": low,
                "ci_high": high,
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _interaction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame.variant.isin(INTERACTION_VARIANTS)].copy()
    selected["control_value"] = selected.variant.map(INTERACTION_VARIANTS).astype(float)
    selected["control"] = "interaction scale"
    return selected


def _dephasing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame.variant.isin(DEPHASING_VARIANTS)].copy()
    selected["control_value"] = selected.variant.map(DEPHASING_VARIANTS).astype(float)
    selected["control"] = "event dephasing"
    return selected


def summarize_bace(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = pd.concat(
        [_interaction_frame(metrics), _dephasing_frame(metrics)], ignore_index=True
    )
    performance = _summary(
        controls,
        ["control", "control_value", "variant", "p"],
        "test_auc",
    )
    correlation = _summary(
        controls,
        ["control", "control_value", "variant", "p"],
        "connected_zz_rms",
    )
    return performance, correlation


def hopping_seed_ratios(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return a target-balanced MAE ratio to the coherent endpoint.

    Each target is first divided by its coherent MAE at the same seed and p;
    targets are then averaged within a seed.  Thus the plotted uncertainty is
    across reservoir seeds rather than treating four targets as independent.
    """
    keys = ["seed", "p", "readout", "decoder", "target"]
    baseline = (
        metrics[metrics.variant == "coherent"][keys + ["mae_ev"]]
        .rename(columns={"mae_ev": "coherent_mae_ev"})
        .drop_duplicates(keys)
    )
    merged = metrics.merge(baseline, on=keys, how="left", validate="many_to_one")
    if merged.coherent_mae_ev.isna().any():
        raise RuntimeError("missing coherent hopping baseline")
    merged["mae_ratio_to_coherent"] = merged.mae_ev / merged.coherent_mae_ev
    return (
        merged.groupby(["variant", "p", "readout", "decoder", "seed"], as_index=False)
        .agg(
            mae_ratio_to_coherent=("mae_ratio_to_coherent", "mean"),
            mean_mae_ev=("mae_ev", "mean"),
        )
    )


def hopping_trajectory_ratios(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build paired, target-balanced ratios at the physical-trajectory level.

    Errors are first averaged within each held-out trajectory.  Every variant is
    then divided by the coherent error for the same reservoir seed, reset rate,
    target, and physical trajectory.  Averaging over targets leaves one paired
    value per reservoir seed and trajectory, which is the unit used by the
    crossed bootstrap below.
    """
    grouped = (
        predictions.groupby(
            ["variant", "seed", "p", "readout", "decoder", "target", "trajectory_id"],
            as_index=False,
        )
        .agg(mae_ev=("absolute_error_ev", "mean"))
    )
    keys = ["seed", "p", "readout", "decoder", "target", "trajectory_id"]
    baseline = (
        grouped[grouped.variant == "coherent"][keys + ["mae_ev"]]
        .rename(columns={"mae_ev": "coherent_mae_ev"})
        .drop_duplicates(keys)
    )
    merged = grouped.merge(baseline, on=keys, how="left", validate="many_to_one")
    if merged.coherent_mae_ev.isna().any():
        raise RuntimeError("missing coherent trajectory-level hopping baseline")
    if (merged.coherent_mae_ev <= 0).any():
        raise RuntimeError("non-positive coherent trajectory-level hopping MAE")
    merged["mae_ratio_to_coherent"] = merged.mae_ev / merged.coherent_mae_ev
    return (
        merged.groupby(
            ["variant", "p", "readout", "decoder", "seed", "trajectory_id"],
            as_index=False,
        )
        .agg(mae_ratio_to_coherent=("mae_ratio_to_coherent", "mean"))
    )


def crossed_seed_trajectory_summary(
    trajectory_level: pd.DataFrame,
    groups: list[str],
    value: str,
    *,
    draws: int = 20_000,
) -> pd.DataFrame:
    """Percentile CI from independent resampling of seeds and trajectories.

    The point estimate is the equally weighted mean of the complete
    seed-by-trajectory matrix.  Each bootstrap draw independently resamples
    reservoir seeds and held-out physical trajectories, preserving pairing
    between every intervention and its coherent reference before this stage.
    """
    rows: list[dict] = []
    for key, group in trajectory_level.groupby(groups, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        matrix = group.pivot(index="seed", columns="trajectory_id", values=value)
        if matrix.isna().any().any():
            raise RuntimeError(f"incomplete seed-by-trajectory matrix for {key}")
        values = matrix.to_numpy(dtype=np.float64)
        stable_key = "|".join(str(item) for item in key).encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(stable_key).digest()[:4], "little")
        rng = np.random.default_rng((20260829 + offset) % (2**32 - 1))
        seed_draws = rng.integers(0, values.shape[0], size=(draws, values.shape[0]))
        trajectory_draws = rng.integers(
            0, values.shape[1], size=(draws, values.shape[1])
        )
        boot = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            boot[draw] = values[np.ix_(seed_draws[draw], trajectory_draws[draw])].mean()
        rows.append(
            {
                **dict(zip(groups, key)),
                "metric": value,
                "mean": float(values.mean()),
                "sd": float(np.std(values.mean(axis=1), ddof=1))
                if values.shape[0] > 1
                else np.nan,
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "n_seeds": int(values.shape[0]),
                "n_trajectories": int(values.shape[1]),
                "bootstrap": "crossed_seed_x_trajectory",
            }
        )
    return pd.DataFrame(rows)


def summarize_hopping(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_level = hopping_seed_ratios(metrics)
    trajectory_level = hopping_trajectory_ratios(predictions)
    controls = pd.concat(
        [_interaction_frame(trajectory_level), _dephasing_frame(trajectory_level)],
        ignore_index=True,
    )
    ratio_summary = crossed_seed_trajectory_summary(
        controls,
        ["control", "control_value", "variant", "p", "readout", "decoder"],
        "mae_ratio_to_coherent",
    )
    target_summary = _summary(
        pd.concat(
            [_interaction_frame(metrics), _dephasing_frame(metrics)], ignore_index=True
        ),
        ["control", "control_value", "variant", "p", "readout", "decoder", "target"],
        "mae_ev",
    )
    return seed_level, trajectory_level, ratio_summary, target_summary


def summarize_hopping_correlation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse target/readout duplicates before summarizing feature geometry."""
    grouped = (
        metrics.groupby(["variant", "p", "seed"], as_index=False)
        .agg(
            connected_zz_rms=("connected_zz_rms", "first"),
            connected_zz_rms_min=("connected_zz_rms", "min"),
            connected_zz_rms_max=("connected_zz_rms", "max"),
        )
    )
    if not np.allclose(
        grouped.connected_zz_rms_min,
        grouped.connected_zz_rms_max,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise RuntimeError("hopping connected-ZZ diagnostic differs across duplicate rows")
    controls = pd.concat(
        [_interaction_frame(grouped), _dephasing_frame(grouped)], ignore_index=True
    )
    return _summary(
        controls,
        ["control", "control_value", "variant", "p"],
        "connected_zz_rms",
    )


def _draw_performance(
    ax: plt.Axes,
    summary: pd.DataFrame,
    control: str,
    ylabel: str,
    *,
    chance: float | None = None,
) -> None:
    frame = summary[summary.control == control]
    for p in P_VALUES:
        sub = frame[np.isclose(frame.p, p)].sort_values("control_value")
        x = sub.control_value.to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        low = sub.ci_low.to_numpy(dtype=float)
        high = sub.ci_high.to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=COLORS[p],
            marker="o" if p == 0 else "s",
            ms=4.0,
            lw=1.45,
            label=P_LABELS[p],
            zorder=3,
        )
        ax.fill_between(x, low, high, color=COLORS[p], alpha=0.14, linewidth=0)
    if chance is not None:
        ax.axhline(chance, color="#999999", lw=0.75, ls="--", zorder=0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(axis="y", color="#E7E7E7", lw=0.55)
    ax.set_ylabel(ylabel)


def _draw_connected(
    ax: plt.Axes,
    summaries: tuple[tuple[str, str, pd.DataFrame], ...],
    control: str,
    *,
    p: float = 0.0,
) -> None:
    for label, color, summary in summaries:
        sub = summary[
            (summary.control == control) & np.isclose(summary.p, p)
        ].sort_values("control_value")
        x = sub.control_value.to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        low = np.maximum(sub.ci_low.to_numpy(dtype=float), np.finfo(float).tiny)
        high = sub.ci_high.to_numpy(dtype=float)
        ax.plot(x, y, color=color, marker="D", ms=3.5, lw=1.35, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(axis="y", color="#E7E7E7", lw=0.55, which="both")
    ax.set_ylabel(r"Connected-$ZZ$ RMS")


def plot_mechanism(
    bace_performance: pd.DataFrame,
    bace_correlation: pd.DataFrame,
    hopping_ratio: pd.DataFrame,
    hopping_correlation: pd.DataFrame,
) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.45))

    _draw_performance(
        axes[0, 0], bace_performance, "interaction scale", "BACE test ROC-AUC"
    )
    axes[0, 0].set_xlabel(r"Interaction scale, $s_J$")
    axes[0, 0].set_title("Drug-molecule event stream")

    _draw_performance(
        axes[0, 1],
        hopping_ratio,
        "interaction scale",
        "Target-balanced MAE ratio",
        chance=1.0,
    )
    axes[0, 1].set_xlabel(r"Interaction scale, $s_J$")
    axes[0, 1].set_title("Surface-hopping stream")

    correlation_summaries = (
        ("BACE", "#6B5AA6", bace_correlation),
        ("Hopping", "#4F7C76", hopping_correlation),
    )
    _draw_connected(axes[0, 2], correlation_summaries, "interaction scale", p=0.0)
    axes[0, 2].set_xlabel(r"Interaction scale, $s_J$")
    axes[0, 2].set_title("Closed-channel factorization check")

    _draw_performance(
        axes[1, 0], bace_performance, "event dephasing", "BACE test ROC-AUC"
    )
    axes[1, 0].set_xlabel(r"Event dephasing, $\lambda$")

    _draw_performance(
        axes[1, 1],
        hopping_ratio,
        "event dephasing",
        "Target-balanced MAE ratio",
        chance=1.0,
    )
    axes[1, 1].set_xlabel(r"Event dephasing, $\lambda$")

    _draw_connected(axes[1, 2], correlation_summaries, "event dephasing", p=0.0)
    axes[1, 2].set_xlabel(r"Event dephasing, $\lambda$")

    for index, ax in enumerate(axes.flat):
        panel_label(ax, chr(ord("a") + index))

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[p],
            marker="o" if p == 0 else "s",
            lw=1.45,
            ms=4,
            label=P_LABELS[p],
        )
        for p in P_VALUES
    ]
    handles.extend(
        [
            Line2D([0], [0], color=color, marker="D", lw=1.35, ms=3.5,
                   label=rf"{label}, closed connected-$ZZ$")
            for label, color, _ in correlation_summaries
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(
        left=0.09, right=0.995, top=0.93, bottom=0.16, wspace=0.38, hspace=0.43
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "fig_mechanism_coherence_interaction"
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


def _paired_delta_summary(
    metrics: pd.DataFrame,
    value: str,
    *,
    higher_is_better: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for p in P_VALUES:
        coherent = metrics[
            (metrics.variant == "coherent") & np.isclose(metrics.p, p)
        ][["seed", value]].rename(columns={value: "coherent"})
        for variant in ["interaction_half", "interaction_off", "dephase_half", "event_dephase"]:
            current = metrics[
                (metrics.variant == variant) & np.isclose(metrics.p, p)
            ][["seed", value]].rename(columns={value: "current"})
            paired = coherent.merge(current, on="seed", validate="one_to_one")
            delta = paired.current - paired.coherent
            if not higher_is_better:
                delta = -delta
            estimate, low, high = bootstrap_mean_ci(
                delta.to_numpy(), seed=20260829 + int(1000 * p) + len(rows)
            )
            rows.append(
                {
                    "variant": variant,
                    "p": p,
                    "oriented_delta_vs_coherent": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_seeds": len(delta),
                    "positive_means_improvement": True,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    bace_metrics: pd.DataFrame,
    hopping_selected: pd.DataFrame,
    hopping_trajectory_level: pd.DataFrame,
    bace_performance: pd.DataFrame,
    bace_correlation: pd.DataFrame,
    hopping_seed_level: pd.DataFrame,
    hopping_ratio: pd.DataFrame,
    hopping_targets: pd.DataFrame,
    hopping_correlation: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    bace_performance.to_csv(SOURCE_DATA / "mechanism_bace_auc.csv", index=False)
    bace_correlation.to_csv(SOURCE_DATA / "mechanism_bace_connected_zz.csv", index=False)
    hopping_seed_level.to_csv(SOURCE_DATA / "mechanism_hopping_seed_ratios.csv", index=False)
    hopping_trajectory_level.to_csv(
        SOURCE_DATA / "mechanism_hopping_trajectory_ratios.csv", index=False
    )
    hopping_ratio.to_csv(SOURCE_DATA / "mechanism_hopping_ratio_summary.csv", index=False)
    hopping_targets.to_csv(SOURCE_DATA / "mechanism_hopping_target_mae.csv", index=False)
    hopping_correlation.to_csv(
        SOURCE_DATA / "mechanism_hopping_connected_zz.csv", index=False
    )

    bace_deltas = _paired_delta_summary(
        bace_metrics, "test_auc", higher_is_better=True
    )
    bace_deltas.to_csv(REPORTS / "bace_paired_deltas.csv", index=False)

    interaction_j0 = bace_correlation[
        (bace_correlation.control == "interaction scale")
        & np.isclose(bace_correlation.control_value, 0.0)
        & np.isclose(bace_correlation.p, 0.0)
    ].iloc[0]
    hopping_interaction_j0 = hopping_correlation[
        (hopping_correlation.control == "interaction scale")
        & np.isclose(hopping_correlation.control_value, 0.0)
        & np.isclose(hopping_correlation.p, 0.0)
    ].iloc[0]
    payload = {
        "protocol": {
            "tasks": ["BACE", "SHNITSEL_A01"],
            "p_values": list(P_VALUES),
            "decoder": "RBF-SVR",
            "observables": "Z+all_ZZ",
            "selection": (
                "validation selects only RBF-SVR hyperparameters; p and final/TM3 "
                "remain complete fixed axes"
            ),
        },
        "j0_closed_connected_zz": {
            "BACE": {
                "mean_rms": float(interaction_j0["mean"]),
                "bootstrap_95ci": [
                    float(interaction_j0.ci_low),
                    float(interaction_j0.ci_high),
                ],
            },
            "hopping": {
                "mean_rms": float(hopping_interaction_j0["mean"]),
                "bootstrap_95ci": [
                    float(hopping_interaction_j0.ci_low),
                    float(hopping_interaction_j0.ci_high),
                ],
            },
            "interpretation": (
                "Numerical factorization check. J=0 leaves the closed state as a "
                "product under single-qubit injection and dynamics; it does not imply "
                "chance-level prediction with an RBF-SVR decoder."
            ),
        },
        "claim_guardrails": [
            "Reset dissipation p and event-wise dephasing lambda are distinct interventions.",
            "Connected ZZ is a covariance diagnostic, not an entanglement witness.",
            "Hopping target ratios are paired and target-balanced within each reservoir seed and physical trajectory.",
            "Hopping confidence intervals use crossed seed-by-trajectory bootstrap resampling.",
            "All claims must be conditioned on the frozen Q, T, W, g and decoder protocol.",
        ],
    }
    (REPORTS / "mechanism_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# Coherence and interaction ablation report",
        "",
        "This report is generated from frozen held-out predictions. The manuscript "
        "must use the numeric tables rather than infer values from the figure.",
        "",
        "## Closed J=0 factorization check",
        "",
        f"Mean connected-ZZ RMS: {float(interaction_j0['mean']):.6g} "
        f"(bootstrap 95% CI {float(interaction_j0.ci_low):.6g} to "
        f"{float(interaction_j0.ci_high):.6g}).",
        f"Hopping mean connected-ZZ RMS: {float(hopping_interaction_j0['mean']):.6g} "
        f"(bootstrap 95% CI {float(hopping_interaction_j0.ci_low):.6g} to "
        f"{float(hopping_interaction_j0.ci_high):.6g}).",
        "",
        "At J=0 the closed circuit is a tensor product of one-qubit channels. "
        "The RBF-SVR can still learn nonlinear functions of the resulting local "
        "features, so reduced connected correlations do not mathematically require "
        "chance-level task performance.",
        "",
        "## Statistical interpretation",
        "",
        "BACE error bands are percentile bootstrap confidence intervals of the seed "
        "mean. Hopping MAE is normalized target by target to the paired coherent "
        "endpoint within each reservoir seed and physical trajectory; intervals "
        "then use independent crossed resampling of reservoir seeds and held-out "
        "physical trajectories.",
    ]
    (REPORTS / "MECHANISM_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    bace_path = RESULTS / "bace_metrics_by_seed.csv"
    hopping_metrics_path = RESULTS / "hopping_metrics_by_seed.csv"
    hopping_predictions_path = RESULTS / "hopping_predictions_all_fixed_axes.csv"
    if not all(
        path.exists()
        for path in (
            bace_path,
            hopping_metrics_path,
            hopping_predictions_path,
        )
    ):
        raise FileNotFoundError(
            "Run `python run_ablation.py evaluate --task all` before plotting."
        )
    bace = pd.read_csv(bace_path)
    hopping_metrics = pd.read_csv(hopping_metrics_path)
    hopping_predictions = pd.read_csv(hopping_predictions_path)
    bace_performance, bace_correlation = summarize_bace(bace)
    (
        hopping_seed_level,
        hopping_trajectory_level,
        hopping_ratio,
        hopping_targets,
    ) = summarize_hopping(hopping_metrics, hopping_predictions)
    hopping_correlation = summarize_hopping_correlation(hopping_metrics)
    fixed_hopping_ratio = hopping_ratio[
        (hopping_ratio.readout == "final")
        & (hopping_ratio.decoder == "RBF-SVR")
    ].copy()
    write_report(
        bace,
        hopping_metrics,
        hopping_trajectory_level,
        bace_performance,
        bace_correlation,
        hopping_seed_level,
        hopping_ratio,
        hopping_targets,
        hopping_correlation,
    )
    plot_mechanism(
        bace_performance, bace_correlation, fixed_hopping_ratio, hopping_correlation
    )
    print(FIGURES / "fig_mechanism_coherence_interaction.pdf")


if __name__ == "__main__":
    main()
