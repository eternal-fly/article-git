from __future__ import annotations

from common import *
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5,
    "axes.labelsize": 8., "xtick.labelsize": 7., "ytick.labelsize": 7., "axes.linewidth": .8,
    "axes.spines.top": True, "axes.spines.right": True, "legend.frameon": True,
    "legend.fancybox": False, "legend.framealpha": 1., "legend.edgecolor": "#bbbbbb",
    "pdf.fonttype": 42, "svg.fonttype": "none"})
COLORS = {"DTC-QRC": "#355c91", "FP-CVR": "#cb7357", "ESN": "#538f88",
    "Selected-QRC": "#355c91", "QRC-p1": "#8b72a6", "Original-ESN": "#84b2ac",
    "Tuned-ESN": "#337c70", "Direct-20": "#c68a3d", "ExtraTrees": "#777777"}
TARGETS = ["gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs"]
TITLES = ["Gap, 1 fs", "Gap, 5 fs", "Gap, 10 fs", "Minimum gap, next 5 fs"]


def letter(ax, s):
    ax.text(-.14, 1.05, s, transform=ax.transAxes, weight="bold", fontsize=10)


def finish(fig, name, caption):
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_axisbelow(True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    hidden_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo-1e-10 <= tick.get_loc() <= hi+1e-10:
                    hidden_ticks.update((tick.label1, tick.label2))
    for artist in fig.findobj(mpl.text.Text):
        if artist not in hidden_ticks and artist.get_visible() and artist.get_text().strip():
            bounds = artist.get_window_extent(renderer)
            if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width+1 or bounds.y1 > height+1:
                outside.append(artist.get_text())
    write_json(OUT / f"{name}_qa.json", dict(
        canvas_inches=list(fig.get_size_inches()), text_outside_canvas=outside,
        full_frames=all(s.get_visible() for ax in fig.axes for s in ax.spines.values()),
        backend="Python/matplotlib", png_dpi=300, pdf_fonttype=42, svg_text="live"))
    if outside:
        raise RuntimeError(f"Text outside {name}: {outside}")
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, facecolor="white")
    (OUT / f"{name}_caption.txt").write_text(caption + "\n")
    plt.close(fig)


def probes():
    data = pd.read_csv(HERE / "probes/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.2))
    for i, (ax, family, title) in enumerate(zip(axes, ("delayed_motif", "distant_pair"), ("Delayed motif", "Distant event pair"))):
        for method, marker in (("DTC-QRC", "o"), ("FP-CVR", "s"), ("ESN", "^")):
            block = data[(data.family == family) & (data.method == method)].sort_values("lag")
            x, y, sd = block.lag.to_numpy(), block.auc_mean.to_numpy(), block.auc_sd.to_numpy()
            ax.plot(x, y, color=COLORS[method], marker=marker, ms=3.8, lw=1.3, label=method)
            ax.fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.12, lw=0)
        ax.axhline(.5, color="#888888", ls="--", lw=.8)
        ax.set(title=title, xlabel="Event separation", ylabel="Test ROC-AUC", ylim=(.4, .9))
        ax.set_xticks(LAGS)
        ax.grid(axis="y", alpha=.22)
        letter(ax, "ab"[i])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .015))
    fig.subplots_adjust(left=.085, right=.975, bottom=.23, top=.88, wspace=.3)
    finish(fig, "01_unique_background_probes", "Q12/T20/p0.10/g0.84/W0.50/L3; final Z+ZZ and the original affine decoder. Each condition has 512 unique complete inputs in 128 counterfactual groups. Original 70/15/15 group splits, six paired seeds. Curves and bands: mean and sample SD. Test/background templates do not overlap training or validation. Pointwise and seven-lag simultaneous intervals are supplied separately. Background diversity changed as well as overlap, so comparison with old results is not a pure split-only intervention.")


def history():
    data = pd.read_csv(HERE / "hopping/paired_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.15), sharey=True)
    for i, (ax, readout) in enumerate(zip(axes, ("final", "tm3"))):
        block = data[(data.readout == readout) & (data.decoder == "rbf_svr") & (data.baseline == "QRC-p1")].set_index("target").loc[TARGETS]
        x = block.delta_mae.to_numpy()
        ax.errorbar(x, np.arange(4), xerr=np.array([x-block.ci_low.to_numpy(), block.ci_high.to_numpy()-x]),
                    fmt="o", capsize=3, color=COLORS["DTC-QRC"], ms=4, lw=1.2)
        ax.axvline(0., color="#888888", ls="--", lw=.9)
        ax.set_yticks(np.arange(4), ["Gap 1 fs", "Gap 5 fs", "Gap 10 fs", "Next-5-fs\nminimum"])
        ax.set(title=f"RBF-SVR, {readout.upper() if i else readout}", xlabel=r"MAE(selected $p$) $-$ MAE($p=1$), eV")
        ax.grid(axis="x", alpha=.22)
        ax.set_ylim(3.6, -.6)
        letter(ax, "ab"[i])
    fig.subplots_adjust(left=.16, right=.97, bottom=.21, top=.87, wspace=.3)
    finish(fig, "02_hopping_history_contribution", "Negative differences favor the validation-selected reset over the single-encoded-event QRC limit p=1. Target/readout-specific p is selected from the original 27 candidates using six-seed mean validation MAE. Points and bars: mean paired MAE difference and pointwise 95% crossed bootstrap intervals over six reservoir seeds and 60 test trajectories (20,000 draws). p=1 still receives a backward-difference velocity inside the last event.")


def hopping_comparison():
    data = pd.read_csv(HERE / "hopping/method_summary.csv")
    data = data[(data.readout == "final") & (data.decoder == "rbf_svr")]
    methods = ["Selected-QRC", "QRC-p1", "Original-ESN", "Tuned-ESN", "Direct-20", "ExtraTrees"]
    if "ExtraTrees" not in set(data.method):
        return
    labels = ["Selected QRC", "QRC, p=1", "Original ESN", "Tuned ESN", "Direct SVR", "ExtraTrees"]
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.25))
    for i, (ax, target, title) in enumerate(zip(axes.flat, TARGETS, TITLES)):
        block = data[data.target == target].set_index("method").loc[methods]
        for j, method in enumerate(methods):
            row = block.loc[method]
            ax.errorbar(row.mae, j, xerr=[[row.mae-row.ci_low], [row.ci_high-row.mae]], fmt="o", ms=4,
                        capsize=2.5, color=COLORS[method], lw=1.15)
        ax.set_yticks(np.arange(len(methods)), labels)
        ax.set(ylim=(5.6, -.6), xlabel="Test trajectory MAE (eV)", title=title)
        ax.grid(axis="x", alpha=.2)
        letter(ax, "abcd"[i])
    fig.subplots_adjust(left=.18, right=.97, bottom=.10, top=.925, wspace=.65, hspace=.55)
    finish(fig, "03_hopping_strengthened_baselines", "Q15/T20, final RBF-SVR unless the method is ExtraTrees. Input-matched direct SVR and ExtraTrees use all 20 frames; no extra target information is supplied. Tuned ESN searches 27 spectral-radius/leak/input-scale configurations at 120 nodes and selects by validation. QRC-p1 explicitly removes across-encoded-event reservoir history. Dots and bars: trajectory-balanced mean MAE and pointwise 95% crossed seed/trajectory bootstrap intervals. The deterministic direct SVR is repeated only for pairing, not counted as independent seeds.")


def windows():
    paths = sorted((HERE / "hopping/direct").glob("window*/metrics.csv"))
    data = pd.concat([pd.read_csv(p) for p in paths])
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.2))
    for i, (ax, decoder) in enumerate(zip(axes, ("linear", "rbf_svr"))):
        for target, title, color, marker in zip(TARGETS, TITLES, ("#355c91", "#538f88", "#c68a3d", "#8b72a6"), ("o", "s", "^", "D")):
            block = data[(data.decoder == decoder) & (data.target == target)].sort_values("window")
            ax.plot(block.window, block.mae_ev, color=color, marker=marker, ms=3.7, lw=1.2, label=title)
        ax.set(xlabel="Trailing encoded frames", ylabel="Test MAE (eV)", title=("Direct Ridge", "Direct RBF-SVR")[i])
        ax.set_xticks([1, 2, 4, 8, 20])
        ax.grid(axis="y", alpha=.2)
        letter(ax, "ab"[i])
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.095, right=.975, bottom=.28, top=.88, wspace=.3)
    finish(fig, "04_direct_input_history_windows", "Direct decoders receive the last 1,2,4,8 or 20 of the original encoded frames, with unchanged train-only preprocessing and the original decoder grid. One encoded frame contains both distance and backward-difference velocity. This is a deliberate information-window ablation, not a same-input superiority comparison. Curves are deterministic held-out MAEs; no artificial seed SD is drawn.")


def graphs():
    data = pd.read_csv(HERE / "graph_direct/comparison_metrics.csv")
    methods = ["open_dtc_qrc", "matched_esn", "Direct-mean", "Direct-full"]
    labels = ["DTC-QRC", "Matched\nESN", "Mean\n+ SVR", "Sequence\n+ SVR"]
    colors = ["#355c91", "#538f88", "#c68a3d", "#8b72a6"]
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 4.95))
    for i, (ax, (dataset, t)) in enumerate(zip(axes.flat, [("BACE", 24), ("BBBP", 24), ("BACE", 30), ("BBBP", 30)])):
        for j, (method, color) in enumerate(zip(methods, colors)):
            values = data[(data.dataset == dataset) & (data.t == t) & (data.method == method)].sort_values("seed").test_auc.to_numpy()
            assert len(values) == 6
            ax.scatter(j + np.linspace(-.1, .1, 6), values, s=8, color=color, alpha=.5)
            ax.errorbar(j, values.mean(), yerr=values.std(ddof=1), color=color, fmt="s", capsize=3, ms=4.3, lw=1.2)
        ax.set_xticks(range(4), labels)
        ax.set(title=f"{dataset}, T={t}", ylabel="Test ROC-AUC", ylim=(.63, .94))
        ax.grid(axis="y", alpha=.2)
        letter(ax, "abcd"[i])
    fig.subplots_adjust(left=.095, right=.975, bottom=.105, top=.915, wspace=.32, hspace=.47)
    finish(fig, "05_graph_input_matched_controls", "BACE/BBBP use the same Q12-compatible local token prefixes, six stored scaffold splits and RBF-SVR grid. QRC p=0.10 and matched ESN expose 78 coordinates; mean pooling exposes 36 and flattening 36T. Small points: individual seeds; squares/bars: mean and sample SD. Each comparison fixes T; direct baselines are input-matched, not width-matched. Paired 95% intervals are supplied in the source table.")


def convergence():
    paths = sorted((HERE / "dephasing").glob("*/lambda*/seed*/convergence.csv"))
    if len(paths) != 24:
        return
    data = pd.concat([pd.read_csv(p) for p in paths])
    write_csv(HERE / "dephasing/convergence_all_seeds.csv", data)
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.15))
    for i, (ax, (task, lam)) in enumerate(zip(axes.flat, [("bace", .5), ("bace", 1.), ("hopping", .5), ("hopping", 1.)])):
        for p, color, marker in ((0., "#355c91", "o"), (.1, "#c68a3d", "s")):
            block = data[(data.task == task) & (data.dephasing == lam) & np.isclose(data.p, p)]
            grouped = block.groupby("trajectories").feature_se_rms.agg(["mean", "std"])
            x = grouped.index.to_numpy()
            y, sd = grouped["mean"].to_numpy(), grouped["std"].to_numpy()
            ax.plot(x, y, color=color, marker=marker, ms=4, lw=1.2, label=f"p={p:.1f}")
            ax.fill_between(x, np.maximum(y-sd, 1e-7), y+sd, alpha=.13, color=color, lw=0)
        ax.set_xscale("log", base=4)
        ax.set_yscale("log")
        ax.set_xticks([8, 32, 128], ["8", "32", "128"])
        ax.set_ylim(.002, .05)
        ax.set_yticks([.002, .005, .01, .02, .05])
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        ax.set(title=f"{'BACE, Q12' if task == 'bace' else 'Hopping, Q15'}, $\\lambda$={lam:g}",
               xlabel="Pauli trajectories", ylabel="RMS feature standard error")
        ax.grid(axis="y", alpha=.2, which="both")
        letter(ax, "abcd"[i])
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.105, right=.975, top=.925, bottom=.165, wspace=.35, hspace=.55)
    finish(fig, "06_dephasing_feature_convergence", "Original BACE Q12/T30 and hopping Q15/T20, final Z+ZZ; lambda0.5/1 and p0/0.1. Each cell uses the same 24 prespecified representative inputs and six reservoir seeds. Standard error is computed across independent Pauli trajectories for each input/observable, then summarized by RMS. Lines/bands: six-seed mean and sample SD of this RMS. Ensembles are nested in trajectory count. This quantifies feature-estimator precision, not full-data AUC/MAE convergence or exact density-matrix error.")


def gap_strata():
    path = HERE / "hopping/gap_strata_metrics.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    methods = ["Selected-QRC", "QRC-p1", "Tuned-ESN", "Direct-20", "ExtraTrees"]
    labels = ["Selected QRC", "QRC, p=1", "Tuned ESN", "Direct SVR", "ExtraTrees"]
    bins = ["0-1", "1-3", ">=3"]
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.3))
    for i, (ax, target, title) in enumerate(zip(axes.flat, TARGETS, TITLES)):
        block = data[data.target == target]
        for method, label, marker in zip(methods, labels, ("o", "D", "^", "s", "v")):
            values = block[block.method == method].set_index("gap_bin_ev").loc[bins]
            ax.plot(np.arange(3), values.trajectory_balanced_mae, marker=marker, ms=3.5,
                    color=COLORS[method], lw=1.15, label=label)
        count = block[block.method == "Selected-QRC"].set_index("gap_bin_ev").loc[bins].n_windows
        ax.set_xticks(np.arange(3), [f"0–1\n(n={count.iloc[0]})", f"1–3\n(n={count.iloc[1]})", f"≥3\n(n={count.iloc[2]})"])
        ax.set(title=title, xlabel="True target gap (eV)", ylabel="Trajectory MAE (eV)")
        ax.grid(axis="y", alpha=.2)
        letter(ax, "abcd"[i])
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.095, right=.975, bottom=.235, top=.925, wspace=.35, hspace=.65)
    finish(fig, "07_hopping_gap_strata", "Exploratory descriptive analysis of frozen final-RBF-SVR models and ExtraTrees. Bins [0,1), [1,3), [3,infinity) eV use the true held-out target; they do not supply an inference-time input or select a model. Each represented physical trajectory is equally weighted within a bin. Lines show model-seed means; no confidence or significance claims are attached to these subgroup curves. n gives test-window counts; distinct-trajectory counts are in the source CSV. The direct SVR has one deterministic fit. These thresholds are not validated chemical-accuracy or warning thresholds.")


if __name__ == "__main__":
    probes()
    history()
    hopping_comparison()
    windows()
    graphs()
    convergence()
    gap_strata()
    print("Exported:", [p.name for p in sorted(OUT.glob("*.png"))])
