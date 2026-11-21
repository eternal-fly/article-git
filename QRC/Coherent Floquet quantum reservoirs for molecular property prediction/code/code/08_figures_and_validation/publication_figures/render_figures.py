"""Build the manuscript figures from the archived data and vector artwork."""
from pathlib import Path
import argparse
import shutil
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from workflow import draw_workflow

MM = 1 / 25.4
INK, RULE, PALE = "#25323b", "#aeb9c0", "#e2e7ea"
BLUE, ORANGE, TEAL, PURPLE = "#0072b2", "#d55e00", "#007c61", "#9b5487"
MODELS = [
    ("dissipative_dtc_qrc", ORANGE, "o", "-", "Dissipative QRC"),
    ("closed_dtc_qrc", BLUE, "s", "--", "Closed QRC"),
    ("matched_esn", TEAL, "^", "-.", "Matched ESN"),
]
LENGTHS = [(12, BLUE, "o"), (30, ORANGE, "s")]
FIGURES = {2: "fig3_graph_stream_same_t", 3: "fig4_hopping_validation_selected",
           4: "fig5_reset_scans_2x3", 5: "fig_gscan_rbf_final", 6: "fig5_dephasing_k8_k32",
           7: "fig5_hardware_dtc_gscan", 8: "figS_hardware_execution",
           9: "fig8_esn_width_single_column", 10: "fig_gscan_rbf_tm3"}

def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.7, "mathtext.fontset": "dejavusans",
        "axes.labelsize": 9, "axes.titlesize": 9, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#61707a", "axes.linewidth": .7, "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": .65, "ytick.major.width": .65, "xtick.major.size": 3,
        "ytick.major.size": 3, "axes.labelpad": 2, "xtick.major.pad": 3,
        "ytick.major.pad": 3, "legend.fontsize": 8.2, "legend.frameon": False,
        "legend.handlelength": 2.1, "legend.handletextpad": .45, "legend.columnspacing": 1.05,
        "legend.borderaxespad": .2, "pdf.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })

def canvas(width, height, rows=1, cols=1, **kwargs):
    return plt.subplots(rows, cols, figsize=(width * MM, height * MM), **kwargs)

def title(ax, panel, label):
    ax.set_title(f"({panel})  {label}", loc="left", fontweight="semibold", pad=4)

def extract(frame, field):
    x = frame[field].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"Missing plotted values in {field}")
    return x

def key(color, marker, dash, label):
    return Line2D([], [], color=color, marker=marker, linestyle=dash, lw=1.15,
                  ms=4.1, markeredgewidth=.6, label=label)

def trace(ax, frame, xcol, mean, color, marker, dash, *, sd=None, bounds=None,
          markevery=1, size=3.4, alpha=.12):
    frame = frame.sort_values(xcol)
    x, y = extract(frame, xcol), extract(frame, mean)
    if len(x) == 0 or len(np.unique(x)) != len(x):
        raise ValueError(f"Missing or repeated {xcol} coordinates")
    if sd:
        spread = extract(frame, sd)
        low, high = y - spread, y + spread
    else:
        low, high = (extract(frame, f) for f in bounds)
    if not np.all(low <= high):
        raise ValueError("Reversed uncertainty bounds")
    ax.fill_between(x, low, high, color=color, alpha=alpha, lw=0, zorder=1)
    line, = ax.plot(x, y, color=color, ls=dash, marker=marker, ms=size, lw=1.17,
                    markevery=markevery, markeredgewidth=.55, zorder=3)
    np.testing.assert_array_equal(line.get_ydata(), y)
    np.testing.assert_array_equal(line.get_xdata(), x)

def legend(fig, handles, xy, ncol, **kwargs):
    return fig.legend(handles=handles, loc="center", bbox_to_anchor=xy, ncol=ncol, **kwargs)

def model_keys():
    return [key(c, m, ls, label) for _, c, m, ls, label in MODELS]

def classification(source):
    data = pd.read_csv(source / "main/graph_stream_q12_curves.csv")
    fig, axes = canvas(86, 109, 2, 1, sharex=True)
    fig.subplots_adjust(left=.19, right=.97, bottom=.11, top=.85, hspace=.43)
    for ax, dataset, letter in zip(axes, ["BACE", "BBBP"], "ab"):
        ax.axvspan(20, 30.8, color="#edf0f1", lw=0)
        ax.axhline(.5, color="#717e86", dashes=(3, 2), lw=.76)
        for model, c, m, ls, _ in MODELS:
            code = "open_dtc_qrc" if model == "dissipative_dtc_qrc" else model
            d = data[(data.dataset == dataset) & (data.system == code)]
            assert len(d) == 10
            trace(ax, d, "t", "test_auc_mean", c, m, ls, sd="test_auc_sd")
        title(ax, letter, dataset)
        ax.set(xlim=(1.2, 30.8), ylim=(.475, .845), yticks=[.5, .6, .7, .8], ylabel="Test ROC-AUC")
    axes[-1].set(xticks=[2, 6, 10, 16, 20, 24, 30], xlabel=r"Visible events, $T$")
    legend(fig, model_keys(), (.54, .957), 2)
    return fig

def gap_forecasts(source):
    data = pd.read_csv(source / "main/hopping_validation_selected_rbf_final.csv")
    targets = ["gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs"]
    labels = ["Gap at\n1 fs", "Gap at\n5 fs", "Gap at\n10 fs", "Minimum gap\n(next 5 fs)"]
    series = [
        ("dissipative_dtc_qrc", "Dissipative\nDTC-QRC", ORANGE, ORANGE, None),
        ("closed_dtc_qrc", "Closed\nDTC-QRC", "#D4E7F2", BLUE, "//"),
        ("matched_esn", "Matched\nESN", "#83B9A9", TEAL, None),
    ]
    fig, ax = canvas(85.45, 79.06)
    fig.subplots_adjust(left=.17, right=.978, bottom=.145, top=.87)
    centers = np.arange(len(targets))
    for index, (model, label, face, edge, hatch) in enumerate(series):
        rows = data.loc[data.system == model].set_index("target", verify_integrity=True).loc[targets]
        means = extract(rows, "test_mae_mean_ev")
        deviations = extract(rows, "test_mae_sd_ev")
        if not rows.n_seeds.eq(6).all() or (deviations < 0).any():
            raise ValueError(f"Expected six-seed means and sample standard deviations for {model}")
        ax.bar(centers + (index - 1) * .235, means, width=.216, yerr=deviations,
               color=face, edgecolor=edge, linewidth=.68, hatch=hatch, label=label,
               error_kw=dict(ecolor=INK, elinewidth=.85, capsize=2.0, capthick=.85),
               zorder=3)
    ax.set(xlim=(-.49, 3.50), ylim=(0, 2.0), yticks=[0, .5, 1, 1.5, 2],
           ylabel="Test MAE (eV)")
    ax.set_xticks(centers, labels)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=PALE, linewidth=.50)
    ax.spines[:].set_visible(True)
    ax.legend(loc="center", bbox_to_anchor=(.5, 1.088), ncol=3, fontsize=8.5,
              handlelength=1.25, handletextpad=.45, columnspacing=1.6, borderpad=0)
    return fig

def dephasing(source):
    data=pd.read_csv(source/"main/fig5_dephasing_k8_k32/plotted_data.csv")
    fig, axes=canvas(86, 94, 2, 2, sharex=True, sharey="col")
    fig.subplots_adjust(left=.158, right=.98, bottom=.19, top=.862, wspace=.58, hspace=.53)
    for letter, ax in zip("abcd", axes.flat):
        d=data[data.panel == letter]
        k=int(d.k.iloc[0])
        is_graph=d.task.iloc[0] == "BACE"
        for p, (c, m, ls) in [(0, (BLUE, "s", "--")), (.1, (ORANGE, "o", "-"))]:
            curve=d[np.isclose(d.p, p)]
            assert len(curve) == 3 and (curve.n_seeds == 6).all()
            trace(ax, curve, "dephasing", "mean", c, m, ls, bounds=("band_low", "band_high"), size=3.6, alpha=.16)
        ax.set_box_aspect(1)
        title(ax, letter, rf"$k={k}$")
        ax.set(xlim=(-.065, 1.065), xticks=[0, .5, 1], ylim=(.4, .85) if is_graph else (.975, 1.36),
               yticks=[.4, .6, .8] if is_graph else [1, 1.15, 1.3],
               ylabel="Test ROC-AUC" if is_graph else "MAE ratio")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        if is_graph:
            ax.axhline(.5, color=RULE, ls=(0, (3, 2)), lw=.7, zorder=0)
    for col, label in enumerate(["BACE", "Ethene"]):
        bounds=axes[0, col].get_position()
        fig.text((bounds.x0+bounds.x1)/2, .963, label, ha="center", va="center", fontweight="semibold", fontsize=9)
    fig.text(.56, .125, r"Event dephasing, $\lambda$", ha="center", va="center", fontsize=9)
    legend(fig, [key(BLUE, "s", "--", r"$p=0$"), key(ORANGE, "o", "-", r"$p=0.10$")], (.54, .049), 2)
    return fig

def hardware_scan(hardware):
    prefix="hardware_dtc_gscan_"
    rows=pd.read_csv(hardware/(prefix+"connected_metrics_by_seed.csv"))
    retain=pd.read_csv(hardware/(prefix+"connected_retention.csv")).query("family == 'CZZ'").sort_values("g")
    batches=pd.read_csv(hardware/(prefix+"hardware_connected_batch_means.csv")).query("family == 'CZZ'")
    paired=pd.read_csv(hardware/(prefix+"hardware_connected_contrast_by_batch.csv"))
    ideals=rows[rows.domain == "ideal"].pivot(index=["g", "seed"], columns="family", values="test_auc")
    ideal_diff=(ideals.CZZ-ideals.Z).groupby(level="g").agg(["mean", "std"])
    hardware_diff=paired.groupby("g").CZZ_minus_Z.agg(["mean", "std"])
    assert len(retain) == 4 and (batches.groupby("g").size() == 3).all() and (ideals.groupby(level="g").size() == 10).all()
    fig, axes=canvas(86, 127, 2, 1, sharex=True)
    fig.subplots_adjust(left=.205, right=.96, bottom=.10, top=.88, hspace=.40)
    g=retain.g
    for ax in axes:
        ax.axvspan(.7, .9, color="#e8ecee", lw=0)
        ax.set(xlim=(.35, 1.05), xticks=[.4, .6, .84, 1.])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    for ax, idmean, idsd, hwmean, hwsd, batch, column in [
            (axes[0], retain.auc_mean_ideal, retain.auc_sd_ideal, retain.auc_mean_hardware,
             retain.auc_sd_hardware, batches, "auc_mean"),
            (axes[1], ideal_diff["mean"], ideal_diff["std"], hardware_diff["mean"],
             hardware_diff["std"], paired, "CZZ_minus_Z")]:
        ax.errorbar(g, idmean, yerr=idsd, color=BLUE, fmt="s--", lw=1.05, ms=4.2,
                    elinewidth=.85, capsize=2.35, capthick=.8)
        ax.errorbar(g, hwmean, yerr=hwsd, color=ORANGE, fmt="^-", lw=1.15, ms=4.5,
                    elinewidth=.85, capsize=2.35, capthick=.8)
        ax.scatter(batch.g, batch[column], s=13, facecolor="white", edgecolor=ORANGE, linewidth=.75, zorder=4)
    title(axes[0], "a", "Connected pair readout")
    axes[0].set(ylabel="Test ROC-AUC", ylim=(.37, .79), yticks=[.4, .5, .6, .7])
    title(axes[1], "b", "Gain over single-qubit readout")
    axes[1].axhline(0, color="#68767e", lw=.75, ls=(0, (3, 2)))
    axes[1].set(ylabel=r"$\mathrm{AUC}_{C^{ZZ}}-\mathrm{AUC}_{Z}$", ylim=(-.225, .225),
                yticks=[-.2, -.1, 0, .1, .2], xlabel=r"Floquet drive, $g$")
    legend(fig, [key(BLUE, "s", "--", "Ideal"), key(ORANGE, "^", "-", "Shenglian"),
                Line2D([], [], marker="o", mfc="white", mec=ORANGE, ls="none", ms=4, label="Batch")],
           (.53, .968), 3, handlelength=1.5, columnspacing=.8)
    return fig

def hardware_controls(hardware):
    data=pd.read_csv(hardware/"mainline_hardware_q6_auc_observable_ablation_en_source_data.csv")
    tasks=sorted(data.task_key.unique())
    assert len(tasks) == 5
    fig, axes=canvas(86, 106, 2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(left=.175, right=.972, bottom=.119, top=.855, hspace=.66)
    groups=[[("Ideal QRC", BLUE, "s", "Ideal"), ("Baihua", TEAL, "D", "Baihua"), ("Shenglian", ORANGE, "^", "Shenglian")],
            [("Z", TEAL, "o", r"$Z$"), ("ZZ", PURPLE, "s", r"$ZZ$"), ("Z+ZZ", BLUE, "D", r"$Z+ZZ$")]]
    legends=[]
    for ax, letter, series in zip(axes, "ab", groups):
        title(ax, letter, "Full readout" if letter == "a" else "Shenglian observables")
        ax.axhline(.5, color="#6e7b84", ls=(0, (3, 2)), lw=.78)
        ax.axvline(2.5, color=PALE, lw=.65)
        for offset, (value, c, m, label) in zip([-.245, 0, .245], series):
            for index, task in enumerate(tasks):
                sub=data[(data.panel == letter)&(data.series == value)&(data.task_key == task)].sort_values("seed")
                values=extract(sub, "test_auc")
                assert len(values) == 10
                mean, sd=sub.nested_seed_mean.iloc[0], sub.nested_seed_sd.iloc[0]
                np.testing.assert_allclose([mean, sd], [np.mean(values), np.std(values, ddof=1)])
                center=index+offset
                ax.scatter(center+np.linspace(-.058, .058, 10), values, s=9, marker=m,
                           color=c, alpha=.31, lw=0, zorder=2)
                ax.errorbar(center, mean, yerr=sd, fmt=m, ms=4.2, color=c, elinewidth=.92,
                            capsize=1.7, capthick=.85, zorder=3)
        handles=[Line2D([], [], ls="none", marker=m, color=c, ms=4.4, label=l) for _, c, m, l in series]
        legends.append(handles)
        ax.set(xlim=(-.58, 4.58), ylim=(.16, .9), yticks=[.2, .4, .6, .8], ylabel="Test ROC-AUC")
    axes[1].set_xticks(range(5), ["$L=3$\nrun 1", "$L=3$\nrun 2", "$L=3$\nrun 3", "$L=1$\nbase", "$L=1$\nfold3"])
    legend(fig, legends[0], (.55, .969), 3, handlelength=.85, columnspacing=.8)
    legend(fig, legends[1], (.56, .518), 3, handlelength=.85)
    return fig

def width_scan(source):
    data=pd.read_csv(source/"main/wide_esn_summary.csv")
    qrc=pd.read_csv(source/"main/wide_esn_qrc_reference.csv")
    fig, axes=canvas(86, 57, 1, 2, sharey=True)
    fig.subplots_adjust(left=.171, right=.98, bottom=.36, top=.855, wspace=.15)
    for ax, dataset, letter in zip(axes, ["BACE", "BBBP"], "ab"):
        for t, c, m in LENGTHS:
            sub=data[(data.dataset == dataset)&(data.t == t)]
            assert len(sub) == 4
            trace(ax, sub, "width", "test_auc_mean", c, m, "-", sd="test_auc_sd", size=3.6)
            ref=qrc[(qrc.dataset == dataset)&(qrc.t == t)].test_auc_mean.item()
            ax.axhline(ref, color=c, ls="--" if t == 12 else ":", lw=1.0)
        title(ax, letter, dataset)
        ax.set(xscale="log", xlim=(60, 2650), ylim=(.63, .845), yticks=[.65, .7, .75, .8])
        ax.set_xticks([78, 500, 1000, 2000], ["78", "500", "1k", "2k"], rotation=40, ha="right")
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    axes[0].set_ylabel("Test ROC-AUC")
    fig.text(.555, .188, "ESN nodes", ha="center", fontsize=9)
    handles=[key(c, m, "-", f"ESN, $T={t}$") for t, c, m in LENGTHS]
    handles.extend([key(c, None, "--" if t == 12 else ":", f"QRC, $T={t}$") for t, c, m in LENGTHS])
    legend(fig, handles, (.53, .082), 2, handlelength=1.5, fontsize=8)
    return fig

def render_archived_figure(source_dir, name, out, png_dir, number):
    source = source_dir / (name + ".pdf")
    destination = out / source.name
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    if png_dir:
        subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-singlefile", "-r", "220",
                        "-png", str(destination), str(png_dir / f"fig{number:02d}")], check=True)


def main():
    local=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=local.parent, help="QRC project root with BFS-QRC archive")
    parser.add_argument("--out", type=Path, default=local/"fig", help="vector PDF destination")
    parser.add_argument("--png-dir", type=Path, help="optional print-size previews")
    parser.add_argument("--only", type=int, nargs="+", choices=range(1, 11),
                        help="render only these manuscript figure numbers")
    args=parser.parse_args()
    selected = set(args.only or range(1, 11))
    src=args.project/"BFS-QRC/PRL/revised_open_dtc_qrc/source_data"
    hardware=args.project/"BFS-QRC/PRL/ch/evidence_tables"
    args.out.mkdir(parents=True, exist_ok=True)
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
    style()
    archive = src.parent / "fig"
    for number in (4, 5, 10):
        if number not in selected:
            continue
        render_archived_figure(archive, FIGURES[number], args.out, args.png_dir, number)
        print(f"Figure {number}: {FIGURES[number]}.pdf", flush=True)
    builders = [
        (1, "fig1_molecular_reservoir", draw_workflow, ()),
        (2, FIGURES[2], classification, (src,)),
        (3, FIGURES[3], gap_forecasts, (src,)),
        (6, FIGURES[6], dephasing, (src,)),
        (7, FIGURES[7], hardware_scan, (hardware,)),
        (8, FIGURES[8], hardware_controls, (hardware,)),
        (9, FIGURES[9], width_scan, (src,)),
    ]
    for i, name, build, params in builders:
        if i not in selected:
            continue
        fig=build(*params)
        fig.savefig(args.out/(name+".pdf"), metadata={"Title":name, "CreationDate":None, "ModDate":None})
        if i in (1, 3):
            fig.savefig(args.out / (name + ".png"), dpi=245 if i == 1 else 400)
        if args.png_dir:
            fig.savefig(args.png_dir/f"fig{i:02d}.png", dpi=220)
        plt.close(fig)
        print(f"Figure {i}: {name}.pdf", flush=True)

if __name__ == "__main__":
    main()

