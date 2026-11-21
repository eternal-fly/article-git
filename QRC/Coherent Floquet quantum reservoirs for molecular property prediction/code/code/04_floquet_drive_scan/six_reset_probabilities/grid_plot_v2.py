"""Layout-only revision of the frozen renderer: preserve the rightmost g tick."""
from grid_common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz

TITLES = dict(zip(TARGET_NAMES, ("Gap, 1 fs", "Gap, 5 fs", "Gap, 10 fs", "Minimum gap, next 5 fs")))
COLORS = ("#3E4B65", "#497EA5", "#67A29B", "#989D64", "#C18B46", "#A16D8B")
STYLES = ("-", "-", "--", "-.", ":", "-")
MARKERS = (None, "o", None, None, None, "s")


def main():
    summary = pd.read_csv(HERE / "summary.csv")
    paired = pd.read_csv(HERE / "paired_delta_mae.csv")
    assert len(summary) == 4800 and summary.n_seeds.eq(6).all()
    assert len(paired) == 4000 and paired.n_seeds.eq(6).all()
    group = ["readout", "decoder", "target", "g"]
    closed = summary[summary.p.eq(0.)][group+["mae_mean"]].rename(columns={"mae_mean": "closed_mean"})
    contrast = summary[summary.p.gt(0.)].merge(closed, on=group, validate="many_to_one")
    contrast = contrast.merge(paired, on=group+["p"], validate="one_to_one")
    assert np.allclose(contrast.mae_mean-contrast.closed_mean, contrast.delta_mae_mean, atol=1e-5, rtol=1e-6)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "axes.linewidth": .8, "axes.spines.top": True,
        "axes.spines.right": True, "legend.frameon": True, "pdf.fonttype": 42, "svg.fonttype": "none"})
    qa, pdfs = [], []
    for readout in READOUTS:
        for decoder in DECODERS:
            for mode in ("raw", "paired"):
                source = summary if mode == "raw" else paired
                sub = source[source.readout.eq(readout) & source.decoder.eq(decoder)]
                fig, axes = plt.subplots(2, 2, figsize=(180/25.4, 158/25.4))
                fig.subplots_adjust(left=.12, right=.96, bottom=.15, top=.83, hspace=.44, wspace=.34)
                for ti, target in enumerate(TARGET_NAMES):
                    ax = axes.flat[ti]
                    for pi, p in enumerate(PS):
                        if mode == "paired" and p == 0:
                            x = np.asarray(GS)
                            y = low = high = np.zeros(len(GS))
                        else:
                            cell = sub[sub.target.eq(target) & np.isclose(sub.p, p)].sort_values("g")
                            assert len(cell) == 50 and np.allclose(cell.g, GS)
                            x = cell.g.to_numpy()
                            if mode == "raw":
                                y = cell.mae_mean.to_numpy()
                                low, high = y-cell.mae_sd.to_numpy(), y+cell.mae_sd.to_numpy()
                            else:
                                y = cell.delta_mae_mean.to_numpy()
                                low, high = cell.pointwise_ci_low.to_numpy(), cell.pointwise_ci_high.to_numpy()
                        ax.fill_between(x, low, high, color=COLORS[pi], alpha=.075, lw=0, zorder=1)
                        ax.plot(x, y, color=COLORS[pi], ls=STYLES[pi], marker=MARKERS[pi],
                                markevery=5, ms=2.2, lw=1.45 if p == 0 else 1.05,
                                label=f"p = {p:.1f}"+(" (closed)" if p == 0 else ""), zorder=3)
                    ax.set(title=TITLES[target], xlabel="Floquet parameter, g", xlim=(0, .98),
                           ylabel="Test trajectory MAE (eV)" if mode == "raw" else "MAE(p) − MAE(0) (eV)")
                    ax.title.set_position((.5, 1.025))
                    ax.set_xticks([0, .2, .4, .6, .8, .98], ["0", "0.2", "0.4", "0.6", "0.8", "0.98"])
                    ax.axvline(.84, ls=":", lw=.8, color="#9A9DA3", zorder=0)
                    ax.grid(axis="y", color="#ECEEF0", lw=.6)
                    ax.set_axisbelow(True)
                    ax.text(-.15, 1.05, "abcd"[ti], transform=ax.transAxes, fontsize=11, fontweight="bold")
                decoder_name = "Ridge" if decoder == "linear" else "RBF-SVR"
                title = "Six fixed reset settings" if mode == "raw" else "Fixed-reset effect relative to p = 0"
                fig.suptitle(f"{title} | {decoder_name}, {readout.upper()}", y=.995, fontsize=10)
                fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=3,
                           bbox_to_anchor=(.55, .958), frameon=True, fancybox=False,
                           handlelength=2.7, columnspacing=1.6, borderpad=.5)
                footer = ("Q = 15 | T = 20 | L = 3 | W = 0.50 | 6 paired reservoir seeds\n"
                          "Raw mean ± sample SD. Fixed p across g; dotted reference at g = 0.84.") if mode == "raw" else (
                          "Negative favors p > 0 at the identical g and T; p = 0 is the zero reference\n"
                          "Pointwise 95% crossed seed/trajectory intervals; simultaneous bands in source CSV.")
                fig.text(.55, .039, footer, ha="center", fontsize=7, linespacing=1.5)
                name = ("mae_g_six_p" if mode == "raw" else "paired_delta_mae_g_six_p")+f"_{readout}_{decoder}"
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                artists = list(fig.texts)+list(fig.legends)
                for ax in axes.flat:
                    assert all(s.get_visible() for s in ax.spines.values())
                    artists += [ax.title, ax.xaxis.label, ax.yaxis.label,
                                *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
                for artist in artists:
                    box = artist.get_window_extent(renderer)
                    assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, (name, str(artist))
                for extension in ("png", "pdf", "svg"):
                    fig.savefig(out/f"{name}.{extension}", dpi=300)
                cwrite(out/f"{name}_source.csv", sub)
                pdfs.append(out/f"{name}.pdf")
                qa.append(dict(name=name, six_p_curves=True, complete_grid=True,
                               full_frames=True, labels_inside_canvas=True, smoothed=False))
                plt.close(fig)
    with fitz.open() as bundle:
        for path in pdfs:
            with fitz.open(path) as document:
                assert len(document) == 1 and "MAE" in document[0].get_text()
                assert "\ufffd" not in document[0].get_text()
                assert all(font[1] in ("ttf", "otf") for font in document[0].get_fonts())
                bundle.insert_pdf(document)
        bundle.save(out / "all_six_fixed_p_figures.pdf")
    jwrite(out / "figure_qa.json", dict(passed=True, figures=qa, manual_visual_review_required=True))
    jwrite(out / "plot_revision.json", dict(
        original_renderer_sha256=sha(HERE / "grid_plot.py"),
        revised_renderer_sha256=sha(HERE / "grid_plot_v2.py"),
        change="right subplot margin 0.985 -> 0.96; no statistical or data changes",
        summary_sha256=sha(HERE / "summary.csv"),
        paired_sha256=sha(HERE / "paired_delta_mae.csv")))



if __name__ == "__main__":
    main()

