"""Predeclared raw MAE-g plots from complete six-seed measured cells only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz

TITLES = dict(zip(TARGET_NAMES, ("Gap, 1 fs", "Gap, 5 fs", "Gap, 10 fs", "Minimum gap, next 5 fs")))


def main():
    source = pd.read_csv(HERE/"summary.csv")
    paired = pd.read_csv(HERE/"paired_delta_mae.csv")
    assert len(source) == 1600 and source.n_seeds.eq(6).all()
    assert len(paired) == 800 and paired.n_seeds.eq(6).all()
    out = HERE/"figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "axes.linewidth": .8, "axes.spines.top": True,
                         "axes.spines.right": True, "legend.frameon": True,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    qa, pdfs = [], []
    def save(fig, name, subset):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = list(fig.texts) + list(fig.legends)
        for ax in fig.axes:
            assert all(s.get_visible() for s in ax.spines.values())
            artists += [ax.xaxis.label, ax.yaxis.label, ax.title, *ax.get_xticklabels(),
                        *ax.get_yticklabels(), *ax.texts]
        for artist in artists:
            if not artist.get_visible() or (hasattr(artist, "get_text") and not artist.get_text()):
                continue
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, (name, str(artist))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out/f"{name}.{ext}", dpi=300)
        cwrite(out/f"{name}_source.csv", subset)
        pdfs.append(out/f"{name}.pdf")
        qa.append(dict(name=name, full_frames=True, labels_inside_canvas=True, six_seeds=True, smoothed=False))
        plt.close(fig)
    def frame(ax, i):
        ax.set(xlabel="Floquet parameter, g", xlim=(0, .98))
        ax.set_xticks([0, .2, .4, .6, .8, .98], ["0", "0.2", "0.4", "0.6", "0.8", "0.98"])
        ax.axvline(.84, ls=":", lw=.8, color="#969BA2")
        ax.grid(axis="y", lw=.6, color="#ECEEF0")
        ax.set_axisbelow(True)
        ax.text(-.15, 1.045, "abcd"[i], transform=ax.transAxes, fontweight="bold", fontsize=11)
    for readout in READOUTS:
        for decoder in DECODERS:
            sub = source[source.readout.eq(readout) & source.decoder.eq(decoder)]
            fig, axes = plt.subplots(2, 2, figsize=(180/25.4, 150/25.4))
            fig.subplots_adjust(left=.11, right=.98, bottom=.15, top=.86, hspace=.44, wspace=.32)
            for i, target in enumerate(TARGET_NAMES):
                ax = axes.flat[i]
                p = float(choices().query("readout == @readout and decoder == @decoder and target == @target").iloc[0].p)
                for condition, color, style, label in (("closed", "#435A89", "-", "Closed, p = 0"),
                                                        ("frozen_p", "#C88A3E", "--", "Previously selected p")):
                    cell = sub[sub.target.eq(target) & sub.condition.eq(condition)].sort_values("g")
                    assert len(cell) == 50
                    x, y, sd = (cell[k].to_numpy() for k in ("g", "mae_mean", "mae_sd"))
                    ax.plot(x, y, color=color, ls=style, lw=1.2, label=label)
                    ax.fill_between(x, y-sd, y+sd, color=color, alpha=.14, lw=0)
                frame(ax, i)
                ax.set_title(f"{TITLES[target]} | fixed p = {p:.2f}", pad=7)
                ax.set_ylabel("Test trajectory MAE (eV)")
            fig.suptitle(f'SHNITSEL | {"Ridge" if decoder=="linear" else "RBF-SVR"}, {readout.upper()}', y=.992, fontsize=10)
            fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                       bbox_to_anchor=(.54, .961), frameon=True, fancybox=False)
            fig.text(.54, .039, "Q = 15 | T = 20 | L = 3 | W = 0.50 | 6 paired reservoir seeds\n"
                     "Raw mean ± sample SD; p frozen across g. Dotted line: reference g = 0.84.",
                     ha="center", fontsize=7, linespacing=1.5)
            save(fig, f"mae_g_{readout}_{decoder}", sub)
            sub = paired[paired.readout.eq(readout) & paired.decoder.eq(decoder)]
            fig, axes = plt.subplots(2, 2, figsize=(180/25.4, 146/25.4))
            fig.subplots_adjust(left=.13, right=.98, bottom=.15, top=.90, hspace=.44, wspace=.34)
            for i, target in enumerate(TARGET_NAMES):
                ax = axes.flat[i]
                cell = sub[sub.target.eq(target)].sort_values("g")
                assert len(cell) == 50
                ax.plot(cell.g, cell.delta_mae_mean, color="#57978E", lw=1.2)
                ax.fill_between(cell.g.to_numpy(), cell.pointwise_ci_low.to_numpy(),
                                cell.pointwise_ci_high.to_numpy(), color="#57978E", alpha=.17, lw=0)
                ax.axhline(0, color="#868B93", ls="--", lw=.7)
                frame(ax, i)
                ax.set(title=TITLES[target], ylabel="Selected-p minus closed MAE (eV)")
            fig.suptitle(f'Paired reset effect | {"Ridge" if decoder=="linear" else "RBF-SVR"}, {readout.upper()}', y=.985, fontsize=10)
            fig.text(.54, .039, "Negative favors the previously selected reset setting; same g and T\n"
                     "Pointwise 95% crossed seed/trajectory bootstrap; simultaneous bands in source CSV.",
                     ha="center", fontsize=7, linespacing=1.5)
            save(fig, f"paired_delta_mae_g_{readout}_{decoder}", sub)
    with fitz.open() as bundle:
        for path in pdfs:
            with fitz.open(path) as doc:
                assert "MAE" in doc[0].get_text()
                bundle.insert_pdf(doc)
        bundle.save(out/"all_hopping_gscan_figures.pdf")
    jwrite(out/"figure_qa.json", dict(passed=True, figures=qa, backend="Python/matplotlib/PyMuPDF",
                                     manual_visual_review_required=True))


if __name__ == "__main__":
    main()
