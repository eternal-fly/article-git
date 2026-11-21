"""Plot unchanged six-p results against explicitly attributed reference regions.

The reference divisions are not phase estimates for the input-driven, resetting
reservoir. This renderer leaves the frozen experiment and previous plots intact.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_phase_reference"
PS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
GS = np.round(np.arange(50) * .02, 2)
TARGET_NAMES = ("gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs")
READOUTS = ("final", "tm3")
DECODERS = ("linear", "rbf_svr")
TITLES = dict(zip(TARGET_NAMES, ("Gap, 1 fs", "Gap, 5 fs", "Gap, 10 fs", "Minimum gap, next 5 fs")))
COLORS = ("#3E4B65", "#497EA5", "#67A29B", "#989D64", "#C18B46", "#A16D8B")
STYLES = ("-", "-", "--", "-.", ":", "-")
MARKERS = (None, "o", None, None, None, "s")
REGIONS = (("NTC", 0., .22, "#F0F4FB", "#4F6690"),
           ("Thermal", .22, .84, "#FCF2EE", "#916657"),
           ("DTC", .84, 1., "#F0F7EB", "#607D4D"))
REFERENCE = {
    "boundary_g": [.22, .84],
    "interpretation": "Literature reference regions, not phase estimates for this driven/reset reservoir",
    "g_convention": "Global single-qubit drive RX(pi*g), matching the reference g convention",
    "ntc_boundary_source": {
        "title": "Robust and efficient quantum reservoir computing with a discrete time crystal",
        "citation": "Zhang et al., Physical Review Applied 26, 014056 (2026)",
        "url": "https://doi.org/10.1103/w2w9-c944",
        "location": "Figure 2 and Section VI (NTC transition reference g=0.22)",
    },
    "dtc_boundary_source": {
        "title": "Observation of time-crystalline eigenstate order on a quantum processor",
        "citation": "Mi et al., Nature 601, 531-536 (2022)",
        "url": "https://doi.org/10.1038/s41586-021-04257-w",
        "public_manuscript": "https://arxiv.org/pdf/2107.13571",
        "location": "Main text following Eq. (1): theoretical g_c about 0.84; Figure 5: experimental interval 0.83-0.88",
        "experimental_transition_interval": [.83, .88],
        "nominal_reference": .84,
    },
    "attribution_note": "Google did not supply the g=0.22 NTC boundary; the full three-region convention follows Zhang et al.",
    "model_note": "The same RX angle convention does not imply the same phase boundaries when coupling/disorder, input drive, reset and evolution depth differ.",
}
CONTRACT = {
    "claim": "Show how the measured MAE-g curves align with externally specified NTC, thermal and DTC reference regions.",
    "evidence": "Four targets per figure; six fixed p values; final and TM3 readout; Ridge and RBF-SVR; raw MAE and paired changes at the same g.",
    "archetype": "quantitative grid",
    "backend": "Python/matplotlib only",
    "size_mm": [180, 168],
    "exports": ["png (300 dpi)", "pdf (editable text)", "svg (editable text)", "source.csv"],
    "statistics": "Six reservoir seeds on one fixed whole-trajectory split; raw mean +/- sample SD. Paired panels use existing pointwise 95% crossed seed/test-trajectory bootstrap intervals, not across-g tests.",
    "integrity": "No smoothing, retraining, data selection, altered curve values, or phase inference from task scores.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def add_reference_regions(ax):
    """A framed header strip keeps phase names clear of curves and error bands."""
    labels, patches = [], []
    for name, left, right, fill, ink in REGIONS:
        end = min(right, float(GS[-1]))
        ax.axvspan(left, end, facecolor=fill, edgecolor="none", zorder=-2)
        patch = Rectangle((left, 1.012), end-left, .075,
                          transform=ax.get_xaxis_transform(), facecolor=fill,
                          edgecolor="#92979F", lw=.45, clip_on=False)
        ax.add_patch(patch)
        patches.append(patch)
        labels.append(ax.text((left+end)/2, 1.0495, name,
                              transform=ax.get_xaxis_transform(), ha="center", va="center",
                              color=ink, fontsize=7.7, fontweight="medium", clip_on=False))
    for boundary in REFERENCE["boundary_g"]:
        ax.axvline(boundary, color="#83858A", lw=.75, ls=(0, (3, 3)), zorder=2)
    return labels, patches


def main():
    inputs = {name: sha(HERE / name) for name in
              ("summary.csv", "paired_delta_mae.csv", "protocol.json")}
    summary = pd.read_csv(HERE / "summary.csv")
    paired = pd.read_csv(HERE / "paired_delta_mae.csv")
    assert len(summary) == 4800 and summary.n_seeds.eq(6).all()
    assert len(paired) == 4000 and paired.n_seeds.eq(6).all()
    group = ["readout", "decoder", "target", "g"]
    closed = summary[summary.p.eq(0.)][group+["mae_mean"]].rename(columns={"mae_mean": "closed_mean"})
    contrast = summary[summary.p.gt(0.)].merge(closed, on=group, validate="many_to_one")
    contrast = contrast.merge(paired, on=group+["p"], validate="one_to_one")
    assert np.allclose(contrast.mae_mean-contrast.closed_mean, contrast.delta_mae_mean, atol=1e-5, rtol=1e-6)
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "figure_contract.json", CONTRACT)
    write_json(OUT / "phase_reference.json", REFERENCE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "axes.linewidth": .8, "axes.spines.top": True,
        "axes.spines.right": True, "legend.frameon": True, "pdf.fonttype": 42, "svg.fonttype": "none"})
    qa, pdfs = [], []
    for readout in READOUTS:
        for decoder in DECODERS:
            for mode in ("raw", "paired"):
                source = summary if mode == "raw" else paired
                sub = source[source.readout.eq(readout) & source.decoder.eq(decoder)]
                fig, axes = plt.subplots(2, 2, figsize=(180/25.4, 168/25.4))
                fig.subplots_adjust(left=.12, right=.96, bottom=.17, top=.80, hspace=.60, wspace=.34)
                phase_labels, phase_patches = [], []
                for ti, target in enumerate(TARGET_NAMES):
                    ax = axes.flat[ti]
                    for pi, p in enumerate(PS):
                        if mode == "paired" and p == 0:
                            x = GS.copy()
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
                        line, = ax.plot(x, y, color=COLORS[pi], ls=STYLES[pi], marker=MARKERS[pi],
                                       markevery=5, ms=2.2, lw=1.45 if p == 0 else 1.05,
                                       label=f"p = {p:.1f}"+(" (closed)" if p == 0 else ""), zorder=3)
                        np.testing.assert_array_equal(line.get_ydata(), y)
                        np.testing.assert_array_equal(line.get_xdata(), x)
                    original_ylim = ax.get_ylim()
                    labels, patches = add_reference_regions(ax)
                    phase_labels.extend(labels)
                    phase_patches.extend(patches)
                    # Background annotations must not change the vertical data scale.
                    ax.set_ylim(original_ylim)
                    ax.set(xlabel="Floquet parameter, g", xlim=(0, .98),
                           ylabel="Test trajectory MAE (eV)" if mode == "raw" else "MAE(p) − MAE(0) (eV)")
                    ax.set_title(TITLES[target], y=1.145, pad=0)
                    ax.set_xticks([0, .2, .4, .6, .8, .98], ["0", "0.2", "0.4", "0.6", "0.8", "0.98"])
                    ax.grid(axis="y", color="#DEE1E5", lw=.55)
                    ax.set_axisbelow(True)
                    ax.text(-.15, 1.145, "abcd"[ti], transform=ax.transAxes, fontsize=11, fontweight="bold")
                    assert len([line for line in ax.lines if line.get_label().startswith("p = ")]) == 6
                decoder_name = "Ridge" if decoder == "linear" else "RBF-SVR"
                title = "Six fixed reset settings" if mode == "raw" else "Fixed-reset effect relative to p = 0"
                fig.suptitle(f"{title} | {decoder_name}, {readout.upper()}", y=.992, fontsize=10)
                fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=3,
                           bbox_to_anchor=(.55, .955), frameon=True, fancybox=False,
                           handlelength=2.7, columnspacing=1.6, borderpad=.5)
                first = ("Q = 15 | T = 20 | L = 3 | W = 0.50 | 6 reservoir seeds; mean ± SD"
                         if mode == "raw" else "Paired ΔMAE at the same g and T; pointwise 95% seed/trajectory intervals")
                footer = (first + "\nLiterature reference regions: boundaries at g = 0.22 and 0.84.\n"
                          "Reference zoning, not phase estimates for the driven/reset reservoir.")
                fig.text(.55, .031, footer, ha="center", fontsize=6.8, linespacing=1.5)
                name = ("mae_g_six_p" if mode == "raw" else "paired_delta_mae_g_six_p")+f"_{readout}_{decoder}"
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                artists = list(fig.texts)+list(fig.legends)+phase_patches
                for ax in axes.flat:
                    assert all(s.get_visible() for s in ax.spines.values())
                    artists += [ax.title, ax.xaxis.label, ax.yaxis.label,
                                *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
                for artist in artists:
                    box = artist.get_window_extent(renderer)
                    assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, (name, str(artist))
                for label, patch in zip(phase_labels, phase_patches):
                    lb, pb = label.get_window_extent(renderer), patch.get_window_extent(renderer)
                    assert pb.contains(lb.x0, lb.y0) and pb.contains(lb.x1, lb.y1), (name, label.get_text())
                for ax in axes.flat:
                    title_box = ax.title.get_window_extent(renderer)
                    for label in ax.texts:
                        if label.get_text() in {region[0] for region in REGIONS}:
                            assert not title_box.overlaps(label.get_window_extent(renderer))
                for extension in ("png", "pdf", "svg"):
                    fig.savefig(OUT/f"{name}.{extension}", dpi=300)
                sub.to_csv(OUT/f"{name}_source.csv", index=False)
                original_source = pd.read_csv(HERE/"figures"/f"{name}_source.csv")
                pd.testing.assert_frame_equal(pd.read_csv(OUT/f"{name}_source.csv"), original_source)
                pdfs.append(OUT/f"{name}.pdf")
                qa.append(dict(name=name, six_p_curves=True, complete_grid=True,
                               full_frames=True, labels_inside_canvas=True,
                               region_labels_framed_and_clear_of_titles=True,
                               unchanged_source_data=True, unchanged_y_range=True, smoothed=False))
                plt.close(fig)
    with fitz.open() as bundle:
        for path in pdfs:
            with fitz.open(path) as document:
                assert len(document) == 1
                text = document[0].get_text()
                assert all(text.count(label) == 4 for label in ("NTC", "Thermal", "DTC"))
                assert "\ufffd" not in text and "0.22" in text and "0.84" in text
                assert all(font[1] in ("ttf", "otf") for font in document[0].get_fonts())
                bundle.insert_pdf(document)
        bundle.save(OUT / "all_six_fixed_p_phase_reference_figures.pdf")
    assert inputs == {name: sha(HERE / name) for name in inputs}
    write_json(OUT / "figure_qa.json", dict(passed=True, figures=qa, manual_visual_review_required=True))
    write_json(OUT / "plot_revision.json", dict(
        prior_renderer_sha256=sha(HERE / "grid_plot_v2.py"),
        renderer_sha256=sha(Path(__file__)), inputs_sha256=inputs,
        change="Only reference shading, labels and layout; original figures retained; identical source data and curves"))
    print(json.dumps({"output": str(OUT), "figures": len(qa), "automated_qa_passed": True}))


if __name__ == "__main__":
    main()
