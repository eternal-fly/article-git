"""Export the audited Distant-pair panel without changing the manuscript.

Figure contract: reproduce the unique-background controlled-probe comparison,
not the earlier repeated-template result. One quantitative comparison panel;
original numeric lag spacing, six paired seeds, mean +/- sample SD. Q12/T20,
final Z+ZZ, original affine decoder, identical 78-feature reservoir interface.
Python/matplotlib only; square data axes with all four spines; PDF/SVG/PNG plus
source rows, statistical summary, caption and QA. No extra baseline or fit.
"""
from __future__ import annotations

import hashlib
import json

from common import HERE, LAGS, SEEDS, np, pd, write_json
from plot_results import COLORS, OUT, finish, plt


NAME = "08_distant_pair_unique_background"
METHODS = (("DTC-QRC", "o", "DTC-QRC"),
           ("FP-CVR", "s", "FP-CVR"),
           ("ESN", "^", "Matched ESN"))


def main():
    source = HERE / "probes/metrics_all_seeds.csv"
    raw = pd.read_csv(source)
    raw = raw[(raw.family == "distant_pair") & (raw.decoder == "affine")
              & raw.method.isin([method for method, _, _ in METHODS])].copy()
    assert len(raw) == len(METHODS) * len(SEEDS) * len(LAGS)
    assert not raw.duplicated(["method", "lag", "seed"]).any()
    assert (raw.q == 12).all() and (raw.t == 20).all()
    assert np.allclose(raw.loc[raw.method == "DTC-QRC", "p"], 0.1)
    for _, block in raw.groupby(["method", "lag"]):
        assert set(block.seed) == set(SEEDS)

    audit = pd.read_csv(HERE / "probes/input_audit.csv")
    audit = audit[audit.family == "distant_pair"]
    assert len(audit) == len(SEEDS) * len(LAGS)
    assert (audit.complete_input_overlap == 0).all()
    assert (audit.group_overlap == 0).all()
    assert (audit.unique_inputs == 512).all()
    assert (audit.unique_backgrounds == 128).all()

    summary = raw.groupby(["method", "lag"], as_index=False).agg(
        auc_mean=("test_auc", "mean"), auc_sd=("test_auc", "std"),
        n_seeds=("seed", "nunique"))
    reference = pd.read_csv(HERE / "probes/summary.csv")
    reference = reference[(reference.family == "distant_pair")
                          & (reference.decoder == "affine")]
    checked = summary.merge(reference, on=["method", "lag"],
                            suffixes=("", "_reference"), validate="one_to_one")
    assert len(checked) == len(summary)
    for field in ("auc_mean", "auc_sd"):
        np.testing.assert_allclose(checked[field], checked[f"{field}_reference"],
                                   rtol=0, atol=1e-12)

    fig = plt.figure(figsize=(3.54, 4.10))
    ax = fig.add_axes([0.18, 0.235, 0.77, 0.77 * 3.54 / 4.10])
    ax.set_box_aspect(1)
    for method, marker, label in METHODS:
        block = summary[summary.method == method].sort_values("lag")
        x = block.lag.to_numpy()
        y, sd = block.auc_mean.to_numpy(), block.auc_sd.to_numpy()
        ax.fill_between(x, y - sd, y + sd, color=COLORS[method], alpha=0.13,
                        linewidth=0, zorder=1)
        ax.plot(x, y, color=COLORS[method], marker=marker, markersize=4.0,
                linewidth=1.4, label=label, zorder=3)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.85, zorder=2)
    ax.set(xlim=(0.45, 16.55), ylim=(0.4, 0.9), xlabel="Event separation",
           ylabel="Test ROC-AUC")
    ax.set_xticks(LAGS)
    ax.set_yticks(np.arange(0.4, 0.91, 0.1))
    ax.grid(axis="y", color="#999999", alpha=0.2, linewidth=0.6)
    ax.tick_params(direction="out", length=3)
    fig.text(0.56, 0.967, "Distant event pair", ha="center", va="center",
             fontsize=10, fontweight="medium")
    fig.text(0.56, 0.932, r"$Q=12$, $T=20$, $p=0.10$", ha="center",
             va="center", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.078), fontsize=7.2,
               handlelength=1.5, handletextpad=0.45, columnspacing=0.85,
               borderpad=0.65)
    fig.text(0.5, 0.034, "Linear readout; mean +/- SD across 6 paired seeds",
             ha="center", va="center", fontsize=7.0)
    caption = (
        "Distant event pair, unique-background controlled probe. Q=12, T=20, "
        "p=0.10, g=0.84, W=0.50, L=3, default interaction scale J=1. Final "
        "Z+ZZ expectations (78 features); original affine decoder for DTC-QRC, "
        "FP-CVR and matched ESN, with identical input events. Each lag uses "
        "512 unique complete inputs in 128 counterfactual groups, with original "
        "70/15/15 group splits and six paired seeds (0,10,20,30,40,50). "
        "Train/validation/test complete inputs and background groups do not "
        "overlap. Lines are arithmetic means; shaded bands are sample standard "
        "deviations across paired seeds, not confidence intervals. Dashed line: "
        "chance ROC-AUC=0.5. Numeric lag spacing and y limits reproduce the "
        "existing two-panel figure. Background diversity also changed relative "
        "to the earlier repeated-template experiment; the new-versus-old "
        "contrast is not a split-only intervention. Source: "
        "probes/metrics_all_seeds.csv; all plotted means and SDs independently "
        "recomputed and checked against probes/summary.csv."
    )
    raw.sort_values(["method", "lag", "seed"]).to_csv(
        OUT / f"{NAME}_source.csv", index=False)
    summary.to_csv(OUT / f"{NAME}_summary.csv", index=False)
    finish(fig, NAME, caption)
    qa_path = OUT / f"{NAME}_qa.json"
    qa = json.loads(qa_path.read_text())
    qa.update(source=str(source.relative_to(HERE)),
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              plotted_rows=len(raw), paired_seeds=list(SEEDS), lags=list(LAGS),
              overlaps_complete_input=0, overlaps_group=0,
              statistics="arithmetic mean +/- sample SD (ddof=1)",
              verified_against_existing_summary=True, square_data_axes=True)
    write_json(qa_path, qa)
    print(summary.to_string(index=False))
    print(OUT / f"{NAME}.png")


if __name__ == "__main__":
    main()
