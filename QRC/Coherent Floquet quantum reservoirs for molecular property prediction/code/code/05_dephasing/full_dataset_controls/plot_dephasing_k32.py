"""Reproduce the two main dephasing panels using the completed K=32 refits.

Figure contract: assess performance changes under event dephasing at a fixed
Monte Carlo budget, without interpreting K=32 as the converged channel limit.
Two standalone quantitative panels: BACE AUC and target-balanced hopping MAE
ratio; final Z+ZZ/RBF-SVR only, p=0 and 0.10, six paired seeds. Python-only,
square full-frame axes, 89.9-mm-wide PDF/SVG/PNG with editable text, source rows
and QA. Lines and bands are means and sample SDs across seed means, matching
the earlier main-panel presentation, not bootstrap confidence intervals.
"""
from __future__ import annotations

import hashlib
import json

from common import HERE, SEEDS, np, pd, write_json
from plot_results import OUT, finish, plt


ANALYSIS = HERE / "dephasing_full_k/analysis/all"
CONDITIONS = (
    (0.0, "#40578f", "o", r"Closed, $p=0$"),
    (0.1, "#c88a39", "s", r"Dissipative, $p=0.10$"),
)


def select(frame):
    """Do not relabel the exact coherent K=0 reference as a sampled K=32 row."""
    mask = ((frame.K == 32) & frame.dephasing.isin([0.5, 1.0])) | (
        (frame.K == 0) & (frame.dephasing == 0.0))
    return frame[mask & (frame.decoder == "rbf_svr")
                 & (frame.readout == "final") & frame.p.isin([0.0, 0.1])].copy()


def validate_summary(rows, reference, value, trajectories):
    keys = ["dephasing", "K", "p"]
    assert len(reference) == 6
    assert (reference.n_seeds == 6).all()
    for key, block in rows.groupby(keys):
        assert set(block.seed) == set(SEEDS)
        if trajectories:
            grid = block.pivot(index="seed", columns="trajectory_id", values=value)
            assert grid.shape == (6, 60) and not grid.isna().any().any()
            means = grid.mean(axis=1)
        else:
            assert len(block) == 6
            means = block[value]
        record = reference
        for name, number in zip(keys, key):
            record = record[record[name] == number]
        assert len(record) == 1
        record = record.iloc[0]
        np.testing.assert_allclose(means.mean(), record["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(means.std(ddof=1), record.sd_across_seed_means,
                                   rtol=0, atol=1e-12)
    return reference.sort_values(["p", "dephasing"])


def draw(summary, raw, *, name, title, parameters, ylabel, ylim, yticks,
         reference_line, caption, sources):
    fig = plt.figure(figsize=(3.54, 4.25))
    ax = fig.add_axes([0.18, 0.26, 0.77, 0.77 * 3.54 / 4.25])
    ax.set_box_aspect(1)
    for p, color, marker, label in CONDITIONS:
        block = summary[summary.p == p].sort_values("dephasing")
        np.testing.assert_array_equal(block.dephasing, [0.0, 0.5, 1.0])
        x, y = block.dephasing.to_numpy(), block["mean"].to_numpy()
        sd = block.sd_across_seed_means.to_numpy()
        assert np.min(y - sd) > ylim[0] and np.max(y + sd) < ylim[1]
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.14,
                        linewidth=0, zorder=1)
        ax.plot(x, y, color=color, marker=marker, markersize=4.0,
                linewidth=1.4, label=label, zorder=3)
    ax.axhline(reference_line, color="#888888", linestyle="--", linewidth=0.85,
               zorder=2)
    ax.set(xlim=(-0.05, 1.05), ylim=ylim,
           xlabel=r"Event dephasing, $\lambda$", ylabel=ylabel)
    ax.set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    ax.set_yticks(yticks)
    ax.tick_params(direction="out", length=3)
    ax.grid(axis="y", color="#999999", alpha=0.2, linewidth=0.6)
    fig.text(0.56, 0.967, title, ha="center", va="center", fontsize=10)
    fig.text(0.56, 0.932, parameters, ha="center", va="center", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=2,
                        bbox_to_anchor=(0.5, 0.091), fontsize=7.0, handlelength=1.7,
                        handletextpad=0.45, columnspacing=1.1, borderpad=0.65)
    fig.text(0.5, 0.061, "Final Z+ZZ; RBF-SVR; mean +/- SD (6 seeds)",
             ha="center", va="center", fontsize=7.0)
    fig.text(0.5, 0.026, r"$K=32$ at $\lambda>0$; exact reference at $\lambda=0$",
             ha="center", va="center", fontsize=6.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert not legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer))
    for text in fig.texts:
        assert not legend_box.overlaps(text.get_window_extent(renderer))
    summary.to_csv(OUT / f"{name}_summary.csv", index=False)
    raw.to_csv(OUT / f"{name}_source.csv", index=False)
    finish(fig, name, caption)
    qa_file = OUT / f"{name}_qa.json"
    qa = json.loads(qa_file.read_text())
    qa.update(
        paired_seeds=list(SEEDS), dephasing_grid=[0.0, 0.5, 1.0],
        K_at_positive_dephasing=32, coherent_reference="exact; source K=0",
        readout="final Z+ZZ", decoder="RBF-SVR", reset_values=[0.0, 0.1],
        statistics="mean +/- sample SD across seed means (ddof=1), not CI",
        raw_rows=len(raw), plotted_conditions=len(summary),
        statistics_recomputed_and_verified=True, square_data_axes=True,
        legend_label_overlap=False,
        source_sha256={str(path.relative_to(HERE)):
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        scientific_caveat="K=32 is a finite-budget result, not a convergence claim",
    )
    write_json(qa_file, qa)
    print(name)
    print(summary[["dephasing", "K", "p", "mean", "sd_across_seed_means"]]
          .to_string(index=False))


def main():
    verification = json.loads((ANALYSIS / "verification.json").read_text())
    assert verification["complete"] and verification["verified"]
    assert verification["numerical_verification_passed"]
    metrics_path = ANALYSIS / "metrics_all_seeds.csv"
    metrics = select(pd.read_csv(metrics_path))
    bace = metrics[metrics.task == "bace"].copy()
    assert len(bace) == 36 and (bace.q == 12).all() and (bace.t == 30).all()
    bace_summary_path = ANALYSIS / "bace_auc_summary.csv"
    summary = validate_summary(bace, select(pd.read_csv(bace_summary_path)),
                                "test_auc", trajectories=False)
    common_caption = (
        "Closed (blue circles, p=0) and dissipative (orange squares, p=0.10) "
        "DTC-QRC, g=0.84, W=0.50, three Floquet cycles per input event. "
        "The lambda=0 control uses exact coherent features. The lambda=0.5 "
        "and 1.0 conditions use K=32 Monte Carlo Pauli trajectories per input "
        "row and suffix, with independently retrained RBF-SVR decoders and "
        "the original train/validation/test split and decoder-selection grid. "
        "Each wire receives Z with probability lambda/2 at the event boundary. "
        "The reset channel is mixed with exact suffix weights. Final Z+ZZ "
        "readout only. Curves and shading: arithmetic mean and sample standard "
        "deviation across six paired seeds (0,10,20,30,40,50), not confidence "
        "intervals. K=32 is a finite Monte Carlo budget, not an assertion of "
        "convergence; K=8 and K=128 are not included. "
    )
    draw(summary, bace, name="09a_k32_bace_dephasing", title="BACE dephasing test",
         parameters=r"$Q=12$, $T=30$; local chemical tokens",
         ylabel="Test ROC-AUC", ylim=(0.40, 0.85), yticks=np.arange(0.4, 0.851, 0.1),
         reference_line=0.5,
         caption=("BACE, Q=12, T=30, canonical-general local chemical tokens, "
                  "78 retained observables. " + common_caption +
                  "Decoder selection: validation AUC, then log loss, then accuracy, "
                  "using the original train-only fit. Dashed line: chance AUC=0.5."),
         sources=[metrics_path, bace_summary_path])

    hopping = metrics[metrics.task == "hopping"]
    assert (hopping.q == 15).all() and (hopping.t == 20).all()
    ratio_path = ANALYSIS / "hopping_target_balanced_trajectory_ratios.csv"
    raw = select(pd.read_csv(ratio_path))
    assert len(raw) == 6 * 6 * 60
    ratio_summary_path = ANALYSIS / "hopping_ratio_summary.csv"
    summary = validate_summary(raw, select(pd.read_csv(ratio_summary_path)),
                                "mae_ratio_to_coherent", trajectories=True)
    draw(summary, raw, name="09b_k32_hopping_dephasing", title="Surface-hopping dephasing test",
         parameters=r"$Q=15$, $T=20$; four prediction targets",
         ylabel="Target-balanced MAE ratio", ylim=(0.97, 1.35),
         yticks=np.arange(1.0, 1.351, 0.05), reference_line=1.0,
         caption=("SHNITSEL A01 surface hopping, Q=15, T=20, 120 retained "
                  "observables, original distance/backward-velocity inputs. " +
                  common_caption + "For each seed, physical test trajectory and "
                  "prediction target, MAE is divided by that of the matched "
                  "lambda=0 model with the same p. Ratios are averaged equally "
                  "over four targets, then 60 physical test trajectories, then "
                  "six seeds. The shaded SD is over the six seed-level means. "
                  "Dashed line: ratio=1; values above 1 indicate worse prediction "
                  "than the same-p coherent control. Validation MAE selects "
                  "decoder parameters, followed by the original train+validation "
                  "refit. No absolute-MAE comparison between p values is implied."),
         sources=[ratio_path, ratio_summary_path])


if __name__ == "__main__":
    main()
