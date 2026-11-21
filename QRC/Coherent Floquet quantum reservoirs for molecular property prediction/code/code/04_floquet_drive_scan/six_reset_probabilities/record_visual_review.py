"""Record the completed assistant visual inspection of all eight actual PNGs.

This record is not an automated inference that visual review has occurred. It
is executed only after all eight exports have been opened and inspected.
"""
from grid_common import *
from datetime import datetime, timezone


def main():
    freeze()
    automatic = json.loads((HERE / "completion_audit.json").read_text())
    numeric = json.loads((HERE / "numerical_audit.json").read_text())
    qa = json.loads((HERE / "figures/figure_qa.json").read_text())
    assert automatic["passed"] and numeric["passed"] and qa["passed"]
    assert len(qa["figures"]) == 8
    names = [entry["name"] for entry in qa["figures"]]
    records = [dict(name=name, inspected_png_sha256=sha(HERE / "figures" / f"{name}.png"),
        passed=True, full_frames=True, rightmost_tick_visible=True,
        no_label_or_legend_collision=True, no_replacement_glyphs_visible=True)
        for name in names]
    now = datetime.now(timezone.utc).isoformat()
    review = dict(passed=True, reviewed_utc=now, reviewer="assistant visual inspection",
        figures=records, comparison="all six fixed p values on identical g and T within each panel",
        statistics="raw: six-seed mean and sample SD; paired: crossed seed/trajectory pointwise 95% intervals",
        notes=["Overlapping data curves were not offset or altered.",
               "Raw MAE panels use task-specific y limits.",
               "The g=0.84 line is a reference, not a fitted phase boundary.",
               "Only the code-level right margin changed after the initial export check."])
    jwrite(HERE / "figures/visual_review.json", review)
    deliverables = [HERE / "figures/all_six_fixed_p_figures.pdf", HERE / "metrics_all_seeds.csv",
                    HERE / "summary.csv", HERE / "paired_delta_mae.csv",
                    HERE / "figures/visual_review.json", HERE / "figures/plot_revision.json"]
    jwrite(HERE / "delivery_status.json", dict(complete=True, updated_utc=now,
        feature_cases=300, decoder_cases=300, metric_rows=28800, six_seed_cells=4800,
        reused_prediction_rows=6900, new_decoder_result_rows=21900, figures=8,
        physical_protocol_unchanged=True, numerical_audit_passed=True,
        automatic_figure_audit_passed=True, visual_review_passed=True,
        fixed_p=list(PS), q=Q, t=T, layers=LAYERS, W=W, seeds=list(SEEDS),
        final_renderer="grid_plot_v2.py",
        artifacts={str(p.relative_to(HERE)): sha(p) for p in deliverables}))
    jwrite(HERE / "status.json", dict(state="complete", stage="all training, numerical audits, exports and visual review complete",
        pid=None, updated_utc=now, completed_features=300, completed_decoders=300,
        expected_cases=300, expected_metric_rows=28800, visual_review_passed=True,
        delivery_report="delivery_status.json"))
    print("COMPLETE: 300 cases, 28800 results, 8 figures; numerical and visual QA passed", flush=True)


if __name__ == "__main__":
    main()
