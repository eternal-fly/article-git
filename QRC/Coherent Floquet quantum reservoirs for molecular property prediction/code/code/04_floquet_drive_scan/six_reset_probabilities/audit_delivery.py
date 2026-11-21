"""Independent artifact and source audit after the six-p scan completes."""
from grid_common import *
import fitz


def main():
    freeze()
    numeric = json.loads((HERE / "numerical_audit.json").read_text())
    assert numeric["passed"] and numeric["rows"] == 28800
    raw = pd.read_csv(HERE / "metrics_all_seeds.csv")
    summary = pd.read_csv(HERE / "summary.csv")
    paired = pd.read_csv(HERE / "paired_delta_mae.csv")
    assert len(raw) == 28800 and len(summary) == 4800 and len(paired) == 4000
    assert set(raw.p) == set(PS) and set(raw.g.round(2)) == set(GS)
    assert raw.q.eq(Q).all() and raw.t.eq(T).all() and raw.layers.eq(LAYERS).all()
    assert np.allclose(raw.W, W) and raw.seed.isin(SEEDS).all()
    assert np.isfinite(raw[["mae_ev", "rmse_ev", "r2", "validation_mae_ev"]]).all().all()
    assert raw.fit_origin.eq("reused-prior-gscan").sum() == 6900
    assert raw.fit_origin.eq("refit-fixed-p-grid").sum() == 21900
    feature_records = []
    for g in GS:
        for seed in SEEDS:
            folder = folder_for(g, seed)
            metadata = json.loads((folder / "features_complete.json").read_text())
            assert metadata["g"] == g and metadata["seed"] == seed and metadata["p"] == .2
            assert sha(folder / "p020_features.npz") == metadata["sha256"]
            with np.load(folder / "p020_features.npz", allow_pickle=False) as z:
                x = z["features"]
                assert x.shape == (2376, 360) and x.dtype == np.float32 and np.isfinite(x).all()
                assert np.abs(x).max() < 1.0002
            assert metadata["maximum_closed_feature_error"] < 5e-5
            decoder = json.loads((folder / "decoder_complete.json").read_text())
            assert decoder["rows"] == 96 and decoder["reused_rows"] == 23
            assert sha(folder / "predictions.npz") == decoder["predictions_sha256"]
            rows = pd.read_csv(folder / "metrics.csv")
            global_rows = raw[np.isclose(raw.g, g) & raw.seed.eq(seed)].reset_index(drop=True)
            pd.testing.assert_frame_equal(rows, global_rows, check_dtype=False, check_exact=False,
                                          atol=1e-12, rtol=1e-12)
            feature_records.append(dict(g=g, seed=seed, sha256=metadata["sha256"],
                 closed_feature_reproduction_error=metadata["maximum_closed_feature_error"]))
    cwrite(HERE / "feature_cache_audit.csv", feature_records)
    keys = ["g", "p", "target", "readout", "decoder"]
    check = raw.groupby(keys).mae_ev.agg(["mean", "std", "count"]).reset_index()
    merged = summary.merge(check, on=keys, validate="one_to_one")
    assert merged["count"].eq(6).all() and np.allclose(merged.mae_mean, merged["mean"], atol=1e-12)
    assert np.allclose(merged.mae_sd, merged["std"], atol=1e-12)
    figure_qa = json.loads((HERE / "figures/figure_qa.json").read_text())
    assert figure_qa["passed"] and len(figure_qa["figures"]) == 8
    for entry in figure_qa["figures"]:
        name = entry["name"]
        readout, decoder = name.split("six_p_")[1].split("_", 1)
        expected = paired if name.startswith("paired_") else summary
        expected = expected[expected.readout.eq(readout) & expected.decoder.eq(decoder)].reset_index(drop=True)
        source = pd.read_csv(HERE / "figures" / f"{name}_source.csv")
        pd.testing.assert_frame_equal(source, expected, check_exact=False, atol=1e-12, rtol=1e-12)
        for extension in ("png", "pdf", "svg"):
            assert (HERE / "figures" / f"{name}.{extension}").stat().st_size > 1000
        with fitz.open(HERE / "figures" / f"{name}.pdf") as document:
            assert len(document) == 1 and "MAE" in document[0].get_text()
            assert "\ufffd" not in document[0].get_text()
    with fitz.open(HERE / "figures/all_six_fixed_p_figures.pdf") as document:
        assert len(document) == 8
    report = dict(passed=True, feature_cases=300, decoder_cases=300, metric_rows=28800,
                  six_seed_cells=4800, fixed_p=list(PS), verified_new_feature_hashes=300,
                  verified_prediction_hashes=300, exactly_reused_predictions=6900,
                  new_fits=21900, verified_figure_sources=8, manual_visual_review_required=True)
    jwrite(HERE / "completion_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
