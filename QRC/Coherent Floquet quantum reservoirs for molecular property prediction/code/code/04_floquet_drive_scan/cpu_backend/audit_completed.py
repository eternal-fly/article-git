"""Independent post-run completeness and prediction-to-figure audit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import fitz


def main():
    protocol = freeze()
    feature_audit = json.loads((HERE / "feature_cache_audit.json").read_text())
    assert feature_audit["passed"] and feature_audit["matched_hashes"] == 300
    raw = pd.read_csv(HERE / "metrics_all_seeds.csv")
    summary = pd.read_csv(HERE / "summary.csv")
    paired = pd.read_csv(HERE / "paired_delta_mae.csv")
    anchors = pd.read_csv(HERE / "anchor_decoder_parity.csv")
    assert len(raw) == 9600 and len(summary) == 1600 and len(paired) == 800
    assert len(anchors) == 192 and anchors.original_hyperparameters_match.all()
    assert summary.n_seeds.eq(6).all() and paired.n_seeds.eq(6).all()
    assert set(raw.g.round(2)) == set(GS) and set(raw.seed) == set(SEEDS)
    assert raw.q.eq(Q).all() and raw.t.eq(T).all() and raw.layers.eq(LAYERS).all()
    assert np.allclose(raw.W, W)
    assert np.isfinite(raw[["mae_ev", "rmse_ev", "r2", "validation_mae_ev"]]).all().all()
    keys = ["g", "seed", "target", "readout", "decoder", "condition"]
    assert not raw.duplicated(keys).any()
    fixed = choices().set_index(["target", "readout", "decoder"]).p
    for row in raw.itertuples():
        expected = 0.0 if row.condition == "closed" else fixed.loc[(row.target, row.readout, row.decoder)]
        assert np.isclose(row.p, expected)
        assert row.feature_dim == (120 if row.readout == "final" else 360)
    checks = []
    data = dataset()
    test = data["split"] == 2
    truth = data["targets"][test].astype(np.float64)
    reference_ids = data["trajectory_id"][test]
    unique_ids, counts = np.unique(reference_ids, return_counts=True)
    assert len(unique_ids) == 60 and np.all(counts == 8)
    for g in GS:
        for seed in SEEDS:
            folder = case_dir(g, seed)
            metrics = pd.read_csv(folder / "metrics.csv")
            aggregate_case = raw[np.isclose(raw.g, g) & raw.seed.eq(seed)].reset_index(drop=True)
            pd.testing.assert_frame_equal(metrics, aggregate_case, check_exact=False,
                                          check_dtype=False, atol=1e-12, rtol=1e-12)
            marker = json.loads((folder / "decoder_complete.json").read_text())
            assert marker["predictions_sha256"] == sha(folder / "predictions.npz")
            assert marker["rows"] == 32 and len(metrics) == 32
            with np.load(folder / "predictions.npz", allow_pickle=False) as z:
                assert z["predictions"].shape == (32, 480)
                assert np.array_equal(z["trajectory_id"], reference_ids)
                assert np.array_equal(z["endpoint"], data["endpoint"][test])
                assert np.array_equal(z["target_values"], truth.astype(np.float32))
                for i, row in metrics.iterrows():
                    for field in ("target", "readout", "decoder", "condition"):
                        assert z[field][i] == row[field]
                    assert np.isclose(z["p"][i], row.p)
                    residual = np.abs(z["predictions"][i].astype(float) - truth[:, TARGET_NAMES.index(row.target)])
                    mae = float(residual.mean())
                    trajectory_mae = float(np.mean([residual[reference_ids == tid].mean() for tid in unique_ids]))
                    error = abs(mae - row.mae_ev)
                    assert np.isclose(mae, row.mae_ev, atol=1e-5, rtol=1e-6), (g, seed, i, error)
                    assert abs(mae - trajectory_mae) < 1e-10
                    checks.append(dict(g=g, seed=seed, target=row.target, readout=row.readout,
                                       decoder=row.decoder, condition=row.condition,
                                       reconstructed_mae=mae, archived_mae=row.mae_ev,
                                       prediction_rounding_mae_error=error))
    cwrite(HERE / "prediction_metric_audit.csv", checks)
    group = ["target", "readout", "decoder", "condition", "p", "g"]
    recomputed = raw.groupby(group).mae_ev.agg(["mean", "std"]).reset_index()
    merged = summary.merge(recomputed, on=group, validate="one_to_one")
    assert np.allclose(merged.mae_mean, merged["mean"], atol=1e-12)
    assert np.allclose(merged.mae_sd, merged["std"], atol=1e-12)
    contrast_keys = ["target", "readout", "decoder", "g"]
    means = summary.pivot(index=contrast_keys, columns="condition", values="mae_mean")
    means["reconstructed_delta"] = means["frozen_p"] - means["closed"]
    contrasts = paired.merge(means[["reconstructed_delta"]].reset_index(),
                             on=contrast_keys, validate="one_to_one")
    assert np.allclose(contrasts.delta_mae_mean, contrasts.reconstructed_delta,
                       atol=1e-5, rtol=1e-6)
    with np.load(HERE / "paired_trajectory_errors.npz", allow_pickle=False) as z:
        assert z["differences"].shape == (16, 6, 50, 60)
        assert np.array_equal(z["trajectory_ids"], unique_ids)
        assert np.array_equal(z["seeds"], SEEDS) and np.allclose(z["g"], GS)
        for i, (readout, decoder, target) in enumerate(z["keys"]):
            cell = paired[paired.readout.eq(readout) & paired.decoder.eq(decoder)
                          & paired.target.eq(target)].sort_values("g")
            assert np.allclose(z["differences"][i].mean(axis=(0, 2)), cell.delta_mae_mean,
                               atol=1e-12, rtol=1e-12)
    for prefix in ("pointwise", "simultaneous"):
        assert (paired[f"{prefix}_ci_low"] <= paired[f"{prefix}_ci_high"]).all()
    figures = HERE / "figures"
    qa = json.loads((figures / "figure_qa.json").read_text())
    assert qa["passed"] and len(qa["figures"]) == 8
    figure_checks = []
    for item in qa["figures"]:
        name = item["name"]
        for extension in ("pdf", "svg", "png"):
            assert (figures / f"{name}.{extension}").stat().st_size > 1000
        csv = pd.read_csv(figures / f"{name}_source.csv")
        expected = paired if name.startswith("paired_") else summary
        readout, decoder = name.split("_g_")[1].split("_", 1)
        expected = expected[expected.readout.eq(readout) & expected.decoder.eq(decoder)].reset_index(drop=True)
        pd.testing.assert_frame_equal(csv, expected, check_exact=False, rtol=1e-12, atol=1e-12)
        with fitz.open(figures / f"{name}.pdf") as document:
            assert len(document) == 1
            text = document[0].get_text()
            assert "MAE" in text and "\ufffd" not in text
            fonts = document[0].get_fonts()
            assert fonts and all(font[1] in ("ttf", "otf") for font in fonts)
            figure_checks.append(dict(name=name, editable_text=True, embedded_fonts=True,
                                      source_data_verified=True, pdf_sha256=sha(figures / f"{name}.pdf")))
    with fitz.open(figures / "all_hopping_gscan_figures.pdf") as document:
        assert len(document) == 8
    report = dict(passed=True, metric_rows=len(raw), complete_six_seed_cells=len(summary),
                  paired_cells=len(paired), verified_anchor_rows=len(anchors),
                  verified_feature_cache_hashes=feature_audit["matched_hashes"],
                  fixed_p_at_all_g=True, original_source_hashes_verified=len(protocol["source_sha256"]),
                  max_prediction_rounding_mae_error=max(row["prediction_rounding_mae_error"] for row in checks),
                  test_windows=480, test_trajectories=60, figures=figure_checks,
                  manual_visual_review_required=True)
    jwrite(HERE / "completion_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
