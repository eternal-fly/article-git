"""Fixed-grid channel tests and full-Q decoder reproduction before launch."""
from grid_common import *
from grid_common import _best_ridge, _refit_ridge
import verify as previous_verify
from evaluate_strict_axes import shard_stem


def main():
    verify_prior()
    records = []
    rng = np.random.default_rng(20260908)
    small = rng.normal(size=(2, 4, 9)).astype(np.float32)
    small[..., 1::3] = 0
    for g in (0., .34, .84, .98):
        bank, _ = backend.suffixes(small, seed=10, g=g, threads=2)
        for p in PS:
            mixed = mix_suffix_bank(bank, p, "tm3", 6)
            density = previous_verify.density_reference(small[:1], 10, g, p)
            error = float(np.max(np.abs(mixed[0]-density)))
            assert error < 1e-5, (g, p, error)
            records.append(dict(check="independent_density_channel", g=g, p=p, error=error))
    data = dataset()
    index = np.asarray([0, 1424, 1896])
    for seed in SEEDS:
        bank, _ = backend.suffixes(data["tokens"][index], seed=seed, g=.84, threads=3)
        with np.load(base.HOP / "features" / f"qrc_post_tm3_suffixes_seed{seed:03d}.npz", allow_pickle=False) as z:
            reference = z["suffixes"][:, index]
        error = float(np.abs(mix_suffix_bank(bank, .2, "tm3", 120)
                                -mix_suffix_bank(reference, .2, "tm3", 120)).max())
        assert error < 3e-5, (seed, error)
        records.append(dict(check="full_Q15_p0.2_anchor", seed=seed, error=error))
    folder = folder_for(.84, 0)
    assert (folder / "decoder_complete.json").exists(), "Run grid_run.py decode-anchor first"
    rows = pd.read_csv(folder / "metrics.csv")
    assert len(rows) == 96 and rows.fit_origin.eq("reused-prior-gscan").sum() == 23
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
        predictions = z["predictions"]
    with np.load(base.HOP / "features/qrc_post_tm3_suffixes_seed000.npz", allow_pickle=False) as z:
        bank = z["suffixes"].astype(np.float32)
    tr, va, te = (data["split"] == i for i in range(3))
    alphas = tuple(base.load_config().payload["decoders"]["ridge_alpha_grid"])
    parity = []
    for i, row in rows.iterrows():
        stem = shard_stem("qrc", 0, row.p, row.readout)
        original = pd.read_csv(stem.with_suffix(".csv"))
        ref = original[original.target.eq(row.target) & original.decoder.eq(row.decoder)].iloc[0]
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            match = np.flatnonzero((z["target_name"] == row.target) & (z["decoder"] == row.decoder))
            assert len(match) == 1
            archive_error = float(np.max(np.abs(predictions[i]-z["predictions"][match[0]])))
        hyperparameters = ("alpha",) if row.decoder == "linear" else ("C", "gamma_scale", "gamma", "epsilon")
        assert all(np.isclose(row[k], ref[k]) for k in hyperparameters), (row.to_dict(), ref.to_dict())
        mae_error = abs(row.mae_ev-ref.mae_ev)
        assert mae_error < 1e-4, (i, row.to_dict(), mae_error)
        live_error = 0.
        if row.decoder == "rbf_svr":
            assert archive_error < 1e-5, (i, archive_error)
        else:
            x = mix_suffix_bank(bank, row.p, row.readout, 120)
            y = data["targets"][:, TARGET_NAMES.index(row.target)].astype(float)
            selected = _best_ridge(x[tr], y[tr], x[va], y[va], alphas)
            assert selected["alpha"] == row.alpha
            live = _refit_ridge(x[tr], y[tr], x[va], y[va], x[te], row.alpha)
            live_error = float(np.max(np.abs(np.asarray(live, np.float32)-predictions[i])))
            assert live_error < 1e-5, (i, live_error)
        parity.append(dict(p=row.p, target=row.target, decoder=row.decoder, readout=row.readout,
                           archive_prediction_error=archive_error, archive_mae_error=mae_error,
                           live_original_ridge_error=live_error, hyperparameters_match=True))
    cwrite(HERE / "anchor_decoder_parity.csv", parity)
    jwrite(HERE / "preflight.json", dict(passed=True, physical_checks=records, decoder_heads=96,
           original_hyperparameters_match=True, maximum_archive_mae_error=max(r["archive_mae_error"] for r in parity),
           maximum_live_ridge_error=max(r["live_original_ridge_error"] for r in parity),
           kernel_sha256=sha(PRIOR / "cpu_kernel.so")))
    freeze()
    print("PASS: all fixed-p density checks, six Q15 p=0.2 anchors, 96 original decoder checks", flush=True)


if __name__ == "__main__":
    main()
