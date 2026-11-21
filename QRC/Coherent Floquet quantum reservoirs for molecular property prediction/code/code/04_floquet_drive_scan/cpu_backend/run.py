"""Resumable CPU scan, original decoder fitting, and paired aggregation."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import multiprocessing
import shutil
import time
import traceback
from a01_core import (_best_ridge, _best_svr, _refit_ridge, _refit_svr, regression_metrics)


def decode(g, seed):
    folder = case_dir(g, seed)
    marker = folder / "decoder_complete.json"
    if marker.exists():
        assert (folder/"metrics.csv").exists() and (folder/"predictions.npz").exists()
        return g, seed, 0.
    started = time.monotonic()
    data = dataset()
    split = data["split"]
    tr, va, te = (split == i for i in range(3))
    targets = data["targets"].astype(np.float64)
    with np.load(folder/"features.npz", allow_pickle=False) as z:
        bank = {k: z[k] for k in z.files}
    cfg = load_config().payload["decoders"]
    alphas = tuple(map(float, cfg["ridge_alpha_grid"]))
    candidates = [(float(c), float(gam), float(e)) for c in cfg["svr_C_grid"]
                  for gam in cfg["svr_gamma_scale_grid"] for e in cfg["svr_epsilon_grid"]]
    rows, predictions = [], []
    for choice in choices().to_dict("records"):
        target, readout, decoder = (choice[k] for k in ("target", "readout", "decoder"))
        y = targets[:, TARGET_NAMES.index(target)]
        for condition, p in (("closed", 0.), ("frozen_p", round(float(choice["p"]), 8))):
            x = np.ascontiguousarray(bank[pkey(p)][:, :120 if readout == "final" else 360])
            if decoder == "linear":
                selected = _best_ridge(x[tr], y[tr], x[va], y[va], alphas)
                pred = _refit_ridge(x[tr], y[tr], x[va], y[va], x[te], selected["alpha"])
            else:
                selected = _best_svr(x[tr], y[tr], x[va], y[va], candidates, 256.)
                pred = _refit_svr(x[tr], y[tr], x[va], y[va], x[te], selected, 256.)
            assert np.isfinite(pred).all()
            rows.append(dict(g=g, seed=seed, q=Q, t=T, layers=LAYERS, W=W, p=p,
                             condition=condition, target=target, readout=readout, decoder=decoder,
                             feature_dim=x.shape[1], **selected, **regression_metrics(y[te], pred)))
            predictions.append(np.asarray(pred, dtype=np.float32))
            jwrite(folder/"decoder_progress.json", dict(g=g, seed=seed, rows_done=len(rows), rows_total=32,
                                                       elapsed_seconds=time.monotonic()-started))
    assert len(rows) == 32
    nwrite(folder/"predictions.npz", predictions=np.stack(predictions),
           target=np.asarray([r["target"] for r in rows]), readout=np.asarray([r["readout"] for r in rows]),
           decoder=np.asarray([r["decoder"] for r in rows]), condition=np.asarray([r["condition"] for r in rows]),
           p=np.asarray([r["p"] for r in rows]), target_values=targets[te].astype(np.float32),
           trajectory_id=data["trajectory_id"][te], endpoint=data["endpoint"][te])
    cwrite(folder/"metrics.csv", rows)
    jwrite(marker, dict(version=VERSION, g=g, seed=seed, rows=32,
                       elapsed_seconds=time.monotonic()-started, predictions_sha256=sha(folder/"predictions.npz")))
    return g, seed, time.monotonic()-started


def anchor_decoder_audit():
    from evaluate_strict_axes import shard_stem
    from a01_core import mix_suffix_bank
    records = []
    data = dataset()
    tr, va, te = (data["split"] == i for i in range(3))
    alphas = tuple(load_config().payload["decoders"]["ridge_alpha_grid"])
    for seed in SEEDS:
        folder = case_dir(.84, seed)
        if not (folder/"decoder_complete.json").exists():
            continue
        metrics = pd.read_csv(folder/"metrics.csv")
        with np.load(folder/"predictions.npz", allow_pickle=False) as z:
            preds = z["predictions"]
        with np.load(HOP/"features"/f"qrc_post_tm3_suffixes_seed{seed:03d}.npz", allow_pickle=False) as z:
            original_bank = z["suffixes"].astype(np.float32)
        for i, row in metrics.iterrows():
            source = shard_stem("qrc", seed, row.p, row.readout)
            ref = pd.read_csv(source.with_suffix(".csv"))
            ref = ref[ref.target.eq(row.target) & ref.decoder.eq(row.decoder)].iloc[0]
            with np.load(source.with_suffix(".npz"), allow_pickle=False) as z:
                index = np.flatnonzero((z["target_name"] == row.target) & (z["decoder"] == row.decoder))
                assert len(index) == 1
                err = float(np.max(np.abs(preds[i] - z["predictions"][index[0]])))
            metric_error = abs(row.mae_ev-ref.mae_ev)
            hyperparameters = ("alpha",) if row.decoder == "linear" else ("C", "gamma_scale", "gamma", "epsilon")
            assert all(np.isclose(row[k], ref[k]) for k in hyperparameters), (seed, row.to_dict(), ref.to_dict())
            live_error = 0.
            if row.decoder == "linear":
                x = mix_suffix_bank(original_bank, row.p, row.readout, 120)
                y = data["targets"][:, TARGET_NAMES.index(row.target)].astype(np.float64)
                live_selection = _best_ridge(x[tr], y[tr], x[va], y[va], alphas)
                assert live_selection["alpha"] == row.alpha
                live = _refit_ridge(x[tr], y[tr], x[va], y[va], x[te], row.alpha)
                live_error = float(np.max(np.abs(np.asarray(live, dtype=np.float32)-preds[i])))
                assert live_error < 1e-5, (seed, row.target, row.readout, live_error)
            else:
                assert err < 1e-5, (seed, row.target, row.readout, err)
            assert metric_error < 1e-4, (seed, row.target, row.readout, row.decoder, err, metric_error)
            records.append(dict(seed=seed, target=row.target, readout=row.readout, decoder=row.decoder,
                                p=row.p, maximum_archive_prediction_error=err, archive_mae_error=metric_error,
                                live_original_ridge_prediction_error=live_error, original_hyperparameters_match=True))
    cwrite(HERE/"anchor_decoder_parity.csv", records)
    return records


def aggregate():
    frames = []
    for g in GS:
        for seed in SEEDS:
            folder = case_dir(g, seed)
            assert (folder/"decoder_complete.json").exists(), str(folder)
            frame = pd.read_csv(folder/"metrics.csv")
            assert len(frame) == 32
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    keys = ["target", "readout", "decoder", "condition", "p", "g"]
    assert len(raw) == 9600 and not raw.duplicated(keys+["seed"]).any()
    summary = raw.groupby(keys).agg(mae_mean=("mae_ev", "mean"), mae_sd=("mae_ev", "std"),
             rmse_mean=("rmse_ev", "mean"), r2_mean=("r2", "mean"),
             validation_mae_mean=("validation_mae_ev", "mean"), n_seeds=("seed", "nunique")).reset_index()
    assert len(summary) == 1600 and summary.n_seeds.eq(6).all()
    cwrite(HERE/"metrics_all_seeds.csv", raw)
    cwrite(HERE/"summary.csv", summary)
    parity = anchor_decoder_audit()
    assert len(parity) == 192
    return summary


def paired_statistics():
    ordered = [(r, d, target) for r in READOUTS for d in DECODERS for target in TARGET_NAMES]
    differences = np.empty((16, 6, 50, 60), dtype=np.float64)
    reference_ids = None
    for gi, g in enumerate(GS):
        for si, seed in enumerate(SEEDS):
            with np.load(case_dir(g, seed)/"predictions.npz", allow_pickle=False) as z:
                ids = np.unique(z["trajectory_id"])
                if reference_ids is None:
                    reference_ids = ids
                assert np.array_equal(reference_ids, ids) and len(ids) == 60
                for ki, (r, d, target) in enumerate(ordered):
                    base = (z["readout"] == r) & (z["decoder"] == d) & (z["target"] == target)
                    truth = z["target_values"][:, TARGET_NAMES.index(target)].astype(float)
                    err = {}
                    for condition in ("closed", "frozen_p"):
                        hit = np.flatnonzero(base & (z["condition"] == condition))
                        assert len(hit) == 1
                        residual = np.abs(z["predictions"][hit[0]].astype(float)-truth)
                        err[condition] = np.array([residual[z["trajectory_id"] == t].mean() for t in ids])
                    differences[ki, si, gi] = err["frozen_p"] - err["closed"]
    nwrite(HERE/"paired_trajectory_errors.npz", differences=differences, trajectory_ids=reference_ids,
           keys=np.asarray(ordered), seeds=np.asarray(SEEDS), g=np.asarray(GS))
    records = []
    rng = np.random.default_rng(20260908)
    seed_counts = rng.multinomial(6, np.full(6, 1/6), size=20000).astype(float)/6
    trajectory_counts = rng.multinomial(60, np.full(60, 1/60), size=20000).astype(float)/60
    for ki, (readout, decoder, target) in enumerate(ordered):
        curve = differences[ki]
        mean = curve.mean(axis=(0, 2))
        boot = np.empty((20000, 50))
        for start in range(0, 20000, 256):
            end = min(start+256, 20000)
            boot[start:end] = np.einsum("bs,sgt,bt->bg", seed_counts[start:end], curve,
                                       trajectory_counts[start:end], optimize=True)
        low, high = np.quantile(boot, [.025, .975], axis=0)
        half = np.quantile(np.max(np.abs(boot-mean), axis=1), .95)
        for gi, g in enumerate(GS):
            records.append(dict(readout=readout, decoder=decoder, target=target, g=g,
                     delta_mae_mean=mean[gi], pointwise_ci_low=low[gi], pointwise_ci_high=high[gi],
                     simultaneous_ci_low=mean[gi]-half, simultaneous_ci_high=mean[gi]+half,
                     n_seeds=6, n_test_trajectories=60, bootstrap_draws=20000))
    cwrite(HERE/"paired_delta_mae.csv", records)


def status(state, stage, started, **extra):
    jwrite(HERE/"status.json", dict(state=state, stage=stage, pid=os.getpid(),
          updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
          completed_features=len(list((HERE/"cases").glob("g*/seed*/features_complete.json"))),
          completed_decoders=len(list((HERE/"cases").glob("g*/seed*/decoder_complete.json"))),
          expected_feature_cases=300, expected_decoder_cases=300, expected_metric_rows=9600, **extra))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "aggregate", "decode-anchor"))
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--decoder-workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=96)
    args = parser.parse_args()
    if args.stage == "decode-anchor":
        print(decode(.84, 0), flush=True)
        print("Anchor parity rows:", len(anchor_decoder_audit()), flush=True)
        return
    if args.stage == "aggregate":
        aggregate(); paired_statistics()
        from plot_results import main as plot
        plot()
        return
    from backend import features, compile_kernel
    lock = (HERE/"runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        freeze()
        verification = json.loads((HERE/"backend_verification.json").read_text())
        assert verification["passed"]
        assert len(anchor_decoder_audit()) >= 32, "Run the anchor decoder parity check before production"
        compile_kernel()
        status("running", "feature extraction and original decoder fitting", started)
        ordering = [.84, 0., .98, .5] + [g for g in GS if g not in (.84, 0., .98, .5)]
        futures = []
        with ProcessPoolExecutor(max_workers=args.decoder_workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for g in ordering:
                for seed in SEEDS:
                    if shutil.disk_usage(HERE).free < 4 * 1024**3:
                        raise RuntimeError("Less than 4 GiB disk remains; resume after space is available")
                    features(g, seed, threads=args.threads, chunk_size=args.chunk_size)
                    futures.append(pool.submit(decode, g, seed))
                    finished = [f for f in futures if f.done()]
                    for f in finished:
                        print("[decoder complete]", f.result(), flush=True)
                    futures = [f for f in futures if f not in finished]
                    status("running", "feature extraction and original decoder fitting", started, current_g=g, current_seed=seed)
            while futures:
                done = [f for f in futures if f.done()]
                for f in done:
                    print("[decoder complete]", f.result(), flush=True)
                futures = [f for f in futures if f not in done]
                status("running", "finishing decoder queue", started)
                if futures:
                    time.sleep(10)
        aggregate()
        status("running", "paired whole-trajectory statistics", started)
        paired_statistics()
        from plot_results import main as plot
        plot()
        status("complete", "9600 rows, anchor parity, paired intervals and plots verified", started, metric_rows=9600)
        print("[complete] 300 feature cases and 9600 decoder rows verified", flush=True)
    except BaseException as error:
        status("failed", str(error), started)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
