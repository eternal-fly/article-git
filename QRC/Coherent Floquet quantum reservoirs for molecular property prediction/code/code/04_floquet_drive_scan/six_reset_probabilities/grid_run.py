"""Resumable six-p decoder extension; prior results are strictly read-only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_common import *
from grid_common import _best_ridge, _best_svr, _refit_ridge, _refit_svr
import argparse
import fcntl
import multiprocessing
import shutil
import time
import traceback
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor


def decode(g, seed):
    folder = folder_for(g, seed)
    if (folder / "decoder_complete.json").exists():
        assert (folder / "metrics.csv").exists() and (folder / "predictions.npz").exists()
        return g, seed, 0.
    started = time.monotonic()
    data = dataset()
    tr, va, te = (data["split"] == i for i in range(3))
    yall = data["targets"].astype(np.float64)
    old = base.case_dir(g, seed)
    previous = pd.read_csv(old / "metrics.csv")
    assert sha(old / "predictions.npz") == json.loads((old / "decoder_complete.json").read_text())["predictions_sha256"]
    with np.load(old / "predictions.npz", allow_pickle=False) as z:
        old_predictions = z["predictions"]
        assert np.array_equal(z["trajectory_id"], data["trajectory_id"][te])
        assert np.array_equal(z["target_values"], yall[te].astype(np.float32))
    with np.load(old / "features.npz", allow_pickle=False) as z:
        bank = {p: z[pkey(p)] for p in PS if p != .2}
    with np.load(folder / "p020_features.npz", allow_pickle=False) as z:
        bank[.2] = z["features"]
    cfg = base.load_config().payload["decoders"]
    alphas = tuple(map(float, cfg["ridge_alpha_grid"]))
    candidates = [(float(c), float(g), float(e)) for c in cfg["svr_C_grid"]
                  for g in cfg["svr_gamma_scale_grid"] for e in cfg["svr_epsilon_grid"]]
    rows, predictions = [], []
    for p in PS:
        for readout in READOUTS:
            block = folder / "blocks" / f"{pkey(p)}_{readout}.npz"
            marker = block.with_suffix(".json")
            if block.exists() and marker.exists():
                report = json.loads(marker.read_text())
                assert report["p"] == p and report["readout"] == readout
                assert sha(block) == report["sha256"]
                with np.load(block, allow_pickle=False) as z:
                    block_predictions = list(z["predictions"])
                block_rows = report["rows"]
                assert len(block_rows) == len(block_predictions) == 8
            else:
                block_rows, block_predictions = [], []
                x = np.ascontiguousarray(bank[p][:, :120 if readout == "final" else 360])
                for decoder in DECODERS:
                    for target in TARGET_NAMES:
                        matched = previous[np.isclose(previous.p, p) & previous.readout.eq(readout)
                                           & previous.decoder.eq(decoder) & previous.target.eq(target)]
                        assert len(matched) <= 1
                        if len(matched):
                            index = matched.index[0]
                            row = {k: v for k, v in matched.iloc[0].to_dict().items() if pd.notna(v)}
                            pred = old_predictions[index]
                            origin = "reused-prior-gscan"
                        else:
                            y = yall[:, TARGET_NAMES.index(target)]
                            if decoder == "linear":
                                selected = _best_ridge(x[tr], y[tr], x[va], y[va], alphas)
                                pred = _refit_ridge(x[tr], y[tr], x[va], y[va], x[te], selected["alpha"])
                            else:
                                selected = _best_svr(x[tr], y[tr], x[va], y[va], candidates, 256.)
                                pred = _refit_svr(x[tr], y[tr], x[va], y[va], x[te], selected, 256.)
                            row = dict(g=g, seed=seed, q=Q, t=T, layers=LAYERS, W=W, p=p,
                                       target=target, readout=readout, decoder=decoder,
                                       feature_dim=x.shape[1], **selected, **regression_metrics(y[te], pred))
                            origin = "refit-fixed-p-grid"
                        assert np.isfinite(pred).all()
                        row.update(condition="closed" if p == 0 else "fixed_p", fit_origin=origin)
                        block_rows.append(row)
                        block_predictions.append(np.asarray(pred, dtype=np.float32))
                        jwrite(folder / "decoder_progress.json", dict(g=g, seed=seed,
                               rows_done=len(rows)+len(block_rows), rows_total=96,
                               elapsed_seconds=time.monotonic()-started))
                nwrite(block, predictions=np.stack(block_predictions))
                jwrite(marker, dict(p=p, readout=readout, rows=block_rows, sha256=sha(block)))
            rows.extend(block_rows)
            predictions.extend(block_predictions)
    assert len(rows) == 96 and sum(r["fit_origin"] == "reused-prior-gscan" for r in rows) == 23
    cwrite(folder / "metrics.csv", rows)
    nwrite(folder / "predictions.npz", predictions=np.stack(predictions),
           target=np.asarray([r["target"] for r in rows]), readout=np.asarray([r["readout"] for r in rows]),
           decoder=np.asarray([r["decoder"] for r in rows]), p=np.asarray([r["p"] for r in rows]),
           target_values=yall[te].astype(np.float32), trajectory_id=data["trajectory_id"][te], endpoint=data["endpoint"][te])
    jwrite(folder / "decoder_complete.json", dict(version=VERSION, g=g, seed=seed, rows=96,
           reused_rows=23, new_rows=73, elapsed_seconds=time.monotonic()-started,
           predictions_sha256=sha(folder / "predictions.npz")))
    return g, seed, time.monotonic()-started


def aggregate():
    frames, errors, reuse_checks = [], [], []
    data = dataset()
    te = data["split"] == 2
    truth = data["targets"][te].astype(float)
    for g in GS:
        for seed in SEEDS:
            folder = folder_for(g, seed)
            assert (folder / "decoder_complete.json").exists()
            rows = pd.read_csv(folder / "metrics.csv")
            assert len(rows) == 96
            with np.load(folder / "predictions.npz", allow_pickle=False) as z:
                assert z["predictions"].shape == (96, 480)
                assert np.array_equal(z["trajectory_id"], data["trajectory_id"][te])
                assert np.array_equal(z["endpoint"], data["endpoint"][te])
                predictions = z["predictions"]
                for i, row in rows.iterrows():
                    assert z["target"][i] == row.target and z["readout"][i] == row.readout
                    assert z["decoder"][i] == row.decoder and np.isclose(z["p"][i], row.p)
                    mae = np.abs(predictions[i].astype(float)-truth[:, TARGET_NAMES.index(row.target)]).mean()
                    error = abs(mae-row.mae_ev)
                    assert np.isclose(mae, row.mae_ev, atol=1e-5, rtol=1e-6)
                    errors.append(error)
            old = pd.read_csv(base.case_dir(g, seed) / "metrics.csv")
            with np.load(base.case_dir(g, seed) / "predictions.npz", allow_pickle=False) as z:
                old_predictions = z["predictions"]
            for i, row in rows[rows.fit_origin.eq("reused-prior-gscan")].iterrows():
                match = old[np.isclose(old.p, row.p) & old.readout.eq(row.readout)
                            & old.decoder.eq(row.decoder) & old.target.eq(row.target)]
                assert len(match) == 1
                ref = match.iloc[0]
                assert np.isclose(row.mae_ev, ref.mae_ev, atol=1e-12, rtol=0)
                hyperparameters = ("alpha",) if row.decoder == "linear" else ("C", "gamma", "gamma_scale", "epsilon")
                assert all(np.isclose(row[k], ref[k], atol=1e-12, rtol=1e-12) for k in hyperparameters)
                assert np.array_equal(predictions[i], old_predictions[match.index[0]])
                reuse_checks.append(dict(g=g, seed=seed, p=row.p, target=row.target,
                                         readout=row.readout, decoder=row.decoder, exact=True))
            frames.append(rows)
    raw = pd.concat(frames, ignore_index=True)
    keys = ["g", "p", "target", "readout", "decoder"]
    assert len(raw) == 28800 and not raw.duplicated(keys+["seed"]).any()
    assert len(reuse_checks) == 6900
    summary = raw.groupby(keys).agg(mae_mean=("mae_ev", "mean"), mae_sd=("mae_ev", "std"),
        rmse_mean=("rmse_ev", "mean"), r2_mean=("r2", "mean"),
        validation_mae_mean=("validation_mae_ev", "mean"), n_seeds=("seed", "nunique")).reset_index()
    assert len(summary) == 4800 and summary.n_seeds.eq(6).all()
    cwrite(HERE / "metrics_all_seeds.csv", raw)
    cwrite(HERE / "summary.csv", summary)
    cwrite(HERE / "exact_reuse_audit.csv", reuse_checks)
    jwrite(HERE / "numerical_audit.json", dict(passed=True, rows=28800, six_seed_cells=4800,
            exact_reused_predictions=6900, maximum_prediction_rounding_mae_error=max(errors)))


def paired_statistics():
    ordered = [(r, d, target, p) for r in READOUTS for d in DECODERS for target in TARGET_NAMES for p in PS[1:]]
    differences = np.empty((80, 6, 50, 60), dtype=np.float64)
    ids_reference = None
    for gi, g in enumerate(GS):
        for si, seed in enumerate(SEEDS):
            with np.load(folder_for(g, seed)/"predictions.npz", allow_pickle=False) as z:
                ids = np.unique(z["trajectory_id"])
                if ids_reference is None:
                    ids_reference = ids
                assert np.array_equal(ids_reference, ids) and len(ids) == 60
                errors = {}
                for i in range(96):
                    target = str(z["target"][i])
                    truth = z["target_values"][:, TARGET_NAMES.index(target)].astype(float)
                    residual = np.abs(z["predictions"][i].astype(float)-truth)
                    errors[(str(z["readout"][i]), str(z["decoder"][i]), target, round(float(z["p"][i]), 2))] = np.array([
                        residual[z["trajectory_id"] == tid].mean() for tid in ids])
                for ki, (r, d, target, p) in enumerate(ordered):
                    differences[ki, si, gi] = errors[(r, d, target, p)] - errors[(r, d, target, 0.)]
    nwrite(HERE/"paired_trajectory_errors.npz", differences=differences, keys=np.asarray(ordered),
           g=np.asarray(GS), seeds=np.asarray(SEEDS), trajectory_ids=ids_reference)
    rng = np.random.default_rng(20260908)
    seed_counts = rng.multinomial(6, np.full(6, 1/6), size=20000).astype(float)/6
    trajectory_counts = rng.multinomial(60, np.full(60, 1/60), size=20000).astype(float)/60
    rows = []
    for ki, (r, d, target, p) in enumerate(ordered):
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
            rows.append(dict(readout=r, decoder=d, target=target, p=p, g=g,
                delta_mae_mean=mean[gi], pointwise_ci_low=low[gi], pointwise_ci_high=high[gi],
                simultaneous_ci_low=mean[gi]-half, simultaneous_ci_high=mean[gi]+half,
                n_seeds=6, n_test_trajectories=60, bootstrap_draws=20000))
        if (ki+1) % 10 == 0:
            print(f"[paired intervals] {ki+1}/80", flush=True)
    cwrite(HERE / "paired_delta_mae.csv", rows)


def status(state, stage, started, **extra):
    jwrite(HERE / "status.json", dict(state=state, stage=stage, pid=os.getpid(),
        updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        completed_features=len(list((HERE/"cases").glob("g*/seed*/features_complete.json"))),
        completed_decoders=len(list((HERE/"cases").glob("g*/seed*/decoder_complete.json"))),
        expected_cases=300, expected_metric_rows=28800, **extra))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "aggregate", "decode-anchor"))
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--decoder-workers", type=int, default=10)
    args = parser.parse_args()
    if args.stage == "decode-anchor":
        from grid_features import features
        features(.84, 0, threads=args.threads)
        print(decode(.84, 0), flush=True)
        return
    if args.stage == "aggregate":
        aggregate(); paired_statistics()
        from grid_plot import main as plot
        plot()
        return
    from grid_features import features
    lock = (HERE / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        freeze()
        assert json.loads((HERE / "preflight.json").read_text())["passed"]
        status("running", "missing p=0.2 features and fixed-p decoder fitting", started)
        ordering = [.84, 0., .98, .5] + [g for g in GS if g not in (.84, 0., .98, .5)]
        futures = []
        with ProcessPoolExecutor(max_workers=args.decoder_workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for g in ordering:
                for seed in SEEDS:
                    if shutil.disk_usage(HERE).free < 4*1024**3:
                        raise RuntimeError("Disk free space below 4 GiB; resume after space is available")
                    features(g, seed, threads=args.threads)
                    futures.append(pool.submit(decode, g, seed))
                    done = [f for f in futures if f.done()]
                    for f in done:
                        print("[decoder complete]", f.result(), flush=True)
                    futures = [f for f in futures if f not in done]
                    status("running", "missing p=0.2 features and fixed-p decoder fitting", started, current_g=g, current_seed=seed)
            while futures:
                done = [f for f in futures if f.done()]
                for f in done:
                    print("[decoder complete]", f.result(), flush=True)
                futures = [f for f in futures if f not in done]
                status("running", "finishing fixed-p decoder queue", started)
                if futures:
                    time.sleep(10)
        status("running", "prediction and exact-reuse audit", started)
        aggregate()
        status("running", "paired whole-trajectory intervals", started)
        paired_statistics()
        status("running", "six-curve visualization", started)
        from grid_plot import main as plot
        plot()
        status("complete", "28800 rows verified and eight six-curve figures exported", started)
        print("[complete] Fixed p grid finished", flush=True)
    except BaseException as error:
        status("failed", str(error), started)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
