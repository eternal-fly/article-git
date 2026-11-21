"""Full-data K=8/32/128 dephasing features and original decoder refits.

Run with --detach for a restartable GPU0/CPU queue; status.json is the live
record. Original manuscript, caches, and representative diagnostics are read-only.
"""
from __future__ import annotations

import argparse
import fcntl
import itertools
import multiprocessing as mp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from common import *
from a01_core import TARGET_NAMES, load_config, regression_metrics
from hopping_controls import tune
from readouts import fit_svr

OUT = HERE / "dephasing_full_k"
K_VALUES = (8, 32, 128)
P_VALUES = (0., .1)
VERSION = "full-data-dephasing-k-v1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.json")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def stamp(**details):
    atomic_json(OUT / "status.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                pid=os.getpid(), **details))


def load_task(task):
    if task == "bace":
        source = FINAL / "16_dissipative_protocol_migration/cache/tokens/BACE_canonical_general_Q24_T30.npz"
        with np.load(source) as z:
            data = {"tokens": np.ascontiguousarray(z["tokens"][:, :30, :36]), "labels": z["labels"]}
        q, t, scale, snapshots = 12, 30, 1., 1
    else:
        source = HOP / "data/processed/a01_windows.npz"
        data = hopping_data()
        q, t, scale, snapshots = 15, 20, .5, 3
    meta = dict(version=VERSION, task=task, q=q, t=t, scale=scale,
                n=len(data["tokens"]), snapshots=snapshots, width=snapshots*q*(q+1)//2,
                g=.84, W=.5, layers=3, interaction_scale=1., source=str(source),
                token_hash=array_hash(data["tokens"]), K_values=list(K_VALUES),
                p_values=list(P_VALUES), backend_sha256=file_hash(HERE / "dephasing_fast_backend.py"),
                rng="pauli-full-k-v1; numpy PCG64 float32 row-major [N,128,T-suffix,Q]; seed hash(q,seed,suffix)")
    return data, meta


def weights(t, p, task):
    if task == "bace":
        w = np.asarray([(1-p)**t] + [p*(1-p)**(t-s) for s in range(1, t)])
        anchor = p
    else:
        w = np.asarray([(1-p)**(t-1)] + [p*(1-p)**(t-1-s) for s in range(1, t)])
        anchor = 0.
    assert np.isclose(w.sum()+anchor, 1., atol=1e-12)
    return w, anchor


def coherent_bank_path(task, seed):
    if task == "bace":
        return MIG / f"cache/qrc_suffix_banks/BACE/canonical_general/Q12/g0.840_W0.500/T30_seed{seed:03d}.npz"
    return HOP / f"features/qrc_post_tm3_suffixes_seed{seed:03d}.npz"


def fingerprint(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


def ensure_meta(folder, meta):
    path = folder / "metadata.json"
    meta = {**meta, "fingerprint": fingerprint(meta)}
    if path.exists():
        old = json.loads(path.read_text())
        if old["fingerprint"] != meta["fingerprint"]:
            raise RuntimeError(f"Refusing incompatible cache: {folder}")
    else:
        atomic_json(path, meta)
    return meta


def prepare_coherent(task, seed):
    data, base = load_task(task)
    folder = OUT / f"features/{task}/lambda0.0/seed{seed:03d}"
    meta = ensure_meta(folder, dict(**base, seed=seed, dephasing=0., exact=True))
    if (folder / "complete.json").exists():
        return folder
    with np.load(coherent_bank_path(task, seed)) as z:
        bank = z["suffixes"].astype(np.float64)
    assert bank.shape == (base["t"], base["n"], base["width"]), bank.shape
    mixed = []
    for p in P_VALUES:
        w, anchor = weights(base["t"], p, task)
        mixed.append(np.tensordot(w, bank, axes=(0, 0)) + anchor)
    features = np.asarray(mixed, dtype=np.float32)[None]
    write_npz(folder / "features.npz", features=features, K=np.asarray([0]),
              p=np.asarray(P_VALUES), standard_error=np.zeros_like(features),
              metadata=np.asarray(json.dumps(meta)))
    atomic_json(folder / "complete.json", dict(fingerprint=meta["fingerprint"], exact=True))
    return folder


def feature_case(task, seed, strength, device, batch_inputs):
    import torch
    from dephasing_fast_backend import StatevectorBackend, draw_masks
    data, base = load_task(task)
    folder = OUT / f"features/{task}/lambda{strength:.1f}/seed{seed:03d}"
    meta = ensure_meta(folder, dict(**base, seed=seed, dephasing=strength, exact=False))
    if (folder / "complete.json").exists():
        return folder
    backend_check = json.loads((OUT / "backend_verified.json").read_text())
    assert backend_check["passed"] and backend_check["source_sha256"] == base["backend_sha256"]
    tokens = np.ascontiguousarray(data["tokens"] * np.float32(base["scale"]))
    n, t, _ = tokens.shape
    q, width = base["q"], base["width"]
    engine = StatevectorBackend(q, seed, device)
    # Per-seed reference check; its noise stream is isolated from production.
    audit_masks = draw_masks(np.random.default_rng(221), 1, 2, t, q, strength)
    actual, norm = engine.run(tokens[:1], audit_masks, tm3=task == "hopping")
    reference, _ = engine.run(tokens[:1], audit_masks, tm3=task == "hopping", reference=True)
    parity = float(np.max(np.abs(actual-reference)))
    assert parity < 5e-5 and norm < 1e-4, (task, seed, parity, norm)
    atomic_json(folder / "parity.json", dict(max_feature_error=parity, norm_error=norm))
    started = time.monotonic()
    for suffix in range(t):
        path = folder / f"suffix{suffix:02d}.npz"
        if path.exists():
            with np.load(path) as z:
                assert str(z["fingerprint"]) == meta["fingerprint"]
                assert z["means"].shape == (3, n, width)
            continue
        means = np.empty((3, n, width), dtype=np.float32)
        variances = np.empty_like(means)
        rng = np.random.default_rng(stable_seed("pauli-full-k-v1", q, seed, suffix))
        max_norm = 0.
        probe = None
        next_update = time.monotonic()
        suffix_started = time.monotonic()
        for start in range(0, n, batch_inputs):
            end = min(start+batch_inputs, n)
            masks = draw_masks(rng, end-start, 128, t, q, strength, suffix)
            chunk = folder / f"chunks/suffix{suffix:02d}/rows{start:06d}-{end:06d}.npz"
            if chunk.exists():
                with np.load(chunk) as z:
                    assert str(z["fingerprint"]) == meta["fingerprint"]
                    means[:, start:end], variances[:, start:end] = z["means"], z["variances"]
                    err = float(z["max_norm_error"])
                    if start == 0:
                        probe = z["trial_probe"].copy()
            else:
                values, err = engine.run(tokens[start:end], masks, suffix, tm3=task == "hopping")
                assert np.isfinite(values).all()
                for j, k in enumerate(K_VALUES):
                    means[j, start:end] = values[:, :k].mean(1, dtype=np.float64)
                    variances[j, start:end] = values[:, :k].var(1, ddof=1, dtype=np.float64)
                if start == 0:
                    probe = values[:2].copy()
                write_npz(chunk, means=means[:, start:end], variances=variances[:, start:end],
                          trial_probe=probe if start == 0 else np.empty(0, dtype=np.float32),
                          max_norm_error=np.asarray(err), fingerprint=np.asarray(meta["fingerprint"]))
            max_norm = max(max_norm, err)
            assert max_norm < 2e-4
            if time.monotonic() >= next_update or end == n:
                stamp(state="running", stage="features", task=task, seed=seed, dephasing=strength,
                      suffix=suffix, total_suffixes=t, samples_done=end, samples_total=n,
                      case_elapsed_seconds=time.monotonic()-started)
                print(f"[features {task} lambda={strength} seed={seed}] suffix={suffix+1}/{t} inputs={end}/{n}", flush=True)
                next_update = time.monotonic()+30
        write_npz(path, means=means, variances=variances, trial_probe=probe,
                  K=np.asarray(K_VALUES), fingerprint=np.asarray(meta["fingerprint"]),
                  max_norm_error=np.asarray(max_norm), elapsed_seconds=np.asarray(time.monotonic()-suffix_started))
    mixed = np.zeros((3, 2, n, width), dtype=np.float64)
    estimator_variance = np.zeros_like(mixed)
    weight_values = [weights(t, p, task) for p in P_VALUES]
    for pi, (_, anchor) in enumerate(weight_values):
        mixed[:, pi] += anchor
    for suffix in range(t):
        with np.load(folder / f"suffix{suffix:02d}.npz") as z:
            assert str(z["fingerprint"]) == meta["fingerprint"]
            mean, variance = z["means"], z["variances"]
            for pi, (w, _) in enumerate(weight_values):
                mixed[:, pi] += w[suffix] * mean
                estimator_variance[:, pi] += w[suffix]**2 * variance / np.asarray(K_VALUES)[:, None, None]
    write_npz(folder / "features.npz", features=mixed.astype(np.float32),
              standard_error=np.sqrt(estimator_variance).astype(np.float32),
              K=np.asarray(K_VALUES), p=np.asarray(P_VALUES), metadata=np.asarray(json.dumps(meta)))
    atomic_json(folder / "complete.json", dict(fingerprint=meta["fingerprint"], elapsed_seconds=time.monotonic()-started))
    return folder


def fit_hopping(x, data, folder, identifiers):
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    choices, screen = tune(x, data["targets"].astype(float), data["split"])
    write_csv(folder / "validation_screen.csv", screen)
    split, targets = data["split"], data["targets"].astype(float)
    trainval, test = split != 2, split == 2
    scaler = StandardScaler().fit(x[trainval])
    xt, xe = scaler.transform(x[trainval]), scaler.transform(x[test])
    predictions, models, rows = [], {}, []
    for choice in choices:
        y = targets[:, TARGET_NAMES.index(choice["target"])]
        if choice["decoder"] == "linear":
            model = Ridge(alpha=choice["alpha"])
        else:
            model = SVR(C=choice["C"], gamma=choice["gamma"], epsilon=choice["epsilon"], cache_size=256)
        model.fit(xt, y[trainval])
        pred = model.predict(xe)
        rows.append(dict(**identifiers, **choice, **regression_metrics(y[test], pred)))
        predictions.append(pred)
        models[(choice["target"], choice["decoder"])] = model
    joblib.dump(dict(scaler=scaler, models=models), folder / "decoder_models.joblib", compress=3)
    write_npz(folder / "predictions.npz", predictions=np.asarray(predictions),
              decoder=np.asarray([c["decoder"] for c in choices]), target=np.asarray([c["target"] for c in choices]),
              test_indices=np.flatnonzero(test), target_values=targets[test],
              trajectory_id=data["trajectory_id"][test], endpoint=data["endpoint"][test])
    write_csv(folder / "metrics.csv", rows)


def decoder_job(job):
    import joblib
    task, strength, seed, k, p = job
    source = OUT / f"features/{task}/lambda{strength:.1f}/seed{seed:03d}"
    folder = OUT / f"decoders/{task}/lambda{strength:.1f}/seed{seed:03d}/K{k:03d}/p{p:.1f}"
    data, meta = load_task(task)
    source_meta = json.loads((source / "metadata.json").read_text())
    complete = folder / "complete.json"
    if complete.exists():
        assert json.loads(complete.read_text())["source_fingerprint"] == source_meta["fingerprint"]
        return f"cached {job}"
    with np.load(source / "features.npz") as z:
        assert json.loads(str(z["metadata"]))["fingerprint"] == source_meta["fingerprint"]
        ki, pi = list(z["K"]).index(k), list(z["p"]).index(p)
        x = z["features"][ki, pi].copy()
    identifiers = dict(task=task, dephasing=strength, seed=seed, K=k, p=p, q=meta["q"], t=meta["t"])
    if task == "bace":
        split = json.loads((FINAL / f"16_dissipative_protocol_migration/cache/splits/BACE/seed{seed:03d}.json").read_text())
        metric, artifact = fit_svr(x, data["labels"], split, seed=seed)
        write_csv(folder / "validation_screen.csv", artifact["screen"])
        joblib.dump(dict(scaler=artifact["scaler"], model=artifact["model"]), folder / "decoder_model.joblib", compress=3)
        write_npz(folder / "predictions.npz", test_scores=artifact["test_scores"],
                  test_indices=artifact["test_indices"], labels=data["labels"][artifact["test_indices"]])
        write_csv(folder / "metrics.csv", [dict(**identifiers, readout="final", decoder="rbf_svr", **metric)])
    else:
        for readout, features in (("final", x[:, :120]), ("tm3", x)):
            fit_hopping(features, data, folder / readout, dict(**identifiers, readout=readout))
    atomic_json(complete, dict(source_fingerprint=source_meta["fingerprint"], job=list(job)))
    return f"fitted {job}"


def summarize_available():
    files = sorted((OUT / "decoders").glob("**/metrics.csv"))
    if files:
        frames = []
        for path in files:
            # Do not expose a partially completed decoder cell.
            parent = path.parent if "bace" in path.parts else path.parent.parent
            if (parent / "complete.json").exists():
                frames.append(pd.read_csv(path))
        if frames:
            write_csv(OUT / "metrics_all_seeds.csv", pd.concat(frames, ignore_index=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=("bace", "hopping", "all"), default="all")
    parser.add_argument("--batch-inputs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--decoders-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != "--detach"]]
        with (OUT / "run.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(OUT / "launcher.json", dict(pid=child.pid, command=command, log=str(OUT / "run.log")))
        print(json.dumps(dict(pid=child.pid, log=str(OUT / "run.log")), ensure_ascii=False))
        return
    lock = (OUT / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = ("bace", "hopping") if args.task == "all" else (args.task,)
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not args.decoders_only:
        torch.cuda.set_device(args.device)
    stamp(state="running", stage="preparation", tasks=list(tasks))
    futures, submitted, missing = [], set(), []
    pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"))
    def submit(task, strength, seed, ks):
        if args.features_only:
            return
        for k, p in itertools.product(ks, P_VALUES):
            job = (task, strength, seed, k, p)
            if job not in submitted:
                futures.append(pool.submit(decoder_job, job))
                submitted.add(job)
    def harvest():
        for future in list(futures):
            if future.done():
                print(f"[decoder] {future.result()}", flush=True)
                futures.remove(future)
        summarize_available()
    try:
        for task in tasks:
            for seed in SEEDS:
                prepare_coherent(task, seed)
                submit(task, 0., seed, (0,))
        for task in tasks:
            for seed in SEEDS:
                for strength in (.5, 1.):
                    if args.decoders_only:
                        folder = OUT / f"features/{task}/lambda{strength:.1f}/seed{seed:03d}"
                        if not (folder / "complete.json").exists():
                            missing.append(str(folder))
                            continue
                    else:
                        feature_case(task, seed, strength, args.device, args.batch_inputs)
                    submit(task, strength, seed, K_VALUES)
                    harvest()
        stamp(state="running", stage="decoders", pending=len(futures))
        for future in as_completed(futures):
            print(f"[decoder] {future.result()}", flush=True)
            summarize_available()
        pool.shutdown()
        summarize_available()
        if not missing and not args.features_only:
            stamp(state="running", stage="prediction_verification", tasks=list(tasks))
            subprocess.run([sys.executable, str(HERE / "analyze_dephasing_full_k.py"),
                            "--task", args.task, "--require-complete"], check=True, cwd=ROOT)
        verified = not missing and not args.features_only
        stamp(state="partial" if missing else "training_complete",
              stage="verified" if verified else "features_only" if args.features_only else "missing_features",
              tasks=list(tasks), decoder_jobs=len(submitted), missing_feature_cells=missing,
              features_only=args.features_only, prediction_verification_passed=verified)
        print("Requested training and prediction verification completed." if verified
              else "Queue ended without full prediction verification; inspect status.json.", flush=True)
    except BaseException as exc:
        stamp(state="failed", error=repr(exc))
        pool.shutdown(wait=False, cancel_futures=True)
        raise


if __name__ == "__main__":
    main()
