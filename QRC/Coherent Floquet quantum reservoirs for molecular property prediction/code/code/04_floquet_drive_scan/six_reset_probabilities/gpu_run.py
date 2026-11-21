"""Resume only unfinished cases on two GPUs and the original CPU decoders."""
from grid_run import *
from concurrent.futures import wait, FIRST_COMPLETED


def gpu_case(g, seed, gpu, microbatch):
    from gpu_suffix_backend import features
    return features(g, seed, gpu, microbatch)


def manifest():
    preflight = json.loads((HERE / "gpu_preflight.json").read_text())
    assert preflight["passed"]
    assert preflight["backend_sha256"] == sha(HERE / "gpu_suffix_backend.py")
    assert preflight["verifier_sha256"] == sha(HERE / "gpu_verify.py")
    paths = [HERE / x for x in ("gpu_suffix_backend.py", "gpu_verify.py", "gpu_run.py", "GPU_MIGRATION.md")]
    value = dict(version="execution-only-dual-GPU-migration-v1",
        original_protocol_sha256=sha(HERE / "protocol.json"),
        sources={p.name: sha(p) for p in paths},
        physical_protocol="unchanged; exact 20 suffixes; p=0,0.2,0.4,0.6,0.8,1",
        decoders="unchanged original CPU Ridge/RBF-SVR; validation hyperparameters only",
        gpus=[0, 1], microbatch={"0": 32, "1": 16})
    path = HERE / "gpu_execution_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == value, "GPU execution source changed"
    else:
        jwrite(path, value)
        jwrite(HERE / "gpu_migration_checkpoint.json", dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            preserved_features=[str(p.relative_to(HERE)) for p in sorted((HERE/"cases").glob("g*/seed*/features_complete.json"))],
            preserved_decoders=[str(p.relative_to(HERE)) for p in sorted((HERE/"cases").glob("g*/seed*/decoder_complete.json"))]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-workers", type=int, default=18)
    args = parser.parse_args()
    lock = (HERE / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        status("starting", "verifying frozen protocol and dual-GPU migration", started)
        freeze()
        manifest()
        assert json.loads((HERE / "preflight.json").read_text())["passed"]
        ordering = [.84, 0., .98, .5] + [g for g in GS if g not in (.84, 0., .98, .5)]
        missing, ready = [], []
        for g in ordering:
            for seed in SEEDS:
                folder = folder_for(g, seed)
                if (folder / "decoder_complete.json").exists():
                    assert (folder / "features_complete.json").exists()
                elif (folder / "features_complete.json").exists():
                    ready.append((g, seed))
                else:
                    missing.append((g, seed))
        print(f"[resume] missing features={len(missing)}; ready decoders={len(ready)}; GPUs=0,1", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.decoder_workers, mp_context=ctx) as decoders, \
             ProcessPoolExecutor(max_workers=1, mp_context=ctx) as gpu0, \
             ProcessPoolExecutor(max_workers=1, mp_context=ctx) as gpu1:
            pools = {0: gpu0, 1: gpu1}
            decoder_jobs = {decoders.submit(decode, g, seed): (g, seed) for g, seed in ready}
            feature_jobs = {}

            def submit_gpu(gpu):
                if not missing:
                    return
                if shutil.disk_usage(HERE).free < 4*1024**3:
                    raise RuntimeError("Disk free space below 4 GiB; checkpoints preserved")
                g, seed = missing.pop(0)
                future = pools[gpu].submit(gpu_case, g, seed, gpu, 32 if gpu == 0 else 16)
                feature_jobs[future] = (gpu, g, seed)

            submit_gpu(0)
            submit_gpu(1)
            while feature_jobs or decoder_jobs:
                done, _ = wait(list(feature_jobs)+list(decoder_jobs), timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if future in feature_jobs:
                        gpu, g, seed = feature_jobs.pop(future)
                        print(f"[GPU{gpu} case complete] g={g:.2f} seed={seed}", flush=True)
                        decoder_jobs[decoders.submit(decode, g, seed)] = (g, seed)
                        submit_gpu(gpu)
                    else:
                        g, seed = decoder_jobs.pop(future)
                        print("[decoder complete]", result, flush=True)
                status("running", "dual-GPU exact features and original CPU decoders" if feature_jobs
                       else "finishing original CPU decoder queue", started,
                    active_gpu_cases=[dict(gpu=v[0], g=v[1], seed=v[2]) for v in feature_jobs.values()],
                    remaining_unassigned_features=len(missing), pending_decoder_cases=len(decoder_jobs),
                    decoder_workers=args.decoder_workers)
        status("running", "prediction and exact-reuse audit", started)
        aggregate()
        status("running", "paired whole-trajectory intervals", started)
        paired_statistics()
        status("running", "six-curve visualization", started)
        from grid_plot import main as plot
        plot()
        from audit_delivery import main as audit
        audit()
        status("complete", "28800 rows audited; eight six-curve figures exported; manual visual QA pending", started)
        print("[complete] dual-GPU continuation finished", flush=True)
    except BaseException as error:
        status("failed", str(error), started)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
