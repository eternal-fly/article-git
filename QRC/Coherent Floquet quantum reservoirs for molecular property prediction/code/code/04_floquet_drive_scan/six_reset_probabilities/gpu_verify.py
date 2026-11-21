"""Preflight on both requested GPUs, before creating any production cache."""
from gpu_suffix_backend import *
from datetime import datetime, timezone


def main():
    tokens = dataset()["tokens"]
    records = []
    for g, seed in ((0., 0), (.42, 50), (.84, 10), (.98, 50)):
        indices = [0, 1424, 1896]
        source = tokens[indices]
        reference, _ = backend.suffixes(source, seed=seed, g=g, threads=3)
        for gpu in (0, 1):
            bank, metadata = suffixes(source, seed=seed, g=g, gpu=gpu, microbatch=2)
            error = float(np.abs(bank-reference).max())
            mixed_error = float(np.abs(mix_suffix_bank(bank, .2, "tm3", 120)-
                                      mix_suffix_bank(reference, .2, "tm3", 120)).max())
            assert error < 3e-5 and mixed_error < 1e-5, (gpu, g, error, mixed_error)
            records.append(dict(gpu=gpu, g=g, seed=seed, suffix_error=error,
                mixed_error=mixed_error, **{k: metadata[k] for k in ("maximum_norm_error", "elapsed_seconds")}))
            print(records[-1], flush=True)
    # Production-sized chunk, including row ordering and GPU batch subdivision.
    reference, _ = backend.suffixes(tokens[:96], seed=20, g=.46, threads=8)
    for gpu, microbatch in ((0, 32), (1, 16)):
        bank, metadata = suffixes(tokens[:96], seed=20, g=.46, gpu=gpu, microbatch=microbatch)
        error = float(np.abs(bank-reference).max())
        assert error < 3e-5, (gpu, error)
        records.append(dict(gpu=gpu, g=.46, seed=20, rows=96, suffix_error=error,
                            elapsed_seconds=metadata["elapsed_seconds"]))
        print(records[-1], flush=True)
    jwrite(HERE / "gpu_preflight.json", dict(passed=True, records=records,
        backend_sha256=sha(HERE / "gpu_suffix_backend.py"),
        verifier_sha256=sha(HERE / "gpu_verify.py"),
        created_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
