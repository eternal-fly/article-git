"""Numerical gates before using the grouped implementation on full data."""
import time
from common import *
from dephasing_fast_backend import StatevectorBackend, draw_masks
import torch


def run():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for task, q, t, scale in (("bace", 12, 30, 1.), ("hopping", 15, 20, .5)):
        if task == "bace":
            with np.load(FINAL / "16_dissipative_protocol_migration/cache/tokens/BACE_canonical_general_Q24_T30.npz") as z:
                tokens = z["tokens"][[0, 17], :t, :3*q]
            path = MIG / "cache/qrc_suffix_banks/BACE/canonical_general/Q12/g0.840_W0.500/T30_seed000.npz"
        else:
            tokens = hopping_data()["tokens"][[0, 17]]
            path = HOP / "features/qrc_post_tm3_suffixes_seed000.npz"
        tokens = np.ascontiguousarray(tokens * np.float32(scale))
        engine = StatevectorBackend(q, 0)
        for suffix in (0, t-1):
            for strength in (0., .5, 1.):
                masks = draw_masks(np.random.default_rng(20260905), 2, 4, t, q, strength, suffix)
                before = time.monotonic()
                actual, norm = engine.run(tokens, masks, suffix, tm3=task == "hopping")
                reference, _ = engine.run(tokens, masks, suffix, tm3=task == "hopping", reference=True)
                error = float(np.max(np.abs(actual-reference)))
                assert error < 5e-5 and norm < 1e-4, (task, suffix, strength, error, norm)
                row = dict(task=task, q=q, t=t, suffix=suffix, dephasing=strength,
                           max_feature_error=error, max_norm_error=norm, elapsed_s=time.monotonic()-before)
                if strength == 0:
                    with np.load(path) as z:
                        coherent = z["suffixes"][suffix, [0, 17]]
                    e = float(np.max(np.abs(actual[:, 0] - coherent)))
                    assert e < 5e-5, e
                    row["original_coherent_bank_error"] = e
                rows.append(row)
                print(row, flush=True)
        before = time.monotonic()
        bench = np.repeat(tokens, 2, axis=0)
        masks = draw_masks(np.random.default_rng(42), len(bench), 128, t, q, .5)
        actual, norm = engine.run(bench, masks, 0, tm3=task == "hopping")
        elapsed = time.monotonic()-before
        print(dict(task=task, benchmark_inputs=len(bench), K=128, full_suffix_seconds=elapsed), flush=True)
        rows.append(dict(task=task, q=q, t=t, benchmark_inputs=len(bench), K=128, elapsed_s=elapsed))
        # Row chunking and K slicing must preserve the exact noise realizations.
        whole = draw_masks(np.random.default_rng(11), 5, 128, t, q, .5, 3)
        rng = np.random.default_rng(11)
        chunked = np.concatenate([draw_masks(rng, n, 128, t, q, .5, 3) for n in (2, 3)])
        assert np.array_equal(whole, chunked)
    write_csv(HERE / "dephasing_full_k/backend_parity.csv", rows)
    write_json(HERE / "dephasing_full_k/backend_verified.json", dict(
        passed=True, feature_tolerance=5e-5, norm_tolerance=1e-4,
        source_sha256=hashlib.sha256((HERE / "dephasing_fast_backend.py").read_bytes()).hexdigest(),
        checks="Q12/Q15, original coherent banks, original per-wire noisy evolution, final/TM3, row-batch-independent noise"))
    print("All backend parity checks passed", flush=True)


if __name__ == "__main__":
    run()
