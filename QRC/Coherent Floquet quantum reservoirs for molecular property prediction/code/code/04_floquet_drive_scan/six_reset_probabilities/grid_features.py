"""Only p=0.2 requires new quantum features; all suffixes remain exact."""
from grid_common import *
import time


def features(g, seed, threads=20, chunk_size=96):
    folder = folder_for(g, seed)
    output = folder / "p020_features.npz"
    marker = folder / "features_complete.json"
    if marker.exists():
        assert output.exists()
        return output
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with np.load(base.case_dir(g, seed) / "features.npz", allow_pickle=False) as z:
        reference = z[pkey(0.)]
    tokens = dataset()["tokens"]
    result = np.empty((len(tokens), 360), np.float32)
    records = []
    if g == .84:
        with np.load(base.HOP / "features" / f"qrc_post_tm3_suffixes_seed{seed:03d}.npz",
                     allow_pickle=False) as z:
            bank = z["suffixes"].astype(np.float32)
        error = float(np.abs(bank[0] - reference).max())
        assert error < 5e-5
        result[:] = mix_suffix_bank(bank, .2, "tm3", 120)
        origin = "original-g0.84-full-suffix-bank"
        records.append(dict(closed_feature_error=error, maximum_norm_error=0.))
    else:
        origin = "verified-CPU-full-suffix-mixture"
        for start in range(0, len(tokens), chunk_size):
            end = min(start + chunk_size, len(tokens))
            chunk = folder / "chunks" / f"rows{start:04d}_{end:04d}.npz"
            note = chunk.with_suffix(".json")
            if chunk.exists() and note.exists():
                with np.load(chunk, allow_pickle=False) as z:
                    values = z["features"]
                record = json.loads(note.read_text())
                assert sha(chunk) == record["sha256"]
            else:
                bank, record = backend.suffixes(tokens[start:end], seed=seed, g=g, threads=threads)
                error = float(np.abs(bank[0] - reference[start:end]).max())
                assert error < 5e-5, (g, seed, start, error)
                values = mix_suffix_bank(bank, .2, "tm3", 120)
                nwrite(chunk, features=values)
                record.update(start=start, end=end, closed_feature_error=error, sha256=sha(chunk))
                jwrite(note, record)
            result[start:end] = values
            records.append(record)
            jwrite(folder / "feature_progress.json", dict(g=g, seed=seed, rows_done=end,
                    rows_total=len(tokens), elapsed_seconds=time.monotonic()-started))
            print(f"[p=0.2 features g={g:.2f} seed={seed}] {end}/{len(tokens)}", flush=True)
    assert np.isfinite(result).all() and np.abs(result).max() < 1.0002
    nwrite(output, features=result)
    jwrite(marker, dict(version=VERSION, g=g, seed=seed, p=.2, rows=2376, width=360,
          source=origin, sha256=sha(output), bytes=output.stat().st_size,
          elapsed_seconds=time.monotonic()-started,
          maximum_closed_feature_error=max(r["closed_feature_error"] for r in records),
          maximum_norm_error=max(r["maximum_norm_error"] for r in records)))
    # Only this case's redundant, task-created checkpoints are removed after the
    # consolidated cache has been saved and fingerprinted. Old results are intact.
    for path in sorted((folder / "chunks").glob("rows*.npz")):
        path.unlink()
    for path in sorted((folder / "chunks").glob("rows*.json")):
        path.unlink()
    return output
