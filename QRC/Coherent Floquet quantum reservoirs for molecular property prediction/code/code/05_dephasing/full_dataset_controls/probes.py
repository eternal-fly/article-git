from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import *
encode_symbol_sequences = load_source("review_original_tasks", CONTROL / "controlled_streams/tasks.py").encode_symbol_sequences


def case_dir(family, lag):
    return HERE / "probes" / family / f"lag{lag:02d}"


def prepare():
    audit = []
    for family in FAMILIES:
        for lag in LAGS:
            original = CONTROL / f"outputs/full/{family}/{family}_lag{lag:02d}"
            with np.load(original / "dataset.npz", allow_pickle=False) as z:
                original_symbols, labels, groups = (z[k] for k in ("symbols", "labels", "groups"))
            symbols = original_symbols.copy()
            background = np.empty((128, 20), dtype=np.int16)
            rng = np.random.default_rng(stable_seed("background-v1", family, lag))
            for group in range(128):
                background[group] = rng.permutation(np.repeat([0, 4, 5, 6], 5))
                for row in np.flatnonzero(groups == group):
                    neutral = original_symbols[row] == 0
                    symbols[row, neutral] = background[group, neutral]
                subset = groups == group
                left, right = symbols[subset & (labels == 0)], symbols[subset & (labels == 1)]
                for code in range(7):
                    assert np.array_equal((left == code).sum(0), (right == code).sum(0))
                # Foreground code and position remain exactly unchanged.
                assert np.array_equal(symbols[subset][original_symbols[subset] != 0],
                                      original_symbols[subset][original_symbols[subset] != 0])
            tokens = encode_symbol_sequences(symbols, 12)
            hashes = np.asarray([array_hash(row) for row in tokens])
            bg_hashes = np.asarray([array_hash(row) for row in background])
            assert len(set(hashes)) == 512 and len(set(bg_hashes)) == 128
            folder = case_dir(family, lag)
            write_npz(folder / "dataset.npz", tokens=tokens, labels=labels, groups=groups,
                      symbols=symbols, original_symbols=original_symbols, backgrounds=background,
                      input_hashes=hashes, background_hashes=bg_hashes)
            for seed in SEEDS:
                split = json.loads((original / f"splits/seed{seed:03d}.json").read_text())
                indices = [np.asarray(split[k]) for k in ("train_indices", "val_indices", "test_indices")]
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    assert not set(hashes[indices[i]]) & set(hashes[indices[j]])
                    assert not set(groups[indices[i]]) & set(groups[indices[j]])
                write_json(folder / f"splits/seed{seed:03d}.json", split)
                audit.append(dict(family=family, lag=lag, seed=seed, q=12, t=20,
                                  unique_inputs=512, unique_backgrounds=128,
                                  train_rows=len(indices[0]), val_rows=len(indices[1]), test_rows=len(indices[2]),
                                  complete_input_overlap=0, group_overlap=0, unary_marginals_balanced=True,
                                  input_sha256=array_hash(tokens)))
    write_csv(HERE / "probes/input_audit.csv", audit)
    print("PASS: 14 datasets, 512 unique inputs each, 84 zero-overlap splits", flush=True)


def features(family, lag, seed, device):
    import torch
    from hamiltonian_qrc import compute_suffix_banks, mix_suffix_bank
    torch.set_num_threads(1)
    folder = case_dir(family, lag)
    output = folder / f"features/qrc_seed{seed:03d}.npz"
    with np.load(folder / "dataset.npz") as z:
        tokens = z["tokens"]
    expected = dict(q=12, t=20, p=.1, g=.84, W=.5, L=3, J_scale=1., seed=seed,
                    token_hash=array_hash(tokens), family=family, lag=lag)
    if output.exists():
        with np.load(output) as z:
            assert json.loads(str(z["metadata"])) == expected
        return
    start = time.monotonic()
    banks, metadata = compute_suffix_banks(tokens, q=12, t_values=(20,), seed=seed,
                         device=device, g=.84, W_scale=.5, layers=3, interaction_scale=1., batch_size=128)
    write_npz(output, features=mix_suffix_bank(banks[20], .1), metadata=np.asarray(json.dumps(expected)),
              backend_metadata=np.asarray(json.dumps(metadata)))
    print(f"[probe QRC] {family} lag={lag} seed={seed} {time.monotonic()-start:.1f}s", flush=True)


def fit_case(job):
    import torch
    from readouts import fit_linear_bank, fit_svr
    crc_backend = load_source("review_original_crc", FINAL / "2_linear_readout/core/crc3_baseline.py")
    esn_backend = load_source("review_original_esn", FINAL / "2_linear_readout/core/esn_baseline.py")
    torch.set_num_threads(1)
    family, lag, seed = job
    folder = case_dir(family, lag)
    output = folder / f"results/seed{seed:03d}"
    if (output / "metrics.csv").exists():
        return str(output)
    with np.load(folder / "dataset.npz") as z:
        tokens, labels = z["tokens"], z["labels"]
    split = json.loads((folder / f"splits/seed{seed:03d}.json").read_text())
    with np.load(folder / f"features/qrc_seed{seed:03d}.npz") as z:
        qrc = z["features"]
    crc, _ = crc_backend.compute_crc3_z_zz_features(tokens, q=12, seed=seed, device="cpu", layers=3)
    esn, _ = esn_backend.compute_esn_standard_features(tokens, q=12, seed=seed, device="cpu", layers=3)
    methods = ["DTC-QRC", "FP-CVR", "ESN"]
    table, artifacts = fit_linear_bank(np.stack([qrc, crc, esn]), labels, split,
                                       seed=seed, device="cpu", epochs=500)
    table["method"] = methods
    table["decoder"] = "affine"
    rows = table.drop(columns="row_value").to_dict("records")
    predictions = [*artifacts["test_probabilities"]]
    pred_methods, decoders = methods.copy(), ["affine"] * 3
    for method, x in (("Direct-last", tokens[:, -1]), ("Direct-mean", tokens.mean(1)),
                      ("Direct-full", tokens.reshape(len(tokens), -1))):
        row, artifact = fit_svr(x, labels, split, seed=seed)
        rows.append(dict(method=method, decoder="RBF-SVR", **row))
        write_csv(output / f"{method}_validation.csv", artifact["screen"])
        predictions.append(artifact["test_scores"])
        pred_methods.append(method)
        decoders.append("RBF-SVR")
    for row in rows:
        row.update(family=family, lag=lag, seed=seed, q=12, t=20, p=.1)
    write_npz(output / "predictions.npz", predictions=np.stack(predictions),
              methods=np.asarray(pred_methods), decoders=np.asarray(decoders),
              test_indices=artifacts["test_indices"], labels=labels[artifacts["test_indices"]])
    write_csv(output / "metrics.csv", rows)
    return f"{family}/lag{lag:02d}/seed{seed:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "features", "fit"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    jobs = [(f, l, s) for f in FAMILIES for l in LAGS for s in SEEDS]
    if args.limit:
        jobs = jobs[:args.limit]
    if args.stage == "prepare":
        prepare()
    elif args.stage == "features":
        for job in jobs:
            features(*job, args.device)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(fit_case, jobs):
                print("[probe fit]", result, flush=True)
        files = sorted((HERE / "probes").glob("*/lag*/results/seed*/metrics.csv"))
        write_csv(HERE / "probes/metrics_all_seeds.csv", pd.concat([pd.read_csv(f) for f in files]))


if __name__ == "__main__":
    main()
