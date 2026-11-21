from concurrent.futures import ProcessPoolExecutor, as_completed
from common import *
from readouts import fit_svr


def run(job):
    dataset, t, seed = job
    folder = HERE / f"graph_direct/{dataset}/T{t:02d}/seed{seed:03d}"
    if (folder / "metrics.csv").exists():
        return str(folder)
    with np.load(FINAL / f"16_dissipative_protocol_migration/cache/tokens/{dataset}_canonical_general_Q24_T30.npz") as z:
        tokens, labels = z["tokens"][:, :t, :36], z["labels"]
    split = json.loads((FINAL / f"16_dissipative_protocol_migration/cache/splits/{dataset}/seed{seed:03d}.json").read_text())
    rows, predictions = [], []
    for method, x in (("Direct-mean", tokens.mean(1)), ("Direct-full", tokens.reshape(len(tokens), -1))):
        result, artifact = fit_svr(x, labels, split, seed=seed)
        rows.append(dict(dataset=dataset, q=12, t=t, seed=seed, method=method, feature_dim=x.shape[1], **result))
        write_csv(folder / f"{method}_validation.csv", artifact["screen"])
        predictions.append(artifact["test_scores"])
    write_npz(folder / "predictions.npz", predictions=np.stack(predictions),
              methods=np.asarray(["Direct-mean", "Direct-full"]), test_indices=np.asarray(split["test_indices"]))
    write_csv(folder / "metrics.csv", rows)
    return f"{dataset} T{t} seed{seed}"


if __name__ == "__main__":
    with ProcessPoolExecutor(4) as pool:
        for output in pool.map(run, [(d, t, s) for d in ("BACE", "BBBP") for t in (24, 30) for s in SEEDS]):
            print(output, flush=True)
    paths = sorted((HERE / "graph_direct").glob("*/T*/seed*/metrics.csv"))
    write_csv(HERE / "graph_direct/metrics_all_seeds.csv", pd.concat([pd.read_csv(f) for f in paths]))
