"""Exploratory, fixed-gap-bin diagnostics; no model selection or new fitting."""
from common import *
from a01_core import TARGET_NAMES
from evaluate_strict_axes import shard_stem


def prediction(path, target, decoder="rbf_svr"):
    with np.load(path) as z:
        names = z["target_name"] if "target_name" in z.files else z["target"]
        hits = np.flatnonzero((names == target) & (z["decoder"] == decoder))
        assert len(hits) == 1, (path, target, hits)
        return z["predictions"][hits[0]].astype(float), z["trajectory_id"]


def main():
    data = hopping_data()
    test = data["split"] == 2
    trajectory = data["trajectory_id"][test]
    bins = (0., 1., 3., np.inf)
    selections = pd.read_csv(HERE / "hopping/qrc_validation_selected_p.csv")
    selections = selections[(selections.readout == "final") & (selections.decoder == "rbf_svr")].set_index("target")
    distributions, rows = [], []
    for j, target in enumerate(TARGET_NAMES):
        truth = data["targets"][test, j].astype(float)
        p = float(selections.loc[target, "p"])
        models = {}
        for name in ("Selected-QRC", "QRC-p1", "Tuned-ESN", "ExtraTrees", "Direct-20"):
            seeds = (-1,) if name == "Direct-20" else SEEDS
            values = []
            for seed in seeds:
                decoder = "rbf_svr"
                if name in ("Selected-QRC", "QRC-p1"):
                    path = shard_stem("qrc", seed, p if name == "Selected-QRC" else 1., "final").with_suffix(".npz")
                elif name == "Tuned-ESN":
                    path = HERE / f"hopping/esn_selected/final/rbf_svr/{target}/seed{seed:03d}/predictions.npz"
                elif name == "ExtraTrees":
                    path = HERE / f"hopping/extra_trees/seed{seed:03d}/predictions.npz"
                    decoder = "tree"
                else:
                    path = HERE / "hopping/direct/window20/predictions.npz"
                pred, ids = prediction(path, target, decoder)
                assert np.array_equal(ids, trajectory)
                values.append(pred)
            models[name] = np.stack(values)
        for lo, hi in zip(bins[:-1], bins[1:]):
            label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}"
            for split, split_name in enumerate(("train", "validation", "test")):
                valid = (data["split"] == split) & (data["targets"][:, j] >= lo) & (data["targets"][:, j] < hi)
                distributions.append(dict(target=target, split=split_name, gap_bin_ev=label,
                    n_windows=int(valid.sum()), n_trajectories=int(len(np.unique(data["trajectory_id"][valid]))),
                    fraction=float(valid.sum() / np.sum(data["split"] == split))))
            valid = (truth >= lo) & (truth < hi)
            ids = np.unique(trajectory[valid])
            for name, values in models.items():
                err = np.abs(values - truth[None])
                if len(ids):
                    balanced = np.stack([err[:, valid & (trajectory == tid)].mean(1) for tid in ids], axis=1)
                    mean, sd = balanced.mean(), balanced.mean(1).std(ddof=1) if len(values) > 1 else np.nan
                    flat_mae = err[:, valid].mean()
                else:
                    mean, sd, flat_mae = np.nan, np.nan, np.nan
                rows.append(dict(target=target, method=name, gap_bin_ev=label,
                    n_windows=int(valid.sum()), n_trajectories=len(ids), n_model_seeds=len(values),
                    trajectory_balanced_mae=float(mean), model_seed_sd=float(sd), window_mae=float(flat_mae)))
    write_csv(HERE / "hopping/gap_strata_distribution.csv", distributions)
    write_csv(HERE / "hopping/gap_strata_metrics.csv", rows)
    write_json(HERE / "hopping/gap_strata_protocol.json", dict(
        status="exploratory descriptive diagnostic after main fits; no refitting or selection",
        bins_ev=["[0,1)", "[1,3)", "[3,infinity)"],
        bin_definition="held-out target value; not an input available at inference",
        note="Thresholds are descriptive bins, not validated chemical-accuracy or warning thresholds. No significance claims.",
        weighting="Each represented test trajectory is equally weighted within its bin; bins can contain unequal counts of windows per trajectory."))
    print("Gap-stratified descriptive metrics written.", flush=True)


if __name__ == "__main__":
    main()
