from __future__ import annotations

import argparse
from common import *
from a01_core import TARGET_NAMES, load_config
from evaluate_strict_axes import _trajectory_errors, shard_stem

PAPER = ROOT / "BFS-QRC/PRL/revised_open_dtc_qrc"


def probe_stats():
    metrics = pd.read_csv(HERE / "probes/metrics_all_seeds.csv")
    summaries, comparisons = [], []
    rng = np.random.default_rng(20260905)
    draws = rng.integers(0, 6, size=(20000, 6))
    for (family, method, decoder), block in metrics.groupby(["family", "method", "decoder"]):
        curve = block.pivot(index="seed", columns="lag", values="test_auc").loc[list(SEEDS), list(LAGS)].to_numpy()
        boot = curve[draws].mean(1)
        mean = curve.mean(0)
        simultaneous_half = np.quantile(np.max(np.abs(boot - mean), axis=1), .95)
        for index, lag in enumerate(LAGS):
            summaries.append(dict(family=family, method=method, decoder=decoder, lag=lag,
                auc_mean=mean[index], auc_sd=curve[:, index].std(ddof=1),
                ci_low=np.quantile(boot[:, index], .025), ci_high=np.quantile(boot[:, index], .975),
                simultaneous_low=mean[index]-simultaneous_half, simultaneous_high=mean[index]+simultaneous_half,
                above_chance_seeds=int(np.sum(curve[:, index] > .5))))
    for family in FAMILIES:
        block = metrics[(metrics.family == family) & (metrics.decoder == "affine")]
        pivot = block.pivot(index=["seed", "lag"], columns="method", values="test_auc")
        for baseline in ("ESN", "FP-CVR"):
            curve = (pivot["DTC-QRC"] - pivot[baseline]).unstack("lag").loc[list(SEEDS), list(LAGS)].to_numpy()
            boot = curve[draws].mean(1)
            mean = curve.mean(0)
            half = np.quantile(np.max(np.abs(boot - mean), axis=1), .95)
            for i, lag in enumerate(LAGS):
                comparisons.append(dict(family=family, baseline=baseline, lag=lag, delta_auc=mean[i],
                     ci_low=np.quantile(boot[:, i], .025), ci_high=np.quantile(boot[:, i], .975),
                     simultaneous_low=mean[i]-half, simultaneous_high=mean[i]+half))
    write_csv(HERE / "probes/summary.csv", summaries)
    write_csv(HERE / "probes/paired_comparisons.csv", comparisons)


def new_errors(path, target, decoder=None):
    with np.load(path) as z:
        hit = z["target"] == target
        if decoder is not None:
            hit &= z["decoder"] == decoder
        chosen = np.flatnonzero(hit)
        assert len(chosen) == 1, (path, target, decoder, chosen)
        prediction = z["predictions"][chosen[0]]
        truth = z["target_values"][:, TARGET_NAMES.index(target)]
        trajectory = z["trajectory_id"]
    ids = np.unique(trajectory)
    return ids, np.asarray([np.abs(prediction[trajectory == t] - truth[trajectory == t]).mean() for t in ids])


def existing_errors(p, readout, decoder, target, system="qrc"):
    result, expected_ids = [], None
    for seed in SEEDS:
        ids, error = _trajectory_errors(system, seed, p, readout, decoder, target)
        if expected_ids is None:
            expected_ids = ids
        assert np.array_equal(ids, expected_ids)
        result.append(error)
    return expected_ids, np.stack(result)


def hopping_stats(require_tuned=False):
    files = sorted((HOP / "outputs/strict_decoder_only/shards").glob("qrc_seed*_*.csv"))
    metrics = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    metrics = metrics[np.isin(np.round(metrics.p, 8), np.round(load_config().p_grid, 8))].copy()
    assert set(metrics.seed) == set(SEEDS)
    keys = ["readout", "decoder", "target"]
    assert len(metrics) == 27 * 6 * 2 * 2 * 4
    assert not metrics.duplicated(keys + ["p", "seed"]).any()
    validation = metrics.groupby(keys+["p"], as_index=False).validation_mae_ev.mean()
    selection = validation.sort_values(["validation_mae_ev", "p"]).groupby(keys, as_index=False).first()
    assert len(selection) == 16
    write_csv(HERE / "hopping/qrc_validation_selected_p.csv", selection)
    effects, summary, gap_records = [], [], []
    for selected in selection.to_dict("records"):
        readout, decoder, target, p = (selected[k] for k in ["readout", "decoder", "target", "p"])
        ids, qrc = existing_errors(p, readout, decoder, target)
        comparisons = {}
        for method, probability, system in (("QRC-p1", 1., "qrc"), ("Closed-QRC", 0., "qrc"), ("Original-ESN", 0., "esn")):
            other_ids, err = existing_errors(probability, readout, decoder, target, system)
            assert np.array_equal(ids, other_ids)
            comparisons[method] = err
        for w in (1, 2, 4, 8, 20):
            path = HERE / f"hopping/direct/window{w:02d}/predictions.npz"
            if path.exists():
                other_ids, err = new_errors(path, target, decoder)
                assert np.array_equal(ids, other_ids)
                comparisons[f"Direct-{w}"] = np.repeat(err[None], 6, axis=0)
        if (HERE / "hopping/esn_selected_metrics.csv").exists():
            rows = []
            for seed in SEEDS:
                path = HERE / f"hopping/esn_selected/{readout}/{decoder}/{target}/seed{seed:03d}/predictions.npz"
                other_ids, err = new_errors(path, target, decoder)
                assert np.array_equal(ids, other_ids)
                rows.append(err)
            comparisons["Tuned-ESN"] = np.stack(rows)
        elif require_tuned:
            raise RuntimeError("Tuned ESN incomplete")
        tree_paths = [HERE / f"hopping/extra_trees/seed{s:03d}/predictions.npz" for s in SEEDS]
        if all(path.exists() for path in tree_paths):
            errors = []
            for path in tree_paths:
                other_ids, err = new_errors(path, target, "tree")
                assert np.array_equal(ids, other_ids)
                errors.append(err)
            comparisons["ExtraTrees"] = np.stack(errors)
        all_methods = dict(comparisons, **{"Selected-QRC": qrc})
        for method, values in all_methods.items():
            stat = crossed_interval(values, (readout, decoder, target, method, "absolute"))
            summary.append(dict(readout=readout, decoder=decoder, target=target, p=p,
                                method=method, mae=stat["mean"], ci_low=stat["ci_low"], ci_high=stat["ci_high"],
                                seed_sd=float(values.mean(1).std(ddof=1)), trajectories=60,
                                deterministic_baseline=method.startswith("Direct-")))
        for baseline, values in comparisons.items():
            delta = qrc - values
            stat = crossed_interval(delta, (readout, decoder, target, baseline, "paired"))
            effects.append(dict(readout=readout, decoder=decoder, target=target, selected_p=p,
                                baseline=baseline, delta_mae=stat["mean"], ci_low=stat["ci_low"], ci_high=stat["ci_high"],
                                better_seeds=int(np.sum(delta.mean(1) < 0))))
        write_npz(HERE / f"hopping/error_matrices/{readout}_{decoder}_{target}.npz", trajectory_id=ids,
                  methods=np.asarray(list(all_methods)), error=np.stack(list(all_methods.values())), selected_p=np.asarray(p))
    write_csv(HERE / "hopping/method_summary.csv", summary)
    write_csv(HERE / "hopping/paired_effects.csv", effects)


def graph_stats():
    direct = pd.read_csv(HERE / "graph_direct/metrics_all_seeds.csv")
    source = pd.read_csv(PAPER / "source_data/main/graph_stream_q12_metrics_all_seeds.csv")
    source = source[source.t.isin((24, 30))]
    source = source.rename(columns={"system": "method"})
    combined = pd.concat([direct, source], ignore_index=True)
    write_csv(HERE / "graph_direct/comparison_metrics.csv", combined)
    rows = []
    rng = np.random.default_rng(20260905)
    draw = rng.integers(0, 6, size=(20000, 6))
    for (dataset, t), df in combined.groupby(["dataset", "t"]):
        pivot = df.pivot(index="seed", columns="method", values="test_auc").loc[list(SEEDS)]
        for method in pivot.columns:
            values = pivot[method].to_numpy()
            delta = pivot.open_dtc_qrc.to_numpy() - values
            boot = delta[draw].mean(1)
            rows.append(dict(dataset=dataset, t=t, method=method, auc=values.mean(), sd=values.std(ddof=1),
                        qrc_minus_baseline=delta.mean(), paired_ci_low=np.quantile(boot, .025), paired_ci_high=np.quantile(boot, .975)))
    write_csv(HERE / "graph_direct/comparison_summary.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-tuned", action="store_true")
    args = parser.parse_args()
    probe_stats()
    graph_stats()
    hopping_stats(args.require_tuned)
    print("Summaries written", flush=True)
