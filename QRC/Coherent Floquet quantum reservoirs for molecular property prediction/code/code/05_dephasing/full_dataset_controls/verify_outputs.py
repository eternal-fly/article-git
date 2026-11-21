"""Read-only verification of predictions, selection, protocol and source hashes."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version

from common import *
from sklearn.metrics import roc_auc_score
from a01_core import TARGET_NAMES, load_config, regression_metrics
from probes import encode_symbol_sequences, case_dir

AUDIT = ROOT / "BFS-QRC/PRL/revised_open_dtc_qrc/pre_supplement_protocol_audit_20260905"
CHECKS = []


def check(name, value, **details):
    CHECKS.append(dict(check=name, passed=bool(value), **details))
    if not value:
        raise AssertionError(CHECKS[-1])


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_checks():
    n_predictions = 0
    for family in FAMILIES:
        for lag in LAGS:
            folder = case_dir(family, lag)
            original = CONTROL / f"outputs/full/{family}/{family}_lag{lag:02d}"
            with np.load(folder / "dataset.npz") as z:
                data = {k: z[k] for k in z.files}
            with np.load(original / "dataset.npz") as z:
                check(f"{family}/{lag}: original foreground, labels and groups", all(
                    np.array_equal(data[k], z[original_key])
                    for k, original_key in (("original_symbols", "symbols"), ("labels", "labels"), ("groups", "groups"))))
            check(f"{family}/{lag}: original encoder", np.array_equal(
                encode_symbol_sequences(data["symbols"], 12), data["tokens"]))
            hashes = np.asarray([array_hash(x) for x in data["tokens"]])
            background_hashes = np.asarray([array_hash(x) for x in data["backgrounds"]])
            check(f"{family}/{lag}: unique complete inputs and backgrounds",
                  len(set(hashes)) == 512 and len(set(background_hashes)) == 128)
            for group in range(128):
                members = data["groups"] == group
                x0 = data["symbols"][members & (data["labels"] == 0)]
                x1 = data["symbols"][members & (data["labels"] == 1)]
                assert all(np.array_equal((x0 == c).sum(0), (x1 == c).sum(0)) for c in range(7))
            check(f"{family}/{lag}: 128 group-wise single-symbol marginals", True)
            for seed in SEEDS:
                split = json.loads((folder / f"splits/seed{seed:03d}.json").read_text())
                original_split = json.loads((original / f"splits/seed{seed:03d}.json").read_text())
                assert split == original_split
                groups = [np.unique(data["groups"][split[key]]) for key in ("train_indices", "val_indices", "test_indices")]
                inputs = [set(hashes[split[key]]) for key in ("train_indices", "val_indices", "test_indices")]
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    assert not inputs[i] & inputs[j]
                    assert not set(background_hashes[groups[i]]) & set(background_hashes[groups[j]])
                with np.load(folder / f"features/qrc_seed{seed:03d}.npz") as z:
                    meta = json.loads(str(z["metadata"]))
                    assert all(meta[k] == v for k, v in dict(q=12, t=20, p=.1, g=.84, W=.5, L=3, J_scale=1., seed=seed).items())
                    assert meta["token_hash"] == array_hash(data["tokens"])
                    assert z["features"].shape == (512, 78) and np.isfinite(z["features"]).all()
                metrics = pd.read_csv(folder / f"results/seed{seed:03d}/metrics.csv").set_index("method")
                with np.load(folder / f"results/seed{seed:03d}/predictions.npz") as z:
                    assert np.array_equal(z["test_indices"], split["test_indices"])
                    assert np.array_equal(z["labels"], data["labels"][split["test_indices"]])
                    for method, pred in zip(z["methods"], z["predictions"]):
                        assert np.isfinite(pred).all()
                        assert abs(roc_auc_score(z["labels"], pred) - metrics.loc[method, "test_auc"]) < 1e-10
                        n_predictions += 1
    check("84 probe cells: splits, physics, finite features, prediction AUC", n_predictions == 504,
          prediction_rows=n_predictions)


def graph_checks():
    count = 0
    for dataset in ("BACE", "BBBP"):
        with np.load(FINAL / f"16_dissipative_protocol_migration/cache/tokens/{dataset}_canonical_general_Q24_T30.npz") as z:
            labels = z["labels"]
        for t in (24, 30):
            for seed in SEEDS:
                folder = HERE / f"graph_direct/{dataset}/T{t}/seed{seed:03d}"
                split = json.loads((FINAL / f"16_dissipative_protocol_migration/cache/splits/{dataset}/seed{seed:03d}.json").read_text())
                metrics = pd.read_csv(folder / "metrics.csv").set_index("method")
                with np.load(folder / "predictions.npz") as z:
                    assert np.array_equal(z["test_indices"], split["test_indices"])
                    for method, pred in zip(z["methods"], z["predictions"]):
                        assert abs(roc_auc_score(labels[z["test_indices"]], pred) - metrics.loc[method, "test_auc"]) < 1e-10
                        expected = 36 if method == "Direct-mean" else 36*t
                        assert metrics.loc[method, "feature_dim"] == expected
                        screen = pd.read_csv(folder / f"{method}_validation.csv")
                        winner = screen.sort_values(["val_auc", "val_loss", "val_acc", "C", "gamma_scale"],
                            ascending=[False, True, False, True, True]).iloc[0]
                        assert all(np.isclose(winner[k], metrics.loc[method, k], rtol=1e-12, atol=1e-12)
                                   for k in ("C", "gamma_scale", "val_auc", "val_loss", "val_acc"))
                        count += 1
    check("graph direct: same T, splits, dimensions and prediction AUC", count == 48, prediction_rows=count)


def regression_prediction_checks(folder, data):
    table = pd.read_csv(folder / "metrics.csv")
    with np.load(folder / "predictions.npz") as z:
        test = data["split"] == 2
        assert np.array_equal(z["target_values"], data["targets"][test])
        assert np.array_equal(z["trajectory_id"], data["trajectory_id"][test])
        assert z["predictions"].shape == (len(table), 480)
        assert np.isfinite(z["predictions"]).all()
        for k, row in enumerate(table.to_dict("records")):
            assert row["target"] == z["target"][k]
            if "decoder" in row:
                assert row["decoder"] == z["decoder"][k]
            truth = z["target_values"][:, TARGET_NAMES.index(row["target"])]
            for name, value in regression_metrics(truth, z["predictions"][k]).items():
                assert np.isclose(value, row[name], rtol=1e-8, atol=1e-8), (folder, name, value, row[name])
    return len(table)


def regression_screen_checks(folder, best_name="metrics.csv"):
    screen = pd.read_csv(folder / "validation_screen.csv")
    best = pd.read_csv(folder / best_name)
    for row in best.to_dict("records"):
        block = screen[screen.target == row["target"]]
        if "readout" in row:
            block = block[block.readout == row["readout"]]
        if "decoder" in row:
            block = block[block.decoder == row["decoder"]]
            params = ["alpha"] if row["decoder"] == "linear" else ["C", "gamma_scale", "epsilon"]
        else:
            params = ["max_features", "min_samples_leaf"]
        winner = block.sort_values(["validation_mae_ev"] + params).iloc[0]
        for key in ["validation_mae_ev"] + params:
            assert np.isclose(winner[key], row[key], rtol=1e-12, atol=1e-12), (folder, key, row, winner.to_dict())
    return len(best)


def hopping_checks():
    data = hopping_data()
    check("hopping original Q15-compatible input shape", data["tokens"].shape == (2376, 20, 45))
    split_ids = [set(data["trajectory_id"][data["split"] == s]) for s in range(3)]
    check("hopping trajectory split unchanged and disjoint", [len(x) for x in split_ids] == [178, 59, 60]
          and not any(split_ids[i] & split_ids[j] for i, j in ((0, 1), (0, 2), (1, 2))))
    count = sum(regression_prediction_checks(HERE / f"hopping/direct/window{w:02d}", data) for w in (1, 2, 4, 8, 20))
    check("direct Ridge/SVR five windows: prediction metrics", count == 40)
    count = sum(regression_prediction_checks(HERE / f"hopping/extra_trees/seed{s:03d}", data) for s in SEEDS)
    check("input-matched ExtraTrees six seeds: prediction metrics", count == 24)
    screen_count = sum(regression_screen_checks(HERE / f"hopping/direct/window{w:02d}") for w in (1, 2, 4, 8, 20))
    screen_count += sum(regression_screen_checks(HERE / f"hopping/extra_trees/seed{s:03d}") for s in SEEDS)
    for config in range(27):
        for seed in SEEDS:
            screen_count += regression_screen_checks(HERE / f"hopping/esn_search/config{config:02d}/seed{seed:03d}", "validation_best.csv")
    check("direct/tree/ESN decoder winners reproduce from full validation candidate screens", screen_count == 2656,
          selected_rows=screen_count)
    search = pd.read_csv(HERE / "hopping/esn_validation_all_configs.csv")
    check("ESN 27 configs x six seeds x 16 target/readout/decoder cells", len(search) == 2592 and
          not search.duplicated(["seed", "config", "readout", "decoder", "target"]).any())
    choices = pd.read_csv(HERE / "hopping/esn_validation_selected.csv")
    average = search.groupby(["readout", "target", "decoder", "config"], as_index=False).validation_mae_ev.mean()
    expected = average.sort_values(["validation_mae_ev", "config"]).groupby(["readout", "target", "decoder"], as_index=False).first()
    pd.testing.assert_frame_equal(choices, expected[choices.columns], check_exact=False, rtol=1e-12, atol=1e-12)
    count = 0
    for row in choices.to_dict("records"):
        for seed in SEEDS:
            count += regression_prediction_checks(HERE / f"hopping/esn_selected/{row['readout']}/{row['decoder']}/{row['target']}/seed{seed:03d}", data)
    check("ESN validation-only config selection and final predictions", count == 96)
    qrc_metrics = pd.concat([pd.read_csv(p) for p in sorted((HOP / "outputs/strict_decoder_only/shards").glob("qrc_seed*_*.csv"))])
    qrc_metrics = qrc_metrics[np.isin(np.round(qrc_metrics.p, 8), np.round(load_config().p_grid, 8))]
    average = qrc_metrics.groupby(["readout", "decoder", "target", "p"], as_index=False).validation_mae_ev.mean()
    expected = average.sort_values(["validation_mae_ev", "p"]).groupby(["readout", "decoder", "target"], as_index=False).first()
    choices = pd.read_csv(HERE / "hopping/qrc_validation_selected_p.csv")
    pd.testing.assert_frame_equal(choices, expected[choices.columns], check_exact=False, rtol=1e-12, atol=1e-12)
    check("QRC p selection uses original 27 validation cells, not test metrics", len(qrc_metrics) == 2592)
    main = pd.read_csv(HERE / "hopping/method_summary.csv")
    effects = pd.read_csv(HERE / "hopping/paired_effects.csv")
    for row in effects.to_dict("records"):
        block = main[(main.readout == row["readout"]) & (main.decoder == row["decoder"]) & (main.target == row["target"])].set_index("method")
        delta = block.loc["Selected-QRC", "mae"] - block.loc[row["baseline"], "mae"]
        assert abs(delta - row["delta_mae"]) < 1e-10
    check("paired effects agree with absolute MAE tables", len(effects) == 160)


def dephasing_checks(allow_running):
    files = sorted((HERE / "dephasing").glob("*/lambda*/seed*/convergence.csv"))
    if len(files) != 24 and allow_running:
        CHECKS.append(dict(check="full-Q convergence grid", passed=None, completed=len(files), expected=24, status="running"))
        return
    check("full-Q convergence complete", len(files) == 24)
    for path in files:
        meta = json.loads((path.parent / "metadata.json").read_text())
        q, t = (12, 30) if meta["task"] == "bace" else (15, 20)
        assert (meta["q"], meta["t"]) == (q, t)
        assert meta["K_values"] == [8, 32, 128] and not meta["original_8_trajectory_cache_reused"]
        with np.load(path.parent / "nested_ensemble_128.npz") as z:
            trial = z["features_by_p_and_trial"].astype(float)
        assert trial.shape == (2, 128, 24, q*(q+1)//2)
        assert np.isfinite(trial).all()
        for i, p in enumerate((0., .1)):
            for k in (8, 32, 128):
                with np.load(path.parent / f"K{k:03d}/p{p:.1f}.npz") as z:
                    assert np.max(np.abs(trial[i, :k].mean(0) - z["features"])) < 1e-7
                    expected = trial[i, :k].std(0, ddof=1) / np.sqrt(k)
                    assert np.max(np.abs(expected - z["standard_error"])) < 1e-7
    data = pd.concat([pd.read_csv(p) for p in files])
    check("nested features and per-coordinate standard errors reproduce", len(data) == 144 and data.coherent_parity_error.max() < 3e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-running", action="store_true")
    args = parser.parse_args()
    original = json.loads((AUDIT / "audit_results.json").read_text())["source_sha256"]
    changed = [p for p, h in original.items() if file_hash(ROOT / p) != h]
    check("pre-audited manuscript and source files unchanged", not changed, n_files=len(original), changed=changed)
    probe_checks()
    graph_checks()
    hopping_checks()
    dephasing_checks(args.allow_running)
    sha = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(HERE.glob("*.py"))}
    sha.update(original)
    sha[str((HOP / "data/processed/a01_windows.npz").relative_to(ROOT))] = file_hash(HOP / "data/processed/a01_windows.npz")
    write_json(HERE / "verification.json", dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        complete=all(row["passed"] is True for row in CHECKS), checks=CHECKS))
    write_json(HERE / "provenance.json", dict(source_sha256=sha, python=sys.version,
        packages={p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib")},
        dependency_scope="Existing project modules are imported read-only; this directory is not a standalone replacement for the repository."))
    print(f"PASS: {sum(r['passed'] is True for r in CHECKS)} checks; {sum(r['passed'] is None for r in CHECKS)} running.", flush=True)


if __name__ == "__main__":
    main()
