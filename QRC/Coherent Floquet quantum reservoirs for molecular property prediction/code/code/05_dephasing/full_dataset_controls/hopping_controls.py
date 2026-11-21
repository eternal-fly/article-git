from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import *
from a01_core import TARGET_NAMES, load_config, regression_metrics, _refit_ridge, _refit_svr

GRID = list(itertools.product((.3, .6, .9), (.1, .65, 1.), (.25, .75, 1.5)))


def tune(x, y, split):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.metrics import pairwise_distances
    cfg = load_config().payload["decoders"]
    scaler = StandardScaler().fit(x[split == 0])
    xt = scaler.transform(x[split == 0])
    xv = scaler.transform(x[split == 1])
    yt, yv = y[split == 0], y[split == 1]
    screens, best = [], {}
    for alpha in cfg["ridge_alpha_grid"]:
        for target, name in enumerate(TARGET_NAMES):
            prediction = Ridge(alpha=alpha).fit(xt, yt[:, target]).predict(xv)
            row = dict(target=name, decoder="linear", alpha=alpha,
                       validation_mae_ev=float(np.mean(np.abs(prediction - yv[:, target]))))
            screens.append(row)
            key = (row["validation_mae_ev"], alpha)
            if (name, "linear") not in best or key < best[name, "linear"][0]:
                best[name, "linear"] = key, row
    # Same RBF kernel and epsilon-SVR objective as the original. Reusing each
    # Gram matrix avoids recalculating identical distances for C/epsilon/target.
    train_distance = pairwise_distances(xt.astype(float), metric="sqeuclidean")
    val_distance = pairwise_distances(xv.astype(float), xt.astype(float), metric="sqeuclidean")
    for gs in cfg["svr_gamma_scale_grid"]:
        gamma = gs / x.shape[1]
        kt, kv = np.exp(-gamma * train_distance), np.exp(-gamma * val_distance)
        for C, epsilon, target in itertools.product(cfg["svr_C_grid"], cfg["svr_epsilon_grid"], range(4)):
            model = SVR(kernel="precomputed", C=C, epsilon=epsilon, cache_size=256)
            prediction = model.fit(kt, yt[:, target]).predict(kv)
            name = TARGET_NAMES[target]
            mae = float(np.mean(np.abs(prediction - yv[:, target])))
            row = dict(target=name, decoder="rbf_svr", C=C, gamma_scale=gs, gamma=gamma,
                       epsilon=epsilon, validation_mae_ev=mae)
            screens.append(row)
            key = (mae, C, gs, epsilon)
            if (name, "rbf_svr") not in best or key < best[name, "rbf_svr"][0]:
                best[name, "rbf_svr"] = key, row
    return [value[1] for value in best.values()], screens


def refit_and_save(x, data, choices, folder, identifiers):
    split, target = data["split"], data["targets"].astype(float)
    train, val, test = [split == i for i in range(3)]
    predictions, rows = [], []
    for choice in choices:
        target_index = TARGET_NAMES.index(choice["target"])
        y = target[:, target_index]
        if choice["decoder"] == "linear":
            pred = _refit_ridge(x[train], y[train], x[val], y[val], x[test], choice["alpha"])
        else:
            pred = _refit_svr(x[train], y[train], x[val], y[val], x[test], choice, 256)
        predictions.append(pred)
        rows.append(dict(**identifiers, **choice, **regression_metrics(y[test], pred)))
    write_npz(folder / "predictions.npz", predictions=np.asarray(predictions),
              decoder=np.asarray([c["decoder"] for c in choices]),
              target=np.asarray([c["target"] for c in choices]),
              target_values=target[test], trajectory_id=data["trajectory_id"][test],
              endpoint=data["endpoint"][test])
    write_csv(folder / "metrics.csv", rows)
    return rows


def esn_features(tokens, seed, config):
    import math
    rho, leak, scale = GRID[config]
    rng = np.random.default_rng(seed)
    rec = rng.normal(size=(120, 120)).astype(np.float32)
    radius = float(np.max(np.abs(np.linalg.eigvals(rec.astype(np.float64)))))
    rec *= np.float32(rho / radius)
    inputs = rng.normal(size=(120, 45)).astype(np.float32) / math.sqrt(45)
    bias = rng.normal(scale=.10, size=120).astype(np.float32)
    x = np.tanh(tokens)
    state = np.zeros((len(tokens), 120), dtype=np.float32)
    for event in range(20):
        injection = scale * (x[:, event] @ inputs.T)
        for _ in range(3):
            state = (1 - leak) * state + leak * np.tanh(state @ rec.T + injection + bias)
    snapshots = [state.copy()]
    for _ in range(2):
        state = (1 - leak) * state + leak * np.tanh(state @ rec.T + bias)
        snapshots.append(state.copy())
    return np.concatenate(snapshots, axis=1)


def direct_job(window):
    data = hopping_data()
    folder = HERE / f"hopping/direct/window{window:02d}"
    if (folder / "decoder_parity_v2.json").exists():
        return f"direct w{window} cached"
    # Drop identically zero Y coordinates. Keep every actual supplied X/Z input.
    # Explicit take preserves sample order.
    x = np.take(data["tokens"].reshape(len(data["tokens"]), 20, 15, 3)[:, -window:], [0, 2], axis=-1)
    x = x.reshape(len(x), -1)
    choices, screen = tune(x, data["targets"].astype(float), data["split"])
    write_csv(folder / "validation_screen.csv", screen)
    refit_and_save(x, data, choices, folder, dict(method="Direct", window=window, seed=-1))
    write_json(folder / "decoder_parity_v2.json", dict(ridge="original float32 single-target fitting", svr="reused float64 RBF Gram matrix"))
    return f"direct w{window} complete"


def esn_job(job):
    seed, config = job
    folder = HERE / f"hopping/esn_search/config{config:02d}/seed{seed:03d}"
    if (folder / "validation_best.csv").exists():
        return f"esn config{config} seed{seed} cached"
    data = hopping_data()
    x = esn_features(data["tokens"], seed, config)
    if GRID[config] == (.9, .65, .75):
        with np.load(HOP / f"features/esn_post_tm3_suffixes_seed{seed:03d}.npz") as z:
            reference = z["suffixes"][0]
        error = float(np.max(np.abs(x - reference)))
        if error > 1e-4:
            raise RuntimeError(f"ESN parity failure: {error}")
        write_json(folder / "original_feature_parity.json", dict(max_abs_error=error, tolerance=1e-4))
    write_npz(folder / "features.npz", features=x)
    rows, screens = [], []
    for readout, values in (("final", x[:, :120]), ("tm3", x)):
        best, screen = tune(values, data["targets"].astype(float), data["split"])
        for collection, dest in ((best, rows), (screen, screens)):
            for row in collection:
                dest.append(dict(seed=seed, config=config, readout=readout, rho=GRID[config][0],
                                 leak=GRID[config][1], input_scale=GRID[config][2], **row))
    write_csv(folder / "validation_screen.csv", screens)
    write_csv(folder / "validation_best.csv", rows)
    return f"esn config{config} seed{seed} complete"


def select_esn():
    files = sorted((HERE / "hopping/esn_search").glob("config*/seed*/validation_best.csv"))
    if len(files) != 162:
        raise RuntimeError(f"Expected 162 search cells, found {len(files)}")
    best = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    write_csv(HERE / "hopping/esn_validation_all_configs.csv", best)
    averaged = best.groupby(["config", "readout", "target", "decoder"], as_index=False).validation_mae_ev.mean()
    selected = averaged.sort_values(["validation_mae_ev", "config"]).groupby(["readout", "target", "decoder"], as_index=False).first()
    write_csv(HERE / "hopping/esn_validation_selected.csv", selected)
    data, rows = hopping_data(), []
    for choice in selected.to_dict("records"):
        config = int(choice["config"])
        for seed in SEEDS:
            source = HERE / f"hopping/esn_search/config{config:02d}/seed{seed:03d}"
            with np.load(source / "features.npz") as z:
                x = z["features"]
            if choice["readout"] == "final":
                x = x[:, :120]
            cell = best[(best.config == config) & (best.seed == seed) & (best.readout == choice["readout"])
                        & (best.target == choice["target"]) & (best.decoder == choice["decoder"])].iloc[0].to_dict()
            params = {k: v for k, v in cell.items() if k not in ["seed", "config", "readout", "rho", "leak", "input_scale"] and pd.notna(v)}
            folder = HERE / f"hopping/esn_selected/{choice['readout']}/{choice['decoder']}/{choice['target']}/seed{seed:03d}"
            rows.extend(refit_and_save(x, data, [params], folder, dict(method="Tuned-ESN", seed=seed,
                        config=config, readout=choice["readout"], rho=GRID[config][0], leak=GRID[config][1], input_scale=GRID[config][2])))
    write_csv(HERE / "hopping/esn_selected_metrics.csv", rows)


def tree_job(seed):
    from sklearn.ensemble import ExtraTreesRegressor
    data = hopping_data()
    x = np.take(data["tokens"].reshape(-1, 20, 15, 3), [0, 2], axis=-1).reshape(len(data["tokens"]), -1)
    split, y = data["split"], data["targets"].astype(float)
    folder = HERE / f"hopping/extra_trees/seed{seed:03d}"
    if (folder / "metrics.csv").exists():
        return f"trees seed{seed} cached"
    rows, preds, screens = [], [], []
    for target, name in enumerate(TARGET_NAMES):
        best = None
        for mf, leaf in itertools.product((.5, 1.), (1, 3)):
            model = ExtraTreesRegressor(n_estimators=400, max_features=mf, min_samples_leaf=leaf, random_state=seed, n_jobs=1)
            pred = model.fit(x[split == 0], y[split == 0, target]).predict(x[split == 1])
            mae = float(np.mean(np.abs(pred - y[split == 1, target])))
            screens.append(dict(target=name, max_features=mf, min_samples_leaf=leaf, validation_mae_ev=mae))
            if best is None or (mae, mf, leaf) < best:
                best = (mae, mf, leaf)
        model = ExtraTreesRegressor(n_estimators=400, max_features=best[1], min_samples_leaf=best[2], random_state=seed, n_jobs=1)
        pred = model.fit(x[split != 2], y[split != 2, target]).predict(x[split == 2])
        rows.append(dict(method="ExtraTrees", seed=seed, target=name, validation_mae_ev=best[0],
                         max_features=best[1], min_samples_leaf=best[2], **regression_metrics(y[split == 2, target], pred)))
        preds.append(pred)
    write_csv(folder / "validation_screen.csv", screens)
    write_npz(folder / "predictions.npz", predictions=np.stack(preds), target=np.asarray(TARGET_NAMES),
              decoder=np.asarray(["tree"] * 4), target_values=y[split == 2], trajectory_id=data["trajectory_id"][split == 2])
    write_csv(folder / "metrics.csv", rows)
    return f"trees seed{seed} complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["direct", "esn", "trees", "select-esn", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "select-esn":
        select_esn()
        return
    if args.stage == "test":
        start = time.monotonic()
        print(direct_job(1), f"{time.monotonic()-start:.1f}s", flush=True)
        return
    fn, jobs = {"direct": (direct_job, [1, 2, 4, 8, 20]),
                "esn": (esn_job, [(s, c) for c in range(27) for s in SEEDS]),
                "trees": (tree_job, list(SEEDS))}[args.stage]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fn, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}]", future.result(), flush=True)


if __name__ == "__main__":
    main()
