from common import *
from hopping_controls import tune, esn_features, GRID
from a01_core import load_config, TARGET_NAMES, _best_ridge, _best_svr
import itertools

data = hopping_data()
x = np.take(data["tokens"].reshape(-1, 20, 15, 3)[:, -1], [0, 2], axis=-1).reshape(len(data["tokens"]), -1)
split = data["split"]
target = data["targets"].astype(float)
choices, _ = tune(x, target, split)
cfg = load_config().payload["decoders"]
grid = list(itertools.product(cfg["svr_C_grid"], cfg["svr_gamma_scale_grid"], cfg["svr_epsilon_grid"]))
rows = []
for index, name in enumerate(TARGET_NAMES):
    for decoder in ("linear", "rbf_svr"):
        args = (x[split == 0], target[split == 0, index], x[split == 1], target[split == 1, index])
        reference = _best_ridge(*args, cfg["ridge_alpha_grid"]) if decoder == "linear" else _best_svr(*args, grid, 256)
        choice = next(c for c in choices if c["target"] == name and c["decoder"] == decoder)
        keys = ["alpha"] if decoder == "linear" else ["C", "gamma_scale", "epsilon"]
        same = all(choice[k] == reference[k] for k in keys)
        error = abs(choice["validation_mae_ev"] - reference["validation_mae_ev"])
        rows.append(dict(target=name, decoder=decoder, same_hyperparameters=same, validation_mae_error=error))
        assert same and error < 1e-5, rows[-1]
for seed in SEEDS:
    values = esn_features(data["tokens"], seed, GRID.index((.9, .65, .75)))
    with np.load(HOP / f"features/esn_post_tm3_suffixes_seed{seed:03d}.npz") as z:
        reference = z["suffixes"][0]
    error = float(np.max(np.abs(values - reference)))
    rows.append(dict(target="ESN features", seed=seed, max_abs_error=error))
    assert error < 1e-4, rows[-1]
write_csv(HERE / "hopping/implementation_parity.csv", rows)
print("PASS", rows, flush=True)
