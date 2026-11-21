#!/usr/bin/env python3
"""Six-seed fixed-history evaluation for the closed hopping reservoirs.

Every history length is retained as a separate test result.  Validation data
select only Ridge/RBF-SVR hyperparameters and never select the physical history
length or reservoir family.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from closed_core import (
    SYSTEMS,
    TARGET_NAMES,
    _best_ridge,
    _best_svr,
    _refit_ridge,
    _refit_svr,
    feature_path,
    load_config,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parent
STRICT_ROOT = ROOT / "outputs" / "strict_decoder_only"
SHARD_ROOT = STRICT_ROOT / "shards"
EXPECTED_SEEDS = (0, 10, 20, 30, 40, 50)
DECODERS = ("linear", "rbf_svr")
PROTOCOL_VERSION = "shnitsel-a01-q15-closed-W0p5-fixed-history-six-seed-v3"


def shard_stem(system: str, seed: int, history: int) -> Path:
    return SHARD_ROOT / f"{system}_seed{int(seed):03d}_T{int(history):02d}"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def evaluate_shard(
    system: str, seed: int, history: int, *, force: bool = False
) -> tuple[Path, Path]:
    cfg = load_config()
    if cfg.seeds != EXPECTED_SEEDS:
        raise ValueError(f"formal seed contract changed: {cfg.seeds}")
    if system not in SYSTEMS or seed not in cfg.seeds or history not in cfg.histories:
        raise ValueError((system, seed, history))
    stem = shard_stem(system, seed, history)
    csv_path, npz_path = stem.with_suffix(".csv"), stem.with_suffix(".npz")
    if csv_path.exists() and npz_path.exists() and not force:
        return csv_path, npz_path

    with np.load(cfg.dataset_path, allow_pickle=False) as package:
        targets = package["targets"].astype(np.float64)
        split = package["split"].astype(np.int8)
        trajectory_id = package["trajectory_id"].astype(np.int32)
        endpoint = package["endpoint"].astype(np.int16)
    train, validation, test = split == 0, split == 1, split == 2
    with np.load(feature_path(system, seed), allow_pickle=False) as package:
        histories = package["histories"].astype(np.int64)
        grid = package["features"].astype(np.float32)
    matches = np.flatnonzero(histories == int(history))
    if len(matches) != 1:
        raise RuntimeError((feature_path(system, seed), history, matches))
    features = grid[int(matches[0])]

    decoder_cfg = cfg.payload["decoders"]
    alpha_grid = tuple(map(float, decoder_cfg["ridge_alpha_grid"]))
    svr_candidates = [
        (float(C), float(gamma), float(epsilon))
        for C in decoder_cfg["svr_C_grid"]
        for gamma in decoder_cfg["svr_gamma_scale_grid"]
        for epsilon in decoder_cfg["svr_epsilon_grid"]
    ]
    cache_size = float(decoder_cfg["svr_cache_size_mb"])
    rows: list[dict] = []
    predictions: list[np.ndarray] = []
    case_decoder: list[str] = []
    case_target: list[str] = []
    for target_index, target_name in enumerate(TARGET_NAMES):
        y = targets[:, target_index]
        ridge = _best_ridge(
            features[train], y[train], features[validation], y[validation], alpha_grid
        )
        ridge_prediction = _refit_ridge(
            features[train], y[train], features[validation], y[validation],
            features[test], ridge["alpha"],
        )
        svr = _best_svr(
            features[train], y[train], features[validation], y[validation],
            svr_candidates, cache_size,
        )
        svr_prediction = _refit_svr(
            features[train], y[train], features[validation], y[validation],
            features[test], svr, cache_size,
        )
        for decoder, selected, prediction in (
            ("linear", ridge, ridge_prediction),
            ("rbf_svr", svr, svr_prediction),
        ):
            row = {
                "task": "SHNITSEL_A01",
                "system": system,
                "seed": int(seed),
                "q": cfg.q,
                "t": int(history),
                "history_fs": float(history * cfg.dt_fs),
                "p": 0.0,
                "W": cfg.W,
                "readout": "final",
                "decoder": decoder,
                "target": target_name,
                "feature_dim": cfg.output_dim,
                "validation_mae_ev": float(selected["validation_mae_ev"]),
                **regression_metrics(y[test], prediction),
                "decoder_selection_rule": "validation MAE; history fixed",
                "protocol_version": PROTOCOL_VERSION,
            }
            for key in ("alpha", "C", "gamma_scale", "gamma", "epsilon"):
                row[key] = selected.get(key, "")
            rows.append(row)
            predictions.append(np.asarray(prediction, dtype=np.float32))
            case_decoder.append(decoder)
            case_target.append(target_name)
        print(f"[{system} seed={seed} T={history}] {target_name} complete", flush=True)

    _atomic_csv(pd.DataFrame(rows), csv_path)
    _atomic_npz(
        npz_path,
        predictions=np.stack(predictions),
        decoder=np.asarray(case_decoder),
        target_name=np.asarray(case_target),
        target_values=targets[test].astype(np.float32),
        trajectory_id=trajectory_id[test],
        endpoint=endpoint[test],
        system=np.asarray(system),
        seed=np.int64(seed),
        history=np.int16(history),
        protocol_version=np.asarray(PROTOCOL_VERSION),
    )
    return csv_path, npz_path


def _trajectory_errors(
    system: str, seed: int, history: int, decoder: str, target: str
) -> tuple[np.ndarray, np.ndarray]:
    path = shard_stem(system, seed, history).with_suffix(".npz")
    with np.load(path, allow_pickle=False) as package:
        matches = np.flatnonzero(
            (package["decoder"].astype(str) == decoder)
            & (package["target_name"].astype(str) == target)
        )
        if len(matches) != 1:
            raise RuntimeError((path, decoder, target, matches))
        prediction = package["predictions"][int(matches[0])].astype(np.float64)
        truth = package["target_values"][:, TARGET_NAMES.index(target)].astype(np.float64)
        trajectory = package["trajectory_id"].astype(np.int64)
    ids = np.unique(trajectory)
    error = np.abs(prediction - truth)
    return ids, np.asarray([error[trajectory == item].mean() for item in ids])


def _exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    distribution = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(distribution) >= observed - 1e-15))


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _crossed_bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> np.ndarray:
    result = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        count = stop - start
        seed_draw = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        trajectory_draw = rng.integers(
            0, values.shape[1], size=(count, values.shape[1])
        )
        sampled = values[seed_draw[:, :, None], trajectory_draw[:, None, :]]
        result[start:stop] = sampled.mean(axis=(1, 2))
    return result


def aggregate(*, bootstrap_draws: int = 20_000) -> tuple[Path, Path, Path]:
    cfg = load_config()
    expected_csv = [
        shard_stem(system, seed, history).with_suffix(".csv")
        for system in SYSTEMS
        for seed in cfg.seeds
        for history in cfg.histories
    ]
    missing = [path for path in expected_csv if not path.exists()]
    if missing:
        raise FileNotFoundError("missing fixed-history shards:\n" + "\n".join(map(str, missing)))
    metrics = pd.concat((pd.read_csv(path) for path in expected_csv), ignore_index=True)
    metrics = metrics.sort_values(
        ["system", "seed", "t", "decoder", "target"]
    ).reset_index(drop=True)
    metrics_path = STRICT_ROOT / "reservoir_metrics_all_fixed_histories.csv"
    _atomic_csv(metrics, metrics_path)
    summary = metrics.groupby(
        ["system", "t", "history_fs", "decoder", "target"], as_index=False
    ).agg(
        mae_ev_mean=("mae_ev", "mean"),
        mae_ev_sample_std=("mae_ev", "std"),
        rmse_ev_mean=("rmse_ev", "mean"),
        r2_mean=("r2", "mean"),
        num_seeds=("seed", "size"),
    )
    summary_path = STRICT_ROOT / "reservoir_summary_all_fixed_histories.csv"
    _atomic_csv(summary, summary_path)

    paired_rows: list[dict] = []
    for history in cfg.histories:
        for decoder in DECODERS:
            for target in TARGET_NAMES:
                for baseline in SYSTEMS:
                    if baseline == "qrc":
                        continue
                    differences = []
                    reference_ids = None
                    for seed in cfg.seeds:
                        qrc_ids, qrc_error = _trajectory_errors(
                            "qrc", seed, history, decoder, target
                        )
                        base_ids, base_error = _trajectory_errors(
                            baseline, seed, history, decoder, target
                        )
                        if not np.array_equal(qrc_ids, base_ids):
                            raise RuntimeError("trajectory mismatch")
                        if reference_ids is None:
                            reference_ids = qrc_ids
                        elif not np.array_equal(reference_ids, qrc_ids):
                            raise RuntimeError("seed trajectory mismatch")
                        differences.append(qrc_error - base_error)
                    values = np.stack(differences)
                    rng = np.random.default_rng(
                        _stable_seed(PROTOCOL_VERSION, history, decoder, target, baseline)
                    )
                    boot = _crossed_bootstrap(values, rng, bootstrap_draws)
                    seed_means = values.mean(axis=1)
                    paired_rows.append({
                        "t": int(history),
                        "history_fs": float(history * cfg.dt_fs),
                        "decoder": decoder,
                        "target": target,
                        "baseline": baseline,
                        "mean_qrc_minus_baseline_trajectory_mae_ev": float(values.mean()),
                        "crossed_bootstrap_95ci_low_ev": float(np.quantile(boot, 0.025)),
                        "crossed_bootstrap_95ci_high_ev": float(np.quantile(boot, 0.975)),
                        "exact_two_sided_seed_sign_flip_p": _exact_sign_flip_p(seed_means),
                        "num_seeds": len(cfg.seeds),
                        "num_test_trajectories": int(values.shape[1]),
                        "bootstrap_draws": bootstrap_draws,
                        "protocol_version": PROTOCOL_VERSION,
                    })
    paired_path = STRICT_ROOT / "qrc_minus_controls_paired_by_fixed_history.csv"
    _atomic_csv(pd.DataFrame(paired_rows), paired_path)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "seeds": list(cfg.seeds),
        "history_steps": list(cfg.histories),
        "systems": list(SYSTEMS),
        "decoders": list(DECODERS),
        "selection_boundary": (
            "Validation selects only decoder hyperparameters. Every history "
            "length and reservoir family remains a complete reported axis."
        ),
        "comparison_rule": "QRC and every control are compared only at identical T.",
        "expected_metric_rows": len(SYSTEMS) * len(cfg.seeds) * len(cfg.histories) * 2 * len(TARGET_NAMES),
    }
    (STRICT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return metrics_path, summary_path, paired_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--system", choices=SYSTEMS, required=True)
    shard.add_argument("--seed", type=int, required=True)
    shard.add_argument("--history", type=int, required=True)
    shard.add_argument("--force", action="store_true")
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    if args.stage == "shard":
        print(evaluate_shard(args.system, args.seed, args.history, force=args.force))
    else:
        print(aggregate())


if __name__ == "__main__":
    main()
