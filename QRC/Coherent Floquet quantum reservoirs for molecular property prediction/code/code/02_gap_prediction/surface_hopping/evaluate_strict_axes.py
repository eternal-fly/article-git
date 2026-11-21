#!/usr/bin/env python3
"""Six-seed, decoder-only-validation evaluation for every fixed hopping axis.

The physical axes p and final/TM3 are never selected by validation data.  Each
fixed slice receives its own validation-selected Ridge or RBF-SVR decoder and
is then refit on train+validation before one held-out trajectory evaluation.
All test cells remain available; target-wise p* is an explicitly post-hoc scan
descriptor protected by a simultaneous band, not a deployable model choice.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from a01_core import (
    TARGET_NAMES,
    _best_ridge,
    _best_svr,
    _refit_ridge,
    _refit_svr,
    feature_path,
    load_config,
    mix_suffix_bank,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parent
STRICT_ROOT = ROOT / "outputs" / "strict_decoder_only"
SHARD_ROOT = STRICT_ROOT / "shards"
PROTOCOL_VERSION = "shnitsel-a01-q15-t20-six-seed-fixed-physical-axes-v2"
COMPARISON_PROTOCOL_VERSION = (
    "shnitsel-a01-q15-t20-qrc-p-axis-vs-standard-esn-six-seed-v5-uniform-p005"
)
READOUTS = ("final", "tm3")
DECODERS = ("linear", "rbf_svr")
EXPECTED_SEEDS = (0, 10, 20, 30, 40, 50)


def p_label(value: float) -> str:
    return f"p{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def shard_stem(system: str, seed: int, p: float, readout: str) -> Path:
    return SHARD_ROOT / f"{system}_seed{int(seed):03d}_{p_label(p)}_{readout}"


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
    system: str,
    seed: int,
    p: float,
    readout: str,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    cfg = load_config()
    if cfg.seeds != EXPECTED_SEEDS:
        raise ValueError(f"formal seed contract changed: {cfg.seeds}")
    if system not in ("qrc", "esn") or seed not in cfg.seeds:
        raise ValueError((system, seed))
    if readout not in READOUTS or not any(np.isclose(p, item) for item in cfg.p_grid):
        raise ValueError((p, readout))
    if system == "esn" and not np.isclose(p, 0.0):
        raise ValueError("the standard ESN always receives the complete T=20 stream (p=0)")
    stem = shard_stem(system, seed, p, readout)
    csv_path, npz_path = stem.with_suffix(".csv"), stem.with_suffix(".npz")
    if csv_path.exists() and npz_path.exists() and not force:
        return csv_path, npz_path

    with np.load(cfg.processed_path, allow_pickle=False) as package:
        targets = package["targets"].astype(np.float64)
        split = package["split"].astype(np.int8)
        trajectory_id = package["trajectory_id"].astype(np.int32)
        endpoint = package["endpoint"].astype(np.int16)
    train, validation, test = split == 0, split == 1, split == 2
    with np.load(feature_path(system, seed), allow_pickle=False) as package:
        bank = package["suffixes"].astype(np.float32)
    features = mix_suffix_bank(bank, float(p), readout, cfg.output_dim)
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
                "t": cfg.history_steps,
                "p": float(p),
                "readout": readout,
                "decoder": decoder,
                "target": target_name,
                "feature_dim": int(features.shape[1]),
                "validation_mae_ev": float(selected["validation_mae_ev"]),
                **regression_metrics(y[test], prediction),
                "decoder_selection_rule": "validation MAE; physical axes fixed",
                "protocol_version": PROTOCOL_VERSION,
            }
            for key in ("alpha", "C", "gamma_scale", "gamma", "epsilon"):
                row[key] = selected.get(key, "")
            rows.append(row)
            predictions.append(np.asarray(prediction, dtype=np.float32))
            case_decoder.append(decoder)
            case_target.append(target_name)
        print(
            f"[{system} seed={seed} p={p:g} {readout}] {target_name} complete",
            flush=True,
        )

    _atomic_csv(pd.DataFrame(rows), csv_path)
    _atomic_npz(
        npz_path,
        predictions=np.stack(predictions),
        decoder=np.asarray(case_decoder),
        target_name=np.asarray(case_target),
        target_values=targets[test].astype(np.float32),
        trajectory_id=trajectory_id[test],
        endpoint=endpoint[test],
        p=np.float64(p),
        readout=np.asarray(readout),
        system=np.asarray(system),
        seed=np.int64(seed),
        protocol_version=np.asarray(PROTOCOL_VERSION),
    )
    return csv_path, npz_path


@lru_cache(maxsize=None)
def _trajectory_errors(
    system: str, seed: int, p: float, readout: str, decoder: str, target: str
) -> tuple[np.ndarray, np.ndarray]:
    path = shard_stem(system, seed, p, readout).with_suffix(".npz")
    with np.load(path, allow_pickle=False) as package:
        matches = np.flatnonzero(
            (package["decoder"].astype(str) == decoder)
            & (package["target_name"].astype(str) == target)
        )
        if len(matches) != 1:
            raise RuntimeError((path, decoder, target, matches))
        prediction = package["predictions"][int(matches[0])].astype(np.float64)
        target_index = TARGET_NAMES.index(target)
        truth = package["target_values"][:, target_index].astype(np.float64)
        trajectory = package["trajectory_id"].astype(np.int64)
    ids = np.unique(trajectory)
    error = np.abs(prediction - truth)
    return ids, np.asarray([error[trajectory == item].mean() for item in ids])


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


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


def _crossed_bootstrap_curves(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> np.ndarray:
    """Resample one shared seed/trajectory draw across the complete p curve."""
    if values.ndim != 3:
        raise ValueError(f"expected p x seed x trajectory values, got {values.shape}")
    result = np.empty((draws, values.shape[0]), dtype=np.float64)
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        count = stop - start
        seed_draw = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        trajectory_draw = rng.integers(
            0, values.shape[2], size=(count, values.shape[2])
        )
        for p_index in range(values.shape[0]):
            sampled = values[p_index][
                seed_draw[:, :, None], trajectory_draw[:, None, :]
            ]
            result[start:stop, p_index] = sampled.mean(axis=(1, 2))
    return result


def aggregate(*, bootstrap_draws: int = 20_000) -> tuple[Path, Path, Path, Path]:
    cfg = load_config()
    expected_csv = [
        shard_stem(system, seed, p, readout).with_suffix(".csv")
        for system in ("qrc",)
        for seed in cfg.seeds
        for p in cfg.p_grid
        for readout in READOUTS
    ]
    expected_csv.extend(
        shard_stem("esn", seed, 0.0, readout).with_suffix(".csv")
        for seed in cfg.seeds
        for readout in READOUTS
    )
    missing = [path for path in expected_csv if not path.exists()]
    if missing:
        raise FileNotFoundError("missing strict shards:\n" + "\n".join(map(str, missing)))
    metrics = pd.concat((pd.read_csv(path) for path in expected_csv), ignore_index=True)
    metrics = metrics.sort_values(
        ["system", "seed", "p", "readout", "decoder", "target"]
    ).reset_index(drop=True)
    metrics_path = STRICT_ROOT / "reservoir_metrics_all_fixed_axes.csv"
    _atomic_csv(metrics, metrics_path)
    summary = metrics.groupby(
        ["system", "p", "readout", "decoder", "target"], as_index=False
    ).agg(
        mae_ev_mean=("mae_ev", "mean"),
        mae_ev_sample_std=("mae_ev", "std"),
        rmse_ev_mean=("rmse_ev", "mean"),
        rmse_ev_sample_std=("rmse_ev", "std"),
        r2_mean=("r2", "mean"),
        r2_sample_std=("r2", "std"),
        num_seeds=("seed", "size"),
    )
    summary_path = STRICT_ROOT / "reservoir_summary_all_fixed_axes.csv"
    _atomic_csv(summary, summary_path)

    paired_rows: list[dict] = []
    closed_paired_rows: list[dict] = []
    effect_matrices: dict[tuple[str, str, str, float], np.ndarray] = {}
    closed_effect_matrices: dict[tuple[str, str, str, float], np.ndarray] = {}
    for p in cfg.p_grid:
        for readout in READOUTS:
            for decoder in DECODERS:
                for target in TARGET_NAMES:
                    differences = []
                    closed_differences = []
                    reference_ids = None
                    for seed in cfg.seeds:
                        qrc_ids, qrc_error = _trajectory_errors(
                            "qrc", seed, p, readout, decoder, target
                        )
                        esn_ids, esn_error = _trajectory_errors(
                            "esn", seed, 0.0, readout, decoder, target
                        )
                        closed_ids, closed_error = _trajectory_errors(
                            "qrc", seed, 0.0, readout, decoder, target
                        )
                        if not (
                            np.array_equal(qrc_ids, esn_ids)
                            and np.array_equal(qrc_ids, closed_ids)
                        ):
                            raise RuntimeError("QRC/baseline trajectory mismatch")
                        if reference_ids is None:
                            reference_ids = qrc_ids
                        elif not np.array_equal(reference_ids, qrc_ids):
                            raise RuntimeError("seed trajectory mismatch")
                        differences.append(qrc_error - esn_error)
                        closed_differences.append(qrc_error - closed_error)
                    values = np.stack(differences)
                    closed_values = np.stack(closed_differences)
                    effect_matrices[(readout, decoder, target, float(p))] = values
                    closed_effect_matrices[
                        (readout, decoder, target, float(p))
                    ] = closed_values
                    rng = np.random.default_rng(
                        _stable_seed(
                            COMPARISON_PROTOCOL_VERSION, p, readout, decoder, target
                        )
                    )
                    boot = _crossed_bootstrap(values, rng, bootstrap_draws)
                    seed_means = values.mean(axis=1)
                    paired_rows.append({
                        "p": float(p),
                        "qrc_p": float(p),
                        "esn_p": 0.0,
                        "readout": readout,
                        "decoder": decoder,
                        "target": target,
                        "mean_qrc_minus_esn_trajectory_mae_ev": float(values.mean()),
                        "crossed_bootstrap_95ci_low_ev": float(np.quantile(boot, 0.025)),
                        "crossed_bootstrap_95ci_high_ev": float(np.quantile(boot, 0.975)),
                        "qrc_better_seed_fraction": float(np.mean(seed_means < 0)),
                        "exact_two_sided_seed_sign_flip_p": _exact_sign_flip_p(seed_means),
                        "num_seeds": len(cfg.seeds),
                        "num_test_trajectories": int(values.shape[1]),
                        "bootstrap_draws": bootstrap_draws,
                        "protocol_version": COMPARISON_PROTOCOL_VERSION,
                    })
                    closed_rng = np.random.default_rng(
                        _stable_seed(
                            COMPARISON_PROTOCOL_VERSION,
                            "qrc-minus-closed-pointwise",
                            p,
                            readout,
                            decoder,
                            target,
                        )
                    )
                    closed_boot = _crossed_bootstrap(
                        closed_values, closed_rng, bootstrap_draws
                    )
                    closed_seed_means = closed_values.mean(axis=1)
                    closed_paired_rows.append({
                        "p": float(p),
                        "qrc_p": float(p),
                        "closed_qrc_p": 0.0,
                        "readout": readout,
                        "decoder": decoder,
                        "target": target,
                        "mean_qrc_minus_closed_trajectory_mae_ev": float(
                            closed_values.mean()
                        ),
                        "crossed_bootstrap_95ci_low_ev": float(
                            np.quantile(closed_boot, 0.025)
                        ),
                        "crossed_bootstrap_95ci_high_ev": float(
                            np.quantile(closed_boot, 0.975)
                        ),
                        "qrc_better_than_closed_seed_fraction": float(
                            np.mean(closed_seed_means < 0)
                        ),
                        "exact_two_sided_seed_sign_flip_p": _exact_sign_flip_p(
                            closed_seed_means
                        ),
                        "num_seeds": len(cfg.seeds),
                        "num_test_trajectories": int(closed_values.shape[1]),
                        "bootstrap_draws": bootstrap_draws,
                        "baseline_definition": (
                            "same-seed, same-readout, same-decoder closed QRC at p=0"
                        ),
                        "protocol_version": COMPARISON_PROTOCOL_VERSION,
                    })
    paired_path = STRICT_ROOT / "qrc_minus_esn_paired_all_fixed_axes.csv"
    paired = pd.DataFrame(paired_rows)
    _atomic_csv(paired, paired_path)
    closed_paired_path = STRICT_ROOT / "qrc_minus_closed_paired_all_fixed_axes.csv"
    _atomic_csv(pd.DataFrame(closed_paired_rows), closed_paired_path)

    optimum_rows: list[dict] = []
    p_values = tuple(map(float, cfg.p_grid))
    for readout in READOUTS:
        for decoder in DECODERS:
            for target in TARGET_NAMES:
                qrc_curve = summary[
                    (summary.system == "qrc")
                    & (summary.readout == readout)
                    & (summary.decoder == decoder)
                    & (summary.target == target)
                ].sort_values("p")
                if len(qrc_curve) != len(p_values) or not np.allclose(
                    qrc_curve.p.to_numpy(float), np.asarray(sorted(p_values))
                ):
                    raise RuntimeError((readout, decoder, target, qrc_curve.p.tolist()))
                ranked_curve = qrc_curve.sort_values(["mae_ev_mean", "p"])
                selected = ranked_curve.iloc[0]
                second = ranked_curve.iloc[1]
                p_star = float(selected.p)
                seed_cells = metrics[
                    (metrics.system == "qrc")
                    & (metrics.readout == readout)
                    & (metrics.decoder == decoder)
                    & (metrics.target == target)
                ]
                seed_optima = seed_cells.loc[
                    seed_cells.sort_values(["seed", "mae_ev", "p"])
                    .groupby("seed", sort=True).head(1).index
                ]
                seed_counts = {
                    f"{float(p):g}": int(count)
                    for p, count in seed_optima.p.value_counts().sort_index().items()
                }
                esn = summary[
                    (summary.system == "esn")
                    & np.isclose(summary.p, 0.0)
                    & (summary.readout == readout)
                    & (summary.decoder == decoder)
                    & (summary.target == target)
                ].iloc[0]
                closed = summary[
                    (summary.system == "qrc")
                    & np.isclose(summary.p, 0.0)
                    & (summary.readout == readout)
                    & (summary.decoder == decoder)
                    & (summary.target == target)
                ].iloc[0]
                effect = paired[
                    np.isclose(paired.p, p_star)
                    & (paired.readout == readout)
                    & (paired.decoder == decoder)
                    & (paired.target == target)
                ].iloc[0]
                matrices = np.stack([
                    effect_matrices[(readout, decoder, target, p)]
                    for p in sorted(p_values)
                ])
                rng = np.random.default_rng(
                    _stable_seed(
                        COMPARISON_PROTOCOL_VERSION, "simultaneous-p-band",
                        readout, decoder, target,
                    )
                )
                boot_curves = _crossed_bootstrap_curves(
                    matrices, rng, bootstrap_draws
                )
                curve_mean = matrices.mean(axis=(1, 2))
                maximum_deviation = np.max(
                    np.abs(boot_curves - curve_mean[None, :]), axis=1
                )
                critical = float(np.quantile(maximum_deviation, 0.95))
                closed_matrices = np.stack([
                    closed_effect_matrices[(readout, decoder, target, p)]
                    for p in sorted(p_values)
                ])
                closed_selected = closed_effect_matrices[
                    (readout, decoder, target, p_star)
                ]
                closed_rng = np.random.default_rng(
                    _stable_seed(
                        COMPARISON_PROTOCOL_VERSION, "p-star-vs-closed-pointwise",
                        readout, decoder, target,
                    )
                )
                closed_boot = _crossed_bootstrap(
                    closed_selected, closed_rng, bootstrap_draws
                )
                closed_curve_rng = np.random.default_rng(
                    _stable_seed(
                        COMPARISON_PROTOCOL_VERSION, "closed-simultaneous-p-band",
                        readout, decoder, target,
                    )
                )
                closed_boot_curves = _crossed_bootstrap_curves(
                    closed_matrices, closed_curve_rng, bootstrap_draws
                )
                closed_curve_mean = closed_matrices.mean(axis=(1, 2))
                closed_maximum_deviation = np.max(
                    np.abs(
                        closed_boot_curves - closed_curve_mean[None, :]
                    ),
                    axis=1,
                )
                closed_critical = float(
                    np.quantile(closed_maximum_deviation, 0.95)
                )
                closed_effect = float(closed_selected.mean())
                closed_seed_means = closed_selected.mean(axis=1)
                optimum_rows.append({
                    "readout": readout,
                    "decoder": decoder,
                    "target": target,
                    "p_star_test_grid_argmin": p_star,
                    "second_best_p_by_six_seed_mean": float(second.p),
                    "p_star_mae_margin_to_second_best_ev": float(
                        second.mae_ev_mean - selected.mae_ev_mean
                    ),
                    "per_seed_p_star_counts_json": json.dumps(
                        seed_counts, sort_keys=True
                    ),
                    "qrc_mae_ev_mean": float(selected.mae_ev_mean),
                    "qrc_mae_ev_sample_std": float(selected.mae_ev_sample_std),
                    "esn_mae_ev_mean": float(esn.mae_ev_mean),
                    "esn_mae_ev_sample_std": float(esn.mae_ev_sample_std),
                    "closed_qrc_mae_ev_mean": float(closed.mae_ev_mean),
                    "closed_qrc_mae_ev_sample_std": float(
                        closed.mae_ev_sample_std
                    ),
                    "mean_qrc_minus_esn_trajectory_mae_ev": float(
                        effect.mean_qrc_minus_esn_trajectory_mae_ev
                    ),
                    "pointwise_95ci_low_ev": float(
                        effect.crossed_bootstrap_95ci_low_ev
                    ),
                    "pointwise_95ci_high_ev": float(
                        effect.crossed_bootstrap_95ci_high_ev
                    ),
                    "simultaneous_95band_low_ev": float(
                        effect.mean_qrc_minus_esn_trajectory_mae_ev - critical
                    ),
                    "simultaneous_95band_high_ev": float(
                        effect.mean_qrc_minus_esn_trajectory_mae_ev + critical
                    ),
                    "simultaneous_band_critical_ev": critical,
                    "qrc_better_seed_fraction": float(
                        effect.qrc_better_seed_fraction
                    ),
                    "exact_two_sided_seed_sign_flip_p": float(
                        effect.exact_two_sided_seed_sign_flip_p
                    ),
                    "mean_p_star_minus_closed_trajectory_mae_ev": closed_effect,
                    "closed_pointwise_95ci_low_ev": float(
                        np.quantile(closed_boot, 0.025)
                    ),
                    "closed_pointwise_95ci_high_ev": float(
                        np.quantile(closed_boot, 0.975)
                    ),
                    "closed_simultaneous_95band_low_ev": float(
                        closed_effect - closed_critical
                    ),
                    "closed_simultaneous_95band_high_ev": float(
                        closed_effect + closed_critical
                    ),
                    "closed_simultaneous_band_critical_ev": closed_critical,
                    "p_star_better_than_closed_seed_fraction": float(
                        np.mean(closed_seed_means < 0)
                    ),
                    "p_star_vs_closed_exact_two_sided_seed_sign_flip_p": (
                        _exact_sign_flip_p(closed_seed_means)
                    ),
                    "num_p_values_scanned": len(p_values),
                    "num_seeds": len(cfg.seeds),
                    "num_test_trajectories": int(matrices.shape[2]),
                    "selection_scope": (
                        "descriptive argmin of six-seed mean test MAE over the "
                        "complete physical p grid; not validation or deployment selection"
                    ),
                    "multiplicity_control": (
                        "95% max-absolute centered crossed-bootstrap band over all p"
                    ),
                    "protocol_version": COMPARISON_PROTOCOL_VERSION,
                })
    optimum_path = STRICT_ROOT / "qrc_p_scan_optima_by_fixed_decoder_readout.csv"
    _atomic_csv(pd.DataFrame(optimum_rows), optimum_path)
    protocol = {
        "protocol_version": COMPARISON_PROTOCOL_VERSION,
        "fixed_cell_protocol_version": PROTOCOL_VERSION,
        "seeds": list(cfg.seeds),
        "p_grid": list(cfg.p_grid),
        "targets": list(TARGET_NAMES),
        "readouts": list(READOUTS),
        "decoders": list(DECODERS),
        "selection_boundary": (
            "Validation selects only Ridge/SVR internal hyperparameters. Every p, "
            "readout and target slice is retained. The separately labelled p* is a "
            "post-hoc descriptive test-grid argmin, not model selection."
        ),
        "esn_reset_rule": (
            "The standard ESN is evaluated once on the complete T=20 stream "
            "(p=0); p is a QRC physical axis and is not applied to the ESN."
        ),
        "closed_reference_rule": (
            "Every QRC p cell is additionally differenced against the paired "
            "same-seed, same-readout and same-decoder closed QRC at p=0."
        ),
        "p_scan_optimum_rule": (
            "For each target within every fixed decoder/readout slice, p* is the "
            "descriptive argmin of six-seed mean test MAE (eV) over the complete grid. "
            "It is not a validation-selected or deployable hyperparameter. Both "
            "pointwise intervals and a scan-wide simultaneous bootstrap band are "
            "reported against the same-T standard ESN and the same-W closed p=0 QRC."
        ),
        "test_information_length": "T=20 for every QRC/ESN comparison",
        "expected_metric_rows": (
            (len(cfg.p_grid) + 1) * len(cfg.seeds) * 2 * 2 * len(TARGET_NAMES)
        ),
    }
    (STRICT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return metrics_path, summary_path, paired_path, optimum_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--system", choices=("qrc", "esn"), required=True)
    shard.add_argument("--seed", type=int, required=True)
    shard.add_argument("--p", type=float, required=True)
    shard.add_argument("--readout", choices=READOUTS, required=True)
    shard.add_argument("--force", action="store_true")
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    if args.stage == "shard":
        print(evaluate_shard(args.system, args.seed, args.p, args.readout, force=args.force))
    else:
        print(aggregate())


if __name__ == "__main__":
    main()
