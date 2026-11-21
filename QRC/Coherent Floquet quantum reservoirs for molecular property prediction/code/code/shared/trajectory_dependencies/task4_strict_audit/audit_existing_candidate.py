"""Strict audit of the existing Transition1x temporal-QRC candidate.

The audit deliberately adds classical controls that were absent from task 3:
raw trajectory models, early DFT-energy history, persistence/trend rules, and
ExtraTrees.  Hyperparameters are chosen on the official validation split only;
the formula-disjoint test split is touched once.  Reservoir seeds are
ensembled, never selected.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
TRANSITION = ROOT.parent
DATA = TRANSITION / "data/processed/task1/transition1x_task1_q8_t6_n3000.npz"
TASK3 = TRANSITION / "task3"
SEEDS = (0, 10, 20)
ALPHAS = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
BOOTSTRAP_REPEATS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def masked_mae(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(y))[mask]))


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> tuple[np.ndarray, float, float]:
    scaler = StandardScaler().fit(x[train])
    x_train = scaler.transform(x[train])
    x_validation = scaler.transform(x[validation])
    candidates = []
    for alpha in ALPHAS:
        model = Ridge(alpha=float(alpha), solver="lsqr").fit(x_train, y[train])
        candidates.append((score(y[validation], model.predict(x_validation), validation), alpha))
    validation_score, alpha = min(candidates)
    train_validation = train | validation
    scaler = StandardScaler().fit(x[train_validation])
    model = Ridge(alpha=float(alpha), solver="lsqr").fit(
        scaler.transform(x[train_validation]), y[train_validation]
    )
    return model.predict(scaler.transform(x[test])), float(alpha), float(validation_score)


def fit_extra_trees(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    seed: int,
) -> tuple[np.ndarray, dict, float]:
    candidates = []
    for leaf in (1, 2, 4, 8, 16):
        for fraction in (0.5, 1.0):
            model = ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=leaf,
                max_features=fraction,
                random_state=seed,
                n_jobs=-1,
            ).fit(x[train], y[train])
            candidates.append(
                (score(y[validation], model.predict(x[validation]), validation), leaf, fraction)
            )
    validation_score, leaf, fraction = min(candidates)
    train_validation = train | validation
    model = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=int(leaf),
        max_features=float(fraction),
        random_state=seed + 1,
        n_jobs=-1,
    ).fit(x[train_validation], y[train_validation])
    return (
        model.predict(x[test]),
        {"min_samples_leaf": int(leaf), "max_features": float(fraction)},
        float(validation_score),
    )


def per_reaction_errors(
    target_rho: np.ndarray,
    target_energy: np.ndarray,
    prediction_rho: np.ndarray,
    prediction_energy: np.ndarray,
    valid: np.ndarray,
    changed: np.ndarray,
) -> dict[str, np.ndarray]:
    absolute = np.abs(prediction_rho - target_rho)
    return {
        "rho_all": np.sum(absolute * valid, axis=1) / np.maximum(valid.sum(axis=1), 1),
        "rho_changed": np.sum(absolute * changed, axis=1) / np.maximum(changed.sum(axis=1), 1),
        "energy": np.abs(prediction_energy - target_energy),
    }


def bootstrap_interval(
    improvement: np.ndarray,
    groups: np.ndarray | None,
    repeats: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    values = np.asarray(improvement, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    if groups is None:
        for index in range(repeats):
            draws[index] = np.mean(values[rng.integers(0, len(values), len(values))])
        estimate = float(np.mean(values))
    else:
        labels = np.unique(groups)
        group_means = np.asarray([np.mean(values[groups == label]) for label in labels])
        for index in range(repeats):
            draws[index] = np.mean(
                group_means[rng.integers(0, len(group_means), len(group_means))]
            )
        estimate = float(np.mean(group_means))
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "confidence": float(confidence),
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    split = arrays["splits"]
    train, validation, test = split == "train", split == "val", split == "test"
    formulas = arrays["formulas"]
    formula_sets = {name: set(formulas[split == name]) for name in ("train", "val", "test")}
    if formula_sets["train"] & formula_sets["val"] or formula_sets["train"] & formula_sets["test"] or formula_sets["val"] & formula_sets["test"]:
        raise AssertionError("formula leakage across official splits")

    target_rho = arrays["target_delta_rho"].astype(np.float64)
    target_energy = arrays["target_delta_energy"].astype(np.float64)
    valid = arrays["valid_pair_mask"]
    changed = valid & arrays["changed_pair_mask"]
    tokens = arrays["tokens"].astype(np.float64)
    energy_history = arrays["relative_energy_paths"][:, : tokens.shape[1]].astype(np.float64)
    x_geometry = tokens.reshape(len(tokens), -1)
    x_energy = np.concatenate([x_geometry, energy_history, np.diff(energy_history, axis=1)], axis=1)

    def rho_score(y_true: np.ndarray, prediction: np.ndarray, rows: np.ndarray) -> float:
        return masked_mae(y_true, prediction, valid[rows])

    def energy_score(y_true: np.ndarray, prediction: np.ndarray, _rows: np.ndarray) -> float:
        return float(np.mean(np.abs(prediction - y_true)))

    predictions: dict[str, dict[str, np.ndarray]] = {}
    selections: list[dict] = []

    # Non-parametric physical controls.
    last_increment = tokens[:, -1].reshape(len(tokens), 8, 3)[:, :, 2] * 0.4
    energy_delta = np.diff(energy_history, axis=1)
    for name, rho_prediction, energy_prediction in (
        ("ZERO", np.zeros_like(target_rho[test]), np.zeros(test.sum())),
        ("PERSIST_LAST", last_increment[test], energy_delta[test, -1]),
        ("PERSIST_MEAN2", np.mean(tokens[:, -2:].reshape(len(tokens), 2, 8, 3)[:, :, :, 2] * 0.4, axis=1)[test], np.mean(energy_delta[test, -2:], axis=1)),
    ):
        predictions[name] = {"rho": rho_prediction, "energy": energy_prediction}

    # Tuned raw trajectory controls.
    for name, features in (("RAW_GEOMETRY", x_geometry), ("RAW_GEOMETRY_ENERGY", x_energy)):
        rho_prediction, rho_alpha, rho_validation = fit_ridge(
            features, target_rho, train, validation, test, rho_score
        )
        energy_prediction, energy_alpha, energy_validation = fit_ridge(
            features, target_energy, train, validation, test, energy_score
        )
        predictions[f"{name}_RIDGE"] = {"rho": rho_prediction, "energy": energy_prediction}
        selections.extend([
            {"model": f"{name}_RIDGE", "target": "rho", "alpha": rho_alpha, "validation_mae": rho_validation},
            {"model": f"{name}_RIDGE", "target": "energy", "alpha": energy_alpha, "validation_mae": energy_validation},
        ])
        rho_prediction, rho_parameters, rho_validation = fit_extra_trees(
            features, target_rho, train, validation, test, rho_score, 260_820
        )
        energy_prediction, energy_parameters, energy_validation = fit_extra_trees(
            features, target_energy, train, validation, test, energy_score, 260_821
        )
        predictions[f"{name}_EXTRA_TREES"] = {"rho": rho_prediction, "energy": energy_prediction}
        selections.extend([
            {"model": f"{name}_EXTRA_TREES", "target": "rho", **rho_parameters, "validation_mae": rho_validation},
            {"model": f"{name}_EXTRA_TREES", "target": "energy", **energy_parameters, "validation_mae": energy_validation},
        ])

    # Existing temporal reservoirs. FC216 projection is fitted on training rows only.
    seeded: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in ("QRC_F", "QRC_FC", "QRC_FC216", "ESN_F")}
    for seed in SEEDS:
        with np.load(TASK3 / f"features/qrc_ablation_seed{seed:03d}.npz", allow_pickle=False) as package:
            feature_sets = {"QRC_F": package["F"].astype(np.float64), "QRC_FC": package["FC"].astype(np.float64)}
        scaler = StandardScaler().fit(feature_sets["QRC_FC"][train])
        standardized = scaler.transform(feature_sets["QRC_FC"])
        pca = PCA(n_components=216, svd_solver="randomized", iterated_power=5, random_state=seed).fit(standardized[train])
        feature_sets["QRC_FC216"] = pca.transform(standardized)
        with np.load(TASK3 / f"features/esn_temporal_seed{seed:03d}.npz", allow_pickle=False) as package:
            feature_sets["ESN_F"] = package["features"].astype(np.float64)
        for model_name, features in feature_sets.items():
            rho_prediction, rho_alpha, rho_validation = fit_ridge(
                features, target_rho, train, validation, test, rho_score
            )
            energy_prediction, energy_alpha, energy_validation = fit_ridge(
                features, target_energy, train, validation, test, energy_score
            )
            seeded[model_name].append({"rho": rho_prediction, "energy": energy_prediction})
            selections.extend([
                {"model": model_name, "seed": seed, "target": "rho", "alpha": rho_alpha, "validation_mae": rho_validation},
                {"model": model_name, "seed": seed, "target": "energy", "alpha": energy_alpha, "validation_mae": energy_validation},
            ])
    for model_name, seed_predictions in seeded.items():
        predictions[model_name] = {
            target: np.mean([item[target] for item in seed_predictions], axis=0)
            for target in ("rho", "energy")
        }

    errors = {
        name: per_reaction_errors(
            target_rho[test], target_energy[test], values["rho"], values["energy"], valid[test], changed[test]
        )
        for name, values in predictions.items()
    }
    metrics = []
    for name, by_metric in errors.items():
        for metric, values in by_metric.items():
            metrics.append({"model": name, "metric": metric, "mae": float(np.mean(values))})

    classical_names = [name for name in predictions if not name.startswith("QRC_") and name != "ESN_F"]
    qrc_names = ("QRC_F", "QRC_FC", "QRC_FC216")
    comparisons = []
    hypothesis_count = len(qrc_names) * 3 * 2
    adjusted_confidence = 1.0 - 0.05 / hypothesis_count
    test_formulas = formulas[test]
    for qrc_name in qrc_names:
        for metric in ("rho_all", "rho_changed", "energy"):
            strongest_classical = min(classical_names, key=lambda name: float(np.mean(errors[name][metric])))
            for baseline in ("ESN_F", strongest_classical):
                improvement = errors[baseline][metric] - errors[qrc_name][metric]
                reaction_ci = bootstrap_interval(improvement, None, BOOTSTRAP_REPEATS, 71_000 + len(comparisons), adjusted_confidence)
                formula_ci = bootstrap_interval(improvement, test_formulas, BOOTSTRAP_REPEATS, 91_000 + len(comparisons), adjusted_confidence)
                comparisons.append({
                    "qrc": qrc_name,
                    "baseline": baseline,
                    "metric": metric,
                    "qrc_mae": float(np.mean(errors[qrc_name][metric])),
                    "baseline_mae": float(np.mean(errors[baseline][metric])),
                    "relative_improvement_percent": float(100.0 * np.mean(improvement) / np.mean(errors[baseline][metric])),
                    "reaction_bootstrap": reaction_ci,
                    "formula_cluster_bootstrap": formula_ci,
                    "strict_win": bool(reaction_ci["lower"] > 0.0 and formula_ci["lower"] > 0.0),
                })

    pairwise_strict_wins = [row for row in comparisons if row["strict_win"]]
    # A commercial advantage is not established by beating only the matched
    # ESN.  The same QRC/metric pair must also beat the strongest raw classical
    # model under both reaction- and formula-cluster intervals.
    full_gate_wins = []
    for qrc_name in qrc_names:
        for metric in ("rho_all", "rho_changed", "energy"):
            rows = [
                row for row in comparisons
                if row["qrc"] == qrc_name and row["metric"] == metric
            ]
            if len(rows) == 2 and all(row["strict_win"] for row in rows):
                full_gate_wins.append({"qrc": qrc_name, "metric": metric, "comparisons": rows})
    payload = {
        "status": "completed",
        "decision": "reject_as_strict_commercial_advantage" if not full_gate_wins else "candidate_survives",
        "scope": "known-reactant-and-product reaction-path continuation; endpoint-conditioned pair selector",
        "dataset": str(DATA),
        "dataset_sha256": sha256(DATA),
        "split_counts": {name: int(np.sum(split == name)) for name in ("train", "val", "test")},
        "formula_counts": {name: len(formula_sets[name]) for name in ("train", "val", "test")},
        "formula_disjoint": True,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "multiplicity_hypotheses": hypothesis_count,
        "metrics": metrics,
        "selections": selections,
        "comparisons": comparisons,
        "pairwise_strict_wins": pairwise_strict_wins,
        "full_gate_wins": full_gate_wins,
        "limitations": [
            "The pair selector uses reactant and product endpoints; this is valid only when both endpoints are known.",
            "The official test set contains only eight formula groups, so formula-cluster intervals are intentionally conservative.",
            "This audit rejects an existing candidate; it does not establish that ExtraTrees is deployable without prospective validation.",
        ],
    }
    (ROOT / "audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "metric", "mae"))
        writer.writeheader(); writer.writerows(metrics)
    np.savez_compressed(
        ROOT / "test_predictions.npz",
        reaction_ids=arrays["reaction_ids"][test],
        formulas=test_formulas,
        target_rho=target_rho[test],
        target_energy=target_energy[test],
        **{f"{name}__rho": values["rho"] for name, values in predictions.items()},
        **{f"{name}__energy": values["energy"] for name, values in predictions.items()},
    )
    best = {(metric): min((row for row in metrics if row["metric"] == metric), key=lambda row: row["mae"]) for metric in ("rho_all", "rho_changed", "energy")}
    lines = [
        "# Strict Transition1x commercial audit",
        "",
        f"Decision: **{payload['decision']}**.",
        "",
        "The original temporal-QRC edge does not count unless it survives early-energy-history and nonlinear raw-trajectory controls.",
        "",
        "## Best held-out-test model by metric",
        "",
        "| Metric | Model | MAE |",
        "|---|---|---:|",
        *[f"| {metric} | {best[metric]['model']} | {best[metric]['mae']:.6f} |" for metric in ("rho_all", "rho_changed", "energy")],
        "",
        "Full hyperparameter choices, paired reaction bootstraps, and eight-formula cluster bootstraps are in `audit.json`.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"], "best": best,
        "pairwise_strict_wins": len(pairwise_strict_wins),
        "full_gate_wins": len(full_gate_wins),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
