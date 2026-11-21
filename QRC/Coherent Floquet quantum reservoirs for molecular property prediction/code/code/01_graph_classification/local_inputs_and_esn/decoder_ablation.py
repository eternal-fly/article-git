"""Decoder-only sensitivity analysis on affine-selected reservoir states.

The reservoir representation is frozen by the completed manuscript affine
protocol.  This module changes only the decoder, which makes the comparison an
auditable test of whether the headline ordering is decoder-dependent.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, pairwise_distances, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from esn_features import load_esn_features
from protocol import PROTOCOL_VERSION, RESULT_ROOT, split_path, token_cache_path
from qrc_features import load_qrc_bank, mix_suffix_bank
from readouts import _indices, metric_bundle, metric_bundles


DECODER_PROTOCOL = "affine-selected-representation-decoder-ablation-v1"
SYSTEMS = (
    "QRC_CLOSED",
    "QRC_OPEN",
    "ESN_MATCHED",
    "ESN_BEST_WIDTH",
)
DECODERS = ("SVR_RBF", "REG_MLP")

SVR_C = (0.1, 1.0, 10.0)
SVR_GAMMA_SCALES = (0.25, 1.0, 4.0)
SVR_EPSILON = 0.01

MLP_WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)
MLP_DROPOUT = 0.20
MLP_EPOCHS = 500
MLP_BATCH_SIZE = 32
MLP_LR = 1e-3


def decoder_result_dir(
    dataset: str, variant: str, system: str, decoder: str, seed: int
) -> Path:
    return (
        RESULT_ROOT
        / "decoder_ablation"
        / dataset.upper()
        / variant
        / system
        / decoder
        / f"seed{int(seed):03d}"
    )


def _selection_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qrc = pd.read_csv(RESULT_ROOT / "qrc_validation_selected_q_t_by_seed.csv")
    matched = pd.read_csv(
        RESULT_ROOT / "esn_matched_validation_selected_q_t_by_seed.csv"
    )
    best = pd.read_csv(RESULT_ROOT / "esn_validation_selected_q_width_t_by_seed.csv")
    return qrc, matched, best


def selected_case_rows() -> pd.DataFrame:
    """Return the 96 frozen representations used by every new decoder."""
    qrc, matched, best = _selection_tables()
    rows: list[dict] = []
    for _, row in qrc.iterrows():
        rows.append(
            {
                "dataset": row.dataset,
                "token_variant": row.token_variant,
                "system": row.system,
                "seed": int(row.seed),
                "q": int(row.q),
                "t": int(row.t),
                "p": float(row.p),
                "width": np.nan,
                "affine_test_auc": float(row.test_auc),
                "affine_test_acc": float(row.test_acc),
                "affine_test_brier": float(row.test_brier),
            }
        )
    for system, table in (("ESN_MATCHED", matched), ("ESN_BEST_WIDTH", best)):
        for _, row in table.iterrows():
            rows.append(
                {
                    "dataset": row.dataset,
                    "token_variant": row.token_variant,
                    "system": system,
                    "seed": int(row.seed),
                    "q": int(row.q),
                    "t": int(row.t),
                    "p": np.nan,
                    "width": int(row.width),
                    "affine_test_auc": float(row.test_auc),
                    "affine_test_acc": float(row.test_acc),
                    "affine_test_brier": float(row.test_brier),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "token_variant", "system", "seed"]
    )
    expected = 2 * 2 * len(SYSTEMS) * 6
    if len(frame) != expected or frame.duplicated(
        ["dataset", "token_variant", "system", "seed"]
    ).any():
        raise RuntimeError(f"invalid frozen decoder cases: {len(frame)} != {expected}")
    return frame.reset_index(drop=True)


def load_case_features(row: dict | pd.Series) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    dataset = str(row["dataset"])
    variant = str(row["token_variant"])
    system = str(row["system"])
    seed = int(row["seed"])
    q = int(row["q"])
    t = int(row["t"])
    with np.load(token_cache_path(dataset, variant), allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    if system.startswith("QRC_"):
        suffixes = load_qrc_bank(
            dataset,
            variant,
            q,
            t,
            seed,
            n_samples=len(labels),
            token_sha256=token_sha,
        )
        probability = 0.0 if system == "QRC_CLOSED" else float(row["p"])
        features = mix_suffix_bank(suffixes, probability).astype(np.float32)
        selection = {"q": q, "t": t, "p": probability, "width": None}
    else:
        width = int(row["width"])
        features = load_esn_features(
            dataset,
            variant,
            q,
            t,
            width,
            seed,
            n_samples=len(labels),
            token_sha256=token_sha,
        ).astype(np.float32)
        selection = {"q": q, "t": t, "p": None, "width": width}
    split = json.loads(split_path(dataset, seed).read_text(encoding="utf-8"))
    return features, labels, split, selection


def _score_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Metrics for a non-probabilistic SVR score, matching prior BACE work."""
    y = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    bounded = np.clip(score, 0.0, 1.0)
    return {
        "auc": float(roc_auc_score(y, score)),
        "acc": float(accuracy_score(y, bounded >= 0.5)),
        "brier": float(np.mean((bounded - y) ** 2)),
        "rmse": float(np.sqrt(mean_squared_error(y, bounded))),
    }


def _base_gamma(x_train_scaled: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    count = min(512, len(x_train_scaled))
    selected = rng.choice(len(x_train_scaled), size=count, replace=False)
    distances = pairwise_distances(x_train_scaled[selected], metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    nonzero = upper[upper > 1e-12]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    return 1.0 / max(median, 1e-12)


def fit_svr(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
    destination: Path,
    decoder_protocol: str = DECODER_PROTOCOL,
) -> dict:
    train, val, test = _indices(split)
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_val = scaler.transform(features[val])
    x_test = scaler.transform(features[test])
    base_gamma = _base_gamma(x_train, seed)
    rows = []
    best_key = None
    best_model = None
    best_row = None
    for c_value in SVR_C:
        for gamma_scale in SVR_GAMMA_SCALES:
            gamma = float(gamma_scale * base_gamma)
            model = SVR(
                kernel="rbf",
                C=float(c_value),
                gamma=gamma,
                epsilon=SVR_EPSILON,
            )
            model.fit(x_train, labels[train])
            metrics = _score_metrics(labels[val], model.predict(x_val))
            row = {
                "C": float(c_value),
                "gamma_scale": float(gamma_scale),
                "gamma": gamma,
                "epsilon": SVR_EPSILON,
                **{f"val_{name}": value for name, value in metrics.items()},
            }
            rows.append(row)
            key = (
                metrics["auc"],
                -metrics["brier"],
                -float(c_value),
                -float(gamma_scale),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_model = model
                best_row = row
    if best_model is None or best_row is None:
        raise RuntimeError("no valid SVR candidate")
    test_score = best_model.predict(x_test)
    test_metrics = _score_metrics(labels[test], test_score)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "protocol_version": PROTOCOL_VERSION,
            "decoder_protocol": decoder_protocol,
            "scaler": scaler,
            "model": best_model,
            "seed": int(seed),
            "selected": best_row,
        },
        destination / "model.joblib",
    )
    pd.DataFrame(rows).to_csv(destination / "validation_screen.csv", index=False)
    pd.DataFrame(
        {
            "sample_index": test,
            "y_true": labels[test],
            "score": test_score,
            "probability_like": np.clip(test_score, 0.0, 1.0),
            "prediction": (np.clip(test_score, 0.0, 1.0) >= 0.5).astype(int),
        }
    ).to_csv(destination / "test_predictions.csv", index=False)
    return {
        "feature_dim": int(features.shape[1]),
        "selected_C": best_row["C"],
        "selected_gamma_scale": best_row["gamma_scale"],
        "selected_gamma": best_row["gamma"],
        "selected_epsilon": best_row["epsilon"],
        **{key: value for key, value in best_row.items() if key.startswith("val_")},
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }


class RegularizedMLP(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hidden = min(256, max(128, int(2 ** math.ceil(math.log2(max(64, input_dim // 2))))))
        hidden2 = min(128, max(64, hidden // 2))
        self.hidden = hidden
        self.hidden2 = hidden2
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Dropout(MLP_DROPOUT),
            torch.nn.Linear(hidden, hidden2),
            torch.nn.GELU(),
            torch.nn.Dropout(MLP_DROPOUT),
            torch.nn.Linear(hidden2, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(1)


def fit_regularized_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
    device: str,
    destination: Path,
    decoder_protocol: str = DECODER_PROTOCOL,
) -> dict:
    train_np, val_np, test_np = _indices(split)
    scaler = StandardScaler().fit(features[train_np])
    scaled = scaler.transform(features).astype(np.float32)
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"regularized MLP requested unavailable device {device}")
    if compute_device.type == "cuda":
        torch.cuda.set_device(compute_device)
    x = torch.from_numpy(scaled).to(compute_device)
    y = torch.from_numpy(labels.astype(np.float32)).to(compute_device)
    train = torch.from_numpy(train_np).to(compute_device)
    val = torch.from_numpy(val_np).to(compute_device)
    test = torch.from_numpy(test_np).to(compute_device)
    candidate_rows = []
    candidate_artifacts = []
    history_rows = []
    for weight_decay in MLP_WEIGHT_DECAYS:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = RegularizedMLP(x.shape[1]).to(compute_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=MLP_LR, weight_decay=float(weight_decay)
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        generator = torch.Generator(device=compute_device).manual_seed(int(seed))
        best_key = None
        best_epoch = -1
        best_state = None
        best_validation = None
        for epoch in range(1, MLP_EPOCHS + 1):
            model.train()
            order = train[
                torch.randperm(len(train), generator=generator, device=compute_device)
            ]
            losses = []
            for start in range(0, len(order), MLP_BATCH_SIZE):
                batch = order[start : start + MLP_BATCH_SIZE]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val_probability = torch.sigmoid(model(x[val])).cpu().numpy()
            validation = metric_bundles(
                labels[val_np], val_probability[None, :]
            )[0]
            key = (validation["auc"], -validation["loss"], validation["acc"])
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_validation = validation
                best_state = copy.deepcopy(
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                )
            history_rows.append(
                {
                    "weight_decay": float(weight_decay),
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    **{f"val_{name}": value for name, value in validation.items()},
                }
            )
        if best_state is None or best_validation is None:
            raise RuntimeError(f"no MLP checkpoint for weight_decay={weight_decay}")
        candidate_rows.append(
            {
                "weight_decay": float(weight_decay),
                "best_epoch": int(best_epoch),
                **{f"val_{name}": value for name, value in best_validation.items()},
            }
        )
        candidate_artifacts.append(
            {
                "weight_decay": float(weight_decay),
                "best_epoch": int(best_epoch),
                "validation": best_validation,
                "state_dict": best_state,
                "hidden": int(model.hidden),
                "hidden2": int(model.hidden2),
            }
        )
        del model, optimizer
    ranking = sorted(
        range(len(candidate_rows)),
        key=lambda index: (
            candidate_rows[index]["val_auc"],
            -candidate_rows[index]["val_loss"],
            candidate_rows[index]["val_acc"],
            candidate_rows[index]["weight_decay"],
        ),
        reverse=True,
    )
    selected_index = ranking[0]
    selected = candidate_artifacts[selected_index]
    model = RegularizedMLP(x.shape[1]).to(compute_device)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    with torch.no_grad():
        test_probability = torch.sigmoid(model(x[test])).cpu().numpy()
    test_metrics = metric_bundle(labels[test_np], test_probability)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "protocol_version": PROTOCOL_VERSION,
            "decoder_protocol": decoder_protocol,
            "scaler": scaler,
        },
        destination / "scaler.joblib",
    )
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "decoder_protocol": decoder_protocol,
            "model": "RegularizedMLP",
            "state_dict": selected["state_dict"],
            "input_dim": int(features.shape[1]),
            "hidden": selected["hidden"],
            "hidden2": selected["hidden2"],
            "dropout": MLP_DROPOUT,
            "weight_decay": selected["weight_decay"],
            "best_epoch": selected["best_epoch"],
            "seed": int(seed),
        },
        destination / "model.pt",
    )
    pd.DataFrame(history_rows).to_csv(destination / "training_history.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(
        destination / "validation_screen.csv", index=False
    )
    pd.DataFrame(
        {
            "sample_index": test_np,
            "y_true": labels[test_np],
            "probability": test_probability,
            "prediction": (test_probability >= 0.5).astype(int),
        }
    ).to_csv(destination / "test_predictions.csv", index=False)
    validation = selected["validation"]
    return {
        "feature_dim": int(features.shape[1]),
        "hidden": selected["hidden"],
        "hidden2": selected["hidden2"],
        "dropout": MLP_DROPOUT,
        "selected_weight_decay": selected["weight_decay"],
        "best_epoch": selected["best_epoch"],
        **{f"val_{name}": value for name, value in validation.items()},
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }


def evaluate_case(
    row: dict | pd.Series,
    decoder: str,
    *,
    device: str = "cpu",
    force: bool = False,
) -> dict:
    dataset = str(row["dataset"])
    variant = str(row["token_variant"])
    system = str(row["system"])
    seed = int(row["seed"])
    destination = decoder_result_dir(dataset, variant, system, decoder, seed)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not force:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_version") == PROTOCOL_VERSION
            and payload.get("decoder_protocol") == DECODER_PROTOCOL
        ):
            return payload
    features, labels, split, selection = load_case_features(row)
    if decoder == "SVR_RBF":
        metrics = fit_svr(
            features, labels, split, seed=seed, destination=destination
        )
    elif decoder == "REG_MLP":
        metrics = fit_regularized_mlp(
            features,
            labels,
            split,
            seed=seed,
            device=device,
            destination=destination,
        )
    else:
        raise ValueError(f"unsupported decoder {decoder}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "decoder_protocol": DECODER_PROTOCOL,
        "selection_protocol": "reservoir fixed by manuscript affine validation selection",
        "dataset": dataset.upper(),
        "token_variant": variant,
        "system": system,
        "decoder": decoder,
        "seed": seed,
        **selection,
        **metrics,
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
