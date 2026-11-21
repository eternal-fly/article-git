"""Paired splitting, linear probes, statistics, and plots for controlled streams."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover - statistics still save without SciPy
    ttest_rel = None
    wilcoxon = None


METHODS = (
    "qrc",
    "crc",
    "crc3_z_zz_matched",
    "esn_standard",
    "esn",
    "raw_full",
    "raw_pca",
    "raw_logreg_cv",
)
METHOD_LABELS = {
    "qrc": "QRC (Z+ZZ)",
    "crc": "CRC (scalar-Q)",
    "crc3_z_zz_matched": "CRC3 (z+zz, no PCA)",
    "esn_standard": "Standard ESN",
    "esn": "Orthogonal ESN",
    "raw_full": "Raw-full",
    "raw_pca": "Raw-PCA",
    "raw_logreg_cv": "Raw standardized LR",
}
METHOD_COLORS = {
    "qrc": "#2F6B9A",
    "crc": "#D07A35",
    "crc3_z_zz_matched": "#C44E52",
    "esn_standard": "#E9A23B",
    "esn": "#3B8C6E",
    "raw_full": "#6F6F6F",
    "raw_pca": "#8E5EA2",
    "raw_logreg_cv": "#B24A63",
}
METHOD_STYLES = {
    "qrc": "-",
    "crc": "--",
    "crc3_z_zz_matched": (0, (5, 1)),
    "esn_standard": ":",
    "esn": "-.",
    "raw_full": ":",
    "raw_pca": (0, (3, 1, 1, 1)),
    "raw_logreg_cv": (0, (5, 2)),
}
SELECTION_RULE = "max(val_auc) > min(val_loss) > max(val_acc)"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _save_figure(fig: plt.Figure, png_path: Path) -> None:
    """Save a preview plus editable-vector publication sources."""

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".svg"), bbox_inches="tight")


def resolve_device(value: str | None) -> torch.device:
    raw = "auto" if value is None else str(value).lower()
    if raw in {"", "auto", "cuda", "gpu"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def grouped_split(
    groups: np.ndarray,
    labels: np.ndarray,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[int]]:
    """Split nuisance-matched groups, never paired variants, across partitions."""

    groups = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three nuisance groups are required")
    rng = np.random.default_rng(int(seed))
    unique = unique[rng.permutation(len(unique))]
    n_train = max(1, int(math.floor(len(unique) * float(train_ratio))))
    n_val = max(1, int(math.floor(len(unique) * float(val_ratio))))
    if n_train + n_val >= len(unique):
        n_train = max(1, len(unique) - 2)
        n_val = 1
    partitions = {
        "train_indices": np.flatnonzero(np.isin(groups, unique[:n_train])).tolist(),
        "val_indices": np.flatnonzero(np.isin(groups, unique[n_train : n_train + n_val])).tolist(),
        "test_indices": np.flatnonzero(np.isin(groups, unique[n_train + n_val :])).tolist(),
    }
    for name, indices in partitions.items():
        if set(np.unique(labels[np.asarray(indices, dtype=np.int64)]).tolist()) != {0, 1}:
            raise RuntimeError(f"{name} lacks one class; increase n_groups")
    return partitions


def stratified_split(
    labels: np.ndarray,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[int]]:
    """Paired method split for optional BACE/BBBP token replay."""

    labels = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(len(labels), dtype=np.int64)
    train_idx, holdout_idx = train_test_split(
        all_indices,
        train_size=float(train_ratio),
        random_state=int(seed),
        stratify=labels,
    )
    relative_val = float(val_ratio) / (1.0 - float(train_ratio))
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=relative_val,
        random_state=int(seed) + 104729,
        stratify=labels[holdout_idx],
    )
    return {
        "train_indices": sorted(int(i) for i in train_idx),
        "val_indices": sorted(int(i) for i in val_idx),
        "test_indices": sorted(int(i) for i in test_idx),
    }


def metric_bundle(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(y, probabilities)),
        "acc": float(accuracy_score(y, predictions)),
    }


def fit_raw_pca(
    raw_features: np.ndarray,
    split: dict[str, list[int]],
    target_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit PCA only on training samples and align width to the QRC readout."""

    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    n_components = min(int(target_dim), raw_features.shape[1], len(train_idx))
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(raw_features[train_idx])
    projected = pca.transform(raw_features).astype(np.float32)
    if n_components < target_dim:
        padded = np.zeros((len(projected), int(target_dim)), dtype=np.float32)
        padded[:, :n_components] = projected
        projected = padded
    return projected, {
        "n_components": int(n_components),
        "target_dim": int(target_dim),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "fit_scope": "train_only",
    }


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, list[int]],
    seed: int,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor]]:
    """FINAL-aligned unstandardized affine probe with validation-only selection."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    compute_device = resolve_device(device)
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(compute_device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(compute_device)
    train_np = np.asarray(split["train_indices"], dtype=np.int64)
    val_np = np.asarray(split["val_indices"], dtype=np.int64)
    test_np = np.asarray(split["test_indices"], dtype=np.int64)
    train_idx = torch.from_numpy(train_np).to(compute_device)
    val_idx = torch.from_numpy(val_np).to(compute_device)
    test_idx = torch.from_numpy(test_np).to(compute_device)
    model = torch.nn.Linear(x.shape[1], 1, bias=True).to(compute_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator(device=compute_device).manual_seed(int(seed))
    best_key: tuple[float, float, float] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_val: dict[str, float] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator, device=compute_device)]
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            batch = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch]).squeeze(1), y[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(x[val_idx]).squeeze(1).cpu().numpy()
        val = metric_bundle(labels[val_np], val_logits)
        key = (val["auc"], -val["loss"], val["acc"])
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_val = val
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_auc": val["auc"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
            }
        )
    if best_state is None or best_val is None:
        raise RuntimeError("No valid linear-probe checkpoint")
    model.load_state_dict(best_state)
    model.to(compute_device).eval()
    with torch.no_grad():
        test_logits = model(x[test_idx]).squeeze(1).cpu().numpy()
    test = metric_bundle(labels[test_np], test_logits)
    probability = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40.0, 40.0)))
    metrics = {
        "feature_dim": int(features.shape[1]),
        "best_epoch": int(best_epoch),
        "val_auc": best_val["auc"],
        "val_loss": best_val["loss"],
        "val_acc": best_val["acc"],
        "test_auc": test["auc"],
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "selection_rule": SELECTION_RULE,
        "standardized": False,
        "regularized": False,
        "train_device": str(compute_device),
    }
    predictions = pd.DataFrame(
        {
            "sample_index": test_np,
            "y_true": labels[test_np],
            "probability": probability,
            "prediction": (probability >= 0.5).astype(np.int64),
        }
    )
    return metrics, pd.DataFrame(history), predictions, best_state


def train_strong_raw_logreg(
    raw_features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, list[int]],
    seed: int,
    c_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train-only standardization plus validation-selected L2 logistic regression."""

    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    val_idx = np.asarray(split["val_indices"], dtype=np.int64)
    test_idx = np.asarray(split["test_indices"], dtype=np.int64)
    scaler = StandardScaler().fit(raw_features[train_idx])
    x_train = scaler.transform(raw_features[train_idx])
    x_val = scaler.transform(raw_features[val_idx])
    x_test = scaler.transform(raw_features[test_idx])
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_model: LogisticRegression | None = None
    best_val: dict[str, float] | None = None
    best_c = float("nan")
    for c in c_grid:
        model = LogisticRegression(
            C=float(c),
            solver="lbfgs",
            max_iter=5000,
            random_state=int(seed),
        )
        model.fit(x_train, labels[train_idx])
        val_logits = model.decision_function(x_val)
        val = metric_bundle(labels[val_idx], val_logits)
        key = (val["auc"], -val["loss"], val["acc"])
        history.append({"C": float(c), **{f"val_{key}": value for key, value in val.items()}})
        if best_key is None or key > best_key:
            best_key = key
            best_model = copy.deepcopy(model)
            best_val = val
            best_c = float(c)
    if best_model is None or best_val is None:
        raise RuntimeError("No valid logistic-regression model")
    test_logits = best_model.decision_function(x_test)
    test = metric_bundle(labels[test_idx], test_logits)
    probability = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40.0, 40.0)))
    metrics = {
        "feature_dim": int(raw_features.shape[1]),
        "best_epoch": -1,
        "best_C": best_c,
        "val_auc": best_val["auc"],
        "val_loss": best_val["loss"],
        "val_acc": best_val["acc"],
        "test_auc": test["auc"],
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "selection_rule": SELECTION_RULE,
        "standardized": True,
        "regularized": True,
        "train_device": "sklearn_cpu",
    }
    predictions = pd.DataFrame(
        {
            "sample_index": test_idx,
            "y_true": labels[test_idx],
            "probability": probability,
            "prediction": (probability >= 0.5).astype(np.int64),
        }
    )
    serializable = {
        "coef": best_model.coef_.astype(np.float32),
        "intercept": best_model.intercept_.astype(np.float32),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "best_C": np.float32(best_c),
    }
    return metrics, pd.DataFrame(history), predictions, serializable


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["setting", "family", "parameter_name", "parameter_value", "method"]
    for values, subset in metrics.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, values))
        row["n_seeds"] = int(len(subset))
        for metric in ("val_auc", "val_acc", "test_auc", "test_acc"):
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1)) if len(subset) > 1 else 0.0
            row[f"{metric}_sem"] = float(subset[metric].sem(ddof=1)) if len(subset) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(n_boot), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_statistics(metrics: pd.DataFrame, reference: str = "qrc") -> pd.DataFrame:
    """Seed-paired QRC differences, bootstrap CIs, t tests, and Wilcoxon tests."""

    rows: list[dict[str, Any]] = []
    for setting, subset in metrics.groupby("setting", sort=False):
        for metric in ("test_auc", "test_acc"):
            pivot = subset.pivot(index="seed", columns="method", values=metric)
            if reference not in pivot:
                continue
            for method in [name for name in pivot.columns if name != reference]:
                pair = pivot[[reference, method]].dropna()
                if pair.empty:
                    continue
                diff = (pair[reference] - pair[method]).to_numpy(dtype=float)
                digest = hashlib.sha256(f"{setting}|{metric}|{method}".encode("utf-8")).hexdigest()
                low, high = _bootstrap_ci(diff, seed=int(digest[:8], 16))
                t_p = float("nan")
                w_p = float("nan")
                if len(diff) >= 2 and ttest_rel is not None:
                    t_p = float(ttest_rel(pair[reference], pair[method]).pvalue)
                if len(diff) >= 2 and wilcoxon is not None and np.any(np.abs(diff) > 0):
                    try:
                        w_p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
                    except ValueError:
                        pass
                first = subset.iloc[0]
                rows.append(
                    {
                        "setting": setting,
                        "family": first["family"],
                        "parameter_name": first["parameter_name"],
                        "parameter_value": first["parameter_value"],
                        "metric": metric,
                        "reference": reference,
                        "comparison": method,
                        "n_pairs": int(len(diff)),
                        "mean_difference": float(diff.mean()),
                        "std_difference": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "wins": int((diff > 0).sum()),
                        "ties": int((diff == 0).sum()),
                        "losses": int((diff < 0).sum()),
                        "paired_t_p": t_p,
                        "wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(rows)


def plot_setting(summary: pd.DataFrame, setting: str, output_path: Path) -> None:
    subset = summary[summary["setting"] == setting].copy()
    if subset.empty:
        return
    order = [method for method in METHODS if method in set(subset["method"])]
    subset = subset.set_index("method").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for axis, metric, label in zip(axes, ("test_auc", "test_acc"), ("Test ROC-AUC", "Test accuracy")):
        y = subset[f"{metric}_mean"].to_numpy()
        err = subset[f"{metric}_std"].to_numpy()
        axis.bar(
            np.arange(len(subset)),
            y,
            yerr=err,
            color=[METHOD_COLORS[m] for m in subset["method"]],
            capsize=3,
            alpha=0.9,
        )
        axis.axhline(0.5, color="#444444", lw=0.8, ls=":")
        axis.set_ylim(max(0.0, min(0.45, float(np.nanmin(y - err)) - 0.05)), 1.02)
        axis.set_ylabel(label)
        axis.set_xticks(np.arange(len(subset)))
        axis.set_xticklabels([METHOD_LABELS[m] for m in subset["method"]], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(setting)
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_memory_curves(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot AUC/ACC against lag, spacing, or prefix length for each task family."""

    paths: list[Path] = []
    scannable = summary[summary["parameter_name"].isin(["lag", "spacing", "prefix_length"])]
    for family, family_frame in scannable.groupby("family", sort=False):
        if family_frame["parameter_value"].nunique() < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
        for method in METHODS:
            subset = family_frame[family_frame["method"] == method].copy()
            if subset.empty:
                continue
            subset["parameter_numeric"] = pd.to_numeric(subset["parameter_value"])
            subset = subset.sort_values("parameter_numeric")
            for axis, metric in zip(axes, ("test_auc", "test_acc")):
                axis.errorbar(
                    subset["parameter_numeric"],
                    subset[f"{metric}_mean"],
                    yerr=subset[f"{metric}_sem"],
                    marker="o",
                    ms=4,
                    lw=1.5,
                    color=METHOD_COLORS[method],
                    linestyle=METHOD_STYLES[method],
                    label=METHOD_LABELS[method],
                )
        x_label = str(family_frame.iloc[0]["parameter_name"]).replace("_", " ")
        for axis, label in zip(axes, ("Test ROC-AUC", "Test accuracy")):
            axis.axhline(0.5, color="#444444", lw=0.8, ls=":")
            axis.set_xlabel(x_label)
            axis.set_ylabel(label)
            axis.set_ylim(0.35, 1.02)
            axis.grid(alpha=0.2)
        axes[1].legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle(f"{family}: performance versus {x_label}")
        path = output_dir / f"{family}_memory_curve.png"
        _save_figure(fig, path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_overview(summary: pd.DataFrame, output_path: Path) -> None:
    pivot = summary.pivot(index="setting", columns="method", values="test_auc_mean")
    pivot = pivot.reindex(columns=[method for method in METHODS if method in pivot.columns])
    fig_height = max(5.0, 0.28 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, fig_height), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.45, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Controlled graph-event tasks: mean test ROC-AUC")
    fig.colorbar(image, ax=ax, label="ROC-AUC", shrink=0.7)
    _save_figure(fig, output_path)
    plt.close(fig)
