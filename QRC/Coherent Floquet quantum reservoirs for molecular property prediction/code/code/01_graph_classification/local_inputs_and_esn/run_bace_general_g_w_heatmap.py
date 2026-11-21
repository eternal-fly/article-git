#!/usr/bin/env python3
"""Full BACE/general g-by-W grid at frozen Q12/T30 and validated p.

This is a parameter-sensitivity map, not a g/W model-selection experiment.
The event-wise reset probability is frozen per seed from the completed affine
validation protocol at the manuscript anchor (g=0.84, W=0.5).  Every g/W test
point is retained and no g/W validation selection is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Mandatory publication/export settings: editable SVG and TrueType PDF text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

from decoder_ablation import fit_svr
from protocol import (
    LAYERS,
    OBSERVABLES,
    P_VALUES,
    PROTOCOL_VERSION,
    QRC_BACKEND,
    RESULT_ROOT,
    SEEDS,
    split_path,
    token_cache_path,
)
from qrc_features import parameter_sha256, refresh_suffix_weights


if str(QRC_BACKEND) not in sys.path:
    sys.path.insert(0, str(QRC_BACKEND))

from gpu_qrc_statevector import (  # noqa: E402
    DTCQRCConfig,
    adapt_token_dim,
    apply_reservoir_layers,
    apply_token_injection,
    build_dtc_diagonal_phase,
    build_observable_signs,
    compute_qrc_features_gpu,
    compute_z_zz_from_state,
    generate_dtc_params,
    real_dtype_from_complex,
    rx_gate,
    set_seed,
    torch_complex_dtype,
)


SCAN_PROTOCOL = "bace-general-q12-t30-open-fixed-p-svr-g-w-grid-v1"
DATASET = "BACE"
VARIANT = "canonical_general"
Q = 12
T = 30
P_VALIDATION_G = 0.84
P_VALIDATION_W = 0.50
G_VALUES = tuple(round(index / 10, 1) for index in range(1, 11))
W_VALUES = tuple(round(index / 10, 1) for index in range(0, 10))
GRID_STEP = 0.1
BATCH_SIZE = 32

SCAN_ROOT = RESULT_ROOT / "bace_general_g_w_heatmap"
FEATURE_ROOT = SCAN_ROOT / "features"
RUN_ROOT = SCAN_ROOT / "runs"
FIGURE_ROOT = SCAN_ROOT / "figures"
P_SOURCE_ROOT = (
    RESULT_ROOT / "qrc" / DATASET / VARIANT / f"Q{Q}" / f"T{T}"
)


def _tag(prefix: str, value: float) -> str:
    return f"{prefix}{float(value):.1f}".replace(".", "p")


def parameter_pairs() -> list[tuple[float, float]]:
    pairs = [(float(g), float(w)) for g in G_VALUES for w in W_VALUES]
    if len(pairs) != 100 or len(set(pairs)) != len(pairs):
        raise AssertionError("invalid 10-by-10 g/W grid")
    return pairs


def validated_p_by_seed() -> dict[int, float]:
    """Load the already completed, per-seed affine validation choices."""
    selected: dict[int, float] = {}
    for seed in SEEDS:
        path = P_SOURCE_ROOT / f"seed{int(seed):03d}" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"stale p-selection result {path}")
        candidates = [
            row
            for row in payload.get("rows", [])
            if row.get("system") == "QRC_OPEN"
            and int(row.get("seed", -1)) == int(seed)
            and int(row.get("q", -1)) == Q
            and int(row.get("t", -1)) == T
            and row.get("dataset") == DATASET
            and row.get("token_variant") == VARIANT
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one frozen-p row in {path}")
        probability = float(candidates[0]["p"])
        if probability not in P_VALUES:
            raise RuntimeError(f"invalid validated p={probability} in {path}")
        selected[int(seed)] = probability
    if set(selected) != set(SEEDS):
        raise RuntimeError("incomplete frozen-p map")
    return selected


def case_rows(mode: str = "full") -> pd.DataFrame:
    pairs = [(G_VALUES[0], W_VALUES[0])] if mode == "smoke" else parameter_pairs()
    seeds = (SEEDS[0],) if mode == "smoke" else SEEDS
    selected_p = validated_p_by_seed()
    frame = pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "token_variant": VARIANT,
                "q": Q,
                "t": T,
                "g": float(g),
                "W": float(w),
                "seed": int(seed),
                "p": float(selected_p[int(seed)]),
            }
            for g, w in pairs
            for seed in seeds
        ]
    )
    expected = len(pairs) * len(seeds)
    if len(frame) != expected or frame.duplicated(["g", "W", "seed"]).any():
        raise RuntimeError("invalid g/W heatmap cases")
    return frame


def feature_path(row: dict | pd.Series) -> Path:
    return (
        FEATURE_ROOT
        / _tag("g", float(row["g"]))
        / _tag("W", float(row["W"]))
        / f"features_seed{int(row['seed']):03d}.npz"
    )


def result_dir(row: dict | pd.Series) -> Path:
    return (
        RUN_ROOT
        / _tag("g", float(row["g"]))
        / _tag("W", float(row["W"]))
        / "SVR_RBF"
        / f"seed{int(row['seed']):03d}"
    )


def _load_inputs() -> tuple[np.ndarray, np.ndarray, str, str]:
    path = token_cache_path(DATASET, VARIANT)
    with np.load(path, allow_pickle=False) as archive:
        tokens = np.ascontiguousarray(
            archive["tokens"][:, :T, : 3 * Q], dtype=np.float32
        )
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if tokens.shape[1:] != (T, 3 * Q):
        raise RuntimeError(f"invalid BACE/general crop {tokens.shape}")
    return tokens, labels, token_sha, str(metadata["dataset_fingerprint"])


def load_feature(row: dict | pd.Series) -> np.ndarray:
    path = feature_path(row)
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        identity = {
            "protocol": str(archive["scan_protocol"].item()),
            "g": float(archive["g"]),
            "W": float(archive["W"]),
            "p": float(archive["p"]),
            "seed": int(archive["seed"]),
            "q": int(archive["q"]),
            "t": int(archive["t"]),
        }
    expected = {
        "protocol": SCAN_PROTOCOL,
        "g": float(np.float32(row["g"])),
        "W": float(np.float32(row["W"])),
        "p": float(np.float32(row["p"])),
        "seed": int(row["seed"]),
        "q": Q,
        "t": T,
    }
    if identity != expected or features.shape[1] != Q * (Q + 1) // 2:
        raise RuntimeError(f"stale or invalid feature cache {path}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"non-finite feature cache {path}")
    return features


@torch.inference_mode()
def compute_fixed_p_features_gpu(
    tokens: np.ndarray,
    cfg: DTCQRCConfig,
    *,
    seed: int,
    probability: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exact final Z/ZZ features for event-wise reset to |0...0>.

    The T pure suffix trajectories are evolved as a batched ensemble.  Their
    observable expectations are mixed with the analytic reset-channel weights;
    no density-matrix approximation and no micro-time-step splitting is used.
    """
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("the suffix-ensemble path requires 0 < p < 1")
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"exact QRC extraction requires visible CUDA, got {device}")
    torch.cuda.set_device(torch_device)
    set_seed(int(seed))
    params = generate_dtc_params(cfg.num_qubits, seed=int(seed), W_scale=cfg.W_scale)
    complex_dtype = torch_complex_dtype(cfg.complex_dtype)
    real_dtype = real_dtype_from_complex(complex_dtype)
    rx = rx_gate(cfg.g * math.pi, device=torch_device, complex_dtype=complex_dtype)
    phase = build_dtc_diagonal_phase(
        cfg.num_qubits,
        torch.from_numpy(params["J"]),
        torch.from_numpy(params["h"]),
        torch_device,
        complex_dtype,
    )
    signs = build_observable_signs(cfg.num_qubits, torch_device, real_dtype)
    mixture_weights = refresh_suffix_weights(T, p)
    pure_weights = torch.as_tensor(
        mixture_weights[:-1], device=torch_device, dtype=real_dtype
    )
    anchor_weight = float(mixture_weights[-1])
    output = np.empty((len(tokens), cfg.output_dim), dtype=np.float32)
    source = torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.float32))

    for batch_start in range(0, len(source), int(batch_size)):
        batch_end = min(batch_start + int(batch_size), len(source))
        x = source[batch_start:batch_end].to(
            device=torch_device, dtype=real_dtype, non_blocking=True
        )
        x = adapt_token_dim(x, cfg.input_dim, cfg.token_dim_policy)
        if cfg.use_tanh_before_encoding:
            x = torch.tanh(x)
        batch = len(x)
        if int(x.shape[1]) != T:
            raise RuntimeError(f"expected T={T}, got {tuple(x.shape)}")
        states = torch.zeros(
            (batch, T, cfg.state_dim), device=torch_device, dtype=complex_dtype
        )
        states[:, :, 0] = 1.0 + 0.0j
        for event in range(T):
            active_count = event + 1
            active = states[:, :active_count, :].reshape(
                batch * active_count, cfg.state_dim
            )
            token = (
                x[:, event, :]
                .unsqueeze(1)
                .expand(batch, active_count, cfg.input_dim)
                .reshape(batch * active_count, cfg.input_dim)
            )
            active = apply_token_injection(active, token, cfg, complex_dtype)
            active = apply_reservoir_layers(active, cfg, rx, phase)
            states[:, :active_count, :] = active.reshape(
                batch, active_count, cfg.state_dim
            )
        suffix_features = compute_z_zz_from_state(
            states.reshape(batch * T, cfg.state_dim), signs
        ).reshape(batch, T, cfg.output_dim)
        mixed = torch.einsum("btd,t->bd", suffix_features, pure_weights)
        mixed = mixed + anchor_weight
        output[batch_start:batch_end] = mixed.detach().cpu().numpy()
        del states, suffix_features, mixed, x
        print(
            f"[exact-p] seed={seed} g={cfg.g:.1f} W={cfg.W_scale:.1f} "
            f"samples={batch_end}/{len(source)}",
            flush=True,
        )
    return output, params


def _legacy_closed_feature(row: dict | pd.Series) -> np.ndarray | None:
    """Reuse exact closed caches only where the old one-factor plan overlaps."""
    if float(row["p"]) != 0.0 or float(row["W"]) != 0.5:
        return None
    try:
        from run_bace_general_g_w_scan import (
            parameter_pairs as legacy_pairs,
            load_feature as load_legacy,
        )

        if (float(row["g"]), float(row["W"])) not in legacy_pairs():
            return None
        return load_legacy(row).copy()
    except (FileNotFoundError, RuntimeError, ValueError):
        return None


@torch.inference_mode()
def compute_feature(
    row: dict | pd.Series,
    *,
    device: str,
    force: bool = False,
) -> Path:
    path = feature_path(row)
    if path.exists() and not force:
        try:
            load_feature(row)
            print(f"[cached] {path}", flush=True)
            return path
        except Exception:
            pass
    tokens, labels, token_sha, dataset_fingerprint = _load_inputs()
    cfg = DTCQRCConfig(
        num_qubits=Q,
        features_per_qubit=3,
        g=float(row["g"]),
        W_scale=float(row["W"]),
        layers_per_injection=LAYERS,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    started = time.time()
    reused = _legacy_closed_feature(row) if not force else None
    if reused is not None:
        features = reused
        params = generate_dtc_params(Q, int(row["seed"]), float(row["W"]))
        method = "reused_closed_statevector_cache"
    elif float(row["p"]) == 0.0:
        features_t, params = compute_qrc_features_gpu(
            torch.from_numpy(tokens),
            cfg,
            seed=int(row["seed"]),
            batch_size=BATCH_SIZE,
            device=device,
            progress=False,
        )
        features = features_t.numpy().astype(np.float32, copy=False)
        method = "closed_statevector_p0"
    else:
        features, params = compute_fixed_p_features_gpu(
            tokens,
            cfg,
            seed=int(row["seed"]),
            probability=float(row["p"]),
            batch_size=BATCH_SIZE,
            device=device,
        )
        method = "exact_eventwise_reset_suffix_ensemble"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        features=np.asarray(features, dtype=np.float32),
        labels=labels,
        scan_protocol=np.asarray(SCAN_PROTOCOL),
        base_protocol=np.asarray(PROTOCOL_VERSION),
        dataset=np.asarray(DATASET),
        token_variant=np.asarray(VARIANT),
        token_sha256=np.asarray(token_sha),
        dataset_fingerprint=np.asarray(dataset_fingerprint),
        q=np.int16(Q),
        t=np.int16(T),
        g=np.float32(row["g"]),
        W=np.float32(row["W"]),
        p=np.float32(row["p"]),
        seed=np.int32(row["seed"]),
        layers=np.int16(LAYERS),
        observables=np.asarray(OBSERVABLES),
        p_selection_source=np.asarray("completed affine validation at g=0.84, W=0.5"),
        feature_method=np.asarray(method),
        elapsed_seconds=np.float64(time.time() - started),
        parameter_sha256=np.asarray(parameter_sha256(params)),
        J=params["J"],
        h=params["h"],
    )
    partial.replace(path)
    load_feature(row)
    print(
        f"[feature] g={row['g']:.1f} W={row['W']:.1f} p={row['p']:.3f} "
        f"seed={int(row['seed'])} method={method} shape={features.shape}",
        flush=True,
    )
    return path


def evaluate_case(row: dict | pd.Series, *, force: bool = False) -> dict:
    destination = result_dir(row)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not force:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("scan_protocol") == SCAN_PROTOCOL:
            return payload
    features = load_feature(row)
    _, labels, token_sha, _ = _load_inputs()
    split = json.loads(split_path(DATASET, int(row["seed"])).read_text(encoding="utf-8"))
    metrics = fit_svr(
        features,
        labels,
        split,
        seed=int(row["seed"]),
        destination=destination,
        decoder_protocol=SCAN_PROTOCOL,
    )
    payload = {
        "scan_protocol": SCAN_PROTOCOL,
        "base_protocol": PROTOCOL_VERSION,
        "dataset": DATASET,
        "token_variant": VARIANT,
        "q": Q,
        "t": T,
        "g": float(row["g"]),
        "W": float(row["W"]),
        "p": float(row["p"]),
        "seed": int(row["seed"]),
        "reservoir": "event-wise reset-channel DTC-QRC",
        "observables": OBSERVABLES,
        "token_sha256": token_sha,
        "q_t_selection": (
            "Q12/T30 frozen from the completed six-seed validation-mean SVR curve"
        ),
        "p_selection": (
            "per-seed p frozen from the completed affine validation scan at "
            "g=0.84 and W=0.5"
        ),
        "g_W_selection": "none; every prespecified test-grid point is retained",
        **metrics,
    }
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def collect(strict: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    missing: list[str] = []
    for row in case_rows("full").to_dict("records"):
        path = result_dir(row) / "metrics.json"
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scan_protocol") != SCAN_PROTOCOL:
            raise RuntimeError(f"stale result {path}")
        rows.append(payload)
    if strict and missing:
        raise FileNotFoundError(f"missing {len(missing)} results; first={missing[0]}")
    return pd.DataFrame(rows)


def summarize(strict: bool = True) -> dict:
    frame = collect(strict=strict).sort_values(["g", "W", "seed"])
    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SCAN_ROOT / "metrics_all.csv", index=False)
    summary = (
        frame.groupby(["g", "W"], as_index=False)
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_brier_mean=("test_brier", "mean"),
            test_brier_std=("test_brier", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            val_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["g", "W"])
    )
    if strict:
        if len(summary) != 100 or not (summary.n_seeds == len(SEEDS)).all():
            raise RuntimeError("incomplete six-seed 10-by-10 summary")
    summary.to_csv(SCAN_ROOT / "test_grid_summary.csv", index=False)
    frame[
        [
            "g", "W", "seed", "p", "test_auc", "test_acc", "test_brier",
            "test_rmse", "val_auc", "selected_C", "selected_gamma_scale",
        ]
    ].to_csv(SCAN_ROOT / "test_grid_by_seed.csv", index=False)
    for metric in (
        "test_auc_mean",
        "test_auc_std",
        "test_acc_mean",
        "test_brier_mean",
    ):
        matrix = summary.pivot(index="g", columns="W", values=metric).reindex(
            index=G_VALUES, columns=W_VALUES
        )
        matrix.to_csv(SCAN_ROOT / f"{metric}_matrix.csv")
    status = {
        "scan_protocol": SCAN_PROTOCOL,
        "dataset": DATASET,
        "token_variant": VARIANT,
        "q": Q,
        "t": T,
        "g_values": list(G_VALUES),
        "W_values": list(W_VALUES),
        "grid_step": GRID_STEP,
        "grid_points": len(parameter_pairs()),
        "seeds": list(SEEDS),
        "fixed_validated_p_by_seed": validated_p_by_seed(),
        "result_rows": len(frame),
        "g_W_validation_selection": False,
        "test_point_selection": False,
        "reporting": "all prespecified test-grid points",
    }
    (SCAN_ROOT / "summary_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    return status


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    text_format: str,
) -> mpl.image.AxesImage:
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    for row, col in np.ndindex(matrix.shape):
        value = float(matrix[row, col])
        red, green, blue, _ = cmap(norm(value))
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        ax.text(
            col,
            row,
            text_format.format(value),
            ha="center",
            va="center",
            fontsize=5.2,
            color="white" if luminance < 0.48 else "#272727",
        )
    ax.set_xticks(range(len(W_VALUES)), [f"{value:.1f}" for value in W_VALUES])
    ax.set_yticks(range(len(G_VALUES)), [f"{value:.1f}" for value in G_VALUES])
    ax.set_xlabel(r"Disorder strength $W$")
    ax.set_ylabel(r"Floquet pulse parameter $g$")
    ax.set_xticks(np.arange(-0.5, len(W_VALUES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(G_VALUES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_heatmaps() -> dict:
    source = pd.read_csv(SCAN_ROOT / "test_grid_summary.csv")
    if len(source) != 100 or not (source.n_seeds == len(SEEDS)).all():
        raise RuntimeError("plotting requires the complete 10-by-10 six-seed summary")
    mean = source.pivot(index="g", columns="W", values="test_auc_mean").reindex(
        index=G_VALUES, columns=W_VALUES
    ).to_numpy()
    spread = source.pivot(index="g", columns="W", values="test_auc_std").reindex(
        index=G_VALUES, columns=W_VALUES
    ).to_numpy()
    if mean.shape != (10, 10) or spread.shape != (10, 10):
        raise RuntimeError("invalid heatmap matrix shape")
    if not np.isfinite(mean).all() or not np.isfinite(spread).all():
        raise RuntimeError("heatmap contains missing or non-finite values")

    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    auc_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "qrc_auc", ["#F4F6F8", "#9FC4DF", "#0F4D92"]
    )
    std_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "qrc_std", ["#F7F7F7", "#B9B9C8", "#484878"]
    )
    auc_padding = max(0.002, 0.03 * float(np.ptp(mean)))
    mean_norm = mpl.colors.Normalize(
        vmin=float(mean.min() - auc_padding), vmax=float(mean.max() + auc_padding)
    )
    spread_norm = mpl.colors.Normalize(vmin=0.0, vmax=float(spread.max()))
    width_in = 183.0 / 25.4
    height_in = 82.0 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width_in, height_in),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.46},
    )
    mean_image = _annotated_heatmap(
        axes[0], mean, cmap=auc_cmap, norm=mean_norm, text_format="{:.3f}"
    )
    spread_image = _annotated_heatmap(
        axes[1], spread, cmap=std_cmap, norm=spread_norm, text_format="{:.3f}"
    )
    axes[0].set_title("Mean test ROC–AUC", pad=5)
    axes[1].set_title("Across-seed s.d.", pad=5)
    axes[0].text(
        -0.16, 1.04, "a", transform=axes[0].transAxes,
        fontsize=8, fontweight="bold", va="bottom",
    )
    axes[1].text(
        -0.16, 1.04, "b", transform=axes[1].transAxes,
        fontsize=8, fontweight="bold", va="bottom",
    )
    cbar_mean = fig.colorbar(mean_image, ax=axes[0], fraction=0.047, pad=0.025)
    cbar_mean.set_label("Test ROC–AUC")
    cbar_spread = fig.colorbar(spread_image, ax=axes[1], fraction=0.047, pad=0.025)
    cbar_spread.set_label("s.d. across 6 seeds")
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    base = FIGURE_ROOT / "bace_general_fixed_p_g_w_test_heatmaps"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    contract = {
        "core_conclusion": (
            "At fixed Q12/T30 and validation-frozen p, the complete test map "
            "quantifies joint g-W sensitivity without selecting a g/W operating point."
        ),
        "archetype": "quantitative grid",
        "role": "robustness and parameter sensitivity",
        "backend": "Python/matplotlib only",
        "final_size_mm": [183, 82],
        "panel_map": {
            "a": "six-seed mean test ROC-AUC at every prespecified grid point",
            "b": "across-seed standard deviation at the same grid points",
        },
        "statistics": {
            "n": "6 paired seeds",
            "center": "arithmetic mean",
            "spread": "sample standard deviation (pandas ddof=1)",
            "split": "unchanged train/validation/test split for each seed",
            "metric": "ROC-AUC on continuous SVR scores",
        },
        "reviewer_risk": (
            "The map is descriptive. No maximum test cell is marked or promoted, "
            "and no g/W validation selection is performed."
        ),
        "source_data": "test_grid_summary.csv and test_grid_by_seed.csv",
    }
    (FIGURE_ROOT / "figure_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
    )
    svg_text = base.with_suffix(".svg").read_text(encoding="utf-8")
    expected_exports = [base.with_suffix(ext) for ext in (".svg", ".pdf", ".tiff", ".png")]
    qa = {
        "python_only": True,
        "matrix_shape": list(mean.shape),
        "all_finite": bool(np.isfinite(mean).all() and np.isfinite(spread).all()),
        "editable_svg_text": "<text" in svg_text,
        "exports": {str(path.name): path.stat().st_size for path in expected_exports},
        "no_g_W_selection_marker": True,
        "passes": bool(
            np.isfinite(mean).all()
            and np.isfinite(spread).all()
            and "<text" in svg_text
            and all(path.exists() and path.stat().st_size > 0 for path in expected_exports)
        ),
    }
    (FIGURE_ROOT / "figure_qa.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not qa["passes"]:
        raise RuntimeError(f"figure QA failed: {qa}")
    return qa


def audit(strict: bool = True) -> dict:
    errors: list[str] = []
    valid_features = 0
    valid_results = 0
    for row in case_rows("full").to_dict("records"):
        try:
            load_feature(row)
            valid_features += 1
        except Exception as error:
            errors.append(f"feature {row}: {error}")
        path = result_dir(row) / "metrics.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = [
                payload.get(key)
                for key in ("val_auc", "test_auc", "test_acc", "test_brier")
            ]
            if (
                payload.get("scan_protocol") == SCAN_PROTOCOL
                and np.isfinite(values).all()
                and payload.get("g_W_selection", "").startswith("none")
            ):
                valid_results += 1
            else:
                errors.append(f"invalid result {path}")
        else:
            errors.append(f"missing result {path}")
    expected = len(parameter_pairs()) * len(SEEDS)
    report = {
        "scan_protocol": SCAN_PROTOCOL,
        "expected": expected,
        "valid_features": valid_features,
        "valid_results": valid_results,
        "errors": errors,
    }
    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    (SCAN_ROOT / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if strict and errors:
        raise AssertionError(errors[0])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("cases", "features", "evaluate", "all", "summarize", "plot", "audit"),
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    cases = case_rows(args.mode).to_dict("records")
    if args.stage == "cases":
        result = {
            "cases": len(cases),
            "grid_points": len({(row["g"], row["W"]) for row in cases}),
            "fixed_validated_p_by_seed": validated_p_by_seed(),
        }
    elif args.stage in ("features", "evaluate", "all"):
        if args.stage in ("features", "all"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA unavailable; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
                )
            for index, row in enumerate(cases, start=1):
                compute_feature(row, device=args.device, force=args.force)
                print(f"feature case {index}/{len(cases)}", flush=True)
        if args.stage in ("evaluate", "all"):
            for index, row in enumerate(cases, start=1):
                evaluate_case(row, force=args.force)
                print(f"SVR case {index}/{len(cases)}", flush=True)
        result = {"completed_cases": len(cases)}
    elif args.stage == "summarize":
        result = summarize(strict=args.strict)
    elif args.stage == "plot":
        result = plot_heatmaps()
    else:
        result = audit(strict=args.strict)
    print(json.dumps({"stage": args.stage, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
