"""Publication-oriented validity controls for molecular BFS event-stream QRC.

This module is intentionally an extension, not a replacement, of
``FINAL/2_linear_readout``.  It imports the same general-token schema, QRC
statevector, scalar CRC, scaffold split, and 500-epoch one-layer probe.

Four independent experiments are provided:

1. isomorphic indexing/traversal robustness;
2. typed-padding, zero-identity-padding, and true length masking;
3. label-conflict diagnostics and a representation-oracle ceiling;
4. original-trained frozen probes evaluated across controlled order changes.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


MODULE_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = MODULE_ROOT.parent
BASELINE_ROOT = FINAL_ROOT / "2_linear_readout"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from core.general_tokens import MAX_Q, SCHEMA_VERSION  # noqa: E402
from core.crc3_baseline import (  # noqa: E402
    CRC3_Z_ZZ_METHOD,
    CRC3_Z_ZZ_PROTOCOL,
    crc3_z_zz_dim,
)
from core.linear_readout import (  # noqa: E402
    DEFAULT_SEEDS,
    TOKEN_VARIANT_GENERAL,
    ExperimentConfig,
    load_or_build_token_cache,
)

from .feature_engine import (
    append_ensemble_variants,
    compute_crc3_z_zz_padding,
    compute_crc3_z_zz_tensor,
    compute_crc_padding,
    compute_crc_tensor,
    evaluate_ensemble_augmented_head,
    evaluate_frozen_head,
    extract_qrc_dual_gpu,
    extract_qrc_order_dual_gpu,
    json_dump,
    load_cache_extended,
    representation_robustness_rows,
    summarize_frozen_metrics,
)
from .token_variants import (
    ORDER_VARIANTS,
    build_isomorphic_token_variants,
    build_order_tensor,
)
from .canonical_tokens import audit_canonical_general_invariance


EXPERIMENTS = ("invariance", "padding", "collision", "cross_order")
PERMUTATION_SEEDS = tuple(range(1000, 1100, 10))


@dataclass(frozen=True)
class ValidityConfig:
    dataset: str
    data_dir: str
    output_dir: str
    experiment: str
    q: int = 16
    t: int = 16
    profile: str = "full"
    epochs: int = 500
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    permutation_seeds: tuple[int, ...] = PERMUTATION_SEEDS
    W: float = 0.6
    g: float = 0.84
    layers: int = 3
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    learning_rate: float = 1e-3
    batch_size: int = 32
    qrc_devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    qrc_batch_size: int | None = None
    crc_device: str = "auto"
    train_device: str = "auto"
    crc_state_scale: float = 0.85
    crc_inject_scale: float = 0.75
    relabel_repeats: int = 2
    random_smiles_repeats: int = 2
    root_repeats: int = 4
    variant_seed: int = 20260710
    block_size: int = 2
    q_values: tuple[int, ...] = tuple(range(1, 21))
    t_values: tuple[int, ...] = tuple(range(1, 31))
    run_qrc: bool = True
    force: bool = False

    def normalized(self) -> "ValidityConfig":
        dataset = self.dataset.upper()
        experiment = self.experiment.lower()
        profile = self.profile.lower()
        if dataset not in {"BACE", "BBBP"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if experiment not in EXPERIMENTS:
            raise ValueError(f"experiment must be one of {EXPERIMENTS}")
        if profile not in {"full", "quick"}:
            raise ValueError("profile must be full or quick")
        q, t = int(self.q), int(self.t)
        epochs = int(self.epochs)
        seeds = tuple(int(seed) for seed in self.seeds)
        permutation_seeds = tuple(int(seed) for seed in self.permutation_seeds)
        q_values = tuple(sorted({int(value) for value in self.q_values}))
        t_values = tuple(sorted({int(value) for value in self.t_values}))
        devices = tuple(str(value) for value in self.qrc_devices)
        if profile == "quick":
            q, t = min(q, 4), min(t, 6)
            epochs = min(epochs, 20)
            seeds = (seeds[0],)
            permutation_seeds = (permutation_seeds[0],)
            q_values = tuple(value for value in (2, 4) if value <= q)
            t_values = tuple(value for value in (2, 4, 6) if value <= t)
            devices = devices[:1]
        else:
            if seeds != DEFAULT_SEEDS:
                raise ValueError(f"Full protocol fixes seeds={DEFAULT_SEEDS}")
            if epochs != 500:
                raise ValueError("Full protocol fixes epochs=500")
            if not math.isclose(float(self.W), 0.6):
                raise ValueError("Canonical validity protocol fixes W=0.6")
        if len(permutation_seeds) != len(seeds):
            raise ValueError("Reservoir/split and permutation seed lists must have equal length")
        if set(permutation_seeds) & set(seeds):
            raise ValueError("Permutation seeds must be disjoint from reservoir/split seeds")
        if not (1 <= q <= min(MAX_Q, 20)) or not (1 <= t <= 30):
            raise ValueError("This FINAL module supports Q=1..20 and T=1..30")
        if not q_values or min(q_values) < 1 or max(q_values) > 20:
            raise ValueError("q_values must lie in 1..20")
        if not t_values or min(t_values) < 1 or max(t_values) > 30:
            raise ValueError("t_values must lie in 1..30")
        return ValidityConfig(
            **{
                **asdict(self),
                "dataset": dataset,
                "data_dir": str(Path(self.data_dir).resolve()),
                "output_dir": str(Path(self.output_dir).resolve()),
                "experiment": experiment,
                "profile": profile,
                "q": q,
                "t": t,
                "epochs": epochs,
                "seeds": seeds,
                "permutation_seeds": permutation_seeds,
                "q_values": q_values,
                "t_values": t_values,
                "qrc_devices": devices,
            }
        )

    def linear_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            dataset=self.dataset,
            data_dir=self.data_dir,
            scan_root=self.output_dir,
            token_variant=TOKEN_VARIANT_GENERAL,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            seeds=self.seeds,
            g=self.g,
            W=self.W,
            layers=self.layers,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            crc_state_scale=self.crc_state_scale,
            crc_inject_scale=self.crc_inject_scale,
            max_cached_events=max(30, self.t),
            qrc_devices=self.qrc_devices,
            qrc_batch_size=self.qrc_batch_size,
            crc_device=self.crc_device,
            train_device=self.train_device,
            run_qrc=self.run_qrc,
            force=self.force,
        )


def _prepare(cfg: ValidityConfig) -> tuple[Path, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path, metadata = load_or_build_token_cache(
        cfg.dataset,
        cfg.data_dir,
        MODULE_ROOT / "cache" / cfg.dataset,
        max_events=30,
        token_variant=TOKEN_VARIANT_GENERAL,
    )
    tokens, labels, smiles, counts = load_cache_extended(cache_path)
    return cache_path, metadata, tokens, labels, smiles, counts


def _write_contract(cfg: ValidityConfig, extra: dict) -> None:
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
            **asdict(cfg),
            "token_variant": TOKEN_VARIANT_GENERAL,
            "token_schema_version": SCHEMA_VERSION,
            "classifier": "single torch.nn.Linear(d,1)",
            "selection": "max(val_auc) > min(val_loss) > max(val_acc)",
            "standardization": False,
            "regularization": False,
            "seed_contract": (
                "split seed = reservoir seed for paired QRC/CRC3/scalar-CRC comparison; permutation_seed is stored "
                "separately and is disjoint from both"
            ),
            **extra,
        }
    if cfg.experiment != "collision":
        payload[CRC3_Z_ZZ_METHOD] = {
            "type": "Qx3_vector_state_with_qubit_and_axis_mixing",
            "input": "same_tanh_3Q_event_coordinates",
            "readout": "direct_z+all_zz_no_pca",
            "feature_dim": crc3_z_zz_dim(cfg.q),
            "protocol_version": CRC3_Z_ZZ_PROTOCOL,
        }
    json_dump(output / "experiment_config.json", payload)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_frozen(summary: pd.DataFrame, output: Path, title: str) -> None:
    if summary.empty:
        return
    _style()
    variants = list(dict.fromkeys(summary.test_variant.astype(str)))
    methods = [
        method
        for method in ("qrc", CRC3_Z_ZZ_METHOD, "crc", "raw_full")
        if method in set(summary.method)
    ]
    colors = {
        "qrc": "#2F6B9A",
        CRC3_Z_ZZ_METHOD: "#2D8C73",
        "crc": "#D07A35",
        "raw_full": "#6F6F6F",
    }
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(180 / 25.4, 75 / 25.4), constrained_layout=True)
    for method in methods:
        table = summary[summary.method == method].set_index("test_variant")
        means = np.asarray([table.loc[name, "test_auc_mean"] for name in variants])
        stds = np.asarray([table.loc[name, "test_auc_std"] for name in variants])
        label = "CRC3-Z/ZZ" if method == CRC3_Z_ZZ_METHOD else method.upper()
        ax.plot(x, means, marker="o", ms=3, lw=1.1, color=colors[method], label=label)
        ax.fill_between(x, means - stds, means + stds, color=colors[method], alpha=0.13)
    ax.axhline(0.5, color="#AAAAAA", ls="--", lw=0.7)
    ax.set_xticks(x, variants, rotation=28, ha="right")
    ax.set_ylabel("Frozen-head test ROC-AUC")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(ncol=len(methods), loc="best")
    _save(fig, output / "figures" / "frozen_cross_variant_auc")


def _crc_path(output: Path, seed: int) -> Path:
    return output / "features" / f"crc_seed{seed:03d}.npz"


def _crc3_path(output: Path, seed: int) -> Path:
    return output / "features" / f"{CRC3_Z_ZZ_METHOD}_seed{seed:03d}.npz"


def _load_valid_crc3_cache(
    path: Path,
    *,
    shape: tuple[int, ...],
    names: Sequence[str],
    q: int,
    t: int,
    seed: int,
    permutation_seed: int | None = None,
) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as package:
            valid = (
                package["features"].shape == shape
                and np.array_equal(package["variant_names"].astype(str), np.asarray(names).astype(str))
                and int(package["q"]) == int(q)
                and int(package["t"]) == int(t)
                and int(package["seed"]) == int(seed)
                and str(package["method"].item()) == CRC3_Z_ZZ_METHOD
                and str(package["baseline_protocol_version"].item()) == CRC3_Z_ZZ_PROTOCOL
                and str(package["readout"].item()) == "z+all_zz_no_pca"
            )
            if permutation_seed is not None:
                valid = valid and int(package["permutation_seed"]) == int(permutation_seed)
            if valid:
                return package["features"].astype(np.float32)
    except Exception:
        pass
    return None


def _completed_with_crc3(
    output: Path,
    cfg: ValidityConfig,
    *,
    require_ensemble: bool = False,
) -> bool:
    summary_path = output / "frozen_cross_variant_summary.csv"
    if not summary_path.exists():
        return False
    try:
        summary = pd.read_csv(summary_path)
        required = {"crc", "raw_full", CRC3_Z_ZZ_METHOD}
        if cfg.run_qrc:
            required.add("qrc")
        if not required.issubset(set(summary.method.astype(str))):
            return False
        for method in required:
            subset = summary[summary.method == method]
            if subset.empty or int(subset.n_seeds.min()) < len(cfg.seeds):
                return False
        if require_ensemble:
            ensemble = pd.read_csv(output / "ensemble_augmented_summary.csv")
            if not required.issubset(set(ensemble.base_method.astype(str))):
                return False
            for method in required:
                subset = ensemble[ensemble.base_method == method]
                if subset.empty or int(subset.n_seeds.min()) < len(cfg.seeds):
                    return False
        return True
    except Exception:
        return False


def run_invariance(config: ValidityConfig) -> pd.DataFrame:
    cfg = config.normalized()
    output = Path(cfg.output_dir)
    completed = output / "frozen_cross_variant_summary.csv"
    if not cfg.force and _completed_with_crc3(output, cfg, require_ensemble=True):
        return pd.read_csv(completed)
    cache_path, metadata, base, labels, smiles, counts = _prepare(cfg)
    tensor_path = output / "inputs" / "isomorphic_token_variants.npz"
    if cfg.force or not tensor_path.exists():
        variants, variant_counts, names, audit = build_isomorphic_token_variants(
            base[:, : cfg.t, : 3 * cfg.q],
            counts,
            smiles,
            cfg.dataset,
            cfg.q,
            cfg.t,
            relabel_repeats=cfg.relabel_repeats,
            random_smiles_repeats=cfg.random_smiles_repeats,
            root_repeats=cfg.root_repeats,
            variant_seed=cfg.variant_seed,
        )
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tensor_path,
            tokens=variants,
            event_counts=variant_counts,
            variant_names=np.asarray(names),
            labels=labels,
            smiles=smiles,
        )
        pd.DataFrame(audit).to_csv(output / "isomorphism_identity_audit.csv", index=False)
    with np.load(tensor_path, allow_pickle=False) as package:
        variants = package["tokens"].astype(np.float32)
        names = package["variant_names"].astype(str).tolist()
    canonical_audit = audit_canonical_general_invariance(
        smiles,
        cfg.dataset,
        q=cfg.q,
        t=cfg.t,
        repeats=max(cfg.relabel_repeats, cfg.random_smiles_repeats),
        seed=cfg.variant_seed,
    )
    canonical_audit.to_csv(output / "canonical_v3_invariance_audit.csv", index=False)
    canonical_audit.groupby("variant", as_index=False).agg(
        n=("sample_index", "size"),
        event_count_equal_rate=("event_count_equal", "mean"),
        token_exact_equal_rate=("token_exact_equal", "mean"),
    ).to_csv(output / "canonical_v3_invariance_summary.csv", index=False)
    _write_contract(
        cfg,
        {
            "hypothesis": "A graph method should not exploit arbitrary atom indices, SMILES enumeration, or a single BFS root.",
            "reference_cache": str(cache_path),
            "isomorphism_check": "canonical isomeric SMILES equality before token acceptance",
            "variant_names": names,
            "ensemble_rule": "average representations after QRC/CRC3/scalar-CRC/raw mapping; never average labels",
            "invariance_conditions": {
                "static_canonical_v3": "invariant-by-canonicalization; complete-graph ordering; reproducibility control only",
                "traversal_ensemble": "invariant-in-expectation; traversal augmentation on train and post-map aggregation on validation/test",
                "dynamic_graph": "use observed event timestamps; neither canonical graph ordering nor random traversal replaces time",
            },
        },
    )
    if cfg.run_qrc:
        extract_qrc_dual_gpu(
            tensor_path,
            output / "features",
            q=cfg.q,
            t=cfg.t,
            seeds=cfg.seeds,
            devices=cfg.qrc_devices,
            g=cfg.g,
            W=cfg.W,
            layers=cfg.layers,
            force=cfg.force,
            batch_size=cfg.qrc_batch_size,
        )

    token_with_ensembles, ensemble_names = append_ensemble_variants(variants, names)
    robustness = representation_robustness_rows(
        token_with_ensembles,
        ensemble_names,
        method="tokens",
        seed=-1,
        reference="original",
    )
    all_metrics = []
    augmented_metrics = []
    raw = token_with_ensembles.reshape(len(ensemble_names), len(labels), -1)
    for seed in cfg.seeds:
        crc_path = _crc_path(output, seed)
        if crc_path.exists() and not cfg.force:
            with np.load(crc_path, allow_pickle=False) as package:
                crc = package["features"].astype(np.float32)
                crc_names = package["variant_names"].astype(str).tolist()
        else:
            crc_base = compute_crc_tensor(
                variants,
                q=cfg.q,
                layers=cfg.layers,
                seed=seed,
                device=cfg.crc_device,
                state_scale=cfg.crc_state_scale,
                inject_scale=cfg.crc_inject_scale,
            )
            crc, crc_names = append_ensemble_variants(crc_base, names)
            crc_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(crc_path, features=crc, variant_names=np.asarray(crc_names), seed=seed)
        crc3_path = _crc3_path(output, seed)
        crc3 = None if cfg.force else _load_valid_crc3_cache(
            crc3_path,
            shape=(len(ensemble_names), len(labels), crc3_z_zz_dim(cfg.q)),
            names=ensemble_names,
            q=cfg.q,
            t=cfg.t,
            seed=seed,
        )
        if crc3 is None:
            crc3_base = compute_crc3_z_zz_tensor(
                variants,
                q=cfg.q,
                layers=cfg.layers,
                seed=seed,
                device=cfg.crc_device,
            )
            crc3, crc3_names = append_ensemble_variants(crc3_base, names)
            if crc3_names != ensemble_names:
                raise RuntimeError("CRC3 and token variant order diverged")
            crc3_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                crc3_path,
                features=crc3,
                variant_names=np.asarray(crc3_names),
                q=np.int16(cfg.q),
                t=np.int16(cfg.t),
                seed=np.int32(seed),
                layers=np.int16(cfg.layers),
                method=np.asarray(CRC3_Z_ZZ_METHOD),
                baseline_protocol_version=np.asarray(CRC3_Z_ZZ_PROTOCOL),
                preprocessing=np.asarray("tanh"),
                readout=np.asarray("z+all_zz_no_pca"),
            )
        features_by_method = {
            CRC3_Z_ZZ_METHOD: crc3,
            "crc": crc,
            "raw_full": raw,
        }
        robustness.extend(
            representation_robustness_rows(
                crc3, ensemble_names, method=CRC3_Z_ZZ_METHOD, seed=seed, reference="original"
            )
        )
        robustness.extend(
            representation_robustness_rows(crc, crc_names, method="crc", seed=seed, reference="original")
        )
        robustness.extend(
            representation_robustness_rows(raw, ensemble_names, method="raw_full", seed=seed, reference="original")
        )
        if cfg.run_qrc:
            with np.load(output / "features" / f"qrc_seed{seed:03d}.npz", allow_pickle=False) as package:
                qrc_base = package["features"].astype(np.float32)
                qrc_base_names = package["variant_names"].astype(str).tolist()
            qrc, qrc_names = append_ensemble_variants(qrc_base, qrc_base_names)
            if qrc_names != ensemble_names:
                raise RuntimeError("QRC and token variant order diverged")
            features_by_method = {"qrc": qrc, **features_by_method}
            robustness.extend(
                representation_robustness_rows(qrc, qrc_names, method="qrc", seed=seed, reference="original")
            )
        metrics = evaluate_frozen_head(
            features_by_method,
            ensemble_names,
            labels,
            smiles,
            seed=seed,
            output_dir=output,
            linear_cfg=cfg.linear_config(),
            q=cfg.q,
            t=cfg.t,
            train_variant="original",
            experiment="isomorphic_traversal",
        )
        all_metrics.append(metrics)
        augmented_metrics.append(
            evaluate_ensemble_augmented_head(
                features_by_method,
                ensemble_names,
                labels,
                smiles,
                seed=seed,
                output_dir=output,
                linear_cfg=cfg.linear_config(),
                q=cfg.q,
                t=cfg.t,
            )
        )
    pd.DataFrame(robustness).to_csv(output / "representation_robustness.csv", index=False)
    augmented = pd.concat(augmented_metrics, ignore_index=True)
    augmented.to_csv(output / "ensemble_augmented_metrics_by_seed.csv", index=False)
    augmented.groupby("base_method", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        test_auc_mean=("test_auc", "mean"),
        test_auc_std=("test_auc", "std"),
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
    ).to_csv(output / "ensemble_augmented_summary.csv", index=False)
    summary = summarize_frozen_metrics(pd.concat(all_metrics, ignore_index=True), output)
    _plot_frozen(summary, output, f"{cfg.dataset}: isomorphic traversal transfer, Q={cfg.q}, T={cfg.t}")
    json_dump(
        output / "figures" / "figure_contract.json",
        {
            "core_conclusion": "Quantify whether arbitrary graph serialization changes tokens and frozen QRC/CRC3/scalar-CRC/raw decisions.",
            "evidence": "Representation cosine/CKA plus original-trained frozen-head AUC over atom relabeling, randomized SMILES, and multi-root ensembles.",
            "source_data": ["representation_robustness.csv", "frozen_cross_variant_summary.csv"],
        },
    )
    return summary


def run_padding_ablation(config: ValidityConfig) -> pd.DataFrame:
    cfg = config.normalized()
    output = Path(cfg.output_dir)
    completed = output / "frozen_cross_variant_summary.csv"
    if not cfg.force and _completed_with_crc3(output, cfg):
        return pd.read_csv(completed)
    cache_path, _, base, labels, smiles, counts = _prepare(cfg)
    typed = np.ascontiguousarray(base[:, : cfg.t, : 3 * cfg.q], dtype=np.float32)
    zero = typed.copy()
    clipped_counts = np.clip(counts.astype(int), 0, cfg.t).astype(np.int16)
    for index, count in enumerate(clipped_counts):
        zero[index, int(count) :] = 0.0
    tensor_path = output / "inputs" / "padding_tokens.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tensor_path,
        typed_tokens=typed,
        zero_tokens=zero,
        event_counts=clipped_counts,
        labels=labels,
        smiles=smiles,
    )
    names = ["typed_pad", "zero_identity_pad", "length_masked"]
    _write_contract(
        cfg,
        {
            "hypothesis": "Apparent memory must not be caused by the number or content of padding steps.",
            "reference_cache": str(cache_path),
            "padding_modes": {
                "typed_pad": "current FINAL [PAD] token followed by normal reservoir evolution",
                "zero_identity_pad": "all-zero U3 injection, while reservoir evolution continues",
                "length_masked": "no injection and no reservoir layer after the last valid event",
            },
        },
    )
    if cfg.run_qrc:
        extract_qrc_dual_gpu(
            tensor_path,
            output / "features",
            q=cfg.q,
            t=cfg.t,
            seeds=cfg.seeds,
            devices=cfg.qrc_devices,
            g=cfg.g,
            W=cfg.W,
            layers=cfg.layers,
            force=cfg.force,
            padding_mode=True,
            batch_size=cfg.qrc_batch_size,
        )
    token_tensor = np.stack([typed, zero, zero])
    raw = token_tensor.reshape(3, len(labels), -1)
    robustness = representation_robustness_rows(
        token_tensor, names, method="tokens", seed=-1, reference="typed_pad"
    )
    all_metrics = []
    for seed in cfg.seeds:
        crc_path = _crc_path(output, seed)
        if crc_path.exists() and not cfg.force:
            with np.load(crc_path, allow_pickle=False) as package:
                crc = package["features"].astype(np.float32)
        else:
            crc = compute_crc_padding(
                typed,
                zero,
                clipped_counts,
                q=cfg.q,
                t=cfg.t,
                layers=cfg.layers,
                seed=seed,
                device=cfg.crc_device,
                state_scale=cfg.crc_state_scale,
                inject_scale=cfg.crc_inject_scale,
            )
            crc_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(crc_path, features=crc, variant_names=np.asarray(names), seed=seed)
        crc3_path = _crc3_path(output, seed)
        crc3 = None if cfg.force else _load_valid_crc3_cache(
            crc3_path,
            shape=(len(names), len(labels), crc3_z_zz_dim(cfg.q)),
            names=names,
            q=cfg.q,
            t=cfg.t,
            seed=seed,
        )
        if crc3 is None:
            crc3 = compute_crc3_z_zz_padding(
                typed,
                zero,
                clipped_counts,
                q=cfg.q,
                t=cfg.t,
                layers=cfg.layers,
                seed=seed,
                device=cfg.crc_device,
            )
            crc3_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                crc3_path,
                features=crc3,
                variant_names=np.asarray(names),
                q=np.int16(cfg.q),
                t=np.int16(cfg.t),
                seed=np.int32(seed),
                layers=np.int16(cfg.layers),
                method=np.asarray(CRC3_Z_ZZ_METHOD),
                baseline_protocol_version=np.asarray(CRC3_Z_ZZ_PROTOCOL),
                preprocessing=np.asarray("tanh"),
                readout=np.asarray("z+all_zz_no_pca"),
            )
        features_by_method = {
            CRC3_Z_ZZ_METHOD: crc3,
            "crc": crc,
            "raw_full": raw,
        }
        robustness.extend(
            representation_robustness_rows(
                crc3, names, method=CRC3_Z_ZZ_METHOD, seed=seed, reference="typed_pad"
            )
        )
        robustness.extend(
            representation_robustness_rows(crc, names, method="crc", seed=seed, reference="typed_pad")
        )
        robustness.extend(
            representation_robustness_rows(raw, names, method="raw_full", seed=seed, reference="typed_pad")
        )
        if cfg.run_qrc:
            with np.load(output / "features" / f"qrc_seed{seed:03d}.npz", allow_pickle=False) as package:
                qrc = package["features"].astype(np.float32)
            features_by_method = {"qrc": qrc, **features_by_method}
            robustness.extend(
                representation_robustness_rows(qrc, names, method="qrc", seed=seed, reference="typed_pad")
            )
        all_metrics.append(
            evaluate_frozen_head(
                features_by_method,
                names,
                labels,
                smiles,
                seed=seed,
                output_dir=output,
                linear_cfg=cfg.linear_config(),
                q=cfg.q,
                t=cfg.t,
                train_variant="typed_pad",
                experiment="padding_ablation",
            )
        )
    pd.DataFrame(robustness).to_csv(output / "representation_robustness.csv", index=False)
    summary = summarize_frozen_metrics(pd.concat(all_metrics, ignore_index=True), output)
    _plot_frozen(summary, output, f"{cfg.dataset}: padding transfer, Q={cfg.q}, T={cfg.t}")
    return summary


def _digest_rows(array: np.ndarray) -> list[bytes]:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return [row.tobytes() for row in contiguous]


def _entropy_binary(positive: int, total: int) -> float:
    if positive in {0, total} or total == 0:
        return 0.0
    p = positive / total
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def _unique_canonical_rows(
    tokens: np.ndarray, labels: np.ndarray, smiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    groups: dict[str, list[int]] = {}
    for index, smi in enumerate(smiles.astype(str)):
        groups.setdefault(smi, []).append(index)
    selected, unique_labels, conflicts = [], [], []
    for smi, indices in sorted(groups.items()):
        y = labels[indices]
        selected.append(indices[0])
        unique_labels.append(int(np.mean(y) >= 0.5))
        conflicts.append(
            {
                "canonical_smiles": smi,
                "source_rows": len(indices),
                "source_positive": int(y.sum()),
                "source_label_conflict": bool(len(np.unique(y)) > 1),
            }
        )
    selected_array = np.asarray(selected, dtype=int)
    return (
        tokens[selected_array],
        np.asarray(unique_labels, dtype=np.int64),
        smiles[selected_array],
        pd.DataFrame(conflicts),
    )


def _collision_row(tokens: np.ndarray, labels: np.ndarray, q: int, t: int) -> tuple[dict, list[dict]]:
    groups: dict[bytes, list[int]] = {}
    flat = tokens[:, :t, : 3 * q].reshape(len(tokens), -1)
    for index, digest in enumerate(_digest_rows(flat)):
        groups.setdefault(digest, []).append(index)
    oracle_probability = np.zeros(len(labels), dtype=float)
    details = []
    collision_samples = conflict_samples = 0
    weighted_entropy = 0.0
    conflict_groups = 0
    for group_index, indices in enumerate(groups.values()):
        y = labels[indices]
        size, positives = len(indices), int(y.sum())
        probability = positives / size
        oracle_probability[indices] = probability
        entropy = _entropy_binary(positives, size)
        weighted_entropy += size * entropy
        if size > 1:
            collision_samples += size
        if 0 < positives < size:
            conflict_groups += 1
            conflict_samples += size
        details.append(
            {
                "group_index": group_index,
                "size": size,
                "positive": positives,
                "negative": size - positives,
                "label_entropy_bits": entropy,
                "label_conflict": bool(0 < positives < size),
            }
        )
    predictions = (oracle_probability >= 0.5).astype(int)
    row = {
        "q": int(q),
        "token_dim": int(3 * q),
        "t": int(t),
        "n_unique_molecules": int(len(tokens)),
        "unique_representations": int(len(groups)),
        "collision_groups": int(sum(len(indices) > 1 for indices in groups.values())),
        "collision_samples": int(collision_samples),
        "collision_rate": float(collision_samples / len(tokens)),
        "label_conflict_groups": int(conflict_groups),
        "label_conflict_samples": int(conflict_samples),
        "label_conflict_rate": float(conflict_samples / len(tokens)),
        "weighted_group_label_entropy_bits": float(weighted_entropy / len(tokens)),
        "representation_oracle_auc": float(roc_auc_score(labels, oracle_probability)),
        "representation_oracle_acc": float(accuracy_score(labels, predictions)),
        "oracle_definition": "full-dataset empirical positive fraction within each exact representation group",
    }
    return row, details


def run_collision_ceiling(config: ValidityConfig) -> pd.DataFrame:
    cfg = config.normalized()
    output = Path(cfg.output_dir)
    completed = output / "collision_label_conflict_Q_T.csv"
    if completed.exists() and not cfg.force:
        return pd.read_csv(completed)
    cache_path, metadata, tokens, labels, smiles, _ = _prepare(cfg)
    unique_tokens, unique_labels, unique_smiles, source_groups = _unique_canonical_rows(tokens, labels, smiles)
    output.mkdir(parents=True, exist_ok=True)
    source_groups.to_csv(output / "source_duplicate_label_audit.csv", index=False)
    rows = []
    selected_details = None
    for q in cfg.q_values:
        for t in cfg.t_values:
            row, details = _collision_row(unique_tokens, unique_labels, q, t)
            row.update({"dataset": cfg.dataset, "schema_version": SCHEMA_VERSION})
            rows.append(row)
            if q == cfg.q and t == cfg.t:
                selected_details = details
    table = pd.DataFrame(rows)
    table.to_csv(completed, index=False)
    if selected_details is not None:
        pd.DataFrame(selected_details).to_csv(
            output / f"collision_groups_Q{cfg.q:02d}_T{cfg.t:02d}.csv", index=False
        )
    _write_contract(
        cfg,
        {
            "reference_cache": str(cache_path),
            "source_metadata": metadata,
            "diagnostic_population": "unique canonical molecules; duplicate source rows audited separately",
            "label_use": "labels are used only after tokenization to diagnose conflict; never to construct or select tokens",
            "oracle_warning": "descriptive in-sample representation ceiling, not a predictive test estimate",
        },
    )
    _style()
    q_values = sorted(table.q.unique())
    t_values = sorted(table.t.unique())
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 58 / 25.4), constrained_layout=True)
    for ax, metric, title, limits in zip(
        axes,
        ("collision_rate", "label_conflict_rate", "representation_oracle_auc"),
        ("Exact collisions", "Label-conflicting collisions", "Representation oracle AUC"),
        ((0, 1), (0, 1), (0.5, 1)),
    ):
        pivot = table.pivot(index="q", columns="t", values=metric).loc[q_values, t_values]
        image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", vmin=limits[0], vmax=limits[1])
        ax.set_xticks(np.arange(len(t_values))[::3], [str(value) for value in t_values[::3]])
        ax.set_yticks(np.arange(len(q_values))[::2], [str(value) for value in q_values[::2]])
        ax.set_xlabel("Events, T")
        ax.set_ylabel("Qubits, Q")
        ax.set_title(title, loc="left", fontweight="bold")
        fig.colorbar(image, ax=ax, pad=0.02)
    _save(fig, output / "figures" / "collision_conflict_oracle_Q_T")
    return table


def run_cross_order(config: ValidityConfig) -> pd.DataFrame:
    cfg = config.normalized()
    output = Path(cfg.output_dir)
    completed = output / "frozen_cross_variant_summary.csv"
    if not cfg.force and _completed_with_crc3(output, cfg):
        return pd.read_csv(completed)
    cache_path, _, base, labels, smiles, counts = _prepare(cfg)
    tokens = np.ascontiguousarray(base[:, : cfg.t, : 3 * cfg.q], dtype=np.float32)
    clipped_counts = np.clip(counts.astype(int), 0, cfg.t).astype(np.int16)
    base_path = output / "inputs" / "base_tokens.npz"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base_path, tokens=tokens, event_counts=clipped_counts, labels=labels, smiles=smiles)
    _write_contract(
        cfg,
        {
            "hypothesis": "A frozen original-trained head separates useful event-order dependence from per-order retraining.",
            "reference_cache": str(cache_path),
            "order_variants": list(ORDER_VARIANTS),
            "block_size": cfg.block_size,
            "permutation_seed_map": {
                str(seed): int(permutation_seed)
                for seed, permutation_seed in zip(cfg.seeds, cfg.permutation_seeds)
            },
            "transfer_protocol": "train and select a head on original only; freeze weights and evaluate every reordered test representation",
        },
    )
    if cfg.run_qrc:
        extract_qrc_order_dual_gpu(
            base_path,
            output / "features",
            q=cfg.q,
            t=cfg.t,
            seeds=cfg.seeds,
            permutation_seeds=cfg.permutation_seeds,
            devices=cfg.qrc_devices,
            g=cfg.g,
            W=cfg.W,
            layers=cfg.layers,
            block_size=cfg.block_size,
            force=cfg.force,
            batch_size=cfg.qrc_batch_size,
        )
    all_metrics = []
    robustness = []
    for seed, permutation_seed in zip(cfg.seeds, cfg.permutation_seeds):
        ordered, names = build_order_tensor(
            tokens,
            clipped_counts,
            permutation_seed=permutation_seed,
            q=cfg.q,
            t=cfg.t,
            block_size=cfg.block_size,
        )
        raw = ordered.reshape(len(names), len(labels), -1)
        crc_path = _crc_path(output, seed)
        if crc_path.exists() and not cfg.force:
            with np.load(crc_path, allow_pickle=False) as package:
                crc = package["features"].astype(np.float32)
        else:
            crc = compute_crc_tensor(
                ordered,
                q=cfg.q,
                layers=cfg.layers,
                seed=seed,
                device=cfg.crc_device,
                state_scale=cfg.crc_state_scale,
                inject_scale=cfg.crc_inject_scale,
            )
            crc_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                crc_path,
                features=crc,
                variant_names=np.asarray(names),
                seed=seed,
                permutation_seed=permutation_seed,
            )
        crc3_path = _crc3_path(output, seed)
        crc3 = None if cfg.force else _load_valid_crc3_cache(
            crc3_path,
            shape=(len(names), len(labels), crc3_z_zz_dim(cfg.q)),
            names=names,
            q=cfg.q,
            t=cfg.t,
            seed=seed,
            permutation_seed=permutation_seed,
        )
        if crc3 is None:
            crc3 = compute_crc3_z_zz_tensor(
                ordered,
                q=cfg.q,
                layers=cfg.layers,
                seed=seed,
                device=cfg.crc_device,
            )
            crc3_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                crc3_path,
                features=crc3,
                variant_names=np.asarray(names),
                q=np.int16(cfg.q),
                t=np.int16(cfg.t),
                seed=np.int32(seed),
                permutation_seed=np.int64(permutation_seed),
                layers=np.int16(cfg.layers),
                method=np.asarray(CRC3_Z_ZZ_METHOD),
                baseline_protocol_version=np.asarray(CRC3_Z_ZZ_PROTOCOL),
                preprocessing=np.asarray("tanh"),
                readout=np.asarray("z+all_zz_no_pca"),
            )
        features_by_method = {
            CRC3_Z_ZZ_METHOD: crc3,
            "crc": crc,
            "raw_full": raw,
        }
        robustness.extend(
            representation_robustness_rows(
                crc3, names, method=CRC3_Z_ZZ_METHOD, seed=seed, reference="original"
            )
        )
        robustness.extend(
            representation_robustness_rows(crc, names, method="crc", seed=seed, reference="original")
        )
        robustness.extend(
            representation_robustness_rows(raw, names, method="raw_full", seed=seed, reference="original")
        )
        if cfg.run_qrc:
            with np.load(output / "features" / f"qrc_seed{seed:03d}.npz", allow_pickle=False) as package:
                qrc = package["features"].astype(np.float32)
                qrc_names = package["variant_names"].astype(str).tolist()
            if qrc_names != names:
                raise RuntimeError("QRC order variant mismatch")
            features_by_method = {"qrc": qrc, **features_by_method}
            robustness.extend(
                representation_robustness_rows(qrc, names, method="qrc", seed=seed, reference="original")
            )
        all_metrics.append(
            evaluate_frozen_head(
                features_by_method,
                names,
                labels,
                smiles,
                seed=seed,
                output_dir=output,
                linear_cfg=cfg.linear_config(),
                q=cfg.q,
                t=cfg.t,
                train_variant="original",
                experiment="frozen_cross_order",
                permutation_seed=permutation_seed,
            )
        )
    pd.DataFrame(robustness).to_csv(output / "representation_order_sensitivity.csv", index=False)
    summary = summarize_frozen_metrics(pd.concat(all_metrics, ignore_index=True), output)
    _plot_frozen(summary, output, f"{cfg.dataset}: frozen original-head order transfer, Q={cfg.q}, T={cfg.t}")
    return summary


def run_plan(plan_path: str | Path, *, profile_override: str | None = None) -> pd.DataFrame:
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if profile_override is not None:
        plan["profile"] = profile_override
        if profile_override == "quick":
            plan["output_dir"] = str(plan_path.parent / "outputs_quick")
    config = ValidityConfig(
        dataset=plan["dataset"],
        data_dir=plan["data_dir"],
        output_dir=plan.get("output_dir", str(plan_path.parent / "outputs")),
        experiment=plan["experiment"],
        q=int(plan.get("q", 16)),
        t=int(plan.get("t", 16)),
        profile=str(plan.get("profile", "full")),
        epochs=int(plan.get("epochs", 500)),
        seeds=tuple(plan.get("seeds", DEFAULT_SEEDS)),
        permutation_seeds=tuple(plan.get("permutation_seeds", PERMUTATION_SEEDS)),
        W=float(plan.get("W", 0.6)),
        g=float(plan.get("g", 0.84)),
        layers=int(plan.get("layers", 3)),
        qrc_devices=tuple(plan.get("qrc_devices", ("cuda:0", "cuda:1"))),
        qrc_batch_size=plan.get("qrc_batch_size"),
        crc_device=str(plan.get("crc_device", "auto")),
        train_device=str(plan.get("train_device", "auto")),
        relabel_repeats=int(plan.get("relabel_repeats", 2)),
        random_smiles_repeats=int(plan.get("random_smiles_repeats", 2)),
        root_repeats=int(plan.get("root_repeats", 4)),
        variant_seed=int(plan.get("variant_seed", 20260710)),
        block_size=int(plan.get("block_size", 2)),
        q_values=tuple(plan.get("q_values", range(1, 21))),
        t_values=tuple(plan.get("t_values", range(1, 31))),
        run_qrc=bool(plan.get("run_qrc", True)),
        force=bool(plan.get("force", False)),
    )
    if config.experiment == "invariance":
        return run_invariance(config)
    if config.experiment == "padding":
        return run_padding_ablation(config)
    if config.experiment == "collision":
        return run_collision_ceiling(config)
    if config.experiment == "cross_order":
        return run_cross_order(config)
    raise ValueError(config.experiment)
