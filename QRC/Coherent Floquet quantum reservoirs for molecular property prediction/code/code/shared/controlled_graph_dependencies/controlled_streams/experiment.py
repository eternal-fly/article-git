"""End-to-end controlled graph-stream experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .evaluation import (
    METHODS,
    fit_raw_pca,
    grouped_split,
    paired_statistics,
    plot_memory_curves,
    plot_overview,
    plot_setting,
    stratified_split,
    summarize_metrics,
    train_linear_probe,
    train_strong_raw_logreg,
)
from .reservoirs import (
    CRC3_Z_ZZ_PROTOCOL,
    ESN_STANDARD_PROTOCOL,
    crc3_z_zz_matched_features,
    crc_features,
    esn_features,
    esn_standard_features,
    qrc_features,
    readout_dim,
)
from .tasks import ControlledDataset, TaskSetting, generate_dataset, task_settings


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = tuple(range(0, 100, 10))
CANONICAL_W = 0.60
CANONICAL_G = 0.84
CANONICAL_LAYERS = 3


@dataclass(frozen=True)
class RunConfig:
    profile: str = "full"
    output_root: str = str(MODULE_ROOT / "outputs")
    q: int = 12
    n_groups: int = 128
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    generation_seed: int = 20260710
    g: float = CANONICAL_G
    W: float = CANONICAL_W
    layers: int = CANONICAL_LAYERS
    epochs: int = 500
    learning_rate: float = 1e-3
    batch_size: int = 64
    devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    train_device: str = "cuda:0"
    methods: tuple[str, ...] = METHODS
    qrc_batch_size: int | None = None
    force: bool = False
    progress: bool = True

    @classmethod
    def for_profile(cls, profile: str, **overrides: Any) -> "RunConfig":
        profile = profile.lower()
        if profile == "smoke":
            base = cls(
                profile="smoke",
                q=6,
                n_groups=6,
                seeds=(0,),
                epochs=8,
                batch_size=32,
                devices=("cpu",),
                train_device="cpu",
                progress=False,
            )
        elif profile == "quick":
            base = cls(
                profile="quick",
                q=8,
                n_groups=48,
                seeds=(0, 10),
                epochs=100,
                batch_size=64,
            )
        elif profile == "full":
            base = cls(profile="full")
        else:
            raise ValueError("profile must be smoke, quick, or full")
        return replace(base, **overrides)

    def validate(self) -> "RunConfig":
        if self.profile not in {"smoke", "quick", "full", "replay"}:
            raise ValueError(f"Unsupported profile {self.profile!r}")
        if self.q < 2:
            raise ValueError("Q must be >= 2")
        if self.n_groups < 3:
            raise ValueError("n_groups must be >= 3")
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        if not self.devices:
            raise ValueError("At least one compute device is required")
        return self


def available_devices() -> tuple[str, ...]:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return tuple(f"cuda:{index}" for index in range(min(2, count)))
    return ("cpu",)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _stable_setting_seed(base_seed: int, setting_name: str) -> int:
    digest = hashlib.sha256(setting_name.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32)


def _save_dataset(path: Path, dataset: ControlledDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "tokens": dataset.tokens.astype(np.float32),
        "labels": dataset.labels.astype(np.int64),
        "groups": dataset.groups.astype(np.int64),
    }
    if dataset.symbols is not None:
        payload["symbols"] = dataset.symbols.astype(np.int16)
    np.savez_compressed(path, **payload)
    json_dump(path.with_suffix(".json"), dataset.metadata)


def _input_audit(dataset: ControlledDataset) -> dict[str, Any]:
    labels = dataset.labels
    flat = dataset.tokens.reshape(len(dataset.tokens), -1)
    audit: dict[str, Any] = {
        "shape": list(dataset.tokens.shape),
        "class_counts": {str(c): int((labels == c).sum()) for c in (0, 1)},
        "raw_class_centroid_l2": float(np.linalg.norm(flat[labels == 1].mean(0) - flat[labels == 0].mean(0))),
        "raw_class_centroid_max_abs": float(np.max(np.abs(flat[labels == 1].mean(0) - flat[labels == 0].mean(0)))),
    }
    if dataset.symbols is not None:
        valid = dataset.symbols[dataset.symbols >= 0]
        max_symbol = int(valid.max()) if len(valid) else -1
        histograms: dict[str, list[int]] = {}
        for label in (0, 1):
            values = dataset.symbols[labels == label]
            values = values[values >= 0]
            histograms[str(label)] = np.bincount(values, minlength=max_symbol + 1).astype(int).tolist()
        audit["aggregate_symbol_histogram_by_label"] = histograms
    return audit


def load_token_replay(
    cache_path: str | Path,
    q: int,
    t: int,
    name: str,
) -> tuple[TaskSetting, ControlledDataset]:
    """Load an existing FINAL BACE/BBBP token cache without regenerating chemistry."""

    cache_path = Path(cache_path).resolve()
    with np.load(cache_path, allow_pickle=False) as package:
        if "tokens" not in package or "labels" not in package:
            raise KeyError(f"{cache_path} must contain tokens and labels")
        tokens = package["tokens"][:, : int(t), : 3 * int(q)].astype(np.float32)
        labels = package["labels"].astype(np.int64)
    if tokens.shape[1] != int(t) or tokens.shape[2] != 3 * int(q):
        raise ValueError(f"Replay cache is too small for Q={q}, T={t}: {tokens.shape}")
    setting = TaskSetting(
        name=f"replay_{name}_Q{q:02d}_T{t:02d}",
        family="token_replay",
        sequence_length=int(t),
        parameter_name="dataset",
        parameter_value=name,
    )
    dataset = ControlledDataset(
        tokens=tokens,
        labels=labels,
        groups=np.arange(len(labels), dtype=np.int64),
        symbols=None,
        metadata={
            "task": setting.to_dict(),
            "representation": "existing_FINAL_token_replay",
            "source_cache": str(cache_path),
            "q": int(q),
            "t": int(t),
            "n_samples": int(len(labels)),
            "class_counts": {str(c): int((labels == c).sum()) for c in (0, 1)},
            "split_policy": "paired stratified sample split; replay is descriptive, not a controlled synthetic task",
        },
    )
    dataset.validate(q)
    return setting, dataset


def _feature_path(setting_dir: Path, method: str, seed: int) -> Path:
    return setting_dir / "features" / f"{method}_seed{int(seed):03d}.npz"


def _valid_feature(
    path: Path,
    n: int,
    d: int,
    *,
    method: str | None = None,
    protocol: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as package:
            valid = package["features"].shape == (int(n), int(d)) and np.isfinite(package["features"]).all()
            if method is not None:
                valid = valid and "method" in package and str(package["method"].item()) == str(method)
            if protocol is not None:
                valid = valid and "protocol" in package and str(package["protocol"].item()) == str(protocol)
            return bool(valid)
    except Exception:
        return False


def _reservoir_feature_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """One top-level spawn-safe worker owns one GPU and a disjoint seed subset."""

    data_path = Path(payload["data_path"])
    setting_dir = Path(payload["setting_dir"])
    with np.load(data_path, allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32)
    q = int(payload["q"])
    output_dim = readout_dim(q)
    methods = tuple(payload["methods"])
    completed: list[tuple[str, int]] = []
    for seed in payload["seeds"]:
        seed = int(seed)
        for method in ("qrc", "crc", "crc3_z_zz_matched", "esn_standard", "esn"):
            if method not in methods:
                continue
            path = _feature_path(setting_dir, method, seed)
            expected_protocol = (
                CRC3_Z_ZZ_PROTOCOL if method == "crc3_z_zz_matched"
                else ESN_STANDARD_PROTOCOL if method == "esn_standard" else None
            )
            if not payload["force"] and _valid_feature(
                path,
                len(tokens),
                output_dim,
                method=method,
                protocol=expected_protocol,
            ):
                completed.append((method, seed))
                continue
            if method == "qrc":
                features, metadata = qrc_features(
                    tokens,
                    q,
                    seed,
                    payload["device"],
                    g=float(payload["g"]),
                    W=float(payload["W"]),
                    layers=int(payload["layers"]),
                    batch_size=payload["qrc_batch_size"],
                    progress=bool(payload["progress"]),
                )
            elif method == "crc":
                features, metadata = crc_features(
                    tokens,
                    q,
                    seed,
                    payload["device"],
                    layers=int(payload["layers"]),
                )
            elif method == "crc3_z_zz_matched":
                features, metadata = crc3_z_zz_matched_features(
                    tokens,
                    q,
                    seed,
                    payload["device"],
                    layers=int(payload["layers"]),
                )
            elif method == "esn_standard":
                features, metadata = esn_standard_features(
                    tokens, q, seed, payload["device"], layers=int(payload["layers"])
                )
            else:
                features, metadata = esn_features(
                    tokens,
                    q,
                    seed,
                    payload["device"],
                    layers=int(payload["layers"]),
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                features=features,
                method=np.asarray(method),
                seed=np.int32(seed),
                q=np.int16(q),
                output_dim=np.int16(output_dim),
                protocol=np.asarray(str(metadata.get("protocol", ""))),
            )
            json_dump(path.with_suffix(".json"), metadata)
            completed.append((method, seed))
    return {"device": payload["device"], "completed": completed}


def extract_reservoir_features(
    data_path: Path,
    setting_dir: Path,
    cfg: RunConfig,
) -> None:
    reservoir_methods = tuple(
        m for m in cfg.methods if m in {"qrc", "crc", "crc3_z_zz_matched", "esn_standard", "esn"}
    )
    if not reservoir_methods:
        return
    assignments = [list(cfg.seeds[index :: len(cfg.devices)]) for index in range(len(cfg.devices))]
    jobs = []
    for device, seeds in zip(cfg.devices, assignments):
        if seeds:
            jobs.append(
                {
                    "device": device,
                    "seeds": seeds,
                    "methods": reservoir_methods,
                    "data_path": str(data_path),
                    "setting_dir": str(setting_dir),
                    "q": cfg.q,
                    "g": cfg.g,
                    "W": cfg.W,
                    "layers": cfg.layers,
                    "qrc_batch_size": cfg.qrc_batch_size,
                    "force": cfg.force,
                    "progress": cfg.progress,
                }
            )
    if len(jobs) == 1:
        _reservoir_feature_worker(jobs[0])
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
        futures = [pool.submit(_reservoir_feature_worker, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


def _load_feature(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as package:
        return package["features"].astype(np.float32)


def _setting_signature(setting: TaskSetting, dataset: ControlledDataset, cfg: RunConfig, replay: bool) -> dict[str, Any]:
    return {
        "setting": setting.to_dict(),
        "dataset_metadata": dataset.metadata,
        "profile": cfg.profile,
        "q": cfg.q,
        "g": cfg.g,
        "W": cfg.W,
        "layers": cfg.layers,
        "seeds": list(cfg.seeds),
        "methods": list(cfg.methods),
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "split": "70/15/15 nuisance-group split" if not replay else "70/15/15 stratified sample split",
        "selection": "max validation AUC, then min loss, then max ACC",
        "qrc_readout": "Z+all_ZZ",
        "crc3_z_zz_protocol": CRC3_Z_ZZ_PROTOCOL,
        "esn_standard_protocol": ESN_STANDARD_PROTOCOL,
        "canonical_parameters": {"W": CANONICAL_W, "g": CANONICAL_G, "layers": CANONICAL_LAYERS},
    }


def _additive_setting_signature_compatible(current: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Permit adding a method without invalidating scientifically identical work.

    Every field other than the method list must match.  The one migration from
    pre-CRC3 manifests is accepted only when the old method set did not contain
    CRC3; after migration the full expected signature is written back.
    """

    old = dict(current)
    new = dict(expected)
    old_methods = set(old.pop("methods", []))
    new_methods = set(new.pop("methods", []))
    if not old_methods.issubset(new_methods):
        return False
    if "crc3_z_zz_protocol" not in old and "crc3_z_zz_matched" not in old_methods:
        old["crc3_z_zz_protocol"] = new.get("crc3_z_zz_protocol")
    if "esn_standard_protocol" not in old and "esn_standard" not in old_methods:
        old["esn_standard_protocol"] = new.get("esn_standard_protocol")
    return old == new


def run_dataset_setting(
    setting: TaskSetting,
    dataset: ControlledDataset,
    cfg: RunConfig,
    *,
    replay: bool = False,
) -> pd.DataFrame:
    """Run every paired method and seed for one controlled setting."""

    cfg.validate()
    setting_dir = Path(cfg.output_root) / cfg.profile / setting.family / setting.name
    setting_dir.mkdir(parents=True, exist_ok=True)
    signature = _setting_signature(setting, dataset, cfg, replay)
    config_path = setting_dir / "setting_config.json"
    if config_path.exists() and not cfg.force:
        current = json.loads(config_path.read_text(encoding="utf-8"))
        if current != signature and not _additive_setting_signature_compatible(current, signature):
            raise RuntimeError(f"Existing config differs at {config_path}; use --force or a new output root")
    json_dump(config_path, signature)
    data_path = setting_dir / "dataset.npz"
    if cfg.force or not data_path.exists():
        _save_dataset(data_path, dataset)
    json_dump(setting_dir / "input_audit.json", _input_audit(dataset))
    extract_reservoir_features(data_path, setting_dir, cfg)

    raw = dataset.tokens.reshape(len(dataset.tokens), -1).astype(np.float32)
    raw_path = setting_dir / "features" / "raw_full.npz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.force or not raw_path.exists():
        np.savez_compressed(raw_path, features=raw)

    rows: list[dict[str, Any]] = []
    predictions_all: list[pd.DataFrame] = []
    for seed in cfg.seeds:
        split = stratified_split(dataset.labels, seed) if replay else grouped_split(dataset.groups, dataset.labels, seed)
        json_dump(setting_dir / "splits" / f"seed{seed:03d}.json", split)
        pca_features: np.ndarray | None = None
        if "raw_pca" in cfg.methods:
            pca_path = _feature_path(setting_dir, "raw_pca", seed)
            if not cfg.force and _valid_feature(pca_path, len(raw), readout_dim(cfg.q)):
                pca_features = _load_feature(pca_path)
            else:
                pca_features, pca_meta = fit_raw_pca(raw, split, readout_dim(cfg.q))
                np.savez_compressed(pca_path, features=pca_features, seed=np.int32(seed), q=np.int16(cfg.q))
                json_dump(pca_path.with_suffix(".json"), pca_meta)

        for method in cfg.methods:
            run_dir = setting_dir / "runs"
            metric_path = run_dir / f"{method}_seed{seed:03d}_metrics.json"
            prediction_path = run_dir / f"{method}_seed{seed:03d}_test_predictions.csv"
            history_path = setting_dir / "histories" / f"{method}_seed{seed:03d}.csv"
            resume_run = (
                not cfg.force
                and metric_path.exists()
                and prediction_path.exists()
                and history_path.exists()
            )
            cached_row = json.loads(metric_path.read_text(encoding="utf-8")) if resume_run else None
            if (
                method == "crc3_z_zz_matched"
                and cached_row is not None
                and cached_row.get("feature_protocol") != CRC3_Z_ZZ_PROTOCOL
            ):
                resume_run = False
            if resume_run and cached_row is not None:
                row = cached_row
                predictions = pd.read_csv(prediction_path)
            else:
                if method == "raw_logreg_cv":
                    metric, history, predictions, checkpoint = train_strong_raw_logreg(raw, dataset.labels, split, seed)
                    checkpoint_path = setting_dir / "checkpoints" / f"{method}_seed{seed:03d}_best.npz"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(checkpoint_path, **checkpoint)
                else:
                    if method in {"qrc", "crc", "crc3_z_zz_matched", "esn_standard", "esn"}:
                        features = _load_feature(_feature_path(setting_dir, method, seed))
                    elif method == "raw_full":
                        features = raw
                    elif method == "raw_pca" and pca_features is not None:
                        features = pca_features
                    else:
                        raise KeyError(method)
                    metric, history, predictions, checkpoint = train_linear_probe(
                        features,
                        dataset.labels,
                        split,
                        seed,
                        epochs=cfg.epochs,
                        learning_rate=cfg.learning_rate,
                        batch_size=cfg.batch_size,
                        device=cfg.train_device,
                    )
                    checkpoint_path = setting_dir / "checkpoints" / f"{method}_seed{seed:03d}_best.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "state_dict": checkpoint,
                            "method": method,
                            "seed": int(seed),
                            "selection": metric["selection_rule"],
                        },
                        checkpoint_path,
                    )
                row = {
                    "setting": setting.name,
                    "family": setting.family,
                    "parameter_name": setting.parameter_name,
                    "parameter_value": setting.parameter_value,
                    "method": method,
                    "seed": int(seed),
                    "q": int(cfg.q),
                    "t": int(dataset.tokens.shape[1]),
                    "feature_protocol": (
                        CRC3_Z_ZZ_PROTOCOL if method == "crc3_z_zz_matched" else None
                    ),
                    **metric,
                }
                json_dump(metric_path, row)
                history_path.parent.mkdir(parents=True, exist_ok=True)
                history.to_csv(history_path, index=False)
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                predictions.assign(
                    setting=setting.name,
                    family=setting.family,
                    method=method,
                    seed=int(seed),
                ).to_csv(prediction_path, index=False)
            rows.append(row)
            predictions_all.append(predictions.assign(setting=setting.name, method=method, seed=int(seed)))
    metrics = pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)
    metrics.to_csv(setting_dir / "metrics_by_seed.csv", index=False)
    pd.concat(predictions_all, ignore_index=True).to_csv(setting_dir / "test_predictions.csv", index=False)
    summary = summarize_metrics(metrics)
    paired = paired_statistics(metrics)
    summary.to_csv(setting_dir / "performance_summary.csv", index=False)
    paired.to_csv(setting_dir / "paired_statistics.csv", index=False)
    plot_setting(summary, setting.name, setting_dir / "figures" / "auc_acc_summary.png")
    return metrics


def run_experiment(
    cfg: RunConfig,
    settings: Iterable[TaskSetting] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete synthetic profile sequentially, using both GPUs within settings."""

    cfg = cfg.validate()
    selected = list(task_settings(cfg.profile) if settings is None else settings)
    root = Path(cfg.output_root) / cfg.profile
    root.mkdir(parents=True, exist_ok=True)
    (root / "ALL_DONE.txt").unlink(missing_ok=True)
    json_dump(
        root / "run_manifest.json",
        {
            "config": asdict(cfg),
            "settings": [setting.to_dict() for setting in selected],
            "execution": "settings sequential; reservoir seeds striped across devices",
        },
    )
    metric_frames: list[pd.DataFrame] = []
    for index, setting in enumerate(selected, start=1):
        print(f"[{index:02d}/{len(selected):02d}] {setting.name}", flush=True)
        # A family-stable seed makes lag/prefix curves nested controls rather than
        # comparisons between unrelated nuisance realizations.
        generation_seed = _stable_setting_seed(cfg.generation_seed, setting.family)
        dataset = generate_dataset(setting, cfg.q, cfg.n_groups, generation_seed)
        metric_frames.append(run_dataset_setting(setting, dataset, cfg))
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    paired = paired_statistics(metrics)
    metrics.to_csv(root / "metrics_all_settings.csv", index=False)
    summary.to_csv(root / "summary_all_settings.csv", index=False)
    paired.to_csv(root / "paired_statistics_all_settings.csv", index=False)
    figure_dir = root / "figures"
    plot_memory_curves(summary, figure_dir)
    plot_overview(summary, figure_dir / "auc_overview.png")
    (root / "ALL_DONE.txt").write_text("Controlled graph-stream profile completed.\n", encoding="utf-8")
    return metrics, summary, paired


def run_replay(
    cfg: RunConfig,
    cache_path: str | Path,
    name: str,
    t: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay_cfg = replace(cfg, profile="replay").validate()
    setting, dataset = load_token_replay(cache_path, replay_cfg.q, int(t), name)
    metrics = run_dataset_setting(setting, dataset, replay_cfg, replay=True)
    summary = summarize_metrics(metrics)
    paired = paired_statistics(metrics)
    return metrics, summary, paired
