"""Strict QRC/CRC/raw linear-readout workflow for BACE and BBBP.

This module is imported by the notebooks.  Keeping multiprocessing workers here
(rather than in notebook cells) makes the two-GPU ``spawn`` workflow reliable.

Comparison contract
-------------------
* One task-independent local BFS event stream is used by all methods.
* A Q-qubit setting consumes the first 3Q token coordinates; a T-event setting
  consumes the first T events.  No token schema is redesigned between settings.
* QRC and CRC both receive ``tanh(tokens)``, use Q internal coordinates, three
  reservoir layers per injected event, and return Z/ZZ-aligned dimensions
  ``Q + Q(Q-1)/2``.  CRC is the old scalar-state reservoir, not a 3Q-state CRC.
* CRC3-z/zz receives the same 3Q coordinates as QRC, evolves a Q-by-3
  classical state, and directly exposes ``[s_i^z, s_i^z s_j^z]``.  It has the
  same readout dimension as QRC and never uses PCA or label information.
* Raw-full is the original token tensor directly flattened to ``T*3Q``; it does
  not pass through a reservoir or an extra preprocessing transform.
* Raw-PCA is fitted on each training split only, then zero-padded when needed to
  the QRC/CRC readout dimension ``Q + Q(Q-1)/2``.
* Every classifier is exactly ``torch.nn.Linear(d, 1)``.  There is no feature
  standardization, hidden layer, class weighting, dropout, or weight decay.
* All heads run 500 epochs.  Selection is lexicographic on validation metrics:
  maximum AUC, then minimum loss, then maximum ACC.  Test data never select an
  epoch and are evaluated only after restoring the selected checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .crc3_baseline import (
    CRC3_Z_ZZ_METHOD,
    CRC3_Z_ZZ_PROTOCOL,
    compute_crc3_z_zz_features,
)
from .esn_baseline import (
    ESN_STANDARD_METHOD,
    ESN_STANDARD_PROTOCOL,
    compute_esn_standard_features,
)
from .general_tokens import MAX_Q, MAX_TOKEN_DIM, SCHEMA_VERSION, tokenize_mol
from .gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu


RDLogger.DisableLog("rdApp.*")

DEFAULT_SEEDS = tuple(range(0, 100, 10))
METHODS = ("qrc", CRC3_Z_ZZ_METHOD, ESN_STANDARD_METHOD, "crc", "raw_full", "raw_pca")
METHOD_LABELS = {
    "qrc": "QRC",
    CRC3_Z_ZZ_METHOD: "CRC3-z/zz",
    ESN_STANDARD_METHOD: "Standard ESN",
    "crc": "CRC",
    "raw_full": "Raw-full",
    "raw_pca": "Raw-PCA",
}
METHOD_COLORS = {
    "qrc": "#2F6B9A",
    CRC3_Z_ZZ_METHOD: "#2A9D8F",
    ESN_STANDARD_METHOD: "#E9A23B",
    "crc": "#D07A35",
    "raw_full": "#6F6F6F",
    "raw_pca": "#8E5EA2",
}
SELECTION_RULE = "max(val_auc) > min(val_loss) > max(val_acc)"
TOKEN_VARIANT_GENERAL = "general_task_independent"
TOKEN_VARIANT_CHEMISTRY = "chemistry_task_aware"


def normalize_token_variant(value: str | None) -> str:
    raw = (value or TOKEN_VARIANT_GENERAL).strip().lower()
    aliases = {
        "general": TOKEN_VARIANT_GENERAL,
        "general_task_independent": TOKEN_VARIANT_GENERAL,
        "task_independent": TOKEN_VARIANT_GENERAL,
        "chemistry": TOKEN_VARIANT_CHEMISTRY,
        "chemistry_task_aware": TOKEN_VARIANT_CHEMISTRY,
        "task_aware": TOKEN_VARIANT_CHEMISTRY,
    }
    if raw not in aliases:
        raise ValueError(f"Unsupported token_variant={value!r}; expected general or chemistry.")
    return aliases[raw]


def token_variant_is_task_aware(token_variant: str | None) -> bool:
    return normalize_token_variant(token_variant) == TOKEN_VARIANT_CHEMISTRY


def token_variant_cache_stem(token_variant: str | None) -> str:
    return "chemistry" if token_variant_is_task_aware(token_variant) else "general"


def token_variant_description(token_variant: str | None) -> str:
    return (
        "task-aware chemistry slots contain fixed label-free dataset-directed descriptors"
        if token_variant_is_task_aware(token_variant)
        else "task-independent local/prefix-causal generic tokens; reserved chemistry slots are fixed to zero"
    )


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    data_dir: str
    scan_root: str
    token_variant: str = TOKEN_VARIANT_GENERAL
    epochs: int = 500
    learning_rate: float = 1e-3
    batch_size: int = 32
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    g: float = 0.84
    W: float = 0.4
    layers: int = 3
    features_per_qubit: int = 3
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    crc_state_scale: float = 0.85
    crc_inject_scale: float = 0.75
    max_cached_events: int = 30
    qrc_devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    qrc_batch_size: int | None = None
    crc_device: str = "auto"
    train_device: str = "auto"
    run_qrc: bool = True
    force: bool = False

    def normalized(self) -> "ExperimentConfig":
        dataset = self.dataset.upper()
        if dataset not in {"BACE", "BBBP"}:
            raise ValueError(f"dataset must be BACE or BBBP, got {self.dataset!r}")
        if self.features_per_qubit != 3:
            raise ValueError("This aligned experiment requires exactly three coordinates per qubit.")
        if self.epochs != 500:
            raise ValueError("The strict protocol fixes epochs=500.")
        seeds = tuple(int(seed) for seed in self.seeds)
        if seeds != DEFAULT_SEEDS:
            raise ValueError(f"The strict protocol fixes seeds={DEFAULT_SEEDS}, got {seeds}.")
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1):
            raise ValueError("Invalid train/validation ratios.")
        token_variant = normalize_token_variant(self.token_variant)
        return ExperimentConfig(
            **{
                **asdict(self),
                "dataset": dataset,
                "data_dir": str(Path(self.data_dir).resolve()),
                "scan_root": str(Path(self.scan_root).resolve()),
                "token_variant": token_variant,
                "seeds": seeds,
                "qrc_devices": tuple(self.qrc_devices),
                "crc_device": str(self.crc_device),
                "train_device": str(self.train_device),
            }
        )


def resolve_torch_device(preferred: str | None = "auto") -> torch.device:
    """Resolve an experiment device string, preferring CUDA whenever possible."""
    value = "auto" if preferred is None else str(preferred).strip().lower()
    if value in {"", "auto", "gpu", "cuda"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _dataset_spec(dataset: str, data_dir: str | Path) -> tuple[Path, str, str]:
    root = Path(data_dir)
    if dataset == "BACE":
        return root / "bace" / "raw" / "bace.csv", "mol", "Class"
    return root / "bbbp" / "raw" / "BBBP.csv", "smiles", "p_np"


def _dataset_fingerprint(smiles: Sequence[str], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for smi, label in zip(smiles, labels):
        digest.update(smi.encode("utf-8"))
        digest.update(bytes([int(label)]))
    return digest.hexdigest()


def load_or_build_token_cache(
    dataset: str,
    data_dir: str | Path,
    cache_dir: str | Path,
    max_events: int = 30,
    token_variant: str = TOKEN_VARIANT_GENERAL,
) -> tuple[Path, dict]:
    """Load all valid labelled rows and encode one fixed Q24 BFS stream."""
    dataset = dataset.upper()
    token_variant = normalize_token_variant(token_variant)
    task_aware = token_variant_is_task_aware(token_variant)
    cache_stem = token_variant_cache_stem(token_variant)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset}_{cache_stem}_tokens_Q24_T{max_events:02d}.npz"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") == SCHEMA_VERSION and metadata.get("token_variant") == token_variant:
            return cache_path, metadata

    csv_path, smiles_col, label_col = _dataset_spec(dataset, data_dir)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    token_rows: list[np.ndarray] = []
    canonical_smiles: list[str] = []
    labels: list[int] = []
    source_row_indices: list[int] = []
    event_counts: list[int] = []
    invalid_rows = 0
    invalid_labels = 0
    for row_index, row in frame.iterrows():
        try:
            label = int(float(row[label_col]))
        except (TypeError, ValueError):
            invalid_labels += 1
            continue
        if label not in (0, 1):
            invalid_labels += 1
            continue
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None or mol.GetNumAtoms() == 0:
            invalid_rows += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        tokens, count = tokenize_mol(mol, dataset, task_aware=task_aware, max_events=max_events)
        token_rows.append(tokens)
        canonical_smiles.append(canonical)
        labels.append(label)
        source_row_indices.append(int(row_index))
        event_counts.append(int(count))
    if not token_rows:
        raise RuntimeError(f"No valid labelled molecules found in {csv_path}")
    tokens = np.stack(token_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(
        cache_path,
        tokens=tokens,
        labels=y,
        smiles=np.asarray(canonical_smiles),
        event_counts=np.asarray(event_counts, dtype=np.int16),
        source_row_indices=np.asarray(source_row_indices, dtype=np.int32),
    )
    metadata = {
        "dataset": dataset,
        "token_variant": token_variant,
        "schema_version": SCHEMA_VERSION,
        "task_aware": bool(task_aware),
        "task_slots": token_variant_description(token_variant),
        "source_csv": str(csv_path.resolve()),
        "source_rows": int(len(frame)),
        "valid_labelled_rows": int(len(y)),
        "invalid_smiles_rows": int(invalid_rows),
        "invalid_label_rows": int(invalid_labels),
        "class_counts": {str(value): int((y == value).sum()) for value in (0, 1)},
        "shape": list(tokens.shape),
        "max_Q": MAX_Q,
        "max_T": int(max_events),
        "max_token_dim": MAX_TOKEN_DIM,
        "fingerprint": _dataset_fingerprint(canonical_smiles, y),
    }
    _json_dump(metadata_path, metadata)
    return cache_path, metadata


def _load_cache(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as package:
        return (
            package["tokens"].astype(np.float32, copy=False),
            package["labels"].astype(np.int64, copy=False),
            package["smiles"].astype(str),
        )


def _murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split_indices(
    smiles: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact old stable scaffold policy, with seed-dependent tie shuffling."""
    groups: dict[str, list[int]] = {}
    for index, smi in enumerate(smiles):
        groups.setdefault(_murcko_scaffold(str(smi)), []).append(index)
    scaffold_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    rng = random.Random(int(seed))
    shuffled: list[tuple[str, list[int]]] = []
    i = 0
    while i < len(scaffold_groups):
        j = i
        size = len(scaffold_groups[i][1])
        while j < len(scaffold_groups) and len(scaffold_groups[j][1]) == size:
            j += 1
        block = scaffold_groups[i:j]
        rng.shuffle(block)
        shuffled.extend(block)
        i = j
    n = len(smiles)
    n_train, n_val = int(n * train_ratio), int(n * val_ratio)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _, indices in shuffled:
        if len(train) + len(indices) <= n_train:
            train.extend(indices)
        elif len(val) + len(indices) <= n_val:
            val.extend(indices)
        else:
            test.extend(indices)
    return tuple(np.asarray(values, dtype=np.int64) for values in (train, val, test))


def _split_payload(
    smiles: Sequence[str], labels: np.ndarray, cfg: ExperimentConfig, seed: int
) -> dict:
    train, val, test = scaffold_split_indices(smiles, cfg.train_ratio, cfg.val_ratio, seed)
    payload = {
        "seed": int(seed),
        "split_type": "stable_scaffold_seeded_tie_shuffle",
        "train_indices": train.tolist(),
        "val_indices": val.tolist(),
        "test_indices": test.tolist(),
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "class_counts": {
            name: {str(c): int((labels[idx] == c).sum()) for c in (0, 1)}
            for name, idx in (("train", train), ("val", val), ("test", test))
        },
    }
    for name, idx in (("validation", val), ("test", test)):
        if len(np.unique(labels[idx])) < 2:
            raise RuntimeError(f"Seed {seed} produced a one-class {name} split: {payload['class_counts']}")
    return payload


def _adaptive_qrc_batch_size(q: int) -> int:
    if q >= 22:
        return 1
    if q >= 20:
        return 2
    if q >= 18:
        return 4
    if q >= 16:
        return 8
    if q >= 13:
        return 16
    if q >= 9:
        return 32
    return 64


def _readout_dim(q: int) -> int:
    return int(q + q * (q - 1) // 2)


def _valid_feature_file(
    path: Path,
    n: int,
    dim: int,
    q: int,
    t: int,
    seed: int,
    token_schema_version: str = SCHEMA_VERSION,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["features"].shape == (n, dim)
                and int(data["q"]) == q
                and int(data["t"]) == t
                and int(data["seed"]) == seed
                and "token_schema_version" in data
                and str(data["token_schema_version"].item()) == str(token_schema_version)
            )
    except Exception:
        return False


def _valid_raw_full_file(
    path: Path,
    n: int,
    dim: int,
    q: int,
    t: int,
    token_schema_version: str = SCHEMA_VERSION,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["features"].shape == (n, dim)
                and int(data["q"]) == q
                and int(data["t"]) == t
                and "token_schema_version" in data
                and str(data["token_schema_version"].item()) == str(token_schema_version)
            )
    except Exception:
        return False


def _qrc_device_worker(job: dict) -> list[str]:
    """One spawned process owns one GPU and handles its assigned seeds serially."""
    device = job["device"]
    q, t = int(job["q"]), int(job["t"])
    tokens, _, _ = _load_cache(job["cache_path"])
    tokens_tensor = torch.from_numpy(np.ascontiguousarray(tokens[:, :t, : 3 * q]))
    out_dir = Path(job["feature_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DTCQRCConfig(
        num_qubits=q,
        features_per_qubit=3,
        g=float(job["g"]),
        W_scale=float(job["W"]),
        layers_per_injection=int(job["layers"]),
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    finished: list[str] = []
    for seed in job["seeds"]:
        seed = int(seed)
        path = out_dir / f"qrc_seed{seed:03d}.npz"
        if not job["force"] and path.exists() and _valid_feature_file(
            path,
            len(tokens_tensor),
            cfg.output_dim,
            q,
            t,
            seed,
            token_schema_version=str(job.get("token_schema_version", SCHEMA_VERSION)),
        ):
            finished.append(str(path))
            continue
        features, params = compute_qrc_features_gpu(
            tokens_tensor,
            cfg=cfg,
            seed=seed,
            batch_size=int(job["batch_size"]),
            device=device,
            progress=True,
        )
        np.savez_compressed(
            path,
            features=features.numpy().astype(np.float32),
            J=params["J"],
            h=params["h"],
            q=np.int16(q),
            t=np.int16(t),
            seed=np.int32(seed),
            token_schema_version=np.asarray(str(job.get("token_schema_version", SCHEMA_VERSION))),
            g=np.float32(job["g"]),
            W=np.float32(job["W"]),
            layers=np.int16(job["layers"]),
            preprocessing=np.asarray("tanh"),
            observables=np.asarray("Z+all_ZZ"),
        )
        finished.append(str(path))
    return finished


def extract_qrc_features_dual_gpu(
    cache_path: Path,
    setting_dir: Path,
    q: int,
    t: int,
    cfg: ExperimentConfig,
    token_schema_version: str = SCHEMA_VERSION,
) -> None:
    devices = tuple(cfg.qrc_devices)
    if not devices:
        raise ValueError("At least one CUDA device is required when run_qrc=True.")
    assignments = [list(cfg.seeds[index:: len(devices)]) for index in range(len(devices))]
    jobs = []
    for device, seeds in zip(devices, assignments):
        if not seeds:
            continue
        jobs.append(
            {
                "device": device,
                "seeds": seeds,
                "cache_path": str(cache_path),
                "feature_dir": str(setting_dir / "features"),
                "q": q,
                "t": t,
                "g": cfg.g,
                "W": cfg.W,
                "layers": cfg.layers,
                "batch_size": cfg.qrc_batch_size or _adaptive_qrc_batch_size(q),
                "force": cfg.force,
                "token_schema_version": str(token_schema_version),
            }
        )
    if len(jobs) == 1:
        _qrc_device_worker(jobs[0])
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
        futures = [pool.submit(_qrc_device_worker, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


@torch.no_grad()
def compute_crc_features(
    tokens: np.ndarray,
    q: int,
    layers: int,
    seed: int,
    state_scale: float = 0.85,
    inject_scale: float = 0.75,
    device: str | torch.device | None = "auto",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vectorized old scalar-Q CRC with final state and all pair products."""
    compute_device = resolve_torch_device(str(device) if device is not None else "auto")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    encoder_cpu = torch.randn((q, 3), generator=generator) / math.sqrt(3.0)
    enc_bias_cpu = torch.randn(q, generator=generator) / math.sqrt(max(1, q))
    W_state_cpu = torch.randn((q, q), generator=generator) / math.sqrt(max(1, q))
    W_inj_cpu = torch.randn((q, q), generator=generator) / math.sqrt(max(1, q))
    bias_cpu = torch.randn(q, generator=generator) / math.sqrt(max(1, q))
    encoder = encoder_cpu.to(compute_device)
    enc_bias = enc_bias_cpu.to(compute_device)
    W_state = W_state_cpu.to(compute_device)
    W_inj = W_inj_cpu.to(compute_device)
    bias = bias_cpu.to(compute_device)
    x = torch.tanh(
        torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.float32)).to(compute_device)
    ).reshape(len(tokens), -1, q, 3)
    injections = torch.sin(torch.einsum("ntqf,qf->ntq", x, encoder) + enc_bias)
    state = torch.zeros((len(tokens), q), dtype=torch.float32, device=compute_device)
    for time_index in range(injections.shape[1]):
        inj = injections[:, time_index, :]
        state = torch.tanh(state_scale * state + inject_scale * inj)
        for _ in range(int(layers)):
            state = torch.tanh(
                state_scale * (state @ W_state.T)
                + inject_scale * (inj @ W_inj.T)
                + bias
            )
    columns = [state]
    if q >= 2:
        pairs = [state[:, i] * state[:, j] for i in range(q) for j in range(i + 1, q)]
        columns.append(torch.stack(pairs, dim=1))
    if compute_device.type == "cuda":
        torch.cuda.synchronize(compute_device)
    features = torch.cat(columns, dim=1).detach().cpu().numpy().astype(np.float32)
    params = {
        "encoder": encoder_cpu.numpy(),
        "enc_bias": enc_bias_cpu.numpy(),
        "W_state": W_state_cpu.numpy(),
        "W_inj": W_inj_cpu.numpy(),
        "bias": bias_cpu.numpy(),
    }
    return features, params


def _metric_bundle(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    auc = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "auc": auc,
        "acc": float(accuracy_score(y, predictions)),
    }


def _linear_logits(model: torch.nn.Linear, features: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(features).squeeze(1).cpu().numpy()


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict,
    method: str,
    seed: int,
    setting_dir: Path,
    cfg: ExperimentConfig,
    q: int,
    t: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Train exactly one affine logit and select only with validation data."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_device = resolve_torch_device(cfg.train_device)
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(train_device)
    y = torch.from_numpy(labels.astype(np.float32)).to(train_device)
    train_np = np.asarray(split["train_indices"], dtype=np.int64)
    val_np = np.asarray(split["val_indices"], dtype=np.int64)
    test_np = np.asarray(split["test_indices"], dtype=np.int64)
    train_idx = torch.from_numpy(train_np).to(train_device)
    val_idx = torch.from_numpy(val_np).to(train_device)
    test_idx = torch.from_numpy(test_np).to(train_device)
    model = torch.nn.Linear(x.shape[1], 1, bias=True).to(train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator(device=train_device).manual_seed(int(seed))
    best_key: tuple[float, float, float] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_val: dict[str, float] | None = None
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator, device=train_device)]
        batch_losses = []
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch]).squeeze(1)
            loss = criterion(logits, y[batch])
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        val_metrics = _metric_bundle(labels[val_np], _linear_logits(model, x[val_idx]))
        key = (val_metrics["auc"], -val_metrics["loss"], val_metrics["acc"])
        if np.isfinite(key[0]) and (best_key is None or key > best_key):
            best_key = key
            best_epoch = epoch
            best_val = val_metrics.copy()
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_acc": val_metrics["acc"],
            }
        )
    if best_state is None or best_val is None:
        raise RuntimeError(f"No valid validation AUC for {method}, seed={seed}, Q={q}, T={t}.")
    model.load_state_dict(best_state)
    model.to(train_device)
    model.eval()
    test_logits = _linear_logits(model, x[test_idx])
    test_metrics = _metric_bundle(labels[test_np], test_logits)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40.0, 40.0)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    checkpoint_dir = setting_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "torch.nn.Linear",
            "state_dict": best_state,
            "input_dim": int(x.shape[1]),
            "seed": int(seed),
            "method": method,
            "q": int(q),
            "t": int(t),
            "best_epoch": int(best_epoch),
            "selection_rule": SELECTION_RULE,
            "standardization": False,
            "weight_decay": 0.0,
            "train_device": str(train_device),
        },
        checkpoint_dir / f"{method}_seed{seed:03d}_best.pt",
    )
    metric_row = {
        "dataset": cfg.dataset,
        "method": method,
        "seed": int(seed),
        "q": int(q),
        "t": int(t),
        "token_dim": int(3 * q),
        "feature_dim": int(x.shape[1]),
        "epochs": int(cfg.epochs),
        "best_epoch": int(best_epoch),
        "val_auc": best_val["auc"],
        "val_loss": best_val["loss"],
        "val_acc": best_val["acc"],
        "test_auc": test_metrics["auc"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "selection_rule": SELECTION_RULE,
        "standardized": False,
        "weight_decay": 0.0,
        "train_device": str(train_device),
    }
    history_frame = pd.DataFrame(history)
    predictions_frame = pd.DataFrame(
        {
            "dataset": cfg.dataset,
            "method": method,
            "seed": seed,
            "q": q,
            "t": t,
            "sample_index": test_np,
            "y_true": labels[test_np],
            "probability": probabilities,
            "prediction": predictions,
        }
    )
    return metric_row, history_frame, predictions_frame


def _setting_signature(
    cfg: ExperimentConfig,
    q: int,
    t: int,
    dataset_fingerprint: str,
    token_schema_version: str = SCHEMA_VERSION,
) -> dict:
    return {
        "dataset": cfg.dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "token_variant": cfg.token_variant,
        "token_schema_version": str(token_schema_version),
        "q": int(q),
        "t": int(t),
        "token_dim": int(3 * q),
        "qrc": {"g": cfg.g, "W": cfg.W, "layers": cfg.layers, "readout": "Z+all_ZZ"},
        CRC3_Z_ZZ_METHOD: {
            "protocol": CRC3_Z_ZZ_PROTOCOL,
            "type": "classical_Q_by_3_state",
            "layers": cfg.layers,
            "input": "same_T_by_3Q_event_stream_as_QRC",
            "preprocessing": "tanh",
            "readout": "z+all_zz_direct",
            "feature_dim": int(_readout_dim(q)),
            "pca": False,
            "standardization": False,
            "label_or_split_dependent_feature_map": False,
        },
        ESN_STANDARD_METHOD: {
            "protocol": ESN_STANDARD_PROTOCOL,
            "type": "fixed_gaussian_echo_state_network",
            "layers": cfg.layers,
            "input": "same_T_by_3Q_event_stream_as_QRC",
            "preprocessing": "tanh",
            "state_and_feature_dim": int(_readout_dim(q)),
            "spectral_radius": 0.90,
            "input_scale": 0.75,
            "leak_rate": 0.65,
            "readout": "final_ESN_state_direct",
            "pca": False,
            "standardization": False,
            "label_or_split_dependent_feature_map": False,
        },
        "crc": {
            "type": "old_scalar_Q_state",
            "layers": cfg.layers,
            "state_scale": cfg.crc_state_scale,
            "inject_scale": cfg.crc_inject_scale,
            "readout": "state+all_pair_products",
            "compute_device": str(resolve_torch_device(cfg.crc_device)),
        },
        "preprocessing": "tanh inside QRC and CRC injection; none for raw_full; train-split PCA for raw_pca",
        "raw_full": "original T-by-3Q tokens directly flattened",
        "raw_pca": {
            "fit": "PCA fitted on train split only for each seed",
            "target_dim": int(_readout_dim(q)),
            "padding": "zero-pad only when PCA components < target_dim",
        },
        "classifier": "single torch.nn.Linear(d,1)",
        "classifier_train_device": str(resolve_torch_device(cfg.train_device)),
        "standardization": False,
        "regularization": False,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "seeds": list(cfg.seeds),
        "split": "80/10/10 stable scaffold; same seed paired across methods",
        "selection_rule": SELECTION_RULE,
    }


def _ensure_setting_config(path: Path, expected: dict, force: bool) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        # Backward-compatible, one-way migration for an additive baseline.
        # Every pre-existing scientific setting must be byte-for-byte equal
        # after removing only the newly introduced CRC3 contract.  This lets a
        # backfill reuse expensive QRC features without silently accepting any
        # other protocol drift.
        additive_keys = {CRC3_Z_ZZ_METHOD, ESN_STANDARD_METHOD}
        current_base = {key: value for key, value in current.items() if key not in additive_keys}
        expected_base = {key: value for key, value in expected.items() if key not in additive_keys}
        # Execution device is provenance, not a scientific hyperparameter.
        # CUDA availability can differ between resume shells while cached
        # numerical features and all training rules remain identical.
        current_base = json.loads(json.dumps(current_base, allow_nan=True))
        expected_base = json.loads(json.dumps(expected_base, allow_nan=True))
        current_base.pop("classifier_train_device", None)
        expected_base.pop("classifier_train_device", None)
        if isinstance(current_base.get("crc"), dict):
            current_base["crc"].pop("compute_device", None)
        if isinstance(expected_base.get("crc"), dict):
            expected_base["crc"].pop("compute_device", None)
        base_equal = json.dumps(current_base, sort_keys=True, allow_nan=True) == json.dumps(
            expected_base, sort_keys=True, allow_nan=True
        )
        present_additions_equal = all(
            key not in current
            or json.dumps(current[key], sort_keys=True, allow_nan=True)
            == json.dumps(expected[key], sort_keys=True, allow_nan=True)
            for key in additive_keys
        )
        additive_baseline_migration = base_equal and present_additions_equal
        if current != expected and not additive_baseline_migration and not force:
            changed = sorted(
                key for key in set(current) | set(expected)
                if json.dumps(current.get(key), sort_keys=True, allow_nan=True)
                != json.dumps(expected.get(key), sort_keys=True, allow_nan=True)
            )
            raise RuntimeError(
                f"Existing setting configuration differs at {path}. "
                f"Changed top-level keys={changed}. "
                "Use a new output folder or set force=True deliberately."
            )
    _json_dump(path, expected)


def _load_or_make_crc(
    tokens: np.ndarray,
    setting_dir: Path,
    q: int,
    t: int,
    seed: int,
    cfg: ExperimentConfig,
    token_schema_version: str = SCHEMA_VERSION,
) -> np.ndarray:
    path = setting_dir / "features" / f"crc_seed{seed:03d}.npz"
    output_dim = q + q * (q - 1) // 2
    if not cfg.force and path.exists() and _valid_feature_file(
        path, len(tokens), output_dim, q, t, seed, token_schema_version
    ):
        with np.load(path, allow_pickle=False) as data:
            return data["features"].astype(np.float32)
    features, params = compute_crc_features(
        tokens,
        q=q,
        layers=cfg.layers,
        seed=seed,
        state_scale=cfg.crc_state_scale,
        inject_scale=cfg.crc_inject_scale,
        device=cfg.crc_device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        q=np.int16(q),
        t=np.int16(t),
        seed=np.int32(seed),
        token_schema_version=np.asarray(str(token_schema_version)),
        layers=np.int16(cfg.layers),
        state_scale=np.float32(cfg.crc_state_scale),
        inject_scale=np.float32(cfg.crc_inject_scale),
        preprocessing=np.asarray("tanh"),
        readout=np.asarray("state+all_pair_products"),
        **params,
    )
    return features


def _valid_crc3_z_zz_file(
    path: Path,
    n: int,
    q: int,
    t: int,
    seed: int,
    layers: int,
    token_schema_version: str,
) -> bool:
    """Strictly validate the no-PCA CRC3 artifact before resuming it."""
    try:
        with np.load(path, allow_pickle=False) as data:
            features = data["features"]
            return bool(
                features.shape == (n, _readout_dim(q))
                and np.isfinite(features).all()
                and int(data["q"]) == int(q)
                and int(data["t"]) == int(t)
                and int(data["seed"]) == int(seed)
                and int(data["layers"]) == int(layers)
                and str(data["token_schema_version"].item()) == str(token_schema_version)
                and str(data["baseline_protocol"].item()) == CRC3_Z_ZZ_PROTOCOL
                and str(data["internal_state"].item()) == "Qx3_xyz"
                and str(data["readout"].item()) == "z+all_zz_direct"
                and int(data["axis_index"]) == 2
                and not bool(data["pca"])
            )
    except Exception:
        return False


def _load_or_make_crc3_z_zz_matched(
    tokens: np.ndarray,
    setting_dir: Path,
    q: int,
    t: int,
    seed: int,
    cfg: ExperimentConfig,
    token_schema_version: str = SCHEMA_VERSION,
) -> np.ndarray:
    """Q-by-3 CRC state with direct z/zz output and no PCA."""
    path = setting_dir / "features" / f"{CRC3_Z_ZZ_METHOD}_seed{seed:03d}.npz"
    if not cfg.force and path.exists() and _valid_crc3_z_zz_file(
        path,
        len(tokens),
        q,
        t,
        seed,
        cfg.layers,
        token_schema_version,
    ):
        with np.load(path, allow_pickle=False) as data:
            return data["features"].astype(np.float32)

    features, params = compute_crc3_z_zz_features(
        tokens,
        q=q,
        layers=cfg.layers,
        seed=seed,
        device=cfg.crc_device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        q=np.int16(q),
        t=np.int16(t),
        seed=np.int32(seed),
        token_schema_version=np.asarray(str(token_schema_version)),
        baseline_protocol=np.asarray(CRC3_Z_ZZ_PROTOCOL),
        layers=np.int16(cfg.layers),
        preprocessing=np.asarray("tanh"),
        internal_state=np.asarray("Qx3_xyz"),
        axis=np.asarray("z"),
        axis_index=np.int8(2),
        readout=np.asarray("z+all_zz_direct"),
        feature_dim=np.int32(_readout_dim(q)),
        pca=np.bool_(False),
        standardization=np.bool_(False),
        label_or_split_dependent_feature_map=np.bool_(False),
        **params,
    )
    return features.astype(np.float32, copy=False)


def _load_or_make_esn_standard(
    tokens: np.ndarray,
    setting_dir: Path,
    q: int,
    t: int,
    seed: int,
    cfg: ExperimentConfig,
    token_schema_version: str = SCHEMA_VERSION,
) -> np.ndarray:
    """Standard dimension-matched ESN; direct state output and no PCA."""
    path = setting_dir / "features" / f"{ESN_STANDARD_METHOD}_seed{seed:03d}.npz"
    if not cfg.force and path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                candidate = data["features"].astype(np.float32)
                valid = bool(
                    candidate.shape == (len(tokens), _readout_dim(q))
                    and np.isfinite(candidate).all()
                    and int(data["q"]) == q
                    and int(data["t"]) == t
                    and int(data["seed"]) == seed
                    and int(data["layers"]) == cfg.layers
                    and str(data["token_schema_version"].item()) == str(token_schema_version)
                    and str(data["baseline_protocol"].item()) == ESN_STANDARD_PROTOCOL
                    and str(data["readout"].item()) == "final_ESN_state_direct"
                    and not bool(data["pca"])
                )
            if valid:
                return candidate
        except Exception:
            pass
    features, params = compute_esn_standard_features(
        tokens,
        q=q,
        layers=cfg.layers,
        seed=seed,
        device=cfg.crc_device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        q=np.int16(q),
        t=np.int16(t),
        seed=np.int32(seed),
        layers=np.int16(cfg.layers),
        token_schema_version=np.asarray(str(token_schema_version)),
        baseline_protocol=np.asarray(ESN_STANDARD_PROTOCOL),
        preprocessing=np.asarray("tanh"),
        state_dim=np.int32(_readout_dim(q)),
        readout=np.asarray("final_ESN_state_direct"),
        pca=np.bool_(False),
        standardization=np.bool_(False),
        label_or_split_dependent_feature_map=np.bool_(False),
        **params,
    )
    return features.astype(np.float32, copy=False)


def _load_or_make_raw_pca(
    raw_features: np.ndarray,
    split: dict,
    setting_dir: Path,
    q: int,
    t: int,
    seed: int,
    cfg: ExperimentConfig,
    token_schema_version: str = SCHEMA_VERSION,
) -> np.ndarray:
    """Train-split PCA compression of raw-full features to QRC readout width."""
    target_dim = _readout_dim(q)
    path = setting_dir / "features" / f"raw_pca_seed{seed:03d}.npz"
    if not cfg.force and path.exists() and _valid_feature_file(
        path, len(raw_features), target_dim, q, t, seed, token_schema_version
    ):
        with np.load(path, allow_pickle=False) as data:
            return data["features"].astype(np.float32)

    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    train_features = np.ascontiguousarray(raw_features[train_idx], dtype=np.float32)
    n_components = int(min(target_dim, raw_features.shape[1], len(train_idx)))
    if n_components < 1:
        raise RuntimeError(f"Cannot fit raw_pca for seed={seed}, Q={q}, T={t}: empty training split.")
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(train_features)
    projected = pca.transform(np.ascontiguousarray(raw_features, dtype=np.float32)).astype(np.float32)
    if n_components < target_dim:
        padded = np.zeros((len(raw_features), target_dim), dtype=np.float32)
        padded[:, :n_components] = projected
        projected = padded

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=projected.astype(np.float32),
        q=np.int16(q),
        t=np.int16(t),
        seed=np.int32(seed),
        token_schema_version=np.asarray(str(token_schema_version)),
        target_dim=np.int16(target_dim),
        n_components=np.int16(n_components),
        raw_dim=np.int32(raw_features.shape[1]),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        preprocessing=np.asarray("train_split_pca_then_zero_pad"),
        readout=np.asarray("raw_flatten_pca_aligned_to_Z+all_ZZ_dim"),
    )
    return projected.astype(np.float32)


def _load_qrc(
    setting_dir: Path,
    q: int,
    t: int,
    seed: int,
    n: int,
    token_schema_version: str = SCHEMA_VERSION,
) -> np.ndarray:
    path = setting_dir / "features" / f"qrc_seed{seed:03d}.npz"
    output_dim = _readout_dim(q)
    if not path.exists() or not _valid_feature_file(
        path, n, output_dim, q, t, seed, token_schema_version
    ):
        raise FileNotFoundError(f"Missing or incompatible QRC features: {path}")
    with np.load(path, allow_pickle=False) as data:
        return data["features"].astype(np.float32)


def _summarize_setting(setting_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method, subset in metrics.groupby("method", sort=False):
        row = {"method": method, "n_seeds": int(len(subset))}
        for metric in ("val_auc", "val_loss", "val_acc", "test_auc", "test_loss", "test_acc"):
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(setting_dir / "performance_summary.csv", index=False)
    pivot = metrics.pivot(index="seed", columns="method", values=["test_auc", "test_acc"])
    paired = pd.DataFrame(index=pivot.index)
    for metric in ("test_auc", "test_acc"):
        if (metric, "qrc") in pivot and (metric, CRC3_Z_ZZ_METHOD) in pivot:
            paired[f"qrc_minus_{CRC3_Z_ZZ_METHOD}_{metric}"] = (
                pivot[(metric, "qrc")] - pivot[(metric, CRC3_Z_ZZ_METHOD)]
            )
        if (metric, "qrc") in pivot and (metric, ESN_STANDARD_METHOD) in pivot:
            paired[f"qrc_minus_{ESN_STANDARD_METHOD}_{metric}"] = (
                pivot[(metric, "qrc")] - pivot[(metric, ESN_STANDARD_METHOD)]
            )
        if (metric, "qrc") in pivot and (metric, "crc") in pivot:
            paired[f"qrc_minus_crc_{metric}"] = pivot[(metric, "qrc")] - pivot[(metric, "crc")]
        if (metric, "qrc") in pivot and (metric, "raw_full") in pivot:
            paired[f"qrc_minus_raw_full_{metric}"] = pivot[(metric, "qrc")] - pivot[(metric, "raw_full")]
        if (metric, "qrc") in pivot and (metric, "raw_pca") in pivot:
            paired[f"qrc_minus_raw_pca_{metric}"] = pivot[(metric, "qrc")] - pivot[(metric, "raw_pca")]
    paired.reset_index().to_csv(setting_dir / "paired_differences.csv", index=False)
    return summary


def run_setting(
    cache_path: Path,
    dataset_metadata: dict,
    setting_dir: Path,
    q: int,
    t: int,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    if not 1 <= q <= MAX_Q:
        raise ValueError(f"Q must lie in [1,{MAX_Q}], got {q}")
    if not 1 <= t <= cfg.max_cached_events:
        raise ValueError(f"T must lie in [1,{cfg.max_cached_events}], got {t}")
    setting_dir.mkdir(parents=True, exist_ok=True)
    token_schema_version = str(dataset_metadata.get("schema_version", SCHEMA_VERSION))
    expected = _setting_signature(
        cfg,
        q,
        t,
        dataset_metadata["fingerprint"],
        token_schema_version=token_schema_version,
    )
    _ensure_setting_config(setting_dir / "setting_config.json", expected, cfg.force)
    tokens_all, labels, smiles = _load_cache(cache_path)
    tokens = np.ascontiguousarray(tokens_all[:, :t, : 3 * q], dtype=np.float32)
    _json_dump(
        setting_dir / "input_manifest.json",
        {
            "shared_token_cache": str(cache_path.resolve()),
            "slice": f"tokens[:, :{t}, :{3*q}]",
            "shape": list(tokens.shape),
            "labels_shape": list(labels.shape),
        },
    )
    for seed in cfg.seeds:
        split = _split_payload(smiles, labels, cfg, seed)
        _json_dump(setting_dir / "splits" / f"seed{seed:03d}.json", split)
    if cfg.run_qrc:
        extract_qrc_features_dual_gpu(
            cache_path,
            setting_dir,
            q,
            t,
            cfg,
            token_schema_version=token_schema_version,
        )
    raw_path = setting_dir / "features" / "raw_full_features.npz"
    raw_dim = int(t * 3 * q)
    if cfg.force or not _valid_raw_full_file(
        raw_path, len(tokens), raw_dim, q, t, token_schema_version
    ):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_features = tokens.reshape(len(tokens), -1).astype(np.float32)
        np.savez_compressed(
            raw_path,
            features=raw_features,
            q=np.int16(q),
            t=np.int16(t),
            token_schema_version=np.asarray(token_schema_version),
            preprocessing=np.asarray("none"),
            readout=np.asarray("direct_flatten"),
        )
    else:
        with np.load(raw_path, allow_pickle=False) as data:
            raw_features = data["features"].astype(np.float32)

    run_dir = setting_dir / "runs"
    history_dir = setting_dir / "histories"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in cfg.seeds:
        split = json.loads((setting_dir / "splits" / f"seed{seed:03d}.json").read_text())
        crc_features = _load_or_make_crc(
            tokens,
            setting_dir,
            q,
            t,
            seed,
            cfg,
            token_schema_version=token_schema_version,
        )
        crc3_z_zz_features = _load_or_make_crc3_z_zz_matched(
            tokens,
            setting_dir,
            q,
            t,
            seed,
            cfg,
            token_schema_version=token_schema_version,
        )
        esn_standard_features = _load_or_make_esn_standard(
            tokens,
            setting_dir,
            q,
            t,
            seed,
            cfg,
            token_schema_version=token_schema_version,
        )
        raw_pca_features = _load_or_make_raw_pca(
            raw_features,
            split,
            setting_dir,
            q,
            t,
            seed,
            cfg,
            token_schema_version=token_schema_version,
        )
        features_by_method = {
            CRC3_Z_ZZ_METHOD: crc3_z_zz_features,
            ESN_STANDARD_METHOD: esn_standard_features,
            "crc": crc_features,
            "raw_full": raw_features,
            "raw_pca": raw_pca_features,
        }
        if cfg.run_qrc:
            features_by_method = {
                "qrc": _load_qrc(
                    setting_dir,
                    q,
                    t,
                    seed,
                    len(tokens),
                    token_schema_version=token_schema_version,
                ),
                **features_by_method,
            }
        for method, features in features_by_method.items():
            metric_path = run_dir / f"{method}_seed{seed:03d}_metrics.json"
            history_path = history_dir / f"{method}_seed{seed:03d}.csv"
            prediction_path = run_dir / f"{method}_seed{seed:03d}_test_predictions.csv"
            checkpoint_path = setting_dir / "checkpoints" / f"{method}_seed{seed:03d}_best.pt"
            if (
                not cfg.force
                and metric_path.exists()
                and history_path.exists()
                and prediction_path.exists()
                and checkpoint_path.exists()
            ):
                metric_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
                prediction_frames.append(pd.read_csv(prediction_path))
                continue
            row, history, predictions = train_linear_probe(
                features, labels, split, method, seed, setting_dir, cfg, q, t
            )
            _json_dump(metric_path, row)
            history.to_csv(history_path, index=False)
            predictions.to_csv(prediction_path, index=False)
            metric_rows.append(row)
            prediction_frames.append(predictions)
    metrics = pd.DataFrame(metric_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    metrics.to_csv(setting_dir / "metrics_by_seed.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(setting_dir / "test_predictions.csv", index=False)
    _summarize_setting(setting_dir, metrics)
    return metrics


def prepare_q_equal_t_layout(scan_root: str | Path) -> list[tuple[int, int, Path]]:
    root = Path(scan_root)
    settings = []
    for value in range(1, 21):
        path = root / f"Q{value:02d}_T{value:02d}"
        path.mkdir(parents=True, exist_ok=True)
        settings.append((value, value, path))
    return settings


def prepare_fixed_q_layout(scan_root: str | Path) -> list[tuple[int, int, Path]]:
    root = Path(scan_root)
    settings = []
    for q in (8, 12, 16, 20):
        for t in range(1, 31):
            path = root / f"Q{q:02d}" / f"T{t:02d}"
            path.mkdir(parents=True, exist_ok=True)
            settings.append((q, t, path))
    return settings


def run_scan(cfg: ExperimentConfig, settings: Iterable[tuple[int, int, Path]]) -> pd.DataFrame:
    cfg = cfg.normalized()
    scan_root = Path(cfg.scan_root)
    cache_path, metadata = load_or_build_token_cache(
        cfg.dataset,
        cfg.data_dir,
        scan_root.parent / "cache",
        max_events=cfg.max_cached_events,
        token_variant=cfg.token_variant,
    )
    _json_dump(scan_root / "scan_config.json", asdict(cfg))
    for index, (q, t, setting_dir) in enumerate(settings, start=1):
        print(f"\n[{cfg.dataset}] setting {index}: Q={q}, T={t}, output={setting_dir}", flush=True)
        run_setting(cache_path, metadata, Path(setting_dir), q, t, cfg)
    return collect_scan_results(scan_root)


def collect_scan_results(scan_root: str | Path) -> pd.DataFrame:
    rows = []
    root = Path(scan_root)
    for path in sorted(root.rglob("performance_summary.csv")):
        config_path = path.parent / "setting_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        table = pd.read_csv(path)
        table.insert(0, "q", int(config["q"]))
        table.insert(1, "t", int(config["t"]))
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True).sort_values(["q", "t", "method"])
    combined.to_csv(root / "scan_performance_summary.csv", index=False)
    return combined


def collect_w_scan_results(scan_root: str | Path) -> pd.DataFrame:
    rows = []
    root = Path(scan_root)
    for path in sorted(root.rglob("performance_summary.csv")):
        config_path = path.parent / "setting_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        table = pd.read_csv(path)
        table.insert(0, "q", int(config["q"]))
        table.insert(1, "t", int(config["t"]))
        table.insert(2, "W", float(config["qrc"]["W"]))
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True).sort_values(["q", "t", "W", "method"])
    combined.to_csv(root / "scan_W_performance_summary.csv", index=False)
    return combined


def _figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}
    )


def plot_single_setting_summary(setting_dir: str | Path, dataset: str | None = None) -> plt.Figure:
    """Plot the per-setting QRC/CRC3/CRC/raw-full/raw-PCA summary.

    This is intentionally small and local: every minimum experiment folder can
    run its own notebook/script and immediately inspect the setting-level
    performance without waiting for the whole scan to finish.
    """
    setting_dir = Path(setting_dir)
    summary_path = setting_dir / "performance_summary.csv"
    config_path = setting_dir / "setting_config.json"
    if not summary_path.exists():
        raise RuntimeError(f"No performance_summary.csv found under {setting_dir}")
    if not config_path.exists():
        raise RuntimeError(f"No setting_config.json found under {setting_dir}")
    summary = pd.read_csv(summary_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_label = (dataset or config.get("dataset", "dataset")).upper()
    q, t = int(config["q"]), int(config["t"])
    methods = [method for method in METHODS if method in set(summary["method"])]
    labels = [METHOD_LABELS[method] for method in methods]

    _figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 62 / 25.4), constrained_layout=True)
    x = np.arange(len(methods), dtype=float)
    for ax, metric, ylabel in zip(axes, ("test_auc", "test_acc"), ("Test ROC-AUC", "Test accuracy")):
        means = np.asarray(
            [summary.loc[summary.method == method, f"{metric}_mean"].iloc[0] for method in methods],
            dtype=float,
        )
        stds = np.asarray(
            [summary.loc[summary.method == method, f"{metric}_std"].iloc[0] for method in methods],
            dtype=float,
        )
        colors = [METHOD_COLORS[method] for method in methods]
        ax.bar(x, means, yerr=stds, color=colors, width=0.66, capsize=2.5, linewidth=0)
        ax.axhline(0.5, color="#B8B8B8", lw=0.7, ls="--", zorder=0)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, loc="left", fontweight="bold")
    fig.suptitle(f"{dataset_label} general BFS single setting: Q={q}, T={t}", x=0.01, ha="left", fontweight="bold")
    _save_figure(fig, setting_dir / "figures" / f"{dataset_label}_general_Q{q:02d}_T{t:02d}_single_setting")
    _json_dump(
        setting_dir / "figures" / "single_setting_figure_contract.json",
        {
            "core_conclusion": "Compare QRC with a no-PCA Q-by-3 CRC3 z/zz match, the old scalar-state CRC, direct flattened tokens, and train-split PCA-compressed tokens at this exact Q/T setting.",
            "evidence": "Mean and standard deviation of test ROC-AUC and accuracy across ten paired seeded scaffold splits.",
            "setting": {"dataset": dataset_label, "q": q, "t": t, "token_dim": int(3 * q)},
            "source_data": "performance_summary.csv",
            "selection_rule": SELECTION_RULE,
        },
    )
    return fig


def run_setting_plan(
    plan_path: str | Path,
    run_qrc: bool | None = None,
    force: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one local minimum experiment folder from its ``setting_plan.json``.

    The generated per-folder notebooks and scripts use this helper so that each
    minimum folder is a self-contained runnable unit, while the numerical QRC,
    CRC, tokenization, split and linear-readout implementations remain shared
    and version-consistent.
    """
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    setting_dir = Path(plan.get("setting_dir", plan_path.parent)).resolve()
    scan_root = Path(plan["scan_root"]).resolve()
    data_dir = Path(plan["data_dir"]).resolve()
    cfg = ExperimentConfig(
        dataset=str(plan["dataset"]).upper(),
        data_dir=str(data_dir),
        scan_root=str(scan_root),
        token_variant=str(plan.get("token_variant", TOKEN_VARIANT_GENERAL)),
        epochs=int(plan.get("epochs", 500)),
        seeds=tuple(int(seed) for seed in plan.get("seeds", DEFAULT_SEEDS)),
        W=float(plan.get("W", 0.4)),
        g=float(plan.get("g", 0.84)),
        layers=int(plan.get("layers", 3)),
        qrc_devices=tuple(plan.get("qrc_devices", ("cuda:0", "cuda:1"))),
        crc_device=str(plan.get("crc_device", "auto")),
        train_device=str(plan.get("train_device", "auto")),
        run_qrc=bool(plan.get("run_qrc", True)) if run_qrc is None else bool(run_qrc),
        force=bool(plan.get("force", False)) if force is None else bool(force),
    ).normalized()
    cache_path, metadata = load_or_build_token_cache(
        cfg.dataset,
        cfg.data_dir,
        scan_root.parent / "cache",
        max_events=cfg.max_cached_events,
        token_variant=cfg.token_variant,
    )
    metrics = run_setting(
        cache_path=cache_path,
        dataset_metadata=metadata,
        setting_dir=setting_dir,
        q=int(plan["q"]),
        t=int(plan["t"]),
        cfg=cfg,
    )
    summary = pd.read_csv(setting_dir / "performance_summary.csv")
    plot_single_setting_summary(setting_dir, cfg.dataset)
    return metrics, summary


SCAN_METRIC_YLIMS = {
    "test_auc": (0.45, 0.90),
    "test_acc": (0.45, 0.90),
}


def _plot_method_lines(
    ax: plt.Axes,
    table: pd.DataFrame,
    x_column: str,
    metric: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    visible_lows: list[float] = []
    visible_highs: list[float] = []
    for method in METHODS:
        subset = table[table.method == method].sort_values(x_column)
        if subset.empty:
            continue
        x = subset[x_column].to_numpy()
        mean = subset[f"{metric}_mean"].to_numpy()
        std = subset[f"{metric}_std"].to_numpy()
        visible_lows.extend((mean - std).tolist())
        visible_highs.extend((mean + std).tolist())
        ax.plot(x, mean, color=METHOD_COLORS[method], marker="o", markersize=2.5, lw=1.2, label=METHOD_LABELS[method])
        ax.fill_between(x, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.14, linewidth=0)
    ax.axhline(0.5, color="#B8B8B8", lw=0.7, ls="--", zorder=0)
    lower, upper = ylim or (0.45, 1.0)
    if visible_lows and visible_highs:
        lower = min(lower, float(np.nanmin(visible_lows)) - 0.015)
        upper = max(upper, float(np.nanmax(visible_highs)) + 0.015)
    ax.set_ylim(max(0.0, lower), min(1.0, upper))


def plot_q_equal_t_scan(scan_root: str | Path, dataset: str) -> plt.Figure:
    root = Path(scan_root)
    table = collect_scan_results(root)
    if table.empty:
        raise RuntimeError(f"No completed settings under {root}")
    _figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 67 / 25.4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("test_auc", "test_acc"), ("Test ROC-AUC", "Test accuracy")):
        _plot_method_lines(ax, table, "q", metric, ylim=SCAN_METRIC_YLIMS.get(metric))
        ax.set_xlabel("Q = T  (token dimension = 3Q)")
        ax.set_ylabel(title)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks([1, 4, 8, 12, 16, 20])
    axes[0].legend(ncol=4, loc="lower right")
    fig.suptitle(f"{dataset.upper()} general BFS: synchronous Q=T scan", x=0.01, ha="left", fontweight="bold")
    _save_figure(fig, root / "figures" / f"{dataset.upper()}_general_Q_equal_T")
    _json_dump(
        root / "figures" / "figure_contract.json",
        {
            "core_conclusion": "Compare QRC, direct no-PCA CRC3-z/zz, scalar CRC, raw-full tokens, and raw-PCA tokens as Q and T grow together.",
            "evidence": "Mean test ROC-AUC and accuracy across ten paired seeded scaffold splits; shading is one standard deviation.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "source_data": "scan_performance_summary.csv",
            "review_risks": [
                "Q=T simultaneously changes token width, event count, reservoir size, and readout dimension.",
                "Raw-full feature dimension T*3Q is intentionally not matched to the reservoir readout dimension.",
                "Raw-PCA is fitted only on each train split and is dimension-aligned to the reservoir readout when possible.",
                "Epoch selection uses validation data only; reported test variability is across paired seeds.",
            ],
        },
    )
    return fig


def plot_fixed_q_scan(scan_root: str | Path, dataset: str) -> plt.Figure:
    root = Path(scan_root)
    table = collect_scan_results(root)
    if table.empty:
        raise RuntimeError(f"No completed settings under {root}")
    _figure_style()
    q_values = (8, 12, 16, 20)
    fig, axes = plt.subplots(2, 4, figsize=(183 / 25.4, 112 / 25.4), sharex=True, sharey="row", constrained_layout=True)
    for column, q in enumerate(q_values):
        subset = table[table.q == q]
        for row, (metric, label) in enumerate((("test_auc", "Test ROC-AUC"), ("test_acc", "Test accuracy"))):
            ax = axes[row, column]
            _plot_method_lines(ax, subset, "t", metric, ylim=SCAN_METRIC_YLIMS.get(metric))
            ax.set_title(f"Q = {q}", loc="left", fontweight="bold")
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
            if column == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Events, T")
    axes[0, 0].legend(ncol=1, loc="lower right")
    fig.suptitle(f"{dataset.upper()} general BFS: fixed-Q event-memory scan", x=0.01, ha="left", fontweight="bold")
    _save_figure(fig, root / "figures" / f"{dataset.upper()}_general_fixed_Q_T")
    _json_dump(
        root / "figures" / "figure_contract.json",
        {
            "core_conclusion": "At fixed reservoir width Q, identify how QRC, direct no-PCA CRC3-z/zz, scalar CRC, raw-full, and raw-PCA performance changes with event-stream length T.",
            "evidence": "Mean test ROC-AUC and accuracy across ten paired seeded scaffold splits; shading is one standard deviation.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "source_data": "scan_performance_summary.csv",
            "review_risks": [
                "T changes how much molecular information is exposed while Q and token width remain fixed.",
                "The raw-full baseline grows with T, whereas QRC/CRC3/CRC/raw-PCA final readout dimension stays Q(Q+1)/2 when raw rank allows it.",
                "Epoch selection uses validation data only; no test metric is optimized.",
            ],
        },
    )
    return fig


def plot_w_scan(scan_root: str | Path, dataset: str) -> plt.Figure:
    root = Path(scan_root)
    table = collect_w_scan_results(root)
    if table.empty:
        raise RuntimeError(f"No completed W-scan settings under {root}")
    _figure_style()
    settings = (
        table[["q", "t"]]
        .drop_duplicates()
        .sort_values(["q", "t"])
        .itertuples(index=False, name=None)
    )
    settings = list(settings)
    fig, axes = plt.subplots(
        2,
        len(settings),
        figsize=(183 / 25.4, 92 / 25.4),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    for column, (q, t) in enumerate(settings):
        subset = table[(table.q == q) & (table.t == t)]
        for row, (metric, label) in enumerate((("test_auc", "Test ROC-AUC"), ("test_acc", "Test accuracy"))):
            ax = axes[row, column]
            _plot_method_lines(ax, subset, "W", metric, ylim=SCAN_METRIC_YLIMS.get(metric))
            ax.set_xlim(-0.025, 0.925)
            ax.set_xticks([0.0, 0.3, 0.6, 0.9])
            ax.set_title(f"Q = {q}, T = {t}", loc="left", fontweight="bold")
            if column == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Reservoir coupling scale, W")
    axes[0, 0].legend(ncol=1, loc="lower right")
    fig.suptitle(f"{dataset.upper()} general BFS: W scan at fixed Q=T", x=0.01, ha="left", fontweight="bold")
    _save_figure(fig, root / "figures" / f"{dataset.upper()}_general_W_scan")
    _json_dump(
        root / "figures" / "figure_contract.json",
        {
            "core_conclusion": "At fixed Q=T, test whether reservoir coupling strength W changes QRC/CRC linear-readout performance.",
            "evidence": "Mean test ROC-AUC and accuracy across ten paired seeded scaffold splits; shading is one standard deviation.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "source_data": "scan_W_performance_summary.csv",
            "review_risks": [
                "Only W changes within each Q=T column; tokens, splits, g, layers, readout and classifier stay fixed.",
                "Raw-full and raw-PCA baselines are W-independent controls and may appear nearly horizontal apart from training stochasticity.",
                "Epoch selection uses validation data only; no test metric is optimized.",
            ],
        },
    )
    return fig
