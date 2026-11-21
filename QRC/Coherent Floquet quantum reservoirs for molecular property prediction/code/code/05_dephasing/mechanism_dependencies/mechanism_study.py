#!/usr/bin/env python3
"""Mechanism and quantum-necessity controls for FINAL graph-event QRC.

This module deliberately keeps the molecular input fixed while reservoir size is
changed.  Only the first ``input_qubits`` receive the same 3-coordinate token;
all additional qubits receive U3(0, 0, 0) and therefore act as ancillas.  It also
provides a compact W-g-L map and controls which remove interactions, disorder,
coherence, or entanglement.

The executable defaults are conservative.  ``--mode quick`` is a pipeline check;
``--mode full`` is the publication plan.  W=0.6 is canonical everywhere except
the explicit phase-map section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
LINEAR_ROOT = FINAL / "2_linear_readout"
sys.path.insert(0, str(LINEAR_ROOT))

from core.gpu_qrc_statevector import (  # noqa: E402
    DTCQRCConfig,
    _basis_z_signs,
    apply_reservoir_layers,
    apply_token_injection,
    build_dtc_diagonal_phase,
    build_observable_signs,
    compute_z_zz_from_state,
    evolve_batch_final_state,
    generate_dtc_params,
    real_dtype_from_complex,
    rx_gate,
    set_seed,
    torch_complex_dtype,
)
from core.crc3_baseline import (  # noqa: E402
    CRC3_Z_ZZ_METHOD,
    CRC3_Z_ZZ_PROTOCOL,
    compute_crc3_z_zz_features,
)
from core.linear_readout import (  # noqa: E402
    load_or_build_token_cache,
    scaffold_split_indices,
)


SEEDS = tuple(range(0, 100, 10))
CANONICAL_W = 0.6
CANONICAL_G = 0.84
CANONICAL_L = 3
CRC3_PROBE_PROTOCOL = "train_only_standardized_logistic_Cgrid_v1"
ABLATIONS = (
    "full",
    "J0",
    "no_disorder",
    "event_dephase",
    "mean_field",
    "clifford_reservoir_proxy",
)


@dataclass(frozen=True)
class StudyPoint:
    dataset: str
    section: str
    total_qubits: int
    input_qubits: int
    events: int
    W: float = CANONICAL_W
    g: float = CANONICAL_G
    layers: int = CANONICAL_L
    variant: str = "full"

    @property
    def key(self) -> str:
        return (
            f"Q{self.total_qubits:02d}_Qin{self.input_qubits:02d}_T{self.events:02d}"
            f"_W{self.W:.2f}_g{self.g:.2f}_L{self.layers}_{self.variant}"
        ).replace(".", "p")


def data_root(dataset: str) -> Path:
    return LINEAR_ROOT / "general" / dataset.upper() / "data"


def load_dataset(
    dataset: str,
    max_events: int = 30,
    token_schema: str = "final_v2",
    token_cache: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replaceable token-loader boundary shared by all mechanism sections."""
    dataset = dataset.upper()
    if token_cache is not None:
        cache = Path(token_cache)
    elif token_schema == "final_v2":
        cache, _ = load_or_build_token_cache(
            dataset, data_root(dataset), HERE / "cache", max_events=max_events,
            token_variant="general",
        )
    elif token_schema == "canonical_v3":
        validity_root = FINAL / "5_graph_stream_validity"
        sys.path.insert(0, str(validity_root))
        from validity_core.canonical_tokens import load_or_build_canonical_token_cache

        cache, _ = load_or_build_canonical_token_cache(
            dataset, data_root(dataset), HERE / "cache", max_events=max_events
        )
    else:
        raise ValueError(f"Unsupported token_schema={token_schema}")
    if not cache.exists():
        raise FileNotFoundError(cache)
    with np.load(cache, allow_pickle=False) as z:
        return z["tokens"].astype(np.float32), z["labels"].astype(np.int64), z["smiles"].astype(str)


def fixed_information_tokens(tokens: np.ndarray, input_qubits: int, total_qubits: int, events: int) -> np.ndarray:
    """Copy a fixed 3*Qin prefix and zero-pad ancilla injection coordinates."""
    if input_qubits > total_qubits:
        raise ValueError("input_qubits cannot exceed total_qubits")
    source = np.ascontiguousarray(tokens[:, :events, : 3 * input_qubits], dtype=np.float32)
    out = np.zeros((len(source), events, 3 * total_qubits), dtype=np.float32)
    out[:, :, : source.shape[-1]] = source
    return out


def _variant_params(point: StudyPoint, seed: int) -> tuple[np.ndarray, np.ndarray, DTCQRCConfig]:
    params = generate_dtc_params(point.total_qubits, seed=seed, W_scale=point.W)
    J, h = params["J"].copy(), params["h"].copy()
    g = point.g
    if point.variant == "J0":
        J[:] = 0.0
    elif point.variant == "no_disorder":
        h[:] = 0.0
    elif point.variant == "clifford_reservoir_proxy":
        # RZZ(pi/2) and RX(pi/2) are Clifford reservoir gates.  Continuous U3
        # input encoding is intentionally retained, so this is a reservoir-only
        # Clifford proxy rather than a classically simulable full Clifford circuit.
        J = np.sign(J) * (math.pi / 2.0)
        h = np.round(2.0 * h) / 2.0
        g = 0.5
    cfg = DTCQRCConfig(
        num_qubits=point.total_qubits,
        features_per_qubit=3,
        g=g,
        W_scale=point.W,
        layers_per_injection=point.layers,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    return J, h, cfg


@torch.no_grad()
def _evolve_with_event_dephase(
    tokens: torch.Tensor,
    cfg: DTCQRCConfig,
    J: torch.Tensor,
    h: torch.Tensor,
    device: torch.device,
    seed: int,
    strength: float = 1.0,
) -> torch.Tensor:
    """One quantum-trajectory realization of Z dephasing after every event.

    A Bernoulli Z flip with probability strength/2 is an unbiased trajectory
    representation of the phase-flip/dephasing channel.  Several trajectories
    are averaged by :func:`compute_features`.
    """
    dtype = torch_complex_dtype(cfg.complex_dtype)
    real = real_dtype_from_complex(dtype)
    x = torch.tanh(tokens.to(device=device, dtype=real))
    state = torch.zeros((len(x), 1 << cfg.num_qubits), dtype=dtype, device=device)
    state[:, 0] = 1.0
    rx = rx_gate(cfg.g * math.pi, device=device, complex_dtype=dtype)
    phase = build_dtc_diagonal_phase(cfg.num_qubits, J, h, device, dtype)
    zsign = _basis_z_signs(cfg.num_qubits, device=device, dtype=real)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    for time in range(x.shape[1]):
        state = apply_token_injection(state, x[:, time], cfg, dtype)
        state = apply_reservoir_layers(state, cfg, rx, phase)
        flips = torch.rand((len(x), cfg.num_qubits), generator=gen, device=device) < (strength / 2.0)
        for wire in range(cfg.num_qubits):
            state = torch.where(flips[:, wire, None], state * zsign[:, wire][None, :], state)
    return state


@torch.no_grad()
def compute_mean_field_features(tokens: np.ndarray, point: StudyPoint, seed: int, device: str) -> np.ndarray:
    """Product-state time-dependent mean-field approximation to the QRC.

    Each qubit retains a two-component pure state.  ZZ interactions are replaced
    by local RZ fields J_ij <Z_j>, so no entanglement can be generated.
    """
    dev = torch.device(device if torch.cuda.is_available() or "cuda" not in device else "cpu")
    J_np, h_np, cfg = _variant_params(point, seed)
    J = torch.tensor(J_np, device=dev)
    h = torch.tensor(h_np, device=dev)
    x = torch.tanh(torch.tensor(tokens, dtype=torch.float32, device=dev)).reshape(len(tokens), -1, cfg.num_qubits, 3)
    psi = torch.zeros((len(tokens), cfg.num_qubits, 2), dtype=torch.complex64, device=dev)
    psi[:, :, 0] = 1.0

    def local_u3(v: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        theta, phi, lam = angles.unbind(-1)
        c, s = torch.cos(theta / 2), torch.sin(theta / 2)
        a, b = v[..., 0], v[..., 1]
        out0 = c * a - torch.exp(1j * lam) * s * b
        out1 = torch.exp(1j * phi) * s * a + torch.exp(1j * (phi + lam)) * c * b
        return torch.stack((out0, out1), dim=-1)

    for time in range(x.shape[1]):
        psi = local_u3(psi, x[:, time])
        for _ in range(cfg.layers_per_injection):
            # RX(g*pi)
            a = cfg.g * math.pi / 2
            p0, p1 = psi[..., 0], psi[..., 1]
            psi = torch.stack((math.cos(a) * p0 - 1j * math.sin(a) * p1,
                               -1j * math.sin(a) * p0 + math.cos(a) * p1), dim=-1)
            z = psi.abs().square()[..., 0] - psi.abs().square()[..., 1]
            angle = math.pi * h[None, :].expand_as(z)
            if cfg.num_qubits > 1:
                angle[:, :-1] += J[None, :] * z[:, 1:]
                angle[:, 1:] += J[None, :] * z[:, :-1]
            psi[..., 0] *= torch.exp(-0.5j * angle)
            psi[..., 1] *= torch.exp(0.5j * angle)
    z = (psi.abs().square()[..., 0] - psi.abs().square()[..., 1]).float()
    cols = [z] + [z[:, i:i+1] * z[:, j:j+1] for i in range(cfg.num_qubits) for j in range(i + 1, cfg.num_qubits)]
    return torch.cat(cols, dim=1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def compute_features(
    tokens: np.ndarray,
    point: StudyPoint,
    seed: int,
    device: str,
    batch_size: int = 16,
    dephase_trajectories: int = 4,
    return_states: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute QRC/control features and optionally retain final states for diagnostics."""
    if point.variant == "mean_field":
        return compute_mean_field_features(tokens, point, seed, device), []
    dev = torch.device(device if torch.cuda.is_available() or "cuda" not in device else "cpu")
    J_np, h_np, cfg = _variant_params(point, seed)
    J = torch.tensor(J_np, dtype=torch.float32)
    h = torch.tensor(h_np, dtype=torch.float32)
    obs = build_observable_signs(cfg.num_qubits, dev, torch.float32)
    x = torch.tensor(tokens, dtype=torch.float32)
    chunks: list[np.ndarray] = []
    saved_states: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        if point.variant == "event_dephase":
            avg = None
            representative = None
            for trajectory in range(dephase_trajectories):
                state = _evolve_with_event_dephase(xb, cfg, J, h, dev, seed * 1009 + trajectory)
                feat = compute_z_zz_from_state(state, obs)
                avg = feat if avg is None else avg + feat
                representative = state
            feat = avg / float(dephase_trajectories)
            state = representative
        else:
            state = evolve_batch_final_state(xb, cfg, J, h, dev)
            feat = compute_z_zz_from_state(state, obs)
        chunks.append(feat.cpu().numpy().astype(np.float32))
        if return_states and len(saved_states) < 2:
            saved_states.extend([s.copy() for s in state[: min(4, len(state))].cpu().numpy()])
        del state, feat
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(chunks), saved_states


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def load_or_compute_crc3_reference(
    tokens: np.ndarray,
    point: StudyPoint,
    seed: int,
    out_root: Path,
    token_schema: str,
    device: str,
) -> tuple[np.ndarray, Path]:
    """Load one W/g/variant-independent CRC3 reference matrix.

    CRC3 has no quantum W, g, dephasing, or interaction-ablation parameter.
    Consequently its cache key contains only the input stream, Q/Qin/T, L, and
    reservoir seed.  Phase-map and quantum-ablation points reuse this one fixed
    reference rather than silently creating a different classical model.
    """

    fingerprint = _array_fingerprint(tokens)
    cache_root = out_root / token_schema / point.dataset / "crc3_z_zz_matched_cache"
    stem = (
        f"Q{point.total_qubits:02d}_Qin{point.input_qubits:02d}_T{point.events:02d}"
        f"_L{point.layers}_seed{seed:03d}_{fingerprint[:16]}"
    )
    path = cache_root / f"{stem}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as package:
                features = package["features"].astype(np.float32)
                valid = (
                    features.shape
                    == (len(tokens), point.total_qubits * (point.total_qubits + 1) // 2)
                    and np.isfinite(features).all()
                    and str(package["protocol"].item()) == CRC3_Z_ZZ_PROTOCOL
                    and str(package["token_fingerprint"].item()) == fingerprint
                )
            if valid:
                return features, path
        except Exception:
            pass
    features, params = compute_crc3_z_zz_features(
        tokens,
        q=point.total_qubits,
        layers=point.layers,
        seed=seed,
        device=device,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        method=np.asarray(CRC3_Z_ZZ_METHOD),
        protocol=np.asarray(CRC3_Z_ZZ_PROTOCOL),
        token_fingerprint=np.asarray(fingerprint),
        total_qubits=np.int16(point.total_qubits),
        input_qubits=np.int16(point.input_qubits),
        events=np.int16(point.events),
        layers=np.int16(point.layers),
        seed=np.int32(seed),
        transform=np.asarray("none"),
        **params,
    )
    return features, path


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = np.square(singular)
    if power.sum() <= 0:
        return 0.0
    p = power / power.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def centered_kernel_alignment(features: np.ndarray, labels: np.ndarray) -> float:
    x = features - features.mean(axis=0, keepdims=True)
    y = labels.astype(np.float64)[:, None]
    y -= y.mean()
    kx, ky = x @ x.T, y @ y.T
    denom = np.linalg.norm(kx) * np.linalg.norm(ky)
    return float(np.sum(kx * ky) / denom) if denom > 0 else float("nan")


def select_input_only_observables(features: np.ndarray, total_qubits: int, input_qubits: int) -> np.ndarray:
    """Keep Z/ZZ observables supported entirely on the injected subsystem.

    This fixes readout dimension while ancillas are added, preventing a growing
    number of measured coordinates from masquerading as a reservoir-size gain.
    """
    columns = list(range(input_qubits))
    cursor = total_qubits
    for i in range(total_qubits):
        for j in range(i + 1, total_qubits):
            if i < input_qubits and j < input_qubits:
                columns.append(cursor)
            cursor += 1
    return features[:, columns]


def half_chain_entropy(state: np.ndarray, q: int) -> float:
    left = q // 2
    matrix = state.reshape(1 << left, 1 << (q - left))
    s = np.linalg.svd(matrix, compute_uv=False)
    p = np.square(s)
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def fit_validation_probe(features: np.ndarray, labels: np.ndarray, split: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """Train-only standardization and validation-only C selection; test once."""
    train, val, test = split
    scaler = StandardScaler().fit(features[train])
    xtr, xva, xte = scaler.transform(features[train]), scaler.transform(features[val]), scaler.transform(features[test])
    best: tuple[float, float, LogisticRegression] | None = None
    for C in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=0)
        model.fit(xtr, labels[train])
        prob = model.predict_proba(xva)[:, 1]
        auc = roc_auc_score(labels[val], prob) if len(np.unique(labels[val])) == 2 else float("nan")
        loss_proxy = float(np.mean(-(labels[val] * np.log(prob + 1e-9) + (1 - labels[val]) * np.log(1 - prob + 1e-9))))
        key = (auc, -loss_proxy)
        if best is None or key > best[:2]:
            best = (auc, -loss_proxy, model)
            best_C = C
    if best is None:
        raise RuntimeError("No valid validation model")
    model = best[2]
    prob = model.predict_proba(xte)[:, 1]
    return {
        "selected_C": float(best_C),
        "val_auc": float(best[0]),
        "test_auc": float(roc_auc_score(labels[test], prob)),
        "test_acc": float(accuracy_score(labels[test], prob >= 0.5)),
    }


def crc3_reference_record(
    features: np.ndarray,
    feature_path: Path,
    labels: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    point: StudyPoint,
) -> dict:
    """Compute or resume the fixed CRC3 probe and representation diagnostics."""

    split_fingerprint = _array_fingerprint(
        labels,
        np.asarray(split[0], dtype=np.int64),
        np.asarray(split[1], dtype=np.int64),
        np.asarray(split[2], dtype=np.int64),
    )
    metric_path = feature_path.with_suffix(".probe.json")
    if metric_path.exists():
        try:
            cached = json.loads(metric_path.read_text(encoding="utf-8"))
            if (
                cached.get("crc3_z_zz_protocol") == CRC3_Z_ZZ_PROTOCOL
                and cached.get("crc3_z_zz_probe_protocol") == CRC3_PROBE_PROTOCOL
                and cached.get("split_fingerprint") == split_fingerprint
            ):
                return cached
        except Exception:
            pass
    probe = fit_validation_probe(features, labels, split)
    fixed = select_input_only_observables(
        features, point.total_qubits, point.input_qubits
    )
    fixed_probe = fit_validation_probe(fixed, labels, split)
    record = {
        "crc3_z_zz_method": CRC3_Z_ZZ_METHOD,
        "crc3_z_zz_protocol": CRC3_Z_ZZ_PROTOCOL,
        "crc3_z_zz_probe_protocol": CRC3_PROBE_PROTOCOL,
        "crc3_z_zz_reference_scope": "independent_of_quantum_W_g_and_variant",
        "crc3_z_zz_feature_path": str(feature_path.resolve()),
        "crc3_z_zz_feature_dim": int(features.shape[1]),
        "crc3_z_zz_fixed_input_readout_dim": int(fixed.shape[1]),
        "crc3_z_zz_effective_rank": effective_rank(features),
        "crc3_z_zz_fixed_input_readout_effective_rank": effective_rank(fixed),
        "crc3_z_zz_kernel_alignment": centered_kernel_alignment(features, labels),
        "crc3_z_zz_fixed_input_readout_kernel_alignment": centered_kernel_alignment(fixed, labels),
        "crc3_z_zz_val_auc": probe["val_auc"],
        "crc3_z_zz_test_auc": probe["test_auc"],
        "crc3_z_zz_test_acc": probe["test_acc"],
        "crc3_z_zz_selected_C": probe["selected_C"],
        "crc3_z_zz_fixed_input_readout_val_auc": fixed_probe["val_auc"],
        "crc3_z_zz_fixed_input_readout_test_auc": fixed_probe["test_auc"],
        "crc3_z_zz_fixed_input_readout_test_acc": fixed_probe["test_acc"],
        "crc3_z_zz_fixed_input_readout_selected_C": fixed_probe["selected_C"],
        "split_fingerprint": split_fingerprint,
    }
    metric_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def perturbation_diagnostics(
    tokens: np.ndarray,
    base_features: np.ndarray,
    point: StudyPoint,
    seed: int,
    device: str,
    max_samples: int,
) -> tuple[list[float], float]:
    """Token-lag influence and normalized reverse-order representation commutator."""
    n = min(max_samples, len(tokens))
    x = tokens[:n]
    base = base_features[:n]
    scale = np.linalg.norm(base, axis=1).mean() + 1e-9
    influence: list[float] = []
    for lag in range(point.events):
        perturbed = x.copy()
        perturbed[:, lag] = 0.0
        feat, _ = compute_features(perturbed, point, seed, device, batch_size=max(1, min(8, n)))
        influence.append(float(np.linalg.norm(feat - base, axis=1).mean() / scale))
    rev, _ = compute_features(x[:, ::-1].copy(), point, seed, device, batch_size=max(1, min(8, n)))
    commutator = float(np.linalg.norm(rev - base, axis=1).mean() / scale)
    return influence, commutator


def build_plan(dataset: str, mode: str, section: str) -> list[StudyPoint]:
    dataset = dataset.upper()
    points: list[StudyPoint] = []
    if mode == "quick":
        if section in {"all", "fixed_q"}:
            points += [StudyPoint(dataset, "fixed_input_q", q, 4, 4) for q in (4, 6)]
        if section in {"all", "phase_map"}:
            points += [StudyPoint(dataset, "phase_map", 6, 4, 4, W=w, g=g, layers=1)
                       for w, g in ((0.0, 0.84), (0.6, 0.60))]
        if section in {"all", "ablations"}:
            points += [StudyPoint(dataset, "ablations", 6, 4, 4, variant=v) for v in ABLATIONS]
        return points
    if section in {"all", "fixed_q"}:
        points += [StudyPoint(dataset, "fixed_input_q", q, 6, 12) for q in (6, 8, 10, 12, 16, 20)]
    if section in {"all", "phase_map"}:
        points += [StudyPoint(dataset, "phase_map", 12, 6, 12, W=w, g=g, layers=L)
                   for w in (0.0, 0.3, 0.6, 0.9)
                   for g in (0.60, 0.72, 0.84, 0.96)
                   for L in (1, 2, 3)]
    if section in {"all", "ablations"}:
        points += [StudyPoint(dataset, "ablations", 12, 6, 12, variant=v) for v in ABLATIONS]
    return points


def run_point(
    point: StudyPoint,
    seed: int,
    tokens_all: np.ndarray,
    labels_all: np.ndarray,
    smiles_all: np.ndarray,
    out_root: Path,
    device: str,
    sample_limit: int | None,
    diagnostics_samples: int,
    force: bool,
    token_schema: str,
) -> dict:
    out = out_root / token_schema / point.dataset / point.section / point.key / f"seed_{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "metrics.json"
    if sample_limit is not None:
        # A deterministic stratified-ish prefix is only for quick/smoke mode.
        rng = np.random.default_rng(20260710)
        idx = np.sort(rng.choice(len(labels_all), min(sample_limit, len(labels_all)), replace=False))
        tokens, labels, smiles = tokens_all[idx], labels_all[idx], smiles_all[idx]
    else:
        tokens, labels, smiles = tokens_all, labels_all, smiles_all
    x = fixed_information_tokens(tokens, point.input_qubits, point.total_qubits, point.events)
    split = scaffold_split_indices(smiles, 0.8, 0.1, seed)
    crc3_features, crc3_path = load_or_compute_crc3_reference(
        x, point, seed, out_root, token_schema, device
    )
    crc3_record = crc3_reference_record(
        crc3_features, crc3_path, labels, split, point
    )
    (out / "crc3_z_zz_matched_reference.json").write_text(
        json.dumps(crc3_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if result_path.exists() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("crc3_z_zz_protocol") == CRC3_Z_ZZ_PROTOCOL
            and existing.get("crc3_z_zz_probe_protocol") == CRC3_PROBE_PROTOCOL
        ):
            return existing
        # Additive migration: retain all completed QRC features, probes, and
        # expensive perturbation diagnostics and append only the fixed CRC3
        # reference fields.
        existing.update(crc3_record)
        result_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
        return existing
    features, states = compute_features(
        x, point, seed, device,
        batch_size=max(1, min(32, 2 ** max(0, 16 - point.total_qubits))),
        dephase_trajectories=2 if sample_limit else 8,
        return_states=True,
    )
    probe = fit_validation_probe(features, labels, split)
    fixed_readout = select_input_only_observables(features, point.total_qubits, point.input_qubits)
    fixed_probe = fit_validation_probe(fixed_readout, labels, split)
    influence, order_comm = perturbation_diagnostics(
        x, features, point, seed, device, diagnostics_samples
    )
    entropy = [half_chain_entropy(s, point.total_qubits) for s in states] if states else []
    row = {
        **asdict(point),
        "seed": int(seed),
        "token_schema": token_schema,
        "n_samples": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "fixed_input_readout_dim": int(fixed_readout.shape[1]),
        "ancilla_qubits": int(point.total_qubits - point.input_qubits),
        "canonical_W_policy": "W=0.6 except explicit phase_map",
        "effective_rank": effective_rank(features),
        "fixed_input_readout_effective_rank": effective_rank(fixed_readout),
        "kernel_alignment": centered_kernel_alignment(features, labels),
        "fixed_input_readout_kernel_alignment": centered_kernel_alignment(fixed_readout, labels),
        "half_chain_entropy_mean": float(np.mean(entropy)) if entropy else 0.0,
        "order_commutator_normalized": order_comm,
        "token_lag_influence": influence,
        "fixed_input_readout_val_auc": fixed_probe["val_auc"],
        "fixed_input_readout_test_auc": fixed_probe["test_auc"],
        "fixed_input_readout_test_acc": fixed_probe["test_acc"],
        "fixed_input_readout_selected_C": fixed_probe["selected_C"],
        **crc3_record,
        **probe,
    }
    J_saved, h_saved, _ = _variant_params(point, seed)
    np.savez_compressed(
        out / "features.npz",
        features=features,
        fixed_input_readout_features=fixed_readout,
        labels=labels,
        J=J_saved,
        h=h_saved,
        token_schema=np.asarray(token_schema),
        total_qubits=np.asarray(point.total_qubits),
        input_qubits=np.asarray(point.input_qubits),
        events=np.asarray(point.events),
        seed=np.asarray(seed),
    )
    result_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    return row


def summarize(rows: list[dict], out_root: Path) -> pd.DataFrame:
    flat = [{**r, "token_lag_influence": json.dumps(r["token_lag_influence"])} for r in rows]
    frame = pd.DataFrame(flat)
    out_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_root / "metrics_by_seed.csv", index=False)
    keys = ["dataset", "section", "total_qubits", "input_qubits", "events", "W", "g", "layers", "variant"]
    metrics = ["test_auc", "test_acc", "fixed_input_readout_test_auc", "fixed_input_readout_test_acc",
               "effective_rank", "fixed_input_readout_effective_rank", "kernel_alignment",
               "half_chain_entropy_mean", "order_commutator_normalized",
               "crc3_z_zz_test_auc", "crc3_z_zz_test_acc",
               "crc3_z_zz_fixed_input_readout_test_auc",
               "crc3_z_zz_fixed_input_readout_test_acc",
               "crc3_z_zz_effective_rank",
               "crc3_z_zz_fixed_input_readout_effective_rank",
               "crc3_z_zz_kernel_alignment"]
    summary = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(out_root / "performance_summary.csv", index=False)
    paired_crc3_statistics(frame).to_csv(
        out_root / "paired_qrc_vs_crc3_z_zz_matched.csv", index=False
    )
    plot_summary(frame, out_root)
    return summary


def paired_crc3_statistics(frame: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    """Seed-paired QRC-minus-fixed-CRC3 differences for every mechanism point."""

    keys = ["dataset", "section", "total_qubits", "input_qubits", "events", "W", "g", "layers", "variant"]
    rows: list[dict] = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        identity = dict(zip(keys, values))
        for metric, baseline_metric in (
            ("test_auc", "crc3_z_zz_test_auc"),
            ("test_acc", "crc3_z_zz_test_acc"),
            ("fixed_input_readout_test_auc", "crc3_z_zz_fixed_input_readout_test_auc"),
            ("fixed_input_readout_test_acc", "crc3_z_zz_fixed_input_readout_test_acc"),
        ):
            pair = group[[metric, baseline_metric]].dropna()
            if pair.empty:
                continue
            delta = (pair[metric] - pair[baseline_metric]).to_numpy(dtype=float)
            digest = hashlib.sha256(
                (json.dumps(identity, sort_keys=True, default=str) + "|" + metric).encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
            if len(delta) == 1:
                low = high = float(delta[0])
            else:
                draws = rng.choice(delta, size=(int(n_boot), len(delta)), replace=True).mean(axis=1)
                low, high = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "reference": "qrc",
                    "comparison": CRC3_Z_ZZ_METHOD,
                    "n_pairs": int(len(delta)),
                    "mean_difference": float(delta.mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "qrc_wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "qrc_losses": int((delta < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(frame: pd.DataFrame, out_root: Path) -> None:
    for section, group in frame.groupby("section"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        if section == "fixed_input_q":
            x, hue = "total_qubits", "variant"
        elif section == "phase_map":
            x, hue = "g", "W"
        else:
            x, hue = "variant", "variant"
        for value, sub in group.groupby(hue):
            agg = sub.groupby(x).agg(test_auc=("test_auc", "mean"), rank=("effective_rank", "mean"), ent=("half_chain_entropy_mean", "mean")).reset_index()
            axes[0].plot(agg[x].astype(str) if x == "variant" else agg[x], agg.test_auc, marker="o", label=str(value))
            axes[1].plot(agg[x].astype(str) if x == "variant" else agg[x], agg["rank"], marker="o")
            axes[2].plot(agg[x].astype(str) if x == "variant" else agg[x], agg.ent, marker="o")
        crc = group.groupby(x).agg(
            auc=("crc3_z_zz_test_auc", "mean"),
            rank=("crc3_z_zz_effective_rank", "mean"),
        ).reset_index()
        crc_x = crc[x].astype(str) if x == "variant" else crc[x]
        axes[0].plot(crc_x, crc.auc, "D--", color="#C44E52", label="CRC3 z/zz (fixed reference)")
        axes[1].plot(crc_x, crc["rank"], "D--", color="#C44E52")
        if section == "fixed_input_q":
            fixed = group.groupby(x)["fixed_input_readout_test_auc"].mean().reset_index()
            axes[0].plot(fixed[x], fixed.fixed_input_readout_test_auc, "s--", color="black",
                         label="fixed input-only readout")
            crc_fixed = group.groupby(x)["crc3_z_zz_fixed_input_readout_test_auc"].mean().reset_index()
            axes[0].plot(
                crc_fixed[x], crc_fixed.crc3_z_zz_fixed_input_readout_test_auc,
                "^:", color="#C44E52", label="CRC3 fixed input-only readout",
            )
        axes[0].set_ylabel("Test AUC")
        axes[1].set_ylabel("Effective rank")
        axes[2].set_ylabel("Half-chain entropy")
        for ax in axes:
            ax.set_xlabel(x)
            ax.grid(alpha=0.25)
            if x == "variant":
                ax.tick_params(axis="x", rotation=35)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out_root / f"{section}_diagnostics.png", dpi=180)
        fig.savefig(out_root / f"{section}_diagnostics.pdf")
        plt.close(fig)


def main(argv: Iterable[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("BACE", "BBBP"), required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--section", choices=("all", "fixed_q", "phase_map", "ablations"), default="all")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    parser.add_argument("--token-schema", choices=("final_v2", "canonical_v3"), default="final_v2")
    parser.add_argument("--token-cache", type=Path, default=None,
                        help="Explicit NPZ with tokens/labels/smiles; overrides schema cache discovery.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not torch.cuda.is_available():
        devices = ["cpu"]
    tokens, labels, smiles = load_dataset(args.dataset, token_schema=args.token_schema, token_cache=args.token_cache)
    seeds = (0,) if args.mode == "quick" else SEEDS
    sample_limit = 64 if args.mode == "quick" else None
    diagnostics_samples = 4 if args.mode == "quick" else 16
    rows: list[dict] = []
    locks = {device: threading.Semaphore(1) for device in devices}

    def guarded(point: StudyPoint, seed: int, device: str) -> dict:
        with locks[device]:
            return run_point(point, seed, tokens, labels, smiles, args.output / args.mode,
                             device, sample_limit, diagnostics_samples, args.force, args.token_schema)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for point in build_plan(args.dataset, args.mode, args.section):
            futures = []
            for i, seed in enumerate(seeds):
                device = devices[i % len(devices)]
                print(f"[{point.dataset}] {point.section} {point.key} seed={seed} device={device}", flush=True)
                futures.append(pool.submit(guarded, point, seed, device))
            for future in as_completed(futures):
                rows.append(future.result())
    return summarize(rows, args.output / args.mode / args.token_schema / args.dataset)


if __name__ == "__main__":
    main()
