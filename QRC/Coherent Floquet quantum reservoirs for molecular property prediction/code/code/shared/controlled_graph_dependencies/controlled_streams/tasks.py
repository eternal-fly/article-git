"""Synthetic controlled graph-event streams with auditable information matching.

The tasks in this module are deliberately label-free at encoding time.  Labels are
defined only by a relation between events (order, lag, parity, or graph motif), while
nuisance quantities such as event counts, unary marginals, and—in the third-order
task—cyclic ordered-pair counts are matched between classes by construction.

Every task returns a dense tensor ``[sample, event, 3 * Q]`` so that the same stream
can be injected into the FINAL QRC, scalar CRC, ESN, and raw baselines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TaskSetting:
    """One independently runnable controlled task setting."""

    name: str
    family: str
    sequence_length: int
    parameter_name: str = "none"
    parameter_value: int | str = 0
    contrast: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ControlledDataset:
    tokens: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    metadata: dict[str, Any]
    symbols: np.ndarray | None = None

    def validate(self, q: int) -> None:
        if self.tokens.ndim != 3 or self.tokens.shape[-1] != 3 * int(q):
            raise ValueError(f"Expected [N,T,{3*q}] tokens, got {self.tokens.shape}")
        n = len(self.tokens)
        if len(self.labels) != n or len(self.groups) != n:
            raise ValueError("tokens, labels, and groups must contain the same samples")
        if set(np.unique(self.labels).tolist()) != {0, 1}:
            raise ValueError("Every controlled setting must be a balanced binary task")
        if not np.isfinite(self.tokens).all():
            raise ValueError("Non-finite token value detected")


# Fixed sparse symbol code.  It is independent of task, label, QRC seed, and split.
SYMBOL_CODE_VERSION = "sparse-local-symbol-code-v1"


def encode_symbol_sequences(symbols: np.ndarray, q: int) -> np.ndarray:
    """Encode integer events into a fixed sparse three-channel Q-wire representation.

    ``-1`` denotes padding and maps to the all-zero identity event.  Non-negative
    symbols activate three deterministic wire/channel locations.  No learned or
    task-specific projection is used.
    """

    symbols = np.asarray(symbols, dtype=np.int16)
    if symbols.ndim != 2:
        raise ValueError(f"symbols must be [N,T], got {symbols.shape}")
    if q < 2:
        raise ValueError("Controlled streams require Q >= 2")
    n, t = symbols.shape
    out = np.zeros((n, t, q, 3), dtype=np.float32)
    for row in range(n):
        for step in range(t):
            code = int(symbols[row, step])
            if code < 0:
                continue
            w0 = code % q
            w1 = (2 * code + 1) % q
            w2 = (3 * code + 2) % q
            out[row, step, w0, 0] += 1.0
            out[row, step, w1, 1] += 0.75
            out[row, step, w2, 2] += 0.50
    return out.reshape(n, t, 3 * q)


def encode_edge_sequences(edge_rows: list[list[tuple[int, int]]], q: int) -> np.ndarray:
    """Encode undirected edge-addition events as local source/destination incidences."""

    if not edge_rows:
        raise ValueError("edge_rows is empty")
    t = len(edge_rows[0])
    out = np.zeros((len(edge_rows), t, q, 3), dtype=np.float32)
    for sample_index, edges in enumerate(edge_rows):
        if len(edges) != t:
            raise ValueError("All edge streams in one setting must have equal length")
        for event_index, (a, b) in enumerate(edges):
            a, b = sorted((int(a), int(b)))
            if not (0 <= a < q and 0 <= b < q):
                raise ValueError(f"Edge {(a,b)} cannot be represented with Q={q}")
            out[sample_index, event_index, a, 0] = 1.0
            out[sample_index, event_index, b, 1] = 1.0
            out[sample_index, event_index, a, 2] = 0.5
            out[sample_index, event_index, b, 2] = 0.5
    return out.reshape(len(edge_rows), t, 3 * q)


def _pack_symbol_dataset(
    sequences: list[np.ndarray],
    labels: list[int],
    groups: list[int],
    q: int,
    setting: TaskSetting,
    extra: dict[str, Any],
) -> ControlledDataset:
    symbols = np.stack(sequences).astype(np.int16)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)
    dataset = ControlledDataset(
        tokens=encode_symbol_sequences(symbols, q),
        labels=y,
        groups=group_array,
        symbols=symbols,
        metadata={
            "task": setting.to_dict(),
            "representation": SYMBOL_CODE_VERSION,
            "n_samples": int(len(y)),
            "class_counts": {str(c): int((y == c).sum()) for c in (0, 1)},
            "n_groups": int(len(np.unique(group_array))),
            **extra,
        },
    )
    dataset.validate(q)
    return dataset


def _roll(sequence: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(np.asarray(sequence, dtype=np.int16), int(shift))


def _second_order(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    t = int(setting.sequence_length)
    if t < 6 or t % 2:
        raise ValueError("second_order requires an even sequence length >= 6")
    low = np.asarray([0] * (t // 2) + [1] * (t // 2), dtype=np.int16)
    high = np.asarray(([0, 1] * (t // 2)), dtype=np.int16)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(n_groups):
        shift = int(rng.integers(0, t))
        complement = int(rng.integers(0, 2))
        for label, base in ((0, low), (1, high)):
            rows.append(_roll(base ^ complement, shift))
            labels.append(label)
            groups.append(group)
    return _pack_symbol_dataset(
        rows,
        labels,
        groups,
        q,
        setting,
        {
            "controlled_claim": "per-sample unary counts match; ordered adjacent-pair statistics differ",
            "class_definition": "low versus high transition density",
            "cyclic_shift_randomized": True,
        },
    )


def _cyclic_hist(bits: tuple[int, ...], order: int) -> tuple[int, ...]:
    n = len(bits)
    counts = np.zeros(2**order, dtype=np.int16)
    for start in range(n):
        code = 0
        for offset in range(order):
            code = (code << 1) | int(bits[(start + offset) % n])
        counts[code] += 1
    return tuple(int(v) for v in counts)


def _linear_hist(bits: tuple[int, ...], order: int) -> tuple[int, ...]:
    counts = np.zeros(2**order, dtype=np.int16)
    for start in range(len(bits) - order + 1):
        code = 0
        for offset in range(order):
            code = (code << 1) | int(bits[start + offset])
        counts[code] += 1
    return tuple(int(v) for v in counts)


def _find_third_order_pair(t: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Find finite strings with identical unary/bigram counts but distinct trigrams."""

    if not 6 <= t <= 16:
        raise ValueError("third_order exhaustive construction supports 6 <= T <= 16")
    groups: dict[tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    for bits in product((0, 1), repeat=t):
        unary = _linear_hist(bits, 1)
        pair = _linear_hist(bits, 2)
        # Matching endpoints prevents a boundary cue from bypassing pair statistics.
        signature = unary + pair + (int(bits[0]), int(bits[-1]))
        groups.setdefault(signature, []).append((bits, _linear_hist(bits, 3)))
    best: tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None
    for signature, candidates in groups.items():
        by_hist: dict[tuple[int, ...], tuple[int, ...]] = {}
        for bits, hist in candidates:
            by_hist.setdefault(hist, bits)
        histograms = list(by_hist)
        for i, hist_a in enumerate(histograms):
            for hist_b in histograms[i + 1 :]:
                score = int(sum(abs(a - b) for a, b in zip(hist_a, hist_b)))
                if score and (best is None or score > best[0]):
                    best = (score, by_hist[hist_a], by_hist[hist_b], signature, hist_a, hist_b)
    if best is None:
        raise RuntimeError(f"Could not construct a third-order matched pair for T={t}")
    _, bits_a, bits_b, signature, hist_a, hist_b = best
    return (
        np.asarray(bits_a, dtype=np.int16),
        np.asarray(bits_b, dtype=np.int16),
        signature,
        hist_a,
        hist_b,
    )


def _third_order(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    t = int(setting.sequence_length)
    base0, base1, low_order_signature, trigram0, trigram1 = _find_third_order_pair(t)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(n_groups):
        # Cycle deterministically through a balanced nuisance orbit.  Complement and
        # reversal preserve equality of the matched lower-order signature.
        complement = group % 2
        reverse = (group // 2) % 2
        for label, base in ((0, base0), (1, base1)):
            transformed = base[::-1] if reverse else base
            rows.append(np.asarray(transformed ^ complement, dtype=np.int16))
            labels.append(label)
            groups.append(group)
    return _pack_symbol_dataset(
        rows,
        labels,
        groups,
        q,
        setting,
        {
            "controlled_claim": "each paired sample has identical finite-string unary counts, ordered-bigram counts, and endpoints, but distinct trigram histograms",
            "low_order_signature": list(low_order_signature),
            "base_class0": base0.tolist(),
            "base_class1": base1.tolist(),
            "trigram_hist_class0": list(trigram0),
            "trigram_hist_class1": list(trigram1),
            "balanced_nuisance_orbit": "common bit complement and common reversal",
        },
    )


def _direction(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    t = int(setting.sequence_length)
    if t < 5:
        raise ValueError("direction requires T >= 5")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(n_groups):
        start = int(rng.integers(0, t - 2))
        for label in (0, 1):
            sequence = np.zeros(t, dtype=np.int16)
            sequence[start : start + 3] = [1, 2, 3] if label == 0 else [1, 3, 2]
            rows.append(sequence)
            labels.append(label)
            groups.append(group)
    return _pack_symbol_dataset(
        rows,
        labels,
        groups,
        q,
        setting,
        {
            "controlled_claim": "event multiset and absolute cue location distribution match; only M-A-B versus M-B-A order differs",
            "symbols": {"neutral": 0, "marker": 1, "A": 2, "B": 3},
        },
    )


def _delayed_motif(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    t = int(setting.sequence_length)
    lag = int(setting.parameter_value)
    if lag < 1 or lag >= t - 2:
        raise ValueError(f"delayed_motif requires 1 <= lag < T-2, got lag={lag}, T={t}")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(n_groups):
        valid_pairs = [
            (a, b)
            for a in range(lag, t)
            for b in range(a + 1, t)
            if (a - lag) not in {a, b} and (b - lag) not in {a, b}
        ]
        if not valid_pairs:
            raise ValueError(f"No nuisance-matched delayed pair exists for lag={lag}, T={t}")
        target_a, target_b = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
        for target, decoy in ((target_a, target_b), (target_b, target_a)):
            marker = target - lag
            for label in (0, 1):
                seq = np.zeros(t, dtype=np.int16)
                seq[marker] = 1
                seq[target] = 2 if label == 0 else 3
                seq[decoy] = 3 if label == 0 else 2
                rows.append(seq)
                labels.append(label)
                groups.append(group)
    return _pack_symbol_dataset(
        rows,
        labels,
        groups,
        q,
        setting,
        {
            "controlled_claim": "symbol counts and absolute symbol-position marginals match exactly within each four-sample group; label is which candidate symbol lies at the marker-relative lag",
            "lag": lag,
            "samples_per_nuisance_group": 4,
        },
    )


def _distant_pair(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    t = int(setting.sequence_length)
    lag = int(setting.parameter_value)
    if lag < 1 or lag >= t:
        raise ValueError(f"distant_pair requires 1 <= lag < T, got lag={lag}, T={t}")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    patterns = ((0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1))
    for group in range(n_groups):
        end_low = max(lag, t - 4)
        end = int(rng.integers(end_low, t))
        start = end - lag
        for bit_a, bit_b, label in patterns:
            seq = np.zeros(t, dtype=np.int16)
            seq[start] = 1 + bit_a
            seq[start + lag] = 1 + bit_b
            rows.append(seq)
            labels.append(label)
            groups.append(group)
    return _pack_symbol_dataset(
        rows,
        labels,
        groups,
        q,
        setting,
        {
            "controlled_claim": "both marked positions have exactly balanced bit marginals in each label; label is equality/XOR of the distant pair",
            "lag": lag,
            "samples_per_nuisance_group": 4,
        },
    )


def _parity(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    t = int(setting.sequence_length)
    spacing = int(setting.parameter_value)
    base_positions = (0, spacing, 2 * spacing)
    if spacing < 1 or base_positions[-1] >= t:
        raise ValueError(f"parity requires 2*spacing < T, got spacing={spacing}, T={t}")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(n_groups):
        end_low = max(base_positions[-1], t - 4)
        end = int(rng.integers(end_low, t))
        positions = (end - 2 * spacing, end - spacing, end)
        for bits in product((0, 1), repeat=3):
            seq = np.zeros(t, dtype=np.int16)
            for position, bit in zip(positions, bits):
                seq[position] = 1 + int(bit)
            rows.append(seq)
            labels.append(int(bits[0] ^ bits[1] ^ bits[2]))
            groups.append(group)
    return _pack_symbol_dataset(
        rows,
        labels,
        groups,
        q,
        setting,
        {
            "controlled_claim": "all one- and two-bit marginals match exactly; label is three-bit parity",
            "spacing": spacing,
            "endpoint_policy": "latest marked bit sampled from the final four positions; earlier bits move with spacing",
            "samples_per_nuisance_group": 8,
        },
    )


GRAPH_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "path": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
    "branch": ((0, 1), (0, 2), (0, 3), (3, 4), (3, 5)),
    # A 5-cycle plus one isolated node: same node and edge counts as the trees.
    "cycle": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)),
}


def _relabel_and_order(
    edges: Iterable[tuple[int, int]], permutation: np.ndarray, edge_order: np.ndarray
) -> list[tuple[int, int]]:
    relabelled = [(int(permutation[a]), int(permutation[b])) for a, b in edges]
    return [relabelled[int(index)] for index in edge_order]


def _static_graph(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    if q < 6:
        raise ValueError("static graph tasks require Q >= 6")
    left, right = setting.contrast.split("_vs_")
    if left not in GRAPH_EDGES or right not in GRAPH_EDGES:
        raise ValueError(f"Unsupported graph contrast: {setting.contrast}")
    edge_rows: list[list[tuple[int, int]]] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(n_groups):
        permutation = rng.permutation(6)
        edge_order = rng.permutation(5)
        for label, graph_name in ((0, left), (1, right)):
            edge_rows.append(_relabel_and_order(GRAPH_EDGES[graph_name], permutation, edge_order))
            labels.append(label)
            groups.append(group)
    tokens = encode_edge_sequences(edge_rows, q)
    dataset = ControlledDataset(
        tokens=tokens,
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
        symbols=None,
        metadata={
            "task": setting.to_dict(),
            "representation": "local-undirected-edge-incidence-v1",
            "controlled_claim": "six nodes and five edge-addition events match; node relabelling and event order are randomized identically within each class pair",
            "contrast": setting.contrast,
            "n_samples": int(len(labels)),
            "n_groups": int(n_groups),
            "class_counts": {"0": int(n_groups), "1": int(n_groups)},
        },
    )
    dataset.validate(q)
    return dataset


def _prefix_graph(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    if q < 6:
        raise ValueError("prefix graph task requires Q >= 6")
    prefix = int(setting.parameter_value)
    if not 1 <= prefix <= 5:
        raise ValueError("prefix_length must lie in [1,5]")
    base = TaskSetting(
        name=setting.name,
        family="static_graph",
        sequence_length=5,
        contrast="path_vs_branch",
    )
    full = _static_graph(base, q, n_groups, rng)
    full.tokens = np.ascontiguousarray(full.tokens[:, :prefix], dtype=np.float32)
    full.metadata = {
        **full.metadata,
        "task": setting.to_dict(),
        "controlled_claim": "predict the final graph motif using only the first k edge-addition events",
        "prefix_length": prefix,
        "full_event_count": 5,
    }
    full.validate(q)
    return full


def _future_event(setting: TaskSetting, q: int, n_groups: int, rng: np.random.Generator) -> ControlledDataset:
    lag = int(setting.parameter_value)
    t = int(setting.sequence_length)
    if lag < 0 or t < lag + 4:
        raise ValueError("future_event requires T >= lag + 4")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(n_groups):
        prefix_neutral = int(rng.integers(0, max(1, t - lag - 2)))
        cue_start = min(prefix_neutral, t - lag - 3)
        for label in (0, 1):
            seq = np.zeros(t, dtype=np.int16)
            seq[cue_start : cue_start + 3] = [1, 2, 3] if label == 0 else [1, 3, 2]
            # The target event itself is withheld: 4 for label 0, 5 for label 1.
            rows.append(seq)
            labels.append(label)
            groups.append(group)
    return _pack_symbol_dataset(
        rows,
        labels,
        groups,
        q,
        setting,
        {
            "controlled_claim": "the future target is withheld; its identity is determined by an earlier AB-versus-BA cue followed by a controlled distractor lag",
            "lag": lag,
            "future_target_map": {"MAB": 4, "MBA": 5},
        },
    )


GENERATORS = {
    "second_order": _second_order,
    "third_order": _third_order,
    "direction": _direction,
    "delayed_motif": _delayed_motif,
    "distant_pair": _distant_pair,
    "parity": _parity,
    "static_graph": _static_graph,
    "prefix_graph": _prefix_graph,
    "future_event": _future_event,
}


def generate_dataset(
    setting: TaskSetting,
    q: int,
    n_groups: int,
    generation_seed: int,
) -> ControlledDataset:
    """Generate one deterministic task dataset, independent of model/split seeds."""

    if setting.family not in GENERATORS:
        raise KeyError(f"Unknown task family {setting.family!r}")
    rng = np.random.default_rng(int(generation_seed))
    dataset = GENERATORS[setting.family](setting, int(q), int(n_groups), rng)
    dataset.metadata.update(
        {
            "q": int(q),
            "token_dim": int(3 * q),
            "generation_seed": int(generation_seed),
        }
    )
    return dataset


def task_settings(profile: str) -> list[TaskSetting]:
    """Return the complete manifest for ``smoke``, ``quick``, or ``full``."""

    profile = profile.lower()
    if profile not in {"smoke", "quick", "full"}:
        raise ValueError("profile must be smoke, quick, or full")
    if profile == "smoke":
        return [
            TaskSetting("second_order_T08", "second_order", 8),
            TaskSetting("third_order_T08", "third_order", 8),
            TaskSetting("direction_AB_BA_T08", "direction", 8),
            TaskSetting("delayed_motif_lag03", "delayed_motif", 10, "lag", 3),
            TaskSetting("distant_pair_lag04", "distant_pair", 10, "lag", 4),
            TaskSetting("parity_spacing02", "parity", 8, "spacing", 2),
            TaskSetting("static_path_vs_cycle", "static_graph", 5, contrast="path_vs_cycle"),
            TaskSetting("prefix_graph_k03", "prefix_graph", 3, "prefix_length", 3),
            TaskSetting("future_event_lag04", "future_event", 8, "lag", 4),
        ]
    if profile == "quick":
        settings = [
            TaskSetting("second_order_T12", "second_order", 12),
            TaskSetting("third_order_T10", "third_order", 10),
            TaskSetting("direction_AB_BA_T12", "direction", 12),
        ]
        settings += [TaskSetting(f"delayed_motif_lag{lag:02d}", "delayed_motif", 16, "lag", lag) for lag in (1, 4, 8)]
        settings += [TaskSetting(f"distant_pair_lag{lag:02d}", "distant_pair", 16, "lag", lag) for lag in (1, 4, 8)]
        settings += [TaskSetting(f"parity_spacing{s:02d}", "parity", 12, "spacing", s) for s in (1, 3, 5)]
        settings += [
            TaskSetting(f"static_{contrast}", "static_graph", 5, contrast=contrast)
            for contrast in ("path_vs_branch", "path_vs_cycle", "branch_vs_cycle")
        ]
        settings += [TaskSetting(f"prefix_graph_k{k:02d}", "prefix_graph", k, "prefix_length", k) for k in (1, 3, 5)]
        settings += [TaskSetting(f"future_event_lag{lag:02d}", "future_event", lag + 4, "lag", lag) for lag in (0, 4, 8)]
        return settings
    settings = [
        TaskSetting("second_order_T16", "second_order", 16),
        TaskSetting("third_order_T12", "third_order", 12),
        TaskSetting("direction_AB_BA_T16", "direction", 16),
    ]
    settings += [TaskSetting(f"delayed_motif_lag{lag:02d}", "delayed_motif", 20, "lag", lag) for lag in (1, 2, 4, 6, 8, 12, 16)]
    settings += [TaskSetting(f"distant_pair_lag{lag:02d}", "distant_pair", 20, "lag", lag) for lag in (1, 2, 4, 6, 8, 12, 16)]
    settings += [TaskSetting(f"parity_spacing{s:02d}", "parity", 20, "spacing", s) for s in (1, 2, 3, 4, 6, 8)]
    settings += [
        TaskSetting(f"static_{contrast}", "static_graph", 5, contrast=contrast)
        for contrast in ("path_vs_branch", "path_vs_cycle", "branch_vs_cycle")
    ]
    settings += [TaskSetting(f"prefix_graph_k{k:02d}", "prefix_graph", k, "prefix_length", k) for k in range(1, 6)]
    settings += [TaskSetting(f"future_event_lag{lag:02d}", "future_event", lag + 4, "lag", lag) for lag in (0, 1, 2, 4, 8, 12, 16)]
    return settings
