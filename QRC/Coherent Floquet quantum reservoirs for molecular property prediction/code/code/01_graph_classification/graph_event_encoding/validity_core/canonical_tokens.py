"""Static canonical reproducibility condition for general BFS tokens.

This is a new schema and intentionally never overwrites historical FINAL v2
caches.  Graph invariance is obtained by converting every valid input to one
canonical isomeric SMILES string, reparsing that string, then applying a stable
index-rooted BFS in the reparsed canonical graph.

The canonical ordering uses the complete static graph and therefore is *not*
prefix-causal. It is a reproducibility/control condition for static molecular
property prediction, not the primary representation for a naturally dynamic
graph. Dynamic data must use observed timestamps; weak-prior static studies
should additionally report permutation/root ensembles (invariance in
expectation).

The 72-slot local event feature layout remains numerically aligned with v2 so
existing QRC/CRC code can consume it. Provenance and cache names make the
traversal change explicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from .token_variants import stable_seed, tokenize_mol_from_root


CANONICAL_SCHEMA_VERSION = "local-bfs-canonical-isomeric-reparse-stable-v3"
CANONICAL_TRAVERSAL_PROVENANCE = {
    "condition": "static canonical reproducibility condition",
    "invariance_type": "invariant-by-canonicalization",
    "causality_warning": "canonical ordering depends on the complete static graph and is not prefix-causal",
    "dynamic_data_policy": "use observed timestamps; do not canonicalize temporal order",
    "canonicalization": "RDKit canonical isomeric SMILES",
    "reparse": "Chem.MolFromSmiles(canonical_isomeric_smiles)",
    "root": "atom index 0 after canonical reparse",
    "neighbor_tie_break": "ascending atom index after canonical reparse",
    "event_order": "root atom; for every BFS visit emit unseen bond then newly discovered atom",
    "ring_closure": "prefix-causal closure flag when an emitted edge reaches an already visited atom",
    "padding": "historical typed [PAD] token; padding ablation is implemented separately",
    "task_slots": "all exactly zero",
    "label_use": "none",
}


@dataclass(frozen=True)
class CanonicalTokenConfig:
    dataset: str
    data_dir: str
    cache_dir: str
    max_events: int = 30

    def normalized(self) -> "CanonicalTokenConfig":
        dataset = self.dataset.upper()
        if dataset not in {"BACE", "BBBP"}:
            raise ValueError(dataset)
        if not 1 <= int(self.max_events) <= 32:
            raise ValueError("max_events must lie in 1..32")
        return CanonicalTokenConfig(
            dataset=dataset,
            data_dir=str(Path(self.data_dir).resolve()),
            cache_dir=str(Path(self.cache_dir).resolve()),
            max_events=int(self.max_events),
        )


def canonical_reparse(mol_or_smiles: Chem.Mol | str) -> tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(str(mol_or_smiles)) if isinstance(mol_or_smiles, (str, np.str_)) else mol_or_smiles
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot canonicalize an empty/invalid molecule")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(canonical)
    if reparsed is None or reparsed.GetNumAtoms() == 0:
        raise RuntimeError(f"Canonical SMILES did not reparse: {canonical}")
    roundtrip = Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=True)
    if roundtrip != canonical:
        raise RuntimeError(f"Unstable canonical round trip: {canonical} -> {roundtrip}")
    return reparsed, canonical


def tokenize_canonical_general(
    mol_or_smiles: Chem.Mol | str,
    dataset: str,
    max_events: int = 30,
) -> tuple[np.ndarray, int, dict]:
    """Return one deterministic static-canonical v3 control stream."""
    reparsed, canonical = canonical_reparse(mol_or_smiles)
    tokens, count = tokenize_mol_from_root(reparsed, dataset, 0, int(max_events), task_aware=False)
    provenance = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "canonical_smiles": canonical,
        "num_atoms": int(reparsed.GetNumAtoms()),
        "num_bonds": int(reparsed.GetNumBonds()),
        "root": 0,
        "event_count": int(count),
        **CANONICAL_TRAVERSAL_PROVENANCE,
    }
    return tokens, count, provenance


def tokenize_canonical_multi_root_general(
    mol_or_smiles: Chem.Mol | str,
    dataset: str,
    max_events: int = 30,
    root_count: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return deterministic per-root streams; aggregate only after a reservoir.

    Root 0 is the canonical v3 stream.  Additional roots follow canonical
    reparsed atom index.  If a molecule has fewer atoms than ``root_count``,
    roots cycle deterministically.  The returned streams must not be averaged
    before QRC/CRC when testing nonlinear multi-root ensembles.
    """
    if root_count < 1:
        raise ValueError("root_count must be positive")
    reparsed, canonical = canonical_reparse(mol_or_smiles)
    roots = [index % reparsed.GetNumAtoms() for index in range(int(root_count))]
    streams, counts = [], []
    for root in roots:
        tokens, count = tokenize_mol_from_root(reparsed, dataset, root, int(max_events), task_aware=False)
        streams.append(tokens)
        counts.append(count)
    provenance = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "canonical_smiles": canonical,
        "root_policy": "first root_count atom indices after canonical reparse; cycle for small graphs",
        "roots": roots,
        "ensemble_rule": "map every root independently, then mean QRC/CRC/raw representation",
        **CANONICAL_TRAVERSAL_PROVENANCE,
    }
    return np.stack(streams).astype(np.float32), np.asarray(counts, dtype=np.int16), provenance


def audit_canonical_general_invariance(
    smiles_values: Sequence[str],
    dataset: str,
    *,
    q: int,
    t: int,
    repeats: int = 2,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Verify v3 equality under atom relabeling and randomized SMILES."""
    rows = []
    for sample_index, value in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise RuntimeError(f"Invalid cached SMILES at {sample_index}")
        reference, reference_count, provenance = tokenize_canonical_general(mol, dataset, t)
        for rep in range(repeats):
            rng = np.random.default_rng(stable_seed(seed, "canonical_audit", sample_index, rep))
            relabelled = Chem.RenumberAtoms(mol, rng.permutation(mol.GetNumAtoms()).astype(int).tolist())
            relabel_tokens, relabel_count, _ = tokenize_canonical_general(relabelled, dataset, t)
            random_smiles = Chem.MolToRandomSmilesVect(
                mol, 1, stable_seed(seed, "canonical_random_smiles", sample_index, rep), True
            )[0]
            random_tokens, random_count, _ = tokenize_canonical_general(str(random_smiles), dataset, t)
            for variant, tokens, count in (
                ("atom_relabel", relabel_tokens, relabel_count),
                ("randomized_smiles", random_tokens, random_count),
            ):
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "repeat": int(rep),
                        "variant": variant,
                        "schema_version": CANONICAL_SCHEMA_VERSION,
                        "canonical_smiles": provenance["canonical_smiles"],
                        "event_count_equal": bool(count == reference_count),
                        "token_exact_equal": bool(
                            np.array_equal(tokens[:t, : 3 * q], reference[:t, : 3 * q])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _dataset_spec(dataset: str, data_dir: Path) -> tuple[Path, str, str]:
    if dataset == "BACE":
        return data_dir / "bace" / "raw" / "bace.csv", "mol", "Class"
    return data_dir / "bbbp" / "raw" / "BBBP.csv", "smiles", "p_np"


def _fingerprint(smiles: Sequence[str], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(CANONICAL_SCHEMA_VERSION.encode("utf-8"))
    for smi, label in zip(smiles, labels):
        digest.update(str(smi).encode("utf-8"))
        digest.update(bytes([int(label)]))
    return digest.hexdigest()


def load_or_build_canonical_token_cache(
    dataset: str,
    data_dir: str | Path,
    cache_dir: str | Path,
    *,
    max_events: int = 30,
    force: bool = False,
) -> tuple[Path, dict]:
    """Build a labelled-row v3 cache without touching any v2 cache."""
    cfg = CanonicalTokenConfig(dataset, str(data_dir), str(cache_dir), max_events).normalized()
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.dataset}_general_canonical_v3_Q24_T{cfg.max_events:02d}.npz"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") == CANONICAL_SCHEMA_VERSION:
            return cache_path, metadata
    csv_path, smiles_column, label_column = _dataset_spec(cfg.dataset, Path(cfg.data_dir))
    frame = pd.read_csv(csv_path)
    token_rows, labels, canonical_smiles, event_counts, source_indices = [], [], [], [], []
    invalid_smiles = invalid_labels = 0
    for source_index, row in frame.iterrows():
        try:
            label = int(float(row[label_column]))
        except (TypeError, ValueError):
            invalid_labels += 1
            continue
        if label not in (0, 1):
            invalid_labels += 1
            continue
        try:
            tokens, count, provenance = tokenize_canonical_general(
                str(row[smiles_column]), cfg.dataset, cfg.max_events
            )
        except ValueError:
            invalid_smiles += 1
            continue
        token_rows.append(tokens)
        labels.append(label)
        canonical_smiles.append(provenance["canonical_smiles"])
        event_counts.append(count)
        source_indices.append(int(source_index))
    if not token_rows:
        raise RuntimeError(f"No valid molecules in {csv_path}")
    tokens = np.stack(token_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(
        cache_path,
        tokens=tokens,
        labels=y,
        smiles=np.asarray(canonical_smiles),
        event_counts=np.asarray(event_counts, dtype=np.int16),
        source_row_indices=np.asarray(source_indices, dtype=np.int32),
        token_schema_version=np.asarray(CANONICAL_SCHEMA_VERSION),
    )
    metadata = {
        "config": asdict(cfg),
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "traversal_provenance": CANONICAL_TRAVERSAL_PROVENANCE,
        "source_csv": str(csv_path.resolve()),
        "source_rows": int(len(frame)),
        "valid_labelled_rows": int(len(y)),
        "invalid_smiles_rows": int(invalid_smiles),
        "invalid_label_rows": int(invalid_labels),
        "shape": list(tokens.shape),
        "fingerprint": _fingerprint(canonical_smiles, y),
        "historical_cache_policy": "v3 filename is distinct; no v2 cache is read, mutated, or overwritten",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return cache_path, metadata
