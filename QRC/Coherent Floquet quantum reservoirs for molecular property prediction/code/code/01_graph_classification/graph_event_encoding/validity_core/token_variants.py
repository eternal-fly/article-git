"""Deterministic isomorphic traversals and order perturbations.

The feature schema is imported from ``FINAL/2_linear_readout``.  Only graph
indexing, BFS root, valid-event order, or padding treatment changes here.
Labels are never consulted while constructing a token stream.
"""

from __future__ import annotations

import hashlib
import sys
from collections import deque
from pathlib import Path

import numpy as np
from rdkit import Chem


MODULE_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = MODULE_ROOT.parent
BASELINE_ROOT = FINAL_ROOT / "2_linear_readout"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from core.general_tokens import (  # noqa: E402
    TASK_DIM,
    TASK_SLOT_INDICES,
    _assemble_token,
    _atom_generic,
    _bond_generic,
    _pad_token,
)


ORDER_VARIANTS = (
    "original",
    "reverse",
    "full_shuffle",
    "block_shuffle",
    "within_depth",
    "local_swap",
    "cyclic_shift",
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)


def tokenize_mol_from_root(
    mol: Chem.Mol,
    dataset: str,
    root: int,
    max_events: int,
    *,
    task_aware: bool = False,
) -> tuple[np.ndarray, int]:
    """Run the exact FINAL local token schema from an explicitly chosen root."""
    if task_aware:
        raise ValueError("5_graph_stream_validity is intentionally restricted to general tokens.")
    if mol is None or mol.GetNumAtoms() == 0:
        return np.stack([_pad_token()] * max_events).astype(np.float32), 0
    root = int(root)
    if not 0 <= root < mol.GetNumAtoms():
        raise ValueError(f"root={root} is outside molecule with {mol.GetNumAtoms()} atoms")
    task_values = np.zeros(TASK_DIM, dtype=np.float32)
    visited = {root}
    emitted_edges: set[tuple[int, int]] = set()
    queue = deque([(root, 0)])
    events = [_assemble_token(_atom_generic(mol.GetAtomWithIdx(root), 0, None), task_values)]
    while queue and len(events) < max_events:
        current, depth = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in sorted(atom.GetNeighbors(), key=lambda value: value.GetIdx()):
            target = neighbor.GetIdx()
            edge = tuple(sorted((current, target)))
            if edge in emitted_edges:
                continue
            emitted_edges.add(edge)
            bond = mol.GetBondBetweenAtoms(current, target)
            closure = target in visited
            events.append(
                _assemble_token(_bond_generic(bond, atom, neighbor, depth + 1, closure), task_values)
            )
            if len(events) >= max_events:
                break
            if not closure:
                visited.add(target)
                events.append(_assemble_token(_atom_generic(neighbor, depth + 1, bond), task_values))
                queue.append((target, depth + 1))
                if len(events) >= max_events:
                    break
    count = len(events)
    events.extend([_pad_token()] * (max_events - count))
    return np.stack(events).astype(np.float32), count


def _canonical_identity(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def build_isomorphic_token_variants(
    reference_tokens: np.ndarray,
    reference_event_counts: np.ndarray,
    smiles: np.ndarray,
    dataset: str,
    q: int,
    t: int,
    *,
    relabel_repeats: int = 2,
    random_smiles_repeats: int = 2,
    root_repeats: int = 4,
    variant_seed: int = 20260710,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Create same-graph token streams with controlled index/traversal changes.

    The first variant is the exact cached FINAL stream.  All alternatives are
    checked by canonical isomeric SMILES before their tokens are accepted.
    """
    names = ["original"]
    names += [f"atom_relabel_r{rep:02d}" for rep in range(relabel_repeats)]
    names += [f"random_smiles_r{rep:02d}" for rep in range(random_smiles_repeats)]
    names += [f"root_r{rep:02d}" for rep in range(root_repeats)]
    n = len(smiles)
    arrays = np.zeros((len(names), n, t, 3 * q), dtype=np.float32)
    counts = np.zeros((len(names), n), dtype=np.int16)
    arrays[0] = np.ascontiguousarray(reference_tokens[:, :t, : 3 * q], dtype=np.float32)
    audit: list[dict] = []
    name_to_index = {name: index for index, name in enumerate(names)}

    for sample_index, smi in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            raise RuntimeError(f"Cannot reconstruct cached molecule {sample_index}: {smi}")
        identity = _canonical_identity(mol)
        counts[0, sample_index] = int(min(int(reference_event_counts[sample_index]), t))

        for rep in range(relabel_repeats):
            rng = np.random.default_rng(stable_seed(variant_seed, "relabel", sample_index, rep))
            permutation = rng.permutation(mol.GetNumAtoms()).astype(int).tolist()
            changed = Chem.RenumberAtoms(mol, permutation)
            accepted = _canonical_identity(changed) == identity
            if not accepted:
                raise RuntimeError("Atom renumbering changed canonical graph identity")
            token, count = tokenize_mol_from_root(changed, dataset, 0, t)
            name = f"atom_relabel_r{rep:02d}"
            arrays[name_to_index[name], sample_index] = token[:, : 3 * q]
            counts[name_to_index[name], sample_index] = count
            audit.append({
                "sample_index": sample_index,
                "variant": name,
                "canonical_identity_preserved": accepted,
                "root": 0,
            })

        # RDKit can very rarely emit a randomized ring SMILES whose parsed
        # E/Z annotation is inverted.  Generate a deterministic candidate pool
        # and accept only candidates that round-trip to the exact canonical
        # isomeric identity; never abort a multi-hour run because one candidate
        # failed the audit.
        random_candidates = list(
            Chem.MolToRandomSmilesVect(
                mol,
                int(max(32, 8 * random_smiles_repeats)),
                int(stable_seed(variant_seed, "random_smiles", sample_index)),
                True,
            )
        )
        random_strings = []
        rejected_random_smiles = 0
        for candidate in random_candidates:
            parsed = Chem.MolFromSmiles(str(candidate))
            if parsed is not None and _canonical_identity(parsed) == identity:
                random_strings.append(str(candidate))
                if len(random_strings) == random_smiles_repeats:
                    break
            else:
                rejected_random_smiles += 1
        if len(random_strings) != random_smiles_repeats:
            raise RuntimeError(
                f"Could not generate {random_smiles_repeats} identity-preserving randomized SMILES "
                f"for sample {sample_index}; accepted={len(random_strings)}, rejected={rejected_random_smiles}"
            )
        for rep, random_smi in enumerate(random_strings):
            changed = Chem.MolFromSmiles(str(random_smi))
            accepted = changed is not None and _canonical_identity(changed) == identity
            if not accepted:
                raise RuntimeError("Randomized SMILES changed canonical graph identity")
            token, count = tokenize_mol_from_root(changed, dataset, 0, t)
            name = f"random_smiles_r{rep:02d}"
            arrays[name_to_index[name], sample_index] = token[:, : 3 * q]
            counts[name_to_index[name], sample_index] = count
            audit.append({
                "sample_index": sample_index,
                "variant": name,
                "canonical_identity_preserved": accepted,
                "root": 0,
                "rejected_candidates_before_acceptance": int(rejected_random_smiles),
            })

        rng = np.random.default_rng(stable_seed(variant_seed, "roots", sample_index))
        roots = [0]
        if mol.GetNumAtoms() > 1:
            alternatives = [value for value in rng.permutation(mol.GetNumAtoms()).astype(int) if value != 0]
            roots.extend(alternatives[: max(0, root_repeats - 1)])
        while len(roots) < root_repeats:
            roots.append(roots[len(roots) % max(1, len(roots))])
        for rep, root in enumerate(roots[:root_repeats]):
            token, count = tokenize_mol_from_root(mol, dataset, int(root), t)
            name = f"root_r{rep:02d}"
            arrays[name_to_index[name], sample_index] = token[:, : 3 * q]
            counts[name_to_index[name], sample_index] = count
            audit.append({
                "sample_index": sample_index,
                "variant": name,
                "canonical_identity_preserved": True,
                "root": int(root),
            })
    return arrays, counts, names, audit


def _generic_to_physical_slot(generic_index: int) -> int:
    current = -1
    for physical in range(72):
        if physical in TASK_SLOT_INDICES:
            continue
        current += 1
        if current == generic_index:
            return physical
    raise ValueError(generic_index)


ATOM_DEPTH_SLOT = _generic_to_physical_slot(18)
BOND_DEPTH_SLOT = _generic_to_physical_slot(15)


def token_depth(token: np.ndarray) -> float:
    if token[0] > 0.5 and ATOM_DEPTH_SLOT < len(token):
        return float(token[ATOM_DEPTH_SLOT])
    if token[1] > 0.5 and BOND_DEPTH_SLOT < len(token):
        return float(token[BOND_DEPTH_SLOT])
    # Very small Q-prefix smoke profiles may truncate the explicit depth slot.
    # Grouping by event type is then the strongest prefix-available control.
    if token[0] > 0.5:
        return -2.0
    if token[1] > 0.5:
        return -3.0
    return -1.0


def apply_order_variant(
    tokens: np.ndarray,
    event_counts: np.ndarray,
    variant: str,
    *,
    permutation_seed: int,
    q: int,
    t: int,
    block_size: int = 2,
) -> np.ndarray:
    """Reorder only valid events; the padded suffix never enters the prefix."""
    if variant not in ORDER_VARIANTS:
        raise ValueError(f"Unknown order variant: {variant}")
    out = np.ascontiguousarray(tokens.copy())
    if variant == "original":
        return out
    for sample_index, raw_count in enumerate(event_counts):
        n_valid = int(min(max(int(raw_count), 0), t))
        if n_valid <= 1:
            continue
        rng = np.random.default_rng(
            stable_seed(permutation_seed, variant, sample_index, q, t, block_size)
        )
        permutation = np.arange(n_valid)
        if variant == "reverse":
            permutation = permutation[::-1]
        elif variant == "full_shuffle":
            permutation = rng.permutation(n_valid)
        elif variant == "block_shuffle":
            blocks = [
                np.arange(start, min(start + block_size, n_valid))
                for start in range(0, n_valid, block_size)
            ]
            permutation = np.concatenate([blocks[index] for index in rng.permutation(len(blocks))])
        elif variant == "within_depth":
            depths = np.asarray([token_depth(tokens[sample_index, pos]) for pos in range(n_valid)])
            for depth in np.unique(depths):
                positions = np.flatnonzero(np.isclose(depths, depth))
                permutation[positions] = rng.permutation(positions)
        elif variant == "local_swap":
            pos = 0
            while pos < n_valid - 1:
                if rng.random() < 0.5:
                    permutation[pos], permutation[pos + 1] = permutation[pos + 1], permutation[pos]
                    pos += 2
                else:
                    pos += 1
        elif variant == "cyclic_shift":
            shift = int(rng.integers(1, n_valid))
            permutation = np.roll(permutation, shift)
        out[sample_index, :n_valid] = tokens[sample_index, permutation]
    return out


def build_order_tensor(
    tokens: np.ndarray,
    event_counts: np.ndarray,
    *,
    permutation_seed: int,
    q: int,
    t: int,
    block_size: int,
) -> tuple[np.ndarray, list[str]]:
    variants = [
        apply_order_variant(
            tokens,
            event_counts,
            name,
            permutation_seed=permutation_seed,
            q=q,
            t=t,
            block_size=block_size,
        )
        for name in ORDER_VARIANTS
    ]
    return np.stack(variants).astype(np.float32), list(ORDER_VARIANTS)
