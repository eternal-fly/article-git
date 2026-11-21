"""Shared local-BFS token design and label-free Q/T collision scan.

The general and chemistry notebooks import this exact module.  The only token
design switch exposed to them is ``task_aware``:

* False: twelve reserved chemistry slots are exactly zero; generic event
  features are local/prefix-causal and avoid full-graph descriptors.
* True: the same slots contain a fixed, label-free, dataset-directed panel.

Traversal, prefix ring-closure handling, normalization, slot order, padding,
Q-prefix policy, and T-prefix policy are otherwise identical.  The general
variant deliberately avoids canonical ranks, full-graph ring sizes, neighbor
aggregate statistics and molecule-level descriptors so that it is a better
proxy for dynamic molecular event streams.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


RDLogger.DisableLog("rdApp.warning")

SCHEMA_VERSION = "local-bfs-prefix-causal-task-slots-v2"
MAX_Q = 24
MAX_T = 32
MAX_TOKEN_DIM = 3 * MAX_Q
TASK_SLOT_INDICES = tuple(range(5, MAX_TOKEN_DIM, 6))  # 5, 11, ..., 71
TASK_DIM = len(TASK_SLOT_INDICES)
GENERIC_DIM = MAX_TOKEN_DIM - TASK_DIM
HALOGENS = {9, 17, 35, 53}


@dataclass(frozen=True)
class ScanConfig:
    dataset: str
    task_aware: bool
    data_dir: str
    output_dir: str
    q_values: tuple[int, ...] = tuple(range(1, MAX_Q + 1))
    t_values: tuple[int, ...] = tuple(range(1, MAX_T + 1))

    def normalized(self) -> "ScanConfig":
        dataset = self.dataset.upper()
        if dataset not in {"BACE", "BBBP"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        q_values = tuple(sorted({int(value) for value in self.q_values}))
        t_values = tuple(sorted({int(value) for value in self.t_values}))
        if not q_values or min(q_values) < 1 or max(q_values) > MAX_Q:
            raise ValueError(f"q_values must lie in [1, {MAX_Q}]")
        if not t_values or min(t_values) < 1 or max(t_values) > MAX_T:
            raise ValueError(f"t_values must lie in [1, {MAX_T}]")
        return ScanConfig(
            dataset=dataset,
            task_aware=bool(self.task_aware),
            data_dir=str(Path(self.data_dir).resolve()),
            output_dir=str(Path(self.output_dir).resolve()),
            q_values=q_values,
            t_values=t_values,
        )


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _z(atom):
    return min(1.0, atom.GetAtomicNum() / 80.0)


def _degree(atom):
    return min(1.0, atom.GetDegree() / 6.0)


def _charge(atom):
    return float(np.clip(atom.GetFormalCharge() / 3.0, -1.0, 1.0))


def _hydrogens(atom):
    return min(1.0, atom.GetTotalNumHs(includeNeighbors=True) / 4.0)


def _hetero(atom):
    return float(atom.GetAtomicNum() not in (1, 6))


def _hydrophobic_like(atom):
    return float(atom.GetFormalCharge() == 0 and (atom.GetAtomicNum() == 6 or atom.GetAtomicNum() in HALOGENS))


def _hbd_like(atom):
    return float(
        atom.GetAtomicNum() in (7, 8, 16)
        and atom.GetTotalNumHs(includeNeighbors=True) > 0
        and atom.GetFormalCharge() >= 0
    )


def _hba_like(atom):
    z = atom.GetAtomicNum()
    if z not in (7, 8, 16) or atom.GetFormalCharge() > 0:
        return 0.0
    if z == 7 and atom.GetIsAromatic() and atom.GetTotalNumHs(includeNeighbors=True) > 0:
        return 0.0
    return 1.0


def _positive_ionizable_like(atom):
    if atom.GetFormalCharge() > 0:
        return 1.0
    return float(
        atom.GetAtomicNum() == 7
        and not atom.GetIsAromatic()
        and atom.GetDegree() >= 3
        and atom.GetFormalCharge() >= 0
    )


def _rotatable_bond_like(bond):
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE or bond.IsInRing():
        return 0.0
    return float(bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1)


def _bond_onehot(bond):
    if bond is None:
        return [0.0] * 4
    kind = bond.GetBondType()
    return [
        float(kind == Chem.rdchem.BondType.SINGLE),
        float(kind == Chem.rdchem.BondType.DOUBLE),
        float(kind == Chem.rdchem.BondType.TRIPLE),
        float(kind not in {
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
        }),
    ]


def _ring_sizes(atom):
    sizes = [
        len(ring)
        for ring in atom.GetOwningMol().GetRingInfo().AtomRings()
        if atom.GetIdx() in ring
    ]
    return (min(sizes), max(sizes), len(sizes)) if sizes else (0, 0, 0)


def _hybrid_flags(atom):
    value = atom.GetHybridization()
    return [
        float(value == Chem.rdchem.HybridizationType.SP),
        float(value == Chem.rdchem.HybridizationType.SP2),
        float(value == Chem.rdchem.HybridizationType.SP3),
    ]


def _element_flags(atom):
    z = atom.GetAtomicNum()
    return [float(z == value) for value in (6, 7, 8, 16, 15)] + [float(z in HALOGENS)]


def _neighbor_element_fractions(neighbors):
    return [
        _mean([float(atom.GetAtomicNum() == value) for atom in neighbors])
        for value in (6, 7, 8, 16, 15)
    ] + [_mean([float(atom.GetAtomicNum() in HALOGENS) for atom in neighbors])]


def _canonical_ranks(mol):
    return list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=True))


def _priority(atom, ranks):
    # Identical traversal priority for general and chemistry variants.
    return (atom.GetDegree(), int(atom.GetIsAromatic()), atom.GetAtomicNum(), -ranks[atom.GetIdx()])


def task_feature_names(dataset):
    dataset = dataset.upper()
    if dataset == "BACE":
        return (
            "basic_center_fraction",
            "hba_fraction",
            "hbd_fraction",
            "aromatic_atom_fraction",
            "hydrophobic_atom_fraction",
            "hetero_atom_fraction",
            "ring_atom_fraction",
            "rotatable_bond_fraction",
            "formal_charge_burden",
            "mol_logp_scaled",
            "tpsa_scaled",
            "molecular_weight_scaled",
        )
    if dataset == "BBBP":
        return (
            "molecular_weight_scaled",
            "mol_logp_scaled",
            "tpsa_scaled",
            "hbd_fraction",
            "hba_fraction",
            "rotatable_bond_fraction",
            "aromatic_atom_fraction",
            "ring_atom_fraction",
            "formal_charge_burden",
            "hetero_atom_fraction",
            "hydrophobic_atom_fraction",
            "fraction_csp3",
        )
    raise ValueError(dataset)


def _task_features(mol, dataset):
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    n_atoms = max(1, len(atoms))
    n_bonds = max(1, len(bonds))
    common = {
        "molecular_weight_scaled": _clip01(Descriptors.MolWt(mol) / 700.0),
        "mol_logp_scaled": _clip01((Crippen.MolLogP(mol) + 5.0) / 15.0),
        "tpsa_scaled": _clip01(rdMolDescriptors.CalcTPSA(mol) / 200.0),
        "hbd_fraction": _clip01(sum(_hbd_like(atom) for atom in atoms) / n_atoms),
        "hba_fraction": _clip01(sum(_hba_like(atom) for atom in atoms) / n_atoms),
        "rotatable_bond_fraction": _clip01(sum(_rotatable_bond_like(bond) for bond in bonds) / n_bonds),
        "aromatic_atom_fraction": _clip01(sum(float(atom.GetIsAromatic()) for atom in atoms) / n_atoms),
        "ring_atom_fraction": _clip01(sum(float(atom.IsInRing()) for atom in atoms) / n_atoms),
        "formal_charge_burden": _clip01(sum(abs(atom.GetFormalCharge()) for atom in atoms) / 3.0),
        "hetero_atom_fraction": _clip01(sum(_hetero(atom) for atom in atoms) / n_atoms),
        "hydrophobic_atom_fraction": _clip01(sum(_hydrophobic_like(atom) for atom in atoms) / n_atoms),
        "fraction_csp3": _clip01(rdMolDescriptors.CalcFractionCSP3(mol)),
        "basic_center_fraction": _clip01(sum(_positive_ionizable_like(atom) for atom in atoms) / n_atoms),
    }
    return np.asarray([common[name] for name in task_feature_names(dataset)], dtype=np.float32)


def _assemble_token(generic_values, task_values):
    generic_values = list(generic_values)[:GENERIC_DIM]
    generic_values += [0.0] * (GENERIC_DIM - len(generic_values))
    task_values = list(task_values)[:TASK_DIM]
    task_values += [0.0] * (TASK_DIM - len(task_values))
    output = np.zeros(MAX_TOKEN_DIM, dtype=np.float32)
    generic_index = 0
    task_index = 0
    for index in range(MAX_TOKEN_DIM):
        if index in TASK_SLOT_INDICES:
            output[index] = task_values[task_index]
            task_index += 1
        else:
            output[index] = generic_values[generic_index]
            generic_index += 1
    return output


def _atom_generic(atom, depth, parent_bond):
    parent_exists = float(parent_bond is not None)
    features = [
        _z(atom), _degree(atom), _charge(atom), _hydrogens(atom),
        float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        *_hybrid_flags(atom), *_element_flags(atom),
        _hetero(atom), min(1.0, depth / 16.0), *_bond_onehot(parent_bond),
        parent_exists,
        float(atom.GetIsotope() > 0), float(atom.GetNoImplicit()),
        min(1.0, atom.GetNumRadicalElectrons() / 2.0), min(1.0, atom.GetMass() / 210.0),
        min(1.0, atom.GetTotalValence() / 8.0),
        min(1.0, atom.GetValence(Chem.ValenceType.EXPLICIT) / 8.0),
        min(1.0, atom.GetValence(Chem.ValenceType.IMPLICIT) / 8.0),
        min(1.0, atom.GetTotalDegree() / 6.0),
        min(1.0, atom.GetTotalNumHs(includeNeighbors=False) / 4.0),
        *[float(atom.GetDegree() == degree) for degree in range(6)],
        float(atom.GetAtomicNum() == 5), float(atom.GetAtomicNum() == 14),
        float(atom.GetAtomicNum() == 34),
        float(atom.GetAtomicNum() not in {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}),
        float(atom.GetFormalCharge() > 0), float(atom.GetFormalCharge() < 0),
        float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
        - float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
    ]
    return [1.0, 0.0, 0.0] + features


def _bond_onehot_summary(bonds):
    bonds = list(bonds)
    return [_mean([_bond_onehot(bond)[index] for bond in bonds]) for index in range(4)]


def _bond_generic(bond, src, dst, depth, ring_closure):
    kind = bond.GetBondType()
    order = 1.0 if kind == Chem.rdchem.BondType.TRIPLE else (
        2.0 / 3.0 if kind == Chem.rdchem.BondType.DOUBLE else 1.0 / 3.0
    )
    features = [
        order, float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
        _z(src), _z(dst), _degree(src), _degree(dst),
        _hetero(src), _hetero(dst),
        _charge(src), _charge(dst), _hydrogens(src), _hydrogens(dst),
        min(1.0, depth / 16.0), *_bond_onehot(bond),
        abs(_z(src) - _z(dst)), abs(_degree(src) - _degree(dst)),
        abs(_charge(src) - _charge(dst)),
        float(src.GetDegree() == 1), float(dst.GetDegree() == 1),
        float(src.GetDegree() >= 3), float(dst.GetDegree() >= 3),
        min(1.0, src.GetMass() / 210.0), min(1.0, dst.GetMass() / 210.0),
        min(1.0, src.GetTotalValence() / 8.0), min(1.0, dst.GetTotalValence() / 8.0),
        float(ring_closure),
        float(src.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        float(dst.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        *_hybrid_flags(src), *_hybrid_flags(dst),
        *_element_flags(src), *_element_flags(dst),
        float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE),
        float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ),
        float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOCIS),
        float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOTRANS),
        float(src.GetFormalCharge() > 0), float(src.GetFormalCharge() < 0),
        float(dst.GetFormalCharge() > 0), float(dst.GetFormalCharge() < 0),
    ]
    return [0.0, 1.0, 0.0] + features


def _pad_token():
    # Task slots remain zero for padding in both variants.
    return _assemble_token([0.0, 0.0, 1.0], np.zeros(TASK_DIM, dtype=np.float32))


def tokenize_mol(mol, dataset, task_aware, max_events=MAX_T):
    if mol is None or mol.GetNumAtoms() == 0:
        return np.stack([_pad_token()] * max_events), 0
    dataset = dataset.upper()
    root = 0
    task_values = _task_features(mol, dataset) if task_aware else np.zeros(TASK_DIM, dtype=np.float32)
    visited = {root}
    emitted_edges = set()
    queue = deque([(root, 0)])
    events = [_assemble_token(_atom_generic(mol.GetAtomWithIdx(root), 0, None), task_values)]
    while queue and len(events) < max_events:
        current, depth = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        neighbors = sorted(atom.GetNeighbors(), key=lambda value: value.GetIdx())
        for neighbor in neighbors:
            target = neighbor.GetIdx()
            edge = tuple(sorted((current, target)))
            if edge in emitted_edges:
                continue
            emitted_edges.add(edge)
            bond = mol.GetBondBetweenAtoms(current, target)
            closure = target in visited
            events.append(_assemble_token(_bond_generic(bond, atom, neighbor, depth + 1, closure), task_values))
            if len(events) >= max_events:
                break
            if not closure:
                visited.add(target)
                events.append(_assemble_token(_atom_generic(neighbor, depth + 1, bond), task_values))
                queue.append((target, depth + 1))
                if len(events) >= max_events:
                    break
    event_count = len(events)
    events.extend([_pad_token()] * (max_events - event_count))
    return np.stack(events).astype(np.float32), event_count


def _dataset_csv(dataset, data_dir):
    data_dir = Path(data_dir)
    if dataset == "BACE":
        return data_dir / "bace" / "raw" / "bace.csv", "mol"
    return data_dir / "bbbp" / "raw" / "BBBP.csv", "smiles"


def load_unique_molecules(dataset, data_dir):
    path, smiles_column = _dataset_csv(dataset, data_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    canonical_to_mol = {}
    invalid = 0
    for value in frame[smiles_column].astype(str):
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        canonical_to_mol.setdefault(canonical, mol)
    canonical = sorted(canonical_to_mol)
    molecules = [canonical_to_mol[value] for value in canonical]
    metadata = {
        "source_csv": str(path.resolve()),
        "source_rows": int(len(frame)),
        "valid_rows": int(len(frame) - invalid),
        "invalid_rows": int(invalid),
        "unique_canonical_molecules": int(len(molecules)),
        "duplicate_valid_rows": int(len(frame) - invalid - len(molecules)),
    }
    return canonical, molecules, metadata


def _digest(array):
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=16).digest()


def _collision_metrics(tokens, q, t):
    groups = defaultdict(int)
    for molecule_tokens in tokens:
        groups[_digest(molecule_tokens[:t, : 3 * q])] += 1
    sizes = np.asarray(list(groups.values()), dtype=int)
    colliding = sizes[sizes > 1]
    n_molecules = int(len(tokens))
    colliding_molecules = int(colliding.sum()) if len(colliding) else 0
    return {
        "q": int(q),
        "token_dim": int(3 * q),
        "t": int(t),
        "n_molecules": n_molecules,
        "unique_representations": int(len(groups)),
        "collision_groups": int(len(colliding)),
        "colliding_molecules": colliding_molecules,
        "collision_rate": float(colliding_molecules / max(1, n_molecules)),
        "max_collision_group_size": int(sizes.max(initial=0)),
    }


def _plot_heatmap(table, output_dir, dataset, task_aware):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    pivot = table.pivot(index="q", columns="t", values="collision_rate").sort_index(ascending=False)
    width = 180 / 25.4
    height = 105 / 25.4
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns))[::2], [str(value) for value in pivot.columns[::2]])
    ax.set_yticks(np.arange(len(pivot.index))[::2], [str(value) for value in pivot.index[::2]])
    ax.set_xlabel("Molecular events, T")
    ax.set_ylabel("Qubits, Q  (token dimension = 3Q)")
    mode = "task-aware chemistry" if task_aware else "task-independent general"
    ax.set_title(f"{dataset}: {mode} token collisions", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Fraction of molecules in collision groups")
    stem = Path(output_dir) / "collision_rate_Q_T_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def run_collision_scan(config):
    config = config.normalized()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical, molecules, source_metadata = load_unique_molecules(config.dataset, config.data_dir)
    encoded = []
    event_counts = []
    for mol in molecules:
        tokens, count = tokenize_mol(mol, config.dataset, config.task_aware, max_events=max(config.t_values))
        encoded.append(tokens)
        event_counts.append(count)
    token_array = np.stack(encoded).astype(np.float32)
    rows = [
        {
            "dataset": config.dataset,
            "task_aware": config.task_aware,
            "schema_version": SCHEMA_VERSION,
            **_collision_metrics(token_array, q, t),
        }
        for q in config.q_values
        for t in config.t_values
    ]
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "collision_scan_Q_T.csv", index=False)
    first_clean = []
    for q, subset in table.groupby("q"):
        clean = subset.loc[subset.collision_rate == 0.0, "t"]
        first_clean.append({
            "q": int(q),
            "token_dim": int(3 * q),
            "first_collision_free_t": int(clean.min()) if len(clean) else None,
            "minimum_collision_rate": float(subset.collision_rate.min()),
        })
    pd.DataFrame(first_clean).to_csv(output_dir / "collision_thresholds_by_Q.csv", index=False)
    np.savez_compressed(
        output_dir / "tokens_Q24_Tmax.npz",
        tokens=token_array,
        event_counts=np.asarray(event_counts, dtype=np.int16),
        canonical_smiles=np.asarray(canonical),
    )
    metadata = {
        "config": asdict(config),
        "schema_version": SCHEMA_VERSION,
        "max_token_dim": MAX_TOKEN_DIM,
        "task_slot_indices_zero_based": list(TASK_SLOT_INDICES),
        "task_feature_names": list(task_feature_names(config.dataset)),
        "general_variant_task_slots": "all zeros",
        "chemistry_variant_task_slots": "dataset-directed, label-free descriptors",
        "collision_definition": "exact float32 equality among distinct canonical molecules",
        "source": source_metadata,
        "event_count": {
            "minimum": int(np.min(event_counts)),
            "median": float(np.median(event_counts)),
            "maximum": int(np.max(event_counts)),
        },
    }
    (output_dir / "scan_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    figure_contract = {
        "core_conclusion": "Show how exact molecular-representation collisions change with event count T and token width 3Q.",
        "evidence": "One Q-by-T heatmap of the fraction of unique canonical molecules belonging to exact float32 collision groups.",
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "source_data": "collision_scan_Q_T.csv",
        "review_risks": [
            "Q changes the accessible prefix of one fixed 72-slot schema.",
            "T changes the accessible prefix of one fixed local-BFS event stream.",
            "Exact collisions do not measure near-collisions or downstream classifier separability.",
        ],
    }
    (output_dir / "figure_contract.json").write_text(json.dumps(figure_contract, indent=2), encoding="utf-8")
    _plot_heatmap(table, output_dir, config.dataset, config.task_aware)
    return table, metadata
