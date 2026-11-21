"""Deterministic SpaTRACE simulation preprocessing for the QRC validation task.

The code intentionally avoids scanpy/anndata: the two public H5AD files used here
store dense matrices, so h5py is sufficient.  The feature construction mirrors the
relevant SpaTRACE preprocessing operations (GCN smoothing, sender-to-receiver LR
features, and forward pseudotime walks) while using a strict cell-disjoint split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph


def _decode(values: Iterable[bytes]) -> np.ndarray:
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def _read_categorical(group: h5py.Group) -> np.ndarray:
    categories = _decode(group["categories"][:])
    codes = np.asarray(group["codes"][:], dtype=np.int64)
    out = np.full(len(codes), "<NA>", dtype=object)
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out.astype(str)


def read_gene_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_simulation_h5ad(path: str | Path, *, single_cell: bool) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        data = {
            "X": np.asarray(handle["X"][:], dtype=np.float32),
            "obs_names": _decode(handle["obs/_index"][:]),
            "var_names": _decode(handle["var/_index"][:]),
            "cell_types": _read_categorical(handle["obs/Cell Types"]),
            "batch": _read_categorical(handle["obs/batch"]),
            "spatial": np.asarray(handle["obsm/spatial"][:], dtype=np.float32),
        }
        if single_cell:
            data["pseudotime"] = np.asarray(handle["obs/dpt_pseudotime"][:], dtype=np.float32)
            data["umap"] = np.asarray(handle["obsm/X_umap"][:], dtype=np.float32)
    return data


def gcn_weighted_knn_smooth(
    X: np.ndarray,
    *,
    k: int = 5,
    alpha: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Match SpaTRACE's residual weighted-kNN smoothing on the full spatial data."""
    X = np.asarray(X, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean", algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    nonzero = distances[distances > 0]
    sigma = float(np.median(nonzero)) if len(nonzero) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    n = len(X)
    rows = np.repeat(np.arange(n), int(k))
    cols = indices.reshape(-1)
    weights = np.exp(-(distances.reshape(-1) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    adjacency = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float32)
    adjacency = adjacency.maximum(adjacency.T)
    adjacency = adjacency + sparse.eye(n, dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degree[degree == 0] = 1.0
    transition = sparse.diags(1.0 / degree.astype(np.float32)) @ adjacency
    smoothed = (1.0 - float(alpha)) * X + float(alpha) * (transition @ X)
    return np.asarray(smoothed, dtype=np.float32), sigma


def build_lr_features(
    X: np.ndarray,
    coordinates: np.ndarray,
    cell_types: np.ndarray,
    gene_names: np.ndarray,
    ligand_names: list[str],
    receptor_names: list[str],
    sender_types: list[str],
    receiver_types: list[str],
    *,
    radius: float = 5.0,
    min_corr: float = 0.3,
    min_nonzero: int = 10,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Construct the batch-aware SpaTRACE LR feature block (one batch here)."""
    gene_to_index = {name: idx for idx, name in enumerate(gene_names)}
    ligands = [name for name in ligand_names if name in gene_to_index]
    receptors = [name for name in receptor_names if name in gene_to_index]
    lig_idx = [gene_to_index[name] for name in ligands]
    rec_idx = [gene_to_index[name] for name in receptors]

    sender_mask = np.isin(cell_types.astype(str), np.asarray(sender_types, dtype=str))
    receiver_mask = np.isin(cell_types.astype(str), np.asarray(receiver_types, dtype=str))
    graph = radius_neighbors_graph(
        np.asarray(coordinates),
        radius=float(radius),
        mode="distance",
        include_self=True,
        n_jobs=-1,
    ).tocsr()
    if graph.nnz:
        graph.data = np.exp(-graph.data).astype(np.float32, copy=False)
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1)
    row_sum[row_sum == 0] = 1.0
    weights = sparse.diags(1.0 / row_sum.astype(np.float32)) @ graph

    ligand_linear = np.expm1(X[:, lig_idx]).astype(np.float32, copy=False)
    ligand_linear[~sender_mask] = 0.0
    ligand_smooth = np.asarray(weights @ ligand_linear, dtype=np.float32)
    receptor_linear = np.expm1(X[:, rec_idx]).astype(np.float32, copy=False)

    kept: list[tuple[str, str, int, int]] = []
    diagnostics: list[dict] = []
    for li, ligand in enumerate(ligands):
        for ri, receptor in enumerate(receptors):
            x = ligand_smooth[receiver_mask, li]
            y = receptor_linear[receiver_mask, ri]
            x_nonzero = int(np.count_nonzero(x > 0))
            y_nonzero = int(np.count_nonzero(y > 0))
            rho = float("nan")
            if x_nonzero >= int(min_nonzero) and y_nonzero >= int(min_nonzero):
                rho = float(spearmanr(x, y).statistic)
            is_kept = bool(np.isfinite(rho) and rho >= float(min_corr))
            diagnostics.append(
                {
                    "ligand": ligand,
                    "receptor": receptor,
                    "spearman": rho,
                    "ligand_nonzero": x_nonzero,
                    "receptor_nonzero": y_nonzero,
                    "kept": is_kept,
                }
            )
            if is_kept:
                kept.append((ligand, receptor, li, ri))

    kept.sort(key=lambda item: (item[0], item[1]))
    names = [f"{ligand}_to_{receptor}" for ligand, receptor, _, _ in kept]
    features = np.zeros((len(X), len(kept)), dtype=np.float32)
    for column, (_, _, li, ri) in enumerate(kept):
        value = np.log1p(ligand_smooth[:, li] * receptor_linear[:, ri]).astype(np.float32)
        value[~receiver_mask] = 0.0
        features[:, column] = value
    return features, names, diagnostics


def make_cell_disjoint_split(
    cell_types: np.ndarray,
    pseudotime: np.ndarray,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    pseudotime_bins: int = 5,
) -> dict[str, np.ndarray]:
    """Stratify by cell type and within-type pseudotime bin, then split cells once."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one.")
    rng = np.random.default_rng(int(seed))
    result: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for cell_type in np.unique(cell_types):
        ct_indices = np.where(cell_types == cell_type)[0]
        quantiles = np.linspace(0, 1, int(pseudotime_bins) + 1)[1:-1]
        boundaries = np.quantile(pseudotime[ct_indices], quantiles)
        bins = np.digitize(pseudotime[ct_indices], boundaries)
        for bin_id in range(int(pseudotime_bins)):
            indices = ct_indices[bins == bin_id].copy()
            rng.shuffle(indices)
            n = len(indices)
            n_validation = max(1, int(round(fractions[1] * n)))
            n_test = max(1, int(round(fractions[2] * n)))
            if n_validation + n_test >= n:
                n_validation = max(1, n // 5)
                n_test = max(1, n // 5)
            result["validation"].extend(indices[:n_validation].tolist())
            result["test"].extend(indices[n_validation : n_validation + n_test].tolist())
            result["train"].extend(indices[n_validation + n_test :].tolist())
    arrays = {key: np.asarray(sorted(values), dtype=np.int64) for key, values in result.items()}
    all_cells = np.concatenate(list(arrays.values()))
    if len(np.unique(all_cells)) != len(cell_types):
        raise RuntimeError("Cell split is not a disjoint partition.")
    return arrays


def derive_temporal_neighbors(
    umap: np.ndarray,
    pseudotime: np.ndarray,
    cell_indices: np.ndarray,
    *,
    k_primary: int = 5,
    fallback_scan: int = 10,
    out_degree: int = 2,
) -> dict[int, list[int]]:
    """Build forward neighbors within one split only, preventing cross-split leakage."""
    local_umap = np.asarray(umap[cell_indices], dtype=np.float32)
    local_pt = np.asarray(pseudotime[cell_indices], dtype=np.float32)
    squared_distance = np.sum((local_umap[:, None, :] - local_umap[None, :, :]) ** 2, axis=2)
    neighbors: dict[int, list[int]] = {}
    for local_i in range(len(cell_indices)):
        ordered = np.argsort(squared_distance[local_i])
        primary = ordered[: int(k_primary)]
        forward = [(int(j), float(local_pt[j])) for j in primary if j != local_i and local_pt[j] > local_pt[local_i]]
        forward.sort(key=lambda item: -item[1])
        chosen = [j for j, _ in forward[: int(out_degree)]]
        if len(chosen) < int(out_degree):
            for j in ordered[1 : 1 + int(fallback_scan)]:
                j = int(j)
                if local_pt[j] > local_pt[local_i] and j not in chosen:
                    chosen.append(j)
                    if len(chosen) >= int(out_degree):
                        break
        neighbors[local_i] = chosen
    return neighbors


def sample_paths(
    neighbors: dict[int, list[int]],
    global_cell_indices: np.ndarray,
    *,
    path_length: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    paths: list[tuple[int, ...]] = []
    for _ in range(int(repeats)):
        for start in range(len(global_cell_indices)):
            path = [start]
            current = start
            for _step in range(int(path_length)):
                choices = neighbors[current]
                if not choices:
                    break
                current = int(rng.choice(choices))
                path.append(current)
            if len(path) == int(path_length) + 1:
                paths.append(tuple(int(global_cell_indices[index]) for index in path))
    # Stable de-duplication keeps the dataset fixed and avoids repeated trajectory leakage.
    unique = list(dict.fromkeys(paths))
    return np.asarray(unique, dtype=np.int64)


def _fit_positive_scale(values: np.ndarray, train_cells: np.ndarray) -> np.ndarray:
    scale = np.max(values[train_cells], axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return scale


def prepare_fixed_dataset(config: dict, output_path: str | Path, *, force: bool = False) -> Path:
    """Create and persist the fixed 3-event -> next-TG regression dataset."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path
    source = Path(config["source_dir"])
    sc_data = read_simulation_h5ad(source / "simulation_sc_adata.h5ad", single_cell=True)
    st_data = read_simulation_h5ad(source / "simulation_st_adata.h5ad", single_cell=False)
    if not np.array_equal(sc_data["var_names"], st_data["var_names"]):
        raise ValueError("The SpaTRACE single-cell and spatial gene orders differ.")

    smoothed, gcn_sigma = gcn_weighted_knn_smooth(
        st_data["X"], k=int(config["gcn_k"]), alpha=float(config["gcn_alpha"])
    )
    ligand_names = read_gene_list(source / "simulation_ligand.txt")
    receptor_names = read_gene_list(source / "simulation_receptor.txt")
    tf_names = sorted(read_gene_list(source / "simulation_tf.txt"), key=int, reverse=True)
    sender_types = read_gene_list(source / "simulation_sender.txt")
    receiver_types = read_gene_list(source / "simulation_receiver.txt")
    lr_all, lr_names, lr_diagnostics = build_lr_features(
        smoothed,
        st_data["spatial"],
        st_data["cell_types"],
        st_data["var_names"],
        ligand_names,
        receptor_names,
        sender_types,
        receiver_types,
        radius=float(config["lr_radius"]),
        min_corr=float(config["lr_min_corr"]),
    )

    st_name_to_index = {name: idx for idx, name in enumerate(st_data["obs_names"])}
    receiver_rows = np.asarray([st_name_to_index[name] for name in sc_data["obs_names"]], dtype=np.int64)
    smoothed_receiver = smoothed[receiver_rows]
    lr_receiver = lr_all[receiver_rows]
    excluded = set(ligand_names) | set(receptor_names) | set(tf_names)
    target_names = sorted([name for name in sc_data["var_names"] if name not in excluded], key=int, reverse=True)
    gene_to_index = {name: idx for idx, name in enumerate(sc_data["var_names"])}
    tf_idx = np.asarray([gene_to_index[name] for name in tf_names], dtype=np.int64)
    target_idx = np.asarray([gene_to_index[name] for name in target_names], dtype=np.int64)

    cell_split = make_cell_disjoint_split(
        sc_data["cell_types"],
        sc_data["pseudotime"],
        seed=int(config["split_seed"]),
        fractions=tuple(config["split_fractions"]),
        pseudotime_bins=int(config["pseudotime_bins"]),
    )
    train_cells = cell_split["train"]
    tf_scale = _fit_positive_scale(smoothed_receiver[:, tf_idx], train_cells)
    lr_scale = _fit_positive_scale(lr_receiver, train_cells)
    target_scale = _fit_positive_scale(smoothed_receiver[:, target_idx], train_cells)
    tf_block = smoothed_receiver[:, tf_idx] / tf_scale
    lr_block = lr_receiver / lr_scale
    target_block = smoothed_receiver[:, target_idx] / target_scale
    cell_features = np.concatenate([tf_block, lr_block, target_block], axis=1).astype(np.float32)
    feature_names = tf_names + lr_names + target_names
    if cell_features.shape[1] != int(config["required_input_dim"]):
        raise ValueError(
            f"Expected {config['required_input_dim']} input features, got {cell_features.shape[1]} "
            f"({len(tf_names)} TF + {len(lr_names)} LR + {len(target_names)} TG)."
        )

    split_paths: dict[str, np.ndarray] = {}
    for offset, split_name in enumerate(("train", "validation", "test")):
        neighbors = derive_temporal_neighbors(
            sc_data["umap"],
            sc_data["pseudotime"],
            cell_split[split_name],
            k_primary=int(config["k_primary"]),
            fallback_scan=int(config["fallback_scan"]),
            out_degree=int(config["out_degree"]),
        )
        split_paths[split_name] = sample_paths(
            neighbors,
            cell_split[split_name],
            path_length=int(config["path_length"]),
            repeats=int(config["path_repeats"][split_name]),
            seed=int(config["path_seed"]) + offset,
        )
        if not len(split_paths[split_name]):
            raise RuntimeError(f"No complete paths were sampled for {split_name}.")

    metadata = {
        "task": "three pseudotime events predict fourth-event target-gene expression",
        "source": "SpaTRACE public simulation data",
        "source_revision": config["source_revision"],
        "split_policy": "cell-disjoint, stratified by cell type and pseudotime bin",
        "feature_order": "TF, LR-pair, TG",
        "dimensions": {
            "tf": len(tf_names),
            "lr_pair": len(lr_names),
            "target_gene": len(target_names),
            "input": int(cell_features.shape[1]),
            "output": len(target_names),
        },
        "gcn_sigma": gcn_sigma,
        "path_counts": {name: int(len(paths)) for name, paths in split_paths.items()},
        "cell_counts": {name: int(len(indices)) for name, indices in cell_split.items()},
        "lr_diagnostics": lr_diagnostics,
        "config": config,
    }
    payload: dict[str, np.ndarray] = {
        "feature_names": np.asarray(feature_names),
        "tf_names": np.asarray(tf_names),
        "lr_names": np.asarray(lr_names),
        "target_names": np.asarray(target_names),
        "tf_scale": tf_scale,
        "lr_scale": lr_scale,
        "target_scale": target_scale,
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    for split_name in ("train", "validation", "test"):
        paths = split_paths[split_name]
        payload[f"{split_name}_paths"] = paths
        payload[f"{split_name}_cell_indices"] = cell_split[split_name]
        payload[f"{split_name}_tokens"] = cell_features[paths[:, :-1]]
        payload[f"{split_name}_targets"] = target_block[paths[:, -1]]
        payload[f"{split_name}_pseudotime"] = sc_data["pseudotime"][paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def fit_linear_angle_normalizer(train_tokens: np.ndarray, angle_scale: float) -> dict[str, np.ndarray | float]:
    """Fit an affine, train-only min-max map; no tanh and no clipping."""
    flat = np.asarray(train_tokens, dtype=np.float32).reshape(-1, train_tokens.shape[-1])
    lower = np.min(flat, axis=0).astype(np.float32)
    upper = np.max(flat, axis=0).astype(np.float32)
    width = upper - lower
    width[width == 0] = 1.0
    slope = (2.0 * float(angle_scale) / width).astype(np.float32)
    intercept = (-float(angle_scale) - slope * lower).astype(np.float32)
    return {"slope": slope, "intercept": intercept, "angle_scale": float(angle_scale)}


def apply_linear_angle_normalizer(tokens: np.ndarray, normalizer: dict) -> np.ndarray:
    return (np.asarray(tokens, dtype=np.float32) * normalizer["slope"] + normalizer["intercept"]).astype(np.float32)

