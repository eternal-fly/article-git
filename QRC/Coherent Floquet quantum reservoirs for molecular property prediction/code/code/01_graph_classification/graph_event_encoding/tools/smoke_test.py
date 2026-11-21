#!/usr/bin/env python3
"""CPU-only deterministic unit smoke; it does not launch a scan or train heads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validity_core.canonical_tokens import CANONICAL_SCHEMA_VERSION, tokenize_canonical_general, tokenize_canonical_multi_root_general
from validity_core.feature_engine import compute_crc3_z_zz_padding, compute_crc3_z_zz_tensor
from validity_core.token_variants import ORDER_VARIANTS, apply_order_variant


def main() -> None:
    smiles = "C1=CC=CC=C1O"
    mol = Chem.MolFromSmiles(smiles)
    first, count, provenance = tokenize_canonical_general(mol, "BACE", 12)
    second, count2, _ = tokenize_canonical_general(Chem.MolFromSmiles("Oc1ccccc1"), "BACE", 12)
    assert CANONICAL_SCHEMA_VERSION in provenance["schema_version"]
    assert count == count2 and np.array_equal(first, second)
    roots, root_counts, _ = tokenize_canonical_multi_root_general(mol, "BACE", 12, 4)
    assert roots.shape == (4, 12, 72) and root_counts.shape == (4,)
    batch = first[None, :, :12]
    counts = np.asarray([count])
    for variant in ORDER_VARIANTS:
        changed = apply_order_variant(
            batch, counts, variant, permutation_seed=1000, q=4, t=12, block_size=2
        )
        assert changed.shape == batch.shape
        assert np.allclose(
            np.sort(changed[0, :count].reshape(count, -1), axis=0),
            np.sort(batch[0, :count].reshape(count, -1), axis=0),
        )
    rng = np.random.default_rng(20260712)
    variants = rng.normal(size=(2, 3, 5, 12)).astype(np.float32)
    crc3 = compute_crc3_z_zz_tensor(
        variants, q=4, layers=2, seed=10, device="cpu"
    )
    crc3_repeat = compute_crc3_z_zz_tensor(
        variants, q=4, layers=2, seed=10, device="cpu"
    )
    assert crc3.shape == (2, 3, 10)
    assert np.array_equal(crc3, crc3_repeat)
    typed = variants[0]
    zero = typed.copy()
    counts_small = np.asarray([5, 3, 1], dtype=np.int16)
    for index, length in enumerate(counts_small):
        zero[index, int(length) :] = 0.0
    crc3_padding = compute_crc3_z_zz_padding(
        typed,
        zero,
        counts_small,
        q=4,
        t=5,
        layers=2,
        seed=10,
        device="cpu",
    )
    assert crc3_padding.shape == (3, 3, 10)
    assert np.isfinite(crc3_padding).all()
    print("graph-stream validity smoke: PASS")


if __name__ == "__main__":
    main()
