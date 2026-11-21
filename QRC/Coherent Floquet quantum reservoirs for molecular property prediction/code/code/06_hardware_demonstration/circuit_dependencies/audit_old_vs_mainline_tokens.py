#!/usr/bin/env python3
"""Reproduce the historical-versus-mainline balanced-400 token audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare_dataset import DATASET_OUTPUT, sha256_file


HERE = Path(__file__).resolve().parent
QRC_ROOT = HERE.parents[1]
OUTPUT = HERE / "old_vs_mainline_token_audit.json"
OLD_BUNDLES = {
    "historical_q6_t6": (
        QRC_ROOT
        / "simulation/experiment/6T6Q3L"
        / "results_qrc_quafu_BACE_bal400_Dongling6_T6_L3_W0p8_g0p84"
        / "bace_bal400_tokens_Dongling6_T6_L3_W0p8_g0p84.npz"
    ),
    "historical_q7_t7": (
        QRC_ROOT
        / "simulation/experiment/7T7Q2L"
        / "results_qrc_quafu_BACE_bal400_Dongling7_T7_L2_W0p8_g0p84"
        / "bace_bal400_tokens_Dongling7_T7_L2_W0p8_g0p84.npz"
    ),
    "historical_q8_t8": (
        QRC_ROOT
        / "simulation/experiment/8T8Q3L"
        / "results_qrc_quafu_BACE_bal400_Dongling8_T8_L3_W0p8_g0p84"
        / "bace_bal400_tokens_Dongling8_T8_L3_W0p8_g0p84.npz"
    ),
}


def scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def load_new() -> dict:
    with np.load(DATASET_OUTPUT, allow_pickle=False) as package:
        return {
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "smiles": package["smiles"].astype(str),
            "tokens": package["tokens"].astype(np.float32),
            "schema": scalar_text(package["token_schema_version"]),
            "metadata": json.loads(scalar_text(package["metadata_json"])),
        }


def main() -> None:
    new = load_new()
    rows = []
    for name, path in OLD_BUNDLES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as package:
            old_indices = package["indices"].astype(np.int64)
            old_labels = package["labels"].astype(np.int64)
            old_smiles = package["smiles"].astype(str)
            old_tokens = package["tokens"].astype(np.float32)
            config = json.loads(scalar_text(package["config_json"]))
            metadata = json.loads(scalar_text(package["metadata_json"]))

        old_row = {int(index): row for row, index in enumerate(old_indices)}
        same_index_set = set(map(int, old_indices)) == set(map(int, new["indices"]))
        if not same_index_set:
            raise AssertionError(f"{name}: balanced subsets differ")
        aligned = np.asarray([old_row[int(index)] for index in new["indices"]])
        cropped = old_tokens[aligned, :6, :18]
        exact = np.equal(cropped, new["tokens"])
        rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "stored_shape": list(old_tokens.shape),
                "same_400_index_set": True,
                "same_row_order": bool(np.array_equal(old_indices, new["indices"])),
                "labels_match_after_index_alignment": bool(
                    np.array_equal(old_labels[aligned], new["labels"])
                ),
                "smiles_match_after_index_alignment": bool(
                    np.array_equal(old_smiles[aligned], new["smiles"])
                ),
                "cropped_q6_t6_scalar_exact_fraction": float(exact.mean()),
                "cropped_q6_t6_row_exact_count": int(
                    np.all(exact, axis=(1, 2)).sum()
                ),
                "historical_token_scheme": config.get("tokens", {}).get("scheme"),
                "historical_config": config,
                "historical_metadata": metadata,
            }
        )

    report = {
        "conclusion": (
            "The historical bundles reuse the same balanced-400 molecules but "
            "do not implement the mainline canonical-v3 token representation."
        ),
        "new_bundle": {
            "path": str(DATASET_OUTPUT.resolve()),
            "sha256": sha256_file(DATASET_OUTPUT),
            "shape": list(new["tokens"].shape),
            "schema": new["schema"],
            "global_descriptors": new["metadata"]["global_descriptors"],
            "task_guided_traversal": new["metadata"]["task_guided_traversal"],
        },
        "historical_bundles": rows,
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[saved] {OUTPUT}")


if __name__ == "__main__":
    main()
