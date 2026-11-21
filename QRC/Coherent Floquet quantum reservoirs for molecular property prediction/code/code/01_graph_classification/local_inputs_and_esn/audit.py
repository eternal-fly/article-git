"""Fail-closed audit for protocol identity, coverage, and token semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from esn_features import load_esn_features
from protocol import (
    DATASETS,
    G,
    GLOBAL_TOKEN_SCHEMA_VERSION,
    LAYERS,
    OBSERVABLES,
    PROTOCOL_VERSION,
    Q_VALUES,
    SEEDS,
    T_VALUES,
    TOKEN_SCHEMA_VERSION,
    TOKEN_VARIANTS,
    W,
    esn_feature_path,
    esn_widths,
    qrc_bank_path,
    split_path,
    token_cache_path,
)
from qrc_features import load_qrc_bank, mix_suffix_bank
from summarize import collect_esn, collect_qrc


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_audit(*, strict: bool = False) -> dict:
    report: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "g": G,
        "W": W,
        "layers": LAYERS,
        "observables": OBSERVABLES,
        "seeds": list(SEEDS),
        "errors": [],
    }
    token_rows = []
    for dataset in DATASETS:
        arrays = {}
        for variant in TOKEN_VARIANTS:
            path = token_cache_path(dataset, variant)
            _assert(path.exists(), f"missing token cache {path}")
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            expected_schema = (
                TOKEN_SCHEMA_VERSION if variant == "canonical_general" else GLOBAL_TOKEN_SCHEMA_VERSION
            )
            _assert(metadata["protocol_version"] == PROTOCOL_VERSION, f"stale token {path}")
            _assert(metadata["schema_version"] == expected_schema, f"wrong schema {path}")
            _assert(metadata["label_used_by_tokenizer"] is False, f"label leakage marker {path}")
            with np.load(path, allow_pickle=False) as archive:
                arrays[variant] = archive["tokens"].astype(np.float32)
                labels = archive["labels"]
                counts = archive["event_counts"]
                stored_hash = str(archive["tokens_sha256"].item())
            _assert(stored_hash == metadata["tokens_sha256"], f"token hash mismatch {path}")
            _assert(tuple(arrays[variant].shape[1:]) == (30, 72), f"token shape {path}")
            _assert(len(labels) == len(counts) == len(arrays[variant]), f"row mismatch {path}")
            token_rows.append({"dataset": dataset, "variant": variant, "rows": len(labels)})
        slots = np.arange(5, 72, 6)
        generic = np.setdiff1d(np.arange(72), slots)
        general = arrays["canonical_general"]
        global_prior = arrays["canonical_global_prior"]
        _assert(np.array_equal(general[:, :, generic], global_prior[:, :, generic]), f"generic drift {dataset}")
        _assert(np.all(general[:, :, slots] == 0.0), f"nonzero general prior slots {dataset}")
        _assert(np.any(global_prior[:, :, slots] != 0.0), f"empty global prior {dataset}")
        for seed in SEEDS:
            payload = json.loads(split_path(dataset, seed).read_text(encoding="utf-8"))
            _assert(payload["protocol_version"] == PROTOCOL_VERSION, f"stale split {dataset}/{seed}")
            union = payload["train_indices"] + payload["val_indices"] + payload["test_indices"]
            _assert(len(union) == len(set(union)) == len(general), f"bad split partition {dataset}/{seed}")
    report["tokens"] = token_rows

    qrc_total = qrc_valid = 0
    esn_total = esn_valid = 0
    for dataset in DATASETS:
        for variant in TOKEN_VARIANTS:
            with np.load(token_cache_path(dataset, variant), allow_pickle=False) as archive:
                n = len(archive["labels"])
                token_sha = str(archive["tokens_sha256"].item())
            for q in Q_VALUES:
                for seed in SEEDS:
                    for t in T_VALUES:
                        qrc_total += 1
                        path = qrc_bank_path(dataset, variant, q, t, seed)
                        if path.exists():
                            bank = load_qrc_bank(
                                dataset, variant, q, t, seed, n_samples=n, token_sha256=token_sha
                            )
                            _assert(np.array_equal(mix_suffix_bank(bank, 0.0), bank[0]), f"p0 mismatch {path}")
                            qrc_valid += 1
                        for width in esn_widths(q):
                            esn_total += 1
                            ep = esn_feature_path(dataset, variant, q, t, width, seed)
                            if ep.exists():
                                load_esn_features(
                                    dataset, variant, q, t, width, seed, n_samples=n, token_sha256=token_sha
                                )
                                esn_valid += 1
    report["qrc_feature_coverage"] = {"valid": qrc_valid, "expected": qrc_total}
    report["esn_feature_coverage"] = {"valid": esn_valid, "expected": esn_total}
    qrc_results = collect_qrc(strict=False)
    esn_results = collect_esn(strict=False)
    report["qrc_result_rows"] = len(qrc_results)
    report["qrc_result_rows_expected"] = qrc_total * 2
    report["esn_result_rows"] = len(esn_results)
    report["esn_result_rows_expected"] = esn_total
    if strict:
        _assert(qrc_valid == qrc_total, f"QRC feature coverage {qrc_valid}/{qrc_total}")
        _assert(esn_valid == esn_total, f"ESN feature coverage {esn_valid}/{esn_total}")
        _assert(len(qrc_results) == qrc_total * 2, "incomplete QRC metrics")
        _assert(len(esn_results) == esn_total, "incomplete ESN metrics")
    destination = Path(__file__).resolve().parent / "audit_report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report

