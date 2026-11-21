"""Audit unchanged ESN controls and assemble the fair main-protocol tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import (
    DATASETS,
    DECODERS,
    ESN_INPUT_SCALE,
    ESN_LEAK_RATE,
    ESN_SPECTRAL_RADIUS,
    LAYERS,
    PROTOCOL_VERSION,
    Q_VALUES,
    RESULT_ROOT,
    SEEDS,
    SOURCE_MIGRATION,
    T_VALUES,
    TOKEN_VARIANTS,
    WIDE_ESN_T_VALUES,
    WIDE_ESN_WIDTHS,
    readout_dim,
    source_token_path,
)


LEGACY_ESN_CACHE = SOURCE_MIGRATION / "cache" / "esn"
LEGACY_ESN_RESULTS = SOURCE_MIGRATION / "results"
OUTPUT = RESULT_ROOT / "main_fixed_q_t"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _feature_path(dataset: str, variant: str, q: int, t: int, width: int, seed: int) -> Path:
    return (
        LEGACY_ESN_CACHE / dataset / variant / f"Q{q:02d}" / f"T{t:02d}"
        / f"esn_width{width:04d}_seed{seed:03d}.npz"
    )


def _linear_result_path(dataset: str, variant: str, q: int, t: int, width: int, seed: int) -> Path:
    return (
        LEGACY_ESN_RESULTS / "esn" / dataset / variant / f"Q{q:02d}" / f"T{t:02d}"
        / f"width{width:04d}" / f"seed{seed:03d}" / "result.json"
    )


def _svr_result_path(dataset: str, variant: str, q: int, t: int, width: int, seed: int) -> Path:
    return (
        LEGACY_ESN_RESULTS / "decoder_q_t_scan" / dataset / variant / "ESN"
        / f"Q{q:02d}" / f"T{t:02d}" / f"width{width:04d}" / "SVR_RBF"
        / f"seed{seed:03d}" / "metrics.json"
    )


def _audit_feature(
    dataset: str, variant: str, q: int, t: int, width: int, seed: int, token_sha: str
) -> None:
    path = _feature_path(dataset, variant, q, t, width, seed)
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"]
        metadata = json.loads(str(archive["metadata_json"].item()))
    expected = {
        "dataset": dataset,
        "variant": variant,
        "q": q,
        "t": t,
        "width": width,
        "seed": seed,
        "token_sha256": token_sha,
        "layers_per_event": LAYERS,
        "spectral_radius": ESN_SPECTRAL_RADIUS,
        "input_scale": ESN_INPUT_SCALE,
        "leak_rate": ESN_LEAK_RATE,
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"incompatible frozen ESN state {path}: {actual} != {expected}")
    if features.ndim != 2 or features.shape[1] != width or not np.isfinite(features).all():
        raise RuntimeError(f"invalid frozen ESN features {path}: {features.shape}")


def _normalize_result(payload: dict, decoder: str, scope: str) -> dict:
    row = dict(payload)
    row["decoder"] = decoder
    row["system"] = "matched_esn" if scope == "matched" else "wide_esn"
    row["comparison_scope"] = scope
    row["source_protocol_version"] = row.pop("protocol_version", None)
    row["protocol_version"] = PROTOCOL_VERSION
    row["p"] = np.nan
    rename = {
        "selected_C": "C",
        "selected_gamma_scale": "gamma_scale",
        "selected_gamma": "gamma",
        "selected_epsilon": "epsilon",
    }
    for old, new in rename.items():
        if old in row:
            row[new] = row.pop(old)
    return row


def collect_esn() -> pd.DataFrame:
    rows = []
    audited = set()
    for dataset in DATASETS:
        for variant in TOKEN_VARIANTS:
            with np.load(source_token_path(dataset, variant), allow_pickle=False) as archive:
                token_sha = str(archive["tokens_sha256"].item())
            for q in Q_VALUES:
                matched = readout_dim(q)
                for t in T_VALUES:
                    widths = [matched]
                    if t in WIDE_ESN_T_VALUES:
                        widths.extend(WIDE_ESN_WIDTHS)
                    for width in widths:
                        scope = "matched" if width == matched else "wide"
                        for seed in SEEDS:
                            feature_key = (dataset, variant, q, t, width, seed)
                            if feature_key not in audited:
                                _audit_feature(*feature_key, token_sha)
                                audited.add(feature_key)
                            linear = json.loads(
                                _linear_result_path(*feature_key).read_text(encoding="utf-8")
                            )
                            svr = json.loads(
                                _svr_result_path(*feature_key).read_text(encoding="utf-8")
                            )
                            rows.append(_normalize_result(linear, "linear", scope))
                            rows.append(_normalize_result(svr, "rbf_svr", scope))
    result = pd.DataFrame(rows)
    identity = ["dataset", "token_variant", "q", "t", "seed", "decoder", "width"]
    if result.duplicated(identity).any():
        raise RuntimeError("duplicate ESN result rows")
    return result


def main() -> None:
    qrc_path = OUTPUT / "qrc_metrics_all_seeds.csv"
    if not qrc_path.exists():
        raise FileNotFoundError(f"QRC training is incomplete: {qrc_path}")
    qrc = pd.read_csv(qrc_path)
    esn = collect_esn()
    matched = esn[esn.comparison_scope == "matched"].copy()
    width = esn[
        (esn["t"].isin(WIDE_ESN_T_VALUES))
        & (esn["width"].isin([readout_dim(q) for q in Q_VALUES] + list(WIDE_ESN_WIDTHS)))
    ].copy()
    main = pd.concat([qrc, matched], ignore_index=True, sort=False)
    group = ["dataset", "token_variant", "q", "t", "seed", "decoder"]
    expected_systems = {"closed_dtc_qrc", "open_dtc_qrc", "matched_esn"}
    bad = [
        key for key, frame in main.groupby(group)
        if set(frame["system"]) != expected_systems
    ]
    if bad:
        raise RuntimeError(f"unpaired main comparison keys: {bad[:5]}")
    summary = (
        main.groupby(["dataset", "token_variant", "q", "t", "decoder", "system"], as_index=False)
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", "std"),
            val_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "token_variant", "q", "t", "decoder", "system"])
    )
    width_summary = (
        width.groupby(
            ["dataset", "token_variant", "q", "t", "decoder", "width"], as_index=False
        )
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", "std"),
            val_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "token_variant", "q", "t", "decoder", "width"])
    )
    _atomic_csv(esn, OUTPUT / "esn_audited_metrics.csv")
    _atomic_csv(main, OUTPUT / "main_pointwise_metrics_all_seeds.csv")
    _atomic_csv(summary, OUTPUT / "main_pointwise_summary.csv")
    _atomic_csv(width, OUTPUT / "esn_width_metrics_all_seeds.csv")
    _atomic_csv(width_summary, OUTPUT / "esn_width_summary.csv")
    expected_qrc = len(DATASETS) * len(TOKEN_VARIANTS) * len(Q_VALUES) * len(T_VALUES) * len(SEEDS) * len(DECODERS) * 2
    expected_matched = len(DATASETS) * len(TOKEN_VARIANTS) * len(Q_VALUES) * len(T_VALUES) * len(SEEDS) * len(DECODERS)
    expected_width = len(DATASETS) * len(TOKEN_VARIANTS) * len(Q_VALUES) * len(WIDE_ESN_T_VALUES) * len(SEEDS) * len(DECODERS) * 4
    audit = {
        "status": "passed",
        "protocol_version": PROTOCOL_VERSION,
        "qrc_rows": len(qrc),
        "expected_qrc_rows": expected_qrc,
        "matched_esn_rows": len(matched),
        "expected_matched_esn_rows": expected_matched,
        "esn_width_rows": len(width),
        "expected_esn_width_rows": expected_width,
        "main_rows": len(main),
        "paired_keys": main.groupby(group).ngroups,
        "systems_per_key": sorted(expected_systems),
        "seeds": list(SEEDS),
        "fixed_open_p": 0.10,
        "legacy_esn_reuse_reason": (
            "ESN receives the identical frozen token arrays and is independent of the QRC "
            "Hamiltonian injection map; token hashes and all ESN dynamics parameters were audited."
        ),
    }
    if (
        len(qrc) != expected_qrc
        or len(matched) != expected_matched
        or len(width) != expected_width
        or not (summary.n_seeds == len(SEEDS)).all()
        or not (width_summary.n_seeds == len(SEEDS)).all()
    ):
        audit["status"] = "failed"
        raise RuntimeError(audit)
    (OUTPUT / "training_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
