#!/usr/bin/env python3
"""Retain and summarize every fixed-Q SVR test point on the T grid.

The primary comparison holds both Q/readout width and molecular event prefix
length T fixed.  Q12 is compared with a 78-node classical reservoir and Q16
with a 136-node classical reservoir.  Validation-selected T endpoints remain
supplementary and are deliberately not used by this summarizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import Q_VALUES, RESULT_ROOT, SEEDS, T_VALUES, readout_dim
from run_decoder_q_t_scan import GRID_DECODER_PROTOCOL, GRID_ROOT


OUTPUT_ROOT = GRID_ROOT / "svr_fixed_q_t"


def load_complete_svr_grid() -> pd.DataFrame:
    rows = []
    for path in GRID_ROOT.rglob("metrics.json"):
        if "SVR_RBF" not in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("decoder_protocol") != GRID_DECODER_PROTOCOL:
            raise RuntimeError(f"stale SVR result {path}")
        rows.append(payload)
    frame = pd.DataFrame(rows)
    expected = 2 * 2 * len(Q_VALUES) * len(T_VALUES) * len(SEEDS) * (2 + 4)
    if len(frame) != expected:
        raise RuntimeError(f"incomplete SVR grid: {len(frame)} != {expected}")
    if not np.isfinite(
        frame[["val_auc", "test_auc", "test_acc", "test_brier"]]
    ).all().all():
        raise RuntimeError("non-finite metrics in complete SVR grid")
    return frame


def matched_width_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected_width = frame.q.map(
        {int(q): readout_dim(int(q)) for q in Q_VALUES}
    )
    matched = frame[
        (frame.system == "ESN")
        & frame.width.eq(expected_width)
    ].copy()
    matched["system"] = "CRC_MATCHED"
    primary = pd.concat(
        [
            frame[frame.system.isin(("QRC_CLOSED", "QRC_OPEN"))],
            matched,
        ],
        ignore_index=True,
    )
    keys = ["dataset", "token_variant", "q", "t", "system", "seed"]
    expected = 2 * 2 * len(Q_VALUES) * len(T_VALUES) * 3 * len(SEEDS)
    if len(primary) != expected or primary.duplicated(keys).any():
        raise RuntimeError("invalid matched-width fixed-Q/T comparison grid")
    return primary


def summarize() -> dict:
    frame = load_complete_svr_grid()
    primary = matched_width_frame(frame)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    primary.sort_values(
        ["dataset", "token_variant", "q", "t", "system", "seed"]
    ).to_csv(OUTPUT_ROOT / "test_by_seed.csv", index=False)

    curve_summary = (
        primary.groupby(
            ["dataset", "token_variant", "q", "t", "system"], as_index=False
        )
        .agg(
            feature_dim=("feature_dim", "first"),
            val_auc_mean=("val_auc", "mean"),
            val_auc_std=("val_auc", "std"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_brier_mean=("test_brier", "mean"),
            test_brier_std=("test_brier", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    curve_summary.to_csv(OUTPUT_ROOT / "test_curve_summary.csv", index=False)

    pair_specs = (
        ("QRC_OPEN", "CRC_MATCHED"),
        ("QRC_CLOSED", "CRC_MATCHED"),
        ("QRC_OPEN", "QRC_CLOSED"),
    )
    paired_rows = []
    keys = ["dataset", "token_variant", "q", "t", "seed"]
    for left, right in pair_specs:
        lhs = primary[primary.system == left]
        rhs = primary[primary.system == right]
        merged = lhs.merge(
            rhs,
            on=keys,
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    **{key: row[key] for key in keys},
                    "comparison": f"{left}_MINUS_{right}",
                    "left_test_auc": float(row.test_auc_left),
                    "right_test_auc": float(row.test_auc_right),
                    "delta_test_auc": float(
                        row.test_auc_left - row.test_auc_right
                    ),
                }
            )
    pairs = pd.DataFrame(paired_rows)
    pairs.to_csv(OUTPUT_ROOT / "paired_test_by_seed.csv", index=False)
    pair_summary = (
        pairs.groupby(
            ["dataset", "token_variant", "q", "t", "comparison"],
            as_index=False,
        )
        .agg(
            left_test_auc_mean=("left_test_auc", "mean"),
            right_test_auc_mean=("right_test_auc", "mean"),
            mean_delta_test_auc=("delta_test_auc", "mean"),
            std_delta_test_auc=("delta_test_auc", "std"),
            favourable_seeds=("delta_test_auc", lambda x: int(np.sum(x > 0))),
            tied_seeds=("delta_test_auc", lambda x: int(np.sum(x == 0))),
            n_seeds=("seed", "nunique"),
        )
    )
    pair_summary.to_csv(OUTPUT_ROOT / "paired_test_summary.csv", index=False)

    esn_width_summary = (
        frame[frame.system == "ESN"]
        .groupby(
            ["dataset", "token_variant", "q", "t", "width"], as_index=False
        )
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_brier_mean=("test_brier", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    esn_width_summary.to_csv(
        OUTPUT_ROOT / "crc_width_scaling_test_curve_summary.csv", index=False
    )

    status = {
        "decoder_protocol": GRID_DECODER_PROTOCOL,
        "interpretation": (
            "fixed-Q, fixed-width, same-event-prefix test curves; "
            "no validation-selected T collapse"
        ),
        "svr_grid_rows": len(frame),
        "primary_by_seed_rows": len(primary),
        "primary_curve_rows": len(curve_summary),
        "paired_by_seed_rows": len(pairs),
        "paired_curve_rows": len(pair_summary),
        "crc_width_curve_rows": len(esn_width_summary),
    }
    (OUTPUT_ROOT / "summary_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    return status


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, sort_keys=True))
