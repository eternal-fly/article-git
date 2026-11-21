#!/usr/bin/env python3
"""Backfill CRC3-z/zz only into settings with ten completed QRC seeds.

This intentionally does not restart scans that the author previously stopped.
Every selected setting is resumed with ``force=False``; existing QRC and other
baseline artifacts are required and reused, while the missing no-PCA CRC3
feature/head artifacts are added.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.crc3_baseline import (  # noqa: E402
    CRC3_Z_ZZ_METHOD,
    CRC3_Z_ZZ_PROTOCOL,
    crc3_z_zz_dim,
)
from core.linear_readout import (  # noqa: E402
    METHODS,
    plot_q_equal_t_scan,
    plot_w_scan,
    run_setting_plan,
)
from core.esn_baseline import ESN_STANDARD_METHOD, ESN_STANDARD_PROTOCOL, esn_standard_dim  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _completed_qrc_plan(plan_path: Path) -> bool:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    feature_dir = plan_path.parent / "features"
    for seed in plan["seeds"]:
        path = feature_dir / f"qrc_seed{int(seed):03d}.npz"
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as package:
                expected_dim = int(plan["q"]) * (int(plan["q"]) + 1) // 2
                if (
                    package["features"].ndim != 2
                    or package["features"].shape[1] != expected_dim
                    or int(package["q"]) != int(plan["q"])
                    or int(package["t"]) != int(plan["t"])
                    or int(package["seed"]) != int(seed)
                    or not np.isfinite(package["features"]).all()
                ):
                    return False
        except Exception:
            return False
    return True


def _selected_plans() -> list[Path]:
    plans = []
    for path in sorted(PROJECT_ROOT.rglob("setting_plan.json")):
        relative = path.relative_to(PROJECT_ROOT)
        if relative.parts[0] != "general":
            continue
        if _completed_qrc_plan(path):
            plans.append(path)
    return plans


def _validate_setting(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    setting = plan_path.parent
    metrics = pd.read_csv(setting / "metrics_by_seed.csv")
    expected = {(method, int(seed)) for method in METHODS for seed in plan["seeds"]}
    actual = {(str(row.method), int(row.seed)) for row in metrics.itertuples(index=False)}
    if len(metrics) != len(expected) or actual != expected:
        raise RuntimeError(
            f"{setting}: incomplete method/seed contract; "
            f"missing={sorted(expected-actual)[:8]}, extra={sorted(actual-expected)[:8]}"
        )
    q, t = int(plan["q"]), int(plan["t"])
    matched = metrics[metrics.method == CRC3_Z_ZZ_METHOD]
    if set(matched.feature_dim.astype(int)) != {crc3_z_zz_dim(q)}:
        raise RuntimeError(f"{setting}: CRC3 output dimension mismatch")
    for seed in plan["seeds"]:
        path = setting / "features" / f"{CRC3_Z_ZZ_METHOD}_seed{int(seed):03d}.npz"
        with np.load(path, allow_pickle=False) as package:
            if (
                package["features"].shape[1] != crc3_z_zz_dim(q)
                or str(package["baseline_protocol"].item()) != CRC3_Z_ZZ_PROTOCOL
                or bool(package["pca"])
                or (int(package["q"]), int(package["t"]), int(package["seed"]))
                != (q, t, int(seed))
            ):
                raise RuntimeError(f"Invalid CRC3 provenance: {path}")
        esn_path = setting / "features" / f"{ESN_STANDARD_METHOD}_seed{int(seed):03d}.npz"
        with np.load(esn_path, allow_pickle=False) as package:
            if (
                package["features"].shape[1] != esn_standard_dim(q)
                or str(package["baseline_protocol"].item()) != ESN_STANDARD_PROTOCOL
                or bool(package["pca"])
                or str(package["readout"].item()) != "final_ESN_state_direct"
                or (int(package["q"]), int(package["t"]), int(package["seed"]))
                != (q, t, int(seed))
            ):
                raise RuntimeError(f"Invalid standard ESN provenance: {esn_path}")


def _refresh_scan_figures() -> None:
    bace = PROJECT_ROOT / "general" / "BACE"
    q_equal_t = bace / "01_Q_equal_T_scan"
    w_scan = bace / "03_W_scan"
    if any(q_equal_t.rglob("performance_summary.csv")):
        plot_q_equal_t_scan(q_equal_t, "BACE")
    if any(w_scan.rglob("performance_summary.csv")):
        plot_w_scan(w_scan, "BACE")


def main() -> None:
    args = parse_args()
    plans = _selected_plans()
    if args.limit is not None:
        plans = plans[: max(0, args.limit)]
    print(f"Selected {len(plans)} completed-QRC settings; unfinished scans are excluded.", flush=True)
    for index, plan_path in enumerate(plans, start=1):
        print(f"[{index:03d}/{len(plans):03d}] {plan_path.parent.relative_to(PROJECT_ROOT)}", flush=True)
        if args.dry_run:
            continue
        run_setting_plan(plan_path, run_qrc=True, force=False)
        _validate_setting(plan_path)
    if not args.dry_run:
        _refresh_scan_figures()
        marker = PROJECT_ROOT / "NEW_CLASSICAL_BASELINES_BACKFILL_DONE.txt"
        marker.write_text(
            f"{CRC3_Z_ZZ_PROTOCOL} + {ESN_STANDARD_PROTOCOL}: {len(plans)} completed-QRC settings; PCA=False; "
            f"finished={datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
