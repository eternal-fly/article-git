#!/usr/bin/env python3
"""Run BACE then BBBP validity plans; each QRC plan uses both GPUs internally."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_ORDER = (
    "BACE/03_collision_ceiling/plan.json",
    "BACE/01_isomorphism_robustness/Q16_T16/plan.json",
    "BACE/02_padding_ablation/Q16_T16/plan.json",
    "BACE/04_frozen_cross_order/Q16_T16/plan.json",
    "BBBP/03_collision_ceiling/plan.json",
    "BBBP/01_isomorphism_robustness/Q16_T16/plan.json",
    "BBBP/02_padding_ablation/Q16_T16/plan.json",
    "BBBP/04_frozen_cross_order/Q16_T16/plan.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", "--mode", dest="profile", choices=("quick", "full"), default="full")
    parser.add_argument("--python", default=os.environ.get("QRC_PYTHON", sys.executable))
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"validity_sequential_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "plan", "profile", "returncode", "started", "finished", "log"])
    for index, relative in enumerate(PLAN_ORDER, start=1):
        plan = ROOT / relative
        log_path = log_dir / f"{index:02d}_{plan.parent.parent.name}_{plan.parent.name}.log"
        started = dt.datetime.now().isoformat(timespec="seconds")
        print(f"[{index}/{len(PLAN_ORDER)}] {plan}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [args.python, str(ROOT / "run_validity.py"), "--plan", str(plan), "--profile", args.profile],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finished = dt.datetime.now().isoformat(timespec="seconds")
        with status_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [index, str(plan), args.profile, process.returncode, started, finished, str(log_path)]
            )
        if process.returncode and not args.continue_on_error:
            raise SystemExit(f"Plan failed ({process.returncode}); see {log_path}")
    (ROOT / "logs" / "latest_validity_run.txt").write_text(str(log_dir), encoding="utf-8")
    print(f"All plans finished. Logs: {log_dir}")


if __name__ == "__main__":
    main()
