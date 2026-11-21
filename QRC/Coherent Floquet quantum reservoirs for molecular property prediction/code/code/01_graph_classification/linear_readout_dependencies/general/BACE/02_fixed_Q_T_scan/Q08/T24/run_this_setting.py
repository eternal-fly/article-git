"""Run this exact Q/T setting.

Edit RUN_QRC/FORCE below if needed, then run:

    /home/ubuntu/miniconda3/envs/Molecule/bin/python run_this_setting.py
"""

from __future__ import annotations

from pathlib import Path
import sys


SETTING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SETTING_DIR
while not (PROJECT_ROOT / "core" / "linear_readout.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Could not locate FINAL/2_linear_readout project root")
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.linear_readout import run_setting_plan


RUN_QRC = True
FORCE = False


def main() -> None:
    metrics, summary = run_setting_plan(
        SETTING_DIR / "setting_plan.json",
        run_qrc=RUN_QRC,
        force=FORCE,
    )
    print("\n=== performance_summary.csv ===")
    print(summary.to_string(index=False))
    print("\n=== last metrics_by_seed rows ===")
    print(metrics.sort_values(["method", "seed"]).tail(12).to_string(index=False))
    print(f"\nOutputs written under: {SETTING_DIR}")


if __name__ == "__main__":
    main()
