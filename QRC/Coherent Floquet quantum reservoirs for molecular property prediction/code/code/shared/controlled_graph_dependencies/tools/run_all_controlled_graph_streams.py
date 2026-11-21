"""Non-interactive full-profile controller for the global FINAL queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from controlled_streams import RunConfig, available_devices, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full",), default="full")
    args = parser.parse_args()
    devices = available_devices()
    cfg = RunConfig.for_profile(
        args.profile,
        output_root=str(MODULE_ROOT / "outputs"),
        devices=devices,
        train_device=devices[0],
    )
    print(f"Controlled graph-stream full profile; devices={devices}; W={cfg.W}", flush=True)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
