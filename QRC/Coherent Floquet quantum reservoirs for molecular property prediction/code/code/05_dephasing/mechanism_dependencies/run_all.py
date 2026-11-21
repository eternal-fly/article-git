#!/usr/bin/env python3
"""Two-GPU dataset runner; each dataset remains internally sequential."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("quick", "full"), default="quick")
    p.add_argument("--section", choices=("all", "fixed_q", "phase_map", "ablations"), default="all")
    p.add_argument("--devices", default="cuda:0,cuda:1")
    p.add_argument("--token-schema", choices=("final_v2", "canonical_v3"), default="canonical_v3")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    devices = [x.strip() for x in a.devices.split(",") if x.strip()]
    if torch.cuda.device_count() < 2:
        devices = ["cuda:0"] if torch.cuda.device_count() == 1 else ["cpu"]
    commands = []
    for i, dataset in enumerate(("BACE", "BBBP")):
        device = devices[i % len(devices)]
        command = [sys.executable, str(HERE / "mechanism_study.py"), "--dataset", dataset,
                   "--mode", a.mode, "--section", a.section, "--devices", device]
        command += ["--token-schema", a.token_schema]
        if a.force:
            command.append("--force")
        commands.append(command)
    if len(devices) >= 2:
        processes = [subprocess.Popen(command, cwd=HERE) for command in commands]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise subprocess.CalledProcessError(next(code for code in codes if code), "dataset worker")
    else:
        for command in commands:
            subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
