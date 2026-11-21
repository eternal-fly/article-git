#!/usr/bin/env python3
"""Run the SHNITSEL A01 Q15 surface-hopping benchmark."""

from __future__ import annotations

import argparse

from a01_core import (
    build_dataset, build_report, compute_feature_bank, evaluate_classical,
    evaluate_reservoirs, load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "features", "reservoirs", "classical", "report", "all-eval"))
    parser.add_argument("--system", choices=("qrc", "esn"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    print(build_dataset(cfg, force=args.force and args.stage == "prepare"), flush=True)
    if args.stage == "prepare":
        return
    if args.stage == "features":
        if args.system is None or args.seed is None:
            parser.error("features requires --system and --seed")
        print(compute_feature_bank(cfg, args.system, args.seed, args.device, force=args.force), flush=True)
    elif args.stage == "reservoirs":
        print(evaluate_reservoirs(cfg), flush=True)
    elif args.stage == "classical":
        print(evaluate_classical(cfg), flush=True)
    elif args.stage == "report":
        print(build_report(cfg), flush=True)
    else:
        print(evaluate_reservoirs(cfg), flush=True)
        print(evaluate_classical(cfg), flush=True)
        print(build_report(cfg), flush=True)


if __name__ == "__main__":
    main()
