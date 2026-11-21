#!/usr/bin/env python3
"""Run the closed coherent SHNITSEL A01 benchmark."""

from __future__ import annotations

import argparse

from closed_core import audit_results, build_report, compute_feature_grid, evaluate_classical, evaluate_reservoirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("features", "reservoirs", "classical", "report", "audit", "all-eval"))
    parser.add_argument(
        "--system",
        choices=("qrc", "esn_standard", "esn_second_order_recurrent", "fp_cvr"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "features":
        if args.system is None or args.seed is None:
            parser.error("features requires --system and --seed")
        print(compute_feature_grid(args.system, args.seed, args.device, force=args.force), flush=True)
    elif args.stage == "reservoirs":
        print(evaluate_reservoirs(), flush=True)
    elif args.stage == "classical":
        print(evaluate_classical(), flush=True)
    elif args.stage == "report":
        print(build_report(), flush=True)
    elif args.stage == "audit":
        print(audit_results(), flush=True)
    else:
        print(evaluate_reservoirs(), flush=True)
        print(evaluate_classical(), flush=True)
        print(build_report(), flush=True)
        print(audit_results(), flush=True)


if __name__ == "__main__":
    main()
