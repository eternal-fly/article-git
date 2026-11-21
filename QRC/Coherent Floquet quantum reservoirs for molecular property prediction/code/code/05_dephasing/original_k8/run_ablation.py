#!/usr/bin/env python3
"""Run feature extraction or RBF-SVR evaluation for the mechanism ablation."""

from __future__ import annotations

import argparse
import json

from evaluate_ablation import evaluate_all, evaluate_bace, evaluate_hopping
from mechanism_features import compute_suffix_bank
from protocol import BACE_SEEDS, HOPPING_SEEDS, VARIANT_BY_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    features = sub.add_parser("features")
    features.add_argument("--task", choices=("bace", "hopping"), required=True)
    features.add_argument("--variant", choices=tuple(VARIANT_BY_NAME), required=True)
    features.add_argument("--seed", type=int, required=True)
    features.add_argument("--device", default="cuda:0")
    features.add_argument("--batch-size", type=int, default=None)
    features.add_argument("--force", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--task", choices=("bace", "hopping", "all"), default="all")
    evaluation.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "features":
        allowed = BACE_SEEDS if args.task == "bace" else HOPPING_SEEDS
        if args.seed not in allowed:
            parser.error(f"seed must be one of {allowed}")
        if args.variant == "coherent":
            parser.error("coherent banks are reused and do not require extraction")
        path = compute_suffix_bank(
            args.task, args.variant, args.seed, args.device,
            force=args.force, batch_size=args.batch_size,
        )
        print(path)
        return
    if args.task == "bace":
        result = [str(path) for path in evaluate_bace(args.force)]
    elif args.task == "hopping":
        result = [str(path) for path in evaluate_hopping(args.force)]
    else:
        result = evaluate_all(args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

