#!/usr/bin/env python3
"""Precompute exact dense ESN spectral-radius normalizations."""

from __future__ import annotations

import argparse

from esn_features import build_or_load_recurrent
from protocol import SEEDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="78,136,500,1000,2000")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    widths = tuple(int(value) for value in args.widths.split(",") if value.strip())
    for width in widths:
        for seed in SEEDS:
            _, metadata = build_or_load_recurrent(width, seed, force=args.force)
            print(
                {
                    "width": width,
                    "seed": seed,
                    "radius_before": metadata["spectral_radius_before_scaling"],
                    "seconds": metadata["eigendecomposition_seconds"],
                },
                flush=True,
            )


if __name__ == "__main__":
    main()

