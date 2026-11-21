#!/usr/bin/env python3
"""Complete the formal tiny-p plus uniform-0.05 reset scan sequentially on CPU.

Only missing QRC shard pairs are evaluated. Existing cells that remain on the
formal grid are reused exactly; former off-grid cells such as p=0.16, 0.32 and
0.64 are ignored by aggregation. After all missing cells finish, the complete
six-seed statistics and simultaneous scan-wide bands are rebuilt.
"""

from __future__ import annotations

import time

import numpy as np

from a01_core import load_config
from evaluate_strict_axes import READOUTS, aggregate, evaluate_shard, shard_stem


TINY_P_GRID = np.asarray([0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])
UNIFORM_P_GRID = np.round(np.arange(0.05, 1.001, 0.05), 2)
FORMAL_P_GRID = np.asarray(sorted(set(TINY_P_GRID) | set(UNIFORM_P_GRID)))


def main() -> None:
    cfg = load_config()
    if len(cfg.p_grid) != len(FORMAL_P_GRID) or not np.allclose(
        cfg.p_grid, FORMAL_P_GRID
    ):
        raise RuntimeError(f"unexpected formal p grid: {cfg.p_grid}")
    if any(np.isclose(cfg.p_grid, p).any() for p in (0.16, 0.32, 0.64)):
        raise RuntimeError("legacy off-grid p cells entered the formal protocol")

    jobs = []
    for seed in cfg.seeds:
        for p in cfg.p_grid:
            for readout in READOUTS:
                stem = shard_stem("qrc", int(seed), float(p), readout)
                if not (stem.with_suffix(".csv").exists()
                        and stem.with_suffix(".npz").exists()):
                    jobs.append((int(seed), float(p), readout))

    print(
        f"formal grid: {len(cfg.p_grid)} p values; "
        f"missing fixed shards: {len(jobs)}",
        flush=True,
    )
    started = time.time()
    for index, (seed, p, readout) in enumerate(jobs, start=1):
        shard_started = time.time()
        csv_path, npz_path = evaluate_shard("qrc", seed, p, readout)
        print(
            f"[uniform-p {index:03d}/{len(jobs):03d}] "
            f"seed={seed} p={p:.4g} readout={readout} "
            f"elapsed={time.time() - shard_started:.1f}s "
            f"csv={csv_path.name} npz={npz_path.name}",
            flush=True,
        )

    outputs = aggregate()
    print(
        f"formal uniform-p scan complete: {len(jobs)} new fixed shards, "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    for output in outputs:
        print(output, flush=True)


if __name__ == "__main__":
    main()
