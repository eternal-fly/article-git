#!/usr/bin/env python3
"""Run one deterministic shard of the resumable mechanism-feature queue."""

from __future__ import annotations

import argparse
import json

from mechanism_features import compute_suffix_bank
from protocol import BACE_SEEDS, HOPPING_SEEDS, VARIANTS, feature_path


def build_shards() -> tuple[list[dict], list[dict]]:
    jobs = []
    for task, seeds in (("bace", BACE_SEEDS), ("hopping", HOPPING_SEEDS)):
        for variant in VARIANTS:
            if variant.name == "coherent":
                continue
            for seed in seeds:
                weight = 1.0
                if variant.dephasing_strength > 0:
                    weight = 8.0 if task == "hopping" else 4.0
                elif task == "hopping":
                    weight = 1.5
                jobs.append(
                    {
                        "task": task,
                        "variant": variant.name,
                        "seed": seed,
                        "weight": weight,
                    }
                )
    jobs.sort(key=lambda row: (-row["weight"], row["task"], row["variant"], row["seed"]))
    shards: tuple[list[dict], list[dict]] = ([], [])
    loads = [0.0, 0.0]
    for job in jobs:
        target = 0 if loads[0] <= loads[1] else 1
        shards[target].append(job)
        loads[target] += job["weight"]
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    jobs = build_shards()[args.shard]
    print(json.dumps({"shard": args.shard, "device": args.device, "jobs": jobs}, indent=2), flush=True)
    for index, job in enumerate(jobs, start=1):
        destination = feature_path(job["task"], job["variant"], job["seed"])
        if destination.exists() and not args.force:
            print(f"[queue {args.shard}] cached {index}/{len(jobs)} {destination}", flush=True)
            continue
        print(f"[queue {args.shard}] start {index}/{len(jobs)} {job}", flush=True)
        path = compute_suffix_bank(
            job["task"], job["variant"], job["seed"], args.device,
            force=args.force, batch_size=128,
        )
        print(f"[queue {args.shard}] complete {index}/{len(jobs)} {path}", flush=True)
    print(f"[queue {args.shard}] all complete", flush=True)


if __name__ == "__main__":
    main()

