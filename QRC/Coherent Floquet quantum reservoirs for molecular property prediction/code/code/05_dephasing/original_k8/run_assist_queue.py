#!/usr/bin/env python3
"""Use a freed GPU to finish the tail of the other deterministic shard."""

from mechanism_features import compute_suffix_bank
from protocol import feature_path


JOBS = [
    ("bace", "interaction_half", seed) for seed in (10, 30, 50)
] + [
    ("bace", "interaction_off", seed) for seed in (10, 30, 50)
]


def main() -> None:
    for index, (task, variant, seed) in enumerate(JOBS, start=1):
        destination = feature_path(task, variant, seed)
        if destination.exists():
            print(f"[assist] cached {index}/{len(JOBS)} {destination}", flush=True)
            continue
        print(f"[assist] start {index}/{len(JOBS)} {task} {variant} seed={seed}", flush=True)
        path = compute_suffix_bank(
            task, variant, seed, "cuda:0", force=False, batch_size=128
        )
        print(f"[assist] complete {index}/{len(JOBS)} {path}", flush=True)
    print("[assist] all complete", flush=True)


if __name__ == "__main__":
    main()
