"""Fast invariant checks for every controlled task manifest (no model training)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from controlled_streams.tasks import generate_dataset, task_settings  # noqa: E402


def linear_hist(row: np.ndarray, order: int) -> tuple[int, ...]:
    counts = np.zeros(2**order, dtype=int)
    for start in range(len(row) - order + 1):
        code = 0
        for offset in range(order):
            code = (code << 1) | int(row[start + offset])
        counts[code] += 1
    return tuple(int(value) for value in counts)


def validate_profile(profile: str) -> None:
    q = 6 if profile == "smoke" else (8 if profile == "quick" else 12)
    for index, setting in enumerate(task_settings(profile)):
        dataset = generate_dataset(setting, q=q, n_groups=4, generation_seed=12345 + index)
        dataset.validate(q)
        for group in np.unique(dataset.groups):
            group_labels = dataset.labels[dataset.groups == group]
            assert set(np.unique(group_labels).tolist()) == {0, 1}, (setting.name, group_labels)
        if dataset.symbols is not None:
            max_symbol = int(dataset.symbols.max())
            hist = []
            for label in (0, 1):
                values = dataset.symbols[dataset.labels == label].reshape(-1)
                hist.append(np.bincount(values, minlength=max_symbol + 1))
            assert np.array_equal(hist[0], hist[1]), (setting.name, hist)
        if setting.family == "third_order":
            for group in np.unique(dataset.groups):
                pair = dataset.symbols[dataset.groups == group]
                assert linear_hist(pair[0], 1) == linear_hist(pair[1], 1)
                assert linear_hist(pair[0], 2) == linear_hist(pair[1], 2)
                assert pair[0, 0] == pair[1, 0] and pair[0, -1] == pair[1, -1]
                assert linear_hist(pair[0], 3) != linear_hist(pair[1], 3)
        print(f"PASS {profile:5s} {setting.name}")


def main() -> None:
    for profile in ("smoke", "quick", "full"):
        validate_profile(profile)
    print("All controlled-task invariants passed.")


if __name__ == "__main__":
    main()

