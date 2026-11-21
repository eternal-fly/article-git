from __future__ import annotations

import numpy as np

from readouts import fit_linear_bank, fit_svr


def _toy() -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 5)).astype(np.float32)
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.int64)
    split = {
        "train_indices": list(range(0, 36)),
        "val_indices": list(range(36, 48)),
        "test_indices": list(range(48, 60)),
    }
    return x, y, split


def test_linear_bank_smoke() -> None:
    x, y, split = _toy()
    table, artifacts = fit_linear_bank(
        np.stack([x, x]), y, split, seed=0, device="cpu",
        row_values=(0.0, 0.1), epochs=4,
    )
    assert len(table) == 2
    assert artifacts["test_probabilities"].shape == (2, 12)


def test_svr_smoke() -> None:
    x, y, split = _toy()
    result, artifacts = fit_svr(x, y, split, seed=0)
    assert 0.0 <= result["test_auc"] <= 1.0
    assert len(artifacts["screen"]) == 9

