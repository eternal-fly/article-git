#!/usr/bin/env python3
"""Mechanical QA for publication exports and source-data traceability."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


def main() -> None:
    manifest = json.loads((FIGURES / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "Python/matplotlib"
    assert manifest["epoch_loss_available"] is False
    for stem in manifest["figures"]:
        for extension in manifest["formats"]:
            path = FIGURES / f"{stem}.{extension}"
            assert path.exists() and path.stat().st_size > 10000, path
        svg = (FIGURES / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg and "font-family" in svg
        with Image.open(FIGURES / f"{stem}.png") as image:
            assert image.width >= 2000 and image.height >= 1700
        with Image.open(FIGURES / f"{stem}.tiff") as image:
            assert image.info.get("compression") == "tiff_lzw"
    metrics = pd.read_csv(FIGURES / "source_data" / "model_metrics.csv")
    assert len(metrics) == 4 * 6 * 3
    qrc_primary = metrics[(metrics.target == "min_gap_next_5fs")
                          & (metrics.method == "qrc") & (metrics.metric == "mae_ev")].iloc[0]
    assert abs(qrc_primary["mean"] - 0.8467088374358632) < 1e-12
    assert (FIGURES / "FIGURE_QA.md").exists() and (FIGURES / "FIGURE_CAPTIONS.md").exists()
    print("Figure audit passed: source data, editable SVG, PDF/PNG and 600-dpi LZW TIFF exports.")


if __name__ == "__main__":
    main()
