#!/usr/bin/env python3
"""Rebuild graph means and paired percentile intervals from the six-seed CSV.

Default mode checks the published tables. --write replaces them. The historical
filename ``long_stream_exact_contrasts`` denotes paired contrasts, not an exact
enumeration of the bootstrap distribution. T=24/30 are descriptive scan cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "main"
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_DRAWS = 20_000
SEEDS = [0, 10, 20, 30, 40, 50]


def rebuild() -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(SOURCE / "graph_stream_q12_metrics_all_seeds.csv")
    assert len(raw) == 360
    assert set(raw.q) == {12} and set(raw.decoder) == {"rbf_svr"}
    assert set(raw.token_variant) == {"canonical_general"}
    curves = raw.groupby(["dataset", "q", "t", "system"], as_index=False).agg(
        test_auc_mean=("test_auc", "mean"), test_auc_sd=("test_auc", "std"),
        validation_auc_mean=("val_auc", "mean"), n_seeds=("seed", "nunique"),
    )
    assert (curves.n_seeds == 6).all()
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(SEEDS), size=(BOOTSTRAP_DRAWS, len(SEEDS))
    )
    contrasts = []
    comparisons = [
        ("dissipative_minus_matched_esn", "open_dtc_qrc", "matched_esn"),
        ("dissipative_minus_closed", "open_dtc_qrc", "closed_dtc_qrc"),
        ("closed_minus_matched_esn", "closed_dtc_qrc", "matched_esn"),
    ]
    for (dataset, t), cell in raw[raw.t.isin([24, 30])].groupby(["dataset", "t"]):
        paired = cell.pivot(index="seed", columns="system", values="test_auc").loc[SEEDS]
        assert not paired.isna().any().any()
        for name, a, b in comparisons:
            delta = (paired[a] - paired[b]).to_numpy()
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975], method="linear")
            contrasts.append(dict(
                dataset=dataset, q=12, t=int(t), comparison=name,
                delta_auc_mean=delta.mean(), ci_low=low, ci_high=high,
                positive_seeds=int((delta > 0).sum()), zero_seeds=int((delta == 0).sum()),
                n_seeds=6, information_scope="same T only",
            ))
    effects = pd.DataFrame(contrasts)
    application = []
    for dataset in ["BACE", "BBBP"]:
        for t in [24, 30]:
            c = curves[(curves.dataset == dataset) & (curves.t == t)].set_index("system")
            e = effects[(effects.dataset == dataset) & (effects.t == t) &
                        (effects.comparison == "dissipative_minus_matched_esn")].iloc[0]
            row = dict(dataset=dataset, t=t, dissipative_p=0.10)
            for prefix, system in [("dissipative_qrc", "open_dtc_qrc"),
                                   ("closed_qrc", "closed_dtc_qrc"), ("matched_esn", "matched_esn")]:
                row[prefix + "_auc_mean"] = c.loc[system, "test_auc_mean"]
                row[prefix + "_auc_sd"] = c.loc[system, "test_auc_sd"]
            row.update(qrc_minus_esn_mean=e.delta_auc_mean,
                       qrc_minus_esn_ci95_low=e.ci_low, qrc_minus_esn_ci95_high=e.ci_high)
            application.append(row)
    return {
        "graph_stream_q12_curves.csv": curves,
        "graph_stream_q12_long_stream_exact_contrasts.csv": effects,
        "graph_stream_q12_application_table.csv": pd.DataFrame(application),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    tables = rebuild()
    for filename, calculated in tables.items():
        path = SOURCE / filename
        if args.write:
            calculated.to_csv(path, index=False)
        else:
            pd.testing.assert_frame_equal(pd.read_csv(path), calculated, check_dtype=False,
                                          rtol=1e-12, atol=1e-14)
    if args.write:
        metadata = dict(method="paired-seed percentile bootstrap", draws=BOOTSTRAP_DRAWS,
                        random_seed=BOOTSTRAP_SEED, generator="numpy.random.default_rng / PCG64",
                        quantile_method="linear", confidence_level=0.95, paired_seeds=SEEDS,
                        selection="T=24/30 are descriptive cells; intervals are pointwise",
                        note="September audit recomputation; point estimates unchanged; no experiments rerun")
        (SOURCE / "graph_statistics_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(tables["graph_stream_q12_application_table.csv"].to_string(index=False))
    print("Graph statistics: " + ("written" if args.write else "verified"))


if __name__ == "__main__":
    main()
