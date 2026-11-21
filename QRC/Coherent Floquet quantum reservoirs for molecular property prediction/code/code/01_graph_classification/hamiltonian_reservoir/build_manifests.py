"""Write immutable CSV/JSON execution manifests for the frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from protocol import (
    CLOSED_P,
    DECODERS,
    G,
    INJECTION_PROTOCOL,
    LAYERS,
    MANIFEST_ROOT,
    OBSERVABLES,
    OBSERVABLE_D_VALUES,
    OBSERVABLE_Q,
    OBSERVABLE_SUBSET_SEEDS,
    OBSERVABLE_T_VALUES,
    OPEN_P,
    PRIMARY_DECODER,
    PROTOCOL_VERSION,
    SEEDS,
    W,
    main_comparison_jobs,
    pt_jobs,
    validate_protocol,
    wg_jobs,
    wide_esn_jobs,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def observable_jobs() -> list[dict]:
    rows = []
    for dataset in ("BACE", "BBBP"):
        for system, p in (("closed_dtc_qrc", CLOSED_P), ("open_dtc_qrc", OPEN_P)):
            for t in OBSERVABLE_T_VALUES:
                for d in OBSERVABLE_D_VALUES:
                    for subset_seed in OBSERVABLE_SUBSET_SEEDS:
                        for seed in SEEDS:
                            rows.append(
                                {
                                    "figure": "fig6_observable_basis_diagnostic",
                                    "dataset": dataset,
                                    "token_variant": "canonical_general",
                                    "system": system,
                                    "q": OBSERVABLE_Q,
                                    "t": t,
                                    "retained_dimension": d,
                                    "observable_subset_seed": subset_seed,
                                    "seed": seed,
                                    "decoder": PRIMARY_DECODER,
                                    "p": p,
                                    "g": G,
                                    "W": W,
                                    "layers": LAYERS,
                                    "families": "ZZZ_enriched|mixed_XY|union",
                                    "defined_metrics": "family_AUC|subspace_overlap|union_increment",
                                    "main_readout_unchanged": OBSERVABLES,
                                }
                            )
        # An ESN has no Pauli basis.  Label-independent subsets of its 136
        # nodes supply a same-D predictive reference, but never enter the QRC
        # subspace-overlap or operator-union definitions.
        for t in OBSERVABLE_T_VALUES:
            for d in OBSERVABLE_D_VALUES:
                for subset_seed in OBSERVABLE_SUBSET_SEEDS:
                    for seed in SEEDS:
                        rows.append(
                            {
                                "figure": "fig6_esn_same_D_predictive_reference",
                                "dataset": dataset,
                                "token_variant": "canonical_general",
                                "system": "matched_esn",
                                "q": OBSERVABLE_Q,
                                "t": t,
                                "retained_dimension": d,
                                "observable_subset_seed": subset_seed,
                                "seed": seed,
                                "decoder": PRIMARY_DECODER,
                                "p": None,
                                "g": None,
                                "W": None,
                                "layers": LAYERS,
                                "families": "ESN_node_subset",
                                "defined_metrics": "predictive_AUC_only",
                                "main_readout_unchanged": OBSERVABLES,
                            }
                        )
    return rows


def write_manifests(destination: Path = MANIFEST_ROOT) -> dict:
    audit = validate_protocol()
    destination.mkdir(parents=True, exist_ok=True)
    main_rows = [job.to_dict() for job in main_comparison_jobs()]
    wide_rows = [job.to_dict() for job in wide_esn_jobs()]
    wg_rows = [job.to_dict() for job in wg_jobs()]
    pt_rows = [job.to_dict() for job in pt_jobs()]
    obs_rows = observable_jobs()
    _write_csv(destination / "main_pointwise_comparisons.csv", main_rows)
    _write_csv(destination / "esn_width_scan_separate.csv", wide_rows)
    _write_csv(destination / "w_g_full_grid.csv", wg_rows)
    _write_csv(destination / "p_by_T_diagnostic.csv", pt_rows)
    _write_csv(destination / "observable_diagnostic.csv", obs_rows)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "injection": INJECTION_PROTOCOL,
        "main_observables": OBSERVABLES,
        "seeds": list(SEEDS),
        "decoders": list(DECODERS),
        "primary_decoder": PRIMARY_DECODER,
        "canonical": {"g": G, "W": W, "layers": LAYERS},
        "channels": {
            "closed": {"p": CLOSED_P},
            "open": {"p": OPEN_P, "selection": "prespecified globally; never selected per seed"},
        },
        "row_counts": {
            **audit,
            "observable_rows": len(obs_rows),
        },
        "interpretation": {
            "joint_q_equals_t": "descriptive scaling only",
            "fixed_q_prefix": "primary pointwise comparison; retain every T test result",
            "wide_esn": "separate capacity-scaling figure",
            "w_g": "all test cells retained; no validation-selected heat-map cell",
            "p_by_T": (
                "fixed Q12 diagnostic; all p/T test cells retained and never used "
                "to choose the main open p"
            ),
            "observable_exception": (
                "X/Y/ZZZ appear only in the Fig. 6 diagnostic; the main architecture remains Z+ZZ"
            ),
        },
    }
    (destination / "protocol_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=MANIFEST_ROOT)
    args = parser.parse_args()
    print(json.dumps(write_manifests(args.destination.resolve()), indent=2))


if __name__ == "__main__":
    main()
