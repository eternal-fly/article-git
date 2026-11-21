"""Strict collection, validation-only T selection, and paired comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from esn_evaluation import ESN_READOUT_PROTOCOL, esn_result_dir
from protocol import (
    DATASETS,
    P_VALUES,
    PROTOCOL_VERSION,
    Q_VALUES,
    RESULT_ROOT,
    SEEDS,
    T_VALUES,
    TOKEN_VARIANTS,
    esn_widths,
)
from readouts import qrc_result_dir
from readouts import P_SELECTION_PROTOCOL


def collect_qrc(strict: bool = True) -> pd.DataFrame:
    rows = []
    missing = []
    for dataset in DATASETS:
        for variant in TOKEN_VARIANTS:
            for q in Q_VALUES:
                for t in T_VALUES:
                    for seed in SEEDS:
                        path = qrc_result_dir(dataset, variant, q, t, seed) / "metrics.json"
                        if not path.exists():
                            missing.append(str(path))
                            continue
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        if (
                            payload.get("protocol_version") != PROTOCOL_VERSION
                            or payload.get("p_selection_protocol") != P_SELECTION_PROTOCOL
                        ):
                            raise RuntimeError(f"stale QRC result {path}")
                        rows.extend(payload["rows"])
    if strict and missing:
        raise FileNotFoundError(f"Missing {len(missing)} QRC result files; first={missing[0]}")
    frame = pd.DataFrame(rows)
    if len(frame):
        if not set(frame.p.astype(float)).issubset(set(P_VALUES)):
            raise RuntimeError("QRC results contain p outside the frozen grid")
        if frame.duplicated(["dataset", "token_variant", "system", "q", "t", "seed"]).any():
            raise RuntimeError("duplicate QRC result keys")
    return frame


def collect_esn(strict: bool = True) -> pd.DataFrame:
    rows = []
    missing = []
    for dataset in DATASETS:
        for variant in TOKEN_VARIANTS:
            for q in Q_VALUES:
                for t in T_VALUES:
                    for width in esn_widths(q):
                        for seed in SEEDS:
                            path = esn_result_dir(dataset, variant, q, t, width, seed) / "result.json"
                            if not path.exists():
                                missing.append(str(path))
                                continue
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            if (
                                payload.get("protocol_version") != PROTOCOL_VERSION
                                or payload.get("esn_readout_protocol") != ESN_READOUT_PROTOCOL
                            ):
                                raise RuntimeError(f"stale ESN result {path}")
                            rows.append(payload)
    if strict and missing:
        raise FileNotFoundError(f"Missing {len(missing)} ESN result files; first={missing[0]}")
    frame = pd.DataFrame(rows)
    if len(frame) and frame.duplicated(
        ["dataset", "token_variant", "q", "t", "width", "seed"]
    ).any():
        raise RuntimeError("duplicate ESN result keys")
    return frame


def validation_select_t(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    selected = []
    for _, group in frame.groupby(group_columns, dropna=False):
        ordered = group.sort_values(
            ["val_auc", "val_loss", "val_acc", "t"],
            ascending=[False, True, False, True],
            kind="mergesort",
        )
        row = ordered.iloc[0].to_dict()
        row["selected_t_by"] = "per-seed validation: max AUC > min loss > max ACC > smaller T"
        selected.append(row)
    return pd.DataFrame(selected)


def summarize_all(*, strict: bool = True) -> dict:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    qrc = collect_qrc(strict=strict)
    esn = collect_esn(strict=strict)
    qrc.to_csv(RESULT_ROOT / "qrc_metrics_all_q_t_seeds.csv", index=False)
    esn.to_csv(RESULT_ROOT / "esn_metrics_all_q_t_seeds.csv", index=False)
    outputs: dict[str, object] = {"qrc_rows": len(qrc), "esn_rows": len(esn)}
    if len(qrc):
        qrc_summary = (
            qrc.groupby(["dataset", "token_variant", "system", "q", "t"], as_index=False)
            .agg(
                test_auc_mean=("test_auc", "mean"),
                test_auc_std=("test_auc", "std"),
                test_acc_mean=("test_acc", "mean"),
                p_mean=("p", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        qrc_summary.to_csv(RESULT_ROOT / "qrc_q_t_summary.csv", index=False)
        qrc_selected = validation_select_t(
            qrc, ["dataset", "token_variant", "system", "q", "seed"]
        )
        qrc_selected.to_csv(RESULT_ROOT / "qrc_validation_selected_t_by_seed.csv", index=False)
        outputs["qrc_selected_rows"] = len(qrc_selected)
        qrc_joint_rows = []
        for _, group in qrc.groupby(
            ["dataset", "token_variant", "system", "seed"]
        ):
            qrc_joint_rows.append(
                group.sort_values(
                    ["val_auc", "val_loss", "val_acc", "q", "t"],
                    ascending=[False, True, False, True, True],
                    kind="mergesort",
                ).iloc[0].to_dict()
            )
        qrc_joint_selected = pd.DataFrame(qrc_joint_rows)
        qrc_joint_selected["selected_q_t_by"] = (
            "per-seed validation: max AUC > min loss > max ACC > smaller Q > smaller T"
        )
        qrc_joint_selected.to_csv(
            RESULT_ROOT / "qrc_validation_selected_q_t_by_seed.csv", index=False
        )
        (
            qrc_joint_selected.groupby(
                ["dataset", "token_variant", "system"], as_index=False
            )
            .agg(
                test_auc_mean=("test_auc", "mean"),
                test_auc_std=("test_auc", "std"),
                test_acc_mean=("test_acc", "mean"),
                val_auc_mean=("val_auc", "mean"),
                selected_q_mean=("q", "mean"),
                selected_t_mean=("t", "mean"),
                selected_p_mean=("p", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(RESULT_ROOT / "qrc_validation_selected_q_t_summary.csv", index=False)
        outputs["qrc_joint_selected_rows"] = len(qrc_joint_selected)
        (
            qrc_selected.groupby(
                ["dataset", "token_variant", "system", "q"], as_index=False
            )
            .agg(
                test_auc_mean=("test_auc", "mean"),
                test_auc_std=("test_auc", "std"),
                test_acc_mean=("test_acc", "mean"),
                val_auc_mean=("val_auc", "mean"),
                selected_t_mean=("t", "mean"),
                selected_p_mean=("p", "mean"),
                nonzero_p_seeds=("p", lambda x: int(np.sum(x > 0))),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(RESULT_ROOT / "qrc_validation_selected_summary.csv", index=False)
        (
            qrc_selected[qrc_selected.system == "QRC_OPEN"]
            .groupby(
                ["dataset", "token_variant", "q", "p"], as_index=False
            )
            .agg(n_seeds=("seed", "nunique"))
        ).to_csv(RESULT_ROOT / "qrc_validation_selected_p_frequency.csv", index=False)
        selected_open = qrc_selected[qrc_selected.system == "QRC_OPEN"]
        selected_closed = qrc_selected[qrc_selected.system == "QRC_CLOSED"]
        selected_open_closed = selected_open.merge(
            selected_closed,
            on=["dataset", "token_variant", "q", "seed"],
            suffixes=("_open", "_closed"),
            validate="one_to_one",
        )
        selected_open_closed["open_minus_closed_test_auc"] = (
            selected_open_closed.test_auc_open - selected_open_closed.test_auc_closed
        )
        selected_open_closed.to_csv(
            RESULT_ROOT / "paired_validation_selected_open_vs_closed.csv", index=False
        )
        (
            selected_open_closed.groupby(
                ["dataset", "token_variant", "q"], as_index=False
            )
            .agg(
                open_test_auc_mean=("test_auc_open", "mean"),
                closed_test_auc_mean=("test_auc_closed", "mean"),
                mean_delta_auc=("open_minus_closed_test_auc", "mean"),
                std_delta_auc=("open_minus_closed_test_auc", "std"),
                favourable_seeds=(
                    "open_minus_closed_test_auc",
                    lambda x: int(np.sum(x > 0)),
                ),
                selected_p_mean=("p_open", "mean"),
                selected_t_open_mean=("t_open", "mean"),
                selected_t_closed_mean=("t_closed", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(
            RESULT_ROOT / "paired_validation_selected_open_vs_closed_summary.csv",
            index=False,
        )
        selected_general = qrc_selected[
            qrc_selected.token_variant == "canonical_general"
        ]
        selected_prior = qrc_selected[
            qrc_selected.token_variant == "canonical_global_prior"
        ]
        selected_prior_general = selected_prior.merge(
            selected_general,
            on=["dataset", "system", "q", "seed"],
            suffixes=("_prior", "_general"),
            validate="one_to_one",
        )
        selected_prior_general["prior_minus_general_test_auc"] = (
            selected_prior_general.test_auc_prior
            - selected_prior_general.test_auc_general
        )
        selected_prior_general.to_csv(
            RESULT_ROOT / "paired_validation_selected_prior_vs_general.csv", index=False
        )
        (
            selected_prior_general.groupby(
                ["dataset", "system", "q"], as_index=False
            )
            .agg(
                prior_test_auc_mean=("test_auc_prior", "mean"),
                general_test_auc_mean=("test_auc_general", "mean"),
                mean_delta_auc=("prior_minus_general_test_auc", "mean"),
                std_delta_auc=("prior_minus_general_test_auc", "std"),
                favourable_seeds=(
                    "prior_minus_general_test_auc",
                    lambda x: int(np.sum(x > 0)),
                ),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(
            RESULT_ROOT / "paired_validation_selected_prior_vs_general_summary.csv",
            index=False,
        )
        closed = qrc[qrc.system == "QRC_CLOSED"]
        opened = qrc[qrc.system == "QRC_OPEN"]
        open_closed = opened.merge(
            closed,
            on=["dataset", "token_variant", "q", "t", "seed"],
            suffixes=("_open", "_closed"),
            validate="one_to_one",
        )
        open_closed["open_minus_closed_test_auc"] = (
            open_closed.test_auc_open - open_closed.test_auc_closed
        )
        open_closed.to_csv(RESULT_ROOT / "paired_open_vs_closed_all_q_t.csv", index=False)
        (
            open_closed.groupby(["dataset", "token_variant", "q", "t"], as_index=False)
            .agg(
                mean_delta_auc=("open_minus_closed_test_auc", "mean"),
                std_delta_auc=("open_minus_closed_test_auc", "std"),
                favourable_seeds=("open_minus_closed_test_auc", lambda x: int(np.sum(x > 0))),
                selected_p_mean=("p_open", "mean"),
                nonzero_p_seeds=("p_open", lambda x: int(np.sum(x > 0))),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(RESULT_ROOT / "paired_open_vs_closed_summary.csv", index=False)
        general = qrc[qrc.token_variant == "canonical_general"]
        prior = qrc[qrc.token_variant == "canonical_global_prior"]
        prior_general = prior.merge(
            general,
            on=["dataset", "system", "q", "t", "seed"],
            suffixes=("_prior", "_general"),
            validate="one_to_one",
        )
        prior_general["prior_minus_general_test_auc"] = (
            prior_general.test_auc_prior - prior_general.test_auc_general
        )
        prior_general.to_csv(RESULT_ROOT / "paired_global_prior_vs_general_all_q_t.csv", index=False)
    if len(esn):
        esn_summary = (
            esn.groupby(["dataset", "token_variant", "q", "t", "width"], as_index=False)
            .agg(
                test_auc_mean=("test_auc", "mean"),
                test_auc_std=("test_auc", "std"),
                test_acc_mean=("test_acc", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        esn_summary.to_csv(RESULT_ROOT / "esn_q_t_width_summary.csv", index=False)
        esn_selected = validation_select_t(
            esn, ["dataset", "token_variant", "q", "width", "seed"]
        )
        esn_selected.to_csv(RESULT_ROOT / "esn_validation_selected_t_by_seed.csv", index=False)
        outputs["esn_selected_rows"] = len(esn_selected)
        (
            esn_selected.groupby(
                ["dataset", "token_variant", "q", "width"], as_index=False
            )
            .agg(
                test_auc_mean=("test_auc", "mean"),
                test_auc_std=("test_auc", "std"),
                test_acc_mean=("test_acc", "mean"),
                val_auc_mean=("val_auc", "mean"),
                selected_t_mean=("t", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(RESULT_ROOT / "esn_validation_selected_t_summary.csv", index=False)
        # Apply the full joint ranking explicitly and conservatively prefer
        # smaller width before smaller T on exact metric ties.
        joint_rows = []
        for _, group in esn.groupby(["dataset", "token_variant", "q", "seed"]):
            joint_rows.append(
                group.sort_values(
                    ["val_auc", "val_loss", "val_acc", "width", "t"],
                    ascending=[False, True, False, True, True],
                    kind="mergesort",
                ).iloc[0].to_dict()
            )
        esn_joint_selected = pd.DataFrame(joint_rows)
        esn_joint_selected["selected_width_t_by"] = (
            "per-seed validation: max AUC > min loss > max ACC > smaller width > smaller T"
        )
        esn_joint_selected.to_csv(
            RESULT_ROOT / "esn_validation_selected_width_t_by_seed.csv", index=False
        )
        outputs["esn_joint_selected_rows"] = len(esn_joint_selected)
        matched_esn = esn[
            esn.apply(
                lambda row: int(row.width) == int(row.q * (row.q + 1) // 2),
                axis=1,
            )
        ]
        matched_joint_rows = []
        for _, group in matched_esn.groupby(
            ["dataset", "token_variant", "seed"]
        ):
            matched_joint_rows.append(
                group.sort_values(
                    ["val_auc", "val_loss", "val_acc", "q", "t"],
                    ascending=[False, True, False, True, True],
                    kind="mergesort",
                ).iloc[0].to_dict()
            )
        esn_matched_joint_selected = pd.DataFrame(matched_joint_rows)
        esn_matched_joint_selected["selected_q_t_by"] = (
            "per-seed validation among matched widths: max AUC > min loss > max ACC > smaller Q > smaller T"
        )
        esn_matched_joint_selected.to_csv(
            RESULT_ROOT / "esn_matched_validation_selected_q_t_by_seed.csv",
            index=False,
        )
        all_joint_rows = []
        for _, group in esn.groupby(["dataset", "token_variant", "seed"]):
            all_joint_rows.append(
                group.sort_values(
                    ["val_auc", "val_loss", "val_acc", "width", "q", "t"],
                    ascending=[False, True, False, True, True, True],
                    kind="mergesort",
                ).iloc[0].to_dict()
            )
        esn_all_joint_selected = pd.DataFrame(all_joint_rows)
        esn_all_joint_selected["selected_q_width_t_by"] = (
            "per-seed validation: max AUC > min loss > max ACC > smaller width > smaller Q > smaller T"
        )
        esn_all_joint_selected.to_csv(
            RESULT_ROOT / "esn_validation_selected_q_width_t_by_seed.csv",
            index=False,
        )
        outputs["esn_matched_joint_q_selected_rows"] = len(esn_matched_joint_selected)
        outputs["esn_all_joint_q_selected_rows"] = len(esn_all_joint_selected)
        (
            esn_joint_selected.groupby(
                ["dataset", "token_variant", "q"], as_index=False
            )
            .agg(
                test_auc_mean=("test_auc", "mean"),
                test_auc_std=("test_auc", "std"),
                test_acc_mean=("test_acc", "mean"),
                val_auc_mean=("val_auc", "mean"),
                selected_t_mean=("t", "mean"),
                selected_width_mean=("width", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(
            RESULT_ROOT / "esn_validation_selected_width_t_summary.csv", index=False
        )
        (
            esn_joint_selected.groupby(
                ["dataset", "token_variant", "q", "width"], as_index=False
            )
            .agg(n_seeds=("seed", "nunique"))
        ).to_csv(RESULT_ROOT / "esn_validation_selected_width_frequency.csv", index=False)
    if len(qrc) and len(esn):
        matched = esn[esn.apply(lambda row: int(row.width) == int(row.q * (row.q + 1) // 2), axis=1)]
        merged = qrc.merge(
            matched,
            on=["dataset", "token_variant", "q", "t", "seed"],
            suffixes=("_qrc", "_esn"),
            validate="many_to_one",
        )
        merged["qrc_minus_matched_esn_test_auc"] = merged.test_auc_qrc - merged.test_auc_esn
        merged.to_csv(RESULT_ROOT / "paired_qrc_vs_matched_esn_all_q_t.csv", index=False)
        paired_summary = (
            merged.groupby(["dataset", "token_variant", "system_qrc", "q", "t"], as_index=False)
            .agg(
                mean_delta_auc=("qrc_minus_matched_esn_test_auc", "mean"),
                std_delta_auc=("qrc_minus_matched_esn_test_auc", "std"),
                favourable_seeds=("qrc_minus_matched_esn_test_auc", lambda x: int(np.sum(x > 0))),
                n_seeds=("seed", "nunique"),
            )
        )
        paired_summary.to_csv(RESULT_ROOT / "paired_qrc_vs_matched_esn_summary.csv", index=False)
        outputs["paired_rows"] = len(merged)
        merged_all_widths = qrc.merge(
            esn,
            on=["dataset", "token_variant", "q", "t", "seed"],
            suffixes=("_qrc", "_esn"),
            validate="many_to_many",
        )
        merged_all_widths["qrc_minus_esn_test_auc"] = (
            merged_all_widths.test_auc_qrc - merged_all_widths.test_auc_esn
        )
        merged_all_widths.to_csv(
            RESULT_ROOT / "paired_qrc_vs_all_esn_widths_all_q_t.csv", index=False
        )
        (
            merged_all_widths.groupby(
                ["dataset", "token_variant", "system_qrc", "q", "t", "width"],
                as_index=False,
            )
            .agg(
                mean_delta_auc=("qrc_minus_esn_test_auc", "mean"),
                std_delta_auc=("qrc_minus_esn_test_auc", "std"),
                favourable_seeds=("qrc_minus_esn_test_auc", lambda x: int(np.sum(x > 0))),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(RESULT_ROOT / "paired_qrc_vs_all_esn_widths_summary.csv", index=False)

        qrc_selected = pd.read_csv(RESULT_ROOT / "qrc_validation_selected_t_by_seed.csv")
        esn_selected = pd.read_csv(RESULT_ROOT / "esn_validation_selected_t_by_seed.csv")
        matched_selected = esn_selected[
            esn_selected.apply(
                lambda row: int(row.width) == int(row.q * (row.q + 1) // 2), axis=1
            )
        ]
        selected_pair = qrc_selected.merge(
            matched_selected,
            on=["dataset", "token_variant", "q", "seed"],
            suffixes=("_qrc", "_esn"),
            validate="many_to_one",
        )
        selected_pair["qrc_minus_matched_esn_test_auc"] = (
            selected_pair.test_auc_qrc - selected_pair.test_auc_esn
        )
        selected_pair.to_csv(
            RESULT_ROOT / "paired_validation_selected_qrc_vs_matched_esn.csv", index=False
        )
        (
            selected_pair.groupby(
                ["dataset", "token_variant", "system_qrc", "q"], as_index=False
            )
            .agg(
                qrc_test_auc_mean=("test_auc_qrc", "mean"),
                esn_test_auc_mean=("test_auc_esn", "mean"),
                mean_delta_auc=("qrc_minus_matched_esn_test_auc", "mean"),
                std_delta_auc=("qrc_minus_matched_esn_test_auc", "std"),
                favourable_seeds=(
                    "qrc_minus_matched_esn_test_auc",
                    lambda x: int(np.sum(x > 0)),
                ),
                selected_qrc_t_mean=("t_qrc", "mean"),
                selected_esn_t_mean=("t_esn", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(
            RESULT_ROOT / "paired_validation_selected_qrc_vs_matched_esn_summary.csv",
            index=False,
        )
        esn_joint = pd.read_csv(RESULT_ROOT / "esn_validation_selected_width_t_by_seed.csv")
        selected_best_pair = qrc_selected.merge(
            esn_joint,
            on=["dataset", "token_variant", "q", "seed"],
            suffixes=("_qrc", "_esn"),
            validate="many_to_one",
        )
        selected_best_pair["qrc_minus_best_width_esn_test_auc"] = (
            selected_best_pair.test_auc_qrc - selected_best_pair.test_auc_esn
        )
        selected_best_pair.to_csv(
            RESULT_ROOT / "paired_validation_selected_qrc_vs_best_width_esn.csv", index=False
        )
        (
            selected_best_pair.groupby(
                ["dataset", "token_variant", "system_qrc", "q"], as_index=False
            )
            .agg(
                qrc_test_auc_mean=("test_auc_qrc", "mean"),
                esn_test_auc_mean=("test_auc_esn", "mean"),
                mean_delta_auc=("qrc_minus_best_width_esn_test_auc", "mean"),
                std_delta_auc=("qrc_minus_best_width_esn_test_auc", "std"),
                favourable_seeds=(
                    "qrc_minus_best_width_esn_test_auc",
                    lambda x: int(np.sum(x > 0)),
                ),
                selected_qrc_t_mean=("t_qrc", "mean"),
                selected_esn_t_mean=("t_esn", "mean"),
                selected_esn_width_mean=("width", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(
            RESULT_ROOT / "paired_validation_selected_qrc_vs_best_width_esn_summary.csv",
            index=False,
        )
        qrc_joint = pd.read_csv(
            RESULT_ROOT / "qrc_validation_selected_q_t_by_seed.csv"
        )
        esn_matched_joint = pd.read_csv(
            RESULT_ROOT / "esn_matched_validation_selected_q_t_by_seed.csv"
        )
        joint_matched_pair = qrc_joint.merge(
            esn_matched_joint,
            on=["dataset", "token_variant", "seed"],
            suffixes=("_qrc", "_esn"),
            validate="many_to_one",
        )
        joint_matched_pair["qrc_minus_matched_esn_test_auc"] = (
            joint_matched_pair.test_auc_qrc - joint_matched_pair.test_auc_esn
        )
        joint_matched_pair.to_csv(
            RESULT_ROOT
            / "paired_validation_selected_q_t_qrc_vs_matched_esn.csv",
            index=False,
        )
        (
            joint_matched_pair.groupby(
                ["dataset", "token_variant", "system_qrc"], as_index=False
            )
            .agg(
                qrc_test_auc_mean=("test_auc_qrc", "mean"),
                esn_test_auc_mean=("test_auc_esn", "mean"),
                mean_delta_auc=("qrc_minus_matched_esn_test_auc", "mean"),
                std_delta_auc=("qrc_minus_matched_esn_test_auc", "std"),
                favourable_seeds=(
                    "qrc_minus_matched_esn_test_auc",
                    lambda x: int(np.sum(x > 0)),
                ),
                selected_qrc_q_mean=("q_qrc", "mean"),
                selected_qrc_t_mean=("t_qrc", "mean"),
                selected_esn_q_mean=("q_esn", "mean"),
                selected_esn_t_mean=("t_esn", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(
            RESULT_ROOT
            / "paired_validation_selected_q_t_qrc_vs_matched_esn_summary.csv",
            index=False,
        )
        esn_all_joint = pd.read_csv(
            RESULT_ROOT / "esn_validation_selected_q_width_t_by_seed.csv"
        )
        joint_best_pair = qrc_joint.merge(
            esn_all_joint,
            on=["dataset", "token_variant", "seed"],
            suffixes=("_qrc", "_esn"),
            validate="many_to_one",
        )
        joint_best_pair["qrc_minus_best_esn_test_auc"] = (
            joint_best_pair.test_auc_qrc - joint_best_pair.test_auc_esn
        )
        joint_best_pair.to_csv(
            RESULT_ROOT / "paired_validation_selected_q_t_qrc_vs_best_esn.csv",
            index=False,
        )
        (
            joint_best_pair.groupby(
                ["dataset", "token_variant", "system_qrc"], as_index=False
            )
            .agg(
                qrc_test_auc_mean=("test_auc_qrc", "mean"),
                esn_test_auc_mean=("test_auc_esn", "mean"),
                mean_delta_auc=("qrc_minus_best_esn_test_auc", "mean"),
                std_delta_auc=("qrc_minus_best_esn_test_auc", "std"),
                favourable_seeds=(
                    "qrc_minus_best_esn_test_auc",
                    lambda x: int(np.sum(x > 0)),
                ),
                selected_qrc_q_mean=("q_qrc", "mean"),
                selected_qrc_t_mean=("t_qrc", "mean"),
                selected_esn_q_mean=("q_esn", "mean"),
                selected_esn_t_mean=("t_esn", "mean"),
                selected_esn_width_mean=("width", "mean"),
                n_seeds=("seed", "nunique"),
            )
        ).to_csv(
            RESULT_ROOT / "paired_validation_selected_q_t_qrc_vs_best_esn_summary.csv",
            index=False,
        )
    (RESULT_ROOT / "summary_status.json").write_text(
        json.dumps({"protocol_version": PROTOCOL_VERSION, **outputs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs
