#!/usr/bin/env python3
"""Reconstruct and train the four Shenglian Q6/T6/L1 acquisition batches.

The primary estimator exactly follows the manuscript mainline: one
bias-containing torch.nn.Linear(D, 1) readout per feature domain, 500 epochs,
and checkpoint selection by val AUC > val loss > val accuracy.  The hardware
unit is an acquisition batch; the ten scaffold/readout seeds are nested
analyses of each fixed feature matrix.
"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QRC_ROOT = ROOT.parent
MAINLINE = QRC_ROOT / "Quafu_mainline_Q6T6L3_experiment"
MAINLINE_LOCAL = MAINLINE / "local"
sys.path.insert(0, str(MAINLINE_LOCAL))
import run_local_analysis as common  # noqa: E402


DATASET = ROOT / "data" / "bace_bal400_canonical_v3_q6_t6.npz"
IDEAL = MAINLINE / "data" / "ideal_bace_bal400_canonical_v3_q6_t6_l1_seed0.npz"
ORIGINAL_SAVED = (
    MAINLINE
    / "local_results_five_tasks"
    / "hardware_features_shenglian_04_q6_t6_l1_base_batch01.npz"
)
OUTPUT = ROOT / "local_analysis_results_r4"
EXTRACTED = OUTPUT / "extracted_raw"
EXPECTED_DATASET_SHA256 = (
    "00b3fdbab64cb95f61b8580bd7e975ee9f1c9d7df1b44ce60ebe61a643a3eb04"
)
EXPECTED_CHAIN = [27, 34, 41, 48, 54, 61]
EXPECTED_SHOTS = 1024
FAMILIES = {
    "Z": np.arange(0, 6, dtype=np.int64),
    "ZZ": np.arange(6, 21, dtype=np.int64),
    "Z+ZZ": np.arange(0, 21, dtype=np.int64),
}
BATCHES = {
    "original": {
        "archive": (
            MAINLINE
            / "downloaded_five_tasks"
            / "shenglian_04_q6_t6_l1_base_batch01_cloud_results.zip"
        ),
        "manifest_glob": "**/shenglian_04_q6_t6_l1_base_batch01_job_manifest.jsonl",
        "result_tag": "04_q6_t6_l1_base_batch01",
    },
    "r2": {
        "archive": ROOT / "sl_q6t6l1_r2_results.zip",
        "manifest_glob": "**/sl_q6t6l1_r2_jobs.jsonl",
        "result_tag": "sl_q6t6l1_r2",
    },
    "r3": {
        "archive": ROOT / "sl_q6t6l1_r3_results.zip",
        "manifest_glob": "**/sl_q6t6l1_r3_jobs.jsonl",
        "result_tag": "sl_q6t6l1_r3",
    },
    "r4": {
        "archive": ROOT / "03_sl_q6t6l1_r4" / "sl_q6t6l1_r4_results.zip",
        "manifest_glob": "**/sl_q6t6l1_r4_jobs.jsonl",
        "result_tag": "sl_q6t6l1_r4",
    },
}


def scalar_text(value) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def load_bundle() -> dict:
    common.DATASET_PATH = DATASET
    return common.load_dataset()


def load_ideal() -> dict:
    with np.load(IDEAL, allow_pickle=False) as package:
        return {
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "token_hashes": package["token_hashes"].astype(str),
            "features": package["features"].astype(np.float32),
        }


def validate_record(record: dict, batch: str) -> None:
    required = {
        "q": 6,
        "t": 6,
        "layers": 1,
        "reservoir_seed": 0,
        "nominal_two_qubit_gates": 60,
    }
    for key, expected in required.items():
        if int(record[key]) != expected:
            raise AssertionError((batch, key, record[key], expected))
    if int(record.get("fold_factor", 1)) != 1:
        raise AssertionError((batch, "fold_factor", record.get("fold_factor")))
    if not math.isclose(float(record["g"]), 0.84, abs_tol=1e-12):
        raise AssertionError((batch, "g", record["g"]))
    if not math.isclose(float(record["W"]), 0.60, abs_tol=1e-12):
        raise AssertionError((batch, "W", record["W"]))
    if list(map(int, record["physical_qubits"])) != EXPECTED_CHAIN:
        raise AssertionError((batch, "physical_qubits"))
    if record.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise AssertionError((batch, "dataset_sha256"))
    if record.get("compile") is not True:
        raise AssertionError((batch, "compile"))
    options = record.get("task_options", {})
    if options.get("compiler") != "quarkcircuit" or options.get("correct") is not False:
        raise AssertionError((batch, "task_options", options))


def locate_raw_file(manifest: Path, raw_name: str) -> Path:
    matches = list(manifest.parent.rglob(raw_name))
    if len(matches) != 1:
        raise FileNotFoundError((raw_name, matches))
    return matches[0]


def reconstruct_batch(batch: str, spec: dict, bundle: dict) -> tuple[np.ndarray, dict]:
    archive = Path(spec["archive"])
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination = EXTRACTED / batch
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise RuntimeError((archive, bad))
        handle.extractall(destination)
    manifests = list(destination.glob(spec["manifest_glob"]))
    if len(manifests) != 1:
        raise FileNotFoundError((batch, manifests))
    manifest = manifests[0]
    records = common.read_jsonl(manifest)
    if len(records) != 400:
        raise AssertionError((batch, len(records)))

    row_by_index = {
        int(original): row for row, original in enumerate(bundle["indices"].tolist())
    }
    observed = []
    calibration_notes = set()
    tids = set()
    for record in records:
        validate_record(record, batch)
        original_index = int(record["original_index"])
        row = row_by_index[original_index]
        if str(record["token_hash"]) != str(bundle["token_hashes"][row]):
            raise AssertionError((batch, "token_hash", original_index))
        raw_name = record.get("raw_file_name")
        if not raw_name:
            raise AssertionError((batch, "raw_file_name", original_index))
        raw_path = locate_raw_file(manifest, raw_name)
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        counts = common.extract_counts(result)
        if not counts:
            raise RuntimeError((batch, "missing counts", raw_path))
        shots = int(round(sum(float(value) for value in counts.values())))
        if shots != EXPECTED_SHOTS:
            raise AssertionError((batch, original_index, shots))
        tid = str(record["tid"])
        if tid in tids:
            raise AssertionError((batch, "duplicate tid", tid))
        tids.add(tid)
        calibration_notes.add(str(record.get("calibration_timestamp_from_image", "")))
        observed.append(
            (
                row,
                original_index,
                int(record["submission_rank"]),
                common.counts_to_features(counts, record),
                shots,
                tid,
            )
        )
    observed.sort(key=lambda value: value[0])
    if [value[1] for value in observed] != bundle["indices"].astype(int).tolist():
        raise AssertionError((batch, "index restoration"))
    ranks = sorted(value[2] for value in observed)
    if ranks != list(range(400)):
        raise AssertionError((batch, "submission ranks"))
    features = np.stack([value[3] for value in observed]).astype(np.float32)
    audit = {
        "batch": batch,
        "result_tag": spec["result_tag"],
        "archive": str(archive.resolve()),
        "manifest": str(manifest.resolve()),
        "record_count": len(records),
        "unique_tid_count": len(tids),
        "shots_min": min(value[4] for value in observed),
        "shots_max": max(value[4] for value in observed),
        "feature_shape": list(features.shape),
        "calibration_notes": sorted(calibration_notes),
        "submission_rank_restored_to_dataset_order": True,
    }
    output_path = OUTPUT / f"hardware_features_shenglian_l1_{batch}.npz"
    np.savez_compressed(
        output_path,
        indices=bundle["indices"].astype(np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        features=features,
        shots=np.asarray([value[4] for value in observed], dtype=np.int64),
        tids=np.asarray([value[5] for value in observed], dtype=str),
        submission_ranks=np.asarray([value[2] for value in observed], dtype=np.int64),
        audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
    )
    return features, audit


def centered_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    denominator = float(
        np.linalg.norm(x.T @ x, ord="fro")
        * np.linalg.norm(y.T @ y, ord="fro")
    )
    return numerator / denominator if denominator > 0 else float("nan")


def feature_geometry(domains: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    names = list(domains)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            x = domains[left]
            y = domains[right]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "centered_linear_cka": centered_linear_cka(x, y),
                    "flattened_pearson": float(
                        np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]
                    ),
                    "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def train_domains(
    bundle: dict,
    domains: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    transfer_rows = []
    labels = bundle["labels"]
    for family, columns in FAMILIES.items():
        family_domains = {
            name: np.ascontiguousarray(features[:, columns], dtype=np.float32)
            for name, features in domains.items()
        }
        for seed in common.SEEDS:
            split = common.split_rows(bundle, seed)
            test = split["test"]
            fitted = {}
            for domain, features in family_domains.items():
                state, fit = common.train_ideal_linear_head(
                    features,
                    labels,
                    split,
                    seed,
                )
                fitted[domain] = state
                scores = common.logits(features[test], state)
                result = common.metrics(labels[test], scores)
                metric_rows.append(
                    {
                        "family": family,
                        "feature_dim": len(columns),
                        "domain": domain,
                        "seed": seed,
                        "n_test": len(test),
                        "best_epoch": fit["best_epoch"],
                        "val_auc": fit["val_auc"],
                        "val_loss": fit["val_loss"],
                        "val_acc": fit["val_acc"],
                        "test_auc": result["auc"],
                        "test_loss": result["loss"],
                        "test_acc": result["acc"],
                        "readout": "single torch.nn.Linear(D,1), bias=True",
                        "selection_rule": common.SELECTION_RULE,
                    }
                )
            for source, state in fitted.items():
                for target, target_features in family_domains.items():
                    scores = common.logits(target_features[test], state)
                    result = common.metrics(labels[test], scores)
                    transfer_rows.append(
                        {
                            "family": family,
                            "feature_dim": len(columns),
                            "source_head_domain": source,
                            "target_feature_domain": target,
                            "seed": seed,
                            "test_auc": result["auc"],
                            "test_loss": result["loss"],
                            "test_acc": result["acc"],
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(transfer_rows)


def build_summaries(
    metrics: pd.DataFrame,
    transfers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary = (
        metrics.groupby(["family", "feature_dim", "domain"], as_index=False)
        .agg(
            nested_seed_count=("seed", "nunique"),
            auc_mean=("test_auc", "mean"),
            auc_sd_nested=("test_auc", "std"),
            acc_mean=("test_acc", "mean"),
            loss_mean=("test_loss", "mean"),
        )
    )
    ideal = summary[summary["domain"] == "ideal"][
        ["family", "auc_mean"]
    ].rename(columns={"auc_mean": "ideal_auc_mean"})
    summary = summary.merge(ideal, on="family", validate="many_to_one")
    summary["auc_minus_ideal"] = summary["auc_mean"] - summary["ideal_auc_mean"]

    transfer_summary = (
        transfers.groupby(
            ["family", "feature_dim", "source_head_domain", "target_feature_domain"],
            as_index=False,
        )
        .agg(
            nested_seed_count=("seed", "nunique"),
            auc_mean=("test_auc", "mean"),
            auc_sd_nested=("test_auc", "std"),
        )
    )

    full = summary[summary["family"] == "Z+ZZ"].set_index("domain")
    hardware_names = list(BATCHES)
    hardware_batch_means = [float(full.loc[name, "auc_mean"]) for name in hardware_names]
    ideal_mean = float(full.loc["ideal", "auc_mean"])
    frozen = transfer_summary[
        (transfer_summary["family"] == "Z+ZZ")
        & (transfer_summary["source_head_domain"] == "ideal")
        & (transfer_summary["target_feature_domain"].isin(hardware_names))
    ]
    off_diagonal = transfer_summary[
        (transfer_summary["family"] == "Z+ZZ")
        & (transfer_summary["source_head_domain"].isin(hardware_names))
        & (transfer_summary["target_feature_domain"].isin(hardware_names))
        & (
            transfer_summary["source_head_domain"]
            != transfer_summary["target_feature_domain"]
        )
    ]
    diagnosis = {
        "statistical_unit": "acquisition-batch mean after averaging ten nested scaffold/readout analyses",
        "hardware_batch_count": len(hardware_names),
        "ideal_auc_mean": ideal_mean,
        "hardware_batch_auc_means": dict(zip(hardware_names, hardware_batch_means)),
        "hardware_across_batch_mean": float(np.mean(hardware_batch_means)),
        "hardware_across_batch_sd": float(np.std(hardware_batch_means, ddof=1)),
        "hardware_minus_ideal_across_batch_mean": float(
            np.mean(hardware_batch_means) - ideal_mean
        ),
        "hardware_batches_above_ideal_mean": int(
            sum(value > ideal_mean for value in hardware_batch_means)
        ),
        "batch_averaged_feature_auc": float(full.loc["batch_mean", "auc_mean"]),
        "batch_averaged_feature_minus_ideal": float(
            full.loc["batch_mean", "auc_mean"] - ideal_mean
        ),
        "ideal_head_to_hardware_auc_mean": float(frozen["auc_mean"].mean()),
        "off_diagonal_hardware_head_transfer_auc_mean": float(
            off_diagonal["auc_mean"].mean()
        ),
        "interpretation_boundary": (
            "Four batches can test whether the original excess is repeatable. "
            "They cannot identify a beneficial noise channel without a matched "
            "noise model/intervention, and same-window acquisitions are not "
            "independent calibration repeats."
        ),
    }
    return summary, transfer_summary, diagnosis


def write_report(
    summary: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    geometry: pd.DataFrame,
    audits: list[dict],
    diagnosis: dict,
) -> None:
    full = summary[summary["family"] == "Z+ZZ"].set_index("domain")
    hardware_names = list(BATCHES)
    report_domains = ["ideal", *hardware_names, "batch_mean"]
    family_rows = summary[summary["domain"].isin(report_domains)].copy()
    lines = [
        "# Shenglian Q6/T6/L1 four-batch audit",
        "",
        "Primary protocol: one bias-containing `torch.nn.Linear(D,1)` per feature "
        "domain, 500 epochs, selected by val AUC > val loss > val accuracy.",
        "",
        "## Full Z+ZZ readout",
        "",
        "| Domain | Mean test AUC | Nested-seed SD | Minus ideal |",
        "|---|---:|---:|---:|",
    ]
    for name in report_domains:
        row = full.loc[name]
        lines.append(
            f"| {name} | {row['auc_mean']:.4f} | {row['auc_sd_nested']:.4f} | "
            f"{row['auc_minus_ideal']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Hardware-batch mean ± sample SD across the four acquisition means: "
            f"{diagnosis['hardware_across_batch_mean']:.4f} ± "
            f"{diagnosis['hardware_across_batch_sd']:.4f}. ",
            f"Batches above the fixed ideal mean: "
            f"{diagnosis['hardware_batches_above_ideal_mean']}/4.",
            "",
            "## Observable families",
            "",
            "| Family | Domain | Mean test AUC | Minus ideal |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in family_rows.sort_values(["family", "domain"]).iterrows():
        lines.append(
            f"| {row['family']} | {row['domain']} | {row['auc_mean']:.4f} | "
            f"{row['auc_minus_ideal']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Transfer diagnostics",
            "",
            f"- Ideal-head → hardware mean AUC: "
            f"{diagnosis['ideal_head_to_hardware_auc_mean']:.4f}.",
            f"- Off-diagonal hardware-head → other hardware-batch mean AUC: "
            f"{diagnosis['off_diagonal_hardware_head_transfer_auc_mean']:.4f}.",
            f"- Readout on the molecule-wise mean of four hardware feature matrices: "
            f"{diagnosis['batch_averaged_feature_auc']:.4f}.",
            "",
            "## Acquisition audit",
            "",
        ]
    )
    for audit in audits:
        lines.append(
            f"- {audit['batch']}: {audit['record_count']} unique records, "
            f"shots {audit['shots_min']}–{audit['shots_max']}, calibration note(s) "
            f"{audit['calibration_notes']}."
        )
    lines.extend(
        [
            "",
            "The ten seeds are nested analyses of a fixed batch and are not ten "
            "hardware repetitions. Acquisition-window interpretation follows the "
            "timestamps recorded in each manifest; order randomization prevents a "
            "molecule-index drift confound but does not by itself create independent "
            "device calibrations.",
            "",
            "No conclusion of beneficial quantum noise is made from AUC alone. A "
            "repeatable domain-specific excess can also arise from systematic feature "
            "deformation that a separately fitted linear head exploits, finite test-set "
            "variation, readout/shot bias, or a genuinely useful stochastic "
            "regularization effect. Frozen-head and cross-batch transfer distinguish "
            "coordinate preservation from batch-specific recalibration but do not "
            "identify a physical noise channel.",
        ]
    )
    (OUTPUT / "ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    ideal = load_ideal()
    if not np.array_equal(ideal["indices"], bundle["indices"]):
        raise AssertionError("ideal indices")
    if not np.array_equal(ideal["token_hashes"], bundle["token_hashes"]):
        raise AssertionError("ideal token hashes")

    hardware = {}
    audits = []
    for batch, spec in BATCHES.items():
        features, audit = reconstruct_batch(batch, spec, bundle)
        hardware[batch] = features
        audits.append(audit)
    with np.load(ORIGINAL_SAVED, allow_pickle=False) as package:
        saved_original = package["features"].astype(np.float32)
    if not np.allclose(saved_original, hardware["original"], atol=1e-7, rtol=0):
        raise AssertionError("Original raw-count reconstruction differs from mainline saved features")

    domains = {
        "ideal": ideal["features"],
        **hardware,
        "batch_mean": np.mean(np.stack(list(hardware.values())), axis=0).astype(np.float32),
    }
    metrics, transfers = train_domains(bundle, domains)
    geometry = feature_geometry(domains)
    summary, transfer_summary, diagnosis = build_summaries(metrics, transfers)

    metrics.to_csv(OUTPUT / "l1_four_batch_metrics_by_nested_seed.csv", index=False)
    summary.to_csv(OUTPUT / "l1_four_batch_summary.csv", index=False)
    transfers.to_csv(OUTPUT / "l1_cross_domain_transfer_by_nested_seed.csv", index=False)
    transfer_summary.to_csv(OUTPUT / "l1_cross_domain_transfer_summary.csv", index=False)
    geometry.to_csv(OUTPUT / "l1_feature_geometry.csv", index=False)
    (OUTPUT / "RAW_ACQUISITION_AUDIT.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "DIAGNOSIS_SUMMARY.json").write_text(
        json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary, transfer_summary, geometry, audits, diagnosis)
    print(summary.to_string(index=False))
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
