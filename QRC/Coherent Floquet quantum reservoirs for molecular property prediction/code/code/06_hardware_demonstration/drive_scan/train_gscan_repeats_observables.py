#!/usr/bin/env python3
"""Train three Shenglian acquisition repeats for Z/ZZ/Z+ZZ feature domains.

Each acquisition repeat is summarized only after averaging the same ten fixed
scaffold/readout seeds.  Cross-repeat variability is therefore computed from
three hardware batches, not from 30 nested readout analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
QRC_ROOT = HERE.parent
COMMON_DIR = QRC_ROOT / "Quafu_mainline_Q6T6L3_experiment" / "local"
BASE_SCRIPT = HERE / "train_gscan_hardware.py"
DATASET_PATH = HERE / "bace_bal400_canonical_v3_q6_t6.npz"
OUTPUT_DIR = HERE / "repeat_observable_training_results"
EXTRACT_DIR = HERE / "extracted_repeat_results"
SEEDS = tuple(range(0, 100, 10))
G_POINTS = {"040": 0.40, "060": 0.60, "100": 1.00}
REPEATS = (1, 2, 3)
FAMILIES = {
    "Z": np.arange(0, 6, dtype=np.int64),
    "ZZ": np.arange(6, 21, dtype=np.int64),
    "Z+ZZ": np.arange(0, 21, dtype=np.int64),
}
G084_BATCHES = {
    "Z": (0.5340023910978194, 0.6050059716995139, 0.5193359637881537),
    "ZZ": (0.6024223063553147, 0.6205758987235650, 0.5288806975201317),
    "Z+ZZ": (0.5873445169440534, 0.6397015361071627, 0.5279731430310715),
}
G084_SOURCE = (
    QRC_ROOT / "Quafu_mainline_Q6T6L3_experiment" / "local_results_five_tasks"
    / "observable_family_ablation" / "observable_family_summary.csv"
)
PROTOCOL = "shenglian-q6t6l3-gscan-three-acquisition-linear-observable-audit-v1"

sys.path.insert(0, str(COMMON_DIR))
import run_local_analysis as common  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "shenglian_gscan_base")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset() -> dict:
    bundle = base.load_dataset()
    common.DATASET_PATH = DATASET_PATH
    return bundle


def result_tag(tag: str, repeat: int) -> str:
    return f"sl_g{tag}" if repeat == 1 else f"sl_g{tag}_r{repeat}"


def archive_path(tag: str, repeat: int) -> Path:
    return HERE / f"{result_tag(tag, repeat)}_results.zip"


def locate_manifest(tag: str, repeat: int) -> Path:
    archive = archive_path(tag, repeat)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination = EXTRACT_DIR / result_tag(tag, repeat)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    candidates = sorted(destination.glob(f"**/{result_tag(tag, repeat)}_jobs.jsonl"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"g={tag}, repeat={repeat}: expected one manifest, got {candidates}"
        )
    return candidates[0]


def validate_record(record: dict, g_value: float, repeat: int) -> None:
    expected = {
        "q": 6,
        "t": 6,
        "layers": 3,
        "reservoir_seed": 0,
        "nominal_two_qubit_gates": 180,
    }
    for key, value in expected.items():
        if int(record[key]) != value:
            raise AssertionError((key, record[key], value))
    if not np.isclose(float(record["g"]), g_value, atol=1e-8):
        raise AssertionError((record["g"], g_value))
    if not np.isclose(float(record["W"]), 0.60, atol=1e-8):
        raise AssertionError((record["W"], 0.60))
    if repeat > 1 and int(record.get("acquisition_round", -1)) != repeat:
        raise AssertionError((record.get("acquisition_round"), repeat))


def build_hardware_features(tag: str, g_value: float, repeat: int, bundle: dict) -> Path:
    feature_dir = OUTPUT_DIR / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    output = feature_dir / f"hardware_{result_tag(tag, repeat)}_z_zz.npz"
    manifest = locate_manifest(tag, repeat)
    records = common.read_jsonl(manifest)
    if len(records) != 400:
        raise RuntimeError(f"{result_tag(tag, repeat)}: {len(records)}/400 records")
    if len({str(record["tid"]) for record in records}) != 400:
        raise AssertionError(f"{result_tag(tag, repeat)}: duplicate task IDs")

    row_by_index = {
        int(original): row for row, original in enumerate(bundle["indices"].tolist())
    }
    raw_dir = manifest.parent / f"raw_{result_tag(tag, repeat)}"
    rows = []
    for record in records:
        validate_record(record, g_value, repeat)
        raw_name = str(record.get("raw_file_name", ""))
        if not raw_name:
            raise RuntimeError(f"{result_tag(tag, repeat)}: missing raw filename")
        raw_path = raw_dir / raw_name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw_result = json.loads(raw_path.read_text(encoding="utf-8"))
        counts = common.extract_counts(raw_result)
        if not counts:
            raise RuntimeError(f"No counts in {raw_path}")
        original = int(record["original_index"])
        dataset_row = row_by_index[original]
        if str(record["token_hash"]) != str(bundle["token_hashes"][dataset_row]):
            raise AssertionError(f"Token mismatch for molecule {original}")
        rows.append(
            (
                dataset_row,
                original,
                common.counts_to_features(counts, record),
                int(round(sum(float(value) for value in counts.values()))),
                str(record["tid"]),
                int(record["submission_rank"]),
            )
        )
    rows.sort(key=lambda value: value[0])
    if [value[1] for value in rows] != bundle["indices"].astype(int).tolist():
        raise AssertionError("Restored molecule order differs from immutable dataset order")
    features = np.stack([value[2] for value in rows]).astype(np.float32)
    if features.shape != (400, 21):
        raise AssertionError(features.shape)
    shots = np.asarray([value[3] for value in rows], dtype=np.int64)
    np.savez_compressed(
        output,
        indices=np.asarray([value[1] for value in rows], dtype=np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        features=features,
        shots=shots,
        tids=np.asarray([value[4] for value in rows], dtype=str),
        submission_ranks=np.asarray([value[5] for value in rows], dtype=np.int64),
        g=np.asarray(g_value, dtype=np.float32),
        repeat=np.asarray(repeat, dtype=np.int64),
        source_manifest=np.asarray(str(manifest.resolve())),
        protocol=np.asarray(PROTOCOL),
    )
    print(
        f"[features] {result_tag(tag, repeat)} shape={features.shape} "
        f"shots={shots.min()}-{shots.max()}"
    )
    return output


def load_features(path: Path, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        if not np.array_equal(package["indices"], bundle["indices"]):
            raise AssertionError(path)
        return package["features"].astype(np.float32), package["shots"].astype(np.int64)


def train_matrix(
    features: np.ndarray,
    bundle: dict,
    tag: str,
    g_value: float,
    repeat: int,
    family: str,
    source: str,
) -> list[dict]:
    selected = np.ascontiguousarray(features[:, FAMILIES[family]], dtype=np.float32)
    rows = []
    for seed in SEEDS:
        split = common.split_rows(bundle, seed)
        state, fit = common.train_ideal_linear_head(
            selected, bundle["labels"], split, seed
        )
        test = split["test"]
        score = common.logits(selected[test], state)
        test_metric = common.metrics(bundle["labels"][test], score)
        rows.append(
            {
                "tag": f"g{tag}",
                "g": g_value,
                "repeat": repeat,
                "source": source,
                "family": family,
                "feature_dim": selected.shape[1],
                "seed": seed,
                "n_test": len(test),
                "best_epoch": fit["best_epoch"],
                "val_auc": fit["val_auc"],
                "val_loss": fit["val_loss"],
                "val_acc": fit["val_acc"],
                "test_auc": test_metric["auc"],
                "test_loss": test_metric["loss"],
                "test_acc": test_metric["acc"],
                "readout": f"single torch.nn.Linear({selected.shape[1]},1) layer",
                "weight_decay": 0.0,
                "selection_rule": common.SELECTION_RULE,
                "protocol": PROTOCOL,
            }
        )
    return rows


def train_tag(tag: str, g_value: float, repeats: tuple[int, ...], bundle: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in repeats:
        feature_path = build_hardware_features(tag, g_value, repeat, bundle)
        features, shots = load_features(feature_path, bundle)
        for family in FAMILIES:
            rows.extend(
                train_matrix(features, bundle, tag, g_value, repeat, family, "hardware")
            )
        print(
            f"[trained] {result_tag(tag, repeat)} all families; "
            f"mean shots={shots.mean():.1f}"
        )
    table = pd.DataFrame(rows).sort_values(["repeat", "family", "seed"])
    table.to_csv(OUTPUT_DIR / f"metrics_g{tag}_repeats_observables.csv", index=False)


def collect_metrics() -> pd.DataFrame:
    tables = []
    for tag in G_POINTS:
        path = OUTPUT_DIR / f"metrics_g{tag}_repeats_observables.csv"
        if path.is_file():
            tables.append(pd.read_csv(path))
    if not tables:
        raise FileNotFoundError("No completed repeat metric tables")
    table = pd.concat(tables, ignore_index=True).sort_values(
        ["g", "repeat", "family", "seed"]
    )
    table.to_csv(OUTPUT_DIR / "all_nested_readout_metrics.csv", index=False)
    return table


def make_summaries(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batch = (
        metrics.groupby(["g", "repeat", "family"], as_index=False)
        .agg(
            auc_mean=("test_auc", "mean"),
            auc_sd_nested_seeds=("test_auc", "std"),
            loss_mean=("test_loss", "mean"),
            acc_mean=("test_acc", "mean"),
            nested_seed_count=("seed", "nunique"),
        )
        .sort_values(["g", "repeat", "family"])
    )
    g084_batch = pd.DataFrame(
        [
            {
                "g": 0.84,
                "repeat": repeat,
                "family": family,
                "auc_mean": auc,
                "auc_sd_nested_seeds": np.nan,
                "loss_mean": np.nan,
                "acc_mean": np.nan,
                "nested_seed_count": 10,
            }
            for family, values in G084_BATCHES.items()
            for repeat, auc in enumerate(values, start=1)
        ]
    )
    batch_all = pd.concat([batch, g084_batch], ignore_index=True).sort_values(
        ["g", "repeat", "family"]
    )
    batch_all.to_csv(OUTPUT_DIR / "hardware_batch_means.csv", index=False)
    recent = (
        batch[(batch["repeat"].isin([2, 3]))]
        .groupby(["g", "family"], as_index=False)
        .agg(
            r2_r3_auc_mean=("auc_mean", "mean"),
            r2_r3_auc_sd=("auc_mean", "std"),
            acquisition_batch_count=("repeat", "nunique"),
        )
        .sort_values(["g", "family"])
    )
    recent.to_csv(OUTPUT_DIR / "hardware_r2_r3_summary.csv", index=False)

    across = (
        batch.groupby(["g", "family"], as_index=False)
        .agg(
            auc_mean=("auc_mean", "mean"),
            auc_sd_across_batches=("auc_mean", "std"),
            auc_min_batch=("auc_mean", "min"),
            auc_max_batch=("auc_mean", "max"),
            hardware_batch_count=("repeat", "nunique"),
            nested_seed_count_per_batch=("nested_seed_count", "min"),
        )
        .sort_values(["g", "family"])
    )
    reference = pd.DataFrame(
        [
            {
                "g": 0.84,
                "family": family,
                "auc_mean": float(np.mean(values)),
                "auc_sd_across_batches": float(np.std(values, ddof=1)),
                "auc_min_batch": float(np.min(values)),
                "auc_max_batch": float(np.max(values)),
                "hardware_batch_count": 3,
                "nested_seed_count_per_batch": 10,
                "source_note": "existing canonical-v3 mainline three-batch source table",
            }
            for family, values in G084_BATCHES.items()
        ]
    )
    across["source_note"] = "recomputed from three downloaded hardware batches"
    full = pd.concat([across, reference], ignore_index=True).sort_values(["g", "family"])
    across.to_csv(OUTPUT_DIR / "hardware_cross_batch_summary_recomputed.csv", index=False)
    full.to_csv(OUTPUT_DIR / "hardware_gscan_z_zz_source_data.csv", index=False)

    pivot = batch_all.pivot_table(index=["g", "repeat"], columns="family", values="auc_mean").reset_index()
    pivot["ZZ_minus_Z"] = pivot["ZZ"] - pivot["Z"]
    pivot["ZplusZZ_minus_best_single"] = pivot["Z+ZZ"] - pivot[["Z", "ZZ"]].max(axis=1)
    pivot.to_csv(OUTPUT_DIR / "within_batch_observable_contrasts.csv", index=False)
    return batch_all, full, pivot


def plot_scan(batch: pd.DataFrame, full: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = {"Z": "#4C78A8", "ZZ": "#E07A5F", "Z+ZZ": "#2A9D8F"}
    markers = {"Z": "o", "ZZ": "s", "Z+ZZ": "D"}
    offsets = {"Z": -0.010, "ZZ": 0.0, "Z+ZZ": 0.010}
    fig, ax = plt.subplots(figsize=(3.50, 2.75), constrained_layout=True)
    for family in ("Z", "ZZ", "Z+ZZ"):
        curve = full[full["family"] == family].sort_values("g")
        ax.plot(
            curve["g"], curve["auc_mean"], color=colors[family], marker=markers[family],
            lw=1.35, ms=4.3, label=family, zorder=3,
        )
        recomputed = curve[np.isfinite(curve["auc_sd_across_batches"])]
        ax.errorbar(
            recomputed["g"], recomputed["auc_mean"],
            yerr=recomputed["auc_sd_across_batches"], fmt="none",
            ecolor=colors[family], elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=2,
        )
        points = batch[batch["family"] == family]
        for repeat in REPEATS:
            part = points[points["repeat"] == repeat]
            ax.scatter(
                part["g"] + offsets[family] + (repeat - 2) * 0.0025,
                part["auc_mean"], s=11, facecolor="white", edgecolor=colors[family],
                linewidth=0.75, alpha=0.95, zorder=4,
            )
    ax.axhline(0.5, color="#777777", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.axvspan(0.7, 0.9, color="#D9D9D9", alpha=0.28, zorder=-1)
    ax.text(0.80, 0.752, "transition-edge window", ha="center", va="top", fontsize=6.2, color="#555555")
    ax.text(0.84, 0.665, "mainline hardware\nbatches", ha="center", va="bottom", fontsize=6.0, color="#444444")
    ax.set(xlabel="Transverse-field parameter, $g$", ylabel="Test ROC–AUC", xlim=(0.35, 1.05), ylim=(0.40, 0.76))
    ax.set_xticks([0.4, 0.6, 0.84, 1.0], ["0.40", "0.60", "0.84", "1.00"])
    ax.legend(loc="lower left", ncol=3, handlelength=1.7, columnspacing=1.0)
    ax.tick_params(direction="out", length=3)
    stem = OUTPUT_DIR / "shenglian_gscan_z_zz_three_batches"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(batch: pd.DataFrame, full: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    lines = [
        "# Shenglian canonical-v3 hardware g scan: three acquisition repeats",
        "",
        "Each hardware-batch point is the mean of the same ten fixed scaffold/readout",
        "seeds. The reported spread is the SD across three hardware acquisition batches.",
        "A single unregularized `Linear(d,1)` readout was selected by validation AUC,",
        "then validation loss, then validation accuracy.",
        "",
        "| g | family | batch 1 | batch 2 | batch 3 | three-batch mean +/- SD |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for g_value in (0.40, 0.60, 0.84, 1.00):
        for family in ("Z", "ZZ", "Z+ZZ"):
            values = batch[(batch["g"] == g_value) & (batch["family"] == family)].sort_values("repeat")["auc_mean"].to_numpy()
            summary = full[(full["g"] == g_value) & (full["family"] == family)].iloc[0]
            lines.append(
                f"| {g_value:.2f} | {family} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} "
                f"| {summary.auc_mean:.3f} +/- {summary.auc_sd_across_batches:.3f} |"
            )
    lines.extend(["", "## Observable-domain contrasts", ""])
    contrast_summary = contrasts.groupby("g").agg(
        ZZ_minus_Z_mean=("ZZ_minus_Z", "mean"),
        combined_minus_best_mean=("ZplusZZ_minus_best_single", "mean"),
    )
    for g_value, row in contrast_summary.iterrows():
        lines.append(
            f"- g={g_value:.2f}: ZZ-Z = {row.ZZ_minus_Z_mean:+.3f}; "
            f"Z+ZZ-best(Z,ZZ) = {row.combined_minus_best_mean:+.3f}."
        )
    lines.extend(
        [
            "",
            "## R2/R3-only descriptive means",
            "",
            "| g | Z | ZZ | Z+ZZ |",
            "|---:|---:|---:|---:|",
        ]
    )
    recent = batch[(batch["g"].isin([0.40, 0.60, 1.00])) & (batch["repeat"].isin([2, 3]))]
    for g_value in (0.40, 0.60, 1.00):
        values = recent[recent["g"] == g_value].groupby("family")["auc_mean"].mean()
        lines.append(
            f"| {g_value:.2f} | {values['Z']:.3f} | {values['ZZ']:.3f} | {values['Z+ZZ']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The ten readout seeds within a batch are nested analyses, not independent",
            "hardware repeats. The three acquisition batches use the same molecules,",
            "tokens, split definitions, qubit chain, and nominal circuit protocol.",
            "",
            f"Dataset SHA-256: `{sha256_file(DATASET_PATH)}`",
            f"g=0.84 source-table SHA-256: `{sha256_file(G084_SOURCE)}`",
        ]
    )
    (OUTPUT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize() -> None:
    metrics = collect_metrics()
    batch, full, contrasts = make_summaries(metrics)
    plot_scan(batch, full)
    write_report(batch, full, contrasts)
    print(full[["g", "family", "auc_mean", "auc_sd_across_batches", "hardware_batch_count"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", choices=tuple(G_POINTS), default=tuple(G_POINTS))
    parser.add_argument("--repeats", nargs="*", type=int, choices=REPEATS, default=REPEATS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summarize_only:
        bundle = load_dataset()
        for tag in args.tags:
            train_tag(tag, G_POINTS[tag], tuple(args.repeats), bundle)
    if not args.no_summarize:
        summarize()


if __name__ == "__main__":
    main()
