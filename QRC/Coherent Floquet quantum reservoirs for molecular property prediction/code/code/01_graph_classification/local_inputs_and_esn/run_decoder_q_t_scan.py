#!/usr/bin/env python3
"""Full fixed-Q, scanned-T migration for SVR and regularized MLP heads.

The reservoir banks are reused exactly.  For QRC_OPEN, p is the value selected
at each Q/T/seed point by the completed affine validation-only p scan.  Thus
this stage expands the decoder-only sensitivity analysis over the complete
Q/T grid without silently performing a second, decoder-specific p search.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from decoder_ablation import DECODERS, fit_regularized_mlp, fit_svr, load_case_features
from protocol import PROTOCOL_VERSION, Q_VALUES, RESULT_ROOT, SEEDS, T_VALUES, readout_dim


GRID_DECODER_PROTOCOL = "fixed-q-scan-t-affine-p-decoder-migration-v1"
GRID_ROOT = RESULT_ROOT / "decoder_q_t_scan"


def grid_case_rows() -> pd.DataFrame:
    """Return all 2,880 reservoir representations in the original Q/T grid."""
    qrc = pd.read_csv(RESULT_ROOT / "qrc_metrics_all_q_t_seeds.csv")
    esn = pd.read_csv(RESULT_ROOT / "esn_metrics_all_q_t_seeds.csv")
    rows = []
    for _, row in qrc.iterrows():
        rows.append(
            {
                "dataset": row.dataset,
                "token_variant": row.token_variant,
                "system": row.system,
                "q": int(row.q),
                "t": int(row.t),
                "seed": int(row.seed),
                "p": float(row.p),
                "width": np.nan,
                "affine_test_auc": float(row.test_auc),
                "affine_test_acc": float(row.test_acc),
                "affine_test_brier": float(row.test_brier),
            }
        )
    for _, row in esn.iterrows():
        rows.append(
            {
                "dataset": row.dataset,
                "token_variant": row.token_variant,
                "system": "ESN",
                "q": int(row.q),
                "t": int(row.t),
                "seed": int(row.seed),
                "p": np.nan,
                "width": int(row.width),
                "affine_test_auc": float(row.test_auc),
                "affine_test_acc": float(row.test_acc),
                "affine_test_brier": float(row.test_brier),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "token_variant", "system", "q", "t", "width", "seed"],
        na_position="first",
    )
    expected = 2 * 2 * len(Q_VALUES) * len(T_VALUES) * len(SEEDS) * (2 + 4)
    keys = ["dataset", "token_variant", "system", "q", "t", "seed"]
    qrc_frame = frame[frame.system != "ESN"]
    esn_frame = frame[frame.system == "ESN"]
    if len(frame) != expected:
        raise RuntimeError(f"invalid Q/T decoder grid: {len(frame)} != {expected}")
    if qrc_frame.duplicated(keys).any() or esn_frame.duplicated(keys + ["width"]).any():
        raise RuntimeError("duplicate Q/T decoder grid keys")
    return frame.reset_index(drop=True)


def _p_tag(probability: float) -> str:
    return f"p{float(probability):.3f}".replace(".", "p")


def grid_result_dir(row: dict | pd.Series, decoder: str) -> Path:
    base = (
        GRID_ROOT
        / str(row["dataset"]).upper()
        / str(row["token_variant"])
        / str(row["system"])
        / f"Q{int(row['q']):02d}"
        / f"T{int(row['t']):02d}"
    )
    representation = (
        _p_tag(float(row["p"]))
        if str(row["system"]).startswith("QRC_")
        else f"width{int(row['width']):04d}"
    )
    return base / representation / decoder / f"seed{int(row['seed']):03d}"


def evaluate_grid_case(
    row: dict | pd.Series,
    decoder: str,
    *,
    device: str = "cpu",
    force: bool = False,
) -> dict:
    destination = grid_result_dir(row, decoder)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not force:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_version") == PROTOCOL_VERSION
            and payload.get("decoder_protocol") == GRID_DECODER_PROTOCOL
        ):
            return payload
    features, labels, split, selection = load_case_features(row)
    if decoder == "SVR_RBF":
        metrics = fit_svr(
            features,
            labels,
            split,
            seed=int(row["seed"]),
            destination=destination,
            decoder_protocol=GRID_DECODER_PROTOCOL,
        )
    elif decoder == "REG_MLP":
        metrics = fit_regularized_mlp(
            features,
            labels,
            split,
            seed=int(row["seed"]),
            device=device,
            destination=destination,
            decoder_protocol=GRID_DECODER_PROTOCOL,
        )
    else:
        raise ValueError(f"unsupported decoder {decoder}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "decoder_protocol": GRID_DECODER_PROTOCOL,
        "selection_protocol": (
            "complete fixed-Q/T representation grid; QRC_OPEN p frozen from "
            "the original affine validation-only p scan"
        ),
        "dataset": str(row["dataset"]).upper(),
        "token_variant": str(row["token_variant"]),
        "system": str(row["system"]),
        "decoder": decoder,
        "seed": int(row["seed"]),
        **selection,
        **metrics,
    }
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _svr_worker(row: dict, force: bool) -> dict:
    return evaluate_grid_case(row, "SVR_RBF", force=force)


def run_svr(cases: list[dict], workers: int, force: bool) -> list[dict]:
    context = mp.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_svr_worker, case, force) for case in cases]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 20 == 0 or completed == len(futures):
                print(f"SVR {completed}/{len(futures)}", flush=True)
    return rows


def _mlp_process_entry(device: str, task_queue, result_queue, force: bool) -> None:
    try:
        completed = 0
        while True:
            case = task_queue.get()
            if case is None:
                break
            evaluate_grid_case(case, "REG_MLP", device=device, force=force)
            completed += 1
            if completed % 10 == 0:
                print(f"[{device}] MLP completed {completed}", flush=True)
        result_queue.put({"ok": True, "device": device, "completed": completed})
    except Exception as error:
        result_queue.put(
            {
                "ok": False,
                "device": device,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )


def run_mlp(cases: list[dict], devices: list[str], force: bool) -> int:
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for case in cases:
        task_queue.put(case)
    for _ in devices:
        task_queue.put(None)
    processes = [
        context.Process(
            target=_mlp_process_entry,
            args=(device, task_queue, result_queue, force),
        )
        for device in devices
    ]
    for process in processes:
        process.start()
    messages = []
    try:
        for _ in processes:
            messages.append(result_queue.get())
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    failures = [message for message in messages if not message["ok"]]
    if failures:
        raise RuntimeError(f"{failures[0]['error']}\n{failures[0]['traceback']}")
    return int(sum(message["completed"] for message in messages))


def run_mlp_direct(cases: list[dict], device: str, force: bool) -> int:
    """Run one recoverable shard in the current process.

    This mode is useful in restricted launch environments where CUDA is
    available to the parent command but not inherited by multiprocessing
    children.  Independent commands should receive disjoint shard indices.
    """
    for completed, case in enumerate(cases, start=1):
        evaluate_grid_case(case, "REG_MLP", device=device, force=force)
        if completed % 10 == 0 or completed == len(cases):
            print(f"[{device}] direct MLP {completed}/{len(cases)}", flush=True)
    return len(cases)


def collect(strict: bool = True) -> pd.DataFrame:
    rows = []
    missing = []
    for case in grid_case_rows().to_dict("records"):
        for decoder in DECODERS:
            path = grid_result_dir(case, decoder) / "metrics.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("protocol_version") != PROTOCOL_VERSION
                or payload.get("decoder_protocol") != GRID_DECODER_PROTOCOL
            ):
                raise RuntimeError(f"stale Q/T decoder result {path}")
            rows.append(payload)
    if strict and missing:
        raise FileNotFoundError(
            f"missing {len(missing)} Q/T decoder results; first={missing[0]}"
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        qrc = frame[frame.system != "ESN"]
        esn = frame[frame.system == "ESN"]
        keys = ["dataset", "token_variant", "system", "decoder", "q", "t", "seed"]
        if qrc.duplicated(keys).any() or esn.duplicated(keys + ["width"]).any():
            raise RuntimeError("duplicate collected Q/T decoder result keys")
    return frame


def _select_best(group: pd.DataFrame, *, include_width: bool = False) -> pd.Series:
    decoder = str(group.decoder.iloc[0])
    ranked = group.copy()
    if decoder == "REG_MLP":
        ranked["secondary"] = -ranked.val_loss
    else:
        ranked["secondary"] = -ranked.val_brier
    sort_columns = ["val_auc", "secondary", "val_acc"]
    ascending = [False, False, False]
    if include_width:
        sort_columns.append("width")
        ascending.append(True)
    sort_columns.append("t")
    ascending.append(True)
    return ranked.sort_values(sort_columns, ascending=ascending).iloc[0]


def summarize(strict: bool = True) -> dict:
    frame = collect(strict=strict)
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(GRID_ROOT / "decoder_q_t_metrics_all.csv", index=False)

    curves = frame.copy()
    curves["width_key"] = curves.width.fillna(-1).astype(int)
    (
        curves.groupby(
            ["dataset", "token_variant", "system", "decoder", "q", "t", "width_key"],
            as_index=False,
        )
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            val_auc_mean=("val_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .to_csv(GRID_ROOT / "decoder_q_t_curves.csv", index=False)
    )

    selected_t_rows = []
    group_columns = [
        "dataset", "token_variant", "system", "decoder", "q", "seed", "width_key"
    ]
    for _, group in curves.groupby(group_columns, sort=False):
        selected_t_rows.append(_select_best(group).to_dict())
    selected_t = pd.DataFrame(selected_t_rows)
    selected_t["selected_t_by"] = (
        "per-seed fixed-Q validation: decoder rule then smaller T"
    )
    selected_t.to_csv(GRID_ROOT / "validation_selected_t_by_q_seed.csv", index=False)

    qrc_selected = selected_t[selected_t.system != "ESN"].copy()
    esn_selected = selected_t[selected_t.system == "ESN"].copy()
    matched = esn_selected[
        esn_selected.apply(lambda row: int(row.width) == readout_dim(int(row.q)), axis=1)
    ].copy()
    matched["system"] = "ESN_MATCHED"

    best_esn_rows = []
    for _, group in curves[curves.system == "ESN"].groupby(
        ["dataset", "token_variant", "decoder", "q", "seed"], sort=False
    ):
        row = _select_best(group, include_width=True).to_dict()
        row["system"] = "ESN_BEST_WIDTH"
        best_esn_rows.append(row)
    best_esn = pd.DataFrame(best_esn_rows)
    best_esn["selected_width_t_by"] = (
        "per-seed fixed-Q validation: decoder rule then smaller width then smaller T"
    )
    best_esn.to_csv(
        GRID_ROOT / "esn_validation_selected_width_t_by_q_seed.csv", index=False
    )

    systems = pd.concat([qrc_selected, matched, best_esn], ignore_index=True)
    systems.to_csv(GRID_ROOT / "fixed_q_validation_selected_by_seed.csv", index=False)
    summary = (
        systems.groupby(
            ["dataset", "token_variant", "decoder", "system", "q"], as_index=False
        )
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_brier_mean=("test_brier", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(GRID_ROOT / "fixed_q_validation_selected_summary.csv", index=False)

    pair_specs = (
        ("QRC_OPEN", "QRC_CLOSED"),
        ("QRC_OPEN", "ESN_MATCHED"),
        ("QRC_OPEN", "ESN_BEST_WIDTH"),
        ("QRC_CLOSED", "ESN_MATCHED"),
        ("QRC_CLOSED", "ESN_BEST_WIDTH"),
    )
    paired_rows = []
    for left, right in pair_specs:
        lhs = systems[systems.system == left]
        rhs = systems[systems.system == right]
        merged = lhs.merge(
            rhs,
            on=["dataset", "token_variant", "decoder", "q", "seed"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "dataset": row.dataset,
                    "token_variant": row.token_variant,
                    "decoder": row.decoder,
                    "q": int(row.q),
                    "comparison": f"{left}_MINUS_{right}",
                    "seed": int(row.seed),
                    "left_test_auc": float(row.test_auc_left),
                    "right_test_auc": float(row.test_auc_right),
                    "delta_test_auc": float(row.test_auc_left - row.test_auc_right),
                }
            )
    pairs = pd.DataFrame(paired_rows)
    pairs.to_csv(GRID_ROOT / "paired_fixed_q_by_seed.csv", index=False)
    (
        pairs.groupby(
            ["dataset", "token_variant", "decoder", "q", "comparison"],
            as_index=False,
        )
        .agg(
            left_auc_mean=("left_test_auc", "mean"),
            right_auc_mean=("right_test_auc", "mean"),
            mean_delta_auc=("delta_test_auc", "mean"),
            std_delta_auc=("delta_test_auc", "std"),
            favourable_seeds=("delta_test_auc", lambda x: int(np.sum(x > 0))),
            n_seeds=("seed", "nunique"),
        )
        .to_csv(GRID_ROOT / "paired_fixed_q_summary.csv", index=False)
    )
    status = {
        "protocol_version": PROTOCOL_VERSION,
        "decoder_protocol": GRID_DECODER_PROTOCOL,
        "metric_rows": len(frame),
        "representation_cases": len(grid_case_rows()),
        "selected_fixed_q_rows": len(systems),
    }
    (GRID_ROOT / "summary_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    return status


def audit(strict: bool = True) -> dict:
    frame = collect(strict=strict)
    expected = len(grid_case_rows()) * len(DECODERS)
    errors = []
    if len(frame) != expected:
        errors.append(f"metric rows {len(frame)} != {expected}")
    if len(frame) and not np.isfinite(
        frame[["val_auc", "test_auc", "test_acc", "test_brier"]]
    ).all().all():
        errors.append("non-finite decoder metrics")
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "decoder_protocol": GRID_DECODER_PROTOCOL,
        "expected_rows": expected,
        "valid_rows": len(frame),
        "errors": errors,
    }
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    (GRID_ROOT / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if strict and errors:
        raise AssertionError(errors[0])
    return report


def _filter_cases(cases: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = cases
    if args.datasets:
        allowed = {value.upper() for value in args.datasets.split(",")}
        selected = selected[selected.dataset.str.upper().isin(allowed)]
    if args.variants:
        allowed = {value.strip() for value in args.variants.split(",")}
        selected = selected[selected.token_variant.isin(allowed)]
    if args.systems:
        allowed = {value.strip().upper() for value in args.systems.split(",")}
        selected = selected[selected.system.str.upper().isin(allowed)]
    if args.q:
        allowed = {int(value) for value in args.q.split(",")}
        selected = selected[selected.q.isin(allowed)]
    if args.t:
        allowed = {int(value) for value in args.t.split(",")}
        selected = selected[selected.t.isin(allowed)]
    if args.shard_count > 1:
        indices = np.arange(len(selected))
        selected = selected.iloc[indices % args.shard_count == args.shard_index]
    return selected.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cases", "svr", "mlp", "summarize", "audit"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--systems", default="")
    parser.add_argument("--q", default="")
    parser.add_argument("--t", default="")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    cases = _filter_cases(grid_case_rows(), args).to_dict("records")
    if args.stage == "cases":
        result = {"cases": len(cases)}
    elif args.stage == "svr":
        result = {"completed": len(run_svr(cases, max(1, args.workers), args.force))}
    elif args.stage == "mlp":
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        if not devices:
            raise ValueError("at least one MLP device is required")
        if args.direct:
            if len(devices) != 1:
                raise ValueError("direct MLP mode accepts exactly one device")
            result = {"completed": run_mlp_direct(cases, devices[0], args.force)}
        else:
            result = {"completed": run_mlp(cases, devices, args.force)}
    elif args.stage == "summarize":
        result = summarize(strict=args.strict)
    else:
        result = audit(strict=args.strict)
    print(json.dumps({"stage": args.stage, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
