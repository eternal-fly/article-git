"""Resumable Hamiltonian-setting QRC feature stages."""

from __future__ import annotations

import argparse
import json

import numpy as np

from hamiltonian_qrc import (
    compute_suffix_banks,
    load_suffix_bank,
    save_suffix_bank,
)
from protocol import (
    DATASETS,
    G,
    JOINT_QT_VALUES,
    Q_VALUES,
    SEEDS,
    T_VALUES,
    TOKEN_VARIANTS,
    W,
    WG_DATASET,
    WG_Q,
    WG_T,
    WG_TOKEN_VARIANT,
    WG_VALUES,
    qrc_bank_path,
    source_token_path,
)


def _csv_tuple(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _load_tokens(dataset: str, variant: str, q: int) -> tuple[np.ndarray, str]:
    path = source_token_path(dataset, variant)
    with np.load(path, allow_pickle=False) as archive:
        values = archive["tokens"].astype(np.float32, copy=False)
        token_sha = str(archive["tokens_sha256"].item())
    return np.ascontiguousarray(values[:, :, : 3 * int(q)], dtype=np.float32), token_sha


def _cached(
    dataset: str, variant: str, q: int, t: int, seed: int, g: float, W_value: float,
    n_samples: int, token_sha: str,
) -> bool:
    path = qrc_bank_path(
        dataset, variant, q, t, seed, g=g, W_value=W_value
    )
    if not path.exists():
        return False
    try:
        load_suffix_bank(
            path, q=q, t=t, seed=seed, g=g, W_scale=W_value,
            n_samples=n_samples,
            expected_metadata={
                "dataset": dataset.upper(),
                "token_variant": variant,
                "source_token_sha256": token_sha,
            },
        )
        return True
    except Exception:
        return False


def run_group(
    dataset: str,
    variant: str,
    q: int,
    t_values: tuple[int, ...],
    seed: int,
    *,
    device: str,
    g: float,
    W_value: float,
    force: bool,
    max_samples: int | None,
    batch_size: int | None,
) -> list[dict]:
    tokens, token_sha = _load_tokens(dataset, variant, q)
    if max_samples is not None:
        tokens = tokens[: int(max_samples)]
    pending = tuple(
        t for t in t_values
        if force or not _cached(
            dataset, variant, q, t, seed, g, W_value, len(tokens), token_sha
        )
    )
    if not pending:
        return [{
            "dataset": dataset, "variant": variant, "q": q, "seed": seed,
            "g": g, "W": W_value, "status": "cached",
        }]
    banks, metadata = compute_suffix_banks(
        tokens,
        q=q,
        t_values=pending,
        seed=seed,
        device=device,
        g=g,
        W_scale=W_value,
        batch_size=batch_size,
    )
    metadata.update(
        {
            "dataset": dataset.upper(),
            "token_variant": variant,
            "source_token_sha256": token_sha,
        }
    )
    rows = []
    for t, bank in banks.items():
        destination = qrc_bank_path(
            dataset, variant, q, t, seed, g=g, W_value=W_value
        )
        save_suffix_bank(destination, bank, metadata, t)
        rows.append({
            "dataset": dataset, "variant": variant, "q": q, "t": t,
            "seed": seed, "g": g, "W": W_value, "status": "computed",
            "path": str(destination.resolve()),
        })
    return rows


def fixed_q(args: argparse.Namespace) -> list[dict]:
    rows = []
    for dataset in args.datasets:
        for variant in args.variants:
            for q in args.q_values:
                for seed in args.seeds:
                    rows += run_group(
                        dataset, variant, q, args.t_values, seed,
                        device=args.device, g=G, W_value=W, force=args.force,
                        max_samples=args.max_samples, batch_size=args.batch_size,
                    )
    return rows


def joint(args: argparse.Namespace) -> list[dict]:
    rows = []
    for q in args.joint_q_values:
        for seed in args.seeds:
            rows += run_group(
                "BACE", "canonical_general", q, (q,), seed,
                device=args.device, g=G, W_value=W, force=args.force,
                max_samples=args.max_samples, batch_size=args.batch_size,
            )
    return rows


def wg(args: argparse.Namespace) -> list[dict]:
    rows = []
    for g in args.wg_values:
        for W_value in args.wg_values:
            for seed in args.seeds:
                rows += run_group(
                    WG_DATASET, WG_TOKEN_VARIANT, WG_Q, (WG_T,), seed,
                    device=args.device, g=g, W_value=W_value,
                    force=args.force, max_samples=args.max_samples,
                    batch_size=args.batch_size,
                )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fixed-q", "joint", "wg"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--variants", default=",".join(TOKEN_VARIANTS))
    parser.add_argument("--q-values", default=",".join(map(str, Q_VALUES)))
    parser.add_argument("--t-values", default=",".join(map(str, T_VALUES)))
    parser.add_argument("--joint-q-values", default=",".join(map(str, JOINT_QT_VALUES)))
    parser.add_argument("--wg-values", default=",".join(map(str, WG_VALUES)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.datasets = tuple(value.upper() for value in _csv_tuple(args.datasets))
    args.variants = _csv_tuple(args.variants)
    args.q_values = _csv_tuple(args.q_values, int)
    args.t_values = _csv_tuple(args.t_values, int)
    args.joint_q_values = _csv_tuple(args.joint_q_values, int)
    args.wg_values = _csv_tuple(args.wg_values, float)
    args.seeds = _csv_tuple(args.seeds, int)
    runners = {"fixed-q": fixed_q, "joint": joint, "wg": wg}
    rows = runners[args.stage](args)
    print(json.dumps({"stage": args.stage, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
