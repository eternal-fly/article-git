#!/usr/bin/env python3
"""Create self-contained Jupyter task folders and one upload archive."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from prepare_dataset import (
    DATASET_OUTPUT,
    IDEAL_L1_OUTPUT,
    IDEAL_L3_OUTPUT,
    PROTOCOL_VERSION,
    build_dataset,
    build_ideal_features,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
CLOUD_ROOT = HERE / "experiment"
COMMON_SOURCE = HERE / "experiment_common.py"


TASKS = (
    {
        "folder": "01_q6_t6_l3_batch01",
        "batch_id": "q6_t6_l3_batch01",
        "layers": 3,
        "ideal": IDEAL_L3_OUTPUT,
        "variants": [{"name": "base", "fold_factor": 1}],
        "order_seed": 31001,
        "paired_interleave": False,
    },
    {
        "folder": "02_q6_t6_l3_batch02",
        "batch_id": "q6_t6_l3_batch02",
        "layers": 3,
        "ideal": IDEAL_L3_OUTPUT,
        "variants": [{"name": "base", "fold_factor": 1}],
        "order_seed": 31002,
        "paired_interleave": False,
    },
    {
        "folder": "03_q6_t6_l3_batch03",
        "batch_id": "q6_t6_l3_batch03",
        "layers": 3,
        "ideal": IDEAL_L3_OUTPUT,
        "variants": [{"name": "base", "fold_factor": 1}],
        "order_seed": 31003,
        "paired_interleave": False,
    },
    {
        "folder": "04_q6_t6_l1_fold_pair_batch01",
        "batch_id": "q6_t6_l1_fold_pair_batch01",
        "layers": 1,
        "ideal": IDEAL_L1_OUTPUT,
        "variants": [
            {"name": "base", "fold_factor": 1},
            {"name": "fold3", "fold_factor": 3},
        ],
        "order_seed": 41001,
        "paired_interleave": True,
    },
)


ROOT_README = """# Quafu mainline hardware-noise tasks

Upload this complete `experiment` directory to Quafu Jupyter.  Each numbered
subdirectory is one self-contained task and must be run from inside that
directory.

| Folder | Scientific unit | Circuits | Nominal CX/circuit | Purpose |
|---|---:|---:|---:|---|
| `01_q6_t6_l3_batch01` | device batch 1 | 400 | 180 | mainline ideal-versus-hardware transfer |
| `02_q6_t6_l3_batch02` | device batch 2 | 400 | 180 | independent device repetition |
| `03_q6_t6_l3_batch03` | device batch 3 | 400 | 180 | independent device repetition |
| `04_q6_t6_l1_fold_pair_batch01` | paired device batch | 800 | 60/180 | interleaved unitary-preserving noise amplification |

Inside every task, run:

1. `01_dataset_and_validate.ipynb`;
2. `02_submit.ipynb`;
3. after every circuit is complete, `03_collect_and_train.ipynb`.

Notebooks 02 and 03 request the API token with `getpass`; the value is not
stored in the notebook.  Run notebook 02 first with `MAX_SUBMIT = 1`; inspect
the compiled circuit and then change it to `None` to resume all missing
submissions.  For `fold3`, the compiled platform circuit must retain 180
two-qubit gates.

The ten scaffold/readout seeds are nested analyses of one hardware feature
batch.  They are not ten device repetitions.  Device-level robustness is
assessed only across the three independent `q6_t6_l3_batch*` folders.
"""


def _source_lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def code_notebook(cells: list[str], title: str) -> dict:
    """Return a dependency-free nbformat-v4 notebook payload."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n"],
            },
            *[
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": _source_lines(cell),
                }
                for cell in cells
            ],
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_notebook(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_notebooks(folder: Path) -> None:
    _write_notebook(
        folder / "01_dataset_and_validate.ipynb",
        code_notebook(
            [
                "from experiment_common import validate_dataset_and_config, dry_run",
                "validation = validate_dataset_and_config('config.json')\nvalidation",
                "# This builds one circuit only and does not submit anything.\npreview = dry_run('config.json')\npreview",
            ],
            "01 — dataset and circuit validation",
        ),
    )
    _write_notebook(
        folder / "02_submit.ipynb",
        code_notebook(
            [
                "from getpass import getpass\n# The entered value is held in memory and is not saved in the notebook source.\nAPI_TOKEN = getpass('New QPU API token: ')\nMAX_SUBMIT = 1  # first run: smoke submission; after inspection use None to resume all missing circuits",
                "from experiment_common import dry_run, submit_jobs",
                "# Always inspect this before the first submission.\ndry_run('config.json')",
                "manifest = submit_jobs(API_TOKEN, 'config.json', max_submit=MAX_SUBMIT)\nmanifest",
            ],
            "02 — randomized Quafu submission",
        ),
    )
    _write_notebook(
        folder / "03_collect_and_train.ipynb",
        code_notebook(
            [
                "from getpass import getpass\n# Re-enter the token; it is not read from or written to the notebook file.\nAPI_TOKEN = getpass('New QPU API token: ')",
                "from experiment_common import collect_jobs, train_and_analyze, shot_noise_bootstrap",
                "feature_files = collect_jobs(API_TOKEN, 'config.json', require_complete=True)\nfeature_files",
                "metrics, summary = train_and_analyze('config.json')\nsummary",
                "# Statistical shot-noise diagnostic; increase N_BOOT to 1000–2000 for the final analysis.\nN_BOOT = 500\ndraws, shot_summary = shot_noise_bootstrap('config.json', n_boot=N_BOOT)\nshot_summary",
            ],
            "03 — count collection, linear readout, and shot bootstrap",
        ),
    )


def write_scripts(folder: Path) -> None:
    (folder / "01_dataset_and_validate.py").write_text(
        "from experiment_common import validate_dataset_and_config, dry_run\n"
        "validate_dataset_and_config('config.json')\n"
        "dry_run('config.json')\n",
        encoding="utf-8",
    )
    (folder / "02_submit.py").write_text(
        "import os\nfrom experiment_common import dry_run, submit_jobs\n"
        "dry_run('config.json')\n"
        "submit_jobs(os.environ['QPU_API_TOKEN'], 'config.json')\n",
        encoding="utf-8",
    )
    (folder / "03_collect_and_train.py").write_text(
        "import os\nfrom experiment_common import collect_jobs, train_and_analyze, shot_noise_bootstrap\n"
        "collect_jobs(os.environ['QPU_API_TOKEN'], 'config.json', require_complete=True)\n"
        "train_and_analyze('config.json')\n"
        "shot_noise_bootstrap('config.json', n_boot=500)\n",
        encoding="utf-8",
    )


def task_config(task: dict, dataset_name: str, ideal_name: str) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "batch_id": task["batch_id"],
        "scientific_role": (
            "independent Q6/T6/L3 device batch"
            if task["layers"] == 3
            else "paired ideal-unitary-preserving noise amplification"
        ),
        "files": {
            "dataset": dataset_name,
            "dataset_sha256": sha256_file(DATASET_OUTPUT),
            "ideal_features": ideal_name,
            "ideal_features_sha256": sha256_file(task["ideal"]),
        },
        "circuit": {
            "q": 6,
            "t": 6,
            "layers": task["layers"],
            "g": 0.84,
            "W": 0.6,
            "reservoir_seed": 0,
            "features_per_qubit": 3,
            "token_schema": "canonical-v3",
            "interaction": "CX-RZ(J_i)-CX",
            "J_distribution": "Uniform(-0.9,0.9)",
            "readout": "Z+all_ZZ",
        },
        "variants": task["variants"],
        "cloud": {
            "chip": "Baihua",
            "repeat": 10,
            "max_two_qubit_gates": 200,
            "mapping_mode": "logical_auto",
            "backend_num_qubits": 6,
            "physical_qubits": [],
            "compile": True,
            "compiler": "quarkcircuit",
            "correct": False,
            "open_dd": None,
        },
        "submission": {
            "order_seed": task["order_seed"],
            "paired_interleave": task["paired_interleave"],
            "sleep_seconds": 0.0,
        },
        "training": {
            "model": "torch.nn.Linear(d,1)",
            "epochs": 500,
            "learning_rate": 0.001,
            "batch_size": 32,
            "weight_decay": 0.0,
            "standardization": False,
            "seeds": list(range(0, 100, 10)),
            "selection_rule": "max(val_auc) > min(val_loss) > max(val_acc)",
            "device": "auto",
        },
        "interpretation": {
            "split_readout_seeds_are_hardware_repeats": False,
            "independent_unit_for_device_robustness": "batch_id",
            "primary_metric": "frozen ideal-readout hardware AUC and paired delta AUC",
            "hardware_refit_metric": "secondary snapshot only",
        },
    }


def task_readme(task: dict) -> str:
    variants = ", ".join(
        f"{value['name']} (fold={value['fold_factor']})"
        for value in task["variants"]
    )
    return f"""# {task['batch_id']}

Self-contained Quafu task for the manuscript mainline canonical-v3 tokens.

- Q=T=6, L={task['layers']}, g=0.84, W=0.6, reservoir seed 0.
- Variants: {variants}.
- Nominal CX count is checked against the 200-gate cap before submission.
- The dataset contains 400 molecules (200/200), sampled once with seed 42.
- Ten scaffold/readout seeds reuse the same hardware feature batch and are not
  independent device repeats.
- No API credential is included. Notebooks 02 and 03 use a non-echoing
  `getpass` prompt, or set `QPU_API_TOKEN` for the Python scripts.

Run in order:

1. `01_dataset_and_validate.ipynb`
2. `02_submit.ipynb`
3. after all jobs finish, `03_collect_and_train.ipynb`

For the fold3 task, circuit generation fails if the installed Quark circuit
object cannot emit barriers. After compilation, verify on the Quafu page that
the folded circuit still contains the intended 180 two-qubit gates; otherwise
the noise-amplification comparison is invalid.
"""


def build() -> Path:
    if not DATASET_OUTPUT.exists():
        build_dataset()
    if not IDEAL_L1_OUTPUT.exists() or not IDEAL_L3_OUTPUT.exists():
        build_ideal_features()
    CLOUD_ROOT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        folder = CLOUD_ROOT / task["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        dataset_target = folder / DATASET_OUTPUT.name
        ideal_target = folder / task["ideal"].name
        shutil.copy2(DATASET_OUTPUT, dataset_target)
        shutil.copy2(task["ideal"], ideal_target)
        shutil.copy2(COMMON_SOURCE, folder / COMMON_SOURCE.name)
        config = task_config(task, dataset_target.name, ideal_target.name)
        (folder / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        (folder / "README.md").write_text(task_readme(task), encoding="utf-8")
        write_notebooks(folder)
        write_scripts(folder)
    (CLOUD_ROOT / "README.md").write_text(ROOT_README, encoding="utf-8")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "credential_included": False,
        "dataset": DATASET_OUTPUT.name,
        "dataset_sha256": sha256_file(DATASET_OUTPUT),
        "tasks": [
            {
                "folder": task["folder"],
                "batch_id": task["batch_id"],
                "q": 6,
                "t": 6,
                "layers": task["layers"],
                "circuits": 400 * len(task["variants"]),
                "variants": [
                    {
                        **variant,
                        "nominal_two_qubit_gates": (
                            2
                            * 6
                            * int(task["layers"])
                            * (6 - 1)
                            * int(variant["fold_factor"])
                        ),
                    }
                    for variant in task["variants"]
                ],
                "config_sha256": sha256_file(
                    CLOUD_ROOT / task["folder"] / "config.json"
                ),
            }
            for task in TASKS
        ],
    }
    (CLOUD_ROOT / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive = HERE / "quafu_mainline_hardware_noise_experiment.zip"
    with zipfile.ZipFile(
        archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(CLOUD_ROOT.rglob("*")):
            relative = path.relative_to(HERE)
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or ".ipynb_checkpoints" in path.parts
                or path.suffix == ".pyc"
                or "runs" in path.relative_to(CLOUD_ROOT).parts
            ):
                continue
            handle.write(path, relative.as_posix())
    print(f"[archive] {archive}")
    return archive


if __name__ == "__main__":
    build()
