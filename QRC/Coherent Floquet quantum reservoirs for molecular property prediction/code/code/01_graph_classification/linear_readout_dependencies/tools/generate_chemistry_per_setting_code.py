"""Generate chemistry/task-aware local code packages for FINAL/2 linear readout."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(0, 100, 10))
TOKEN_VARIANT = "chemistry_task_aware"


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _notebook(dataset: str, q: int, t: int, setting_dir: Path) -> dict:
    intro = f"""# {dataset} chemistry BFS linear readout — Q={q}, T={t}

This notebook runs the local chemistry/task-aware token variant for this exact
setting.  The scan, QRC, CRC, raw-full/raw-PCA baselines and linear training logic are
identical to the general branch; only `token_variant` changes to
`chemistry_task_aware`.
"""
    imports = f"""from pathlib import Path
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt
import torch

SETTING_DIR = Path(r"{setting_dir.resolve()}").resolve()
PLAN_PATH = SETTING_DIR / "setting_plan.json"
PROJECT_ROOT = Path(r"{PROJECT_ROOT.resolve()}").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.linear_readout import run_setting_plan, plot_single_setting_summary

print("setting:", SETTING_DIR)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
"""
    config = """RUN_QRC = True
FORCE = False

plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
display(pd.Series(plan, name="setting_plan"))
"""
    protocol = """## Protocol locked in this folder

- Token stream: chemistry/task-aware BFS event stream.
- Traversal, padding, Q-prefix and T-prefix are identical to the general branch.
- Task-aware chemistry slots contain fixed, label-free, dataset-directed descriptors.
- QRC/CRC3-z/zz/CRC/raw-full/raw-PCA/training/selection are exactly the same as the general branch; CRC3-z/zz is direct and never uses PCA.
"""
    run = """metrics, summary = run_setting_plan(
    PLAN_PATH,
    run_qrc=RUN_QRC,
    force=FORCE,
)

display(summary)
display(metrics.sort_values(["method", "seed"]).tail(12))
"""
    visualize = """fig = plot_single_setting_summary(SETTING_DIR, plan["dataset"])
plt.show()

print("summary:", SETTING_DIR / "performance_summary.csv")
print("predictions:", SETTING_DIR / "test_predictions.csv")
"""
    return {
        "cells": [
            _cell("markdown", intro),
            _cell("code", imports),
            _cell("code", config),
            _cell("markdown", protocol),
            _cell("code", run),
            _cell("code", visualize),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Molecule", "language": "python", "name": "molecule"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _scan_notebook(dataset: str, scan_root: Path, scan_kind: str) -> dict:
    title = "Q=T scan" if scan_kind == "q_equal_t" else "fixed-Q/T scan"
    imports_extra = "prepare_q_equal_t_layout, plot_q_equal_t_scan" if scan_kind == "q_equal_t" else "prepare_fixed_q_layout, plot_fixed_q_scan"
    prepare_call = "prepare_q_equal_t_layout(SCAN_ROOT)" if scan_kind == "q_equal_t" else "prepare_fixed_q_layout(SCAN_ROOT)"
    plot_call = "plot_q_equal_t_scan(SCAN_ROOT, CFG.dataset)" if scan_kind == "q_equal_t" else "plot_fixed_q_scan(SCAN_ROOT, CFG.dataset)"
    intro = f"""# {dataset} chemistry BFS — {title}

Chemistry/task-aware token variant.  All reservoir, CRC, raw-full/raw-PCA and linear
readout logic is the same as the general branch.
"""
    imports = f"""from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT.resolve()}").resolve()
SCAN_ROOT = Path(r"{scan_root.resolve()}").resolve()
DATA_DIR = PROJECT_ROOT / "chemistry" / "{dataset}" / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from core.linear_readout import (
    DEFAULT_SEEDS, ExperimentConfig,
    run_scan, collect_scan_results,
    {imports_extra},
)

print("project:", PROJECT_ROOT)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
"""
    config = f"""SETTINGS = {prepare_call}
CFG = ExperimentConfig(
    dataset="{dataset}",
    data_dir=str(DATA_DIR),
    scan_root=str(SCAN_ROOT),
    token_variant="{TOKEN_VARIANT}",
    epochs=500,
    seeds=DEFAULT_SEEDS,
    W=0.4,
    g=0.84,
    layers=3,
    qrc_devices=("cuda:0", "cuda:1"),
    crc_device="cuda:0",
    train_device="cuda:0",
    run_qrc=True,
    force=False,
)

print(CFG)
print("settings:", len(SETTINGS), SETTINGS[0], "...", SETTINGS[-1])
"""
    run = """results = run_scan(CFG, SETTINGS)
display(results)
"""
    plot = f"""summary = collect_scan_results(SCAN_ROOT)
display(summary)

fig = {plot_call}
plt.show()
"""
    return {
        "cells": [
            _cell("markdown", intro),
            _cell("code", imports),
            _cell("code", config),
            _cell("code", run),
            _cell("code", plot),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Molecule", "language": "python", "name": "molecule"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _script() -> str:
    return '''"""Run this exact chemistry/task-aware Q/T setting."""

from __future__ import annotations

from pathlib import Path
import sys


SETTING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SETTING_DIR
while not (PROJECT_ROOT / "core" / "linear_readout.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Could not locate FINAL/2_linear_readout project root")
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.linear_readout import run_setting_plan


RUN_QRC = True
FORCE = False


def main() -> None:
    metrics, summary = run_setting_plan(
        SETTING_DIR / "setting_plan.json",
        run_qrc=RUN_QRC,
        force=FORCE,
    )
    print("\\n=== performance_summary.csv ===")
    print(summary.to_string(index=False))
    print("\\n=== last metrics_by_seed rows ===")
    print(metrics.sort_values(["method", "seed"]).tail(12).to_string(index=False))
    print(f"\\nOutputs written under: {SETTING_DIR}")


if __name__ == "__main__":
    main()
'''


def _readme(dataset: str, q: int, t: int, scan_kind: str, setting_dir: Path) -> str:
    return f"""# {dataset} chemistry BFS linear readout — Q={q}, T={t}

这个目录是 chemistry/task-aware tokens 分支的最小实验单元。
除 tokens 的任务相关 slots 打开以外，QRC、CRC、raw-full/raw-PCA baseline、训练和可视化逻辑均与 general 分支一致。

- scan kind: `{scan_kind}`
- token_variant: `{TOKEN_VARIANT}`
- QRC: `W=0.4`, `g=0.84`, `L=3`
- seeds: `0,10,...,90`

```bash
cd "{setting_dir.resolve()}"
/home/ubuntu/miniconda3/envs/Molecule/bin/python run_this_setting.py
```
"""


def _plan(dataset: str, q: int, t: int, scan_kind: str, scan_root: Path, setting_dir: Path) -> dict:
    return {
        "dataset": dataset,
        "token_variant": TOKEN_VARIANT,
        "scan_kind": scan_kind,
        "q": q,
        "t": t,
        "token_dim": 3 * q,
        "project_root": str(PROJECT_ROOT.resolve()),
        "scan_root": str(scan_root.resolve()),
        "setting_dir": str(setting_dir.resolve()),
        "data_dir": str((PROJECT_ROOT / "chemistry" / dataset / "data").resolve()),
        "epochs": 500,
        "seeds": SEEDS,
        "W": 0.4,
        "g": 0.84,
        "layers": 3,
        "qrc_devices": ["cuda:0", "cuda:1"],
        "crc_device": "cuda:0",
        "train_device": "cuda:0",
        "run_qrc": True,
        "force": False,
        "methods": ["qrc", "crc3_z_zz_matched", "esn_standard", "crc", "raw_full", "raw_pca"],
        "notes": [
            "Chemistry/task-aware tokens; logic otherwise identical to general branch.",
            "Task slots contain fixed label-free dataset-directed descriptors.",
        ],
    }


def _write_setting(dataset: str, q: int, t: int, scan_kind: str, scan_root: Path, setting_dir: Path) -> None:
    setting_dir.mkdir(parents=True, exist_ok=True)
    (setting_dir / "setting_plan.json").write_text(
        json.dumps(_plan(dataset, q, t, scan_kind, scan_root, setting_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (setting_dir / "run_this_setting.py").write_text(_script(), encoding="utf-8")
    (setting_dir / "README.md").write_text(_readme(dataset, q, t, scan_kind, setting_dir), encoding="utf-8")
    (setting_dir / "run_this_setting.ipynb").write_text(
        json.dumps(_notebook(dataset, q, t, setting_dir), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    written = 0
    for dataset in ("BACE", "BBBP"):
        dataset_root = PROJECT_ROOT / "chemistry" / dataset
        qeq_root = dataset_root / "01_Q_equal_T_scan"
        fixed_root = dataset_root / "02_fixed_Q_T_scan"
        for value in range(1, 21):
            _write_setting(dataset, value, value, "Q_equal_T", qeq_root, qeq_root / f"Q{value:02d}_T{value:02d}")
            written += 1
        for q in (8, 12, 16, 20):
            for t in range(1, 31):
                _write_setting(dataset, q, t, "fixed_Q_T_scan", fixed_root, fixed_root / f"Q{q:02d}" / f"T{t:02d}")
                written += 1
        (dataset_root / f"{dataset}_chemistry_Q_equal_T_scan.ipynb").write_text(
            json.dumps(_scan_notebook(dataset, qeq_root, "q_equal_t"), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (dataset_root / f"{dataset}_chemistry_fixed_Q_T_scan.ipynb").write_text(
            json.dumps(_scan_notebook(dataset, fixed_root, "fixed_q"), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"Generated chemistry local code packages for {written} minimum setting folders.")


if __name__ == "__main__":
    main()
