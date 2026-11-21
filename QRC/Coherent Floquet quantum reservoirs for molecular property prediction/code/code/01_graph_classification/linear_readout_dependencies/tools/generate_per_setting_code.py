"""Generate runnable code packages inside every minimum Q/T setting folder.

The earlier scan notebooks keep the whole sweep convenient, but this project
also needs the original GPU-simulate/BACE style: every smallest experiment
folder should contain its own code entrypoint, configuration and README.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(0, 100, 10))
SOURCE_LAYOUT_REFERENCE = "/home/ubuntu/Workspace/Fei/wlf/Article/QRC/GPU-simulate/BACE"


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _notebook(dataset: str, q: int, t: int, setting_dir: Path) -> dict:
    setting_dir_literal = str(setting_dir.resolve())
    project_root_literal = str(PROJECT_ROOT.resolve())
    markdown_intro = f"""# {dataset} general BFS single setting — Q={q}, T={t}

This notebook is the local executable code for this minimum experiment folder.
It follows the original `GPU-simulate/BACE` per-setting style, but uses the
strict FINAL protocol: task-independent local-BFS tokens without global generic
priors, QRC/CRC dimensional alignment, and a single unregularized linear readout selected by validation
`AUC > loss > ACC`.
"""
    imports = f"""from pathlib import Path
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt
import torch

SETTING_DIR = Path(r"{setting_dir_literal}").resolve()
PLAN_PATH = SETTING_DIR / "setting_plan.json"
PROJECT_ROOT = Path(r"{project_root_literal}").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.linear_readout import run_setting_plan, plot_single_setting_summary

print("setting:", SETTING_DIR)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
"""
    config = """RUN_QRC = True   # set False only for CPU/debugging of CRC/raw
FORCE = False   # resume old QRC/controls and backfill only missing methods

plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
display(pd.Series(plan, name="setting_plan"))
"""
    protocol = """## Protocol locked in this folder

- Token stream: general, task-independent local-BFS event stream with no generic molecule-level descriptors, canonical ranks, full-graph ring sizes or neighbor aggregate statistics.
- Token slice: first `T` molecular events and first `3Q` token coordinates.
- QRC: `W=0.4`, `g=0.84`, `L=3`, tanh injection preprocessing, Z + all-ZZ readout.
- CRC: old scalar-Q reservoir, same input preprocessing, same number of reservoir layers, same output dimension as QRC.
- Raw-full baseline: direct flattening of the original `T × 3Q` token tensor.
- CRC3-z/zz matched: Q-by-3 classical state with direct z and all zz products; exactly Q(Q+1)/2 outputs and no PCA.
- Standard ESN: fixed Gaussian echo-state network with spectral radius 0.9 and direct QRC-dimension state; no PCA.
- Raw-PCA baseline: PCA fitted on each train split only, then compressed/padded to the QRC/CRC readout dimension.
- Classifier: exactly one `torch.nn.Linear(d, 1)`, no standardization, no hidden layer, no regularization.
- Seeds/splits: `0,10,...,90`; QRC/CRC parameters, scaffold split and training all change with seed.
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

summary_path = SETTING_DIR / "performance_summary.csv"
pred_path = SETTING_DIR / "test_predictions.csv"
print("summary:", summary_path)
print("predictions:", pred_path)
"""
    return {
        "cells": [
            _cell("markdown", markdown_intro),
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


def _script() -> str:
    return '''"""Run this exact Q/T setting.

Edit RUN_QRC/FORCE below if needed, then run:

    /home/ubuntu/miniconda3/envs/Molecule/bin/python run_this_setting.py
"""

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
    return f"""# {dataset} general BFS linear readout — Q={q}, T={t}

这个目录是一个最小实验单元，不只是结果输出目录。组织方式参考
`{SOURCE_LAYOUT_REFERENCE}` 中每个 Q/T 文件夹自带代码的思路，但这里的
tokens、QRC、CRC、训练流程均对齐 FINAL 的严格比较协议。

## 本目录代码

- `setting_plan.json`：本设置的本地配置，固定 dataset、Q、T、scan root、data dir、seeds、W/g/L 等。
- `run_this_setting.ipynb`：推荐入口；打开后从上到下运行，会完成训练和结果可视化。
- `run_this_setting.py`：脚本入口；等价运行本设置。

## 协议

- scan kind: `{scan_kind}`
- tokens: 通用、任务无关 BFS 事件流；本设置读取前 `{t}` 个事件和前 `{3*q}` 个 token 维度。
- QRC: `W=0.4`, `g=0.84`, `L=3`, Z + all-ZZ readout。
- CRC: 旧版 scalar-Q 储层；输入预处理、layers、输出维度与 QRC 对齐。
- CRC3-z/zz matched: Q×3 内部态直接读取 z 与全 zz，输出严格为 Q(Q+1)/2；不做 PCA。
- Standard ESN: 固定高斯回声态网络，谱半径0.9，直接输出QRC对齐维度的末态；不做PCA。
- raw-full baseline: 直接展平原始 tokens 后接单层线性读出。
- raw-PCA baseline: 每个 seed 仅用 train split 拟合 PCA，并压缩/零填充到 QRC/CRC 读出维度。
- classifier: 单层 `torch.nn.Linear(d, 1)`；不做标准化、正则、隐藏层或 class weighting。
- seeds: `0,10,...,90`；数据划分、QRC/CRC 参数、训练初始化均随 seed 变化。
- best epoch: validation `AUC > loss > ACC`，测试集只用于最终汇报。

## 运行

```bash
cd "{setting_dir.resolve()}"
/home/ubuntu/miniconda3/envs/Molecule/bin/python run_this_setting.py
```

或直接打开 `run_this_setting.ipynb`。

输出会写在当前目录下的 `features/`, `splits/`, `histories/`, `checkpoints/`,
`runs/`, `figures/` 以及 `performance_summary.csv` 等文件中。
"""


def _plan(dataset: str, q: int, t: int, scan_kind: str, scan_root: Path, setting_dir: Path) -> dict:
    dataset_root = PROJECT_ROOT / "general" / dataset
    return {
        "dataset": dataset,
        "token_variant": "general_task_independent",
        "scan_kind": scan_kind,
        "q": q,
        "t": t,
        "token_dim": 3 * q,
        "project_root": str(PROJECT_ROOT.resolve()),
        "scan_root": str(scan_root.resolve()),
        "setting_dir": str(setting_dir.resolve()),
        "data_dir": str((dataset_root / "data").resolve()),
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
        "source_layout_reference": SOURCE_LAYOUT_REFERENCE,
        "notes": [
            "This folder is a local runnable unit.",
            "Numerical kernels are imported from the project-level core package to keep all settings protocol-identical.",
        ],
    }


def _write_setting_code(dataset: str, q: int, t: int, scan_kind: str, scan_root: Path, setting_dir: Path) -> None:
    setting_dir.mkdir(parents=True, exist_ok=True)
    plan = _plan(dataset, q, t, scan_kind, scan_root, setting_dir)
    (setting_dir / "setting_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (setting_dir / "run_this_setting.py").write_text(_script(), encoding="utf-8")
    (setting_dir / "README.md").write_text(_readme(dataset, q, t, scan_kind, setting_dir), encoding="utf-8")
    notebook = _notebook(dataset, q, t, setting_dir)
    (setting_dir / "run_this_setting.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    written = 0
    for dataset in ("BACE", "BBBP"):
        dataset_root = PROJECT_ROOT / "general" / dataset
        qeq_root = dataset_root / "01_Q_equal_T_scan"
        fixed_root = dataset_root / "02_fixed_Q_T_scan"
        for value in range(1, 21):
            _write_setting_code(
                dataset=dataset,
                q=value,
                t=value,
                scan_kind="Q_equal_T",
                scan_root=qeq_root,
                setting_dir=qeq_root / f"Q{value:02d}_T{value:02d}",
            )
            written += 1
        for q in (8, 12, 16, 20):
            for t in range(1, 31):
                _write_setting_code(
                    dataset=dataset,
                    q=q,
                    t=t,
                    scan_kind="fixed_Q_T_scan",
                    scan_root=fixed_root,
                    setting_dir=fixed_root / f"Q{q:02d}" / f"T{t:02d}",
                )
                written += 1
    print(f"Generated local code packages for {written} minimum setting folders.")


if __name__ == "__main__":
    main()
