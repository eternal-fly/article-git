"""Generate BACE/general W-scan local code packages.

Folder layout:

general/BACE/03_W_scan/
  Q12_T12/W0p0 ... W0p9
  Q16_T16/W0p0 ... W0p9

Each W folder is a complete minimum experiment unit with its own plan,
script, notebook and README.  Numerical work is delegated to
``core.linear_readout.run_setting_plan`` so the protocol remains identical to
the other FINAL/2 linear-readout scans.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACE_ROOT = PROJECT_ROOT / "general" / "BACE"
SCAN_ROOT = BACE_ROOT / "03_W_scan"
SEEDS = list(range(0, 100, 10))
SETTINGS = ((12, 12), (16, 16))
W_VALUES = tuple(round(index / 10, 1) for index in range(10))
SOURCE_LAYOUT_REFERENCE = "/home/ubuntu/Workspace/Fei/wlf/Article/QRC/GPU-simulate/BACE"


def w_label(value: float) -> str:
    return f"W{value:.1f}".replace(".", "p")


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _script() -> str:
    return '''"""Run this exact BACE/general W-scan setting."""

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


def _notebook(q: int, t: int, W: float, setting_dir: Path) -> dict:
    intro = f"""# BACE general BFS W scan — Q={q}, T={t}, W={W:.1f}

This notebook is the local executable code for one minimum W-scan folder.
Only `W` differs across sibling folders; tokens, splits, `g`, `L`, QRC/CRC
readout, raw baselines, and the single linear readout are protocol-identical.
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
    protocol = f"""## Protocol locked in this folder

- Dataset/token stream: BACE, general task-independent local-BFS events.
- Token slice: first `{t}` events and first `{3 * q}` token coordinates.
- QRC/CRC scale: `W={W:.1f}`, `g=0.84`, `L=3`.
- QRC readout: Z + all-ZZ; CRC3 matched readout: direct z + all zz from a Q×3 state (no PCA); CRC readout: scalar-Q state + all pair products.
- Baselines: raw-full flattened tokens and train-split raw-PCA.
- Classifier: exactly one `torch.nn.Linear(d, 1)`, no standardization, no hidden layer, no regularization.
- Seeds/splits: `0,10,...,90`; QRC/CRC parameters, scaffold split and training all change with seed.
- Best epoch: validation `AUC > loss > ACC`.
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


def _readme(q: int, t: int, W: float, setting_dir: Path) -> str:
    return f"""# BACE general BFS W scan — Q={q}, T={t}, W={W:.1f}

这个目录是 W 扫描中的一个最小实验单元。组织方式参考
`{SOURCE_LAYOUT_REFERENCE}` 中每个实验文件夹自带代码的思路；数值实现
复用 FINAL/2 的共享核心，确保与 Q/T 扫描完全同协议。

## 本目录代码

- `setting_plan.json`：固定 dataset、Q、T、W、g、L、seeds、data dir、scan root 等。
- `run_this_setting.ipynb`：推荐入口；打开后从上到下运行，完成训练和本设置可视化。
- `run_this_setting.py`：脚本入口；等价运行本设置。

## 协议

- scan kind: `W_scan`
- tokens: BACE general task-independent local-BFS event stream。
- token slice: first `{t}` events and first `{3 * q}` token dimensions。
- QRC/CRC: `W={W:.1f}`, `g=0.84`, `L=3`。
- QRC readout: Z + all-ZZ。
- CRC: old scalar-Q reservoir；输入预处理、layers、输出维度与 QRC 对齐。
- raw-full baseline: 直接展平原始 tokens 后接单层线性读出。
- raw-PCA baseline: 每个 seed 仅用 train split 拟合 PCA，并压缩/零填充到 QRC/CRC 读出维度。
- classifier: 单层 `torch.nn.Linear(d, 1)`；不做标准化、正则、隐藏层或 class weighting。
- seeds: `0,10,...,90`。
- best epoch: validation `AUC > loss > ACC`。

## 运行

```bash
cd "{setting_dir.resolve()}"
/home/ubuntu/miniconda3/envs/Molecule/bin/python run_this_setting.py
```

输出会写在当前目录下的 `features/`, `splits/`, `histories/`,
`checkpoints/`, `runs/`, `figures/` 以及 `performance_summary.csv` 等文件中。
"""


def _plan(q: int, t: int, W: float, setting_dir: Path) -> dict:
    return {
        "dataset": "BACE",
        "token_variant": "general_task_independent",
        "scan_kind": "W_scan",
        "q": q,
        "t": t,
        "token_dim": 3 * q,
        "project_root": str(PROJECT_ROOT.resolve()),
        "scan_root": str(SCAN_ROOT.resolve()),
        "setting_dir": str(setting_dir.resolve()),
        "data_dir": str((BACE_ROOT / "data").resolve()),
        "epochs": 500,
        "seeds": SEEDS,
        "W": float(W),
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
            "This folder is a local runnable W-scan unit.",
            "Only W changes across sibling folders with the same Q/T setting.",
            "Numerical kernels are imported from the project-level core package to keep all settings protocol-identical.",
        ],
    }


def _write_setting(q: int, t: int, W: float) -> Path:
    setting_dir = SCAN_ROOT / f"Q{q:02d}_T{t:02d}" / w_label(W)
    setting_dir.mkdir(parents=True, exist_ok=True)
    (setting_dir / "setting_plan.json").write_text(
        json.dumps(_plan(q, t, W, setting_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (setting_dir / "run_this_setting.py").write_text(_script(), encoding="utf-8")
    (setting_dir / "README.md").write_text(_readme(q, t, W, setting_dir), encoding="utf-8")
    (setting_dir / "run_this_setting.ipynb").write_text(
        json.dumps(_notebook(q, t, W, setting_dir), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return setting_dir


def _scan_notebook() -> dict:
    intro = """# BACE general BFS — W scan

This scan fixes `(Q,T)=(12,12)` and `(16,16)` and sweeps `W=0.0,...,0.9`.
Each minimum folder owns its own runnable code; this notebook is the scan-level
convenience entrypoint and visualization wrapper.
"""
    imports = f"""from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(r"{PROJECT_ROOT.resolve()}").resolve()
SCAN_ROOT = Path(r"{SCAN_ROOT.resolve()}").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.linear_readout import collect_w_scan_results, plot_w_scan

print("project:", PROJECT_ROOT)
print("scan:", SCAN_ROOT)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
"""
    list_settings = """settings = sorted(SCAN_ROOT.glob("Q*_T*/W*"))
print("n settings:", len(settings))
for path in settings:
    print(path.relative_to(SCAN_ROOT))
"""
    visualize = """table = collect_w_scan_results(SCAN_ROOT)
display(table.head())
if len(table):
    fig = plot_w_scan(SCAN_ROOT, "BACE")
    plt.show()
    display(table.groupby(["q", "t", "method"])[["test_auc_mean", "test_acc_mean"]].max())
else:
    print("No completed W-scan settings yet. Run individual W folders or run_w_scan_sequential.py first.")
"""
    return {
        "cells": [
            _cell("markdown", intro),
            _cell("code", imports),
            _cell("code", list_settings),
            _cell("code", visualize),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Molecule", "language": "python", "name": "molecule"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _runner() -> str:
    return f'''"""Run BACE/general W-scan folders sequentially."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

SCAN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCAN_ROOT
while not (PROJECT_ROOT / "core" / "linear_readout.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Could not locate FINAL/2_linear_readout project root")
    PROJECT_ROOT = PROJECT_ROOT.parent
PYTHON = Path(sys.executable).resolve()
LOG_ROOT = PROJECT_ROOT / "logs"
SETTINGS = {SETTINGS!r}
W_VALUES = {W_VALUES!r}
TOTAL_JOBS = len(SETTINGS) * len(W_VALUES)


@dataclass(frozen=True)
class Job:
    order: int
    q: int
    t: int
    W: float
    setting_dir: Path


def w_label(value: float) -> str:
    return f"W{{value:.1f}}".replace(".", "p")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def iter_jobs() -> list[Job]:
    jobs = []
    order = 0
    for q, t in SETTINGS:
        for W in W_VALUES:
            order += 1
            jobs.append(Job(order, q, t, W, SCAN_ROOT / f"Q{{q:02d}}_T{{t:02d}}" / w_label(W)))
    return jobs


def write_status_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "q", "t", "W", "setting_dir", "status", "start_time", "end_time", "duration_sec", "returncode", "log_path"])


def append_status(path: Path, job: Job, status: str, start: float, end: float, returncode: int, log_path: Path) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            job.order, job.q, job.t, f"{{job.W:.1f}}", str(job.setting_dir), status,
            datetime.fromtimestamp(start).isoformat(timespec="seconds"),
            datetime.fromtimestamp(end).isoformat(timespec="seconds"),
            f"{{end - start:.1f}}", returncode, str(log_path),
        ])


def validate_jobs(jobs: list[Job]) -> None:
    missing = [job.setting_dir / "run_this_setting.py" for job in jobs if not (job.setting_dir / "run_this_setting.py").exists()]
    if missing:
        preview = "\\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing run_this_setting.py in {{len(missing)}} folders; first entries:\\n{{preview}}")


def run_job(job: Job, run_dir: Path, status_path: Path) -> None:
    log_path = run_dir / f"{{job.order:03d}}_BACE_W_scan_Q{{job.q:02d}}_T{{job.t:02d}}_{{w_label(job.W)}}.log"
    start = time.time()
    print(f"[{{now()}}] START {{job.order:03d}}/{{TOTAL_JOBS}} BACE W_scan Q={{job.q}} T={{job.t}} W={{job.W:.1f}}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {{now()}} START BACE W_scan Q={{job.q}} T={{job.t}} W={{job.W:.1f}}\\n")
        log.write(f"# setting_dir = {{job.setting_dir}}\\n")
        log.flush()
        completed = subprocess.run(
            [str(PYTHON), "run_this_setting.py"],
            cwd=str(job.setting_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        log.write(f"\\n# {{now()}} END returncode={{completed.returncode}}\\n")
    end = time.time()
    if completed.returncode == 0:
        append_status(status_path, job, "done", start, end, completed.returncode, log_path)
        print(f"[{{now()}}] DONE  {{job.order:03d}}/{{TOTAL_JOBS}} BACE W_scan Q={{job.q}} T={{job.t}} W={{job.W:.1f}} ({{end - start:.1f}}s)", flush=True)
        return
    append_status(status_path, job, "failed", start, end, completed.returncode, log_path)
    print(f"[{{now()}}] FAIL  {{job.order:03d}}/{{TOTAL_JOBS}} BACE W_scan Q={{job.q}} T={{job.t}} W={{job.W:.1f}}; log={{log_path}}", flush=True)
    raise SystemExit(completed.returncode)


def aggregate_figures() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.linear_readout import plot_w_scan

    print(f"[{{now()}}] Aggregating BACE W-scan figure", flush=True)
    plot_w_scan(SCAN_ROOT, "BACE")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_ROOT / f"bace_general_w_scan_{{timestamp}}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = LOG_ROOT / "latest_bace_general_w_scan_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir) + "\\n", encoding="utf-8")
    status_path = run_dir / "status.csv"
    write_status_header(status_path)

    jobs = iter_jobs()
    validate_jobs(jobs)
    print(f"[{{now()}}] W-scan run directory: {{run_dir}}", flush=True)
    print(f"[{{now()}}] Python executable: {{PYTHON}}", flush=True)
    print(f"[{{now()}}] Total settings: {{len(jobs)}}", flush=True)

    for job in jobs:
        run_job(job, run_dir, status_path)

    aggregate_figures()
    (run_dir / "ALL_DONE.txt").write_text(f"Completed all settings at {{now()}}\\n", encoding="utf-8")
    print(f"[{{now()}}] ALL DONE. Logs: {{run_dir}}", flush=True)


if __name__ == "__main__":
    main()
'''


def _scan_readme() -> str:
    return f"""# BACE general BFS W scan

固定两个尺度：

- `Q12_T12`
- `Q16_T16`

每个尺度下扫描 `W=0.0,0.1,...,0.9`，每个 W 独立成一个最小实验文件夹。

## 文件夹结构

```text
03_W_scan/
  Q12_T12/
    W0p0/
    ...
    W0p9/
  Q16_T16/
    W0p0/
    ...
    W0p9/
```

## 运行

单个 setting：

```bash
cd "{(SCAN_ROOT / 'Q16_T16' / 'W0p4').resolve()}"
/home/ubuntu/miniconda3/envs/Molecule/bin/python run_this_setting.py
```

整个 W scan 顺序运行：

```bash
cd "{SCAN_ROOT.resolve()}"
/home/ubuntu/miniconda3/envs/Molecule/bin/python run_w_scan_sequential.py
```

完成后会生成：

- `scan_W_performance_summary.csv`
- `figures/BACE_general_W_scan.(png/pdf/svg/tiff)`

协议与 FINAL/2 其它 general linear-readout 扫描完全一致，只改变 `W`。
"""


def main() -> None:
    written = 0
    for q, t in SETTINGS:
        for W in W_VALUES:
            _write_setting(q, t, W)
            written += 1
    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    (SCAN_ROOT / "BACE_general_W_scan.ipynb").write_text(
        json.dumps(_scan_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (SCAN_ROOT / "run_w_scan_sequential.py").write_text(_runner(), encoding="utf-8")
    (SCAN_ROOT / "README.md").write_text(_scan_readme(), encoding="utf-8")
    print(f"Generated BACE W-scan code for {written} minimum setting folders under {SCAN_ROOT}.")


if __name__ == "__main__":
    main()
