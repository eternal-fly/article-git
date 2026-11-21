"""Build the user-facing Molecule notebook with CONFIG as its first cell."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "01_spatrace_dtc_qrc_algorithm_validation.ipynb"


cells = [
    nbf.v4.new_code_cell(
        """CONFIG = {
    "data": {
        "source_dir": "data/source",
        "source_revision": "SpaTRACE a553bed733a0c9743e60081951a4cfe174503942",
        "fixed_dataset": "data/fixed_spatrace_3to1_cell_disjoint.npz",
        "force_rebuild": False,
        "required_input_dim": 60,
        "gcn_k": 5,
        "gcn_alpha": 0.30,
        "lr_radius": 5.0,
        "lr_min_corr": 0.30,
        "split_seed": 2026,
        "split_fractions": [0.70, 0.15, 0.15],
        "pseudotime_bins": 5,
        "path_length": 3,
        "path_repeats": {"train": 2, "validation": 1, "test": 1},
        "path_seed": 42,
        "k_primary": 5,
        "fallback_scan": 10,
        "out_degree": 2,
    },
    "qrc": {
        "num_qubits": 20,
        "features_per_qubit": 3,
        "g": 0.40,
        "W": 0.60,
        "L": 3,
        "angle_scale": 1.5707963267948966,
        "complex_dtype": "complex64",
        "reservoir_seeds": [1000, 1010, 1020, 1030, 1040],
    },
    "training": {
        "training_seeds": [2000, 2001, 2002, 2003, 2004, 2005],
        "device": "cuda:0",
        "epochs": 400,
        "batch_size": 64,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "patience": 60,
        "minimum_delta": 1e-6,
    },
    "esn": {
        "hidden_size": 210,
        "spectral_radius": 0.90,
        "input_scale": 0.50,
        "leak_rate": 1.0,
    },
    "runtime": {
        "devices": ["cuda:0"],
        "qrc_batch_size": 256,
        "force_recompute_features": False,
        "run_injection_only_control": True,
        "cache_dir": "cache/g0p4",
        "output_dir": "outputs/g0p4",
    },
}
"""
    ),
    nbf.v4.new_markdown_cell(
        """# SpaTRACE × DTC-QRC：第一层算法验证

固定任务是：沿严格递增伪时序路径读取前三个细胞事件的完整 60 维状态（6 TF + 8 LR + 46 TG），预测第 4 个事件的 46 维 TG 表达。数据划分在细胞层面互斥并固定落盘。

输入不再使用 `tanh + U3`。每个量子比特接收三个特征，并执行一次
$U_{in}=\exp[-\frac{i}{2}(\theta_xX+\theta_yY+\theta_zZ)]$；随后仍是原 BFS-QRC 的 $L=3$ 层 DTC Floquet 演化。"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image

from data_pipeline import prepare_fixed_dataset
from experiment import (
    extract_injection_only_features,
    extract_qrc_features,
    load_dataset_with_linear_angles,
    plot_summary,
    run_injection_audit,
    run_nested_analysis,
)

ROOT = Path.cwd().resolve()
DATA_CONFIG = dict(CONFIG["data"])
DATA_CONFIG["source_dir"] = str((ROOT / DATA_CONFIG["source_dir"]).resolve())
DATASET_PATH = (ROOT / CONFIG["data"]["fixed_dataset"]).resolve()
CACHE_DIR = (ROOT / CONFIG["runtime"]["cache_dir"]).resolve()
OUTPUT_DIR = (ROOT / CONFIG["runtime"]["output_dir"]).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Python:", sys.executable)
print("Version:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert "Molecule" in sys.executable, "This notebook must run with the Molecule environment."
assert CONFIG["qrc"]["num_qubits"] * CONFIG["qrc"]["features_per_qubit"] == 60
"""
    ),
    nbf.v4.new_code_cell(
        """fixed_path = prepare_fixed_dataset(
    DATA_CONFIG,
    DATASET_PATH,
    force=CONFIG["data"]["force_rebuild"],
)
metadata = json.loads(fixed_path.with_suffix(".json").read_text(encoding="utf-8"))
print(json.dumps(metadata["dimensions"], ensure_ascii=False, indent=2))
print("path counts:", metadata["path_counts"])
print("cell counts:", metadata["cell_counts"])
print("kept LR pairs:", metadata["dimensions"]["lr_pair"])
assert metadata["dimensions"] == {"tf": 6, "lr_pair": 8, "target_gene": 46, "input": 60, "output": 46}
"""
    ),
    nbf.v4.new_code_cell(
        """data, normalization_audit = load_dataset_with_linear_angles(
    DATASET_PATH,
    angle_scale=CONFIG["qrc"]["angle_scale"],
    output_dir=OUTPUT_DIR,
)
hamiltonian_audit = run_injection_audit(data, OUTPUT_DIR)
print("linear normalization audit:")
display(pd.Series(normalization_audit))
print("Hamiltonian injection audit:")
display(pd.Series(hamiltonian_audit))
assert normalization_audit["affine_identity_max_error"] < 1e-5
assert hamiltonian_audit["max_unitarity_error"] < 1e-10
"""
    ),
    nbf.v4.new_code_cell(
        """qrc_feature_files = extract_qrc_features(
    data,
    dataset_path=DATASET_PATH,
    qrc_config=CONFIG["qrc"],
    reservoir_seeds=CONFIG["qrc"]["reservoir_seeds"],
    devices=CONFIG["runtime"]["devices"],
    batch_size=CONFIG["runtime"]["qrc_batch_size"],
    cache_dir=CACHE_DIR,
    force=CONFIG["runtime"]["force_recompute_features"],
)
print("QRC caches:")
for path in qrc_feature_files:
    print(" -", path.name)
"""
    ),
    nbf.v4.new_code_cell(
        """injection_only_file = None
if CONFIG["runtime"]["run_injection_only_control"]:
    injection_only_file = extract_injection_only_features(
        data,
        qrc_config=CONFIG["qrc"],
        device=CONFIG["runtime"]["devices"][0],
        batch_size=CONFIG["runtime"]["qrc_batch_size"],
        cache_dir=CACHE_DIR,
        force=CONFIG["runtime"]["force_recompute_features"],
    )
    print("Injection-only cache:", injection_only_file.name)
"""
    ),
    nbf.v4.new_code_cell(
        """nested_metrics, summary, paired_summary = run_nested_analysis(
    data,
    qrc_feature_files=qrc_feature_files,
    injection_only_file=injection_only_file,
    reservoir_seeds=CONFIG["qrc"]["reservoir_seeds"],
    training_seeds=CONFIG["training"]["training_seeds"],
    readout_config=CONFIG["training"],
    esn_config=CONFIG["esn"],
    angle_scale=CONFIG["qrc"]["angle_scale"],
    output_dir=OUTPUT_DIR,
)
display(summary[["method", "n_runs", "n_reservoirs", "macro_nrmse_mean", "macro_nrmse_sd_between_reservoirs", "macro_r2_mean", "macro_pearson_mean"]])
display(paired_summary)
"""
    ),
    nbf.v4.new_code_cell(
        """figure_path = plot_summary(summary, OUTPUT_DIR / "macro_nrmse_comparison.png")
run_manifest = {
    "config": CONFIG,
    "dataset": str(DATASET_PATH),
    "qrc_feature_files": [str(path) for path in qrc_feature_files],
    "injection_only_file": str(injection_only_file) if injection_only_file else None,
    "normalization_audit": normalization_audit,
    "hamiltonian_audit": hamiltonian_audit,
}
(OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
display(Image(filename=str(figure_path)))
print("Completed. Results:", OUTPUT_DIR)
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python (Molecule)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
