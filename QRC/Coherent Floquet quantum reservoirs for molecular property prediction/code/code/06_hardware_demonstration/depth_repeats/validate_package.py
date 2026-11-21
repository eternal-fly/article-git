#!/usr/bin/env python3
import hashlib
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED = "00b3fdbab64cb95f61b8580bd7e975ee9f1c9d7df1b44ce60ebe61a643a3eb04"
SPECS = {
    "01_sl_q6t6l1_r2": ("SL_Q6T6L1_r2_submit_collect.ipynb", "sl_q6t6l1_r2", 43002),
    "02_sl_q6t6l1_r3": ("SL_Q6T6L1_r3_submit_collect.ipynb", "sl_q6t6l1_r3", 43003),
}
orders = {}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for folder, (notebook_name, tag, order_seed) in SPECS.items():
    task = ROOT / folder
    datasets = list(task.glob("*.npz"))
    if len(datasets) != 1 or digest(datasets[0]) != EXPECTED:
        raise AssertionError((folder, datasets))
    with np.load(datasets[0], allow_pickle=False) as data:
        if data["tokens"].shape != (400, 6, 18):
            raise AssertionError(data["tokens"].shape)
        if np.bincount(data["labels"].astype(int), minlength=2).tolist() != [200, 200]:
            raise AssertionError("class balance")
        indices = data["indices"].astype(int)
    permutation = np.random.default_rng(order_seed).permutation(400)
    order = indices[permutation]
    if len(np.unique(order)) != 400 or np.array_equal(order, indices):
        raise AssertionError((folder, "invalid randomized order"))
    orders[folder] = order
    nb = json.loads((task / notebook_name).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    required = (
        'API_TOKEN = "PASTE_QUAFU_API_TOKEN_HERE"',
        'CALIBRATED_AT = "PASTE_CURRENT_CALIBRATION_TIMESTAMP_OR_ID_HERE"',
        f"SUBMISSION_ORDER_SEED = {order_seed}",
        f"DATASET_FILE = 'bace_bal400_canonical_v3_q6_t6__{tag}.npz'",
        "Q = 6", "T = 6", "LAYERS = 1", "G = 0.84", "W = 0.60",
        "RESERVOIR_SEED = 0", f'RESULT_TAG = "{tag}"',
        'metadata["nominal_two_qubit_gates"] != 60',
        'print("Topology validation passed; nominal CX count = 60")',
        "rng.permutation(len(BUNDLE[\"indices\"]))",
        "qc.rz(float(params[\"J\"][logical]), right)",
        "qc.rz(float(params[\"h\"][logical]) * math.pi, int(physical))",
        "from quark.quafu import Task",
        '"compiler": "quarkcircuit"', '"correct": False',
        "manager.run(task, repeat=REPEAT)",
    )
    missing = [value for value in required if value not in source]
    if missing:
        raise AssertionError((folder, missing))
    forbidden = (
        "LAYERS = 3", "G = 0.40", "fold_factor", "getpass(",
        "CALIBRATED_AT = '2026-07-28 19:25:54'",
    )
    present = [value for value in forbidden if value in source]
    if present:
        raise AssertionError((folder, present))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                raise AssertionError(f"Notebook is not clean: {notebook_name}")

if np.array_equal(orders["01_sl_q6t6l1_r2"], orders["02_sl_q6t6l1_r3"]):
    raise AssertionError("The two acquisition orders must differ")

master = ROOT / "data" / "bace_bal400_canonical_v3_q6_t6.npz"
if digest(master) != EXPECTED:
    raise AssertionError("Master data hash")

manifest_path = ROOT / "FILE_MANIFEST.csv"
with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
manifested = {row["relative_path"] for row in rows}
actual = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file() and path != manifest_path
}
if manifested != actual:
    raise AssertionError(("manifest coverage", sorted(manifested ^ actual)))
for row in rows:
    path = ROOT / row["relative_path"]
    if int(row["bytes"]) != path.stat().st_size or row["sha256"] != digest(path):
        raise AssertionError(("manifest mismatch", row["relative_path"]))
print("Validation passed: two clean 400-circuit Q6/T6/L1 repeats; 800 jobs total.")
