#!/usr/bin/env python3
"""Build one two-backend package using the supplied Quafu notebook conventions."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import build_complete_five_task_packages as base


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE / "quafu_strict_template_combined"
ARCHIVE = HERE / "Quafu_Baihua_Shenglian_complete_5tasks_strict_template.zip"
PROTOCOL_VERSION = (
    "2026-07-28-mainline-canonical-v3-q6t6-five-task-strict-template-v4"
)


def requirements_markdown() -> str:
    return """## Cloud-side requirements

The notebook follows the supplied Quafu template and uses only `numpy` and
`quarkstudio`/`quark`. The API credential is held in the environment variable
`QPU_API_TOKEN`. The first configuration cell prompts for it with `getpass`
when the variable is absent; the value is not saved into the notebook.

The cloud stage constructs QLISP circuits, calls `Task.run`, calls
`Task.result`, and saves raw result JSON files. Token generation, observable
reconstruction, model training, AUC calculation, and plotting remain local.
"""


def config_source(
    backend_key: str,
    task: dict,
    dataset_hash: str,
) -> str:
    backend = base.BACKENDS[backend_key]
    submission_tasks = base.submission_tasks_for(task)
    paired = len(submission_tasks) == 2
    dataset_name = base.task_dataset_name(task)
    expected_cx = (
        2 * 6 * int(task["layers"]) * 5 * int(task["fold_factor"])
    )
    stem = f"{backend_key}_{task['key']}"
    return f"""# ===============================
# 1) Configuration
# ===============================
CONFIG = {{
    "run": {{
        # Choose: "dry_build", "submit_qrc_jobs", "collect_qrc_results".
        "mode": "submit_qrc_jobs",
        "molecule_limit": None,
        "max_submit_per_run": None,
        "overwrite_jobs": False,
        "overwrite_results": False,
        "poll_sleep_sec": 0.0,
    }},
    "tokens": {{
        "max_events": 6,
        "token_dim": 18,
        "use_tanh_before_encoding": True,
    }},
    "qrc": {{
        "num_qubits": 6,
        "features_per_qubit": 3,
        "g": 0.84,
        "W_scale": 0.60,
        "layers_per_injection": {int(task["layers"])},
        "fold_factor": {int(task["fold_factor"])},
        "reservoir_seed": 0,
        "J_low": -0.9,
        "J_high": 0.9,
        "chip": {backend["chip"]!r},
        "repeat": 1,
        "mapping_mode": "physical_direct",
        "backend_num_qubits": {int(backend["backend_num_qubits"])},
        "physical_qubits": {backend["chain"]!r},
        "compile": True,
        "compiler": "quarkcircuit",
        "correct": False,
        "open_dd": None,
        "task_name_prefix": {stem!r},
        "full_bit_order": "rightmost_is_c0",
        "compact_bit_order": "rightmost_is_first_measured",
    }},
    "submission": {{
        "order_seed": {int(task["order_seed"])},
        "label_blind_random_order": True,
        "interleaved_l1_pair": {paired!r},
    }},
    "artifacts": {{
        "token_file": {dataset_name!r},
        "save_dir": {f"{stem}_cloud_results"!r},
        "job_manifest_path": {f"{stem}_cloud_results/{stem}_job_manifest.jsonl"!r},
        "raw_result_dir": {f"{stem}_cloud_results/raw_counts_{stem}"!r},
        "download_zip_name": {f"{stem}_cloud_results.zip"!r},
    }},
}}

# The supplied template reads QPU_API_TOKEN from the environment.
import os
from getpass import getpass
if not os.getenv("QPU_API_TOKEN", ""):
    os.environ["QPU_API_TOKEN"] = getpass("QPU_API_TOKEN: ")

PROTOCOL_VERSION = {PROTOCOL_VERSION!r}
BACKEND_KEY = {backend_key!r}
CHIP = CONFIG["qrc"]["chip"]
BACKEND_NUM_QUBITS = CONFIG["qrc"]["backend_num_qubits"]
PHYSICAL_QUBITS = CONFIG["qrc"]["physical_qubits"]
CALIBRATED_AT = {backend["calibrated_at"]!r}
CURRENT_TASK_KEY = {task["key"]!r}
CURRENT_TASK_FOLDER = {task["folder"]!r}
CURRENT_TASK = {task!r}
SUBMISSION_TASKS = {submission_tasks!r}
IS_INTERLEAVED_L1_PAIR = {paired!r}
Q = CONFIG["qrc"]["num_qubits"]
T = CONFIG["tokens"]["max_events"]
G = CONFIG["qrc"]["g"]
W = CONFIG["qrc"]["W_scale"]
RESERVOIR_SEED = CONFIG["qrc"]["reservoir_seed"]
REPEAT = CONFIG["qrc"]["repeat"]
DATASET_FILE = CONFIG["artifacts"]["token_file"]
EXPECTED_DATASET_SHA256 = {dataset_hash!r}
EXPECTED_CURRENT_CX = {expected_cx}
"""


def transform_payload(
    backend_key: str,
    task: dict,
    dataset_hash: str,
) -> dict:
    payload = base.notebook_payload(backend_key, task, dataset_hash)
    payload["cells"].insert(1, base.markdown_cell(requirements_markdown()))
    payload["cells"][2] = base.code_cell(
        config_source(backend_key, task, dataset_hash)
    )

    imports = "".join(payload["cells"][3]["source"])
    imports = imports.replace(
        base.PROTOCOL_VERSION,
        PROTOCOL_VERSION,
    )
    imports = imports.replace("import shutil\n", "import shutil\nimport time\n")
    payload["cells"][3] = base.code_cell(imports)

    circuit = "".join(payload["cells"][5]["source"])
    old_preview = """preview_circuit, preview_metadata = build_circuit(BUNDLE["tokens"][0])
preview_cx = sum(
    isinstance(value, (tuple, list)) and value and str(value[0]).lower() == "cx"
    for value in preview_circuit.to_qlisp
)
if preview_cx != EXPECTED_CURRENT_CX:
    raise AssertionError((preview_cx, EXPECTED_CURRENT_CX))
print(
    f"Topology/circuit validation passed; current nominal CX count = "
    f"{EXPECTED_CURRENT_CX}"
)
"""
    new_preview = """preview_circuit, preview_metadata = build_circuit(BUNDLE["tokens"][0])
if int(preview_metadata["nominal_two_qubit_gates"]) != int(EXPECTED_CURRENT_CX):
    raise AssertionError(
        (
            preview_metadata["nominal_two_qubit_gates"],
            EXPECTED_CURRENT_CX,
        )
    )
print(
    f"Topology/circuit validation passed; current nominal CX count = "
    f"{EXPECTED_CURRENT_CX}"
)
print("QLISP preview (format is backend-version dependent):")
preview_qlisp = preview_circuit.to_qlisp
if isinstance(preview_qlisp, (list, tuple)):
    for item in list(preview_qlisp)[:20]:
        print(item)
    print(f"Total QLISP items: {len(preview_qlisp)}")
else:
    print(str(preview_qlisp)[:2000])
    print(f"QLISP object type: {type(preview_qlisp)}")
"""
    if old_preview not in circuit:
        raise AssertionError("The generated preview block changed unexpectedly")
    circuit = circuit.replace(old_preview, new_preview)
    payload["cells"][5] = base.code_cell(circuit)

    acquisition = "".join(payload["cells"][6]["source"])
    old_manager = """def task_manager():
    if not API_TOKEN:
        raise RuntimeError("API token is empty")
    from quark import Task
    return Task(API_TOKEN)
"""
    new_manager = """def init_quafu_task_manager():
    token = os.getenv("QPU_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "QPU_API_TOKEN is empty. Enter it in the configuration cell."
        )
    from quark import Task
    return Task(token)


def task_manager():
    # Compatibility alias; submission/collection use the supplied template API.
    return init_quafu_task_manager()
"""
    if old_manager not in acquisition:
        raise AssertionError("The generated task-manager block changed unexpectedly")
    acquisition = acquisition.replace(old_manager, new_manager)
    acquisition = acquisition.replace(
        "    manager = task_manager()\n",
        "    tmgr = init_quafu_task_manager()\n",
    )
    acquisition = acquisition.replace(
        "        tid = manager.run(task_payload, repeat=REPEAT)\n",
        "        tid = tmgr.run(task_payload, repeat=REPEAT)\n",
    )
    acquisition = acquisition.replace(
        "                result = manager.result(record[\"tid\"])\n",
        "                result = tmgr.result(record[\"tid\"])\n",
    )

    acquisition = acquisition.replace(
        """    rows = rng.permutation(len(BUNDLE["indices"])).astype(int)
    specs = []
""",
        """    rows = rng.permutation(len(BUNDLE["indices"])).astype(int)
    molecule_limit = CONFIG.get("run", {}).get("molecule_limit", None)
    if molecule_limit is not None:
        rows = rows[: int(molecule_limit)]
    specs = []
""",
    )
    acquisition = acquisition.replace(
        """    submitted = 0
    for spec in specs:
""",
        """    submitted = 0
    max_submit = CONFIG.get("run", {}).get("max_submit_per_run", None)
    max_submit = None if max_submit is None else int(max_submit)
    overwrite = bool(CONFIG.get("run", {}).get("overwrite_jobs", False))
    poll_sleep = float(CONFIG.get("run", {}).get("poll_sleep_sec", 0.0))
    for spec in specs:
        if max_submit is not None and submitted >= max_submit:
            break
""",
    )
    acquisition = acquisition.replace(
        """        if old is not None and old.get("tid") is not None:
            continue
""",
        """        if old is not None and old.get("tid") is not None and not overwrite:
            continue
""",
    )
    acquisition = acquisition.replace(
        """        print(
            f"[{submitted:04d}] global_rank={spec['global_submission_rank']:04d} "
            f"task={task_key} id={spec['original_index']} tid={tid}"
        )
""",
        """        print(
            f"[{submitted:04d}] global_rank={spec['global_submission_rank']:04d} "
            f"task={task_key} id={spec['original_index']} tid={tid}"
        )
        if poll_sleep > 0:
            time.sleep(poll_sleep)
""",
    )
    acquisition += """


# Supplied-template stage names.
def submit_qrc_jobs(config=CONFIG):
    return submit_all_randomized()


def collect_qrc_results(config=CONFIG):
    return collect_raw_results()
"""
    payload["cells"][6] = base.code_cell(acquisition)

    execute = """# ===============================
# 7) Execute configured cloud stage
# ===============================
def execute_configured_stage(config: dict):
    mode = config.get("run", {}).get("mode", "dry_build")
    if mode == "dry_build":
        return preview_qlisp
    if mode == "submit_qrc_jobs":
        return submit_qrc_jobs(config)
    if mode == "collect_qrc_results":
        return collect_qrc_results(config)
    raise ValueError(f"Unknown mode: {mode}")


stage_output = execute_configured_stage(CONFIG)
stage_output
"""
    payload["cells"][7] = base.code_cell(execute)
    payload["cells"][0] = base.markdown_cell(
        f"""# Quafu cloud DTC-QRC acquisition — {base.BACKENDS[backend_key]['chip']} / {task['key']}

This notebook follows the structure and Quafu API call pattern of the supplied
`02_qrc_quafu_Baihua12_T12_modified.ipynb`, with only the manuscript protocol,
NPZ path, physical chain, task naming, and randomized ordering updated.

It uses Q=T=6, g=0.84, W=0.60, reservoir seed 0, and the task-specific L/fold
configuration. The cloud saves job manifests and raw task results only.
"""
    )
    return payload


def build() -> Path:
    for source in (base.SOURCE_DATASET, base.SOURCE_IDEAL_L1, base.SOURCE_IDEAL_L3):
        if not source.is_file():
            raise FileNotFoundError(source)
    dataset_hash = base.sha256_file(base.SOURCE_DATASET)
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_backends = {}
    for backend_key, backend in base.BACKENDS.items():
        backend_root = PACKAGE_ROOT / backend_key
        backend_root.mkdir(parents=True, exist_ok=True)
        task_rows = []
        for task in base.TASKS:
            task_dir = backend_root / task["folder"]
            task_dir.mkdir(parents=True, exist_ok=True)
            dataset_name = base.task_dataset_name(task)
            dataset_path = task_dir / dataset_name
            shutil.copy2(base.SOURCE_DATASET, dataset_path)
            notebook_name = (
                f"{backend['chip']}_{task['key']}_strict_template.ipynb"
            )
            notebook_path = task_dir / notebook_name
            base.write_notebook(
                notebook_path,
                transform_payload(backend_key, task, dataset_hash),
            )
            task_rows.append(
                {
                    "task_key": task["key"],
                    "folder": task["folder"],
                    "circuits": 400,
                    "layers": task["layers"],
                    "fold_factor": task["fold_factor"],
                    "nominal_two_qubit_gates": (
                        2
                        * 6
                        * int(task["layers"])
                        * 5
                        * int(task["fold_factor"])
                    ),
                    "order_seed": task["order_seed"],
                    "dataset": dataset_name,
                    "dataset_sha256": base.sha256_file(dataset_path),
                    "notebook": notebook_name,
                    "notebook_sha256": base.sha256_file(notebook_path),
                }
            )
        manifest_backends[backend_key] = {
            "chip": backend["chip"],
            "physical_qubits": backend["chain"],
            "task_folder_count": 5,
            "npz_count": 5,
            "notebook_count": 5,
            "circuits": 2000,
            "tasks": task_rows,
        }

    readme = """# Quafu complete two-backend strict-template package

This archive contains `baihua/` and `shenglian/`. Each backend contains five
task folders, five processed NPZ files, five notebooks, and 2,000 circuits.

Run every notebook from inside its own task folder. In the configuration cell:

- `CONFIG["run"]["mode"] = "dry_build"` previews one circuit;
- `CONFIG["run"]["mode"] = "submit_qrc_jobs"` submits randomized jobs;
- `CONFIG["run"]["mode"] = "collect_qrc_results"` downloads raw results.

The configuration cell requests `QPU_API_TOKEN` with a hidden prompt when the
environment variable is absent. No token is stored in the archive.

L3 folders 01–03 each submit 400 circuits. For the coordinated L1 pair, run
submit in either folder 04 or 05 once; it submits 800 base/fold3 circuits in
per-molecule interleaved order. Then collect separately in folders 04 and 05.
"""
    (PACKAGE_ROOT / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_template": "02_qrc_quafu_Baihua12_T12_modified.ipynb",
        "credential_included": False,
        "cloud_contract": "Task.run/Task.result and raw-result storage only",
        "backend_folder_count": 2,
        "circuits_per_backend": 2000,
        "total_circuits": 4000,
        "backends": manifest_backends,
    }
    (PACKAGE_ROOT / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(
        ARCHIVE,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as handle:
        for path in sorted(PACKAGE_ROOT.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(PACKAGE_ROOT).as_posix())
    output = {
        "archive": str(ARCHIVE),
        "archive_sha256": base.sha256_file(ARCHIVE),
        "manifest": manifest,
    }
    (HERE / "STRICT_TEMPLATE_COMBINED_MANIFEST.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))
    return ARCHIVE


if __name__ == "__main__":
    build()
