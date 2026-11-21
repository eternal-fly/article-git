"""Build the four Shenglian Q6/T6/L3 hardware-g-scan notebooks.

The source is the previously uploaded Shenglian cloud-acquisition notebook.
Only the scan identity, g value, output names, and a robust QuarkStudio Task
import are changed.  Dataset, physical chain, circuit depth, W, reservoir
seed, and label-blind randomized molecule order remain fixed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE.parent
    / "Quafu_mainline_Q6T6L3_experiment"
    / "upload_shenglian"
    / "Shenglian_Q6T6L3_random_submit_collect_raw.ipynb"
)

SCAN_POINTS = (
    ("040", 0.40),
    ("060", 0.60),
    ("084", 0.84),
    ("100", 1.00),
)


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


def build_one(template: dict, tag: str, g_value: float) -> Path:
    notebook = copy.deepcopy(template)
    result_tag = f"sl_g{tag}"

    set_source(
        notebook["cells"][0],
        f"""# Shenglian Q6/T6/L3 hardware scan: g={g_value:.2f}

This notebook is one of four matched hardware acquisitions using the same
processed 400-molecule canonical-v3 NPZ, Shenglian qubit chain, W, reservoir
seed, circuit depth, and deterministic label-blind randomized submission
order. Only the global rotation parameter is changed.

`ACTION="submit"` submits 400 circuits. `ACTION="collect"` downloads only
this g point's raw counts and creates `{result_tag}_results.zip` for local
feature reconstruction and training.
""",
    )

    config = source_text(notebook["cells"][1])
    config = replace_once(
        config,
        "# Edit only ACTION, then run all cells.",
        "# Edit only ACTION, then run all cells. API token is entered securely below.",
    )
    config = replace_once(
        config,
        "G = 0.84",
        f"G = {g_value:.2f}",
    )
    config = replace_once(
        config,
        'RESULT_ROOT_NAME = f"{BACKEND_KEY}_q6t6l3_cloud_results"',
        f'RESULT_TAG = "{result_tag}"\nRESULT_ROOT_NAME = f"{{RESULT_TAG}}_results"',
    )
    config = replace_once(
        config,
        'MANIFEST_NAME = f"{BACKEND_KEY}_q6t6l3_job_manifest.jsonl"',
        'MANIFEST_NAME = f"{RESULT_TAG}_jobs.jsonl"',
    )
    config = replace_once(
        config,
        'RAW_DIR_NAME = f"raw_counts_{BACKEND_KEY}_q6t6l3"',
        'RAW_DIR_NAME = f"raw_{RESULT_TAG}"',
    )
    config = replace_once(
        config,
        'DOWNLOAD_ZIP_NAME = f"{BACKEND_KEY}_q6t6l3_cloud_results.zip"',
        'DOWNLOAD_ZIP_NAME = f"{RESULT_TAG}_results.zip"',
    )
    set_source(notebook["cells"][1], config)

    paths = source_text(notebook["cells"][2])
    paths = replace_once(
        paths,
        'PROTOCOL_VERSION = "2026-07-28-mainline-canonical-v3-q6t6l3-cloud-acquisition-v2"',
        'PROTOCOL_VERSION = "2026-07-31-shenglian-canonical-v3-q6t6l3-gscan-v1"',
    )
    paths = replace_once(
        paths,
        'ORDER_PATH = RESULT_ROOT / f"{BACKEND_KEY}_q6t6l3_submission_order.json"',
        'ORDER_PATH = RESULT_ROOT / f"{RESULT_TAG}_order.json"',
    )
    paths = replace_once(
        paths,
        'SUMMARY_PATH = RESULT_ROOT / f"{BACKEND_KEY}_q6t6l3_collection_summary.json"',
        'SUMMARY_PATH = RESULT_ROOT / f"{RESULT_TAG}_summary.json"',
    )
    set_source(notebook["cells"][2], paths)

    cloud = source_text(notebook["cells"][5])
    cloud = replace_once(
        cloud,
        "    from quark import Task\n    return Task(API_TOKEN)",
        "    try:\n"
        "        from quark.quafu import Task\n"
        "    except ImportError:\n"
        "        from quark import Task\n"
        "    return Task(API_TOKEN)",
    )
    cloud = replace_once(
        cloud,
        'f"{BACKEND_KEY}_q6t6l3_rank{spec[\'submission_rank\']:05d}"',
        'f"{RESULT_TAG}_rank{spec[\'submission_rank\']:05d}"',
    )
    cloud = replace_once(
        cloud,
        '                "backend": CHIP,\n                "seed": SUBMISSION_ORDER_SEED,',
        '                "backend": CHIP,\n'
        '                "result_tag": RESULT_TAG,\n'
        '                "g": G,\n'
        '                "seed": SUBMISSION_ORDER_SEED,',
    )
    cloud = replace_once(
        cloud,
        '            "backend_key": BACKEND_KEY,\n            "chip": CHIP,',
        '            "backend_key": BACKEND_KEY,\n'
        '            "result_tag": RESULT_TAG,\n'
        '            "chip": CHIP,',
    )
    set_source(notebook["cells"][5], cloud)

    for index, cell in enumerate(notebook["cells"]):
        cell.setdefault("id", f"gscan-cell-{index:02d}")
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    output = HERE / f"SL_g{tag}.ipynb"
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    template = json.loads(SOURCE.read_text(encoding="utf-8"))
    outputs = [build_one(template, tag, value) for tag, value in SCAN_POINTS]
    for output in outputs:
        print(output.name)


if __name__ == "__main__":
    main()
