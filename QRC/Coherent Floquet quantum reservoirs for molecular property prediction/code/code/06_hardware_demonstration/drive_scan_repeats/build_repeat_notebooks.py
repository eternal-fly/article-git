"""Build Shenglian g-scan acquisition rounds 2 and 3."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "Shenglian_Q6T6L3_g_scan_cloud"
G_POINTS = {"040": 0.40, "060": 0.60, "100": 1.00}
ROUNDS = {2: 42002, 3: 42003}


def text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_text(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def replace_once(value: str, old: str, new: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old!r}")
    return value.replace(old, new, 1)


def build_one(template: dict, g_tag: str, g_value: float, round_id: int, order_seed: int) -> Path:
    notebook = copy.deepcopy(template)
    result_tag = f"sl_g{g_tag}_r{round_id}"

    set_text(
        notebook["cells"][0],
        f"""# Shenglian Q6/T6/L3 hardware scan: g={g_value:.2f}, round {round_id}

This is acquisition round {round_id} for the matched Shenglian hardware g scan.
It uses the unchanged processed 400-molecule NPZ, Q6/T6/L3 circuit, W=0.60,
reservoir seed 0, physical chain, and label-blind randomized submission logic.

Paste the API token directly into `API_TOKEN` in the configuration cell.
Set `ACTION=\"submit\"` for the 400 submissions and later set
`ACTION=\"collect\"` to create `{result_tag}_results.zip`.
""",
    )

    config = text(notebook["cells"][1])
    config = replace_once(
        config,
        "# Edit only ACTION, then run all cells. API token is entered securely below.",
        "# Edit ACTION and paste the API token below, then run all cells.",
    )
    config = replace_once(
        config,
        'from getpass import getpass\nAPI_TOKEN = getpass("New Quafu API token: ")',
        'API_TOKEN = "PASTE_QUAFU_API_TOKEN_HERE"',
    )
    config = replace_once(
        config,
        "SUBMISSION_ORDER_SEED = 20260728",
        f"SUBMISSION_ORDER_SEED = {order_seed}",
    )
    config = replace_once(
        config,
        f'RESULT_TAG = "sl_g{g_tag}"',
        f'RESULT_TAG = "{result_tag}"\nACQUISITION_ROUND = {round_id}',
    )
    set_text(notebook["cells"][1], config)

    helpers = text(notebook["cells"][2])
    helpers = replace_once(
        helpers,
        'PROTOCOL_VERSION = "2026-07-31-shenglian-canonical-v3-q6t6l3-gscan-v1"',
        'PROTOCOL_VERSION = "2026-07-31-shenglian-canonical-v3-q6t6l3-gscan-repeats-v1"',
    )
    set_text(notebook["cells"][2], helpers)

    cloud = text(notebook["cells"][5])
    cloud = replace_once(
        cloud,
        "    if not API_TOKEN:\n        raise RuntimeError(\"API token is empty\")",
        "    if (not API_TOKEN) or API_TOKEN == \"PASTE_QUAFU_API_TOKEN_HERE\":\n"
        "        raise RuntimeError(\"Paste the Quafu API token into API_TOKEN first\")",
    )
    cloud = replace_once(
        cloud,
        '                "result_tag": RESULT_TAG,\n                "g": G,',
        '                "result_tag": RESULT_TAG,\n'
        '                "acquisition_round": ACQUISITION_ROUND,\n'
        '                "g": G,',
    )
    cloud = replace_once(
        cloud,
        '            "result_tag": RESULT_TAG,\n            "chip": CHIP,',
        '            "result_tag": RESULT_TAG,\n'
        '            "acquisition_round": ACQUISITION_ROUND,\n'
        '            "chip": CHIP,',
    )
    cloud = replace_once(
        cloud,
        '        "backend": CHIP,\n        "physical_qubits": list(PHYSICAL_QUBITS),',
        '        "backend": CHIP,\n'
        '        "result_tag": RESULT_TAG,\n'
        '        "acquisition_round": ACQUISITION_ROUND,\n'
        '        "g": G,\n'
        '        "physical_qubits": list(PHYSICAL_QUBITS),',
    )
    set_text(notebook["cells"][5], cloud)

    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"r{round_id}-g{g_tag}-cell-{index:02d}"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    output = HERE / f"SL_g{g_tag}_r{round_id}.ipynb"
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    outputs = []
    for round_id, order_seed in ROUNDS.items():
        for g_tag, g_value in G_POINTS.items():
            source = BASE_DIR / f"SL_g{g_tag}.ipynb"
            template = json.loads(source.read_text(encoding="utf-8"))
            outputs.append(build_one(template, g_tag, g_value, round_id, order_seed))
    for output in outputs:
        print(output.name)

    expected = {
        f"SL_g{g_tag}_r{round_id}.ipynb": (g_value, round_id, order_seed)
        for round_id, order_seed in ROUNDS.items()
        for g_tag, g_value in G_POINTS.items()
    }
    for output in outputs:
        notebook = json.loads(output.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        g_value, round_id, order_seed = expected[output.name]
        result_tag = output.stem.lower()
        assertions = (
            f"G = {g_value:.2f}" in source,
            f"ACQUISITION_ROUND = {round_id}" in source,
            f"SUBMISSION_ORDER_SEED = {order_seed}" in source,
            f'RESULT_TAG = "{result_tag}"' in source,
            f"{result_tag}_results.zip" in source,
            'API_TOKEN = "PASTE_QUAFU_API_TOKEN_HERE"' in source,
            "getpass" not in source,
        )
        if not all(assertions):
            raise AssertionError((output.name, assertions))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{output.name}:cell{index}", "exec")

    dataset = HERE / "bace_bal400_canonical_v3_q6_t6.npz"
    source_dataset = BASE_DIR / dataset.name
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(dataset) != digest(source_dataset):
        raise AssertionError("Dataset copy differs from round-1 package")
    print(f"Validated {len(outputs)} notebooks; dataset SHA-256={digest(dataset)}")


if __name__ == "__main__":
    main()
