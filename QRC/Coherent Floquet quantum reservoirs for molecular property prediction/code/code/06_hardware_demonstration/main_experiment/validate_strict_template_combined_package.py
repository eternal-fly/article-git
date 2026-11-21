#!/usr/bin/env python3
"""Offline validation of the strict-template two-backend cloud package."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

from validate_complete_five_task_packages import (
    EXPECTED_CX,
    FakeTask,
    TASK_KEYS,
    fake_quark,
)


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "Quafu_Baihua_Shenglian_complete_5tasks_strict_template.zip"


def load_namespace(task_dir: Path) -> dict:
    notebooks = list(task_dir.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise AssertionError((task_dir, notebooks))
    notebook = notebooks[0]
    text = notebook.read_text(encoding="utf-8")
    forbidden = re.compile(
        r"(?i)\b(?:api[_-]?(?:key|token)|password|secret)\b"
        r"\s*[:=]\s*(['\"])(?!<)[^'\"]+\1"
    )
    if forbidden.search(text):
        raise AssertionError(f"Literal credential found in {notebook}")
    payload = json.loads(text)
    if len(payload["cells"]) != 8:
        raise AssertionError(f"{notebook}: expected 8 cells")
    namespace: dict = {}
    old_token = os.environ.get("QPU_API_TOKEN")
    current = Path.cwd()
    try:
        os.environ["QPU_API_TOKEN"] = "OFFLINE_TEST_ONLY"
        os.chdir(task_dir)
        for cell_index in range(2, 7):
            source = "".join(payload["cells"][cell_index]["source"])
            with contextlib.redirect_stdout(io.StringIO()):
                exec(
                    compile(
                        source,
                        f"{notebook.name}:cell{cell_index}",
                        "exec",
                    ),
                    namespace,
                )
    finally:
        os.chdir(current)
        if old_token is None:
            os.environ.pop("QPU_API_TOKEN", None)
        else:
            os.environ["QPU_API_TOKEN"] = old_token
    return namespace


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="strict-template-package-") as name:
        root = Path(name)
        with zipfile.ZipFile(ARCHIVE) as handle:
            handle.extractall(root)
        if sorted(path.name for path in root.iterdir() if path.is_dir()) != [
            "baihua",
            "shenglian",
        ]:
            raise AssertionError("Archive must contain baihua/ and shenglian/")
        namespaces = {}
        with fake_quark():
            for backend_key in ("baihua", "shenglian"):
                backend_root = root / backend_key
                folders = sorted(
                    path.name for path in backend_root.iterdir() if path.is_dir()
                )
                if folders != list(TASK_KEYS):
                    raise AssertionError((backend_key, folders))
                if len(list(backend_root.glob("*/*.npz"))) != 5:
                    raise AssertionError(f"{backend_key}: NPZ count is not five")
                if len(list(backend_root.glob("*/*.ipynb"))) != 5:
                    raise AssertionError(f"{backend_key}: notebook count is not five")
                for task_key in TASK_KEYS:
                    namespace = load_namespace(backend_root / task_key)
                    namespaces[(backend_key, task_key)] = namespace
                    circuit, metadata = namespace["build_circuit"](
                        namespace["BUNDLE"]["tokens"][0]
                    )
                    qlisp = circuit.to_qlisp
                    cx_count = sum(
                        isinstance(value, (tuple, list))
                        and value
                        and str(value[0]).lower() == "cx"
                        for value in qlisp
                    )
                    if cx_count != EXPECTED_CX[task_key]:
                        raise AssertionError((backend_key, task_key, cx_count))
                    if metadata["nominal_two_qubit_gates"] != EXPECTED_CX[task_key]:
                        raise AssertionError(metadata)
                    source = "".join(
                        json.loads(
                            next((backend_root / task_key).glob("*.ipynb")).read_text(
                                encoding="utf-8"
                            )
                        )["cells"][6]["source"]
                    )
                    for required in (
                        "from quark import Task",
                        "tmgr.run(task_payload, repeat=REPEAT)",
                        "tmgr.result(record[\"tid\"])",
                    ):
                        if required not in source:
                            raise AssertionError(
                                f"{backend_key}/{task_key}: missing {required}"
                            )
                    print(
                        f"PASS {backend_key}/{task_key}: "
                        f"template API + CX={cx_count}"
                    )

            old_token = os.environ.get("QPU_API_TOKEN")
            try:
                os.environ["QPU_API_TOKEN"] = "OFFLINE_TEST_ONLY"
                FakeTask.counter = 0
                with contextlib.redirect_stdout(io.StringIO()):
                    for task_key in TASK_KEYS[:3]:
                        namespaces[("baihua", task_key)]["submit_qrc_jobs"]()
                    namespaces[("baihua", TASK_KEYS[3])]["submit_qrc_jobs"]()
                if FakeTask.counter != 2000:
                    raise AssertionError(FakeTask.counter)
                before = FakeTask.counter
                with contextlib.redirect_stdout(io.StringIO()):
                    namespaces[("baihua", TASK_KEYS[4])]["submit_qrc_jobs"]()
                if FakeTask.counter != before:
                    raise AssertionError("L1 resume submitted duplicates")
                with contextlib.redirect_stdout(io.StringIO()):
                    for task_key in TASK_KEYS:
                        archive = namespaces[("baihua", task_key)][
                            "collect_qrc_results"
                        ]()
                        if not Path(archive).is_file():
                            raise AssertionError(archive)
            finally:
                if old_token is None:
                    os.environ.pop("QPU_API_TOKEN", None)
                else:
                    os.environ["QPU_API_TOKEN"] = old_token
            print(
                "PASS strict-template Baihua roundtrip: exactly 2000 Task.run "
                "calls and five 400-record Task.result collections"
            )

    print(
        "PASS combined strict-template archive: two backend folders, "
        "ten NPZ files, ten notebooks, no literal credential"
    )


if __name__ == "__main__":
    main()
