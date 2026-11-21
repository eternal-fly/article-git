"""Execute the validation notebook with the current (Molecule) Python environment."""

from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "01_spatrace_dtc_qrc_algorithm_validation.ipynb"
EXECUTED = ROOT / "01_spatrace_dtc_qrc_algorithm_validation.executed.ipynb"

if "Molecule" not in sys.executable:
    raise RuntimeError(f"Use /home/ubuntu/miniconda3/envs/Molecule/bin/python, got {sys.executable}")

notebook = nbformat.read(SOURCE, as_version=4)
client = NotebookClient(
    notebook,
    timeout=None,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=False,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, EXECUTED)
print(EXECUTED)

