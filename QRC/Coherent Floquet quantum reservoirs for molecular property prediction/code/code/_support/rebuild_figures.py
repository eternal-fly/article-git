"""Render the ten publication figures in the prepared release runtime."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    renderer = ROOT / "tools/publication_figures/render_figures.py"
    output = ROOT / "runs/figures"
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / "runs/matplotlib"))
    command = [sys.executable, str(renderer), "--project", str(ROOT), "--out", str(output)]
    subprocess.run(command + sys.argv[1:], env=env, check=True)


if __name__ == "__main__":
    main()
