"""Launch only this authorized, resumable CPU scan."""
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import jwrite

if __name__ == "__main__":
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLCONFIGDIR"] = str(HERE/".mplconfig")
    with (HERE/"run.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, "-u", str(HERE/"run.py"), "run",
                                    "--threads", "24", "--decoder-workers", "3", "--chunk-size", "96"],
                                    cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    jwrite(HERE/"launch.json", dict(pid=process.pid, execution="CPU: 24 feature threads + 3 decoder workers",
                                   log="run.log"))
    print(f"Launched PID {process.pid}", flush=True)
