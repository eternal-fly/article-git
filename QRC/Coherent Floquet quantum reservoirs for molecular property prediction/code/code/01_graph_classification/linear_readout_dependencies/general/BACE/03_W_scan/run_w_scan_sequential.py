"""Run BACE/general W-scan folders sequentially."""

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
SETTINGS = ((12, 12), (16, 16))
W_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TOTAL_JOBS = len(SETTINGS) * len(W_VALUES)


@dataclass(frozen=True)
class Job:
    order: int
    q: int
    t: int
    W: float
    setting_dir: Path


def w_label(value: float) -> str:
    return f"W{value:.1f}".replace(".", "p")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def iter_jobs() -> list[Job]:
    jobs = []
    order = 0
    for q, t in SETTINGS:
        for W in W_VALUES:
            order += 1
            jobs.append(Job(order, q, t, W, SCAN_ROOT / f"Q{q:02d}_T{t:02d}" / w_label(W)))
    return jobs


def write_status_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "q", "t", "W", "setting_dir", "status", "start_time", "end_time", "duration_sec", "returncode", "log_path"])


def append_status(path: Path, job: Job, status: str, start: float, end: float, returncode: int, log_path: Path) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            job.order, job.q, job.t, f"{job.W:.1f}", str(job.setting_dir), status,
            datetime.fromtimestamp(start).isoformat(timespec="seconds"),
            datetime.fromtimestamp(end).isoformat(timespec="seconds"),
            f"{end - start:.1f}", returncode, str(log_path),
        ])


def validate_jobs(jobs: list[Job]) -> None:
    missing = [job.setting_dir / "run_this_setting.py" for job in jobs if not (job.setting_dir / "run_this_setting.py").exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing run_this_setting.py in {len(missing)} folders; first entries:\n{preview}")


def run_job(job: Job, run_dir: Path, status_path: Path) -> None:
    log_path = run_dir / f"{job.order:03d}_BACE_W_scan_Q{job.q:02d}_T{job.t:02d}_{w_label(job.W)}.log"
    start = time.time()
    print(f"[{now()}] START {job.order:03d}/{TOTAL_JOBS} BACE W_scan Q={job.q} T={job.t} W={job.W:.1f}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {now()} START BACE W_scan Q={job.q} T={job.t} W={job.W:.1f}\n")
        log.write(f"# setting_dir = {job.setting_dir}\n")
        log.flush()
        completed = subprocess.run(
            [str(PYTHON), "run_this_setting.py"],
            cwd=str(job.setting_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        log.write(f"\n# {now()} END returncode={completed.returncode}\n")
    end = time.time()
    if completed.returncode == 0:
        append_status(status_path, job, "done", start, end, completed.returncode, log_path)
        print(f"[{now()}] DONE  {job.order:03d}/{TOTAL_JOBS} BACE W_scan Q={job.q} T={job.t} W={job.W:.1f} ({end - start:.1f}s)", flush=True)
        return
    append_status(status_path, job, "failed", start, end, completed.returncode, log_path)
    print(f"[{now()}] FAIL  {job.order:03d}/{TOTAL_JOBS} BACE W_scan Q={job.q} T={job.t} W={job.W:.1f}; log={log_path}", flush=True)
    raise SystemExit(completed.returncode)


def aggregate_figures() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.linear_readout import plot_w_scan

    print(f"[{now()}] Aggregating BACE W-scan figure", flush=True)
    plot_w_scan(SCAN_ROOT, "BACE")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_ROOT / f"bace_general_w_scan_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = LOG_ROOT / "latest_bace_general_w_scan_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir) + "\n", encoding="utf-8")
    status_path = run_dir / "status.csv"
    write_status_header(status_path)

    jobs = iter_jobs()
    validate_jobs(jobs)
    print(f"[{now()}] W-scan run directory: {run_dir}", flush=True)
    print(f"[{now()}] Python executable: {PYTHON}", flush=True)
    print(f"[{now()}] Total settings: {len(jobs)}", flush=True)

    for job in jobs:
        run_job(job, run_dir, status_path)

    aggregate_figures()
    (run_dir / "ALL_DONE.txt").write_text(f"Completed all settings at {now()}\n", encoding="utf-8")
    print(f"[{now()}] ALL DONE. Logs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
