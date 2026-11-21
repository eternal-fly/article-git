"""Run all general BACE/BBBP minimum setting folders sequentially.

Order:
1. BACE Q=T scan, Q=T=1..20
2. BACE fixed-Q/T scan, Q in 8,12,16,20 and T=1..30
3. BBBP Q=T scan, Q=T=1..20
4. BBBP fixed-Q/T scan, Q in 8,12,16,20 and T=1..30

Every setting is executed through that folder's own ``run_this_setting.py``.
The runner only handles ordering, logging and final scan-level aggregation.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
LOG_ROOT = PROJECT_ROOT / "logs"
FIXED_Q_VALUES = (8, 12, 16, 20)
TOTAL_JOBS = 280


@dataclass(frozen=True)
class SettingJob:
    order: int
    dataset: str
    scan_kind: str
    q: int
    t: int
    setting_dir: Path


def iter_jobs() -> list[SettingJob]:
    jobs: list[SettingJob] = []
    order = 0
    for dataset in ("BACE", "BBBP"):
        dataset_root = PROJECT_ROOT / "general" / dataset
        qeq_root = dataset_root / "01_Q_equal_T_scan"
        for value in range(1, 21):
            order += 1
            jobs.append(
                SettingJob(
                    order=order,
                    dataset=dataset,
                    scan_kind="Q_equal_T",
                    q=value,
                    t=value,
                    setting_dir=qeq_root / f"Q{value:02d}_T{value:02d}",
                )
            )
        fixed_root = dataset_root / "02_fixed_Q_T_scan"
        for q in FIXED_Q_VALUES:
            for t in range(1, 31):
                order += 1
                jobs.append(
                    SettingJob(
                        order=order,
                        dataset=dataset,
                        scan_kind="fixed_Q_T_scan",
                        q=q,
                        t=t,
                        setting_dir=fixed_root / f"Q{q:02d}" / f"T{t:02d}",
                    )
                )
    return jobs


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(job: SettingJob) -> str:
    return f"{job.order:03d}_{job.dataset}_{job.scan_kind}_Q{job.q:02d}_T{job.t:02d}.log"


def _write_status_header(status_path: Path) -> None:
    with status_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order",
                "dataset",
                "scan_kind",
                "q",
                "t",
                "setting_dir",
                "status",
                "start_time",
                "end_time",
                "duration_sec",
                "returncode",
                "log_path",
            ]
        )


def _append_status(status_path: Path, job: SettingJob, status: str, start: float, end: float, returncode: int, log_path: Path) -> None:
    with status_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                job.order,
                job.dataset,
                job.scan_kind,
                job.q,
                job.t,
                str(job.setting_dir),
                status,
                datetime.fromtimestamp(start).isoformat(timespec="seconds"),
                datetime.fromtimestamp(end).isoformat(timespec="seconds"),
                f"{end - start:.1f}",
                returncode,
                str(log_path),
            ]
        )


def _validate_jobs(jobs: list[SettingJob]) -> None:
    if len(jobs) != TOTAL_JOBS:
        raise RuntimeError(f"Expected {TOTAL_JOBS} setting jobs, got {len(jobs)}")
    missing = [job.setting_dir / "run_this_setting.py" for job in jobs if not (job.setting_dir / "run_this_setting.py").exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing run_this_setting.py in {len(missing)} folders; first entries:\n{preview}")


def _run_job(job: SettingJob, run_dir: Path, status_path: Path) -> None:
    log_path = run_dir / _safe_name(job)
    start = time.time()
    print(
        f"[{_now()}] START {job.order:03d}/{TOTAL_JOBS} {job.dataset} {job.scan_kind} Q={job.q} T={job.t}",
        flush=True,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {_now()} START {job.dataset} {job.scan_kind} Q={job.q} T={job.t}\n")
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
        log.write(f"\n# {_now()} END returncode={completed.returncode}\n")
    end = time.time()
    if completed.returncode == 0:
        _append_status(status_path, job, "done", start, end, completed.returncode, log_path)
        print(
            f"[{_now()}] DONE  {job.order:03d}/{TOTAL_JOBS} {job.dataset} {job.scan_kind} Q={job.q} T={job.t} "
            f"({end - start:.1f}s)",
            flush=True,
        )
        return
    _append_status(status_path, job, "failed", start, end, completed.returncode, log_path)
    print(
        f"[{_now()}] FAIL  {job.order:03d}/{TOTAL_JOBS} {job.dataset} {job.scan_kind} Q={job.q} T={job.t}; "
        f"log={log_path}",
        flush=True,
    )
    raise SystemExit(completed.returncode)


def _aggregate_figures() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.linear_readout import plot_fixed_q_scan, plot_q_equal_t_scan

    for dataset in ("BACE", "BBBP"):
        dataset_root = PROJECT_ROOT / "general" / dataset
        print(f"[{_now()}] Aggregating scan figures for {dataset}", flush=True)
        plot_q_equal_t_scan(dataset_root / "01_Q_equal_T_scan", dataset)
        plot_fixed_q_scan(dataset_root / "02_fixed_Q_T_scan", dataset)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_ROOT / f"general_sequential_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = LOG_ROOT / "latest_general_sequential_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir) + "\n", encoding="utf-8")
    status_path = run_dir / "status.csv"
    _write_status_header(status_path)

    jobs = iter_jobs()
    _validate_jobs(jobs)
    print(f"[{_now()}] Sequential run directory: {run_dir}", flush=True)
    print(f"[{_now()}] Python executable: {PYTHON}", flush=True)
    print(f"[{_now()}] Total settings: {len(jobs)}", flush=True)

    for job in jobs:
        _run_job(job, run_dir, status_path)

    _aggregate_figures()
    (run_dir / "ALL_DONE.txt").write_text(f"Completed all settings at {_now()}\n", encoding="utf-8")
    print(f"[{_now()}] ALL DONE. Logs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
