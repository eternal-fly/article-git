"""Controlled graph-event stream benchmarks for FINAL QRC."""

from .experiment import (
    CANONICAL_G,
    CANONICAL_LAYERS,
    CANONICAL_W,
    RunConfig,
    available_devices,
    run_experiment,
    run_replay,
)
from .tasks import ControlledDataset, TaskSetting, generate_dataset, task_settings

__all__ = [
    "CANONICAL_G",
    "CANONICAL_LAYERS",
    "CANONICAL_W",
    "ControlledDataset",
    "RunConfig",
    "TaskSetting",
    "available_devices",
    "generate_dataset",
    "run_experiment",
    "run_replay",
    "task_settings",
]

