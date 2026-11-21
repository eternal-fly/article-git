#!/usr/bin/env python3
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(path for path in HERE.parents if path.name == "5_graph_stream_validity")
sys.path.insert(0, str(ROOT))
from validity_core import run_plan

run_plan(HERE.with_name("plan.json"))
