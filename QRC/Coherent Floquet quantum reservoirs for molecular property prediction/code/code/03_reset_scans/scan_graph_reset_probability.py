#!/usr/bin/env python3
"""Documented release entry point for the graph-reset workflow."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_support'))
from release_runtime import launch

if __name__ == '__main__':
    launch('graph-reset')
