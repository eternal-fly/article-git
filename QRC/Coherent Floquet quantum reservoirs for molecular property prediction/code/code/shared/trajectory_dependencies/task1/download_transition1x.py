#!/usr/bin/env python3
"""Parallel, resumable downloader for the official Transition1x HDF5 file."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import h5py


URL = "https://ndownloader.figshare.com/files/36035789"
EXPECTED_SIZE = 6_623_929_000
DEFAULT_TARGET = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Transition1x.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=4)
    return parser.parse_args()


def split_ranges(start: int, end: int, count: int) -> list[tuple[int, int]]:
    remaining = end - start + 1
    count = min(max(1, int(count)), remaining)
    base, extra = divmod(remaining, count)
    ranges = []
    cursor = start
    for index in range(count):
        length = base + (1 if index < extra else 0)
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    return ranges


def discover_covered_parts(
    parts_dir: Path,
) -> list[tuple[int, int, Path]]:
    """Return the actually downloaded (not merely declared) byte intervals."""
    covered = []
    pattern = re.compile(r".*_(\d+)_(\d+)\.bin$")
    for part_path in sorted(parts_dir.glob("*.bin")):
        match = pattern.fullmatch(part_path.name)
        if match is None:
            continue
        declared_start, declared_end = map(int, match.groups())
        size = part_path.stat().st_size
        expected = declared_end - declared_start + 1
        if size > expected:
            raise RuntimeError(
                f"{part_path}: {size} bytes exceeds declared range size {expected}"
            )
        if size:
            covered.append(
                (declared_start, declared_start + size - 1, part_path)
            )

    covered.sort(key=lambda item: item[0])
    for previous, current in zip(covered, covered[1:]):
        if current[0] <= previous[1]:
            raise RuntimeError(
                "overlapping downloaded parts: "
                f"{previous[2].name} and {current[2].name}"
            )
    return covered


def find_gaps(
    start: int,
    end: int,
    covered: list[tuple[int, int, Path]],
) -> list[tuple[int, int]]:
    gaps = []
    cursor = start
    for part_start, part_end, _ in covered:
        if part_end < start or part_start > end:
            continue
        if part_start < start:
            raise RuntimeError("a part overlaps the preserved target prefix")
        if cursor < part_start:
            gaps.append((cursor, part_start - 1))
        cursor = max(cursor, part_end + 1)
    if cursor <= end:
        gaps.append((cursor, end))
    return gaps


def repartition_gaps(
    gaps: list[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    """Split the largest missing intervals until all workers have a range."""
    ranges = list(gaps)
    count = max(1, int(count))
    while len(ranges) < count:
        splittable = [
            (end - start + 1, index)
            for index, (start, end) in enumerate(ranges)
            if end > start
        ]
        if not splittable:
            break
        _, index = max(splittable)
        start, end = ranges.pop(index)
        midpoint = (start + end) // 2
        ranges.extend([(start, midpoint), (midpoint + 1, end)])
    return sorted(ranges)


def download_range(
    index: int,
    start: int,
    end: int,
    part_path: Path,
    chunk_bytes: int,
) -> tuple[int, Path, int, float]:
    expected = end - start + 1
    started = time.perf_counter()
    attempts = 0
    while True:
        existing = part_path.stat().st_size if part_path.exists() else 0
        if existing > expected:
            raise RuntimeError(f"{part_path}: existing part is larger than its range")
        if existing == expected:
            return index, part_path, existing, time.perf_counter() - started
        attempts += 1
        if attempts > 200:
            raise RuntimeError(f"{part_path}: exceeded 200 resumable attempts")

        request_start = start + existing
        command = [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            "300",
            "--speed-limit",
            "32768",
            "--speed-time",
            "30",
            "--range",
            f"{request_start}-{end}",
            URL,
        ]
        with part_path.open("ab") as handle:
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.PIPE,
                check=False,
                text=False,
            )
        size = part_path.stat().st_size
        if size > expected:
            raise RuntimeError(
                f"{part_path}: server ignored range; expected at most "
                f"{expected} bytes, got {size}"
            )
        if result.returncode and size == existing:
            time.sleep(min(2.0 * attempts, 20.0))


def validate_hdf5(path: Path) -> None:
    if path.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(
            f"downloaded size is {path.stat().st_size}, expected {EXPECTED_SIZE}"
        )
    with h5py.File(path, "r") as handle:
        required = {"data", "train", "val", "test"}
        if set(handle.keys()) != required:
            raise RuntimeError(
                f"unexpected root groups: {sorted(handle.keys())}; "
                f"expected {sorted(required)}"
            )


def write_download_metadata(path: Path) -> Path:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata_path = path.with_suffix(path.suffix + ".download.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_url": URL,
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "hdf5_root_groups": ["data", "test", "train", "val"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def main() -> None:
    args = parse_args()
    target = args.target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix_size = target.stat().st_size if target.exists() else 0
    if prefix_size > EXPECTED_SIZE:
        raise RuntimeError(
            f"{target} is larger than the official file: {prefix_size} bytes"
        )
    if prefix_size == EXPECTED_SIZE:
        validate_hdf5(target)
        metadata_path = write_download_metadata(target)
        parts_dir = target.parent / ".transition1x_parts"
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        print(
            f"[download] already complete and valid: {target}\n"
            f"[download] metadata: {metadata_path}"
        )
        return

    parts_dir = target.parent / ".transition1x_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    covered_before = discover_covered_parts(parts_dir)
    missing = find_gaps(
        prefix_size, EXPECTED_SIZE - 1, covered_before
    )
    ranges = repartition_gaps(
        missing, int(args.connections)
    )
    covered_bytes = prefix_size + sum(
        end - start + 1 for start, end, _ in covered_before
    )
    print(
        f"[download] preserving {covered_bytes / 1024**3:.3f} GiB; "
        f"downloading {len(ranges)} missing ranges with "
        f"{args.connections} connections",
        flush=True,
    )
    if not ranges:
        completed = []
    else:
        completed = None
    futures = []
    if completed is None:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(args.connections)
        ) as executor:
            for index, (start, end) in enumerate(ranges):
                part_path = (
                    parts_dir / f"gap_{index:02d}_{start}_{end}.bin"
                )
                futures.append(
                    executor.submit(
                        download_range,
                        index,
                        start,
                        end,
                        part_path,
                        int(args.chunk_mib) * 1024 * 1024,
                    )
                )
            completed = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed.append(result)
                print(
                    f"[download] range {result[0]} complete: "
                    f"{result[2] / 1024**3:.3f} GiB in {result[3]:.1f}s",
                    flush=True,
                )

    covered_after = discover_covered_parts(parts_dir)
    remaining = find_gaps(prefix_size, EXPECTED_SIZE - 1, covered_after)
    if remaining:
        raise RuntimeError(f"download finished with missing intervals: {remaining}")

    assembled = target.with_suffix(target.suffix + ".complete")
    with assembled.open("wb") as output:
        if prefix_size:
            with target.open("rb") as prefix:
                shutil.copyfileobj(prefix, output, length=16 * 1024 * 1024)
        for _, _, part_path in covered_after:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, output, length=16 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())

    validate_hdf5(assembled)
    os.replace(assembled, target)
    validate_hdf5(target)
    metadata_path = write_download_metadata(target)
    shutil.rmtree(parts_dir)
    print(
        f"[download] complete and validated: {target}\n"
        f"[download] metadata: {metadata_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
