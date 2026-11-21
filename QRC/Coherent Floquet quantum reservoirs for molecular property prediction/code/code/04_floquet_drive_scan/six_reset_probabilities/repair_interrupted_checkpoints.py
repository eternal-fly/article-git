"""Quarantine empty interrupted blocks, then verify every surviving NPZ/JSON.

Never alter a completed case or a nonempty data artifact during recovery.
"""
from grid_common import *
import fcntl
import zipfile
from datetime import datetime, timezone


def main():
    lock = (HERE / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = HERE / "cases"
    empty = sorted(p for p in root.rglob("*") if p.is_file() and p.stat().st_size == 0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = HERE / "interrupted_checkpoint_quarantine" / stamp
    for path in empty:
        assert path.parent.name in ("chunks", "blocks"), path
        folder = path.parent.parent
        marker = folder / ("features_complete.json" if path.parent.name == "chunks" else "decoder_complete.json")
        assert not marker.exists(), f"Completed case has empty artifact: {path}"
    moved = []
    for path in empty:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        moved.append(str(target.relative_to(HERE)))
    counts = dict(json=0, npz=0, hashes=0)
    for path in sorted(root.rglob("*.json")):
        record = json.loads(path.read_text())
        counts["json"] += 1
        if path.parent.name in ("chunks", "blocks") and "sha256" in record:
            assert sha(path.with_suffix(".npz")) == record["sha256"], path
            counts["hashes"] += 1
        elif path.name == "features_complete.json":
            assert sha(path.parent / "p020_features.npz") == record["sha256"], path
            counts["hashes"] += 1
        elif path.name == "decoder_complete.json":
            assert sha(path.parent / "predictions.npz") == record["predictions_sha256"], path
            counts["hashes"] += 1
    for path in sorted(root.rglob("*.npz")):
        if ".partial." in path.name:
            continue
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, path
        counts["npz"] += 1
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  quarantined_empty_files=moved, verified=counts,
                  completed_cases_modified=False)
    jwrite(HERE / "interrupted_checkpoint_recovery.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
