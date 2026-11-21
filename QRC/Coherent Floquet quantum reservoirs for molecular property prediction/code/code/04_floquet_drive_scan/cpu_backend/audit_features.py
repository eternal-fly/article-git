"""Read-only verification of all completed feature caches."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *


def main():
    expected = widths()
    records = []
    for g in GS:
        for seed in SEEDS:
            folder = case_dir(g, seed)
            path = folder / "features.npz"
            marker = json.loads((folder / "features_complete.json").read_text())
            assert marker["g"] == g and marker["seed"] == seed
            assert marker["rows"] == 2376 and marker["version"] == VERSION
            assert path.stat().st_size == marker["bytes"]
            assert sha(path) == marker["sha256"], str(path)
            maximum = 0.0
            with np.load(path, allow_pickle=False) as bank:
                assert set(bank.files) == {pkey(p) for p in expected}
                for p, width in expected.items():
                    x = bank[pkey(p)]
                    assert x.shape == (2376, width) and x.dtype == np.float32
                    assert np.isfinite(x).all()
                    maximum = max(maximum, float(np.abs(x).max()))
            assert maximum <= 1.0002, (g, seed, maximum)
            records.append(dict(g=g, seed=seed, source=marker["source"],
                                maximum_absolute_expectation=maximum,
                                maximum_norm_error=marker.get("maximum_norm_error"),
                                bytes=marker["bytes"], sha256=marker["sha256"]))
            if len(records) % 50 == 0:
                print(f"Feature caches verified: {len(records)}/300", flush=True)
    assert len(records) == 300
    cwrite(HERE / "feature_cache_audit.csv", records)
    report = dict(passed=True, feature_cases=300, matched_hashes=300,
                  expected_input_windows_per_case=2376,
                  sources=pd.Series([r["source"] for r in records]).value_counts().to_dict(),
                  maximum_absolute_expectation=max(r["maximum_absolute_expectation"] for r in records),
                  finite=True, shapes_verified=True)
    jwrite(HERE / "feature_cache_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
