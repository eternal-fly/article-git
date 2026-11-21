"""Read-only verification of all consolidated feature caches during decoding."""
from grid_common import *
from collections import Counter
from datetime import datetime, timezone


def main():
    records = []
    origins = Counter()
    for g in GS:
        for seed in SEEDS:
            folder = folder_for(g, seed)
            marker = json.loads((folder / "features_complete.json").read_text())
            path = folder / "p020_features.npz"
            assert marker["g"] == g and marker["seed"] == seed and marker["p"] == .2
            assert sha(path) == marker["sha256"], path
            with np.load(path, allow_pickle=False) as archive:
                values = archive["features"]
                assert values.shape == (2376, 360) and values.dtype == np.float32
                assert np.isfinite(values).all() and np.abs(values).max() < 1.0002
            assert marker["maximum_closed_feature_error"] < 5e-5
            assert marker["maximum_norm_error"] < 5e-4
            origins[marker["source"]] += 1
            records.append(dict(g=g, seed=seed, source=marker["source"],
                gpu=marker.get("gpu"), sha256=marker["sha256"],
                closed_feature_error=marker["maximum_closed_feature_error"],
                norm_error=marker["maximum_norm_error"]))
    cwrite(HERE / "completed_features_early_audit.csv", records)
    report = dict(passed=True, cases=len(records), expected_cases=300,
        created_utc=datetime.now(timezone.utc).isoformat(), origins=dict(origins),
        maximum_closed_feature_error=max(r["closed_feature_error"] for r in records),
        maximum_norm_error=max(r["norm_error"] for r in records),
        finalized_feature_files_modified=False)
    jwrite(HERE / "completed_features_early_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
