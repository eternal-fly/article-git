"""Verify full-data dephasing refits and summarize paired estimator convergence.

Only reads completed experiment cells. This does not select a physical parameter,
alter the manuscript, or overwrite the legacy K=8 results.
"""
from __future__ import annotations

import argparse
import itertools
from datetime import datetime, timezone

from common import *
from a01_core import TARGET_NAMES, regression_metrics
from readouts import score_metrics, split_indices
from run_dephasing_full_k import (
    OUT, K_VALUES, P_VALUES, atomic_json, coherent_bank_path, fingerprint, load_task, weights,
)


def close(a, b, label, atol=1e-9, rtol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=rtol):
        delta = float(np.max(np.abs(a-b))) if a.shape == b.shape else "shape mismatch"
        raise AssertionError(f"{label}: {a.shape} / {b.shape}, error={delta}")
    return float(np.max(np.abs(a-b))) if a.size else 0.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def source_audit():
    path = ROOT / "BFS-QRC/PRL/revised_open_dtc_qrc/pre_supplement_protocol_audit_20260905/audit_results.json"
    reference = json.loads(path.read_text())["source_sha256"]
    rows = []
    for name, expected in reference.items():
        actual = digest(ROOT / name)
        rows.append(dict(path=name, expected_sha256=expected, actual_sha256=actual, unchanged=actual == expected))
    return rows


def classify_source_audit(rows):
    # These two publication artifacts are not read by feature extraction,
    # fitting, prediction, or statistical aggregation. Keep their drift visible
    # without treating concurrent manuscript edits as a failed experiment.
    documents = {
        "BFS-QRC/PRL/revised_open_dtc_qrc/DTC_QRC_PRAPPLIED_open_en.tex",
        "BFS-QRC/PRL/revised_open_dtc_qrc/DTC_QRC_PRAPPLIED_open_en.pdf",
    }
    changed = [row for row in rows if not row["unchanged"]]
    document_changes = [row for row in changed if row["path"] in documents]
    blocking_changes = [row for row in changed if row["path"] not in documents]
    return dict(document_changes=document_changes, blocking_source_changes=blocking_changes,
                experiment_source_manifest_unchanged=not blocking_changes,
                unchanged_manifest_entries=sum(bool(row["unchanged"]) for row in rows),
                document_drift_warning=bool(document_changes))


def verify_features(task, strength, seed):
    data, base = load_task(task)
    folder = OUT / f"features/{task}/lambda{strength:.1f}/seed{seed:03d}"
    meta = json.loads((folder / "metadata.json").read_text())
    expected = dict(**base, seed=seed, dephasing=strength, exact=strength == 0)
    assert meta["fingerprint"] == fingerprint(expected)
    assert json.loads((folder / "complete.json").read_text())["fingerprint"] == meta["fingerprint"]
    with np.load(folder / "features.npz") as z:
        features, standard_error = z["features"], z["standard_error"]
        ks = tuple(z["K"].tolist())
        close(z["p"], P_VALUES, "reset grid", atol=0, rtol=0)
        assert json.loads(str(z["metadata"])) == meta
    assert ks == ((0,) if strength == 0 else K_VALUES)
    assert features.shape == (len(ks), 2, base["n"], base["width"])
    assert standard_error.shape == features.shape
    assert np.isfinite(features).all() and np.isfinite(standard_error).all()
    assert np.max(np.abs(features)) < 1.001 and np.min(standard_error) >= 0
    mix = np.zeros_like(features, dtype=float)
    variance = np.zeros_like(mix)
    wp = [weights(base["t"], p, task) for p in P_VALUES]
    for pi, (_, anchor) in enumerate(wp):
        mix[:, pi] += anchor
    probe_error, norm_error = 0., 0.
    if strength == 0:
        with np.load(coherent_bank_path(task, seed)) as z:
            bank = z["suffixes"].astype(float)
        for pi, (w, anchor) in enumerate(wp):
            mix[0, pi] = np.tensordot(w, bank, axes=(0, 0)) + anchor
    else:
        parity = json.loads((folder / "parity.json").read_text())
        assert parity["max_feature_error"] < 5e-5 and parity["norm_error"] < 1e-4
        for suffix in range(base["t"]):
            with np.load(folder / f"suffix{suffix:02d}.npz") as z:
                assert str(z["fingerprint"]) == meta["fingerprint"]
                mean, var, probe = z["means"], z["variances"], z["trial_probe"]
                assert mean.shape == var.shape == (3, base["n"], base["width"])
                assert probe.shape == (2, 128, base["width"])
                assert tuple(z["K"].tolist()) == K_VALUES
                assert np.isfinite(mean).all() and np.isfinite(var).all() and (var >= 0).all()
                norm_error = max(norm_error, float(z["max_norm_error"]))
            for ki, k in enumerate(K_VALUES):
                error = close(mean[ki, :2], probe[:, :k].mean(1, dtype=float), "nested K mean", atol=1e-7)
                probe_error = max(probe_error, error)
                close(var[ki, :2], probe[:, :k].var(1, ddof=1, dtype=float), "nested K variance", atol=1e-7)
            for pi, (w, _) in enumerate(wp):
                mix[:, pi] += w[suffix]*mean
                variance[:, pi] += w[suffix]**2*var/np.asarray(K_VALUES)[:, None, None]
        assert norm_error < 2e-4
    mix_error = close(features, mix.astype(np.float32), "exact suffix mixture", atol=0, rtol=0)
    close(standard_error, np.sqrt(variance).astype(np.float32), "mixture estimator SE", atol=0, rtol=0)
    checks = dict(task=task, dephasing=strength, seed=seed, source_fingerprint=meta["fingerprint"],
                  mix_error=mix_error, probe_mean_error=probe_error, max_norm_error=norm_error, passed=True)
    precision = []
    for ki, k in enumerate(ks):
        for pi, p in enumerate(P_VALUES):
            for readout, width in (("final", base["q"]*(base["q"]+1)//2), ("tm3", base["width"])):
                if task == "bace" and readout == "tm3":
                    continue
                x, se = features[ki, pi, :, :width], standard_error[ki, pi, :, :width]
                signal = float(np.sqrt(np.mean(x.var(0, dtype=float))))
                rms_se = float(np.sqrt(np.mean(se.astype(float)**2)))
                precision.append(dict(task=task, dephasing=strength, seed=seed, K=k, p=p, readout=readout,
                                      estimator_rms_se=rms_se, feature_sample_sd_rms=signal,
                                      se_to_sample_sd=rms_se/signal if signal > 0 else np.nan))
    return data, features, ks, checks, precision


def selected_bace(screen):
    assert len(screen) == 9
    return screen.sort_values(["val_auc", "val_loss", "val_acc", "C", "gamma_scale"],
                              ascending=[False, True, False, True, True]).iloc[0]


def verify_bace(folder, x, data, seed):
    import joblib
    split_path = FINAL / f"16_dissipative_protocol_migration/cache/splits/BACE/seed{seed:03d}.json"
    train, val, test = split_indices(json.loads(split_path.read_text()))
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
    rows = pd.read_csv(folder / "metrics.csv")
    assert len(rows) == 1
    row = rows.iloc[0]
    winner = selected_bace(pd.read_csv(folder / "validation_screen.csv"))
    for key in winner.index:
        close(row[key], winner[key], f"BACE selected {key}")
    artifact = joblib.load(folder / "decoder_model.joblib")
    scaler, model = artifact["scaler"], artifact["model"]
    close(scaler.mean_, x[train].mean(0, dtype=float), "train-only scaler", atol=1e-8)
    assert int(scaler.n_samples_seen_) == len(train)
    for key in ("C", "gamma", "epsilon"):
        close(model.get_params()[key], row[key], f"BACE model parameter {key}")
    with np.load(folder / "predictions.npz") as z:
        close(z["test_indices"], test, "BACE test indices", atol=0, rtol=0)
        close(z["labels"], data["labels"][test], "BACE test labels", atol=0, rtol=0)
        scores = z["test_scores"]
    error = close(scores, model.predict(scaler.transform(x[test])), "BACE saved model inference")
    for key, value in score_metrics(data["labels"][test], scores).items():
        close(row[f"test_{key}"], value, f"BACE test {key}")
    for key, value in score_metrics(data["labels"][val], model.predict(scaler.transform(x[val]))).items():
        close(row[f"val_{key}"], value, f"BACE validation {key}")
    return rows, [], error


def verify_hopping(folder, x, data):
    import joblib
    trainval, test = data["split"] != 2, data["split"] == 2
    split_sets = [set(data["trajectory_id"][data["split"] == i]) for i in range(3)]
    assert all(not split_sets[a] & split_sets[b] for a, b in itertools.combinations(range(3), 2))
    assert [len(s) for s in split_sets] == [178, 59, 60]
    rows = pd.read_csv(folder / "metrics.csv")
    screen = pd.read_csv(folder / "validation_screen.csv")
    assert len(rows) == 8 and len(screen) == 4*(7+4*4*2)
    artifact = joblib.load(folder / "decoder_models.joblib")
    scaler, models = artifact["scaler"], artifact["models"]
    close(scaler.mean_, x[trainval].mean(0, dtype=float), "train+validation scaler", atol=1e-8)
    assert int(scaler.n_samples_seen_) == int(trainval.sum())
    with np.load(folder / "predictions.npz") as z:
        predictions, names, decoders = z["predictions"], z["target"], z["decoder"]
        close(z["test_indices"], np.flatnonzero(test), "hopping test indices", atol=0, rtol=0)
        close(z["target_values"], data["targets"][test], "hopping targets", atol=0, rtol=0)
        assert np.array_equal(z["trajectory_id"], data["trajectory_id"][test])
        close(z["endpoint"], data["endpoint"][test], "hopping window endpoints", atol=0, rtol=0)
    assert predictions.shape == (8, int(test.sum()))
    trajectories, error = [], 0.
    for i, row in rows.iterrows():
        name, decoder = row["target"], row["decoder"]
        assert name == names[i] and decoder == decoders[i]
        candidates = screen[(screen.target == name) & (screen.decoder == decoder)]
        params = ["alpha"] if decoder == "linear" else ["C", "gamma_scale", "epsilon"]
        winner = candidates.sort_values(["validation_mae_ev", *params]).iloc[0]
        for key in [*params, "validation_mae_ev"]:
            close(row[key], winner[key], f"hopping selected {key}")
        model = models[(name, decoder)]
        for key in (["alpha"] if decoder == "linear" else ["C", "gamma", "epsilon"]):
            close(model.get_params()[key], row[key], f"hopping model parameter {key}")
        error = max(error, close(predictions[i], model.predict(scaler.transform(x[test])),
                                 "hopping saved model inference", atol=2e-6))
        y = data["targets"][test, TARGET_NAMES.index(name)].astype(float)
        for key, value in regression_metrics(y, predictions[i]).items():
            close(row[key], value, f"hopping test {key}")
        absolute_error = np.abs(predictions[i]-y)
        identifiers = {key: row[key] for key in ("task", "dephasing", "seed", "K", "p", "q", "t", "readout", "target", "decoder")}
        for tid in sorted(split_sets[2]):
            mask = data["trajectory_id"][test] == tid
            trajectories.append(dict(**identifiers, trajectory_id=tid, n_windows=int(mask.sum()),
                                     mae_ev=float(absolute_error[mask].mean())))
    return rows, trajectories, error


def coherent_parity(metrics):
    rows = []
    for _, row in metrics[metrics.dephasing == 0].iterrows():
        seed = int(row.seed)
        if row.task == "bace":
            path = HERE / f"dephasing/legacy8_decoder_audit/coherent/seed{seed:03d}/metrics.csv"
            reference = pd.read_csv(path)
            reference = reference[np.isclose(reference.p, row.p)]
            fields, tolerance = ["C", "gamma_scale", "epsilon", "test_auc", "val_auc"], 1e-9
        else:
            name = "hopping_linear_metrics_by_seed.csv" if row.decoder == "linear" else "hopping_metrics_by_seed.csv"
            path = ROOT / "QRC-J-mechanism-diagnostics/experiments/18_coherence_interaction_ablation/results" / name
            reference = pd.read_csv(path)
            reference = reference[(reference.variant == "coherent") & (reference.seed == seed)
                                  & np.isclose(reference.p, row.p) & (reference.readout == row.readout)
                                  & (reference.target == row.target)]
            fields = (["alpha"] if row.decoder == "linear" else ["C", "gamma_scale", "epsilon"])
            fields += ["mae_ev", "rmse_ev", "r2", "validation_mae_ev"]
            tolerance = 1e-5 if row.decoder == "linear" else 1e-8
        assert len(reference) == 1
        error = max(close(row[field], reference.iloc[0][field], f"original coherent {field}", atol=tolerance)
                    for field in fields)
        rows.append(dict(task=row.task, seed=seed, p=row.p, readout=row.readout, decoder=row.decoder,
                         target=row.get("target", "BACE"), max_metric_difference=error, tolerance=tolerance,
                         reference=str(path.relative_to(ROOT)), passed=True))
    return rows


def interval(values, label):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    assert values.shape[0] == 6 and np.isfinite(values).all()
    result = crossed_interval(values, label)
    return dict(**result, sd_across_seed_means=float(values.mean(1).std(ddof=1)), n_seeds=6,
                n_trajectories=values.shape[1] if values.shape[1] > 1 else np.nan,
                ci_method="crossed_seed_trajectory_bootstrap" if values.shape[1] > 1 else "paired_seed_bootstrap",
                bootstrap_draws=20000)


def matrix(group, value):
    if "trajectory_id" in group:
        pivot = group.pivot(index="seed", columns="trajectory_id", values=value).sort_index()
        assert tuple(pivot.index) == SEEDS and pivot.shape[1] == 60 and not pivot.isna().any().any()
        return pivot.to_numpy()
    assert len(group) == 6 and tuple(group.sort_values("seed").seed.astype(int)) == SEEDS
    return group.sort_values("seed")[value].to_numpy()[:, None]


def summarize(frame, keys, value, out, prefix):
    summaries, deltas = [], []
    if frame.empty:
        return
    group_keys = ["dephasing", "K", "p", *keys]
    complete = {}
    for key, group in frame.groupby(group_keys, dropna=False, sort=True):
        if set(group.seed) != set(SEEDS):
            continue
        values = matrix(group, value)
        complete[key] = values
        info = dict(zip(group_keys, key))
        summaries.append(dict(**info, metric=value, **interval(values, f"{prefix}|{key}|summary")))
    for key, values in complete.items():
        strength, k, p, *other = key
        comparisons = []
        if strength != 0:
            comparisons.append(((0., 0, p, *other), "dephased_minus_coherent"))
            if k in (8, 32):
                comparisons.append(((strength, 128, p, *other), f"K{k}_minus_K128"))
        for baseline_key, comparison in comparisons:
            if baseline_key not in complete:
                continue
            delta = values-complete[baseline_key]
            deltas.append(dict(**dict(zip(group_keys, key)), metric=value, comparison=comparison,
                               reference_dephasing=baseline_key[0], reference_K=baseline_key[1],
                               **interval(delta, f"{prefix}|{key}|{comparison}")))
    write_csv(out / f"{prefix}_summary.csv", summaries)
    write_csv(out / f"{prefix}_paired_differences.csv", deltas)


def target_balanced_ratios(trajectories):
    keys = ["seed", "p", "readout", "decoder", "target", "trajectory_id"]
    ref = trajectories[trajectories.dephasing == 0][keys+["mae_ev"]].rename(columns={"mae_ev": "coherent_mae_ev"})
    merged = trajectories.merge(ref, on=keys, validate="many_to_one", how="left")
    assert merged.coherent_mae_ev.notna().all() and (merged.coherent_mae_ev > 0).all()
    merged["mae_ratio_to_coherent"] = merged.mae_ev / merged.coherent_mae_ev
    groups = ["dephasing", "K", "p", "readout", "decoder", "seed", "trajectory_id"]
    counts = merged.groupby(groups).target.nunique()
    assert (counts == 4).all()
    return merged.groupby(groups, as_index=False).mae_ratio_to_coherent.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "bace", "hopping"), default="all")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    out = OUT / "analysis" / args.task
    tasks = ("bace", "hopping") if args.task == "all" else (args.task,)
    missing, feature_checks, decoder_checks, precision, metric_frames, trajectory_rows = [], [], [], [], [], []
    for task, seed, strength in itertools.product(tasks, SEEDS, (0., .5, 1.)):
        folder = OUT / f"features/{task}/lambda{strength:.1f}/seed{seed:03d}"
        if not (folder / "complete.json").exists():
            missing.append(str(folder.relative_to(OUT)))
            continue
        data, features, ks, check, stats = verify_features(task, strength, seed)
        feature_checks.append(check)
        precision.extend(stats)
        for ki, k in enumerate(ks):
            for pi, p in enumerate(P_VALUES):
                cell = OUT / f"decoders/{task}/lambda{strength:.1f}/seed{seed:03d}/K{k:03d}/p{p:.1f}"
                if not (cell / "complete.json").exists():
                    missing.append(str(cell.relative_to(OUT)))
                    continue
                marker = json.loads((cell / "complete.json").read_text())
                assert marker["source_fingerprint"] == check["source_fingerprint"]
                assert marker["job"] == [task, strength, seed, k, p]
                x = features[ki, pi]
                maximum_error = 0.
                for readout in (("final",) if task == "bace" else ("final", "tm3")):
                    if task == "bace":
                        rows, trajectories, error = verify_bace(cell, x, data, seed)
                    else:
                        xr = x[:, :120] if readout == "final" else x
                        rows, trajectories, error = verify_hopping(cell / readout, xr, data)
                    expected = dict(task=task, dephasing=strength, seed=seed, K=k, p=p,
                                    q=12 if task == "bace" else 15, t=30 if task == "bace" else 20, readout=readout)
                    for key, value in expected.items():
                        assert (rows[key] == value).all(), (cell, key)
                    maximum_error = max(maximum_error, error)
                    metric_frames.append(rows)
                    trajectory_rows.extend(trajectories)
                decoder_checks.append(dict(task=task, dephasing=strength, seed=seed, K=k, p=p,
                                           max_inference_error=maximum_error, passed=True))
        print(f"[verify] {task} seed={seed} lambda={strength}", flush=True)
    if args.require_complete and missing:
        raise RuntimeError(f"Incomplete full-data retraining: {len(missing)} missing cells; first={missing[0]}")
    assert metric_frames, "No completed decoder outputs to verify"
    metrics = pd.concat(metric_frames, ignore_index=True)
    if not missing:
        assert len(feature_checks) == 18*len(tasks)
        assert len(decoder_checks) == 84*len(tasks)
        assert len(metrics) == sum(84 if task == "bace" else 1344 for task in tasks)
    source_rows = source_audit()
    unchanged = all(row["unchanged"] for row in source_rows)
    source_classification = classify_source_audit(source_rows)
    parity = coherent_parity(metrics)
    write_csv(out / "metrics_all_seeds.csv", metrics)
    write_csv(out / "feature_verification.csv", feature_checks)
    write_csv(out / "feature_precision.csv", precision)
    write_csv(out / "decoder_verification.csv", decoder_checks)
    write_csv(out / "coherent_original_parity.csv", parity)
    write_csv(out / "original_source_hash_audit.csv", source_rows)
    summarize(metrics[metrics.task == "bace"], ["readout", "decoder"], "test_auc", out, "bace_auc")
    if trajectory_rows:
        trajectory = pd.DataFrame(trajectory_rows)
        write_csv(out / "hopping_trajectory_mae.csv", trajectory)
        summarize(trajectory, ["readout", "decoder", "target"], "mae_ev", out, "hopping_mae")
        ratios = target_balanced_ratios(trajectory)
        write_csv(out / "hopping_target_balanced_trajectory_ratios.csv", ratios)
        summarize(ratios, ["readout", "decoder"], "mae_ratio_to_coherent", out, "hopping_ratio")
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), tasks=list(tasks),
                  complete=not missing, verified=source_classification["experiment_source_manifest_unchanged"],
                  numerical_verification_passed=True, **source_classification,
                  missing=missing, n_feature_cells=len(feature_checks),
                  n_decoder_cells=len(decoder_checks), n_metric_rows=len(metrics), original_sources_unchanged=unchanged,
                  interpretation="K controls Monte Carlo feature accuracy, not physical noise. K=0 denotes the exact coherent reference only.",
                  bootstrap="20000 pointwise paired-seed BACE / crossed seed-by-physical-trajectory hopping draws; conditional on fixed fits",
                  code_sha256={name: digest(HERE / name) for name in (
                      "analyze_dephasing_full_k.py", "run_dephasing_full_k.py", "dephasing_fast_backend.py", "test_dephasing_full_k.py")})
    atomic_json(out / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not source_classification["experiment_source_manifest_unchanged"]:
        raise RuntimeError("A non-document source differs from the pre-supplement audit; inspect original_source_hash_audit.csv")
    if source_classification["document_drift_warning"]:
        print("WARNING: manuscript TEX/PDF changed outside this experiment; hashes retained in the audit. Numerical verification passed.", flush=True)


if __name__ == "__main__":
    main()
