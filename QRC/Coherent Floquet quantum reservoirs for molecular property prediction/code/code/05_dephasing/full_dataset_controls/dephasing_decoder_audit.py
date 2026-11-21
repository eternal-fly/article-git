from concurrent.futures import ProcessPoolExecutor
from common import *
from hamiltonian_qrc import mix_suffix_bank
from readouts import fit_svr

ARCH = ROOT / "QRC-J-mechanism-diagnostics/experiments/18_coherence_interaction_ablation"


def run(job):
    variant, seed = job
    folder = HERE / f"dephasing/legacy8_decoder_audit/{variant}/seed{seed:03d}"
    if (folder / "metrics.csv").exists():
        return
    if variant == "coherent":
        path = MIG / f"cache/qrc_suffix_banks/BACE/canonical_general/Q12/g0.840_W0.500/T30_seed{seed:03d}.npz"
    else:
        path = ARCH / f"features/bace/{variant}/seed{seed:03d}.npz"
    with np.load(path) as z:
        bank = z["suffixes"]
    with np.load(FINAL / "16_dissipative_protocol_migration/cache/tokens/BACE_canonical_general_Q24_T30.npz") as z:
        y = z["labels"]
    split = json.loads((FINAL / f"16_dissipative_protocol_migration/cache/splits/BACE/seed{seed:03d}.json").read_text())
    rows, predictions = [], []
    for p in (0., .1):
        metric, artifact = fit_svr(mix_suffix_bank(bank, p), y, split, seed=seed)
        rows.append(dict(variant=variant, seed=seed, p=p, q=12, t=30, **metric))
        predictions.append(artifact["test_scores"])
        write_csv(folder / f"p{p:.1f}_validation_screen.csv", artifact["screen"])
    write_npz(folder / "predictions.npz", predictions=np.stack(predictions), test_indices=artifact["test_indices"], p=np.asarray([0., .1]))
    write_csv(folder / "metrics.csv", rows)


if __name__ == "__main__":
    jobs = [(v, s) for v in ("coherent", "dephase_half", "event_dephase") for s in SEEDS]
    with ProcessPoolExecutor(6) as pool:
        for i, _ in enumerate(pool.map(run, jobs), 1):
            print(f"decoder audit {i}/{len(jobs)}", flush=True)
    paths = sorted((HERE / "dephasing/legacy8_decoder_audit").glob("*/seed*/metrics.csv"))
    new = pd.concat([pd.read_csv(p) for p in paths])
    old = pd.read_csv(ARCH / "results/bace_metrics_by_seed.csv")
    combined = new.merge(old, on=["variant", "seed", "p", "q", "t"], suffixes=("_new", "_old"), validate="one_to_one")
    combined["changed_hyperparameters"] = (combined.C_new != combined.C_old) | (combined.gamma_scale_new != combined.gamma_scale_old)
    combined["delta_test_auc"] = combined.test_auc_new - combined.test_auc_old
    write_csv(HERE / "dephasing/legacy8_decoder_audit.csv", combined)
