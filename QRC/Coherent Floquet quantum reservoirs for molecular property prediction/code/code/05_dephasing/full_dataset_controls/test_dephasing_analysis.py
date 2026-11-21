"""CPU-only tests of selection, suffix weights, and paired statistical units."""
import tempfile
import unittest

from analyze_dephasing_full_k import *


class AnalysisTests(unittest.TestCase):
    def test_only_two_explicit_manuscripts_are_nonblocking(self):
        prefix = "BFS-QRC/PRL/revised_open_dtc_qrc/"
        rows = [dict(path=prefix+name, unchanged=False, expected_sha256="old", actual_sha256="new")
                for name in ("DTC_QRC_PRAPPLIED_open_en.tex", "DTC_QRC_PRAPPLIED_open_en.pdf")]
        result = classify_source_audit(rows)
        self.assertTrue(result["experiment_source_manifest_unchanged"])
        self.assertTrue(result["document_drift_warning"])
        self.assertEqual(len(result["document_changes"]), 2)
        for name in ("core.py", "results.csv", "other.pdf", "other.tex"):
            result = classify_source_audit(rows+[dict(path=prefix+name, unchanged=False)])
            self.assertFalse(result["experiment_source_manifest_unchanged"])
            self.assertEqual(len(result["blocking_source_changes"]), 1)

    def test_bace_validation_priority_and_deterministic_ties(self):
        rows = []
        for c, gamma in itertools.product((.1, 1., 10.), (.25, 1., 4.)):
            rows.append(dict(C=c, gamma_scale=gamma, val_auc=.7, val_loss=.5, val_acc=.8))
        screen = pd.DataFrame(rows)
        winner = selected_bace(screen)
        self.assertEqual((winner.C, winner.gamma_scale), (.1, .25))
        screen.loc[8, "val_auc"] = .71
        screen.loc[8, "val_loss"] = 100.
        self.assertEqual(selected_bace(screen).name, 8)
        screen.loc[8, "val_auc"] = .7
        screen.loc[7, "val_loss"] = .4
        screen.loc[7, "val_acc"] = .1
        self.assertEqual(selected_bace(screen).name, 7)
        screen.loc[6, "val_loss"] = .4
        self.assertEqual(selected_bace(screen).name, 6)

    def test_reset_weights_and_endpoints(self):
        for task, t in (("bace", 30), ("hopping", 20)):
            for p in (0., .1, 1.):
                w, anchor = weights(t, p, task)
                self.assertAlmostEqual(w.sum()+anchor, 1.)
                self.assertTrue((w >= 0).all())
            w, anchor = weights(t, 0., task)
            self.assertEqual(w[0], 1.)
            self.assertEqual(w[1:].sum()+anchor, 0.)
        w, anchor = weights(30, 1., "bace")
        self.assertEqual(anchor, 1.)
        self.assertEqual(w.sum(), 0.)
        w, anchor = weights(20, 1., "hopping")
        self.assertEqual(anchor, 0.)
        self.assertEqual(w[-1], 1.)

    def test_target_balancing_is_within_trajectory(self):
        rows = []
        # Analytic per-target ratios differ from a pooled ratio of mean errors.
        target_errors = (1., 2., 4., 8.)
        ratios = (1., 2., 3., 4.)
        for seed, tid, target in itertools.product(SEEDS, range(60), range(4)):
            baseline = target_errors[target]*(1+tid/10)*(1+seed/100)
            ident = dict(seed=seed, trajectory_id=tid, target=TARGET_NAMES[target],
                         p=.1, readout="final", decoder="rbf_svr")
            rows.append(dict(**ident, dephasing=0., K=0, mae_ev=baseline))
            rows.append(dict(**ident, dephasing=.5, K=128, mae_ev=baseline*ratios[target]))
        result = target_balanced_ratios(pd.DataFrame(rows))
        close(result[result.K == 128].mae_ratio_to_coherent.to_numpy(), np.full(360, 2.5), "target-balanced toy")
        self.assertNotAlmostEqual(sum(a*b for a, b in zip(target_errors, ratios))/sum(target_errors), 2.5)
        values = matrix(result[result.K == 128], "mae_ratio_to_coherent")
        stats = interval(values, "unit-test-target-ratio")
        self.assertEqual(values.shape, (6, 60))
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["ci_low"], 2.5)
        self.assertAlmostEqual(stats["ci_high"], 2.5)

    def test_k_pairing_and_partial_seed_exclusion(self):
        rows = []
        for strength, k, base in ((0., 0, .8), (.5, 8, .59), (.5, 32, .60), (.5, 128, .61)):
            for seed in SEEDS:
                rows.append(dict(dephasing=strength, K=k, p=.1, seed=seed, test_auc=base+seed/1000))
        rows.append(dict(dephasing=1., K=128, p=.1, seed=0, test_auc=.5))
        with tempfile.TemporaryDirectory(prefix="qrc-dephasing-analysis-") as folder:
            out = Path(folder)
            summarize(pd.DataFrame(rows), [], "test_auc", out, "toy")
            summary = pd.read_csv(out / "toy_summary.csv")
            self.assertEqual(len(summary), 4)
            pairs = pd.read_csv(out / "toy_paired_differences.csv")
            self.assertEqual(len(pairs), 5)
            row = pairs[pairs.comparison == "K32_minus_K128"].iloc[0]
            self.assertAlmostEqual(row["mean"], -.01)
            self.assertAlmostEqual(row["ci_low"], -.01)
            self.assertAlmostEqual(row["ci_high"], -.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
