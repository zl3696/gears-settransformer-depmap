"""Check reuse of validation targets, scorer behavior, and paired statistics."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_linear import evaluate_saved, independent_metric_check
from evaluate_validation import align_predictions, load_scorer, make_scorer_inputs, score_inputs
from summarize_validation import metric_statistics


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = staticmethod(load_scorer())

    def setUp(self):
        self.pairs = np.array(["A+B", "A+C"])
        self.genes = np.array([f"id{i}" for i in range(24)])
        self.names = np.array([f"g{i}" for i in range(24)])
        self.truth = np.arange(48, dtype=float).reshape(2, 24)
        self.indices = np.tile(np.arange(20), (2, 1))

    def test_perfect_and_constant_vectors_use_actual_scorer(self):
        inputs = make_scorer_inputs(self.pairs, self.truth.copy(), self.truth, self.indices)
        scores, _, _, _ = score_inputs(self.scorer, inputs)
        self.assertEqual(scores["mse"], 0)
        self.assertEqual(scores["mse_de"], 0)
        self.assertAlmostEqual(scores["pearson"], 1)
        self.assertAlmostEqual(scores["pearson_de"], 1)
        self.assertEqual(independent_metric_check(inputs, scores)["status"], "passed")
        inputs = make_scorer_inputs(self.pairs, np.zeros_like(self.truth), np.zeros_like(self.truth), self.indices)
        scores, _, flags, _ = score_inputs(self.scorer, inputs)
        self.assertEqual(scores["pearson"], 0)
        self.assertEqual(scores["pearson_de"], 0)
        self.assertTrue(all(f["pearson_de_undefined_to_zero"] for f in flags.values()))

    def test_reuses_saved_targets_without_opening_source_h5ad(self):
        baseline = make_scorer_inputs(self.pairs, self.truth + 2, self.truth, self.indices)
        expected, _, _, _ = score_inputs(self.scorer, baseline)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "targets.npz"
            np.savez(target, **baseline, de_indices=self.indices, gene_ids=self.genes, gene_names=self.names)
            with patch("h5py.File", side_effect=AssertionError("Raw H5AD access forbidden")):
                result = evaluate_saved(self.scorer, target, self.pairs, self.genes, self.names, self.truth + 1, expected)
            self.assertEqual(result[2]["mse"], 1)
            self.assertEqual(result[6]["mse"], 4)
            with self.assertRaisesRegex(ValueError, "Target condition"):
                evaluate_saved(self.scorer, target, self.pairs[::-1], self.genes, self.names, self.truth, expected)

    def test_de_slice_mismatch_is_rejected(self):
        inputs = make_scorer_inputs(self.pairs, self.truth, self.truth, self.indices)
        inputs["truth_de"] = inputs["truth_de"] + 1
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bad.npz"
            np.savez(target, **inputs, de_indices=self.indices, gene_ids=self.genes, gene_names=self.names)
            with self.assertRaisesRegex(ValueError, "DE slice"):
                evaluate_saved(self.scorer, target, self.pairs, self.genes, self.names, self.truth, {})

    def test_identifier_alignment_and_missing_gene_rejection(self):
        actual = align_predictions(self.pairs[::-1], self.genes[::-1], self.truth[::-1, ::-1], self.pairs, self.genes)
        np.testing.assert_array_equal(actual, self.truth)
        with self.assertRaisesRegex(ValueError, "panel differs"):
            align_predictions(self.pairs, self.genes[:-1], self.truth[:, :-1], self.pairs, self.genes)

    def test_statistics_use_five_split_sample_variance(self):
        rows = [{"split_seed": i, **{k: float(i) for k in ("mse", "mse_de", "pearson", "pearson_de")}} for i in range(1, 6)]
        for row in metric_statistics(rows):
            self.assertEqual(row["mean"], 3)
            self.assertEqual(row["sample_variance"], 2.5)
        with self.assertRaises(ValueError):
            metric_statistics(rows[:1])


if __name__ == "__main__":
    unittest.main()
