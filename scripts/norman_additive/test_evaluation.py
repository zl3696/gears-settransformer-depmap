"""Verify compatibility with the actual GEARS scorer before real outcome access."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from additive_baseline import ROOT
from evaluate_validation import (
    TEAM_SCORER, align_predictions, de_indices_for_condition, load_scorer,
    make_scorer_inputs, read_validation_targets, score_inputs, select_validation_cells,
)


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = staticmethod(load_scorer())
        cls.team_scorer = staticmethod(load_scorer(TEAM_SCORER, TEAM_SCORER))

    def test_hand_calculated_mse_and_absolute_expression_correlation(self):
        truth = np.arange(24, dtype=float)[None, :] + 1
        pred = truth * 2 + 3
        idx = np.arange(20)[None, :]
        data = make_scorer_inputs(["A+B"], pred, truth, idx)
        metrics, _, _, _ = score_inputs(self.scorer, data)
        self.assertAlmostEqual(metrics["mse"], np.mean((truth + 3) ** 2))
        self.assertAlmostEqual(metrics["mse_de"], np.mean((truth[:, :20] + 3) ** 2))
        self.assertAlmostEqual(metrics["pearson"], 1)
        self.assertAlmostEqual(metrics["pearson_de"], 1)
        self.assertEqual(metrics, score_inputs(self.team_scorer, data)[0])

    def test_perfect_prediction_and_error_outside_de_genes(self):
        truth = np.arange(24, dtype=float)[None, :]
        idx = np.arange(20)[None, :]
        data = make_scorer_inputs(["A+B"], truth.copy(), truth, idx)
        metrics = score_inputs(self.scorer, data)[0]
        self.assertEqual(metrics["mse"], 0)
        self.assertEqual(metrics["mse_de"], 0)
        pred = truth.copy()
        pred[0, 23] += 12
        metrics = score_inputs(self.scorer, make_scorer_inputs(["A+B"], pred, truth, idx))[0]
        self.assertEqual(metrics["mse"], 6)
        self.assertEqual(metrics["mse_de"], 0)

    def test_constant_vectors_keep_gears_zero_conversion_and_flags(self):
        pred, truth = np.ones((1, 24)), np.arange(24, dtype=float)[None, :]
        data = make_scorer_inputs(["A+B"], pred, truth, np.arange(20)[None, :])
        metrics, _, flags, _ = score_inputs(self.scorer, data)
        self.assertEqual(metrics["pearson"], 0)
        self.assertEqual(metrics["pearson_de"], 0)
        self.assertTrue(all(flags["A+B"].values()))

    def test_condition_mean_adapter_matches_unequal_cell_counts(self):
        rng = np.random.default_rng(8)
        labels = np.array(["A+B"] * 2 + ["C+D"] * 7)
        pred, truth = rng.normal(size=(9, 24)), rng.normal(size=(9, 24))
        cell_data = {"pert_cat": labels, "pred": pred, "truth": truth,
                     "pred_de": pred[:, :20], "truth_de": truth[:, :20]}
        expected, per_condition = self.scorer(cell_data)
        means = [np.stack([x[labels == p].mean(0) for p in ("A+B", "C+D")]) for x in (pred, truth)]
        actual = score_inputs(self.scorer, make_scorer_inputs(["A+B", "C+D"], *means,
                                                            np.tile(np.arange(20), (2, 1))))[0]
        for metric in actual:
            self.assertAlmostEqual(actual[metric], expected[metric], places=14)
            self.assertAlmostEqual(actual[metric], np.mean([r[metric] for r in per_condition.values()]), places=14)

    def test_gene_and_condition_reordering_aligns_by_ids(self):
        conditions, genes = np.array(["A+B", "C+D"]), np.array([f"g{i}" for i in range(24)])
        values = np.arange(48).reshape(2, 24)
        reordered = align_predictions(conditions[::-1], genes[::-1], values[::-1, ::-1], conditions, genes)
        np.testing.assert_array_equal(reordered, values)
        for bad_conditions, bad_genes in ((["A+B", "A+B"], genes), (conditions, ["x"] * 24),
                                          (["A+B", "E+F"], genes)):
            with self.assertRaises(ValueError):
                align_predictions(bad_conditions, bad_genes, values, conditions, genes)

    def test_invalid_de_indices_and_nonfinite_inputs_rejected(self):
        truth, pred = np.arange(24, dtype=float)[None, :], np.ones((1, 24))
        for idx in (np.zeros((1, 20), dtype=int), np.arange(19)[None, :], np.arange(20)[None, :] + 9):
            with self.assertRaises(ValueError):
                make_scorer_inputs(["A+B"], pred, truth, idx)
        pred[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            make_scorer_inputs(["A+B"], pred, truth, np.arange(20)[None, :])

    def test_released_de_membership_order_and_missing_gene_failure(self):
        genes = np.array([f"g{i}" for i in range(24)])
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            with h5py.File(Path(temporary) / "toy.h5", "w") as h:
                rankings = h.create_group("rankings")
                rankings.create_dataset("A+B", data=genes[::-1].astype("S"))
                idx, ranked = de_indices_for_condition(rankings, "A+B", genes)
                np.testing.assert_array_equal(idx, np.arange(4, 24))
                np.testing.assert_array_equal(ranked, genes[::-1][:20])
                with self.assertRaisesRegex(ValueError, "do not map"):
                    de_indices_for_condition(rankings, "A+B", np.array(["missing"] * 24))

    def test_validation_only_reader_ignores_poisoned_test_values_and_rankings(self):
        genes = np.array([f"g{i}" for i in range(24)])
        rows = [{"partition": p, "canonical_condition": c, "condition_type": "pair", "source_row": str(i),
                 "cell_id": f"cell{i}", "source_condition": c, "source_condition_name": c}
                for i, (p, c) in enumerate((("val", "A+B"), ("test", "C+D"), ("val", "A+B")))]
        manifest = {"pair_conditions": {"val": ["A+B"]}, "target_cell_counts": {"val": 2},
                    "canonical_conditions": {"train": ["A", "B"]}}
        pairs, selected = select_validation_cells(rows, manifest)
        dense = np.stack([np.arange(1, 25), np.full(24, np.nan), np.arange(1, 25) + 2])
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            with h5py.File(Path(temporary) / "toy.h5", "w") as h:
                x = h.create_group("X")
                x.attrs["shape"] = dense.shape
                x.create_dataset("data", data=dense.ravel())
                x.create_dataset("indices", data=np.tile(np.arange(24), 3))
                x.create_dataset("indptr", data=np.arange(4) * 24)
                h.create_dataset("var/gene_id", data=genes.astype("S"))
                for field, key in (("cell_barcode", "cell_id"), ("condition", "source_condition"),
                                   ("condition_name", "source_condition_name")):
                    h.create_dataset(f"obs/{field}", data=np.array([r[key] for r in rows], dtype="S"))
                h.create_dataset("uns/rank_genes_groups_cov_all/A+B", data=genes.astype("S"))
                # A malformed test ranking would fail if the reader tried to use it.
                h.create_dataset("uns/rank_genes_groups_cov_all/C+D", data=np.array(["poison"], dtype="S"))
                observed, counts, idx, _, names, entries = read_validation_targets(h, selected, pairs, genes)
                np.testing.assert_array_equal(observed[0], np.arange(1, 25) + 1)
                np.testing.assert_array_equal(counts, [2])
                self.assertEqual(names, ["A+B"])
                self.assertEqual(entries, 48)
                self.assertEqual(idx.shape, (1, 20))
                with self.assertRaisesRegex(ValueError, "Only validation"):
                    read_validation_targets(h, rows, pairs, genes)


if __name__ == "__main__":
    unittest.main()
