"""Synthetic tests for formula fidelity, target symmetry, and training isolation."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from linear_baseline import condition_embeddings, fit_model, predict, read_training_profiles, select_training_cells


class LinearModelTests(unittest.TestCase):
    def setUp(self):
        self.names = np.array(["A", "B", "C", "D"] + [f"g{i}" for i in range(20)])
        self.conditions = ["ctrl", "A", "B", "C", "A+B", "A+C"]
        self.means = np.random.default_rng(101).normal(size=(len(self.conditions), len(self.names)))

    def test_literal_reference_solver_and_pca_orientation(self):
        model = fit_model(self.means, self.conditions, self.names, k=3)
        y = self.means.T
        g, b, offset = model["gene_embeddings"], model["training_embeddings"], y.mean(1)
        expected = np.linalg.inv(g.T @ g + .1 * np.eye(3)) @ g.T @ (y - offset[:, None]) @ b.T @ np.linalg.inv(b @ b.T + .1 * np.eye(3))
        np.testing.assert_allclose(model["coefficients"], expected, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(g, (y - y.mean(0)) @ model["pca_components"].T, atol=1e-12)
        np.testing.assert_allclose(model["intercept"], offset)
        np.testing.assert_allclose(np.linalg.norm(g, axis=0), model["pca_singular_values"])
        np.testing.assert_allclose(g.mean(0), 0, atol=1e-12)

    def test_control_offset_algebra_and_pair_symmetry(self):
        model = fit_model(self.means, self.conditions, self.names, k=3)
        values = predict(model, ["A", "B", "A+B", "B+A", "ctrl"], self.names)
        np.testing.assert_allclose(values[2], values[3])
        np.testing.assert_allclose(values[2], values[0] + values[1] - model["intercept"])
        np.testing.assert_allclose(values[4], model["intercept"])
        control = self.means[0]
        centered_response = self.means.T - control[:, None]
        np.testing.assert_allclose(control + centered_response.mean(1), model["intercept"])

    def test_axis_reordering_preserves_predictions(self):
        first = fit_model(self.means, self.conditions, self.names, k=3)
        gp, cp = np.arange(len(self.names))[::-1], np.arange(len(self.conditions))[::-1]
        second = fit_model(self.means[cp][:, gp], np.array(self.conditions)[cp], self.names[gp], k=3)
        expected = predict(first, ["A+B", "B+C"], self.names)
        actual = predict(second, ["A+B", "B+C"], self.names[gp])[:, gp]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_training_pair_outcomes_can_change_fit(self):
        original = fit_model(self.means, self.conditions, self.names, k=3)
        changed = self.means.copy()
        changed[-1] += np.arange(len(self.names))
        alternative = fit_model(changed, self.conditions, self.names, k=3)
        self.assertFalse(np.allclose(predict(original, ["B+C"], self.names), predict(alternative, ["B+C"], self.names)))

    def test_reject_missing_duplicate_targets_and_nonfinite_inputs(self):
        embedding = np.ones((len(self.names), 2))
        for condition in ["A+missing", "A+A", "A+B+C"]:
            with self.assertRaises(ValueError):
                condition_embeddings([condition], self.names, embedding)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            condition_embeddings(["A"], ["A", "A"], np.ones((2, 2)))
        means = self.means.copy()
        means[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            fit_model(means, self.conditions, self.names, k=3)

    def test_negative_predictions_are_preserved(self):
        model = {"gene_embeddings": np.eye(2), "coefficients": -np.eye(2), "intercept": np.zeros(2)}
        np.testing.assert_array_equal(predict(model, ["A+B"], ["A", "B"]), [[-1, -1]])


class TrainingIsolationTests(unittest.TestCase):
    def test_only_training_rows_and_pooled_alias_means_are_read(self):
        names = np.array(["A", "B", "C"])
        ids = np.array(["id0", "id1", "id2"])
        data = np.array([[1., 2., 3.], [np.nan] * 3, [3., 4., 5.], [np.nan] * 3, [2., 1., 4.]])
        labels = ["A+ctrl", "A+B", "ctrl+A", "B+C", "ctrl"]
        cells = [{"partition": "train", "source_row": str(i), "cell_id": f"cell{i}",
                  "source_condition": labels[i], "canonical_condition": c}
                 for i, c in [(0, "A"), (2, "A"), (4, "ctrl")]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.h5ad"
            with h5py.File(path, "w") as h:
                x, obs, var = h.create_group("X"), h.create_group("obs"), h.create_group("var")
                sparse = csr_matrix(data)
                x.attrs["shape"] = sparse.shape
                x.attrs["encoding-type"] = "csr_matrix"
                for key in ("data", "indices", "indptr"):
                    x.create_dataset(key, data=getattr(sparse, key))
                dt = h5py.string_dtype()
                for group, key, values in [(var, "gene_id", ids), (var, "gene_name", names),
                                            (obs, "condition", labels), (obs, "cell_barcode", [f"cell{i}" for i in range(5)])]:
                    group.create_dataset(key, data=list(values), dtype=dt)
                # Only these three groups are made available to the reader.
                allowlisted = {key: h[key] for key in ("X", "obs", "var")}
                means, counts, entries = read_training_profiles(allowlisted, cells, ["A", "ctrl"], ids, names)
                np.testing.assert_array_equal(means, [[2, 3, 4], [2, 1, 4]])
                np.testing.assert_array_equal(counts, [2, 1])
                self.assertEqual(entries, 9)
                cells[0]["partition"] = "val"
                with self.assertRaisesRegex(ValueError, "Only training"):
                    read_training_profiles(allowlisted, cells, ["A", "ctrl"], ids, names)

    def test_training_selector_rejects_overlap_or_missing_conditions(self):
        rows = [{"partition": "train", "source_row": "1", "cell_id": "a", "canonical_condition": "A"},
                {"partition": "val", "source_row": "2", "cell_id": "b", "canonical_condition": "A+B"}]
        manifest = {"canonical_conditions": {"train": ["A"], "val": ["A+B"], "test": ["A+C"]}, "target_cell_counts": {"train": 1}}
        names, selected = select_training_cells(rows, manifest)
        self.assertEqual(names, ["A"])
        self.assertEqual(len(selected), 1)
        manifest["canonical_conditions"]["test"] = ["A"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_training_cells(rows, manifest)


if __name__ == "__main__":
    unittest.main()
