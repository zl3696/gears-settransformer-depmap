"""Check the statistical baseline and expression-access boundaries on toy data."""

from pathlib import Path
import json
import tempfile
import unittest

import h5py
import numpy as np

from additive_baseline import (
    ROOT, aggregate_selected_rows, check_frozen_bundle, check_manifests,
    digest, json_bytes, predict_pairs,
)


class AdditiveBaselineTest(unittest.TestCase):
    def test_formula_order_invariance_and_unclipped_predictions(self):
        means = {"ctrl": np.array([4., 3., 0.]), "A": np.array([6., 0., 1.]),
                 "B": np.array([5., 1., 2.])}
        actual = predict_pairs(means, ["A+B", "B+A"])
        np.testing.assert_array_equal(actual, [[7., -2., 3.], [7., -2., 3.]])

    def test_inactive_single_and_both_inactive(self):
        control = np.array([1., 2.])
        active = np.array([3., 0.])
        means = {"ctrl": control, "A": active, "B": control, "C": control}
        np.testing.assert_array_equal(predict_pairs(means, ["A+B", "B+C"]), [active, control])

    def test_missing_or_repeated_target_rejected(self):
        means = {"ctrl": np.array([1.]), "A": np.array([2.])}
        for pair in ("A+B", "A+A"):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                predict_pairs(means, [pair])

    def test_only_training_rows_decoded_and_alias_cells_weighted(self):
        # Pair rows contain NaNs: reading them into the fit would fail finiteness.
        dense = np.array([[4., 0.], [6., 0.], [np.nan, 99.],
                          [8., 2.], [5., 1.], [np.nan, -99.]])
        row_indices, columns = np.nonzero(dense)
        indptr = np.r_[0, np.cumsum(np.bincount(row_indices, minlength=len(dense)))]
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            with h5py.File(Path(temporary) / "toy.h5", "w") as h:
                x = h.create_group("X")
                x.attrs["shape"] = dense.shape
                x.attrs["encoding-type"] = "csr_matrix"
                x.create_dataset("data", data=dense[row_indices, columns])
                x.create_dataset("indices", data=columns)
                x.create_dataset("indptr", data=indptr)
                means, counts, entries = aggregate_selected_rows(x, [0, 1, 3, 4], [0, 1, 1, 2], 3)
                np.testing.assert_array_equal(means, [[4., 0.], [7., 1.], [5., 1.]])
                np.testing.assert_array_equal(counts, [1, 2, 1])
                self.assertEqual(entries, 6)
                # Including the forbidden pair demonstrates that the fixture catches leakage.
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    aggregate_selected_rows(x, [0, 1, 2, 3, 4], [0, 1, 1, 1, 2], 3)

    def test_invalid_row_selections_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            with h5py.File(Path(temporary) / "toy.h5", "w") as h:
                x = h.create_group("X")
                x.attrs["shape"] = (3, 2)
                for rows in ([1, 0], [0, 0], [-1], [3], []):
                    with self.subTest(rows=rows), self.assertRaises(ValueError):
                        aggregate_selected_rows(x, rows, [0] * len(rows), 1)

    def test_held_out_single_and_overlapping_partitions_rejected(self):
        conditions = {c: {"condition_type": kind, "cell_count": 1}
                      for c, kind in (("ctrl", "control"), ("A", "single"),
                                      ("B", "single"), ("A+B", "pair"))}
        cells = [{"canonical_condition": c} for c in conditions]
        manifest = {"canonical_conditions": {"train": ["ctrl", "A", "B"], "val": ["A+B"], "test": []},
                    "pair_conditions": {"train": [], "val": ["A+B"], "test": []},
                    "target_cell_counts": {"train": 3, "val": 1, "test": 0}}
        check_manifests([manifest], conditions, cells)
        invalid = json.loads(json.dumps(manifest))
        invalid["canonical_conditions"]["train"].remove("B")
        invalid["canonical_conditions"]["test"].append("B")
        with self.assertRaisesRegex(ValueError, "outside training"):
            check_manifests([invalid], conditions, cells)
        invalid = json.loads(json.dumps(manifest))
        invalid["canonical_conditions"]["test"].append("A+B")
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_manifests([invalid], conditions, cells)

    def test_frozen_artifact_tampering_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            directory = Path(temporary)
            artifact = directory / "conditions.csv"
            artifact.write_text("original\n")
            hashes = {artifact.name: digest(artifact)}
            (directory / "freeze.json").write_bytes(json_bytes({
                "artifact_sha256": hashes,
                "bundle_sha256": hashlib.sha256(json_bytes(hashes)).hexdigest(),
            }))
            check_frozen_bundle(directory)
            artifact.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Frozen split artifact changed"):
                check_frozen_bundle(directory)


if __name__ == "__main__":
    unittest.main()
