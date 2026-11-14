"""Check final-test boundaries and configuration consistency on synthetic data."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from additive_baseline import ROOT
from evaluate_test import read_test_targets, select_test_cells
from evaluate_validation import METRICS, read_validation_targets
from summarize_test import SHARED_CONFIG_KEYS, check_compatible


class FinalEvaluationTest(unittest.TestCase):
    def test_test_selection_requires_saved_pairs_counts_and_training_singles(self):
        rows = [{"partition": "test", "canonical_condition": "A+B", "condition_type": "pair"},
                {"partition": "val", "canonical_condition": "C+D", "condition_type": "pair"}]
        manifest = {"pair_conditions": {"test": ["A+B"]}, "target_cell_counts": {"test": 1},
                    "canonical_conditions": {"train": ["A", "B"]}}
        self.assertEqual(select_test_cells(rows, manifest), (["A+B"], rows[:1]))
        manifest["canonical_conditions"]["train"] = ["A"]
        with self.assertRaisesRegex(ValueError, "single is not in training"):
            select_test_cells(rows, manifest)
        manifest["canonical_conditions"]["train"] = ["A", "B"]
        manifest["target_cell_counts"]["test"] = 2
        with self.assertRaisesRegex(ValueError, "cell count mismatch"):
            select_test_cells(rows, manifest)
        manifest["target_cell_counts"]["test"] = 1
        manifest["pair_conditions"]["test"] = ["C+D"]
        with self.assertRaisesRegex(ValueError, "saved condition list"):
            select_test_cells(rows, manifest)

    def test_test_reader_ignores_poisoned_non_test_rows_and_matches_validation_math(self):
        genes = np.array([f"g{i}" for i in range(24)])
        rows = [{"partition": p, "canonical_condition": c, "source_row": str(i),
                 "cell_id": f"cell{i}", "source_condition": c, "source_condition_name": c}
                for i, (p, c) in enumerate((("test", "A+B"), ("val", "C+D"),
                                            ("test", "A+B"), ("train", "E+F")))]
        dense = np.stack([np.arange(1, 25), np.full(24, np.nan),
                          np.arange(1, 25) + 2, np.full(24, np.nan)])
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            with h5py.File(Path(temporary) / "toy.h5", "w") as h:
                x = h.create_group("X")
                x.attrs["shape"] = dense.shape
                x.create_dataset("data", data=dense.ravel())
                x.create_dataset("indices", data=np.tile(np.arange(24), 4))
                x.create_dataset("indptr", data=np.arange(5) * 24)
                h.create_dataset("var/gene_id", data=genes.astype("S"))
                for field, key in (("cell_barcode", "cell_id"), ("condition", "source_condition"),
                                   ("condition_name", "source_condition_name")):
                    h.create_dataset(f"obs/{field}", data=np.array([r[key] for r in rows], dtype="S"))
                h.create_dataset("uns/rank_genes_groups_cov_all/A+B", data=genes[::-1].astype("S"))
                for key in ("C+D", "E+F"):
                    h.create_dataset(f"uns/rank_genes_groups_cov_all/{key}", data=np.array(["poison"], dtype="S"))
                selected = [rows[0], rows[2]]
                result = read_test_targets(h, selected, ["A+B"], genes)
                np.testing.assert_array_equal(result[0][0], np.arange(1, 25) + 1)
                np.testing.assert_array_equal(result[1], [2])
                np.testing.assert_array_equal(result[2][0], np.arange(4, 24))
                self.assertEqual(result[4:], (["A+B"], 48))
                # Identical selected cells under the old validation reader must give identical arrays.
                validation_rows = [{**r, "partition": "val"} for r in selected]
                expected = read_validation_targets(h, validation_rows, ["A+B"], genes)
                for a, b in zip(result[:4], expected[:4]):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(result[4:], expected[4:])
                with self.assertRaisesRegex(ValueError, "Only test"):
                    read_test_targets(h, rows, ["A+B"], genes)

    def test_summary_rejects_mixed_partitions_and_changed_scoring_dependencies(self):
        base = {key: "same" for key in SHARED_CONFIG_KEYS}
        base.update(partition="test", metrics=list(METRICS), test_outcomes_permitted=True)
        configs = [{**base, "split_seed": seed} for seed in range(1, 6)]
        self.assertEqual(check_compatible(configs), base)
        configs[0]["partition"] = "val"
        with self.assertRaisesRegex(ValueError, "final-test configurations"):
            check_compatible(configs)
        configs[0]["partition"] = "test"
        configs[3]["validation_adapter_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "differs across splits"):
            check_compatible(configs)


if __name__ == "__main__":
    unittest.main()
