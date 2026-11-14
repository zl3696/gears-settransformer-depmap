"""Guard the matched test population, control pairing, and metric semantics."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("norman_neural_test", REPOSITORY / "scripts/norman_neural_test.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.cells = [{"source_row": "1", "cell_id": "target", "source_condition": "A+B"}]
        self.schedule = [{"target_source_row": "1", "target_cell_id": "target", "control_source_row": "0",
                          "control_cell_id": "control", "input_index": "0"}]
        self.barcodes = np.array(["control", "target"])
        self.conditions = np.array(["ctrl", "A+B"])

    def resolve(self, schedule=None, assignments=None, conditions=None):
        return runner.assigned_controls(self.cells, schedule or self.schedule, assignments or self.schedule,
                                        self.barcodes, self.conditions if conditions is None else conditions)

    def test_exact_control_and_audit_identity(self):
        controls, audit = self.resolve()
        np.testing.assert_array_equal(controls, [0])
        self.assertEqual(audit[0]["target_cell_id"], "target")

    def test_wrong_source_label_or_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "assigned training control"):
            self.resolve(conditions=np.array(["A", "A+B"]))
        with self.assertRaisesRegex(ValueError, "source-label mismatch"):
            self.resolve(conditions=np.array(["ctrl", "B+C"]))

    def test_changed_schedule_and_duplicate_inputs_are_rejected(self):
        changed = [{**self.schedule[0], "input_index": "1"}]
        with self.assertRaisesRegex(ValueError, "assignments differ"):
            self.resolve(assignments=changed)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.resolve(schedule=self.schedule * 2)

    def test_reader_skips_poisoned_non_control_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.h5"
            with h5py.File(path, "w") as h:
                x = h.create_group("X")
                x.attrs["shape"] = (4, 3)
                x.attrs["encoding-type"] = "csr_matrix"
                x["indptr"] = [0, 2, 3, 5, 6]
                x["indices"] = [0, 2, 0, 1, 2, 1]
                x["data"] = [1., 2., np.nan, 3., 4., np.nan]
                h["counts"] = [np.nan]
                values, entries = runner.read_control_rows(x, [0, 2])
                np.testing.assert_array_equal(values, [[1, 0, 2], [0, 3, 4]])
                self.assertEqual(values.dtype, np.float32)
                self.assertEqual(entries, 4)
                with self.assertRaisesRegex(ValueError, "Invalid control expression"):
                    runner.read_control_rows(x, [1])
                with self.assertRaisesRegex(ValueError, "unique and sorted"):
                    runner.read_control_rows(x, [2, 0])


class StreamingTests(unittest.TestCase):
    def test_cell_means_keep_fixed_batch_positions_and_partial_final_batch(self):
        class PositionModel(torch.nn.Module):
            def forward(self, batch):
                n = len(batch.ptr) - 1
                return batch.x.reshape(n, 24) + torch.arange(n).reshape(-1, 1)

        controls = np.arange(72, dtype=np.float32).reshape(3, 24)
        inputs = np.array([10, 20, 30] * 11 + [10, 20])
        cells = [{"canonical_condition": "A+B" if i % 3 else "B+C"} for i in range(35)]
        pairs = ["A+B", "B+C"]
        result = runner.predict_means(PositionModel(), controls, np.array([10, 20, 30]), inputs,
                                      cells, pairs, ["A", "B", "C"], 32)
        expected = np.stack([controls[[10, 20, 30].index(int(inputs[i]))] + i % 32 for i in range(35)])
        for j, pair in enumerate(pairs):
            mask = np.array([row["canonical_condition"] == pair for row in cells])
            np.testing.assert_array_equal(result[0][j], expected[mask].astype(np.float64).mean(0))
            self.assertEqual(result[1][j], mask.sum())
        with self.assertRaisesRegex(ValueError, "batch size 32"):
            runner.predict_means(PositionModel(), controls, np.array([10, 20, 30]), inputs,
                                 cells, pairs, ["A", "B", "C"], 16)

    def test_published_blob_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact"
            path.write_bytes(b"original")
            artifact = {"size": path.stat().st_size, "git_blob_sha1": runner.blob_sha(path)}
            runner.check_artifact(path, artifact)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Published artifact differs"):
                runner.check_artifact(path, artifact)


class ScoreTests(unittest.TestCase):
    def test_actual_scorer_preserves_equal_pair_weight_and_de_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner.shared.ROOT = Path(tmp)
            scorer = runner.shared.load_scorer(reference=REPOSITORY / "gears/inference.py",
                                               team=REPOSITORY / "gears/inference.py")
            truth = np.arange(48, dtype=float).reshape(2, 24)
            prediction = truth + np.array([[1], [3]])
            indices = np.tile(np.r_[0:10, 14:24], (2, 1))
            inputs = runner.shared.make_scorer_inputs(["A+B", "A+C"], prediction, truth, indices)
            scores, _, _, _ = runner.shared.score_inputs(scorer, inputs)
            self.assertEqual(scores["mse"], 5)
            self.assertEqual(scores["mse_de"], 5)
            runner.independent_scores(inputs, scores)
            inputs = runner.shared.make_scorer_inputs(["A+B", "A+C"], np.zeros_like(truth), np.zeros_like(truth), indices)
            scores, _, flags, _ = runner.shared.score_inputs(scorer, inputs)
            self.assertEqual(scores["pearson_de"], 0)
            self.assertTrue(all(row["pearson_de_undefined_to_zero"] for row in flags.values()))


if __name__ == "__main__":
    unittest.main()
