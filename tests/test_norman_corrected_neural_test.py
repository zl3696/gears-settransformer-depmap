"""Guard checkpoint identity, validation selection, and cohort routing."""

from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))
import norman_corrected_neural_test as runner


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        self.training = {"training": {"epochs": 20}}
        config = self.root / "configs/norman_training.json"
        config.write_text(json.dumps(self.training))
        plan = self.root / "configs/norman_runs.csv"
        plan.write_text("run_id\nexample\n")
        self.protocol = {"split_bundle_sha256": "frozen", "selection_metric": "mse_de"}
        history = [{"mse_de": 2.0} for _ in range(20)]
        history[4] = {"mse_de": 1.0}
        self.record = {"run_id": "run", "model": "gears", "split_seed": 1,
                       "training_seed": 101, "split_bundle_sha256": "frozen",
                       "protocol_sha256": runner.digest(config), "validation_history": history,
                       "selection_metric": "mse_de", "test_evaluated": False,
                       "best_epoch": 5, "selected_validation": history[4],
                       "run_plan_sha256": runner.digest(plan)}

    def check(self, record=None):
        runner.check_training_record(record or self.record, self.training, self.root,
                                     "run", "gears", 1, self.protocol)

    def test_valid_published_checkpoint(self):
        self.check()

    def test_wrong_split_or_previous_test_exposure_is_rejected(self):
        for field, value in (("split_seed", 2), ("test_evaluated", True),
                             ("selection_metric", "normalized_mse_de")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check({**self.record, field: value})

    def test_nonminimum_epoch_or_mismatched_validation_is_rejected(self):
        for change in ({"best_epoch": 1}, {"selected_validation": {"mse_de": 3.0}}):
            with self.assertRaisesRegex(ValueError, "validation minimum"):
                self.check({**self.record, **change})

    def test_changed_training_plan_is_rejected(self):
        (self.root / "configs/norman_runs.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "run plan changed"):
            self.check()


class SourceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = runner.read_json(runner.PROTOCOL_PATH)

    def test_each_model_uses_its_pinned_cohort(self):
        for model in self.protocol["models"]:
            source = runner.source_for_model(self.protocol, model)
            expected = "residual_comparison_v1" if "residual" in model else "coexpression_shared_v1"
            self.assertEqual(source["result_namespace"], expected)
            path = runner.checkpoint_directory(Path("/external"), model + "_split1_train101", source)
            self.assertIn(expected, path.parts)
            self.assertEqual(sum(a["path"].endswith("/model.pt") for a in source["artifacts"]),
                             5 * len(source["models"]))

    def test_missing_or_ambiguous_model_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one source"):
            runner.source_for_model(self.protocol, "unknown")
        changed = copy.deepcopy(self.protocol)
        changed["sources"].append(changed["sources"][0])
        with self.assertRaisesRegex(ValueError, "exactly one source"):
            runner.source_for_model(changed, "gears")


if __name__ == "__main__":
    unittest.main()
