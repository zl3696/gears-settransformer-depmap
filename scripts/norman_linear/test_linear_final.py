"""Check final-test boundaries and publication without changing sealed fits."""

from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_linear_test import validate_test_contract, publish_test, evaluate_saved, METRICS
from evaluate_validation import load_scorer, make_scorer_inputs, score_inputs
from summarize_linear_test import check_compatible


class FinalContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"split_seed": 1, "pair_conditions": {"test": ["A+B"]}}
        self.config = {"partition": "test", "split_seed": 1, "metrics": list(METRICS),
                       "test_outcomes_permitted": True, "source_sha256": "source",
                       "split_bundle_sha256": "bundle", "conditions": ["A+B"]}
        self.summary = {"partition": "test", "split_seed": 1, "status": "final_test_evaluated"}

    def test_validation_bundle_cannot_be_used_as_final_test(self):
        validate_test_contract(self.config, self.summary, self.manifest, 1, "source", "bundle")
        for changes in ({"partition": "val"}, {"test_outcomes_permitted": False}):
            config = {**self.config, **changes}
            with self.assertRaisesRegex(ValueError, "authorized final-test"):
                validate_test_contract(config, self.summary, self.manifest, 1, "source", "bundle")

    def test_wrong_seed_source_or_pair_population_is_rejected(self):
        for changes in ({"split_seed": 2}, {"source_sha256": "different"},
                        {"split_bundle_sha256": "different"}, {"conditions": ["A+C"]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_test_contract({**self.config, **changes}, self.summary, self.manifest, 1, "source", "bundle")

    def test_summary_rejects_refit_tuning_and_mixed_contracts(self):
        keys = ("model", "source_sha256", "split_bundle_sha256", "runner_sha256",
                "shared_linear_evaluation_helper_sha256", "shared_evaluation_helper_sha256", "test_selection_helper_sha256",
                "original_scorer_sha256", "team_scorer_sha256", "de_policy", "undefined_pearson_policy", "condition_weighting",
                "mean_dtype", "protocol_scope")
        configs = [{**{key: "fixed" for key in keys}, "split_seed": seed, "partition": "test", "metrics": list(METRICS),
                    "test_outcomes_permitted": True, "refit_performed": False, "hyperparameter_tuning": False}
                   for seed in range(1, 6)]
        self.assertEqual(check_compatible(configs)["partition"], "test")
        for changes in ({"refit_performed": True}, {"hyperparameter_tuning": True}, {"model": "different"}, {"partition": "val"}):
            bad = deepcopy(configs)
            bad[3].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_compatible(bad)

    def test_final_publication_uses_test_labels_and_exact_saved_targets(self):
        scorer = load_scorer()
        pairs = np.array(["A+B", "A+C"])
        genes, names = np.array([f"id{i}" for i in range(24)]), np.array([f"g{i}" for i in range(24)])
        truth = np.stack([np.arange(24, dtype=float), np.zeros(24)])
        indices = np.tile(np.arange(20), (2, 1))
        additive = make_scorer_inputs(pairs, truth + .25, truth, indices)
        additive_scores, _, _, _ = score_inputs(scorer, additive)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets, stage = root / "targets", root / "result"
            targets.mkdir()
            stage.mkdir()
            np.savez(targets / "scoring_inputs.npz", **additive, de_indices=indices, gene_ids=genes, gene_names=names)
            for filename in ("scoring_genes.csv", "evaluation_cells.csv.gz"):
                (targets / filename).write_bytes(b"synthetic target artifact")
            config = {"model": "fitted_linear_pair_adaptation", "split_seed": 1,
                      "prior_validation_exposed_test_conditions": []}
            (stage / "evaluation_config.json").write_text(json.dumps(config))
            cells = [{"canonical_condition": str(pair)} for pair in pairs]
            with patch("h5py.File", side_effect=AssertionError("Raw H5AD access forbidden")), \
                 patch("linear_baseline.fit_model", side_effect=AssertionError("Refitting forbidden")):
                result = evaluate_saved(scorer, targets / "scoring_inputs.npz", pairs, genes, names, truth + .5, additive_scores)
                summary = publish_test(stage, config, cells, genes, names, targets, result)
            self.assertEqual(summary["partition"], "test")
            self.assertEqual(summary["status"], "final_test_evaluated")
            self.assertFalse(summary["refit_performed"])
            self.assertEqual(summary["metrics"]["mse"], .25)
            self.assertEqual(summary["additive_metrics"]["mse"], .0625)
            self.assertEqual(summary["undefined_to_zero_counts"]["pearson_de_undefined_to_zero"], 1)
            self.assertEqual((stage / "scoring_genes.csv").read_bytes(), (targets / "scoring_genes.csv").read_bytes())
            with np.load(stage / "scoring_inputs.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["truth"], truth)


if __name__ == "__main__":
    unittest.main()
