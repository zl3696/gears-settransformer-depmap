"""Check five-split reporting and reject incompatible scoring configurations."""

import unittest
import numpy as np

from evaluate_validation import METRICS
from summarize_validation import SHARED_CONFIG_KEYS, check_compatible, metric_statistics


class ValidationSummaryTest(unittest.TestCase):
    def test_sample_variance_uses_denominator_four(self):
        rows = [{"split_seed": seed, **{key: float(seed) for key in METRICS}} for seed in range(1, 6)]
        for row in metric_statistics(rows):
            self.assertEqual(row["mean"], 3)
            self.assertEqual(row["sample_variance"], 2.5)
            self.assertAlmostEqual(row["sample_standard_deviation"], np.sqrt(2.5))

    def test_incomplete_duplicate_and_nonfinite_scores_rejected(self):
        rows = [{"split_seed": seed, **{key: float(seed) for key in METRICS}} for seed in range(1, 6)]
        for invalid in (rows[:4], rows[:4] + [rows[0]]):
            with self.assertRaises(ValueError):
                metric_statistics(invalid)
        rows[0]["mse"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            metric_statistics(rows)

    def test_mixed_scorer_or_prediction_configuration_rejected(self):
        base = {key: "same" for key in SHARED_CONFIG_KEYS}
        base.update(partition="val", metrics=list(METRICS))
        configs = [{**base, "split_seed": seed} for seed in range(1, 6)]
        self.assertEqual(check_compatible(configs), base)
        configs[3]["predictions_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "differs across splits"):
            check_compatible(configs)


if __name__ == "__main__":
    unittest.main()
