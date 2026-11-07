"""Condition leakage, RNG, and frozen-snapshot integrity regression tests."""

from pathlib import Path
import hashlib
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from norman_splits import (DEFAULT_SPLITS, canonical, check_hashes, json_bytes,
                           load_metadata, pair_partitions, verify)


class NormanSplitTests(unittest.TestCase):
    def test_condition_aliases(self):
        self.assertEqual(canonical("B+A"), canonical("A+B"))
        self.assertEqual(canonical("A+ctrl"), canonical("ctrl+A"))
        self.assertEqual(canonical("ctrl"), "ctrl")

    def test_invalid_conditions(self):
        for label in ("", "A+", "A+A", "A+B+C", "ctrl+ctrl"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                canonical(label)

    def test_disjoint_pair_counts(self):
        pairs = [f"A{i:03d}+B{i:03d}" for i in range(128)]
        result = pair_partitions(pairs, 1)
        self.assertEqual({k: len(v) for k, v in result.items()}, {"train": 76, "val": 26, "test": 26})
        self.assertEqual(set(pairs), set().union(*map(set, result.values())))

    def test_rng_and_slice_policy(self):
        pairs = [f"A{i:03d}+B{i:03d}" for i in range(128)]
        expected = pair_partitions(pairs, 3)
        np.random.seed(456)
        np.random.random(100)
        self.assertEqual(expected, pair_partitions(list(reversed(pairs)), 3))
        self.assertNotEqual(expected, pair_partitions(pairs, 4))
        shuffled = np.random.Generator(np.random.PCG64(3)).permutation(sorted(pairs)).tolist()
        self.assertEqual(expected["test"], sorted(shuffled[:26]))
        self.assertEqual(expected["val"], sorted(shuffled[26:52]))

    def test_population_drift(self):
        for pairs in (["A+B"] * 128, [f"A{i}+B" for i in range(127)], [f"A{i}" for i in range(128)]):
            with self.assertRaises(ValueError):
                pair_partitions(pairs, 1)

    def test_singles_and_controls_are_retained(self):
        cells, counts, aliases = load_metadata(DEFAULT_SPLITS)
        self.assertEqual(len(cells), 89357)
        self.assertEqual(counts["ctrl"], 7353)
        self.assertEqual(sum(map(len, aliases.values())), 277)

    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
            output = Path(temporary)
            artifact = output / "assignment.csv"
            artifact.write_text("cell_id,partition\ncell1,test\n")
            hashes = {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}
            (output / "freeze.json").write_bytes(json_bytes({
                "artifact_sha256": hashes,
                "bundle_sha256": hashlib.sha256(json_bytes(hashes)).hexdigest(),
            }))
            artifact.write_text("cell_id,partition\ncell1,train\n")
            with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
                check_hashes(output)

    def test_frozen_snapshot_rebuild(self):
        report = verify(DEFAULT_SPLITS, os.environ.get("TMPDIR"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["hashed_artifacts"], 37)
        self.assertTrue(report["recomputed_split_and_control_files_identical"])


if __name__ == "__main__":
    unittest.main()
