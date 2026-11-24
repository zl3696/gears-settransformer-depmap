"""Evaluation invariants and end-to-end runs on synthetic individual cells."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from replogle_data.io import digest, write_json
from replogle_data.workflow import prepare
from replogle_evaluation.artifacts import (check_seal, load_arrays, load_protocol,
                                          read_json, seal, verify_result)
from replogle_evaluation.dataset import EvaluationData
from replogle_evaluation.metrics import (correlation, de_indices, paired_conditions,
                                        score, statistics)
from replogle_evaluation.prediction import load_selected
from replogle_evaluation.summary import summarize
from replogle_evaluation.workflow import evaluate, output_directory
from replogle_training.dataset import SparseExpression
from replogle_training.engine import run
import test_replogle_data as data_tests
import test_replogle_training as training_tests


def scoring_fixture():
    truth = np.array([[1., 2., 4.], [2., 5., 3.], [2., 3., 5.], [4., 7., 5.]])
    return dict(truth=truth, prediction=truth + np.array([1., 3., 2., 4.])[:, None],
                de_indices=np.tile([0, 2], (4, 1)), counts=np.array([1, 100, 200, 3]),
                lines=np.array([0, 0, 1, 1]), targets=np.array(['P', 'Q', 'P', 'Q']),
                control_means=np.zeros((2, 3)), input_means=np.zeros((4, 3)), low_cell_count=np.asarray(30))


class MetricTests(unittest.TestCase):
    def test_original_metrics_condition_identity_and_equal_weights(self):
        aggregate, rows, similarity = score(scoring_fixture())
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(similarity), 2)
        self.assertEqual(aggregate['by_cell_line']['K562']['mse'], 5.)
        self.assertEqual(aggregate['by_cell_line']['RPE1']['mse'], 10.)
        self.assertEqual(aggregate['metrics']['mse'], 7.5)
        self.assertEqual(aggregate['metrics']['mse_de'], 7.5)
        self.assertAlmostEqual(aggregate['metrics']['pearson'], 1.)

    def test_context_response_removes_assigned_control_offsets(self):
        arrays = scoring_fixture()
        controls = np.array([[10., 20., 30.], [1., 2., 3.]])
        effect = np.array([[1., -2., 3.], [-3., 2., 1.]])
        arrays['control_means'] = controls
        arrays['truth'] = controls[arrays['lines']] + effect[[0, 1, 0, 1]]
        arrays['input_means'] = controls[arrays['lines']] + np.array([[7., -4., 2.], [2., 1., 7.], [-2., 9., 3.], [8., 0., 4.]])
        arrays['prediction'] = arrays['input_means'] + effect[[0, 1, 0, 1]]
        aggregate, rows, similarity = score(arrays)
        self.assertAlmostEqual(aggregate['metrics']['response_pearson'], 1.)
        self.assertGreater(aggregate['metrics']['mse'], 0)
        self.assertEqual(aggregate['cross_line']['response_difference_mse'], 0.)
        for row in similarity:
            self.assertAlmostEqual(row['predicted_similarity'], 1.)
            self.assertAlmostEqual(row['similarity_absolute_error'], 0.)
        self.assertTrue(all(r['control_offset_rms'] > 0 for r in rows))

    def test_undefined_correlations_remain_flagged_and_pair_common_valid(self):
        arrays = scoring_fixture()
        arrays['prediction'][0] = 1
        aggregate, first, _ = score(arrays)
        self.assertEqual(first[0]['pearson'], 0)
        self.assertTrue(first[0]['pearson_undefined_to_zero'])
        self.assertIsNone(first[0]['response_pearson'])
        self.assertEqual(aggregate['cross_line']['similarity_valid_targets'], 1)
        _, other, _ = score(scoring_fixture())
        difference, paired = paired_conditions(first, other)
        self.assertIsNone(paired[0]['response_pearson'])
        self.assertEqual(difference['by_cell_line']['K562']['valid_counts']['response_pearson'], 1)
        self.assertIsNone(correlation([1, 1], [1, 2]))
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1]), -1.)

    def test_welch_zero_variance_and_stable_ties(self):
        indices = de_indices(np.array([[0., 2., 3., 0.]]), np.zeros((1, 4)), np.array([1]),
                             np.zeros((2, 4)), np.zeros((2, 4)), np.array([2, 2]), np.array([0]), 3)
        np.testing.assert_array_equal(indices, [[1, 2, 0]])
        self.assertEqual(statistics([1., 3.])['sample_variance'], 2.)
        self.assertIsNone(statistics([None])['mean'])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        sources, _ = data_tests.WorkflowTests().fixture(self.root)
        self.bundle = self.root / 'bundle'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(*sources, self.bundle, self.root / 'config.json', self.root / 'go.csv')

    def tearDown(self):
        self.temporary.cleanup()

    def test_copy_effect_matches_hand_calculation_and_reads_training_only(self):
        for task in ('k562_to_rpe1', 'rpe1_to_k562'):
            data = EvaluationData(self.bundle, task, 1, 'test')
            accessed = []
            original = SparseExpression.read
            def tracked(reader, rows):
                accessed.extend(np.asarray(rows).tolist())
                return original(reader, rows)
            try:
                with patch.object(SparseExpression, 'read', tracked):
                    prediction, references = data.copy_effect()
                self.assertTrue(set(accessed) <= set(data.train['rows']))
                self.assertFalse(set(accessed) & set(data.view['rows']))
                source = 0 if task == 'k562_to_rpe1' else 1
                destination = 1 - source
                controls, _, _ = data.control_statistics()
                target = data.target_ids[0]
                rows = data.train['rows'][(data.train['cell_line_ids'] == source) & (data.train['target_ids'] == target)]
                effect = data.training_expression.read(rows).toarray().astype(np.float64).mean(0) - controls[source]
                np.testing.assert_allclose(prediction[0], controls[destination] + effect)
                np.testing.assert_array_equal(references[0], controls[destination])
                with self.assertRaisesRegex(ValueError, 'held-out test'):
                    data.training_expression.read(data.view['rows'][:1])
            finally:
                data.close()

    def test_copy_effect_missing_source_fails_and_main_is_rejected(self):
        data = EvaluationData(self.bundle, 'k562_to_rpe1', 1, 'val')
        try:
            keep = ~((data.train['cell_line_ids'] == 0) & (data.train['target_ids'] == data.target_ids[0]))
            data.train = {key: value[keep] for key, value in data.train.items()}
            with self.assertRaisesRegex(ValueError, 'Missing source'):
                data.copy_effect()
        finally:
            data.close()
        data = EvaluationData(self.bundle, 'main', 1, 'val')
        try:
            with self.assertRaisesRegex(ValueError, 'inapplicable'):
                data.copy_effect()
        finally:
            data.close()

    def test_cli_baseline_validation_test_verify_regenerate_and_summary(self):
        output = self.root / 'evaluation'
        command = [sys.executable, str(REPO / 'scripts/replogle_test.py')]
        common = ['--output-root', str(output), '--tasks', 'k562_to_rpe1', '--variants', 'Copy-effect', '--split-seeds', '1']
        inputs = ['--data-dir', str(self.bundle)]
        plan = subprocess.run(command + ['plan'] + common + inputs, cwd=self.root, check=True, capture_output=True, text=True)
        self.assertIn('"checkpoint_ready": true', plan.stdout)
        self.assertFalse(output.exists())
        for partition in ('val', 'test'):
            for action in ('run', 'regenerate'):
                subprocess.run(command + [action] + common + inputs + ['--partition', partition],
                               cwd=self.root, check=True, capture_output=True, text=True)
        for action in ('verify', 'summary'):
            subprocess.run(command + [action] + common + ['--partition', 'test'],
                           cwd=self.root, check=True, capture_output=True, text=True)
        with self.assertRaisesRegex(ValueError, 'validation evaluation'):
            evaluate(self.bundle, None, self.root / 'ungated', 'k562_to_rpe1', 1, 'Copy-effect', 'test', load_protocol())

    def test_resealed_de_index_corruption_is_recomputed(self):
        output = self.root / 'de-corruption'
        evaluate(self.bundle, None, output, 'k562_to_rpe1', 1, 'Copy-effect', 'val', load_protocol())
        directory = output_directory(output, 'k562_to_rpe1', 1, 'Copy-effect', 'val')
        arrays = load_arrays(directory / 'scoring_inputs.npz')
        arrays['de_indices'] = arrays['de_indices'][:, ::-1]
        np.savez_compressed(directory / 'scoring_inputs.npz', **arrays)
        seal(directory)
        with self.assertRaisesRegex(ValueError, 'DE ranking'):
            verify_result(directory)

    def test_test_gate_rejects_changed_policy_before_outcome_access(self):
        output = self.root / 'policy-gate'
        protocol = load_protocol()
        evaluate(self.bundle, None, output, 'k562_to_rpe1', 1, 'Copy-effect', 'val', protocol)
        protocol = dict(protocol, batch_size=2)
        with patch.object(EvaluationData, 'outcomes', side_effect=AssertionError('Test outcomes accessed')):
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                evaluate(self.bundle, None, output, 'k562_to_rpe1', 1, 'Copy-effect', 'test', protocol)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = training_tests.TrainingTests()
        cls.fixture.setUp()
        cls.training = cls.fixture.root / 'training'
        cls.output = cls.fixture.root / 'evaluation'
        cls.protocol = load_protocol()
        cls.protocol.update(threads=1, batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            for task in ('main', 'k562_to_rpe1', 'rpe1_to_k562'):
                for variant in ('A', 'B', 'C'):
                    run(cls.fixture.data, cls.training, task, variant, 1, cls.fixture.config_path,
                        cls.fixture.annotation, cls.fixture.prior if variant == 'C' else None, 'cpu')

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def test_all_nine_neural_paths_both_baselines_and_matched_summaries(self):
        for task in ('main', 'k562_to_rpe1', 'rpe1_to_k562'):
            variants = ['A', 'B', 'C'] + ([] if task == 'main' else ['Copy-effect'])
            for partition in ('val', 'test'):
                for variant in variants:
                    result = evaluate(self.fixture.data, self.training, self.output, task, 1, variant, partition, self.protocol)
                    self.assertEqual(result['status'], 'complete')
                    self.assertEqual(result['test_evaluated'], partition == 'test')
                    if variant != 'Copy-effect' and partition == 'val':
                        self.assertTrue(result['training_validation_verified'])
                result = summarize(self.output, task, [1], variants, partition)
                check_seal(Path(result['directory']))
                self.assertFalse(result['complete_five_split'])
                if task == 'main':
                    self.assertTrue((Path(result['directory']) / 'similarity_split1.png').is_file())
            for variant in variants:
                result = evaluate(self.fixture.data, self.training, self.output, task, 1, variant, 'test', self.protocol, regenerate=True)
                self.assertEqual(result['status'], 'passed')

    def test_checkpoint_wrong_identity_minimum_and_hash_are_rejected(self):
        directory = self.training / 'runs/joint_unseen_A_split1_train101'
        data = EvaluationData(self.fixture.data, 'main', 1, 'val')
        result_path = directory / 'results.json'
        original = result_path.read_bytes()
        try:
            for key, value in [('split_seed', 2), ('best_epoch', 999), ('best_checkpoint_sha256', 'changed')]:
                changed = json.loads(original)
                changed[key] = value
                write_json(result_path, changed)
                with self.assertRaises(ValueError):
                    load_selected(directory, data, 'A', self.protocol)
                result_path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'variant identity'):
                load_selected(directory, data, 'B', self.protocol)
        finally:
            result_path.write_bytes(original)
            data.close()

    def test_saved_array_corruption_is_rejected(self):
        output = self.fixture.root / 'tampering'
        evaluate(self.fixture.data, self.training, output, 'main', 1, 'A', 'val', self.protocol)
        directory = output_directory(output, 'main', 1, 'A', 'val')
        with (directory / 'scoring_inputs.npz').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            verify_result(directory)

    def test_validation_adapter_checks_the_recorded_training_metrics(self):
        directory = self.training / 'runs/joint_unseen_A_split1_train101'
        path = directory / 'results.json'
        original = path.read_bytes()
        try:
            changed = json.loads(original)
            changed['selected_validation']['mse'] += 10
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'Validation adapter differs'):
                evaluate(self.fixture.data, self.training, self.fixture.root / 'wrong-validation',
                         'main', 1, 'A', 'val', self.protocol)
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
