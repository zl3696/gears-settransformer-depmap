"""End-to-end training and leakage checks on tiny individual-cell fixtures."""

import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from gears.depmap import align_features, load_features, save_features
from gears.model import GEARS_Model
from gears.replogle_model import ReplogleModel
from replogle_data.io import digest, write_json
from replogle_data.workflow import prepare
from replogle_training.artifacts import load_checkpoint, preprocessing
from replogle_training.dataset import Bundle
from replogle_training.engine import loss, run, validation
from replogle_training.graphs import build_arrays
import test_replogle_data as data_tests


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = dict(num_genes=5, num_perts=3, hidden_size=8, uncertainty=False,
                         decoder_hidden_size=4, num_go_gnn_layers=1, num_gene_gnn_layers=1,
                         no_perturb=False, device='cpu', pert_agg='sum',
                         G_coexpress=torch.tensor([[0, 1, 2], [1, 2, 3]]), G_coexpress_weight=torch.ones(3),
                         G_go=torch.tensor([[0, 1], [1, 2]]), G_go_weight=torch.ones(2),
                         perturbation_genes=['P0', 'P1', 'P2'])
        self.temporary = tempfile.TemporaryDirectory()
        features = save_features(Path(self.temporary.name) / 'features.npz', ['P0', 'P1'], [[1., 2.], [3., 4.]], {})
        self.aligned = align_features(features, self.args['perturbation_genes'])

    def tearDown(self):
        self.temporary.cleanup()

    def model(self, variant):
        torch.manual_seed(101)
        return ReplogleModel(self.args, variant, self.aligned if variant == 'C' else None).eval()

    def test_initial_outputs_rng_and_shared_parameters_match(self):
        first = self.model('A')
        rng = torch.get_rng_state()
        initial = {key: value.clone() for key, value in first.state_dict().items()}
        x, nodes, contexts = torch.ones(4, 5), torch.tensor([0, 0, 2, -1]), torch.tensor([0, 1, 1, 0])
        expected = first(x, nodes, contexts)
        for variant in ('B', 'C'):
            other = self.model(variant)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in initial.items():
                torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
            torch.testing.assert_close(expected, other(x, nodes, contexts), atol=0, rtol=0)
        self.assertFalse(any('context' in k or 'depmap' in k for k in first.state_dict()))

    def test_baseline_matches_original_single_cell_and_ignores_context(self):
        baseline = self.model('A')
        torch.manual_seed(101)
        original = GEARS_Model(self.args).eval()
        x = torch.ones(1, 5)
        graph = SimpleNamespace(x=x.reshape(-1, 1), pert_idx=[[0]], batch=torch.zeros(5, dtype=torch.long))
        torch.testing.assert_close(baseline(x, torch.tensor([0]), torch.tensor([0])), original(graph), atol=0, rtol=0)
        together = baseline(x.repeat(2, 1), torch.tensor([0, 0]), torch.tensor([0, 1]))
        torch.testing.assert_close(together[0], together[1], atol=0, rtol=0)
        torch.testing.assert_close(together[:1], original(graph), atol=1e-6, rtol=1e-5)

    def test_active_extensions_controls_and_missing_bias(self):
        baseline, contextual, prior = self.model('A'), self.model('B'), self.model('C')
        x = torch.ones(4, 5)
        nodes, contexts = torch.tensor([0, 0, 2, -1]), torch.tensor([0, 1, 1, 1])
        with torch.no_grad():
            contextual.context_embedding.weight[1].fill_(1)
            prior.depmap_projection.weight.fill_(.2)
            prior.depmap_projection.bias.fill_(.7)
        b = contextual(x, nodes, contexts)
        a, c = baseline(x, nodes, contexts), prior(x, nodes, contexts)
        self.assertFalse(torch.allclose(b[0], b[1]))
        self.assertFalse(torch.allclose(c[:2], a[:2]))
        torch.testing.assert_close(c[2:], a[2:], atol=0, rtol=0)
        torch.testing.assert_close(b[-1], a[-1], atol=0, rtol=0)

    def test_loss_is_cell_level_and_groups_contexts_separately(self):
        prediction = torch.zeros(3, 2, requires_grad=True)
        batch = dict(y=torch.tensor([[1., 0.], [3., 0.], [0., 4.]]), groups=np.array([1, 1, 4]))
        value = loss(prediction, batch, torch.ones(6, 2, dtype=torch.bool), torch.zeros(2, 2), 2, 0)
        self.assertAlmostEqual(float(value), 74.25)
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_validation_weights_conditions_then_lines(self):
        class Zero(torch.nn.Module):
            def forward(self, x, nodes, contexts):
                return torch.zeros_like(x)
        class Fake:
            n_groups, target_count = 6, 2
            def batches(self, *args, **kwargs):
                yield dict(x=torch.ones(5, 2), nodes=torch.zeros(5, dtype=torch.long),
                           contexts=torch.tensor([0, 0, 1, 1, 1]), groups=np.array([1, 2, 4, 4, 4]))
        means = np.zeros((6, 2)); means[1] = 1; means[2] = 3; means[4] = 2
        arrays = dict(validation_means=means, validation_counts=np.array([0, 1, 1, 0, 3, 0]),
                      validation_de=np.tile([0, 1], (6, 1)))
        scores = validation(Zero(), Fake(), arrays, 8, 'cpu')
        self.assertAlmostEqual(scores['mse'], 4.5)
        self.assertAlmostEqual(scores['mse_de'], 4.5)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        sources, _ = data_tests.WorkflowTests().fixture(self.root)
        self.data = self.root / 'bundle'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(*sources, self.data, self.root / 'config.json', self.root / 'go.csv')
        self.annotation = self.root / 'annotations.pkl'
        self.annotation.write_bytes(pickle.dumps({f'T{i}': {'shared', f'term{i}'} for i in range(9)}))
        self.config = json.loads((REPO / 'configs/replogle_training.json').read_text())
        self.config.update(epochs=2, hidden_size=8, threads=1, batch_size=8, validation_batch_size=4,
                           go_k=2, coexpression_k=2, coexpression_threshold=0,
                           gene2go_sha256=digest(self.annotation))
        self.config_path = self.root / 'training.json'
        write_json(self.config_path, self.config)
        self.prior = self.root / 'features.npz'
        save_features(self.prior, [f'T{i}' for i in range(8)], np.arange(32).reshape(8, 4) / 7, {})

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_nine_task_variant_paths_and_self_contained_checkpoints(self):
        output = self.root / 'runtime'
        cache_keys = {}
        for task in ('main', 'k562_to_rpe1', 'rpe1_to_k562'):
            for variant in ('A', 'B', 'C'):
                with self.subTest(task=task, variant=variant), contextlib.redirect_stdout(io.StringIO()):
                    result = run(self.data, output, task, variant, 1, self.config_path,
                                 self.annotation, self.prior if variant == 'C' else '/missing/ignored.npz', 'cpu')
                    self.assertFalse(result['test_evaluated'])
                    self.assertTrue(result['checkpoint_reload_verified'])
                    self.assertEqual(result['epochs_completed'], 2)
                    cache_keys.setdefault(task, set()).add(result['preprocessing']['key'])
                    checkpoint = output / 'runs' / result['run_id'] / 'best.pt'
                    model, payload = load_checkpoint(checkpoint)
                    if variant in ('B', 'C'):
                        self.assertGreater(float(model.context_embedding.weight.abs().sum()), 0)
                    if variant == 'C':
                        self.assertGreater(float(model.depmap_projection.weight.abs().sum()), 0)
                        expected = align_features(load_features(self.prior), payload['model_config']['perturbation_genes'])
                        np.testing.assert_array_equal(model.depmap_features.numpy(), expected.values)
                    with self.assertRaises(ValueError):
                        load_checkpoint(checkpoint, expression_genes=payload['model_config']['expression_genes'][::-1])
        self.assertTrue(all(len(keys) == 1 for keys in cache_keys.values()))
        self.assertEqual(len({next(iter(k)) for k in cache_keys.values()}), 3)
        self.prior.unlink()
        for path in (output / 'runs').glob('*C*/best.pt'):
            model, _ = load_checkpoint(path)
            self.assertEqual(model.variant, 'C')

    def test_test_rows_are_blocked_and_graphs_use_training_only(self):
        bundle = Bundle(self.data, 'main', 1)
        try:
            with np.load(bundle.view_dir / 'test.npz') as data:
                with self.assertRaisesRegex(ValueError, 'held-out test'):
                    bundle.expression.read(data['rows'][:1])
            with contextlib.redirect_stdout(io.StringIO()):
                arrays = build_arrays(bundle, self.config, self.annotation)
            rows = bundle.views['train']['rows']
            x = bundle.expression.read(rows).toarray()
            corr = np.abs(np.nan_to_num(np.corrcoef(x.T)))
            for (source, target), weight in zip(arrays['coexpression_edges'].T, arrays['coexpression_weights']):
                self.assertAlmostEqual(float(weight), float(corr[source, target]), places=6)
            self.assertTrue(np.all(arrays['training_counts'][np.unique(bundle.groups('val'))] == 0))
            for line in range(2):
                group = line * (bundle.target_count + 1)
                np.testing.assert_allclose(arrays['control_means'][line], x[bundle.groups('train') == group].mean(0), rtol=1e-6)
            first = bundle.expression.read(rows[:3]).toarray()
            bundle.expression.preload()
            np.testing.assert_array_equal(bundle.expression.read(rows[:3]).toarray(), first)
        finally:
            bundle.close()

    def test_cache_corruption_is_rejected(self):
        bundle = Bundle(self.data, 'main', 1)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _, info = preprocessing(bundle, self.config, self.annotation, self.root / 'cache')
            path = Path(info['path']) / 'arrays.npz'
            with path.open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                preprocessing(bundle, self.config, self.annotation, self.root / 'cache')
        finally:
            bundle.close()

    def test_cli_plan_and_actual_training_from_another_directory(self):
        output = self.root / 'cli-runtime'
        base = [sys.executable, str(REPO / 'scripts/replogle_train.py')]
        common = ['--data-dir', str(self.data), '--output-root', str(output), '--gene2go', str(self.annotation),
                  '--depmap-features', str(self.prior), '--config', str(self.config_path), '--device', 'cpu']
        plan = subprocess.run(base + ['plan', '--split-seeds', '1', '2'] + common,
                              cwd=self.root, check=True, capture_output=True, text=True)
        self.assertEqual(len(plan.stdout.splitlines()), 18)
        self.assertFalse(output.exists())
        subprocess.run(base + ['train', '--task', 'k562_to_rpe1', '--variant', 'C', '--split-seed', '1'] + common,
                       cwd=self.root, check=True, capture_output=True, text=True)
        result = json.loads((output / 'runs/k562_to_rpe1_C_split1_train101/results.json').read_text())
        self.assertFalse(result['test_evaluated'])
        self.assertTrue(result['checkpoint_reload_verified'])


if __name__ == '__main__':
    unittest.main()
