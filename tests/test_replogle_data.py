"""Check data leakage boundaries, exact sparse preservation, and portable exports."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from replogle_data.io import canonical, digest, write_json
from replogle_data.protocol import (PROTOCOLS, cell_partitions, fixed_controls,
                                   gene_partition, sampling_weights)
from replogle_data.workflow import export_view, prepare, verify


class ProtocolTests(unittest.TestCase):
    def test_target_partitions_and_directional_transfer(self):
        genes = [f'T{i}' for i in range(20)]
        roles = gene_partition(genes, 1)
        self.assertEqual([len(roles[x]) for x in ('train', 'val', 'test')], [14, 3, 3])
        self.assertEqual(roles, gene_partition(genes[::-1], 1))
        self.assertNotEqual(roles, gene_partition(genes, 2))
        labels = np.asarray((['ctrl'] + genes) * 2)
        lines = np.repeat([0, 1], 21)
        joint = cell_partitions(labels, lines, roles, 'joint_unseen')
        self.assertTrue(np.array_equal(joint[:21], joint[21:]))
        for protocol, source in [('k562_to_rpe1', 0), ('rpe1_to_k562', 1)]:
            assigned = cell_partitions(labels, lines, roles, protocol)
            self.assertTrue(np.all(assigned[lines == source] == 'train'))
            self.assertTrue(np.array_equal(assigned[lines != source], joint[lines != source]))

    def test_controls_and_balancing_do_not_mix_cell_lines(self):
        labels = np.asarray(['ctrl', 'ctrl', 'A', 'A', 'A', 'B', 'ctrl', 'A', 'B', 'B'])
        lines = np.asarray([0] * 6 + [1] * 4)
        controls = fixed_controls(labels, lines, 1000)
        self.assertTrue(np.all(labels[controls] == 'ctrl'))
        self.assertTrue(np.array_equal(lines[controls], lines))
        weights = sampling_weights(labels, lines)
        for line in range(2):
            self.assertAlmostEqual(weights[lines == line].sum(), .5)
            for gene in ('ctrl', 'A', 'B'):
                self.assertAlmostEqual(weights[(lines == line) & (labels == gene)].sum(), 1 / 6)

    def test_invalid_protocol_inputs_fail(self):
        for label in ('', 'A+B', 'A+A', 'ctrl+ctrl', ' A+ctrl'):
            with self.assertRaises(ValueError):
                canonical(label)
        self.assertEqual(canonical('ctrl+A'), canonical('A+ctrl'))
        with self.assertRaises(ValueError):
            gene_partition(['A', 'A'], 1)


class WorkflowTests(unittest.TestCase):
    def fixture(self, directory):
        config = dict(schema_version=1, name='synthetic', split_seeds=[1, 2], validation_fraction=.15,
                      test_fraction=.15, control_seed=1000, training_seed=101, low_cell_count_flag=30,
                      protocols=list(PROTOCOLS), source_sha256={}, source_urls={})
        paths, matrices = [], []
        for line, genes in [('k562', ['E1', 'E2', 'E3', 'E4']), ('rpe1', ['E4', 'E2', 'E1', 'E5'])]:
            conditions = ['ctrl'] * 2 + [f'T{i}+ctrl' for i in range(10) for _ in range(3)]
            if line == 'k562':
                conditions += ['ONLY_K562+ctrl']
            x = np.random.default_rng(13).integers(0, 6, (len(conditions), 4)).astype(np.float32)
            path = directory / (line + '.h5ad')
            obs = pd.DataFrame({'condition': conditions}, index=[f'cell{i}' for i in range(len(conditions))])
            var = pd.DataFrame({'gene_id': genes, 'gene_name': ['G' + g for g in genes]}, index=genes)
            if line == 'k562':
                # A duplicate symbol outside the shared Ensembl panel is harmless.
                var.loc['E3', 'gene_name'] = 'GE1'
            data = ad.AnnData(X=csr_matrix(x), obs=obs, var=var)
            data.layers['unread_counts'] = csr_matrix(x * 100)
            data.uns['rank_genes_groups_cov_all'] = {'must_not_copy': 'outcome-derived'}
            data.write_h5ad(path)
            config['source_sha256'][line] = digest(path)
            config['source_urls'][line] = 'synthetic'
            paths.append(path)
            matrices.append(x)
        write_json(directory / 'config.json', config)
        (directory / 'go.csv').write_text('perturbation_index,gene_name\n' + ''.join(f'{i},T{i}\n' for i in range(9)))
        return paths, matrices

    def test_full_prepare_verify_export_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources, matrices = self.fixture(root)
            output = root / 'bundle'
            result = prepare(*sources, output, root / 'config.json', root / 'go.csv')
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['paired_targets'], 9)
            self.assertEqual(result['dataset_views'], 6)
            self.assertEqual(result['expression_genes'], 3)
            matrix = ad.read_h5ad(output / 'expression.h5ad')
            self.assertEqual(list(matrix.var_names), ['E1', 'E2', 'E4'])
            expected = np.concatenate([matrices[0][:29, [0, 1, 3]], matrices[1][:29, [2, 1, 0]]])
            np.testing.assert_array_equal(matrix.X.toarray(), expected)
            self.assertFalse(matrix.uns)
            self.assertFalse(matrix.layers)
            for protocol in PROTOCOLS:
                with np.load(output / 'views' / protocol / 'seed_1/test.npz') as view:
                    self.assertTrue(np.all(matrix.obs.iloc[view['input_control_rows']].target == 'ctrl'))
            exported = root / 'exports/rpe1_test.h5ad'
            info = export_view(output, 'joint_unseen', 1, 'test', exported, 'rpe1')
            self.assertEqual(info['cells'], 3)
            data = ad.read_h5ad(exported)
            self.assertTrue(np.all(data.obs.cell_line == 'RPE1'))
            self.assertTrue(np.all(data.obs.target != 'ctrl'))
            command = [sys.executable, str(REPO / 'scripts/replogle_data.py'), 'verify', '--data-dir', str(output)]
            completed = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
            self.assertIn('passed', completed.stdout)
            with self.assertRaises(ValueError):
                prepare(*sources, output, root / 'config.json', root / 'go.csv')
            (output / 'paired_targets.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify(output)


if __name__ == '__main__':
    unittest.main()
