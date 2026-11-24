"""Guarded single-cell access to frozen training/validation views only."""

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import vstack
import torch

from replogle_data.io import digest, read_column, read_csr, require
from replogle_data.workflow import verify

TASKS = {'main': 'joint_unseen', 'joint_unseen': 'joint_unseen',
         'k562_to_rpe1': 'k562_to_rpe1', 'rpe1_to_k562': 'rpe1_to_k562'}


class SparseExpression:
    def __init__(self, path, allowed_rows):
        self.handle = h5py.File(path, 'r')
        self.n_rows, self.n_genes = map(int, self.handle['X'].attrs['shape'])
        self.allowed = np.zeros(self.n_rows, dtype=bool)
        self.allowed[np.asarray(allowed_rows, dtype=np.int64)] = True
        self.matrix = self.lookup = None

    def read(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        require(rows.ndim == 1 and len(rows) > 0 and np.all((rows >= 0) & (rows < self.n_rows)), 'Invalid expression rows')
        require(np.all(self.allowed[rows]), 'Attempt to read a held-out test expression row')
        if self.matrix is not None:
            return self.matrix[self.lookup[rows]]
        ordered, inverse = np.unique(rows, return_inverse=True)
        breaks = np.flatnonzero(np.diff(ordered) != 1) + 1
        runs = np.split(ordered, breaks)
        blocks = [read_csr(self.handle, int(run[0]), int(run[-1]) + 1) for run in runs]
        result = vstack(blocks, format='csr')[inverse]
        require(np.isfinite(result.data).all(), 'Nonfinite expression')
        return result

    def preload(self):
        rows = np.flatnonzero(self.allowed)
        blocks = [self.read(rows[i:i + 2048]) for i in range(0, len(rows), 2048)]
        self.matrix = vstack(blocks, format='csr')
        self.lookup = np.full(self.n_rows, -1, dtype=np.int64)
        self.lookup[rows] = np.arange(len(rows))

    def close(self):
        self.handle.close()


class Bundle:
    def __init__(self, directory, task, seed, preload=False):
        self.directory = Path(directory).resolve()
        require(task in TASKS, 'Unknown Replogle task')
        self.protocol = TASKS[task]
        self.seed = seed
        # Hashing the matrix does not decode held-out expression. The verifier
        # reconstructs split metadata; compare_sources is deliberately false.
        self.verification = verify(self.directory, compare_sources=False)
        self.bundle_sha256 = self.verification['bundle_sha256']
        config = json.loads((self.directory / 'config.json').read_text())
        require(seed in config['split_seeds'], 'Split seed is not in the frozen bundle')
        self.view_dir = self.directory / 'views' / self.protocol / f'seed_{seed}'
        self.manifest = json.loads((self.view_dir / 'manifest.json').read_text())
        self.genes = pd.read_csv(self.directory / 'genes.csv', keep_default_na=False).gene_id.tolist()
        self.perturbations = pd.read_csv(self.directory / 'perturbation_nodes.csv', keep_default_na=False).gene_name.tolist()
        targets = pd.read_csv(self.directory / 'paired_targets.csv', keep_default_na=False).target.tolist()
        lookup = {g: i for i, g in enumerate(self.perturbations)}
        self.target_nodes = np.asarray([lookup[g] for g in targets], dtype=np.int64)
        self.n_groups = 2 * (len(targets) + 1)
        self.target_count = len(targets)
        self.views = {}
        for part in ('train', 'val'):
            with np.load(self.view_dir / (part + '.npz'), allow_pickle=False) as data:
                self.views[part] = {k: data[k].copy() for k in ('rows', 'input_control_rows', 'cell_line_ids', 'target_ids')}
                if part == 'train':
                    self.views[part]['sampling_weights'] = data['sampling_weights'].copy()
        allowed = np.unique(np.concatenate([view['rows'] for view in self.views.values()]))
        self.expression = SparseExpression(self.directory / 'expression.h5ad', allowed)
        with h5py.File(self.directory / 'expression.h5ad', 'r') as handle:
            self.contexts = read_column(handle['obs'], 'cell_line_id').astype(np.int64)
        self.contract = dict(bundle_sha256=self.bundle_sha256, protocol=self.protocol, split_seed=seed,
                             train_view_sha256=digest(self.view_dir / 'train.npz'),
                             val_view_sha256=digest(self.view_dir / 'val.npz'),
                             expression_genes=self.genes, perturbation_genes=self.perturbations,
                             context_names=['K562', 'RPE1'])
        if preload:
            self.expression.preload()

    def groups(self, part):
        view = self.views[part]
        if 'groups' not in view:
            view['groups'] = view['cell_line_ids'] * (self.target_count + 1) + view['target_ids'] + 1
        return view['groups']

    def batches(self, part, batch_size, positions=None, device='cpu'):
        require(part in ('train', 'val'), 'Training entry point does not expose test expression')
        view = self.views[part]
        if positions is None:
            positions = np.arange(len(view['rows']))
        for start in range(0, len(positions), batch_size):
            selected = positions[start:start + batch_size]
            targets = view['target_ids'][selected]
            nodes = np.full(len(selected), -1, dtype=np.int64)
            nodes[targets >= 0] = self.target_nodes[targets[targets >= 0]]
            inputs = self.expression.read(view['input_control_rows'][selected]).toarray()
            outcomes = self.expression.read(view['rows'][selected]).toarray()
            yield dict(x=torch.as_tensor(inputs, dtype=torch.float32, device=device),
                       y=torch.as_tensor(outcomes, dtype=torch.float32, device=device),
                       nodes=torch.as_tensor(nodes, dtype=torch.long, device=device),
                       contexts=torch.as_tensor(view['cell_line_ids'][selected], dtype=torch.long, device=device),
                       groups=self.groups(part)[selected], targets=targets)

    def close(self):
        self.expression.close()
