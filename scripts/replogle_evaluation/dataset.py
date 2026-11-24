"""Evaluation-only row access; prediction generation reads training rows only."""

import json

import numpy as np
import pandas as pd
import torch

from replogle_data.io import digest, require
from replogle_training.dataset import Bundle, SparseExpression
from .metrics import de_indices


class EvaluationData:
    def __init__(self, directory, task, seed, partition):
        require(partition in ('val', 'test'), 'Evaluation partition must be val or test')
        # Reuse the frozen-data verifier and exact checkpoint contract. Never
        # expand the training consumer's permitted expression rows.
        bundle = Bundle(directory, task, seed)
        try:
            self.directory = bundle.directory
            self.protocol, self.seed = bundle.protocol, seed
            self.contract = bundle.contract
            self.genes, self.perturbations = bundle.genes, bundle.perturbations
            self.target_nodes = bundle.target_nodes.copy()
            self.train = {key: value.copy() for key, value in bundle.views['train'].items()}
            self.view_dir = bundle.view_dir
        finally:
            bundle.close()
        self.partition = partition
        self.config = json.loads((self.directory / 'config.json').read_text())
        self.target_names = pd.read_csv(self.directory / 'paired_targets.csv', keep_default_na=False).target.to_numpy(str)
        self.target_count = len(self.target_names)
        with np.load(self.view_dir / (partition + '.npz'), allow_pickle=False) as data:
            self.view = {k: data[k].copy() for k in ('rows', 'input_control_rows', 'cell_line_ids', 'target_ids')}
        require(len(self.view['rows']) > 0 and np.all(self.view['target_ids'] >= 0), 'Empty or control-containing evaluation view')
        codes = self.view['cell_line_ids'] * self.target_count + self.view['target_ids']
        self.codes, self.groups = np.unique(codes, return_inverse=True)
        self.lines, self.target_ids = self.codes // self.target_count, self.codes % self.target_count
        self.targets = self.target_names[self.target_ids]
        self.counts = np.bincount(self.groups, minlength=len(self.codes))
        require(len(set(self.view['rows'])) == len(codes), 'Duplicate evaluation rows')
        controls = self.train['rows'][self.train['target_ids'] == -1]
        control_set = set(controls.tolist())
        lookup = {int(row): int(line) for row, line in zip(self.train['rows'], self.train['cell_line_ids'])
                  if int(row) in control_set}
        require(all(lookup.get(int(row)) == int(line) for row, line in
                    zip(self.view['input_control_rows'], self.view['cell_line_ids'])), 'Input is not a same-line training control')
        self.training_expression = SparseExpression(self.directory / 'expression.h5ad', self.train['rows'])
        self._controls = None

    @property
    def identity(self):
        return dict(data=self.contract, partition=self.partition, view_sha256=digest(self.view_dir / (self.partition + '.npz')))

    def close(self):
        self.training_expression.close()

    def inputs(self, positions):
        return self.training_expression.read(self.view['input_control_rows'][positions]).toarray()

    def prediction_batches(self, batch_size, positions=None, device='cpu'):
        if positions is None:
            positions = np.arange(len(self.groups))
        for start in range(0, len(positions), batch_size):
            selected = positions[start:start + batch_size]
            targets = self.view['target_ids'][selected]
            yield dict(positions=selected, groups=self.groups[selected],
                       x=torch.as_tensor(self.inputs(selected), dtype=torch.float32, device=device),
                       nodes=torch.as_tensor(self.target_nodes[targets], dtype=torch.long, device=device),
                       contexts=torch.as_tensor(self.view['cell_line_ids'][selected], dtype=torch.long, device=device))

    def control_statistics(self):
        if self._controls is not None:
            return self._controls
        means, variances, counts = [], [], []
        for line in range(2):
            rows = self.train['rows'][(self.train['cell_line_ids'] == line) & (self.train['target_ids'] == -1)]
            require(len(rows) >= 2, 'Each cell line requires at least two training controls')
            total, squares = np.zeros(len(self.genes)), np.zeros(len(self.genes))
            for start in range(0, len(rows), 512):
                values = self.training_expression.read(rows[start:start + 512]).toarray().astype(np.float64)
                total += values.sum(0)
                squares += (values ** 2).sum(0)
            means.append(total / len(rows))
            variances.append(np.maximum(squares - total ** 2 / len(rows), 0) / (len(rows) - 1))
            counts.append(len(rows))
        self._controls = np.asarray(means), np.asarray(variances), np.asarray(counts)
        return self._controls

    def outcomes(self, number):
        """Decode only the declared evaluation rows, after the run is recorded."""
        total = np.zeros((len(self.codes), len(self.genes)), dtype=np.float64)
        squares = np.zeros_like(total)
        reader = SparseExpression(self.directory / 'expression.h5ad', self.view['rows'])
        try:
            for start in range(0, len(self.groups), 512):
                values = reader.read(self.view['rows'][start:start + 512]).toarray().astype(np.float64)
                groups = self.groups[start:start + len(values)]
                np.add.at(total, groups, values)
                np.add.at(squares, groups, values ** 2)
        finally:
            reader.close()
        means = total / self.counts[:, None]
        variances = np.maximum(squares - total ** 2 / self.counts[:, None], 0) / np.maximum(self.counts[:, None] - 1, 1)
        controls, control_variances, control_counts = self.control_statistics()
        indices = de_indices(means, variances, self.counts, controls, control_variances, control_counts, self.lines, number)
        return dict(truth=means, variances=variances, counts=self.counts.copy(), lines=self.lines.copy(), targets=self.targets.copy(),
                    genes=np.asarray(self.genes), control_means=controls, control_variances=control_variances,
                    control_counts=control_counts, de_indices=indices, low_cell_count=np.asarray(self.config['low_cell_count_flag']))

    def copy_effect(self):
        """Training-only means over actual cells, never replacement draws."""
        require(self.protocol != 'joint_unseen', 'Copy-effect is inapplicable to joint_unseen')
        source = 0 if self.protocol == 'k562_to_rpe1' else 1
        destination = 1 - source
        require(np.all(self.lines == destination), 'Transfer panel contains the source line')
        wanted = np.isin(self.train['target_ids'], self.target_ids) & (self.train['cell_line_ids'] == source)
        rows = self.train['rows'][wanted]
        targets = self.train['target_ids'][wanted]
        total = np.zeros((self.target_count, len(self.genes)), dtype=np.float64)
        count = np.bincount(targets, minlength=self.target_count)
        require(np.all(count[self.target_ids] > 0), 'Missing source training response for a scored target')
        for start in range(0, len(rows), 512):
            values = self.training_expression.read(rows[start:start + 512]).toarray().astype(np.float64)
            np.add.at(total, targets[start:start + len(values)], values)
        controls, _, _ = self.control_statistics()
        effect = total[self.target_ids] / count[self.target_ids, None] - controls[source]
        references = np.repeat(controls[destination][None, :], len(self.codes), axis=0)
        return effect + references, references
