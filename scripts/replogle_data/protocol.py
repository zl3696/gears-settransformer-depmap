"""Condition-level split rules shared by all Replogle model variants."""

import numpy as np

from .io import require

LINES = ('k562', 'rpe1')
PARTITIONS = ('train', 'val', 'test')
PROTOCOLS = ('joint_unseen', 'k562_to_rpe1', 'rpe1_to_k562')


def gene_partition(targets, seed, validation_fraction=.15, test_fraction=.15):
    targets = sorted(targets)
    require(len(targets) == len(set(targets)), 'Duplicate targets')
    require(0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1, 'Invalid split fractions')
    n_val, n_test = int(len(targets) * validation_fraction), int(len(targets) * test_fraction)
    require(min(n_val, n_test, len(targets) - n_val - n_test) > 0, 'Empty target partition')
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(targets).tolist()
    return dict(train=sorted(shuffled[n_test + n_val:]),
                val=sorted(shuffled[n_test:n_test + n_val]), test=sorted(shuffled[:n_test]))


def cell_partitions(labels, context_ids, target_roles, protocol):
    require(protocol in PROTOCOLS, f'Unknown protocol: {protocol}')
    lookup = {g: role for role, genes in target_roles.items() for g in genes}
    require(all(g == 'ctrl' or g in lookup for g in labels), 'Unexpected target')
    result = np.asarray([lookup.get(g, 'train') for g in labels], dtype='<U5')
    if protocol != 'joint_unseen':
        source_id = 0 if protocol == 'k562_to_rpe1' else 1
        result[context_ids == source_id] = 'train'
    require(np.all(result[labels == 'ctrl'] == 'train'), 'Controls must stay in training')
    return result


def fixed_controls(labels, context_ids, seed):
    rows = np.arange(len(labels), dtype=np.int64)
    assigned = rows.copy()
    for line in range(2):
        controls = rows[(labels == 'ctrl') & (context_ids == line)]
        targets = rows[(labels != 'ctrl') & (context_ids == line)]
        require(len(controls) > 0, 'Missing cell-line control bank')
        assigned[targets] = np.random.Generator(np.random.PCG64(seed + line)).choice(
            controls, len(targets), replace=True)
    return assigned


def sampling_weights(labels, context_ids):
    """Choose a line uniformly, then a condition uniformly, then one of its cells."""
    weights = np.zeros(len(labels), dtype=np.float64)
    for line in range(2):
        positions = np.flatnonzero(context_ids == line)
        conditions, inverse, counts = np.unique(labels[positions], return_inverse=True, return_counts=True)
        require(len(conditions) > 0, 'Training partition lacks a cell line')
        weights[positions] = 1. / (2 * len(conditions) * counts[inverse])
    require(np.isclose(weights.sum(), 1), 'Sampling weights do not sum to one')
    return weights
