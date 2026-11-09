"""
Checks for the frozen Norman protocol in the GEARS code: saved control
assignments, axis checks, normalized MSE and test-free training.

Uses tiny synthetic data, so no dataset download is needed:

    python -m pytest tests/test_norman_protocol.py -v
"""
import gzip
import hashlib
import json
import os
import pickle
import tempfile

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pytest
from torch_geometric.data import Data, DataLoader

from gears import GEARS, PertData
from gears.inference import normalized_mse_de

GENES = ['G0', 'G1', 'G2', 'G3', 'G4']
PERTS = np.array(['A', 'B', 'C'])
CONDITIONS = ['ctrl', 'ctrl', 'ctrl', 'A+ctrl', 'A+ctrl', 'A+B', 'A+B']
CONTROL_ROWS = [0, 1, 2, 2, 0, 1, 1]


def make_adata():
    X = sp.csr_matrix(np.arange(len(CONDITIONS) * len(GENES), dtype=np.float32)
                      .reshape(len(CONDITIONS), len(GENES)))
    obs = pd.DataFrame({'condition': CONDITIONS,
                        'condition_name': ['K562_' + c for c in CONDITIONS]},
                       index=['cell{}'.format(i) for i in range(len(CONDITIONS))])
    var = pd.DataFrame({'gene_name': GENES}, index=['ENSG{}'.format(i) for i in range(len(GENES))])
    return anndata.AnnData(X=X, obs=obs, var=var)


def write_frozen_dir(directory):
    cell_ids = ['cell{}'.format(i) for i in range(len(CONDITIONS))]
    tables = {
        'expression_genes.csv': pd.DataFrame({'expression_index': range(len(GENES)),
                                              'gene_id': ['ENSG{}'.format(i) for i in range(len(GENES))],
                                              'gene_name': GENES}),
        'perturbation_nodes.csv': pd.DataFrame({'perturbation_index': range(len(PERTS)),
                                                'gene_name': PERTS, 'expression_index': ''}),
        'cells.csv.gz': pd.DataFrame({'cell_id': cell_ids, 'source_condition': CONDITIONS}),
        'control_assignments.csv.gz': pd.DataFrame({'target_cell_id': cell_ids,
                                                    'control_cell_id': [cell_ids[i] for i in CONTROL_ROWS]}),
    }
    sha256 = {}
    for name, table in tables.items():
        path = os.path.join(directory, name)
        table.to_csv(path, index=False, compression='gzip' if name.endswith('.gz') else None)
        sha256[name] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    with open(os.path.join(directory, 'freeze.json'), 'w') as f:
        json.dump({'artifact_sha256': sha256}, f)


def make_pert_data(adata):
    # skip PertData.__init__, which downloads the GO annotations
    pert_data = PertData.__new__(PertData)
    pert_data.adata = adata
    pert_data.pert_names = PERTS
    pert_data.ctrl_adata = adata[adata.obs['condition'] == 'ctrl']
    pert_data.frozen_split_dir = None
    pert_data.control_rows = None
    return pert_data


# ---------- 1. frozen control assignments and axes ----------

def test_cells_use_saved_controls():
    adata = make_adata()
    pert_data = make_pert_data(adata)
    with tempfile.TemporaryDirectory() as directory:
        write_frozen_dir(directory)
        pert_data.load_frozen_protocol(directory)

    for pert in ['A+ctrl', 'A+B']:
        rows = np.where(adata.obs['condition'] == pert)[0]
        graphs = pert_data.create_cell_graph_dataset(adata, pert)
        assert len(graphs) == len(rows)
        for row, graph in zip(rows, graphs):
            control = adata.X[CONTROL_ROWS[row]].toarray().ravel()
            assert np.array_equal(graph.x.numpy().ravel(), control)
            assert np.array_equal(graph.y.numpy().ravel(), adata.X[row].toarray().ravel())


def test_custom_split_defaults_to_frozen_seed_file():
    pert_data = make_pert_data(make_adata())
    with tempfile.TemporaryDirectory() as directory:
        os.mkdir(os.path.join(directory, 'seed_3'))
        split = {'train': ['ctrl', 'A+ctrl'], 'val': ['A+B'], 'test': []}
        with open(os.path.join(directory, 'seed_3', 'gears_custom_split.pkl'), 'wb') as f:
            import pickle
            pickle.dump(split, f)
        pert_data.frozen_split_dir = directory
        pert_data.prepare_split(split='custom', seed=3)
    assert pert_data.set2conditions == split


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_reordered_genes_are_rejected():
    adata = make_adata()
    adata = adata[:, ::-1].copy()
    pert_data = make_pert_data(adata)
    with tempfile.TemporaryDirectory() as directory:
        write_frozen_dir(directory)
        expect_value_error(lambda: pert_data.load_frozen_protocol(directory))


def test_modified_frozen_file_is_rejected():
    pert_data = make_pert_data(make_adata())
    with tempfile.TemporaryDirectory() as directory:
        write_frozen_dir(directory)
        path = os.path.join(directory, 'control_assignments.csv.gz')
        with gzip.open(path, 'at') as f:
            f.write('extra,row\n')
        expect_value_error(lambda: pert_data.load_frozen_protocol(directory))


# ---------- 2. normalized MSE ----------

def test_normalized_mse_de():
    ctrl = np.array([1., 1., 1., 1.])
    truth_de = np.array([[3., 5.], [3., 5.], [2., 2.]])
    results = {'pert_cat': np.array(['A+B', 'A+B', 'ctrl']),
               'truth_de': truth_de,
               'de_idx': np.array([[0, 2], [0, 2], [-1, -1]])}

    results['pred_de'] = truth_de.copy()
    assert normalized_mse_de(results, ctrl) == 0

    results['pred_de'] = np.ones_like(truth_de)
    assert np.isclose(normalized_mse_de(results, ctrl), 1)


# ---------- 3. training selects on normalized MSE without the test set ----------

class FailingLoader:
    def __iter__(self):
        raise AssertionError('test loader must not be used during training')


def make_gears(pert_agg='sum', **model_options):
    torch.manual_seed(0)
    conditions = ['ctrl', 'A+ctrl', 'B+ctrl', 'A+B']
    X = np.random.RandomState(0).rand(len(conditions) * 4, len(GENES)).astype(np.float32)
    obs = pd.DataFrame({'condition': np.repeat(conditions, 4),
                        'condition_name': np.repeat(['K562_' + c for c in conditions], 4)})
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame({'gene_name': GENES}))
    adata.uns['non_zeros_gene_idx'] = {'K562_' + c: np.arange(len(GENES)) for c in conditions}

    def graphs(condition, pert_idx):
        return [Data(x=torch.rand(len(GENES), 1), y=torch.rand(1, len(GENES)),
                     pert_idx=pert_idx, de_idx=[0, 1], pert=condition) for _ in range(8)]
    train = graphs('A+ctrl', [0]) + graphs('B+ctrl', [1]) + graphs('ctrl', [-1])
    val = graphs('A+B', [0, 1])

    pert_data = type('FakePertData', (), {})()
    pert_data.dataloader = {'train_loader': DataLoader(train, batch_size=4, shuffle=True, drop_last=True),
                            'val_loader': DataLoader(val, batch_size=4),
                            'test_loader': FailingLoader()}
    pert_data.adata = adata
    pert_data.node_map = {g: i for i, g in enumerate(GENES)}
    pert_data.node_map_pert = {p: i for i, p in enumerate(PERTS)}
    pert_data.data_path = pert_data.dataset_name = pert_data.split = None
    pert_data.seed = pert_data.train_gene_set_size = pert_data.subgroup = None
    pert_data.set2conditions = {'train': conditions[:3], 'val': ['A+B']}
    pert_data.gene_names = pd.Series(GENES)
    pert_data.pert_names = PERTS
    pert_data.default_pert_graph = True

    model = GEARS(pert_data, device='cpu')
    model.model_initialize(hidden_size=8, seed=101, pert_agg=pert_agg, **model_options,
                           G_coexpress=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                           G_coexpress_weight=torch.rand(3),
                           G_go=torch.tensor([[0, 1], [1, 2]]),
                           G_go_weight=torch.rand(2))
    return model


@pytest.mark.parametrize('aggregation', ['sum', 'set_transformer', 'residual_set_transformer'])
def test_train_selects_on_normalized_mse_without_test(aggregation):
    model = make_gears(aggregation)
    model.train(epochs=3, selection_metric='normalized_mse_de', run_test=False)

    assert len(model.val_history) == 3
    scores = [h['normalized_mse_de'] for h in model.val_history]
    assert model.best_epoch == int(np.argmin(scores)) + 1


def test_unknown_selection_metric_is_rejected():
    model = make_gears()
    expect_value_error(lambda: model.train(epochs=1, selection_metric='pearson'))


@pytest.mark.parametrize('aggregation', ['sum', 'set_transformer', 'residual_set_transformer'])
@pytest.mark.parametrize('policy', ['shared', 'legacy', 'missing'])
def test_checkpoint_preserves_batching_policy(tmp_path, aggregation, policy):
    chosen = 'legacy' if policy == 'missing' else policy
    original = make_gears(aggregation, coexpression_batching=chosen)
    if aggregation == 'residual_set_transformer':
        # Round-trip an active correction, not just its initial all-zero state.
        with torch.no_grad():
            original.best_model.set_residual_projection.weight.normal_(std=0.1)
    batch = next(iter(original.dataloader['val_loader']))
    # save_model saves best_model, as it does after training.
    original.best_model.eval()
    with torch.no_grad():
        expected = original.best_model(batch)
    checkpoint = tmp_path / 'checkpoint'
    original.save_model(str(checkpoint))
    if policy == 'missing':
        with (checkpoint / 'config.pkl').open('rb') as stream:
            config = pickle.load(stream)
        del config['coexpression_batching']
        with (checkpoint / 'config.pkl').open('wb') as stream:
            pickle.dump(config, stream)
    restored = make_gears()
    restored.load_pretrained(str(checkpoint))
    assert restored.config['coexpression_batching'] == chosen
    restored.best_model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.best_model(batch), expected)
    # Loading and re-saving must retain complete, loadable configuration.
    restored.save_model(str(tmp_path / 'resaved'))
    restored.load_pretrained(str(tmp_path / 'resaved'))
    assert restored.config['coexpression_batching'] == chosen
    assert restored.config['num_genes'] == len(GENES)


def test_new_models_default_to_corrected_batching():
    model = make_gears()
    assert model.config['coexpression_batching'] == 'shared'
