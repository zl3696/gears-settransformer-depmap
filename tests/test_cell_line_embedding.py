"""Cell-line conditioning, matched controls, metrics, and portable checkpoints."""
import pickle

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch_geometric.data import Batch, Data

from gears import GEARS, PertData
from gears.cell_line import cell_line_loss
from gears.depmap import save_features
from gears.inference import compute_metrics, normalized_mse_de
from gears.utils import loss_fct, uncertainty_loss_fct
from test_norman_protocol import GENES, PERTS, make_gears

MODES = ['sum', 'set_transformer', 'residual_set_transformer']
LINES = ['K562', 'RPE1']


def data(sparse=False, names=None):
    conditions = ['ctrl', 'A+ctrl', 'B+ctrl', 'A+B', 'C+ctrl']
    rows, values = [], []
    for line_id, line in enumerate(LINES):
        for condition_id, condition in enumerate(conditions):
            for replicate in range(4):
                rows.append((line, condition, line + '_' + condition))
                values.append(np.arange(1, 6) + line_id * 10 + condition_id * .3
                              * np.array([1, -1, 2, -2, 3]) + replicate * .01)
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.asarray(values, dtype=np.float32)) if sparse else np.asarray(values, dtype=np.float32),
        obs=pd.DataFrame(rows, columns=['cell_type', 'condition', 'condition_name']),
        var=pd.DataFrame({'gene_name': GENES}, index=['g' + str(i) for i in range(5)]))
    adata.uns['rank_genes_groups_cov_all'] = {
        line + '_' + condition: list(adata.var_names)[::1 if line == 'K562' else -1]
        for line in LINES for condition in conditions if condition != 'ctrl'}
    p = PertData.__new__(PertData)  # avoid download in constructor
    p.adata, p.pert_names = adata, PERTS.copy()
    p.node_map_pert = {gene: i for i, gene in enumerate(PERTS)}
    p.data_path = p.dataset_name = p.frozen_split_dir = p.control_rows = None
    p.split, p.seed, p.train_gene_set_size, p.subgroup = 'custom', 1, None, None
    p.default_pert_graph = True
    p.set2conditions = {'train': conditions[:3], 'val': ['A+B'], 'test': ['C+ctrl']}
    p.prepare_cell_lines(cell_line_names=names, seed=15)
    p.get_dataloader(batch_size=4)
    return p


def model(mode='sum', enabled=True, features=None, uncertainty=False, names=None, sparse=False):
    p = data(sparse=sparse, names=names)
    m = GEARS(p, device='cpu')
    m.model_initialize(hidden_size=8, seed=101, pert_agg=mode,
                       use_cell_line=enabled, uncertainty=uncertainty,
                       use_depmap=features is not None, depmap_path=features,
                       G_coexpress=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                       G_coexpress_weight=torch.ones(3),
                       G_go=torch.tensor([[0, 1], [1, 2]]), G_go_weight=torch.ones(2))
    return m


def batch(perts, lines):
    return Batch.from_data_list([
        Data(x=torch.ones(5, 1), pert_idx=pert, cell_line=line,
             cell_line_id=torch.tensor([LINES.index(line)]))
        for pert, line in zip(perts, lines)])


@pytest.fixture
def features(tmp_path):
    path = tmp_path / 'features.npz'
    save_features(path, ['A', 'B'], [[1., -.5, .2], [-.1, .8, -.3]], {'source': 'synthetic'})
    return path


@pytest.mark.parametrize('mode', MODES)
def test_disabled_preserves_legacy_and_rng(mode):
    original = make_gears(mode)
    rng = torch.get_rng_state()
    disabled = make_gears(mode, use_cell_line=False)
    assert torch.equal(rng, torch.get_rng_state())
    assert original.model.state_dict().keys() == disabled.model.state_dict().keys()
    assert not any('cell_line' in key for key in disabled.model.state_dict())
    for key, value in original.model.state_dict().items():
        assert torch.equal(value, disabled.model.state_dict()[key])


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('depmap', [False, True])
@pytest.mark.parametrize('training', [False, True])
def test_zero_initialization_outputs_gradients_and_rng(mode, depmap, training, features):
    path = features if depmap else None
    base = model(mode, False, path).model.train(training)
    rng = torch.get_rng_state()
    enabled = model(mode, True, path).model.train(training)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[key])
    x = batch([[0], [0], [1, 0], [-1]], ['K562', 'RPE1', 'RPE1', 'K562'])
    expected, actual = base(x), enabled(x)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for name, parameter in base.named_parameters():
        other = dict(enabled.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(other.grad, parameter.grad)
    gradient = enabled.cell_line_embedding.weight.grad
    assert torch.isfinite(gradient).all() and torch.all(gradient.abs().sum(1) > 0)


@pytest.mark.parametrize('mode', MODES)
def test_active_labels_change_same_perturbation_and_preserve_controls_and_order(mode):
    m = model(mode).model.eval()
    base = model(mode, False).model.eval()
    with torch.no_grad():
        m.cell_line_embedding.weight[0].fill_(.3)
        m.cell_line_embedding.weight[1].fill_(-.5)
        perts, lines = [[0], [0], [0, 1], [-1]], ['K562', 'RPE1', 'K562', 'RPE1']
        actual = m(batch(perts, lines))
        assert not torch.allclose(actual[0], actual[1])
        torch.testing.assert_close(actual[-1], base(batch(perts, lines))[-1], atol=0, rtol=0)
        torch.testing.assert_close(actual, m(batch([[0], [0], [1, 0], [-1]], lines)))
        separate = torch.cat([m(batch([p], [line])) for p, line in zip(perts, lines)])
        torch.testing.assert_close(actual, separate, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(m(batch([[-1], [-1]], LINES)), base(batch([[-1], [-1]], LINES)))


@pytest.mark.parametrize('sparse', [False, True])
def test_graph_controls_labels_de_indices_and_sampling(sparse):
    p, repeated = data(sparse=sparse), data(sparse=sparse)
    np.testing.assert_array_equal(p.cell_line_control_rows, repeated.cell_line_control_rows)
    assert p.cell_line_names == LINES
    for condition, graphs in p.dataset_processed.items():
        for g in graphs:
            assert g.cell_line_id.item() == LINES.index(g.cell_line)
            if condition == 'ctrl':
                torch.testing.assert_close(g.x.flatten(), g.y.flatten())
            else:
                assert g.x.mean() < 4 if g.cell_line == 'K562' else g.x.mean() > 12
                assert g.de_idx == list(range(5))[::1 if g.cell_line == 'K562' else -1]
    assert len(set(p.dataloader['train_loader'].dataset[i].cell_line
                   for i in range(len(p.dataloader['train_loader'].dataset)))) == 2


@pytest.mark.parametrize('uncertainty', [False, True])
def test_loss_uses_separate_controls_and_masks_with_equal_context_weights(uncertainty):
    pred = torch.tensor([[2., 3.], [4., 5.], [11., 8.]], requires_grad=True)
    truth, logvar = torch.tensor([[1., 3.], [2., 4.], [8., 12.]]), torch.ones_like(pred)
    perts, lines = ['A+ctrl', 'B+ctrl', 'A+ctrl'], ['K562', 'K562', 'RPE1']
    controls = {'K562': torch.tensor([1., 2.]), 'RPE1': torch.tensor([10., 10.])}
    filters = {'K562': {'A+ctrl': [0], 'B+ctrl': [1]}, 'RPE1': {'A+ctrl': [1]}}
    expected = 0
    for i, (pert, line) in enumerate(zip(perts, lines)):
        kwargs = dict(ctrl=controls[line], dict_filter=filters[line], direction_lambda=.2)
        expected += (uncertainty_loss_fct(pred[i:i+1], logvar[i:i+1], truth[i:i+1], [pert], reg=.3, **kwargs)
                     if uncertainty else loss_fct(pred[i:i+1], truth[i:i+1], [pert], **kwargs)) / 3
    actual = cell_line_loss(pred, truth, perts, lines, controls, filters, .2,
                            logvar if uncertainty else None, .3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all()


def test_metrics_do_not_cancel_opposite_cell_line_errors():
    truth = np.array([[2., 4.], [12., 14.]])
    pred = truth + np.array([[2., 2.], [-2., -2.]])
    results = dict(pred=pred, truth=truth, pred_de=pred, truth_de=truth,
                   de_idx=np.array([[0, 1], [0, 1]]), pert_cat=np.array(['A+ctrl'] * 2),
                   cell_line=np.array(LINES))
    metrics, contexts = compute_metrics(results)
    assert metrics['mse'] == metrics['mse_de'] == 4
    assert set(contexts) == {('K562', 'A+ctrl'), ('RPE1', 'A+ctrl')}
    assert normalized_mse_de(results, {'K562': np.array([0., 2.]), 'RPE1': np.array([10., 12.])}) == 1
    with pytest.raises(ValueError, match='specific control'):
        normalized_mse_de(results, np.array([5., 7.]))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('depmap', [False, True])
def test_training_and_checkpoint_round_trip(mode, depmap, features, tmp_path):
    m = model(mode, features=features if depmap else None)
    m.train(epochs=2, run_test=False)
    assert torch.all(m.best_model.cell_line_embedding.weight.abs().sum(1) > 0)
    assert not torch.equal(*m.best_model.cell_line_embedding.weight)
    assert all(np.isfinite(row['mse_de']) for row in m.val_history)
    assert np.isfinite(m.test()['mse_de'])
    assert set(m.test_metrics_by_cell_line) == set(LINES)
    x = batch([[0], [0], [0, 1]], ['K562', 'RPE1', 'RPE1'])
    m.best_model.eval()
    expected = m.best_model(x)
    m.save_model(str(tmp_path / 'checkpoint'))
    features.unlink()
    loaded = model(enabled=False)
    loaded.load_pretrained(str(tmp_path / 'checkpoint'))
    loaded.best_model.eval()
    torch.testing.assert_close(loaded.best_model(x), expected, atol=0, rtol=0)
    assert loaded.config['cell_line_metadata']['names'] == LINES
    loaded.save_model(str(tmp_path / 'resaved'))
    loaded.load_pretrained(str(tmp_path / 'resaved'))


def test_uncertainty_training_and_prediction_caches_are_per_line():
    m = model(uncertainty=True, sparse=True)
    m.train(epochs=1, run_test=False)
    first, first_unc = m.predict([['A']], cell_line='K562')
    second, second_unc = m.predict([['A']], cell_line='RPE1')
    assert not np.allclose(first['A'], second['A'])
    np.testing.assert_array_equal(m.predict([['A']], cell_line='K562')[0]['A'], first['A'])
    assert set(m.saved_pred) == {('K562', 'A'), ('RPE1', 'A')}
    assert set(m.saved_logvar_sum) == set(m.saved_pred)
    assert first_unc['A'] > 0 and second_unc['A'] > 0
    with pytest.raises(ValueError, match='Choose cell_line'):
        m.predict([['A']])
    with pytest.raises(ValueError, match='unseen'):
        m.predict([['A']], cell_line='unknown')


@pytest.mark.parametrize('enabled', [False, True])
def test_prediction_controls_match_line_for_both_baseline_and_conditioned(enabled):
    m = model(enabled=enabled)
    m.best_model.no_perturb = True
    assert m.predict([['A']], cell_line='K562')['A'].mean() < 4
    assert m.predict([['A']], cell_line='RPE1')['A'].mean() > 12


def test_invalid_labels_and_checkpoint_vocabularies_fail(tmp_path):
    with pytest.raises(ValueError, match='prepare_cell_lines'):
        make_gears(use_cell_line=True)
    with pytest.raises(ValueError, match='boolean'):
        make_gears(use_cell_line='false')
    m = model()
    x = batch([[0], [1]], LINES)
    x.cell_line_id = torch.tensor([0, 2])
    with pytest.raises(ValueError, match='valid integer'):
        m.model(x)
    x.cell_line_id = torch.tensor([1, 0])
    with pytest.raises(ValueError, match='vocabulary'):
        m.model(x)
    del x.cell_line_id
    with pytest.raises(ValueError, match='valid integer'):
        m.model(x)
    m.save_model(str(tmp_path / 'checkpoint'))
    with pytest.raises(ValueError, match='vocabulary or gene order'):
        model(names=LINES[::-1]).load_pretrained(str(tmp_path / 'checkpoint'))
    changed = model()
    changed.gene_list = changed.gene_list[::-1]
    with pytest.raises(ValueError, match='gene order'):
        changed.load_pretrained(str(tmp_path / 'checkpoint'))
    config_path = tmp_path / 'checkpoint' / 'config.pkl'
    with config_path.open('rb') as f:
        config = pickle.load(f)
    del config['cell_line_metadata']
    with config_path.open('wb') as f:
        pickle.dump(config, f)
    with pytest.raises(ValueError, match='lacks label vocabulary'):
        model().load_pretrained(str(tmp_path / 'checkpoint'))


def test_preparation_rejects_missing_controls_labels_and_de_rankings():
    p = data()
    p.adata.obs.loc[p.adata.obs.index[0], 'cell_type'] = None
    with pytest.raises(ValueError, match='nonempty string'):
        p.prepare_cell_lines()
    p = data()
    p.adata = p.adata[~((p.adata.obs.cell_type == 'RPE1') & (p.adata.obs.condition == 'ctrl'))].copy()
    with pytest.raises(ValueError, match='No controls'):
        p.prepare_cell_lines()
    p = data()
    del p.adata.uns['rank_genes_groups_cov_all']['K562_A+ctrl']
    with pytest.raises(ValueError, match='DE ranking'):
        p.prepare_cell_lines()
    p = data()
    p.control_rows = np.zeros(p.adata.n_obs, dtype=int)
    with pytest.raises(ValueError, match='target cell line'):
        p.prepare_cell_lines()


def test_untrained_cell_line_and_legacy_checkpoint(tmp_path):
    m = model()
    dataset = m.dataloader['train_loader'].dataset
    dataset[:] = [g for g in dataset if g.cell_line == 'K562']
    with pytest.raises(ValueError, match='examples in training'):
        m.train(epochs=1, run_test=False)
    old = make_gears()
    old.save_model(str(tmp_path / 'old'))
    path = tmp_path / 'old' / 'config.pkl'
    with path.open('rb') as f:
        config = pickle.load(f)
    config.pop('use_cell_line')
    with path.open('wb') as f:
        pickle.dump(config, f)
    restored = make_gears()
    restored.load_pretrained(str(tmp_path / 'old'))
    assert not restored.config['use_cell_line']


def test_load_bypasses_stale_unlabeled_cache(tmp_path):
    original = data(sparse=True)
    directory = tmp_path / 'joint'
    directory.mkdir()
    original.adata.write_h5ad(directory / 'perturb_processed.h5ad')
    cache = directory / 'data_pyg'
    cache.mkdir()
    (cache / 'cell_graphs.pkl').write_bytes(b'not a usable graph cache')
    loaded = PertData.__new__(PertData)
    loaded.data_path = str(tmp_path)
    loaded.set_pert_genes = lambda: setattr(loaded, 'pert_names', PERTS.copy())
    loaded.load(data_path=str(directory), cell_line_key='cell_type',
                cell_line_names=LINES, cell_line_seed=15)
    np.testing.assert_array_equal(loaded.cell_line_control_rows, original.cell_line_control_rows)
    assert all(hasattr(g, 'cell_line_id') for graphs in loaded.dataset_processed.values() for g in graphs)
    assert (cache / 'cell_graphs.pkl').read_bytes() == b'not a usable graph cache'
    loaded.dataloader = object()
    loaded.prepare_cell_lines(seed=15)
    assert not hasattr(loaded, 'dataloader')


def test_unconditioned_baseline_trains_and_scores_with_line_matched_data():
    m = model(enabled=False)
    m.train(epochs=1, run_test=False)
    assert not hasattr(m.best_model, 'cell_line_embedding')
    assert np.isfinite(m.test()['normalized_mse_de'])
    assert set(m.test_pert_metrics) == {('K562', 'C+ctrl'), ('RPE1', 'C+ctrl')}
