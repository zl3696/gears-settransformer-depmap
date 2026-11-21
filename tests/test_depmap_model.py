"""Behavioral coverage of optional DepMap across all aggregation modes."""
import pickle

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from gears.depmap import save_features
from test_norman_protocol import GENES, make_gears

MODES = ['sum', 'set_transformer', 'residual_set_transformer']


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / 'features.npz'
    save_features(path, ['B', 'A'], [[-.2, .4, -1., .3], [-1., .2, .5, -.7]],
                  {'source_version': 'synthetic-test'})
    return path


def batch(perts):
    return Batch.from_data_list([Data(x=torch.ones(len(GENES), 1), pert_idx=p) for p in perts])


@pytest.mark.parametrize('mode', MODES)
def test_disabled_is_identical_and_needs_no_features(mode):
    baseline = make_gears(mode)
    expected_rng = torch.get_rng_state()
    disabled = make_gears(mode, use_depmap=False, depmap_path='/does/not/exist.npz')
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert baseline.model.state_dict().keys() == disabled.model.state_dict().keys()
    assert not any('depmap' in k for k in disabled.model.state_dict())
    for name, value in baseline.model.state_dict().items():
        assert torch.equal(value, disabled.model.state_dict()[name])
    baseline.model.eval(); disabled.model.eval()
    x = batch([[0], [0, 1], [-1]])
    torch.testing.assert_close(baseline.model(x), disabled.model(x), atol=0, rtol=0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('training', [False, True])
def test_zero_initialization_preserves_outputs_and_shared_gradients(bundle, mode, training):
    baseline = make_gears(mode).model.train(training)
    expected_rng = torch.get_rng_state()
    enabled = make_gears(mode, use_depmap=True, depmap_path=bundle).model.train(training)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name]), name
    x = batch([[0], [1], [0, 1], [-1]])
    expected, actual = baseline(x), enabled(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().mean().backward(); actual.square().mean().backward()
    for name, param in baseline.named_parameters():
        grad = dict(enabled.named_parameters())[name].grad
        if param.grad is None:
            assert grad is None
        else:
            torch.testing.assert_close(grad, param.grad, atol=1e-6, rtol=1e-5)
    grad = enabled.depmap_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert not enabled.depmap_features.requires_grad
    assert 'depmap_features' not in dict(enabled.named_parameters())


@pytest.mark.parametrize('mode', MODES)
def test_active_correction_respects_missing_controls_and_set_order(bundle, mode):
    baseline = make_gears(mode).model.eval()
    enabled = make_gears(mode, use_depmap=True, depmap_path=bundle).model.eval()
    with torch.no_grad():
        enabled.depmap_projection.weight.fill_(.2)
        enabled.depmap_projection.bias.fill_(.7)
        x = batch([[-1], [2], [0], [0, 1]])
        actual, expected = enabled(x), baseline(x)
        torch.testing.assert_close(actual[:2], expected[:2], atol=0, rtol=0)
        assert not torch.allclose(actual[2:], expected[2:])
        reversed_pairs = enabled(batch([[-1], [2], [0], [1, 0]]))
        torch.testing.assert_close(actual, reversed_pairs)
        separate = torch.cat([enabled(batch([p])) for p in [[-1], [2], [0], [0, 1]]])
        torch.testing.assert_close(actual, separate, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(enabled(batch([[-1], [-1]])), baseline(batch([[-1], [-1]])))


@pytest.mark.parametrize('mode', MODES)
def test_short_training_and_checkpoint_without_source_file(bundle, tmp_path, mode):
    original = make_gears(mode, use_depmap=True, depmap_path=bundle)
    features = original.model.depmap_features.clone()
    original.train(epochs=2, selection_metric='mse_de', run_test=False)
    assert len(original.val_history) == 2 and original.best_epoch is not None
    assert original.best_model.depmap_projection.weight.abs().sum() > 0
    torch.testing.assert_close(original.best_model.depmap_features, features, atol=0, rtol=0)
    original.best_model.eval()
    x = batch([[0], [0, 1], [2], [-1]])
    expected = original.best_model(x).detach()
    checkpoint = tmp_path / 'checkpoint'
    original.save_model(str(checkpoint))
    bundle.unlink()
    restored = make_gears()
    restored.load_pretrained(str(checkpoint))
    restored.model.eval()
    torch.testing.assert_close(restored.model(x), expected, atol=0, rtol=0)
    assert restored.config['use_depmap'] and restored.config['pert_agg'] == mode
    assert restored.config['depmap_metadata']['missing_genes'] == ['C']
    restored.save_model(str(tmp_path / 'resaved'))
    restored.load_pretrained(str(tmp_path / 'resaved'))
    restored.model.eval()
    torch.testing.assert_close(restored.model(x), expected, atol=0, rtol=0)


def test_checkpoint_rejects_incompatible_axes_and_corrupted_features(bundle, tmp_path):
    original = make_gears(use_depmap=True, depmap_path=bundle)
    directory = tmp_path / 'checkpoint'
    original.save_model(str(directory))
    restored = make_gears()
    restored.pert_list = ['B', 'A', 'C']
    with pytest.raises(ValueError, match='order'):
        restored.load_pretrained(str(directory))
    restored = make_gears()
    restored.gene_list = restored.gene_list[::-1]
    with pytest.raises(ValueError, match='order'):
        restored.load_pretrained(str(directory))
    state = torch.load(directory / 'model.pt', weights_only=True)
    state['depmap_features'][0, 0] += 1
    torch.save(state, directory / 'model.pt')
    with pytest.raises(ValueError, match='hash'):
        make_gears().load_pretrained(str(directory))


@pytest.mark.parametrize('mode', MODES)
def test_old_checkpoint_without_depmap_fields_still_loads(tmp_path, mode):
    original = make_gears(mode)
    original.save_model(str(tmp_path / 'old'))
    path = tmp_path / 'old' / 'config.pkl'
    with path.open('rb') as stream:
        config = pickle.load(stream)
    config.pop('use_depmap')
    with path.open('wb') as stream:
        pickle.dump(config, stream)
    restored = make_gears()
    restored.load_pretrained(str(tmp_path / 'old'))
    assert not restored.config['use_depmap']
    for name, value in original.best_model.state_dict().items():
        assert torch.equal(value, restored.best_model.state_dict()[name])


def test_enabled_requires_valid_input(bundle):
    with pytest.raises(ValueError, match='exactly one'):
        make_gears(use_depmap=True)
    with pytest.raises(ValueError, match='exactly one'):
        make_gears(use_depmap=True, depmap_path=bundle, depmap_data=object())
    with pytest.raises(ValueError, match='boolean'):
        make_gears(use_depmap='false')


def test_source_row_permutation_preserves_predictions_with_active_projection(bundle, tmp_path):
    reordered = tmp_path / 'reordered.npz'
    save_features(reordered, ['A', 'B'], [[-1., .2, .5, -.7], [-.2, .4, -1., .3]], {})
    first = make_gears(use_depmap=True, depmap_path=bundle).model.eval()
    second = make_gears(use_depmap=True, depmap_path=reordered).model.eval()
    with torch.no_grad():
        for model in (first, second):
            model.depmap_projection.weight.fill_(.3)
            model.depmap_projection.bias.fill_(.2)
        torch.testing.assert_close(first(batch([[0], [1], [0, 1]])),
                                   second(batch([[0], [1], [0, 1]])), atol=0, rtol=0)
