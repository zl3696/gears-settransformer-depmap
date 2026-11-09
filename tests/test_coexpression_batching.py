"""Regression checks for shared gene-graph propagation (no real data needed)."""
from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch

from test_set_transformer import NUM_GENES, make_args, make_batch, model_module


def corrected_model(aggregation, layers=1):
    torch.manual_seed(17)
    args = make_args(aggregation)
    args.update(coexpression_batching='shared', num_gene_gnn_layers=layers)
    return model_module.GEARS_Model(args).eval()


@pytest.mark.parametrize('aggregation', ['sum', 'set_transformer', 'residual_set_transformer'])
@pytest.mark.parametrize('layers', [0, 1, 2])
def test_predictions_ignore_batch_position_and_size(aggregation, layers):
    model = corrected_model(aggregation, layers)
    cells = make_batch([[-1], [2], [1, 3], [4], [0, 5]]).to_data_list()
    order = [4, 2, 0, 3, 1]
    with torch.no_grad():
        together = model(Batch.from_data_list(cells))
        reordered = model(Batch.from_data_list([cells[i] for i in order]))
        separate = torch.cat([model(Batch.from_data_list([cell])) for cell in cells])
    torch.testing.assert_close(reordered, together[order], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(separate, together, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('aggregation', ['sum', 'set_transformer', 'residual_set_transformer'])
@pytest.mark.parametrize('train_mode', [False, True])
def test_matches_explicit_disconnected_graphs_and_gradients(aggregation, train_mode):
    # Independent reference: repeat the embeddings AND offset all edges for
    # each cell. The efficient shared-graph version must be equivalent.
    shared = corrected_model(aggregation, layers=2).train(train_mode)
    reference = deepcopy(shared)
    reference.coexpression_batching = 'legacy'
    batch = make_batch([[-1], [2], [1, 3], [0, 5]])
    reference.G_coexpress = torch.cat([
        shared.G_coexpress + i * NUM_GENES for i in range(4)], dim=1)
    reference.G_coexpress_weight = shared.G_coexpress_weight.repeat(4)
    actual = shared(batch)
    expected = reference(batch)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for name in ['emb_pos.weight', 'layers_emb_pos.0.lin.weight',
                 'layers_emb_pos.1.lin.weight']:
        gradient = dict(shared.named_parameters())[name].grad
        reference_gradient = dict(reference.named_parameters())[name].grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, reference_gradient, atol=2e-5, rtol=2e-3)


def test_legacy_reproduces_the_position_bug():
    model = corrected_model('sum')
    model.coexpression_batching = 'legacy'
    cells = make_batch([[1, 3], [2]]).to_data_list()
    with torch.no_grad():
        original = model(Batch.from_data_list(cells))
        reversed_output = model(Batch.from_data_list(cells[::-1])).flip(0)
    assert not torch.allclose(original, reversed_output, atol=1e-5)


def test_invalid_batching_policy_is_rejected():
    args = make_args('sum')
    args['coexpression_batching'] = 'unknown'
    with pytest.raises(ValueError, match='coexpression_batching'):
        model_module.GEARS_Model(args)
