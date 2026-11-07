"""
Sanity checks for the Set Transformer perturbation aggregation.

Uses a tiny synthetic graph, so no dataset is needed. Only torch and
torch_geometric are required:

    python -m pytest tests/test_set_transformer.py -v
or
    python tests/test_set_transformer.py
"""
import importlib.util
import os
import subprocess
import tempfile

import torch
from torch_geometric.data import Data, Batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# last commit before the Set Transformer changes, used as the original GEARS reference
ORIGINAL_COMMIT = 'f374e43'

NUM_GENES = 10
NUM_PERTS = 6
HIDDEN_SIZE = 8


def load_module(name, path):
    # load model.py directly so that gears/__init__.py (scanpy etc.) is not imported
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_module('model', os.path.join(REPO_ROOT, 'gears', 'model.py'))


def load_original_module():
    source = subprocess.run(['git', 'show', ORIGINAL_COMMIT + ':gears/model.py'],
                            cwd=REPO_ROOT, capture_output=True, text=True,
                            check=True).stdout
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(source)
    try:
        return load_module('original_model', f.name)
    finally:
        os.remove(f.name)


def random_graph(num_nodes, num_edges, generator):
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=generator)
    edge_weight = torch.rand(num_edges, generator=generator)
    return edge_index, edge_weight


def make_args(pert_agg):
    generator = torch.Generator().manual_seed(0)
    G_coexpress, G_coexpress_weight = random_graph(NUM_GENES, 30, generator)
    G_go, G_go_weight = random_graph(NUM_PERTS, 15, generator)
    return {'hidden_size': HIDDEN_SIZE,
            'num_go_gnn_layers': 1,
            'num_gene_gnn_layers': 1,
            'decoder_hidden_size': 16,
            'uncertainty': False,
            'G_go': G_go,
            'G_go_weight': G_go_weight,
            'G_coexpress': G_coexpress,
            'G_coexpress_weight': G_coexpress_weight,
            'device': 'cpu',
            'num_genes': NUM_GENES,
            'num_perts': NUM_PERTS,
            'no_perturb': False,
            'pert_agg': pert_agg}


def make_model(pert_agg, seed=0):
    torch.manual_seed(seed)
    model = model_module.GEARS_Model(make_args(pert_agg))
    # eval mode: BatchNorm uses running stats, so each sample is computed independently
    model.eval()
    return model


def make_batch(pert_idx_list, seed=0):
    """
    :param pert_idx_list: one list of perturbation indices per sample, [-1] for control
    """
    generator = torch.Generator().manual_seed(seed)
    data_list = []
    for pert_idx in pert_idx_list:
        x = torch.rand(NUM_GENES, 1, generator=generator)
        data_list.append(Data(x=x, y=x.T, pert_idx=pert_idx))
    return Batch.from_data_list(data_list)


# ---------- 1. order invariance ----------

def test_set_modules_are_order_invariant():
    torch.manual_seed(0)
    set_attention = model_module.SetAttentionBlock(HIDDEN_SIZE, HIDDEN_SIZE, num_heads=4).eval()
    set_pooling = model_module.PoolingByMultiheadAttention(HIDDEN_SIZE, num_heads=4).eval()

    elements = torch.randn(5, 2, HIDDEN_SIZE)
    mask = torch.ones(5, 2, dtype=torch.bool)
    with torch.no_grad():
        output = set_pooling(set_attention(elements, mask), mask)
        output_reversed = set_pooling(set_attention(elements.flip(1), mask), mask)

    assert torch.allclose(output, output_reversed, atol=1e-6)


def test_model_is_order_invariant():
    model = make_model('set_transformer')
    with torch.no_grad():
        output = model(make_batch([[1, 3], [2], [-1]]))
        output_swapped = model(make_batch([[3, 1], [2], [-1]]))

    assert torch.allclose(output, output_swapped, atol=1e-6)


# ---------- 2. shape and mask ----------

def test_mixed_batch_shape():
    model = make_model('set_transformer')
    with torch.no_grad():
        output = model(make_batch([[-1], [2], [1, 3], [4], [0, 5]]))

    assert output.shape == (5, NUM_GENES)
    assert torch.isfinite(output).all()


def test_padding_values_are_ignored():
    torch.manual_seed(0)
    set_attention = model_module.SetAttentionBlock(HIDDEN_SIZE, HIDDEN_SIZE, num_heads=4).eval()
    set_pooling = model_module.PoolingByMultiheadAttention(HIDDEN_SIZE, num_heads=4).eval()

    single = torch.randn(1, 1, HIDDEN_SIZE)
    padded_zeros = torch.cat([single, torch.zeros(1, 1, HIDDEN_SIZE)], dim=1)
    padded_noise = torch.cat([single, 100 * torch.randn(1, 1, HIDDEN_SIZE)], dim=1)
    mask = torch.tensor([[True, False]])

    with torch.no_grad():
        output_unpadded = set_pooling(set_attention(single))
        output_zeros = set_pooling(set_attention(padded_zeros, mask), mask)
        output_noise = set_pooling(set_attention(padded_noise, mask), mask)

    assert torch.allclose(output_unpadded, output_zeros, atol=1e-6)
    assert torch.allclose(output_unpadded, output_noise, atol=1e-6)


def test_single_gene_sample_unaffected_by_padding():
    # sample 0 is a single-gene perturbation; in the mixed batch it gets padded to length 2
    model = make_model('set_transformer')
    with torch.no_grad():
        output_mixed = model(make_batch([[2], [1, 3], [-1]]))
        output_singles = model(make_batch([[2], [1], [-1]]))

    assert torch.allclose(output_mixed[0], output_singles[0], atol=1e-6)


# ---------- 3. pert_agg='sum' matches original GEARS ----------

def test_sum_matches_original_gears():
    original_module = load_original_module()
    args = make_args('sum')

    torch.manual_seed(0)
    original_model = original_module.GEARS_Model(args)
    torch.manual_seed(0)
    model = model_module.GEARS_Model(args)

    assert original_model.state_dict().keys() == model.state_dict().keys()
    model.load_state_dict(original_model.state_dict())

    batch = make_batch([[-1], [2], [1, 3], [4], [0, 5]])
    for train_mode in [True, False]:
        original_model.train(train_mode)
        model.train(train_mode)
        with torch.no_grad():
            assert torch.equal(original_model(batch), model(batch))


# ---------- 4. same seed gives paired runs ----------

def test_same_seed_shares_init_and_batch_order():
    model_sum = make_model('sum', seed=1)
    rng_after_sum = torch.get_rng_state()
    model_set = make_model('set_transformer', seed=1)
    rng_after_set = torch.get_rng_state()

    # same RNG state after init -> same DataLoader shuffle order in training
    assert torch.equal(rng_after_sum, rng_after_set)

    state_sum = model_sum.state_dict()
    state_set = model_set.state_dict()
    shared = state_sum.keys() & state_set.keys()
    assert len(shared) > 0
    for key in shared:
        assert torch.equal(state_sum[key], state_set[key]), key


def test_invalid_pert_agg_raises():
    try:
        make_model('mean')
    except ValueError:
        return
    raise AssertionError("pert_agg='mean' should raise ValueError")


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in list(globals().items())
             if name.startswith('test_') and callable(fn)]
    for name, fn in tests:
        fn()
        print('passed', name)
    print('all {} tests passed'.format(len(tests)))
