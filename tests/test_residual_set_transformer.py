"""Behavioral tests for original GEARS fusion plus an attention correction."""
import subprocess

import pytest
import torch

from test_coexpression_batching import corrected_model
from test_set_transformer import REPO_ROOT, load_module, make_args, make_batch


@pytest.mark.parametrize('train_mode', [False, True])
@pytest.mark.parametrize('perts', [
    [[-1], [2], [1, 3], [0, 5]], [[-1], [-1]], [[2], [-1]],
])
def test_zero_residual_starts_as_original_fusion(train_mode, perts):
    baseline = corrected_model('sum').train(train_mode)
    rng_baseline = torch.get_rng_state()
    residual = corrected_model('residual_set_transformer').train(train_mode)
    assert torch.equal(rng_baseline, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, residual.state_dict()[name]), name
    batch = make_batch(perts)
    expected, actual = baseline(batch), residual(batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for name, parameter in baseline.named_parameters():
        other = dict(residual.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-5)


def test_projection_then_attention_learn():
    model = corrected_model('residual_set_transformer').train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = make_batch([[1, 3], [2], [0, 5], [-1]])
    for step in range(2):
        optimizer.zero_grad()
        output = model(batch)
        loss = (output - 0.3).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        projection_grad = model.set_residual_projection.weight.grad
        assert torch.isfinite(projection_grad).all() and projection_grad.abs().sum() > 0
        attention_grads = [parameter.grad for name, parameter in model.named_parameters()
                           if name.startswith(('set_attention.', 'set_pooling.'))]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in attention_grads)
        gradient_total = sum(grad.abs().sum() for grad in attention_grads)
        assert gradient_total == 0 if step == 0 else gradient_total > 0
        optimizer.step()


def learned_residual():
    model = corrected_model('residual_set_transformer')
    with torch.no_grad():
        model.set_residual_projection.weight.normal_(std=0.1)
        model.set_residual_projection.bias.fill_(0.2)
    return model


def test_active_residual_respects_set_order_padding_and_controls():
    model = learned_residual()
    baseline = corrected_model('sum')
    perts = [[1, 3], [2], [-1]]
    with torch.no_grad():
        output = model(make_batch(perts))
        reversed_set = model(make_batch([[3, 1], [2], [-1]]))
        unpadded_single = model(make_batch([[1], [2], [-1]]))
        original_fusion = baseline(make_batch(perts))
        controls = make_batch([[-1], [-1]])
        torch.testing.assert_close(model(controls), baseline(controls))
    torch.testing.assert_close(output, reversed_set)
    torch.testing.assert_close(output[1], unpadded_single[1])
    torch.testing.assert_close(output[2], original_fusion[2])
    assert not torch.allclose(output[:2], original_fusion[:2])


@pytest.mark.parametrize('aggregation', ['sum', 'set_transformer'])
def test_existing_variants_match_batch_fix_commit(tmp_path, aggregation):
    # Protect both existing modes when factoring out the shared attention code.
    source = subprocess.run(['git', 'show', 'c154fa1:gears/model.py'], cwd=REPO_ROOT,
                            check=True, capture_output=True, text=True).stdout
    reference_path = tmp_path / 'reference.py'
    reference_path.write_text(source)
    reference_module = load_module('batch_fixed_reference', str(reference_path))
    args = make_args(aggregation)
    args['coexpression_batching'] = 'shared'
    current = corrected_model(aggregation)
    reference = reference_module.GEARS_Model(args)
    reference.load_state_dict(current.state_dict())
    batch = make_batch([[-1], [2], [1, 3], [0, 5]])
    for training in (True, False):
        current.train(training)
        reference.train(training)
        torch.testing.assert_close(current(batch), reference(batch), atol=0, rtol=0)
