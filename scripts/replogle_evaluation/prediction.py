"""Strict validation-selected checkpoint loading and streaming inference."""

import json

import numpy as np
import torch

from replogle_data.io import digest, require
from replogle_data.workflow import REPOSITORY
from replogle_training.artifacts import load_checkpoint


def run_directory(training_root, protocol, variant, seed, training_seed):
    return training_root / 'runs' / f'{protocol}_{variant}_split{seed}_train{training_seed}'


def load_selected(directory, data, variant, protocol):
    files = ('run.json', 'history.json', 'results.json', 'best.pt', 'best.pt.sha256')
    require(all((directory / name).is_file() for name in files), f'Complete trained checkpoint missing: {directory}')
    run, history, result = [json.loads((directory / name).read_text()) for name in files[:3]]
    config = run['configuration']
    identity = directory.name
    require(run['state'] == 'complete' and not (directory / 'failure.json').exists(), 'Training run is incomplete or failed')
    require(run['run_id'] == result['run_id'] == identity and run['variant'] == result['variant'] == variant,
            'Checkpoint run/variant identity differs')
    require(result['protocol'] == data.protocol and result['split_seed'] == data.seed and run['data'] == data.contract,
            'Checkpoint task/split/data contract differs')
    require(config['training_seed'] == protocol['training_seed'] and config['de_genes'] == protocol['de_genes'],
            'Training seed or validation DE setting differs')
    require(config['selection_metric'] == result['selection_metric'] == 'mse_de', 'Checkpoint must use validation mse_de')
    require(run['test_evaluated'] is False and result['test_evaluated'] is False and result['checkpoint_reload_verified'],
            'Training exposure/reload record differs')
    require(len(history) == result['epochs_completed'] == config['epochs'] and
            [row['epoch'] for row in history] == list(range(1, len(history) + 1)), 'Training history is incomplete')
    values = [row['validation']['mse_de'] for row in history]
    require(np.isfinite(values).all(), 'Invalid validation history')
    best = int(np.argmin(values))
    require(result['best_epoch'] == best + 1 and result['selection_value'] == values[best], 'Checkpoint is not the first validation minimum')
    require(np.isclose(result['selected_validation']['mse_de'], values[best], rtol=1e-6, atol=1e-8), 'Reloaded validation score differs')
    require(digest(directory / 'best.pt') == result['best_checkpoint_sha256'], 'Checkpoint result hash differs')
    for name, expected in run['source_sha256'].items():
        require(digest(REPOSITORY / name) == expected, f'Training source changed: {name}')
    model, payload = load_checkpoint(directory / 'best.pt', protocol['device'], data.genes, data.perturbations, data.contract)
    require(payload['variant'] == variant and payload['epoch'] == result['best_epoch'] and
            payload['selection_value'] == result['selection_value'], 'Checkpoint selection payload differs')
    require(payload['model_config']['context_names'] == ['K562', 'RPE1'], 'Checkpoint cell-line order differs')
    controls, _, _ = data.control_statistics()
    np.testing.assert_allclose(payload['control_means'].cpu().numpy(), controls, rtol=1e-6, atol=1e-7)
    model.eval()
    return model, {name: digest(directory / name) for name in files}, config, result


def stream(model, data, batch_size, device):
    sums = np.zeros((len(data.codes), len(data.genes)), dtype=np.float64)
    input_sums = np.zeros_like(sums)
    counts = np.zeros(len(data.codes), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for batch in data.prediction_batches(batch_size, device=device):
            prediction = model(batch['x'], batch['nodes'], batch['contexts']).cpu().numpy()
            require(prediction.shape == tuple(batch['x'].shape) and np.isfinite(prediction).all(), 'Invalid prediction')
            np.add.at(sums, batch['groups'], prediction.astype(np.float64))
            np.add.at(input_sums, batch['groups'], batch['x'].cpu().numpy().astype(np.float64))
            counts += np.bincount(batch['groups'], minlength=len(counts))
    require(np.array_equal(counts, data.counts), 'Missing or duplicate predicted cells')
    return sums / counts[:, None], input_sums / counts[:, None]


def batching_audit(model, data, protocol):
    """Compare original/reversed order, a different batch size, and singleton tail."""
    positions = np.arange(min(33, len(data.groups)))
    def predict(order, size):
        output = np.zeros((len(positions), len(data.genes)))
        with torch.inference_mode():
            for batch in data.prediction_batches(size, positions=order, device=protocol['device']):
                output[batch['positions']] = model(batch['x'], batch['nodes'], batch['contexts']).cpu().numpy()
        return output
    reference = predict(positions, max(len(positions), 1))
    repeated = predict(positions, max(len(positions), 1))
    alternative = predict(positions[::-1], 7)
    singleton = predict(positions, 1)
    require(np.isfinite(reference).all(), 'Nonfinite batch audit')
    for other in (repeated, alternative, singleton):
        require(np.allclose(reference, other, rtol=protocol['batch_audit_rtol'], atol=protocol['batch_audit_atol']),
                'Evaluation predictions depend on batch order or size')
    return dict(status='passed', cells=len(positions),
                max_absolute_difference=float(max(np.max(np.abs(reference - p)) for p in (repeated, alternative, singleton))))
