"""Validation-first evaluation, baseline generation, and reproducible artifacts."""

from importlib.metadata import version
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from replogle_data.io import digest, require, write_json
from replogle_data.workflow import REPOSITORY
from replogle_training.dataset import TASKS
from .artifacts import (check_seal, load_arrays, read_json, seal, source_hashes,
                        verify_result, write_rows)
from .dataset import EvaluationData
from .metrics import score
from .prediction import batching_audit, load_selected, run_directory, stream


def output_directory(root, task, seed, variant, partition):
    return Path(root) / 'runs' / TASKS[task] / f'seed_{seed}' / variant / partition


def require_output_root(output_root, data_dir, training_root=None):
    output_root, data_dir = Path(output_root).resolve(), Path(data_dir).resolve()
    protected = [REPOSITORY, data_dir]
    if training_root is not None:
        protected.append(Path(training_root).resolve())
    require(all(not output_root.is_relative_to(p) and not p.is_relative_to(output_root) for p in protected),
            'Evaluation root must be separate from checkout, data, and training roots')


def targets(data, root, protocol):
    directory = Path(root) / 'targets' / data.protocol / f'seed_{data.seed}' / data.partition
    expected = dict(identity=data.identity, de_genes=protocol['de_genes'], source_sha256=source_hashes())
    if directory.exists():
        check_seal(directory)
        require(read_json(directory / 'configuration.json') == expected, 'Cached scoring target contract differs')
    else:
        arrays = data.outcomes(protocol['de_genes'])
        arrays.update(cell_rows=data.view['rows'], assigned_control_rows=data.view['input_control_rows'],
                      cell_lines=data.view['cell_line_ids'], cell_target_ids=data.view['target_ids'],
                      all_targets=data.target_names,
                      training_control_rows=data.train['rows'][data.train['target_ids'] == -1],
                      training_control_lines=data.train['cell_line_ids'][data.train['target_ids'] == -1])
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent, prefix='building-') as temporary:
            stage = Path(temporary) / 'payload'
            stage.mkdir()
            write_json(stage / 'configuration.json', expected)
            np.savez_compressed(stage / 'targets.npz', **arrays)
            seal(stage)
            stage.rename(directory)
    return load_arrays(directory / 'targets.npz'), digest(directory / 'manifest.json')


def evaluate(data_dir, training_root, output_root, task, seed, variant, partition, protocol, regenerate=False):
    require(variant in ('A', 'B', 'C', 'Copy-effect'), 'Unknown evaluation variant')
    require(task in TASKS and seed in protocol['split_seeds'], 'Unknown task or split')
    require(not (TASKS[task] == 'joint_unseen' and variant == 'Copy-effect'), 'Copy-effect is inapplicable to joint_unseen')
    require_output_root(output_root, data_dir, training_root)
    directory = output_directory(output_root, task, seed, variant, partition)
    if directory.exists() and not regenerate:
        verify_result(directory)
        saved = read_json(directory / 'evaluation_config.json')
        require(saved['evaluation_protocol'] == protocol and saved['data_directory'] == str(Path(data_dir).resolve()),
                'Existing evaluation invocation differs')
        if variant != 'Copy-effect':
            source = run_directory(Path(training_root), TASKS[task], variant, seed, protocol['training_seed'])
            require(saved['checkpoint_sha256'] == {name: digest(source / name) for name in saved['checkpoint_sha256']},
                    'Existing evaluation checkpoint differs')
        return read_json(directory / 'summary.json')
    require(not regenerate or directory.is_dir(), 'Regeneration requires a completed evaluation bundle')
    # Final testing requires a completed validation adapter check for this exact
    # model/baseline and policy. Scoring cannot select or alter a checkpoint.
    gate_hash = None
    if partition == 'test':
        gate = output_directory(output_root, task, seed, variant, 'val')
        require((gate / 'manifest.json').is_file(), 'Run and verify validation evaluation before final testing')
        verify_result(gate)
        gate_hash = digest(gate / 'manifest.json')
    torch.set_num_threads(protocol['threads'])
    data = EvaluationData(data_dir, task, seed, partition)
    try:
        model, checkpoint_hashes, training_config, training_result = None, {}, None, None
        if variant != 'Copy-effect':
            require(training_root is not None, 'A/B/C evaluation requires --training-root')
            source = run_directory(Path(training_root), data.protocol, variant, seed, protocol['training_seed'])
            model, checkpoint_hashes, training_config, training_result = load_selected(source, data, variant, protocol)
        config = dict(schema_version=1, protocol=data.protocol, split_seed=seed, variant=variant, partition=partition,
                      identity=data.identity, data_directory=str(data.directory), evaluation_protocol=protocol,
                      checkpoint_sha256=checkpoint_hashes, training_configuration=training_config,
                      source_sha256=source_hashes(), validation_gate_sha256=gate_hash,
                      retraining_performed=False, hyperparameter_tuning=False,
                      environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                       scipy=version('scipy'), matplotlib=version('matplotlib')))
        if gate_hash:
            prior = read_json(gate / 'evaluation_config.json')
            require(all(config[k] == prior[k] for k in ('checkpoint_sha256', 'evaluation_protocol', 'training_configuration')),
                    'Test configuration differs from validated adapter')
            require(config['identity']['data'] == prior['identity']['data'] and
                    config['data_directory'] == prior['data_directory'], 'Test data differs from validated adapter')
        if regenerate:
            verify_result(directory)
            require(read_json(directory / 'evaluation_config.json') == config, 'Regeneration configuration differs')
        else:
            directory.mkdir(parents=True)
            # This file exists before predictions and before any outcome decode.
            write_json(directory / 'evaluation_config.json', config)
        try:
            if model is None:
                prediction, inputs = data.copy_effect()
                audit = dict(status='not_applicable', reason='Deterministic training-condition means')
            else:
                audit = batching_audit(model, data, protocol)
                prediction, inputs = stream(model, data, protocol['batch_size'], protocol['device'])
            arrays, target_hash = targets(data, output_root, protocol)
            arrays.update(prediction=prediction, input_means=inputs)
            aggregate, conditions, similarity = score(arrays)
            validation_matches = None
            if model is not None and partition == 'val':
                for key in ('mse', 'mse_de'):
                    require(np.isclose(aggregate['metrics'][key], training_result['selected_validation'][key],
                                       rtol=protocol['batch_audit_rtol'], atol=protocol['batch_audit_atol']),
                            f'Validation adapter differs from selected training checkpoint: {key}')
                validation_matches = True
            summary = dict(status='complete', protocol=data.protocol, split_seed=seed, variant=variant,
                           partition=partition, scores=aggregate, target_manifest_sha256=target_hash,
                           evaluated_cells=int(data.counts.sum()), evaluated_conditions=len(conditions),
                           test_evaluated=partition == 'test', training_validation_verified=validation_matches)
            if regenerate:
                stored = load_arrays(directory / 'scoring_inputs.npz')
                require(set(stored) == set(arrays), 'Regeneration array fields differ')
                for key, value in arrays.items():
                    if key in ('prediction', 'input_means'):
                        require(np.allclose(value, stored[key], rtol=protocol['regeneration_rtol'], atol=protocol['regeneration_atol']),
                                f'Regenerated inference differs: {key}')
                    else:
                        require(np.array_equal(value, stored[key]), f'Regenerated scoring contract differs: {key}')
                original = read_json(directory / 'summary.json')
                # Metrics are checked with the same numerical tolerance as predictions.
                def compare(a, b):
                    if isinstance(a, dict):
                        return set(a) == set(b) and all(compare(a[k], b[k]) for k in a)
                    if isinstance(a, (float, int)) and not isinstance(a, bool):
                        return bool(np.isclose(a, b, rtol=protocol['regeneration_rtol'], atol=protocol['regeneration_atol']))
                    return a == b
                require(compare(summary, original), 'Regenerated metrics differ')
                report = dict(status='passed', evaluation_manifest_sha256=digest(directory / 'manifest.json'),
                              inference_and_metrics_regenerated=True, batching_audit=audit)
                reports = Path(output_root) / 'regeneration' / data.protocol / f'seed_{seed}' / variant
                reports.mkdir(parents=True, exist_ok=True)
                write_json(reports / (partition + '.json'), report)
                return report
            np.savez_compressed(directory / 'scoring_inputs.npz', **arrays)
            write_rows(directory / 'per_condition_metrics.csv', conditions)
            write_rows(directory / 'per_target_similarity.csv', similarity)
            write_json(directory / 'batching_audit.json', audit)
            write_json(directory / 'summary.json', summary)
            seal(directory)
            verify_result(directory)
            return summary
        except BaseException as error:
            if not regenerate:
                write_json(directory / 'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise
    finally:
        data.close()
