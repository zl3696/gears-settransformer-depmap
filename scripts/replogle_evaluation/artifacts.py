"""Sealed evaluation bundles with numerical verification independent of inference."""

import csv
import json
from pathlib import Path

import numpy as np

from replogle_data.io import digest, require, write_json
from replogle_data.workflow import REPOSITORY

DEFAULT = REPOSITORY / 'configs/replogle_evaluation.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def load_protocol(path=DEFAULT):
    protocol = read_json(path)
    # Numerical rules are versioned together; callers may copy this config only
    # to adjust device/batch/threads, not silently redefine an existing protocol.
    fixed = read_json(DEFAULT)
    require(set(protocol) == set(fixed), 'Evaluation configuration fields differ')
    for key in set(fixed) - {'batch_size', 'device', 'threads'}:
        require(protocol[key] == fixed[key], f'Unsupported evaluation rule: {key}')
    require(protocol['batch_size'] >= 1 and protocol['threads'] >= 1, 'Invalid compute budget')
    require(protocol['device'] in ('cpu', 'mps', 'cuda'), 'Select an explicit evaluation device')
    return protocol


def source_hashes():
    paths = [REPOSITORY / 'scripts/replogle_test.py', DEFAULT, REPOSITORY / 'gears/inference.py',
             * (REPOSITORY / 'scripts/replogle_evaluation').glob('*.py'),
             REPOSITORY / 'scripts/replogle_training/dataset.py', REPOSITORY / 'scripts/replogle_training/artifacts.py',
             REPOSITORY / 'gears/replogle_model.py', REPOSITORY / 'gears/model.py', REPOSITORY / 'gears/depmap.py',
             * (REPOSITORY / 'scripts/replogle_data').glob('*.py')]
    return {p.relative_to(REPOSITORY).as_posix(): digest(p) for p in sorted(paths)}


def write_rows(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else ['target']
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def seal(directory):
    files = {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    write_json(directory / 'manifest.json', dict(schema_version=1, files=files))


def check_seal(directory):
    manifest = read_json(directory / 'manifest.json')
    require(manifest['schema_version'] == 1, 'Unsupported artifact manifest')
    names = {p.name for p in directory.iterdir() if p.is_file() and p.name != 'manifest.json'}
    require(names == set(manifest['files']), 'Evaluation artifact file set differs')
    for name, expected in manifest['files'].items():
        require(Path(name).name == name and digest(directory / name) == expected, f'Evaluation artifact hash differs: {name}')
    return manifest


def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def verify_result(directory, check_sources=True):
    from .metrics import de_indices, score
    directory = Path(directory)
    check_seal(directory)
    config = read_json(directory / 'evaluation_config.json')
    if check_sources:
        require(config['source_sha256'] == source_hashes(), 'Evaluation source changed; use the original evaluator')
    arrays = load_arrays(directory / 'scoring_inputs.npz')
    require(arrays['genes'].tolist() == config['identity']['data']['expression_genes'], 'Saved expression gene order differs')
    require(len(set(arrays['all_targets'])) == len(arrays['all_targets']), 'Duplicate target vocabulary')
    require(np.all((arrays['cell_lines'] >= 0) & (arrays['cell_lines'] <= 1)) and
            np.all((arrays['cell_target_ids'] >= 0) & (arrays['cell_target_ids'] < len(arrays['all_targets']))),
            'Invalid saved context/target identifiers')
    control_lookup = dict(zip(arrays['training_control_rows'].tolist(), arrays['training_control_lines'].tolist()))
    require(all(control_lookup.get(int(row)) == int(line) for row, line in
                zip(arrays['assigned_control_rows'], arrays['cell_lines'])), 'Saved same-line training-control schedule differs')
    # Stored means must be compatible with the saved exact cell schedule.
    codes = arrays['cell_lines'] * len(arrays['all_targets']) + arrays['cell_target_ids']
    unique, inverse = np.unique(codes, return_inverse=True)
    require(np.array_equal(unique // len(arrays['all_targets']), arrays['lines']) and
            np.array_equal(arrays['all_targets'][unique % len(arrays['all_targets'])], arrays['targets']), 'Saved condition/cell axes differ')
    require(np.array_equal(np.bincount(inverse), arrays['counts']), 'Saved cell counts differ')
    require(len(set(arrays['cell_rows'])) == len(arrays['cell_rows']), 'Duplicate saved target cells')
    require(np.array_equal(arrays['targets'], arrays['all_targets'][unique % len(arrays['all_targets'])]), 'Saved target identity differs')
    ranked = de_indices(arrays['truth'], arrays['variances'], arrays['counts'], arrays['control_means'],
                        arrays['control_variances'], arrays['control_counts'], arrays['lines'], config['evaluation_protocol']['de_genes'])
    require(np.array_equal(ranked, arrays['de_indices']), 'Recomputed DE ranking differs')
    aggregate, conditions, similarity = score(arrays)
    require(read_json(directory / 'summary.json')['scores'] == aggregate, 'Recomputed metrics differ')
    # Re-encode CSV values to verify tables as well as numerical arrays.
    for name, rows in [('per_condition_metrics.csv', conditions), ('per_target_similarity.csv', similarity)]:
        saved = read_rows(directory / name)
        expected = [{key: '' if value is None else str(value) for key, value in row.items()} for row in rows]
        require(saved == expected, f'Recomputed table differs: {name}')
    return dict(status='passed', directory=str(directory), conditions=len(conditions), test_evaluated=config['partition'] == 'test')
