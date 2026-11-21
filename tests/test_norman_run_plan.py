"""Ensure the corrected comparison is runnable without changing frozen splits."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import shlex

import numpy as np

from gears.depmap import file_sha256, save_features

REPO = Path(__file__).resolve().parents[1]


def test_corrected_run_plan_and_portable_launcher(tmp_path):
    protocol = json.loads((REPO / 'configs/norman_training.json').read_text())
    with (REPO / protocol['run_plan']).open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    assert len(runs) == 5 * len(protocol['models'])
    assert len({row['run_id'] for row in runs}) == len(runs)
    assert len({row['result_path'] for row in runs}) == len(runs)
    with (REPO / 'splits/norman/planned_runs.csv').open(newline='') as stream:
        historical_paths = {row['result_path'] for row in csv.DictReader(stream)}
    for model in protocol['models']:
        assert {int(row['split_seed']) for row in runs if row['model'] == model} == set(range(1, 6))
    for row in runs:
        assert row['result_path'] not in historical_paths
        assert int(row['training_seed']) == protocol['training_seed']
        assert row['shared_configuration_sha256'] == protocol['split_configuration_sha256']
        assert row['manifest_path'] == f"seed_{row['split_seed']}/manifest.json"
        assert Path(row['checkpoint_path']).name == 'model.pt'
        assert Path(row['result_path']).parent == Path(row['checkpoint_path']).parent
    command = [sys.executable, str(REPO / 'scripts/run_norman_all.py'), '--dry-run',
               '--splits', '1', '--data-path', str(tmp_path / 'data'),
               '--output-root', str(tmp_path / 'results')]
    result = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert len(result.stdout.splitlines()) == len(protocol['models'])
    assert not (tmp_path / 'data').exists()
    assert not (tmp_path / 'results').exists()
    for row in runs:
        if row['split_seed'] == '1':
            assert row['run_id'] in result.stdout
    result = subprocess.run(command + ['--models', 'not_a_model'], cwd=tmp_path, capture_output=True)
    assert result.returncode != 0


def test_depmap_plan_supports_six_combinations_without_overwriting_old_runs(tmp_path):
    path = REPO / 'configs/norman_depmap_training.json'
    protocol = json.loads(path.read_text())
    with (REPO / protocol['run_plan']).open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    assert len(runs) == 30
    assert {(v['pert_agg'], v['use_depmap']) for v in protocol['models'].values()} == {
        (mode, enabled) for mode in ('sum', 'set_transformer', 'residual_set_transformer')
        for enabled in (False, True)}
    with (REPO / 'configs/norman_runs.csv').open(newline='') as stream:
        old_paths = {r['result_path'] for r in csv.DictReader(stream)}
    assert len({r['result_path'] for r in runs}) == 30
    assert not old_paths & {r['result_path'] for r in runs}
    assert {r['model_configuration'] for r in runs} == {'configs/norman_depmap_training.json'}
    for name in protocol['models']:
        assert {int(r['split_seed']) for r in runs if r['model'] == name} == set(range(1, 6))
    original = json.loads((REPO / 'configs/norman_training.json').read_text())
    for key in ('split_dir', 'split_bundle_sha256', 'split_configuration_sha256',
                'training_seed', 'model', 'graphs', 'training', 'checkpoint_selection'):
        assert protocol[key] == original[key]
    features = tmp_path / 'features.npz'
    command = [sys.executable, str(REPO / 'scripts/run_norman_all.py'), '--dry-run',
               '--protocol', str(path), '--splits', '1', '--data-path', str(tmp_path / 'data'),
               '--output-root', str(tmp_path / 'results')]
    assert subprocess.run(command, cwd=tmp_path, capture_output=True).returncode != 0
    result = subprocess.run(command + ['--depmap-features', str(features)], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 6
    assert sum('--depmap-features' in c for c in commands) == 3
    for c in commands:
        assert c[c.index('--protocol') + 1] == str(path)
    assert not (tmp_path / 'results').exists()


def test_launcher_rejects_existing_result_from_different_depmap_features(tmp_path):
    path = REPO / 'configs/norman_depmap_training.json'
    protocol = json.loads(path.read_text())
    with (REPO / protocol['run_plan']).open(newline='') as stream:
        row = next(r for r in csv.DictReader(stream) if r['model'] == 'gears_depmap' and r['split_seed'] == '1')
    features = tmp_path / 'features.npz'
    save_features(features, ['A'], np.ones((1, 2)), {})
    result_path = tmp_path / row['result_path']
    result_path.parent.mkdir(parents=True)
    for name in ('config.pkl', 'model.pt'):
        (result_path.parent / name).touch()
    result = dict(protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  run_plan_sha256=file_sha256(REPO / protocol['run_plan']),
                  run_id=row['run_id'], use_depmap=True, depmap_bundle_sha256='wrong')
    result_path.write_text(json.dumps(result))
    command = [sys.executable, str(REPO / 'scripts/run_norman_all.py'),
               '--protocol', str(path), '--splits', '1', '--models', 'gears_depmap',
               '--data-path', str(tmp_path / 'data'), '--output-root', str(tmp_path),
               '--depmap-features', str(features)]
    rejected = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'another protocol' in rejected.stderr
    result['depmap_bundle_sha256'] = file_sha256(features)
    result_path.write_text(json.dumps(result))
    skipped = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert 'Skip completed' in skipped.stdout
