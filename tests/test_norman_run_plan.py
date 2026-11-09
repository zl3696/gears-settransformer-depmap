"""Ensure the corrected comparison is runnable without changing frozen splits."""
import csv
import json
from pathlib import Path
import subprocess
import sys

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
