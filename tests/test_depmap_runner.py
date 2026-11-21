"""Exercise the actual training CLI on synthetic data, including saved provenance."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from gears.depmap import file_sha256, save_features
from test_norman_protocol import make_gears

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('enabled', [False, True])
def test_training_entrypoint_routes_depmap_and_saves_provenance(tmp_path, monkeypatch, enabled):
    spec = importlib.util.spec_from_file_location('depmap_training_runner', REPO / 'scripts/norman_train.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    protocol = json.loads((REPO / 'configs/norman_depmap_training.json').read_text())
    protocol['training']['epochs'] = 1
    protocol['model']['hidden_size'] = 8
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    features = tmp_path / 'features.npz'
    if enabled:
        save_features(features, ['A', 'B'], [[-.5, .2], [-1., .4]], {'source_version': 'test'})
    model = make_gears()
    model.saved_pred['stale'] = 'old model prediction'

    class SyntheticPertData:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, **kwargs):
            pass

        def prepare_split(self, **kwargs):
            pass

        def get_dataloader(self, **kwargs):
            pass

    graphs = {'fit_cells': 24}
    for name, config_name in [('coexpress', 'G_coexpress'), ('go', 'G_go')]:
        edge, weight = model.config[config_name], model.config[config_name + '_weight']
        graphs[name] = dict(edge_index=edge, edge_weight=weight, sha256=runner.tensor_sha256(edge, weight))
    monkeypatch.setattr(runner, 'PertData', SyntheticPertData)
    monkeypatch.setattr(runner, 'GEARS', lambda *args, **kwargs: model)
    monkeypatch.setattr(runner, 'shared_graphs', lambda *args: graphs)
    run_id = ('gears_depmap' if enabled else 'gears') + '_split1_train101'
    argv = ['norman_train.py', run_id, '--protocol', str(protocol_path), '--device', 'cpu',
            '--data-path', str(tmp_path / 'data'), '--output-root', str(tmp_path)]
    if enabled:
        argv += ['--depmap-features', str(features)]
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    directory = tmp_path / 'outputs/norman-benchmark/depmap_comparison_v1' / run_id
    result = json.loads((directory / 'results.json').read_text())
    assert result['use_depmap'] == enabled
    assert result['test_evaluated'] is False
    assert result['best_epoch'] == 1 and result['selection_metric'] == 'mse_de'
    assert result['protocol_sha256'] == file_sha256(protocol_path)
    assert model.saved_pred == {}
    restored = make_gears()
    restored.load_pretrained(str(directory))
    assert restored.config['use_depmap'] == enabled
    if enabled:
        assert result['depmap_bundle_sha256'] == file_sha256(features)
        assert result['depmap']['matched_genes'] == 2
        assert result['depmap']['missing_genes'] == ['C']
    else:
        assert result['depmap_bundle_sha256'] is None
