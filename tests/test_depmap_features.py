"""DepMap input integrity and exact identifier alignment, independent of training."""
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gears.depmap import AlignedDepMap, align_features, load_features, save_features

REPO = Path(__file__).resolve().parents[1]


def test_feature_order_does_not_change_alignment(tmp_path):
    genes = ['B', 'A', 'EXTRA']
    values = np.arange(12).reshape(3, 4) / 7
    first = save_features(tmp_path / 'one.npz', genes, values, {'source_version': 'test'})
    second = save_features(tmp_path / 'two.npz', genes[::-1], values[::-1], {'source_version': 'test'})
    a = align_features(load_features(tmp_path / 'one.npz'), ['A', 'MISSING', 'B'])
    b = align_features(second, ['A', 'MISSING', 'B'])
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.mask, [True, False, True])
    assert a.metadata['aligned_sha256'] == b.metadata['aligned_sha256']
    assert a.metadata['missing_genes'] == ['MISSING']
    assert a.metadata['matched_genes'] == 2
    assert np.all(a.values[1] == 0)
    assert first.metadata['feature_dim'] == 4
    with pytest.raises(ValueError, match='No perturbation genes match'):
        align_features(first, ['absent'])
    with pytest.raises(ValueError, match='Duplicate'):
        align_features(first, ['A', 'A'])


@pytest.mark.parametrize('genes,values', [
    (['A', 'A'], [[1], [2]]), ([' A'], [[1]]), ([''], [[1]]),
    (['A'], [[float('nan')]]), (['A'], [[float('inf')]]),
    (['A'], [[1e100]]), (['A'], [['bad']]), (['A'], [[1j]]),
    (['A'], [[]]), (['A'], [1]), (['A'], [[1], [2]]),
])
def test_malformed_features_rejected(tmp_path, genes, values):
    with pytest.raises(ValueError):
        save_features(tmp_path / 'bad.npz', genes, values, {})


def test_bundle_and_aligned_corruption_rejected(tmp_path):
    path = tmp_path / 'good.npz'
    features = save_features(path, ['A', 'B'], [[1, 2], [3, 4]], {})
    with pytest.raises(FileExistsError):
        save_features(path, ['A'], [[1]], {})
    with np.load(path, allow_pickle=False) as bundle:
        contents = {k: bundle[k].copy() for k in bundle.files}
    contents['values'][0, 0] += 1
    np.savez(tmp_path / 'corrupt.npz', **contents)
    with pytest.raises(ValueError, match='hash'):
        load_features(tmp_path / 'corrupt.npz')
    aligned = align_features(features, ['A', 'B', 'C'])
    with pytest.raises(ValueError, match='order'):
        aligned.validate(['B', 'A', 'C'])
    aligned.values[0, 0] += 1
    with pytest.raises(ValueError, match='hash'):
        aligned.validate(['A', 'B', 'C'])
    aligned = align_features(features, ['A', 'B', 'C'])
    with pytest.raises(ValueError, match='boolean'):
        AlignedDepMap(aligned.values, aligned.mask.astype(int), aligned.metadata).validate(['A', 'B', 'C'])


def test_morph_conversion_from_any_directory(tmp_path):
    source = tmp_path / 'morph.pkl'
    with source.open('wb') as stream:
        pickle.dump({'A': np.array([-.5, .3]), 'B': np.array([-1., -.1])}, stream)
    nodes = tmp_path / 'nodes.csv'
    nodes.write_text('gene_name\nB\nC\nA\n')
    output = tmp_path / 'out' / 'features.npz'
    command = [sys.executable, str(REPO / 'scripts/prepare_depmap.py'),
               '--input-pickle', str(source), '--output', str(output),
               '--perturbation-nodes', str(nodes), '--source-version', 'synthetic-test']
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    features = load_features(output)
    assert features.metadata['source_version'] == 'synthetic-test'
    assert features.metadata['reference_cell_line_ids'] is None
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['coverage']['matched_genes'] == 2
    assert report['coverage']['missing_genes'] == ['C']
    assert subprocess.run(command, cwd=tmp_path, capture_output=True).returncode != 0
