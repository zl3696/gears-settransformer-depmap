"""Convert MORPH's trusted gene-to-vector pickle to a validated, safe NPZ bundle.

Only use --input-pickle with a trusted source: Python pickle can execute code.
The GEARS runtime reads the resulting NPZ with allow_pickle=False.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gears.depmap import align_features, file_sha256, save_features

MORPH_SOURCE = 'https://drive.google.com/file/d/1qhPMgarLgEgOkh8KwInrJuNoIAiMk46-/view'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-pickle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-url', default=MORPH_SOURCE)
    parser.add_argument('--source-version', default='unspecified; identified by source SHA256')
    parser.add_argument('--perturbation-nodes', type=Path,
                        help='Optional CSV with gene_name in GEARS perturbation-node order')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(REPO):
        parser.error('Feature bundles must be outside the source checkout')
    with args.input_pickle.open('rb') as stream:
        profiles = pickle.load(stream)
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError('MORPH input must be a nonempty gene-symbol-to-vector dictionary')
    genes = list(profiles)
    vectors = [np.asarray(profiles[g]) for g in genes]
    if any(v.ndim != 1 for v in vectors) or len({v.shape for v in vectors}) != 1:
        raise ValueError('MORPH profiles must be one-dimensional vectors of equal length')
    features = save_features(args.output, genes, np.stack(vectors), {
        'source_url': args.source_url, 'source_version': args.source_version,
        'source_filename': args.input_pickle.name,
        'source_sha256': file_sha256(args.input_pickle),
        'preprocessing': 'MORPH processed values preserved; float32 conversion only',
        'gene_mapping': 'exact case-sensitive symbols; no automatic aliases',
        'reference_cell_line_ids': None,
        'feature_order': 'original MORPH vector order; cell-line identifiers not supplied',
    })
    report = dict(features.metadata, output=str(args.output.resolve()),
                  bundle_sha256=file_sha256(args.output))
    if args.perturbation_nodes:
        with args.perturbation_nodes.open(newline='') as stream:
            nodes = [row['gene_name'] for row in csv.DictReader(stream)]
        aligned = align_features(features, nodes)
        report['coverage'] = {k: aligned.metadata[k] for k in
                              ('matched_genes', 'missing_genes', 'aligned_sha256')}
        report['coverage']['total_genes'] = len(nodes)
    report_path = args.output.with_suffix('.json')
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    summary = {k: report[k] for k in ('output', 'gene_count', 'feature_dim', 'source_sha256')}
    if 'coverage' in report:
        summary['coverage'] = {k: report['coverage'][k] for k in ('matched_genes', 'total_genes')}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
