"""Build and audit pooled single-cell expression and immutable dataset views."""

import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .io import (digest, metadata, read_column, read_csr, read_index, require,
                 source_blocks, write_json, write_table)
from .protocol import (LINES, PARTITIONS, PROTOCOLS, cell_partitions, fixed_controls,
                       gene_partition, sampling_weights)

REPOSITORY = Path(__file__).resolve().parents[2]


def write_matrix(path, obs, var, blocks):
    """Write a standard AnnData H5AD with a bounded-memory CSR expression stream."""
    ad.AnnData(X=csr_matrix((len(obs), len(var)), dtype=np.float32), obs=obs, var=var).write_h5ad(path)
    with h5py.File(path, 'r+') as handle:
        group = handle['X']
        for name in ('data', 'indices', 'indptr'):
            del group[name]
        values = group.create_dataset('data', shape=(0,), maxshape=(None,), dtype='<f4',
                                      chunks=True, compression='gzip', compression_opts=4, shuffle=True)
        indices = group.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='<i4',
                                       chunks=True, compression='gzip', compression_opts=4, shuffle=True)
        pointers = group.create_dataset('indptr', shape=(len(obs) + 1,), dtype='<i8')
        n_rows = n_values = 0
        for block in blocks:
            require(block.shape[1] == len(var), 'Expression block gene count differs')
            end = n_values + block.nnz
            values.resize((end,)); indices.resize((end,))
            values[n_values:end] = block.data
            indices[n_values:end] = block.indices
            pointers[n_rows:n_rows + block.shape[0] + 1] = block.indptr + n_values
            n_rows += block.shape[0]
            n_values = end
        require(n_rows == len(obs), 'Expression stream row count differs')


def prepare(k562, rpe1, output, config_path, go_path, depmap_genes=None):
    output = Path(output).resolve()
    require(not output.is_relative_to(REPOSITORY), 'Runtime data must stay outside the checkout')
    require(not output.exists(), 'Output exists; choose a new version or run verify')
    config = json.loads(Path(config_path).read_text())
    require(config['schema_version'] == 1 and tuple(config['protocols']) == PROTOCOLS,
            'Unsupported data protocol')
    sources = {}
    for line, path in zip(LINES, (k562, rpe1)):
        print(f'Checking {line} source hash and metadata', flush=True)
        require(digest(path) == config['source_sha256'][line], f'{line} source hash differs')
        sources[line] = metadata(path)
    with Path(go_path).open(newline='') as stream:
        go_genes = [row['gene_name'] for row in csv.DictReader(stream)]
    require(len(go_genes) == len(set(go_genes)), 'GO vocabulary has duplicate genes')
    counts = {line: sources[line]['counts'] for line in LINES}
    shared = (set(counts['k562']) & set(counts['rpe1'])) - {'ctrl'}
    targets = sorted(shared & set(go_genes))
    roles = gene_partition(targets, config['split_seeds'][0], config['validation_fraction'], config['test_fraction'])
    del roles
    depmap = set(json.loads(Path(depmap_genes).read_text())) if depmap_genes else None
    common_ids = sorted(set(sources['k562']['gene_ids']) & set(sources['rpe1']['gene_ids']))
    require(bool(common_ids), 'No shared measured genes')
    columns, gene_symbols = {}, None
    for line in LINES:
        source = sources[line]
        lookup = {g: i for i, g in enumerate(source['gene_ids'])}
        columns[line] = np.asarray([lookup[g] for g in common_ids], dtype=np.int64)
        names = source['symbols'][columns[line]].tolist()
        require(len(names) == len(set(names)), 'Shared expression panel has ambiguous gene symbols')
        require(gene_symbols is None or names == gene_symbols,
                'Shared Ensembl identifiers disagree on gene symbols')
        gene_symbols = names
        source['rows'] = np.flatnonzero(np.isin(source['labels'], targets + ['ctrl']))
    output.mkdir(parents=True)
    write_json(output / 'config.json', config)
    write_table(output / 'genes.csv', ('expression_index', 'gene_id', 'gene_name', 'k562_source_column', 'rpe1_source_column'),
                ((i, g, gene_symbols[i], columns['k562'][i], columns['rpe1'][i]) for i, g in enumerate(common_ids)))
    write_table(output / 'perturbation_nodes.csv', ('perturbation_index', 'gene_name'), enumerate(go_genes))
    write_table(output / 'paired_targets.csv', ('target_id', 'target', 'k562_ncells', 'rpe1_ncells',
                'low_cell_count', 'depmap_exact_match', 'k562_source_labels', 'rpe1_source_labels'),
                ((i, g, counts['k562'][g], counts['rpe1'][g],
                  min(counts[x][g] for x in LINES) < config['low_cell_count_flag'],
                  '' if depmap is None else g in depmap,
                  json.dumps(sources['k562']['aliases'][g]), json.dumps(sources['rpe1']['aliases'][g]))
                 for i, g in enumerate(targets)))
    all_targets = (set(counts['k562']) | set(counts['rpe1'])) - {'ctrl'}
    write_table(output / 'excluded_targets.csv', ('target', 'k562_ncells', 'rpe1_ncells', 'reason'),
                ((g, counts['k562'][g], counts['rpe1'][g],
                  'absent_in_one_cell_line' if g not in shared else 'absent_from_fixed_go_vocabulary')
                 for g in sorted(all_targets - set(targets))))
    frames = []
    for line_id, line in enumerate(LINES):
        source, rows = sources[line], sources[line]['rows']
        labels = source['labels'][rows]
        frames.append(pd.DataFrame(dict(
            source_row=rows, source_cell_id=source['barcodes'][rows], source_condition=source['source_labels'][rows],
            cell_line=line.upper(), cell_line_id=np.full(len(rows), line_id, dtype=np.int8), target=labels,
            condition=np.where(labels == 'ctrl', 'ctrl', np.char.add(labels, '+ctrl')),
            condition_key=[line.upper() + '::' + g for g in labels]),
            index=[line.upper() + '::' + g for g in source['barcodes'][rows]]))
    obs = pd.concat(frames)
    require(obs.index.is_unique, 'Namespaced cell identifiers are not unique')
    obs.index.name = 'cell_id'
    var = pd.DataFrame(dict(gene_name=gene_symbols, gene_id=common_ids), index=common_ids)
    obs.to_csv(output / 'cells.csv.gz', compression=dict(method='gzip', mtime=0), index=True)
    labels = obs['target'].to_numpy()
    contexts = obs['cell_line_id'].to_numpy()
    target_lookup = {g: i for i, g in enumerate(targets)}
    target_ids = np.asarray([target_lookup.get(g, -1) for g in labels], dtype=np.int32)
    control_rows = fixed_controls(labels, contexts, config['control_seed'])
    np.save(output / 'input_control_rows.npy', control_rows, allow_pickle=False)
    np.savez_compressed(output / 'controls.npz', **{line: np.flatnonzero((labels == 'ctrl') & (contexts == i))
                                                  for i, line in enumerate(LINES)})
    print(f'Writing {len(obs):,} cells x {len(var):,} shared expression genes; {len(targets)} paired targets', flush=True)
    def expression_blocks():
        for line in LINES:
            for _, block in source_blocks(sources[line]['path'], sources[line]['rows'], columns[line]):
                yield block
            print(f'Wrote {line} expression', flush=True)
    write_matrix(output / 'expression.h5ad', obs, var, expression_blocks())
    summaries = []
    for seed in config['split_seeds']:
        target_roles = gene_partition(targets, seed, config['validation_fraction'], config['test_fraction'])
        for protocol in PROTOCOLS:
            directory = output / 'views' / protocol / f'seed_{seed}'
            directory.mkdir(parents=True)
            assigned = cell_partitions(labels, contexts, target_roles, protocol)
            panels = {}
            for part in PARTITIONS:
                rows = np.flatnonzero(assigned == part)
                arrays = dict(rows=rows, input_control_rows=control_rows[rows],
                              cell_line_ids=contexts[rows], target_ids=target_ids[rows])
                arrays.update({line + '_rows': rows[contexts[rows] == i] for i, line in enumerate(LINES)})
                if part == 'train':
                    arrays['sampling_weights'] = sampling_weights(labels[rows], contexts[rows])
                np.savez_compressed(directory / (part + '.npz'), **arrays)
                panels[part] = {line: dict(cells=int(np.sum(contexts[rows] == i)),
                    targets=len(set(labels[rows[contexts[rows] == i]]) - {'ctrl'})) for i, line in enumerate(LINES)}
            manifest = dict(protocol=protocol, split_seed=seed, training_seed=config['training_seed'],
                expression_path='../../../expression.h5ad', target_partitions=target_roles, panels=panels,
                control_policy='All controls belong to training; inputs always use a fixed same-line training control',
                sampling='Equal line, equal condition within line, equal cell within condition',
                shared_by_variants=['A', 'B', 'C'])
            write_json(directory / 'manifest.json', manifest)
            summaries.append(dict(protocol=protocol, split_seed=seed, panels=panels))
    records = []
    for protocol in PROTOCOLS:
        for seed in config['split_seeds']:
            for name, label, prior in [('A', False, False), ('B', True, False), ('C', True, True)]:
                records.append((f'{protocol}_{name}_split{seed}_train{config["training_seed"]}',
                    protocol, seed, name, label, prior, f'views/{protocol}/seed_{seed}', 'not_started'))
    write_table(output / 'planned_runs.csv', ('run_id', 'protocol', 'split_seed', 'variant',
                'use_cell_line_label', 'use_depmap', 'view_directory', 'status'), records)
    provenance = dict(schema_version=1, name=config['name'], source_release='GEARS-filtered single-cell H5AD; not PIE pseudobulk',
        source_files={line: dict(path=str(sources[line]['path']), sha256=config['source_sha256'][line],
                               url=config['source_urls'][line], shape=list(sources[line]['shape'])) for line in LINES},
        preparation_source_sha256={p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        configuration_sha256=digest(config_path), go_vocabulary_sha256=digest(go_path),
        depmap_gene_vocabulary_sha256=digest(depmap_genes) if depmap_genes else None,
        expression_scale='Stored processed float32 X preserved exactly; upstream normalization target unverified; no second log transform',
        feature_selection='Intersection of released upstream HVG panels by Ensembl ID; no new HVG selection or scaling',
        outcome_exposure='Released HVG panels were preselected upstream; full-data PCA was performed on both releases before splitting. No prior PCA transformations are reused.',
        graph_policy='No expression-derived graph is fitted here. Future graphs and fitted preprocessing must use only permitted training observations for each protocol/split.',
        matching='Exact target symbols in both sources and fixed GO vocabulary; no automatic aliases; perturbation nodes need not be measured outputs',
        low_cell_count_policy='Flag only; retain all eligible paired targets regardless of cell count or DepMap availability',
        counts=dict(shared_targets_before_go=len(shared), paired_targets=len(targets), output_genes=len(var), cells=len(obs),
                    low_cell_count_targets=sum(min(counts[x][g] for x in LINES) < config['low_cell_count_flag'] for g in targets)),
        protocols=summaries,
        remaining='Context-aware GEARS loader, label embeddings, training and evaluation code/DE rankings remain separate work; no metrics or training were run.')
    for line in LINES:
        require(digest(sources[line]['path']) == config['source_sha256'][line], 'Source changed during preparation')
    write_json(output / 'provenance.json', provenance)
    artifacts = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    bundle = hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()
    write_json(output / 'freeze.json', dict(artifact_sha256=artifacts, bundle_sha256=bundle))
    report = verify(output, compare_sources=True)
    write_json(output / 'verification.json', report)
    return report


def verify(directory, compare_sources=False):
    directory = Path(directory).resolve()
    frozen = json.loads((directory / 'freeze.json').read_text())
    artifacts = frozen['artifact_sha256']
    require(hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest() == frozen['bundle_sha256'],
            'Freeze index digest differs')
    for name, expected in artifacts.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory), 'Artifact path escapes dataset')
        require(digest(path) == expected, f'Artifact hash mismatch: {name}')
    config = json.loads((directory / 'config.json').read_text())
    paired = pd.read_csv(directory / 'paired_targets.csv', keep_default_na=False)
    genes = pd.read_csv(directory / 'genes.csv', keep_default_na=False)
    targets = paired['target'].tolist()
    require(len(targets) == len(set(targets)), 'Paired targets are duplicated')
    target_lookup = {g: i for i, g in enumerate(targets)}
    with h5py.File(directory / 'expression.h5ad', 'r') as handle:
        labels = read_column(handle['obs'], 'target')
        contexts = read_column(handle['obs'], 'cell_line_id').astype(np.int8)
        cell_ids = read_index(handle['obs'])
        require(len(cell_ids) == len(set(cell_ids)), 'Duplicate pooled cell identifier')
        require(read_index(handle['var']).tolist() == genes.gene_id.tolist(), 'Expression gene order differs')
        require(tuple(handle['X'].attrs['shape']) == (len(labels), len(genes)), 'Expression shape differs')
        source_rows = read_column(handle['obs'], 'source_row').astype(np.int64)
        table = pd.read_csv(directory / 'cells.csv.gz', keep_default_na=False)
        require(table.cell_id.tolist() == cell_ids.tolist(), 'Cell table and matrix order differ')
        require(table.target.tolist() == labels.tolist(), 'Cell table and matrix targets differ')
        target_ids = np.asarray([target_lookup.get(g, -1) for g in labels], dtype=np.int32)
        for line_id, line in enumerate(LINES):
            observed = pd.Series(labels[contexts == line_id]).value_counts()
            require(all(int(observed[g]) == int(n) for g, n in zip(targets, paired[line + '_ncells'])),
                    'Paired target cell counts differ from the matrix')
        control_rows = np.load(directory / 'input_control_rows.npy', allow_pickle=False)
        require(np.array_equal(control_rows, fixed_controls(labels, contexts, config['control_seed'])),
                'Control schedule cannot be reconstructed')
        require(np.all(labels[control_rows] == 'ctrl') and np.array_equal(contexts[control_rows], contexts),
                'A cell uses an incompatible control')
        if compare_sources:
            provenance = json.loads((directory / 'provenance.json').read_text())
            for line_id, line in enumerate(LINES):
                source = Path(provenance['source_files'][line]['path'])
                require(digest(source) == config['source_sha256'][line], f'{line} source changed')
                pooled = np.flatnonzero(contexts == line_id)
                columns = genes[line + '_source_column'].to_numpy(dtype=np.int64)
                for offset, expected in source_blocks(source, source_rows[pooled], columns):
                    actual = read_csr(handle, int(pooled[offset]), int(pooled[offset]) + expected.shape[0])
                    require(np.array_equal(actual.indptr, expected.indptr)
                            and np.array_equal(actual.indices, expected.indices)
                            and np.array_equal(actual.data, expected.data), 'Expression differs from source')
                print(f'Verified every retained {line} expression value against its source', flush=True)
    views = 0
    for seed in config['split_seeds']:
        target_roles = gene_partition(targets, seed, config['validation_fraction'], config['test_fraction'])
        for protocol in PROTOCOLS:
            panel = directory / 'views' / protocol / f'seed_{seed}'
            manifest = json.loads((panel / 'manifest.json').read_text())
            require(manifest['target_partitions'] == target_roles, 'Target partition cannot be reconstructed')
            require((panel / manifest['expression_path']).resolve() == directory / 'expression.h5ad',
                    'Dataset view expression reference is invalid')
            assigned = cell_partitions(labels, contexts, target_roles, protocol)
            parts = []
            for part in PARTITIONS:
                with np.load(panel / (part + '.npz'), allow_pickle=False) as data:
                    rows = data['rows']
                    require(np.array_equal(rows, np.flatnonzero(assigned == part)), 'Dataset row membership differs')
                    require(np.array_equal(data['input_control_rows'], control_rows[rows]), 'View controls differ')
                    require(np.array_equal(data['cell_line_ids'], contexts[rows]), 'View labels differ')
                    require(np.array_equal(data['target_ids'], target_ids[rows]), 'View target identifiers differ')
                    for i, line in enumerate(LINES):
                        require(np.array_equal(data[line + '_rows'], rows[contexts[rows] == i]), 'Per-line view differs')
                    if part == 'train':
                        require(np.array_equal(data['sampling_weights'], sampling_weights(labels[rows], contexts[rows])),
                                'Training weights differ')
                    parts.append(rows)
            require(np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels))), 'Cells are lost or duplicated')
            views += 1
    return dict(status='passed', bundle_sha256=frozen['bundle_sha256'], artifacts=len(artifacts),
                dataset_views=views, paired_targets=len(targets), expression_genes=len(genes), cells=len(labels),
                exact_source_expression_verified=compare_sources, split_and_control_reconstruction=True)


def export_view(directory, protocol, seed, partition, output, cell_line=None):
    """Materialize an independent H5AD for tools that cannot consume index views."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.exists(), 'Export exists')
    require(not output.is_relative_to(REPOSITORY) and not output.is_relative_to(directory),
            'Export must stay outside the checkout and frozen bundle')
    with np.load(directory / 'views' / protocol / f'seed_{seed}' / (partition + '.npz'), allow_pickle=False) as data:
        rows = data[cell_line + '_rows' if cell_line else 'rows'].copy()
    require(len(rows) > 0, 'Requested dataset view is empty')
    matrix = ad.read_h5ad(directory / 'expression.h5ad', backed='r')
    try:
        obs, var = matrix.obs.iloc[rows].copy(), matrix.var.copy()
    finally:
        matrix.file.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    blocks = (block for _, block in source_blocks(directory / 'expression.h5ad', rows, np.arange(len(var))))
    write_matrix(output, obs, var, blocks)
    return dict(path=str(output), cells=len(rows), genes=len(var), protocol=protocol, split_seed=seed,
                partition=partition, cell_line=cell_line, sha256=digest(output))
