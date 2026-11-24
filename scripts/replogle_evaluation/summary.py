"""Matched split statistics and per-perturbation similarity figures."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from replogle_data.io import digest, require, write_json
from replogle_data.workflow import REPOSITORY
from replogle_training.dataset import TASKS
from .artifacts import (check_seal, load_arrays, read_json, seal, source_hashes,
                        verify_result, write_rows)
from .metrics import CORE, paired_conditions, score, statistics
from .workflow import output_directory


def similarity_plot(runs, destination, seed, partition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'A': ('#0072B2', 'o'), 'B': ('#D55E00', 's'), 'C': ('#009E73', '^')}
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 4.3), layout='constrained', squeeze=False)
        for axis, (variant, rows) in zip(axes[0], runs.items()):
            valid = [r for r in rows if r['similarity_valid']]
            color, marker = styles[variant]
            axis.plot([-1, 1], [-1, 1], '--', color='#555555', linewidth=1, zorder=1)
            axis.scatter([r['measured_similarity'] for r in valid], [r['predicted_similarity'] for r in valid],
                         color=color, marker=marker, s=28, alpha=.8, edgecolor='white', linewidth=.3, zorder=2)
            axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect='equal',
                     xlabel='Measured K562–RPE1 response Pearson', ylabel='Predicted K562–RPE1 response Pearson',
                     title=f'{variant} | valid {len(valid)}/{len(rows)}')
            mae = float(np.mean([r['similarity_absolute_error'] for r in valid])) if valid else None
            axis.text(.04, .96, f'MAE = {mae:.3f}' if mae is not None else 'MAE undefined', transform=axis.transAxes, va='top')
        fig.suptitle(f'Joint held-out perturbations | {partition} | split {seed}')
        fig.savefig(destination.with_suffix('.png'), dpi=300, facecolor='white')
        fig.savefig(destination.with_suffix('.pdf'), facecolor='white')
        plt.close(fig)


def summarize(root, task, seeds, variants, partition):
    root = Path(root).resolve()
    require(not root.is_relative_to(REPOSITORY) and not REPOSITORY.is_relative_to(root),
            'Evaluation summaries must stay outside the checkout')
    protocol = TASKS[task]
    require(len(set(seeds)) == len(seeds) and len(set(variants)) == len(variants), 'Duplicate summary selections')
    require(variants and seeds, 'Empty summary selection')
    require(not (protocol == 'joint_unseen' and 'Copy-effect' in variants), 'Copy-effect is inapplicable to joint_unseen')
    seed_tag, variant_tag = '-'.join(map(str, sorted(seeds))), '-'.join(variants)
    destination = Path(root) / 'summaries' / protocol / partition / f'seeds_{seed_tag}_variants_{variant_tag}'
    records, split_rows, paired_rows, paired_details, similarity_data = {}, [], [], [], {}
    sources = {}
    for seed in seeds:
        reference_arrays, reference_config = None, None
        for variant in variants:
            directory = output_directory(root, task, seed, variant, partition)
            verify_result(directory)
            sources[str(directory.relative_to(root))] = digest(directory / 'manifest.json')
            arrays = load_arrays(directory / 'scoring_inputs.npz')
            config = read_json(directory / 'evaluation_config.json')
            if reference_arrays is None:
                reference_arrays, reference_config = arrays, config
            else:
                for key in ('truth', 'counts', 'genes', 'targets', 'lines', 'de_indices', 'control_means',
                            'cell_rows', 'assigned_control_rows', 'cell_lines', 'cell_target_ids'):
                    require(np.array_equal(arrays[key], reference_arrays[key]), f'Unmatched scoring population: {key}')
                require(config['identity'] == reference_config['identity'] and
                        config['evaluation_protocol'] == reference_config['evaluation_protocol'], 'Unmatched evaluation policy')
            aggregate, conditions, similarity = score(arrays)
            records[seed, variant] = (aggregate, conditions, similarity, config)
            for scope, values in [('overall', aggregate['metrics']), *aggregate['by_cell_line'].items(), ('cross_line', aggregate['cross_line'])]:
                for metric in (*CORE, 'response_pearson', 'similarity_mae', 'response_difference_mse'):
                    if metric in values:
                        split_rows.append(dict(split_seed=seed, variant=variant, scope=scope, metric=metric, value=values[metric]))
            if similarity:
                similarity_data.setdefault(seed, {})[variant] = similarity
        configurations = [records[seed, v][3]['training_configuration'] for v in variants if v != 'Copy-effect']
        require(not configurations or all(c == configurations[0] for c in configurations), 'Unmatched A/B/C training configurations')
        comparisons = [('B', 'A'), ('C', 'B'), ('B', 'Copy-effect'), ('C', 'Copy-effect')]
        for model, reference in comparisons:
            if model not in variants or reference not in variants:
                continue
            first, second = records[seed, model], records[seed, reference]
            aggregate, details = paired_conditions(first[1], second[1])
            for scope, values in [('overall', aggregate['metrics']), *aggregate['by_cell_line'].items()]:
                for metric in (*CORE, 'response_pearson'):
                    paired_rows.append(dict(split_seed=seed, model=model, reference=reference, scope=scope,
                                            metric=metric, value=values[metric]))
            for row in details:
                paired_details.append(dict(split_seed=seed, model=model, reference=reference, unit='condition', **row))
            if first[2]:
                other = {r['target']: r for r in second[2]}
                require(set(other) == {r['target'] for r in first[2]}, 'Paired similarity target sets differ')
                errors, difference = [], []
                for row in first[2]:
                    partner = other[row['target']]
                    error = (row['similarity_absolute_error'] - partner['similarity_absolute_error']
                             if row['similarity_valid'] and partner['similarity_valid'] else None)
                    diff = row['response_difference_mse'] - partner['response_difference_mse']
                    if error is not None:
                        errors.append(error)
                    difference.append(diff)
                    paired_details.append(dict(split_seed=seed, model=model, reference=reference, unit='cross_line_target',
                                               target=row['target'], similarity_mae=error, response_difference_mse=diff))
                for metric, values in [('similarity_mae', errors), ('response_difference_mse', difference)]:
                    paired_rows.append(dict(split_seed=seed, model=model, reference=reference, scope='cross_line', metric=metric,
                                            value=float(np.mean(values)) if values else None, valid_targets=len(values)))
    expected = dict(protocol=protocol, partition=partition, seeds=sorted(seeds), variants=variants,
                    complete_five_split=sorted(seeds) == [1, 2, 3, 4, 5], sources=sources, source_sha256=source_hashes())
    if destination.exists():
        check_seal(destination)
        require(read_json(destination / 'configuration.json') == expected, 'Existing summary contract differs')
        return dict(status='verified_existing', directory=str(destination))
    destination.mkdir(parents=True)
    write_json(destination / 'configuration.json', expected)
    write_rows(destination / 'split_metrics.csv', split_rows)
    write_rows(destination / 'paired_split_differences.csv', paired_rows,
               ['split_seed', 'model', 'reference', 'scope', 'metric', 'value', 'valid_targets'])
    fields = list(dict.fromkeys(key for row in paired_details for key in row))
    write_rows(destination / 'paired_per_target_differences.csv', paired_details, fields or ['target'])
    statistics_rows = []
    for rows, keys, filename in [(split_rows, ('variant', 'scope', 'metric'), 'metric_summary.csv'),
                                 (paired_rows, ('model', 'reference', 'scope', 'metric'), 'paired_difference_summary.csv')]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in keys)].append(row['value'])
        summary_rows = [dict(zip(keys, identity), **statistics(values)) for identity, values in sorted(grouped.items())]
        write_rows(destination / filename, summary_rows)
        if filename == 'metric_summary.csv':
            statistics_rows = summary_rows
    for seed, runs in similarity_data.items():
        write_rows(destination / f'similarity_split{seed}.csv',
                   [dict(variant=variant, **row) for variant, values in runs.items() for row in values])
        similarity_plot(runs, destination / f'similarity_split{seed}', seed, partition)
    if similarity_data:
        write_json(destination / 'figure_metadata.json', dict(
            source_tables=[f'similarity_split{seed}.csv' for seed in similarity_data],
            source_evaluation_manifest_sha256=sources, formats=['png', 'pdf'], png_dpi=300,
            x='Measured K562–RPE1 response Pearson', y='Predicted K562–RPE1 response Pearson',
            limits=[-1.05, 1.05], identity_reference=True, smoothing=False,
            missing='Undefined pairs retained in CSV, omitted from scatter, counted in panel titles',
            sampling_unit='One held-out target within a split; splits overlap',
            description='Separate variant panels compare measured and predicted cross-line response similarity. Closeness to the diagonal indicates similarity recovery; underlying per-target values are provided in CSV.'))
    text = ['# Replogle evaluation', '', f'Protocol: {protocol}; partition: {partition}; split seeds: {sorted(seeds)}.', '',
            'Each split weights perturbations equally, then available cell lines equally. SD is sample SD across splits; panels overlap and are not biological replicates.', '',
            'Expression metrics use raw frozen-input predictions. Response/context metrics subtract the actual assigned-input mean for neural models and the destination training-control mean for Copy-effect. Measured responses subtract the same-line full training-control mean.', '',
            '| Variant | Scope | Metric | Mean | Sample SD | Valid splits |', '|---|---|---|---:|---:|---:|']
    def number(value):
        return 'undefined' if value is None else f'{value:.6g}'
    text.extend(f'| {r["variant"]} | {r["scope"]} | {r["metric"]} | {number(r["mean"])} | {number(r["sample_sd"])} | {r["n"]} |'
                for r in statistics_rows)
    text += ['', 'Similarity figures: one point per held-out perturbation, measured versus predicted cross-line response Pearson, identity line, identical axes. Undefined pairs are omitted from the plot and counted in the panel title; CSV retains every target. No across-target correlation is required.', '',
             'Paired response/similarity comparisons use common-valid targets. Within-line response correlations and cross-line similarity are descriptive; technical screen differences can contribute. Upstream HVG selection and previous full-data PCA exposure remain limitations.', '',
             'Lower MSE/similarity error and higher Pearson are favorable. Difference tables are model minus reference. Copy-effect is a training-only source-response baseline for transfer tasks; source-line training observations are not held-out destination outcomes.', '']
    (destination / 'report.md').write_text('\n'.join(text))
    seal(destination)
    return dict(status='complete', directory=str(destination), complete_five_split=expected['complete_five_split'])
