"""Condition-weighted GEARS metrics and control-relative context endpoints."""

import warnings

import numpy as np

from replogle_data.io import require

LINES = ('K562', 'RPE1')
CORE = ('mse', 'mse_de', 'pearson', 'pearson_de')


def correlation(first, second):
    """Return null for a constant vector, preserving missingness in JSON/CSV."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    require(first.shape == second.shape and first.size >= 2, 'Invalid correlation axes')
    require(np.isfinite(first).all() and np.isfinite(second).all(), 'Nonfinite correlation input')
    a, b = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denominator, -1, 1)) if denominator > 0 else None


def de_indices(means, variances, counts, controls, control_variances, control_counts, lines, number):
    """The training validator's absolute Welch ranking, with stable index ties."""
    result = []
    for mean, variance, n, line in zip(means, variances, counts, lines):
        error = np.sqrt(variance / n + control_variances[line] / control_counts[line])
        difference = mean - controls[line]
        score = np.divide(difference, error, out=np.zeros_like(difference), where=error > 0)
        special = (error == 0) & (difference != 0)
        score[special] = np.sign(difference[special]) * np.inf
        result.append(np.lexsort((np.arange(len(score)), -np.abs(score)))[:min(number, len(score))])
    return np.asarray(result, dtype=np.int64)


def summarize_conditions(rows, keys=(*CORE, 'response_pearson')):
    by_line = {}
    for line in LINES:
        selected = [row for row in rows if row['cell_line'] == line]
        if not selected:
            continue
        scores = {'conditions': len(selected), 'valid_counts': {}}
        for key in keys:
            values = [r[key] for r in selected if r[key] is not None]
            scores[key] = float(np.mean(values)) if values else None
            scores['valid_counts'][key] = len(values)
        by_line[line] = scores
    overall = {}
    for key in keys:
        values = [row[key] for row in by_line.values()]
        # A missing whole line must not silently change the equal-line estimand.
        overall[key] = float(np.mean(values)) if all(v is not None for v in values) else None
    return dict(metrics=overall, by_cell_line=by_line)


def score(arrays):
    """Reuse the actual GEARS scorer, checking each result independently."""
    from gears.inference import compute_metrics

    prediction, truth, indices = (arrays[k] for k in ('prediction', 'truth', 'de_indices'))
    n, genes = truth.shape
    require(prediction.shape == truth.shape and n > 0 and genes >= 2, 'Scoring shape mismatch')
    require(indices.ndim == 2 and len(indices) == n and indices.shape[1] >= 2, 'Invalid DE shape')
    require(np.issubdtype(indices.dtype, np.integer) and np.all((indices >= 0) & (indices < genes)), 'Invalid DE indices')
    require(all(len(set(row)) == len(row) for row in indices), 'Duplicate DE indices')
    require(all(np.isfinite(arrays[k]).all() for k in ('prediction', 'truth', 'input_means', 'control_means')), 'Nonfinite scoring arrays')
    names = [f'{LINES[int(line)]}::{target}' for line, target in zip(arrays['lines'], arrays['targets'])]
    require(len(set(names)) == n, 'Duplicate context-specific condition')
    pred_de = np.take_along_axis(prediction, indices, axis=1)
    truth_de = np.take_along_axis(truth, indices, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, historical = compute_metrics(dict(pert_cat=np.asarray(names), pred=prediction, truth=truth,
                                             pred_de=pred_de, truth_de=truth_de))
    rows = []
    observed_response = truth - arrays['control_means'][arrays['lines']]
    predicted_response = prediction - arrays['input_means']
    for i, name in enumerate(names):
        row = dict(condition=name, cell_line=LINES[int(arrays['lines'][i])], target=str(arrays['targets'][i]),
                   cells=int(arrays['counts'][i]), low_cell_count=bool(arrays['counts'][i] < arrays['low_cell_count']),
                   control_offset_rms=float(np.sqrt(np.mean((arrays['input_means'][i] - arrays['control_means'][arrays['lines'][i]]) ** 2))))
        for suffix, a, b in (('', prediction[i], truth[i]), ('_de', pred_de[i], truth_de[i])):
            corr = correlation(a, b)
            independently = {'mse' + suffix: float(np.mean((a - b) ** 2)), 'pearson' + suffix: 0. if corr is None else corr}
            for key, value in independently.items():
                require(np.isclose(historical[name][key], value, rtol=1e-10, atol=1e-12), 'Independent GEARS score differs')
                row[key] = float(historical[name][key])
            row['pearson' + suffix + '_undefined_to_zero'] = corr is None
        row['response_pearson'] = correlation(predicted_response[i], observed_response[i])
        row['response_pearson_undefined'] = row['response_pearson'] is None
        rows.append(row)
    similarity = []
    if set(arrays['lines'].tolist()) == {0, 1}:
        lookup = {(int(line), str(target)): i for i, (line, target) in enumerate(zip(arrays['lines'], arrays['targets']))}
        left = {target for line, target in lookup if line == 0}
        right = {target for line, target in lookup if line == 1}
        require(left == right, 'Cross-line target sets differ')
        for target in sorted(left):
            a, b = lookup[0, target], lookup[1, target]
            measured = correlation(observed_response[a], observed_response[b])
            predicted = correlation(predicted_response[a], predicted_response[b])
            valid = measured is not None and predicted is not None
            difference = (predicted_response[a] - predicted_response[b]) - (observed_response[a] - observed_response[b])
            similarity.append(dict(target=target, k562_cells=int(arrays['counts'][a]), rpe1_cells=int(arrays['counts'][b]),
                measured_similarity=measured, predicted_similarity=predicted, similarity_valid=valid,
                similarity_absolute_error=abs(predicted - measured) if valid else None,
                response_difference_mse=float(np.mean(difference ** 2))))
    aggregate = summarize_conditions(rows)
    aggregate['cross_line'] = {}
    if similarity:
        values = [r['similarity_absolute_error'] for r in similarity if r['similarity_valid']]
        aggregate['cross_line'] = dict(targets=len(similarity), similarity_valid_targets=len(values),
            similarity_mae=float(np.mean(values)) if values else None,
            response_difference_mse=float(np.mean([r['response_difference_mse'] for r in similarity])))
    return aggregate, rows, similarity


def paired_conditions(first, reference):
    """Common-valid pairing, with equal condition weighting followed by lines."""
    lookup = {r['condition']: r for r in reference}
    require(set(lookup) == {r['condition'] for r in first}, 'Paired condition sets differ')
    rows = []
    for row in first:
        other = lookup[row['condition']]
        result = {k: row[k] for k in ('condition', 'cell_line', 'target')}
        for key in (*CORE, 'response_pearson'):
            result[key] = row[key] - other[key] if row[key] is not None and other[key] is not None else None
        rows.append(result)
    return summarize_conditions(rows), rows


def statistics(values):
    finite = [v for v in values if v is not None]
    return dict(n=len(finite), mean=float(np.mean(finite)) if finite else None,
                sample_variance=float(np.var(finite, ddof=1)) if len(finite) > 1 else None,
                sample_sd=float(np.std(finite, ddof=1)) if len(finite) > 1 else None)
