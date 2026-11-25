"""Training-only expression moments and fixed-prior GO graph construction."""

import json
import pickle

import numpy as np
from scipy.sparse import csr_matrix

from replogle_data.io import digest, require

COEXPRESSION_METHOD = 'mean_absolute_within_line_pearson_v1'


def top_edges(scores, k, threshold):
    """Directed top-k edges into each query gene, ties resolved by gene index."""
    sources, destinations, values = [], [], []
    for target, row in enumerate(scores):
        candidates = np.flatnonzero(row > threshold)
        order = np.lexsort((candidates, -row[candidates]))[:k]
        chosen = candidates[order]
        sources.extend(chosen.tolist())
        destinations.extend([target] * len(chosen))
        values.extend(row[chosen].tolist())
    return np.asarray([sources, destinations], dtype=np.int64), np.asarray(values, dtype=np.float32)


def go_graph(genes, annotations, k, threshold):
    require(k >= 1 and 0 <= threshold < 1, 'Invalid GO graph settings')
    require(all(g in annotations and len(annotations[g]) for g in genes), 'GO annotations lack a perturbation node')
    terms = {term: i for i, term in enumerate(sorted({t for g in genes for t in annotations[g]}))}
    row, col = [], []
    for i, gene in enumerate(genes):
        tags = sorted(set(annotations[gene]))
        row.extend([i] * len(tags)); col.extend(terms[t] for t in tags)
    incidence = csr_matrix((np.ones(len(row), dtype=np.int32), (row, col)), shape=(len(genes), len(terms)))
    sizes = np.asarray(incidence.sum(axis=1)).ravel()
    edges, weights = [], []
    for start in range(0, len(genes), 128):
        overlap = (incidence[start:start + 128] @ incidence.T).toarray()
        scores = overlap / (sizes[start:start + 128, None] + sizes[None, :] - overlap)
        edge, weight = top_edges(scores, k, threshold)
        edge[1] += start
        edges.append(edge); weights.append(weight)
    return np.concatenate(edges, axis=1), np.concatenate(weights)


def moments(bundle, part, cross_product=False):
    """Condition moments and, optionally, centered cross-products per cell line."""
    n_genes = len(bundle.genes)
    sums = np.zeros((bundle.n_groups, n_genes), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(bundle.n_groups, dtype=np.int64)
    n_lines = len(bundle.contract['context_names'])
    cross = np.zeros((n_lines, n_genes, n_genes), dtype=np.float64) if cross_product else None
    line_means = np.zeros((n_lines, n_genes), dtype=np.float64)
    line_counts = np.zeros(n_lines, dtype=np.int64)
    rows = bundle.views[part]['rows']
    groups = bundle.groups(part)
    for start in range(0, len(rows), 512):
        x = bundle.expression.read(rows[start:start + 512]).toarray().astype(np.float64)
        group = groups[start:start + len(x)]
        np.add.at(sums, group, x)
        np.add.at(squares, group, x * x)
        counts += np.bincount(group, minlength=bundle.n_groups)
        if cross is not None:
            contexts = bundle.views[part]['cell_line_ids'][start:start + len(x)]
            for line in np.unique(contexts):
                values = x[contexts == line]
                block_mean = values.mean(axis=0)
                centered = values - block_mean
                previous = line_counts[line]
                total = previous + len(values)
                delta = block_mean - line_means[line]
                # Merge centered blocks to avoid subtracting large raw moments.
                cross[line] += centered.T @ centered
                cross[line] += np.outer(delta, delta) * (previous * len(values) / total)
                line_means[line] += delta * (len(values) / total)
                line_counts[line] = total
    return sums, squares, counts, cross


def build_arrays(bundle, config, gene2go_path):
    require(digest(gene2go_path) == config['gene2go_sha256'], 'GO annotation source hash differs')
    # Only load the trusted, hash-pinned official annotation pickle.
    with open(gene2go_path, 'rb') as stream:
        annotations = pickle.load(stream)
    print('Building fixed-prior GO graph', flush=True)
    go_edges, go_weights = go_graph(bundle.perturbations, annotations, config['go_k'] + 1, config['go_threshold'])
    print('Fitting equal-line mean absolute Pearson graph and loss masks using training cells only', flush=True)
    train_sum, train_sq, train_n, cross = moments(bundle, 'train', cross_product=True)
    require(np.all(train_n.reshape(len(cross), -1).sum(axis=1) >= 2),
            'Each line requires at least two observed training cells')
    scores = np.zeros_like(cross[0])
    for covariance in cross:
        scale = np.sqrt(np.maximum(np.diag(covariance), 0))
        denominator = np.outer(scale, scale)
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
        # Preserve GEARS's unsigned edges: take absolute values before averaging.
        scores += np.abs(np.clip(correlation, -1, 1)) / len(cross)
    edges, weights = top_edges(scores, config['coexpression_k'] + 1, config['coexpression_threshold'])
    control_groups = np.asarray([0, bundle.target_count + 1])
    require(np.all(train_n[control_groups] >= 2), 'Each line requires at least two observed training controls')
    controls = train_sum[control_groups] / train_n[control_groups, None]
    control_var = np.maximum(train_sq[control_groups] - train_sum[control_groups] ** 2 / train_n[control_groups, None], 0)
    control_var /= (train_n[control_groups, None] - 1)
    mask = train_sum > 0
    mask[control_groups] = True
    print('Preparing validation-only condition means and DE indices', flush=True)
    val_sum, val_sq, val_n, _ = moments(bundle, 'val')
    means = val_sum / np.maximum(val_n[:, None], 1)
    de = np.full((bundle.n_groups, min(config['de_genes'], len(bundle.genes))), -1, dtype=np.int64)
    for group in np.flatnonzero(val_n):
        line = group // (bundle.target_count + 1)
        n = val_n[group]
        var = np.maximum(val_sq[group] - val_sum[group] ** 2 / n, 0) / max(n - 1, 1)
        se = np.sqrt(var / n + control_var[line] / train_n[control_groups[line]])
        difference = means[group] - controls[line]
        score = np.divide(difference, se, out=np.zeros_like(difference), where=se > 0)
        score[(se == 0) & (difference != 0)] = np.sign(difference[(se == 0) & (difference != 0)]) * np.inf
        de[group] = np.lexsort((np.arange(len(score)), -np.abs(score)))[:de.shape[1]]
    require(np.all(mask[train_n > 0].any(axis=1)), 'A training condition has no measurable expression')
    return dict(go_edges=go_edges, go_weights=go_weights, coexpression_edges=edges, coexpression_weights=weights,
                control_means=controls.astype(np.float32), loss_masks=mask, training_counts=train_n,
                validation_means=means.astype(np.float64), validation_counts=val_n, validation_de=de)
