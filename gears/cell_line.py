"""Cell-line labels, matched controls, and context-specific training helpers."""
import numpy as np
import torch
from scipy.sparse import issparse


def validate_cell_line_names(names):
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('cell_line_names must be a nonempty ordered list of unique strings')
    return list(names)


def validate_labels(adata, key, names=None):
    if key not in adata.obs:
        raise ValueError(f'Missing cell-line label column: {key}')
    labels = adata.obs[key].tolist()
    if not labels or any(not isinstance(n, str) or not n.strip() for n in labels):
        raise ValueError('Every cell must have a nonempty string cell-line label')
    names = validate_cell_line_names(sorted(set(labels)) if names is None else names)
    if set(labels) != set(names):
        raise ValueError('cell_line_names must match the observed cell-line labels exactly')
    return names


def dense(values):
    return values.toarray() if issparse(values) else np.asarray(values)


def prepare_cell_lines(pert_data, key='cell_type', names=None, seed=0):
    """Rebuild graphs in memory; never reuse the legacy pooled-control cache.

    Existing condition-level splits keep all lines of a perturbation together.
    Each graph carries its line string and vocabulary ID. DE rankings must be
    supplied for each line-condition, on the same expression-gene axis.
    """
    adata = pert_data.adata
    names = validate_labels(adata, key, names)
    labels = np.asarray(adata.obs[key])
    conditions = np.asarray(adata.obs['condition'])
    if 'condition_name' not in adata.obs:
        raise ValueError('Cell-line data requires condition_name metadata')
    metadata = adata.obs[[key, 'condition', 'condition_name']].drop_duplicates()
    if (metadata.duplicated([key, 'condition']).any()
            or metadata.duplicated('condition_name').any()):
        raise ValueError('condition_name must identify exactly one cell-line/condition pair')
    rankings = adata.uns.get('rank_genes_groups_cov_all', {})
    gene_indices = {gene: i for i, gene in enumerate(adata.var_names)}
    if len(gene_indices) != adata.n_vars:
        raise ValueError('Expression gene identifiers must be unique')
    width = min(20, adata.n_vars)
    if width < 2:
        raise ValueError('Cell-line evaluation requires at least two expression genes')
    rng = np.random.default_rng(seed)
    saved_controls = getattr(pert_data, 'control_rows', None)
    if saved_controls is not None:
        saved_controls = np.asarray(saved_controls)
        if (saved_controls.shape != (adata.n_obs,) or saved_controls.dtype.kind not in 'iu'
                or np.any(saved_controls < 0) or np.any(saved_controls >= adata.n_obs)):
            raise ValueError('Frozen control rows do not match the cell axis')
        if np.any(conditions[saved_controls] != 'ctrl') or np.any(labels[saved_controls] != labels):
            raise ValueError('Frozen controls must belong to the target cell line')
    dataset = {p: [] for p in dict.fromkeys(conditions)}
    control_rows = np.empty(adata.n_obs, dtype=np.int64)
    for line_id, line in enumerate(names):
        controls = np.flatnonzero((labels == line) & (conditions == 'ctrl'))
        if not len(controls):
            raise ValueError(f'No controls available for cell line {line}')
        for condition in dict.fromkeys(conditions[labels == line]):
            rows = np.flatnonzero((labels == line) & (conditions == condition))
            if condition == 'ctrl':
                indices, pert_idx, paired = [-1] * width, None, rows
            else:
                full_name = adata.obs['condition_name'].iloc[rows[0]]
                top = list(rankings.get(full_name, []))[:width]
                if len(top) != width or len(set(top)) != width or any(g not in gene_indices for g in top):
                    raise ValueError(f'Missing or invalid per-line DE ranking for {full_name}')
                indices = [gene_indices[g] for g in top]
                pert_idx = pert_data.get_pert_idx(condition)
                if not pert_idx:
                    raise ValueError(f'No perturbation graph nodes for {condition}')
                paired = (saved_controls[rows] if saved_controls is not None
                          else rng.choice(controls, len(rows), replace=True))
            control_rows[rows] = paired
            xs, ys = dense(adata.X[paired]), dense(adata.X[rows])
            for i in range(len(rows)):
                graph = pert_data.create_cell_graph(xs[i:i+1], ys[i:i+1], indices, condition, pert_idx)
                graph.cell_line_id = torch.tensor([line_id], dtype=torch.long)
                graph.cell_line = line
                dataset[condition].append(graph)
    pert_data.cell_line_key = key
    pert_data.cell_line_names = names
    pert_data.cell_line_control_seed = seed
    pert_data.cell_line_control_rows = control_rows
    pert_data.dataset_processed = dataset
    pert_data.ctrl_adata = adata[conditions == 'ctrl']
    pert_data.gene_names = adata.var.gene_name
    # A rebuilt vocabulary/control schedule invalidates already-created loaders.
    if hasattr(pert_data, 'dataloader'):
        del pert_data.dataloader


def context_statistics(adata, key, names, device):
    """Line-specific control means and nonzero masks, without pooled metadata."""
    controls, filters = {}, {}
    for line in names:
        subset = adata[np.asarray(adata.obs[key] == line)]
        ctrl = subset[subset.obs['condition'] == 'ctrl']
        if not ctrl.n_obs:
            raise ValueError(f'No controls available for cell line {line}')
        controls[line] = torch.as_tensor(np.asarray(ctrl.X.mean(axis=0)).reshape(-1),
                                        dtype=torch.float32, device=device)
        filters[line] = {}
        for condition in subset.obs['condition'].unique():
            if condition == 'ctrl':
                continue
            values = subset[subset.obs['condition'] == condition].X
            indices = np.flatnonzero(np.asarray(values.mean(axis=0)).reshape(-1) != 0)
            if not len(indices):
                raise ValueError(f'No nonzero target genes for {line}/{condition}')
            filters[line][condition] = indices
    return controls, filters


def cell_line_loss(pred, y, perts, lines, controls, filters, direction_lambda,
                   logvar=None, uncertainty_reg=1):
    """Equal weight for each observed (line, condition), as in GEARS loss."""
    from .utils import loss_fct, uncertainty_loss_fct
    lines, perts = np.asarray(lines), np.asarray(perts)
    total, count = pred.sum() * 0, 0
    for line in np.unique(lines):
        indices = np.flatnonzero(lines == line)
        weight = len(set(perts[indices]))
        kwargs = dict(ctrl=controls[line], dict_filter=filters[line], direction_lambda=direction_lambda)
        if logvar is None:
            loss = loss_fct(pred[indices], y[indices], perts[indices], **kwargs)
        else:
            loss = uncertainty_loss_fct(pred[indices], logvar[indices], y[indices], perts[indices],
                                        reg=uncertainty_reg, **kwargs)
        total = total + weight * loss
        count += weight
    return total / count


def results_for_cell_line(results, line):
    indices = np.flatnonzero(results['cell_line'] == line)
    return {key: values[indices] for key, values in results.items() if key != 'cell_line'}
