# Optional cell-line label embeddings

## Scope and model

`feature/cell-line-embedding` builds on DepMap commit `b24cd81`.
`GEARS.model_initialize(use_cell_line=True)` learns one hidden-size vector per
known cell line. It is independent of `use_depmap` and of all three `pert_agg`
modes (`sum`, `set_transformer`, `residual_set_transformer`).

For each perturbed cell, after aggregating its perturbation embeddings and
before the shared nonlinear decoder:

```text
decoder_input[cell, gene] = base_gene_embedding[gene]
                          + aggregated_perturbation[cell]
                          + cell_line_embedding[label[cell]]
```

The label vector is added once per cell, not once per perturbed gene. Controls
bypass this added perturbation term. The nonlinear decoder can learn interactions
between a perturbation and its line. DepMap still supplies the same fixed fitness
profile for a gene in either line; it does not identify the current cell line.

Label weights start at zero. Enabled and disabled models have identical initial
predictions, shared parameters, and random-number state under the same seed;
label weights receive gradients immediately. All parameters train jointly.
Training retains GEARS BatchNorm, so batch composition still affects outputs
in training mode. Evaluation uses its stored statistics.

`use_cell_line=False` is the default and adds no parameters or label requirement
to the model. Old checkpoints without the switch remain unconditioned. Label
embeddings support known training cell lines, not zero-shot unseen cell lines.

## Data and controls

`gears/cell_line.py` owns validation, labeled graph preparation, matched controls,
and context-specific loss helpers. Use the same prepared data for every ablation,
including the baseline with labels disabled:

```python
from gears import GEARS, PertData

data = PertData('/absolute/external/data')
data.load(data_path='/absolute/external/data/processed_joint_dataset',
          cell_line_key='cell_type', cell_line_names=['K562', 'RPE1'],
          cell_line_seed=15)
data.prepare_split(split='custom', seed=1,
                   split_dict_path='/absolute/external/condition_split.pkl')
data.get_dataloader(batch_size=32)
model = GEARS(data, device='cuda')
model.model_initialize(use_cell_line=True, use_depmap=False, pert_agg='sum', seed=101)
model.train(epochs=20, run_test=False)
predictions = model.predict([['GENE_SYMBOL']], cell_line='K562')
```

Replace the example dataset, split and gene with actual inputs. The dataset
directory must contain `perturb_processed.h5ad` with:

- A shared, identically ordered expression-gene axis and `var['gene_name']`.
- `obs['condition']` in GEARS notation, a nonempty label for every cell in the
  chosen column, and controls (`condition == 'ctrl'`) in each line.
- A `condition_name` uniquely identifying each (cell line, condition).
- `uns['rank_genes_groups_cov_all']` with per-line DE rankings keyed by
  `condition_name`. Each non-control needs at least min(20, number of genes)
  unique measured gene IDs. Rankings define evaluation genes and must not be
  used to fit preprocessing or graphs on held-out target expression.

Alternatively, after loading/processing data, call
`data.prepare_cell_lines('cell_type', ['K562', 'RPE1'], seed=15)` before creating
loaders. This rebuilds graphs in memory and invalidates existing loaders.
Both entry points bypass the legacy pooled-control graph cache. The sorted
observed labels are used if an explicit ordered vocabulary is omitted.

Every graph carries `cell_line` and a `cell_line_id` tensor. Sampling uses a local
seeded generator and only controls of the same line. `cell_line_control_rows`
records the pairing; preserve the ordered input cells, seed, and assignments
with experiment artifacts. Supplied frozen assignments must match the target
line. Loss control means and nonzero masks are computed separately per line;
only the current training contexts' masks enter the loss.

## Comparison and evaluation

| Model | `use_cell_line` | `use_depmap` |
| --- | --- | --- |
| GEARS baseline | false | false |
| GEARS with labels | true | false |
| GEARS with labels and DepMap | true | true |

For the last row also pass `depmap_path='/absolute/features.npz'`; see
[DepMap](depmap.md). Use `pert_agg='sum'` for the primary A/B/C comparison, or
select another aggregation method independently. Keep the same data, controls,
splits, graph fitting, training budget and seed for all variants.

Training requires perturbed training examples in every enabled label category.
Loss averages over (line, perturbation) contexts. `evaluate()` retains labels;
`compute_metrics()` uses context means and returns `(line, perturbation)` keys.
`normalized_mse_de()` requires a dictionary of line-specific control means.
This prevents opposite errors in K562 and RPE1 from cancelling in a pooled mean.

For prepared data, `model.test()` returns core metrics and saves `test_results`,
`test_metrics`, `test_pert_metrics`, and `test_metrics_by_cell_line`. It does not
run the inherited single-line GI/dropout/subgroup reports. Multi-line
`GI_predict()` and `plot_perturbation()` reject ambiguous use. Prediction requires
an explicit `cell_line` for multi-line data and caches results separately by line,
even when embeddings are disabled. Unknown labels fail rather than map to zero.

**Split boundary:** existing `prepare_split()` still partitions perturbation
conditions; all lines of that condition stay together. This branch does not
generate the proposed held-out gene–cell-line combinations. Joint Replogle
assembly, train-fitted preprocessing/graphs, versioned gene–line split manifests,
and the per-perturbation cross-line response-similarity report remain TODO.
Do not describe an ordinary condition split as that cross-line experiment.
The existing Norman launchers and frozen run plans are unchanged.

## Checkpoints and tests

`model.pt` saves learned label weights. `config.pkl` saves `use_cell_line` and
`cell_line_metadata` (label column, ordered names, expression genes, perturbation
genes), including for the unconditioned baseline trained on prepared data.
Prepare the same vocabulary before `load_pretrained()`. Reordered labels/genes
or missing metadata fail explicitly; no silent reassignment of embedding rows.
DepMap-enabled checkpoints remain independent of the original feature file.

From any directory, with repository requirements and pytest installed:

```bash
repo_dir='/absolute/path/to/gears-settransformer-depmap'
PYTHONPATH="$repo_dir" PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -p no:cacheprovider "$repo_dir/tests/test_cell_line_embedding.py" -q
```

Tests cover all aggregation modes with DepMap on/off, zero initialization and
gradients, active label effects, controls, set/batch invariance in evaluation,
dense/sparse data, line-specific losses/metrics/caches, short joint training,
uncertainty, vocabulary validation, legacy loading and checkpoint restoration.
The inherited uncertainty loss now uses out-of-place accumulation so CPU autograd
can train it; its numerical formula is unchanged. No new dependencies are needed.
Synthetic checks establish implementation behavior, not biological accuracy.
