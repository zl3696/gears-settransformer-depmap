# Paired Replogle single-cell data

## Scope and source

The user selected individual-cell training for pooled variants A (GEARS), B
(GEARS plus cell-line labels), and C (the same labels plus DepMap), and approved
both known-perturbation transfer directions on 24 November 2026. This workflow
prepares data only. It does not train a model, select checkpoints, fit expression
graphs, compute DE rankings, or evaluate predictions.

Use the GEARS-filtered K562 and RPE1 H5AD releases from Harvard Dataverse files
7458695 and 7458694. `configs/replogle_data.json` pins their H5AD hashes. These
are different releases from the PIE Hugging Face condition-mean dataset; do not
transfer PIE's target counts, expression scale, or DE tables onto these cells.

`scripts/replogle_data.py` is a standalone CLI. Its implementation lives in
`scripts/replogle_data/` and imports no GEARS model. Dependencies are NumPy,
SciPy, pandas, h5py, and anndata; the existing course GEARS environment has them.

## Population and preprocessing

- Intersect target symbols exactly across lines, then require membership in the
  fixed external GO vocabulary. Do not automatically infer aliases. Retain all
  cells of every eligible condition, including low-count conditions.
- Match measured expression genes by unique Ensembl ID; require agreeing gene
  symbols and freeze a sorted shared gene axis. The target/GO vocabulary is a
  separate axis: a perturbed gene need not be a measured expression output.
  K562 contains two `TBCE` symbols at different Ensembl IDs; match by ID before
  checking symbol uniqueness on the shared panel. Never deduplicate source
  columns by symbol or merge their values automatically.
- Preserve float32 values from stored sparse `X` exactly. Do not renormalize,
  log-transform again, impute unmeasured genes, or reuse previous PCA fits.
  The upstream normalization target remains unverified.
- Released expression panels were already selected upstream as HVGs; previous
  full-dataset PCA exposed outcomes before splitting. Record both limitations.
  The new intersection uses identifiers, not newly fitted expression statistics.
- Keep line labels, source barcodes/rows/labels, and namespaced cell IDs. A
  `condition_key` such as `K562::GENE` identifies a context-specific condition;
  `condition=GENE+ctrl` alone is insufficient when pooling the lines.
- `paired_targets.csv` records cell counts, aliases, low-count flags (below 30
  in either line), and optional exact DepMap coverage. These flags do not filter
  the target set. Do not restrict A/B observations to genes available to C.

## Frozen splits and controls

Five target-level splits use PCG64 seeds 1–5. Validation and test each receive
`floor(0.15 * N)` targets; training receives the remainder. Shuffle the sorted
target names. Every source-label alias and every cell of a context-specific
target stays together. All variants use the same views and training seed 101.

| Protocol | Source-line observations | Destination-line observations |
|---|---|---|
| `joint_unseen` | Both lines follow the same train/val/test gene partition | Same partition |
| `k562_to_rpe1` | All selected K562 targets in training | RPE1 follows the gene partition |
| `rpe1_to_k562` | All selected RPE1 targets in training | K562 follows the gene partition |

Both transfer protocols have both contexts represented in training. They test a
known perturbation in a known context; they do not test an unseen cell line.
Their validation/test panels contain only the destination line. Keep all three
protocols independent: source-line outcomes intentionally exposed in a transfer
run must not leak into main-run fitting, preprocessing, or adaptive tuning.
Freeze model choices before examining final tests across protocols or splits.

All controls belong to training. Each perturbed cell has one fixed, randomly
assigned control from the same line, using seeds 1000/1001; controls map to
themselves. Validation/test inputs may reuse this observed training-control
bank. The supplied training weights select a line uniformly, a condition
(including control) uniformly within that line, and a cell uniformly within
that condition. The [training consumer](replogle-training.md) uses these weights
for replacement sampling and keeps each epoch's original training-cell budget.

## Bundle and consumers

`expression.h5ad` stores all retained cells once, with shared expression columns
and minimal metadata. It copies no layers, released DE rankings, PCA outputs,
or other `uns` entries. `cells.csv.gz`, `genes.csv`, `paired_targets.csv`,
`excluded_targets.csv`, and `perturbation_nodes.csv` describe its axes and audit.

Each `views/<protocol>/seed_<n>/` has a manifest and `train.npz`, `val.npz`, and
`test.npz`. NPZ fields include global expression `rows`, `input_control_rows`,
`cell_line_ids`, `target_ids` (-1 for control), and per-line global row indices.
Training also includes `sampling_weights`. Load NPZ with `allow_pickle=False`;
take expression rows from the shared H5AD. Preserve row order when associating
weights, targets, labels, and control inputs. Do not interpret index views as
additional biological observations.

`target_ids` index `paired_targets.csv`, not the larger GO graph. The training
loader resolves the target symbol through `perturbation_nodes.csv` to obtain
GEARS node indices. Read expression in batches; do not densify the pooled matrix or
prebuild every PyG cell graph in memory.

`planned_runs.csv` enumerates the 45 future A/B/C runs; all start as
`not_started`. `freeze.json` pins payload hashes; `verification.json` records
integrity checks. Runtime data stays outside Git. The CLI never overwrites an
existing dataset or export. A partial directory without a passing verification
record must not be used for training.

The original single-line GEARS loader does not consume this format. Use the
separate [Replogle training entry](replogle-training.md) for context-aware A/B/C,
training-only graphs/masks, validation-only checkpoint selection, and fixed
DepMap projection. Final-test prediction scoring remains separate work. The
historical Norman manifests and numerical sources are unchanged.

## Commands from any directory

Set `repo_dir`, `runtime_dir`, and `python_bin` to absolute locations. In the
course workspace, `runtime_dir` is `Codex Outputs/`; its `.venvs/gears/bin/python`
is the existing interpreter. Source files remain read-only.

```bash
"$python_bin" "$repo_dir/scripts/replogle_data.py" prepare \
  --k562 "$runtime_dir/data/replogle_k562_essential/perturb_processed.h5ad" \
  --rpe1 "$runtime_dir/data/replogle_rpe1_essential/perturb_processed.h5ad" \
  --output-dir "$runtime_dir/outputs/replogle-paired-singlecell-v1" \
  --depmap-genes "$runtime_dir/tmp/morph-gears-review/depmap_genes.json"
"$python_bin" "$repo_dir/scripts/replogle_data.py" verify \
  --data-dir "$runtime_dir/outputs/replogle-paired-singlecell-v1" --compare-sources
"$python_bin" -m unittest discover -s "$repo_dir/tests" -p 'test_replogle_data.py' -v
```

Preparation checks source hashes before and after writing and compares every
retained sparse expression value, row, and gene column against the source.
Verification reconstructs all partitions, controls, target IDs, and weights.
Without `--compare-sources`, verification needs only the portable bundle.

Tools requiring a separate H5AD can materialize any view, optionally by line:

```bash
"$python_bin" "$repo_dir/scripts/replogle_data.py" export \
  --data-dir "$runtime_dir/outputs/replogle-paired-singlecell-v1" \
  --protocol joint_unseen --seed 1 --partition test --cell-line rpe1 \
  --output "$runtime_dir/outputs/replogle-exports/rpe1_test.h5ad"
```

An exported validation/test H5AD contains target observations only; its control
bank and input schedule remain in the frozen bundle. Never sample inputs from
another line or silently recalculate control assignments during loading.
