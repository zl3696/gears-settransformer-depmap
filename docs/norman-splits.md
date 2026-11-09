# Norman condition splits

The versioned snapshot in `splits/norman/` is shared by original GEARS and all
three variants. It contains metadata and custom split files for the processed
Norman K562 CRISPRa release, without expression matrices or model checkpoints.

## Population and protocol

- Source: GEARS Harvard Dataverse file 6154020; original H5AD SHA-256 and GO
  annotation hashes are recorded in `provenance.json`.
- Default GO filtering retains 89,357 cells, 5,045 expression genes, 102 single
  conditions, 128 double conditions, and 7,353 controls. Both singles exist for
  every retained pair. The default perturbation-node panel contains 9,853 genes.
- All 54,931 single/control cells stay in every training partition. Only pair
  conditions are split: **76 training / 26 validation / 26 test**.
- Canonicalize pair orientation and single/control orientation first. All 277
  source labels map to 231 canonical conditions; aliases never cross partitions.
- For each seed 1–5, apply one NumPy `Generator(PCG64(seed))` permutation to the
  same sorted 128-pair list. Positions 0–25 are test, 26–51 validation, and 52–127
  training. Saved condition lists are sorted within each partition.
- Intended training seed: 101 for every model/split. Control seed: 1000, using
  one draw with replacement per non-control target in retained source order.
  Control targets map to themselves and consume no draw. Evaluation reuses the
  same assignments; these independent cells are not before/after matches.

| Split seed | Training cells | Validation cells | Test cells |
|---:|---:|---:|---:|
| 1 | 76,724 | 5,419 | 7,214 |
| 2 | 74,686 | 7,770 | 6,901 |
| 3 | 75,199 | 7,818 | 6,340 |
| 4 | 74,101 | 7,220 | 8,036 |
| 5 | 76,086 | 6,468 | 6,803 |

Keep the source's processed expression values and gene order unchanged. The
source's `A549` field is a placeholder; the study used K562. Prior dataset-wide
descriptive PCA exposure is recorded in provenance and must not become model
preprocessing. No expression arrays or released DE rankings were decoded during
split preparation.

## Files and validation

- `seed_N/gears_custom_split.pkl`: original source-label lists accepted by
  `PertData.prepare_split(split="custom", seed=N, split_dict_path=...)`.
- `seed_N/manifest.json`, `condition_assignments.csv`, `cell_assignments.csv.gz`:
  canonical/source condition lists, dependency hashes, and per-cell partitions.
- `conditions.csv`, `source_condition_labels.csv`, `cells.csv.gz`: frozen metadata
  from which the portable script reconstructs the assignments. Source/retained
  row indices are zero-based. `excluded_cells.csv.gz` records 1,848 excluded cells.
- `expression_genes.csv`, `perturbation_nodes.csv`: ordered model axes.
- `control_assignments.csv.gz`, `evaluation_control_schedule.csv.gz`: fixed
  inputs for all retained targets and the 34,426 pair targets respectively.
- `graph_metadata.json` in each seed folder records allowed fitting populations,
  not fitted edges. Upstream coexpression uses the training controls/singles.
- `planned_runs.csv`: 20 intended neural-model fits; all remain `not_started`.
- `freeze.json`: 37 payload hashes and bundle digest
  `e1e90cbac40b2ea90695b761af5c5f4103baf00ff6380dcd0e3571c65891ec34`.
- `validation_report.json`: original preparation audit, including both original
  and team GEARS custom-loader checks. Its source revisions describe preparation
  provenance and are not requirements to check out those revisions for portable
  verification. `summary.json` contains population and split counts.

`scripts/norman_splits.py` verifies every payload hash and independently rebuilds
all five split pickles, condition/cell assignments, and both control schedules.
Rebuilt files must match the frozen bytes. The script needs NumPy; NumPy 1.26.4
and Python 3.12 were used for the original freeze. Gzip byte identity can also
depend on the Python/zlib version. Raw H5AD extraction and GO filtering are the
recorded upstream preparation step; this script rebuilds from frozen metadata.

## Commands from any working directory

Set absolute paths for your checkout and an external runtime directory. In the
course workspace, use `Codex Outputs/tmp` for scratch and `Codex Outputs/outputs`
for rebuilt files. The runtime directory must already exist.

```bash
repo_dir='/absolute/path/to/gears-settransformer-depmap'
runtime_dir='/absolute/path/outside/the/checkout/runtime'
python "$repo_dir/scripts/norman_splits.py" verify --temp-dir "$runtime_dir"
TMPDIR="$runtime_dir" python -m unittest discover -s "$repo_dir/tests" -p 'test_norman_splits.py' -v
python "$repo_dir/scripts/norman_splits.py" generate --temp-dir "$runtime_dir" --output-dir "$runtime_dir/norman-splits-rebuilt"
```

Generation refuses to replace an existing destination or write runtime outputs
inside this source checkout. The committed snapshot is protocol source material;
keep all subsequent caches, expression data, and model results outside Git.

## Training boundaries

Training settings live in `configs/norman_training.json`, outside the frozen
snapshot. It pins the snapshot by bundle and `config.json` digests and freezes:

- training seed 101 for every model and split (`model_initialize(seed=...)`);
- GEARS default budget: 20 epochs, lr 1e-3, weight decay 5e-4, StepLR gamma 0.5,
  batch size 32;
- GEARS default graphs: coexpression k=20 and threshold 0.4, default GO graph
  with k=20;
- checkpoint selection on validation normalized MSE: for each validation pair,
  MSE(predicted mean, true mean) / MSE(control mean, true mean) over its top 20
  DE genes, averaged over pairs. Values below 1 beat predicting no change.

`PertData.load(data_name="norman", frozen_split_dir=...)` checks the
expression-gene, perturbation-node and retained-cell order and the file hashes
against this snapshot, then pairs every target with its saved control from
`control_assignments.csv.gz`, for training and evaluation alike. The cell-graph
cache is keyed by that file's digest, so a random-control `cell_graphs.pkl` is
never reused. `prepare_split(split="custom", seed=N)` then reads
`seed_N/gears_custom_split.pkl`.

`scripts/norman_train.py <run_id>` runs one `configs/norman_runs.csv` row (currently
`gears`, `gears_set_transformer`, and `gears_residual_set_transformer`). It fits the coexpression graph on the
training controls and singles, checks the fit population against
`graph_metadata.json`, saves both graphs with tensor digests per split, and
reuses them for every model. `GEARS.train(selection_metric="normalized_mse_de",
run_test=False)` keeps the test loader out of training; `--test` (or
`GEARS.test()`) evaluates the selected checkpoint afterwards.

```bash
python "$repo_dir/scripts/norman_train.py" gears_split1_train101 \
    --data-path "$runtime_dir/data" --output-root "$runtime_dir"
```

Corrected runs use shared coexpression propagation and separate output paths;
see [model variants](model-variants.md) for checkpoint compatibility and tests.
The frozen historical run plan is not modified. TODO: settle final test scoring
definitions and add the DepMap variants.

Fit any new preprocessing/graphs/loss masks on permitted training observations.
Keep held-out outcomes and their rankings out of model inputs and tuning. The
five predefined splits overlap across repetitions: freeze choices before test
inspection, and use synthetic data or training singles/controls for adaptive
pilot development. A pair used for training in one split can be test in another.
