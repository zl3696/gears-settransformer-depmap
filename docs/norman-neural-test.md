# Norman neural final tests matched to the baselines

This document covers the historical `b2390d26` checkpoints. The newer corrected
batching and residual checkpoints use the separate
[corrected neural workflow](norman-corrected-neural-test.md).

## Evaluation contract

The user authorized local final testing of all ten published GEARS and Set
Transformer checkpoints on 9 November 2026, using the same tests as the completed
additive and fitted linear baselines. `scripts/norman_neural_test.py` supplies an
inference-only entry point; it never calls a trainer or changes a checkpoint.

`configs/norman_neural_test.json` pins source commit
`b2390d26e752431a6550cf136e12c1b97be00bea`, Git blob hashes for every downloaded
checkpoint/source artifact, the source H5AD, frozen split bundle, exact baseline
scorer and CPU inference settings. Downloads and runtime results stay outside Git.
The evaluation branch starts from the linear baseline branch because its exact
scorer, split snapshot and preserved helper implementations are dependencies.
The published neural model source is a hash-verified external dependency, not a
second editable project implementation.

- Use the exact baseline test cells and 26 canonical pairs per split, all 5,045
  expression genes and the same 20 DE indices per pair.
- Reuse verified additive `scoring_inputs.npz` measured condition means and DE
  indices directly. Check their hashes, cell lists, gene axes and historical scores.
- Read only assigned training-control rows from H5AD `X`. Never load the dense
  counts layer or materialize all per-cell GEARS graphs. Test expression is not
  reread from H5AD. No normalization, clipping or new DE calculation is performed.
- Match `evaluation_control_schedule.csv.gz` to `control_assignments.csv.gz` by
  target ID; require `input_index=0`. Input expression is the saved control for
  each target. Both genes of the canonical pair supply perturbation indices.
- Run the published model and graphs on CPU, four threads, batch size 32,
  increasing target source-row order, no shuffle. Average cell predictions in
  float64 within each canonical pair, then use the unchanged baseline GEARS
  `compute_metrics`: `mse`, `mse_de`, `pearson`, `pearson_de`. Weight pairs equally;
  summarize five split scores with sample variance/SD (`ddof=1`).
- Preserve undefined-Pearson-to-zero behavior and record affected pairs.

## Interpretation constraints

Published neural checkpoints were selected using validation `normalized_mse_de`,
as recorded by their trainer. The final test metrics remain the four historical
baseline metrics; no checkpoint is reselected using test outcomes.

The published coexpression model supplies unoffset graph edges to a stacked
batch, so graph propagation applies to the first sample of each batch. This
runner retains that trained implementation and freezes source-row order and
batch size 32 for both neural variants. Results depend on this policy; do not
silently change batch size, reorder cells, or correct graph propagation during
testing. A corrected implementation requires a separately labeled experiment
and retraining, with its protocol decided independently of these test scores.

Test outcomes were already exposed by baseline evaluations and some overlapping
validation panels, and dataset-wide PCA preceded splitting. Report split
variability, not independent biological-replicate or training-seed uncertainty.
Do not tune later variants to these scores across splits. DepMap remains separate.

## Runtime files and checks

Paths below are relative to an explicit external `--runtime-root`:

- `data/norman-neural-checkpoints-b2390d26/`: immutable downloaded checkpoints,
  model source, training configuration and download manifest.
- `outputs/norman-neural-test/<run_id>/`: persisted pre-scoring configuration,
  condition-mean scoring arrays, prediction sums/counts, per-pair scores, exact
  baseline cell/gene lists, control/batch schedule, first-batch predictions,
  timing/memory/provenance and verification reports.
- `outputs/norman-neural-test/test_summary/`: all four models' split scores,
  mean/sample variance/SD, paired split differences and a concise report.

`verify` checks hashes, exact baseline targets/cells/genes, reaggregates saved
predictions, reproduces the historical additive score and the neural scores, and
checks metrics independently with NumPy. `regenerate` repeats all cell inference
and requires exactly identical condition means and scoring arrays. A completed
`run` verifies and skips an existing bundle; incomplete output requires inspection.
Existing baseline bundles and user notebooks are read-only.

## Completed local run

All ten final-test runs completed on 9 November 2026 on the M5 MacBook Air with
16 GB memory. Each output passes exact baseline-target/cell/gene matching,
historical additive-score reproduction, neural-score regeneration from saved
arrays and independent NumPy metric checks. Seven synthetic regression tests
cover pairing, excluded-row access, fixed batching, aggregation and scoring.
Both neural variants' split 1 runs also passed full cell-inference regeneration:
condition means and scoring arrays were exactly identical after checkpoint reload.

The local report (`outputs/norman-neural-test/test_summary/report.md` under the runtime directory)
contains all four models' five-split mean/SD; sibling CSVs preserve full precision,
split-level scores and paired differences. Additive has the best five-split mean
on all four metrics. Set Transformer has worse five-split means than original
GEARS on all four metrics for these fixed checkpoints and batching policy.
These test scores do not authorize hyperparameter changes or checkpoint reselection.

## Commands from any directory

```bash
NORMAN_PYTHON='/path/to/project/Codex Outputs/.venvs/gears/bin/python'
NORMAN_REPOSITORY='/path/to/project/Code/gears-settransformer-depmap'
NORMAN_RUNTIME='/path/to/project/Codex Outputs'

PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_neural_test.py" download --runtime-root "$NORMAN_RUNTIME"
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" -m unittest discover -s "$NORMAN_REPOSITORY/tests" -p 'test_norman_neural_test.py' -v
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_neural_test.py" run --runtime-root "$NORMAN_RUNTIME"
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_neural_test.py" verify --runtime-root "$NORMAN_RUNTIME"
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_neural_test.py" summary --runtime-root "$NORMAN_RUNTIME"
```

`run`, `verify`, and `regenerate` accept `--model gears` or
`--model gears_set_transformer`, and `--split-seed 1` through `5`. `--source` may
override the default H5AD path, but its hash must still match the baseline source.
The existing Python 3.12 GEARS environment supplies all dependencies.
