# Model variants and coexpression batching

## Shared graph correction

`GEARS.model_initialize(coexpression_batching="shared")` is the default for new
models. In `gears/model.py`, coexpression SGConv layers process the N gene
embeddings once, then repeat the result for B cells. This is equivalent to B
disconnected copies of the same graph, including gradients, with less work.

The inherited implementation repeats embeddings before convolution but keeps
edges indexed only from 0 to N-1. Later cells therefore receive only self-loop
updates. `coexpression_batching="legacy"` preserves this behavior for historical
reproduction. GO propagation, graph weights, the 0.2 coexpression multiplier,
loss, decoder and optimizer are unchanged.

Batch-position and batch-size invariance apply at evaluation time. Training
still uses the original BatchNorm layers, so batch composition can affect
training outputs. Do not interpret the correction as removing BatchNorm.

## Checkpoints and experiments

- The batching policy is saved in `config.pkl` and the Norman `results.json`.
- Loading a checkpoint with no policy explicitly selects `legacy`. The low-level
  `GEARS_Model(args)` also treats a missing field as legacy; new code should use
  `model_initialize` or pass the setting explicitly.
- Retrain corrected models; old weights and metrics are not corrected results.
- `pert_agg="sum"` keeps original sum-plus-MLP fusion; `"set_transformer"`
  uses the existing attention replacement. Both use the same batching policy.
- `fix/coexpression-batching` starts from `set-transformer` at `b2390d2`.
  The original GEARS reference is `master` at `f374e43`.

`configs/norman_runs.csv` lists corrected runs, selected by
`configs/norman_training.json`. They write to
`outputs/norman-benchmark/coexpression_shared_v1/<run_id>/` under the external
output root. The frozen `splits/norman/planned_runs.csv` and its hashes remain
unchanged. Splits, controls, graph fitting, training seed 101, training budget
and validation checkpoint selection are shared; see [Norman splits](norman-splits.md).

## Checks and commands

From any directory, set `repo_dir` to an absolute checkout path and use an
environment with `requirements.txt` and pytest installed:

```bash
repo_dir='/absolute/path/to/gears-settransformer-depmap'
PYTHONPATH="$repo_dir" python -m pytest -p no:cacheprovider "$repo_dir/tests"
python "$repo_dir/scripts/norman_train.py" gears_split1_train101 \
  --data-path /absolute/external/data --output-root /absolute/external/results
python "$repo_dir/scripts/run_norman_all.py" --splits 1 --dry-run \
  --data-path /absolute/external/data --output-root /absolute/external/results
```

Synthetic tests cover reordered and resized evaluation batches, an explicit
disconnected-graph reference in training/evaluation with gradient comparisons,
historical output compatibility, checkpoint loading/re-saving, masks and
validation-only checkpoint selection. They establish implementation behavior;
full Norman retraining is required to measure predictive performance.

The batch launcher reads the current run plan, supports `--models`/`--splits`,
and skips only completed runs with matching protocol hashes and checkpoints.
It uses the invoking Python environment and no machine-specific paths. Remove
`--dry-run` to train; omit `--splits` to run all five frozen splits.
