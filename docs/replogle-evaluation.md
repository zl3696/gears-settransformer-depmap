# Replogle evaluation

## Scope and entry point

`scripts/replogle_test.py {plan,run,verify,regenerate,summary}` evaluates the
[frozen paired data](replogle-data.md) and validation-selected
[A/B/C models](replogle-training.md). It never trains, selects checkpoints, changes
control assignments, or fits expression transformations. All outputs are external
to the checkout, frozen data and training roots.

The main task holds out the same 126 targets in both lines per split; report
K562, RPE1 and their equally weighted summary. Both transfer tasks score only
the destination line, with an additional **Copy-effect** baseline. Five splits
yield 45 neural evaluations and 10 baseline evaluations per partition. Both
contexts occur in training; transfer does not hold out an entire cell line.

`configs/replogle_evaluation.json` versions numerical rules. A copied config may
adjust batch size, device and threads; other numerical changes require a new
protocol. Defaults are CPU, batch size 128 and four threads. Training seed is 101.
All model and protocol choices must be fixed before any final tests across tasks
or overlapping splits. Validation evaluation is required before the matching test.

## Prediction and control centering

Neural inference uses each cell's frozen same-line training-control input.
Accumulate predictions and inputs in float64 by `(cell line, target)`, retaining
the original view order and cell counts. Denote a condition's raw prediction
mean by `P`, observed mean by `Y`, assigned-input mean by `I`, and that line's
full training-control mean by `C`.

- The historical four **expression** metrics score `P` against `Y`, preserving
  the training validator's raw-expression convention.
- Primary **response/context** metrics compare `P - I` with `Y - C`. Equivalently,
  recenter the prediction to `P - I + C`, then subtract `C` from both profiles.
  This removes finite control-assignment offsets without changing assignments.
- Save `P`, `I`, `C`, cell/control schedules, genes, counts, truth and DE indices.
  Per-condition `control_offset_rms` quantifies `I - C`; raw control-relative
  predictions `P - C` remain reconstructible and are not called learned context.

For source S and destination T, Copy-effect uses actual training cells:

```text
effect[p] = mean_train(S, p) - mean_train(S, control)
P[T, p] = mean_train(T, control) + effect[p]
I[T, p] = mean_train(T, control)
```

No destination perturbed validation/test expression enters baseline prediction
generation. Missing source responses fail explicitly. Copy-effect is inapplicable
to `main`/`joint_unseen`; borrowing outcomes from a transfer task is forbidden.
A's learned shared response is a separate comparator from this observed source
response. For A, `P - I` is context invariant up to floating-point error.

## Endpoints and weighting

- Reuse `gears.inference.compute_metrics`: `mse`, `mse_de`, `pearson`, `pearson_de`.
  Independent NumPy calculations check every condition's four scores.
- DE indices are generated once per task/split/partition, shared across variants.
  Use the training validator's top-20 absolute Welch ranking against same-line
  training controls, expression-index tie breaking, singleton variance zero,
  and infinite absolute scores for nonzero differences with zero standard error.
  Test DE rankings are scoring-only and never replace validation selection.
- `response_pearson` compares predicted and observed response within a line.
- Main only: for every shared target, calculate measured and predicted cross-line
  response Pearson over all shared expression genes. Primary cross-line endpoint
  is mean absolute difference between those two similarities (`similarity_mae`).
  Plot measured x versus predicted y, with identity line and identical [-1, 1]
  correlation scales; retain every target in the source CSV.
- `response_difference_mse` compares predicted versus measured response-difference
  vectors, K562 minus RPE1, and captures magnitude as well as pattern differences.

Average condition scores equally within line, then lines equally. Do not weight
metrics by cell count or treat repeated targets across splits as independent.
Report each split, mean/sample variance/SD (`ddof=1`), and valid split counts.
Partial summaries are labeled explicitly and never called five-split results.

Historical absolute-expression undefined Pearson becomes zero with flags.
Undefined response/similarity correlations remain null, with valid-condition or
valid-target counts. A wholly undefined line makes an equal-line response summary
undefined. Paired B-minus-A, C-minus-B, B-minus-Copy-effect and C-minus-Copy-effect
comparisons use common-valid conditions/targets; all exclusions remain in tables.
No across-target correlation is required when model similarity scores are constant.
Keep low-count conditions and missing-prior targets in the primary population.

## Provenance and verification

Before predictions/outcome decoding, save the evaluation configuration. Check
run/variant/task/split identity, first validation minimum, completed history,
checkpoint hashes, ordered expression/GO/context axes, training-control means,
and training-source hashes. The validation adapter must reproduce selected training
`mse` and `mse_de` within recorded batch-audit tolerances before its gate can pass.
C restores its aligned fixed prior from its checkpoint.
The trainer's immutable records keep their original `test_evaluated=false` flags;
new evaluation records track actual outcome access separately.

`run` verifies existing completed results instead of overwriting them. Inspect
incomplete directories before replacing them; a missing checkpoint fails before
CLI evaluation begins. `verify` checks saved hashes, counts, tables and independent
scores without inference or expression decoding. `regenerate` reloads the same
checkpoint/baseline and repeats full prediction/aggregation, comparing arrays and
metrics at recorded tolerances. Reports go outside sealed result directories.
The first up-to-33 cells undergo repeat/order/size/singleton inference checks.

Under the evaluation root:

- `targets/<protocol>/seed_<n>/<partition>/`: immutable shared scoring arrays.
- `runs/<protocol>/seed_<n>/<variant>/<partition>/`: pre-access configuration,
  scoring arrays, per-condition metrics, per-target similarities, audit and hashes.
- `regeneration/<protocol>/seed_<n>/<variant>/<partition>.json`: repeat checks.
- `summaries/<protocol>/<partition>/.../`: split metrics, descriptive and paired
  statistics, common-valid per-target differences, report and main PNG/PDF figures.

Figures are general research exports, not journal-certified files. They preserve
raw scatter positions, missing counts, full data tables and separate split panels.
Split SD is not biological replication. Screen-specific technical effects,
upstream HVG selection, unverified normalization and prior full-data PCA exposure
limit interpretation; gains cannot be attributed solely to biological context.

## Commands from any directory

```bash
repo_dir='/absolute/path/to/gears-settransformer-depmap'
runtime_dir='/absolute/path/to/Codex Outputs'
python_bin="$runtime_dir/.venvs/gears/bin/python"
common=(--data-dir "$runtime_dir/outputs/replogle-paired-singlecell-v1"
        --training-root "$runtime_dir/outputs/replogle-training-v1"
        --output-root "$runtime_dir/outputs/replogle-evaluation-v1")
"$python_bin" "$repo_dir/scripts/replogle_test.py" plan "${common[@]}"
"$python_bin" "$repo_dir/scripts/replogle_test.py" run "${common[@]}" --partition val
"$python_bin" "$repo_dir/scripts/replogle_test.py" run "${common[@]}" --partition test
"$python_bin" "$repo_dir/scripts/replogle_test.py" verify \
  --output-root "$runtime_dir/outputs/replogle-evaluation-v1" --partition test
"$python_bin" "$repo_dir/scripts/replogle_test.py" summary \
  --output-root "$runtime_dir/outputs/replogle-evaluation-v1" --partition test
"$python_bin" "$repo_dir/scripts/replogle_test.py" regenerate "${common[@]}" \
  --tasks main --variants A --split-seeds 1 --partition test
PYTHONDONTWRITEBYTECODE=1 "$python_bin" -m unittest discover \
  -s "$repo_dir/tests" -p 'test_replogle*.py' -v
```

Narrow any command with `--tasks`, `--variants`, `--split-seeds`, and `--partition`.
To run Copy-effect before trained neural checkpoints exist, select transfer tasks
and `--variants Copy-effect`; no `--training-root` is needed. Explicitly requesting
Copy-effect with `main` is an error. Synthetic tests cover all nine neural paths,
both baselines, validation gates, selection identity, row access, offset centering,
undefined correlations, shared targets, matched summaries, tampering and regeneration.
