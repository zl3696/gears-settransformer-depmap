# Norman fitted linear baseline: implementation and reproducibility

Five fits and separate five-split validation/final-test evaluations are complete.
The [results](../results/norman/linear/README.md) contain all four metrics, per-pair
scores, fitted models, predictions, training means, variance/SD, paired additive
comparisons, and provenance. The linear pair adaptation is worse than additive in
every split/metric. The [matched neural test workflow](norman-neural-test.md)
compares published neural checkpoints against the same final-test panels.

## Model and evaluation contract

This model adapts the fitted linear baseline of [Ahlmann-Eltze, Huber and Anders
(2025)](https://doi.org/10.1038/s41592-025-02772-6) to our unseen-pair task. The paper's
direct unseen-pair baseline is additive; its fitted linear benchmark concerns
unseen singles. Summing embeddings for pairs is an explicit project adaptation.

For each frozen split, pool aliases by cell and construct Y with 5,045 gene rows
and 179 canonical training-condition columns: 102 singles, 76 pairs, and control.
Use unmodified processed expression and equal weight per condition mean. Fit
centered, unscaled PCA with genes as observations, preserving score scaling;
G has 10 columns. All 102 targets map uniquely to the output gene panel.

For target A, use its row of G as p_A; use p_A + p_B for A+B and zero for control.
Stack training embeddings as columns of B. Let b be the row mean of Y. Fit:

```text
W = inverse(G.T @ G + 0.1 I) @ G.T @ (Y - b[:, None])
W = W @ B.T @ inverse(B @ B.T + 0.1 I)
prediction = b + G @ W @ p_condition
```

Implementation uses linear solves. Keep negative predictions; no extra
normalization, clipping, whitening, validation refit, or parameter search occurs.
The pair model remains additive around its fitted intercept, with no pair-specific
interaction term. Every fit is isolated to its own training partition.

Use the existing GEARS `compute_metrics`: raw MSE on all 5,045 genes and on the
condition's 20 DE genes; absolute-expression Pearson on the same two gene sets.
Reuse additive's measured targets and DE indices. Keep the first 20 released DE
IDs, mapped into expression-axis order, and the undefined-Pearson-to-zero policy
with explicit flags. Average conditions equally within a split, then use sample
variance/SD across the five split scores (`ddof=1`). Validation and test stay separate.

## Source and runtime boundary

- `scripts/norman_linear.py` provides portable `verify`, `tests`, `fit`, `audit`,
  scoring, aggregation, and full `reproduce` commands.
- `scripts/norman_linear/` preserves all nine evaluated Python source/test files
  byte for byte. Their original configuration hashes remain valid.
- Existing `scripts/norman_additive/` sources and results are reused unchanged.
  The numerical source expects historical hyphenated sibling paths. The portable
  entry point creates byte-identical source mirrors under an explicit external
  runtime root, binds the repository split/scorer paths, and leaves math unchanged.
- `results/norman/linear/` is a curated result snapshot. Its manifest distinguishes
  retained files from omitted per-cell lists and duplicate scoring arrays. Existing
  historical summaries retain their original paths, hashes, and stage statuses.
- Runtime sources, measured arrays, checkpoints, caches, and fresh results belong
  outside the checkout. In the course workspace, keep them under `Codex Outputs/`.

Python 3.12, NumPy, h5py, and the existing GEARS/SciPy/scikit-learn environment are
used. No new dependency is required. The compact export can be verified without
opening the H5AD; reconstructing measured target profiles requires that source.

## Commands from any directory

Use absolute paths. Collaborators should substitute their own checkout, Python,
source data, and a fresh external runtime directory.

```bash
NORMAN_PYTHON='/Users/lisa/Desktop/Barnard/26 Fall/ML for Functional Genomics/ML Project /Codex Outputs/.venvs/gears/bin/python'
NORMAN_REPOSITORY='/Users/lisa/Desktop/Barnard/26 Fall/ML for Functional Genomics/ML Project /Code/gears-settransformer-depmap'
NORMAN_LINEAR_RUNTIME='/Users/lisa/Desktop/Barnard/26 Fall/ML for Functional Genomics/ML Project /Codex Outputs/tmp/norman-linear-reproduction'
NORMAN_SOURCE='/Users/lisa/Desktop/Barnard/26 Fall/ML for Functional Genomics/ML Project /Codex Outputs/data/norman/perturb_processed.h5ad'

VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_linear.py" verify
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_linear.py" tests --runtime-root "$NORMAN_LINEAR_RUNTIME"
```

Reproduce the completed benchmark from the source data in a fresh runtime root:

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_linear.py" reproduce --runtime-root "$NORMAN_LINEAR_RUNTIME" --source "$NORMAN_SOURCE"
```

This rebuilds and verifies additive reference means/validation/test targets, fits
and verifies all five linear models, runs independent fit audits, evaluates and
verifies both partitions, and reconstructs both aggregate summaries. It compares
all 115 fitted-array fields and 40 split-level metric values with the committed
snapshot. The completed portable rerun passed these comparisons and both aggregate
comparisons; its report is committed with the results. It refuses to reuse a runtime containing existing linear fits.

Individual steps are also available after rebuilding additive targets using the
[additive entry point](norman-additive.md#commands-from-any-directory):

```bash
for split_seed in 1 2 3 4 5; do
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_linear.py" fit --split-seed "$split_seed" --runtime-root "$NORMAN_LINEAR_RUNTIME" --source "$NORMAN_SOURCE"
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_linear.py" fit --action verify --split-seed "$split_seed" --runtime-root "$NORMAN_LINEAR_RUNTIME" --source "$NORMAN_SOURCE"
done
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_linear.py" audit --runtime-root "$NORMAN_LINEAR_RUNTIME"
```

Scoring commands are `validation` and `test`, each taking `--split-seed` and
`--runtime-root`. Their aggregate commands are `validation-summary` and
`test-summary`. Add `--action verify` to regenerate an existing bundle. Final test
requires the corresponding verified validation run and the unchanged predictions.
All run commands refuse to overwrite existing bundles.

Runtime output is under `RUNTIME/outputs/norman-linear/seed_N/` and
`RUNTIME/outputs/norman-linear-evaluation/seed_N/{val,test}/`, with separate
`validation_summary/` and `test_summary/` aggregates. `RUNTIME/reproduction_verification.json`
records the source-data reproduction comparison. Mirrored sources live under
`RUNTIME/scripts/`; they are not a second editable implementation.

## Outcome access and future development

The user authorized the completed final evaluation with model and scorer fixed.
The overlapping panels exposed 120 distinct pair outcomes across validation/test,
and descriptive PCA preceded splitting. Neither the snapshot nor a reproduction
creates globally untouched test outcomes. Do not adapt later models using these
scores across splits. Outstanding neural architecture, batching, and training
choices remain separate.

Keep the evaluated source and result hashes unchanged. Numerical or protocol
changes require a new versioned output and an explicit account of prior exposure.
