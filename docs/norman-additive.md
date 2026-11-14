# Norman additive baseline: implementation and reproducibility

The additive baseline is fitted and evaluated on all five frozen Norman splits.
Validation and final test are separate; see the [results](../results/norman/additive/README.md)
for all four metrics, per-split scores, mean/sample variance/SD, and caveats.
The [fitted linear pair adaptation](norman-linear.md) is now implemented and
evaluated separately. The [matched neural test workflow](norman-neural-test.md)
reuses these exact targets and scorer for the published GEARS/Set Transformer checkpoints.

## Model and scoring contract

For each pair A+B, predict `mu_A + mu_B - mu_control`. Fit only the shared 54,931
retained training single/control cells: 102 singles and 7,353 controls. No pair
expression, released ranking, or counts-layer values enter fitting. Stored
processed `X` is unchanged; keep negative predictions and use float64 means.
All five splits share these references, so one prediction array covers 128 pairs
and 5,045 expression genes. The [frozen splits](norman-splits.md) determine which
26 pairs are validation/test in each seed; never score all 128 as held out.

This implements the additive formula in [Ahlmann-Eltze, Huber and Anders (2025)](https://doi.org/10.1038/s41592-025-02772-6).
Their fitted linear model is a separate baseline; its explicit project pair
adaptation is described in the [linear workflow](norman-linear.md).

The exact existing `compute_metrics` in `gears/inference.py` supplies:

| Key | Condition-level calculation | Better |
|---|---|---|
| `mse` | Raw MSE between predicted/measured means across 5,045 genes | Lower |
| `mse_de` | Raw MSE on the condition's same 20 DE genes | Lower |
| `pearson` | Absolute-expression correlation across 5,045 genes | Higher |
| `pearson_de` | Absolute-expression correlation on those 20 DE genes | Higher |

Average per-pair scores equally within a split, then summarize five split scores
with `ddof=1`. Reuse the first 20 released `rank_genes_groups_cov_all` IDs for the
source condition name, mapped by membership into expression-axis order. Require
20 valid unique indices and finite inputs. Preserve GEARS's undefined-Pearson-to-zero
rule, with separate flags/counts. No new DE algorithm or normalized/delta metric
is introduced. The source's `A549` placeholder names remain ranking provenance
keys; the biological data are K562 CRISPRa.

## Source and artifact map

- `scripts/norman_additive.py` is the portable entry point. Its `verify` command
  validates the compact committed export, hashes, additive formula, gene axes,
  frozen populations, per-pair aggregation, and sample statistics without H5AD.
- `scripts/norman_additive/` preserves all nine original Python source/test files
  byte for byte. Their hashes match the historical configurations. Use the entry
  point rather than invoking those files directly: it binds external runtime
  paths and this checkout's identical GEARS scorer without changing numerical code.
- `results/norman/additive/` contains the explicit reviewed results snapshot.
  `provenance.json` hashes retained artifacts and source files. Historical summaries
  are unchanged, including their original local paths and generation-time status;
  those paths are provenance, not portable execution defaults.
- Per-cell fitting/scoring allowlists and duplicate scoring-input NPZs remain
  local. Their historical hashes are retained, and the omitted filenames are
  explicit in provenance. Frozen cell assignments already identify the cells;
  a full rerun recreates allowlists, scoring inputs, and exact DE indices.

Raw H5AD, virtual environments, graph caches, logs, and new runtime outputs stay
outside Git. Python 3.12, NumPy, h5py, and the existing GEARS/scipy/sklearn/scanpy
environment are used; no additional package was installed for this workflow.
The `verify` path imports NumPy/h5py but does not load GEARS or expression data.

## Commands from any directory

Use absolute paths. These values match the local course workspace; collaborators
should set them to their own checkout, environment, source H5AD, and an external
runtime directory. Use a fresh runtime root for a new reproduction.

```bash
NORMAN_PYTHON='/path/to/project/Codex Outputs/.venvs/gears/bin/python'
NORMAN_REPOSITORY='/path/to/project/Code/gears-settransformer-depmap'
NORMAN_RUNTIME='/path/to/project/Codex Outputs/tmp/norman-additive-reproduction'
NORMAN_SOURCE='/path/to/project/Codex Outputs/data/norman/perturb_processed.h5ad'

"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" verify
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" tests --runtime-root "$NORMAN_RUNTIME"
```

Generate and independently regenerate predictions:

```bash
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" predict --runtime-root "$NORMAN_RUNTIME" --source "$NORMAN_SOURCE"
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" predict --action verify --runtime-root "$NORMAN_RUNTIME" --source "$NORMAN_SOURCE"
```

Run/verify all validation panels and their aggregate before final testing:

```bash
for seed in 1 2 3 4 5; do
  "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" validation --split-seed "$seed" --runtime-root "$NORMAN_RUNTIME" --source "$NORMAN_SOURCE"
  "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" validation --action verify --split-seed "$seed" --runtime-root "$NORMAN_RUNTIME" --source "$NORMAN_SOURCE"
done
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" validation-summary --runtime-root "$NORMAN_RUNTIME"
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" validation-summary --action verify --runtime-root "$NORMAN_RUNTIME"
```

The user explicitly authorized the completed additive final tests after its
predictor/scorer were fixed. To reproduce that already-scored final benchmark:

```bash
for seed in 1 2 3 4 5; do
  "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" test --split-seed "$seed" --runtime-root "$NORMAN_RUNTIME" --source "$NORMAN_SOURCE"
  "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" test --action verify --split-seed "$seed" --runtime-root "$NORMAN_RUNTIME" --source "$NORMAN_SOURCE"
done
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" test-summary --runtime-root "$NORMAN_RUNTIME"
"$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_additive.py" test-summary --action verify --runtime-root "$NORMAN_RUNTIME"
```

Commands refuse to overwrite run bundles. `--action verify` reruns the saved
calculation and writes its verification report under the external runtime root.
Runtime files are under `RUNTIME/outputs/norman-additive/` and
`RUNTIME/outputs/norman-additive-evaluation/seed_N/{val,test}/`, with separate
`validation_summary/` and `test_summary/` aggregates. `verify` on the committed
export checks retained artifacts and reaggregates metrics; it does not claim to
recompute omitted measured profiles without the H5AD.

## Verification and development constraints

All 21 preserved regression tests cover training-only fitting, poisoned excluded
rows/rankings, exact formula and unclipped predictions, DE mapping, actual scorer
agreement, equal condition weighting, missing/invalid inputs, and split statistics.
Original predictions, ten evaluation runs, and both aggregates regenerated
identically. Independent test metrics agree within 2.23e-16; repeated condition
profiles/DE indices agree exactly across validation/test. Preserve these sources
and snapshots; use an explicitly versioned protocol/output for numerical changes.

The overlapping panels exposed 120 unique pair outcomes in validation/test combined.
Do not adapt later architectures or hyperparameters to these scores across splits.
Additive final testing does not lock outstanding neural choices. Neural training,
schedule-aware inputs, and matched comparisons remain separate development work.
