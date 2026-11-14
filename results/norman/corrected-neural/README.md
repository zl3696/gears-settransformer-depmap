# Corrected GEARS and residual Set Transformer: Norman final tests

[Comparison report](test_summary/report.md) | [Split scores](test_summary/split_metrics.csv) | [Mean/variance/SD](test_summary/metric_summary.csv) | [Paired differences](test_summary/paired_difference_summary.csv)

All 15 new checkpoints were tested and verified on 12 November 2026 against the
unchanged additive/linear final-test panels: three neural variants, five splits,
26 held-out pairs per split, 5,045 genes and 20 DE indices per pair. The evaluation
contains 35,294 cell predictions per variant (105,882 total); conditions overlap
across splits. Means weight pairs equally; sample statistics use ddof=1.

Additive has the best five-split mean on all four metrics. Among neural variants,
corrected GEARS has the lowest mean DE MSE, replacement Set Transformer the lowest
mean all-gene MSE, and residual Set Transformer the highest mean all-gene Pearson.
These are descriptive comparisons, not independent biological replication.

## Source and protocol

- GEARS and replacement Set Transformer: published commit `858a96bb1fa92ffaf4bd35d4944d61b9a6960ed9`, namespace `coexpression_shared_v1`.
- Residual Set Transformer: published commit `b112de0833294b4ae0048549fe3d1903a974323b`, namespace `residual_comparison_v1`.
- All use shared coexpression propagation, training seed 101, 20 training epochs,
  and their published validation `mse_de`-selected checkpoints.
- The cohorts' training protocols differ only by registration of the residual
  mode; their run-plan hashes remain distinct. Existing weights are preserved.
- Test inference uses CPU, four threads, fixed source-row order and batch size 32.
  Saved frozen controls, measured test means, DE indices and four-metric scorer
  exactly match the baseline contract. No retraining or reselection occurred.

The immutable [evaluation protocol](../../../configs/norman_corrected_neural_test.json)
pins every source/checkpoint Git blob and size. Test outcomes had previously been
exposed by baseline tests and overlapping validation panels; descriptive PCA
preceded splitting. Do not tune subsequent models to these scores across splits.

## Files and checks

`evaluation/<run_id>/` preserves per-pair scores, evaluation configuration, original
summary, independent metric check and verification report for every new run. The
three split-1 folders also include exact full-inference regeneration evidence.
`test_summary/` includes the two unchanged baselines in the five-model comparison.
`batching_audit.json` records batch-order/size checks for all three model types.
`final_verification_report.json` records the 15-run audit, 13 synthetic checks,
three inference regenerations and independent aggregation checks.

`provenance.json` hashes this compact export and its evaluator dependencies.
Original runtime summaries retain hashes of omitted prediction/scoring arrays,
cell/gene lists and schedules. Those hashes document the full local run; this
compact export deliberately does not contain every runtime artifact. Checkpoint
weights, raw expression data, environments, logs, per-cell schedules and large
numerical arrays remain outside Git. Hash verification of this export checks the
retained package; full prediction regeneration requires the external runtime.

See the [workflow and verification commands](../../../docs/norman-corrected-neural-test.md).
