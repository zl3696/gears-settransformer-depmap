# Fitted linear baseline Norman: fitted models and evaluation results

Completed and verified on 7 November 2026. This export contains five fitted models,
all pair predictions, training condition means, and separate validation/final-test
results. It follows [Ahlmann-Eltze, Huber and Anders (2025)](https://doi.org/10.1038/s41592-025-02772-6),
with an explicit summed-embedding adaptation from unseen singles to pairs.
It does not reproduce that paper's unseen-single benchmark.

Each fit uses its own frozen training partition: 102 singles, 76 pairs, and control
(179 condition means), with 5,045 output genes. Centered, unscaled PCA treats genes
as observations; K = 10 and both factorized-solver penalties are fixed at 0.1.
Pairs sum target-gene embeddings; the intercept is the training-condition mean.
Fitting is closed-form, with no epochs, neural checkpoint, or loss curve.

**The fixed linear pair adaptation is worse than additive on all four metrics in
every validation split and every final-test split.** No tuning or refitting followed
these scores. Matched trained GEARS/Set Transformer comparisons remain pending.

The four metrics are the existing GEARS `mse`, `mse_de`, `pearson`, and `pearson_de`.
MSE is raw; Pearson uses absolute expression. The DE metrics use the same 20 released
genes and measured targets as additive. Average per-pair metrics equally, then use
sample variance/SD across five split scores (`ddof=1`).

## Validation

| Split seed | Target cells | `mse` | `mse_de` | `pearson` | `pearson_de` |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,419 | 0.0043385227 | 0.2714158049 | 0.9867733531 | 0.7860218068 |
| 2 | 7,770 | 0.0041996004 | 0.2944525437 | 0.9872351950 | 0.8769438368 |
| 3 | 7,818 | 0.0045889379 | 0.2771750850 | 0.9860882669 | 0.8839308446 |
| 4 | 7,220 | 0.0050018525 | 0.2462438947 | 0.9848813369 | 0.9125938922 |
| 5 | 6,468 | 0.0054015216 | 0.2217825180 | 0.9838544358 | 0.8647347958 |

| Metric | Mean | Sample SD | Sample variance |
|---|---:|---:|---:|
| `mse` | 0.0047060870 | 0.0004940806 | 2.441156443e-07 |
| `mse_de` | 0.2622139693 | 0.0284498681 | 0.0008093949952 |
| `pearson` | 0.9857665175 | 0.0013886914 | 1.928463819e-06 |
| `pearson_de` | 0.8648450352 | 0.0474439563 | 0.002250928989 |

Full precision: [split scores](evaluation/validation_summary/split_metrics.csv), [statistics](evaluation/validation_summary/metric_summary.csv), and [paired additive differences](evaluation/validation_summary/paired_difference_summary.csv).

## Final test

| Split seed | Target cells | `mse` | `mse_de` | `pearson` | `pearson_de` |
|---:|---:|---:|---:|---:|---:|
| 1 | 7,214 | 0.0044361432 | 0.2406550496 | 0.9867856076 | 0.8982950368 |
| 2 | 6,901 | 0.0049084381 | 0.2870593752 | 0.9852534278 | 0.9040880807 |
| 3 | 6,340 | 0.0056657592 | 0.3038532110 | 0.9833334082 | 0.8324152467 |
| 4 | 8,036 | 0.0034277450 | 0.2432433949 | 0.9895897789 | 0.8794623320 |
| 5 | 6,803 | 0.0056098032 | 0.2662907138 | 0.9831810804 | 0.7875465910 |

| Metric | Mean | Sample SD | Sample variance |
|---|---:|---:|---:|
| `mse` | 0.0048095777 | 0.0009262958 | 8.58023967e-07 |
| `mse_de` | 0.2682203489 | 0.0274409695 | 0.0007530068053 |
| `pearson` | 0.9856286606 | 0.0026659246 | 7.107154015e-06 |
| `pearson_de` | 0.8603614575 | 0.0494944767 | 0.002449703225 |

Full precision: [split scores](evaluation/test_summary/split_metrics.csv), [statistics](evaluation/test_summary/metric_summary.csv), and [paired additive differences](evaluation/test_summary/paired_difference_summary.csv).

## Interpretation and prior exposure

Each panel contains 26 pairs. The five validation panels cover 88 unique pairs and
23,925 cells; the five test panels cover 88 unique pairs and 22,779 cells. Splits
overlap and are not independent biological replicates. Reported SD describes
partition variability, not stochastic-training variability.

Of 88 unique test pairs, 56 previously occurred in validation panels of other splits;
validation/test together expose 120 distinct pairs. Dataset-wide descriptive PCA
preceded formal splitting. The existing additive tests exposed the same outcomes.
Do not describe these as globally untouched test outcomes or tune later models to
them across splits. Final linear scoring used the existing validated fits without
adding validation cells to training. Neural protocol decisions remain separate.

GEARS converts undefined Pearson values to zero. Validation `pearson_de` conversion
counts are 4/1/1/0/1 across seeds 1-5; test counts are 0/0/2/1/3. All affected pairs
remain in their averages and are flagged in the pair-score CSVs. No all-gene
correlation required conversion; no metric or DE ranking was changed after scoring.

## Files

| Location | Contents |
|---|---|
| `fits/seed_N/` | `model.npz`, all 128 pair predictions, 179 training means/counts, original configuration/summary, regeneration report, independent fit audit |
| `evaluation/seed_N/{val,test}/` | Four per-pair scores, matched additive scores/differences, exact DE-gene IDs, original configuration/summary, numerical and regeneration checks |
| `evaluation/{validation_summary,test_summary}/` | Five split scores, mean/sample variance/SD, pair-score tables, paired differences and their statistics |
| `evaluation/*verification_report.json` | Historical task-level validation and final-test verification records |
| `provenance.json` | Retained-file/source hashes, source/split identifiers, and explicit omitted-file policy |
| `reproduction_verification.json` | Fresh portable source-data reproduction of five fits, ten scoring panels, and both aggregates |
| `portable_export_checks.json` | 17 original tests and rejection checks for altered results and in-checkout runtimes |

Original arrays, numerical sources, historical summaries, paths, hashes, and
stage-specific statuses are preserved. For example, a fit summary describes the
prediction-only stage; later summaries record scoring. Historical local paths are
provenance, not portable defaults. The inherited additive results snapshot retains
its earlier historical statement that fitted linear was not yet implemented; this
linear package records the later implementation and results.

Per-cell fit/evaluation allowlists and duplicate `scoring_inputs.npz` files are
omitted deliberately. Their hashes remain in historical summaries. Frozen split
metadata identifies the cells; full source-data reproduction reconstructs the
omitted arrays and rankings outside the checkout. Raw H5AD data, environments,
caches, and new runtime outputs are not committed.

All 17 original linear tests pass. Five fit and ten evaluation regenerations,
independent covariance/solver/metric checks, and both aggregates passed locally.
Independent final-test metrics agree with GEARS within 1.45e-15. The portable export
verifier checks models, predictions, gene/condition axes, population counts, DE
identity, pair-score aggregation, and paired statistics without the H5AD. It does
not claim to reconstruct omitted observed expression without the source data.

A fresh portable rerun from the H5AD reproduced all 115 fitted-array fields, 40
split-level metric values, ten scoring panels, and both aggregate summaries within
`rtol=1e-10`, `atol=1e-12`. The committed reproduction report records every stage.

Use the [portable verification and reproduction workflow](../../../docs/norman-linear.md).
