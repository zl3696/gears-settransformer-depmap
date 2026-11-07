# Additive baseline Norman: validation and final-test results

Completed and verified on 7 November 2026. All five frozen split seeds use 26
validation and 26 test pairs; their results are reported separately. Predictions
are `mu_A + mu_B - mu_control`, fitted only on the 54,931 shared training
single/control cells. Fitting has no epochs, neural checkpoint, or pair-outcome
input. See the [implementation and commands](../../../docs/norman-additive.md).

The four metrics are the unchanged GEARS `mse`, `mse_de`, `pearson`, and
`pearson_de`. MSE is raw; Pearson uses absolute expression. DE metrics use the
same first 20 released genes selected by GEARS. Each split score averages its
26 pair scores equally. Across-split sample variance/SD use `ddof=1`.

## Validation

| Split seed | Target cells | `mse` | `mse_de` | `pearson` | `pearson_de` |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,419 | 0.0019596453 | 0.0596661817 | 0.9943273979 | 0.8346772669 |
| 2 | 7,770 | 0.0019952615 | 0.0675737580 | 0.9939500855 | 0.9459761818 |
| 3 | 7,818 | 0.0015130298 | 0.0395407728 | 0.9954855633 | 0.9343912244 |
| 4 | 7,220 | 0.0023909996 | 0.0796946929 | 0.9930781440 | 0.9801454122 |
| 5 | 6,468 | 0.0021801096 | 0.0579794109 | 0.9934897189 | 0.9440387641 |

| Metric | Mean | Sample SD | Sample variance |
|---|---:|---:|---:|
| `mse` | 0.0020078092 | 0.0003255020 | 1.059515317e-07 |
| `mse_de` | 0.0608909633 | 0.0146973455 | 0.000216011964 |
| `pearson` | 0.9940661819 | 0.0009226164 | 8.512210638e-07 |
| `pearson_de` | 0.9278457699 | 0.0548848577 | 0.003012347602 |

Full precision: [split scores](evaluation/validation_summary/split_metrics.csv), [sample statistics](evaluation/validation_summary/metric_summary.csv), and [130 per-pair scores](evaluation/validation_summary/per_condition_metrics.csv).

## Final test

| Split seed | Target cells | `mse` | `mse_de` | `pearson` | `pearson_de` |
|---:|---:|---:|---:|---:|---:|
| 1 | 7,214 | 0.0018672347 | 0.0554272570 | 0.9946676385 | 0.9878953870 |
| 2 | 6,901 | 0.0017752318 | 0.0609139787 | 0.9949252472 | 0.9865967354 |
| 3 | 6,340 | 0.0025817327 | 0.0778738141 | 0.9928587111 | 0.9063357393 |
| 4 | 8,036 | 0.0012811939 | 0.0474419511 | 0.9961165973 | 0.9487976683 |
| 5 | 6,803 | 0.0019434841 | 0.0513531268 | 0.9944347885 | 0.8741536031 |

| Metric | Mean | Sample SD | Sample variance |
|---|---:|---:|---:|
| `mse` | 0.0018897754 | 0.0004652121 | 2.16422318e-07 |
| `mse_de` | 0.0586020255 | 0.0118734990 | 0.0001409799778 |
| `pearson` | 0.9946005965 | 0.0011693810 | 1.367452019e-06 |
| `pearson_de` | 0.9407558266 | 0.0500219874 | 0.002502199221 |

Full precision: [split scores](evaluation/test_summary/split_metrics.csv), [sample statistics](evaluation/test_summary/metric_summary.csv), and [130 per-pair scores](evaluation/test_summary/per_condition_metrics.csv).

## Interpretation and outcome exposure

The validation panels cover 88 unique pairs and 23,925 unique target cells. The
final-test panels cover 88 unique pairs and 22,779 unique cells. Repeated pairs
are not independent biological replicates; SD reflects partition variability,
not training-seed uncertainty. All five splits share the same additive fit.
These scores alone do not establish improvement over GEARS or Set Transformer;
matched trained neural results remain pending. The fitted linear baseline is a
separate unimplemented method.

The scorer converts undefined Pearson to zero. Validation DE correlations required
this 4, 1, 1, 0, and 1 times in seeds 1–5; final test required it 0, 0, 2, 1, and 3
times. All remain in the means and are flagged in the per-pair CSVs. No all-gene
Pearson required conversion. Final-test affected conditions are:

- Seed 3: `CBL+UBASH3A`, `KIF18B+KIF2C`.
- Seed 4: `SAMD1+TGFBR2`.
- Seed 5: `C3orf72+FOXL2`, `CBL+UBASH3A`, `CEBPB+JUN`.

Measured expression is zero at all 20 released DE genes in those cases. No
alternative ranking or metric was chosen after observing this behavior.

Of 88 unique test pairs, 56 already had validation outcomes exposed through other
predefined splits (16, 15, 18, 16, and 12 per test seed). Validation/test together
cover 120 unique pairs. Earlier descriptive dataset-wide PCA/previews are also
recorded in frozen provenance. Do not call these globally untouched outcomes or
adapt later model settings to them across overlapping splits. The user authorized
additive final testing with its predictor/scorer fixed; outstanding neural choices
remain pending.

## Files and verification

| Location | Contents |
|---|---|
| `predictions/` | Compact pair predictions, single/control means, five-split membership, historical fit summary and checks |
| `evaluation/seed_N/{val,test}/` | Original configuration, four per-pair scores, exact DE-gene lists, historical summary and regeneration report |
| `evaluation/validation_summary/` | Five split scores, mean/sample variance/SD, merged per-pair scores, shared provenance and verification |
| `evaluation/test_summary/` | Same final-test summaries plus the independent NumPy metric check |
| `provenance.json` | Hash manifest for retained files/source, input hashes, and explicit omitted-runtime-file policy |
| `reproduction_verification.json` | Fresh portable source-data rerun matches both prediction arrays, all ten split scores, and both aggregate statistics |

All 21 preserved regression tests pass. Original predictions, all ten evaluation
runs, and both aggregates regenerated identically. Independent NumPy test scores
agree within 2.23e-16. All 140 repeated-condition array checks across both
partitions passed; source and frozen split hashes were reverified.

The historical summaries retain their original hashes, paths, timestamps, and
stage-specific statuses. In particular, the prediction summary describes the
prediction-only stage; these validation/test bundles record later evaluation.
Per-cell allowlists and duplicate `scoring_inputs.npz` files are deliberately
omitted from this compact export; their original hashes remain recorded. The
frozen split metadata identifies cells, and the full source-data workflow
recreates these files outside the repository. Raw H5AD data, environments, caches,
checkpoints, and new runtime outputs are not part of this snapshot.

Use the portable [verification and reproduction commands](../../../docs/norman-additive.md#commands-from-any-directory).
Export verification checks hashes, exact additive formula, gene membership,
populations, and metric/statistic aggregation; source-data reruns additionally
regenerate measured condition means and scoring inputs. Original evaluated source
files are preserved byte for byte under `scripts/norman_additive/`.
