# Replogle evaluation

Protocol: joint_unseen; partition: test; split seeds: [1, 2, 3, 4, 5].

Each split weights perturbations equally, then available cell lines equally. SD is sample SD across splits; panels overlap and are not biological replicates.

Expression metrics use raw frozen-input predictions. Response/context metrics subtract the actual assigned-input mean for neural models and the destination training-control mean for Copy-effect. Measured responses subtract the same-line full training-control mean.

| Variant | Scope | Metric | Mean | Sample SD | Valid splits |
|---|---|---|---:|---:|---:|
| A | K562 | mse | 0.0127994 | 0.00100031 | 5 |
| A | K562 | mse_de | 0.12686 | 0.00806959 | 5 |
| A | K562 | pearson | 0.976908 | 0.00192393 | 5 |
| A | K562 | pearson_de | 0.954103 | 0.00683261 | 5 |
| A | K562 | response_pearson | 0.344316 | 0.0128828 | 5 |
| A | RPE1 | mse | 0.01923 | 0.000662828 | 5 |
| A | RPE1 | mse_de | 0.135161 | 0.0230499 | 5 |
| A | RPE1 | pearson | 0.968355 | 0.00111092 | 5 |
| A | RPE1 | pearson_de | 0.753402 | 0.0368538 | 5 |
| A | RPE1 | response_pearson | 0.720985 | 0.0173387 | 5 |
| A | cross_line | response_difference_mse | 0.0276407 | 0.000181514 | 5 |
| A | cross_line | similarity_mae | 0.650083 | 0.0129928 | 5 |
| A | overall | mse | 0.0160147 | 0.000254394 | 5 |
| A | overall | mse_de | 0.13101 | 0.0138037 | 5 |
| A | overall | pearson | 0.972632 | 0.000605525 | 5 |
| A | overall | pearson_de | 0.853752 | 0.0162737 | 5 |
| A | overall | response_pearson | 0.532651 | 0.0148149 | 5 |
| B | K562 | mse | 0.00972546 | 0.00137391 | 5 |
| B | K562 | mse_de | 0.124451 | 0.00469804 | 5 |
| B | K562 | pearson | 0.983403 | 0.00198965 | 5 |
| B | K562 | pearson_de | 0.954823 | 0.00876647 | 5 |
| B | K562 | response_pearson | 0.417373 | 0.0310205 | 5 |
| B | RPE1 | mse | 0.0181306 | 0.00159088 | 5 |
| B | RPE1 | mse_de | 0.107329 | 0.0223362 | 5 |
| B | RPE1 | pearson | 0.969764 | 0.00268813 | 5 |
| B | RPE1 | pearson_de | 0.733293 | 0.0379954 | 5 |
| B | RPE1 | response_pearson | 0.729605 | 0.020225 | 5 |
| B | cross_line | response_difference_mse | 0.0159584 | 0.00126465 | 5 |
| B | cross_line | similarity_mae | 0.401183 | 0.0541373 | 5 |
| B | overall | mse | 0.013928 | 0.00145543 | 5 |
| B | overall | mse_de | 0.11589 | 0.0131855 | 5 |
| B | overall | pearson | 0.976583 | 0.00231956 | 5 |
| B | overall | pearson_de | 0.844058 | 0.0151452 | 5 |
| B | overall | response_pearson | 0.573489 | 0.0243187 | 5 |
| C | K562 | mse | 0.00966065 | 0.000427687 | 5 |
| C | K562 | mse_de | 0.135405 | 0.00411513 | 5 |
| C | K562 | pearson | 0.98318 | 0.000708376 | 5 |
| C | K562 | pearson_de | 0.952943 | 0.0080649 | 5 |
| C | K562 | response_pearson | 0.372649 | 0.0166002 | 5 |
| C | RPE1 | mse | 0.0194413 | 0.00122357 | 5 |
| C | RPE1 | mse_de | 0.101378 | 0.0220681 | 5 |
| C | RPE1 | pearson | 0.967327 | 0.00192071 | 5 |
| C | RPE1 | pearson_de | 0.720693 | 0.0309264 | 5 |
| C | RPE1 | response_pearson | 0.717429 | 0.0165513 | 5 |
| C | cross_line | response_difference_mse | 0.016643 | 0.000373563 | 5 |
| C | cross_line | similarity_mae | 0.429581 | 0.0480842 | 5 |
| C | overall | mse | 0.014551 | 0.000814055 | 5 |
| C | overall | mse_de | 0.118392 | 0.0126728 | 5 |
| C | overall | pearson | 0.975253 | 0.00128005 | 5 |
| C | overall | pearson_de | 0.836818 | 0.0123442 | 5 |
| C | overall | response_pearson | 0.545039 | 0.0133274 | 5 |

Similarity figures: one point per held-out perturbation, measured versus predicted cross-line response Pearson, identity line, identical axes. Undefined pairs are omitted from the plot and counted in the panel title; CSV retains every target. No across-target correlation is required.

Paired response/similarity comparisons use common-valid targets. Within-line response correlations and cross-line similarity are descriptive; technical screen differences can contribute. Upstream HVG selection and previous full-data PCA exposure remain limitations.

Lower MSE/similarity error and higher Pearson are favorable. Difference tables are model minus reference. Copy-effect is a training-only source-response baseline for transfer tasks; source-line training observations are not held-out destination outcomes.
