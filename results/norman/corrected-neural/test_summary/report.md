# Norman matched final-test results

All models use the same five test panels, cells, measured means, genes, DE indices and GEARS scorer.

| Model | MSE (mean +/- sample SD) | DE MSE | Pearson | DE Pearson |
|---|---:|---:|---:|---:|
| additive | 0.001890 +/- 0.000465 | 0.058602 +/- 0.011873 | 0.994601 +/- 0.001169 | 0.940756 +/- 0.050022 |
| linear | 0.004810 +/- 0.000926 | 0.268220 +/- 0.027441 | 0.985629 +/- 0.002666 | 0.860361 +/- 0.049494 |
| gears | 0.005109 +/- 0.001178 | 0.114550 +/- 0.004649 | 0.987247 +/- 0.002535 | 0.913989 +/- 0.044213 |
| gears_set_transformer | 0.004760 +/- 0.000806 | 0.173365 +/- 0.021806 | 0.986892 +/- 0.002004 | 0.893831 +/- 0.048523 |
| gears_residual_set_transformer | 0.004963 +/- 0.001117 | 0.119467 +/- 0.011359 | 0.987458 +/- 0.002468 | 0.911824 +/- 0.043233 |

MSE metrics are raw; Pearson uses absolute expression. Each split averages its 26 pair scores equally; sample statistics use ddof=1.

All three neural variants retain their published corrected shared coexpression batching. Fixed source-row order and batch size 32 match the earlier baseline test contract.

Published neural checkpoints were selected using validation mse_de. GEARS and replacement Set Transformer use coexpression_shared_v1; residual Set Transformer uses residual_comparison_v1. The baseline outcomes, controls and four metrics are unchanged. No reselection, refit or tuning was performed.

Panels overlap and describe split variability, not independent biological replicates. Test outcomes were previously exposed by the baseline evaluations and some validation panels.
