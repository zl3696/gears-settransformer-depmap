# Replogle joint-unseen A/B/C: five-split final tests

[Report](test_summary/report.md) | [Mean/variance/SD](test_summary/metric_summary.csv) | [Split scores](test_summary/split_metrics.csv) | [Paired differences](test_summary/paired_difference_summary.csv) | [Per-target differences](test_summary/paired_per_target_differences.csv)

All 15 `main` (`joint_unseen`) models (variants A/B/C × split seeds 1–5, training
seed 101) were trained and then evaluated on 26 November 2026 with the unchanged
[evaluation workflow](../../../docs/replogle-evaluation.md). Each split tests 126
held-out targets in both K562 and RPE1 (252 conditions). All 15 test outputs passed
`replogle_test.py verify`, and every reloaded checkpoint reproduced its training
validation score before test scoring. Transfer tasks were not run.

## Main results (test, overall, mean ± sample SD over five splits)

| Metric | A | B | C |
|---|---:|---:|---:|
| `mse_de` (lower better) | 0.1310 ± 0.0138 | **0.1159 ± 0.0132** | 0.1184 ± 0.0127 |
| `mse` | 0.0160 ± 0.0003 | **0.0139 ± 0.0015** | 0.0146 ± 0.0008 |
| `pearson_de` (higher better) | **0.8538 ± 0.0163** | 0.8441 ± 0.0151 | 0.8368 ± 0.0123 |
| `response_pearson` | 0.5327 ± 0.0148 | **0.5735 ± 0.0243** | 0.5450 ± 0.0133 |
| Cross-line `similarity_mae` | 0.6501 ± 0.0130 | **0.4012 ± 0.0541** | 0.4296 ± 0.0481 |

Paired differences (same split, model minus reference):

- **B − A**: `mse_de` −0.0151 ± 0.0051, lower in all five splits and mostly from
  RPE1 (−0.0278 ± 0.0027). `response_pearson` +0.041 ± 0.019 and cross-line
  `similarity_mae` −0.249 ± 0.055, also consistent in all five splits. The
  cell-line embedding mainly helps the model separate the two lines' responses.
- **C − B**: `mse_de` +0.0025 ± 0.0098 (lower in 2 of 5 splits), `response_pearson`
  −0.028 ± 0.028, `similarity_mae` +0.028 ± 0.086. Under this protocol and budget
  the linear DepMap projection gives no consistent gain over B.
- `pearson_de` is slightly lower for B and C than A; differences are within one SD.

Splits share targets and are not independent biological replicates; SDs are
descriptive, not significance tests. See [report.md](test_summary/report.md) for
per-line scores and metric definitions.

## Source and protocol

- Code: commit `938846bf2dde7502cf5dd722701a3fb51c7a0af2` (branch
  `replogle-10-epochs`), [training config](../../../configs/replogle_training.json)
  with 10 epochs; all other values as committed. Selection: lowest validation
  `mse_de`; `configuration.json` records evaluator source hashes.
- Data bundle `replogle-paired-singlecell-v1`, bundle SHA256
  `8f27751a4ae187f40b45b19fbf92b8f9ea980a70582e3fdbf5048befb42f7028`.
- Variant C DepMap prior: committed `priors/depmap/morph_depmap.npz`, SHA256
  `9223da67cae34929ff6780aa40e07052b4e92dde684c2c76fb633a87fd648507`.
- Graphs: `mean_absolute_within_line_pearson_v1`, shared across A/B/C per split.
- Evaluation: [evaluation config](../../../configs/replogle_evaluation.json) with only
  the permitted runtime fields changed (`device: cuda`, `threads: 16`); order was
  validation gate → test → verify → summary.
- Environment: Windows, RTX 3060, Python 3.10.22, PyTorch 2.11.0+cu128, NumPy 1.26.4.

Selected best epochs (validation `mse_de`), splits 1–5:
A 3/10/4/6/5, B 1/10/9/6/2, C 3/2/3/8/3.

## Files

`test_summary/` is the evaluator's `summary` output copied unchanged.
`manifest.json` lists the SHA256 of every file in it; `.gitattributes` stores
these files byte-for-byte so those hashes remain valid. `similarity_split{1-5}`
PNG/PDF plot measured vs predicted cross-line response similarity per target.

Checkpoints, per-run prediction arrays, run logs and raw expression remain outside
Git in the external runtime root; `configuration.json` keeps the hash of each
per-run test manifest they came from.
