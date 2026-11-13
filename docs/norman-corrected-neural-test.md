# Corrected-batching Norman neural final tests

## Scope and pinned inputs

The user authorized testing the 15 newly published checkpoints on 12 November
2026. `scripts/norman_corrected_neural_test.py` evaluates three variants across
the five existing frozen splits. It performs inference only and keeps historical
baseline and neural result bundles unchanged.

`configs/norman_corrected_neural_test.json` pins every downloaded artifact by
Git blob hash and size, the expression source, split bundle, scorer, and CPU
inference policy. The source cohorts are:

| Variants | Published commit | Result namespace |
|---|---|---|
| GEARS; replacement Set Transformer | `858a96bb1fa92ffaf4bd35d4944d61b9a6960ed9` | `coexpression_shared_v1` |
| Residual Set Transformer | `b112de0833294b4ae0048549fe3d1903a974323b` | `residual_comparison_v1` |

The published model modules and graphs are loaded directly from verified external
downloads. All 15 training records contain 20 epochs, training seed 101,
`coexpression_batching="shared"`, validation selection by `mse_de`, and
`test_evaluated=false`. Testing is recorded in new local bundles; the immutable
published training records retain their original flags.

The cohorts have distinct training-protocol and run-plan hashes. Their common
model/graph hyperparameters, training budget, frozen splits, controls, and
validation selection rule match. The residual training protocol additionally
registers its third aggregation mode. These are the existing published weights;
testing does not retrain the other two modes under the residual run plan.

## Evaluation contract

Reuse the [historical matched-test contract](norman-neural-test.md): identical
test cells, 26 canonical pairs per split, frozen control inputs, 5,045 genes,
20 DE indices per pair, measured means, equal pair weighting, and unchanged
`mse`, `mse_de`, `pearson`, `pearson_de` scorer. Read only assigned training
controls from H5AD; reuse the verified baseline test truth arrays.

The new runner imports the unchanged historical runner's pairing, control-row
reader, streaming predictor, and scoring helpers. Historical entry points,
protocol hashes, baseline numerical sources, and output bundles stay intact.

Use CPU, four threads, batch size 32 and increasing target source-row order.
Preserve each checkpoint's published shared coexpression propagation. The earlier
legacy model's batch-position defect and normalized-DE checkpoint selection do
not describe these new checkpoints. The published trainers used test batch size
128; this evaluator fixes 32 to match the earlier baseline inference schedule.
Corrected shared propagation is intended to be invariant to evaluation batch
position and size; BatchNorm remains in evaluation mode.

A local audit of the first 32 split-1 cells passed for all three published model
types: reversing batch order gave identical predictions; changing batch size
from 32 to 7 changed predictions by at most `1.431e-6` (absolute/relative
tolerance `1e-5`). Evidence is saved as `batching_audit.json` under the output root.

No training, checkpoint reselection, normalization change, DE recalculation, or
test-dependent tuning occurs. Outcomes were previously exposed in baseline
tests, overlapping validation panels, and descriptive PCA. The five panels
measure split variability, not independent biological or training-seed
replication. Keep this exposure boundary when developing later variants.

## Outputs and verification

Under an explicit external runtime root:

- `data/norman-neural-checkpoints-858a96bb/` and
  `data/norman-neural-checkpoints-b112de08/`: pinned source, checkpoints and
  download manifests.
- `outputs/norman-corrected-neural-test/<run_id>/`: pre-inference configuration,
  condition-mean predictions and counts, exact scoring inputs, per-pair scores,
  cell/gene lists, frozen prediction schedule, timing and verification evidence.
- `outputs/norman-corrected-neural-test/test_summary/`: all five models' split
  scores, mean/sample variance/SD, paired differences, and `report.md`.

Every run checks training identity and recorded validation-minimum selection,
graph hashes, strict state-dict loading, exact baseline targets/cells/genes,
the saved frozen control/batch schedule, historical additive-score reproduction,
and independent NumPy metric calculations. Summary statistics use `ddof=1`.
Six new synthetic tests guard source routing and training-record acceptance;
the seven historical tests guard control pairing, excluded-row access, streaming
aggregation, hash tampering, and metric semantics.

`verify` rescans saved artifacts and scores without model inference. `regenerate`
repeats inference and requires identical scoring arrays and metrics. A completed
`run` verifies existing bundles before skipping them. Inspect incomplete output
before replacing it. Source/protocol changes invalidate existing bundle checks.

## Completed local evaluation

All 15 test panels completed and verified on 12 November 2026. They contain
105,882 cell predictions in total (35,294 per neural variant), aggregated into
26 pair scores per split. All 13 synthetic checks pass. Split-1 full inference
regeneration passes with identical scoring arrays and metrics for each variant.
The historical ten-run evaluator also still passes verification.

The [committed comparison report](../results/norman/corrected-neural/test_summary/report.md)
contains five-model means/SD; sibling `split_metrics.csv`, `metric_summary.csv`,
and `paired_difference_summary.csv` preserve full precision.
Additive has the best mean on all four metrics. Among neural variants, corrected
GEARS has the lowest mean DE MSE; replacement Set Transformer has the lowest
mean all-gene MSE; residual Set Transformer has the highest mean all-gene Pearson.
These are descriptive comparisons under the existing outcome-exposure limits.
The existing Python 3.12 GEARS environment supplies all dependencies.

The [compact result export](../results/norman/corrected-neural/README.md) preserves
all 15 configurations, summaries, per-pair scores and verification records, plus
the three regeneration reports and five-model summary. `provenance.json` hashes
every retained file and its evaluator dependencies. Runtime arrays and source
checkpoints stay external; original summary hashes also cover omitted files.

## Commands from any directory

```bash
NORMAN_PYTHON='/Users/lisa/Desktop/Barnard/26 Fall/ML for Functional Genomics/ML Project /Codex Outputs/.venvs/gears/bin/python'
NORMAN_REPOSITORY='/Users/lisa/Desktop/Barnard/26 Fall/ML for Functional Genomics/ML Project /Code/gears-settransformer-depmap'
NORMAN_RUNTIME='/Users/lisa/Desktop/Barnard/26 Fall/ML for Functional Genomics/ML Project /Codex Outputs'

PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" -m unittest discover -s "$NORMAN_REPOSITORY/tests" -p 'test_norman*neural_test.py' -v
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_corrected_neural_test.py" download --runtime-root "$NORMAN_RUNTIME"
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_corrected_neural_test.py" run --runtime-root "$NORMAN_RUNTIME"
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_corrected_neural_test.py" verify --runtime-root "$NORMAN_RUNTIME"
PYTHONDONTWRITEBYTECODE=1 "$NORMAN_PYTHON" "$NORMAN_REPOSITORY/scripts/norman_corrected_neural_test.py" summary --runtime-root "$NORMAN_RUNTIME"
```

To check only the compact export, Python's standard library is sufficient. Set
`NORMAN_REPOSITORY` to an absolute checkout path, then run from any directory:

```bash
python3 - "$NORMAN_REPOSITORY" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
repo = Path(sys.argv[1])
root = repo / 'results/norman/corrected-neural'
manifest = json.loads((root / 'provenance.json').read_text())
for base, field in [(root, 'export_file_sha256'), (repo, 'repository_source_sha256')]:
    for name, expected in manifest[field].items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected, name
print('Compact export and evaluator source hashes verified')
PY
```

`run`, `verify`, and `regenerate` accept `--model gears`,
`--model gears_set_transformer`, or `--model gears_residual_set_transformer`,
and `--split-seed 1` through `5`. Use `regenerate` with one model/split for a
full checkpoint-reload reproducibility check.
