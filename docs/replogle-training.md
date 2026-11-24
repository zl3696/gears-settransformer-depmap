# Pooled Replogle training

## Entry point and scope

`scripts/replogle_train.py` trains an independent model for each task, variant,
and split. It consumes the immutable [single-cell bundle](replogle-data.md),
including its control assignments and sampling weights. It does not score the
final test set. Historical `GEARS`, `PertData`, and Norman checkpoint paths are
unchanged; this is a separate consumer with corrected shared-graph batching.

| CLI task | Training targets | Validation targets | Test targets, reserved |
|---|---|---|---|
| `main` (`joint_unseen`) | 591 K562 + the same 591 RPE1 | Same 126 in both lines | Same 126 in both lines |
| `k562_to_rpe1` | All 843 K562 + 591 RPE1 | 126 RPE1 | 126 RPE1 |
| `rpe1_to_k562` | All 843 RPE1 + 591 K562 | 126 K562 | 126 K562 |

All controls are training observations. Both contexts occur in every training
task; transfer holds out a known target's response in the destination context.
Five split seeds and three variants yield 45 separate models, each with training
seed 101. Never use transfer outcomes to adapt a main-run model or its settings.

## Variants and model behavior

`gears/replogle_model.py` subclasses the existing GEARS model and reuses its
original sum fusion, MLPs, and gene-specific/cross-gene decoder. Its forward path
supports one target per cell; it is not a Norman double-perturbation adapter.

| Variant | Learned additions | Fixed inputs |
|---|---|---|
| A | None | Shared expression/GO graphs and same-line control input |
| B | Two cell-line embeddings | Same as A |
| C | B plus a linear DepMap projection | Same as B plus aligned DepMap profiles/mask |

B/C add the cell-line vector to the fused perturbation vector before nonlinear
decoding. C adds a projected fixed profile to the selected target embedding
after GO propagation and before fusion. Missing profiles give zero correction,
including projection bias; no target is removed. Controls receive neither
correction. A does not receive an explicit context feature; labels still govern
control matching, condition grouping, loss references, and reporting for all models.

Both additions start at zero and preserve the base initialization/RNG sequence,
so A/B/C start with identical predictions. Fixed graphs and C's profiles/mask
are registered buffers. `gears/depmap.py` is copied unchanged from team commit
`b24cd8135d2a3d7c90eb20c1fecff34e702be9ac` (Git blob
`3c164f1b5b6bf57eafedae1c3ba21f78dd6f2730`). The upstream DepMap release and
reference-cell identities remain unverified; see the prepared prior provenance.

## Fitting and checkpoint selection

- `dataset.py` verifies frozen hashes/assignments, maps paired target IDs to GO
  nodes by symbol, and reads sparse H5AD rows. Only train/validation expression
  is exposed. Hash verification can read test file bytes and index metadata;
  it does not decode test expression. `--preload` caches only permitted rows as
  sparse CSR in RAM; otherwise expression is read on demand.
- `graphs.py` constructs a GO Jaccard graph from trusted, SHA-pinned
  `gene2go_all.pkl`. The coexpression graph uses absolute Pearson correlations
  over this task/split's training cells only. Original GEARS conventions retain
  up to `k + 1` edges above threshold, with deterministic gene-index ties.
  Defaults are k=20, GO threshold=0.1 and coexpression threshold=0.4.
- Training-only nonzero-expression masks and per-line control means define the
  original fourth-power/direction loss, applied to individual observed cells.
  Each `(cell line, target)` is a separate condition, including controls. Loss
  averages within conditions, then over conditions present in the batch.
- Each epoch samples `N_train` cells with replacement using the exact frozen
  weights: equal line mass, equal condition mass within line, equal cell mass
  within condition. It preserves the sample budget and avoids singleton batches.
- Validation averages predictions by condition, calculates MSE per condition,
  averages conditions within each line, then weights available lines equally.
  Transfer validation therefore uses its destination line alone. Default best
  checkpoint selection is the lowest validation top-20-DE MSE (`mse_de`).
- Validation DE genes are scoring-only: rank absolute Welch t statistics against
  same-line training controls, breaking ties by expression index. A singleton
  condition has variance zero; zero standard error with nonzero difference has
  infinite absolute rank score. This is an explicit ranking convention, not a
  claim of reliable low-count p-values. No released full-data DE list is reused.
- No extra normalization, PCA, feature standardization, or test-derived fitting
  occurs. Upstream HVG selection, earlier PCA exposure, and unverified upstream
  normalization remain recorded limitations of the prepared data.

`configs/replogle_training.json` fixes 20 epochs, batch size 32, hidden size 64,
Adam lr=0.001, weight decay=0.0005, per-epoch LR multiplier 0.5, direction weight
0.1, gradient-value clipping at 1, and validation batch size 128. Change a copied
configuration before formal runs and use a new output root; do not adapt it to
test results. No full Replogle fit or final-test evaluation has been completed.

## Commands from any directory

Set these variables to absolute locations. Dependencies are the existing GEARS
stack (PyTorch, PyG, Scanpy, anndata, h5py, NumPy, SciPy, pandas). Tested locally
with Python 3.12, PyTorch 2.2.2, NumPy 1.26.4 and SciPy 1.14.1.

```bash
repo_dir='/absolute/path/to/gears-settransformer-depmap'
runtime_dir='/absolute/path/to/Codex Outputs'
python_bin="$runtime_dir/.venvs/gears/bin/python"
common=(--data-dir "$runtime_dir/outputs/replogle-paired-singlecell-v1"
        --output-root "$runtime_dir/outputs/replogle-training-v1"
        --gene2go "$runtime_dir/tmp/morph-gears-review/gene2go_all.pkl")

# Each command launches exactly one model. Choose A, B, or C as needed.
"$python_bin" "$repo_dir/scripts/replogle_train.py" train "${common[@]}" \
  --task main --variant A --split-seed 1
"$python_bin" "$repo_dir/scripts/replogle_train.py" train "${common[@]}" \
  --task k562_to_rpe1 --variant B --split-seed 1
"$python_bin" "$repo_dir/scripts/replogle_train.py" train "${common[@]}" \
  --task rpe1_to_k562 --variant C --split-seed 1 \
  --depmap-features "$runtime_dir/outputs/replogle-depmap-v1/morph_depmap.npz"

# Print all 45 commands; printing does not create or run models.
"$python_bin" "$repo_dir/scripts/replogle_train.py" plan "${common[@]}" \
  --depmap-features "$runtime_dir/outputs/replogle-depmap-v1/morph_depmap.npz"
```

Add `--execute` to `plan` to run sequentially. Narrow it with `--tasks main`,
`--variants A B`, or `--split-seeds 1`. A/B never load DepMap even if the argument
is supplied. `--device auto` chooses CUDA, then MPS, then CPU; an explicit
`--device cpu` is useful for reproducibility. `--preload` trades RAM for faster
random-row input access; the complete expression matrix is never densified.

## Runtime artifacts and verification

Outputs must be outside the checkout and frozen bundle:

- `preprocessing/<content-key>/`: graph/mask/control and validation-scoring
  arrays, input contract, and hashes. A/B/C share this cache for a task/split;
  different protocols/splits have separate keys. A live builder lock fails
  clearly; start parallel variants after the common cache has been built.
- `runs/<protocol>_<variant>_split<n>_train101/`: `run.json` records configuration,
  data/code hashes, Git commit, environment, and state; `history.json` records
  epoch loss/validation metrics; `results.json` records selected validation and
  checkpoint hash. Errors leave `failure.json`.
- `best.pt` and `last.pt`, each with a SHA sidecar: ordered expression/GO axes,
  context names, model configuration/state, graphs, per-line control means and
  C's fixed features/mask/metadata. Reload checks hashes, gene order, contract,
  and aligned prior integrity; C does not need the original NPZ to restore.

Best-checkpoint validation is repeated after reload and must match. Existing
run directories are refused rather than overwritten; optimizer resume is not
implemented. Start a deliberate rerun in a new output root. Frozen
`planned_runs.csv` remains immutable; actual status belongs in runtime run files.

```bash
"$python_bin" -m unittest discover -s "$repo_dir/tests" -p 'test_replogle*.py' -v
```

The 13 data/training regression tests include two epochs for all nine task/variant
combinations, real CLI dispatch, base-model equivalence, initial prediction/RNG
matching, context conditioning, missing-prior masking, training-only graph fitting,
test-read guards, equal-line validation, cache corruption, and self-contained
checkpoint restoration. They use tiny fixtures, not the final benchmark data.
On this Mac, CLI subprocess tests require normal OpenMP shared-memory access.
