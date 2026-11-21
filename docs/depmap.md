# Optional DepMap residuals

## Architecture and API

`GEARS.model_initialize(use_depmap=True, depmap_path="/absolute/features.npz")`
adds fixed gene dependency profiles through a learned linear residual. It is
independent of `pert_agg`, which still accepts `sum`, `set_transformer`, or
`residual_set_transformer`. Train a separate model for each configuration.

After GO propagation and before perturbation aggregation:

```text
h'_g = h_g + available_g * (W d_g + b)
```

`W` and `b` start at zero. Shared weights and the random-number stream match the
corresponding model without DepMap. Only genes used in the current batch are
projected. The projection and GEARS weights train jointly; input profiles remain
fixed device-aware buffers. Masking happens after projection so missing genes
receive no bias correction. Controls receive no injected DepMap perturbation.
Original BatchNorm behavior, decoder, loss, and GO propagation are preserved.

`use_depmap=False` is the default, including for old checkpoints. It creates no
DepMap parameters or buffers and reads no feature file. An enabled model must
receive either `depmap_path` or a validated `AlignedDepMap` as `depmap_data`.

This is a gene prior, not a cell-line identifier. Optional
[cell-line label embeddings](cell-line-embedding.md) are implemented on the
subsequent feature branch; full joint Replogle K562/RPE1 training remains pending.

## Feature preparation and alignment

The selected source is MORPH's
[processed data](https://github.com/uhlerlab/MORPH/blob/main/test_demo.md):
[`depmap_crispr_gene_effect_processed.pkl`](https://drive.google.com/file/d/1qhPMgarLgEgOkh8KwInrJuNoIAiMk46-/view).
The inspected file contains 18,610 gene symbols mapped to 1,100-dimensional
float64 vectors, with no nonfinite entries. Its SHA256 is
`83dc5007b1d9d58f60d55e158a5338a21250adfa600715ebb1773067da040c2a`.

The file does not identify the reference cell lines or DepMap release. Preserve
its original vector order and use the source hash to identify this exact input;
do not invent cell-line names or claim a particular release. TODO: recover the
reference-cell-line identities and upstream release from the source authors.

`scripts/prepare_depmap.py` converts this trusted pickle to an NPZ containing
`genes`, `values`, and JSON `metadata`, plus a JSON provenance/coverage report.
Pickle conversion is only for trusted source files. Model initialization loads
NPZ with `allow_pickle=False`. Preparation preserves values apart from float32
conversion; it fits no scaling, PCA, imputation, or other target-data transform.

`gears/depmap.py` validates dimensions, finite real values, unique identifiers,
and content hashes, then matches exact case-sensitive symbols to `pert_list`.
Expression-output order is not used for alignment. No automatic alias mapping
is performed. Missing genes have zero rows and a false availability mask;
partially missing profiles, ambiguous identifiers, and zero total overlap fail.
Source and prepared bundles must remain outside the source checkout.

On the frozen Norman vocabulary, 9,734/9,853 graph nodes match. Among 102
experimental target genes, 99 match; `C19orf26`, `C3orf72`, and `SAMD1` use the
missing-gene fallback. Coverage is not a measure of predictive accuracy.

From any directory, after setting absolute locations:

```bash
repo_dir='/absolute/path/to/gears-settransformer-depmap'
runtime_dir='/absolute/path/to/external/runtime'
python "$repo_dir/scripts/prepare_depmap.py" \
  --input-pickle "$runtime_dir/data/depmap/depmap_crispr_gene_effect_processed.pkl" \
  --output "$runtime_dir/data/depmap/morph_depmap.npz" \
  --source-version 'MORPH processed file; identify exact version by recorded hash' \
  --perturbation-nodes "$repo_dir/splits/norman/perturbation_nodes.csv"
```

Outputs are never silently overwritten. Reuse the existing validated bundle or
choose a new output name when preparing another release.

## Checkpoints

`model.pt` includes projection weights, aligned feature values, and the mask.
`config.pkl` records ordered perturbation and expression genes, feature width,
matched/missing coverage, source provenance, and the aligned-content hash.
`load_pretrained()` rebuilds the model from these files without the original
NPZ/download. It rejects reordered genes or modified features/masks. Use the
wrapper's save/load methods to retain these checks; a bare state dictionary
does not carry the gene-order contract. As with existing GEARS checkpoints,
only load trusted checkpoint files.

## Six-mode Norman comparison

`configs/norman_depmap_training.json` and `configs/norman_depmap_runs.csv` define
30 runs: three aggregation choices, DepMap on/off, and five frozen splits.
Model names are `gears`, `gears_depmap`, `gears_set_transformer`,
`gears_set_transformer_depmap`, `gears_residual_set_transformer`, and
`gears_residual_set_transformer_depmap`.

They retain the current shared preprocessing, graphs, training seed/budget,
and validation top-20 DE MSE checkpoint selection. Normalized MSE remains
recorded. Outputs use `outputs/norman-benchmark/depmap_comparison_v1/` under the
external output root. Existing run plans and frozen split files are unchanged.

```bash
python "$repo_dir/scripts/run_norman_all.py" \
  --protocol "$repo_dir/configs/norman_depmap_training.json" \
  --depmap-features "$runtime_dir/data/depmap/morph_depmap.npz" \
  --data-path "$runtime_dir/data" --output-root "$runtime_dir" \
  --splits 1 --dry-run
```

Remove `--dry-run` to train. Omit `--splits` for all five splits; use `--models`
to select any subset. The default device is CUDA; `--device cpu` is supported.
Test data is accessed only with `--test`. For one run:

```bash
python "$repo_dir/scripts/norman_train.py" gears_depmap_split1_train101 \
  --protocol "$repo_dir/configs/norman_depmap_training.json" \
  --depmap-features "$runtime_dir/data/depmap/morph_depmap.npz" \
  --data-path "$runtime_dir/data" --output-root "$runtime_dir"
```

Results record feature-bundle and aligned-feature hashes and coverage. The
launcher rejects an existing result if its protocol, run plan, or DepMap input
differs. It never silently reuses a run trained with another feature file.

## Verification

```bash
PYTHONPATH="$repo_dir" PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -p no:cacheprovider "$repo_dir/tests" -q
```

Tests cover disabled/legacy compatibility, exact baseline initialization,
alignment under row reordering, missing genes with nonzero projection bias,
controls, singles/pairs, set-order and evaluation-batch invariance, gradients,
fixed buffers, short training in every aggregation mode, save/load without the
source file, corrupted/reordered checkpoint rejection, and portable launchers.
The regression reference `8edf2ff` is the batching-fix commit in the current
remote history; the older `c154fa1` disappeared in the team's history rewrite.

CPU tests pass. Full DepMap benchmark training and biological evaluation remain
pending; synthetic training and real-feature forward/backward checks establish
implementation behavior only. No additional packages are required beyond the
existing repository requirements and pytest for tests.
