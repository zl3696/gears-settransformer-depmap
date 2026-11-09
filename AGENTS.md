# GEARS Set Transformer / DepMap project

## Overview
- `gears/` contains the GEARS model, data interface, training, and evaluation code.
- The Set Transformer aggregation is enabled with `pert_agg="set_transformer"`;
  the default `"sum"` path is retained. DepMap integration remains pending.
- `splits/norman/` is the frozen shared condition-split protocol for all models.
- New models use corrected shared coexpression batching. Historical checkpoints
  without a batching setting load in legacy mode; never silently reinterpret them.
- In the local course layout, also read the [workspace instructions](../../AGENTS.md).

## Development rules
- Preserve existing model behavior unless a requested change requires otherwise.
- Keep all aliases/cells of a condition together; every held-out pair retains
  both constituent singles in training. Fit preprocessing only on training data.
- Keep downloaded expression data, runtime caches, checkpoints, and results out
  of Git. In the local course workspace, write them under `Codex Outputs/`.
- Treat committed split metadata as immutable protocol source; version a changed
  protocol explicitly rather than editing frozen manifests in place.
- Keep annotations and technical documentation in English. Do not edit personal
  notebooks or assignment files without explicit authorization.

## Commands and documentation map
- [Model variants](docs/model-variants.md): batching semantics, checkpoint
  compatibility, experiment configuration and synthetic regression checks.
- [Norman splits](docs/norman-splits.md): split rules, provenance, file map,
  GEARS integration limits, and portable verify/generate/test commands.
- `scripts/norman_splits.py verify` audits hashes and reconstructs assignments.
- `scripts/norman_splits.py generate --output-dir /absolute/external/path`
  rebuilds a snapshot outside the checkout; add `--temp-dir` for local scratch.
- `python -m unittest discover -s /absolute/repository/tests -p 'test_norman_splits.py' -v`
  runs the split regression suite from any directory.
- `scripts/norman_train.py <run_id> --data-path ... --output-root ...` trains one
  planned run with the frozen controls, shared graphs and
  `configs/norman_training.json`; the test set is only scored with `--test`.
- `python -m pytest tests/test_norman_protocol.py tests/test_set_transformer.py tests/test_coexpression_batching.py`
  runs the training-protocol and model checks on synthetic data.
- [README](README.md): upstream GEARS installation and examples.
- Final test scoring and DepMap variants remain TODO.
