# GEARS Set Transformer / DepMap project

## Overview
- `gears/` contains the GEARS model, data interface, training, and evaluation code.
- The Set Transformer aggregation is enabled with `pert_agg="set_transformer"`;
  the default `"sum"` path is retained. DepMap integration remains pending.
- `splits/norman/` is the frozen shared condition-split protocol for all models.
- Additive and fitted linear baselines are fitted and evaluated on all five
  validation/test panels; matched neural accuracy comparisons remain pending.

## Development rules
- Preserve existing model behavior unless a requested change requires otherwise.
- Keep all aliases/cells of a condition together; every held-out pair retains
  both constituent singles in training. Fit preprocessing only on training data.
- Keep downloaded expression data, runtime caches, checkpoints, and new runtime
  outputs out of Git. In the local course workspace, write them under `Codex Outputs/`.
  The explicitly requested `results/norman/additive/` and `results/norman/linear/`
  are curated results snapshots; keep future exports deliberate and documented.
- Preserve `scripts/norman_additive/`, `scripts/norman_linear/`, and their historical
  result hashes. Use portable entry points with external runtime roots; version
  numerical changes.
- Treat committed split metadata as immutable protocol source; version a changed
  protocol explicitly rather than editing frozen manifests in place.
- Keep annotations and technical documentation in English. Do not edit personal
  notebooks or assignment files without explicit authorization.

## Commands and documentation map
- [Norman splits](docs/norman-splits.md): split rules, provenance, file map,
  GEARS integration limits, and portable verify/generate/test commands.
- `scripts/norman_splits.py verify` audits hashes and reconstructs assignments.
- `scripts/norman_splits.py generate --output-dir /absolute/external/path`
  rebuilds a snapshot outside the checkout; add `--temp-dir` for local scratch.
- `python -m unittest discover -s /absolute/repository/tests -p 'test_norman_splits.py' -v`
  runs the split regression suite from any directory.
- [README](README.md): upstream GEARS installation and examples.
- [Norman additive baseline](docs/norman-additive.md): fitting/scoring rules,
  immutable source snapshot, portable verify/test/rerun commands, and dependencies.
- [Additive baseline results](results/norman/additive/README.md): five-split
  validation/test scores, sample statistics, artifacts, and exposure limitations.
- `python /absolute/repository/scripts/norman_additive.py verify` validates the
  committed export without reading expression outcomes.
- [Fitted linear workflow](docs/norman-linear.md): explicit pair adaptation,
  preserved sources, portable tests/verification, and full source-data reproduction.
- [Fitted linear results](results/norman/linear/README.md): five fitted models,
  predictions, training means, separate validation/test results, and paired comparisons.
- `python /absolute/repository/scripts/norman_linear.py verify` checks the compact
  linear export; `reproduce --runtime-root /absolute/external/path --source /absolute/data.h5ad`
  reconstructs the completed benchmark outside the checkout.
- Formal neural training/evaluation commands and the schedule-aware adapter remain TODO.
