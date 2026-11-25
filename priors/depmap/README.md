# Fixed MORPH DepMap prior

`morph_depmap.npz` is the prepared fixed input for Replogle variant C (GEARS
with cell-line and DepMap embeddings). It is included in Git at the user's
request so teammates receive it with the training code. A/B do not load it.
Raw expression, runtime caches and trained checkpoints remain external.

## Contents and provenance

- `morph_depmap.npz`: 18,610 gene symbols and an 18,610 x 1,100 float32 matrix,
  with `genes`, `values` and JSON `metadata` arrays. Load with
  `gears.depmap.load_features`; it validates identifiers, dimensions and content.
- `morph_depmap.json`: portable source/hash metadata and coverage against the
  frozen 9,853-node vocabulary (9,734 matched). Its `output` is repository-relative.
- Source: MORPH's `depmap_crispr_gene_effect_processed.pkl`, available from the
  [MORPH processed-data link](https://drive.google.com/file/d/1qhPMgarLgEgOkh8KwInrJuNoIAiMk46-/view).
  Preparation used the published DepMap implementation at `b24cd8135d2a3d7c90eb20c1fecff34e702be9ac`.
- Values and feature order are preserved apart from the original float32
  conversion. This committed NPZ is byte-for-byte identical to the validated
  prepared bundle; it is not fitted to Replogle training/validation/test outcomes.
- The upstream DepMap release and reference-cell identities remain unverified.
  Preserve the supplied 1,100-column order; do not assign cell-line names to it.

NPZ SHA256:
`9223da67cae34929ff6780aa40e07052b4e92dde684c2c76fb633a87fd648507`

Source pickle SHA256:
`83dc5007b1d9d58f60d55e158a5338a21250adfa600715ebb1773067da040c2a`

## Usage and verification

Set `repo_dir` to the absolute checkout path and pass
`--depmap-features "$repo_dir/priors/depmap/morph_depmap.npz"` when launching
variant C. The [training guide](../../docs/replogle-training.md) provides full
A/B/C commands. The prepared single-cell bundle and pinned GO annotations are
separate inputs documented in the [data guide](../../docs/replogle-data.md).

To verify the committed bytes from any working directory:

```bash
repo_dir='/absolute/path/to/gears-settransformer-depmap'
python - "$repo_dir/priors/depmap/morph_depmap.npz" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], 'rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
expected = '9223da67cae34929ff6780aa40e07052b4e92dde684c2c76fb633a87fd648507'
if digest.hexdigest() != expected:
    raise SystemExit('DepMap bundle hash mismatch')
print('DepMap bundle SHA256 verified')
PY
```
