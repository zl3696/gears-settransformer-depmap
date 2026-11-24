"""Validated, identifier-aligned DepMap priors. Runtime bundles never use pickle."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def _identifiers(items, name):
    items = list(items)
    if not items or any(not isinstance(x, str) or not x or x.strip() != x for x in items):
        raise ValueError(f"{name} must be nonempty strings without surrounding whitespace")
    if len(set(items)) != len(items):
        raise ValueError(f"Duplicate {name}; resolve ambiguous identifiers explicitly")
    return items


def _matrix(values):
    values = np.asarray(values)
    if values.ndim != 2 or not all(values.shape) or values.dtype.kind not in 'fiu':
        raise ValueError("DepMap values must be a nonempty, real numeric matrix")
    if not np.isfinite(values).all():
        raise ValueError("DepMap profiles contain nonfinite values; partial profiles are not imputed")
    with np.errstate(over='ignore'):
        values = np.array(values, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(values).all():
        raise ValueError("DepMap values overflow float32")
    return values


def content_sha256(genes, values, mask=None):
    """Digest the exact ordered model inputs, independently of archive timestamps."""
    digest = hashlib.sha256(json.dumps(list(genes), ensure_ascii=True).encode())
    digest.update(np.asarray(values.shape, dtype='<i8').tobytes())
    digest.update(np.asarray(values, dtype='<f4').tobytes(order='C'))
    if mask is not None:
        digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DepMapFeatures:
    genes: list
    values: np.ndarray
    metadata: dict


@dataclass
class AlignedDepMap:
    values: np.ndarray
    mask: np.ndarray
    metadata: dict

    def validate(self, perturbation_genes):
        genes = _identifiers(perturbation_genes, 'perturbation genes')
        values = _matrix(self.values)
        mask = np.asarray(self.mask)
        if self.metadata.get('perturbation_genes') != genes:
            raise ValueError("DepMap perturbation-gene order differs from the model")
        if values.shape != (len(genes), self.metadata.get('feature_dim')):
            raise ValueError("DepMap feature dimensions differ from metadata")
        if mask.dtype != np.bool_ or mask.shape != (len(genes),):
            raise ValueError("DepMap availability mask must be a boolean vector in perturbation order")
        if not mask.any():
            raise ValueError("No perturbation genes match the DepMap features")
        if np.any(values[~mask] != 0):
            raise ValueError("Unavailable DepMap rows must contain zeros")
        if content_sha256(genes, values, mask) != self.metadata.get('aligned_sha256'):
            raise ValueError("DepMap aligned features or mask differ from their saved hash")
        if (self.metadata.get('matched_genes') != int(mask.sum())
                or self.metadata.get('missing_genes') != [g for g, present in zip(genes, mask) if not present]):
            raise ValueError("DepMap coverage metadata differs from the mask")
        return self


def save_features(path, genes, values, source_metadata):
    """Write a portable float32 bundle; caller supplies source/version provenance."""
    genes = _identifiers(genes, 'DepMap genes')
    values = _matrix(values)
    if values.shape[0] != len(genes):
        raise ValueError("DepMap gene count does not match feature rows")
    metadata = dict(source_metadata)
    metadata.update(schema_version=1, gene_count=len(genes), feature_dim=values.shape[1],
                    content_sha256=content_sha256(genes, values))
    serialized = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing another feature release.
    with path.open('xb') as stream:
        np.savez_compressed(stream, genes=np.asarray(genes), values=values,
                            metadata=np.asarray(serialized))
    return DepMapFeatures(genes, values, metadata)


def load_features(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'genes', 'values', 'metadata'}:
            raise ValueError("Expected a prepared DepMap NPZ with genes, values and metadata")
        if archive['genes'].ndim != 1:
            raise ValueError("DepMap genes must be a one-dimensional array")
        genes = _identifiers(archive['genes'].tolist(), 'DepMap genes')
        values = _matrix(archive['values'])
        metadata = json.loads(str(archive['metadata'].item()))
    if not isinstance(metadata, dict) or metadata.get('schema_version') != 1:
        raise ValueError("Unsupported DepMap bundle schema")
    if values.shape != (len(genes), metadata.get('feature_dim')) or metadata.get('gene_count') != len(genes):
        raise ValueError("DepMap bundle dimensions differ from metadata")
    if content_sha256(genes, values) != metadata.get('content_sha256'):
        raise ValueError("DepMap bundle features differ from their saved hash")
    return DepMapFeatures(genes, values, metadata)


def align_features(features, perturbation_genes):
    """Exact, case-sensitive gene-symbol matching; absent genes get zero correction."""
    genes = _identifiers(perturbation_genes, 'perturbation genes')
    source_genes = _identifiers(features.genes, 'DepMap genes')
    source = _matrix(features.values)
    if source.shape[0] != len(source_genes):
        raise ValueError("DepMap gene count does not match feature rows")
    lookup = {gene: i for i, gene in enumerate(source_genes)}
    mask = np.asarray([g in lookup for g in genes], dtype=bool)
    values = np.zeros((len(genes), source.shape[1]), dtype=np.float32)
    for i, gene in enumerate(genes):
        if mask[i]:
            values[i] = source[lookup[gene]]
    metadata = dict(schema_version=1, source=dict(features.metadata),
                    perturbation_genes=genes, feature_dim=source.shape[1],
                    matched_genes=int(mask.sum()),
                    missing_genes=[g for g, present in zip(genes, mask) if not present],
                    aligned_sha256=content_sha256(genes, values, mask))
    return AlignedDepMap(values, mask, metadata).validate(genes)
