"""Read sparse source expression without loading layers or outcome-derived metadata."""

from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_table(path, fields, rows):
    import csv
    path = Path(path)
    with path.open('wb') as raw:
        if path.suffix == '.gz':
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream, lineterminator='\n')
                    writer.writerow(fields)
                    writer.writerows(rows)
        else:
            with io.TextIOWrapper(raw, encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(fields)
                writer.writerows(rows)


def read_column(group, key):
    obj = group[key]
    if isinstance(obj, h5py.Group):
        codes = obj['codes'][:]
        require(np.all(codes >= 0), f'Missing categorical values: {key}')
        return obj['categories'][:].astype(str)[codes]
    if '__categories' in group and key in group['__categories']:
        codes = obj[:]
        require(np.all(codes >= 0), f'Missing categorical values: {key}')
        return group['__categories'][key][:].astype(str)[codes]
    return obj[:].astype(str)


def read_index(group):
    key = group.attrs.get('_index', '_index')
    if isinstance(key, bytes):
        key = key.decode()
    return read_column(group, key if key in group else 'index')


def canonical(label):
    tokens = label.split('+')
    genes = [x for x in tokens if x != 'ctrl']
    require(all(tokens) and len(tokens) <= 2 and len(genes) <= 1,
            f'Expected a single-target condition or ctrl: {label}')
    require(label == 'ctrl' or (len(genes) == 1 and genes[0].strip() == genes[0]),
            f'Malformed condition: {label}')
    return genes[0] if genes else 'ctrl'


def metadata(path):
    with h5py.File(path, 'r') as handle:
        obs, var = handle['obs'], handle['var']
        source_labels = read_column(obs, 'condition')
        labels = np.asarray([canonical(x) for x in source_labels])
        ids = read_column(var, 'gene_id') if 'gene_id' in var else read_index(var)
        symbols = read_column(var, 'gene_name')
        barcodes = read_index(obs)
        shape = tuple(int(x) for x in handle['X'].attrs['shape'])
        require(shape == (len(labels), len(ids)), 'Source metadata and matrix shapes differ')
        require(handle['X'].attrs['encoding-type'] == 'csr_matrix', 'Expected sparse CSR X')
        require(len(set(ids)) == len(ids), 'Expression Ensembl identifiers must be unique')
        require(len(set(barcodes)) == len(barcodes), 'Source cell identifiers are not unique')
        require('ctrl' in labels, 'Source has no controls')
    counts = Counter(labels)
    aliases = {g: sorted(set(source_labels[labels == g])) for g in counts}
    return dict(path=Path(path).resolve(), labels=labels, source_labels=source_labels,
                gene_ids=ids, symbols=symbols, barcodes=barcodes,
                counts=counts, aliases=aliases, shape=shape)


def read_csr(handle, start, stop):
    group = handle['X']
    pointers = group['indptr'][start:stop + 1].astype(np.int64)
    begin, end = int(pointers[0]), int(pointers[-1])
    return csr_matrix((group['data'][begin:end], group['indices'][begin:end], pointers - begin),
                      shape=(stop - start, int(group.attrs['shape'][1])))


def source_blocks(source, rows, columns, block_rows=1024):
    """Yield selected source rows, retaining exact float32 values and ordered columns."""
    with h5py.File(source, 'r') as handle:
        for offset in range(0, len(rows), block_rows):
            selected = rows[offset:offset + block_rows]
            start, stop = int(selected[0]), int(selected[-1]) + 1
            block = read_csr(handle, start, stop)[selected - start][:, columns].tocsr()
            block.sum_duplicates()
            block.eliminate_zeros()
            block.sort_indices()
            require(block.dtype == np.float32, 'Source X must already be float32')
            require(np.isfinite(block.data).all() and np.all(block.data >= 0),
                    'Source contains nonfinite or negative processed expression')
            yield offset, block
