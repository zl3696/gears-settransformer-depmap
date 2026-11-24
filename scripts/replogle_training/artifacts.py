"""Content-addressed, model-independent preprocessing and portable checkpoints."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from gears.replogle_model import restore_model
from replogle_data.io import digest, require, write_json
from .graphs import build_arrays


def preprocessing(bundle, config, annotation_path, output_root):
    keys = ('coexpression_k', 'coexpression_threshold', 'go_k', 'go_threshold', 'gene2go_sha256', 'de_genes')
    inputs = dict(data=bundle.contract, graph_config={k: config[k] for k in keys},
                  implementation_sha256={name: digest(Path(__file__).with_name(name))
                                         for name in ('graphs.py', 'dataset.py')})
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    base = Path(output_root) / 'preprocessing'
    directory = base / key
    base.mkdir(parents=True, exist_ok=True)
    if not directory.exists():
        lock = base / (key + '.lock')
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise ValueError(f'Preprocessing is already being built: {lock}') from error
        os.close(descriptor)
        try:
            if not directory.exists():
                arrays = build_arrays(bundle, config, annotation_path)
                with tempfile.TemporaryDirectory(dir=base, prefix='building-') as temporary:
                    stage = Path(temporary) / 'payload'
                    stage.mkdir()
                    np.savez_compressed(stage / 'arrays.npz', **arrays)
                    write_json(stage / 'metadata.json', dict(input_key=key, inputs=inputs,
                               arrays_sha256=digest(stage / 'arrays.npz')))
                    stage.rename(directory)
        finally:
            lock.unlink(missing_ok=True)
    metadata = json.loads((directory / 'metadata.json').read_text())
    require(metadata['input_key'] == key and metadata['inputs'] == inputs, 'Preprocessing contract differs')
    require(digest(directory / 'arrays.npz') == metadata['arrays_sha256'], 'Preprocessing cache is corrupted')
    with np.load(directory / 'arrays.npz', allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    return arrays, dict(key=key, path=str(directory), arrays_sha256=metadata['arrays_sha256'])


def save_checkpoint(path, model, config, contract, epoch, metric, references):
    path = Path(path)
    payload = dict(format_version=1, variant=model.variant, model_config=config,
                   contract=contract, epoch=epoch, selection_value=metric,
                   depmap_metadata=model.depmap_metadata,
                   state_dict={k: value.detach().cpu().clone() for k, value in model.state_dict().items()},
                   control_means=torch.as_tensor(references).cpu())
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    path.with_suffix(path.suffix + '.sha256').write_text(digest(path) + '\n')


def load_checkpoint(path, device='cpu', expression_genes=None, perturbation_genes=None, expected_contract=None):
    path = Path(path)
    require(digest(path) == path.with_suffix(path.suffix + '.sha256').read_text().strip(), 'Checkpoint hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(payload['format_version'] == 1, 'Unsupported checkpoint format')
    if expected_contract is not None:
        require(payload['contract'] == expected_contract, 'Checkpoint data contract differs')
    model = restore_model(payload, device, expression_genes, perturbation_genes)
    return model, payload
