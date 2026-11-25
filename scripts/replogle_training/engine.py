"""Single-cell optimization with context-aware grouping and validation-only selection."""

import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn

from gears.depmap import align_features, load_features
from gears.replogle_model import ReplogleModel
from replogle_data.io import digest, require, write_json
from replogle_data.workflow import REPOSITORY
from .artifacts import load_checkpoint, preprocessing, save_checkpoint
from .dataset import Bundle, TASKS


def loss(prediction, batch, masks, control_means, target_count, direction_lambda):
    """GEARS autofocus/direction loss, grouping each (line, target) independently."""
    losses = []
    for group in np.unique(batch['groups']):
        rows = torch.as_tensor(np.flatnonzero(batch['groups'] == group), device=prediction.device)
        mask = masks[int(group)]
        actual, predicted = batch['y'][rows][:, mask], prediction[rows][:, mask]
        reference = control_means[int(group) // (target_count + 1), mask]
        losses.append((actual - predicted).pow(4).mean() + direction_lambda *
                      (torch.sign(actual - reference) - torch.sign(predicted - reference)).pow(2).mean())
    return torch.stack(losses).mean()


def validation(model, bundle, arrays, batch_size, device):
    model.eval()
    sums = np.zeros_like(arrays['validation_means'], dtype=np.float64)
    counts = np.zeros(bundle.n_groups, dtype=np.int64)
    with torch.no_grad():
        for batch in bundle.batches('val', batch_size, device=device):
            pred = model(batch['x'], batch['nodes'], batch['contexts']).cpu().numpy().astype(np.float64)
            require(np.isfinite(pred).all(), 'Validation prediction is nonfinite')
            np.add.at(sums, batch['groups'], pred)
            counts += np.bincount(batch['groups'], minlength=bundle.n_groups)
    require(np.array_equal(counts, arrays['validation_counts']), 'Validation cell counts differ')
    per_line = {}
    for line, name in enumerate(('K562', 'RPE1')):
        groups = np.flatnonzero((counts > 0) & (np.arange(bundle.n_groups) // (bundle.target_count + 1) == line))
        if not len(groups):
            continue
        predicted = sums[groups] / counts[groups, None]
        error = predicted - arrays['validation_means'][groups]
        de_error = np.take_along_axis(error, arrays['validation_de'][groups], axis=1)
        per_line[name] = dict(mse=float(np.mean(np.mean(error ** 2, axis=1))),
                              mse_de=float(np.mean(np.mean(de_error ** 2, axis=1))), conditions=len(groups))
    require(bool(per_line), 'No validation conditions')
    return dict(mse=float(np.mean([v['mse'] for v in per_line.values()])),
                mse_de=float(np.mean([v['mse_de'] for v in per_line.values()])), by_cell_line=per_line)


def choose_device(name):
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def run(data_dir, output_root, task, variant, split_seed, config_path, gene2go, depmap_features=None,
        device='auto', preload=False):
    require(variant in ('A', 'B', 'C') and task in TASKS, 'Unknown task or variant')
    output_root = Path(output_root).resolve()
    data_dir = Path(data_dir).resolve()
    require(not output_root.is_relative_to(REPOSITORY) and not output_root.is_relative_to(data_dir),
            'Runtime outputs must be outside the checkout and frozen data')
    config = json.loads(Path(config_path).read_text())
    require(config['schema_version'] == 1 and config['selection_metric'] in ('mse', 'mse_de'), 'Unsupported training protocol')
    require(config['epochs'] >= 1 and config['batch_size'] >= 2 and config['validation_batch_size'] >= 1,
            'Invalid training budget')
    require(config['threads'] >= 1 and config['de_genes'] >= 1, 'Invalid compute or DE setting')
    require(digest(gene2go) == config['gene2go_sha256'], 'GO annotation source hash differs')
    features, depmap_hash = None, None
    if variant == 'C':
        require(depmap_features is not None, 'Variant C requires --depmap-features')
        features = load_features(depmap_features)
        depmap_hash = digest(depmap_features)
    torch.set_num_threads(config['threads'])
    device = choose_device(device)
    run_id = f'{TASKS[task]}_{variant}_split{split_seed}_train{config["training_seed"]}'
    directory = output_root / 'runs' / run_id
    require(not directory.exists(), f'Run already exists: {directory}')
    bundle = Bundle(data_dir, task, split_seed, preload)
    try:
        require(len(bundle.views['train']['rows']) >= 2, 'Training requires at least two cells')
        aligned = align_features(features, bundle.perturbations) if features is not None else None
        code_paths = [Path(__file__).resolve().parents[1] / 'replogle_train.py',
                      *Path(__file__).parent.glob('*.py'), REPOSITORY / 'gears/replogle_model.py',
                      REPOSITORY / 'gears/model.py', REPOSITORY / 'gears/depmap.py',
                      * (REPOSITORY / 'scripts/replogle_data').glob('*.py')]
        code_hashes = {p.relative_to(REPOSITORY).as_posix(): digest(p) for p in code_paths}
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, capture_output=True, text=True)
        directory.mkdir(parents=True)
        provenance = dict(run_id=run_id, variant=variant, data=bundle.contract, configuration=config,
            configuration_sha256=digest(config_path), depmap_bundle_sha256=depmap_hash,
            source_sha256=code_hashes, git_commit=git.stdout.strip(),
            environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=device),
            selection_rule='Mean condition MSE, then equal mean across available validation cell lines',
            de_rule='Top genes by absolute Welch t statistic against same-line training controls; stable gene-index ties; scoring only',
            test_evaluated=False, state='preparing')
        write_json(directory / 'run.json', provenance)
        arrays, cache = preprocessing(bundle, config, gene2go, output_root)
        provenance.update(preprocessing=cache, state='training')
        write_json(directory / 'run.json', provenance)
        print(f"Variant {variant}: coexpression_method={cache['coexpression_method']}, "
              f"shared_graph_cache={cache['key']}", flush=True)
        model_config = {k: config[k] for k in ('hidden_size', 'decoder_hidden_size', 'num_go_gnn_layers', 'num_gene_gnn_layers')}
        model_config.update(num_genes=len(bundle.genes), num_perts=len(bundle.perturbations), no_perturb=False,
                            uncertainty=False, pert_agg='sum', expression_genes=bundle.genes,
                            perturbation_genes=bundle.perturbations, context_names=['K562', 'RPE1'])
        args = dict(model_config, device='cpu', G_coexpress=torch.as_tensor(arrays['coexpression_edges']),
                    G_coexpress_weight=torch.as_tensor(arrays['coexpression_weights']),
                    G_go=torch.as_tensor(arrays['go_edges']), G_go_weight=torch.as_tensor(arrays['go_weights']))
        torch.manual_seed(config['training_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['training_seed'])
        model = ReplogleModel(args, variant, aligned).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['scheduler_gamma'])
        masks = torch.as_tensor(arrays['loss_masks'], device=device)
        reference = torch.as_tensor(arrays['control_means'], device=device)
        weights = torch.as_tensor(bundle.views['train']['sampling_weights'], dtype=torch.double)
        history, best, best_epoch = [], float('inf'), None
        started = time.monotonic()
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            generator = torch.Generator().manual_seed(config['training_seed'] + epoch)
            draws = torch.multinomial(weights, len(weights), replacement=True, generator=generator).numpy()
            # Divide the exact epoch sample budget into batches without a singleton.
            chunks = np.array_split(draws, min(math.ceil(len(draws) / config['batch_size']), len(draws) // 2))
            loss_sum = 0.
            for selected in chunks:
                batch = next(bundle.batches('train', len(selected), positions=selected, device=device))
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch['x'], batch['nodes'], batch['contexts'])
                value = loss(prediction, batch, masks, reference, bundle.target_count, config['direction_lambda'])
                require(bool(torch.isfinite(value)), 'Training loss is nonfinite')
                value.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                        'A training gradient is nonfinite')
                nn.utils.clip_grad_value_(model.parameters(), config['clip_grad_value'])
                optimizer.step()
                loss_sum += float(value.detach().cpu())
            scores = validation(model, bundle, arrays, config['validation_batch_size'], device)
            value = scores[config['selection_metric']]
            require(np.isfinite(value), 'Checkpoint selection metric is nonfinite')
            row = dict(epoch=epoch, training_loss=loss_sum / len(chunks), learning_rate=optimizer.param_groups[0]['lr'],
                       validation=scores, samples_drawn=len(draws))
            history.append(row)
            if value < best:
                best, best_epoch = value, epoch
                save_checkpoint(directory / 'best.pt', model, model_config, bundle.contract, epoch, value, arrays['control_means'])
            save_checkpoint(directory / 'last.pt', model, model_config, bundle.contract, epoch, value, arrays['control_means'])
            write_json(directory / 'history.json', history)
            scheduler.step()
            print(json.dumps(dict(run_id=run_id, **row)), flush=True)
        restored, payload = load_checkpoint(directory / 'best.pt', device, bundle.genes, bundle.perturbations, bundle.contract)
        repeated = validation(restored, bundle, arrays, config['validation_batch_size'], device)
        require(np.isclose(repeated[config['selection_metric']], best, rtol=1e-6, atol=1e-8), 'Restored checkpoint validation differs')
        result = dict(run_id=run_id, variant=variant, protocol=bundle.protocol, split_seed=split_seed,
                      best_epoch=best_epoch, selection_metric=config['selection_metric'], selection_value=best,
                      selected_validation=repeated, training_seconds=time.monotonic() - started,
                      epochs_completed=len(history), test_evaluated=False, checkpoint_reload_verified=True,
                      best_checkpoint_sha256=digest(directory / 'best.pt'), preprocessing=cache,
                      depmap_metadata=model.depmap_metadata)
        write_json(directory / 'results.json', result)
        provenance['state'] = 'complete'
        write_json(directory / 'run.json', provenance)
        return result
    except BaseException as error:
        if directory.exists():
            write_json(directory / 'failure.json', dict(type=type(error).__name__, message=str(error), test_evaluated=False))
        raise
    finally:
        bundle.close()
