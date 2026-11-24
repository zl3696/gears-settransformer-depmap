#!/usr/bin/env python3
"""Evaluate frozen Replogle models and the training-only Copy-effect baseline."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run', 'verify', 'regenerate', 'summary'):
        command = commands.add_parser(name)
        command.add_argument('--output-root', type=Path, required=True)
        command.add_argument('--tasks', nargs='+', choices=('main', 'joint_unseen', 'k562_to_rpe1', 'rpe1_to_k562'),
                             default=['main', 'k562_to_rpe1', 'rpe1_to_k562'])
        command.add_argument('--variants', nargs='+', choices=('A', 'B', 'C', 'Copy-effect'))
        command.add_argument('--split-seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
        command.add_argument('--partition', choices=('val', 'test'), default='val')
        if name in ('plan', 'run', 'regenerate'):
            command.add_argument('--data-dir', type=Path, required=True)
            command.add_argument('--training-root', type=Path)
            command.add_argument('--config', type=Path, default=REPOSITORY / 'configs/replogle_evaluation.json')
    args = parser.parse_args()
    for key in ('data_dir', 'training_root', 'output_root', 'config'):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, value.resolve())
    # Native libraries must receive thread/cache settings before their imports.
    protocol_path = getattr(args, 'config', REPOSITORY / 'configs/replogle_evaluation.json')
    protocol = json.loads(protocol_path.read_text())
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(protocol['threads'])
    with tempfile.TemporaryDirectory(prefix='replogle-evaluation-cache-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        from replogle_evaluation.artifacts import load_protocol, verify_result
        from replogle_evaluation.prediction import run_directory
        from replogle_evaluation.summary import summarize
        from replogle_evaluation.workflow import evaluate, output_directory, require_output_root
        from replogle_training.dataset import TASKS
        protocol = load_protocol(protocol_path)
        tasks = list(dict.fromkeys('main' if TASKS[t] == 'joint_unseen' else t for t in args.tasks))
        if len(set(args.split_seeds)) != len(args.split_seeds) or not set(args.split_seeds) <= set(protocol['split_seeds']):
            parser.error('Select unique split seeds from 1–5')
        if args.variants is not None and len(set(args.variants)) != len(args.variants):
            parser.error('Select unique variants')
        selections = []
        for task in tasks:
            variants = args.variants or (['A', 'B', 'C'] if task == 'main' else ['A', 'B', 'C', 'Copy-effect'])
            if task == 'main' and 'Copy-effect' in variants:
                parser.error('Copy-effect is inapplicable to main; select transfer tasks')
            selections.extend((task, seed, variant) for seed in args.split_seeds for variant in variants)
        if args.command in ('plan', 'run', 'regenerate'):
            require_output_root(args.output_root, args.data_dir, args.training_root)
            frozen = json.loads((args.data_dir / 'config.json').read_text())
            if not set(args.split_seeds) <= set(frozen['split_seeds']):
                parser.error('Split seed is missing from the frozen data bundle')
            inventory = []
            for task, seed, variant in selections:
                checkpoint = None if variant == 'Copy-effect' or args.training_root is None else run_directory(
                    args.training_root, TASKS[task], variant, seed, protocol['training_seed'])
                needed = ('run.json', 'history.json', 'results.json', 'best.pt', 'best.pt.sha256')
                ready = variant == 'Copy-effect' or (checkpoint is not None and all((checkpoint / n).is_file() for n in needed))
                validation = output_directory(args.output_root, task, seed, variant, 'val')
                inventory.append(dict(task=task, split_seed=seed, variant=variant, checkpoint_ready=bool(ready),
                                      validation_ready=(validation / 'manifest.json').is_file()))
            if args.command == 'plan':
                print(json.dumps(dict(partition=args.partition, evaluations=inventory), indent=2))
                return
            missing = [row for row in inventory if not row['checkpoint_ready'] or
                       (args.partition == 'test' and not row['validation_ready'])]
            if missing:
                parser.error('Missing complete checkpoints or required validation evaluations: ' + json.dumps(missing))
        if args.command == 'summary':
            for task in tasks:
                variants = args.variants or (['A', 'B', 'C'] if task == 'main' else ['A', 'B', 'C', 'Copy-effect'])
                print(json.dumps(summarize(args.output_root, task, args.split_seeds, variants, args.partition)))
        else:
            for task, seed, variant in selections:
                if args.command == 'verify':
                    result = verify_result(output_directory(args.output_root, task, seed, variant, args.partition))
                else:
                    result = evaluate(args.data_dir, args.training_root, args.output_root, task, seed, variant,
                                      args.partition, protocol, regenerate=args.command == 'regenerate')
                print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
