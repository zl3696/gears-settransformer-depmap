"""Launch main or either transfer task independently on frozen single-cell views."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
TASKS = ('main', 'joint_unseen', 'k562_to_rpe1', 'rpe1_to_k562')
DEFAULT = REPOSITORY / 'configs/replogle_training.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'plan'):
        command = commands.add_parser(name)
        command.add_argument('--data-dir', type=Path, required=True)
        command.add_argument('--output-root', type=Path, required=True)
        command.add_argument('--gene2go', type=Path, required=True, help='Trusted hash-pinned gene2go_all.pkl')
        command.add_argument('--depmap-features', type=Path, help='Prepared fixed NPZ prior, required for C')
        command.add_argument('--config', type=Path, default=DEFAULT)
        command.add_argument('--device', default='auto')
        command.add_argument('--preload', action='store_true', help='Cache permitted train/validation expression as sparse CSR in RAM')
        if name == 'train':
            command.add_argument('--task', choices=TASKS, required=True)
            command.add_argument('--variant', choices=['A', 'B', 'C'], required=True)
            command.add_argument('--split-seed', type=int, required=True)
        else:
            command.add_argument('--tasks', nargs='+', choices=TASKS, default=['main', 'k562_to_rpe1', 'rpe1_to_k562'])
            command.add_argument('--variants', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
            command.add_argument('--split-seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
            command.add_argument('--execute', action='store_true', help='Execute printed commands sequentially; otherwise plan only')
    args = parser.parse_args()
    need_prior = args.variant == 'C' if args.command == 'train' else 'C' in args.variants
    if need_prior and args.depmap_features is None:
        parser.error('Variant C requires --depmap-features')
    for name in ('data_dir', 'output_root', 'gene2go', 'config', 'depmap_features'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.output_root.is_relative_to(REPOSITORY) or args.output_root.is_relative_to(args.data_dir):
        parser.error('Output root must be outside the source checkout and frozen data')
    if args.command == 'plan':
        frozen = json.loads((args.data_dir / 'config.json').read_text())
        if not set(args.split_seeds) <= set(frozen['split_seeds']):
            parser.error('Split seed is not in the frozen data')
        commands_seen = set()
        for task in args.tasks:
            protocol = 'joint_unseen' if task == 'main' else task
            for seed in args.split_seeds:
                view = args.data_dir / 'views' / protocol / f'seed_{seed}'
                if not all((view / file).is_file() for file in ('manifest.json', 'train.npz', 'val.npz')):
                    parser.error(f'Training/validation view is missing: {view}')
                for variant in args.variants:
                    identity = (protocol, seed, variant)
                    if identity in commands_seen:
                        continue
                    commands_seen.add(identity)
                    command = [sys.executable, str(Path(__file__).resolve()), 'train', '--task', task,
                               '--variant', variant, '--split-seed', str(seed), '--data-dir', str(args.data_dir),
                               '--output-root', str(args.output_root), '--gene2go', str(args.gene2go),
                               '--config', str(args.config), '--device', args.device]
                    if variant == 'C':
                        command += ['--depmap-features', str(args.depmap_features)]
                    if args.preload:
                        command.append('--preload')
                    print(shlex.join(command), flush=True)
                    if args.execute:
                        subprocess.run(command, check=True)
        return
    # Configure native-library thread counts before importing the training stack.
    config = json.loads(args.config.read_text())
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(config['threads'])
    from replogle_training.engine import run
    result = run(args.data_dir, args.output_root, args.task, args.variant, args.split_seed,
                 args.config, args.gene2go, args.depmap_features, args.device, args.preload)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
