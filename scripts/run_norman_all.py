"""Run the configured Norman experiments, skipping matching completed runs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'configs/norman_training.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models', nargs='+', help='Model names from the training config; default: all')
    parser.add_argument('--splits', nargs='+', type=int, choices=range(1, 6), help='Default: all five')
    parser.add_argument('--test', action='store_true', help='Score the selected checkpoint after training')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training or writing files')
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.models and set(args.models) - protocol['models'].keys():
        parser.error('Unknown model; choose from ' + ', '.join(protocol['models']))
    output_root = args.output_root.resolve()
    data_path = args.data_path.resolve()
    if output_root.is_relative_to(REPO) or data_path.is_relative_to(REPO):
        parser.error('Data and outputs must be outside the source checkout')
    with (REPO / protocol['run_plan']).open(newline='') as stream:
        runs = [row for row in csv.DictReader(stream)
                if (not args.models or row['model'] in args.models)
                and (not args.splits or int(row['split_seed']) in args.splits)]
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    env_dir = Path(sys.executable).parent
    env = dict(os.environ)
    # Retain Windows conda support without machine-specific paths.
    env['PATH'] = os.pathsep.join([str(env_dir), str(env_dir / 'Library' / 'bin'),
                                   str(env_dir / 'Scripts'), env.get('PATH', '')])
    for run in runs:
        command = [sys.executable, str(REPO / 'scripts/norman_train.py'), run['run_id'],
                   '--data-path', str(data_path), '--output-root', str(output_root),
                   '--device', args.device]
        if args.test:
            command.append('--test')
        if args.dry_run:
            print(shlex.join(command))
            continue
        result_path = output_root / run['result_path']
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result.get('protocol_sha256') != protocol_hash
                    or result.get('run_id') != run['run_id']
                    or not all((result_path.parent / name).is_file() for name in ('config.pkl', 'model.pt'))
                    or (args.test and not result.get('test_evaluated'))):
                raise ValueError(f'Existing run is incomplete or uses another protocol: {result_path}')
            print(f"Skip completed {run['run_id']}", flush=True)
            continue
        print(f"Start {run['run_id']}", flush=True)
        subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == '__main__':
    main()
