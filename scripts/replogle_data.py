"""Prepare, verify, or materialize paired Replogle single-cell dataset views."""

import argparse
import json
from pathlib import Path

from replogle_data.protocol import PARTITIONS, PROTOCOLS
from replogle_data.workflow import REPOSITORY, export_view, prepare, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('prepare')
    build.add_argument('--k562', type=Path, required=True)
    build.add_argument('--rpe1', type=Path, required=True)
    build.add_argument('--output-dir', type=Path, required=True)
    build.add_argument('--config', type=Path, default=REPOSITORY / 'configs/replogle_data.json')
    build.add_argument('--go-vocabulary', type=Path, default=REPOSITORY / 'splits/norman/perturbation_nodes.csv')
    build.add_argument('--depmap-genes', type=Path)
    check = commands.add_parser('verify')
    check.add_argument('--data-dir', type=Path, required=True)
    check.add_argument('--compare-sources', action='store_true')
    export = commands.add_parser('export')
    export.add_argument('--data-dir', type=Path, required=True)
    export.add_argument('--protocol', choices=PROTOCOLS, required=True)
    export.add_argument('--seed', type=int, required=True)
    export.add_argument('--partition', choices=PARTITIONS, required=True)
    export.add_argument('--cell-line', choices=['k562', 'rpe1'])
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.k562, args.rpe1, args.output_dir, args.config, args.go_vocabulary, args.depmap_genes)
    elif args.command == 'verify':
        result = verify(args.data_dir, args.compare_sources)
    else:
        result = export_view(args.data_dir, args.protocol, args.seed, args.partition, args.output, args.cell_line)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
