"""Verify or rebuild the frozen Norman condition splits from shared metadata.

Only NumPy and the standard library are needed. Expression values are not read.
The checked-in snapshot contains metadata, source hashes, and GEARS custom split
pickles; runtime output directories must be outside the source checkout.
"""

from collections import Counter
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import io
import json
import pickle
import shutil
import tempfile

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = REPOSITORY / "splits/norman"
PARTITIONS = ("train", "val", "test")
SEEDS = (1, 2, 3, 4, 5)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with path.open("wb") as raw:
        if path.suffix == ".gz":
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    writer = csv.writer(text, lineterminator="\n")
                    writer.writerow(fields)
                    writer.writerows(rows)
        else:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(fields)
                writer.writerows(rows)


def canonical(label):
    tokens = label.split("+")
    require(all(tokens) and len(tokens) <= 2, f"Malformed condition: {label}")
    genes = [gene for gene in tokens if gene != "ctrl"]
    require(len(genes) == len(set(genes)), f"Repeated target: {label}")
    require(label == "ctrl" or bool(genes), f"Malformed control: {label}")
    return "+".join(sorted(genes)) or "ctrl"


def condition_type(condition):
    return "control" if condition == "ctrl" else "pair" if "+" in condition else "single"


def pair_partitions(pairs, seed):
    ordered = sorted(pairs)
    require(len(ordered) == len(set(ordered)) == 128, "Expected 128 unique pair conditions")
    require(all(condition_type(canonical(p)) == "pair" for p in ordered), "Expected two-gene conditions")
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(ordered).tolist()
    return {"train": sorted(shuffled[52:]), "val": sorted(shuffled[26:52]),
            "test": sorted(shuffled[:26])}


def check_hashes(directory):
    frozen = json.loads((directory / "freeze.json").read_text())
    expected = hashlib.sha256(json_bytes(frozen["artifact_sha256"])).hexdigest()
    require(expected == frozen["bundle_sha256"], "Freeze index digest mismatch")
    for name, expected in frozen["artifact_sha256"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "Artifact path escapes snapshot")
        require(digest(path) == expected, f"Artifact hash mismatch: {name}")
    return frozen


def load_metadata(directory):
    cells = read_csv(directory / "cells.csv.gz")
    counts = Counter(row["canonical_condition"] for row in cells)
    require(len(cells) == len({r["cell_id"] for r in cells}) == 89357, "Retained cell population differs")
    require([int(r["retained_row"]) for r in cells] == list(range(len(cells))), "Retained cell order differs")
    source_rows = [int(r["source_row"]) for r in cells]
    require(source_rows == sorted(set(source_rows)), "Source rows are duplicated or reordered")
    require(all(canonical(r["source_condition"]) == r["canonical_condition"] for r in cells), "Source-label aliases disagree")
    singles = {c for c in counts if condition_type(c) == "single"}
    pairs = {c for c in counts if condition_type(c) == "pair"}
    require(len(singles) == 102 and len(pairs) == 128 and counts["ctrl"] == 7353,
            "Condition/control counts differ from the Norman plan")
    require(all(set(p.split("+")) <= singles for p in pairs), "A pair lacks its single-gene conditions")
    aliases = {c: set() for c in counts}
    for row in cells:
        aliases[row["canonical_condition"]].add(row["source_condition"])
    return cells, counts, {c: sorted(labels) for c, labels in aliases.items()}


def rebuild_files(output, cells, counts, aliases):
    """Recompute condition/cell assignments and fixed input schedules, not outcomes."""
    pairs = [c for c in counts if condition_type(c) == "pair"]
    shared_train = [c for c in counts if condition_type(c) != "pair"]
    cell_fields = tuple(cells[0])
    condition_fields = ("canonical_condition", "condition_type", "target_a", "target_b", "cell_count", "source_labels_json")
    control_indices = np.array([i for i, row in enumerate(cells) if row["canonical_condition"] == "ctrl"])
    noncontrol = np.array([i for i, row in enumerate(cells) if row["canonical_condition"] != "ctrl"])
    selected = np.arange(len(cells))
    selected[noncontrol] = np.random.Generator(np.random.PCG64(1000)).choice(control_indices, len(noncontrol), replace=True)
    control_fields = ("target_source_row", "target_cell_id", "control_source_row", "control_cell_id", "input_index")
    controls = [(row["source_row"], row["cell_id"], cells[j]["source_row"], cells[j]["cell_id"], 0)
                for row, j in zip(cells, selected)]
    write_csv(output / "control_assignments.csv.gz", control_fields, controls)
    write_csv(output / "evaluation_control_schedule.csv.gz", control_fields,
              (row for row, cell in zip(controls, cells) if condition_type(cell["canonical_condition"]) == "pair"))
    summaries = []
    for seed in SEEDS:
        directory = output / f"seed_{seed}"
        directory.mkdir(exist_ok=True)
        pairs_by_partition = pair_partitions(pairs, seed)
        conditions = {k: sorted(pairs_by_partition[k] + (shared_train if k == "train" else [])) for k in PARTITIONS}
        # Preserve NumPy string scalars used by the original frozen pickle.
        source_conditions = {k: sorted(np.str_(label) for c in conditions[k] for label in aliases[c]) for k in PARTITIONS}
        (directory / "gears_custom_split.pkl").write_bytes(pickle.dumps(source_conditions, protocol=4))
        partition_for = {c: k for k in PARTITIONS for c in conditions[k]}
        write_csv(directory / "cell_assignments.csv.gz", cell_fields + ("partition",),
                  (tuple(row.values()) + (partition_for[row["canonical_condition"]],) for row in cells))
        write_csv(directory / "condition_assignments.csv", condition_fields + ("partition",),
                  ((c, condition_type(c), c.split("+")[0] if c != "ctrl" else "",
                    c.split("+")[1] if "+" in c else "", counts[c], json.dumps(aliases[c]), partition_for[c])
                   for c in sorted(counts)))
        summaries.append({"split_seed": seed, "pair_counts": {k: len(v) for k, v in pairs_by_partition.items()},
                          "target_cell_counts": dict(Counter(partition_for[r["canonical_condition"]] for r in cells))})
    return summaries


def verify(directory=DEFAULT_SPLITS, temp_dir=None):
    directory = Path(directory).resolve()
    frozen = check_hashes(directory)
    cells, counts, aliases = load_metadata(directory)
    with tempfile.TemporaryDirectory(prefix="norman-splits-", dir=temp_dir) as temporary:
        regenerated = Path(temporary)
        summaries = rebuild_files(regenerated, cells, counts, aliases)
        for path in regenerated.rglob("*"):
            if path.is_file():
                name = path.relative_to(regenerated).as_posix()
                require(digest(path) == frozen["artifact_sha256"][name], f"Rebuilt split/control artifact differs: {name}")
    for summary in summaries:
        manifest = json.loads((directory / f"seed_{summary['split_seed']}/manifest.json").read_text())
        require(manifest["target_cell_counts"] == summary["target_cell_counts"], "Manifest cell counts differ")
        require(manifest["pair_conditions"] == pair_partitions([c for c in counts if condition_type(c) == "pair"], summary["split_seed"]), "Manifest pair lists differ")
        require(all(c in manifest["canonical_conditions"]["train"] for c in counts if condition_type(c) != "pair"), "Singles/controls absent from training")
    return {"status": "passed", "bundle_sha256": frozen["bundle_sha256"],
            "hashed_artifacts": len(frozen["artifact_sha256"]), "retained_cells": len(cells),
            "source_condition_labels": sum(map(len, aliases.values())), "splits": summaries,
            "recomputed_split_and_control_files_identical": True}


def generate(directory, output, temp_dir=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.is_relative_to(REPOSITORY), "Runtime outputs must be outside the source checkout")
    require(not output.exists(), "Output already exists; use verify to audit frozen files")
    report = verify(directory, temp_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells, counts, aliases = load_metadata(directory)
    # Publish only after the regenerated files pass the original frozen hashes.
    with tempfile.TemporaryDirectory(prefix="norman-splits-build-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        shutil.copytree(directory, stage)
        rebuild_files(stage, cells, counts, aliases)
        check_hashes(stage)
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "generate"))
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--temp-dir", type=Path, help="Optional existing directory for verification scratch files")
    parser.add_argument("--output-dir", type=Path, help="Required for generate; must be outside the checkout")
    args = parser.parse_args()
    if args.command == "generate":
        if args.output_dir is None:
            parser.error("generate requires --output-dir")
        report = generate(args.split_dir, args.output_dir, args.temp_dir)
    else:
        report = verify(args.split_dir, args.temp_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
