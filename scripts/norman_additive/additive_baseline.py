"""Predict Norman pairs from training single/control means and frozen splits.

This runner never reads pair expression values, released rankings, or counts.
Use the shared GEARS evaluator later to score these condition-level predictions.
"""

from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import csv
import gzip
import hashlib
import json
import platform
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/norman/perturb_processed.h5ad"
SPLITS = ROOT / "outputs/norman-splits"
OUTPUT = ROOT / "outputs/norman-additive"
PARTITIONS = ("train", "val", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_column(group, key):
    obj = group[key]
    if isinstance(obj, h5py.Group):
        categories, codes = obj["categories"][:].astype(str), obj["codes"][:]
    elif "__categories" in group and key in group["__categories"]:
        categories, codes = group["__categories"][key][:].astype(str), obj[:]
    else:
        return obj[:].astype(str)
    require(np.all(codes >= 0), f"Missing metadata: {key}")
    return categories[codes]


def check_frozen_bundle(directory):
    frozen = json.loads((directory / "freeze.json").read_text())
    hashes = frozen["artifact_sha256"]
    require(hashlib.sha256(json_bytes(hashes)).hexdigest() == frozen["bundle_sha256"],
            "Split bundle digest mismatch")
    for relative, expected in hashes.items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory), "Split artifact escapes its directory")
        require(digest(path) == expected, f"Frozen split artifact changed: {relative}")
    return frozen


def check_manifests(manifests, conditions, cells):
    """Require the identical complete single/control population in every split."""
    names = set(conditions)
    singles_controls = {c for c, row in conditions.items() if row["condition_type"] != "pair"}
    pairs = names - singles_controls
    for manifest in manifests:
        sets = {p: set(manifest["canonical_conditions"][p]) for p in PARTITIONS}
        require(set.union(*sets.values()) == names, "Split population mismatch")
        require(sum(map(len, sets.values())) == len(names), "Split partitions overlap")
        require(singles_controls <= sets["train"], "A single/control is outside training")
        for pair in pairs:
            a, b = pair.split("+")
            require(a in sets["train"] and b in sets["train"],
                    f"Both singles must be in training: {pair}")
        for partition in PARTITIONS:
            require(set(manifest["pair_conditions"][partition]) == sets[partition] & pairs,
                    "Pair lists disagree with canonical partitions")
            expected_cells = sum(int(conditions[c]["cell_count"]) for c in sets[partition])
            require(expected_cells == manifest["target_cell_counts"][partition],
                    "Manifest cell count mismatch")
    counts = Counter(row["canonical_condition"] for row in cells)
    require(counts == Counter({c: int(row["cell_count"]) for c, row in conditions.items()}),
            "Frozen cell and condition tables disagree")


def aggregate_selected_rows(x, source_rows, condition_codes, n_conditions):
    """Sum only allowlisted CSR rows, including zeros, in float64.

    Read contiguous runs of selected rows. Never read a data/indices slice
    spanning an excluded row, even when its condition is training in a split.
    """
    shape = tuple(map(int, x.attrs["shape"]))
    require(x.attrs.get("encoding-type", "csr_matrix") == "csr_matrix", "Expected CSR X")
    rows = np.asarray(source_rows, dtype=np.int64)
    codes = np.asarray(condition_codes, dtype=np.int64)
    require(len(rows) > 0 and len(rows) == len(codes), "Empty or inconsistent row selection")
    require(np.all(np.diff(rows) > 0), "Source rows must be unique and increasing")
    require(rows[0] >= 0 and rows[-1] < shape[0], "Source row outside X")
    require(np.all((codes >= 0) & (codes < n_conditions)), "Invalid condition code")
    indptr = x["indptr"][:]
    require(len(indptr) == shape[0] + 1 and np.all(np.diff(indptr) >= 0), "Invalid CSR pointers")
    sums = np.zeros((n_conditions, shape[1]), dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(rows) > 1) + 1, len(rows)]
    decoded_entries = 0
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        start, stop = int(rows[first]), int(rows[last - 1]) + 1
        left, right = int(indptr[start]), int(indptr[stop])
        values = x["data"][left:right]
        columns = x["indices"][left:right]
        require(np.isfinite(values).all(), "Non-finite training expression")
        require(np.all((columns >= 0) & (columns < shape[1])), "Invalid CSR gene index")
        labels = np.repeat(codes[first:last], np.diff(indptr[start:stop + 1]))
        np.add.at(sums, (labels, columns), values)
        decoded_entries += len(values)
    counts = np.bincount(codes, minlength=n_conditions)
    require(np.all(counts > 0), "A required training condition has no cells")
    return sums / counts[:, None], counts, decoded_entries


def predict_pairs(means, pairs):
    control = means["ctrl"]
    predictions = []
    for pair in pairs:
        a, b = pair.split("+")
        require(a != b and a in means and b in means, f"Unavailable/invalid singles: {pair}")
        # Keep negative predictions: clipping would change the cited baseline.
        predictions.append(means[a] + means[b] - control)
    return np.stack(predictions)


def build(source, split_dir):
    frozen = check_frozen_bundle(split_dir)
    provenance = json.loads((split_dir / "provenance.json").read_text())
    config = json.loads((split_dir / "config.json").read_text())
    before = source.stat()
    source_hash = digest(source)
    require(source_hash == provenance["input_sha256"]["data/norman/perturb_processed.h5ad"],
            "Source H5AD differs from the frozen split source")
    cells = read_csv(split_dir / "cells.csv.gz")
    condition_rows = read_csv(split_dir / "conditions.csv")
    conditions = {row["canonical_condition"]: row for row in condition_rows}
    require(len(conditions) == len(condition_rows), "Duplicate canonical condition")
    genes = read_csv(split_dir / "expression_genes.csv")
    manifests = [json.loads((split_dir / f"seed_{seed}/manifest.json").read_text())
                 for seed in config["split_seeds"]]
    check_manifests(manifests, conditions, cells)
    fit_cells = [row for row in cells if row["condition_type"] in ("single", "control")]
    fit_names = sorted({row["canonical_condition"] for row in fit_cells})
    lookup = {name: i for i, name in enumerate(fit_names)}
    source_rows = np.array([int(row["source_row"]) for row in fit_cells])
    codes = np.array([lookup[row["canonical_condition"]] for row in fit_cells])
    gene_ids = np.array([row["gene_id"] for row in genes])
    gene_names = np.array([row["gene_name"] for row in genes])
    require([int(row["expression_index"]) for row in genes] == list(range(len(genes))),
            "Expression gene order mismatch")
    with h5py.File(source, "r") as h:
        require(list(map(int, h["X"].attrs["shape"])) == provenance["source_shape"],
                "Source shape mismatch")
        require(np.array_equal(read_column(h["var"], "gene_id"), gene_ids), "Gene IDs reordered")
        require(np.array_equal(read_column(h["var"], "gene_name"), gene_names), "Gene names reordered")
        barcodes, labels = read_column(h["obs"], "cell_barcode"), read_column(h["obs"], "condition")
        require(all(barcodes[int(r["source_row"])] == r["cell_id"] and
                    labels[int(r["source_row"])] == r["source_condition"] for r in cells),
                "Frozen source row/cell/condition alignment mismatch")
        mean_array, counts, entries = aggregate_selected_rows(h["X"], source_rows, codes, len(fit_names))
    means = dict(zip(fit_names, mean_array))
    require(all(int(conditions[c]["cell_count"]) == int(counts[i]) for i, c in enumerate(fit_names)),
            "Training mean cell counts mismatch")
    pairs = sorted(c for c, row in conditions.items() if row["condition_type"] == "pair")
    predictions = predict_pairs(means, pairs)
    require(np.isfinite(predictions).all(), "Non-finite pair prediction")
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Source file changed during prediction")
    memberships = []
    for manifest in manifests:
        partition_for = {c: p for p in PARTITIONS for c in manifest["pair_conditions"][p]}
        memberships.extend({"split_seed": manifest["split_seed"], "prediction_row": i,
                            "canonical_condition": c, "partition": partition_for[c],
                            "target_cell_count": int(conditions[c]["cell_count"])}
                           for i, c in enumerate(pairs))
    arrays = {"conditions": np.array(pairs), "gene_ids": gene_ids, "gene_names": gene_names,
              "additive_mean": predictions, "control_mean": means["ctrl"]}
    fit_arrays = {"conditions": np.array(fit_names), "mean_expression": mean_array,
                  "cell_counts": counts, "gene_ids": gene_ids, "gene_names": gene_names}
    summary = {
        "status": "predictions_verified_not_scored", "formula": "mu_A + mu_B - mu_control",
        "source_sha256": source_hash, "split_bundle_sha256": frozen["bundle_sha256"],
        "source": str(source), "split_directory": str(split_dir),
        "expression_policy": provenance["expression_policy"],
        "training_conditions": len(fit_names), "training_single_conditions": len(fit_names) - 1,
        "training_cells": len(fit_cells), "control_cells": int(counts[lookup["ctrl"]]),
        "prediction_shape": list(predictions.shape), "split_seeds": config["split_seeds"],
        "pair_counts_per_split": {str(m["split_seed"]): m["pair_counts"] for m in manifests},
        "training_expression_entries_decoded": entries,
        "pair_expression_rows_decoded": 0, "released_rankings_decoded": False,
        "negative_prediction_entries_preserved": int(np.sum(predictions < 0)),
        "prediction_reference": "Full training-control mean; one mean profile per pair",
        "control_schedule": "Not sampled: condition-mean baseline does not use per-cell control inputs",
        "shared_across_splits": "All five splits have identical fit singles/controls; predictions reused",
        "scoring_status": "No validation/test scores; common evaluator and gene-ranking rules pending",
        "prior_exposure": provenance["prior_exposure"],
        "reference": "https://doi.org/10.1038/s41592-025-02772-6",
        "reference_code": "https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/main/benchmark/src/run_additive_model.py",
    }
    return arrays, fit_arrays, memberships, fit_cells, summary


def write_outputs(output, payload):
    arrays, fit_arrays, memberships, fit_cells, summary = payload
    require(not output.exists(), "Output exists; use verify or a new versioned output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="norman-additive-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        np.savez_compressed(stage / "predictions.npz", **arrays)
        np.savez_compressed(stage / "training_means.npz", **fit_arrays)
        with (stage / "split_membership.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(memberships[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(memberships)
        # The allowlist records every row whose expression contributed to fitting.
        with gzip.open(stage / "fit_cells.csv.gz", "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fit_cells[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(fit_cells)
        summary.update({"created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                        "python": platform.python_version(),
                        "versions": {p: version(p) for p in ("numpy", "h5py")},
                        "runner_sha256": digest(Path(__file__)),
                        "artifact_sha256": {p.name: digest(p) for p in stage.iterdir()}})
        (stage / "summary.json").write_bytes(json_bytes(summary))
        stage.rename(output)
    return summary


def verify_outputs(output, payload):
    arrays, fit_arrays, memberships, fit_cells, current = payload
    saved = json.loads((output / "summary.json").read_text())
    for name, expected in saved["artifact_sha256"].items():
        require(digest(output / name) == expected, f"Output artifact changed: {name}")
    require(saved["runner_sha256"] == digest(Path(__file__)), "Runner changed; version/rebuild outputs")
    for key, value in current.items():
        require(saved[key] == value, f"Prediction provenance changed: {key}")
    for name, expected in (("predictions.npz", arrays), ("training_means.npz", fit_arrays)):
        with np.load(output / name, allow_pickle=False) as actual:
            require(set(actual.files) == set(expected), f"Array schema changed: {name}")
            for key, values in expected.items():
                require(np.array_equal(actual[key], values), f"Regenerated array differs: {name}/{key}")
    require(read_csv(output / "split_membership.csv") ==
            [{k: str(v) for k, v in row.items()} for row in memberships], "Split memberships changed")
    require(read_csv(output / "fit_cells.csv.gz") == fit_cells, "Training row allowlist changed")
    return {"status": "passed", "identical_prediction_regeneration": True,
            "prediction_shape": current["prediction_shape"], "split_seeds": current["split_seeds"],
            "pair_expression_rows_decoded": 0, "validation_or_test_scores_computed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "verify"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source, split_dir, output = (p.resolve() for p in (args.source, args.split_dir, args.output_dir))
    require(output.is_relative_to(ROOT / "outputs"), "Keep runtime outputs inside Codex Outputs/outputs")
    require(not output.is_relative_to(split_dir) and not split_dir.is_relative_to(output),
            "Prediction output and frozen split directories must be separate")
    if args.command == "predict":
        require(not output.exists(), "Output exists; use verify or a new versioned output directory")
    payload = build(source, split_dir)
    result = write_outputs(output, payload) if args.command == "predict" else verify_outputs(output, payload)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
