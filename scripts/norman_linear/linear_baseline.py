"""Fit the planned PCA/factorized linear pair baseline on frozen training rows."""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import gzip
import json
import platform
import sys
import tempfile

import h5py
import numpy as np

HELPERS = Path(__file__).resolve().parents[1] / "norman-additive"
sys.path.insert(0, str(HELPERS))
from additive_baseline import (
    ROOT, SOURCE, SPLITS, aggregate_selected_rows, check_frozen_bundle,
    digest, json_bytes, read_column, read_csv, require,
)

OUTPUT = ROOT / "outputs/norman-linear"
K = 10
PENALTY = 0.1
RTOL, ATOL = 1e-10, 1e-12
REFERENCE_COMMIT = "bfa6eeea2bd145a1af2ec0127a2e808cc38456a9"


def condition_embeddings(conditions, gene_names, gene_embeddings):
    """Use target-gene rows, summing two rows for canonical or reversed pairs."""
    names = list(map(str, gene_names))
    require(len(set(names)) == len(names), "Ambiguous expression gene names")
    require(gene_embeddings.ndim == 2 and len(gene_embeddings) == len(names),
            "Gene embedding axis mismatch")
    lookup = {name: i for i, name in enumerate(names)}
    columns = []
    for condition in conditions:
        targets = [] if condition == "ctrl" else str(condition).split("+")
        require(len(targets) <= 2 and len(set(targets)) == len(targets),
                f"Invalid condition: {condition}")
        require(all(t in lookup for t in targets), f"Target absent from expression panel: {condition}")
        columns.append(sum((gene_embeddings[lookup[t]] for t in targets),
                           np.zeros(gene_embeddings.shape[1], dtype=np.float64)))
    return np.stack(columns, axis=1)


def fit_model(mean_expression, training_conditions, gene_names, k=K, penalty=PENALTY):
    """PCA uses genes as rows and condition means as features, without scaling."""
    means = np.asarray(mean_expression, dtype=np.float64)
    require(means.shape == (len(training_conditions), len(gene_names)), "Training axes mismatch")
    require(len(set(training_conditions)) == len(training_conditions), "Duplicate training condition")
    require(np.isfinite(means).all(), "Non-finite training means")
    require(0 < k < min(means.shape) and penalty > 0, "Invalid PCA dimension or penalty")
    y = means.T
    feature_center = y.mean(axis=0)
    u, singular, vt = np.linalg.svd(y - feature_center, full_matrices=False)
    # Fix the sign ambiguity using the largest absolute gene loading per component.
    signs = np.sign(u[np.argmax(np.abs(u[:, :k]), axis=0), np.arange(k)])
    signs[signs == 0] = 1
    g = u[:, :k] * singular[:k] * signs
    components = vt[:k] * signs[:, None]
    b = condition_embeddings(training_conditions, gene_names, g)
    intercept = y.mean(axis=1)
    left = np.linalg.solve(g.T @ g + penalty * np.eye(k), g.T @ (y - intercept[:, None]))
    w = np.linalg.solve(b @ b.T + penalty * np.eye(k), (left @ b.T).T).T
    require(np.isfinite(w).all() and np.isfinite(g).all(), "Non-finite fitted model")
    total_variation = float(np.sum(singular ** 2))
    require(total_variation > 0, "Training profiles have no PCA variation")
    return {
        "gene_embeddings": g, "pca_feature_center": feature_center,
        "pca_components": components, "pca_singular_values": singular[:k],
        "pca_explained_variance_ratio": singular[:k] ** 2 / total_variation,
        "training_embeddings": b, "coefficients": w, "intercept": intercept,
    }


def predict(model, conditions, gene_names):
    embeddings = condition_embeddings(conditions, gene_names, model["gene_embeddings"])
    values = (model["intercept"][:, None] +
              model["gene_embeddings"] @ model["coefficients"] @ embeddings).T
    require(np.isfinite(values).all(), "Non-finite prediction")
    return values


def select_training_cells(assignments, manifest):
    expected = set(manifest["canonical_conditions"]["train"])
    held_out = set(manifest["canonical_conditions"]["val"] + manifest["canonical_conditions"]["test"])
    require(not expected & held_out, "Training/held-out conditions overlap")
    cells = sorted((r for r in assignments if r["partition"] == "train"),
                   key=lambda r: int(r["source_row"]))
    require({r["canonical_condition"] for r in cells} == expected, "Training condition set mismatch")
    require(len(cells) == manifest["target_cell_counts"]["train"], "Training cell count mismatch")
    require(len({r["cell_id"] for r in cells}) == len(cells), "Duplicate training cell ID")
    return sorted(expected), cells


def read_training_profiles(h, cells, conditions, gene_ids, gene_names):
    """Decode only allowlisted training CSR rows; never access rankings or layers."""
    require(all(r["partition"] == "train" for r in cells), "Only training rows may be fitted")
    require(np.array_equal(read_column(h["var"], "gene_id"), gene_ids), "Source gene IDs changed")
    require(np.array_equal(read_column(h["var"], "gene_name"), gene_names), "Source gene names changed")
    barcodes, labels = read_column(h["obs"], "cell_barcode"), read_column(h["obs"], "condition")
    require(all(barcodes[int(r["source_row"])] == r["cell_id"] and
                labels[int(r["source_row"])] == r["source_condition"] for r in cells),
            "Training source-row alignment changed")
    lookup = {c: i for i, c in enumerate(conditions)}
    rows = np.array([int(r["source_row"]) for r in cells])
    codes = np.array([lookup[r["canonical_condition"]] for r in cells])
    return aggregate_selected_rows(h["X"], rows, codes, len(conditions))


def prepare(source, split_dir, seed):
    """Check metadata and save a complete configuration before expression access."""
    frozen = check_frozen_bundle(split_dir)
    provenance = json.loads((split_dir / "provenance.json").read_text())
    source_hash = digest(source)
    require(source_hash == provenance["input_sha256"]["data/norman/perturb_processed.h5ad"],
            "Source H5AD differs from frozen source")
    manifest_path = split_dir / f"seed_{seed}/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assignments = read_csv(split_dir / f"seed_{seed}/cell_assignments.csv.gz")
    conditions, cells = select_training_cells(assignments, manifest)
    condition_rows = read_csv(split_dir / "conditions.csv")
    metadata = {r["canonical_condition"]: r for r in condition_rows}
    singles = sorted(c for c, r in metadata.items() if r["condition_type"] == "single")
    pairs = sorted(c for c, r in metadata.items() if r["condition_type"] == "pair")
    require(len(singles) == 102 and len(pairs) == 128 and len(conditions) == 179,
            "Unexpected frozen training population")
    require(manifest["pair_counts"] == {"train": 76, "val": 26, "test": 26}, "Split counts changed")
    require(set(singles + ["ctrl"]) <= set(conditions), "A single/control is outside training")
    genes = read_csv(split_dir / "expression_genes.csv")
    gene_ids, gene_names = (np.array([r[key] for r in genes]) for key in ("gene_id", "gene_name"))
    require(len(genes) == 5045 and len(set(gene_ids)) == 5045, "Expression gene panel changed")
    require([int(r["expression_index"]) for r in genes] == list(range(5045)), "Gene axis changed")
    require(len(set(gene_names)) == len(gene_names) and set(singles) <= set(gene_names),
            "Missing or ambiguous target-gene mapping")
    configuration = {
        "model": "fitted_linear_pair_adaptation", "split_seed": seed,
        "training_seed": 101, "stochastic_training_used": False,
        "pca_dimensions": K, "gene_penalty": PENALTY, "perturbation_penalty": PENALTY,
        "pca": "Deterministic NumPy reduced SVD; genes as observations; condition-column centering; no scaling/whitening",
        "pair_embedding": "Sum the two target-gene rows of G; control embedding is zero",
        "intercept": "Equal-condition mean of absolute training expression",
        "condition_weighting": "One mean column per canonical training condition",
        "expression_policy": "Preserve processed X, all 5045 genes, no clipping or extra normalization",
        "training_conditions": conditions, "training_cells": len(cells),
        "source_sha256": source_hash, "split_bundle_sha256": frozen["bundle_sha256"],
        "manifest_sha256": digest(manifest_path),
        "training_assignments_sha256": digest(split_dir / f"seed_{seed}/cell_assignments.csv.gz"),
        "runner_sha256": digest(Path(__file__)),
        "aggregation_helper_sha256": digest(HELPERS / "additive_baseline.py"),
        "reference_commit": REFERENCE_COMMIT,
        "reference": "https://doi.org/10.1038/s41592-025-02772-6",
        "validation_or_test_rows_permitted_for_fitting": False,
        "rankings_permitted_for_fitting": False,
        "prior_exposure": provenance["prior_exposure"] + [
            "Five additive validation panels expose 88 unique pair outcomes; no adaptive tuning of this fixed model"],
        "versions": {p: version(p) for p in ("numpy", "h5py")},
        "python": platform.python_version(), "machine": platform.machine(),
        "regeneration_rtol": RTOL, "regeneration_atol": ATOL,
    }
    return configuration, cells, conditions, pairs, gene_ids, gene_names, metadata, manifest


def build(source, context):
    config, cells, conditions, pairs, genes, names, metadata, manifest = context
    before = source.stat()
    with h5py.File(source, "r") as h:
        means, counts, entries = read_training_profiles(h, cells, conditions, genes, names)
    require(all(int(metadata[c]["cell_count"]) == int(counts[i]) for i, c in enumerate(conditions)),
            "Training mean cell counts mismatch")
    model = fit_model(means, conditions, names)
    values = predict(model, pairs, names)
    partitions = {c: p for p in ("train", "val", "test") for c in manifest["pair_conditions"][p]}
    targets = sorted(c for c, r in metadata.items() if r["condition_type"] == "single")
    model.update({"gene_ids": genes, "gene_names": names, "training_conditions": np.array(conditions),
                  "target_names": np.array(targets),
                  "target_expression_indices": np.array([list(names).index(c) for c in targets])})
    predictions = {"conditions": np.array(pairs), "gene_ids": genes, "gene_names": names,
                   "predicted_mean": values, "partitions": np.array([partitions[c] for c in pairs])}
    profiles = {"conditions": np.array(conditions), "mean_expression": means,
                "cell_counts": counts, "gene_ids": genes, "gene_names": names}
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Source changed during fitting")
    return model, predictions, profiles, entries


def verify_artifacts(directory):
    summary = json.loads((directory / "summary.json").read_text())
    for name, expected in summary["artifact_sha256"].items():
        require(digest(directory / name) == expected, f"Linear artifact changed: {name}")
    return summary


def verify(directory, payload):
    summary = verify_artifacts(directory)
    for filename, expected in zip(("model.npz", "predictions.npz", "training_means.npz"), payload[:3]):
        with np.load(directory / filename, allow_pickle=False) as saved:
            require(set(saved.files) == set(expected), f"Array schema changed: {filename}")
            for key, values in expected.items():
                actual = saved[key]
                equal = (actual.shape == values.shape and np.allclose(actual, values, rtol=RTOL, atol=ATOL)) if values.dtype.kind in "fc" else np.array_equal(actual, values)
                require(equal, f"Regenerated array differs: {filename}/{key}")
    require(summary["training_expression_entries_decoded"] == payload[3], "Training access count changed")
    return {"status": "passed", "prediction_and_fit_regeneration": True,
            "rtol": RTOL, "atol": ATOL, "held_out_rows_read_for_fitting": 0,
            "rankings_read_for_fitting": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "verify"))
    parser.add_argument("--split-seed", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source, splits = args.source.resolve(), args.split_dir.resolve()
    output = (args.output_dir or OUTPUT / f"seed_{args.split_seed}").resolve()
    require(output.is_relative_to(ROOT / "outputs"), "Keep linear outputs inside Codex Outputs/outputs")
    require(not output.is_relative_to(splits) and not splits.is_relative_to(output), "Separate outputs from frozen splits")
    require(not output.exists() if args.command == "fit" else output.is_dir(),
            "Fit requires a new output directory; verify requires an existing bundle")
    context = prepare(source, splits, args.split_seed)
    if args.command == "verify":
        require(json.loads((output / "fit_config.json").read_text()) == context[0], "Fit configuration changed")
        require(read_csv(output / "fit_cells.csv.gz") == context[1], "Training allowlist changed")
        result = verify(output, build(source, context))
        (output / "verification_report.json").write_bytes(json_bytes(result))
        print(json.dumps({"split_seed": args.split_seed, **result}, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="linear-fit-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        (stage / "fit_config.json").write_bytes(json_bytes(context[0]))
        model, predictions, profiles, entries = build(source, context)
        for filename, arrays in (("model.npz", model), ("predictions.npz", predictions), ("training_means.npz", profiles)):
            np.savez_compressed(stage / filename, **arrays)
        with gzip.open(stage / "fit_cells.csv.gz", "wt", newline="") as stream:
            import csv
            writer = csv.DictWriter(stream, fieldnames=list(context[1][0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(context[1])
        summary = {"status": "fitted_predictions_not_scored", "model": context[0]["model"],
                   "split_seed": args.split_seed, "source_sha256": context[0]["source_sha256"],
                   "split_bundle_sha256": context[0]["split_bundle_sha256"],
                   "training_cells": len(context[1]), "training_conditions": len(context[2]),
                   "training_pairs": 76, "training_singles": 102, "expression_genes": len(context[4]),
                   "prediction_shape": list(predictions["predicted_mean"].shape),
                   "training_expression_entries_decoded": entries,
                   "held_out_expression_rows_decoded_for_fitting": 0, "rankings_decoded_for_fitting": 0,
                   "negative_prediction_entries_preserved": int(np.sum(predictions["predicted_mean"] < 0)),
                   "pca_explained_variance_ratio_sum": float(model["pca_explained_variance_ratio"].sum()),
                   "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                   "artifact_sha256": {p.name: digest(p) for p in stage.iterdir()}}
        (stage / "summary.json").write_bytes(json_bytes(summary))
        stage.rename(output)
    print(json.dumps({"split_seed": args.split_seed, "output": str(output),
                      "training_cells": summary["training_cells"], "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
