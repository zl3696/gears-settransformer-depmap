"""Evaluate frozen additive predictions on one saved Norman validation split.

Reuse the actual GEARS compute_metrics function. Only validation expression rows
and their first 20 released DE gene IDs are decoded. Final test scoring is not
part of this runner. All source data and prediction/split bundles stay read-only.
"""

from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import gzip
import importlib.util
import json
import os
import tempfile
import warnings

import h5py
import numpy as np

from additive_baseline import (
    ROOT, SOURCE, SPLITS, OUTPUT as PREDICTIONS, aggregate_selected_rows,
    check_frozen_bundle, digest, json_bytes, read_column, read_csv, require,
)

METRICS = ("mse", "mse_de", "pearson", "pearson_de")
REFERENCE_SCORER = ROOT / "src/GEARS/gears/inference.py"
TEAM_SCORER = ROOT.parent / "Code/gears-settransformer-depmap/gears/inference.py"


def load_scorer(reference=REFERENCE_SCORER, team=TEAM_SCORER):
    """Import the existing implementation without changing the installed package."""
    require(digest(reference) == digest(team), "Original and team GEARS scorers differ")
    cache = ROOT / "tmp/norman-additive-evaluation"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    spec = importlib.util.spec_from_file_location("norman_shared_gears_inference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_metrics


def align_predictions(conditions, gene_ids, values, wanted_conditions, wanted_genes):
    """Align by identifiers and reject ambiguous/missing axes."""
    conditions, gene_ids = list(conditions), list(gene_ids)
    wanted_conditions, wanted_genes = list(wanted_conditions), list(wanted_genes)
    require(len(set(conditions)) == len(conditions), "Duplicate prediction condition")
    require(len(set(gene_ids)) == len(gene_ids), "Duplicate prediction gene ID")
    require(len(set(wanted_conditions)) == len(wanted_conditions), "Duplicate evaluation condition")
    require(len(set(wanted_genes)) == len(wanted_genes), "Duplicate evaluation gene ID")
    require(values.shape == (len(conditions), len(gene_ids)), "Prediction shape/axis mismatch")
    require(set(wanted_conditions) <= set(conditions), "Missing prediction condition")
    require(set(wanted_genes) == set(gene_ids), "Prediction expression panel differs")
    row_lookup, col_lookup = ({v: i for i, v in enumerate(axis)} for axis in (conditions, gene_ids))
    result = values[np.ix_([row_lookup[c] for c in wanted_conditions],
                          [col_lookup[g] for g in wanted_genes])]
    require(np.isfinite(result).all(), "Non-finite prediction")
    return result


def select_validation_cells(assignments, manifest):
    pairs = sorted(manifest["pair_conditions"]["val"])
    cells = [row for row in assignments if row["partition"] == "val"]
    require(len(pairs) == len(set(pairs)) > 0, "Invalid validation condition list")
    require({row["canonical_condition"] for row in cells} == set(pairs),
            "Validation cells differ from the saved condition list")
    require(all(row["condition_type"] == "pair" for row in cells), "Non-pair validation target")
    require(len(cells) == manifest["target_cell_counts"]["val"], "Validation cell count mismatch")
    for pair in pairs:
        require(all(g in manifest["canonical_conditions"]["train"] for g in pair.split("+")),
                f"A constituent single is not in training: {pair}")
    return pairs, cells


def de_indices_for_condition(rankings, condition_name, gene_ids):
    """Match PertData's first-20 membership selection and expression-axis order."""
    require(condition_name in rankings, f"Missing released DE ranking: {condition_name}")
    dataset = rankings[condition_name]
    require(isinstance(dataset, h5py.Dataset) and dataset.ndim == 1 and len(dataset) >= 20,
            "Invalid DE ranking storage")
    # Do not decode rankings for other conditions or genes beyond the first 20.
    ranked = dataset[:20].astype(str)
    require(len(set(ranked)) == 20, "Duplicate DE gene IDs")
    indices = np.flatnonzero(np.isin(gene_ids, ranked))
    require(len(indices) == 20, "The first 20 DE genes do not map to the expression panel")
    return indices, ranked


def read_validation_targets(h, cells, pairs, gene_ids):
    """Read only selected validation expression rows and condition-specific rankings."""
    require(all(row["partition"] == "val" for row in cells), "Only validation targets may be read")
    require(np.array_equal(read_column(h["var"], "gene_id"), gene_ids), "Source gene order differs")
    barcodes = read_column(h["obs"], "cell_barcode")
    labels = read_column(h["obs"], "condition")
    names = read_column(h["obs"], "condition_name")
    for row in cells:
        index = int(row["source_row"])
        require((barcodes[index], labels[index], names[index]) ==
                (row["cell_id"], row["source_condition"], row["source_condition_name"]),
                "Source cell/condition alignment differs")
    lookup = {pair: i for i, pair in enumerate(pairs)}
    rows = np.array([int(row["source_row"]) for row in cells])
    codes = np.array([lookup[row["canonical_condition"]] for row in cells])
    observed, counts, entries = aggregate_selected_rows(h["X"], rows, codes, len(pairs))
    indices, ranked_ids, source_names = [], [], []
    for pair in pairs:
        selected = [row for row in cells if row["canonical_condition"] == pair]
        keys = {row["source_condition_name"] for row in selected}
        require(len(keys) == 1, "A pair has ambiguous source DE-ranking keys")
        key = next(iter(keys))
        de_idx, ranked = de_indices_for_condition(h["uns"]["rank_genes_groups_cov_all"], key, gene_ids)
        indices.append(de_idx)
        ranked_ids.append(ranked)
        source_names.append(key)
    return observed, counts, np.stack(indices), np.stack(ranked_ids), source_names, entries


def make_scorer_inputs(pairs, predicted, observed, de_indices):
    require(predicted.ndim == observed.ndim == 2 and predicted.shape == observed.shape,
            "Predicted and observed profiles must have matching two-dimensional shapes")
    require(len(pairs) == len(set(pairs)) == len(predicted), "Condition axis mismatch")
    require(np.isfinite(predicted).all() and np.isfinite(observed).all(), "Non-finite scoring input")
    require(de_indices.shape == (len(pairs), 20) and np.issubdtype(de_indices.dtype, np.integer),
            "Expected 20 integer DE indices per condition")
    require(np.all((de_indices >= 0) & (de_indices < predicted.shape[1])), "DE index outside gene axis")
    require(all(len(set(row)) == 20 for row in de_indices), "Duplicate DE indices")
    return {"pert_cat": np.asarray(pairs), "pred": predicted, "truth": observed,
            "pred_de": np.take_along_axis(predicted, de_indices, axis=1),
            "truth_de": np.take_along_axis(observed, de_indices, axis=1)}


def score_inputs(scorer, inputs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aggregate, by_condition = scorer(inputs)
    require(set(aggregate) == set(METRICS), "GEARS core metric keys changed")
    aggregate = {key: float(aggregate[key]) for key in METRICS}
    by_condition = {condition: {key: float(values[key]) for key in METRICS}
                    for condition, values in by_condition.items()}
    require(all(np.isfinite(v) for v in aggregate.values()), "Non-finite aggregate score")
    flags = {}
    for i, pair in enumerate(inputs["pert_cat"]):
        flags[str(pair)] = {
            "pearson_undefined_to_zero": bool(np.ptp(inputs["pred"][i]) == 0 or
                                               np.ptp(inputs["truth"][i]) == 0),
            "pearson_de_undefined_to_zero": bool(np.ptp(inputs["pred_de"][i]) == 0 or
                                                  np.ptp(inputs["truth_de"][i]) == 0),
        }
    return aggregate, by_condition, flags, dict(Counter(type(w.message).__name__ for w in caught))


def prepare_inputs(source, split_dir, prediction_dir, seed):
    frozen = check_frozen_bundle(split_dir)
    prediction_summary = json.loads((prediction_dir / "summary.json").read_text())
    for name, expected in prediction_summary["artifact_sha256"].items():
        require(digest(prediction_dir / name) == expected, f"Prediction artifact changed: {name}")
    require(prediction_summary["split_bundle_sha256"] == frozen["bundle_sha256"],
            "Predictions use a different split bundle")
    require(digest(source) == prediction_summary["source_sha256"], "Source H5AD changed")
    manifest_path = split_dir / f"seed_{seed}/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pairs, cells = select_validation_cells(read_csv(split_dir / f"seed_{seed}/cell_assignments.csv.gz"), manifest)
    genes = read_csv(split_dir / "expression_genes.csv")
    gene_ids, gene_names = (np.array([row[key] for row in genes]) for key in ("gene_id", "gene_name"))
    with np.load(prediction_dir / "predictions.npz", allow_pickle=False) as saved:
        predicted = align_predictions(saved["conditions"], saved["gene_ids"], saved["additive_mean"],
                                      pairs, gene_ids)
    configuration = {
        "model": "additive", "split_seed": seed, "partition": "val", "metrics": list(METRICS),
        "conditions": pairs, "source_sha256": prediction_summary["source_sha256"],
        "split_bundle_sha256": frozen["bundle_sha256"], "manifest_sha256": digest(manifest_path),
        "predictions_sha256": digest(prediction_dir / "predictions.npz"),
        "prediction_summary_sha256": digest(prediction_dir / "summary.json"),
        "runner_sha256": digest(Path(__file__)), "aggregation_helper_sha256": digest(Path(__file__).with_name("additive_baseline.py")),
        "original_scorer_sha256": digest(REFERENCE_SCORER), "team_scorer_sha256": digest(TEAM_SCORER),
        "scorer": "Existing GEARS compute_metrics; one condition-mean row per pair",
        "mean_dtype": "float64", "condition_weighting": "equal weight per pair",
        "de_policy": "First 20 released rank_genes_groups_cov_all IDs; expression-axis membership order",
        "undefined_pearson_policy": "Use existing scorer's NaN-to-zero conversion; record affected pairs",
        "test_outcomes_permitted": False,
    }
    return pairs, cells, gene_ids, gene_names, predicted, configuration


def evaluate(source, cells, pairs, gene_ids, predicted, scorer):
    before = source.stat()
    with h5py.File(source, "r") as h:
        observed, counts, indices, ranked, source_names, entries = read_validation_targets(h, cells, pairs, gene_ids)
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Source changed during evaluation")
    inputs = make_scorer_inputs(pairs, predicted, observed, indices)
    metrics, by_condition, flags, caught = score_inputs(scorer, inputs)
    return inputs, indices, ranked, source_names, counts, metrics, by_condition, flags, caught, entries


def write_csv(path, rows):
    import csv
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def publish(stage, configuration, cells, genes, names, result):
    inputs, indices, ranked, source_names, counts, metrics, by_condition, flags, caught, entries = result
    pairs = inputs["pert_cat"].tolist()
    np.savez_compressed(stage / "scoring_inputs.npz", **inputs, de_indices=indices, gene_ids=genes, gene_names=names)
    rows = [{"model": "additive", "split_seed": configuration["split_seed"], "partition": "val",
             "canonical_condition": pair, "target_cells": int(counts[i]),
             **by_condition[pair], **flags[pair]} for i, pair in enumerate(pairs)]
    write_csv(stage / "per_condition_metrics.csv", rows)
    scoring_genes = []
    for i, pair in enumerate(pairs):
        rank_lookup = {gene: rank + 1 for rank, gene in enumerate(ranked[i])}
        for position, index in enumerate(indices[i]):
            scoring_genes.append({"canonical_condition": pair, "source_condition_name": source_names[i],
                                  "scoring_position": position, "released_rank": rank_lookup[genes[index]],
                                  "expression_index": int(index), "gene_id": genes[index], "gene_name": names[index]})
    write_csv(stage / "scoring_genes.csv", scoring_genes)
    with gzip.open(stage / "evaluation_cells.csv.gz", "wt", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)
    summary = {
        "status": "validation_evaluated", "model": "additive", "split_seed": configuration["split_seed"],
        "partition": "val", "metrics": metrics, "evaluated_pairs": len(pairs),
        "evaluated_cells": len(cells), "expression_genes": len(genes), "de_genes_per_pair": 20,
        "aggregation": "Arithmetic mean of pair-level metrics, equal weight per pair",
        "undefined_to_zero_counts": {key: sum(row[key] for row in flags.values()) for key in next(iter(flags.values()))},
        "scorer_warning_counts": caught, "validation_expression_entries_decoded": entries,
        "validation_ranking_keys_decoded": source_names, "test_expression_rows_decoded_in_this_run": 0,
        "test_ranking_keys_decoded_in_this_run": 0,
        "cross_split_exposure": "Validation pairs in this split may be test in other predefined splits; do not adapt later model choices to these scores",
        "prior_exposure": "Dataset-wide descriptive PCA and earlier previews are recorded in the frozen split provenance",
        "comparison_status": "Additive only; no trained GEARS/Set Transformer accuracy results available locally",
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "versions": {name: version(name) for name in ("numpy", "h5py", "scipy", "scikit-learn")},
        "configuration_sha256": digest(stage / "evaluation_config.json"),
        "artifact_sha256": {path.name: digest(path) for path in stage.iterdir()},
    }
    (stage / "summary.json").write_bytes(json_bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--split-seed", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTIONS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source, split_dir, prediction_dir = (p.resolve() for p in (args.source, args.split_dir, args.prediction_dir))
    output = (args.output_dir or ROOT / f"outputs/norman-additive-evaluation/seed_{args.split_seed}/val").resolve()
    require(output.is_relative_to(ROOT / "outputs"), "Keep evaluation outputs inside Codex Outputs/outputs")
    for directory in (split_dir, prediction_dir):
        require(not output.is_relative_to(directory) and not directory.is_relative_to(output), "Separate evaluation outputs from input bundles")
    require((not output.exists()) if args.command == "run" else output.is_dir(),
            "Run requires a new output directory; verify requires an existing bundle")
    scorer = load_scorer()
    pairs, cells, genes, names, predicted, config = prepare_inputs(source, split_dir, prediction_dir, args.split_seed)
    if args.command == "verify":
        require(json.loads((output / "evaluation_config.json").read_text()) == config, "Evaluation configuration changed")
        summary = json.loads((output / "summary.json").read_text())
        for name, expected in summary["artifact_sha256"].items():
            require(digest(output / name) == expected, f"Evaluation artifact changed: {name}")
        result = evaluate(source, cells, pairs, genes, predicted, scorer)
        require(summary["metrics"] == result[5], "Regenerated aggregate metrics differ")
        with np.load(output / "scoring_inputs.npz", allow_pickle=False) as saved:
            for key, values in {**result[0], "de_indices": result[1], "gene_ids": genes, "gene_names": names}.items():
                require(np.array_equal(saved[key], values), f"Regenerated scoring input differs: {key}")
        print(json.dumps({"status": "passed", "identical_score_and_input_regeneration": True,
                          "partition": "val", "split_seed": args.split_seed, "metrics": result[5]}, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="validation-build-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        # Persist the complete scoring configuration before decoding any outcomes.
        (stage / "evaluation_config.json").write_bytes(json_bytes(config))
        result = evaluate(source, cells, pairs, genes, predicted, scorer)
        summary = publish(stage, config, cells, genes, names, result)
        stage.rename(output)
    print(json.dumps({"output": str(output), **summary}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
