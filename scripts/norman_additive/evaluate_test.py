"""Score frozen additive predictions on a saved final-test partition.

The prediction builder, validation adapter, and GEARS scorer remain unchanged.
Only this split's test expression rows and first-20 released DE IDs are decoded.
"""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import csv
import gzip
import json
import tempfile

import h5py
import numpy as np

from additive_baseline import (
    ROOT, SOURCE, SPLITS, OUTPUT as PREDICTIONS, aggregate_selected_rows,
    check_frozen_bundle, digest, json_bytes, read_column, read_csv, require,
)
from evaluate_validation import (
    METRICS, REFERENCE_SCORER, TEAM_SCORER, align_predictions,
    de_indices_for_condition, load_scorer, make_scorer_inputs, score_inputs, write_csv,
)

EVALUATIONS = ROOT / "outputs/norman-additive-evaluation"


def select_test_cells(assignments, manifest):
    pairs = sorted(manifest["pair_conditions"]["test"])
    cells = [row for row in assignments if row["partition"] == "test"]
    require(len(pairs) == len(set(pairs)) > 0, "Invalid test condition list")
    require({row["canonical_condition"] for row in cells} == set(pairs),
            "Test cells differ from the saved condition list")
    require(all(row["condition_type"] == "pair" for row in cells), "Non-pair test target")
    require(len(cells) == manifest["target_cell_counts"]["test"], "Test cell count mismatch")
    for pair in pairs:
        require(all(g in manifest["canonical_conditions"]["train"] for g in pair.split("+")),
                f"A constituent single is not in training: {pair}")
    return pairs, cells


def read_test_targets(h, cells, pairs, gene_ids):
    """Aggregate exactly the selected test rows; keep validation/train values unread."""
    require(len(cells) > 0 and all(row["partition"] == "test" for row in cells),
            "Only test targets may be read")
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
        keys = {row["source_condition_name"] for row in cells if row["canonical_condition"] == pair}
        require(len(keys) == 1, "A pair has ambiguous source DE-ranking keys")
        key = next(iter(keys))
        de_idx, ranked = de_indices_for_condition(h["uns"]["rank_genes_groups_cov_all"], key, gene_ids)
        indices.append(de_idx)
        ranked_ids.append(ranked)
        source_names.append(key)
    return observed, counts, np.stack(indices), np.stack(ranked_ids), source_names, entries


def prepare_inputs(source, split_dir, prediction_dir, seed):
    frozen = check_frozen_bundle(split_dir)
    prediction_summary = json.loads((prediction_dir / "summary.json").read_text())
    for name, expected in prediction_summary["artifact_sha256"].items():
        require(digest(prediction_dir / name) == expected, f"Prediction artifact changed: {name}")
    builder = Path(__file__).with_name("additive_baseline.py")
    require(digest(builder) == prediction_summary["runner_sha256"], "Prediction builder changed")
    require(prediction_summary["split_bundle_sha256"] == frozen["bundle_sha256"],
            "Predictions use a different split bundle")
    require(digest(source) == prediction_summary["source_sha256"], "Source H5AD changed")
    manifest_path = split_dir / f"seed_{seed}/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["split_seed"] == seed, "Manifest seed differs")
    pairs, cells = select_test_cells(read_csv(split_dir / f"seed_{seed}/cell_assignments.csv.gz"), manifest)
    genes = read_csv(split_dir / "expression_genes.csv")
    gene_ids, gene_names = (np.array([row[key] for row in genes]) for key in ("gene_id", "gene_name"))
    with np.load(prediction_dir / "predictions.npz", allow_pickle=False) as saved:
        predicted = align_predictions(saved["conditions"], saved["gene_ids"], saved["additive_mean"], pairs, gene_ids)
    # This record contains already-scored condition IDs, not additional expression outcomes.
    prior_path = EVALUATIONS / "validation_summary/summary.json"
    prior = json.loads(prior_path.read_text())
    require(prior["status"] == "all_five_validation_splits_evaluated" and
            prior["shared_configuration"]["split_bundle_sha256"] == frozen["bundle_sha256"],
            "Prior validation record differs")
    adapter_hash = digest(Path(__file__).with_name("evaluate_validation.py"))
    require(adapter_hash == prior["shared_configuration"]["runner_sha256"], "Validation adapter changed")
    for key, expected in (("predictions_sha256", digest(prediction_dir / "predictions.npz")),
                          ("original_scorer_sha256", digest(REFERENCE_SCORER)),
                          ("team_scorer_sha256", digest(TEAM_SCORER))):
        require(prior["shared_configuration"][key] == expected, f"Scoring differs from validation: {key}")
    config = {
        "model": "additive", "split_seed": seed, "partition": "test", "metrics": list(METRICS),
        "conditions": pairs, "source_sha256": prediction_summary["source_sha256"],
        "split_bundle_sha256": frozen["bundle_sha256"], "manifest_sha256": digest(manifest_path),
        "predictions_sha256": digest(prediction_dir / "predictions.npz"),
        "prediction_summary_sha256": digest(prediction_dir / "summary.json"),
        "runner_sha256": digest(Path(__file__)), "aggregation_helper_sha256": digest(builder),
        "validation_adapter_sha256": adapter_hash,
        "prior_validation_summary_sha256": digest(prior_path),
        "prior_validation_exposed_test_conditions": sorted(set(pairs) & set(prior["validation_conditions_accessed"])),
        "original_scorer_sha256": digest(REFERENCE_SCORER), "team_scorer_sha256": digest(TEAM_SCORER),
        "scorer": "Existing GEARS compute_metrics; one condition-mean row per pair",
        "mean_dtype": "float64", "condition_weighting": "equal weight per pair",
        "de_policy": "First 20 released rank_genes_groups_cov_all IDs; expression-axis membership order",
        "undefined_pearson_policy": "Use existing scorer's NaN-to-zero conversion; record affected pairs",
        "test_outcomes_permitted": True,
        "protocol_scope": "Fixed additive predictor and scoring contract; user authorized final tests; neural model protocol remains pending",
    }
    return pairs, cells, gene_ids, gene_names, predicted, config


def evaluate(source, cells, pairs, gene_ids, predicted, scorer):
    before = source.stat()
    with h5py.File(source, "r") as h:
        observed, counts, indices, ranked, source_names, entries = read_test_targets(h, cells, pairs, gene_ids)
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Source changed during evaluation")
    inputs = make_scorer_inputs(pairs, predicted, observed, indices)
    metrics, by_condition, flags, caught = score_inputs(scorer, inputs)
    return inputs, indices, ranked, source_names, counts, metrics, by_condition, flags, caught, entries


def publish(stage, config, cells, genes, names, result):
    inputs, indices, ranked, source_names, counts, metrics, by_condition, flags, caught, entries = result
    pairs = inputs["pert_cat"].tolist()
    np.savez_compressed(stage / "scoring_inputs.npz", **inputs, de_indices=indices, gene_ids=genes, gene_names=names)
    rows = [{"model": "additive", "split_seed": config["split_seed"], "partition": "test",
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
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)
    summary = {
        "status": "final_test_evaluated", "model": "additive", "split_seed": config["split_seed"],
        "partition": "test", "metrics": metrics, "evaluated_pairs": len(pairs),
        "evaluated_cells": len(cells), "expression_genes": len(genes), "de_genes_per_pair": 20,
        "aggregation": "Arithmetic mean of pair-level metrics, equal weight per pair",
        "undefined_to_zero_counts": {key: sum(row[key] for row in flags.values()) for key in next(iter(flags.values()))},
        "scorer_warning_counts": caught, "test_expression_entries_decoded": entries,
        "test_ranking_keys_decoded": source_names, "non_test_expression_rows_decoded_in_this_run": 0,
        "non_test_ranking_keys_decoded_in_this_run": 0,
        "prior_validation_exposed_test_conditions": config["prior_validation_exposed_test_conditions"],
        "prior_exposure": "Overlapping validation panels plus earlier descriptive PCA/previews; do not claim globally untouched test outcomes",
        "comparison_status": "Additive only; matched trained GEARS/Set Transformer results remain pending",
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
    output = (args.output_dir or EVALUATIONS / f"seed_{args.split_seed}/test").resolve()
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
        require(summary["configuration_sha256"] == digest(output / "evaluation_config.json"), "Configuration hash differs")
        for name, expected in summary["artifact_sha256"].items():
            require(digest(output / name) == expected, f"Evaluation artifact changed: {name}")
        result = evaluate(source, cells, pairs, genes, predicted, scorer)
        require(summary["metrics"] == result[5], "Regenerated aggregate metrics differ")
        with np.load(output / "scoring_inputs.npz", allow_pickle=False) as saved:
            for key, values in {**result[0], "de_indices": result[1], "gene_ids": genes, "gene_names": names}.items():
                require(np.array_equal(saved[key], values), f"Regenerated scoring input differs: {key}")
        rows = read_csv(output / "per_condition_metrics.csv")
        for row in rows:
            pair = row["canonical_condition"]
            require(all(float(row[key]) == result[6][pair][key] for key in METRICS), "Regenerated per-pair scores differ")
        print(json.dumps({"status": "passed", "identical_score_and_input_regeneration": True,
                          "partition": "test", "split_seed": args.split_seed, "metrics": result[5]}, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-build-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        # Freeze hashes, definitions, and condition IDs before reading test outcomes.
        (stage / "evaluation_config.json").write_bytes(json_bytes(config))
        result = evaluate(source, cells, pairs, genes, predicted, scorer)
        summary = publish(stage, config, cells, genes, names, result)
        stage.rename(output)
    print(json.dumps({"output": str(output), **summary}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
