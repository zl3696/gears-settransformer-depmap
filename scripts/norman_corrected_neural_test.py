"""Test the 15 published corrected-batching Norman neural checkpoints.

Reuse immutable legacy pairing, control streaming and scoring helpers. Preserve
all baseline data and historical neural outputs. Never call a trainer.
"""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import gc
import importlib.util
import json
import resource
import shutil

import h5py
import numpy as np
import torch
import norman_neural_test as legacy
from norman_neural_test import (
    CPUUnpickler, METRICS, assigned_controls, check_artifact, check_bundle,
    check_frozen_bundle, digest, independent_scores, json_bytes, load_targets,
    metric_statistics, predict_means, read_column, read_control_rows, read_csv,
    read_json, require, shared, tensor_digest,
)

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY / "configs/norman_corrected_neural_test.json"


def source_for_model(protocol, model):
    matches = [source for source in protocol["sources"] if model in source["models"]]
    require(len(matches) == 1, f"Expected exactly one source for model: {model}")
    return matches[0]


def checkpoint_directory(root, run_id, source):
    return root / "results/norman" / source["result_namespace"] / run_id


def check_training_record(record, training, root, run_id, model_name, seed, protocol):
    require(record["run_id"] == run_id and record["model"] == model_name and record["split_seed"] == seed and
            record["training_seed"] == 101 and record["split_bundle_sha256"] == protocol["split_bundle_sha256"],
            "Checkpoint run/split identity differs")
    require(record["protocol_sha256"] == digest(root / "configs/norman_training.json") and
            len(record["validation_history"]) == training["training"]["epochs"] == 20,
            "Checkpoint training protocol differs")
    require(record["selection_metric"] == protocol["selection_metric"] and not record["test_evaluated"],
            "Published checkpoint selection/exposure differs")
    best = int(np.argmin([row[record["selection_metric"]] for row in record["validation_history"]]))
    require(record["best_epoch"] == best + 1 and record["selected_validation"] == record["validation_history"][best],
            "Checkpoint was not the recorded validation minimum")
    require(record["run_plan_sha256"] == digest(root / "configs/norman_runs.csv"), "Training run plan changed")


def load_model(root, run_id, model_name, seed, protocol, source):
    directory = checkpoint_directory(root, run_id, source)
    record = read_json(directory / "results.json")
    training = read_json(root / "configs/norman_training.json")
    check_training_record(record, training, root, run_id, model_name, seed, protocol)
    with (directory / "config.pkl").open("rb") as stream:
        config = CPUUnpickler(stream).load()
    require(config["num_genes"] == 5045 and config["num_perts"] == 9853 and config["hidden_size"] == 64,
            "Checkpoint dimensions differ")
    require(config["pert_agg"] == training["models"][model_name]["pert_agg"], "Wrong aggregation variant")
    for name, prefix in (("coexpress", "G_coexpress"), ("go", "G_go")):
        require(tensor_digest(config[prefix], config[prefix + "_weight"]) == record["graph_sha256"][name],
                "Checkpoint graph differs from training record")
    require(config["coexpression_batching"] == record["coexpression_batching"] == protocol["coexpression_batching"],
            "Checkpoint coexpression policy differs")
    config["device"] = "cpu"
    spec = importlib.util.spec_from_file_location("norman_published_model", root / "gears/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GEARS_Model(config)
    weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if next(iter(weights)).startswith("module."):
        weights = {key[7:]: value for key, value in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, record



def run_one(runtime, source, protocol, model_name, seed, regenerate=False):
    splits = REPOSITORY / "splits/norman"
    source_spec = source_for_model(protocol, model_name)
    root = legacy.checkpoint_root(runtime, source_spec)
    run_id = f"{model_name}_split{seed}_train101"
    output = runtime / protocol["output_directory"] / run_id
    if output.exists() and not regenerate:
        verify_one(runtime, protocol, model_name, seed)
        print(json.dumps({"verified_existing": run_id}), flush=True)
        return
    targets, baseline, target_config, pairs, cells, genes, names, arrays = load_targets(runtime, splits, seed, protocol)
    model, training_record = load_model(root, run_id, model_name, seed, protocol, source_spec)
    nodes = [row["gene_name"] for row in read_csv(splits / "perturbation_nodes.csv")]
    with h5py.File(source, "r") as h:
        require(np.array_equal(read_column(h["var"], "gene_id"), genes) and
                np.array_equal(read_column(h["var"], "gene_name"), names), "Source gene axis differs")
        inputs, audit = assigned_controls(cells, read_csv(splits / "evaluation_control_schedule.csv.gz"),
                                          read_csv(splits / "control_assignments.csv.gz"),
                                          read_column(h["obs"], "cell_barcode"), read_column(h["obs"], "condition"))
        control_rows = np.unique(inputs)
        controls, entries = read_control_rows(h["X"], control_rows)
    config = {"model": model_name, "run_id": run_id, "split_seed": seed, "partition": "test",
              "metrics": list(METRICS), "protocol_sha256": digest(PROTOCOL_PATH),
              "runner_sha256": digest(Path(__file__)), "model_source_sha256": digest(root / "gears/model.py"),
              "source_commit": source_spec["source_commit"], "result_namespace": source_spec["result_namespace"],
              "legacy_helpers_sha256": digest(REPOSITORY / "scripts/norman_neural_test.py"),
              "scorer_sha256": digest(REPOSITORY / "gears/inference.py"),
              "shared_helper_sha256": digest(REPOSITORY / "scripts/norman_additive/evaluate_validation.py"),
              "checkpoint_sha256": {name: digest(checkpoint_directory(root, run_id, source_spec) / name)
                                    for name in ("model.pt", "config.pkl", "results.json")},
              "source_h5ad_sha256": protocol["source_h5ad_sha256"], "split_bundle_sha256": protocol["split_bundle_sha256"],
              "baseline_target_summary_sha256": digest(targets / "summary.json"),
              "baseline_target_inputs_sha256": digest(targets / "scoring_inputs.npz"),
              "inference": protocol["inference"], "condition_weighting": "equal weight per pair",
              "conditions": pairs, "de_policy": target_config["de_policy"],
              "undefined_pearson_policy": target_config["undefined_pearson_policy"],
              "checkpoint_best_epoch": training_record["best_epoch"],
              "checkpoint_selection_metric": training_record["selection_metric"],
              "test_outcomes_permitted": True, "retraining_performed": False, "hyperparameter_tuning": False}
    if not regenerate:
        output.mkdir(parents=True)
        # Persist the fixed evaluation contract before running predictions or scoring.
        (output / "evaluation_config.json").write_bytes(json_bytes(config))
    else:
        require(read_json(output / "evaluation_config.json") == config, "Regeneration configuration differs")
    print(json.dumps({"start_run": run_id, "test_cells": len(cells), "pairs": len(pairs)}), flush=True)
    prediction, counts, sums, first, seconds = predict_means(model, controls, control_rows, inputs, cells, pairs,
                                                           nodes, protocol["inference"]["batch_size"])
    expected_counts = np.array([sum(r["canonical_condition"] == pair for r in cells) for pair in pairs])
    require(np.array_equal(counts, expected_counts), "Target-cell weighting differs from baseline")
    scorer = shared.load_scorer(reference=REPOSITORY / "gears/inference.py", team=REPOSITORY / "gears/inference.py")
    scoring = shared.make_scorer_inputs(pairs, prediction, arrays["truth"], arrays["de_indices"])
    scores, per_pair, flags, caught = shared.score_inputs(scorer, scoring)
    original, _, _, _ = shared.score_inputs(scorer, arrays)
    require(original == baseline["metrics"], "Historical additive test score was not reproduced")
    independent = independent_scores(scoring, scores)
    if regenerate:
        with np.load(output / "scoring_inputs.npz", allow_pickle=False) as saved:
            for key, value in scoring.items():
                require(np.array_equal(saved[key], value), f"Regenerated prediction/scoring input differs: {key}")
        require(read_json(output / "summary.json")["metrics"] == scores, "Regenerated metrics differ")
        (output / "regeneration_report.json").write_bytes(json_bytes({"status": "passed",
            "all_test_cell_predictions_and_means_regenerated": True, "identical_scoring_arrays": True}))
        return
    np.savez_compressed(output / "scoring_inputs.npz", **scoring, de_indices=arrays["de_indices"],
                        gene_ids=genes, gene_names=names, predicted_sums=sums, predicted_counts=counts)
    np.savez_compressed(output / "first_batch_predictions.npz", prediction=first)
    rows = [{"model": model_name, "split_seed": seed, "partition": "test", "canonical_condition": pair,
             "target_cells": int(counts[i]), **per_pair[pair], **flags[pair]} for i, pair in enumerate(pairs)]
    shared.write_csv(output / "per_condition_metrics.csv", rows)
    for name in ("evaluation_cells.csv.gz", "scoring_genes.csv"):
        shutil.copyfile(targets / name, output / name)
    shared.write_csv(output / "prediction_schedule.csv", [{**row, "batch_index": i // 32, "batch_position": i % 32}
                                                         for i, row in enumerate(audit)])
    (output / "independent_metrics_check.json").write_bytes(json_bytes({"status": "passed", "metrics": independent}))
    summary = {"status": "final_test_evaluated", "model": model_name, "run_id": run_id, "split_seed": seed,
               "partition": "test", "metrics": scores, "evaluated_pairs": len(pairs), "evaluated_cells": len(cells),
               "expression_genes": len(genes), "de_genes_per_pair": 20, "inference_seconds": seconds,
               "control_rows_read": len(control_rows), "control_sparse_entries_read": entries,
               "test_expression_rows_read_from_h5ad": 0, "counts_layer_read": False,
               "truth_and_de_indices": "Exact existing verified additive final-test scoring arrays",
               "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
               "undefined_to_zero_counts": {key: sum(f[key] for f in flags.values()) for key in next(iter(flags.values()))},
               "scorer_warning_counts": caught, "prior_exposure": "Existing baseline tests, overlapping validation panels, and descriptive PCA",
               "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
               "versions": {name: version(name) for name in ("torch", "torch-geometric", "numpy", "h5py", "scipy", "scikit-learn")},
               "configuration_sha256": digest(output / "evaluation_config.json"),
               "artifact_sha256": {p.name: digest(p) for p in output.iterdir()}}
    (output / "summary.json").write_bytes(json_bytes(summary))
    verify_one(runtime, protocol, model_name, seed)
    print(json.dumps({"finished_run": run_id, "metrics": scores, "inference_seconds": seconds}), flush=True)
    del model, controls, arrays
    gc.collect()


def verify_one(runtime, protocol, model_name, seed):
    output = runtime / protocol["output_directory"] / f"{model_name}_split{seed}_train101"
    summary = check_bundle(output)
    config = read_json(output / "evaluation_config.json")
    require(config["protocol_sha256"] == digest(PROTOCOL_PATH) and config["runner_sha256"] == digest(Path(__file__)),
            "Evaluation source/config changed")
    require(config["inference"] == protocol["inference"], "Inference policy changed")
    require(config["legacy_helpers_sha256"] == digest(REPOSITORY / "scripts/norman_neural_test.py") and
            config["shared_helper_sha256"] == digest(REPOSITORY / "scripts/norman_additive/evaluate_validation.py"),
            "Shared neural/scoring helpers changed")
    source_spec = source_for_model(protocol, model_name)
    root = legacy.checkpoint_root(runtime, source_spec)
    require(config["source_commit"] == source_spec["source_commit"] and
            config["model_source_sha256"] == digest(root / "gears/model.py"), "Published model source changed")
    require(config["model"] == model_name and config["split_seed"] == seed and config["partition"] == "test",
            "Evaluation identity changed")
    for name, expected in config["checkpoint_sha256"].items():
        require(digest(checkpoint_directory(root, config["run_id"], source_spec) / name) == expected, "Checkpoint changed")
    targets, baseline, _, pairs, cells, genes, names, original = load_targets(runtime, REPOSITORY / "splits/norman", seed, protocol)
    require(digest(targets / "summary.json") == config["baseline_target_summary_sha256"] and
            digest(targets / "scoring_inputs.npz") == config["baseline_target_inputs_sha256"], "Baseline targets changed")
    with np.load(output / "scoring_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("pert_cat", "pred", "truth", "pred_de", "truth_de")}
        for key in ("truth", "truth_de", "de_indices", "gene_ids", "gene_names", "pert_cat"):
            require(np.array_equal(saved[key], original[key]), f"Neural/baseline test input differs: {key}")
        require(np.array_equal(saved["pred"], saved["predicted_sums"] / saved["predicted_counts"][:, None]),
                "Prediction aggregation differs")
        require(np.array_equal(saved["pred_de"], np.take_along_axis(saved["pred"], saved["de_indices"], axis=1)),
                "Predicted DE slice differs")
    for name in ("evaluation_cells.csv.gz", "scoring_genes.csv"):
        require(digest(output / name) == digest(targets / name), "Test cells/scoring genes differ from baseline")
    require(summary["evaluated_pairs"] == len(pairs) == 26 and summary["evaluated_cells"] == len(cells), "Test size differs")
    scorer = shared.load_scorer(reference=REPOSITORY / "gears/inference.py", team=REPOSITORY / "gears/inference.py")
    scores, per_pair, _, _ = shared.score_inputs(scorer, inputs)
    require(scores == summary["metrics"], "Rescored neural metrics differ")
    require(shared.score_inputs(scorer, original)[0] == baseline["metrics"], "Historical baseline score differs")
    for row in read_csv(output / "per_condition_metrics.csv"):
        require(all(float(row[key]) == per_pair[row["canonical_condition"]][key] for key in METRICS), "Per-pair score differs")
    schedule = read_csv(output / "prediction_schedule.csv")
    expected_schedule = {row["target_cell_id"]: row for row in read_csv(REPOSITORY / "splits/norman/evaluation_control_schedule.csv.gz")}
    require(len(schedule) == len(cells), "Saved prediction schedule size changed")
    for i, (row, cell) in enumerate(zip(schedule, cells)):
        expected = {**cell, **expected_schedule[cell["cell_id"]], "batch_index": str(i // 32), "batch_position": str(i % 32)}
        require(row == expected, "Saved control/batch schedule differs")
    independent_scores(inputs, scores)
    (output / "verification_report.json").write_bytes(json_bytes({"status": "passed", "exact_baseline_targets_cells_genes": True,
        "baseline_metrics_reproduced": True, "saved_neural_scores_reproduced": True, "independent_numpy_metrics": True,
        "frozen_control_and_batch_schedule": True}))
    return summary


def summarize(runtime, protocol):
    output = runtime / protocol["output_directory"] / "test_summary"
    output.mkdir(parents=True, exist_ok=True)
    split_rows = []
    for model, directory in (("additive", "norman-additive-evaluation"), ("linear", "norman-linear-evaluation")):
        for seed in protocol["split_seeds"]:
            bundle = runtime / f"outputs/{directory}/seed_{seed}/test"
            summary = check_bundle(bundle)
            split_rows.append({"model": model, "split_seed": seed, "test_cells": summary["evaluated_cells"], **summary["metrics"]})
    for model in protocol["models"]:
        for seed in protocol["split_seeds"]:
            summary = verify_one(runtime, protocol, model, seed)
            split_rows.append({"model": model, "split_seed": seed, "test_cells": summary["evaluated_cells"], **summary["metrics"]})
    statistics, differences = [], []
    models = ["additive", "linear", *protocol["models"]]
    for model in models:
        statistics.extend({"model": model, **row} for row in metric_statistics([r for r in split_rows if r["model"] == model]))
    for model in protocol["models"]:
        for reference in ("additive", "linear", "gears", "gears_set_transformer"):
            if model == reference:
                continue
            a = [r for r in split_rows if r["model"] == model]
            b = [r for r in split_rows if r["model"] == reference]
            delta = [{"split_seed": r["split_seed"], **{k: r[k] - s[k] for k in METRICS}} for r, s in zip(a, b)]
            differences.extend({"model": model, "reference": reference, **row} for row in metric_statistics(delta))
    shared.write_csv(output / "split_metrics.csv", split_rows)
    shared.write_csv(output / "metric_summary.csv", statistics)
    shared.write_csv(output / "paired_difference_summary.csv", differences)
    table = ["# Norman matched final-test results", "", "All models use the same five test panels, cells, measured means, genes, DE indices and GEARS scorer.", "",
             "| Model | MSE (mean +/- sample SD) | DE MSE | Pearson | DE Pearson |", "|---|---:|---:|---:|---:|"]
    for model in models:
        lookup = {r["metric"]: r for r in statistics if r["model"] == model}
        values = [f"{lookup[k]['mean']:.6f} +/- {lookup[k]['sample_standard_deviation']:.6f}" for k in METRICS]
        table.append("| " + " | ".join([model, *values]) + " |")
    table += ["", "MSE metrics are raw; Pearson uses absolute expression. Each split averages its 26 pair scores equally; sample statistics use ddof=1.", "",
              "All three neural variants retain their published corrected shared coexpression batching. Fixed source-row order and batch size 32 match the earlier baseline test contract.", "",
              "Published neural checkpoints were selected using validation mse_de. GEARS and replacement Set Transformer use coexpression_shared_v1; residual Set Transformer uses residual_comparison_v1. The baseline outcomes, controls and four metrics are unchanged. No reselection, refit or tuning was performed.", "",
              "Panels overlap and describe split variability, not independent biological replicates. Test outcomes were previously exposed by the baseline evaluations and some validation panels.", ""]
    (output / "report.md").write_text("\n".join(table))
    summary = {"status": "all_fifteen_corrected_neural_test_runs_verified", "models": models, "split_seeds": protocol["split_seeds"],
               "metrics": statistics, "protocol_sha256": digest(PROTOCOL_PATH), "matched_baseline_contract": True,
               "artifact_sha256": {p.name: digest(p) for p in output.iterdir() if p.name != "summary.json"}}
    (output / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "run", "verify", "regenerate", "summary"))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", choices=("gears", "gears_set_transformer", "gears_residual_set_transformer"))
    parser.add_argument("--split-seed", type=int, choices=range(1, 6))
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    require(not runtime.is_relative_to(REPOSITORY) and not REPOSITORY.is_relative_to(runtime), "Runtime must be separate from source checkout")
    protocol = read_json(PROTOCOL_PATH)
    shared.ROOT = runtime
    torch.set_num_threads(protocol["inference"]["threads"])
    torch.manual_seed(101)
    if args.command == "download":
        for source in protocol["sources"]:
            legacy.download(runtime, {**source, "repository": protocol["repository"]})
        return
    frozen = check_frozen_bundle(REPOSITORY / "splits/norman")
    require(frozen["bundle_sha256"] == protocol["split_bundle_sha256"], "Frozen splits changed")
    for source_spec in protocol["sources"]:
        root = legacy.checkpoint_root(runtime, source_spec)
        for artifact in source_spec["artifacts"]:
            check_artifact(root / artifact["path"], artifact)
    if args.command == "summary":
        summarize(runtime, protocol)
        return
    source = (args.source or runtime / "data/norman/perturb_processed.h5ad").resolve()
    before = source.stat()
    if args.command in ("run", "regenerate"):
        require(digest(source) == protocol["source_h5ad_sha256"], "Source H5AD differs from baseline")
    for model in ([args.model] if args.model else protocol["models"]):
        for seed in ([args.split_seed] if args.split_seed else protocol["split_seeds"]):
            if args.command == "verify":
                verify_one(runtime, protocol, model, seed)
            else:
                run_one(runtime, source, protocol, model, seed, regenerate=args.command == "regenerate")
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Source changed during evaluation")


if __name__ == "__main__":
    main()
