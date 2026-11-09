"""Evaluate published neural checkpoints on the unchanged Norman baseline tests.

Only assigned training-control expression rows are read from H5AD. Measured test
means and DE indices come from the verified additive test bundles. No fitting,
checkpoint selection, graph correction, or test-dependent tuning is performed.
"""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import gc
import hashlib
import importlib.util
import io
import json
import pickle
import resource
import shutil
import sys
import time
import urllib.request

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch, Data

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY / "configs/norman_neural_test.json"
sys.path.insert(0, str(REPOSITORY / "scripts/norman_additive"))
from additive_baseline import check_frozen_bundle, digest, json_bytes, read_column, read_csv, require
import evaluate_validation as shared
from evaluate_test import select_test_cells
from summarize_validation import metric_statistics

METRICS = shared.METRICS


def read_json(path):
    return json.loads(Path(path).read_text())


def blob_sha(path):
    path = Path(path)
    value = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_artifact(path, artifact):
    require(path.stat().st_size == artifact["size"] and blob_sha(path) == artifact["git_blob_sha1"],
            f"Published artifact differs: {path}")


def checkpoint_root(runtime, protocol):
    return runtime / "data" / ("norman-neural-checkpoints-" + protocol["source_commit"][:8])


def download(runtime, protocol):
    """Download only explicitly pinned public artifacts, never execute a trainer."""
    root = checkpoint_root(runtime, protocol)
    root.mkdir(parents=True, exist_ok=True)
    for artifact in protocol["artifacts"]:
        target = root / artifact["path"]
        if target.exists():
            check_artifact(target, artifact)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/{protocol['repository']}/"
               f"{protocol['source_commit']}/{artifact['path']}")
        temporary = target.with_name(target.name + ".download")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            check_artifact(temporary, artifact)
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps({"downloaded": artifact["path"], "verified_git_blob": True}), flush=True)
    (root / "download_manifest.json").write_bytes(json_bytes(protocol))


def check_bundle(directory):
    summary = read_json(directory / "summary.json")
    require(summary["configuration_sha256"] == digest(directory / "evaluation_config.json"),
            f"Configuration changed: {directory}")
    for name, expected in summary["artifact_sha256"].items():
        require(digest(directory / name) == expected, f"Baseline/output artifact changed: {directory / name}")
    return summary


def load_targets(runtime, splits, seed, protocol):
    """Require the exact historical targets, cells, genes, and metric contract."""
    target_dir = runtime / f"outputs/norman-additive-evaluation/seed_{seed}/test"
    summary = check_bundle(target_dir)
    config = read_json(target_dir / "evaluation_config.json")
    require(read_json(target_dir / "verification_report.json")["status"] == "passed",
            "Baseline targets have not been verified")
    require(config["partition"] == "test" and config["split_seed"] == seed and
            config["metrics"] == list(METRICS), "Wrong baseline test panel")
    require(config["source_sha256"] == protocol["source_h5ad_sha256"] and
            config["split_bundle_sha256"] == protocol["split_bundle_sha256"], "Baseline source/split mismatch")
    require(config["original_scorer_sha256"] == config["team_scorer_sha256"] ==
            protocol["scorer_sha256"] == digest(REPOSITORY / "gears/inference.py"), "Baseline scorer changed")
    manifest = read_json(splits / f"seed_{seed}/manifest.json")
    pairs, cells = select_test_cells(read_csv(splits / f"seed_{seed}/cell_assignments.csv.gz"), manifest)
    require(cells == read_csv(target_dir / "evaluation_cells.csv.gz") and pairs == config["conditions"],
            "Baseline test cell/condition selection differs")
    genes = read_csv(splits / "expression_genes.csv")
    ids = np.array([r["gene_id"] for r in genes])
    names = np.array([r["gene_name"] for r in genes])
    with np.load(target_dir / "scoring_inputs.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    require(np.array_equal(arrays["pert_cat"], pairs) and np.array_equal(arrays["gene_ids"], ids) and
            np.array_equal(arrays["gene_names"], names), "Baseline target axes changed")
    require(np.array_equal(arrays["truth_de"], np.take_along_axis(arrays["truth"], arrays["de_indices"], axis=1)),
            "Baseline DE slice differs")
    return target_dir, summary, config, pairs, cells, ids, names, arrays


def assigned_controls(cells, schedule, assignments, source_barcodes, source_conditions):
    """Resolve one frozen control per target and reject any pairing mismatch."""
    schedule_map = {row["target_cell_id"]: row for row in schedule}
    assignment_map = {row["target_cell_id"]: row for row in assignments}
    require(len(schedule_map) == len(schedule) and len(assignment_map) == len(assignments),
            "Duplicate control assignment or input_index")
    target_rows = [int(row["source_row"]) for row in cells]
    require(np.all(np.diff(target_rows) > 0), "Test cells must remain in increasing source-row order")
    controls, audit = [], []
    for cell in cells:
        row = schedule_map[cell["cell_id"]]
        require(row == assignment_map[cell["cell_id"]] and int(row["input_index"]) == 0,
                "Evaluation/training frozen control assignments differ")
        target, control = int(row["target_source_row"]), int(row["control_source_row"])
        require(target == int(cell["source_row"]) and source_barcodes[target] == cell["cell_id"] and
                source_conditions[target] == cell["source_condition"], "Target row/cell/source-label mismatch")
        require(source_barcodes[control] == row["control_cell_id"] and source_conditions[control] == "ctrl",
                "Model input is not its assigned training control")
        controls.append(control)
        audit.append({**cell, **row})
    return np.array(controls), audit


def read_control_rows(x, rows):
    """Read only allowlisted CSR rows into a bounded float32 control cache."""
    rows = np.asarray(rows, dtype=np.int64)
    require(len(rows) > 0 and np.all(np.diff(rows) > 0), "Control rows must be unique and sorted")
    shape = tuple(x.attrs["shape"])
    require(x.attrs["encoding-type"] == "csr_matrix" and 0 <= rows[0] <= rows[-1] < shape[0],
            "Invalid control CSR selection")
    pointers = x["indptr"][:]
    result = np.zeros((len(rows), shape[1]), dtype=np.float32)
    boundaries = np.r_[0, np.flatnonzero(np.diff(rows) > 1) + 1, len(rows)]
    entries = 0
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        start, stop = rows[first], rows[last - 1] + 1
        left, right = int(pointers[start]), int(pointers[stop])
        values, columns = x["data"][left:right], x["indices"][left:right]
        require(np.isfinite(values).all() and np.all((columns >= 0) & (columns < shape[1])),
                "Invalid control expression")
        row_numbers = np.repeat(np.arange(first, last), np.diff(pointers[start:stop + 1]))
        np.add.at(result, (row_numbers, columns), values)
        entries += len(values)
    return result, entries


class CPUUnpickler(pickle.Unpickler):
    """Load hash-verified published config tensors onto CPU."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
        return super().find_class(module, name)


def tensor_digest(*tensors):
    value = hashlib.sha256()
    for tensor in tensors:
        value.update(tensor.cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def load_model(root, run_id, model_name, seed, protocol):
    directory = root / "results/norman" / run_id
    record = read_json(directory / "results.json")
    training = read_json(root / "configs/norman_training.json")
    require(record["run_id"] == run_id and record["model"] == model_name and record["split_seed"] == seed and
            record["training_seed"] == 101 and record["split_bundle_sha256"] == protocol["split_bundle_sha256"],
            "Checkpoint run/split identity differs")
    require(record["protocol_sha256"] == digest(root / "configs/norman_training.json") and
            len(record["validation_history"]) == training["training"]["epochs"] == 20,
            "Checkpoint training protocol differs")
    require(record["selection_metric"] == "normalized_mse_de" and not record["test_evaluated"],
            "Published checkpoint selection/exposure differs")
    with (directory / "config.pkl").open("rb") as stream:
        config = CPUUnpickler(stream).load()
    require(config["num_genes"] == 5045 and config["num_perts"] == 9853 and config["hidden_size"] == 64,
            "Checkpoint dimensions differ")
    require(config["pert_agg"] == training["models"][model_name]["pert_agg"], "Wrong aggregation variant")
    for name, prefix in (("coexpress", "G_coexpress"), ("go", "G_go")):
        require(tensor_digest(config[prefix], config[prefix + "_weight"]) == record["graph_sha256"][name],
                "Checkpoint graph differs from training record")
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


def predict_means(model, controls, control_rows, inputs, cells, pairs, perturbation_names, batch_size):
    """Stream fixed batches and average cell predictions in float64 per pair."""
    require(batch_size == 32, "This frozen inference protocol requires batch size 32")
    control_lookup = {int(row): i for i, row in enumerate(control_rows)}
    node_lookup = {str(name): i for i, name in enumerate(perturbation_names)}
    pair_lookup = {pair: i for i, pair in enumerate(pairs)}
    sums = np.zeros((len(pairs), controls.shape[1]), dtype=np.float64)
    counts = np.zeros(len(pairs), dtype=np.int64)
    first_prediction = None
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(cells), batch_size):
            stop = min(start + batch_size, len(cells))
            examples = [Data(x=torch.from_numpy(controls[control_lookup[int(inputs[i])]].copy()).reshape(-1, 1),
                             pert_idx=[node_lookup[g] for g in cells[i]["canonical_condition"].split("+")])
                        for i in range(start, stop)]
            batch = Batch.from_data_list(examples)
            prediction = model(batch).cpu().numpy()
            require(prediction.shape == (stop - start, controls.shape[1]) and np.isfinite(prediction).all(),
                    "Invalid neural prediction")
            if first_prediction is None:
                first_prediction = prediction.copy()
                require(np.array_equal(model(batch).cpu().numpy(), prediction), "Repeated first-batch inference differs")
            codes = np.array([pair_lookup[row["canonical_condition"]] for row in cells[start:stop]])
            np.add.at(sums, codes, prediction.astype(np.float64))
            np.add.at(counts, codes, 1)
            if start == 0 or stop == len(cells) or (start // batch_size + 1) % 50 == 0:
                print(json.dumps({"inferred_cells": stop, "total_cells": len(cells),
                                  "seconds": round(time.perf_counter() - start_time, 2)}), flush=True)
    require(np.all(counts > 0) and counts.sum() == len(cells), "Missing/duplicate predicted targets")
    return sums / counts[:, None], counts, sums, first_prediction, time.perf_counter() - start_time


def independent_scores(inputs, scores):
    values = {key: [] for key in METRICS}
    for i in range(len(inputs["pert_cat"])):
        for suffix in ("", "_de"):
            a, b = inputs["pred" + suffix][i], inputs["truth" + suffix][i]
            values["mse" + suffix].append(float(np.mean((a - b) ** 2)))
            ac, bc = a - a.mean(), b - b.mean()
            denominator = np.linalg.norm(ac) * np.linalg.norm(bc)
            values["pearson" + suffix].append(float(ac @ bc / denominator) if denominator else 0.0)
    result = {key: float(np.mean(value)) for key, value in values.items()}
    require(all(np.isclose(result[key], scores[key], rtol=1e-12, atol=1e-14) for key in METRICS),
            "Independent NumPy metrics differ from baseline scorer")
    return result


def run_one(runtime, source, protocol, model_name, seed, regenerate=False):
    splits = REPOSITORY / "splits/norman"
    root = checkpoint_root(runtime, protocol)
    run_id = f"{model_name}_split{seed}_train101"
    output = runtime / "outputs/norman-neural-test" / run_id
    if output.exists() and not regenerate:
        verify_one(runtime, protocol, model_name, seed)
        print(json.dumps({"verified_existing": run_id}), flush=True)
        return
    targets, baseline, target_config, pairs, cells, genes, names, arrays = load_targets(runtime, splits, seed, protocol)
    model, training_record = load_model(root, run_id, model_name, seed, protocol)
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
              "scorer_sha256": digest(REPOSITORY / "gears/inference.py"),
              "shared_helper_sha256": digest(REPOSITORY / "scripts/norman_additive/evaluate_validation.py"),
              "checkpoint_sha256": {name: digest(root / "results/norman" / run_id / name)
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
    output = runtime / "outputs/norman-neural-test" / f"{model_name}_split{seed}_train101"
    summary = check_bundle(output)
    config = read_json(output / "evaluation_config.json")
    require(config["protocol_sha256"] == digest(PROTOCOL_PATH) and config["runner_sha256"] == digest(Path(__file__)),
            "Evaluation source/config changed")
    require(config["inference"] == protocol["inference"], "Inference policy changed")
    root = checkpoint_root(runtime, protocol)
    for name, expected in config["checkpoint_sha256"].items():
        require(digest(root / "results/norman" / config["run_id"] / name) == expected, "Checkpoint changed")
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
    independent_scores(inputs, scores)
    (output / "verification_report.json").write_bytes(json_bytes({"status": "passed", "exact_baseline_targets_cells_genes": True,
        "baseline_metrics_reproduced": True, "saved_neural_scores_reproduced": True, "independent_numpy_metrics": True}))
    return summary


def summarize(runtime, protocol):
    output = runtime / "outputs/norman-neural-test/test_summary"
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
        for reference in ("additive", "linear", "gears"):
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
              "The trained neural coexpression implementation is preserved, including its batch-position dependence. Fixed source-row order and batch size 32 apply to both neural models.", "",
              "Published neural checkpoints were selected using validation normalized_mse_de; their final test scores here use the historical four baseline metrics. No reselection, refit or tuning was performed.", "",
              "Panels overlap and describe split variability, not independent biological replicates. Test outcomes were previously exposed by the baseline evaluations and some validation panels.", ""]
    (output / "report.md").write_text("\n".join(table))
    summary = {"status": "all_ten_neural_test_runs_verified", "models": models, "split_seeds": protocol["split_seeds"],
               "metrics": statistics, "protocol_sha256": digest(PROTOCOL_PATH), "matched_baseline_contract": True,
               "artifact_sha256": {p.name: digest(p) for p in output.iterdir() if p.name != "summary.json"}}
    (output / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "run", "verify", "regenerate", "summary"))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", choices=("gears", "gears_set_transformer"))
    parser.add_argument("--split-seed", type=int, choices=range(1, 6))
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    require(not runtime.is_relative_to(REPOSITORY) and not REPOSITORY.is_relative_to(runtime), "Runtime must be separate from source checkout")
    protocol = read_json(PROTOCOL_PATH)
    shared.ROOT = runtime
    torch.set_num_threads(protocol["inference"]["threads"])
    torch.manual_seed(101)
    if args.command == "download":
        download(runtime, protocol)
        return
    frozen = check_frozen_bundle(REPOSITORY / "splits/norman")
    require(frozen["bundle_sha256"] == protocol["split_bundle_sha256"], "Frozen splits changed")
    root = checkpoint_root(runtime, protocol)
    for artifact in protocol["artifacts"]:
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
