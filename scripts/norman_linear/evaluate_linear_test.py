"""Evaluate sealed linear predictions on the authorized final-test partitions."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import json
import shutil
import tempfile

import numpy as np

from linear_baseline import ROOT, SPLITS, OUTPUT, HELPERS, digest, json_bytes, read_csv, require, verify_artifacts
from evaluate_linear import EVALUATIONS, TARGETS, evaluate_saved, independent_metric_check
from evaluate_validation import METRICS, REFERENCE_SCORER, TEAM_SCORER, align_predictions, load_scorer, write_csv
from evaluate_test import select_test_cells
from additive_baseline import check_frozen_bundle


def validate_test_contract(config, summary, manifest, seed, source_hash, bundle_hash):
    require(config["partition"] == summary["partition"] == "test" and
            summary["status"] == "final_test_evaluated" and config["test_outcomes_permitted"] is True,
            "Expected an authorized final-test target bundle")
    require(config["split_seed"] == summary["split_seed"] == manifest["split_seed"] == seed,
            "Test target split seed differs")
    require(config["metrics"] == list(METRICS), "Final-test metrics differ")
    require(config["source_sha256"] == source_hash and config["split_bundle_sha256"] == bundle_hash,
            "Test targets and model use different source/splits")
    require(config["conditions"] == sorted(manifest["pair_conditions"]["test"]),
            "Test target conditions differ from manifest")


def prepare_test(split_dir, prediction_dir, target_dir, validation_dir, seed):
    frozen = check_frozen_bundle(split_dir)
    fit = verify_artifacts(prediction_dir)
    fit_config = json.loads((prediction_dir / "fit_config.json").read_text())
    require(fit["split_seed"] == fit_config["split_seed"] == seed, "Wrong fitted model split")
    require(fit_config["runner_sha256"] == digest(Path(__file__).with_name("linear_baseline.py")), "Fitter changed")
    require(fit_config["aggregation_helper_sha256"] == digest(HELPERS / "additive_baseline.py"), "Training helper changed")
    require(fit_config["pca_dimensions"] == 10 and fit_config["gene_penalty"] == fit_config["perturbation_penalty"] == .1,
            "Fixed linear hyperparameters changed")
    require(fit["split_bundle_sha256"] == fit_config["split_bundle_sha256"] == frozen["bundle_sha256"], "Fit split bundle changed")
    for filename, key in (("verification_report.json", "prediction_and_fit_regeneration"), ("independent_fit_check.json", "status")):
        report = json.loads((prediction_dir / filename).read_text())
        require(report["status"] == "passed" and bool(report[key]), "Fit has not passed its numerical audits")
    previous = verify_artifacts(validation_dir)
    previous_config = json.loads((validation_dir / "evaluation_config.json").read_text())
    require(previous["partition"] == previous_config["partition"] == "val" and previous["split_seed"] == seed,
            "Wrong prior validation bundle")
    require(previous["configuration_sha256"] == digest(validation_dir / "evaluation_config.json"), "Validation configuration changed")
    require(previous_config["predictions_sha256"] == digest(prediction_dir / "predictions.npz") and
            previous_config["fit_configuration_sha256"] == digest(prediction_dir / "fit_config.json"),
            "Final predictions differ from the validated frozen fit")
    target_summary = verify_artifacts(target_dir)
    target_config = json.loads((target_dir / "evaluation_config.json").read_text())
    require(target_summary["configuration_sha256"] == digest(target_dir / "evaluation_config.json"), "Test target configuration changed")
    manifest_path = split_dir / f"seed_{seed}/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_test_contract(target_config, target_summary, manifest, seed, fit["source_sha256"], frozen["bundle_sha256"])
    require(fit_config["training_conditions"] == sorted(manifest["canonical_conditions"]["train"]), "Fit population differs from frozen training partition")
    scorer_hash = digest(REFERENCE_SCORER)
    require(scorer_hash == digest(TEAM_SCORER) == previous_config["original_scorer_sha256"] ==
            target_config["original_scorer_sha256"] == target_config["team_scorer_sha256"], "Shared scorer changed")
    target_verification = json.loads((target_dir / "verification_report.json").read_text())
    require(target_verification["status"] == "passed" and target_verification["partition"] == "test" and
            target_verification["split_seed"] == seed and target_verification["metrics"] == target_summary["metrics"],
            "Additive test target bundle is not verified")
    pairs, cells = select_test_cells(read_csv(split_dir / f"seed_{seed}/cell_assignments.csv.gz"), manifest)
    require(read_csv(target_dir / "evaluation_cells.csv.gz") == cells, "Saved test target cells differ")
    genes = read_csv(split_dir / "expression_genes.csv")
    gene_ids, gene_names = (np.array([r[key] for r in genes]) for key in ("gene_id", "gene_name"))
    with np.load(prediction_dir / "predictions.npz", allow_pickle=False) as saved:
        predicted = align_predictions(saved["conditions"], saved["gene_ids"], saved["predicted_mean"], pairs, gene_ids)
        partitions = dict(zip(saved["conditions"], saved["partitions"]))
        require(all(partitions[c] == "test" for c in pairs), "Prediction partition is not test")
    config = {
        "model": "fitted_linear_pair_adaptation", "partition": "test", "split_seed": seed,
        "metrics": list(METRICS), "conditions": pairs, "test_outcomes_permitted": True,
        "protocol_scope": "User-authorized final evaluation of the existing fixed linear fits; neural protocol remains separate",
        "refit_performed": False, "hyperparameter_tuning": False,
        "source_sha256": fit["source_sha256"], "split_bundle_sha256": frozen["bundle_sha256"],
        "manifest_sha256": digest(manifest_path), "predictions_sha256": digest(prediction_dir / "predictions.npz"),
        "fit_configuration_sha256": digest(prediction_dir / "fit_config.json"),
        "model_sha256": digest(prediction_dir / "model.npz"),
        "prior_linear_validation_summary_sha256": digest(validation_dir / "summary.json"),
        "target_summary_sha256": digest(target_dir / "summary.json"),
        "target_configuration_sha256": digest(target_dir / "evaluation_config.json"),
        "target_verification_sha256": digest(target_dir / "verification_report.json"),
        "target_inputs_sha256": digest(target_dir / "scoring_inputs.npz"),
        "runner_sha256": digest(Path(__file__)),
        "shared_linear_evaluation_helper_sha256": digest(Path(__file__).with_name("evaluate_linear.py")),
        "shared_evaluation_helper_sha256": digest(HELPERS / "evaluate_validation.py"),
        "test_selection_helper_sha256": digest(HELPERS / "evaluate_test.py"),
        "original_scorer_sha256": scorer_hash, "team_scorer_sha256": scorer_hash,
        "condition_weighting": "equal weight per pair", "mean_dtype": "float64",
        "de_policy": target_config["de_policy"], "undefined_pearson_policy": target_config["undefined_pearson_policy"],
        "outcome_source": "Existing verified additive final-test scoring arrays; no raw H5AD access",
        "prior_validation_exposed_test_conditions": target_config["prior_validation_exposed_test_conditions"],
    }
    return config, pairs, cells, gene_ids, gene_names, predicted, target_summary


def publish_test(stage, config, cells, genes, names, target_dir, result):
    inputs, indices, scores, by_pair, flags, warnings, additive_scores, additive_pairs = result
    np.savez_compressed(stage / "scoring_inputs.npz", **inputs, de_indices=indices, gene_ids=genes, gene_names=names)
    counts = {pair: sum(r["canonical_condition"] == pair for r in cells) for pair in inputs["pert_cat"]}
    rows = [{"model": config["model"], "split_seed": config["split_seed"], "partition": "test",
             "canonical_condition": str(pair), "target_cells": counts[pair], **by_pair[pair], **flags[pair],
             **{"additive_" + k: additive_pairs[pair][k] for k in METRICS},
             **{"difference_from_additive_" + k: by_pair[pair][k] - additive_pairs[pair][k] for k in METRICS}}
            for pair in inputs["pert_cat"]]
    write_csv(stage / "per_condition_metrics.csv", rows)
    for filename in ("scoring_genes.csv", "evaluation_cells.csv.gz"):
        shutil.copyfile(target_dir / filename, stage / filename)
    (stage / "independent_metrics_check.json").write_bytes(json_bytes(independent_metric_check(inputs, scores)))
    summary = {
        "status": "final_test_evaluated", "model": config["model"], "partition": "test",
        "split_seed": config["split_seed"], "metrics": scores, "additive_metrics": additive_scores,
        "difference_from_additive": {k: scores[k] - additive_scores[k] for k in METRICS},
        "evaluated_pairs": len(inputs["pert_cat"]), "evaluated_cells": len(cells),
        "expression_genes": len(genes), "de_genes_per_pair": 20,
        "undefined_to_zero_counts": {k: sum(f[k] for f in flags.values()) for k in next(iter(flags.values()))},
        "scorer_warning_counts": warnings, "raw_h5ad_opened": False,
        "new_expression_rows_decoded_from_h5ad": 0, "new_ranking_keys_decoded": 0,
        "test_scores_computed": True, "refit_performed": False,
        "prior_validation_exposed_test_conditions": config["prior_validation_exposed_test_conditions"],
        "prior_exposure": "Overlapping validation panels, existing additive final tests, and descriptive PCA; no claim of globally untouched test outcomes",
        "neural_comparison_status": "Matched trained GEARS/Set Transformer predictions remain unavailable",
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "configuration_sha256": digest(stage / "evaluation_config.json"),
        "artifact_sha256": {p.name: digest(p) for p in stage.iterdir()},
    }
    (stage / "summary.json").write_bytes(json_bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--split-seed", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    splits = args.split_dir.resolve()
    predictions = (args.prediction_dir or OUTPUT / f"seed_{args.split_seed}").resolve()
    targets = (args.target_dir or TARGETS / f"seed_{args.split_seed}/test").resolve()
    validation = (args.validation_dir or EVALUATIONS / f"seed_{args.split_seed}/val").resolve()
    output = (args.output_dir or EVALUATIONS / f"seed_{args.split_seed}/test").resolve()
    require(output.is_relative_to(ROOT / "outputs"), "Keep final-test outputs under Codex Outputs/outputs")
    for path in (splits, predictions, targets, validation):
        require(not output.is_relative_to(path) and not path.is_relative_to(output), "Separate final tests from input bundles")
    require(not output.exists() if args.command == "run" else output.is_dir(), "Run requires new output; verify requires existing output")
    config, pairs, cells, genes, names, predicted, target_summary = prepare_test(splits, predictions, targets, validation, args.split_seed)
    scorer = load_scorer()
    if args.command == "verify":
        require(json.loads((output / "evaluation_config.json").read_text()) == config, "Final-test configuration changed")
        summary = verify_artifacts(output)
        result = evaluate_saved(scorer, targets / "scoring_inputs.npz", pairs, genes, names, predicted, target_summary["metrics"])
        require(summary["metrics"] == result[2] and summary["additive_metrics"] == result[6], "Regenerated final scores differ")
        with np.load(output / "scoring_inputs.npz", allow_pickle=False) as saved:
            for key, values in {**result[0], "de_indices": result[1], "gene_ids": genes, "gene_names": names}.items():
                require(np.array_equal(saved[key], values), f"Regenerated scoring input differs: {key}")
        for row in read_csv(output / "per_condition_metrics.csv"):
            require(all(float(row[k]) == result[3][row["canonical_condition"]][k] for k in METRICS), "Per-pair metrics differ")
        report = {"status": "passed", "partition": "test", "split_seed": args.split_seed,
                  "identical_score_and_input_regeneration": True, "metrics": result[2]}
        (output / "verification_report.json").write_bytes(json_bytes(report))
        print(json.dumps(report, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="linear-final-test-", dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        # Persist the fixed model/scoring contract before decoding saved test arrays.
        (stage / "evaluation_config.json").write_bytes(json_bytes(config))
        result = evaluate_saved(scorer, targets / "scoring_inputs.npz", pairs, genes, names, predicted, target_summary["metrics"])
        summary = publish_test(stage, config, cells, genes, names, targets, result)
        stage.rename(output)
    print(json.dumps({"split_seed": args.split_seed, "partition": "test", "metrics": summary["metrics"],
                      "difference_from_additive": summary["difference_from_additive"]}, indent=2))


if __name__ == "__main__":
    main()
