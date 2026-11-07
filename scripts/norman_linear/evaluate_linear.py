"""Score the fitted linear baseline against existing validation targets only."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import json
import shutil
import tempfile

import numpy as np

from linear_baseline import ROOT, SPLITS, OUTPUT, HELPERS, digest, json_bytes, read_csv, require, verify_artifacts
from evaluate_validation import (
    METRICS, REFERENCE_SCORER, TEAM_SCORER, align_predictions, load_scorer,
    make_scorer_inputs, score_inputs, select_validation_cells, write_csv,
)

TARGETS = ROOT / "outputs/norman-additive-evaluation"
EVALUATIONS = ROOT / "outputs/norman-linear-evaluation"


def prepare_evaluation(split_dir, prediction_dir, target_dir, seed):
    from additive_baseline import check_frozen_bundle
    frozen = check_frozen_bundle(split_dir)
    fit = verify_artifacts(prediction_dir)
    fit_config = json.loads((prediction_dir / "fit_config.json").read_text())
    require(fit_config["runner_sha256"] == digest(Path(__file__).with_name("linear_baseline.py")), "Linear fitter changed")
    require(fit_config["aggregation_helper_sha256"] == digest(HELPERS / "additive_baseline.py"), "Training helper changed")
    require(fit["split_seed"] == fit_config["split_seed"] == seed, "Wrong linear split")
    require(fit["split_bundle_sha256"] == frozen["bundle_sha256"], "Linear fit used another split bundle")
    verification = json.loads((prediction_dir / "verification_report.json").read_text())
    require(verification["status"] == "passed" and verification["prediction_and_fit_regeneration"], "Fit is not regeneration-verified")
    target_summary = json.loads((target_dir / "summary.json").read_text())
    target_config = json.loads((target_dir / "evaluation_config.json").read_text())
    for name, expected in target_summary["artifact_sha256"].items():
        require(digest(target_dir / name) == expected, f"Saved validation artifact changed: {name}")
    require(target_summary["configuration_sha256"] == digest(target_dir / "evaluation_config.json"), "Target configuration hash changed")
    require(target_config["partition"] == "val" and target_config["split_seed"] == seed and
            target_config["metrics"] == list(METRICS), "Targets must be this split's four-metric validation bundle")
    require(target_config["split_bundle_sha256"] == frozen["bundle_sha256"] and
            target_config["source_sha256"] == fit["source_sha256"], "Target/fit source mismatch")
    require(digest(REFERENCE_SCORER) == digest(TEAM_SCORER) == target_config["original_scorer_sha256"] == target_config["team_scorer_sha256"], "Shared scorer changed")
    manifest_path = split_dir / f"seed_{seed}/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pairs, cells = select_validation_cells(read_csv(split_dir / f"seed_{seed}/cell_assignments.csv.gz"), manifest)
    require(pairs == target_config["conditions"], "Saved targets use another pair list")
    require(read_csv(target_dir / "evaluation_cells.csv.gz") == cells, "Saved target-cell population differs")
    genes = read_csv(split_dir / "expression_genes.csv")
    gene_ids, gene_names = (np.array([r[key] for r in genes]) for key in ("gene_id", "gene_name"))
    with np.load(prediction_dir / "predictions.npz", allow_pickle=False) as saved:
        predicted = align_predictions(saved["conditions"], saved["gene_ids"], saved["predicted_mean"], pairs, gene_ids)
        partitions = dict(zip(saved["conditions"], saved["partitions"]))
        require(all(partitions[c] == "val" for c in pairs), "Prediction partition mismatch")
    configuration = {
        "model": "fitted_linear_pair_adaptation", "split_seed": seed, "partition": "val",
        "metrics": list(METRICS), "conditions": pairs,
        "source_sha256": fit["source_sha256"], "split_bundle_sha256": frozen["bundle_sha256"],
        "manifest_sha256": digest(manifest_path), "fit_configuration_sha256": digest(prediction_dir / "fit_config.json"),
        "prediction_summary_sha256": digest(prediction_dir / "summary.json"),
        "predictions_sha256": digest(prediction_dir / "predictions.npz"),
        "fit_verification_sha256": digest(prediction_dir / "verification_report.json"),
        "target_summary_sha256": digest(target_dir / "summary.json"),
        "target_inputs_sha256": digest(target_dir / "scoring_inputs.npz"),
        "target_genes_sha256": digest(target_dir / "scoring_genes.csv"),
        "runner_sha256": digest(Path(__file__)),
        "shared_evaluation_helper_sha256": digest(HELPERS / "evaluate_validation.py"),
        "original_scorer_sha256": digest(REFERENCE_SCORER), "team_scorer_sha256": digest(TEAM_SCORER),
        "condition_weighting": "equal weight per pair", "mean_dtype": "float64",
        "de_policy": target_config["de_policy"], "undefined_pearson_policy": target_config["undefined_pearson_policy"],
        "outcome_source": "Existing additive validation scoring_inputs.npz; no raw H5AD access",
        "test_outcomes_permitted": False, "hyperparameter_tuning": False,
    }
    return configuration, pairs, cells, gene_ids, gene_names, predicted, target_summary


def evaluate_saved(scorer, target_file, pairs, genes, names, predicted, expected_additive):
    with np.load(target_file, allow_pickle=False) as saved:
        require(np.array_equal(saved["pert_cat"], pairs), "Target condition order changed")
        require(np.array_equal(saved["gene_ids"], genes) and np.array_equal(saved["gene_names"], names), "Target gene axes changed")
        truth, indices = saved["truth"].copy(), saved["de_indices"].copy()
        require(np.array_equal(saved["truth_de"], np.take_along_axis(truth, indices, axis=1)), "Target DE slice changed")
        additive = {key: saved[key].copy() for key in ("pert_cat", "pred", "truth", "pred_de", "truth_de")}
    inputs = make_scorer_inputs(pairs, predicted, truth, indices)
    scores, by_pair, flags, warnings = score_inputs(scorer, inputs)
    additive_scores, additive_pairs, _, _ = score_inputs(scorer, additive)
    require(additive_scores == expected_additive, "Reused targets do not reproduce additive scores")
    return inputs, indices, scores, by_pair, flags, warnings, additive_scores, additive_pairs


def independent_metric_check(inputs, scores):
    """Check the actual GEARS scorer using direct NumPy arithmetic."""
    values = {key: [] for key in METRICS}
    for pred, truth, pred_de, truth_de in zip(inputs["pred"], inputs["truth"], inputs["pred_de"], inputs["truth_de"]):
        for suffix, a, b in (("", pred, truth), ("_de", pred_de, truth_de)):
            values["mse" + suffix].append(float(np.mean((a - b) ** 2)))
            ac, bc = a - a.mean(), b - b.mean()
            denominator = np.linalg.norm(ac) * np.linalg.norm(bc)
            values["pearson" + suffix].append(float(ac @ bc / denominator) if denominator > 0 else 0.0)
    independent = {key: float(np.mean(v)) for key, v in values.items()}
    require(all(np.isclose(independent[key], scores[key], rtol=1e-12, atol=1e-14) for key in METRICS), "Independent metric check failed")
    return {"status": "passed", "method": "Direct NumPy squared error and centered-vector correlation",
            "metrics": independent, "rtol": 1e-12, "atol": 1e-14}


def publish(stage, config, cells, genes, names, target_dir, result):
    inputs, indices, scores, by_pair, flags, warnings, additive_scores, additive_pairs = result
    np.savez_compressed(stage / "scoring_inputs.npz", **inputs, de_indices=indices, gene_ids=genes, gene_names=names)
    counts = {pair: sum(r["canonical_condition"] == pair for r in cells) for pair in inputs["pert_cat"]}
    rows = [{"model": config["model"], "split_seed": config["split_seed"], "partition": "val",
             "canonical_condition": str(pair), "target_cells": counts[pair], **by_pair[pair], **flags[pair],
             **{"additive_" + key: additive_pairs[pair][key] for key in METRICS},
             **{"difference_from_additive_" + key: by_pair[pair][key] - additive_pairs[pair][key] for key in METRICS}}
            for pair in inputs["pert_cat"]]
    write_csv(stage / "per_condition_metrics.csv", rows)
    for filename in ("scoring_genes.csv", "evaluation_cells.csv.gz"):
        shutil.copyfile(target_dir / filename, stage / filename)
    (stage / "independent_metrics_check.json").write_bytes(json_bytes(independent_metric_check(inputs, scores)))
    summary = {
        "status": "validation_evaluated", "model": config["model"], "partition": "val",
        "split_seed": config["split_seed"], "metrics": scores, "additive_metrics": additive_scores,
        "difference_from_additive": {k: scores[k] - additive_scores[k] for k in METRICS},
        "evaluated_pairs": len(inputs["pert_cat"]), "evaluated_cells": len(cells),
        "expression_genes": len(genes), "de_genes_per_pair": 20,
        "undefined_to_zero_counts": {key: sum(f[key] for f in flags.values()) for key in next(iter(flags.values()))},
        "scorer_warning_counts": warnings, "raw_h5ad_opened_for_scoring": False,
        "new_validation_expression_rows_decoded": 0, "new_ranking_keys_decoded": 0,
        "test_scores_computed": False,
        "cross_split_exposure": "Reuse previously scored validation panels; some pairs are test in other splits. No adaptive tuning.",
        "neural_comparison_status": "No matched trained GEARS/Set Transformer predictions available locally",
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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    splits = args.split_dir.resolve()
    prediction_dir = (args.prediction_dir or OUTPUT / f"seed_{args.split_seed}").resolve()
    target_dir = (args.target_dir or TARGETS / f"seed_{args.split_seed}/val").resolve()
    output = (args.output_dir or EVALUATIONS / f"seed_{args.split_seed}/val").resolve()
    require(output.is_relative_to(ROOT / "outputs"), "Keep evaluation outputs inside Codex Outputs/outputs")
    for directory in (splits, prediction_dir, target_dir):
        require(not output.is_relative_to(directory) and not directory.is_relative_to(output), "Separate evaluation from input bundles")
    require(not output.exists() if args.command == "run" else output.is_dir(),
            "Run requires a new output directory; verify requires an existing bundle")
    context = prepare_evaluation(splits, prediction_dir, target_dir, args.split_seed)
    config, pairs, cells, genes, names, predicted, target_summary = context
    scorer = load_scorer()
    if args.command == "verify":
        require(json.loads((output / "evaluation_config.json").read_text()) == config, "Evaluation configuration changed")
        summary = verify_artifacts(output)
        result = evaluate_saved(scorer, target_dir / "scoring_inputs.npz", pairs, genes, names, predicted, target_summary["metrics"])
        require(summary["metrics"] == result[2], "Regenerated metrics differ")
        with np.load(output / "scoring_inputs.npz", allow_pickle=False) as saved:
            for key, values in {**result[0], "de_indices": result[1], "gene_ids": genes, "gene_names": names}.items():
                require(np.array_equal(saved[key], values), f"Regenerated scoring input differs: {key}")
        report = {"status": "passed", "identical_score_and_input_regeneration": True,
                  "split_seed": args.split_seed, "metrics": result[2]}
        (output / "verification_report.json").write_bytes(json_bytes(report))
        print(json.dumps(report, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="linear-score-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        (stage / "evaluation_config.json").write_bytes(json_bytes(config))
        result = evaluate_saved(scorer, target_dir / "scoring_inputs.npz", pairs, genes, names, predicted, target_summary["metrics"])
        summary = publish(stage, config, cells, genes, names, target_dir, result)
        stage.rename(output)
    print(json.dumps({"split_seed": args.split_seed, "metrics": summary["metrics"],
                      "difference_from_additive": summary["difference_from_additive"]}, indent=2))


if __name__ == "__main__":
    main()
