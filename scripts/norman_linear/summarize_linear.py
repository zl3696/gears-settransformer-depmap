"""Summarize five verified linear validation runs and matched additive differences."""

from pathlib import Path
import argparse
import json
import tempfile

import numpy as np

from linear_baseline import ROOT, SPLITS, digest, json_bytes, read_csv, require, verify_artifacts
from evaluate_linear import EVALUATIONS, METRICS, write_csv
from summarize_validation import metric_statistics
from additive_baseline import check_frozen_bundle


def collect(evaluation_root, split_dir):
    frozen = check_frozen_bundle(split_dir)
    rows, differences, pair_rows, hashes, configs = [], [], [], {}, []
    pairs_accessed = set()
    for seed in range(1, 6):
        directory = evaluation_root / f"seed_{seed}/val"
        summary = verify_artifacts(directory)
        config = json.loads((directory / "evaluation_config.json").read_text())
        verified = json.loads((directory / "verification_report.json").read_text())
        require(summary["status"] == "validation_evaluated" and config["split_seed"] == summary["split_seed"] == seed,
                "Wrong evaluation state/split")
        require(config["partition"] == "val" and config["metrics"] == list(METRICS), "Wrong partition/metrics")
        require(config["split_bundle_sha256"] == frozen["bundle_sha256"], "Changed frozen split bundle")
        require(summary["configuration_sha256"] == digest(directory / "evaluation_config.json"), "Configuration hash changed")
        require(verified["status"] == "passed" and verified["identical_score_and_input_regeneration"] and verified["metrics"] == summary["metrics"], "Evaluation is not verified")
        per_pair = read_csv(directory / "per_condition_metrics.csv")
        require(len(per_pair) == 26 and {r["canonical_condition"] for r in per_pair} == set(config["conditions"]), "Pair population changed")
        for key in METRICS:
            require(np.isclose(np.mean([float(r[key]) for r in per_pair]), summary["metrics"][key], rtol=1e-14, atol=1e-15), "Pair/aggregate metric mismatch")
            require(np.isclose(np.mean([float(r["difference_from_additive_" + key]) for r in per_pair]), summary["difference_from_additive"][key], rtol=1e-12, atol=1e-14), "Paired difference mismatch")
        rows.append({"split_seed": seed, "partition": "val", "evaluated_pairs": 26,
                     "evaluated_cells": summary["evaluated_cells"], **summary["metrics"], **summary["undefined_to_zero_counts"]})
        differences.append({"split_seed": seed, **summary["difference_from_additive"]})
        pair_rows.extend(per_pair)
        pairs_accessed.update(config["conditions"])
        configs.append(config)
        for filename in ("summary.json", "evaluation_config.json", "verification_report.json"):
            hashes[f"seed_{seed}/val/{filename}"] = digest(directory / filename)
    shared_keys = ("model", "partition", "metrics", "source_sha256", "split_bundle_sha256", "runner_sha256",
                   "shared_evaluation_helper_sha256", "original_scorer_sha256", "team_scorer_sha256",
                   "condition_weighting", "mean_dtype", "de_policy", "undefined_pearson_policy")
    shared = {k: configs[0][k] for k in shared_keys}
    require(all({k: c[k] for k in shared_keys} == shared for c in configs), "Across-split scoring contract differs")
    summary = {
        "status": "all_five_validation_splits_evaluated", "model": shared["model"], "partition": "val",
        "split_seeds": list(range(1, 6)), "pair_evaluations": len(pair_rows), "unique_validation_pairs": len(pairs_accessed),
        "statistics": metric_statistics(rows), "difference_from_additive_statistics": metric_statistics(differences),
        "paired_difference_definition": "Linear minus additive within each matching split; negative MSE or positive Pearson favors linear",
        "interpretation": "Sample variance/SD of overlapping validation panels; not biological replication or training-seed variability",
        "shared_configuration": shared, "input_summary_sha256": hashes,
        "summary_script_sha256": digest(Path(__file__)), "statistics_helper_sha256": digest(Path(__file__).parents[1] / "norman-additive/summarize_validation.py"),
        "test_scores_computed": False, "neural_comparison_status": "Matched trained neural predictions not available",
    }
    return rows, differences, pair_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATIONS)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluation_root, splits = args.evaluation_root.resolve(), args.split_dir.resolve()
    output = (args.output_dir or evaluation_root / "validation_summary").resolve()
    require(output.is_relative_to(ROOT / "outputs") and not output.is_relative_to(splits), "Invalid summary output directory")
    require(output != evaluation_root and not evaluation_root.is_relative_to(output), "Summary cannot replace evaluation root")
    require(not output.exists() if args.command == "run" else output.is_dir(), "Run requires new output; verify requires existing output")
    rows, differences, pairs, summary = collect(evaluation_root, splits)
    if args.command == "verify":
        saved = verify_artifacts(output)
        require({k: saved[k] for k in summary} == summary, "Regenerated summary differs")
        for filename, values in (("split_metrics.csv", rows), ("paired_differences.csv", differences),
                                  ("per_condition_metrics.csv", pairs), ("metric_summary.csv", summary["statistics"]),
                                  ("paired_difference_summary.csv", summary["difference_from_additive_statistics"])):
            require(read_csv(output / filename) == [{k: str(v) for k, v in r.items()} for r in values], f"Regenerated table differs: {filename}")
        report = {"status": "passed", "identical_summary_regeneration": True}
        (output / "verification_report.json").write_bytes(json_bytes(report))
        print(json.dumps(report))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="linear-summary-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for filename, values in (("split_metrics.csv", rows), ("paired_differences.csv", differences),
                                  ("per_condition_metrics.csv", pairs), ("metric_summary.csv", summary["statistics"]),
                                  ("paired_difference_summary.csv", summary["difference_from_additive_statistics"])):
            write_csv(stage / filename, values)
        summary["artifact_sha256"] = {p.name: digest(p) for p in stage.iterdir()}
        (stage / "summary.json").write_bytes(json_bytes(summary))
        stage.rename(output)
    print(json.dumps({k: summary[k] for k in ("statistics", "difference_from_additive_statistics")}, indent=2))


if __name__ == "__main__":
    main()
