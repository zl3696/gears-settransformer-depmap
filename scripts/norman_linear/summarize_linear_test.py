"""Aggregate verified final-test linear scores and matched additive differences."""

from pathlib import Path
import argparse
import json
import tempfile

import numpy as np

from linear_baseline import ROOT, SPLITS, digest, json_bytes, read_csv, require, verify_artifacts
from evaluate_linear import EVALUATIONS
from evaluate_linear_test import METRICS, write_csv
from summarize_validation import metric_statistics
from additive_baseline import check_frozen_bundle


def check_compatible(configs):
    require(len(configs) == 5 and [c["split_seed"] for c in configs] == list(range(1, 6)), "Exactly five ordered test configurations required")
    require(all(c["partition"] == "test" and c["metrics"] == list(METRICS) and
                c["test_outcomes_permitted"] is True and c["refit_performed"] is False and
                c["hyperparameter_tuning"] is False for c in configs), "Expected fixed authorized final-test configurations")
    keys = ("model", "partition", "metrics", "source_sha256", "split_bundle_sha256", "runner_sha256",
            "shared_linear_evaluation_helper_sha256", "shared_evaluation_helper_sha256", "test_selection_helper_sha256",
            "original_scorer_sha256", "team_scorer_sha256", "de_policy", "undefined_pearson_policy", "condition_weighting",
            "mean_dtype", "test_outcomes_permitted", "refit_performed", "hyperparameter_tuning", "protocol_scope")
    shared = {k: configs[0][k] for k in keys}
    require(all({k: c[k] for k in keys} == shared for c in configs), "Across-split final scoring contracts differ")
    return shared


def collect(evaluation_root, split_dir):
    frozen = check_frozen_bundle(split_dir)
    rows, additive_rows, deltas, pair_rows, configs, hashes = [], [], [], [], [], {}
    pairs_accessed, cells_accessed, previously_exposed = set(), set(), set()
    for seed in range(1, 6):
        directory = evaluation_root / f"seed_{seed}/test"
        summary = verify_artifacts(directory)
        config = json.loads((directory / "evaluation_config.json").read_text())
        verified = json.loads((directory / "verification_report.json").read_text())
        require(summary["status"] == "final_test_evaluated" and summary["partition"] == "test" and
                config["split_seed"] == summary["split_seed"] == seed, "Wrong final evaluation state/split")
        require(config["split_bundle_sha256"] == frozen["bundle_sha256"], "Frozen split bundle changed")
        require(summary["configuration_sha256"] == digest(directory / "evaluation_config.json"), "Final configuration hash changed")
        require(verified["status"] == "passed" and verified["identical_score_and_input_regeneration"] and
                verified["partition"] == "test" and verified["split_seed"] == seed and
                verified["metrics"] == summary["metrics"], "Final scores are not verified")
        manifest_path = split_dir / f"seed_{seed}/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        require(config["manifest_sha256"] == digest(manifest_path) and
                config["conditions"] == sorted(manifest["pair_conditions"]["test"]), "Test partition changed")
        per_pair = read_csv(directory / "per_condition_metrics.csv")
        require(len(per_pair) == 26 and {r["canonical_condition"] for r in per_pair} == set(config["conditions"]) and
                all(r["partition"] == "test" and int(r["split_seed"]) == seed for r in per_pair), "Invalid scored pair population")
        cells = read_csv(directory / "evaluation_cells.csv.gz")
        require(len(cells) == summary["evaluated_cells"] == manifest["target_cell_counts"]["test"] and
                all(r["partition"] == "test" for r in cells), "Invalid test cell population")
        for key in METRICS:
            for column, expected in ((key, summary["metrics"][key]), ("additive_" + key, summary["additive_metrics"][key]),
                                      ("difference_from_additive_" + key, summary["difference_from_additive"][key])):
                require(np.isclose(np.mean([float(r[column]) for r in per_pair]), expected, rtol=1e-12, atol=1e-14), "Pair/aggregate metric mismatch")
        rows.append({"split_seed": seed, "partition": "test", "evaluated_pairs": 26, "evaluated_cells": len(cells),
                     **summary["metrics"], **summary["undefined_to_zero_counts"]})
        additive_rows.append({"split_seed": seed, **summary["additive_metrics"]})
        deltas.append({"split_seed": seed, **summary["difference_from_additive"]})
        pair_rows.extend(per_pair)
        pairs_accessed.update(config["conditions"])
        cells_accessed.update(r["cell_id"] for r in cells)
        previously_exposed.update(config["prior_validation_exposed_test_conditions"])
        configs.append(config)
        for filename in ("summary.json", "evaluation_config.json", "verification_report.json"):
            hashes[f"seed_{seed}/test/{filename}"] = digest(directory / filename)
    shared = check_compatible(configs)
    summary = {
        "status": "all_five_final_test_splits_evaluated", "model": shared["model"], "partition": "test",
        "split_seeds": list(range(1, 6)), "statistics": metric_statistics(rows),
        "additive_statistics": metric_statistics(additive_rows), "difference_from_additive_statistics": metric_statistics(deltas),
        "pair_evaluations": len(pair_rows), "unique_test_pairs": len(pairs_accessed), "unique_test_target_cells": len(cells_accessed),
        "test_conditions_accessed": sorted(pairs_accessed), "previously_validation_exposed_test_pairs": len(previously_exposed),
        "prior_validation_exposure_by_test_split": {str(c["split_seed"]): len(c["prior_validation_exposed_test_conditions"]) for c in configs},
        "paired_difference_definition": "Linear minus additive in matching test splits; negative MSE/positive Pearson favors linear",
        "interpretation": "Sample variance/SD across overlapping test panels; not independent biological replication or training-seed variability",
        "test_scores_computed": True, "refit_performed": False, "hyperparameter_tuning": False,
        "neural_comparison_status": "Matched trained neural predictions unavailable",
        "shared_configuration": shared, "input_summary_sha256": hashes,
        "summary_script_sha256": digest(Path(__file__)), "statistics_helper_sha256": digest(Path(__file__).parents[1] / "norman-additive/summarize_validation.py"),
    }
    return rows, deltas, pair_rows, summary


def tables(rows, deltas, pairs, summary):
    return {"split_metrics.csv": rows, "metric_summary.csv": summary["statistics"],
            "additive_metric_summary.csv": summary["additive_statistics"], "paired_differences.csv": deltas,
            "paired_difference_summary.csv": summary["difference_from_additive_statistics"], "per_condition_metrics.csv": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATIONS)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluation_root, splits = args.evaluation_root.resolve(), args.split_dir.resolve()
    output = (args.output_dir or evaluation_root / "test_summary").resolve()
    require(output.is_relative_to(ROOT / "outputs") and not output.is_relative_to(splits) and not splits.is_relative_to(output), "Invalid final-summary output path")
    require(output != evaluation_root and not evaluation_root.is_relative_to(output), "Summary cannot replace evaluation root")
    require(not output.exists() if args.command == "run" else output.is_dir(), "Run requires new output; verify requires existing output")
    rows, deltas, pairs, summary = collect(evaluation_root, splits)
    if args.command == "verify":
        saved = verify_artifacts(output)
        require({k: saved[k] for k in summary} == summary, "Final summary regeneration differs")
        for filename, values in tables(rows, deltas, pairs, summary).items():
            require(read_csv(output / filename) == [{k: str(v) for k, v in r.items()} for r in values], f"Regenerated table differs: {filename}")
        report = {"status": "passed", "identical_summary_regeneration": True, "partition": "test"}
        (output / "verification_report.json").write_bytes(json_bytes(report))
        print(json.dumps(report))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="linear-test-summary-", dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        for filename, values in tables(rows, deltas, pairs, summary).items():
            write_csv(stage / filename, values)
        summary["artifact_sha256"] = {p.name: digest(p) for p in stage.iterdir()}
        (stage / "summary.json").write_bytes(json_bytes(summary))
        stage.rename(output)
    print(json.dumps({k: summary[k] for k in ("statistics", "additive_statistics", "difference_from_additive_statistics", "unique_test_pairs")}, indent=2))


if __name__ == "__main__":
    main()
