"""Aggregate five verified additive final-test runs without new outcome reads."""

from pathlib import Path
import argparse
import json
import tempfile

import numpy as np

from additive_baseline import ROOT, SPLITS, check_frozen_bundle, digest, json_bytes, read_csv, require
from evaluate_test import EVALUATIONS
from evaluate_validation import METRICS, write_csv
from summarize_validation import SHARED_CONFIG_KEYS as VALIDATION_KEYS, metric_statistics

SHARED_CONFIG_KEYS = (*VALIDATION_KEYS, "validation_adapter_sha256",
                      "prior_validation_summary_sha256", "test_outcomes_permitted", "protocol_scope")


def check_compatible(configs):
    require(len(configs) == 5 and [c["split_seed"] for c in configs] == [1, 2, 3, 4, 5],
            "Exactly five ordered test configurations are required")
    require(all(c["partition"] == "test" and c["metrics"] == list(METRICS) and
                c["test_outcomes_permitted"] is True for c in configs), "Expected authorized final-test configurations")
    shared = {key: configs[0][key] for key in SHARED_CONFIG_KEYS}
    require(all({key: c[key] for key in SHARED_CONFIG_KEYS} == shared for c in configs),
            "Scoring inputs, code, or metric configuration differs across splits")
    return shared


def collect(evaluation_root, split_dir):
    frozen = check_frozen_bundle(split_dir)
    split_rows, all_pair_rows, configs, input_hashes = [], [], [], {}
    cells_accessed, pairs_accessed = set(), set()
    for seed in range(1, 6):
        directory = evaluation_root / f"seed_{seed}/test"
        summary = json.loads((directory / "summary.json").read_text())
        config = json.loads((directory / "evaluation_config.json").read_text())
        verification = json.loads((directory / "verification_report.json").read_text())
        require(verification["status"] == "passed" and verification["identical_score_and_input_regeneration"] and
                verification["partition"] == "test" and verification["split_seed"] == seed,
                f"Split {seed} test regeneration is not verified")
        require(summary["status"] == "final_test_evaluated" and summary["partition"] == "test" and
                summary["split_seed"] == config["split_seed"] == seed, "Evaluation summary/configuration differs")
        require(config["split_bundle_sha256"] == frozen["bundle_sha256"], "Split source differs")
        manifest_path = split_dir / f"seed_{seed}/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        require(config["manifest_sha256"] == digest(manifest_path) and
                set(config["conditions"]) == set(manifest["pair_conditions"]["test"]), "Test partition differs")
        require(summary["configuration_sha256"] == digest(directory / "evaluation_config.json"), "Configuration hash differs")
        for name, expected in summary["artifact_sha256"].items():
            require(digest(directory / name) == expected, f"Split {seed} artifact changed: {name}")
        pair_rows = read_csv(directory / "per_condition_metrics.csv")
        require(len(pair_rows) == len({row["canonical_condition"] for row in pair_rows}) == summary["evaluated_pairs"] == 26,
                "Expected 26 unique test pairs per split")
        require({r["canonical_condition"] for r in pair_rows} == set(config["conditions"]) and
                all(r["partition"] == "test" and int(r["split_seed"]) == seed for r in pair_rows), "Scored conditions differ")
        cells = read_csv(directory / "evaluation_cells.csv.gz")
        require(len(cells) == summary["evaluated_cells"] == manifest["target_cell_counts"]["test"] and
                all(r["partition"] == "test" for r in cells), "Scored test cells differ")
        for metric in METRICS:
            average = np.mean([float(row[metric]) for row in pair_rows])
            require(np.isclose(average, summary["metrics"][metric], rtol=1e-14, atol=1e-15),
                    f"Pair-level/aggregate mismatch: split {seed}, {metric}")
        require(verification["metrics"] == summary["metrics"], "Verified metrics differ")
        split_rows.append({"split_seed": seed, "partition": "test", "evaluated_pairs": 26,
                           "evaluated_cells": len(cells), **summary["metrics"], **summary["undefined_to_zero_counts"]})
        all_pair_rows.extend(pair_rows)
        pairs_accessed.update(config["conditions"])
        cells_accessed.update(row["cell_id"] for row in cells)
        configs.append(config)
        for name in ("summary.json", "evaluation_config.json", "verification_report.json"):
            input_hashes[f"seed_{seed}/test/{name}"] = digest(directory / name)
    shared = check_compatible(configs)
    prior_path = evaluation_root / "validation_summary/summary.json"
    require(digest(prior_path) == shared["prior_validation_summary_sha256"], "Prior validation record changed")
    prior = json.loads(prior_path.read_text())
    previously_exposed = set(prior["validation_conditions_accessed"])
    for config in configs:
        require(config["prior_validation_exposed_test_conditions"] == sorted(set(config["conditions"]) & previously_exposed),
                "Prior-exposure record differs")
    summary = {
        "status": "all_five_final_test_splits_evaluated", "model": "additive", "partition": "test",
        "split_seeds": [1, 2, 3, 4, 5], "statistics": metric_statistics(split_rows),
        "pair_evaluations": len(all_pair_rows), "unique_test_pairs": len(pairs_accessed),
        "unique_test_target_cells": len(cells_accessed), "test_conditions_accessed": sorted(pairs_accessed),
        "previously_validation_exposed_test_pairs": len(pairs_accessed & previously_exposed),
        "test_pairs_not_previously_scored_in_validation": len(pairs_accessed - previously_exposed),
        "prior_validation_exposure_by_test_split": {str(c["split_seed"]): len(c["prior_validation_exposed_test_conditions"]) for c in configs},
        "unique_pairs_scored_across_validation_and_test": len(pairs_accessed | previously_exposed),
        "final_test_scores_computed": True,
        "interpretation": "Sample variance/SD across overlapping test panels; not independent biological replicates or training-seed variability",
        "comparison_status": "Additive only; no matched trained neural accuracy results",
        "shared_configuration": shared, "input_summary_sha256": input_hashes,
        "summary_script_sha256": digest(Path(__file__)),
        "statistics_helper_sha256": digest(Path(__file__).with_name("summarize_validation.py")),
    }
    return split_rows, all_pair_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATIONS)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path, default=EVALUATIONS / "test_summary")
    args = parser.parse_args()
    evaluation_root, split_dir, output = (p.resolve() for p in (args.evaluation_root, args.split_dir, args.output_dir))
    require(output.is_relative_to(ROOT / "outputs"), "Keep summaries inside Codex Outputs/outputs")
    split_rows, pair_rows, summary = collect(evaluation_root, split_dir)
    if args.command == "verify":
        saved = json.loads((output / "summary.json").read_text())
        require({key: saved[key] for key in summary} == summary, "Regenerated test summary differs")
        for name, expected in saved["artifact_sha256"].items():
            require(digest(output / name) == expected, f"Summary artifact changed: {name}")
        print(json.dumps({"status": "passed", "five_split_summary_regeneration_identical": True}, indent=2))
        return
    require(not output.exists(), "Summary exists; use verify or a new versioned output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-summary-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        write_csv(stage / "split_metrics.csv", split_rows)
        write_csv(stage / "metric_summary.csv", summary["statistics"])
        write_csv(stage / "per_condition_metrics.csv", pair_rows)
        summary["artifact_sha256"] = {path.name: digest(path) for path in stage.iterdir()}
        (stage / "summary.json").write_bytes(json_bytes(summary))
        stage.rename(output)
    print(json.dumps({"output": str(output), "statistics": summary["statistics"],
                      "pair_evaluations": summary["pair_evaluations"], "unique_test_pairs": summary["unique_test_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
