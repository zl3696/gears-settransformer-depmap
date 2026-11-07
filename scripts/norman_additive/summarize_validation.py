"""Combine five verified validation runs without reading new source outcomes."""

from pathlib import Path
import argparse
import json
import tempfile

import numpy as np

from additive_baseline import ROOT, SPLITS, check_frozen_bundle, digest, json_bytes, read_csv, require
from evaluate_validation import METRICS, write_csv

EVALUATIONS = ROOT / "outputs/norman-additive-evaluation"
SHARED_CONFIG_KEYS = (
    "model", "partition", "metrics", "source_sha256", "split_bundle_sha256",
    "predictions_sha256", "prediction_summary_sha256", "runner_sha256",
    "aggregation_helper_sha256", "original_scorer_sha256", "team_scorer_sha256",
    "scorer", "mean_dtype", "condition_weighting", "de_policy", "undefined_pearson_policy",
)


def check_compatible(configurations):
    require(len(configurations) == 5, "Exactly five validation configurations are required")
    require([c["split_seed"] for c in configurations] == [1, 2, 3, 4, 5], "Expected split seeds 1–5 in order")
    require(all(c["partition"] == "val" and c["metrics"] == list(METRICS) for c in configurations),
            "Expected the same four validation metrics")
    shared = {key: configurations[0][key] for key in SHARED_CONFIG_KEYS}
    require(all({key: c[key] for key in SHARED_CONFIG_KEYS} == shared for c in configurations),
            "Scoring inputs, code, or metric configuration differs across splits")
    return shared


def metric_statistics(split_rows):
    require(len(split_rows) == 5 and {int(r["split_seed"]) for r in split_rows} == set(range(1, 6)),
            "Exactly five unique split scores are required")
    rows = []
    for metric in METRICS:
        values = np.array([float(row[metric]) for row in split_rows], dtype=np.float64)
        require(np.isfinite(values).all(), f"Non-finite split score: {metric}")
        rows.append({"metric": metric, "n_splits": 5, "mean": float(values.mean()),
                     "sample_variance": float(values.var(ddof=1)),
                     "sample_standard_deviation": float(values.std(ddof=1))})
    return rows


def collect(evaluation_root, split_directory):
    frozen = check_frozen_bundle(split_directory)
    split_rows, all_pair_rows, configs, input_hashes = [], [], [], {}
    cells_accessed, pairs_accessed = set(), set()
    for seed in range(1, 6):
        directory = evaluation_root / f"seed_{seed}/val"
        summary = json.loads((directory / "summary.json").read_text())
        config = json.loads((directory / "evaluation_config.json").read_text())
        verification = json.loads((directory / "verification_report.json").read_text())
        require(verification["status"] == "passed" and verification["identical_score_and_input_regeneration"],
                f"Split {seed} regeneration is not verified")
        require(summary["status"] == "validation_evaluated" and summary["split_seed"] == config["split_seed"] == seed,
                "Evaluation summary/configuration differs")
        require(config["split_bundle_sha256"] == frozen["bundle_sha256"], "Split source differs")
        require(summary["configuration_sha256"] == digest(directory / "evaluation_config.json"), "Configuration hash differs")
        for name, expected in summary["artifact_sha256"].items():
            require(digest(directory / name) == expected, f"Split {seed} artifact changed: {name}")
        pair_rows = read_csv(directory / "per_condition_metrics.csv")
        require(len(pair_rows) == len({row["canonical_condition"] for row in pair_rows}) == summary["evaluated_pairs"] == 26,
                "Expected 26 unique validation pairs per split")
        require({r["canonical_condition"] for r in pair_rows} == set(config["conditions"]), "Scored conditions differ")
        for metric in METRICS:
            average = np.mean([float(row[metric]) for row in pair_rows])
            require(np.isclose(average, summary["metrics"][metric], rtol=1e-14, atol=1e-15),
                    f"Pair-level/aggregate score mismatch: split {seed}, {metric}")
        require(verification["metrics"] == summary["metrics"], "Verified scores differ from summary")
        split_rows.append({"split_seed": seed, "partition": "val", "evaluated_pairs": 26,
                           "evaluated_cells": summary["evaluated_cells"], **summary["metrics"],
                           **summary["undefined_to_zero_counts"]})
        all_pair_rows.extend(pair_rows)
        pairs_accessed.update(config["conditions"])
        cells_accessed.update(row["cell_id"] for row in read_csv(directory / "evaluation_cells.csv.gz"))
        configs.append(config)
        for name in ("summary.json", "evaluation_config.json", "verification_report.json"):
            input_hashes[f"seed_{seed}/val/{name}"] = digest(directory / name)
    shared = check_compatible(configs)
    coverage = {}
    for seed in range(1, 6):
        manifest = json.loads((split_directory / f"seed_{seed}/manifest.json").read_text())
        coverage[str(seed)] = len(set(manifest["pair_conditions"]["test"]) & pairs_accessed)
    summary = {
        "status": "all_five_validation_splits_evaluated", "model": "additive", "partition": "val",
        "split_seeds": [1, 2, 3, 4, 5], "statistics": metric_statistics(split_rows),
        "pair_evaluations": len(all_pair_rows), "unique_validation_pairs": len(pairs_accessed),
        "unique_validation_target_cells": len(cells_accessed),
        "validation_conditions_accessed": sorted(pairs_accessed),
        "test_conditions_with_outcomes_exposed_via_other_validation_splits": coverage,
        "final_test_scores_computed": False,
        "interpretation": "Sample variance/SD across overlapping validation panels; not independent biological replicates or training-seed variability",
        "shared_configuration": shared, "input_summary_sha256": input_hashes,
        "summary_script_sha256": digest(Path(__file__)),
    }
    return split_rows, all_pair_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATIONS)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path, default=EVALUATIONS / "validation_summary")
    args = parser.parse_args()
    evaluation_root, split_dir, output = (p.resolve() for p in (args.evaluation_root, args.split_dir, args.output_dir))
    require(output.is_relative_to(ROOT / "outputs"), "Keep summaries inside Codex Outputs/outputs")
    split_rows, pair_rows, summary = collect(evaluation_root, split_dir)
    if args.command == "verify":
        saved = json.loads((output / "summary.json").read_text())
        require({key: saved[key] for key in summary} == summary, "Regenerated validation summary differs")
        for name, expected in saved["artifact_sha256"].items():
            require(digest(output / name) == expected, f"Summary artifact changed: {name}")
        print(json.dumps({"status": "passed", "five_split_summary_regeneration_identical": True}, indent=2))
        return
    require(not output.exists(), "Summary exists; use verify or a new versioned output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="validation-summary-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        write_csv(stage / "split_metrics.csv", split_rows)
        write_csv(stage / "metric_summary.csv", summary["statistics"])
        write_csv(stage / "per_condition_metrics.csv", pair_rows)
        summary["artifact_sha256"] = {path.name: digest(path) for path in stage.iterdir()}
        (stage / "summary.json").write_bytes(json_bytes(summary))
        stage.rename(output)
    print(json.dumps({"output": str(output), "statistics": summary["statistics"],
                      "pair_evaluations": summary["pair_evaluations"],
                      "unique_validation_pairs": summary["unique_validation_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
