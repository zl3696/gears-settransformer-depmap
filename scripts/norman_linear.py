"""Verify the curated linear results or reproduce the preserved implementation.

Original numerical sources remain byte-identical in scripts/norman_linear.
An external runtime mirror preserves their historical relative paths; no raw
expression data, caches, or new run outputs are written into this checkout.
"""

from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
SNAPSHOT = REPOSITORY / "scripts/norman_linear"
HELPERS = REPOSITORY / "scripts/norman_additive"
RESULTS = REPOSITORY / "results/norman/linear"
METRICS = ("mse", "mse_de", "pearson", "pearson_de")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def additive_export():
    spec = importlib.util.spec_from_file_location("norman_additive_export", REPOSITORY / "scripts/norman_additive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_results():
    """Check hashes, fitted math, populations, scores, and paired statistics."""
    archive = additive_export()
    baseline_check = archive.verify_results()
    provenance = read_json(RESULTS / "provenance.json")
    for relative, expected in provenance["artifact_sha256"].items():
        require(digest(archive.checked_path(RESULTS, relative)) == expected, f"Result changed: {relative}")
    for relative, expected in provenance["repository_source_sha256"].items():
        require(digest(archive.checked_path(REPOSITORY, relative)) == expected, f"Source changed: {relative}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SNAPSHOT))
    linear = importlib.import_module("linear_baseline")
    frozen = linear.check_frozen_bundle(REPOSITORY / "splits/norman")
    require(frozen["bundle_sha256"] == provenance["split_bundle_sha256"], "Frozen split bundle differs")
    condition_rows = {r["canonical_condition"]: r for r in archive.read_csv(REPOSITORY / "splits/norman/conditions.csv")}
    genes = archive.read_csv(REPOSITORY / "splits/norman/expression_genes.csv")
    gene_ids = np.array([r["gene_id"] for r in genes])
    gene_names = np.array([r["gene_name"] for r in genes])
    with np.load(archive.RESULTS / "predictions/training_means.npz", allow_pickle=False) as saved:
        shared_names, shared_means = saved["conditions"].copy(), saved["mean_expression"].copy()
    partitions = {p: {"rows": [], "scores": [], "differences": [], "pairs": set()} for p in ("val", "test")}
    for seed in range(1, 6):
        fit = RESULTS / f"fits/seed_{seed}"
        fit_summary, fit_config = read_json(fit / "summary.json"), read_json(fit / "fit_config.json")
        archive.check_archived_artifacts(fit, fit_summary)
        manifest_path = REPOSITORY / f"splits/norman/seed_{seed}/manifest.json"
        manifest = read_json(manifest_path)
        require(fit_config["runner_sha256"] == digest(SNAPSHOT / "linear_baseline.py") and
                fit_config["source_sha256"] == provenance["source_sha256"] and
                fit_config["split_bundle_sha256"] == frozen["bundle_sha256"] and
                fit_config["manifest_sha256"] == digest(manifest_path), "Fit source/configuration differs")
        require(fit_config["pca_dimensions"] == 10 and fit_config["gene_penalty"] == fit_config["perturbation_penalty"] == .1,
                "Fixed hyperparameters differ")
        require(read_json(fit / "verification_report.json")["prediction_and_fit_regeneration"] and
                read_json(fit / "independent_fit_check.json")["status"] == "passed", "Fit audits not verified")
        with np.load(fit / "model.npz", allow_pickle=False) as z:
            model = {k: z[k].copy() for k in z.files}
        with np.load(fit / "predictions.npz", allow_pickle=False) as z:
            predicted = {k: z[k].copy() for k in z.files}
        with np.load(fit / "training_means.npz", allow_pickle=False) as z:
            profiles = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(profiles["gene_ids"], gene_ids)
        np.testing.assert_array_equal(predicted["gene_ids"], gene_ids)
        np.testing.assert_array_equal(model["gene_names"], gene_names)
        require(profiles["conditions"].tolist() == sorted(manifest["canonical_conditions"]["train"]), "Training conditions differ")
        require(profiles["mean_expression"].shape == (179, 5045) and predicted["predicted_mean"].shape == (128, 5045), "Array shape differs")
        np.testing.assert_array_equal(profiles["cell_counts"], [int(condition_rows[c]["cell_count"]) for c in profiles["conditions"]])
        lookup = {c: i for i, c in enumerate(profiles["conditions"])}
        np.testing.assert_array_equal(profiles["mean_expression"][[lookup[c] for c in shared_names]], shared_means)
        regenerated = linear.fit_model(profiles["mean_expression"], profiles["conditions"], gene_names)
        for key, value in regenerated.items():
            np.testing.assert_allclose(model[key], value, rtol=1e-10, atol=1e-12)
        expected = linear.predict(model, predicted["conditions"], gene_names)
        np.testing.assert_allclose(predicted["predicted_mean"], expected, rtol=1e-10, atol=1e-12)
        membership = {c: p for p in ("train", "val", "test") for c in manifest["pair_conditions"][p]}
        require(predicted["partitions"].tolist() == [membership[c] for c in predicted["conditions"]], "Prediction partition mapping differs")
        for partition in partitions:
            directory = RESULTS / f"evaluation/seed_{seed}/{partition}"
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "evaluation_config.json")
            archive.check_archived_artifacts(directory, summary)
            require(summary["partition"] == config["partition"] == partition and summary["split_seed"] == seed,
                    "Scoring identity differs")
            require(config["conditions"] == sorted(manifest["pair_conditions"][partition]) and config["metrics"] == list(METRICS), "Scoring population/metrics differ")
            require(config["predictions_sha256"] == digest(fit / "predictions.npz") and
                    config["fit_configuration_sha256"] == digest(fit / "fit_config.json"), "Scoring used another fitted model")
            runner = "evaluate_linear.py" if partition == "val" else "evaluate_linear_test.py"
            require(config["runner_sha256"] == digest(SNAPSHOT / runner) and
                    config["original_scorer_sha256"] == config["team_scorer_sha256"] == digest(REPOSITORY / "gears/inference.py"), "Evaluation source differs")
            require(summary["configuration_sha256"] == digest(directory / "evaluation_config.json"), "Configuration hash differs")
            report = read_json(directory / "verification_report.json")
            require(report["status"] == "passed" and report["identical_score_and_input_regeneration"] and report["metrics"] == summary["metrics"], "Score regeneration unverified")
            require(read_json(directory / "independent_metrics_check.json")["status"] == "passed", "Independent metric check unverified")
            baseline_dir = archive.RESULTS / f"evaluation/seed_{seed}/{partition}"
            baseline = {r["canonical_condition"]: r for r in archive.read_csv(baseline_dir / "per_condition_metrics.csv")}
            require(digest(directory / "scoring_genes.csv") == digest(baseline_dir / "scoring_genes.csv"), "DE genes differ from additive")
            rows = archive.read_csv(directory / "per_condition_metrics.csv")
            require(len(rows) == 26 and {r["canonical_condition"] for r in rows} == set(config["conditions"]), "Per-pair population differs")
            require(sum(int(r["target_cells"]) for r in rows) == manifest["target_cell_counts"][partition], "Evaluation cell counts differ")
            for row in rows:
                require(row["partition"] == partition and int(row["split_seed"]) == seed, "Per-pair partition differs")
                for key in METRICS:
                    original = float(baseline[row["canonical_condition"]][key])
                    require(float(row["additive_" + key]) == original, "Matched additive score differs")
                    require(np.isclose(float(row["difference_from_additive_" + key]), float(row[key]) - original,
                                       rtol=1e-12, atol=1e-14), "Paired difference differs")
            for key in METRICS:
                require(np.isclose(np.mean([float(r[key]) for r in rows]), summary["metrics"][key], rtol=1e-14, atol=1e-15), "Pair-score average differs")
            for flag, metric in (("pearson_undefined_to_zero", "pearson"), ("pearson_de_undefined_to_zero", "pearson_de")):
                affected = [r for r in rows if r[flag] == "True"]
                require(len(affected) == summary["undefined_to_zero_counts"][flag] and all(float(r[metric]) == 0 for r in affected), "Undefined-correlation policy differs")
            partitions[partition]["rows"].extend(rows)
            partitions[partition]["scores"].append(summary["metrics"])
            partitions[partition]["differences"].append(summary["difference_from_additive"])
            partitions[partition]["pairs"].update(config["conditions"])
    for partition, state in partitions.items():
        directory = RESULTS / "evaluation" / ("validation_summary" if partition == "val" else "test_summary")
        summary = read_json(directory / "summary.json")
        archive.check_archived_artifacts(directory, summary)
        require(read_json(directory / "verification_report.json")["identical_summary_regeneration"], "Aggregate regeneration unverified")
        require(archive.read_csv(directory / "per_condition_metrics.csv") == state["rows"], "Merged pair table differs")
        require(len(state["pairs"]) == 88 and len(state["rows"]) == 130, "Aggregate coverage differs")
        for collection, field, filename in ((state["scores"], "statistics", "metric_summary.csv"),
                                             (state["differences"], "difference_from_additive_statistics", "paired_difference_summary.csv")):
            table = archive.read_csv(directory / filename)
            for key in METRICS:
                values = np.array([r[key] for r in collection])
                row = next(r for r in summary[field] if r["metric"] == key)
                csv_row = next(r for r in table if r["metric"] == key)
                for name, value in (("mean", values.mean()), ("sample_variance", values.var(ddof=1)),
                                    ("sample_standard_deviation", values.std(ddof=1))):
                    require(row[name] == float(csv_row[name]) == value, "Across-split statistics differ")
    return {"status": "passed", "artifact_hashes_verified": len(provenance["artifact_sha256"]),
            "source_hashes_verified": len(provenance["repository_source_sha256"]), "fitted_models_checked": 5,
            "prediction_profiles_checked": 640, "evaluation_panels_checked": 10, "metrics_and_paired_statistics_checked": True,
            "raw_expression_outcomes_read": False, "additive_export_verified": baseline_check["status"] == "passed"}


def configure_runtime(runtime, source):
    runtime = runtime.resolve()
    require(not runtime.is_relative_to(REPOSITORY) and not REPOSITORY.is_relative_to(runtime), "Use an external runtime root")
    (runtime / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(runtime / "tmp")
    sys.dont_write_bytecode = True
    mirrors = {}
    for origin, name in ((SNAPSHOT, "norman-linear"), (HELPERS, "norman-additive")):
        mirror = runtime / "scripts" / name
        mirror.mkdir(parents=True, exist_ok=True)
        for original in origin.glob("*.py"):
            target = mirror / original.name
            if target.exists():
                require(digest(target) == digest(original), "Runtime source mirror changed; use a fresh runtime root")
            else:
                shutil.copy2(original, target)
        mirrors[name] = mirror
    sys.path.insert(0, str(mirrors["norman-additive"]))
    sys.path.insert(0, str(mirrors["norman-linear"]))
    builder = importlib.import_module("additive_baseline")
    builder.ROOT, builder.SPLITS = runtime, REPOSITORY / "splits/norman"
    builder.SOURCE, builder.OUTPUT = source, runtime / "outputs/norman-additive"
    validation = importlib.import_module("evaluate_validation")
    scorer = REPOSITORY / "gears/inference.py"
    require(digest(scorer) == read_json(RESULTS / "provenance.json")["gears_scorer_sha256"], "GEARS scorer changed")
    validation.REFERENCE_SCORER = validation.TEAM_SCORER = scorer
    original_loader = validation.load_scorer

    def portable_loader(reference=scorer, team=scorer):
        return original_loader(reference, team)

    validation.load_scorer = portable_loader
    linear = importlib.import_module("linear_baseline")
    linear.ROOT, linear.SPLITS, linear.SOURCE = runtime, builder.SPLITS, source
    linear.OUTPUT = runtime / "outputs/norman-linear"
    return linear, mirrors["norman-linear"]


def reproduce(runtime, source):
    """Regenerate additive targets, five fits, all ten panels, and both summaries."""
    require(not (runtime / "outputs/norman-linear").exists(), "Reproduction requires a fresh runtime root")
    stages = []

    def run(entry, command, seed=None, action="run"):
        arguments = [sys.executable, str(entry), command, "--runtime-root", str(runtime), "--source", str(source)]
        if seed is not None:
            arguments += ["--split-seed", str(seed)]
        if action == "verify":
            arguments += ["--action", "verify"]
        result = subprocess.run(arguments, text=True, capture_output=True)
        require(result.returncode == 0, f"Reproduction failed at {entry.name}/{command}/{seed}: {result.stderr}\n{result.stdout}")
        stages.append({"entry": entry.name, "command": command, "split_seed": seed, "action": action})
        print(json.dumps({"completed": stages[-1]}), flush=True)

    additive = REPOSITORY / "scripts/norman_additive.py"
    for action in ("run", "verify"):
        run(additive, "predict", action=action)
    for partition in ("validation", "test"):
        for seed in range(1, 6):
            for action in ("run", "verify"):
                run(additive, partition, seed, action)
        for action in ("run", "verify"):
            run(additive, partition + "-summary", action=action)
    entry = Path(__file__).resolve()
    for seed in range(1, 6):
        for action in ("run", "verify"):
            run(entry, "fit", seed, action)
    run(entry, "audit")
    for partition in ("validation", "test"):
        for seed in range(1, 6):
            for action in ("run", "verify"):
                run(entry, partition, seed, action)
        for action in ("run", "verify"):
            run(entry, partition + "-summary", action=action)
    array_checks, score_checks = 0, 0
    for seed in range(1, 6):
        for filename in ("model.npz", "predictions.npz", "training_means.npz"):
            with np.load(RESULTS / f"fits/seed_{seed}" / filename, allow_pickle=False) as expected, \
                 np.load(runtime / f"outputs/norman-linear/seed_{seed}" / filename, allow_pickle=False) as actual:
                require(set(expected.files) == set(actual.files), "Reproduced array schema differs")
                for key in expected.files:
                    if expected[key].dtype.kind in "fc":
                        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-12)
                    else:
                        np.testing.assert_array_equal(actual[key], expected[key])
                    array_checks += 1
        for partition in ("val", "test"):
            expected = read_json(RESULTS / f"evaluation/seed_{seed}/{partition}/summary.json")
            actual = read_json(runtime / f"outputs/norman-linear-evaluation/seed_{seed}/{partition}/summary.json")
            for key in METRICS:
                require(np.isclose(actual["metrics"][key], expected["metrics"][key], rtol=1e-10, atol=1e-12), "Reproduced metric differs")
                score_checks += 1
    for name in ("validation_summary", "test_summary"):
        expected = read_json(RESULTS / "evaluation" / name / "summary.json")
        actual = read_json(runtime / "outputs/norman-linear-evaluation" / name / "summary.json")
        for field in ("statistics", "difference_from_additive_statistics"):
            for a, b in zip(actual[field], expected[field]):
                require(a["metric"] == b["metric"], "Reproduced metric order differs")
                for key in ("mean", "sample_variance", "sample_standard_deviation"):
                    require(np.isclose(a[key], b[key], rtol=1e-10, atol=1e-12), "Reproduced statistic differs")
    report = {"status": "passed", "fresh_source_data_reproduction": True, "source_sha256": digest(source),
              "fitted_models": 5, "array_fields_checked": array_checks, "metric_values_checked": score_checks,
              "evaluation_panels": 10, "aggregate_summaries": 2, "completed_stages": stages,
              "numerical_rtol": 1e-10, "numerical_atol": 1e-12,
              "portable_entry_sha256": digest(entry), "split_bundle_sha256": read_json(RESULTS / "provenance.json")["split_bundle_sha256"]}
    (runtime / "reproduction_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "completed_stages"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "tests", "fit", "audit", "validation", "test", "validation-summary", "test-summary", "reproduce"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--split-seed", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--action", choices=("run", "verify"), default="run")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_results(), indent=2))
        return
    require(args.runtime_root is not None, "Supply an external --runtime-root")
    runtime = args.runtime_root.resolve()
    require(not runtime.is_relative_to(REPOSITORY) and not REPOSITORY.is_relative_to(runtime), "Use a runtime root outside this checkout")
    source = (args.source or runtime / "data/norman/perturb_processed.h5ad").resolve()
    if args.command in ("fit", "reproduce"):
        require(source.is_file(), "Supply --source pointing to the frozen H5AD")
    if args.command == "reproduce":
        reproduce(runtime, source)
        return
    linear, mirror = configure_runtime(runtime, source)
    if args.command == "tests":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(mirror), pattern="test_*.py"))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    names = {"fit": "linear_baseline", "audit": "audit_linear", "validation": "evaluate_linear",
             "test": "evaluate_linear_test", "validation-summary": "summarize_linear", "test-summary": "summarize_linear_test"}
    module = importlib.import_module(names[args.command])
    if args.command == "audit":
        module.main()
        return
    action = ("fit" if args.action == "run" else "verify") if args.command == "fit" else args.action
    sys.argv = [str(__file__), action]
    if args.command in ("fit", "validation", "test"):
        sys.argv += ["--split-seed", str(args.split_seed)]
    if args.command == "fit":
        sys.argv += ["--source", str(source)]
    module.main()


if __name__ == "__main__":
    main()
