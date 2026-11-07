"""Verify the committed additive results or run the preserved implementation.

Run outputs and temporary files belong in an explicit directory outside this
checkout. The files in scripts/norman_additive are byte-identical to the evaluated
implementation; this entry point supplies portable paths without changing math.
"""

from pathlib import Path
import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import sys
import unittest

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / "results/norman/additive"
SNAPSHOT = REPOSITORY / "scripts/norman_additive"
METRICS = ("mse", "mse_de", "pearson", "pearson_de")
OMITTED = {"fit_cells.csv.gz", "evaluation_cells.csv.gz", "scoring_inputs.npz"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def checked_path(root, relative):
    path = (root / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()),
            f"Artifact path escapes its bundle: {relative}")
    return path


def check_archived_artifacts(directory, summary):
    """Check retained historical hashes; allow only explicitly omitted runtime files."""
    for name, expected in summary["artifact_sha256"].items():
        path = checked_path(directory, name)
        if not path.exists():
            require(name in OMITTED, f"Missing archived artifact: {path}")
        else:
            require(digest(path) == expected, f"Archived artifact changed: {path}")


def verify_results():
    """Verify the compact export without reading any expression outcomes."""
    provenance = read_json(RESULTS / "provenance.json")
    for name, expected in provenance["artifact_sha256"].items():
        require(digest(checked_path(RESULTS, name)) == expected, f"Result changed: {name}")
    for name, expected in provenance["repository_source_sha256"].items():
        require(digest(checked_path(REPOSITORY, name)) == expected, f"Source changed: {name}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SNAPSHOT))
    from additive_baseline import check_frozen_bundle, read_csv as read_any_csv
    frozen = check_frozen_bundle(REPOSITORY / "splits/norman")
    require(frozen["bundle_sha256"] == provenance["split_bundle_sha256"], "Split bundle differs")
    fit_dir = RESULTS / "predictions"
    fit_summary = read_json(fit_dir / "summary.json")
    check_archived_artifacts(fit_dir, fit_summary)
    require(read_json(fit_dir / "verification_report.json")["identical_prediction_regeneration"],
            "Original prediction regeneration was not verified")
    require(fit_summary["source_sha256"] == provenance["source_sha256"], "Expression source differs")
    require(fit_summary["runner_sha256"] == digest(SNAPSHOT / "additive_baseline.py"), "Builder differs")
    with np.load(fit_dir / "predictions.npz", allow_pickle=False) as p, \
            np.load(fit_dir / "training_means.npz", allow_pickle=False) as f:
        genes = p["gene_ids"].copy()
        conditions = p["conditions"].tolist()
        require(p["additive_mean"].shape == (128, 5045), "Prediction shape differs")
        np.testing.assert_array_equal(genes, f["gene_ids"])
        means = dict(zip(f["conditions"], f["mean_expression"]))
        expected = np.stack([means[a] + means[b] - means["ctrl"] for a, b in (c.split("+") for c in conditions)])
        np.testing.assert_array_equal(p["additive_mean"], expected)
        np.testing.assert_array_equal(p["control_mean"], means["ctrl"])
        require(int(f["cell_counts"].sum()) == 54931, "Training population differs")
    gene_rows = read_csv(REPOSITORY / "splits/norman/expression_genes.csv")
    np.testing.assert_array_equal(genes, [r["gene_id"] for r in gene_rows])
    memberships = read_csv(fit_dir / "split_membership.csv")
    require(len(memberships) == 640, "Prediction membership table differs")
    cells_by_condition = {}
    for row in read_any_csv(REPOSITORY / "splits/norman/cells.csv.gz"):
        cells_by_condition.setdefault(row["canonical_condition"], set()).add(row["cell_id"])
    coverage, summaries = {}, {}
    for partition, aggregate_name in (("val", "validation_summary"), ("test", "test_summary")):
        all_rows, split_metrics, unique_pairs = [], [], set()
        for seed in range(1, 6):
            directory = RESULTS / f"evaluation/seed_{seed}/{partition}"
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "evaluation_config.json")
            report = read_json(directory / "verification_report.json")
            manifest_path = REPOSITORY / f"splits/norman/seed_{seed}/manifest.json"
            manifest = read_json(manifest_path)
            pairs = set(manifest["pair_conditions"][partition])
            require(set(config["conditions"]) == pairs and len(pairs) == 26, "Scored pairs differ")
            require(config["partition"] == summary["partition"] == partition and
                    config["split_seed"] == summary["split_seed"] == seed, "Scoring identity differs")
            require(config["manifest_sha256"] == digest(manifest_path) and
                    config["source_sha256"] == provenance["source_sha256"] and
                    config["split_bundle_sha256"] == frozen["bundle_sha256"], "Scoring source differs")
            require(config["predictions_sha256"] == digest(fit_dir / "predictions.npz") and
                    config["prediction_summary_sha256"] == digest(fit_dir / "summary.json"), "Prediction provenance differs")
            require(config["metrics"] == list(METRICS), "Scoring metrics differ")
            runner = "evaluate_validation.py" if partition == "val" else "evaluate_test.py"
            require(config["runner_sha256"] == digest(SNAPSHOT / runner), "Evaluation runner differs")
            require(config["original_scorer_sha256"] == config["team_scorer_sha256"] ==
                    digest(REPOSITORY / "gears/inference.py"), "GEARS scorer differs")
            require(summary["configuration_sha256"] == digest(directory / "evaluation_config.json"), "Configuration hash differs")
            check_archived_artifacts(directory, summary)
            require(report["status"] == "passed" and report["identical_score_and_input_regeneration"] and
                    report["metrics"] == summary["metrics"], "Historical regeneration report differs")
            rows = read_csv(directory / "per_condition_metrics.csv")
            require(len(rows) == len({r["canonical_condition"] for r in rows}) == 26 and
                    {r["canonical_condition"] for r in rows} == pairs, "Per-pair population differs")
            require(sum(int(r["target_cells"]) for r in rows) == manifest["target_cell_counts"][partition], "Target counts differ")
            require({r["canonical_condition"] for r in memberships if int(r["split_seed"]) == seed and
                    r["partition"] == partition} == pairs, "Prediction membership differs")
            metrics = {key: float(np.mean([float(r[key]) for r in rows])) for key in METRICS}
            for key in METRICS:
                require(np.isfinite([float(r[key]) for r in rows]).all(), "Non-finite score")
                require(np.isclose(metrics[key], summary["metrics"][key], rtol=1e-14, atol=1e-15), "Per-pair mean differs")
            for flag, metric in (("pearson_undefined_to_zero", "pearson"), ("pearson_de_undefined_to_zero", "pearson_de")):
                affected = [r for r in rows if r[flag] == "True"]
                require(len(affected) == summary["undefined_to_zero_counts"][flag] and
                        all(float(r[metric]) == 0 for r in affected), "Undefined-correlation policy differs")
            scoring_genes = read_csv(directory / "scoring_genes.csv")
            require(len(scoring_genes) == 520, "Scoring gene count differs")
            for pair in pairs:
                selected = [r for r in scoring_genes if r["canonical_condition"] == pair]
                require(len(selected) == len({r["gene_id"] for r in selected}) == 20 and
                        {int(r["released_rank"]) for r in selected} == set(range(1, 21)), "DE membership differs")
                require(all(genes[int(r["expression_index"])] == r["gene_id"] for r in selected), "DE axis differs")
            split_metrics.append(summary["metrics"])
            all_rows.extend(rows)
            unique_pairs.update(pairs)
        combined_dir = RESULTS / "evaluation" / aggregate_name
        aggregate = read_json(combined_dir / "summary.json")
        check_archived_artifacts(combined_dir, aggregate)
        require(read_json(combined_dir / "verification_report.json")["five_split_summary_regeneration_identical"], "Aggregate regeneration unverified")
        require(read_csv(combined_dir / "per_condition_metrics.csv") == all_rows, "Combined per-pair table differs")
        split_rows = read_csv(combined_dir / "split_metrics.csv")
        require([int(r["split_seed"]) for r in split_rows] == [1, 2, 3, 4, 5], "Split order differs")
        statistic_rows = read_csv(combined_dir / "metric_summary.csv")
        for key in METRICS:
            values = np.array([r[key] for r in split_metrics])
            np.testing.assert_array_equal(values, [float(r[key]) for r in split_rows])
            row = next(r for r in statistic_rows if r["metric"] == key)
            saved = next(r for r in aggregate["statistics"] if r["metric"] == key)
            for field, expected in (("mean", values.mean()), ("sample_variance", values.var(ddof=1)),
                                    ("sample_standard_deviation", values.std(ddof=1))):
                require(float(row[field]) == saved[field] == expected, f"Split statistics differ: {key}/{field}")
        coverage[partition] = unique_pairs
        require(len(unique_pairs) == aggregate[f'unique_{"validation" if partition == "val" else "test"}_pairs'] == 88,
                "Unique pair coverage differs")
        require(len(set.union(*(cells_by_condition[p] for p in unique_pairs))) ==
                aggregate[f'unique_{"validation" if partition == "val" else "test"}_target_cells'], "Unique target-cell coverage differs")
        summaries[partition] = aggregate
    require(len(coverage["test"] & coverage["val"]) == summaries["test"]["previously_validation_exposed_test_pairs"] == 56,
            "Prior validation exposure differs")
    require(len(coverage["test"] | coverage["val"]) == 120, "Total outcome coverage differs")
    return {"status": "passed", "artifact_hashes_verified": len(provenance["artifact_sha256"]),
            "additive_formula_regeneration_identical": True, "partitions": ["val", "test"],
            "splits_per_partition": 5, "pair_evaluations_per_partition": 130,
            "metrics_and_statistics_reaggregated": True, "source_expression_outcomes_read": False}


def configure_runtime(runtime, source):
    """Bind preserved modules to external runtime paths and this checkout's scorer."""
    runtime = runtime.resolve()
    require(not runtime.is_relative_to(REPOSITORY) and not REPOSITORY.is_relative_to(runtime),
            "Use a runtime directory outside this checkout")
    (runtime / "tmp").mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SNAPSHOT))
    builder = importlib.import_module("additive_baseline")
    builder.ROOT, builder.SPLITS = runtime, REPOSITORY / "splits/norman"
    builder.SOURCE, builder.OUTPUT = source, runtime / "outputs/norman-additive"
    validation = importlib.import_module("evaluate_validation")
    scorer = REPOSITORY / "gears/inference.py"
    expected = read_json(RESULTS / "provenance.json")["gears_scorer_sha256"]
    require(digest(scorer) == expected, "Version the protocol before changing the GEARS scorer")
    validation.REFERENCE_SCORER = validation.TEAM_SCORER = scorer
    original_load = validation.load_scorer

    def portable_load(reference=scorer, team=scorer):
        return original_load(reference, team)

    validation.load_scorer = portable_load
    return builder, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "tests", "predict", "validation", "test", "validation-summary", "test-summary"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--split-seed", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--action", choices=("run", "verify"), default="run")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_results(), indent=2))
        return
    require(args.runtime_root is not None, "Supply --runtime-root outside the checkout")
    if args.command in ("predict", "validation", "test"):
        require(args.source is not None and args.source.is_file(), "Supply --source pointing to the frozen H5AD")
    source = (args.source or args.runtime_root / "data/norman/perturb_processed.h5ad").resolve()
    builder, validation = configure_runtime(args.runtime_root, source)
    if args.command == "tests":
        suite = unittest.defaultTestLoader.discover(str(SNAPSHOT), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    modules = {"predict": builder, "validation": validation}
    for command, name in (("test", "evaluate_test"), ("validation-summary", "summarize_validation"), ("test-summary", "summarize_test")):
        modules[command] = importlib.import_module(name)
    command = ("predict" if args.action == "run" else "verify") if args.command == "predict" else args.action
    sys.argv = [str(__file__), command]
    if args.command in ("validation", "test"):
        sys.argv.extend(["--split-seed", str(args.split_seed)])
    if args.action == "verify":
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            modules[args.command].main()
        report = json.loads(capture.getvalue())
        outputs = args.runtime_root.resolve() / "outputs"
        if args.command == "predict":
            directory = outputs / "norman-additive"
        elif args.command in ("validation", "test"):
            partition = "val" if args.command == "validation" else "test"
            directory = outputs / f"norman-additive-evaluation/seed_{args.split_seed}/{partition}"
        else:
            name = "validation_summary" if args.command == "validation-summary" else "test_summary"
            directory = outputs / "norman-additive-evaluation" / name
        (directory / "verification_report.json").write_bytes(builder.json_bytes(report))
        print(json.dumps(report, indent=2))
    else:
        modules[args.command].main()


if __name__ == "__main__":
    main()
