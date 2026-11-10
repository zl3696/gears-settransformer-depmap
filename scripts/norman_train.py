"""Train one planned Norman run on the frozen condition splits.

Reads the run from configs/norman_runs.csv and the training protocol from
configs/norman_training.json. Cells use the saved control assignments, graphs
are fitted once per split and shared by all models, and the checkpoint is
selected on validation top-20 DE MSE. The test set is only evaluated with
--test. Outputs must be written outside the source checkout.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from gears import GEARS, PertData
from gears.utils import GeneSimNetwork, file_sha256, get_similarity_network

PROTOCOL = REPOSITORY / "configs/norman_training.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_sha256(*tensors):
    result = hashlib.sha256()
    for tensor in tensors:
        result.update(tensor.cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def planned_run(run_plan, run_id):
    with run_plan.open(newline="") as stream:
        runs = {row["run_id"]: row for row in csv.DictReader(stream)}
    require(run_id in runs, f"Unknown run_id: {run_id}")
    return runs[run_id]


def check_protocol(protocol, split_dir, run):
    frozen = json.loads((split_dir / "freeze.json").read_text())
    require(frozen["bundle_sha256"] == protocol["split_bundle_sha256"], "Split snapshot differs from the training protocol")
    require(file_sha256(split_dir / "config.json") == protocol["split_configuration_sha256"], "Split config.json differs from the training protocol")
    require(run["shared_configuration_sha256"] == protocol["split_configuration_sha256"], "Planned run uses another split configuration")
    require(int(run["training_seed"]) == protocol["training_seed"], "Planned run uses another training seed")
    require(run["model"] in protocol["models"], f"Model not implemented yet: {run['model']}")


def shared_graphs(pert_data, protocol, split_dir, split_seed, graph_path):
    """Fit the coexpression and GO graphs once per split; every model reuses the saved tensors."""
    if graph_path.exists():
        graphs = torch.load(graph_path)
        for name in ("coexpress", "go"):
            require(tensor_sha256(graphs[name]["edge_index"], graphs[name]["edge_weight"]) == graphs[name]["sha256"],
                    f"Saved {name} graph is corrupted: {graph_path}")
        return graphs

    metadata = json.loads((split_dir / f"seed_{split_seed}/graph_metadata.json").read_text())
    fit_conditions = [c for c in pert_data.set2conditions["train"] if "ctrl" in c]
    fit_cells = int(pert_data.adata.obs["condition"].isin(fit_conditions).sum())
    require(fit_cells == metadata["coexpression"]["fit_cells"], "Coexpression fit population differs from graph_metadata.json")

    hyper = protocol["graphs"]
    # the split name only keys the coexpression cache file
    cache_split = "frozen_" + protocol["split_bundle_sha256"][:12]
    common = dict(adata=pert_data.adata, threshold=hyper["coexpress_threshold"],
                  data_path=pert_data.data_path, data_name=pert_data.dataset_name,
                  split=cache_split, seed=split_seed, train_gene_set_size=None,
                  set2conditions=pert_data.set2conditions)
    coexpress = GeneSimNetwork(get_similarity_network("co-express", k=hyper["num_similar_genes_co_express_graph"], **common),
                               pert_data.gene_names.values.tolist(), node_map=pert_data.node_map)
    go = GeneSimNetwork(get_similarity_network("go", k=hyper["num_similar_genes_go_graph"], pert_list=pert_data.pert_names,
                                               default_pert_graph=hyper["default_pert_graph"], **common),
                        pert_data.pert_names.tolist(), node_map=pert_data.node_map_pert)

    graphs = {"fit_cells": fit_cells}
    for name, network in (("coexpress", coexpress), ("go", go)):
        graphs[name] = {"edge_index": network.edge_index, "edge_weight": network.edge_weight,
                        "sha256": tensor_sha256(network.edge_index, network.edge_weight)}
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, graph_path)
    return graphs


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, capture_output=True, text=True)
    return result.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="run_id from configs/norman_runs.csv")
    parser.add_argument("--data-path", type=Path, required=True, help="GEARS data directory (downloads and caches)")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory that planned output paths are relative to")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test", action="store_true", help="Also evaluate the selected checkpoint on the test set")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    require(not output_root.is_relative_to(REPOSITORY), "Outputs must be outside the source checkout")
    require(not args.data_path.resolve().is_relative_to(REPOSITORY), "Data must be outside the source checkout")
    protocol = json.loads(PROTOCOL.read_text())
    split_dir = REPOSITORY / protocol["split_dir"]
    run_plan = REPOSITORY / protocol["run_plan"]
    run = planned_run(run_plan, args.run_id)
    check_protocol(protocol, split_dir, run)
    split_seed = int(run["split_seed"])
    run_dir = (output_root / run["result_path"]).parent
    require(not run_dir.exists(), f"Run directory already exists: {run_dir}")

    pert_data = PertData(str(args.data_path))
    pert_data.load(data_name="norman", frozen_split_dir=str(split_dir))
    pert_data.prepare_split(split="custom", seed=split_seed)
    pert_data.get_dataloader(batch_size=protocol["training"]["batch_size"],
                             test_batch_size=protocol["training"]["test_batch_size"])
    graph_path = output_root / "outputs/norman-benchmark/graphs" / f"split{split_seed}.pt"
    graphs = shared_graphs(pert_data, protocol, split_dir, split_seed, graph_path)

    model = GEARS(pert_data, device=args.device)
    graph_hyper = {k: v for k, v in protocol["graphs"].items() if k != "default_pert_graph"}
    model.model_initialize(**protocol["model"], **graph_hyper,
                           pert_agg=protocol["models"][run["model"]]["pert_agg"],
                           seed=protocol["training_seed"],
                           G_coexpress=graphs["coexpress"]["edge_index"],
                           G_coexpress_weight=graphs["coexpress"]["edge_weight"],
                           G_go=graphs["go"]["edge_index"],
                           G_go_weight=graphs["go"]["edge_weight"])
    training = protocol["training"]
    model.train(epochs=training["epochs"], lr=training["lr"], weight_decay=training["weight_decay"],
                selection_metric=protocol["checkpoint_selection"]["metric"], run_test=args.test)

    require(model.best_epoch is not None, "No epoch produced a finite validation metric")
    run_dir.mkdir(parents=True)
    model.save_model(str(run_dir))
    selected = model.val_history[model.best_epoch - 1]
    results = {"run_id": args.run_id, "model": run["model"], "split_seed": split_seed,
               "training_seed": protocol["training_seed"], "git_commit": git_commit(),
               "pert_agg": model.config["pert_agg"],
               "coexpression_batching": model.config["coexpression_batching"],
               "run_plan_sha256": file_sha256(run_plan),
               "protocol_sha256": file_sha256(PROTOCOL), "split_bundle_sha256": protocol["split_bundle_sha256"],
               "graph_sha256": {name: graphs[name]["sha256"] for name in ("coexpress", "go")},
               "coexpression_fit_cells": graphs["fit_cells"],
               "selection_metric": protocol["checkpoint_selection"]["metric"], "best_epoch": model.best_epoch,
               "selected_validation": selected, "validation_history": model.val_history,
               "test_evaluated": args.test, "checkpoint_files": ["config.pkl", "model.pt"]}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("run_id", "best_epoch", "selected_validation")}, indent=2))


if __name__ == "__main__":
    main()
