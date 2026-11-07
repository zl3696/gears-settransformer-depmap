"""Independently audit saved fits without reading held-out expression or rankings."""

from pathlib import Path
import json

import numpy as np

from linear_baseline import ROOT, OUTPUT, digest, json_bytes, require, verify_artifacts


def main():
    additive = ROOT / "outputs/norman-additive/training_means.npz"
    with np.load(additive, allow_pickle=False) as saved:
        reference = {key: saved[key].copy() for key in saved.files}
    for seed in range(1, 6):
        directory = OUTPUT / f"seed_{seed}"
        verify_artifacts(directory)
        with np.load(directory / "model.npz", allow_pickle=False) as saved:
            model = {key: saved[key].copy() for key in saved.files}
        with np.load(directory / "training_means.npz", allow_pickle=False) as saved:
            means = {key: saved[key].copy() for key in saved.files}
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            predictions = {key: saved[key].copy() for key in saved.files}
        require(np.array_equal(means["gene_ids"], reference["gene_ids"]), "Shared mean gene axes differ")
        lookup = {str(c): i for i, c in enumerate(means["conditions"])}
        indices = [lookup[str(c)] for c in reference["conditions"]]
        shared_error = float(np.max(np.abs(means["mean_expression"][indices] - reference["mean_expression"])))
        require(shared_error == 0 and np.array_equal(means["cell_counts"][indices], reference["cell_counts"]),
                "Single/control means differ from independently checked additive means")
        y = means["mean_expression"].T
        centered = y - y.mean(0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
        k = model["gene_embeddings"].shape[1]
        v = eigenvectors[:, -k:]
        stored_v = model["pca_components"].T
        projection_error = float(np.max(np.abs(v @ v.T - stored_v @ stored_v.T)))
        require(projection_error < 1e-8, "Independent covariance PCA subspace differs")
        np.testing.assert_allclose(eigenvalues[-k:][::-1], model["pca_singular_values"] ** 2, rtol=1e-10, atol=1e-10)
        g, b, offset = model["gene_embeddings"], model["training_embeddings"], y.mean(1)
        w = np.linalg.inv(g.T @ g + .1 * np.eye(k)) @ g.T @ (y - offset[:, None]) @ b.T @ np.linalg.inv(b @ b.T + .1 * np.eye(k))
        coefficient_error = float(np.max(np.abs(w - model["coefficients"])))
        np.testing.assert_allclose(w, model["coefficients"], rtol=1e-10, atol=1e-12)
        # Independently assemble every pair from target indices and the literal formula.
        gene_lookup = {str(c): i for i, c in enumerate(model["gene_names"])}
        pair_b = np.stack([g[gene_lookup[a]] + g[gene_lookup[c]] for a, c in
                           (str(pair).split("+") for pair in predictions["conditions"])], axis=1)
        literal_predictions = (offset[:, None] + g @ w @ pair_b).T
        prediction_error = float(np.max(np.abs(literal_predictions - predictions["predicted_mean"])))
        np.testing.assert_allclose(literal_predictions, predictions["predicted_mean"], rtol=1e-10, atol=1e-12)
        report = {
            "status": "passed", "split_seed": seed, "shared_single_control_conditions_checked": len(indices),
            "single_control_mean_max_absolute_difference": shared_error,
            "independent_pca_method": "Symmetric eigendecomposition of centered training-condition covariance",
            "pca_projection_max_absolute_difference": projection_error,
            "literal_two_sided_solver_coefficient_max_absolute_difference": coefficient_error,
            "literal_pair_prediction_max_absolute_difference": prediction_error,
            "raw_h5ad_opened": False, "held_out_outcomes_read": False, "rankings_read": False,
            "audit_script_sha256": digest(Path(__file__)),
            "model_sha256": digest(directory / "model.npz"),
            "training_means_sha256": digest(directory / "training_means.npz"),
            "additive_training_means_sha256": digest(additive),
        }
        (directory / "independent_fit_check.json").write_bytes(json_bytes(report))
        print(json.dumps({"split_seed": seed, "status": "passed", "prediction_max_error": prediction_error}), flush=True)


if __name__ == "__main__":
    main()
