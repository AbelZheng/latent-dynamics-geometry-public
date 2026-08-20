#!/usr/bin/env python3
"""Deterministic static rank-two Gaussian latent recovery demonstration.

The script requires only NumPy and Pillow.  Data generation, preprocessing,
fitting/selection, scoring, artifact writing, and verification are deliberately
separate.  Running with no flags performs both generation and verification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MASTER_SEED = 20260819
DATE = "2026-08-19"
P = 8
K = 2
SPLIT_SIZES = {"train": 700, "validation": 350, "test": 500}
CONDITION_ORDER = (
    "matched_diagonal",
    "repeated_signal_eigenvalues",
    "near_zero_uniqueness",
    "correlated_residuals",
    "high_variance_nuisance",
    "outliers",
)
FA_FLOORS = (1e-5, 1e-3)
FA_RESTARTS = (0, 1, 2)
FA_MAX_ITER = 250
FA_TOL = 1e-8
RANDOM_PROJECTIONS = 7
TOL = {"symmetry": 1e-10, "psd": 1e-9, "poison": 1e-12, "determinism": 0.0}
ARTIFACT_NAMES = ("metrics.csv", "diagnostics.json", "summary.png")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "static-linear-latent-recovery"
METRIC_FIELDS = (
    "split", "condition", "preprocessing", "method", "target", "metric", "value"
)


def rng_for(*codes: int) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((MASTER_SEED,) + tuple(codes)))
    )


def orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(matrix, dtype=float))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q[:, : matrix.shape[1]] * signs[: matrix.shape[1]]


def base_basis() -> np.ndarray:
    raw = np.array(
        [
            [1.0, 0.2], [0.7, -0.5], [0.4, 1.0], [-0.6, 0.8],
            [0.9, 0.3], [-0.2, -0.9], [0.5, -0.4], [-0.7, -0.2],
        ],
        dtype=float,
    )
    return orthonormal_columns(raw)


def condition_spec(name: str) -> dict:
    q = base_basis()
    strengths = np.array([2.0, 1.25])
    uniqueness = np.array([0.12, 0.20, 0.32, 0.48, 0.70, 0.95, 1.25, 1.60])
    loading = q * strengths
    residual = np.diag(uniqueness)
    outlier_fraction = 0.0
    outlier_scale = 0.0
    diagonal_residual_applicable = True
    if name == "matched_diagonal":
        pass
    elif name == "repeated_signal_eigenvalues":
        loading = q * 1.65
        uniqueness = np.full(P, 0.30)
        residual = np.diag(uniqueness)
    elif name == "near_zero_uniqueness":
        uniqueness = np.array([1e-5, 2e-5, 5e-5, 1e-4, 0.02, 0.04, 0.08, 0.12])
        residual = np.diag(uniqueness)
    elif name == "correlated_residuals":
        rho = 0.42
        corr = rho ** np.abs(np.subtract.outer(np.arange(P), np.arange(P)))
        residual = np.sqrt(uniqueness)[:, None] * corr * np.sqrt(uniqueness)[None, :]
        diagonal_residual_applicable = False
    elif name == "high_variance_nuisance":
        uniqueness = uniqueness.copy()
        uniqueness[-1] = 7.5
        residual = np.diag(uniqueness)
    elif name == "outliers":
        outlier_fraction = 0.025
        outlier_scale = 9.0
        diagonal_residual_applicable = False
    else:
        raise KeyError(name)
    covariance = loading @ loading.T + residual
    return {
        "name": name,
        "mean": np.zeros(P),
        "loading": loading,
        "uniqueness": np.diag(residual).copy(),
        "residual_covariance": residual,
        "nominal_covariance": covariance,
        "outlier_fraction": outlier_fraction,
        "outlier_scale": outlier_scale,
        "diagonal_residual_applicable": diagonal_residual_applicable,
    }


def generate_split(spec: dict, split: str) -> dict:
    """Generate one split independently while retaining latent truth."""
    split_code = {"train": 1, "validation": 2, "test": 3}[split]
    condition_code = CONDITION_ORDER.index(spec["name"]) + 1
    n = SPLIT_SIZES[split]
    latent = rng_for(10, condition_code, split_code).normal(size=(n, K))
    residual = rng_for(11, condition_code, split_code).multivariate_normal(
        np.zeros(P), spec["residual_covariance"], size=n, check_valid="raise"
    )
    observed = spec["mean"] + latent @ spec["loading"].T + residual
    outlier_mask = np.zeros(n, dtype=bool)
    if spec["outlier_fraction"] > 0.0:
        count = max(1, int(round(n * spec["outlier_fraction"])))
        order = rng_for(12, condition_code, split_code).permutation(n)
        outlier_mask[order[:count]] = True
        directions = rng_for(13, condition_code, split_code).normal(size=(count, P))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        observed[outlier_mask] += spec["outlier_scale"] * directions
    return {"observed": observed, "latent": latent, "outlier_mask": outlier_mask}


def generate_condition(name: str) -> dict:
    spec = condition_spec(name)
    return {"spec": spec, **{split: generate_split(spec, split) for split in SPLIT_SIZES}}


def fit_standardizer(x: np.ndarray) -> dict:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {"mean": mean, "scale": scale}


def transform(x: np.ndarray, prep: dict) -> np.ndarray:
    return (x - prep["mean"]) / prep["scale"]


def inverse_transform(x: np.ndarray, prep: dict) -> np.ndarray:
    return prep["mean"] + x * prep["scale"]


def sample_covariance(centered: np.ndarray) -> np.ndarray:
    return centered.T @ centered / centered.shape[0]


def symmetric_eigh(matrix: np.ndarray):
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def fit_pca(x: np.ndarray) -> dict:
    mean = np.mean(x, axis=0)
    covariance = sample_covariance(x - mean)
    values, vectors = symmetric_eigh(covariance)
    return {"mean": mean, "basis": vectors[:, :K], "eigenvalues": values, "sample_covariance": covariance}


def fit_ppca(x: np.ndarray) -> dict:
    pca = fit_pca(x)
    sigma2 = max(float(np.mean(pca["eigenvalues"][K:])), 1e-8)
    signal = np.maximum(pca["eigenvalues"][:K] - sigma2, 0.0)
    loading = pca["basis"] * np.sqrt(signal)
    covariance = loading @ loading.T + sigma2 * np.eye(P)
    return {
        "mean": pca["mean"], "loading": loading, "noise_variance": sigma2,
        "covariance": covariance, "basis": pca["basis"],
    }


def gaussian_nll(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    centered = x - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        return float("inf")
    solved = np.linalg.solve(covariance, centered.T).T
    return float(0.5 * (P * math.log(2.0 * math.pi) + logdet + np.mean(np.sum(centered * solved, axis=1))))


def fa_initialization(covariance: np.ndarray, restart: int, floor: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = symmetric_eigh(covariance)
    baseline = max(float(np.mean(values[K:])), floor)
    signal = np.maximum(values[:K] - baseline, floor)
    loading = vectors[:, :K] * np.sqrt(signal)
    if restart > 0:
        jitter = rng_for(30, restart).normal(scale=0.08 * math.sqrt(max(values[0], floor)), size=(P, K))
        loading = loading + jitter
    uniqueness = np.maximum(np.diag(covariance - loading @ loading.T), floor)
    return loading, uniqueness


def fit_fa_em(x: np.ndarray, floor: float, restart: int) -> dict:
    """Bounded deterministic EM for a zero-mean latent Gaussian FA model."""
    mean = np.mean(x, axis=0)
    centered = x - mean
    covariance_sample = sample_covariance(centered)
    loading, uniqueness = fa_initialization(covariance_sample, restart, floor)
    history = []
    converged = False
    previous = float("inf")
    for iteration in range(1, FA_MAX_ITER + 1):
        covariance = loading @ loading.T + np.diag(uniqueness)
        inv_cov = np.linalg.inv(covariance)
        beta = loading.T @ inv_cov
        expected_zz = np.eye(K) - beta @ loading + beta @ covariance_sample @ beta.T
        cross_xz = covariance_sample @ beta.T
        loading_new = cross_xz @ np.linalg.inv(expected_zz)
        uniqueness_new = np.maximum(np.diag(covariance_sample - loading_new @ cross_xz.T), floor)
        covariance_new = loading_new @ loading_new.T + np.diag(uniqueness_new)
        current = gaussian_nll(x, mean, covariance_new)
        history.append(current)
        loading, uniqueness = loading_new, uniqueness_new
        if np.isfinite(previous) and abs(previous - current) <= FA_TOL * max(1.0, abs(previous)):
            converged = True
            break
        previous = current
    covariance = loading @ loading.T + np.diag(uniqueness)
    return {
        "mean": mean, "loading": loading, "uniqueness": uniqueness,
        "covariance": covariance, "iterations": iteration, "converged": converged,
        "train_nll": gaussian_nll(x, mean, covariance),
        "history": history,
        "floor": floor, "restart": restart,
    }


def select_fa(train: np.ndarray, validation: np.ndarray) -> tuple[dict, list[dict]]:
    candidates = []
    fitted = []
    for floor in FA_FLOORS:
        for restart in FA_RESTARTS:
            model = fit_fa_em(train, floor, restart)
            validation_nll = gaussian_nll(validation, model["mean"], model["covariance"])
            record = {
                "floor": floor, "restart": restart, "converged": bool(model["converged"]),
                "iterations": int(model["iterations"]), "train_nll": model["train_nll"],
                "validation_nll": validation_nll,
                "final_train_improvement": float(model["history"][0] - model["history"][-1]),
                "maximum_train_nll_increase": float(max(
                    [0.0] + [b - a for a, b in zip(model["history"][:-1], model["history"][1:])]
                )),
            }
            candidates.append(record)
            fitted.append(model)
    index = min(
        range(len(candidates)),
        key=lambda i: (candidates[i]["validation_nll"], candidates[i]["floor"], candidates[i]["restart"]),
    )
    selected = fitted[index]
    selected["validation_nll"] = candidates[index]["validation_nll"]
    return selected, candidates


def posterior_reconstruction(x: np.ndarray, mean: np.ndarray, loading: np.ndarray, residual: np.ndarray) -> np.ndarray:
    covariance = loading @ loading.T + residual
    beta = loading.T @ np.linalg.inv(covariance)
    scores = (x - mean) @ beta.T
    return mean + scores @ loading.T


def projection_reconstruction(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return mean + (x - mean) @ basis @ basis.T


def principal_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    ql = orthonormal_columns(left)
    qr = orthonormal_columns(right)
    singular = np.linalg.svd(ql.T @ qr, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth, "fro") / max(np.linalg.norm(truth, "fro"), 1e-15))


def offdiag_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    mask = ~np.eye(P, dtype=bool)
    return float(np.sqrt(np.mean((estimate[mask] - truth[mask]) ** 2)))


def anderson_rubin_rank_diagnostic(loading: np.ndarray) -> dict:
    """Check a sufficient row-deletion rank condition for rank-two FA.

    After deleting each row, search for two disjoint two-row submatrices whose
    ranks are both two.  The witnesses are deterministic lexicographic minima.
    """
    loading = np.asarray(loading, dtype=float)
    deleted_records = {}
    for deleted in range(loading.shape[0]):
        remaining = [row for row in range(loading.shape[0]) if row != deleted]
        witness = None
        for first in itertools.combinations(remaining, K):
            if np.linalg.matrix_rank(loading[list(first), :]) < K:
                continue
            rest = [row for row in remaining if row not in first]
            for second in itertools.combinations(rest, K):
                if np.linalg.matrix_rank(loading[list(second), :]) == K:
                    witness = {"first_rows": list(first), "second_rows": list(second)}
                    break
            if witness is not None:
                break
        deleted_records[str(deleted)] = {"pass": witness is not None, "witness": witness}
    return {
        "criterion": "after deleting any one row, two disjoint remaining two-row submatrices each have rank two",
        "per_deleted_row": deleted_records,
        "overall_pass": bool(all(record["pass"] for record in deleted_records.values())),
        "identification_scope": "sufficient structural identification of diagonal uniqueness and the common-loading structure up to orthogonal loading rotation; not a finite-sample recovery guarantee",
    }


def add_metric(rows: list[dict], split: str, condition: str, preprocessing: str,
               method: str, target: str, metric: str, value: float) -> None:
    rows.append({
        "split": split, "condition": condition, "preprocessing": preprocessing,
        "method": method, "target": target, "metric": metric, "value": float(value),
    })


def fixed_basis() -> np.ndarray:
    raw = np.zeros((P, K))
    raw[0, 0] = 1.0
    raw[1, 1] = 1.0
    return raw


def random_bases() -> list[np.ndarray]:
    return [orthonormal_columns(rng_for(40, i).normal(size=(P, K))) for i in range(RANDOM_PROJECTIONS)]


def population_principal_basis(spec: dict) -> np.ndarray:
    return symmetric_eigh(spec["nominal_covariance"])[1][:, :K]


def score_condition(condition: dict, rows: list[dict]) -> tuple[dict, dict]:
    spec = condition["spec"]
    train = condition["train"]["observed"]
    validation = condition["validation"]["observed"]
    test = condition["test"]["observed"]
    pca = fit_pca(train)
    ppca = fit_ppca(train)
    fa, candidates = select_fa(train, validation)
    fixed = fixed_basis()
    random = random_bases()
    pop_pca = population_principal_basis(spec)
    true_loading = spec["loading"]
    rank_diagnostic = anderson_rubin_rank_diagnostic(true_loading)
    uniqueness_applicable = bool(spec["diagonal_residual_applicable"] and rank_diagnostic["overall_pass"])
    reference_method = "nominal_clean_gaussian_reference" if spec["name"] == "outliers" else "oracle_true_gaussian"
    models = {
        "mean": {"reconstruction": np.broadcast_to(pca["mean"], test.shape)},
        "fixed_projection": {"reconstruction": projection_reconstruction(test, pca["mean"], fixed)},
        "pca": {"reconstruction": projection_reconstruction(test, pca["mean"], pca["basis"])},
        "ppca": {
            "reconstruction": posterior_reconstruction(test, ppca["mean"], ppca["loading"], ppca["noise_variance"] * np.eye(P)),
            "mean": ppca["mean"], "covariance": ppca["covariance"], "loading": ppca["loading"],
            "residual": ppca["noise_variance"] * np.eye(P),
        },
        "fa_selected": {
            "reconstruction": posterior_reconstruction(test, fa["mean"], fa["loading"], np.diag(fa["uniqueness"])),
            "mean": fa["mean"], "covariance": fa["covariance"], "loading": fa["loading"],
            "residual": np.diag(fa["uniqueness"]),
        },
        reference_method: {
            "reconstruction": posterior_reconstruction(test, spec["mean"], spec["loading"], spec["residual_covariance"]),
            "mean": spec["mean"], "covariance": spec["nominal_covariance"], "loading": spec["loading"],
            "residual": spec["residual_covariance"],
        },
    }
    random_mse = []
    for index, basis in enumerate(random):
        reconstructed = projection_reconstruction(test, pca["mean"], basis)
        value = float(np.mean((test - reconstructed) ** 2))
        random_mse.append(value)
        add_metric(rows, "test", spec["name"], "center_train_only",
                   f"random_projection_{index}", "original_coordinate_reconstruction",
                   "heldout_reconstruction_mse", value)
    for method, model in models.items():
        reconstruction_target = (
            "nominal_clean_gaussian_reconstruction"
            if method == "nominal_clean_gaussian_reference"
            else "original_coordinate_reconstruction"
        )
        outlier_condition = spec["name"] == "outliers"
        nll_target = (
            "contaminated_observation_distribution_gaussian_score"
            if outlier_condition else "observable_covariance"
        )
        covariance_error_target = (
            "nominal_clean_gaussian_covariance"
            if outlier_condition else "observable_covariance"
        )
        residual_target = (
            "nominal_clean_gaussian_residual_covariance"
            if outlier_condition else "residual_covariance"
        )
        add_metric(rows, "test", spec["name"], "center_train_only", method,
                   reconstruction_target, "heldout_reconstruction_mse",
                   np.mean((test - model["reconstruction"]) ** 2))
        if "covariance" in model:
            add_metric(rows, "test", spec["name"], "center_train_only", method,
                       nll_target, "heldout_gaussian_nll",
                       gaussian_nll(test, model["mean"], model["covariance"]))
            add_metric(rows, "population", spec["name"], "none", method,
                       covariance_error_target, "relative_frobenius_error",
                       relative_frobenius(model["covariance"], spec["nominal_covariance"]))
            add_metric(rows, "population", spec["name"], "none", method,
                       residual_target, "offdiagonal_rmse",
                       offdiag_rmse(model["residual"], spec["residual_covariance"]))
    for summary, value in (
        ("mean", np.mean(random_mse)), ("minimum", np.min(random_mse)),
        ("maximum", np.max(random_mse)), ("standard_deviation", np.std(random_mse)),
    ):
        add_metric(rows, "test", spec["name"], "center_train_only", "random_projection",
                   "original_coordinate_reconstruction", f"heldout_reconstruction_mse_{summary}", value)
    pca_angles = principal_angles(pca["basis"], pop_pca)
    ppca_angles = principal_angles(ppca["basis"], pop_pca)
    fa_angles = principal_angles(fa["loading"], true_loading)
    for label, angles, method, target in (
        ("pca", pca_angles, "pca", "population_principal_subspace"),
        ("ppca", ppca_angles, "ppca", "population_principal_subspace"),
        ("fa", fa_angles, "fa_selected", "true_common_loading_subspace"),
    ):
        add_metric(rows, "train", spec["name"], "center_train_only", method, target,
                   "principal_angle_max_degrees", np.max(angles))
        add_metric(rows, "train", spec["name"], "center_train_only", method, target,
                   "principal_angle_rms_degrees", np.sqrt(np.mean(angles ** 2)))
    if uniqueness_applicable:
        delta = fa["uniqueness"] - spec["uniqueness"]
        add_metric(rows, "train", spec["name"], "center_train_only", "fa_selected",
                   "identified_diagonal_uniqueness", "uniqueness_rmse", np.sqrt(np.mean(delta ** 2)))
        add_metric(rows, "train", spec["name"], "center_train_only", "fa_selected",
                   "identified_diagonal_uniqueness", "uniqueness_relative_error",
                   np.linalg.norm(delta) / max(np.linalg.norm(spec["uniqueness"]), 1e-15))
    selected_record = {
        "floor": fa["floor"], "restart": fa["restart"], "converged": bool(fa["converged"]),
        "iterations": int(fa["iterations"]), "train_nll": fa["train_nll"],
        "validation_nll": fa["validation_nll"],
    }
    for metric, value in (
        ("selected_floor", fa["floor"]), ("selected_restart", fa["restart"]),
        ("selected_iterations", fa["iterations"]), ("selected_converged", float(fa["converged"])),
        ("selected_train_nll", fa["train_nll"]), ("selected_validation_nll", fa["validation_nll"]),
    ):
        add_metric(rows, "validation", spec["name"], "center_train_only", "fa_selected",
                   "bounded_multistart_selection", metric, value)
    return {
        "pca": pca, "ppca": ppca, "fa": fa, "models": models,
        "pca_angles": pca_angles, "ppca_angles": ppca_angles, "fa_angles": fa_angles,
        "rank_identification": rank_diagnostic,
        "uniqueness_metrics_applicable": uniqueness_applicable,
        "random_projection_mse": random_mse,
    }, {"selected": selected_record, "candidates": candidates}


def scaling_leakage_diagnostic(condition: dict, rows: list[dict]) -> dict:
    train = condition["train"]["observed"]
    test = condition["test"]["observed"]
    proper = fit_standardizer(train)
    leaked = fit_standardizer(np.vstack([train, test]))

    def path(prep: dict) -> tuple[float, np.ndarray, np.ndarray]:
        train_z = transform(train, prep)
        test_z = transform(test, prep)
        model = fit_pca(train_z)
        reconstruction = inverse_transform(
            projection_reconstruction(test_z, model["mean"], model["basis"]), prep
        )
        return float(np.mean((test - reconstruction) ** 2)), model["basis"], reconstruction

    proper_mse, proper_basis, proper_reconstruction = path(proper)
    leaked_mse, leaked_basis, leaked_reconstruction = path(leaked)
    add_metric(rows, "test", "scaling_leakage_negative_control", "standardize_train_only_valid",
               "pca", "original_coordinate_reconstruction", "heldout_reconstruction_mse", proper_mse)
    add_metric(rows, "test", "scaling_leakage_negative_control", "standardize_train_plus_test_invalid",
               "pca", "original_coordinate_reconstruction", "heldout_reconstruction_mse", leaked_mse)
    add_metric(rows, "test", "scaling_leakage_negative_control", "invalid_minus_valid",
               "pca", "original_coordinate_reconstruction", "heldout_reconstruction_mse_gap", leaked_mse - proper_mse)

    poisoned = test.copy()
    poisoned[:, 0] += 50.0
    poisoned[:, 1] *= 5.0
    proper_poison = fit_standardizer(train)
    leaked_poison = fit_standardizer(np.vstack([train, poisoned]))
    return {
        "valid_path": "feature mean and scale fitted on training observations only",
        "invalid_path": "feature mean and scale fitted on pooled training and test observations; intentionally leaked negative control",
        "proper_mse_original_coordinates": proper_mse,
        "leaked_mse_original_coordinates": leaked_mse,
        "leaked_minus_proper_mse": leaked_mse - proper_mse,
        "proper_test_poison_mean_max_abs_change": float(np.max(np.abs(proper_poison["mean"] - proper["mean"]))),
        "proper_test_poison_scale_max_abs_change": float(np.max(np.abs(proper_poison["scale"] - proper["scale"]))),
        "leaked_test_poison_mean_max_abs_change": float(np.max(np.abs(leaked_poison["mean"] - leaked["mean"]))),
        "leaked_test_poison_scale_max_abs_change": float(np.max(np.abs(leaked_poison["scale"] - leaked["scale"]))),
        "proper_vs_leaked_basis_angle_max_degrees": float(np.max(principal_angles(proper_basis, leaked_basis))),
        "proper_vs_leaked_reconstruction_max_abs": float(np.max(np.abs(proper_reconstruction - leaked_reconstruction))),
        "interpretation": "The pooled path is invalid regardless of whether its finite-sample test MSE is numerically better or worse.",
    }


def array_max_abs(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def model_parameter_discrepancies(left: dict, right: dict, keys: tuple[str, ...]) -> dict:
    records = {}
    for key in keys:
        records[key] = array_max_abs(left[key], right[key])
    records["maximum"] = max(records.values(), default=0.0)
    return records


def observation_only_snapshot(bundle: dict, evaluation_observed: np.ndarray) -> dict:
    """Fit every observation-only model without accepting latent arrays or test fit data."""
    train = bundle["train"]["observed"]
    validation = bundle["validation"]["observed"]
    pca = fit_pca(train)
    ppca = fit_ppca(train)
    candidates = []
    candidate_records = []
    for floor in FA_FLOORS:
        for restart in FA_RESTARTS:
            model = fit_fa_em(train, floor, restart)
            validation_nll = gaussian_nll(validation, model["mean"], model["covariance"])
            model["validation_nll"] = validation_nll
            candidates.append(model)
            candidate_records.append({
                "floor": floor,
                "restart": restart,
                "converged": bool(model["converged"]),
                "iterations": int(model["iterations"]),
                "train_nll": model["train_nll"],
                "validation_nll": validation_nll,
                "final_train_improvement": float(model["history"][0] - model["history"][-1]),
                "maximum_train_nll_increase": float(max(
                    [0.0] + [b - a for a, b in zip(model["history"][:-1], model["history"][1:])]
                )),
            })
    selected_index = min(
        range(len(candidate_records)),
        key=lambda i: (
            candidate_records[i]["validation_nll"],
            candidate_records[i]["floor"],
            candidate_records[i]["restart"],
        ),
    )
    selected = candidates[selected_index]
    reconstructions = {
        "pca": projection_reconstruction(evaluation_observed, pca["mean"], pca["basis"]),
        "ppca": posterior_reconstruction(
            evaluation_observed, ppca["mean"], ppca["loading"],
            ppca["noise_variance"] * np.eye(P),
        ),
        "fa_selected": posterior_reconstruction(
            evaluation_observed, selected["mean"], selected["loading"],
            np.diag(selected["uniqueness"]),
        ),
    }
    for index, model in enumerate(candidates):
        reconstructions[f"fa_candidate_{index}"] = posterior_reconstruction(
            evaluation_observed, model["mean"], model["loading"],
            np.diag(model["uniqueness"]),
        )
    return {
        "pca": pca,
        "ppca": ppca,
        "fa_candidates": candidates,
        "fa_candidate_records": candidate_records,
        "selected_index": selected_index,
        "fa_selected": selected,
        "reconstructions": reconstructions,
    }


def record_max_abs(left: dict, right: dict) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in left)


def compare_observation_only_snapshots(left: dict, right: dict) -> dict:
    pca = model_parameter_discrepancies(
        left["pca"], right["pca"], ("mean", "basis", "eigenvalues", "sample_covariance")
    )
    ppca = model_parameter_discrepancies(
        left["ppca"], right["ppca"],
        ("mean", "loading", "noise_variance", "covariance", "basis"),
    )
    candidate_details = {}
    for index, (left_model, right_model, left_record, right_record) in enumerate(zip(
        left["fa_candidates"], right["fa_candidates"],
        left["fa_candidate_records"], right["fa_candidate_records"],
    )):
        parameters = model_parameter_discrepancies(
            left_model, right_model, ("mean", "loading", "uniqueness", "covariance")
        )
        candidate_details[str(index)] = {
            "parameter_max_abs_discrepancy": parameters["maximum"],
            "selection_record_max_abs_discrepancy": record_max_abs(left_record, right_record),
        }
    selected_parameters = model_parameter_discrepancies(
        left["fa_selected"], right["fa_selected"], ("mean", "loading", "uniqueness", "covariance")
    )
    reconstruction_details = {
        method: array_max_abs(left["reconstructions"][method], right["reconstructions"][method])
        for method in left["reconstructions"]
    }
    return {
        "pca_fitted_parameters": pca,
        "ppca_fitted_parameters": ppca,
        "fa_candidates": candidate_details,
        "fa_candidate_parameter_max_abs_discrepancy": max(
            item["parameter_max_abs_discrepancy"] for item in candidate_details.values()
        ),
        "fa_candidate_selection_record_max_abs_discrepancy": max(
            item["selection_record_max_abs_discrepancy"] for item in candidate_details.values()
        ),
        "fa_selected_parameters": selected_parameters,
        "fa_selected_index_unchanged": left["selected_index"] == right["selected_index"],
        "test_reconstructions_by_method_max_abs_discrepancy": reconstruction_details,
        "test_reconstruction_max_abs_discrepancy": max(reconstruction_details.values()),
        "overall_max_abs_discrepancy": max(
            pca["maximum"], ppca["maximum"], selected_parameters["maximum"],
            max(item["parameter_max_abs_discrepancy"] for item in candidate_details.values()),
            max(item["selection_record_max_abs_discrepancy"] for item in candidate_details.values()),
            max(reconstruction_details.values()),
        ),
    }


def leakage_and_generation_diagnostics(all_data: dict, fitted: dict) -> dict:
    del fitted  # Diagnostics recompute independently from the retained observation arrays.

    def copy_bundle(condition: dict) -> dict:
        return {
            split: {
                "observed": condition[split]["observed"].copy(),
                "latent": condition[split]["latent"].copy(),
                "outlier_mask": condition[split]["outlier_mask"].copy(),
            }
            for split in SPLIT_SIZES
        }

    poison_records = {}
    split_checks = {}
    any_equal_split_prefix = False
    for condition_index, name in enumerate(CONDITION_ORDER):
        condition = all_data[name]
        baseline_bundle = copy_bundle(condition)
        original_test = baseline_bundle["test"]["observed"].copy()
        baseline = observation_only_snapshot(baseline_bundle, original_test)

        test_poison_bundle = copy_bundle(condition)
        test_poison_bundle["test"]["observed"][:] = rng_for(70, condition_index).normal(
            loc=1e4, scale=1e3, size=test_poison_bundle["test"]["observed"].shape
        )
        test_poison = observation_only_snapshot(test_poison_bundle, original_test)

        latent_poison_bundle = copy_bundle(condition)
        for split_index, split in enumerate(SPLIT_SIZES):
            latent_poison_bundle[split]["latent"][:] = rng_for(71, condition_index, split_index).normal(
                loc=-1e5, scale=2e4, size=latent_poison_bundle[split]["latent"].shape
            )
        latent_poison = observation_only_snapshot(latent_poison_bundle, original_test)

        poison_records[name] = {
            "test_observation_poison": {
                "poisoned_test_norm": float(np.linalg.norm(test_poison_bundle["test"]["observed"])),
                **compare_observation_only_snapshots(baseline, test_poison),
            },
            "latent_truth_poison": {
                "poisoned_all_latent_arrays_norm": float(sum(
                    np.linalg.norm(latent_poison_bundle[split]["latent"]) for split in SPLIT_SIZES
                )),
                **compare_observation_only_snapshots(baseline, latent_poison),
            },
        }

        spec = condition["spec"]
        for split in SPLIT_SIZES:
            regenerated = generate_split(spec, split)
            key = f"{name}:{split}"
            split_checks[key] = {
                "observed_max_abs": array_max_abs(regenerated["observed"], condition[split]["observed"]),
                "latent_max_abs": array_max_abs(regenerated["latent"], condition[split]["latent"]),
                "outlier_mask_equal": bool(np.array_equal(regenerated["outlier_mask"], condition[split]["outlier_mask"])),
            }
        prefix_length = min(SPLIT_SIZES["validation"], SPLIT_SIZES["train"])
        any_equal_split_prefix = any_equal_split_prefix or bool(np.array_equal(
            condition["train"]["observed"][:prefix_length],
            condition["validation"]["observed"][:prefix_length],
        ))

    return {
        "poison_regressions": poison_records,
        "poison_scope": {
            "conditions": list(CONDITION_ORDER),
            "methods": ["pca", "ppca", "six_fa_candidates", "selected_fa", "all_test_reconstructions"],
            "test_observation_poison_evaluation": "test arrays are poisoned, while reconstruction comparisons use the same original test observations so only fitted-state leakage can change outputs",
            "latent_poison_evaluation": "every retained train/validation/test latent array is poisoned; observation-only fit and reconstruction must remain unchanged",
        },
        "independent_split_regeneration": {
            "checks": split_checks,
            "all_exact": bool(all(
                record["observed_max_abs"] == 0.0
                and record["latent_max_abs"] == 0.0
                and record["outlier_mask_equal"]
                for record in split_checks.values()
            )),
            "any_train_validation_prefix_arrays_equal": any_equal_split_prefix,
        },
        "retained_latent_truth_shape": {
            name: {split: list(all_data[name][split]["latent"].shape) for split in SPLIT_SIZES}
            for name in CONDITION_ORDER
        },
    }

def covariance_diagnostics(all_data: dict, fitted: dict) -> dict:
    records = {}
    for name in CONDITION_ORDER:
        matrices = {
            "population": all_data[name]["spec"]["nominal_covariance"],
            "ppca": fitted[name]["ppca"]["covariance"],
            "fa": fitted[name]["fa"]["covariance"],
        }
        for label, matrix in matrices.items():
            records[f"{name}:{label}"] = {
                "symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
                "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)))),
                "all_finite": bool(np.all(np.isfinite(matrix))),
            }
    return {
        "records": records,
        "max_symmetry_error": max(x["symmetry_max_abs"] for x in records.values()),
        "minimum_eigenvalue": min(x["minimum_eigenvalue"] for x in records.values()),
        "all_finite": bool(all(x["all_finite"] for x in records.values())),
    }


def failure_summaries(all_data: dict, fitted: dict, rows: list[dict]) -> dict:
    del all_data
    def metric(condition: str, method: str, name: str) -> float:
        return next(r["value"] for r in rows if r["condition"] == condition and r["method"] == method and r["metric"] == name)
    summaries = {}
    for name in CONDITION_ORDER:
        reference_method = "nominal_clean_gaussian_reference" if name == "outliers" else "oracle_true_gaussian"
        selected_converged = bool(fitted[name]["fa"]["converged"])
        summaries[name] = {
            "pca_population_subspace_max_angle_degrees": metric(name, "pca", "principal_angle_max_degrees"),
            "ppca_population_subspace_max_angle_degrees": metric(name, "ppca", "principal_angle_max_degrees"),
            "fa_loading_subspace_max_angle_degrees": metric(name, "fa_selected", "principal_angle_max_degrees"),
            "pca_test_reconstruction_mse": metric(name, "pca", "heldout_reconstruction_mse"),
            "fa_test_reconstruction_mse": metric(name, "fa_selected", "heldout_reconstruction_mse"),
            "fa_test_gaussian_nll": metric(name, "fa_selected", "heldout_gaussian_nll"),
            "reference_method": reference_method,
            "reference_test_gaussian_nll": metric(name, reference_method, "heldout_gaussian_nll"),
            "fa_selected_converged": selected_converged,
            "fa_result_status": (
                "converged_bounded_fit"
                if selected_converged
                else "bounded_optimizer_nonconvergence_computational_failure"
            ),
        }
    summaries["interpretation"] = {
        "repeated_signal_eigenvalues": "Only the complete rank-two eigenspace is compared; displayed axes within it are not treated as identified.",
        "correlated_residuals": "Diagonal-noise FA is misspecified; its fitted diagonal residual cannot reproduce true residual off-diagonals.",
        "high_variance_nuisance": "The selected FA fit reached the fixed iteration bound; its FA result is a bounded optimizer nonconvergence/computational failure, not evidence of method-level population failure. PCA is scored against its population principal-subspace target.",
        "outliers": "Gaussian likelihood and covariance estimators are intentionally exposed to non-Gaussian contamination; the nominal oracle is not a contamination model.",
        "near_zero_uniqueness": "The selected FA fit reached the fixed iteration bound near the uniqueness boundary; this is recorded as bounded optimizer nonconvergence/computational failure. The floor remains a numerical/modeling choice.",
    }
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["value"] = format(record["value"], ".17g")
            writer.writerow(record)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float) -> str:
    return f"{value:.4g}"


def make_summary(path: Path, all_data: dict, fitted: dict, diagnostics: dict, rows: list[dict]) -> None:
    width, height = 1500, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 14), "LDG static linear latent recovery — deterministic diagnostic", fill="black", font=font)

    def panel(box, title):
        draw.rectangle(box, outline="black", width=2)
        draw.text((box[0] + 8, box[1] + 7), title, fill="black", font=font)

    panel((24, 44, 750, 500), "A. Matched test observations projected onto true loading coordinates")
    matched = all_data["matched_diagonal"]
    x = matched["test"]["observed"][:180]
    scores = x @ orthonormal_columns(matched["spec"]["loading"])
    xmin, xmax = float(scores[:, 0].min()), float(scores[:, 0].max())
    ymin, ymax = float(scores[:, 1].min()), float(scores[:, 1].max())
    for a, b in scores:
        px = 50 + (a - xmin) / max(xmax - xmin, 1e-12) * 670
        py = 465 - (b - ymin) / max(ymax - ymin, 1e-12) * 380
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(50, 100, 170))
    draw.text((42, 478), "Display only; acceptance uses numeric held-out and subspace metrics.", fill="black", font=font)

    panel((775, 44, 1475, 500), "B. Whole-subspace angle diagnostics (degrees)")
    for i, name in enumerate(CONDITION_ORDER):
        pca_angle = diagnostics["failure_conditions"][name]["pca_population_subspace_max_angle_degrees"]
        fa_angle = diagnostics["failure_conditions"][name]["fa_loading_subspace_max_angle_degrees"]
        draw.text((790, 82 + i * 62), f"{name}", fill="black", font=font)
        draw.rectangle((1055, 80 + i * 62, 1055 + min(300, pca_angle * 5), 96 + i * 62), fill=(31, 119, 180))
        draw.rectangle((1055, 101 + i * 62, 1055 + min(300, fa_angle * 5), 117 + i * 62), fill=(214, 39, 40))
        draw.text((1370, 82 + i * 62), f"P {fmt(pca_angle)}", fill="black", font=font)
        draw.text((1370, 103 + i * 62), f"F {fmt(fa_angle)}", fill="black", font=font)
    draw.text((790, 468), "blue=PCA vs population PCA target; red=FA vs true loading subspace", fill="black", font=font)

    panel((24, 525, 750, 950), "C. Matched held-out metrics")
    keys = [
        ("mean", "heldout_reconstruction_mse"), ("fixed_projection", "heldout_reconstruction_mse"),
        ("random_projection", "heldout_reconstruction_mse_mean"), ("pca", "heldout_reconstruction_mse"),
        ("ppca", "heldout_gaussian_nll"), ("fa_selected", "heldout_gaussian_nll"),
        ("oracle_true_gaussian", "heldout_gaussian_nll"),
    ]
    for i, (method, metric_name) in enumerate(keys):
        value = next(r["value"] for r in rows if r["condition"] == "matched_diagonal" and r["method"] == method and r["metric"] == metric_name)
        draw.text((42, 570 + i * 48), f"{method} | {metric_name}: {fmt(value)}", fill="black", font=font)

    panel((775, 525, 1475, 950), "D. Verification and interpretation boundaries")
    cov = diagnostics["covariance_checks"]
    leak = diagnostics["scaling_leakage_negative_control"]
    gen = diagnostics["leakage_and_generation"]
    poison_max = max(
        record[poison]["overall_max_abs_discrepancy"]
        for record in gen["poison_regressions"].values()
        for poison in ("test_observation_poison", "latent_truth_poison")
    )
    lines = [
        f"covariance max asymmetry: {fmt(cov['max_symmetry_error'])}",
        f"covariance minimum eigenvalue: {fmt(cov['minimum_eigenvalue'])}",
        f"split regeneration exact: {gen['independent_split_regeneration']['all_exact']}",
        f"all-condition poison maximum discrepancy: {fmt(poison_max)}",
        f"valid/leaked scaling MSE: {fmt(leak['proper_mse_original_coordinates'])} / {fmt(leak['leaked_mse_original_coordinates'])}",
        "Pooled train+test scaling is invalid, irrespective of its score.",
        "Repeated eigenvalues identify an eigenspace, not preferred axes.",
        "PCA targets variance; diagonal FA assumes diagonal residual noise.",
        "Recovery here is simulation-conditional, not mechanistic evidence.",
    ]
    for i, line in enumerate(lines):
        draw.text((792, 570 + i * 36), line, fill="black", font=font)
    image.save(path, format="PNG", optimize=False)


def build(root: Path, quiet: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    all_data = {name: generate_condition(name) for name in CONDITION_ORDER}
    rows: list[dict] = []
    fitted = {}
    convergence = {}
    for name in CONDITION_ORDER:
        fitted[name], convergence[name] = score_condition(all_data[name], rows)
    scaling = scaling_leakage_diagnostic(all_data["matched_diagonal"], rows)
    leakage = leakage_and_generation_diagnostics(all_data, fitted)
    covariance = covariance_diagnostics(all_data, fitted)
    failures = failure_summaries(all_data, fitted, rows)
    rank_identification = {name: fitted[name]["rank_identification"] for name in CONDITION_ORDER}
    uniqueness_applicability = {
        name: {
            "diagonal_residual_specification_applicable": bool(all_data[name]["spec"]["diagonal_residual_applicable"]),
            "anderson_rubin_sufficient_rank_pass": bool(rank_identification[name]["overall_pass"]),
            "uniqueness_metrics_emitted": bool(fitted[name]["uniqueness_metrics_applicable"]),
            "rule": "emit uniqueness error only when the analyzed condition has an applicable diagonal residual specification and the sufficient row-deletion rank diagnostic passes",
        }
        for name in CONDITION_ORDER
    }
    diagnostics = {
        "generator": {
            "equation": "x = mean + Lambda z + epsilon; z ~ N(0,I_2)",
            "ambient_dimension": P, "latent_rank": K, "split_sizes": SPLIT_SIZES,
            "independent_splits": True, "retained_latent_truth": True,
            "conditions": {
                name: {
                    "loading": all_data[name]["spec"]["loading"].tolist(),
                    "residual_covariance": all_data[name]["spec"]["residual_covariance"].tolist(),
                    "outlier_fraction": all_data[name]["spec"]["outlier_fraction"],
                    "outlier_scale": all_data[name]["spec"]["outlier_scale"],
                    "diagonal_residual_specification_applicable": all_data[name]["spec"]["diagonal_residual_applicable"],
                    "anderson_rubin_sufficient_rank_pass": rank_identification[name]["overall_pass"],
                    "uniqueness_metrics_identified_and_reported": fitted[name]["uniqueness_metrics_applicable"],
                } for name in CONDITION_ORDER
            },
        },
        "fixed_and_random_projections": {
            "fixed_orthonormal_basis": fixed_basis().tolist(),
            "random_projection_count": RANDOM_PROJECTIONS,
            "random_bases": [basis.tolist() for basis in random_bases()],
            "fit_policy": "fixed before observing outcomes; no fitting or selection on validation/test data",
        },
        "fa_multistart_selection": convergence,
        "structural_identification": {
            "rank_diagnostics": rank_identification,
            "uniqueness_metric_applicability": uniqueness_applicability,
            "structural_vs_finite_recovery": "Passing the sufficient rank condition supports structural identification up to orthogonal loading rotation under the diagonal-residual FA specification; it does not guarantee finite-sample accuracy or optimizer convergence.",
        },
        "scaling_leakage_negative_control": scaling,
        "leakage_and_generation": leakage,
        "covariance_checks": covariance,
        "failure_conditions": failures,
        "scientific_limits": [
            "PCA is evaluated against the population covariance principal subspace, not presumed to recover the factor-loading subspace.",
            "Repeated eigenvalues identify the whole repeated eigenspace; no arbitrary within-eigenspace axis is scored.",
            "FA uniqueness metrics are emitted only when the diagonal-residual specification is applicable and the computed Anderson-Rubin-style sufficient row-deletion rank check passes.",
            "The bounded EM search is not a claim of global maximum-likelihood optimization.",
            "For outliers, the generating clean-Gaussian covariance is labeled nominal_clean_gaussian_reference and is not called the covariance of the contaminated observable law.",
            "Low reconstruction error or subspace alignment is not evidence of mechanism, causality, intrinsic dimension, or a unique latent generator.",
        ],
    }
    write_csv(root / "metrics.csv", rows)
    (root / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    make_summary(root / "summary.png", all_data, fitted, diagnostics, rows)
    hashes = {name: sha256(root / name) for name in ARTIFACT_NAMES}
    manifest = {
        "schema_version": 1, "project": "Latent Dynamics & Geometry",
        "artifact": "static-linear-latent-recovery", "date": DATE,
        "master_seed": MASTER_SEED,
        "rng": "numpy.Generator(PCG64(SeedSequence((master_seed, stage, condition, split))))",
        "runtime": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pillow": getattr(sys.modules.get("PIL"), "__version__", "unknown"),
            "platform": platform.platform(),
        },
        "dimensions": {"ambient": P, "latent": K}, "split_sizes": SPLIT_SIZES,
        "condition_order": list(CONDITION_ORDER),
        "methods": [
            "mean", "fixed_orthonormal_projection", f"{RANDOM_PROJECTIONS}_deterministic_random_rank_matched_projections",
            "ordinary_covariance_pca", "probabilistic_pca_isotropic_noise",
            "bounded_diagonal_noise_fa_em", "oracle_true_gaussian",
            "nominal_clean_gaussian_reference_for_outlier_condition",
        ],
        "applicability_rules": {
            "pca_subspace_target": "population principal subspace of the nominal covariance for each declared condition",
            "ppca_subspace_target": "same population principal subspace; closed-form PPCA uses the fitted PCA basis",
            "uniqueness_error": "reported only when diagonal residual specification is applicable and the computed sufficient row-deletion rank diagnostic passes",
            "outlier_reference_labels": {
                "method": "nominal_clean_gaussian_reference",
                "covariance_target": "nominal_clean_gaussian_covariance",
                "scope": "clean generating Gaussian reference, not the covariance model of the contaminated observable law",
            },
            "nonconverged_fa": "reported as bounded optimizer nonconvergence/computational failure, not method-level population failure",
        },
        "fa_selection": {
            "fit_data": "train observations only", "selection_data": "validation observations only",
            "criterion": "minimum validation Gaussian NLL with deterministic floor/restart tie break",
            "floors": list(FA_FLOORS), "restarts": list(FA_RESTARTS),
            "max_iterations": FA_MAX_ITER, "relative_nll_tolerance": FA_TOL,
        },
        "test_latent_truth_use": "scoring subspace/recovery diagnostics only; never fitting or selection",
        "tolerances": TOL,
        "commands": {
            "python_executable": sys.executable,
            "generate": f"{sys.executable} toy-models/static-linear-latent-recovery.py --generate",
            "verify": f"{sys.executable} toy-models/static-linear-latent-recovery.py --verify",
        },
        "artifact_files": ["manifest.json", *ARTIFACT_NAMES], "files_sha256": hashes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not quiet:
        matched_failure = failures["matched_diagonal"]
        print(json.dumps({
            "artifact_root": str(root), "metrics_rows": len(rows),
            "matched_pca_population_angle_max_degrees": matched_failure["pca_population_subspace_max_angle_degrees"],
            "matched_fa_loading_angle_max_degrees": matched_failure["fa_loading_subspace_max_angle_degrees"],
            "matched_fa_test_nll": matched_failure["fa_test_gaussian_nll"],
        }, indent=2))
    return diagnostics


def compare_roots(left: Path, right: Path) -> list[str]:
    return [name for name in ("manifest.json", *ARTIFACT_NAMES) if (left / name).read_bytes() != (right / name).read_bytes()]


def verify(root: Path, recompute: bool = True) -> None:
    errors = []
    for name in ("manifest.json", *ARTIFACT_NAMES):
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(errors))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if sha256(root / name) != expected:
            errors.append(f"hash mismatch: {name}")
    if manifest.get("tolerances") != TOL:
        errors.append("manifest tolerance mismatch")
    covariance = diagnostics["covariance_checks"]
    if covariance["max_symmetry_error"] > TOL["symmetry"]:
        errors.append("covariance symmetry tolerance failed")
    if covariance["minimum_eigenvalue"] < -TOL["psd"] or not covariance["all_finite"]:
        errors.append("covariance PSD/finiteness check failed")
    leakage = diagnostics["leakage_and_generation"]
    poison_records = leakage.get("poison_regressions", {})
    if set(poison_records) != set(CONDITION_ORDER):
        errors.append("poison regression conditions are incomplete")
    for condition, condition_record in poison_records.items():
        for poison_label in ("test_observation_poison", "latent_truth_poison"):
            record = condition_record.get(poison_label, {})
            if not record.get("fa_selected_index_unchanged", False):
                errors.append(f"{condition}:{poison_label} changed selected FA candidate")
            for method_label in ("pca_fitted_parameters", "ppca_fitted_parameters", "fa_selected_parameters"):
                if record.get(method_label, {}).get("maximum", float("inf")) > TOL["poison"]:
                    errors.append(f"{condition}:{poison_label}:{method_label} exceeded poison tolerance")
            if len(record.get("fa_candidates", {})) != len(FA_FLOORS) * len(FA_RESTARTS):
                errors.append(f"{condition}:{poison_label} lacks six FA candidate checks")
            if record.get("fa_candidate_parameter_max_abs_discrepancy", float("inf")) > TOL["poison"]:
                errors.append(f"{condition}:{poison_label} changed FA candidate parameters")
            if record.get("fa_candidate_selection_record_max_abs_discrepancy", float("inf")) > TOL["poison"]:
                errors.append(f"{condition}:{poison_label} changed FA candidate records")
            reconstruction_records = record.get("test_reconstructions_by_method_max_abs_discrepancy", {})
            expected_reconstructions = {"pca", "ppca", "fa_selected"} | {
                f"fa_candidate_{index}" for index in range(len(FA_FLOORS) * len(FA_RESTARTS))
            }
            if set(reconstruction_records) != expected_reconstructions:
                errors.append(f"{condition}:{poison_label} reconstruction checks incomplete")
            if record.get("test_reconstruction_max_abs_discrepancy", float("inf")) > TOL["poison"]:
                errors.append(f"{condition}:{poison_label} changed test reconstructions")
            if record.get("overall_max_abs_discrepancy", float("inf")) > TOL["poison"]:
                errors.append(f"{condition}:{poison_label} overall poison regression failed")
    split = leakage["independent_split_regeneration"]
    if not split["all_exact"] or split["any_train_validation_prefix_arrays_equal"]:
        errors.append("independent deterministic split check failed")
    scaling = diagnostics["scaling_leakage_negative_control"]
    if scaling["proper_test_poison_mean_max_abs_change"] > TOL["poison"] or scaling["proper_test_poison_scale_max_abs_change"] > TOL["poison"]:
        errors.append("proper training-only scaling changed under test poison")
    if scaling["leaked_test_poison_mean_max_abs_change"] <= 1.0 or scaling["leaked_test_poison_scale_max_abs_change"] <= 1.0:
        errors.append("invalid pooled scaling did not respond to test poison")
    for name, selection in diagnostics["fa_multistart_selection"].items():
        if len(selection["candidates"]) != len(FA_FLOORS) * len(FA_RESTARTS):
            errors.append(f"incomplete FA multistart record: {name}")
        if any(candidate["maximum_train_nll_increase"] > 1e-10 for candidate in selection["candidates"]):
            errors.append(f"FA EM likelihood monotonicity failed: {name}")
        selected = selection["selected"]
        best = min(selection["candidates"], key=lambda x: (x["validation_nll"], x["floor"], x["restart"]))
        if selected["floor"] != best["floor"] or selected["restart"] != best["restart"]:
            errors.append(f"FA selection record mismatch: {name}")

    structural = diagnostics["structural_identification"]
    for name in CONDITION_ORDER:
        rank_record = structural["rank_diagnostics"][name]
        loading = np.asarray(diagnostics["generator"]["conditions"][name]["loading"], dtype=float)
        computed_all_pass = True
        for deleted_text, deletion in rank_record["per_deleted_row"].items():
            deleted = int(deleted_text)
            witness = deletion["witness"]
            valid = witness is not None
            if valid:
                first = witness["first_rows"]
                second = witness["second_rows"]
                valid = (
                    deleted not in first and deleted not in second
                    and len(first) == K and len(second) == K
                    and set(first).isdisjoint(second)
                    and np.linalg.matrix_rank(loading[first, :]) == K
                    and np.linalg.matrix_rank(loading[second, :]) == K
                )
            if bool(deletion["pass"]) != bool(valid):
                errors.append(f"rank witness validity mismatch: {name}:deleted_{deleted}")
            computed_all_pass = computed_all_pass and bool(valid)
        if bool(rank_record["overall_pass"]) != computed_all_pass:
            errors.append(f"overall rank diagnostic mismatch: {name}")
        applicability = structural["uniqueness_metric_applicability"][name]
        expected_emit = bool(
            applicability["diagonal_residual_specification_applicable"]
            and applicability["anderson_rubin_sufficient_rank_pass"]
        )
        if applicability["uniqueness_metrics_emitted"] != expected_emit:
            errors.append(f"uniqueness applicability rule mismatch: {name}")

    with (root / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(METRIC_FIELDS):
            errors.append(f"metrics CSV fields mismatch: {reader.fieldnames}")
        rows = list(reader)
    composite_keys = set()
    for index, row in enumerate(rows, start=2):
        if set(row) != set(METRIC_FIELDS):
            errors.append(f"metrics CSV row {index} does not have exactly seven fields")
            continue
        if any(not str(row[field]).strip() for field in METRIC_FIELDS):
            errors.append(f"metrics CSV row {index} has an empty field")
        try:
            value = float(row["value"])
            if not math.isfinite(value):
                errors.append(f"metrics CSV row {index} has non-finite value")
        except ValueError:
            errors.append(f"metrics CSV row {index} has non-numeric value")
        key = tuple(row[field] for field in METRIC_FIELDS[:-1])
        if key in composite_keys:
            errors.append(f"duplicate metrics composite key at row {index}: {key}")
        composite_keys.add(key)
    required = {
        "heldout_reconstruction_mse", "heldout_gaussian_nll", "principal_angle_max_degrees",
        "principal_angle_rms_degrees", "relative_frobenius_error", "offdiagonal_rmse",
        "selected_validation_nll",
    }
    present = {row["metric"] for row in rows}
    if not required.issubset(present):
        errors.append(f"required metrics missing: {sorted(required - present)}")
    for name in CONDITION_ORDER:
        for metric_name in ("principal_angle_max_degrees", "principal_angle_rms_degrees"):
            pca_value = next(float(row["value"]) for row in rows if row["condition"] == name and row["method"] == "pca" and row["metric"] == metric_name)
            ppca_value = next(float(row["value"]) for row in rows if row["condition"] == name and row["method"] == "ppca" and row["metric"] == metric_name)
            if abs(pca_value - ppca_value) > TOL["poison"]:
                errors.append(f"PPCA/PCA principal-subspace angle mismatch: {name}:{metric_name}")
    uniqueness_conditions = {row["condition"] for row in rows if row["metric"].startswith("uniqueness_")}
    allowed = {
        name for name in CONDITION_ORDER
        if structural["uniqueness_metric_applicability"][name]["uniqueness_metrics_emitted"]
    }
    if uniqueness_conditions != allowed:
        errors.append("uniqueness metrics emitted outside or missing from applicable conditions")
    outlier_rows = [row for row in rows if row["condition"] == "outliers"]
    if any(row["method"] == "oracle_true_gaussian" for row in outlier_rows):
        errors.append("outlier condition retains ambiguous oracle_true_gaussian label")
    nominal_rows = [row for row in outlier_rows if row["method"] == "nominal_clean_gaussian_reference"]
    if not nominal_rows or not any(row["target"] == "nominal_clean_gaussian_covariance" for row in nominal_rows):
        errors.append("outlier nominal clean-Gaussian reference labels are missing")
    if any(
        row["metric"] == "relative_frobenius_error" and row["target"] == "observable_covariance"
        for row in outlier_rows
    ):
        errors.append("outlier covariance errors retain ambiguous observable_covariance target")
    outlier_covariance_errors = [
        row for row in outlier_rows if row["metric"] == "relative_frobenius_error"
    ]
    if not outlier_covariance_errors or any(
        row["target"] != "nominal_clean_gaussian_covariance"
        for row in outlier_covariance_errors
    ):
        errors.append("outlier covariance errors are not consistently labelled nominal clean Gaussian")
    for name in ("high_variance_nuisance", "near_zero_uniqueness"):
        failure = diagnostics["failure_conditions"][name]
        if failure["fa_selected_converged"] or failure["fa_result_status"] != "bounded_optimizer_nonconvergence_computational_failure":
            errors.append(f"nonconverged FA failure is not explicitly computational: {name}")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="ldg-static-linear-verify-") as temp:
            regenerated = Path(temp) / "artifact"
            build(regenerated, quiet=True)
            errors.extend(f"byte mismatch: {name}" for name in compare_roots(root, regenerated))
    if errors:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"verified": True, "artifact_root": str(root), "deterministic_recomputation": recompute}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="generate the four bounded artifacts")
    parser.add_argument("--verify", action="store_true", help="verify hashes, diagnostics, and deterministic regeneration")
    parser.add_argument("--no-recompute", action="store_true", help="skip byte-identical temporary regeneration")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    generate, check = args.generate, args.verify
    if not generate and not check:
        generate = check = True
    if generate:
        build(args.artifact_root)
    if check:
        verify(args.artifact_root, recompute=not args.no_recompute)


if __name__ == "__main__":
    main()
