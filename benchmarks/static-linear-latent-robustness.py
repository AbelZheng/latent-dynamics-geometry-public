#!/usr/bin/env python3
"""P4-U02 deterministic static linear latent recovery benchmark.

The benchmark imports the stable Phase 3 static-linear implementation without
editing it. Generation, bounded train/validation selection, inference, and
scoring are separate. Only NumPy and Pillow are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MASTER_SEED = 20260820
REPLICATES = 4
P = 8
K = 2
TOL = {"poison": 1e-12, "oracle": 1e-10, "ordering": 0.0}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PHASE3_PATH = REPO_ROOT / "toy-models" / "static-linear-latent-recovery.py"
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "static-linear-latent-robustness"
ARTIFACT_NAMES = ("results.csv", "diagnostics.json", "summary.png")
STATUS_VALUES = ("ok", "nonconvergence", "invalid", "inapplicable")

BASE_UNIQUENESS = (0.12, 0.20, 0.32, 0.48, 0.70, 0.95, 1.25, 1.60)
SCENARIOS = (
    {"id": "reference", "class": "reference", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "heterogeneous_diagonal"},
    {"id": "small_sample", "class": "sample_size", "n_train": 60, "n_validation": 60, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "heterogeneous_diagonal"},
    {"id": "large_sample", "class": "sample_size", "n_train": 600, "n_validation": 200, "n_test": 240,
     "strengths": (2.0, 1.25), "residual": "heterogeneous_diagonal"},
    {"id": "weak_separation", "class": "signal_eigenvalue_separation", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (1.25, 0.95), "residual": "homogeneous_diagonal"},
    {"id": "repeated_signal", "class": "signal_eigenvalue_separation", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (1.50, 1.50), "residual": "homogeneous_diagonal"},
    {"id": "heterogeneous_uniqueness", "class": "heterogeneous_uniqueness", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "heterogeneous_diagonal"},
    {"id": "near_zero_uniqueness", "class": "near_zero_uniqueness", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "near_zero_diagonal"},
    {"id": "correlated_residuals", "class": "correlated_residuals", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "correlated"},
    {"id": "high_nuisance", "class": "high_nuisance", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "high_nuisance"},
    {"id": "outliers", "class": "outliers", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "heterogeneous_diagonal", "outlier_fraction": 0.025, "outlier_scale": 9.0},
    {"id": "preprocessing_leakage", "class": "preprocessing_leakage", "n_train": 240, "n_validation": 120, "n_test": 180,
     "strengths": (2.0, 1.25), "residual": "heterogeneous_diagonal"},
)

METHODS = (
    {"id": "mean", "family": "mean", "information_set": "train_only", "parameter_source": "train_fit", "preprocessing": "center_train_only"},
    {"id": "fixed_projection", "family": "fixed_projection", "information_set": "fixed_reference", "parameter_source": "predeclared", "preprocessing": "center_train_only"},
    {"id": "random_projection", "family": "random_projection", "information_set": "fixed_reference", "parameter_source": "deterministic_seed", "preprocessing": "center_train_only"},
    {"id": "pca", "family": "pca", "information_set": "train_only", "parameter_source": "train_fit", "preprocessing": "center_train_only"},
    {"id": "ppca", "family": "ppca", "information_set": "train_only", "parameter_source": "train_fit", "preprocessing": "center_train_only"},
    {"id": "fa_selected", "family": "diagonal_fa", "information_set": "train_validation", "parameter_source": "selected_train_validation", "preprocessing": "center_train_only"},
    {"id": "true_gaussian_reference", "family": "true_parameter_gaussian", "information_set": "oracle_parameters", "parameter_source": "known_true", "preprocessing": "none"},
    {"id": "proper_scaled_pca", "family": "preprocessing_control", "information_set": "train_only", "parameter_source": "train_fit", "preprocessing": "standardize_train_only"},
    {"id": "leaked_scaled_pca_control", "family": "preprocessing_control", "information_set": "invalid_test_preprocessing", "parameter_source": "train_test_pooled", "preprocessing": "invalid_pooled_train_test"},
)

METRICS = (
    "heldout_reconstruction_mse",
    "latent_signal_reconstruction_mse",
    "heldout_gaussian_nll",
    "covariance_relative_frobenius",
    "principal_subspace_angle_max_degrees",
    "loading_subspace_angle_max_degrees",
    "uniqueness_rmse",
    "selected_validation_nll",
    "optimizer_iterations",
    "identification_applicable",
    "preprocessing_test_poison_parameter_change",
)

FIELDS = (
    "row_key", "scenario", "scenario_class", "replicate", "provenance", "method",
    "method_family", "information_set", "parameter_source", "preprocessing",
    "target", "metric", "value", "status", "reason", "seed_id", "n_train",
    "n_validation", "n_test", "signal_strength_1", "signal_strength_2",
    "residual_kind", "selected_floor", "selected_restart", "selected_converged",
)


def load_phase3():
    spec = importlib.util.spec_from_file_location("ldg_phase3_static_linear", PHASE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load stable Phase 3 implementation: {PHASE3_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P3 = load_phase3()


def scenario_code(scenario_id: str) -> int:
    return int.from_bytes(hashlib.sha256(scenario_id.encode("utf-8")).digest()[:4], "big")


def rng_for(scenario_id: str, replicate: int, stage: int, split: str = "train") -> np.random.Generator:
    split_code = {"train": 1, "validation": 2, "test": 3}[split]
    seed = np.random.SeedSequence((MASTER_SEED, scenario_code(scenario_id), replicate, stage, split_code))
    return np.random.Generator(np.random.PCG64(seed))


def seed_id(scenario_id: str, replicate: int) -> str:
    return f"pcg64:{MASTER_SEED}:{scenario_code(scenario_id)}:{replicate}"


def scenario_spec(scenario: dict) -> dict:
    basis = P3.base_basis()
    strengths = np.asarray(scenario["strengths"], dtype=float)
    loading = basis * strengths
    residual_kind = scenario["residual"]
    uniqueness = np.asarray(BASE_UNIQUENESS, dtype=float)
    diagonal_applicable = True
    if residual_kind == "homogeneous_diagonal":
        uniqueness = np.full(P, 0.45)
        residual = np.diag(uniqueness)
    elif residual_kind == "near_zero_diagonal":
        uniqueness = np.asarray((1e-5, 2e-5, 5e-5, 1e-4, 0.02, 0.04, 0.08, 0.12))
        residual = np.diag(uniqueness)
    elif residual_kind == "correlated":
        rho = 0.42
        corr = rho ** np.abs(np.subtract.outer(np.arange(P), np.arange(P)))
        residual = np.sqrt(uniqueness)[:, None] * corr * np.sqrt(uniqueness)[None, :]
        diagonal_applicable = False
    elif residual_kind == "high_nuisance":
        uniqueness = uniqueness.copy()
        uniqueness[-1] = 7.5
        residual = np.diag(uniqueness)
    elif residual_kind == "heterogeneous_diagonal":
        residual = np.diag(uniqueness)
    else:
        raise KeyError(residual_kind)
    outlier_fraction = float(scenario.get("outlier_fraction", 0.0))
    if outlier_fraction:
        diagonal_applicable = False
    return {
        "mean": np.zeros(P),
        "loading": loading,
        "uniqueness": np.diag(residual).copy(),
        "residual_covariance": residual,
        "nominal_covariance": loading @ loading.T + residual,
        "diagonal_identification_applicable": diagonal_applicable,
        "outlier_fraction": outlier_fraction,
        "outlier_scale": float(scenario.get("outlier_scale", 0.0)),
    }


def generate_split(scenario: dict, replicate: int, split: str) -> dict:
    """Generation stage: one independent split with retained scoring truth."""
    spec = scenario_spec(scenario)
    n = int(scenario[f"n_{split}"])
    latent = rng_for(scenario["id"], replicate, 10, split).normal(size=(n, K))
    residual = rng_for(scenario["id"], replicate, 11, split).multivariate_normal(
        np.zeros(P), spec["residual_covariance"], size=n, check_valid="raise"
    )
    signal = latent @ spec["loading"].T
    observed = spec["mean"] + signal + residual
    outlier_mask = np.zeros(n, dtype=bool)
    if spec["outlier_fraction"]:
        count = max(1, int(round(n * spec["outlier_fraction"])))
        order = rng_for(scenario["id"], replicate, 12, split).permutation(n)
        outlier_mask[order[:count]] = True
        directions = rng_for(scenario["id"], replicate, 13, split).normal(size=(count, P))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        observed[outlier_mask] += spec["outlier_scale"] * directions
    return {"observed": observed, "latent": latent, "signal": signal, "outlier_mask": outlier_mask}


def generate_replicate(scenario: dict, replicate: int) -> dict:
    return {split: generate_split(scenario, replicate, split) for split in ("train", "validation", "test")}


def restricted_select_fa(train_observations: np.ndarray, validation_observations: np.ndarray) -> tuple[dict, list[dict]]:
    """Selection stage: accepts train/validation observations and nothing else."""
    return P3.select_fa(np.asarray(train_observations), np.asarray(validation_observations))


def random_basis(scenario: dict, replicate: int) -> np.ndarray:
    return P3.orthonormal_columns(rng_for(scenario["id"], replicate, 40).normal(size=(P, K)))


def fit_scaled_pca(train: np.ndarray, scaler_data: np.ndarray) -> dict:
    scaler = P3.fit_standardizer(scaler_data)
    model = P3.fit_pca(P3.transform(train, scaler))
    return {"scaler": scaler, "model": model}


def fit_lawful_states(train_observations: np.ndarray, validation_observations: np.ndarray) -> dict:
    """Fit/select using train and validation observations only."""
    train = np.asarray(train_observations)
    validation = np.asarray(validation_observations)
    pca = P3.fit_pca(train)
    ppca = P3.fit_ppca(train)
    fa, candidates = restricted_select_fa(train, validation)
    proper_scaled = fit_scaled_pca(train, train)
    return {
        "train_mean": np.mean(train, axis=0),
        "pca": pca,
        "ppca": ppca,
        "fa_selected": fa,
        "fa_candidates": candidates,
        "proper_scaled_pca": proper_scaled,
    }


def assemble_frozen_states(scenario: dict, replicate: int, lawful_fitted: dict) -> dict:
    """Join lawful fitted states with predeclared fixed and oracle states."""
    return {
        **lawful_fitted,
        "fixed_basis": P3.fixed_basis(),
        "random_basis": random_basis(scenario, replicate),
        "reference_spec": scenario_spec(scenario),
    }


def recover_frozen_states(fitted: dict, test_observations: np.ndarray) -> dict:
    """Recover from frozen lawful fitted states plus test observations only."""
    test = np.asarray(test_observations)
    spec = fitted["reference_spec"]
    pca = fitted["pca"]
    ppca = fitted["ppca"]
    fa = fitted["fa_selected"]
    proper_scaled = fitted["proper_scaled_pca"]
    models = {
        "mean": {"reconstruction": np.broadcast_to(fitted["train_mean"], test.shape)},
        "fixed_projection": {"reconstruction": P3.projection_reconstruction(test, fitted["train_mean"], fitted["fixed_basis"]), "basis": fitted["fixed_basis"]},
        "random_projection": {"reconstruction": P3.projection_reconstruction(test, fitted["train_mean"], fitted["random_basis"]), "basis": fitted["random_basis"]},
        "pca": {"reconstruction": P3.projection_reconstruction(test, pca["mean"], pca["basis"]), "basis": pca["basis"]},
        "ppca": {
            "reconstruction": P3.posterior_reconstruction(test, ppca["mean"], ppca["loading"], ppca["noise_variance"] * np.eye(P)),
            "mean": ppca["mean"], "covariance": ppca["covariance"], "basis": ppca["basis"],
            "loading": ppca["loading"], "residual": ppca["noise_variance"] * np.eye(P),
        },
        "fa_selected": {
            "reconstruction": P3.posterior_reconstruction(test, fa["mean"], fa["loading"], np.diag(fa["uniqueness"])),
            "mean": fa["mean"], "covariance": fa["covariance"], "loading": fa["loading"],
            "residual": np.diag(fa["uniqueness"]), "uniqueness": fa["uniqueness"],
            "selected": fa,
        },
        "true_gaussian_reference": {
            "reconstruction": P3.posterior_reconstruction(test, spec["mean"], spec["loading"], spec["residual_covariance"]),
            "mean": spec["mean"], "covariance": spec["nominal_covariance"], "loading": spec["loading"],
            "residual": spec["residual_covariance"], "uniqueness": spec["uniqueness"],
        },
    }
    scaler, model = proper_scaled["scaler"], proper_scaled["model"]
    transformed = P3.transform(test, scaler)
    reconstructed_scaled = P3.projection_reconstruction(transformed, model["mean"], model["basis"])
    models["proper_scaled_pca"] = {
        "reconstruction": P3.inverse_transform(reconstructed_scaled, scaler),
        "basis_scaled": model["basis"], "scaler": scaler,
    }
    return models


def recover_invalid_leaked_control(train_observations: np.ndarray, test_observations: np.ndarray) -> dict:
    """Explicit invalid control that pools train and test during preprocessing."""
    train = np.asarray(train_observations)
    test = np.asarray(test_observations)
    fitted = fit_scaled_pca(train, np.vstack((train, test)))
    scaler, model = fitted["scaler"], fitted["model"]
    transformed = P3.transform(test, scaler)
    reconstructed_scaled = P3.projection_reconstruction(transformed, model["mean"], model["basis"])
    return {
        "reconstruction": P3.inverse_transform(reconstructed_scaled, scaler),
        "basis_scaled": model["basis"], "scaler": scaler, "invalid_fitted_state": fitted,
    }


def status_for_exception(exc: Exception) -> str:
    if isinstance(exc, (FloatingPointError, np.linalg.LinAlgError)):
        return "nonconvergence"
    return "invalid"


def target_for_metric(method: str, metric: str, scenario: dict) -> str:
    if metric == "heldout_reconstruction_mse":
        return "contaminated_observed_coordinate_reconstruction" if scenario["id"] == "outliers" else "observed_coordinate_reconstruction"
    if metric == "latent_signal_reconstruction_mse":
        return "true_common_signal"
    if metric == "heldout_gaussian_nll":
        return "contaminated_observation_gaussian_score" if scenario["id"] == "outliers" else "observable_distribution"
    if metric == "covariance_relative_frobenius":
        return "nominal_clean_gaussian_covariance" if scenario["id"] == "outliers" else "observable_covariance"
    if metric == "principal_subspace_angle_max_degrees":
        return "population_principal_subspace"
    if metric == "loading_subspace_angle_max_degrees":
        return "identified_true_common_loading_subspace"
    if metric == "uniqueness_rmse":
        return "identified_diagonal_uniqueness"
    if metric in ("selected_validation_nll", "optimizer_iterations"):
        return "bounded_fa_selection"
    if metric == "identification_applicable":
        return "diagonal_fa_identification_applicability"
    if metric == "preprocessing_test_poison_parameter_change":
        return "preprocessing_leakage_diagnostic"
    raise KeyError(metric)


def provenance_for_metric(metric: str) -> str:
    if metric in ("heldout_reconstruction_mse", "latent_signal_reconstruction_mse", "heldout_gaussian_nll"):
        return "test"
    if metric == "selected_validation_nll":
        return "validation"
    if metric in ("covariance_relative_frobenius", "principal_subspace_angle_max_degrees",
                  "loading_subspace_angle_max_degrees", "uniqueness_rmse", "optimizer_iterations"):
        return "fit_diagnostic"
    if metric == "identification_applicable":
        return "scenario_diagnostic"
    if metric == "preprocessing_test_poison_parameter_change":
        return "leakage_diagnostic"
    raise KeyError(metric)


def applicability(method: str, metric: str, scenario: dict, identification_applicable: bool) -> tuple[bool, str]:
    preprocessing = method in ("proper_scaled_pca", "leaked_scaled_pca_control")
    if preprocessing and scenario["id"] != "preprocessing_leakage":
        return False, "preprocessing control is predeclared only for preprocessing_leakage"
    if metric in ("heldout_reconstruction_mse", "latent_signal_reconstruction_mse"):
        return True, ""
    if metric in ("heldout_gaussian_nll", "covariance_relative_frobenius"):
        return (method in ("ppca", "fa_selected", "true_gaussian_reference"),
                "method does not define a full Gaussian covariance model")
    if metric == "principal_subspace_angle_max_degrees":
        return (method in ("pca", "ppca"), "principal-subspace target is reserved for PCA and PPCA")
    if metric in ("loading_subspace_angle_max_degrees", "uniqueness_rmse"):
        if method not in ("fa_selected", "true_gaussian_reference"):
            return False, "method is not assigned the identified diagonal-FA target"
        if not identification_applicable:
            return False, "diagonal_FA_identification_assumptions_do_not_apply"
        return True, ""
    if metric in ("selected_validation_nll", "optimizer_iterations"):
        return (method == "fa_selected", "method does not perform bounded FA selection")
    if metric == "identification_applicable":
        return (method in ("fa_selected", "true_gaussian_reference"),
                "identification applicability is specific to diagonal FA targets")
    if metric == "preprocessing_test_poison_parameter_change":
        return (scenario["id"] == "preprocessing_leakage" and preprocessing,
                "metric is reserved for preprocessing leakage controls")
    raise KeyError(metric)


def row_base(scenario: dict, replicate: int, method_record: dict, selected: dict) -> dict:
    return {
        "scenario": scenario["id"], "scenario_class": scenario["class"], "replicate": replicate,
        "method": method_record["id"], "method_family": method_record["family"],
        "information_set": method_record["information_set"], "parameter_source": method_record["parameter_source"],
        "preprocessing": method_record["preprocessing"], "seed_id": seed_id(scenario["id"], replicate),
        "n_train": scenario["n_train"], "n_validation": scenario["n_validation"], "n_test": scenario["n_test"],
        "signal_strength_1": scenario["strengths"][0], "signal_strength_2": scenario["strengths"][1],
        "residual_kind": scenario["residual"], "selected_floor": selected.get("floor", ""),
        "selected_restart": selected.get("restart", ""), "selected_converged": selected.get("converged", ""),
    }


def add_row(rows: list[dict], base: dict, scenario: dict, metric: str, value: float | None,
            status: str = "ok", reason: str = "") -> None:
    if status not in STATUS_VALUES:
        raise ValueError(status)
    row = dict(base)
    row["target"] = target_for_metric(row["method"], metric, scenario)
    row["provenance"] = provenance_for_metric(metric)
    if status == "ok":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if not np.isfinite(numeric):
            status, reason, value = "invalid", "nonfinite_metric_value", None
        else:
            value = numeric
    row.update({"metric": metric, "value": "" if value is None else float(value), "status": status, "reason": reason})
    row["row_key"] = "|".join((row["scenario"], str(row["replicate"]), row["provenance"], row["method"], metric))
    rows.append(row)


def preprocessing_poison_change(method: str, train: np.ndarray, test: np.ndarray) -> float:
    poisoned = test.copy()
    poisoned[:, 0] += 50.0
    poisoned[:, 1] *= 7.0
    if method == "proper_scaled_pca":
        left = fit_scaled_pca(train, train)
        right = fit_scaled_pca(train, train)
    elif method == "leaked_scaled_pca_control":
        left = fit_scaled_pca(train, np.vstack((train, test)))
        right = fit_scaled_pca(train, np.vstack((train, poisoned)))
    else:
        raise KeyError(method)
    discrepancies = [
        np.max(np.abs(left["scaler"]["mean"] - right["scaler"]["mean"])),
        np.max(np.abs(left["scaler"]["scale"] - right["scaler"]["scale"])),
        np.max(np.abs(left["model"]["basis"] @ left["model"]["basis"].T - right["model"]["basis"] @ right["model"]["basis"].T)),
    ]
    return float(max(discrepancies))


def score_method(rows: list[dict], scenario: dict, replicate: int, method_record: dict,
                 selected: dict, model: dict | None, exception: Exception | None,
                 test_observed: np.ndarray, test_signal: np.ndarray, train_observed: np.ndarray) -> None:
    """Scoring stage: joins retained signal truth after inference is frozen."""
    method = method_record["id"]
    base = row_base(scenario, replicate, method_record, selected)
    spec = scenario_spec(scenario)
    rank_pass = bool(P3.anderson_rubin_rank_diagnostic(spec["loading"])["overall_pass"])
    identification_applicable = bool(spec["diagonal_identification_applicable"] and rank_pass)
    fa_nonconverged = method == "fa_selected" and not bool(selected.get("converged", False))
    reconstruction = np.asarray(model.get("reconstruction")) if model is not None and "reconstruction" in model else None
    reconstruction_valid = bool(
        reconstruction is not None and reconstruction.shape == test_observed.shape and np.all(np.isfinite(reconstruction))
    )
    pop_basis = P3.population_principal_basis({"nominal_covariance": spec["nominal_covariance"]})
    for metric in METRICS:
        try:
            is_applicable, inapplicable_reason = applicability(method, metric, scenario, identification_applicable)
            if not is_applicable:
                add_row(rows, base, scenario, metric, None, "inapplicable", inapplicable_reason)
                continue
            if metric == "identification_applicable":
                add_row(rows, base, scenario, metric, 1.0 if identification_applicable else 0.0)
                continue
            if exception is not None:
                add_row(rows, base, scenario, metric, None, status_for_exception(exception), f"{type(exception).__name__}: {exception}")
                continue
            if model is None:
                add_row(rows, base, scenario, metric, None, "invalid", "missing_inference_result")
                continue
            if fa_nonconverged and metric == "optimizer_iterations":
                add_row(rows, base, scenario, metric, selected["iterations"], "nonconvergence", "bounded_fa_optimizer_did_not_converge")
                continue
            if fa_nonconverged:
                add_row(rows, base, scenario, metric, None, "nonconvergence", "bounded_fa_optimizer_did_not_converge")
            elif metric == "heldout_reconstruction_mse":
                if not reconstruction_valid:
                    add_row(rows, base, scenario, metric, None, "invalid", "nonfinite_or_malformed_reconstruction")
                else:
                    add_row(rows, base, scenario, metric, np.mean((test_observed - reconstruction) ** 2))
            elif metric == "latent_signal_reconstruction_mse":
                if not reconstruction_valid:
                    add_row(rows, base, scenario, metric, None, "invalid", "nonfinite_or_malformed_reconstruction")
                else:
                    centered = reconstruction - spec["mean"]
                    add_row(rows, base, scenario, metric, np.mean((test_signal - centered) ** 2))
            elif metric == "heldout_gaussian_nll":
                value = P3.gaussian_nll(test_observed, model["mean"], model["covariance"])
                add_row(rows, base, scenario, metric, value)
            elif metric == "covariance_relative_frobenius":
                add_row(rows, base, scenario, metric, P3.relative_frobenius(model["covariance"], spec["nominal_covariance"]))
            elif metric == "principal_subspace_angle_max_degrees":
                add_row(rows, base, scenario, metric, np.max(P3.principal_angles(model["basis"], pop_basis)))
            elif metric == "loading_subspace_angle_max_degrees":
                add_row(rows, base, scenario, metric, np.max(P3.principal_angles(model["loading"], spec["loading"])))
            elif metric == "uniqueness_rmse":
                add_row(rows, base, scenario, metric, np.sqrt(np.mean((model["uniqueness"] - spec["uniqueness"]) ** 2)))
            elif metric == "selected_validation_nll":
                add_row(rows, base, scenario, metric, selected["validation_nll"])
            elif metric == "optimizer_iterations":
                add_row(rows, base, scenario, metric, selected["iterations"])
            elif metric == "preprocessing_test_poison_parameter_change":
                add_row(rows, base, scenario, metric, preprocessing_poison_change(method, train_observed, test_observed))
            else:
                raise KeyError(metric)
        except Exception as exc:  # scoring failures must become explicit rows
            add_row(rows, base, scenario, metric, None, status_for_exception(exc), f"scoring_{type(exc).__name__}: {exc}")


def evaluate_replicate(scenario: dict, replicate: int) -> tuple[list[dict], dict]:
    data = generate_replicate(scenario, replicate)
    train = data["train"]["observed"]
    validation = data["validation"]["observed"]
    test = data["test"]["observed"]
    try:
        lawful_fitted = fit_lawful_states(train, validation)
        fitted = assemble_frozen_states(scenario, replicate, lawful_fitted)
        models = recover_frozen_states(fitted, test)
        models["leaked_scaled_pca_control"] = recover_invalid_leaked_control(train, test)
        candidates = fitted["fa_candidates"]
        selected = fitted["fa_selected"]
        inference_exception = None
    except Exception as exc:
        models, candidates, selected, inference_exception = {}, [], {}, exc
    rows = []
    for method in METHODS:
        model = models.get(method["id"])
        exception = inference_exception if method["id"] in ("pca", "ppca", "fa_selected") else None
        score_method(rows, scenario, replicate, method, selected, model, exception,
                     test, data["test"]["signal"], train)
    selection_record = {
        "scenario": scenario["id"], "replicate": replicate,
        "selected": {key: selected.get(key) for key in ("floor", "restart", "converged", "iterations", "train_nll", "validation_nll")},
        "candidates": candidates,
    }
    return rows, selection_record


def canonical_key(row: dict) -> tuple:
    return (row["scenario"], int(row["replicate"]), row["provenance"], row["method"], row["metric"])


def format_number(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("nonfinite serialized numeric")
    return format(numeric, ".17g")


def serialized_row(row: dict) -> dict:
    output = {}
    numeric = {"replicate", "n_train", "n_validation", "n_test", "signal_strength_1", "signal_strength_2",
               "selected_floor", "selected_restart", "value"}
    for field in FIELDS:
        value = row.get(field, "")
        output[field] = format_number(value) if field in numeric else str(value).lower() if field == "selected_converged" and isinstance(value, bool) else str(value)
    return output


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((serialized_row(row) for row in rows), key=canonical_key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"results schema mismatch: {reader.fieldnames}")
        rows = list(reader)
    if any(set(row) != set(FIELDS) for row in rows):
        raise ValueError("result row has incomplete or unknown fields")
    return rows


def validate_result_row(row: dict) -> None:
    if set(row) != set(FIELDS):
        raise ValueError("field set mismatch")
    expected_key = "|".join((row["scenario"], row["replicate"], row["provenance"], row["method"], row["metric"]))
    if row["row_key"] != expected_key:
        raise ValueError("row_key mismatch")
    if row["scenario"] not in {s["id"] for s in SCENARIOS}:
        raise ValueError("unknown scenario")
    if row["method"] not in {m["id"] for m in METHODS}:
        raise ValueError("unknown method")
    if row["metric"] not in METRICS:
        raise ValueError("unknown metric")
    if row["status"] not in STATUS_VALUES:
        raise ValueError("unknown status")
    if row["status"] == "ok":
        if row["reason"] or row["value"] == "" or not math.isfinite(float(row["value"])):
            raise ValueError("invalid ok value/reason")
    elif not row["reason"] or row["value"] != "" and row["status"] == "inapplicable":
        raise ValueError("invalid non-ok value/reason")
    if row["status"] == "nonconvergence" and row["value"] != "" and row["metric"] != "optimizer_iterations":
        raise ValueError("nonconvergence value is retained only for optimizer_iterations")


def validate_resume_rows(prior_rows: list[dict], fresh_rows: list[dict]) -> dict[str, dict]:
    fresh = {serialized_row(row)["row_key"]: serialized_row(row) for row in fresh_rows}
    if len(fresh) != len(fresh_rows):
        raise RuntimeError("fresh rows contain duplicate keys")
    accepted = {}
    for row in prior_rows:
        validate_result_row(row)
        key = row["row_key"]
        if key in accepted:
            raise ValueError(f"duplicate resume key: {key}")
        if key not in fresh:
            raise ValueError(f"unknown resume key: {key}")
        if row != fresh[key]:
            raise ValueError(f"stale or corrupted resume row: {key}")
        accepted[key] = row
    return accepted


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["method"], row["metric"]), []).append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        values = np.asarray([float(row["value"]) for row in group if row["status"] == "ok"], dtype=float)
        counts = {status: sum(row["status"] == status for row in group) for status in STATUS_VALUES}
        output.append({
            "scenario": key[0], "method": key[1], "metric": key[2], "n_rows": len(group), "n_ok": len(values),
            "q10": None if not len(values) else float(np.quantile(values, 0.10)),
            "median": None if not len(values) else float(np.quantile(values, 0.50)),
            "q90": None if not len(values) else float(np.quantile(values, 0.90)),
            "ok_rate": counts["ok"] / len(group),
            "failure_rate": (counts["invalid"] + counts["nonconvergence"]) / len(group),
            "applicability_rate": 1.0 - counts["inapplicable"] / len(group),
            "status_counts": counts,
        })
    return output


def tree_max_abs(left, right) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max((tree_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((tree_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape:
            return float("inf")
        return float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
    if isinstance(left, (bool, str)) or isinstance(right, (bool, str)):
        return 0.0 if left == right else float("inf")
    return abs(float(left) - float(right))


def lawful_fitted_snapshot(fitted: dict) -> dict:
    return {
        "pca": fitted["pca"],
        "ppca": fitted["ppca"],
        "fa_selected": fitted["fa_selected"],
        "fa_candidates": fitted["fa_candidates"],
        "proper_scaled_pca": fitted["proper_scaled_pca"],
    }


def fixed_recovery_snapshot(models: dict) -> dict:
    return {method: {"reconstruction": models[method]["reconstruction"]}
            for method in ("pca", "ppca", "fa_selected", "proper_scaled_pca")}


def poison_diagnostics() -> dict:
    scenario, replicate = SCENARIOS[0], 0
    data = generate_replicate(scenario, replicate)
    train = data["train"]["observed"]
    validation = data["validation"]["observed"]
    fixed_evaluation = data["test"]["observed"].copy()
    base_fitted = fit_lawful_states(train, validation)
    base_recovery = recover_frozen_states(assemble_frozen_states(scenario, replicate, base_fitted), fixed_evaluation)

    test_poisoned = generate_replicate(scenario, replicate)
    test_poisoned["test"]["observed"] = test_poisoned["test"]["observed"].copy()
    test_poisoned["test"]["observed"][:, 0] += 1e6
    test_fitted = fit_lawful_states(test_poisoned["train"]["observed"], test_poisoned["validation"]["observed"])
    test_fixed_recovery = recover_frozen_states(assemble_frozen_states(scenario, replicate, test_fitted), fixed_evaluation)

    latent_poisoned = generate_replicate(scenario, replicate)
    for bundle in latent_poisoned.values():
        bundle["latent"] = np.full_like(bundle["latent"], 1e9)
        bundle["signal"] = np.full_like(bundle["signal"], -1e9)
    latent_fitted = fit_lawful_states(latent_poisoned["train"]["observed"], latent_poisoned["validation"]["observed"])
    latent_fixed_recovery = recover_frozen_states(assemble_frozen_states(scenario, replicate, latent_fitted), fixed_evaluation)

    base_fit_snapshot = lawful_fitted_snapshot(base_fitted)
    base_recovery_snapshot = fixed_recovery_snapshot(base_recovery)
    return {
        "selection_signature": str(inspect.signature(restricted_select_fa)),
        "lawful_fit_signature": str(inspect.signature(fit_lawful_states)),
        "frozen_recovery_signature": str(inspect.signature(recover_frozen_states)),
        "invalid_leaked_control_signature": str(inspect.signature(recover_invalid_leaked_control)),
        "lawful_fit_excludes_test_and_truth": "test" not in str(inspect.signature(fit_lawful_states)) and "latent" not in str(inspect.signature(fit_lawful_states)),
        "frozen_recovery_excludes_train_validation_truth": all(token not in str(inspect.signature(recover_frozen_states)) for token in ("train", "validation", "latent", "signal")),
        "test_observation_poison_complete_fitted_state_max_abs": tree_max_abs(base_fit_snapshot, lawful_fitted_snapshot(test_fitted)),
        "test_observation_poison_fixed_recovery_max_abs": tree_max_abs(base_recovery_snapshot, fixed_recovery_snapshot(test_fixed_recovery)),
        "latent_truth_poison_complete_fitted_state_max_abs": tree_max_abs(base_fit_snapshot, lawful_fitted_snapshot(latent_fitted)),
        "latent_truth_poison_fixed_recovery_max_abs": tree_max_abs(base_recovery_snapshot, fixed_recovery_snapshot(latent_fixed_recovery)),
        "poisoned_test_changed": bool(np.max(np.abs(test_poisoned["test"]["observed"] - fixed_evaluation)) > 1e5),
        "poisoned_truth_changed": bool(np.max(np.abs(latent_poisoned["test"]["signal"] - data["test"]["signal"])) > 1e8),
    }


def true_reference_diagnostics() -> dict:
    scenario, replicate = SCENARIOS[0], 0
    data = generate_replicate(scenario, replicate)
    lawful_fitted = fit_lawful_states(data["train"]["observed"], data["validation"]["observed"])
    fitted = assemble_frozen_states(scenario, replicate, lawful_fitted)
    returned = recover_frozen_states(fitted, data["test"]["observed"])["true_gaussian_reference"]
    truth = scenario_spec(scenario)
    direct_covariance = truth["loading"] @ truth["loading"].T + truth["residual_covariance"]
    centered = data["test"]["observed"] - truth["mean"]
    direct_reconstruction = truth["mean"] + centered @ np.linalg.solve(direct_covariance, truth["loading"]) @ truth["loading"].T
    return {
        "returned_mean_max_abs": float(np.max(np.abs(returned["mean"] - truth["mean"]))),
        "returned_loading_max_abs": float(np.max(np.abs(returned["loading"] - truth["loading"]))),
        "returned_residual_max_abs": float(np.max(np.abs(returned["residual"] - truth["residual_covariance"]))),
        "returned_uniqueness_max_abs": float(np.max(np.abs(returned["uniqueness"] - np.diag(truth["residual_covariance"])))),
        "returned_covariance_vs_direct_max_abs": float(np.max(np.abs(returned["covariance"] - direct_covariance))),
        "returned_reconstruction_vs_direct_max_abs": float(np.max(np.abs(returned["reconstruction"] - direct_reconstruction))),
    }


def scoring_fault_diagnostics() -> dict:
    scenario, replicate = SCENARIOS[0], 0
    data = generate_replicate(scenario, replicate)
    lawful_fitted = fit_lawful_states(data["train"]["observed"], data["validation"]["observed"])
    fitted = assemble_frozen_states(scenario, replicate, lawful_fitted)
    models = recover_frozen_states(fitted, data["test"]["observed"])
    selected = lawful_fitted["fa_selected"]
    method = next(m for m in METHODS if m["id"] == "ppca")
    broken = dict(models["ppca"])
    broken["reconstruction"] = broken["reconstruction"].copy()
    broken["reconstruction"][0, 0] = np.inf
    broken["covariance"] = broken["covariance"].copy()
    broken["covariance"][0, 0] = np.nan
    rows = []
    with np.errstate(all="ignore"):
        score_method(rows, scenario, replicate, method, selected, broken, None, data["test"]["observed"], data["test"]["signal"], data["train"]["observed"])
    missing = []
    score_method(missing, scenario, replicate, method, selected, None, None, data["test"]["observed"], data["test"]["signal"], data["train"]["observed"])
    invalid_exception = []
    score_method(invalid_exception, scenario, replicate, method, selected, None, ValueError("injected_invalid"),
                 data["test"]["observed"], data["test"]["signal"], data["train"]["observed"])
    nonconvergence_exception = []
    score_method(nonconvergence_exception, scenario, replicate, method, selected, None, np.linalg.LinAlgError("injected_nonconvergence"),
                 data["test"]["observed"], data["test"]["signal"], data["train"]["observed"])
    by_metric = {row["metric"]: row for row in rows}
    applicable_metrics = [metric for metric in METRICS if applicability("ppca", metric, scenario, True)[0]]
    return {
        "nonfinite_reconstruction_status": by_metric["heldout_reconstruction_mse"]["status"],
        "nonfinite_covariance_nll_status": by_metric["heldout_gaussian_nll"]["status"],
        "nonfinite_covariance_error_status": by_metric["covariance_relative_frobenius"]["status"],
        "missing_result_applicable_invalid_inapplicable_preserved": bool(all(
            row["status"] == ("invalid" if row["metric"] in applicable_metrics else "inapplicable") for row in missing
        )),
        "inference_invalid_exception_applicable_invalid_inapplicable_preserved": bool(all(
            row["status"] == ("invalid" if row["metric"] in applicable_metrics else "inapplicable") for row in invalid_exception
        )),
        "inference_linalg_exception_applicable_nonconvergence_inapplicable_preserved": bool(all(
            row["status"] == ("nonconvergence" if row["metric"] in applicable_metrics else "inapplicable") for row in nonconvergence_exception
        )),
        "machine_readable_reasons_present": bool(all(row["reason"] for row in rows if row["status"] in ("invalid", "nonconvergence"))),
    }


def verification_diagnostics() -> dict:
    scenario, replicate = next(s for s in SCENARIOS if s["id"] == "preprocessing_leakage"), 0
    data = generate_replicate(scenario, replicate)
    train, test = data["train"]["observed"], data["test"]["observed"]
    proper = preprocessing_poison_change("proper_scaled_pca", train, test)
    leaked = preprocessing_poison_change("leaked_scaled_pca_control", train, test)
    return {
        "capability_and_poison": poison_diagnostics(),
        "preprocessing_leakage": {
            "proper_train_only_test_poison_parameter_change": proper,
            "invalid_pooled_test_poison_parameter_change": leaked,
        },
        "true_parameter_reference": true_reference_diagnostics(),
        "scoring_fault_injection": scoring_fault_diagnostics(),
    }


def make_summary(path: Path, summaries: list[dict], checks: dict) -> None:
    width, height = 1500, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), "P4-U02: Static linear latent robustness (median with q10-q90)", fill="black", font=font)
    boxes = [(25, 55, 735, 475), (765, 55, 1475, 475), (25, 505, 735, 925), (765, 505, 1475, 925)]
    lookup = {(s["scenario"], s["method"], s["metric"]): s for s in summaries}
    colors = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75)]

    def panel(box, title, scenario, metric, methods, footer):
        draw.rectangle(box, outline="black", width=2)
        draw.text((box[0]+10, box[1]+8), title, fill="black", font=font)
        records = [(m, lookup.get((scenario, m, metric))) for m in methods]
        records = [(m, r) for m, r in records if r and r["median"] is not None]
        max_value = max((r["q90"] for _, r in records), default=1.0) * 1.12
        x0, y0, x1, y1 = box[0]+175, box[1]+45, box[2]-25, box[3]-35
        for i, (method, record) in enumerate(records):
            y = y0 + (i + 0.5) * (y1-y0) / max(len(records), 1)
            scale = (x1-x0) / max(max_value, 1e-12)
            draw.text((box[0]+10, y-6), method, fill="black", font=font)
            draw.line((x0+record["q10"]*scale, y, x0+record["q90"]*scale, y), fill=colors[i % len(colors)], width=4)
            x = x0+record["median"]*scale
            draw.ellipse((x-4, y-4, x+4, y+4), fill=colors[i % len(colors)])
        draw.text((box[0]+10, box[3]-20), footer, fill="black", font=font)

    panel(boxes[0], "A. Shared observed-reconstruction target: reference", "reference", "heldout_reconstruction_mse",
          ["mean", "fixed_projection", "random_projection", "pca", "ppca", "fa_selected"],
          "One target only; true-parameter oracle is not ranked here.")
    panel(boxes[1], "B. Population principal-subspace target: weak separation", "weak_separation", "principal_subspace_angle_max_degrees",
          ["pca", "ppca"], "PCA/PPCA target only; not the FA loading target.")
    panel(boxes[2], "C. Identified loading-subspace target: reference", "reference", "loading_subspace_angle_max_degrees",
          ["fa_selected", "true_gaussian_reference"], "Reference is a correctness anchor, not a deployable winner.")
    box = boxes[3]
    draw.rectangle(box, outline="black", width=2)
    draw.text((box[0]+10, box[1]+8), "D. Leakage and applicability diagnostics", fill="black", font=font)
    leakage = checks["preprocessing_leakage"]
    lines = [
        "Train-only and invalid pooled preprocessing are not peers.",
        f"proper scaler change after test poison: {leakage['proper_train_only_test_poison_parameter_change']:.4g}",
        f"invalid pooled scaler/model change: {leakage['invalid_pooled_test_poison_parameter_change']:.4g}",
        "Correlated residuals and outliers invalidate diagonal-FA",
        "uniqueness/loading identification metrics; rows remain explicit.",
        "No unlike targets are pooled; no universal winner is computed.",
    ]
    for i, line in enumerate(lines):
        draw.text((box[0]+20, box[1]+60+i*48), line, fill="black", font=font)
    draw.text((box[0]+10, box[3]-20), "q10/median/q90 use four independent replicates; descriptive only.", fill="black", font=font)
    image.save(path, format="PNG", optimize=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, reverse: bool = False, resume: bool = False, quiet: bool = False) -> dict:
    start = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    work = [(scenario, replicate) for scenario in SCENARIOS for replicate in range(REPLICATES)]
    if reverse:
        work.reverse()
    fresh_rows, selections = [], []
    for scenario, replicate in work:
        rows, selection = evaluate_replicate(scenario, replicate)
        fresh_rows.extend(rows)
        selections.append(selection)
    prior_rows = read_results(root / "results.csv") if resume else []
    prior = validate_resume_rows(prior_rows, fresh_rows) if resume else {}
    rows = [prior.get(serialized_row(row)["row_key"], row) for row in fresh_rows]
    write_results(root / "results.csv", rows)
    canonical_rows = read_results(root / "results.csv")
    summaries = summarize(canonical_rows)
    checks = verification_diagnostics()
    status_counts = {status: sum(row["status"] == status for row in canonical_rows) for status in STATUS_VALUES}
    nonconverged_rows = [row for row in canonical_rows if row["status"] == "nonconvergence"]
    nonconvergence_audit = {
        "row_count": len(nonconverged_rows),
        "all_selected_fa": bool(all(row["method"] == "fa_selected" for row in nonconverged_rows)),
        "no_identification_applicability_rows": bool(all(row["metric"] != "identification_applicable" for row in nonconverged_rows)),
        "by_scenario_metric": [
            {"scenario": scenario, "metric": metric, "count": sum(
                row["scenario"] == scenario and row["metric"] == metric for row in nonconverged_rows
            )}
            for scenario, metric in sorted({(row["scenario"], row["metric"]) for row in nonconverged_rows})
        ],
    }
    diagnostics = {
        "schema_version": 1,
        "summary_grouping": ["scenario", "method", "metric"],
        "quantiles": [0.10, 0.50, 0.90],
        "summaries": summaries,
        "status_counts": status_counts,
        "nonconvergence_applicability_audit": nonconvergence_audit,
        "selection_by_replicate": sorted(selections, key=lambda x: (x["scenario"], x["replicate"])),
        "verification_checks": checks,
        "scientific_limits": [
            "Results are conditional on this rank-two Gaussian simulator, bounded scenarios, method-specific targets, and four replicates.",
            "PCA and PPCA are scored against the population covariance principal subspace; diagonal FA is scored against the loading subspace only where its identification assumptions apply.",
            "Correlated residuals and outlier contamination make diagonal-uniqueness identification metrics inapplicable rather than failed recovery scores.",
            "The bounded FA EM search is not a global optimization guarantee; nonconvergence is a computational label.",
            "The pooled train-test preprocessing control is intentionally invalid and is not a candidate method.",
            "No result establishes mechanism, causality, intrinsic dimension, a unique latent generator, real-data validity, or universal superiority.",
        ],
    }
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    make_summary(root / "summary.png", summaries, checks)
    hashes = {name: sha256(root / name) for name in ARTIFACT_NAMES}
    manifest = {
        "schema_version": 1,
        "object_id": "P4-U02",
        "artifact": "static-linear-latent-robustness",
        "date": "2026-08-20",
        "maturity": "L1",
        "status": "stable",
        "scientific_question": "Where are bounded static rank-two latent recovery methods applicable, robust, or brittle under declared synthetic scenarios and method-specific targets?",
        "ground_truth": "x = mean + Lambda z + epsilon, z~N(0,I_2), with declared residual and contamination stresses",
        "phase3_reuse": str(PHASE3_PATH.relative_to(REPO_ROOT)),
        "phase3_sha256": sha256(PHASE3_PATH),
        "stages": ["generation", "selection", "inference", "scoring"],
        "split_policy": "independent train, validation, and test samples per replicate; test latent/signal truth is scoring-only",
        "master_seed": MASTER_SEED,
        "seed_mapping": "PCG64(SeedSequence(master_seed, sha256(scenario_id)[:4], replicate, stage, split_code))",
        "replicates": REPLICATES,
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "metrics": list(METRICS),
        "status_vocabulary": list(STATUS_VALUES),
        "selection": {
            "method": "bounded diagonal-noise FA EM imported from stable Phase 3",
            "fit_data": "train observations only", "selection_data": "validation observations only",
            "test_inputs_allowed": False, "latent_truth_allowed": False,
            "candidate_floors": list(P3.FA_FLOORS), "candidate_restarts": list(P3.FA_RESTARTS),
            "max_iterations": P3.FA_MAX_ITER, "relative_nll_tolerance": P3.FA_TOL,
        },
        "capability_separation": {
            "lawful_fit": "fit_lawful_states receives train observations and validation observations only",
            "frozen_recovery": "recover_frozen_states receives frozen fitted/predeclared states and test observations only",
            "invalid_control": "recover_invalid_leaked_control is a separately named negative control that intentionally pools train and test during preprocessing",
        },
        "identification_rule": "loading-subspace and uniqueness recovery rows are applicable only for diagonal residual scenarios with a passing stable Phase 3 Anderson-Rubin-style sufficient row-deletion rank diagnostic",
        "provenance_vocabulary": ["test", "validation", "fit_diagnostic", "scenario_diagnostic", "leakage_diagnostic"],
        "metric_provenance": {metric: provenance_for_metric(metric) for metric in METRICS},
        "canonical_row_order": ["scenario", "replicate", "provenance", "method", "metric"],
        "row_key": ["scenario", "replicate", "provenance", "method", "metric"],
        "aggregation": {
            "within_replicate": "each metric pools the declared held-out test samples for one scenario replicate",
            "across_replicates": "q10, median, and q90 across four independent replicate-level values",
            "rates": "ok, failure=(invalid+nonconvergence), and applicability=(1-inapplicable) rates retain method and metric strata",
        },
        "resume": "partial rows must match schema, key, metadata, status/value semantics, and freshly recomputed canonical bytes exactly; duplicate, unknown, stale, corrupted, or incomplete rows are rejected",
        "failure_label_semantics": "method×metric×scenario applicability is resolved before convergence; inapplicable rows have no value; nonconvergence is assigned only to applicable selected-FA outputs, and optimizer_iterations retains the reached iteration count",
        "tolerances": TOL,
        "artifact_files": ["manifest.json", *ARTIFACT_NAMES],
        "files_sha256": hashes,
        "commands": {
            "python_executable": sys.executable,
            "generate": f"{sys.executable} benchmarks/static-linear-latent-robustness.py --generate",
            "verify": f"{sys.executable} benchmarks/static-linear-latent-robustness.py --verify",
            "resume": f"{sys.executable} benchmarks/static-linear-latent-robustness.py --generate --resume",
        },
        "runtime": {"python_executable": sys.executable, "python": sys.version.split()[0], "numpy": np.__version__, "pillow": getattr(sys.modules.get("PIL"), "__version__", "unknown"), "platform": platform.platform()},
        "interpretation_boundary": "Scenario/target/metric conditional; not mechanism, causality, intrinsic dimension, real-data validity, or universal superiority.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - start
    result = {"artifact_root": str(root), "rows": len(rows), "runtime_seconds": elapsed, "hashes": hashes}
    if not quiet:
        print(json.dumps(result, indent=2))
    return result


def compare_roots(left: Path, right: Path) -> list[str]:
    return [name for name in ("manifest.json", *ARTIFACT_NAMES) if (left / name).read_bytes() != (right / name).read_bytes()]


def recompute_summaries(rows: list[dict]) -> list[dict]:
    return summarize(rows)


def validate_artifacts(root: Path) -> list[str]:
    errors = []
    expected = {"manifest.json", *ARTIFACT_NAMES}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        return [f"artifact set mismatch: expected {sorted(expected)}, got {sorted(actual)}"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    rows = read_results(root / "results.csv")
    for row in rows:
        try:
            validate_result_row(row)
        except (ValueError, TypeError) as exc:
            errors.append(f"invalid row {row.get('row_key', '<missing>')}: {exc}")
    for name, digest in manifest.get("files_sha256", {}).items():
        if sha256(root / name) != digest:
            errors.append(f"hash mismatch: {name}")
    if manifest.get("phase3_sha256") != sha256(PHASE3_PATH):
        errors.append("stable Phase 3 source hash mismatch")
    if manifest.get("metric_provenance") != {metric: provenance_for_metric(metric) for metric in METRICS}:
        errors.append("manifest metric provenance mismatch")
    if manifest.get("row_key") != ["scenario", "replicate", "provenance", "method", "metric"]:
        errors.append("manifest row-key schema mismatch")
    keys = [row["row_key"] for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("row keys are not unique")
    if rows != sorted(rows, key=canonical_key):
        errors.append("results are not in canonical order")
    expected_keys = {
        f"{scenario['id']}|{replicate}|{provenance_for_metric(metric)}|{method['id']}|{metric}"
        for scenario in SCENARIOS for replicate in range(REPLICATES) for method in METHODS for metric in METRICS
    }
    if set(keys) != expected_keys:
        errors.append(f"raw-row Cartesian product mismatch: missing={len(expected_keys-set(keys))}, extra={len(set(keys)-expected_keys)}")
    required_classes = {"sample_size", "signal_eigenvalue_separation", "heterogeneous_uniqueness", "near_zero_uniqueness", "correlated_residuals", "high_nuisance", "outliers", "preprocessing_leakage"}
    represented = {row["scenario_class"] for row in rows}
    if not required_classes.issubset(represented):
        errors.append(f"scenario classes missing: {sorted(required_classes-represented)}")
    required_families = {"mean", "fixed_projection", "random_projection", "pca", "ppca", "diagonal_fa", "true_parameter_gaussian"}
    families = {row["method_family"] for row in rows}
    if not required_families.issubset(families):
        errors.append(f"method families missing: {sorted(required_families-families)}")
    if diagnostics.get("summaries") != recompute_summaries(rows):
        errors.append("diagnostic summaries do not match raw rows")
    counts = {status: sum(row["status"] == status for row in rows) for status in STATUS_VALUES}
    if diagnostics.get("status_counts") != counts:
        errors.append("status counts do not match raw rows")
    if any(row["provenance"] != provenance_for_metric(row["metric"]) for row in rows):
        errors.append("metric provenance mismatch")
    scenario_lookup = {scenario["id"]: scenario for scenario in SCENARIOS}
    for row in rows:
        scenario = scenario_lookup[row["scenario"]]
        identification = bool(scenario_spec(scenario)["diagonal_identification_applicable"] and
                              P3.anderson_rubin_rank_diagnostic(scenario_spec(scenario)["loading"])["overall_pass"])
        expected_applicable, _ = applicability(row["method"], row["metric"], scenario, identification)
        if not expected_applicable and row["status"] != "inapplicable":
            errors.append(f"inapplicable row received {row['status']}: {row['row_key']}")
        if row["status"] == "nonconvergence" and (row["method"] != "fa_selected" or not expected_applicable):
            errors.append(f"nonconvergence assigned outside applicable selected FA: {row['row_key']}")
    nonconverged_rows = [row for row in rows if row["status"] == "nonconvergence"]
    audit = diagnostics.get("nonconvergence_applicability_audit", {})
    if audit.get("row_count") != len(nonconverged_rows) or not audit.get("all_selected_fa") or not audit.get("no_identification_applicability_rows"):
        errors.append("nonconvergence applicability audit mismatch")
    checks = diagnostics["verification_checks"]
    poison = checks["capability_and_poison"]
    if not poison["lawful_fit_excludes_test_and_truth"] or not poison["frozen_recovery_excludes_train_validation_truth"]:
        errors.append("lawful fit/recovery capability boundary includes forbidden inputs")
    for name in ("test_observation_poison_complete_fitted_state_max_abs", "test_observation_poison_fixed_recovery_max_abs",
                 "latent_truth_poison_complete_fitted_state_max_abs", "latent_truth_poison_fixed_recovery_max_abs"):
        if poison[name] > TOL["poison"]:
            errors.append(f"complete fitted-state or fixed-recovery poison check failed: {name}")
    if not poison["poisoned_test_changed"] or not poison["poisoned_truth_changed"]:
        errors.append("poison fixtures did not materially change forbidden data")
    prep = checks["preprocessing_leakage"]
    if prep["proper_train_only_test_poison_parameter_change"] > TOL["poison"]:
        errors.append("proper train-only preprocessing changed under test poison")
    if prep["invalid_pooled_test_poison_parameter_change"] <= 1.0:
        errors.append("invalid pooled preprocessing did not visibly respond to test poison")
    oracle = checks["true_parameter_reference"]
    if any(value > TOL["oracle"] for value in oracle.values()):
        errors.append("true-parameter reference failed exact target checks")
    faults = checks["scoring_fault_injection"]
    if not (faults["nonfinite_reconstruction_status"] == "invalid" and faults["nonfinite_covariance_nll_status"] == "invalid"
            and faults["nonfinite_covariance_error_status"] == "invalid" and faults["missing_result_applicable_invalid_inapplicable_preserved"]
            and faults["inference_invalid_exception_applicable_invalid_inapplicable_preserved"]
            and faults["inference_linalg_exception_applicable_nonconvergence_inapplicable_preserved"]
            and faults["machine_readable_reasons_present"]):
        errors.append("scoring failure/nonfinite fault labels failed")
    for scenario_id in ("correlated_residuals", "outliers"):
        affected = [row for row in rows if row["scenario"] == scenario_id and row["method"] == "fa_selected" and row["metric"] in ("loading_subspace_angle_max_degrees", "uniqueness_rmse")]
        if any(row["status"] != "inapplicable" for row in affected):
            errors.append(f"identification metrics not inapplicable under {scenario_id}")
    for scenario in SCENARIOS:
        for replicate in range(REPLICATES):
            for metric in ("principal_subspace_angle_max_degrees",):
                pca = next(row for row in rows if row["scenario"] == scenario["id"] and row["replicate"] == str(replicate) and row["method"] == "pca" and row["metric"] == metric)
                ppca = next(row for row in rows if row["scenario"] == scenario["id"] and row["replicate"] == str(replicate) and row["method"] == "ppca" and row["metric"] == metric)
                if pca["status"] == "ok" and ppca["status"] == "ok" and abs(float(pca["value"]) - float(ppca["value"])) > TOL["oracle"]:
                    errors.append(f"PCA/PPCA principal target mismatch: {scenario['id']}:{replicate}")
    return errors


def adversarial_resume_checks(full_rows: list[dict], root: Path) -> list[str]:
    failures = []
    def must_reject(label: str, action) -> None:
        try:
            action()
        except (ValueError, RuntimeError):
            return
        failures.append(f"adversarial resume case accepted: {label}")
    fresh_rows, _ = evaluate_replicate(SCENARIOS[0], 0)
    fresh_map = {serialized_row(row)["row_key"]: serialized_row(row) for row in fresh_rows}
    subset = [row for row in full_rows if row["row_key"] in fresh_map]
    corrupted = [dict(row) for row in subset[:2]]
    corrupted[0]["value"] = "999" if corrupted[0]["status"] == "ok" else ""
    must_reject("corrupted", lambda: validate_resume_rows(corrupted, fresh_rows))
    stale = [dict(row) for row in subset[:2]]
    stale[0]["n_train"] = "999"
    must_reject("stale", lambda: validate_resume_rows(stale, fresh_rows))
    duplicate = [dict(subset[0]), dict(subset[0])]
    must_reject("duplicate", lambda: validate_resume_rows(duplicate, fresh_rows))
    unknown = [dict(subset[0])]
    unknown[0]["row_key"] += "|unknown"
    must_reject("unknown", lambda: validate_resume_rows(unknown, fresh_rows))
    incomplete = root / "incomplete.csv"
    with incomplete.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS[:-1], lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: subset[0][field] for field in FIELDS[:-1]})
    must_reject("incomplete", lambda: read_results(incomplete))
    return failures


def verify(root: Path) -> None:
    errors = validate_artifacts(root)
    with tempfile.TemporaryDirectory(prefix="ldg-p4-u02-") as temporary:
        temporary = Path(temporary)
        normal, reverse, resumed = temporary / "normal", temporary / "reverse", temporary / "resumed"
        build(normal, quiet=True)
        build(reverse, reverse=True, quiet=True)
        errors.extend(f"byte regeneration mismatch: {name}" for name in compare_roots(root, normal))
        errors.extend(f"reversed traversal mismatch: {name}" for name in compare_roots(normal, reverse))
        resumed.mkdir()
        full_rows = read_results(normal / "results.csv")
        write_results(resumed / "results.csv", full_rows[:len(full_rows)//2])
        build(resumed, resume=True, quiet=True)
        errors.extend(f"resume mismatch: {name}" for name in compare_roots(normal, resumed))
        errors.extend(adversarial_resume_checks(full_rows, temporary))
    if errors:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(errors))
    print(json.dumps({
        "verified": True, "artifact_root": str(root),
        "checks": ["exact_artifact_set", "hashes", "canonical_unique_complete_rows", "summary_recomputation",
                   "byte_regeneration", "reversed_order", "valid_partial_resume", "adversarial_resume_rejection",
                   "restricted_selection_api", "test_and_latent_poison", "preprocessing_leakage_control",
                   "true_parameter_reference", "failure_and_nonfinite_fault_injection"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    generate, check = args.generate, args.verify
    if not generate and not check:
        generate = check = True
    if generate:
        build(args.artifact_root, reverse=args.reverse, resume=args.resume)
    if check:
        verify(args.artifact_root)


if __name__ == "__main__":
    main()
