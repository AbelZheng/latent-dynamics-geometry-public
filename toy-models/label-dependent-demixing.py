#!/usr/bin/env python3
"""Deterministic label-dependent demixing generation/recovery experiment.

NumPy and Pillow only. Generation, marginalization, fitting/selection,
trial-level decoding, scoring, artifact writing, and verification are separate.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
P, T = 12, 24
MARGINALS = ("time", "stimulus_time", "decision_time", "interaction_time")
CONDITIONS = ("balanced", "nearly_collinear", "missing_unbalanced_cell")
CELL_ORDER = ((-1, -1), (-1, 1), (1, -1), (1, 1))
BALANCED_COUNTS = {"train": 40, "validation": 20, "test": 30}
UNBALANCED_COUNTS = {
    "train": (40, 30, 16, 0), "validation": (20, 15, 8, 0), "test": (30, 30, 30, 30)
}
LAMBDA_GRID = (1e-4, 1e-2, 1.0, 100.0)
DECODER_GRID = (1e-3, 1e-1, 10.0)
TOL = {"algebra": 1e-10, "poison": 1e-12, "finite": 0.0, "determinism": 0.0}
ARTIFACT_NAMES = ("metrics.csv", "diagnostics.json", "summary.png")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "label-dependent-demixing"
METRIC_FIELDS = ("split", "condition", "method", "marginal", "target", "metric", "value")


def rng_for(*codes: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence((MASTER_SEED,) + tuple(codes))))


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x)


def raw_mixing_vectors() -> dict[str, np.ndarray]:
    return {
        "time": np.array([1.0, .3, -.2, .5, 0, .4, -.6, .2, .1, -.3, .7, -.1]),
        "stimulus_time": np.array([.2, 1, .4, -.3, .6, -.2, .3, -.7, .1, .5, -.4, .2]),
        "decision_time": np.array([-.4, .2, 1, .5, -.2, .7, -.1, .3, -.6, .2, .4, -.5]),
        "interaction_time": np.array([.5, -.3, .2, 1, .4, -.6, .2, .1, .7, -.2, .3, .4]),
    }


def mixing_vectors(collinear: bool = False) -> dict[str, np.ndarray]:
    raw = raw_mixing_vectors()
    vectors = {key: unit(value) for key, value in raw.items()}
    if collinear:
        orthogonal = vectors["decision_time"] - vectors["stimulus_time"] * np.dot(vectors["stimulus_time"], vectors["decision_time"])
        vectors["decision_time"] = unit(vectors["stimulus_time"] + 0.06 * unit(orthogonal))
    return vectors


def temporal_profiles() -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, T)
    profiles = {
        "time": 0.7 * np.sin(2 * np.pi * t) + 0.35 * (t - .5),
        "stimulus_time": 1.25 * np.exp(-((t - .35) / .18) ** 2) - .25,
        "decision_time": 1.1 / (1 + np.exp(-12 * (t - .58))) - .55,
        "interaction_time": .85 * np.sin(3 * np.pi * t) * np.exp(-1.3 * t),
    }
    return profiles


def condition_spec(name: str) -> dict:
    vectors = mixing_vectors(name == "nearly_collinear")
    profiles = temporal_profiles()
    amplitudes = {"time": 1.0, "stimulus_time": 1.35, "decision_time": 1.15, "interaction_time": .75}
    components = {key: amplitudes[key] * profiles[key][:, None] * vectors[key][None, :] for key in MARGINALS}
    covariance = .16 * np.eye(P) + .10 * np.outer(unit(np.arange(1, P + 1, dtype=float)), unit(np.arange(1, P + 1, dtype=float)))
    return {"name": name, "raw_vectors": raw_mixing_vectors(), "vectors": vectors, "profiles": profiles,
            "amplitudes": amplitudes, "components": components, "noise_covariance": covariance}


def split_counts(condition: str, split: str) -> tuple[int, ...]:
    if condition == "missing_unbalanced_cell":
        return UNBALANCED_COUNTS[split]
    return (BALANCED_COUNTS[split],) * 4


def generate_split(spec: dict, split: str) -> dict:
    condition_code = CONDITIONS.index(spec["name"]) + 1
    split_code = {"train": 1, "validation": 2, "test": 3}[split]
    observed, labels, trial_ids, signals = [], [], [], []
    counts = split_counts(spec["name"], split)
    for cell_index, ((stimulus, decision), count) in enumerate(zip(CELL_ORDER, counts)):
        for local_index in range(count):
            signal = (
                spec["components"]["time"]
                + stimulus * spec["components"]["stimulus_time"]
                + decision * spec["components"]["decision_time"]
                + stimulus * decision * spec["components"]["interaction_time"]
            )
            noise = rng_for(10, condition_code, split_code, cell_index, local_index).multivariate_normal(
                np.zeros(P), spec["noise_covariance"], size=T
            )
            observed.append(signal + noise)
            signals.append(signal)
            labels.append((stimulus, decision))
            trial_ids.append(condition_code * 1_000_000 + split_code * 100_000 + cell_index * 10_000 + local_index)
    return {
        "observed": np.asarray(observed), "labels": np.asarray(labels, dtype=int),
        "trial_ids": np.asarray(trial_ids, dtype=int), "latent_signal": np.asarray(signals),
        "declared_cell_counts": np.asarray(counts, dtype=int),
    }


def generate_condition(name: str) -> dict:
    spec = condition_spec(name)
    return {"spec": spec, **{split: generate_split(spec, split) for split in ("train", "validation", "test")}}


def cell_counts(labels: np.ndarray) -> np.ndarray:
    return np.array([np.sum((labels[:, 0] == s) & (labels[:, 1] == d)) for s, d in CELL_ORDER], dtype=int)


def estimate_condition_means(observed: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return 2 x 2 x T x P means; use weighted LS if a cell is absent/unbalanced."""
    counts = cell_counts(labels)
    balanced = bool(np.all(counts == counts[0]) and counts[0] > 0)
    if balanced:
        means = np.empty((2, 2, T, P))
        for index, (s, d) in enumerate(CELL_ORDER):
            means[(s + 1) // 2, (d + 1) // 2] = np.mean(observed[(labels[:, 0] == s) & (labels[:, 1] == d)], axis=0)
        design_rank = 4
        method = "balanced_cell_means"
    else:
        design = np.column_stack([np.ones(len(labels)), labels[:, 0], labels[:, 1], labels[:, 0] * labels[:, 1]])
        coefficients = np.linalg.pinv(design) @ observed.reshape(len(labels), -1)
        full_design = np.array([[1, s, d, s * d] for s, d in CELL_ORDER], dtype=float)
        predicted = (full_design @ coefficients).reshape(4, T, P)
        means = predicted.reshape(2, 2, T, P)
        design_rank = int(np.linalg.matrix_rank(design))
        method = "trial_weighted_cell_ls"
    return means, {
        "counts": counts.tolist(), "balanced": balanced, "all_cells_present": bool(np.all(counts > 0)),
        "design_rank": design_rank, "full_effect_design_rank": 4,
        "balanced_estimator_valid": balanced, "estimator": method,
    }


def marginal_decomposition(means: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """Balanced two-factor marginalization after one global grand-mean removal."""
    time_mean = np.mean(means, axis=(0, 1), keepdims=True)
    grand_mean = np.mean(time_mean, axis=2, keepdims=True)
    centered = means - grand_mean
    time = time_mean - grand_mean
    stim = np.mean(means, axis=1, keepdims=True) - time_mean
    decision = np.mean(means, axis=0, keepdims=True) - time_mean
    interaction = means - time_mean - stim - decision
    parts = {
        "time": np.broadcast_to(time, means.shape).copy(),
        "stimulus_time": np.broadcast_to(stim, means.shape).copy(),
        "decision_time": np.broadcast_to(decision, means.shape).copy(),
        "interaction_time": interaction,
    }
    discrepancy = float(np.max(np.abs(sum(parts.values()) - centered)))
    return parts, centered, grand_mean.reshape(P), discrepancy


def flatten_means(means: np.ndarray) -> np.ndarray:
    return means.reshape(4 * T, P).T


def flatten_parts(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: flatten_means(parts[key]) for key in MARGINALS}


def truth_marginals(spec: dict) -> dict[str, np.ndarray]:
    signs = {
        "time": np.ones(4),
        "stimulus_time": np.array([s for s, _ in CELL_ORDER]),
        "decision_time": np.array([d for _, d in CELL_ORDER]),
        "interaction_time": np.array([s * d for s, d in CELL_ORDER]),
    }
    components = dict(spec["components"])
    components["time"] = components["time"] - np.mean(components["time"], axis=0, keepdims=True)
    return {key: np.stack([sign * components[key] for sign in signs[key]]).reshape(4 * T, P).T for key in MARGINALS}


def subspace_angle_max(left: np.ndarray, right: np.ndarray) -> float:
    ql, _ = np.linalg.qr(left)
    qr, _ = np.linalg.qr(right)
    singular = np.linalg.svd(ql.T @ qr, compute_uv=False)
    return float(np.degrees(np.max(np.arccos(np.clip(singular, -1.0, 1.0)))))


def pairwise_frobenius(parts: dict[str, np.ndarray]) -> dict[str, float]:
    records = {}
    for i, left in enumerate(MARGINALS):
        for right in MARGINALS[i + 1:]:
            records[f"{left}__{right}"] = float(abs(np.sum(parts[left] * parts[right])))
    return records


def leading_left(matrix: np.ndarray, rank: int = 1) -> np.ndarray:
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank]


def fit_global_pca(x: np.ndarray) -> dict:
    basis = leading_left(x, len(MARGINALS))
    return {"basis": basis}


def fit_separate_pca(parts: dict[str, np.ndarray]) -> dict[str, dict]:
    return {key: {"basis": leading_left(parts[key])} for key in MARGINALS}


def fit_dpca(x: np.ndarray, parts: dict[str, np.ndarray], regularization: float) -> dict[str, dict]:
    gram = x @ x.T + regularization * np.eye(P)
    records = {}
    for key in MARGINALS:
        cross = parts[key] @ x.T
        a = np.linalg.solve(gram, cross.T).T
        u = leading_left(a @ x)
        decoder = u.T @ a
        records[key] = {"A": a, "U": u, "D": decoder, "regularization": regularization}
    return records


def apply_marginal_model(method: str, model, x: np.ndarray, marginal: str) -> np.ndarray:
    if method == "condition_mean_baseline":
        return model["time"] if marginal == "time" else np.zeros_like(x)
    if method == "global_pca":
        return model["basis"] @ model["basis"].T @ x
    if method == "separate_marginal_pca":
        basis = model[marginal]["basis"]
        return basis @ basis.T @ x
    if method == "regularized_dpca":
        record = model[marginal]
        return record["U"] @ record["D"] @ x
    raise KeyError(method)


def normalized_mse(estimate: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((estimate - target) ** 2) / max(np.mean(target ** 2), 1e-15))


def direction_angle(left: np.ndarray, right: np.ndarray) -> float:
    left, right = unit(left.reshape(-1)), unit(right.reshape(-1))
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(left, right)), 0.0, 1.0))))


def marginal_variance(parts: dict[str, np.ndarray]) -> dict[str, float]:
    energies = {key: float(np.sum(value ** 2)) for key, value in parts.items()}
    total = max(sum(energies.values()), 1e-15)
    return {key: value / total for key, value in energies.items()}


def select_dpca(train_x: np.ndarray, train_parts: dict[str, np.ndarray], validation_x: np.ndarray,
                validation_parts: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    candidates = []
    models = []
    for regularization in LAMBDA_GRID:
        model = fit_dpca(train_x, train_parts, regularization)
        errors = {
            key: normalized_mse(apply_marginal_model("regularized_dpca", model, validation_x, key), validation_parts[key])
            for key in MARGINALS
        }
        candidates.append({"lambda": regularization, "validation_mean_normalized_mse": float(np.mean(list(errors.values()))), "by_marginal": errors})
        models.append(model)
    index = min(range(len(candidates)), key=lambda i: (candidates[i]["validation_mean_normalized_mse"], candidates[i]["lambda"]))
    return models[index], candidates


def ridge_classifier(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
                     validation_y: np.ndarray) -> tuple[dict, list[dict]]:
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    x = (train_x - mean) / scale
    xv = (validation_x - mean) / scale
    kernel = x @ x.T
    candidates = []
    weights = []
    for regularization in DECODER_GRID:
        gram = kernel + regularization * np.eye(len(x))
        dual = np.linalg.solve(gram, train_y)
        weight = x.T @ dual
        accuracy = float(np.mean(np.where(xv @ weight >= 0.0, 1, -1) == validation_y))
        candidates.append({"lambda": regularization, "validation_accuracy": accuracy,
                           "gram_diagnostics": matrix_diagnostics(gram)})
        weights.append(weight)
    index = min(range(len(candidates)), key=lambda i: (-candidates[i]["validation_accuracy"], candidates[i]["lambda"]))
    return {"mean": mean, "scale": scale, "weight": weights[index], "lambda": candidates[index]["lambda"]}, candidates


def trial_features(observed: np.ndarray) -> np.ndarray:
    return observed.reshape(len(observed), T * P)


def classifier_accuracy(model: dict, observed: np.ndarray, labels: np.ndarray) -> float:
    x = (trial_features(observed) - model["mean"]) / model["scale"]
    return float(np.mean(np.where(x @ model["weight"] >= 0.0, 1, -1) == labels))


def fit_decoders(condition: dict) -> dict:
    train, validation = condition["train"], condition["validation"]
    records = {}
    for index, label in enumerate(("stimulus", "decision")):
        model, candidates = ridge_classifier(
            trial_features(train["observed"]), train["labels"][:, index],
            trial_features(validation["observed"]), validation["labels"][:, index],
        )
        records[label] = {"model": model, "candidates": candidates}
    return records


def add_metric(rows, split, condition, method, marginal, target, metric, value):
    rows.append({"split": split, "condition": condition, "method": method, "marginal": marginal,
                 "target": target, "metric": metric, "value": float(value)})


def recovery_target(condition: str, target: str) -> str:
    return f"invalid_rank_deficient_balanced_recovery__{target}" if condition == "missing_unbalanced_cell" else target


def prepare_condition(condition: dict) -> dict:
    prepared = {}
    algebra = {}; orthogonality = {}; grand_means = {}
    cells = {}
    for split in ("train", "validation", "test"):
        means, cell_record = estimate_condition_means(condition[split]["observed"], condition[split]["labels"])
        parts, centered, grand_mean, discrepancy = marginal_decomposition(means)
        flat_parts = flatten_parts(parts)
        prepared[split] = {"means": means, "centered_means": centered, "grand_mean": grand_mean,
                           "X": flatten_means(centered), "parts": flat_parts}
        cells[split] = cell_record
        algebra[split] = discrepancy
        grand_means[split] = grand_mean.tolist()
        orthogonality[split] = pairwise_frobenius(flat_parts)
    return {"splits": prepared, "cells": cells, "marginal_sum_max_abs": algebra,
            "pairwise_frobenius_inner_products": orthogonality, "grand_means": grand_means}


def score_models(name: str, condition: dict, prepared: dict, rows: list[dict]) -> tuple[dict, dict]:
    train, validation, test = (prepared["splits"][key] for key in ("train", "validation", "test"))
    models = {
        "condition_mean_baseline": {"time": train["parts"]["time"]},
        "global_pca": fit_global_pca(train["X"]),
        "separate_marginal_pca": fit_separate_pca(train["parts"]),
    }
    dpca, selection = select_dpca(train["X"], train["parts"], validation["X"], validation["parts"])
    models["regularized_dpca"] = dpca
    truth = truth_marginals(condition["spec"])

    population_fractions = marginal_variance(truth)
    for marginal, fraction in population_fractions.items():
        add_metric(rows, "population", name, "generating_truth", marginal, recovery_target(name, "noiseless_population_marginal"), "marginal_variance_fraction", fraction)
    empirical_fractions = {}
    truth_recovery = {}
    for split_name, split_record in prepared["splits"].items():
        empirical_fractions[split_name] = marginal_variance(split_record["parts"])
        truth_recovery[split_name] = {}
        for marginal in MARGINALS:
            add_metric(rows, split_name, name, "condition_means", marginal,
                       recovery_target(name, "empirical_marginal"), "marginal_variance_fraction",
                       empirical_fractions[split_name][marginal])
            value = normalized_mse(split_record["parts"][marginal], truth[marginal])
            truth_recovery[split_name][marginal] = value
            add_metric(rows, split_name, name, "condition_means", marginal,
                       recovery_target(name, "noiseless_true_marginal"), "empirical_vs_truth_normalized_mse", value)

    reconstructions = {}
    for method, model in models.items():
        reconstructions[method] = {}
        for marginal in MARGINALS:
            model_input = test["parts"][marginal] if method in ("global_pca", "separate_marginal_pca") else test["X"]
            estimate = apply_marginal_model(method, model, model_input, marginal)
            reconstructions[method][marginal] = estimate
            add_metric(rows, "test", name, method, marginal, recovery_target(name, "heldout_empirical_marginal"),
                       "normalized_mse", normalized_mse(estimate, test["parts"][marginal]))
            if method not in ("condition_mean_baseline", "global_pca"):
                direction = model[marginal]["basis"][:, 0] if method == "separate_marginal_pca" else model[marginal]["U"][:, 0]
                add_metric(rows, "train", name, method, marginal, recovery_target(name, "true_marginal_direction"),
                           "principal_angle_degrees", direction_angle(direction, condition["spec"]["vectors"][marginal]))
        combined = sum(reconstructions[method].values())
        add_metric(rows, "test", name, method, "all", recovery_target(name, "heldout_condition_means"),
                   "combined_reconstruction_normalized_mse", normalized_mse(combined, test["X"]))
    true_span = np.column_stack([condition["spec"]["vectors"][key] for key in MARGINALS])
    add_metric(rows, "train", name, "global_pca", "all", recovery_target(name, "span_of_all_true_marginal_directions"),
               "principal_subspace_angle_max_degrees", subspace_angle_max(models["global_pca"]["basis"], true_span))

    confusion_profiles = {}
    for method in ("separate_marginal_pca", "regularized_dpca"):
        profile = {"raw_map_output_energy": {}, "normalized_confusion_fraction": {}, "decoder_score_energy": {},
                   "assigned_marginal_fraction": {}, "demixing_index": {}}
        for recovered in MARGINALS:
            raw = {}; scores = {}
            for source in MARGINALS:
                output = apply_marginal_model(method, models[method], truth[source], recovered)
                raw[source] = float(np.sum(output ** 2))
                decoder = (models[method][recovered]["basis"].T if method == "separate_marginal_pca" else models[method][recovered]["D"])
                scores[source] = float(np.sum((decoder @ truth[source]) ** 2))
                add_metric(rows, "test", name, method, recovered, recovery_target(name, f"true_{source}_input"), "raw_map_output_energy", raw[source])
                add_metric(rows, "test", name, method, recovered, recovery_target(name, f"true_{source}_input"), "decoder_score_energy", scores[source])
            total = max(sum(raw.values()), 1e-15)
            normalized = {source: raw[source] / total for source in MARGINALS}
            for source in MARGINALS:
                add_metric(rows, "test", name, method, recovered, recovery_target(name, f"true_{source}_input"), "normalized_confusion_fraction", normalized[source])
            assigned = normalized[recovered]
            demixing = max(normalized.values())
            add_metric(rows, "test", name, method, recovered, recovery_target(name, "true_marginal_inputs"), "assigned_marginal_fraction", assigned)
            add_metric(rows, "test", name, method, recovered, recovery_target(name, "true_marginal_inputs"), "demixing_index", demixing)
            profile["raw_map_output_energy"][recovered] = raw
            profile["normalized_confusion_fraction"][recovered] = normalized
            profile["decoder_score_energy"][recovered] = scores
            profile["assigned_marginal_fraction"][recovered] = assigned
            profile["demixing_index"][recovered] = demixing
        confusion_profiles[method] = profile

    for candidate in selection:
        add_metric(rows, "validation", name, "regularized_dpca", "all", recovery_target(name, "lambda_selection"),
                   f"validation_mean_normalized_mse_lambda_{candidate['lambda']:g}", candidate["validation_mean_normalized_mse"])
    selected_lambda = next(record["regularization"] for record in dpca.values())
    add_metric(rows, "validation", name, "regularized_dpca", "all", recovery_target(name, "lambda_selection"), "selected_lambda", selected_lambda)
    decoders = fit_decoders(condition)
    for label, record in decoders.items():
        index = 0 if label == "stimulus" else 1
        accuracy = classifier_accuracy(record["model"], condition["test"]["observed"], condition["test"]["labels"][:, index])
        add_metric(rows, "test", name, "trial_level_ridge_classifier", label, recovery_target(name, "heldout_trial_labels"), "accuracy", accuracy)
        add_metric(rows, "validation", name, "trial_level_ridge_classifier", label, recovery_target(name, "regularization_selection"), "selected_lambda", record["model"]["lambda"])
    for split_name in ("train", "validation", "test"):
        cell = prepared["cells"][split_name]
        add_metric(rows, split_name, name, "cell_design_diagnostic", "all", "trial_cell_structure", "minimum_cell_count", min(cell["counts"]))
        add_metric(rows, split_name, name, "cell_design_diagnostic", "all", "trial_cell_structure", "maximum_cell_count", max(cell["counts"]))
        add_metric(rows, split_name, name, "cell_design_diagnostic", "all", "trial_cell_structure", "effect_design_rank", cell["design_rank"])
        add_metric(rows, split_name, name, "cell_design_diagnostic", "all", "trial_cell_structure", "balanced_indicator", float(cell["balanced"]))
    return {"models": models, "selection": selection, "reconstructions": reconstructions, "decoders": decoders,
            "confusion_profiles": confusion_profiles, "truth_recovery": truth_recovery,
            "population_variance_fractions": population_fractions, "empirical_variance_fractions": empirical_fractions}, {"truth": truth}

def shuffle_label_pairs(labels: np.ndarray, condition_code: int) -> np.ndarray:
    permutation = rng_for(50, condition_code).permutation(len(labels))
    return labels[permutation].copy()


def shuffle_control(condition: dict, rows: list[dict]) -> dict:
    shuffled_condition = {"spec": condition["spec"]}
    for split in ("train", "validation", "test"):
        shuffled_condition[split] = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in condition[split].items()
        }
    original_labels = shuffled_condition["train"]["labels"].copy()
    original_counts = cell_counts(original_labels)
    shuffled_condition["train"]["labels"] = shuffle_label_pairs(
        original_labels, CONDITIONS.index(condition["spec"]["name"]) + 1
    )
    shuffled_counts = cell_counts(shuffled_condition["train"]["labels"])
    prepared = prepare_condition(shuffled_condition)
    train, validation, test = (prepared["splits"][key] for key in ("train", "validation", "test"))
    model, selection = select_dpca(train["X"], train["parts"], validation["X"], validation["parts"])
    selected = min(selection, key=lambda x: (x["validation_mean_normalized_mse"], x["lambda"]))
    errors = {}
    for marginal in MARGINALS:
        estimate = apply_marginal_model("regularized_dpca", model, test["X"], marginal)
        errors[marginal] = normalized_mse(estimate, test["parts"][marginal])
        add_metric(rows, "test", condition["spec"]["name"], "training_joint_label_pair_shuffle_dpca", marginal,
                   "heldout_true_label_marginal", "normalized_mse", errors[marginal])
    add_metric(rows, "validation", condition["spec"]["name"], "training_joint_label_pair_shuffle_dpca", "all",
               "shuffle_lambda_selection", "selected_lambda", selected["lambda"])
    shuffled_decoders = fit_decoders(shuffled_condition)
    decoder_records = {}
    for label, column in (("stimulus", 0), ("decision", 1)):
        record = shuffled_decoders[label]
        accuracy = classifier_accuracy(record["model"], condition["test"]["observed"], condition["test"]["labels"][:, column])
        decoder_records[label] = {"selected_lambda": record["model"]["lambda"], "candidates": record["candidates"], "test_accuracy": accuracy}
        add_metric(rows, "test", condition["spec"]["name"], "training_joint_label_pair_shuffle_decoder", label,
                   "heldout_true_trial_labels", "accuracy", accuracy)
        add_metric(rows, "validation", condition["spec"]["name"], "training_joint_label_pair_shuffle_decoder", label,
                   "shuffle_regularization_selection", "selected_lambda", record["model"]["lambda"])
    return {
        "shuffle_type": "joint permutation of complete (stimulus, decision) label pairs on training trials only",
        "training_labels_changed_fraction": float(np.mean(shuffled_condition["train"]["labels"] != original_labels)),
        "original_cell_counts": original_counts.tolist(), "shuffled_cell_counts": shuffled_counts.tolist(),
        "cell_counts_exactly_preserved": bool(np.array_equal(original_counts, shuffled_counts)),
        "training_observations_max_abs_change": max_abs(shuffled_condition["train"]["observed"], condition["train"]["observed"]),
        "validation_labels_max_abs_change": max_abs(shuffled_condition["validation"]["labels"], condition["validation"]["labels"]),
        "test_labels_max_abs_change": max_abs(shuffled_condition["test"]["labels"], condition["test"]["labels"]),
        "shuffle_only_uses_training_labels": bool(
            max_abs(shuffled_condition["train"]["observed"], condition["train"]["observed"]) == 0.0
            and max_abs(shuffled_condition["validation"]["labels"], condition["validation"]["labels"]) == 0.0
            and max_abs(shuffled_condition["test"]["labels"], condition["test"]["labels"]) == 0.0
            and np.mean(shuffled_condition["train"]["labels"] != original_labels) > 0.0
        ),
        "selected_lambda": selected["lambda"], "selection_candidates": selection,
        "decoder_branches": decoder_records,
        "shuffled_cell_record": prepared["cells"]["train"],
        "marginal_sum_max_abs": prepared["marginal_sum_max_abs"]["train"],
        "test_normalized_mse": errors,
    }


def deterministic_percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile, method="linear"))


def decoder_shuffle_nulls(condition: dict, rows: list[dict], count: int = 32) -> dict:
    primary = fit_decoders(condition)
    records = {}
    for label, column in (("stimulus", 0), ("decision", 1)):
        observed_accuracy = classifier_accuracy(primary[label]["model"], condition["test"]["observed"], condition["test"]["labels"][:, column])
        null_accuracies = []
        selected_lambdas = []
        candidate_records = []
        counts_preserved = []
        original = condition["train"]["labels"][:, column]
        for shuffle_index in range(count):
            permuted = original[rng_for(55, column, shuffle_index).permutation(len(original))]
            model, candidates = ridge_classifier(
                trial_features(condition["train"]["observed"]), permuted,
                trial_features(condition["validation"]["observed"]), condition["validation"]["labels"][:, column],
            )
            null_accuracies.append(classifier_accuracy(model, condition["test"]["observed"], condition["test"]["labels"][:, column]))
            selected_lambdas.append(model["lambda"])
            candidate_records.append(candidates)
            counts_preserved.append(bool(np.array_equal(np.sort(permuted), np.sort(original))))
        summary = {
            "shuffle_count": count, "observed_accuracy": observed_accuracy,
            "null_median": deterministic_percentile(null_accuracies, 50.0),
            "null_percentile_2_5": deterministic_percentile(null_accuracies, 2.5),
            "null_percentile_97_5": deterministic_percentile(null_accuracies, 97.5),
            "null_exceedance_count_ge_observed": int(np.sum(np.asarray(null_accuracies) >= observed_accuracy)),
            "null_accuracies": null_accuracies, "selected_lambdas": selected_lambdas,
            "candidate_records": candidate_records,
            "label_count_preserved_each_shuffle": bool(all(counts_preserved)),
        }
        records[label] = summary
        for metric in ("null_median", "null_percentile_2_5", "null_percentile_97_5", "null_exceedance_count_ge_observed"):
            add_metric(rows, "test", "decoder_training_label_shuffle_null", "trial_level_ridge_classifier", label,
                       "heldout_trial_labels", metric, summary[metric])
    return records

def pseudo_population(split: dict, condition_code: int, split_code: int) -> dict:
    observed = split["observed"].copy()
    for cell_index, (stimulus, decision) in enumerate(CELL_ORDER):
        indices = np.flatnonzero((split["labels"][:, 0] == stimulus) & (split["labels"][:, 1] == decision))
        for feature in range(P):
            permutation = rng_for(60, condition_code, split_code, cell_index, feature).permutation(len(indices))
            observed[indices, :, feature] = observed[indices[permutation], :, feature]
    return {**split, "observed": observed}


def pseudo_population_control(condition: dict, rows: list[dict]) -> dict:
    condition_code = CONDITIONS.index(condition["spec"]["name"]) + 1
    pseudo = {"spec": condition["spec"]}
    for split_code, split in enumerate(("train", "validation", "test"), 1):
        pseudo[split] = pseudo_population(condition[split], condition_code, split_code)
    original_prepared, pseudo_prepared = prepare_condition(condition), prepare_condition(pseudo)
    mean_difference = max(np.max(np.abs(original_prepared["splits"][split]["means"] - pseudo_prepared["splits"][split]["means"])) for split in ("train", "validation", "test"))
    original_decoders, pseudo_decoders = fit_decoders(condition), fit_decoders(pseudo)
    original_fit = snapshot_observation_only(condition)
    pseudo_fit = snapshot_observation_only(pseudo)
    map_difference = snapshot_difference(original_fit, pseudo_fit)
    accuracy = {}
    for label, index in (("stimulus", 0), ("decision", 1)):
        original_accuracy = classifier_accuracy(original_decoders[label]["model"], condition["test"]["observed"], condition["test"]["labels"][:, index])
        pseudo_accuracy = classifier_accuracy(pseudo_decoders[label]["model"], pseudo["test"]["observed"], pseudo["test"]["labels"][:, index])
        accuracy[label] = {"simultaneous": original_accuracy, "pseudo_population": pseudo_accuracy, "gap": pseudo_accuracy - original_accuracy}
        add_metric(rows, "test", "pseudo_population_control", "trial_level_ridge_classifier", label, "pseudo_population_trial_labels", "accuracy", pseudo_accuracy)
        add_metric(rows, "test", "pseudo_population_control", "trial_level_ridge_classifier", label, "pseudo_minus_simultaneous", "accuracy_gap", pseudo_accuracy - original_accuracy)
    original_cov = np.cov(condition["train"]["observed"].reshape(-1, P), rowvar=False)
    pseudo_cov = np.cov(pseudo["train"]["observed"].reshape(-1, P), rowvar=False)
    offdiag = ~np.eye(P, dtype=bool)
    map_parameter_change = max(
        [map_difference["pca_projector"]]
        + list(map_difference["separate_marginal_projectors"].values())
        + [value for candidate in map_difference["dpca_all_lambda_candidates"].values()
           for record in candidate["by_marginal"].values() for key, value in record.items() if key != "validation_score"]
    )
    reconstruction_change = max(map_difference["test_reconstructions"].values())
    map_only_change = max(map_parameter_change, reconstruction_change)
    for metric, value in (
        ("condition_mean_map_parameter_max_abs_change", map_parameter_change),
        ("condition_mean_reconstruction_max_abs_change", reconstruction_change),
    ):
        add_metric(rows, "test", "pseudo_population_control", "regularized_dpca", "all", "condition_mean_maps", metric, value)
    label_comparisons = {split: bool(np.array_equal(pseudo[split]["labels"], condition[split]["labels"])) for split in ("train", "validation", "test")}
    return {"condition_means_max_abs_change": float(mean_difference),
            "labels_preserved_by_split": label_comparisons,
            "labels_exactly_preserved": bool(all(label_comparisons.values())),
            "offdiagonal_covariance_max_abs_change": float(np.max(np.abs(original_cov[offdiag] - pseudo_cov[offdiag]))),
            "condition_mean_map_difference": map_difference,
            "condition_mean_map_only_max_abs_discrepancy": map_only_change,
            "condition_mean_maps_equal_within_algebra_tolerance": bool(map_only_change <= TOL["algebra"]),
            "map_invariance_scope": "condition-mean fits and reconstructions are equal within the declared algebra tolerance, not required to be byte-identical",
            "decoder_accuracy": accuracy}


def single_trial_reconstruction(model: dict, observed: np.ndarray, training_grand_mean: np.ndarray) -> np.ndarray:
    centered = observed - training_grand_mean[None, None, :]
    x = centered.transpose(2, 0, 1).reshape(P, -1)
    recovered = sum(apply_marginal_model("regularized_dpca", model, x, marginal) for marginal in MARGINALS)
    return recovered.reshape(P, len(observed), T).transpose(1, 2, 0)


def averaging_mismatch(condition: dict, fitted: dict, prepared: dict, rows: list[dict]) -> dict:
    test = condition["test"]
    model = fitted["models"]["regularized_dpca"]
    condition_mean_error = normalized_mse(sum(fitted["reconstructions"]["regularized_dpca"].values()), prepared["splits"]["test"]["X"])
    training_grand_mean = prepared["splits"]["train"]["grand_mean"]
    centered_observed = test["observed"] - training_grand_mean[None, None, :]
    centered_signal = test["latent_signal"] - training_grand_mean[None, None, :]
    trial_reconstruction = single_trial_reconstruction(model, test["observed"], training_grand_mean)
    noisy_error = float(np.mean((trial_reconstruction - centered_observed) ** 2) / np.mean(centered_observed ** 2))
    signal_error = float(np.mean((trial_reconstruction - centered_signal) ** 2) / np.mean(centered_signal ** 2))
    add_metric(rows, "test", "trial_averaging_mismatch", "regularized_dpca", "all", "heldout_condition_means", "condition_mean_normalized_mse", condition_mean_error)
    add_metric(rows, "test", "trial_averaging_mismatch", "regularized_dpca", "all", "centered_noiseless_latent_signal_effect_target", "single_trial_latent_signal_normalized_mse", signal_error)
    add_metric(rows, "test", "trial_averaging_mismatch", "regularized_dpca", "all", "centered_noisy_observed_trial_target", "single_trial_noisy_observation_normalized_mse", noisy_error)
    decoder_accuracy = {}
    for label, index in (("stimulus", 0), ("decision", 1)):
        value = classifier_accuracy(fitted["decoders"][label]["model"], test["observed"], test["labels"][:, index])
        decoder_accuracy[label] = value
        add_metric(rows, "test", "trial_averaging_mismatch", "trial_level_ridge_classifier", label, "heldout_single_trial_labels", "accuracy", value)
    return {"training_grand_mean": training_grand_mean.tolist(), "condition_mean_normalized_mse": condition_mean_error,
            "single_trial_latent_signal_normalized_mse": signal_error,
            "single_trial_noisy_observation_normalized_mse": noisy_error,
            "noisy_single_trial_minus_condition_mean": noisy_error - condition_mean_error,
            "single_trial_decoder_accuracy": decoder_accuracy,
            "interpretation": "A map selected on averaged condition marginals is not validated as a single-trial reconstruction or decoder by that mean-level score."}


def decoder_candidate_states(condition: dict, label: str, column: int) -> dict:
    train_x = trial_features(condition["train"]["observed"])
    validation_x = trial_features(condition["validation"]["observed"])
    train_y = condition["train"]["labels"][:, column]
    validation_y = condition["validation"]["labels"][:, column]
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    x = (train_x - mean) / scale
    xv = (validation_x - mean) / scale
    kernel = x @ x.T
    candidates = []
    for regularization in DECODER_GRID:
        gram = kernel + regularization * np.eye(len(x))
        weight = x.T @ np.linalg.solve(gram, train_y)
        accuracy = float(np.mean(np.where(xv @ weight >= 0.0, 1, -1) == validation_y))
        candidates.append({"lambda": regularization, "weight": weight, "validation_accuracy": accuracy})
    selected_index = min(range(len(candidates)), key=lambda i: (-candidates[i]["validation_accuracy"], candidates[i]["lambda"]))
    return {"mean": mean, "scale": scale, "candidates": candidates, "selected_index": selected_index, "selected": candidates[selected_index], "label": label}


def snapshot_observation_only(condition: dict, evaluation_condition: dict | None = None) -> dict:
    prepared = prepare_condition(condition)
    evaluation = condition if evaluation_condition is None else evaluation_condition
    evaluation_prepared = prepared if evaluation_condition is None else prepare_condition(evaluation_condition)
    train, validation, test = prepared["splits"]["train"], prepared["splits"]["validation"], evaluation_prepared["splits"]["test"]
    pca = fit_global_pca(train["X"])
    separate = fit_separate_pca(train["parts"])
    dpca_candidates = []
    for regularization in LAMBDA_GRID:
        model = fit_dpca(train["X"], train["parts"], regularization)
        scores = {m: normalized_mse(apply_marginal_model("regularized_dpca", model, validation["X"], m), validation["parts"][m]) for m in MARGINALS}
        dpca_candidates.append({"lambda": regularization, "model": model, "by_marginal": scores, "mean_score": float(np.mean(list(scores.values())))})
    selected_index = min(range(len(dpca_candidates)), key=lambda i: (dpca_candidates[i]["mean_score"], dpca_candidates[i]["lambda"]))
    dpca = dpca_candidates[selected_index]["model"]
    decoders = {label: decoder_candidate_states(condition, label, column) for label, column in (("stimulus", 0), ("decision", 1))}
    recon = {}
    for marginal in MARGINALS:
        recon[f"global_pca:{marginal}"] = pca["basis"] @ pca["basis"].T @ test["parts"][marginal]
        recon[f"separate_marginal_pca:{marginal}"] = apply_marginal_model("separate_marginal_pca", separate, test["parts"][marginal], marginal)
        recon[f"regularized_dpca:{marginal}"] = apply_marginal_model("regularized_dpca", dpca, test["X"], marginal)
    decoder_predictions = {}
    for label, column in (("stimulus", 0), ("decision", 1)):
        state = decoders[label]
        features = (trial_features(evaluation["test"]["observed"]) - state["mean"]) / state["scale"]
        decoder_predictions[label] = np.where(features @ state["selected"]["weight"] >= 0.0, 1, -1)
    tensors = {}
    for split in ("train", "validation"):
        tensors[f"{split}:condition_means"] = prepared["splits"][split]["means"]
        tensors[f"{split}:centered_means"] = prepared["splits"][split]["centered_means"]
        tensors[f"{split}:grand_mean"] = prepared["splits"][split]["grand_mean"]
        for marginal in MARGINALS:
            tensors[f"{split}:marginal:{marginal}"] = prepared["splits"][split]["parts"][marginal]
    return {"tensors": tensors, "pca_projector": pca["basis"] @ pca["basis"].T,
            "separate_projectors": {m: separate[m]["basis"] @ separate[m]["basis"].T for m in MARGINALS},
            "dpca_candidates": dpca_candidates, "dpca_selected_index": selected_index,
            "decoders": decoders, "decoder_predictions": decoder_predictions, "reconstructions": recon}


def max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def snapshot_difference(a: dict, b: dict) -> dict:
    tensor_by_key = {key: max_abs(a["tensors"][key], b["tensors"][key]) for key in a["tensors"]}
    pca = max_abs(a["pca_projector"], b["pca_projector"])
    separate = {m: max_abs(a["separate_projectors"][m], b["separate_projectors"][m]) for m in MARGINALS}
    dpca = {}
    for index, (left, right) in enumerate(zip(a["dpca_candidates"], b["dpca_candidates"])):
        by_marginal = {}
        for marginal in MARGINALS:
            lm, rm = left["model"][marginal], right["model"][marginal]
            by_marginal[marginal] = {"A": max_abs(lm["A"], rm["A"]),
                                     "U_projector": max_abs(lm["U"] @ lm["U"].T, rm["U"] @ rm["U"].T),
                                     "D": max_abs(lm["D"], rm["D"]),
                                     "validation_score": abs(left["by_marginal"][marginal] - right["by_marginal"][marginal])}
        dpca[str(index)] = {"lambda": left["lambda"], "by_marginal": by_marginal,
                            "mean_validation_score": abs(left["mean_score"] - right["mean_score"])}
    decoder = {}
    for label in ("stimulus", "decision"):
        left, right = a["decoders"][label], b["decoders"][label]
        candidates = {}
        for index, (lc, rc) in enumerate(zip(left["candidates"], right["candidates"])):
            candidates[str(index)] = {"weight": max_abs(lc["weight"], rc["weight"]),
                                      "validation_accuracy": abs(lc["validation_accuracy"] - rc["validation_accuracy"]),
                                      "lambda": abs(lc["lambda"] - rc["lambda"])}
        decoder[label] = {"mean": max_abs(left["mean"], right["mean"]), "scale": max_abs(left["scale"], right["scale"]),
                          "candidates": candidates, "selected_index_unchanged": left["selected_index"] == right["selected_index"],
                          "prediction": max_abs(a["decoder_predictions"][label], b["decoder_predictions"][label])}
    reconstruction = {key: max_abs(a["reconstructions"][key], b["reconstructions"][key]) for key in a["reconstructions"]}
    numeric = list(tensor_by_key.values()) + [pca] + list(separate.values()) + list(reconstruction.values())
    numeric += [v for candidate in dpca.values() for record in candidate["by_marginal"].values() for v in record.values()]
    numeric += [candidate["mean_validation_score"] for candidate in dpca.values()]
    for record in decoder.values():
        numeric += [record["mean"], record["scale"], record["prediction"]]
        numeric += [v for candidate in record["candidates"].values() for v in candidate.values()]
    return {"condition_mean_and_marginal_tensors": tensor_by_key, "pca_projector": pca,
            "separate_marginal_projectors": separate, "dpca_all_lambda_candidates": dpca,
            "dpca_selected_index_unchanged": a["dpca_selected_index"] == b["dpca_selected_index"],
            "decoder_all_candidates": decoder, "test_reconstructions": reconstruction,
            "overall_max_abs_discrepancy": max(numeric, default=0.0)}

def leakage_diagnostics(all_data: dict) -> dict:
    records = {}
    split_records = {}
    for condition_index, name in enumerate(CONDITIONS):
        condition = all_data[name]
        baseline = snapshot_observation_only(condition)
        test_poison = {"spec": copy.deepcopy(condition["spec"])}
        latent_poison = {"spec": copy.deepcopy(condition["spec"])}
        for split_index, split in enumerate(("train", "validation", "test")):
            test_poison[split] = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in condition[split].items()}
            latent_poison[split] = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in condition[split].items()}
            latent_poison[split]["latent_signal"][:] = rng_for(70, condition_index, split_index).normal(loc=1e4, size=latent_poison[split]["latent_signal"].shape)
        test_poison["test"]["observed"][:] = rng_for(71, condition_index).normal(loc=-1e4, size=test_poison["test"]["observed"].shape)
        test_poison["test"]["labels"][:] = test_poison["test"]["labels"][::-1]
        for truth_index, key in enumerate(MARGINALS):
            latent_poison["spec"]["vectors"][key][:] = rng_for(72, condition_index, truth_index).normal(loc=2e3, size=P)
            latent_poison["spec"]["profiles"][key][:] = rng_for(73, condition_index, truth_index).normal(loc=-2e3, size=T)
            latent_poison["spec"]["components"][key][:] = rng_for(74, condition_index, truth_index).normal(loc=3e3, size=(T, P))
        # Compare outputs on the same original test set; only fit/selection leakage can change them.
        records[name] = {"test_trials_and_labels_poison": snapshot_difference(baseline, snapshot_observation_only(test_poison, condition)),
                         "all_truth_arrays_poison": snapshot_difference(baseline, snapshot_observation_only(latent_poison)),
                         "truth_poison_scope": "all retained latent_signal arrays plus copied spec directions, profiles, and components"}
        records[name]["test_trials_and_labels_poison"]["poisoned_test_observation_norm"] = float(np.linalg.norm(test_poison["test"]["observed"]))
        records[name]["test_trials_and_labels_poison"]["test_label_changed_fraction"] = float(np.mean(test_poison["test"]["labels"] != condition["test"]["labels"]))
        records[name]["all_truth_arrays_poison"]["poisoned_truth_norm"] = float(sum(
            np.linalg.norm(latent_poison[split]["latent_signal"]) for split in ("train", "validation", "test")
        ) + sum(np.linalg.norm(latent_poison["spec"]["components"][key]) for key in MARGINALS))
        for split in ("train", "validation", "test"):
            regenerated = generate_split(condition["spec"], split)
            split_records[f"{name}:{split}"] = {"observed_max_abs": max_abs(regenerated["observed"], condition[split]["observed"]),
                                                 "latent_max_abs": max_abs(regenerated["latent_signal"], condition[split]["latent_signal"]),
                                                 "labels_equal": bool(np.array_equal(regenerated["labels"], condition[split]["labels"])),
                                                 "trial_ids_equal": bool(np.array_equal(regenerated["trial_ids"], condition[split]["trial_ids"]))}
    ids = {split: set(all_data["balanced"][split]["trial_ids"].tolist()) for split in ("train", "validation", "test")}
    all_shapes_valid = bool(all(all_data[name][split]["observed"].shape[1:] == (T, P) for name in CONDITIONS for split in ("train", "validation", "test")))
    whole_trial_disjoint = bool(ids["train"].isdisjoint(ids["validation"]) and ids["train"].isdisjoint(ids["test"]) and ids["validation"].isdisjoint(ids["test"]))
    return {"poison_regressions": records, "deterministic_split_regeneration": split_records,
            "all_regeneration_exact": bool(all(r["observed_max_abs"] == 0 and r["latent_max_abs"] == 0 and r["labels_equal"] and r["trial_ids_equal"] for r in split_records.values())),
            "whole_trial_split_ids_disjoint": whole_trial_disjoint,
            "all_trials_retain_complete_time_axis": all_shapes_valid,
            "no_adjacent_time_split_leakage": bool(whole_trial_disjoint and all_shapes_valid),
            "split_unit": "whole independent trial; all 24 adjacent time points remain in one split"}


def generator_algebra_checks(all_data: dict) -> dict:
    records = {}
    for name in CONDITIONS:
        spec = all_data[name]["spec"]
        for split in ("train", "validation", "test"):
            labels = all_data[name][split]["labels"]
            expected = np.stack([
                spec["components"]["time"]
                + stimulus * spec["components"]["stimulus_time"]
                + decision * spec["components"]["decision_time"]
                + stimulus * decision * spec["components"]["interaction_time"]
                for stimulus, decision in labels
            ])
            records[f"{name}:{split}"] = max_abs(expected, all_data[name][split]["latent_signal"])
    return {"by_condition_split_max_abs": records, "maximum": max(records.values())}


def matrix_diagnostics(matrix: np.ndarray) -> dict:
    symmetric = 0.5 * (matrix + matrix.T)
    return {"symmetry_max_abs": max_abs(matrix, matrix.T), "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(symmetric))),
            "condition_number": float(np.linalg.cond(matrix)), "all_finite": bool(np.all(np.isfinite(matrix)))}


def numerical_checks(fitted: dict, all_data: dict, prepared: dict) -> dict:
    pca_oracles = {}; marginal_scoring_oracles = {}; dpca_oracles = {}; dpca_grams = {}; decoder_grams = {}; decoder_selection = {}; lambda_selection = {}
    all_finite = True
    for name in CONDITIONS:
        train = prepared[name]["splits"]["train"]; validation = prepared[name]["splits"]["validation"]
        x = train["X"]
        covariance = x @ x.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigen_basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:len(MARGINALS)]]
        svd_basis = leading_left(x, len(MARGINALS)); fitted_basis = fitted[name]["models"]["global_pca"]["basis"]
        projector = fitted_basis @ fitted_basis.T
        pca_oracles[name] = {"fitted_vs_svd_projector_max_abs": max_abs(projector, svd_basis @ svd_basis.T),
                             "fitted_vs_covariance_eigen_projector_max_abs": max_abs(projector, eigen_basis @ eigen_basis.T),
                             "projector_symmetry_max_abs": max_abs(projector, projector.T),
                             "projector_idempotence_max_abs": max_abs(projector @ projector, projector)}
        marginal_scoring_oracles[name] = {}
        test = prepared[name]["splits"]["test"]
        for marginal in MARGINALS:
            global_expected = projector @ test["parts"][marginal]
            separate_basis = fitted[name]["models"]["separate_marginal_pca"][marginal]["basis"]
            separate_expected = separate_basis @ separate_basis.T @ test["parts"][marginal]
            marginal_scoring_oracles[name][marginal] = {
                "global_pca_P_Xphi_max_abs": max_abs(fitted[name]["reconstructions"]["global_pca"][marginal], global_expected),
                "separate_pca_Pphi_Xphi_max_abs": max_abs(fitted[name]["reconstructions"]["separate_marginal_pca"][marginal], separate_expected),
            }
        selected_record = min(fitted[name]["selection"], key=lambda r: (r["validation_mean_normalized_mse"], r["lambda"]))
        stored_lambda = next(iter(fitted[name]["models"]["regularized_dpca"].values()))["regularization"]
        lambda_selection[name] = {"recomputed_selected_lambda": selected_record["lambda"], "stored_selected_lambda": stored_lambda,
                                  "difference": abs(selected_record["lambda"] - stored_lambda)}
        for regularization in LAMBDA_GRID:
            gram = x @ x.T + regularization * np.eye(P)
            dpca_grams[f"{name}:lambda_{regularization:g}"] = matrix_diagnostics(gram)
            model = fit_dpca(x, train["parts"], regularization)
            for marginal in MARGINALS:
                record = model[marginal]
                cross = train["parts"][marginal] @ x.T
                expected_a = np.linalg.solve(gram, cross.T).T
                target = expected_a @ x
                independent_u = leading_left(target)
                expected_d = record["U"].T @ record["A"]
                dpca_oracles[f"{name}:lambda_{regularization:g}:{marginal}"] = {
                    "A_solve_formula_max_abs": max_abs(record["A"], expected_a),
                    "selected_U_projector_vs_independent_svd_max_abs": max_abs(record["U"] @ record["U"].T, independent_u @ independent_u.T),
                    "D_formula_max_abs": max_abs(record["D"], expected_d),
                    "map_reconstruction_max_abs": max_abs(record["U"] @ record["D"] @ x, (record["U"] @ record["U"].T) @ record["A"] @ x),
                    "numerical_rank": int(np.linalg.matrix_rank(record["U"] @ record["D"], tol=1e-10)),
                    "all_finite": bool(all(np.all(np.isfinite(record[key])) for key in ("A", "U", "D"))),
                }
        decoder_selection[name] = {}
        for label, column in (("stimulus", 0), ("decision", 1)):
            state = decoder_candidate_states(all_data[name], label, column)
            best = min(state["candidates"], key=lambda r: (-r["validation_accuracy"], r["lambda"]))
            stored = fitted[name]["decoders"][label]["model"]["lambda"]
            decoder_selection[name][label] = {"recomputed_selected_lambda": best["lambda"], "stored_selected_lambda": stored,
                                               "difference": abs(best["lambda"] - stored)}
            train_features = trial_features(all_data[name]["train"]["observed"])
            standardized = (train_features - state["mean"]) / state["scale"]
            kernel = standardized @ standardized.T
            for candidate in state["candidates"]:
                gram = kernel + candidate["lambda"] * np.eye(len(kernel))
                decoder_grams[f"primary:{name}:{label}:lambda_{candidate['lambda']:g}"] = matrix_diagnostics(gram)
    noise_records = {name: matrix_diagnostics(all_data[name]["spec"]["noise_covariance"]) for name in CONDITIONS}
    all_records = list(dpca_grams.values()) + list(decoder_grams.values()) + list(noise_records.values())
    all_finite = all(record["all_finite"] for record in all_records) and all(record["all_finite"] for record in dpca_oracles.values())
    return {"all_model_arrays_finite": bool(all_finite), "pca_projector_oracles": pca_oracles,
            "marginal_scoring_oracles": marginal_scoring_oracles,
            "dpca_map_oracles_all_lambdas": dpca_oracles, "dpca_gram_all_lambdas": dpca_grams,
            "decoder_gram_primary_all_candidates": decoder_grams, "noise_covariance_records": noise_records,
            "dpca_lambda_selection_oracles": lambda_selection, "decoder_selection_oracles": decoder_selection,
            "maximum_gram_asymmetry": max(record["symmetry_max_abs"] for record in all_records),
            "minimum_gram_eigenvalue": min(record["minimum_eigenvalue"] for record in all_records),
            "maximum_gram_condition_number": max(record["condition_number"] for record in all_records)}

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = dict(row); record["value"] = format(record["value"], ".17g"); writer.writerow(record)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_summary(path: Path, diagnostics: dict, rows: list[dict]) -> None:
    image = Image.new("RGB", (1500, 960), "white"); draw = ImageDraw.Draw(image); font = ImageFont.load_default()
    draw.text((24, 14), "LDG label-dependent demixing — deterministic diagnostic", fill="black", font=font)
    def panel(box, title):
        draw.rectangle(box, outline="black", width=2); draw.text((box[0]+8, box[1]+7), title, fill="black", font=font)

    panel((24,44,740,470), "A. Balanced stimulus marginal: true vs selected-dPCA time course")
    course = diagnostics["visualization_data"]["stimulus_time_course"]
    values = np.asarray(course["true"] + course["estimated"])
    lo, hi = float(values.min()), float(values.max()); span = max(hi-lo, 1e-12)
    px = lambda i: 55 + i/(T-1)*650
    py = lambda value: 430 - (value-lo)/span*330
    draw.line([(px(i),py(v)) for i,v in enumerate(course["true"])], fill=(0,0,0), width=3)
    draw.line([(px(i),py(v)) for i,v in enumerate(course["estimated"])], fill=(214,39,40), width=3)
    draw.text((50,445), "black=true noiseless marginal; red=selected dPCA; one declared stimulus-positive cell", fill="black", font=font)

    panel((765,44,1475,470), "B. Complete balanced dPCA normalized confusion matrix")
    matrix = diagnostics["confusion_profiles"]["balanced"]["regularized_dpca"]["normalized_confusion_fraction"]
    cell, x0, y0 = 78, 1035, 105
    for col, source in enumerate(MARGINALS):
        draw.text((x0+col*cell+5,75), source.replace("_time", "")[:9], fill="black", font=font)
    for row, recovered in enumerate(MARGINALS):
        draw.text((780,y0+row*cell+42), recovered.replace("_time", "")[:11], fill="black", font=font)
        for col, source in enumerate(MARGINALS):
            value = matrix[recovered][source]
            shade = int(255*(1-min(max(value,0.0),1.0)))
            color = (255, shade, shade)
            box=(x0+col*cell,y0+row*cell,x0+(col+1)*cell-4,y0+(row+1)*cell-4)
            draw.rectangle(box,fill=color,outline="black")
            draw.text((box[0]+17,box[1]+31),f"{value:.3f}",fill="black",font=font)
    draw.text((1000,440), "columns=true source; rows=recovered map", fill="black", font=font)

    panel((24,495,740,930), "C. Negative controls and transfer boundaries")
    shuffle=diagnostics["training_label_shuffle"]; pseudo=diagnostics["pseudo_population_control"]; mismatch=diagnostics["trial_averaging_mismatch"]
    nulls=diagnostics["decoder_training_label_shuffle_nulls"]
    lines=[f"joint shuffle counts preserved: {shuffle['cell_counts_exactly_preserved']}; lambda {shuffle['selected_lambda']}",
           f"joint shuffle decoder accuracy S/D: {shuffle['decoder_branches']['stimulus']['test_accuracy']:.3g} / {shuffle['decoder_branches']['decision']['test_accuracy']:.3g}",
           f"32-shuffle null medians S/D: {nulls['stimulus']['null_median']:.3g} / {nulls['decision']['null_median']:.3g}",
           f"pseudo-pop mean/map discrepancy: {pseudo['condition_means_max_abs_change']:.3g} / {pseudo['condition_mean_map_only_max_abs_discrepancy']:.3g}",
           f"mean / latent-trial / noisy-trial NMSE: {mismatch['condition_mean_normalized_mse']:.3g} / {mismatch['single_trial_latent_signal_normalized_mse']:.3g} / {mismatch['single_trial_noisy_observation_normalized_mse']:.3g}",
           f"missing-cell train design rank: {diagnostics['cell_balance']['missing_unbalanced_cell']['train']['design_rank']} / 4",
           "Missing-cell recovery metrics are explicitly labelled invalid_rank_deficient_balanced_recovery."]
    for i,line in enumerate(lines): draw.text((42,540+i*48),line,fill="black",font=font)

    panel((765,495,1475,930), "D. Executable oracles and interpretation limits")
    poison=max(r[p]["overall_max_abs_discrepancy"] for r in diagnostics["leakage"]["poison_regressions"].values() for p in ("test_trials_and_labels_poison","all_truth_arrays_poison"))
    numerical=diagnostics["numerical_checks"]
    lines=[f"truth-generation algebra max: {diagnostics['generator_algebra']['maximum']:.3g}",
           f"poison maximum discrepancy: {poison:.3g}",
           f"marginal-sum maximum: {max(diagnostics['marginal_algebra'].values()):.3g}",
           f"all-grid Gram minimum eigenvalue: {numerical['minimum_gram_eigenvalue']:.3g}",
           f"dPCA map maximum rank: {max(v['numerical_rank'] for v in numerical['dpca_map_oracles_all_lambdas'].values())}",
           "Separate PCA and global PCA score each held-out marginal X_phi, not full X.",
           "Labels supervise this decomposition; they are not discovered variables.",
           "Encoder and decoder axes may be nonorthogonal; decoding is a separate endpoint.",
           "Demixing is reconstruction, not independence, causal separation, or mechanism."]
    for i,line in enumerate(lines): draw.text((785,540+i*42),line,fill="black",font=font)
    image.save(path,format="PNG",optimize=False)


def seed_registry() -> dict:
    return {"policy":"numpy PCG64 with SeedSequence words (master_seed, stage, condition, split, cell, local trial or shuffle index)",
            "words":{"generation":10,"joint_pair_shuffle":50,"decoder_single_label_shuffle":55,
                     "pseudo_population_featurewise_trajectory_permutation":60,"truth_poison_latent":70,
                     "test_observation_label_poison":71,"truth_poison_vectors":72,"truth_poison_profiles":73,"truth_poison_components":74}}


def build(root: Path, quiet: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    all_data={name:generate_condition(name) for name in CONDITIONS}; rows=[]; prepared={}; fitted={}; auxiliary={}
    for name in CONDITIONS:
        prepared[name]=prepare_condition(all_data[name]); fitted[name],auxiliary[name]=score_models(name,all_data[name],prepared[name],rows)
    shuffle=shuffle_control(all_data["balanced"],rows)
    decoder_nulls=decoder_shuffle_nulls(all_data["balanced"],rows)
    pseudo=pseudo_population_control(all_data["balanced"],rows)
    mismatch=averaging_mismatch(all_data["balanced"],fitted["balanced"],prepared["balanced"],rows)
    leakage=leakage_diagnostics(all_data); numerical=numerical_checks(fitted,all_data,prepared)
    generator_algebra=generator_algebra_checks(all_data)
    true_stimulus=auxiliary["balanced"]["truth"]["stimulus_time"]
    estimated_stimulus=fitted["balanced"]["reconstructions"]["regularized_dpca"]["stimulus_time"]
    vector=all_data["balanced"]["spec"]["vectors"]["stimulus_time"]
    cell_slice=slice(2*T,3*T)
    visualization={"stimulus_time_course":{"true":(vector @ true_stimulus[:,cell_slice]).tolist(),
                                                   "estimated":(vector @ estimated_stimulus[:,cell_slice]).tolist(),
                                                   "cell":{"stimulus":1,"decision":-1}}}
    diagnostics={
        "generator":{"ambient_dimension":P,"time_points":T,"labels":{"stimulus":[-1,1],"decision":[-1,1]},
                     "split_unit":"independent whole trials","conditions":list(CONDITIONS),
                     "raw_mixing_vectors":{k:v.tolist() for k,v in raw_mixing_vectors().items()},
                     "normalized_mixing_vectors":{name:{key:value.tolist() for key,value in all_data[name]["spec"]["vectors"].items()} for name in CONDITIONS},
                     "amplitudes":all_data["balanced"]["spec"]["amplitudes"],
                     "temporal_profiles":{key:value.tolist() for key,value in temporal_profiles().items()},
                     "noise_covariance":all_data["balanced"]["spec"]["noise_covariance"].tolist(),
                     "retained_truth":"latent trial signal and exact marginal directions/profiles/components; scoring only"},
        "cell_balance":{name:prepared[name]["cells"] for name in CONDITIONS},
        "grand_means":{name:prepared[name]["grand_means"] for name in CONDITIONS},
        "marginal_algebra":{f"{name}:{split}":prepared[name]["marginal_sum_max_abs"][split] for name in CONDITIONS for split in ("train","validation","test")},
        "marginal_pairwise_frobenius_inner_products":{f"{name}:{split}":prepared[name]["pairwise_frobenius_inner_products"][split] for name in CONDITIONS for split in ("train","validation","test")},
        "generator_algebra":generator_algebra,
        "empirical_truth_recovery":{name:fitted[name]["truth_recovery"] for name in CONDITIONS},
        "marginal_variance_fractions":{name:{"population":fitted[name]["population_variance_fractions"],"empirical":fitted[name]["empirical_variance_fractions"]} for name in CONDITIONS},
        "dpca_selection":{name:fitted[name]["selection"] for name in CONDITIONS},
        "confusion_profiles":{name:fitted[name]["confusion_profiles"] for name in CONDITIONS},
        "training_label_shuffle":shuffle,"decoder_training_label_shuffle_nulls":decoder_nulls,
        "pseudo_population_control":pseudo,"trial_averaging_mismatch":mismatch,
        "leakage":leakage,"numerical_checks":numerical,"visualization_data":visualization,
        "scientific_limits":["dPCA components optimize regularized marginal reconstruction; they do not establish statistical independence or causal separation.",
                             "Label decoding is a separate trial-level endpoint and is not evidence that a recovered marginal is a unique representation.",
                             "The missing-cell weighted least-squares design is rank deficient; all associated recovery targets use invalid_rank_deficient_balanced_recovery.",
                             "Pseudo-population condition-mean maps are equal only within the declared algebra tolerance; simultaneous trial covariance changes.",
                             "Condition-mean recovery does not establish single-trial reconstruction."],
    }
    write_csv(root/"metrics.csv",rows)
    (root/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    make_summary(root/"summary.png",diagnostics,rows)
    hashes={name:sha256(root/name) for name in ARTIFACT_NAMES}
    split_cell_counts={name:{split:list(split_counts(name,split)) for split in ("train","validation","test")} for name in CONDITIONS}
    design_ranks={name:{split:prepared[name]["cells"][split]["design_rank"] for split in ("train","validation","test")} for name in CONDITIONS}
    raw=raw_mixing_vectors(); raw_matrix=np.column_stack([raw[k] for k in MARGINALS])
    normalized=mixing_vectors(False); normalized_matrix=np.column_stack([normalized[k] for k in MARGINALS])
    condition_vector_registry={}
    for condition_name in CONDITIONS:
        vectors=all_data[condition_name]["spec"]["vectors"]
        matrix=np.column_stack([vectors[key] for key in MARGINALS])
        condition_vector_registry[condition_name]={"normalized_vectors":{key:vectors[key].tolist() for key in MARGINALS},
                                                   "normalized_vector_gram":(matrix.T@matrix).tolist()}
    manifest={"schema_version":1,"project":"Latent Dynamics & Geometry","artifact":"label-dependent-demixing","date":DATE,
              "master_seed":MASTER_SEED,"seed_registry":seed_registry(),
              "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"pillow":getattr(sys.modules.get("PIL"),"__version__","unknown"),"platform":platform.platform()},
              "dimensions":{"ambient":P,"time":T},"conditions":list(CONDITIONS),"marginals":list(MARGINALS),
              "generator":{"raw_vectors":{k:v.tolist() for k,v in raw.items()},"normalized_vectors":{k:v.tolist() for k,v in normalized.items()},
                           "raw_vector_gram":(raw_matrix.T@raw_matrix).tolist(),"normalized_vector_gram":(normalized_matrix.T@normalized_matrix).tolist(),
                           "condition_vector_registry":condition_vector_registry,
                           "amplitudes":all_data["balanced"]["spec"]["amplitudes"],"profiles":{k:v.tolist() for k,v in temporal_profiles().items()},
                           "noise_covariance":all_data["balanced"]["spec"]["noise_covariance"].tolist(),"split_cell_counts":split_cell_counts},
              "formulas":{"centering":"mu = mean over cells and time; centered X = cell means - mu",
                          "marginalization":"X_time=mu_t-mu; X_stim=mean_d X-mu_t; X_dec=mean_s X-mu_t; X_int=X-mu_t-X_stim-X_dec",
                          "grouping":"four balanced cells ordered (-1,-1),(-1,+1),(+1,-1),(+1,+1)",
                          "dpca":"solve (X X^T + lambda I) A_phi^T = (X_phi X^T)^T; U_phi=leading left singular vector of A_phi X; D_phi=U_phi^T A_phi"},
              "methods":{"global_pca_rank":4,"separate_marginal_pca_rank":1,"dpca_rank_per_marginal":1,
                         "decoder_tie_rule":"maximize validation accuracy, then smallest lambda",
                         "dpca_tie_rule":"minimize mean validation marginal NMSE, then smallest lambda"},
              "grids":{"dpca_lambda":list(LAMBDA_GRID),"decoder_lambda":list(DECODER_GRID),"decoder_null_shuffles":32},
              "missing_cell_design_ranks":design_ranks["missing_unbalanced_cell"],
              "pseudo_population_convention":"within each label cell, independently permute complete T-point trial trajectories for each feature; preserve labels and cell means, destroy simultaneous feature trial identity",
              "applicability":{"global_and_separate_pca_marginal_scoring":"P @ X_phi for each held-out marginal",
                               "confusion":"raw map energy, row-normalized 4x4 fractions, decoder-score energy, assigned diagonal fraction, and max-based demixing index",
                               "decoding":"separate endpoint with joint-pair shuffle and 32 independent single-label nulls",
                               "missing_cell":"all recovery targets explicitly prefixed invalid_rank_deficient_balanced_recovery",
                               "single_trial_transfer":"center by training grand mean; noiseless latent-signal and noisy-observation targets are separate",
                               "pseudo_population":"condition-mean maps/reconstructions equal within algebra tolerance, not byte identity"},
              "tolerances":TOL,"artifact_files":["manifest.json",*ARTIFACT_NAMES],"files_sha256":hashes,
              "commands":{"generate":f"{sys.executable} toy-models/label-dependent-demixing.py --generate","verify":f"{sys.executable} toy-models/label-dependent-demixing.py --verify"}}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    if not quiet:
        selected=next(record["regularization"] for record in fitted["balanced"]["models"]["regularized_dpca"].values())
        print(json.dumps({"artifact_root":str(root),"metrics_rows":len(rows),"balanced_selected_lambda":selected,
                          "max_marginal_algebra_error":max(diagnostics["marginal_algebra"].values())},indent=2))
    return diagnostics

def compare_roots(left: Path, right: Path) -> list[str]:
    return [name for name in ("manifest.json",*ARTIFACT_NAMES) if (left/name).read_bytes() != (right/name).read_bytes()]


def verify(root: Path, recompute: bool = True) -> None:
    errors=[]
    for name in ("manifest.json",*ARTIFACT_NAMES):
        if not (root/name).is_file(): errors.append(f"missing {name}")
    if errors: raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    manifest=json.loads((root/"manifest.json").read_text()); diagnostics=json.loads((root/"diagnostics.json").read_text())
    for name,expected in manifest["files_sha256"].items():
        if sha256(root/name)!=expected: errors.append(f"hash mismatch: {name}")
    if manifest.get("tolerances")!=TOL: errors.append("manifest tolerance mismatch")
    required_manifest=("seed_registry","generator","formulas","methods","grids","missing_cell_design_ranks","pseudo_population_convention","applicability")
    if any(key not in manifest for key in required_manifest): errors.append("manifest scientific/runtime registry incomplete")
    generator_manifest=manifest.get("generator",{})
    for key in ("raw_vectors","normalized_vectors","raw_vector_gram","normalized_vector_gram","amplitudes","profiles","noise_covariance","split_cell_counts"):
        if key not in generator_manifest: errors.append(f"manifest generator field missing: {key}")
    if manifest.get("applicability",{}).get("missing_cell")!="all recovery targets explicitly prefixed invalid_rank_deficient_balanced_recovery":
        errors.append("manifest invalid-recovery vocabulary mismatch")

    if diagnostics["generator_algebra"]["maximum"]>TOL["algebra"]: errors.append("stored latent signal does not equal exact four-effect sum")
    if max(diagnostics["marginal_algebra"].values())>TOL["algebra"]: errors.append("marginal-sum algebra failed")
    for key,records in diagnostics["marginal_pairwise_frobenius_inner_products"].items():
        condition,split=key.split(":")
        if diagnostics["cell_balance"][condition][split]["balanced"] and max(records.values(),default=0.0)>TOL["algebra"]:
            errors.append(f"balanced marginal orthogonality failed: {key}")
    for name in CONDITIONS:
        for split in ("train","validation","test"):
            record=diagnostics["cell_balance"][name][split]
            if sum(record["counts"])<=0: errors.append(f"empty split: {name}:{split}")
        if name!="missing_unbalanced_cell" and not all(diagnostics["cell_balance"][name][split]["balanced"] for split in ("train","validation","test")):
            errors.append(f"balanced condition not balanced: {name}")
    missing=diagnostics["cell_balance"]["missing_unbalanced_cell"]
    if missing["train"]["all_cells_present"] or missing["train"]["balanced_estimator_valid"] or missing["train"]["design_rank"]>=4:
        errors.append("missing-cell diagnostic not rank deficient/invalid")

    shuffle=diagnostics["training_label_shuffle"]
    if (not shuffle["shuffle_only_uses_training_labels"] or not shuffle["cell_counts_exactly_preserved"]
            or shuffle["original_cell_counts"]!=shuffle["shuffled_cell_counts"]
            or shuffle["training_observations_max_abs_change"]>TOL["poison"]
            or shuffle["validation_labels_max_abs_change"]>0 or shuffle["test_labels_max_abs_change"]>0):
        errors.append("joint label-pair shuffle independence/count preservation failed")
    if shuffle["training_labels_changed_fraction"]<=0: errors.append("joint label-pair shuffle did not change training labels")
    best_shuffle=min(shuffle["selection_candidates"],key=lambda r:(r["validation_mean_normalized_mse"],r["lambda"]))
    if best_shuffle["lambda"]!=shuffle["selected_lambda"]: errors.append("shuffle dPCA lambda selection was not recomputed")
    for label,branch in shuffle["decoder_branches"].items():
        best=min(branch["candidates"],key=lambda r:(-r["validation_accuracy"],r["lambda"]))
        if best["lambda"]!=branch["selected_lambda"] or not math.isfinite(branch["test_accuracy"]):
            errors.append(f"joint-shuffle decoder branch invalid: {label}")
        for candidate in branch["candidates"]:
            gram=candidate["gram_diagnostics"]
            if not gram["all_finite"] or gram["symmetry_max_abs"]>TOL["algebra"] or gram["minimum_eigenvalue"]<=0:
                errors.append(f"joint-shuffle decoder Gram invalid: {label}")

    for label,null in diagnostics["decoder_training_label_shuffle_nulls"].items():
        if null["shuffle_count"]!=32 or len(null["null_accuracies"])!=32 or len(null["candidate_records"])!=32 or not null["label_count_preserved_each_shuffle"]:
            errors.append(f"decoder null scope/count failed: {label}")
        for percentile,key in ((50,"null_median"),(2.5,"null_percentile_2_5"),(97.5,"null_percentile_97_5")):
            if abs(null[key]-deterministic_percentile(null["null_accuracies"],percentile))>TOL["poison"]:
                errors.append(f"decoder null percentile mismatch: {label}:{key}")
        if null["null_exceedance_count_ge_observed"]!=sum(value>=null["observed_accuracy"] for value in null["null_accuracies"]):
            errors.append(f"decoder null exceedance mismatch: {label}")
        for selected,candidates in zip(null["selected_lambdas"],null["candidate_records"]):
            best=min(candidates,key=lambda r:(-r["validation_accuracy"],r["lambda"]))
            if selected!=best["lambda"]: errors.append(f"decoder null lambda selection mismatch: {label}")
            for candidate in candidates:
                gram=candidate["gram_diagnostics"]
                if not gram["all_finite"] or gram["symmetry_max_abs"]>TOL["algebra"] or gram["minimum_eigenvalue"]<=0:
                    errors.append(f"decoder-null Gram invalid: {label}")

    pseudo=diagnostics["pseudo_population_control"]
    if (pseudo["condition_means_max_abs_change"]>TOL["algebra"] or not pseudo["labels_exactly_preserved"]
            or not all(pseudo["labels_preserved_by_split"].values()) or pseudo["offdiagonal_covariance_max_abs_change"]<=0
            or not pseudo["condition_mean_maps_equal_within_algebra_tolerance"]
            or pseudo["condition_mean_map_only_max_abs_discrepancy"]>TOL["algebra"]):
        errors.append("pseudo-population equality-within-tolerance control failed")
    mismatch=diagnostics["trial_averaging_mismatch"]
    if "single_trial_latent_signal_normalized_mse" not in mismatch or "single_trial_noisy_observation_normalized_mse" not in mismatch:
        errors.append("single-trial transfer targets are incomplete")

    leakage=diagnostics["leakage"]
    if not leakage["all_regeneration_exact"] or not leakage["whole_trial_split_ids_disjoint"] or not leakage["no_adjacent_time_split_leakage"]:
        errors.append("split/regeneration leakage check failed")
    expected_tensor_keys={f"{split}:{kind}" for split in ("train","validation") for kind in ("condition_means","centered_means","grand_mean")}
    expected_tensor_keys|={f"{split}:marginal:{m}" for split in ("train","validation") for m in MARGINALS}
    for condition,record in leakage["poison_regressions"].items():
        for poison in ("test_trials_and_labels_poison","all_truth_arrays_poison"):
            item=record[poison]
            if item["overall_max_abs_discrepancy"]>TOL["poison"] or not item["dpca_selected_index_unchanged"]:
                errors.append(f"poison regression failed: {condition}:{poison}")
            if set(item["condition_mean_and_marginal_tensors"])!=expected_tensor_keys:
                errors.append(f"poison tensor scope incomplete: {condition}:{poison}")
            if len(item["dpca_all_lambda_candidates"])!=len(LAMBDA_GRID): errors.append(f"poison dPCA candidates incomplete: {condition}:{poison}")
            for candidate in item["dpca_all_lambda_candidates"].values():
                if set(candidate["by_marginal"])!=set(MARGINALS): errors.append(f"poison dPCA marginal scope incomplete: {condition}:{poison}")
            for label in ("stimulus","decision"):
                branch=item["decoder_all_candidates"][label]
                if len(branch["candidates"])!=len(DECODER_GRID) or not branch["selected_index_unchanged"]:
                    errors.append(f"poison decoder candidate scope incomplete: {condition}:{poison}:{label}")
        if record["test_trials_and_labels_poison"]["test_label_changed_fraction"]<=0: errors.append(f"test-label poison did not change labels: {condition}")
        if record["all_truth_arrays_poison"]["poisoned_truth_norm"]<=0: errors.append(f"truth poison missing: {condition}")

    numerical=diagnostics["numerical_checks"]
    if not numerical["all_model_arrays_finite"] or numerical["maximum_gram_asymmetry"]>TOL["algebra"] or numerical["minimum_gram_eigenvalue"]<=0:
        errors.append("all-grid Gram/finiteness checks failed")
    if len(numerical["dpca_gram_all_lambdas"])!=len(CONDITIONS)*len(LAMBDA_GRID): errors.append("dPCA Gram grid incomplete")
    if len(numerical["decoder_gram_primary_all_candidates"])!=len(CONDITIONS)*2*len(DECODER_GRID): errors.append("primary decoder Gram grid incomplete")
    for name,record in numerical["pca_projector_oracles"].items():
        if max(record.values())>TOL["algebra"]: errors.append(f"PCA eigensystem/SVD projector oracle failed: {name}")
    for condition,records in numerical["marginal_scoring_oracles"].items():
        if set(records)!=set(MARGINALS) or max(value for record in records.values() for value in record.values())>TOL["algebra"]:
            errors.append(f"PCA marginal-scoring oracle failed: {condition}")
    for key,record in numerical["dpca_map_oracles_all_lambdas"].items():
        if record["numerical_rank"]>1 or not record["all_finite"] or max(record[k] for k in ("A_solve_formula_max_abs","selected_U_projector_vs_independent_svd_max_abs","D_formula_max_abs","map_reconstruction_max_abs"))>TOL["algebra"]:
            errors.append(f"dPCA formula/rank/U-projector oracle failed: {key}")
    if any(record["difference"]>TOL["poison"] for record in numerical["dpca_lambda_selection_oracles"].values()): errors.append("dPCA lambda selection recomputation failed")
    if any(record["difference"]>TOL["poison"] for condition in numerical["decoder_selection_oracles"].values() for record in condition.values()): errors.append("decoder lambda selection recomputation failed")

    for condition,methods in diagnostics["confusion_profiles"].items():
        for method in ("separate_marginal_pca","regularized_dpca"):
            profile=methods.get(method,{})
            for matrix_name in ("raw_map_output_energy","normalized_confusion_fraction","decoder_score_energy"):
                matrix=profile.get(matrix_name,{})
                if set(matrix)!=set(MARGINALS) or any(set(matrix[row])!=set(MARGINALS) for row in matrix):
                    errors.append(f"confusion matrix incomplete: {condition}:{method}:{matrix_name}")
            normalized=profile.get("normalized_confusion_fraction",{})
            for row in MARGINALS:
                if row in normalized and abs(sum(normalized[row].values())-1.0)>TOL["algebra"]:
                    errors.append(f"normalized confusion row does not sum to one: {condition}:{method}:{row}")
                if row in normalized:
                    assigned=normalized[row][row]; demixing=max(normalized[row].values())
                    if abs(profile["assigned_marginal_fraction"][row]-assigned)>TOL["poison"] or abs(profile["demixing_index"][row]-demixing)>TOL["poison"]:
                        errors.append(f"confusion summary definition mismatch: {condition}:{method}:{row}")

    with (root/"metrics.csv").open(encoding="utf-8",newline="") as handle:
        reader=csv.DictReader(handle); rows=list(reader)
        if reader.fieldnames!=list(METRIC_FIELDS): errors.append("metrics fields mismatch")
    seen=set()
    for index,row in enumerate(rows,2):
        if set(row)!=set(METRIC_FIELDS) or any(not row[field].strip() for field in METRIC_FIELDS): errors.append(f"invalid metrics row {index}")
        try:
            if not math.isfinite(float(row["value"])): errors.append(f"nonfinite metric row {index}")
        except ValueError: errors.append(f"nonnumeric metric row {index}")
        key=tuple(row[field] for field in METRIC_FIELDS[:-1])
        if key in seen: errors.append(f"duplicate metric key: {key}")
        seen.add(key)
    required={"normalized_mse","empirical_vs_truth_normalized_mse","combined_reconstruction_normalized_mse","marginal_variance_fraction","principal_angle_degrees","principal_subspace_angle_max_degrees","raw_map_output_energy","normalized_confusion_fraction","decoder_score_energy","assigned_marginal_fraction","demixing_index","accuracy","selected_lambda","single_trial_latent_signal_normalized_mse","single_trial_noisy_observation_normalized_mse"}
    if not required.issubset({r["metric"] for r in rows}): errors.append("required metrics missing")
    for condition in CONDITIONS:
        population=[r for r in rows if r["condition"]==condition and r["split"]=="population" and r["metric"]=="marginal_variance_fraction"]
        empirical=[r for r in rows if r["condition"]==condition and r["method"]=="condition_means" and r["metric"]=="marginal_variance_fraction" and r["split"] in ("train","validation","test")]
        truth_rows=[r for r in rows if r["condition"]==condition and r["metric"]=="empirical_vs_truth_normalized_mse"]
        if len(population)!=4 or len(empirical)!=12 or len(truth_rows)!=12: errors.append(f"truth/variance metric coverage incomplete: {condition}")
    if any(r["method"]=="global_pca" and r["metric"]=="principal_angle_degrees" for r in rows): errors.append("global PCA retains unsupported per-label direction claims")
    for row in rows:
        if row["condition"]=="missing_unbalanced_cell" and row["method"]!="cell_design_diagnostic" and not row["target"].startswith("invalid_rank_deficient_balanced_recovery"):
            errors.append(f"missing-cell recovery metric lacks invalid target label: {row}")
    transfer_targets={r["target"] for r in rows if r["condition"]=="trial_averaging_mismatch" and r["method"]=="regularized_dpca" and r["metric"].startswith("single_trial_")}
    if transfer_targets!={"centered_noiseless_latent_signal_effect_target","centered_noisy_observed_trial_target"}: errors.append("single-trial transfer target labels incomplete")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="ldg-demixing-verify-") as temp:
            regenerated=Path(temp)/"artifact"; build(regenerated,quiet=True); errors.extend(f"byte mismatch: {name}" for name in compare_roots(root,regenerated))
    if errors: raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    print(json.dumps({"verified":True,"artifact_root":str(root),"deterministic_recomputation":recompute},indent=2))

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--generate",action="store_true"); parser.add_argument("--verify",action="store_true"); parser.add_argument("--no-recompute",action="store_true"); parser.add_argument("--artifact-root",type=Path,default=DEFAULT_ROOT); args=parser.parse_args()
    generate,check=args.generate,args.verify
    if not generate and not check: generate=check=True
    if generate: build(args.artifact_root)
    if check: verify(args.artifact_root,recompute=not args.no_recompute)


if __name__=="__main__": main()
