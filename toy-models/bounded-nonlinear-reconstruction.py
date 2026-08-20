#!/usr/bin/env python3
"""Deterministic bounded nonlinear reconstruction demonstration.

Candidate G compares reconstruction methods under fixed architectures, seeds,
optimizer budgets, and validation checkpoint selection.  It requires only
NumPy and Pillow.  Running without flags generates and verifies the four
bounded artifacts.
"""
from __future__ import annotations

import argparse
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
P = 6
D = 1
SEEDS = (0, 1, 2, 3, 4)
CONDITIONS = (
    "nominal_curved",
    "full_dimensional_gaussian",
    "high_variance_nuisance",
    "few_sample_high_capacity",
    "shifted_range",
)
SPLIT_SIZES = {
    "nominal_curved": {"train": 320, "validation": 160, "test": 240},
    "full_dimensional_gaussian": {"train": 320, "validation": 160, "test": 240},
    "high_variance_nuisance": {"train": 320, "validation": 160, "test": 240},
    "few_sample_high_capacity": {"train": 40, "validation": 80, "test": 240},
    "shifted_range": {"train": 320, "validation": 160, "test": 240},
}
NOISE_SD = 0.06
QUADRATURE_NODES = 256
COLLISION_S_DISTANCE = 0.50
COLLISION_CODE_RANGE_FRACTION = 0.05
NUISANCE_SD = 2.5
SMALL_HIDDEN = 8
LARGE_HIDDEN = 32
MAX_EPOCHS = {"linear_ae": 900, "nonlinear_ae": 900, "frozen_random_encoder": 900}
PATIENCE = {"linear_ae": 120, "nonlinear_ae": 120, "frozen_random_encoder": 120}
LEARNING_RATE = {"linear_ae": 0.025, "nonlinear_ae": 0.012, "frozen_random_encoder": 0.015}
MIN_DELTA = 1e-10
TOL = {
    "gradient_relative": 3e-5,
    "poison": 0.0,
    "checkpoint": 2e-12,
    "linear_pca_projector": 7e-2,
    "linear_pca_angle_degrees": 3.0,
    "linear_pca_mse": 2e-3,
}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "bounded-nonlinear-reconstruction"
ARTIFACT_NAMES = ("metrics.csv", "diagnostics.json", "summary.png")
METRIC_FIELDS = (
    "split", "condition", "preprocessing", "method", "seed", "selection",
    "target", "metric", "value",
)


def rng_for(*codes: int) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((MASTER_SEED,) + tuple(codes)))
    )


def condition_code(name: str) -> int:
    return CONDITIONS.index(name) + 1


def split_code(split: str) -> int:
    return {"train": 1, "validation": 2, "test": 3}[split]


def curved_map(s: np.ndarray) -> np.ndarray:
    """Known one-parameter observation map, before additive noise."""
    s = np.asarray(s, dtype=float)
    cos_center = math.sin(1.5) / 1.5  # E[cos(s)] for Uniform[-1.5, 1.5]
    return np.column_stack(
        (
            s,
            s ** 2,
            0.35 * np.sin(2.0 * s),
            0.2 * s ** 3,
            np.cos(s) - cos_center,
            0.4 * np.sin(3.0 * s),
        )
    )


def sample_s(name: str, split: str, n: int) -> np.ndarray:
    rng = rng_for(10, condition_code(name), split_code(split))
    if name == "shifted_range" and split == "test":
        side = rng.integers(0, 2, size=n)
        magnitude = rng.uniform(1.5, 2.4, size=n)
        return np.where(side == 0, -magnitude, magnitude)
    return rng.uniform(-1.5, 1.5, size=n)


def curved_population_moments() -> dict:
    """Compute nominal observable moments by fixed Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(QUADRATURE_NODES)
    s = 1.5 * nodes
    probability_weights = 0.5 * weights
    mapped = curved_map(s)
    mean = probability_weights @ mapped
    centered = mapped - mean
    signal_covariance = (centered * probability_weights[:, None]).T @ centered
    observable_covariance = signal_covariance + (NOISE_SD ** 2) * np.eye(P)
    eigenvalues = np.linalg.eigvalsh(observable_covariance)[::-1]
    return {
        "nodes": nodes,
        "weights": weights,
        "mapped_s": s,
        "mean": mean,
        "signal_covariance": signal_covariance,
        "observable_covariance": observable_covariance,
        "eigenvalues_descending": eigenvalues,
    }


def full_gaussian_parameters() -> tuple[np.ndarray, np.ndarray]:
    moments = curved_population_moments()
    return moments["mean"].copy(), moments["observable_covariance"].copy()


def generate_split(name: str, split: str) -> dict:
    n = SPLIT_SIZES[name][split]
    if name == "full_dimensional_gaussian":
        mean, covariance = full_gaussian_parameters()
        x = rng_for(11, condition_code(name), split_code(split)).multivariate_normal(
            mean, covariance, size=n, check_valid="raise"
        )
        return {
            "observed": x,
            "s": np.full(n, np.nan),
            "noiseless": np.full((n, P), np.nan),
            "generator_kind": "moment_matched_full_dimensional_gaussian_no_1d_truth",
        }
    s = sample_s(name, split, n)
    noiseless = curved_map(s)
    noise = rng_for(12, condition_code(name), split_code(split)).normal(
        scale=NOISE_SD, size=(n, P)
    )
    x = noiseless + noise
    if name == "high_variance_nuisance":
        nuisance = rng_for(13, condition_code(name), split_code(split)).normal(
            scale=NUISANCE_SD, size=(n, 2)
        )
        x[:, 4:6] += nuisance
    return {
        "observed": x,
        "s": s,
        "noiseless": noiseless,
        "generator_kind": "known_curved_1d_map",
    }


def generate_all() -> dict:
    data = {}
    for name in CONDITIONS:
        if name == "shifted_range":
            continue
        data[name] = {
            split: generate_split(name, split)
            for split in ("train", "validation", "test")
        }
    data["shifted_range"] = {
        "train": data["nominal_curved"]["train"],
        "validation": data["nominal_curved"]["validation"],
        "test": generate_split("shifted_range", "test"),
    }
    return data


def fit_standardizer(x: np.ndarray) -> dict:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {"mean": mean, "scale": scale}


def transform(x: np.ndarray, prep: dict) -> np.ndarray:
    return (x - prep["mean"]) / prep["scale"]


def inverse_transform(x: np.ndarray, prep: dict) -> np.ndarray:
    return prep["mean"] + x * prep["scale"]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def fit_mean(x: np.ndarray) -> dict:
    return {"kind": "mean", "mean": np.mean(x, axis=0), "parameter_count": P}


def fit_pca(x: np.ndarray) -> dict:
    mean = np.mean(x, axis=0)
    centered = x - mean
    covariance = centered.T @ centered / x.shape[0]
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1]
    basis = vectors[:, order[:D]]
    pivot = int(np.argmax(np.abs(basis[:, 0])))
    if basis[pivot, 0] < 0:
        basis[:, 0] *= -1.0
    return {
        "kind": "pca", "mean": mean, "basis": basis,
        "eigenvalues": values[order], "parameter_count": P + P * D,
    }


def predict_and_code(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kind = model["kind"]
    if kind == "mean":
        return np.broadcast_to(model["mean"], x.shape).copy(), np.zeros((x.shape[0], 1))
    if kind == "pca":
        code = (x - model["mean"]) @ model["basis"]
        return model["mean"] + code @ model["basis"].T, code
    if kind == "linear_ae":
        p = model["params"]
        code = x @ p["we"] + p["be"]
        return code @ p["wd"] + p["bd"], code
    if kind == "nonlinear_ae":
        p = model["params"]
        h1 = np.tanh(x @ p["w1"] + p["b1"])
        code = h1 @ p["w2"] + p["b2"]
        h3 = np.tanh(code @ p["w3"] + p["b3"])
        return h3 @ p["w4"] + p["b4"], code
    if kind == "frozen_random_encoder":
        f, p = model["frozen"], model["params"]
        h1 = np.tanh(x @ f["w1"] + f["b1"])
        code = h1 @ f["w2"] + f["b2"]
        h3 = np.tanh(code @ p["w3"] + p["b3"])
        return h3 @ p["w4"] + p["b4"], code
    raise KeyError(kind)


def linear_loss_grad(params: dict, x: np.ndarray) -> tuple[float, dict]:
    z = x @ params["we"] + params["be"]
    y = z @ params["wd"] + params["bd"]
    residual = y - x
    loss = float(np.mean(residual ** 2))
    dy = 2.0 * residual / residual.size
    grads = {
        "wd": z.T @ dy,
        "bd": np.sum(dy, axis=0),
    }
    dz = dy @ params["wd"].T
    grads["we"] = x.T @ dz
    grads["be"] = np.sum(dz, axis=0)
    return loss, grads


def nonlinear_loss_grad(params: dict, x: np.ndarray) -> tuple[float, dict]:
    h1 = np.tanh(x @ params["w1"] + params["b1"])
    z = h1 @ params["w2"] + params["b2"]
    h3 = np.tanh(z @ params["w3"] + params["b3"])
    y = h3 @ params["w4"] + params["b4"]
    residual = y - x
    loss = float(np.mean(residual ** 2))
    dy = 2.0 * residual / residual.size
    grads = {"w4": h3.T @ dy, "b4": np.sum(dy, axis=0)}
    dh3 = dy @ params["w4"].T
    da3 = dh3 * (1.0 - h3 ** 2)
    grads["w3"] = z.T @ da3
    grads["b3"] = np.sum(da3, axis=0)
    dz = da3 @ params["w3"].T
    grads["w2"] = h1.T @ dz
    grads["b2"] = np.sum(dz, axis=0)
    dh1 = dz @ params["w2"].T
    da1 = dh1 * (1.0 - h1 ** 2)
    grads["w1"] = x.T @ da1
    grads["b1"] = np.sum(da1, axis=0)
    return loss, grads


def frozen_decoder_loss_grad(params: dict, frozen: dict, x: np.ndarray) -> tuple[float, dict]:
    h1 = np.tanh(x @ frozen["w1"] + frozen["b1"])
    z = h1 @ frozen["w2"] + frozen["b2"]
    h3 = np.tanh(z @ params["w3"] + params["b3"])
    y = h3 @ params["w4"] + params["b4"]
    residual = y - x
    loss = float(np.mean(residual ** 2))
    dy = 2.0 * residual / residual.size
    grads = {"w4": h3.T @ dy, "b4": np.sum(dy, axis=0)}
    dh3 = dy @ params["w4"].T
    da3 = dh3 * (1.0 - h3 ** 2)
    grads["w3"] = z.T @ da3
    grads["b3"] = np.sum(da3, axis=0)
    return loss, grads


def copy_params(params: dict) -> dict:
    return {key: np.array(value, copy=True) for key, value in params.items()}


def count_params(params: dict) -> int:
    return int(sum(np.asarray(value).size for value in params.values()))


def adam_train(
    initial: dict,
    loss_grad,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    method: str,
    extra=None,
) -> dict:
    params = copy_params(initial)
    first_moment = {key: np.zeros_like(value) for key, value in params.items()}
    second_moment = {key: np.zeros_like(value) for key, value in params.items()}
    best_params = copy_params(params)
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    lr = LEARNING_RATE[method]
    for epoch in range(1, MAX_EPOCHS[method] + 1):
        if extra is None:
            train_loss, grads = loss_grad(params, x_train)
        else:
            train_loss, grads = loss_grad(params, extra, x_train)
        for key in params:
            grad = grads[key]
            first_moment[key] = 0.9 * first_moment[key] + 0.1 * grad
            second_moment[key] = 0.999 * second_moment[key] + 0.001 * grad ** 2
            mhat = first_moment[key] / (1.0 - 0.9 ** epoch)
            vhat = second_moment[key] / (1.0 - 0.999 ** epoch)
            params[key] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        if extra is None:
            validation_loss = loss_grad(params, x_validation)[0]
        else:
            validation_loss = loss_grad(params, extra, x_validation)[0]
        history.append((epoch, float(train_loss), float(validation_loss)))
        if validation_loss < best_loss - MIN_DELTA:
            best_loss = float(validation_loss)
            best_epoch = epoch
            best_params = copy_params(params)
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE[method]:
            break
    return {
        "params": best_params,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
    }


def initialize_linear(x: np.ndarray, seed: int) -> dict:
    pca = fit_pca(x)
    rng = rng_for(30, seed)
    we = pca["basis"] + rng.normal(scale=0.02, size=(P, 1))
    wd = pca["basis"].T + rng.normal(scale=0.02, size=(1, P))
    return {"we": we, "be": np.zeros(1), "wd": wd, "bd": np.zeros(P)}


def initialize_nonlinear(hidden: int, seed: int) -> dict:
    rng = rng_for(31, hidden, seed)
    return {
        "w1": rng.normal(scale=math.sqrt(2.0 / (P + hidden)), size=(P, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(scale=math.sqrt(2.0 / (hidden + 1)), size=(hidden, 1)),
        "b2": np.zeros(1),
        "w3": rng.normal(scale=math.sqrt(2.0 / (hidden + 1)), size=(1, hidden)),
        "b3": np.zeros(hidden),
        "w4": rng.normal(scale=math.sqrt(2.0 / (hidden + P)), size=(hidden, P)),
        "b4": np.zeros(P),
    }


def initialize_frozen(hidden: int, seed: int) -> tuple[dict, dict]:
    rng = rng_for(32, hidden, seed)
    frozen = {
        "w1": rng.normal(scale=1.0 / math.sqrt(P), size=(P, hidden)),
        "b1": rng.normal(scale=0.08, size=hidden),
        "w2": rng.normal(scale=1.0 / math.sqrt(hidden), size=(hidden, 1)),
        "b2": rng.normal(scale=0.05, size=1),
    }
    params = {
        "w3": rng.normal(scale=math.sqrt(2.0 / (hidden + 1)), size=(1, hidden)),
        "b3": np.zeros(hidden),
        "w4": rng.normal(scale=math.sqrt(2.0 / (hidden + P)), size=(hidden, P)),
        "b4": np.zeros(P),
    }
    return frozen, params


def train_seeded_method(method: str, x_train: np.ndarray, x_validation: np.ndarray, seed: int, hidden: int) -> dict:
    if method == "linear_ae":
        result = adam_train(initialize_linear(x_train, seed), linear_loss_grad, x_train, x_validation, method)
        return {
            "kind": method, "params": result["params"],
            "parameter_count": count_params(result["params"]),
            "trainable_parameter_count": count_params(result["params"]),
            **{key: result[key] for key in ("best_epoch", "best_validation_loss", "epochs_run", "history")},
        }
    if method == "nonlinear_ae":
        result = adam_train(initialize_nonlinear(hidden, seed), nonlinear_loss_grad, x_train, x_validation, method)
        return {
            "kind": method, "params": result["params"], "hidden": hidden,
            "parameter_count": count_params(result["params"]),
            "trainable_parameter_count": count_params(result["params"]),
            **{key: result[key] for key in ("best_epoch", "best_validation_loss", "epochs_run", "history")},
        }
    if method == "frozen_random_encoder":
        frozen, initial = initialize_frozen(hidden, seed)
        frozen_initial = copy_params(frozen)
        result = adam_train(initial, frozen_decoder_loss_grad, x_train, x_validation, method, extra=frozen)
        return {
            "kind": method, "params": result["params"], "frozen": frozen,
            "frozen_initial": frozen_initial, "hidden": hidden,
            "parameter_count": count_params(result["params"]) + count_params(frozen),
            "trainable_parameter_count": count_params(result["params"]),
            **{key: result[key] for key in ("best_epoch", "best_validation_loss", "epochs_run", "history")},
        }
    raise KeyError(method)


def model_signature(model: dict) -> np.ndarray:
    arrays = []
    for key in ("mean", "basis"):
        if key in model:
            arrays.append(np.asarray(model[key]).ravel())
    for group in ("params", "frozen", "frozen_initial"):
        if group in model:
            for key in sorted(model[group]):
                arrays.append(np.asarray(model[group][key]).ravel())
    arrays.extend(
        [
            np.array([float(model.get("best_epoch", 0))]),
            np.array([float(model.get("best_validation_loss", 0.0))]),
        ]
    )
    return np.concatenate(arrays) if arrays else np.zeros(1)


def fit_condition(bundle: dict, name: str) -> dict:
    """Fit using only train observations and validation observations."""
    prep = fit_standardizer(bundle["train"]["observed"])
    x_train = transform(bundle["train"]["observed"], prep)
    x_validation = transform(bundle["validation"]["observed"], prep)
    hidden = LARGE_HIDDEN if name == "few_sample_high_capacity" else SMALL_HIDDEN
    models = {
        "mean": {"deterministic": fit_mean(x_train)},
        "pca": {"deterministic": fit_pca(x_train)},
    }
    for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
        candidates = {
            seed: train_seeded_method(method, x_train, x_validation, seed, hidden)
            for seed in SEEDS
        }
        best_seed = min(
            SEEDS,
            key=lambda seed: (candidates[seed]["best_validation_loss"], seed),
        )
        models[method] = {"candidates": candidates, "best_seed": best_seed, "selected": candidates[best_seed]}
    return {"preprocessing": prep, "hidden": hidden, "models": models}


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


def code_collision_diagnostic(code: np.ndarray, s: np.ndarray) -> dict:
    code = np.asarray(code, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    code_range = max(float(np.ptp(code)), 1e-12)
    dc = np.abs(code[:, None] - code[None, :])
    ds = np.abs(s[:, None] - s[None, :])
    upper = np.triu(np.ones((code.size, code.size), dtype=bool), 1)
    close_code = dc <= COLLISION_CODE_RANGE_FRACTION * code_range
    candidate_pair_count = int(np.sum(upper))
    close_code_pair_count = int(np.sum(upper & close_code))
    collision_pair_count = int(np.sum(upper & close_code & (ds > COLLISION_S_DISTANCE)))
    fraction = collision_pair_count / close_code_pair_count if close_code_pair_count else 0.0
    return {
        "s_distance_threshold": COLLISION_S_DISTANCE,
        "code_distance_fraction_of_range": COLLISION_CODE_RANGE_FRACTION,
        "code_range": code_range,
        "candidate_pair_count": candidate_pair_count,
        "qualifying_close_code_pair_count": close_code_pair_count,
        "collision_pair_count": collision_pair_count,
        "collision_fraction_of_close_code_pairs": float(fraction),
    }


def add_row(rows: list[dict], split: str, condition: str, method: str, seed, selection: str, target: str, metric: str, value: float) -> None:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric {condition}/{method}/{metric}: {value}")
    rows.append(
        {
            "split": split,
            "condition": condition,
            "preprocessing": "training_only_feature_standardization",
            "method": method,
            "seed": str(seed),
            "selection": selection,
            "target": target,
            "metric": metric,
            "value": value,
        }
    )


def score_one_model(rows: list[dict], name: str, bundle: dict, prep: dict, method: str, model: dict, seed, selection: str) -> dict:
    split_scores = {}
    for split in ("train", "validation", "test"):
        observed = bundle[split]["observed"]
        x = transform(observed, prep)
        reconstructed_t, code = predict_and_code(model, x)
        reconstructed_o = inverse_transform(reconstructed_t, prep)
        transformed_mse = mse(reconstructed_t, x)
        original_mse = mse(reconstructed_o, observed)
        add_row(rows, split, name, method, seed, selection, "observations", "reconstruction_mse_transformed", transformed_mse)
        add_row(rows, split, name, method, seed, selection, "observations", "reconstruction_mse_original_units", original_mse)
        for feature in range(P):
            add_row(rows, split, name, method, seed, selection, f"feature_{feature}", "reconstruction_mse_original_units", mse(reconstructed_o[:, feature], observed[:, feature]))
        add_row(rows, split, name, method, seed, selection, "code_d1", "code_mean", float(np.mean(code)))
        add_row(rows, split, name, method, seed, selection, "code_d1", "code_variance", float(np.var(code)))
        add_row(rows, split, name, method, seed, selection, "code_d1", "code_range", float(np.ptp(code)))
        add_row(rows, split, name, method, seed, selection, "code_d1", "code_dead_or_collapsed", float(np.var(code) < 1e-6 or np.ptp(code) < 1e-4))
        if name != "full_dimensional_gaussian":
            s = bundle[split]["s"]
            add_row(rows, split, name, method, seed, selection, "generator_parameter_s", "absolute_pearson_code_s", abs_corr(code, s))
            add_row(rows, split, name, method, seed, selection, "generator_parameter_s", "absolute_spearman_code_s", abs_corr(rankdata(code.ravel()), rankdata(s)))
            collision = code_collision_diagnostic(code, s)
            for collision_metric in (
                "s_distance_threshold", "code_distance_fraction_of_range", "code_range",
                "candidate_pair_count", "qualifying_close_code_pair_count",
                "collision_pair_count", "collision_fraction_of_close_code_pairs",
            ):
                add_row(rows, split, name, method, seed, selection, "generator_parameter_s_simulation_diagnostic", collision_metric, collision[collision_metric])
            add_row(rows, split, name, method, seed, selection, "generator_parameter_s", "code_collision_rate", collision["collision_fraction_of_close_code_pairs"])
            add_row(rows, split, name, method, seed, selection, "noiseless_generator", "reconstruction_mse_original_units", mse(reconstructed_o, bundle[split]["noiseless"]))
        if name == "high_variance_nuisance":
            add_row(rows, split, name, method, seed, selection, "structural_features_0_3", "reconstruction_mse_original_units", mse(reconstructed_o[:, :4], observed[:, :4]))
            add_row(rows, split, name, method, seed, selection, "nuisance_contaminated_features_4_5", "reconstruction_mse_original_units", mse(reconstructed_o[:, 4:6], observed[:, 4:6]))
        split_scores[split] = {"transformed_mse": transformed_mse, "original_mse": original_mse}
    add_row(rows, "selection", name, method, seed, selection, "optimizer", "selected_epoch", float(model.get("best_epoch", 0)))
    add_row(rows, "selection", name, method, seed, selection, "optimizer", "selected_validation_loss", float(model.get("best_validation_loss", split_scores["validation"]["transformed_mse"])))
    add_row(rows, "selection", name, method, seed, selection, "architecture", "parameter_count_total", float(model["parameter_count"]))
    add_row(rows, "selection", name, method, seed, selection, "architecture", "parameter_count_trainable", float(model.get("trainable_parameter_count", model["parameter_count"])))
    add_row(rows, "test", name, method, seed, selection, "observations", "generalization_gap_original_units_test_minus_train", split_scores["test"]["original_mse"] - split_scores["train"]["original_mse"])
    return split_scores


def score_condition(rows: list[dict], name: str, bundle: dict, fit: dict) -> dict:
    prep = fit["preprocessing"]
    summaries = {}
    for method in ("mean", "pca"):
        model = fit["models"][method]["deterministic"]
        summaries[method] = score_one_model(rows, name, bundle, prep, method, model, "deterministic", "deterministic_fit")
    for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
        per_seed = {}
        for seed in SEEDS:
            model = fit["models"][method]["candidates"][seed]
            per_seed[seed] = score_one_model(rows, name, bundle, prep, method, model, seed, "validation_checkpoint_per_seed")
        best_seed = fit["models"][method]["best_seed"]
        selected_model = fit["models"][method]["selected"]
        summaries[method] = score_one_model(rows, name, bundle, prep, method, selected_model, "selected", f"best_validation_seed_{best_seed}")
        test_values = np.array([per_seed[seed]["test"]["original_mse"] for seed in SEEDS])
        for metric, value in (
            ("seed_test_mse_mean", np.mean(test_values)),
            ("seed_test_mse_std", np.std(test_values)),
            ("seed_test_mse_min", np.min(test_values)),
            ("seed_test_mse_max", np.max(test_values)),
        ):
            add_row(rows, "test", name, method, "aggregate", "five_predeclared_seeds", "seed_distribution", metric, float(value))
    return summaries


def gradient_check(loss_grad, params: dict, x: np.ndarray, extra=None) -> dict:
    if extra is None:
        loss, grads = loss_grad(params, x)
    else:
        loss, grads = loss_grad(params, extra, x)
    records = []
    epsilon = 1e-6
    keys = sorted(params)
    for number in range(8):
        key = keys[number % len(keys)]
        index = tuple(0 for _ in params[key].shape)
        if params[key].size > 1:
            flat_index = (number * 7 + 1) % params[key].size
            index = np.unravel_index(flat_index, params[key].shape)
        plus = copy_params(params)
        minus = copy_params(params)
        plus[key][index] += epsilon
        minus[key][index] -= epsilon
        if extra is None:
            lp = loss_grad(plus, x)[0]
            lm = loss_grad(minus, x)[0]
        else:
            lp = loss_grad(plus, extra, x)[0]
            lm = loss_grad(minus, extra, x)[0]
        numerical = (lp - lm) / (2.0 * epsilon)
        analytic = float(grads[key][index])
        relative = abs(analytic - numerical) / max(1.0, abs(analytic), abs(numerical))
        records.append({"parameter": key, "index": list(index), "analytic": analytic, "numerical": numerical, "relative_error": relative})
    return {"loss": loss, "maximum_relative_error": max(r["relative_error"] for r in records), "records": records}


def gradient_diagnostics() -> dict:
    x = rng_for(80).normal(size=(7, P))
    linear = initialize_linear(x, 3)
    nonlinear = initialize_nonlinear(SMALL_HIDDEN, 3)
    frozen, decoder = initialize_frozen(SMALL_HIDDEN, 3)
    return {
        "linear_ae": gradient_check(linear_loss_grad, linear, x),
        "nonlinear_ae": gradient_check(nonlinear_loss_grad, nonlinear, x),
        "frozen_random_encoder_decoder": gradient_check(frozen_decoder_loss_grad, decoder, x, extra=frozen),
    }


def fit_signature(fit: dict) -> dict:
    result = {}
    for method in ("mean", "pca"):
        result[method] = model_signature(fit["models"][method]["deterministic"])
    for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
        result[method] = np.concatenate(
            [
                np.array([fit["models"][method]["best_seed"]], dtype=float),
                *[model_signature(fit["models"][method]["candidates"][seed]) for seed in SEEDS],
            ]
        )
    result["preprocessing"] = np.concatenate((fit["preprocessing"]["mean"], fit["preprocessing"]["scale"]))
    return result


def compare_signatures(a: dict, b: dict) -> dict:
    per_method = {}
    for key in a:
        if a[key].shape != b[key].shape:
            discrepancy = float("inf")
        else:
            discrepancy = float(np.max(np.abs(a[key] - b[key]))) if a[key].size else 0.0
        per_method[key] = discrepancy
    return {"per_method_max_abs_discrepancy": per_method, "overall_max_abs_discrepancy": max(per_method.values())}


def histories_diagnostic(a: dict, b: dict) -> dict:
    exact = True
    maximum = 0.0
    for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
        for seed in SEEDS:
            ha = np.asarray(a["models"][method]["candidates"][seed]["history"], dtype=float)
            hb = np.asarray(b["models"][method]["candidates"][seed]["history"], dtype=float)
            same = ha.shape == hb.shape and np.array_equal(ha, hb)
            exact = exact and same
            if ha.shape == hb.shape and ha.size:
                maximum = max(maximum, float(np.max(np.abs(ha - hb))))
            elif ha.shape != hb.shape:
                maximum = float("inf")
    return {"all_complete_histories_exactly_equal": bool(exact), "maximum_abs_discrepancy": maximum}


def fit_reconstruction_diagnostic(a: dict, b: dict, bundle: dict) -> dict:
    maximum_reconstruction = 0.0
    maximum_code = 0.0
    for split in ("train", "validation"):
        xa = transform(bundle[split]["observed"], a["preprocessing"])
        xb = transform(bundle[split]["observed"], b["preprocessing"])
        for method in ("mean", "pca"):
            ma = a["models"][method]["deterministic"]
            mb = b["models"][method]["deterministic"]
            ra, za = predict_and_code(ma, xa)
            rb, zb = predict_and_code(mb, xb)
            maximum_reconstruction = max(maximum_reconstruction, float(np.max(np.abs(ra - rb))))
            maximum_code = max(maximum_code, float(np.max(np.abs(za - zb))))
        for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
            for seed in SEEDS:
                ma = a["models"][method]["candidates"][seed]
                mb = b["models"][method]["candidates"][seed]
                ra, za = predict_and_code(ma, xa)
                rb, zb = predict_and_code(mb, xb)
                maximum_reconstruction = max(maximum_reconstruction, float(np.max(np.abs(ra - rb))))
                maximum_code = max(maximum_code, float(np.max(np.abs(za - zb))))
    return {
        "train_validation_reconstruction_max_abs_discrepancy": maximum_reconstruction,
        "train_validation_code_max_abs_discrepancy": maximum_code,
    }


def poison_diagnostics(all_data: dict, fits: dict) -> dict:
    records = {}
    for name in CONDITIONS:
        base = all_data[name]
        test_poison = {
            split: {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in base[split].items()}
            for split in base
        }
        test_poison["test"]["observed"] = test_poison["test"]["observed"] * 1000.0 + 777.0
        latent_poison = {
            split: {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in base[split].items()}
            for split in base
        }
        for split in latent_poison:
            latent_poison[split]["s"] = np.full_like(latent_poison[split]["s"], 123456.0)
            latent_poison[split]["noiseless"] = np.full_like(latent_poison[split]["noiseless"], -654321.0)
        baseline_fit = fits[name]
        baseline_signature = fit_signature(baseline_fit)
        test_fit = fit_condition(test_poison, name)
        latent_fit = fit_condition(latent_poison, name)
        test_signature = compare_signatures(baseline_signature, fit_signature(test_fit))
        latent_signature = compare_signatures(baseline_signature, fit_signature(latent_fit))
        test_history = histories_diagnostic(baseline_fit, test_fit)
        latent_history = histories_diagnostic(baseline_fit, latent_fit)
        latent_reconstruction = fit_reconstruction_diagnostic(baseline_fit, latent_fit, base)
        test_signature.update({
            "complete_checkpoint_histories": test_history,
        })
        latent_signature.update({
            "complete_checkpoint_histories": latent_history,
            "train_validation_outputs": latent_reconstruction,
        })
        records[name] = {
            "test_observation_poison": test_signature,
            "latent_truth_poison": latent_signature,
        }
    return records


def checkpoint_diagnostics(fits: dict, all_data: dict) -> dict:
    maximum = 0.0
    records = {}
    for name in CONDITIONS:
        prep = fits[name]["preprocessing"]
        validation = transform(all_data[name]["validation"]["observed"], prep)
        records[name] = {}
        for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
            records[name][method] = {}
            for seed in SEEDS:
                model = fits[name]["models"][method]["candidates"][seed]
                prediction = predict_and_code(model, validation)[0]
                recomputed = mse(prediction, validation)
                discrepancy = abs(recomputed - model["best_validation_loss"])
                replay_loss = float("inf")
                replay_epoch = 0
                for epoch, _train_loss, validation_loss in model["history"]:
                    if validation_loss < replay_loss - MIN_DELTA:
                        replay_loss = validation_loss
                        replay_epoch = epoch
                replay_discrepancy = abs(replay_loss - model["best_validation_loss"])
                maximum = max(maximum, discrepancy, replay_discrepancy)
                records[name][method][str(seed)] = {
                    "selected_epoch": model["best_epoch"],
                    "epochs_run": model["epochs_run"],
                    "stored_validation_loss": model["best_validation_loss"],
                    "recomputed_validation_loss": recomputed,
                    "absolute_discrepancy": discrepancy,
                    "history_replay_epoch": replay_epoch,
                    "history_replay_validation_loss": replay_loss,
                    "history_replay_absolute_discrepancy": replay_discrepancy,
                    "history_replay_epoch_matches": bool(replay_epoch == model["best_epoch"]),
                }
            recorded_seed = fits[name]["models"][method]["best_seed"]
            replay_seed = min(
                SEEDS,
                key=lambda candidate_seed: (
                    fits[name]["models"][method]["candidates"][candidate_seed]["best_validation_loss"],
                    candidate_seed,
                ),
            )
            records[name][method]["seed_selection"] = {
                "recorded_seed": recorded_seed,
                "replayed_seed": replay_seed,
                "matches": bool(recorded_seed == replay_seed),
            }
    all_history_epochs_match = all(
        records[name][method][str(seed)]["history_replay_epoch_matches"]
        for name in CONDITIONS for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder") for seed in SEEDS
    )
    all_seed_selections_match = all(
        records[name][method]["seed_selection"]["matches"]
        for name in CONDITIONS for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder")
    )
    return {
        "maximum_absolute_discrepancy": maximum,
        "all_history_replay_epochs_match": all_history_epochs_match,
        "all_seed_selections_match": all_seed_selections_match,
        "records": records,
    }


def linear_pca_diagnostics(fits: dict, all_data: dict) -> dict:
    records = {}
    for name in CONDITIONS:
        prep = fits[name]["preprocessing"]
        x_train = transform(all_data[name]["train"]["observed"], prep)
        pca = fits[name]["models"]["pca"]["deterministic"]
        linear = fits[name]["models"]["linear_ae"]["selected"]
        pca_prediction = predict_and_code(pca, x_train)[0]
        linear_prediction = predict_and_code(linear, x_train)[0]
        projector_pca = pca["basis"] @ pca["basis"].T
        composite = linear["params"]["we"] @ linear["params"]["wd"]
        projector_error = float(np.linalg.norm(composite - projector_pca, ord="fro"))
        encoder_direction = linear["params"]["we"][:, 0]
        cosine = abs(float(np.dot(encoder_direction, pca["basis"][:, 0]))) / max(float(np.linalg.norm(encoder_direction)), 1e-15)
        subspace_angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        pca_training_mse = mse(pca_prediction, x_train)
        linear_training_mse = mse(linear_prediction, x_train)
        mse_gap = abs(linear_training_mse - pca_training_mse)
        relative_mse_gap = mse_gap / max(abs(pca_training_mse), 1e-15)
        records[name] = {
            "selected_seed": fits[name]["models"]["linear_ae"]["best_seed"],
            "composite_projector_frobenius_error": projector_error,
            "encoder_to_pca_subspace_angle_degrees": subspace_angle,
            "pca_training_objective_transformed_mse": pca_training_mse,
            "linear_ae_training_objective_transformed_mse": linear_training_mse,
            "training_objective_transformed_mse_absolute_gap": mse_gap,
            "training_objective_transformed_mse_relative_gap": relative_mse_gap,
            "passes_declared_tolerance": bool(
                projector_error <= TOL["linear_pca_projector"]
                and subspace_angle <= TOL["linear_pca_angle_degrees"]
                and mse_gap <= TOL["linear_pca_mse"]
            ),
            "interpretation_if_false": "bounded linear-autoencoder optimization failure; not a failure of the linear autoencoder/PCA equivalence result",
        }
    return records


def adam_determinism_diagnostic(all_data: dict, fits: dict) -> dict:
    name = "nominal_curved"
    prep = fits[name]["preprocessing"]
    train = transform(all_data[name]["train"]["observed"], prep)
    validation = transform(all_data[name]["validation"]["observed"], prep)
    repeated = train_seeded_method("nonlinear_ae", train, validation, 2, SMALL_HIDDEN)
    reference = fits[name]["models"]["nonlinear_ae"]["candidates"][2]
    discrepancy = float(np.max(np.abs(model_signature(repeated) - model_signature(reference))))
    history_equal = repeated["history"] == reference["history"]
    return {"condition": name, "method": "nonlinear_ae", "seed": 2, "model_max_abs_discrepancy": discrepancy, "history_exactly_equal": history_equal}


def regeneration_diagnostic(all_data: dict) -> dict:
    regenerated = generate_all()
    records = {}
    all_exact = True
    for name in CONDITIONS:
        records[name] = {}
        for split in ("train", "validation", "test"):
            exact = all(
                np.array_equal(all_data[name][split][key], regenerated[name][split][key], equal_nan=True)
                for key in ("observed", "s", "noiseless")
            )
            records[name][split] = exact
            all_exact = all_exact and exact
    return {"all_exact": all_exact, "records": records}


def parameter_count_diagnostic(fits: dict) -> dict:
    records = {}
    for name in CONDITIONS:
        records[name] = {}
        for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
            model = fits[name]["models"][method]["selected"]
            if method == "linear_ae":
                expected_total = P + 1 + P + P
                expected_trainable = expected_total
            elif method == "nonlinear_ae":
                h = fits[name]["hidden"]
                expected_total = P * h + h + h + 1 + h + h + h * P + P
                expected_trainable = expected_total
            else:
                h = fits[name]["hidden"]
                encoder = P * h + h + h + 1
                decoder = h + h + h * P + P
                expected_total = encoder + decoder
                expected_trainable = decoder
            records[name][method] = {
                "recorded_total": model["parameter_count"],
                "expected_total": expected_total,
                "recorded_trainable": model["trainable_parameter_count"],
                "expected_trainable": expected_trainable,
                "pass": bool(model["parameter_count"] == expected_total and model["trainable_parameter_count"] == expected_trainable),
            }
    return records


def frozen_encoder_immutability_diagnostic(fits: dict) -> dict:
    maximum = 0.0
    exact = True
    records = {}
    for name in CONDITIONS:
        records[name] = {}
        for seed in SEEDS:
            model = fits[name]["models"]["frozen_random_encoder"]["candidates"][seed]
            key_records = {}
            for key in sorted(model["frozen"]):
                before = model["frozen_initial"][key]
                after = model["frozen"][key]
                discrepancy = float(np.max(np.abs(before - after)))
                equal = bool(np.array_equal(before, after))
                maximum = max(maximum, discrepancy)
                exact = exact and equal
                key_records[key] = {"bitwise_equal": equal, "max_abs_discrepancy": discrepancy}
            records[name][str(seed)] = {
                "initialized_frozen_arrays": {key: model["frozen_initial"][key].tolist() for key in sorted(model["frozen_initial"])},
                "post_training_frozen_arrays": {key: model["frozen"][key].tolist() for key in sorted(model["frozen"])},
                "comparison": key_records,
            }
    return {"all_bitwise_equal": bool(exact), "maximum_abs_discrepancy": maximum, "records": records}


def initialization_registry() -> dict:
    return {
        "linear_ae": {
            "encoder_weights": "rank-1 training PCA basis plus N(0,0.02^2) seed noise",
            "decoder_weights": "transpose of rank-1 training PCA basis plus N(0,0.02^2) seed noise",
            "encoder_bias": "zeros",
            "decoder_bias": "zeros",
            "rng": "rng_for(30, seed)",
        },
        "nonlinear_ae": {
            "weight_distribution": "independent centered Normal with Glorot-like scale sqrt(2/(fan_in+fan_out))",
            "w1_scale": "sqrt(2/(6+H))",
            "w2_scale": "sqrt(2/(H+1))",
            "w3_scale": "sqrt(2/(1+H))",
            "w4_scale": "sqrt(2/(H+6))",
            "all_biases": "zeros",
            "rng": "rng_for(31, hidden_width, seed)",
        },
        "frozen_random_encoder": {
            "encoder_w1": "N(0,1/6)",
            "encoder_b1": "N(0,0.08^2)",
            "encoder_w2": "N(0,1/H)",
            "encoder_b2": "N(0,0.05^2)",
            "decoder_w3": "N(0,2/(H+1))",
            "decoder_b3": "zeros",
            "decoder_w4": "N(0,2/(H+6))",
            "decoder_b4": "zeros",
            "rng": "rng_for(32, hidden_width, seed)",
        },
    }


def architecture_diagnostic(fits: dict, all_data: dict) -> dict:
    expected_keys = {
        "mean": {"mean"},
        "pca": {"mean", "basis", "eigenvalues"},
        "linear_ae": {"we", "be", "wd", "bd"},
        "nonlinear_ae": {"w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"},
        "frozen_random_encoder_trainable": {"w3", "b3", "w4", "b4"},
        "frozen_random_encoder_fixed": {"w1", "b1", "w2", "b2"},
    }
    graph_edges = {
        "linear_ae": ["x->code", "code->output"],
        "nonlinear_ae": ["x->hidden_encoder", "hidden_encoder->code", "code->hidden_decoder", "hidden_decoder->output"],
        "frozen_random_encoder": ["x->fixed_hidden_encoder", "fixed_hidden_encoder->code", "code->trained_hidden_decoder", "trained_hidden_decoder->output"],
    }
    unexpected = []
    shape_records = []
    behavioral_no_skip_records = []
    all_shapes = True
    for name in CONDITIONS:
        sample = transform(all_data[name]["train"]["observed"][:7], fits[name]["preprocessing"])
        for method in ("mean", "pca"):
            model = fits[name]["models"][method]["deterministic"]
            _reconstruction, code = predict_and_code(model, sample)
            all_shapes = all_shapes and code.shape == (sample.shape[0], 1)
            shape_records.append({"condition": name, "method": method, "seed": "deterministic", "observed_shape": list(code.shape)})
        for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
            for seed in SEEDS:
                model = fits[name]["models"][method]["candidates"][seed]
                _reconstruction, code = predict_and_code(model, sample)
                all_shapes = all_shapes and code.shape == (sample.shape[0], 1)
                shape_records.append({"condition": name, "method": method, "seed": seed, "observed_shape": list(code.shape)})
                if method == "linear_ae":
                    extra = sorted(set(model["params"]) - expected_keys["linear_ae"])
                elif method == "nonlinear_ae":
                    extra = sorted(set(model["params"]) - expected_keys["nonlinear_ae"])
                else:
                    extra = sorted(set(model["params"]) - expected_keys["frozen_random_encoder_trainable"])
                    extra += sorted(set(model["frozen"]) - expected_keys["frozen_random_encoder_fixed"])
                if extra:
                    unexpected.append({"condition": name, "method": method, "seed": seed, "unexpected_keys": extra})
                zeroed = {key: value for key, value in model.items()}
                if method == "linear_ae":
                    zeroed["params"] = {key: np.array(value, copy=True) for key, value in model["params"].items()}
                    zeroed["params"]["wd"][:] = 0.0; zeroed["params"]["bd"][:] = 0.0
                elif method == "nonlinear_ae":
                    zeroed["params"] = {key: np.array(value, copy=True) for key, value in model["params"].items()}
                    zeroed["params"]["w4"][:] = 0.0; zeroed["params"]["b4"][:] = 0.0
                else:
                    zeroed["params"] = {key: np.array(value, copy=True) for key, value in model["params"].items()}
                    zeroed["frozen"] = {key: np.array(value, copy=True) for key, value in model["frozen"].items()}
                    zeroed["params"]["w4"][:] = 0.0; zeroed["params"]["b4"][:] = 0.0
                zeroed_output, _ = predict_and_code(zeroed, sample[:2])
                behavioral_no_skip_records.append({
                    "condition": name, "method": method, "seed": seed,
                    "distinct_input_norm": float(np.linalg.norm(sample[0] - sample[1])),
                    "zero_decoder_output_difference": float(np.max(np.abs(zeroed_output[0] - zeroed_output[1]))),
                    "zero_decoder_output_max_abs": float(np.max(np.abs(zeroed_output))),
                    "passes": bool(
                        np.linalg.norm(sample[0] - sample[1]) > 0.0
                        and np.max(np.abs(zeroed_output[0] - zeroed_output[1])) == 0.0
                        and np.max(np.abs(zeroed_output)) == 0.0
                    ),
                })
            selected_model = fits[name]["models"][method]["selected"]
            _selected_reconstruction, selected_code = predict_and_code(selected_model, sample)
            all_shapes = all_shapes and selected_code.shape == (sample.shape[0], 1)
            shape_records.append({
                "condition": name, "method": method, "seed": "selected",
                "selected_seed": fits[name]["models"][method]["best_seed"],
                "observed_shape": list(selected_code.shape),
            })
    direct_input_output_edges = [edge for edges in graph_edges.values() for edge in edges if edge == "x->output"]
    behavioral_no_skip_pass = bool(all(record["passes"] for record in behavioral_no_skip_records))
    return {
        "bottleneck_dimension": D,
        "all_runtime_code_shapes_n_by_1": bool(all_shapes),
        "runtime_code_shape_records": shape_records,
        "allowed_parameter_keys": {key: sorted(value) for key, value in expected_keys.items()},
        "unexpected_parameter_keys": unexpected,
        "forward_graph_edges": graph_edges,
        "direct_input_to_output_edges": direct_input_output_edges,
        "behavioral_no_skip_records": behavioral_no_skip_records,
        "behavioral_no_skip_pass": behavioral_no_skip_pass,
        "skip_paths_derived": bool(direct_input_output_edges or unexpected or not behavioral_no_skip_pass),
        "small_architecture": "6->8->1->8->6",
        "few_sample_high_capacity_architecture": "6->32->1->32->6",
        "linear_architecture": "6->1->6",
        "frozen_random_encoder": "6->H->1 fixed; 1->H->6 trained",
        "condition_hidden_widths": {name: fits[name]["hidden"] for name in CONDITIONS},
        "capacity_registry": {
            "ordinary_width_8": {
                "nonlinear_total_and_trainable": 135,
                "frozen_encoder_fixed": 65,
                "frozen_decoder_trainable": 70,
                "frozen_total": 135,
            },
            "few_sample_width_32": {
                "nonlinear_total_and_trainable": 519,
                "frozen_encoder_fixed": 257,
                "frozen_decoder_trainable": 262,
                "frozen_total": 519,
            },
        },
    }


def moment_matching_diagnostic(all_data: dict) -> dict:
    quadrature = curved_population_moments()
    target_mean, target_covariance = full_gaussian_parameters()
    recomputed = curved_population_moments()
    target_mean_discrepancy = float(np.max(np.abs(target_mean - recomputed["mean"])))
    target_covariance_discrepancy = float(np.max(np.abs(target_covariance - recomputed["observable_covariance"])))
    samples = {}
    for split in ("train", "validation", "test"):
        x = all_data["full_dimensional_gaussian"][split]["observed"]
        sample_mean = np.mean(x, axis=0)
        centered = x - sample_mean
        sample_covariance = centered.T @ centered / x.shape[0]
        samples[split] = {
            "sample_mean": sample_mean.tolist(),
            "sample_covariance": sample_covariance.tolist(),
            "sample_mean_max_abs_deviation_from_target": float(np.max(np.abs(sample_mean - target_mean))),
            "sample_covariance_max_abs_deviation_from_target": float(np.max(np.abs(sample_covariance - target_covariance))),
        }
    return {
        "quadrature_family": "numpy.polynomial.legendre.leggauss",
        "quadrature_nodes": QUADRATURE_NODES,
        "uniform_interval": [-1.5, 1.5],
        "probability_weight_rule": "mapped nodes s=1.5*x; probability weights=w/2",
        "target_mean": target_mean.tolist(),
        "target_signal_covariance": quadrature["signal_covariance"].tolist(),
        "target_observable_covariance_including_noise_0.06_squared_I": target_covariance.tolist(),
        "target_observable_covariance_eigenvalues_descending": quadrature["eigenvalues_descending"].tolist(),
        "minimum_target_eigenvalue": float(np.min(quadrature["eigenvalues_descending"])),
        "executable_target_mean_max_abs_discrepancy": target_mean_discrepancy,
        "executable_target_covariance_max_abs_discrepancy": target_covariance_discrepancy,
        "finite_sample_deviations": samples,
        "no_s_truth_in_full_dimensional_bundle": bool(
            all(np.all(np.isnan(all_data["full_dimensional_gaussian"][split]["s"])) for split in ("train", "validation", "test"))
        ),
    }


def shifted_reuse_diagnostic(all_data: dict, fits: dict) -> dict:
    split_records = {}
    for split in ("train", "validation"):
        split_records[split] = {
            key: bool(np.array_equal(
                all_data["nominal_curved"][split][key],
                all_data["shifted_range"][split][key],
                equal_nan=True,
            ))
            for key in ("observed", "s", "noiseless")
        }
        split_records[split]["same_bundle_object"] = bool(
            all_data["nominal_curved"][split] is all_data["shifted_range"][split]
        )
    signatures = compare_signatures(fit_signature(fits["nominal_curved"]), fit_signature(fits["shifted_range"]))
    selections_equal = all(
        fits["nominal_curved"]["models"][method]["best_seed"]
        == fits["shifted_range"]["models"][method]["best_seed"]
        for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder")
    )
    histories = histories_diagnostic(fits["nominal_curved"], fits["shifted_range"])
    return {
        "train_validation_exact": split_records,
        "same_fit_object": bool(fits["nominal_curved"] is fits["shifted_range"]),
        "fit_signature": signatures,
        "complete_checkpoint_histories": histories,
        "selected_seeds_equal": bool(selections_equal),
        "only_test_bundle_differs": bool(
            all(all(value for key, value in record.items() if key != "same_bundle_object") for record in split_records.values())
            and not np.array_equal(all_data["nominal_curved"]["test"]["observed"], all_data["shifted_range"]["test"]["observed"])
        ),
    }


def selected_metric(rows: list[dict], condition: str, method: str, split: str, metric: str, target: str = "observations") -> float:
    matches = [
        row["value"] for row in rows
        if row["condition"] == condition and row["method"] == method and row["split"] == split
        and row["seed"] in ("selected", "deterministic") and row["metric"] == metric and row["target"] == target
    ]
    if len(matches) != 1:
        raise ValueError((condition, method, split, metric, target, len(matches)))
    return float(matches[0])


def scientific_summaries(rows: list[dict], fits: dict) -> dict:
    summaries = {}
    for name in CONDITIONS:
        summaries[name] = {
            method: {
                "test_original_mse": selected_metric(rows, name, method, "test", "reconstruction_mse_original_units"),
                "test_transformed_mse": selected_metric(rows, name, method, "test", "reconstruction_mse_transformed"),
            }
            for method in ("mean", "pca", "linear_ae", "nonlinear_ae", "frozen_random_encoder")
        }
        summaries[name]["selected_seeds"] = {
            method: fits[name]["models"][method]["best_seed"]
            for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder")
        }
    summaries["few_sample_memorization_control"] = {
        method: {
            "train_original_mse": selected_metric(rows, "few_sample_high_capacity", method, "train", "reconstruction_mse_original_units"),
            "test_original_mse": selected_metric(rows, "few_sample_high_capacity", method, "test", "reconstruction_mse_original_units"),
            "test_minus_train_gap": selected_metric(rows, "few_sample_high_capacity", method, "test", "generalization_gap_original_units_test_minus_train"),
        }
        for method in ("pca", "linear_ae", "nonlinear_ae", "frozen_random_encoder")
    }
    summaries["full_dimensional_control_interpretation"] = (
        "This condition has no one-dimensional generator target. A one-dimensional bottleneck score is only reconstruction compression performance and cannot support a one-dimensional-structure, manifold, or intrinsic-dimension claim."
    )
    summaries["shifted_range_interpretation"] = (
        "The shifted test split lies outside the training/validation parameter range; its score is extrapolation under a frozen training-only fit, not in-range generalization."
    )
    return summaries


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["value"] = format(float(record["value"]), ".17g")
            writer.writerow(record)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float) -> str:
    return f"{value:.4g}"


def make_summary(path: Path, all_data: dict, fits: dict, diagnostics: dict, rows: list[dict]) -> None:
    width, height = 1500, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 14), "LDG bounded nonlinear reconstruction - deterministic diagnostic", fill="black", font=font)

    def panel(box, title):
        draw.rectangle(box, outline="black", width=2)
        draw.text((box[0] + 8, box[1] + 7), title, fill="black", font=font)

    panel((24, 44, 750, 500), "A. Nominal curved test: generator and selected reconstructions")
    name = "nominal_curved"
    prep = fits[name]["preprocessing"]
    order = np.argsort(all_data[name]["test"]["s"])
    s = all_data[name]["test"]["s"][order]
    truth = all_data[name]["test"]["noiseless"][order, 1]
    methods = ("pca", "nonlinear_ae", "frozen_random_encoder")
    colors = {"truth": (0, 0, 0), "pca": (31, 119, 180), "nonlinear_ae": (214, 39, 40), "frozen_random_encoder": (44, 160, 44)}
    curves = {"truth": truth}
    x_t = transform(all_data[name]["test"]["observed"], prep)
    for method in methods:
        model = fits[name]["models"][method]["deterministic"] if method == "pca" else fits[name]["models"][method]["selected"]
        rec = inverse_transform(predict_and_code(model, x_t)[0], prep)
        curves[method] = rec[order, 1]
    ymin = min(float(np.min(v)) for v in curves.values())
    ymax = max(float(np.max(v)) for v in curves.values())
    for label, values in curves.items():
        points = []
        for sx, vy in zip(s, values):
            px = 50 + (sx + 1.5) / 3.0 * 670
            py = 460 - (vy - ymin) / max(ymax - ymin, 1e-12) * 360
            points.append((px, py))
        draw.line(points, fill=colors[label], width=2)
    draw.text((48, 475), "black=generator; blue=PCA; red=nonlinear AE; green=frozen encoder", fill="black", font=font)

    panel((775, 44, 1475, 500), "B. Selected test MSE in original units (lower is better)")
    methods_all = ("mean", "pca", "linear_ae", "nonlinear_ae", "frozen_random_encoder")
    for i, condition in enumerate(CONDITIONS):
        draw.text((790, 78 + i * 78), condition, fill="black", font=font)
        values = [selected_metric(rows, condition, method, "test", "reconstruction_mse_original_units") for method in methods_all]
        scale = max(values)
        for j, (method, value) in enumerate(zip(methods_all, values)):
            y = 98 + i * 78 + j * 10
            draw.rectangle((1010, y, 1010 + 290 * value / max(scale, 1e-12), y + 7), fill=(40 + 35 * j, 80, 180 - 25 * j))
            draw.text((1310, y - 2), f"{method[:8]} {fmt(value)}", fill="black", font=font)

    panel((24, 525, 750, 970), "C. Five-seed selected nonlinear AE test-MSE distributions")
    for i, condition in enumerate(CONDITIONS):
        records = [
            row for row in rows if row["condition"] == condition and row["method"] == "nonlinear_ae"
            and row["split"] == "test" and row["metric"] == "reconstruction_mse_original_units"
            and row["target"] == "observations" and row["seed"] in {str(seed) for seed in SEEDS}
        ]
        values = [float(row["value"]) for row in records]
        draw.text((42, 565 + i * 72), f"{condition}: min {fmt(min(values))} | mean {fmt(float(np.mean(values)))} | max {fmt(max(values))}", fill="black", font=font)
        lo, hi = min(values), max(values)
        for value in values:
            px = 55 + (value - lo) / max(hi - lo, 1e-12) * 640
            draw.ellipse((px - 4, 590 + i * 72 - 4, px + 4, 590 + i * 72 + 4), fill=(214, 39, 40))

    panel((775, 525, 1475, 970), "D. Verification and interpretation boundaries")
    poison = diagnostics["leakage_poison_checks"]
    poison_max = max(
        record[kind]["overall_max_abs_discrepancy"]
        for record in poison.values() for kind in ("test_observation_poison", "latent_truth_poison")
    )
    grad_max = max(record["maximum_relative_error"] for record in diagnostics["gradient_checks"].values())
    few = diagnostics["scientific_summaries"]["few_sample_memorization_control"]["nonlinear_ae"]
    lines = [
        f"all-condition poison maximum: {fmt(poison_max)}",
        f"gradient-check maximum relative error: {fmt(grad_max)}",
        f"checkpoint recomputation maximum: {fmt(diagnostics['checkpoint_recomputation']['maximum_absolute_discrepancy'])}",
        f"Adam repeat discrepancy: {fmt(diagnostics['adam_determinism']['model_max_abs_discrepancy'])}",
        f"few-sample nonlinear train/test MSE: {fmt(few['train_original_mse'])} / {fmt(few['test_original_mse'])}",
        "All preprocessing is fit on training observations only.",
        "Validation selects checkpoints and seeds; test never selects.",
        "The full-dimensional Gaussian has no 1D generator target.",
        "Generator-parameter correlation is simulation-side only.",
        "Reconstruction is not manifold, intrinsic-dimension, or mechanism recovery.",
    ]
    for i, line in enumerate(lines):
        draw.text((792, 570 + i * 35), line, fill="black", font=font)
    image.save(path, format="PNG", optimize=False)


def build(root: Path, quiet: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    all_data = generate_all()
    fits = {
        name: fit_condition(all_data[name], name)
        for name in CONDITIONS
        if name != "shifted_range"
    }
    fits["shifted_range"] = fits["nominal_curved"]
    rows: list[dict] = []
    score_summaries = {name: score_condition(rows, name, all_data[name], fits[name]) for name in CONDITIONS}
    gradients = gradient_diagnostics()
    poison = poison_diagnostics(all_data, fits)
    checkpoints = checkpoint_diagnostics(fits, all_data)
    linear_pca = linear_pca_diagnostics(fits, all_data)
    adam = adam_determinism_diagnostic(all_data, fits)
    regeneration = regeneration_diagnostic(all_data)
    parameter_counts = parameter_count_diagnostic(fits)
    architecture = architecture_diagnostic(fits, all_data)
    frozen_immutability = frozen_encoder_immutability_diagnostic(fits)
    initialization = initialization_registry()
    moment_matching = moment_matching_diagnostic(all_data)
    shifted_reuse = shifted_reuse_diagnostic(all_data, fits)
    scientific = scientific_summaries(rows, fits)
    diagnostics = {
        "generator": {
            "ambient_dimension": P,
            "bottleneck_dimension": D,
            "curved_equation": "[s, s^2, 0.35 sin(2s), 0.2 s^3, cos(s)-E_train cos(s), 0.4 sin(3s)] + epsilon",
            "nominal_s_range": [-1.5, 1.5],
            "shifted_test_s_support": "symmetric exterior [-2.4,-1.5] union [1.5,2.4]",
            "noise_sd": NOISE_SD,
            "nuisance_sd_features_4_5": NUISANCE_SD,
            "split_sizes": SPLIT_SIZES,
            "independent_split_rng": True,
            "full_dimensional_control": "full-rank Gaussian observations exactly population-moment matched by fixed quadrature to the nominal noisy curved observable law; no s truth exists",
            "full_dimensional_moment_source": "256-node deterministic Gauss-Legendre quadrature including 0.06^2 I observation noise",
        },
        "optimizer_contract": {
            "optimizer": "manual deterministic full-batch Adam",
            "seeds": list(SEEDS),
            "maximum_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rates": LEARNING_RATE,
            "minimum_validation_improvement": MIN_DELTA,
            "checkpoint_rule": "minimum validation transformed-coordinate MSE per seed",
            "seed_selection_rule": "minimum selected validation MSE, then lower seed",
            "test_excluded_from_checkpoint_and_seed_selection": True,
        },
        "score_summaries": score_summaries,
        "scientific_summaries": scientific,
        "gradient_checks": gradients,
        "leakage_poison_checks": poison,
        "checkpoint_recomputation": checkpoints,
        "linear_autoencoder_vs_pca": linear_pca,
        "adam_determinism": adam,
        "exact_split_regeneration": regeneration,
        "parameter_counts": parameter_counts,
        "architecture_checks": architecture,
        "frozen_encoder_immutability": frozen_immutability,
        "initialization_registry": initialization,
        "moment_matched_full_dimensional_gaussian": moment_matching,
        "shifted_range_exact_reuse": shifted_reuse,
        "scientific_limits": [
            "Low held-out reconstruction error is not evidence of a manifold, intrinsic dimension, unique coordinates, mechanism, or causality.",
            "Code correlation with s and code collisions are generator-side simulation diagnostics unavailable as ground truth in ordinary observations.",
            "A one-dimensional bottleneck applied to full-dimensional Gaussian data does not establish one-dimensional latent structure.",
            "The shifted-range condition evaluates frozen-model extrapolation outside training support, not interpolation.",
            "Few-sample high-capacity performance is interpreted with its train-test gap and seed dispersion; a low training loss alone is not recovery.",
            "The frozen random encoder is a capacity/control comparison, not a learned representation claim.",
        ],
    }
    write_csv(root / "metrics.csv", rows)
    (root / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n", encoding="utf-8"
    )
    make_summary(root / "summary.png", all_data, fits, diagnostics, rows)
    hashes = {name: sha256(root / name) for name in ARTIFACT_NAMES}
    manifest = {
        "schema_version": 1,
        "project": "Latent Dynamics & Geometry",
        "artifact": "bounded-nonlinear-reconstruction",
        "date": DATE,
        "master_seed": MASTER_SEED,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": getattr(sys.modules.get("PIL"), "__version__", "unknown"),
            "platform": platform.platform(),
        },
        "dimensions": {"ambient": P, "bottleneck": D},
        "conditions": list(CONDITIONS),
        "split_sizes": SPLIT_SIZES,
        "methods": ["mean", "rank_1_pca", "linear_autoencoder", "small_or_bounded_high_capacity_tanh_autoencoder", "frozen_random_tanh_encoder_trained_decoder"],
        "architectures": architecture,
        "initialization_registry": initialization,
        "moment_matching_contract": {
            "quadrature_family": "numpy.polynomial.legendre.leggauss",
            "quadrature_nodes": QUADRATURE_NODES,
            "observable_covariance_includes_noise_variance": NOISE_SD ** 2,
        },
        "shifted_range_fit_policy": "reuse exact nominal train/validation bundles, standardizer, candidate histories/checkpoints, selected seeds, and fitted models; only test differs",
        "optimizer_contract": diagnostics["optimizer_contract"],
        "metric_schema": list(METRIC_FIELDS),
        "metric_rows": len(rows),
        "tolerances": TOL,
        "artifact_hashes_sha256": hashes,
        "default_command": "python3 toy-models/bounded-nonlinear-reconstruction.py",
        "explicit_commands": [
            "python3 toy-models/bounded-nonlinear-reconstruction.py --generate",
            "python3 toy-models/bounded-nonlinear-reconstruction.py --verify",
            "python3 toy-models/bounded-nonlinear-reconstruction.py --generate --verify",
        ],
        "artifact_files": ["manifest.json", *ARTIFACT_NAMES],
        "interpretation_boundary": "Reconstruction and simulation-side code diagnostics do not prove a manifold, intrinsic dimension, recovered coordinate, mechanism, or causal generator.",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n", encoding="utf-8"
    )
    if not quiet:
        nominal = scientific["nominal_curved"]
        print(f"wrote {root}")
        print(f"nominal PCA test MSE: {nominal['pca']['test_original_mse']:.6f}")
        print(f"nominal nonlinear AE test MSE: {nominal['nonlinear_ae']['test_original_mse']:.6f}")
        print(f"shifted nonlinear AE test MSE: {scientific['shifted_range']['nonlinear_ae']['test_original_mse']:.6f}")
        print(f"metric rows: {len(rows)}")
    return {"manifest": manifest, "diagnostics": diagnostics, "rows": rows}


def verify_metrics(path: Path, expected_rows: int) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise AssertionError("unexpected metrics header")
        seen = set()
        rows = list(reader)
        if len(rows) != expected_rows:
            raise AssertionError(f"metric row count mismatch: {len(rows)} != {expected_rows}")
        for line_number, row in enumerate(rows, start=2):
            if any(row[field] == "" for field in METRIC_FIELDS):
                raise AssertionError(f"empty metric label at line {line_number}")
            value = float(row["value"])
            if not np.isfinite(value):
                raise AssertionError(f"non-finite metric at line {line_number}")
            key = tuple(row[field] for field in METRIC_FIELDS[:-1])
            if key in seen:
                raise AssertionError(f"duplicate metric key at line {line_number}: {key}")
            seen.add(key)


def verify_collision_metrics(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = {
        "s_distance_threshold", "code_distance_fraction_of_range", "code_range",
        "candidate_pair_count", "qualifying_close_code_pair_count",
        "collision_pair_count", "collision_fraction_of_close_code_pairs",
    }
    groups = {}
    for row in rows:
        if row["target"] != "generator_parameter_s_simulation_diagnostic" or row["metric"] not in names:
            continue
        key = (row["condition"], row["split"], row["method"], row["seed"], row["selection"])
        groups.setdefault(key, {})[row["metric"]] = float(row["value"])
    if not groups:
        raise AssertionError("collision diagnostics missing")
    for key, values in groups.items():
        if set(values) != names:
            raise AssertionError(f"incomplete collision diagnostic: {key}")
        condition, split, _method, _seed, _selection = key
        n = SPLIT_SIZES[condition][split]
        expected_candidates = n * (n - 1) // 2
        if values["s_distance_threshold"] != COLLISION_S_DISTANCE:
            raise AssertionError(f"collision s threshold mismatch: {key}")
        if values["code_distance_fraction_of_range"] != COLLISION_CODE_RANGE_FRACTION:
            raise AssertionError(f"collision code threshold mismatch: {key}")
        if int(values["candidate_pair_count"]) != expected_candidates:
            raise AssertionError(f"collision candidate count mismatch: {key}")
        close = int(values["qualifying_close_code_pair_count"])
        collision = int(values["collision_pair_count"])
        if not (0 <= collision <= close <= expected_candidates):
            raise AssertionError(f"collision counts invalid: {key}")
        expected_fraction = collision / close if close else 0.0
        if abs(values["collision_fraction_of_close_code_pairs"] - expected_fraction) > 1e-15:
            raise AssertionError(f"collision fraction mismatch: {key}")



def verify(root: Path, quiet: bool = False) -> None:
    required = ("manifest.json", *ARTIFACT_NAMES)
    for name in required:
        if not (root / name).is_file():
            raise AssertionError(f"missing artifact: {name}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    if manifest["artifact"] != "bounded-nonlinear-reconstruction":
        raise AssertionError("wrong artifact identifier")
    if manifest["dimensions"] != {"ambient": P, "bottleneck": D}:
        raise AssertionError("dimension contract mismatch")
    for name, expected in manifest["artifact_hashes_sha256"].items():
        actual = sha256(root / name)
        if actual != expected:
            raise AssertionError(f"hash mismatch for {name}")
    verify_metrics(root / "metrics.csv", manifest["metric_rows"])
    verify_collision_metrics(root / "metrics.csv")
    if manifest["date"] != DATE:
        raise AssertionError("artifact date mismatch")
    moment = diagnostics["moment_matched_full_dimensional_gaussian"]
    if moment["quadrature_nodes"] != QUADRATURE_NODES:
        raise AssertionError("quadrature node contract mismatch")
    if moment["executable_target_mean_max_abs_discrepancy"] != 0.0 or moment["executable_target_covariance_max_abs_discrepancy"] != 0.0:
        raise AssertionError("moment target construction mismatch")
    if moment["minimum_target_eigenvalue"] <= 0.0 or not moment["no_s_truth_in_full_dimensional_bundle"]:
        raise AssertionError("full-dimensional Gaussian rank/truth contract failed")
    shifted = diagnostics["shifted_range_exact_reuse"]
    if not shifted["same_fit_object"] or not shifted["selected_seeds_equal"] or not shifted["only_test_bundle_differs"]:
        raise AssertionError("shifted-range exact reuse failed")
    if shifted["fit_signature"]["overall_max_abs_discrepancy"] != 0.0:
        raise AssertionError("shifted-range fit signature differs")
    if not shifted["complete_checkpoint_histories"]["all_complete_histories_exactly_equal"]:
        raise AssertionError("shifted-range histories differ")
    grad_max = max(item["maximum_relative_error"] for item in diagnostics["gradient_checks"].values())
    if grad_max > TOL["gradient_relative"]:
        raise AssertionError(f"gradient check failed: {grad_max}")
    poison_max = max(
        item[kind]["overall_max_abs_discrepancy"]
        for item in diagnostics["leakage_poison_checks"].values()
        for kind in ("test_observation_poison", "latent_truth_poison")
    )
    if poison_max > TOL["poison"]:
        raise AssertionError(f"leakage poison failed: {poison_max}")
    for condition, poison_record in diagnostics["leakage_poison_checks"].items():
        test_history = poison_record["test_observation_poison"]["complete_checkpoint_histories"]
        latent_history = poison_record["latent_truth_poison"]["complete_checkpoint_histories"]
        latent_outputs = poison_record["latent_truth_poison"]["train_validation_outputs"]
        if not test_history["all_complete_histories_exactly_equal"] or test_history["maximum_abs_discrepancy"] != 0.0:
            raise AssertionError(f"test-poison history mismatch: {condition}")
        if not latent_history["all_complete_histories_exactly_equal"] or latent_history["maximum_abs_discrepancy"] != 0.0:
            raise AssertionError(f"latent-poison history mismatch: {condition}")
        if max(latent_outputs.values()) != 0.0:
            raise AssertionError(f"latent-poison train/validation output mismatch: {condition}")
    checkpoint = diagnostics["checkpoint_recomputation"]
    if checkpoint["maximum_absolute_discrepancy"] > TOL["checkpoint"]:
        raise AssertionError("checkpoint recomputation failed")
    if not checkpoint["all_history_replay_epochs_match"] or not checkpoint["all_seed_selections_match"]:
        raise AssertionError("checkpoint-history or validation-seed replay failed")
    if diagnostics["adam_determinism"]["model_max_abs_discrepancy"] != 0.0 or not diagnostics["adam_determinism"]["history_exactly_equal"]:
        raise AssertionError("Adam determinism failed")
    if not diagnostics["exact_split_regeneration"]["all_exact"]:
        raise AssertionError("split regeneration failed")
    architecture = diagnostics["architecture_checks"]
    if architecture["bottleneck_dimension"] != 1 or not architecture["all_runtime_code_shapes_n_by_1"]:
        raise AssertionError("runtime bottleneck shape contract failed")
    if architecture["skip_paths_derived"] or architecture["unexpected_parameter_keys"] or architecture["direct_input_to_output_edges"]:
        raise AssertionError("derived architecture/no-skip contract failed")
    if not architecture["behavioral_no_skip_pass"] or not architecture["behavioral_no_skip_records"]:
        raise AssertionError("behavioral zero-decoder no-skip check failed")
    for record in architecture["behavioral_no_skip_records"]:
        if not record["passes"] or record["distinct_input_norm"] <= 0 or record["zero_decoder_output_difference"] != 0.0 or record["zero_decoder_output_max_abs"] != 0.0:
            raise AssertionError(f"behavioral no-skip record failed: {record}")
    expected_capacity = {
        "ordinary_width_8": {"nonlinear_total_and_trainable": 135, "frozen_encoder_fixed": 65, "frozen_decoder_trainable": 70, "frozen_total": 135},
        "few_sample_width_32": {"nonlinear_total_and_trainable": 519, "frozen_encoder_fixed": 257, "frozen_decoder_trainable": 262, "frozen_total": 519},
    }
    if architecture["capacity_registry"] != expected_capacity:
        raise AssertionError("capacity registry mismatch")
    frozen = diagnostics["frozen_encoder_immutability"]
    if not frozen["all_bitwise_equal"] or frozen["maximum_abs_discrepancy"] != 0.0:
        raise AssertionError("frozen encoder changed during decoder training")
    if diagnostics["initialization_registry"] != initialization_registry():
        raise AssertionError("initialization registry mismatch")
    for condition in CONDITIONS:
        for method in ("linear_ae", "nonlinear_ae", "frozen_random_encoder"):
            if not diagnostics["parameter_counts"][condition][method]["pass"]:
                raise AssertionError(f"parameter count failed: {condition}/{method}")
    failed_linear = [name for name, item in diagnostics["linear_autoencoder_vs_pca"].items() if not item["passes_declared_tolerance"]]
    if failed_linear:
        raise AssertionError(f"bounded linear AE optimization failed PCA equivalence tolerance: {failed_linear}")
    with tempfile.TemporaryDirectory(prefix="ldg-bnr-verify-") as temporary:
        temp_root = Path(temporary)
        build(temp_root, quiet=True)
        for name in required:
            if (root / name).read_bytes() != (temp_root / name).read_bytes():
                raise AssertionError(f"byte regeneration mismatch: {name}")
    if not quiet:
        print("verification passed: hashes, finite unique metrics, gradients, leakage, checkpoints, architecture, PCA equivalence, and byte regeneration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="generate bounded artifacts")
    parser.add_argument("--verify", action="store_true", help="verify existing artifacts and deterministic regeneration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT, help="artifact directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate = args.generate or not (args.generate or args.verify)
    verify_requested = args.verify or not (args.generate or args.verify)
    if generate:
        build(args.output_dir)
    if verify_requested:
        verify(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
