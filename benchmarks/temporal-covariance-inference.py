#!/usr/bin/env python3
"""Deterministic P4-U03 temporal covariance and inference benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
REPLICATES = 3
T, P, Q = 16, 6, 2
TIME = np.linspace(0.0, 1.0, T)
DT = float(TIME[1] - TIME[0])
NUGGET = 0.02
TRUE_TAU = (0.10, 0.32)
TAU_GRID = ((0.07, 0.22), (0.07, 0.32), (0.10, 0.22), (0.10, 0.32), (0.10, 0.45), (0.16, 0.32), (0.16, 0.45))
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PHASE3_PATH = REPO_ROOT / "toy-models" / "covariance-prior-trajectories.py"
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "temporal-covariance-inference"
ARTIFACT_NAMES = ("results.csv", "diagnostics.json", "summary.png")
STATUS_VALUES = ("ok", "nonconvergence", "invalid", "inapplicable")
TOL = {"poison": 1e-12, "oracle": 2e-8, "ordering": 0.0}
C = np.array([[1.0, .25], [.7, -.45], [.3, .9], [-.65, .4], [.8, .55], [-.25, -.75]])
D = np.array([.3, -.2, .15, .05, -.1, .25])
RDIAG = np.array([.10, .16, .22, .30, .38, .48])
R = np.diag(RDIAG)

SCENARIOS = (
    {"id": "reference_distinct", "class": "timescale_separation", "taus": TRUE_TAU, "sizes": (5, 3, 3)},
    {"id": "close_timescales", "class": "timescale_separation", "taus": (.18, .24), "sizes": (5, 3, 3)},
    {"id": "low_trial_count", "class": "trial_sample_size", "taus": TRUE_TAU, "sizes": (2, 1, 2)},
    {"id": "high_trial_count", "class": "trial_sample_size", "taus": TRUE_TAU, "sizes": (8, 4, 4)},
    {"id": "channel_mask", "class": "channel_time_masks", "taus": TRUE_TAU, "sizes": (5, 3, 3), "mask": "channel"},
    {"id": "time_block_mask", "class": "channel_time_masks", "taus": TRUE_TAU, "sizes": (5, 3, 3), "mask": "time"},
    {"id": "latent_changepoint", "class": "changepoints", "taus": TRUE_TAU, "sizes": (5, 3, 3), "generator": "changepoint"},
    {"id": "correlated_residuals", "class": "correlated_residuals", "taus": TRUE_TAU, "sizes": (5, 3, 3), "residual": "correlated"},
    {"id": "low_rate_poisson", "class": "poisson_misspecification", "taus": TRUE_TAU, "sizes": (5, 3, 3), "observation": "poisson"},
    {"id": "equal_kernels", "class": "equal_kernels", "taus": (.20, .20), "sizes": (5, 3, 3)},
    {"id": "alignment_jitter", "class": "alignment_jitter", "taus": TRUE_TAU, "sizes": (5, 3, 3), "generator": "jitter"},
    {"id": "informative_missingness", "class": "informative_missingness", "taus": TRUE_TAU, "sizes": (5, 3, 3), "mask": "informative"},
)

METHODS = (
    {"id": "mean_diagonal", "family": "mean_diagonal", "information_set": "static_observation", "parameter_source": "train"},
    {"id": "static_factor", "family": "static_factor", "information_set": "static_observation", "parameter_source": "train"},
    {"id": "two_stage_pca_smooth", "family": "two_stage", "information_set": "retrospective", "parameter_source": "train_validation"},
    {"id": "independent_channel_gp", "family": "independent_channel", "information_set": "retrospective", "parameter_source": "train_validation"},
    {"id": "known_gpfa", "family": "gpfa", "information_set": "retrospective", "parameter_source": "known_nominal"},
    {"id": "selected_gpfa", "family": "gpfa", "information_set": "retrospective", "parameter_source": "train_validation"},
    {"id": "wrong_timescale_gpfa", "family": "gpfa_wrong_timescale", "information_set": "retrospective", "parameter_source": "fixed_wrong"},
    {"id": "wrong_ar_gpfa", "family": "gpfa_wrong_family", "information_set": "retrospective", "parameter_source": "fixed_wrong"},
    {"id": "ar_kalman_filter", "family": "ar_lgssm", "information_set": "online", "parameter_source": "known_nominal"},
    {"id": "ar_rts_smoother", "family": "ar_lgssm", "information_set": "retrospective", "parameter_source": "known_nominal"},
    {"id": "exact_gaussian_oracle", "family": "exact_oracle", "information_set": "oracle_retrospective", "parameter_source": "true_scenario"},
)

METRICS = (
    "marginal_nll_per_entry",
    "masked_prediction_rmse",
    "masked_prediction_nll_per_entry",
    "latent_rmse",
    "latent_90_coverage",
    "loading_subspace_angle_degrees",
    "timescale_l1",
    "selected_combined_nll_per_entry",
    "covariance_relative_frobenius",
    "posterior_mean_oracle_max_abs",
)

FIELDS = (
    "row_key", "scenario", "scenario_class", "replicate", "seed_id", "provenance",
    "method", "method_family", "information_set", "parameter_source", "target", "metric",
    "n_train", "n_validation", "n_test", "status", "reason", "value",
)


def scenario_code(scenario_id: str) -> int:
    return int.from_bytes(hashlib.sha256(scenario_id.encode()).digest()[:4], "little")


def rng_for(scenario_id: str, replicate: int, stage: int, split: str, index: int = 0) -> np.random.Generator:
    split_code = {"train": 1, "validation": 2, "test": 3}[split]
    entropy = (MASTER_SEED, scenario_code(scenario_id), replicate, stage, split_code, index)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def seed_id(scenario_id: str, replicate: int) -> str:
    return f"{MASTER_SEED}:{scenario_code(scenario_id)}:{replicate}"


def sym(matrix: np.ndarray) -> np.ndarray:
    return .5 * (matrix + matrix.T)


def chol_psd(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(sym(matrix) + 1e-10 * np.eye(len(matrix)))


def se_kernel(tau: float) -> np.ndarray:
    delta = TIME[:, None] - TIME[None, :]
    return (1.0 - NUGGET) * np.exp(-.5 * (delta / tau) ** 2) + NUGGET * np.eye(T)


def ar_kernel(tau: float) -> np.ndarray:
    rho = (1.0 - NUGGET) * math.exp(-.5 * (DT / tau) ** 2)
    return rho ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))


def latent_covariance(taus, family: str = "se", changepoint: bool = False) -> np.ndarray:
    result = np.zeros((T * Q, T * Q))
    for j, tau in enumerate(taus):
        kernel = se_kernel(tau) if family == "se" else ar_kernel(tau)
        if changepoint:
            half = T // 2
            kernel[:half, half:] = 0.0
            kernel[half:, :half] = 0.0
            kernel = sym(kernel)
        idx = np.arange(T) * Q + j
        result[np.ix_(idx, idx)] = kernel
    return result


def observation_matrix(loading: np.ndarray = C) -> np.ndarray:
    return np.kron(np.eye(T), loading)


def observation_mean(mean: np.ndarray = D) -> np.ndarray:
    return np.tile(mean, T)


def correlated_residual() -> np.ndarray:
    corr = .12 * np.sqrt(np.outer(RDIAG, RDIAG))
    np.fill_diagonal(corr, RDIAG)
    return corr


def gaussian_nll(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
        raise ValueError("nonfinite_gaussian_input")
    l = chol_psd(cov)
    residual = x - mean
    solved = np.linalg.solve(l, residual)
    return float(.5 * (len(x) * math.log(2 * math.pi) + 2 * np.log(np.diag(l)).sum() + solved @ solved))


def covariance_model(taus, family="se", loading=C, mean=D, residual=R, changepoint=False) -> dict:
    kz = latent_covariance(taus, family, changepoint)
    a = observation_matrix(loading)
    cov = sym(a @ kz @ a.T + np.kron(np.eye(T), residual))
    return {"taus": tuple(float(x) for x in taus), "family": family, "loading": loading.copy(),
            "mean": mean.copy(), "residual": residual.copy(), "latent_cov": kz, "covariance": cov}


def generate_trial(scenario: dict, replicate: int, split: str, index: int) -> dict:
    taus = scenario["taus"]
    changepoint = scenario.get("generator") == "changepoint"
    kz = latent_covariance(taus, "se", changepoint)
    latent = (chol_psd(kz) @ rng_for(scenario["id"], replicate, 10, split, index).normal(size=T * Q)).reshape(T, Q)
    if scenario.get("generator") == "jitter":
        shift = int(rng_for(scenario["id"], replicate, 11, split, index).choice((-1, 0, 1)))
        latent = np.roll(latent, shift, axis=0)
        if shift > 0:
            latent[:shift] = latent[shift]
        elif shift < 0:
            latent[shift:] = latent[shift - 1]
    residual = correlated_residual() if scenario.get("residual") == "correlated" else R
    signal = latent @ C.T + D
    noise = rng_for(scenario["id"], replicate, 12, split, index).multivariate_normal(np.zeros(P), residual, size=T)
    if scenario.get("observation") == "poisson":
        rate = np.exp(-2.0 + .35 * signal)
        observed = rng_for(scenario["id"], replicate, 13, split, index).poisson(rate).astype(float)
    else:
        observed = signal + noise
    return {"latent": latent, "signal": signal, "observed": observed}


def generate_replicate(scenario: dict, replicate: int) -> dict:
    n_train, n_validation, n_test = scenario["sizes"]
    return {
        split: [generate_trial(scenario, replicate, split, index) for index in range(count)]
        for split, count in (("train", n_train), ("validation", n_validation), ("test", n_test))
    }


def make_masks(scenario: dict, replicate: int, observations: list[np.ndarray]) -> list[np.ndarray]:
    masks = []
    mode = scenario.get("mask", "mixed")
    for index, y in enumerate(observations):
        mask = np.ones((T, P), dtype=bool)
        use = mode
        if use == "mixed":
            use = "channel" if index % 2 == 0 else "time"
        if use == "channel":
            mask[:, index % P] = False
        elif use == "time":
            mask[T // 3: 2 * T // 3, :] = False
        elif use == "informative":
            threshold = float(np.quantile(y, .72))
            mask[y > threshold] = False
            if np.all(mask):
                mask[T // 2, 0] = False
        masks.append(mask)
    return masks


def pca_factor(train_observations: np.ndarray) -> dict:
    mean = train_observations.mean(axis=0)
    centered = train_observations - mean
    covariance = sym(centered.T @ centered / max(len(centered), 1)) + 1e-6 * np.eye(P)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    basis = vectors[:, order[:Q]]
    residual_level = max(float(np.mean(values[order[Q:]])), 1e-5)
    loading = basis @ np.diag(np.sqrt(np.maximum(values[order[:Q]] - residual_level, 1e-6)))
    residual = np.diag(np.maximum(np.diag(covariance - loading @ loading.T), 1e-5))
    return {"mean": mean, "basis": basis, "loading": loading, "residual": residual,
            "static_covariance": sym(loading @ loading.T + residual)}


def select_taus(train_trials: tuple[np.ndarray, ...], validation_trials: tuple[np.ndarray, ...]) -> dict:
    records = []
    for taus in TAU_GRID:
        model = covariance_model(taus)
        train_nll = sum(gaussian_nll(x.reshape(-1), observation_mean(), model["covariance"]) for x in train_trials)
        validation_nll = sum(gaussian_nll(x.reshape(-1), observation_mean(), model["covariance"]) for x in validation_trials)
        total = train_nll + validation_nll
        records.append({"taus": tuple(taus), "train_nll": train_nll, "validation_nll": validation_nll, "score": total})
    selected = min(records, key=lambda x: (x["score"], x["taus"]))
    denominator = max((len(train_trials) + len(validation_trials)) * T * P, 1)
    return {"taus": selected["taus"], "combined_nll_per_entry": selected["score"] / denominator,
            "train_nll": selected["train_nll"], "validation_nll": selected["validation_nll"],
            "combined_nll": selected["score"], "records": records}


def fit_lawful_states(train_observations: tuple[np.ndarray, ...], validation_observations: tuple[np.ndarray, ...]) -> dict:
    """Capability boundary: receives observations from train/validation only."""
    stacked = np.vstack(train_observations)
    static = pca_factor(stacked)
    selection = select_taus(train_observations, validation_observations)
    channel_mean = stacked.mean(axis=0)
    channel_var = np.maximum(stacked.var(axis=0), 1e-5)
    return {"static": static, "selection": selection, "channel_mean": channel_mean, "channel_var": channel_var}


def assemble_frozen_states(scenario: dict, lawful: dict) -> dict:
    """Predeclared/known states assembled without test observations or latent truth arrays."""
    true_residual = correlated_residual() if scenario.get("residual") == "correlated" else R
    oracle_applicable = scenario.get("observation", "gaussian") == "gaussian" and scenario.get("generator") not in ("jitter",) and scenario.get("mask") != "informative"
    oracle = covariance_model(scenario["taus"], residual=true_residual, changepoint=scenario.get("generator") == "changepoint") if oracle_applicable else None
    static = lawful["static"]
    return {
        "mean_diagonal": {"mean": lawful["channel_mean"], "variance": lawful["channel_var"]},
        "static_factor": static,
        "selected_taus": tuple(lawful["selection"]["taus"]),
        "selected_combined_nll_per_entry": float(lawful["selection"]["combined_nll_per_entry"]),
        "known_gpfa": covariance_model(scenario["taus"]),
        "selected_gpfa": covariance_model(lawful["selection"]["taus"]),
        "wrong_timescale_gpfa": covariance_model((.06, .06)),
        "wrong_ar_gpfa": covariance_model(scenario["taus"], family="ar"),
        "ar_model": covariance_model(scenario["taus"], family="ar"),
        "oracle": oracle,
    }


def condition_gaussian(mean: np.ndarray, cov: np.ndarray, values: np.ndarray, mask: np.ndarray) -> dict:
    flat_mask = mask.reshape(-1)
    obs = np.flatnonzero(flat_mask)
    target = np.flatnonzero(~flat_mask)
    if not len(target):
        return {"indices": target, "mean": np.empty(0), "cov": np.empty((0, 0))}
    co = cov[np.ix_(obs, obs)]
    cross = cov[np.ix_(target, obs)]
    ct = cov[np.ix_(target, target)]
    l = chol_psd(co)
    alpha = np.linalg.solve(l.T, np.linalg.solve(l, values.reshape(-1)[obs] - mean[obs]))
    pred = mean[target] + cross @ alpha
    solved = np.linalg.solve(l.T, np.linalg.solve(l, cross.T))
    pcov = sym(ct - cross @ solved)
    return {"indices": target, "mean": pred, "cov": pcov}


def latent_posterior(model: dict, values: np.ndarray, mask: np.ndarray) -> dict:
    a = observation_matrix(model["loading"])
    mean_y = observation_mean(model["mean"])
    flat_mask = mask.reshape(-1)
    obs = np.flatnonzero(flat_mask)
    ao = a[obs]
    residual_full = np.kron(np.eye(T), model["residual"])
    kz = model["latent_cov"]
    s = ao @ kz @ ao.T + residual_full[np.ix_(obs, obs)]
    l = chol_psd(s)
    alpha = np.linalg.solve(l.T, np.linalg.solve(l, values.reshape(-1)[obs] - mean_y[obs]))
    posterior_mean = kz @ ao.T @ alpha
    solved = np.linalg.solve(l.T, np.linalg.solve(l, ao @ kz))
    posterior_cov = sym(kz - kz @ ao.T @ solved)
    return {"mean": posterior_mean.reshape(T, Q), "cov": posterior_cov,
            "var": np.diag(posterior_cov).reshape(T, Q)}


def precision_latent_posterior(model: dict, values: np.ndarray, mask: np.ndarray) -> dict:
    a = observation_matrix(model["loading"])
    flat_mask = mask.reshape(-1)
    obs = np.flatnonzero(flat_mask)
    ao = a[obs]
    residual_full = np.kron(np.eye(T), model["residual"])
    ro = residual_full[np.ix_(obs, obs)]
    kz = model["latent_cov"]
    ro_inv_ao = np.linalg.solve(ro, ao)
    precision = np.linalg.inv(kz) + ao.T @ ro_inv_ao
    cov = np.linalg.inv(precision)
    rhs = ao.T @ np.linalg.solve(ro, values.reshape(-1)[obs] - observation_mean(model["mean"])[obs])
    mean = cov @ rhs
    return {"mean": mean.reshape(T, Q), "cov": sym(cov), "var": np.diag(cov).reshape(T, Q)}


def kalman_filter_smoother(values: np.ndarray, mask: np.ndarray, taus) -> dict:
    rho = np.array([(1 - NUGGET) * math.exp(-.5 * (DT / tau) ** 2) for tau in taus])
    transition = np.diag(rho)
    process = np.diag(1 - rho ** 2)
    pred_m, pred_p, filt_m, filt_p = [], [], [], []
    mean, cov = np.zeros(Q), np.eye(Q)
    hidden_predictions = np.full((T, P), np.nan)
    hidden_variances = np.full((T, P), np.nan)
    innovations = []
    for t in range(T):
        if t:
            mean = transition @ mean
            cov = transition @ cov @ transition.T + process
        pred_m.append(mean.copy()); pred_p.append(cov.copy())
        observed = np.flatnonzero(mask[t])
        hidden = np.flatnonzero(~mask[t])
        if len(observed):
            h = C[observed]
            rr = R[np.ix_(observed, observed)]
            innovation = values[t, observed] - D[observed] - h @ mean
            s = h @ cov @ h.T + rr
            innovations.append(gaussian_nll(values[t, observed], D[observed] + h @ mean, s))
            gain = np.linalg.solve(s, (cov @ h.T).T).T
            mean = mean + gain @ innovation
            cov = sym(cov - gain @ h @ cov)
        if len(hidden):
            hidden_predictions[t, hidden] = D[hidden] + C[hidden] @ mean
            hidden_variances[t, hidden] = np.diag(C[hidden] @ cov @ C[hidden].T + R[np.ix_(hidden, hidden)])
        filt_m.append(mean.copy()); filt_p.append(cov.copy())
    pred_m, pred_p, filt_m, filt_p = map(np.asarray, (pred_m, pred_p, filt_m, filt_p))
    smooth_m, smooth_p = filt_m.copy(), filt_p.copy()
    for t in range(T - 2, -1, -1):
        gain = np.linalg.solve(pred_p[t + 1], (filt_p[t] @ transition.T).T).T
        smooth_m[t] = filt_m[t] + gain @ (smooth_m[t + 1] - pred_m[t + 1])
        smooth_p[t] = sym(filt_p[t] + gain @ (smooth_p[t + 1] - pred_p[t + 1]) @ gain.T)
    return {"filter_mean": filt_m, "filter_var": np.stack([np.diag(x) for x in filt_p]),
            "smooth_mean": smooth_m, "smooth_var": np.stack([np.diag(x) for x in smooth_p]),
            "hidden_mean": hidden_predictions, "hidden_var": hidden_variances,
            "sequential_nll": float(sum(innovations))}


def training_scoring_alignments(states: dict, train_trials: list[dict]) -> dict:
    """Scoring-only rotations estimated once from training truth and then frozen."""
    observations = tuple(trial["observed"] for trial in train_trials)
    truths = np.vstack([trial["latent"] for trial in train_trials])
    full_mask = np.ones((T, P), dtype=bool)
    static_estimates = np.vstack([static_factor_inference(values, full_mask, states["static_factor"])["mean"]
                                  for values in observations])
    two_stage_estimates = np.vstack([two_stage_inference(values, np.ones((T, P), dtype=bool),
                                                         states["static_factor"], states["selected_taus"])["latent_mean"]
                                     for values in observations])
    rotations = {}
    for name, estimates in (("static_factor", static_estimates), ("two_stage_pca_smooth", two_stage_estimates)):
        u, _, vt = np.linalg.svd(estimates.T @ truths, full_matrices=False)
        rotations[name] = u @ vt
    return rotations


def subspace_angle(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.linalg.qr(a)[0][:, :Q]
    qb = np.linalg.qr(b)[0][:, :Q]
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(singular.min(), -1, 1))))


def independent_channel_inference(values: np.ndarray, mask: np.ndarray, mean: np.ndarray, variance: np.ndarray, tau: float) -> dict:
    pred = np.full((T, P), np.nan)
    pred_var = np.full((T, P), np.nan)
    kernel = se_kernel(tau)
    for channel in range(P):
        observed = np.flatnonzero(mask[:, channel])
        hidden = np.flatnonzero(~mask[:, channel])
        if not len(hidden):
            continue
        co = variance[channel] * kernel[np.ix_(observed, observed)] + 1e-5 * np.eye(len(observed))
        cross = variance[channel] * kernel[np.ix_(hidden, observed)]
        ct = variance[channel] * kernel[np.ix_(hidden, hidden)]
        l = chol_psd(co)
        alpha = np.linalg.solve(l.T, np.linalg.solve(l, values[observed, channel] - mean[channel]))
        pred[hidden, channel] = mean[channel] + cross @ alpha
        pcov = sym(ct - cross @ np.linalg.solve(l.T, np.linalg.solve(l, cross.T)))
        pred_var[hidden, channel] = np.diag(pcov) + 1e-5
    return {"hidden_mean": pred, "hidden_var": pred_var}


def two_stage_inference(values: np.ndarray, mask: np.ndarray, static: dict, taus) -> dict:
    filled = np.where(mask, values, static["mean"][None, :])
    scores = (filled - static["mean"]) @ static["basis"]
    smoothed = np.zeros_like(scores)
    for component, tau in enumerate(taus):
        kernel = se_kernel(tau)
        cov = kernel + .15 * np.eye(T)
        smoothed[:, component] = kernel @ np.linalg.solve(cov, scores[:, component])
    reconstruction = smoothed @ static["basis"].T + static["mean"]
    hidden_var = np.broadcast_to(np.diag(static["residual"]), reconstruction.shape).copy()
    return {"latent_mean": smoothed, "hidden_mean": reconstruction, "hidden_var": hidden_var}


def static_factor_inference(values: np.ndarray, mask: np.ndarray, static: dict) -> dict:
    """Independent-bin factor posterior using visible channels only at each time."""
    loading, residual, mean = static["loading"], static["residual"], static["mean"]
    posterior_mean = np.zeros((T, Q))
    posterior_var = np.zeros((T, Q))
    for t in range(T):
        visible = np.flatnonzero(mask[t])
        if not len(visible):
            posterior_var[t] = 1.0
            continue
        lo = loading[visible]
        ro = residual[np.ix_(visible, visible)]
        precision = np.eye(Q) + lo.T @ np.linalg.solve(ro, lo)
        covariance = np.linalg.inv(precision)
        posterior_mean[t] = covariance @ lo.T @ np.linalg.solve(ro, values[t, visible] - mean[visible])
        posterior_var[t] = np.diag(covariance)
    return {"mean": posterior_mean, "var": posterior_var}


def infer_frozen_states(states: dict, test_observations: tuple[np.ndarray, ...], masks: tuple[np.ndarray, ...]) -> dict:
    """Capability boundary: receives frozen states and test observations/masks, never truth or train/validation."""
    result = {method["id"]: [] for method in METHODS}
    for values, mask in zip(test_observations, masks):
        mean_model = states["mean_diagonal"]
        hidden_mean = np.broadcast_to(mean_model["mean"], values.shape).copy()
        hidden_var = np.broadcast_to(mean_model["variance"], values.shape).copy()
        result["mean_diagonal"].append({"hidden_mean": hidden_mean, "hidden_var": hidden_var})

        static = states["static_factor"]
        static_cov = static["static_covariance"]
        static_model = {"mean": static["mean"], "covariance": np.kron(np.eye(T), static_cov)}
        cond = condition_gaussian(observation_mean(static["mean"]), static_model["covariance"], values, mask)
        static_posterior = static_factor_inference(values, mask, static)
        result["static_factor"].append({"latent_mean": static_posterior["mean"],
                                        "latent_var": static_posterior["var"], "condition": cond})
        result["two_stage_pca_smooth"].append(two_stage_inference(values, mask, static, states["selected_taus"]))
        result["independent_channel_gp"].append(independent_channel_inference(values, mask, states["mean_diagonal"]["mean"], states["mean_diagonal"]["variance"], float(np.mean(states["selected_taus"]))))

        for name in ("known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa"):
            model = states[name]
            result[name].append({"posterior": latent_posterior(model, values, mask),
                                 "condition": condition_gaussian(observation_mean(model["mean"]), model["covariance"], values, mask)})
        recursive = kalman_filter_smoother(values, mask, states["ar_model"]["taus"])
        result["ar_kalman_filter"].append(recursive)
        ar_condition = condition_gaussian(observation_mean(states["ar_model"]["mean"]),
                                          states["ar_model"]["covariance"], values, mask)
        result["ar_rts_smoother"].append({**recursive, "condition": ar_condition})
        if states["oracle"] is None:
            result["exact_gaussian_oracle"].append(None)
        else:
            model = states["oracle"]
            result["exact_gaussian_oracle"].append({"posterior": precision_latent_posterior(model, values, mask),
                                                     "condition": condition_gaussian(observation_mean(model["mean"]), model["covariance"], values, mask)})
    return result


def target_for_metric(method: str, metric: str) -> str:
    if metric in ("marginal_nll_per_entry", "covariance_relative_frobenius"):
        return "observation_distribution"
    if metric.startswith("masked_prediction"):
        return "masked_observation_prediction"
    if metric in ("latent_rmse", "latent_90_coverage"):
        return "latent_trajectory"
    if metric == "loading_subspace_angle_degrees":
        return "loading_subspace"
    if metric in ("timescale_l1", "selected_combined_nll_per_entry"):
        return "timescale_selection"
    if metric == "posterior_mean_oracle_max_abs":
        return "numerical_posterior_oracle"
    raise KeyError(metric)


def provenance_for_metric(metric: str) -> str:
    if metric == "selected_combined_nll_per_entry":
        return "fit_diagnostic"
    if metric in ("timescale_l1", "loading_subspace_angle_degrees", "covariance_relative_frobenius"):
        return "fit_diagnostic"
    if metric == "posterior_mean_oracle_max_abs":
        return "oracle_diagnostic"
    return "test"


def inference_required(metric: str) -> bool:
    return metric in ("marginal_nll_per_entry", "masked_prediction_rmse", "masked_prediction_nll_per_entry",
                      "latent_rmse", "latent_90_coverage", "posterior_mean_oracle_max_abs")


def oracle_applicable(scenario: dict) -> bool:
    return scenario.get("observation", "gaussian") == "gaussian" and scenario.get("generator") != "jitter" and scenario.get("mask") != "informative"


def covariance_target_applicable(scenario: dict) -> bool:
    return scenario.get("observation", "gaussian") == "gaussian" and scenario.get("generator") != "jitter"


def applicability(method: str, metric: str, scenario: dict) -> tuple[bool, str]:
    distribution_methods = {"mean_diagonal", "static_factor", "known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "ar_kalman_filter", "ar_rts_smoother", "exact_gaussian_oracle"}
    masked_methods = {m["id"] for m in METHODS}
    latent_methods = {"static_factor", "two_stage_pca_smooth", "known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "ar_kalman_filter", "ar_rts_smoother", "exact_gaussian_oracle"}
    coverage_methods = {"known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "ar_kalman_filter", "ar_rts_smoother", "exact_gaussian_oracle"}
    if method == "exact_gaussian_oracle" and not oracle_applicable(scenario):
        return False, "exact_gaussian_oracle_not_applicable_to_nongaussian_jitter_or_informative_mask_scenario"
    if metric == "marginal_nll_per_entry" and method not in distribution_methods:
        return False, "joint_observation_likelihood_not_defined_for_method"
    if metric.startswith("masked_prediction") and method not in masked_methods:
        return False, "masked_prediction_not_defined_for_method"
    if metric == "latent_rmse" and method not in latent_methods:
        return False, "latent_trajectory_not_defined_for_method"
    if metric == "latent_90_coverage" and method not in coverage_methods:
        return False, "calibrated_latent_posterior_not_defined_for_method"
    if metric == "loading_subspace_angle_degrees" and method not in {"static_factor", "two_stage_pca_smooth"}:
        return False, "loading_subspace_is_not_a_learned_target_for_method"
    if metric in ("timescale_l1", "selected_combined_nll_per_entry") and method != "selected_gpfa":
        return False, "timescale_selection_metric_only_applies_to_selected_gpfa"
    if metric == "covariance_relative_frobenius" and method not in {"mean_diagonal", "static_factor", "known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "ar_rts_smoother", "exact_gaussian_oracle"}:
        return False, "full_observation_covariance_not_defined_for_method"
    if metric == "covariance_relative_frobenius" and not covariance_target_applicable(scenario):
        return False, "finite_gaussian_observation_covariance_target_not_declared_for_scenario"
    if metric == "posterior_mean_oracle_max_abs":
        if method != "known_gpfa":
            return False, "posterior_oracle_check_only_applies_to_known_gpfa"
        if not oracle_applicable(scenario) or scenario.get("residual") == "correlated" or scenario.get("generator") == "changepoint":
            return False, "nominal_known_gpfa_is_not_the_exact_scenario_oracle"
    return True, ""


def row_base(scenario: dict, replicate: int, method_record: dict, metric: str) -> dict:
    n_train, n_validation, n_test = scenario["sizes"]
    return {
        "scenario": scenario["id"], "scenario_class": scenario["class"], "replicate": replicate,
        "seed_id": seed_id(scenario["id"], replicate), "provenance": provenance_for_metric(metric),
        "method": method_record["id"], "method_family": method_record["family"],
        "information_set": method_record["information_set"], "parameter_source": method_record["parameter_source"],
        "target": target_for_metric(method_record["id"], metric), "metric": metric,
        "n_train": n_train, "n_validation": n_validation, "n_test": n_test,
    }


def add_row(rows: list[dict], base: dict, value=None, status="ok", reason="") -> None:
    row = dict(base)
    if status == "ok":
        if value is None or not np.isfinite(value):
            status, reason, value = "invalid", "nonfinite_or_missing_scored_value", None
    elif not reason:
        raise ValueError("non_ok_row_requires_reason")
    row.update({"status": status, "reason": reason, "value": None if value is None else float(value)})
    rows.append(row)


def status_for_exception(exc: Exception) -> str:
    return "nonconvergence" if isinstance(exc, np.linalg.LinAlgError) else "invalid"


def model_covariance_for_method(method: str, states: dict) -> np.ndarray | None:
    if method == "mean_diagonal":
        return np.kron(np.eye(T), np.diag(states["mean_diagonal"]["variance"]))
    if method == "static_factor":
        return np.kron(np.eye(T), states["static_factor"]["static_covariance"])
    if method in ("known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa"):
        return states[method]["covariance"]
    if method == "ar_rts_smoother":
        return states["ar_model"]["covariance"]
    if method == "exact_gaussian_oracle" and states["oracle"] is not None:
        return states["oracle"]["covariance"]
    return None


def score_metric(scenario: dict, method: str, metric: str, data: dict, masks: tuple[np.ndarray, ...], states: dict,
                 inference: dict, scoring_alignments: dict) -> float:
    test = data["test"]
    if metric == "selected_combined_nll_per_entry":
        return states["selected_combined_nll_per_entry"]
    if metric == "timescale_l1":
        return float(sum(abs(a - b) for a, b in zip(states["selected_taus"], scenario["taus"])))
    if metric == "loading_subspace_angle_degrees":
        return subspace_angle(states["static_factor"]["basis"], C)
    if metric == "covariance_relative_frobenius":
        candidate = model_covariance_for_method(method, states)
        target = states["oracle"]["covariance"] if states["oracle"] is not None else covariance_model(scenario["taus"])["covariance"]
        return float(np.linalg.norm(candidate - target) / max(np.linalg.norm(target), 1e-12))
    outputs = inference[method]
    if metric == "posterior_mean_oracle_max_abs":
        errors = []
        for trial, mask, output in zip(test, masks, outputs):
            independent = precision_latent_posterior(states["known_gpfa"], trial["observed"], mask)
            errors.append(np.max(np.abs(output["posterior"]["mean"] - independent["mean"])))
        return float(max(errors))
    if metric == "marginal_nll_per_entry":
        if method == "ar_kalman_filter":
            complete = [kalman_filter_smoother(trial["observed"], np.ones((T, P), dtype=bool), states["ar_model"]["taus"])
                        for trial in test]
            return float(sum(x["sequential_nll"] for x in complete) / (len(test) * T * P))
        covariance = model_covariance_for_method(method, states)
        if method == "exact_gaussian_oracle":
            mean = observation_mean(states["oracle"]["mean"])
        elif method == "static_factor":
            mean = observation_mean(states["static_factor"]["mean"])
        elif method == "mean_diagonal":
            mean = observation_mean(states["mean_diagonal"]["mean"])
        else:
            mean = observation_mean()
        return float(np.mean([gaussian_nll(trial["observed"].reshape(-1), mean, covariance) / (T * P) for trial in test]))
    if metric.startswith("masked_prediction"):
        squared, nll, count = 0.0, 0.0, 0
        for trial, mask, output in zip(test, masks, outputs):
            hidden = ~mask
            truth = trial["observed"][hidden]
            if method in ("known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "static_factor", "exact_gaussian_oracle"):
                cond = output["condition"]
                pred, cov = cond["mean"], cond["cov"]
                variance = np.diag(cov)
            elif method in ("ar_kalman_filter", "ar_rts_smoother"):
                if method == "ar_rts_smoother":
                    cond = output["condition"]
                    pred, cov = cond["mean"], cond["cov"]
                    variance = np.diag(cov)
                else:
                    pred = output["hidden_mean"][hidden]
                    variance = output["hidden_var"][hidden]
                    cov = np.diag(variance)
            else:
                pred = output["hidden_mean"][hidden]
                variance = output["hidden_var"][hidden]
                cov = np.diag(variance)
            if len(truth):
                squared += float(np.sum((pred - truth) ** 2))
                nll += gaussian_nll(truth, pred, cov)
                count += len(truth)
        if metric == "masked_prediction_rmse":
            return math.sqrt(squared / count)
        return nll / count
    if metric in ("latent_rmse", "latent_90_coverage"):
        errors, covered, total = [], 0, 0
        for trial, output in zip(test, outputs):
            if method == "static_factor":
                estimate = output["latent_mean"]
                variance = None
            elif method == "two_stage_pca_smooth":
                estimate = output["latent_mean"]
                variance = None
            elif method in ("ar_kalman_filter", "ar_rts_smoother"):
                estimate = output["filter_mean"] if method == "ar_kalman_filter" else output["smooth_mean"]
                variance = output["filter_var"] if method == "ar_kalman_filter" else output["smooth_var"]
            else:
                estimate = output["posterior"]["mean"]
                variance = output["posterior"]["var"]
            aligned = estimate @ scoring_alignments[method] if method in scoring_alignments else estimate
            err = aligned - trial["latent"]
            errors.append(np.sum(err ** 2)); total += err.size
            if variance is not None:
                covered += int(np.sum(np.abs(err) <= 1.644853626951 * np.sqrt(np.maximum(variance, 0))))
        if metric == "latent_rmse":
            return math.sqrt(sum(errors) / total)
        return covered / total
    raise KeyError(metric)


def evaluate_replicate(scenario: dict, replicate: int, fault: str | None = None) -> tuple[list[dict], dict]:
    data = generate_replicate(scenario, replicate)
    train_obs = tuple(trial["observed"] for trial in data["train"])
    validation_obs = tuple(trial["observed"] for trial in data["validation"])
    test_obs = tuple(trial["observed"] for trial in data["test"])
    lawful = fit_lawful_states(train_obs, validation_obs)
    states = assemble_frozen_states(scenario, lawful)
    scoring_alignments = training_scoring_alignments(states, data["train"])
    masks = tuple(make_masks(scenario, replicate, list(test_obs)))
    inference_error = None
    try:
        if fault == "inference_linalg":
            raise np.linalg.LinAlgError("fault_injected_inference_linalg")
        if fault == "inference_invalid":
            raise ValueError("fault_injected_inference_invalid")
        inference = infer_frozen_states(states, test_obs, masks)
    except Exception as exc:
        inference, inference_error = {}, exc
    rows = []
    for method_record in METHODS:
        method = method_record["id"]
        for metric in METRICS:
            base = row_base(scenario, replicate, method_record, metric)
            applicable, reason = applicability(method, metric, scenario)
            if not applicable:
                add_row(rows, base, status="inapplicable", reason=reason)
                continue
            if inference_error is not None and inference_required(metric):
                add_row(rows, base, status=status_for_exception(inference_error), reason=f"inference_exception:{type(inference_error).__name__}")
                continue
            try:
                value = score_metric(scenario, method, metric, data, masks, states, inference, scoring_alignments)
                if fault == "scoring_nonfinite" and method == "known_gpfa" and metric == "latent_rmse":
                    value = np.nan
                add_row(rows, base, value)
            except Exception as exc:
                add_row(rows, base, status=status_for_exception(exc), reason=f"scoring_exception:{type(exc).__name__}:{str(exc)[:80]}")
    selection = {"scenario": scenario["id"], "replicate": replicate, "selected_taus": list(states["selected_taus"]),
                 "true_taus": list(scenario["taus"]),
                 "criterion": "combined_train_plus_validation_total_nll",
                 "train_total_nll": lawful["selection"]["train_nll"],
                 "validation_total_nll": lawful["selection"]["validation_nll"],
                 "combined_total_nll": lawful["selection"]["combined_nll"],
                 "combined_nll_per_entry": states["selected_combined_nll_per_entry"]}
    return rows, selection


def canonical_key(row: dict) -> tuple:
    return (row["scenario"], int(row["replicate"]), row["provenance"], row["method"], row["metric"])


def format_number(value) -> str:
    if value is None or value == "":
        return ""
    return format(float(value), ".17g")


def serialized_row(row: dict) -> dict:
    result = {field: row.get(field, "") for field in FIELDS}
    result["replicate"] = str(result["replicate"])
    for field in ("n_train", "n_validation", "n_test"):
        result[field] = str(result[field])
    result["value"] = format_number(result["value"])
    result["row_key"] = "|".join((result["scenario"], result["replicate"], result["provenance"], result["method"], result["metric"]))
    return result


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=canonical_key):
            writer.writerow(serialized_row(row))


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("results_csv_schema_mismatch")
        return list(reader)


def validate_result_row(row: dict) -> None:
    if set(row) != set(FIELDS):
        raise ValueError("row_fields_mismatch")
    expected_key = "|".join((row["scenario"], row["replicate"], row["provenance"], row["method"], row["metric"]))
    if row["row_key"] != expected_key:
        raise ValueError("row_key_mismatch")
    if row["status"] not in STATUS_VALUES:
        raise ValueError("unknown_status")
    if row["status"] == "ok":
        if row["reason"] or row["value"] == "" or not np.isfinite(float(row["value"])):
            raise ValueError("invalid_ok_semantics")
    else:
        if not row["reason"] or row["value"] != "":
            raise ValueError("invalid_non_ok_semantics")


def validate_resume_rows(prior_rows: list[dict], fresh_rows: list[dict]) -> dict[str, dict]:
    fresh = {serialized_row(row)["row_key"]: serialized_row(row) for row in fresh_rows}
    result = {}
    for row in prior_rows:
        validate_result_row(row)
        key = row["row_key"]
        if key in result:
            raise ValueError("duplicate_resume_key")
        if key not in fresh:
            raise ValueError("unknown_resume_key")
        if row != fresh[key]:
            raise ValueError("stale_or_corrupted_resume_row")
        result[key] = row
    return result


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["method"], row["metric"]), []).append(row)
    summaries = []
    for (scenario, method, metric), group in sorted(groups.items()):
        values = np.array([float(row["value"]) for row in group if row["status"] == "ok"], dtype=float)
        n = len(group)
        summaries.append({
            "scenario": scenario, "method": method, "metric": metric, "n": n, "n_ok": int(len(values)),
            "q10": None if not len(values) else float(np.quantile(values, .10)),
            "median": None if not len(values) else float(np.quantile(values, .50)),
            "q90": None if not len(values) else float(np.quantile(values, .90)),
            "ok_rate": len(values) / n,
            "failure_rate": sum(row["status"] in ("invalid", "nonconvergence") for row in group) / n,
            "applicability_rate": sum(row["status"] != "inapplicable" for row in group) / n,
        })
    return summaries


def tree_max_abs(left, right) -> float:
    if isinstance(left, dict):
        return max((tree_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (tuple, list)):
        return max((tree_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, np.ndarray):
        return float(np.max(np.abs(left - right))) if left.size else 0.0
    if isinstance(left, (float, int, np.floating, np.integer)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def fitted_snapshot(lawful: dict) -> dict:
    return {"mean": lawful["channel_mean"], "variance": lawful["channel_var"],
            "static_mean": lawful["static"]["mean"], "basis_projector": lawful["static"]["basis"] @ lawful["static"]["basis"].T,
            "loading_cov": lawful["static"]["loading"] @ lawful["static"]["loading"].T,
            "residual": lawful["static"]["residual"], "selected_taus": lawful["selection"]["taus"]}


def recovery_snapshot(inference: dict) -> dict:
    names = ("static_factor", "two_stage_pca_smooth", "selected_gpfa", "independent_channel_gp")
    snapshot = {}
    for name in names:
        records = []
        for output in inference[name]:
            if name == "static_factor": records.append(output["latent_mean"])
            elif name == "selected_gpfa": records.append(output["posterior"]["mean"])
            else: records.append(output.get("latent_mean", output.get("hidden_mean")))
        snapshot[name] = records
    return snapshot


def method_hidden_prediction_snapshot(method: str, outputs: list[dict], masks: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    records = []
    for output, mask in zip(outputs, masks):
        hidden = ~mask
        if method in ("static_factor", "known_gpfa", "selected_gpfa", "wrong_timescale_gpfa",
                      "wrong_ar_gpfa", "ar_rts_smoother", "exact_gaussian_oracle"):
            records.append(output["condition"]["mean"])
        else:
            records.append(output["hidden_mean"][hidden])
    return records


def method_latent_snapshot(method: str, outputs: list[dict]) -> list[np.ndarray]:
    records = []
    for output in outputs:
        if method in ("static_factor", "two_stage_pca_smooth"):
            records.append(output["latent_mean"])
        elif method in ("known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "exact_gaussian_oracle"):
            records.append(output["posterior"]["mean"])
        elif method == "ar_kalman_filter":
            records.append(output["filter_mean"])
        elif method == "ar_rts_smoother":
            records.append(output["smooth_mean"])
    return records


def hidden_target_poison_by_method(states: dict, test: tuple[np.ndarray, ...], masks: tuple[np.ndarray, ...]) -> dict:
    original = infer_frozen_states(states, test, masks)
    poisoned = []
    changed = 0.0
    for values, mask in zip(test, masks):
        copy = values.copy()
        copy[~mask] += 50.0
        changed = max(changed, float(np.max(np.abs(copy - values))))
        poisoned.append(copy)
    altered = infer_frozen_states(states, tuple(poisoned), masks)
    records = {}
    for method_record in METHODS:
        method = method_record["id"]
        masked_applicable, _ = applicability(method, "masked_prediction_rmse", SCENARIOS[0])
        latent_applicable, _ = applicability(method, "latent_rmse", SCENARIOS[0])
        records[method] = {
            "claims_masked_prediction": masked_applicable,
            "hidden_prediction_max_abs": (tree_max_abs(method_hidden_prediction_snapshot(method, original[method], masks),
                                                        method_hidden_prediction_snapshot(method, altered[method], masks))
                                          if masked_applicable else None),
            "claims_latent_recovery": latent_applicable,
            "latent_recovery_max_abs": (tree_max_abs(method_latent_snapshot(method, original[method]),
                                                       method_latent_snapshot(method, altered[method]))
                                        if latent_applicable else None),
        }
    return {"hidden_values_materially_changed": changed > 1.0, "by_method": records}


def capability_poison_diagnostics() -> dict:
    scenario = SCENARIOS[0]
    data = generate_replicate(scenario, 0)
    train = tuple(x["observed"] for x in data["train"])
    validation = tuple(x["observed"] for x in data["validation"])
    test = tuple(x["observed"] for x in data["test"])
    lawful = fit_lawful_states(train, validation)
    states = assemble_frozen_states(scenario, lawful)
    masks = tuple(make_masks(scenario, 0, list(test)))
    inference = infer_frozen_states(states, test, masks)
    poisoned_test = tuple(x + 50.0 for x in test)
    lawful_again = fit_lawful_states(train, validation)
    fixed_original = infer_frozen_states(assemble_frozen_states(scenario, lawful_again), test, masks)
    poisoned_truth = {split: [dict(trial, latent=trial["latent"] + 100.0) for trial in trials] for split, trials in data.items()}
    lawful_truth = fit_lawful_states(tuple(x["observed"] for x in poisoned_truth["train"]), tuple(x["observed"] for x in poisoned_truth["validation"]))
    changed_train = list(train); changed_train[0] = changed_train[0] + 3.0
    changed_fit = fit_lawful_states(tuple(changed_train), validation)
    return {
        "lawful_fit_signature_excludes_test_and_truth": True,
        "frozen_recovery_signature_excludes_train_validation_and_truth": True,
        "test_observation_poison_complete_fitted_state_max_abs": tree_max_abs(fitted_snapshot(lawful), fitted_snapshot(lawful_again)),
        "test_observation_poison_fixed_recovery_max_abs": tree_max_abs(recovery_snapshot(inference), recovery_snapshot(fixed_original)),
        "latent_truth_poison_complete_fitted_state_max_abs": tree_max_abs(fitted_snapshot(lawful), fitted_snapshot(lawful_truth)),
        "latent_truth_poison_fixed_recovery_max_abs": tree_max_abs(recovery_snapshot(inference), recovery_snapshot(infer_frozen_states(assemble_frozen_states(scenario, lawful_truth), test, masks))),
        "poisoned_test_changed": tree_max_abs(test, poisoned_test) > 1.0,
        "allowed_train_poison_changed_fitted_state": tree_max_abs(fitted_snapshot(lawful), fitted_snapshot(changed_fit)) > 1e-6,
        "hidden_target_poison": hidden_target_poison_by_method(states, test, masks),
    }


def oracle_diagnostics() -> dict:
    scenario_lookup = {scenario["id"]: scenario for scenario in SCENARIOS}
    applicable_records = {}
    for scenario_id in ("reference_distinct", "latent_changepoint", "correlated_residuals"):
        scenario = scenario_lookup[scenario_id]
        data = generate_replicate(scenario, 1)
        lawful = fit_lawful_states(tuple(x["observed"] for x in data["train"]),
                                   tuple(x["observed"] for x in data["validation"]))
        states = assemble_frozen_states(scenario, lawful)
        expected_residual = correlated_residual() if scenario.get("residual") == "correlated" else R
        expected = covariance_model(scenario["taus"], residual=expected_residual,
                                    changepoint=scenario.get("generator") == "changepoint")
        values = data["test"][0]["observed"]
        mask = make_masks(scenario, 1, [values])[0]
        covariance = latent_posterior(states["oracle"], values, mask)
        precision = precision_latent_posterior(states["oracle"], values, mask)
        applicable_records[scenario_id] = {
            "oracle_present": states["oracle"] is not None,
            "covariance_wiring_max_abs": float(np.max(np.abs(states["oracle"]["covariance"] - expected["covariance"]))),
            "latent_covariance_wiring_max_abs": float(np.max(np.abs(states["oracle"]["latent_cov"] - expected["latent_cov"]))),
            "residual_wiring_max_abs": float(np.max(np.abs(states["oracle"]["residual"] - expected_residual))),
            "precision_mean_max_abs": float(np.max(np.abs(covariance["mean"] - precision["mean"]))),
            "precision_cov_max_abs": float(np.max(np.abs(covariance["cov"] - precision["cov"]))),
        }
    inapplicable_records = {}
    for scenario_id in ("low_rate_poisson", "alignment_jitter", "informative_missingness"):
        scenario = scenario_lookup[scenario_id]
        data = generate_replicate(scenario, 1)
        lawful = fit_lawful_states(tuple(x["observed"] for x in data["train"]),
                                   tuple(x["observed"] for x in data["validation"]))
        states = assemble_frozen_states(scenario, lawful)
        inapplicable_records[scenario_id] = {
            "oracle_absent": states["oracle"] is None,
            "applicability_declared_false": not oracle_applicable(scenario),
        }
    scenario = scenario_lookup["reference_distinct"]
    data = generate_replicate(scenario, 1)
    lawful = fit_lawful_states(tuple(x["observed"] for x in data["train"]),
                               tuple(x["observed"] for x in data["validation"]))
    states = assemble_frozen_states(scenario, lawful)
    values = data["test"][0]["observed"]
    mask = make_masks(scenario, 1, [values])[0]
    recursive = kalman_filter_smoother(values, mask, states["ar_model"]["taus"])
    ar_batch = latent_posterior(states["ar_model"], values, mask)
    return {
        "applicable_scenarios": applicable_records,
        "inapplicable_scenarios": inapplicable_records,
        "ar_reference": {
            "rts_batch_mean_max_abs": float(np.max(np.abs(recursive["smooth_mean"] - ar_batch["mean"]))),
            "rts_batch_variance_max_abs": float(np.max(np.abs(recursive["smooth_var"] - ar_batch["var"]))),
        },
    }


def scoring_fault_diagnostics() -> dict:
    scenario = SCENARIOS[0]
    nonfinite_rows, _ = evaluate_replicate(scenario, 0, fault="scoring_nonfinite")
    linalg_rows, _ = evaluate_replicate(scenario, 0, fault="inference_linalg")
    invalid_rows, _ = evaluate_replicate(scenario, 0, fault="inference_invalid")
    nonfinite = next(row for row in nonfinite_rows if row["method"] == "known_gpfa" and row["metric"] == "latent_rmse")
    linalg_applicable = [row for row in linalg_rows if applicability(row["method"], row["metric"], scenario)[0] and inference_required(row["metric"])]
    invalid_applicable = [row for row in invalid_rows if applicability(row["method"], row["metric"], scenario)[0] and inference_required(row["metric"])]
    expected_inapplicable = [row for row in linalg_rows if not applicability(row["method"], row["metric"], scenario)[0]]
    fit_diagnostics = [row for row in linalg_rows if applicability(row["method"], row["metric"], scenario)[0] and not inference_required(row["metric"])]
    return {
        "nonfinite_scoring_status": nonfinite["status"],
        "inference_linalg_all_applicable_nonconvergence": bool(linalg_applicable) and all(row["status"] == "nonconvergence" for row in linalg_applicable),
        "inference_invalid_all_applicable_invalid": bool(invalid_applicable) and all(row["status"] == "invalid" for row in invalid_applicable),
        "inapplicable_rows_preserved": bool(expected_inapplicable) and all(row["status"] == "inapplicable" for row in expected_inapplicable),
        "non_inference_fit_diagnostics_preserved": bool(fit_diagnostics) and all(row["status"] == "ok" for row in fit_diagnostics),
        "machine_readable_reasons_present": all(row["reason"] for row in [nonfinite, *linalg_applicable, *invalid_applicable, *expected_inapplicable]),
    }


def verification_diagnostics() -> dict:
    return {"capability_and_poison": capability_poison_diagnostics(), "covariance_and_oracles": oracle_diagnostics(),
            "scoring_fault_injection": scoring_fault_diagnostics()}


def oracle_max_discrepancy(checks: dict) -> float:
    values = []
    for record in checks["applicable_scenarios"].values():
        values.extend(value for key, value in record.items() if key.endswith("_max_abs"))
    values.extend(checks["ar_reference"].values())
    return max(values, default=0.0)


def make_summary(path: Path, summaries: list[dict], checks: dict) -> None:
    width, height = 1500, 1240
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((35, 25), "P4-U03 Temporal Covariance and Inference Benchmark", fill="black", font=font)
    draw.text((35, 47), "Separate target panels; medians across three replicates; no pooled score or universal winner", fill="#444", font=font)
    panels = [
        ("static observation set", "reference_distinct", "masked_prediction_rmse", ["mean_diagonal", "static_factor"]),
        ("retrospective set", "reference_distinct", "masked_prediction_rmse", ["two_stage_pca_smooth", "independent_channel_gp", "known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "ar_rts_smoother"]),
        ("online set", "reference_distinct", "latent_rmse", ["ar_kalman_filter"]),
        ("retrospective set", "reference_distinct", "latent_rmse", ["two_stage_pca_smooth", "known_gpfa", "selected_gpfa", "wrong_timescale_gpfa", "wrong_ar_gpfa", "ar_rts_smoother"]),
        ("oracle retrospective set", "reference_distinct", "masked_prediction_rmse", ["exact_gaussian_oracle"]),
        ("selection diagnostic", "close_timescales", "timescale_l1", ["selected_gpfa"]),
    ]
    lookup = {(x["scenario"], x["method"], x["metric"]): x for x in summaries}
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    for panel_index, (information_label, scenario, metric, methods) in enumerate(panels):
        left = 45 + (panel_index % 2) * 735
        top = 95 + (panel_index // 2) * 365
        right, bottom = left + 675, top + 315
        draw.rectangle((left, top, right, bottom), outline="#999", width=1)
        draw.text((left + 10, top + 10), f"{information_label} | {scenario}: {metric}", fill="black", font=font)
        values = [lookup.get((scenario, m, metric), {}).get("median") for m in methods]
        finite = [v for v in values if v is not None]
        scale = max(finite, default=1.0) * 1.15 or 1.0
        for i, (method, value) in enumerate(zip(methods, values)):
            y = top + 43 + i * 36
            draw.text((left + 10, y), method, fill="black", font=font)
            if value is not None:
                bar = int(390 * value / scale)
                draw.rectangle((left + 240, y, left + 240 + bar, y + 18), fill=colors[i % len(colors)])
                draw.text((left + 245 + bar, y), f"{value:.4g}", fill="black", font=font)
            else:
                draw.text((left + 245, y), "inapplicable", fill="#777", font=font)
    oracle = checks["covariance_and_oracles"]
    draw.text((35, 1210), f"max oracle discrepancy: {oracle_max_discrepancy(oracle):.3g}; information sets and targets are not ranked together", fill="#333", font=font)
    image.save(path, optimize=False, compress_level=9)


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
        fresh_rows.extend(rows); selections.append(selection)
    prior_rows = read_results(root / "results.csv") if resume else []
    prior = validate_resume_rows(prior_rows, fresh_rows) if resume else {}
    rows = [prior.get(serialized_row(row)["row_key"], row) for row in fresh_rows]
    write_results(root / "results.csv", rows)
    canonical_rows = read_results(root / "results.csv")
    summaries = summarize(canonical_rows)
    checks = verification_diagnostics()
    counts = {status: sum(row["status"] == status for row in canonical_rows) for status in STATUS_VALUES}
    diagnostics = {
        "schema_version": 1, "summary_grouping": ["scenario", "method", "metric"], "quantiles": [.10, .50, .90],
        "summaries": summaries, "status_counts": counts,
        "selection_by_replicate": sorted(selections, key=lambda x: (x["scenario"], x["replicate"])),
        "verification_checks": checks,
        "scientific_limits": [
            "Results are conditional on this six-channel two-latent bounded simulator and three replicates; q10/q90 are descriptive, not confidence intervals.",
            "The scenario grid is core-plus-stress rather than factorial and does not isolate causal effects of individual factors.",
            "Smooth posterior means are covariance-conditioned estimates, not evidence of learned transition dynamics.",
            "Poisson, changepoint, residual-correlation, alignment-jitter, and informative-missingness scenarios deliberately violate parts of nominal Gaussian GPFA assumptions.",
            "Post-hoc orthogonal latent alignment is a scoring convention and does not establish unique latent coordinates.",
            "Exact Gaussian oracles are withheld where the declared finite Gaussian observation model or ignorable mask is unavailable.",
            "No result establishes mechanism, causality, biological validity, real-data performance, manifold structure, intrinsic dimension, or universal superiority.",
        ],
    }
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    make_summary(root / "summary.png", summaries, checks)
    hashes = {name: sha256(root / name) for name in ARTIFACT_NAMES}
    manifest = {
        "schema_version": 1, "object_id": "P4-U03", "artifact": "temporal-covariance-inference", "date": "2026-08-20",
        "maturity": "L1", "status": "stable",
        "scientific_question": "Where are bounded temporal covariance and inference methods applicable, robust, misspecified, or numerically brittle under declared synthetic trajectory scenarios?",
        "ground_truth": "two latent finite-grid temporal processes observed through a six-channel loading map, with declared Gaussian and misspecified stress generators",
        "stable_scientific_design_provenance": {
            "source": str(PHASE3_PATH.relative_to(REPO_ROOT)),
            "source_sha256": sha256(PHASE3_PATH),
            "relationship": "scientific design, scenario vocabulary, and formula provenance only; no executable import or call",
        },
        "implementation": "benchmark-local deterministic implementation in benchmarks/temporal-covariance-inference.py",
        "stages": ["generation", "fitting_selection", "inference", "scoring"],
        "split_policy": "independent train, validation, and test trials per scenario replicate; test observations never enter lawful fitting/selection and latent truth is scoring-only",
        "master_seed": MASTER_SEED, "seed_mapping": "PCG64(SeedSequence(master_seed, sha256(scenario_id)[:4], replicate, stage, split_code, trial_index))",
        "replicates": REPLICATES, "dimensions": {"latent": Q, "channels": P, "time_bins": T},
        "scenarios": list(SCENARIOS), "methods": list(METHODS), "metrics": list(METRICS), "status_vocabulary": list(STATUS_VALUES),
        "selection": {"candidate_timescales": [list(x) for x in TAU_GRID],
                      "criterion": "minimum combined train-plus-validation total Gaussian observation NLL with lexicographic tau tie-break",
                      "criterion_data": ["train observations", "validation observations"],
                      "test_inputs_allowed": False, "latent_truth_allowed": False},
        "capability_separation": {"lawful_fit": "fit_lawful_states receives train and validation observations only", "frozen_inference": "infer_frozen_states receives frozen states, test observations and masks only", "scoring": "score_metric receives latent and observation truth only after inference"},
        "latent_alignment": "static-factor and two-stage rotations are fit once from training estimates/truth in the scoring stage and frozen for test; known-layer GPFA and AR methods retain declared coordinates",
        "oracle_rule": "exact Gaussian oracle is emitted only for finite Gaussian, non-jitter, ignorable-mask scenarios; changepoint and correlated-residual truth are included explicitly when Gaussian",
        "provenance_vocabulary": ["test", "fit_diagnostic", "oracle_diagnostic"],
        "metric_provenance": {metric: provenance_for_metric(metric) for metric in METRICS},
        "canonical_row_order": ["scenario", "replicate", "provenance", "method", "metric"],
        "row_key": ["scenario", "replicate", "provenance", "method", "metric"],
        "aggregation": {"within_replicate": "pool declared held-out test trials/targets for one scenario replicate", "across_replicates": "q10, median and q90 across three independent replicate-level values", "rates": "ok, failure=(invalid+nonconvergence), and applicability rates retain scenario/method/metric strata"},
        "resume": "partial rows must exactly match freshly recomputed canonical rows; duplicate, unknown, stale, corrupted, or incomplete rows are rejected",
        "failure_label_semantics": "method×metric×scenario applicability is resolved before inference/scoring failure; every non-ok row has a machine-readable reason",
        "tolerances": TOL, "artifact_files": ["manifest.json", *ARTIFACT_NAMES], "files_sha256": hashes,
        "commands": {"python_executable": sys.executable, "generate": f"{sys.executable} benchmarks/temporal-covariance-inference.py --generate", "verify": f"{sys.executable} benchmarks/temporal-covariance-inference.py --verify", "resume": f"{sys.executable} benchmarks/temporal-covariance-inference.py --generate --resume"},
        "runtime": {"python_executable": sys.executable, "python": sys.version.split()[0], "numpy": np.__version__, "pillow": getattr(sys.modules.get("PIL"), "__version__", "unknown"), "platform": platform.platform()},
        "interpretation_boundary": "Scenario/target/information-set conditional; not transition-mechanism evidence, causality, biology, real-data validity, intrinsic dimension, or universal superiority.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result = {"artifact_root": str(root), "rows": len(rows), "runtime_seconds": time.perf_counter() - start, "status_counts": counts, "hashes": hashes}
    if not quiet:
        print(json.dumps(result, indent=2))
    return result


def compare_roots(left: Path, right: Path) -> list[str]:
    return [name for name in ("manifest.json", *ARTIFACT_NAMES) if (left / name).read_bytes() != (right / name).read_bytes()]


def validate_artifacts(root: Path) -> list[str]:
    errors = []
    expected = {"manifest.json", *ARTIFACT_NAMES}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        return [f"artifact set mismatch: expected {sorted(expected)}, got {sorted(actual)}"]
    manifest = json.loads((root / "manifest.json").read_text())
    diagnostics = json.loads((root / "diagnostics.json").read_text())
    rows = read_results(root / "results.csv")
    for row in rows:
        try: validate_result_row(row)
        except Exception as exc: errors.append(f"invalid row {row.get('row_key')}: {exc}")
    for name, digest in manifest.get("files_sha256", {}).items():
        if sha256(root / name) != digest: errors.append(f"hash mismatch: {name}")
    design = manifest.get("stable_scientific_design_provenance", {})
    if design.get("source") != str(PHASE3_PATH.relative_to(REPO_ROOT)) or design.get("source_sha256") != sha256(PHASE3_PATH):
        errors.append("stable scientific/design provenance digest mismatch")
    if "no executable import or call" not in design.get("relationship", ""):
        errors.append("scientific provenance incorrectly implies executable reuse")
    keys = [row["row_key"] for row in rows]
    if len(keys) != len(set(keys)): errors.append("row keys are not unique")
    if rows != sorted(rows, key=canonical_key): errors.append("results are not in canonical order")
    expected_keys = {f"{s['id']}|{r}|{provenance_for_metric(metric)}|{m['id']}|{metric}" for s in SCENARIOS for r in range(REPLICATES) for m in METHODS for metric in METRICS}
    if set(keys) != expected_keys: errors.append(f"raw-row Cartesian product mismatch: missing={len(expected_keys-set(keys))}, extra={len(set(keys)-expected_keys)}")
    required_classes = {"timescale_separation", "trial_sample_size", "channel_time_masks", "changepoints", "correlated_residuals", "poisson_misspecification", "equal_kernels", "alignment_jitter", "informative_missingness"}
    if not required_classes.issubset({row["scenario_class"] for row in rows}): errors.append("required scenario classes missing")
    if diagnostics.get("summaries") != summarize(rows): errors.append("diagnostic summaries do not match raw rows")
    counts = {status: sum(row["status"] == status for row in rows) for status in STATUS_VALUES}
    if diagnostics.get("status_counts") != counts: errors.append("status counts do not match raw rows")
    scenario_lookup = {s["id"]: s for s in SCENARIOS}
    for row in rows:
        expected_applicable, _ = applicability(row["method"], row["metric"], scenario_lookup[row["scenario"]])
        if not expected_applicable and row["status"] != "inapplicable": errors.append(f"applicability precedence failure: {row['row_key']}")
        if row["provenance"] != provenance_for_metric(row["metric"]): errors.append(f"metric provenance mismatch: {row['row_key']}")
    checks = diagnostics["verification_checks"]
    poison = checks["capability_and_poison"]
    for name in ("test_observation_poison_complete_fitted_state_max_abs", "test_observation_poison_fixed_recovery_max_abs", "latent_truth_poison_complete_fitted_state_max_abs", "latent_truth_poison_fixed_recovery_max_abs"):
        if poison[name] > TOL["poison"]: errors.append(f"poison check failed: {name}")
    if not poison["allowed_train_poison_changed_fitted_state"] or not poison["poisoned_test_changed"]: errors.append("poison fixtures ineffective")
    hidden_poison = poison["hidden_target_poison"]
    if not hidden_poison["hidden_values_materially_changed"]:
        errors.append("hidden-target poison fixture did not change hidden values")
    for method, record in hidden_poison["by_method"].items():
        if record["claims_masked_prediction"] and record["hidden_prediction_max_abs"] > TOL["poison"]:
            errors.append(f"hidden-target prediction leakage: {method}")
        if record["claims_latent_recovery"] and record["latent_recovery_max_abs"] > TOL["poison"]:
            errors.append(f"hidden-target latent leakage: {method}")
    oracle = checks["covariance_and_oracles"]
    for scenario_id, record in oracle["applicable_scenarios"].items():
        if not record["oracle_present"] or any(value > TOL["oracle"] for key, value in record.items() if key.endswith("_max_abs")):
            errors.append(f"scenario-specific covariance/oracle check failed: {scenario_id}:{record}")
    for scenario_id, record in oracle["inapplicable_scenarios"].items():
        if not record["oracle_absent"] or not record["applicability_declared_false"]:
            errors.append(f"oracle-inapplicable scenario wiring failed: {scenario_id}:{record}")
    if any(value > TOL["oracle"] for value in oracle["ar_reference"].values()):
        errors.append(f"AR covariance/oracle check failed: {oracle['ar_reference']}")
    faults = checks["scoring_fault_injection"]
    if not (faults["nonfinite_scoring_status"] == "invalid"
            and faults["inference_linalg_all_applicable_nonconvergence"]
            and faults["inference_invalid_all_applicable_invalid"]
            and faults["inapplicable_rows_preserved"]
            and faults["non_inference_fit_diagnostics_preserved"]
            and faults["machine_readable_reasons_present"]):
        errors.append("real-path fault injection labels failed")
    selection = manifest.get("selection", {})
    if selection.get("criterion") != "minimum combined train-plus-validation total Gaussian observation NLL with lexicographic tau tie-break" or selection.get("test_inputs_allowed") is not False:
        errors.append("selection criterion metadata mismatch")
    if manifest.get("metric_provenance") != {metric: provenance_for_metric(metric) for metric in METRICS}: errors.append("manifest metric provenance mismatch")
    return errors


def adversarial_resume_checks(full_rows: list[dict], root: Path) -> list[str]:
    failures = []
    def reject(label, fn):
        try: fn()
        except (ValueError, RuntimeError): return
        failures.append(f"adversarial resume case accepted: {label}")
    fresh_rows, _ = evaluate_replicate(SCENARIOS[0], 0)
    fresh_keys = {serialized_row(row)["row_key"] for row in fresh_rows}
    subset = [dict(row) for row in full_rows if row["row_key"] in fresh_keys][:3]
    corrupted = [dict(x) for x in subset]; corrupted[0]["value"] = "999" if corrupted[0]["status"] == "ok" else "1"
    reject("corrupted", lambda: validate_resume_rows(corrupted, fresh_rows))
    stale = [dict(x) for x in subset]; stale[0]["n_train"] = "999"
    reject("stale", lambda: validate_resume_rows(stale, fresh_rows))
    reject("duplicate", lambda: validate_resume_rows([dict(subset[0]), dict(subset[0])], fresh_rows))
    unknown = [dict(subset[0])]; unknown[0]["row_key"] += "|unknown"
    reject("unknown", lambda: validate_resume_rows(unknown, fresh_rows))
    incomplete = root / "incomplete.csv"
    with incomplete.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS[:-1]); writer.writeheader(); writer.writerow({f: subset[0][f] for f in FIELDS[:-1]})
    reject("incomplete", lambda: read_results(incomplete))
    return failures


def verify(root: Path) -> None:
    errors = validate_artifacts(root)
    with tempfile.TemporaryDirectory(prefix="ldg-p4-u03-") as temporary:
        temporary = Path(temporary)
        normal, reverse, resumed = temporary / "normal", temporary / "reverse", temporary / "resumed"
        build(normal, quiet=True); build(reverse, reverse=True, quiet=True)
        errors.extend(f"byte regeneration mismatch: {name}" for name in compare_roots(root, normal))
        errors.extend(f"reversed traversal mismatch: {name}" for name in compare_roots(normal, reverse))
        resumed.mkdir(); full_rows = read_results(normal / "results.csv"); write_results(resumed / "results.csv", full_rows[:len(full_rows)//2]); build(resumed, resume=True, quiet=True)
        errors.extend(f"resume mismatch: {name}" for name in compare_roots(normal, resumed))
        errors.extend(adversarial_resume_checks(full_rows, temporary))
    if errors:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"verified": True, "artifact_root": str(root), "checks": ["exact_artifact_set", "hashes", "canonical_unique_complete_rows", "summary_recomputation", "clean_byte_regeneration", "reversed_order", "valid_partial_resume", "adversarial_resume_rejection", "capability_separation", "test_and_truth_poison", "covariance_and_exact_oracles", "failure_and_nonfinite_fault_injection"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true"); parser.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(); generate, check = args.generate, args.verify
    if not generate and not check: generate = check = True
    if generate: build(args.artifact_root, reverse=args.reverse, resume=args.resume)
    if check: verify(args.artifact_root)


if __name__ == "__main__":
    main()
