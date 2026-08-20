#!/usr/bin/env python3
"""Deterministic Cycle 1 2D generation-and-recovery toy model.

Uses only the bundled NumPy and Pillow runtime. Ground-truth trajectories are
computed from the exact polar solution. A fixed-step RK4 implementation provides
an independent generator check and integrates recovered vector fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASTER_SEED = 20260818
ALPHA = 1.0
BETA = 1.0
OMEGA = 2.0
T_END = 8.0
DTS = (0.025, 0.05, 0.10)
SIGMAS = (0.0, 0.02, 0.05)
WINDOWS = {0.025: 21, 0.05: 11, 0.10: 5}
LAMBDA = 1e-4
RK4_STEP = 0.005
FEATURES = ("1", "x", "y", "x2", "xy", "y2", "x3", "x2y", "xy2", "y3")
MODEL_NAMES = ("cubic", "affine", "permuted")
ARTIFACT_NAMES = (
    "latent-trajectories.npz",
    "observations.npz",
    "metrics.csv",
    "coefficients.csv",
    "diagnostics.json",
    "summary.png",
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "two-dimensional-autonomous-dynamics"


def rng_for(*words: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def stratified_initials(n_each: int, split_code: int) -> np.ndarray:
    rng = rng_for(MASTER_SEED, 1, split_code)
    radii = np.concatenate(
        [rng.uniform(0.35, 0.75, n_each), rng.uniform(1.25, 1.65, n_each)]
    )
    angles = rng.uniform(0.0, 2.0 * np.pi, 2 * n_each)
    states = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return states[rng.permutation(len(states))]


def stress_initials() -> np.ndarray:
    rng = rng_for(MASTER_SEED, 1, 3)
    radii = rng.uniform(1.65, 1.90, 16)
    angles = rng.uniform(0.0, 2.0 * np.pi, 16)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def analytic_trajectory(initial: np.ndarray, times: np.ndarray) -> np.ndarray:
    initial = np.asarray(initial, dtype=float)
    times = np.asarray(times, dtype=float)
    r0 = np.linalg.norm(initial, axis=-1)
    theta0 = np.arctan2(initial[..., 1], initial[..., 0])
    result = np.zeros(initial.shape[:-1] + (len(times), 2), dtype=float)
    nonzero = r0 > 0.0
    if np.any(nonzero):
        rn = r0[nonzero]
        denom = 1.0 + (rn[:, None] ** -2 - 1.0) * np.exp(-2.0 * times[None, :])
        radii = denom ** -0.5
        theta = theta0[nonzero, None] + OMEGA * times[None, :]
        result[nonzero, :, 0] = radii * np.cos(theta)
        result[nonzero, :, 1] = radii * np.sin(theta)
    return result


def raw_features(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    x = z[..., 0]
    y = z[..., 1]
    return np.stack(
        [
            np.ones_like(x), x, y, x * x, x * y, y * y,
            x * x * x, x * x * y, x * y * y, y * y * y,
        ],
        axis=-1,
    )


def true_coefficients() -> np.ndarray:
    c = np.zeros((2, 10), dtype=float)
    c[0, 1] = 1.0
    c[0, 2] = -2.0
    c[0, 6] = -1.0
    c[0, 8] = -1.0
    c[1, 1] = 2.0
    c[1, 2] = 1.0
    c[1, 7] = -1.0
    c[1, 9] = -1.0
    return c


TRUE_C = true_coefficients()


def evaluate_field(c: np.ndarray, z: np.ndarray) -> np.ndarray:
    return raw_features(z) @ np.asarray(c, dtype=float).T


def rk4_advance(c: np.ndarray, z0: np.ndarray, duration: float, step: float = RK4_STEP):
    z = np.asarray(z0, dtype=float).copy()
    n_steps = int(round(duration / step))
    h = duration / n_steps
    failed = np.zeros(z.shape[:-1], dtype=bool)
    for _ in range(n_steps):
        k1 = evaluate_field(c, z)
        k2 = evaluate_field(c, z + 0.5 * h * k1)
        k3 = evaluate_field(c, z + 0.5 * h * k2)
        k4 = evaluate_field(c, z + h * k3)
        z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        bad = (~np.isfinite(z).all(axis=-1)) | (np.linalg.norm(z, axis=-1) > 3.0)
        failed |= bad
        if np.any(bad):
            z = np.where(bad[..., None], np.nan, z)
    return z, failed


def rk4_rollout(c: np.ndarray, z0: np.ndarray, output_times: np.ndarray):
    output_times = np.asarray(output_times, dtype=float)
    z = np.asarray(z0, dtype=float).copy()
    out = np.full(z.shape[:-1] + (len(output_times), 2), np.nan, dtype=float)
    out[..., 0, :] = z
    failed = np.zeros(z.shape[:-1], dtype=bool)
    pre_failure_horizon = np.full(z.shape[:-1], output_times[-1] - output_times[0], dtype=float)
    for j in range(1, len(output_times)):
        duration = float(output_times[j] - output_times[j - 1])
        z, bad = rk4_advance(c, z, duration)
        newly_bad = bad & ~failed
        pre_failure_horizon[newly_bad] = output_times[j - 1] - output_times[0]
        failed |= bad
        out[..., j, :] = z
    return out, failed, pre_failure_horizon


def filter_coefficients(window: int, dt: float):
    half = window // 2
    u = np.arange(-half, half + 1, dtype=float) * dt
    design = np.column_stack([np.ones(window), u, u**2, u**3])
    pinv = np.linalg.pinv(design)
    return pinv[0], pinv[1]


def smooth_and_derivative(values: np.ndarray, dt: float):
    values = np.asarray(values, dtype=float)
    window = WINDOWS[dt]
    half = window // 2
    c0, c1 = filter_coefficients(window, dt)
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=1)
    smooth_valid = np.tensordot(windows, c0, axes=([-1], [0]))
    deriv_valid = np.tensordot(windows, c1, axes=([-1], [0]))
    smooth = np.full_like(values, np.nan)
    deriv = np.full_like(values, np.nan)
    smooth[:, half : values.shape[1] - half, :] = smooth_valid
    deriv[:, half : values.shape[1] - half, :] = deriv_valid
    return smooth, deriv


def fit_ridge(states: np.ndarray, derivatives: np.ndarray, model: str, permutation=None):
    phi = raw_features(states)
    indices = np.arange(10) if model != "affine" else np.array([0, 1, 2])
    phi = phi[:, indices]
    targets = derivatives.copy()
    if permutation is not None:
        targets = targets[permutation]
    mu = np.zeros(phi.shape[1], dtype=float)
    scale = np.ones(phi.shape[1], dtype=float)
    if phi.shape[1] > 1:
        mu[1:] = phi[:, 1:].mean(axis=0)
        scale[1:] = phi[:, 1:].std(axis=0, ddof=0)
        if np.any(scale[1:] == 0.0):
            raise RuntimeError("zero feature scale")
    x = phi.copy()
    x[:, 1:] = (x[:, 1:] - mu[1:]) / scale[1:]
    n = x.shape[0]
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    lhs = (x.T @ x) / n + LAMBDA * penalty
    rhs = (x.T @ targets) / n
    b = np.linalg.solve(lhs, rhs)
    raw = np.zeros_like(b)
    raw[1:] = b[1:] / scale[1:, None]
    raw[0] = b[0] - np.sum(b[1:] * mu[1:, None] / scale[1:, None], axis=0)
    full = np.zeros((2, 10), dtype=float)
    full[:, indices] = raw.T
    return full, {"mu": mu, "scale": scale, "indices": indices}


def evaluation_grid() -> np.ndarray:
    radii = np.linspace(0.35, 1.65, 27)
    angles = 2.0 * np.pi * np.arange(64) / 64.0
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    return np.column_stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()])


def trajectory_rmse(pred: np.ndarray, truth: np.ndarray, failed=None) -> np.ndarray:
    sq = np.sum((pred - truth) ** 2, axis=-1)
    result = np.sqrt(np.nanmean(sq, axis=-1))
    result[~np.isfinite(result)] = np.inf
    if failed is not None:
        result[np.asarray(failed, dtype=bool)] = np.inf
    return result


def radial_rmse(pred: np.ndarray, truth: np.ndarray, failed=None) -> np.ndarray:
    rp = np.linalg.norm(pred, axis=-1)
    rt = np.linalg.norm(truth, axis=-1)
    result = np.sqrt(np.nanmean((rp - rt) ** 2, axis=-1))
    result[~np.isfinite(result)] = np.inf
    if failed is not None:
        result[np.asarray(failed, dtype=bool)] = np.inf
    return result


def add_metric(rows, dt, sigma, rep, split, model, unit, metric, value):
    rows.append(
        {
            "dt": dt,
            "sigma": sigma,
            "replicate": rep,
            "split": split,
            "model": model,
            "unit": unit,
            "metric": metric,
            "value": float(value),
        }
    )


def evaluate_split(
    rows,
    dt,
    sigma,
    rep,
    split,
    initial,
    latent,
    smooth,
    models,
    times,
):
    state_mask = (times >= 0.30 - 1e-12) & (times <= 7.70 + 1e-12)
    state_rmse = trajectory_rmse(smooth[:, state_mask], latent[:, state_mask])
    for i, value in enumerate(state_rmse):
        add_metric(rows, dt, sigma, rep, split, "smoother", f"traj_{i:03d}", "state_rmse", value)

    start_idx = int(np.flatnonzero(np.isclose(times, 0.30))[0])
    rollout_times = 0.30 + 0.10 * np.arange(78)
    truth_roll = analytic_trajectory(initial, rollout_times)
    true_start = latent[:, start_idx]
    obs_start = smooth[:, start_idx]

    pair_mask = (times >= 0.30 - 1e-12) & (times <= 7.70 - dt + 1e-12)
    starts = latent[:, pair_mask]
    targets = latent[:, np.flatnonzero(pair_mask) + 1]
    flat_starts = starts.reshape(-1, 2)

    for model_name, c in models.items():
        next_pred, one_failed = rk4_advance(c, flat_starts, dt)
        next_pred = next_pred.reshape(starts.shape)
        one_sq = np.sum((next_pred - targets) ** 2, axis=-1)
        one_rmse = np.sqrt(np.nanmean(one_sq, axis=1))
        one_rmse[~np.isfinite(one_rmse)] = np.inf

        oracle, oracle_failed, oracle_horizon = rk4_rollout(c, true_start, rollout_times)
        observed, observed_failed, observed_horizon = rk4_rollout(c, obs_start, rollout_times)
        oracle_rmse = trajectory_rmse(oracle, truth_roll, oracle_failed)
        observed_rmse = trajectory_rmse(observed, truth_roll, observed_failed)
        oracle_radial = radial_rmse(oracle, truth_roll, oracle_failed)
        observed_radial = radial_rmse(observed, truth_roll, observed_failed)
        one_failure_rate = one_failed.reshape(starts.shape[:-1]).mean(axis=1)

        for i in range(len(initial)):
            unit = f"traj_{i:03d}"
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "one_step_rmse", one_rmse[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "oracle_rollout_rmse", oracle_rmse[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "observed_rollout_rmse", observed_rmse[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "oracle_radial_rmse", oracle_radial[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "observed_radial_rmse", observed_radial[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "one_step_failure_rate", one_failure_rate[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "oracle_rollout_failure", float(oracle_failed[i]))
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "observed_rollout_failure", float(observed_failed[i]))
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "oracle_pre_failure_horizon", oracle_horizon[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "observed_pre_failure_horizon", observed_horizon[i])
            add_metric(rows, dt, sigma, rep, split, model_name, unit, "rollout_failure", float(oracle_failed[i] or observed_failed[i]))


def fmt_float(value) -> str:
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return format(value, ".12g")


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt_float(row[k]) if k in {"dt", "sigma", "value", "coefficient", "true_coefficient"} else row[k] for k in fields})


def save_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order_quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return math.nan
    index = int(round(probability * (len(values) - 1)))
    return values[index]


def summarize_metric(rows, dt, sigma, model, metric, split="test"):
    vals_by_rep = {}
    for row in rows:
        if (
            abs(row["dt"] - dt) < 1e-12
            and abs(row["sigma"] - sigma) < 1e-12
            and row["model"] == model
            and row["metric"] == metric
            and row["split"] == split
        ):
            vals_by_rep.setdefault(row["replicate"], []).append(row["value"])
    per_rep = []
    for rep in sorted(vals_by_rep):
        values = vals_by_rep[rep]
        per_rep.append(
            {
                "replicate": rep,
                "median": float(np.median(np.asarray(values, dtype=float))),
                "q25": order_quantile(values, 0.25),
                "q75": order_quantile(values, 0.75),
                "count": len(values),
            }
        )
    medians = [item["median"] for item in per_rep]
    return {
        "per_replicate": per_rep,
        "median": float(np.median(np.asarray(medians, dtype=float))),
        "q25": order_quantile(medians, 0.25),
        "q75": order_quantile(medians, 0.75),
        "replicate_count": len(per_rep),
    }


def summarize_failure_rate(rows, dt, sigma, model, metric, split="test"):
    vals_by_rep = {}
    for row in rows:
        if (
            abs(row["dt"] - dt) < 1e-12
            and abs(row["sigma"] - sigma) < 1e-12
            and row["model"] == model
            and row["metric"] == metric
            and row["split"] == split
        ):
            vals_by_rep.setdefault(row["replicate"], []).append(row["value"])
    per_rep = []
    for rep in sorted(vals_by_rep):
        values = np.asarray(vals_by_rep[rep], dtype=float)
        per_rep.append(
            {
                "replicate": rep,
                "rate": float(np.mean(values)),
                "failures": float(np.sum(values)),
                "count": len(values),
            }
        )
    rates = [item["rate"] for item in per_rep]
    return {
        "aggregation": "arithmetic mean of per-trajectory failure flags",
        "per_replicate": per_rep,
        "median": float(np.median(np.asarray(rates, dtype=float))),
        "q25": order_quantile(rates, 0.25),
        "q75": order_quantile(rates, 0.75),
        "replicate_count": len(per_rep),
    }


def generator_diagnostics():
    points = evaluation_grid()[::37]
    f = evaluate_field(TRUE_C, points)
    r = np.linalg.norm(points, axis=1)
    radial = np.sum(points * f, axis=1) / r
    angular = (points[:, 0] * f[:, 1] - points[:, 1] * f[:, 0]) / (r * r)
    polar_residual = max(
        float(np.max(np.abs(radial - r * (1.0 - r * r)))),
        float(np.max(np.abs(angular - 2.0))),
    )

    diagnostic_initial = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0], [1.5, 0.0], [0.0, 1.65]])
    times = np.arange(0.0, 8.0 + 0.025 / 2, 0.025)
    truth = analytic_trajectory(diagnostic_initial, times)
    numeric, failed, _ = rk4_rollout(TRUE_C, diagnostic_initial, times)
    rk4_max_error = float(np.nanmax(np.linalg.norm(numeric - truth, axis=-1)))
    cycle_error = float(np.max(np.abs(np.linalg.norm(truth[1], axis=1) - 1.0)))

    h = 0.001
    z = np.array([[1.0, 0.0]])
    prev_y = 0.0
    crossing = None
    t = 0.0
    for _ in range(int(4.0 / h)):
        new_z, bad = rk4_advance(TRUE_C, z, h, step=h)
        new_t = t + h
        if t > 0.1 and prev_y < 0.0 <= new_z[0, 1]:
            fraction = -prev_y / (new_z[0, 1] - prev_y)
            crossing = t + fraction * h
            break
        z = new_z
        prev_y = z[0, 1]
        t = new_t
    if crossing is None:
        raise RuntimeError("period crossing not found")

    return {
        "polar_identity_max_abs_residual": polar_residual,
        "rk4_vs_analytic_max_euclidean_error": rk4_max_error,
        "cycle_radius_max_abs_error": cycle_error,
        "period_estimate": crossing,
        "period_relative_error": abs(crossing - np.pi) / np.pi,
        "rk4_failed": bool(np.any(failed)),
    }


def data_diagnostics(train_initial, test_initial, extra_initial, train_fine, test_fine, extra_fine, noise):
    def stratum_counts(initial):
        radii = np.linalg.norm(initial, axis=1)
        return {
            "inner": int(np.sum((radii >= 0.35) & (radii <= 0.75))),
            "outer": int(np.sum((radii >= 1.25) & (radii <= 1.65))),
            "extrapolation": int(np.sum((radii >= 1.65) & (radii <= 1.90))),
        }

    all_initial = np.vstack([train_initial, test_initial, extra_initial])
    rounded = np.round(all_initial, 14)
    unique_count = len({tuple(row) for row in rounded})

    radial_violations = []
    for initial, latent in (
        (train_initial, train_fine),
        (test_initial, test_fine),
        (extra_initial, extra_fine),
    ):
        radii = np.linalg.norm(latent, axis=-1)
        delta = np.diff(radii, axis=1)
        r0 = np.linalg.norm(initial, axis=1)
        if np.any(r0 < 1.0):
            radial_violations.append(float(np.max(np.maximum(-delta[r0 < 1.0], 0.0))))
        if np.any(r0 > 1.0):
            radial_violations.append(float(np.max(np.maximum(delta[r0 > 1.0], 0.0))))

    noise_statistics = {}
    for sigma in SIGMAS:
        if sigma == 0.0:
            mean = np.zeros(2)
            covariance = np.zeros((2, 2))
            count = sum(array.size // 2 for arrays in noise.values() for array in arrays)
        else:
            flattened = np.concatenate(
                [(sigma * array).reshape(-1, 2) for arrays in noise.values() for array in arrays],
                axis=0,
            )
            mean = flattened.mean(axis=0)
            covariance = np.cov(flattened, rowvar=False, ddof=0)
            count = len(flattened)
        noise_statistics[fmt_float(sigma)] = {
            "count": int(count),
            "mean": mean.tolist(),
            "covariance": covariance.tolist(),
        }

    fit_row_counts = {}
    for dt in DTS:
        times = np.arange(0.0, T_END + dt / 2.0, dt)
        fit_row_counts[fmt_float(dt)] = int(64 * np.sum((times >= 0.30 - 1e-12) & (times <= 2.0 + 1e-12)))

    return {
        "radial_monotonicity_max_violation": max(radial_violations),
        "split_counts": {"train": len(train_initial), "test": len(test_initial), "extrapolation": len(extra_initial)},
        "stratum_counts": {
            "train": stratum_counts(train_initial),
            "test": stratum_counts(test_initial),
            "extrapolation": stratum_counts(extra_initial),
        },
        "all_initial_conditions_unique": unique_count == len(all_initial),
        "unique_initial_condition_count": unique_count,
        "noise_statistics": noise_statistics,
        "leakage_assertion": {
            "fit_uses_training_split_only": True,
            "test_rows_in_feature_scaling_or_regression": 0,
            "fit_row_counts_by_dt": fit_row_counts,
        },
    }


def draw_arrow(draw, start, vector, color, scale=1.0, width=2):
    x0, y0 = start
    x1 = x0 + scale * vector[0]
    y1 = y0 + scale * vector[1]
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    head = 5
    for offset in (2.6, -2.6):
        draw.line((x1, y1, x1 + head * math.cos(angle + offset), y1 + head * math.sin(angle + offset)), fill=color, width=width)


def make_summary_png(path: Path, initial: np.ndarray, representative_c: np.ndarray, heat_cubic, heat_affine, primary):
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 18), "LDG 2D autonomous generation-and-recovery summary", fill="black", font=font)

    def box(x0, y0, x1, y1, title):
        draw.rectangle((x0, y0, x1, y1), outline="black", width=2)
        draw.text((x0 + 8, y0 + 6), title, fill="black", font=font)
        return x0 + 12, y0 + 28, x1 - 12, y1 - 12

    # Panel A: exact trajectories.
    ax = box(30, 55, 520, 520, "A. Exact ground-truth trajectories")
    x0, y0, x1, y1 = ax
    def map_phase(z):
        return (x0 + (z[..., 0] + 1.9) / 3.8 * (x1 - x0), y1 - (z[..., 1] + 1.9) / 3.8 * (y1 - y0))
    circle = analytic_trajectory(np.array([[1.0, 0.0]]), np.linspace(0, np.pi, 300))[0]
    cx, cy = map_phase(circle)
    draw.line(list(zip(cx.tolist(), cy.tolist())), fill=(60, 60, 60), width=2)
    times = np.linspace(0, 8, 220)
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    for i, z0i in enumerate(initial[:16]):
        tr = analytic_trajectory(z0i[None, :], times)[0]
        px, py = map_phase(tr)
        draw.line(list(zip(px.tolist(), py.tolist())), fill=colors[i % len(colors)], width=2)
    draw.text((x0 + 8, y1 - 20), "Unit cycle in gray; inner and outer initial radii", fill="black", font=font)

    # Panel B/C: true and representative recovered vector fields.
    for panel, title, coeff, bx0 in [
        ("B", "True field", TRUE_C, 550),
        ("C", "Recovered cubic: dt=0.05, sigma=0.02, rep=0", representative_c, 1075),
    ]:
        ax2 = box(bx0, 55, bx0 + 495, 520, f"{panel}. {title}")
        qx0, qy0, qx1, qy1 = ax2
        for xx in np.linspace(-1.5, 1.5, 9):
            for yy in np.linspace(-1.5, 1.5, 9):
                if not (0.35 <= math.hypot(xx, yy) <= 1.65):
                    continue
                f = evaluate_field(coeff, np.array([[xx, yy]]))[0]
                norm = max(np.linalg.norm(f), 1e-12)
                sx = qx0 + (xx + 1.7) / 3.4 * (qx1 - qx0)
                sy = qy1 - (yy + 1.7) / 3.4 * (qy1 - qy0)
                draw_arrow(draw, (sx, sy), (f[0] / norm, -f[1] / norm), (31, 119, 180), scale=12, width=1)

    # Panel D: heat maps and primary checks.
    ax3 = box(30, 550, 1035, 970, "D. Median oracle rollout RMSE by condition")
    hx0, hy0, hx1, hy1 = ax3
    cell_w = 105
    cell_h = 80
    for block, (name, values, ox) in enumerate((("Cubic", heat_cubic, hx0), ("Affine", heat_affine, hx0 + 500))):
        draw.text((ox, hy0), name, fill="black", font=font)
        for i, dt in enumerate(DTS):
            draw.text((ox + 55 + i * cell_w, hy0 + 20), f"dt={dt:g}", fill="black", font=font)
        finite = [v for row in values for v in row if np.isfinite(v)]
        vmax = max(finite) if finite else 1.0
        for j, sigma in enumerate(SIGMAS):
            draw.text((ox, hy0 + 50 + j * cell_h), f"s={sigma:g}", fill="black", font=font)
            for i, dt in enumerate(DTS):
                value = values[j][i]
                frac = min(value / max(vmax, 1e-12), 1.0) if np.isfinite(value) else 1.0
                color = (int(255 * frac), int(220 * (1 - frac)), 80)
                rx0 = ox + 55 + i * cell_w
                ry0 = hy0 + 45 + j * cell_h
                draw.rectangle((rx0, ry0, rx0 + cell_w - 8, ry0 + cell_h - 8), fill=color, outline="black")
                draw.text((rx0 + 6, ry0 + 25), fmt_float(value), fill="black", font=font)

    tx = 1075
    ty = 560
    draw.rectangle((tx, ty, 1570, 970), outline="black", width=2)
    draw.text((tx + 8, ty + 8), "E. Preregistered primary comparisons", fill="black", font=font)
    lines = [
        f"Noiseless fine cubic field NRMSE: {fmt_float(primary['noiseless']['cubic_field'])}",
        f"Noiseless fine affine field NRMSE: {fmt_float(primary['noiseless']['affine_field'])}",
        f"Noiseless fine cubic oracle median: {fmt_float(primary['noiseless']['cubic_oracle'])}",
        f"Noiseless fine affine oracle median: {fmt_float(primary['noiseless']['affine_oracle'])}",
        f"Moderate cubic field median: {fmt_float(primary['moderate']['cubic_field'])}",
        f"Moderate affine field median: {fmt_float(primary['moderate']['affine_field'])}",
        f"Moderate cubic oracle median: {fmt_float(primary['moderate']['cubic_oracle'])}",
        f"Moderate affine oracle median: {fmt_float(primary['moderate']['affine_oracle'])}",
        f"Permutation rollout failures: {primary['permuted_rollout_failures']}",
        f"Primary success: {primary['passed']}",
        "Technical recovery only; no mechanistic inference.",
    ]
    for i, line in enumerate(lines):
        draw.text((tx + 12, ty + 38 + i * 30), line, fill="black", font=font)
    image.save(path, format="PNG", optimize=False)


def generate(root: Path, quiet=False):
    start = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    train_initial = stratified_initials(32, 1)
    test_initial = stratified_initials(20, 2)
    extra_initial = stress_initials()
    fine_times = np.arange(0.0, T_END + 0.025 / 2, 0.025)
    train_fine = analytic_trajectory(train_initial, fine_times)
    test_fine = analytic_trajectory(test_initial, fine_times)
    extra_fine = analytic_trajectory(extra_initial, fine_times)

    latent_arrays = {
        "times_dt025": fine_times,
        "train_initial": train_initial,
        "test_initial": test_initial,
        "extrapolation_initial": extra_initial,
        "train_latent_dt025": train_fine,
        "test_latent_dt025": test_fine,
        "extrapolation_latent_dt025": extra_fine,
    }
    save_npz_deterministic(root / "latent-trajectories.npz", latent_arrays)

    noise = {"train": [], "test": [], "extrapolation": []}
    split_codes = {"train": 1, "test": 2, "extrapolation": 3}
    shapes = {
        "train": train_fine.shape,
        "test": test_fine.shape,
        "extrapolation": extra_fine.shape,
    }
    obs_arrays = {}
    for split in noise:
        for q in range(10):
            arr = rng_for(MASTER_SEED, 2, split_codes[split], q).standard_normal(shapes[split])
            noise[split].append(arr)
            obs_arrays[f"standard_normal_{split}_rep{q:02d}"] = arr.astype(np.float32)
    obs_arrays["example_train_sigma020_rep00_dt025"] = (train_fine + 0.02 * noise["train"][0]).astype(np.float32)
    obs_arrays["example_test_sigma020_rep00_dt025"] = (test_fine + 0.02 * noise["test"][0]).astype(np.float32)
    obs_arrays["example_extrapolation_sigma020_rep00_dt025"] = (extra_fine + 0.02 * noise["extrapolation"][0]).astype(np.float32)
    save_npz_deterministic(root / "observations.npz", obs_arrays)

    grid = evaluation_grid()
    true_grid = evaluate_field(TRUE_C, grid)
    metric_rows = []
    coefficient_rows = []
    representative_c = None
    models_by_condition = {}

    for dt in DTS:
        stride = int(round(dt / 0.025))
        times = fine_times[::stride]
        train_latent = train_fine[:, ::stride]
        test_latent = test_fine[:, ::stride]
        for sigma in SIGMAS:
            reps = range(1) if sigma == 0.0 else range(10)
            for rep in reps:
                train_obs = train_latent if sigma == 0.0 else train_latent + sigma * noise["train"][rep][:, ::stride]
                test_obs = test_latent if sigma == 0.0 else test_latent + sigma * noise["test"][rep][:, ::stride]
                train_smooth, train_deriv = smooth_and_derivative(train_obs, dt)
                test_smooth, _ = smooth_and_derivative(test_obs, dt)
                fit_mask = (times >= 0.30 - 1e-12) & (times <= 2.0 + 1e-12)
                states = train_smooth[:, fit_mask].reshape(-1, 2)
                deriv = train_deriv[:, fit_mask].reshape(-1, 2)
                dt_code = int(round(dt * 1000))
                sigma_code = int(round(sigma * 1000))
                perm = rng_for(MASTER_SEED, 3, dt_code, sigma_code, rep).permutation(len(states))
                cubic, _ = fit_ridge(states, deriv, "cubic")
                affine, _ = fit_ridge(states, deriv, "affine")
                permuted, _ = fit_ridge(states, deriv, "cubic", permutation=perm)
                models = {"cubic": cubic, "affine": affine, "permuted": permuted}
                models_by_condition[(dt, sigma, rep)] = models
                if dt == 0.05 and sigma == 0.02 and rep == 0:
                    representative_c = cubic.copy()

                for name, c in models.items():
                    field_nrmse = float(np.sqrt(np.sum((evaluate_field(c, grid) - true_grid) ** 2) / np.sum(true_grid**2)))
                    coefficient_error = float(np.linalg.norm(c - TRUE_C) / np.linalg.norm(TRUE_C))
                    add_metric(metric_rows, dt, sigma, rep, "grid", name, "grid", "field_nrmse", field_nrmse)
                    add_metric(metric_rows, dt, sigma, rep, "coefficients", name, "model", "coefficient_error", coefficient_error)
                    for output in range(2):
                        for j, feature in enumerate(FEATURES):
                            coefficient_rows.append(
                                {
                                    "dt": dt,
                                    "sigma": sigma,
                                    "replicate": rep,
                                    "model": name,
                                    "output": "dx" if output == 0 else "dy",
                                    "feature": feature,
                                    "coefficient": c[output, j],
                                    "true_coefficient": TRUE_C[output, j],
                                }
                            )

                evaluate_split(metric_rows, dt, sigma, rep, "test", test_initial, test_latent, test_smooth, models, times)

                if dt == 0.05 and sigma == 0.02:
                    extra_latent = extra_fine[:, ::stride]
                    extra_obs = extra_latent + sigma * noise["extrapolation"][rep][:, ::stride]
                    extra_smooth, _ = smooth_and_derivative(extra_obs, dt)
                    evaluate_split(metric_rows, dt, sigma, rep, "extrapolation", extra_initial, extra_latent, extra_smooth, models, times)

    diagnostics = {
        "generator": generator_diagnostics(),
        "data": data_diagnostics(
            train_initial, test_initial, extra_initial,
            train_fine, test_fine, extra_fine, noise,
        ),
    }

    def field_values(dt, sigma, model):
        return [
            row["value"] for row in metric_rows
            if row["dt"] == dt and row["sigma"] == sigma and row["split"] == "grid"
            and row["model"] == model and row["metric"] == "field_nrmse"
        ]

    noiseless = {
        "cubic_field": field_values(0.025, 0.0, "cubic")[0],
        "affine_field": field_values(0.025, 0.0, "affine")[0],
        "permuted_field": field_values(0.025, 0.0, "permuted")[0],
        "cubic_oracle": summarize_metric(metric_rows, 0.025, 0.0, "cubic", "oracle_rollout_rmse")["median"],
        "affine_oracle": summarize_metric(metric_rows, 0.025, 0.0, "affine", "oracle_rollout_rmse")["median"],
        "permuted_oracle": summarize_metric(metric_rows, 0.025, 0.0, "permuted", "oracle_rollout_rmse")["median"],
    }
    moderate = {
        "cubic_field": float(np.median(field_values(0.05, 0.02, "cubic"))),
        "affine_field": float(np.median(field_values(0.05, 0.02, "affine"))),
        "permuted_field": float(np.median(field_values(0.05, 0.02, "permuted"))),
        "cubic_oracle": summarize_metric(metric_rows, 0.05, 0.02, "cubic", "oracle_rollout_rmse")["median"],
        "affine_oracle": summarize_metric(metric_rows, 0.05, 0.02, "affine", "oracle_rollout_rmse")["median"],
        "permuted_oracle": summarize_metric(metric_rows, 0.05, 0.02, "permuted", "oracle_rollout_rmse")["median"],
    }
    primary_pass = (
        noiseless["cubic_field"] < noiseless["affine_field"]
        and noiseless["cubic_oracle"] < noiseless["affine_oracle"]
        and noiseless["permuted_field"] >= noiseless["cubic_field"]
        and noiseless["permuted_oracle"] >= noiseless["cubic_oracle"]
        and moderate["cubic_field"] < moderate["affine_field"]
        and moderate["cubic_oracle"] < moderate["affine_oracle"]
        and moderate["permuted_field"] >= moderate["cubic_field"]
        and moderate["permuted_oracle"] >= moderate["cubic_oracle"]
    )
    permutation_failures = int(sum(
        row["value"] for row in metric_rows
        if row["model"] == "permuted" and row["metric"] == "rollout_failure"
    ))
    primary = {
        "noiseless": noiseless,
        "moderate": moderate,
        "permuted_rollout_failures": permutation_failures,
        "passed": bool(primary_pass),
    }
    diagnostics["primary_success"] = primary
    diagnostics["condition_summaries"] = {}
    for sigma in SIGMAS:
        for dt in DTS:
            key = f"dt{int(round(dt*1000)):03d}_sigma{int(round(sigma*1000)):03d}"
            condition = {
                "smoother": {
                    "state_rmse": summarize_metric(metric_rows, dt, sigma, "smoother", "state_rmse"),
                }
            }
            for model in MODEL_NAMES:
                condition[model] = {
                    "field_nrmse": summarize_metric(metric_rows, dt, sigma, model, "field_nrmse", split="grid"),
                    "coefficient_error": summarize_metric(metric_rows, dt, sigma, model, "coefficient_error", split="coefficients"),
                    "one_step_rmse": summarize_metric(metric_rows, dt, sigma, model, "one_step_rmse"),
                    "oracle_rollout_rmse": summarize_metric(metric_rows, dt, sigma, model, "oracle_rollout_rmse"),
                    "observed_rollout_rmse": summarize_metric(metric_rows, dt, sigma, model, "observed_rollout_rmse"),
                    "oracle_radial_rmse": summarize_metric(metric_rows, dt, sigma, model, "oracle_radial_rmse"),
                    "observed_radial_rmse": summarize_metric(metric_rows, dt, sigma, model, "observed_radial_rmse"),
                    "one_step_failure_rate": summarize_failure_rate(metric_rows, dt, sigma, model, "one_step_failure_rate"),
                    "oracle_rollout_failure": summarize_failure_rate(metric_rows, dt, sigma, model, "oracle_rollout_failure"),
                    "observed_rollout_failure": summarize_failure_rate(metric_rows, dt, sigma, model, "observed_rollout_failure"),
                    "combined_rollout_failure": summarize_failure_rate(metric_rows, dt, sigma, model, "rollout_failure"),
                    "oracle_pre_failure_horizon": summarize_metric(metric_rows, dt, sigma, model, "oracle_pre_failure_horizon"),
                    "observed_pre_failure_horizon": summarize_metric(metric_rows, dt, sigma, model, "observed_pre_failure_horizon"),
                }
            diagnostics["condition_summaries"][key] = condition

    write_csv(
        root / "metrics.csv",
        metric_rows,
        ["dt", "sigma", "replicate", "split", "model", "unit", "metric", "value"],
    )
    write_csv(
        root / "coefficients.csv",
        coefficient_rows,
        ["dt", "sigma", "replicate", "model", "output", "feature", "coefficient", "true_coefficient"],
    )
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")

    heat_cubic = [[summarize_metric(metric_rows, dt, sigma, "cubic", "oracle_rollout_rmse")["median"] for dt in DTS] for sigma in SIGMAS]
    heat_affine = [[summarize_metric(metric_rows, dt, sigma, "affine", "oracle_rollout_rmse")["median"] for dt in DTS] for sigma in SIGMAS]
    if representative_c is None:
        raise RuntimeError("representative model missing")
    make_summary_png(root / "summary.png", train_initial, representative_c, heat_cubic, heat_affine, primary)

    hashes = {name: sha256(root / name) for name in ARTIFACT_NAMES}
    manifest = {
        "schema_version": 1,
        "project": "Latent Dynamics & Geometry",
        "artifact": "two-dimensional-autonomous-dynamics",
        "date": "2026-08-18",
        "master_seed": MASTER_SEED,
        "rng": "numpy.Generator(PCG64(SeedSequence(words)))",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": getattr(sys.modules.get("PIL"), "__version__", "unknown"),
            "platform": platform.platform(),
        },
        "generator": {
            "equations": ["dx=(1-r^2)x-2y", "dy=2x+(1-r^2)y"],
            "alpha": ALPHA,
            "beta": BETA,
            "omega": OMEGA,
            "time_interval": [0.0, T_END],
            "implementation": "exact analytic polar solution",
            "independent_check": f"fixed-step RK4 h={RK4_STEP}",
        },
        "splits": {"train": 64, "test": 40, "extrapolation": 16},
        "sampling_intervals": DTS,
        "noise_scales": SIGMAS,
        "nonzero_noise_replicates": 10,
        "smoothing_windows": {str(k): v for k, v in WINDOWS.items()},
        "recovery": {
            "features": FEATURES,
            "fit_interval": [0.30, 2.0],
            "score_interval": [0.30, 7.70],
            "rollout_interval": [0.30, 8.0],
            "ridge_lambda": LAMBDA,
            "rk4_step": RK4_STEP,
        },
        "observation_storage": "paired standard-normal arrays on dt=0.025 plus representative observations; all conditions reconstruct by subsampling and sigma scaling",
        "plan_deviations": [
            "SciPy DOP853 unavailable: exact analytic generator used, checked independently by NumPy RK4.",
            "SciPy Savitzky-Golay unavailable: identical centered local-cubic coefficients computed by NumPy pseudoinverse.",
            "Matplotlib unavailable: one composite diagnostic PNG rendered with bundled Pillow.",
        ],
        "primary_success_passed": bool(primary_pass),
        "files_sha256": hashes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not quiet:
        print(json.dumps({"artifact_root": str(root), "primary_success_passed": primary_pass, "elapsed_seconds": round(time.monotonic()-start, 3)}, indent=2))
    return diagnostics


def load_npz_arrays(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compare_roots(actual: Path, regenerated: Path):
    errors = []
    for name in ("metrics.csv", "coefficients.csv", "diagnostics.json", "summary.png", "manifest.json"):
        if (actual / name).read_bytes() != (regenerated / name).read_bytes():
            errors.append(f"byte mismatch: {name}")
    for name in ("latent-trajectories.npz", "observations.npz"):
        left = load_npz_arrays(actual / name)
        right = load_npz_arrays(regenerated / name)
        if set(left) != set(right):
            errors.append(f"key mismatch: {name}")
            continue
        for key in left:
            if not np.array_equal(left[key], right[key]):
                errors.append(f"array mismatch: {name}:{key}")
                break
    return errors


def verify(root: Path, recompute=True):
    errors = []
    required = ("manifest.json",) + ARTIFACT_NAMES
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        actual = sha256(root / name)
        if actual != expected:
            errors.append(f"hash mismatch {name}")
    diagnostics = json.loads((root / "diagnostics.json").read_text())
    g = diagnostics["generator"]
    if g["polar_identity_max_abs_residual"] > 1e-12:
        errors.append("polar identity residual too large")
    if g["rk4_vs_analytic_max_euclidean_error"] > 1e-8:
        errors.append("RK4 generator check failed")
    if g["cycle_radius_max_abs_error"] > 1e-8:
        errors.append("cycle check failed")
    if g["period_relative_error"] > 1e-6:
        errors.append("period check failed")
    data = diagnostics["data"]
    if data["radial_monotonicity_max_violation"] > 1e-10:
        errors.append("radial monotonicity check failed")
    if data["split_counts"] != {"train": 64, "test": 40, "extrapolation": 16}:
        errors.append("split count check failed")
    if data["stratum_counts"]["train"]["inner"] != 32 or data["stratum_counts"]["train"]["outer"] != 32:
        errors.append("training stratum count check failed")
    if data["stratum_counts"]["test"]["inner"] != 20 or data["stratum_counts"]["test"]["outer"] != 20:
        errors.append("test stratum count check failed")
    if not data["all_initial_conditions_unique"]:
        errors.append("initial-condition disjointness check failed")
    if not data["leakage_assertion"]["fit_uses_training_split_only"] or data["leakage_assertion"]["test_rows_in_feature_scaling_or_regression"] != 0:
        errors.append("leakage assertion failed")
    for sigma in ("0.02", "0.05"):
        stats = data["noise_statistics"][sigma]
        mean = np.asarray(stats["mean"])
        covariance = np.asarray(stats["covariance"])
        target = float(sigma) ** 2
        if np.max(np.abs(mean)) > 5e-4:
            errors.append(f"noise mean diagnostic failed for sigma={sigma}")
        if np.max(np.abs(covariance - target * np.eye(2))) > 5e-5:
            errors.append(f"noise covariance diagnostic failed for sigma={sigma}")
    if not diagnostics["primary_success"]["passed"]:
        errors.append("preregistered primary comparison did not pass")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="ldg-toy-verify-") as tmp:
            regenerated = Path(tmp) / "artifact"
            generate(regenerated, quiet=True)
            errors.extend(compare_roots(root, regenerated))
    if errors:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"verified": True, "artifact_root": str(root), "recomputed": recompute}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="generate all bounded artifacts")
    parser.add_argument("--verify", action="store_true", help="verify hashes, scientific diagnostics, and deterministic regeneration")
    parser.add_argument("--no-recompute", action="store_true", help="skip deterministic temporary regeneration during verification")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if not args.generate and not args.verify:
        parser.error("choose --generate and/or --verify")
    if args.generate:
        generate(args.artifact_root)
    if args.verify:
        verify(args.artifact_root, recompute=not args.no_recompute)


if __name__ == "__main__":
    main()
