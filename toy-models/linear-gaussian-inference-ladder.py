#!/usr/bin/env python3
"""Deterministic scalar LGSSM generation/recovery inference ladder.

Only NumPy and Pillow are required. Generation, masking, parameter selection,
inference, scoring, and artifact verification are kept as separate stages.
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
TRUE = {"a": 0.85, "q": 0.10, "r": 0.40}
SPLIT_SIZES = {"train": 12, "validation": 6, "test": 8}
SEQUENCE_LENGTH = 80
MISSING_BLOCK = (30, 45)  # half-open, zero based
SHORT_ORACLE_LENGTH = 24
GRID = tuple(
    (a, q, r)
    for a in (0.75, 0.85, 0.95)
    for q in (0.05, 0.10, 0.20)
    for r in (0.20, 0.40, 0.80)
)
WRONG = {
    "q_too_small_r_too_large": {"a": 0.85, "q": 0.025, "r": 0.80},
    "q_too_large_r_too_small": {"a": 0.85, "q": 0.40, "r": 0.10},
}
NEAR_UNIT = {"a": 0.99, "q": 0.02, "r": 0.40}
NEAR_UNIT_HORIZONS = (1, 10, 50, 100, 200)
TOL = {"oracle": 1e-10, "variance": 1e-12, "covariance": 1e-12, "determinism": 0.0}
ARTIFACT_NAMES = ("metrics.csv", "diagnostics.json", "summary.png")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "linear-gaussian-inference-ladder"
METRIC_FIELDS = (
    "split", "condition", "parameter_source", "method", "inference_target",
    "region", "metric", "value",
)


def rng_for(*codes: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence((MASTER_SEED,) + codes)))


def stationary_variance(a: float, q: float) -> float:
    if not (abs(a) < 1.0 and q >= 0.0):
        raise ValueError("stationary scalar prior requires |a|<1 and q>=0")
    return q / (1.0 - a * a)


def generate_sequences(n: int, length: int, params: dict, split_code: int):
    """Generate independent sequences, resetting z[0] from the declared prior."""
    a, q, r = params["a"], params["q"], params["r"]
    latent = np.empty((n, length), dtype=float)
    observed = np.empty_like(latent)
    p0 = stationary_variance(a, q)
    for i in range(n):
        rng = rng_for(10, split_code, i)
        latent[i, 0] = rng.normal(0.0, math.sqrt(p0))
        observed[i, 0] = latent[i, 0] + rng.normal(0.0, math.sqrt(r))
        for t in range(1, length):
            latent[i, t] = a * latent[i, t - 1] + rng.normal(0.0, math.sqrt(q))
            observed[i, t] = latent[i, t] + rng.normal(0.0, math.sqrt(r))
    return latent, observed


def kalman_filter(x, mask, params, prior_mean=0.0, prior_var=None):
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    a, q, r = params["a"], params["q"], params["r"]
    if prior_var is None:
        prior_var = stationary_variance(a, q)
    n = len(x)
    pm = np.empty(n); pv = np.empty(n); fm = np.empty(n); fv = np.empty(n)
    innovation = np.full(n, np.nan); innovation_var = np.full(n, np.nan)
    nll = np.full(n, np.nan)
    for t in range(n):
        if t == 0:
            pm[t], pv[t] = prior_mean, prior_var
        else:
            pm[t], pv[t] = a * fm[t - 1], a * a * fv[t - 1] + q
        if mask[t]:
            innovation[t] = x[t] - pm[t]
            innovation_var[t] = pv[t] + r
            gain = pv[t] / innovation_var[t]
            fm[t] = pm[t] + gain * innovation[t]
            fv[t] = (1.0 - gain) * pv[t]
            nll[t] = 0.5 * (math.log(2.0 * math.pi * innovation_var[t]) + innovation[t] ** 2 / innovation_var[t])
        else:
            fm[t], fv[t] = pm[t], pv[t]
        if not (np.isfinite(fv[t]) and fv[t] >= -TOL["covariance"]):
            raise FloatingPointError("non-finite or negative filter variance")
    return {"pred_mean": pm, "pred_var": pv, "mean": fm, "var": fv,
            "innovation": innovation, "innovation_var": innovation_var, "nll": nll}


def rts_smoother(filt, params):
    a = params["a"]
    sm = filt["mean"].copy(); sv = filt["var"].copy()
    for t in range(len(sm) - 2, -1, -1):
        gain = filt["var"][t] * a / filt["pred_var"][t + 1]
        sm[t] += gain * (sm[t + 1] - filt["pred_mean"][t + 1])
        sv[t] += gain * gain * (sv[t + 1] - filt["pred_var"][t + 1])
        if not (np.isfinite(sv[t]) and sv[t] >= -TOL["covariance"]):
            raise FloatingPointError("non-finite or negative smoother variance")
    return {"mean": sm, "var": sv}


def prior_covariance(length: int, params: dict, prior_var=None):
    a, q = params["a"], params["q"]
    if prior_var is None:
        prior_var = stationary_variance(a, q)
    cov = np.empty((length, length), dtype=float)
    variances = np.empty(length); variances[0] = prior_var
    for t in range(1, length):
        variances[t] = a * a * variances[t - 1] + q
    for i in range(length):
        for j in range(length):
            lo, hi = min(i, j), max(i, j)
            cov[i, j] = (a ** (hi - lo)) * variances[lo]
    return cov


def batch_condition(x, mask, params, prior_mean=0.0, prior_var=None):
    """Direct joint-Gaussian conditioning; used only as a bounded oracle."""
    x = np.asarray(x, dtype=float); mask = np.asarray(mask, dtype=bool)
    n = len(x); mean0 = np.array([prior_mean * params["a"] ** t for t in range(n)])
    cov0 = prior_covariance(n, params, prior_var)
    obs = np.flatnonzero(mask)
    if len(obs) == 0:
        return {"mean": mean0, "cov": cov0, "var": np.diag(cov0).copy()}
    cross = cov0[:, obs]
    obs_cov = cov0[np.ix_(obs, obs)] + params["r"] * np.eye(len(obs))
    solved_resid = np.linalg.solve(obs_cov, x[obs] - mean0[obs])
    solved_cross = np.linalg.solve(obs_cov, cross.T)
    mean = mean0 + cross @ solved_resid
    cov = cov0 - cross @ solved_cross
    cov = 0.5 * (cov + cov.T)
    return {"mean": mean, "cov": cov, "var": np.diag(cov).copy()}


def innovation_nll(observed, masks, params):
    total = 0.0; count = 0
    for x, mask in zip(observed, masks):
        out = kalman_filter(x, mask, params)
        good = np.isfinite(out["nll"])
        total += float(np.sum(out["nll"][good])); count += int(np.sum(good))
    return total / count


def select_grid(data):
    masks = {s: np.ones_like(data[s][1], dtype=bool) for s in ("train", "validation")}
    scores = []
    for a, q, r in GRID:
        params = {"a": a, "q": q, "r": r}
        train = innovation_nll(data["train"][1], masks["train"], params)
        validation = innovation_nll(data["validation"][1], masks["validation"], params)
        scores.append({"a": a, "q": q, "r": r, "train_nll": train,
                       "validation_nll": validation, "selection_score": train + validation})
    selected = min(scores, key=lambda d: (d["selection_score"], d["a"], d["q"], d["r"]))
    return {k: selected[k] for k in ("a", "q", "r")}, scores


def persistence(x, mask):
    result = np.zeros(len(x)); last = 0.0
    for t in range(len(x)):
        if mask[t]: last = x[t]
        result[t] = last
    return result


def open_loop(x, mask, params):
    """Assimilate the first available observation, then propagate without updates."""
    x = np.asarray(x, dtype=float); mask = np.asarray(mask, dtype=bool)
    result = np.zeros(len(x), dtype=float)
    observed = np.flatnonzero(mask)
    if len(observed) == 0:
        return result
    first = int(observed[0])
    p0 = stationary_variance(params["a"], params["q"])
    gain = p0 / (p0 + params["r"])
    result[first] = gain * x[first]
    for t in range(first + 1, len(x)):
        result[t] = params["a"] * result[t - 1]
    return result


def lag1(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    adjacent = mask[:-1] & mask[1:] & np.isfinite(values[:-1]) & np.isfinite(values[1:])
    left = values[:-1][adjacent]
    right = values[1:][adjacent]
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def add_metric(rows, split, condition, source, method, target, region, metric, value):
    rows.append({"split": split, "condition": condition, "parameter_source": source,
                 "method": method, "inference_target": target, "region": region,
                 "metric": metric, "value": float(value)})


def score_probabilistic(rows, split, condition, source, method, target, latent, result, regions):
    for region, idx in regions.items():
        if not np.any(idx): continue
        err = result["mean"][:, idx] - latent[:, idx]
        var = result["var"][:, idx]
        add_metric(rows, split, condition, source, method, target, region, "latent_rmse", np.sqrt(np.mean(err ** 2)))
        add_metric(rows, split, condition, source, method, target, region, "interval_95_coverage",
                   np.mean(np.abs(err) <= 1.959963984540054 * np.sqrt(var)))
        add_metric(rows, split, condition, source, method, target, region, "mean_posterior_variance", np.mean(var))


def infer_dataset(observed, masks, params):
    filters = []; smoothers = []
    for x, mask in zip(observed, masks):
        f = kalman_filter(x, mask, params); s = rts_smoother(f, params)
        filters.append(f); smoothers.append(s)
    stack = lambda records, key: np.stack([r[key] for r in records])
    return {
        "filter": {k: stack(filters, k) for k in filters[0]},
        "smoother": {k: stack(smoothers, k) for k in smoothers[0]},
    }


def array_discrepancy(left, right):
    """Return finite-value discrepancy plus an exact NaN-pattern check."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    nan_pattern_equal = bool(np.array_equal(np.isnan(left), np.isnan(right)))
    finite = np.isfinite(left) & np.isfinite(right)
    maximum = float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0
    return maximum, nan_pattern_equal


def leakage_reset_diagnostics(data, masks, selected):
    """Computed poison, sequence-reset, and hidden-update diagnostics."""
    selected_baseline, baseline_scores = select_grid(data)
    if selected_baseline != selected:
        raise RuntimeError("selection diagnostic baseline disagrees with build selection")
    poison_test = {split: (latent.copy(), observed.copy()) for split, (latent, observed) in data.items()}
    poison_test["test"] = (
        poison_test["test"][0],
        poison_test["test"][1] + np.linspace(1.0e4, 2.0e4, SEQUENCE_LENGTH)[None, :],
    )
    selected_test_poison, test_poison_scores = select_grid(poison_test)

    poison_latent = {split: (latent.copy(), observed.copy()) for split, (latent, observed) in data.items()}
    for code, split in enumerate(("train", "validation", "test"), 1):
        poison_latent[split] = (
            poison_latent[split][0] + code * 1.0e6,
            poison_latent[split][1],
        )
    selected_latent_poison, latent_poison_scores = select_grid(poison_latent)

    baseline = infer_dataset(data["test"][1], masks["test"], selected)
    latent_poison_recovery = infer_dataset(poison_latent["test"][1], masks["test"], selected_latent_poison)
    recovery_by_key = {}
    for stage in ("filter", "smoother"):
        for key in baseline[stage]:
            maximum, nan_equal = array_discrepancy(baseline[stage][key], latent_poison_recovery[stage][key])
            recovery_by_key[f"{stage}.{key}"] = {
                "max_abs_discrepancy": maximum,
                "nan_pattern_equal": nan_equal,
            }

    independent_by_key = {}
    for i, (x, mask) in enumerate(zip(data["test"][1], masks["test"])):
        direct_filter = kalman_filter(x, mask, selected)
        direct_smoother = rts_smoother(direct_filter, selected)
        for stage, direct in (("filter", direct_filter), ("smoother", direct_smoother)):
            for key, values in direct.items():
                label = f"{stage}.{key}"
                maximum, nan_equal = array_discrepancy(baseline[stage][key][i], values)
                record = independent_by_key.setdefault(
                    label, {"max_abs_discrepancy": 0.0, "nan_pattern_equal": True}
                )
                record["max_abs_discrepancy"] = max(record["max_abs_discrepancy"], maximum)
                record["nan_pattern_equal"] = bool(record["nan_pattern_equal"] and nan_equal)

    hidden_masks = np.ones_like(data["test"][1], dtype=bool)
    hidden_masks[:, slice(*MISSING_BLOCK)] = False
    hidden = infer_dataset(data["test"][1], hidden_masks, TRUE)["filter"]
    hidden_idx = ~hidden_masks
    hidden_mean_max = float(np.max(np.abs(hidden["mean"][hidden_idx] - hidden["pred_mean"][hidden_idx])))
    hidden_var_max = float(np.max(np.abs(hidden["var"][hidden_idx] - hidden["pred_var"][hidden_idx])))
    hidden_absent = {
        key: bool(np.all(np.isnan(hidden[key][hidden_idx])))
        for key in ("innovation", "innovation_var", "nll")
    }

    def score_max(left, right):
        return float(max(
            abs(a["selection_score"] - b["selection_score"])
            for a, b in zip(left, right)
        ))

    return {
        "test_observation_poison": {
            "selected_parameters_unchanged": bool(selected_test_poison == selected),
            "grid_selection_score_max_abs_discrepancy": score_max(baseline_scores, test_poison_scores),
        },
        "latent_poison": {
            "selected_parameters_unchanged": bool(selected_latent_poison == selected),
            "grid_selection_score_max_abs_discrepancy": score_max(baseline_scores, latent_poison_scores),
            "recovery_by_key": recovery_by_key,
            "recovery_max_abs_discrepancy": max(v["max_abs_discrepancy"] for v in recovery_by_key.values()),
            "recovery_nan_patterns_all_equal": bool(all(v["nan_pattern_equal"] for v in recovery_by_key.values())),
        },
        "independent_sequence_invocation": {
            "by_key": independent_by_key,
            "max_abs_discrepancy": max(v["max_abs_discrepancy"] for v in independent_by_key.values()),
            "nan_patterns_all_equal": bool(all(v["nan_pattern_equal"] for v in independent_by_key.values())),
        },
        "hidden_update": {
            "hidden_time_count": int(np.sum(hidden_idx)),
            "mean_equals_prediction_max_abs": hidden_mean_max,
            "variance_equals_prediction_max_abs": hidden_var_max,
            "innovation_fields_absent": hidden_absent,
            "all_innovation_fields_absent": bool(all(hidden_absent.values())),
        },
    }


def evaluate_condition(rows, split, condition, source, latent, observed, masks, params):
    inferred = infer_dataset(observed, masks, params)
    inside = np.zeros(observed.shape[1], dtype=bool); inside[slice(*MISSING_BLOCK)] = True
    regions = {"all": np.ones(observed.shape[1], dtype=bool)}
    if condition == "missing_block": regions.update({"inside_missing_block": inside, "outside_missing_block": ~inside})
    score_probabilistic(rows, split, condition, source, "kalman_filter", "online_state", latent, inferred["filter"], regions)
    score_probabilistic(rows, split, condition, source, "rts_smoother", "retrospective_state", latent, inferred["smoother"], regions)
    for method, estimates in (
        ("stationary_mean", np.zeros_like(latent)),
        ("persistence", np.stack([persistence(x, m) for x, m in zip(observed, masks)])),
        ("open_loop", np.stack([open_loop(x, m, params) for x, m in zip(observed, masks)])),
    ):
        for region, idx in regions.items():
            add_metric(rows, split, condition, source if method == "open_loop" else "baseline", method,
                       "point_state", region, "latent_rmse", np.sqrt(np.mean((estimates[:, idx] - latent[:, idx]) ** 2)))
    innovations = inferred["filter"]["innovation"]
    standardized = innovations / np.sqrt(inferred["filter"]["innovation_var"])
    good = np.isfinite(standardized)
    add_metric(rows, split, condition, source, "kalman_filter", "one_step_observation", "observed",
               "predictive_nll", np.nanmean(inferred["filter"]["nll"]))
    add_metric(rows, split, condition, source, "kalman_filter", "innovation", "observed", "innovation_mean", np.mean(standardized[good]))
    add_metric(rows, split, condition, source, "kalman_filter", "innovation", "observed", "innovation_variance", np.var(standardized[good]))
    per_seq_lag = [lag1(v, m) for v, m in zip(standardized, masks)]
    add_metric(rows, split, condition, source, "kalman_filter", "innovation", "observed", "innovation_lag1", np.nanmean(per_seq_lag))
    return inferred


def oracle_diagnostics(x, params):
    x = x[:SHORT_ORACLE_LENGTH]; full_mask = np.ones(len(x), dtype=bool)
    filt = kalman_filter(x, full_mask, params); smooth = rts_smoother(filt, params)
    batch = batch_condition(x, full_mask, params)
    prefix_mean = []; prefix_var = []
    for t in range(len(x)):
        b = batch_condition(x[:t + 1], np.ones(t + 1, dtype=bool), params)
        prefix_mean.append(b["mean"][-1]); prefix_var.append(b["var"][-1])
    prefix_length = 13
    future_changed = x.copy()
    future_changed[prefix_length:] += np.linspace(5.0, 9.0, len(x) - prefix_length)
    original_prefix = kalman_filter(x, full_mask, params)
    changed_prefix = kalman_filter(future_changed, full_mask, params)
    causal_by_key = {}
    for key in original_prefix:
        maximum, nan_equal = array_discrepancy(
            original_prefix[key][:prefix_length], changed_prefix[key][:prefix_length]
        )
        causal_by_key[key] = {
            "max_abs_discrepancy": maximum,
            "nan_pattern_equal": nan_equal,
            "original_nan_count": int(np.sum(np.isnan(original_prefix[key][:prefix_length]))),
            "perturbed_nan_count": int(np.sum(np.isnan(changed_prefix[key][:prefix_length]))),
        }
    sym = float(np.max(np.abs(batch["cov"] - batch["cov"].T)))
    eigmin = float(np.min(np.linalg.eigvalsh(batch["cov"])))
    return {
        "filter_batch_mean_max_abs": float(np.max(np.abs(filt["mean"] - prefix_mean))),
        "filter_batch_variance_max_abs": float(np.max(np.abs(filt["var"] - prefix_var))),
        "rts_batch_mean_max_abs": float(np.max(np.abs(smooth["mean"] - batch["mean"]))),
        "rts_batch_variance_max_abs": float(np.max(np.abs(smooth["var"] - batch["var"]))),
        "smoother_minus_filter_variance_max": float(np.max(smooth["var"] - filt["var"])),
        "causal_prefix_length": prefix_length,
        "causal_prefix_by_key": causal_by_key,
        "causal_prefix_change_max_abs": max(v["max_abs_discrepancy"] for v in causal_by_key.values()),
        "causal_prefix_nan_patterns_all_equal": bool(all(v["nan_pattern_equal"] for v in causal_by_key.values())),
        "batch_covariance_symmetry_max_abs": sym,
        "batch_covariance_min_eigenvalue": eigmin,
        "all_covariances_finite": bool(np.all(np.isfinite(batch["cov"])) and np.all(np.isfinite(filt["var"])) and np.all(np.isfinite(smooth["var"]))),
    }


def reset_boundary_diagnostic(latent, observed, params):
    proper_second = kalman_filter(observed[1], np.ones(SEQUENCE_LENGTH, bool), params)
    joined_x = np.concatenate([observed[0], observed[1]])
    wrong = kalman_filter(joined_x, np.ones(len(joined_x), bool), params)
    wrong_second = wrong["mean"][SEQUENCE_LENGTH:]
    k = 8
    return {
        "boundary_prediction_from_previous_sequence": float(params["a"] * kalman_filter(observed[0], np.ones(SEQUENCE_LENGTH, bool), params)["mean"][-1]),
        "correct_reset_prior_mean": 0.0,
        "first_state_estimate_abs_difference": float(abs(wrong_second[0] - proper_second["mean"][0])),
        "first_8_wrong_rmse": float(np.sqrt(np.mean((wrong_second[:k] - latent[1, :k]) ** 2))),
        "first_8_proper_rmse": float(np.sqrt(np.mean((proper_second["mean"][:k] - latent[1, :k]) ** 2))),
        "artificial_continuity_abs": float(abs(wrong["pred_mean"][SEQUENCE_LENGTH] - 0.0)),
    }


def near_unit_diagnostic():
    latent, observed = generate_sequences(4, 160, NEAR_UNIT, 40)
    mask = np.ones_like(observed, bool)
    stationary = infer_dataset(observed, mask, NEAR_UNIT)
    tight = [kalman_filter(x, m, NEAR_UNIT, prior_var=0.05) for x, m in zip(observed, mask)]
    tight_mean = np.stack([x["mean"] for x in tight]); tight_var = np.stack([x["var"] for x in tight])
    stationary_p0 = stationary_variance(NEAR_UNIT["a"], NEAR_UNIT["q"])
    open_loop_variances = {}
    for label, initial_variance in (("stationary", stationary_p0), ("tight", 0.05)):
        values = {}
        for horizon in NEAR_UNIT_HORIZONS:
            variance = initial_variance
            for _ in range(horizon):
                variance = NEAR_UNIT["a"] ** 2 * variance + NEAR_UNIT["q"]
            values[str(horizon)] = float(variance)
        open_loop_variances[label] = values
    open_loop_values = [v for values in open_loop_variances.values() for v in values.values()]
    return {
        "parameters": NEAR_UNIT,
        "stationary_prior_variance": stationary_p0,
        "max_filter_variance": float(np.max(stationary["filter"]["var"])),
        "max_smoother_variance": float(np.max(stationary["smoother"]["var"])),
        "tight_vs_stationary_first_20_mean_max_abs": float(np.max(np.abs(tight_mean[:, :20] - stationary["filter"]["mean"][:, :20]))),
        "tight_vs_stationary_first_20_variance_max_abs": float(np.max(np.abs(tight_var[:, :20] - stationary["filter"]["var"][:, :20]))),
        "open_loop_predictive_variance": open_loop_variances,
        "open_loop_horizons": list(NEAR_UNIT_HORIZONS),
        "open_loop_all_finite_nonnegative": bool(np.all(np.isfinite(open_loop_values)) and min(open_loop_values) >= -TOL["covariance"]),
        "all_finite_nonnegative": bool(np.all(np.isfinite(tight_var)) and np.min(tight_var) >= -TOL["covariance"]),
    }


def informative_missingness_diagnostic(latent, observed):
    threshold = 0.9
    mask = np.abs(observed) <= threshold
    inferred = infer_dataset(observed, mask, TRUE)
    missing = ~mask
    return {
        "mechanism": "observation omitted when abs(x_t)>0.9; missingness is therefore non-ignorable under the ordinary observation model",
        "threshold": threshold,
        "missing_fraction": float(np.mean(missing)),
        "mean_abs_latent_when_missing": float(np.mean(np.abs(latent[missing]))),
        "mean_abs_latent_when_observed": float(np.mean(np.abs(latent[mask]))),
        "skipped_update_filter_rmse": float(np.sqrt(np.mean((inferred["filter"]["mean"] - latent) ** 2))),
        "warning": "Skipped Kalman updates condition only on observed x values and ignore information carried by this state/observation-dependent mask; the result is a diagnostic, not a correction.",
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row); out["value"] = format(out["value"], ".17g")
            writer.writerow(out)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(v):
    return f"{v:.4g}"


def make_summary(path, latent, observed, inferred, diagnostics, rows):
    width, height = 1500, 980
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image); font = ImageFont.load_default()
    draw.text((25, 15), "LDG scalar LGSSM inference ladder — deterministic diagnostic", fill="black", font=font)
    def panel(box, title):
        draw.rectangle(box, outline="black", width=2); draw.text((box[0]+8, box[1]+6), title, fill="black", font=font)
    panel((25, 45, 1020, 520), "A. Test sequence 0: missing block, online filter vs retrospective smoother")
    x0,y0,x1,y1=45,80,1000,500
    vals=np.concatenate([latent[0], observed[0], inferred["filter"]["mean"][0], inferred["smoother"]["mean"][0]])
    lo,hi=float(vals.min()-0.5),float(vals.max()+0.5)
    px=lambda t: x0+t/(SEQUENCE_LENGTH-1)*(x1-x0)
    py=lambda v: y1-(v-lo)/(hi-lo)*(y1-y0)
    mb0,mb1=MISSING_BLOCK; draw.rectangle((px(mb0),y0,px(mb1-1),y1),fill=(238,238,238))
    for arr,c in ((latent[0],(0,0,0)),(inferred["filter"]["mean"][0],(31,119,180)),(inferred["smoother"]["mean"][0],(214,39,40))):
        draw.line([(px(t),py(v)) for t,v in enumerate(arr)],fill=c,width=2)
    observation_color=(160,160,160)
    draw.line([(px(t),py(observed[0,t])) for t in range(0,mb0)],fill=observation_color,width=2)
    draw.line([(px(t),py(observed[0,t])) for t in range(mb1,SEQUENCE_LENGTH)],fill=observation_color,width=2)
    for t in range(SEQUENCE_LENGTH):
        sd=1.96*math.sqrt(inferred["filter"]["var"][0,t]); draw.line((px(t),py(inferred["filter"]["mean"][0,t]-sd),px(t),py(inferred["filter"]["mean"][0,t]+sd)),fill=(170,210,240))
    draw.text((55,470),"black truth; gray observations (not drawn in block); blue filter; red smoother",fill="black",font=font)
    draw.text((55,487),"shaded = withheld block; intervals condition on fixed parameters",fill="black",font=font)
    panel((1040,45,1475,520),"B. Numerical and leakage oracles")
    oracle=diagnostics["oracles"]
    lines=[f"filter vs prefix batch mean: {fmt(oracle['filter_batch_mean_max_abs'])}",f"filter vs prefix batch var: {fmt(oracle['filter_batch_variance_max_abs'])}",f"RTS vs full batch mean: {fmt(oracle['rts_batch_mean_max_abs'])}",f"RTS vs full batch var: {fmt(oracle['rts_batch_variance_max_abs'])}",f"max smoother-filter var: {fmt(oracle['smoother_minus_filter_variance_max'])}",f"causal prefix change: {fmt(oracle['causal_prefix_change_max_abs'])}",f"selected grid: {diagnostics['selection']['selected']}","Selection uses train+validation observations only.","Smoothing is retrospective, not online."]
    for i,line in enumerate(lines): draw.text((1052,82+i*38),line,fill="black",font=font)
    panel((25,545,760,950),"C. Test metrics by parameter condition")
    wanted={("correct_model","kalman_filter","latent_rmse"),("correct_model","rts_smoother","latent_rmse"),("q_too_small_r_too_large","kalman_filter","latent_rmse"),("q_too_large_r_too_small","kalman_filter","latent_rmse"),("learned_grid","kalman_filter","latent_rmse"),("correct_model","kalman_filter","predictive_nll")}
    shown=[]
    for r in rows:
        key=(r["condition"],r["method"],r["metric"])
        if r["split"]=="test" and r["region"] in ("all","observed") and key in wanted: shown.append(r)
    for i,r in enumerate(shown): draw.text((42,580+i*42),f"{r['condition']} | {r['method']} | {r['metric']}: {fmt(r['value'])}",fill="black",font=font)
    panel((785,545,1475,950),"D. Failure diagnostics and interpretation limit")
    reset=diagnostics["reset_boundary_negative_control"]; near=diagnostics["near_unit_root_stress"]; info=diagnostics["informative_missingness"]
    lines=[f"reset artificial continuity: {fmt(reset['artificial_continuity_abs'])}",f"reset proper/wrong first-8 RMSE: {fmt(reset['first_8_proper_rmse'])} / {fmt(reset['first_8_wrong_rmse'])}",f"near-unit stationary prior variance: {fmt(near['stationary_prior_variance'])}",f"near-unit init sensitivity: {fmt(near['tight_vs_stationary_first_20_mean_max_abs'])}",f"informative missing fraction: {fmt(info['missing_fraction'])}",f"|latent| missing/observed: {fmt(info['mean_abs_latent_when_missing'])} / {fmt(info['mean_abs_latent_when_observed'])}","Ordinary skipped updates do not model informative missingness.","Low RMSE or a smooth path is not mechanistic evidence."]
    for i,line in enumerate(lines): draw.text((800,580+i*42),line,fill="black",font=font)
    image.save(path,format="PNG",optimize=False)


def build(root: Path, quiet=False):
    root.mkdir(parents=True, exist_ok=True)
    data = {}
    for code, split in enumerate(("train", "validation", "test"), 1):
        data[split] = generate_sequences(SPLIT_SIZES[split], SEQUENCE_LENGTH, TRUE, code)
    selected, grid_scores = select_grid(data)
    rows=[]; masks_complete={s:np.ones_like(data[s][1],bool) for s in data}
    leakage_resets = leakage_reset_diagnostics(data, masks_complete, selected)
    known_results={}
    for split,(latent,observed) in data.items():
        known_results[(split,"correct_model")]=evaluate_condition(rows,split,"correct_model","known_true",latent,observed,masks_complete[split],TRUE)
        evaluate_condition(rows,split,"learned_grid","selected_train_validation",latent,observed,masks_complete[split],selected)
        for name,params in WRONG.items(): evaluate_condition(rows,split,name,"predeclared_wrong",latent,observed,masks_complete[split],params)
    test_latent,test_observed=data["test"]
    missing_masks=np.ones_like(test_observed,bool); missing_masks[:,slice(*MISSING_BLOCK)]=False
    missing_inferred=evaluate_condition(rows,"test","missing_block","known_true",test_latent,test_observed,missing_masks,TRUE)
    oracle=oracle_diagnostics(test_observed[0],TRUE)
    for metric,value in oracle.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool): add_metric(rows,"test","correct_model","known_true","oracle_check","posterior_marginal","short_sequence",metric,value)
    reset=reset_boundary_diagnostic(test_latent,test_observed,TRUE)
    for metric,value in reset.items(): add_metric(rows,"test","reset_boundary_negative_control","intentionally_wrong_continuation","kalman_filter","online_state","boundary",metric,value)
    near=near_unit_diagnostic()
    for metric in ("stationary_prior_variance","max_filter_variance","max_smoother_variance","tight_vs_stationary_first_20_mean_max_abs","tight_vs_stationary_first_20_variance_max_abs"):
        add_metric(rows,"stress","near_unit_root","known_stress","kalman_filter" if "smoother" not in metric else "rts_smoother","numerical_stress","all",metric,near[metric])
    for prior_label, values in near["open_loop_predictive_variance"].items():
        for horizon, value in values.items():
            add_metric(rows,"stress","near_unit_root",f"known_stress_{prior_label}_prior","open_loop","predictive_variance",f"horizon_{horizon}","open_loop_predictive_variance",value)
    informative=informative_missingness_diagnostic(test_latent,test_observed)
    for metric in ("missing_fraction","mean_abs_latent_when_missing","mean_abs_latent_when_observed","skipped_update_filter_rmse"):
        add_metric(rows,"test","informative_missingness","known_true_unmodeled_mask","kalman_filter","diagnostic_only","all",metric,informative[metric])
    def lookup(condition,method,metric):
        return next(r["value"] for r in rows if r["split"]=="test" and r["condition"]==condition and r["method"]==method and r["region"]=="all" and r["metric"]==metric)
    gaps={
        "filter_latent_rmse": lookup("learned_grid","kalman_filter","latent_rmse")-lookup("correct_model","kalman_filter","latent_rmse"),
        "smoother_latent_rmse": lookup("learned_grid","rts_smoother","latent_rmse")-lookup("correct_model","rts_smoother","latent_rmse"),
        "filter_interval_95_coverage": lookup("learned_grid","kalman_filter","interval_95_coverage")-lookup("correct_model","kalman_filter","interval_95_coverage"),
        "smoother_interval_95_coverage": lookup("learned_grid","rts_smoother","interval_95_coverage")-lookup("correct_model","rts_smoother","interval_95_coverage"),
        "predictive_nll": next(r["value"] for r in rows if r["split"]=="test" and r["condition"]=="learned_grid" and r["metric"]=="predictive_nll")-next(r["value"] for r in rows if r["split"]=="test" and r["condition"]=="correct_model" and r["metric"]=="predictive_nll"),
    }
    for metric,value in gaps.items(): add_metric(rows,"test","learned_vs_known_gap","selected_train_validation_minus_known","comparison","mixed", "all",metric,value)
    diagnostics={
        "selection":{"criterion":"minimum mean train innovation NLL + mean validation innovation NLL; deterministic lexicographic tie break","selected":selected,"grid_scores":grid_scores},
        "oracles":oracle,
        "leakage_and_resets":leakage_resets,
        "reset_boundary_negative_control":reset,
        "near_unit_root_stress":near,
        "informative_missingness":informative,
        "learned_vs_known_gap":gaps,
        "missing_block":{"start_inclusive":MISSING_BLOCK[0],"end_exclusive":MISSING_BLOCK[1],"filter_information":"observations through current time only; prediction-only inside block","smoother_information":"all nonmissing observations in sequence; retrospective, not online"},
        "scientific_limits":["Finite-grid selection is a controlled baseline, not general maximum-likelihood estimation.","Posterior intervals are conditional on fixed parameters and omit parameter uncertainty.","Smoother gains from future observations and is not an online competitor to filtering.","Smooth recovery is not evidence of mechanism, causality, or unique latent structure."],
    }
    write_csv(root/"metrics.csv",rows)
    (root/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    make_summary(root/"summary.png",test_latent,test_observed,missing_inferred,diagnostics,rows)
    hashes={name:sha256(root/name) for name in ARTIFACT_NAMES}
    manifest={
        "schema_version":1,"project":"Latent Dynamics & Geometry","artifact":"linear-gaussian-inference-ladder","date":"2026-08-19","master_seed":MASTER_SEED,
        "rng":"numpy.Generator(PCG64(SeedSequence((master_seed, stage, split, sequence))))",
        "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"pillow":getattr(sys.modules.get("PIL"),"__version__","unknown"),"platform":platform.platform()},
        "equations":["z[t+1] = a*z[t] + w[t], w[t] ~ Normal(0,q)","x[t] = z[t] + v[t], v[t] ~ Normal(0,r)","z[0] ~ Normal(0,q/(1-a^2))"],
        "true_parameters":TRUE,"split_sizes":SPLIT_SIZES,"sequence_length":SEQUENCE_LENGTH,"explicit_reset_per_sequence":bool(leakage_resets["independent_sequence_invocation"]["max_abs_discrepancy"] <= TOL["oracle"]),
        "missing_block":{"start_inclusive":MISSING_BLOCK[0],"end_exclusive":MISSING_BLOCK[1]},"wrong_parameter_conditions":WRONG,"near_unit_root_stress":NEAR_UNIT,
        "parameter_grid":[{"a":a,"q":q,"r":r} for a,q,r in GRID],"selection":{"data":"train and validation observations only","criterion":diagnostics["selection"]["criterion"],"selected":selected},
        "methods":["stationary_mean","persistence","open_loop","known_parameter_kalman_filter","known_parameter_rts_smoother","direct_batch_gaussian_conditioning","finite_grid_selected_filter_and_smoother"],
        "tolerances":TOL,"commands":{"python_executable":sys.executable,"generate":f"{sys.executable} toy-models/linear-gaussian-inference-ladder.py --generate","verify":f"{sys.executable} toy-models/linear-gaussian-inference-ladder.py --verify"},
        "artifact_files":["manifest.json",*ARTIFACT_NAMES],"files_sha256":hashes,
    }
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    if not quiet: print(json.dumps({"artifact_root":str(root),"selected_parameters":selected,"oracle_max":max(oracle[k] for k in ("filter_batch_mean_max_abs","filter_batch_variance_max_abs","rts_batch_mean_max_abs","rts_batch_variance_max_abs")),"metrics_rows":len(rows)},indent=2))
    return diagnostics


def compare_roots(left,right):
    return [name for name in ("manifest.json",*ARTIFACT_NAMES) if (left/name).read_bytes()!=(right/name).read_bytes()]


def verify(root: Path, recompute=True):
    errors=[]
    for name in ("manifest.json",*ARTIFACT_NAMES):
        if not (root/name).is_file(): errors.append(f"missing {name}")
    if errors: raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    manifest=json.loads((root/"manifest.json").read_text()); diagnostics=json.loads((root/"diagnostics.json").read_text())
    for name,expected in manifest["files_sha256"].items():
        if sha256(root/name)!=expected: errors.append(f"hash mismatch: {name}")
    o=diagnostics["oracles"]
    for key in ("filter_batch_mean_max_abs","filter_batch_variance_max_abs","rts_batch_mean_max_abs","rts_batch_variance_max_abs"):
        if o[key]>TOL["oracle"]: errors.append(f"oracle tolerance failed: {key}={o[key]}")
    if o["smoother_minus_filter_variance_max"]>TOL["variance"]: errors.append("smoother variance exceeds filter variance")
    if o["causal_prefix_change_max_abs"]>TOL["oracle"]: errors.append("causal-prefix invariance failed")
    if not o["causal_prefix_nan_patterns_all_equal"]: errors.append("causal-prefix NaN patterns differ")
    for key, record in o["causal_prefix_by_key"].items():
        if record["max_abs_discrepancy"]>TOL["oracle"]: errors.append(f"causal-prefix output changed: {key}")
        if not record["nan_pattern_equal"] or record["original_nan_count"] != record["perturbed_nan_count"]:
            errors.append(f"causal-prefix NaN mismatch: {key}")
    if o["batch_covariance_symmetry_max_abs"]>TOL["covariance"] or o["batch_covariance_min_eigenvalue"] < -TOL["covariance"]: errors.append("batch covariance validity failed")
    if not o["all_covariances_finite"]: errors.append("non-finite covariance")
    leak=diagnostics["leakage_and_resets"]
    test_poison=leak["test_observation_poison"]
    if not test_poison["selected_parameters_unchanged"] or test_poison["grid_selection_score_max_abs_discrepancy"]>TOL["oracle"]:
        errors.append("test-observation poison changed train/validation selection")
    latent_poison=leak["latent_poison"]
    if not latent_poison["selected_parameters_unchanged"] or latent_poison["grid_selection_score_max_abs_discrepancy"]>TOL["oracle"]:
        errors.append("latent poison changed grid selection")
    if latent_poison["recovery_max_abs_discrepancy"]>TOL["oracle"] or not latent_poison["recovery_nan_patterns_all_equal"]:
        errors.append("latent poison changed recovery outputs")
    for key, record in latent_poison["recovery_by_key"].items():
        if record["max_abs_discrepancy"]>TOL["oracle"] or not record["nan_pattern_equal"]:
            errors.append(f"latent poison recovery mismatch: {key}")
    independent=leak["independent_sequence_invocation"]
    if independent["max_abs_discrepancy"]>TOL["oracle"] or not independent["nan_patterns_all_equal"]:
        errors.append("multi-sequence inference differs from independent invocation")
    for key, record in independent["by_key"].items():
        if record["max_abs_discrepancy"]>TOL["oracle"] or not record["nan_pattern_equal"]:
            errors.append(f"independent sequence mismatch: {key}")
    hidden=leak["hidden_update"]
    if hidden["mean_equals_prediction_max_abs"]>TOL["oracle"] or hidden["variance_equals_prediction_max_abs"]>TOL["variance"]:
        errors.append("hidden-time filter update differs from prediction")
    if not hidden["all_innovation_fields_absent"] or not all(hidden["innovation_fields_absent"].values()):
        errors.append("hidden-time innovation fields are not absent")
    near=diagnostics["near_unit_root_stress"]
    if not near["all_finite_nonnegative"] or not near["open_loop_all_finite_nonnegative"]:
        errors.append("near-unit stress numerical check failed")
    expected_horizons={str(h) for h in NEAR_UNIT_HORIZONS}
    for prior_label in ("stationary","tight"):
        values=near["open_loop_predictive_variance"].get(prior_label,{})
        if set(values)!=expected_horizons or any((not np.isfinite(v)) or v < -TOL["covariance"] for v in values.values()):
            errors.append(f"near-unit open-loop variance invalid: {prior_label}")
    if manifest.get("tolerances") != TOL:
        errors.append("manifest tolerances do not match executable tolerances")
    metric_rows=list(csv.DictReader((root/"metrics.csv").open(encoding="utf-8")))
    required_gap_metrics={"filter_interval_95_coverage","smoother_interval_95_coverage"}
    present_gap_metrics={r["metric"] for r in metric_rows if r["condition"]=="learned_vs_known_gap"}
    if not required_gap_metrics.issubset(present_gap_metrics):
        errors.append("learned-versus-known coverage-gap metrics missing")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="ldg-lgssm-verify-") as tmp:
            regenerated=Path(tmp)/"artifact"; build(regenerated,quiet=True)
            errors.extend(f"byte mismatch: {name}" for name in compare_roots(root,regenerated))
    if errors: raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    print(json.dumps({"verified":True,"artifact_root":str(root),"deterministic_recomputation":recompute},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true",help="generate the four bounded artifacts")
    parser.add_argument("--verify",action="store_true",help="verify hashes, numerical oracles, leakage/reset assertions, and deterministic regeneration")
    parser.add_argument("--no-recompute",action="store_true",help="skip temporary byte-identical regeneration")
    parser.add_argument("--artifact-root",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    generate=args.generate; check=args.verify
    if not generate and not check: generate=check=True
    if generate: build(args.artifact_root)
    if check: verify(args.artifact_root,recompute=not args.no_recompute)


if __name__ == "__main__":
    main()
