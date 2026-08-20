#!/usr/bin/env python3
"""P4-U01 deterministic scalar Linear-Gaussian robustness benchmark.

The benchmark wraps the stable Phase 3 scalar LGSSM routines. Generation,
finite-grid selection, inference, and scoring are separate stages. Only NumPy
and Pillow are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASTER_SEED = 20260820
REPLICATES = 5
SPLIT_SIZES = {"train": 2, "validation": 2, "test": 2}
GRID = tuple(
    (a, q, r)
    for a in (0.60, 0.85, 0.97)
    for q in (0.025, 0.10, 0.40)
    for r in (0.10, 0.40, 0.80)
)
TOL = {"oracle": 2e-9, "variance": 1e-11, "ordering": 0.0}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "linear-gaussian-robustness"
PHASE3_PATH = REPO_ROOT / "toy-models" / "linear-gaussian-inference-ladder.py"
ARTIFACT_NAMES = ("results.csv", "diagnostics.json", "summary.png")
STATUS_VALUES = ("ok", "nonconvergence", "invalid", "inapplicable")

SCENARIOS = (
    {"id": "stable_low_a", "class": "transition_stability", "a": 0.60, "q": 0.10, "r": 0.40, "length": 60},
    {"id": "stable_moderate", "class": "transition_stability", "a": 0.85, "q": 0.10, "r": 0.40, "length": 60},
    {"id": "stable_high_a", "class": "transition_stability", "a": 0.97, "q": 0.10, "r": 0.40, "length": 60},
    {"id": "process_dominant", "class": "noise_ratio", "a": 0.85, "q": 0.40, "r": 0.10, "length": 60},
    {"id": "observation_dominant", "class": "noise_ratio", "a": 0.85, "q": 0.025, "r": 0.80, "length": 60},
    {"id": "short_sequence", "class": "sequence_length", "a": 0.85, "q": 0.10, "r": 0.40, "length": 24},
    {"id": "long_sequence", "class": "sequence_length", "a": 0.85, "q": 0.10, "r": 0.40, "length": 120},
    {"id": "missing_block", "class": "missing_blocks", "a": 0.85, "q": 0.10, "r": 0.40, "length": 80, "missing": [28, 48]},
    {"id": "wrong_parameters", "class": "wrong_parameters", "a": 0.85, "q": 0.10, "r": 0.40, "length": 60,
     "wrong": {"a": 0.60, "q": 0.025, "r": 0.80}},
    {"id": "reset_boundary", "class": "reset_boundary_failure", "a": 0.85, "q": 0.10, "r": 0.40, "length": 60},
    {"id": "near_unit_root", "class": "near_unit_root_stress", "a": 0.99, "q": 0.02, "r": 0.40, "length": 120},
)

METHODS = (
    {"id": "stationary_mean", "family": "stationary_mean", "information_set": "fixed_reference", "parameter_source": "none"},
    {"id": "persistence", "family": "persistence_open_loop", "information_set": "online_observation_history", "parameter_source": "none"},
    {"id": "open_loop", "family": "persistence_open_loop", "information_set": "online_first_observation", "parameter_source": "known_true"},
    {"id": "known_kalman_filter", "family": "known_parameter_kalman", "information_set": "online_current_and_past", "parameter_source": "known_true"},
    {"id": "known_rts_smoother", "family": "fixed_interval_rts", "information_set": "retrospective_full_interval", "parameter_source": "known_true"},
    {"id": "batch_conditioning_oracle", "family": "direct_batch_conditioning", "information_set": "oracle_retrospective_full_interval", "parameter_source": "known_true"},
    {"id": "selected_kalman_filter", "family": "finite_grid_selected", "information_set": "online_current_and_past", "parameter_source": "selected_train_validation"},
    {"id": "selected_rts_smoother", "family": "finite_grid_selected", "information_set": "retrospective_full_interval", "parameter_source": "selected_train_validation"},
    {"id": "misspecified_kalman_filter", "family": "wrong_parameter_control", "information_set": "online_current_and_past", "parameter_source": "predeclared_wrong"},
    {"id": "misspecified_rts_smoother", "family": "wrong_parameter_control", "information_set": "retrospective_full_interval", "parameter_source": "predeclared_wrong"},
    {"id": "false_continuation_filter", "family": "reset_boundary_negative_control", "information_set": "invalid_cross_sequence_history", "parameter_source": "known_true"},
)

METRICS = (
    "latent_rmse",
    "predictive_nll",
    "interval_95_coverage",
    "mean_posterior_variance",
    "inside_missing_block_rmse",
    "outside_missing_block_rmse",
    "selection_parameter_l1",
    "oracle_mean_discrepancy",
    "reset_first_state_abs_difference",
)

FIELDS = (
    "row_key", "scenario", "scenario_class", "replicate", "split", "method",
    "method_family", "information_set", "parameter_source", "target", "metric",
    "value", "status", "reason", "seed_id", "true_a", "true_q", "true_r",
    "sequence_length", "missing_start", "missing_end", "selected_a", "selected_q",
    "selected_r", "inference_a", "inference_q", "inference_r",
)


def load_phase3():
    spec = importlib.util.spec_from_file_location("ldg_phase3_lgssm", PHASE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load stable Phase 3 implementation: {PHASE3_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P3 = load_phase3()


def scenario_code(scenario_id: str) -> int:
    return int.from_bytes(hashlib.sha256(scenario_id.encode("utf-8")).digest()[:4], "big")


def rng_for(scenario_id: str, replicate: int, split: str, sequence: int) -> np.random.Generator:
    split_code = {"train": 1, "validation": 2, "test": 3}[split]
    seed = np.random.SeedSequence((MASTER_SEED, scenario_code(scenario_id), replicate, split_code, sequence))
    return np.random.Generator(np.random.PCG64(seed))


def seed_id(scenario_id: str, replicate: int) -> str:
    return f"pcg64:{MASTER_SEED}:{scenario_code(scenario_id)}:{replicate}"


def params_for(scenario: dict) -> dict:
    return {k: float(scenario[k]) for k in ("a", "q", "r")}


def generate_split(scenario: dict, replicate: int, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generation stage: independent sequences and a predeclared observation mask."""
    params = params_for(scenario)
    n, length = SPLIT_SIZES[split], int(scenario["length"])
    latent = np.empty((n, length), dtype=float)
    observed = np.empty_like(latent)
    mask = np.ones_like(latent, dtype=bool)
    p0 = P3.stationary_variance(params["a"], params["q"])
    for sequence in range(n):
        rng = rng_for(scenario["id"], replicate, split, sequence)
        latent[sequence, 0] = rng.normal(0.0, math.sqrt(p0))
        observed[sequence, 0] = latent[sequence, 0] + rng.normal(0.0, math.sqrt(params["r"]))
        for t in range(1, length):
            latent[sequence, t] = params["a"] * latent[sequence, t - 1] + rng.normal(0.0, math.sqrt(params["q"]))
            observed[sequence, t] = latent[sequence, t] + rng.normal(0.0, math.sqrt(params["r"]))
    if scenario.get("missing") is not None:
        start, end = scenario["missing"]
        mask[:, start:end] = False
    return latent, observed, mask


def generate_replicate(scenario: dict, replicate: int) -> dict:
    return {split: generate_split(scenario, replicate, split) for split in SPLIT_SIZES}


def selection_score(observations: np.ndarray, masks: np.ndarray, params: dict) -> float:
    total, count = 0.0, 0
    for x, mask in zip(observations, masks):
        out = P3.kalman_filter(x, mask, params)
        good = np.isfinite(out["nll"])
        total += float(np.sum(out["nll"][good]))
        count += int(np.sum(good))
    if count == 0:
        raise ValueError("selection split has no observed values")
    return total / count


def select_parameters(train_observations: np.ndarray, train_masks: np.ndarray,
                      validation_observations: np.ndarray, validation_masks: np.ndarray) -> tuple[dict, list[dict]]:
    """Select from explicit train/validation observations and masks only."""
    records = []
    for a, q, r in GRID:
        params = {"a": a, "q": q, "r": r}
        train = selection_score(train_observations, train_masks, params)
        validation = selection_score(validation_observations, validation_masks, params)
        records.append({"a": a, "q": q, "r": r, "train_nll": train,
                        "validation_nll": validation, "score": train + validation})
    chosen = min(records, key=lambda x: (x["score"], x["a"], x["q"], x["r"]))
    return {k: float(chosen[k]) for k in ("a", "q", "r")}, records


def infer_sequences(observed: np.ndarray, masks: np.ndarray, method: str, params: dict) -> dict:
    """Inference stage: observations, masks and frozen parameters only."""
    if method == "stationary_mean":
        return {"mean": np.zeros_like(observed)}
    if method == "persistence":
        return {"mean": np.stack([P3.persistence(x, m) for x, m in zip(observed, masks)])}
    if method == "open_loop":
        return {"mean": np.stack([P3.open_loop(x, m, params) for x, m in zip(observed, masks)])}
    if method.endswith("kalman_filter"):
        records = [P3.kalman_filter(x, m, params) for x, m in zip(observed, masks)]
        return {key: np.stack([record[key] for record in records]) for key in records[0]}
    if method.endswith("rts_smoother"):
        records = []
        for x, m in zip(observed, masks):
            filt = P3.kalman_filter(x, m, params)
            records.append(P3.rts_smoother(filt, params))
        return {key: np.stack([record[key] for record in records]) for key in records[0]}
    if method == "batch_conditioning_oracle":
        records = [P3.batch_condition(x, m, params) for x, m in zip(observed, masks)]
        return {"mean": np.stack([r["mean"] for r in records]), "var": np.stack([r["var"] for r in records])}
    raise KeyError(method)


def false_continuation(observed: np.ndarray, masks: np.ndarray, params: dict) -> dict:
    joined_x = observed.reshape(-1)
    joined_mask = masks.reshape(-1)
    joined = P3.kalman_filter(joined_x, joined_mask, params)
    length = observed.shape[1]
    return {key: joined[key].reshape(observed.shape) for key in joined}


def status_for_exception(exc: Exception) -> str:
    if isinstance(exc, (FloatingPointError, np.linalg.LinAlgError)):
        return "nonconvergence"
    return "invalid"


def state_target_for(method: str) -> str:
    if method in ("known_kalman_filter", "selected_kalman_filter", "misspecified_kalman_filter", "false_continuation_filter"):
        return "online_state"
    if method in ("known_rts_smoother", "selected_rts_smoother", "misspecified_rts_smoother", "batch_conditioning_oracle"):
        return "retrospective_state"
    return "point_state"


def target_for_metric(method: str, metric: str) -> str:
    if metric == "predictive_nll":
        return "one_step_observation_prediction"
    if metric == "selection_parameter_l1":
        return "selected_parameter_tuple"
    if metric == "reset_first_state_abs_difference":
        return "boundary_reset_diagnostic"
    if metric == "oracle_mean_discrepancy":
        return "numerical_correctness_reference"
    return state_target_for(method)


def row_base(scenario: dict, replicate: int, method_record: dict, selected: dict, inference_params: dict | None) -> dict:
    missing = scenario.get("missing", ["", ""])
    return {
        "scenario": scenario["id"], "scenario_class": scenario["class"], "replicate": replicate,
        "split": "test", "method": method_record["id"], "method_family": method_record["family"],
        "information_set": method_record["information_set"], "parameter_source": method_record["parameter_source"],
        "target": state_target_for(method_record["id"]), "seed_id": seed_id(scenario["id"], replicate),
        "true_a": scenario["a"], "true_q": scenario["q"], "true_r": scenario["r"],
        "sequence_length": scenario["length"], "missing_start": missing[0], "missing_end": missing[1],
        "selected_a": selected["a"], "selected_q": selected["q"], "selected_r": selected["r"],
        "inference_a": "" if inference_params is None else inference_params["a"],
        "inference_q": "" if inference_params is None else inference_params["q"],
        "inference_r": "" if inference_params is None else inference_params["r"],
    }


def add_row(rows: list[dict], base: dict, metric: str, value: float | None, status: str = "ok", reason: str = "") -> None:
    if status not in STATUS_VALUES:
        raise ValueError(status)
    row = dict(base)
    row["target"] = target_for_metric(row["method"], metric)
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
    row["row_key"] = "|".join((row["scenario"], str(row["replicate"]), row["split"], row["method"], metric))
    rows.append(row)


def score_method(rows: list[dict], scenario: dict, replicate: int, method_record: dict, selected: dict,
                 latent: np.ndarray, observed: np.ndarray, masks: np.ndarray, result: dict | None,
                 inference_params: dict | None, exception: Exception | None, known_filter: dict | None,
                 known_smoother: dict | None) -> None:
    """Scoring stage: latent truth is joined only after inference is frozen."""
    method = method_record["id"]
    base = row_base(scenario, replicate, method_record, selected, inference_params)
    active_wrong = scenario["id"] == "wrong_parameters" and method.startswith("misspecified_")
    active_reset = scenario["id"] == "reset_boundary" and method == "false_continuation_filter"
    if method.startswith("misspecified_") and not active_wrong:
        for metric in METRICS:
            add_row(rows, base, metric, None, "inapplicable", "wrong-parameter control is predeclared only for wrong_parameters")
        return
    if method == "false_continuation_filter" and not active_reset:
        for metric in METRICS:
            add_row(rows, base, metric, None, "inapplicable", "false continuation is predeclared only for reset_boundary")
        return
    if exception is not None:
        label = status_for_exception(exception)
        for metric in METRICS:
            add_row(rows, base, metric, None, label, f"{type(exception).__name__}: {exception}")
        return
    if result is None:
        for metric in METRICS:
            add_row(rows, base, metric, None, "invalid", "missing_inference_result")
        return
    mean = np.asarray(result.get("mean"), dtype=float) if "mean" in result else None
    mean_valid = mean is not None and mean.shape == latent.shape and bool(np.all(np.isfinite(mean)))
    err = mean - latent if mean_valid else None
    probabilistic = "var" in result
    variance = np.asarray(result.get("var"), dtype=float) if probabilistic else None
    variance_valid = bool(
        probabilistic and variance is not None and variance.shape == latent.shape
        and np.all(np.isfinite(variance)) and np.all(variance >= 0.0)
    )
    filter_like = method.endswith("kalman_filter") or method == "false_continuation_filter"
    selected_method = method.startswith("selected_")
    missing = scenario.get("missing")
    for metric in METRICS:
        try:
            if metric == "latent_rmse":
                if not mean_valid:
                    add_row(rows, base, metric, None, "invalid", "nonfinite_or_malformed_state_mean")
                else:
                    add_row(rows, base, metric, float(np.sqrt(np.mean(err ** 2))))
            elif metric == "predictive_nll":
                if filter_like and "nll" in result:
                    nll = np.asarray(result["nll"], dtype=float)
                    if nll.shape != masks.shape or not np.all(np.isfinite(nll[masks])):
                        add_row(rows, base, metric, None, "invalid", "nonfinite_or_malformed_observed_time_predictive_nll")
                    else:
                        add_row(rows, base, metric, float(np.mean(nll[masks])))
                else:
                    add_row(rows, base, metric, None, "inapplicable", "method does not return online predictive densities")
            elif metric == "interval_95_coverage":
                if not probabilistic:
                    add_row(rows, base, metric, None, "inapplicable", "point reference has no posterior variance")
                elif not mean_valid:
                    add_row(rows, base, metric, None, "invalid", "nonfinite_or_malformed_state_mean")
                elif not variance_valid:
                    add_row(rows, base, metric, None, "invalid", "nonfinite_negative_or_malformed_posterior_variance")
                else:
                    coverage = np.mean(np.abs(err) <= 1.959963984540054 * np.sqrt(variance))
                    add_row(rows, base, metric, float(coverage))
            elif metric == "mean_posterior_variance":
                if not probabilistic:
                    add_row(rows, base, metric, None, "inapplicable", "point reference has no posterior variance")
                elif not variance_valid:
                    add_row(rows, base, metric, None, "invalid", "nonfinite_negative_or_malformed_posterior_variance")
                else:
                    add_row(rows, base, metric, float(np.mean(variance)))
            elif metric in ("inside_missing_block_rmse", "outside_missing_block_rmse"):
                if missing is None:
                    add_row(rows, base, metric, None, "inapplicable", "scenario has no predeclared missing block")
                elif not mean_valid:
                    add_row(rows, base, metric, None, "invalid", "nonfinite_or_malformed_state_mean")
                else:
                    idx = np.zeros(latent.shape[1], dtype=bool)
                    idx[slice(*missing)] = True
                    if metric.startswith("outside"):
                        idx = ~idx
                    add_row(rows, base, metric, float(np.sqrt(np.mean(err[:, idx] ** 2))))
            elif metric == "selection_parameter_l1":
                if selected_method:
                    truth = params_for(scenario)
                    value = abs(selected["a"] - truth["a"]) + abs(selected["q"] - truth["q"]) + abs(selected["r"] - truth["r"])
                    add_row(rows, base, metric, float(value))
                else:
                    add_row(rows, base, metric, None, "inapplicable", "method does not select parameters")
            elif metric == "oracle_mean_discrepancy":
                if method == "known_kalman_filter" and known_filter is not None:
                    discrepancies = []
                    for i, (x, mask) in enumerate(zip(observed, masks)):
                        prefix = [P3.batch_condition(x[:t + 1], mask[:t + 1], params_for(scenario))["mean"][-1] for t in range(len(x))]
                        discrepancies.append(np.max(np.abs(known_filter["mean"][i] - np.asarray(prefix))))
                    add_row(rows, base, metric, float(max(discrepancies)))
                elif method == "known_rts_smoother" and known_smoother is not None:
                    batch = infer_sequences(observed, masks, "batch_conditioning_oracle", params_for(scenario))
                    add_row(rows, base, metric, float(np.max(np.abs(known_smoother["mean"] - batch["mean"]))))
                elif method == "batch_conditioning_oracle":
                    add_row(rows, base, metric, 0.0)
                else:
                    add_row(rows, base, metric, None, "inapplicable", "metric is reserved for known recursion versus batch oracle")
            elif metric == "reset_first_state_abs_difference":
                if method == "false_continuation_filter" and known_filter is not None and mean_valid:
                    add_row(rows, base, metric, float(abs(mean[1, 0] - known_filter["mean"][1, 0])))
                elif method == "false_continuation_filter":
                    add_row(rows, base, metric, None, "invalid", "nonfinite_or_malformed_boundary_state_mean")
                else:
                    add_row(rows, base, metric, None, "inapplicable", "metric is reserved for reset-boundary negative control")
        except (ArithmeticError, FloatingPointError, IndexError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            add_row(rows, base, metric, None, status_for_exception(exc), f"scoring_exception:{type(exc).__name__}:{exc}")


def evaluate_replicate(scenario: dict, replicate: int) -> tuple[list[dict], dict]:
    data = generate_replicate(scenario, replicate)
    selected, grid_scores = select_parameters(
        data["train"][1], data["train"][2],
        data["validation"][1], data["validation"][2],
    )
    latent, observed, masks = data["test"]
    true_params = params_for(scenario)
    results = {}
    failures = {}
    for method_record in METHODS:
        method = method_record["id"]
        if method.startswith("misspecified_") and scenario["id"] != "wrong_parameters":
            continue
        if method == "false_continuation_filter" and scenario["id"] != "reset_boundary":
            continue
        params = None
        if method in ("open_loop", "known_kalman_filter", "known_rts_smoother", "batch_conditioning_oracle", "false_continuation_filter"):
            params = true_params
        elif method.startswith("selected_"):
            params = selected
        elif method.startswith("misspecified_"):
            params = scenario["wrong"]
        try:
            results[method] = false_continuation(observed, masks, params) if method == "false_continuation_filter" else infer_sequences(observed, masks, method, params or true_params)
        except Exception as exc:  # labels must be retained rather than silently omitted
            failures[method] = exc
    rows = []
    for method_record in METHODS:
        method = method_record["id"]
        params = None
        if method in ("open_loop", "known_kalman_filter", "known_rts_smoother", "batch_conditioning_oracle", "false_continuation_filter"):
            params = true_params
        elif method.startswith("selected_"):
            params = selected
        elif method.startswith("misspecified_") and scenario.get("wrong"):
            params = scenario["wrong"]
        score_method(rows, scenario, replicate, method_record, selected, latent, observed, masks,
                     results.get(method), params, failures.get(method), results.get("known_kalman_filter"),
                     results.get("known_rts_smoother"))
    diagnostics = {
        "scenario": scenario["id"], "replicate": replicate, "selected": selected,
        "best_selection_score": float(min(x["score"] for x in grid_scores)),
    }
    return rows, diagnostics


def canonical_key(row: dict) -> tuple:
    return (row["scenario"], int(row["replicate"]), row["split"], row["method"], row["metric"])


def format_number(value) -> str:
    if value == "" or value is None:
        return ""
    return format(float(value), ".17g")


NUMERIC_RESULT_FIELDS = (
    "value", "true_a", "true_q", "true_r", "selected_a", "selected_q", "selected_r",
    "inference_a", "inference_q", "inference_r",
)


def serialized_row(row: dict) -> dict:
    out = {field: str(row.get(field, "")) for field in FIELDS}
    for field in NUMERIC_RESULT_FIELDS:
        out[field] = format_number(row.get(field, ""))
    out["replicate"] = str(row.get("replicate", ""))
    out["sequence_length"] = str(row.get("sequence_length", ""))
    out["missing_start"] = str(row.get("missing_start", ""))
    out["missing_end"] = str(row.get("missing_end", ""))
    return out


def write_results(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=canonical_key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(serialized_row(row))


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("partial results header does not match benchmark schema")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("partial results header does not match benchmark schema")
    return rows


def validate_result_row(row: dict) -> None:
    if set(row) != set(FIELDS):
        raise ValueError("result row fields do not match benchmark schema")
    if row["status"] not in STATUS_VALUES:
        raise ValueError(f"unknown result status: {row['status']}")
    if row["metric"] not in METRICS:
        raise ValueError(f"unknown result metric: {row['metric']}")
    scenarios = {s["id"]: s for s in SCENARIOS}
    methods = {m["id"]: m for m in METHODS}
    if row["scenario"] not in scenarios or row["method"] not in methods:
        raise ValueError("unknown scenario or method in result row")
    try:
        replicate = int(row["replicate"])
    except ValueError as exc:
        raise ValueError("invalid replicate") from exc
    if not 0 <= replicate < REPLICATES or row["split"] != "test":
        raise ValueError("invalid replicate or split")
    expected_key = "|".join((row["scenario"], row["replicate"], row["split"], row["method"], row["metric"]))
    if row["row_key"] != expected_key:
        raise ValueError("row key does not match row metadata")
    scenario, method = scenarios[row["scenario"]], methods[row["method"]]
    expected_metadata = {
        "scenario_class": scenario["class"], "method_family": method["family"],
        "information_set": method["information_set"], "parameter_source": method["parameter_source"],
        "target": target_for_metric(method["id"], row["metric"]),
        "seed_id": seed_id(scenario["id"], replicate), "sequence_length": str(scenario["length"]),
    }
    for field, expected in expected_metadata.items():
        if row[field] != str(expected):
            raise ValueError(f"stale or corrupted metadata: {field}")
    if row["status"] == "ok":
        if row["reason"]:
            raise ValueError("ok row must not carry a failure reason")
        try:
            value = float(row["value"])
        except ValueError as exc:
            raise ValueError("ok row has nonnumeric value") from exc
        if not np.isfinite(value):
            raise ValueError("ok row has nonfinite value")
    else:
        if not row["reason"] or row["value"] != "":
            raise ValueError("non-ok row must have a reason and blank value")
    for field in NUMERIC_RESULT_FIELDS[1:]:
        if row[field] != "" and not np.isfinite(float(row[field])):
            raise ValueError(f"nonfinite metadata field: {field}")


def validate_resume_rows(prior_rows: list[dict], fresh_rows: list[dict]) -> dict[str, dict]:
    fresh = {row["row_key"]: serialized_row(row) for row in fresh_rows}
    if len(fresh) != len(fresh_rows):
        raise RuntimeError("fresh result rows contain duplicate keys")
    reused = {}
    for row in prior_rows:
        validate_result_row(row)
        key = row["row_key"]
        if key in reused:
            raise ValueError(f"duplicate partial result row: {key}")
        if key not in fresh:
            raise ValueError(f"unknown or stale partial result row: {key}")
        if row != fresh[key]:
            mismatches = [field for field in FIELDS if row[field] != fresh[key][field]]
            raise ValueError(f"corrupted or stale partial result row {key}: {','.join(mismatches)}")
        reused[key] = row
    return reused


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["scenario"], row["method"], row["metric"])
        grouped.setdefault(key, []).append(row)
    output = []
    for key in sorted(grouped):
        group = grouped[key]
        values = np.asarray([float(r["value"]) for r in group if r["status"] == "ok"], dtype=float)
        total = len(group)
        applicable = sum(r["status"] != "inapplicable" for r in group)
        failures = sum(r["status"] in ("nonconvergence", "invalid") for r in group)
        output.append({
            "scenario": key[0], "method": key[1], "metric": key[2], "n_rows": total,
            "n_ok": int(len(values)), "median": None if len(values) == 0 else float(np.median(values)),
            "q10": None if len(values) == 0 else float(np.quantile(values, 0.10)),
            "q90": None if len(values) == 0 else float(np.quantile(values, 0.90)),
            "ok_rate": len(values) / total, "failure_rate": failures / total,
            "applicability_rate": applicable / total,
        })
    return output


def array_discrepancy(a, b) -> tuple[float, bool]:
    a, b = np.asarray(a), np.asarray(b)
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    good = np.isfinite(a) & np.isfinite(b)
    return (float(np.max(np.abs(a[good] - b[good]))) if np.any(good) else 0.0, same_nan)


def verification_diagnostics() -> dict:
    scenario = next(s for s in SCENARIOS if s["id"] == "stable_moderate")
    data = generate_replicate(scenario, 0)
    selected, baseline_scores = select_parameters(
        data["train"][1], data["train"][2],
        data["validation"][1], data["validation"][2],
    )
    latent, observed, masks = data["test"]
    true_params = params_for(scenario)
    known_filter = infer_sequences(observed, masks, "known_kalman_filter", true_params)
    known_smoother = infer_sequences(observed, masks, "known_rts_smoother", true_params)
    batch = infer_sequences(observed, masks, "batch_conditioning_oracle", true_params)
    independent = []
    for i in range(len(observed)):
        f = P3.kalman_filter(observed[i], masks[i], true_params)
        independent.append(max(array_discrepancy(known_filter[key][i], f[key])[0] for key in f))
    prefix = 17
    poisoned = observed.copy()
    poisoned[:, prefix:] += np.linspace(1e3, 2e3, observed.shape[1] - prefix)
    poison_filter = infer_sequences(poisoned, masks, "known_kalman_filter", true_params)
    causal = max(array_discrepancy(known_filter[key][:, :prefix], poison_filter[key][:, :prefix])[0] for key in known_filter)
    poisoned_test_observations = observed + 1e6
    selected_test_poison, test_scores = select_parameters(
        data["train"][1], data["train"][2],
        data["validation"][1], data["validation"][2],
    )
    poisoned_latent_truth = {k: v[0] + 1e6 for k, v in data.items()}
    selected_latent_poison, latent_scores = select_parameters(
        data["train"][1], data["train"][2],
        data["validation"][1], data["validation"][2],
    )
    score_delta_test = max(abs(a["score"] - b["score"]) for a, b in zip(baseline_scores, test_scores))
    score_delta_latent = max(abs(a["score"] - b["score"]) for a, b in zip(baseline_scores, latent_scores))
    reset_scenario = next(s for s in SCENARIOS if s["id"] == "reset_boundary")
    reset_data = generate_replicate(reset_scenario, 0)["test"]
    reset_proper = infer_sequences(reset_data[1], reset_data[2], "known_kalman_filter", params_for(reset_scenario))
    reset_wrong = false_continuation(reset_data[1], reset_data[2], params_for(reset_scenario))
    diagnostics = {
        "phase3_source": str(PHASE3_PATH.relative_to(REPO_ROOT)),
        "oracle": {
            "filter_prefix_batch_mean_max_abs": max(
                max(abs(known_filter["mean"][i, t] - P3.batch_condition(observed[i, :t+1], masks[i, :t+1], true_params)["mean"][-1]) for t in range(observed.shape[1]))
                for i in range(len(observed))
            ),
            "rts_batch_mean_max_abs": float(np.max(np.abs(known_smoother["mean"] - batch["mean"]))),
            "rts_batch_variance_max_abs": float(np.max(np.abs(known_smoother["var"] - batch["var"]))),
            "smoother_minus_filter_variance_max": float(np.max(known_smoother["var"] - known_filter["var"])),
            "causal_prefix_max_abs": causal,
        },
        "leakage": {
            "restricted_selection_arguments": ["train_observations", "train_masks", "validation_observations", "validation_masks"],
            "test_observation_poison_not_passed_shape": list(poisoned_test_observations.shape),
            "test_observation_poison_selected_unchanged": selected_test_poison == selected,
            "test_observation_poison_score_max_abs": score_delta_test,
            "latent_truth_poison_not_passed_splits": sorted(poisoned_latent_truth),
            "latent_truth_poison_selected_unchanged": selected_latent_poison == selected,
            "latent_truth_poison_score_max_abs": score_delta_latent,
        },
        "reset": {
            "independent_invocation_max_abs": float(max(independent)),
            "false_continuation_first_state_abs_difference": float(abs(reset_wrong["mean"][1, 0] - reset_proper["mean"][1, 0])),
        },
    }
    diagnostics["scoring_fault_injection"] = scoring_fault_diagnostics(scenario, selected, latent, observed, masks)
    return diagnostics


def scoring_fault_diagnostics(scenario: dict, selected: dict, latent: np.ndarray,
                              observed: np.ndarray, masks: np.ndarray) -> dict:
    method = next(m for m in METHODS if m["id"] == "known_kalman_filter")
    params = params_for(scenario)
    baseline = infer_sequences(observed, masks, method["id"], params)
    negative_variance = {key: value.copy() for key, value in baseline.items()}
    negative_variance["var"][0, 0] = -1.0
    variance_rows = []
    score_method(variance_rows, scenario, 0, method, selected, latent, observed, masks,
                 negative_variance, params, None, baseline, None)
    nonfinite_nll = {key: value.copy() for key, value in baseline.items()}
    first_observed = tuple(np.argwhere(masks)[0])
    nonfinite_nll["nll"][first_observed] = np.inf
    nll_rows = []
    score_method(nll_rows, scenario, 0, method, selected, latent, observed, masks,
                 nonfinite_nll, params, None, baseline, None)
    missing_result_rows = []
    score_method(missing_result_rows, scenario, 0, method, selected, latent, observed, masks,
                 None, params, None, baseline, None)
    by_metric_variance = {row["metric"]: row for row in variance_rows}
    by_metric_nll = {row["metric"]: row for row in nll_rows}
    return {
        "negative_variance_interval_status": by_metric_variance["interval_95_coverage"]["status"],
        "negative_variance_mean_status": by_metric_variance["mean_posterior_variance"]["status"],
        "nonfinite_nll_status": by_metric_nll["predictive_nll"]["status"],
        "missing_result_all_invalid": bool(all(row["status"] == "invalid" for row in missing_result_rows)),
        "machine_readable_reasons_present": bool(all(
            row["reason"] for row in (
                by_metric_variance["interval_95_coverage"],
                by_metric_variance["mean_posterior_variance"],
                by_metric_nll["predictive_nll"],
            )
        )),
    }


def make_summary(path: Path, summaries: list[dict], checks: dict) -> None:
    width, height = 1500, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), "P4-U01: Linear-Gaussian robustness benchmark (median with q10-q90)", fill="black", font=font)
    panels = [
        ("A. Online information only: stable_moderate latent RMSE", "stable_moderate", "latent_rmse", ["persistence", "open_loop", "known_kalman_filter", "selected_kalman_filter"]),
        ("B. Retrospective information only: stable_moderate latent RMSE", "stable_moderate", "latent_rmse", ["known_rts_smoother", "selected_rts_smoother"]),
        ("C. Online information only: wrong-parameter control", "wrong_parameters", "latent_rmse", ["known_kalman_filter", "misspecified_kalman_filter"]),
    ]
    boxes = [(25, 55, 735, 475), (765, 55, 1475, 475), (25, 505, 735, 925), (765, 505, 1475, 925)]
    lookup = {(s["scenario"], s["method"], s["metric"]): s for s in summaries}
    colors = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75),(227,119,194),(127,127,127)]
    for box, (title, scenario, metric, methods) in zip(boxes[:3], panels):
        draw.rectangle(box, outline="black", width=2)
        draw.text((box[0]+10, box[1]+8), title, fill="black", font=font)
        records = [(m, lookup.get((scenario, m, metric))) for m in methods]
        records = [(m, r) for m, r in records if r and r["median"] is not None]
        max_value = max((r["q90"] for _, r in records), default=1.0) * 1.12
        x0, y0, x1, y1 = box[0]+175, box[1]+45, box[2]-25, box[3]-30
        for i, (method, record) in enumerate(records):
            y = y0 + (i + 0.5) * (y1-y0) / max(len(records), 1)
            scale = (x1-x0) / max(max_value, 1e-12)
            draw.text((box[0]+10, y-6), method, fill="black", font=font)
            draw.line((x0+record["q10"]*scale, y, x0+record["q90"]*scale, y), fill=colors[i % len(colors)], width=4)
            x = x0+record["median"]*scale
            draw.ellipse((x-4, y-4, x+4, y+4), fill=colors[i % len(colors)])
        draw.text((box[0]+10, box[3]-18), "Within one information-set class; q10/median/q90 across five replicates.", fill="black", font=font)
    box = boxes[3]
    draw.rectangle(box, outline="black", width=2)
    draw.text((box[0]+10, box[1]+8), "D. Batch conditioning oracle: numerical correctness reference only", fill="black", font=font)
    oracle = checks["oracle"]
    lines = [
        "The batch oracle is not plotted beside deployable methods.",
        "It checks exact Gaussian recursion on bounded sequences:",
        f"filter vs prefix-batch mean max abs: {oracle['filter_prefix_batch_mean_max_abs']:.4g}",
        f"RTS vs full-batch mean max abs: {oracle['rts_batch_mean_max_abs']:.4g}",
        f"RTS vs full-batch variance max abs: {oracle['rts_batch_variance_max_abs']:.4g}",
        f"max(smoother variance - filter variance): {oracle['smoother_minus_filter_variance_max']:.4g}",
        "No oracle-versus-online performance rank is implied.",
    ]
    for i, line in enumerate(lines):
        draw.text((box[0]+20, box[1]+55+i*42), line, fill="black", font=font)
    draw.text((box[0]+10, box[3]-18), "Correctness reference; not a candidate method or universal winner.", fill="black", font=font)
    image.save(path, format="PNG", optimize=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, reverse: bool = False, resume: bool = False, quiet: bool = False) -> dict:
    start = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    work = [(s, r) for s in SCENARIOS for r in range(REPLICATES)]
    if reverse:
        work.reverse()
    fresh_rows = []
    selection_records = []
    for scenario, replicate in work:
        new_rows, selection = evaluate_replicate(scenario, replicate)
        selection_records.append(selection)
        fresh_rows.extend(new_rows)
    prior_rows = read_results(root / "results.csv") if resume else []
    prior = validate_resume_rows(prior_rows, fresh_rows) if resume else {}
    rows = [prior.get(row["row_key"], row) for row in fresh_rows]
    keys = [r["row_key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate result row keys")
    write_results(root / "results.csv", rows)
    canonical_rows = read_results(root / "results.csv")
    summaries = summarize(canonical_rows)
    checks = verification_diagnostics()
    label_counts = {label: sum(r["status"] == label for r in canonical_rows) for label in STATUS_VALUES}
    diagnostics = {
        "schema_version": 1,
        "summary_grouping": ["scenario", "method", "metric"],
        "quantiles": [0.10, 0.50, 0.90],
        "summaries": summaries,
        "status_counts": label_counts,
        "selection_by_replicate": sorted(selection_records, key=lambda x: (x["scenario"], x["replicate"])),
        "verification_checks": checks,
        "scientific_limits": [
            "All rankings are conditional on this scalar simulator, declared scenario, metric, replicate design, and information set.",
            "Retrospective smoothers and the direct batch oracle use future observations and are not online competitors.",
            "Finite-grid selection is a bounded train/validation baseline, not general parameter learning.",
            "Results do not establish mechanism, causality, biological validity, real-data validity, or universal method superiority.",
            "No cross-task scalar score or universal winner is computed.",
        ],
    }
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    make_summary(root / "summary.png", summaries, checks)
    hashes = {name: sha256(root / name) for name in ARTIFACT_NAMES}
    manifest = {
        "schema_version": 1,
        "object_id": "P4-U01",
        "artifact": "linear-gaussian-robustness",
        "date": "2026-08-20",
        "maturity": "L1",
        "status": "stable",
        "scientific_question": "Where are scalar LGSSM references, filters, smoothers, batch conditioning, and finite-grid plug-in inference applicable, robust, or brittle under declared synthetic scenarios and information sets?",
        "ground_truth": "known scalar stationary LGSSM: z[t+1]=a*z[t]+w[t], x[t]=z[t]+v[t]",
        "phase3_reuse": str(PHASE3_PATH.relative_to(REPO_ROOT)),
        "phase3_source_provenance": {
            "path": str(PHASE3_PATH.relative_to(REPO_ROOT)),
            "role": "actual_executable_routine_reuse",
            "sha256": sha256(PHASE3_PATH),
        },
        "stages": ["generation", "selection", "inference", "scoring"],
        "split_policy": "independent train, validation, and test sequences per replicate; test truth is scoring-only",
        "master_seed": MASTER_SEED,
        "seed_mapping": "PCG64(SeedSequence(master_seed, sha256(scenario_id)[:4], replicate, split_code, sequence))",
        "replicates": REPLICATES,
        "split_sizes": SPLIT_SIZES,
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "metrics": list(METRICS),
        "status_vocabulary": list(STATUS_VALUES),
        "parameter_grid": [{"a": a, "q": q, "r": r} for a, q, r in GRID],
        "selection": {"criterion": "mean train innovation NLL + mean validation innovation NLL; lexicographic tie break", "test_inputs_allowed": False, "latent_truth_allowed": False},
        "canonical_row_order": ["scenario", "replicate", "split", "method", "metric"],
        "row_key": ["scenario", "replicate", "split", "method", "metric"],
        "aggregation": {
            "within_replicate": "pool the two test sequences and applicable time points before computing each metric",
            "predictive_nll": "mean one-step observation predictive NLL over observed test times only",
            "across_replicates": "q10, median, and q90 across five independent replicate-level values",
            "selection_parameter_l1": "unnormalized |a_hat-a|+|q_hat-q|+|r_hat-r|; mixed scales limit cross-parameter interpretation",
        },
        "resume": "partial rows must match schema, key, metadata, status/value semantics, and freshly recomputed canonical bytes exactly; duplicate, unknown, stale, corrupted, or incomplete rows are rejected",
        "tolerances": TOL,
        "artifact_files": ["manifest.json", *ARTIFACT_NAMES],
        "files_sha256": hashes,
        "commands": {
            "python_executable": sys.executable,
            "generate": f"{sys.executable} benchmarks/linear-gaussian-robustness.py --generate",
            "verify": f"{sys.executable} benchmarks/linear-gaussian-robustness.py --verify",
            "resume": f"{sys.executable} benchmarks/linear-gaussian-robustness.py --generate --resume",
        },
        "runtime": {"python_executable": sys.executable, "python": sys.version.split()[0], "numpy": np.__version__, "pillow": getattr(sys.modules.get("PIL"), "__version__", "unknown"), "platform": platform.platform()},
        "interpretation_boundary": "Simulator/task/metric/information-set conditional; not mechanism, causality, real-data validity, or universal superiority.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - start
    result = {"artifact_root": str(root), "rows": len(rows), "runtime_seconds": elapsed, "hashes": hashes}
    if not quiet:
        print(json.dumps(result, indent=2))
    return result


def compare_roots(left: Path, right: Path) -> list[str]:
    return [name for name in ("manifest.json", *ARTIFACT_NAMES) if (left / name).read_bytes() != (right / name).read_bytes()]


def validate_artifacts(root: Path) -> list[str]:
    errors = []
    expected = {"manifest.json", *ARTIFACT_NAMES}
    actual = {p.name for p in root.iterdir() if p.is_file()}
    if actual != expected:
        errors.append(f"artifact set mismatch: expected {sorted(expected)}, got {sorted(actual)}")
        return errors
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    rows = read_results(root / "results.csv")
    for row in rows:
        try:
            validate_result_row(row)
        except ValueError as exc:
            errors.append(f"invalid result row {row.get('row_key', '<missing>')}: {exc}")
    for name, digest in manifest["files_sha256"].items():
        if sha256(root / name) != digest:
            errors.append(f"hash mismatch: {name}")
    source = manifest.get("phase3_source_provenance", {})
    expected_source_path = str(PHASE3_PATH.relative_to(REPO_ROOT))
    if source.get("path") != expected_source_path:
        errors.append("Phase 3 source path mismatch")
    if source.get("role") != "actual_executable_routine_reuse":
        errors.append("Phase 3 source role mismatch")
    if source.get("sha256") != sha256(PHASE3_PATH):
        errors.append("Phase 3 source digest mismatch")
    keys = [r["row_key"] for r in rows]
    if len(keys) != len(set(keys)):
        errors.append("row keys are not unique")
    if rows != sorted(rows, key=canonical_key):
        errors.append("results are not in canonical order")
    if set(r["status"] for r in rows) - set(STATUS_VALUES):
        errors.append("unknown status label")
    if any(r["status"] != "ok" and not r["reason"] for r in rows):
        errors.append("non-ok row without reason")
    if any(r["status"] == "ok" and r["value"] == "" for r in rows):
        errors.append("ok row without numeric value")
    represented = {r["scenario_class"] for r in rows}
    required = {"transition_stability", "noise_ratio", "sequence_length", "missing_blocks", "wrong_parameters", "reset_boundary_failure", "near_unit_root_stress"}
    if not required.issubset(represented):
        errors.append(f"scenario classes missing: {sorted(required-represented)}")
    families = {r["method_family"] for r in rows if r["status"] == "ok"}
    required_families = {"stationary_mean", "persistence_open_loop", "known_parameter_kalman", "fixed_interval_rts", "direct_batch_conditioning", "finite_grid_selected"}
    if not required_families.issubset(families):
        errors.append(f"method families missing: {sorted(required_families-families)}")
    checks = diagnostics["verification_checks"]
    oracle = checks["oracle"]
    for name in ("filter_prefix_batch_mean_max_abs", "rts_batch_mean_max_abs", "rts_batch_variance_max_abs", "causal_prefix_max_abs"):
        if oracle[name] > TOL["oracle"]:
            errors.append(f"oracle tolerance failed: {name}={oracle[name]}")
    if oracle["smoother_minus_filter_variance_max"] > TOL["variance"]:
        errors.append("smoother variance exceeds filter variance")
    leakage = checks["leakage"]
    if not leakage["test_observation_poison_selected_unchanged"] or leakage["test_observation_poison_score_max_abs"] > TOL["oracle"]:
        errors.append("test-observation poison changed selection")
    if not leakage["latent_truth_poison_selected_unchanged"] or leakage["latent_truth_poison_score_max_abs"] > TOL["oracle"]:
        errors.append("latent-truth poison changed selection")
    reset = checks["reset"]
    if reset["independent_invocation_max_abs"] > TOL["oracle"]:
        errors.append("independent reset invocation mismatch")
    if reset["false_continuation_first_state_abs_difference"] <= 1e-8:
        errors.append("false-continuation negative control did not visibly differ")
    faults = checks["scoring_fault_injection"]
    if not (
        faults["negative_variance_interval_status"] == "invalid"
        and faults["negative_variance_mean_status"] == "invalid"
        and faults["nonfinite_nll_status"] == "invalid"
        and faults["missing_result_all_invalid"]
        and faults["machine_readable_reasons_present"]
    ):
        errors.append("scoring fault-injection labels failed")
    return errors


def adversarial_resume_checks(full_rows: list[dict], root: Path) -> list[str]:
    failures = []

    def must_reject(label: str, action) -> None:
        try:
            action()
        except (ValueError, RuntimeError):
            return
        failures.append(f"adversarial resume case was accepted: {label}")

    fresh_rows, _ = evaluate_replicate(SCENARIOS[0], 0)
    fresh_map = {row["row_key"]: row for row in fresh_rows}
    subset = [row for row in full_rows if row["row_key"] in fresh_map]
    corrupted = [dict(row) for row in subset[:2]]
    corrupted[0]["value"] = "999" if corrupted[0]["status"] == "ok" else corrupted[0]["reason"] + "-corrupt"
    must_reject("corrupted value", lambda: validate_resume_rows(corrupted, fresh_rows))
    stale = [dict(row) for row in subset[:2]]
    stale[0]["true_a"] = "0.123"
    must_reject("stale metadata", lambda: validate_resume_rows(stale, fresh_rows))
    duplicate = [dict(subset[0]), dict(subset[0])]
    must_reject("duplicate key", lambda: validate_resume_rows(duplicate, fresh_rows))
    unknown = [dict(subset[0])]
    unknown[0]["row_key"] = unknown[0]["row_key"] + "|unknown"
    must_reject("unknown key", lambda: validate_resume_rows(unknown, fresh_rows))
    incomplete = root / "incomplete.csv"
    with incomplete.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS[:-1], lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: subset[0][field] for field in FIELDS[:-1]})
    must_reject("incomplete schema", lambda: read_results(incomplete))
    return failures


def verify(root: Path) -> None:
    errors = validate_artifacts(root)
    with tempfile.TemporaryDirectory(prefix="ldg-p4-u01-") as tmp:
        tmp = Path(tmp)
        normal = tmp / "normal"
        reverse = tmp / "reverse"
        resumed = tmp / "resumed"
        build(normal, quiet=True)
        build(reverse, reverse=True, quiet=True)
        errors.extend(f"byte regeneration mismatch: {name}" for name in compare_roots(root, normal))
        errors.extend(f"reversed traversal mismatch: {name}" for name in compare_roots(normal, reverse))
        resumed.mkdir()
        full_rows = read_results(normal / "results.csv")
        write_results(resumed / "results.csv", full_rows[:len(full_rows)//2])
        build(resumed, resume=True, quiet=True)
        errors.extend(f"resume mismatch: {name}" for name in compare_roots(normal, resumed))
        errors.extend(adversarial_resume_checks(full_rows, tmp))
    if errors:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"verified": True, "artifact_root": str(root), "checks": ["hashes", "unique_keys", "canonical_order", "byte_regeneration", "reversed_order", "valid_partial_resume", "adversarial_resume_rejection", "restricted_selection_api", "poison", "causal_prefix", "reset", "batch_oracles", "scoring_fault_injection"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reverse", action="store_true", help="reverse traversal; canonical bytes must remain unchanged")
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
