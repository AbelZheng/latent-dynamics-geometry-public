#!/usr/bin/env python3
"""P4-U05 deterministic nonlinear-reconstruction and graph-geometry benchmark.

The reconstruction and graph tasks have separate scenarios, methods, targets,
metrics, and summaries. NumPy and Pillow are the only runtime dependencies.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASTER_SEED = 20260820
REPLICATES = 3
STATUS_VALUES = ("ok", "nonconvergence", "invalid", "inapplicable")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "nonlinear-graph-robustness"
ARTIFACT_NAMES = ("results.csv", "diagnostics.json", "summary.png")
PHASE3_SOURCES = (
    REPO_ROOT / "toy-models" / "bounded-nonlinear-reconstruction.py",
    REPO_ROOT / "toy-models" / "graph-geometry-recovery.py",
)


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P3_GR = load_source(PHASE3_SOURCES[1], "ldg_p3_graph")

# --------------------------- reconstruction task ---------------------------
REC_P = 6
REC_SCENARIOS = (
    {"id": "reference_curved", "class": "curvature", "curvature": 1.0, "sizes": (120, 60, 90)},
    {"id": "low_curvature", "class": "curvature", "curvature": 0.25, "sizes": (120, 60, 90)},
    {"id": "high_curvature", "class": "curvature", "curvature": 1.65, "sizes": (120, 60, 90)},
    {"id": "moment_matched_gaussian", "class": "support_family", "gaussian": True, "sizes": (120, 60, 90)},
    {"id": "small_sample_high_capacity", "class": "sample_capacity", "curvature": 1.0, "sizes": (28, 42, 90), "hidden": 18},
    {"id": "high_variance_nuisance", "class": "nuisance", "curvature": 1.0, "sizes": (120, 60, 90), "nuisance": 2.4},
    {"id": "support_shift", "class": "support_shift", "curvature": 1.0, "sizes": (120, 60, 90), "shift": True},
)
REC_METHODS = (
    ("mean", "mean", "train_fit"),
    ("pca", "pca", "train_fit"),
    ("linear_autoencoder", "linear_autoencoder", "analytic_train_fit"),
    ("nonlinear_autoencoder", "nonlinear_autoencoder", "train_validation_selected"),
    ("frozen_encoder", "frozen_random_encoder", "train_validation_selected"),
)
REC_METRICS = (
    "observed_reconstruction_mse",
    "noiseless_signal_mse",
    "code_variance",
    "code_abs_spearman_truth",
    "code_collision_rate",
    "selected_validation_mse",
    "parameter_count",
)

# ------------------------------- graph task --------------------------------
HAIRPIN_N = {"train": 108, "validation": 42, "test": 54}
CIRCLE_N = 64
GRAPH_SCENARIOS = (
    {"id": "hairpin_reference", "class": "reference", "domain": "hairpin", "k": 6},
    {"id": "hairpin_sparse_gap", "class": "sampling_gap", "domain": "hairpin", "k": 6, "sampling": "gap"},
    {"id": "hairpin_shortcuts", "class": "shortcuts", "domain": "hairpin", "k": 6, "radius": 0.06},
    {"id": "hairpin_nonuniform", "class": "nonuniform_density", "domain": "hairpin", "k": 6, "sampling": "nonuniform"},
    {"id": "hairpin_disconnected", "class": "connectivity", "domain": "hairpin", "k": 1},
    {"id": "hairpin_large_k", "class": "neighborhood", "domain": "hairpin", "k": 16},
    {"id": "hairpin_nearly_complete", "class": "neighborhood", "domain": "hairpin", "k": 100},
    {"id": "hairpin_metric_scaling", "class": "metric_scaling", "domain": "hairpin", "k": 6, "transform": (1.0, 1.0, 3.0)},
    {"id": "circle_reference", "class": "reference", "domain": "circle", "epsilon": 0.18},
    {"id": "circle_nonuniform", "class": "nonuniform_density", "domain": "circle", "epsilon": 0.18, "sampling": "nonuniform"},
    {"id": "circle_bandwidth_small", "class": "bandwidth", "domain": "circle", "epsilon": 0.008},
    {"id": "circle_bandwidth_large", "class": "bandwidth", "domain": "circle", "epsilon": 4.0},
    {"id": "circle_metric_scaling", "class": "metric_scaling", "domain": "circle", "epsilon": 0.18, "transform": (1.0, 1.0, 3.0)},
)
GRAPH_METHODS = (
    ("mean", "mean", "train_scoring_reference"),
    ("pca", "pca", "train_fit"),
    ("ambient_mds", "classical_mds", "train_fit_gower_extension"),
    ("isomap", "isomap", "train_fit_gower_extension"),
    ("graph_diagnostics", "knn_graph", "train_fit"),
    ("diffusion_maps", "diffusion_maps", "train_fit"),
    ("direct_operator", "markov_operator", "train_fit"),
    ("nystrom_extension", "diffusion_maps_extension", "frozen_train_fit"),
)
GRAPH_METRICS = (
    "heldout_coordinate_rmse",
    "graph_distance_relative_rmse",
    "component_count",
    "material_shortcut_rate",
    "infinite_pair_rate",
    "mds_negative_eigenmass",
    "diffusion_eigenspace_angle_max_degrees",
    "diffusion_truncation_relative_rmse",
    "direct_spectral_distance_max_abs",
    "heldout_extension_rmse",
    "operator_row_sum_max_abs",
)

FIELDS = (
    "row_key", "task", "scenario", "scenario_class", "replicate", "seed_id",
    "provenance", "split", "evaluation_stage", "method", "method_family",
    "information_set", "parameter_source", "target", "metric", "status",
    "reason", "value", "n_train", "n_validation", "n_test",
    "selected_hyperparameter", "domain",
)


def scenario_code(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def rng_for(task: str, scenario: str, replicate: int, stage: int, split: str = "train", index: int = 0):
    split_code = {"train": 1, "validation": 2, "test": 3, "diagnostic": 4}[split]
    entropy = (MASTER_SEED, scenario_code(task), scenario_code(scenario), replicate, stage, split_code, index)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def seed_id(task: str, scenario: str, replicate: int) -> str:
    return f"pcg64:{MASTER_SEED}:{scenario_code(task)}:{scenario_code(scenario)}:{replicate}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_max_abs(left, right) -> float:
    if isinstance(left, dict):
        return max((tree_max_abs(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, (tuple, list)):
        return max((tree_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0


# ------------------------ reconstruction implementation ---------------------
def curved_map(s: np.ndarray, curvature: float) -> np.ndarray:
    s = np.asarray(s, float)
    c = float(curvature)
    cos_center = math.sin(1.5) / 1.5
    return np.column_stack((
        s,
        c * s ** 2,
        .35 * c * np.sin(2 * s),
        .2 * c * s ** 3,
        c * (np.cos(s) - cos_center),
        .4 * c * np.sin(3 * s),
    ))


def population_moments(curvature: float) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(128)
    s = 1.5 * nodes
    x = curved_map(s, curvature)
    w = .5 * weights
    mean = w @ x
    xc = x - mean
    cov = (xc * w[:, None]).T @ xc + .06 ** 2 * np.eye(REC_P)
    return mean, cov


def generate_rec_split(scenario: dict, replicate: int, split: str) -> dict:
    n = scenario["sizes"][("train", "validation", "test").index(split)]
    if scenario.get("gaussian"):
        mean, cov = population_moments(1.0)
        x = rng_for("reconstruction", scenario["id"], replicate, 10, split).multivariate_normal(mean, cov, n)
        return {"observed": x, "truth": None, "noiseless": None}
    rng = rng_for("reconstruction", scenario["id"], replicate, 11, split)
    if scenario.get("shift") and split == "test":
        side = rng.integers(0, 2, n)
        mag = rng.uniform(1.5, 2.35, n)
        s = np.where(side == 0, -mag, mag)
    else:
        s = rng.uniform(-1.5, 1.5, n)
    noiseless = curved_map(s, scenario.get("curvature", 1.0))
    x = noiseless + rng_for("reconstruction", scenario["id"], replicate, 12, split).normal(0, .06, (n, REC_P))
    if scenario.get("nuisance"):
        x[:, 4:] += rng_for("reconstruction", scenario["id"], replicate, 13, split).normal(0, scenario["nuisance"], (n, 2))
    return {"observed": x, "truth": s, "noiseless": noiseless}


def generate_rec(scenario: dict, replicate: int) -> dict:
    if scenario.get("shift"):
        base = dict(next(s for s in REC_SCENARIOS if s["id"] == "reference_curved"))
        train = generate_rec_split(base, replicate, "train")
        validation = generate_rec_split(base, replicate, "validation")
        test = generate_rec_split(scenario, replicate, "test")
        return {"train": train, "validation": validation, "test": test}
    return {split: generate_rec_split(scenario, replicate, split) for split in ("train", "validation", "test")}


def standardizer(x: np.ndarray) -> dict:
    scale = np.std(x, axis=0)
    return {"mean": np.mean(x, axis=0), "scale": np.where(scale > 1e-12, scale, 1.0)}


def transform(x, prep): return (x - prep["mean"]) / prep["scale"]
def inverse_transform(x, prep): return prep["mean"] + x * prep["scale"]


def pca_model(x: np.ndarray) -> dict:
    mean = x.mean(0); xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:1].T
    if basis[np.argmax(np.abs(basis[:, 0])), 0] < 0: basis *= -1
    return {"kind": "pca", "mean": mean, "basis": basis, "parameter_count": 12}


def init_nonlinear(hidden: int, rng, frozen: bool = False) -> dict:
    encoder = {
        "w1": rng.normal(0, math.sqrt(2/(REC_P+hidden)), (REC_P, hidden)), "b1": np.zeros(hidden),
        "w2": rng.normal(0, math.sqrt(2/(hidden+1)), (hidden, 1)), "b2": np.zeros(1),
    }
    decoder = {
        "w3": rng.normal(0, math.sqrt(2/(hidden+1)), (1, hidden)), "b3": np.zeros(hidden),
        "w4": rng.normal(0, math.sqrt(2/(hidden+REC_P)), (hidden, REC_P)), "b4": np.zeros(REC_P),
    }
    return {"encoder": encoder, "decoder": decoder, "frozen": frozen}


def copy_network(net):
    return {group: {k: np.array(v, copy=True) for k, v in net[group].items()} for group in ("encoder", "decoder")} | {"frozen": net["frozen"]}


def network_forward(net, x):
    e, d = net["encoder"], net["decoder"]
    h1 = np.tanh(x @ e["w1"] + e["b1"]); z = h1 @ e["w2"] + e["b2"]
    h3 = np.tanh(z @ d["w3"] + d["b3"]); y = h3 @ d["w4"] + d["b4"]
    return y, z, (h1, h3)


def network_loss_grad(net, x):
    y, z, (h1, h3) = network_forward(net, x); residual = y - x
    dy = 2 * residual / residual.size
    d = net["decoder"]; e = net["encoder"]
    gd = {"w4": h3.T @ dy, "b4": dy.sum(0)}
    da3 = (dy @ d["w4"].T) * (1 - h3**2)
    gd.update({"w3": z.T @ da3, "b3": da3.sum(0)})
    grads = {"decoder": gd}
    if not net["frozen"]:
        dz = da3 @ d["w3"].T
        ge = {"w2": h1.T @ dz, "b2": dz.sum(0)}
        da1 = (dz @ e["w2"].T) * (1 - h1**2)
        ge.update({"w1": x.T @ da1, "b1": da1.sum(0)})
        grads["encoder"] = ge
    return float(np.mean(residual**2)), grads


def train_network(train: np.ndarray, validation: np.ndarray, hidden: int, rng, frozen: bool) -> dict:
    net = init_nonlinear(hidden, rng, frozen)
    groups = ("decoder",) if frozen else ("encoder", "decoder")
    m = {g: {k: np.zeros_like(v) for k, v in net[g].items()} for g in groups}
    v = {g: {k: np.zeros_like(val) for k, val in net[g].items()} for g in groups}
    best = copy_network(net); best_loss = float("inf"); best_epoch = 0; wait = 0
    epochs = 260 if hidden <= 8 else 340; patience = 45; lr = .012 if not frozen else .016
    stop_reason = "epoch_budget"
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        _, grads = network_loss_grad(net, train)
        for g in groups:
            for k in net[g]:
                m[g][k] = .9*m[g][k] + .1*grads[g][k]
                v[g][k] = .999*v[g][k] + .001*grads[g][k]**2
                net[g][k] -= lr*(m[g][k]/(1-.9**epoch))/(np.sqrt(v[g][k]/(1-.999**epoch))+1e-8)
        val = network_loss_grad(net, validation)[0]
        if val < best_loss - 1e-10:
            best, best_loss, best_epoch, wait = copy_network(net), val, epoch, 0
        else:
            wait += 1
        if wait >= patience:
            stop_reason = "patience"
            break
    best.update({"kind": "frozen_encoder" if frozen else "nonlinear_autoencoder", "best_validation_mse": best_loss,
                 "best_epoch": best_epoch, "epochs_run": epochs_run, "max_epoch": epochs,
                 "patience": patience, "stop_reason": stop_reason, "hidden": hidden,
                 "parameter_count": sum(a.size for g in ("encoder", "decoder") for a in best[g].values())})
    return best


def fit_rec_states(train_observations: np.ndarray, validation_observations: np.ndarray, scenario_id: str, replicate: int, hidden: int) -> dict:
    prep = standardizer(train_observations)
    train = transform(train_observations, prep); val = transform(validation_observations, prep)
    pca = pca_model(train)
    states = {
        "preprocessing": prep,
        "mean": {"kind": "mean", "mean": train.mean(0), "parameter_count": REC_P, "best_validation_mse": float(np.mean((val-train.mean(0))**2))},
        "pca": pca,
        "linear_autoencoder": {**pca, "kind": "linear_autoencoder"},
    }
    for method, frozen in (("nonlinear_autoencoder", False), ("frozen_encoder", True)):
        candidates = []
        candidate_seed_ids = []
        for candidate in range(2):
            rng = rng_for("reconstruction", scenario_id, replicate, 30 + int(frozen), "train", candidate)
            candidates.append(train_network(train, val, hidden, rng, frozen))
            candidate_seed_ids.append(f"pcg64:{MASTER_SEED}:{scenario_code('reconstruction')}:{scenario_code(scenario_id)}:{replicate}:{30 + int(frozen)}:1:{candidate}")
        selected = min(enumerate(candidates), key=lambda z: (z[1]["best_validation_mse"], z[0]))
        selection_record = {
            "candidate_indices": list(range(len(candidates))),
            "candidate_seed_ids": candidate_seed_ids,
            "candidate_validation_losses": [float(x["best_validation_mse"]) for x in candidates],
            "candidate_best_epochs": [int(x["best_epoch"]) for x in candidates],
            "candidate_epochs_run": [int(x["epochs_run"]) for x in candidates],
            "candidate_max_epochs": [int(x["max_epoch"]) for x in candidates],
            "candidate_stop_reasons": [x["stop_reason"] for x in candidates],
            "selected_candidate": int(selected[0]),
            "selected_seed_id": candidate_seed_ids[selected[0]],
            "selected_validation_loss": float(selected[1]["best_validation_mse"]),
            "selected_best_epoch": int(selected[1]["best_epoch"]),
            "selected_max_epoch": int(selected[1]["max_epoch"]),
            "selected_stop_reason": selected[1]["stop_reason"],
            "hidden_width": int(hidden),
            "tie_rule": "minimum_validation_loss_then_lower_candidate_index",
            "epoch_cap_semantics": "diagnostic_only_not_automatic_nonconvergence",
        }
        states[method] = selected[1] | {"selected_candidate": selected[0], "selection_record": selection_record}
    for method in ("pca", "linear_autoencoder"):
        pred, _ = rec_predict(states[method], val)
        states[method]["best_validation_mse"] = float(np.mean((pred-val)**2))
    return states


def rec_predict(model: dict, x: np.ndarray):
    if model["kind"] == "mean": return np.broadcast_to(model["mean"], x.shape).copy(), np.zeros((len(x), 1))
    if model["kind"] in ("pca", "linear_autoencoder"):
        z = (x-model["mean"]) @ model["basis"]
        return model["mean"] + z @ model["basis"].T, z
    return network_forward(model, x)[:2]


def recover_rec(states: dict, test_observations: np.ndarray, fault: str | None = None) -> dict:
    x = transform(test_observations, states["preprocessing"]); result = {}
    for method, _, _ in REC_METHODS:
        try:
            if fault == "linalg" and method == "nonlinear_autoencoder": raise np.linalg.LinAlgError("injected_recovery_failure")
            pred, code = rec_predict(states[method], x)
            if fault == "nonfinite" and method == "nonlinear_autoencoder": pred = pred.copy(); pred[0, 0] = np.nan
            result[method] = {"reconstruction": inverse_transform(pred, states["preprocessing"]), "code": code, "error": None}
        except Exception as exc:
            result[method] = {"reconstruction": None, "code": None, "error": exc}
    return result


def rankdata(x):
    order = np.argsort(x, kind="mergesort"); ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def abs_spearman(x, y):
    if np.std(x) < 1e-15 or np.std(y) < 1e-15: return 0.0
    return abs(float(np.corrcoef(rankdata(np.asarray(x).ravel()), rankdata(np.asarray(y).ravel()))[0, 1]))


def collision_rate(code, truth):
    code = np.asarray(code).ravel(); truth = np.asarray(truth).ravel(); upper = np.triu(np.ones((len(code), len(code)), bool), 1)
    close = np.abs(code[:, None]-code[None, :]) <= .05*max(float(np.ptp(code)), 1e-12)
    denom = int(np.sum(upper & close)); bad = int(np.sum(upper & close & (np.abs(truth[:, None]-truth[None, :]) > .5)))
    return bad/max(denom, 1)


def rec_applicable(method: str, metric: str, scenario: dict) -> tuple[bool, str]:
    if scenario.get("gaussian") and metric in ("noiseless_signal_mse", "code_abs_spearman_truth", "code_collision_rate"):
        return False, "no_scalar_or_noiseless_truth_for_full_dimensional_gaussian"
    return True, ""


def rec_target(metric):
    return {
        "observed_reconstruction_mse": "observed_test_reconstruction",
        "noiseless_signal_mse": "known_noiseless_test_signal",
        "code_variance": "deterministic_test_code_activity",
        "code_abs_spearman_truth": "scoring_only_generator_parameter",
        "code_collision_rate": "scoring_only_generator_collision_diagnostic",
        "selected_validation_mse": "validation_selection_diagnostic",
        "parameter_count": "architecture_diagnostic",
    }[metric]


def rec_metric(method, metric, scenario, data, states, recovery):
    if metric == "selected_validation_mse": return states[method]["best_validation_mse"]
    if metric == "parameter_count": return states[method]["parameter_count"]
    if recovery[method]["error"] is not None: raise recovery[method]["error"]
    rec = recovery[method]["reconstruction"]; code = recovery[method]["code"]
    if metric == "observed_reconstruction_mse": return float(np.mean((rec-data["test"]["observed"])**2))
    if metric == "noiseless_signal_mse": return float(np.mean((rec-data["test"]["noiseless"])**2))
    if metric == "code_variance": return float(np.var(code))
    if metric == "code_abs_spearman_truth": return abs_spearman(code, data["test"]["truth"])
    if metric == "code_collision_rate": return collision_rate(code, data["test"]["truth"])
    raise KeyError(metric)


# ---------------------------- graph implementation --------------------------
def graph_sizes(scenario): return (HAIRPIN_N["train"], HAIRPIN_N["validation"], HAIRPIN_N["test"]) if scenario["domain"] == "hairpin" else (CIRCLE_N, CIRCLE_N//2, CIRCLE_N)


def generate_hairpin_split(scenario, replicate, split):
    n = HAIRPIN_N[split]; radius = scenario.get("radius", P3_GR.RADIUS); total = 6 + math.pi*radius
    seed_scenario = "hairpin_reference" if scenario.get("transform") else scenario["id"]
    rng = rng_for("graph", seed_scenario, replicate, 50, split)
    if scenario.get("sampling") == "gap" and split == "train":
        lo, hi = 3-.18, 3+math.pi*radius+.18; pieces = []
        while sum(len(a) for a in pieces) < n:
            z = rng.uniform(0, total, n); pieces.append(z[(z < lo) | (z > hi)])
        ell = np.sort(np.concatenate(pieces)[:n])
    elif scenario.get("sampling") == "nonuniform":
        u = rng.uniform(size=n); ell = np.empty(n); left = u < .65
        ell[left] = total*(u[left]/.65)**2*.55
        ell[~left] = total*(.55+((u[~left]-.65)/.35)**.55*.45); ell.sort()
    else: ell = np.sort(rng.uniform(0, total, n))
    q, off = P3_GR.embed_isometry(); x = P3_GR.hairpin_planar(ell, radius) @ q.T + off
    x += rng_for("graph", seed_scenario, replicate, 51, split).normal(0, .01, x.shape)
    if scenario.get("transform"): x = x @ np.diag(scenario["transform"])
    return {"observed": x, "truth": ell, "total": total}


def circle_train_theta(scenario, replicate):
    n = CIRCLE_N
    seed_scenario = "circle_reference" if scenario.get("transform") else scenario["id"]
    if scenario.get("sampling") == "nonuniform":
        u = np.sort(rng_for("graph", seed_scenario, replicate, 60, "train").uniform(size=n))
        return 2*math.pi*u**2.4
    phase = rng_for("graph", seed_scenario, replicate, 61, "train").uniform(0, 2*math.pi/n)
    return np.sort((phase+2*math.pi*np.arange(n)/n)%(2*math.pi))


def circular_midpoints(theta):
    theta = np.sort(np.asarray(theta, float))
    return (theta + .5*((np.roll(theta, -1)-theta)%(2*math.pi)))%(2*math.pi)


def generate_circle_split(scenario, replicate, split):
    n = graph_sizes(scenario)[("train", "validation", "test").index(split)]
    seed_scenario = "circle_reference" if scenario.get("transform") else scenario["id"]
    if split == "train":
        theta = circle_train_theta(scenario, replicate)
    elif split == "test":
        theta = circular_midpoints(circle_train_theta(scenario, replicate))
    else:
        phase = rng_for("graph", seed_scenario, replicate, 62, split).uniform(0, 2*math.pi/n); theta = (phase+2*math.pi*(np.arange(n)+.5)/n)%(2*math.pi)
    x = P3_GR.circle_points(theta)
    if scenario.get("transform"): x = x @ np.diag(scenario["transform"])
    return {"observed": x, "truth": theta}


def generate_graph(scenario, replicate):
    fn = generate_hairpin_split if scenario["domain"] == "hairpin" else generate_circle_split
    return {split: fn(scenario, replicate, split) for split in ("train", "validation", "test")}


def fit_graph_states(train_observations: np.ndarray, scenario: dict) -> dict:
    if scenario["domain"] == "hairpin":
        k = min(scenario.get("k", 6), len(train_observations)-1)
        fit = P3_GR.fit_hairpin(train_observations, k=k)
        return {"domain": "hairpin", "fit": fit, "train_observations": np.array(train_observations, copy=True)}
    model = P3_GR.diffusion_fit(train_observations, scenario.get("epsilon", .18))
    return {"domain": "circle", "model": model, "train_observations": np.array(train_observations, copy=True)}


def fit_scoring_alignment(states: dict, train_truth: np.ndarray) -> dict:
    if states["domain"] != "hairpin": return {}
    fit = states["fit"]; out = {"mean": float(np.mean(train_truth))}
    for method, coords in (("pca", fit["pca"]), ("ambient_mds", fit["ambient_mds"]["coords"])):
        out[method] = P3_GR.fit_affine(coords, train_truth)
    if fit["isomap"] is not None: out["isomap"] = P3_GR.fit_affine(fit["isomap"]["coords"], train_truth)
    return out


def recover_graph(states: dict, test_observations: np.ndarray, fault: str | None = None) -> dict:
    if fault == "linalg": raise np.linalg.LinAlgError("injected_graph_recovery_failure")
    if states["domain"] == "hairpin":
        ext = P3_GR.extend_hairpin(states["fit"], test_observations, states["train_observations"])
        if fault == "nonfinite": ext["pca"] = ext["pca"].copy(); ext["pca"][0] = np.nan
        return ext
    psi, transition = P3_GR.nystrom(states["model"], test_observations)
    if fault == "nonfinite": psi = psi.copy(); psi[0, 1] = np.nan
    return {"psi": psi, "transition": transition}


def graph_applicable(scenario, method, metric):
    domain = scenario["domain"]
    hairpin_coord = method in ("mean", "pca", "ambient_mds", "isomap") and metric == "heldout_coordinate_rmse"
    hairpin_diag = method == "graph_diagnostics" and metric in ("graph_distance_relative_rmse", "component_count", "material_shortcut_rate", "infinite_pair_rate")
    hairpin_mds = method in ("ambient_mds", "isomap") and metric == "mds_negative_eigenmass"
    circle_dm = method == "diffusion_maps" and metric in ("diffusion_eigenspace_angle_max_degrees", "diffusion_truncation_relative_rmse")
    circle_direct = method == "direct_operator" and metric in ("direct_spectral_distance_max_abs", "operator_row_sum_max_abs")
    circle_ext = method == "nystrom_extension" and metric == "heldout_extension_rmse"
    if domain == "hairpin":
        return (hairpin_coord or hairpin_diag or hairpin_mds), "metric_not_applicable_to_hairpin_method"
    return (circle_dm or circle_direct or circle_ext), "metric_not_applicable_to_circle_method"


def graph_target(metric):
    return {
        "heldout_coordinate_rmse": "scoring_aligned_known_hairpin_arc_length",
        "graph_distance_relative_rmse": "known_hairpin_pairwise_arc_separation",
        "component_count": "finite_training_graph_connectivity_diagnostic",
        "material_shortcut_rate": "known_hairpin_path_underestimation_diagnostic",
        "infinite_pair_rate": "finite_training_graph_disconnection_diagnostic",
        "mds_negative_eigenmass": "classical_mds_euclideanity_diagnostic",
        "diffusion_eigenspace_angle_max_degrees": "l2_pi_cosine_sine_eigenspace_diagnostic",
        "diffusion_truncation_relative_rmse": "full_finite_operator_diffusion_distance",
        "direct_spectral_distance_max_abs": "finite_operator_numerical_oracle",
        "heldout_extension_rmse": "heldout_cosine_sine_extension_after_train_alignment",
        "operator_row_sum_max_abs": "markov_operator_numerical_oracle",
    }[metric]


def graph_metric(scenario, method, metric, data, states, recovery, alignment):
    if scenario["domain"] == "hairpin":
        fit = states["fit"]; truth = data["test"]["truth"]
        if metric == "heldout_coordinate_rmse":
            if method == "mean": pred = np.full(len(truth), alignment["mean"])
            else:
                raw = recovery[{"pca":"pca", "ambient_mds":"ambient_cmds", "isomap":"isomap"}[method]]
                if raw is None: raise ValueError("disconnected_isomap_extension")
                pred = P3_GR.apply_affine(raw, alignment[method])
            return P3_GR.rmse(pred, truth)
        gm = P3_GR.graph_metrics(fit["adj"], fit["graph_dist"], data["train"]["truth"])
        if metric == "graph_distance_relative_rmse": return gm["graph_distance_relative_rmse"]
        if metric == "component_count": return gm["component_count"]
        if metric == "material_shortcut_rate": return gm["material_path_shortcut_rate"]
        if metric == "infinite_pair_rate":
            den = gm["finite_pair_count"] + gm["infinite_pair_count"]
            return gm["infinite_pair_count"]/max(den, 1)
        if metric == "mds_negative_eigenmass":
            model = fit["ambient_mds"] if method == "ambient_mds" else fit["isomap"]
            if model is None: raise ValueError("disconnected_isomap_mds")
            return model["negative_eigenmass"]
    model = states["model"]; theta = data["train"]["truth"]
    if metric == "diffusion_eigenspace_angle_max_degrees":
        truth = np.column_stack((np.cos(theta), np.sin(theta))); sp = np.sqrt(model["pi"])[:, None]
        return float(np.max(P3_GR.principal_angles(sp*model["psi"][:,1:3], sp*truth)))
    if metric == "diffusion_truncation_relative_rmse":
        full, trunc = P3_GR.diffusion_distances(model), P3_GR.diffusion_distances(model, 2); pairs = np.triu(np.ones(full.shape, bool), 1)
        return float(np.sqrt(np.mean((full[pairs]-trunc[pairs])**2))/max(np.sqrt(np.mean(full[pairs]**2)), 1e-15))
    if metric == "direct_spectral_distance_max_abs": return float(np.max(np.abs(P3_GR.diffusion_distances(model)-P3_GR.direct_diffusion_distances(model))))
    if metric == "operator_row_sum_max_abs": return float(np.max(np.abs(model["p"].sum(1)-1)))
    if metric == "heldout_extension_rmse":
        train_truth = np.column_stack((np.cos(theta), np.sin(theta))); modes = model["psi"][:,1:3]
        coef = np.linalg.lstsq(modes, train_truth, rcond=None)[0]
        pred = recovery["psi"][:,1:3] @ coef
        held = np.column_stack((np.cos(data["test"]["truth"]), np.sin(data["test"]["truth"])))
        return P3_GR.rmse(pred, held)
    raise KeyError((scenario["id"], method, metric))


# ----------------------------- row production -------------------------------
def base_row(task, scenario, replicate, method, metric):
    methods = REC_METHODS if task == "reconstruction" else GRAPH_METHODS
    family, source = next((fam, src) for mid, fam, src in methods if mid == method)
    sizes = scenario["sizes"] if task == "reconstruction" else graph_sizes(scenario)
    if task == "reconstruction":
        provenance = "validation" if metric == "selected_validation_mse" else "architecture_diagnostic" if metric == "parameter_count" else "test"
        split = "validation" if metric == "selected_validation_mse" else "diagnostic" if metric == "parameter_count" else "test"
        target = rec_target(metric); info = "train_validation_fit_frozen_test_recovery"
        selected = "two_seed_validation_selection" if method in ("nonlinear_autoencoder", "frozen_encoder") else "deterministic"
        domain = "observed_vectors"
    else:
        provenance = "test_extension" if metric in ("heldout_coordinate_rmse", "heldout_extension_rmse") else "train_diagnostic"
        split = "test" if provenance == "test_extension" else "train"; target = graph_target(metric)
        info = "frozen_train_graph_heldout_extension" if provenance == "test_extension" else "training_graph_operator_diagnostic"
        selected = f"k={scenario.get('k')}" if scenario["domain"] == "hairpin" else f"epsilon={scenario.get('epsilon')}"
        domain = scenario["domain"]
    key = f"{task}|{scenario['id']}|{replicate}|{method}|{metric}"
    return {"row_key": key, "task": task, "scenario": scenario["id"], "scenario_class": scenario["class"],
            "replicate": replicate, "seed_id": seed_id(task, scenario["id"], replicate), "provenance": provenance,
            "split": split, "evaluation_stage": provenance, "method": method, "method_family": family,
            "information_set": info, "parameter_source": source, "target": target, "metric": metric,
            "status": "inapplicable", "reason": "", "value": None, "n_train": sizes[0], "n_validation": sizes[1],
            "n_test": sizes[2], "selected_hyperparameter": selected, "domain": domain}


def set_value(row, value):
    value = float(value)
    if not np.isfinite(value): raise ValueError("nonfinite_scoring_value")
    row.update(status="ok", reason="", value=value)


def mark_failure(row, exc):
    row.update(status="nonconvergence" if isinstance(exc, np.linalg.LinAlgError) else "invalid", reason=f"{type(exc).__name__}:{exc}", value=None)


def evaluate_rec(scenario, replicate, fault=None):
    data = generate_rec(scenario, replicate)
    hidden = scenario.get("hidden", 8)
    fit_seed_scenario = "reference_curved" if scenario.get("shift") else scenario["id"]
    states = fit_rec_states(data["train"]["observed"], data["validation"]["observed"], fit_seed_scenario, replicate, hidden)
    recovery = recover_rec(states, data["test"]["observed"], fault)
    rows = []
    for method, _, _ in REC_METHODS:
        for metric in REC_METRICS:
            row = base_row("reconstruction", scenario, replicate, method, metric)
            applicable, reason = rec_applicable(method, metric, scenario)
            if not applicable: row["reason"] = reason
            else:
                try: set_value(row, rec_metric(method, metric, scenario, data, states, recovery))
                except Exception as exc: mark_failure(row, exc)
            rows.append(row)
    return rows, {"data": data, "states": states, "recovery": recovery}


def evaluate_graph(scenario, replicate, fault=None):
    data = generate_graph(scenario, replicate); states = fit_graph_states(data["train"]["observed"], scenario)
    alignment = fit_scoring_alignment(states, data["train"]["truth"])
    recovery_error = None
    try: recovery = recover_graph(states, data["test"]["observed"], fault)
    except Exception as exc: recovery, recovery_error = None, exc
    rows = []
    for method, _, _ in GRAPH_METHODS:
        for metric in GRAPH_METRICS:
            row = base_row("graph", scenario, replicate, method, metric)
            applicable, reason = graph_applicable(scenario, method, metric)
            if not applicable: row["reason"] = reason
            elif scenario["domain"] == "hairpin" and method == "isomap" and states["fit"]["isomap"] is None and metric in ("heldout_coordinate_rmse", "mds_negative_eigenmass"):
                row["reason"] = "isomap_inapplicable_for_disconnected_training_graph"
            elif recovery_error is not None and metric in ("heldout_coordinate_rmse", "heldout_extension_rmse"): mark_failure(row, recovery_error)
            else:
                try: set_value(row, graph_metric(scenario, method, metric, data, states, recovery, alignment))
                except Exception as exc: mark_failure(row, exc)
            rows.append(row)
    return rows, {"data": data, "states": states, "recovery": recovery, "alignment": alignment}


def all_fresh_rows(reverse=False):
    jobs = [("reconstruction", s, r) for s in REC_SCENARIOS for r in range(REPLICATES)] + [("graph", s, r) for s in GRAPH_SCENARIOS for r in range(REPLICATES)]
    if reverse: jobs.reverse()
    rows = []
    for task, scenario, rep in jobs:
        new, _ = evaluate_rec(scenario, rep) if task == "reconstruction" else evaluate_graph(scenario, rep)
        rows.extend(new)
    return sorted(rows, key=lambda r: r["row_key"])


def format_number(value): return "" if value is None else format(float(value), ".17g")


def serialized(row):
    out = {field: row[field] for field in FIELDS}; out["value"] = format_number(row["value"]); return out


def write_results(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n"); writer.writeheader()
        for row in rows: writer.writerow(serialized(row))


def read_results(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["replicate"] = int(row["replicate"]); row["n_train"] = int(row["n_train"]); row["n_validation"] = int(row["n_validation"]); row["n_test"] = int(row["n_test"])
        row["value"] = None if row["value"] == "" else float(row["value"])
    return rows


def validate_row(row):
    if set(row) != set(FIELDS): raise ValueError("incomplete_or_extra_fields")
    if row["status"] not in STATUS_VALUES: raise ValueError("unknown_status")
    if row["status"] == "ok":
        if row["reason"] or row["value"] is None or not np.isfinite(row["value"]): raise ValueError("invalid_ok_row")
    elif not row["reason"] or row["value"] is not None: raise ValueError("invalid_non_ok_row")
    task = row["task"]
    scenarios = REC_SCENARIOS if task == "reconstruction" else GRAPH_SCENARIOS
    methods = REC_METHODS if task == "reconstruction" else GRAPH_METHODS
    metrics = REC_METRICS if task == "reconstruction" else GRAPH_METRICS
    if row["scenario"] not in {s["id"] for s in scenarios} or row["method"] not in {m[0] for m in methods} or row["metric"] not in metrics: raise ValueError("unknown_row_identity")
    if row["row_key"] != f"{task}|{row['scenario']}|{row['replicate']}|{row['method']}|{row['metric']}": raise ValueError("row_key_mismatch")


def validate_resume(prior, fresh):
    expected = {r["row_key"]: r for r in fresh}; found = {}
    for row in prior:
        validate_row(row)
        if row["row_key"] in found: raise ValueError("duplicate_resume_key")
        if row["row_key"] not in expected: raise ValueError("unknown_resume_key")
        if serialized(row) != serialized(expected[row["row_key"]]): raise ValueError("stale_or_corrupted_resume_row")
        found[row["row_key"]] = row
    return found


def summarize(rows):
    groups = {}
    for row in rows: groups.setdefault((row["task"], row["scenario"], row["method"], row["metric"]), []).append(row)
    out = []
    for key, records in sorted(groups.items()):
        vals = np.array([r["value"] for r in records if r["status"] == "ok"], float); n = len(records)
        counts = {s: sum(r["status"] == s for r in records) for s in STATUS_VALUES}
        out.append({"task": key[0], "scenario": key[1], "method": key[2], "metric": key[3], "n": n, "ok_count": counts["ok"],
                    "median": None if not len(vals) else float(np.median(vals)), "q10": None if not len(vals) else float(np.quantile(vals,.1)),
                    "q90": None if not len(vals) else float(np.quantile(vals,.9)), "ok_rate": counts["ok"]/n,
                    "failure_rate": (counts["invalid"]+counts["nonconvergence"])/n, "applicability_rate": (n-counts["inapplicable"])/n,
                    "status_counts": counts})
    return out


# ------------------------------- diagnostics --------------------------------
def model_snapshot(states):
    arrays = []
    def walk(x):
        if isinstance(x, np.ndarray):
            arrays.append(np.nan_to_num(x.ravel(), nan=9e250, posinf=8e250, neginf=-8e250))
        elif isinstance(x, dict):
            for k in sorted(x):
                if k not in ("best_epoch",): walk(x[k])
        elif isinstance(x, (float, int, np.floating, np.integer)): arrays.append(np.array([x], float))
    walk(states); return np.concatenate(arrays) if arrays else np.zeros(1)


def capability_diagnostics():
    rs, rep = REC_SCENARIOS[0], 0; rd = generate_rec(rs, rep)
    fit1 = fit_rec_states(rd["train"]["observed"], rd["validation"]["observed"], rs["id"], rep, 8)
    poisoned_test = rd["test"]["observed"]*999+77
    fit2 = fit_rec_states(rd["train"]["observed"], rd["validation"]["observed"], rs["id"], rep, 8)
    rec1 = recover_rec(fit1, rd["test"]["observed"]); rec2 = recover_rec(fit1, rd["test"]["observed"])
    truth_poison = rd["test"]["truth"] + 999
    gs, grep = next(s for s in GRAPH_SCENARIOS if s["id"] == "hairpin_reference"), 0; gd = generate_graph(gs, grep)
    gf1 = fit_graph_states(gd["train"]["observed"], gs); gf2 = fit_graph_states(gd["train"]["observed"], gs)
    ge1 = recover_graph(gf1, gd["test"]["observed"])
    poisoned_truth = gd["test"]["truth"] + 1234
    ge2 = recover_graph(gf1, gd["test"]["observed"])
    return {
        "reconstruction_test_observation_poison_fit_max_abs": tree_max_abs(model_snapshot(fit1), model_snapshot(fit2)),
        "reconstruction_truth_poison_recovery_max_abs": tree_max_abs(rec1, rec2),
        "reconstruction_truth_poison_truth_change": float(np.max(np.abs(truth_poison-rd["test"]["truth"]))),
        "reconstruction_poisoned_test_not_passed_to_fit": True,
        "graph_heldout_observation_poison_fit_max_abs": tree_max_abs(model_snapshot(gf1), model_snapshot(gf2)),
        "graph_truth_poison_extension_max_abs": tree_max_abs(ge1, ge2),
        "graph_truth_poison_truth_change": float(np.max(np.abs(poisoned_truth-gd["test"]["truth"]))),
        "restricted_fit_signatures": {"reconstruction": ["train_observations", "validation_observations"], "graph": ["train_observations", "scenario_without_truth"]},
        "restricted_recovery_signatures": {"reconstruction": ["frozen_states", "test_observations"], "graph": ["frozen_states_including_training_anchors", "test_observations"]},
        "unused_poison_checksum": float(np.sum(poisoned_test)),
    }


def support_reuse_diagnostic():
    ref = REC_SCENARIOS[0]; shift = next(s for s in REC_SCENARIOS if s["id"] == "support_shift")
    a = generate_rec(ref, 0); b = generate_rec(shift, 0)
    sa = fit_rec_states(a["train"]["observed"], a["validation"]["observed"], ref["id"], 0, 8)
    sb = fit_rec_states(b["train"]["observed"], b["validation"]["observed"], ref["id"], 0, 8)
    return {"train_observations_exact": bool(np.array_equal(a["train"]["observed"], b["train"]["observed"])),
            "validation_observations_exact": bool(np.array_equal(a["validation"]["observed"], b["validation"]["observed"])),
            "fitted_state_max_abs": tree_max_abs(model_snapshot(sa), model_snapshot(sb))}


def graph_oracles():
    hs = next(s for s in GRAPH_SCENARIOS if s["id"] == "hairpin_reference"); hd = generate_graph(hs,0); hf = fit_graph_states(hd["train"]["observed"],hs)["fit"]
    adj, dist = hf["adj"], hf["graph_dist"]; finite = np.isfinite(dist)
    rng = rng_for("graph","oracle",0,90,"diagnostic"); tri = 0.0
    for _ in range(300):
        i,j,k = rng.integers(0,len(dist),3)
        if np.isfinite(dist[i,j]) and np.isfinite(dist[i,k]) and np.isfinite(dist[k,j]): tri=max(tri,float(dist[i,j]-dist[i,k]-dist[k,j]))
    replay = P3_GR.gower_extend(hf["ambient_dist"], hf["ambient_mds"])
    iso_replay = P3_GR.gower_extend(hf["graph_dist"], hf["isomap"])
    cs = next(s for s in GRAPH_SCENARIOS if s["id"]=="circle_reference"); cd=generate_graph(cs,0); cm=fit_graph_states(cd["train"]["observed"],cs)["model"]
    nys,_=P3_GR.nystrom(cm,cm["x"]); full=P3_GR.diffusion_distances(cm); direct=P3_GR.direct_diffusion_distances(cm)
    perm=rng_for("graph","permutation",0,91,"diagnostic").permutation(len(hd["train"]["observed"])); inv=np.argsort(perm)
    hp=P3_GR.fit_hairpin(hd["train"]["observed"][perm],k=hs["k"])
    cp=rng_for("graph","permutation",0,92,"diagnostic").permutation(len(cd["train"]["observed"])); cinv=np.argsort(cp)
    cmp=P3_GR.diffusion_fit(cd["train"]["observed"][cp],cs["epsilon"])
    q1=np.linalg.qr(cm["psi"][:,1:3])[0]; q2=np.linalg.qr(cmp["psi"][cinv,1:3])[0]
    return {
        "adjacency_symmetry_max_abs": float(np.max(np.abs(adj[np.isfinite(adj)]-adj.T[np.isfinite(adj)]))),
        "shortest_path_symmetry_max_abs": float(np.max(np.abs(dist[finite]-dist.T[finite]))),
        "shortest_path_diagonal_max_abs": float(np.max(np.abs(np.diag(dist)))),
        "sampled_triangle_positive_violation": max(0.0,tri),
        "ambient_mds_training_extension_replay_max_abs": float(np.max(np.abs(replay-hf["ambient_mds"]["coords"]))),
        "isomap_training_extension_replay_max_abs": float(np.max(np.abs(iso_replay-hf["isomap"]["coords"]))),
        "diffusion_training_nystrom_replay_max_abs": float(np.max(np.abs(nys[:,:3]-cm["psi"][:,:3]))),
        "diffusion_direct_spectral_max_abs": float(np.max(np.abs(full-direct))),
        "markov_row_sum_max_abs": float(np.max(np.abs(cm["p"].sum(1)-1))),
        "hairpin_relabel_graph_distance_max_abs": float(np.max(np.abs(hf["graph_dist"]-hp["graph_dist"][inv][:,inv]))),
        "circle_relabel_projector_max_abs": float(np.max(np.abs(q1@q1.T-q2@q2.T))),
    }


def circle_midpoint_diagnostics():
    records = []
    maximum = 0.0
    for scenario in GRAPH_SCENARIOS:
        if scenario["domain"] != "circle": continue
        for replicate in range(REPLICATES):
            data = generate_graph(scenario, replicate)
            expected = np.sort(circular_midpoints(data["train"]["truth"]))
            observed = np.sort(data["test"]["truth"])
            discrepancy = float(np.max(np.abs(expected-observed)))
            maximum = max(maximum, discrepancy)
            records.append({"scenario":scenario["id"],"replicate":replicate,"count":len(observed),"max_abs_error":discrepancy})
    return {"all_circle_test_points_are_training_grid_midpoints": maximum <= 1e-14,
            "maximum_midpoint_max_abs_error": maximum, "records": records}


def network_selection_diagnostics():
    records = []
    for scenario in REC_SCENARIOS:
        for replicate in range(REPLICATES):
            data = generate_rec(scenario, replicate)
            fit_seed_scenario = "reference_curved" if scenario.get("shift") else scenario["id"]
            states = fit_rec_states(data["train"]["observed"], data["validation"]["observed"], fit_seed_scenario,
                                    replicate, scenario.get("hidden", 8))
            for method in ("nonlinear_autoencoder", "frozen_encoder"):
                record = dict(states[method]["selection_record"])
                record.update({"scenario":scenario["id"],"replicate":replicate,"method":method,
                               "fit_seed_scenario":fit_seed_scenario})
                records.append(record)
    return {"record_count":len(records),"expected_record_count":len(REC_SCENARIOS)*REPLICATES*2,"records":records}


def provenance_role_diagnostics():
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in ("P3_NL", "P3_GR"):
                calls.append(f"{node.func.value.id}.{node.func.attr}")
    nonlinear_calls = sorted(x for x in calls if x.startswith("P3_NL."))
    graph_calls = sorted(set(x for x in calls if x.startswith("P3_GR.")))
    return {
        "nonlinear_source_role":"hash_only_design_provenance",
        "nonlinear_runtime_module_loaded": "P3_NL" in globals(),
        "nonlinear_ast_runtime_calls": nonlinear_calls,
        "graph_source_role":"actual_executable_routine_reuse",
        "graph_runtime_module_loaded": "P3_GR" in globals(),
        "graph_ast_runtime_calls": graph_calls,
        "graph_required_runtime_calls_present": all(name in graph_calls for name in (
            "P3_GR.fit_hairpin", "P3_GR.extend_hairpin", "P3_GR.diffusion_fit", "P3_GR.nystrom")),
    }


def metric_scaling_diagnostic():
    base = next(s for s in GRAPH_SCENARIOS if s["id"]=="hairpin_reference"); scaled=next(s for s in GRAPH_SCENARIOS if s["id"]=="hairpin_metric_scaling")
    a=generate_graph(base,0); b=generate_graph(scaled,0)
    fa=fit_graph_states(a["train"]["observed"],base)["fit"]; fb=fit_graph_states(b["train"]["observed"],scaled)["fit"]
    ea=np.isfinite(fa["adj"]); eb=np.isfinite(fb["adj"])
    return {"undirected_edge_symmetric_difference": int(np.sum(ea!=eb)//2), "transform": list(scaled["transform"])}


def fault_diagnostics():
    rr_nf,_=evaluate_rec(REC_SCENARIOS[0],0,"nonfinite"); rr_la,_=evaluate_rec(REC_SCENARIOS[0],0,"linalg")
    gr_nf,_=evaluate_graph(GRAPH_SCENARIOS[0],0,"nonfinite"); gr_la,_=evaluate_graph(GRAPH_SCENARIOS[0],0,"linalg")
    def counts(rows): return {s:sum(r["status"]==s for r in rows) for s in STATUS_VALUES}
    return {"reconstruction_nonfinite":counts(rr_nf),"reconstruction_linalg":counts(rr_la),"graph_nonfinite":counts(gr_nf),"graph_linalg":counts(gr_la),
            "applicability_precedes_failure": all(r["status"]=="inapplicable" for r in rr_nf+rr_la+gr_nf+gr_la if "not_applicable" in r["reason"] or r["reason"].startswith("no_scalar"))}


def verification_diagnostics():
    return {"capability_and_poison":capability_diagnostics(),"support_shift_fitted_reuse":support_reuse_diagnostic(),"graph_operator_oracles":graph_oracles(),
            "circle_midpoint_integrity":circle_midpoint_diagnostics(),"network_selection":network_selection_diagnostics(),
            "phase3_provenance_roles":provenance_role_diagnostics(),"metric_scaling":metric_scaling_diagnostic(),"fault_injection":fault_diagnostics()}


# -------------------------- artifacts and verifier --------------------------
def make_summary(path, summaries, checks):
    width,height=1280,820; im=Image.new("RGB",(width,height),"white"); d=ImageDraw.Draw(im); font=ImageFont.load_default()
    d.text((28,20),"P4-U05 Nonlinear Reconstruction and Graph Geometry (separate targets)",fill="black",font=font)
    panels=[
      ("Reconstruction: observed test MSE","reconstruction","reference_curved","observed_reconstruction_mse",["mean","pca","linear_autoencoder","nonlinear_autoencoder","frozen_encoder"]),
      ("Reconstruction: support-shift test MSE","reconstruction","support_shift","observed_reconstruction_mse",["mean","pca","nonlinear_autoencoder","frozen_encoder"]),
      ("Hairpin: held-out aligned coordinate RMSE","graph","hairpin_reference","heldout_coordinate_rmse",["mean","pca","ambient_mds","isomap"]),
      ("Circle: finite operator diagnostics","graph","circle_reference","diffusion_truncation_relative_rmse",["diffusion_maps"]),
    ]
    lookup={(s["task"],s["scenario"],s["method"],s["metric"]):s for s in summaries}
    for idx,(title,task,scenario,metric,methods) in enumerate(panels):
        x0=35+(idx%2)*620;y0=65+(idx//2)*365;d.rectangle((x0,y0,x0+580,y0+315),outline="#999");d.text((x0+12,y0+10),title,fill="black",font=font)
        vals=[lookup[(task,scenario,m,metric)]["median"] or 0 for m in methods]; vmax=max(vals+[1e-12])
        for j,(m,v) in enumerate(zip(methods,vals)):
            summary=lookup[(task,scenario,m,metric)]; yy=y0+50+j*48;bar=405*v/vmax
            d.rectangle((x0+125,yy,x0+125+bar,yy+20),fill="#4c78a8" if task=="reconstruction" else "#59a14f")
            d.text((x0+8,yy+4),m,fill="black",font=font)
            d.text((x0+132+bar,yy+4),f"{v:.4g} (n={summary['ok_count']}/{summary['n']})",fill="black",font=font)
        if idx==3:
            extras=[("eigenspace angle",lookup[(task,scenario,"diffusion_maps","diffusion_eigenspace_angle_max_degrees")]["median"]),
                    ("direct/spectral",lookup[(task,scenario,"direct_operator","direct_spectral_distance_max_abs")]["median"]),
                    ("Nyström extension",lookup[(task,scenario,"nystrom_extension","heldout_extension_rmse")]["median"])]
            extra_keys=[("diffusion_maps","diffusion_eigenspace_angle_max_degrees"),("direct_operator","direct_spectral_distance_max_abs"),("nystrom_extension","heldout_extension_rmse")]
            for j,((name,v),(method_name,metric_name)) in enumerate(zip(extras,extra_keys)):
                summary=lookup[(task,scenario,method_name,metric_name)]
                d.text((x0+18,y0+115+j*42),f"{name}: {v:.4g} (n={summary['ok_count']}/{summary['n']})",fill="black",font=font)
    d.text((35,790),"No pooled score or universal winner. Reconstruction is not manifold evidence; graph diagnostics are finite-sample/operator checks.",fill="#8b0000",font=font)
    im.save(path,optimize=False,compress_level=9)


def source_records():
    return [
        {"path":str(PHASE3_SOURCES[0].relative_to(REPO_ROOT)),"sha256":hashlib.sha256(PHASE3_SOURCES[0].read_bytes()).hexdigest(),"role":"hash_only_design_provenance_no_runtime_calls"},
        {"path":str(PHASE3_SOURCES[1].relative_to(REPO_ROOT)),"sha256":hashlib.sha256(PHASE3_SOURCES[1].read_bytes()).hexdigest(),"role":"actual_executable_routine_reuse"},
    ]


def expected_keys():
    out=[]
    for s in REC_SCENARIOS:
      for r in range(REPLICATES):
       for m,_,_ in REC_METHODS:
        for metric in REC_METRICS: out.append(f"reconstruction|{s['id']}|{r}|{m}|{metric}")
    for s in GRAPH_SCENARIOS:
      for r in range(REPLICATES):
       for m,_,_ in GRAPH_METHODS:
        for metric in GRAPH_METRICS: out.append(f"graph|{s['id']}|{r}|{m}|{metric}")
    return sorted(out)


def build(root, reverse=False, resume=False, quiet=False):
    root.mkdir(parents=True,exist_ok=True); fresh=all_fresh_rows(reverse); rows=fresh
    if resume and (root/"results.csv").exists():
        prior=read_results(root/"results.csv"); reused=validate_resume(prior,fresh); rows=[reused.get(r["row_key"],r) for r in fresh]
    write_results(root/"results.csv",rows); summaries=summarize(rows); checks=verification_diagnostics()
    status_counts={s:sum(r["status"]==s for r in rows) for s in STATUS_VALUES}
    diagnostics={"schema_version":1,"status_counts":status_counts,"row_count":len(rows),"summary_group_count":len(summaries),"summaries":summaries,"verification_checks":checks,
      "scientific_limits":["two tasks remain separate","three replicates yield descriptive quantiles only","bounded scenario grids are not factorial causal designs","reconstruction error and codes do not establish a manifold, topology, intrinsic dimension, representation, mechanism, or causality","graph paths and diffusion coordinates are finite graph/operator outputs, not proof of geodesic, Riemannian, manifold, or topological recovery","no cross-task score or universal winner"]}
    (root/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True,allow_nan=False)+"\n")
    make_summary(root/"summary.png",summaries,checks)
    hashes={n:sha256(root/n) for n in ARTIFACT_NAMES}
    manifest={"schema_version":1,"benchmark":"P4-U05 nonlinear reconstruction and graph geometry","maturity":"L1","status":"stable","master_seed":MASTER_SEED,"replicates":REPLICATES,
      "tasks":{"reconstruction":{"scenarios":list(REC_SCENARIOS),"methods":[m[0] for m in REC_METHODS],"metrics":list(REC_METRICS)},"graph_geometry":{"scenarios":list(GRAPH_SCENARIOS),"methods":[m[0] for m in GRAPH_METHODS],"metrics":list(GRAPH_METRICS)}},
      "stages":{"generation":"split-specific observations plus scoring-only truth","fitting_selection":"restricted train/validation observations for reconstruction; train observations for graph/operator fits","recovery":"frozen fitted states and held-out observations only","scoring":"truth and diagnostics added only after recovery"},
      "status_vocabulary":list(STATUS_VALUES),"row_fields":list(FIELDS),"row_count":len(rows),"canonical_order":"lexicographic row_key","summary":"replicate-level median,q10,q90 and status/applicability rates within task/scenario/method/metric only",
      "phase3_sources":source_records(),"artifact_files":["manifest.json",*ARTIFACT_NAMES],"artifact_hashes_sha256":hashes,
      "commands":{"generate":f"{sys.executable} benchmarks/nonlinear-graph-robustness.py --generate","verify":f"{sys.executable} benchmarks/nonlinear-graph-robustness.py --verify","resume":f"{sys.executable} benchmarks/nonlinear-graph-robustness.py --generate --resume"},
      "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"pillow":getattr(sys.modules.get("PIL"),"__version__","unknown"),"platform":platform.platform(),"executable":sys.executable},
      "interpretation_boundary":"No pooled score. Reconstruction, codes, graph distances, eigenspaces, and extensions do not establish manifold, topology, intrinsic dimension, unique coordinates, representation, dynamics, mechanism, causality, real-data validity, or universal superiority."}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if not quiet: print(f"wrote {root}\nrows={len(rows)} status={status_counts}")
    return manifest,diagnostics,rows


def validate_artifacts(root):
    errors=[]; expected={"manifest.json",*ARTIFACT_NAMES}; actual={p.name for p in root.iterdir() if p.is_file()}
    if actual!=expected: errors.append(f"artifact inventory mismatch {sorted(actual)}")
    try: manifest=json.loads((root/"manifest.json").read_text()); diag=json.loads((root/"diagnostics.json").read_text()); rows=read_results(root/"results.csv")
    except Exception as exc: return [f"artifact read failure:{exc}"]
    if tuple(rows[0].keys())!=FIELDS: errors.append("field order mismatch")
    keys=[]
    for row in rows:
        try: validate_row(row)
        except Exception as exc: errors.append(f"invalid row {row.get('row_key')}:{exc}")
        keys.append(row["row_key"])
    if keys!=sorted(keys): errors.append("noncanonical ordering")
    if len(keys)!=len(set(keys)): errors.append("duplicate keys")
    if keys!=expected_keys(): errors.append("Cartesian key mismatch")
    recomputed=summarize(rows)
    if recomputed!=diag.get("summaries"): errors.append("summary mismatch")
    counts={s:sum(r["status"]==s for r in rows) for s in STATUS_VALUES}
    if counts!=diag.get("status_counts"): errors.append("status count mismatch")
    for name in ARTIFACT_NAMES:
        if manifest.get("artifact_hashes_sha256",{}).get(name)!=sha256(root/name): errors.append(f"hash mismatch {name}")
    checks=diag.get("verification_checks",{}); cap=checks.get("capability_and_poison",{}); oracle=checks.get("graph_operator_oracles",{}); support=checks.get("support_shift_fitted_reuse",{}); scale=checks.get("metric_scaling",{}); fault=checks.get("fault_injection",{})
    midpoint=checks.get("circle_midpoint_integrity",{}); selection=checks.get("network_selection",{}); roles=checks.get("phase3_provenance_roles",{})
    for key in ("reconstruction_test_observation_poison_fit_max_abs","reconstruction_truth_poison_recovery_max_abs","graph_heldout_observation_poison_fit_max_abs","graph_truth_poison_extension_max_abs"):
        if cap.get(key,1)>1e-12: errors.append(f"poison failure {key}")
    if not support.get("train_observations_exact") or not support.get("validation_observations_exact") or support.get("fitted_state_max_abs",1)>1e-12: errors.append("support-shift fitted reuse failure")
    for key in ("adjacency_symmetry_max_abs","shortest_path_symmetry_max_abs","shortest_path_diagonal_max_abs","sampled_triangle_positive_violation","ambient_mds_training_extension_replay_max_abs","isomap_training_extension_replay_max_abs","diffusion_training_nystrom_replay_max_abs","diffusion_direct_spectral_max_abs","markov_row_sum_max_abs","hairpin_relabel_graph_distance_max_abs","circle_relabel_projector_max_abs"):
        if oracle.get(key,1)>2e-8: errors.append(f"oracle failure {key}")
    if scale.get("undirected_edge_symmetric_difference",0)<=0: errors.append("metric scaling did not change graph")
    if not midpoint.get("all_circle_test_points_are_training_grid_midpoints") or midpoint.get("maximum_midpoint_max_abs_error",1)>1e-14:
        errors.append("circle held-out midpoint integrity failure")
    expected_selection_count=len(REC_SCENARIOS)*REPLICATES*2
    records=selection.get("records",[])
    if selection.get("record_count")!=expected_selection_count or selection.get("expected_record_count")!=expected_selection_count or len(records)!=expected_selection_count:
        errors.append("network selection record count mismatch")
    expected_selection_keys={(s["id"],r,m) for s in REC_SCENARIOS for r in range(REPLICATES) for m in ("nonlinear_autoencoder","frozen_encoder")}
    observed_selection_keys=set()
    for record in records:
        key=(record.get("scenario"),record.get("replicate"),record.get("method")); observed_selection_keys.add(key)
        indices=record.get("candidate_indices",[]); losses=record.get("candidate_validation_losses",[]); seeds=record.get("candidate_seed_ids",[])
        selected=record.get("selected_candidate")
        best_epochs=record.get("candidate_best_epochs",[]); epochs_runs=record.get("candidate_epochs_run",[])
        max_epochs=record.get("candidate_max_epochs",[]); stop_reasons=record.get("candidate_stop_reasons",[])
        if not indices or not (len(indices)==len(losses)==len(seeds)==len(best_epochs)==len(epochs_runs)==len(max_epochs)==len(stop_reasons)):
            errors.append(f"incomplete network selection candidates {key}"); continue
        expected_selected=min(range(len(indices)),key=lambda i:(losses[i],indices[i]))
        if selected!=indices[expected_selected] or record.get("selected_validation_loss")!=losses[expected_selected] or record.get("selected_seed_id")!=seeds[expected_selected]:
            errors.append(f"network selection argmin/tie failure {key}")
        if record.get("selected_best_epoch")!=best_epochs[expected_selected] or record.get("selected_max_epoch")!=max_epochs[expected_selected] or record.get("selected_stop_reason")!=stop_reasons[expected_selected]:
            errors.append(f"selected network diagnostics mismatch {key}")
        for best_epoch,epochs_run,max_epoch,stop_reason in zip(best_epochs,epochs_runs,max_epochs,stop_reasons):
            if not (0 < best_epoch <= epochs_run <= max_epoch): errors.append(f"invalid epoch selection record {key}")
            if stop_reason not in ("patience","epoch_budget") or (stop_reason=="epoch_budget" and epochs_run!=max_epoch): errors.append(f"invalid stop reason {key}")
        if record.get("epoch_cap_semantics")!="diagnostic_only_not_automatic_nonconvergence": errors.append(f"epoch cap semantics mismatch {key}")
    if observed_selection_keys!=expected_selection_keys: errors.append("network selection key coverage mismatch")
    if roles.get("nonlinear_source_role")!="hash_only_design_provenance" or roles.get("nonlinear_runtime_module_loaded") or roles.get("nonlinear_ast_runtime_calls"):
        errors.append("nonlinear provenance role mismatch")
    if roles.get("graph_source_role")!="actual_executable_routine_reuse" or not roles.get("graph_runtime_module_loaded") or not roles.get("graph_required_runtime_calls_present"):
        errors.append("graph executable reuse role mismatch")
    source_roles={x.get("path"):x.get("role") for x in manifest.get("phase3_sources",[])}
    if source_roles.get("toy-models/bounded-nonlinear-reconstruction.py")!="hash_only_design_provenance_no_runtime_calls" or source_roles.get("toy-models/graph-geometry-recovery.py")!="actual_executable_routine_reuse":
        errors.append("manifest Phase 3 provenance roles mismatch")
    frozen_k={"hairpin_reference","hairpin_sparse_gap","hairpin_shortcuts","hairpin_nonuniform","hairpin_metric_scaling"}
    scenario_map={s["id"]:s for s in manifest.get("tasks",{}).get("graph_geometry",{}).get("scenarios",[])}
    if any(scenario_map.get(name,{}).get("k")!=6 for name in frozen_k): errors.append("nominal-related hairpin k freeze mismatch")
    if not fault.get("applicability_precedes_failure"): errors.append("failure/applicability precedence failure")
    if fault.get("reconstruction_nonfinite",{}).get("invalid",0)<=0 or fault.get("reconstruction_linalg",{}).get("nonconvergence",0)<=0 or fault.get("graph_nonfinite",{}).get("invalid",0)<=0 or fault.get("graph_linalg",{}).get("nonconvergence",0)<=0: errors.append("fault routing failure")
    return errors


def compare_roots(a,b): return [n for n in ("manifest.json",*ARTIFACT_NAMES) if (a/n).read_bytes()!=(b/n).read_bytes()]


def adversarial_resume(rows, root):
    errors=[]
    cases=[]
    corrupt=[dict(r) for r in rows[:40]]; corrupt[0]["value"] = 999 if corrupt[0]["status"]=="ok" else corrupt[0]["reason"]+"x"; cases.append(("corrupt",corrupt))
    duplicate=[dict(r) for r in rows[:40]]+[dict(rows[0])]; cases.append(("duplicate",duplicate))
    unknown=[dict(r) for r in rows[:40]]; unknown[0]["row_key"]="unknown"; cases.append(("unknown",unknown))
    incomplete=[dict(r) for r in rows[:40]]; incomplete[0].pop("target"); cases.append(("incomplete",incomplete))
    fresh=all_fresh_rows()
    for name,case in cases:
        try: validate_resume(case,fresh); errors.append(f"{name} resume accepted")
        except Exception: pass
    return errors


def verify(root):
    errors=validate_artifacts(root); rows=read_results(root/"results.csv")
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); clean=td/"clean"; reverse=td/"reverse"; resume=td/"resume"
        build(clean,quiet=True); build(reverse,reverse=True,quiet=True)
        resume.mkdir(); partial=rows[::7]; write_results(resume/"results.csv",partial); build(resume,resume=True,quiet=True)
        errors += [f"clean regeneration {x}" for x in compare_roots(root,clean)]
        errors += [f"reverse traversal {x}" for x in compare_roots(root,reverse)]
        errors += [f"partial resume {x}" for x in compare_roots(root,resume)]
        errors += adversarial_resume(rows,td)
    if errors: raise SystemExit("verification failed:\n- "+"\n- ".join(errors))
    print(f"verified {root}: {len(rows)} rows; byte-identical clean/reverse/resume; poison/oracle/fault checks PASS")


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--generate",action="store_true"); p.add_argument("--verify",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--reverse",action="store_true"); p.add_argument("--root",type=Path,default=DEFAULT_ROOT); return p.parse_args()


def main():
    args=parse_args(); do_generate=args.generate or not args.verify
    if do_generate: build(args.root,reverse=args.reverse,resume=args.resume)
    if args.verify or not args.generate: verify(args.root)


if __name__=="__main__": main()
