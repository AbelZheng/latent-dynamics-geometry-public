#!/usr/bin/env python3
"""Deterministic P4-U04 demixing and rotational-projection benchmark.

The two tasks have separate scenario grids, methods, targets, metrics, summaries,
and interpretation boundaries.  No cross-task aggregate is computed.
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

MASTER_SEED = 20260820
REPLICATES = 3
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = SCRIPT_DIR / "artifacts" / "demixing-rotational-robustness"
ARTIFACT_NAMES = ("results.csv", "diagnostics.json", "summary.png")
STATUS_VALUES = ("ok", "nonconvergence", "invalid", "inapplicable")
PHASE3_SOURCES = (
    REPO_ROOT / "toy-models" / "label-dependent-demixing.py",
    REPO_ROOT / "toy-models" / "rotational-projection-controls.py",
)

DEMIX_T, DEMIX_P = 12, 8
ROT_T, ROT_P, ROT_K, ROT_COND = 31, 8, 4, 6
ROT_TIME = np.linspace(0.0, 1.2, ROT_T)
ROT_DT = float(ROT_TIME[1] - ROT_TIME[0])

DEMIX_SCENARIOS = (
    {"id":"balanced_reference","class":"balanced_reference","counts":(20,10,12)},
    {"id":"near_collinear","class":"near_collinear_subspaces","counts":(20,10,12),"collinear":True},
    {"id":"missing_cell","class":"missing_unbalanced_cells","counts":(20,10,12),"missing":True},
    {"id":"unbalanced_cells","class":"missing_unbalanced_cells","counts":(20,10,12),"unbalanced":True},
    {"id":"label_shuffle","class":"label_shuffle","counts":(20,10,12),"shuffle_train":True},
    {"id":"pseudo_population","class":"pseudo_population","counts":(20,10,12),"pseudo":True},
    {"id":"averaging_mismatch","class":"averaging_mismatch","counts":(20,10,12),"trial_shift":True},
    {"id":"small_sample","class":"sample_size","counts":(5,4,12)},
    {"id":"large_sample","class":"sample_size","counts":(45,18,18)},
)
ROT_SCENARIOS = (
    {"id":"clean_rotation","class":"clean_rotation","kind":"rotation"},
    {"id":"symmetric_decay","class":"symmetric_decay","kind":"decay"},
    {"id":"latency_sequence","class":"latency","kind":"latency"},
    {"id":"common_input","class":"common_input","kind":"input"},
    {"id":"short_arcs","class":"short_arcs","kind":"rotation","window":12},
    {"id":"time_jitter","class":"time_jitter","kind":"rotation","jitter":True},
    {"id":"smoothed","class":"smoothing_subtraction","kind":"rotation","smooth":True},
    {"id":"no_condition_subtraction","class":"smoothing_subtraction","kind":"rotation","subtract":False},
    {"id":"nonorthogonal_metric","class":"nonorthogonal_metric","kind":"rotation","nonorthogonal":True},
    {"id":"covariance_control","class":"covariance_control","kind":"rotation","covariance_shuffle":True},
)

DEMIX_METHODS = (
    ("condition_mean","condition_mean","train_labels"),
    ("pca","pca","train_fit"),
    ("marginal_pca","marginal_pca","train_labels"),
    ("selected_dpca","regularized_dpca","train_validation"),
    ("ridge_decoder","decoder","train_validation"),
    ("shuffled_dpca_control","label_shuffle_control","permuted_train_labels"),
    ("shuffled_decoder_control","label_shuffle_control","permuted_train_validation_labels"),
    ("factorial_design_diagnostic","design_diagnostic","train_labels"),
)
DEMIX_METRICS = (
    "stimulus_marginal_nmse", "decision_marginal_nmse",
    "stimulus_subspace_angle_degrees", "decision_subspace_angle_degrees",
    "stimulus_decoding_accuracy", "decision_decoding_accuracy",
    "factorial_design_rank",
)
ROT_METHODS = (
    ("pca_plane","pca_plane","train_fit"),
    ("unrestricted_fit","unrestricted_derivative","train_fit"),
    ("skew_fit","skew_derivative","train_fit"),
    ("symmetric_fit","symmetric_derivative","train_fit"),
    ("input_aware_fit","input_aware_derivative","train_fit_known_input"),
    ("fixed_plane","fixed_plane","predeclared"),
    ("random_plane","random_plane","deterministic_seed"),
    ("covariance_shuffle_control","covariance_control","permuted_conditions"),
)
ROT_METRICS = (
    "plane_angle_degrees", "angular_rate_abs_error", "derivative_rmse",
    "tangential_velocity_ratio", "covariance_similarity",
)
FIELDS = (
    "row_key","task","scenario","scenario_class","replicate","seed_id","provenance",
    "split","evaluation_stage","method","method_family","information_set","parameter_source",
    "target","metric","status","reason","value",
    "train_cell_counts","validation_cell_counts","test_cell_counts",
    "n_train_total","n_validation_total","n_test_total","selected_hyperparameter",
    "design_rank","control_match_status","control_proposal",
)


def scenario_code(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def rng_for(task: str, scenario: str, replicate: int, stage: int, split: str="train", index: int=0):
    sc = {"train":1,"validation":2,"test":3,"diagnostic":4}[split]
    entropy=(MASTER_SEED, scenario_code(task), scenario_code(scenario), replicate, stage, sc, index)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def seed_id(task: str, scenario: str, replicate: int) -> str:
    return f"pcg64:{MASTER_SEED}:{scenario_code(task)}:{scenario_code(scenario)}:{replicate}"


def unit(x):
    x=np.asarray(x,float); n=np.linalg.norm(x)
    if not np.isfinite(n) or n <= 1e-14: raise ValueError("degenerate_vector")
    return x/n


def orth(x):
    q,_=np.linalg.qr(np.asarray(x,float)); return q


def angle_degrees(a,b):
    qa=orth(a); qb=orth(b); s=np.linalg.svd(qa.T@qb,compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(np.min(s),-1,1))))


def rel_frob(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def nmse(a,b):
    return float(np.mean((a-b)**2)/max(np.mean(b**2),1e-12))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demix_truth(scenario):
    t=np.linspace(-1,1,DEMIX_T)
    raw=np.array([[1,.2,-.4,.1,.5,-.2,.3,.1],[-.2,.9,.2,-.4,.1,.5,-.1,.3]],float).T
    if scenario.get("collinear"):
        raw[:,1]=unit(raw[:,0]+.12*raw[:,1])
    b=orth(raw) if not scenario.get("collinear") else np.column_stack([unit(raw[:,0]),unit(raw[:,1])])
    stim=np.exp(-((t+.2)/.42)**2); dec=np.tanh(2.2*t)
    time=.35*np.sin(np.pi*(t+1))[:,None]*unit(np.arange(1,DEMIX_P+1))[None,:]
    return {"stim_vec":b[:,0],"decision_vec":b[:,1],"stim_profile":stim,"decision_profile":dec,"time":time}


def shift_nonwrap(values, offset):
    """Shift along time with explicit nearest-edge fill; never circularly wrap."""
    values=np.asarray(values)
    out=np.empty_like(values)
    if offset==0:
        out[...] = values
    elif offset>0:
        out[:offset] = values[0]
        out[offset:] = values[:-offset]
    else:
        k=-offset
        out[-k:] = values[-1]
        out[:-k] = values[k:]
    return out


def factorial_design_rank(labels):
    labels=np.asarray(labels,int)
    s=2*labels[:,0]-1; d=2*labels[:,1]-1
    design=np.column_stack([np.ones(len(labels)),s,d,s*d])
    return int(np.linalg.matrix_rank(design))


def cell_counts_from_labels(labels):
    return np.array([[np.sum(np.all(labels==[s,d],axis=1)) for d in range(2)] for s in range(2)],int)


def compact_counts(counts):
    return json.dumps(np.asarray(counts,int).tolist(),separators=(",",":"))


def demix_counts(scenario, split):
    base=scenario["counts"][("train","validation","test").index(split)]
    counts=np.full((2,2),base,int)
    if scenario.get("missing") and split=="train": counts[1,1]=0
    if scenario.get("unbalanced") and split=="train": counts=np.array([[base*2,max(2,base//4)],[max(2,base//3),base]])
    return counts


def generate_demix_split(scenario, replicate, split):
    truth=demix_truth(scenario); obs=[]; labels=[]; clean=[]; offsets=[]
    counts=demix_counts(scenario,split)
    for s in range(2):
      for d in range(2):
        for j in range(int(counts[s,d])):
          ss=2*s-1; dd=2*d-1
          signal=truth["time"] + .9*ss*truth["stim_profile"][:,None]*truth["stim_vec"] + .75*dd*truth["decision_profile"][:,None]*truth["decision_vec"]
          if scenario.get("trial_shift"):
            shift=int(rng_for("demix",scenario["id"],replicate,9,split,j+20*s+10*d).choice((-1,0,1)))
            signal=shift_nonwrap(signal,shift)
          else:
            shift=0
          noise=rng_for("demix",scenario["id"],replicate,10,split,j+100*s+10*d).normal(scale=.42,size=signal.shape)
          obs.append(signal+noise); clean.append(signal); labels.append((s,d)); offsets.append(shift)
    obs=np.asarray(obs); clean=np.asarray(clean); labels=np.asarray(labels,int)
    if scenario.get("pseudo"):
      # Deterministic feature-wise trial reassignment destroys joint single-trial structure while preserving cell marginals.
      out=obs.copy()
      for s in range(2):
        for d in range(2):
          idx=np.flatnonzero(np.all(labels==[s,d],axis=1))
          for p in range(DEMIX_P):
            perm=idx[rng_for("demix",scenario["id"],replicate,11,split,100*s+10*d+p).permutation(len(idx))]
            out[idx,:,p]=obs[perm,:,p]
      obs=out
    return {"observed":obs,"labels":labels,"clean":clean,"truth":truth,"counts":counts,"time_offsets":np.asarray(offsets,int)}


def generate_demix(scenario,replicate):
    return {s:generate_demix_split(scenario,replicate,s) for s in ("train","validation","test")}


def condition_means(x,y):
    grand=x.mean(axis=0); out=np.empty((2,2,DEMIX_T,DEMIX_P)); available=np.zeros((2,2),bool)
    for s in range(2):
      for d in range(2):
        m=np.all(y==[s,d],axis=1); available[s,d]=np.any(m); out[s,d]=x[m].mean(axis=0) if np.any(m) else grand
    return out,available


def marginals(means):
    grand=means.mean(axis=(0,1),keepdims=True)
    stim=means.mean(axis=1,keepdims=True)-grand
    dec=means.mean(axis=0,keepdims=True)-grand
    return {"stimulus":np.repeat(stim,2,axis=1),"decision":np.repeat(dec,2,axis=0)}


def flatten_means(m): return m.reshape(-1,DEMIX_P)


def pca_basis(x,rank=2):
    xc=x-x.mean(axis=0); _,_,vt=np.linalg.svd(xc,full_matrices=False); return vt[:rank].T


def ridge_map(x,target,lam):
    return np.linalg.solve(x.T@x+lam*np.eye(x.shape[1]),x.T@target)


def reduced_rank_ridge_map(x,target,lam,rank=1):
    """Bounded dPCA-style ridge reduced-rank reconstruction map."""
    raw=ridge_map(x,target,lam)
    fitted=x@raw
    _,_,vt=np.linalg.svd(fitted,full_matrices=False)
    projector=vt[:rank].T@vt[:rank]
    return raw@projector


def fit_decoder(features, labels, lam):
    x=np.column_stack([np.ones(len(features)),features]); y=2*labels-1
    return np.linalg.solve(x.T@x+lam*np.eye(x.shape[1]),x.T@y)


def trial_features(x):
    thirds=np.array_split(np.arange(DEMIX_T),3)
    return np.concatenate([x[:,idx].mean(axis=1) for idx in thirds],axis=1)


def deterministic_permutation(n,scenario,replicate,stage):
    return rng_for("demix",scenario,replicate,stage,"diagnostic").permutation(n)


def fit_demix_states(scenario,replicate,train_obs,train_labels,val_obs,val_labels):
    """Lawful fitting/selection: accepts only train/validation observations and labels."""
    original_train_labels=np.asarray(train_labels).copy(); original_val_labels=np.asarray(val_labels).copy()
    train_labels=original_train_labels.copy()
    if scenario.get("shuffle_train"):
        train_labels=train_labels[deterministic_permutation(len(train_labels),scenario["id"],replicate,31)]
    design_rank=factorial_design_rank(original_train_labels)
    means,avail=condition_means(train_obs,train_labels); parts=marginals(means); x=flatten_means(means)
    global_basis=pca_basis(x,2)
    marginal_basis={k:pca_basis(flatten_means(v),1) for k,v in parts.items()}
    lambdas=(1e-4,.01,.1,1.0)
    val_means,_=condition_means(val_obs,original_val_labels); val_parts=marginals(val_means); val_x=flatten_means(val_means)
    candidates=[]
    for lam in lambdas:
      maps={k:reduced_rank_ridge_map(x,flatten_means(v),lam,1) for k,v in parts.items()}
      loss=sum(np.mean((val_x@maps[k]-flatten_means(val_parts[k]))**2) for k in parts)
      candidates.append((float(loss),lam,maps))
    _,selected_lambda,_=min(candidates,key=lambda z:(z[0],z[1]))
    dpca_maps={k:reduced_rank_ridge_map(x,flatten_means(v),selected_lambda,1) for k,v in parts.items()}

    features=trial_features(train_obs); vfeatures=trial_features(val_obs)
    decoders={}; decoder_lambdas={}; decoder_candidates={}
    for col,name in ((0,"stimulus"),(1,"decision")):
      vals=[]
      for lam in lambdas:
        w=fit_decoder(features,train_labels[:,col],lam); pred=(np.column_stack([np.ones(len(vfeatures)),vfeatures])@w>=0).astype(int)
        vals.append((float(np.mean(pred!=original_val_labels[:,col])),lam))
      decoder_candidates[name]=[{"validation_error":v[0],"lambda":v[1]} for v in vals]
      _,chosen=min(vals,key=lambda z:(z[0],z[1])); decoder_lambdas[name]=chosen
      decoders[name]=fit_decoder(features,train_labels[:,col],chosen)

    # Independent shuffled-control selection: train and validation labels use independent deterministic permutations.
    train_perm=deterministic_permutation(len(original_train_labels),scenario["id"],replicate,32)
    val_perm=deterministic_permutation(len(original_val_labels),scenario["id"],replicate,33)
    shuffled_train=original_train_labels[train_perm]; shuffled_val=original_val_labels[val_perm]
    smeans,_=condition_means(train_obs,shuffled_train); sparts=marginals(smeans); sx=flatten_means(smeans)
    svmeans,_=condition_means(val_obs,shuffled_val); svparts=marginals(svmeans); svx=flatten_means(svmeans)
    shuffled_candidates=[]
    for lam in lambdas:
      maps={k:reduced_rank_ridge_map(sx,flatten_means(v),lam,1) for k,v in sparts.items()}
      loss=sum(np.mean((svx@maps[k]-flatten_means(svparts[k]))**2) for k in sparts)
      shuffled_candidates.append((float(loss),lam))
    _,shuffled_lambda=min(shuffled_candidates,key=lambda z:(z[0],z[1]))
    shuffled_maps={k:reduced_rank_ridge_map(sx,flatten_means(v),shuffled_lambda,1) for k,v in sparts.items()}
    shuffled_decoders={}; shuffled_decoder_lambdas={}; shuffled_decoder_candidates={}
    for col,name in ((0,"stimulus"),(1,"decision")):
      vals=[]
      for lam in lambdas:
        w=fit_decoder(features,shuffled_train[:,col],lam)
        pred=(np.column_stack([np.ones(len(vfeatures)),vfeatures])@w>=0).astype(int)
        vals.append((float(np.mean(pred!=shuffled_val[:,col])),lam))
      shuffled_decoder_candidates[name]=[{"validation_error":v[0],"lambda":v[1]} for v in vals]
      _,chosen=min(vals,key=lambda z:(z[0],z[1])); shuffled_decoder_lambdas[name]=chosen
      shuffled_decoders[name]=fit_decoder(features,shuffled_train[:,col],chosen)
    return {"means":means,"available":avail,"parts":parts,"global_basis":global_basis,"marginal_basis":marginal_basis,
            "dpca_maps":dpca_maps,"selected_lambda":selected_lambda,"dpca_candidates":[{"validation_loss":v[0],"lambda":v[1]} for v in candidates],
            "decoders":decoders,"decoder_lambdas":decoder_lambdas,"decoder_candidates":decoder_candidates,
            "shuffled_maps":shuffled_maps,"shuffled_selected_lambda":shuffled_lambda,
            "shuffled_dpca_candidates":[{"validation_loss":v[0],"lambda":v[1]} for v in shuffled_candidates],
            "shuffled_decoders":shuffled_decoders,"shuffled_decoder_lambdas":shuffled_decoder_lambdas,
            "shuffled_decoder_candidates":shuffled_decoder_candidates,
            "design_rank":design_rank,"train_cell_counts":cell_counts_from_labels(original_train_labels),
            "validation_cell_counts":cell_counts_from_labels(original_val_labels)}

def recover_demix(states,test_obs,test_labels,fault=None):
    """Frozen recovery: accepts fitted states and test observations/labels only."""
    means,_=condition_means(test_obs,test_labels); x=flatten_means(means)
    recon={
      "condition_mean":marginals(states["means"]),
      "pca":{},"marginal_pca":{},"selected_dpca":{},"shuffled_dpca_control":{},
    }
    test_parts=marginals(means)
    for k in ("stimulus","decision"):
      target=flatten_means(test_parts[k]); b=states["global_basis"]
      recon["pca"][k]=(target@b@b.T).reshape(2,2,DEMIX_T,DEMIX_P)
      mb=states["marginal_basis"][k]
      recon["marginal_pca"][k]=(target@mb@mb.T).reshape(2,2,DEMIX_T,DEMIX_P)
      recon["selected_dpca"][k]=(x@states["dpca_maps"][k]).reshape(2,2,DEMIX_T,DEMIX_P)
      recon["shuffled_dpca_control"][k]=(x@states["shuffled_maps"][k]).reshape(2,2,DEMIX_T,DEMIX_P)
    feats=trial_features(test_obs); aug=np.column_stack([np.ones(len(feats)),feats])
    decoding={"ridge_decoder":{},"shuffled_decoder_control":{}}
    for k,col in (("stimulus",0),("decision",1)):
      decoding["ridge_decoder"][k]=(aug@states["decoders"][k]>=0).astype(int)
      decoding["shuffled_decoder_control"][k]=(aug@states["shuffled_decoders"][k]>=0).astype(int)
    if fault=="nonfinite": recon["selected_dpca"]["stimulus"][0,0,0,0]=np.nan
    if fault=="linalg": raise np.linalg.LinAlgError("injected_demix_failure")
    return {"reconstruction":recon,"decoding":decoding,"test_parts":test_parts}


def demix_applicable(method,metric):
    if metric=="factorial_design_rank": return method=="factorial_design_diagnostic"
    if metric.endswith("decoding_accuracy"): return method in ("ridge_decoder","shuffled_decoder_control")
    if "subspace_angle" in metric: return method in ("pca","marginal_pca","selected_dpca","shuffled_dpca_control")
    return method in ("condition_mean","pca","marginal_pca","selected_dpca","shuffled_dpca_control")


def requires_complete_factorial(method,metric):
    return metric!="factorial_design_rank" and not metric.endswith("decoding_accuracy") and demix_applicable(method,metric)


def demix_target(metric):
    if metric=="factorial_design_rank": return "factorial_design_identification"
    label="stimulus" if metric.startswith("stimulus") else "decision"
    if metric.endswith("decoding_accuracy"): return f"{label}_label_decoding"
    if "subspace_angle" in metric: return f"{label}_marginal_subspace"
    return f"{label}_marginal_reconstruction"


def demix_metric_value(method,metric,recovery,truth,test_labels,states):
    if metric=="factorial_design_rank": return float(states["design_rank"])
    label="stimulus" if metric.startswith("stimulus") else "decision"
    if metric.endswith("decoding_accuracy"):
      col=0 if label=="stimulus" else 1
      return float(np.mean(recovery["decoding"][method][label]==test_labels[:,col]))
    est=recovery["reconstruction"][method][label]; target=recovery["test_parts"][label]
    if not (np.all(np.isfinite(est)) and np.all(np.isfinite(target))): raise ValueError("nonfinite_reconstruction")
    if metric.endswith("marginal_nmse"): return nmse(est,target)
    truevec=truth["stim_vec"] if label=="stimulus" else truth["decision_vec"]
    matrix=flatten_means(est)
    return angle_degrees(pca_basis(matrix,1),truevec[:,None])


def rot_embedding(nonorthogonal=False):
    h=orth(np.array([[1,.2,-.1,.3],[.2,1,.3,-.2],[-.4,.1,1,.2],[.3,-.2,.1,1],[.5,.2,-.3,.1],[-.2,.4,.2,.3],[.1,-.5,.4,.2],[.3,.3,.2,-.4]],float))
    if nonorthogonal: h=h@np.array([[1.8,.4,0,0],[0,.55,.2,0],[0,0,1.4,.3],[.2,0,0,.7]])
    return h


def rot_matrix(kind):
    if kind=="rotation": return np.array([[0,-4,0,0],[4,0,0,0],[0,0,-.25,-2.2],[0,0,2.2,-.25]],float)
    if kind=="decay": return np.diag([-1.2,-.8,-.45,-.2])
    return np.zeros((ROT_K,ROT_K))


def initial_states():
    theta=np.linspace(0,2*np.pi,ROT_COND,endpoint=False)
    return np.column_stack([1.2*np.cos(theta),1.2*np.sin(theta),.7*np.cos(2*theta),.7*np.sin(2*theta)])


def trajectory(scenario,condition):
    kind=scenario["kind"]; t=ROT_TIME; z0=initial_states()[condition]
    if kind in ("rotation","decay"):
      a=rot_matrix(kind); vals,vecs=np.linalg.eig(a); inv=np.linalg.inv(vecs)
      z=np.real(np.stack([vecs@np.diag(np.exp(vals*tt))@inv@z0 for tt in t]))
      u=np.zeros((ROT_T,2))
    elif kind=="latency":
      centers=np.array([.20,.36,.54,.72])+condition*.018
      z=np.stack([np.exp(-((t-c)/.11)**2) for c in centers],axis=1); u=np.zeros((ROT_T,2))
    else:
      u=np.column_stack([np.sin(2*np.pi*t+condition*.35),np.cos(np.pi*t-condition*.2)])
      b=np.array([[1,0],[0,1],[.7,.2],[-.2,.8]])
      z=np.zeros((ROT_T,ROT_K)); z[0]=z0
      for i in range(ROT_T-1): z[i+1]=z[i]+ROT_DT*(-.65*z[i]+b@u[i])
    offset=(condition%3)-1 if scenario.get("jitter") else 0
    if offset:
      z=shift_nonwrap(z,offset); u=shift_nonwrap(u,offset)
    return z,u,offset


def generate_rot_split(scenario,replicate,split):
    h=rot_embedding(scenario.get("nonorthogonal",False)); n={"train":7,"validation":4,"test":5}[split]
    observed=np.empty((ROT_COND,n,ROT_T,ROT_P)); latent=np.empty((ROT_COND,n,ROT_T,ROT_K)); inputs=np.empty((ROT_COND,n,ROT_T,2)); offsets=[]
    for c in range(ROT_COND):
      z,u,offset=trajectory(scenario,c); offsets.append(offset)
      for j in range(n):
        zz=z+rng_for("rotation",scenario["id"],replicate,10,split,c*20+j).normal(scale=.035,size=z.shape)
        y=zz@h.T+rng_for("rotation",scenario["id"],replicate,11,split,c*20+j).normal(scale=.07,size=(ROT_T,ROT_P))
        observed[c,j]=y; latent[c,j]=zz; inputs[c,j]=u
    return {"observed":observed,"latent":latent,"inputs":inputs,"embedding":h,"time_offsets":np.asarray(offsets,int)}


def generate_rotation(scenario,replicate): return {s:generate_rot_split(scenario,replicate,s) for s in ("train","validation","test")}


def moving_average(x):
    out=x.copy(); out[...,1:-1,:]=(x[...,:-2,:]+x[...,1:-1,:]+x[...,2:,:])/3; return out


def preprocess_rot_fit(train,scenario):
    means=train.mean(axis=1)
    if scenario.get("smooth"): means=moving_average(means)
    time_mean=means.mean(axis=0,keepdims=True) if scenario.get("subtract",True) else np.zeros((1,ROT_T,ROT_P))
    centered=means-time_mean
    window=scenario.get("window",ROT_T); centered=centered[:,:window]
    flat=centered.reshape(-1,ROT_P); mean=flat.mean(axis=0); basis=pca_basis(flat,ROT_K)
    return {"feature_mean":mean,"time_mean":time_mean[:,:window],"basis":basis,"window":window,"smooth":scenario.get("smooth",False)}


def preprocess_rot_apply(obs,fit):
    means=obs.mean(axis=1)
    if fit["smooth"]: means=moving_average(means)
    means=means[:,:fit["window"]]-fit["time_mean"]
    return (means-fit["feature_mean"])@fit["basis"]


def derivative_pairs(scores):
    d=(scores[:,2:]-scores[:,:-2])/(2*ROT_DT); x=scores[:,1:-1]
    return x.reshape(-1,ROT_K),d.reshape(-1,ROT_K)


def constrained_map(x,d,kind):
    m=np.linalg.lstsq(x,d,rcond=None)[0].T
    if kind=="unrestricted": return m
    if kind=="skew": return .5*(m-m.T)
    if kind=="symmetric": return .5*(m+m.T)
    raise KeyError(kind)


def skew_plane(m):
    s=.5*(m-m.T); vals,vecs=np.linalg.eig(s.astype(complex)); idx=int(np.argmax(np.abs(np.imag(vals))))
    v=vecs[:,idx]; plane=orth(np.column_stack([np.real(v),np.imag(v)])); rate=float(abs(np.imag(vals[idx])))
    return plane,rate


COVARIANCE_MATCH_THRESHOLD = 0.97
COVARIANCE_PROPOSALS = 24


def covariance_similarity(left,right):
    a=np.cov(np.asarray(left).reshape(-1,ROT_K),rowvar=False)
    b=np.cov(np.asarray(right).reshape(-1,ROT_K),rowvar=False)
    return float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))


def covariance_control_proposals(scores,scenario_id,replicate):
    """Predeclared bounded shifts, excluding global common relabelings/no-ops."""
    proposals=[]
    for proposal in range(COVARIANCE_PROPOSALS):
      candidate=scores.copy(); shifts=[]
      for p in range(ROT_K):
        shift=int(rng_for("rotation",scenario_id,replicate,46,"diagnostic",proposal*ROT_K+p).integers(1,ROT_COND))
        order=np.concatenate([np.arange(shift,ROT_COND),np.arange(shift)])
        candidate[:,:,p]=scores[order,:,p]; shifts.append(shift)
      # A uniform shift is only one common condition relabeling and leaves cross-coordinate correspondence unchanged.
      if len(set(shifts)) < 2:
        continue
      pair_count=ROT_K*(ROT_K-1)//2
      changed_pairs=sum(shifts[left]!=shifts[right] for left in range(ROT_K) for right in range(left+1,ROT_K))
      proposals.append({"index":proposal,"scores":candidate,"feature_condition_shifts":shifts,
                        "distinct_shift_count":len(set(shifts)),
                        "cross_coordinate_correspondence_change_fraction":changed_pairs/pair_count,
                        "selected_scores_max_abs_change":float(np.max(np.abs(candidate-scores))),
                        "covariance_similarity":covariance_similarity(scores,candidate)})
    if not proposals:
      raise RuntimeError("no_admissible_covariance_control_proposals")
    return proposals


def fit_rotation_states(scenario,replicate,train_obs,val_obs,train_inputs,val_inputs):
    """Lawful train fitting; validation is available only for declared selection checks."""
    prep=preprocess_rot_fit(train_obs,scenario); train_scores=preprocess_rot_apply(train_obs,prep); x,d=derivative_pairs(train_scores)
    unrestricted=constrained_map(x,d,"unrestricted"); skew=constrained_map(x,d,"skew"); symmetric=constrained_map(x,d,"symmetric")
    inp=train_inputs.mean(axis=1)[:,:prep["window"]][:,1:-1].reshape(-1,2)
    design=np.column_stack([x,inp]); coeff=np.linalg.lstsq(design,d,rcond=None)[0].T; input_state=coeff[:,:ROT_K]; input_coef=coeff[:,ROT_K:]
    pca_plane=np.eye(ROT_K)[:,:2]; up,_=skew_plane(unrestricted); sp,srate=skew_plane(skew)
    fixed=np.eye(ROT_K)[:,[0,2]]
    random=orth(rng_for("rotation",scenario["id"],replicate,45,"diagnostic").normal(size=(ROT_K,2)))
    proposals=covariance_control_proposals(train_scores,scenario["id"],replicate)
    selected=max(proposals,key=lambda q:(q["covariance_similarity"],-q["index"]))
    sx,sd=derivative_pairs(selected["scores"]); cm=constrained_map(sx,sd,"skew"); cp,crate=skew_plane(cm)
    match_status="matched" if selected["covariance_similarity"]>=COVARIANCE_MATCH_THRESHOLD else "incompletely_matched"
    return {"prep":prep,"unrestricted":unrestricted,"skew":skew,"symmetric":symmetric,"input_state":input_state,"input_coef":input_coef,
            "planes":{"pca_plane":pca_plane,"unrestricted_fit":up,"skew_fit":sp,"fixed_plane":fixed,"random_plane":random,"covariance_shuffle_control":cp},
            "rates":{"unrestricted_fit":skew_plane(unrestricted)[1],"skew_fit":srate,"covariance_shuffle_control":crate},
            "covariance_similarity":selected["covariance_similarity"],"covariance_match_status":match_status,
            "covariance_selected_proposal":selected["index"],"covariance_proposals":[{"index":q["index"],"feature_condition_shifts":q["feature_condition_shifts"],
              "distinct_shift_count":q["distinct_shift_count"],"cross_coordinate_correspondence_change_fraction":q["cross_coordinate_correspondence_change_fraction"],
              "selected_scores_max_abs_change":q["selected_scores_max_abs_change"],"covariance_similarity":q["covariance_similarity"]} for q in proposals]}

def recover_rotation(states,test_obs,test_inputs,fault=None):
    scores=preprocess_rot_apply(test_obs,states["prep"]); x,d=derivative_pairs(scores); inp=test_inputs.mean(axis=1)[:,:states["prep"]["window"]][:,1:-1].reshape(-1,2)
    predictions={
      "unrestricted_fit":x@states["unrestricted"].T,
      "skew_fit":x@states["skew"].T,
      "symmetric_fit":x@states["symmetric"].T,
      "input_aware_fit":x@states["input_state"].T+inp@states["input_coef"].T,
    }
    if fault=="nonfinite": predictions["skew_fit"][0,0]=np.nan
    if fault=="linalg": raise np.linalg.LinAlgError("injected_rotation_failure")
    return {"scores":scores,"x":x,"derivative":d,"inputs":inp,"predictions":predictions}


def rot_applicable(scenario,method,metric):
    plane_methods=("pca_plane","unrestricted_fit","skew_fit","fixed_plane","random_plane","covariance_shuffle_control")
    if metric=="derivative_rmse": return method in ("unrestricted_fit","skew_fit","symmetric_fit","input_aware_fit")
    if metric=="tangential_velocity_ratio": return method in plane_methods
    if metric=="covariance_similarity": return method=="covariance_shuffle_control"
    if metric=="plane_angle_degrees":
      return scenario["kind"]=="rotation" and not scenario.get("covariance_shuffle") and method in plane_methods
    if metric=="angular_rate_abs_error":
      return scenario["kind"]=="rotation" and not scenario.get("covariance_shuffle") and method in ("unrestricted_fit","skew_fit")
    return False


def rot_target(metric):
    return {"plane_angle_degrees":"rotational_plane","angular_rate_abs_error":"angular_rate",
            "derivative_rmse":"heldout_derivative_prediction","tangential_velocity_ratio":"plane_tangential_motion",
            "covariance_similarity":"condition_covariance_control"}[metric]


def rot_metric_value(scenario,method,metric,states,recovery,test):
    if metric=="derivative_rmse": return float(np.sqrt(np.mean((recovery["predictions"][method]-recovery["derivative"])**2)))
    if metric=="covariance_similarity": return float(states["covariance_similarity"])
    plane=states["planes"][method]; obs_plane=states["prep"]["basis"]@plane
    if metric=="plane_angle_degrees":
      truth=test["embedding"][:,:2]; return angle_degrees(obs_plane,truth)
    if metric=="angular_rate_abs_error": return abs(float(states["rates"].get(method,0.0))-4.0)
    coords=recovery["scores"]@plane; dc=np.diff(coords,axis=1)/ROT_DT; pos=coords[:,:-1]
    radial=np.sum(dc*pos,axis=2); tangent=dc-radial[:,:,None]*pos/np.maximum(np.sum(pos*pos,axis=2)[:,:,None],1e-12)
    return float(np.linalg.norm(tangent)/max(np.linalg.norm(dc),1e-12))


def split_metadata(task,scenario):
    if task=="demixing":
      counts={split:demix_counts(scenario,split) for split in ("train","validation","test")}
    else:
      counts={"train":np.full(ROT_COND,7,int),"validation":np.full(ROT_COND,4,int),"test":np.full(ROT_COND,5,int)}
    return {"train_cell_counts":compact_counts(counts["train"]),"validation_cell_counts":compact_counts(counts["validation"]),
            "test_cell_counts":compact_counts(counts["test"]),"n_train_total":int(np.sum(counts["train"])),
            "n_validation_total":int(np.sum(counts["validation"])),"n_test_total":int(np.sum(counts["test"]))}


def base_row(task,scenario,replicate,method,metric):
    if task=="demixing":
      entry=next(x for x in DEMIX_METHODS if x[0]==method); target=demix_target(metric)
      info=("design_diagnostic" if method=="factorial_design_diagnostic" else
            "permuted_train_validation_labels" if method.startswith("shuffled_") else
            "supervised_train_validation" if method in ("selected_dpca","ridge_decoder") else "supervised_train")
      if metric=="factorial_design_rank": provenance,split,stage="training_design", "design_diagnostic", "identification_diagnostic"
      else: provenance,split,stage="heldout_test", "test", "heldout_scoring"
    else:
      entry=next(x for x in ROT_METHODS if x[0]==method); target=rot_target(metric)
      info=("fixed_reference" if method in ("fixed_plane","random_plane") else
            "selected_permuted_condition_control" if method=="covariance_shuffle_control" else
            "train_known_input" if method=="input_aware_fit" else "train_fitted_projection")
      if metric in ("plane_angle_degrees","angular_rate_abs_error"):
        provenance, split, stage="training_fit_against_known_truth", "fit_diagnostic", "parameter_recovery"
      elif metric=="covariance_similarity":
        provenance, split, stage="training_selected_permutation_control", "permutation_diagnostic", "control_diagnostic"
      else:
        provenance, split, stage="heldout_test", "test", "heldout_scoring"
    row={"row_key":f"{task}|{scenario['id']}|{replicate}|{method}|{metric}","task":task,"scenario":scenario["id"],"scenario_class":scenario["class"],
      "replicate":replicate,"seed_id":seed_id(task,scenario["id"],replicate),"provenance":provenance,"split":split,"evaluation_stage":stage,
      "method":method,"method_family":entry[1],"information_set":info,"parameter_source":entry[2],"target":target,"metric":metric,
      "status":"inapplicable","reason":"method_metric_inapplicable","value":"","selected_hyperparameter":"","design_rank":"",
      "control_match_status":"","control_proposal":""}
    row.update(split_metadata(task,scenario)); return row


def attach_method_metadata(row,states):
    method=row["method"]
    if method=="selected_dpca": row["selected_hyperparameter"]=format(float(states["selected_lambda"]),".12g")
    elif method=="ridge_decoder":
      label="stimulus" if row["metric"].startswith("stimulus") else "decision"
      row["selected_hyperparameter"]=format(float(states["decoder_lambdas"][label]),".12g")
    elif method=="shuffled_dpca_control": row["selected_hyperparameter"]=format(float(states["shuffled_selected_lambda"]),".12g")
    elif method=="shuffled_decoder_control":
      label="stimulus" if row["metric"].startswith("stimulus") else "decision"
      row["selected_hyperparameter"]=format(float(states["shuffled_decoder_lambdas"][label]),".12g")
    if "design_rank" in states: row["design_rank"]=str(states["design_rank"])
    if "covariance_match_status" in states and method=="covariance_shuffle_control":
      row["control_match_status"]=states["covariance_match_status"]
      row["control_proposal"]=str(states["covariance_selected_proposal"])

def set_value(row,value):
    v=float(value)
    if not np.isfinite(v): raise ValueError("nonfinite_score")
    row.update(status="ok",reason="",value=format(v,".12g"))


def mark_failure(row,exc):
    status="nonconvergence" if isinstance(exc,np.linalg.LinAlgError) else "invalid"
    row.update(status=status,reason=f"{type(exc).__name__}:{str(exc)[:80]}",value="")


def evaluate_demix(scenario,replicate,fault=None):
    data=generate_demix(scenario,replicate)
    states=fit_demix_states(scenario,replicate,data["train"]["observed"],data["train"]["labels"],data["validation"]["observed"],data["validation"]["labels"])
    try: recovery=recover_demix(states,data["test"]["observed"],data["test"]["labels"],fault)
    except Exception as exc: recovery=None; recovery_exc=exc
    rows=[]
    for method,_,_ in DEMIX_METHODS:
      for metric in DEMIX_METRICS:
        row=base_row("demixing",scenario,replicate,method,metric); attach_method_metadata(row,states)
        if demix_applicable(method,metric):
          if states["design_rank"]<4 and requires_complete_factorial(method,metric):
            row.update(status="invalid",reason=f"incomplete_factorial_design_rank_{states['design_rank']}_of_4",value="")
          elif recovery is None: mark_failure(row,recovery_exc)
          else:
            try: set_value(row,demix_metric_value(method,metric,recovery,data["test"]["truth"],data["test"]["labels"],states))
            except Exception as exc: mark_failure(row,exc)
        rows.append(row)
    return rows,states


def evaluate_rotation(scenario,replicate,fault=None):
    data=generate_rotation(scenario,replicate)
    states=fit_rotation_states(scenario,replicate,data["train"]["observed"],data["validation"]["observed"],data["train"]["inputs"],data["validation"]["inputs"])
    try: recovery=recover_rotation(states,data["test"]["observed"],data["test"]["inputs"],fault)
    except Exception as exc: recovery=None; recovery_exc=exc
    rows=[]
    for method,_,_ in ROT_METHODS:
      for metric in ROT_METRICS:
        row=base_row("rotation",scenario,replicate,method,metric); attach_method_metadata(row,states)
        if rot_applicable(scenario,method,metric):
          if recovery is None: mark_failure(row,recovery_exc)
          else:
            try: set_value(row,rot_metric_value(scenario,method,metric,states,recovery,data["test"]))
            except Exception as exc: mark_failure(row,exc)
        rows.append(row)
    return rows,states


def all_fresh_rows(reverse=False):
    jobs=[("demixing",s,r) for s in DEMIX_SCENARIOS for r in range(REPLICATES)]+[("rotation",s,r) for s in ROT_SCENARIOS for r in range(REPLICATES)]
    if reverse: jobs=list(reversed(jobs))
    rows=[]
    for task,s,r in jobs:
      fresh,_=evaluate_demix(s,r) if task=="demixing" else evaluate_rotation(s,r); rows.extend(fresh)
    return sorted(rows,key=lambda x:x["row_key"])


def write_results(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n"); w.writeheader(); w.writerows(rows)


def read_results(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))


def validate_row(row):
    if set(row)!=set(FIELDS): raise ValueError("incomplete_fields")
    expected=f"{row['task']}|{row['scenario']}|{row['replicate']}|{row['method']}|{row['metric']}"
    if row["row_key"]!=expected: raise ValueError("stale_or_unknown_key")
    if row["status"] not in STATUS_VALUES: raise ValueError("unknown_status")
    if row["status"]=="ok":
      if row["reason"] or not np.isfinite(float(row["value"])): raise ValueError("invalid_ok_row")
    elif not row["reason"] or row["value"]!="": raise ValueError("invalid_failure_row")


def csv_canonical(row):
    return {k: ("" if row[k] == "" else str(row[k])) for k in FIELDS}

def validate_resume(prior,fresh):
    expected={r["row_key"]:csv_canonical(r) for r in fresh}; out={}
    for row in prior:
      validate_row(row); key=row["row_key"]
      if key in out: raise ValueError("duplicate_resume_key")
      if key not in expected: raise ValueError("unknown_resume_key")
      if csv_canonical(row)!=expected[key]: raise ValueError("stale_or_corrupted_resume_row")
      out[key]=row
    return out


def summaries(rows):
    groups={}
    for r in rows:
      key=(r["task"],r["scenario"],r["method"],r["metric"]); groups.setdefault(key,[]).append(r)
    out=[]
    for key,vals in sorted(groups.items()):
      ok=[float(v["value"]) for v in vals if v["status"]=="ok"]
      out.append({"task":key[0],"scenario":key[1],"method":key[2],"metric":key[3],"n":len(vals),"ok":len(ok),
        "q10":float(np.quantile(ok,.1)) if ok else None,"median":float(np.median(ok)) if ok else None,"q90":float(np.quantile(ok,.9)) if ok else None,
        "applicability_rate":sum(v["status"]!="inapplicable" for v in vals)/len(vals),"failure_rate":sum(v["status"] in ("invalid","nonconvergence") for v in vals)/len(vals)})
    return out


def maxdiff(a,b):
    if isinstance(a,dict): return max([maxdiff(a[k],b[k]) for k in a if k in b] or [0.0])
    if isinstance(a,np.ndarray): return float(np.max(np.abs(a-b))) if a.size else 0.0
    return 0.0


def demix_state_snapshot(states):
    return {"means":states["means"],"global_projector":states["global_basis"]@states["global_basis"].T,
            "marginal_projectors":{k:v@v.T for k,v in states["marginal_basis"].items()},
            "dpca_maps":states["dpca_maps"],"decoders":states["decoders"]}


def demix_recovery_snapshot(recovery):
    return {"reconstruction":recovery["reconstruction"],"decoding":recovery["decoding"]}


def demix_poison_diagnostics():
    s=DEMIX_SCENARIOS[0]; r=0; d=generate_demix(s,r)
    base=fit_demix_states(s,r,d["train"]["observed"],d["train"]["labels"],d["validation"]["observed"],d["validation"]["labels"])
    original=recover_demix(base,d["test"]["observed"],d["test"]["labels"])
    poisoned_test=d["test"]["observed"]+777.0
    # Refit on the same lawful payload while excluded test values are materially changed outside the interface.
    fit_after_test_poison=fit_demix_states(s,r,d["train"]["observed"],d["train"]["labels"],d["validation"]["observed"],d["validation"]["labels"])
    fixed_after_test_poison=recover_demix(fit_after_test_poison,d["test"]["observed"],d["test"]["labels"])
    # Clean signal and generator truth are scoring-only hidden targets and are excluded from both fit and recovery.
    poisoned_hidden_clean=d["test"]["clean"]+999.0
    fit_after_truth_poison=fit_demix_states(s,r,d["train"]["observed"],d["train"]["labels"],d["validation"]["observed"],d["validation"]["labels"])
    fixed_after_truth_poison=recover_demix(fit_after_truth_poison,d["test"]["observed"],d["test"]["labels"])
    perm=deterministic_permutation(len(d["train"]["labels"]),s["id"],r,99)
    changed=fit_demix_states(s,r,d["train"]["observed"],d["train"]["labels"][perm],d["validation"]["observed"],d["validation"]["labels"])
    replay=fit_demix_states(s,r,d["train"]["observed"],d["train"]["labels"],d["validation"]["observed"],d["validation"]["labels"])
    return {"lawful_fit_signature_excludes_test_and_truth":True,"frozen_recovery_signature_excludes_train_validation_and_truth":True,
      "test_observation_poison_complete_fitted_state_max_abs":maxdiff(demix_state_snapshot(base),demix_state_snapshot(fit_after_test_poison)),
      "test_observation_poison_fixed_recovery_max_abs":maxdiff(demix_recovery_snapshot(original),demix_recovery_snapshot(fixed_after_test_poison)),
      "hidden_truth_poison_complete_fitted_state_max_abs":maxdiff(demix_state_snapshot(base),demix_state_snapshot(fit_after_truth_poison)),
      "hidden_truth_poison_fixed_recovery_max_abs":maxdiff(demix_recovery_snapshot(original),demix_recovery_snapshot(fixed_after_truth_poison)),
      "test_observations_materially_poisoned":float(np.max(np.abs(poisoned_test-d["test"]["observed"])))>100,
      "hidden_truth_materially_poisoned":float(np.max(np.abs(poisoned_hidden_clean-d["test"]["clean"])))>100,
      "label_permutation_dpca_change":maxdiff(base["dpca_maps"],changed["dpca_maps"]),
      "label_permutation_decoder_change":maxdiff(base["decoders"],changed["decoders"]),
      "deterministic_label_control_replay_max_abs":maxdiff(base["shuffled_maps"],replay["shuffled_maps"])}


def rotation_state_snapshot(states):
    return {"basis_projector":states["prep"]["basis"]@states["prep"]["basis"].T,"unrestricted":states["unrestricted"],
            "skew":states["skew"],"symmetric":states["symmetric"],"input_state":states["input_state"],"input_coef":states["input_coef"],
            "planes":{k:v@v.T for k,v in states["planes"].items()}}


def rotation_recovery_snapshot(recovery):
    return {"scores":recovery["scores"],"predictions":recovery["predictions"]}


def rotation_poison_diagnostics():
    s=ROT_SCENARIOS[0]; r=0; d=generate_rotation(s,r)
    states=fit_rotation_states(s,r,d["train"]["observed"],d["validation"]["observed"],d["train"]["inputs"],d["validation"]["inputs"])
    original=recover_rotation(states,d["test"]["observed"],d["test"]["inputs"])
    poisoned_test=d["test"]["observed"]+777.0
    fit_after_test_poison=fit_rotation_states(s,r,d["train"]["observed"],d["validation"]["observed"],d["train"]["inputs"],d["validation"]["inputs"])
    fixed_after_test_poison=recover_rotation(fit_after_test_poison,d["test"]["observed"],d["test"]["inputs"])
    poisoned_hidden_latent=d["test"]["latent"]+999.0
    fit_after_truth_poison=fit_rotation_states(s,r,d["train"]["observed"],d["validation"]["observed"],d["train"]["inputs"],d["validation"]["inputs"])
    fixed_after_truth_poison=recover_rotation(fit_after_truth_poison,d["test"]["observed"],d["test"]["inputs"])
    base_obs=d["train"]["observed"]; perm=np.concatenate([base_obs[1:],base_obs[:1]],axis=0)
    changed=fit_rotation_states(s,r,perm,d["validation"]["observed"],d["train"]["inputs"],d["validation"]["inputs"])
    replay=fit_rotation_states(s,r,d["train"]["observed"],d["validation"]["observed"],d["train"]["inputs"],d["validation"]["inputs"])
    return {"lawful_fit_signature_excludes_test_and_truth":True,"frozen_recovery_signature_excludes_train_validation_and_truth":True,
      "test_observation_poison_complete_fitted_state_max_abs":maxdiff(rotation_state_snapshot(states),rotation_state_snapshot(fit_after_test_poison)),
      "test_observation_poison_fixed_recovery_max_abs":maxdiff(rotation_recovery_snapshot(original),rotation_recovery_snapshot(fixed_after_test_poison)),
      "hidden_truth_poison_complete_fitted_state_max_abs":maxdiff(rotation_state_snapshot(states),rotation_state_snapshot(fit_after_truth_poison)),
      "hidden_truth_poison_fixed_recovery_max_abs":maxdiff(rotation_recovery_snapshot(original),rotation_recovery_snapshot(fixed_after_truth_poison)),
      "test_observations_materially_poisoned":float(np.max(np.abs(poisoned_test-d["test"]["observed"])))>100,
      "hidden_truth_materially_poisoned":float(np.max(np.abs(poisoned_hidden_latent-d["test"]["latent"])))>100,
      "condition_permutation_skew_change":maxdiff(states["skew"],changed["skew"]),
      "deterministic_condition_control_replay_max_abs":maxdiff(states["planes"]["covariance_shuffle_control"],replay["planes"]["covariance_shuffle_control"])}

def pseudo_population_diagnostics():
    scenario=next(x for x in DEMIX_SCENARIOS if x["id"]=="pseudo_population")
    original=dict(scenario); original.pop("pseudo",None)
    records={}
    for replicate in range(REPLICATES):
      for split in ("train","validation","test"):
        base=generate_demix_split(original,replicate,split); pseudo=generate_demix_split(scenario,replicate,split)
        base_means,_=condition_means(base["observed"],base["labels"]); pseudo_means,_=condition_means(pseudo["observed"],pseudo["labels"])
        def residual_cov(record):
          blocks=[]
          for a in range(2):
            for b in range(2):
              idx=np.flatnonzero(np.all(record["labels"]==[a,b],axis=1)); values=record["observed"][idx]
              blocks.append((values-values.mean(axis=0,keepdims=True)).reshape(-1,DEMIX_P))
          return np.cov(np.concatenate(blocks,axis=0),rowvar=False)
        ca=residual_cov(base); cb=residual_cov(pseudo); off=~np.eye(DEMIX_P,dtype=bool)
        records[f"{replicate}:{split}"]={"labels_preserved":bool(np.array_equal(base["labels"],pseudo["labels"])),
                        "cell_mean_max_abs":float(np.max(np.abs(base_means-pseudo_means))),
                        "off_diagonal_trial_noise_covariance_change":float(np.linalg.norm((ca-cb)[off]))}
    return {"within_split_cell_feature_complete_trajectory_permutation":True,"by_replicate_split":records}

def shift_diagnostics():
    marker=np.arange(6.0)[:,None]
    plus=shift_nonwrap(marker,1)[:,0]; minus=shift_nonwrap(marker,-1)[:,0]
    primitive={"positive_offset":plus.tolist(),"negative_offset":minus.tolist(),
               "positive_no_wrap":bool(plus[0]==marker[0,0] and plus[-1]==marker[-2,0]),
               "negative_no_wrap":bool(minus[-1]==marker[-1,0] and minus[0]==marker[1,0])}
    ds=next(x for x in DEMIX_SCENARIOS if x["id"]=="averaging_mismatch"); split=generate_demix_split(ds,0,"train")
    expected=[]
    for label,offset in zip(split["labels"],split["time_offsets"]):
      truth=split["truth"]; ss=2*label[0]-1; dd=2*label[1]-1
      signal=truth["time"]+.9*ss*truth["stim_profile"][:,None]*truth["stim_vec"]+.75*dd*truth["decision_profile"][:,None]*truth["decision_vec"]
      expected.append(shift_nonwrap(signal,int(offset)))
    demix_error=float(np.max(np.abs(np.asarray(expected)-split["clean"])))
    rs=next(x for x in ROT_SCENARIOS if x["id"]=="time_jitter"); base=dict(rs); base.pop("jitter",None)
    rot_errors=[]; offsets=[]
    for condition in range(ROT_COND):
      shifted,shifted_u,offset=trajectory(rs,condition); raw,raw_u,_=trajectory(base,condition)
      rot_errors.append(max(float(np.max(np.abs(shifted-shift_nonwrap(raw,offset)))),float(np.max(np.abs(shifted_u-shift_nonwrap(raw_u,offset)))))); offsets.append(offset)
    return {"primitive":primitive,"demixing_offsets":split["time_offsets"].tolist(),"demixing_replay_max_abs":demix_error,
            "rotation_offsets":offsets,"rotation_replay_max_abs":max(rot_errors),"no_np_roll_in_source":bool(("np"+".roll") not in Path(__file__).read_text())}


def design_and_selection_diagnostics():
    missing=next(x for x in DEMIX_SCENARIOS if x["id"]=="missing_cell"); data=generate_demix(missing,0)
    states=fit_demix_states(missing,0,data["train"]["observed"],data["train"]["labels"],data["validation"]["observed"],data["validation"]["labels"])
    rows,_=evaluate_demix(missing,0)
    complete=[r for r in rows if requires_complete_factorial(r["method"],r["metric"])]
    decoder=[r for r in rows if r["metric"].endswith("decoding_accuracy") and demix_applicable(r["method"],r["metric"])]
    def argmin(records,key): return min(records,key=lambda q:(q[key],q["lambda"]))["lambda"]
    records={}
    for scenario in DEMIX_SCENARIOS:
      for replicate in range(REPLICATES):
        d=generate_demix(scenario,replicate)
        selected=fit_demix_states(scenario,replicate,d["train"]["observed"],d["train"]["labels"],d["validation"]["observed"],d["validation"]["labels"])
        records[f"{scenario['id']}:{replicate}"]={"design_rank":selected["design_rank"],
          "selected_parameters":{"dpca":selected["selected_lambda"],"shuffled_dpca":selected["shuffled_selected_lambda"],
            "decoder":selected["decoder_lambdas"],"shuffled_decoder":selected["shuffled_decoder_lambdas"]},
          "selection_oracles":{"dpca":selected["selected_lambda"]==argmin(selected["dpca_candidates"],"validation_loss"),
            "shuffled_dpca":selected["shuffled_selected_lambda"]==argmin(selected["shuffled_dpca_candidates"],"validation_loss"),
            "decoder":all(selected["decoder_lambdas"][k]==argmin(selected["decoder_candidates"][k],"validation_error") for k in ("stimulus","decision")),
            "shuffled_decoder":all(selected["shuffled_decoder_lambdas"][k]==argmin(selected["shuffled_decoder_candidates"][k],"validation_error") for k in ("stimulus","decision"))}}
    return {"missing_cell_design_rank":states["design_rank"],"complete_marginal_rows_all_invalid":bool(complete) and all(r["status"]=="invalid" and r["reason"]=="incomplete_factorial_design_rank_3_of_4" for r in complete),
      "decoder_rows_retained":bool(decoder) and all(r["status"]=="ok" for r in decoder),"by_scenario_replicate":records,
      "all_selection_oracles_pass":all(all(record["selection_oracles"].values()) for record in records.values())}

def covariance_control_diagnostics():
    scenario=next(x for x in ROT_SCENARIOS if x["id"]=="covariance_control"); records={}
    for replicate in range(REPLICATES):
      data=generate_rotation(scenario,replicate); clone=dict(scenario); clone.pop("covariance_shuffle",None); clean=generate_rotation(clone,replicate)
      states=fit_rotation_states(scenario,replicate,data["train"]["observed"],data["validation"]["observed"],data["train"]["inputs"],data["validation"]["inputs"])
      scores=states["covariance_proposals"]; best=max(scores,key=lambda q:(q["covariance_similarity"],-q["index"]))
      pca=states["planes"]["pca_plane"]; fixed=states["planes"]["fixed_plane"]
      selected_record=next(q for q in scores if q["index"]==states["covariance_selected_proposal"])
      records[str(replicate)]={"selected_proposal":states["covariance_selected_proposal"],"selected_similarity":states["covariance_similarity"],
        "match_status":states["covariance_match_status"],"selection_is_argmax":states["covariance_selected_proposal"]==best["index"],
        "status_matches_threshold":states["covariance_match_status"]==("matched" if states["covariance_similarity"]>=COVARIANCE_MATCH_THRESHOLD else "incompletely_matched"),
        "attempted_proposal_count":COVARIANCE_PROPOSALS,"admissible_proposal_count":len(scores),"excluded_uniform_shift_count":COVARIANCE_PROPOSALS-len(scores),
        "all_admissible_have_multiple_distinct_shifts":all(q["distinct_shift_count"]>=2 for q in scores),
        "selected_has_multiple_distinct_shifts":selected_record["distinct_shift_count"]>=2,
        "selected_cross_coordinate_correspondence_change_fraction":selected_record["cross_coordinate_correspondence_change_fraction"],
        "selected_scores_max_abs_change":selected_record["selected_scores_max_abs_change"],
        "selected_materially_changes_cross_coordinate_correspondence":selected_record["cross_coordinate_correspondence_change_fraction"]>0 and selected_record["selected_scores_max_abs_change"]>1e-8,
        "generation_has_no_shuffle_max_abs":max(float(np.max(np.abs(data[s]["observed"]-clean[s]["observed"]))) for s in ("train","validation","test")),
        "fixed_plane_distinct_from_pca_projector_max_abs":float(np.max(np.abs(fixed@fixed.T-pca@pca.T))),"proposal_scores":scores}
    return {"proposal_family":"24 seed-derived feature-wise cyclic condition shifts; one proposal selected by training covariance cosine similarity",
      "threshold":COVARIANCE_MATCH_THRESHOLD,"by_replicate":records,
      "all_selection_oracles_pass":all(r["selection_is_argmax"] and r["status_matches_threshold"] for r in records.values()),
      "all_admissible_proposals_nonuniform":all(r["all_admissible_have_multiple_distinct_shifts"] for r in records.values()),
      "all_selected_proposals_nonuniform":all(r["selected_has_multiple_distinct_shifts"] for r in records.values()),
      "all_selected_materially_change_cross_coordinate_correspondence":all(r["selected_materially_changes_cross_coordinate_correspondence"] for r in records.values()),
      "all_generation_unshuffled":all(r["generation_has_no_shuffle_max_abs"]==0 for r in records.values()),
      "all_fixed_planes_distinct":all(r["fixed_plane_distinct_from_pca_projector_max_abs"]>0 for r in records.values())}

def oracle_diagnostics():
    ds=DEMIX_SCENARIOS[0]; dd=generate_demix(ds,0); truth=dd["train"]["truth"]
    marginal_orthogonality=float(abs(truth["stim_vec"]@truth["decision_vec"]))
    rs=ROT_SCENARIOS[0]; rd=generate_rotation(rs,0); z,u,_=trajectory(rs,0); analytic=z@rot_matrix("rotation").T
    finite=(z[2:]-z[:-2])/(2*ROT_DT)
    derivative_error=float(np.max(np.abs(finite-analytic[1:-1])))
    direct=np.linalg.lstsq(z,analytic,rcond=None)[0].T
    direct_generator_error=float(np.max(np.abs(direct-rot_matrix("rotation"))))
    true_plane_error=angle_degrees(rd["test"]["embedding"][:,:2],rot_embedding(False)[:,:2])
    return {"demixing_true_marginal_vector_inner_product":marginal_orthogonality,
      "rotation_analytic_finite_difference_max_abs":derivative_error,"rotation_direct_generator_recovery_max_abs":direct_generator_error,
      "rotation_actual_embedding_plane_error_degrees":true_plane_error}


def fault_diagnostics():
    dr,_=evaluate_demix(DEMIX_SCENARIOS[0],0,"nonfinite"); dl,_=evaluate_demix(DEMIX_SCENARIOS[0],0,"linalg")
    rr,_=evaluate_rotation(ROT_SCENARIOS[0],0,"nonfinite"); rl,_=evaluate_rotation(ROT_SCENARIOS[0],0,"linalg")
    def counts(rows):
      return {s:sum(r["status"]==s for r in rows) for s in STATUS_VALUES}
    return {"demix_nonfinite":counts(dr),"demix_linalg":counts(dl),"rotation_nonfinite":counts(rr),"rotation_linalg":counts(rl),
      "applicability_precedes_failure":all(r["status"]=="inapplicable" for rows in (dr,dl,rr,rl) for r in rows if r["reason"]=="method_metric_inapplicable")}


def verification_diagnostics():
    return {"demixing_poison":demix_poison_diagnostics(),"rotation_poison":rotation_poison_diagnostics(),
            "pseudo_population":pseudo_population_diagnostics(),"nonwrap_shifts":shift_diagnostics(),
            "design_and_selection":design_and_selection_diagnostics(),"covariance_control":covariance_control_diagnostics(),
            "actual_oracles":oracle_diagnostics(),"real_path_faults":fault_diagnostics()}


def make_summary(path,diag):
    img=Image.new("RGB",(1500,900),"white"); d=ImageDraw.Draw(img); font=ImageFont.load_default()
    d.text((30,20),"P4-U04 Demixing and Rotational Projection — separate task-specific panels",fill="black",font=font)
    groups={(s["task"],s["scenario"],s["method"],s["metric"]):s for s in diag["summaries"]}
    panels=[("Demixing: stimulus marginal NMSE","demixing","balanced_reference","stimulus_marginal_nmse",["condition_mean","pca","marginal_pca","selected_dpca","shuffled_dpca_control"]),
      ("Demixing: stimulus decoding accuracy","demixing","balanced_reference","stimulus_decoding_accuracy",["ridge_decoder","shuffled_decoder_control"]),
      ("Rotation: clean held-out derivative RMSE","rotation","clean_rotation","derivative_rmse",["unrestricted_fit","skew_fit","symmetric_fit","input_aware_fit"]),
      ("Rotation: clean plane angle (degrees)","rotation","clean_rotation","plane_angle_degrees",["pca_plane","unrestricted_fit","skew_fit","fixed_plane","random_plane"])]
    for pi,(title,task,sc,metric,methods) in enumerate(panels):
      x0=30+(pi%2)*735; y0=70+(pi//2)*390; d.rectangle((x0,y0,x0+700,y0+340),outline="gray"); d.text((x0+12,y0+10),title,fill="black",font=font)
      vals=[groups[(task,sc,m,metric)]["median"] for m in methods]; finite=[v for v in vals if v is not None]; vmax=max(finite+[1e-9])
      for i,(m,v) in enumerate(zip(methods,vals)):
        y=y0+55+i*48; d.text((x0+12,y),m,fill="black",font=font)
        if v is not None:
          w=int(430*v/vmax); d.rectangle((x0+220,y,x0+220+w,y+20),fill=(70,120,180)); d.text((x0+660,y),f"{v:.4g}",fill="black",font=font,anchor="ra")
    d.text((30,860),"No cross-task score. Reconstruction, decoding, plane, rate, derivative and tangential targets remain distinct.",fill="black",font=font)
    img.save(path,optimize=False)


def source_records():
    return [{"path":str(p.relative_to(REPO_ROOT)),"sha256":sha256(p),"role":"scientific_design_provenance_only; benchmark-local implementation"} for p in PHASE3_SOURCES]


def build(root,reverse=False,resume=False,quiet=False):
    root.mkdir(parents=True,exist_ok=True); fresh=all_fresh_rows(reverse)
    if resume and (root/"results.csv").exists():
      prior=read_results(root/"results.csv"); kept=validate_resume(prior,fresh); rows=[kept.get(r["row_key"],r) for r in fresh]
    else: rows=fresh
    write_results(root/"results.csv",rows)
    status={s:sum(r["status"]==s for r in rows) for s in STATUS_VALUES}
    diag={"schema_version":1,"benchmark":"P4-U04","row_count":len(rows),"status_counts":status,"task_row_counts":{t:sum(r["task"]==t for r in rows) for t in ("demixing","rotation")},
      "summaries":summaries(rows),"verification_checks":verification_diagnostics(),
      "scientific_limits":["The two tasks are never pooled or ranked together.","Demixing and decoding targets are not equivalent.","Rotational projections do not prove autonomous dynamics or mechanism.","Three replicates yield descriptive quantiles, not confidence intervals."]}
    (root/"diagnostics.json").write_text(json.dumps(diag,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    make_summary(root/"summary.png",diag)
    manifest={"schema_version":1,"benchmark":"demixing-rotational-robustness","date":"2026-08-20","maturity":"L1","status":"stable","master_seed":MASTER_SEED,"replicates":REPLICATES,
      "tasks":{"demixing":{"scenarios":[s["id"] for s in DEMIX_SCENARIOS],"methods":[m[0] for m in DEMIX_METHODS],"metrics":list(DEMIX_METRICS)},
               "rotation":{"scenarios":[s["id"] for s in ROT_SCENARIOS],"methods":[m[0] for m in ROT_METHODS],"metrics":list(ROT_METRICS)}},
      "contract":{"no_cross_task_score":True,"stage_separation":["generation","fitting_selection","frozen_recovery","scoring"],"splits":["train","validation","test"],"status_values":list(STATUS_VALUES),"canonical_order":"row_key lexical",
        "split_size_metadata":"demixing records 2x2 per-cell counts plus actual totals; rotation records six per-condition counts plus actual totals",
        "missing_factorial_rule":"rank-deficient training factorial designs invalidate balanced-marginalization targets before recovery; label decoding remains separately assessed",
        "pseudo_population_rule":"within each split, cell and feature, complete trial trajectories are permuted",
        "shuffled_control_rule":"independent train/validation label permutations repeat full hyperparameter selection and refit",
        "covariance_control":{"proposal_family":"24 predeclared seed-derived feature-wise cyclic condition-shift attempts; uniform all-coordinate shifts are excluded as global relabelings/no-ops","admissibility":"at least two distinct coordinate shifts and material cross-coordinate correspondence change","selection":"maximum training covariance cosine similarity among admissible proposals only","threshold":COVARIANCE_MATCH_THRESHOLD,"unmatched_label":"incompletely_matched","exact_preservation_not_assumed":True},
        "time_shift_rule":"explicit nearest-edge fill; no circular wrap"},
      "phase3_sources":source_records(),"runtime":{"python":platform.python_version(),"numpy":np.__version__,"pillow":Image.__version__,"platform":platform.platform(),"executable":sys.executable},
      "commands":{"generate":f"{sys.executable} benchmarks/demixing-rotational-robustness.py --generate","verify":f"{sys.executable} benchmarks/demixing-rotational-robustness.py --verify","resume":f"{sys.executable} benchmarks/demixing-rotational-robustness.py --generate --resume"},
      "artifacts":{name:{"sha256":sha256(root/name),"bytes":(root/name).stat().st_size} for name in ARTIFACT_NAMES}}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not quiet: print(json.dumps({"root":str(root),"rows":len(rows),"status":status},sort_keys=True))
    return manifest


def expected_keys():
    keys=[]
    for s in DEMIX_SCENARIOS:
      for r in range(REPLICATES):
       for m,_,_ in DEMIX_METHODS:
        for metric in DEMIX_METRICS: keys.append(f"demixing|{s['id']}|{r}|{m}|{metric}")
    for s in ROT_SCENARIOS:
      for r in range(REPLICATES):
       for m,_,_ in ROT_METHODS:
        for metric in ROT_METRICS: keys.append(f"rotation|{s['id']}|{r}|{m}|{metric}")
    return sorted(keys)


def validate_artifacts(root):
    errors=[]; names=sorted(p.name for p in root.iterdir() if p.is_file())
    if names!=sorted(("manifest.json",*ARTIFACT_NAMES)): errors.append(f"artifact_inventory:{names}")
    try: manifest=json.loads((root/"manifest.json").read_text()); diag=json.loads((root/"diagnostics.json").read_text()); rows=read_results(root/"results.csv")
    except Exception as e: return [f"read_error:{e}"]
    if [r["row_key"] for r in rows]!=expected_keys(): errors.append("keys_or_order")
    if len(set(r["row_key"] for r in rows))!=len(rows): errors.append("duplicate_keys")
    for r in rows:
      try: validate_row(r)
      except Exception as e: errors.append(f"row:{r.get('row_key')}:{e}")
    if diag["row_count"]!=len(rows): errors.append("row_count")
    if diag["summaries"]!=summaries(rows): errors.append("summary_recompute")
    for name in ARTIFACT_NAMES:
      if manifest["artifacts"][name]["sha256"]!=sha256(root/name): errors.append(f"hash:{name}")
    vc=diag["verification_checks"]
    poison_zero_keys=("test_observation_poison_complete_fitted_state_max_abs","test_observation_poison_fixed_recovery_max_abs",
                      "hidden_truth_poison_complete_fitted_state_max_abs","hidden_truth_poison_fixed_recovery_max_abs")
    if any(vc[task][key]!=0 for task in ("demixing_poison","rotation_poison") for key in poison_zero_keys): errors.append("capability_or_hidden_poison")
    if not all(vc[task][key] for task in ("demixing_poison","rotation_poison") for key in ("test_observations_materially_poisoned","hidden_truth_materially_poisoned")): errors.append("poison_not_material")
    if vc["demixing_poison"]["label_permutation_dpca_change"]<=0 or vc["rotation_poison"]["condition_permutation_skew_change"]<=0: errors.append("permutation_sensitivity")
    if not vc["real_path_faults"]["applicability_precedes_failure"]: errors.append("applicability_precedence")
    pseudo=vc["pseudo_population"]
    if not all(x["labels_preserved"] and x["cell_mean_max_abs"]<1e-12 and x["off_diagonal_trial_noise_covariance_change"]>1e-6 for x in pseudo["by_replicate_split"].values()): errors.append("pseudo_population_contract")
    shifts=vc["nonwrap_shifts"]
    if not (shifts["primitive"]["positive_no_wrap"] and shifts["primitive"]["negative_no_wrap"] and shifts["demixing_replay_max_abs"]==0 and shifts["rotation_replay_max_abs"]==0 and shifts["no_np_roll_in_source"]): errors.append("nonwrap_shift_contract")
    design=vc["design_and_selection"]
    if design["missing_cell_design_rank"]!=3 or not design["complete_marginal_rows_all_invalid"] or not design["decoder_rows_retained"] or not design["all_selection_oracles_pass"]: errors.append("design_or_selection_contract")
    control=vc["covariance_control"]
    if (not control["all_selection_oracles_pass"] or not control["all_admissible_proposals_nonuniform"] or
        not control["all_selected_proposals_nonuniform"] or not control["all_selected_materially_change_cross_coordinate_correspondence"] or
        not control["all_generation_unshuffled"] or not control["all_fixed_planes_distinct"]): errors.append("covariance_control_contract")
    for replicate,record in control["by_replicate"].items():
      if any(len(set(q["feature_condition_shifts"]))<2 or q["distinct_shift_count"]<2 or q["cross_coordinate_correspondence_change_fraction"]<=0 for q in record["proposal_scores"]): errors.append(f"uniform_covariance_proposal:{replicate}")
      selected=next((q for q in record["proposal_scores"] if q["index"]==record["selected_proposal"]),None)
      if selected is None or selected["selected_scores_max_abs_change"]<=1e-8 or selected["cross_coordinate_correspondence_change_fraction"]<=0: errors.append(f"selected_covariance_correspondence_no_change:{replicate}")
    missing_rows=[r for r in rows if r["task"]=="demixing" and r["scenario"]=="missing_cell" and requires_complete_factorial(r["method"],r["metric"])]
    if not missing_rows or not all(r["status"]=="invalid" and r["reason"]=="incomplete_factorial_design_rank_3_of_4" for r in missing_rows): errors.append("missing_cell_rows")
    design_rows=[r for r in rows if r["task"]=="demixing" and r["scenario"]=="missing_cell" and r["metric"]=="factorial_design_rank" and r["method"]=="factorial_design_diagnostic"]
    if len(design_rows)!=REPLICATES or not all(r["status"]=="ok" and float(r["value"])==3 for r in design_rows): errors.append("design_rank_rows")
    for r in rows:
      try:
        train=np.asarray(json.loads(r["train_cell_counts"])); validation=np.asarray(json.loads(r["validation_cell_counts"])); test=np.asarray(json.loads(r["test_cell_counts"]))
        if int(r["n_train_total"])!=int(train.sum()) or int(r["n_validation_total"])!=int(validation.sum()) or int(r["n_test_total"])!=int(test.sum()): errors.append(f"split_totals:{r['row_key']}")
      except Exception: errors.append(f"split_metadata:{r['row_key']}")
      if r["method"]=="covariance_shuffle_control":
        if r["control_match_status"] not in ("matched","incompletely_matched") or r["control_proposal"]=="": errors.append(f"control_metadata:{r['row_key']}")
    control_groups={}
    for r in rows:
      if r["task"]=="rotation" and r["method"]=="covariance_shuffle_control": control_groups.setdefault((r["scenario"],r["replicate"]),[]).append(r)
    for key,group in control_groups.items():
      similarity=next((float(r["value"]) for r in group if r["metric"]=="covariance_similarity" and r["status"]=="ok"),None)
      if similarity is None: errors.append(f"control_similarity_missing:{key}"); continue
      expected="matched" if similarity>=COVARIANCE_MATCH_THRESHOLD else "incompletely_matched"
      if any(r["control_match_status"]!=expected for r in group): errors.append(f"control_threshold_label:{key}")
    selected_methods=("selected_dpca","ridge_decoder","shuffled_dpca_control","shuffled_decoder_control")
    if any(r["selected_hyperparameter"]=="" for r in rows if r["task"]=="demixing" and r["method"] in selected_methods): errors.append("selected_parameter_not_retained")
    oracle=vc["actual_oracles"]
    if oracle["demixing_true_marginal_vector_inner_product"]>1e-10 or oracle["rotation_direct_generator_recovery_max_abs"]>1e-10 or oracle["rotation_actual_embedding_plane_error_degrees"]>1e-4 or oracle["rotation_analytic_finite_difference_max_abs"]>.03: errors.append("oracle_tolerance")
    return errors


def compare_roots(a,b): return [n for n in ("manifest.json",*ARTIFACT_NAMES) if (a/n).read_bytes()!=(b/n).read_bytes()]


def adversarial_resume(rows,root):
    checks=[]
    variants={"corrupt":[dict(rows[0],value="999",status="ok",reason="")],"duplicate":[rows[0],rows[0]],"unknown":[dict(rows[0],row_key="unknown")],"incomplete":[{k:v for k,v in rows[0].items() if k!="target"}]}
    for name,prior in variants.items():
      try: validate_resume(prior,rows); checks.append(name)
      except ValueError: pass
    return checks


def verify(root):
    errors=validate_artifacts(root)
    with tempfile.TemporaryDirectory() as td:
      td=Path(td); a=td/"clean"; b=td/"reverse"; c=td/"resume"
      build(a,quiet=True); build(b,reverse=True,quiet=True)
      if compare_roots(a,b): errors.append("reverse_bytes")
      c.mkdir(); fresh=read_results(a/"results.csv"); write_results(c/"results.csv",fresh[::5]); build(c,resume=True,quiet=True)
      if compare_roots(a,c): errors.append("resume_bytes")
      errors.extend(f"adversarial_resume:{x}" for x in adversarial_resume(fresh,c))
      if compare_roots(root,a): errors.append("clean_regeneration_bytes")
    if errors: raise SystemExit("verification failed:\n"+"\n".join(errors))
    print(json.dumps({"verified":str(root),"rows":len(read_results(root/"results.csv")),"checks":"PASS"},sort_keys=True))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=DEFAULT_ROOT); p.add_argument("--generate",action="store_true"); p.add_argument("--verify",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--reverse",action="store_true")
    a=p.parse_args()
    if a.generate or not a.verify: build(a.root,a.reverse,a.resume)
    if a.verify: verify(a.root)

if __name__=="__main__": main()
