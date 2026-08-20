#!/usr/bin/env python3
"""Deterministic finite graph/operator recovery demonstration (P3-U07).

NumPy and Pillow only. Generation, graph/operator fitting, extension, scoring,
and verification are separated. With no flags, generate and verify artifacts.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASTER_SEED=20260820; DATE="2026-08-20"; P=3
BRANCH=3.0; RADIUS=0.175; GAP=0.35; TOTAL=6.0+math.pi*RADIUS
NOISE_SD=0.01; NOMINAL_K=6; LONG_ARC_EDGE_THRESHOLD=0.70; PATH_SHORTCUT_UNDERESTIMATION_THRESHOLD=0.50
HAIRPIN_SIZES={"train":180,"validation":80,"test":80}; CIRCLE_N=96
EPSILON=0.18; CIRCLE_TIME=1
SCRIPT_DIR=Path(__file__).resolve().parent
DEFAULT_ROOT=SCRIPT_DIR/"artifacts"/"graph-geometry-recovery"
ARTIFACT_NAMES=("metrics.csv","diagnostics.json","summary.png")
FIELDS=("domain","condition","split","method","target","metric","value")
TOL={"symmetry":1e-12,"diagonal":1e-12,"triangle":1e-10,"oracle":2e-10,"fourier":2e-10,"projector":2e-10,"poison":0.0}


def rng_for(*codes): return np.random.Generator(np.random.PCG64(np.random.SeedSequence((MASTER_SEED,)+tuple(codes))))

def embed_isometry():
    raw=np.array([[1.0,.3],[-.2,.9],[.45,-.35]]); q,_=np.linalg.qr(raw)
    return q[:,:2],np.array([.4,-.7,.2])

def hairpin_planar(ell,r=RADIUS):
    ell=np.asarray(ell,float); out=np.empty((ell.size,2)); c=math.pi*r
    a=ell<=BRANCH; b=(ell>BRANCH)&(ell<=BRANCH+c); d=ell>BRANCH+c
    out[a]=np.column_stack((np.zeros(a.sum()),ell[a]))
    u=ell[b]-BRANCH; theta=math.pi-u/r
    out[b]=np.column_stack((r+r*np.cos(theta),BRANCH+r*np.sin(theta)))
    v=ell[d]-(BRANCH+c); out[d]=np.column_stack((np.full(d.sum(),2*r),BRANCH-v))
    return out

def hairpin_observed(ell,r=RADIUS,noise=True,seed_codes=(0,)):
    q,off=embed_isometry(); x=hairpin_planar(ell,r)@q.T+off
    if noise: x=x+rng_for(*seed_codes).normal(scale=NOISE_SD,size=x.shape)
    return x

def sample_ell(split,n,condition="nominal"):
    code={"train":1,"validation":2,"test":3}[split]; rng=rng_for(10,code,{"nominal":1,"nonuniform":2,"sparse_gap":3}.get(condition,1))
    if condition=="nonuniform":
        u=rng.uniform(size=n); ell=np.empty(n); left=u<.65
        ell[left]=TOTAL*(u[left]/.65)**2*.55
        ell[~left]=TOTAL*(.55+((u[~left]-.65)/.35)**.55*.45)
    elif condition=="sparse_gap":
        lo=BRANCH-.18; hi=BRANCH+math.pi*RADIUS+.18; pieces=[]
        while sum(len(z) for z in pieces)<n:
            z=rng.uniform(0,TOTAL,size=n); pieces.append(z[(z<lo)|(z>hi)])
        ell=np.concatenate(pieces)[:n]
    else: ell=rng.uniform(0,TOTAL,size=n)
    return np.sort(ell)

def generate_hairpin():
    nominal={}
    for split,n in HAIRPIN_SIZES.items():
        e=sample_ell(split,n); nominal[split]={"ell":e,"x":hairpin_observed(e,seed_codes=(11,{"train":1,"validation":2,"test":3}[split]))}
    controls={}
    for name in ("sparse_gap","nonuniform"):
        e=sample_ell("train",HAIRPIN_SIZES["train"],name); controls[name]={"ell":e,"x":hairpin_observed(e,seed_codes=(12,1 if name=="sparse_gap" else 2))}
    reduced_radius=0.06; reduced_total=6.0+math.pi*reduced_radius
    nominal_quantiles=nominal["train"]["ell"]/TOTAL; reduced_ell=nominal_quantiles*reduced_total
    controls["reduced_branch_separation"]={"ell":reduced_ell,"x":hairpin_observed(reduced_ell,r=reduced_radius,seed_codes=(12,3)),"radius":reduced_radius,"total_length":reduced_total,"endpoint":np.array([2.0*reduced_radius,0.0]),"nominal_quantiles":nominal_quantiles}
    return {"nominal":nominal,"controls":controls}

def pairwise_dist(x,y=None):
    y=x if y is None else y; d=x[:,None,:]-y[None,:,:]; return np.sqrt(np.maximum(np.sum(d*d,axis=2),0.0))

def knn_candidates(x,k,scale=1.0):
    dist=pairwise_dist(x)*scale; n=len(x); k=min(max(int(k),1),n-1)
    candidate=dist.copy(); np.fill_diagonal(candidate,np.inf)
    order=np.argsort(candidate,axis=1,kind="mergesort")[:,:k]
    return dist,order

def knn_graph(x,k,scale=1.0):
    dist,order=knn_candidates(x,k,scale); n=len(x)
    adj=np.full((n,n),np.inf); np.fill_diagonal(adj,0.0)
    for i in range(n):
        for j in order[i]:
            w=dist[i,j]
            if w<adj[i,j]: adj[i,j]=adj[j,i]=w
    return adj

def components(adj):
    n=len(adj); seen=np.zeros(n,bool); groups=[]
    for start in range(n):
        if seen[start]: continue
        stack=[start];seen[start]=True;group=[]
        while stack:
            i=stack.pop();group.append(i)
            for j in np.flatnonzero(np.isfinite(adj[i])&(np.arange(n)!=i)):
                if not seen[j]: seen[j]=True;stack.append(int(j))
        groups.append(group)
    return groups

def floyd(adj):
    d=adj.copy()
    for k in range(len(d)): d=np.minimum(d,d[:,k,None]+d[None,k,:])
    return d

def cmdscale(dist,dim=1):
    n=len(dist); d2=dist**2; row=d2.mean(1); grand=d2.mean(); b=-.5*(d2-row[:,None]-row[None,:]+grand)
    vals,vecs=np.linalg.eigh((b+b.T)/2); order=np.argsort(vals)[::-1]; vals=vals[order];vecs=vecs[:,order]
    pos=np.maximum(vals[:dim],0); coords=vecs[:,:dim]*np.sqrt(pos)
    neg=float(np.sum(np.abs(vals[vals<0]))); strain=float(np.sqrt(np.mean((b-coords@coords.T)**2)))
    return {"coords":coords,"values":vals,"vectors":vecs,"rowmean_d2":row,"grandmean_d2":grand,"negative_eigenmass":neg,"strain":strain}

def gower_extend(dnew,model,dim=1):
    d2=dnew**2; b=-.5*(d2-d2.mean(1)[:,None]-model["rowmean_d2"][None,:]+model["grandmean_d2"])
    vals=model["values"][:dim]; return b@model["vectors"][:,:dim]/np.sqrt(np.maximum(vals,1e-15))

def fit_affine(z,y):
    a=np.column_stack((z.ravel(),np.ones(len(z)))); coef=np.linalg.lstsq(a,y,rcond=None)[0]; return coef

def apply_affine(z,coef): return coef[0]*z.ravel()+coef[1]

def fit_hairpin(x,k=NOMINAL_K,scale=1.0):
    mean=x.mean(0); xc=x-mean; _,_,vt=np.linalg.svd(xc,full_matrices=False); basis=vt[:1].T; pca=xc@basis
    ambient=pairwise_dist(x)*scale; amb_mds=cmdscale(ambient)
    adj=knn_graph(x,k,scale); graph=floyd(adj); comps=components(adj)
    iso=cmdscale(graph) if len(comps)==1 else None
    return {"mean":mean,"basis":basis,"pca":pca,"ambient_dist":ambient,"ambient_mds":amb_mds,"adj":adj,"graph_dist":graph,"components":comps,"isomap":iso,"k":k,"scale":scale}

def extend_hairpin(model,xnew,xtrain):
    centered=xnew-model["mean"]; pca=centered@model["basis"]
    ambient=pairwise_dist(xnew,xtrain)*model["scale"]; amb=gower_extend(ambient,model["ambient_mds"])
    iso=None
    if model["isomap"] is not None:
        raw=pairwise_dist(xnew,xtrain)*model["scale"]; k=model["k"]; attach=np.argsort(raw,axis=1,kind="mergesort")[:,:k]
        gd=np.empty_like(raw)
        for i in range(len(xnew)): gd[i]=np.min(raw[i,attach[i]][:,None]+model["graph_dist"][attach[i]],axis=0)
        iso=gower_extend(gd,model["isomap"])
    return {"pca":pca,"ambient_cmds":amb,"isomap":iso}

def graph_metrics(adj,gd,ell):
    n=len(adj); edge=np.isfinite(adj)&(np.arange(n)[:,None]<np.arange(n)[None,:]); edge_count=int(edge.sum())
    delta=np.abs(ell[:,None]-ell[None,:]); edge_weight=np.where(edge,adj,0.0)
    long_edges=edge&(delta>LONG_ARC_EDGE_THRESHOLD); long_count=int(long_edges.sum())
    material=edge&((delta-edge_weight)>PATH_SHORTCUT_UNDERESTIMATION_THRESHOLD); shortcut_count=int(material.sum())
    pairs=np.triu(np.ones((n,n),bool),1); finite=pairs&np.isfinite(gd)
    rel=float(np.sqrt(np.mean((gd[finite]-delta[finite])**2))/max(np.sqrt(np.mean(delta[finite]**2)),1e-15)) if finite.any() else 0.0
    corr=float(np.corrcoef(gd[finite],delta[finite])[0,1]) if finite.sum()>2 else 0.0
    return {"edge_count":edge_count,"long_arc_separation_edge_count":long_count,"long_arc_separation_edge_rate":long_count/max(edge_count,1),"long_arc_separation_threshold":LONG_ARC_EDGE_THRESHOLD,"long_arc_separation_denominator_edges":edge_count,"material_path_shortcut_count":shortcut_count,"material_path_shortcut_rate":shortcut_count/max(edge_count,1),"material_path_shortcut_underestimation_threshold":PATH_SHORTCUT_UNDERESTIMATION_THRESHOLD,"material_path_shortcut_denominator_edges":edge_count,"graph_distance_relative_rmse":rel,"graph_distance_correlation":corr,"component_count":len(components(adj)),"finite_pair_count":int(finite.sum()),"infinite_pair_count":int((pairs&~np.isfinite(gd)).sum())}

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

def fit_score_hairpin(data):
    train=data["nominal"]["train"]; fit=fit_hairpin(train["x"]); exts={s:extend_hairpin(fit,data["nominal"][s]["x"],train["x"]) for s in ("train","validation","test")}
    exts["train"]={"pca":fit["pca"],"ambient_cmds":fit["ambient_mds"]["coords"],"isomap":fit["isomap"]["coords"]}
    aligned={};coefs={}
    for method in ("pca","ambient_cmds","isomap"):
        coefs[method]=fit_affine(exts["train"][method],train["ell"]);aligned[method]={s:apply_affine(exts[s][method],coefs[method]) for s in exts}
    return fit,exts,aligned,coefs

# Circle diffusion operator.
def circle_points(theta):
    q,_=embed_isometry(); return np.column_stack((np.cos(theta),np.sin(theta)))@q.T

def diffusion_fit(x,epsilon=EPSILON):
    d=pairwise_dist(x); w=np.exp(-(d**2)/epsilon); q=w.sum(1); k=w/(q[:,None]*q[None,:]); degree=k.sum(1)
    p=k/degree[:,None]; pi=degree/np.sum(degree); s=k/np.sqrt(degree[:,None]*degree[None,:])
    vals,phi=np.linalg.eigh((s+s.T)/2); order=np.argsort(vals)[::-1];vals=vals[order];phi=phi[:,order]
    psi=phi*np.sqrt(np.sum(degree))/np.sqrt(degree[:,None])
    return {"x":x,"epsilon":epsilon,"w":w,"q":q,"k":k,"degree":degree,"p":p,"pi":pi,"s":s,"values":vals,"phi":phi,"psi":psi}

def nystrom(model,xnew):
    dist=pairwise_dist(xnew,model["x"]); w=np.exp(-(dist**2)/model["epsilon"]); qx=w.sum(1); k=w/(qx[:,None]*model["q"][None,:]); dx=k.sum(1); p=k/dx[:,None]
    vals=model["values"]; out=(p@model["psi"])/np.where(np.abs(vals)>1e-14,vals,1.0)[None,:]; return out,p

def principal_angles(a,b):
    qa=np.linalg.qr(a)[0];qb=np.linalg.qr(b)[0]; sv=np.linalg.svd(qa.T@qb,compute_uv=False);return np.degrees(np.arccos(np.clip(sv,-1,1)))

def diffusion_distances(model,modes=None):
    vals=model["values"][1:];psi=model["psi"][:,1:]
    if modes is not None: vals=vals[:modes];psi=psi[:,:modes]
    z=psi*(vals**CIRCLE_TIME); return pairwise_dist(z)

def direct_diffusion_distances(model):
    transition=np.linalg.matrix_power(model["p"],CIRCLE_TIME)
    weighted=transition/np.sqrt(model["pi"])[None,:]
    return pairwise_dist(weighted)

def circle_analysis(theta,epsilon=EPSILON):
    x=circle_points(theta); model=diffusion_fit(x,epsilon); truth=np.column_stack((np.cos(theta),np.sin(theta)))
    sqrt_pi=np.sqrt(model["pi"])[:,None]
    angles=principal_angles(sqrt_pi*model["psi"][:,1:3],sqrt_pi*truth)
    pc=model["p"]@truth; gram=truth.T@(model["pi"][:,None]*truth); rhs=truth.T@(model["pi"][:,None]*pc); coefficient=np.linalg.solve(gram,rhs)
    residual_matrix=pc-truth@coefficient
    residual=math.sqrt(float(np.sum(model["pi"][:,None]*residual_matrix**2)))/max(math.sqrt(float(np.sum(model["pi"][:,None]*truth**2))),1e-15)
    full=diffusion_distances(model); direct=direct_diffusion_distances(model); trunc=diffusion_distances(model,2); pairs=np.triu(np.ones(full.shape,bool),1)
    return model,{"l2_pi_principal_angles_degrees":angles,"l2_pi_p_invariant_subspace_residual":float(residual),"l2_pi_formula":"angles between sqrt(pi)*psi[:,1:3] and sqrt(pi)*[cos(theta),sin(theta)]; residual min_A ||P C-C A||_F,pi / ||C||_F,pi","diffusion_truncation_relative_rmse":float(np.sqrt(np.mean((trunc[pairs]-full[pairs])**2))/max(np.sqrt(np.mean(full[pairs]**2)),1e-15)),"spectral_direct_diffusion_distance_max_abs_error":float(np.max(np.abs(full-direct)))}

def generate_circle():
    theta=2*math.pi*np.arange(CIRCLE_N)/CIRCLE_N; held=(theta+math.pi*2/CIRCLE_N/2)%(2*math.pi)
    rng=rng_for(50); u=np.sort(rng.uniform(size=CIRCLE_N)); nonuniform=2*math.pi*(u**2.4)
    return {"uniform_theta":theta,"heldout_theta":held,"nonuniform_theta":nonuniform}


def add(rows,domain,condition,split,method,target,metric,value):
    value=float(value)
    if not np.isfinite(value): raise ValueError((condition,metric,value))
    rows.append(dict(domain=domain,condition=condition,split=split,method=method,target=target,metric=metric,value=value))

def signature(obj):
    arrays=[]
    def walk(x):
        if isinstance(x,np.ndarray): arrays.append(np.nan_to_num(x.ravel(),nan=9e299,posinf=8e299,neginf=-8e299))
        elif isinstance(x,dict):
            for k in sorted(x):
                if k not in ("ell","theta"): walk(x[k])
        elif isinstance(x,(float,int,np.floating,np.integer)): arrays.append(np.array([x],float))
    walk(obj);return np.concatenate(arrays) if arrays else np.zeros(1)

def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n");w.writeheader()
        for r in rows: z=dict(r);z["value"]=format(z["value"],".17g");w.writerow(z)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def json_default(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.integer,np.floating)): return x.item()
    raise TypeError(type(x).__name__)

def analytic_hairpin_oracle():
    c=math.pi*RADIUS
    joins=[BRANCH,BRANCH+c]
    continuity=[]
    for j in joins:
        left=hairpin_planar(np.array([j-1e-10]))[0]; exact=hairpin_planar(np.array([j]))[0];right=hairpin_planar(np.array([j+1e-10]))[0]
        continuity.append(max(float(np.linalg.norm(left-exact)),float(np.linalg.norm(right-exact))))
    grid=np.concatenate((np.linspace(.01,BRANCH-.01,80),np.linspace(BRANCH+.01,BRANCH+c-.01,80),np.linspace(BRANCH+c+.01,TOTAL-.01,80)))
    h=1e-6; speed=np.linalg.norm((hairpin_planar(grid+h)-hairpin_planar(grid-h))/(2*h),axis=1)
    q,_=embed_isometry()
    return {"join_continuity_max_error":max(continuity),"unit_speed_max_abs_error":float(np.max(np.abs(speed-1))),"isometry_gram_max_abs_error":float(np.max(np.abs(q.T@q-np.eye(2)))),"total_length":TOTAL}

def knn_tie_oracle():
    x=np.array([[0.0,0.0],[0.0,0.0],[1.0,0.0],[-1.0,0.0]])
    dist,order=knn_candidates(x,2); adj=knn_graph(x,2)
    no_self=bool(all(i not in order[i] for i in range(len(x))))
    stable_lower_index=bool(np.array_equal(order[0],np.array([1,2])) and np.array_equal(order[1],np.array([0,2])))
    union_symmetry=bool(np.array_equal(np.isfinite(adj),np.isfinite(adj.T)))
    retained=[]; max_weight_error=0.0
    for i in range(len(x)):
        for j in order[i]:
            retained.append([int(i),int(j),float(dist[i,j])])
            max_weight_error=max(max_weight_error,abs(float(adj[i,j])-float(dist[i,j])))
    return {"synthetic_points":x,"k":2,"directed_candidate_indices":order,"directed_candidate_records":retained,"no_self_neighbor_candidates":no_self,"stable_lower_index_tie_behavior":stable_lower_index,"union_symmetry":union_symmetry,"retained_edge_weight_max_abs_error":max_weight_error}

def reduced_separation_oracle(bundle):
    r=float(bundle["radius"]); total=float(bundle["total_length"]); endpoint=hairpin_planar(np.array([total]),r=r)[0]
    quantile_error=float(np.max(np.abs(bundle["ell"]/total-bundle["nominal_quantiles"])))
    lower_violation=max(0.0,float(-np.min(bundle["ell"])));upper_violation=max(0.0,float(np.max(bundle["ell"])-total))
    return {"radius":r,"total_length":total,"expected_endpoint":bundle["endpoint"],"evaluated_endpoint":endpoint,"endpoint_max_abs_error":float(np.max(np.abs(endpoint-bundle["endpoint"]))),"sample_min_ell":float(np.min(bundle["ell"])),"sample_max_ell":float(np.max(bundle["ell"])),"domain_upper_bound":total,"max_domain_violation":max(lower_violation,upper_violation),"nominal_quantile_reuse_max_abs_error":quantile_error,"straight_branch_length":BRANCH}

def graph_oracles(fit):
    adj=fit["adj"];d=fit["graph_dist"]; finite=np.isfinite(d)
    symmetry=float(np.max(np.abs(adj[np.isfinite(adj)]-adj.T[np.isfinite(adj)])))
    dsym=float(np.max(np.abs(d[finite]-d.T[finite])))
    diag=float(np.max(np.abs(np.diag(d))))
    rng=rng_for(70); tri=0.0
    for _ in range(500):
        i,j,k=rng.integers(0,len(d),3)
        if np.isfinite(d[i,j]) and np.isfinite(d[i,k]) and np.isfinite(d[k,j]): tri=max(tri,float(d[i,j]-d[i,k]-d[k,j]))
    return {"adjacency_symmetry_max_abs_error":symmetry,"shortest_path_symmetry_max_abs_error":dsym,"shortest_path_diagonal_max_abs_error":diag,"triangle_max_positive_violation_sampled":max(0.0,tri),"finite_for_connected":bool(len(fit["components"])!=1 or np.all(np.isfinite(d))),"explicit_inf_for_disconnected":bool(len(fit["components"])==1 or np.isinf(d).any())}

def method_oracles(fit,trainx):
    z1=fit["pca"].ravel();z2=fit["ambient_mds"]["coords"].ravel();coef=fit_affine(z2,z1); replay=apply_affine(z2,coef)
    p1=np.outer(z1/np.linalg.norm(z1),z1/np.linalg.norm(z1));p2=np.outer(z2/np.linalg.norm(z2),z2/np.linalg.norm(z2))
    amb_replay=gower_extend(fit["ambient_dist"],fit["ambient_mds"])
    iso_replay=gower_extend(fit["graph_dist"],fit["isomap"]) if fit["isomap"] is not None else None
    return {"pca_ambient_cmds_coordinate_affine_rmse":rmse(replay,z1),"pca_ambient_cmds_sample_projector_max_abs_error":float(np.max(np.abs(p1-p2))),"ambient_cmds_training_extension_replay_max_abs_error":float(np.max(np.abs(amb_replay-fit["ambient_mds"]["coords"]))),"isomap_training_extension_replay_max_abs_error":float(np.max(np.abs(iso_replay-fit["isomap"]["coords"]))) if iso_replay is not None else 0.0}

def circle_oracles(model,theta):
    replay,_=nystrom(model,model["x"]); replay_modes=slice(0,3)
    replay_error=float(np.max(np.abs(replay[:,replay_modes]-model["psi"][:,replay_modes])))
    n=len(theta); first=model["w"][0]; circulant=np.vstack([np.roll(first,i) for i in range(n)])
    firstp=model["p"][0]; circulantp=np.vstack([np.roll(firstp,i) for i in range(n)])
    row_sum_error=float(np.max(np.abs(model["p"].sum(1)-1.0)))
    stationary_error=float(np.max(np.abs(model["pi"]@model["p"]-model["pi"])))
    detailed=model["pi"][:,None]*model["p"]
    reversibility_error=float(np.max(np.abs(detailed-detailed.T)))
    gram=model["psi"].T@(model["pi"][:,None]*model["psi"])
    gram_error=float(np.max(np.abs(gram-np.eye(len(model["psi"])))))
    spectral=diffusion_distances(model);direct=direct_diffusion_distances(model)
    return {"affinity_circulant_max_abs_error":float(np.max(np.abs(model["w"]-circulant))),"markov_circulant_max_abs_error":float(np.max(np.abs(model["p"]-circulantp))),"diffusion_training_nystrom_replay_max_abs_error":replay_error,"markov_row_sum_max_abs_error":row_sum_error,"stationary_distribution_max_abs_error":stationary_error,"detailed_balance_max_abs_error":reversibility_error,"l2_pi_eigenfunction_gram_max_abs_error":gram_error,"spectral_direct_diffusion_distance_max_abs_error":float(np.max(np.abs(spectral-direct)))}

def permutation_oracle(hair,fit,circle_theta,circle_model):
    perm=rng_for(71).permutation(len(hair["x"])); inv=np.argsort(perm); f2=fit_hairpin(hair["x"][perm],fit["k"])
    hair_err=float(np.max(np.abs(fit["graph_dist"]-f2["graph_dist"][inv][:,inv])))
    cp=rng_for(72).permutation(len(circle_theta));cinv=np.argsort(cp);m2=diffusion_fit(circle_points(circle_theta[cp]),circle_model["epsilon"])
    p1=model_projector(circle_model["psi"][:,1:3]);p2=model_projector(m2["psi"][cinv,1:3])
    return {"hairpin_graph_distance_relabel_max_abs_error":hair_err,"circle_two_mode_projector_relabel_max_abs_error":float(np.max(np.abs(p1-p2)))}

def model_projector(x):
    q=np.linalg.qr(x)[0];return q@q.T

def exact_regeneration(hair,circle):
    h2=generate_hairpin();c2=generate_circle(); exact=True;records={}
    for split in HAIRPIN_SIZES:
        ok=np.array_equal(hair["nominal"][split]["ell"],h2["nominal"][split]["ell"]) and np.array_equal(hair["nominal"][split]["x"],h2["nominal"][split]["x"]);records[split]=ok;exact&=ok
    for name in hair["controls"]:
        ok=np.array_equal(hair["controls"][name]["ell"],h2["controls"][name]["ell"]) and np.array_equal(hair["controls"][name]["x"],h2["controls"][name]["x"]);records[name]=ok;exact&=ok
    for key in circle:
        ok=np.array_equal(circle[key],c2[key]);records[key]=ok;exact&=ok
    return {"all_exact":bool(exact),"records":records}

def leakage_oracles(hair,fit,exts,circle,cm):
    poisoned={s:{"x":hair["nominal"][s]["x"].copy(),"ell":np.full_like(hair["nominal"][s]["ell"],9999.)} for s in HAIRPIN_SIZES}
    f2=fit_hairpin(poisoned["train"]["x"]); e2={s:extend_hairpin(f2,poisoned[s]["x"],poisoned["train"]["x"]) for s in HAIRPIN_SIZES}
    e2["train"]={"pca":f2["pca"],"ambient_cmds":f2["ambient_mds"]["coords"],"isomap":f2["isomap"]["coords"]}
    truth_fit=float(np.max(np.abs(signature(fit)-signature(f2)))); truth_ext=max(float(np.max(np.abs(exts[s][m]-e2[s][m]))) for s in HAIRPIN_SIZES for m in ("pca","ambient_cmds","isomap"))
    poisoned_test={split:{"x":hair["nominal"][split]["x"].copy(),"ell":hair["nominal"][split]["ell"].copy()} for split in HAIRPIN_SIZES};poisoned_test["test"]["x"]=poisoned_test["test"]["x"]*100+77
    f3=fit_hairpin(poisoned_test["train"]["x"]); test_fit=float(np.max(np.abs(signature(fit)-signature(f3))))
    train_x=cm["x"].copy(); held_x=circle_points(circle["heldout_theta"]); original_extension,_=nystrom(cm,held_x)
    poisoned_theta=(circle["uniform_theta"]+0.371)%(2*math.pi); poisoned_truth=np.column_stack((np.cos(poisoned_theta),np.sin(poisoned_theta))); original_truth=np.column_stack((np.cos(circle["uniform_theta"]),np.sin(circle["uniform_theta"])))
    cm2=diffusion_fit(train_x,cm["epsilon"]); poisoned_extension,_=nystrom(cm2,held_x)
    return {"truth_poison_hairpin_fit_signature_max_abs_error":truth_fit,"truth_poison_hairpin_raw_extensions_max_abs_error":truth_ext,"test_observation_poison_training_fit_signature_max_abs_error":test_fit,"circle_truth_poison_truth_max_abs_change":float(np.max(np.abs(poisoned_truth-original_truth))),"circle_truth_poison_fit_signature_max_abs_error":float(np.max(np.abs(signature(cm)-signature(cm2)))),"circle_truth_poison_raw_nystrom_max_abs_error":float(np.max(np.abs(original_extension-poisoned_extension))),"circle_truth_poison_observed_train_x_exact":bool(np.array_equal(train_x,cm2["x"])),"circle_truth_poison_observed_heldout_x_unchanged":True}

def build(root,quiet=False):
    root.mkdir(parents=True,exist_ok=True); rows=[]; hair=generate_hairpin();fit,exts,aligned,coefs=fit_score_hairpin(hair)
    gm=graph_metrics(fit["adj"],fit["graph_dist"],hair["nominal"]["train"]["ell"])
    for k,v in gm.items(): add(rows,"hairpin","nominal","train","isomap_graph","arc_length",k,v)
    for name,model in (("ambient_cmds",fit["ambient_mds"]),("isomap",fit["isomap"])):
        add(rows,"hairpin","nominal","train",name,"mds","negative_eigenmass",model["negative_eigenmass"]);add(rows,"hairpin","nominal","train",name,"mds","strain",model["strain"])
    for method in ("pca","ambient_cmds","isomap"):
        for split in HAIRPIN_SIZES: add(rows,"hairpin","nominal",split,method,"affine_aligned_arc_length","coordinate_rmse",rmse(aligned[method][split],hair["nominal"][split]["ell"]))
    mean=train_mean=hair["nominal"]["train"]["ell"].mean()
    for split in HAIRPIN_SIZES: add(rows,"hairpin","nominal",split,"mean","arc_length","coordinate_rmse",rmse(np.full(HAIRPIN_SIZES[split],mean),hair["nominal"][split]["ell"]))
    anisotropic_transform=np.diag([1.0,1.0,3.0]); anisotropic_bundle={"ell":hair["nominal"]["train"]["ell"],"x":hair["nominal"]["train"]["x"]@anisotropic_transform}
    controls={
      "sparse_connector_gap":(hair["controls"]["sparse_gap"],NOMINAL_K),"reduced_branch_separation":(hair["controls"]["reduced_branch_separation"],NOMINAL_K),"nonuniform_sampling":(hair["controls"]["nonuniform"],NOMINAL_K),
      "disconnected_small_k":(hair["nominal"]["train"],1),"wrong_k":(hair["nominal"]["train"],18),"nearly_complete_large_k":(hair["nominal"]["train"],len(hair["nominal"]["train"]["x"])-2),"anisotropic_feature_scaling_z3":(anisotropic_bundle,NOMINAL_K)}
    control_diag={}
    nominal_edges=np.isfinite(fit["adj"])&(~np.eye(len(fit["adj"]),dtype=bool))
    for name,(bundle,k) in controls.items():
        f=fit_hairpin(bundle["x"],k); m=graph_metrics(f["adj"],f["graph_dist"],bundle["ell"])
        if name=="anisotropic_feature_scaling_z3":
            transformed_edges=np.isfinite(f["adj"])&(~np.eye(len(f["adj"]),dtype=bool)); changed=int(np.sum(nominal_edges!=transformed_edges)//2)
            m.update({"ambient_feature_transform":anisotropic_transform,"undirected_neighbor_edge_symmetric_difference_count":changed,"undirected_neighbor_edge_symmetric_difference_rate":changed/max(int(np.sum(nominal_edges)//2),1)})
        control_diag[name]=m
        for key,val in m.items():
            if np.isscalar(val): add(rows,"hairpin",name,"train","isomap_graph","control",key,val)
        if f["isomap"] is not None:
            add(rows,"hairpin",name,"train","isomap","mds","negative_eigenmass",f["isomap"]["negative_eigenmass"]);add(rows,"hairpin",name,"train","isomap","mds","strain",f["isomap"]["strain"])
    circle=generate_circle(); cm,cscore=circle_analysis(circle["uniform_theta"]); heldpsi,_=nystrom(cm,circle_points(circle["heldout_theta"])); truth=np.column_stack((np.cos(circle["uniform_theta"]),np.sin(circle["uniform_theta"]))); modes=cm["psi"][:,1:3]; coef=np.linalg.solve(modes.T@(cm["pi"][:,None]*modes),modes.T@(cm["pi"][:,None]*truth)); heldtruth=np.column_stack((np.cos(circle["heldout_theta"]),np.sin(circle["heldout_theta"]))); heldrmse=rmse(heldpsi[:,1:3]@coef,heldtruth)
    for i,a in enumerate(cscore["l2_pi_principal_angles_degrees"]): add(rows,"circle","uniform","train","diffusion_maps","l2_pi_fourier_two_mode_eigenspace",f"l2_pi_principal_angle_{i+1}_degrees",a)
    add(rows,"circle","uniform","train","diffusion_maps","operator","l2_pi_p_invariant_subspace_residual",cscore["l2_pi_p_invariant_subspace_residual"]);add(rows,"circle","uniform","train","diffusion_maps","diffusion_distance","two_mode_truncation_relative_rmse",cscore["diffusion_truncation_relative_rmse"]);add(rows,"circle","uniform","train","diffusion_maps","diffusion_distance","spectral_direct_max_abs_error",cscore["spectral_direct_diffusion_distance_max_abs_error"]);add(rows,"circle","uniform","test","diffusion_maps_nystrom","cos_sin_modes","heldout_midpoint_rmse",heldrmse)
    for i,v in enumerate(cm["values"][:8]): add(rows,"circle","uniform","train","diffusion_maps","spectrum",f"eigenvalue_{i}",v)
    circle_controls={"nonuniform_density":(circle["nonuniform_theta"],EPSILON),"epsilon_too_small":(circle["uniform_theta"],.005),"epsilon_too_large":(circle["uniform_theta"],5.0)};ccdiag={}
    for name,(theta,eps) in circle_controls.items():
        mod,sc=circle_analysis(theta,eps)
        full_control_distance=diffusion_distances(mod); control_pairs=np.triu(np.ones(full_control_distance.shape,bool),1)
        full_distance_rms=float(np.sqrt(np.mean(full_control_distance[control_pairs]**2)))
        ccdiag[name]={"epsilon":eps,**sc,"leading_eigenvalues":mod["values"][:4],"full_diffusion_distance_rms":full_distance_rms}
        add(rows,"circle",name,"train","diffusion_maps","operator","l2_pi_p_invariant_subspace_residual",sc["l2_pi_p_invariant_subspace_residual"]);add(rows,"circle",name,"train","diffusion_maps","diffusion_distance","two_mode_truncation_relative_rmse",sc["diffusion_truncation_relative_rmse"])
        add(rows,"circle",name,"train","diffusion_maps","diffusion_distance","full_spectrum_pairwise_rms",full_distance_rms)
        for i,v in enumerate(mod["values"][:4]): add(rows,"circle",name,"train","diffusion_maps","spectrum",f"eigenvalue_{i}",v)
        for i,a in enumerate(sc["l2_pi_principal_angles_degrees"]): add(rows,"circle",name,"train","diffusion_maps","l2_pi_fourier_two_mode_eigenspace",f"l2_pi_principal_angle_{i+1}_degrees",a)
    hor=analytic_hairpin_oracle();go=graph_oracles(fit);tie=knn_tie_oracle();reduced_oracle=reduced_separation_oracle(hair["controls"]["reduced_branch_separation"]);mo=method_oracles(fit,hair["nominal"]["train"]["x"]);co=circle_oracles(cm,circle["uniform_theta"]);po=permutation_oracle(hair["nominal"]["train"],fit,circle["uniform_theta"],cm);regen=exact_regeneration(hair,circle);leak=leakage_oracles(hair,fit,exts,circle,cm)
    for key in ("markov_row_sum_max_abs_error","stationary_distribution_max_abs_error","detailed_balance_max_abs_error","l2_pi_eigenfunction_gram_max_abs_error","spectral_direct_diffusion_distance_max_abs_error"):
        add(rows,"circle","uniform","train","diffusion_maps","markov_operator_oracle",key,co[key])
    for key in ("radius","total_length","endpoint_max_abs_error","sample_min_ell","sample_max_ell","domain_upper_bound","max_domain_violation","nominal_quantile_reuse_max_abs_error","straight_branch_length"):
        add(rows,"hairpin","reduced_branch_separation","train","generator_oracle","reduced_geometry",key,reduced_oracle[key])
    for key in ("no_self_neighbor_candidates","stable_lower_index_tie_behavior","union_symmetry","retained_edge_weight_max_abs_error"):
        add(rows,"hairpin","synthetic_exact_ties","train","knn_oracle","tie_handling",key,float(tie[key]))
    diagnostics={"generator":{"master_seed":MASTER_SEED,"hairpin":{"branch_length":BRANCH,"radius":RADIUS,"gap":GAP,"total_length":TOTAL,"noise_sd":NOISE_SD,"sizes":HAIRPIN_SIZES,"fixed_isometry":embed_isometry()[0],"fixed_offset":embed_isometry()[1]},"circle":{"training_count":CIRCLE_N,"heldout_midpoints":CIRCLE_N,"epsilon":EPSILON,"alpha":1,"diffusion_time":CIRCLE_TIME}},"nominal_hairpin":{"k":NOMINAL_K,"graph_metrics":gm,"affine_coefficients":coefs},"hairpin_controls":control_diag,"circle_nominal":{"scores":cscore,"heldout_nystrom_rmse":heldrmse,"eigenvalues":cm["values"],"markov_transition_p":cm["p"],"stationary_pi":cm["pi"],"degree":cm["degree"]},"circle_controls":ccdiag,"oracles":{"analytic_hairpin":hor,"reduced_branch_separation":reduced_oracle,"knn_exact_ties":tie,"graph":go,"pca_cmds_and_extension":mo,"circle":co,"permutation_relabel":po,"exact_regeneration":regen,"leakage":leak},"circle_representation_contract":{"weighting":"L2(pi)","principal_angles_formula":"Euclidean principal angles after multiplying both right eigenfunctions and cos/sin truth by sqrt(pi)","invariant_residual_formula":"min_A ||P C-C A||_F,pi / ||C||_F,pi"},"interpretation_boundary":"Finite graph/operator recovery only; no proof of topology, manifold, intrinsic dimension, geodesic or Riemannian geometry, unique coordinates, dynamics, mechanism, or causality."}
    write_csv(root/"metrics.csv",rows);(root/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True,allow_nan=False,default=json_default)+"\n")
    make_summary(root/"summary.png",hair,fit,aligned,circle,cm,diagnostics)
    hashes={n:sha(root/n) for n in ARTIFACT_NAMES};manifest={"schema_version":1,"artifact":"graph-geometry-recovery","date":DATE,"master_seed":MASTER_SEED,"runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"pillow":getattr(sys.modules.get("PIL"),"__version__","unknown"),"platform":platform.platform()},"hairpin":{"branch_length":BRANCH,"radius":RADIUS,"gap":GAP,"total_length":TOTAL,"noise_sd":NOISE_SD,"nominal_k":NOMINAL_K,"split_sizes":HAIRPIN_SIZES,"long_arc_separation_threshold":LONG_ARC_EDGE_THRESHOLD,"material_path_shortcut_underestimation_threshold":PATH_SHORTCUT_UNDERESTIMATION_THRESHOLD,"reduced_separation":{"radius":0.06,"total_length":6.0+math.pi*0.06,"sampling":"reuse nominal sorted quantiles ell/TOTAL","endpoint":[0.12,0.0]}},"circle":{"n":CIRCLE_N,"epsilon":EPSILON,"alpha":1,"diffusion_time":CIRCLE_TIME,"representation_scoring":"L2(pi)-weighted right-eigenfunction angles and P-invariant-subspace residual"},"methods":["mean","pca_1d","ambient_cmds_1d","isomap","diffusion_maps_alpha_1","gower_extension","nystrom_extension"],"default_command":f"{sys.executable} toy-models/graph-geometry-recovery.py","explicit_commands":{"generate":f"{sys.executable} toy-models/graph-geometry-recovery.py --generate","verify":f"{sys.executable} toy-models/graph-geometry-recovery.py --verify","generate_and_verify":f"{sys.executable} toy-models/graph-geometry-recovery.py --generate --verify"},"anisotropic_control":{"condition":"anisotropic_feature_scaling_z3","transform":[[1,0,0],[0,1,0],[0,0,3]]},"metric_rows":len(rows),"metric_fields":FIELDS,"tolerances":TOL,"artifact_hashes_sha256":hashes,"artifact_files":["manifest.json",*ARTIFACT_NAMES],"interpretation_boundary":diagnostics["interpretation_boundary"]};(root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False,default=json_default)+"\n")
    if not quiet: print(f"wrote {root}\nnominal Isomap test RMSE: {rmse(aligned['isomap']['test'],hair['nominal']['test']['ell']):.6f}\ncircle Nystrom heldout RMSE: {heldrmse:.6g}\nmetric rows: {len(rows)}")
    return manifest,diagnostics,rows

def make_summary(path,hair,fit,aligned,circle,cm,diag):
    im=Image.new("RGB",(1500,980),"white");d=ImageDraw.Draw(im);font=ImageFont.load_default();d.text((22,14),"LDG graph geometry recovery - finite diagnostic",fill="black",font=font)
    def panel(box,title): d.rectangle(box,outline="black",width=2);d.text((box[0]+8,box[1]+7),title,fill="black",font=font)
    panel((22,42,740,480),"A. Hairpin training graph and fixed kNN edges")
    xy=hair["nominal"]["train"]["x"][:,:2];lo=xy.min(0);hi=xy.max(0)
    def pix(z): return (45+(z[0]-lo[0])/max(hi[0]-lo[0],1e-12)*660,445-(z[1]-lo[1])/max(hi[1]-lo[1],1e-12)*360)
    edge=np.isfinite(fit["adj"])&(np.arange(len(xy))[:,None]<np.arange(len(xy))[None,:])
    for i,j in np.argwhere(edge): d.line((pix(xy[i]),pix(xy[j])),fill=(210,210,210),width=1)
    for z in xy:
        x,y=pix(z);d.ellipse((x-2,y-2,x+2,y+2),fill=(31,119,180))
    panel((765,42,1475,480),"B. Hairpin held-out aligned coordinate errors")
    colors={"pca":(31,119,180),"ambient_cmds":(44,160,44),"isomap":(214,39,40)}
    truth=hair["nominal"]["test"]["ell"];order=np.argsort(truth)
    for method in colors:
        pts=[]
        for i in order:
            px=795+truth[i]/TOTAL*630; py=445-aligned[method]["test"][i]/TOTAL*350;pts.append((px,py))
        d.line(pts,fill=colors[method],width=2)
    d.line((795,445,1425,95),fill="black",width=1);d.text((790,455),"black=truth; blue=PCA; green=ambient cMDS; red=Isomap",fill="black",font=font)
    panel((22,505,740,950),"C. Circle first two nonconstant diffusion modes")
    psi=cm["psi"][:,1:3];theta=circle["uniform_theta"]
    for i in range(len(theta)):
        x=380+psi[i,0]/max(np.max(np.abs(psi[:,0])),1e-12)*260;y=730-psi[i,1]/max(np.max(np.abs(psi[:,1])),1e-12)*180
        d.ellipse((x-3,y-3,x+3,y+3),fill=(120,int(80+150*i/len(theta)),180))
    d.text((40,920),"Display of finite operator modes; eigenspace metrics are numerical.",fill="black",font=font)
    panel((765,505,1475,950),"D. Numeric controls and interpretation boundary")
    gm=diag["nominal_hairpin"]["graph_metrics"];cs=diag["circle_nominal"]["scores"];o=diag["oracles"]
    lines=[f"nominal components: {gm['component_count']}",f"nominal shortcut rate: {gm['material_path_shortcut_rate']:.4g}",f"graph-distance relative RMSE: {gm['graph_distance_relative_rmse']:.4g}",f"circle max principal angle (deg): {max(cs['l2_pi_principal_angles_degrees']):.4g}",f"circle L2(pi) P residual: {cs['l2_pi_p_invariant_subspace_residual']:.3g}",f"Nystrom held-out RMSE: {diag['circle_nominal']['heldout_nystrom_rmse']:.4g}",f"PCA/cMDS projector error: {o['pca_cmds_and_extension']['pca_ambient_cmds_sample_projector_max_abs_error']:.3g}",f"relabel max error: {max(o['permutation_relabel'].values()):.3g}","Finite graph/operator recovery only.","No topology, manifold, intrinsic dimension, geodesic,","Riemannian geometry, dynamics, mechanism, or causal claim."]
    for i,line in enumerate(lines): d.text((785,550+32*i),line,fill="black",font=font)
    im.save(path,format="PNG",optimize=False)

def verify_metrics(path,expected):
    with path.open(newline="",encoding="utf-8") as f: reader=csv.DictReader(f);rows=list(reader)
    if tuple(reader.fieldnames or ())!=FIELDS or len(rows)!=expected: raise AssertionError("metric schema/count")
    seen=set()
    for r in rows:
        if any(r[k]=="" for k in FIELDS): raise AssertionError("empty metric label")
        if not np.isfinite(float(r["value"])): raise AssertionError("nonfinite metric")
        key=tuple(r[k] for k in FIELDS[:-1])
        if key in seen: raise AssertionError(f"duplicate metric {key}")
        seen.add(key)

def verify_metric_consistency(path,diag):
    with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    lookup={(r["domain"],r["condition"],r["split"],r["method"],r["target"],r["metric"]):float(r["value"]) for r in rows}
    graph_records={"nominal":diag["nominal_hairpin"]["graph_metrics"],**diag["hairpin_controls"]}
    graph_metrics_to_check=("edge_count","long_arc_separation_edge_count","long_arc_separation_edge_rate","long_arc_separation_threshold","long_arc_separation_denominator_edges","material_path_shortcut_count","material_path_shortcut_rate","material_path_shortcut_underestimation_threshold","material_path_shortcut_denominator_edges","graph_distance_relative_rmse","graph_distance_correlation","component_count")
    for condition,record in graph_records.items():
        target="arc_length" if condition=="nominal" else "control"
        for metric in graph_metrics_to_check:
            key=("hairpin",condition,"train","isomap_graph",target,metric)
            if key not in lookup or lookup[key]!=float(record[metric]): raise AssertionError(f"graph diagnostics/metrics mismatch: {key}")
    circle_records={"uniform":diag["circle_nominal"]["scores"],**diag["circle_controls"]}
    for condition,record in circle_records.items():
        for index,value in enumerate(record["l2_pi_principal_angles_degrees"],start=1):
            key=("circle",condition,"train","diffusion_maps","l2_pi_fourier_two_mode_eigenspace",f"l2_pi_principal_angle_{index}_degrees")
            if key not in lookup or lookup[key]!=float(value): raise AssertionError(f"circle angle diagnostics/metrics mismatch: {key}")
        key=("circle",condition,"train","diffusion_maps","operator","l2_pi_p_invariant_subspace_residual")
        if key not in lookup or lookup[key]!=float(record["l2_pi_p_invariant_subspace_residual"]): raise AssertionError(f"circle residual diagnostics/metrics mismatch: {condition}")


def verify(root,recompute=True):
    required=("manifest.json",*ARTIFACT_NAMES)
    if sorted(p.name for p in root.iterdir() if p.is_file())!=sorted(required): raise AssertionError("artifact inventory is not exactly four files")
    man=json.loads((root/"manifest.json").read_text());diag=json.loads((root/"diagnostics.json").read_text())
    if man["date"]!=DATE or man["master_seed"]!=MASTER_SEED: raise AssertionError("manifest contract")
    for n,h in man["artifact_hashes_sha256"].items():
        if sha(root/n)!=h: raise AssertionError(f"hash {n}")
    verify_metrics(root/"metrics.csv",man["metric_rows"])
    verify_metric_consistency(root/"metrics.csv",diag)
    o=diag["oracles"]
    if o["analytic_hairpin"]["join_continuity_max_error"]>2e-9 or o["analytic_hairpin"]["unit_speed_max_abs_error"]>2e-8: raise AssertionError("hairpin analytic oracle")
    g=o["graph"]
    if g["adjacency_symmetry_max_abs_error"]>TOL["symmetry"] or g["shortest_path_symmetry_max_abs_error"]>TOL["symmetry"] or g["shortest_path_diagonal_max_abs_error"]>TOL["diagonal"] or g["triangle_max_positive_violation_sampled"]>TOL["triangle"]: raise AssertionError("graph algebra oracle")
    if not g["finite_for_connected"] or not diag["hairpin_controls"]["disconnected_small_k"]["infinite_pair_count"]>0: raise AssertionError("connectivity/inf oracle")
    tie=o["knn_exact_ties"]
    if not tie["no_self_neighbor_candidates"] or not tie["stable_lower_index_tie_behavior"] or not tie["union_symmetry"] or tie["retained_edge_weight_max_abs_error"]>TOL["oracle"]: raise AssertionError("exact-tie kNN oracle")
    reduced=o["reduced_branch_separation"]
    expected_reduced_total=6.0+math.pi*0.06
    if abs(reduced["radius"]-0.06)>TOL["oracle"] or abs(reduced["total_length"]-expected_reduced_total)>TOL["oracle"] or reduced["endpoint_max_abs_error"]>TOL["oracle"] or reduced["max_domain_violation"]>TOL["oracle"] or reduced["nominal_quantile_reuse_max_abs_error"]>TOL["oracle"]: raise AssertionError("reduced-separation generator oracle")
    required_graph_metrics={"long_arc_separation_edge_count","long_arc_separation_edge_rate","long_arc_separation_threshold","long_arc_separation_denominator_edges","material_path_shortcut_count","material_path_shortcut_rate","material_path_shortcut_underestimation_threshold","material_path_shortcut_denominator_edges"}
    for condition,record in {"nominal":diag["nominal_hairpin"]["graph_metrics"],**diag["hairpin_controls"]}.items():
        if not required_graph_metrics.issubset(record): raise AssertionError(f"missing edge semantics: {condition}")
        if record["long_arc_separation_threshold"]!=LONG_ARC_EDGE_THRESHOLD or record["material_path_shortcut_underestimation_threshold"]!=PATH_SHORTCUT_UNDERESTIMATION_THRESHOLD: raise AssertionError(f"edge threshold mismatch: {condition}")
        if record["long_arc_separation_denominator_edges"]!=record["edge_count"] or record["material_path_shortcut_denominator_edges"]!=record["edge_count"]: raise AssertionError(f"edge denominator mismatch: {condition}")
    m=o["pca_cmds_and_extension"]
    if m["pca_ambient_cmds_sample_projector_max_abs_error"]>TOL["projector"] or m["ambient_cmds_training_extension_replay_max_abs_error"]>TOL["oracle"] or m["isomap_training_extension_replay_max_abs_error"]>TOL["oracle"]: raise AssertionError("MDS/PCA oracle")
    c=o["circle"]
    if c["affinity_circulant_max_abs_error"]>TOL["fourier"] or c["markov_circulant_max_abs_error"]>TOL["fourier"] or c["diffusion_training_nystrom_replay_max_abs_error"]>TOL["oracle"]: raise AssertionError("circle operator oracle")
    for key in ("markov_row_sum_max_abs_error","stationary_distribution_max_abs_error","detailed_balance_max_abs_error","l2_pi_eigenfunction_gram_max_abs_error","spectral_direct_diffusion_distance_max_abs_error"):
        if c[key]>TOL["oracle"]: raise AssertionError(f"circle Markov/diffusion oracle: {key}")
    if diag["circle_nominal"]["scores"]["l2_pi_p_invariant_subspace_residual"]>TOL["fourier"] or diag["circle_nominal"]["scores"]["spectral_direct_diffusion_distance_max_abs_error"]>TOL["oracle"]: raise AssertionError("weighted Fourier/direct diffusion residual")
    if diag["circle_representation_contract"]["weighting"]!="L2(pi)": raise AssertionError("circle representation weighting")
    for condition,score in {"uniform":diag["circle_nominal"]["scores"],**diag["circle_controls"]}.items():
        angles=score["l2_pi_principal_angles_degrees"]; residual=score["l2_pi_p_invariant_subspace_residual"]
        if len(angles)!=2 or not np.all(np.isfinite(angles)) or not np.isfinite(residual): raise AssertionError(f"nonfinite L2(pi) circle score: {condition}")
    anisotropic=diag["hairpin_controls"]["anisotropic_feature_scaling_z3"]
    if anisotropic["undirected_neighbor_edge_symmetric_difference_count"]<=0: raise AssertionError("anisotropic transform did not change graph neighbors")
    leak=o["leakage"]
    zero_leak_keys=("truth_poison_hairpin_fit_signature_max_abs_error","truth_poison_hairpin_raw_extensions_max_abs_error","test_observation_poison_training_fit_signature_max_abs_error","circle_truth_poison_fit_signature_max_abs_error","circle_truth_poison_raw_nystrom_max_abs_error")
    if max(leak[key] for key in zero_leak_keys)>TOL["poison"] or leak["circle_truth_poison_truth_max_abs_change"]<=0 or not leak["circle_truth_poison_observed_train_x_exact"] or not leak["circle_truth_poison_observed_heldout_x_unchanged"]: raise AssertionError("executable leakage/truth-poison oracle")
    if max(o["permutation_relabel"].values())>TOL["oracle"] or not o["exact_regeneration"]["all_exact"]: raise AssertionError("relabel/regeneration oracle")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="ldg-ggr-") as td:
            build(Path(td),quiet=True)
            for n in required:
                if (root/n).read_bytes()!=(Path(td)/n).read_bytes(): raise AssertionError(f"byte regeneration {n}")
    print("verification passed: graph/operator algebra, extensions, leakage, relabeling, metrics, hashes, and byte regeneration")

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--generate",action="store_true");p.add_argument("--verify",action="store_true");p.add_argument("--no-recompute",action="store_true");p.add_argument("--artifact-root",type=Path,default=DEFAULT_ROOT);a=p.parse_args();g=a.generate;v=a.verify
    if not g and not v:g=v=True
    if g:build(a.artifact_root)
    if v:verify(a.artifact_root,recompute=not a.no_recompute)
    return 0
if __name__=="__main__": raise SystemExit(main())
