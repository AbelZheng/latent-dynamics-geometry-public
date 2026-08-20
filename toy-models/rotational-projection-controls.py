#!/usr/bin/env python3
"""Deterministic rotational-projection controls using only NumPy and Pillow."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASTER_SEED=20260819; DATE="2026-08-19"; K=6; P=10; NT=61; DT=0.04
TIME=np.arange(NT)*DT; NCOND=8; TRIALS={"train":10,"validation":5,"test":7}
SCENARIOS=("positive_full_arc","positive_short_window","jittered_alignment","smoothed","no_condition_mean_subtraction","symmetric_decay","latency_sequence","input_driven_state_only","input_driven_input_aware","nonorthogonal_mixing","covariance_condition_shuffle")
ARTIFACT_NAMES=("metrics.csv","diagnostics.json","summary.png")
FIELDS=("split","condition","preprocessing","method","target","metric","value")
TOL={"algebra":1e-9,"poison":1e-12,"skew":1e-10,"finite_difference":0.003,"determinism":0.0}
ROOT=Path(__file__).resolve().parent; DEFAULT_ROOT=ROOT/"artifacts"/"rotational-projection-controls"

def rng_for(*words): return np.random.Generator(np.random.PCG64(np.random.SeedSequence((MASTER_SEED,)+tuple(words))))
def blockdiag(*blocks):
    n=sum(len(b) for b in blocks); out=np.zeros((n,n)); i=0
    for b in blocks: out[i:i+len(b),i:i+len(b)]=b; i+=len(b)
    return out
def generator_matrices():
    rot=lambda w:np.array([[0.,-w],[w,0.]])
    return {"positive":blockdiag(rot(2.),rot(.7),np.diag([-.4,-.9])),"symmetric":np.diag([-1.2,-.8,-.5,-.35,-.2,-.1])}
def embedding():
    q,r=np.linalg.qr(rng_for(1).normal(size=(P,K))); return q*np.where(np.diag(r)<0,-1,1)
def nonorthogonal_embedding(h):
    s=np.array([[1,.35,0,0,0,0],[0,1.6,.2,0,0,0],[0,0,.7,.25,0,0],[0,0,0,1.3,.2,0],[0,0,0,0,.8,.3],[0,0,0,0,0,1.8]])
    return h@s
def initial_states(): return rng_for(2).normal(size=(NCOND,K))*np.array([1.2,1.2,.8,.8,.55,.55])
def analytic_positive(z0,t):
    z=np.empty((len(t),K));
    for j,w in enumerate((2.,.7)):
        c=np.cos(w*t); s=np.sin(w*t); a,b=z0[2*j:2*j+2]
        z[:,2*j]=c*a-s*b; z[:,2*j+1]=s*a+c*b
    z[:,4]=z0[4]*np.exp(-.4*t); z[:,5]=z0[5]*np.exp(-.9*t); return z
def analytic_derivative_positive(z): return z@generator_matrices()["positive"].T
def symmetric_trajectory(z0,t):
    rates=np.diag(generator_matrices()["symmetric"]); return z0[None,:]*np.exp(t[:,None]*rates)
def input_signal(condition,t):
    phase=.35*condition; return np.column_stack([np.sin(1.3*t+phase),np.exp(-((t-(.8+.04*condition))/.28)**2)])
def input_trajectory(z0,t,condition):
    a=np.diag([-.25,-.3,-.35,-.4,-.45,-.5]); b=np.array([[.8,0],[0,.7],[.5,.3],[-.3,.5],[.2,-.4],[.4,.25]])
    z=np.empty((len(t),K)); z[0]=z0; u=input_signal(condition,t)
    for i in range(len(t)-1): z[i+1]=z[i]+(t[i+1]-t[i])*(a@z[i]+b@u[i])
    return z,u,a,b
def latency_observed(condition,t,p):
    centers=np.linspace(.35,1.7,p)+.025*condition
    widths=np.linspace(.13,.25,p); signs=np.where(np.arange(p)%2, -1.,1.)
    return signs[None,:]*np.exp(-((t[:,None]-centers[None,:])/widths[None,:])**2)*(1+.08*condition)
def clean_condition(name):
    h=embedding(); mix=nonorthogonal_embedding(h) if name=="nonorthogonal_mixing" else h; z0s=initial_states(); xs=[]; zs=[]; us=[]
    for c,z0 in enumerate(z0s):
        if name=="symmetric_decay": z=symmetric_trajectory(z0,TIME); x=z@mix.T; u=np.zeros((NT,2))
        elif name=="latency_sequence": z=np.zeros((NT,K)); x=latency_observed(c,TIME,P); u=np.zeros((NT,2))
        elif name.startswith("input_driven"): z,u,_,_=input_trajectory(z0,TIME,c); x=z@mix.T
        else: z=analytic_positive(z0,TIME); x=z@mix.T; u=np.zeros((NT,2))
        xs.append(x); zs.append(z); us.append(u)
    return np.array(xs),np.array(zs),np.array(us),mix,h
def generate_trials(base_name,split,jitter=False):
    x,z,u,mix,h=clean_condition(base_name); n=TRIALS[split]; code={"train":1,"validation":2,"test":3}[split]
    out=np.empty((NCOND,n,NT,P)); latent=np.empty((NCOND,n,NT,K)); shifts=np.zeros((NCOND,n),dtype=int)
    for c in range(NCOND):
        for r in range(n):
            shift=int(rng_for(40,code,c,r).integers(-3,4)) if jitter else 0
            shifts[c,r]=shift
            if jitter:
                shifted_time=TIME+shift*DT
                trial_z=analytic_positive(initial_states()[c],shifted_time)
                trial_x=trial_z@mix.T
            else:
                trial_z=z[c]; trial_x=x[c]
            noise_code=8 if base_name.startswith("input_driven") else (SCENARIOS.index(base_name) if base_name in SCENARIOS else 99)
            noise=rng_for(10,noise_code,code,c,r).normal(scale=.12,size=(NT,P))
            out[c,r]=trial_x+noise; latent[c,r]=trial_z
    return {"observed":out,"clean":x,"latent":latent,"input":u,"mix":mix,"isometric_H":h,
            "trial_ids":np.arange(NCOND*n).reshape(NCOND,n)+code*100000,"jitter_shifts":shifts,
            "jitter_policy":"analytic evaluation at TIME + integer_shift*DT before adding the unchanged deterministic trial noise; no wrapping"}

def moving_average(x):
    y=x.copy(); y[...,1:-1,:]=(x[...,:-2,:]+x[...,1:-1,:]+x[...,2:,:])/3; return y
def condition_means(data): return np.mean(data["observed"],axis=1)
def preprocess_fit(means,subtract=True,smooth=False,window=None):
    y=moving_average(means) if smooth else means.copy(); idx=np.arange(NT) if window is None else np.asarray(window)
    y=y[:,idx]
    time_mean=np.mean(y,axis=0,keepdims=True) if subtract else np.zeros((1,len(idx),P)); y=y-time_mean
    center=np.mean(y,axis=(0,1)); yc=y-center
    matrix=yc.reshape(-1,P); _,_,vt=np.linalg.svd(matrix,full_matrices=False); basis=vt[:K].T
    return {"basis":basis,"center":center,"time_mean":time_mean,"indices":idx,"subtract":subtract,"smooth":smooth}
def preprocess_apply(means,fit):
    y=moving_average(means) if fit["smooth"] else means.copy(); y=y[:,fit["indices"]]-fit["time_mean"]-fit["center"]
    return y@fit["basis"]
def derivatives(scores,times):
    x=[]; d=[]; groups=[]
    for c,path in enumerate(scores):
        dt=np.diff(times); x.append((path[1:]+path[:-1])/2); d.append(np.diff(path,axis=0)/dt[:,None]); groups.extend([c]*(len(path)-1))
    return np.vstack(x),np.vstack(d),np.array(groups)
def fit_linear(x,d): return np.linalg.lstsq(x,d,rcond=None)[0].T
def matrix_basis(kind,d):
    mats=[]
    if kind=="skew":
        for i in range(d):
            for j in range(i+1,d): m=np.zeros((d,d));m[i,j]=1;m[j,i]=-1;mats.append(m)
    else:
        for i in range(d):
            for j in range(i,d): m=np.zeros((d,d));m[i,j]=m[j,i]=1 if i==j else .5;mats.append(m)
    return mats
def fit_constrained(x,d,kind):
    mats=matrix_basis(kind,x.shape[1]); design=np.column_stack([(x@m.T).reshape(-1) for m in mats]); coef=np.linalg.lstsq(design,d.reshape(-1),rcond=None)[0]
    return sum(c*m for c,m in zip(coef,mats)),coef
def fit_input_aware(x,d,u):
    design=np.column_stack([x,u]); coef=np.linalg.lstsq(design,d,rcond=None)[0]; return coef[:x.shape[1]].T,coef[x.shape[1]:].T

def fit_affine(x,d,u=None):
    design=np.column_stack([x,np.ones(len(x))]) if u is None else np.column_stack([x,u,np.ones(len(x))])
    coef=np.linalg.lstsq(design,d,rcond=None)[0]
    state=coef[:x.shape[1]].T
    if u is None:return {"state":state,"input":None,"intercept":coef[-1],"design":design,"coef":coef}
    return {"state":state,"input":coef[x.shape[1]:x.shape[1]+u.shape[1]].T,"intercept":coef[-1],"design":design,"coef":coef}

def input_secants(scores,inputs):
    x=[];d=[];u=[];groups=[]
    for c,path in enumerate(scores):
        x.append(path[:-1]);d.append(np.diff(path,axis=0)/DT);u.append(inputs[c,:-1]);groups.extend([c]*(len(path)-1))
    return np.vstack(x),np.vstack(d),np.vstack(u),np.array(groups)
def deterministic_vector_from_projector(projector):
    norms=np.linalg.norm(projector,axis=0); index=int(np.argmax(norms)); q=projector[:,index]/norms[index]
    pivot=int(np.argmax(np.abs(q))); q*=1.0 if q[pivot]>=0 else -1.0
    return q

def extract_skew_planes(m,count=2):
    values,vectors=np.linalg.eigh(-(m@m)); order=np.argsort(values)[::-1]; planes=[]
    for rank in range(count):
        pair=order[2*rank:2*rank+2]; projector=vectors[:,pair]@vectors[:,pair].T
        q1=deterministic_vector_from_projector(projector)
        rate=math.sqrt(max(float(q1@(-m@m)@q1),0.0))
        if rate<=1e-12:
            q2=deterministic_vector_from_projector(projector-q1[:,None]@q1[None,:]); signed=0.0
        else:
            q2=m@q1/rate; q2/=np.linalg.norm(q2); signed=-float(q1@m@q2)
        q=np.column_stack([q1,q2])
        planes.append({"basis":q,"projector":q@q.T,"signed_rate":signed,"absolute_rate":abs(signed),
                       "eigenvalue":float(np.mean(values[pair])),"orientation_bilinear":float(q1@m@q2)})
    return planes

def plane_angle(a,b):
    qa,_=np.linalg.qr(a);qb,_=np.linalg.qr(b);s=np.linalg.svd(qa.T@qb,compute_uv=False);return float(np.degrees(np.max(np.arccos(np.clip(s,-1,1)))))
def prediction_metrics(d,p):
    mse=float(np.mean((d-p)**2)); denom=float(np.sum((d-np.mean(d,axis=0))**2)); r2=1-float(np.sum((d-p)**2))/max(denom,1e-15);return mse,r2
def plane_metrics(condition_scores,times,plane):
    projected_sum=0.;total_sum=0.; tangential_sum=0.;radial_sum=0.;valid_count=0;perpendicular_count=0;spans=[];signed=[];angular_rates=[]
    for path in condition_scores:
        x=(path[1:]+path[:-1])/2; d=np.diff(path,axis=0)/np.diff(times)[:,None]
        xp=x@plane; dp=d@plane; radius=np.linalg.norm(xp,axis=1); valid=radius>1e-8
        projected_sum+=float(np.sum(xp**2));total_sum+=float(np.sum(x**2))
        if np.any(valid):
            xv=xp[valid];dv=dp[valid];rv=radius[valid];unit_r=xv/rv[:,None]
            radial_component=np.sum(dv*unit_r,axis=1)[:,None]*unit_r;tangent_component=dv-radial_component
            radial_sum+=float(np.sum(radial_component**2));tangential_sum+=float(np.sum(tangent_component**2));valid_count+=int(np.sum(valid))
            norms=np.linalg.norm(dv,axis=1);cosine=np.abs(np.sum(xv*dv,axis=1))/(rv*norms+1e-15)
            perpendicular_count+=int(np.sum(cosine<=math.sin(math.radians(15))))
            angle=np.unwrap(np.arctan2(xv[:,1],xv[:,0]));spans.append(float(np.max(angle)-np.min(angle)))
            rates=(xv[:,0]*dv[:,1]-xv[:,1]*dv[:,0])/(rv**2)
            angular_rates.append(float(np.mean(rates)));signed.append(float(np.mean(xv[:,0]*dv[:,1]-xv[:,1]*dv[:,0])))
        else:spans.append(0.);angular_rates.append(0.);signed.append(0.)
    mean_t=tangential_sum/max(valid_count,1);mean_r=radial_sum/max(valid_count,1)
    return {"plane_occupancy":projected_sum/max(total_sum,1e-15),"tangential_energy":mean_t,"radial_energy":mean_r,
            "tangential_radial_energy_ratio":mean_t/max(mean_r,1e-15),"perpendicular_fraction_within_15deg":perpendicular_count/max(valid_count,1),
            "condition_spans_radians":spans,"condition_span_mean_radians":float(np.mean(spans)),"condition_span_median_radians":float(np.median(spans)),
            "condition_span_min_radians":float(np.min(spans)),"condition_span_max_radians":float(np.max(spans)),
            "signed_direction_consistency":float(max(np.mean(np.array(signed)>=0),np.mean(np.array(signed)<=0))),
            "condition_signed_angular_rate_mean":float(np.mean(angular_rates)),"condition_signed_angular_rate_sd":float(np.std(angular_rates)),
            "signed_cross_product_by_condition":signed,"condition_signed_angular_rates":angular_rates,"valid_radius_pair_count":valid_count}

def fixed_plane(): return np.eye(K)[:,:2]
def random_planes():
    out=[]
    for i in range(5): q,_=np.linalg.qr(rng_for(30,i).normal(size=(K,2)));out.append(q)
    return out
def add(rows,split,condition,prep,method,target,metric,value): rows.append(dict(split=split,condition=condition,preprocessing=prep,method=method,target=target,metric=metric,value=float(value)))

def covariance_similarity(left,right):
    a=np.cov(left.reshape(-1,P),rowvar=False);b=np.cov(right.reshape(-1,P),rowvar=False)
    return float(1.0-np.linalg.norm(a-b,"fro")**2/max(np.linalg.norm(a,"fro")**2,1e-15))

def covariance_shuffle(means,split_code):
    proposals=[];best=None
    for proposal in range(128):
        out=np.empty_like(means);perms=[]
        for feature in range(P):
            perm=rng_for(41,split_code,proposal,feature).permutation(NCOND);perms.append(perm.tolist());out[:,:,feature]=means[perm,:,feature]
        score=covariance_similarity(means,out);record={"proposal":proposal,"covariance_similarity":score,"feature_permutations":perms}
        proposals.append(record)
        if best is None or score>best["covariance_similarity"] or (score==best["covariance_similarity"] and proposal<best["proposal"]):best={**record,"shuffled":out}
    multiset=True
    for f in range(P):
        before=sorted(hashlib.sha256(means[c,:,f].tobytes()).hexdigest() for c in range(NCOND));after=sorted(hashlib.sha256(best["shuffled"][c,:,f].tobytes()).hexdigest() for c in range(NCOND));multiset &= before==after
    distribution=float(np.max(np.abs(np.sort(means.reshape(-1,P),axis=0)-np.sort(best["shuffled"].reshape(-1,P),axis=0))))
    return best["shuffled"],{"proposal_scores":[{"proposal":r["proposal"],"covariance_similarity":r["covariance_similarity"]} for r in proposals],"selected_proposal":best["proposal"],"selected_feature_permutations":best["feature_permutations"],"threshold":.95,"achieved_similarity":best["covariance_similarity"],"threshold_miss":best["covariance_similarity"]<.95,"feature_trajectory_multisets_preserved":bool(multiset),"marginal_feature_distribution_max_abs":distribution}

def scenario_config(name):
    base=name
    if name in ("positive_full_arc","positive_short_window","jittered_alignment","smoothed","no_condition_mean_subtraction","covariance_condition_shuffle"):base="positive_full_arc"
    subtract=name not in ("no_condition_mean_subtraction","input_driven_state_only","input_driven_input_aware")
    return {"base":base,"subtract":subtract,"smooth":name=="smoothed","window":np.arange(18) if name=="positive_short_window" else None,
            "jitter":name=="jittered_alignment","shuffle":name=="covariance_condition_shuffle","input_control":name.startswith("input_driven")}

def plane_error(a,b):
    qa,_=np.linalg.qr(a); qb,_=np.linalg.qr(b); singular=np.linalg.svd(qa.T@qb,compute_uv=False)
    angles=np.arccos(np.clip(singular,-1,1)); return {"max_angle_degrees":float(np.degrees(np.max(angles))),"rms_sine_error":float(np.sqrt(np.mean(np.sin(angles)**2))),"projector_frobenius_error":float(np.linalg.norm(qa@qa.T-qb@qb.T,"fro"))}

def match_two_planes(recovered,true_planes):
    candidates=[]
    for permutation in ((0,1),(1,0)):
        errors=[plane_error(recovered[i]["basis"],true_planes[permutation[i]]) for i in range(2)]
        candidates.append({"permutation":list(permutation),"errors":errors,"total_max_angle":sum(e["max_angle_degrees"] for e in errors)})
    return min(candidates,key=lambda r:(r["total_max_angle"],r["permutation"]))

def emit_plane_metrics(rows,name,method,target,scores,times,plane,preprocessing="train_fit"):
    record=plane_metrics(scores,times,plane)
    for metric in ("plane_occupancy","tangential_energy","radial_energy","tangential_radial_energy_ratio","perpendicular_fraction_within_15deg",
                   "condition_span_mean_radians","condition_span_median_radians","condition_span_min_radians","condition_span_max_radians",
                   "signed_direction_consistency","condition_signed_angular_rate_mean","condition_signed_angular_rate_sd"):
        add(rows,"test",name,preprocessing,method,target,metric,record[metric])
    for c,value in enumerate(record["condition_spans_radians"]):add(rows,"test",name,preprocessing,method,f"condition_{c}","condition_span_radians",value)
    return record

def input_expected_coefficients(fit,mix):
    a=np.diag([-.25,-.3,-.35,-.4,-.45,-.5]);b=np.array([[.8,0],[0,.7],[.5,.3],[-.3,.5],[.2,-.4],[.4,.25]])
    w=fit["basis"];r=mix.T@w;rinv=np.linalg.inv(r);center_score=fit["center"]@w
    m_t=rinv@a.T@r; g_t=b.T@r; intercept=center_score@m_t
    return {"M":m_t.T,"G":g_t.T,"intercept":intercept,"R":r,"A":a,"B":b}

def fit_split_planes(means,cfg):
    fit=preprocess_fit(means,cfg["subtract"],cfg["smooth"],cfg["window"]);scores=preprocess_apply(means,fit);times=TIME[fit["indices"]]
    x,d,_=derivatives(scores,times);sk,_=fit_constrained(x,d,"skew");return fit,scores,extract_skew_planes(sk,2)

def observed_plane_records(fit,planes): return [{"basis":fit["basis"]@p["basis"],"signed_rate":p["signed_rate"]} for p in planes]

def compare_observed_plane_sets(reference,candidate):
    match=match_two_planes(reference,[p["basis"] for p in candidate]);out=[]
    for i,j in enumerate(match["permutation"]):out.append({"angle_degrees":match["errors"][i]["max_angle_degrees"],"signed_rate_difference":candidate[j]["signed_rate"]-reference[i]["signed_rate"],"absolute_rate_difference":abs(candidate[j]["signed_rate"]-reference[i]["signed_rate"])})
    return {"permutation":match["permutation"],"planes":out,"total_angle":match["total_max_angle"]}

def evaluate_scenario(name,rows):
    cfg=scenario_config(name);data={s:generate_trials(cfg["base"],s,jitter=cfg["jitter"]) for s in TRIALS};means={s:condition_means(data[s]) for s in TRIALS};control={}
    if cfg["jitter"]:
        control["trial_level_shifts"]={s:data[s]["jitter_shifts"].tolist() for s in TRIALS};control["multiple_distinct_shifts"]={s:len(np.unique(data[s]["jitter_shifts"]))>1 for s in TRIALS};control["no_circular_wrap"]={s:True for s in TRIALS};control["shift_policy"]=data["train"]["jitter_policy"]
    if cfg["shuffle"]:
        for split_code,split in enumerate(means,1):means[split],control[split]=covariance_shuffle(means[split],split_code)
        control["all_feature_trajectory_multisets_preserved"]=all(control[s]["feature_trajectory_multisets_preserved"] for s in means);control["maximum_marginal_feature_distribution_max_abs"]=max(control[s]["marginal_feature_distribution_max_abs"] for s in means)
    fit=preprocess_fit(means["train"],cfg["subtract"],cfg["smooth"],cfg["window"]);scores={s:preprocess_apply(means[s],fit) for s in means};times=TIME[fit["indices"]]
    input_model=None;input_expected=None;input_oracle=None
    if cfg["input_control"]:
        secants={s:input_secants(scores[s],data[s]["input"]) for s in scores};xtr,dtr,utr,_=secants["train"]
        state_affine=fit_affine(xtr,dtr);un=state_affine["state"];state_intercept=state_affine["intercept"]
        aware_affine=fit_affine(xtr,dtr,utr);input_expected=input_expected_coefficients(fit,data["train"]["mix"])
        input_model=(aware_affine["state"],aware_affine["input"],aware_affine["intercept"]) if name=="input_driven_input_aware" else None
        input_oracle={"M_error":float(np.linalg.norm(aware_affine["state"]-input_expected["M"])),"G_error":float(np.linalg.norm(aware_affine["input"]-input_expected["G"])),"intercept_error":float(np.linalg.norm(aware_affine["intercept"]-input_expected["intercept"]))} if name=="input_driven_input_aware" else None
        if input_oracle is not None:
            clean_scores=(data["train"]["clean"]-fit["center"])@fit["basis"];cx,cd,cu,_=input_secants(clean_scores,data["train"]["input"]);expected_pred=cx@input_expected["M"].T+cu@input_expected["G"].T+input_expected["intercept"];input_oracle["euler_local_consistency_max_abs"]=float(np.max(np.abs(cd-expected_pred)))
        xd={s:(secants[s][0],secants[s][1],secants[s][3]) for s in secants}
    else:
        xd={s:derivatives(scores[s],times) for s in scores};xtr,dtr,_=xd["train"];un=fit_linear(xtr,dtr);state_intercept=np.zeros(K)
    sk,skcoef=fit_constrained(xtr,dtr,"skew");sy,sycoef=fit_constrained(xtr,dtr,"symmetric");planes=extract_skew_planes(sk,2)
    for split in ("train","validation","test"):
        x,d,_=xd[split]
        predictions={"unrestricted_affine" if cfg["input_control"] else "unrestricted":x@un.T+state_intercept,"skew_constrained":x@sk.T,"symmetric_constrained":x@sy.T}
        if input_model is not None:
            u=input_secants(scores[split],data[split]["input"])[2];predictions["input_aware_affine"]=x@input_model[0].T+u@input_model[1].T+input_model[2]
        for method,pred in predictions.items():
            mse,r2=prediction_metrics(d,pred);add(rows,split,name,"train_fit",method,"derivative_secant" if cfg["input_control"] else "heldout_derivative","derivative_mse",mse);add(rows,split,name,"train_fit",method,"derivative_secant" if cfg["input_control"] else "heldout_derivative","derivative_r2",r2)
    plane_records=[]
    for index,plane in enumerate(planes,1):
        label=f"skew_plane_{index}";metric=emit_plane_metrics(rows,name,label,"projected_trajectory",scores["test"],times,plane["basis"])
        for key in ("signed_rate","absolute_rate"):add(rows,"test",name,"train_fit",label,"skew_generator",key,plane[key])
        plane_records.append(metric)
    positive_derived=cfg["base"]=="positive_full_arc" and name!="nonorthogonal_mixing";mapped_stress=name=="nonorthogonal_mixing";matching=None
    if positive_derived or mapped_stress:
        true_planes=[fit["basis"].T@data["train"]["mix"][:,0:2],fit["basis"].T@data["train"]["mix"][:,2:4]];matching=match_two_planes(planes,true_planes)
        for recovered_index,true_index in enumerate(matching["permutation"]):
            label=f"skew_plane_{recovered_index+1}";target=("mapped_nonorthogonal_true_plane" if mapped_stress else "true_positive_pca_plane")+f"_{true_index+1}";error=matching["errors"][recovered_index]
            for metric,key in (("plane_max_angle_degrees","max_angle_degrees"),("plane_rms_sine_error","rms_sine_error"),("plane_projector_frobenius_error","projector_frobenius_error")):add(rows,"test",name,"train_fit",label,target,metric,error[key])
            if not mapped_stress:
                true_rate=(2.0,.7)[true_index];add(rows,"test",name,"train_fit",label,target,"signed_rate_error",planes[recovered_index]["signed_rate"]-true_rate);add(rows,"test",name,"train_fit",label,target,"absolute_rate_error",abs(planes[recovered_index]["absolute_rate"]-true_rate));add(rows,"test",name,"train_fit",label,target,"relative_rate_error",abs(planes[recovered_index]["absolute_rate"]-true_rate)/true_rate)
    baselines=[("fixed_plane",fixed_plane())]+[(f"random_plane_{i}",q) for i,q in enumerate(random_planes())]
    for label,plane in baselines:emit_plane_metrics(rows,name,label,"baseline_plane",scores["test"],times,plane)
    stability={}
    if positive_derived:
        split_records={"train":observed_plane_records(fit,planes)}
        for split in ("validation","test"):
            sfit,_,splanes=fit_split_planes(means[split],cfg);split_records[split]=observed_plane_records(sfit,splanes);comparison=compare_observed_plane_sets(split_records["train"],split_records[split]);stability[f"train_{split}"]=comparison
            for i,p in enumerate(comparison["planes"],1):add(rows,split,name,"independent_fit",f"skew_plane_{i}",f"train_{split}_observed_space_stability","plane_angle_degrees",p["angle_degrees"]);add(rows,split,name,"independent_fit",f"skew_plane_{i}",f"train_{split}_observed_space_stability","signed_rate_difference",p["signed_rate_difference"])
        loo=[]
        full_observed=split_records["train"]
        for held in range(NCOND):
            reduced=np.delete(means["train"],held,axis=0);lfit,_,lplanes=fit_split_planes(reduced,cfg);comparison=compare_observed_plane_sets(full_observed,observed_plane_records(lfit,lplanes));loo.append({"held_out_condition":held,**comparison})
            for i,p in enumerate(comparison["planes"],1):add(rows,"train",name,"leave_one_condition_out",f"skew_plane_{i}",f"held_out_condition_{held}","plane_angle_degrees",p["angle_degrees"]);add(rows,"train",name,"leave_one_condition_out",f"skew_plane_{i}",f"held_out_condition_{held}","absolute_rate_change",p["absolute_rate_difference"])
        stability["leave_one_condition_out"]=loo
    single_trial={}
    if positive_derived:
        for pi,plane in enumerate(planes,1):
            vals=[]
            for c in range(NCOND):
                trial_observed=data["test"]["observed"][c]
                if fit["smooth"]:trial_observed=moving_average(trial_observed)
                trial_observed=trial_observed[:,fit["indices"]]
                transformed=(trial_observed-fit["time_mean"][0]-fit["center"])@fit["basis"]
                for trial in transformed:vals.append(plane_metrics(trial[None,:,:],times,plane["basis"]))
            pooled={k:float(np.mean([v[k] for v in vals])) for k in ("tangential_energy","radial_energy","tangential_radial_energy_ratio","perpendicular_fraction_within_15deg")}
            single_trial[f"plane_{pi}"]=pooled
            for metric,value in pooled.items():add(rows,"test",name,"frozen_training_plane_single_trial",f"skew_plane_{pi}","test_trials",metric,value)
    input_rank=0
    if cfg["input_control"]:input_rank=int(np.linalg.matrix_rank(input_secants(scores["train"],data["train"]["input"])[2]))
    if input_oracle:
        for metric,value in input_oracle.items():add(rows,"train",name,"expected_coefficient_oracle","input_aware_affine","known_euler_model",metric,value)
    return {"fit":fit,"means":means,"scores":scores,"matrices":{"unrestricted":un,"skew":sk,"symmetric":sy},"intercept":state_intercept,"coefficients":{"skew":skcoef,"symmetric":sycoef},"planes":planes,"plane_metrics":plane_records,"matching":matching,"stability":stability,"single_trial":single_trial,"applicability":{"positive_true_planes":positive_derived,"nonorthogonal_mapped_plane_stress":mapped_stress,"true_plane_metrics_omitted":not(positive_derived or mapped_stress),"latent_rate_errors_omitted":mapped_stress,"autonomous_interpretation":cfg["base"]=="positive_full_arc","input_aware":name=="input_driven_input_aware","state_only_input_excluded":name=="input_driven_state_only","latency_has_no_state_law":cfg["base"]=="latency_sequence"},"control":control,"data":data,"derivatives":xd,"input_model":input_model,"input_expected":input_expected,"input_oracle":input_oracle,"input_design_rank":input_rank}

def matrix_fit_diagnostics(design,target,prediction):
    residual=target-prediction
    return {"design_rank":int(np.linalg.matrix_rank(design)),"design_condition_number":float(np.linalg.cond(design)),
            "training_sse":float(np.sum(residual**2)),"residual_orthogonality_max_abs":float(np.max(np.abs(design.T@residual))),
            "all_finite":bool(np.all(np.isfinite(design)) and np.all(np.isfinite(prediction)))}

def oracles(results):
    h=embedding();iso=float(np.max(np.abs(h.T@h-np.eye(K))));a=generator_matrices()["positive"]
    block_sign={"a01":float(a[0,1]),"a10":float(a[1,0]),"a23":float(a[2,3]),"a32":float(a[3,2]),"pass":bool(a[0,1]==-2 and a[1,0]==2 and a[2,3]==-.7 and a[3,2]==.7)}
    z=analytic_positive(initial_states()[0],TIME);analytic=analytic_derivative_positive(z);fd=np.diff(z,axis=0)/DT;mid=(analytic[1:]+analytic[:-1])/2;fd_err=float(np.max(np.abs(fd-mid)))
    records={};max_constraint=0;max_basis=0;max_pca=0;max_plane_invariance=0;min_orientation=float("inf");max_rate_identity=0;max_mq=0
    for name,r in results.items():
        x,d,g=r["derivatives"]["train"];un=r["matrices"]["unrestricted"];sk=r["matrices"]["skew"];sy=r["matrices"]["symmetric"]
        skew_err=float(np.max(np.abs(sk+sk.T)));sym_err=float(np.max(np.abs(sy-sy.T)));max_constraint=max(max_constraint,skew_err,sym_err)
        skew_mats=matrix_basis("skew",K);skew_design=np.column_stack([(x@m.T).reshape(-1) for m in skew_mats]);skew_coef=np.linalg.lstsq(skew_design,d.reshape(-1),rcond=None)[0];skew_rec=sum(c*m for c,m in zip(skew_coef,skew_mats))
        sym_mats=matrix_basis("symmetric",K);sym_design=np.column_stack([(x@m.T).reshape(-1) for m in sym_mats]);sym_coef=np.linalg.lstsq(sym_design,d.reshape(-1),rcond=None)[0];sym_rec=sum(c*m for c,m in zip(sym_coef,sym_mats))
        un_rec=fit_affine(x,d)["state"] if name.startswith("input_driven") else np.linalg.lstsq(x,d,rcond=None)[0].T
        basis_err=max(float(np.max(np.abs(skew_rec-sk))),float(np.max(np.abs(sym_rec-sy))),float(np.max(np.abs(un_rec-un))));max_basis=max(max_basis,basis_err)
        fit=r["fit"];y=r["means"]["train"]
        if fit["smooth"]:y=moving_average(y)
        y=y[:,fit["indices"]]-fit["time_mean"]-fit["center"];_,singular,vt=np.linalg.svd(y.reshape(-1,P),full_matrices=False);independent=vt[:K].T;pca_err=float(np.max(np.abs(fit["basis"]@fit["basis"].T-independent@independent.T)));max_pca=max(max_pca,pca_err)
        losses={"unrestricted":float(np.sum((d-x@un.T-r["intercept"])**2)),"skew":float(np.sum((d-x@sk.T)**2)),"symmetric":float(np.sum((d-x@sy.T)**2))}
        transitions=bool(np.all(g[:-1]<=g[1:]));reextract=extract_skew_planes(sk,2);plane_checks=[]
        for stored,fresh in zip(r["planes"],reextract):
            invariance=float(np.max(np.abs(stored["projector"]-fresh["projector"])));orientation=stored["orientation_bilinear"];rate_identity=abs(stored["signed_rate"]+orientation)
            q=stored["basis"];rate=stored["signed_rate"];j=np.array([[0.,-rate],[rate,0.]]);mq=float(np.max(np.abs(sk@q-q@j)))
            rotation=np.array([[0.,-1.],[1.,0.]]);variants=[q@rotation,q[:,::-1],q*np.array([[-1.,1.]])];angle_invariance=max(plane_angle(q,v) for v in variants)
            max_plane_invariance=max(max_plane_invariance,invariance);min_orientation=min(min_orientation,orientation);max_rate_identity=max(max_rate_identity,rate_identity);max_mq=max(max_mq,mq)
            plane_checks.append({"projector_reextract_max_abs":invariance,"orientation_bilinear":orientation,"orientation_negative":orientation<0,"signed_rate_positive":rate>=0,"signed_rate_identity_error":rate_identity,"orthonormality_max_abs":float(np.max(np.abs(q.T@q-np.eye(2)))),"MQ_equals_QJ_max_abs":mq,"angle_invariance_rotation_swap_sign_max_degrees":angle_invariance})
        match_check=None
        if r["matching"] is not None:
            true_planes=[fit["basis"].T@r["data"]["train"]["mix"][:,0:2],fit["basis"].T@r["data"]["train"]["mix"][:,2:4]];totals=[sum(plane_angle(r["planes"][i]["basis"],true_planes[p[i]]) for i in range(2)) for p in ((0,1),(1,0))];match_check={"difference":abs(r["matching"]["total_max_angle"]-min(totals))}
        pair_expected=NCOND*(len(fit["indices"])-1)
        predictors={"unrestricted":matrix_fit_diagnostics(np.column_stack([x,np.ones(len(x))]) if np.any(r["intercept"]) else x,d,x@un.T+r["intercept"]),
                    "skew":matrix_fit_diagnostics(skew_design,d.reshape(-1,1),(skew_design@skew_coef).reshape(-1,1)),
                    "symmetric":matrix_fit_diagnostics(sym_design,d.reshape(-1,1),(sym_design@sym_coef).reshape(-1,1))}
        if r["input_model"] is not None:
            ix,idiff,iu,_=input_secants(r["scores"]["train"],r["data"]["train"]["input"]);idesign=np.column_stack([ix,iu,np.ones(len(ix))]);ipred=ix@r["input_model"][0].T+iu@r["input_model"][1].T+r["input_model"][2];predictors["input_aware_affine"]=matrix_fit_diagnostics(idesign,idiff,ipred)
        split_ids={split:set(r["data"][split]["trial_ids"].reshape(-1).tolist()) for split in TRIALS}
        records[name]={"skew_constraint":skew_err,"symmetric_constraint":sym_err,"basis_recompute_max_abs":basis_err,"pca_svd_projector_max_abs":pca_err,"pca_rank":fit["basis"].shape[1],"pca_raw_numerical_rank":int(np.sum(singular>1e-10)),"pca_retained_fraction":float(np.sum(singular[:K]**2)/np.sum(singular**2)),"losses":losses,"unrestricted_sse_le_skew":losses["unrestricted"]<=losses["skew"]+TOL["algebra"],"unrestricted_sse_le_symmetric":losses["unrestricted"]<=losses["symmetric"]+TOL["algebra"],"condition_groups_monotone_no_cross_transition":transitions,"derivative_pair_count":len(x),"expected_derivative_pair_count":pair_expected,"trial_id_splits_disjoint":bool(split_ids["train"].isdisjoint(split_ids["validation"]) and split_ids["train"].isdisjoint(split_ids["test"]) and split_ids["validation"].isdisjoint(split_ids["test"])),"planes":plane_checks,"matching_recompute":match_check,"predictor_diagnostics":predictors}
    jitter=results["jittered_alignment"]["control"];ids={split:set(results["positive_full_arc"]["data"][split]["trial_ids"].reshape(-1).tolist()) for split in TRIALS}
    branch={"smoothed_indices_equal":bool(np.array_equal(results["smoothed"]["fit"]["indices"],results["positive_full_arc"]["fit"]["indices"])),"short_window_length":len(results["positive_short_window"]["fit"]["indices"]),"input_subtraction_disabled":not results["input_driven_state_only"]["fit"]["subtract"] and not results["input_driven_input_aware"]["fit"]["subtract"],"no_subtraction_branch":not results["no_condition_mean_subtraction"]["fit"]["subtract"]}
    return {"positive_block_sign_algebra":block_sign,"isometry_max_abs":iso,"analytic_midpoint_fd_max_abs":fd_err,"records":records,"max_constraint":max_constraint,"max_basis_recompute":max_basis,"max_pca_svd_projector":max_pca,"max_plane_projector_reextract":max_plane_invariance,"minimum_orientation_bilinear":min_orientation,"max_signed_rate_identity_error":max_rate_identity,"max_MQ_equals_QJ":max_mq,"jitter_checks":{"multiple_shifts_all_splits":all(jitter["multiple_distinct_shifts"].values()),"no_circular_wrap_all_splits":all(jitter["no_circular_wrap"].values()),"split_shift_arrays_differ":jitter["trial_level_shifts"]["train"]!=jitter["trial_level_shifts"]["test"]},"trial_id_splits_disjoint":bool(ids["train"].isdisjoint(ids["validation"]) and ids["train"].isdisjoint(ids["test"]) and ids["validation"].isdisjoint(ids["test"])),"preprocessing_branch_isolation":branch}

def scenario_snapshot(name,data,fixed_test,truth_bundle=None):
    cfg=scenario_config(name);means={s:condition_means(data[s]) for s in TRIALS};control={}
    if cfg["jitter"]:control["jitter_shifts"]={s:data[s]["jitter_shifts"].copy() for s in TRIALS}
    if cfg["shuffle"]:
        for code,split in enumerate(("train","validation"),1):means[split],control[split]=covariance_shuffle(means[split],code)
    fit=preprocess_fit(means["train"],cfg["subtract"],cfg["smooth"],cfg["window"]);scores={s:preprocess_apply(means[s],fit) for s in ("train","validation")};times=TIME[fit["indices"]]
    if cfg["input_control"]:x,d,u,_=input_secants(scores["train"],data["train"]["input"]);unfit=fit_affine(x,d);aware=fit_affine(x,d,u)
    else:x,d,_=derivatives(scores["train"],times);unfit={"state":fit_linear(x,d),"intercept":np.zeros(K)};aware=None
    sk,_=fit_constrained(x,d,"skew");sy,_=fit_constrained(x,d,"symmetric");planes=extract_skew_planes(sk,2)
    original_means=condition_means(fixed_test);original_scores=preprocess_apply(original_means,fit)
    if cfg["input_control"]:px,_,pu,_=input_secants(original_scores,fixed_test["input"]);prediction=px@(aware["state"] if aware else unfit["state"]).T+(pu@aware["input"].T if aware else 0)+(aware["intercept"] if aware else unfit["intercept"])
    else:px,_,_=derivatives(original_scores,times);prediction=px@unfit["state"].T+unfit["intercept"]
    return {"indices":fit["indices"],"time_mean":fit["time_mean"],"center":fit["center"],"pca_projector":fit["basis"]@fit["basis"].T,"unrestricted":unfit["state"],"intercept":unfit["intercept"],"skew":sk,"symmetric":sy,"plane_projectors":[p["projector"] for p in planes],"plane_rates":[p["signed_rate"] for p in planes],"input_state":None if aware is None else aware["state"],"input_coeff":None if aware is None else aware["input"],"input_intercept":None if aware is None else aware["intercept"],"fixed_original_test_prediction":prediction,"fixed_plane":fixed_plane(),"random_planes":random_planes(),"control":control,"truth_bundle_received_norm":0. if truth_bundle is None else float(sum(np.linalg.norm(v) for v in truth_bundle.values()))}

def snapshot_difference(a,b):
    fields={}
    for key in ("indices","time_mean","center","pca_projector","unrestricted","intercept","skew","symmetric","fixed_original_test_prediction","fixed_plane"):
        fields[key]=float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key]))))
    fields["plane_projectors"]=max(float(np.max(np.abs(x-y))) for x,y in zip(a["plane_projectors"],b["plane_projectors"]));fields["plane_rates"]=max(abs(x-y) for x,y in zip(a["plane_rates"],b["plane_rates"]));fields["random_planes"]=max(float(np.max(np.abs(x-y))) for x,y in zip(a["random_planes"],b["random_planes"]))
    for key in ("input_state","input_coeff","input_intercept"):fields[key]=0. if a[key] is None and b[key] is None else float(np.max(np.abs(a[key]-b[key])))
    control_diff=0.
    if "jitter_shifts" in a["control"]:control_diff=max(float(np.max(np.abs(a["control"]["jitter_shifts"][s]-b["control"]["jitter_shifts"][s]))) for s in TRIALS)
    for split in ("train","validation"):
        if split in a["control"]:
            control_diff=max(control_diff,max(abs(x["covariance_similarity"]-y["covariance_similarity"]) for x,y in zip(a["control"][split]["proposal_scores"],b["control"][split]["proposal_scores"])),float(a["control"][split]["selected_proposal"]!=b["control"][split]["selected_proposal"]))
    fields["jitter_shuffle_registry"]=control_diff
    return {"by_field":fields,"maximum":max(fields.values())}

def poison_check():
    records={}
    for name in SCENARIOS:
        cfg=scenario_config(name);data={s:generate_trials(cfg["base"],s,jitter=cfg["jitter"]) for s in TRIALS};fixed_test=data["test"];baseline=scenario_snapshot(name,data,fixed_test)
        poisoned={s:{k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in data[s].items()} for s in TRIALS};poisoned["test"]["observed"][:]=rng_for(70,SCENARIOS.index(name)).normal(loc=1e4,size=poisoned["test"]["observed"].shape)
        truth={"positive":generator_matrices()["positive"].copy(),"symmetric":generator_matrices()["symmetric"].copy(),"embedding":embedding().copy(),"nonorthogonal":nonorthogonal_embedding(embedding()).copy(),"initial_states":initial_states().copy(),"latent":poisoned["test"]["latent"].copy()}
        for i,key in enumerate(truth):truth[key][:]=rng_for(71,SCENARIOS.index(name),i).normal(loc=-1e4,size=truth[key].shape)
        comparison=scenario_snapshot(name,poisoned,fixed_test,truth)
        records[name]={"poisoned_test_observation_change":float(np.max(np.abs(poisoned["test"]["observed"]-data["test"]["observed"]))),"all_truth_poison_norm":float(sum(np.linalg.norm(v) for v in truth.values())),"truth_poison_received_by_harness_norm":comparison["truth_bundle_received_norm"],"fit_difference":snapshot_difference(baseline,comparison)}
    return {"records":records,"maximum_fit_discrepancy":max(r["fit_difference"]["maximum"] for r in records.values())}

def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n");w.writeheader()
        for r in rows:q=dict(r);q["value"]=format(q["value"],".17g");w.writerow(q)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def make_png(path,rows,diag,results):
    im=Image.new("RGB",(1500,960),"white");d=ImageDraw.Draw(im);font=ImageFont.load_default();d.text((24,14),"LDG rotational projection controls — deterministic diagnostic",fill="black",font=font)
    def panel(b,t):d.rectangle(b,outline="black",width=2);d.text((b[0]+8,b[1]+7),t,fill="black",font=font)
    panel((24,44,760,470),"A. Positive test condition mean in recovered plane 1")
    s=results["positive_full_arc"]["scores"]["test"][0]@results["positive_full_arc"]["planes"][0]["basis"];lo=s.min(0);hi=s.max(0);pts=[(55+(x-lo[0])/max(hi[0]-lo[0],1e-12)*670,430-(y-lo[1])/max(hi[1]-lo[1],1e-12)*330) for x,y in s];d.line(pts,fill=(31,119,180),width=3)
    panel((785,44,1475,470),"B. Held-out derivative R2 by condition")
    for i,name in enumerate(SCENARIOS):
        unrestricted="unrestricted_affine" if name.startswith("input_driven") else "unrestricted"
        vals=[next(float(r["value"]) for r in rows if r["split"]=="test" and r["condition"]==name and r["method"]==method and r["metric"]=="derivative_r2") for method in (unrestricted,"skew_constrained")]
        d.text((800,78+i*32),f"{name[:27]:27s} un {vals[0]: .3f}  skew {vals[1]: .3f}",fill="black",font=font)
    panel((24,495,760,930),"C. Both-plane recovery and whole-trajectory shuffle")
    positive=results["positive_full_arc"]; match=positive["matching"]; shuffle=results["covariance_condition_shuffle"]["control"]
    lines=[]
    for i,true_index in enumerate(match["permutation"]):
        plane=positive["planes"][i];err=match["errors"][i]; true_rate=(2.,.7)[true_index]
        lines.append(f"plane {i+1} -> true {true_index+1}: angle {err['max_angle_degrees']:.3g} deg; signed rate {plane['signed_rate']:.3g}; abs error {abs(plane['absolute_rate']-true_rate):.3g}")
    pmetric=positive["plane_metrics"][0]
    lines += [f"plane 1 occupancy/tangent-radial/perpendicular: {pmetric['plane_occupancy']:.3g} / {pmetric['tangential_radial_energy_ratio']:.3g} / {pmetric['perpendicular_fraction_within_15deg']:.3g}",
              f"preprocess controls jitter/smooth/no-sub R2: {next(float(r['value']) for r in rows if r['condition']=='jittered_alignment' and r['split']=='test' and r['method']=='skew_constrained' and r['metric']=='derivative_r2'):.3g} / {next(float(r['value']) for r in rows if r['condition']=='smoothed' and r['split']=='test' and r['method']=='skew_constrained' and r['metric']=='derivative_r2'):.3g} / {next(float(r['value']) for r in rows if r['condition']=='no_condition_mean_subtraction' and r['split']=='test' and r['method']=='skew_constrained' and r['metric']=='derivative_r2'):.3g}",
              f"shuffle covariance similarity train/val/test: {shuffle['train']['achieved_similarity']:.4f} / {shuffle['validation']['achieved_similarity']:.4f} / {shuffle['test']['achieved_similarity']:.4f} (threshold .95 MISS)",
              f"shuffle threshold .95 misses: {sum(shuffle[s]['threshold_miss'] for s in ('train','validation','test'))}",
              "State projections are not posterior distributions or mechanism evidence."]
    for i,x in enumerate(lines):d.text((42,540+i*48),x,fill="black",font=font)
    panel((785,495,1475,930),"D. Executable oracles and interpretation")
    o=diag["oracles"];lines=[f"H isometry max: {o['isometry_max_abs']:.3g}",f"analytic/midpoint-FD max: {o['analytic_midpoint_fd_max_abs']:.3g}",f"constraint / LS basis max: {o['max_constraint']:.3g} / {o['max_basis_recompute']:.3g}",f"two-plane projector reextract max: {o['max_plane_projector_reextract']:.3g}",f"poison complete-fit discrepancy: {diag['leakage']['maximum_fit_discrepancy']:.3g}","Latency and input-driven controls omit true autonomous-plane metrics.","A rotational projection is not proof of oscillator, autonomy, or mechanism."]
    for i,x in enumerate(lines):d.text((800,540+i*48),x,fill="black",font=font)
    im.save(path,format="PNG",optimize=False)

def seed_registry():
    return {"embedding":1,"initial_states":2,"trial_noise":10,"random_planes":30,"trial_jitter":40,
            "shuffle_proposals":41,"test_observation_poison":70,"truth_poison":71,
            "policy":"numpy PCG64 SeedSequence(master_seed, stage, split/scenario, condition, trial/proposal/feature)"}

def build(root,quiet=False):
    root.mkdir(parents=True,exist_ok=True);rows=[];results={}
    for name in SCENARIOS:results[name]=evaluate_scenario(name,rows)
    oracle=oracles(results);leak=poison_check();applicability={name:results[name]["applicability"] for name in SCENARIOS}
    state,aware=results["input_driven_state_only"],results["input_driven_input_aware"]
    shared_input={"train_observed_max_abs":float(np.max(np.abs(state["data"]["train"]["observed"]-aware["data"]["train"]["observed"]))),
                  "validation_observed_max_abs":float(np.max(np.abs(state["data"]["validation"]["observed"]-aware["data"]["validation"]["observed"]))),
                  "test_observed_max_abs":float(np.max(np.abs(state["data"]["test"]["observed"]-aware["data"]["test"]["observed"]))),
                  "clean_max_abs":float(np.max(np.abs(state["data"]["train"]["clean"]-aware["data"]["train"]["clean"]))),
                  "means_max_abs":max(float(np.max(np.abs(state["means"][split]-aware["means"][split]))) for split in TRIALS),
                  "preprocessing_center_max_abs":float(np.max(np.abs(state["fit"]["center"]-aware["fit"]["center"]))),
                  "preprocessing_pca_projector_max_abs":float(np.max(np.abs(state["fit"]["basis"]@state["fit"]["basis"].T-aware["fit"]["basis"]@aware["fit"]["basis"].T))),
                  "trial_ids_equal":all(np.array_equal(state["data"][split]["trial_ids"],aware["data"][split]["trial_ids"]) for split in TRIALS),
                  "regression_convention":"same byte-identical dataset; left-endpoint PCA scores and inputs; affine [x,1] versus [x,u,1]; no time-varying across-condition subtraction"}
    input_design={name:{"input_rank":results[name]["input_design_rank"],"input_aware_fit_present":results[name]["input_model"] is not None,
                        "input_coefficients_all_finite":bool(results[name]["input_model"] is None or all(np.all(np.isfinite(x)) for x in results[name]["input_model"])),
                        "expected_coefficient_oracle":results[name]["input_oracle"]} for name in ("input_driven_state_only","input_driven_input_aware")}
    plane_registry={name:{"planes":[{"projector":p["projector"].tolist(),"basis":p["basis"].tolist(),"signed_rate":p["signed_rate"],"absolute_rate":p["absolute_rate"],"orientation_bilinear":p["orientation_bilinear"]} for p in results[name]["planes"]],"matching":results[name]["matching"],"observed_space_stability":results[name]["stability"],"single_trial_metrics":results[name]["single_trial"]} for name in SCENARIOS}
    omitted={name:[] for name in SCENARIOS}
    for name in SCENARIOS:
        if results[name]["applicability"]["true_plane_metrics_omitted"]:omitted[name].append("true_plane_angles_and_rate_errors")
        if results[name]["applicability"]["latent_rate_errors_omitted"]:omitted[name].append("latent_signed_absolute_relative_rate_errors")
        if not results[name]["applicability"]["positive_true_planes"]:omitted[name].append("positive_train_validation_test_and_loo_stability")
    solve_diagnostics={name:oracle["records"][name]["predictor_diagnostics"] for name in SCENARIOS}
    diagnostics={"oracles":oracle,"leakage":leak,"applicability":applicability,"omitted_metrics":omitted,"shared_input_control":shared_input,
                 "input_design":input_design,"plane_registry":plane_registry,"solve_diagnostics":solve_diagnostics,
                 "scenario_controls":{n:results[n]["control"] for n in SCENARIOS},
                 "scientific_limits":["Both recovered invariant planes are ordered by skew rate and matched jointly to declared true planes only where applicable.",
                                      "A rotational projection is a state-space projection, not a posterior distribution and not proof of an oscillator, autonomous dynamics, or mechanism.",
                                      "Nonorthogonal mixing retains mapped-plane angles and Euclidean fitted rates but omits latent rate-error claims.",
                                      "The covariance-shuffle threshold miss is retained explicitly rather than described as a successful match."]}
    write_csv(root/"metrics.csv",rows);(root/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True,allow_nan=False)+"\n");make_png(root/"summary.png",rows,diagnostics,results);hashes={n:sha(root/n) for n in ARTIFACT_NAMES}
    h=embedding();non=nonorthogonal_embedding(h);z0=initial_states();dist=np.linalg.norm(z0[:,None,:]-z0[None,:,:],axis=2);nonzero=dist[np.triu_indices(NCOND,1)]
    initial_summary={"rank":int(np.linalg.matrix_rank(z0)),"minimum_pairwise_distance":float(np.min(nonzero)),"maximum_pairwise_distance":float(np.max(nonzero)),
                     "fast_plane_energy_fraction":float(np.sum(z0[:,:2]**2)/np.sum(z0**2)),"slow_plane_energy_fraction":float(np.sum(z0[:,2:4]**2)/np.sum(z0**2))}
    shuffle_selected={split:{key:results["covariance_condition_shuffle"]["control"][split][key] for key in ("selected_proposal","achieved_similarity","threshold","threshold_miss","selected_feature_permutations")} for split in TRIALS}
    manifest={"schema_version":1,"artifact":"rotational-projection-controls","date":DATE,"master_seed":MASTER_SEED,"seed_registry":seed_registry(),
              "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"pillow":getattr(sys.modules.get('PIL'),'__version__','unknown'),"platform":platform.platform()},
              "dimensions":{"latent":K,"observed":P,"time_points":NT,"conditions":NCOND},"trial_counts":TRIALS,
              "time":{"dt":DT,"interval":[float(TIME[0]),float(TIME[-1])]},"initial_states":z0.tolist(),"initial_state_summary":initial_summary,
              "generators":{k:v.tolist() for k,v in generator_matrices().items()},
              "embeddings":{"isometric":h.tolist(),"isometric_gram":(h.T@h).tolist(),"isometric_condition_number":float(np.linalg.cond(h)),"nonorthogonal":non.tolist(),"nonorthogonal_gram":(non.T@non).tolist(),"nonorthogonal_condition_number":float(np.linalg.cond(non))},
              "windows":{"full_indices":list(range(NT)),"short_indices":list(range(18))},"smoothing_kernel":[1/3,1/3,1/3],"fixed_plane":fixed_plane().tolist(),"random_planes":[p.tolist() for p in random_planes()],
              "jitter":{"policy":"independent trial integer shifts -3..3; analytic physical-time evaluation at TIME+shift*DT; no wrapping","records":"diagnostics.scenario_controls.jittered_alignment.trial_level_shifts"},
              "shuffle":{"statistic":"1 - ||C-Cp||_F^2 / ||C||_F^2 on flattened feature covariance","proposal_count_per_split":128,"operation":"independent whole-trajectory condition permutation for each feature; all 61 bins move together","selection":"maximum similarity only","threshold":.95,"selected_records":shuffle_selected},
              "input_control":{"integration":"forward Euler latent system","regression":"shared byte-identical dataset; left-endpoint PCA score/input secants; affine state-only [x,1] and input-aware [x,u,1]","across_condition_time_mean_subtraction":False,"expected_coefficient_formula":"R=H^T W; M^T=R^-1 A^T R; G^T=B^T R; intercept=center_score M^T"},
              "pipeline":{"condition_means":"split-specific means; training-only preprocessing fit","pca_rank":K,"derivatives":"midpoint/forward difference except input controls use left-endpoint Euler secants; never cross condition boundaries","unrestricted":"least squares; never antisymmetrized","skew":"explicit skew-basis least squares","symmetric":"explicit symmetric-basis least squares","planes":"two invariant planes from -M_skew^2 with q1^T M q2<0 and signed rate=-q1^T M q2","stability":"common observed feature space W_split Q_split; train-validation, train-test, and leave-one-condition-out"},
              "solve_diagnostics":solve_diagnostics,"applicability":applicability,"omitted_metrics":omitted,"tolerances":TOL,
              "artifact_files":["manifest.json",*ARTIFACT_NAMES],"files_sha256":hashes,"commands":{"generate":f"{sys.executable} toy-models/rotational-projection-controls.py --generate","verify":f"{sys.executable} toy-models/rotational-projection-controls.py --verify"}}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if not quiet:
        p=results["positive_full_arc"];print(json.dumps({"artifact_root":str(root),"metrics_rows":len(rows),"positive_plane_angles":[e["max_angle_degrees"] for e in p["matching"]["errors"]],"positive_signed_rates":[x["signed_rate"] for x in p["planes"]],"input_aware_test_r2":next(r["value"] for r in rows if r["condition"]=="input_driven_input_aware" and r["split"]=="test" and r["method"]=="input_aware_affine" and r["metric"]=="derivative_r2")},indent=2))
    return diagnostics

def compare(a,b):return[n for n in ("manifest.json",*ARTIFACT_NAMES) if (a/n).read_bytes()!=(b/n).read_bytes()]
def verify(root,recompute=True):
    errors=[]
    for name in ("manifest.json",*ARTIFACT_NAMES):
        if not(root/name).is_file():errors.append(f"missing {name}")
    if errors:raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    manifest=json.loads((root/"manifest.json").read_text());diagnostics=json.loads((root/"diagnostics.json").read_text())
    for name,expected in manifest["files_sha256"].items():
        if sha(root/name)!=expected:errors.append(f"hash mismatch {name}")
    for key in ("trial_counts","initial_states","initial_state_summary","embeddings","seed_registry","windows","smoothing_kernel","random_planes","jitter","shuffle","input_control","pipeline","solve_diagnostics","applicability","omitted_metrics"):
        if key not in manifest:errors.append(f"manifest missing {key}")
    if manifest.get("trial_counts")!=TRIALS:errors.append("trial counts mismatch")
    if manifest.get("shuffle",{}).get("statistic")!="1 - ||C-Cp||_F^2 / ||C||_F^2 on flattened feature covariance":errors.append("shuffle statistic mismatch")
    if manifest.get("initial_state_summary",{}).get("rank",0)<K:errors.append("initial state rank insufficient")

    oracle=diagnostics["oracles"]
    if not oracle["positive_block_sign_algebra"]["pass"]:errors.append("positive generator block/sign algebra failed")
    if oracle["isometry_max_abs"]>TOL["algebra"] or oracle["analytic_midpoint_fd_max_abs"]>TOL["finite_difference"]:errors.append("isometry/finite-difference oracle failed")
    if oracle["max_constraint"]>TOL["skew"] or oracle["max_basis_recompute"]>TOL["algebra"] or oracle["max_pca_svd_projector"]>TOL["algebra"]:errors.append("constraint/basis/PCA oracle failed")
    if oracle["max_plane_projector_reextract"]>TOL["algebra"] or oracle["minimum_orientation_bilinear"]>=0 or oracle["max_signed_rate_identity_error"]>TOL["algebra"] or oracle["max_MQ_equals_QJ"]>TOL["algebra"]:errors.append("plane invariance/orientation/rate oracle failed")
    if not all(oracle["jitter_checks"].values()) or not oracle["trial_id_splits_disjoint"]:errors.append("jitter/split-ID oracle failed")
    if not all(oracle["preprocessing_branch_isolation"].values()) or oracle["preprocessing_branch_isolation"]["short_window_length"]!=18:errors.append("preprocessing branch isolation failed")
    for name,record in oracle["records"].items():
        if not record["condition_groups_monotone_no_cross_transition"] or not record["trial_id_splits_disjoint"]:errors.append(f"condition/split boundary failed {name}")
        if record["derivative_pair_count"]!=record["expected_derivative_pair_count"]:errors.append(f"derivative pair count failed {name}")
        if not record["unrestricted_sse_le_skew"] or not record["unrestricted_sse_le_symmetric"]:errors.append(f"nested SSE ordering failed {name}")
        if record["pca_rank"]!=K or not (0<=record["pca_retained_fraction"]<=1):errors.append(f"PCA rank/occupancy failed {name}")
        for plane in record["planes"]:
            if not plane["orientation_negative"] or not plane["signed_rate_positive"] or plane["orthonormality_max_abs"]>TOL["algebra"] or plane["MQ_equals_QJ_max_abs"]>TOL["algebra"] or plane["angle_invariance_rotation_swap_sign_max_degrees"]>5e-6:errors.append(f"plane algebra/invariance failed {name}")
        if record["matching_recompute"] is not None and record["matching_recompute"]["difference"]>TOL["algebra"]:errors.append(f"matching recompute failed {name}")
        for predictor,pd in record["predictor_diagnostics"].items():
            if pd["design_rank"]<=0 or not math.isfinite(pd["design_condition_number"]) or pd["training_sse"]<0 or not pd["all_finite"] or pd["residual_orthogonality_max_abs"]>1e-8:errors.append(f"predictor solve diagnostics failed {name}:{predictor}")

    shared=diagnostics["shared_input_control"]
    for key in ("train_observed_max_abs","validation_observed_max_abs","test_observed_max_abs","clean_max_abs","means_max_abs","preprocessing_center_max_abs","preprocessing_pca_projector_max_abs"):
        if shared[key]>TOL["poison"]:errors.append(f"input datasets/preprocessing differ: {key}")
    if not shared["trial_ids_equal"]:errors.append("input trial IDs differ")
    input_design=diagnostics["input_design"]
    if input_design["input_driven_state_only"]["input_aware_fit_present"] or not input_design["input_driven_input_aware"]["input_aware_fit_present"]:errors.append("input branch fit presence failed")
    aware=input_design["input_driven_input_aware"]["expected_coefficient_oracle"]
    if aware is None or aware["euler_local_consistency_max_abs"]>TOL["algebra"] or not all(math.isfinite(aware[k]) for k in ("M_error","G_error","intercept_error")):errors.append("input expected-coordinate oracle failed")

    poison=diagnostics["leakage"]
    if poison["maximum_fit_discrepancy"]>TOL["poison"] or set(poison["records"])!=set(SCENARIOS):errors.append("full-scenario poison failed")
    expected_fields={"indices","time_mean","center","pca_projector","unrestricted","intercept","skew","symmetric","fixed_original_test_prediction","fixed_plane","plane_projectors","plane_rates","random_planes","input_state","input_coeff","input_intercept","jitter_shuffle_registry"}
    for name,record in poison["records"].items():
        if record["poisoned_test_observation_change"]<=0 or record["all_truth_poison_norm"]<=0 or record["truth_poison_received_by_harness_norm"]<=0:errors.append(f"poison arrays did not change/pass harness {name}")
        if set(record["fit_difference"]["by_field"])!=expected_fields or record["fit_difference"]["maximum"]>TOL["poison"]:errors.append(f"poison fitted state mismatch {name}")

    jitter=diagnostics["scenario_controls"]["jittered_alignment"]
    if not all(jitter["multiple_distinct_shifts"].values()) or not all(jitter["no_circular_wrap"].values()):errors.append("jitter records invalid")
    shuffle=diagnostics["scenario_controls"]["covariance_condition_shuffle"]
    for split in TRIALS:
        record=shuffle[split]
        if len(record["proposal_scores"])!=128 or len(record["selected_feature_permutations"])!=P:errors.append(f"shuffle registry incomplete {split}")
        best=max(record["proposal_scores"],key=lambda x:(x["covariance_similarity"],-x["proposal"]))
        if best["proposal"]!=record["selected_proposal"] or abs(best["covariance_similarity"]-record["achieved_similarity"])>TOL["algebra"]:errors.append(f"shuffle reselection failed {split}")
        if not record["feature_trajectory_multisets_preserved"] or record["marginal_feature_distribution_max_abs"]>TOL["algebra"]:errors.append(f"shuffle invariants failed {split}")
        if record["threshold_miss"]!=(record["achieved_similarity"]<.95):errors.append(f"shuffle threshold record failed {split}")

    applicability=diagnostics["applicability"]
    for name in ("latency_sequence","symmetric_decay","input_driven_state_only","input_driven_input_aware"):
        if not applicability[name]["true_plane_metrics_omitted"]:errors.append(f"true-plane omission failed {name}")
    if not applicability["nonorthogonal_mixing"]["nonorthogonal_mapped_plane_stress"] or not applicability["nonorthogonal_mixing"]["latent_rate_errors_omitted"]:errors.append("nonorthogonal applicability failed")

    with (root/"metrics.csv").open(newline="") as handle:
        reader=csv.DictReader(handle);rows=list(reader)
        if reader.fieldnames!=list(FIELDS):errors.append("CSV fields mismatch")
    seen=set()
    for index,row in enumerate(rows,2):
        if set(row)!=set(FIELDS) or any(not row[k].strip() for k in FIELDS):errors.append(f"invalid row {index}")
        try:
            if not math.isfinite(float(row["value"])):errors.append(f"nonfinite row {index}")
        except ValueError:errors.append(f"nonnumeric row {index}")
        key=tuple(row[k] for k in FIELDS[:-1])
        if key in seen:errors.append(f"duplicate key {key}")
        seen.add(key)
    metrics={row["metric"] for row in rows}
    required={"derivative_mse","derivative_r2","plane_occupancy","tangential_energy","radial_energy","tangential_radial_energy_ratio","perpendicular_fraction_within_15deg","condition_span_radians","condition_span_mean_radians","condition_span_median_radians","condition_span_min_radians","condition_span_max_radians","signed_direction_consistency","condition_signed_angular_rate_mean","condition_signed_angular_rate_sd","signed_rate","absolute_rate"}
    if not required.issubset(metrics):errors.append("required metrics missing")
    for name in SCENARIOS:
        expected_primary="unrestricted_affine" if name.startswith("input_driven") else "unrestricted"
        for split in TRIALS:
            for method in (expected_primary,"skew_constrained","symmetric_constrained"):
                if not all(any(row["condition"]==name and row["split"]==split and row["method"]==method and row["metric"]==metric for row in rows) for metric in ("derivative_mse","derivative_r2")):errors.append(f"derivative split coverage missing {name}:{split}:{method}")
        for method in ("skew_plane_1","skew_plane_2","fixed_plane",*[f"random_plane_{i}" for i in range(5)]):
            subset=[row for row in rows if row["condition"]==name and row["method"]==method]
            needed={"plane_occupancy","tangential_energy","radial_energy","tangential_radial_energy_ratio","perpendicular_fraction_within_15deg","condition_span_mean_radians","condition_span_median_radians","condition_span_min_radians","condition_span_max_radians","signed_direction_consistency","condition_signed_angular_rate_mean","condition_signed_angular_rate_sd"}
            if not needed.issubset({row["metric"] for row in subset}) or sum(row["metric"]=="condition_span_radians" for row in subset)!=NCOND:errors.append(f"plane metric coverage missing {name}:{method}")
    positive_names=("positive_full_arc","positive_short_window","jittered_alignment","smoothed","no_condition_mean_subtraction","covariance_condition_shuffle")
    for name in positive_names:
        if len([r for r in rows if r["condition"]==name and r["metric"]=="relative_rate_error"])!=2:errors.append(f"relative rate rows missing {name}")
        for target_split in ("validation","test"):
            if len([r for r in rows if r["condition"]==name and r["target"]==f"train_{target_split}_observed_space_stability" and r["metric"]=="plane_angle_degrees"])!=2:errors.append(f"split stability missing {name}:{target_split}")
        if len([r for r in rows if r["condition"]==name and r["preprocessing"]=="leave_one_condition_out" and r["metric"]=="plane_angle_degrees"])!=NCOND*2:errors.append(f"LOO coverage missing {name}")
        if len([r for r in rows if r["condition"]==name and r["preprocessing"]=="frozen_training_plane_single_trial" and r["metric"]=="tangential_energy"])!=2:errors.append(f"single-trial plane metrics missing {name}")
    if any(row["condition"]=="nonorthogonal_mixing" and row["metric"] in ("signed_rate_error","absolute_rate_error","relative_rate_error") for row in rows):errors.append("nonorthogonal latent rate errors present")
    for name in ("latency_sequence","symmetric_decay","input_driven_state_only","input_driven_input_aware"):
        if any(row["condition"]==name and row["metric"] in ("plane_max_angle_degrees","signed_rate_error","absolute_rate_error","relative_rate_error") for row in rows):errors.append(f"inapplicable true-plane metrics present {name}")
    for metric in ("M_error","G_error","intercept_error","euler_local_consistency_max_abs"):
        if not any(row["condition"]=="input_driven_input_aware" and row["metric"]==metric for row in rows):errors.append(f"input coefficient metric missing {metric}")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="ldg-rotation-") as tmp:
            regenerated=Path(tmp)/"artifact";build(regenerated,True);errors += [f"byte mismatch {name}" for name in compare(root,regenerated)]
    if errors:raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    print(json.dumps({"verified":True,"artifact_root":str(root),"deterministic_recomputation":recompute},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument("--generate",action="store_true");p.add_argument("--verify",action="store_true");p.add_argument("--no-recompute",action="store_true");p.add_argument("--artifact-root",type=Path,default=DEFAULT_ROOT);a=p.parse_args();g,v=a.generate,a.verify
    if not g and not v:g=v=True
    if g:build(a.artifact_root)
    if v:verify(a.artifact_root,not a.no_recompute)
if __name__=="__main__":main()
