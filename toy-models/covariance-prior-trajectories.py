#!/usr/bin/env python3
"""Deterministic covariance-prior trajectory generation and recovery ladder."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASTER_SEED=20260819; DATE="2026-08-19"; Q=2; P=6; T=24
TIME=np.linspace(0.,1.,T); DT=float(TIME[1]-TIME[0]); TRUE_TAU=(.08,.30); NUGGET=.02
SPLITS={"train":8,"validation":4,"test":5}; TAU_GRID=tuple((a,b) for a in (.06,.08,.14) for b in (.18,.30,.45))
COMMON_TAU=(.14,.14); ARTIFACT_NAMES=("metrics.csv","diagnostics.json","summary.png")
FIELDS=("split","condition","method","target","region","metric","value")
TOL={"algebra":1e-9,"psd":1e-9,"poison":1e-12,"determinism":0.0}
ROOT=Path(__file__).resolve().parent; DEFAULT_ROOT=ROOT/"artifacts"/"covariance-prior-trajectories"
C=np.array([[1.0,.25],[.7,-.45],[.3,.9],[-.65,.4],[.8,.55],[-.25,-.75]])
D=np.array([.3,-.2,.15,.05,-.1,.25]); RDIAG=np.array([.10,.16,.22,.30,.38,.48]); R=np.diag(RDIAG)

def rng_for(*words):return np.random.Generator(np.random.PCG64(np.random.SeedSequence((MASTER_SEED,)+tuple(words))))
def se_kernel(tau,t=TIME):
    delta=t[:,None]-t[None,:];return (1-NUGGET)*np.exp(-.5*(delta/tau)**2)+NUGGET*np.eye(len(t))
def latent_cov(taus,kernel="se"):
    cov=np.zeros((Q*T,Q*T))
    for j,tau in enumerate(taus):
        if kernel=="se":k=se_kernel(tau)
        elif kernel=="identity":k=np.eye(T)
        elif kernel=="ar":
            rho=(1-NUGGET)*math.exp(-.5*(DT/tau)**2);k=rho**np.abs(np.subtract.outer(np.arange(T),np.arange(T)))
        else:raise KeyError(kernel)
        idx=np.arange(T)*Q+j;cov[np.ix_(idx,idx)]=k
    return cov
def observation_matrix(c=C):return np.kron(np.eye(T),c)
def observation_mean(d=D):return np.tile(d,T)
def observation_cov(taus,c=C,r=R,kernel="se"):
    a=observation_matrix(c);return a@latent_cov(taus,kernel)@a.T+np.kron(np.eye(T),r)
def chol_psd(matrix):return np.linalg.cholesky(.5*(matrix+matrix.T))
def generate_trial(taus,split_code,index,c=C,d=D,r=R,condition="primary"):
    z=np.empty((T,Q))
    for j,tau in enumerate(taus):z[:,j]=chol_psd(se_kernel(tau))@rng_for(10,split_code,index,j).normal(size=T)
    noise=rng_for(11,split_code,index).multivariate_normal(np.zeros(P),r,size=T)
    y=z@c.T+d+noise
    return {"latent":z,"observed":y,"signal":z@c.T+d,"condition":condition,"trial_id":split_code*1000+index}
def generate_dataset(taus=TRUE_TAU,c=C,d=D,r=R):
    data={}
    for code,split in enumerate(("train","validation","test"),1):data[split]=[generate_trial(taus,code,i,c,d,r) for i in range(SPLITS[split])]
    return data
def mask_design():
    masks=[];records=[]
    for i in range(SPLITS["test"]):
        mask=np.ones((T,P),bool)
        if i<2: mask[:,i]=False;records.append({"trial":i,"type":"complete_channel","channel":i})
        else: mask[8:15,:]=False;records.append({"trial":i,"type":"contiguous_time_block","start":8,"end_exclusive":15})
        masks.append(mask)
    return masks,records
def gaussian_nll(x,mean,cov):
    l=chol_psd(cov);res=x-mean;sol=np.linalg.solve(l,res);return float(.5*(len(x)*math.log(2*math.pi)+2*np.sum(np.log(np.diag(l)))+sol@sol))
def condition_gaussian(mean,cov,observed,mask):
    mask=np.asarray(mask,bool).reshape(-1);obs=np.flatnonzero(mask);target=np.flatnonzero(~mask);result={"target_indices":target}
    if not len(target):return {**result,"mean":np.empty(0),"cov":np.empty((0,0)),"nll":0.}
    co=cov[np.ix_(obs,obs)];ct=cov[np.ix_(target,target)];cross=cov[np.ix_(target,obs)]
    l=chol_psd(co);alpha=np.linalg.solve(l.T,np.linalg.solve(l,observed.reshape(-1)[obs]-mean[obs]));cm=mean[target]+cross@alpha
    solved=np.linalg.solve(l.T,np.linalg.solve(l,cross.T));cc=.5*((ct-cross@solved)+(ct-cross@solved).T)
    return {**result,"mean":cm,"cov":cc,"nll":gaussian_nll(observed.reshape(-1)[target],cm,cc)}
def latent_posterior(y,mask,taus,kernel="se",c=C,d=D,r=R):
    kz=latent_cov(taus,kernel);a=observation_matrix(c);mean_y=observation_mean(d);flat=y.reshape(-1);m=np.asarray(mask,bool).reshape(-1);obs=np.flatnonzero(m)
    ao=a[obs];ro=np.diag(np.tile(np.diag(r),T)[obs]);s=ao@kz@ao.T+ro;l=chol_psd(s);alpha=np.linalg.solve(l.T,np.linalg.solve(l,flat[obs]-mean_y[obs]));pm=kz@ao.T@alpha
    solved=np.linalg.solve(l.T,np.linalg.solve(l,ao@kz));pc=.5*((kz-kz@ao.T@solved)+(kz-kz@ao.T@solved).T)
    return {"mean":pm.reshape(T,Q),"cov":pc,"var":np.diag(pc).reshape(T,Q)}
def precision_posterior(y,mask,taus,c=C,d=D,r=R):
    kz=latent_cov(taus);a=observation_matrix(c);m=np.asarray(mask,bool).reshape(-1);obs=np.flatnonzero(m);ao=a[obs];rv=np.tile(np.diag(r),T)[obs];precision=np.linalg.inv(kz)+ao.T@(ao/rv[:,None]);cov=np.linalg.inv(precision);rhs=ao.T@((y.reshape(-1)[obs]-observation_mean(d)[obs])/rv);mean=cov@rhs
    return mean,cov

def ar_parameters(taus):
    rho=np.array([(1-NUGGET)*math.exp(-.5*(DT/tau)**2) for tau in taus]);return rho,1-rho**2

def kalman_rts(y,mask,taus,c=C,d=D,r=R):
    rho,q=ar_parameters(taus);f=np.diag(rho);process=np.diag(q);pm=[];pp=[];fm=[];fp=[]
    mean=np.zeros(Q);cov=np.eye(Q)
    for t in range(T):
        if t>0:mean=f@mean;cov=f@cov@f.T+process
        pm.append(mean.copy());pp.append(cov.copy());obs=np.flatnonzero(mask[t])
        if len(obs):
            h=c[obs];rr=r[np.ix_(obs,obs)];innovation=y[t,obs]-d[obs]-h@mean;s=h@cov@h.T+rr;k=np.linalg.solve(s,(cov@h.T).T).T
            mean=mean+k@innovation;cov=cov-k@h@cov;cov=.5*(cov+cov.T)
        fm.append(mean.copy());fp.append(cov.copy())
    pm,pp,fm,fp=map(np.asarray,(pm,pp,fm,fp));sm=fm.copy();sp=fp.copy();cross=np.zeros((T-1,Q,Q));gains=np.zeros((T-1,Q,Q))
    for t in range(T-2,-1,-1):
        gain=np.linalg.solve(pp[t+1],(fp[t]@f.T).T).T;gains[t]=gain;sm[t]=fm[t]+gain@(sm[t+1]-pm[t+1]);sp[t]=fp[t]+gain@(sp[t+1]-pp[t+1])@gain.T;sp[t]=.5*(sp[t]+sp[t].T);cross[t]=gain@sp[t+1]
    return {"mean":sm,"covariances":sp,"var":np.stack([np.diag(x) for x in sp]),"adjacent_cross":cross,"smoother_gains":gains,"filter_mean":fm,"filter_cov":fp}

def rts_full_covariance(result):
    full=np.zeros((Q*T,Q*T))
    for t in range(T):full[t*Q:(t+1)*Q,t*Q:(t+1)*Q]=result["covariances"][t]
    for i in range(T-2,-1,-1):
        for j in range(i+1,T):
            block=result["smoother_gains"][i]@full[(i+1)*Q:(i+2)*Q,j*Q:(j+1)*Q];full[i*Q:(i+1)*Q,j*Q:(j+1)*Q]=block;full[j*Q:(j+1)*Q,i*Q:(i+1)*Q]=block.T
    return full

def ar_batch_blocks(y,mask,taus):
    post=latent_posterior(y,mask,taus,"ar");cov=post["cov"]
    blocks=np.stack([cov[t*Q:(t+1)*Q,t*Q:(t+1)*Q] for t in range(T)])
    cross=np.stack([cov[t*Q:(t+1)*Q,(t+1)*Q:(t+2)*Q] for t in range(T-1)])
    return post["mean"],blocks,cross,cov
def align_latent(estimate,truth):
    candidates=[]
    for perm in ((0,1),(1,0)):
        for signs in ((1,1),(1,-1),(-1,1),(-1,-1)):
            e=estimate[:,perm]*np.array(signs);candidates.append((float(np.mean((e-truth)**2)),e,perm,signs))
    return min(candidates,key=lambda x:(x[0],x[2],x[3]))
def select_grid(data):
    records=[]
    for taus in TAU_GRID:
        cov=observation_cov(taus);mean=observation_mean();totals={}
        for split in ("train","validation"):totals[split]=float(sum(gaussian_nll(trial["observed"].reshape(-1),mean,cov) for trial in data[split]))
        combined=totals["train"]+totals["validation"]
        records.append({"tau_1":taus[0],"tau_2":taus[1],"train_total_nll":totals["train"],"validation_total_nll":totals["validation"],"combined_total_nll":combined,
                        "train_nll_per_entry":totals["train"]/(len(data["train"])*T*P),"validation_nll_per_entry":totals["validation"]/(len(data["validation"])*T*P),
                        "combined_nll_per_entry":combined/((len(data["train"])+len(data["validation"]))*T*P),"selection_score":combined})
    ordered=sorted(records,key=lambda x:(x["combined_total_nll"],x["tau_1"],x["tau_2"]))
    for rank,record in enumerate(ordered,1):record["rank"]=rank
    return (ordered[0]["tau_1"],ordered[0]["tau_2"]),records
def add(rows,split,condition,method,target,region,metric,value):rows.append(dict(split=split,condition=condition,method=method,target=target,region=region,metric=metric,value=float(value)))
def latent_time_regions(mask):
    hidden=np.all(~np.asarray(mask,bool),axis=1);return {"masked_time_region":hidden,"unmasked_time_region":~hidden}
def score_probabilistic(rows,data,masks,condition,method,taus,kernel="se"):
    covy=observation_cov(taus,kernel=kernel);mean=observation_mean();rmses=[];cover=[];vars_=[];rough=[];full_nll=[]
    masked={"masked_channel":{"nll":[],"sse":[],"count":0,"covered":[],"coverage_count":0,"variance":[]},"masked_block":{"nll":[],"sse":[],"count":0,"covered":[],"coverage_count":0,"variance":[]}};latent_regions={k:{"sse":0.,"count":0,"covered":0} for k in ("masked_time_region","unmasked_time_region")}
    for i,trial in enumerate(data["test"]):
        full_nll.append(gaussian_nll(trial["observed"].reshape(-1),mean,covy)/(T*P))
        post=latent_posterior(trial["observed"],masks[i],taus,kernel);err=post["mean"]-trial["latent"];rmses.append(np.mean(err**2));cover.append(np.mean(np.abs(err)<=1.9599639845*np.sqrt(post["var"])));vars_.append(np.mean(post["var"]));rough.append(np.mean(np.diff(post["mean"],axis=0)**2))
        for region,index in latent_time_regions(masks[i]).items():
            if np.any(index):latent_regions[region]["sse"]+=float(np.sum(err[index]**2));latent_regions[region]["count"]+=int(np.sum(index)*Q);latent_regions[region]["covered"]+=int(np.sum(np.abs(err[index])<=1.9599639845*np.sqrt(post["var"][index])))
        conditional=condition_gaussian(mean,covy,trial["observed"],masks[i]);target=trial["observed"].reshape(-1)[conditional["target_indices"]];err_y=conditional["mean"]-target;ctype="masked_channel" if i<2 else "masked_block";record=masked[ctype]
        record["nll"].append(conditional["nll"]);record["sse"].append(float(np.sum(err_y**2)));record["count"]+=len(target);record["covered"].append(int(np.sum(np.abs(err_y)<=1.9599639845*np.sqrt(np.diag(conditional["cov"])))));record["coverage_count"]+=len(target);record["variance"].extend(np.diag(conditional["cov"]).tolist())
    add(rows,"test",condition,method,"observation_distribution","complete_trial","marginal_nll_per_entry",np.mean(full_nll))
    for region,record in masked.items():
        if record["count"]:add(rows,"test",condition,method,"masked_observation_prediction",region,"predictive_nll_per_entry",sum(record["nll"])/record["count"]);add(rows,"test",condition,method,"masked_observation_prediction",region,"predictive_rmse",math.sqrt(sum(record["sse"])/record["count"]));add(rows,"test",condition,method,"masked_observation_prediction",region,"predictive_95_coverage",sum(record["covered"])/record["coverage_count"]);add(rows,"test",condition,method,"masked_observation_prediction",region,"mean_predictive_variance",np.mean(record["variance"]));add(rows,"test",condition,method,"masked_observation_prediction",region,"hidden_target_count",record["count"])
    add(rows,"test",condition,method,"latent_posterior","all","latent_rmse",np.sqrt(np.mean(rmses)));add(rows,"test",condition,method,"latent_posterior","all","latent_95_coverage",np.mean(cover));add(rows,"test",condition,method,"latent_posterior","all","mean_posterior_variance",np.mean(vars_));add(rows,"test",condition,method,"latent_posterior","all","posterior_roughness",np.mean(rough))
    for region,record in latent_regions.items():
        if record["count"]:add(rows,"test",condition,method,"latent_posterior",region,"latent_rmse",math.sqrt(record["sse"]/record["count"]));add(rows,"test",condition,method,"latent_posterior",region,"latent_95_coverage",record["covered"]/record["count"])

def score_recursive_ar(rows,data,masks):
    rms=[];coverage=[];variance=[];rough=[];full_nll=[];covy=observation_cov(TRUE_TAU,kernel="ar");mean=observation_mean();masked={"masked_channel":[],"masked_block":[]};latent_regions={k:{"sse":0.,"count":0,"covered":0} for k in ("masked_time_region","unmasked_time_region")}
    for i,(trial,mask) in enumerate(zip(data["test"],masks)):
        result=kalman_rts(trial["observed"],mask,TRUE_TAU);err=result["mean"]-trial["latent"];rms.append(np.mean(err**2));coverage.append(np.mean(np.abs(err)<=1.9599639845*np.sqrt(result["var"])));variance.append(np.mean(result["var"]));rough.append(np.mean(np.diff(result["mean"],axis=0)**2));full_nll.append(gaussian_nll(trial["observed"].reshape(-1),mean,covy)/(T*P))
        for latent_region,index in latent_time_regions(mask).items():
            if np.any(index):latent_regions[latent_region]["sse"]+=float(np.sum(err[index]**2));latent_regions[latent_region]["count"]+=int(np.sum(index)*Q);latent_regions[latent_region]["covered"]+=int(np.sum(np.abs(err[index])<=1.9599639845*np.sqrt(result["var"][index])))
        cond=condition_gaussian(mean,covy,trial["observed"],mask);target=trial["observed"].reshape(-1)[cond["target_indices"]];err_y=cond["mean"]-target;region="masked_channel" if i<2 else "masked_block";masked[region].append((cond["nll"],np.sum(err_y**2),len(target),np.sum(np.abs(err_y)<=1.9599639845*np.sqrt(np.diag(cond["cov"]))),np.sum(np.diag(cond["cov"]))))
    add(rows,"test","primary","ar_lgssm_rts","observation_distribution","complete_trial","marginal_nll_per_entry",np.mean(full_nll))
    for region,records in masked.items():
        count=sum(x[2] for x in records);add(rows,"test","primary","ar_lgssm_rts","masked_observation_prediction",region,"predictive_nll_per_entry",sum(x[0] for x in records)/count);add(rows,"test","primary","ar_lgssm_rts","masked_observation_prediction",region,"predictive_rmse",math.sqrt(sum(x[1] for x in records)/count));add(rows,"test","primary","ar_lgssm_rts","masked_observation_prediction",region,"predictive_95_coverage",sum(x[3] for x in records)/count);add(rows,"test","primary","ar_lgssm_rts","masked_observation_prediction",region,"mean_predictive_variance",sum(x[4] for x in records)/count);add(rows,"test","primary","ar_lgssm_rts","masked_observation_prediction",region,"hidden_target_count",count)
    add(rows,"test","primary","ar_lgssm_rts","latent_posterior","all","latent_rmse",math.sqrt(np.mean(rms)));add(rows,"test","primary","ar_lgssm_rts","latent_posterior","all","latent_95_coverage",np.mean(coverage));add(rows,"test","primary","ar_lgssm_rts","latent_posterior","all","mean_posterior_variance",np.mean(variance));add(rows,"test","primary","ar_lgssm_rts","latent_posterior","all","posterior_roughness",np.mean(rough))
    for region,record in latent_regions.items():
        if record["count"]:add(rows,"test","primary","ar_lgssm_rts","latent_posterior",region,"latent_rmse",math.sqrt(record["sse"]/record["count"]));add(rows,"test","primary","ar_lgssm_rts","latent_posterior",region,"latent_95_coverage",record["covered"]/record["count"])

def score_fa_model(rows,data,masks,model,method,condition="primary",target_data=None,alignment=None):
    target_data=data if target_data is None else target_data
    rmse=[];coverage=[];rough=[];masked_sse={"masked_channel":[],"masked_block":[]};masked_nll={"masked_channel":[],"masked_block":[]};masked_cov={"masked_channel":[],"masked_block":[]};full=[]
    covariance=model["covariance"]
    for i,trial in enumerate(data["test"]):
        target_trial=target_data["test"][i];scores,post_cov=fa_bin_posterior(trial["observed"],model);rotation=np.eye(Q) if alignment is None else alignment["rotation"];aligned=scores@rotation;transformed_cov=rotation.T@post_cov@rotation;rmse.append(np.mean((aligned-target_trial["latent"])**2));coverage.append(np.mean(np.abs(aligned-target_trial["latent"])<=1.9599639845*np.sqrt(np.diag(transformed_cov))[None,:]));rough.append(np.mean(np.diff(aligned,axis=0)**2));full.append(iid_gaussian_nll(target_trial["observed"],model["mean"],covariance))
        region="masked_channel" if i<2 else "masked_block"
        for t in range(T):
            obs=np.flatnonzero(masks[i][t]);hidden=np.flatnonzero(~masks[i][t])
            if not len(hidden):continue
            co=covariance[np.ix_(obs,obs)];cross=covariance[np.ix_(hidden,obs)];ct=covariance[np.ix_(hidden,hidden)];l=chol_psd(co);pred=model["mean"][hidden]+cross@np.linalg.solve(l.T,np.linalg.solve(l,trial["observed"][t,obs]-model["mean"][obs]));pc=ct-cross@np.linalg.solve(l.T,np.linalg.solve(l,cross.T));err=pred-target_trial["observed"][t,hidden];masked_sse[region].append(np.sum(err**2));masked_nll[region].append(gaussian_nll(target_trial["observed"][t,hidden],pred,pc));masked_cov[region].append((int(np.sum(np.abs(err)<=1.9599639845*np.sqrt(np.diag(pc)))),len(hidden),float(np.sum(np.diag(pc)))))
    add(rows,"test",condition,method,"observation_distribution","complete_trial","marginal_nll_per_entry",np.mean(full));add(rows,"test",condition,method,"aligned_latent_scores","all","latent_rmse",math.sqrt(np.mean(rmse)));add(rows,"test",condition,method,"aligned_latent_scores","all","latent_95_coverage",np.mean(coverage));add(rows,"test",condition,method,"aligned_latent_scores","all","posterior_roughness",np.mean(rough));add(rows,"test",condition,method,"loading_subspace","all","loading_subspace_angle_degrees",subspace_angle(model["loading"],C))
    for region in masked_sse:
        count=sum(x[1] for x in masked_cov[region])
        if count:add(rows,"test",condition,method,"masked_observation_prediction",region,"predictive_rmse",math.sqrt(sum(masked_sse[region])/count));add(rows,"test",condition,method,"masked_observation_prediction",region,"predictive_nll_per_entry",sum(masked_nll[region])/count);add(rows,"test",condition,method,"masked_observation_prediction",region,"predictive_95_coverage",sum(x[0] for x in masked_cov[region])/count);add(rows,"test",condition,method,"masked_observation_prediction",region,"mean_predictive_variance",sum(x[2] for x in masked_cov[region])/count);add(rows,"test",condition,method,"masked_observation_prediction",region,"hidden_target_count",count)

def score_mean_diagonal(rows,data,masks,model):
    full=[];regions={"masked_channel":[],"masked_block":[]}
    for i,trial in enumerate(data["test"]):
        full.append(float(np.mean(.5*(np.log(2*math.pi*model["variance"])+(trial["observed"]-model["mean"])**2/model["variance"]))))
        hidden=~masks[i];err=trial["observed"][hidden]-np.broadcast_to(model["mean"],(T,P))[hidden];var=np.broadcast_to(model["variance"],(T,P))[hidden];region="masked_channel" if i<2 else "masked_block";regions[region].append((np.sum(.5*(np.log(2*math.pi*var)+err**2/var)),np.sum(err**2),len(err),np.sum(np.abs(err)<=1.9599639845*np.sqrt(var)),np.sum(var)))
    add(rows,"test","primary","mean_diagonal_gaussian","observation_distribution","complete_trial","marginal_nll_per_entry",np.mean(full))
    for region,recs in regions.items():
        n=sum(x[2] for x in recs);add(rows,"test","primary","mean_diagonal_gaussian","masked_observation_prediction",region,"predictive_nll_per_entry",sum(x[0] for x in recs)/n);add(rows,"test","primary","mean_diagonal_gaussian","masked_observation_prediction",region,"predictive_rmse",math.sqrt(sum(x[1] for x in recs)/n));add(rows,"test","primary","mean_diagonal_gaussian","masked_observation_prediction",region,"predictive_95_coverage",sum(x[3] for x in recs)/n);add(rows,"test","primary","mean_diagonal_gaussian","masked_observation_prediction",region,"mean_predictive_variance",sum(x[4] for x in recs)/n);add(rows,"test","primary","mean_diagonal_gaussian","masked_observation_prediction",region,"hidden_target_count",n)

def fit_pca(train):
    x=np.vstack([t["observed"] for t in train]);mean=np.mean(x,0);_,_,vt=np.linalg.svd(x-mean,full_matrices=False);return mean,vt[:Q].T

def iid_gaussian_nll(x,mean,cov):
    return gaussian_nll((x-mean).reshape(-1),np.zeros(x.size),np.kron(np.eye(len(x)),cov))/x.size

def fa_em_arrays(train_x,validation_x,start_id,floor=1e-6,max_iter=300,tol=1e-9):
    mean=np.mean(train_x,axis=0);x=train_x-mean;s=x.T@x/len(x);vals,vecs=np.linalg.eigh(s);order=np.argsort(vals)[::-1]
    noise=max(float(np.mean(vals[order][Q:])),floor);loading=vecs[:,order[:Q]]*np.sqrt(np.maximum(vals[order[:Q]]-noise,floor))
    if start_id:
        loading=loading+rng_for(30,start_id).normal(scale=.08*np.sqrt(max(vals[order[0]],floor)),size=(P,Q))
    uniqueness=np.maximum(np.diag(s-loading@loading.T),floor);history=[];previous=None;converged=False
    for iteration in range(1,max_iter+1):
        cov=loading@loading.T+np.diag(uniqueness);inv=np.linalg.inv(cov);beta=loading.T@inv
        ezz=np.eye(Q)-beta@loading+beta@s@beta.T;cross=s@beta.T
        new_loading=cross@np.linalg.inv(ezz);new_unique=np.maximum(np.diag(s-new_loading@cross.T),floor)
        new_cov=new_loading@new_loading.T+np.diag(new_unique);current=iid_gaussian_nll(train_x,mean,new_cov);history.append(current)
        loading,uniqueness=new_loading,new_unique
        if previous is not None and abs(previous-current)<=tol*max(1.,abs(previous)):converged=True;break
        previous=current
    cov=loading@loading.T+np.diag(uniqueness);validation_nll=iid_gaussian_nll(validation_x,mean,cov)
    return {"start_id":start_id,"mean":mean,"loading":loading,"uniqueness":uniqueness,"covariance":cov,"history":history,
            "train_nll_per_entry":iid_gaussian_nll(train_x,mean,cov),"validation_nll_per_entry":validation_nll,"iterations":iteration,
            "converged":converged,"floor":floor,"maximum_nll_increase":float(max([0.]+[b-a for a,b in zip(history[:-1],history[1:])]))}

def select_fa(train,validation):
    tx=np.vstack([t["observed"] for t in train]);vx=np.vstack([t["observed"] for t in validation]);starts=[fa_em_arrays(tx,vx,i) for i in range(6)]
    selected=min(starts,key=lambda r:(r["train_nll_per_entry"],r["validation_nll_per_entry"],r["start_id"]))
    return selected,starts

def fa_bin_posterior(y,model):
    load=model["loading"];psi=np.diag(model["uniqueness"]);cov=load@load.T+psi;beta=load.T@np.linalg.inv(cov);mean=(y-model["mean"])@beta.T;post_cov=np.eye(Q)-beta@load
    return mean,post_cov

def mean_diagonal_model(train):
    x=np.vstack([t["observed"] for t in train]);return {"mean":np.mean(x,0),"variance":np.var(x,axis=0)+1e-8}

def procrustes_align(estimate,truth):
    u,_,vt=np.linalg.svd(estimate.T@truth);rotation=u@vt;return estimate@rotation,rotation

def fit_training_alignment(method,data,model):
    estimates=[];truth=[]
    for trial in data["train"]:
        if method in ("static_pca","two_stage_channel_gp_then_pca"):scores=(trial["observed"]-model["mean"])@model["basis"]
        else:scores,_=fa_bin_posterior(trial["observed"],model)
        estimates.append(scores);truth.append(trial["latent"])
    estimate=np.vstack(estimates);target=np.vstack(truth);aligned,rotation=procrustes_align(estimate,target)
    return {"rotation":rotation,"training_objective_mse":float(np.mean((aligned-target)**2)),"fit_split":"train","test_truth_used":False}

def learned_path_metrics(rows,method,data,model,alignment,posterior_cov=None,condition="primary"):
    rms=[];covered=[]
    rotation=alignment["rotation"]
    for trial in data["test"]:
        if "basis" in model:scores=(trial["observed"]-model["mean"])@model["basis"]
        else:scores,bin_cov=fa_bin_posterior(trial["observed"],model);posterior_cov=bin_cov
        aligned=scores@rotation;err=aligned-trial["latent"];rms.append(np.mean(err**2))
        if posterior_cov is not None:
            transformed=rotation.T@posterior_cov@rotation;covered.append(np.mean(np.abs(err)<=1.9599639845*np.sqrt(np.diag(transformed))[None,:]))
    add(rows,"test",condition,method,"train_frozen_procrustes_latent_path","all","latent_rmse",math.sqrt(np.mean(rms)))
    if covered:add(rows,"test",condition,method,"train_frozen_procrustes_latent_path","all","latent_95_coverage",np.mean(covered))
    return {"test_rmse":float(math.sqrt(np.mean(rms))),"test_coverage":None if not covered else float(np.mean(covered))}

def subspace_angle(a,b):
    qa,_=np.linalg.qr(a);qb,_=np.linalg.qr(b);s=np.linalg.svd(qa.T@qb,compute_uv=False);return float(np.degrees(np.max(np.arccos(np.clip(s,-1,1)))))
def subspace_metrics(a,b):
    qa,_=np.linalg.qr(a);qb,_=np.linalg.qr(b);singular=np.linalg.svd(qa.T@qb,compute_uv=False);sines=np.sqrt(np.maximum(0,1-singular**2));return {"max_angle_degrees":float(np.degrees(np.max(np.arcsin(np.clip(sines,0,1))))),"rms_sine_error":float(np.sqrt(np.mean(sines**2))),"projector_frobenius_error":float(np.linalg.norm(qa@qa.T-qb@qb.T,"fro"))}
def covariance_record(matrix):
    matrix=np.asarray(matrix,float);sym=.5*(matrix+matrix.T);return {"shape":list(matrix.shape),"all_finite":bool(np.all(np.isfinite(matrix))),"symmetry_max_abs":float(np.max(np.abs(matrix-matrix.T))),"minimum_eigenvalue":float(np.min(np.linalg.eigvalsh(sym)))}
def channel_smooth(y,mask,taus=TRUE_TAU):
    out=np.empty_like(y)
    for ch in range(P):
        signal=sum(C[ch,j]**2*se_kernel(taus[j]) for j in range(Q));cov=signal+RDIAG[ch]*np.eye(T);m=mask[:,ch];o=np.flatnonzero(m)
        if len(o):
            l=chol_psd(cov[np.ix_(o,o)]);alpha=np.linalg.solve(l.T,np.linalg.solve(l,y[o,ch]-D[ch]));out[:,ch]=D[ch]+signal[:,o]@alpha
        else:out[:,ch]=D[ch]
    return out

def independent_channel_predictive(data,masks):
    records={"masked_channel":[],"masked_block":[]}
    for i,(trial,mask) in enumerate(zip(data["test"],masks)):
        region="masked_channel" if i<2 else "masked_block"
        for ch in range(P):
            hidden=np.flatnonzero(~mask[:,ch]);observed=np.flatnonzero(mask[:,ch])
            if not len(hidden):continue
            signal=sum(C[ch,j]**2*se_kernel(TRUE_TAU[j]) for j in range(Q));cov=signal+RDIAG[ch]*np.eye(T)
            if len(observed):
                co=cov[np.ix_(observed,observed)];cross=signal[np.ix_(hidden,observed)];l=chol_psd(co);mean=D[ch]+cross@np.linalg.solve(l.T,np.linalg.solve(l,trial["observed"][observed,ch]-D[ch]));pc=cov[np.ix_(hidden,hidden)]-cross@np.linalg.solve(l.T,np.linalg.solve(l,cross.T))
            else:mean=np.full(len(hidden),D[ch]);pc=cov[np.ix_(hidden,hidden)]
            err=mean-trial["observed"][hidden,ch];records[region].append((gaussian_nll(trial["observed"][hidden,ch],mean,pc),float(np.sum(err**2)),len(hidden),int(np.sum(np.abs(err)<=1.9599639845*np.sqrt(np.diag(pc)))),float(np.mean(np.diag(pc)))))
    return records

def deterministic_two_stage_hidden(rows,data,masks,mean,basis,method,fa_model=None):
    records={"masked_channel":[],"masked_block":[]}
    for i,(trial,mask) in enumerate(zip(data["test"],masks)):
        smooth=channel_smooth(trial["observed"],mask)
        if fa_model is None:reconstruction=mean+(smooth-mean)@basis@basis.T
        else:
            scores,_=fa_bin_posterior(smooth,fa_model);reconstruction=fa_model["mean"]+scores@fa_model["loading"].T
        hidden=~mask;err=reconstruction[hidden]-trial["observed"][hidden];region="masked_channel" if i<2 else "masked_block";records[region].append((float(np.sum(err**2)),len(err)))
    for region,recs in records.items():add(rows,"test","primary",method,"deterministic_hidden_raw_reconstruction",region,"predictive_rmse",math.sqrt(sum(x[0] for x in recs)/sum(x[1] for x in recs)))
def static_scores(y,mean,basis):return (y-mean)@basis
def stress_dataset(name):
    data=generate_dataset();out=[]
    corr=.28**np.abs(np.subtract.outer(np.arange(P),np.arange(P)));rr=np.sqrt(RDIAG)[:,None]*corr*np.sqrt(RDIAG)[None,:]
    for i,trial in enumerate(data["test"]):
        z=trial["latent"].copy();y=trial["observed"].copy()
        if name=="changepoint":z[T//2:,0]+=2.;y=z@C.T+D+rng_for(60,i).normal(scale=np.sqrt(RDIAG),size=(T,P))
        elif name=="correlated_residuals":y=z@C.T+D+rng_for(61,i).multivariate_normal(np.zeros(P),rr,size=T)
        elif name=="poisson_sqrt":
            rate=np.exp(np.clip(z@C.T+D,-1.5,1.1));counts=rng_for(62,i).poisson(rate);y=np.sqrt(counts+.25)
        elif name=="equal_kernels":
            tr=generate_trial((.18,.18),3,i);z,y=tr["latent"],tr["observed"]
        elif name=="alignment_jitter":
            shift=int(rng_for(63,i).integers(-2,3));shifted=TIME+shift*DT;nominal=np.empty((T,Q));observed_latent=np.empty((T,Q))
            for j,tau in enumerate(TRUE_TAU):
                union,inverse=np.unique(np.concatenate([TIME,shifted]),return_inverse=True);draw=chol_psd(se_kernel(tau,union))@rng_for(65,i,j).normal(size=len(union));nominal[:,j]=draw[inverse[:T]];observed_latent[:,j]=draw[inverse[T:]]
            z=nominal;y=observed_latent@C.T+D+rng_for(64,i).normal(scale=np.sqrt(RDIAG),size=(T,P))
        out.append({"latent":z,"observed":y,"signal":z@C.T+D,"condition":name,**({"shift":shift,"observed_latent":observed_latent} if name=="alignment_jitter" else {})})
    return out

def stress_split_dataset(name):
    base=generate_dataset();result={}
    corr=.28**np.abs(np.subtract.outer(np.arange(P),np.arange(P)));rr=np.sqrt(RDIAG)[:,None]*corr*np.sqrt(RDIAG)[None,:]
    for split_code,split in enumerate(SPLITS,21):
        result[split]=[]
        for i,trial in enumerate(base[split]):
            z=trial["latent"].copy()
            if name=="changepoint":z[T//2:,0]+=2.;y=z@C.T+D+rng_for(90,split_code,i).normal(scale=np.sqrt(RDIAG),size=(T,P))
            else:y=z@C.T+D+rng_for(91,split_code,i).multivariate_normal(np.zeros(P),rr,size=T)
            result[split].append({"latent":z,"observed":y,"signal":z@C.T+D,"trial_id":split_code*1000+i,"condition":name})
    return result

def equal_kernel_dataset():
    data={}
    for code,split in enumerate(("train","validation","test"),11):
        data[split]=[]
        for i in range(SPLITS[split]):
            trial=generate_trial((.18,.18),code,i,condition="equal_kernels");trial["trial_id"]=code*1000+i;data[split].append(trial)
    return data
def score_stress(rows,name,trials):
    masks=[np.ones((T,P),bool) for _ in trials];rms=[];nll=[]
    for trial,mask in zip(trials,masks):
        post=latent_posterior(trial["observed"],mask,TRUE_TAU);rms.append(np.mean((post["mean"]-trial["latent"])**2));nll.append(gaussian_nll(trial["observed"].reshape(-1),observation_mean(),observation_cov(TRUE_TAU))/(T*P))
    add(rows,"stress",name,"oracle_gpfa_misspecified","latent_recovery","all","latent_rmse",np.sqrt(np.mean(rms)));add(rows,"stress",name,"oracle_gpfa_misspecified","observation_distribution","all","marginal_nll_per_entry",np.mean(nll))
    return {"latent_rmse":float(np.sqrt(np.mean(rms))),"nll_per_entry":float(np.mean(nll))}

def residual_diagnostics(trials,condition):
    residuals=[];step=[]
    for trial in trials:
        post=latent_posterior(trial["observed"],np.ones((T,P),bool),TRUE_TAU);pred=post["mean"]@C.T+D;res=trial["observed"]-pred;residuals.append(res)
        if condition=="changepoint":
            error=post["mean"]-trial["latent"];covered=np.abs(error)<=1.9599639845*np.sqrt(post["var"])
            step.append({"around_step_abs_mean":float(np.mean(np.abs(res[10:15]))),"outside_step_abs_mean":float(np.mean(np.abs(np.vstack([res[:8],res[17:]])))),"around_step_latent_rmse":float(np.sqrt(np.mean(error[10:15]**2))),"outside_step_latent_rmse":float(np.sqrt(np.mean(np.vstack([error[:8],error[17:]])**2))),"around_step_coverage":float(np.mean(covered[10:15])),"outside_step_coverage":float(np.mean(np.vstack([covered[:8],covered[17:]])))})
    array=np.vstack(residuals);cov=np.cov(array,rowvar=False);lag=[]
    for trial_res in residuals:
        for ch in range(P):lag.append(np.corrcoef(trial_res[:-1,ch],trial_res[1:,ch])[0,1])
    return {"residual_mean":np.mean(array,axis=0).tolist(),"residual_marginal_variance":np.var(array,axis=0).tolist(),"cross_channel_covariance":cov.tolist(),
            "maximum_absolute_offdiagonal":float(np.max(np.abs(cov[~np.eye(P,dtype=bool)]))),"mean_lag1_autocorrelation":float(np.nanmean(lag)),
            "model_misspecified_correlated_residual":condition=="correlated_residuals","changepoint_oversmoothing":step}

def posterior_nesting(data,masks):
    records=[]
    for trial,mask in zip(data["test"],masks):
        complete=latent_posterior(trial["observed"],np.ones((T,P),bool),TRUE_TAU);masked=latent_posterior(trial["observed"],mask,TRUE_TAU)
        records.append({"minimum_masked_minus_complete_variance":float(np.min(masked["var"]-complete["var"])),"mean_masked_minus_complete_variance":float(np.mean(masked["var"]-complete["var"]))})
    return records
def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n");w.writeheader();[w.writerow({**r,"value":format(r["value"],".17g")}) for r in rows]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def make_png(path,rows,diag):
    im=Image.new("RGB",(1500,940),"white");d=ImageDraw.Draw(im);font=ImageFont.load_default();d.text((24,14),"LDG covariance-prior trajectories — deterministic diagnostic",fill="black",font=font)
    def panel(b,t):d.rectangle(b,outline="black",width=2);d.text((b[0]+8,b[1]+7),t,fill="black",font=font)
    panel((24,44,740,450),"A. Masked trial latent truth and oracle posterior")
    plot=diag["masked_plot"];truth=np.array(plot["truth"]);mean=np.array(plot["posterior_mean"]);sd=np.array(plot["posterior_sd"]);lo=float(np.min(np.r_[truth,mean-1.96*sd]));hi=float(np.max(np.r_[truth,mean+1.96*sd]));px=lambda i:50+i/(T-1)*660;py=lambda x:420-(x-lo)/max(hi-lo,1e-12)*330
    d.line([(px(i),py(v)) for i,v in enumerate(truth)],fill="black",width=3);d.line([(px(i),py(v)) for i,v in enumerate(mean)],fill=(31,119,180),width=3)
    for i in range(T):d.line((px(i),py(mean[i]-1.96*sd[i]),px(i),py(mean[i]+1.96*sd[i])),fill=(170,210,240))
    d.text((45,430),"black=true latent; blue=oracle posterior mean; intervals fixed-parameter retrospective",fill="black",font=font)
    panel((765,44,1475,450),"B. Masked prediction NLL")
    methods=("oracle_gpfa_true_tau","selected_tau_gpfa","wrong_common_tau")
    for i,m in enumerate(methods[:3]):
        channel=next(float(r["value"]) for r in rows if r["condition"]=="primary" and r["method"]==m and r["metric"]=="predictive_nll_per_entry" and r["region"]=="masked_channel");block=next(float(r["value"]) for r in rows if r["condition"]=="primary" and r["method"]==m and r["metric"]=="predictive_nll_per_entry" and r["region"]=="masked_block");d.text((785,95+i*75),f"{m}: channel {channel:.3f}; block {block:.3f}",fill="black",font=font)
    panel((24,475,740,910),"C. Stress outcomes and warnings")
    for i,(name,v) in enumerate(diag["stress_results"].items()):d.text((42,520+i*45),f"{name}: RMSE {v['latent_rmse']:.3f}; NLL {v['nll_per_entry']:.3f}",fill="black",font=font)
    d.text((42,800),"Informative missingness: skipped-mask Gaussian inference is warning-only, not a correction.",fill="black",font=font)
    panel((765,475,1475,910),"D. Oracles and interpretation limits")
    o=diag["oracles"];lines=[f"known tau {list(TRUE_TAU)} and selected tau {diag['selection']['selected']} coincide in this realization",f"kernel minimum eigenvalue: {o['kernel_min_eigenvalue']:.3g}",f"posterior direct/precision mean: {o['direct_precision_mean_max_abs']:.3g}",f"posterior direct/precision covariance: {o['direct_precision_cov_max_abs']:.3g}",f"poison learned-state discrepancy: {o['poison_learned_state_max_abs']:.3g}","Posterior is fixed-parameter retrospective; parameter-selection uncertainty omitted.","Smooth posterior means are covariance-prior conditional, not discovered dynamics."]
    for i,x in enumerate(lines):d.text((785,520+i*48),x,fill="black",font=font)
    im.save(path,format="PNG",optimize=False)

def learned_snapshot(data):
    selected,grid=select_grid(data);pmean,pbasis=fit_pca(data["train"]);fa,starts=select_fa(data["train"],data["validation"])
    smooth={split:[{"observed":channel_smooth(t["observed"],np.ones((T,P),bool))} for t in data[split]] for split in ("train","validation")}
    spmean,spbasis=fit_pca(smooth["train"]);sfa,sstarts=select_fa(smooth["train"],smooth["validation"])
    models={"static_pca":{"mean":pmean,"basis":pbasis},"bounded_diagonal_fa_em":fa,"two_stage_channel_gp_then_pca":{"mean":spmean,"basis":spbasis},"two_stage_channel_gp_then_fa":sfa}
    alignments={name:fit_training_alignment(name,data if not name.startswith("two_stage") else {**data,"train":[{**t,"observed":channel_smooth(t["observed"],np.ones((T,P),bool))} for t in data["train"]]},model) for name,model in models.items()}
    return {"selected":selected,"grid":grid,"pca_mean":pmean,"pca_projector":pbasis@pbasis.T,"fa":fa,"fa_starts":starts,"alignments":alignments,
            "smooth_pca_mean":spmean,"smooth_pca_projector":spbasis@spbasis.T,"smooth_fa":sfa,"smooth_fa_starts":sstarts}

def snapshot_difference(a,b):
    values=[float(np.max(np.abs(a["pca_mean"]-b["pca_mean"]))),float(np.max(np.abs(a["pca_projector"]-b["pca_projector"]))),
            float(np.max(np.abs(a["smooth_pca_mean"]-b["smooth_pca_mean"]))),float(np.max(np.abs(a["smooth_pca_projector"]-b["smooth_pca_projector"])))]
    values += [max(abs(x[k]-y[k]) for k in ("train_nll_per_entry","validation_nll_per_entry","selection_score")) for x,y in zip(a["grid"],b["grid"])]
    for key in ("fa","smooth_fa"):
        values += [float(np.max(np.abs(a[key][field]-b[key][field]))) for field in ("mean","loading","uniqueness","covariance")]
    for key in ("fa_starts","smooth_fa_starts"):
        for x,y in zip(a[key],b[key]):values += [float(np.max(np.abs(x[field]-y[field]))) for field in ("mean","loading","uniqueness","covariance")]+[abs(x["train_nll_per_entry"]-y["train_nll_per_entry"]),abs(x["validation_nll_per_entry"]-y["validation_nll_per_entry"]),float(x["start_id"]!=y["start_id"])]
    for key in a["alignments"]:values.append(float(np.max(np.abs(a["alignments"][key]["rotation"]-b["alignments"][key]["rotation"]))));values.append(abs(a["alignments"][key]["training_objective_mse"]-b["alignments"][key]["training_objective_mse"]))
    values.append(float(a["selected"]!=b["selected"]));return max(values)

def snapshot_observation_only_difference(a,b):
    values=[float(np.max(np.abs(a["pca_mean"]-b["pca_mean"]))),float(np.max(np.abs(a["pca_projector"]-b["pca_projector"]))),
            float(np.max(np.abs(a["smooth_pca_mean"]-b["smooth_pca_mean"]))),float(np.max(np.abs(a["smooth_pca_projector"]-b["smooth_pca_projector"])))]
    values += [max(abs(x[k]-y[k]) for k in ("train_nll_per_entry","validation_nll_per_entry","selection_score")) for x,y in zip(a["grid"],b["grid"])]
    for key in ("fa","smooth_fa"):
        values += [float(np.max(np.abs(a[key][field]-b[key][field]))) for field in ("mean","loading","uniqueness","covariance")]
    for key in ("fa_starts","smooth_fa_starts"):
        for x,y in zip(a[key],b[key]):
            values += [float(np.max(np.abs(x[field]-y[field]))) for field in ("mean","loading","uniqueness","covariance")]
            values += [abs(x["train_nll_per_entry"]-y["train_nll_per_entry"]),abs(x["validation_nll_per_entry"]-y["validation_nll_per_entry"]),float(x["start_id"]!=y["start_id"])]
    values.append(float(a["selected"]!=b["selected"]));return max(values)

def snapshot_alignment_difference(a,b):
    return max([0.]+[max(float(np.max(np.abs(a["alignments"][key]["rotation"]-b["alignments"][key]["rotation"]))),
                               abs(a["alignments"][key]["training_objective_mse"]-b["alignments"][key]["training_objective_mse"]))
                         for key in a["alignments"]])

def fixed_original_output_snapshot(snapshot,data,masks):
    taus=snapshot["selected"];posts=[];predictions=[]
    cov=observation_cov(taus);mean=observation_mean()
    for trial,mask in zip(data["test"],masks):
        post=latent_posterior(trial["observed"],mask,taus);pred=condition_gaussian(mean,cov,trial["observed"],mask)
        posts.append((post["mean"],post["cov"]));predictions.append((pred["mean"],pred["cov"]))
    return {"posts":posts,"predictions":predictions}

def fixed_output_difference(a,b):
    values=[0.]
    for left,right in zip(a["posts"],b["posts"]):
        values.extend(float(np.max(np.abs(x-y))) for x,y in zip(left,right))
    for left,right in zip(a["predictions"],b["predictions"]):
        values.extend(float(np.max(np.abs(x-y))) for x,y in zip(left,right))
    return max(values)

def build(root,quiet=False):
    root.mkdir(parents=True,exist_ok=True);data=generate_dataset();masks,mask_records=mask_design();selected,grid=select_grid(data);rows=[]
    for method,taus,kernel in (("oracle_gpfa_true_tau",TRUE_TAU,"se"),("selected_tau_gpfa",selected,"se"),("wrong_common_tau",COMMON_TAU,"se"),("independent_bin",TRUE_TAU,"identity")):score_probabilistic(rows,data,masks,"primary",method,taus,kernel)
    score_recursive_ar(rows,data,masks)
    for j,tau in enumerate(selected):add(rows,"validation","primary","selected_tau_gpfa",f"latent_{j+1}","timescale","selected_tau",tau);add(rows,"validation","primary","selected_tau_gpfa",f"latent_{j+1}","timescale","absolute_tau_error",abs(tau-TRUE_TAU[j]))
    ranked=sorted(grid,key=lambda x:(x["combined_total_nll"],x["tau_1"],x["tau_2"]));gap=ranked[1]["combined_total_nll"]-ranked[0]["combined_total_nll"]
    for candidate in grid:
        label=f"tau_{candidate['tau_1']}_{candidate['tau_2']}";add(rows,"validation","primary","selected_tau_gpfa",label,"selection_provenance","train_total_nll",candidate["train_total_nll"]);add(rows,"validation","primary","selected_tau_gpfa",label,"selection_provenance","validation_total_nll",candidate["validation_total_nll"]);add(rows,"validation","primary","selected_tau_gpfa",label,"selection_provenance","combined_total_nll",candidate["combined_total_nll"]);add(rows,"validation","primary","selected_tau_gpfa",label,"selection_provenance","rank",candidate["rank"])
    add(rows,"validation","primary","selected_tau_gpfa","grid","selection_provenance","next_candidate_total_nll_gap",gap)
    for j,tau in enumerate(selected):add(rows,"validation","primary","selected_tau_gpfa",f"latent_{j+1}","timescale","relative_tau_error",abs(tau-TRUE_TAU[j])/TRUE_TAU[j])

    pmean,pbasis=fit_pca(data["train"]);fa,fa_starts=select_fa(data["train"],data["validation"]);mean_diag=mean_diagonal_model(data["train"])
    smooth={split:[] for split in SPLITS}
    for split in SPLITS:
        for i,trial in enumerate(data[split]):
            mask=masks[i] if split=="test" else np.ones((T,P),bool);smooth[split].append({"observed":channel_smooth(trial["observed"],mask),"latent":trial["latent"]})
    spmean,spbasis=fit_pca(smooth["train"]);sfa,sfa_starts=select_fa(smooth["train"],smooth["validation"])
    alignment_models={"static_pca":{"mean":pmean,"basis":pbasis},"bounded_diagonal_fa_em":fa,"two_stage_channel_gp_then_pca":{"mean":spmean,"basis":spbasis},"two_stage_channel_gp_then_fa":sfa}
    alignments={name:fit_training_alignment(name,data if not name.startswith("two_stage") else smooth,model) for name,model in alignment_models.items()}
    for method,basis in (("static_pca",pbasis),("bounded_diagonal_fa_em",fa["loading"]),("two_stage_channel_gp_then_pca",spbasis),("two_stage_channel_gp_then_fa",sfa["loading"])):
        sm=subspace_metrics(basis,C);add(rows,"train","primary",method,"loading_subspace","all","loading_subspace_angle_degrees",sm["max_angle_degrees"]);add(rows,"train","primary",method,"loading_subspace","all","loading_subspace_rms_sine_error",sm["rms_sine_error"]);add(rows,"train","primary",method,"loading_subspace","all","loading_subspace_projector_frobenius_error",sm["projector_frobenius_error"])
    score_fa_model(rows,data,masks,fa,"bounded_diagonal_fa_em",alignment=alignments["bounded_diagonal_fa_em"]);score_mean_diagonal(rows,data,masks,mean_diag)
    learned_path_metrics(rows,"static_pca",data,alignment_models["static_pca"],alignments["static_pca"])
    learned_path_metrics(rows,"two_stage_channel_gp_then_pca",smooth,alignment_models["two_stage_channel_gp_then_pca"],alignments["two_stage_channel_gp_then_pca"])
    learned_path_metrics(rows,"two_stage_channel_gp_then_fa",smooth,sfa,alignments["two_stage_channel_gp_then_fa"],condition="primary")
    deterministic_two_stage_hidden(rows,data,masks,spmean,spbasis,"two_stage_channel_gp_then_pca")
    deterministic_two_stage_hidden(rows,data,masks,sfa["mean"],sfa["loading"],"two_stage_channel_gp_then_fa",fa_model=sfa)
    # Independent-channel smoother and explicit hidden-channel no-cross-channel behavior.
    smooth_rmse=[];hidden_prior=[];pinv=np.linalg.pinv(C.T)
    for i,(trial,mask) in enumerate(zip(data["test"],masks)):
        ys=channel_smooth(trial["observed"],mask);smooth_rmse.append(np.mean(((ys-D)@pinv-trial["latent"])**2))
        if i<2:hidden_prior.append(float(np.max(np.abs(ys[:,i]-D[i]))))
    add(rows,"test","primary","independent_channel_gp_smoothing","latent_recovery","all","latent_rmse",math.sqrt(np.mean(smooth_rmse)));add(rows,"test","primary","independent_channel_gp_smoothing","fully_hidden_channel","masked_channel","hidden_channel_prior_offset_max_abs",max(hidden_prior))
    channel_predictions=independent_channel_predictive(data,masks)
    for region,recs in channel_predictions.items():
        count=sum(x[2] for x in recs);add(rows,"test","primary","independent_channel_gp_smoothing","masked_observation_prediction",region,"predictive_nll_per_entry",sum(x[0] for x in recs)/count);add(rows,"test","primary","independent_channel_gp_smoothing","masked_observation_prediction",region,"predictive_rmse",math.sqrt(sum(x[1] for x in recs)/count));add(rows,"test","primary","independent_channel_gp_smoothing","masked_observation_prediction",region,"predictive_95_coverage",sum(x[3] for x in recs)/count);add(rows,"test","primary","independent_channel_gp_smoothing","masked_observation_prediction",region,"mean_predictive_variance",np.mean([x[4] for x in recs]));add(rows,"test","primary","independent_channel_gp_smoothing","masked_observation_prediction",region,"hidden_target_count",count)

    informative=[];missing_magnitude=[]
    for trial in data["test"]:
        mask=np.abs(trial["observed"])<1.25;post=latent_posterior(trial["observed"],mask,TRUE_TAU);informative.append(np.mean((post["mean"]-trial["latent"])**2));missing_magnitude.append((np.mean(np.abs(trial["observed"])[~mask]),np.mean(np.abs(trial["observed"])[mask]),np.mean(~mask)))
    add(rows,"test","informative_missingness","oracle_gpfa_skipped_mask_warning","latent_recovery_warning_only","all","latent_rmse",math.sqrt(np.mean(informative)))

    changepoint_data=stress_split_dataset("changepoint");correlated_data=stress_split_dataset("correlated_residuals")
    stress_trials={"changepoint":changepoint_data["test"],"correlated_residuals":correlated_data["test"],"poisson_sqrt":stress_dataset("poisson_sqrt"),"alignment_jitter":stress_dataset("alignment_jitter")};stress={name:score_stress(rows,name,trials) for name,trials in stress_trials.items()}
    correlated_fa,correlated_starts=select_fa(correlated_data["train"],correlated_data["validation"]);correlated_alignment=fit_training_alignment("bounded_diagonal_fa_em",correlated_data,correlated_fa);score_fa_model(rows,correlated_data,[np.ones((T,P),bool) for _ in correlated_data["test"]],correlated_fa,"correlated_residual_static_fa",condition="correlated_residuals",alignment=correlated_alignment)
    correlated_smooth={split:[{"observed":channel_smooth(t["observed"],np.ones((T,P),bool)),"latent":t["latent"]} for t in correlated_data[split]] for split in SPLITS};correlated_sfa,correlated_sstarts=select_fa(correlated_smooth["train"],correlated_smooth["validation"]);correlated_salign=fit_training_alignment("two_stage_channel_gp_then_fa",correlated_smooth,correlated_sfa);learned_path_metrics(rows,"correlated_residual_two_stage_fa",correlated_smooth,correlated_sfa,correlated_salign,condition="correlated_residuals");add(rows,"stress","correlated_residuals","correlated_residual_two_stage_fa","loading_subspace","all","loading_subspace_angle_degrees",subspace_angle(correlated_sfa["loading"],C))
    # Changepoint block spans the step.
    cp_masks=[]
    for _ in changepoint_data["test"]:m=np.ones((T,P),bool);m[9:16]=False;cp_masks.append(m)
    score_probabilistic(rows,changepoint_data,cp_masks,"changepoint","oracle_gpfa_step_mask_misspecified",TRUE_TAU)
    residuals={"primary":residual_diagnostics(data["test"],"primary"),"correlated_residuals":residual_diagnostics(stress_trials["correlated_residuals"],"correlated_residuals"),"changepoint":residual_diagnostics(stress_trials["changepoint"],"changepoint")}
    for residual_condition in ("primary","correlated_residuals"):
        rd=residuals[residual_condition];split_label="test" if residual_condition=="primary" else "stress"
        add(rows,split_label,residual_condition,"oracle_gpfa_residual_diagnostic","observation_residual","all","residual_mean_abs_max",np.max(np.abs(rd["residual_mean"])));add(rows,split_label,residual_condition,"oracle_gpfa_residual_diagnostic","observation_residual","all","residual_variance_mean",np.mean(rd["residual_marginal_variance"]));add(rows,split_label,residual_condition,"oracle_gpfa_residual_diagnostic","observation_residual","all","residual_max_offdiagonal_covariance",rd["maximum_absolute_offdiagonal"]);add(rows,split_label,residual_condition,"oracle_gpfa_residual_diagnostic","observation_residual","all","residual_lag1_autocorrelation",rd["mean_lag1_autocorrelation"])
    cp=residuals["changepoint"]["changepoint_oversmoothing"]
    add(rows,"stress","changepoint","oracle_gpfa_oversmoothing_diagnostic","latent_recovery","around_step","latent_rmse",np.mean([x["around_step_latent_rmse"] for x in cp]));add(rows,"stress","changepoint","oracle_gpfa_oversmoothing_diagnostic","latent_recovery","outside_step","latent_rmse",np.mean([x["outside_step_latent_rmse"] for x in cp]));add(rows,"stress","changepoint","oracle_gpfa_oversmoothing_diagnostic","latent_recovery","around_step","latent_95_coverage",np.mean([x["around_step_coverage"] for x in cp]))
    # Equal-kernel independent splits and train-only alignments.
    equal_data=equal_kernel_dataset();equal_masks=[np.ones((T,P),bool) for _ in equal_data["test"]];score_probabilistic(rows,equal_data,equal_masks,"equal_kernels","oracle_gpfa_equal_tau",(.18,.18))
    epmean,epbasis=fit_pca(equal_data["train"]);efa,efa_starts=select_fa(equal_data["train"],equal_data["validation"]);ealign=fit_training_alignment("static_pca",equal_data,{"mean":epmean,"basis":epbasis});learned_path_metrics(rows,"equal_kernel_static_pca",equal_data,{"mean":epmean,"basis":epbasis},ealign,condition="equal_kernels");add(rows,"stress","equal_kernels","equal_kernel_static_pca","joint_2d_loading_subspace","all","loading_subspace_angle_degrees",subspace_angle(epbasis,C))
    stress["equal_kernels"]={"latent_rmse":next(r["value"] for r in rows if r["condition"]=="equal_kernels" and r["method"]=="oracle_gpfa_equal_tau" and r["metric"]=="latent_rmse"),"nll_per_entry":next(r["value"] for r in rows if r["condition"]=="equal_kernels" and r["method"]=="oracle_gpfa_equal_tau" and r["metric"]=="marginal_nll_per_entry")}

    # Core covariance and posterior oracles.
    kernel_diag=max(float(np.max(np.abs(np.diag(se_kernel(tau))-1))) for tau in TRUE_TAU);signal_diag=1-NUGGET
    kernel_eigs=[np.min(np.linalg.eigvalsh(se_kernel(tau))) for tau in TRUE_TAU];stacked=observation_cov(TRUE_TAU);short=data["test"][0];mask=masks[0];direct=latent_posterior(short["observed"],mask,TRUE_TAU);pm,pc=precision_posterior(short["observed"],mask,TRUE_TAU)
    kz=latent_cov(TRUE_TAU);a=observation_matrix();joint=np.block([[kz,kz@a.T],[a@kz,stacked]]);obs=np.flatnonzero(mask.reshape(-1));co=stacked[np.ix_(obs,obs)];cross=kz@a[obs].T;l=chol_psd(co);generic_mean=cross@np.linalg.solve(l.T,np.linalg.solve(l,short["observed"].reshape(-1)[obs]-observation_mean()[obs]));generic_cov=kz-cross@np.linalg.solve(l.T,np.linalg.solve(l,cross.T))
    ar_oracles={}
    for label,trial_mask in (("complete",np.ones((T,P),bool)),("masked",masks[0])):
        recursive=kalman_rts(short["observed"],trial_mask,TRUE_TAU);bm,bc,bx,_=ar_batch_blocks(short["observed"],trial_mask,TRUE_TAU);ar_oracles[label]={"mean_max_abs":float(np.max(np.abs(recursive["mean"]-bm))),"marginal_cov_max_abs":float(np.max(np.abs(recursive["covariances"]-bc))),"adjacent_cross_max_abs":float(np.max(np.abs(recursive["adjacent_cross"]-bx)))}
    recursive=kalman_rts(short["observed"],mask,TRUE_TAU);direct_hidden=condition_gaussian(observation_mean(),observation_cov(TRUE_TAU,kernel="ar"),short["observed"],mask);recursive_mean=[];recursive_var=[]
    for index in direct_hidden["target_indices"]:
        t=index//P;ch=index%P;recursive_mean.append(D[ch]+C[ch]@recursive["mean"][t]);recursive_var.append(C[ch]@recursive["covariances"][t]@C[ch]+RDIAG[ch])
    hidden_equivalence={"mean_max_abs":float(np.max(np.abs(np.asarray(recursive_mean)-direct_hidden["mean"]))),"variance_max_abs":float(np.max(np.abs(np.asarray(recursive_var)-np.diag(direct_hidden["cov"]))))}
    full_latent_rts=rts_full_covariance(recursive);hidden_indices=direct_hidden["target_indices"];hidden_a=observation_matrix()[hidden_indices];hidden_r=np.diag(np.tile(RDIAG,T)[hidden_indices]);recursive_hidden_full=hidden_a@full_latent_rts@hidden_a.T+hidden_r
    hidden_equivalence["full_covariance_max_abs"]=float(np.max(np.abs(recursive_hidden_full-direct_hidden["cov"])))
    nesting=posterior_nesting(data,masks)
    # Generator, stacking, complete/masked, independent-bin, reset, and perturbation oracles.
    generator_signal_max=max(float(np.max(np.abs(trial["signal"]-(trial["latent"]@C.T+D)))) for split in SPLITS for trial in data[split])
    stacking_max=max(float(np.max(np.abs(observation_matrix()@trial["latent"].reshape(-1)+observation_mean()-trial["signal"].reshape(-1)))) for split in SPLITS for trial in data[split])
    posterior_forms={}
    for label,trial_mask in (("complete",np.ones((T,P),bool)),("masked",masks[0])):
        direct_form=latent_posterior(short["observed"],trial_mask,TRUE_TAU);precision_mean,precision_cov=precision_posterior(short["observed"],trial_mask,TRUE_TAU);obs_idx=np.flatnonzero(trial_mask.reshape(-1));cross=kz@a[obs_idx].T;obs_cov=stacked[np.ix_(obs_idx,obs_idx)];ll=chol_psd(obs_cov);joint_mean=cross@np.linalg.solve(ll.T,np.linalg.solve(ll,short["observed"].reshape(-1)[obs_idx]-observation_mean()[obs_idx]));joint_cov=kz-cross@np.linalg.solve(ll.T,np.linalg.solve(ll,cross.T));posterior_forms[label]={"direct_precision_mean_max_abs":float(np.max(np.abs(direct_form["mean"].reshape(-1)-precision_mean))),"direct_precision_cov_max_abs":float(np.max(np.abs(direct_form["cov"]-precision_cov))),"direct_joint_mean_max_abs":float(np.max(np.abs(direct_form["mean"].reshape(-1)-joint_mean))),"direct_joint_cov_max_abs":float(np.max(np.abs(direct_form["cov"]-joint_cov)))}
    independent_joint=latent_posterior(short["observed"],masks[0],TRUE_TAU,"identity");separate_mean=[];separate_cov=np.zeros((Q*T,Q*T))
    for t in range(T):
        obs=np.flatnonzero(masks[0][t]);h=C[obs];rr=R[np.ix_(obs,obs)];s=h@h.T+rr;k=np.linalg.solve(s,h).T;separate_mean.append(k@(short["observed"][t,obs]-D[obs]));separate_cov[t*Q:(t+1)*Q,t*Q:(t+1)*Q]=np.eye(Q)-k@h
    independent_bin_equivalence={"mean_max_abs":float(np.max(np.abs(independent_joint["mean"]-np.asarray(separate_mean)))),"covariance_max_abs":float(np.max(np.abs(independent_joint["cov"]-separate_cov)))}
    hidden_perturbed=short["observed"].copy();hidden_perturbed[~mask]+=777.;observed_perturbed=short["observed"].copy();observed_perturbed[mask]+=0.25
    base_hidden=latent_posterior(short["observed"],mask,TRUE_TAU);hidden_post=latent_posterior(hidden_perturbed,mask,TRUE_TAU);observed_post=latent_posterior(observed_perturbed,mask,TRUE_TAU)
    base_pred=condition_gaussian(observation_mean(),stacked,short["observed"],mask);hidden_pred=condition_gaussian(observation_mean(),stacked,hidden_perturbed,mask);observed_pred=condition_gaussian(observation_mean(),stacked,observed_perturbed,mask)
    mask_perturbation={"observed_indices":np.flatnonzero(mask.reshape(-1)).tolist(),"hidden_indices":np.flatnonzero(~mask.reshape(-1)).tolist(),"observed_count":int(np.sum(mask)),"hidden_count":int(np.sum(~mask)),"hidden_value_change_norm":float(np.linalg.norm(hidden_perturbed-short["observed"])),"observed_value_change_norm":float(np.linalg.norm(observed_perturbed-short["observed"])),"hidden_posterior_mean_max_abs":float(np.max(np.abs(base_hidden["mean"]-hidden_post["mean"]))),"hidden_posterior_cov_max_abs":float(np.max(np.abs(base_hidden["cov"]-hidden_post["cov"]))),"hidden_prediction_mean_max_abs":float(np.max(np.abs(base_pred["mean"]-hidden_pred["mean"]))),"hidden_prediction_cov_max_abs":float(np.max(np.abs(base_pred["cov"]-hidden_pred["cov"]))),"observed_perturb_posterior_mean_change":float(np.max(np.abs(base_hidden["mean"]-observed_post["mean"]))),"observed_perturb_prediction_mean_change":float(np.max(np.abs(base_pred["mean"]-observed_pred["mean"]))) }
    loop_posts=[latent_posterior(trial["observed"],trial_mask,TRUE_TAU) for trial,trial_mask in zip(data["test"],masks)];reset_max=0.
    for i,(trial,trial_mask) in enumerate(zip(data["test"],masks)):
        independent=latent_posterior(trial["observed"],trial_mask,TRUE_TAU);reset_max=max(reset_max,float(np.max(np.abs(loop_posts[i]["mean"]-independent["mean"]))),float(np.max(np.abs(loop_posts[i]["cov"]-independent["cov"]))))
    trial_ids={split:[trial["trial_id"] for trial in data[split]] for split in SPLITS};trial_id_disjoint=bool(set(trial_ids["train"]).isdisjoint(trial_ids["validation"]) and set(trial_ids["train"]).isdisjoint(trial_ids["test"]) and set(trial_ids["validation"]).isdisjoint(trial_ids["test"]))
    # Fully hidden-channel other-channel perturbation.
    hidden_trial=data["test"][0];hidden_mask=masks[0];base_channel=channel_smooth(hidden_trial["observed"],hidden_mask);other_changed=hidden_trial["observed"].copy();other_changed[:,1:]+=rng_for(86).normal(scale=5,size=(T,P-1));perturbed_channel=channel_smooth(other_changed,hidden_mask);signal0=sum(C[0,j]**2*se_kernel(TRUE_TAU[j]) for j in range(Q));hidden_channel_variance=np.diag(signal0+RDIAG[0]*np.eye(T));cross_channel_perturbation={"mean_max_abs":float(np.max(np.abs(base_channel[:,0]-perturbed_channel[:,0]))),"variance_max_abs":0.0,"prior_offset_max_abs":float(np.max(np.abs(base_channel[:,0]-D[0]))),"prior_variance_mean":float(np.mean(hidden_channel_variance)),"perturbation_norm":float(np.linalg.norm(other_changed-hidden_trial["observed"])),"uses_other_channels":bool(np.max(np.abs(base_channel[:,0]-perturbed_channel[:,0]))>TOL["algebra"])}
    # Hidden predictive equivalence against direct conditioning already used by score routines.
    predictive_provenance="95% predictive intervals use conditional Gaussian covariance for held-out observations; latent coverage uses posterior marginal variance; warning-only stresses have no acceptance role"

    def copy_data_bundle(source):
        return {split:[{**trial,"observed":trial["observed"].copy(),"latent":trial["latent"].copy(),"signal":trial["signal"].copy()} for trial in source[split]] for split in SPLITS}
    baseline=learned_snapshot(data);baseline_outputs=fixed_original_output_snapshot(baseline,data,masks)
    observation_poison_data=copy_data_bundle(data)
    for i,trial in enumerate(observation_poison_data["test"]):trial["observed"][:]=rng_for(80,i).normal(loc=1e4,size=(T,P))
    observation_poison_snapshot=learned_snapshot(observation_poison_data)
    observation_fit_difference=snapshot_difference(baseline,observation_poison_snapshot)
    observation_fixed_output_difference=fixed_output_difference(baseline_outputs,fixed_original_output_snapshot(observation_poison_snapshot,data,masks))
    all_latent_poison_data=copy_data_bundle(data);poisoned_all_latent_norm=0.
    for split_code,split in enumerate(SPLITS,1):
        for i,trial in enumerate(all_latent_poison_data[split]):
            trial["latent"][:]=rng_for(81,split_code,i).normal(loc=-1e4,size=(T,Q));poisoned_all_latent_norm+=float(np.linalg.norm(trial["latent"]))
    all_latent_poison_snapshot=learned_snapshot(all_latent_poison_data)
    latent_observation_fit_difference=snapshot_observation_only_difference(baseline,all_latent_poison_snapshot)
    training_alignment_expected_change=snapshot_alignment_difference(baseline,all_latent_poison_snapshot)
    latent_fixed_output_difference=fixed_output_difference(baseline_outputs,fixed_original_output_snapshot(all_latent_poison_snapshot,data,masks))
    test_latent_poison_data=copy_data_bundle(data)
    for i,trial in enumerate(test_latent_poison_data["test"]):trial["latent"][:]=rng_for(87,i).normal(loc=2e4,size=(T,Q))
    test_latent_poison_snapshot=learned_snapshot(test_latent_poison_data)
    test_truth_alignment_difference=snapshot_alignment_difference(baseline,test_latent_poison_snapshot)
    poison_registry={"test_observation_poison":{"changed_norm":float(sum(np.linalg.norm(t["observed"]-data["test"][i]["observed"]) for i,t in enumerate(observation_poison_data["test"]))),"complete_fit_max_abs":observation_fit_difference,"fixed_original_output_max_abs":observation_fixed_output_difference},
                     "all_latent_truth_poison":{"changed_norm":poisoned_all_latent_norm,"observation_only_fit_max_abs":latent_observation_fit_difference,"training_alignment_expected_change":training_alignment_expected_change,"fixed_original_output_max_abs":latent_fixed_output_difference},
                     "test_latent_truth_poison":{"changed_norm":float(sum(np.linalg.norm(t["latent"]-data["test"][i]["latent"]) for i,t in enumerate(test_latent_poison_data["test"]))),"training_alignment_max_abs":test_truth_alignment_difference}}
    regeneration=generate_dataset();regen=max(float(np.max(np.abs(data[s][i][key]-regeneration[s][i][key]))) for s in SPLITS for i in range(SPLITS[s]) for key in ("latent","observed"))
    trial_block=np.kron(np.eye(2),latent_cov(TRUE_TAU));cross_trial_block=float(np.max(np.abs(trial_block[:Q*T,Q*T:])))
    covariance_registry={}
    for j,tau in enumerate(TRUE_TAU):covariance_registry[f"kernel_{j}"]=covariance_record(se_kernel(tau))
    covariance_registry["observation_complete"]=covariance_record(stacked);covariance_registry["observation_masked"]=covariance_record(stacked[np.ix_(np.flatnonzero(mask.reshape(-1)),np.flatnonzero(mask.reshape(-1)))])
    covariance_registry["posterior_complete"]=covariance_record(latent_posterior(short["observed"],np.ones((T,P),bool),TRUE_TAU)["cov"]);covariance_registry["posterior_masked"]=covariance_record(direct["cov"])
    covariance_registry["selected_fa_covariance"]=covariance_record(fa["covariance"]);covariance_registry["selected_fa_posterior"]=covariance_record(fa_bin_posterior(short["observed"],fa)[1]);covariance_registry["two_stage_fa_covariance"]=covariance_record(sfa["covariance"]);covariance_registry["two_stage_fa_posterior"]=covariance_record(fa_bin_posterior(smooth["test"][0]["observed"],sfa)[1])
    rho,q=ar_parameters(TRUE_TAU);covariance_registry["ar_process"]=covariance_record(np.diag(q));ar_complete=kalman_rts(short["observed"],np.ones((T,P),bool),TRUE_TAU);ar_masked=kalman_rts(short["observed"],mask,TRUE_TAU)
    for label,result in (("complete",ar_complete),("masked",ar_masked)):
        for t in range(T):covariance_registry[f"ar_filter_{label}_{t}"]=covariance_record(result["filter_cov"][t]);covariance_registry[f"ar_smoother_{label}_{t}"]=covariance_record(result["covariances"][t])
    covariance_registry["hidden_predictive_covariance"]=covariance_record(direct_hidden["cov"])
    covariance_registry["ar_rts_full_latent_covariance"]=covariance_record(full_latent_rts);covariance_registry["ar_recursive_hidden_predictive_full_covariance"]=covariance_record(recursive_hidden_full)
    fa_registry={"primary":{"selected_start":fa["start_id"],"starts":[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in record.items() if k not in ("mean","loading","uniqueness","covariance")} for record in fa_starts]},
                 "two_stage":{"selected_start":sfa["start_id"],"starts":[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in record.items() if k not in ("mean","loading","uniqueness","covariance")} for record in sfa_starts]}}
    jitter_records={"shifts":[t["shift"] for t in stress_trials["alignment_jitter"]],"nominal_observed_latent_max_abs_difference":[float(np.max(np.abs(t["latent"]-t["observed_latent"]))) for t in stress_trials["alignment_jitter"]],"zero_shift_exact_identity":all(float(np.max(np.abs(t["latent"]-t["observed_latent"])))==0.0 for t in stress_trials["alignment_jitter"] if t["shift"]==0),"no_wrap":True,"construction":"one GP draw on the unique union of nominal and shifted physical times; shared times index the exact same draw including nugget; observations use shifted path; scoring uses nominal path"}
    alignment_registry={name:{"rotation":record["rotation"].tolist(),"training_objective_mse":record["training_objective_mse"],"fit_split":"train","test_truth_used":False} for name,record in alignments.items()}
    equal_registry={"tau":[.18,.18],"split_trial_ids":{split:[trial["trial_id"] for trial in equal_data[split]] for split in SPLITS},"static_pca_alignment":{"rotation":ealign["rotation"].tolist(),"training_objective_mse":ealign["training_objective_mse"],"test_truth_used":False},"coordinate_tau_identity_emitted":False}
    diagnostics={"selection":{"selected":list(selected),"grid":grid,"criterion":"minimum combined total train+validation trial NLL","next_candidate_total_nll_gap":gap,"selected_absolute_errors":[abs(selected[j]-TRUE_TAU[j]) for j in range(Q)],"selected_relative_errors":[abs(selected[j]-TRUE_TAU[j])/TRUE_TAU[j] for j in range(Q)]},"trial_ids":{"primary":trial_ids,"split_disjoint":trial_id_disjoint,"reset_policy":"independent GP draw and prior per trial"},"masks":mask_records,"fa_em":fa_registry,"train_only_alignments":alignment_registry,"equal_kernel":equal_registry,
                 "masked_plot":{"trial":0,"latent_dimension":0,"truth":short["latent"][:,0].tolist(),"posterior_mean":direct["mean"][:,0].tolist(),"posterior_sd":np.sqrt(direct["var"][:,0]).tolist(),"mask_type":"complete_channel_observation_mask"},
                 "stress_results":stress,"residual_diagnostics":residuals,"posterior_nesting":nesting,"alignment_jitter":jitter_records,
                 "informative_missingness":{"mechanism":"mask observed only when abs(y)<1.25; skipped-mask inference ignores information in the mask","warning_only":True,"missing_fraction":float(np.mean([x[2] for x in missing_magnitude])),"mean_abs_inside_missing":float(np.mean([x[0] for x in missing_magnitude])),"mean_abs_outside_missing":float(np.mean([x[1] for x in missing_magnitude])),"coverage_acceptance":False},
                 "covariance_registry":covariance_registry,"generator_oracles":{"signal_max_abs":generator_signal_max,"time_major_kron_stacking_max_abs":stacking_max},"posterior_form_oracles":posterior_forms,"independent_bin_per_time_equivalence":independent_bin_equivalence,"mask_perturbation":mask_perturbation,"per_trial_reset_max_abs":reset_max,
                 "oracles":{"kernel_diagonal_max_abs_from_one":kernel_diag,"kernel_signal_diagonal":signal_diag,"kernel_nugget_diagonal":NUGGET,"kernel_min_eigenvalue":float(min(kernel_eigs)),"stacked_covariance_symmetry_max_abs":float(np.max(np.abs(stacked-stacked.T))),"stacked_covariance_min_eigenvalue":float(np.min(np.linalg.eigvalsh(stacked))),
                            "direct_precision_mean_max_abs":float(np.max(np.abs(direct["mean"].reshape(-1)-pm))),"direct_precision_cov_max_abs":float(np.max(np.abs(direct["cov"]-pc))),"generic_joint_mean_max_abs":float(np.max(np.abs(direct["mean"].reshape(-1)-generic_mean))),"generic_joint_cov_max_abs":float(np.max(np.abs(direct["cov"]-generic_cov))),
                            "ar_recursive_batch":ar_oracles,"hidden_predictive_direct_equivalence":hidden_equivalence,"poison_registry":poison_registry,"poison_learned_state_max_abs":max(observation_fit_difference,latent_observation_fit_difference,test_truth_alignment_difference),"poisoned_test_norm":poison_registry["test_observation_poison"]["changed_norm"],"poisoned_all_latent_norm":poisoned_all_latent_norm,"fixed_original_posterior_prediction_max_abs":max(observation_fixed_output_difference,latent_fixed_output_difference),"independent_regeneration_max_abs":regen,"mask_targets_excluded_by_perturbation":mask_perturbation["hidden_posterior_mean_max_abs"]<=TOL["poison"] and mask_perturbation["hidden_prediction_mean_max_abs"]<=TOL["poison"] and mask_perturbation["observed_perturb_posterior_mean_change"]>0,"selected_grid_exact_recompute":baseline["selected"]==selected,"trial_block_covariance_cross_trial_max_abs":cross_trial_block,"alignment_scoring_only":"orthogonal Procrustes used only for reported learned latent path scores; all matrices fitted on training truth only"},
                 "predictive_coverage_provenance":predictive_provenance,"independent_channel_hidden_channel":cross_channel_perturbation,
                 "methods":{"ar_lgssm_rts":"recursive vector Kalman filter and RTS smoother with arbitrary observation masks and per-trial resets","bounded_diagonal_fa_em":"six-start diagonal-noise FA EM, floor 1e-6, <=300 iterations, relative train-NLL tolerance 1e-9","two_stage_channel_gp_then_fa":"independent-channel GP smoothing followed by separately selected bounded FA on pooled smoothed train bins","mean_diagonal_gaussian":"training mean plus diagonal marginal variance","independent_channel_gp_smoothing":"each channel uses only its own observed samples; fully hidden channel remains at offset prior mean"},
                 "scientific_limits":["Smooth posterior means are conditional on a chosen covariance prior and are not discovered transition dynamics.","Equal kernels identify only the joint two-dimensional subspace; Procrustes alignment is scoring-only.","Informative missingness and Poisson-square-root conditions are warning/approximation diagnostics without coverage acceptance.","Correlated residual diagnostics explicitly mark the diagonal-residual model as misspecified."]}
    write_csv(root/"metrics.csv",rows);(root/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True,allow_nan=False)+"\n");make_png(root/"summary.png",rows,diagnostics);hashes={n:sha(root/n) for n in ARTIFACT_NAMES}
    manifest={"schema_version":1,"artifact":"covariance-prior-trajectories","date":DATE,"master_seed":MASTER_SEED,"runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"pillow":getattr(sys.modules.get('PIL'),'__version__','unknown'),"platform":platform.platform()},"dimensions":{"latent":Q,"observed":P,"time":T},"time_grid":TIME.tolist(),"split_trials":SPLITS,"kernel":{"formula":"(1-nugget)*exp(-0.5*delta^2/tau^2)+nugget*I","true_tau":list(TRUE_TAU),"nugget":NUGGET,"diagonal":1.0,"ar_rho_formula":"(1-nugget)*exp(-0.5*dt^2/tau^2)","ar_q_formula":"1-rho^2"},"loading_C":C.tolist(),"offset_d":D.tolist(),"residual_R":R.tolist(),"tau_grid":[list(x) for x in TAU_GRID],"selection":"train+validation observation marginal NLL only","masks":mask_records,"fa_em":{"starts":6,"floor":1e-6,"max_iterations":300,"relative_train_nll_tolerance":1e-9,"selection":"lowest train NLL, then lower validation NLL, then start ID","records":"diagnostics.fa_em"},"methods":{"oracle_gpfa_true_tau":"known C,d,R,tau exact dense posterior","selected_tau_gpfa":"finite train+validation NLL-selected tau exact posterior","wrong_common_tau":"predeclared wrong covariance family","independent_bin":"K=I static-time posterior","ar_lgssm_rts":diagnostics["methods"]["ar_lgssm_rts"],"bounded_diagonal_fa_em":diagnostics["methods"]["bounded_diagonal_fa_em"],"independent_channel_gp_smoothing":diagnostics["methods"]["independent_channel_gp_smoothing"],"two_stage_channel_gp_then_pca":"independent-channel smoothing then rank-two PCA","two_stage_channel_gp_then_fa":diagnostics["methods"]["two_stage_channel_gp_then_fa"],"static_pca":"pooled-bin static PCA with scoring-only alignment","mean_diagonal_gaussian":diagnostics["methods"]["mean_diagonal_gaussian"]},"stress_registry":{"changepoint":"step oversmoothing residual diagnostic","correlated_residuals":"diagonal residual model misspecified","poisson_sqrt":"approximate misspecified Gaussian score; no coverage acceptance","equal_kernels":"joint subspace and scoring-only Procrustes; no coordinate tau identity","alignment_jitter":"joint nominal/shifted GP sampling without clipping or interpolation","informative_missingness":"warning-only skipped-mask inference"},"coverage_provenance":predictive_provenance,"seed_registry":{"policy":"PCG64 SeedSequence(master_seed, stage, split, trial, latent_dimension)","latent_draw":10,"observation_noise":11,"fa_starts":30,"changepoint_noise":90,"correlated_residual_noise":91,"poisson":62,"jitter_shift":63,"jitter_noise":64,"jitter_union_draw":65,"test_observation_poison":80,"latent_poison":81,"spec_C_poison":82,"spec_R_poison":83,"kernel_poison":84,"truth_poison":85,"cross_channel_perturbation":86},"trial_registry":diagnostics["trial_ids"],"grid_score_records":grid,"grid_next_candidate_total_nll_gap":gap,"train_alignment_registry":alignment_registry,"equal_kernel_registry":equal_registry,"ar_registry":{"rho":ar_parameters(TRUE_TAU)[0].tolist(),"q":ar_parameters(TRUE_TAU)[1].tolist(),"initial_mean":[0,0],"initial_covariance":np.eye(Q).tolist(),"adjacent_cross_covariance_convention":"Cov(z_t,z_{t+1}|y) blocks; recursive RTS compared to direct AR batch"},"stress_parameters":{"changepoint":{"step_index":T//2,"step_size_latent_1":2.0,"mask_start":9,"mask_end_exclusive":16},"correlated_residuals":{"correlation_decay":0.28},"poisson_sqrt":{"transform":"sqrt(count+0.25)"},"alignment_jitter":{"integer_shift_range":[-2,2],"joint_unique_union_sampling":True},"informative_missingness":{"threshold_abs_y":1.25}},"oracle_registry_pointers":{"covariances":"diagnostics.covariance_registry","posterior_forms":"diagnostics.posterior_form_oracles","mask_perturbation":"diagnostics.mask_perturbation","poison":"diagnostics.oracles","resets":"diagnostics.per_trial_reset_max_abs"},"applicability":{"equal_kernels":"joint 2D subspace only","informative_missingness":"no coverage acceptance","independent_channel_hidden":"cannot use other channels; prior offset behavior","ar_lgssm_rts":"recursive results used; batch covariance is oracle only"},"tolerances":TOL,"artifact_files":["manifest.json",*ARTIFACT_NAMES],"files_sha256":hashes,"commands":{"generate":f"{sys.executable} toy-models/covariance-prior-trajectories.py --generate","verify":f"{sys.executable} toy-models/covariance-prior-trajectories.py --verify"}}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if not quiet:print(json.dumps({"artifact_root":str(root),"metrics_rows":len(rows),"selected_tau":selected,"selected_fa_start":fa["start_id"],"selected_two_stage_fa_start":sfa["start_id"]},indent=2))
    return diagnostics

def compare_roots(a,b):return[n for n in ("manifest.json",*ARTIFACT_NAMES) if (a/n).read_bytes()!=(b/n).read_bytes()]
def verify(root,recompute=True):
    errors=[]
    for name in ("manifest.json",*ARTIFACT_NAMES):
        if not(root/name).is_file():errors.append(f"missing {name}")
    if errors:raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    manifest=json.loads((root/"manifest.json").read_text());diag=json.loads((root/"diagnostics.json").read_text());oracle=diag["oracles"]
    for name,expected in manifest["files_sha256"].items():
        if sha(root/name)!=expected:errors.append(f"hash mismatch {name}")
    for key in ("kernel","fa_em","methods","stress_registry","coverage_provenance","applicability","masks","seed_registry","trial_registry","grid_score_records","train_alignment_registry","equal_kernel_registry","ar_registry","oracle_registry_pointers"):
        if key not in manifest:errors.append(f"manifest missing {key}")
    if manifest.get("kernel",{}).get("diagonal")!=1.0 or manifest.get("kernel",{}).get("nugget")!=NUGGET:errors.append("manifest kernel convention mismatch")
    if oracle["kernel_diagonal_max_abs_from_one"]>TOL["algebra"] or abs(oracle["kernel_signal_diagonal"]-(1-NUGGET))>TOL["algebra"] or abs(oracle["kernel_nugget_diagonal"]-NUGGET)>TOL["algebra"]:errors.append("kernel signal/nugget/diagonal oracle failed")
    if oracle["kernel_min_eigenvalue"]<=0 or oracle["stacked_covariance_min_eigenvalue"]<=-TOL["psd"] or oracle["stacked_covariance_symmetry_max_abs"]>TOL["algebra"]:errors.append("kernel/stacked covariance PSD failed")
    for label,record in diag["posterior_form_oracles"].items():
        if max(record.values())>TOL["algebra"]:errors.append(f"posterior form oracle failed {label}")
    if max(diag["independent_bin_per_time_equivalence"].values())>TOL["algebra"]:errors.append("independent-bin per-time equivalence failed")
    for label,record in oracle["ar_recursive_batch"].items():
        if max(record.values())>TOL["algebra"]:errors.append(f"recursive RTS/batch oracle failed {label}")
    if max(oracle["hidden_predictive_direct_equivalence"].values())>TOL["algebra"]:errors.append("hidden predictive direct-equivalence failed")
    if any(record["minimum_masked_minus_complete_variance"]<-TOL["algebra"] for record in diag["posterior_nesting"]):errors.append("posterior variance nesting failed")
    if not oracle["mask_targets_excluded_by_perturbation"] or oracle["trial_block_covariance_cross_trial_max_abs"]!=0 or oracle["independent_regeneration_max_abs"]!=0 or diag["per_trial_reset_max_abs"]>TOL["algebra"]:errors.append("mask/reset/regeneration oracle failed")
    perturb=diag["mask_perturbation"]
    if perturb["hidden_value_change_norm"]<=0 or perturb["observed_value_change_norm"]<=0 or max(perturb[k] for k in ("hidden_posterior_mean_max_abs","hidden_posterior_cov_max_abs","hidden_prediction_mean_max_abs","hidden_prediction_cov_max_abs"))>TOL["poison"] or perturb["observed_perturb_posterior_mean_change"]<=0 or perturb["observed_perturb_prediction_mean_change"]<=0:errors.append("hidden/observed perturbation oracle failed")
    if diag["generator_oracles"]["signal_max_abs"]>TOL["algebra"] or diag["generator_oracles"]["time_major_kron_stacking_max_abs"]>TOL["algebra"]:errors.append("generator/stacking algebra failed")
    if not diag["trial_ids"]["split_disjoint"]:errors.append("trial ID splits not disjoint")
    for name,record in diag["covariance_registry"].items():
        if not record["all_finite"] or record["symmetry_max_abs"]>TOL["algebra"] or record["minimum_eigenvalue"]<-TOL["psd"]:errors.append(f"covariance registry failed {name}")

    for scope in ("primary","two_stage"):
        registry=diag["fa_em"][scope];starts=registry["starts"]
        if len(starts)!=6 or {x["start_id"] for x in starts}!=set(range(6)):errors.append(f"FA start registry incomplete {scope}")
        for record in starts:
            if record["floor"]!=1e-6 or record["iterations"]>300 or record["maximum_nll_increase"]>TOL["algebra"] or len(record["history"])!=record["iterations"]:errors.append(f"FA convergence/history failed {scope}:{record['start_id']}")
        best=min(starts,key=lambda r:(r["train_nll_per_entry"],r["validation_nll_per_entry"],r["start_id"]))
        if registry["selected_start"]!=best["start_id"]:errors.append(f"FA selection mismatch {scope}")
    if manifest["fa_em"]!={"starts":6,"floor":1e-6,"max_iterations":300,"relative_train_nll_tolerance":1e-9,"selection":"lowest train NLL, then lower validation NLL, then start ID","records":"diagnostics.fa_em"}:errors.append("FA manifest mismatch")

    best=min(diag["selection"]["grid"],key=lambda x:(x["selection_score"],x["tau_1"],x["tau_2"]))
    if diag["selection"]["selected"]!=[best["tau_1"],best["tau_2"]] or not oracle["selected_grid_exact_recompute"]:errors.append("selected tau grid mismatch")
    ranked=sorted(diag["selection"]["grid"],key=lambda x:(x["combined_total_nll"],x["tau_1"],x["tau_2"]))
    if any(record["rank"]!=i+1 for i,record in enumerate(ranked)) or abs(diag["selection"]["next_candidate_total_nll_gap"]-(ranked[1]["combined_total_nll"]-ranked[0]["combined_total_nll"]))>TOL["algebra"]:errors.append("grid ranks/gap mismatch")
    for name,record in diag["train_only_alignments"].items():
        rotation=np.asarray(record["rotation"])
        if record["fit_split"]!="train" or record["test_truth_used"] or np.max(np.abs(rotation.T@rotation-np.eye(Q)))>TOL["algebra"] or record["training_objective_mse"]<0:errors.append(f"train-only alignment invalid {name}")
    equal_ids=diag["equal_kernel"]["split_trial_ids"]
    if not (set(equal_ids["train"]).isdisjoint(equal_ids["validation"]) and set(equal_ids["train"]).isdisjoint(equal_ids["test"]) and set(equal_ids["validation"]).isdisjoint(equal_ids["test"])) or diag["equal_kernel"]["coordinate_tau_identity_emitted"]:errors.append("equal-kernel registry invalid")
    poison=oracle["poison_registry"]
    if poison["test_observation_poison"]["changed_norm"]<=0 or poison["test_observation_poison"]["complete_fit_max_abs"]>TOL["poison"] or poison["test_observation_poison"]["fixed_original_output_max_abs"]>TOL["poison"]:errors.append("test-observation poison failed")
    if poison["all_latent_truth_poison"]["changed_norm"]<=0 or poison["all_latent_truth_poison"]["observation_only_fit_max_abs"]>TOL["poison"] or poison["all_latent_truth_poison"]["fixed_original_output_max_abs"]>TOL["poison"] or poison["all_latent_truth_poison"]["training_alignment_expected_change"]<=0:errors.append("all-latent truth poison failed")
    if poison["test_latent_truth_poison"]["changed_norm"]<=0 or poison["test_latent_truth_poison"]["training_alignment_max_abs"]>TOL["poison"]:errors.append("test-truth alignment poison failed")
    jitter=diag["alignment_jitter"]
    if not jitter["no_wrap"] or not jitter["zero_shift_exact_identity"] or not any(abs(x)>0 for x in jitter["shifts"]) or not any(x>0 for x in jitter["nominal_observed_latent_max_abs_difference"]):errors.append("alignment jitter construction failed")
    hidden_channel=diag["independent_channel_hidden_channel"]
    if hidden_channel["prior_offset_max_abs"]>TOL["algebra"] or hidden_channel["mean_max_abs"]>TOL["algebra"] or hidden_channel["variance_max_abs"]>TOL["algebra"] or hidden_channel["perturbation_norm"]<=0 or hidden_channel["uses_other_channels"]:errors.append("independent-channel hidden behavior failed")
    if not diag["residual_diagnostics"]["correlated_residuals"]["model_misspecified_correlated_residual"]:errors.append("correlated residual misspecification label missing")
    if not diag["residual_diagnostics"]["changepoint"]["changepoint_oversmoothing"]:errors.append("changepoint oversmoothing diagnostics missing")
    if not diag["informative_missingness"]["warning_only"] or diag["informative_missingness"]["coverage_acceptance"] or diag["informative_missingness"]["missing_fraction"]<=0 or diag["informative_missingness"]["mean_abs_inside_missing"]<=diag["informative_missingness"]["mean_abs_outside_missing"]:errors.append("informative missingness warning/magnitude failed")

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
        if key in seen:errors.append(f"duplicate {key}")
        seen.add(key)
    required={"marginal_nll_per_entry","latent_rmse","latent_95_coverage","mean_posterior_variance","posterior_roughness","predictive_nll_per_entry","predictive_rmse","predictive_95_coverage","mean_predictive_variance","hidden_target_count","loading_subspace_angle_degrees","loading_subspace_rms_sine_error","loading_subspace_projector_frobenius_error","selected_tau","absolute_tau_error","relative_tau_error","train_total_nll","validation_total_nll","combined_total_nll","rank","residual_max_offdiagonal_covariance","residual_lag1_autocorrelation"}
    if not required.issubset({row["metric"] for row in rows}):errors.append("required metrics missing")
    for method in ("oracle_gpfa_true_tau","selected_tau_gpfa","wrong_common_tau","ar_lgssm_rts","bounded_diagonal_fa_em","mean_diagonal_gaussian","independent_channel_gp_smoothing"):
        for region in ("masked_channel","masked_block"):
            if not all(any(row["method"]==method and row["region"]==region and row["metric"]==metric for row in rows) for metric in ("predictive_nll_per_entry","predictive_rmse","predictive_95_coverage","mean_predictive_variance","hidden_target_count")):errors.append(f"masked metrics missing {method}:{region}")
    for method in ("oracle_gpfa_true_tau","selected_tau_gpfa","wrong_common_tau","independent_bin","ar_lgssm_rts"):
        for region in ("masked_time_region","unmasked_time_region"):
            if not all(any(row["method"]==method and row["region"]==region and row["metric"]==metric for row in rows) for metric in ("latent_rmse","latent_95_coverage")):errors.append(f"latent region metric missing {method}:{region}")
    for method in ("two_stage_channel_gp_then_pca","two_stage_channel_gp_then_fa"):
        for region in ("masked_channel","masked_block"):
            if not any(row["method"]==method and row["region"]==region and row["metric"]=="predictive_rmse" and row["target"]=="deterministic_hidden_raw_reconstruction" for row in rows):errors.append(f"two-stage deterministic hidden RMSE missing {method}:{region}")
            if any(row["method"]==method and row["region"]==region and row["metric"] in ("predictive_nll_per_entry","predictive_95_coverage") for row in rows):errors.append(f"incoherent two-stage probabilistic metric present {method}:{region}")
    if any(row["condition"]=="equal_kernels" and "tau" in row["metric"] for row in rows):errors.append("equal-kernel coordinate tau identity metric present")
    if not any(row["condition"]=="equal_kernels" and row["method"]=="equal_kernel_static_pca" and row["metric"]=="latent_rmse" for row in rows):errors.append("equal-kernel Procrustes path score missing")
    if any(row["condition"] in ("informative_missingness","poisson_sqrt") and "coverage" in row["metric"] for row in rows):errors.append("warning/approximation coverage acceptance row present")
    if not any(row["method"]=="two_stage_channel_gp_then_fa" and row["metric"]=="loading_subspace_angle_degrees" for row in rows):errors.append("two-stage FA metrics missing")
    for candidate in diag["selection"]["grid"]:
        label=f"tau_{candidate['tau_1']}_{candidate['tau_2']}"
        for metric in ("train_total_nll","validation_total_nll","combined_total_nll","rank"):
            if not any(row["method"]=="selected_tau_gpfa" and row["target"]==label and row["metric"]==metric for row in rows):errors.append(f"grid metric missing {label}:{metric}")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="ldg-gpfa-") as tmp:
            regenerated=Path(tmp)/"artifact";build(regenerated,True);errors += [f"byte mismatch {name}" for name in compare_roots(root,regenerated)]
    if errors:raise SystemExit("Verification failed:\n- "+"\n- ".join(errors))
    print(json.dumps({"verified":True,"artifact_root":str(root),"deterministic_recomputation":recompute},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument("--generate",action="store_true");p.add_argument("--verify",action="store_true");p.add_argument("--no-recompute",action="store_true");p.add_argument("--artifact-root",type=Path,default=DEFAULT_ROOT);a=p.parse_args();g,v=a.generate,a.verify
    if not g and not v:g=v=True
    if g:build(a.artifact_root)
    if v:verify(a.artifact_root,not a.no_recompute)
if __name__=="__main__":main()
