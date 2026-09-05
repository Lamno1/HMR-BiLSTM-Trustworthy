"""E5-A subtype and E5-B patient-conditional geometry forensic (CPU-only)."""
from __future__ import annotations

import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.signal as sig
import torch
import wfdb
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmr_multipath import HMRMultiPath
from train_inter_patient import CONFIG
from validation.preprocess_aami_rr_crossdb import AAMI, highpass_zero_phase

OUT = ROOT / "results" / "forensic"
POOL = np.load(ROOT / "data" / "processed" / "lopo" / "pool.npz")
META = json.loads((ROOT / "data" / "processed" / "lopo" / "records.json").read_text())["records"]
SUBS = ("A", "a", "J", "S")


def softmax(z):
    z = z.astype(float) - z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)


def artifact_paths(model):
    d, tag = (("mp_lopo", "vrce") if model == "r2" else ("lopo_cv", "flat"))
    return sorted((ROOT/"results"/d).glob(f"fold[0-4]_seed*_{tag}_preds.npz"))


def raw_symbols(record_name):
    db, name = record_name.split("/")
    raw = ROOT/"data"/"raw"/("mitdb" if db == "mitdb" else "incartdb")
    ann = wfdb.rdann(str(raw/name), "atr")
    rec = wfdb.rdrecord(str(raw/name))
    fs = rec.fs
    if db == "mitdb":
        n125 = len(sig.resample_poly(rec.p_signal[:, 0], 125, int(round(fs))))
        ans=[]
        for samp, sym in zip(ann.sample, ann.symbol):
            if sym in AAMI:
                p=int(samp*125/fs)
                if p-90 >= 0 and p+97 <= n125: ans.append((int(samp),sym))
        return ans
    names=[x.strip().upper() for x in rec.sig_name]; li=names.index("II")
    s=sig.resample_poly(highpass_zero_phase(rec.p_signal[:,li].astype(float),fs),125,int(round(fs)))
    ans=[]
    for samp,sym in zip(ann.sample,ann.symbol):
        if sym not in AAMI: continue
        p0=int(samp*125/fs); lo,hi=max(0,p0-15),min(len(s),p0+15)
        if hi-lo < 10: continue
        p=lo+int(np.argmax(np.abs(s[lo:hi]-np.median(s[lo:hi]))))
        if p-90 >= 0 and p+97 <= len(s): ans.append((int(samp),sym))
    return ans


def build_symbol_map():
    y, rec = POOL["y"].astype(int), POOL["record"].astype(int)
    symbols=np.full(len(y),"?",dtype="<U1"); samples=np.full(len(y),-1,dtype=int)
    audit=[]
    for rid in np.unique(rec):
        idx=np.flatnonzero(rec==rid); name=META[str(int(rid))]["name"]
        try: pairs=raw_symbols(name)
        except Exception as e:
            audit.append({"record":name,"joined":0,"pool":len(idx),"status":"read_error","error":repr(e)}); continue
        sy=np.array([s for _,s in pairs]); mapped=np.array([AAMI[s] for s in sy])
        ok=len(idx)==len(pairs) and np.array_equal(y[idx],mapped)
        if ok:
            symbols[idx]=sy; samples[idx]=[p for p,_ in pairs]
        audit.append({"record":name,"joined":len(idx) if ok else 0,"pool":len(idx),"raw_filtered":len(pairs),"status":"exact" if ok else "sequence_mismatch"})
    return symbols,samples,audit


def metric(ybin, score, pred_s):
    n=int(ybin.sum())
    return {"support":n,"auc":float(roc_auc_score(ybin,score)) if 0<n<len(ybin) else None,
            "ap":float(average_precision_score(ybin,score)) if n else None,
            "argmax_f1":float(f1_score(ybin,pred_s,zero_division=0))}


def run_e5a(symbols,samples,audit):
    y=POOL["y"].astype(int); rows=[]; composition=[]; sanity=[]
    for model in ("r2","flat"):
        for p in artifact_paths(model):
            d=np.load(p,allow_pickle=True); meta=json.loads(str(d["meta_json"])); idx=np.array(meta["test_indices"])
            probs=softmax(d["te_logits"]); pred=probs.argmax(1); fold=int(meta["fold"]); seed=int(meta["seed"])
            src=POOL["source"][idx].astype(int)
            mit=src==0
            sanity.append({"model":model,"fold":fold,"seed":seed,**metric(y[idx][mit]==1,probs[mit,1],pred[mit]==1)})
            for scope,mask in (("mitbih",mit),("incart",src==1)):
                sy=symbols[idx]
                for sub in SUBS:
                    pos=mask&(sy==sub)
                    if pos.sum()>=20:
                        rows.append({"model":model,"fold":fold,"seed":seed,"scope":scope,"subtype":sub,**metric(pos,probs[:,1],pred==1)})
            if model=="r2" and seed==42:
                sy=symbols[idx]; sm=(y[idx]==1)
                composition.append({"fold":fold,"n_S":int(sm.sum()),"unjoined_S":int((sm&(sy=="?")).sum()),
                                    "symbols":dict(Counter(sy[sm]).most_common()),
                                    "records":dict(Counter(META[str(int(r))]["name"] for r in POOL["record"][idx][sm]).most_common())})
    means=[]
    keys=sorted(set((r["model"],r["fold"],r["scope"],r["subtype"]) for r in rows))
    for key in keys:
        rs=[r for r in rows if (r["model"],r["fold"],r["scope"],r["subtype"])==key]
        means.append({"model":key[0],"fold":key[1],"scope":key[2],"subtype":key[3],"n_seeds":len(rs),
                      "support":rs[0]["support"],**{m:float(np.mean([r[m] for r in rs])) for m in ("auc","ap","argmax_f1")}})
    spreads=[]
    for model in ("r2","flat"):
        for scope in ("mitbih","incart"):
            for sub in SUBS:
                rs=[r for r in means if r["model"]==model and r["scope"]==scope and r["subtype"]==sub]
                if len(rs)>=2: spreads.append({"model":model,"scope":scope,"subtype":sub,"n_folds":len(rs),"fold_auc":{str(r["fold"]):r["auc"] for r in rs},"auc_spread":float(max(r["auc"] for r in rs)-min(r["auc"] for r in rs))})
    e1=[.733,.980,.951,.950,.816]
    r2san=[r for r in sanity if r["model"]=="r2"]
    reproduced=[float(np.mean([r["auc"] for r in r2san if r["fold"]==f])) for f in range(5)]
    report={"experiment":"E5-A","mapping_audit":audit,"mapping_summary":{"pool_beats":len(y),"joined":int((symbols!="?").sum()),"unjoined":int((symbols=="?").sum()),"note_e":"preprocessing maps raw e to AAMI-N; excluded from S subtypes"},
            "per_run":rows,"two_seed_means":means,"cross_fold_spreads":spreads,"composition_seed42":composition,
            "sanity":{"reproduced_r2_mitbih_auc":reproduced,"e1_rounded_reference":e1,"max_abs_vs_e1_rounded":float(np.max(np.abs(np.array(reproduced)-e1)))}}
    (OUT/"E5_subtype.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def load_embeddings(path,d,idx):
    meta=json.loads(str(d["meta_json"])); ck=path.with_name(path.name.replace("_preds.npz","_model.pt"))
    state=torch.load(ck,map_location="cpu",weights_only=False)
    model=HMRMultiPath(hidden_size=CONFIG["hidden_size"],dropout=CONFIG["dropout"],
                       cnn_out_channels=CONFIG["cnn_out_channels"],num_layers=2,
                       use_v1=True,use_rhythm=True,use_grl=False,rr_dim=3)
    model.load_state_dict(state["model_state"]); model.eval()
    x=POOL["X"][idx].astype(np.float32); src=POOL["source"][idx].astype(int)
    for s in np.unique(src):
        mu,sd=meta["x_zscore"][str(int(s))]; x[src==s]=(x[src==s]-mu)/sd
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),512): out.append(model.embed(torch.from_numpy(x[i:i+512]))[0].numpy())
    return np.concatenate(out)


def dist(a,b,kind):
    if kind=="euclidean": return np.linalg.norm(a-b,axis=1)
    return 1-(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)


def geometry_run(rep,y,rec,pred,fold,seed,space):
    beats=[]; centroids={}; records=[]
    for rid in np.unique(rec[y==3]):
        m=rec==rid
        if (m&(y==0)).sum()<2 or (m&(y==2)).sum()<2: continue
        cn=rep[m&(y==0)].mean(0); cv=rep[m&(y==2)].mean(0); cf=rep[m&(y==3)].mean(0)
        centroids[int(rid)]=cf; records.append(int(rid))
        for kind in ("euclidean","cosine"):
            fi=np.flatnonzero(m&(y==3)); dn=dist(rep[fi],np.repeat(cn[None],len(fi),0),kind); dv=dist(rep[fi],np.repeat(cv[None],len(fi),0),kind)
            inter=1-np.abs(dn-dv)/(dn+dv+1e-12)
            for i,j in enumerate(fi): beats.append({"fold":fold,"seed":seed,"space":space,"distance":kind,"record_id":int(rid),"record":META[str(int(rid))]["name"],"pred":int(pred[j]),"d_N":float(dn[i]),"d_V":float(dv[i]),"intermediacy":float(inter[i])})
    patients=[]
    for rid in records:
        others=[v for k,v in centroids.items() if k!=rid]
        iso=float(np.mean(dist(np.repeat(centroids[rid][None],len(others),0),np.stack(others),"cosine"))) if others else None
        m=rec==rid; tf=m&(y==3); pf=m&(pred==3)
        patients.append({"fold":fold,"seed":seed,"space":space,"record":META[str(rid)]["name"],"n_F":int(tf.sum()),"iso_cosine_within_fold":iso,
                         "F_recall":float((tf&(pred==3)).sum()/tf.sum()),"F_precision":float((tf&(pred==3)).sum()/pf.sum()) if pf.sum() else None})
    return beats,patients


def cross_tabs(beats):
    out=[]
    for space in sorted(set(x["space"] for x in beats)):
      for kind in ("euclidean","cosine"):
        rs=[x for x in beats if x["space"]==space and x["distance"]==kind]
        for variable in ("intermediacy","d_N"):
            vals=np.array([x[variable] for x in rs]); cuts=np.quantile(vals,[1/3,2/3]); bins=np.digitize(vals,cuts)
            tab=[]
            for b in range(3):
                q=[r for r,z in zip(rs,bins) if z==b]
                tab.append({"tertile":b+1,"n":len(q),"pred_F":float(np.mean([r["pred"]==3 for r in q])),"pred_N":float(np.mean([r["pred"]==0 for r in q]))})
            out.append({"space":space,"distance":kind,"variable":variable,"cuts":cuts.tolist(),"cells":tab})
    return out


def run_e5b():
    allbeats=[]; patients=[]
    for path in artifact_paths("r2"):
        d=np.load(path,allow_pickle=True); meta=json.loads(str(d["meta_json"])); full_idx=np.array(meta["test_indices"])
        full_y=POOL["y"][full_idx].astype(int); full_rec=POOL["record"][full_idx].astype(int)
        f_records=np.unique(full_rec[full_y==3]); keep=np.isin(full_rec,f_records)
        idx=full_idx[keep]; y=full_y[keep]; rec=full_rec[keep]; pred=d["te_logits"][keep].argmax(1)
        for space,rep in (("logits",d["te_logits"][keep].astype(float)),("embedding",load_embeddings(path,d,idx))):
            b,p=geometry_run(rep,y,rec,pred,int(meta["fold"]),int(meta["seed"]),space); allbeats+=b; patients+=p
    corrs=[]
    for space in ("logits","embedding"):
        ps=[p for p in patients if p["space"]==space and p["iso_cosine_within_fold"] is not None]
        for target in ("F_recall","F_precision"):
            q=[p for p in ps if p[target] is not None]
            rho,pv=spearmanr([p["iso_cosine_within_fold"] for p in q],[p[target] for p in q]) if len(q)>=3 else (np.nan,np.nan)
            corrs.append({"space":space,"target":target,"n_record_seed":len(q),"spearman_rho":float(rho),"pvalue_descriptive":float(pv)})
    report={"experiment":"E5-B","centroid_scope":"outer-test N and V beats from same record; no F labels used for N/V centroids",
            "spaces":{"logits":"aligned semantic 5-D space; primary for iso","embedding":"shared-trunk embedding; cross-tabs valid within model, iso computed only against F records in same fold/model"},
            "warning":"independent fold embedding coordinates are not comparable; all iso values are within-fold. Protocol sign contradiction: iso-high means unlike peers, so failure predicts negative—not positive—correlation with recall.",
            "cross_tabs":cross_tabs(allbeats),"patient_rows":patients,"iso_correlations":corrs,
            "record_cross_tabs":{recname:cross_tabs([x for x in allbeats if x["record"]==recname])
                                 for recname in ("mitdb/208","mitdb/213")},
            "records_208_213":[x for x in allbeats if x["record"] in ("mitdb/208","mitdb/213")]}
    (OUT/"E5_fgeometry.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    symbols,samples,audit=build_symbol_map(); a=run_e5a(symbols,samples,audit); b=run_e5b()
    print("E5-A",a["mapping_summary"],a["sanity"])
    print("E5-B correlations",json.dumps(b["iso_correlations"],indent=2))

if __name__=="__main__": main()
