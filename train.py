import json, numpy as np, pandas as pd
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

ORBIT_ENERGY={"LEO":0.0,"ISS":0.05,"SSO":0.35,"Polar":0.45,"GTO":1.0}
df=pd.read_csv("data_launches.csv")
df["orbit_energy"]=df.orbit.map(ORBIT_ENERGY)
df["is_rtls"]=(df.landing_type=="RTLS").astype(int)
FEATURES=["flight_number","payload_mass_kg","orbit_energy","is_rtls","reuse_count"]
X,y=df[FEATURES],df.landing_success

# ---- honest finding: temporal confounding ----
rng=np.random.RandomState(7); idx=rng.permutation(len(df))
cut=int(.75*len(df)); tr,te=idx[:cut],idx[cut:]
def auc_of(cols):
    m=CalibratedClassifierCV(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),method="sigmoid",cv=5)
    m.fit(X.iloc[tr][cols],y.iloc[tr]); return roc_auc_score(y.iloc[te],m.predict_proba(X.iloc[te][cols])[:,1])
auc_full=auc_of(FEATURES); auc_time=auc_of(["flight_number"]); auc_mission=auc_of(["payload_mass_kg","orbit_energy","is_rtls","reuse_count"])
print(f"AUC full={auc_full:.3f}  time-only(flight#)={auc_time:.3f}  mission-only={auc_mission:.3f}")

# ---- deploy calibrated model on full features ----
cal=CalibratedClassifierCV(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),method="sigmoid",cv=5).fit(X,y)
members=[]
for cc in cal.calibrated_classifiers_:
    pipe=cc.estimator; sc=pipe.named_steps["standardscaler"]; clf=pipe.named_steps["logisticregression"]
    calib=getattr(cc,"calibrators",None) or getattr(cc,"calibrators_"); cobj=calib[0]
    members.append(dict(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=clf.coef_[0].tolist(),
                        intercept=float(clf.intercept_[0]),a=float(cobj.a_),b=float(cobj.b_)))
def fwd(Xdf):
    Xv=Xdf[FEATURES].to_numpy(float);out=np.zeros(len(Xv))
    for m in members:
        z=(Xv-m["mean"])/m["scale"];d=z@np.array(m["coef"])+m["intercept"];out+=expit(-(m["a"]*d+m["b"]))
    return out/len(members)
print("max|mine-skl|=",f"{np.max(np.abs(fwd(X)-cal.predict_proba(X)[:,1])):.2e}")

# ---- extra diagnostics for the demo + figures (held-out split) ----
p_te=fwd(X.iloc[te]); y_te=y.iloc[te].values
fpr,tpr,_=roc_curve(y_te,p_te)
roc_pts=[[round(float(a),3),round(float(b),3)] for a,b in zip(fpr[::max(1,len(fpr)//40)],tpr[::max(1,len(tpr)//40)])]
frac,meanp=calibration_curve(y_te,p_te,n_bins=8,strategy="quantile")
calib=[[round(float(m),3),round(float(f),3)] for m,f in zip(meanp,frac)]
cm=confusion_matrix(y_te,(p_te>=0.5).astype(int)).tolist()
# mean standardized logistic coef across members (sign/magnitude of each feature)
import numpy as _np
mean_coef=_np.mean([m["coef"] for m in members],axis=0).tolist()
EXTRAS=dict(roc=roc_pts,calibration=calib,confusion=cm,mean_coef=[round(c,3) for c in mean_coef],
            auc_full=round(float(auc_full),3),auc_time=round(float(auc_time),3),auc_mission=round(float(auc_mission),3))
model=dict(features=FEATURES,orbit_energy=ORBIT_ENERGY,members=members,
  metrics=dict(n=len(df),overall_success=round(float(y.mean()),3),
               auc_full=round(float(auc_full),3),auc_time=round(float(auc_time),3),auc_mission=round(float(auc_mission),3)),
  year_rates={int(k):round(float(v),2) for k,v in df.groupby("year").landing_success.mean().items()},
  diagnostics=EXTRAS,
  econ=dict(booster_cost_m=37.0,other_cost_m=30.0,refurb_cost_m=1.2,expendable_extra_m=0.0,payload_kg=15600))
json.dump(model,open("web_model.json","w"))
print("wrote web_model.json",round(len(json.dumps(model))/1024,1),"KB")
for ex,fv in [("2014 GTO droneship, fresh",[8,4800,1.0,0,0]),("2022 LEO RTLS, reused x5",[200,9000,0.0,1,5]),("2022 GTO droneship heavy",[205,7000,1.0,0,8])]:
    p=fwd(pd.DataFrame([dict(zip(FEATURES,fv))]))[0]; print(f"  {ex}: {p*100:.0f}%")
