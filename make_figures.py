"""Generate result figures for the README (reads data_launches.csv)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
ORBIT_ENERGY={"LEO":0.0,"ISS":0.05,"SSO":0.35,"Polar":0.45,"GTO":1.0}
plt.rcParams.update({"figure.facecolor":"#0b0f17","axes.facecolor":"#0e1420","savefig.facecolor":"#0b0f17",
  "text.color":"#e8eef9","axes.labelcolor":"#cdd9ef","xtick.color":"#8aa0bf","ytick.color":"#8aa0bf",
  "axes.edgecolor":"#1f2a3c","font.size":11,"axes.titlecolor":"#e8eef9"})
df=pd.read_csv("data_launches.csv"); df["orbit_energy"]=df.orbit.map(ORBIT_ENERGY); df["is_rtls"]=(df.landing_type=="RTLS").astype(int)
F=["flight_number","payload_mass_kg","orbit_energy","is_rtls","reuse_count"]; X,y=df[F],df.landing_success
rng=np.random.RandomState(7); idx=rng.permutation(len(df)); cut=int(.75*len(df)); tr,te=idx[:cut],idx[cut:]
mk=lambda: CalibratedClassifierCV(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),method="sigmoid",cv=5)
m=mk().fit(X.iloc[tr],y.iloc[tr]); p=m.predict_proba(X.iloc[te])[:,1]; yte=y.iloc[te].values

# 1) success by year
plt.figure(figsize=(7,3.6)); byr=df.groupby("year").landing_success.mean()
plt.bar(byr.index,byr.values,color=["#ff5d6c" if v<.5 else "#ffb454" if v<.8 else "#27d08a" for v in byr.values])
plt.ylim(0,1); plt.ylabel("landing success rate"); plt.title("Falcon 9 booster-landing success climbed 0% → ~98%")
plt.tight_layout(); plt.savefig("results/figures/success_by_year.png",dpi=120); plt.close()

# 2) ROC + calibration
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
fpr,tpr,_=roc_curve(yte,p); ax[0].plot(fpr,tpr,color="#5b8cff",lw=2.5,label=f"AUC={roc_auc_score(yte,p):.3f}")
ax[0].plot([0,1],[0,1],"--",color="#8aa0bf",alpha=.5); ax[0].set_title("ROC (held-out)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
frac,mp=calibration_curve(yte,p,n_bins=8,strategy="quantile"); ax[1].plot(mp,frac,"o-",color="#27d08a")
ax[1].plot([0,1],[0,1],"--",color="#8aa0bf",alpha=.5); ax[1].set_title("Calibration (reliability)"); ax[1].set_xlabel("predicted"); ax[1].set_ylabel("observed")
plt.tight_layout(); plt.savefig("results/figures/roc_calibration.png",dpi=120); plt.close()

# 3) AUC comparison: full vs time-only vs mission-only
def auc(cols): mm=mk().fit(X.iloc[tr][cols],y.iloc[tr]); return roc_auc_score(yte,mm.predict_proba(X.iloc[te][cols])[:,1])
vals=[auc(F),auc(["flight_number"]),auc(["payload_mass_kg","orbit_energy","is_rtls","reuse_count"])]
plt.figure(figsize=(7,3.6)); bars=plt.bar(["full\n(year+mission)","flight # only\n(a clock)","mission only\n(no time)"],vals,color=["#5b8cff","#ffb454","#8aa0bf"])
plt.ylim(0.5,1); plt.ylabel("ROC-AUC"); plt.title("The model is mostly a clock: time alone ≈ full model")
for b,v in zip(bars,vals): plt.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.3f}",ha="center",color="#e8eef9")
plt.tight_layout(); plt.savefig("results/figures/auc_comparison.png",dpi=120); plt.close()
print("wrote 3 figures to results/figures/")
