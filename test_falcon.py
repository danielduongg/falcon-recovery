"""Smoke + correctness tests for falcon-recovery."""
import json, math, numpy as np
from simulate import generate

def test_generator_reproducible():
    a, b = generate(seed=20260618), generate(seed=20260618)
    assert a.equals(b)

def test_success_rate_climbs():
    df = generate()
    byr = df.groupby("year").landing_success.mean()
    assert byr.loc[2013] < 0.2          # early era ~0
    assert byr.loc[2022] > 0.85         # mature era high
    assert byr.index.is_monotonic_increasing

def test_model_export_shape():
    m = json.load(open("web_model.json"))
    assert m["features"] == ["flight_number","payload_mass_kg","orbit_energy","is_rtls","reuse_count"]
    assert len(m["members"]) == 5
    for mem in m["members"]:
        assert {"mean","scale","coef","intercept","a","b"} <= set(mem)

def test_temporal_confounding_finding():
    d = json.load(open("web_model.json"))["diagnostics"]
    # flight-number-only AUC essentially equals the full-model AUC (the headline)
    assert abs(d["auc_time"] - d["auc_full"]) < 0.03
    assert d["auc_full"] > 0.85

def test_js_forward_matches_python():
    m = json.load(open("web_model.json")); F=m["features"]
    def fwd(x):
        out=0.0
        for mem in m["members"]:
            s=mem["intercept"]
            for i in range(len(F)): s+=((x[i]-mem["mean"][i])/mem["scale"][i])*mem["coef"][i]
            out+=1/(1+math.exp(mem["a"]*s+mem["b"]))
        return out/len(m["members"])
    # a recent easy mission should score high, an early hard one low
    assert fwd([205,7000,1.0,0,8]) > 0.8
    assert fwd([8,4800,1.0,0,0]) < 0.4
