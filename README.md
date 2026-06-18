# 🚀 falcon-recovery — Will the booster land?

A calibrated classifier for **Falcon 9 first-stage recovery**, with a live in-browser demo and a deliberately honest finding about *why* it works.

### ▶️ [Live demo](https://danielduongg.github.io/falcon-recovery/)

Set a mission — year, orbit, recovery mode (Landing Zone vs droneship), payload mass, booster reuse — and read the landing probability, plus the **cost-per-kg-to-orbit** as reuse amortizes the booster.

## The honest finding: the model is mostly a clock 🕐

Landing success climbed from **0% (2013–14) to ~98% (2019+)** as the program matured. So a model given only the **flight number** scores essentially the same as the full model:

| Model | ROC-AUC |
|---|---|
| Full (year + mission features) | **0.916** |
| **Flight number only** (a clock) | **0.916** |
| Mission features only (no time) | 0.864 |

Knowing *when* a flight happened predicts recovery as well as anything about the mission. That's **temporal confounding / time leakage** — a failure mode that quietly inflates "accuracy" across real-world ML whenever a maturing process is encoded in a date-like feature. The demo lets you feel it: hold a hard GTO droneship mission fixed and drag the year from 2022 back to 2014 — the probability collapses.

## Method

- Historically-grounded data generator calibrated to public per-year landing rates (`simulate.py`); columns mirror the SpaceX API (`v4` launches + cores + payloads) so you can swap in live data.
- Calibrated logistic regression (Platt scaling, 5-fold) — `train.py`.
- The trained model is exported to JSON and re-implemented in ~10 lines of JavaScript; the browser scores match scikit-learn to machine precision (verified in `train.py`).

## Run it

```bash
pip install -r requirements.txt
python simulate.py      # build the dataset
python train.py         # train, evaluate, export web_model.json
python build_demo.py    # bake the model into index.html
```

> Educational project on a grounded synthetic dataset; point it at the live SpaceX API for real launches. Not affiliated with SpaceX.
