# Project folder structure

```
your-project/
├── app.py                          <- YOUR Streamlit app (add the app_integration_example.py block to its top)
├── app_integration_example.py      <- copy the top block of this into your real app.py, then delete this file
├── requirements.txt
│
├── backend/
│   │
│   │   ── RUNTIME-CRITICAL (your app needs these to make predictions) ──
│   ├── inference.py                <- the seam: load_model(), forecast(), explain()
│   ├── model.py                    <- WorldModel architecture (encoder/transition/decoder)
│   ├── mitre_mapping.py            <- MITRE stage id <-> name lookup
│   ├── response_mapping.py         <- MITRE stage -> recommended response lookup
│   ├── ctu13_world_model.pt        <- the trained checkpoint (weights + normalization + threshold)
│   │
│   │   ── TRAINING-ONLY (not needed to run the app, only to retrain) ──
│   ├── ctu13_pipeline.py           <- windows raw CTU-13 flow data into model input
│   ├── ctu13_packet_features.py    <- extracts real TTL/retransmit features from PCAP
│   ├── train_real.py               <- trains on real data, produces ctu13_world_model.pt
│   ├── baseline.py                 <- the logistic regression comparison
│   ├── data.py                     <- synthetic data generator (early-stage smoke testing)
│   ├── train.py                    <- trains on synthetic data (smoke test only)
│   └── feature_pipeline.py         <- CIC-IDS2018 loader (unused -- blocked on missing IP
│                                       columns in that dataset, documented in-project)
│
└── data/                           <- TRAINING-ONLY, not needed to run the app
    ├── ctu13_multiscenario_trimmed.csv   (45.7 MB -- flow-level, all 4 scenarios)
    ├── pf_scenario1.csv                  (real packet features, Aug 10)
    ├── pf_scenario2.csv                  (real packet features, Aug 11)
    ├── pf_scenario9.csv                  (real packet features, Aug 17)
    └── pf_scenario10.csv                 (real packet features, Aug 18 -- held-out test day)
```

## What to actually do

1. Drop the `backend/` folder into your project root, next to your existing `app.py`.
2. Open `app_integration_example.py`, copy the top block (the `sys.path` line
   through `model = get_model()`) into the top of your real `app.py`.
3. Wherever your UI currently has placeholder/mock numbers, call `forecast(model, ...)`
   and `explain(model, ...)` instead — see the bottom of the example file for exact
   input/output shapes.
4. `pip install -r requirements.txt`.
5. `streamlit run app.py`.

You do **not** need to touch anything inside `backend/` — every file there already
works together exactly as tested throughout this project, using plain flat imports.
Don't rename or move files within `backend/` without checking cross-imports first.

## If you never plan to retrain

You can delete `data/` and the training-only files listed above entirely — the
app only needs the five files marked RUNTIME-CRITICAL plus `ctu13_world_model.pt`.
Keeping them in the repo is still worth it for the "Source Code" deliverable
though, since it shows the full pipeline that produced the checkpoint, not just
the checkpoint itself.

## Model input format (for `forecast()`/`explain()`)

Each window is a list of 11 real-valued features, in this exact order
(also available at runtime as `model.feature_cols` after `load_model()`):

```
[flow_count, unique_dst_ports, unique_dst_hosts, syn_ack_ratio,
 iat_mean, iat_var, bytes_out, bytes_in, ttl_var, retransmits,
 has_packet_coverage]
```

Pass **raw, real-unit values** — do not normalize them yourself. `inference.py`
applies the exact normalization the model was trained with internally, using
stats saved inside the checkpoint.

## Real results this checkpoint achieves

- Trained on CTU-13 Scenarios 1/2/9 (Aug 10/11/17), tested on Scenario 10
  (Aug 18) — a full capture day never seen during training.
- World model F1 0.4254 vs. logistic-regression baseline F1 0.2491.
- Under genuine blind rollout (only 3 real windows of context, then forecasting
  12 steps with zero new real data), roughly 7-9 of 30 held-out infected
  sequences are correctly forecast — a real, honestly-measured result, not
  a cherry-picked one. See this project's architecture doc for the full
  methodology and caveats (small-sample variance across seeds, real-data
  provenance corrections, etc.).
