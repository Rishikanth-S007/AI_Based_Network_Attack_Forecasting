"""
The single seam between the ML backend and ANY frontend. Nothing here
knows or cares whether it's called from Streamlit, a CLI, or a FastAPI
endpoint serving a JS app built elsewhere (Figma Make, etc.) -- every
function takes plain, real-scale (NOT pre-normalized) data in, and
returns plain, JSON-serializable dicts out.

load_model() now loads the REAL checkpoint trained on CTU-13 (Scenarios
1/2/9 -> train, Scenario 10 -> held-out test, F1 ~0.43). The checkpoint
carries everything needed to use the model correctly, not just weights:
the exact feature normalization stats it was trained with, the tuned
alert threshold, and the real feature schema (11 features, including
has_packet_coverage) -- so any caller can hand this raw, real-world
numbers and get back a correct result without having to replicate our
preprocessing.

Two entry points:
  forecast(model, raw_context_window, k_steps) -> risk/stage/response
      per imagined future step
  explain(model, raw_window_features)          -> feature attribution +
      one plain-language sentence for a single window
"""
import torch
import numpy as np
from captum.attr import IntegratedGradients

from model import WorldModel
from mitre_mapping import ID_TO_STAGE
from response_mapping import STAGE_TO_RESPONSE


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WorldModel(n_features=ckpt["n_features"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # attach everything needed to use this model correctly -- normalization
    # stats, the tuned threshold, and real feature names -- so callers never
    # need to know these details or risk getting them wrong
    model.feature_mean = ckpt["feature_mean"]
    model.feature_std = ckpt["feature_std"]
    model.threshold = ckpt["threshold"]
    model.feature_cols = ckpt["feature_cols"]
    model.seq_len = ckpt["seq_len"]
    print(f"loaded checkpoint: {ckpt['trained_on']}")
    print(f"  test F1 {ckpt['test_f1']:.4f} vs baseline {ckpt['baseline_f1']:.4f}, "
          f"alert threshold {ckpt['threshold']:.3f}")
    return model


def _normalize(model, raw_windows):
    """raw_windows: array-like [T, n_features] in real units. Applies the
    exact normalization this model was trained with."""
    arr = np.array(raw_windows, dtype=np.float32)
    mean = model.feature_mean.reshape(-1)
    std = model.feature_std.reshape(-1)
    return (arr - mean) / std


def forecast(model, raw_context_window, k_steps=10):
    """
    raw_context_window: list of lists, shape [T_context][n_features], in
    REAL units (e.g. actual flow counts, not pre-normalized) -- exactly
    what a dashboard would have on hand from live traffic.

    Returns a JSON-ready dict: a per-step forecast, each with risk,
    predicted MITRE stage, whether it crosses this model's tuned alert
    threshold, and the advisory response for that stage.
    """
    x = torch.tensor(np.expand_dims(_normalize(model, raw_context_window), 0), dtype=torch.float32)
    with torch.no_grad():
        teacher_forced_latents, _ = model.teacher_forced_rollout(x)
        teacher_forced_risk_logits, _ = model.decoder(teacher_forced_latents)
        observed_risks = torch.sigmoid(teacher_forced_risk_logits[0]).tolist()
        risks, stage_logits, _ = model.imagine_forward(x, k_steps=k_steps)
    risks = risks[0].tolist()
    stage_ids = stage_logits[0].argmax(dim=-1).tolist()

    history = [
        {
            "step": index - len(observed_risks),
            "risk": round(risk, 4),
        }
        for index, risk in enumerate(observed_risks)
    ]
    steps = []
    for i, (risk, stage_id) in enumerate(zip(risks, stage_ids)):
        stage_name = ID_TO_STAGE[stage_id]
        steps.append({
            "step": i + 1,
            "risk": round(risk, 4),
            "alert": risk > model.threshold,
            "predicted_stage": stage_name,
            "recommended_response": STAGE_TO_RESPONSE.get(stage_name, []),
        })

    peak_idx = risks.index(max(risks))
    return {
        "k_steps": k_steps,
        "history": history,
        "current_risk": history[-1]["risk"],
        "forecast": steps,
        "peak_risk": round(risks[peak_idx], 4),
        "peak_stage": ID_TO_STAGE[stage_ids[peak_idx]],
        "threshold_used": round(model.threshold, 4),
    }


def explain(model, raw_window_features, top_k=3):
    """
    raw_window_features: list of length n_features, ONE observed window,
    in real units.

    Returns per-feature attribution (via Captum Integrated Gradients on
    encoder -> risk head, a same-step model, not through the full
    rollout -- see the reasoning earlier in this project) plus a plain-
    language sentence built from the top-k contributing features.
    """
    def encode_and_score(x):
        z = model.encoder(x)
        risk_logit, _ = model.decoder(z)
        return risk_logit.unsqueeze(-1)

    normed = _normalize(model, [raw_window_features])
    x = torch.tensor(normed, dtype=torch.float32, requires_grad=True)
    baseline = torch.zeros_like(x)  # zero = the dataset mean, post-normalization

    ig = IntegratedGradients(encode_and_score)
    attributions = ig.attribute(x, baseline, target=0).squeeze(0).tolist()

    scored = sorted(zip(model.feature_cols, attributions), key=lambda p: abs(p[1]), reverse=True)
    total_abs = sum(abs(attribution) for _, attribution in scored)
    top = scored[:top_k]

    with torch.no_grad():
        risk_logit, stage_logits = model.decoder(model.encoder(x))
        stage_name = ID_TO_STAGE[stage_logits.argmax(dim=-1).item()]

    normalized_all = [
        {
            "feature": name,
            "attribution": round(attribution, 4),
            "impact_percent": round(
                (abs(attribution) / total_abs) * 100 if total_abs else 0.0,
                2,
            ),
        }
        for name, attribution in scored
    ]
    normalized_top = normalized_all[:top_k]
    positive_top = [item for item in normalized_all if item["attribution"] > 0]
    summary_features = (positive_top or normalized_all)[:2]
    feature_list = " and ".join(item["feature"].replace("_", " ") for item in summary_features)
    sentence = (
        f"Risk score driven primarily by {feature_list}, "
        f"consistent with {stage_name.replace('_', ' ')} behavior."
    )

    return {
        "predicted_stage": stage_name,
        "top_features": normalized_top,
        "explanation": sentence,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ctu13_pipeline import load_ctu13_csv, window_by_host, FEATURE_COLS
    from ctu13_packet_features import merge_packet_features
    import pandas as pd

    print("loading the real trained checkpoint...\n")
    model = load_model("ctu13_world_model.pt")
    SEQ_LEN = model.seq_len

    print("\npulling one REAL held-out (Aug 18) infected-host window sequence "
          "to demonstrate on, in raw (un-normalized) units...")
    df = load_ctu13_csv("ctu13_multiscenario_trimmed.csv")
    windowed = window_by_host(df)
    packet_dfs = [pd.read_csv(f"pf_scenario{s}.csv") for s in [1, 2, 9, 10]]
    for p in packet_dfs:
        p["window"] = pd.to_datetime(p["window"])
    packet_df = pd.concat(packet_dfs, ignore_index=True)
    windowed = merge_packet_features(windowed, packet_df)

    infected_hosts = [
        "147.32.84.165", "147.32.84.191", "147.32.84.192", "147.32.84.193", "147.32.84.204",
        "147.32.84.205", "147.32.84.206", "147.32.84.207", "147.32.84.208", "147.32.84.209",
    ]

    def get_contexts(host):
        """All 3-window contexts from this host's Aug 18 sequence chunks
        (matching how build_sequences actually chunks it -- non-
        overlapping 15-window blocks), not just the very first one."""
        rows = windowed[(windowed["src_addr"] == host) & (windowed["date"].astype(str) == "2011-08-18")]
        rows = rows.sort_values("window")
        contexts = []
        for start in range(0, len(rows) - SEQ_LEN + 1, SEQ_LEN):
            chunk = rows.iloc[start:start + SEQ_LEN]
            contexts.append(chunk[FEATURE_COLS].fillna(0).values[:3].tolist())
        return contexts

    print("scanning held-out Aug 18 infected hosts for one detection and one miss, "
          "to show both honestly rather than cherry-pick...\n")

    detection_example, miss_example = None, None
    for host in infected_hosts:
        for ctx in get_contexts(host):
            result = forecast(model, ctx, k_steps=8)
            hit = any(step["alert"] for step in result["forecast"])
            if hit and detection_example is None:
                detection_example = (host, ctx, result)
            if not hit and miss_example is None:
                miss_example = (host, ctx, result)
        if detection_example and miss_example:
            break

    for label, example in [("DETECTION example", detection_example), ("MISS example", miss_example)]:
        if example is None:
            print(f"--- {label}: none found in this scan ---\n")
            continue
        host, ctx, result = example
        print(f"--- {label}: {host}, first 3 real windows of Aug 18 ---")
        for step in result["forecast"][:5]:
            print(" ", step)
        print(f"  peak risk {result['peak_risk']} at stage '{result['peak_stage']}', "
              f"threshold {result['threshold_used']}")
        print(f"  explain() on window 3: {explain(model, ctx[-1])}")
        print()
