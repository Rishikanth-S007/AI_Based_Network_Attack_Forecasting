"""
EXAMPLE ONLY -- shows exactly how to wire your existing Streamlit app to
the backend folder. Copy the top block into the top of YOUR actual
app.py; don't run this file itself.

This is the entire integration -- no changes needed to any file inside
backend/, they already work together as-is (every path in this project
has been tested with this exact flat layout).
"""
import sys
import os

# 1. Make the backend folder importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# 2. Import the three functions your UI actually calls
from inference import load_model, forecast, explain

# 3. Load the model ONCE, e.g. with Streamlit's cache so it's not
#    reloaded on every widget interaction
import streamlit as st

@st.cache_resource
def get_model():
    checkpoint_path = os.path.join(os.path.dirname(__file__), "backend", "ctu13_world_model.pt")
    return load_model(checkpoint_path)

model = get_model()

# --- from here on, this is just example usage inside your UI code ---

# raw_window: a list of feature values in REAL units, in this exact order
# (see backend/ctu13_pipeline.py's FEATURE_COLS, or just check
# model.feature_cols after loading -- it's saved in the checkpoint):
#   [flow_count, unique_dst_ports, unique_dst_hosts, syn_ack_ratio,
#    iat_mean, iat_var, bytes_out, bytes_in, ttl_var, retransmits,
#    has_packet_coverage]
#
# raw_context: a list of 3+ such windows -- the model's real, observed
# traffic history for one host, oldest first

example_context = [
    [20, 3, 2, 1.0, 0.5, 0.05, 50000, 60000, 0.0, 0.0, 0.0],
    [22, 3, 2, 1.0, 0.5, 0.05, 48000, 61000, 0.0, 0.0, 0.0],
    [25, 60, 40, 3.5, 0.5, 0.05, 52000, 59000, 0.0, 0.0, 0.0],  # port fan-out starting
]

result = forecast(model, example_context, k_steps=8)
# result = {
#   "k_steps": 8,
#   "forecast": [ {"step":1, "risk":0.97, "alert":True,
#                  "predicted_stage":"command_and_control",
#                  "recommended_response":[...]}, ... ],
#   "peak_risk": 0.97,
#   "peak_stage": "command_and_control",
#   "threshold_used": 0.288,
# }
# -> feed result["forecast"] directly into your probability-timeline chart

explanation = explain(model, example_context[-1])
# explanation = {
#   "predicted_stage": "command_and_control",
#   "top_features": [{"feature":"unique_dst_ports","attribution":1.83}, ...],
#   "explanation": "Flagged primarily due to unique dst ports and unique
#                   dst hosts, consistent with command and control behavior."
# }
# -> feed explanation["explanation"] directly into your explainability panel
