import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "project_package" / "backend"
CHECKPOINT_PATH = BACKEND_DIR / "ctu13_world_model.pt"
SAMPLE_PATH = ROOT / "data" / "sample_scenarios.json"
sys.path.insert(0, str(BACKEND_DIR))

from inference import explain, forecast, load_model


st.set_page_config(
    page_title="Attack Forecast Console Dashboard",
    page_icon="Shield",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model():
    return load_model(str(CHECKPOINT_PATH))


@st.cache_data
def load_sample_scenarios():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample scenario file not found: {SAMPLE_PATH}")
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return payload["scenarios"]


def validate_feature_csv(frame: pd.DataFrame, model):
    """Validate a raw feature-window CSV without substituting sample data."""
    required = list(model.feature_cols)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model feature column(s): {', '.join(missing)}")

    values = frame[required]
    non_numeric = [column for column in required if not pd.api.types.is_numeric_dtype(values[column])]
    if non_numeric:
        raise ValueError(f"Feature column(s) must be numeric: {', '.join(non_numeric)}")
    nulls = [column for column in required if values[column].isna().any()]
    if nulls:
        raise ValueError(f"Feature column(s) contain null values: {', '.join(nulls)}")

    time_column = next(
        (column for column in ("window", "timestamp", "time", "StartTime") if column in frame.columns),
        None,
    )
    if time_column:
        times = pd.to_datetime(frame[time_column], errors="coerce")
        if times.isna().any():
            raise ValueError(f"Time column '{time_column}' contains invalid timestamps")
        if not times.is_monotonic_increasing:
            raise ValueError(f"Rows must be sorted by time column '{time_column}'")

    host_column = next(
        (column for column in ("host", "host_id", "src_addr", "SrcAddr") if column in frame.columns),
        None,
    )
    if host_column:
        host_counts = frame.groupby(host_column, dropna=False).size()
        if host_counts.max() < model.seq_len:
            raise ValueError(
                f"At least {model.seq_len} rows are required for one host; "
                f"the largest host group has {int(host_counts.max())}"
            )
        selected_host = host_counts.idxmax()
        selected = frame[frame[host_column] == selected_host]
    else:
        if len(frame) < model.seq_len:
            raise ValueError(
                f"At least {model.seq_len} rows are required for one host; received {len(frame)}"
            )
        selected = frame

    return selected[required].astype("float32").tail(model.seq_len).values.tolist(), host_column


def run_forecast(model, context, k_steps, threshold_override=None):
    """Run the model seam once and shape the response for dashboard sections."""
    result = forecast(model, context, k_steps=k_steps)
    if threshold_override is not None:
        threshold = float(threshold_override)
        for step in result["forecast"]:
            step["alert"] = step["risk"] > threshold
        result["threshold_used"] = threshold
    explanation = explain(model, context[-1])
    return {"forecast": result, "explanation": explanation, "context": context}


def stage_label(stage):
    return stage.replace("_", " ").title()


def risk_percent(value):
    return f"{value * 100:.1f}%"


def render_kpi(label, value, subtext, sub_color="#64748B"):
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:14px 16px;">
          <div style="font-size:.78rem;color:#64748B;">{label}</div>
          <div style="font-size:1.7rem;font-weight:800;color:#0F172A;">{value}</div>
          <div style="font-size:.75rem;color:{sub_color};font-weight:600;">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeline(forecast_rows):
    colors = {
        "benign": ("#15803D", "#F0FDF4", "#DCFCE7"),
        "reconnaissance": ("#A16207", "#FEFCE8", "#FEF08A"),
        "lateral_movement": ("#DC2626", "#FEF2F2", "#FECDD3"),
        "command_and_control": ("#C2410C", "#FFF7ED", "#FFEDD5"),
        "exfiltration": ("#C2410C", "#FFF7ED", "#FFEDD5"),
    }
    parts = []
    for row in forecast_rows:
        stage = row["predicted_stage"]
        text_color, background, border = colors.get(stage, colors["benign"])
        border_width = "2px" if row["alert"] else "1px"
        parts.append(
            f"<div style='flex:1;border-radius:6px;padding:8px 4px;text-align:center;"
            f"font-size:.72rem;font-weight:600;color:{text_color};background:{background};"
            f"border:{border_width} solid {border};'>+{row['step']} "
            f"{stage_label(stage)}</div>"
        )
    st.markdown(
        "<div style='display:flex;gap:8px;margin-bottom:24px;'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def render_flagged_rows(rows, host):
    flagged = [row for row in rows if row["alert"]]
    if not flagged:
        st.info("No forecast window crosses the active alert threshold.")
        return
    table = pd.DataFrame(
        [
            {
                "Window": f"t +{row['step']}",
                "Host": host,
                "Predicted stage": stage_label(row["predicted_stage"]),
                "Risk score": f"{row['risk']:.4f}",
            }
            for row in flagged
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


model = get_model()
samples = load_sample_scenarios()

if "sidebar_mode" not in st.session_state:
    st.session_state.sidebar_mode = "sample"
if "inference_state" not in st.session_state:
    st.session_state.inference_state = "not_run"
if "inference_result" not in st.session_state:
    st.session_state.inference_result = None
if "inference_host" not in st.session_state:
    st.session_state.inference_host = ""

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:700;color:#0F172A;'>Attack Forecast</div>"
        "<div style='font-size:.75rem;font-weight:600;letter-spacing:.08em;color:#64748B;"
        "margin-bottom:16px;'>CONSOLE</div>",
        unsafe_allow_html=True,
    )
    tab_sample, tab_upload = st.columns(2)
    with tab_sample:
        if st.button(
            "Sample scenario",
            use_container_width=True,
            type="primary" if st.session_state.sidebar_mode == "sample" else "secondary",
        ):
            st.session_state.sidebar_mode = "sample"
            st.session_state.inference_state = "not_run"
            st.session_state.inference_result = None
            st.rerun()
    with tab_upload:
        if st.button(
            "Upload file",
            use_container_width=True,
            type="primary" if st.session_state.sidebar_mode == "upload" else "secondary",
        ):
            st.session_state.sidebar_mode = "upload"
            st.session_state.inference_state = "not_run"
            st.session_state.inference_result = None
            st.rerun()

    if st.session_state.sidebar_mode == "upload":
        st.markdown("##### Upload feature-window CSV")
        uploaded_file = st.file_uploader(
            "CSV with the 11 model features per window",
            type=["csv"],
            help="Required columns are read from the loaded checkpoint. PCAP and log files are not supported.",
        )
        upload_context = None
        upload_host = "uploaded-host"
        if uploaded_file is not None:
            try:
                uploaded_frame = pd.read_csv(uploaded_file)
                upload_context, host_column = validate_feature_csv(uploaded_frame, model)
                if host_column:
                    counts = uploaded_frame.groupby(host_column, dropna=False).size()
                    upload_host = str(counts.idxmax())
                st.success(f"Validated: {uploaded_file.name}")
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                st.error(f"CSV validation failed: {exc}")
    else:
        scenario_id = st.selectbox(
            "Model / Scenario Selection",
            options=[sample["id"] for sample in samples],
            format_func=lambda value: next(sample["label"] for sample in samples if sample["id"] == value),
            label_visibility="collapsed",
        )
        selected_sample = next(sample for sample in samples if sample["id"] == scenario_id)
        upload_context = selected_sample["context"]
        upload_host = selected_sample["host_id"]

    alert_threshold = st.slider(
        "Alert threshold override (%)",
        min_value=0,
        max_value=100,
        value=round(float(model.threshold) * 100),
        step=1,
        help=f"Model checkpoint default: {model.threshold:.3f}.",
    )
    horizon_label = st.selectbox("Forecast horizon", ["+4 windows", "+8 windows", "+12 windows", "+24 windows"], index=1)
    horizon = int(horizon_label.split()[0][1:])

    run_clicked = st.button("Run inference", use_container_width=True, type="secondary")
    if run_clicked:
        if upload_context is None:
            st.session_state.inference_state = "failed"
            st.session_state.inference_result = None
        else:
            st.session_state.inference_state = "running"
            with st.spinner("Running world-model forecast..."):
                try:
                    st.session_state.inference_result = run_forecast(
                        model,
                        upload_context,
                        horizon,
                        threshold_override=alert_threshold / 100,
                    )
                    st.session_state.inference_host = upload_host
                    st.session_state.inference_state = "complete"
                except (ValueError, RuntimeError, KeyError) as exc:
                    st.session_state.inference_state = "failed"
                    st.session_state.inference_result = None
                    st.error(f"Inference failed: {exc}")

result = st.session_state.inference_result
st.markdown(
    "<div style='font-size:1.6rem;font-weight:700;color:#0F172A;'>Attack forecast console</div>"
    "<div style='font-size:.82rem;color:#64748B;margin-bottom:16px;'>"
    "Real CTU-13 checkpoint - flat backend integration</div>",
    unsafe_allow_html=True,
)

if st.session_state.inference_state == "not_run":
    st.info("Select a sample or upload a feature CSV, then click Run inference.")
elif st.session_state.inference_state == "failed":
    st.error("Inference failed. Correct the input or review the specific error in the sidebar.")
elif st.session_state.inference_state == "running":
    st.warning("Inference is running.")

if result is not None and st.session_state.inference_state == "complete":
    forecast_result = result["forecast"]
    rows = forecast_result["forecast"]
    explanation = result["explanation"]
    host = st.session_state.inference_host
    first_risk = rows[0]["risk"]
    peak_risk = forecast_result["peak_risk"]
    first_alert = next((row["step"] for row in rows if row["alert"]), None)
    alert_count = sum(row["alert"] for row in rows)
    alert_text = f"t +{first_alert}" if first_alert else "not reached"
    banner_color = "#FFF7ED" if alert_count else "#F0FDF4"
    banner_border = "#EA580C" if alert_count else "#16A34A"
    banner_text = (
        f"+ Elevated risk forecast for {host} - {alert_count} of {len(rows)} future windows cross the threshold"
        if alert_count
        else f"No elevated risk forecast for {host} - no future window crosses the threshold"
    )
    st.markdown(
        f"<div style='background:{banner_color};border:1px solid #FED7AA;border-left:4px solid "
        f"{banner_border};border-radius:6px;padding:10px 14px;color:#9A3412;font-size:.85rem;"
        f"font-weight:500;margin-bottom:20px;'>{banner_text}</div>",
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_kpi("Current forecast risk", risk_percent(first_risk), f"Peak {risk_percent(peak_risk)}", "#EA580C" if alert_count else "#16A34A")
    with kpi2:
        render_kpi("Predicted MITRE stage", stage_label(forecast_result["peak_stage"]), f"Peak forecast for {host}")
    with kpi3:
        render_kpi("First alert window", alert_text, f"Threshold {forecast_result['threshold_used']:.3f}", "#EA580C" if first_alert else "#16A34A")
    with kpi4:
        render_kpi("Hosts monitored", "1", "Selected input host")

    st.markdown("#### Probability of attack rollout across time horizon")
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=[f"t +{row['step']}" for row in rows],
            y=[row["risk"] * 100 for row in rows],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#1E3A8A", width=2.5, dash="dash"),
            marker=dict(size=7, color="#2563EB"),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    chart.add_hline(
        y=forecast_result["threshold_used"] * 100,
        line_width=1.5,
        line_dash="dot",
        line_color="#DC2626",
        annotation_text="alert threshold",
    )
    chart.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=15, b=40),
        height=300,
        yaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="#F1F5F9"),
        xaxis=dict(title="Forecast window", gridcolor="#F1F5F9"),
        hovermode="x unified",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.markdown("#### Predicted MITRE stage timeline")
    render_timeline(rows)

    st.markdown("#### Flagged windows")
    render_flagged_rows(rows, host)

    st.markdown("#### Explainability - latest observed window")
    feature_lines = []
    for item in explanation["top_features"]:
        score = abs(item["attribution"])
        width = min(score / max(1.0, max(abs(x["attribution"]) for x in explanation["top_features"])) * 100, 100)
        feature_lines.append(
            f"<div style='margin-bottom:12px;'><div style='display:flex;justify-content:space-between;"
            f"font-size:.8rem;font-weight:500;color:#334155;margin-bottom:4px;'>"
            f"<span>{item['feature'].replace('_', ' ')}</span><span>{item['attribution']:.4f}</span></div>"
            f"<div style='width:100%;height:6px;background:#E2E8F0;border-radius:3px;overflow:hidden;'>"
            f"<div style='height:100%;width:{width:.1f}%;background:#1E3A8A;border-radius:3px;'></div></div></div>"
        )
    st.markdown(
        f"<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:20px;"
        f"margin-bottom:24px;'><div style='font-size:.92rem;font-weight:700;color:#0F172A;margin-bottom:8px;'>"
        f"Predicted stage: {stage_label(explanation['predicted_stage'])}</div><div style='font-size:.82rem;"
        f"color:#475569;line-height:1.5;margin-bottom:16px;'>{explanation['explanation']}</div>"
        + "".join(feature_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Advisory response guidance")
    guidance = next((row["recommended_response"] for row in rows if row["alert"]), [])
    if guidance:
        for item in guidance:
            st.markdown(f"- {item}")
    else:
        st.caption("No response action is recommended while the forecast remains below threshold.")

st.markdown("#### World model vs. baseline")
wm_col1, wm_col2 = st.columns(2)
with wm_col1:
    st.metric("World model F1", "0.4254", "Real held-out CTU-13 result")
    st.caption("Active checkpoint: CTU-13 Scenarios 1/2/9, tested on Scenario 10.")
with wm_col2:
    st.metric("Baseline F1", "0.2491", "Logistic regression comparison")
    st.caption("The comparison is fixed benchmark metadata, not a live inference result.")
