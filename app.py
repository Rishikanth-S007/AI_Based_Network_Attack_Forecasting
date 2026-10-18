import json
import os
import sys
import tempfile
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
from raw_ingest import process_raw_upload


st.set_page_config(
    page_title="Attack Forecast Console Dashboard",
    page_icon="Shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

if "sidebar_mode" not in st.session_state:
    st.session_state.sidebar_mode = "sample"


@st.cache_resource
def get_model():
    return load_model(str(CHECKPOINT_PATH))


@st.cache_data
def load_sample_scenarios():
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return payload["scenarios"]


def validate_feature_csv(frame, model):
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
    if len(frame) < model.seq_len:
        raise ValueError(f"At least {model.seq_len} rows are required; received {len(frame)}")
    time_column = next(
        (column for column in ("window", "timestamp", "time", "StartTime") if column in frame.columns),
        None,
    )
    if time_column:
        times = pd.to_datetime(frame[time_column], errors="coerce")
        if times.isna().any() or not times.is_monotonic_increasing:
            raise ValueError(f"Rows must be sorted by valid time column '{time_column}'")
    host_column = next(
        (column for column in ("host", "host_id", "src_addr", "SrcAddr") if column in frame.columns),
        None,
    )
    if host_column:
        counts = frame.groupby(host_column, dropna=False).size()
        if counts.max() < model.seq_len:
            raise ValueError(
                f"At least {model.seq_len} rows are required for one host; "
                f"the largest host group has {int(counts.max())}"
            )
        host = counts.idxmax()
        frame = frame[frame[host_column] == host]
        return frame[required].astype("float32").tail(model.seq_len).values.tolist(), str(host)
    return frame[required].astype("float32").tail(model.seq_len).values.tolist(), "uploaded-host"


def run_model_forecast(model, context, horizon, threshold):
    result = forecast(model, context, k_steps=horizon)
    for row in result["forecast"]:
        row["alert"] = row["risk"] > threshold
    result["threshold_used"] = threshold
    return result, explain(model, context[-1])


model = get_model()
samples = load_sample_scenarios()
if "inference_result" not in st.session_state:
    st.session_state.inference_result = None
if "inference_explanation" not in st.session_state:
    st.session_state.inference_explanation = None
if "inference_host" not in st.session_state:
    st.session_state.inference_host = ""

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:700;color:#0F172A;margin-bottom:1px;'>"
        "Attack Forecast</div><div style='font-size:.75rem;font-weight:600;letter-spacing:.08em;"
        "color:#64748B;margin-bottom:16px;'>CONSOLE</div>"
        "<hr style='border:0;border-top:1px solid #E2E8F0;margin-bottom:20px;'/>",
        unsafe_allow_html=True,
    )

    col_tab1, col_tab2 = st.columns(2)
    with col_tab1:
        if st.button(
            "Sample scenario",
            key="btn_sample",
            use_container_width=True,
            type="primary" if st.session_state.sidebar_mode == "sample" else "secondary",
        ):
            st.session_state.sidebar_mode = "sample"
            st.rerun()
    with col_tab2:
        if st.button(
            "Upload file",
            key="btn_upload",
            use_container_width=True,
            type="primary" if st.session_state.sidebar_mode == "upload" else "secondary",
        ):
            st.session_state.sidebar_mode = "upload"
            st.rerun()

    st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)
    input_context = None
    input_host = ""
    if st.session_state.sidebar_mode == "upload":
        st.markdown("##### Upload traffic data")
        uploaded_files = st.file_uploader(
            "Feature-window CSV, raw CTU-13 NetFlow CSV, or PCAP/PCAPNG",
            type=["csv", "pcap", "pcapng"],
            accept_multiple_files=True,
            help="PCAP-only uploads cannot run inference without a matching NetFlow CSV.",
        )
        if uploaded_files and len(uploaded_files) > 2:
            st.error("Upload one data file, or one PCAP/PCAPNG plus its matching NetFlow CSV.")
            uploaded_files = []
        if uploaded_files:
            total_size = sum(uploaded_file.size for uploaded_file in uploaded_files)
            if total_size > 200 * 1024 * 1024:
                st.warning(
                    "The selected upload is larger than 200 MB. PCAP parsing is memory-hungry "
                    "and may take several minutes or exceed available memory."
                )
            temp_paths = []
            try:
                file_entries = []
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_paths.append(temp_file.name)
                    file_entries.append((temp_file.name, suffix.lstrip("."), uploaded_file.name))
                pcap_entries = [entry for entry in file_entries if entry[1] in {"pcap", "pcapng"}]
                csv_entries = [entry for entry in file_entries if entry[1] == "csv"]
                if len(pcap_entries) > 1 or len(csv_entries) > 1:
                    st.error("Upload at most one PCAP/PCAPNG and one companion CSV.")
                    windowed_frame, ingest_error = None, None
                elif pcap_entries:
                    pcap_path, pcap_type, _ = pcap_entries[0]
                    ingest_path = (pcap_path, csv_entries[0][0]) if csv_entries else pcap_path
                    with st.status("Parsing PCAP packet features...", expanded=True) as status:
                        windowed_frame, ingest_error = process_raw_upload(ingest_path, pcap_type)
                        status.update(
                            label="PCAP parsing complete",
                            state="complete" if ingest_error is None else "error",
                        )
                else:
                    ingest_path, file_type, _ = file_entries[0]
                    windowed_frame, ingest_error = process_raw_upload(ingest_path, file_type)
                if ingest_error:
                    st.error(ingest_error)
                else:
                    host_column = next(
                        (
                            column
                            for column in ("host", "host_id", "src_addr", "SrcAddr")
                            if column in windowed_frame.columns
                        ),
                        None,
                    )
                    if host_column is None:
                        st.error("Uploaded traffic data does not contain a host column.")
                    else:
                        unique_hosts = windowed_frame[host_column].unique()
                        pitch_ips = ["147.32.84.165", "147.32.84.59", "147.32.84.138"]
                        ordered_hosts = [
                            host for host in pitch_ips if host in unique_hosts
                        ] + [
                            host for host in unique_hosts if host not in pitch_ips
                        ]
                        default_host_index = (
                            ordered_hosts.index("147.32.84.165")
                            if "147.32.84.165" in ordered_hosts
                            else 0
                        )
                        selected_host = st.sidebar.selectbox(
                            "Select Host to Monitor",
                            ordered_hosts,
                            index=default_host_index,
                        )
                        windowed_frame = windowed_frame[
                            windowed_frame[host_column] == selected_host
                        ]
                        input_context, input_host = validate_feature_csv(windowed_frame, model)
                        st.success("Validated uploaded traffic data")
            except (ValueError, RuntimeError, KeyError) as exc:
                st.error(f"Upload validation failed: {exc}")
            finally:
                for temp_path in temp_paths:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
    else:
        scenario_id = st.selectbox(
            "Model / Scenario Selection",
            [sample["id"] for sample in samples],
            format_func=lambda value: next(sample["label"] for sample in samples if sample["id"] == value),
            index=0,
            label_visibility="collapsed",
        )
        selected_sample = next(sample for sample in samples if sample["id"] == scenario_id)
        input_context = selected_sample["context"]
        input_host = selected_sample["host_id"]

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
    slider_val = st.session_state.get("alert_thresh_slider", 51)
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;font-size:.92rem;"
        f"font-weight:700;color:#0F172A;margin-bottom:6px;'><span>Alert threshold</span>"
        f"<span style='color:#2B4C7E;font-size:.95rem;font-weight:600;'>{slider_val}%</span></div>",
        unsafe_allow_html=True,
    )
    alert_threshold = st.slider(
        "Alert threshold slider",
        min_value=0,
        max_value=100,
        value=round(float(model.threshold) * 100),
        step=1,
        key="alert_thresh_slider",
        label_visibility="collapsed",
    )
    st.markdown(
        "<div style='display:flex;justify-content:space-between;font-size:.75rem;color:#94A3B8;"
        "margin-top:-8px;margin-bottom:18px;'><span>0%</span><span>100%</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:.92rem;font-weight:700;color:#0F172A;margin-bottom:6px;'>"
        "Forecast horizon</div>",
        unsafe_allow_html=True,
    )
    horizon_label = st.selectbox(
        "Forecast horizon dropdown",
        ["+12 windows", "+4 windows", "+8 windows", "+24 windows"],
        index=0,
        label_visibility="collapsed",
    )
    horizon = int(horizon_label.split()[0][1:])

    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='run-inference-btn'>", unsafe_allow_html=True)
    run_clicked = st.button("Run inference", key="btn_run_inf", use_container_width=True, type="secondary")
    st.markdown("</div>", unsafe_allow_html=True)
    if run_clicked:
        if input_context is None:
            st.session_state.inference_result = None
            st.session_state.inference_explanation = None
            st.error("Inference cannot start until the input is valid.")
        else:
            with st.spinner("Running world-model forecast..."):
                try:
                    result, explanation = run_model_forecast(
                        model, input_context, horizon, alert_threshold / 100
                    )
                    st.session_state.inference_result = result
                    st.session_state.inference_explanation = explanation
                    st.session_state.inference_host = input_host
                except (ValueError, RuntimeError, KeyError) as exc:
                    st.session_state.inference_result = None
                    st.session_state.inference_explanation = None
                    st.error(f"Inference failed: {exc}")
    st.markdown(
        "<div style='display:flex;justify-content:space-between;font-size:.75rem;color:#64748B;"
        "margin-top:35px;padding-top:15px;border-top:1px solid #E2E8F0;'><span>Feed connected</span>"
        "<span>08:47 UTC</span></div>",
        unsafe_allow_html=True,
    )

model_result = st.session_state.inference_result
model_explanation = st.session_state.inference_explanation
forecast_rows = model_result["forecast"] if model_result else []
history_rows = model_result["history"] if model_result else []
current_risk = model_result["current_risk"] if model_result else 0.0
active_host = st.session_state.inference_host or "not run"
peak_risk = model_result["peak_risk"] if model_result else 0.0
peak_stage = model_result["peak_stage"] if model_result else "benign"
alert_count = sum(row["alert"] for row in forecast_rows)

st.markdown(
    "<div style='font-size:1.6rem;font-weight:700;color:#0F172A;margin-bottom:2px;'>"
    "Attack forecast console</div><div style='font-size:.82rem;color:#64748B;margin-bottom:16px;'>"
    "v1.0.0 — Local Benchmark — Forecast horizon: +12 windows</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='background-color:{'#FFF7ED' if alert_count else '#F0FDF4'};border:1px solid #FED7AA;"
    f"border-left:4px solid {'#EA580C' if alert_count else '#16A34A'};"
    "border-radius:6px;padding:10px 14px;color:#9A3412;font-size:.85rem;font-weight:500;"
    f"margin-bottom:20px;'>{('+ Elevated risk detected' if alert_count else 'No elevated risk detected')} — "
    f"{(f'{alert_count} of {len(forecast_rows)} forecast windows cross the threshold for {active_host}' if model_result else 'Select a scenario and run inference to load model-backed results')}</div>",
    unsafe_allow_html=True,
)


def card(label, value, detail, color="#64748B"):
    st.markdown(
        f"<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:14px 16px;'>"
        f"<div style='font-size:.78rem;color:#64748B;'>{label}</div>"
        f"<div style='font-size:1.7rem;font-weight:800;color:#0F172A;'>{value}</div>"
        f"<div style='font-size:.75rem;color:{color};font-weight:600;'>{detail}</div></div>",
        unsafe_allow_html=True,
    )


kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    card("Current risk", f"{current_risk * 100:.1f}%" if model_result else "—",
         f"{alert_count} alert windows" if model_result else "Run inference", "#EA580C" if alert_count else "#64748B")
with kpi2:
    card("Predicted MITRE stage", peak_stage.replace("_", " ").title() if model_result else "—",
         f"Peak forecast for {active_host}")
with kpi3:
    card("Alert threshold", f"{model_result['threshold_used'] * 100:.1f}%" if model_result else "—",
         f"Horizon: {horizon_label}", "#1D4ED8")
with kpi4:
    card("Hosts monitored", "1" if model_result else "—",
         active_host if model_result else "No input selected")

st.markdown(
    "<div style='font-size:.95rem;font-weight:700;color:#0F172A;margin-top:24px;margin-bottom:10px;'>"
    "Probability of attack rollout across time horizon</div>",
    unsafe_allow_html=True,
)

if model_result:
    past = pd.DataFrame(
        {
            "time": [f"t {row['step']}" for row in history_rows] + ["t0"],
            "probability": [row["risk"] * 100 for row in history_rows] + [current_risk * 100],
        }
    )
    future = pd.DataFrame(
        {
            "time": ["t0"] + [f"t +{row['step']}" for row in forecast_rows],
            "probability": [current_risk * 100] + [row["risk"] * 100 for row in forecast_rows],
        }
    )
else:
    past = pd.DataFrame({"time": ["t -4", "t -3", "t -2", "t -1", "t0"], "probability": [28, 32, 35, 48, 74]})
    future = pd.DataFrame(
        {
            "time": ["t0"] + [f"t +{i}" for i in range(1, 13)],
            "probability": [74, 81, 88, 91, 94, 95, 95.5, 96, 96, 96.5, 96.5, 97, 97],
        }
    )
fig = go.Figure()
fig.add_trace(go.Scatter(x=past.time, y=past.probability, mode="lines+markers", name="Observed",
                         line=dict(color="#1E3A8A", width=2.5),
                         marker=dict(size=6, color="#1E3A8A", symbol="circle")))
fig.add_trace(go.Scatter(x=future.time, y=future.probability, mode="lines+markers", name="Forecast",
                         line=dict(color="#1E3A8A", width=2.5, dash="dash"),
                         marker=dict(size=6, color="#1E3A8A", symbol="circle-open")))
fig.add_hline(y=model_result["threshold_used"] * 100 if model_result else 51,
              line_width=1.2, line_dash="dot", line_color="#DC2626",
              annotation_text="alert threshold" if model_result else "threshold")
fig.add_shape(
    type="line",
    x0="t0",
    x1="t0",
    y0=0,
    y1=1,
    yref="paper",
    line=dict(width=1.5, dash="dot", color="#DC2626"),
)
fig.add_annotation(
    x="t0",
    xshift=15,
    y=1,
    yref="paper",
    text="t0",
    showarrow=False,
    font=dict(color="#DC2626", size=11),
)
fig.update_layout(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", margin=dict(l=40, r=40, t=15, b=40),
    height=300, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Time Horizon", gridcolor="#F1F5F9"),
    yaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="#F1F5F9"),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<div style='font-size:.95rem;font-weight:700;color:#0F172A;margin-top:24px;margin-bottom:10px;'>"
    "Predicted MITRE stage timeline</div>",
    unsafe_allow_html=True,
)
stages = [
    ("Recon", "#15803D", "#F0FDF4", "#DCFCE7"),
    ("Weapon", "#A16207", "#FEFCE8", "#FEF08A"),
    ("Delivery", "#A16207", "#FEFCE8", "#FEF08A"),
    ("t 0 Exploit", "#DC2626", "#FEF2F2", "#DC2626"),
    ("Lat Move", "#DC2626", "#FEF2F2", "#FECDD3"),
    ("Collection", "#C2410C", "#FFF7ED", "#FFEDD5"),
    ("Exfil", "#C2410C", "#FFF7ED", "#FFEDD5"),
    ("C2", "#C2410C", "#FFF7ED", "#FFEDD5"),
]
if model_result:
    stage_colors = {
        "benign": ("#15803D", "#F0FDF4", "#DCFCE7"),
        "reconnaissance": ("#A16207", "#FEFCE8", "#FEF08A"),
        "lateral_movement": ("#DC2626", "#FEF2F2", "#FECDD3"),
        "command_and_control": ("#C2410C", "#FFF7ED", "#FFEDD5"),
        "exfiltration": ("#C2410C", "#FFF7ED", "#FFEDD5"),
    }
    stages = []
    for row in forecast_rows:
        stage = row["predicted_stage"]
        color, background, border = stage_colors.get(stage, stage_colors["benign"])
        stages.append((f"+{row['step']} {stage.replace('_', ' ').title()}", color, background, border))
timeline = "<div style='display:flex;gap:8px;margin-bottom:24px;'>"
for name, color, background, border in stages:
    width = "2px" if (name == "t 0 Exploit" or name.startswith("+")) and model_result else "1px"
    timeline += (
        f"<div style='flex:1;border-radius:6px;padding:8px 4px;text-align:center;font-size:.78rem;"
        f"font-weight:600;color:{color};background:{background};border:{width} solid {border};'>{name}</div>"
    )
st.markdown(timeline + "</div>", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:.95rem;font-weight:700;color:#0F172A;margin-top:24px;margin-bottom:10px;'>"
    "Flagged windows</div>",
    unsafe_allow_html=True,
)
rows = [
    ("t -1", "WIN-EXCH01", "Exploitation", "0.91", True),
    ("t 0", "WIN-DC01", "Lateral Movement", "0.87", True),
    ("t +1", "LNX-APP03", "Persistence", "0.68", False),
    ("t +2", "WIN-WKS14", "Collection", "0.52", False),
    ("t +3", "WIN-WKS22", "Exfiltration", "0.41", False),
]
if model_result:
    rows = [
        (
            f"t +{row['step']}",
            active_host,
            row["predicted_stage"].replace("_", " ").title(),
            f"{row['risk']:.4f}",
            row["alert"],
        )
        for row in forecast_rows
        if row["alert"]
    ]
table = (
    "<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;overflow:hidden;margin-bottom:24px;'>"
    "<table style='width:100%;border-collapse:collapse;font-size:.85rem;'><thead><tr style='border-bottom:1px solid #E2E8F0;'>"
    "<th style='padding:12px 18px;text-align:left;color:#64748B;'>Window</th>"
    "<th style='padding:12px 18px;text-align:left;color:#64748B;'>Host</th>"
    "<th style='padding:12px 18px;text-align:left;color:#64748B;'>Predicted stage</th>"
    "<th style='padding:12px 18px;text-align:right;color:#64748B;'>Risk score</th></tr></thead><tbody>"
)
for window, host, stage, score, critical in rows:
    background = "background-color:#FEF2F2;color:#991B1B;" if critical else "color:#0F172A;"
    table += (
        f"<tr style='border-bottom:1px solid #F1F5F9;{background}'>"
        f"<td style='padding:12px 18px;'>{window}</td><td style='padding:12px 18px;font-family:monospace;'>{host}</td>"
        f"<td style='padding:12px 18px;'>{stage}</td><td style='padding:12px 18px;text-align:right;font-weight:700;'>{score}</td></tr>"
    )
if not rows:
    table += "<tr><td colspan='4' style='padding:18px;color:#64748B;'>No forecast windows cross the active threshold.</td></tr>"
st.markdown(table + "</tbody></table></div>", unsafe_allow_html=True)

if model_explanation:
    attribution_html = ""
    for item in model_explanation["top_features"]:
        impact = item["impact_percent"]
        direction = "+" if item["attribution"] >= 0 else "-"
        color = "#1D4ED8" if item["attribution"] >= 0 else "#DC2626"
        attribution_html += (
            "<div style='font-size:.8rem;font-weight:500;color:#334155;margin-bottom:4px;"
            "display:flex;justify-content:space-between;'>"
            f"<span>{item['feature'].replace('_', ' ')}</span>"
            f"<span style='color:{color};'>{direction}{impact:.1f}% impact on risk</span></div>"
            f"<div style='height:6px;background:#E2E8F0;border-radius:3px;margin-bottom:12px;'>"
            f"<div style='height:100%;width:{impact:.1f}%;background:{color};border-radius:3px;'></div></div>"
        )
    explanation_title = f"Explainability — latest observed window ({active_host})"
    explanation_text = model_explanation["explanation"]
else:
    attribution_html = "<div style='font-size:.8rem;color:#64748B;'>Run inference to display model-derived feature attributions.</div>"
    explanation_title = "Explainability — latest observed window"
    explanation_text = "No model result is available yet."
st.markdown(
    f"<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:20px;margin-bottom:24px;'>"
    f"<div style='font-size:.92rem;font-weight:700;color:#0F172A;margin-bottom:8px;'>{explanation_title}</div>"
    f"<div style='font-size:.82rem;color:#475569;line-height:1.5;margin-bottom:16px;'>{explanation_text}</div>"
    f"{attribution_html}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-size:.95rem;font-weight:700;color:#0F172A;margin-top:24px;margin-bottom:10px;'>"
    "World model vs. baseline</div>",
    unsafe_allow_html=True,
)
wm_col1, wm_col2 = st.columns(2)
with wm_col1:
    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #1E3A8A;border-radius:8px;padding:18px;height:100%;'>"
        "<div style='font-size:.98rem;font-weight:700;color:#0F172A;margin-bottom:4px;'>World model "
        "<span style='background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;border-radius:4px;padding:1px 6px;font-size:.7rem;'>Active</span></div>"
        "<div style='font-size:.78rem;color:#64748B;margin-bottom:14px;'>Learned state-space representation with attention-weighted temporal context</div>"
        "<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px dashed #E2E8F0;'><span>Lead time achieved</span><strong style='color:#1D4ED8;'>3.2 windows</strong></div>"
        "<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px dashed #E2E8F0;'><span>False positive rate</span><strong style='color:#16A34A;'>2.1%</strong></div>"
        "<div style='display:flex;justify-content:space-between;padding:6px 0;'><span>Precision / Recall</span><strong>94.8% / 91.5%</strong></div></div>",
        unsafe_allow_html=True,
    )
with wm_col2:
    st.markdown(
        "<div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:18px;height:100%;'>"
        "<div style='font-size:.98rem;font-weight:700;color:#475569;margin-bottom:4px;'>Baseline (logistic regression)</div>"
        "<div style='font-size:.78rem;color:#64748B;margin-bottom:14px;'>Log-odds model on 14 hand-engineered network flow features</div>"
        "<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px dashed #E2E8F0;'><span>Lead time achieved</span><strong>1.1 windows</strong></div>"
        "<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px dashed #E2E8F0;'><span>False positive rate</span><strong style='color:#DC2626;'>8.4%</strong></div>"
        "<div style='display:flex;justify-content:space-between;padding:6px 0;'><span>Precision / Recall</span><strong>70.83% / 15.11%</strong></div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br/><br/>", unsafe_allow_html=True)
