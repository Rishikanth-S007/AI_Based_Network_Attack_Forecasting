import numpy as np
import pandas as pd
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
from datetime import datetime, timedelta

def get_kpi_summary_metrics(scenario: str):
    """
    Returns exact Light Theme KPI metrics matching the Figma design.
    """
    return {
        "current_risk": "74%",
        "risk_delta": "+12% vs t-1",
        "predicted_mitre_stage": "T1021",
        "mitre_stage_sub": "Lateral Movement",
        "lead_time": "3.2 win",
        "lead_time_sub": "vs baseline 1.1",
        "hosts_monitored": "847",
        "hosts_sub": "All systems active",
    }

def generate_forecast_timeseries(horizon_windows: int = 4):
    """
    Generates time-series data matching the light-theme Plotly rollout chart.
    """
    past_windows = 4
    past_times = [f"t -{past_windows - i}" for i in range(past_windows)] + ["t 0"]
    past_vals = [28.0, 32.0, 35.0, 48.0, 74.0]
    
    future_times = [f"t +{i+1}" for i in range(horizon_windows)]
    future_vals = [81.0, 88.0, 91.0, 94.0][:horizon_windows]
    
    df_past = pd.DataFrame({
        "time": past_times,
        "probability": past_vals,
        "type": "Observed"
    })

    df_future = pd.DataFrame({
        "time": [past_times[-1]] + future_times,
        "probability": [past_vals[-1]] + future_vals,
        "type": "Forecast"
    })

    return df_past, df_future

def get_mitre_stages():
    """
    Returns stage names, status colors, and active stage flags for timeline row.
    """
    return [
        {"name": "Recon", "style": "mitre-green", "active": False},
        {"name": "Weapon", "style": "mitre-yellow", "active": False},
        {"name": "Delivery", "style": "mitre-yellow", "active": False},
        {"name": "t 0 Exploit", "style": "mitre-red mitre-active", "active": True},
        {"name": "Lat Move", "style": "mitre-red", "active": False},
        {"name": "Collection", "style": "mitre-orange", "active": False},
        {"name": "Exfil", "style": "mitre-orange", "active": False},
        {"name": "C2", "style": "mitre-orange", "active": False},
    ]

def get_host_prediction_rows():
    """
    Returns Host Prediction Data rows for clean HTML rendering matching Figma.
    """
    return [
        {
            "window": "t -1",
            "host": "WIN-EXCH01",
            "stage": "Lat Move",
            "risk": 0.92,
            "is_critical": True,
        },
        {
            "window": "t 0",
            "host": "WIN-DC01",
            "stage": "Exploit",
            "risk": 0.87,
            "is_critical": True,
        },
        {
            "window": "t +1",
            "host": "WIN-WS06",
            "stage": "Lat Move",
            "risk": 0.78,
            "is_critical": False,
        },
        {
            "window": "t +2",
            "host": "WIN-WS04",
            "stage": "Recon",
            "risk": 0.34,
            "is_critical": False,
        },
        {
            "window": "t +3",
            "host": "WIN-WS05",
            "stage": "Recon",
            "risk": 0.22,
            "is_critical": False,
        },
    ]

def get_feature_attributions():
    """
    Returns Feature Attribution data with uniform deep blue bar styling.
    """
    return [
        {"feature": "SMB Lateral Traffic Spikes", "score": 0.84, "color_class": "bar-inner-deepblue"},
        {"feature": "Failed Authentication Rate (Kerberos)", "score": 0.72, "color_class": "bar-inner-deepblue"},
        {"feature": "Anomalous Process Creation (lsass.exe)", "score": 0.56, "color_class": "bar-inner-deepblue"},
        {"feature": "Outbound Network Connections (Port 445)", "score": 0.41, "color_class": "bar-inner-deepblue"},
    ]

def get_model_comparison_metrics():
    """
    Returns metric values comparing World Model vs Baseline.
    """
    return {
        "world_model": {
            "name": "World Model (Proposed)",
            "lead_time": "3.2 windows",
            "fp_rate": "2.1%",
            "precision": "94.8%",
            "recall": "91.5%",
        },
        "baseline_model": {
            "name": "Baseline Model (Rule-Based)",
            "lead_time": "1.1 windows",
            "fp_rate": "8.4%",
            "precision": "71.2%",
            "recall": "68.0%",
        }
    }
