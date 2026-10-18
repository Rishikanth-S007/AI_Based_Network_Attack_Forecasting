"""Adapters for raw dashboard uploads.

This module deliberately delegates CTU-13 parsing, windowing, and packet
feature extraction to the existing backend pipeline.
"""
from pathlib import Path

import pandas as pd
import dpkt

from ctu13_packet_features import extract_packet_features, merge_packet_features
from ctu13_pipeline import FEATURE_COLS, load_ctu13_csv, window_by_host


RAW_NETFLOW_COLUMNS = [
    "StartTime", "SrcAddr", "DstAddr", "Dport", "State", "TotBytes", "SrcBytes"
]


def _sort_windowed(frame):
    """Keep upload rows in chronological window order for dashboard validation."""
    if "window" not in frame.columns:
        return frame
    windowed_df = frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(windowed_df["window"]):
        parsed_window = pd.to_datetime(windowed_df["window"], errors="coerce")
        if not parsed_window.isna().any():
            windowed_df["window"] = parsed_window
    return windowed_df.sort_values("window").reset_index(drop=True)


def process_raw_upload(file_path, file_type):
    """Return ``(windowed_dataframe, error_message)`` for one upload.

    PCAP-only uploads are parsed to validate the capture, but are rejected
    because packet extraction provides only the two packet-level features;
    the model also requires nine flow-level features.
    """
    normalized_type = str(file_type).lower().lstrip(".")
    path = Path(file_path) if isinstance(file_path, (str, Path)) else None

    try:
        if normalized_type == "csv":
            frame = pd.read_csv(path)
            found = list(frame.columns)
            if set(FEATURE_COLS).issubset(frame.columns):
                return _sort_windowed(frame), None
            if set(RAW_NETFLOW_COLUMNS).issubset(frame.columns):
                return _sort_windowed(window_by_host(load_ctu13_csv(path))), None
            return None, (
                "CSV schema not recognized. Expected either the 11 model feature "
                f"columns {FEATURE_COLS} or raw CTU-13 columns {RAW_NETFLOW_COLUMNS}; "
                f"found {found}."
            )

        if normalized_type in {"pcap", "pcapng"}:
            companion_csv = None
            if isinstance(file_path, (tuple, list)):
                if len(file_path) != 2:
                    return None, "A PCAP upload may have at most one companion NetFlow CSV."
                pcap_path, companion_csv = map(Path, file_path)
                path = pcap_path
            packet_df = extract_packet_features(str(path))
            if packet_df.empty:
                return None, (
                    "PCAP parsing found no IPv4 packet windows. PCAP-only inference "
                    "cannot run because flow-level features are unavailable."
                )
            if companion_csv is not None:
                try:
                    flow_df = load_ctu13_csv(companion_csv)
                except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    return None, f"Could not process companion NetFlow CSV: {exc}"
                windowed_df = merge_packet_features(window_by_host(flow_df), packet_df)
                return _sort_windowed(windowed_df), None
            return None, (
                "PCAP parsed successfully, but PCAP-only inference cannot run: "
                "packet data provides only ttl_var and retransmits, while the model "
                "also requires the nine flow-level features. Upload a matching "
                "CTU-13 NetFlow CSV with the capture."
            )

        return None, "Unsupported upload type. Expected .csv, .pcap, or .pcapng."
    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        dpkt.dpkt.NeedData,
        dpkt.dpkt.UnpackError,
    ) as exc:
        return None, f"Could not process upload: {exc}"
