"""
Real data loader: turns a CICFlowMeter-schema CSV (the format CIC-IDS2017/
2018 ships in) into the same windowed (X, y_attack, y_stage) shapes that
data.py's synthetic generator produces -- so nothing in model.py, train.py,
or inference.py needs to change once real data is wired in.

Built and smoke-tested against a small SYNTHETIC csv that mimics the real
column names (see the bottom of this file) -- the actual dataset hasn't
been uploaded yet. Point `load_flow_csv` at a real file next. CICFlowMeter's
exact column naming drifts slightly across dataset releases (e.g. "Dst
Port" vs "Destination Port"), so the very first thing to do with a real
file is check `df.columns` against REQUIRED_COLUMNS below before trusting
anything downstream.

Known gap, by design: TTL variance and retransmit count are packet-level
features that don't exist in a flow-only CSV -- they're filled with 0
here (a genuinely empty/uninformative value, not a misleading one, since
it applies identically to every window) until the PCAP-based packet-level
step is wired in on top of this.
"""
import numpy as np
import pandas as pd

from mitre_mapping import LABEL_TO_STAGE, STAGE_TO_ID

WINDOW_SECONDS = 30

REQUIRED_COLUMNS = [
    "Src IP", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std",
    "SYN Flag Cnt", "ACK Flag Cnt",
    "Label",
]

FEATURE_COLS = [
    "flow_count", "unique_dst_ports", "unique_dst_hosts",
    "syn_ack_ratio", "iat_mean", "iat_var",
    "bytes_out", "bytes_in", "ttl_var", "retransmits",
]


def load_flow_csv(path):
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. "
            f"Actual columns in this file: {list(df.columns)}. "
            "CICFlowMeter column names vary slightly by dataset release -- "
            "update REQUIRED_COLUMNS and window_by_host() to match."
        )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    return df


def window_by_host(df, window_seconds=WINDOW_SECONDS):
    """Aggregates raw flow rows into one row per (host, time window)."""
    df = df.copy()
    df["window"] = df["Timestamp"].dt.floor(f"{window_seconds}s")

    rows = []
    for (src_ip, window), g in df.groupby(["Src IP", "window"]):
        syn = g["SYN Flag Cnt"].sum()
        ack = g["ACK Flag Cnt"].sum()
        rows.append({
            "src_ip": src_ip,
            "window": window,
            "flow_count": len(g),
            "unique_dst_ports": g["Dst Port"].nunique(),
            "unique_dst_hosts": g["Dst IP"].nunique(),
            "syn_ack_ratio": float(syn) / (float(ack) + 1e-6),
            "iat_mean": g["Flow IAT Mean"].mean(),
            "iat_var": g["Flow IAT Std"].pow(2).mean(),
            "bytes_out": g["TotLen Fwd Pkts"].sum(),
            "bytes_in": g["TotLen Bwd Pkts"].sum(),
            "ttl_var": 0.0,       # packet-level only -- see README Phase 2b
            "retransmits": 0.0,   # packet-level only -- see README Phase 2b
            "raw_labels": g["Label"].unique().tolist(),
        })
    return pd.DataFrame(rows).sort_values(["src_ip", "window"]).reset_index(drop=True)


def label_window(raw_labels):
    """A window is labeled with the first non-benign stage found among its
    flows (via mitre_mapping's LABEL_TO_STAGE), else benign. Labels mapped
    to None (DDoS/DoS -- see mitre_mapping.py) are skipped, not counted."""
    for label in raw_labels:
        stage = LABEL_TO_STAGE.get(label)
        if stage and stage != "benign":
            return STAGE_TO_ID[stage]
    return STAGE_TO_ID["benign"]


def build_sequences(windowed_df, seq_len=40):
    """Chops each host's window history into fixed-length sequences the
    model can consume. Hosts with fewer than seq_len windows are skipped
    for now -- padding can be added once we see real sequence-length
    distributions from an actual file."""
    X, y_stage = [], []
    for src_ip, g in windowed_df.groupby("src_ip"):
        g = g.sort_values("window")
        if len(g) < seq_len:
            continue
        for start in range(0, len(g) - seq_len + 1, seq_len):
            chunk = g.iloc[start:start + seq_len]
            X.append(chunk[FEATURE_COLS].fillna(0).values.astype(np.float32))
            y_stage.append(chunk["raw_labels"].apply(label_window).values)

    if not X:
        raise ValueError(
            f"No host had {seq_len} consecutive windows -- lower seq_len, "
            f"shrink WINDOW_SECONDS, or check that Timestamp parsed correctly."
        )

    X = np.stack(X)
    y_stage = np.stack(y_stage)
    y_attack = (y_stage != STAGE_TO_ID["benign"]).astype(np.float32)

    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std

    return X, y_attack, y_stage


def _make_synthetic_csv(path, n_flows=3000, seed=0):
    """Builds a tiny CSV matching REQUIRED_COLUMNS, purely to smoke-test
    the parsing/windowing logic above before a real file exists. Not
    meant to resemble real attack behavior the way data.py's generator
    does -- that's a different, deliberately more realistic synthetic set."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-02-28 09:00:00")
    rows = []
    for i in range(n_flows):
        ts = start + pd.Timedelta(seconds=int(rng.uniform(0, 3600)))
        is_attack = i > n_flows * 0.7 and i < n_flows * 0.85
        rows.append({
            "Src IP": "192.168.1.105",
            "Dst IP": f"10.0.0.{rng.integers(1, 200) if is_attack else rng.integers(1, 5)}",
            "Dst Port": int(rng.integers(1, 65535)) if is_attack else 443,
            "Protocol": 6,
            "Timestamp": ts,
            "Flow Duration": rng.uniform(10, 5000),
            "Tot Fwd Pkts": rng.integers(1, 20),
            "Tot Bwd Pkts": rng.integers(0, 3) if is_attack else rng.integers(5, 20),
            "TotLen Fwd Pkts": rng.uniform(40, 1000),
            "TotLen Bwd Pkts": rng.uniform(0, 200) if is_attack else rng.uniform(500, 5000),
            "Flow IAT Mean": rng.uniform(1, 50),
            "Flow IAT Std": rng.uniform(0.1, 10),
            "SYN Flag Cnt": 1,
            "ACK Flag Cnt": 0 if is_attack else 1,
            "Label": "PortScan" if is_attack else "BENIGN",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    import tempfile, os

    tmp_path = os.path.join(tempfile.gettempdir(), "synthetic_flow_mimic.csv")
    print(f"building a synthetic CICFlowMeter-schema csv at {tmp_path} to smoke-test parsing...\n")
    _make_synthetic_csv(tmp_path)

    df = load_flow_csv(tmp_path)
    print(f"loaded {len(df)} raw flow rows, columns check passed")

    windowed = window_by_host(df, window_seconds=30)
    print(f"windowed into {len(windowed)} (host, time-window) rows")
    print(windowed.head(3)[["src_ip", "window", "flow_count", "unique_dst_ports", "syn_ack_ratio"]])

    X, y_attack, y_stage = build_sequences(windowed, seq_len=10)
    print(f"\nbuilt sequences: X {X.shape}, y_attack {y_attack.shape}, y_stage {y_stage.shape}")
    print(f"attack windows found: {int(y_attack.sum())} / {y_attack.size}")
