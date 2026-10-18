"""
Real data loader for CTU-13's Argus/NetFlow schema -- meaningfully
different from CICFlowMeter's schema (feature_pipeline.py), so it gets
its own file rather than overloading that one with two parsing paths.

Produces the same (X, y_attack, y_stage) shapes as everything else in
this project, built and verified against the actual uploaded
ctu13_scenario10.csv, not a synthetic mimic -- this is real data.

Known, deliberate approximations, stated plainly rather than hidden:
- syn_ack_ratio is derived from Argus's State string (e.g. "S_RA",
  "FSPA_FSPA", "CON"), not explicit flag counts like CICFlowMeter gives.
  Counting 'S' and 'A' characters in the state string is a reasonable
  proxy, not an exact equivalent.
- iat_mean/iat_var are computed between consecutive FLOW start times
  per host per window (Argus doesn't expose packet-level IAT the way
  CICFlowMeter does) -- this actually fits our purposes well, since
  regular, low-variance flow timing is exactly the beaconing signature
  we want to catch for Command & Control.
- ttl_var and retransmits stay 0.0 placeholders, same as
  feature_pipeline.py -- packet-level only, needs PCAP.

Real finding worth stating here too: the 10 infected hosts in this
scenario have essentially no pre-infection traffic AS THEMSELVES in this
capture (verified: one host had 9,579 botnet flows vs. 1 background
flow) -- these were pre-infected before the capture started. That rules
out a "clean host gets compromised mid-capture" lead-time test on this
scenario specifically. What this data DOES support: does the model
recognize C2 behavior on a held-out infected host it never trained on.
Split train/test by HOST, not by time, for exactly that reason.
"""
import numpy as np
import pandas as pd

from mitre_mapping import STAGE_TO_ID

WINDOW_SECONDS = 30

REQUIRED_COLUMNS = ["StartTime", "SrcAddr", "DstAddr", "Dport", "State",
                    "TotBytes", "SrcBytes"]

FEATURE_COLS = [
    "flow_count", "unique_dst_ports", "unique_dst_hosts",
    "syn_ack_ratio", "iat_mean", "iat_var",
    "bytes_out", "bytes_in", "ttl_var", "retransmits",
    "has_packet_coverage",
]

# this file ships an already-computed coarse_label (background/normal/
# botnet) -- reuse it rather than re-deriving from the 300+ granular
# raw Label strings ourselves.
COARSE_TO_STAGE = {
    "background": "benign",
    "normal": "benign",
    "botnet": "command_and_control",
}


def load_ctu13_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Actual: {list(df.columns)}")
    if "coarse_label" not in df.columns:
        df["coarse_label"] = "normal"
    df["StartTime"] = pd.to_datetime(df["StartTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["StartTime"])
    return df


def window_by_host(df, window_seconds=WINDOW_SECONDS, internal_prefix="147.32."):
    """Vectorized aggregation -- the original row-by-row Python loop timed
    out on this file's 151,256 unique SrcAddr values, almost all one-off
    external background hosts. Filtering to internal (147.32.*) hosts
    first cuts that to 413 while keeping all 10 infected hosts and 77%
    of the actual data volume."""
    df = df[df["SrcAddr"].str.startswith(internal_prefix)].copy()
    df["window"] = df["StartTime"].dt.floor(f"{window_seconds}s")
    df = df.sort_values(["SrcAddr", "window", "StartTime"])

    # per-flow gap to the previous flow within the same host+window --
    # vectorized stand-in for packet-level IAT (see module docstring)
    df["gap"] = df.groupby(["SrcAddr", "window"])["StartTime"].diff().dt.total_seconds()
    df["is_syn"] = df["State"].astype(str).str.contains("S")
    df["is_ack"] = df["State"].astype(str).str.contains("A")
    df["is_botnet"] = df["coarse_label"] == "botnet"

    agg = df.groupby(["SrcAddr", "window"]).agg(
        flow_count=("Dport", "size"),
        unique_dst_ports=("Dport", "nunique"),
        unique_dst_hosts=("DstAddr", "nunique"),
        syn_count=("is_syn", "sum"),
        ack_count=("is_ack", "sum"),
        bytes_out=("SrcBytes", "sum"),
        tot_bytes=("TotBytes", "sum"),
        iat_mean=("gap", "mean"),
        iat_var=("gap", "var"),
        any_botnet=("is_botnet", "any"),
    ).reset_index()

    agg["syn_ack_ratio"] = agg["syn_count"] / (agg["ack_count"] + 1e-6)
    agg["bytes_in"] = agg["tot_bytes"] - agg["bytes_out"]
    agg["iat_mean"] = agg["iat_mean"].fillna(0.0)
    agg["iat_var"] = agg["iat_var"].fillna(0.0)
    agg["ttl_var"] = 0.0
    agg["retransmits"] = 0.0
    agg["has_packet_coverage"] = 0.0  # set to 1.0 by merge_packet_features wherever real
    # packet data was actually merged in -- lets the model distinguish "confirmed zero"
    # from "no data", instead of both looking identical
    agg["stage"] = agg["any_botnet"].map(
        {True: STAGE_TO_ID["command_and_control"], False: STAGE_TO_ID["benign"]}
    )
    agg["date"] = agg["window"].dt.date
    return agg.rename(columns={"SrcAddr": "src_addr"}).sort_values(["src_addr", "window"]).reset_index(drop=True)


def build_sequences(windowed_df, seq_len=40):
    """Chunks each host's window history into fixed-length sequences.
    Groups by (host, DATE), not just host -- with multi-day data, the
    same physical host can be infected on several different days, and a
    sequence silently spanning two different capture days would blur
    exactly the boundary a day-based train/test split depends on."""
    X, y_stage, hosts, dates = [], [], [], []
    for (src_addr, date), g in windowed_df.groupby(["src_addr", "date"]):
        g = g.sort_values("window")
        if len(g) < seq_len:
            continue
        for start in range(0, len(g) - seq_len + 1, seq_len):
            chunk = g.iloc[start:start + seq_len]
            X.append(chunk[FEATURE_COLS].fillna(0).values.astype(np.float32))
            y_stage.append(chunk["stage"].values)
            hosts.append(src_addr)
            dates.append(date)

    if not X:
        raise ValueError(f"No (host, date) had {seq_len} consecutive windows.")

    X = np.stack(X)
    y_stage = np.stack(y_stage)
    y_attack = (y_stage != STAGE_TO_ID["benign"]).astype(np.float32)

    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std

    return X, y_attack, y_stage, np.array(hosts), np.array(dates), mean, std

    return X, y_attack, y_stage, np.array(hosts), np.array(dates)


if __name__ == "__main__":
    print("loading real ctu13_scenario10.csv...")
    df = load_ctu13_csv("/mnt/user-data/uploads/ctu13_scenario10.csv")
    print(f"loaded {len(df)} real flow rows")

    print("windowing by host...")
    windowed = window_by_host(df)
    print(f"windowed into {len(windowed)} (host, time-window) rows across {windowed['src_addr'].nunique()} hosts")

    X, y_attack, y_stage, hosts, dates, mean, std = build_sequences(windowed, seq_len=40)
    print(f"\nbuilt sequences: X {X.shape}, y_attack {y_attack.shape}")
    print(f"unique hosts with full sequences: {len(set(hosts))}")
    print(f"sequences containing any C2 activity: {int((y_attack.sum(axis=1) > 0).sum())} / {len(X)}")
