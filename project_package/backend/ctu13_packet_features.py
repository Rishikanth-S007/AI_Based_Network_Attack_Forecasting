"""
Extracts real packet-level features -- TTL variance, retransmit count --
from PCAP file(s), replacing the 0.0 placeholders that ctu13_pipeline.py's
window_by_host has been using for these two features.

Handles both classic pcap and pcapng formats (auto-detected by magic
bytes -- we've hit both from different sources in this project) and
accepts either a single path or a list of paths to be treated as one
continuous stream (for captures split into parts to fit an upload limit).

Uses hand-rolled byte-offset parsing instead of full dpkt object
construction -- meaningfully faster once processing millions of packets,
which mattered once Scenario 10's flood traffic pushed runtime past a
single-call time budget.

Retransmission detection: a TCP packet counts as a retransmission if the
exact same (src, sport, dst, dport, seq) combination has been seen
before. Simple heuristic, not a full TCP state machine -- stated plainly
rather than hidden, same as every other approximation in this project.
"""
import socket
from collections import defaultdict

import numpy as np
import pandas as pd
import dpkt

WINDOW_SECONDS = 30


def ip_to_str(addr):
    return socket.inet_ntoa(addr)


def _open_reader(f):
    magic = f.read(4)
    f.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        return dpkt.pcapng.Reader(f)
    return dpkt.pcap.Reader(f)


def _fast_parse(buf):
    """Hand-rolled header parsing via direct byte offsets instead of
    constructing full dpkt.ethernet/ip/tcp objects. Only looks at the
    handful of header bytes we actually need."""
    if len(buf) < 34:
        return None
    eth_type = buf[12] << 8 | buf[13]
    if eth_type != 0x0800:  # IPv4 only
        return None
    ip_start = 14
    ihl = (buf[ip_start] & 0x0F) * 4
    ttl = buf[ip_start + 8]
    proto = buf[ip_start + 9]
    src = buf[ip_start + 12:ip_start + 16]
    result = {"ttl": ttl, "src": src, "proto": proto}
    if proto == 6:  # TCP
        tcp_start = ip_start + ihl
        if len(buf) >= tcp_start + 8:
            sport = buf[tcp_start] << 8 | buf[tcp_start + 1]
            dport = buf[tcp_start + 2] << 8 | buf[tcp_start + 3]
            seq = int.from_bytes(buf[tcp_start + 4:tcp_start + 8], "big")
            dst = buf[ip_start + 16:ip_start + 20]
            result.update({"sport": sport, "dport": dport, "seq": seq, "dst": dst})
    return result


def extract_packet_features(pcap_paths, window_seconds=WINDOW_SECONDS):
    """pcap_paths: a single path, or a list of paths treated as ONE
    continuous stream in order. Using a list matters for correctness,
    not just convenience -- retransmit detection shares a `seen_seqs`
    set across the whole call, so a retransmission split across two
    part-files is only caught correctly if both parts share that state."""
    if isinstance(pcap_paths, str):
        pcap_paths = [pcap_paths]

    seen_seqs = set()
    ttl_by_window = defaultdict(list)
    retrans_by_window = defaultdict(int)
    n_packets = 0
    n_tcp = 0

    for path in pcap_paths:
        with open(path, "rb") as f:
            reader = _open_reader(f)
            for ts, buf in reader:
                n_packets += 1
                parsed = _fast_parse(buf)
                if parsed is None:
                    continue
                src = ip_to_str(parsed["src"])
                window = pd.Timestamp(ts, unit="s").floor(f"{window_seconds}s")
                ttl_by_window[(src, window)].append(parsed["ttl"])

                if "seq" in parsed:
                    n_tcp += 1
                    key = (src, parsed["sport"], ip_to_str(parsed["dst"]), parsed["dport"], parsed["seq"])
                    if key in seen_seqs:
                        retrans_by_window[(src, window)] += 1
                    else:
                        seen_seqs.add(key)
        print(f"  ...processed {path.split('/')[-1]}, running total: {n_packets} packets")

    rows = []
    for (src, window), ttls in ttl_by_window.items():
        rows.append({
            "src_addr": src,
            "window": window,
            "ttl_var": float(np.var(ttls)) if len(ttls) > 1 else 0.0,
            "retransmits": retrans_by_window.get((src, window), 0),
        })
    df = pd.DataFrame(rows)
    print(f"  parsed {n_packets} packets total ({n_tcp} TCP) across {len(pcap_paths)} file(s), "
          f"{len(df)} (host, window) combinations with packet coverage")
    return df


def merge_packet_features(windowed_df, packet_df):
    """Left-merge real packet features into the existing flow-level
    windowed dataframe, on (src_addr, window).

    Two corrections applied here, both added after the first full run
    showed merging raw packet features made results WORSE, not better:
    - retransmits is log1p-scaled before merging. A single observed
      window with 441 retransmits (vs. a typical 0-5) was dominating the
      dataset-wide normalization and compressing every other value's
      scale into near-indistinguishability. log1p keeps the signal
      (441 still reads as clearly elevated) without one outlier warping
      everything else.
    - has_packet_coverage is set to 1.0 for rows that actually got a
      real match here, 0.0 otherwise -- coverage is only ~10% of rows
      (we only ever captured packets for the known infected hosts, never
      background traffic), so without this flag "ttl_var/retransmits are
      0" is ambiguous between "confirmed clean" and "we have no idea."
    """
    packet_df = packet_df.copy()
    packet_df["retransmits"] = np.log1p(packet_df["retransmits"])
    packet_df["ttl_var"] = np.log1p(packet_df["ttl_var"])

    merged = windowed_df.merge(
        packet_df, on=["src_addr", "window"], how="left", suffixes=("", "_real")
    )
    coverage = merged["ttl_var_real"].notna().sum()
    print(f"  packet-level coverage: {coverage}/{len(merged)} rows in the flow-level "
          f"windowed data ({coverage/len(merged)*100:.1f}%)")
    merged["has_packet_coverage"] = merged["ttl_var_real"].notna().astype(float)
    merged["ttl_var"] = merged["ttl_var_real"].fillna(merged["ttl_var"])
    merged["retransmits"] = merged["retransmits_real"].fillna(merged["retransmits"])
    return merged.drop(columns=["ttl_var_real", "retransmits_real"])


def _make_synthetic_pcap(path):
    with open(path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        base_ts = 1000000.0
        packets = [
            {"ttl": 64, "seq": 100, "sport": 12345},
            {"ttl": 63, "seq": 101, "sport": 12345},
            {"ttl": 65, "seq": 101, "sport": 12345},
            {"ttl": 64, "seq": 102, "sport": 12345},
            {"ttl": 40, "seq": 103, "sport": 12345},
        ]
        for i, p in enumerate(packets):
            tcp = dpkt.tcp.TCP(sport=p["sport"], dport=80, seq=p["seq"], flags=0x18)
            tcp.data = b""
            ip = dpkt.ip.IP(src=socket.inet_aton("147.32.84.165"),
                             dst=socket.inet_aton("8.8.8.8"), ttl=p["ttl"], p=6)
            ip.data = tcp
            ip.len = len(bytes(ip))
            eth = dpkt.ethernet.Ethernet(
                src=b"\x00" * 6, dst=b"\x00" * 6, type=dpkt.ethernet.ETH_TYPE_IP
            )
            eth.data = ip
            writer.writepkt(bytes(eth), ts=base_ts + i)


if __name__ == "__main__":
    import tempfile, os
    tmp_path = os.path.join(tempfile.gettempdir(), "synthetic_test.pcap")
    print(f"building a synthetic pcap at {tmp_path} to smoke-test parsing...\n")
    _make_synthetic_pcap(tmp_path)
    df = extract_packet_features(tmp_path)
    print()
    print(df.to_string())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["retransmits"] == 1, f"expected 1 retransmit, got {row['retransmits']}"
    assert row["ttl_var"] > 0, "expected nonzero TTL variance"
    print("\nsmoke test passed")
