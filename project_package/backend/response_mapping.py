"""
Stage -> recommended first response. Advisory only — this is read by a
human analyst through the dashboard, it never triggers an automated
network action. That distinction is deliberate: a forecasting system
that's occasionally wrong is fine; one that's occasionally wrong AND
takes autonomous action on your network is a different, much riskier
system, and it's not what this PS asked for.

Actions are drawn from MITRE ATT&CK's own published Mitigations
(attack.mitre.org/mitigations/) for the relevant tactic, not invented.
"""

STAGE_TO_RESPONSE = {
    "benign": [],
    "reconnaissance": [
        "Rate-limit or temporarily block the scanning source IP",
        "Tighten inbound firewall rules on the probed ports/hosts",
        "Consider a decoy/honeypot on the pattern being scanned (MITRE M1030)",
    ],
    "lateral_movement": [
        "Isolate the affected host segment from the rest of the network (MITRE M1030 — Network Segmentation)",
        "Force credential rotation for accounts active on the affected hosts",
        "Require MFA for lateral admin protocols (SMB/RDP/WinRM) if not already enforced (MITRE M1032)",
        "Restrict SMB/RDP between the affected segment and the rest of the network",
    ],
    "command_and_control": [
        "Block the destination IP/domain at the firewall or DNS sinkhole (MITRE M1031)",
        "Isolate the beaconing host from the network",
        "Capture a forensic snapshot of the host BEFORE remediation, not after",
        "Kill the beaconing process once the snapshot is captured",
    ],
    "exfiltration": [
        "Block outbound traffic to the flagged destination immediately",
        "Enforce DLP / rate-limit egress from the affected host (MITRE M1057)",
        "Snapshot the affected data stores for forensics and scope assessment",
        "Notify data-protection / compliance stakeholders per incident policy",
    ],
}
