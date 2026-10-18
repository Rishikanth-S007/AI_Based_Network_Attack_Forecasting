import pandas as pd
import sys

# Point to your backend folder
sys.path.insert(0, "project_package/backend")
from ctu13_pipeline import window_by_host

# Load the demo slice
print("Loading demo slice...")
df_slice = pd.read_csv("capture20110811_demo_slice.csv")

# FIX 1: Convert time
df_slice['StartTime'] = pd.to_datetime(df_slice['StartTime'])

# FIX 2: Inject the missing column the pipeline expects
df_slice['coarse_label'] = 'normal'

# Run the windowing function
print("Windowing traffic...")
windowed = window_by_host(df_slice)

# Isolate .59 and sort by time window
ip_col = 'host' if 'host' in windowed.columns else 'src_addr'
host_59_windows = windowed[windowed[ip_col] == '147.32.84.59'].sort_values('window')

# Print the features
print("\n--- The Exact Features Fed to the LSTM ---")
print(host_59_windows[['window', 'flow_count', 'unique_dst_ports', 'unique_dst_hosts', 'syn_ack_ratio']].head(10).to_string())