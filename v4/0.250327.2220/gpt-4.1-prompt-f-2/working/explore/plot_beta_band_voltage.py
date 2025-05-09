# This script loads the Beta_Band_Voltage data from the provided NWB file and creates a timeseries plot.
# It also prints basic statistics about the data.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Settings
url = "https://api.dandiarchive.org/api/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/download/"
fig_path = "explore/beta_band_voltage_timeseries.png"

# Load NWB
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract Beta Band Voltage data and timestamps
beta = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"]

data = beta.data[:]
timestamps = beta.timestamps[:]

print(f"Beta_Band_Voltage.data shape: {data.shape}")
print(f"Beta_Band_Voltage.timestamps shape: {timestamps.shape}")
print(f"Data mean: {data.mean():.4f}; std: {data.std():.4f}; min: {data.min():.4f}; max: {data.max():.4f}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(timestamps, data, color="tab:blue", lw=1)
plt.xlabel("Time (s)")
plt.ylabel("Beta Band Voltage (V)")
plt.title("Beta Band Voltage vs. Time")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
print(f"Saved plot to {fig_path}")