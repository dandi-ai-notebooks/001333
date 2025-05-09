# This script loads the Beta_Band_Voltage data from the specified NWB file and generates a histogram
# (distribution plot) of the signal values to show the spread and central tendency.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/download/"
fig_path = "explore/beta_band_voltage_histogram.png"

# Load NWB
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

beta = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"]

data = beta.data[:]
plt.figure(figsize=(6, 4))
plt.hist(data, bins=40, color='tab:purple', edgecolor='black')
plt.xlabel("Beta Band Voltage (V)")
plt.ylabel("Count")
plt.title("Beta Band Voltage Signal Histogram")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
print(f"Saved histogram plot to {fig_path}")