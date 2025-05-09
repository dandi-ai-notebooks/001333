# Script to load a subset of LFP data and plot it

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/5409700b-e080-44e6-a6db-1d3e8890cd6c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data
lfp_data = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].data
rate = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].rate

# Load a subset of data (e.g., first 50000 samples)
subset_size = 50000
lfp_subset = lfp_data[0:subset_size]
timestamps = np.arange(subset_size) / rate

# Get electrode info
electrodes_df = nwb.electrodes.to_dataframe()

# Plot the LFP subset
plt.figure(figsize=(12, 6))
plt.plot(timestamps, lfp_subset)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (volts)")
plt.title("Subset of LFP Data")
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig("explore/lfp_subset_plot.png")
plt.close()

io.close()