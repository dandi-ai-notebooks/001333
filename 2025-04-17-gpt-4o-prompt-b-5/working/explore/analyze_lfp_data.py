# This script is for exploring the LFP data within the chosen NWB file.
# It will extract and visualize LFP data from the electrical series
# and will save plots of the data visualization as PNG files, without showing them.

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

# Load the NWB file from the remote URL
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwbfile = io.read()

# Extract LFP data from the specified electrical series
lfp_data = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data[:]
timestamps = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].timestamps[:]

# Create a plot of the LFP data
plt.figure(figsize=(10, 4))
plt.plot(timestamps[:1000], lfp_data[:1000], label="LFP Beta Band Voltage")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("LFP Beta Band Voltage")
plt.legend()

# Save plot as a PNG file
plt.savefig('explore/lfp_beta_band_voltage.png')
plt.close()