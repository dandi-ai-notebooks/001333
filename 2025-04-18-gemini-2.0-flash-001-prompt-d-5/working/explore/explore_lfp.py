import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to explore LFP data from the NWB file and generate a plot.

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the LFP data
lfp_data = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data
lfp_timestamps = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].timestamps

# Select a subset of the data for plotting
num_samples = 1000
start_index = 0
end_index = start_index + num_samples
lfp_data_subset = lfp_data[start_index:end_index]
lfp_timestamps_subset = lfp_timestamps[start_index:end_index]

# Create a time series plot of the LFP data
plt.figure(figsize=(10, 6))
plt.plot(lfp_timestamps_subset, lfp_data_subset)
plt.xlabel("Time (s)")
plt.ylabel("Beta Band Voltage (V)")
plt.title("LFP Data - Beta Band Voltage")
plt.grid(True)

# Save the plot to a file
plt.savefig("explore/lfp_plot.png")
print("LFP plot saved to explore/lfp_plot.png")