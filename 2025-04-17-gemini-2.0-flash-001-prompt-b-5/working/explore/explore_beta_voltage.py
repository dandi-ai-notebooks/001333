import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to load and plot Beta_Band_Voltage data from an NWB file

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the Beta_Band_Voltage data
data = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data
timestamps = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].timestamps

# Get the first 1000 data points
num_points = 1000
data_subset = data[:num_points]
timestamps_subset = timestamps[:num_points]

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(timestamps_subset, data_subset)
plt.xlabel("Time (s)")
plt.ylabel("Beta Band Voltage (V)")
plt.title("Beta Band Voltage Over Time")
plt.savefig("explore/beta_voltage.png")
plt.close()

print("Beta Band Voltage plot created successfully in explore/beta_voltage.png")