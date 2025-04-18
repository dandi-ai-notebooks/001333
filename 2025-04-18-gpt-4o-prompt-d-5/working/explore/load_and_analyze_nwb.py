# This script loads the NWB file from the remote server, extracts Beta_Band_Voltage data from it,
# and generates a time series plot that is saved to an image file for further analysis.

import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile
import numpy as np

# Load the remote NWB file
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
nwb_io = pynwb.NWBHDF5IO(file=h5_file)
nwb = nwb_io.read()

# Accessing Beta_Band_Voltage data
beta_band_voltage = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"]

# Fetching data and timestamps
data = beta_band_voltage.data[:]
timestamps = beta_band_voltage.timestamps[:]

# Plotting the Beta Band Voltage over time
plt.figure(figsize=(10, 4))
plt.plot(timestamps, data)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Beta Band Voltage Over Time')
plt.grid(True)
plt.savefig('explore/beta_band_voltage_plot.png')
plt.close()

# Release resources
nwb_io.close()
h5_file.close()