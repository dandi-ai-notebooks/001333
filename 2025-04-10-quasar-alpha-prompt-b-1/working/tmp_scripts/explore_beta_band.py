# This script loads a segment of Beta Band Voltage data and timestamps from the NWB file remotely,
# and generates a time series plot saved as PNG for review purposes.

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

es = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"]

# Load first 200 points for quick exploration
data = es.data[:200]
timestamps = es.timestamps[:200]

plt.figure(figsize=(10, 4))
plt.plot(timestamps, data)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Beta Band Voltage (first 200 samples)')
plt.tight_layout()
plt.savefig('tmp_scripts/beta_band_timeseries.png')
plt.close()