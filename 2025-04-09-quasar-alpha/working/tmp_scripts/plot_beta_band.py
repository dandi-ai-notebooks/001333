# This script loads the NWB file and creates a time series plot of the Beta Band Voltage
# It saves the figure to 'tmp_scripts/beta_band_timeseries.png'
# Useful to inspect the beta-band filtered LFP activity over time in one session

import remfile
import h5py
import pynwb
import matplotlib.pyplot as plt

# Load the NWB file over HTTP using remfile/h5py/pynwb
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
file = remfile.File(url)
f = h5py.File(file, 'r')
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

lfp_es = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"]
data = lfp_es.data[:]
timestamps = lfp_es.timestamps[:]

# Plot
plt.figure(figsize=(12, 4))
plt.plot(timestamps, data, lw=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Beta Band Voltage Time Series')
plt.tight_layout()
plt.savefig('tmp_scripts/beta_band_timeseries.png')
plt.close()