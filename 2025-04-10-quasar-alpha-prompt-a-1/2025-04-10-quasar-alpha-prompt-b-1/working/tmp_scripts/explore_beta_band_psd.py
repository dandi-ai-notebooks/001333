# This script loads a segment of the Beta Band Voltage from the NWB file and generates
# a power spectral density plot saved as PNG, to illustrate characteristic oscillatory frequencies.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile
from scipy.signal import welch

url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

es = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"]
data = es.data[:200]
timestamps = es.timestamps[:200]

# Estimate sampling rate from timestamps
dt = np.diff(timestamps)
fs = 1.0 / np.median(dt)

freqs, psd = welch(data, fs=fs, nperseg=min(128, len(data)))

plt.figure(figsize=(8, 4))
plt.semilogy(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Power Spectral Density of Beta Band Voltage (first 200 samples)')
plt.tight_layout()
plt.savefig('tmp_scripts/beta_band_psd.png')
plt.close()