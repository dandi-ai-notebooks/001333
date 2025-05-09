# This script computes and plots the power spectral density (PSD) of the first 10,000 samples of LFP data.
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

LFP = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
data = LFP.data[:10000]
rate = LFP.rate

# Welch's method for power spectral density
f, Pxx = welch(data, fs=rate, nperseg=2048)

plt.figure(figsize=(8, 5))
plt.semilogy(f, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V²/Hz)")
plt.title("Power Spectral Density of First 10,000 LFP Samples")
plt.tight_layout()
plt.savefig("explore/plot_lfp_psd.png")
plt.close()

io.close()
h5_file.close()
remote_file.close()