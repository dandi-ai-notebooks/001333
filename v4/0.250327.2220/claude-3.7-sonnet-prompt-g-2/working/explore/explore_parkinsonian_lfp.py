# This script explores the LFP data from a parkinsonian subject
# We will load the data, plot the time series, and perform frequency analysis

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set up plot style
plt.figure(figsize=(12, 8))

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/6aa013b8-536c-4556-9730-94b71ae26c55/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the LFP data
lfp_data = nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data

# Get a sample of the data (first 10,000 points) to avoid loading too much data
sample_data = lfp_data[0:10000]

# Get the sampling rate
sampling_rate = nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].rate

# Create a time vector for plotting
time = np.arange(len(sample_data)) / sampling_rate

# Plot the LFP time series
plt.subplot(2, 1, 1)
plt.plot(time, sample_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title('Parkinsonian Subject LFP Time Series (First 5 seconds)')

# Perform frequency analysis using Welch's method
f, Pxx = signal.welch(sample_data, fs=sampling_rate, nperseg=1024)

# Plot the power spectral density
plt.subplot(2, 1, 2)
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Power Spectral Density of Parkinsonian Subject LFP')
plt.xlim(0, 100)  # Limit to 0-100 Hz for visibility

plt.tight_layout()
plt.savefig('explore/parkinsonian_lfp_analysis.png')