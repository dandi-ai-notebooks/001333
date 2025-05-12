# This script compares LFP data from both healthy and parkinsonian subjects
# We will plot both signals on the same plot for direct comparison

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set up plot style
plt.figure(figsize=(14, 10))

# ---------- Load the healthy subject data ----------
healthy_url = "https://api.dandiarchive.org/api/assets/3cf468a1-4933-4aa6-b1c3-4a261b3ee6db/download/"
healthy_remote_file = remfile.File(healthy_url)
healthy_h5_file = h5py.File(healthy_remote_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5_file)
healthy_nwb = healthy_io.read()

# Access the LFP data
healthy_lfp_data = healthy_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data

# Get a sample of the data (first 10,000 points)
healthy_sample = healthy_lfp_data[0:10000]

# Get the sampling rate
sampling_rate = healthy_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].rate

# Create a time vector for plotting
time = np.arange(len(healthy_sample)) / sampling_rate

# ---------- Load the parkinsonian subject data ----------
parkinsonian_url = "https://api.dandiarchive.org/api/assets/6aa013b8-536c-4556-9730-94b71ae26c55/download/"
parkinsonian_remote_file = remfile.File(parkinsonian_url)
parkinsonian_h5_file = h5py.File(parkinsonian_remote_file)
parkinsonian_io = pynwb.NWBHDF5IO(file=parkinsonian_h5_file)
parkinsonian_nwb = parkinsonian_io.read()

# Access the LFP data
parkinsonian_lfp_data = parkinsonian_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data

# Get a sample of the data (first 10,000 points)
parkinsonian_sample = parkinsonian_lfp_data[0:10000]

# ---------- Plot the time series comparison ----------
plt.subplot(2, 1, 1)
plt.plot(time, healthy_sample, label='Healthy Subject', alpha=0.7)
plt.plot(time, parkinsonian_sample, label='Parkinsonian Subject', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title('Comparison: Healthy vs. Parkinsonian LFP Time Series (First 5 seconds)')
plt.legend()

# ---------- Compute and plot frequency analysis ----------
# Compute Welch's PSD for both signals
f_healthy, Pxx_healthy = signal.welch(healthy_sample, fs=sampling_rate, nperseg=1024)
f_parkinsonian, Pxx_parkinsonian = signal.welch(parkinsonian_sample, fs=sampling_rate, nperseg=1024)

# Plot the power spectral density comparison
plt.subplot(2, 1, 2)
plt.semilogy(f_healthy, Pxx_healthy, label='Healthy Subject')
plt.semilogy(f_parkinsonian, Pxx_parkinsonian, label='Parkinsonian Subject')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Comparison: Power Spectral Density of Healthy vs. Parkinsonian LFP')
plt.xlim(0, 100)  # Limit to 0-100 Hz for visibility
plt.legend()

plt.tight_layout()
plt.savefig('explore/healthy_parkinsonian_comparison.png')

# ---------- Compute and plot focused beta band comparison ----------
plt.figure(figsize=(10, 6))

# Focus on the beta band (13-30 Hz)
beta_mask = (f_healthy >= 13) & (f_healthy <= 30)
plt.plot(f_healthy[beta_mask], Pxx_healthy[beta_mask], label='Healthy Subject', linewidth=2)
plt.plot(f_parkinsonian[beta_mask], Pxx_parkinsonian[beta_mask], label='Parkinsonian Subject', linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Beta Band (13-30 Hz) Power Comparison')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('explore/beta_band_comparison.png')