# This script compares LFP data from different sessions of healthy and parkinsonian subjects
# We'll use different files than our previous comparisons

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set up plot style
plt.figure(figsize=(14, 10))

# ---------- Load a different healthy subject data ----------
healthy_url = "https://api.dandiarchive.org/api/assets/d92648ad-a2f8-4ec6-a125-363f45aa7f35/download/"
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

# ---------- Load a different parkinsonian subject data ----------
parkinsonian_url = "https://api.dandiarchive.org/api/assets/28fc91eb-eca2-4c8f-ba41-1cb7a38bcd50/download/"
parkinsonian_remote_file = remfile.File(parkinsonian_url)
parkinsonian_h5_file = h5py.File(parkinsonian_remote_file)
parkinsonian_io = pynwb.NWBHDF5IO(file=parkinsonian_h5_file)
parkinsonian_nwb = parkinsonian_io.read()

# Access the LFP data
parkinsonian_lfp_data = parkinsonian_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data

# Get a sample of the data (first 10,000 points)
parkinsonian_sample = parkinsonian_lfp_data[0:10000]

# Print subject information to verify we're looking at different subjects
print(f"Healthy Subject ID: {healthy_nwb.subject.subject_id}")
print(f"Parkinsonian Subject ID: {parkinsonian_nwb.subject.subject_id}")

# ---------- Plot the time series comparison ----------
plt.subplot(2, 1, 1)
plt.plot(time, healthy_sample, label='Healthy Subject', alpha=0.7)
plt.plot(time, parkinsonian_sample, label='Parkinsonian Subject', alpha=0.7)

# Highlight key differences with fill between
plt.fill_between(time, healthy_sample, parkinsonian_sample, where=(healthy_sample < parkinsonian_sample), 
                 interpolate=True, color='red', alpha=0.1)
plt.fill_between(time, healthy_sample, parkinsonian_sample, where=(healthy_sample >= parkinsonian_sample), 
                 interpolate=True, color='blue', alpha=0.1)

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
plt.plot(f_healthy, Pxx_healthy, label='Healthy Subject')
plt.plot(f_parkinsonian, Pxx_parkinsonian, label='Parkinsonian Subject')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Comparison: Power Spectral Density of Healthy vs. Parkinsonian LFP')
plt.xlim(0, 100)  # Limit to 0-100 Hz for visibility
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('explore/different_sessions_comparison.png')

# ---------- Compute and plot focused beta band comparison ----------
plt.figure(figsize=(10, 6))

# Focus on the beta band (13-30 Hz)
beta_mask = (f_healthy >= 13) & (f_healthy <= 30)
plt.plot(f_healthy[beta_mask], Pxx_healthy[beta_mask], label='Healthy Subject', linewidth=2)
plt.plot(f_parkinsonian[beta_mask], Pxx_parkinsonian[beta_mask], label='Parkinsonian Subject', linewidth=2)

# Shade the area where Parkinsonian > Healthy
diff = Pxx_parkinsonian[beta_mask] - Pxx_healthy[beta_mask]
plt.fill_between(f_healthy[beta_mask], Pxx_healthy[beta_mask], Pxx_parkinsonian[beta_mask], 
                 where=(Pxx_parkinsonian[beta_mask] > Pxx_healthy[beta_mask]), 
                 interpolate=True, color='red', alpha=0.3, label='Increased Beta Power in PD')

plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Beta Band (13-30 Hz) Power Comparison')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('explore/different_sessions_beta_comparison.png')