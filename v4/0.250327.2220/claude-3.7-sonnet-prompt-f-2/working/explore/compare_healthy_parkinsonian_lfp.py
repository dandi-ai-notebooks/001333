"""
This script compares LFP signals between healthy and parkinsonian datasets
to visualize differences, particularly in the beta band (13-30 Hz) which is
mentioned as a biomarker for Parkinson's disease.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load a healthy LFP file
healthy_url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
healthy_remote_file = remfile.File(healthy_url)
healthy_h5_file = h5py.File(healthy_remote_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5_file)
healthy_nwb = healthy_io.read()

# Load a parkinsonian LFP file
parkinsonian_url = "https://api.dandiarchive.org/api/assets/5535e23a-9029-43c5-80fb-0fb596541a81/download/"
parkinsonian_remote_file = remfile.File(parkinsonian_url)
parkinsonian_h5_file = h5py.File(parkinsonian_remote_file)
parkinsonian_io = pynwb.NWBHDF5IO(file=parkinsonian_h5_file)
parkinsonian_nwb = parkinsonian_io.read()

# Get LFP data (only load first 10000 samples to keep processing manageable)
healthy_lfp = healthy_nwb.processing["ecephys"]["LFP"]["LFP"].data[0:10000]
parkinsonian_lfp = parkinsonian_nwb.processing["ecephys"]["LFP"]["LFP"].data[0:10000]

# Get sampling rate
sampling_rate = healthy_nwb.processing["ecephys"]["LFP"]["LFP"].rate

# Print basic information
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Healthy LFP shape: {healthy_lfp.shape}")
print(f"Parkinsonian LFP shape: {parkinsonian_lfp.shape}")
print(f"Duration of shown signal: {len(healthy_lfp) / sampling_rate:.2f} seconds")

# Plot time domain signals
plt.figure(figsize=(12, 6))
time_axis = np.arange(len(healthy_lfp)) / sampling_rate
plt.subplot(2, 1, 1)
plt.plot(time_axis, healthy_lfp)
plt.title('Healthy LFP Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')

plt.subplot(2, 1, 2)
plt.plot(time_axis, parkinsonian_lfp)
plt.title('Parkinsonian LFP Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')

plt.tight_layout()
plt.savefig('explore/lfp_time_domain.png')

# Compute and plot power spectra
def compute_psd(data, fs):
    f, pxx = signal.welch(data, fs, nperseg=2048)
    return f, pxx

healthy_f, healthy_pxx = compute_psd(healthy_lfp, sampling_rate)
parkinsonian_f, parkinsonian_pxx = compute_psd(parkinsonian_lfp, sampling_rate)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.semilogy(healthy_f, healthy_pxx)
plt.title('Power Spectrum of Healthy LFP')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.xlim(0, 100)  # Limit to 100 Hz for better visualization

# Highlight beta band (13-30 Hz)
plt.axvspan(13, 30, alpha=0.3, color='green', label='Beta Band')
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogy(parkinsonian_f, parkinsonian_pxx)
plt.title('Power Spectrum of Parkinsonian LFP')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.xlim(0, 100)  # Limit to 100 Hz

# Highlight beta band (13-30 Hz)
plt.axvspan(13, 30, alpha=0.3, color='green', label='Beta Band')
plt.legend()

plt.tight_layout()
plt.savefig('explore/lfp_power_spectra.png')

# Compare beta band power directly
beta_mask = (healthy_f >= 13) & (healthy_f <= 30)
healthy_beta_power = np.sum(healthy_pxx[beta_mask])
parkinsonian_beta_power = np.sum(parkinsonian_pxx[beta_mask])

print("\nBeta Band Power Comparison:")
print(f"Healthy Beta Power: {healthy_beta_power:.6e}")
print(f"Parkinsonian Beta Power: {parkinsonian_beta_power:.6e}")
print(f"Ratio (Parkinsonian/Healthy): {parkinsonian_beta_power/healthy_beta_power:.2f}")

# Create a bar chart comparing beta power
plt.figure(figsize=(8, 6))
plt.bar(['Healthy', 'Parkinsonian'], [healthy_beta_power, parkinsonian_beta_power])
plt.title('Beta Band Power Comparison')
plt.ylabel('Power in 13-30 Hz Band')
plt.savefig('explore/beta_power_comparison.png')