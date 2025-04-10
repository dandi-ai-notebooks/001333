# This script explores the LFP data from healthy and parkinsonian subjects
# It loads a subset of the data to avoid memory issues when streaming from remote

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb
from scipy import signal

# URLs for one healthy and one parkinsonian LFP file
healthy_lfp_url = "https://api.dandiarchive.org/api/assets/5ee1bca7-179f-4ce4-b6a1-6f767caf496a/download/"
parkinson_lfp_url = "https://api.dandiarchive.org/api/assets/e1a67d80-9f06-4e36-8630-ee5e8e023845/download/"

# Load the files
healthy_file = remfile.File(healthy_lfp_url)
healthy_h5 = h5py.File(healthy_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5)
healthy_nwb = healthy_io.read()

parkinson_file = remfile.File(parkinson_lfp_url)
parkinson_h5 = h5py.File(parkinson_file)
parkinson_io = pynwb.NWBHDF5IO(file=parkinson_h5)
parkinson_nwb = parkinson_io.read()

# Get metadata
healthy_lfp_rate = healthy_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].rate
parkinson_lfp_rate = parkinson_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].rate

print(f"Healthy LFP sampling rate: {healthy_lfp_rate} Hz")
print(f"Parkinsonian LFP sampling rate: {parkinson_lfp_rate} Hz")

# Get data shape without loading everything
healthy_data_shape = healthy_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].data.shape
parkinson_data_shape = parkinson_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].data.shape

print(f"Healthy LFP data shape: {healthy_data_shape}")
print(f"Parkinsonian LFP data shape: {parkinson_data_shape}")

# Load only a subset of the data (2 seconds worth) to avoid memory issues
# Assuming 2000 Hz sampling rate, 2 seconds = 4000 samples
subset_size = 4000
start_idx = 0  # Start from the beginning

# Load subsets
healthy_lfp_subset = healthy_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].data[start_idx:start_idx+subset_size]
parkinson_lfp_subset = parkinson_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].data[start_idx:start_idx+subset_size]

print("\nLFP Subset Statistics:")
print("Healthy LFP mean:", np.mean(healthy_lfp_subset))
print("Healthy LFP std:", np.std(healthy_lfp_subset))
print("Parkinsonian LFP mean:", np.mean(parkinson_lfp_subset))
print("Parkinsonian LFP std:", np.std(parkinson_lfp_subset))

# Create time axis for the subset (in seconds)
time_subset = np.arange(subset_size) / healthy_lfp_rate

# Plot the LFP data subset
fig = plt.figure(figsize=(12, 10))

# Time series of LFP data
plt.subplot(2, 1, 1)
plt.plot(time_subset, healthy_lfp_subset, label='Healthy', alpha=0.8)
plt.plot(time_subset, parkinson_lfp_subset, label='Parkinsonian', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('LFP Time Series (2-second subset): Healthy vs Parkinsonian')
plt.legend()
plt.grid(True)

# Compute spectrograms for detailed frequency analysis over time
plt.subplot(2, 1, 2)

# Define the frequency range of interest (0-100 Hz) and parameters for spectrogram
fs = healthy_lfp_rate
nperseg = 256  # Window length for STFT
noverlap = 200  # Overlap between windows
nfft = 1024  # Length of FFT

# Compute spectrogram for both signals
f_h, t_h, Sxx_h = signal.spectrogram(healthy_lfp_subset, fs=fs, nperseg=nperseg, 
                                     noverlap=noverlap, nfft=nfft)
f_p, t_p, Sxx_p = signal.spectrogram(parkinson_lfp_subset, fs=fs, nperseg=nperseg, 
                                     noverlap=noverlap, nfft=nfft)

# Calculate ratio of PD to Healthy to highlight differences
# Adding small constant to avoid division by zero
ratio = np.log10(Sxx_p / (Sxx_h + 1e-15))

# Plot the log ratio (PD/Healthy)
# Limit to frequencies below 100Hz
freq_limit = 100
freq_idx = np.where(f_h <= freq_limit)[0]

plt.pcolormesh(t_p, f_h[freq_idx], ratio[freq_idx, :], cmap='RdBu_r', vmin=-2, vmax=2)
plt.colorbar(label='Log10 Power Ratio (PD/Healthy)')
plt.axhline(y=13, color='black', linestyle='--', alpha=0.7, label='Beta Band Start (13 Hz)')
plt.axhline(y=30, color='black', linestyle='--', alpha=0.7, label='Beta Band End (30 Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram Power Ratio (PD vs Healthy)')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('tmp_scripts/lfp_comparison.png')

# Create an additional figure focusing on power spectral density
plt.figure(figsize=(10, 6))

# Compute power spectral density
freq_h, psd_h = signal.welch(healthy_lfp_subset, fs=fs, nperseg=1024, scaling='spectrum')
freq_p, psd_p = signal.welch(parkinson_lfp_subset, fs=fs, nperseg=1024, scaling='spectrum')

# Focus on 0-100 Hz range
freq_limit = 100
freq_idx = np.where(freq_h <= freq_limit)[0]

# Plot PSD
plt.semilogy(freq_h[freq_idx], psd_h[freq_idx], label='Healthy', alpha=0.8)
plt.semilogy(freq_p[freq_idx], psd_p[freq_idx], label='Parkinsonian', alpha=0.8)
plt.axvspan(13, 30, color='yellow', alpha=0.2, label='Beta Band (13-30 Hz)')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title('Power Spectral Density: Healthy vs Parkinsonian')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_psd_comparison.png')