# This script explores the Beta_Band_Voltage data from healthy and parkinsonian subjects
# It compares the data between these two groups to visualize differences

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb

# URLs for one healthy and one parkinsonian beta file
healthy_beta_url = "https://api.dandiarchive.org/api/assets/945ddecb-afa5-4531-ad6a-ed92d5425817/download/"
parkinson_beta_url = "https://api.dandiarchive.org/api/assets/2ad9ddfe-e956-43c7-8f73-653250268865/download/"

# Load the files
healthy_file = remfile.File(healthy_beta_url)
healthy_h5 = h5py.File(healthy_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5)
healthy_nwb = healthy_io.read()

parkinson_file = remfile.File(parkinson_beta_url)
parkinson_h5 = h5py.File(parkinson_file)
parkinson_io = pynwb.NWBHDF5IO(file=parkinson_h5)
parkinson_nwb = parkinson_io.read()

# Get the Beta_Band_Voltage data
healthy_data = healthy_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data[:]
healthy_timestamps = healthy_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].timestamps[:]

parkinson_data = parkinson_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data[:]
parkinson_timestamps = parkinson_nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].timestamps[:]

print("Healthy Beta Data Shape:", healthy_data.shape)
print("Parkinsonian Beta Data Shape:", parkinson_data.shape)

print("\nHealthy Beta Data Statistics:")
print("Mean:", np.mean(healthy_data))
print("Std:", np.std(healthy_data))
print("Min:", np.min(healthy_data))
print("Max:", np.max(healthy_data))

print("\nParkinsonian Beta Data Statistics:")
print("Mean:", np.mean(parkinson_data))
print("Std:", np.std(parkinson_data))
print("Min:", np.min(parkinson_data))
print("Max:", np.max(parkinson_data))

# Plot the data
plt.figure(figsize=(12, 8))

# Time series plot
plt.subplot(2, 1, 1)
plt.plot(healthy_timestamps, healthy_data, label='Healthy', alpha=0.8)
plt.plot(parkinson_timestamps, parkinson_data, label='Parkinsonian', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Beta Band Voltage: Healthy vs Parkinsonian')
plt.legend()
plt.grid(True)

# Frequency domain comparison
plt.subplot(2, 1, 2)
healthy_fft = np.abs(np.fft.rfft(healthy_data))
parkinson_fft = np.abs(np.fft.rfft(parkinson_data))

# Create frequency axis (assuming same sampling rate for both)
sample_spacing = np.mean(np.diff(healthy_timestamps))
freqs = np.fft.rfftfreq(len(healthy_data), sample_spacing)

# Only plot frequencies up to 50 Hz
max_freq_idx = np.searchsorted(freqs, 50)
plt.plot(freqs[:max_freq_idx], healthy_fft[:max_freq_idx], label='Healthy', alpha=0.8)
plt.plot(freqs[:max_freq_idx], parkinson_fft[:max_freq_idx], label='Parkinsonian', alpha=0.8)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum of Beta Band Signals')
plt.axvspan(13, 30, color='yellow', alpha=0.2, label='Beta Band (13-30 Hz)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('tmp_scripts/beta_comparison.png')