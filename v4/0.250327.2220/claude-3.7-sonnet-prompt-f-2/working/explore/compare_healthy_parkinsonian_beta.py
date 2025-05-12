"""
This script compares Beta Band Voltage signals between healthy and parkinsonian datasets
to visualize differences in the beta band (13-30 Hz) which is mentioned as a biomarker
for Parkinson's disease.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load a healthy beta file
healthy_url = "https://api.dandiarchive.org/api/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/download/"
healthy_remote_file = remfile.File(healthy_url)
healthy_h5_file = h5py.File(healthy_remote_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5_file)
healthy_nwb = healthy_io.read()

# Load a parkinsonian beta file
parkinsonian_url = "https://api.dandiarchive.org/api/assets/6b17c99d-19b9-4846-b1c9-671d9b187149/download/"
parkinsonian_remote_file = remfile.File(parkinsonian_url)
parkinsonian_h5_file = h5py.File(parkinsonian_remote_file)
parkinsonian_io = pynwb.NWBHDF5IO(file=parkinsonian_h5_file)
parkinsonian_nwb = parkinsonian_io.read()

# Get Beta Band Voltage data (load all data since it's only 1400 samples)
healthy_beta = healthy_nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"].data[:]
healthy_timestamps = healthy_nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"].timestamps[:]
parkinsonian_beta = parkinsonian_nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"].data[:]
parkinsonian_timestamps = parkinsonian_nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"].timestamps[:]

# Print basic information
print(f"Healthy Beta shape: {healthy_beta.shape}")
print(f"Parkinsonian Beta shape: {parkinsonian_beta.shape}")
print(f"Duration of healthy signal: {healthy_timestamps[-1] - healthy_timestamps[0]:.2f} seconds")
print(f"Duration of parkinsonian signal: {parkinsonian_timestamps[-1] - parkinsonian_timestamps[0]:.2f} seconds")

# Calculate basic statistics
print("\nBasic Statistics:")
print(f"Healthy Beta - Mean: {np.mean(healthy_beta):.6f}, Std: {np.std(healthy_beta):.6f}, Min: {np.min(healthy_beta):.6f}, Max: {np.max(healthy_beta):.6f}")
print(f"Parkinsonian Beta - Mean: {np.mean(parkinsonian_beta):.6f}, Std: {np.std(parkinsonian_beta):.6f}, Min: {np.min(parkinsonian_beta):.6f}, Max: {np.max(parkinsonian_beta):.6f}")

# Plot time domain signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(healthy_timestamps, healthy_beta)
plt.title('Healthy Beta Band Voltage Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')

plt.subplot(2, 1, 2)
plt.plot(parkinsonian_timestamps, parkinsonian_beta)
plt.title('Parkinsonian Beta Band Voltage Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')

plt.tight_layout()
plt.savefig('explore/beta_time_domain.png')

# Create a direct comparison of signals (normalized)
healthy_beta_norm = (healthy_beta - np.mean(healthy_beta)) / np.std(healthy_beta)
parkinsonian_beta_norm = (parkinsonian_beta - np.mean(parkinsonian_beta)) / np.std(parkinsonian_beta)

# Plot a section of the two signals for direct comparison
plt.figure(figsize=(12, 6))
section_length = 500  # Use a section of the data for clearer visualization
plt.plot(healthy_timestamps[:section_length], healthy_beta_norm[:section_length], label='Healthy (normalized)')
plt.plot(parkinsonian_timestamps[:section_length], parkinsonian_beta_norm[:section_length], label='Parkinsonian (normalized)')
plt.title('Comparison of Normalized Beta Band Voltage Signals')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('explore/beta_comparison.png')

# Calculate and plot the amplitude distribution (histogram)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(healthy_beta, bins=30, alpha=0.7, label='Healthy')
plt.title('Amplitude Distribution - Healthy')
plt.xlabel('Amplitude (V)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(parkinsonian_beta, bins=30, alpha=0.7, label='Parkinsonian')
plt.title('Amplitude Distribution - Parkinsonian')
plt.xlabel('Amplitude (V)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('explore/beta_amplitude_distribution.png')

# Calculate power
healthy_power = np.sum(healthy_beta**2) / len(healthy_beta)
parkinsonian_power = np.sum(parkinsonian_beta**2) / len(parkinsonian_beta)

print("\nSignal Power Comparison:")
print(f"Healthy Beta Power: {healthy_power:.6e}")
print(f"Parkinsonian Beta Power: {parkinsonian_power:.6e}")
print(f"Ratio (Parkinsonian/Healthy): {parkinsonian_power/healthy_power:.2f}")

# Create a bar chart comparing average power
plt.figure(figsize=(8, 6))
plt.bar(['Healthy', 'Parkinsonian'], [healthy_power, parkinsonian_power])
plt.title('Beta Band Signal Power Comparison')
plt.ylabel('Power (V²)')
plt.savefig('explore/beta_band_power_comparison.png')