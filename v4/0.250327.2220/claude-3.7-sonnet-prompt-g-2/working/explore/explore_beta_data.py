# This script explores the Beta ARV data from both healthy and parkinsonian subjects
# to understand the structure and characteristics of the beta data

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set up plot style
plt.figure(figsize=(14, 10))

# ---------- Load healthy beta data ----------
healthy_beta_url = "https://api.dandiarchive.org/api/assets/73214862-df4b-452b-a35c-d1f3bdb68180/download/"
healthy_remote_file = remfile.File(healthy_beta_url)
healthy_h5_file = h5py.File(healthy_remote_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5_file)
healthy_beta_nwb = healthy_io.read()

# Access the Beta ARV data (convert h5py dataset to numpy array with [:])
healthy_beta_data = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].data[:]
healthy_beta_timestamps = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].timestamps[:]

print(f"Healthy Beta ARV Data Shape: {healthy_beta_data.shape}")
print(f"Healthy Beta ARV Timestamps Shape: {healthy_beta_timestamps.shape}")

# ---------- Load parkinsonian beta data ----------
parkinsonian_beta_url = "https://api.dandiarchive.org/api/assets/712fd6c0-5543-476d-9493-7bdb652acdf2/download/"
parkinsonian_remote_file = remfile.File(parkinsonian_beta_url)
parkinsonian_h5_file = h5py.File(parkinsonian_remote_file)
parkinsonian_io = pynwb.NWBHDF5IO(file=parkinsonian_h5_file)
parkinsonian_beta_nwb = parkinsonian_io.read()

# Access the Beta ARV data (convert h5py dataset to numpy array with [:])
parkinsonian_beta_data = parkinsonian_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].data[:]
parkinsonian_beta_timestamps = parkinsonian_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].timestamps[:]

print(f"Parkinsonian Beta ARV Data Shape: {parkinsonian_beta_data.shape}")
print(f"Parkinsonian Beta ARV Timestamps Shape: {parkinsonian_beta_timestamps.shape}")

# Access sampling information
print(f"Beta ARV data uses timestamps instead of fixed rate. Time range: {healthy_beta_timestamps[0]} to {healthy_beta_timestamps[-1]} seconds")

# ---------- Plot the Beta ARV data----------
plt.subplot(2, 1, 1)
plt.plot(healthy_beta_timestamps, healthy_beta_data, label='Healthy Subject')
plt.plot(parkinsonian_beta_timestamps, parkinsonian_beta_data, label='Parkinsonian Subject')
plt.xlabel('Time (s)')
plt.ylabel('Beta ARV (V)')
plt.title('Comparison: Healthy vs. Parkinsonian Beta ARV')
plt.legend()
plt.grid(True, alpha=0.3)

# ---------- Compute mean and variance ----------
healthy_beta_mean = np.mean(healthy_beta_data)
healthy_beta_std = np.std(healthy_beta_data)
parkinsonian_beta_mean = np.mean(parkinsonian_beta_data)
parkinsonian_beta_std = np.std(parkinsonian_beta_data)

print(f"Healthy Beta ARV: Mean = {healthy_beta_mean:.8f}V, Std = {healthy_beta_std:.8f}V")
print(f"Parkinsonian Beta ARV: Mean = {parkinsonian_beta_mean:.8f}V, Std = {parkinsonian_beta_std:.8f}V")

# ---------- Plot the Beta ARV distributions ----------
plt.subplot(2, 1, 2)
plt.hist(healthy_beta_data, bins=30, alpha=0.5, label='Healthy Subject')
plt.hist(parkinsonian_beta_data, bins=30, alpha=0.5, label='Parkinsonian Subject')
plt.xlabel('Beta ARV (V)')
plt.ylabel('Count')
plt.title('Distribution of Beta ARV Values')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('explore/beta_arv_comparison.png')

# Create a separate plot to look at the temporal evolution of beta power
plt.figure(figsize=(12, 6))
plt.plot(healthy_beta_timestamps, healthy_beta_data, marker='.', linestyle='-', markersize=5, alpha=0.7, label='Healthy Subject')
plt.plot(parkinsonian_beta_timestamps, parkinsonian_beta_data, marker='.', linestyle='-', markersize=5, alpha=0.7, label='Parkinsonian Subject')
plt.xlabel('Time (s)')
plt.ylabel('Beta ARV (V)')
plt.title('Temporal Evolution of Beta ARV')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('explore/beta_arv_temporal.png')