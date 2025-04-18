"""
This script explores the Beta_Band_Voltage data from a selected NWB file 
in the Parkinson's Electrophysiological Signal Dataset (PESD).
It loads the data, plots the time series, and generates a power spectral density plot
to examine the beta band frequencies (13-30 Hz).
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
file_path = "sub-healthy-simulated-beta/sub-healthy-simulated-beta_ses-1044_ecephys.nwb"

# Load the NWB file
print(f"Loading NWB file from {file_path}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract basic metadata
print(f"Session description: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject description: {nwb.subject.description}")

# Get the Beta_Band_Voltage data
lfp = nwb.processing["ecephys"].data_interfaces["LFP"]
beta_band_voltage = lfp.electrical_series["Beta_Band_Voltage"]

# Extract data for analysis
data = beta_band_voltage.data[:]
timestamps = beta_band_voltage.timestamps[:]

print(f"Data shape: {data.shape}")
print(f"Timestamps shape: {timestamps.shape}")
print(f"Data unit: {beta_band_voltage.unit}")
print(f"Timestamps unit: {beta_band_voltage.timestamps_unit}")

# Create a time series plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps, data)
plt.title(f"Beta Band Voltage - {nwb.subject.subject_id}")
plt.xlabel(f"Time ({beta_band_voltage.timestamps_unit})")
plt.ylabel(f"Voltage ({beta_band_voltage.unit})")
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/beta_band_voltage_timeseries.png")
plt.close()

# Calculate and plot the power spectral density
fs = 1.0 / np.mean(np.diff(timestamps))  # Calculate sampling frequency
print(f"Estimated sampling frequency: {fs} Hz")

# Calculate the power spectral density using Welch's method
f, Pxx = signal.welch(data, fs, nperseg=min(256, len(data)))

# Plot the power spectral density
plt.figure(figsize=(10, 6))
plt.semilogy(f, Pxx)
plt.axvspan(13, 30, color='yellow', alpha=0.3) # Highlight beta band (13-30 Hz)
plt.title(f"Power Spectral Density - {nwb.subject.subject_id}")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/beta_band_voltage_psd.png")
plt.close()

# Get information about the electrodes
electrodes_df = nwb.electrodes.to_dataframe()
print("\nElectrode information:")
print(electrodes_df)