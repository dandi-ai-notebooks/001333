"""
This script compares data from different sessions of the same subject to understand
the variability and consistency of the recordings.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Define multiple sessions to compare from the healthy-simulated-beta subject
sessions = {
    "1044": "1d94c7ad-dbaf-43ea-89f2-1b2518fab158",
    "1046": "e0fa57b2-02a4-4c20-92df-d7eb64b60170",
    "162": "c5f536b1-8500-48dc-904b-584efd33a72a" 
}

# Function to load and process NWB file
def process_nwb(asset_id, session_id):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    print(f"Loading session {session_id} from {url}...")
    
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    # Get the Beta_Band_Voltage data
    lfp = nwb.processing["ecephys"].data_interfaces["LFP"]
    beta_band_voltage = lfp.electrical_series["Beta_Band_Voltage"]
    
    # Extract data for analysis (using a sample of the data for quicker processing)
    data = beta_band_voltage.data[:]
    timestamps = beta_band_voltage.timestamps[:]
    
    # Calculate power spectral density
    fs = 1.0 / np.mean(np.diff(timestamps))
    f, Pxx = signal.welch(data, fs, nperseg=min(256, len(data)))
    
    return {
        "session_id": session_id,
        "data": data,
        "timestamps": timestamps,
        "f": f,
        "Pxx": Pxx,
        "sampling_rate": fs,
        "nwb": nwb
    }

# Process all sessions
session_data = {}
for session_id, asset_id in sessions.items():
    session_data[session_id] = process_nwb(asset_id, session_id)

# Create a figure comparing the time series data
plt.figure(figsize=(15, 10))
for session_id, data in session_data.items():
    # Plot the first 300 points to make the visualization clearer
    plt.plot(data["timestamps"][:300], data["data"][:300], label=f"Session {session_id}")

plt.title("Beta Band Voltage Comparison Across Sessions")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (volts)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/session_comparison_timeseries.png")
plt.close()

# Create a figure comparing the power spectral density
plt.figure(figsize=(12, 8))
for session_id, data in session_data.items():
    plt.semilogy(data["f"], data["Pxx"], label=f"Session {session_id}")

# Highlight the beta band (13-30 Hz)
plt.axvspan(13, 30, color='yellow', alpha=0.3)
plt.title("Power Spectral Density Comparison Across Sessions")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (V^2/Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/session_comparison_psd.png")
plt.close()

# Calculate the correlation between sessions
correlation_matrix = np.zeros((len(sessions), len(sessions)))
sessions_list = list(sessions.keys())

for i, session1 in enumerate(sessions_list):
    for j, session2 in enumerate(sessions_list):
        # Calculate correlation between the first 300 points of each session
        corr = np.corrcoef(
            session_data[session1]["data"][:300],
            session_data[session2]["data"][:300]
        )[0, 1]
        correlation_matrix[i, j] = corr

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(label='Correlation coefficient')
plt.title("Correlation Between Sessions")
plt.xticks(np.arange(len(sessions)), sessions_list)
plt.yticks(np.arange(len(sessions)), sessions_list)
for i in range(len(sessions)):
    for j in range(len(sessions)):
        plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", 
                 ha="center", va="center", color="white")
plt.tight_layout()
plt.savefig("explore/session_correlation.png")
plt.close()

# Print some summary statistics
print("\nSummary Statistics:")
for session_id, data in session_data.items():
    voltage_data = data["data"]
    print(f"\nSession {session_id}:")
    print(f"Min voltage: {np.min(voltage_data):.8f} V")
    print(f"Max voltage: {np.max(voltage_data):.8f} V")
    print(f"Mean voltage: {np.mean(voltage_data):.8f} V")
    print(f"Standard deviation: {np.std(voltage_data):.8f} V")
    
    # Calculate beta band power (13-30 Hz)
    beta_mask = (data["f"] >= 13) & (data["f"] <= 30)
    beta_power = np.sum(data["Pxx"][beta_mask]) 
    total_power = np.sum(data["Pxx"])
    print(f"Beta band power: {beta_power:.8f}")
    print(f"Beta power ratio: {beta_power/total_power:.4f}")