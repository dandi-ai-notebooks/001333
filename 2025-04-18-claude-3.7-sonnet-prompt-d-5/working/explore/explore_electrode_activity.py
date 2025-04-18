"""
This script explores the electrode activity in the NWB file, specifically examining
the data from multiple electrodes and their relationships.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
file_path = "sub-healthy-simulated-beta/sub-healthy-simulated-beta_ses-1044_ecephys.nwb"

# Load the NWB file
print(f"Loading NWB file from {file_path}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the electrodes information
electrodes_df = nwb.electrodes.to_dataframe()

# Print all NWB processing modules to understand what data types are available
print("\nProcessing modules in the NWB file:")
for module_name, module in nwb.processing.items():
    print(f"\nModule: {module_name}")
    print(f"Description: {module.description}")
    print("Data interfaces:")
    for interface_name, interface in module.data_interfaces.items():
        print(f"  - {interface_name} ({type(interface).__name__})")
        # If it's an LFP interface, list its electrical series
        if isinstance(interface, pynwb.ecephys.LFP):
            print(f"    Electrical series:")
            for series_name, series in interface.electrical_series.items():
                print(f"      - {series_name} (shape: {series.data.shape}, unit: {series.unit})")

# Get the Beta_Band_Voltage data
lfp = nwb.processing["ecephys"].data_interfaces["LFP"]
beta_band_voltage = lfp.electrical_series["Beta_Band_Voltage"]

# Get a subset of the data for faster processing
data = beta_band_voltage.data[:]
timestamps = beta_band_voltage.timestamps[:]
fs = 1.0 / np.mean(np.diff(timestamps))  # Calculate sampling frequency

# Print data shape and other information
print(f"\nBeta_Band_Voltage data shape: {data.shape}")
print(f"Number of electrodes: {len(electrodes_df)}")
print(f"Sample frequency: {fs} Hz")

# Check if any other data types are available (e.g., ARV data)
print("\nChecking for other data types in the NWB file...")
for group_name, group in h5_file.items():
    print(f"Group: {group_name}")
    if isinstance(group, h5py.Group):
        for subgroup_name, subgroup in group.items():
            print(f"  - {subgroup_name}")

# Plot the Beta_Band_Voltage data for the first 200 samples across all electrodes
# We'll use a subset to make the plot clearer
plt.figure(figsize=(15, 8))
plt.plot(timestamps[:200], data[:200])
plt.title("Beta Band Voltage - First 200 Samples")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (volts)")
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/beta_voltage_first_200.png")
plt.close()

# Create a heatmap of the electrode data
# Let's examine if there's a correlation between electrode positions
plt.figure(figsize=(10, 8))
plt.imshow(np.corrcoef(data[:200]), cmap='viridis', aspect='auto')
plt.colorbar(label='Correlation coefficient')
plt.title("Correlation between Beta Band Voltage Signals")
plt.savefig("explore/electrode_correlation.png")
plt.close()

# Print what's in the subject data
print("\nSubject information:")
for attr in dir(nwb.subject):
    if not attr.startswith('_'):
        try:
            value = getattr(nwb.subject, attr)
            if not callable(value):
                print(f"{attr}: {value}")
        except:
            pass