"""
This script examines the electrode information in a sample NWB file
to understand the recording setup.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np

# Load a sample healthy LFP file
url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode information
print("=== Electrode Information ===")
electrodes_df = nwb.electrodes.to_dataframe()
print(electrodes_df)

print("\n=== Electrode Group Information ===")
for name, group in nwb.electrode_groups.items():
    print(f"Group: {name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device.description}")