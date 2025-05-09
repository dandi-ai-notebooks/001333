# This script loads the electrodes table from the NWB file and prints it as a pandas DataFrame to inspect electrode info.

import pynwb
import h5py
import remfile
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/download/"

# Load NWB
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract electrodes table as DataFrame
df = nwb.electrodes.to_dataframe()
print("Electrodes table:")
print(df)