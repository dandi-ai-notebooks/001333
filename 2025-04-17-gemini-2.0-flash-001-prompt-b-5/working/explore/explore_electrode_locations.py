import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to load and plot electrode locations from an NWB file

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the electrode locations (x, y, z)
electrodes = nwb.electrodes.to_dataframe()
print(electrodes.columns)
exit() # Print dataframe columns and exit