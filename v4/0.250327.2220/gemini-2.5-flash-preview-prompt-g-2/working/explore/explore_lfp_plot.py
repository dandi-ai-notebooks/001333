# This script loads a short segment of LFP data and plots it.
# This helps in visualizing the raw LFP signal.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the LFP data
lfp_series = nwb.processing['ecephys']['LFP']['LFP']

# Load a short segment of the raw 1D LFP data and plot it.
# Given that the LFP data is a 1D array and we don't have clear
# information on how to separate by electrode from the file info tool,
# we will plot a segment of the raw data as is.

sampling_rate = lfp_series.rate
start_index = 0
end_index = 2000 # Load the first 2000 data points (1 second at 2000 Hz)

# Load a short segment of the raw 1D data
data_segment = lfp_series.data[start_index:end_index]

# Create a time vector for the segment
time = (np.arange(start_index, end_index) / sampling_rate) + lfp_series.starting_time

# Plot the raw 1D data segment
plt.figure(figsize=(10, 4))
plt.plot(time, data_segment)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Short Segment of Raw 1D LFP Data')
plt.grid(True)
plt.savefig('explore/lfp_raw_segment.png') # Save plot to file
# plt.show() # Do not show the plot, save it to file instead

io.close()