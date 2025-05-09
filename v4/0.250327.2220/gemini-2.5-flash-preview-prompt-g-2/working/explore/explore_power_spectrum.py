# This script loads a segment of LFP data for one electrode and plots its power spectrum.
# This helps in visualizing the frequency content, particularly the beta band.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the LFP data and electrodes info
lfp_series = nwb.processing['ecephys']['LFP']['LFP']
electrodes_df = nwb.electrodes.to_dataframe()

# Define segment to load (e.g., first 10 seconds)
sampling_rate = lfp_series.rate
start_time = 0
duration = 3.225 # seconds (entire dataset duration)
start_index = int(start_time * sampling_rate)
end_index = int((start_time + duration) * sampling_rate)

# Load the data segment (raw 1D data)
# Since we are loading the entire dataset duration (6450 time points),
# we load the entire 1D array (77400 data points).
data_segment_1d = lfp_series.data[:]

# Reshape the data segment into a 2D array (time, electrodes)
num_electrodes = electrodes_df.shape[0]
# The number of time points is the total number of points divided by the number of electrodes.
num_time_points_segment = data_segment_1d.shape[0] // num_electrodes
data_segment_2d = data_segment_1d.reshape(num_time_points_segment, num_electrodes)

# Select data for one electrode (e.g., electrode 0)
electrode_index_for_spectrum = 0
electrode_data_for_spectrum = data_segment_2d[:, electrode_index_for_spectrum]

# Compute the power spectrum using Welch's method
nperseg = int(sampling_rate) # Use 1 second segments for Welch
frequencies, power_spectrum = welch(electrode_data_for_spectrum, fs=sampling_rate, nperseg=nperseg)

# Plot the power spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, power_spectrum) # Use semilogy for better visualization
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title(f'Power Spectrum of LFP Data (Electrode {electrode_index_for_spectrum}) - First {duration} seconds')
plt.grid(True)
plt.xlim([0, 100]) # Limit frequency range for better visualization
plt.savefig(f'explore/lfp_power_spectrum_electrode{electrode_index_for_spectrum}.png') # Save plot to file
# plt.show() # Do not show plot

io.close()