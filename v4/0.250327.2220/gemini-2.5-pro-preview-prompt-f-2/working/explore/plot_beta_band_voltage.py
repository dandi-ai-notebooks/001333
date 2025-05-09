# explore/plot_beta_band_voltage.py
# This script loads an NWB file and plots the Beta_Band_Voltage data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # Connect to DANDI and get the NWB file
    url = "https://api.dandiarchive.org/api/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    # Access the Beta_Band_Voltage data and timestamps
    beta_band_voltage = nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"]
    data = beta_band_voltage.data[:]
    timestamps = beta_band_voltage.timestamps[:]

    # Create the plot
    sns.set_theme()
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (volts)")
    plt.title("Beta Band Voltage")
    plt.grid(True)
    
    # Save the plot
    plt.savefig("explore/beta_band_voltage.png")
    print("Plot saved to explore/beta_band_voltage.png")

if __name__ == "__main__":
    main()