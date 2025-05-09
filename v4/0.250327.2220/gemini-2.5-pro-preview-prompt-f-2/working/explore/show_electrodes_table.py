# explore/show_electrodes_table.py
# This script loads an NWB file and prints the electrodes table as a pandas DataFrame.

import pynwb
import h5py
import remfile
import pandas as pd

def main():
    # Connect to DANDI and get the NWB file
    url = "https://api.dandiarchive.org/api/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    # Access the electrodes table and convert to DataFrame
    electrodes_df = nwb.electrodes.to_dataframe()

    # Print the DataFrame
    print("Electrodes Table:")
    print(electrodes_df)

if __name__ == "__main__":
    main()