# This script loads the electrode table from the NWB file and prints it.
# This helps in understanding the electrode configuration.

import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the electrodes table and print it
electrodes_df = nwb.electrodes.to_dataframe()
print(electrodes_df)

io.close()