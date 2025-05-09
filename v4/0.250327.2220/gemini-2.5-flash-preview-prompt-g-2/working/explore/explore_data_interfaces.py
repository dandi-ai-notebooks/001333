# This script lists the data interfaces available under the 'ecephys' processing module.
# This helps identify other processed data available in the NWB file.

import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the 'ecephys' processing module
ecephys_module = nwb.processing.get('ecephys')

if ecephys_module:
    print("Data interfaces in 'ecephys' processing module:")
    for name, data_interface in ecephys_module.data_interfaces.items():
        print(f"- {name}: {type(data_interface).__name__}")
else:
    print("'ecephys' processing module not found.")

io.close()