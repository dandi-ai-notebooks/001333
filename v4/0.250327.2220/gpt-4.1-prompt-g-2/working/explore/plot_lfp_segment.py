# This script plots the first 5000 samples of the LFP data from the specified NWB file, to give a quick look at temporal structure.
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

LFP = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
data = LFP.data[:5000]
rate = LFP.rate
times = np.arange(len(data)) / rate

plt.figure(figsize=(10, 4))
plt.plot(times, data, color="k", linewidth=0.7)
plt.xlabel("Time (s)")
plt.ylabel(f"LFP ({LFP.unit})")
plt.title("First 5,000 samples of LFP signal")
plt.tight_layout()
plt.savefig("explore/plot_lfp_segment.png")
plt.close()

io.close()
h5_file.close()
remote_file.close()