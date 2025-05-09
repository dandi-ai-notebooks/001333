# This script loads the NWB file for sub-healthy-simulated-lfp_ses-162_ecephys.nwb from DANDI
# and prints metadata, electrode table information, and basic info about the LFP data.
import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

# Remote NWB file URL
url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start time:", nwb.session_start_time)
print("Experiment description:", nwb.experiment_description)
print("Lab:", getattr(nwb, "lab", ''))
print("Institution:", getattr(nwb, "institution", ''))
print("Subject ID:", nwb.subject.subject_id)
print("Subject species:", nwb.subject.species)
print("Subject age:", nwb.subject.age)
print("Subject description:", nwb.subject.description)
print("Related publications:", nwb.related_publications)
print("Keywords:", nwb.keywords[:])
print("File creation date(s):", nwb.file_create_date)

print("\nElectrode table (first 5 rows):")
try:
    df = nwb.electrodes.to_dataframe()
    print(df.head())
except Exception as e:
    print("Could not load electrode table:", repr(e))

if "ecephys" in nwb.processing:
    ecephys = nwb.processing["ecephys"]
    if "LFP" in ecephys.data_interfaces:
        LFP = ecephys.data_interfaces["LFP"]
        if "LFP" in LFP.electrical_series:
            LFP_1 = LFP.electrical_series["LFP"]
            print("\nLFP dataset: shape:", LFP_1.data.shape, "dtype:", LFP_1.data.dtype)
            print("LFP start time:", LFP_1.starting_time)
            print("LFP rate:", LFP_1.rate)
            print("LFP unit:", LFP_1.unit)
            # Print some statistics over the whole dataset (using a small chunk if large)
            N = LFP_1.data.shape[0]
            sample_n = min(10_000, N)
            d_sample = LFP_1.data[:sample_n]
            print(f"First {sample_n} LFP data points: mean={np.mean(d_sample):.4g}, std={np.std(d_sample):.4g}, min={np.min(d_sample):.4g}, max={np.max(d_sample):.4g}")
            print("First 10 data points:", d_sample[:10])
        else:
            print("No LFP 'LFP' data found in electrical_series.")
    else:
        print("No 'LFP' data_interface found in 'ecephys' processing module.")
else:
    print("No 'ecephys' processing module found.")

io.close()
h5_file.close()
remote_file.close()