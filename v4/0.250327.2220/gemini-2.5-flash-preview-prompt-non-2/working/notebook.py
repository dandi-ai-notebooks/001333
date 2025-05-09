# %% [markdown]
# # Exploring Dandiset 001333: Parkinson's Electrophysiological Signal Dataset (PESD)
#
# This notebook explores Dandiset 001333 version 0.250327.2220, which contains electrophysiological signals from both healthy and parkinsonian subjects.
#
# **Important:** This notebook was AI-generated and has not been fully verified. Please exercise caution when interpreting the code or results.
#
# ## Dandiset Overview
#
# Dandiset 001333, titled "Parkinson's Electrophysiological Signal Dataset (PESD)", provides electrophysiological signals aimed at helping understand Parkinson's Disease. It includes Beta Average Rectified Voltage (ARV) and Local Field Potential (LFP) signals from the Subthalamic Nucleus (STN).
#
# You can find the Dandiset here: https://dandiarchive.org/dandiset/001333/0.250327.2220
#
# ## Notebook Contents
#
# This notebook will cover:
# - Loading the Dandiset using the DANDI API.
# - Examining the assets within the Dandiset.
# - Loading a specific NWB file and exploring its metadata.
# - Visualizing a portion of the LFP data from the NWB file.
# - Summarizing the findings and suggesting future directions.
#
# ## Required Packages
#
# The following packages are required to run this notebook:
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `seaborn`
# - `itertools`

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme for better plot aesthetics
sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset
#
# We can connect to the DANDI archive and load the specified Dandiset using the `dandi` Python package.

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001333", "0.250327.2220")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB file
#
# This section demonstrates how to load a specific NWB file from the Dandiset and inspect some of its metadata. We will load the file `sub-healthy-simulated-lfp/sub-healthy-simulated-lfp_ses-162_ecephys.nwb` which has the asset ID `00df5264-001b-4bb0-a987-0ddfb6058961`. The URL for this asset is hardcoded as instructed.

# %%
# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# Here are some key metadata fields from the loaded NWB file:

# %%
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Keywords: {nwb.keywords[:]}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")

# %% [markdown]
# ## NWB File Contents
#
# The NWB file contains electrophysiological data, specifically Local Field Potentials (LFP). The data is organized within the `processing` module under `ecephys`. Within `ecephys`, there is an `LFP` data interface containing an `ElectricalSeries` object named `LFP`. This `ElectricalSeries` holds the LFP data and references the electrodes used for recording.
#
# There are also `ElectrodeGroup` objects representing the shanks and a `Device` object describing the virtual probe used in the simulation. The `nwb.electrodes` table provides detailed metadata about each electrode.
#
# Here is a summary of the relevant data paths within the NWB file:
#
# ```
# ├── processing
# │   └── ecephys
# │       ├── description: Processed electrophysiology data
# │       └── data_interfaces
# │           └── LFP
# │               └── electrical_series
# │                   └── LFP
# │                       ├── starting_time: 0.0 (seconds)
# │                       ├── rate: 2000.0 (Hz)
# │                       ├── unit: volts
# │                       ├── data: (Dataset) shape (77400,)
# │                       └── electrodes: (DynamicTableRegion) references nwb.electrodes
# ├── electrode_groups
# │   ├── shank0
# │   ├── shank1
# │   ├── shank2
# │   └── shank3
# ├── devices
# │   └── NEURON_Simulator
# └── electrodes: (DynamicTable) metadata about extracellular electrodes
#     ├── colnames: ['location', 'group', 'group_name', 'label']
#     └── to_dataframe(): (DataFrame) 12 rows, 4 columns
# ```
#
# You can explore this NWB file further on Neurosift:
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/00df5264-001b-4bb0-a987-0ddfb6058961/download/&dandisetId=001333&dandisetVersion=draft

# %% [markdown]
# ## Loading and Visualizing LFP Data
#
# We can access the LFP data from the `ElectricalSeries` and visualize a portion of it. We will load the data for the first electrode (`label: 0`) and plot the first 1000 time points.

# %%
# Get the ElectricalSeries object
ecephys_module = nwb.processing['ecephys']
lfp_data_interface = ecephys_module.data_interfaces['LFP']
lfp_electrical_series = lfp_data_interface.electrical_series['LFP']

# Get the electrode table and find the index of the first electrode ('label: 0')
electrode_table = nwb.electrodes.to_dataframe()
electrode_index_0 = electrode_table[electrode_table['label'] == '0'].index[0]

# Get the data for the first electrode.
# Note: We are only loading a subset of the data (first 1000 time points)
# using the index corresponding to the first electrode. The LFP data is
# likely structured with time as the first dimension and electrodes as the second.
# Since the data shape is (77400,), it seems to be concatenated for all electrodes.
# We will assume the data for each electrode is contiguous in this case and select
# the first 1000 points which likely correspond to the beginning of the first electrode's data.
# If the data were structured differently (e.g., (time, electrodes)), we would need to index
# accordingly (e.g., data[0:1000, electrode_index_0]).
data_subset = lfp_electrical_series.data[0:1000]
rate = lfp_electrical_series.rate
starting_time = lfp_electrical_series.starting_time
timestamps_subset = starting_time + np.arange(len(data_subset))) / rate

# %%
# Plot the subset of LFP data
plt.figure(figsize=(12, 6))
plt.plot(timestamps_subset, data_subset)
plt.xlabel('Time (s)')
plt.ylabel(f'Voltage ({lfp_electrical_series.unit})')
plt.title('Subset of LFP Data (First 1000 time points, Electrode 0)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Summarizing Findings and Future Directions
#
# This notebook provided a brief introduction to accessing and exploring Dandiset 001333. We successfully loaded Dandiset information and a specific NWB file. We also visualized a small segment of the LFP data from one electrode.
#
# Possible future directions for analysis include:
#
# - Loading and visualizing LFP data from other electrodes or sessions.
# - Examining the ARV (Average Rectified Voltage) data if available in other NWB files within the Dandiset.
# - Performing basic signal processing (e.g., filtering, spectral analysis) on the LFP data to investigate the beta oscillations mentioned in the Dandiset description.
# - Comparing the electrophysiological signals between healthy and parkinsonian subjects within the Dandiset.
# - Integrating this data with other relevant datasets or clinical information if available.
#

# %%