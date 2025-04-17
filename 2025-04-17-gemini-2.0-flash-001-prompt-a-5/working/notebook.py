# %% [markdown]
# # Exploring Dandiset 001333: Parkinson's Electrophysiological Signal Dataset (PESD)

# %% [markdown]
# **Disclaimer:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This notebook explores Dandiset 001333, the "Parkinson's Electrophysiological Signal Dataset (PESD)". This dataset contains electrophysiological signals from both healthy and parkinsonian subjects, focusing on beta oscillations in the subthalamic nucleus (STN) as biomarkers for Parkinson's Disease (PD). The dataset includes Beta Average Rectified Voltage (ARV) signals (frequency domain) and Local Field Potential (LFP) signals from the STN (time domain). More details about the data can be found in the associated publication: "Preliminary Results of Neuromorphic Controller Design and a Parkinson's Disease Dataset Building for Closed-Loop Deep Brain Stimulation" (https://arxiv.org/abs/2407.17756). The metadata can be accessed at https://dandiarchive.org/dandiset/001333
#
# You can explore the dandiset using the neurosift interface: https://neurosift.app/dandiset/001333

# %% [markdown]
# ## Scope of this Notebook
#
# This notebook demonstrates how to:
#
# 1.  Load the Dandiset using the DANDI API.
# 2.  Access and explore the metadata of an NWB file within the Dandiset.
# 3.  Load and visualize electrophysiological data (LFP signals) from the NWB file.

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook. Ensure that they are installed in your environment.
#
# *   `pynwb`
# *   `h5py`
# *   `remfile`
# *   `matplotlib`

# %% [markdown]
# ## Loading the Dandiset using the DANDI API

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001333")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading and Exploring an NWB File
#
# This section demonstrates how to load an NWB file from the Dandiset and explore its metadata. We will load the file `sub-healthy-simulated-beta/sub-healthy-simulated-beta_ses-1044_ecephys.nwb`.
#

# %%
# This script shows how to load the NWB file at https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/ in Python using PyNWB

import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb.session_description # (str) Parkinson's Electrophysiological Signal Dataset (PESD) Generated from Simulation
nwb.identifier # (str) 84828db4-a3a3-4b2e-abff-6db2b404dd68
nwb.session_start_time # (datetime) 2025-04-03T12:30:26.094607-04:00
nwb.timestamps_reference_time # (datetime) 2025-04-03T12:30:26.094607-04:00
nwb.file_create_date # (list) [datetime.datetime(2025, 4, 3, 12, 30, 26, 128020, tzinfo=tzoffset(None, -14400))]
nwb.experimenter # (tuple) ['Ananna Biswas']
nwb.related_publications # (tuple) ['https://arxiv.org/abs/2407.17756', 'DOI: 10.3389/fnins.2020.00166']
nwb.keywords # (StrDataset) shape (4,); dtype object
# nwb.keywords[:] # Access all data
# nwb.keywords[0:10] # Access first 10 elements
# First few values of nwb.keywords: ['ecephys' 'LFP' "Parkinson's Disease" 'Beta Band']
nwb.processing # (LabelledDict)
nwb.processing["ecephys"] # (ProcessingModule)
nwb.processing["ecephys"].description # (str) Processed electrophysiology data
nwb.processing["ecephys"].data_interfaces # (LabelledDict)
nwb.processing["ecephys"].data_interfaces["LFP"] # (LFP)
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series # (LabelledDict)
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"] # (ElectricalSeries)
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].resolution # (float64) -1.0
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].comments # (str) no comments
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].description # (str) no description
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].conversion # (float64) 1.0
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].offset # (float64) 0.0
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].unit # (str) volts
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes # (DynamicTableRegion)
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.description # (str) all electrodes
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table # (DynamicTable)
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table.description # (str) metadata about extracellular electrodes
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table.colnames # (tuple) ['location', 'group', 'group_name', 'label']
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table.id # (ElementIdentifiers)
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table.location # (VectorData) the location of channel within the subject e.g. brain region
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table.group # (VectorData) a reference to the ElectrodeGroup this electrode is a part of
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table.group_name # (VectorData) the name of the ElectrodeGroup this electrode is a part of
nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].electrodes.table.label # (VectorData) label of electrode
nwb.electrode_groups # (LabelledDict)
nwb.electrode_groups["shank0"] # (ElectrodeGroup)
nwb.electrode_groups["shank0"].description # (str) Simulated electrode group for shank 0
nwb.electrode_groups["shank0"].location # (str) Simulated Cortico-basal-ganglia network of brain
nwb.electrode_groups["shank0"].device # (Device)
nwb.electrode_groups["shank0"].device.description # (str) Virtual probe used in NEURON simulation
nwb.electrode_groups["shank0"].device.manufacturer # (str) N/A
nwb.electrode_groups["shank1"] # (ElectrodeGroup)
nwb.electrode_groups["shank1"].description # (str) Simulated electrode group for shank 1
nwb.electrode_groups["shank1"].location # (str) Simulated Cortico-basal-ganglia network of brain
nwb.electrode_groups["shank1"].device # (Device)
nwb.electrode_groups["shank1"].device.description # (str) Virtual probe used in NEURON simulation
nwb.electrode_groups["shank1"].device.manufacturer # (str) N/A
nwb.electrode_groups["shank2"] # (ElectrodeGroup)
nwb.electrode_groups["shank2"].description # (str) Simulated electrode group for shank 2
nwb.electrode_groups["shank2"].location # (str) Simulated Cortico-basal-ganglia network of brain
nwb.electrode_groups["shank2"].device # (Device)
nwb.electrode_groups["shank2"].device.description # (str) Virtual probe used in NEURON simulation
nwb.electrode_groups["shank2"].device.manufacturer # (str) N/A
nwb.electrode_groups["shank3"] # (ElectrodeGroup)
nwb.electrode_groups["shank3"].description # (str) Simulated electrode group for shank 3
nwb.electrode_groups["shank3"].location # (str) Simulated Cortico-basal-ganglia network of brain
nwb.electrode_groups["shank3"].device # (Device)
nwb.electrode_groups["shank3"].device.description # (str) Virtual probe used in NEURON simulation
nwb.electrode_groups["shank3"].device.manufacturer # (str) N/A
nwb.devices # (LabelledDict)
nwb.devices["NEURON_Simulator"] # (Device)
nwb.devices["NEURON_Simulator"].description # (str) Virtual probe used in NEURON simulation
nwb.devices["NEURON_Simulator"].manufacturer # (str) N/A
nwb.experiment_description # (str) The PESD dataset is generated from a cortico-basal-ganglia network for a Parkinsonian computation...
nwb.lab # (str) BrainX Lab
nwb.institution # (str) Michigan Technological University
nwb.electrodes # (DynamicTable)
nwb.electrodes.description # (str) metadata about extracellular electrodes
nwb.electrodes.colnames # (tuple) ['location', 'group', 'group_name', 'label']
nwb.electrodes.columns # (tuple)
nwb.electrodes.id # (ElementIdentifiers)
nwb.electrodes.location # (VectorData) the location of channel within the subject e.g. brain region
nwb.electrodes.group # (VectorData) a reference to the ElectrodeGroup this electrode is a part of
nwb.electrodes.group_name # (VectorData) the name of the ElectrodeGroup this electrode is a part of
nwb.electrodes.label # (VectorData) label of electrode
nwb.subject # (Subject)
nwb.subject.age # (str) P0D
nwb.subject.age__reference # (str) birth
nwb.subject.description # (str) This is a simulated dataset generated from a computational model.
nwb.subject.sex # (str) U
nwb.subject.species # (str) Homo sapiens
nwb.subject.subject_id # (str) healthy-simulated-beta

# %% [markdown]
# ## Loading and Visualizing Electrophysiological Data
#
# This section demonstrates how to load and visualize electrophysiological data (LFP signals) from the NWB file. We will load the "Beta_Band_Voltage" ElectricalSeries from the LFP data interface.

# %%
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

data = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data
timestamps = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].timestamps

# Access all data
data[:]
timestamps[:]

# Access first 10 elements
data[0:10]
timestamps[0:10]

# Plot the first 1000 data points
plt.figure(figsize=(10, 5))
plt.plot(timestamps[0:1000], data[0:1000])
plt.xlabel("Time (s)")
plt.ylabel("Beta Band Voltage (V)")
plt.title("LFP Signal (Beta Band Voltage)")
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook provided a basic introduction to exploring the Parkinson's Electrophysiological Signal Dataset (PESD) using the DANDI API and PyNWB. We demonstrated how to load the Dandiset, access metadata, and visualize electrophysiological data from an NWB file.
#
# Further analysis could involve:
#
# *   Exploring other NWB files in the Dandiset.
# *   Analyzing the frequency content of the LFP signals.
# *   Comparing data from healthy and parkinsonian subjects.
# *   Developing more advanced visualizations to highlight key features of the data.