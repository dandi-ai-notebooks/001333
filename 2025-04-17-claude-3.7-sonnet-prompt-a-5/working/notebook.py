# %% [markdown]
# # Exploring Dandiset 001333: Parkinson's Electrophysiological Signal Dataset (PESD)
# 
# > **⚠️ CAUTION**: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.
# 
# ## Overview
# 
# This dataset contains electrophysiological signals from both healthy and parkinsonian subjects, with a focus on beta oscillations (13-30 Hz) in the subthalamic nucleus (STN), which are typically used as pathological biomarkers for Parkinson's Disease symptoms.
# 
# The dataset includes two types of signals:
# 1. **Beta Average Rectified Voltage (ARV)**: Signals in the frequency domain
# 2. **Local Field Potential (LFP) from the Subthalamic Nucleus (STN)**: Signals in the time domain
# 
# You can explore this dataset in Neurosift: [https://neurosift.app/dandiset/001333](https://neurosift.app/dandiset/001333)
# 
# ## What We'll Cover
# 
# In this notebook, we will:
# 1. Connect to the DANDI Archive and load the dataset
# 2. Explore the structure and metadata of the NWB files
# 3. Load and visualize LFP data from the STN
# 4. Analyze beta oscillations in the signals
# 5. Compare data across multiple recordings
# 
# ## Required Packages

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
from scipy import signal
import warnings

# Set seaborn styling for plots
sns.set_theme()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Connecting to the DANDI Archive
# 
# First, let's connect to the DANDI archive and load the Dandiset information.

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001333")
metadata = dandiset.get_metadata()
assets = list(dandiset.get_assets())

print(f"Dandiset ID: {dandiset.identifier}")
print(f"Dandiset Name: {metadata.name}")
print(f"Found {len(assets)} assets in the dataset")

# Display the first 5 assets
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Exploring NWB File Structure
# 
# Let's load one of the NWB files to explore its structure and understand the data it contains.
# 
# We'll use the first file in our assets list: `sub-healthy-simulated-beta/sub-healthy-simulated-beta_ses-1044_ecephys.nwb`

# %%
# Get the first asset and its URL
asset = assets[0]
print(f"Loading file: {asset.path}")
print(f"Asset ID: {asset.identifier}")

# Construct the URL for accessing this asset
url = f"https://api.dandiarchive.org/api/assets/{asset.identifier}/download/"
print(f"URL: {url}")

# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## File Metadata
# 
# Let's look at the metadata for this file to understand what it contains.

# %%
# Display basic metadata
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experimenter: {', '.join(nwb.experimenter)}")
print(f"Keywords: {nwb.keywords[:]}")
print(f"Related publications: {', '.join(nwb.related_publications)}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")

# Display subject information
print("\nSubject Information:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Description: {nwb.subject.description}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")

# %% [markdown]
# ## Exploring the Electrode Setup
# 
# Let's examine the electrode configuration used in the recordings.

# %%
# Get electrodes data
electrodes_df = nwb.electrodes.to_dataframe()
print("Electrodes:")
print(electrodes_df)

# Display electrode groups information
print("\nElectrode Groups:")
for name, group in nwb.electrode_groups.items():
    print(f"\n{name}:")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device}")

# %% [markdown]
# ## Exploring Data Structure
# 
# Let's examine the data structure available in the file.

# %%
# Check available processing modules
print("Processing modules:")
for module_name, module in nwb.processing.items():
    print(f"\n{module_name}:")
    print(f"  Description: {module.description}")
    print("  Data interfaces:")
    for interface_name in module.data_interfaces.keys():
        print(f"    - {interface_name}")

# Inspect the structure of the LFP data
print("\nExploring the LFP data structure:")
lfp = nwb.processing["ecephys"].data_interfaces["LFP"]

# Helper function to print container details
def print_container_details(container, indent=0):
    indent_str = "  " * indent
    print(f"{indent_str}Type: {type(container).__name__}")
    
    # If it's a container with attributes, print them
    if hasattr(container, "fields") and callable(container.fields):
        print(f"{indent_str}Fields: {container.fields}")
        
    # If it's a container with children, print them
    if hasattr(container, "children") and callable(container.children):
        print(f"{indent_str}Children:")
        for child in container.children():
            print(f"{indent_str}  - {child}")
    
    # If it has a data attribute, print info about it
    if hasattr(container, "data"):
        if hasattr(container.data, "shape"):
            print(f"{indent_str}Data shape: {container.data.shape}")
        else:
            print(f"{indent_str}Data: {type(container.data).__name__}")
    
    # If it has a timestamps attribute, print info about it
    if hasattr(container, "timestamps") and container.timestamps is not None:
        print(f"{indent_str}Timestamps shape: {container.timestamps.shape}")
    
    # If it's a dictionary-like object, print keys
    try:
        if hasattr(container, "keys") and callable(container.keys):
            print(f"{indent_str}Keys: {list(container.keys())}")
    except:
        pass

# Print details about the LFP object
print_container_details(lfp)

# Check if LFP has electrical_series
if hasattr(lfp, 'electrical_series'):
    print("\nLFP electrical series:")
    for name, series in lfp.electrical_series.items():
        print(f"\n  {name}:")
        print_container_details(series, indent=2)

# %% [markdown]
# ## Accessing LFP Data
# 
# Now that we've explored the structure, let's access the LFP data.

# %%
# Function to safely get electrical series data
def get_electrical_series(nwb_file):
    """Extract electrical series data from an NWB file."""
    try:
        # Try standard path for LFP data
        if 'ecephys' in nwb_file.processing:
            ecephys = nwb_file.processing['ecephys']
            
            if 'LFP' in ecephys.data_interfaces:
                lfp = ecephys.data_interfaces['LFP']
                
                # Check if LFP has electrical_series attribute
                if hasattr(lfp, 'electrical_series'):
                    # Get the first electrical series if available
                    for series_name, series_data in lfp.electrical_series.items():
                        print(f"Found electrical series: {series_name}")
                        return series_data
                # Check if LFP itself is the electrical series
                elif hasattr(lfp, 'data'):
                    print("LFP object itself contains data")
                    return lfp
        
        # If we haven't returned yet, search for any ElectricalSeries
        print("Searching for ElectricalSeries objects...")
        for module_name in nwb_file.processing:
            module = nwb_file.processing[module_name]
            for interface_name in module.data_interfaces:
                interface = module.data_interfaces[interface_name]
                if isinstance(interface, pynwb.ecephys.ElectricalSeries):
                    print(f"Found ElectricalSeries: {interface_name} in {module_name}")
                    return interface
                
        raise ValueError("Could not find any ElectricalSeries data")
    
    except Exception as e:
        print(f"Error accessing electrical series: {e}")
        return None

# Get the electrical series from the file
electrical_series = get_electrical_series(nwb)

if electrical_series is not None:
    print("\nElectrical Series Details:")
    print(f"  Data shape: {electrical_series.data.shape}")
    
    if hasattr(electrical_series, 'unit'):
        print(f"  Unit: {electrical_series.unit}")
    
    # Check for timestamps
    if hasattr(electrical_series, 'timestamps') and electrical_series.timestamps is not None:
        print(f"  Timestamps shape: {electrical_series.timestamps.shape}")
        time_data = electrical_series.timestamps[:]
    else:
        print("  No timestamps attribute, constructing time vector from rate")
        if hasattr(electrical_series, 'rate'):
            rate = electrical_series.rate
            if rate:
                n_samples = electrical_series.data.shape[0]
                time_data = np.arange(n_samples) / rate
                print(f"  Created timestamps using rate: {rate}")
            else:
                print("  Rate is not defined")
                # Try to find starting_time and create timestamps
                time_data = np.arange(electrical_series.data.shape[0])
        else:
            print("  No rate attribute found")
            time_data = np.arange(electrical_series.data.shape[0])
    
    # Get a subset of data for visualization
    max_points = 1000  # Limit for visualization
    data = electrical_series.data[:]
    
    if len(data.shape) > 1:
        # If multi-channel, take the first channel
        print(f"  Multi-channel data detected ({data.shape[1]} channels), using first channel")
        data = data[:, 0]
        print(f"  Selected data shape: {data.shape}")

    # Create a time subset if the data is large
    if len(data) > max_points:
        # Downsample by taking every Nth point
        downsample_factor = len(data) // max_points
        data = data[::downsample_factor]
        if len(time_data) > max_points:
            time_data = time_data[::downsample_factor]
        print(f"  Downsampled to {len(data)} points")
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(time_data[:len(data)], data)
    if hasattr(electrical_series, 'unit'):
        ylabel = f'Voltage ({electrical_series.unit})'
    else:
        ylabel = 'Voltage'
        
    plt.title('LFP Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# %% [markdown]
# ## Spectral Analysis of the LFP Data
# 
# Let's compute and plot the power spectral density of the signal to analyze its frequency components.

# %%
if electrical_series is not None:
    # Get data for spectral analysis
    data = electrical_series.data[:]
    
    if len(data.shape) > 1:
        # If multi-channel, average across all channels
        data = np.mean(data, axis=1)
        print(f"Averaged across {data.shape[1]} channels for spectral analysis")
    
    # Get sampling rate
    if hasattr(electrical_series, 'timestamps') and electrical_series.timestamps is not None:
        timestamps = electrical_series.timestamps[:]
        sampling_rate = 1.0 / np.mean(np.diff(timestamps))
        print(f"Calculated sampling rate: {sampling_rate:.2f} Hz")
    elif hasattr(electrical_series, 'rate') and electrical_series.rate:
        sampling_rate = electrical_series.rate
        print(f"Using specified sampling rate: {sampling_rate} Hz")
    else:
        # Assume a default rate
        sampling_rate = 1000.0
        print(f"Using default sampling rate: {sampling_rate} Hz")
    
    # Compute power spectral density
    window_size = min(1024, len(data) // 2)
    f, Pxx = signal.welch(data, sampling_rate, nperseg=window_size)
    
    # Plot the PSD
    plt.figure(figsize=(12, 6))
    plt.semilogy(f, Pxx)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (V²/Hz)')
    plt.grid(True)
    plt.axvspan(13, 30, alpha=0.3, color='red', label='Beta band (13-30 Hz)')
    plt.legend()
    # Focus on frequencies up to 100 Hz
    plt.xlim(0, min(100, max(f)))
    plt.show()
    
    # Calculate beta band power
    beta_mask = (f >= 13) & (f <= 30)
    beta_power = np.mean(Pxx[beta_mask])
    total_power = np.mean(Pxx)
    
    print(f"Beta band (13-30 Hz) power: {beta_power:.6e}")
    print(f"Ratio of beta band power to total power: {beta_power/total_power:.4f}")

# %% [markdown]
# ## Comparing Multiple Recordings
# 
# Now let's load data from multiple recordings and compare their beta band power.

# %%
# Function to calculate beta band power from a file
def calculate_beta_power(asset, max_freq=None):
    """
    Calculate beta band power from an NWB file.
    
    Parameters:
    -----------
    asset : Asset
        The DANDI asset to load
    max_freq : float, optional
        Maximum frequency to include in the analysis
        
    Returns:
    --------
    dict
        Dictionary with session_id, beta_power, and other metadata
    """
    session_id = asset.path.split('_')[-2].replace('ses-', '')
    url = f"https://api.dandiarchive.org/api/assets/{asset.identifier}/download/"
    
    print(f"Processing {asset.path}...")
    
    try:
        remote_file = remfile.File(url)
        h5_file = h5py.File(remote_file)
        io = pynwb.NWBHDF5IO(file=h5_file)
        nwb_file = io.read()
        
        # Get electrical series
        electrical_series = get_electrical_series(nwb_file)
        
        if electrical_series is None:
            return {
                'session_id': session_id,
                'error': "Could not find electrical series"
            }
        
        # Get data
        data = electrical_series.data[:]
        
        if len(data.shape) > 1:
            # If multi-channel, average across channels
            data = np.mean(data, axis=1)
        
        # Get sampling rate
        if hasattr(electrical_series, 'timestamps') and electrical_series.timestamps is not None:
            timestamps = electrical_series.timestamps[:]
            sampling_rate = 1.0 / np.mean(np.diff(timestamps))
        elif hasattr(electrical_series, 'rate') and electrical_series.rate:
            sampling_rate = electrical_series.rate
        else:
            # Assume a default rate
            sampling_rate = 1000.0
        
        # Calculate PSD
        window_size = min(1024, len(data) // 2)
        f, Pxx = signal.welch(data, sampling_rate, nperseg=window_size)
        
        # Limit frequency if requested
        if max_freq is not None:
            mask = f <= max_freq
            f = f[mask]
            Pxx = Pxx[mask]
        
        # Calculate beta band power
        beta_mask = (f >= 13) & (f <= 30)
        beta_power = np.mean(Pxx[beta_mask])
        total_power = np.mean(Pxx)
        
        result = {
            'session_id': session_id,
            'beta_power': beta_power,
            'total_power': total_power,
            'ratio': beta_power/total_power,
            'subject_id': nwb_file.subject.subject_id
        }
        
        # Clean up
        io.close()
        h5_file.close()
        remote_file.close()
        
        return result
    
    except Exception as e:
        print(f"Error processing {asset.path}: {e}")
        return {
            'session_id': session_id,
            'error': str(e)
        }

# Process a subset of the files (first 5 for demonstration)
results = []

for i, asset in enumerate(assets[:5]):
    result = calculate_beta_power(asset)
    if 'error' not in result:
        results.append(result)

# Convert results to DataFrame
if results:
    df = pd.DataFrame(results)
    
    # Plot beta powers
    plt.figure(figsize=(12, 6))
    plt.bar(df['session_id'], df['beta_power'])
    plt.title('Beta Band Power (13-30 Hz) Across Recordings')
    plt.xlabel('Session ID')
    plt.ylabel('Mean Power in Beta Band (V²/Hz)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot beta power ratio
    plt.figure(figsize=(12, 6))
    plt.bar(df['session_id'], df['ratio'])
    plt.title('Beta Band Power Ratio Across Recordings')
    plt.xlabel('Session ID')
    plt.ylabel('Beta Power / Total Power')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Display the data table
    print("Analysis Results:")
    print(df)

# %% [markdown]
# ## Time-Frequency Analysis
# 
# Let's perform time-frequency analysis to visualize how the signal's frequency components change over time.

# %%
if electrical_series is not None:
    # Get data for time-frequency analysis
    data = electrical_series.data[:]
    
    if len(data.shape) > 1:
        # If multi-channel, take first channel
        data = data[:, 0]
    
    # Get sampling rate
    if hasattr(electrical_series, 'timestamps') and electrical_series.timestamps is not None:
        timestamps = electrical_series.timestamps[:]
        sampling_rate = 1.0 / np.mean(np.diff(timestamps))
    elif hasattr(electrical_series, 'rate') and electrical_series.rate:
        sampling_rate = electrical_series.rate
    else:
        # Assume a default rate
        sampling_rate = 1000.0
    
    # Get timestamps if available
    timestamps = None
    if hasattr(electrical_series, 'timestamps') and electrical_series.timestamps is not None:
        timestamps = electrical_series.timestamps[:]
    
    # Limit the analysis to first N points if the data is large
    max_points = 10000
    if len(data) > max_points:
        data = data[:max_points]
        if timestamps is not None and len(timestamps) > max_points:
            timestamps = timestamps[:max_points]
    
    # Compute the spectrogram
    window_size = min(256, len(data) // 10)
    f, t, Sxx = signal.spectrogram(data, sampling_rate, nperseg=window_size)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.title('Spectrogram of LFP Data')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    
    # Highlight beta band
    plt.axhline(y=13, color='r', linestyle='-', alpha=0.7, label='Beta band lower bound (13 Hz)')
    plt.axhline(y=30, color='r', linestyle='-', alpha=0.7, label='Beta band upper bound (30 Hz)')
    plt.legend()
    
    # Focus on frequencies up to 100 Hz
    plt.ylim(0, min(100, max(f)))
    
    plt.show()

# %% [markdown]
# ## Summary
# 
# In this notebook, we've explored the Parkinson's Electrophysiological Signal Dataset (PESD) from Dandiset 001333:
# 
# 1. We loaded NWB files from the dataset using the DANDI API
# 2. We explored the metadata and structure of these files
# 3. We visualized LFP data and analyzed its spectral components
# 4. We compared data across multiple recordings and calculated beta band power
# 5. We performed time-frequency analysis to visualize spectral components over time
# 
# The dataset contains simulated electrophysiological signals that model beta oscillations in the subthalamic nucleus, which are relevant to Parkinson's Disease research. The beta band (13-30 Hz) shows notable power, consistent with the dataset description - beta oscillations in the STN are typically used as pathological biomarkers for PD symptoms.
# 
# ## Future Directions
# 
# Further analysis of this dataset could include:
# 
# 1. Comparing the beta band power across all recordings to identify patterns
# 2. Implementing more advanced time-frequency analyses
# 3. Correlating beta oscillation characteristics with other data features
# 4. Using the dataset for testing and validating algorithms for Parkinson's Disease detection or Deep Brain Stimulation control