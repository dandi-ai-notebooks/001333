# %% [markdown]
# # Exploring Dandiset 001333: Parkinson's Electrophysiological Signal Dataset (PESD)
#
# **⚠️ DISCLAIMER: This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results. Always validate findings independently before using for research purposes.**

# %% [markdown]
# ## Overview
#
# The Parkinson's Electrophysiological Signal Dataset (PESD) contains electrophysiological signals from both healthy and parkinsonian subjects. This dataset is particularly valuable for understanding the neural signatures of Parkinson's Disease (PD) and developing therapeutic interventions.
#
# This dataset was generated from a computational model and contains two types of signals for each subject group:
#
# 1. **Beta Average Rectified Voltage (ARV)**: These signals are in the frequency domain and represent the average rectified value of the beta band, calculated by rectifying the filtered LFP signal using a fourth-order Chebyshev band-pass filter with an 8 Hz bandwidth.
#
# 2. **Local Field Potential (LFP) from the Subthalamic Nucleus (STN)**: These signals are in the time domain and derived from the synchronized activity of neuron populations between the cortex, STN, and thalamus.
#
# The beta oscillations (13 to 30 Hz) in the subthalamic nucleus are typically used as pathological biomarkers for PD symptoms. Parkinsonian signals show a relatively high power density at the beta frequency compared to healthy subjects.
#
# **Dandiset Link**: [https://dandiarchive.org/dandiset/001333/0.250327.2220](https://dandiarchive.org/dandiset/001333/0.250327.2220)

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook:

# %%
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import seaborn as sns
import warnings

# Set seaborn style for better visualizations
sns.set_theme()

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', message="Ignoring cached namespace")

# %% [markdown]
# ## Connecting to the DANDI Archive
#
# We'll start by connecting to the DANDI archive and getting information about the dandiset:

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001333", "0.250327.2220")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Dandiset description: {metadata['description'][:300]}...")
print(f"Dandiset citation: {metadata['citation']}")

# %% [markdown]
# ## Exploring the Assets in the Dandiset
#
# This dandiset contains multiple NWB files representing different subjects and sessions. Let's list some of the assets to understand the organization:

# %%
# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# The assets are organized by subject type (healthy vs. parkinsonian) and by signal type (LFP vs. beta).
#
# - `sub-healthy-simulated-beta`: Beta ARV data from healthy subjects
# - `sub-healthy-simulated-lfp`: LFP data from healthy subjects
# - `sub-parkinson-simulated-beta`: Beta ARV data from parkinsonian subjects
# - `sub-parkinson-simulated-lfp`: LFP data from parkinsonian subjects
#
# Each session represents a different recording, indicated by `ses-XXX` in the filename.
#
# For this exploration, we've selected specific NWB files from different sessions to analyze. The selection was based on examining the file structure and ensuring we have a mix of both healthy and parkinsonian data, as well as both beta and LFP signal types.

# %% [markdown]
# ## Loading and Exploring LFP Data
#
# Let's load an LFP data file from a healthy subject to explore its structure:

# %%
# Load a healthy subject LFP file
healthy_lfp_url = "https://api.dandiarchive.org/api/assets/3cf468a1-4933-4aa6-b1c3-4a261b3ee6db/download/"
remote_file = remfile.File(healthy_lfp_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")
print(f"Related publications: {nwb.related_publications}")

# %% [markdown]
# You can explore this file in Neurosift (an interactive NWB viewer) using the following link:
# 
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/3cf468a1-4933-4aa6-b1c3-4a261b3ee6db/download/&dandisetId=001333&dandisetVersion=0.250327.2220](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/3cf468a1-4933-4aa6-b1c3-4a261b3ee6db/download/&dandisetId=001333&dandisetVersion=0.250327.2220)
#
# **Note about data access performance**: Working with remote NWB files can be slow, especially for large datasets. For comprehensive analyses, you might want to download the files locally or use cloud computing resources with faster access to the data.

# %% [markdown]
# Let's examine the structure of the NWB file by looking at the processing modules and available data interfaces:

# %%
# Print information about the processing modules
print("Processing modules:")
for module_name, module in nwb.processing.items():
    print(f"  - {module_name}: {module.description}")
    print("    Data interfaces:")
    for interface_name, interface in module.data_interfaces.items():
        print(f"      - {interface_name}")
        if hasattr(interface, "electrical_series"):
            print("        Electrical series:")
            for es_name, es in interface.electrical_series.items():
                print(f"          - {es_name}: shape {es.data.shape}, rate {es.rate if hasattr(es, 'rate') else 'N/A'} Hz")

# Let's also look at the electrode information
print("\nElectrode information:")
electrode_df = nwb.electrodes.to_dataframe()
print(electrode_df.head())  
print(f"Total electrodes: {len(electrode_df)}")

# %% [markdown]
# ## Visualizing LFP Data
#
# Now, let's visualize a sample of the LFP data from both healthy and parkinsonian subjects:

# %%
# Load parkinsonian subject LFP data
parkinsonian_lfp_url = "https://api.dandiarchive.org/api/assets/28fc91eb-eca2-4c8f-ba41-1cb7a38bcd50/download/"
parkinsonian_remote_file = remfile.File(parkinsonian_lfp_url)
parkinsonian_h5_file = h5py.File(parkinsonian_remote_file)
parkinsonian_io = pynwb.NWBHDF5IO(file=parkinsonian_h5_file)
parkinsonian_nwb = parkinsonian_io.read()

# Get a sample of LFP data from both subjects (first 10,000 points, which at 2000 Hz equals 5 seconds)
# Using [:] to convert h5py dataset to numpy array
healthy_lfp_data = nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data[0:10000]
parkinsonian_lfp_data = parkinsonian_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data[0:10000]

# Get the sampling rate
sampling_rate = nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].rate

# Create a time vector
time = np.arange(len(healthy_lfp_data)) / sampling_rate

# Plot the LFP time series
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, healthy_lfp_data, label='Healthy Subject')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title('Healthy Subject LFP Time Series')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, parkinsonian_lfp_data, label='Parkinsonian Subject')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title('Parkinsonian Subject LFP Time Series')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# We can see that both signals show oscillatory activity. Let's analyze them in the frequency domain to better understand their differences.

# %% [markdown]
# ## Frequency Analysis of LFP Data
#
# Let's perform frequency analysis using Welch's method to compute the power spectral density of the LFP signals:

# %%
# Compute Welch's PSD for both signals
f_healthy, Pxx_healthy = signal.welch(healthy_lfp_data, fs=sampling_rate, nperseg=1024)
f_parkinsonian, Pxx_parkinsonian = signal.welch(parkinsonian_lfp_data, fs=sampling_rate, nperseg=1024)

# Plot the power spectral density
plt.figure(figsize=(12, 6))
plt.semilogy(f_healthy, Pxx_healthy, label='Healthy Subject')
plt.semilogy(f_parkinsonian, Pxx_parkinsonian, label='Parkinsonian Subject')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Power Spectral Density of LFP')
plt.xlim(0, 100)  # Limit to 0-100 Hz for visibility
plt.grid(True)
plt.legend()
plt.show()

# Focus on the beta band (13-30 Hz)
plt.figure(figsize=(10, 6))
beta_mask = (f_healthy >= 13) & (f_healthy <= 30)
plt.plot(f_healthy[beta_mask], Pxx_healthy[beta_mask], label='Healthy Subject', linewidth=2)
plt.plot(f_parkinsonian[beta_mask], Pxx_parkinsonian[beta_mask], label='Parkinsonian Subject', linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Beta Band (13-30 Hz) Power Comparison')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# The power spectral density plots suggest differences in the beta frequency band (13-30 Hz) between healthy and parkinsonian subjects. For this particular sample, the parkinsonian subject appears to show elevated power in the beta band compared to the healthy subject. 
# 
# It's important to note that these observations are based on a limited sample of data and may not necessarily represent statistical differences across the entire dataset. We'll examine more samples later to get a more comprehensive understanding.

# %% [markdown]
# ## Exploring Beta ARV Data
# 
# The dataset also contains Beta Average Rectified Voltage (ARV) data, which is a processed form of the LFP data focusing on the beta band. The Beta ARV is calculated by:
# 
# 1. Filtering the LFP signal using a fourth-order Chebyshev band-pass filter centered at the peak beta frequency with an 8 Hz bandwidth
# 2. Rectifying the filtered signal (taking absolute values)
# 3. Computing the average rectified value over time windows
# 
# This provides a more direct measure of beta band activity that is often used as a biomarker for Parkinson's disease. Let's explore this data:

# %%
# Load a healthy subject Beta ARV file
healthy_beta_url = "https://api.dandiarchive.org/api/assets/73214862-df4b-452b-a35c-d1f3bdb68180/download/"
healthy_remote_file = remfile.File(healthy_beta_url)
healthy_h5_file = h5py.File(healthy_remote_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5_file)
healthy_beta_nwb = healthy_io.read()

# Print basic information
print(f"Subject ID: {healthy_beta_nwb.subject.subject_id}")

# Access the Beta ARV data (using [:] to convert h5py dataset to numpy array)
healthy_beta_data = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].data[:]
healthy_beta_timestamps = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].timestamps[:]

print(f"Beta ARV Data Shape: {healthy_beta_data.shape}")
print(f"Beta ARV Timestamps Shape: {healthy_beta_timestamps.shape}")
print(f"Time range: {healthy_beta_timestamps[0]:.2f} to {healthy_beta_timestamps[-1]:.2f} seconds")

# %% [markdown]
# Let's load a parkinsonian Beta ARV file for comparison:

# %%
# Load a parkinsonian subject Beta ARV file
parkinsonian_beta_url = "https://api.dandiarchive.org/api/assets/712fd6c0-5543-476d-9493-7bdb652acdf2/download/"
parkinsonian_remote_file = remfile.File(parkinsonian_beta_url)
parkinsonian_h5_file = h5py.File(parkinsonian_remote_file)
parkinsonian_io = pynwb.NWBHDF5IO(file=parkinsonian_h5_file)
parkinsonian_beta_nwb = parkinsonian_io.read()

# Print basic information
print(f"Subject ID: {parkinsonian_beta_nwb.subject.subject_id}")

# Access the Beta ARV data (using [:] to convert h5py dataset to numpy array)
parkinsonian_beta_data = parkinsonian_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].data[:]
parkinsonian_beta_timestamps = parkinsonian_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].timestamps[:]

print(f"Beta ARV Data Shape: {parkinsonian_beta_data.shape}")
print(f"Beta ARV Timestamps Shape: {parkinsonian_beta_timestamps.shape}")

# %% [markdown]
# Now, let's visualize and compare the Beta ARV data from both subjects:

# %%
# Plot the Beta ARV data
plt.figure(figsize=(12, 6))
plt.plot(healthy_beta_timestamps, healthy_beta_data, label='Healthy Subject')
plt.plot(parkinsonian_beta_timestamps, parkinsonian_beta_data, label='Parkinsonian Subject')
plt.xlabel('Time (s)')
plt.ylabel('Beta ARV (V)')
plt.title('Comparison: Healthy vs. Parkinsonian Beta ARV')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Let's compute the statistics of the Beta ARV data to quantify the differences:

# %%
# Compute mean and standard deviation
healthy_beta_mean = np.mean(healthy_beta_data)
healthy_beta_std = np.std(healthy_beta_data)
parkinsonian_beta_mean = np.mean(parkinsonian_beta_data)
parkinsonian_beta_std = np.std(parkinsonian_beta_data)

print(f"Healthy Beta ARV: Mean = {healthy_beta_mean:.8f}V, Std = {healthy_beta_std:.8f}V")
print(f"Parkinsonian Beta ARV: Mean = {parkinsonian_beta_mean:.8f}V, Std = {parkinsonian_beta_std:.8f}V")
print(f"Ratio (Parkinsonian/Healthy): Mean = {parkinsonian_beta_mean/healthy_beta_mean:.2f}, Std = {parkinsonian_beta_std/healthy_beta_std:.2f}")

# Plot the Beta ARV distributions
plt.figure(figsize=(10, 6))
plt.hist(healthy_beta_data, bins=30, alpha=0.5, label='Healthy Subject')
plt.hist(parkinsonian_beta_data, bins=30, alpha=0.5, label='Parkinsonian Subject')
plt.xlabel('Beta ARV (V)')
plt.ylabel('Count')
plt.title('Distribution of Beta ARV Values')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# The Beta ARV data from these specific samples shows that the parkinsonian subject has higher beta band activity compared to the healthy subject. For this pair of subjects, the mean Beta ARV in the parkinsonian subject is approximately 3.6 times higher than in the healthy subject, with distinct distributions.
#
# This observation is consistent with what we would expect based on the dataset description, which indicates that parkinsonian signals typically show higher power in the beta band. However, it's important to note that this is based on a single pair of subjects, and further statistical analysis with more samples would be needed to generalize these findings.

# %% [markdown]
# ## Comparing Multiple Sessions
#
# So far, we've only looked at individual sessions. Let's compare data across multiple sessions within each subject group to investigate variability. We'll make sure to load different session data for a proper comparison:

# %%
# Define asset IDs for different sessions
healthy_sessions = [
    {"asset_id": "3cf468a1-4933-4aa6-b1c3-4a261b3ee6db", "session": "54"},
    {"asset_id": "d92648ad-a2f8-4ec6-a125-363f45aa7f35", "session": "218"},
    {"asset_id": "b29a7d22-0d2c-4067-9436-73599e70b112", "session": "279"}
]

parkinsonian_sessions = [
    {"asset_id": "6aa013b8-536c-4556-9730-94b71ae26c55", "session": "51"},
    {"asset_id": "5535e23a-9029-43c5-80fb-0fb596541a81", "session": "111"},
    {"asset_id": "28fc91eb-eca2-4c8f-ba41-1cb7a38bcd50", "session": "93"}
]

# Function to load data from a session
def load_session_data(session_info):
    """
    Load LFP data from a specific session.
    
    Parameters:
    -----------
    session_info : dict
        Dictionary with asset_id and session information
        
    Returns:
    --------
    data : array
        LFP data
    session_id : str
        Session ID
    """
    asset_id = session_info["asset_id"]
    session = session_info["session"]
    
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    # Get a subset of the data to avoid loading too much
    # Using different starting points to ensure we get different data samples
    start = int(session) % 10000  # Use session number to get different starting points
    if start + 4000 > nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data.shape[0]:
        start = 0
    
    data = nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data[start:start+4000]
    
    return data, f"{nwb.subject.subject_id}-{session}"

# Load data from each session
healthy_data = []
parkinsonian_data = []

for session_info in healthy_sessions:
    data, session_id = load_session_data(session_info)
    healthy_data.append((data, session_id))
    print(f"Loaded {session_id}, data shape: {data.shape}")

for session_info in parkinsonian_sessions:
    data, session_id = load_session_data(session_info)
    parkinsonian_data.append((data, session_id))
    print(f"Loaded {session_id}, data shape: {data.shape}")

# Plot a comparison of sessions
plt.figure(figsize=(15, 10))

# Plot healthy sessions
for i, (data, subject_id) in enumerate(healthy_data):
    plt.subplot(2, 3, i+1)
    plt.plot(data)
    plt.title(f"Healthy Session: {subject_id}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (V)")

# Plot parkinsonian sessions
for i, (data, subject_id) in enumerate(parkinsonian_data):
    plt.subplot(2, 3, i+4)
    plt.plot(data)
    plt.title(f"Parkinsonian Session: {subject_id}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (V)")

plt.tight_layout()
plt.show()

# %% [markdown]
# We've loaded data from different time segments of each session to ensure we're examining distinct data. Now let's compute the power spectral density for each session and compare them:

# %%
plt.figure(figsize=(15, 10))

# Compute and plot PSD for healthy sessions
plt.subplot(2, 1, 1)
for i, (data, subject_id) in enumerate(healthy_data):
    f, Pxx = signal.welch(data, fs=2000, nperseg=1024)
    plt.semilogy(f, Pxx, label=f"Healthy Session: {subject_id}")

plt.title("Power Spectral Density: Healthy Sessions")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V^2/Hz)")
plt.xlim(0, 100)
plt.grid(True)
plt.legend()

# Compute and plot PSD for parkinsonian sessions
plt.subplot(2, 1, 2)
for i, (data, subject_id) in enumerate(parkinsonian_data):
    f, Pxx = signal.welch(data, fs=2000, nperseg=1024)
    plt.semilogy(f, Pxx, label=f"Parkinsonian Session: {subject_id}")

plt.title("Power Spectral Density: Parkinsonian Sessions")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V^2/Hz)")
plt.xlim(0, 100)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Focused Analysis of Beta Band

# %%
plt.figure(figsize=(10, 6))

# Plot the beta band (13-30 Hz) for all sessions
beta_mask = (f >= 13) & (f <= 30)

# Plot healthy sessions in shades of blue
blue_colors = ['#1f77b4', '#7bafd2', '#9ecae1']
for i, (data, subject_id) in enumerate(healthy_data):
    f, Pxx = signal.welch(data, fs=2000, nperseg=1024)
    plt.plot(f[beta_mask], Pxx[beta_mask], label=f"Healthy: {subject_id}", 
             linestyle='--', color=blue_colors[i], linewidth=2)

# Plot parkinsonian sessions in shades of orange
orange_colors = ['#ff7f0e', '#ffbb78', '#ffa54f']
for i, (data, subject_id) in enumerate(parkinsonian_data):
    f, Pxx = signal.welch(data, fs=2000, nperseg=1024)
    plt.plot(f[beta_mask], Pxx[beta_mask], label=f"Parkinsonian: {subject_id}",
             color=orange_colors[i], linewidth=2)

# Add vertical line to highlight the parkinsonian beta peak
plt.axvline(x=27, linestyle=':', color='red', label='Typical PD Beta Peak', alpha=0.7)

plt.title("Beta Band (13-30 Hz) Power Comparison Across Sessions")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V^2/Hz)")
plt.grid(True)
plt.legend(fontsize=8)
plt.show()

# %% [markdown]
# From this plot, we observe variability in the beta band power both within and between groups. Some parkinsonian sessions show higher beta power than healthy sessions, while others show similar levels. This variability is expected in electrophysiological data, even within the same condition.
# 
# It's important to note that our sample size (3 sessions per group) is very small, and a larger sample would be needed to draw statistically robust conclusions about group differences.

# %% [markdown]
# ## Advanced Visualization: Time-Frequency Analysis
#
# To better understand how the frequency content changes over time, we can perform a time-frequency analysis using a spectrogram:

# %%
# Function to compute and plot spectrogram
def plot_spectrogram(data, sampling_rate, title):
    """
    Compute and plot a spectrogram for the data.
    
    Parameters:
    -----------
    data : array
        Time series data
    sampling_rate : float
        Sampling rate in Hz
    title : str
        Title for the plot
    """
    # Compute spectrogram with improved parameters for better visualization
    f, t, Sxx = signal.spectrogram(data, fs=sampling_rate, nperseg=256, noverlap=200)
    
    # Apply log10 to improve visualization
    # Add small value to avoid log(0)
    Sxx_log = 10 * np.log10(Sxx + 1e-12)
    
    # Plot
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='PSD [dB/Hz]')
    plt.ylim(0, 100)  # Limit the frequency range for better visibility

# Create a figure with two spectrograms
plt.figure(figsize=(12, 10))

# Healthy subject spectrogram
plt.subplot(2, 1, 1)
plot_spectrogram(healthy_data[0][0], 2000, f"Spectrogram: Healthy Subject {healthy_data[0][1]}")

# Parkinsonian subject spectrogram
plt.subplot(2, 1, 2)
plot_spectrogram(parkinsonian_data[0][0], 2000, f"Spectrogram: Parkinsonian Subject {parkinsonian_data[0][1]}")

plt.tight_layout()
plt.show()

# %% [markdown]
# Let's improve our visualization of the beta band in the spectrogram:

# %%
# Create a figure with two spectrograms, focused on beta band with improved visualization
plt.figure(figsize=(12, 10))

# Enhance spectrograms to better show differences in beta band
def plot_beta_spectrogram(data, sampling_rate, title, beta_highlight=False):
    """
    Plot a spectrogram with enhanced beta band visualization.
    
    Parameters:
    -----------
    data : array
        Time series data
    sampling_rate : float
        Sampling rate in Hz
    title : str
        Title for the plot
    beta_highlight : bool
        Whether to highlight the beta band
    """
    # Use smaller window and more overlap for better time-frequency resolution
    f, t, Sxx = signal.spectrogram(data, fs=sampling_rate, nperseg=256, noverlap=224)
    
    # Apply log10 to improve visualization
    Sxx_log = 10 * np.log10(Sxx + 1e-12)
    
    # Plot spectrogram
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
    
    # Highlight beta band if requested
    if beta_highlight:
        plt.axhline(y=13, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='red', linestyle='--', alpha=0.7)
    
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='PSD [dB/Hz]')
    plt.ylim(10, 35)  # Focus on beta band and surrounding frequencies

# Healthy subject spectrogram
plt.subplot(2, 1, 1)
plot_beta_spectrogram(healthy_data[0][0], 2000, f"Beta Band Spectrogram: Healthy Subject {healthy_data[0][1]}", True)

# Parkinsonian subject spectrogram
plt.subplot(2, 1, 2)
plot_beta_spectrogram(parkinsonian_data[0][0], 2000, f"Beta Band Spectrogram: Parkinsonian Subject {parkinsonian_data[0][1]}", True)

plt.tight_layout()
plt.show()

# %% [markdown]
# The spectrograms don't show dramatic differences between healthy and parkinsonian subjects in the beta band for these particular data segments. This is interesting given the clearer differences we saw in the Beta ARV data. There are several possible explanations for this:
# 
# 1. The spectrogram might not be the most sensitive method for detecting beta band differences in these particular data segments
# 2. The beta differences might be more apparent in other time segments
# 3. The Beta ARV calculation method used to generate those files may be more sensitive to specific beta band features than standard spectrograms
# 4. The differences may be more subtle and require more advanced analysis techniques to visualize clearly
# 
# This highlights the importance of using multiple analysis approaches when working with complex neural data.

# %% [markdown]
# Let's also look at a direct comparison of spectrograms between all sessions:

# %%
# Create a figure that compares all sessions with spectrograms
plt.figure(figsize=(15, 12))

# Plot healthy sessions
for i, (data, subject_id) in enumerate(healthy_data):
    plt.subplot(2, 3, i+1)
    f, t, Sxx = signal.spectrogram(data, fs=2000, nperseg=256, noverlap=200)
    Sxx_log = 10 * np.log10(Sxx + 1e-12)
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
    plt.title(f"Healthy: {subject_id}")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(10, 35)  # Focus on beta band and surrounding
    plt.colorbar(label='PSD [dB/Hz]')

# Plot parkinsonian sessions
for i, (data, subject_id) in enumerate(parkinsonian_data):
    plt.subplot(2, 3, i+4)
    f, t, Sxx = signal.spectrogram(data, fs=2000, nperseg=256, noverlap=200)
    Sxx_log = 10 * np.log10(Sxx + 1e-12)
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
    plt.title(f"Parkinsonian: {subject_id}")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(10, 35)  # Focus on beta band and surrounding
    plt.colorbar(label='PSD [dB/Hz]')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Statistical Comparison of Beta Power
#
# Let's perform a statistical analysis to quantify the differences in beta band power between healthy and parkinsonian subjects, keeping in mind the limitations of our small sample size:

# %%
import scipy.stats as stats

# Function to extract beta band power from a time series
def compute_beta_power(data, fs=2000):
    """
    Compute the average power in the beta band (13-30 Hz).
    
    Parameters:
    -----------
    data : array
        Time series data
    fs : float
        Sampling rate
        
    Returns:
    --------
    beta_power : float
        Average power in the beta band
    """
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    beta_mask = (f >= 13) & (f <= 30)
    return np.mean(Pxx[beta_mask])

# Compute beta power for each session
healthy_beta_powers = [compute_beta_power(data) for data, _ in healthy_data]
parkinsonian_beta_powers = [compute_beta_power(data) for data, _ in parkinsonian_data]

print("Beta band power (V^2/Hz):")
print("Healthy subjects:", healthy_beta_powers)
print("Parkinsonian subjects:", parkinsonian_beta_powers)
print(f"Mean healthy: {np.mean(healthy_beta_powers):.4e}")
print(f"Mean parkinsonian: {np.mean(parkinsonian_beta_powers):.4e}")
print(f"Ratio (parkinsonian/healthy): {np.mean(parkinsonian_beta_powers)/np.mean(healthy_beta_powers):.2f}x")

# Perform t-test to assess statistical significance
t_stat, p_val = stats.ttest_ind(healthy_beta_powers, parkinsonian_beta_powers)
print(f"\nt-test: t={t_stat:.4f}, p={p_val:.4f}")
if p_val < 0.05:
    print("The difference in beta power between healthy and parkinsonian subjects is statistically significant (p < 0.05).")
else:
    print("The difference in beta power is not statistically significant (p >= 0.05).")
    print("Note: This is likely due to our very small sample size (n=3 per group) and the variability in the data.")

# Visualize the beta power comparison
plt.figure(figsize=(8, 6))
labels = ['Healthy', 'Parkinsonian']
powers = [healthy_beta_powers, parkinsonian_beta_powers]

plt.boxplot(powers, labels=labels)
plt.title('Beta Band Power Comparison (Small Sample Size, n=3 per group)')
plt.ylabel('Power (V^2/Hz)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add individual data points
for i, data in enumerate([healthy_beta_powers, parkinsonian_beta_powers], 1):
    plt.scatter([i] * len(data), data, alpha=0.7, s=60)

plt.show()

# %% [markdown]
# **Important Note on Statistical Results**: The t-test does not show a statistically significant difference between the groups (p > 0.05). However, this is likely due to our very small sample size (n=3 per group) and the high variability in the data. The limited number of samples severely restricts our ability to detect statistically significant differences, even if real differences exist in the underlying population.
# 
# In real research, you would need many more samples to draw robust statistical conclusions. The trending difference we observe (parkinsonian mean > healthy mean) is consistent with the dataset description and prior research on Parkinson's disease, but our analysis here should be considered exploratory rather than confirmatory.
# 
# The Beta ARV data (examined earlier) provides an alternative measure that showed clearer differences between the parkinsonian and healthy conditions, possibly because it was specifically designed to isolate and quantify the beta activity most relevant to Parkinson's disease.

# %% [markdown]
# ## Relationship Between Beta ARV and LFP Data
# 
# Let's briefly examine the relationship between the raw LFP data and the derived Beta ARV data to better understand how they relate:

# %%
# Load a pair of matching LFP and Beta ARV data for the same session
# For healthy subject
healthy_lfp_url = "https://api.dandiarchive.org/api/assets/3cf468a1-4933-4aa6-b1c3-4a261b3ee6db/download/"
healthy_remote_file = remfile.File(healthy_lfp_url)
healthy_h5_file = h5py.File(healthy_remote_file)
healthy_io = pynwb.NWBHDF5IO(file=healthy_h5_file)
healthy_nwb = healthy_io.read()

healthy_beta_url = "https://api.dandiarchive.org/api/assets/73214862-df4b-452b-a35c-d1f3bdb68180/download/"
healthy_beta_remote_file = remfile.File(healthy_beta_url)
healthy_beta_h5_file = h5py.File(healthy_beta_remote_file)
healthy_beta_io = pynwb.NWBHDF5IO(file=healthy_beta_h5_file)
healthy_beta_nwb = healthy_beta_io.read()

# Verify they're from the same subject
print(f"Healthy LFP Subject: {healthy_nwb.subject.subject_id}")
print(f"Healthy Beta Subject: {healthy_beta_nwb.subject.subject_id}")

# Get a segment of LFP data at the same time as the Beta ARV
# Beta timestamps start at 10.02 seconds, so we need to offset our LFP data
beta_start_time = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].timestamps[0]
beta_end_time = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].timestamps[-1]
print(f"Beta ARV time range: {beta_start_time:.2f}s to {beta_end_time:.2f}s")

# Calculate LFP indices corresponding to beta time range
lfp_sampling_rate = healthy_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].rate
start_idx = int(beta_start_time * lfp_sampling_rate)
end_idx = int(beta_end_time * lfp_sampling_rate)

# Get LFP data for this time range
lfp_data = healthy_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['LFP'].data[start_idx:end_idx]
lfp_time = np.linspace(beta_start_time, beta_end_time, len(lfp_data))

# Get Beta ARV data
beta_data = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].data[:]
beta_time = healthy_beta_nwb.processing['ecephys'].data_interfaces['LFP'].electrical_series['Beta_Band_Voltage'].timestamps[:]

# Plot them together
plt.figure(figsize=(12, 8))

# Plot LFP data
plt.subplot(2, 1, 1)
plt.plot(lfp_time[:5000], lfp_data[:5000])
plt.title('Raw LFP Data (Healthy Subject)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True, alpha=0.3)

# Plot Beta ARV data for the same time period
plt.subplot(2, 1, 2)
beta_mask = (beta_time >= beta_start_time) & (beta_time <= (beta_start_time + 2.5))  # ~2.5 seconds to match LFP plot
plt.plot(beta_time[beta_mask], beta_data[beta_mask])
plt.title('Derived Beta ARV Data (Healthy Subject)')
plt.xlabel('Time (s)')
plt.ylabel('Beta ARV (V)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# This visualization illustrates the relationship between the raw LFP data and the derived Beta ARV data. The Beta ARV represents a processed form of the LFP signal that specifically captures the average rectified activity in the beta frequency band. While the raw LFP contains signals across all frequency bands, the Beta ARV isolates and quantifies the beta activity that is particularly relevant for Parkinson's disease.
# 
# The different time scales (2000 Hz for LFP vs. ~40 Hz for Beta ARV) reflect the different sampling rates of these signals, with the Beta ARV being a more compressed representation focused on changes in beta power over time.

# %% [markdown]
# ## Summary and Findings
#
# In this notebook, we explored the Parkinson's Electrophysiological Signal Dataset (PESD), which contains simulated electrophysiological signals from both healthy and parkinsonian subjects. Here are the key findings:
#
# 1. **Dataset Structure**: The dataset is organized by subject type (healthy vs. parkinsonian) and signal type (LFP vs. beta ARV).
#
# 2. **LFP Data**: The Local Field Potential (LFP) data represents raw electrophysiological activity and is sampled at 2000 Hz. Both healthy and parkinsonian subjects show oscillatory activity.
#
# 3. **Beta Band Activity**: Our exploratory analysis suggests that parkinsonian subjects tend to show increased power in the beta frequency band (13-30 Hz) compared to healthy subjects, although there is considerable variability between sessions. This trend is consistent with the known pathophysiology of Parkinson's disease, where abnormal beta oscillations in the subthalamic nucleus are a feature often associated with motor symptoms.
#
# 4. **Beta ARV Data**: The Beta Average Rectified Voltage (ARV) data provides a processed measure of beta band activity. In our case study comparing a healthy and parkinsonian subject, we found that the mean Beta ARV in the parkinsonian subject was approximately 3.6 times higher than in the healthy subject, with clear separation in their distributions.
#
# 5. **Statistical Considerations**: Our statistical analysis with a small sample (n=3 per group) did not reach statistical significance (p > 0.05), highlighting the limitations of working with small sample sizes. Larger samples would be needed for robust statistical inference.
#
# 6. **Visualization Methods**: Different visualization approaches (spectrograms, power spectral density plots, time series) provided complementary perspectives on the data, with some methods showing clearer differences than others.

# %% [markdown]
# ## Future Directions
#
# This dataset provides valuable data for researchers interested in:
#
# 1. **Developing Biomarkers**: The differences in beta band activity could be explored further to develop biomarkers for Parkinson's disease.
#
# 2. **Testing Algorithms**: Researchers can use this dataset to test and validate signal processing algorithms or machine learning models for detecting or classifying parkinsonian vs. healthy neural activity.
#
# 3. **Simulating Interventions**: The dataset could be used to simulate the effects of different interventions, such as deep brain stimulation, on pathological neural activity.
#
# 4. **Educational Purposes**: The dataset provides examples of the electrophysiological differences between healthy and parkinsonian neural activity, making it valuable for teaching and learning.
#
# Further analyses could include:
#
# - More advanced signal processing techniques (e.g., phase-amplitude coupling)
# - Machine learning classification of healthy vs. parkinsonian signals
# - Correlation analyses between LFP and Beta ARV data
# - Exploration of different frequency bands beyond beta
# - Modeling the transition from healthy to parkinsonian states
# - Larger-scale statistical analyses with more sessions to obtain more robust findings
#
# **Computational Considerations**: Note that working with remote NWB files can be slow for extensive analyses. For comprehensive studies, researchers may want to download the dataset locally or use cloud computing resources that provide faster access to the data.