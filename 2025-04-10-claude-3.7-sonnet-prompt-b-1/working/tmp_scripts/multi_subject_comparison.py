# This script compares data across multiple subjects to better understand the dataset

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb
import pandas as pd
from scipy import signal
import requests
import random
from collections import defaultdict

# Get asset list for multiple subjects
def get_assets_by_subject_type():
    url = "https://api.dandiarchive.org/api/dandisets/001333/versions/draft/assets/?page_size=200"
    assets_by_type = defaultdict(list)
    
    while url:
        response = requests.get(url)
        data = response.json()
        
        for asset in data['results']:
            path = asset['path']
            parts = path.split('/')
            if len(parts) > 0 and parts[0].startswith('sub-'):
                subject_type = parts[0]
                assets_by_type[subject_type].append({
                    'path': path,
                    'asset_id': asset['asset_id'],
                    'url': f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
                })
        
        url = data.get('next')
        if not url:
            break
    
    return assets_by_type

def sample_assets(assets_by_type, sample_size=3):
    """Sample assets from each subject type"""
    sampled_assets = {}
    for subject_type, assets in assets_by_type.items():
        if len(assets) > sample_size:
            sampled_assets[subject_type] = random.sample(assets, sample_size)
        else:
            sampled_assets[subject_type] = assets
    return sampled_assets

def compute_beta_power(url):
    """Compute beta band (13-30 Hz) power for a given asset"""
    try:
        file = remfile.File(url)
        h5 = h5py.File(file)
        io = pynwb.NWBHDF5IO(file=h5)
        nwb = io.read()
        
        # Check if it's a beta or LFP file
        if "Beta_Band_Voltage" in nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series:
            data = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data[:]
            # For beta files, just compute the mean power (squared amplitude)
            beta_power = np.mean(np.square(data))
            
        elif "LFP" in nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series:
            # For LFP files, get a subset and compute spectral power in beta band
            data_subset = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].data[0:4000]
            fs = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"].rate
            
            # Compute power spectral density
            freq, psd = signal.welch(data_subset, fs=fs, nperseg=1024, scaling='spectrum')
            
            # Extract beta band (13-30 Hz)
            beta_idx = np.where((freq >= 13) & (freq <= 30))[0]
            beta_power = np.mean(psd[beta_idx])
        
        return beta_power, nwb.subject.subject_id
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, None

# Let's examine electrode information from one file to better understand the recording setup
def examine_electrode_info(url):
    try:
        file = remfile.File(url)
        h5 = h5py.File(file)
        io = pynwb.NWBHDF5IO(file=h5)
        nwb = io.read()
        
        # Get electrode information
        electrode_groups = list(nwb.electrode_groups.keys())
        
        # Create a table of electrode information if available
        electrode_data = []
        if hasattr(nwb, 'electrodes') and len(nwb.electrodes.id) > 0:
            for i in range(len(nwb.electrodes.id)):
                electrode_info = {}
                for col in nwb.electrodes.colnames:
                    try:
                        electrode_info[col] = nwb.electrodes[col][i]
                    except:
                        electrode_info[col] = None
                electrode_data.append(electrode_info)
            
        return electrode_groups, electrode_data
    except Exception as e:
        print(f"Error examining electrodes {url}: {e}")
        return [], []

# Main function
def main():
    # Get assets by subject type
    print("Fetching assets...")
    assets_by_type = get_assets_by_subject_type()
    print(f"Found {sum(len(v) for v in assets_by_type.values())} assets across {len(assets_by_type)} subject types")
    
    # Sample assets to avoid processing all files
    sampled_assets = sample_assets(assets_by_type)
    
    # Look at electrode information from one file
    print("\nExamining electrode information...")
    sample_url = sampled_assets["sub-healthy-simulated-beta"][0]["url"]
    electrode_groups, electrode_data = examine_electrode_info(sample_url)
    
    print(f"Electrode groups: {electrode_groups}")
    if electrode_data:
        print("\nElectrode data sample:")
        df = pd.DataFrame(electrode_data[:5])  # Show first 5 rows
        print(df)
    
    # Compute beta power for each sampled asset
    print("\nComputing beta power for sample files...")
    results = []
    
    for subject_type, assets in sampled_assets.items():
        for asset in assets:
            print(f"Processing {asset['path']}...")
            beta_power, subject_id = compute_beta_power(asset['url'])
            if beta_power is not None:
                results.append({
                    'subject_type': subject_type,
                    'asset_id': asset['asset_id'],
                    'path': asset['path'],
                    'beta_power': beta_power,
                    'subject_id': subject_id
                })
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    print("\nResults summary:")
    print(results_df.groupby('subject_type')[['beta_power']].describe())
    
    # Plot beta power by subject type as a boxplot
    plt.figure(figsize=(12, 6))
    
    # Sort the dataframe by subject type to ensure consistent order
    plot_data = []
    labels = []
    
    for subject_type in sorted(results_df['subject_type'].unique()):
        plot_data.append(results_df[results_df['subject_type'] == subject_type]['beta_power'])
        # Create readable labels
        label = subject_type.replace('sub-', '').replace('-', ' ')
        labels.append(label)
    
    # Create boxplot
    plt.boxplot(plot_data, labels=labels)
    plt.title('Beta Power Comparison Across Subject Types')
    plt.ylabel('Beta Power (Logarithmic Scale)')
    plt.yscale('log')  # Use log scale for better visualization
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tmp_scripts/multi_subject_comparison.png')
    
    # Create individual subject plots for each file type
    plt.figure(figsize=(14, 8))
    
    # Split the figure into beta and lfp types
    beta_df = results_df[results_df['subject_type'].str.contains('beta')]
    lfp_df = results_df[results_df['subject_type'].str.contains('lfp')]
    
    plt.subplot(1, 2, 1)
    for subject_type in sorted(beta_df['subject_type'].unique()):
        data = beta_df[beta_df['subject_type'] == subject_type]
        label = subject_type.replace('sub-', '').replace('-', ' ')
        plt.scatter(data['subject_id'], data['beta_power'], label=label, alpha=0.7)
    
    plt.xlabel('Subject ID')
    plt.ylabel('Beta Power')
    plt.title('Beta Files: Individual Subject Comparison')
    plt.yscale('log')  # Use log scale for better visualization
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for subject_type in sorted(lfp_df['subject_type'].unique()):
        data = lfp_df[lfp_df['subject_type'] == subject_type]
        label = subject_type.replace('sub-', '').replace('-', ' ')
        plt.scatter(data['subject_id'], data['beta_power'], label=label, alpha=0.7)
    
    plt.xlabel('Subject ID')
    plt.ylabel('Beta Power')
    plt.title('LFP Files: Individual Subject Comparison')
    plt.yscale('log')  # Use log scale for better visualization
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tmp_scripts/individual_subject_comparison.png')

if __name__ == "__main__":
    main()