"""
This script explores the structure of an NWB file from Dandiset 001333 to better understand
what types of data are available.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# URL of the NWB file to explore
URL = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"

def load_nwb(url):
    """Load an NWB file from a URL."""
    print(f"Loading NWB file from {url}...")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    print("NWB file loaded successfully")
    return nwb, h5_file

def print_nwb_info(nwb):
    """Print basic information about the NWB file."""
    print("\n=== NWB File Information ===")
    print(f"Session description: {nwb.session_description}")
    print(f"Identifier: {nwb.identifier}")
    print(f"Session start time: {nwb.session_start_time}")
    print(f"Experiment description: {nwb.experiment_description}")
    print(f"Lab: {nwb.lab}")
    print(f"Institution: {nwb.institution}")
    print(f"Keywords: {nwb.keywords[:]}")
    print(f"Related publications: {nwb.related_publications}")
    print(f"Experimenter: {nwb.experimenter}")
    
    # Subject information
    print("\n=== Subject Information ===")
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Species: {nwb.subject.species}")
    print(f"Sex: {nwb.subject.sex}")
    print(f"Age: {nwb.subject.age}")
    print(f"Description: {nwb.subject.description}")

def explore_data(nwb, h5_file):
    """Explore the data in the NWB file."""
    print("\n=== Data Exploration ===")
    
    # Check whether it contains LFP data
    if "ecephys" in nwb.processing:
        print("\nProcessing modules:")
        print(f"- ecephys: {nwb.processing['ecephys'].description}")
        
        if "LFP" in nwb.processing["ecephys"].data_interfaces:
            lfp = nwb.processing["ecephys"].data_interfaces["LFP"]
            print("\nLFP electrical series:")
            
            for es_name, es in lfp.electrical_series.items():
                print(f"\n- {es_name}:")
                print(f"  Description: {es.description}")
                print(f"  Unit: {es.unit}")
                print(f"  Data shape: {es.data.shape}")
                print(f"  Timestamps shape: {es.timestamps.shape}")
                
                # Load a sample of the data for analysis
                print("  Loading sample data...")
                data_sample = es.data[:]  # Get all data (it's a small file)
                timestamps_sample = es.timestamps[:]
                
                print(f"  Data statistics:")
                print(f"    Min: {np.min(data_sample):.6f}")
                print(f"    Max: {np.max(data_sample):.6f}")
                print(f"    Mean: {np.mean(data_sample):.6f}")
                print(f"    Std: {np.std(data_sample):.6f}")
                
                print(f"  Timestamps statistics:")
                print(f"    Start: {timestamps_sample[0]:.6f} seconds")
                print(f"    End: {timestamps_sample[-1]:.6f} seconds")
                print(f"    Duration: {timestamps_sample[-1] - timestamps_sample[0]:.6f} seconds")
                
                # Plot the data
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps_sample, data_sample)
                plt.title(f"LFP {es_name}")
                plt.xlabel("Time (seconds)")
                plt.ylabel(f"Voltage ({es.unit})")
                plt.grid(True)
                plt.savefig(f"lfp_{es_name}_timeseries.png")
                plt.close()
                
                # Also generate frequency domain plot
                plt.figure(figsize=(12, 6))
                fs = 1.0 / (timestamps_sample[1] - timestamps_sample[0])  # Sampling frequency
                n = len(data_sample)
                freqs = np.fft.rfftfreq(n, d=1/fs)
                fft_vals = np.abs(np.fft.rfft(data_sample))
                
                plt.semilogy(freqs, fft_vals)
                plt.title(f"LFP {es_name} - Frequency Domain")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Amplitude")
                plt.grid(True)
                plt.savefig(f"lfp_{es_name}_frequency.png")
                plt.close()
                
                # Generate a spectrogram
                plt.figure(figsize=(12, 6))
                plt.specgram(data_sample, Fs=fs, cmap='viridis')
                plt.title(f"LFP {es_name} - Spectrogram")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Frequency (Hz)")
                plt.colorbar(label="Power/Frequency (dB/Hz)")
                plt.savefig(f"lfp_{es_name}_spectrogram.png")
                plt.close()

    # Check electrodes
    if nwb.electrodes is not None:
        print("\nElectrodes information:")
        electrode_df = nwb.electrodes.to_dataframe()
        print(f"Number of electrodes: {len(electrode_df)}")
        print("Electrode locations:")
        locations = electrode_df['location'].unique()
        for loc in locations:
            count = len(electrode_df[electrode_df['location'] == loc])
            print(f"- {loc}: {count} electrodes")
        
        print("\nElectrode groups:")
        for group_name, group in nwb.electrode_groups.items():
            print(f"- {group_name}: {group.description} (Location: {group.location})")

def main():
    try:
        # Load the NWB file
        nwb, h5_file = load_nwb(URL)
        
        # Print general information about the NWB file
        print_nwb_info(nwb)
        
        # Explore the data in the NWB file
        explore_data(nwb, h5_file)
        
        print("\nExploration complete!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()