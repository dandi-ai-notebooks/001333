"""
This script compares the data between two NWB files from the same subject type
to see if there are differences between sessions.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# URLs of the NWB files to compare
URL1 = "https://api.dandiarchive.org/api/assets/1d94c7ad-dbaf-43ea-89f2-1b2518fab158/download/"
URL2 = "https://api.dandiarchive.org/api/assets/e0fa57b2-02a4-4c20-92df-d7eb64b60170/download/"

def load_nwb(url):
    """Load an NWB file from a URL."""
    print(f"Loading NWB file from {url}...")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    print(f"NWB file loaded successfully: {nwb.identifier}")
    return nwb, h5_file

def extract_data(nwb):
    """Extract the LFP data from the NWB file."""
    lfp_data = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].data[:]
    timestamps = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["Beta_Band_Voltage"].timestamps[:]
    subject_id = nwb.subject.subject_id
    session_id = nwb.identifier
    
    return {
        "data": lfp_data,
        "timestamps": timestamps,
        "subject_id": subject_id,
        "session_id": session_id
    }

def compare_time_series(data1, data2):
    """Compare the time series data between two NWB files."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot the first time series
    axes[0].plot(data1["timestamps"], data1["data"], label=f"Session: {data1['session_id'][:8]}...")
    axes[0].set_title(f"Time Series - {data1['subject_id']} - Session 1")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Voltage (volts)")
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot the second time series
    axes[1].plot(data2["timestamps"], data2["data"], label=f"Session: {data2['session_id'][:8]}...")
    axes[1].set_title(f"Time Series - {data2['subject_id']} - Session 2")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Voltage (volts)")
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot both time series for direct comparison
    axes[2].plot(data1["timestamps"], data1["data"], label=f"Session 1", alpha=0.7)
    axes[2].plot(data2["timestamps"], data2["data"], label=f"Session 2", alpha=0.7)
    axes[2].set_title("Direct Comparison of Time Series")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Voltage (volts)")
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("time_series_comparison.png")
    plt.close()
    
    print("Time series comparison plot saved to time_series_comparison.png")

def compare_frequency_domain(data1, data2):
    """Compare the frequency domain data between two NWB files."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Calculate sampling frequency
    fs1 = 1.0 / (data1["timestamps"][1] - data1["timestamps"][0])
    fs2 = 1.0 / (data2["timestamps"][1] - data2["timestamps"][0])
    
    # Calculate the FFTs
    freqs1 = np.fft.rfftfreq(len(data1["data"]), d=1/fs1)
    fft_vals1 = np.abs(np.fft.rfft(data1["data"]))
    
    freqs2 = np.fft.rfftfreq(len(data2["data"]), d=1/fs2)
    fft_vals2 = np.abs(np.fft.rfft(data2["data"]))
    
    # Plot the first frequency domain
    axes[0].semilogy(freqs1, fft_vals1, label=f"Session: {data1['session_id'][:8]}...")
    axes[0].set_title(f"Frequency Domain - {data1['subject_id']} - Session 1")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot the second frequency domain
    axes[1].semilogy(freqs2, fft_vals2, label=f"Session: {data2['session_id'][:8]}...")
    axes[1].set_title(f"Frequency Domain - {data2['subject_id']} - Session 2")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot both frequency domains for direct comparison
    axes[2].semilogy(freqs1, fft_vals1, label=f"Session 1", alpha=0.7)
    axes[2].semilogy(freqs2, fft_vals2, label=f"Session 2", alpha=0.7)
    axes[2].set_title("Direct Comparison of Frequency Domains")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("frequency_domain_comparison.png")
    plt.close()
    
    print("Frequency domain comparison plot saved to frequency_domain_comparison.png")

def compare_spectrograms(data1, data2):
    """Compare the spectrograms between two NWB files."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Calculate sampling frequency
    fs1 = 1.0 / (data1["timestamps"][1] - data1["timestamps"][0])
    fs2 = 1.0 / (data2["timestamps"][1] - data2["timestamps"][0])
    
    # Plot the first spectrogram
    axes[0].set_title(f"Spectrogram - {data1['subject_id']} - Session 1")
    axes[0].specgram(data1["data"], Fs=fs1, cmap='viridis')
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Frequency (Hz)")
    
    # Plot the second spectrogram
    axes[1].set_title(f"Spectrogram - {data2['subject_id']} - Session 2")
    axes[1].specgram(data2["data"], Fs=fs2, cmap='viridis')
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")
    
    plt.colorbar(axes[0].images[0], ax=axes[0], label="Power/Frequency (dB/Hz)")
    plt.colorbar(axes[1].images[0], ax=axes[1], label="Power/Frequency (dB/Hz)")
    
    plt.tight_layout()
    plt.savefig("spectrogram_comparison.png")
    plt.close()
    
    print("Spectrogram comparison plot saved to spectrogram_comparison.png")

def calculate_statistics(data1, data2):
    """Calculate statistics to quantitatively compare the datasets."""
    stats1 = {
        "min": np.min(data1["data"]),
        "max": np.max(data1["data"]),
        "mean": np.mean(data1["data"]),
        "std": np.std(data1["data"]),
        "median": np.median(data1["data"]),
        "duration": data1["timestamps"][-1] - data1["timestamps"][0]
    }
    
    stats2 = {
        "min": np.min(data2["data"]),
        "max": np.max(data2["data"]),
        "mean": np.mean(data2["data"]),
        "std": np.std(data2["data"]),
        "median": np.median(data2["data"]),
        "duration": data2["timestamps"][-1] - data2["timestamps"][0]
    }
    
    print("\n=== Statistical Comparison ===")
    print(f"Session 1 ({data1['session_id'][:8]}...):")
    for key, value in stats1.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nSession 2 ({data2['session_id'][:8]}...):")
    for key, value in stats2.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nRelative differences (Session2/Session1 - 1):")
    for key in stats1.keys():
        if stats1[key] != 0:  # Avoid division by zero
            diff = (stats2[key] / stats1[key] - 1) * 100
            print(f"  {key}: {diff:.2f}%")

def main():
    try:
        # Load the NWB files
        nwb1, h5_file1 = load_nwb(URL1)
        nwb2, h5_file2 = load_nwb(URL2)
        
        # Extract the data
        data1 = extract_data(nwb1)
        data2 = extract_data(nwb2)
        
        # Compare the data
        compare_time_series(data1, data2)
        compare_frequency_domain(data1, data2)
        compare_spectrograms(data1, data2)
        calculate_statistics(data1, data2)
        
        # Close the files
        h5_file1.close()
        h5_file2.close()
        
        print("\nComparison complete!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()