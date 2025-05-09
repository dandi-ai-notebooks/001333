# This script plots the Beta_Band_Voltage data from the NWB file.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r') # Add 'r' mode for read-only
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Add mode='r'
nwb = io.read()

# Get Beta_Band_Voltage data and timestamps
beta_band_voltage_series = nwb.processing["ecephys"]["LFP"].electrical_series["Beta_Band_Voltage"]
data = beta_band_voltage_series.data[:]
timestamps = beta_band_voltage_series.timestamps[:]

# Create plot
sns.set_theme()
plt.figure(figsize=(12, 6))
plt.plot(timestamps, data)
plt.xlabel(f"Time ({beta_band_voltage_series.timestamps_unit})")
plt.ylabel(f"Voltage ({beta_band_voltage_series.unit})")
plt.title("Beta Band Voltage")
plt.grid(True)
plt.savefig("explore/beta_band_voltage_plot.png")
plt.close()

print("Plot saved to explore/beta_band_voltage_plot.png")

io.close() # Close the NWBHDF5IO object
# It's also good practice to close the h5py file if remfile doesn't handle it,
# though remfile often manages the underlying file object.
# remote_file.close() # remfile.File might not have a close method, or it might be automatic.