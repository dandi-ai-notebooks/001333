"""
This script analyzes the asset information from the Parkinson's Electrophysiological Signal Dataset (PESD)
to identify the different subject types and their characteristics.
"""

import json
import os
import re
from collections import Counter, defaultdict

# Load the asset information
with open('explore/assets/dandiset_assets.json', 'r') as f:
    assets_data = json.load(f)

# Extract subject information
subjects = []
subject_files = defaultdict(list)

print(f"Total number of assets: {assets_data['results']['count']}")
print()

# Extract subject names from file paths
for asset in assets_data['results']['results']:
    if 'path' in asset:
        path = asset['path']
        match = re.match(r'(sub-[^/]+)/', path)
        if match:
            subject = match.group(1)
            subjects.append(subject)
            subject_files[subject].append(path)

# Count the occurrences of each subject
subject_counts = Counter(subjects)

print("Subject Types:")
for subject, count in subject_counts.items():
    print(f"{subject}: {count} files")
print()

# Analyze the first few files for each subject
print("Sample Files:")
for subject, files in subject_files.items():
    print(f"\n{subject}:")
    for i, file in enumerate(files[:5]):   # Print only first 5 files
        print(f"  {i+1}. {file}")
    if len(files) > 5:
        print(f"  ... ({len(files)-5} more files)")
print()

# Extract session information
sessions = []
for subject, files in subject_files.items():
    for file in files:
        match = re.search(r'ses-(\d+)_', file)
        if match:
            session = match.group(1)
            sessions.append(session)

# Count the occurrences of each session
session_counts = Counter(sessions)

print("Number of Unique Sessions:", len(session_counts))
print()

# Create dictionary of file sizes by subject
file_sizes = defaultdict(list)
for asset in assets_data['results']['results']:
    if 'path' in asset and 'size' in asset:
        path = asset['path']
        match = re.match(r'(sub-[^/]+)/', path)
        if match:
            subject = match.group(1)
            file_sizes[subject].append(asset['size'])

# Calculate average file size by subject
print("Average File Size by Subject:")
for subject, sizes in file_sizes.items():
    avg_size = sum(sizes) / len(sizes)
    print(f"{subject}: {avg_size:.2f} bytes")