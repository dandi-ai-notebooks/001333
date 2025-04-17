"""
This script examines the full list of assets to understand what types of subjects
or data are available in the Dandiset 001333.
"""

import json
import subprocess
import re
from collections import Counter

# Get all assets
result = subprocess.run(['python', '../tools_cli.py', 'dandiset-assets', '001333'], 
                        capture_output=True, text=True)
data = json.loads(result.stdout)

# Extract subject categories
subject_patterns = set()
full_paths = []

for asset in data['results']['results']:
    path = asset['path']
    full_paths.append(path)
    
    # Extract subject pattern (e.g., sub-healthy-simulated-beta)
    match = re.match(r'(sub-[^/]+)', path)
    if match:
        subject_patterns.add(match.group(1))

# Analyze and print results
print(f"Total assets: {data['results']['count']}")
print(f"\nSubject categories found ({len(subject_patterns)}):")
for pattern in sorted(subject_patterns):
    count = sum(1 for path in full_paths if path.startswith(pattern))
    print(f"- {pattern}: {count} assets")

# Sample paths
print("\nSample paths for each subject category:")
for pattern in sorted(subject_patterns):
    for path in full_paths:
        if path.startswith(pattern):
            print(f"- {path}")
            break

# Look for any patterns in file naming
print("\nAnalyzing file naming patterns:")
extensions = Counter([path.split('.')[-1] for path in full_paths])
print(f"File extensions: {dict(extensions)}")

session_patterns = set()
for path in full_paths:
    match = re.search(r'ses-([^_]+)', path)
    if match:
        session_patterns.add(match.group(1))
print(f"Session patterns: {sorted(session_patterns) if len(session_patterns) < 20 else f'{len(session_patterns)} unique session IDs'}")

# Analyze data types indicated in filenames
datatypes = set()
for path in full_paths:
    parts = path.split('_')
    if len(parts) > 1:
        last_part = parts[-1].split('.')[0]  # Remove file extension
        datatypes.add(last_part)
print(f"Data types: {sorted(datatypes)}")