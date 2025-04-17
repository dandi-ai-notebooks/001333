"""
This script examines the full list of assets with pagination to understand what types of subjects
or data are available in the Dandiset 001333.
"""

import json
import subprocess
import re
from collections import Counter

# Function to get assets with pagination
def get_assets_with_pagination(dandiset_id, limit=100, skip=0, max_results=1000):
    all_assets = []
    total_count = None
    
    while total_count is None or skip < total_count:
        if skip >= max_results:
            print(f"Reached maximum result limit of {max_results}")
            break
            
        cmd = ['python', '../tools_cli.py', 'dandiset-assets', dandiset_id, 
               '--limit', str(limit), '--skip', str(skip)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        try:
            data = json.loads(result.stdout)
            
            # Get the total count if we don't have it yet
            if total_count is None:
                total_count = data['results']['count']
                print(f"Total assets reported by API: {total_count}")
            
            # Add the current page results
            page_results = data['results']['results']
            all_assets.extend(page_results)
            print(f"Retrieved {len(page_results)} assets. Total so far: {len(all_assets)}")
            
            # Move to next page
            skip += limit
            
            # If we got fewer results than the limit, we're done
            if len(page_results) < limit:
                break
                
        except json.JSONDecodeError:
            print(f"Error parsing response: {result.stdout[:100]}...")
            break
            
    return all_assets, total_count

# Get assets with pagination
all_assets, total_count = get_assets_with_pagination('001333', limit=100, max_results=500)

# Extract subject categories
subject_patterns = set()
full_paths = []

for asset in all_assets:
    path = asset['path']
    full_paths.append(path)
    
    # Extract subject pattern (e.g., sub-healthy-simulated-beta)
    match = re.match(r'(sub-[^/]+)', path)
    if match:
        subject_patterns.add(match.group(1))

# Count assets per subject type
subject_counts = {}
for pattern in subject_patterns:
    subject_counts[pattern] = sum(1 for path in full_paths if path.startswith(pattern))

# Analyze and print results
print(f"\nRetrieved {len(all_assets)} assets out of {total_count} total")
print(f"\nSubject categories found ({len(subject_patterns)}):")
for pattern, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"- {pattern}: {count} assets")

# Sample paths for each subject category
print("\nSample paths for each subject category:")
for pattern in sorted(subject_patterns):
    found = False
    for path in full_paths:
        if path.startswith(pattern):
            print(f"- {path}")
            found = True
            break
    if not found:
        print(f"- No sample found for {pattern}")

# Look for any patterns in file naming
print("\nAnalyzing file naming patterns:")
extensions = Counter([path.split('.')[-1] for path in full_paths])
print(f"File extensions: {dict(extensions)}")

# Analyze session patterns
sessions_per_subject = {}
for pattern in subject_patterns:
    sessions = set()
    for path in full_paths:
        if path.startswith(pattern):
            match = re.search(r'ses-([^_]+)', path)
            if match:
                sessions.add(match.group(1))
    sessions_per_subject[pattern] = sessions

print("\nSessions per subject type:")
for subject, sessions in sessions_per_subject.items():
    session_count = len(sessions)
    session_display = sorted(sessions) if session_count < 10 else f"{session_count} unique session IDs"
    print(f"- {subject}: {session_display}")

# Analyze data types indicated in filenames
datatypes_per_subject = {}
for pattern in subject_patterns:
    datatypes = set()
    for path in full_paths:
        if path.startswith(pattern):
            parts = path.split('_')
            if len(parts) > 1:
                last_part = parts[-1].split('.')[0]  # Remove file extension
                datatypes.add(last_part)
    datatypes_per_subject[pattern] = datatypes

print("\nData types per subject type:")
for subject, datatypes in datatypes_per_subject.items():
    print(f"- {subject}: {sorted(datatypes)}")