# This script analyzes the asset paths in the Dandiset to understand data organization
# It will identify different subject types and session patterns

import requests
import json
from collections import Counter

def main():
    # Get all assets from the Dandiset
    url = "https://api.dandiarchive.org/api/dandisets/001333/versions/draft/assets/?page_size=1000"
    all_assets = []
    
    while url:
        response = requests.get(url)
        data = response.json()
        all_assets.extend(data['results'])
        url = data.get('next')
    
    # Analyze subject types
    subject_types = set()
    for asset in all_assets:
        path = asset['path']
        parts = path.split('/')
        if len(parts) > 0:
            subject_folder = parts[0]
            if subject_folder.startswith('sub-'):
                subject_types.add(subject_folder)
    
    print(f"Found {len(all_assets)} assets in the Dandiset")
    print(f"Subject types: {sorted(list(subject_types))}")
    
    # Count assets per subject type
    subject_counts = Counter()
    for asset in all_assets:
        path = asset['path']
        parts = path.split('/')
        if len(parts) > 0 and parts[0].startswith('sub-'):
            subject_counts[parts[0]] += 1
    
    print("\nAssets per subject type:")
    for subject, count in subject_counts.most_common():
        print(f"{subject}: {count}")
    
    # Analyze session patterns
    session_patterns = set()
    for asset in all_assets:
        path = asset['path']
        filename = path.split('/')[-1]
        if '_ses-' in filename:
            session_part = filename.split('_ses-')[1].split('_')[0]
            session_patterns.add(session_part)
    
    print(f"\nSession patterns: {len(session_patterns)} unique patterns")
    print(f"Example sessions: {list(session_patterns)[:5]}...")

if __name__ == "__main__":
    main()