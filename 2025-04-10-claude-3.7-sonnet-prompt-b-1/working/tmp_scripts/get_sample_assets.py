# This script gets sample assets for each subject type in the Dandiset

import requests

def main():
    # Get assets from the Dandiset with pagination
    url = "https://api.dandiarchive.org/api/dandisets/001333/versions/draft/assets/?page_size=100"
    subject_samples = {}
    
    while url and len(subject_samples) < 5:  # We have 5 subject types to find
        print(f"Fetching from {url}")
        response = requests.get(url)
        data = response.json()
        
        for asset in data['results']:
            path = asset['path']
            parts = path.split('/')
            if len(parts) > 0:
                subject_folder = parts[0]
                if subject_folder.startswith('sub-') and subject_folder not in subject_samples:
                    subject_samples[subject_folder] = {
                        'path': path,
                        'asset_id': asset['asset_id']
                    }
                    print(f"Found sample for {subject_folder}")
        
        url = data.get('next')
        if not url:
            print("No more pages to fetch")
    
    print("\nSample assets for each subject type:")
    for subject, sample in subject_samples.items():
        print(f"{subject}:")
        print(f"  Path: {sample['path']}")
        print(f"  Asset ID: {sample['asset_id']}")
        print(f"  URL: https://api.dandiarchive.org/api/assets/{sample['asset_id']}/download/")
        print()

if __name__ == "__main__":
    main()