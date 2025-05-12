"""
This script aims to list all assets in Dandiset 001333 to identify different types of files including parkinsonian data.
"""

from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001333", "0.250327.2220")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List all assets in the Dandiset
assets = dandiset.get_assets()
asset_count = 0
subject_types = set()

print("\nAssets by subject type:")
for asset in assets:
    asset_count += 1
    subject_type = asset.path.split('/')[0]
    subject_types.add(subject_type)
    print(f"{asset_count}. {asset.path} (ID: {asset.identifier})")

print(f"\nTotal assets: {asset_count}")
print(f"Subject types: {sorted(list(subject_types))}")