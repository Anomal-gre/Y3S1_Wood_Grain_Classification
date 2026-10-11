"""
Script to download and extract Wood Species Dataset from Zenodo (Record 2545611)
Reference: Barmpoutis et al. (2018), DOI: 10.5281/zenodo.2545611
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

ZENODO_RECORD_ID = "2545611"
DOWNLOAD_URL = "https://zenodo.org/api/records/2545611/files/Wood%20Dataset.zip/content"
FALLBACK_URL = "https://zenodo.org/records/2545611/files/Wood%20Dataset.zip?download=1"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, target_path):
    print(f"Downloading from {url} to {target_path}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=target_path.name) as t:
            urllib.request.urlretrieve(url, filename=target_path, reporthook=t.update_to)
    except Exception as e:
        print(f"Failed with primary URL: {e}. Trying fallback...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=target_path.name) as t:
            urllib.request.urlretrieve(FALLBACK_URL, filename=target_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset"
    zip_path = data_dir / "Wood_Dataset.zip"
    raw_dir = data_dir / "raw"

    if (raw_dir / "Wood Dataset").exists() or any(raw_dir.glob("*/*.jpg")) or any(raw_dir.glob("*.jpg")):
        print(f"Dataset already extracted at {raw_dir}.")
        return

    if not zip_path.exists():
        download_file(DOWNLOAD_URL, zip_path)
    else:
        print(f"Using existing archive: {zip_path}")

    extract_zip(zip_path, raw_dir)
    print(f"Dataset is ready at {raw_dir}")

if __name__ == "__main__":
    main()
