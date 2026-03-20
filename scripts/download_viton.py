"""
VITON Dataset Downloader
Downloads and extracts VITON dataset from official sources.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile

# VITON dataset sources
VITON_SOURCES = {
    "viton-hd": {
        "name": "VITON-HD",
        "urls": {
            "train_cloth": "https://github.com/shadow2496/VITON-HD/releases/download/v1.0/train_cloth.zip",
            "train_image": "https://github.com/shadow2496/VITON-HD/releases/download/v1.0/train_image.zip",
            "test_cloth": "https://github.com/shadow2496/VITON-HD/releases/download/v1.0/test_cloth.zip",
            "test_image": "https://github.com/shadow2496/VITON-HD/releases/download/v1.0/test_image.zip",
        },
        "note": "High-resolution (1024×768) - ~15GB total"
    },
    "viton": {
        "name": "VITON Original",
        "urls": {
            "dataset": "https://drive.google.com/uc?id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo",
        },
        "note": "Original resolution (256×192) - ~1.5GB"
    },
}

def download_file(url: str, dest_path: Path, chunk_size=8192):
    """Download file with progress bar."""
    print(f"\nDownloading: {url}")
    print(f"Destination: {dest_path}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)

    print(f"✓ Downloaded: {dest_path.name}")
    return dest_path

def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract zip or tar.gz archive."""
    print(f"\nExtracting: {archive_path.name}")

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif archive_path.name.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(dest_dir)

    print(f"✓ Extracted to: {dest_dir}")

    # Clean up archive
    archive_path.unlink()
    print(f"✓ Cleaned up: {archive_path.name}")

def download_viton_dataset(dataset_type="viton-hd", dataset_root="dataset"):
    """Download and setup VITON dataset."""
    dataset_root = Path(dataset_root)
    dataset_info = VITON_SOURCES.get(dataset_type)

    if not dataset_info:
        print(f"Error: Unknown dataset type '{dataset_type}'")
        print(f"Available: {list(VITON_SOURCES.keys())}")
        return False

    print(f"\n{'='*60}")
    print(f"Downloading {dataset_info['name']}")
    print(f"Note: {dataset_info['note']}")
    print(f"{'='*60}")

    # Download all components
    for key, url in dataset_info['urls'].items():
        filename = url.split('/')[-1]
        dest_path = dataset_root / "downloads" / filename

        if dest_path.exists():
            print(f"⏭ Skipping existing: {filename}")
            continue

        try:
            download_file(url, dest_path)

            # Extract to appropriate location
            if 'cloth' in key:
                extract_archive(dest_path, dataset_root / "train")
            elif 'image' in key:
                extract_archive(dest_path, dataset_root / "train")
            else:
                extract_archive(dest_path, dataset_root)

        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False

    print(f"\n{'='*60}")
    print(f"✓ VITON dataset setup complete!")
    print(f"Location: {dataset_root.absolute()}")
    print(f"{'='*60}")
    return True

def check_dataset_structure(dataset_root="dataset"):
    """Verify dataset structure and print stats."""
    dataset_root = Path(dataset_root)

    print(f"\n{'='*60}")
    print(f"Dataset Structure Check")
    print(f"{'='*60}")

    required_dirs = [
        "train/cloth",
        "train/cloth-mask",
        "train/image",
        "train/image-parse-v3",
    ]

    for dir_path in required_dirs:
        full_path = dataset_root / dir_path
        if full_path.exists():
            count = len(list(full_path.glob("*.jpg")))
            print(f"✓ {dir_path:30s} {count:5d} images")
        else:
            print(f"✗ {dir_path:30s} MISSING")

    # Check for pairs file
    pairs_file = dataset_root / "train_pairs.txt"
    if pairs_file.exists():
        with open(pairs_file) as f:
            pairs_count = len(f.readlines())
        print(f"✓ train_pairs.txt                {pairs_count:5d} pairs")
    else:
        print(f"✗ train_pairs.txt                 MISSING")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VITON dataset")
    parser.add_argument(
        "--type",
        choices=["viton", "viton-hd"],
        default="viton-hd",
        help="Dataset version to download"
    )
    parser.add_argument(
        "--root",
        default="dataset",
        help="Dataset root directory"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check dataset structure, don't download"
    )

    args = parser.parse_args()

    if args.check:
        check_dataset_structure(args.root)
    else:
        success = download_viton_dataset(args.type, args.root)
        if success:
            check_dataset_structure(args.root)
        else:
            print("\n✗ Dataset download failed")
            sys.exit(1)
