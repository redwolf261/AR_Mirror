"""
VITON Dataset Preparation Script

Downloads and prepares the High-Resolution VITON-Zalando dataset
for use with AR sizing pipeline.

Usage:
    python prepare_viton_dataset.py --download
    python prepare_viton_dataset.py --validate
    python prepare_viton_dataset.py --create-mapping
"""

import argparse
import json
import sys
from pathlib import Path
from src.viton.viton_integration import VITONDatasetManager, VITONGarmentLoader


def download_dataset(output_dir: str = "viton_data"):
    """
    Download VITON dataset from Kaggle
    Requires: kaggle CLI authenticated
    """
    print("=" * 60)
    print("VITON Dataset Download")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_path.absolute()}")
    print("\nPrerequisites:")
    print("1. Install Kaggle CLI: pip install kaggle")
    print("2. Setup authentication: kaggle.com/account -> Create API Token")
    print("3. Place kaggle.json in ~/.kaggle/")
    print()
    
    try:
        import kaggle
        print("✓ Kaggle CLI found")
    except ImportError:
        print("✗ Kaggle CLI not installed")
        print("\nInstall with: pip install kaggle")
        return False
    
    print("\nDownloading dataset (this may take 10-30 minutes)...")
    print("Dataset: marquis03/high-resolution-viton-zalando-dataset")
    print()
    
    try:
        # Download
        kaggle.api.dataset_download_files(
            'marquis03/high-resolution-viton-zalando-dataset',
            path=str(output_path),
            unzip=True
        )
        
        print(f"\n✓ Dataset downloaded to: {output_path}")
        print("\nValidating download...")
        
        manager = VITONDatasetManager(output_dir)
        is_valid, missing = manager.validate_dataset()
        
        if is_valid:
            stats = manager.get_dataset_stats()
            print("\n✓ Dataset validation passed!")
            print("\nDataset statistics:")
            for dir_name, info in stats['directories'].items():
                if info['exists']:
                    print(f"  {dir_name}: {info['file_count']} files")
        else:
            print(f"\n✗ Dataset incomplete. Missing: {missing}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check Kaggle authentication: kaggle competitions list")
        print("2. Ensure you've accepted dataset terms on Kaggle website")
        print("3. Check internet connection")
        return False


def validate_dataset(dataset_dir: str = "viton_data"):
    """Validate existing VITON dataset"""
    print("=" * 60)
    print("VITON Dataset Validation")
    print("=" * 60)
    
    manager = VITONDatasetManager(dataset_dir)
    
    print(f"\nChecking: {Path(dataset_dir).absolute()}")
    
    is_valid, missing = manager.validate_dataset()
    
    if is_valid:
        print("\n✓ Dataset structure is valid!")
        
        stats = manager.get_dataset_stats()
        print("\nDataset contents:")
        for dir_name, info in stats['directories'].items():
            if info['exists']:
                status = "✓"
                count = f"{info['file_count']} files"
            else:
                status = "✗"
                count = "missing"
            print(f"  {status} {dir_name}: {count}")
        
        return True
    else:
        print(f"\n✗ Dataset incomplete!")
        print(f"\nMissing directories: {', '.join(missing)}")
        print("\nPlease run: python prepare_viton_dataset.py --download")
        return False


def create_mapping_config(dataset_dir: str = "viton_data", 
                         output_file: str = "viton_config.json"):
    """Create SKU to VITON ID mapping configuration"""
    print("=" * 60)
    print("Create VITON Mapping Configuration")
    print("=" * 60)
    
    try:
        loader = VITONGarmentLoader(dataset_dir)
        config = loader.create_sku_mapping(output_file)
        
        print(f"\n✓ Created mapping config: {output_file}")
        print(f"\nMapped {len(config['sku_to_viton_mapping'])} SKUs")
        print("\nSample mappings:")
        for sku, viton_id in list(config['sku_to_viton_mapping'].items())[:5]:
            print(f"  {sku} -> {viton_id}")
        
        print(f"\nEdit {output_file} to customize SKU mappings")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to create mapping: {e}")
        return False


def test_integration(dataset_dir: str = "viton_data"):
    """Test VITON integration with sample garment"""
    print("=" * 60)
    print("Test VITON Integration")
    print("=" * 60)
    
    try:
        from src.viton.viton_integration import demo_viton_loader
        demo_viton_loader(dataset_dir)
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VITON dataset for AR sizing pipeline"
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download VITON dataset from Kaggle'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing dataset'
    )
    
    parser.add_argument(
        '--create-mapping',
        action='store_true',
        help='Create SKU to VITON ID mapping config'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test integration with sample garment'
    )
    
    parser.add_argument(
        '--dataset-dir',
        default='viton_data',
        help='VITON dataset directory (default: viton_data)'
    )
    
    parser.add_argument(
        '--output-config',
        default='viton_config.json',
        help='Output config file (default: viton_config.json)'
    )
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not (args.download or args.validate or args.create_mapping or args.test):
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start Guide")
        print("=" * 60)
        print("\n1. Download dataset:")
        print("   python prepare_viton_dataset.py --download")
        print("\n2. Validate download:")
        print("   python prepare_viton_dataset.py --validate")
        print("\n3. Create mapping config:")
        print("   python prepare_viton_dataset.py --create-mapping")
        print("\n4. Test integration:")
        print("   python prepare_viton_dataset.py --test")
        print()
        return
    
    success = True
    
    if args.download:
        success = download_dataset(args.dataset_dir) and success
    
    if args.validate:
        success = validate_dataset(args.dataset_dir) and success
    
    if args.create_mapping:
        success = create_mapping_config(args.dataset_dir, args.output_config) and success
    
    if args.test:
        success = test_integration(args.dataset_dir) and success
    
    if success:
        print("\n✓ All operations completed successfully!")
    else:
        print("\n✗ Some operations failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
