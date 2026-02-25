"""
VITON Dataset Integration Module

Handles loading and mapping of High-Resolution VITON-Zalando dataset
for realistic garment visualization in AR sizing pipeline.

Dataset: https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VITONGarmentLoader:
    """
    Loads high-resolution garment images from VITON dataset.
    Replaces placeholder assets with real product photography.
    """
    
    def __init__(self, viton_root: str, config_path: str = "viton_config.json"):
        """
        Initialize VITON garment loader
        
        Args:
            viton_root: Path to VITON dataset root directory
            config_path: Path to SKU mapping configuration
        """
        self.viton_root = Path(viton_root)
        self.config_path = Path(config_path)
        
        # VITON dataset structure (supports both train/ and root-level folders)
        if (self.viton_root / "train").exists():
            # Dataset has train/test structure
            self.product_dir = self.viton_root / "train" / "cloth"
            self.person_dir = self.viton_root / "train" / "image"
            self.mask_dir = self.viton_root / "train" / "cloth-mask"
            self.edge_dir = self.viton_root / "train" / "image-parse-v3"
            self.pose_dir = self.viton_root / "train" / "openpose_json"
        else:
            # Flat structure
            self.product_dir = self.viton_root / "train_clothing"
            self.person_dir = self.viton_root / "train_img"
            self.mask_dir = self.viton_root / "train_mask"
            self.edge_dir = self.viton_root / "train_edge"
            self.pose_dir = self.viton_root / "train_openpose"
        
        # Cache for loaded images
        self.image_cache: Dict[str, np.ndarray] = {}
        self.mask_cache: Dict[str, np.ndarray] = {}
        
        # Load configuration
        self.sku_mapping = self._load_sku_mapping()
        self.available_products = self._scan_available_products()
        
        logger.info(f"✓ VITON dataset loaded from: {viton_root}")
        logger.info(f"✓ Found {len(self.available_products)} product images")
    
    def _load_sku_mapping(self) -> Dict[str, str]:
        """Load SKU to VITON ID mapping from config"""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return self._get_default_mapping()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get('sku_to_viton_mapping', {})
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_mapping()
    
    def _get_default_mapping(self) -> Dict[str, str]:
        """Default SKU mapping for testing"""
        return {
            "TSH-001": "00001_00",
            "TSH-002": "00002_00",
            "SHT-001": "00003_00",
            "SHT-002": "00004_00",
            "JKT-001": "00005_00",
            "JKT-002": "00006_00",
            "HOD-001": "00007_00",
            "HOD-002": "00008_00",
            "SWT-001": "00009_00",
            "SWT-002": "00010_00",
        }
    
    def _scan_available_products(self) -> List[str]:
        """Scan VITON dataset for available product images"""
        if not self.product_dir.exists():
            logger.warning(f"Product directory not found: {self.product_dir}")
            return []
        
        products = []
        for img_path in self.product_dir.glob("*.jpg"):
            products.append(img_path.stem)
        
        return sorted(products)
    
    def get_garment_image(self, sku: str, size: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load garment image for given SKU
        
        Args:
            sku: Product SKU code (e.g., "TSH-001")
            size: Size label (currently unused, for future size-specific loading)
        
        Returns:
            BGR image array or None if not found
        """
        cache_key = f"{sku}_{size or 'default'}"
        
        # Check cache
        if cache_key in self.image_cache:
            return self.image_cache[cache_key].copy()
        
        # Map SKU to VITON ID
        viton_id = self.sku_mapping.get(sku)
        if not viton_id:
            logger.warning(f"No VITON mapping for SKU: {sku}")
            return None
        
        # Load image
        img_path = self.product_dir / f"{viton_id}.jpg"
        if not img_path.exists():
            logger.warning(f"VITON image not found: {img_path}")
            return None
        
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return None
        
        # Cache and return
        self.image_cache[cache_key] = img
        logger.info(f"✓ Loaded VITON garment: {sku} -> {viton_id}")
        return img.copy()
    
    def get_garment_mask(self, sku: str) -> Optional[np.ndarray]:
        """
        Load segmentation mask for garment
        
        Args:
            sku: Product SKU code
        
        Returns:
            Grayscale mask or None if not found
        """
        if sku in self.mask_cache:
            return self.mask_cache[sku].copy()
        
        viton_id = self.sku_mapping.get(sku)
        if not viton_id:
            return None
        
        mask_path = self.mask_dir / f"{viton_id}.jpg"
        if not mask_path.exists():
            return None
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            self.mask_cache[sku] = mask
        
        return mask.copy() if mask is not None else None
    
    def get_person_image(self, person_id: str) -> Optional[np.ndarray]:
        """
        Load person image from VITON dataset (for testing/reference)
        
        Args:
            person_id: VITON person ID
        
        Returns:
            BGR image array or None
        """
        img_path = self.person_dir / f"{person_id}.jpg"
        if not img_path.exists():
            return None
        
        return cv2.imread(str(img_path))
    
    def list_available_skus(self) -> List[str]:
        """Get list of SKUs that have VITON mappings"""
        return list(self.sku_mapping.keys())
    
    def list_available_viton_products(self) -> List[str]:
        """Get list of available VITON product IDs"""
        return self.available_products
    
    @property
    def product_count(self) -> int:
        """Get count of available VITON products"""
        return len(self.available_products)
    
    def get_garment_info(self, sku: str) -> Dict:
        """
        Get metadata about garment
        
        Args:
            sku: Product SKU
        
        Returns:
            Dictionary with garment information
        """
        viton_id = self.sku_mapping.get(sku)
        img = self.get_garment_image(sku)
        
        info = {
            'sku': sku,
            'viton_id': viton_id,
            'available': img is not None,
            'has_mask': self.get_garment_mask(sku) is not None
        }
        
        if img is not None:
            info['dimensions'] = {
                'height': img.shape[0],
                'width': img.shape[1],
                'channels': img.shape[2] if len(img.shape) > 2 else 1
            }
        
        return info
    
    def create_sku_mapping(self, output_path: str = "viton_config.json"):
        """
        Create a mapping configuration file
        
        Args:
            output_path: Where to save the config
        """
        # Get first N available products
        available = self.list_available_viton_products()[:50]
        
        # Create sample mapping
        config = {
            "viton_dataset_root": str(self.viton_root),
            "sku_to_viton_mapping": {},
            "category_mapping": {
                "TSH": "t-shirt",
                "SHT": "shirt",
                "JKT": "jacket",
                "HOD": "hoodie",
                "SWT": "sweater",
                "PLO": "polo",
                "SKU": "generic"
            },
            "available_viton_ids": available[:20]
        }
        
        # Auto-generate some mappings
        sku_prefixes = ["TSH", "SHT", "JKT", "HOD", "SWT"]
        for i, viton_id in enumerate(available[:20]):
            prefix = sku_prefixes[i % len(sku_prefixes)]
            sku = f"{prefix}-{str(i+1).zfill(3)}"
            config["sku_to_viton_mapping"][sku] = viton_id
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Created mapping config: {output_path}")
        return config


class VITONDatasetManager:
    """
    Manages VITON dataset download, extraction, and validation
    """
    
    def __init__(self, dataset_root: str = "viton_data"):
        self.dataset_root = Path(dataset_root)
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        Check if VITON dataset is properly downloaded and structured
        
        Returns:
            (is_valid, missing_components)
        """
        required_dirs = [
            "train_img",
            "train_clothing",
            "train_mask",
            "train_edge",
            "train_openpose"
        ]
        
        missing = []
        for dir_name in required_dirs:
            dir_path = self.dataset_root / dir_name
            if not dir_path.exists():
                missing.append(dir_name)
        
        is_valid = len(missing) == 0
        return is_valid, missing
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            'root': str(self.dataset_root),
            'exists': self.dataset_root.exists(),
            'directories': {}
        }
        
        if not self.dataset_root.exists():
            return stats
        
        dirs_to_check = [
            "train_img",
            "train_clothing",
            "train_mask",
            "train_edge",
            "train_openpose"
        ]
        
        for dir_name in dirs_to_check:
            dir_path = self.dataset_root / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                stats['directories'][dir_name] = {
                    'exists': True,
                    'file_count': file_count
                }
            else:
                stats['directories'][dir_name] = {'exists': False}
        
        return stats


def demo_viton_loader(viton_root: str, sku: str = "TSH-001"):
    """
    Demo function to test VITON loader
    
    Args:
        viton_root: Path to VITON dataset
        sku: SKU to load
    """
    print("=" * 60)
    print("VITON Garment Loader Demo")
    print("=" * 60)
    
    # Initialize loader
    loader = VITONGarmentLoader(viton_root)
    
    # Show available SKUs
    print(f"\nAvailable SKUs: {loader.list_available_skus()[:10]}")
    print(f"Available VITON products: {len(loader.list_available_viton_products())}")
    
    # Load specific garment
    print(f"\nLoading garment: {sku}")
    img = loader.get_garment_image(sku)
    
    if img is not None:
        print(f"✓ Loaded image: {img.shape}")
        
        # Get info
        info = loader.get_garment_info(sku)
        print(f"Info: {json.dumps(info, indent=2)}")
        
        # Display
        cv2.imshow('VITON Garment', img)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"✗ Failed to load garment: {sku}")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python viton_integration.py <viton_dataset_path> [sku]")
        print("Example: python viton_integration.py ./viton_data TSH-001")
        sys.exit(1)
    
    viton_path = sys.argv[1]
    sku = sys.argv[2] if len(sys.argv) > 2 else "TSH-001"
    
    demo_viton_loader(viton_path, sku)
