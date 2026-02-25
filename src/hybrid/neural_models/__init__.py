"""
Neural Model Integration Module
Handles loading and management of HR-VITON neural models (GMM and TOM)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NeuralModelManager:
    """
    Manages HR-VITON neural models (Garment Matching Module and Try-On Module)
    Handles model loading, validation, and inference setup
    """
    
    # Expected model information
    MODELS = {
        'gmm': {
            'name': 'Garment Matching Module (GMM)',
            'filename': 'hr_viton_gmm.pth',
            'expected_size_mb': 120,
            'url': 'https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ/view',
            'description': 'Generates thin-plate spline warping parameters'
        },
        'tom': {
            'name': 'Try-On Module (TOM)',
            'filename': 'hr_viton_tom.pth',
            'expected_size_mb': 380,
            'url': 'https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy/view',
            'description': 'Renders warped garment onto person'
        }
    }
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.loaded_models = {}
        self.model_states = {}
        
        self._detect_models()
    
    def _detect_models(self):
        """Scan for available models and verify"""
        for model_key, model_info in self.MODELS.items():
            model_path = self.models_dir / model_info['filename']
            
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                self.model_states[model_key] = {
                    'present': True,
                    'path': str(model_path),
                    'size_mb': size_mb,
                    'verified': size_mb > model_info['expected_size_mb'] * 0.8  # Allow 20% variance
                }
                logger.info(f"[MODELS] Found {model_key}: {size_mb:.1f}MB")
            else:
                self.model_states[model_key] = {
                    'present': False,
                    'path': str(model_path),
                    'size_mb': 0,
                    'verified': False
                }
                logger.warning(f"[MODELS] Missing {model_key}: {model_path}")
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all models"""
        return self.model_states
    
    def is_model_available(self, model_key: str) -> bool:
        """Check if specific model is available and verified"""
        state = self.model_states.get(model_key, {})
        return state.get('verified', False)
    
    def are_all_models_available(self) -> bool:
        """Check if all models are available"""
        return all(state.get('verified', False) for state in self.model_states.values())
    
    def load_gmm(self, device: str = 'cpu') -> Optional[Any]:
        """
        Load GMM model for warping
        
        Args:
            device: 'cpu' or 'cuda'
            
        Returns:
            Loaded model or None if not available
        """
        if not self.is_model_available('gmm'):
            logger.warning("[GMM] Model not available - using geometric fallback")
            return None
        
        try:
            import torch
            
            gmm_path = self.model_states['gmm']['path']
            logger.info(f"[GMM] Loading from {gmm_path}...")
            
            # Load model checkpoint
            checkpoint = torch.load(gmm_path, map_location=device)
            
            # Initialize model architecture
            from src.hybrid.neural_models.gmm_model import GMM
            gmm_model = GMM()
            gmm_model.load_state_dict(checkpoint)
            gmm_model.to(device)
            gmm_model.eval()
            
            self.loaded_models['gmm'] = gmm_model
            logger.info(f"[GMM] Model loaded successfully on {device}")
            
            return gmm_model
        
        except ImportError as e:
            logger.error(f"[GMM] PyTorch not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[GMM] Failed to load model: {e}")
            return None
    
    def load_tom(self, device: str = 'cpu') -> Optional[Any]:
        """
        Load TOM model for rendering
        
        Args:
            device: 'cpu' or 'cuda'
            
        Returns:
            Loaded model or None if not available
        """
        if not self.is_model_available('tom'):
            logger.warning("[TOM] Model not available - using simple compositing")
            return None
        
        try:
            import torch
            
            tom_path = self.model_states['tom']['path']
            logger.info(f"[TOM] Loading from {tom_path}...")
            
            # Load model checkpoint
            checkpoint = torch.load(tom_path, map_location=device)
            
            # Initialize model architecture
            from src.hybrid.neural_models.tom_model import TOM
            tom_model = TOM()
            tom_model.load_state_dict(checkpoint)
            tom_model.to(device)
            tom_model.eval()
            
            self.loaded_models['tom'] = tom_model
            logger.info(f"[TOM] Model loaded successfully on {device}")
            
            return tom_model
        
        except ImportError as e:
            logger.error(f"[TOM] PyTorch not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[TOM] Failed to load model: {e}")
            return None
    
    def load_all_models(self, device: str = 'cpu') -> Tuple[Optional[Any], Optional[Any]]:
        """Load all available models"""
        gmm = self.load_gmm(device)
        tom = self.load_tom(device)
        
        return gmm, tom
    
    def get_download_instructions(self) -> str:
        """Get instructions for downloading models"""
        instructions = """
NEURAL MODEL DOWNLOAD INSTRUCTIONS
{'='*60}

The system requires two neural models for optimal quality:

1. GARMENT MATCHING MODULE (GMM)
   Name: hr_viton_gmm.pth
   Size: ~120 MB
   Download from: https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ
   Place in: models/hr_viton_gmm.pth

2. TRY-ON MODULE (TOM)
   Name: hr_viton_tom.pth
   Size: ~380 MB
   Download from: https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy
   Place in: models/hr_viton_tom.pth

STEPS TO DOWNLOAD:
1. Click the Google Drive links above
2. Click "Download" button
3. Save files to models/ directory
4. Verify with: python -c "from src.hybrid.neural_models import NeuralModelManager; m = NeuralModelManager(); print(m.get_model_status())"

CURRENT STATUS:
{'='*60}
"""
        for model_key, state in self.model_states.items():
            status = "[READY]" if state['verified'] else "[MISSING]"
            model_info = self.MODELS[model_key]
            instructions += f"\n{status} {model_info['name']}"
            instructions += f"\n      Size: {state['size_mb']:.1f}MB"
            instructions += f"\n      Path: {state['path']}\n"
        
        return instructions
    
    def validate_models(self) -> Dict[str, bool]:
        """Validate loaded models"""
        validation = {}
        
        # Check GMM
        if 'gmm' in self.loaded_models:
            try:
                import torch
                gmm = self.loaded_models['gmm']
                # Test inference
                dummy_input = torch.randn(1, 3, 256, 192)  # Typical input
                with torch.no_grad():
                    _ = gmm(dummy_input)
                validation['gmm'] = True
            except Exception as e:
                logger.error(f"[GMM] Validation failed: {e}")
                validation['gmm'] = False
        
        # Check TOM
        if 'tom' in self.loaded_models:
            try:
                import torch
                tom = self.loaded_models['tom']
                # Test inference
                dummy_input = torch.randn(1, 3, 512, 384)  # Typical input
                with torch.no_grad():
                    _ = tom(dummy_input)
                validation['tom'] = True
            except Exception as e:
                logger.error(f"[TOM] Validation failed: {e}")
                validation['tom'] = False
        
        return validation


class PerformanceComparator:
    """
    Compare performance: Geometric vs Neural Models
    """
    
    @staticmethod
    def estimate_performance(use_neural: bool = False, use_gpu: bool = False) -> Dict[str, float]:
        """Estimate performance metrics for different configurations"""
        
        # Baseline: Geometric fallback (measured from stress tests)
        base_metrics = {
            'segmentation_ms': 40.9,      # CPU
            'warping_ms': 3.0,             # Geometric
            'rendering_ms': 16.8,
            'temporal_ms': 0.7,
            'total_ms': 62.1,
            'fps': 16.1
        }
        
        if use_gpu:
            # GPU acceleration (estimated)
            base_metrics['segmentation_ms'] *= 0.35  # 65% speedup from GPU
            base_metrics['rendering_ms'] *= 0.5      # 50% speedup
        
        if use_neural and not use_gpu:
            # Neural models on CPU (estimated slower)
            base_metrics['warping_ms'] = 25.0        # Neural TPS warping
            base_metrics['rendering_ms'] = 40.0      # Neural blending
        
        elif use_neural and use_gpu:
            # Neural models on GPU (optimal)
            base_metrics['warping_ms'] = 8.0         # Neural on GPU
            base_metrics['rendering_ms'] = 15.0      # Neural on GPU
        
        # Recalculate totals
        base_metrics['total_ms'] = (
            base_metrics['segmentation_ms'] + 
            base_metrics['warping_ms'] + 
            base_metrics['rendering_ms'] + 
            base_metrics['temporal_ms']
        )
        base_metrics['fps'] = 1000.0 / base_metrics['total_ms']
        
        return base_metrics


def print_neural_models_status():
    """Print status of neural models"""
    manager = NeuralModelManager()
    
    print("\n" + "="*60)
    print("NEURAL MODEL INTEGRATION STATUS")
    print("="*60 + "\n")
    
    status = manager.get_model_status()
    
    for model_key, state in status.items():
        model_info = manager.MODELS[model_key]
        status_str = "[READY]" if state['verified'] else "[MISSING]"
        
        print(f"{status_str} {model_info['name']}")
        print(f"    File: {model_info['filename']}")
        print(f"    Size: {state['size_mb']:.1f}MB")
        print(f"    Path: {state['path']}")
        print()
    
    if manager.are_all_models_available():
        print("✅ All neural models ready for integration")
    else:
        print(manager.get_download_instructions())
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60 + "\n")
    
    configs = [
        ("CPU + Geometric", False, False),
        ("CPU + Neural", True, False),
        ("GPU + Geometric", False, True),
        ("GPU + Neural", True, True)
    ]
    
    for config_name, use_neural, use_gpu in configs:
        metrics = PerformanceComparator.estimate_performance(use_neural, use_gpu)
        print(f"{config_name:20} → {metrics['fps']:5.1f} FPS ({metrics['total_ms']:6.1f}ms)")
    
    print()


if __name__ == "__main__":
    print_neural_models_status()
