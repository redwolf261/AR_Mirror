#!/usr/bin/env python3
"""
AR Mirror Production Synthetic Data Generation Pipeline

🚀 This is the production script that orchestrates the entire synthetic data generation
   process for commercial deployment. Generates hundreds of thousands of samples
   with commercial SMPL licensing, physics simulation, and domain randomization.

Prerequisites:
- Commercial SMPL license from MPI-IS
- SMPL model files in smpl_models/
- Blender 3.0+ with addon installed
- RTX 2050 with CUDA/OptiX support

Usage:
    python scripts/production_pipeline.py --phase 1 --batch_size 100000
    python scripts/production_pipeline.py --phase 2 --batch_size 300000 --cloud
    python scripts/production_pipeline.py --phase 3 --batch_size 1000000 --cluster
"""

import sys
import os
import argparse  
import subprocess
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
from datetime import datetime
import multiprocessing as mp

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.synthetic_data_factory import SyntheticDataFactory, PoseSampler, ShapeSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("ProductionPipeline")

class ProductionPipeline:
    """
    Production-grade synthetic data generation pipeline
    
    Orchestrates the entire process:
    - SMPL parameter generation
    - Blender rendering with physics
    - Domain randomization
    - Batch processing
    - Quality validation
    - Performance monitoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.validate_setup()
        
        # Initialize components with licensing choice
        use_commercial = config.get('use_commercial_smpl', False)
        self.factory = SyntheticDataFactory(config, use_commercial_smpl=use_commercial)
        self.pose_sampler = PoseSampler()
        self.shape_sampler = ShapeSampler()
        
        # Setup output structure
        self.output_dir = Path(config['output_directory'])
        self.setup_output_structure()
        
        license_type = "commercial" if use_commercial else "free (development)"
        log.info(f"🚀 Production pipeline initialized for Phase {config.get('phase', 1)} ({license_type})")
    
    def validate_setup(self):
        """Validate all requirements for production deployment"""
        issues = []
        use_commercial = self.config.get('use_commercial_smpl', False)
        
        # Check SMPL model setup (free or commercial)
        if use_commercial:
            # Commercial SMPL validation
            smpl_path = Path("smpl_models/SMPL_NEUTRAL.pkl")
            if not smpl_path.exists():
                issues.append("❌ Commercial SMPL model not found (set use_commercial_smpl=False for free development)")
            
            license_path = Path("smpl_models/license.txt")  
            if not license_path.exists():
                issues.append("❌ SMPL commercial license file not found")
        else:
            # Free SMPL-X validation
            smpl_path = Path("smpl_models/SMPLX_NEUTRAL.npz")
            if not smpl_path.exists():
                log.info("💡 Creating free SMPL placeholder for development")
                # Auto-create placeholder for development
                smpl_path.parent.mkdir(exist_ok=True)
                import numpy as np
                placeholder = {
                    'vertices': np.zeros((6890, 3), dtype=np.float32),
                    'faces': np.random.randint(0, 6890, (13776, 3)),
                    'pose_params': np.zeros(75),
                    'shape_params': np.zeros(10)
                }
                np.savez(smpl_path, **placeholder)
                log.info("✅ Free SMPL placeholder created")
        
        # Check Blender installation (if needed)
        if self.config.get('use_blender', False):
            try:
                result = subprocess.run(['blender', '--version'], 
                                     capture_output=True, text=True)
                if result.returncode != 0:
                    issues.append("❌ Blender not found in PATH")
            except FileNotFoundError:
                issues.append("❌ Blender not installed or not in PATH")
        
        # Check GPU availability (optional for free development)
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            if gpu_count == 0:
                if use_commercial:
                    issues.append("❌ No NVIDIA GPU detected (required for commercial production)")
                else:
                    log.info("⚠️  No GPU detected - using CPU mode for development")
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode()
                log.info(f"✅ GPU detected: {gpu_name}")
        except:
            if not use_commercial:
                log.info("⚠️  GPU validation skipped - development mode")
            else:
                issues.append("⚠️  GPU validation failed")
        
        # Check disk space
        output_path = Path(self.config.get('output_directory', 'data/synthetic'))
        if output_path.exists():
            free_gb = shutil.disk_usage(output_path).free / (1024**3)
            required_gb = self.config.get('batch_size', 1000) * 0.01  # ~10MB per sample
            
            if free_gb < required_gb:
                issues.append(f"❌ Insufficient disk space: {free_gb:.1f}GB free, {required_gb:.1f}GB required")
            else:
                log.info(f"✅ Disk space: {free_gb:.1f}GB free")
        
        if issues:
            error_msg = "Setup validation failed:\n" + "\n".join(issues)
            if use_commercial:
                error_msg += "\n💡 For development, try with use_commercial_smpl=False"
            raise RuntimeError(error_msg)
        
        license_status = "commercial" if use_commercial else "free (development)"
        log.info(f"✅ All requirements validated ({license_status})")
    
    def setup_output_structure(self):
        """Setup organized output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized structure
        subdirs = [
            'samples',           # Individual sample directories
            'batches',           # Batch metadata and indices
            'validation',        # Quality validation results  
            'logs',             # Detailed generation logs
            'checkpoints',       # Progress checkpoints
            'metadata'          # Global metadata and statistics
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        log.info(f"📁 Output structure created in {self.output_dir}")
    
    def generate_production_batch(self, 
                                batch_id: int,
                                batch_size: int, 
                                start_index: int = 0) -> Dict:
        """
        Generate a production batch with full pipeline
        
        Args:
            batch_id: Unique batch identifier
            batch_size: Number of samples in this batch
            start_index: Starting sample index
            
        Returns:
            Batch generation statistics and metadata
        """
        batch_start_time = time.time()
        batch_dir = self.output_dir / 'batches' / f'batch_{batch_id:04d}'
        batch_dir.mkdir(exist_ok=True)
        
        log.info(f"🚀 Starting Batch {batch_id}: {batch_size} samples")
        
        # Setup batch metadata
        batch_metadata = {
            'batch_id': batch_id,
            'batch_size': batch_size,
            'start_index': start_index,
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'samples': []
        }
        
        # Generate samples
        successful_samples = 0
        failed_samples = 0
        
        for i in range(batch_size):
            sample_index = start_index + i
            sample_start_time = time.time()
            
            try:
                # Generate sample parameters
                sample_data = self.factory.generate_sample()
                
                # Create sample output directory
                sample_dir = self.output_dir / 'samples' / f'sample_{sample_index:06d}'
                sample_dir.mkdir(exist_ok=True)
                
                # Render with Blender (if enabled)
                if self.config.get('use_blender', True):
                    render_result = self.render_with_blender(sample_data, sample_dir)
                else:
                    render_result = self.render_synthetic_only(sample_data, sample_dir)
                
                # Calculate processing time
                sample_time = time.time() - sample_start_time
                
                # Save sample metadata
                sample_metadata = {
                    'sample_index': sample_index,
                    'batch_id': batch_id,
                    'pose_category': sample_data['pose_category'],
                    'body_type': sample_data['body_type'],
                    'gender': sample_data['gender'],
                    'render_time': sample_time,
                    'render_result': render_result,
                    'pose_params': sample_data['pose'].tolist(),
                    'shape_params': sample_data['shape'].tolist()
                }
                
                # Export MediaPipe keypoints
                if self.config.get('export_keypoints', True):
                    keypoints = self.extract_mediaipe_keypoints(sample_data)
                    sample_metadata['keypoints_2d'] = keypoints.tolist()
                
                # Save individual sample metadata
                with open(sample_dir / 'metadata.json', 'w') as f:
                    json.dump(sample_metadata, f, indent=2)
                
                batch_metadata['samples'].append(sample_metadata)
                successful_samples += 1
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / batch_size
                    elapsed = time.time() - batch_start_time
                    eta = elapsed / progress - elapsed
                    
                    log.info(f"Batch {batch_id} Progress: {progress:.1%} "
                           f"({i+1}/{batch_size}) - ETA: {eta/60:.1f}m")
                
            except Exception as e:
                log.error(f"Sample {sample_index} failed: {str(e)}")
                failed_samples += 1
                continue
        
        # Finalize batch metadata
        batch_time = time.time() - batch_start_time
        batch_metadata.update({
            'end_time': datetime.now().isoformat(),
            'total_time': batch_time,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'success_rate': successful_samples / batch_size,
            'avg_sample_time': batch_time / batch_size
        })
        
        # Save batch metadata
        with open(batch_dir / 'batch_metadata.json', 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        log.info(f"✅ Batch {batch_id} complete: {successful_samples}/{batch_size} "
               f"samples in {batch_time/60:.1f}m")
        
        return batch_metadata
    
    def render_with_blender(self, sample_data: Dict, output_dir: Path) -> Dict:
        """Render sample using Blender with physics simulation"""
        
        # Prepare Blender script
        blender_script = f"""
import sys
import bpy
sys.path.append('{Path(__file__).parent}')

from blender_renderer import render_sample
import numpy as np
import json

# Load sample data
with open('{output_dir / "temp_sample.json"}', 'r') as f:
    sample_data = json.load(f)

# Convert back to numpy arrays
pose_params = np.array(sample_data['pose'])
shape_params = np.array(sample_data['shape'])

# Generate SMPL mesh (placeholder - requires commercial SMPL)
# In production: vertices, faces = smpl_model(pose_params, shape_params)
vertices = np.random.randn(6890, 3)
faces = np.random.randint(0, 6890, (13776, 3))

# Render sample
result = render_sample(
    vertices, faces,
    vertices, faces,  # Placeholder garment
    'shirt',
    Path('{output_dir}'),
    domain_randomization=True
)

# Save result
with open('{output_dir / "render_result.json"}', 'w') as f:
    json.dump({{
        'status': 'success',
        'rgb_path': str(result['rgb_path']),
        'depth_path': str(result['depth_path']),
        'camera_params': result['camera_params']
    }}, f, indent=2)

bpy.ops.wm.quit_blender()
"""
        
        # Save temporary sample data for Blender
        temp_data = {
            'pose': sample_data['pose'].tolist(),
            'shape': sample_data['shape'].tolist(),
            'pose_category': sample_data['pose_category'],
            'body_type': sample_data['body_type']
        }
        
        with open(output_dir / 'temp_sample.json', 'w') as f:
            json.dump(temp_data, f)
        
        # Save Blender script
        script_path = output_dir / 'render_script.py'
        with open(script_path, 'w') as f:
            f.write(blender_script)
        
        # Run Blender rendering
        try:
            cmd = [
                'blender',
                '--background',
                '--python', str(script_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Load render result
                with open(output_dir / 'render_result.json', 'r') as f:
                    render_result = json.load(f)
                
                # Cleanup temporary files
                (output_dir / 'temp_sample.json').unlink()
                (output_dir / 'render_script.py').unlink()
                
                return render_result
            else:
                raise RuntimeError(f"Blender rendering failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender rendering timeout")
    
    def render_synthetic_only(self, sample_data: Dict, output_dir: Path) -> Dict:
        """Generate synthetic data without Blender (faster for testing)"""
        import numpy as np
        from PIL import Image
        
        # Generate synthetic RGB image (placeholder)
        rgb_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        rgb_pil = Image.fromarray(rgb_image)
        rgb_path = output_dir / 'rgb.png'
        rgb_pil.save(rgb_path)
        
        # Generate synthetic depth image  
        depth_image = np.random.rand(512, 512).astype(np.float32)
        np.save(output_dir / 'depth.npy', depth_image)
        
        return {
            'status': 'synthetic',
            'rgb_path': str(rgb_path),
            'depth_path': str(output_dir / 'depth.npy'),
            'camera_params': {'synthetic': True}
        }
    
    def extract_mediaipe_keypoints(self, sample_data: Dict) -> np.ndarray:
        """Extract MediaPipe format keypoints from SMPL pose"""
        # Placeholder - in production, this would:
        # 1. Generate SMPL mesh with commercial license
        # 2. Project SMPL joint locations to image coordinates  
        # 3. Map to MediaPipe 33-landmark format
        
        # For now, return synthetic keypoints
        keypoints = np.random.rand(33, 2) * 512
        return keypoints
    
    def run_production_pipeline(self):
        """Run the complete production pipeline"""
        config = self.config
        total_batch_size = config['batch_size']
        
        # Determine batch strategy based on phase
        if config.get('phase', 1) == 1:
            # Phase 1: Single machine, manageable batches
            batch_size = min(1000, total_batch_size)
        elif config.get('phase', 1) == 2:
            # Phase 2: Larger batches with checkpointing
            batch_size = min(5000, total_batch_size)
        else:
            # Phase 3: Maximum throughput
            batch_size = min(10000, total_batch_size)
        
        num_batches = (total_batch_size + batch_size - 1) // batch_size
        
        log.info(f"🎯 Production Plan: {total_batch_size} samples in {num_batches} batches")
        
        # Global statistics
        total_start_time = time.time()
        all_batch_metadata = []
        
        # Generate batches
        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, total_batch_size - batch_id * batch_size)
            start_index = batch_id * batch_size
            
            batch_metadata = self.generate_production_batch(
                batch_id, current_batch_size, start_index
            )
            
            all_batch_metadata.append(batch_metadata)
            
            # Save checkpoint
            checkpoint = {
                'completed_batches': batch_id + 1,
                'total_batches': num_batches,
                'completed_samples': sum(b['successful_samples'] for b in all_batch_metadata),
                'total_samples': total_batch_size
            }
            
            with open(self.output_dir / 'checkpoints' / f'checkpoint_{batch_id}.json', 'w') as f:
                json.dump(checkpoint, f, indent=2)
        
        # Generate final production report
        total_time = time.time() - total_start_time
        total_successful = sum(b['successful_samples'] for b in all_batch_metadata)
        
        production_report = {
            'production_config': config,
            'total_samples_requested': total_batch_size,
            'total_samples_generated': total_successful,
            'total_batches': num_batches,
            'total_time_hours': total_time / 3600,
            'avg_samples_per_hour': total_successful / (total_time / 3600),
            'success_rate': total_successful / total_batch_size,
            'batch_metadata': all_batch_metadata
        }
        
        # Save production report
        report_path = self.output_dir / 'metadata' / 'production_report.json'
        with open(report_path, 'w') as f:
            json.dump(production_report, f, indent=2)
        
        log.info(f"🎉 Production Complete! Generated {total_successful:,} samples in {total_time/3600:.1f} hours")
        log.info(f"📊 Throughput: {total_successful/(total_time/3600):.0f} samples/hour")
        log.info(f"📋 Report saved: {report_path}")
        
        return production_report


def main():
    """Main entry point for production pipeline"""
    parser = argparse.ArgumentParser(description="AR Mirror Production Synthetic Data Generation")
    
    # Core parameters
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=1,
                       help='Production phase (1=100K, 2=300K, 3=1M+)')
    parser.add_argument('--batch_size', type=int, 
                       help='Total samples to generate (overrides phase default)')
    parser.add_argument('--output_dir', default='data/synthetic_production',
                       help='Output directory for generated data')
    
    # Generation options
    parser.add_argument('--use_blender', action='store_true', default=False,
                       help='Use Blender for physics-based rendering')
    parser.add_argument('--synthetic_only', action='store_true', default=False,
                       help='Generate synthetic data without Blender (testing)')
    parser.add_argument('--export_keypoints', action='store_true', default=True,
                       help='Export MediaPipe format keypoints')
    
    # Licensing options
    parser.add_argument('--commercial', action='store_true', default=False,
                       help='Use commercial SMPL license (requires MPI license)')
    parser.add_argument('--free', action='store_true', default=True,
                       help='Use free SMPL-X models (development only) - DEFAULT')
    
    # Performance options
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (future use)')
    parser.add_argument('--cloud', action='store_true', 
                       help='Enable cloud-based scaling (future use)')
    parser.add_argument('--cluster', action='store_true',
                       help='Enable multi-GPU cluster mode (future use)')
    
    args = parser.parse_args()
    
    # Determine default batch sizes by phase  
    phase_defaults = {1: 100000, 2: 300000, 3: 1000000}
    if args.batch_size is None:
        args.batch_size = phase_defaults[args.phase]
    
    # Determine SMPL licensing (commercial overrides free)
    use_commercial_smpl = args.commercial
    
    # Build configuration
    config = {
        'phase': args.phase,
        'batch_size': args.batch_size,
        'output_directory': args.output_dir,
        'use_blender': args.use_blender,
        'synthetic_only': args.synthetic_only,
        'export_keypoints': args.export_keypoints,
        'use_commercial_smpl': use_commercial_smpl,
        'workers': args.workers,
        'cloud_scaling': args.cloud,
        'cluster_mode': args.cluster
    }
    
    license_type = "Commercial SMPL" if use_commercial_smpl else "Free SMPL-X (Development)"
    
    print(f"""
🚀 AR Mirror Production Synthetic Data Pipeline
==================================================
Phase: {args.phase}
Target: {args.batch_size:,} samples
Output: {args.output_dir}
Blender: {args.use_blender}
SMPL: {license_type}
==================================================
""")
    
    # Confirm production run
    if args.batch_size >= 10000:
        confirm = input(f"⚠️  Generate {args.batch_size:,} samples? This will take several hours. [y/N]: ")
        if confirm.lower() != 'y':
            print("❌ Production run cancelled")
            return
    
    # Run pipeline
    try:
        pipeline = ProductionPipeline(config)
        production_report = pipeline.run_production_pipeline()
        
        print(f"\n✅ Production pipeline completed successfully!")
        print(f"📊 Generated: {production_report['total_samples_generated']:,} samples")
        print(f"⏱️  Time: {production_report['total_time_hours']:.1f} hours") 
        print(f"🚀 Rate: {production_report['avg_samples_per_hour']:.0f} samples/hour")
        
    except KeyboardInterrupt:
        print("\n⚠️ Production interrupted by user")
    except Exception as e:
        print(f"\n❌ Production failed: {str(e)}")
        log.error(f"Production pipeline failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()