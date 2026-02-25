"""
VITON Integration Demo
=====================
Demonstrates the difference between placeholder assets and VITON dataset

Usage:
    python viton_demo.py --mode [placeholder|viton|compare]
    
Examples:
    python viton_demo.py --mode placeholder  # Show placeholder garments only
    python viton_demo.py --mode viton        # Show VITON dataset only
    python viton_demo.py --mode compare      # Show side-by-side comparison
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import argparse
from src.legacy.sizing_pipeline import SizingPipeline
from src.viton.viton_integration import VITONGarmentLoader


def create_comparison_view(placeholder_frame: np.ndarray, viton_frame: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison view"""
    h1, w1 = placeholder_frame.shape[:2]
    h2, w2 = viton_frame.shape[:2]
    
    # Resize to same height
    target_h = min(h1, h2)
    scale1 = target_h / h1
    scale2 = target_h / h2
    
    placeholder_resized = cv2.resize(placeholder_frame, (int(w1 * scale1), target_h))
    viton_resized = cv2.resize(viton_frame, (int(w2 * scale2), target_h))
    
    # Create combined frame
    combined = np.hstack([placeholder_resized, viton_resized])
    
    # Add labels
    cv2.putText(combined, "Placeholder Assets", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    cv2.putText(combined, "VITON Dataset", (int(w1 * scale1 + 20), 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return combined


def run_placeholder_demo():
    """Demo with placeholder assets only"""
    print("\n" + "="*60)
    print("PLACEHOLDER ASSETS DEMO")
    print("="*60)
    
    pipeline = SizingPipeline(
        "data/garments/garment_database.json",
        "data/logs",
        use_viton=False
    )
    
    cap = cv2.VideoCapture(0)
    
    print("Controls:")
    print("  Q - Quit")
    print("  Arrow keys - Change garment")
    print("  N/P - Next/Previous size")
    print("  G - Toggle garment overlay")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_frame, result = pipeline.process_frame(frame)
        
        # Add demo label
        cv2.putText(output_frame, "Mode: Placeholder Assets", (10, output_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('VITON Demo - Placeholder', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 83:  # Right arrow
            pipeline.cycle_garment(1)
        elif key == 81:  # Left arrow
            pipeline.cycle_garment(-1)
        elif key == ord('n'):
            # Cycle through sizes
            sizes = ['S', 'M', 'L', 'XL']
            current_idx = sizes.index(pipeline.current_size_override or 'M')
            pipeline.current_size_override = sizes[(current_idx + 1) % len(sizes)]
        elif key == ord('p'):
            sizes = ['S', 'M', 'L', 'XL']
            current_idx = sizes.index(pipeline.current_size_override or 'M')
            pipeline.current_size_override = sizes[(current_idx - 1) % len(sizes)]
        elif key == ord('g'):
            pipeline.enable_garment_overlay = not pipeline.enable_garment_overlay
    
    cap.release()
    cv2.destroyAllWindows()


def run_viton_demo():
    """Demo with VITON dataset"""
    print("\n" + "="*60)
    print("VITON DATASET DEMO")
    print("="*60)
    
    viton_root = "dataset"  # Your VITON dataset location
    if not Path(viton_root).exists():
        print(f"❌ VITON dataset not found at: {viton_root}")
        print("Please download the dataset first")
        return
    
    pipeline = SizingPipeline(
        "data/garments/garment_database.json",  # Size database for matching
        "data/logs",
        viton_root=viton_root,
        use_viton=True,
        inventory_path="data/garments/garment_inventory.json"  # Inventory for display
    )
    
    if not pipeline.use_viton:
        print("❌ VITON integration failed. Check logs above.")
        return
    
    cap = cv2.VideoCapture(0)
    
    # Force cycle to first garment to ensure one is loaded
    if pipeline.inventory:
        pipeline.cycle_garment(0)
        print(f"📦 Loaded first garment: {pipeline.current_sku}")
        print(f"🎨 Garment overlay: {'ON' if pipeline.enable_garment_overlay else 'OFF'}")
    
    print("Controls:")
    print("  Q - Quit")
    print("  Arrow keys - Change garment")
    print("  N/P - Next/Previous size")
    print("  G - Toggle garment overlay")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_frame, result = pipeline.process_frame(frame)
        
        # Add demo label with garment info
        label_text = f"Mode: VITON Dataset | Garment: {pipeline.current_sku} | Overlay: {'ON' if pipeline.enable_garment_overlay else 'OFF'}"
        cv2.putText(output_frame, label_text, (10, output_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('VITON Demo - Dataset', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 83:  # Right arrow
            pipeline.cycle_garment(1)
        elif key == 81:  # Left arrow
            pipeline.cycle_garment(-1)
        elif key == ord('n'):
            sizes = ['S', 'M', 'L', 'XL']
            current_idx = sizes.index(pipeline.current_size_override or 'M')
            pipeline.current_size_override = sizes[(current_idx + 1) % len(sizes)]
        elif key == ord('p'):
            sizes = ['S', 'M', 'L', 'XL']
            current_idx = sizes.index(pipeline.current_size_override or 'M')
            pipeline.current_size_override = sizes[(current_idx - 1) % len(sizes)]
        elif key == ord('g'):
            pipeline.enable_garment_overlay = not pipeline.enable_garment_overlay
    
    cap.release()
    cv2.destroyAllWindows()


def run_comparison_demo():
    """Side-by-side comparison demo"""
    print("\n" + "="*60)
    print("COMPARISON DEMO (Placeholder vs VITON)")
    print("="*60)
    
    viton_root = "dataset"  # Your VITON dataset location
    if not Path(viton_root).exists():
        print(f"❌ VITON dataset not found at: {viton_root}")
        print("Please download the dataset first")
        return
    
    # Create two pipelines
    pipeline_placeholder = SizingPipeline(
        "data/garments/garment_database.json",
        "data/logs",
        use_viton=False
    )
    
    pipeline_viton = SizingPipeline(
        "data/garments/garment_database.json",
        "data/logs",
        viton_root=viton_root,
        use_viton=True
    )
    
    if not pipeline_viton.use_viton:
        print("❌ VITON integration failed. Check logs above.")
        return
    
    cap = cv2.VideoCapture(0)
    
    print("Controls:")
    print("  Q - Quit")
    print("  Arrow keys - Change garment")
    print("  N/P - Next/Previous size")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with both pipelines
        output_placeholder, _ = pipeline_placeholder.process_frame(frame.copy())
        output_viton, _ = pipeline_viton.process_frame(frame.copy())
        
        # Create comparison view
        comparison_frame = create_comparison_view(output_placeholder, output_viton)
        
        cv2.imshow('VITON Demo - Comparison', comparison_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 83:  # Right arrow
            pipeline_placeholder.cycle_garment(1)
            pipeline_viton.cycle_garment(1)
        elif key == 81:  # Left arrow
            pipeline_placeholder.cycle_garment(-1)
            pipeline_viton.cycle_garment(-1)
        elif key == ord('n'):
            sizes = ['S', 'M', 'L', 'XL']
            current_idx = sizes.index(pipeline_placeholder.current_size_override or 'M')
            new_size = sizes[(current_idx + 1) % len(sizes)]
            pipeline_placeholder.current_size_override = new_size
            pipeline_viton.current_size_override = new_size
        elif key == ord('p'):
            sizes = ['S', 'M', 'L', 'XL']
            current_idx = sizes.index(pipeline_placeholder.current_size_override or 'M')
            new_size = sizes[(current_idx - 1) % len(sizes)]
            pipeline_placeholder.current_size_override = new_size
            pipeline_viton.current_size_override = new_size
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="VITON Integration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python viton_demo.py --mode placeholder
  python viton_demo.py --mode viton
  python viton_demo.py --mode compare
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['placeholder', 'viton', 'compare'],
        default='compare',
        help='Demo mode (default: compare)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'placeholder':
        run_placeholder_demo()
    elif args.mode == 'viton':
        run_viton_demo()
    elif args.mode == 'compare':
        run_comparison_demo()


if __name__ == "__main__":
    main()
