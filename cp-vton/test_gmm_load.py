"""
Step 2: Verify GMM checkpoint loads correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from networks import GMM

print("=" * 60)
print("CP-VTON GMM Checkpoint Loading Test")
print("=" * 60)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create GMM model
print("\n[1/4] Creating GMM architecture...")
class Args:
    fine_height = 256
    fine_width = 192
    grid_size = 5

opt = Args()
model = GMM(opt)
print("✓ GMM architecture created")

# Load checkpoint
print("\n[2/4] Loading checkpoint...")
checkpoint_path = "checkpoints/gmm_train_new/gmm_final.pth"
if not os.path.exists(checkpoint_path):
    print(f"✗ Checkpoint not found at: {checkpoint_path}")
    sys.exit(1)

ckpt = torch.load(checkpoint_path, map_location=device)
print(f"✓ Checkpoint loaded from disk")
print(f"  Keys in checkpoint: {len(ckpt.keys())}")

# Load weights into model
print("\n[3/4] Loading weights into model...")
try:
    model.load_state_dict(ckpt)
    print("✓ Weights loaded successfully")
except Exception as e:
    print(f"✗ Failed to load weights: {e}")
    sys.exit(1)

# Move to device and set eval mode
print("\n[4/4] Preparing model for inference...")
model.to(device)
model.eval()
print(f"✓ Model ready on {device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

print("\n" + "=" * 60)
print("✓ GMM CHECKPOINT LOADED SUCCESSFULLY")
print("=" * 60)
print("\nReady for inference integration.")
