"""
GMM TPS Warper - Replaces resize-paste logic with geometric deformation
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

# Add CP-VTON to path
cp_vton_path = Path(__file__).parent / "cp-vton"
sys.path.insert(0, str(cp_vton_path))

from networks import GMM


class GMMWarper:
    """TPS-based cloth warping using CP-VTON+ GMM"""
    
    def __init__(self, checkpoint_path="cp-vton/checkpoints/gmm_train_new/gmm_final.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model config (CP-VTON+ standard)
        class Args:
            fine_height = 256
            fine_width = 192
            grid_size = 5
        
        self.opt = Args()
        
        # Load model
        print(f"Loading GMM from {checkpoint_path}...")
        self.model = GMM(self.opt)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()
        print(f"GMM loaded on {self.device}")
        
    @torch.no_grad()
    def warp_cloth(self, cloth_rgb, cloth_mask, pose_map, agnostic):
        """
        Warp cloth using TPS transformation
        
        Args:
            cloth_rgb: (H, W, 3) numpy array, float32 [0,1]
            cloth_mask: (H, W) numpy array, float32 [0,1]
            pose_map: (18, 256, 192) numpy array, float32 [0,1]
            agnostic: (22, 256, 192) numpy array, float32 [-1,1]
                     = concat([shape, im_h, pose_map])
        
        Returns:
            warped_cloth: (H, W, 3) numpy array, float32 [0,1]
            warped_mask: (H, W) numpy array, float32 [0,1]
        """
        # Resize inputs to GMM resolution
        H, W = self.opt.fine_height, self.opt.fine_width
        cloth_rgb_resized = cv2.resize(cloth_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        cloth_mask_resized = cv2.resize(cloth_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Convert to torch tensors
        # Cloth: (H,W,3) -> (3,H,W) -> (1,3,H,W), normalize to [-1,1]
        cloth_t = torch.from_numpy(cloth_rgb_resized).permute(2, 0, 1).unsqueeze(0)
        cloth_t = cloth_t * 2.0 - 1.0  # [0,1] -> [-1,1]
        cloth_t = cloth_t.to(self.device).float()
        
        # Cloth mask: (H,W) -> (1,H,W) -> (1,1,H,W)
        cm_t = torch.from_numpy(cloth_mask_resized).unsqueeze(0).unsqueeze(0)
        cm_t = cm_t.to(self.device).float()
        
        # Agnostic: (22,H,W) -> (1,22,H,W)
        agnostic_t = torch.from_numpy(agnostic).unsqueeze(0)
        agnostic_t = agnostic_t.to(self.device).float()
        
        # GMM forward pass
        grid, theta = self.model(agnostic_t, cm_t)
        
        # Warp cloth and mask using predicted grid
        warped_cloth_t = F.grid_sample(cloth_t, grid, padding_mode='border', align_corners=True)
        warped_mask_t = F.grid_sample(cm_t, grid, padding_mode='zeros', align_corners=True)
        
        # Convert back to numpy
        warped_cloth = warped_cloth_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        warped_cloth = (warped_cloth + 1.0) / 2.0  # [-1,1] -> [0,1]
        warped_cloth = np.clip(warped_cloth, 0, 1)
        
        warped_mask = warped_mask_t.squeeze(0).squeeze(0).cpu().numpy()
        warped_mask = np.clip(warped_mask, 0, 1)
        
        return warped_cloth, warped_mask


def build_agnostic_representation(pose_map, person_shape, person_head):
    """
    Build cloth-agnostic representation for GMM input
    
    Args:
        pose_map: (18, 256, 192) float32 [0,1]
        person_shape: (256, 192, 3) float32 [0,1] - body silhouette
        person_head: (256, 192, 3) float32 [0,1] - preserved head region
    
    Returns:
        agnostic: (22, 256, 192) float32 [-1,1]
    """
    # Convert shape to grayscale and normalize to [-1,1]
    if person_shape.ndim == 3:
        shape_gray = cv2.cvtColor((person_shape * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        shape_gray = shape_gray.astype(np.float32) / 255.0
    else:
        shape_gray = person_shape
    shape_gray = shape_gray * 2.0 - 1.0  # [0,1] -> [-1,1]
    
    # Convert head to grayscale and normalize to [-1,1]
    if person_head.ndim == 3:
        head_gray = cv2.cvtColor((person_head * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        head_gray = head_gray.astype(np.float32) / 255.0
    else:
        head_gray = person_head
    head_gray = head_gray * 2.0 - 1.0  # [0,1] -> [-1,1]
    
    # Stack: shape(1) + head(1) + pose(18) + reserved(2) = 22 channels
    # For now, use zeros for reserved channels
    agnostic = np.zeros((22, 256, 192), dtype=np.float32)
    agnostic[0] = shape_gray
    agnostic[1] = head_gray
    agnostic[2:20] = pose_map  # 18 pose channels
    # agnostic[20:22] remain zero (reserved)
    
    return agnostic


if __name__ == "__main__":
    # Test GMM warper with dummy data
    print("Testing GMM Warper...")
    
    warper = GMMWarper()
    
    # Create dummy inputs
    cloth_rgb = np.random.rand(768, 1024, 3).astype(np.float32)
    cloth_mask = np.random.rand(768, 1024).astype(np.float32)
    pose_map = np.random.rand(18, 256, 192).astype(np.float32)
    agnostic = np.random.randn(22, 256, 192).astype(np.float32)
    
    print("Running inference...")
    warped_cloth, warped_mask = warper.warp_cloth(cloth_rgb, cloth_mask, pose_map, agnostic)
    
    print(f"Warped cloth shape: {warped_cloth.shape}")
    print(f"Warped mask shape: {warped_mask.shape}")
    print("GMM Warper test passed!")
