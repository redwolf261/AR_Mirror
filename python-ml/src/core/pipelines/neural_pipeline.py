import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os
import cv2

# Add path for core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from cp_vton.networks import GMM, UnetGenerator
from src.core.live_pose_converter import LivePoseConverter
from src.core.body_segmenter import BodySegmenter

class NeuralPipeline:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gmm = None
        self.tom = None
        self.initialized = False
        
        # Helpers
        self.pose_converter = LivePoseConverter(256, 192)
        self.body_segmenter = BodySegmenter()
        
        # Transforms (ImageNet norm is standard, but CP-VTON usually uses 0.5/0.5)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_models(self, checkpoint_dir):
        """ Loads GMM and TOM from checkpoints """
        try:
            print(f"Loading neural models on {self.device}...")
            
            # Load GMM
            self.gmm = GMM(opt=None)
            gmm_path = os.path.join(checkpoint_dir, 'gmm_final.pth')
            if os.path.exists(gmm_path):
                # Handle CPU loading if needed
                state_dict = torch.load(gmm_path, map_location=self.device)
                self.gmm.load_state_dict(state_dict)
                self.gmm.to(self.device)
                self.gmm.eval()
                print("✓ GMM loaded")
            else:
                print(f"⚠ Warning: GMM checkpoint not found at {gmm_path}")

            # Load TOM
            self.tom = UnetGenerator(25, 4, 6, ngf=64) 
            tom_path = os.path.join(checkpoint_dir, 'tom_final.pth')
            if os.path.exists(tom_path):
                state_dict = torch.load(tom_path, map_location=self.device)
                self.tom.load_state_dict(state_dict)
                self.tom.to(self.device)
                self.tom.eval()
                print("✓ TOM loaded")
                self.initialized = True
            else:
                print(f"⚠ Warning: TOM checkpoint not found at {tom_path}")
            
            return self.initialized
        except Exception as e:
            print(f"❌ Failed to load neural models: {e}")
            return False

    def process_frame(self, frame, landmarks, cloth_image_path):
        """ 
        Main inference loop
        frame: numpy array (H,W,3) RGB
        landmarks: mediapipe landmarks list
        cloth_image_path: path to cloth image
        """
        if not self.initialized or self.gmm is None:
            return frame # Fallback

        try:
            # 1. Preprocess Cloth
            # CACHE FIX: Check if we have processed this cloth before
            if not hasattr(self, 'cloth_cache'):
                self.cloth_cache = {}
            
            if cloth_image_path in self.cloth_cache:
                c_tensor = self.cloth_cache[cloth_image_path]
            else:
                c = Image.open(cloth_image_path).convert('RGB')
                c = c.resize((192, 256))
                c_tensor = self.transform(c).unsqueeze(0).to(self.device)  # type: ignore[union-attr]  # (1,3,256,192)
                self.cloth_cache[cloth_image_path] = c_tensor

            # 2. Preprocess Pose
            pose_map = self.pose_converter.convert_to_heatmap(landmarks)  # type: ignore[attr-defined]  # (18, 256, 192)
            pose_tensor = torch.from_numpy(pose_map).unsqueeze(0).to(self.device) # (1,18,256,192)

            # 3. Preprocess Body (Segment & Agnostic)
            # Resize frame for processing
            frame_resized = cv2.resize(frame, (192, 256))
            frame_tensor = self.transform(Image.fromarray(frame_resized)).unsqueeze(0).to(self.device)  # type: ignore[union-attr]  # (1,3,256,192)
            
            # Get body mask
            body_mask = self.body_segmenter.segment(frame_resized) # (256, 192, 1)
            body_mask_tensor = torch.from_numpy(body_mask.transpose(2,0,1)).unsqueeze(0).to(self.device) # (1,1,256,192)

            # Create "Head" representation (Masked body)
            # In data pre-processing, 'agnostic' usually masks out the bounding box of the cloth.
            # Here we just pass the full body as a rough approximation for the 'head' channels if we don't have a parser.
            # Ideally: Head = Frame * HeadMask. We use Frame * BodyMask as a poor man's approximation.
            head_tensor = frame_tensor * body_mask_tensor # (1,3,256,192)

            # 4. GMM Inference (Warp cloth)
            # Input A: Pose (18) + Body Shape (1) + Head (3) = 22 channels
            inputA = torch.cat([pose_tensor, body_mask_tensor, head_tensor], 1) # (1, 22, 256, 192) 
            inputB = c_tensor # (1, 3, 256, 192)

            grid = self.gmm(inputA, inputB)
            warped_cloth = F.grid_sample(c_tensor, grid, padding_mode='border')

            # 5. TOM Inference (Synthesis)
            # TOM input: Pose(18) + Agnostic(3) + WarpedCloth(3) + Shape(1) ?
            # Standard TOM GMM-based: cat([pose, agnostic, warped_cloth], 1) -> 18+3+3=24 ??
            # My UnetGenerator in networks.py was initialized with `input_nc=25`.
            # Let's check: 18 (Pose) + 3 (Head/Agnostic) + 3 (Warped) + 1 (Shape) = 25.
            tom_input = torch.cat([pose_tensor, head_tensor, warped_cloth, body_mask_tensor], 1)
            output = self.tom(tom_input) # (1, 4, 256, 192) -> Computed image + composition mask

            # 6. Post-process
            # Render output (first 3 channels)
            out_img = output[0, :3, :, :].cpu().detach().numpy()
            out_img = (out_img * 0.5 + 0.5) * 255.0
            out_img = out_img.transpose(1, 2, 0).astype(np.uint8)
            
            # Resize back to original frame (optional, or just overlay 256x192)
            out_img_resized = cv2.resize(out_img, (frame.shape[1], frame.shape[0]))
            
            return out_img_resized
        except Exception as e:
            print(f"Inference error: {e}")
            return frame
