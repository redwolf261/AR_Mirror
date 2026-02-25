import cv2
import mediapipe as mp
import numpy as np

class BodySegmenter:
    """ Produces body mask for GMM input using MediaPipe Selfie Segmentation """
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation  # type: ignore[attr-defined]
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # 1 = Landscape (faster)

    def segment(self, image):
        """
        Args:
            image: RGB numpy array
        Returns:
            mask: Binary mask (0 or 1), shape (H, W, 1)
        """
        results = self.segmenter.process(image)
        if results.segmentation_mask is None:
            return np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)

        # Threshold to binary
        mask = results.segmentation_mask > 0.5
        return mask.astype(np.float32)[:, :, np.newaxis]
