import numpy as np
import cv2
from typing import Dict, Tuple

class EyeGeometry:
    @staticmethod
    def landmarks_to_points(landmarks, indices, img_w, img_h) -> np.ndarray:
        return np.array(
            [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in indices],
            dtype=np.float32
        )

    @staticmethod
    def compute_geometry(pts: np.ndarray) -> Dict:
        center = pts.mean(axis=0)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min

        return {
            "center": center,
            "width": float(width),
            "height": float(height),
            "aspect_ratio": float(height / (width + 1e-6))
        }

    @staticmethod
    def polygon_mask(image_shape, pts: np.ndarray) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        return mask

    @staticmethod
    def symmetry_axis(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        center = pts.mean(axis=0)
        pts_centered = pts - center
        _, _, vh = np.linalg.svd(pts_centered)
        return center, vh[0]
