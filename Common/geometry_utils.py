#Geometry uitlilities
import numpy as np
import cv2
from typing import Dict, Tuple

def landmarks_to_points(landmarks, indices, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert MediaPipe landmarks to Nx2 numpy array (pixel coordinates)
    """
    return np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in indices
    ], dtype=np.float32)

def compute_eye_geometry(pts: np.ndarray) -> Dict:
    """
    Compute basic geometric properties of eye region
    """
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

def polygon_mask(image_shape: Tuple[int, int, int], pts: np.ndarray) -> np.ndarray:
    """
    Create binary mask from polygon points
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask

def normalize_crop(image, box, output_size=128):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None, None, None

    scale_x = output_size / w
    scale_y = output_size / h

    forward = np.array([
        [scale_x, 0, -x1 * scale_x],
        [0, scale_y, -y1 * scale_y]
    ], dtype=np.float32)

    inverse = np.array([
        [1 / scale_x, 0, x1],
        [0, 1 / scale_y, y1]
    ], dtype=np.float32)

    aligned = cv2.resize(crop, (output_size, output_size))

    return aligned, forward, inverse


def eye_symmetry_axis(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute approximate symmetry axis of eye
    Returns: (point_on_axis, direction_vector)
    """
    center = pts.mean(axis=0)

    # PCA để tìm trục chính
    pts_centered = pts - center
    _, _, vh = np.linalg.svd(pts_centered)

    direction = vh[0]  # principal axis
    return center, direction
