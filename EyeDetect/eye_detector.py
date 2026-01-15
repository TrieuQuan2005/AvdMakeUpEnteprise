import numpy as np
import cv2
from typing import Dict, Tuple, Optional

from Common.geometry_utils import (
    landmarks_to_points,
    compute_eye_geometry,
    polygon_mask,
    normalize_crop,
    eye_symmetry_axis
)

# =========================
# MediaPipe Eye Landmark Indices
# =========================

LEFT_EYE_EXT_IDX = [
    33, 133, 160, 159, 158, 157, 173,
    70, 63, 105, 66, 107,
    145, 153, 154, 155, 157, 158, 159,
    6, 197
]

RIGHT_EYE_EXT_IDX = [
    362, 263, 387, 386, 385, 384, 398,
    336, 296, 334, 293, 300,
    374, 380, 381, 382, 384, 385, 386,
    6, 417
]


# =========================
# Internal Helpers
# =========================

def get_eye_box(
    landmarks,
    indices,
    img_w: int,
    img_h: int
) -> Tuple[int, int, int, int]:
    """
    Compute padded bounding box for eye region
    """
    pts = landmarks_to_points(landmarks, indices, img_w, img_h)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    ew = x_max - x_min
    eh = y_max - y_min

    x1 = int(x_min - 0.10 * ew)
    x2 = int(x_max + 0.25 * ew)
    y1 = int(y_min - 0.32 * eh)
    y2 = int(y_max + 0.39 * eh)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    return x1, y1, x2, y2


def extract_eye_region(
    image: np.ndarray,
    landmarks,
    indices,
    output_size: int = 128
) -> Optional[Dict]:
    """
    Extract full EyeRegion representation for one eye
    """
    h, w, _ = image.shape

    # Landmark → pixel points
    pts = landmarks_to_points(landmarks, indices, w, h)

    if pts.shape[0] < 3:
        return None

    # Geometry
    geometry = compute_eye_geometry(pts)

    # Bounding box
    box = get_eye_box(landmarks, indices, w, h)

    # Mask
    mask = polygon_mask(image.shape, pts)

    # Normalized crop
    aligned, forward_tf, inverse_tf = normalize_crop(
        image=image,
        box=box,
        output_size=output_size
    )

    if aligned is None:
        return None

    # Symmetry axis
    sym_center, sym_dir = eye_symmetry_axis(pts)

    return {
        "geometry": geometry,
        "box": box,
        "masks": {
            "eye_region": mask
        },
        "normalized": {
            "aligned_crop": aligned,
            "forward_transform": forward_tf,
            "inverse_transform": inverse_tf
        },
        "structure": {
            "symmetry_center": sym_center,
            "symmetry_direction": sym_dir
        }
    }


# =========================
# Public API
# =========================

def detect_eye_regions(
    image_bgr: np.ndarray,
    face_mesh
) -> Optional[Dict]:
    """
    Detect and extract EyeRegion representations for left & right eye

    Returns:
        {
            "left_eye": EyeRegion,
            "right_eye": EyeRegion,
            "face_landmarks": raw landmarks
        }
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    left_eye = extract_eye_region(
        image=image_bgr,
        landmarks=landmarks,
        indices=LEFT_EYE_EXT_IDX
    )

    right_eye = extract_eye_region(
        image=image_bgr,
        landmarks=landmarks,
        indices=RIGHT_EYE_EXT_IDX
    )

    if left_eye is None or right_eye is None:
        return None

    return {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "face_landmarks": landmarks
    }
