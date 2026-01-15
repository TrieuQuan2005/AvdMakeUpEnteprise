from PIL import Image
import numpy as np
import mediapipe as mp
import cv2

# =========================
# Image / FaceMesh Utils
# =========================

mp_face_mesh = mp.solutions.face_mesh


def load_image(path: str) -> np.ndarray:
    """
    Load image from disk and convert to OpenCV BGR format.
    - Handles RGB conversion
    - Safe for MediaPipe / OpenCV pipeline

    Returns:
        img_bgr (H, W, 3) uint8
    """
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_face_mesh(static: bool = False) -> mp_face_mesh.FaceMesh:
    """
    Create MediaPipe FaceMesh instance.

    Args:
        static: True for single image, False for video/stream

    Returns:
        MediaPipe FaceMesh object
    """
    return mp_face_mesh.FaceMesh(
        static_image_mode=static,
        max_num_faces=1,
        refine_landmarks=True
    )
