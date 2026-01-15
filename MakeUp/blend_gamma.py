import cv2
import numpy as np
#Blend mềm – không “đè màu” – giống filter Instagram hơn

def feather_mask(mask: np.ndarray, ksize: int = 31) -> np.ndarray:
    """
    Smooth mask edges
    """
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0) / 255.0


def gamma_blend(
    roi: np.ndarray,
    makeup: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.8
) -> np.ndarray:
    """
    Gamma-aware alpha blending
    """
    alpha = feather_mask(mask)

    blended = (
        roi * (1 - alpha[..., None] * strength)
        + makeup * (alpha[..., None] * strength)
    )

    return blended.astype(np.uint8)
