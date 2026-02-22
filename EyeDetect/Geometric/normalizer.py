import numpy as np
import cv2
from typing import Tuple, Optional

class EyeNormalizer:
    def __init__(self, output_size: int = 128):
        self.output_size = output_size

    def normalize(self,image: np.ndarray,box: Tuple[int, int, int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        h, w = crop.shape[:2]

        sx = self.output_size / w
        sy = self.output_size / h

        forward = np.array([
            [sx, 0, -x1 * sx],
            [0, sy, -y1 * sy]
        ], dtype=np.float32)

        inverse = np.array([
            [1 / sx, 0, x1],
            [0, 1 / sy, y1]
        ], dtype=np.float32)

        aligned = cv2.resize(crop, (self.output_size, self.output_size))
        return aligned, forward, inverse
