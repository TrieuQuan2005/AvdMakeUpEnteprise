import numpy as np

from EyeDetect.ValueObjects.AffineTransform import AffineTransform
from EyeDetect.ValueObjects.BoundingBox import BoundingBox

class NormalizedEye:
    def __init__(
        self,
        image: np.ndarray,
        transform: AffineTransform,
        box: BoundingBox
    ):
        self.image = image
        self.transform = transform
        self.box = box