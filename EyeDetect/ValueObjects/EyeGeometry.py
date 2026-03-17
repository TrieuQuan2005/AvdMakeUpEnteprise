import numpy as np

class EyeGeometry:
    def __init__(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        aspect_ratio: float
    ):
        self.center = center
        self.width = width
        self.height = height
        self.aspect_ratio = aspect_ratio

    @property
    def openness(self):
        return self.height / (self.width + 1e-6)