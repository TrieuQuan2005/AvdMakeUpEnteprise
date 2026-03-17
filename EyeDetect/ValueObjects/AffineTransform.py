import numpy as np

class AffineTransform:
    def __init__(self, forward: np.ndarray, inverse: np.ndarray):
        self.forward = forward      # image → normalized
        self.inverse = inverse      # normalized → image