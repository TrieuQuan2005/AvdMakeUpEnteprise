import numpy as np


class TemporalSmoother:
    def __init__(self, momentum: float = 0.8):
        self.momentum = momentum
        self.prev = None
        self.prev_shape = None

    def smooth(self, current: np.ndarray) -> np.ndarray:
        if (
            self.prev is None
            or self.prev_shape != current.shape
        ):
            self.prev = current.astype(np.float32)
            self.prev_shape = current.shape
            return current

        smoothed = (
            self.momentum * self.prev
            + (1 - self.momentum) * current
        )

        self.prev = smoothed
        return smoothed.astype(np.uint8)
