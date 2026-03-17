import numpy as np

from EyeDetect.Enums.EyeSide import EyeSide
from EyeDetect.ValueObjects.BoundingBox import BoundingBox
from EyeDetect.ValueObjects.EyeGeometry import EyeGeometry
from EyeDetect.ValueObjects.NormalizedEye import NormalizedEye


class EyeRegion:
    def __init__(
        self,
        side: EyeSide,
        points: np.ndarray,
        geometry: EyeGeometry,
        box: BoundingBox,
        mask: np.ndarray = None,
        normalized: NormalizedEye = None
    ):
        self.side = side
        self.points = points
        self.geometry = geometry
        self.box = box
        self.mask = mask
        self.normalized = normalized

        # future-proof
        self.frame_id = None
        self.timestamp = None