from typing import Optional, Dict
import numpy as np

from EyeDetect.Geometric.geometry import EyeGeometry
from EyeDetect.Geometric.normalizer import EyeNormalizer

class EyeRegionExtractor:
    def __init__(self, output_size: int = 128):
        self.normalizer = EyeNormalizer(output_size)

    @staticmethod
    def _get_eye_box( pts, img_w, img_h):
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        ew = x_max - x_min
        eh = y_max - y_min

        x1 = int(max(0, x_min - 0.10 * ew))
        x2 = int(min(img_w - 1, x_max + 0.25 * ew))
        y1 = int(max(0, y_min - 0.32 * eh))
        y2 = int(min(img_h - 1, y_max + 0.39 * eh))

        return x1, y1, x2, y2

    def extract(self,image: np.ndarray,landmarks,indices) -> Optional[Dict]:
        h, w = image.shape[:2]

        pts = EyeGeometry.landmarks_to_points(landmarks, indices, w, h)
        if pts.shape[0] < 3:
            return None

        geometry = EyeGeometry.compute_geometry(pts)
        box = self._get_eye_box(pts, w, h)

        mask = EyeGeometry.polygon_mask(image.shape, pts)
        norm = self.normalizer.normalize(image, box)
        if norm is None:
            return None

        aligned, forward_tf, inverse_tf = norm
        sym_center, sym_dir = EyeGeometry.symmetry_axis(pts)

        return {
            "geometry": geometry,
            "box": box,
            "masks": {"eye_region": mask},
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
