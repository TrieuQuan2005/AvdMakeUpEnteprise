import numpy as np
import cv2
from typing import Optional
from EyeDetect.ValueObjects.AffineTransform import AffineTransform
from EyeDetect.ValueObjects.BoundingBox import BoundingBox
from EyeDetect.ValueObjects.EyeGeometry import EyeGeometry
from EyeDetect.ValueObjects.NormalizedEye import NormalizedEye

class EyeGeometricService:
    @staticmethod
    def landmarks_to_points(landmarks, indices, img_w, img_h) -> np.ndarray:
        pts = []
        for i in indices:
            if i >= len(landmarks):
                continue
            pts.append([
                landmarks[i].x * img_w,
                landmarks[i].y * img_h
            ])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def compute_geometry(pts: np.ndarray) -> EyeGeometry:
        center = pts.mean(axis=0)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min

        return EyeGeometry(
            center=center,
            width=width,
            height=height,
            aspect_ratio= float(height / (width + 1e-6))
        )

    @staticmethod
    def polygon_mask(pts: np.ndarray, box) -> np.ndarray:
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        w = x2 - x1
        h = y2 - y1

        pts_local = pts.copy()
        pts_local[:, 0] -= x1
        pts_local[:, 1] -= y1

        # Smooth contour (optional nhưng nên có)
        pts_smooth = cv2.approxPolyDP(
            pts_local.astype(np.int32),
            epsilon=1.5,
            closed=True
        )

        hull = cv2.convexHull(pts_smooth)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [hull], 1.0)

        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        return mask

    @staticmethod
    def eye_direction(pts):
        return pts[0] - pts[1]

    @staticmethod
    def normalize(
            image: np.ndarray,
            box: BoundingBox,
            output_size: int
    ) -> Optional[NormalizedEye]:
        x1, y1, x2, y2 = box.as_tuple()
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        h, w = crop.shape[:2]

        sx = output_size / w
        sy = output_size / h

        forward = np.array([
            [sx, 0, -x1 * sx],
            [0, sy, -y1 * sy]
        ], dtype=np.float32)

        inverse = np.array([
            [1 / sx, 0, x1],
            [0, 1 / sy, y1]
        ], dtype=np.float32)

        aligned = cv2.resize(crop, (output_size, output_size))

        transform = AffineTransform(forward, inverse)

        return NormalizedEye(
            image=aligned,
            transform=transform,
            box=box
        )

    @staticmethod
    def get_eye_box(pts, img_w, img_h, pad: int = 2) -> BoundingBox:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        x1 = int(max(0, x_min - pad))
        y1 = int(max(0, y_min - pad))
        x2 = int(min(img_w - 1, x_max + pad))
        y2 = int(min(img_h - 1, y_max + pad))

        return BoundingBox(x1, y1, x2, y2)

def order_points(pts: np.ndarray):
    center = np.mean(pts, axis=0)

    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])

    sorted_idx = np.argsort(angles)

    return pts[sorted_idx]