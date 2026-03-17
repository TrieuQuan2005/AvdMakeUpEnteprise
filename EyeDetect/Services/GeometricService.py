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
    def polygon_mask(image_shape, pts: np.ndarray) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return mask.astype(np.float32) / 255.0

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
    def get_eye_box(pts, img_w, img_h) -> BoundingBox:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        ew = x_max - x_min
        eh = y_max - y_min

        x1 = int(max(0, x_min - 0.10 * ew))
        x2 = int(min(img_w - 1, x_max + 0.25 * ew))
        y1 = int(max(0, y_min - 0.32 * eh))
        y2 = int(min(img_h - 1, y_max + 0.39 * eh))

        return BoundingBox(x1, y1, x2, y2)
