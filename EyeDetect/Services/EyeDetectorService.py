import cv2
import numpy as np
from typing import Optional, Tuple

from EyeDetect.ValueObjects.EyeLandMark import EyeLandMark
from EyeDetect.Enums.EyeSide import EyeSide

from EyeDetect.Services.GeometricService import EyeGeometricService
from EyeDetect.ValueObjects.EyeRegion import EyeRegion


class EyeDetectorService:
    def __init__(self, face_mesh, output_size: int = 128):
        self.face_mesh = face_mesh
        self.output_size = output_size

    # PUBLIC API
    def detect(self, frame_bgr) -> Optional[Tuple[EyeRegion, EyeRegion]]:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        # ===== EXTRACT =====
        left_eye = self._extract_eye(
            frame_bgr, landmarks, EyeLandMark.LEFT_EYE_FULL, w, h, EyeSide.LEFT
        )

        right_eye = self._extract_eye(
            frame_bgr, landmarks, EyeLandMark.RIGHT_EYE_FULL, w, h, EyeSide.RIGHT
        )

        left_eyeball = self._extract_eye(
            frame_bgr, landmarks, EyeLandMark.LEFT_EYEBALL, w, h, EyeSide.LEFT
        )

        right_eyeball = self._extract_eye(
            frame_bgr, landmarks, EyeLandMark.RIGHT_EYEBALL, w, h, EyeSide.RIGHT
        )

        # SAFE CHECK
        if any(x is None for x in [left_eye, right_eye, left_eyeball, right_eyeball]):
            return None

        # REMOVE EYEBALL REGION
        left_eye = self._remove_eyeball(left_eye, left_eyeball)
        right_eye = self._remove_eyeball(right_eye, right_eyeball)

        return left_eye, right_eye

    # INTERNAL
    def _extract_eye(
        self,
        image,
        landmarks,
        indices,
        w,
        h,
        side: EyeSide
    ) -> Optional[EyeRegion]:

        # 1. Landmarks → points
        pts = EyeGeometricService.landmarks_to_points(
            landmarks, indices, w, h
        )

        if pts.shape[0] < 3:
            return None

        # 2. Geometry
        geometry = EyeGeometricService.compute_geometry(pts)

        # 3. Bounding box
        box = EyeGeometricService.get_eye_box(pts, w, h)

        # 4. Mask
        mask = EyeGeometricService.polygon_mask(pts, box)

        # 5. Normalize
        normalized = EyeGeometricService.normalize(
            image,
            box,
            self.output_size
        )

        if normalized is None:
            return None

        return EyeRegion(
            side=side,
            points=pts,
            geometry=geometry,
            box=box,
            mask=mask,
            normalized=normalized
        )

    # REMOVE EYEBALL
    @staticmethod
    def _remove_eyeball(
        eye_region: EyeRegion,
        eyeball_region: EyeRegion
    ) -> EyeRegion:

        outer_mask = eye_region.mask
        inner_small = eyeball_region.mask

        H, W = outer_mask.shape
        inner_mask = np.zeros((H, W), dtype=np.float32)

        # OFFSET (align 2 bounding boxes)
        dx = int(round(eyeball_region.box.x1 - eye_region.box.x1))
        dy = int(round(eyeball_region.box.y1 - eye_region.box.y1))

        h, w = inner_small.shape

        # TARGET REGION
        x1 = max(0, dx)
        y1 = max(0, dy)
        x2 = min(W, dx + w)
        y2 = min(H, dy + h)

        # SOURCE REGION
        sx1 = max(0, -dx)
        sy1 = max(0, -dy)
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        # PASTE
        inner_mask[y1:y2, x1:x2] = inner_small[sy1:sy2, sx1:sx2]

        inner_mask = (inner_mask > 0.5).astype(np.float32)

        # SUBTRACT
        clean_mask = np.clip(outer_mask - inner_mask, 0.0, 1.0)

        # Feather lại cho mượt
        clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)

        eye_region.mask = clean_mask

        return eye_region