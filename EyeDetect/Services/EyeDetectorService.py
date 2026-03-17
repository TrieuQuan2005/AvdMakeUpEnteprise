import cv2
from typing import Optional, Tuple

from EyeDetect.Enums.EyeIndex import LEFT_EYE_EXT_IDX, RIGHT_EYE_EXT_IDX
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

        left_eye = self._extract_eye(
            frame_bgr, landmarks, LEFT_EYE_EXT_IDX, w, h, EyeSide.LEFT
        )

        right_eye = self._extract_eye(
            frame_bgr, landmarks, RIGHT_EYE_EXT_IDX, w, h, EyeSide.RIGHT
        )

        if left_eye is None or right_eye is None:
            return None

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

        # 1. Points
        pts = EyeGeometricService.landmarks_to_points(
            landmarks, indices, w, h
        )

        if pts.shape[0] < 3:
            return None

        geometry = EyeGeometricService.compute_geometry(pts)
        box = EyeGeometricService.get_eye_box(pts, w, h)
        mask = EyeGeometricService.polygon_mask(image.shape, pts)
        normalized = EyeGeometricService.normalize(
            image,
            box,
            self.output_size
        )

        if normalized is None:
            return None

        # 7. Build EyeRegion (CONTRACT)
        eye_region = EyeRegion(
            side=side,
            points=pts,
            geometry=geometry,
            box=box,
            mask=mask,
            normalized=normalized
        )
        return eye_region