import cv2
from typing import Optional, Dict

from EyeDetect.extractor.eye_region_extractor import EyeRegionExtractor
from .constants import LEFT_EYE_EXT_IDX, RIGHT_EYE_EXT_IDX

class EyeDetector:
    def __init__(self, face_mesh, output_size: int = 128):
        self.face_mesh = face_mesh
        self.extractor = EyeRegionExtractor(output_size)

    def detect(self, frame_bgr) -> Optional[Dict]:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = self.extractor.extract(
            frame_bgr, landmarks, LEFT_EYE_EXT_IDX
        )
        right_eye = self.extractor.extract(
            frame_bgr, landmarks, RIGHT_EYE_EXT_IDX
        )

        if left_eye is None or right_eye is None:
            return None

        return {
            "face_landmarks": landmarks,
            "left_eye": left_eye,
            "right_eye": right_eye
        }
