import cv2
import numpy as np

from EyeDetect.Services.EyeDetectorService import EyeDetectorService
from MakeUp.EyeMakeupSpec import EyeMakeupSpec
from MakeUp.EyeMakeupRenderer import EyeMakeupRenderer

import mediapipe as mp


# =========================
# Init FaceMesh
# =========================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

detector = EyeDetectorService(face_mesh)


# =========================
# Init Makeup
# =========================
spec = EyeMakeupSpec(
    eyeshadow_color=np.array([180, 50, 150]),
    alpha=0.7,
    blur=13,
    gradient_strength=0.8
)

renderer = EyeMakeupRenderer(spec)


# =========================
# Load image
# =========================
frame = cv2.imread("test_face.jpg")  # đổi path
frame = cv2.resize(frame, (640, 640))


# =========================
# Detect eyes
# =========================
eyes = detector.detect(frame)

if eyes is None:
    print("❌ Không detect được mắt")
    exit()

left_eye, right_eye = eyes





# =========================
# Apply makeup từng mắt
# =========================


def draw_bbox(frame, eye_region, color=(0, 255, 0)):
    x1 = int(eye_region.box.x1)
    y1 = int(eye_region.box.y1)
    x2 = int(eye_region.box.x2)
    y2 = int(eye_region.box.y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

result = frame.copy()

draw_bbox(result,eye_region=left_eye)
draw_bbox(result,eye_region=right_eye)
result = renderer.apply(result, left_eye)
result = renderer.apply(result, right_eye)
# =========================
# Show
# =========================
cv2.imshow("Original", frame)
cv2.imshow("Makeup", result)

cv2.waitKey(0)
cv2.destroyAllWindows()