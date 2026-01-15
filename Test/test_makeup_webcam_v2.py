import cv2
import numpy as np

from EyeDetect.eye_detector import detect_eye_regions
from Common.image_utils import create_face_mesh, load_image
from MakeUp.color_tranfer import color_transfer
from MakeUp.blend_gamma import gamma_blend
from MakeUp.temporal_smoothing import TemporalSmoother


def apply_makeup(frame, eye_data, ref_eye, smoother):
    box = eye_data["box"]
    mask = eye_data["masks"]["eye_region"]

    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return frame

    roi = frame[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    ref_resized = cv2.resize(ref_eye, (w, h))

    # 1. Temporal smoothing
    ref_resized = smoother.smooth(ref_resized)

    # 2. Color transfer
    ref_color = color_transfer(ref_resized, roi)

    # 3. Gamma blend
    blended = gamma_blend(roi, ref_color, mask_crop)

    frame[y1:y2, x1:x2] = blended
    return frame


def main():
    cap = cv2.VideoCapture(0)
    face_mesh = create_face_mesh(static=False)

    ref_img = load_image("../Data/makeup.jpg")
    ref_eyes = detect_eye_regions(ref_img, face_mesh)

    if ref_eyes is None:
        print("❌ Không detect được mắt ảnh reference")
        return

    ref_left = ref_eyes["left_eye"]["normalized"]["aligned_crop"]
    ref_right = ref_eyes["right_eye"]["normalized"]["aligned_crop"]

    smoother_left = TemporalSmoother()
    smoother_right = TemporalSmoother()

    print("🎨 Refined Makeup Filter – nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eyes = detect_eye_regions(frame, face_mesh)
        if eyes:
            frame = apply_makeup(frame, eyes["left_eye"], ref_left, smoother_left)
            frame = apply_makeup(frame, eyes["right_eye"], ref_right, smoother_right)

        cv2.imshow("Stage 2 – Refined Makeup", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
