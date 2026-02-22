import cv2
import mediapipe as mp

from EyeDetect.detector import EyeDetector
from MakeUp.MakeupEngine import MakeupEngine
from MakeUp.MakeupRegion.MakeupRegion import MakeupRegion


def main():
    cap = cv2.VideoCapture(0)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ===== INIT MODULES =====
    eye_detector = EyeDetector(face_mesh)
    makeup_engine = MakeupEngine(strength=0.8)

    # ===== LOAD REFERENCE (GIỐNG CODE CŨ) =====
    ref_img = cv2.imread("../Data/makeup.jpg")
    ref_eyes = eye_detector.detect(ref_img)

    if ref_eyes is None:
        print(" Không detect được mắt trong ảnh reference")
        return

    ref_left_eye = ref_eyes["left_eye"]["normalized"]["aligned_crop"]
    ref_right_eye = ref_eyes["right_eye"]["normalized"]["aligned_crop"]

    print("Makeup filter started – nhấn ESC để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eyes = eye_detector.detect(frame)

        if eyes:
            # LEFT EYE
            left = eyes["left_eye"]
            left_region = MakeupRegion(
                box=left["box"],
                mask=left["masks"]["eye_region"]
            )
            frame = makeup_engine.apply_region(
                frame,
                left_region,
                ref_left_eye
            )

            # RIGHT EYE
            right = eyes["right_eye"]
            right_region = MakeupRegion(
                box=right["box"],
                mask=right["masks"]["eye_region"]
            )
            frame = makeup_engine.apply_region(
                frame,
                right_region,
                ref_right_eye
            )

        cv2.imshow("Makeup Filter – Refactored", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
