import cv2
import mediapipe as mp

from EyeDetect.detector import EyeDetector

# =========================
# Camera utils
# =========================

def open_first_available_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[INFO] Using camera index {i}")
            return cap
        cap.release()
    raise RuntimeError("No available camera found")

def draw_eye_region(frame, eye_data, color=(0, 255, 0)):
    x1, y1, x2, y2 = eye_data["box"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    center = eye_data["structure"]["symmetry_center"]
    direction = eye_data["structure"]["symmetry_direction"]

    p1 = center - direction * 50
    p2 = center + direction * 50

    cv2.line(
        frame,
        tuple(p1.astype(int)),
        tuple(p2.astype(int)),
        (255, 0, 0),
        2
    )

def main():
    cap = open_first_available_camera()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    eye_detector = EyeDetector(face_mesh=face_mesh, output_size=128)

    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = eye_detector.detect(frame)

        if result is not None:
            draw_eye_region(frame, result["left_eye"], color=(0, 255, 0))
            draw_eye_region(frame, result["right_eye"], color=(0, 255, 0))

        cv2.imshow("EyeDetector - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
