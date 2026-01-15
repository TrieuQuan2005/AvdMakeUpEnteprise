import cv2
import numpy as np

from EyeDetect.eye_detector import detect_eye_regions
from Common.image_utils import create_face_mesh


def draw_eye_debug(frame, eye_data, eye_name):
    box = eye_data["box"]
    mask = eye_data["masks"]["eye_region"]

    x1, y1, x2, y2 = box

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        eye_name,
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Overlay mask (red)
    overlay = frame.copy()
    overlay[mask > 0] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam")
        return

    face_mesh = create_face_mesh(static=False)

    print("📷 Webcam started – nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eyes = detect_eye_regions(frame, face_mesh)

        if eyes:
            if eyes.get("left_eye"):
                draw_eye_debug(frame, eyes["left_eye"], "LEFT EYE")

            if eyes.get("right_eye"):
                draw_eye_debug(frame, eyes["right_eye"], "RIGHT EYE")

        cv2.imshow("Eye Detection Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
