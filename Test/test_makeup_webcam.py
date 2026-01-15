import cv2

from EyeDetect.eye_detector import detect_eye_regions
from Common.image_utils import create_face_mesh, load_image
from MakeUp.apply_makeup import apply_makeup



def main():
    cap = cv2.VideoCapture(0)
    face_mesh = create_face_mesh(static=False)

    # Load reference makeup image
    ref_img = load_image("../Data/makeup.jpg")
    ref_eyes = detect_eye_regions(ref_img, face_mesh)

    if ref_eyes is None:
        print("Không detect được mắt trong ảnh reference")
        return

    ref_left_eye = ref_eyes["left_eye"]["normalized"]["aligned_crop"]
    ref_right_eye = ref_eyes["right_eye"]["normalized"]["aligned_crop"]

    print("🎨 Makeup filter started – nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eyes = detect_eye_regions(frame, face_mesh)

        if eyes:
            frame = apply_makeup(frame, eyes["left_eye"], ref_left_eye)
            frame = apply_makeup(frame, eyes["right_eye"], ref_right_eye)

        cv2.imshow("Makeup Filter (Stage 2)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
