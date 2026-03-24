# app_realtime.py

import cv2
import torch
import os
import torch.nn.functional as F

from AdvMakeup.Models.GanNetwork import GanNetwork
from AdvMakeup.Models.FaceReconizationModels.FaceNet.FaceNetWrapper import FaceNetWrapper
from AdvMakeup.MakeupAttackInference import RealtimeMakeupAttack
from EyeDetect.Services.EyeDetectorService import EyeDetectorService

import mediapipe as mp


# ===== LossId =====
def compute_loss_id(emb1, emb2):
    cos_sim = F.cosine_similarity(emb1, emb2)
    loss_id = 1 - cos_sim
    return loss_id.item(), cos_sim.item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ===== load GAN =====
    gan = GanNetwork(device)
    G = gan.generator

    G.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "AdvMakeup", "checkpoints", "G_best.pth"),
        map_location=device
    ))

    # ===== load FaceNet =====
    fr_model = FaceNetWrapper(device=device).eval()

    # ===== mediapipe =====
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    eye_detector = EyeDetectorService(mp_face_mesh)

    # ===== inference =====
    infer = RealtimeMakeupAttack(G, fr_model, eye_detector, device)

    # ===== victim =====
    victim_path = os.path.join(BASE_DIR, "AdvMakeup", "Trieuquan.jpg")

    victim_img = cv2.imread(victim_path)
    if victim_img is None:
        raise ValueError("Không đọc được ảnh victim")

    victim_img = cv2.resize(victim_img, (160, 160))

    victim_tensor = torch.from_numpy(victim_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    victim_emb = infer.get_embedding(victim_tensor)

    # ===== webcam =====
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== attack =====
        output = infer.run_frame(frame, victim_emb)

        # ===== detect lại mắt trên output =====
        result = infer.eye_detector.detect(output)

        cos_sim, loss_id = None, None

        if result is not None:
            left_eye, right_eye = result

            # ===== tạo face crop từ 2 mắt =====
            x1 = min(left_eye.box.x1, right_eye.box.x1)
            y1 = min(left_eye.box.y1, right_eye.box.y1)
            x2 = max(left_eye.box.x2, right_eye.box.x2)
            y2 = max(left_eye.box.y2, right_eye.box.y2)

            if x2 > x1 and y2 > y1:
                face_crop = output[y1:y2, x1:x2]

                try:
                    face_crop = cv2.resize(face_crop, (160, 160))

                    face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).unsqueeze(0).float().to(device)

                    attacker_emb = infer.get_embedding(face_tensor)

                    loss_id, cos_sim = compute_loss_id(attacker_emb, victim_emb)

                except:
                    pass

        # ===== hiển thị =====
        if cos_sim is not None:
            cv2.putText(output, f"CosSim: {cos_sim:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(output, f"LossId: {loss_id:.3f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(output, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ===== show =====
        cv2.imshow("Makeup Attack", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()