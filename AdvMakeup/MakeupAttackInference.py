# realtime_inference.py

import torch
import cv2
import numpy as np
import torchvision.transforms as T


def preprocess_facenet(x):
    return (x / 127.5) - 1.0


class RealtimeMakeupAttack:
    def __init__(self, generator, fr_model, eye_detector, device):
        self.device = device
        self.G = generator.to(device).eval()
        self.fr_model = fr_model.to(device).eval()
        self.eye_detector = eye_detector

        self.transform = T.Compose([
            T.ToTensor()
        ])

    def get_embedding(self, img_tensor):
        with torch.no_grad():
            emb = self.fr_model(preprocess_facenet(img_tensor))
        return emb

    def run_frame(self, frame_bgr, victim_emb):
        result = self.eye_detector.detect(frame_bgr)

        if result is None:
            return frame_bgr

        left_eye, right_eye = result
        frame_out = frame_bgr.copy()

        for eye in [left_eye, right_eye]:

            if eye.normalized is None or eye.mask is None:
                continue

            eye_img = eye.normalized.image   # (H, W, 3)
            mask = eye.mask                 # (H, W)

            h, w = eye_img.shape[:2]
            mask = cv2.resize(mask, (w, h))

            mask = mask.astype(np.float32)

            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            mask = np.clip(mask, 0, 1)

            img_tensor = self.transform(eye_img).unsqueeze(0) * 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

            img_tensor = img_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)

            victim_emb_batch = victim_emb.repeat(1, 1)

            with torch.no_grad():
                perturb = self.G(img_tensor, mask_tensor, victim_emb_batch)

                perturb = perturb.mean(dim=1, keepdim=True)

                perturb = torch.clamp(perturb, -3, 3)

                # ===== thử các kiểu =====
                #x_adv = img_tensor * (1 - mask_tensor) + (img_tensor + perturb) * mask_tensor
                #x_adv = img_tensor + 50 * mask_tensor #Makeup mo
                #x_adv = img_tensor * (1 - mask_tensor) + (img_tensor + 50 * perturb) * mask_tensor # Vo mau
                x_adv = img_tensor * (1 - mask_tensor) + (img_tensor + 20 * perturb) * mask_tensor # Vo mau

                x_adv = torch.clamp(x_adv, 0, 255)

            adv_np = x_adv[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            x1, y1, x2, y2 = eye.box.as_tuple()

            if x2 <= x1 or y2 <= y1:
                continue

            adv_np = cv2.resize(adv_np, (x2 - x1, y2 - y1))

            frame_out[y1:y2, x1:x2] = adv_np

            # ===== debug =====
            print("Mask min/max:", mask.min(), mask.max())
            print("Perturb mean:", perturb.abs().mean().item())

        return frame_out