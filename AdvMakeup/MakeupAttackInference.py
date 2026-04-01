import torch
import cv2
import numpy as np
import torchvision.transforms as T


class RealtimeMakeupAttack:
    def __init__(self, generator, fr_model, eye_detector, device):
        self.device = device

        self.G = generator.to(device).eval()
        self.fr_model = fr_model.to(device).eval()
        self.eye_detector = eye_detector

        self.transform = T.ToTensor()

        self.victim_emb = None
        self.makeup = None

    # PUBLIC API
    def set_victim_embedding(self, emb):
        self.victim_emb = emb.to(self.device)

    def set_makeup(self, makeup_bgr):
        makeup_rgb = cv2.cvtColor(makeup_bgr, cv2.COLOR_BGR2RGB)

        makeup = self.transform(makeup_rgb).unsqueeze(0) * 255.0
        self.makeup = makeup.to(self.device)

    def clear_makeup(self):
        self.makeup = None

    def __call__(self, frame_bgr):
        return self.run_frame(frame_bgr)

    # INTERNAL
    def run_frame(self, frame_bgr):

        if self.victim_emb is None:
            return frame_bgr

        if self.makeup is None:
            return frame_bgr

        result = self.eye_detector.detect(frame_bgr)

        if result is None:
            return frame_bgr

        left_eye, right_eye = result
        frame_out = frame_bgr.copy()

        for eye in [left_eye, right_eye]:

            if eye.normalized is None or eye.mask is None:
                continue

            eye_img = eye.normalized.image
            mask = eye.mask

            h, w = eye_img.shape[:2]
            mask = cv2.resize(mask, (w, h))

            mask = mask.astype(np.float32)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            mask = np.clip(mask, 0, 1)

            img_tensor = self.transform(eye_img).unsqueeze(0) * 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

            img_tensor = img_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)

            with torch.no_grad():

                x_adv = self.G(
                    img_tensor,
                    mask_tensor,
                    self.victim_emb,
                    self.makeup
                )

                x_adv = torch.clamp(x_adv, 0, 255)

            adv_np = x_adv[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            x1, y1, x2, y2 = eye.box.as_tuple()

            if x2 <= x1 or y2 <= y1:
                continue

            adv_np = cv2.resize(adv_np, (x2 - x1, y2 - y1))
            frame_out[y1:y2, x1:x2] = adv_np

        return frame_out