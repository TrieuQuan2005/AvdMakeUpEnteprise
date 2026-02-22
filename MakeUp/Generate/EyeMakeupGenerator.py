import torch

class EyeMakeupGenerator:
    def __init__(self, generator, device="cpu"):
        self.generator = generator.to(device).eval()
        self.device = device

    @torch.no_grad()
    def generate(self, eye_crop, style_code, mask_eye_space, symmetry):
        x = self._to_tensor(eye_crop)
        m = self._to_tensor(mask_eye_space, is_mask=True)

        delta = self.generator(x, style_code)
        delta = self._apply_symmetry(delta, symmetry)

        out = x + delta * m.repeat(1, 3, 1, 1)
        return self._to_numpy(out)

    @staticmethod
    def _apply_symmetry(delta, symmetry):
        # Hook for symmetry-aware constraint (used in training / Phase 3)
        return delta

    def _to_tensor(self, img, is_mask=False):
        t = torch.from_numpy(img).float().to(self.device)
        if not is_mask:
            t = t.permute(2, 0, 1) / 255.0
        else:
            t = t.unsqueeze(0)
        return t.unsqueeze(0)

    @staticmethod
    def _to_numpy(t):
        t = t.squeeze(0).permute(1, 2, 0)
        return (t.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
