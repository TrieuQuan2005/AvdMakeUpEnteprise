import cv2
import numpy as np

from MakeUp.EyeMakeupSpec import EyeMakeupSpec


class EyeMakeupRenderer:
    def __init__(self, spec: EyeMakeupSpec):
        self.spec = spec

    def apply(self, frame: np.ndarray, eye_region) -> np.ndarray:
        result = frame.astype(np.float32).copy()

        mask = np.clip(eye_region.mask, 0.0, 1.0)
        if mask is None:
            return result.astype(np.uint8)

        # refine
        mask = self._refine_eyelid_mask(mask, eye_region)

        # gradient
        mask = self._apply_gradient(mask)

        # blur
        k = self.spec.blur if self.spec.blur % 2 == 1 else self.spec.blur + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

        # overlay
        overlay = np.zeros_like(result)
        overlay[:] = self.spec.eyeshadow_color

        # blend
        mask_3c = np.expand_dims(mask, axis=-1)
        blend = mask_3c * self.spec.alpha

        result = result * (1 - blend) + overlay * blend

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _refine_eyelid_mask(mask, eye_region):
        h, w = mask.shape

        center_y = int(eye_region.geometry.center[1] - eye_region.box.y1)
        center_y = np.clip(center_y, 0, h - 1)

        refined = mask.copy()

        # giảm phần dưới (tròng mắt)
        refined[center_y:, :] *= 0.2

        # cắt phần trên (lông mày)
        top_cut = int(h * 0.25)
        refined[:top_cut, :] *= 0.3

        return refined

    def _apply_gradient(self, mask):
        if self.spec.gradient_strength <= 0:
            return mask

        h, w = mask.shape

        # vertical gradient (trên đậm dưới nhạt)
        grad_y = np.linspace(1, 0, h).reshape(h, 1)
        grad_y = np.repeat(grad_y, w, axis=1)

        # horizontal gradient (giữa đậm, 2 bên nhạt)
        x = np.linspace(-1, 1, w)
        grad_x = 1 - np.abs(x)
        grad_x = np.repeat(grad_x.reshape(1, w), h, axis=0)

        # combine
        grad = 0.7 * grad_y + 0.3 * grad_x

        return mask * (
                (1 - self.spec.gradient_strength) +
                self.spec.gradient_strength * grad
        )
