import cv2
import numpy as np


def color_transfer(src: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Transfer color statistics from target → src (LAB space)

    src: makeup texture
    target: eye ROI on face
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        s_mean, s_std = src_lab[..., i].mean(), src_lab[..., i].std()
        t_mean, t_std = tgt_lab[..., i].mean(), tgt_lab[..., i].std()

        src_lab[..., i] = (src_lab[..., i] - s_mean) * (t_std / (s_std + 1e-6)) + t_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)
