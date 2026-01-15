import numpy as np
import cv2


def feather_mask(mask, ksize=15):
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0) / 255.0


def apply_makeup(frame, eye_data, ref_eye):
    box = eye_data["box"]
    mask = eye_data["masks"]["eye_region"]

    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1

    if h <= 0 or w <= 0:
        return frame

    # Resize reference eye to current eye size
    ref_resized = cv2.resize(ref_eye, (w, h))

    # Crop frame
    roi = frame[y1:y2, x1:x2]

    # Prepare mask
    mask_crop = mask[y1:y2, x1:x2]

    #alpha = feather_mask(mask_crop)
    alpha = cv2.GaussianBlur(mask_crop, (31, 31), 0) / 255.0

    # Blend
    blended = roi * (1 - alpha[..., None]) + ref_resized * alpha[..., None]

    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame