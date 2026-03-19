import cv2
import numpy as np
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gram_matrix(y):
    b, ch, h, w = y.size()
    features = y.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def compute_laplacian_gradient(img_tensor, lap_filter):
    grads = []
    for c in range(3):
        channel = img_tensor[:, c:c+1, :, :]
        grads.append(lap_filter(channel))
    return grads


def compute_gt_gradient(source, target, mask, offset, lap_filter):
    device = source.device
    x, y = offset
    h, w = mask.shape

    # Compute gradients
    source_grad = compute_laplacian_gradient(source, lap_filter)
    target_grad = compute_laplacian_gradient(target, lap_filter)

    # Canvas mask
    canvas_mask = torch.zeros((1, 1, target.shape[2], target.shape[3]), device=device)
    canvas_mask[:, :, x:x+h, y:y+w] = mask.unsqueeze(0).unsqueeze(0)

    result = []
    for sg, tg in zip(source_grad, target_grad):
        # foreground (source)
        fg = torch.zeros_like(tg)
        fg[:, :, x:x+h, y:y+w] = sg[:, :, :h, :w] * mask

        # background (target)
        bg = tg * (1 - canvas_mask)

        result.append(fg + bg)

    return result

def preprocess_image(img, device=None):
    device = device or get_device()

    if img.max() > 1:
        img = img / 255.0

    tensor = torch.from_numpy(img).float().to(device)

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor

def deprocess_image(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img.clip(0, 1)
    return img

def normalize_eye(eye_region) -> np.ndarray:

    roi = eye_region.get_roi_image()  # bạn cần implement

    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype(np.float32) / 255.0

    return roi

def postprocess( patch: np.ndarray) -> np.ndarray:

    patch = np.clip(patch, 0, 1)
    patch = (patch * 255).astype(np.uint8)
    return patch