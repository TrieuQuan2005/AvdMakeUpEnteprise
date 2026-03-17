import numpy as np


class EyeMakeupSpec:
    def __init__(
        self,
        eyeshadow_color: np.ndarray = None,   # RGB
        alpha: float = 0.6,                   # độ đậm
        blur: int = 11,                       # feather strength (kernel size)
        gradient_strength: float = 0.5,       # 0 = flat, 1 = full gradient
        gradient_direction: str = "vertical"  # vertical | radial (future)
    ):
        self.eyeshadow_color = (
            eyeshadow_color if eyeshadow_color is not None
            else np.array([180, 80, 120], dtype=np.float32)
        )

        self.alpha = alpha
        self.blur = blur if blur % 2 == 1 else blur + 1  # kernel phải lẻ
        self.gradient_strength = gradient_strength
        self.gradient_direction = gradient_direction