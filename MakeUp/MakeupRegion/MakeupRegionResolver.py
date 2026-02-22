import cv2

class MakeupRegionResolver:
    def resolve(self, phase1_output: dict, eye: str):
        eye_data = phase1_output[eye]

        return {
            "eye_crop": eye_data["normalized"]["aligned_crop"],

            "mask_eye_space": self._warp_mask_to_eye_space(
                eye_data["masks"]["eye_region"], eye_data["normalized"]["forward_transform"]),
            "inverse_transform": eye_data["normalized"]["inverse_transform"],
            "symmetry": eye_data["structure"]
        }

    @staticmethod
    def _warp_mask_to_eye_space(mask, m):
        return cv2.warpAffine(mask, m, (128, 128))
