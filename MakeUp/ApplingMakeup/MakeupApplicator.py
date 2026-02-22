import torch
from StyleInject import MakeupGenerator
class MakeupApplicator:
    def __init__(self, gan_path, device):
        self.device = device
        self.model = MakeupGenerator()
        self.model.load_state_dict(torch.load(gan_path))
        self.model.to(device).eval()

    @torch.no_grad()
    def apply(self, eye_crop, mask, style_vec):
        return self.model(
            eye_crop.unsqueeze(0),
            mask.unsqueeze(0),
            torch.tensor(style_vec).unsqueeze(0).to(self.device)
        ).squeeze(0)
