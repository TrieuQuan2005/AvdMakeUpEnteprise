import torch
import torch.nn.functional as f


class EmbeddingAttacker:
    def __init__(
        self,
        step_size: float = 0.2,
        max_iter: int = 300,
        epsilon: float = 5.0,
        device: str = "cpu"
    ):
        self.step_size = step_size
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.device = device

    @staticmethod
    def _ensure_2d(x):
        return x.unsqueeze(0) if x.dim() == 1 else x

    def attack(
        self,
        e_you: torch.Tensor,
        e_target: torch.Tensor,
        verbose: bool = True
    ):
        e_you = self._ensure_2d(e_you).to(self.device)
        e_target = self._ensure_2d(e_target).to(self.device)

        # Normalize embeddings
        e_you = f.normalize(e_you, dim=1)
        e_target = f.normalize(e_target, dim=1)

        # Init delta toward target
        delta = (e_target - e_you).detach()
        delta = delta / delta.norm(dim=1, keepdim=True) * 0.01
        delta.requires_grad_()

        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for i in range(self.max_iter):
            optimizer.zero_grad()

            e_adv = f.normalize(e_you + delta, dim=1)
            loss = -f.cosine_similarity(e_adv, e_target, dim=1).mean()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)

            if verbose and i % 20 == 0:
                sim = f.cosine_similarity(e_adv, e_target, dim=1).item()
                print(f"[{i:03d}] cos(e_adv, target) = {sim:.4f}")

        # final normalized adversarial embedding
        e_adv = f.normalize(e_you + delta, dim=1).detach()
        return e_adv.squeeze(0)
