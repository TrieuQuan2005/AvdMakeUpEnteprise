import torch

from attack.embedding_attack import EmbeddingAttacker
from metrics.similarity import cosine_similarity
from config import Config


def main():
    cfg = Config()

    e_you = torch.randn(cfg.embedding_dim)
    e_target = torch.randn(cfg.embedding_dim)

    print("Baseline similarity:")
    print("cos(e_you, e_target) =",
          cosine_similarity(e_you, e_target).item())

    attacker = EmbeddingAttacker(
        step_size=cfg.step_size,
        max_iter=cfg.max_iter,
        epsilon=cfg.epsilon,
        #lambda_self=cfg.lambda_self,
        device=cfg.device
    )

    e_adv = attacker.attack(e_you, e_target)

    print("\nAfter attack:")
    print("cos(e_adv, e_target) =",
          cosine_similarity(e_adv, e_target).item())
    print("cos(e_adv, e_you) =",
          cosine_similarity(e_adv, e_you).item())


if __name__ == "__main__":
    main()
