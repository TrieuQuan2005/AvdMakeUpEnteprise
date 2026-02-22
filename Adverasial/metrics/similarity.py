import torch
import torch.nn.functional as f


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = f.normalize(a, dim=-1)
    b = f.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - cosine_similarity(a, b)
