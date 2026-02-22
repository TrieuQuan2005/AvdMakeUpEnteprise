import torch


class Config:
    embedding_dim = 512
    step_size = 0.1
    max_iter = 300
    epsilon = 0.5
    lambda_self = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

