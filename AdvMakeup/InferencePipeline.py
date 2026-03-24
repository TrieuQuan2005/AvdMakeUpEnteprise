# pipeline.py

import torch
from torchvision.utils import save_image


def run_inference(
    inferencer,
    attacker_path,
    mask_path,
    victim_path,
    save_path
):
    attacker = inferencer.load_image(attacker_path)
    mask = inferencer.load_mask(mask_path)
    victim = inferencer.load_image(victim_path)

    victim_emb = inferencer.get_embedding(victim)

    x_adv = inferencer.attack(attacker, mask, victim_emb)

    save_image(x_adv / 255.0, save_path)
    print(f"[DONE] Saved: {save_path}")