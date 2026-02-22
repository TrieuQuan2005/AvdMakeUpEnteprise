import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MakeUp.MakeupStyle.model.style_encoder import StyleEncoder
from MakeUp.MakeupStyle.model.style_decoder import StyleDecoder

# Trainer
class StyleTrainer:
    def __init__(
        self,
        dataloader: DataLoader,
        style_dim: int = 128,
        device: str = "cuda"
    ):
        self.device = device
        self.dataloader = dataloader

        self.encoder = StyleEncoder(style_dim).to(device)
        self.decoder = StyleDecoder(style_dim).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()),
            lr=1e-4
        )

    # Losses
    def reconstruction_loss(self, recon, target):
        return F.l1_loss(recon, target)

    def style_regularization(self, z, is_makeup):
        """
        Encourage:
        - makeup style vectors -> spread
        - non-makeup style vectors -> compact
        """
        if z.size(0) < 2:
            return torch.tensor(0.0, device=z.device)

        mean = z.mean(dim=0, keepdim=True)
        dist = ((z - mean) ** 2).sum(dim=1)

        if is_makeup:
            return -dist.mean()   # maximize variance
        else:
            return dist.mean()    # minimize variance

    # Train step
    def train_epoch(self):
        self.encoder.train()
        self.decoder.train()

        total_loss = 0.0

        for batch in self.dataloader:
            images = batch["image"].to(self.device)
            is_makeup = batch["is_makeup"].to(self.device)

            z = self.encoder(images)
            recon = self.decoder(z)

            loss_rec = self.reconstruction_loss(recon, images)

            loss_style = (
                self.style_regularization(z[is_makeup == 1], True) +
                self.style_regularization(z[is_makeup == 0], False)
            )

            loss = loss_rec + 0.1 * loss_style

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    # Main loop
    def train(self, epochs: int = 50, save_path="style_encoder.pth"):
        for epoch in range(epochs):
            loss = self.train_epoch()
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss:.4f}")

        torch.save(self.encoder.state_dict(), save_path)
        print(f"Style Encoder saved to {save_path}")
